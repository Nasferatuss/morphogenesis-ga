"""Train agent: GA training pipeline with curriculum and benchmark integration.

This module hosts the ``train_ga`` function that was previously a 3.5k-line
monolith inside ``run_train.py``. The extraction is wholesale — the function
body, its 21 nested closures, and all captured state move here unchanged to
preserve behaviour. Further decomposition (splitting closures into reusable
services) is follow-up work; the goal of this step is purely to get the
monolith out of the CLI entry point.

Behaviour is guarded by ``tests/test_golden.py`` and the fitness/simulate test
suites; any regression in the refactor is caught by those tests.
"""
from __future__ import annotations

import argparse
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

from agents.train_agent import benchmark_runner as _benchmark_module
from agents.train_agent import mini_transfer as _mini_transfer_module
from agents.train_agent.evaluator import EvaluatorState
from agents.train_agent.adaptive_mutation import (
    AdaptiveMutationConfig,
    adapt_mutation_std,
)
from agents.train_agent.mutation import (
    TMutationConfig,
)
from agents.train_agent.mutation import (
    compute_phase_cleanup_ratio as _agent_phase_cleanup,
)
from agents.train_agent.mutation import (
    resolve_mutation_std as _agent_resolve_mut_std,
)
from agents.train_agent.mutation import (
    resolve_stem_penalty_multiplier as _agent_resolve_stem_penalty,
)
from agents.train_agent.scoring import (
    compute_benchmark_dict_score as _agent_bench_dict_score,
)
from agents.train_agent.scoring import (
    compute_benchmark_score as _agent_bench_score,
)
from agents.train_agent.scoring import (
    simple_mean as _agent_simple_mean,
)
from agents.train_agent.scoring import (
    simple_variance as _agent_simple_variance,
)
from agents.train_agent.t_weights import scale_t_weights as _agent_scale_t_weights
from core.memory.metrics_logger import (
    export_tensorboard_plots,
    write_generations_csv,
    write_run_summary,
)
from core.services.cleanup import resolve_cleanup_ratio
from core.services.factory import (
    build_stability_config,
    prepare_model,
    prepare_world,
)
from core.services.stats import summarize
from core.services.visualizer import build_visualizer
from core.services.writer import build_writer
from src.benchmark import BenchmarkConfig
from src.fitness import compute_fitness
from src.ga import (
    clone_model,
    crossover,
    evaluate_population,
    init_population,
    mutate,
    select_elite,
)
from src.model import LittleLM
from src.simulate import simulate
from src.targets import make_target
from src.viz import GifRecorder, render_grid_to_rgb

TENSORBOARD_PNG_TAGS: Tuple[str, ...] = (
    "fitness/best",
    "fitness/mean",
    "iou/best",
    "iou/mean",
    "stats/alive_best",
    "stats/area_mean",
    "coverage/best",
    "coverage/mean",
)


def train_ga(

    cfg: Dict[str, Any],

    args: argparse.Namespace,

    device: torch.device,

    default_target_mask: torch.Tensor,

    seed: int,

    default_target_name: str,

    default_target_area: float,

) -> Path:

    ga_cfg = cfg.get("ga", {})

    world_cfg = cfg.get("world", {})

    model_cfg = cfg.get("model", {})

    viz_cfg = cfg.get("viz", {})

    simulate_cfg = cfg.get("simulate", {})

    fitness_cfg = cfg.get("fitness", {})

    curriculum_cfg = cfg.get("curriculum", {})

    benchmark_cfg = cfg.get("benchmark", {})

    default_target_label = str(default_target_name).lower() if default_target_name else ""

    stability_source = (

        benchmark_cfg.get("stability")

        or benchmark_cfg.get("baseline")

        or benchmark_cfg.get("target")

        or benchmark_cfg

    )

    train_stability_cfg = build_stability_config(stability_source)

    t_phase_weights_cfg = fitness_cfg.get("t_phase_weights")

    if not isinstance(t_phase_weights_cfg, dict):

        t_phase_weights_cfg = {}

    base_t_phase_weights = {k: float(v) for k, v in t_phase_weights_cfg.items()}

    t_phase_weights = base_t_phase_weights

    coverage_cfg = fitness_cfg.get("coverage", {})

    coverage_target_floor_base = float(coverage_cfg.get("target_floor", 0.55))

    coverage_reward_weight_base = float(coverage_cfg.get("reward_weight", 0.6))

    coverage_penalty_weight_base = float(coverage_cfg.get("penalty_weight", 1.2))

    sparse_area_floor_base = float(coverage_cfg.get("area_ratio_floor", 0.6))

    sparse_penalty_weight_base = float(coverage_cfg.get("area_penalty_weight", 0.9))

    coverage_target_floor = coverage_target_floor_base

    coverage_reward_weight = coverage_reward_weight_base

    coverage_penalty_weight = coverage_penalty_weight_base

    sparse_area_floor = sparse_area_floor_base

    sparse_penalty_weight = sparse_penalty_weight_base

    alpha_fp = float(fitness_cfg.get("alpha_fp", 1.0))

    beta_fn = float(fitness_cfg.get("beta_fn", 0.5))

    gamma_area = float(fitness_cfg.get("gamma_area", 0.15))

    stem_penalty_scale = float(fitness_cfg.get("stem_penalty_scale", 0.05))

    stem_cleanup_multiplier = float(fitness_cfg.get("stem_cleanup_multiplier", 1.0))

    stem_corridor_start_scale = float(fitness_cfg.get("stem_corridor_start_scale", 2.0))

    stem_corridor_end_scale = float(fitness_cfg.get("stem_corridor_end_scale", 1.1))

    t_symmetry_weight = float(fitness_cfg.get("t_symmetry_weight", 0.0))

    t_trunk_weight = float(fitness_cfg.get("t_trunk_weight", 0.0))

    late_t_cfg = fitness_cfg.get("late_t", {})

    late_fp_multiplier = float(late_t_cfg.get("fp_multiplier", 1.0))

    late_symmetry_bonus = float(late_t_cfg.get("symmetry_weight", 0.0))

    late_trunk_bonus = float(late_t_cfg.get("trunk_weight", 0.0))

    late_clean_component_weight = float(late_t_cfg.get("clean_component_weight", 0.0))

    late_clean_fp_weight = float(late_t_cfg.get("clean_fp_weight", 0.0))

    late_t_start_frac = max(0.0, min(1.0, float(late_t_cfg.get("start_frac", 0.65))))

    geometry_focus_cfg = fitness_cfg.get("geometry_focus", {})

    geometry_transition_reward_scale = float(geometry_focus_cfg.get("transition_reward_scale", 0.7))

    geometry_transition_penalty_scale = float(geometry_focus_cfg.get("transition_penalty_scale", 0.45))

    geometry_transition_cleanliness_scale = float(geometry_focus_cfg.get("transition_cleanliness_scale", 0.5))

    geometry_late_structure_boost = float(geometry_focus_cfg.get("late_structure_boost", 1.3))

    geometry_late_penalty_boost = float(geometry_focus_cfg.get("late_penalty_boost", 1.15))

    geometry_late_cleanliness_boost = float(geometry_focus_cfg.get("late_cleanliness_boost", 1.1))

    geometry_early_structure_scale = float(geometry_focus_cfg.get("early_structure_scale", 1.0))

    geometry_early_penalty_scale = float(geometry_focus_cfg.get("early_penalty_scale", 1.0))

    geometry_early_clean_scale = float(geometry_focus_cfg.get("early_clean_scale", 1.0))

    geometry_early_stop_frac = max(0.0, min(1.0, float(geometry_focus_cfg.get("early_stop_frac", 0.0))))

    cleanliness_cfg = fitness_cfg.get("cleanliness", {})

    clean_component_start = float(cleanliness_cfg.get("component_weight_start", 0.04))

    clean_component_target = float(

        cleanliness_cfg.get("component_weight_target", late_clean_component_weight)

    )

    clean_fp_start = float(cleanliness_cfg.get("fp_weight_start", 0.03))

    clean_fp_target = float(cleanliness_cfg.get("fp_weight_target", late_clean_fp_weight))

    clean_ramp_start = float(cleanliness_cfg.get("ramp_start", 0.2))

    clean_ramp_end = float(cleanliness_cfg.get("ramp_end", 0.85))

    empty_collapse_cfg = fitness_cfg.get("empty_collapse", {})

    empty_collapse_area_floor = float(empty_collapse_cfg.get("area_floor", 0.0))

    empty_collapse_coverage_floor = float(empty_collapse_cfg.get("coverage_floor", 0.0))

    empty_collapse_penalty_weight = float(empty_collapse_cfg.get("penalty_weight", 0.0))

    empty_collapse_suppress_scale = float(empty_collapse_cfg.get("suppress_scale", 0.0))

    empty_candidate_area_floor = float(empty_collapse_cfg.get("candidate_area_floor", empty_collapse_area_floor))

    empty_candidate_coverage_floor = float(

        empty_collapse_cfg.get("candidate_coverage_floor", empty_collapse_coverage_floor)

    )

    phase_transition_cfg = cfg.get("phase_transition", {})

    transition_cleanliness_scale = float(phase_transition_cfg.get("cleanliness_scale", 0.6))

    transition_coverage_floor = float(phase_transition_cfg.get("coverage_floor", 0.4))

    transition_mutation_std = float(phase_transition_cfg.get("mutation_std", 0.015))

    transition_bridge_weight = float(phase_transition_cfg.get("bridge_weight", 0.7))

    transition_bridge_min = float(phase_transition_cfg.get("bridge_min_weight", transition_bridge_weight))

    transition_gate_power = float(phase_transition_cfg.get("gate_power", 1.0))

    transition_reward_bias = float(phase_transition_cfg.get("reward_bias", 0.5))

    transition_penalty_relief = float(phase_transition_cfg.get("penalty_relief", 0.5))

    transition_t_phase_gate_start = float(phase_transition_cfg.get("t_phase_gate_start", 0.25))

    transition_structure_gate_start = float(phase_transition_cfg.get("structure_gate_start", 0.35))

    transition_clean_gate_start = float(phase_transition_cfg.get("cleanliness_gate_start", 0.0))

    transition_collapse_area_floor = float(phase_transition_cfg.get("collapse_area_floor", 0.0))

    transition_collapse_coverage_floor = float(phase_transition_cfg.get("collapse_coverage_floor", 0.0))

    transition_collapse_penalty_scale = float(phase_transition_cfg.get("collapse_penalty_scale", 0.5))

    transition_collapse_suppress_scale = float(phase_transition_cfg.get("collapse_suppress_scale", 0.5))

    transition_bridge_hold_frac = max(0.0, min(0.95, float(phase_transition_cfg.get("bridge_hold_frac", 0.25))))

    transition_mutation_warmup_frac = max(0.0, min(1.0, float(phase_transition_cfg.get("mutation_warmup_frac", 0.45))))

    transition_mutation_warmup_power = max(0.1, float(phase_transition_cfg.get("mutation_warmup_power", 1.0)))

    population_size = int(ga_cfg.get("population_size", 32))

    elite_frac = float(ga_cfg.get("elite_frac", 0.2))

    mutation_prob = float(ga_cfg.get("mutation_prob", 0.1))

    mutation_std = float(ga_cfg.get("mutation_std", 0.01))

    adaptive_mutation_cfg = AdaptiveMutationConfig.from_dict(
        ga_cfg.get("adaptive_mutation")
    )

    n_elites_unchanged = max(0, int(ga_cfg.get("n_elites_unchanged", 4)))

    eval_steps = int(ga_cfg.get("eval_steps", world_cfg.get("steps", 200)))

    viz_every = max(1, int(ga_cfg.get("viz_every_generations", 5)))

    checkpoint_every = max(0, int(ga_cfg.get("checkpoint_every_generations", 10)))

    plateau_generations = max(0, int(ga_cfg.get("plateau_generations", 0)))

    plateau_min_delta = float(ga_cfg.get("plateau_min_delta", 1e-4))

    plateau_stop_entire_run = bool(ga_cfg.get("plateau_stop_entire_run", False))

    plateau_warmup_generations_t = max(0, int(ga_cfg.get("plateau_warmup_generations_T", 0)))

    plateau_patience_t = max(0, int(ga_cfg.get("plateau_patience_t", plateau_generations)))

    t_sparse_guard_cfg = ga_cfg.get("t_sparse_guard", {})

    t_sparse_guard_enabled = bool(t_sparse_guard_cfg.get("enabled", False))

    float(t_sparse_guard_cfg.get("coverage_floor", 0.4))

    float(t_sparse_guard_cfg.get("area_floor", 0.4))

    float(t_sparse_guard_cfg.get("penalty", 2.5))

    float(t_sparse_guard_cfg.get("area_weight", 1.0))

    t_sparse_guard_stop_frac = max(0.0, min(1.0, float(t_sparse_guard_cfg.get("stop_frac", 0.4))))

    t_sparse_guard_reward_boost = float(t_sparse_guard_cfg.get("reward_boost", 1.0))

    t_sparse_guard_penalty_boost = float(t_sparse_guard_cfg.get("penalty_boost", 1.0))

    t_sparse_guard_cleanliness_scale = float(t_sparse_guard_cfg.get("cleanliness_scale", 1.0))

    late_t_guard_cfg = ga_cfg.get("late_t_guard", {})

    late_t_guard_enabled = bool(late_t_guard_cfg.get("enabled", False))

    late_t_guard_coverage_floor = float(late_t_guard_cfg.get("coverage_floor", 0.5))

    late_t_guard_trunk_floor = float(late_t_guard_cfg.get("trunk_floor", 0.35))

    late_t_guard_junction_floor = float(late_t_guard_cfg.get("junction_floor", 0.35))

    late_t_guard_area_floor = float(late_t_guard_cfg.get("area_floor", 0.6))

    late_t_guard_penalty = float(late_t_guard_cfg.get("penalty", 2.0))

    late_t_guard_iou_margin = float(late_t_guard_cfg.get("iou_margin", 0.05))

    late_t_guard_cooldown = max(0, int(late_t_guard_cfg.get("cooldown_generations", 8)))

    late_t_guard_area_weight = float(late_t_guard_cfg.get("area_weight", 1.0))

    late_t_guard_reinject = bool(late_t_guard_cfg.get("reinject_champion", True))

    late_t_guard_cleanliness_scale = float(late_t_guard_cfg.get("cleanliness_scale", 0.75))

    late_t_guard_alignment_floor = float(late_t_guard_cfg.get("alignment_floor", 0.4))

    late_t_guard_bar_floor = float(late_t_guard_cfg.get("bar_floor", 0.35))

    late_t_guard_empty_floor = float(late_t_guard_cfg.get("empty_area_floor", 0.25))

    late_t_guard_history_window = max(1, int(late_t_guard_cfg.get("history_window", 24)))

    late_t_guard_recent_window = max(1, int(late_t_guard_cfg.get("recent_window", 6)))

    late_t_guard_min_viable_history = max(1, int(late_t_guard_cfg.get("min_viable_history", 5)))

    late_t_guard_regression_iou_drop = float(late_t_guard_cfg.get("regression_drop_iou", 0.12))

    late_t_guard_regression_coverage_drop = float(late_t_guard_cfg.get("regression_drop_coverage", 0.25))

    late_t_guard_regression_area_drop = float(late_t_guard_cfg.get("regression_drop_area", 0.25))

    late_t_guard_regression_trunk_drop = float(late_t_guard_cfg.get("regression_drop_trunk", 0.3))

    late_t_guard_regression_junction_drop = float(late_t_guard_cfg.get("regression_drop_junction", 0.3))

    late_t_guard_gate_strength = float(late_t_guard_cfg.get("gate_strength", 0.85))

    late_t_guard_regression_gate_boost = float(late_t_guard_cfg.get("regression_gate_boost", 1.3))

    late_t_guard_pool_size = max(1, int(late_t_guard_cfg.get("champion_pool_size", 3)))

    late_t_guard_reinject_fraction = max(0.0, min(1.0, float(late_t_guard_cfg.get("reinject_fraction", 0.1))))

    late_t_guard_reinject_max_slots = max(1, int(late_t_guard_cfg.get("reinject_max_slots", 4)))

    stem_phase_cleanup_start_frac = max(0.0, min(1.0, float(ga_cfg.get("stem_phase_cleanup_start_frac", 0.75))))

    stem_phase_cleanup_power = max(0.1, float(ga_cfg.get("stem_phase_cleanup_power", 1.5)))

    mini_transfer_cfg = ga_cfg.get("mini_transfer", {})

    mini_transfer_enabled = bool(mini_transfer_cfg.get("enabled", False))

    mini_transfer_top_k = max(1, int(mini_transfer_cfg.get("top_k", 4)))

    mini_transfer_every = max(1, int(mini_transfer_cfg.get("every_generations", 5)))

    mini_transfer_eval_steps = int(mini_transfer_cfg.get("eval_steps", eval_steps))

    mini_transfer_seeds_cfg = mini_transfer_cfg.get("seeds")

    if mini_transfer_seeds_cfg:

        mini_transfer_seeds = [int(val) for val in mini_transfer_seeds_cfg]

    else:

        mini_transfer_seeds = [seed + 91 + idx * 2 for idx in range(3)]

    t_mutation_std_start = float(ga_cfg.get("t_mutation_std_start", mutation_std * 0.5))

    t_mutation_std_end = float(ga_cfg.get("t_mutation_std_end", mutation_std * 0.25))

    t_mutation_std_floor = float(ga_cfg.get("t_mutation_std_floor", 0.0015))

    t_mutation_std_cap = float(

        ga_cfg.get("t_mutation_std_cap", max(t_mutation_std_start, mutation_std * 0.75))

    )

    t_mutation_warmup_frac = max(1e-3, float(ga_cfg.get("t_mutation_warmup_frac", 0.35)))

    t_mutation_ease_power = max(0.5, float(ga_cfg.get("t_mutation_ease_power", 1.2)))

    late_t_guard_mutation_cap = float(late_t_guard_cfg.get("regression_mutation_cap", t_mutation_std_cap * 0.6))

    champion_cfg = ga_cfg.get("champion", {})

    champion_min_train_iou = float(champion_cfg.get("min_train_iou", 0.22))

    champion_min_coverage = float(champion_cfg.get("min_coverage", 0.45))

    champion_min_area_ratio = float(champion_cfg.get("min_area_ratio", 0.55))

    sparse_rescue_cfg = ga_cfg.get("sparse_rescue", {})

    sparse_rescue_enabled = bool(sparse_rescue_cfg.get("enabled", False))

    sparse_rescue_area_floor = float(

        sparse_rescue_cfg.get("area_ratio_floor", max(0.5, sparse_area_floor))

    )

    sparse_rescue_coverage_floor = float(

        sparse_rescue_cfg.get("coverage_floor", max(0.45, coverage_target_floor))

    )

    sparse_rescue_share_threshold = float(sparse_rescue_cfg.get("share_threshold", 0.6))

    sparse_rescue_mutation_boost = float(sparse_rescue_cfg.get("mutation_std_boost", 0.02))

    t_sparse_rescue_cap = float(

        sparse_rescue_cfg.get("t_phase_mutation_cap", t_mutation_std_cap)

    )

    sparse_rescue_cleanliness_scale = float(sparse_rescue_cfg.get("cleanliness_scale", 0.4))

    benchmark_modes: List[Dict[str, Any]] = []

    def build_benchmark_mode(mode_name: str, mode_cfg: Optional[Dict[str, Any]]) -> None:

        if not mode_cfg or not mode_cfg.get("enabled", False):

            return

        seeds = (

            mode_cfg.get("fixed_seeds")

            or benchmark_cfg.get("fixed_seeds")

            or [seed + 17 * idx for idx in range(5)]

        )

        seeds = [int(val) for val in seeds]

        reach_thr = float(

            mode_cfg.get(

                "reach_threshold_iou",

                train_stability_cfg.get("reach_threshold_iou", 0.0),

            )

        )

        stabilize_thr = float(

            mode_cfg.get(

                "stabilize_threshold_iou",

                train_stability_cfg.get("stabilize_threshold_iou", reach_thr),

            )

        )

        hold_window = int(

            mode_cfg.get(

                "hold_window_steps", train_stability_cfg.get("hold_window_steps", 0)

            )

        )

        plateau_delta = float(mode_cfg.get("benchmark_plateau_min_delta", 1e-3))

        cfg_obj = BenchmarkConfig(

            mode=mode_name,

            name=str(mode_cfg.get("name", f"benchmark_{mode_name}")),

            fixed_seeds=seeds,

            reach_threshold_iou=reach_thr,

            stabilize_threshold_iou=stabilize_thr,

            hold_window_steps=hold_window,

            min_reach_rate=float(mode_cfg.get("benchmark_pass_min_reach_rate", 0.7)),

            min_stabilized_rate=float(mode_cfg.get("benchmark_pass_min_stabilized_rate", 0.5)),

            min_mean_final_iou=float(mode_cfg.get("benchmark_pass_min_mean_final_iou", 0.7)),

            eval_steps=int(mode_cfg.get("eval_steps", eval_steps)),

            plateau_min_delta=plateau_delta,

        )

        mode_state = {

            "mode": mode_name,

            "config": cfg_obj,

            "run_every": int(mode_cfg.get("every_n_generations", 0)),

            "run_on_phase_end": bool(mode_cfg.get("run_on_phase_end", True)),

            "run_on_plateau": bool(mode_cfg.get("run_on_plateau", False)),

            "run_on_final": bool(mode_cfg.get("run_on_final", True)),

            "last_gen": None,

            "plateau_runs": 0,

            "plateau_limit": max(1, int(mode_cfg.get("plateau_runs", 3))),

            "best_stabilized": 0.0,

            "best_final_iou": 0.0,

            "champion_min_score": float(

                mode_cfg.get(

                    "champion_min_score", 0.3 if mode_name == "baseline" else 0.8

                )

            ),

        }

        benchmark_modes.append(mode_state)

    if "baseline" in benchmark_cfg or "target" in benchmark_cfg:

        build_benchmark_mode("baseline", benchmark_cfg.get("baseline"))

        build_benchmark_mode("target", benchmark_cfg.get("target"))

    elif benchmark_cfg.get("enabled"):

        build_benchmark_mode("target", benchmark_cfg)

    benchmark_enabled = bool(benchmark_modes)

    target_mode_state = next((mode for mode in benchmark_modes if mode["mode"] == "target"), None)

    warmup = int(simulate_cfg.get("anti_extinction_warmup_steps", 0))

    die_cap_schedule = simulate_cfg.get("die_cap_schedule")

    die_cap_frac = simulate_cfg.get("post_warmup_die_cap_frac", None)

    late_cleanup_start_frac = max(0.0, min(1.0, float(simulate_cfg.get("late_cleanup_start_frac", 0.8))))

    writer, run_dir, _ = build_writer(cfg.get("logging", {}).get("run_name", "baseline"))

    gifs_dir = run_dir / "gifs"

    gifs_train_dir = gifs_dir / "train"

    gifs_dir / "benchmark"

    checkpoints_dir = run_dir / "checkpoints"

    benchmark_dir = run_dir / "benchmark"

    run_dir / "reports"

    generations_csv_path = run_dir / "generations.csv"

    run_summary_path = run_dir / "run_summary.json"

    champion_path = run_dir / "champion.pt"

    late_t_champion_path = run_dir / "late_t_champion.pt"

    if checkpoint_every > 0:

        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    width = int(world_cfg.get("width", 15))

    height = int(world_cfg.get("height", 15))

    base_key = str(default_target_name).lower()

    target_cache: Dict[str, torch.Tensor] = {base_key: default_target_mask}

    target_area_cache: Dict[str, float] = {base_key: default_target_area}

    generation_records: List[Dict[str, Any]] = []

    phase_records: List[Dict[str, Any]] = []

    benchmark_history: List[Dict[str, Any]] = []

    benchmark_results_map: Dict[str, Dict[str, Any]] = {}

    target_progress = {"best_stabilized": 0.0, "best_mean_final_iou": 0.0}

    run_status = "training"

    status_timeline: List[Dict[str, Any]] = [{"status": run_status, "generation": 0}]

    best_overall_metrics: Optional[Dict[str, Any]] = None

    last_mini_transfer_summary: Optional[Dict[str, Any]] = None

    def update_status(new_status: str, generation: int) -> None:

        nonlocal run_status

        if new_status == run_status:

            return

        run_status = new_status

        status_timeline.append({"status": new_status, "generation": generation})

    _simple_mean = _agent_simple_mean
    _simple_variance = _agent_simple_variance

    def resolve_target_mask(name: str) -> Tuple[torch.Tensor, float]:

        key = str(name or default_target_name).lower()

        if key not in target_cache:

            mask = make_target(name, height, width)

            target_cache[key] = mask

            target_area_cache[key] = float(mask.to(dtype=torch.bool).sum().item())

        return target_cache[key], target_area_cache[key]

    benchmark_passed = False

    stop_training = False

    champion_score: Optional[float] = None

    champion_mode: Optional[str] = None

    champion_details: Optional[Dict[str, Any]] = None

    champion_source: Optional[str] = None

    sparse_collapse_active = False

    sparse_collapse_flag = 0

    sparse_population_share = 0.0

    late_t_guard_viable = False

    late_t_champion_model = None

    late_t_champion_metrics = None

    late_t_champion_generation = None

    late_t_regression_counter = 0

    late_t_metrics_history: Deque[Dict[str, float]] = deque(maxlen=late_t_guard_history_window if late_t_guard_enabled else 1)

    late_t_champion_pool: Deque[Dict[str, Any]] = deque(maxlen=late_t_guard_pool_size if late_t_guard_enabled else 1)

    compute_benchmark_score = _agent_bench_score
    compute_benchmark_dict_score = _agent_bench_dict_score

    # Benchmark evaluation (run_benchmark_mode + execute_benchmarks) was
    # extracted to agents/train_agent/benchmark_runner.py. The mutable
    # nonlocal state (benchmark_passed, stop_training, champion_score,
    # champion_mode, champion_details, champion_source) is consolidated
    # into BenchmarkState which is passed by reference. The thin wrappers
    # below build the immutable BenchmarkContext per-call (because some
    # referenced objects like latest_candidate_path are defined later in
    # pipeline.py and need to be captured at call-time, not init-time).
    _bench_state = _benchmark_module.BenchmarkState()

    def run_benchmark_mode(mode_state: Dict[str, Any], reason: str) -> None:
        nonlocal benchmark_passed, stop_training, champion_score, champion_mode, champion_details, champion_source
        ctx = _benchmark_module.BenchmarkContext(
            run_dir=run_dir,
            latest_candidate_path=latest_candidate_path,
            champion_path=champion_path,
            benchmark_dir=benchmark_dir,
            model_cfg=model_cfg,
            world_cfg=world_cfg,
            simulate_cfg=simulate_cfg,
            fitness_cfg=fitness_cfg,
            default_target_mask=default_target_mask,
            default_target_area=default_target_area,
            device=device,
            t_phase_weights=t_phase_weights if t_phase_weights else None,
            writer=writer,
            target_progress=target_progress,
            benchmark_history=benchmark_history,
            benchmark_results_map=benchmark_results_map,
            prepare_model_fn=prepare_model,
            compute_benchmark_score_fn=compute_benchmark_score,
            update_status_fn=update_status,
        )
        _benchmark_module.run_benchmark_mode(
            mode_state, reason,
            ctx=ctx, state=_bench_state,
            global_gen=global_gen,
            latest_candidate_meta=latest_candidate_meta,
        )
        # Sync nonlocals back from mutable state
        benchmark_passed = _bench_state.benchmark_passed
        stop_training = _bench_state.stop_training
        champion_score = _bench_state.champion_score
        champion_mode = _bench_state.champion_mode
        champion_details = _bench_state.champion_details
        champion_source = _bench_state.champion_source

    def execute_benchmarks(triggered: List[Tuple[Dict[str, Any], str]]) -> None:
        nonlocal benchmark_passed, stop_training, champion_score, champion_mode, champion_details, champion_source
        _benchmark_module.execute_benchmarks(
            triggered,
            benchmark_enabled=benchmark_enabled,
            ctx=_benchmark_module.BenchmarkContext(
                run_dir=run_dir,
                latest_candidate_path=latest_candidate_path,
                champion_path=champion_path,
                benchmark_dir=benchmark_dir,
                model_cfg=model_cfg,
                world_cfg=world_cfg,
                simulate_cfg=simulate_cfg,
                fitness_cfg=fitness_cfg,
                default_target_mask=default_target_mask,
                default_target_area=default_target_area,
                device=device,
                t_phase_weights=t_phase_weights if t_phase_weights else None,
                writer=writer,
                target_progress=target_progress,
                benchmark_history=benchmark_history,
                benchmark_results_map=benchmark_results_map,
                prepare_model_fn=prepare_model,
                compute_benchmark_score_fn=compute_benchmark_score,
                update_status_fn=update_status,
            ),
            state=_bench_state,
            global_gen=global_gen,
            latest_candidate_meta=latest_candidate_meta,
        )
        # Sync nonlocals back from mutable state
        benchmark_passed = _bench_state.benchmark_passed
        stop_training = _bench_state.stop_training
        champion_score = _bench_state.champion_score
        champion_mode = _bench_state.champion_mode
        champion_details = _bench_state.champion_details
        champion_source = _bench_state.champion_source

    # Mini-transfer evaluation (cross-seed candidate robustness) was
    # extracted to agents/train_agent/mini_transfer.py. The context
    # dataclass is built lazily below — AFTER scale_t_weights_values()
    # is defined — so transfer_weights can be computed using the
    # current t-weight scaling state at call time. The original closure
    # captured 30+ variables; consolidating them into a dataclass makes
    # the extraction testable without passing a 30-arg signature.
    def run_mini_transfer_eval(candidate_indices: List[int]) -> Optional[Dict[str, Any]]:
        transfer_weights = scale_t_weights_values(
            geometry_late_structure_boost,
            geometry_late_penalty_boost,
            geometry_late_cleanliness_boost,
        )
        stability_cfg_eval = (
            train_stability_cfg if train_stability_cfg.get("enabled") else None
        )
        ctx = _mini_transfer_module.MiniTransferContext(
            device=device,
            world_cfg=world_cfg,
            eval_steps=mini_transfer_eval_steps,
            warmup=warmup,
            die_cap_schedule=die_cap_schedule,
            die_cap_frac=die_cap_frac,
            late_cleanup_start_frac=late_cleanup_start_frac,
            alpha_fp=alpha_fp,
            beta_fn=beta_fn,
            gamma_area=gamma_area,
            stem_penalty_scale=stem_penalty_scale,
            stem_cleanup_multiplier=stem_cleanup_multiplier,
            stem_corridor_start_scale=stem_corridor_start_scale,
            stem_corridor_end_scale=stem_corridor_end_scale,
            t_symmetry_weight=t_symmetry_weight,
            t_trunk_weight=t_trunk_weight,
            clean_component_target=clean_component_target,
            clean_fp_target=clean_fp_target,
            coverage_reward_weight=coverage_reward_weight,
            coverage_target_floor=coverage_target_floor,
            coverage_penalty_weight=coverage_penalty_weight,
            sparse_area_floor=sparse_area_floor,
            sparse_penalty_weight=sparse_penalty_weight,
            default_target_mask=default_target_mask,
            default_target_area=default_target_area,
            default_target_label=default_target_label,
            transfer_t_weights=transfer_weights,
            stability_cfg_eval=stability_cfg_eval,
            seeds=mini_transfer_seeds,
            simple_mean=_simple_mean,
            simple_variance=_simple_variance,
            prepare_world_fn=prepare_world,
        )
        return _mini_transfer_module.run_mini_transfer_eval(
            candidate_indices, population, ctx
        )

    phases: List[Dict[str, Any]]

    if curriculum_cfg.get("enabled"):

        phases = curriculum_cfg.get("phases", [])

        if not phases:

            raise ValueError("Curriculum enabled but no phases defined in config.")

    else:

        phases = [

            {

                "name": f"phase_{default_target_name}",

                "target": default_target_name,

                "generations": int(ga_cfg.get("generations", 0)),

            }

        ]

    # Model factory for init_population / mutation rebuild paths. Inlined
    # from a former make_model() closure; prepare_model already returns
    # a model in eval() mode, so no explicit .eval() call is needed.
    population = init_population(
        population_size, lambda: prepare_model(model_cfg)
    )

    best_overall_score = float("-inf")

    best_path = run_dir / "best.pt"

    latest_candidate_path = run_dir / "latest_candidate.pt"

    latest_candidate_meta = {"generation": 0, "phase": None}

    eval_counter = 0

    global_gen = 0

    phase_best_scores: Dict[str, float] = {}

    phase_best_iou_tracker: Dict[str, float] = {}

    gif_cfg = viz_cfg.get("gif", {})

    gif_enabled = bool(gif_cfg.get("enabled", False))

    gif_every = max(1, int(gif_cfg.get("every_generations", viz_every)))

    gif_frame_stride = max(1, int(gif_cfg.get("frame_every_steps", viz_cfg.get("render_every_steps", 10))))

    gif_max_frames = int(gif_cfg.get("max_frames", 300))

    gif_name = gif_cfg.get("out_name", "best.gif")

    gif_keep_history = bool(gif_cfg.get("keep_history", False))

    _t_mut_cfg = TMutationConfig(
        ease_power=t_mutation_ease_power,
        std_start=t_mutation_std_start,
        std_end=t_mutation_std_end,
        std_floor=t_mutation_std_floor,
        std_cap=t_mutation_std_cap,
        warmup_frac=t_mutation_warmup_frac,
    )

    def resolve_mutation_std(phase_target: str, phase_name: str, phase_step: int, total_steps: int) -> float:
        return _agent_resolve_mut_std(
            phase_target, phase_name, phase_step, total_steps,
            default_std=mutation_std, t_cfg=_t_mut_cfg,
        )

    def resolve_stem_penalty_multiplier(phase_target: str, phase_name: str, phase_step: int, total_steps: int) -> float:
        return _agent_resolve_stem_penalty(phase_target, phase_name, phase_step, total_steps)

    def compute_phase_cleanup_ratio(phase_target: str, phase_step: int, total_steps: int) -> float:
        return _agent_phase_cleanup(
            phase_target, phase_step, total_steps,
            cleanup_start_frac=stem_phase_cleanup_start_frac,
            cleanup_power=stem_phase_cleanup_power,
        )

    def scale_t_weights_values(reward_scale: float, penalty_scale: float, clean_scale: float) -> Optional[Dict[str, float]]:
        return _agent_scale_t_weights(base_t_phase_weights, reward_scale, penalty_scale, clean_scale)

    t_weight_reward_scale = 1.0

    t_weight_penalty_scale = 1.0

    t_weight_clean_scale = 1.0

    def set_t_weight_scales(reward: float, penalty: float, clean: float) -> None:

        nonlocal t_weight_reward_scale, t_weight_penalty_scale, t_weight_clean_scale

        t_weight_reward_scale = max(0.0, float(reward))

        t_weight_penalty_scale = max(0.0, float(penalty))

        t_weight_clean_scale = max(0.0, float(clean))

    set_t_weight_scales(1.0, 1.0, 1.0)

    def make_evaluator(

        target_mask: torch.Tensor,

        target_area_val: float,

        *,

        target_label: str,

        blend_from: Optional[Dict[str, Any]] = None,

    ):

        target_bool = target_mask.to(dtype=torch.bool)

        label_lower = str(target_label or "").lower()

        target_sym_weight = t_symmetry_weight if label_lower == "t" else 0.0

        target_trunk_weight = t_trunk_weight if label_lower == "t" else 0.0

        enable_t_metrics = label_lower == "t"

        # Mutable evaluator state — former 15 nonlocal variables + 6 setter
        # closures consolidated into EvaluatorState dataclass (see
        # agents/train_agent/evaluator.py). Setters are now methods on the
        # dataclass; the evaluator closure reads state through `_eval_state`.
        _eval_state = EvaluatorState()

        blend_state: Optional[Dict[str, Any]] = None

        if blend_from is not None:

            blend_mask = blend_from.get("mask")

            blend_area = float(blend_from.get("area", 0.0))

            blend_label = str(blend_from.get("label") or "").lower()

            blend_bool = blend_mask.to(dtype=torch.bool) if blend_mask is not None else None

            if blend_bool is not None:

                blend_state = {

                    "mask": blend_bool,

                    "area": blend_area,

                    "sym_weight": t_symmetry_weight if blend_label == "t" else 0.0,

                    "trunk_weight": t_trunk_weight if blend_label == "t" else 0.0,

                }

        def evaluator(model: LittleLM, idx: int) -> Dict[str, Any]:

            nonlocal eval_counter

            world_seed = seed + eval_counter

            eval_counter += 1

            world = prepare_world(world_cfg, world_seed)

            model.to(device)

            sim_result = simulate(

                world=world,

                model=model,

                target_mask=target_bool,

                steps=eval_steps,

                writer=None,

                viz=None,

                render_every=eval_steps,

                device=device,

                anti_extinction_warmup_steps=warmup,

                die_cap_schedule=die_cap_schedule,

                post_warmup_die_cap_frac=die_cap_frac,

                verbose=False,

                late_cleanup_start_frac=late_cleanup_start_frac,

                stability_cfg=train_stability_cfg if enable_t_metrics and train_stability_cfg.get("enabled") else None,

            )

            cleanup_ratio = resolve_cleanup_ratio(sim_result, eval_steps, late_cleanup_start_frac)

            model.to("cpu")

            scaled_stem_penalty = stem_penalty_scale * _eval_state.stem_scale_multiplier

            alpha_fp_dynamic = alpha_fp

            if label_lower == "t":

                alpha_fp_dynamic = alpha_fp * (1.0 + 0.08 * cleanup_ratio)

                alpha_fp_dynamic *= max(0.0, _eval_state.late_stage_fp_multiplier)

            sym_weight = target_sym_weight

            trunk_weight = target_trunk_weight

            clean_component_weight = 0.0

            clean_fp_weight = 0.0

            if label_lower == "t":

                sym_weight += _eval_state.late_stage_symmetry_bonus

                trunk_weight += _eval_state.late_stage_trunk_bonus

                clean_component_weight = _eval_state.late_stage_clean_component

                clean_fp_weight = _eval_state.late_stage_clean_fp

            collapse_area_floor_local = _eval_state.collapse_area_floor_value if enable_t_metrics else 0.0

            collapse_coverage_floor_local = _eval_state.collapse_coverage_floor_value if enable_t_metrics else 0.0

            dynamic_t_weights = None

            if enable_t_metrics:

                dynamic_t_weights = scale_t_weights_values(

                    t_weight_reward_scale,

                    t_weight_penalty_scale,

                    t_weight_clean_scale,

                )

            metrics = compute_fitness(

                world.grid,

                target_bool,

                alpha_fp=alpha_fp_dynamic,

                beta_fn=beta_fn,

                alive_end=sim_result["alive_end"],

                target_area=target_area_val,

                gamma_area=gamma_area,

                stem_penalty_scale=scaled_stem_penalty,

                stem_cleanup_multiplier=stem_cleanup_multiplier,

                stem_cleanup_ratio=cleanup_ratio,

                stem_phase_cleanup_ratio=_eval_state.phase_cleanup_ratio,

                stem_corridor_start_scale=stem_corridor_start_scale,

                stem_corridor_end_scale=stem_corridor_end_scale,

                t_symmetry_weight=sym_weight,

                t_trunk_weight=trunk_weight,

                clean_component_weight=clean_component_weight,

                clean_fp_weight=clean_fp_weight,

                stability_metrics=sim_result if enable_t_metrics and train_stability_cfg.get("enabled") else None,

                t_benchmark_weights=dynamic_t_weights if enable_t_metrics else None,

                expect_t_shape=enable_t_metrics,

                coverage_weight=coverage_reward_weight,

                coverage_floor=coverage_target_floor,

                coverage_penalty_weight=coverage_penalty_weight,

                sparse_area_floor=sparse_area_floor,

                sparse_penalty_weight=sparse_penalty_weight,

                t_phase_gate=_eval_state.t_phase_gate_value if enable_t_metrics else 1.0,

                t_structure_gate=_eval_state.t_structure_gate_value if enable_t_metrics else 1.0,

                t_cleanliness_gate=_eval_state.t_cleanliness_gate_value if enable_t_metrics else 1.0,

                collapse_area_floor=collapse_area_floor_local,

                collapse_coverage_floor=collapse_coverage_floor_local,

                collapse_penalty_weight=_eval_state.collapse_penalty_weight_value if enable_t_metrics else 0.0,

                collapse_bonus_suppression=_eval_state.collapse_suppress_value if enable_t_metrics else 0.0,

            )

            fitness_val = metrics["fitness"]

            if blend_state is not None and blend_state.get("mask") is not None:

                blend_metrics = compute_fitness(

                    world.grid,

                    blend_state["mask"],

                    alpha_fp=alpha_fp,

                    beta_fn=beta_fn,

                    alive_end=sim_result["alive_end"],

                    target_area=blend_state.get("area"),

                    gamma_area=gamma_area,

                    stem_penalty_scale=scaled_stem_penalty,

                    stem_cleanup_multiplier=stem_cleanup_multiplier,

                    stem_cleanup_ratio=cleanup_ratio,

                    stem_phase_cleanup_ratio=_eval_state.phase_cleanup_ratio,

                    stem_corridor_start_scale=stem_corridor_start_scale,

                    stem_corridor_end_scale=stem_corridor_end_scale,

                    t_symmetry_weight=blend_state.get("sym_weight", 0.0),

                    t_trunk_weight=blend_state.get("trunk_weight", 0.0),

                    clean_component_weight=0.0,

                    clean_fp_weight=0.0,

                    coverage_weight=coverage_reward_weight,

                    coverage_floor=coverage_target_floor,

                    coverage_penalty_weight=coverage_penalty_weight,

                    sparse_area_floor=sparse_area_floor,

                    sparse_penalty_weight=sparse_penalty_weight,

                    t_phase_gate=1.0,

                    t_structure_gate=1.0,

                    t_cleanliness_gate=1.0,

                    collapse_area_floor=empty_collapse_area_floor if default_target_label == "t" else 0.0,

                    collapse_coverage_floor=empty_collapse_coverage_floor if default_target_label == "t" else 0.0,

                    collapse_penalty_weight=empty_collapse_penalty_weight if default_target_label == "t" else 0.0,

                    collapse_bonus_suppression=empty_collapse_suppress_scale if default_target_label == "t" else 0.0,

                )

                blend_fitness = blend_metrics["fitness"]

                fitness_val = (1.0 - _eval_state.blend_weight) * blend_fitness + _eval_state.blend_weight * fitness_val

            result = {

                "fitness": fitness_val,

                "iou": metrics["iou"],

                "area": metrics["area"],

                "alive_end": sim_result["alive_end"],

                "anti_ext": sim_result["anti_extinction_triggers"],

                "divisions": sim_result["total_divisions"],

                "deaths": sim_result["total_deaths"],

                "world_seed": world_seed,

                "base_score": metrics["base_score"],

                "alive_bonus": metrics["alive_bonus"],

                "extinction_penalty": metrics["extinction_penalty"],

                "base_iou": metrics["iou"],

                "fp": metrics["fp"],

                "fn": metrics["fn"],

                "area_penalty2": metrics["area_penalty2"],

                "num_components": metrics["num_components"],

                "largest_component_area": metrics["largest_component_area"],

                "largest_component_ratio": metrics["largest_component_ratio"],

                "stem_count": metrics["stem_count"],

                "component_bonus": metrics["component_bonus"],

                "fragment_penalty": metrics["fragment_penalty"],

                "com_penalty": metrics["com_penalty"],

                "stem_penalty": metrics["stem_penalty"],

                "t_symmetry": metrics["t_symmetry"],

                "t_trunk_continuity": metrics["t_trunk_continuity"],

                "late_clean_penalty": metrics.get("late_clean_penalty", 0.0),

                "target_coverage_ratio": metrics.get("target_coverage_ratio", 0.0),

                "coverage_penalty": metrics.get("coverage_penalty", 0.0),

                "sparse_collapse_penalty": metrics.get("sparse_collapse_penalty", 0.0),

            }

            return result

        return evaluator, _eval_state

    phase_id_map: Dict[str, int] = {}

    for phase in phases:

        phase_name = str(phase.get("name", f"phase_{global_gen + 1}"))

        configured_target = str(phase.get("target", default_target_name))

        blend_from_target = phase.get("blend_from_target")

        blend_to_target = str(phase.get("blend_to_target", configured_target))

        phase_target = blend_to_target

        generations = int(phase.get("generations", ga_cfg.get("generations", 0)))

        if generations <= 0:

            print(f"[WARN] Phase '{phase_name}' has no generations configured. Skipping.")

            continue

        target_mask, phase_target_area = resolve_target_mask(phase_target)

        blend_info = None

        if blend_from_target:

            blend_from_name = str(blend_from_target)

            blend_mask, blend_area = resolve_target_mask(blend_from_name)

            blend_info = {

                "mask": blend_mask,

                "area": blend_area,

                "label": blend_from_name,

            }

        evaluator, eval_state = make_evaluator(

            target_mask,

            phase_target_area,

            target_label=phase_target,

            blend_from=blend_info,

        )

        phase_record = {

            "name": phase_name,

            "target": phase_target,

            "generations": generations,

            "start_generation": global_gen,

            "best_iou": 0.0,

            "best_fitness": float("-inf"),

            "best_generation": None,

        }

        target_bool = target_mask.to(dtype=torch.bool)

        phase_key = phase_target.lower()

        phase_slug = str(phase.get("log_key") or phase_name).lower().replace(' ', '_')

        if phase_slug.startswith("phase_"):

            phase_slug = phase_slug[6:]

        if not phase_slug:

            phase_slug = phase_key

        phase_name_lower = phase_name.lower()

        phase_is_transition = blend_info is not None

        phase_is_t = phase_key == "t"

        phase_is_pure_t = phase_is_t and not phase_is_transition

        phase_is_refine = ("refine" in phase_slug) or ("refine" in phase_name_lower)

        phase_plateau_generations = plateau_generations

        if phase_is_pure_t and plateau_patience_t > 0:

            phase_plateau_generations = plateau_patience_t

        phase_id = phase_id_map.setdefault(phase_key, len(phase_id_map))

        writer.add_text("curriculum/phase", phase_name, global_step=global_gen)

        writer.add_scalar("curriculum/phase_id", phase_id, global_step=global_gen)

        writer.add_scalar("stats/target_area", phase_target_area, global_step=global_gen)

        phase_best_iou = float('-inf')

        phase_best_area_gap = float('inf')

        phase_best_trunk_score = 0.0

        phase_best_components = float('inf')

        gens_since_improve = 0

        plateau_triggered = False

        for phase_step in range(generations):

            global_gen += 1

            print(f"[GEN {global_gen}/{phase_name}] evaluating pop={population_size} ...")

            writer.add_scalar("curriculum/phase_id", phase_id, global_step=global_gen)

            writer.add_scalar("stats/target_area", phase_target_area, global_step=global_gen)

            writer.add_text("phase/name", phase_name, global_step=global_gen)

            phase_cleanup_ratio = compute_phase_cleanup_ratio(phase_target, phase_step, generations)

            eval_state.set_phase_cleanup_ratio(phase_cleanup_ratio)

            writer.add_scalar("ga/stem_phase_cleanup_ratio", phase_cleanup_ratio, global_step=global_gen)

            blend_weight = 1.0

            transition_progress = 0.0

            transition_ease = 1.0

            if phase_is_transition:

                transition_progress = min(1.0, (phase_step + 1) / max(1, generations))

                eased_progress = transition_progress

                if transition_bridge_hold_frac > 0.0:

                    if eased_progress <= transition_bridge_hold_frac:

                        eased_progress = 0.0

                    else:

                        hold_span = max(1e-6, 1.0 - transition_bridge_hold_frac)

                        eased_progress = (eased_progress - transition_bridge_hold_frac) / hold_span

                transition_ease = eased_progress ** transition_gate_power

                target_weight = transition_bridge_min + (transition_bridge_weight - transition_bridge_min) * transition_ease

                blend_weight = max(min(transition_bridge_weight, target_weight), transition_bridge_min)

            eval_state.set_blend_weight(blend_weight)

            coverage_target_floor = coverage_target_floor_base

            coverage_reward_weight = coverage_reward_weight_base

            coverage_penalty_weight = coverage_penalty_weight_base

            sparse_area_floor = sparse_area_floor_base

            sparse_penalty_weight = sparse_penalty_weight_base

            reward_gate = 1.0

            penalty_gate = 1.0

            phase_progress = (phase_step + 1) / max(1, generations)

            early_sparse_guard_active = (

                t_sparse_guard_enabled

                and phase_is_pure_t

                and not phase_is_transition

                and not phase_is_refine

                and phase_progress <= t_sparse_guard_stop_frac

            )

            early_geometry_phase = (

                phase_is_pure_t

                and not phase_is_transition

                and not phase_is_refine

                and phase_progress <= geometry_early_stop_frac

            )

            late_stage_active = False

            late_regression_active = late_t_guard_enabled and late_t_regression_counter > 0

            late_stage_active = False

            if phase_is_pure_t and (phase_is_refine or phase_progress >= late_t_start_frac):

                late_stage_active = True

            if phase_is_transition:

                coverage_target_floor = min(coverage_target_floor_base, transition_coverage_floor)

                sparse_area_floor = min(sparse_area_floor_base, transition_coverage_floor)

                reward_gate = transition_reward_bias + (1.0 - transition_reward_bias) * transition_ease

                penalty_gate = transition_penalty_relief + (1.0 - transition_penalty_relief) * transition_ease

            weight_reward_scale = 1.0

            weight_penalty_scale = 1.0

            weight_clean_scale = 1.0

            if phase_is_transition:

                weight_reward_scale = geometry_transition_reward_scale + (1.0 - geometry_transition_reward_scale) * transition_ease

                weight_penalty_scale = geometry_transition_penalty_scale + (1.0 - geometry_transition_penalty_scale) * transition_ease

                weight_clean_scale = geometry_transition_cleanliness_scale + (1.0 - geometry_transition_cleanliness_scale) * transition_ease

            elif early_geometry_phase:

                weight_reward_scale = geometry_early_structure_scale

                weight_penalty_scale = geometry_early_penalty_scale

                weight_clean_scale = geometry_early_clean_scale

            elif phase_is_pure_t and late_stage_active:

                weight_reward_scale = geometry_late_structure_boost

                weight_penalty_scale = geometry_late_penalty_boost

                weight_clean_scale = geometry_late_cleanliness_boost

            set_t_weight_scales(weight_reward_scale, weight_penalty_scale, weight_clean_scale)

            coverage_reward_weight = coverage_reward_weight_base * reward_gate

            coverage_penalty_weight = coverage_penalty_weight_base * penalty_gate

            sparse_penalty_weight = sparse_penalty_weight_base * penalty_gate

            if early_sparse_guard_active:

                coverage_reward_weight *= t_sparse_guard_reward_boost

                coverage_penalty_weight *= t_sparse_guard_penalty_boost

            t_phase_gate_value = 1.0

            t_structure_gate_value = 1.0

            t_cleanliness_gate_value = 1.0

            collapse_area_floor_active = empty_collapse_area_floor if phase_is_t else 0.0

            collapse_coverage_floor_active = empty_collapse_coverage_floor if phase_is_t else 0.0

            collapse_penalty_weight_active = empty_collapse_penalty_weight if phase_is_t else 0.0

            collapse_bonus_suppress_active = empty_collapse_suppress_scale if phase_is_t else 0.0

            if phase_is_transition:

                t_phase_gate_value = transition_t_phase_gate_start + (1.0 - transition_t_phase_gate_start) * transition_ease

                t_structure_gate_value = transition_structure_gate_start + (1.0 - transition_structure_gate_start) * transition_ease

                t_cleanliness_gate_value = transition_clean_gate_start + (1.0 - transition_clean_gate_start) * transition_ease

                collapse_area_floor_active = max(0.0, min(empty_collapse_area_floor, transition_collapse_area_floor))

                collapse_coverage_floor_active = max(0.0, min(empty_collapse_coverage_floor, transition_collapse_coverage_floor))

                collapse_penalty_weight_active = max(0.0, empty_collapse_penalty_weight * transition_collapse_penalty_scale * penalty_gate)

                collapse_bonus_suppress_active = max(0.0, empty_collapse_suppress_scale * transition_collapse_suppress_scale)

            cleanliness_factor = 0.0

            cleanliness_scale_current = 1.0

            if phase_is_transition:

                cleanliness_scale_current = transition_cleanliness_scale

            if early_sparse_guard_active:

                cleanliness_scale_current *= t_sparse_guard_cleanliness_scale

            if late_regression_active and phase_is_pure_t:

                cleanliness_scale_current *= late_t_guard_cleanliness_scale

            if phase_is_t:

                if phase_progress >= clean_ramp_end:

                    cleanliness_factor = 1.0

                elif phase_progress <= clean_ramp_start:

                    cleanliness_factor = 0.0

                else:

                    span = max(1e-6, clean_ramp_end - clean_ramp_start)

                    cleanliness_factor = (phase_progress - clean_ramp_start) / span

            cleanliness_factor *= cleanliness_scale_current

            if sparse_rescue_enabled and sparse_collapse_active:

                cleanliness_factor *= sparse_rescue_cleanliness_scale

            cleanliness_factor = max(0.0, min(1.0, cleanliness_factor))

            if late_stage_active:

                stage_fp_mult = late_fp_multiplier

                stage_sym_bonus = late_symmetry_bonus

                stage_trunk_bonus = late_trunk_bonus

            else:

                stage_fp_mult = 1.0

                stage_sym_bonus = 0.0

                stage_trunk_bonus = 0.0

            stage_clean_component = 0.0

            stage_clean_fp = 0.0

            if phase_is_t:

                stage_clean_component = clean_component_start + (

                    clean_component_target - clean_component_start

                ) * cleanliness_factor

                stage_clean_fp = clean_fp_start + (clean_fp_target - clean_fp_start) * cleanliness_factor

            eval_state.set_late_stage_modifiers(

                stage_fp_mult,

                stage_sym_bonus,

                stage_trunk_bonus,

                stage_clean_component,

                stage_clean_fp,

            )

            late_enforcer_cfg = None

            if (

                late_t_guard_enabled

                and phase_is_t

                and late_stage_active

            ):

                gate_strength = late_t_guard_gate_strength

                if late_regression_active:

                    gate_strength *= late_t_guard_regression_gate_boost

                late_enforcer_cfg = {

                    "gate": gate_strength,

                    "trunk_min": late_t_guard_trunk_floor,

                    "alignment_min": late_t_guard_alignment_floor,

                    "junction_min": late_t_guard_junction_floor,

                    "coverage_min": late_t_guard_coverage_floor,

                    "area_min": late_t_guard_area_floor,

                    "empty_floor": late_t_guard_empty_floor,

                    "bar_min": late_t_guard_bar_floor,

                    "penalty": late_t_guard_penalty,

                    "area_weight": late_t_guard_area_weight,

                }

            eval_state.set_late_structure_enforcer(late_enforcer_cfg)

            eval_state.set_transition_controls(

                t_phase_gate_value,

                t_structure_gate_value,

                t_cleanliness_gate_value,

                collapse_area_floor_active,

                collapse_coverage_floor_active,

                collapse_penalty_weight_active,

                collapse_bonus_suppress_active,

            )

            writer.add_scalar("ga/late_stage_active", 1 if late_stage_active else 0, global_step=global_gen)

            stem_scale_mult = resolve_stem_penalty_multiplier(phase_target, phase_name, phase_step, generations)

            eval_state.set_stem_scale_multiplier(stem_scale_mult)

            fitnesses, metrics = evaluate_population(population, evaluator)

            if late_t_guard_enabled and late_t_regression_counter > 0 and (phase_is_pure_t or phase_is_refine) and phase_target_area > 0:
                for idx, metric in enumerate(metrics):
                    coverage_val = max(0.0, float(metric.get("target_coverage_ratio", 0.0)))
                    area_val = max(0.0, float(metric.get("area", 0.0)))
                    area_ratio = area_val / phase_target_area if phase_target_area > 0 else 0.0
                    trunk_val = max(0.0, float(metric.get("t_trunk_coverage", 0.0)))
                    junction_val = max(0.0, float(metric.get("t_junction_score", 0.0)))
                    alignment_val = max(0.0, float(metric.get("t_trunk_alignment_score", 0.0)))
                    bar_val = max(0.0, float(metric.get("t_top_bar_coverage", 0.0)))
                    pseudo_penalty = 0.0
                    if coverage_val >= late_t_guard_coverage_floor * 0.5 and trunk_val < late_t_guard_trunk_floor:
                        pseudo_penalty += late_t_guard_penalty * (late_t_guard_trunk_floor - trunk_val)
                    if junction_val < late_t_guard_junction_floor:
                        pseudo_penalty += 0.6 * late_t_guard_penalty * (late_t_guard_junction_floor - junction_val)
                    if area_ratio < late_t_guard_area_floor:
                        pseudo_penalty += 0.5 * late_t_guard_penalty * late_t_guard_area_weight * (late_t_guard_area_floor - area_ratio)
                    if coverage_val < late_t_guard_coverage_floor * 0.5:
                        pseudo_penalty += 0.25 * late_t_guard_penalty * (late_t_guard_coverage_floor * 0.5 - coverage_val)
                    if alignment_val < late_t_guard_alignment_floor:
                        pseudo_penalty += 0.35 * late_t_guard_penalty * (late_t_guard_alignment_floor - alignment_val)
                    if bar_val >= late_t_guard_bar_floor and trunk_val < late_t_guard_trunk_floor * 0.5:
                        pseudo_penalty += 0.5 * late_t_guard_penalty * (late_t_guard_trunk_floor * 0.5 - trunk_val)
                    if area_ratio < max(0.05, late_t_guard_empty_floor):
                        pseudo_penalty += 0.4 * late_t_guard_penalty * (max(0.05, late_t_guard_empty_floor) - area_ratio)
                    if pseudo_penalty > 0.0:
                        fitnesses[idx] -= pseudo_penalty
                        metric["fitness"] = fitnesses[idx]
                        metric["late_t_penalty"] = pseudo_penalty
            current_mutation_std = resolve_mutation_std(phase_target, phase_name, phase_step, generations)

            if phase_is_transition:

                mutation_cap = transition_mutation_std

                if transition_mutation_warmup_frac > 0.0 and mutation_std > transition_mutation_std:

                    warmup_window = max(1, int(generations * transition_mutation_warmup_frac))

                    warmup_progress = min(1.0, (phase_step + 1) / warmup_window)

                    warmup_gate = warmup_progress ** transition_mutation_warmup_power

                    start_cap = mutation_std

                    mutation_cap = transition_mutation_std + (start_cap - transition_mutation_std) * (1.0 - warmup_gate)

                current_mutation_std = min(current_mutation_std, mutation_cap)

            if sparse_rescue_enabled and sparse_collapse_flag:
                boost_target = mutation_std + sparse_rescue_mutation_boost
                if phase_is_t:
                    boost_target = min(boost_target, t_sparse_rescue_cap)
                current_mutation_std = max(current_mutation_std, boost_target)

            if (
                late_regression_active
                and (phase_is_pure_t or phase_is_refine)
                and late_t_guard_mutation_cap > 0.0
            ):
                current_mutation_std = min(current_mutation_std, late_t_guard_mutation_cap)

            if phase_is_t:
                current_mutation_std = min(current_mutation_std, t_mutation_std_cap)

            # Adaptive mutation — only fires if ga.adaptive_mutation.strategy != "none"
            # (default). Applied AFTER all hand-tuned caps so it can only boost σ,
            # not override existing safety limits. For the reproducibility freeze
            # (configs/reproducibility_freeze_v1.yaml) this is a strict passthrough.
            current_mutation_std, _adaptive_diag = adapt_mutation_std(
                current_mutation_std,
                fitnesses=fitnesses,
                gens_since_improve=gens_since_improve,
                cfg=adaptive_mutation_cfg,
            )
            if writer is not None and adaptive_mutation_cfg.strategy != "none":
                writer.add_scalar(
                    "ga/adaptive_mutation_std",
                    float(_adaptive_diag["adjusted_std"]),
                    global_step=global_gen,
                )
                writer.add_scalar(
                    "ga/fitness_variance",
                    float(_adaptive_diag["variance"]),
                    global_step=global_gen,
                )
                writer.add_scalar(
                    "ga/adaptive_diversity_fired",
                    1 if _adaptive_diag["diversity_boost_applied"] else 0,
                    global_step=global_gen,
                )
                writer.add_scalar(
                    "ga/adaptive_plateau_fired",
                    1 if _adaptive_diag["plateau_boost_applied"] else 0,
                    global_step=global_gen,
                )

            best_idx = int(max(range(len(fitnesses)), key=lambda idx: fitnesses[idx]))

            best_metrics = metrics[best_idx]

            best_model = population[best_idx]

            latest_candidate_meta["generation"] = global_gen

            latest_candidate_meta["phase"] = phase_name

            try:

                torch.save(best_model.state_dict(), latest_candidate_path)

            except Exception as err:  # pragma: no cover - diagnostics only

                print(f"[WARN] Failed to save latest candidate: {err}")

            # --- Early extraction of best metrics (needed by late T-guard below) ---
            best_fit = best_metrics["fitness"]
            best_iou = best_metrics["iou"]
            alive_best = best_metrics["alive_end"]
            area_best = best_metrics["area"]
            target_coverage_best = best_metrics.get("target_coverage_ratio", 0.0)
            area_ratio_best = area_best / phase_target_area if phase_target_area > 0 else 0.0
            empty_candidate_blocked = False
            if phase_is_t and (
                target_coverage_best < empty_candidate_coverage_floor
                or area_ratio_best < empty_candidate_area_floor
            ):
                empty_candidate_blocked = True

            if late_t_guard_enabled and phase_is_pure_t and not phase_is_transition:
                coverage_val = max(0.0, target_coverage_best)
                trunk_val = max(0.0, best_metrics.get("t_trunk_coverage", 0.0))
                junction_val = max(0.0, best_metrics.get("t_junction_score", 0.0))
                alignment_val = max(0.0, best_metrics.get("t_trunk_alignment_score", 0.0))
            if late_t_guard_enabled and (phase_is_pure_t or phase_is_refine):
                late_t_metrics_history.append(
                    {
                        "generation": global_gen,
                        "iou": best_iou,
                        "coverage": target_coverage_best,
                        "area_ratio": area_ratio_best,
                        "trunk": best_metrics.get("t_trunk_coverage", 0.0),
                        "junction": best_metrics.get("t_junction_score", 0.0),
                        "alignment": best_metrics.get("t_trunk_alignment_score", 0.0),
                    }
                )
                if (
                    coverage_val >= late_t_guard_coverage_floor
                    and trunk_val >= late_t_guard_trunk_floor
                    and junction_val >= late_t_guard_junction_floor
                    and area_ratio_best >= late_t_guard_area_floor
                ):
                    late_t_guard_viable = True
                    late_t_champion_generation = global_gen
                    champion_metrics = {
                        "iou": best_iou,
                        "coverage": coverage_val,
                        "area_ratio": area_ratio_best,
                        "trunk": trunk_val,
                        "junction": junction_val,
                        "alignment": alignment_val,
                    }
                    late_t_champion_metrics = champion_metrics
                    champion_clone = clone_model(best_model)
                    champion_clone.to("cpu")
                    late_t_champion_model = champion_clone
                    late_t_champion_pool.appendleft(
                        {
                            "generation": global_gen,
                            "metrics": champion_metrics,
                            "model": champion_clone,
                        }
                    )
                    if late_t_guard_reinject:
                        try:
                            torch.save(best_model.state_dict(), late_t_champion_path)
                        except Exception as err:  # pragma: no cover - diagnostics only
                            print(f"[WARN] Failed to save late T champion: {err}")
                    late_t_regression_counter = 0
            regression_triggered = False
            if late_t_guard_enabled and (phase_is_pure_t or phase_is_refine):
                coverage_val = max(0.0, target_coverage_best)
                trunk_val = max(0.0, best_metrics.get("t_trunk_coverage", 0.0))
                junction_val = max(0.0, best_metrics.get("t_junction_score", 0.0))
                alignment_val = max(0.0, best_metrics.get("t_trunk_alignment_score", 0.0))
                area_ratio_best = area_best / phase_target_area if phase_target_area > 0 else 0.0
                history_drop = False
                champion_drop = False
                if late_t_guard_viable and len(late_t_metrics_history) >= late_t_guard_min_viable_history:
                    history_entries = list(late_t_metrics_history)
                    recent_entries = history_entries[-late_t_guard_recent_window :] if late_t_guard_recent_window > 0 else history_entries
                    peak_iou = max(entry.get("iou", 0.0) for entry in history_entries)
                    peak_cov = max(entry.get("coverage", 0.0) for entry in history_entries)
                    peak_area = max(entry.get("area_ratio", 0.0) for entry in history_entries)
                    peak_trunk = max(entry.get("trunk", 0.0) for entry in history_entries)
                    peak_junction = max(entry.get("junction", 0.0) for entry in history_entries)
                    recent_mean_iou = sum(entry.get("iou", 0.0) for entry in recent_entries) / max(1, len(recent_entries))
                    if peak_iou - recent_mean_iou >= late_t_guard_regression_iou_drop:
                        history_drop = True
                    if coverage_val < peak_cov * (1.0 - late_t_guard_regression_coverage_drop):
                        history_drop = True
                    if area_ratio_best < peak_area * (1.0 - late_t_guard_regression_area_drop):
                        history_drop = True
                    if trunk_val < peak_trunk * (1.0 - late_t_guard_regression_trunk_drop):
                        history_drop = True
                    if junction_val < peak_junction * (1.0 - late_t_guard_regression_junction_drop):
                        history_drop = True
                if late_t_guard_viable and late_t_champion_metrics:
                    prev_iou = late_t_champion_metrics.get("iou", 0.0)
                    if prev_iou - best_iou > late_t_guard_iou_margin:
                        champion_drop = True
                near_empty = area_ratio_best < late_t_guard_empty_floor or coverage_val < late_t_guard_empty_floor
                pseudo_t_shape = best_metrics.get("t_top_bar_coverage", 0.0) >= max(0.3, late_t_guard_bar_floor) and trunk_val < late_t_guard_trunk_floor * 0.5
                weak_trunk = trunk_val < late_t_guard_trunk_floor * 0.85 or alignment_val < late_t_guard_alignment_floor * 0.85
                detection_ready = late_t_guard_viable
                if detection_ready and (champion_drop or history_drop or near_empty or pseudo_t_shape or weak_trunk):
                    late_t_regression_counter = late_t_guard_cooldown
                    regression_triggered = True
                elif late_t_regression_counter > 0:
                    late_t_regression_counter = max(0, late_t_regression_counter - 1)
            if regression_triggered:
                best_metrics["late_regression_flag"] = 1.0
            else:
                best_metrics["late_regression_flag"] = 1.0 if late_t_regression_counter > 0 else 0.0

            # best_fit, best_iou, alive_best, area_best — extracted earlier (before late T-guard)

            if best_fit > phase_record.get("best_fitness", float("-inf")):

                phase_record["best_fitness"] = best_fit

                phase_record["best_generation"] = global_gen

            if best_iou > phase_record.get("best_iou", 0.0):

                phase_record["best_iou"] = best_iou

                phase_record["best_generation"] = global_gen

            can_update_best_overall = True

            if phase_is_t and empty_candidate_blocked:

                coverage_gate = target_coverage_best

                if (

                    coverage_gate < champion_min_coverage

                    or area_ratio_best < champion_min_area_ratio

                ):

                    can_update_best_overall = False

            if can_update_best_overall and (

                best_overall_metrics is None

                or best_fit > best_overall_metrics.get("fitness", float("-inf"))

            ):

                best_overall_metrics = {

                    "generation": global_gen,

                    "phase": phase_name,

                    "fitness": best_fit,

                    "iou": best_iou,

                    "first_reach_step": best_metrics.get("first_reach_step"),

                    "stabilized_success": best_metrics.get("stabilized_success", False),

                    "hold_score": best_metrics.get("hold_score", 0.0),

                    "stability_score": best_metrics.get("stability_score", 0.0),

                    "model_phase": phase_name,

                    "model_generation": global_gen,

                }

            mean_fitness, p95_fitness = summarize(fitnesses, 95.0)

            alive_mean, _ = summarize([m["alive_end"] for m in metrics], 95.0)

            area_mean, _ = summarize([m["area"] for m in metrics], 95.0)

            anti_ext_mean, _ = summarize([m["anti_ext"] for m in metrics], 95.0)

            # target_coverage_best, area_ratio_best, empty_candidate_blocked — extracted earlier

            coverage_values = [m.get("target_coverage_ratio", 0.0) for m in metrics]

            coverage_mean = _simple_mean(coverage_values)

            num_components_best = best_metrics["num_components"]

            stem_count_best = best_metrics["stem_count"]

            largest_component_ratio_best = best_metrics["largest_component_ratio"]

            fp_rate_best = best_metrics["fp"]

            fn_rate_best = best_metrics["fn"]

            t_symmetry_best = best_metrics.get("t_symmetry", 0.0)

            t_trunk_best = best_metrics.get("t_trunk_continuity", 0.0)

            phase_iou_tracker = phase_best_iou_tracker.get(phase_slug, float('-inf'))

            if best_iou > phase_iou_tracker:

                phase_best_iou_tracker[phase_slug] = best_iou

                phase_iou_tracker = best_iou

            writer.add_scalar(f"phase/best_iou_{phase_slug}", phase_iou_tracker if phase_iou_tracker > float('-inf') else best_iou, global_step=global_gen)

            area_gap_best = abs(area_ratio_best - 1.0)

            morph_improved = False

            if area_gap_best + 1e-4 < phase_best_area_gap:

                phase_best_area_gap = area_gap_best

                morph_improved = True

            if (phase_is_t or phase_is_transition) and t_trunk_best > phase_best_trunk_score + 1e-3:

                phase_best_trunk_score = t_trunk_best

                morph_improved = True

            if float(num_components_best) + 1e-6 < phase_best_components:

                phase_best_components = float(num_components_best)

                morph_improved = True

            if best_iou > phase_best_iou + plateau_min_delta:

                phase_best_iou = best_iou

                gens_since_improve = 0

            elif morph_improved:

                gens_since_improve = 0

            else:

                gens_since_improve += 1

            writer.add_scalar("fitness/best", best_fit, global_step=global_gen)

            writer.add_scalar("fitness/mean", mean_fitness, global_step=global_gen)

            writer.add_scalar("fitness/p95", p95_fitness, global_step=global_gen)

            area_ratios = [

                m["area"] / phase_target_area if phase_target_area > 0 else 0.0

                for m in metrics

            ]

            sparse_flags = [

                1.0

                if (

                    coverage_values[idx] < sparse_rescue_coverage_floor

                    or area_ratios[idx] < sparse_rescue_area_floor

                )

                else 0.0

                for idx in range(len(metrics))

            ]

            sparse_population_share = sum(sparse_flags) / max(1, len(sparse_flags))

            sparse_collapse_flag = 1 if sparse_population_share >= sparse_rescue_share_threshold else 0

            writer.add_scalar("fitness/base_iou", best_metrics["base_iou"], global_step=global_gen)

            writer.add_scalar("fitness/base", best_metrics["base_score"], global_step=global_gen)

            writer.add_scalar("fitness/fp", best_metrics["fp"], global_step=global_gen)

            writer.add_scalar("fitness/fn", best_metrics["fn"], global_step=global_gen)

            writer.add_scalar("fitness/alive_bonus", best_metrics["alive_bonus"], global_step=global_gen)

            writer.add_scalar(

                "fitness/extinction_penalty", best_metrics["extinction_penalty"], global_step=global_gen

            )

            writer.add_scalar("fitness/area_penalty2", best_metrics["area_penalty2"], global_step=global_gen)

            writer.add_scalar("fitness/component_bonus", best_metrics["component_bonus"], global_step=global_gen)

            writer.add_scalar("fitness/fragment_penalty", best_metrics["fragment_penalty"], global_step=global_gen)

            writer.add_scalar("fitness/com_penalty", best_metrics["com_penalty"], global_step=global_gen)

            writer.add_scalar("fitness/stem_penalty", best_metrics["stem_penalty"], global_step=global_gen)

            writer.add_scalar("fitness/total", best_fit, global_step=global_gen)

            writer.add_scalar("fitness/late_clean_penalty", best_metrics.get("late_clean_penalty", 0.0), global_step=global_gen)

            writer.add_scalar("fitness/empty_collapse_penalty", best_metrics.get("empty_collapse_penalty", 0.0), global_step=global_gen)

            writer.add_scalar("ga/mutation_std", current_mutation_std, global_step=global_gen)

            writer.add_scalar("ga/late_regression_active", 1 if late_t_regression_counter > 0 else 0, global_step=global_gen)

            writer.add_scalar("ga/plateau_since_best", gens_since_improve, global_step=global_gen)

            writer.add_scalar("ga/plateau_generations_active", phase_plateau_generations, global_step=global_gen)

            writer.add_scalar("stats/alive_end_best", alive_best, global_step=global_gen)

            writer.add_scalar("stats/alive_end_mean", alive_mean, global_step=global_gen)

            writer.add_scalar("stats/area_ab_best", area_best, global_step=global_gen)

            writer.add_scalar("stats/area_ab_mean", area_mean, global_step=global_gen)

            area_ratio_mean = _simple_mean(area_ratios)

            writer.add_scalar("stats/area_ratio_best", area_ratio_best, global_step=global_gen)

            writer.add_scalar("stats/area_ratio_mean", area_ratio_mean, global_step=global_gen)

            writer.add_scalar("stats/target_coverage_best", target_coverage_best, global_step=global_gen)

            writer.add_scalar("stats/target_coverage_mean", coverage_mean, global_step=global_gen)

            writer.add_scalar("area_ratio_best", area_ratio_best, global_step=global_gen)

            writer.add_scalar("area_ratio_mean", area_ratio_mean, global_step=global_gen)

            writer.add_scalar("target_coverage_best", target_coverage_best, global_step=global_gen)

            writer.add_scalar("target_coverage_mean", coverage_mean, global_step=global_gen)

            writer.add_scalar("early_sparse_penalty", best_metrics.get("early_sparse_penalty", 0.0), global_step=global_gen)

            writer.add_scalar("late_t_penalty", best_metrics.get("late_t_penalty", 0.0), global_step=global_gen)
            writer.add_scalar("fitness/late_structure_penalty", best_metrics.get("late_structure_penalty", 0.0), global_step=global_gen)
            writer.add_scalar("fitness/pseudo_t_penalty", best_metrics.get("pseudo_t_penalty", 0.0), global_step=global_gen)

            writer.add_scalar("top_bar_score", best_metrics.get("t_top_bar_coverage", 0.0), global_step=global_gen)

            writer.add_scalar("trunk_coverage_score", best_metrics.get("t_trunk_coverage", 0.0), global_step=global_gen)

            writer.add_scalar("bar_endpoint_clutter", best_metrics.get("t_bar_endpoint_clutter", 0.0), global_step=global_gen)

            writer.add_scalar("stats/num_components_best", num_components_best, global_step=global_gen)

            writer.add_scalar("stats/component_count_best", num_components_best, global_step=global_gen)

            writer.add_scalar("stats/largest_component_ratio_best", largest_component_ratio_best, global_step=global_gen)

            writer.add_scalar("stats/stem_count_best", stem_count_best, global_step=global_gen)

            writer.add_scalar("stats/collapse_gate", best_metrics.get("collapse_gate", 1.0), global_step=global_gen)

            writer.add_scalar("ga/sparse_population_share", sparse_population_share, global_step=global_gen)

            writer.add_scalar("ga/sparse_collapse_flag", sparse_collapse_flag, global_step=global_gen)

            writer.add_scalar("sparse_population_share", sparse_population_share, global_step=global_gen)

            writer.add_scalar("sparse_collapse_flag", sparse_collapse_flag, global_step=global_gen)

            if phase_is_t or phase_is_transition:

                writer.add_scalar("stats/t_symmetry", best_metrics["t_symmetry"], global_step=global_gen)

                writer.add_scalar("stats/t_trunk_continuity", best_metrics["t_trunk_continuity"], global_step=global_gen)

            writer.add_scalar(

                "debug/anti_extinction_triggers_mean", anti_ext_mean, global_step=global_gen

            )

            if sparse_rescue_enabled:

                sparse_collapse_active = bool(sparse_collapse_flag)

            if phase_is_t:

                writer.add_scalar(

                    "geometry/top_bar_alignment",

                    best_metrics.get("t_top_bar_alignment_score", 0.0),

                    global_step=global_gen,

                )

                writer.add_scalar(

                    "geometry/trunk_alignment",

                    best_metrics.get("t_trunk_alignment_score", 0.0),

                    global_step=global_gen,

                )

                writer.add_scalar(

                    "geometry/junction_score",

                    best_metrics.get("t_junction_score", 0.0),

                    global_step=global_gen,

                )

                writer.add_scalar(

                    "geometry/off_axis_penalty",

                    best_metrics.get("t_off_axis_penalty", 0.0),

                    global_step=global_gen,

                )

                writer.add_scalar(

                    "geometry/cleanliness_penalty",

                    best_metrics.get("t_cleanliness_penalty", 0.0),

                    global_step=global_gen,

                )

            mini_transfer_summary = None

            if (

                mini_transfer_enabled

                and phase_is_t

                and ((phase_step + 1) % mini_transfer_every == 0 or phase_is_refine)

            ):

                ranked_indices = sorted(

                    range(len(fitnesses)), key=lambda idx: fitnesses[idx], reverse=True

                )

                eligible_indices = [

                    idx

                    for idx in ranked_indices

                    if metrics[idx].get("target_coverage_ratio", 0.0) >= champion_min_coverage

                    and (

                        metrics[idx]["area"] / phase_target_area if phase_target_area > 0 else 0.0

                    )

                    >= champion_min_area_ratio

                    and metrics[idx]["iou"] >= champion_min_train_iou

                ]

                if not eligible_indices:

                    candidate_indices = []

                else:

                    candidate_indices = eligible_indices[:mini_transfer_top_k]

                mini_transfer_summary = run_mini_transfer_eval(candidate_indices)

                if mini_transfer_summary:

                    last_mini_transfer_summary = mini_transfer_summary

                    writer.add_scalar(

                        "mini_transfer/mean_final_iou",

                        mini_transfer_summary["mean_final_iou"],

                        global_step=global_gen,

                    )

                    writer.add_scalar(

                        "mini_transfer/mean_best_iou",

                        mini_transfer_summary["mean_best_iou"],

                        global_step=global_gen,

                    )

                    writer.add_scalar(

                        "mini_transfer/success_rate",

                        mini_transfer_summary["success_rate"],

                        global_step=global_gen,

                    )

                    writer.add_scalar(

                        "mini_transfer/variance",

                        mini_transfer_summary["variance"],

                        global_step=global_gen,

                    )

            baseline_score = compute_benchmark_dict_score(

                benchmark_results_map.get("baseline")

            )

            if mini_transfer_summary is not None and target_coverage_best >= champion_min_coverage:

                composite_score = (

                    0.4 * max(0.0, best_iou)

                    + 0.45 * mini_transfer_summary["mini_score"]

                    + 0.25 * baseline_score

                    - 0.15 * mini_transfer_summary["variance"]

                )

                current_champion_score = champion_score if champion_score is not None else float("-inf")

                if composite_score > current_champion_score + 1e-4:

                    torch.save(

                        population[mini_transfer_summary["candidate_index"]].state_dict(),

                        champion_path,

                    )

                    champion_score = composite_score

                    champion_mode = "composite"

                    champion_source = "composite"

                    champion_details = {

                        "score": champion_score,

                        "source": champion_source,

                        "generation": global_gen,

                        "train_iou": best_iou,

                        "mini_transfer": mini_transfer_summary,

                        "baseline_score": baseline_score,

                    }

            if champion_score is not None:

                writer.add_scalar("champion/score", champion_score, global_step=global_gen)

            plateau_limit_display = "inf" if phase_plateau_generations <= 0 else str(phase_plateau_generations)

            t_diag_str = ""

            if phase_is_t:

                t_diag_str = f" t_sym={t_symmetry_best:.3f} trunk={t_trunk_best:.3f}"

            clean_diag = best_metrics.get("late_clean_penalty", 0.0)

            print(

                f"[GEN {global_gen}] phase={phase_name} best_fit={best_fit:.4f} best_iou={best_iou:.4f} "

                f"fp={fp_rate_best:.4f} fn={fn_rate_best:.4f} stem_best={stem_count_best}{t_diag_str} "

                f"mean_fit={mean_fitness:.4f} p95_fit={p95_fitness:.4f} alive_best={alive_best} area_best={area_best:.1f}/"

                f"{phase_target_area:.1f} comps={num_components_best} mut_std={current_mutation_std:.4f} "

                f"clean_pen={clean_diag:.4f} plateau_gens={gens_since_improve}/{plateau_limit_display}"

            )

            generation_records.append(

                {

                    "generation": global_gen,

                    "phase": phase_name,

                    "best_fitness": best_fit,

                    "mean_fitness": mean_fitness,

                    "best_iou": best_iou,

                    "hold_score": best_metrics.get("hold_score", 0.0),

                    "stability_score": best_metrics.get("stability_score", 0.0),

                    "stabilized_success": best_metrics.get("stabilized_success", False),

                    "first_reach_step": best_metrics.get("first_reach_step"),

                    "hold_length": best_metrics.get("hold_length"),

                    "run_status": run_status,

                    "benchmark_target_best_stabilized": target_progress["best_stabilized"],

                    "benchmark_target_best_mean_final_iou": target_progress["best_mean_final_iou"],

                    "mini_transfer_mean_best_iou": (

                        mini_transfer_summary["mean_best_iou"] if mini_transfer_summary else None

                    ),

                    "mini_transfer_mean_final_iou": (

                        mini_transfer_summary["mean_final_iou"] if mini_transfer_summary else None

                    ),

                    "mini_transfer_success_rate": (

                        mini_transfer_summary["success_rate"] if mini_transfer_summary else None

                    ),

                    "mini_transfer_variance": (

                        mini_transfer_summary["variance"] if mini_transfer_summary else None

                    ),

                    "mini_transfer_score": (

                        mini_transfer_summary["mini_score"] if mini_transfer_summary else None

                    ),

                    "champion_score": champion_score,

                    "champion_source": champion_source,

                    "target_coverage_best": target_coverage_best,

                    "target_coverage_mean": coverage_mean,

                    "area_ratio_mean": area_ratio_mean,

                    "bar_coverage_best": best_metrics.get("t_top_bar_coverage", 0.0),

                    "trunk_coverage_best": best_metrics.get("t_trunk_coverage", 0.0),

                    "top_bar_score": best_metrics.get("t_top_bar_coverage", 0.0),

                    "trunk_coverage_score": best_metrics.get("t_trunk_coverage", 0.0),

                    "coverage_penalty": best_metrics.get("coverage_penalty", 0.0),

                    "sparse_collapse_penalty": best_metrics.get("sparse_collapse_penalty", 0.0),

                    "empty_collapse_penalty": best_metrics.get("empty_collapse_penalty", 0.0),

                    "collapse_gate": best_metrics.get("collapse_gate", 1.0),

                    "sparse_population_share": sparse_population_share,

                    "sparse_collapse_flag": sparse_collapse_flag,

                    "top_bar_alignment_score": best_metrics.get(

                        "t_top_bar_alignment_score", 0.0

                    ),

                    "trunk_alignment_score": best_metrics.get(

                        "t_trunk_alignment_score", 0.0

                    ),

                    "junction_score": best_metrics.get("t_junction_score", 0.0),

                    "off_axis_penalty": best_metrics.get("t_off_axis_penalty", 0.0),

                    "side_clutter_penalty": best_metrics.get("t_side_clutter_ratio", 0.0),

                    "bar_endpoint_clutter": best_metrics.get("t_bar_endpoint_clutter", 0.0),

                    "early_sparse_penalty": best_metrics.get("early_sparse_penalty", 0.0),

                    "late_t_penalty": best_metrics.get("late_t_penalty", 0.0),
                    "late_regression_flag": best_metrics.get("late_regression_flag", 0.0),
                    "late_structure_penalty": best_metrics.get("late_structure_penalty", 0.0),
                    "pseudo_t_penalty": best_metrics.get("pseudo_t_penalty", 0.0),

                }

            )

            prev_phase_best = phase_best_scores.get(phase_slug, float('-inf'))

            if best_metrics["fitness"] > prev_phase_best:

                phase_best_scores[phase_slug] = best_metrics["fitness"]

                phase_best_path = run_dir / f"best_{phase_slug}.pt"

                torch.save(best_model.state_dict(), phase_best_path)

                print(f"[GA] New best for phase '{phase_name}' saved to {phase_best_path}")

            hold_score = float(best_metrics.get("hold_score", 0.0))

            collapse_penalty = float(best_metrics.get("collapse_penalty", 0.0))

            stability_boost = 0.3 * max(0.0, hold_score - collapse_penalty)

            candidate_score = best_fit + (stability_boost if phase_is_t else 0.0)

            if phase_is_t and hold_score < 0.05:

                candidate_score -= 0.2

            if empty_candidate_blocked:

                candidate_score -= 0.3

            train_candidate_valid = True

            if phase_is_t:

                train_candidate_valid = (

                    not empty_candidate_blocked

                    and best_iou >= champion_min_train_iou

                    and target_coverage_best >= champion_min_coverage

                    and area_ratio_best >= champion_min_area_ratio

                )

            if train_candidate_valid and candidate_score > best_overall_score:

                best_overall_score = candidate_score

                torch.save(best_model.state_dict(), best_path)

                print(f"[GA] New best model saved to {best_path}")

            if checkpoint_every > 0 and global_gen % checkpoint_every == 0:

                ckpt_path = checkpoints_dir / f"gen_{global_gen:05d}.pt"

                torch.save(best_model.state_dict(), ckpt_path)

                print(f"[GA] Saved checkpoint: {ckpt_path}")

            plateau_ready = True

            if (phase_is_t or phase_is_transition) and (phase_step + 1) <= max(1, plateau_warmup_generations_t):

                plateau_ready = False

            active_plateau_generations = phase_plateau_generations

            if (

                active_plateau_generations > 0

                and plateau_ready

                and gens_since_improve >= active_plateau_generations

                and not plateau_triggered

            ):

                plateau_triggered = True

                plateau_msg = (

                    f"[GA] Plateau stop at GEN {global_gen} (phase={phase_name}): no IoU improvement for "

                    f"{active_plateau_generations} generations."

                )

                print(plateau_msg)

                writer.add_text("ga/stop_reason", plateau_msg, global_step=global_gen)

                writer.add_scalar("ga/plateau_trigger", 1, global_step=global_gen)

                break

            capture_this_gen = gif_enabled and (global_gen % gif_every == 0)

            should_visualize = (not args.no_viz) and (global_gen % viz_every == 0)

            if should_visualize or capture_this_gen:

                viz = None

                if should_visualize:

                    viz = build_visualizer(

                        cfg=viz_cfg,

                        width=int(world_cfg.get("width", 15)),

                        height=int(world_cfg.get("height", 15)),

                        cell_size=int(viz_cfg.get("cell_px", 24)),

                        target_mask=target_bool,

                        force_disable=False,

                    )

                gif_recorder: Optional[GifRecorder] = None

                if capture_this_gen:

                    gif_recorder = GifRecorder()

                    gif_recorder.start_capture(gif_max_frames)

                render_stride = max(1, int(viz_cfg.get("render_every_steps", 10)))

                def capture_grid_state(grid_tensor: torch.Tensor, step_idx: int) -> None:

                    if gif_recorder is None:

                        return

                    try:

                        frame = render_grid_to_rgb(grid_tensor, target_bool)

                        gif_recorder.capture_frame(frame)

                    except Exception as err:  # pragma: no cover - diagnostics only

                        print(f"[WARN] Failed to render GIF frame at step {step_idx}: {err}")

                print(f"[GEN {global_gen}] Visualizing best-of-generation (phase={phase_name})")

                viz_world = prepare_world(world_cfg, best_metrics["world_seed"])

                viz_model = clone_model(best_model)

                viz_model.to(device)

                simulate(

                    world=viz_world,

                    model=viz_model,

                    target_mask=target_bool,

                    steps=eval_steps,

                    writer=None,

                    viz=viz,

                    render_every=render_stride,

                    device=device,

                    anti_extinction_warmup_steps=warmup,

                    die_cap_schedule=die_cap_schedule,

                    post_warmup_die_cap_frac=die_cap_frac,

                    verbose=True,

                    frame_capture=capture_grid_state if gif_recorder is not None else None,

                    capture_every=gif_frame_stride,

                    late_cleanup_start_frac=late_cleanup_start_frac,

                    stability_cfg=train_stability_cfg if phase_is_t and train_stability_cfg.get("enabled") else None,

                )

                if gif_recorder is not None:

                    gifs_train_dir.mkdir(parents=True, exist_ok=True)

                    gif_path = gifs_train_dir / gif_name

                    gif_recorder.save_gif(gif_path, fps=int(viz_cfg.get("fps", 20)))

                    print(f"[GEN {global_gen}] Saved GIF: {gif_path}")

                    if gif_keep_history:

                        history_path = gifs_train_dir / f"gen_{global_gen:05d}_best.gif"

                        try:

                            shutil.copyfile(gif_path, history_path)

                            print(f"[GEN {global_gen}] Saved GIF history: {history_path}")

                        except Exception as err:

                            print(f"[WARN] Failed to copy GIF history: {err}")

                if viz is not None:

                    viz.close()

                viz_model.to("cpu")

            elites, elite_indices = select_elite(population, fitnesses, elite_frac)

            preserve_count = min(len(elite_indices), n_elites_unchanged)

            new_population: List[LittleLM] = [

                clone_model(population[idx]) for idx in elite_indices[:preserve_count]

            ]

            while len(new_population) < population_size:

                parent_a = elites[int(torch.randint(0, len(elites), (1,)).item())]

                parent_b = elites[int(torch.randint(0, len(elites), (1,)).item())]

                child = crossover(parent_a, parent_b)

                mutate(child, mutation_prob, current_mutation_std)

                child.eval()

                new_population.append(child)

            population = new_population[:population_size]

            if (
                late_t_guard_enabled
                and late_t_regression_counter > 0
                and late_t_guard_reinject
                and (phase_is_pure_t or phase_is_refine)
            ):
                champions = list(late_t_champion_pool)
                if not champions and late_t_champion_model is not None:
                    champions = [{"model": late_t_champion_model, "generation": late_t_champion_generation, "metrics": late_t_champion_metrics}]
                if champions:
                    slots_available = population_size - preserve_count
                    reinject_slots = max(1, int(population_size * late_t_guard_reinject_fraction))
                    reinject_slots = min(reinject_slots, late_t_guard_reinject_max_slots, slots_available, len(champions))
                    for slot in range(reinject_slots):
                        entry = champions[slot % len(champions)]
                        target_idx = population_size - 1 - slot
                        population[target_idx] = clone_model(entry["model"])
                        population[target_idx].eval()
            triggered_modes: List[Tuple[Dict[str, Any], str]] = []

            if benchmark_enabled and phase_is_t:

                for mode_state in benchmark_modes:

                    reason: Optional[str] = None

                    run_every = mode_state["run_every"]

                    if run_every > 0 and global_gen > 0 and global_gen % run_every == 0:

                        reason = f"every_{run_every}"

                    if mode_state["run_on_phase_end"] and (phase_step + 1) == generations:

                        reason = reason or "phase_end"

                    if plateau_triggered and mode_state["run_on_plateau"]:

                        reason = reason or "plateau"

                    if reason and mode_state.get("last_gen") != global_gen:

                        triggered_modes.append((mode_state, reason))

            execute_benchmarks(triggered_modes)

            if benchmark_passed:

                break

        phase_record["end_generation"] = global_gen

        phase_records.append(phase_record)

        if stop_training:

            break

        if plateau_triggered and plateau_stop_entire_run:

            should_halt = True

            if (

                benchmark_enabled

                and not benchmark_passed

                and target_mode_state is not None

                and target_mode_state["plateau_runs"] < target_mode_state["plateau_limit"]

            ):

                should_halt = False

                print(

                    "[GA] Plateau detected but benchmark progress not converged; continuing for more data."

                )

            if should_halt:

                print(f"[GA] Plateau stop configured to end run after phase {phase_name}.")

                break

    if benchmark_enabled and not benchmark_passed:

        final_modes = [

            (mode_state, "final")

            for mode_state in benchmark_modes

            if mode_state.get("run_on_final", True)

        ]

        execute_benchmarks(final_modes)

    update_status("completed", global_gen)

    writer.close()
    export_tensorboard_plots(run_dir, TENSORBOARD_PNG_TAGS)

    write_generations_csv(generations_csv_path, generation_records)

    run_summary_payload = {

        "run_name": cfg.get("logging", {}).get("run_name", "baseline"),

        "status": run_status,

        "status_timeline": status_timeline,

        "best_overall": best_overall_metrics,

        "phase_records": phase_records,

        "benchmark_history": benchmark_history,

        "artifacts": {

            "best_model": str((run_dir / "best.pt").resolve()),

            "champion_model": str(champion_path.resolve()) if champion_path.exists() else None,

            "latest_candidate_model": str(latest_candidate_path.resolve()) if latest_candidate_path.exists() else None,

            "late_t_champion_model": str(late_t_champion_path.resolve()) if late_t_champion_path.exists() else None,

            "checkpoints": str(checkpoints_dir.resolve()),

            "generations_csv": str(generations_csv_path.resolve()),

            "benchmark_dir": str(benchmark_dir.resolve()),

        },

        "latest_candidate": {

            "generation": latest_candidate_meta.get("generation"),

            "phase": latest_candidate_meta.get("phase"),

        },

        "late_t_champion": ({

            "generation": late_t_champion_generation,

            "metrics": late_t_champion_metrics,

        } if late_t_champion_metrics else None),

        "late_t_champion_pool": ([
            {
                "generation": entry.get("generation"),
                "metrics": entry.get("metrics"),
            }
            for entry in list(late_t_champion_pool)
        ] if late_t_champion_pool else None),

        "late_t_history": (list(late_t_metrics_history) if late_t_metrics_history else None),

        "stability": train_stability_cfg,

        "benchmark": {

            "enabled": benchmark_enabled,

            "modes": benchmark_results_map,

            "champion_mode": champion_mode,

            "champion_score": champion_score,

        },

        "mini_transfer": last_mini_transfer_summary,

        "champion": champion_details,

    }

    write_run_summary(run_summary_path, run_summary_payload)

    return best_path

