from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Sequence

import torch

from .fitness import compute_fitness
from .ga import clone_model
from .model import LittleLM
from .simulate import simulate
from .world import World


@dataclass
class BenchmarkConfig:
    mode: str
    name: str
    fixed_seeds: Sequence[int]
    reach_threshold_iou: float
    stabilize_threshold_iou: float
    hold_window_steps: int
    min_reach_rate: float
    min_stabilized_rate: float
    min_mean_final_iou: float
    eval_steps: int
    plateau_min_delta: float = 1e-3


@dataclass
class BenchmarkSeedResult:
    seed: int
    best_iou: float
    final_iou: float
    first_reach_step: Optional[int]
    hold_length: float
    hold_window: float
    stabilized_success: bool
    alive_final: int
    area_final: float
    components_final: float
    t_sym_final: float
    trunk_final: float
    fp_final: float
    fn_final: float
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def reached(self) -> bool:
        return self.first_reach_step is not None and self.first_reach_step >= 0


@dataclass
class BenchmarkSummary:
    success_rate_reach: float
    success_rate_stabilized: float
    mean_first_reach_step: float
    median_first_reach_step: float
    mean_best_iou: float
    mean_final_iou: float
    std_final_iou: float
    mean_hold_length: float
    reproducibility_score: float
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRun:
    config: BenchmarkConfig
    summary: BenchmarkSummary
    per_seed: List[BenchmarkSeedResult]
    output_dir: Path

    @property
    def summary_path(self) -> Path:
        return self.output_dir / f"{self.config.name}_summary.json"

    @property
    def per_seed_path(self) -> Path:
        return self.output_dir / f"{self.config.name}_per_seed.csv"


def _prepare_world(world_cfg: Dict[str, Any], seed: int) -> World:
    world = World(
        width=int(world_cfg.get("width", 15)),
        height=int(world_cfg.get("height", 15)),
        init_cells=int(world_cfg.get("init_cells", 50)),
        seed=seed,
    )
    world.reset()
    return world


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def _safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def _reproducibility_score(
    stabilized_rate: float,
    std_final_iou: float,
    std_first_reach_norm: float,
) -> float:
    penalty = 0.5 * std_final_iou + 0.4 * std_first_reach_norm
    return max(0.0, min(1.0, stabilized_rate - penalty))


def summarize_benchmark(
    config: BenchmarkConfig,
    per_seed: Sequence[BenchmarkSeedResult],
) -> BenchmarkSummary:
    reach_flags = [1.0 if result.reached else 0.0 for result in per_seed]
    stabilized_flags = [1.0 if result.stabilized_success else 0.0 for result in per_seed]
    success_rate_reach = _safe_mean(reach_flags)
    success_rate_stabilized = _safe_mean(stabilized_flags)
    first_reach_steps = [float(result.first_reach_step) for result in per_seed if result.reached]
    best_ious = [result.best_iou for result in per_seed]
    final_ious = [result.final_iou for result in per_seed]
    hold_lengths = [result.hold_length for result in per_seed]
    median_first_reach = _safe_median(first_reach_steps)
    mean_first_reach = _safe_mean(first_reach_steps)
    std_final_iou = _safe_std(final_ious)
    reach_norm_values = [step / max(1.0, config.eval_steps) for step in first_reach_steps]
    std_first_reach_norm = _safe_std(reach_norm_values)
    reproducibility = _reproducibility_score(
        success_rate_stabilized,
        std_final_iou,
        std_first_reach_norm,
    )

    verdict = "benchmark_passed"
    if success_rate_reach < config.min_reach_rate:
        verdict = "benchmark_failed"
    if success_rate_stabilized < config.min_stabilized_rate:
        verdict = "benchmark_failed"
    if _safe_mean(final_ious) < config.min_mean_final_iou:
        verdict = "benchmark_failed"

    return BenchmarkSummary(
        success_rate_reach=success_rate_reach,
        success_rate_stabilized=success_rate_stabilized,
        mean_first_reach_step=mean_first_reach,
        median_first_reach_step=median_first_reach,
        mean_best_iou=_safe_mean(best_ious),
        mean_final_iou=_safe_mean(final_ious),
        std_final_iou=std_final_iou,
        mean_hold_length=_safe_mean(hold_lengths),
        reproducibility_score=reproducibility,
        verdict=verdict,
    )


def write_benchmark_artifacts(run: BenchmarkRun) -> None:
    output_dir = run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir = output_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "config": {
            "name": run.config.name,
            "reach_threshold_iou": run.config.reach_threshold_iou,
            "stabilize_threshold_iou": run.config.stabilize_threshold_iou,
            "hold_window_steps": run.config.hold_window_steps,
            "min_reach_rate": run.config.min_reach_rate,
            "min_stabilized_rate": run.config.min_stabilized_rate,
            "min_mean_final_iou": run.config.min_mean_final_iou,
            "eval_steps": run.config.eval_steps,
            "fixed_seeds": list(run.config.fixed_seeds),
        },
        "summary": run.summary.to_dict(),
    }
    run.summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    csv_path = run.per_seed_path
    fieldnames = [
        "seed",
        "verdict",
        "best_iou",
        "final_iou",
        "first_reach_step",
        "reached",
        "stabilized_success",
        "hold_length",
        "hold_window",
        "alive_final",
        "area_final",
        "components_final",
        "t_sym_final",
        "trunk_final",
        "fp_final",
        "fn_final",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed_result in run.per_seed:
            writer.writerow(
                {
                    "seed": seed_result.seed,
                    "verdict": seed_result.verdict,
                    "best_iou": seed_result.best_iou,
                    "final_iou": seed_result.final_iou,
                    "first_reach_step": seed_result.first_reach_step,
                    "reached": seed_result.reached,
                    "stabilized_success": seed_result.stabilized_success,
                    "hold_length": seed_result.hold_length,
                    "hold_window": seed_result.hold_window,
                    "alive_final": seed_result.alive_final,
                    "area_final": seed_result.area_final,
                    "components_final": seed_result.components_final,
                    "t_sym_final": seed_result.t_sym_final,
                    "trunk_final": seed_result.trunk_final,
                    "fp_final": seed_result.fp_final,
                    "fn_final": seed_result.fn_final,
                }
            )

    for seed_result in run.per_seed:
        seed_path = seeds_dir / f"seed_{seed_result.seed:04d}.json"
        seed_path.write_text(json.dumps(seed_result.to_dict(), indent=2), encoding="utf-8")


def evaluate_t_benchmark(
    model: LittleLM,
    *,
    benchmark_cfg: BenchmarkConfig,
    world_cfg: Dict[str, Any],
    simulate_cfg: Dict[str, Any],
    fitness_cfg: Dict[str, Any],
    target_mask: torch.Tensor,
    target_area: float,
    device: torch.device,
    output_dir: Path,
    stability_weights: Optional[Dict[str, float]] = None,
) -> BenchmarkRun:
    per_seed_results: List[BenchmarkSeedResult] = []
    warmup = int(simulate_cfg.get("anti_extinction_warmup_steps", 0))
    die_cap_schedule = simulate_cfg.get("die_cap_schedule")
    die_cap_frac = simulate_cfg.get("post_warmup_die_cap_frac", None)
    late_cleanup_start_frac = float(simulate_cfg.get("late_cleanup_start_frac", 0.8))

    alpha_fp = float(fitness_cfg.get("alpha_fp", 1.0))
    beta_fn = float(fitness_cfg.get("beta_fn", 0.5))
    gamma_area = float(fitness_cfg.get("gamma_area", 0.15))
    stem_penalty_scale = float(fitness_cfg.get("stem_penalty_scale", 0.05))
    stem_cleanup_multiplier = float(fitness_cfg.get("stem_cleanup_multiplier", 1.0))
    stem_corridor_start_scale = float(fitness_cfg.get("stem_corridor_start_scale", 2.0))
    stem_corridor_end_scale = float(fitness_cfg.get("stem_corridor_end_scale", 1.1))
    t_symmetry_weight = float(fitness_cfg.get("t_symmetry_weight", 0.0))
    t_trunk_weight = float(fitness_cfg.get("t_trunk_weight", 0.0))
    clean_component_weight = float(fitness_cfg.get("late_t", {}).get("clean_component_weight", 0.0))
    clean_fp_weight = float(fitness_cfg.get("late_t", {}).get("clean_fp_weight", 0.0))
    empty_collapse_cfg = fitness_cfg.get("empty_collapse", {})
    collapse_area_floor = float(empty_collapse_cfg.get("area_floor", 0.0))
    collapse_coverage_floor = float(empty_collapse_cfg.get("coverage_floor", 0.0))
    collapse_penalty_weight = float(empty_collapse_cfg.get("penalty_weight", 0.0))
    collapse_suppress_scale = float(empty_collapse_cfg.get("suppress_scale", 0.0))

    stability_cfg = {
        "enabled": True,
        "reach_threshold_iou": benchmark_cfg.reach_threshold_iou,
        "stabilize_threshold_iou": benchmark_cfg.stabilize_threshold_iou,
        "hold_window_steps": benchmark_cfg.hold_window_steps,
    }

    for seed in benchmark_cfg.fixed_seeds:
        world = _prepare_world(world_cfg, int(seed))
        eval_model = clone_model(model)
        eval_model.to(device)
        sim_result = simulate(
            world=world,
            model=eval_model,
            target_mask=target_mask,
            steps=benchmark_cfg.eval_steps,
            writer=None,
            viz=None,
            render_every=benchmark_cfg.eval_steps,
            device=device,
            anti_extinction_warmup_steps=warmup,
            die_cap_schedule=die_cap_schedule,
            post_warmup_die_cap_frac=die_cap_frac,
            verbose=False,
            late_cleanup_start_frac=late_cleanup_start_frac,
            stability_cfg=stability_cfg,
        )
        eval_model.to("cpu")
        metrics = compute_fitness(
            world.grid,
            target_mask,
            alpha_fp=alpha_fp,
            beta_fn=beta_fn,
            alive_end=sim_result["alive_end"],
            target_area=target_area,
            gamma_area=gamma_area,
            stem_penalty_scale=stem_penalty_scale,
            stem_cleanup_multiplier=stem_cleanup_multiplier,
            stem_cleanup_ratio=sim_result.get("late_cleanup_ratio", 0.0),
            stem_corridor_start_scale=stem_corridor_start_scale,
            stem_corridor_end_scale=stem_corridor_end_scale,
            t_symmetry_weight=t_symmetry_weight,
            t_trunk_weight=t_trunk_weight,
            clean_component_weight=clean_component_weight,
            clean_fp_weight=clean_fp_weight,
            stability_metrics=sim_result,
            t_benchmark_weights=stability_weights,
            expect_t_shape=True,
            t_phase_gate=1.0,
            t_structure_gate=1.0,
            t_cleanliness_gate=1.0,
            collapse_area_floor=collapse_area_floor,
            collapse_coverage_floor=collapse_coverage_floor,
            collapse_penalty_weight=collapse_penalty_weight,
            collapse_bonus_suppression=collapse_suppress_scale,
        )
        first_reach_step = metrics.get("first_reach_step")
        if isinstance(first_reach_step, float) and first_reach_step.is_integer():
            first_reach_step = int(first_reach_step)
        verdict = "failed_to_reach"
        if first_reach_step is not None and first_reach_step >= 0:
            verdict = "reached"
            if metrics.get("stabilized_success"):
                verdict = "stabilized"
            else:
                verdict = "reached_but_not_stable"
        per_seed_results.append(
            BenchmarkSeedResult(
                seed=int(seed),
                best_iou=float(sim_result.get("best_iou", metrics["iou"])),
                final_iou=float(metrics["iou"]),
                first_reach_step=first_reach_step,
                hold_length=float(sim_result.get("hold_length_achieved", 0.0)),
                hold_window=float(sim_result.get("hold_window_steps", benchmark_cfg.hold_window_steps)),
                stabilized_success=bool(metrics.get("stabilized_success", False)),
                alive_final=int(sim_result.get("alive_end", 0)),
                area_final=float(metrics["area"]),
                components_final=float(metrics["num_components"]),
                t_sym_final=float(metrics.get("t_symmetry", 0.0)),
                trunk_final=float(metrics.get("t_trunk_continuity", 0.0)),
                fp_final=float(metrics.get("fp", 0.0)),
                fn_final=float(metrics.get("fn", 0.0)),
                verdict=verdict,
            )
        )

    summary = summarize_benchmark(benchmark_cfg, per_seed_results)
    run = BenchmarkRun(
        config=benchmark_cfg,
        summary=summary,
        per_seed=per_seed_results,
        output_dir=output_dir,
    )
    write_benchmark_artifacts(run)
    return run
