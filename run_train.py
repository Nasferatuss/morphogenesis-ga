import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:

    import numpy as np

except ImportError:  # pragma: no cover

    np = None

import torch
from torch.utils.tensorboard import SummaryWriter

from agents.eval_agent.pipeline import play_best as _eval_play_best
from agents.train_agent.pipeline import train_ga as _train_agent_pipeline
from core.memory.metrics_logger import (
    export_tensorboard_plots as _core_export_tb_plots,
)
from core.memory.metrics_logger import (
    write_generations_csv as _core_write_csv,
)
from core.memory.metrics_logger import (
    write_run_summary as _core_write_summary,
)
from core.services.cleanup import compute_late_cleanup_ratio as _core_cleanup_ratio
from core.services.cleanup import resolve_cleanup_ratio as _core_resolve_cleanup
from core.services.config_loader import load_config as _core_load_config
from core.services.factory import build_stability_config as _core_build_stability
from core.services.factory import prepare_model as _core_prepare_model
from core.services.factory import prepare_world as _core_prepare_world
from core.services.stats import summarize as _core_summarize
from core.services.visualizer import build_visualizer as _core_build_visualizer
from core.services.writer import build_writer as _core_build_writer
from src.fitness import compute_fitness
from src.model import LittleLM
from src.simulate import simulate
from src.targets import make_target
from src.utils import select_device, set_seed
from src.world import World

TENSORBOARD_PNG_TAGS: Sequence[str] = (
    "fitness/best",
    "fitness/mean",
    "iou/best",
    "iou/mean",
    "stats/alive_best",
    "stats/area_mean",
    "coverage/best",
    "coverage/mean",
)

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description=(

            "Morphogenesis world simulator with GA training and curriculum support.\n\n"

            "Examples:\n"

            "  python run_train.py --config configs/baseline_T.yaml --train-ga\n"

            "  python run_train.py --config configs/curriculum.yaml --train-ga\n"

            "  python run_train.py --config configs/curriculum.yaml --train-ga --no-viz"

        ),

        formatter_class=argparse.RawDescriptionHelpFormatter,

    )

    parser.add_argument("--config", required=True, help="Path to YAML config")

    parser.add_argument(

        "--no-viz",

        action="store_true",

        help="Disable pygame visualization regardless of config",

    )

    parser.add_argument(

        "--steps",

        type=int,

        default=None,

        help="Override number of simulation steps from config",

    )

    parser.add_argument(

        "--train-ga",

        action="store_true",

        help="Run genetic algorithm training instead of a single simulation",

    )

    parser.add_argument(

        "--play-best",

        type=str,

        default=None,

        help="Path to best.pt to replay with visualization",

    )

    return parser.parse_args()

def load_config(path: str) -> Dict[str, Any]:
    return _core_load_config(path)

def build_writer(run_name: str) -> Tuple[SummaryWriter, Path, str]:
    return _core_build_writer(run_name)


def build_visualizer(
    cfg: Dict[str, Any],
    width: int,
    height: int,
    cell_size: int,
    target_mask: torch.Tensor,
    force_disable: bool,
):
    return _core_build_visualizer(cfg, width, height, cell_size, target_mask, force_disable)


def export_tensorboard_plots(run_dir: Path, tags: Sequence[str]) -> None:
    _core_export_tb_plots(run_dir, tags)


def write_generations_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    _core_write_csv(path, records)


def write_run_summary(path: Path, payload: Dict[str, Any]) -> None:
    _core_write_summary(path, payload)

def compute_late_cleanup_ratio(
    steps_done: int, planned_steps: int, cleanup_start_frac: float = 0.8
) -> float:
    return _core_cleanup_ratio(steps_done, planned_steps, cleanup_start_frac)


def resolve_cleanup_ratio(
    sim_result: Dict[str, Any], planned_steps: int, cleanup_start_frac: float
) -> float:
    return _core_resolve_cleanup(sim_result, planned_steps, cleanup_start_frac)


def prepare_world(world_cfg: Dict[str, Any], seed: int) -> World:
    return _core_prepare_world(world_cfg, seed)


def prepare_model(model_cfg: Dict[str, Any]) -> LittleLM:
    return _core_prepare_model(model_cfg)


def build_stability_config(benchmark_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _core_build_stability(benchmark_cfg)

def run_single_simulation(

    cfg: Dict[str, Any],

    args: argparse.Namespace,

    device: torch.device,

    target_mask: torch.Tensor,

    seed: int,

) -> None:

    world_cfg = cfg.get("world", {})

    model_cfg = cfg.get("model", {})

    viz_cfg = cfg.get("viz", {})

    simulate_cfg = cfg.get("simulate", {})

    fitness_cfg = cfg.get("fitness", {})

    benchmark_cfg = cfg.get("benchmark", {})

    target_cfg = cfg.get("target", {})

    target_label = str(target_cfg.get("name", "T")).lower()

    empty_collapse_cfg = fitness_cfg.get("empty_collapse", {})

    empty_collapse_area_floor = float(empty_collapse_cfg.get("area_floor", 0.0))

    empty_collapse_coverage_floor = float(empty_collapse_cfg.get("coverage_floor", 0.0))

    empty_collapse_penalty_weight = float(empty_collapse_cfg.get("penalty_weight", 0.0))

    empty_collapse_suppress_scale = float(empty_collapse_cfg.get("suppress_scale", 0.0))

    stability_source = (

        benchmark_cfg.get("stability")

        or benchmark_cfg.get("baseline")

        or benchmark_cfg.get("target")

        or benchmark_cfg

    )

    stability_cfg = build_stability_config(stability_source)

    t_phase_weights = fitness_cfg.get("t_phase_weights") if target_label == "t" else None

    steps_cfg = int(world_cfg.get("steps", 0))

    steps = steps_cfg if args.steps is None else max(0, args.steps)

    if args.steps is not None:

        print(f"[INFO] Overriding steps from config ({steps_cfg}) -> {steps}")

    alpha_fp = float(fitness_cfg.get("alpha_fp", 1.0))

    beta_fn = float(fitness_cfg.get("beta_fn", 0.5))

    gamma_area = float(fitness_cfg.get("gamma_area", 0.15))

    stem_penalty_scale = float(fitness_cfg.get("stem_penalty_scale", 0.05))

    stem_cleanup_multiplier = float(fitness_cfg.get("stem_cleanup_multiplier", 1.0))

    stem_corridor_start_scale = float(fitness_cfg.get("stem_corridor_start_scale", 2.0))

    stem_corridor_end_scale = float(fitness_cfg.get("stem_corridor_end_scale", 1.1))

    t_symmetry_weight = float(fitness_cfg.get("t_symmetry_weight", 0.0))

    t_trunk_weight = float(fitness_cfg.get("t_trunk_weight", 0.0))

    coverage_cfg = fitness_cfg.get("coverage", {})

    coverage_target_floor_base = float(coverage_cfg.get("target_floor", 0.55))

    coverage_reward_weight_base = float(coverage_cfg.get("reward_weight", 0.6))

    coverage_penalty_weight_base = float(coverage_cfg.get("penalty_weight", 1.0))

    sparse_area_floor_base = float(coverage_cfg.get("area_ratio_floor", 0.6))

    sparse_penalty_weight_base = float(coverage_cfg.get("area_penalty_weight", 0.8))

    coverage_target_floor = coverage_target_floor_base

    coverage_reward_weight = coverage_reward_weight_base

    coverage_penalty_weight = coverage_penalty_weight_base

    sparse_area_floor = sparse_area_floor_base

    sparse_penalty_weight = sparse_penalty_weight_base

    target_area = float(target_mask.to(dtype=torch.bool).sum().item())

    world = prepare_world(world_cfg, seed)

    model = prepare_model(model_cfg).to(device)

    viz = build_visualizer(

        cfg=viz_cfg,

        width=world.width,

        height=world.height,

        cell_size=int(viz_cfg.get("cell_px", 24)),

        target_mask=target_mask,

        force_disable=args.no_viz,

    )

    writer, _, _ = build_writer(cfg.get("logging", {}).get("run_name", "baseline"))

    writer.add_scalar("stats/target_area", target_area, 0)

    render_every = max(1, int(viz_cfg.get("render_every_steps", 10)))

    warmup = int(simulate_cfg.get("anti_extinction_warmup_steps", 0))

    die_cap_schedule = simulate_cfg.get("die_cap_schedule")

    die_cap_frac = simulate_cfg.get("post_warmup_die_cap_frac", None)

    late_cleanup_start_frac = max(0.0, min(1.0, float(simulate_cfg.get("late_cleanup_start_frac", 0.8))))

    sim_result = simulate(

        world=world,

        model=model,

        target_mask=target_mask,

        steps=steps,

        writer=writer,

        viz=viz,

        render_every=render_every,

        device=device,

        anti_extinction_warmup_steps=warmup,

        die_cap_schedule=die_cap_schedule,

        post_warmup_die_cap_frac=die_cap_frac,

        verbose=True,

        late_cleanup_start_frac=late_cleanup_start_frac,

        stability_cfg=stability_cfg if stability_cfg.get("enabled") else None,

    )

    cleanup_ratio = resolve_cleanup_ratio(sim_result, steps, late_cleanup_start_frac)

    fitness_metrics = compute_fitness(

        world.grid,

        target_mask,

        alpha_fp=alpha_fp,

        beta_fn=beta_fn,

        alive_end=sim_result["alive_end"],

        target_area=target_area,

        gamma_area=gamma_area,

        stem_penalty_scale=stem_penalty_scale,

        stem_cleanup_multiplier=stem_cleanup_multiplier,

        stem_cleanup_ratio=cleanup_ratio,

        stem_corridor_start_scale=stem_corridor_start_scale,

        stem_corridor_end_scale=stem_corridor_end_scale,

        t_symmetry_weight=t_symmetry_weight,

        t_trunk_weight=t_trunk_weight,

        stability_metrics=sim_result if stability_cfg.get("enabled") else None,

        t_benchmark_weights=t_phase_weights,

        expect_t_shape=target_label == "t",

        coverage_weight=coverage_reward_weight,

        coverage_floor=coverage_target_floor,

        coverage_penalty_weight=coverage_penalty_weight,

        sparse_area_floor=sparse_area_floor,

        sparse_penalty_weight=sparse_penalty_weight,

        t_phase_gate=1.0,

        t_structure_gate=1.0,

        t_cleanliness_gate=1.0,

        collapse_area_floor=empty_collapse_area_floor if target_label == "t" else 0.0,

        collapse_coverage_floor=empty_collapse_coverage_floor if target_label == "t" else 0.0,

        collapse_penalty_weight=empty_collapse_penalty_weight if target_label == "t" else 0.0,

        collapse_bonus_suppression=empty_collapse_suppress_scale if target_label == "t" else 0.0,

    )

    iou = fitness_metrics["iou"]

    base_score = fitness_metrics["base_score"]

    fp_rate = fitness_metrics["fp"]

    fn_rate = fitness_metrics["fn"]

    alive_bonus = fitness_metrics["alive_bonus"]

    extinction_penalty = fitness_metrics["extinction_penalty"]

    area_ab = fitness_metrics["area"]

    print(

        "[RESULT] IoU={:.4f} | base={:.4f} | fp={:.4f} | fn={:.4f} | "

        "alive_bonus={:.4f} | ext_penalty={:.4f} | area_ab={:.1f} | target_area={:.1f}".format(

            iou,

            base_score,

            fp_rate,

            fn_rate,

            alive_bonus,

            extinction_penalty,

            area_ab,

            target_area,

        )

    )

    print(

        f"[RESULT] Divisions={sim_result['total_divisions']} | Deaths={sim_result['total_deaths']} | "

        f"Alive end={sim_result['alive_end']}"

    )

    if viz is not None:

        viz.render(world.grid, target_mask, step=sim_result["steps"], iou=sim_result["iou"])

        viz.close()

    writer.close()

def summarize(values: List[float], percentile: float) -> Tuple[float, float]:
    return _core_summarize(values, percentile)

def train_ga(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    default_target_mask: torch.Tensor,
    seed: int,
    default_target_name: str,
    default_target_area: float,
) -> Path:
    return _train_agent_pipeline(
        cfg, args, device, default_target_mask, seed,
        default_target_name, default_target_area,
    )

def play_best(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    target_mask: torch.Tensor,
    seed: int,
) -> None:
    _eval_play_best(cfg, args, device, target_mask, seed)

def main() -> None:

    args = parse_args()

    config_path = Path(args.config)

    if not config_path.exists():

        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(str(config_path))

    seed = int(cfg.get("seed", 42))

    set_seed(seed)

    print(f"[INFO] Loaded config from {config_path}")

    device_pref = cfg.get("device", {}).get("prefer", "cuda")

    device = select_device(device_pref)

    print(f"[INFO] Using device: {device}")

    world_cfg = cfg.get("world", {})

    width = int(world_cfg.get("width", 15))

    height = int(world_cfg.get("height", 15))

    target_cfg = cfg.get("target", {})

    target_name = str(target_cfg.get("name", "T"))

    target_mask = make_target(target_name, height, width)

    target_area = float(target_mask.to(dtype=torch.bool).sum().item())

    print(f"[INFO] target.name={target_name} target_area={target_area:.1f} grid={width}x{height}")

    if args.play_best:

        play_best(cfg, args, device, target_mask, seed)

        return

    if args.train_ga:

        if not cfg.get("ga", {}).get("enabled", False):

            print("[WARN] GA disabled in config but --train-ga supplied. Continuing anyway.")

        best_path = train_ga(cfg, args, device, target_mask, seed, target_name, target_area)

        print(f"[GA] Training complete. Best model saved at {best_path}")

        return

    run_single_simulation(cfg, args, device, target_mask, seed)

if __name__ == "__main__":

    main()

