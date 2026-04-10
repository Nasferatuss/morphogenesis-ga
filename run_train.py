"""Morphogenesis-GA CLI entry point.

Thin orchestrator that parses CLI arguments, loads a YAML config, builds the
default target mask, and dispatches to one of three pipelines:

* ``--play-best``       → ``agents.eval_agent.pipeline.play_best``
* ``--train-ga``        → ``agents.train_agent.pipeline.train_ga``
* (neither flag)        → ``core.services.simulator.run_single_simulation``

All heavy lifting (GA training, single-shot simulation, checkpoint replay,
metrics logging, visualisation) lives in ``core/`` and ``agents/`` modules.
This file intentionally stays small so it can be replaced by a proper
``main.py`` package entry point when the v1.0 packaging work lands.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agents.eval_agent.pipeline import play_best
from agents.train_agent.pipeline import train_ga
from core.services.config_loader import load_config
from core.services.simulator import run_single_simulation
from src.targets import make_target
from src.utils import select_device, set_seed


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
    target_area = float(target_mask.to(dtype=bool).sum().item())
    print(
        f"[INFO] target.name={target_name} target_area={target_area:.1f} "
        f"grid={width}x{height}"
    )

    if args.play_best:
        play_best(cfg, args, device, target_mask, seed)
        return

    if args.train_ga:
        if not cfg.get("ga", {}).get("enabled", False):
            print("[WARN] GA disabled in config but --train-ga supplied. Continuing anyway.")
        best_path = train_ga(
            cfg, args, device, target_mask, seed, target_name, target_area
        )
        print(f"[GA] Training complete. Best model saved at {best_path}")
        return

    run_single_simulation(cfg, args, device, target_mask, seed)


if __name__ == "__main__":
    main()
