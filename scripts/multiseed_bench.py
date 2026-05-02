"""Multi-seed benchmark utility for morphogenesis-ga checkpoints.

Usage:
    python scripts/multiseed_bench.py <checkpoint_path> [--seeds N] [--steps N] [--target T]

Example:
    python scripts/multiseed_bench.py runs/exp_t_stabilize_*/best.pt

Loads a LittleLM checkpoint, runs the simulation across N world seeds (default 10),
and reports mean / min / max best-IoU plus reach-rate thresholds. Auto-detects
whether the model was trained with positional awareness.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch

# Make repo root importable when script is run directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import LittleLM  # noqa: E402
from src.simulate import simulate  # noqa: E402
from src.targets import make_target  # noqa: E402
from src.utils import set_seed  # noqa: E402
from src.world import World  # noqa: E402

DEFAULT_DIE_CAP_SCHEDULE = [
    {"until_step": 60, "cap_frac": 0.20},
    {"until_step": 130, "cap_frac": 0.40},
    {"until_step": 999999, "cap_frac": 0.35},
]


def load_model_auto(checkpoint_path: Path, embed_dim: int, num_heads: int) -> LittleLM:
    """Load a LittleLM checkpoint, auto-detecting spatial awareness."""
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    has_spatial = any(k.startswith("spatial_proj") for k in ckpt)
    model = LittleLM(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_position=has_spatial,
    )
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def run_benchmark(
    model: LittleLM,
    target_name: str,
    width: int,
    height: int,
    init_cells: int,
    n_seeds: int,
    steps: int,
    warmup: int,
    late_cleanup_start_frac: float,
) -> List[Tuple[int, float, float]]:
    """Run simulation across N seeds, return (seed, best_iou, final_iou) tuples."""
    target = make_target(target_name, height, width)
    results = []
    for seed in range(n_seeds):
        set_seed(seed)
        world = World(width=width, height=height, init_cells=init_cells, seed=seed)
        world.reset()
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=steps,
            writer=None,
            viz=None,
            render_every=999,
            device=torch.device("cpu"),
            anti_extinction_warmup_steps=warmup,
            die_cap_schedule=DEFAULT_DIE_CAP_SCHEDULE,
            late_cleanup_start_frac=late_cleanup_start_frac,
            verbose=False,
        )
        results.append((seed, float(result["best_iou"]), float(result["iou"])))
    return results


def print_report(
    results: List[Tuple[int, float, float]],
    checkpoint_path: Path,
    baseline_iou: float,
) -> None:
    """Print a one-screen benchmark report with delta vs locked baseline."""
    bests = [r[1] for r in results]
    finals = [r[2] for r in results]
    n = len(results)
    best_mean = sum(bests) / n

    print(f"\n=== Multi-seed benchmark: {checkpoint_path.name} ===")
    print(f"Path: {checkpoint_path}")
    print(f"Seeds: {n}")
    print()
    print(f"{'seed':<6} {'best_iou':<12} {'final_iou':<12}")
    print("-" * 32)
    for seed, best, final in results:
        print(f"{seed:<6} {best:<12.4f} {final:<12.4f}")
    print()
    print(f"Best IoU   : mean={best_mean:.4f}  min={min(bests):.4f}  max={max(bests):.4f}")
    print(f"Final IoU  : mean={sum(finals)/n:.4f}  min={min(finals):.4f}  max={max(finals):.4f}")
    print()
    for thr in (0.20, 0.30, 0.40, 0.50):
        reach = sum(1 for v in bests if v >= thr) / n
        bar = "#" * int(reach * 20)
        print(f"reach>={thr:.2f} : {int(reach*100):>3}%  {bar}")
    print()
    delta = best_mean - baseline_iou
    pct = 100.0 * delta / baseline_iou if baseline_iou > 0 else 0.0
    sign = "+" if delta >= 0 else ""
    if delta >= 0.02:
        verdict = "BEATS baseline (>= +0.02 delta)"
    elif delta >= 0:
        verdict = "within noise (below +0.02 delta)"
    else:
        verdict = "UNDER baseline"
    print(
        f"vs baseline {baseline_iou:.4f}: {sign}{delta:.4f} ({sign}{pct:.1f}%) — {verdict}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to best.pt")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps (default: 200)")
    parser.add_argument("--target", type=str, default="T", help="Target shape name (default: T)")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--init-cells", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--late-cleanup", type=float, default=0.75)
    parser.add_argument(
        "--baseline-iou",
        type=float,
        default=0.329,
        help=(
            "Reference MS-10 IoU to compare against (default 0.329, the "
            "post-RNG-fix locked baseline from docs/DECISION_2026-04-19.md). "
            "Report prints delta and verdict."
        ),
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    model = load_model_auto(args.checkpoint, args.embed_dim, args.num_heads)
    results = run_benchmark(
        model=model,
        target_name=args.target,
        width=args.width,
        height=args.height,
        init_cells=args.init_cells,
        n_seeds=args.seeds,
        steps=args.steps,
        warmup=args.warmup,
        late_cleanup_start_frac=args.late_cleanup,
    )
    print_report(results, args.checkpoint, args.baseline_iou)
    return 0


if __name__ == "__main__":
    sys.exit(main())
