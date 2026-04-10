"""Dump ASCII visualization of a trained model's final grid on one or more seeds.

Usage:
    python scripts/dump_grid.py <checkpoint> [--seeds 0,1,2] [--steps 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fitness import compute_iou  # noqa: E402
from src.model import LittleLM  # noqa: E402
from src.simulate import simulate  # noqa: E402
from src.targets import make_target  # noqa: E402
from src.utils import set_seed  # noqa: E402
from src.world import CELL_A, CELL_B, CELL_EMPTY, CELL_STEM, World  # noqa: E402

DIE_CAP_SCHEDULE = [
    {"until_step": 60, "cap_frac": 0.20},
    {"until_step": 130, "cap_frac": 0.40},
    {"until_step": 999999, "cap_frac": 0.35},
]


def load_model(ckpt_path: Path) -> LittleLM:
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    has_spatial = any(k.startswith("spatial_proj") for k in ckpt)
    # Detect embed_dim from token_embed.weight
    embed_dim = int(ckpt["token_embed.weight"].shape[1]) if "token_embed.weight" in ckpt else 32
    model = LittleLM(embed_dim=embed_dim, num_heads=4, use_position=has_spatial)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def render(grid: torch.Tensor, target: torch.Tensor) -> str:
    """ASCII rendering.

    Legend:
      .  empty cell (not in target)
      x  empty cell (IN target — missed)
      S  stem cell
      A  CELL_A in target (correct)
      a  CELL_A NOT in target (overflow)
      B  CELL_B in target (correct)
      b  CELL_B NOT in target (overflow)
    """
    h, w = grid.shape
    lines = []
    for y in range(h):
        row = []
        for x in range(w):
            val = grid[y, x].item()
            in_target = bool(target[y, x].item())
            if val == CELL_EMPTY:
                row.append("x" if in_target else ".")
            elif val == CELL_STEM:
                row.append("S")
            elif val == CELL_A:
                row.append("A" if in_target else "a")
            elif val == CELL_B:
                row.append("B" if in_target else "b")
            else:
                row.append("?")
        lines.append(" ".join(row))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated list of world seeds (default 0,1,2)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--target", type=str, default="T")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--init-cells", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    model = load_model(args.checkpoint)
    target = make_target(args.target, args.height, args.width)

    print(f"\n{'='*50}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Positional: {hasattr(model, 'spatial_proj')}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}")
    print(f"\nTarget ({args.target}):")
    print(render(torch.zeros_like(target, dtype=torch.long), target).replace(".", "_").replace("x", "#"))

    for seed in seed_list:
        set_seed(seed)
        world = World(width=args.width, height=args.height, init_cells=args.init_cells, seed=seed)
        world.reset()
        result = simulate(
            world=world, model=model, target_mask=target, steps=args.steps,
            writer=None, viz=None, render_every=999, device=torch.device("cpu"),
            anti_extinction_warmup_steps=args.warmup,
            die_cap_schedule=DIE_CAP_SCHEDULE, late_cleanup_start_frac=0.75,
            verbose=False,
        )
        final_iou = compute_iou(world.grid, target)
        print(f"\n--- seed={seed}  final_iou={final_iou:.4f}  best_iou={result['best_iou']:.4f}"
              f"  alive={result['alive_end']} ---")
        print(render(world.grid, target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
