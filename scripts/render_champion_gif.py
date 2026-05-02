"""Render an animated GIF of a trained model growing the target shape.

Usage:
    python scripts/render_champion_gif.py <checkpoint> [--seed 0] [--output champion.gif]

Loads a LittleLM checkpoint, runs a single-seed simulation, captures every
frame, and saves the trajectory as a GIF. The visual matches the on-screen
pygame renderer:
    - black     = empty cell
    - grey      = stem
    - green     = differentiated cell IN target
    - orange    = differentiated cell OUTSIDE target (false positive)
    - dark blue = empty IN target (false negative)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import LittleLM  # noqa: E402
from src.simulate import simulate  # noqa: E402
from src.targets import make_target  # noqa: E402
from src.utils import set_seed  # noqa: E402
from src.viz import render_grid_to_rgb  # noqa: E402
from src.world import World  # noqa: E402

DIE_CAP_SCHEDULE = [
    {"until_step": 60, "cap_frac": 0.20},
    {"until_step": 130, "cap_frac": 0.40},
    {"until_step": 999999, "cap_frac": 0.35},
]


def load_model(ckpt_path: Path) -> LittleLM:
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    has_spatial = any(k.startswith("spatial_proj") for k in ckpt)
    embed_dim = int(ckpt["token_embed.weight"].shape[1]) if "token_embed.weight" in ckpt else 32
    model = LittleLM(embed_dim=embed_dim, num_heads=4, use_position=has_spatial)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output GIF path (default: docs/assets/champion_seed<N>.gif)")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--target", type=str, default="T")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--init-cells", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--late-cleanup", type=float, default=0.6522)
    parser.add_argument("--capture-every", type=int, default=2,
                        help="Capture 1 frame every N sim steps (default 2)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render-size", type=int, default=384,
                        help="Pixel size of each grid frame (default 384)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    output = args.output
    if output is None:
        output_dir = REPO_ROOT / "docs" / "assets"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"champion_seed{args.seed}.gif"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)
    target = make_target(args.target, args.height, args.width)

    frames: List = []

    def capture(grid: torch.Tensor, step_idx: int) -> None:
        rgb = render_grid_to_rgb(grid, target_mask=target, target_size=args.render_size)
        frames.append(rgb)

    set_seed(args.seed)
    world = World(
        width=args.width, height=args.height,
        init_cells=args.init_cells, seed=args.seed,
    )
    world.reset()

    capture(world.grid, 0)

    print(f"[INFO] Running sim: ckpt={args.checkpoint.name}, seed={args.seed}, steps={args.steps}")
    result = simulate(
        world=world, model=model, target_mask=target, steps=args.steps,
        writer=None, viz=None, render_every=999, device=torch.device("cpu"),
        anti_extinction_warmup_steps=args.warmup,
        die_cap_schedule=DIE_CAP_SCHEDULE,
        late_cleanup_start_frac=args.late_cleanup,
        verbose=False,
        frame_capture=capture,
        capture_every=args.capture_every,
    )

    print(f"[INFO] Captured {len(frames)} frames | best_iou={result['best_iou']:.4f}")

    try:
        from imageio import v2 as imageio
    except ImportError:
        print("[ERROR] imageio is required. Run: pip install 'imageio[pillow]'", file=sys.stderr)
        return 1

    duration = 1.0 / max(1, args.fps)
    print(f"[INFO] Writing GIF: {output} (fps={args.fps}, {len(frames)} frames)")
    imageio.mimsave(str(output), frames, duration=duration, loop=0)
    print(f"[OK] Saved: {output} ({output.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
