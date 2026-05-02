"""Render an animated 2x5 grid showing 10 seeds simulating in parallel.

Usage:
    python scripts/render_multiseed_montage.py <checkpoint> [--output multiseed.gif]

Demonstrates seed-consistency of a trained model: runs 10 simulations on
seeds 0..9 in parallel (in time, sequential capture), tiles their grids
into a 2-row by 5-col montage, and writes one GIF where all 10 seeds
evolve simultaneously.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
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


def collect_seed_frames(
    model: LittleLM, target: torch.Tensor, seed: int,
    width: int, height: int, init_cells: int,
    steps: int, warmup: int, late_cleanup: float,
    capture_every: int, render_size: int,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []

    def capture(grid: torch.Tensor, step_idx: int) -> None:
        frames.append(render_grid_to_rgb(grid, target_mask=target, target_size=render_size))

    set_seed(seed)
    world = World(width=width, height=height, init_cells=init_cells, seed=seed)
    world.reset()
    capture(world.grid, 0)
    simulate(
        world=world, model=model, target_mask=target, steps=steps,
        writer=None, viz=None, render_every=999, device=torch.device("cpu"),
        anti_extinction_warmup_steps=warmup,
        die_cap_schedule=DIE_CAP_SCHEDULE,
        late_cleanup_start_frac=late_cleanup,
        verbose=False,
        frame_capture=capture,
        capture_every=capture_every,
    )
    return frames


def tile_montage(frames_per_seed: List[List[np.ndarray]], rows: int, cols: int,
                 padding: int = 8, bg_color: int = 30) -> List[np.ndarray]:
    n_frames = min(len(seq) for seq in frames_per_seed)
    h, w, _ = frames_per_seed[0][0].shape
    canvas_h = rows * h + (rows + 1) * padding
    canvas_w = cols * w + (cols + 1) * padding
    out: List[np.ndarray] = []
    for f_idx in range(n_frames):
        canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
        for s_idx, seq in enumerate(frames_per_seed):
            r, c = divmod(s_idx, cols)
            y0 = padding + r * (h + padding)
            x0 = padding + c * (w + padding)
            canvas[y0:y0 + h, x0:x0 + w] = seq[f_idx]
        out.append(canvas)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--target", type=str, default="T")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--init-cells", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--late-cleanup", type=float, default=0.6522)
    parser.add_argument("--capture-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render-size", type=int, default=160)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) != args.rows * args.cols:
        print(f"[WARN] {len(seeds)} seeds but rows*cols = {args.rows*args.cols}; using min")
    output = args.output or REPO_ROOT / "docs" / "assets" / "multiseed_montage.gif"
    output.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)
    target = make_target(args.target, args.height, args.width)

    print(f"[INFO] Collecting frames for {len(seeds)} seeds...")
    frames_per_seed: List[List[np.ndarray]] = []
    for seed in seeds:
        frames = collect_seed_frames(
            model, target, seed,
            args.width, args.height, args.init_cells,
            args.steps, args.warmup, args.late_cleanup,
            args.capture_every, args.render_size,
        )
        print(f"  seed={seed}: {len(frames)} frames")
        frames_per_seed.append(frames)

    print(f"[INFO] Tiling into {args.rows}x{args.cols} montage...")
    montage = tile_montage(frames_per_seed, args.rows, args.cols)

    try:
        from imageio import v2 as imageio
    except ImportError:
        print("[ERROR] imageio is required. Run: pip install 'imageio[pillow]'", file=sys.stderr)
        return 1

    duration = 1.0 / max(1, args.fps)
    print(f"[INFO] Writing GIF: {output} ({len(montage)} frames @ {args.fps} fps)")
    imageio.mimsave(str(output), montage, duration=duration, loop=0)
    print(f"[OK] Saved: {output} ({output.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
