"""Render a side-by-side comparison GIF of two checkpoints on the same seed.

Usage:
    python scripts/render_comparison.py <legacy_ckpt> <champion_ckpt> [--seed 0]

Useful for showing the before/after improvement: legacy model on the left,
champion on the right, both growing in lock-step on the same world seed.
A label band on top shows the model name.
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
    # Legacy checkpoints (pre bc0490a) have head with 5 actions; new ones have 9.
    ckpt_actions = int(ckpt["head.weight"].shape[0])
    if ckpt_actions != model.head.out_features:
        import torch.nn as nn
        model.head = nn.Linear(embed_dim, ckpt_actions)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def collect_frames(model: LittleLM, target: torch.Tensor, seed: int,
                   width: int, height: int, init_cells: int,
                   steps: int, warmup: int, late_cleanup: float,
                   capture_every: int, render_size: int) -> List[np.ndarray]:
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


def render_text_band(width: int, height: int, text: str,
                     bg: int = 30, fg: int = 230) -> np.ndarray:
    """Render a simple text band using PIL if available, else solid colour."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return np.full((height, width, 3), bg, dtype=np.uint8)
    img = Image.new("RGB", (width, height), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, height - 14))
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2 - 2), text,
              fill=(fg, fg, fg), font=font)
    return np.array(img)


def stitch_side_by_side(left: List[np.ndarray], right: List[np.ndarray],
                        left_label: str, right_label: str,
                        gap: int = 12, label_h: int = 36) -> List[np.ndarray]:
    n = min(len(left), len(right))
    h, w, _ = left[0].shape
    canvas_w = w * 2 + gap
    canvas_h = h + label_h
    band_left = render_text_band(w, label_h, left_label)
    band_right = render_text_band(w, label_h, right_label)

    out: List[np.ndarray] = []
    for i in range(n):
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
        canvas[:label_h, :w] = band_left
        canvas[:label_h, w + gap:] = band_right
        canvas[label_h:, :w] = left[i]
        canvas[label_h:, w + gap:] = right[i]
        out.append(canvas)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_checkpoint", type=Path)
    parser.add_argument("champion_checkpoint", type=Path)
    parser.add_argument("--legacy-label", type=str, default="legacy (MS-10 0.329)")
    parser.add_argument("--champion-label", type=str, default="champion (MS-10 0.440)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--target", type=str, default="T")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--init-cells", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--late-cleanup", type=float, default=0.6522)
    parser.add_argument("--capture-every", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render-size", type=int, default=320)
    args = parser.parse_args()

    for ckpt in (args.legacy_checkpoint, args.champion_checkpoint):
        if not ckpt.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt}", file=sys.stderr)
            return 2

    output = args.output or REPO_ROOT / "docs" / "assets" / "comparison.gif"
    output.parent.mkdir(parents=True, exist_ok=True)

    target = make_target(args.target, args.height, args.width)

    print(f"[INFO] Running legacy: {args.legacy_checkpoint.name}")
    legacy_model = load_model(args.legacy_checkpoint)
    legacy_frames = collect_frames(
        legacy_model, target, args.seed,
        args.width, args.height, args.init_cells,
        args.steps, args.warmup, args.late_cleanup,
        args.capture_every, args.render_size,
    )

    print(f"[INFO] Running champion: {args.champion_checkpoint.name}")
    champion_model = load_model(args.champion_checkpoint)
    champion_frames = collect_frames(
        champion_model, target, args.seed,
        args.width, args.height, args.init_cells,
        args.steps, args.warmup, args.late_cleanup,
        args.capture_every, args.render_size,
    )

    print(f"[INFO] Stitching {min(len(legacy_frames), len(champion_frames))} frames...")
    composite = stitch_side_by_side(
        legacy_frames, champion_frames,
        args.legacy_label, args.champion_label,
    )

    try:
        from imageio import v2 as imageio
    except ImportError:
        print("[ERROR] imageio is required. Run: pip install 'imageio[pillow]'", file=sys.stderr)
        return 1

    duration = 1.0 / max(1, args.fps)
    print(f"[INFO] Writing GIF: {output}")
    imageio.mimsave(str(output), composite, duration=duration, loop=0)
    print(f"[OK] Saved: {output} ({output.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
