"""Build a leaderboard of all historical best.pt checkpoints.

Scans `runs/` for every subdir containing `best.pt`, loads each model,
runs a multi-seed benchmark, and produces a ranked Markdown table.

Usage:
    python scripts/build_leaderboard.py [--out docs/leaderboard.md] [--seeds 5]

Fast mode uses 5 seeds by default (to keep total runtime manageable across
48+ checkpoints). Use --seeds 10 for the full standard benchmark.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

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


@dataclass
class LeaderboardRow:
    run_name: str
    checkpoint_path: Path
    has_spatial: bool
    n_params: int
    ms_mean: float
    ms_min: float
    ms_max: float
    reach30: float
    reach40: float
    load_error: Optional[str] = None


def count_params(model: LittleLM) -> int:
    return sum(p.numel() for p in model.parameters())


def try_load_model(checkpoint_path: Path, embed_dim: int, num_heads: int):
    """Attempt to load a checkpoint across multiple architectures.

    Older runs may have been trained before positional encoding was added,
    or with different embed_dim / num_heads. We try a few sensible options.
    """
    try:
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    except Exception as err:
        return None, f"load_fail: {type(err).__name__}"

    has_spatial = any(k.startswith("spatial_proj") for k in ckpt)

    # Detect embed_dim from token_embed weight if present
    token_key = "token_embed.weight"
    if token_key in ckpt:
        embed_dim = int(ckpt[token_key].shape[1])

    for heads_try in (num_heads, 4, 2, 8):
        try:
            model = LittleLM(embed_dim=embed_dim, num_heads=heads_try, use_position=has_spatial)
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if unexpected:
                continue
            model.eval()
            return model, None
        except Exception:
            continue
    return None, "shape_mismatch"


def benchmark_checkpoint(
    model: LittleLM,
    n_seeds: int,
    steps: int,
    target_name: str = "T",
) -> List[float]:
    target = make_target(target_name, 15, 15)
    bests = []
    for seed in range(n_seeds):
        set_seed(seed)
        world = World(width=15, height=15, init_cells=50, seed=seed)
        world.reset()
        try:
            result = simulate(
                world=world,
                model=model,
                target_mask=target,
                steps=steps,
                writer=None,
                viz=None,
                render_every=999,
                device=torch.device("cpu"),
                anti_extinction_warmup_steps=25,
                die_cap_schedule=DEFAULT_DIE_CAP_SCHEDULE,
                late_cleanup_start_frac=0.75,
                verbose=False,
            )
            bests.append(float(result["best_iou"]))
        except Exception:
            bests.append(0.0)
    return bests


def scan_checkpoints(runs_dir: Path) -> List[Path]:
    """Find every best.pt under runs/."""
    return sorted(runs_dir.glob("*/best.pt"))


def build_rows(
    checkpoints: List[Path],
    n_seeds: int,
    steps: int,
) -> List[LeaderboardRow]:
    rows = []
    for i, ckpt_path in enumerate(checkpoints):
        run_name = ckpt_path.parent.name
        print(f"[{i+1}/{len(checkpoints)}] {run_name} ...", flush=True)
        model, err = try_load_model(ckpt_path, embed_dim=32, num_heads=4)
        if err:
            rows.append(LeaderboardRow(
                run_name=run_name,
                checkpoint_path=ckpt_path,
                has_spatial=False,
                n_params=0,
                ms_mean=0.0, ms_min=0.0, ms_max=0.0,
                reach30=0.0, reach40=0.0,
                load_error=err,
            ))
            continue
        bests = benchmark_checkpoint(model, n_seeds, steps)
        has_spatial = hasattr(model, "spatial_proj")
        rows.append(LeaderboardRow(
            run_name=run_name,
            checkpoint_path=ckpt_path,
            has_spatial=has_spatial,
            n_params=count_params(model),
            ms_mean=sum(bests) / len(bests),
            ms_min=min(bests),
            ms_max=max(bests),
            reach30=sum(1 for v in bests if v >= 0.30) / len(bests),
            reach40=sum(1 for v in bests if v >= 0.40) / len(bests),
        ))
    return rows


def format_markdown(rows: List[LeaderboardRow], n_seeds: int) -> str:
    rows_ok = [r for r in rows if r.load_error is None]
    rows_fail = [r for r in rows if r.load_error is not None]
    rows_ok.sort(key=lambda r: r.ms_mean, reverse=True)

    lines = []
    lines.append("# Checkpoint Leaderboard")
    lines.append("")
    lines.append(f"Generated by `scripts/build_leaderboard.py` across **{len(rows)}** historical checkpoints.")
    lines.append(f"Benchmark: {n_seeds} world seeds, target T on 15×15 grid, 200 sim steps, standard 3-phase die_cap.")
    lines.append("")
    lines.append(f"- Loaded successfully: **{len(rows_ok)}**")
    lines.append(f"- Failed to load: **{len(rows_fail)}**")
    lines.append("")
    lines.append("## Top 20 by multi-seed mean IoU")
    lines.append("")
    lines.append("| Rank | Run | Pos? | Params | MS mean | MS min | MS max | reach30 | reach40 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for rank, row in enumerate(rows_ok[:20], start=1):
        pos = "✓" if row.has_spatial else "—"
        lines.append(
            f"| {rank} | `{row.run_name}` | {pos} | {row.n_params:,} | "
            f"**{row.ms_mean:.4f}** | {row.ms_min:.4f} | {row.ms_max:.4f} | "
            f"{int(row.reach30*100)}% | {int(row.reach40*100)}% |"
        )
    lines.append("")

    if len(rows_ok) > 20:
        lines.append(f"## Remaining {len(rows_ok) - 20} loadable checkpoints (compact)")
        lines.append("")
        lines.append("| Rank | Run | MS mean | reach30 |")
        lines.append("|---|---|---|---|")
        for rank, row in enumerate(rows_ok[20:], start=21):
            lines.append(f"| {rank} | `{row.run_name}` | {row.ms_mean:.4f} | {int(row.reach30*100)}% |")
        lines.append("")

    if rows_fail:
        lines.append("## Failed to load")
        lines.append("")
        lines.append("| Run | Error |")
        lines.append("|---|---|")
        for row in rows_fail:
            lines.append(f"| `{row.run_name}` | {row.load_error} |")
        lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")
    if rows_ok:
        best = rows_ok[0]
        spatial_rows = [r for r in rows_ok if r.has_spatial]
        legacy_rows = [r for r in rows_ok if not r.has_spatial]
        lines.append(f"- **Top checkpoint:** `{best.run_name}` — MS mean IoU {best.ms_mean:.4f}")
        lines.append(f"- **Positional models:** {len(spatial_rows)} / {len(rows_ok)}")
        if spatial_rows and legacy_rows:
            avg_spatial = sum(r.ms_mean for r in spatial_rows) / len(spatial_rows)
            avg_legacy = sum(r.ms_mean for r in legacy_rows) / len(legacy_rows)
            lines.append(f"- **Average MS IoU (positional):** {avg_spatial:.4f}")
            lines.append(f"- **Average MS IoU (legacy, no position):** {avg_legacy:.4f}")
            lines.append(f"- **Positional advantage:** {(avg_spatial - avg_legacy):+.4f}")
    lines.append("")
    lines.append("---")
    lines.append("*Regenerate with: `python scripts/build_leaderboard.py`*")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "leaderboard.md")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds per checkpoint (default 5 for speed; 10 for standard bench)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only benchmark first N checkpoints (0 = all)")
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"[ERROR] runs dir not found: {args.runs_dir}", file=sys.stderr)
        return 2

    checkpoints = scan_checkpoints(args.runs_dir)
    if args.limit > 0:
        checkpoints = checkpoints[:args.limit]
    print(f"[INFO] Found {len(checkpoints)} checkpoints to benchmark")

    rows = build_rows(checkpoints, n_seeds=args.seeds, steps=args.steps)
    markdown = format_markdown(rows, n_seeds=args.seeds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(f"\n[INFO] Leaderboard written to {args.out}")

    rows_ok = sorted([r for r in rows if r.load_error is None], key=lambda r: r.ms_mean, reverse=True)
    if rows_ok:
        print("\nTop 5 by MS mean:")
        for i, r in enumerate(rows_ok[:5], 1):
            print(f"  {i}. {r.run_name}  mean={r.ms_mean:.4f}  reach30={int(r.reach30*100)}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
