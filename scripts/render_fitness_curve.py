"""Render a fitness/IoU trajectory plot from a generations.csv file.

Usage:
    python scripts/render_fitness_curve.py <run_dir> [--output curve.png]

Reads runs/<dir>/generations.csv and plots best_iou per generation, with
phase boundaries shaded and key milestones annotated. Output is a clean
matplotlib PNG suitable for embedding in articles.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv(csv_path: Path) -> Tuple[List[int], List[float], List[str]]:
    gens: List[int] = []
    ious: List[float] = []
    phases: List[str] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                gens.append(int(row.get("generation", row.get("gen", 0))))
                ious.append(float(row.get("best_iou", 0.0)))
                phases.append(str(row.get("phase", "")))
            except (TypeError, ValueError):
                continue
    return gens, ious, phases


def phase_ranges(gens: List[int], phases: List[str]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    if not gens:
        return out
    cur_phase = phases[0]
    cur_start = gens[0]
    for i in range(1, len(gens)):
        if phases[i] != cur_phase:
            out.append((cur_phase, cur_start, gens[i - 1]))
            cur_phase = phases[i]
            cur_start = gens[i]
    out.append((cur_phase, cur_start, gens[-1]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-iou", type=float, default=0.440)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--width", type=float, default=11.0)
    parser.add_argument("--height", type=float, default=4.5)
    args = parser.parse_args()

    csv_path = args.run_dir / "generations.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found", file=sys.stderr)
        return 2

    gens, ious, phases = read_csv(csv_path)
    if not gens:
        print(f"[ERROR] No data in {csv_path}", file=sys.stderr)
        return 2

    output = args.output or args.run_dir / "fitness_curve.png"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib required. Run: pip install matplotlib", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=130)

    phase_colors = {
        "phase_cross": "#2563eb",
        "phase_transition": "#9333ea",
        "phase_T": "#16a34a",
        "phase_T_refine": "#ca8a04",
    }
    fallback = "#6b7280"

    ranges = phase_ranges(gens, phases)
    for phase, start, end in ranges:
        color = phase_colors.get(phase, fallback)
        ax.axvspan(start, end, alpha=0.08, color=color)
        mid = (start + end) / 2
        ax.text(mid, 0.02, phase, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=color, alpha=0.85)

    ax.plot(gens, ious, color="#0f172a", linewidth=1.2, label="best IoU per gen")

    peak_idx = max(range(len(ious)), key=lambda i: ious[i])
    ax.scatter([gens[peak_idx]], [ious[peak_idx]],
               color="#dc2626", zorder=5, s=40,
               label=f"peak {ious[peak_idx]:.3f} @ gen {gens[peak_idx]}")

    if args.baseline_iou > 0:
        ax.axhline(y=args.baseline_iou, color="#7c3aed", linestyle="--",
                   linewidth=1, alpha=0.6,
                   label=f"baseline MS-10 {args.baseline_iou:.3f}")

    ax.set_xlabel("generation")
    ax.set_ylabel("best IoU")
    ax.set_title(args.title or f"Training trajectory · {args.run_dir.name}")
    ax.set_ylim(0, max(0.65, max(ious) * 1.1))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(output, dpi=130, bbox_inches="tight")
    print(f"[OK] Saved: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
