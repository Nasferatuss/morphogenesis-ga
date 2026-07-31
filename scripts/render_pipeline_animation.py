"""Render an animated schematic of the morphogenesis-ga pipeline.

Usage:
    python scripts/render_pipeline_animation.py [--output pipeline.gif]

Produces a matplotlib animation suitable for LinkedIn / blog headers.
Shows the inference pipeline of one cell decision, with travelling
particles flowing through:

    [grid + 8 neighbours] -> [LittleLM 5K params] -> [5 actions] -> [grid update]

and a feedback loop labelled "GA / curriculum (cross -> T)".

The output is a self-contained GIF, no project imports needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "docs" / "assets" / "pipeline.gif")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-sec", type=float, default=4.0,
                        help="Total animation length in seconds")
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=4.5)
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError:
        print("[ERROR] matplotlib required. Run: pip install matplotlib", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    BG = "#0b1220"
    NODE_BG = "#111827"
    TEXT_PRIMARY = "#e5e7eb"
    TEXT_SECONDARY = "#9ca3af"
    PARTICLE_TRAIL = "#34d399"
    PARTICLE_HEAD = "#fbbf24"
    ARROW_COLOR = "#475569"
    ACCENT = "#60a5fa"

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.set_axis_off()

    nodes = [
        {"pos": (1.5, 2.6), "w": 2.0, "h": 1.6,
         "title": "8 соседей", "subtitle": "+ позиция (y, x)", "color": "#1d4ed8"},
        {"pos": (5.0, 2.6), "w": 2.2, "h": 1.6,
         "title": "LittleLM", "subtitle": "5 065 параметров", "color": "#9333ea"},
        {"pos": (8.7, 2.6), "w": 2.0, "h": 1.6,
         "title": "5 действий", "subtitle": "stay / A / B / divide / die", "color": "#16a34a"},
    ]

    for node in nodes:
        cx, cy = node["pos"]
        x = cx - node["w"] / 2
        y = cy - node["h"] / 2
        box = FancyBboxPatch(
            (x, y), node["w"], node["h"],
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor=NODE_BG, edgecolor=node["color"], linewidth=1.6, zorder=2,
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.25, node["title"],
                ha="center", va="center", fontsize=14, color=TEXT_PRIMARY,
                weight="bold", zorder=3)
        ax.text(cx, cy - 0.30, node["subtitle"],
                ha="center", va="center", fontsize=9, color=TEXT_SECONDARY,
                zorder=3)

    arrow_segments = []
    for i in range(len(nodes) - 1):
        sx = nodes[i]["pos"][0] + nodes[i]["w"] / 2 + 0.05
        ex = nodes[i + 1]["pos"][0] - nodes[i + 1]["w"] / 2 - 0.05
        y = nodes[i]["pos"][1]
        ax.add_patch(FancyArrowPatch(
            (sx, y), (ex, y),
            arrowstyle="-|>", mutation_scale=14,
            color=ARROW_COLOR, linewidth=1.6, zorder=1,
        ))
        arrow_segments.append((sx, ex, y))

    loop_y_top = 0.85
    loop_y_arrow = 0.92
    last_x_right = nodes[-1]["pos"][0] + nodes[-1]["w"] / 2
    first_x_left = nodes[0]["pos"][0] - nodes[0]["w"] / 2
    ax.plot(
        [last_x_right, last_x_right, first_x_left, first_x_left],
        [nodes[-1]["pos"][1] - nodes[-1]["h"] / 2, loop_y_top, loop_y_top,
         nodes[0]["pos"][1] - nodes[0]["h"] / 2],
        color=ACCENT, linewidth=1.4, alpha=0.55, zorder=1,
    )
    ax.add_patch(FancyArrowPatch(
        (first_x_left + 0.02, loop_y_top), (first_x_left + 0.02, nodes[0]["pos"][1] - nodes[0]["h"] / 2 - 0.02),
        arrowstyle="-|>", mutation_scale=12, color=ACCENT, linewidth=1.4, alpha=0.7, zorder=1,
    ))
    ax.text((last_x_right + first_x_left) / 2, loop_y_arrow,
            "GA  ·  curriculum  cross → T",
            ha="center", va="bottom", fontsize=10, color=ACCENT, alpha=0.85)

    ax.text(6.0, 4.10, "morphogenesis-ga · inference loop",
            ha="center", va="center", fontsize=12, color=TEXT_PRIMARY, alpha=0.85)

    particle_trail, = ax.plot([], [], "-", color=PARTICLE_TRAIL,
                              linewidth=2.6, alpha=0.85, zorder=4)
    particle_head, = ax.plot([], [], "o", color=PARTICLE_HEAD,
                             markersize=11, markeredgecolor="#fde68a",
                             markeredgewidth=1.0, zorder=5)

    def particle_path():
        path = []
        n_per_seg = 60
        for sx, ex, y in arrow_segments:
            for k in range(n_per_seg):
                t = k / n_per_seg
                path.append((sx + (ex - sx) * t, y))
        last = nodes[-1]
        for k in range(20):
            t = k / 20
            path.append((last["pos"][0] - last["w"] / 2 + (last["w"]) * (1 - t), last["pos"][1]))
        path.append((nodes[0]["pos"][0], nodes[0]["pos"][1]))
        return path

    path = particle_path()
    n_frames = max(1, int(args.duration_sec * args.fps))

    def update(frame_idx: int):
        progress = (frame_idx % n_frames) / n_frames
        i = int(progress * len(path))
        head = path[min(i, len(path) - 1)]
        trail_start = max(0, i - 12)
        trail = path[trail_start:i + 1]
        if trail:
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            particle_trail.set_data(xs, ys)
        else:
            particle_trail.set_data([], [])
        particle_head.set_data([head[0]], [head[1]])
        return particle_trail, particle_head

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps,
                         blit=True, repeat=True)

    print(f"[INFO] Writing GIF: {args.output} ({n_frames} frames @ {args.fps} fps)")
    writer = PillowWriter(fps=args.fps)
    anim.save(str(args.output), writer=writer, dpi=130,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"[OK] Saved: {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
