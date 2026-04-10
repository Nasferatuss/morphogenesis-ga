"""Metrics persistence: CSV generation logs, run summaries, TensorBoard plot export."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def write_generations_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write per-generation metrics to CSV, preserving key insertion order."""
    if not records:
        return

    fieldnames: List[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_run_summary(path: Path, payload: Dict[str, Any]) -> None:
    """Write run summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_tensorboard_plots(run_dir: Path, tags: Sequence[str]) -> None:
    """Export key TensorBoard scalar tags as PNG plots."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as err:  # pragma: no cover
        print(f"[WARN] Unable to export TensorBoard plots: {err}")
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as err:  # pragma: no cover
        print(f"[WARN] Matplotlib not available for plot export: {err}")
        return
    if not run_dir.exists():
        return
    event_files = list(run_dir.glob('events.out.tfevents.*'))
    if not event_files:
        print(f"[WARN] No TensorBoard event files found in {run_dir}")
        return
    accumulator = EventAccumulator(str(run_dir))
    try:
        accumulator.Reload()
    except Exception as err:  # pragma: no cover
        print(f"[WARN] Failed to read TensorBoard events: {err}")
        return
    scalar_tags = set(accumulator.Tags().get('scalars', []))
    for tag in tags:
        if tag not in scalar_tags:
            continue
        scalars = accumulator.Scalars(tag)
        if not scalars:
            continue
        steps = [item.step for item in scalars]
        values = [item.value for item in scalars]
        plt.figure(figsize=(6, 3.5))
        plt.plot(steps, values, label=tag, linewidth=1.4)
        plt.xlabel('step')
        plt.ylabel(tag)
        plt.title(tag)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        safe_name = tag.replace('/', '_') + '.png'
        out_path = run_dir / safe_name
        try:
            plt.savefig(out_path, dpi=180)
        except Exception as err:  # pragma: no cover
            print(f"[WARN] Failed to save plot for {tag}: {err}")
        plt.close()
