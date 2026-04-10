"""Late cleanup ratio computation for simulation end-phase death pressure."""
from __future__ import annotations

from typing import Any, Dict


def compute_late_cleanup_ratio(
    steps_done: int, planned_steps: int, cleanup_start_frac: float = 0.8
) -> float:
    """Calculate cleanup ratio based on how far into the late-cleanup window we are."""
    if planned_steps <= 0:
        return 0.0
    start_frac = max(0.0, min(1.0, float(cleanup_start_frac)))
    cleanup_start = max(1, int(planned_steps * start_frac))
    if cleanup_start > planned_steps:
        cleanup_start = planned_steps
    if steps_done < cleanup_start:
        return 0.0
    window_span = max(1, planned_steps - cleanup_start + 1)
    late_steps = steps_done - cleanup_start + 1
    ratio = late_steps / window_span
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return float(ratio)


def resolve_cleanup_ratio(
    sim_result: Dict[str, Any], planned_steps: int, cleanup_start_frac: float
) -> float:
    """Resolve cleanup ratio from simulation result or compute it."""
    ratio = sim_result.get("late_cleanup_ratio")
    if ratio is not None:
        try:
            clamped = float(ratio)
        except (TypeError, ValueError):
            clamped = 0.0
        else:
            if clamped < 0.0:
                return 0.0
            if clamped > 1.0:
                return 1.0
            return clamped
    steps_done = int(sim_result.get("steps", planned_steps))
    return compute_late_cleanup_ratio(steps_done, planned_steps, cleanup_start_frac)
