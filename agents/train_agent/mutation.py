"""Mutation and penalty resolution for curriculum-aware GA training."""
from __future__ import annotations

from typing import NamedTuple


class TMutationConfig(NamedTuple):
    """Parameters controlling T-phase mutation schedule."""
    ease_power: float = 1.5
    std_start: float = 0.012
    std_end: float = 0.004
    std_floor: float = 0.002
    std_cap: float = 0.015
    warmup_frac: float = 0.15


def resolve_mutation_std(
    phase_target: str,
    phase_name: str,
    phase_step: int,
    total_steps: int,
    *,
    default_std: float = 0.01,
    t_cfg: TMutationConfig = TMutationConfig(),
) -> float:
    """Resolve mutation std for current phase/step with curriculum awareness."""
    key = str(phase_target or "").lower()
    name_key = str(phase_name or "").lower()

    if "refine" in name_key:
        return 0.0015
    if key == "cross" or name_key.startswith("phase_cross"):
        return 0.02
    if "transition" in name_key or name_key.startswith("phase_transition"):
        return 0.015
    if key == "t" or name_key.startswith("phase_t"):
        progress = (phase_step + 1) / max(1, total_steps)
        eased = progress ** t_cfg.ease_power
        base_std = t_cfg.std_start + (t_cfg.std_end - t_cfg.std_start) * eased
        base_std = max(t_cfg.std_floor, base_std)
        warmup_gate = min(1.0, progress / max(1e-6, t_cfg.warmup_frac))
        gated_std = t_cfg.std_floor + (base_std - t_cfg.std_floor) * warmup_gate
        return max(t_cfg.std_floor, min(t_cfg.std_cap, gated_std))
    return default_std


def resolve_stem_penalty_multiplier(
    phase_target: str,
    phase_name: str,
    phase_step: int,
    total_steps: int,
) -> float:
    """Resolve stem penalty multiplier based on curriculum phase."""
    key = str(phase_target or "").lower()
    name_key = str(phase_name or "").lower()

    if "transition" in name_key or name_key.startswith("phase_transition"):
        return 0.6
    if key == "t" or name_key.startswith("phase_t"):
        easing_steps = max(1, min(total_steps, 20))
        if phase_step < easing_steps:
            start, end = 0.6, 1.0
            ratio = (phase_step + 1) / easing_steps
            return start + (end - start) * max(0.0, min(1.0, ratio))
        return 1.0
    return 1.0


def compute_phase_cleanup_ratio(
    phase_target: str,
    phase_step: int,
    total_steps: int,
    *,
    cleanup_start_frac: float = 0.6,
    cleanup_power: float = 1.5,
) -> float:
    """Compute phase-specific stem cleanup ratio for T-phase."""
    key = str(phase_target or "").lower()
    if key != "t":
        return 0.0
    if cleanup_start_frac >= 1.0:
        return 0.0
    progress = (phase_step + 1) / max(1, total_steps)
    if progress <= cleanup_start_frac:
        return 0.0
    span = max(1e-6, 1.0 - cleanup_start_frac)
    ratio = (progress - cleanup_start_frac) / span
    ratio = max(0.0, min(1.0, ratio))
    return ratio ** cleanup_power
