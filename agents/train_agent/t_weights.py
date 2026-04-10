"""T-shape structure weight scaling for curriculum phases."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

T_STRUCTURE_REWARD_KEYS: Sequence[str] = (
    "top_bar_weight",
    "trunk_weight",
    "symmetry_weight",
    "bar_alignment_weight",
    "trunk_alignment_weight",
    "junction_weight",
)

T_STRUCTURE_PENALTY_KEYS: Sequence[str] = (
    "excess_below_weight",
    "excess_side_weight",
    "overshoot_weight",
    "undershoot_weight",
    "fragment_weight",
    "trunk_extra_weight",
    "side_clutter_weight",
    "off_axis_weight",
    "endpoint_clutter_weight",
)


def scale_t_weights(
    base_weights: Optional[Dict[str, float]],
    reward_scale: float,
    penalty_scale: float,
    clean_scale: float,
) -> Optional[Dict[str, float]]:
    """Scale T-shape structure weights by reward/penalty/cleanliness factors."""
    if not base_weights:
        return None
    scaled = {k: float(v) for k, v in base_weights.items()}
    for key in T_STRUCTURE_REWARD_KEYS:
        if key in scaled:
            scaled[key] *= reward_scale
    for key in T_STRUCTURE_PENALTY_KEYS:
        if key in scaled:
            scaled[key] *= penalty_scale
    if "cleanliness_weight" in scaled:
        scaled["cleanliness_weight"] *= clean_scale
    return scaled
