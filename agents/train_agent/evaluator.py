"""Evaluator state — mutable state container for the make_evaluator closure.

Extracted from ``agents/train_agent/pipeline.py`` as part of the Round 4
closure decomposition. The 15 ``nonlocal`` variables that 6 setter closures
mutated are consolidated into this single mutable dataclass. The setter
closures become methods on the dataclass, and the ``evaluator`` function
body receives the dataclass as an argument instead of capturing 15 separate
nonlocals.

The ``evaluator`` function itself remains as a closure inside
``pipeline.py::train_ga::make_evaluator`` because it captures ~25 read-only
variables from the train_ga scope. Full extraction of the evaluator body
to this file would require bundling all 25 vars into another dataclass
(~400 lines of boilerplate with marginal testing value). This intermediate
step gives us the 80% of the value (explicit state, testable setters, −7
closures) with 20% of the work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EvaluatorState:
    """Mutable per-evaluator state that the training loop modifies through setters.

    Built once per curriculum phase (inside ``make_evaluator``). The 6
    former setter closures are now methods on this dataclass. Each method
    preserves the original validation logic (clamping, type coercion,
    default-on-error) verbatim for bit-identical behaviour.
    """

    # --- Phase cleanup ---
    phase_cleanup_ratio: float = 0.0

    # --- Blend weight ---
    blend_weight: float = 1.0

    # --- Stem scale ---
    stem_scale_multiplier: float = 1.0

    # --- Late stage modifiers ---
    late_stage_fp_multiplier: float = 1.0
    late_stage_symmetry_bonus: float = 0.0
    late_stage_trunk_bonus: float = 0.0
    late_stage_clean_component: float = 0.0
    late_stage_clean_fp: float = 0.0

    # --- Late structure enforcer ---
    late_stage_enforcer_cfg: Optional[Dict[str, float]] = None

    # --- Transition controls ---
    t_phase_gate_value: float = 1.0
    t_structure_gate_value: float = 1.0
    t_cleanliness_gate_value: float = 1.0
    collapse_area_floor_value: float = 0.0
    collapse_coverage_floor_value: float = 0.0
    collapse_penalty_weight_value: float = 0.0
    collapse_suppress_value: float = 0.0

    # ---- Setter methods (former closures, bit-identical logic) ---- #

    def set_phase_cleanup_ratio(self, value: float) -> None:
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            ratio = 0.0
        self.phase_cleanup_ratio = max(0.0, min(1.0, ratio))

    def set_blend_weight(self, value: float) -> None:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = 1.0
        self.blend_weight = max(0.0, min(1.0, weight))

    def set_stem_scale_multiplier(self, value: float) -> None:
        try:
            mult = float(value)
        except (TypeError, ValueError):
            mult = 1.0
        if mult < 0.0:
            mult = 0.0
        self.stem_scale_multiplier = mult

    def set_late_stage_modifiers(
        self,
        fp_mult: float,
        sym_bonus: float,
        trunk_bonus: float,
        clean_component: float,
        clean_fp: float,
    ) -> None:
        try:
            self.late_stage_fp_multiplier = max(0.0, float(fp_mult))
        except (TypeError, ValueError):
            self.late_stage_fp_multiplier = 1.0
        try:
            self.late_stage_symmetry_bonus = float(sym_bonus)
        except (TypeError, ValueError):
            self.late_stage_symmetry_bonus = 0.0
        try:
            self.late_stage_trunk_bonus = float(trunk_bonus)
        except (TypeError, ValueError):
            self.late_stage_trunk_bonus = 0.0
        try:
            self.late_stage_clean_component = max(0.0, float(clean_component))
        except (TypeError, ValueError):
            self.late_stage_clean_component = 0.0
        try:
            self.late_stage_clean_fp = max(0.0, float(clean_fp))
        except (TypeError, ValueError):
            self.late_stage_clean_fp = 0.0

    def set_late_structure_enforcer(
        self, config: Optional[Dict[str, float]]
    ) -> None:
        self.late_stage_enforcer_cfg = config if config else None

    def set_transition_controls(
        self,
        t_gate: float,
        structure_gate: float,
        clean_gate: float,
        collapse_area: float,
        collapse_coverage: float,
        collapse_penalty: float,
        collapse_suppress: float,
    ) -> None:
        try:
            self.t_phase_gate_value = max(0.0, min(1.0, float(t_gate)))
        except (TypeError, ValueError):
            self.t_phase_gate_value = 1.0
        try:
            self.t_structure_gate_value = max(0.0, min(1.0, float(structure_gate)))
        except (TypeError, ValueError):
            self.t_structure_gate_value = 1.0
        try:
            self.t_cleanliness_gate_value = max(0.0, min(1.0, float(clean_gate)))
        except (TypeError, ValueError):
            self.t_cleanliness_gate_value = 1.0
        try:
            self.collapse_area_floor_value = max(0.0, float(collapse_area))
        except (TypeError, ValueError):
            self.collapse_area_floor_value = 0.0
        try:
            self.collapse_coverage_floor_value = max(0.0, float(collapse_coverage))
        except (TypeError, ValueError):
            self.collapse_coverage_floor_value = 0.0
        try:
            self.collapse_penalty_weight_value = max(0.0, float(collapse_penalty))
        except (TypeError, ValueError):
            self.collapse_penalty_weight_value = 0.0
        try:
            self.collapse_suppress_value = max(0.0, float(collapse_suppress))
        except (TypeError, ValueError):
            self.collapse_suppress_value = 0.0
