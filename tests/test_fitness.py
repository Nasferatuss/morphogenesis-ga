"""Tests for src/fitness.py.

Covers the public surface (``compute_iou``, ``compute_fitness``) plus the
private helpers that dominate the uncovered branches: ``_connected_components``,
``_derive_stability_scores``, ``_compute_t_diagnostics``, and
``_compute_t_structure_breakdown``.

All cases construct explicit grids so expectations stay readable even when the
underlying scoring function is complex.
"""
from __future__ import annotations

import torch

from src.fitness import (
    _compute_t_diagnostics,
    _compute_t_structure_breakdown,
    _connected_components,
    _derive_stability_scores,
    compute_fitness,
    compute_iou,
)
from src.world import CELL_A, CELL_B, CELL_EMPTY, CELL_STEM


def _empty_grid(h: int = 5, w: int = 5) -> torch.Tensor:
    return torch.full((h, w), CELL_EMPTY, dtype=torch.int64)


def _t_target(h: int = 7, w: int = 7) -> torch.Tensor:
    """Canonical T shape: a top bar of width 5 and a trunk of height 4."""
    mask = torch.zeros((h, w), dtype=torch.bool)
    # top bar (row 1, cols 1..5)
    mask[1, 1:6] = True
    # trunk (rows 2..5, col 3)
    mask[2:6, 3] = True
    return mask


# -----------------------------------------------------------------------------
# compute_iou
# -----------------------------------------------------------------------------

class TestComputeIou:
    def test_perfect_match(self) -> None:
        grid = _empty_grid(3, 3)
        grid[0, 0] = CELL_A
        grid[0, 1] = CELL_B
        mask = torch.zeros((3, 3), dtype=torch.bool)
        mask[0, 0] = True
        mask[0, 1] = True
        assert compute_iou(grid, mask) == 1.0

    def test_empty_union_returns_zero(self) -> None:
        # exercises the union==0 branch (line 17)
        grid = _empty_grid(3, 3)
        mask = torch.zeros((3, 3), dtype=torch.bool)
        assert compute_iou(grid, mask) == 0.0

    def test_partial_overlap(self) -> None:
        grid = _empty_grid(3, 3)
        grid[0, 0] = CELL_A
        grid[0, 1] = CELL_A
        mask = torch.zeros((3, 3), dtype=torch.bool)
        mask[0, 0] = True
        mask[1, 0] = True
        # intersection=1 ({(0,0)}), union=3 ({(0,0),(0,1),(1,0)})
        assert abs(compute_iou(grid, mask) - 1.0 / 3.0) < 1e-6

    def test_ignores_stem_cells(self) -> None:
        grid = _empty_grid(3, 3)
        grid[0, 0] = CELL_STEM
        mask = torch.zeros((3, 3), dtype=torch.bool)
        mask[0, 0] = True
        assert compute_iou(grid, mask) == 0.0


# -----------------------------------------------------------------------------
# _connected_components
# -----------------------------------------------------------------------------

class TestConnectedComponents:
    def test_empty_mask(self) -> None:
        mask = torch.zeros((4, 4), dtype=torch.bool)
        components, largest = _connected_components(mask)
        assert components == 0
        assert largest == 0

    def test_single_component(self) -> None:
        mask = torch.zeros((4, 4), dtype=torch.bool)
        mask[0, 0:3] = True  # horizontal bar of 3
        components, largest = _connected_components(mask)
        assert components == 1
        assert largest == 3

    def test_two_disjoint_components(self) -> None:
        mask = torch.zeros((4, 4), dtype=torch.bool)
        mask[0, 0] = True
        mask[3, 3] = True
        components, largest = _connected_components(mask)
        assert components == 2
        assert largest == 1

    def test_l_shape_single_component(self) -> None:
        mask = torch.zeros((4, 4), dtype=torch.bool)
        mask[0, 0:3] = True
        mask[1:3, 0] = True  # down the left column
        components, largest = _connected_components(mask)
        assert components == 1
        assert largest == 5


# -----------------------------------------------------------------------------
# _derive_stability_scores
# -----------------------------------------------------------------------------

class TestDeriveStabilityScores:
    def test_none_returns_zeros(self) -> None:
        scores = _derive_stability_scores(None)
        assert scores["hold_score"] == 0.0
        assert scores["stability_score"] == 0.0
        assert scores["collapse_penalty"] == 0.0
        assert scores["stabilized_success"] is False
        assert scores["first_reach_step"] is None

    def test_stabilized_success_full_score(self) -> None:
        metrics = {
            "stabilized_success": True,
            "hold_length_achieved": 10,
            "hold_window_steps": 10,
            "best_iou": 0.9,
            "stabilize_threshold_iou": 0.8,
            "first_reach_step": 50,
        }
        scores = _derive_stability_scores(metrics)
        assert scores["stability_score"] == 1.0
        assert scores["hold_score"] == 1.0
        # stabilized + reached → no collapse penalty
        assert scores["collapse_penalty"] == 0.0
        assert scores["stabilized_success"] is True

    def test_partial_hold_no_success(self) -> None:
        metrics = {
            "stabilized_success": False,
            "hold_length_achieved": 3,
            "hold_window_steps": 10,
            "best_iou": 0.4,
            "stabilize_threshold_iou": 0.8,
            "first_reach_step": None,
        }
        scores = _derive_stability_scores(metrics)
        assert scores["hold_score"] == 0.3
        # base_score = 0.5, stability_score = 0.5*0.3 + 0.5*0.5 = 0.4
        assert abs(scores["stability_score"] - 0.4) < 1e-6
        assert scores["collapse_penalty"] == 0.0  # first_reach_step is None

    def test_collapse_penalty_when_reached_then_lost(self) -> None:
        metrics = {
            "stabilized_success": False,
            "hold_length_achieved": 2,
            "hold_window_steps": 10,
            "best_iou": 0.85,
            "stabilize_threshold_iou": 0.8,
            "first_reach_step": 30,
            "stability_failures": 5,
        }
        scores = _derive_stability_scores(metrics)
        assert scores["collapse_penalty"] > 0.0

    def test_zero_hold_window_falls_back_to_success_flag(self) -> None:
        metrics = {
            "stabilized_success": True,
            "hold_length_achieved": 0,
            "hold_window_steps": 0,
            "best_iou": 0.9,
            "stabilize_threshold_iou": 0.8,
            "first_reach_step": 1,
        }
        scores = _derive_stability_scores(metrics)
        assert scores["hold_score"] == 1.0


# -----------------------------------------------------------------------------
# _compute_t_diagnostics
# -----------------------------------------------------------------------------

class TestComputeTDiagnostics:
    def test_empty_target_returns_zero(self) -> None:
        ab = torch.zeros((5, 5), dtype=torch.bool)
        target = torch.zeros((5, 5), dtype=torch.bool)
        sym, trunk = _compute_t_diagnostics(ab, target)
        assert sym == 0.0 and trunk == 0.0

    def test_perfect_t_match(self) -> None:
        target = _t_target()
        sym, trunk = _compute_t_diagnostics(target.clone(), target)
        assert sym == 1.0
        assert trunk == 1.0

    def test_trunk_missing(self) -> None:
        target = _t_target()
        ab = torch.zeros_like(target)
        ab[1, 1:6] = True  # only bar present
        sym, trunk = _compute_t_diagnostics(ab, target)
        assert sym == 1.0
        assert trunk == 0.0

    def test_asymmetric_bar(self) -> None:
        target = _t_target()
        ab = torch.zeros_like(target)
        ab[1, 1:4] = True  # only half of bar
        ab[2:6, 3] = True
        sym, _ = _compute_t_diagnostics(ab, target)
        assert 0.0 <= sym < 1.0


# -----------------------------------------------------------------------------
# _compute_t_structure_breakdown
# -----------------------------------------------------------------------------

class TestComputeTStructureBreakdown:
    def test_empty_target_returns_default_breakdown(self) -> None:
        ab = torch.zeros((5, 5), dtype=torch.bool)
        target = torch.zeros((5, 5), dtype=torch.bool)
        result = _compute_t_structure_breakdown(
            ab, target, area_ratio=1.0, num_components=1.0
        )
        assert result["top_bar_coverage"] == 0.0
        assert result["trunk_coverage"] == 0.0

    def test_perfect_t_match(self) -> None:
        target = _t_target()
        ab = target.clone()
        result = _compute_t_structure_breakdown(
            ab, target, area_ratio=1.0, num_components=1.0
        )
        assert result["top_bar_coverage"] == 1.0
        assert result["trunk_coverage"] == 1.0
        assert result["top_bar_symmetry"] == 1.0
        assert result["excess_below_ratio"] == 0.0
        assert result["excess_side_ratio"] == 0.0
        assert result["area_overshoot"] == 0.0
        assert result["area_undershoot"] == 0.0

    def test_excess_below_detected(self) -> None:
        target = _t_target()
        ab = target.clone()
        ab[6, 3] = True  # extend trunk below target y_max
        result = _compute_t_structure_breakdown(
            ab, target, area_ratio=1.1, num_components=1.0
        )
        assert result["excess_below_ratio"] > 0.0

    def test_side_clutter_detected(self) -> None:
        target = _t_target()
        ab = target.clone()
        ab[3, 2] = True  # neighbour to trunk
        result = _compute_t_structure_breakdown(
            ab, target, area_ratio=1.05, num_components=1.0
        )
        assert result["side_clutter_ratio"] > 0.0

    def test_area_overshoot_undershoot(self) -> None:
        target = _t_target()
        over = _compute_t_structure_breakdown(
            target.clone(), target, area_ratio=1.5, num_components=1.0
        )
        assert over["area_overshoot"] == 0.5
        assert over["area_undershoot"] == 0.0

        under = _compute_t_structure_breakdown(
            target.clone(), target, area_ratio=0.7, num_components=1.0
        )
        assert under["area_undershoot"] > 0.0
        assert under["area_overshoot"] == 0.0

    def test_disconnected_fragments_normalised(self) -> None:
        target = _t_target()
        ab = target.clone()
        result = _compute_t_structure_breakdown(
            ab, target, area_ratio=1.0, num_components=3.0
        )
        # 3 components → fragments = (3-1)/3 ≈ 0.667
        assert abs(result["disconnected_fragments"] - 2.0 / 3.0) < 1e-6


# -----------------------------------------------------------------------------
# compute_fitness: core behaviour
# -----------------------------------------------------------------------------

class TestComputeFitnessCore:
    def _basic_square_grid_and_target(self):
        grid = _empty_grid(5, 5)
        grid[1:3, 1:3] = CELL_A  # 4 cells
        target = torch.zeros((5, 5), dtype=torch.bool)
        target[1:3, 1:3] = True
        return grid, target

    def test_perfect_square_match(self) -> None:
        grid, target = self._basic_square_grid_and_target()
        metrics = compute_fitness(grid, target)
        assert metrics["iou"] == 1.0
        assert metrics["fp"] == 0.0
        assert metrics["fn"] == 0.0
        assert metrics["base_score"] > 0.0
        assert metrics["num_components"] == 1.0

    def test_no_overlap(self) -> None:
        grid = _empty_grid(5, 5)
        grid[0, 0] = CELL_A
        target = torch.zeros((5, 5), dtype=torch.bool)
        target[4, 4] = True
        metrics = compute_fitness(grid, target)
        assert metrics["iou"] == 0.0
        assert metrics["fp"] > 0.0
        assert metrics["fn"] > 0.0

    def test_alive_bonus_and_extinction_penalty(self) -> None:
        grid, target = self._basic_square_grid_and_target()
        # alive_end=0 → extinction penalty triggered
        ext = compute_fitness(grid, target, alive_end=0.0, target_area=4.0)
        assert ext["extinction_penalty"] == 0.5
        assert ext["alive_bonus"] == 0.0

        # alive_end matching target → full alive bonus
        alive = compute_fitness(grid, target, alive_end=4.0, target_area=4.0)
        assert alive["alive_bonus"] > 0.0
        assert alive["extinction_penalty"] == 0.0

    def test_area_penalty_applied(self) -> None:
        grid, target = self._basic_square_grid_and_target()
        # target_area of 2 means ratio = 4/2 = 2 → penalty = gamma_area * |2-1|
        metrics = compute_fitness(grid, target, target_area=2.0, gamma_area=0.1)
        assert metrics["area_penalty2"] > 0.0

    def test_stem_penalty_only_when_cleanup_ratio_positive(self) -> None:
        grid = _empty_grid(10, 10)
        grid[0, 0] = CELL_A  # minimal shape
        # fill almost everything else with stem cells → well past corridor
        for y in range(10):
            for x in range(10):
                if (y, x) != (0, 0):
                    grid[y, x] = CELL_STEM
        target = torch.zeros((10, 10), dtype=torch.bool)
        target[0, 0] = True
        no_cleanup = compute_fitness(
            grid,
            target,
            target_area=1.0,
            stem_penalty_scale=1.0,
            stem_cleanup_ratio=0.0,
            stem_corridor_start_scale=1.0,
            stem_corridor_end_scale=1.0,
        )
        with_cleanup = compute_fitness(
            grid,
            target,
            target_area=1.0,
            stem_penalty_scale=1.0,
            stem_cleanup_ratio=1.0,
            stem_corridor_start_scale=1.0,
            stem_corridor_end_scale=1.0,
        )
        # Cleanup ratio of 0 disables the penalty regardless of excess.
        assert no_cleanup["stem_penalty"] == 0.0
        assert with_cleanup["stem_penalty"] > 0.0

    def test_returns_all_documented_keys(self) -> None:
        grid, target = self._basic_square_grid_and_target()
        metrics = compute_fitness(grid, target, alive_end=4.0, target_area=4.0)
        required = {
            "fitness",
            "iou",
            "area",
            "base_score",
            "alive_bonus",
            "extinction_penalty",
            "fp",
            "fn",
            "num_components",
            "largest_component_ratio",
            "stem_count",
            "component_bonus",
            "fragment_penalty",
            "com_penalty",
            "stem_penalty",
            "hold_score",
            "stability_score",
            "collapse_penalty",
            "t_phase_score",
            "t_structure_bonus",
            "late_structure_penalty",
        }
        assert required.issubset(metrics.keys())


# -----------------------------------------------------------------------------
# compute_fitness: T-phase, coverage, collapse branches
# -----------------------------------------------------------------------------

class TestComputeFitnessTPhase:
    def test_expect_t_shape_populates_breakdown(self) -> None:
        target = _t_target()
        grid = torch.full_like(target, CELL_EMPTY, dtype=torch.int64)
        grid[target] = CELL_A
        metrics = compute_fitness(grid, target, expect_t_shape=True)
        assert metrics["t_top_bar_coverage"] == 1.0
        assert metrics["t_trunk_coverage"] == 1.0
        assert metrics["t_geometry_gate"] > 0.0

    def test_benchmark_weights_produce_structure_bonus(self) -> None:
        target = _t_target()
        grid = torch.full_like(target, CELL_EMPTY, dtype=torch.int64)
        grid[target] = CELL_A
        weights = {"w_shape": 1.0, "top_bar_weight": 0.5, "trunk_weight": 0.5}
        metrics = compute_fitness(
            grid,
            target,
            expect_t_shape=True,
            t_benchmark_weights=weights,
            coverage_floor=0.5,
        )
        assert metrics["t_phase_score"] > 0.0
        assert metrics["t_structure_bonus"] > 0.0

    def test_coverage_penalty_when_below_floor(self) -> None:
        target = _t_target()
        grid = torch.full_like(target, CELL_EMPTY, dtype=torch.int64)
        grid[1, 1] = CELL_A  # only 1 cell, well below coverage floor
        metrics = compute_fitness(
            grid,
            target,
            coverage_weight=0.5,
            coverage_floor=0.8,
            coverage_penalty_weight=1.0,
        )
        assert metrics["coverage_penalty"] > 0.0

    def test_sparse_collapse_penalty(self) -> None:
        target = _t_target()
        grid = torch.full_like(target, CELL_EMPTY, dtype=torch.int64)
        grid[1, 1] = CELL_A
        metrics = compute_fitness(
            grid,
            target,
            target_area=float(target.sum().item()),
            sparse_area_floor=0.8,
            sparse_penalty_weight=2.0,
        )
        assert metrics["sparse_collapse_penalty"] > 0.0

    def test_empty_collapse_penalty_triggered(self) -> None:
        target = _t_target()
        grid = torch.full_like(target, CELL_EMPTY, dtype=torch.int64)
        # zero AB cells → both area and coverage way below floors
        metrics = compute_fitness(
            grid,
            target,
            target_area=float(target.sum().item()),
            collapse_area_floor=0.5,
            collapse_coverage_floor=0.5,
            collapse_penalty_weight=2.0,
            collapse_bonus_suppression=1.0,
        )
        assert metrics["empty_collapse_penalty"] > 0.0
        assert metrics["collapse_gate"] < 1.0

    def test_late_t_enforce_branch(self) -> None:
        target = _t_target()
        grid = torch.full_like(target, CELL_EMPTY, dtype=torch.int64)
        grid[1, 1:6] = CELL_A  # only the bar, no trunk → trunk_score ≈ 0
        metrics = compute_fitness(
            grid,
            target,
            expect_t_shape=True,
            late_t_enforce={
                "gate": 1.0,
                "penalty": 1.0,
                "trunk_weight": 1.0,
                "trunk_min": 0.8,
                "alignment_weight": 1.0,
                "alignment_min": 0.8,
                "junction_weight": 1.0,
                "junction_min": 0.8,
                "coverage_weight": 1.0,
                "coverage_min": 0.9,
                "area_weight": 1.0,
                "area_min": 0.9,
                "empty_weight": 1.0,
                "empty_floor": 0.5,
                "bar_min": 0.3,
                "pseudo_weight": 1.0,
            },
        )
        assert metrics["late_structure_penalty"] > 0.0
        assert metrics["weak_trunk_penalty"] > 0.0
        # pseudo-T: bar present but trunk missing → pseudo penalty triggers
        assert metrics["pseudo_t_penalty"] > 0.0

    def test_clean_component_and_fp_weights(self) -> None:
        grid = _empty_grid(6, 6)
        grid[0, 0] = CELL_A  # isolated fragment 1
        grid[5, 5] = CELL_A  # isolated fragment 2
        target = torch.zeros((6, 6), dtype=torch.bool)
        target[0, 0] = True
        metrics = compute_fitness(
            grid, target, clean_component_weight=1.0, clean_fp_weight=1.0
        )
        assert metrics["late_clean_penalty"] > 0.0
