"""Tests for src/simulate.py.

Covers the pure helpers (``_normalize_die_cap_schedule``, ``_current_die_cap``,
``_late_cleanup_ratio``) and the main ``simulate`` function. The simulate tests
use tiny deterministic "constant-action" models so we can drive specific code
paths (extinction, anti-extinction safeguard, stability tracking, die-cap
throttling) without pulling in a trained checkpoint.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from src.model import NUM_ACTIONS, NUM_NEIGHBORS
from src.simulate import (
    _current_die_cap,
    _late_cleanup_ratio,
    _normalize_die_cap_schedule,
    simulate,
)
from src.world import ACTION_BECOME_A, ACTION_DIE, ACTION_DIVIDE, ACTION_STAY, World


# -----------------------------------------------------------------------------
# Helper models — deterministic, always emit one action
# -----------------------------------------------------------------------------


class _ConstantActionModel(nn.Module):
    """Outputs logits with a single hot index at ``action``."""

    def __init__(self, action: int) -> None:
        super().__init__()
        self.action = action

    def forward(self, neighbor_tokens: torch.Tensor) -> torch.Tensor:
        batch = neighbor_tokens.shape[0]
        logits = torch.full((batch, NUM_ACTIONS), -10.0)
        logits[:, self.action] = 10.0
        return logits


def _tiny_world(size: int = 4, cells: int = 3, seed: int = 0) -> World:
    w = World(width=size, height=size, init_cells=cells, seed=seed)
    w.reset()
    return w


def _target(size: int = 4) -> torch.Tensor:
    mask = torch.zeros((size, size), dtype=torch.bool)
    mask[0, 0] = True
    return mask


# -----------------------------------------------------------------------------
# _normalize_die_cap_schedule
# -----------------------------------------------------------------------------


class TestNormalizeDieCapSchedule:
    def test_empty_and_no_fallback(self) -> None:
        assert _normalize_die_cap_schedule(None, None) == []
        assert _normalize_die_cap_schedule([], None) == []

    def test_only_fallback_cap(self) -> None:
        out = _normalize_die_cap_schedule(None, 0.25)
        assert len(out) == 1
        until, cap = out[0]
        assert cap == 0.25
        assert until >= 1_000_000  # sentinel for "forever"

    def test_fallback_cap_clamped(self) -> None:
        assert _normalize_die_cap_schedule(None, 1.5)[0][1] == 1.0
        assert _normalize_die_cap_schedule(None, -0.2)[0][1] == 0.0

    def test_valid_schedule_sorted(self) -> None:
        schedule = [
            {"until_step": 50, "cap_frac": 0.1},
            {"until_step": 10, "cap_frac": 0.5},
        ]
        out = _normalize_die_cap_schedule(schedule, None)
        assert out == [(10, 0.5), (50, 0.1)]

    def test_invalid_entries_skipped(self) -> None:
        schedule = [
            "not-a-dict",
            {"until_step": "bad", "cap_frac": 0.2},
            {"until_step": 10, "cap_frac": "bad"},
            {"until_step": 5},  # missing cap
            {"cap_frac": 0.3},  # missing until
            {"until_step": 20, "cap_frac": 0.4},  # valid
        ]
        out = _normalize_die_cap_schedule(schedule, None)
        assert out == [(20, 0.4)]

    def test_valid_schedule_overrides_fallback(self) -> None:
        schedule = [{"until_step": 10, "cap_frac": 0.3}]
        out = _normalize_die_cap_schedule(schedule, 0.9)
        assert out == [(10, 0.3)]


# -----------------------------------------------------------------------------
# _current_die_cap
# -----------------------------------------------------------------------------


class TestCurrentDieCap:
    def test_empty_schedule_returns_none(self) -> None:
        assert _current_die_cap([], 5) is None

    def test_picks_first_matching_step(self) -> None:
        schedule = [(10, 0.5), (20, 0.1)]
        assert _current_die_cap(schedule, 5) == 0.5
        assert _current_die_cap(schedule, 10) == 0.5
        assert _current_die_cap(schedule, 15) == 0.1

    def test_beyond_last_returns_last(self) -> None:
        schedule = [(10, 0.5), (20, 0.1)]
        assert _current_die_cap(schedule, 999) == 0.1


# -----------------------------------------------------------------------------
# _late_cleanup_ratio
# -----------------------------------------------------------------------------


class TestLateCleanupRatio:
    def test_zero_planned_steps(self) -> None:
        assert _late_cleanup_ratio(10, 0, 0.8) == 0.0

    def test_before_cleanup_window(self) -> None:
        assert _late_cleanup_ratio(5, 100, 0.8) == 0.0

    def test_at_start_of_window(self) -> None:
        # start = 100 * 0.8 = 80; at step 80 → (80-80+1)/(100-80+1) = 1/21
        ratio = _late_cleanup_ratio(80, 100, 0.8)
        assert 0.0 < ratio < 0.1

    def test_at_end_of_window(self) -> None:
        assert _late_cleanup_ratio(100, 100, 0.8) == 1.0

    def test_beyond_window_clamped(self) -> None:
        assert _late_cleanup_ratio(200, 100, 0.8) == 1.0

    def test_out_of_range_start_frac_clamped(self) -> None:
        # start_frac > 1.0 should clamp to 1.0 → cleanup_start = planned_steps
        assert _late_cleanup_ratio(100, 100, 1.5) == 1.0


# -----------------------------------------------------------------------------
# simulate — fast path (steps=0)
# -----------------------------------------------------------------------------


class TestSimulateFastPath:
    def test_steps_zero_returns_initial_state(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=0,
            writer=None,
            viz=None,
            render_every=10,
            device=torch.device("cpu"),
            verbose=True,  # exercise print branch
        )
        assert result["steps"] == 0
        assert result["total_divisions"] == 0
        assert result["total_deaths"] == 0
        assert result["anti_extinction_triggers"] == 0
        assert result["alive_end"] == world.count_stem_cells()

    def test_steps_zero_with_stability_cfg(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=0,
            writer=None,
            viz=None,
            render_every=10,
            device=torch.device("cpu"),
            stability_cfg={
                "enabled": True,
                "reach_threshold_iou": 0.0,
                "stabilize_threshold_iou": 0.0,
                "hold_window_steps": 0,
            },
        )
        # reach_threshold=0 so first_reach is the very first step
        assert result["first_reach_step"] == 0


# -----------------------------------------------------------------------------
# simulate — main loop
# -----------------------------------------------------------------------------


class TestSimulateMainLoop:
    def test_all_stay_runs_full_length(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=3,
            writer=None,
            viz=None,
            render_every=1,
            device=torch.device("cpu"),
        )
        assert result["steps"] == 3
        assert result["total_deaths"] == 0

    def test_all_die_post_warmup_triggers_early_exit(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_DIE)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=5,
            writer=None,
            viz=None,
            render_every=10,
            device=torch.device("cpu"),
            anti_extinction_warmup_steps=0,  # no warmup → straight to extinction
            verbose=True,
        )
        # extinct at step 1 → steps_done = 1
        assert result["steps"] == 1
        assert result["alive_end"] == 0

    def test_anti_extinction_safeguard_triggers_in_warmup(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_DIE)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=3,
            writer=None,
            viz=None,
            render_every=10,
            device=torch.device("cpu"),
            anti_extinction_warmup_steps=5,  # stays in warmup whole sim
            verbose=True,
        )
        assert result["anti_extinction_triggers"] >= 1

    def test_divide_action_increases_cell_count(self) -> None:
        world = _tiny_world(size=6, cells=2)
        target = _target(size=6)
        model = _ConstantActionModel(ACTION_DIVIDE)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=2,
            writer=None,
            viz=None,
            render_every=1,
            device=torch.device("cpu"),
        )
        assert result["total_divisions"] > 0

    def test_die_cap_throttles_deaths(self) -> None:
        world = _tiny_world(size=6, cells=6)
        target = _target(size=6)
        model = _ConstantActionModel(ACTION_DIE)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=5,
            writer=None,
            viz=None,
            render_every=10,
            device=torch.device("cpu"),
            anti_extinction_warmup_steps=0,
            die_cap_schedule=[{"until_step": 100, "cap_frac": 0.1}],
        )
        # With cap=0.1 and 6 stems, only 0-1 cells may die per step — survives longer
        assert result["steps"] >= 2

    def test_no_stems_from_start_exits_early(self) -> None:
        world = World(width=4, height=4, init_cells=0, seed=0)
        world.reset()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=5,
            writer=None,
            viz=None,
            render_every=1,
            device=torch.device("cpu"),
            verbose=True,
        )
        assert result["steps"] == 1  # immediate exit


# -----------------------------------------------------------------------------
# simulate — stability tracking
# -----------------------------------------------------------------------------


class TestSimulateStability:
    def test_stability_tracked_when_cfg_enabled(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=3,
            writer=None,
            viz=None,
            render_every=1,
            device=torch.device("cpu"),
            stability_cfg={
                "enabled": True,
                "reach_threshold_iou": 0.0,  # always reached
                "stabilize_threshold_iou": 0.0,
                "hold_window_steps": 2,
            },
        )
        assert result["first_reach_step"] == 0
        assert result["hold_length_achieved"] >= 2
        assert result["stabilized_success"] is True

    def test_stability_hold_window_zero_success_on_reach(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=2,
            writer=None,
            viz=None,
            render_every=1,
            device=torch.device("cpu"),
            stability_cfg={"enabled": True, "reach_threshold_iou": 0.0},
        )
        assert result["stabilized_success"] is True


# -----------------------------------------------------------------------------
# simulate — instrumentation (writer, viz, frame_capture)
# -----------------------------------------------------------------------------


class TestSimulateInstrumentation:
    def test_writer_receives_scalars(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        writer = MagicMock()
        simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=2,
            writer=writer,
            viz=None,
            render_every=1,
            device=torch.device("cpu"),
        )
        # add_scalar called for initial state + each step
        assert writer.add_scalar.called

    def test_viz_receives_render_calls(self) -> None:
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        viz = MagicMock()
        simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=2,
            writer=None,
            viz=viz,
            render_every=1,
            device=torch.device("cpu"),
        )
        assert viz.render.called

    def test_frame_capture_exception_disables_capture(self) -> None:
        captured: List[int] = []

        def bad_capture(grid: torch.Tensor, step: int) -> None:
            captured.append(step)
            raise RuntimeError("boom")

        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=3,
            writer=None,
            viz=None,
            render_every=1,
            device=torch.device("cpu"),
            frame_capture=bad_capture,
            capture_every=1,
        )
        # First capture fails → frame_capture disabled → only one call
        assert len(captured) == 1

    def test_frame_capture_called_at_stride(self) -> None:
        captured: List[int] = []
        world = _tiny_world()
        target = _target()
        model = _ConstantActionModel(ACTION_STAY)
        simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=4,
            writer=None,
            viz=None,
            render_every=10,
            device=torch.device("cpu"),
            frame_capture=lambda g, s: captured.append(s),
            capture_every=2,
        )
        # step 0 initial, then 2, 4 → 3 calls minimum
        assert 0 in captured
        assert 2 in captured
