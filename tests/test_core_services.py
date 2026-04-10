"""Tests for core/services/ extracted modules."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.services.cleanup import compute_late_cleanup_ratio, resolve_cleanup_ratio
from core.services.factory import build_stability_config, prepare_model, prepare_world
from core.services.stats import summarize
from core.services.visualizer import build_visualizer
from core.services.writer import build_writer


class TestComputeLateCleanupRatio:
    def test_zero_steps(self):
        assert compute_late_cleanup_ratio(0, 100) == 0.0

    def test_before_cleanup_window(self):
        # Default cleanup_start_frac=0.8, so cleanup starts at step 80
        assert compute_late_cleanup_ratio(50, 100) == 0.0

    def test_at_cleanup_start(self):
        ratio = compute_late_cleanup_ratio(80, 100)
        assert ratio > 0.0
        assert ratio < 1.0

    def test_at_end(self):
        ratio = compute_late_cleanup_ratio(100, 100)
        assert ratio == pytest.approx(1.0)

    def test_negative_planned(self):
        assert compute_late_cleanup_ratio(50, -1) == 0.0

    def test_zero_planned(self):
        assert compute_late_cleanup_ratio(50, 0) == 0.0

    def test_custom_start_frac(self):
        # Start at 50%
        assert compute_late_cleanup_ratio(40, 100, 0.5) == 0.0
        assert compute_late_cleanup_ratio(75, 100, 0.5) > 0.0


class TestResolveCleanupRatio:
    def test_from_sim_result(self):
        sim = {"late_cleanup_ratio": 0.5, "steps": 100}
        assert resolve_cleanup_ratio(sim, 100, 0.8) == 0.5

    def test_clamped_above_1(self):
        sim = {"late_cleanup_ratio": 1.5}
        assert resolve_cleanup_ratio(sim, 100, 0.8) == 1.0

    def test_clamped_below_0(self):
        sim = {"late_cleanup_ratio": -0.5}
        assert resolve_cleanup_ratio(sim, 100, 0.8) == 0.0

    def test_fallback_to_compute(self):
        sim = {"steps": 90}  # no late_cleanup_ratio key
        ratio = resolve_cleanup_ratio(sim, 100, 0.8)
        assert ratio > 0.0


class TestPrepareWorld:
    def test_default_config(self):
        world = prepare_world({}, seed=42)
        assert world.width == 15
        assert world.height == 15

    def test_custom_config(self):
        cfg = {"width": 10, "height": 10, "init_cells": 20}
        world = prepare_world(cfg, seed=0)
        assert world.width == 10
        assert world.height == 10
        stem_count = (world.grid == 1).sum().item()  # CELL_STEM = 1
        assert stem_count == 20

    def test_deterministic(self):
        w1 = prepare_world({"width": 5, "height": 5, "init_cells": 5}, seed=42)
        w2 = prepare_world({"width": 5, "height": 5, "init_cells": 5}, seed=42)
        assert torch.equal(w1.grid, w2.grid)


class TestPrepareModel:
    def test_default_config(self):
        model = prepare_model({})
        assert model.embed_dim == 32
        assert model.num_heads == 4

    def test_custom_config(self):
        model = prepare_model({"embed_dim": 16, "heads": 2})
        assert model.embed_dim == 16
        assert model.num_heads == 2

    def test_eval_mode(self):
        model = prepare_model({})
        assert not model.training


class TestBuildStabilityConfig:
    def test_enabled_by_reach(self):
        cfg = build_stability_config({"reach_threshold_iou": 0.85})
        assert cfg["enabled"] is True
        assert cfg["reach_threshold_iou"] == 0.85

    def test_disabled_by_default(self):
        cfg = build_stability_config({"reach_threshold_iou": 0.0})
        assert cfg["enabled"] is False

    def test_hold_window(self):
        cfg = build_stability_config({"hold_window_steps": 20})
        assert cfg["enabled"] is True
        assert cfg["hold_window_steps"] == 20


class TestSummarize:
    def test_empty_list(self):
        mean, perc = summarize([], 95.0)
        assert mean == 0.0
        assert perc == 0.0

    def test_single_value(self):
        mean, perc = summarize([5.0], 95.0)
        assert mean == 5.0
        assert perc == 5.0

    def test_multiple_values(self):
        mean, perc = summarize([1.0, 2.0, 3.0, 4.0, 5.0], 50.0)
        assert mean == pytest.approx(3.0)

    def test_percentile_extremes(self):
        values = list(range(1, 101))
        vals_float = [float(v) for v in values]
        _, p99 = summarize(vals_float, 99.0)
        assert p99 >= 99.0


class TestBuildWriter:
    def test_creates_run_dir_and_writer(self, tmp_path: Path) -> None:
        writer, log_dir, run_id = build_writer("unit_test", runs_root=str(tmp_path))
        try:
            assert log_dir.exists()
            assert str(log_dir).startswith(str(tmp_path))
            assert "unit_test" in run_id
            # writer is usable
            writer.add_scalar("unit/metric", 1.0, 0)
            writer.flush()
        finally:
            writer.close()

    def test_run_id_format_contains_name(self, tmp_path: Path) -> None:
        writer, _, run_id = build_writer("custom_run", runs_root=str(tmp_path))
        try:
            assert run_id.startswith("custom_run_")
        finally:
            writer.close()


class TestBuildVisualizer:
    def _target(self) -> torch.Tensor:
        return torch.zeros((4, 4), dtype=torch.bool)

    def test_force_disable_returns_none(self, capsys) -> None:
        result = build_visualizer(
            cfg={"mode": "pygame"},
            width=4,
            height=4,
            cell_size=20,
            target_mask=self._target(),
            force_disable=True,
        )
        assert result is None
        assert "Visualization disabled" in capsys.readouterr().out

    def test_unsupported_mode_returns_none(self, capsys) -> None:
        result = build_visualizer(
            cfg={"mode": "opengl"},
            width=4,
            height=4,
            cell_size=20,
            target_mask=self._target(),
            force_disable=False,
        )
        assert result is None
        assert "unsupported" in capsys.readouterr().out

    def test_default_mode_is_pygame(self, capsys) -> None:
        # force_disable=True bypasses actual pygame import but exercises the
        # mode=default branch (cfg has no "mode" key → defaults to "pygame").
        result = build_visualizer(
            cfg={},
            width=4,
            height=4,
            cell_size=20,
            target_mask=self._target(),
            force_disable=True,
        )
        assert result is None
