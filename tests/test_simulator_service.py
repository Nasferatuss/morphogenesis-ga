"""Tests for core/services/simulator.py.

Covers the ``run_single_simulation`` entry point with a tiny world, a
``mode='off'`` (unsupported) visualiser config to skip the pygame path, and
a zero-step fast path plus a short multi-step run.
"""
from __future__ import annotations

import argparse

import pytest
import torch

from core.services.simulator import run_single_simulation


def _baseline_cfg() -> dict:
    return {
        "seed": 7,
        "world": {"width": 5, "height": 5, "init_cells": 3, "steps": 0},
        "model": {"embed_dim": 16, "heads": 2},
        "viz": {"mode": "off", "render_every_steps": 10, "cell_px": 4, "fps": 5},
        "simulate": {
            "anti_extinction_warmup_steps": 0,
            "late_cleanup_start_frac": 0.8,
        },
        "fitness": {
            "alpha_fp": 1.0,
            "beta_fn": 0.5,
            "gamma_area": 0.1,
            "empty_collapse": {},
            "coverage": {},
        },
        "benchmark": {},
        "target": {"name": "square"},
        "logging": {"run_name": "unit_sim"},
    }


def _target(size: int = 5) -> torch.Tensor:
    mask = torch.zeros((size, size), dtype=torch.bool)
    mask[1:3, 1:3] = True
    return mask


class TestRunSingleSimulation:
    def test_zero_steps_fast_path(self, tmp_path, monkeypatch, capsys) -> None:
        # Redirect runs/ into tmp so we don't leave artefacts behind.
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(no_viz=True, steps=0)
        run_single_simulation(
            _baseline_cfg(), args, torch.device("cpu"), _target(), seed=7
        )
        output = capsys.readouterr().out
        assert "[RESULT] IoU=" in output
        assert "[RESULT] Divisions=" in output

    def test_override_steps_prints_info(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(no_viz=True, steps=2)
        run_single_simulation(
            _baseline_cfg(), args, torch.device("cpu"), _target(), seed=7
        )
        out = capsys.readouterr().out
        assert "Overriding steps" in out
        assert "[RESULT] IoU=" in out

    def test_t_target_branch(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _baseline_cfg()
        cfg["target"]["name"] = "T"
        cfg["fitness"]["t_phase_weights"] = {"w_shape": 1.0}
        args = argparse.Namespace(no_viz=True, steps=1)
        run_single_simulation(cfg, args, torch.device("cpu"), _target(), seed=7)
        assert "[RESULT] IoU=" in capsys.readouterr().out

    def test_stability_enabled_path(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _baseline_cfg()
        cfg["benchmark"] = {
            "enabled": True,
            "reach_threshold_iou": 0.0,
            "stabilize_threshold_iou": 0.0,
            "hold_window_steps": 1,
        }
        args = argparse.Namespace(no_viz=True, steps=2)
        run_single_simulation(cfg, args, torch.device("cpu"), _target(), seed=7)
        assert "[RESULT] IoU=" in capsys.readouterr().out
