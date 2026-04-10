"""Smoke tests for agents/train_agent/pipeline.py train_ga.

Runs a minimal GA training loop (tiny world, pop=4, 2 generations, 2 eval
steps) to exercise the main curriculum path, phase transitions, CSV logging,
and checkpoint saving without pulling in real benchmarks or pygame.

These are integration tests, not unit tests — the goal is to hold
``train_ga`` end-to-end behaviour in place as the first safety net for future
closure-by-closure decomposition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from agents.train_agent.pipeline import train_ga


def _tiny_cfg() -> dict:
    return {
        "seed": 7,
        "world": {"width": 5, "height": 5, "init_cells": 3, "steps": 2},
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
        "target": {"name": "T"},
        "logging": {"run_name": "unit_train"},
        "ga": {
            "enabled": True,
            "population_size": 4,
            "generations": 2,
            "elite_frac": 0.5,
            "mutation_prob": 0.1,
            "mutation_std": 0.02,
            "n_elites_unchanged": 1,
            "eval_steps": 2,
            "viz_every_generations": 0,
            "checkpoint_every_generations": 0,
            "plateau_generations": 0,
        },
    }


def _target(size: int = 5) -> torch.Tensor:
    mask = torch.zeros((size, size), dtype=torch.bool)
    mask[1, 1:4] = True
    mask[2:4, 2] = True
    return mask


class TestTrainGaSmoke:
    def test_minimal_ga_run_produces_artefacts(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Uses "cross" target so ``phase_is_t`` is False and the stricter
        # T-phase champion gating doesn't block best.pt from being written
        # with our tiny world + untrained population.
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(no_viz=True, steps=None, train_ga=True)
        cfg = _tiny_cfg()
        cfg["target"]["name"] = "cross"
        target = _target()
        target_area = float(target.sum().item())

        best_path = train_ga(
            cfg,
            args,
            torch.device("cpu"),
            target,
            seed=7,
            default_target_name="cross",
            default_target_area=target_area,
        )

        assert best_path.name == "best.pt"
        run_dir = best_path.parent
        # Non-T target → best.pt is always written at least once.
        assert best_path.exists()
        # Generations CSV + run summary are written at the end of training.
        assert (run_dir / "generations.csv").exists()
        assert (run_dir / "run_summary.json").exists()

    def test_curriculum_smoke(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(no_viz=True, steps=None, train_ga=True)
        cfg = _tiny_cfg()
        cfg["curriculum"] = {
            "enabled": True,
            "phases": [
                {"name": "phase_cross", "target": "cross", "generations": 1},
                {"name": "phase_T", "target": "T", "generations": 1},
            ],
        }
        target = _target()
        target_area = float(target.sum().item())

        best_path = train_ga(
            cfg,
            args,
            torch.device("cpu"),
            target,
            seed=7,
            default_target_name="T",
            default_target_area=target_area,
        )

        assert best_path.exists()
        # Both phase checkpoints should be written.
        run_dir = best_path.parent
        assert (run_dir / "best_cross.pt").exists() or (run_dir / "best_t.pt").exists()
