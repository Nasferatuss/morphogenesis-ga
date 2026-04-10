"""Tests for agents/eval_agent/pipeline.py.

Covers the ``play_best`` entry point with a real checkpoint, a headless
(``mode='off'``) visualiser config, and a tiny world so the full replay path
runs in milliseconds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from agents.eval_agent.pipeline import play_best
from src.model import LittleLM
from src.utils import set_seed


def _make_checkpoint(tmp_path: Path) -> Path:
    """Save a small randomly-initialised LittleLM checkpoint."""
    set_seed(7)
    model = LittleLM(embed_dim=16, num_heads=2)
    model.eval()
    path = tmp_path / "best.pt"
    torch.save(model.state_dict(), path)
    return path


def _baseline_cfg() -> dict:
    return {
        "world": {"width": 5, "height": 5, "init_cells": 3, "steps": 3},
        "model": {"embed_dim": 16, "heads": 2},
        "viz": {"mode": "off", "render_every_steps": 10, "cell_px": 4, "fps": 5},
        "simulate": {
            "anti_extinction_warmup_steps": 0,
            "late_cleanup_start_frac": 0.8,
        },
        "ga": {"eval_steps": 3},
        "fitness": {
            "alpha_fp": 1.0,
            "beta_fn": 0.5,
            "gamma_area": 0.1,
            "empty_collapse": {},
            "coverage": {},
        },
        "benchmark": {},
        "target": {"name": "square"},
    }


def _target() -> torch.Tensor:
    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[1:3, 1:3] = True
    return mask


class TestPlayBest:
    def test_raises_when_no_viz_flag_set(self, tmp_path: Path) -> None:
        ckpt = _make_checkpoint(tmp_path)
        args = argparse.Namespace(no_viz=True, play_best=str(ckpt))
        with pytest.raises(RuntimeError, match="requires visualization"):
            play_best(_baseline_cfg(), args, torch.device("cpu"), _target(), seed=0)

    def test_raises_when_checkpoint_missing(self, tmp_path: Path) -> None:
        args = argparse.Namespace(no_viz=False, play_best=str(tmp_path / "missing.pt"))
        with pytest.raises(FileNotFoundError):
            play_best(_baseline_cfg(), args, torch.device("cpu"), _target(), seed=0)

    def test_runs_end_to_end_with_headless_viz(self, tmp_path: Path, capsys) -> None:
        ckpt = _make_checkpoint(tmp_path)
        args = argparse.Namespace(no_viz=False, play_best=str(ckpt))
        play_best(_baseline_cfg(), args, torch.device("cpu"), _target(), seed=0)
        output = capsys.readouterr().out
        # Headless viz warns, then final [PLAY] result line is emitted.
        assert "[PLAY] IoU=" in output

    def test_t_target_branch(self, tmp_path: Path, capsys) -> None:
        ckpt = _make_checkpoint(tmp_path)
        cfg = _baseline_cfg()
        cfg["target"]["name"] = "T"
        cfg["fitness"]["t_phase_weights"] = {"w_shape": 1.0}
        args = argparse.Namespace(no_viz=False, play_best=str(ckpt))
        play_best(cfg, args, torch.device("cpu"), _target(), seed=0)
        assert "[PLAY] IoU=" in capsys.readouterr().out

    def test_gif_recorder_path(self, tmp_path: Path) -> None:
        ckpt = _make_checkpoint(tmp_path)
        cfg = _baseline_cfg()
        cfg["viz"]["gif"] = {
            "enabled": True,
            "frame_every_steps": 1,
            "max_frames": 10,
        }
        args = argparse.Namespace(no_viz=False, play_best=str(ckpt))
        play_best(cfg, args, torch.device("cpu"), _target(), seed=0)
        gif_path = tmp_path / "gifs" / "play_best.gif"
        assert gif_path.exists()
