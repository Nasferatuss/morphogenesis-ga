from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.utils import create_run_dir, select_device, set_seed


class TestSetSeed:
    def test_torch_deterministic(self):
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)

    def test_different_seeds_differ(self):
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(99)
        t2 = torch.randn(10)
        assert not torch.equal(t1, t2)


class TestSelectDevice:
    def test_cpu_explicit(self):
        dev = select_device("cpu")
        assert dev == torch.device("cpu")

    def test_auto_returns_valid_device(self):
        dev = select_device("auto")
        assert dev.type in ("cpu", "cuda")

    def test_none_defaults(self):
        dev = select_device(None)
        assert dev.type in ("cpu", "cuda")


class TestCreateRunDir:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path, run_id = create_run_dir(tmpdir, "test_run")
            assert path.exists()
            assert path.is_dir()
            assert "test_run" in run_id

    def test_name_sanitization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, run_id = create_run_dir(tmpdir, "my run name")
            assert " " not in run_id
