"""Tests for core/ extracted modules."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.memory.metrics_logger import (
    export_tensorboard_plots,
    write_generations_csv,
    write_run_summary,
)
from core.services.config_loader import load_config, validate_config


class TestLoadConfig:
    def test_load_valid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("world:\n  height: 15\n  width: 15\n")
            f.flush()
            cfg = load_config(f.name)
        assert cfg["world"]["height"] == 15

    def test_load_baseline_config(self):
        cfg = load_config("configs/baseline_T.yaml")
        assert "world" in cfg
        assert "ga" in cfg
        assert "fitness" in cfg


class TestValidateConfig:
    def test_valid_config(self):
        cfg = load_config("configs/baseline_T.yaml")
        validate_config(cfg)  # should not raise

    def test_missing_section(self):
        with pytest.raises(ValueError, match="missing required"):
            validate_config({"world": {}})

    def test_invalid_population_size(self):
        cfg = load_config("configs/baseline_T.yaml")
        cfg["ga"]["population_size"] = 2
        with pytest.raises(ValueError, match="population_size"):
            validate_config(cfg)


class TestWriteGenerationsCsv:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            records = [
                {"generation": 0, "fitness": 0.1, "iou": 0.05},
                {"generation": 1, "fitness": 0.2, "iou": 0.10},
            ]
            write_generations_csv(path, records)
            content = path.read_text()
            assert "generation" in content
            assert "0.1" in content

    def test_empty_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.csv"
            write_generations_csv(path, [])
            assert not path.exists()


class TestWriteRunSummary:
    def test_write_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            write_run_summary(path, {"status": "complete", "best_iou": 0.85})
            content = path.read_text()
            assert '"status": "complete"' in content


class TestExportTensorboardPlots:
    def test_missing_directory_returns_silently(self, tmp_path: Path) -> None:
        # Directory does not exist → early return, no raise.
        export_tensorboard_plots(tmp_path / "nope", ("fitness/best",))

    def test_empty_directory_warns_and_returns(self, tmp_path: Path, capsys) -> None:
        export_tensorboard_plots(tmp_path, ("fitness/best",))
        captured = capsys.readouterr()
        assert "No TensorBoard event files" in captured.out

    def test_exports_png_for_each_existing_tag(self, tmp_path: Path) -> None:
        pytest.importorskip("tensorboard")
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tmp_path))
        for step in range(3):
            writer.add_scalar("fitness/best", 0.1 * step, step)
            writer.add_scalar("iou/best", 0.05 * step, step)
        writer.flush()
        writer.close()

        export_tensorboard_plots(tmp_path, ("fitness/best", "iou/best", "missing/tag"))

        assert (tmp_path / "fitness_best.png").exists()
        assert (tmp_path / "iou_best.png").exists()
        # missing tag produces no file
        assert not (tmp_path / "missing_tag.png").exists()
