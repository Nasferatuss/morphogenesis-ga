"""Tests for core/memory/mlflow_tracker.py.

Covers the soft-dependency wrapper in both environments:

- When ``mlflow`` IS NOT installed / NOT enabled: every public method
  is a silent no-op, ``is_active`` is False, and the ``start_run``
  context manager still yields cleanly.
- When ``mlflow`` IS installed and enabled: calls are forwarded to the
  mlflow client. We mock the mlflow module to verify call sequences
  without spinning up a real MLflow tracking server (which would need
  a file URI + cleanup and slows down CI).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from core.memory.mlflow_tracker import MLflowConfig, MLflowTracker


# --------------------------------------------------------------------------- #
# MLflowConfig.from_dict
# --------------------------------------------------------------------------- #


class TestMLflowConfigFromDict:
    def test_none_disabled_default(self) -> None:
        cfg = MLflowConfig.from_dict(None)
        assert cfg.enabled is False
        assert cfg.tracking_uri is None
        assert cfg.experiment_name == "morphogenesis-ga"
        assert cfg.tags == {}

    def test_empty_dict_disabled(self) -> None:
        assert MLflowConfig.from_dict({}).enabled is False

    def test_enabled_flag(self) -> None:
        assert MLflowConfig.from_dict({"enabled": True}).enabled is True

    def test_tracking_uri_parsed(self) -> None:
        cfg = MLflowConfig.from_dict(
            {"enabled": True, "tracking_uri": "file:./mlruns"}
        )
        assert cfg.tracking_uri == "file:./mlruns"

    def test_experiment_name_parsed(self) -> None:
        cfg = MLflowConfig.from_dict(
            {"enabled": True, "experiment_name": "hpo_sweep_v3"}
        )
        assert cfg.experiment_name == "hpo_sweep_v3"

    def test_tags_parsed(self) -> None:
        cfg = MLflowConfig.from_dict(
            {"enabled": True, "tags": {"kind": "smoke", "phase": "dev"}}
        )
        assert cfg.tags == {"kind": "smoke", "phase": "dev"}

    def test_tags_none_becomes_empty_dict(self) -> None:
        cfg = MLflowConfig.from_dict({"enabled": True, "tags": None})
        assert cfg.tags == {}


# --------------------------------------------------------------------------- #
# Disabled tracker (no mlflow call should ever happen)
# --------------------------------------------------------------------------- #


class TestDisabledTracker:
    def test_default_tracker_disabled(self) -> None:
        tracker = MLflowTracker()
        assert tracker.is_active is False
        assert tracker.config.enabled is False

    def test_explicit_disabled_cfg(self) -> None:
        tracker = MLflowTracker(MLflowConfig(enabled=False))
        assert tracker.is_active is False

    def test_from_cfg_with_none(self) -> None:
        tracker = MLflowTracker.from_cfg(None)
        assert tracker.is_active is False

    def test_from_cfg_with_empty_dict(self) -> None:
        tracker = MLflowTracker.from_cfg({})
        assert tracker.is_active is False

    def test_start_run_noop_is_context_manager(self) -> None:
        """start_run on a disabled tracker must still work as a with-statement."""
        tracker = MLflowTracker()
        entered = False
        with tracker.start_run(run_name="noop") as t:
            entered = True
            assert t is tracker
            assert t.is_active is False
        assert entered is True

    def test_log_methods_are_noop(self) -> None:
        """Disabled tracker accepts every log call silently."""
        tracker = MLflowTracker()
        with tracker.start_run("noop"):
            # None of these should raise
            tracker.log_params({"alpha_fp": 2.5, "beta_fn": 2.0})
            tracker.log_metric("iou", 0.42, step=3)
            tracker.log_metrics({"best_iou": 0.42, "mean_iou": 0.33}, step=3)
            tracker.log_artifact("nonexistent_file.pt")
            tracker.set_tag("kind", "test")

    def test_log_methods_noop_without_start_run(self) -> None:
        """Calls outside start_run are also silent no-ops."""
        tracker = MLflowTracker()
        tracker.log_params({"alpha_fp": 2.5})
        tracker.log_metric("iou", 0.42)


# --------------------------------------------------------------------------- #
# Enabled tracker with mocked mlflow module
# --------------------------------------------------------------------------- #


def _make_mock_mlflow() -> MagicMock:
    """Build a MagicMock that looks enough like the mlflow module."""
    mock = MagicMock()
    # start_run must act as a context manager
    mock.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock


class TestEnabledTrackerMocked:
    def _build(self, **cfg_overrides: Any) -> tuple[MLflowTracker, MagicMock]:
        """Build an enabled tracker with a mock mlflow module injected."""
        mock_mlflow = _make_mock_mlflow()
        cfg = MLflowConfig(enabled=True, **cfg_overrides)
        tracker = MLflowTracker(cfg)
        # Inject the mock after construction so _load_mlflow's real
        # import doesn't interfere.
        tracker._mlflow = mock_mlflow  # noqa: SLF001 - test-only monkeypatch
        return tracker, mock_mlflow

    def test_start_run_calls_set_experiment(self) -> None:
        tracker, mock = self._build(experiment_name="test_exp")
        with tracker.start_run(run_name="r1"):
            assert tracker.is_active is True
            mock.set_experiment.assert_called_once_with("test_exp")
            mock.start_run.assert_called_once_with(run_name="r1", nested=False)
        assert tracker.is_active is False  # cleared after exit

    def test_start_run_sets_tracking_uri_when_provided(self) -> None:
        tracker, mock = self._build(tracking_uri="file:./mlruns_test")
        with tracker.start_run("r1"):
            mock.set_tracking_uri.assert_called_once_with("file:./mlruns_test")

    def test_start_run_applies_tags(self) -> None:
        tracker, mock = self._build(tags={"kind": "smoke", "phase": "dev"})
        with tracker.start_run("r1"):
            mock.set_tags.assert_called_once_with(
                {"kind": "smoke", "phase": "dev"}
            )

    def test_log_params_forwards_to_mlflow(self) -> None:
        tracker, mock = self._build()
        with tracker.start_run("r1"):
            tracker.log_params({"alpha_fp": 2.5, "beta_fn": 2.0})
        mock.log_params.assert_called_once_with(
            {"alpha_fp": 2.5, "beta_fn": 2.0}
        )

    def test_log_params_coerces_non_scalar(self) -> None:
        tracker, mock = self._build()
        with tracker.start_run("r1"):
            tracker.log_params({"schedule": [1, 2, 3], "name": "phase_T"})
        called_with = mock.log_params.call_args[0][0]
        assert called_with["schedule"] == "[1, 2, 3]"
        assert called_with["name"] == "phase_T"

    def test_log_metric_forwards(self) -> None:
        tracker, mock = self._build()
        with tracker.start_run("r1"):
            tracker.log_metric("iou", 0.42, step=10)
        mock.log_metric.assert_called_once_with("iou", 0.42, step=10)

    def test_log_metrics_batch(self) -> None:
        tracker, mock = self._build()
        with tracker.start_run("r1"):
            tracker.log_metrics({"iou": 0.42, "fitness": -0.5}, step=5)
        mock.log_metrics.assert_called_once_with(
            {"iou": 0.42, "fitness": -0.5}, step=5
        )

    def test_log_artifact_existing_file(self, tmp_path) -> None:
        tracker, mock = self._build()
        f = tmp_path / "best.pt"
        f.write_bytes(b"fake")
        with tracker.start_run("r1"):
            tracker.log_artifact(f)
        mock.log_artifact.assert_called_once_with(str(f), artifact_path=None)

    def test_log_artifact_missing_file_skipped(self, tmp_path) -> None:
        tracker, mock = self._build()
        missing = tmp_path / "nonexistent.pt"
        with tracker.start_run("r1"):
            tracker.log_artifact(missing)
        mock.log_artifact.assert_not_called()

    def test_set_tag(self) -> None:
        tracker, mock = self._build()
        with tracker.start_run("r1"):
            tracker.set_tag("champion", "trial17")
        mock.set_tag.assert_called_once_with("champion", "trial17")

    def test_log_calls_outside_start_run_are_noop(self) -> None:
        tracker, mock = self._build()
        tracker.log_metric("iou", 0.42)
        mock.log_metric.assert_not_called()


# --------------------------------------------------------------------------- #
# Missing mlflow package (simulated via _load_mlflow returning None)
# --------------------------------------------------------------------------- #


class TestSoftDependency:
    def test_enabled_but_mlflow_missing_degrades(self) -> None:
        """If config.enabled=True but mlflow missing, tracker silently no-ops."""
        cfg = MLflowConfig(enabled=True)
        with patch(
            "core.memory.mlflow_tracker._load_mlflow", return_value=None
        ):
            tracker = MLflowTracker(cfg)
            assert tracker._mlflow is None  # noqa: SLF001
            with tracker.start_run("r1"):
                assert tracker.is_active is False
                # All log calls must still be safe
                tracker.log_metric("iou", 0.42)
                tracker.log_params({"alpha_fp": 2.5})

    def test_is_active_false_without_mlflow(self) -> None:
        cfg = MLflowConfig(enabled=True)
        with patch(
            "core.memory.mlflow_tracker._load_mlflow", return_value=None
        ):
            tracker = MLflowTracker(cfg)
            assert tracker.is_active is False
