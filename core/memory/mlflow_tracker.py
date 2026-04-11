"""MLflow experiment tracker — soft-dependency wrapper.

Notion tech stack specifies MLflow as the canonical experiment tracker
for Phase 2+. This module provides a thin wrapper around
``mlflow.set_experiment / start_run / log_metric / log_param / log_artifact``
so the rest of the codebase never imports ``mlflow`` directly.

## Soft dependency

``mlflow`` is an **optional** project dependency (see ``pyproject.toml``
``[project.optional-dependencies.tracker]``). If it's not installed,
this module's public API returns a no-op ``MLflowTracker`` instance
that silently accepts all calls. This means:

- Normal ``run_train.py`` / ``train_ga`` runs work without mlflow installed
- ``pytest`` / CI work without mlflow installed
- Adding ``logging.mlflow.enabled: true`` to a YAML config only activates
  MLflow if the package is actually available; otherwise a single warning
  is printed and training continues normally

Install via::

    pip install "mlflow>=2.15"

or equivalently::

    pip install -e ".[tracker]"

## Usage pattern

    from core.memory.mlflow_tracker import MLflowTracker

    tracker = MLflowTracker.from_cfg(cfg.get("logging", {}).get("mlflow", {}))
    with tracker.start_run(run_name="my_experiment"):
        tracker.log_params({"lr": 0.01, "pop_size": 48})
        for gen in range(30):
            tracker.log_metric("iou", best_iou, step=gen)
        tracker.log_artifact("runs/best.pt")

``start_run`` is a context manager; if MLflow is disabled or unavailable
the ``with`` block still executes but all logging is no-op.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

_LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Soft dependency detection
# --------------------------------------------------------------------------- #


def _load_mlflow() -> Optional[Any]:
    """Try to import mlflow. Return the module on success, None on failure."""
    try:
        import mlflow  # type: ignore[import-untyped]
    except Exception as err:  # pragma: no cover - env-dependent
        _LOGGER.info(
            "mlflow import failed (%s) — MLflow tracking disabled. "
            "Install with `pip install mlflow>=2.15` to enable.",
            err,
        )
        return None
    return mlflow


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MLflowConfig:
    """Parsed configuration bundle for the tracker.

    Built from a YAML ``logging.mlflow`` section. Missing / None / empty
    dict all resolve to a disabled tracker (soft-default).

    Attributes
    ----------
    enabled
        Whether to attempt MLflow logging at all. When False (default),
        the tracker is a no-op even if mlflow is installed.
    tracking_uri
        MLflow tracking server URI. ``None`` uses MLflow's default
        (``./mlruns`` directory relative to CWD).
    experiment_name
        MLflow experiment under which runs are grouped. Created
        automatically if missing. Defaults to ``"morphogenesis-ga"``.
    tags
        Run-level tags applied to every started run. Useful for
        distinguishing sweep vs smoke vs manual runs.
    """

    enabled: bool = False
    tracking_uri: Optional[str] = None
    experiment_name: str = "morphogenesis-ga"
    tags: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "MLflowConfig":
        """Build from a YAML-parsed dict (or ``None`` / empty)."""
        if not cfg:
            return cls()
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            tracking_uri=cfg.get("tracking_uri"),
            experiment_name=str(cfg.get("experiment_name", "morphogenesis-ga")),
            tags=dict(cfg.get("tags") or {}),
        )


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #


class MLflowTracker:
    """Thin wrapper around the MLflow client.

    All public methods are no-ops when either:

    - ``cfg.enabled`` is False (disabled by config)
    - ``mlflow`` is not installed in the environment
    - ``start_run`` has not yet been entered

    The ``start_run`` context manager handles run lifecycle cleanly:
    if the tracker is disabled / unavailable, ``with tracker.start_run(...)``
    still runs the body but skips all MLflow calls.
    """

    def __init__(self, cfg: Optional[MLflowConfig] = None) -> None:
        self._cfg = cfg or MLflowConfig()
        self._mlflow = _load_mlflow() if self._cfg.enabled else None
        self._active = False  # True while inside a start_run context
        if self._cfg.enabled and self._mlflow is None:
            _LOGGER.warning(
                "MLflow requested via config but package is not installed. "
                "Tracking will silently no-op. Install `mlflow>=2.15` to fix."
            )

    @classmethod
    def from_cfg(cls, cfg_dict: Optional[Dict[str, Any]]) -> "MLflowTracker":
        """Convenience: build a tracker from a raw dict (typically from YAML)."""
        return cls(MLflowConfig.from_dict(cfg_dict))

    @property
    def is_active(self) -> bool:
        """True iff MLflow is installed, enabled, AND inside a start_run context."""
        return self._active and self._mlflow is not None

    @property
    def config(self) -> MLflowConfig:
        """Return the parsed config (useful for debugging / logging)."""
        return self._cfg

    # ----- Lifecycle ---------------------------------------------------- #

    @contextlib.contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
    ) -> Iterator["MLflowTracker"]:
        """Context manager: start an MLflow run, yield self, end run cleanly.

        No-ops to a plain ``yield self`` when MLflow is unavailable or
        disabled. Errors inside the ``with`` body are NOT swallowed —
        MLflow's own context manager will mark the run as ``FAILED``
        and re-raise.
        """
        if self._mlflow is None or not self._cfg.enabled:
            # Soft-no-op: still yield self so calling code's `with` works.
            self._active = False
            yield self
            return

        try:
            if self._cfg.tracking_uri:
                self._mlflow.set_tracking_uri(self._cfg.tracking_uri)
            self._mlflow.set_experiment(self._cfg.experiment_name)
            with self._mlflow.start_run(run_name=run_name, nested=nested):
                if self._cfg.tags:
                    self._mlflow.set_tags(self._cfg.tags)
                self._active = True
                try:
                    yield self
                finally:
                    self._active = False
        except Exception as err:  # pragma: no cover - env-dependent
            _LOGGER.error(
                "MLflow start_run failed (%s) — degrading to no-op for this run.",
                err,
            )
            self._active = False
            yield self

    # ----- Logging calls ------------------------------------------------ #

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a batch of run parameters (hyperparameters)."""
        if not self.is_active:
            return
        try:
            # Coerce non-JSON-native types to str to avoid MLflow rejections
            safe = {k: self._coerce_param(v) for k, v in params.items()}
            self._mlflow.log_params(safe)  # type: ignore[union-attr]
        except Exception as err:  # pragma: no cover
            _LOGGER.warning("MLflow log_params failed: %s", err)

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None
    ) -> None:
        """Log a single metric value, optionally at a specific step."""
        if not self.is_active:
            return
        try:
            self._mlflow.log_metric(key, float(value), step=step)  # type: ignore[union-attr]
        except Exception as err:  # pragma: no cover
            _LOGGER.warning("MLflow log_metric(%s) failed: %s", key, err)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log a batch of metrics at a single step (cheaper than N log_metric)."""
        if not self.is_active:
            return
        try:
            safe = {k: float(v) for k, v in metrics.items()}
            self._mlflow.log_metrics(safe, step=step)  # type: ignore[union-attr]
        except Exception as err:  # pragma: no cover
            _LOGGER.warning("MLflow log_metrics failed: %s", err)

    def log_artifact(
        self, path: Any, artifact_path: Optional[str] = None
    ) -> None:
        """Log a single file or directory as an artifact.

        ``path`` may be ``str`` or ``pathlib.Path``. Missing files are
        silently skipped (useful when logging optional outputs).
        """
        if not self.is_active:
            return
        file_path = Path(path)
        if not file_path.exists():
            _LOGGER.debug("MLflow log_artifact skipped — missing: %s", file_path)
            return
        try:
            self._mlflow.log_artifact(str(file_path), artifact_path=artifact_path)  # type: ignore[union-attr]
        except Exception as err:  # pragma: no cover
            _LOGGER.warning("MLflow log_artifact(%s) failed: %s", file_path, err)

    def set_tag(self, key: str, value: str) -> None:
        """Set a single run tag."""
        if not self.is_active:
            return
        try:
            self._mlflow.set_tag(key, str(value))  # type: ignore[union-attr]
        except Exception as err:  # pragma: no cover
            _LOGGER.warning("MLflow set_tag(%s) failed: %s", key, err)

    # ----- Internal helpers --------------------------------------------- #

    @staticmethod
    def _coerce_param(value: Any) -> Any:
        """Coerce non-scalar params to a string so MLflow won't reject them."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
