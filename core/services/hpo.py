"""Hyperparameter optimization service (Optuna-backed).

Library module used by the ``scripts/hpo_sweep.py`` CLI. Contains the
pieces that can be unit-tested without actually launching Optuna:
the search space loader, the dotted-path param setter, the objective
extractor, and the single-trial runner.

The CLI itself (``scripts/hpo_sweep.py``) is a thin argparse wrapper
that creates an ``optuna.Study`` and calls ``run_trial`` in a closure.

Why this lives in ``core/services/``: HPO is a form of calibration —
the Notion roadmap target layout explicitly mentions
``core/services/calibration_service.py``. This module is that slot
under a more descriptive name until the full calibration workflow
lands.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

# Optuna is imported lazily inside ``suggest_params`` so the module
# stays importable (and testable) even in environments without optuna.


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """Single-parameter search spec parsed from YAML."""

    path: str
    kind: str  # "float" | "int" | "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        if self.kind not in {"float", "int", "categorical"}:
            raise ValueError(
                f"ParamSpec {self.path}: unknown kind {self.kind!r}"
            )
        if self.kind in {"float", "int"}:
            if self.low is None or self.high is None:
                raise ValueError(
                    f"ParamSpec {self.path}: {self.kind} needs low + high"
                )
            if self.low >= self.high:
                raise ValueError(
                    f"ParamSpec {self.path}: low {self.low} >= high {self.high}"
                )
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(
                    f"ParamSpec {self.path}: categorical needs non-empty choices"
                )


def load_search_space(path: Path) -> List[ParamSpec]:
    """Parse a search space YAML into an ordered list of ``ParamSpec``.

    The YAML must contain a top-level ``parameters`` dict mapping
    dotted-path strings to per-parameter spec dicts. See
    ``configs/hpo/search_space_v1.yaml`` for the authoritative example.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "parameters" not in raw:
        raise ValueError(
            f"Search space {path}: top-level 'parameters' dict missing"
        )
    specs: List[ParamSpec] = []
    for param_path, spec_dict in raw["parameters"].items():
        if not isinstance(spec_dict, dict):
            raise ValueError(f"Parameter {param_path}: spec must be a dict")
        kind = str(spec_dict.get("type", ""))
        specs.append(
            ParamSpec(
                path=str(param_path),
                kind=kind,
                low=spec_dict.get("low"),
                high=spec_dict.get("high"),
                log=bool(spec_dict.get("log", False)),
                choices=spec_dict.get("choices"),
            )
        )
    return specs


def suggest_params(trial: Any, specs: List[ParamSpec]) -> Dict[str, Any]:
    """Call the appropriate ``trial.suggest_*`` for each spec and
    return a ``{dotted_path: value}`` mapping.

    ``trial`` is typed ``Any`` so the module stays importable without
    optuna; at runtime it is an ``optuna.Trial`` instance.
    """
    params: Dict[str, Any] = {}
    for spec in specs:
        if spec.kind == "float":
            params[spec.path] = trial.suggest_float(
                spec.path, float(spec.low), float(spec.high), log=spec.log
            )
        elif spec.kind == "int":
            params[spec.path] = trial.suggest_int(
                spec.path, int(spec.low), int(spec.high)
            )
        elif spec.kind == "categorical":
            params[spec.path] = trial.suggest_categorical(
                spec.path, spec.choices or []
            )
    return params


# ---------------------------------------------------------------------------
# Dotted-path config merge
# ---------------------------------------------------------------------------


def apply_params(cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of ``cfg`` with ``params`` applied at dotted paths.

    Missing intermediate dicts are auto-created so adaptive_mutation.*
    works even on a base config that doesn't have that section yet.
    """
    merged = copy.deepcopy(cfg)
    for path, value in params.items():
        keys = path.split(".")
        node: Dict[str, Any] = merged
        for key in keys[:-1]:
            existing = node.get(key)
            if not isinstance(existing, dict):
                existing = {}
                node[key] = existing
            node = existing
        node[keys[-1]] = value
    return merged


# ---------------------------------------------------------------------------
# Objective extraction
# ---------------------------------------------------------------------------


VALID_OBJECTIVES = {
    "best_iou",
    "final_best_iou",
    "best_fitness",
    "composite",
    "multi_seed_iou",
}

# Objectives that need the best.pt checkpoint and post-training eval.
# The rest read from ``generations.csv`` only.
CHECKPOINT_OBJECTIVES = {"multi_seed_iou"}


def compute_objective(run_dir: Path, objective: str = "best_iou") -> float:
    """Read ``generations.csv`` from ``run_dir`` and compute the scalar.

    Objectives:
    - ``best_iou``: max ``best_iou`` across all generations
    - ``final_best_iou``: ``best_iou`` of the last generation
    - ``best_fitness``: max ``best_fitness`` across all generations
    - ``composite``: max best_iou + 0.5 × max stability_score

    Note: ``multi_seed_iou`` does NOT go through this function — it needs
    the best.pt checkpoint, not just the CSV. Use
    ``compute_multi_seed_objective`` for that case.

    Returns ``float("-inf")`` if the run failed (no CSV or no rows) —
    Optuna will treat that as the worst possible trial.
    """
    if objective not in VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective {objective!r}; valid: {sorted(VALID_OBJECTIVES)}"
        )
    if objective in CHECKPOINT_OBJECTIVES:
        raise ValueError(
            f"Objective {objective!r} requires compute_multi_seed_objective(), "
            f"not compute_objective()."
        )
    csv_path = run_dir / "generations.csv"
    if not csv_path.exists():
        return float("-inf")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("-inf")

    if objective == "best_iou":
        return max(float(r.get("best_iou", 0.0) or 0.0) for r in rows)
    if objective == "final_best_iou":
        return float(rows[-1].get("best_iou", 0.0) or 0.0)
    if objective == "best_fitness":
        return max(float(r.get("best_fitness", 0.0) or 0.0) for r in rows)
    # composite
    best_iou = max(float(r.get("best_iou", 0.0) or 0.0) for r in rows)
    best_stab = max(
        float(r.get("stability_score", 0.0) or 0.0) for r in rows
    )
    return best_iou + 0.5 * best_stab


def compute_multi_seed_objective(
    best_pt_path: Path,
    base_cfg: Dict[str, Any],
    *,
    n_seeds: int = 5,
    device: Any = None,
) -> float:
    """Load a trained checkpoint and benchmark across ``n_seeds`` world seeds.

    Returns the mean ``best_iou`` across seeds — the same metric used by
    ``scripts/multiseed_bench.py`` and by ``docs/leaderboard.md``. This is the
    HANDOFF-recommended "gold standard" objective that avoids training-peak
    overfitting.

    Fallback chain: if the canonical ``best.pt`` is missing (some trials
    never pass train_ga's champion-gating on T phase), try in order:
    ``best_t_refine.pt`` → ``best_t.pt`` → ``best_transition.pt`` →
    ``best_cross.pt`` → ``latest_candidate.pt``. This keeps weak trials
    scored instead of collapsing to -inf (which would blind TPE).

    Uses seeds ``0..n_seeds-1`` (matches ``scripts/multiseed_bench.py``).

    Returns ``float("-inf")`` only on genuine failure (no checkpoints at all,
    load error, simulate crash).
    """
    # Fallback chain for the case where train_ga's champion gating didn't
    # fire (trial config is weak, best.pt not saved but phase-best.pt exists).
    candidate_paths = [best_pt_path]
    parent = best_pt_path.parent
    for fallback_name in (
        "best_t_refine.pt",
        "best_t.pt",
        "best_transition.pt",
        "best_cross.pt",
        "latest_candidate.pt",
    ):
        candidate_paths.append(parent / fallback_name)

    checkpoint_path: Optional[Path] = None
    for candidate in candidate_paths:
        if candidate.exists():
            checkpoint_path = candidate
            break
    if checkpoint_path is None:
        return float("-inf")

    try:
        from scripts.multiseed_bench import (  # noqa: PLC0415
            load_model_auto,
            run_benchmark,
        )
    except Exception as err:  # pragma: no cover - diagnostics
        print(f"[HPO] multi_seed_iou import failed: {err}")
        return float("-inf")

    world_cfg = base_cfg.get("world", {})
    model_cfg = base_cfg.get("model", {})
    simulate_cfg = base_cfg.get("simulate", {})
    target_cfg = base_cfg.get("target", {})

    width = int(world_cfg.get("width", 15))
    height = int(world_cfg.get("height", 15))
    init_cells = int(world_cfg.get("init_cells", 50))
    steps = int(world_cfg.get("steps", 200))
    warmup = int(simulate_cfg.get("anti_extinction_warmup_steps", 25))
    late_cleanup = float(simulate_cfg.get("late_cleanup_start_frac", 0.75))
    target_name = str(target_cfg.get("name", "T"))
    embed_dim = int(model_cfg.get("embed_dim", 32))
    num_heads = int(model_cfg.get("heads", 4))

    try:
        model = load_model_auto(checkpoint_path, embed_dim, num_heads)
    except Exception as err:  # pragma: no cover
        print(f"[HPO] multi_seed_iou load failed: {err}")
        return float("-inf")

    try:
        results = run_benchmark(
            model=model,
            target_name=target_name,
            width=width,
            height=height,
            init_cells=init_cells,
            n_seeds=int(n_seeds),
            steps=steps,
            warmup=warmup,
            late_cleanup_start_frac=late_cleanup,
        )
    except Exception as err:  # pragma: no cover
        print(f"[HPO] multi_seed_iou benchmark failed: {err}")
        return float("-inf")

    if not results:
        return float("-inf")
    best_ious = [r[1] for r in results]
    _ = device  # accepted for parity; multiseed_bench runs on CPU
    return float(sum(best_ious) / len(best_ious))


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------


TrainGaFn = Callable[..., Path]


def run_trial(
    *,
    base_cfg: Dict[str, Any],
    params: Dict[str, Any],
    trial_dir: Path,
    objective: str,
    train_ga_fn: TrainGaFn,
    target_mask: Any,
    target_area: float,
    device: Any,
    seed: int,
    default_target_name: str,
    multi_seed_n: int = 5,
) -> float:
    """Apply ``params`` to ``base_cfg``, run ``train_ga``, return the objective.

    The trial runs inside ``trial_dir`` (via an os.chdir context) so all
    ``runs/`` artefacts land under that directory. This keeps concurrent
    Optuna studies from fighting over the top-level ``runs/`` folder.

    Parameters
    ----------
    base_cfg
        Parsed YAML dict — the starting point before HPO overrides.
    params
        ``{dotted_path: value}`` mapping from ``suggest_params``.
    trial_dir
        Working directory for this trial. Must exist.
    objective
        One of ``VALID_OBJECTIVES``. For ``multi_seed_iou``, the trained
        best.pt is loaded and benchmarked across ``multi_seed_n`` world
        seeds (the HANDOFF-recommended "gold standard" metric). For all
        other objectives, the value is read from ``generations.csv``.
    train_ga_fn
        Callable with the same signature as
        ``agents.train_agent.pipeline.train_ga``. Injected for
        testability — tests pass a mock.
    target_mask, target_area, device, seed, default_target_name
        Passed through to ``train_ga_fn`` unchanged.
    multi_seed_n
        Number of seeds for ``multi_seed_iou``. Ignored for other
        objectives. Default 5 (matches ``docs/leaderboard.md`` methodology).

    Returns
    -------
    float
        The objective value. ``float("-inf")`` on any failure.
    """
    import argparse
    import os

    # Resolve params against base cfg. For multi_seed_iou we also need
    # the post-override cfg to pass into the benchmark (so world.steps
    # / late_cleanup_start_frac overrides are respected during eval).
    trial_cfg = apply_params(base_cfg, params)
    original_cwd = os.getcwd()
    os.chdir(trial_dir)
    try:
        args = argparse.Namespace(no_viz=True, steps=None, train_ga=True)
        try:
            best_path = train_ga_fn(
                trial_cfg,
                args,
                device,
                target_mask,
                seed=seed,
                default_target_name=default_target_name,
                default_target_area=target_area,
            )
        except Exception as err:  # pragma: no cover - diagnostics
            print(f"[HPO] Trial failed with exception: {err}")
            return float("-inf")

        if objective in CHECKPOINT_OBJECTIVES:
            # best_path is resolved inside trial_dir (train_ga wrote under
            # its local runs/). Pass the post-override cfg so world.steps
            # and other swept values are respected during eval.
            return compute_multi_seed_objective(
                best_path,
                trial_cfg,
                n_seeds=multi_seed_n,
                device=device,
            )

        run_dir = best_path.parent
        return compute_objective(run_dir, objective=objective)
    finally:
        os.chdir(original_cwd)
