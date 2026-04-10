"""Tests for ``core/services/hpo.py``.

Covers the pure helpers (``ParamSpec`` validation, ``load_search_space``,
``apply_params`` dotted-path merge, ``compute_objective``, and the
trial runner) without actually launching Optuna. A single ``slow``
integration test drives an end-to-end 2-trial study using a mocked
``train_ga`` to verify the Optuna → search-space → runner glue.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from core.services.hpo import (
    CHECKPOINT_OBJECTIVES,
    VALID_OBJECTIVES,
    ParamSpec,
    apply_params,
    compute_multi_seed_objective,
    compute_objective,
    load_search_space,
    run_trial,
    suggest_params,
)


# ---------------------------------------------------------------------------
# ParamSpec validation
# ---------------------------------------------------------------------------


class TestParamSpecValidation:
    def test_float_valid(self) -> None:
        spec = ParamSpec(path="ga.mutation_prob", kind="float", low=0.05, high=0.3)
        assert spec.kind == "float"
        assert spec.low == 0.05

    def test_int_valid(self) -> None:
        spec = ParamSpec(path="ga.plateau_threshold", kind="int", low=2, high=6)
        assert spec.kind == "int"

    def test_categorical_valid(self) -> None:
        spec = ParamSpec(
            path="ga.adaptive_mutation.strategy",
            kind="categorical",
            choices=["none", "diversity"],
        )
        assert spec.choices == ["none", "diversity"]

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown kind"):
            ParamSpec(path="x", kind="nonsense")

    def test_float_missing_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="needs low \\+ high"):
            ParamSpec(path="x", kind="float")

    def test_low_equal_high_raises(self) -> None:
        with pytest.raises(ValueError, match="low .* >= high"):
            ParamSpec(path="x", kind="float", low=1.0, high=1.0)

    def test_categorical_empty_choices_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty choices"):
            ParamSpec(path="x", kind="categorical", choices=[])


# ---------------------------------------------------------------------------
# load_search_space
# ---------------------------------------------------------------------------


class TestLoadSearchSpace:
    def test_loads_minimal_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "ss.yaml"
        path.write_text(
            """
name: test
parameters:
  ga.mutation_prob:
    type: float
    low: 0.05
    high: 0.3
  ga.adaptive_mutation.strategy:
    type: categorical
    choices: [none, diversity]
""",
            encoding="utf-8",
        )
        specs = load_search_space(path)
        assert len(specs) == 2
        paths = {s.path for s in specs}
        assert paths == {"ga.mutation_prob", "ga.adaptive_mutation.strategy"}

    def test_missing_parameters_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("name: test\n", encoding="utf-8")
        with pytest.raises(ValueError, match="top-level 'parameters'"):
            load_search_space(path)

    def test_non_dict_param_spec_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "parameters:\n  ga.mutation_prob: not_a_dict\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="spec must be a dict"):
            load_search_space(path)

    def test_real_v1_search_space(self) -> None:
        """Smoke: the shipped v1 search space parses cleanly."""
        path = Path(__file__).resolve().parent.parent / "configs" / "hpo" / "search_space_v1.yaml"
        specs = load_search_space(path)
        assert len(specs) >= 10
        strategy_spec = next(
            s for s in specs if s.path == "ga.adaptive_mutation.strategy"
        )
        assert strategy_spec.kind == "categorical"
        assert "diversity_plateau" in (strategy_spec.choices or [])


# ---------------------------------------------------------------------------
# suggest_params with a mock trial
# ---------------------------------------------------------------------------


class _MockTrial:
    """Deterministic stand-in for ``optuna.Trial`` — returns midpoints."""

    def suggest_float(self, path: str, low: float, high: float, log: bool = False) -> float:
        return (low + high) / 2.0

    def suggest_int(self, path: str, low: int, high: int) -> int:
        return (low + high) // 2

    def suggest_categorical(self, path: str, choices: List[Any]) -> Any:
        return choices[0]


class TestSuggestParams:
    def test_float_midpoint(self) -> None:
        specs = [ParamSpec(path="x", kind="float", low=0.0, high=1.0)]
        params = suggest_params(_MockTrial(), specs)
        assert params == {"x": 0.5}

    def test_int_midpoint(self) -> None:
        specs = [ParamSpec(path="n", kind="int", low=2, high=6)]
        assert suggest_params(_MockTrial(), specs) == {"n": 4}

    def test_categorical_first_choice(self) -> None:
        specs = [ParamSpec(path="s", kind="categorical", choices=["a", "b", "c"])]
        assert suggest_params(_MockTrial(), specs) == {"s": "a"}

    def test_multi_param(self) -> None:
        specs = [
            ParamSpec(path="a.b", kind="float", low=0.0, high=1.0),
            ParamSpec(path="c", kind="int", low=1, high=9),
        ]
        params = suggest_params(_MockTrial(), specs)
        assert params == {"a.b": 0.5, "c": 5}


# ---------------------------------------------------------------------------
# apply_params (dotted-path merge)
# ---------------------------------------------------------------------------


class TestApplyParams:
    def test_shallow_override(self) -> None:
        cfg = {"ga": {"mutation_prob": 0.1}}
        merged = apply_params(cfg, {"ga.mutation_prob": 0.25})
        assert merged["ga"]["mutation_prob"] == 0.25
        # Original untouched (deep copy)
        assert cfg["ga"]["mutation_prob"] == 0.1

    def test_nested_creation(self) -> None:
        cfg: Dict[str, Any] = {"ga": {}}
        merged = apply_params(
            cfg, {"ga.adaptive_mutation.strategy": "diversity"}
        )
        assert merged["ga"]["adaptive_mutation"]["strategy"] == "diversity"

    def test_overwrites_non_dict_with_dict(self) -> None:
        # If an intermediate path collides with a non-dict value,
        # the setter replaces it with a fresh dict.
        cfg = {"ga": {"adaptive_mutation": "legacy_string"}}
        merged = apply_params(
            cfg, {"ga.adaptive_mutation.strategy": "plateau"}
        )
        assert merged["ga"]["adaptive_mutation"] == {"strategy": "plateau"}

    def test_multiple_params(self) -> None:
        cfg: Dict[str, Any] = {"ga": {"mutation_prob": 0.1}, "fitness": {}}
        merged = apply_params(
            cfg,
            {
                "ga.mutation_prob": 0.2,
                "fitness.alpha_fp": 1.5,
                "fitness.beta_fn": 1.8,
            },
        )
        assert merged["ga"]["mutation_prob"] == 0.2
        assert merged["fitness"]["alpha_fp"] == 1.5
        assert merged["fitness"]["beta_fn"] == 1.8

    def test_empty_params_passthrough(self) -> None:
        cfg = {"ga": {"mutation_prob": 0.1}}
        merged = apply_params(cfg, {})
        assert merged == cfg
        assert merged is not cfg  # still a copy


# ---------------------------------------------------------------------------
# compute_objective
# ---------------------------------------------------------------------------


def _write_generations_csv(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "generations.csv"
    fieldnames = list(rows[0].keys()) if rows else ["generation"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestComputeObjective:
    def test_missing_csv_returns_neg_inf(self, tmp_path: Path) -> None:
        assert compute_objective(tmp_path, "best_iou") == float("-inf")

    def test_empty_csv_returns_neg_inf(self, tmp_path: Path) -> None:
        (tmp_path / "generations.csv").write_text("generation\n", encoding="utf-8")
        assert compute_objective(tmp_path, "best_iou") == float("-inf")

    def test_best_iou_takes_max(self, tmp_path: Path) -> None:
        _write_generations_csv(
            tmp_path,
            [
                {"generation": 1, "best_iou": "0.1", "best_fitness": "-1.0"},
                {"generation": 2, "best_iou": "0.3", "best_fitness": "-0.5"},
                {"generation": 3, "best_iou": "0.2", "best_fitness": "-0.8"},
            ],
        )
        assert compute_objective(tmp_path, "best_iou") == pytest.approx(0.3)

    def test_final_best_iou_takes_last(self, tmp_path: Path) -> None:
        _write_generations_csv(
            tmp_path,
            [
                {"generation": 1, "best_iou": "0.5"},
                {"generation": 2, "best_iou": "0.2"},
            ],
        )
        assert compute_objective(tmp_path, "final_best_iou") == pytest.approx(0.2)

    def test_best_fitness_takes_max(self, tmp_path: Path) -> None:
        _write_generations_csv(
            tmp_path,
            [
                {"generation": 1, "best_fitness": "-1.5"},
                {"generation": 2, "best_fitness": "-0.3"},
                {"generation": 3, "best_fitness": "-0.9"},
            ],
        )
        assert compute_objective(tmp_path, "best_fitness") == pytest.approx(-0.3)

    def test_composite(self, tmp_path: Path) -> None:
        _write_generations_csv(
            tmp_path,
            [
                {"generation": 1, "best_iou": "0.3", "stability_score": "0.4"},
                {"generation": 2, "best_iou": "0.5", "stability_score": "0.2"},
            ],
        )
        # composite = max(iou) + 0.5 * max(stab) = 0.5 + 0.5 * 0.4 = 0.7
        assert compute_objective(tmp_path, "composite") == pytest.approx(0.7)

    def test_unknown_objective_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown objective"):
            compute_objective(tmp_path, "totally_fake")

    def test_valid_objectives_set(self) -> None:
        assert "best_iou" in VALID_OBJECTIVES
        assert "composite" in VALID_OBJECTIVES
        assert "multi_seed_iou" in VALID_OBJECTIVES

    def test_multi_seed_iou_in_checkpoint_objectives(self) -> None:
        """multi_seed_iou must route through compute_multi_seed_objective."""
        assert "multi_seed_iou" in CHECKPOINT_OBJECTIVES
        assert "best_iou" not in CHECKPOINT_OBJECTIVES

    def test_compute_objective_rejects_multi_seed_iou(self, tmp_path: Path) -> None:
        """compute_objective must refuse checkpoint-based objectives."""
        _write_generations_csv(
            tmp_path, [{"generation": 1, "best_iou": "0.3"}]
        )
        with pytest.raises(ValueError, match="requires compute_multi_seed_objective"):
            compute_objective(tmp_path, "multi_seed_iou")


# ---------------------------------------------------------------------------
# compute_multi_seed_objective
# ---------------------------------------------------------------------------


class TestComputeMultiSeedObjective:
    def test_missing_checkpoint_returns_neg_inf(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.pt"
        result = compute_multi_seed_objective(missing, {})
        assert result == float("-inf")

    def test_import_or_load_failure_returns_neg_inf(self, tmp_path: Path) -> None:
        """Any exception during load or benchmark is caught → -inf.

        We create an empty file so ``exists()`` passes but
        ``torch.load`` will fail, exercising the load-failure path.
        """
        bad_ckpt = tmp_path / "empty.pt"
        bad_ckpt.write_bytes(b"")
        result = compute_multi_seed_objective(bad_ckpt, {}, n_seeds=2)
        assert result == float("-inf")


# ---------------------------------------------------------------------------
# run_trial — mocked train_ga
# ---------------------------------------------------------------------------


class TestRunTrial:
    def test_happy_path_uses_mock_train_ga(self, tmp_path: Path) -> None:
        captured: Dict[str, Any] = {}

        def fake_train_ga(
            cfg: Dict[str, Any],
            args: Any,
            device: Any,
            target_mask: Any,
            *,
            seed: int,
            default_target_name: str,
            default_target_area: float,
        ) -> Path:
            # Record what we got so we can assert on it
            captured["cfg_mutation_prob"] = cfg["ga"]["mutation_prob"]
            captured["cfg_strategy"] = cfg["ga"]["adaptive_mutation"]["strategy"]
            captured["seed"] = seed
            captured["default_target_name"] = default_target_name
            # Simulate a successful train run by writing a fake generations.csv
            run_dir = Path("fake_run")
            run_dir.mkdir(exist_ok=True)
            _write_generations_csv(
                run_dir,
                [
                    {"generation": 1, "best_iou": "0.1", "best_fitness": "-0.5"},
                    {"generation": 2, "best_iou": "0.25", "best_fitness": "-0.2"},
                ],
            )
            return run_dir / "best.pt"

        base_cfg = {
            "ga": {"mutation_prob": 0.1},
            "fitness": {},
        }
        params = {
            "ga.mutation_prob": 0.22,
            "ga.adaptive_mutation.strategy": "plateau",
        }

        trial_dir = tmp_path / "trial_0"
        trial_dir.mkdir()

        value = run_trial(
            base_cfg=base_cfg,
            params=params,
            trial_dir=trial_dir,
            objective="best_iou",
            train_ga_fn=fake_train_ga,
            target_mask=SimpleNamespace(),
            target_area=25.0,
            device=SimpleNamespace(),
            seed=7,
            default_target_name="T",
        )

        assert value == pytest.approx(0.25)
        # train_ga was called with the HPO-applied cfg
        assert captured["cfg_mutation_prob"] == 0.22
        assert captured["cfg_strategy"] == "plateau"
        assert captured["seed"] == 7
        assert captured["default_target_name"] == "T"

    def test_exception_returns_neg_inf(self, tmp_path: Path) -> None:
        def broken_train_ga(*args: Any, **kwargs: Any) -> Path:
            raise RuntimeError("boom")

        trial_dir = tmp_path / "trial_0"
        trial_dir.mkdir()
        value = run_trial(
            base_cfg={"ga": {}},
            params={"ga.mutation_prob": 0.1},
            trial_dir=trial_dir,
            objective="best_iou",
            train_ga_fn=broken_train_ga,
            target_mask=None,
            target_area=1.0,
            device=None,
            seed=1,
            default_target_name="T",
        )
        assert value == float("-inf")

    def test_cwd_restored_after_trial(self, tmp_path: Path) -> None:
        import os

        def noop_train_ga(*args: Any, **kwargs: Any) -> Path:
            return Path("nonexistent") / "best.pt"

        original_cwd = os.getcwd()
        trial_dir = tmp_path / "trial_0"
        trial_dir.mkdir()
        run_trial(
            base_cfg={},
            params={},
            trial_dir=trial_dir,
            objective="best_iou",
            train_ga_fn=noop_train_ga,
            target_mask=None,
            target_area=1.0,
            device=None,
            seed=1,
            default_target_name="T",
        )
        assert os.getcwd() == original_cwd

    def test_multi_seed_iou_routes_to_checkpoint_objective(
        self, tmp_path: Path
    ) -> None:
        """When objective=multi_seed_iou, run_trial must call
        compute_multi_seed_objective (not compute_objective).

        We verify by pointing at a nonexistent best.pt — the
        multi-seed path returns -inf for a missing checkpoint,
        while compute_objective(best_iou) would read a fake
        generations.csv and return a real number. Different return
        values prove the right branch was taken.
        """
        def fake_train_ga(*args: Any, **kwargs: Any) -> Path:
            # Write a generations.csv that WOULD give 0.5 under best_iou,
            # then return a best.pt path that doesn't actually exist.
            run_dir = Path("fake_run_multi")
            run_dir.mkdir(exist_ok=True)
            _write_generations_csv(
                run_dir,
                [{"generation": 1, "best_iou": "0.5", "best_fitness": "-0.1"}],
            )
            return run_dir / "best.pt"  # file not created → -inf

        trial_dir = tmp_path / "trial_0"
        trial_dir.mkdir()
        value = run_trial(
            base_cfg={"ga": {}, "world": {}, "simulate": {}, "target": {}, "model": {}},
            params={},
            trial_dir=trial_dir,
            objective="multi_seed_iou",
            train_ga_fn=fake_train_ga,
            target_mask=None,
            target_area=1.0,
            device=None,
            seed=1,
            default_target_name="T",
            multi_seed_n=2,
        )
        # multi_seed_iou on missing checkpoint → -inf, proving the
        # correct branch (compute_multi_seed_objective) was taken.
        assert value == float("-inf")

    def test_best_iou_still_works_with_default_multi_seed_n(
        self, tmp_path: Path
    ) -> None:
        """multi_seed_n default must not break the standard best_iou path."""
        def fake_train_ga(*args: Any, **kwargs: Any) -> Path:
            run_dir = Path("fake_run_legacy")
            run_dir.mkdir(exist_ok=True)
            _write_generations_csv(
                run_dir, [{"generation": 1, "best_iou": "0.42"}]
            )
            return run_dir / "best.pt"

        trial_dir = tmp_path / "trial_0"
        trial_dir.mkdir()
        value = run_trial(
            base_cfg={"ga": {}},
            params={},
            trial_dir=trial_dir,
            objective="best_iou",
            train_ga_fn=fake_train_ga,
            target_mask=None,
            target_area=1.0,
            device=None,
            seed=1,
            default_target_name="T",
            # multi_seed_n omitted → uses default 5 → ignored for best_iou
        )
        assert value == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Integration: actual Optuna study with a mocked train_ga (2 trials)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOptunaIntegration:
    def test_two_trial_study_with_mock_train_ga(self, tmp_path: Path) -> None:
        """Drive a real optuna.Study for 2 trials — verifies full glue."""
        optuna = pytest.importorskip("optuna")

        def deterministic_train_ga(
            cfg: Dict[str, Any],
            args: Any,
            device: Any,
            target_mask: Any,
            *,
            seed: int,
            default_target_name: str,
            default_target_area: float,
        ) -> Path:
            # Reward high mutation_prob so Optuna has a clear direction
            score = float(cfg["ga"]["mutation_prob"])
            run_dir = Path("mock_run")
            run_dir.mkdir(exist_ok=True)
            _write_generations_csv(
                run_dir, [{"generation": 1, "best_iou": f"{score:.6f}"}]
            )
            return run_dir / "best.pt"

        specs = [
            ParamSpec(path="ga.mutation_prob", kind="float", low=0.05, high=0.3)
        ]
        base_cfg: Dict[str, Any] = {"ga": {}}

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )

        def objective(trial: "optuna.Trial") -> float:
            params = suggest_params(trial, specs)
            trial_dir = tmp_path / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            return run_trial(
                base_cfg=base_cfg,
                params=params,
                trial_dir=trial_dir,
                objective="best_iou",
                train_ga_fn=deterministic_train_ga,
                target_mask=None,
                target_area=1.0,
                device=None,
                seed=42,
                default_target_name="T",
            )

        study.optimize(objective, n_trials=2)
        assert len(study.trials) == 2
        for t in study.trials:
            assert t.value is not None
            assert 0.05 <= t.params["ga.mutation_prob"] <= 0.3
            # deterministic_train_ga makes value == mutation_prob, modulo
            # the 6-decimal CSV roundtrip in our mock
            assert t.value == pytest.approx(
                t.params["ga.mutation_prob"], abs=1e-5
            )
