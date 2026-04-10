"""Tests for src/benchmark.py.

Covers the pure statistical helpers, ``summarize_benchmark`` verdict logic,
artifact writing, and an end-to-end ``evaluate_t_benchmark`` smoke test with
a randomly initialised model and tiny world/seed count.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.benchmark import (
    BenchmarkConfig,
    BenchmarkRun,
    BenchmarkSeedResult,
    BenchmarkSummary,
    _reproducibility_score,
    _safe_mean,
    _safe_median,
    _safe_std,
    evaluate_t_benchmark,
    summarize_benchmark,
    write_benchmark_artifacts,
)
from src.model import LittleLM
from src.utils import set_seed


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


class TestSafeStatisticHelpers:
    def test_mean_empty_returns_zero(self) -> None:
        assert _safe_mean([]) == 0.0

    def test_mean_non_empty(self) -> None:
        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0

    def test_median_empty_returns_zero(self) -> None:
        assert _safe_median([]) == 0.0

    def test_median_non_empty(self) -> None:
        assert _safe_median([1.0, 3.0, 2.0]) == 2.0

    def test_std_single_value_returns_zero(self) -> None:
        assert _safe_std([5.0]) == 0.0

    def test_std_multiple_values(self) -> None:
        assert _safe_std([0.0, 2.0]) > 0.0


class TestReproducibilityScore:
    def test_perfect_stable_zero_std(self) -> None:
        assert _reproducibility_score(1.0, 0.0, 0.0) == 1.0

    def test_penalty_subtracted(self) -> None:
        # 0.8 - (0.5*0.2 + 0.4*0.1) = 0.8 - 0.14 = 0.66
        score = _reproducibility_score(0.8, 0.2, 0.1)
        assert abs(score - 0.66) < 1e-6

    def test_clamped_to_zero(self) -> None:
        # Heavy penalty drives score below zero, clamp kicks in.
        assert _reproducibility_score(0.1, 1.0, 1.0) == 0.0

    def test_clamped_to_one(self) -> None:
        # Stabilized rate > 1 with zero penalty should clamp to 1.
        assert _reproducibility_score(1.5, 0.0, 0.0) == 1.0


# -----------------------------------------------------------------------------
# BenchmarkSeedResult / BenchmarkSummary dataclass behaviour
# -----------------------------------------------------------------------------


def _make_seed_result(**overrides) -> BenchmarkSeedResult:
    defaults = dict(
        seed=0,
        best_iou=0.8,
        final_iou=0.75,
        first_reach_step=50,
        hold_length=5.0,
        hold_window=10.0,
        stabilized_success=True,
        alive_final=20,
        area_final=25.0,
        components_final=1.0,
        t_sym_final=0.9,
        trunk_final=0.85,
        fp_final=0.02,
        fn_final=0.03,
        verdict="stabilized",
    )
    defaults.update(overrides)
    return BenchmarkSeedResult(**defaults)


class TestBenchmarkSeedResult:
    def test_reached_true_when_first_reach_step_set(self) -> None:
        assert _make_seed_result(first_reach_step=10).reached is True

    def test_reached_false_when_none(self) -> None:
        assert _make_seed_result(first_reach_step=None).reached is False

    def test_to_dict_round_trip(self) -> None:
        result = _make_seed_result()
        data = result.to_dict()
        assert data["seed"] == 0
        assert data["verdict"] == "stabilized"
        assert "best_iou" in data


# -----------------------------------------------------------------------------
# summarize_benchmark
# -----------------------------------------------------------------------------


def _make_config(**overrides) -> BenchmarkConfig:
    defaults = dict(
        mode="test",
        name="unit_test",
        fixed_seeds=(1, 2, 3),
        reach_threshold_iou=0.5,
        stabilize_threshold_iou=0.6,
        hold_window_steps=5,
        min_reach_rate=0.8,
        min_stabilized_rate=0.5,
        min_mean_final_iou=0.4,
        eval_steps=100,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


class TestSummarizeBenchmark:
    def test_all_passing_verdict(self) -> None:
        config = _make_config()
        per_seed = [
            _make_seed_result(seed=s, stabilized_success=True, final_iou=0.7)
            for s in (1, 2, 3)
        ]
        summary = summarize_benchmark(config, per_seed)
        assert summary.verdict == "benchmark_passed"
        assert summary.success_rate_reach == 1.0
        assert summary.success_rate_stabilized == 1.0
        assert summary.mean_final_iou == 0.7

    def test_failed_reach_rate(self) -> None:
        config = _make_config(min_reach_rate=0.9)
        per_seed = [
            _make_seed_result(seed=1, first_reach_step=None, stabilized_success=False),
            _make_seed_result(seed=2, first_reach_step=10),
            _make_seed_result(seed=3, first_reach_step=None, stabilized_success=False),
        ]
        summary = summarize_benchmark(config, per_seed)
        assert summary.verdict == "benchmark_failed"

    def test_failed_stabilized_rate(self) -> None:
        config = _make_config(min_stabilized_rate=1.0)
        per_seed = [
            _make_seed_result(seed=1, stabilized_success=True),
            _make_seed_result(seed=2, stabilized_success=False),
        ]
        summary = summarize_benchmark(config, per_seed)
        assert summary.verdict == "benchmark_failed"

    def test_failed_mean_final_iou(self) -> None:
        config = _make_config(min_mean_final_iou=0.9)
        per_seed = [
            _make_seed_result(seed=s, final_iou=0.5) for s in (1, 2, 3)
        ]
        summary = summarize_benchmark(config, per_seed)
        assert summary.verdict == "benchmark_failed"

    def test_empty_per_seed(self) -> None:
        config = _make_config(min_reach_rate=0.0, min_stabilized_rate=0.0, min_mean_final_iou=0.0)
        summary = summarize_benchmark(config, [])
        assert summary.success_rate_reach == 0.0
        assert summary.mean_final_iou == 0.0


# -----------------------------------------------------------------------------
# BenchmarkRun + write_benchmark_artifacts
# -----------------------------------------------------------------------------


class TestWriteBenchmarkArtifacts:
    def test_writes_summary_json_and_per_seed_csv(self, tmp_path: Path) -> None:
        config = _make_config()
        per_seed = [_make_seed_result(seed=s) for s in (1, 2, 3)]
        summary = summarize_benchmark(config, per_seed)
        run = BenchmarkRun(
            config=config,
            summary=summary,
            per_seed=per_seed,
            output_dir=tmp_path,
        )
        write_benchmark_artifacts(run)

        assert run.summary_path.exists()
        payload = json.loads(run.summary_path.read_text(encoding="utf-8"))
        assert payload["config"]["name"] == "unit_test"
        assert "summary" in payload

        assert run.per_seed_path.exists()
        csv_content = run.per_seed_path.read_text(encoding="utf-8").splitlines()
        assert csv_content[0].startswith("seed,")
        assert len(csv_content) == len(per_seed) + 1  # header + rows

        # per-seed JSON snapshots
        for result in per_seed:
            seed_file = tmp_path / "seeds" / f"seed_{result.seed:04d}.json"
            assert seed_file.exists()
            data = json.loads(seed_file.read_text(encoding="utf-8"))
            assert data["seed"] == result.seed


# -----------------------------------------------------------------------------
# evaluate_t_benchmark smoke test
# -----------------------------------------------------------------------------


class TestEvaluateBenchmarkSmoke:
    def test_runs_end_to_end_with_random_model(self, tmp_path: Path) -> None:
        set_seed(7)
        model = LittleLM(embed_dim=16, num_heads=2)
        model.eval()

        target_mask = torch.zeros((5, 5), dtype=torch.bool)
        target_mask[1:3, 1:3] = True
        target_area = float(target_mask.sum().item())

        config = BenchmarkConfig(
            mode="stability",
            name="smoke",
            fixed_seeds=(1, 2),
            reach_threshold_iou=0.0,  # trivially reachable
            stabilize_threshold_iou=0.0,
            hold_window_steps=0,
            min_reach_rate=0.0,
            min_stabilized_rate=0.0,
            min_mean_final_iou=0.0,
            eval_steps=5,
        )

        run = evaluate_t_benchmark(
            model,
            benchmark_cfg=config,
            world_cfg={"width": 5, "height": 5, "init_cells": 3},
            simulate_cfg={"anti_extinction_warmup_steps": 0, "late_cleanup_start_frac": 0.8},
            fitness_cfg={"alpha_fp": 1.0, "beta_fn": 0.5, "empty_collapse": {}, "late_t": {}},
            target_mask=target_mask,
            target_area=target_area,
            device=torch.device("cpu"),
            output_dir=tmp_path,
            stability_weights={"w_shape": 1.0},
        )

        assert isinstance(run.summary, BenchmarkSummary)
        assert len(run.per_seed) == 2
        assert run.summary_path.exists()
        assert run.per_seed_path.exists()
        for seed_result in run.per_seed:
            assert seed_result.verdict in {
                "failed_to_reach",
                "reached",
                "reached_but_not_stable",
                "stabilized",
            }
