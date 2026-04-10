"""Tests for agents/train_agent/ extracted modules."""
from __future__ import annotations

import pytest

from agents.train_agent.mutation import (
    TMutationConfig,
    compute_phase_cleanup_ratio,
    resolve_mutation_std,
    resolve_stem_penalty_multiplier,
)
from agents.train_agent.scoring import (
    compute_benchmark_dict_score,
    simple_mean,
    simple_variance,
)
from agents.train_agent.t_weights import (
    T_STRUCTURE_PENALTY_KEYS,
    T_STRUCTURE_REWARD_KEYS,
    scale_t_weights,
)


class TestResolveMutationStd:
    def test_cross_phase(self):
        assert resolve_mutation_std("cross", "phase_cross", 0, 50) == 0.02

    def test_transition_phase(self):
        assert resolve_mutation_std("t", "phase_transition", 0, 16) == 0.015

    def test_refine_phase(self):
        assert resolve_mutation_std("t", "phase_T_refine", 0, 30) == 0.0015

    def test_t_phase_start(self):
        t_cfg = TMutationConfig(std_start=0.012, std_end=0.004, std_floor=0.002, std_cap=0.015)
        val = resolve_mutation_std("t", "phase_T", 0, 100, t_cfg=t_cfg)
        assert 0.002 <= val <= 0.015

    def test_t_phase_end(self):
        t_cfg = TMutationConfig(std_start=0.012, std_end=0.004, std_floor=0.002, std_cap=0.015)
        val = resolve_mutation_std("t", "phase_T", 99, 100, t_cfg=t_cfg)
        assert 0.002 <= val <= 0.015
        # At end, std should be closer to std_end than to std_start
        assert val <= t_cfg.std_start

    def test_default_fallback(self):
        assert resolve_mutation_std("unknown", "unknown", 0, 50, default_std=0.05) == 0.05


class TestResolveStemPenaltyMultiplier:
    def test_transition(self):
        assert resolve_stem_penalty_multiplier("t", "phase_transition", 0, 16) == 0.6

    def test_t_phase_early(self):
        val = resolve_stem_penalty_multiplier("t", "phase_T", 0, 100)
        assert 0.6 <= val < 1.0

    def test_t_phase_late(self):
        val = resolve_stem_penalty_multiplier("t", "phase_T", 50, 100)
        assert val == 1.0

    def test_cross_phase(self):
        assert resolve_stem_penalty_multiplier("cross", "phase_cross", 0, 50) == 1.0


class TestComputePhaseCleanupRatio:
    def test_non_t_returns_zero(self):
        assert compute_phase_cleanup_ratio("cross", 50, 100) == 0.0

    def test_t_early_returns_zero(self):
        assert compute_phase_cleanup_ratio("t", 0, 100, cleanup_start_frac=0.6) == 0.0

    def test_t_late_returns_positive(self):
        val = compute_phase_cleanup_ratio("t", 80, 100, cleanup_start_frac=0.6)
        assert val > 0.0

    def test_t_end_near_one(self):
        val = compute_phase_cleanup_ratio("t", 99, 100, cleanup_start_frac=0.6, cleanup_power=1.0)
        assert val > 0.9


class TestSimpleMeanVariance:
    def test_mean_empty(self):
        assert simple_mean([]) == 0.0

    def test_mean_values(self):
        assert simple_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_variance_empty(self):
        assert simple_variance([]) == 0.0

    def test_variance_single(self):
        assert simple_variance([5.0]) == 0.0

    def test_variance_values(self):
        # [1, 2, 3] -> mean=2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
        assert simple_variance([1.0, 2.0, 3.0]) == pytest.approx(2.0 / 3.0)


class TestComputeBenchmarkDictScore:
    def test_none(self):
        assert compute_benchmark_dict_score(None) == 0.0

    def test_empty(self):
        assert compute_benchmark_dict_score({}) == 0.0

    def test_full_scores(self):
        d = {
            "success_rate_stabilized": 1.0,
            "success_rate_reach": 1.0,
            "mean_final_iou": 1.0,
            "mean_best_iou": 1.0,
        }
        score = compute_benchmark_dict_score(d)
        assert score == pytest.approx(1.5 + 1.0 + 1.0 + 0.25)


class TestScaleTWeights:
    def test_none_base(self):
        assert scale_t_weights(None, 1.0, 1.0, 1.0) is None

    def test_empty_base(self):
        assert scale_t_weights({}, 1.0, 1.0, 1.0) is None

    def test_reward_scaling(self):
        base = {"top_bar_weight": 1.0, "trunk_weight": 0.5}
        result = scale_t_weights(base, reward_scale=2.0, penalty_scale=1.0, clean_scale=1.0)
        assert result["top_bar_weight"] == 2.0
        assert result["trunk_weight"] == 1.0

    def test_penalty_scaling(self):
        base = {"excess_below_weight": 1.0}
        result = scale_t_weights(base, reward_scale=1.0, penalty_scale=0.5, clean_scale=1.0)
        assert result["excess_below_weight"] == 0.5

    def test_cleanliness_scaling(self):
        base = {"cleanliness_weight": 1.0}
        result = scale_t_weights(base, reward_scale=1.0, penalty_scale=1.0, clean_scale=3.0)
        assert result["cleanliness_weight"] == 3.0

    def test_key_constants_not_empty(self):
        assert len(T_STRUCTURE_REWARD_KEYS) >= 4
        assert len(T_STRUCTURE_PENALTY_KEYS) >= 5
