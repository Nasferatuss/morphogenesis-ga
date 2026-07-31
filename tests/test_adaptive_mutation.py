"""Unit tests for agents/train_agent/adaptive_mutation.py.

Covers the config parser (``AdaptiveMutationConfig.from_dict``), the
population variance helper, and all three live strategies
(``diversity``, ``plateau``, ``diversity_plateau``) plus the
``none`` passthrough.
"""
from __future__ import annotations

import pytest

from agents.train_agent.adaptive_mutation import (
    AdaptiveMutationConfig,
    _population_variance,
    adapt_mutation_std,
)

# -----------------------------------------------------------------------------
# AdaptiveMutationConfig.from_dict
# -----------------------------------------------------------------------------


class TestAdaptiveMutationConfigFromDict:
    def test_none_returns_default_none_strategy(self) -> None:
        cfg = AdaptiveMutationConfig.from_dict(None)
        assert cfg.strategy == "none"

    def test_empty_dict_returns_default_none(self) -> None:
        cfg = AdaptiveMutationConfig.from_dict({})
        assert cfg.strategy == "none"

    def test_unknown_strategy_falls_back_to_none(self) -> None:
        cfg = AdaptiveMutationConfig.from_dict({"strategy": "mystery"})
        assert cfg.strategy == "none"

    def test_valid_strategies_parsed(self) -> None:
        for name in ("none", "diversity", "plateau", "diversity_plateau"):
            cfg = AdaptiveMutationConfig.from_dict({"strategy": name})
            assert cfg.strategy == name

    def test_numeric_fields_parsed(self) -> None:
        cfg = AdaptiveMutationConfig.from_dict(
            {
                "strategy": "diversity_plateau",
                "diversity_threshold": 0.1,
                "diversity_boost": 0.03,
                "plateau_threshold": 5,
                "plateau_boost": 0.04,
                "std_cap": 0.2,
            }
        )
        assert cfg.diversity_threshold == 0.1
        assert cfg.diversity_boost == 0.03
        assert cfg.plateau_threshold == 5
        assert cfg.plateau_boost == 0.04
        assert cfg.std_cap == 0.2


# -----------------------------------------------------------------------------
# _population_variance helper
# -----------------------------------------------------------------------------


class TestPopulationVariance:
    def test_empty_returns_zero(self) -> None:
        assert _population_variance([]) == 0.0

    def test_single_value_returns_zero(self) -> None:
        assert _population_variance([3.0]) == 0.0

    def test_constant_population_zero_variance(self) -> None:
        assert _population_variance([2.0, 2.0, 2.0, 2.0]) == 0.0

    def test_symmetric_population(self) -> None:
        # Values [-1, 1] → mean 0, variance (1+1)/2 = 1.0
        assert _population_variance([-1.0, 1.0]) == pytest.approx(1.0)

    def test_known_value(self) -> None:
        # Values [0, 0, 0, 4] → mean 1, deviations [-1,-1,-1,3]
        # → squared [1,1,1,9] → variance 12/4 = 3.0
        assert _population_variance([0.0, 0.0, 0.0, 4.0]) == pytest.approx(3.0)


# -----------------------------------------------------------------------------
# adapt_mutation_std — strategy behaviour
# -----------------------------------------------------------------------------


class TestAdaptMutationStdNoneStrategy:
    def test_none_passthrough(self) -> None:
        cfg = AdaptiveMutationConfig(strategy="none")
        adjusted, diag = adapt_mutation_std(
            0.02, fitnesses=[0.1, 0.2, 0.3], gens_since_improve=10, cfg=cfg
        )
        assert adjusted == 0.02
        assert diag["adjusted_std"] == 0.02
        assert diag["base_std"] == 0.02
        assert diag["diversity_boost_applied"] is False
        assert diag["plateau_boost_applied"] is False

    def test_none_ignores_empty_fitnesses(self) -> None:
        cfg = AdaptiveMutationConfig(strategy="none")
        adjusted, _ = adapt_mutation_std(
            0.05, fitnesses=[], gens_since_improve=99, cfg=cfg
        )
        assert adjusted == 0.05


class TestAdaptMutationStdDiversity:
    def test_low_variance_triggers_boost(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="diversity", diversity_threshold=1.0, diversity_boost=0.01
        )
        # all values identical → variance = 0 < 1.0 → boost triggers
        adjusted, diag = adapt_mutation_std(
            0.02, fitnesses=[0.5, 0.5, 0.5], gens_since_improve=0, cfg=cfg
        )
        assert adjusted == pytest.approx(0.03)
        assert diag["diversity_boost_applied"] is True
        assert diag["plateau_boost_applied"] is False

    def test_high_variance_no_boost(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="diversity", diversity_threshold=0.1, diversity_boost=0.01
        )
        # wide spread → variance high → no boost
        adjusted, diag = adapt_mutation_std(
            0.02, fitnesses=[-1.0, 0.0, 1.0], gens_since_improve=0, cfg=cfg
        )
        assert adjusted == pytest.approx(0.02)
        assert diag["diversity_boost_applied"] is False

    def test_diversity_strategy_ignores_plateau_signal(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="diversity", diversity_threshold=0.0, diversity_boost=0.01
        )
        # plateau counter huge, but diversity threshold = 0 → no boost
        adjusted, diag = adapt_mutation_std(
            0.02, fitnesses=[0.5, 0.5], gens_since_improve=999, cfg=cfg
        )
        # variance = 0 < 0.0 is False (strictly less) → no boost
        assert adjusted == pytest.approx(0.02)
        assert diag["plateau_boost_applied"] is False


class TestAdaptMutationStdPlateau:
    def test_plateau_threshold_triggers_boost(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="plateau", plateau_threshold=3, plateau_boost=0.02
        )
        adjusted, diag = adapt_mutation_std(
            0.01, fitnesses=[0.1, 0.2, 0.3], gens_since_improve=3, cfg=cfg
        )
        assert adjusted == pytest.approx(0.03)
        assert diag["plateau_boost_applied"] is True

    def test_below_threshold_no_boost(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="plateau", plateau_threshold=5, plateau_boost=0.02
        )
        adjusted, diag = adapt_mutation_std(
            0.01, fitnesses=[0.1, 0.2], gens_since_improve=4, cfg=cfg
        )
        assert adjusted == pytest.approx(0.01)
        assert diag["plateau_boost_applied"] is False

    def test_plateau_strategy_ignores_diversity_signal(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="plateau", plateau_threshold=3, plateau_boost=0.02
        )
        # variance is 0, but we're not in plateau → no boost
        adjusted, diag = adapt_mutation_std(
            0.01, fitnesses=[0.5, 0.5], gens_since_improve=1, cfg=cfg
        )
        assert adjusted == pytest.approx(0.01)
        assert diag["diversity_boost_applied"] is False


class TestAdaptMutationStdDiversityPlateau:
    def test_both_signals_boost_takes_max(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="diversity_plateau",
            diversity_threshold=1.0,
            diversity_boost=0.01,
            plateau_threshold=3,
            plateau_boost=0.05,
        )
        # Both fire: variance=0 < 1.0, gens_since=5 >= 3
        # Combined takes max(base+0.01, base+0.05) = base+0.05
        adjusted, diag = adapt_mutation_std(
            0.02, fitnesses=[0.5, 0.5], gens_since_improve=5, cfg=cfg
        )
        assert adjusted == pytest.approx(0.07)
        assert diag["diversity_boost_applied"] is True
        assert diag["plateau_boost_applied"] is True

    def test_only_diversity_fires(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="diversity_plateau",
            diversity_threshold=1.0,
            diversity_boost=0.03,
            plateau_threshold=10,
            plateau_boost=0.05,
        )
        adjusted, diag = adapt_mutation_std(
            0.01, fitnesses=[0.5, 0.5], gens_since_improve=2, cfg=cfg
        )
        assert adjusted == pytest.approx(0.04)
        assert diag["diversity_boost_applied"] is True
        assert diag["plateau_boost_applied"] is False

    def test_only_plateau_fires(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="diversity_plateau",
            diversity_threshold=0.0,
            diversity_boost=0.03,
            plateau_threshold=3,
            plateau_boost=0.05,
        )
        adjusted, diag = adapt_mutation_std(
            0.01, fitnesses=[-1.0, 1.0], gens_since_improve=4, cfg=cfg
        )
        assert adjusted == pytest.approx(0.06)
        assert diag["diversity_boost_applied"] is False
        assert diag["plateau_boost_applied"] is True


class TestAdaptMutationStdCap:
    def test_std_cap_clamps_boost(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="plateau",
            plateau_threshold=1,
            plateau_boost=1.0,  # absurdly large
            std_cap=0.05,
        )
        adjusted, diag = adapt_mutation_std(
            0.04, fitnesses=[0.1], gens_since_improve=5, cfg=cfg
        )
        assert adjusted == pytest.approx(0.05)
        assert diag["plateau_boost_applied"] is True

    def test_base_already_at_cap_no_change(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="diversity",
            diversity_threshold=1.0,
            diversity_boost=0.5,
            std_cap=0.02,
        )
        adjusted, _ = adapt_mutation_std(
            0.02, fitnesses=[0.5, 0.5], gens_since_improve=0, cfg=cfg
        )
        # Boost would push to 0.52 but cap is 0.02
        assert adjusted == pytest.approx(0.02)


class TestAdaptMutationStdDiagnostics:
    def test_diagnostics_structure(self) -> None:
        cfg = AdaptiveMutationConfig(
            strategy="diversity_plateau",
            diversity_threshold=0.5,
            plateau_threshold=3,
        )
        _, diag = adapt_mutation_std(
            0.02, fitnesses=[0.1, 0.2, 0.3], gens_since_improve=2, cfg=cfg
        )
        # Must always have these 6 keys regardless of whether boosts fired.
        assert set(diag.keys()) == {
            "strategy",
            "base_std",
            "variance",
            "plateau_boost_applied",
            "diversity_boost_applied",
            "adjusted_std",
        }
        assert diag["strategy"] == "diversity_plateau"
        assert diag["base_std"] == 0.02
        assert diag["variance"] >= 0.0
