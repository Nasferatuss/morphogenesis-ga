"""Adaptive mutation std strategies for GA training.

Three pluggable strategies for adjusting mutation σ on the fly based on
population statistics. Default strategy is ``"none"`` — strict passthrough
of the base std computed by existing curriculum schedules, which keeps
``TestGoldenBaselineV1`` (the reproducibility freeze) bit-identical.

Strategies
----------
- ``none`` (default) — no adaptation, returns ``base_std`` unchanged.
- ``diversity`` — boost σ when population fitness variance drops below
  ``diversity_threshold`` (population collapsed into clone-like region).
- ``plateau`` — boost σ when ``gens_since_improve`` ≥ ``plateau_threshold``
  (best fitness hasn't improved for N generations).
- ``diversity_plateau`` — apply the max of both boosts; fires if either
  signal is triggered.

Both boosts are additive to ``base_std``, then clamped to ``std_cap``.

Why adaptive σ exists
---------------------
``agents/train_agent/mutation.py`` ships hand-tuned curriculum schedules
with 6+ hyperparameters (``t_mutation_std_start/end/floor/cap``,
``warmup_frac``, ``ease_power``). These schedules are fragile across
targets — cross/T/damage-recover each historically needed re-tuning, and
the 2026-04-09 audit flagged fragile curriculum transitions as one of
the top three causes of the 0.256 IoU ceiling (see
``docs/phase0_audit_report.md``).

Adaptive σ reduces the need for manual curriculum tuning by letting the
GA react to its own population state. It is intentionally additive to
the existing schedule, not a replacement — schedules still set the base;
adaptive logic only boosts when the signal fires.

Self-adaptive σ per individual (CMA-ES style) is deferred to a v2 spike
because it requires deeper surgery on ``src.ga.mutate`` (each genome
would carry its own σ as an extra parameter). See
``docs/research/adaptive_mutation_report.md`` for the deferral rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

_VALID_STRATEGIES = {"none", "diversity", "plateau", "diversity_plateau"}


@dataclass(frozen=True)
class AdaptiveMutationConfig:
    """Configuration bundle for ``adapt_mutation_std``.

    Attributes
    ----------
    strategy
        One of ``"none"``, ``"diversity"``, ``"plateau"``,
        ``"diversity_plateau"``. Unknown values are coerced to ``"none"``
        with a warning-free silent fallback (keeps configs forward-compat).
    diversity_threshold
        Population fitness variance below this → diversity boost fires.
        Compared against the *population* variance (sum((f-mean)^2) / n).
    diversity_boost
        Additive boost applied to ``base_std`` when diversity boost fires.
    plateau_threshold
        Minimum number of generations since last best-fitness improvement
        before plateau boost fires.
    plateau_boost
        Additive boost applied to ``base_std`` when plateau boost fires.
    std_cap
        Absolute upper bound on the adjusted std, after all boosts.
    """

    strategy: str = "none"
    diversity_threshold: float = 0.05
    diversity_boost: float = 0.015
    plateau_threshold: int = 3
    plateau_boost: float = 0.02
    std_cap: float = 0.1

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "AdaptiveMutationConfig":
        """Build from a YAML-parsed dict (or ``None``).

        ``None``, empty dict, or unknown strategy all resolve to the
        ``"none"`` passthrough strategy so existing configs remain
        bit-identical.
        """
        if not cfg:
            return cls()
        strategy = str(cfg.get("strategy", "none"))
        if strategy not in _VALID_STRATEGIES:
            strategy = "none"
        return cls(
            strategy=strategy,
            diversity_threshold=float(cfg.get("diversity_threshold", 0.05)),
            diversity_boost=float(cfg.get("diversity_boost", 0.015)),
            plateau_threshold=int(cfg.get("plateau_threshold", 3)),
            plateau_boost=float(cfg.get("plateau_boost", 0.02)),
            std_cap=float(cfg.get("std_cap", 0.1)),
        )


def _population_variance(values: Sequence[float]) -> float:
    """Population (not sample) variance. Returns 0.0 for <2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return float(sum((v - mean) ** 2 for v in values) / len(values))


def adapt_mutation_std(
    base_std: float,
    *,
    fitnesses: Sequence[float],
    gens_since_improve: int,
    cfg: AdaptiveMutationConfig,
) -> Tuple[float, Dict[str, Any]]:
    """Return ``(adjusted_std, diagnostics)``.

    Parameters
    ----------
    base_std
        The std computed by existing curriculum schedules / caps / boosts
        in the pipeline. Adaptive logic only ever *boosts* this value
        (clamped to ``cfg.std_cap``) — it never reduces it, so existing
        fitness caps still apply as lower bounds.
    fitnesses
        Current generation's per-genome fitness values. Used for the
        diversity signal.
    gens_since_improve
        Generations since last best-fitness improvement. Used for the
        plateau signal.
    cfg
        :class:`AdaptiveMutationConfig` bundle.

    Returns
    -------
    adjusted_std
        Possibly boosted std, clamped to ``cfg.std_cap``.
    diagnostics
        Dict with keys ``strategy``, ``base_std``, ``variance``,
        ``plateau_boost_applied``, ``diversity_boost_applied``,
        ``adjusted_std``. Intended for TensorBoard logging and the
        ablation report.
    """
    diagnostics: Dict[str, Any] = {
        "strategy": cfg.strategy,
        "base_std": float(base_std),
        "variance": 0.0,
        "plateau_boost_applied": False,
        "diversity_boost_applied": False,
        "adjusted_std": float(base_std),
    }

    if cfg.strategy == "none":
        return float(base_std), diagnostics

    variance = _population_variance(fitnesses)
    diagnostics["variance"] = variance

    boosted = float(base_std)

    if cfg.strategy in ("diversity", "diversity_plateau"):
        if variance < cfg.diversity_threshold:
            boosted = max(boosted, base_std + cfg.diversity_boost)
            diagnostics["diversity_boost_applied"] = True

    if cfg.strategy in ("plateau", "diversity_plateau"):
        if gens_since_improve >= cfg.plateau_threshold:
            boosted = max(boosted, base_std + cfg.plateau_boost)
            diagnostics["plateau_boost_applied"] = True

    adjusted = min(boosted, cfg.std_cap)
    diagnostics["adjusted_std"] = adjusted
    return adjusted, diagnostics
