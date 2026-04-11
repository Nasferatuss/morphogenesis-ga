"""Mini-transfer evaluation for cross-seed candidate robustness.

Periodically during training, a short benchmark runs the top-K candidates
against a separate set of world seeds. The goal is to catch models that
overfit to the training-time RNG sequence — candidates with high single-
seed fitness but low multi-seed robustness get penalised in champion
selection.

This module was extracted from ``agents/train_agent/pipeline.py`` as part
of the closure-decomposition work. The original closure captured 30+
variables from ``train_ga``'s scope; here they are consolidated into
:class:`MiniTransferContext` so the function signature stays manageable
and the module is unit-testable.

⚠️ **Pre-existing behaviour preserved verbatim**
----------------------------------------------------
The original closure had a scoping bug: ``compute_fitness``, the
``best_ious/final_ious/success_flags`` appends, and the summary math all
live OUTSIDE the ``for seed_val in mini_transfer_seeds`` inner loop.
Effectively, only the last seed's ``sim_result`` is evaluated and the
lists always end up with exactly one entry. The bug is preserved here
bit-for-bit so extraction does not change observable training behaviour
(relevant for reproducibility tests). Fixing the bug is tracked as a
follow-up; when done, it MUST be paired with a baseline v3 bump because
it will shift fitness scores for any config that enables mini-transfer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from src.fitness import compute_fitness
from src.ga import clone_model
from src.model import LittleLM
from src.simulate import simulate


@dataclass
class MiniTransferContext:
    """All per-run config needed to evaluate mini-transfer candidates.

    Built once in ``train_ga`` and reused across every mini-transfer
    invocation. All fields are immutable references to config values
    computed at the start of training; no training-loop state
    (population, counters, etc.) lives here — those are passed as
    explicit arguments to :func:`run_mini_transfer_eval`.
    """

    # Environment
    device: torch.device
    world_cfg: Dict[str, Any]

    # Simulation params
    eval_steps: int
    warmup: int
    die_cap_schedule: Optional[Sequence[Dict[str, Any]]]
    die_cap_frac: Optional[float]
    late_cleanup_start_frac: float

    # Fitness weights (scalar)
    alpha_fp: float
    beta_fn: float
    gamma_area: float
    stem_penalty_scale: float
    stem_cleanup_multiplier: float
    stem_corridor_start_scale: float
    stem_corridor_end_scale: float
    t_symmetry_weight: float
    t_trunk_weight: float
    clean_component_target: float
    clean_fp_target: float

    # Coverage / sparse thresholds
    coverage_reward_weight: float
    coverage_target_floor: float
    coverage_penalty_weight: float
    sparse_area_floor: float
    sparse_penalty_weight: float

    # Target
    default_target_mask: torch.Tensor
    default_target_area: float
    default_target_label: str

    # T-phase weight scaling (computed from geometry_late_* boosts)
    transfer_t_weights: Optional[Dict[str, float]]

    # Stability tracking
    stability_cfg_eval: Optional[Dict[str, Any]]

    # Mini-transfer specific
    seeds: Sequence[int]

    # Statistics helpers (injected to avoid hard-coding which simple_mean
    # implementation to use — pipeline.py binds these to agents.train_agent.scoring)
    simple_mean: Callable[[Sequence[float]], float]
    simple_variance: Callable[[Sequence[float]], float]

    # Factory for building a fresh world (injected to stay core-free;
    # pipeline.py binds this to core.services.factory.prepare_world)
    prepare_world_fn: Callable[[Dict[str, Any], int], Any]


def run_mini_transfer_eval(
    candidate_indices: List[int],
    population: List[LittleLM],
    ctx: MiniTransferContext,
) -> Optional[Dict[str, Any]]:
    """Evaluate top candidates across mini-transfer seeds and rank them.

    Returns a summary dict with the winner's stats and the full per-
    candidate results list, or ``None`` when there are no candidates
    or when no evaluations succeeded. This is the 1-for-1 extracted
    body of the former ``run_mini_transfer_eval`` closure in
    ``train_ga``; the observable behaviour (including the pre-existing
    last-seed-only scoping bug) is unchanged.

    Parameters
    ----------
    candidate_indices
        Indices into ``population`` identifying which models to score.
        Empty list → returns ``None`` without any evaluation work.
    population
        The GA population; candidates are indexed out of it.
    ctx
        Bundled per-run config (see :class:`MiniTransferContext`).
    """
    if not candidate_indices:
        return None

    summary_results: List[Dict[str, Any]] = []

    for cand_idx in candidate_indices:
        candidate = population[cand_idx]
        candidate_model = clone_model(candidate)
        candidate_model.to(ctx.device)

        best_ious: List[float] = []
        final_ious: List[float] = []
        success_flags: List[float] = []

        # NOTE (preserved bug): the inner seed loop overwrites
        # ``transfer_world`` / ``sim_result`` without collecting per-seed
        # metrics. Only the last seed's result feeds ``compute_fitness``
        # below. Do NOT "fix" this without a baseline v3 bump — it would
        # change fitness numbers for any config that enables mini-transfer.
        for seed_val in ctx.seeds:
            transfer_world = ctx.prepare_world_fn(ctx.world_cfg, int(seed_val))
            sim_result = simulate(
                world=transfer_world,
                model=candidate_model,
                target_mask=ctx.default_target_mask,
                steps=ctx.eval_steps,
                writer=None,
                viz=None,
                render_every=ctx.eval_steps,
                device=ctx.device,
                anti_extinction_warmup_steps=ctx.warmup,
                die_cap_schedule=ctx.die_cap_schedule,
                post_warmup_die_cap_frac=ctx.die_cap_frac,
                verbose=False,
                late_cleanup_start_frac=ctx.late_cleanup_start_frac,
                stability_cfg=ctx.stability_cfg_eval,
            )

        metrics = compute_fitness(
            transfer_world.grid,
            ctx.default_target_mask,
            alpha_fp=ctx.alpha_fp,
            beta_fn=ctx.beta_fn,
            alive_end=sim_result["alive_end"],
            target_area=ctx.default_target_area,
            gamma_area=ctx.gamma_area,
            stem_penalty_scale=ctx.stem_penalty_scale,
            stem_cleanup_multiplier=ctx.stem_cleanup_multiplier,
            stem_cleanup_ratio=sim_result.get("late_cleanup_ratio", 0.0),
            stem_corridor_start_scale=ctx.stem_corridor_start_scale,
            stem_corridor_end_scale=ctx.stem_corridor_end_scale,
            t_symmetry_weight=ctx.t_symmetry_weight,
            t_trunk_weight=ctx.t_trunk_weight,
            clean_component_weight=ctx.clean_component_target,
            clean_fp_weight=ctx.clean_fp_target,
            stability_metrics=sim_result if ctx.stability_cfg_eval else None,
            t_benchmark_weights=ctx.transfer_t_weights,
            expect_t_shape=(ctx.default_target_label == "t"),
            coverage_weight=ctx.coverage_reward_weight,
            coverage_floor=ctx.coverage_target_floor,
            coverage_penalty_weight=ctx.coverage_penalty_weight,
            sparse_area_floor=ctx.sparse_area_floor,
            sparse_penalty_weight=ctx.sparse_penalty_weight,
            t_phase_gate=1.0,
            t_structure_gate=1.0,
            t_cleanliness_gate=1.0,
            collapse_area_floor=0.0,
            collapse_coverage_floor=0.0,
            collapse_penalty_weight=0.0,
            collapse_bonus_suppression=0.0,
        )

        best_ious.append(float(sim_result.get("best_iou", metrics["iou"])))
        final_ious.append(float(metrics["iou"]))
        success_flags.append(1.0 if metrics.get("stabilized_success") else 0.0)

        candidate_model.to("cpu")

        mean_best = ctx.simple_mean(best_ious)
        mean_final = ctx.simple_mean(final_ious)
        variance = ctx.simple_variance(final_ious)
        success_rate = ctx.simple_mean(success_flags)

        mini_score = (
            mean_final
            + 0.5 * mean_best
            + 0.5 * success_rate
            - 0.3 * variance
        )

        summary_results.append(
            {
                "candidate_index": cand_idx,
                "mean_best_iou": mean_best,
                "mean_final_iou": mean_final,
                "variance": variance,
                "success_rate": success_rate,
                "mini_score": mini_score,
                "evaluations": len(ctx.seeds),
            }
        )

    if not summary_results:
        return None

    best_result = max(summary_results, key=lambda entry: entry["mini_score"])
    summary = dict(best_result)
    summary["candidates"] = summary_results
    return summary
