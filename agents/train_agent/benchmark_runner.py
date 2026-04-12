"""Benchmark evaluation runner for GA training.

Extracted from ``agents/train_agent/pipeline.py`` as part of the
closure-decomposition work (Round 3). The original code lived as three
nested closures inside ``train_ga``:

- ``build_benchmark_mode(mode_name, mode_cfg)`` → config builder
- ``run_benchmark_mode(mode_state, reason)`` → single-mode evaluation
- ``execute_benchmarks(triggered)`` → batch dispatcher

The closures captured ~26 variables from ``train_ga`` scope, 6 of which
were ``nonlocal`` (mutated). Those 6 are consolidated into
:class:`BenchmarkState` (mutable, passed by reference).
Read-only config is bundled in :class:`BenchmarkContext` (frozen-ish,
built once per ``train_ga`` invocation).

``build_benchmark_mode`` remains as a pure function (no captured state,
only reads ``train_stability_cfg`` and top-level benchmark config).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from src.benchmark import BenchmarkSummary, evaluate_t_benchmark
from src.model import LittleLM


# ---------------------------------------------------------------------------
# Mutable state — 6 former nonlocals
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkState:
    """Mutable state that ``run_benchmark_mode`` writes back into the
    training loop. Corresponds to the 6 ``nonlocal`` declarations of
    the original closure.

    Passed **by reference** so mutations are visible to the caller
    (``train_ga``). This is the replacement for the ``nonlocal`` keyword
    — instead of implicitly sharing a local variable, we explicitly
    share a mutable container.
    """

    benchmark_passed: bool = False
    stop_training: bool = False
    champion_score: Optional[float] = None
    champion_mode: Optional[str] = None
    champion_details: Optional[Dict[str, Any]] = None
    champion_source: Optional[str] = None


# ---------------------------------------------------------------------------
# Read-only context — ~20 former free variables
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkContext:
    """Read-only config and object references needed by the benchmark
    runner. Built once in ``train_ga`` after all config parsing is done.

    All mutable runtime state (``global_gen``, ``latest_candidate_meta``,
    ``benchmark_results_map``, etc.) is accessed through explicit
    arguments to :func:`run_benchmark_mode` rather than being stored
    here. This keeps the context immutable-in-spirit even though
    Python dataclasses don't enforce deep immutability.
    """

    # Paths
    run_dir: Path
    latest_candidate_path: Path
    champion_path: Path
    benchmark_dir: Path

    # Model factory config
    model_cfg: Dict[str, Any]

    # Simulation / fitness config (passed through to evaluate_t_benchmark)
    world_cfg: Dict[str, Any]
    simulate_cfg: Dict[str, Any]
    fitness_cfg: Dict[str, Any]

    # Target
    default_target_mask: torch.Tensor
    default_target_area: float

    # Device
    device: torch.device

    # T-phase weights for stability evaluation
    t_phase_weights: Optional[Dict[str, float]]

    # TensorBoard writer (for per-gen scalar logging)
    writer: SummaryWriter

    # Target progress dict (mutable dict — mutations visible to caller)
    target_progress: Dict[str, float]

    # Benchmark history list (mutable — appends visible to caller)
    benchmark_history: List[Dict[str, Any]]

    # Benchmark results map (mutable — mutations visible to caller)
    benchmark_results_map: Dict[str, Dict[str, Any]]

    # Injected callables
    prepare_model_fn: Callable[[Dict[str, Any]], LittleLM]
    compute_benchmark_score_fn: Callable[[BenchmarkSummary], float]
    update_status_fn: Callable[[str, int], None]


# ---------------------------------------------------------------------------
# run_benchmark_mode (the big closure, now a top-level function)
# ---------------------------------------------------------------------------


def run_benchmark_mode(
    mode_state: Dict[str, Any],
    reason: str,
    *,
    ctx: BenchmarkContext,
    state: BenchmarkState,
    global_gen: int,
    latest_candidate_meta: Dict[str, Any],
) -> None:
    """Run one benchmark mode and update shared state.

    This is the 1-for-1 extraction of the former
    ``train_ga::run_benchmark_mode`` closure. Observable behaviour is
    preserved verbatim.
    """
    best_checkpoint_path = ctx.run_dir / "best.pt"

    benchmark_model_path: Optional[Path] = None
    benchmark_model_source = None
    benchmark_model_phase = None
    benchmark_model_generation = None

    if ctx.latest_candidate_path.exists():
        benchmark_model_path = ctx.latest_candidate_path
        benchmark_model_source = "latest_candidate"
        benchmark_model_phase = latest_candidate_meta.get("phase")
        benchmark_model_generation = latest_candidate_meta.get("generation")
    elif best_checkpoint_path.exists():
        benchmark_model_path = best_checkpoint_path
        benchmark_model_source = "best"
    elif ctx.champion_path.exists():
        benchmark_model_path = ctx.champion_path
        benchmark_model_source = "champion"

    if benchmark_model_path is None or not benchmark_model_path.exists():
        print("[WARN] Cannot run benchmark yet; candidate checkpoint missing.")
        return

    try:
        checkpoint = torch.load(benchmark_model_path, map_location="cpu")
    except Exception as err:  # pragma: no cover - IO guard
        print(f"[WARN] Failed to load checkpoint for benchmark: {err}")
        return

    bench_model = ctx.prepare_model_fn(ctx.model_cfg)
    bench_model.load_state_dict(checkpoint)
    bench_model.eval()

    mode_dir = ctx.benchmark_dir / mode_state["mode"]

    bench_run = evaluate_t_benchmark(
        bench_model,
        benchmark_cfg=mode_state["config"],
        world_cfg=ctx.world_cfg,
        simulate_cfg=ctx.simulate_cfg,
        fitness_cfg=ctx.fitness_cfg,
        target_mask=ctx.default_target_mask,
        target_area=ctx.default_target_area,
        device=ctx.device,
        output_dir=mode_dir,
        stability_weights=ctx.t_phase_weights if ctx.t_phase_weights else None,
    )

    bench_model.to("cpu")

    mode_state["last_gen"] = global_gen

    summary_dict = bench_run.summary.to_dict()

    ctx.benchmark_results_map[mode_state["mode"]] = summary_dict

    improved = False

    if (
        bench_run.summary.success_rate_stabilized
        > mode_state["best_stabilized"] + mode_state["config"].plateau_min_delta
    ):
        mode_state["best_stabilized"] = bench_run.summary.success_rate_stabilized
        improved = True

    if (
        bench_run.summary.mean_final_iou
        > mode_state["best_final_iou"] + mode_state["config"].plateau_min_delta
    ):
        mode_state["best_final_iou"] = bench_run.summary.mean_final_iou
        improved = True

    if improved:
        mode_state["plateau_runs"] = 0
    else:
        mode_state["plateau_runs"] += 1

    history_dir = mode_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    reason_slug = reason.replace(" ", "_")
    snapshot_dir = history_dir / f"gen_{global_gen:05d}_{reason_slug}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    summary_snapshot = snapshot_dir / bench_run.summary_path.name
    per_seed_snapshot = snapshot_dir / bench_run.per_seed_path.name

    shutil.copyfile(bench_run.summary_path, summary_snapshot)
    shutil.copyfile(bench_run.per_seed_path, per_seed_snapshot)

    seeds_src_dir = bench_run.output_dir / "seeds"
    seeds_snapshot_dir = snapshot_dir / "seeds"
    seeds_history_path = seeds_src_dir

    if seeds_src_dir.exists():
        shutil.copytree(seeds_src_dir, seeds_snapshot_dir, dirs_exist_ok=True)
        seeds_history_path = seeds_snapshot_dir

    ctx.benchmark_history.append(
        {
            "generation": global_gen,
            "mode": mode_state["mode"],
            "reason": reason,
            "verdict": bench_run.summary.verdict,
            "summary": summary_dict,
            "summary_path": str(summary_snapshot),
            "per_seed_path": str(per_seed_snapshot),
            "latest_summary_path": str(bench_run.summary_path),
            "latest_per_seed_path": str(bench_run.per_seed_path),
            "snapshot_dir": str(snapshot_dir),
            "seeds_path": str(seeds_history_path.resolve()),
            "model_path": str(benchmark_model_path.resolve()),
            "model_source": benchmark_model_source,
            "model_generation": benchmark_model_generation,
            "model_phase": benchmark_model_phase,
        }
    )

    ctx.writer.add_scalar(
        f"benchmark/{mode_state['mode']}/mean_best_iou",
        bench_run.summary.mean_best_iou,
        global_step=global_gen,
    )

    ctx.writer.add_scalar(
        f"benchmark/{mode_state['mode']}/mean_final_iou",
        bench_run.summary.mean_final_iou,
        global_step=global_gen,
    )

    score = ctx.compute_benchmark_score_fn(bench_run.summary)
    current_champion_score = (
        state.champion_score
        if state.champion_score is not None
        else float("-inf")
    )

    if (
        score >= mode_state["champion_min_score"]
        and score > current_champion_score + 1e-6
    ):
        shutil.copyfile(best_checkpoint_path, ctx.champion_path)
        state.champion_score = score
        state.champion_mode = mode_state["mode"]
        state.champion_source = f"benchmark:{mode_state['mode']}"
        state.champion_details = {
            "score": state.champion_score,
            "source": state.champion_source,
            "generation": global_gen,
            "benchmark_summary": summary_dict,
            "model_path": str(benchmark_model_path.resolve()),
            "model_source": benchmark_model_source,
        }

    if mode_state["mode"] == "target":
        ctx.target_progress["best_stabilized"] = max(
            ctx.target_progress["best_stabilized"],
            bench_run.summary.success_rate_stabilized,
        )
        ctx.target_progress["best_mean_final_iou"] = max(
            ctx.target_progress["best_mean_final_iou"],
            bench_run.summary.mean_final_iou,
        )

        if bench_run.summary.verdict == "benchmark_passed":
            state.benchmark_passed = True
            state.stop_training = True
            ctx.update_status_fn("benchmark_passed", global_gen)
        else:
            ctx.update_status_fn("benchmark_failed", global_gen)


# ---------------------------------------------------------------------------
# execute_benchmarks (thin dispatcher, also extracted)
# ---------------------------------------------------------------------------


def execute_benchmarks(
    triggered: List[Tuple[Dict[str, Any], str]],
    *,
    benchmark_enabled: bool,
    ctx: BenchmarkContext,
    state: BenchmarkState,
    global_gen: int,
    latest_candidate_meta: Dict[str, Any],
) -> None:
    """Run all triggered benchmark modes sequentially.

    Thin wrapper that was a separate closure in ``train_ga``. Extracted
    alongside ``run_benchmark_mode`` for completeness.
    """
    if not benchmark_enabled or not triggered:
        return

    ctx.update_status_fn("benchmark_pending", global_gen)
    ctx.update_status_fn("benchmark_running", global_gen)

    for mode_state, reason in triggered:
        run_benchmark_mode(
            mode_state,
            reason,
            ctx=ctx,
            state=state,
            global_gen=global_gen,
            latest_candidate_meta=latest_candidate_meta,
        )

    if not state.benchmark_passed:
        ctx.update_status_fn("training", global_gen)
