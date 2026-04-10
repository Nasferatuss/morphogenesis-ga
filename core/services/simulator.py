"""Single-shot simulation service.

Runs one simulation of a world+model against a target mask, computes fitness,
logs to TensorBoard, and prints the result line. Used by the ``run_train.py``
CLI when GA training and checkpoint replay are both disabled.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

import torch

from core.services.cleanup import resolve_cleanup_ratio
from core.services.factory import (
    build_stability_config,
    prepare_model,
    prepare_world,
)
from core.services.visualizer import build_visualizer
from core.services.writer import build_writer
from src.fitness import compute_fitness
from src.simulate import simulate


def run_single_simulation(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    target_mask: torch.Tensor,
    seed: int,
) -> None:
    """Run one end-to-end simulation + fitness evaluation + TB logging."""
    world_cfg = cfg.get("world", {})
    model_cfg = cfg.get("model", {})
    viz_cfg = cfg.get("viz", {})
    simulate_cfg = cfg.get("simulate", {})
    fitness_cfg = cfg.get("fitness", {})
    benchmark_cfg = cfg.get("benchmark", {})
    target_cfg = cfg.get("target", {})
    target_label = str(target_cfg.get("name", "T")).lower()

    empty_collapse_cfg = fitness_cfg.get("empty_collapse", {})
    empty_collapse_area_floor = float(empty_collapse_cfg.get("area_floor", 0.0))
    empty_collapse_coverage_floor = float(empty_collapse_cfg.get("coverage_floor", 0.0))
    empty_collapse_penalty_weight = float(empty_collapse_cfg.get("penalty_weight", 0.0))
    empty_collapse_suppress_scale = float(empty_collapse_cfg.get("suppress_scale", 0.0))

    stability_source = (
        benchmark_cfg.get("stability")
        or benchmark_cfg.get("baseline")
        or benchmark_cfg.get("target")
        or benchmark_cfg
    )
    stability_cfg = build_stability_config(stability_source)

    t_phase_weights = fitness_cfg.get("t_phase_weights") if target_label == "t" else None

    steps_cfg = int(world_cfg.get("steps", 0))
    steps = steps_cfg if args.steps is None else max(0, args.steps)
    if args.steps is not None:
        print(f"[INFO] Overriding steps from config ({steps_cfg}) -> {steps}")

    alpha_fp = float(fitness_cfg.get("alpha_fp", 1.0))
    beta_fn = float(fitness_cfg.get("beta_fn", 0.5))
    gamma_area = float(fitness_cfg.get("gamma_area", 0.15))
    stem_penalty_scale = float(fitness_cfg.get("stem_penalty_scale", 0.05))
    stem_cleanup_multiplier = float(fitness_cfg.get("stem_cleanup_multiplier", 1.0))
    stem_corridor_start_scale = float(fitness_cfg.get("stem_corridor_start_scale", 2.0))
    stem_corridor_end_scale = float(fitness_cfg.get("stem_corridor_end_scale", 1.1))
    t_symmetry_weight = float(fitness_cfg.get("t_symmetry_weight", 0.0))
    t_trunk_weight = float(fitness_cfg.get("t_trunk_weight", 0.0))

    coverage_cfg = fitness_cfg.get("coverage", {})
    coverage_target_floor = float(coverage_cfg.get("target_floor", 0.55))
    coverage_reward_weight = float(coverage_cfg.get("reward_weight", 0.6))
    coverage_penalty_weight = float(coverage_cfg.get("penalty_weight", 1.0))
    sparse_area_floor = float(coverage_cfg.get("area_ratio_floor", 0.6))
    sparse_penalty_weight = float(coverage_cfg.get("area_penalty_weight", 0.8))

    target_area = float(target_mask.to(dtype=torch.bool).sum().item())

    world = prepare_world(world_cfg, seed)
    model = prepare_model(model_cfg).to(device)

    viz = build_visualizer(
        cfg=viz_cfg,
        width=world.width,
        height=world.height,
        cell_size=int(viz_cfg.get("cell_px", 24)),
        target_mask=target_mask,
        force_disable=args.no_viz,
    )

    writer, _, _ = build_writer(cfg.get("logging", {}).get("run_name", "baseline"))
    writer.add_scalar("stats/target_area", target_area, 0)

    render_every = max(1, int(viz_cfg.get("render_every_steps", 10)))
    warmup = int(simulate_cfg.get("anti_extinction_warmup_steps", 0))
    die_cap_schedule = simulate_cfg.get("die_cap_schedule")
    die_cap_frac = simulate_cfg.get("post_warmup_die_cap_frac", None)
    late_cleanup_start_frac = max(
        0.0, min(1.0, float(simulate_cfg.get("late_cleanup_start_frac", 0.8)))
    )

    sim_result = simulate(
        world=world,
        model=model,
        target_mask=target_mask,
        steps=steps,
        writer=writer,
        viz=viz,
        render_every=render_every,
        device=device,
        anti_extinction_warmup_steps=warmup,
        die_cap_schedule=die_cap_schedule,
        post_warmup_die_cap_frac=die_cap_frac,
        verbose=True,
        late_cleanup_start_frac=late_cleanup_start_frac,
        stability_cfg=stability_cfg if stability_cfg.get("enabled") else None,
    )

    cleanup_ratio = resolve_cleanup_ratio(sim_result, steps, late_cleanup_start_frac)

    fitness_metrics = compute_fitness(
        world.grid,
        target_mask,
        alpha_fp=alpha_fp,
        beta_fn=beta_fn,
        alive_end=sim_result["alive_end"],
        target_area=target_area,
        gamma_area=gamma_area,
        stem_penalty_scale=stem_penalty_scale,
        stem_cleanup_multiplier=stem_cleanup_multiplier,
        stem_cleanup_ratio=cleanup_ratio,
        stem_corridor_start_scale=stem_corridor_start_scale,
        stem_corridor_end_scale=stem_corridor_end_scale,
        t_symmetry_weight=t_symmetry_weight,
        t_trunk_weight=t_trunk_weight,
        stability_metrics=sim_result if stability_cfg.get("enabled") else None,
        t_benchmark_weights=t_phase_weights,
        expect_t_shape=target_label == "t",
        coverage_weight=coverage_reward_weight,
        coverage_floor=coverage_target_floor,
        coverage_penalty_weight=coverage_penalty_weight,
        sparse_area_floor=sparse_area_floor,
        sparse_penalty_weight=sparse_penalty_weight,
        t_phase_gate=1.0,
        t_structure_gate=1.0,
        t_cleanliness_gate=1.0,
        collapse_area_floor=empty_collapse_area_floor if target_label == "t" else 0.0,
        collapse_coverage_floor=empty_collapse_coverage_floor if target_label == "t" else 0.0,
        collapse_penalty_weight=empty_collapse_penalty_weight if target_label == "t" else 0.0,
        collapse_bonus_suppression=empty_collapse_suppress_scale if target_label == "t" else 0.0,
    )

    print(
        "[RESULT] IoU={:.4f} | base={:.4f} | fp={:.4f} | fn={:.4f} | "
        "alive_bonus={:.4f} | ext_penalty={:.4f} | area_ab={:.1f} | target_area={:.1f}".format(
            fitness_metrics["iou"],
            fitness_metrics["base_score"],
            fitness_metrics["fp"],
            fitness_metrics["fn"],
            fitness_metrics["alive_bonus"],
            fitness_metrics["extinction_penalty"],
            fitness_metrics["area"],
            target_area,
        )
    )
    print(
        f"[RESULT] Divisions={sim_result['total_divisions']} | "
        f"Deaths={sim_result['total_deaths']} | Alive end={sim_result['alive_end']}"
    )

    if viz is not None:
        viz.render(world.grid, target_mask, step=sim_result["steps"], iou=sim_result["iou"])
        viz.close()
    writer.close()
