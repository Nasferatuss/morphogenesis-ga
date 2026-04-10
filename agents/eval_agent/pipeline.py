"""Eval agent: load a trained checkpoint and replay it in a visualizer."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from core.services.cleanup import resolve_cleanup_ratio
from core.services.factory import (
    build_stability_config,
    prepare_model,
    prepare_world,
)
from core.services.visualizer import build_visualizer
from src.fitness import compute_fitness
from src.simulate import simulate
from src.viz import GifRecorder, render_grid_to_rgb


def play_best(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    target_mask: torch.Tensor,
    seed: int,
) -> None:
    """Load `args.play_best` checkpoint and run it through the simulator with visualization."""
    if args.no_viz:
        raise RuntimeError(
            "--play-best requires visualization. Remove --no-viz to view the replay."
        )

    model_path = Path(args.play_best)
    if not model_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found: {model_path}")

    world_cfg = cfg.get("world", {})
    model_cfg = cfg.get("model", {})
    viz_cfg = cfg.get("viz", {})
    simulate_cfg = cfg.get("simulate", {})
    ga_cfg = cfg.get("ga", {})
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

    t_phase_weights = fitness_cfg.get("t_phase_weights")
    if not isinstance(t_phase_weights, dict):
        t_phase_weights = {}

    steps = int(ga_cfg.get("eval_steps", world_cfg.get("steps", 500)))
    warmup = int(simulate_cfg.get("anti_extinction_warmup_steps", 0))
    die_cap_schedule = simulate_cfg.get("die_cap_schedule")
    die_cap_frac = simulate_cfg.get("post_warmup_die_cap_frac", None)
    late_cleanup_start_frac = max(
        0.0, min(1.0, float(simulate_cfg.get("late_cleanup_start_frac", 0.8)))
    )

    # Load checkpoint first to detect whether it was trained with spatial
    # awareness (presence of spatial_proj keys) and build a matching model.
    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt_has_spatial = any(k.startswith("spatial_proj") for k in checkpoint)
    model_cfg_with_spatial = dict(model_cfg)
    model_cfg_with_spatial["use_position"] = ckpt_has_spatial
    model = prepare_model(model_cfg_with_spatial)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    world = prepare_world(world_cfg, seed)

    viz = build_visualizer(
        cfg=viz_cfg,
        width=world.width,
        height=world.height,
        cell_size=int(viz_cfg.get("cell_px", 24)),
        target_mask=target_mask,
        force_disable=args.no_viz,
    )
    if viz is None and not args.no_viz:
        print("[WARN] Visualization could not be created. Continuing without a window.")

    gif_cfg = viz_cfg.get("gif", {})
    gif_enabled = bool(gif_cfg.get("enabled", False))
    gif_frame_stride = max(
        1,
        int(
            gif_cfg.get(
                "frame_every_steps", max(1, int(viz_cfg.get("render_every_steps", 10)))
            )
        ),
    )
    gif_max_frames = int(gif_cfg.get("max_frames", 300))
    gif_recorder: Optional[GifRecorder] = None
    if gif_enabled:
        gif_recorder = GifRecorder()
        gif_recorder.start_capture(gif_max_frames)

    gif_dir = model_path.parent / "gifs"

    def capture_playback_frame(grid_tensor: torch.Tensor, step_idx: int) -> None:
        if gif_recorder is None:
            return
        try:
            frame = render_grid_to_rgb(grid_tensor, target_mask)
            gif_recorder.capture_frame(frame)
        except Exception as err:  # pragma: no cover - diagnostics only
            print(f"[WARN] Failed to render playback GIF frame at step {step_idx}: {err}")

    sim_result = simulate(
        world=world,
        model=model,
        target_mask=target_mask,
        steps=steps,
        writer=None,
        viz=viz,
        render_every=max(1, int(viz_cfg.get("render_every_steps", 10))),
        device=device,
        anti_extinction_warmup_steps=warmup,
        die_cap_schedule=die_cap_schedule,
        post_warmup_die_cap_frac=die_cap_frac,
        verbose=True,
        frame_capture=capture_playback_frame if gif_recorder is not None else None,
        capture_every=gif_frame_stride,
        late_cleanup_start_frac=late_cleanup_start_frac,
        stability_cfg=stability_cfg if stability_cfg.get("enabled") else None,
    )

    cleanup_ratio = resolve_cleanup_ratio(sim_result, steps, late_cleanup_start_frac)

    if viz is not None:
        viz.close()

    if gif_recorder is not None:
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gif_dir / "play_best.gif"
        gif_recorder.save_gif(gif_path, fps=int(viz_cfg.get("fps", 20)))
        print(f"[PLAY] Saved GIF: {gif_path}")

    model.to("cpu")

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

    metrics = compute_fitness(
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
        expect_t_shape=(target_label == "t"),
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
        f"[PLAY] IoU={metrics['iou']:.4f} | total={metrics['fitness']:.4f} | "
        f"base={metrics['base_score']:.4f} | fp={metrics['fp']:.4f} | fn={metrics['fn']:.4f} | "
        f"alive_bonus={metrics['alive_bonus']:.4f} | ext_penalty={metrics['extinction_penalty']:.4f} | "
        f"area_ab={metrics['area']:.1f}/{target_area:.1f}"
    )
