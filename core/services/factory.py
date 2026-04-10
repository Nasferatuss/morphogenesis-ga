"""Factory functions for creating World and LittleLM instances from config dicts."""
from __future__ import annotations

from typing import Any, Dict

from src.model import LittleLM
from src.world import World


def prepare_world(world_cfg: Dict[str, Any], seed: int) -> World:
    """Create and reset a World instance from config."""
    world = World(
        width=int(world_cfg.get("width", 15)),
        height=int(world_cfg.get("height", 15)),
        init_cells=int(world_cfg.get("init_cells", 50)),
        seed=seed,
    )
    world.reset()
    return world


def prepare_model(model_cfg: Dict[str, Any]) -> LittleLM:
    """Create a LittleLM model from config in eval mode.

    Reads ``use_position`` flag from config (default False for backwards
    compatibility with legacy checkpoints). Set ``use_position: true`` in
    YAML config to enable spatial awareness.
    """
    model = LittleLM(
        embed_dim=int(model_cfg.get("embed_dim", 32)),
        num_heads=int(model_cfg.get("heads", 4)),
        use_position=bool(model_cfg.get("use_position", False)),
    )
    model.eval()
    return model


def build_stability_config(benchmark_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build stability tracking config from benchmark section."""
    reach = float(benchmark_cfg.get("reach_threshold_iou", 0.7))
    stabilize = float(benchmark_cfg.get("stabilize_threshold_iou", reach))
    hold_steps = int(benchmark_cfg.get("hold_window_steps", 0))
    enabled = bool(benchmark_cfg.get("enabled", False)) or reach > 0.0 or hold_steps > 0
    return {
        "enabled": enabled,
        "reach_threshold_iou": reach,
        "stabilize_threshold_iou": stabilize,
        "hold_window_steps": hold_steps,
    }
