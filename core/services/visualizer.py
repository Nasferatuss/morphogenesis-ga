"""Visualizer factory for pygame-based world rendering."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def build_visualizer(
    cfg: Dict[str, Any],
    width: int,
    height: int,
    cell_size: int,
    target_mask: torch.Tensor,
    force_disable: bool,
) -> Optional[Any]:
    """Construct a pygame Visualizer from config, or None if disabled/unsupported."""
    viz_mode = str(cfg.get("mode", "pygame")).lower()
    if force_disable:
        print("[INFO] Visualization disabled via flag.")
        return None
    if viz_mode != "pygame":
        print(f"[INFO] Visualization mode '{viz_mode}' unsupported. Disabled.")
        return None

    from src.viz import Visualizer

    fps = int(cfg.get("fps", 0))
    return Visualizer(
        width=width,
        height=height,
        cell_size=cell_size,
        target_mask=target_mask,
        fps=fps,
    )
