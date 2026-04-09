
from __future__ import annotations

from typing import Callable, Dict

import torch


def make_target_T(height: int, width: int) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool)
    bar_row = max(1, height // 6)
    bar_thickness = max(1, height // 10)
    bar_start = max(0, width // 8)
    bar_end = width - bar_start
    mask[bar_row : bar_row + bar_thickness, bar_start:bar_end] = True

    stem_width = max(1, width // 10)
    center = width // 2
    half = max(0, stem_width // 2)
    col_start = max(0, center - half)
    col_end = min(width, center + half + 1)
    mask[bar_row:, col_start:col_end] = True
    return mask


def make_target_cross(height: int, width: int) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool)
    arm_thickness = max(1, min(height, width) // 6)
    center_y = height // 2
    center_x = width // 2
    row_start = max(0, center_y - arm_thickness // 2)
    row_end = min(height, center_y + (arm_thickness + 1) // 2)
    col_start = max(0, center_x - arm_thickness // 2)
    col_end = min(width, center_x + (arm_thickness + 1) // 2)
    mask[row_start:row_end, :] = True
    mask[:, col_start:col_end] = True
    return mask


_TARGET_FACTORIES: Dict[str, Callable[[int, int], torch.Tensor]] = {
    "t": make_target_T,
    "tee": make_target_T,
    "cross": make_target_cross,
}


def make_target(name: str, height: int, width: int) -> torch.Tensor:
    if not name:
        key = "t"
    else:
        key = str(name).lower()
    factory = _TARGET_FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"Unknown target mask: {name}")
    return factory(height, width)


def get_target(name: str, width: int, height: int) -> torch.Tensor:
    return make_target(name, height, width)
