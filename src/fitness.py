from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .world import CELL_A, CELL_B, CELL_EMPTY, CELL_STEM, NEIGHBOR_OFFSETS


def compute_iou(grid: torch.Tensor, target_mask: torch.Tensor) -> float:
    shape_mask = torch.logical_or(grid == CELL_A, grid == CELL_B)
    target_bool = target_mask.to(dtype=torch.bool, device=shape_mask.device)
    intersection = torch.logical_and(shape_mask, target_bool).sum().item()
    union = torch.logical_or(shape_mask, target_bool).sum().item()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _connected_components(mask: torch.Tensor) -> Tuple[int, int]:
    mask_bool = mask.to(dtype=torch.bool).cpu()
    height, width = mask_bool.shape
    visited = torch.zeros_like(mask_bool, dtype=torch.bool)
    components = 0
    largest_area = 0
    offsets = tuple(NEIGHBOR_OFFSETS)

    for y in range(height):
        for x in range(width):
            if not mask_bool[y, x] or visited[y, x]:
                continue
            components += 1
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask_bool[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            largest_area = max(largest_area, area)
    return components, largest_area


def _center_of_mass(mask: torch.Tensor, height: int, width: int) -> Tuple[float, float]:
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return (float(height - 1) / 2.0, float(width - 1) / 2.0)
    y_mean = float(coords[:, 0].to(dtype=torch.float32).mean().item())
    x_mean = float(coords[:, 1].to(dtype=torch.float32).mean().item())
    return y_mean, x_mean


def _normalized_distance(
    point_a: Tuple[float, float], point_b: Tuple[float, float], height: int, width: int
) -> float:
    dy = point_a[0] - point_b[0]
    dx = point_a[1] - point_b[1]
    diag = math.hypot(height, width)
    if diag <= 0:
        return 0.0
    return float(math.hypot(dy, dx) / diag)




def _compute_t_diagnostics(
    ab_mask: torch.Tensor,
    target_mask: torch.Tensor,
    grid: Optional[torch.Tensor] = None,
    stem_trunk_presence: bool = False,
) -> Tuple[float, float]:
    """Compute T-shape symmetry and trunk continuity scores.

    When *grid* is provided and *stem_trunk_presence* is True, stem cells
    (CELL_STEM) in the trunk column count as full presence (same as A/B).
    This gives the GA a strong binary signal: "anything in the trunk column
    is good", encouraging DIVIDE_S → stem placement as a first step toward
    trunk construction. GA cannot optimize subtle weighted gradients within
    its generational budget, so binary presence outperforms weighted credit.
    """
    target_bool = target_mask.to(dtype=torch.bool)
    if target_bool.sum().item() <= 0:
        return 0.0, 0.0
    height, width = target_bool.shape
    row_counts = target_bool.sum(dim=1)
    max_row_val = row_counts.max()
    if max_row_val.item() <= 0:
        return 0.0, 0.0
    bar_rows = torch.nonzero(row_counts == max_row_val, as_tuple=False)
    if bar_rows.numel() == 0:
        return 0.0, 0.0
    bar_row = int(bar_rows[0].item())
    bar_indices = torch.nonzero(target_bool[bar_row], as_tuple=False).flatten()
    if bar_indices.numel() == 0:
        return 0.0, 0.0
    bar_min = int(bar_indices.min().item())
    bar_max = int(bar_indices.max().item())
    if bar_max < bar_min:
        return 0.0, 0.0
    bar_segment = ab_mask[bar_row, bar_min : bar_max + 1].to(dtype=torch.float32)
    if bar_segment.numel() == 0:
        symmetry_score = 0.0
    else:
        flipped = torch.flip(bar_segment, dims=[0])
        diff = torch.abs(bar_segment - flipped)
        symmetry_score = float(1.0 - float(diff.mean().item()))
    symmetry_score = max(0.0, min(1.0, symmetry_score))

    if bar_row + 1 >= height:
        return symmetry_score, 0.0
    lower_target = target_bool[bar_row + 1 :, :]
    if lower_target.numel() == 0 or lower_target.sum().item() <= 0:
        return symmetry_score, 0.0
    col_counts = lower_target.sum(dim=0)
    max_col_val = col_counts.max()
    if max_col_val.item() <= 0:
        return symmetry_score, 0.0
    trunk_cols = torch.nonzero(col_counts == max_col_val, as_tuple=False)
    if trunk_cols.numel() == 0:
        return symmetry_score, 0.0
    trunk_col = int(trunk_cols[0].item())
    trunk_target = lower_target[:, trunk_col]
    trunk_len = int(trunk_target.sum().item())
    if trunk_len <= 0:
        return symmetry_score, 0.0

    # Binary presence vector: A/B = True, stem = True (if enabled)
    pred_trunk_ab = ab_mask[bar_row + 1 :, trunk_col]
    if stem_trunk_presence and grid is not None:
        pred_trunk_stem = (grid[bar_row + 1 :, trunk_col] == CELL_STEM)
        trunk_present = torch.logical_or(pred_trunk_ab, pred_trunk_stem)
    else:
        trunk_present = pred_trunk_ab

    trunk_target_bool = trunk_target.to(dtype=torch.bool)
    coverage = 0.0
    if trunk_len > 0:
        coverage = float(torch.logical_and(trunk_present, trunk_target_bool).sum().item()) / float(trunk_len)

    # Continuity: longest consecutive run of presence in trunk target cells
    presence_list = trunk_present.to("cpu").tolist()
    target_list = trunk_target_bool.to("cpu").tolist()
    longest_run = 0
    current_run = 0
    for pres_val, target_val in zip(presence_list, target_list):
        if not target_val:
            continue
        if pres_val:
            current_run += 1
        else:
            if current_run > longest_run:
                longest_run = current_run
            current_run = 0
    if current_run > longest_run:
        longest_run = current_run
    continuity = float(longest_run) / float(trunk_len) if trunk_len > 0 else 0.0
    trunk_score = max(0.0, min(1.0, 0.5 * (coverage + continuity)))
    return symmetry_score, trunk_score


def _derive_stability_scores(
    stability_metrics: Optional[Dict[str, float]]
) -> Dict[str, float]:
    if not stability_metrics:
        return {
            "hold_score": 0.0,
            "stability_score": 0.0,
            "collapse_penalty": 0.0,
            "hold_length": 0.0,
            "hold_window": 0.0,
            "stabilized_success": False,
            "first_reach_step": None,
            "stability_failures": 0.0,
        }
    hold_length = float(max(0.0, stability_metrics.get("hold_length_achieved", 0.0) or 0.0))
    hold_window = float(max(0.0, stability_metrics.get("hold_window_steps", 0.0) or 0.0))
    stabilized_success = bool(stability_metrics.get("stabilized_success", False))
    best_iou = float(max(0.0, stability_metrics.get("best_iou", 0.0) or 0.0))
    stabilize_threshold = float(
        max(1e-6, stability_metrics.get("stabilize_threshold_iou", 1.0) or 1.0)
    )
    first_reach_step = stability_metrics.get("first_reach_step")
    stability_failures = float(max(0.0, stability_metrics.get("stability_failures", 0.0) or 0.0))

    if hold_window <= 0.0:
        hold_score = 1.0 if stabilized_success else 0.0
    else:
        hold_score = min(1.0, hold_length / hold_window)

    if stabilized_success:
        stability_score = 1.0
    else:
        base_score = min(1.0, best_iou / stabilize_threshold)
        stability_score = 0.5 * hold_score + 0.5 * base_score

    collapse_penalty = 0.0
    if first_reach_step is not None and not stabilized_success:
        collapse_penalty = max(0.0, 1.0 - hold_score)
        failure_norm = min(1.0, stability_failures / max(1.0, hold_window or 1.0))
        collapse_penalty *= 1.0 + 0.5 * failure_norm

    return {
        "hold_score": float(hold_score),
        "stability_score": float(stability_score),
        "collapse_penalty": float(collapse_penalty),
        "hold_length": hold_length,
        "hold_window": hold_window,
        "stabilized_success": stabilized_success,
        "first_reach_step": first_reach_step,
        "stability_failures": stability_failures,
    }


def _compute_t_structure_breakdown(
    ab_mask: torch.Tensor,
    target_bool: torch.Tensor,
    *,
    area_ratio: float,
    num_components: float,
) -> Dict[str, float]:
    breakdown = {
        "top_bar_coverage": 0.0,
        "top_bar_symmetry": 0.0,
        "top_bar_alignment_score": 0.0,
        "top_bar_length_ratio": 0.0,
        "trunk_coverage": 0.0,
        "trunk_alignment_score": 0.0,
        "junction_score": 0.0,
        "bar_endpoint_clutter": 0.0,
        "excess_below_ratio": 0.0,
        "excess_side_ratio": 0.0,
        "trunk_extra_ratio": 0.0,
        "side_clutter_ratio": 0.0,
        "off_axis_mass_ratio": 0.0,
        "area_overshoot": max(0.0, area_ratio - 1.0),
        "area_undershoot": max(0.0, 1.0 - area_ratio),
        "disconnected_fragments": max(0.0, num_components - 1.0),
    }
    target_area = float(target_bool.sum().item())
    if target_area <= 0:
        return breakdown

    height, width = target_bool.shape
    coords = torch.nonzero(target_bool, as_tuple=False)
    int(coords[:, 0].min().item()) if coords.numel() > 0 else 0
    y_max = int(coords[:, 0].max().item()) if coords.numel() > 0 else height - 1
    x_min = int(coords[:, 1].min().item()) if coords.numel() > 0 else 0
    x_max = int(coords[:, 1].max().item()) if coords.numel() > 0 else width - 1

    row_counts = target_bool.sum(dim=1)
    bar_row = int(torch.argmax(row_counts).item()) if row_counts.numel() > 0 else 0
    bar_indices = torch.nonzero(target_bool[bar_row], as_tuple=False).flatten()
    if bar_indices.numel() > 0:
        bar_left = int(bar_indices.min().item())
        bar_right = int(bar_indices.max().item())
    else:
        bar_left, bar_right = x_min, x_max
    max(1.0, float(bar_right - bar_left + 1))
    bar_mask = torch.zeros_like(target_bool)
    bar_mask[bar_row, bar_left : bar_right + 1] = target_bool[bar_row, bar_left : bar_right + 1]
    predicted_bar_overlap = torch.logical_and(ab_mask, bar_mask).sum().item()
    breakdown["top_bar_coverage"] = float(predicted_bar_overlap / max(1.0, bar_mask.sum().item()))

    bar_segment = ab_mask[bar_row, bar_left : bar_right + 1].to(dtype=torch.float32)
    if bar_segment.numel() > 0:
        flipped = torch.flip(bar_segment, dims=[0])
        diff = torch.abs(bar_segment - flipped)
        breakdown["top_bar_symmetry"] = float(1.0 - float(diff.mean().item()))
    breakdown["top_bar_symmetry"] = max(0.0, min(1.0, breakdown["top_bar_symmetry"]))

    pred_rows_sum = ab_mask.sum(dim=1)
    if pred_rows_sum.numel() > 0:
        pred_top_row_indices = torch.nonzero(pred_rows_sum > 0, as_tuple=False)
        if pred_top_row_indices.numel() > 0:
            pred_top_row = int(pred_top_row_indices.min().item())
            alignment_gap = abs(pred_top_row - bar_row) / max(1.0, float(height))
            breakdown["top_bar_alignment_score"] = max(0.0, 1.0 - alignment_gap * 3.0)

    pred_bar_row = ab_mask[bar_row]
    pred_bar_len = int(pred_bar_row.sum().item())
    target_bar_len = max(1.0, float(bar_right - bar_left + 1))
    breakdown["top_bar_length_ratio"] = min(1.5, pred_bar_len / target_bar_len if target_bar_len > 0 else 0.0)
    if pred_bar_len > 0:
        pred_bar_center = torch.nonzero(pred_bar_row, as_tuple=False).to(dtype=torch.float32)
        pred_bar_center = float(pred_bar_center.mean().item()) if pred_bar_center.numel() > 0 else float(bar_left + bar_right) / 2.0
    else:
        pred_bar_center = float(bar_left + bar_right) / 2.0
    target_bar_center = float(bar_left + bar_right) / 2.0
    bar_center_offset = abs(pred_bar_center - target_bar_center) / max(1.0, float(width))
    breakdown["top_bar_alignment_score"] *= max(0.0, 1.0 - 4.0 * bar_center_offset)

    endpoint_clutter = 0.0
    if bar_indices.numel() > 0:
        for endpoint in (bar_left, bar_right):
            x0 = max(0, endpoint - 1)
            x1 = min(width, endpoint + 2)
            y0 = max(0, bar_row - 1)
            y1 = min(height, bar_row + 2)
            pred_patch = ab_mask[y0:y1, x0:x1]
            target_patch = target_bool[y0:y1, x0:x1]
            clutter = torch.logical_and(pred_patch, torch.logical_not(target_patch)).sum().item()
            endpoint_clutter += clutter
    breakdown["bar_endpoint_clutter"] = float(endpoint_clutter / max(1.0, target_area))

    trunk_col = None
    if bar_row + 1 < height:
        lower_target = target_bool[bar_row + 1 :, :]
        col_counts = lower_target.sum(dim=0)
        if col_counts.numel() > 0:
            trunk_col = int(torch.argmax(col_counts).item())
            trunk_mask = torch.zeros_like(target_bool)
            trunk_mask[bar_row + 1 :, trunk_col] = lower_target[:, trunk_col]
            trunk_target_count = max(1.0, float(trunk_mask.sum().item()))
            trunk_overlap = torch.logical_and(ab_mask, trunk_mask).sum().item()
            breakdown["trunk_coverage"] = float(trunk_overlap / trunk_target_count)
            trunk_mask_bool = trunk_mask.to(dtype=torch.bool)
            trunk_target_col = trunk_mask_bool[:, trunk_col]
            trunk_pred_col = ab_mask[:, trunk_col]
            trunk_extra = torch.logical_and(
                trunk_pred_col[bar_row + 1 :],
                torch.logical_not(trunk_target_col[bar_row + 1 :]),
            ).sum().item()
            breakdown["trunk_extra_ratio"] = float(trunk_extra / max(1.0, target_area))
            trunk_rows_mask = trunk_target_col[bar_row + 1 :]
            side_clutter = 0.0
            if trunk_rows_mask.numel() > 0:
                for offset in (-1, 1):
                    neighbor_col = trunk_col + offset
                    if 0 <= neighbor_col < width:
                        neighbor_pred = ab_mask[bar_row + 1 :, neighbor_col]
                        side_clutter += torch.logical_and(neighbor_pred, trunk_rows_mask).sum().item()
            breakdown["side_clutter_ratio"] = float(side_clutter / max(1.0, target_area))
            trunk_pred_mask = ab_mask[bar_row + 1 :, :]
            trunk_columns = torch.sum(trunk_pred_mask, dim=0)
            if trunk_columns.numel() > 0:
                pred_trunk_col = int(torch.argmax(trunk_columns).item())
                trunk_alignment_gap = abs(pred_trunk_col - trunk_col) / max(1.0, float(width))
                breakdown["trunk_alignment_score"] = max(0.0, 1.0 - trunk_alignment_gap * 3.0)
            trunk_target_band = target_bool[bar_row : bar_row + 2, max(0, trunk_col - 1) : min(width, trunk_col + 2)]
            trunk_pred_band = ab_mask[bar_row : bar_row + 2, max(0, trunk_col - 1) : min(width, trunk_col + 2)]
            if trunk_target_band.numel() > 0:
                breakdown["junction_score"] = float(
                    torch.logical_and(trunk_pred_band, trunk_target_band).sum().item()
                ) / float(max(1.0, trunk_target_band.sum().item()))
    if trunk_col is None:
        breakdown["trunk_alignment_score"] = 0.0

    pred_bool = ab_mask.to(dtype=torch.bool)
    if y_max + 1 < height:
        excess_lower = pred_bool[y_max + 1 :, :].sum().item()
        breakdown["excess_below_ratio"] = float(excess_lower / target_area)
    if x_min > 0 or x_max < width - 1:
        left_excess = pred_bool[:, :x_min].sum().item() if x_min > 0 else 0.0
        right_excess = pred_bool[:, x_max + 1 :].sum().item() if x_max + 1 < width else 0.0
        breakdown["excess_side_ratio"] = float((left_excess + right_excess) / target_area)
    if trunk_col is not None:
        off_axis_mask = pred_bool[bar_row + 1 :, :]
        axis_mask = torch.zeros_like(off_axis_mask)
        axis_min = max(0, trunk_col - 1)
        axis_max = min(width, trunk_col + 2)
        axis_mask[:, axis_min:axis_max] = True
        off_axis_mass = torch.logical_and(off_axis_mask, torch.logical_not(axis_mask)).sum().item()
        breakdown["off_axis_mass_ratio"] = float(off_axis_mass / max(1.0, target_area))

    if breakdown["disconnected_fragments"] > 0.0:
        breakdown["disconnected_fragments"] = float(
            breakdown["disconnected_fragments"] / max(1.0, num_components)
        )

    return breakdown

def compute_fitness(
    grid: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    alpha_fp: float = 1.0,
    beta_fn: float = 0.5,
    alive_end: Optional[float] = None,
    target_area: Optional[float] = None,
    gamma_area: float = 0.15,
    stem_penalty_scale: float = 0.05,
    stem_cleanup_multiplier: float = 1.0,
    stem_cleanup_ratio: float = 0.0,
    stem_phase_cleanup_ratio: float = 0.0,
    stem_corridor_start_scale: float = 2.0,
    stem_corridor_end_scale: float = 1.1,
    t_symmetry_weight: float = 0.0,
    t_trunk_weight: float = 0.0,
    clean_component_weight: float = 0.0,
    clean_fp_weight: float = 0.0,
    stability_metrics: Optional[Dict[str, float]] = None,
    t_benchmark_weights: Optional[Dict[str, float]] = None,
    expect_t_shape: bool = False,
    coverage_weight: float = 0.0,
    coverage_floor: float = 0.0,
    coverage_penalty_weight: float = 0.0,
    sparse_area_floor: float = 0.0,
    sparse_penalty_weight: float = 0.0,
    t_phase_gate: float = 1.0,
    t_structure_gate: float = 1.0,
    t_cleanliness_gate: float = 1.0,
    collapse_area_floor: float = 0.0,
    collapse_coverage_floor: float = 0.0,
    collapse_penalty_weight: float = 0.0,
    collapse_bonus_suppression: float = 0.0,
    late_t_enforce: Optional[Dict[str, float]] = None,
    stem_trunk_bonus_weight: float = 0.0,
    stem_trunk_discount: float = 0.0,
    stem_trunk_presence: bool = False,
) -> Dict[str, float]:
    ab_mask = torch.logical_or(grid == CELL_A, grid == CELL_B)
    target_tensor = target_mask.to(device=grid.device)
    target_bool = target_tensor.to(dtype=torch.bool)
    target_cells = float(target_bool.sum().item())
    intersection = torch.logical_and(ab_mask, target_bool).sum().item()
    union = torch.logical_or(ab_mask, target_bool).sum().item()
    iou = float(intersection / union) if union > 0 else 0.0
    target_coverage_ratio = float(intersection) / max(1.0, target_cells)

    area_ab = float(ab_mask.sum().item())
    total_cells = float(target_tensor.numel()) if target_tensor.numel() > 0 else 1.0

    fp = float(torch.logical_and(ab_mask, torch.logical_not(target_bool)).sum().item())
    fn = float(torch.logical_and(torch.logical_not(ab_mask), target_bool).sum().item())
    fp_rate = fp / total_cells
    fn_rate = fn / total_cells

    base_score = float(iou - alpha_fp * fp_rate - beta_fn * fn_rate)
    fitness = base_score

    alive_bonus = 0.0
    extinction_penalty = 0.0
    target_area_val = float(target_area) if target_area is not None else None
    alive_denom = target_area_val if target_area_val and target_area_val > 0 else 1.0
    if alive_end is not None:
        alive_value = max(0.0, float(alive_end))
        alive_bonus = 0.06 * min(1.0, alive_value / alive_denom)
        if alive_value == 0.0:
            extinction_penalty = 0.5

    area_ratio = 1.0
    area_penalty2 = 0.0
    target_area_positive = target_area_val if target_area_val and target_area_val > 0 else None
    if target_area_positive is not None:
        area_ratio = area_ab / target_area_positive if target_area_positive > 0 else 1.0
        penalty_scale = max(0.0, float(gamma_area))
        area_penalty2 = penalty_scale * abs(area_ratio - 1.0)

    height, width = grid.shape[-2:]
    grid_cells = float(max(1, height * width))
    num_components, largest_component_area = _connected_components(ab_mask)
    largest_component_ratio = (
        largest_component_area / max(1.0, area_ab) if area_ab > 0 else 0.0
    )
    component_bonus = 0.18 * largest_component_ratio
    fragment_penalty = 0.035 * max(0.0, float(num_components - 1))

    pred_com = _center_of_mass(ab_mask, height, width)
    target_com = _center_of_mass(target_bool, height, width)
    com_penalty = 0.10 * _normalized_distance(pred_com, target_com, height, width)

    stem_mask = (grid == CELL_STEM)
    stem_count = int(stem_mask.sum().item())

    effective_stem_scale = max(0.0, float(stem_penalty_scale))
    cleanup_ratio = min(1.0, max(0.0, float(stem_cleanup_ratio)))
    phase_cleanup_ratio = min(1.0, max(0.0, float(stem_phase_cleanup_ratio)))
    combined_cleanup = max(cleanup_ratio, phase_cleanup_ratio)
    cleanup_extra = max(0.0, float(stem_cleanup_multiplier) - 1.0)
    start_scale = max(0.0, float(stem_corridor_start_scale))
    end_scale = max(0.0, float(stem_corridor_end_scale))
    if end_scale > start_scale:
        end_scale = start_scale
    baseline_area = target_area_positive if target_area_positive is not None else 0.25 * grid_cells
    baseline_area = max(1.0, baseline_area)
    corridor_scale = start_scale - (start_scale - end_scale) * combined_cleanup
    corridor_scale = max(end_scale, corridor_scale)
    allowed_stems = baseline_area * corridor_scale
    normalized_excess = max(0.0, float(stem_count) - allowed_stems) / baseline_area
    cleanup_pressure = combined_cleanup ** 0.5
    stem_penalty = effective_stem_scale * cleanup_pressure * normalized_excess
    stem_penalty *= 1.0 + cleanup_extra * combined_cleanup

    stray_component_gap = max(0.0, 1.0 - largest_component_ratio)
    stability_info = _derive_stability_scores(stability_metrics)
    fp_noise_ratio = 0.0
    if area_ab > 0:
        fp_noise_ratio = fp / area_ab
    clean_component_penalty = max(0.0, float(clean_component_weight)) * stray_component_gap
    clean_fp_penalty = max(0.0, float(clean_fp_weight)) * fp_noise_ratio
    late_clean_penalty = clean_component_penalty + clean_fp_penalty

    track_t_metrics = bool(expect_t_shape or t_benchmark_weights)
    if track_t_metrics:
        t_breakdown = _compute_t_structure_breakdown(
            ab_mask,
            target_bool,
            area_ratio=area_ratio,
            num_components=float(num_components),
        )
    else:
        t_breakdown = {
            "top_bar_coverage": 0.0,
            "top_bar_symmetry": 0.0,
            "top_bar_alignment_score": 0.0,
            "top_bar_length_ratio": 0.0,
            "trunk_coverage": 0.0,
            "trunk_alignment_score": 0.0,
            "junction_score": 0.0,
            "excess_below_ratio": 0.0,
            "excess_side_ratio": 0.0,
            "trunk_extra_ratio": 0.0,
            "side_clutter_ratio": 0.0,
            "off_axis_mass_ratio": 0.0,
            "area_overshoot": 0.0,
            "area_undershoot": 0.0,
            "disconnected_fragments": 0.0,
        }

    coverage_reward = max(0.0, float(coverage_weight)) * target_coverage_ratio
    coverage_deficit = max(0.0, float(coverage_floor) - target_coverage_ratio)
    coverage_penalty = max(0.0, float(coverage_penalty_weight)) * coverage_deficit
    sparse_progress = min(
        target_coverage_ratio,
        area_ratio if target_area_positive is not None else target_coverage_ratio,
    )
    sparse_deficit = max(0.0, float(sparse_area_floor) - sparse_progress)
    sparse_collapse_penalty = max(0.0, float(sparse_penalty_weight)) * sparse_deficit

    collapse_area_deficit = 0.0
    if collapse_area_floor > 0.0:
        collapse_area_deficit = max(0.0, float(collapse_area_floor) - area_ratio)
    collapse_coverage_deficit = 0.0
    if collapse_coverage_floor > 0.0:
        collapse_coverage_deficit = max(0.0, float(collapse_coverage_floor) - target_coverage_ratio)
    collapse_trigger = max(collapse_area_deficit, collapse_coverage_deficit)
    collapse_gate = 1.0
    empty_collapse_penalty = 0.0
    if collapse_trigger > 0.0:
        penalty_strength = max(0.0, float(collapse_penalty_weight))
        if penalty_strength > 0.0:
            empty_collapse_penalty = penalty_strength * collapse_trigger
        suppress_scale = max(0.0, float(collapse_bonus_suppression))
        if suppress_scale > 0.0:
            collapse_gate = max(0.0, 1.0 - suppress_scale * collapse_trigger)

    t_phase_gate_value = max(0.0, min(1.0, float(t_phase_gate)))
    t_structure_gate_value = max(0.0, min(1.0, float(t_structure_gate)))
    t_cleanliness_gate_value = max(0.0, min(1.0, float(t_cleanliness_gate)))

    # Stem-trunk bonus: reward stem cells in the target trunk column as
    # partial trunk construction. Directional divide actions place stem
    # cells in the trunk column via DIVIDE_S, but IoU and trunk_weight
    # only see CELL_A/CELL_B. This bonus bridges the gap: stem placement
    # is an intermediate step toward trunk construction. Default 0.0 so
    # legacy configs are unchanged. See commit bc0490a for the discovery.
    stem_trunk_bonus_val = 0.0
    if stem_trunk_bonus_weight > 0.0 and expect_t_shape:
        # Find trunk column (same logic as _compute_t_diagnostics)
        row_counts = target_bool.sum(dim=1)
        if row_counts.numel() > 0:
            bar_row = int(torch.argmax(row_counts).item())
            if bar_row + 1 < height:
                lower_target = target_bool[bar_row + 1:, :]
                col_counts = lower_target.sum(dim=0)
                if col_counts.numel() > 0 and col_counts.max().item() > 0:
                    trunk_col = int(torch.argmax(col_counts).item())
                    trunk_target_cells = lower_target[:, trunk_col]
                    trunk_len = float(trunk_target_cells.sum().item())
                    if trunk_len > 0:
                        # Count ANY non-empty cell (stem, A, or B) in
                        # the trunk target column as partial credit
                        trunk_stem_or_ab = (grid[bar_row + 1:, trunk_col] != CELL_EMPTY)
                        trunk_presence = float(
                            torch.logical_and(trunk_stem_or_ab, trunk_target_cells).sum().item()
                        )
                        stem_trunk_ratio = trunk_presence / trunk_len
                        stem_trunk_bonus_val = float(stem_trunk_bonus_weight) * stem_trunk_ratio

    fitness = (
        fitness
        + alive_bonus
        + coverage_reward
        - extinction_penalty
        - area_penalty2
        + component_bonus
        - fragment_penalty
        - com_penalty
        - stem_penalty
        - late_clean_penalty
        - coverage_penalty
        - sparse_collapse_penalty
        - empty_collapse_penalty
        + stem_trunk_bonus_val
    )

    t_phase_score = 0.0
    t_structure_bonus = 0.0
    geometry_gate = 0.0
    junction_score = t_breakdown.get("junction_score", 0.0)
    top_alignment = t_breakdown.get("top_bar_alignment_score", 0.0)
    trunk_alignment = t_breakdown.get("trunk_alignment_score", 0.0)
    off_axis_penalty = min(
        1.0,
        t_breakdown.get("off_axis_mass_ratio", 0.0)
        + t_breakdown.get("side_clutter_ratio", 0.0)
        + 0.5 * t_breakdown.get("trunk_extra_ratio", 0.0),
    )
    coverage_gate = 1.0
    if coverage_floor > 0.0:
        coverage_gate = max(
            0.0,
            min(1.0, (target_coverage_ratio - coverage_floor) / max(1e-6, 1.0 - coverage_floor)),
        )
    geometry_gate = max(0.0, min(1.0, 0.5 * (top_alignment + trunk_alignment) - 0.6 * off_axis_penalty))
    geometry_gate *= coverage_gate
    geometry_gate *= collapse_gate
    gated_hold_score = stability_info["hold_score"] * geometry_gate
    gated_stability_score = stability_info["stability_score"] * max(
        0.0, min(1.0, 0.5 * geometry_gate + 0.5 * junction_score)
    )

    if t_benchmark_weights:
        weights = {
            "w_shape": 1.0,
            "w_hold": 0.5,
            "w_stability": 0.4,
            "w_compact": 0.3,
            "w_fp": 0.8,
            "w_fn": 0.8,
            "w_frag": 0.4,
            "w_collapse": 0.6,
            "top_bar_weight": 0.3,
            "trunk_weight": 0.3,
            "symmetry_weight": 0.2,
            "bar_alignment_weight": 0.2,
            "trunk_alignment_weight": 0.2,
            "junction_weight": 0.25,
            "excess_below_weight": 0.4,
            "excess_side_weight": 0.3,
            "overshoot_weight": 0.25,
            "undershoot_weight": 0.25,
            "fragment_weight": 0.3,
            "trunk_extra_weight": 0.35,
            "side_clutter_weight": 0.35,
            "off_axis_weight": 0.3,
            "endpoint_clutter_weight": 0.2,
            "cleanliness_weight": 0.4,
        }
        weights.update({k: float(v) for k, v in t_benchmark_weights.items()})
        phase_gate = max(0.0, min(1.0, t_phase_gate_value * collapse_gate))
        structure_gate = max(0.0, min(1.0, t_structure_gate_value * collapse_gate))
        clean_gate = max(0.0, min(1.0, t_cleanliness_gate_value * collapse_gate))
        frag_penalty_score = max(0.0, float(num_components - 1.0) / max(1.0, num_components))
        compactness_score = largest_component_ratio
        t_phase_score = phase_gate * (
            weights["w_shape"] * iou
            + weights["w_hold"] * gated_hold_score
            + weights["w_stability"] * gated_stability_score
            + weights["w_compact"] * compactness_score
            - weights["w_fp"] * fp_rate
            - weights["w_fn"] * fn_rate
            - weights["w_frag"] * frag_penalty_score
            - weights["w_collapse"] * stability_info["collapse_penalty"]
        )
        fitness += t_phase_score
        t_structure_bonus = structure_gate * (
            weights["top_bar_weight"] * t_breakdown["top_bar_coverage"] * coverage_gate
            + weights["trunk_weight"] * t_breakdown["trunk_coverage"]
            + weights["symmetry_weight"] * t_breakdown["top_bar_symmetry"]
            + weights["bar_alignment_weight"] * t_breakdown["top_bar_alignment_score"]
            + weights["trunk_alignment_weight"] * t_breakdown["trunk_alignment_score"]
            + weights["junction_weight"] * junction_score
            - weights["excess_below_weight"] * t_breakdown["excess_below_ratio"]
            - weights["excess_side_weight"] * t_breakdown["excess_side_ratio"]
            - weights["overshoot_weight"] * t_breakdown["area_overshoot"]
            - weights["undershoot_weight"] * t_breakdown["area_undershoot"]
            - weights["fragment_weight"] * t_breakdown["disconnected_fragments"]
            - weights["trunk_extra_weight"] * t_breakdown["trunk_extra_ratio"]
            - weights["side_clutter_weight"] * t_breakdown["side_clutter_ratio"]
            - weights["off_axis_weight"] * off_axis_penalty
            - weights["endpoint_clutter_weight"] * t_breakdown.get("bar_endpoint_clutter", 0.0)
        )
        fitness += t_structure_bonus
        cleanliness_weight = max(0.0, float(weights.get("cleanliness_weight", 0.0)))
        cleanliness_penalty = clean_gate * cleanliness_weight * (
            0.6 * t_breakdown["excess_below_ratio"]
            + 0.5 * t_breakdown["excess_side_ratio"]
            + 0.4 * t_breakdown["area_overshoot"]
            + 0.6 * t_breakdown["disconnected_fragments"]
            + 0.5 * t_breakdown["trunk_extra_ratio"]
            + 0.45 * t_breakdown["side_clutter_ratio"]
            + 0.4 * t_breakdown.get("off_axis_mass_ratio", 0.0)
            + 0.35 * t_breakdown.get("bar_endpoint_clutter", 0.0)
        )
        fitness -= cleanliness_penalty
    else:
        cleanliness_penalty = 0.0

    if stability_info["hold_score"] > 0.0 and geometry_gate < 0.5:
        fitness -= (1.0 - geometry_gate) * stability_info["hold_score"] * 0.4

    t_symmetry, t_trunk_continuity = _compute_t_diagnostics(
        ab_mask, target_bool,
        grid=grid if expect_t_shape else None,
        stem_trunk_presence=stem_trunk_presence if expect_t_shape else False,
    )
    symmetry_weight = max(0.0, float(t_symmetry_weight))
    trunk_weight = max(0.0, float(t_trunk_weight))
    morph_gate = max(0.0, min(1.0, collapse_gate * t_structure_gate_value))
    t_morph_bonus = morph_gate * (symmetry_weight * t_symmetry + trunk_weight * t_trunk_continuity)
    fitness += t_morph_bonus
    late_structure_penalty = 0.0
    weak_trunk_component = 0.0
    junction_miss_component = 0.0
    pseudo_t_penalty = 0.0
    late_alignment_component = 0.0
    late_coverage_component = 0.0
    late_area_component = 0.0
    late_empty_component = 0.0
    if expect_t_shape and late_t_enforce:
        gate = max(0.0, float(late_t_enforce.get("gate", 0.0)))
        if gate > 0.0:
            penalty_scale = max(0.0, float(late_t_enforce.get("penalty", 1.0)))
            trunk_weight_local = max(0.0, float(late_t_enforce.get("trunk_weight", 1.2)))
            alignment_weight_local = max(0.0, float(late_t_enforce.get("alignment_weight", 1.0)))
            junction_weight_local = max(0.0, float(late_t_enforce.get("junction_weight", 0.9)))
            coverage_weight_local = max(0.0, float(late_t_enforce.get("coverage_weight", 0.8)))
            area_weight_local = max(0.0, float(late_t_enforce.get("area_weight", 0.8)))
            empty_weight_local = max(0.0, float(late_t_enforce.get("empty_weight", 0.7)))
            pseudo_weight_local = max(0.0, float(late_t_enforce.get("pseudo_weight", 1.0)))
            trunk_min = max(0.0, float(late_t_enforce.get("trunk_min", 0.4)))
            alignment_min = max(0.0, float(late_t_enforce.get("alignment_min", 0.4)))
            junction_min = max(0.0, float(late_t_enforce.get("junction_min", 0.35)))
            coverage_min = max(0.0, float(late_t_enforce.get("coverage_min", 0.5)))
            area_min = max(0.0, float(late_t_enforce.get("area_min", 0.5)))
            empty_floor = max(0.0, float(late_t_enforce.get("empty_floor", 0.2)))
            bar_min = max(0.0, float(late_t_enforce.get("bar_min", 0.35)))
            trunk_score = t_breakdown.get("trunk_coverage", 0.0)
            alignment_score = t_breakdown.get("trunk_alignment_score", 0.0)
            coverage_shortfall = max(0.0, coverage_min - target_coverage_ratio)
            area_shortfall = max(0.0, area_min - area_ratio)
            weak_trunk_component = trunk_weight_local * max(0.0, trunk_min - trunk_score)
            late_alignment_component = alignment_weight_local * max(0.0, alignment_min - alignment_score)
            junction_miss_component = junction_weight_local * max(0.0, junction_min - junction_score)
            late_coverage_component = coverage_weight_local * coverage_shortfall
            late_area_component = area_weight_local * area_shortfall
            late_empty_component = empty_weight_local * max(0.0, empty_floor - min(area_ratio, target_coverage_ratio))
            pseudo_gap = 0.0
            if t_breakdown.get("top_bar_coverage", 0.0) >= bar_min and trunk_score < trunk_min * 0.6:
                pseudo_gap = trunk_min * 0.6 - trunk_score
            pseudo_t_penalty = pseudo_weight_local * pseudo_gap
            total_penalty = (
                weak_trunk_component
                + late_alignment_component
                + junction_miss_component
                + late_coverage_component
                + late_area_component
                + late_empty_component
                + pseudo_t_penalty
            )
            late_structure_penalty = gate * penalty_scale * total_penalty
            if late_structure_penalty > 0.0:
                fitness -= late_structure_penalty
                scale = gate * penalty_scale
                weak_trunk_component *= scale
                junction_miss_component *= scale
                pseudo_t_penalty *= scale
                late_alignment_component *= scale
                late_coverage_component *= scale
                late_area_component *= scale
                late_empty_component *= scale
    return {
        "fitness": fitness,
        "iou": iou,
        "area": area_ab,
        "area_penalty2": area_penalty2,
        "base_score": base_score,
        "alive_bonus": alive_bonus,
        "extinction_penalty": extinction_penalty,
        "fp": fp_rate,
        "fn": fn_rate,
        "num_components": float(num_components),
        "largest_component_area": float(largest_component_area),
        "largest_component_ratio": largest_component_ratio,
        "stem_count": float(stem_count),
        "component_bonus": component_bonus,
        "fragment_penalty": fragment_penalty,
        "com_penalty": com_penalty,
        "stem_penalty": stem_penalty,
        "t_symmetry": t_symmetry,
        "t_trunk_continuity": t_trunk_continuity,
        "late_clean_penalty": late_clean_penalty,
        "t_top_bar_coverage": t_breakdown.get("top_bar_coverage", 0.0),
        "t_top_bar_symmetry": t_breakdown.get("top_bar_symmetry", 0.0),
        "t_top_bar_alignment_score": t_breakdown.get("top_bar_alignment_score", 0.0),
        "t_top_bar_length_ratio": t_breakdown.get("top_bar_length_ratio", 0.0),
        "t_trunk_coverage": t_breakdown.get("trunk_coverage", 0.0),
        "t_trunk_alignment_score": trunk_alignment,
        "t_junction_score": junction_score,
        "t_trunk_extra_ratio": t_breakdown.get("trunk_extra_ratio", 0.0),
        "t_side_clutter_ratio": t_breakdown.get("side_clutter_ratio", 0.0),
        "t_bar_endpoint_clutter": t_breakdown.get("bar_endpoint_clutter", 0.0),
        "t_off_axis_penalty": off_axis_penalty,
        "t_cleanliness_penalty": cleanliness_penalty,
        "t_geometry_gate": geometry_gate,
        "target_coverage_ratio": target_coverage_ratio,
        "coverage_penalty": coverage_penalty,
        "sparse_collapse_penalty": sparse_collapse_penalty,
        "empty_collapse_penalty": empty_collapse_penalty,
        "collapse_gate": collapse_gate,
        "hold_score": stability_info["hold_score"],
        "stability_score": stability_info["stability_score"],
        "collapse_penalty": stability_info["collapse_penalty"],
        "stabilized_success": stability_info["stabilized_success"],
        "first_reach_step": stability_info["first_reach_step"],
        "stability_failures": stability_info["stability_failures"],
        "hold_length": stability_info["hold_length"],
        "hold_window": stability_info["hold_window"],
        "t_phase_score": t_phase_score,
        "t_structure_bonus": t_structure_bonus,
        # t_top_bar_coverage, t_top_bar_symmetry, t_trunk_coverage — already set above (L735-739)
        "t_excess_below_ratio": t_breakdown["excess_below_ratio"],
        "t_excess_side_ratio": t_breakdown["excess_side_ratio"],
        "t_area_overshoot": t_breakdown["area_overshoot"],
        "t_area_undershoot": t_breakdown["area_undershoot"],
        "t_disconnected_fragments": t_breakdown["disconnected_fragments"],
        "late_structure_penalty": late_structure_penalty,
        "weak_trunk_penalty": weak_trunk_component,
        "junction_miss_penalty": junction_miss_component,
        "pseudo_t_penalty": pseudo_t_penalty,
        "late_alignment_penalty": late_alignment_component,
        "late_coverage_penalty": late_coverage_component,
        "late_area_penalty": late_area_component,
        "late_empty_penalty": late_empty_component,
        "stem_trunk_bonus": stem_trunk_bonus_val,
        "t_morph_bonus": t_morph_bonus,
    }
