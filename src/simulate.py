
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from .fitness import compute_iou
from .world import ACTION_DIE, ACTION_STAY, CELL_A, CELL_B, World


def _normalize_die_cap_schedule(
    schedule: Optional[Sequence[Dict[str, float]]],
    fallback_cap: Optional[float],
) -> Sequence[Tuple[int, float]]:
    normalized = []
    if schedule:
        for entry in schedule:
            if not isinstance(entry, dict):
                continue
            until = entry.get("until_step")
            cap = entry.get("cap_frac")
            if cap is None or until is None:
                continue
            try:
                until_val = int(until)
            except (TypeError, ValueError):
                continue
            try:
                cap_val = float(cap)
            except (TypeError, ValueError):
                continue
            cap_clamped = max(0.0, min(1.0, cap_val))
            normalized.append((until_val, cap_clamped))
    if not normalized and fallback_cap is not None:
        cap_clamped = max(0.0, min(1.0, float(fallback_cap)))
        normalized.append((int(1e9), cap_clamped))
    normalized.sort(key=lambda pair: pair[0])
    return normalized


def _current_die_cap(schedule: Sequence[Tuple[int, float]], step_idx: int) -> Optional[float]:
    if not schedule:
        return None
    for until_step, cap in schedule:
        if step_idx <= until_step:
            return cap
    return schedule[-1][1]




def _late_cleanup_ratio(
    steps_done: int, planned_steps: int, cleanup_start_frac: float = 0.8
) -> float:
    if planned_steps <= 0:
        return 0.0
    start_frac = max(0.0, min(1.0, float(cleanup_start_frac)))
    cleanup_start = max(1, int(planned_steps * start_frac))
    if cleanup_start > planned_steps:
        cleanup_start = planned_steps
    if steps_done < cleanup_start:
        return 0.0
    window_span = max(1, planned_steps - cleanup_start + 1)
    late_steps = steps_done - cleanup_start + 1
    ratio = late_steps / window_span
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return float(ratio)

def simulate(
    world: World,
    model: torch.nn.Module,
    target_mask: torch.Tensor,
    steps: int,
    writer: Optional[SummaryWriter],
    viz: Optional[object],
    render_every: int,
    device: torch.device,
    anti_extinction_warmup_steps: int = 0,
    die_cap_schedule: Optional[Sequence[Dict[str, float]]] = None,
    post_warmup_die_cap_frac: Optional[float] = None,
    verbose: bool = False,
    log_every: int = 50,
    frame_capture: Optional[Callable[[torch.Tensor, int], None]] = None,
    capture_every: int = 1,
    late_cleanup_start_frac: float = 0.8,
    stability_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    render_every = max(1, render_every)
    warmup_steps = max(0, int(anti_extinction_warmup_steps))
    log_every = max(1, int(log_every))
    capture_stride = max(1, int(capture_every))

    def emit_capture(step_idx: int) -> None:
        nonlocal frame_capture
        if frame_capture is None:
            return
        try:
            frame_capture(world.grid.clone(), step_idx)
        except Exception as err:
            print(f"[WARN] Frame capture failed at step {step_idx}: {err}")
            frame_capture = None
    normalized_schedule = _normalize_die_cap_schedule(
        die_cap_schedule, post_warmup_die_cap_frac
    )
    iou = compute_iou(world.grid, target_mask)
    best_iou = float(iou)
    best_iou_step = 0
    stability_cfg = stability_cfg or {}
    reach_threshold = float(stability_cfg.get("reach_threshold_iou", 0.0))
    stabilize_threshold = float(
        stability_cfg.get("stabilize_threshold_iou", reach_threshold)
    )
    hold_window = max(0, int(stability_cfg.get("hold_window_steps", 0)))
    track_stability = (
        bool(stability_cfg.get("enabled", False))
        or reach_threshold > 0.0
        or stabilize_threshold > 0.0
        or hold_window > 0
    )
    first_reach_step: Optional[int] = None
    current_hold = 0
    max_hold = 0
    stabilized_success = False
    stability_failures = 0
    stability_reacquire_count = 0
    steps_done = 0
    total_divisions = 0
    total_deaths = 0
    anti_ext_triggers = 0

    if writer is not None:
        writer.add_scalar("fitness/iou", iou, 0)
        writer.add_scalar("stats/num_alive", world.count_stem_cells(), 0)
        writer.add_scalar("stats/divisions", 0, 0)
        writer.add_scalar("stats/deaths", 0, 0)

    if viz is not None:
        viz.render(world.grid, target_mask, step=0, iou=iou)
    emit_capture(0)

    if track_stability:
        if iou >= reach_threshold and first_reach_step is None:
            first_reach_step = 0
        if first_reach_step is not None:
            if iou >= stabilize_threshold:
                current_hold += 1
                max_hold = max(max_hold, current_hold)
            else:
                current_hold = 0
        if not stabilized_success:
            if hold_window <= 0 and first_reach_step is not None:
                stabilized_success = True
            elif hold_window > 0 and max_hold >= hold_window:
                stabilized_success = True

    if steps <= 0:
        if verbose:
            print("[INFO] Fast mode active (steps=0). Skipping simulation loop.")
        ab_mask = torch.logical_or(world.grid == CELL_A, world.grid == CELL_B)
        alive_total = int((world.grid != 0).sum().item())
        area_ab = int(ab_mask.sum().item())
        cleanup_ratio = _late_cleanup_ratio(0, steps, late_cleanup_start_frac)
        return {
            "steps": 0,
            "iou": iou,
            "total_divisions": total_divisions,
            "total_deaths": total_deaths,
            "alive_end": alive_total,
            "alive_total": alive_total,
            "area_ab": area_ab,
            "anti_extinction_triggers": anti_ext_triggers,
            "late_cleanup_ratio": cleanup_ratio,
            "best_iou": best_iou,
            "best_iou_step": best_iou_step,
            "first_reach_step": first_reach_step,
            "hold_length_achieved": max_hold,
            "stabilized_success": stabilized_success,
            "stability_failures": stability_failures,
            "stability_reacquire_count": stability_reacquire_count,
            "reach_threshold_iou": reach_threshold,
            "stabilize_threshold_iou": stabilize_threshold,
            "hold_window_steps": hold_window,
        }

    if verbose:
        print("[INFO] Starting simulation loop...")
    extinct = False
    with torch.no_grad():
        for step_idx in range(1, steps + 1):
            positions = world.get_stem_positions()
            if not positions:
                if verbose:
                    print(
                        f"[WARN] No stem cells remaining at step {step_idx}, stopping early."
                    )
                extinct = True
                steps_done = step_idx
                break

            prev_grid = world.grid.clone()
            rng_state = world.rng.getstate()

            neighbor_tokens = world.get_neighbor_states(positions).to(device)
            logits = model(neighbor_tokens)
            actions = torch.argmax(logits, dim=1).to("cpu").tolist()

            living_before = len(positions)
            cap_fraction = None
            if step_idx > warmup_steps:
                cap_fraction = _current_die_cap(normalized_schedule, step_idx)
            if cap_fraction is not None and living_before > 0:
                die_indices = [idx for idx, action in enumerate(actions) if action == ACTION_DIE]
                if die_indices:
                    target_die = cap_fraction * float(living_before)
                    allowed_die = int(target_die)
                    fractional_part = target_die - allowed_die
                    if (
                        fractional_part > 0
                        and allowed_die < living_before
                        and world.rng.random() < fractional_part
                    ):
                        allowed_die += 1
                    allowed_die = max(0, min(allowed_die, living_before))
                    allowed_die = min(allowed_die, len(die_indices))
                    if allowed_die < len(die_indices):
                        world.rng.shuffle(die_indices)
                        for idx in die_indices[allowed_die:]:
                            actions[idx] = ACTION_STAY
            step_stats = world.apply_actions(positions, actions)

            living_after = world.count_stem_cells()
            safeguard_applied = False
            if living_after == 0:
                if step_idx <= warmup_steps:
                    safe_actions = [
                        ACTION_STAY if action == ACTION_DIE else action for action in actions
                    ]
                    world.grid = prev_grid.clone()
                    world.rng.setstate(rng_state)
                    step_stats = world.apply_actions(positions, safe_actions)
                    living_after = world.count_stem_cells()
                    anti_ext_triggers += 1
                    safeguard_applied = True
                    if verbose:
                        print(f"Anti-extinction triggered at step {step_idx}")
                else:
                    if verbose:
                        print(
                            f"[WARN] Population extinct at step {step_idx}, stopping early."
                        )
                    extinct = True
                    steps_done = step_idx
                    break

            total_divisions += step_stats["divisions"]
            total_deaths += step_stats["deaths"]

            iou = compute_iou(world.grid, target_mask)
            if iou > best_iou:
                best_iou = float(iou)
                best_iou_step = step_idx
            if track_stability:
                if first_reach_step is None and iou >= reach_threshold:
                    first_reach_step = step_idx
                    current_hold = 0
                if first_reach_step is not None:
                    if iou >= stabilize_threshold:
                        current_hold += 1
                        if current_hold == 1 and stability_failures > 0:
                            stability_reacquire_count += 1
                        if current_hold > max_hold:
                            max_hold = current_hold
                    else:
                        if current_hold > 0:
                            stability_failures += 1
                        current_hold = 0
                if not stabilized_success:
                    if hold_window <= 0 and first_reach_step is not None:
                        stabilized_success = True
                    elif hold_window > 0 and max_hold >= hold_window:
                        stabilized_success = True

            if writer is not None:
                writer.add_scalar("fitness/iou", iou, step_idx)
                writer.add_scalar("stats/num_alive", living_after, step_idx)
                writer.add_scalar("stats/divisions", step_stats["divisions"], step_idx)
                writer.add_scalar("stats/deaths", step_stats["deaths"], step_idx)
                writer.add_scalar(
                    "stats/anti_extinction", 1 if safeguard_applied else 0, step_idx
                )

            if viz is not None and step_idx % render_every == 0:
                viz.render(world.grid, target_mask, step=step_idx, iou=iou)
            if frame_capture is not None and step_idx % capture_stride == 0:
                emit_capture(step_idx)

            steps_done = step_idx
            if verbose and (step_idx % log_every == 0 or step_idx == 1):
                print(f"[INFO] Step {step_idx} :: IoU={iou:.3f} :: alive={living_after}")
        else:
            steps_done = steps

    if viz is not None and steps_done % render_every != 0:
        viz.render(world.grid, target_mask, step=steps_done, iou=iou)
    if frame_capture is not None and steps_done % capture_stride != 0:
        emit_capture(steps_done)

    if extinct and steps_done == 0:
        steps_done = 0

    ab_mask = torch.logical_or(world.grid == CELL_A, world.grid == CELL_B)
    area_ab = int(ab_mask.sum().item())
    alive_total = int((world.grid != 0).sum().item())

    cleanup_ratio = _late_cleanup_ratio(steps_done, steps, late_cleanup_start_frac)
    return {
        "steps": steps_done,
        "iou": iou,
        "total_divisions": total_divisions,
        "total_deaths": total_deaths,
        "alive_end": alive_total,
        "alive_total": alive_total,
        "area_ab": area_ab,
        "anti_extinction_triggers": anti_ext_triggers,
        "late_cleanup_ratio": cleanup_ratio,
        "best_iou": best_iou,
        "best_iou_step": best_iou_step,
        "first_reach_step": first_reach_step,
        "hold_length_achieved": max_hold,
        "stabilized_success": stabilized_success,
        "stability_failures": stability_failures,
        "stability_reacquire_count": stability_reacquire_count,
        "reach_threshold_iou": reach_threshold,
        "stabilize_threshold_iou": stabilize_threshold,
        "hold_window_steps": hold_window,
    }



