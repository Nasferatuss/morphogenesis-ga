"""Benchmark scoring and statistical helpers for GA training."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.benchmark import BenchmarkSummary


def compute_benchmark_score(summary: BenchmarkSummary) -> float:
    """Compute composite score from benchmark summary."""
    return (
        1.5 * summary.success_rate_stabilized
        + summary.success_rate_reach
        + summary.mean_final_iou
        + 0.25 * summary.mean_best_iou
    )


def compute_benchmark_dict_score(summary_dict: Optional[Dict[str, Any]]) -> float:
    """Compute composite score from a dict representation of benchmark summary."""
    if not summary_dict:
        return 0.0
    return (
        1.5 * float(summary_dict.get("success_rate_stabilized", 0.0))
        + float(summary_dict.get("success_rate_reach", 0.0))
        + float(summary_dict.get("mean_final_iou", 0.0))
        + 0.25 * float(summary_dict.get("mean_best_iou", 0.0))
    )


def simple_mean(values: List[float]) -> float:
    """Mean of a list of floats, 0.0 if empty."""
    return float(sum(values) / len(values)) if values else 0.0


def simple_variance(values: List[float]) -> float:
    """Population variance of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean_val = simple_mean(values)
    return float(sum((val - mean_val) ** 2 for val in values) / len(values))
