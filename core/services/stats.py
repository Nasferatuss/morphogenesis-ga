"""Statistical helper functions for GA metrics."""
from __future__ import annotations

from typing import List, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def summarize(values: List[float], percentile: float) -> Tuple[float, float]:
    """Return (mean, percentile_value) for a list of floats."""
    if not values:
        return 0.0, 0.0
    if np is not None:
        mean_val = float(np.mean(values))
        perc = float(np.percentile(values, percentile))
    else:
        mean_val = float(sum(values) / len(values))
        sorted_vals = sorted(values)
        idx = int(round((percentile / 100.0) * (len(sorted_vals) - 1)))
        perc = float(sorted_vals[idx])
    return mean_val, perc
