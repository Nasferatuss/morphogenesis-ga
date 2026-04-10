from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy may be missing in tiny envs
    np = None


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: str) -> torch.device:
    prefer = (prefer or "cuda").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_run_dir(base_dir: str, run_name: str) -> Tuple[Path, str]:
    safe_name = (run_name or "run").strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{safe_name}_{timestamp}"
    path = Path(base_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path, run_id
