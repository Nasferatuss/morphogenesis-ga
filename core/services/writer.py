"""TensorBoard writer construction and run directory bootstrap."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.tensorboard import SummaryWriter

from src.utils import create_run_dir


def build_writer(run_name: str, runs_root: str = "runs") -> Tuple[SummaryWriter, Path, str]:
    """Create a run directory and return a TensorBoard SummaryWriter bound to it."""
    log_dir, run_id = create_run_dir(runs_root, run_name)
    print(f"[INFO] TensorBoard logdir: {log_dir}")
    print(f"[INFO] Run ID: {run_id}")
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer, log_dir, run_id
