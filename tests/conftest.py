from __future__ import annotations

import pytest
import torch

from src.utils import set_seed
from src.world import World


@pytest.fixture
def seeded():
    """Ensure deterministic tests."""
    set_seed(42)


@pytest.fixture
def small_world(seeded):
    """5x5 world with 5 initial stem cells, deterministic."""
    w = World(width=5, height=5, init_cells=5, seed=42)
    w.reset()
    return w


@pytest.fixture
def standard_world(seeded):
    """15x15 world matching default configs."""
    w = World(width=15, height=15, init_cells=50, seed=42)
    w.reset()
    return w
