from __future__ import annotations

import torch

from src.world import (
    ACTION_BECOME_A,
    ACTION_BECOME_B,
    ACTION_DIE,
    ACTION_DIVIDE,
    ACTION_STAY,
    CELL_A,
    CELL_B,
    CELL_EMPTY,
    CELL_STEM,
    World,
)


class TestWorldInit:
    def test_grid_shape(self, small_world):
        assert small_world.grid.shape == (5, 5)

    def test_init_cells_count(self, small_world):
        stem_count = (small_world.grid == CELL_STEM).sum().item()
        assert stem_count == 5

    def test_init_cells_capped_by_grid_size(self):
        w = World(width=2, height=2, init_cells=100, seed=0)
        w.reset()
        assert (w.grid == CELL_STEM).sum().item() == 4

    def test_deterministic_reset(self):
        w1 = World(width=5, height=5, init_cells=5, seed=42)
        w1.reset()
        w2 = World(width=5, height=5, init_cells=5, seed=42)
        w2.reset()
        assert torch.equal(w1.grid, w2.grid)

    def test_different_seeds_different_grids(self):
        w1 = World(width=10, height=10, init_cells=20, seed=1)
        w1.reset()
        w2 = World(width=10, height=10, init_cells=20, seed=2)
        w2.reset()
        assert not torch.equal(w1.grid, w2.grid)


class TestGetNeighborStates:
    def test_empty_positions(self, small_world):
        result = small_world.get_neighbor_states([])
        assert result.shape == (0, 8)

    def test_center_cell_has_8_neighbors(self, small_world):
        result = small_world.get_neighbor_states([(2, 2)])
        assert result.shape == (1, 8)

    def test_corner_cell_boundary(self):
        w = World(width=3, height=3, init_cells=0, seed=0)
        w.reset()
        w.grid[1, 1] = CELL_A
        result = w.get_neighbor_states([(0, 0)])
        # Only (1,1) is non-empty among 3 valid neighbors of (0,0)
        assert (result[0] == CELL_A).sum().item() == 1


class TestApplyActions:
    def test_stay_preserves_cell(self):
        w = World(width=3, height=3, init_cells=0, seed=0)
        w.reset()
        w.grid[1, 1] = CELL_STEM
        stats = w.apply_actions([(1, 1)], [ACTION_STAY])
        assert w.grid[1, 1].item() == CELL_STEM
        assert stats["divisions"] == 0
        assert stats["deaths"] == 0

    def test_become_a(self):
        w = World(width=3, height=3, init_cells=0, seed=0)
        w.reset()
        w.grid[1, 1] = CELL_STEM
        w.apply_actions([(1, 1)], [ACTION_BECOME_A])
        assert w.grid[1, 1].item() == CELL_A

    def test_become_b(self):
        w = World(width=3, height=3, init_cells=0, seed=0)
        w.reset()
        w.grid[1, 1] = CELL_STEM
        w.apply_actions([(1, 1)], [ACTION_BECOME_B])
        assert w.grid[1, 1].item() == CELL_B

    def test_die(self):
        w = World(width=3, height=3, init_cells=0, seed=0)
        w.reset()
        w.grid[1, 1] = CELL_STEM
        stats = w.apply_actions([(1, 1)], [ACTION_DIE])
        assert w.grid[1, 1].item() == CELL_EMPTY
        assert stats["deaths"] == 1

    def test_divide_creates_new_stem(self):
        w = World(width=3, height=3, init_cells=0, seed=42)
        w.reset()
        w.grid[1, 1] = CELL_STEM
        stats = w.apply_actions([(1, 1)], [ACTION_DIVIDE])
        stem_count = (w.grid == CELL_STEM).sum().item()
        assert stem_count == 2
        assert stats["divisions"] == 1

    def test_divide_no_space(self):
        """Division fails when all neighbors are occupied."""
        w = World(width=3, height=3, init_cells=0, seed=0)
        w.reset()
        w.grid.fill_(CELL_A)
        w.grid[1, 1] = CELL_STEM
        stats = w.apply_actions([(1, 1)], [ACTION_DIVIDE])
        assert stats["divisions"] == 0

    def test_non_stem_ignored(self):
        """Actions on non-stem cells are ignored."""
        w = World(width=3, height=3, init_cells=0, seed=0)
        w.reset()
        w.grid[1, 1] = CELL_A
        w.apply_actions([(1, 1)], [ACTION_DIE])
        assert w.grid[1, 1].item() == CELL_A  # unchanged


class TestCountStemCells:
    def test_count(self, small_world):
        assert small_world.count_stem_cells() == 5

    def test_empty_grid(self):
        w = World(width=3, height=3, init_cells=0, seed=0)
        w.reset()
        assert w.count_stem_cells() == 0
