from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import torch

CELL_EMPTY = 0
CELL_STEM = 1
CELL_A = 2
CELL_B = 3

ACTION_STAY = 0
ACTION_BECOME_A = 1
ACTION_BECOME_B = 2
ACTION_DIVIDE = 3
ACTION_DIE = 4

NEIGHBOR_OFFSETS: Sequence[Tuple[int, int]] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


class World:
    def __init__(self, width: int, height: int, init_cells: int, seed: int) -> None:
        self.width = width
        self.height = height
        self.init_cells = min(init_cells, width * height)
        self.rng = random.Random(seed)
        self.grid = torch.zeros((self.height, self.width), dtype=torch.long)

    def reset(self) -> torch.Tensor:
        self.grid.fill_(CELL_EMPTY)
        coords = [(y, x) for y in range(self.height) for x in range(self.width)]
        self.rng.shuffle(coords)
        for y, x in coords[: self.init_cells]:
            self.grid[y, x] = CELL_STEM
        return self.grid

    def get_stem_positions(self) -> List[Tuple[int, int]]:
        positions = torch.nonzero(self.grid == CELL_STEM, as_tuple=False)
        return [(int(y.item()), int(x.item())) for y, x in positions]

    def get_neighbor_states(
        self, positions: Sequence[Tuple[int, int]], grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not positions:
            return torch.zeros((0, len(NEIGHBOR_OFFSETS)), dtype=torch.long)
        base_grid = grid if grid is not None else self.grid
        data = torch.zeros((len(positions), len(NEIGHBOR_OFFSETS)), dtype=torch.long)
        for idx, (y, x) in enumerate(positions):
            for offset_idx, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    data[idx, offset_idx] = int(base_grid[ny, nx])
        return data

    def apply_actions(
        self, positions: Sequence[Tuple[int, int]], actions: Sequence[int]
    ) -> dict:
        prev_grid = self.grid.clone()
        divisions = 0
        deaths = 0
        for (y, x), action in zip(positions, actions):
            if prev_grid[y, x].item() != CELL_STEM:
                continue
            if action == ACTION_BECOME_A:
                self.grid[y, x] = CELL_A
            elif action == ACTION_BECOME_B:
                self.grid[y, x] = CELL_B
            elif action == ACTION_DIVIDE:
                target = self._sample_empty_neighbor((y, x), prev_grid)
                if target is not None:
                    ny, nx = target
                    if self.grid[ny, nx] == CELL_EMPTY:
                        self.grid[ny, nx] = CELL_STEM
                        divisions += 1
            elif action == ACTION_DIE:
                self.grid[y, x] = CELL_EMPTY
                deaths += 1
            else:
                # stay keeps the cell as stem
                pass
        return {"divisions": divisions, "deaths": deaths}

    def _sample_empty_neighbor(
        self, pos: Tuple[int, int], grid_snapshot: torch.Tensor
    ) -> Optional[Tuple[int, int]]:
        y, x = pos
        empty_cells = []
        for dy, dx in NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                if grid_snapshot[ny, nx].item() == CELL_EMPTY:
                    empty_cells.append((ny, nx))
        if not empty_cells:
            return None
        return self.rng.choice(empty_cells)

    def count_stem_cells(self) -> int:
        return int((self.grid == CELL_STEM).sum().item())

    def clone(self) -> torch.Tensor:
        return self.grid.clone()
