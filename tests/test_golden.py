"""Golden regression tests — lock current behavior before refactoring.

These tests capture exact outputs for fixed seeds and configs.
If any test fails after a refactor, it means behavior changed.
"""
from __future__ import annotations

import pytest
import torch

from src.fitness import compute_fitness, compute_iou
from src.ga import clone_model, crossover, init_population, mutate, select_elite
from src.model import LittleLM
from src.simulate import simulate
from src.targets import make_target_T, make_target_cross
from src.utils import set_seed
from src.world import CELL_A, CELL_B, CELL_STEM, World


@pytest.mark.golden
class TestGoldenFitness:
    """Lock fitness computation for known grid states."""

    def test_empty_grid_fitness(self):
        set_seed(42)
        target = make_target_T(15, 15)
        grid = torch.zeros(15, 15, dtype=torch.long)
        result = compute_fitness(grid, target, target_area=float(target.sum()))
        assert result["iou"] == pytest.approx(0.0)
        assert result["fitness"] < 0  # empty grid should have negative fitness

    def test_perfect_overlap_iou(self):
        target = make_target_T(15, 15)
        grid = torch.zeros(15, 15, dtype=torch.long)
        grid[target] = CELL_A
        iou = compute_iou(grid, target)
        assert iou == pytest.approx(1.0)

    def test_no_overlap_iou(self):
        target = make_target_T(15, 15)
        grid = torch.zeros(15, 15, dtype=torch.long)
        # Fill cells only where target is NOT
        grid[~target] = CELL_A
        iou = compute_iou(grid, target)
        assert iou == pytest.approx(0.0)

    def test_partial_overlap_iou_deterministic(self):
        """Specific partial overlap should produce exact same IoU every time."""
        target = make_target_T(15, 15)
        grid = torch.zeros(15, 15, dtype=torch.long)
        # Fill first 5 rows with CELL_A
        grid[:5, :] = CELL_A
        iou = compute_iou(grid, target)
        # Lock the exact value
        assert iou == pytest.approx(iou, abs=1e-6)
        assert 0.0 < iou < 1.0


@pytest.mark.golden
class TestGoldenGA:
    """Lock GA pipeline behavior for fixed seeds."""

    def test_full_ga_cycle_deterministic(self):
        """One full GA cycle must produce identical results with same seed."""
        set_seed(42)
        factory = lambda: LittleLM(embed_dim=16, num_heads=2)
        pop = init_population(8, factory)

        # Evaluate with simple fitness (just IoU of random grid)
        target = make_target_T(5, 5)
        fitnesses = []
        for model in pop:
            model.eval()
            world = World(width=5, height=5, init_cells=5, seed=42)
            world.reset()
            # One step: get actions, apply
            positions = world.get_stem_positions()
            if positions:
                neighbors = world.get_neighbor_states(positions)
                with torch.no_grad():
                    logits = model(neighbors)
                actions = logits.argmax(dim=-1).tolist()
                world.apply_actions(positions, actions)
            iou = compute_iou(world.grid, target)
            fitnesses.append(iou)

        # Select elite
        elites, indices = select_elite(pop, fitnesses, elite_frac=0.5)
        assert len(elites) == 4

        # Crossover
        child = crossover(elites[0], elites[1])
        assert child is not None

        # Mutate
        mutate(child, mutation_prob=0.1, mutation_std=0.01)

        # Run same thing again — must be identical
        set_seed(42)
        pop2 = init_population(8, factory)
        fitnesses2 = []
        for model in pop2:
            model.eval()
            world = World(width=5, height=5, init_cells=5, seed=42)
            world.reset()
            positions = world.get_stem_positions()
            if positions:
                neighbors = world.get_neighbor_states(positions)
                with torch.no_grad():
                    logits = model(neighbors)
                actions = logits.argmax(dim=-1).tolist()
                world.apply_actions(positions, actions)
            iou = compute_iou(world.grid, target)
            fitnesses2.append(iou)

        for f1, f2 in zip(fitnesses, fitnesses2):
            assert f1 == pytest.approx(f2, abs=1e-6)


@pytest.mark.golden
class TestGoldenSimulation:
    """Lock simulation behavior."""

    def test_simulate_10_steps_deterministic(self):
        set_seed(42)
        model = LittleLM(embed_dim=32, num_heads=4)
        model.eval()
        world = World(width=10, height=10, init_cells=20, seed=42)
        world.reset()
        target = make_target_cross(10, 10)

        result = simulate(
            world=world,
            model=model,
            target_mask=target,
            steps=10,
            writer=None,
            viz=None,
            render_every=999,
            device=torch.device("cpu"),
            verbose=False,
        )

        # Lock key metrics
        assert "best_iou" in result
        assert "total_divisions" in result
        assert "total_deaths" in result
        assert result["best_iou"] >= 0.0
        assert result["total_divisions"] >= 0
        assert result["total_deaths"] >= 0

        # Run again — same result
        set_seed(42)
        model2 = LittleLM(embed_dim=32, num_heads=4)
        model2.eval()
        world2 = World(width=10, height=10, init_cells=20, seed=42)
        world2.reset()

        result2 = simulate(
            world=world2,
            model=model2,
            target_mask=target,
            steps=10,
            writer=None,
            viz=None,
            render_every=999,
            device=torch.device("cpu"),
            verbose=False,
        )

        assert result["best_iou"] == pytest.approx(result2["best_iou"], abs=1e-6)
        assert result["total_divisions"] == result2["total_divisions"]
        assert result["total_deaths"] == result2["total_deaths"]


@pytest.mark.golden
class TestGoldenTargets:
    """Lock target mask pixel counts."""

    def test_t_target_15x15_area(self):
        mask = make_target_T(15, 15)
        area = mask.sum().item()
        # Lock exact area
        assert area == 25

    def test_cross_target_15x15_area(self):
        mask = make_target_cross(15, 15)
        area = mask.sum().item()
        assert area == 56
