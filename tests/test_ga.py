from __future__ import annotations

import torch

from src.ga import clone_model, crossover, init_population, mutate, select_elite
from src.model import LittleLM
from src.utils import set_seed


class TestCloneModel:
    def test_clone_independent(self):
        model = LittleLM(embed_dim=16, num_heads=2)
        cloned = clone_model(model)
        # Same weights
        for p1, p2 in zip(model.parameters(), cloned.parameters()):
            assert torch.equal(p1, p2)
        # Modification to clone doesn't affect original
        with torch.no_grad():
            for p in cloned.parameters():
                p.fill_(999.0)
        any_unchanged = any(
            not torch.equal(p, torch.full_like(p, 999.0))
            for p in model.parameters()
        )
        assert any_unchanged


class TestInitPopulation:
    def test_population_size(self):
        pop = init_population(8, lambda: LittleLM(embed_dim=16, num_heads=2))
        assert len(pop) == 8

    def test_models_are_independent(self):
        pop = init_population(4, lambda: LittleLM(embed_dim=16, num_heads=2))
        # Each model should have different random weights
        w0 = list(pop[0].parameters())[0].data
        w1 = list(pop[1].parameters())[0].data
        assert not torch.equal(w0, w1)


class TestSelectElite:
    def test_select_top_half(self):
        pop = init_population(4, lambda: LittleLM(embed_dim=16, num_heads=2))
        fitnesses = [0.1, 0.9, 0.5, 0.3]
        elites, indices = select_elite(pop, fitnesses, elite_frac=0.5)
        assert len(elites) == 2
        assert indices[0] == 1  # highest fitness
        assert indices[1] == 2  # second highest

    def test_minimum_one_elite(self):
        pop = init_population(4, lambda: LittleLM(embed_dim=16, num_heads=2))
        fitnesses = [0.1, 0.2, 0.3, 0.4]
        elites, _ = select_elite(pop, fitnesses, elite_frac=0.01)
        assert len(elites) >= 1


class TestCrossover:
    def test_child_shape_matches_parents(self):
        p1 = LittleLM(embed_dim=16, num_heads=2)
        p2 = LittleLM(embed_dim=16, num_heads=2)
        child = crossover(p1, p2)
        for cp, pp in zip(child.parameters(), p1.parameters()):
            assert cp.shape == pp.shape

    def test_child_different_from_both_parents(self):
        """Child should mix parameters from both parents."""
        set_seed(42)
        p1 = LittleLM(embed_dim=16, num_heads=2)
        p2 = LittleLM(embed_dim=16, num_heads=2)
        child = crossover(p1, p2)
        # Child shouldn't be identical to either parent
        same_as_p1 = all(
            torch.equal(cp, pp) for cp, pp in zip(child.parameters(), p1.parameters())
        )
        same_as_p2 = all(
            torch.equal(cp, pp) for cp, pp in zip(child.parameters(), p2.parameters())
        )
        assert not same_as_p1 or not same_as_p2


class TestMutate:
    def test_mutate_changes_weights(self):
        set_seed(42)
        model = LittleLM(embed_dim=16, num_heads=2)
        orig_params = [p.clone() for p in model.parameters()]
        mutate(model, mutation_prob=1.0, mutation_std=0.1)
        changed = any(
            not torch.equal(orig, new)
            for orig, new in zip(orig_params, model.parameters())
        )
        assert changed

    def test_zero_prob_no_mutation(self):
        model = LittleLM(embed_dim=16, num_heads=2)
        orig_params = [p.clone() for p in model.parameters()]
        mutate(model, mutation_prob=0.0, mutation_std=0.1)
        for orig, new in zip(orig_params, model.parameters()):
            assert torch.equal(orig, new)

    def test_deterministic_mutation(self):
        set_seed(42)
        m1 = LittleLM(embed_dim=16, num_heads=2)
        set_seed(99)
        mutate(m1, mutation_prob=0.5, mutation_std=0.01)
        params1 = [p.clone() for p in m1.parameters()]

        set_seed(42)
        m2 = LittleLM(embed_dim=16, num_heads=2)
        set_seed(99)
        mutate(m2, mutation_prob=0.5, mutation_std=0.01)

        for p1, p2 in zip(params1, m2.parameters()):
            assert torch.equal(p1, p2)
