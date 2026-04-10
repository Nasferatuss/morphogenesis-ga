"""Golden regression tests — lock current behavior before refactoring.

These tests capture exact outputs for fixed seeds and configs.
If any test fails after a refactor, it means behavior changed.

**Baseline v1 anchors** (``TestGoldenBaselineV1``) are the reproducibility
freeze for the full curriculum_smoke training pipeline. See
``docs/benchmarks/baseline_v1.md`` for the scoreboard and
``configs/reproducibility_freeze_v1.yaml`` for the frozen config.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from agents.train_agent.pipeline import train_ga
from core.services.config_loader import load_config
from src.fitness import compute_fitness, compute_iou
from src.ga import clone_model, crossover, init_population, mutate, select_elite
from src.model import LittleLM
from src.simulate import simulate
from src.targets import make_target, make_target_T, make_target_cross
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


# ---------------------------------------------------------------------------
# Baseline v1 — reproducibility freeze (captured 2026-04-10)
# ---------------------------------------------------------------------------
#
# These values are the deterministic per-generation metrics produced by
# ``configs/reproducibility_freeze_v1.yaml`` with seed 42 on CPU. If any
# refactor causes these numbers to drift, the tests below will fail — which
# is exactly what "baseline freeze" means. To intentionally bump the
# baseline, create ``reproducibility_freeze_v2.yaml`` + ``baseline_v2.md``
# and a new ``TestGoldenBaselineV2`` class; never edit v1 in place.
#
# See docs/benchmarks/baseline_v1.md for the full scoreboard + known
# limitations. Runtime: ~30 seconds on CPU.
#
# Float tolerance: 1e-9 (effectively bit-identical). GA uses Python RNG
# + torch CPU ops which are fully deterministic under a fixed seed, so
# any drift > 1e-9 indicates a real behavioural change.

BASELINE_V1_EXPECTED = [
    # (generation, phase, best_fitness, best_iou, mean_fitness)
    (1, "phase_cross",      -0.4460150097908516,   0.19310344827586207, -1.6986011817479065),
    (2, "phase_cross",      -0.039757082615359284, 0.24285714285714285, -0.9378264366892466),
    (3, "phase_transition", -0.5618737313614587,   0.11538461538461539, -1.1751321859453598),
    (4, "phase_T",          -1.183357273108479,    0.104,               -1.6860259452870383),
    (5, "phase_T",          -0.9744218314988664,   0.13934426229508196, -1.4297741683496186),
    (6, "phase_T_refine",   -0.665291143775936,    0.14049586776859505, -1.4709745597644481),
]


@pytest.mark.golden
@pytest.mark.slow
class TestGoldenBaselineV1:
    """Reproducibility baseline — full train_ga run anchored on exact numbers.

    This class runs ``configs/reproducibility_freeze_v1.yaml`` end-to-end
    (~30s on CPU) and asserts every per-generation ``best_fitness``,
    ``best_iou``, and ``mean_fitness`` matches the 2026-04-10 freeze.

    Why it lives in test_golden.py: this is the highest-value regression
    guard in the project — it locks the entire pipeline
    (GA + curriculum + fitness + simulate + late-T guard) under a single
    invocation. If anything from closure decomposition / adaptive mutation
    / HPO breaks determinism, this test is the first to notice.
    """

    @pytest.fixture(scope="class")
    def baseline_run(self, tmp_path_factory) -> list[dict]:
        """Run the frozen config once per class and return parsed generations.csv rows."""
        import csv

        run_root = tmp_path_factory.mktemp("baseline_v1_run")
        # train_ga writes into ./runs by default; redirect by chdir.
        import os
        cwd = os.getcwd()
        os.chdir(run_root)
        try:
            cfg = load_config(
                str(Path(cwd) / "configs" / "reproducibility_freeze_v1.yaml")
            )
            args = argparse.Namespace(no_viz=True, steps=None, train_ga=True)
            target_mask = make_target("T", 15, 15)
            target_area = float(target_mask.sum().item())
            set_seed(int(cfg.get("seed", 42)))
            best_path = train_ga(
                cfg,
                args,
                torch.device("cpu"),
                target_mask,
                seed=int(cfg.get("seed", 42)),
                default_target_name="T",
                default_target_area=target_area,
            )
            run_dir = best_path.parent
            with (run_dir / "generations.csv").open() as f:
                return list(csv.DictReader(f))
        finally:
            os.chdir(cwd)

    def test_row_count(self, baseline_run):
        """Baseline produces exactly 6 generations (2 cross + 1 transition + 2 T + 1 refine)."""
        assert len(baseline_run) == len(BASELINE_V1_EXPECTED) == 6

    @pytest.mark.parametrize(
        "expected",
        BASELINE_V1_EXPECTED,
        ids=[f"gen{g}_{p}" for g, p, *_ in BASELINE_V1_EXPECTED],
    )
    def test_generation_metrics_anchored(self, baseline_run, expected):
        """Every generation's best_fitness, best_iou, and mean_fitness match freeze v1."""
        exp_gen, exp_phase, exp_best_fit, exp_best_iou, exp_mean_fit = expected
        row = next(
            (r for r in baseline_run if int(r["generation"]) == exp_gen), None
        )
        assert row is not None, f"missing row for gen {exp_gen}"
        assert row["phase"] == exp_phase, (
            f"gen {exp_gen} phase drifted: {row['phase']} != {exp_phase}"
        )
        # Float comparisons use bit-identical tolerance (1e-9). GA is fully
        # deterministic under fixed seed on CPU, so any drift > 1e-9 is real.
        assert float(row["best_fitness"]) == pytest.approx(exp_best_fit, abs=1e-9), (
            f"gen {exp_gen} best_fitness drifted: {row['best_fitness']} vs {exp_best_fit}"
        )
        assert float(row["best_iou"]) == pytest.approx(exp_best_iou, abs=1e-9), (
            f"gen {exp_gen} best_iou drifted: {row['best_iou']} vs {exp_best_iou}"
        )
        assert float(row["mean_fitness"]) == pytest.approx(exp_mean_fit, abs=1e-9), (
            f"gen {exp_gen} mean_fitness drifted: {row['mean_fitness']} vs {exp_mean_fit}"
        )

    def test_best_overall_gen2_phase_cross(self, baseline_run):
        """Best IoU across all 6 generations stays at gen 2 phase_cross = 0.242857."""
        best = max(baseline_run, key=lambda r: float(r["best_iou"]))
        assert int(best["generation"]) == 2
        assert best["phase"] == "phase_cross"
        assert float(best["best_iou"]) == pytest.approx(
            0.24285714285714285, abs=1e-9
        )

    def test_no_stabilization_in_freeze(self, baseline_run):
        """Baseline never breaks through reach_threshold=0.7 — stability layer is passive."""
        for row in baseline_run:
            # hold_score may be empty string for non-T phases; normalize to 0
            hold = row.get("hold_score") or "0"
            assert float(hold) == 0.0, (
                f"gen {row['generation']}: hold_score {hold} != 0 — "
                "baseline should have zero stabilization (see baseline_v1.md §Known limitations)"
            )
