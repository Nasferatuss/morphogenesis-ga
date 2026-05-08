# Morphogenesis-GA — Claude Code Project Guide

This file is the operational guide for Claude Code in this repository.

> **Must-read rule files:**
> - [`.claude/rules/experimental-findings.md`](.claude/rules/experimental-findings.md)
>   — locked baseline, gotcha catalog, champion recipe. Read before
>   proposing any training run.
> - [`docs/DECISION_2026-04-19.md`](docs/DECISION_2026-04-19.md) — running
>   log of experiments, dead ends, open questions, next experiment.
> - [`.claude/skills/experiment-loop/SKILL.md`](.claude/skills/experiment-loop/SKILL.md)
>   — 8-step Pre-Experiment Gate.

---

## Project Overview

**Morphogenesis-GA** is a neuroevolution platform that evolves neural
network controllers (LittleLM transformers) to govern cellular
self-organization into target morphologies. Genetic algorithms train
populations of micro-controllers; each cell reads 8 neighbours and
chooses an action (stay, differentiate A/B, divide N/S/E/W, die). The
system uses curriculum learning to progressively evolve toward complex
shapes.

## Architecture

```
run_train.py        — Thin CLI entry point
src/                — Domain primitives
  world.py          — 2D grid simulation (with directional divide actions)
  model.py          — LittleLM transformer controller
  ga.py             — GA engine (selection, crossover, mutation)
  simulate.py       — Step-by-step simulation loop
  fitness.py        — Multi-objective fitness metrics
  targets.py        — Target shape definitions (T, cross, bitmap)
  viz.py            — Pygame renderer + GIF recording
  benchmark.py      — Multi-seed evaluation framework
  utils.py          — Seed, device, run directory helpers
core/
  services/         — simulator, config_loader, factory, cleanup,
                      stats, writer, visualizer, hpo
  memory/           — metrics_logger, mlflow_tracker (soft dep)
agents/
  train_agent/      — train_ga pipeline + supporting modules
  eval_agent/       — play_best checkpoint replay
benchmarks/         — cross/, t_shape/ — formal specs
configs/            — YAML experiment configurations
runs/               — Training artifacts (gitignored)
scripts/            — multiseed_bench, build_leaderboard, dump_grid,
                      hpo_sweep, render_*.py for article assets
tests/              — pytest with golden-anchor reproducibility tests
docs/
  adr/              — 5 architectural decisions
  assets/           — GIFs for the Habr article
  DECISION_2026-04-19.md — Live experiment decision loop
```

## Key Commands

```bash
# Environment setup
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[viz,tb,dev,hpo]"

# Smoke test (no training, no viz)
python run_train.py --config configs/baseline_T.yaml --steps 0 --no-viz

# Minimal GA training (curriculum smoke, ~30s on CPU)
python run_train.py --config configs/curriculum_smoke.yaml --train-ga --no-viz

# Full curriculum training
python run_train.py --config configs/curriculum_hpo_v1.yaml --train-ga

# Replay best model
python run_train.py --play-best runs/<run_id>/best.pt

# Multi-seed benchmark (MANDATORY before declaring a new champion)
python scripts/multiseed_bench.py runs/<run_id>/best.pt

# TensorBoard
tensorboard --logdir runs --port 6006

# Tests + coverage
pytest tests/ --cov=src --cov=core --cov=agents --cov-report=term-missing

# Linting
ruff check src/ core/ agents/ run_train.py tests/
mypy src/ core/ agents/ --ignore-missing-imports
```

## Coding Standards

- **Python 3.12+**, `from __future__ import annotations` in every module
- 4-space indentation, `snake_case` functions/variables, `PascalCase` classes
- Logging: `[INFO]`, `[WARN]`, `[ERROR]` prefixes matching `run_train.py` style
- Pure helpers preferred; in-place mutation only where precedent exists
- Config keys: lowercase `snake_case` in YAML
- Type annotations on public functions
- **Core never imports from agents** (ADR-001); agents may import from core

## Testing

- Framework: `pytest` with `pytest-cov`
- Seed determinism: use `utils.set_seed()` so GA mutations replicate
- **Golden tests** in `tests/test_golden.py` are the refactor safety net.
  Anchors: `TestGoldenBaselineV1` (9 @ 1e-9, CPU) +
  `TestGoldenBaselineV2` (33 @ 1e-6, CUDA)
- Integration: `python run_train.py --config configs/curriculum_smoke.yaml --train-ga --no-viz`

## Git & PR Conventions

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Never commit: `.venv/`, `runs/` (except gitignore exceptions for the
  champion checkpoints), `__pycache__/`, `*.pyc`
- Branch naming: `feat/<short-desc>`, `fix/<short-desc>`, `refactor/<short-desc>`

## Experiment Journal

Every training run with a fitness-delta hypothesis is recorded in
[`docs/DECISION_2026-04-19.md`](docs/DECISION_2026-04-19.md): hypothesis,
success criterion, result, verdict. The
[`experiment-loop`](.claude/skills/experiment-loop/SKILL.md) skill
enforces an 8-step gate before launch (hypothesis → prior check →
success criterion → config → diff → run+bench → record → decide).

`scripts/multiseed_bench.py` accepts `--baseline-iou` (default 0.329,
the locked MS-10) and prints the delta vs baseline.

## Important Constraints

- **Reproducibility**: same seed must produce identical fitness trajectory
  ±0.001 (golden anchors: 1e-9 for v1 CPU, 1e-6 for v2 CUDA)
- **GPU awareness**: code must work on both CPU and CUDA; use
  `utils.select_device()`
- **Windows primary**: development environment is Windows 11
