# Morphogenesis-GA — Claude Code Project Guide

This file is the operational guide for Claude Code in this repository.
Heavyweight governance lives in dedicated rule files — read them before
planning non-trivial work.

> **Must-read rule files** (under [`.claude/rules/`](.claude/rules/)):
> - [`experimental-findings.md`](.claude/rules/experimental-findings.md)
>   — locked baseline, 10 critical gotchas, champion recipe. Read before
>   proposing any training run, HPO sweep, or ablation.
> - [`roadmap-status.md`](.claude/rules/roadmap-status.md) — Notion-backed
>   7-phase roadmap + current Sprint status + R&D Kanban mapping. Read
>   before planning feature work.
>
> **Live experiment state:** [`docs/DECISION_2026-04-19.md`](docs/DECISION_2026-04-19.md)
> is the running log of experiments, dead ends, open questions, and the
> single next experiment. If this file and the Decision Doc disagree,
> the Decision Doc wins.
>
> **Roadmap source of truth:** Notion —
> [🌍 Дорожная карта]([redacted])
> and [🔬 Research & Ops Backlog]([redacted]),
> both under [🧮 Morphogenesis-GA]([redacted]).

---

## Project Overview

**Morphogenesis-GA** (public name: _Morphogenesis-GA_ /
internal: _Morphogenesis-GA_) is a neuroevolution
platform that evolves neural network controllers (LittleLM transformers)
to govern cellular self-organization into target morphologies. Genetic
algorithms train populations of micro-controllers; each cell reads 8
neighbours and chooses an action (stay, differentiate A/B, divide N/S/E/W,
die). The system uses curriculum learning to progressively evolve toward
complex shapes.

**North Star Metric:** _Scenario Success Rate (SSR)_ — fraction of
benchmark scenarios in which the policy reaches the target, holds it
under noise, recovers from damage, within a step budget. Not just a
"pretty GIF" — reproducible, measurable robustness.

**Strategic framing:** the `cross → T` curriculum is a _benchmark_, not
the final product. The product target is an **interactive sandbox** with
programmable self-organization, self-healing demos, and explainable
reports.

## Architecture (current state, post Sprint 1A refactor)

```
run_train.py                      — Thin CLI (~112 lines): parse_args + main
src/                              — Legacy domain primitives (stable, well-tested)
  world.py                        — 2D grid simulation (with directional divide actions)
  model.py                        — LittleLM transformer controller
  ga.py                           — GA engine (selection, crossover, mutation)
  simulate.py                     — Step-by-step simulation loop (93% cov)
  fitness.py                      — Multi-objective fitness metrics (97% cov)
  targets.py                      — Target shape definitions (T, cross, bitmap)
  viz.py                          — Pygame renderer + GIF recording
  benchmark.py                    — Multi-seed evaluation framework (99% cov)
  utils.py                        — Seed, device, run directory helpers
core/
  services/                       — simulator, config_loader, factory, cleanup,
                                    stats, writer, visualizer, hpo, benchmark_runner,
                                    mini_transfer
  memory/                         — metrics_logger, mlflow_tracker (soft dependency)
agents/
  train_agent/
    pipeline.py                   — train_ga (closures 21 → 17, Round 4 deferred)
    mutation.py, scoring.py, t_weights.py, adaptive_mutation.py
    evaluator.py / EvaluatorState — extracted via 495a42b
  eval_agent/
    pipeline.py                   — play_best checkpoint replay (98% cov)
benchmarks/
  cross/, t_shape/                — formal specs (57302d6, Sprint 1B partial)
configs/                          — YAML experiment configurations
runs/                             — Training artifacts (gitignored)
scripts/
  multiseed_bench.py              — Multi-seed IoU benchmark (CLI)
  build_leaderboard.py            — Rank every runs/*/best.pt by MS mean
  dump_grid.py                    — ASCII viz of final grid
  hpo_sweep.py                    — Optuna HPO CLI
tests/                            — pytest: 311 fast + 9 v1 anchors + 33 v2 anchors
docs/
  adr/                            — 5 architectural decisions
  benchmarks/                     — baseline_v1.md, baseline_v2.md
  research/                       — hpo_v1_report.md, hpo_v2_report.md, adaptive_mutation_report.md
  DECISION_2026-04-19.md          — Live experiment decision loop
```

**What does NOT yet exist:** `core/brain/`, `core/domain/`,
`core/workflows/` (Temporal), `core/api/` (FastAPI), `frontend/`
(Next.js 16), full `benchmarks/{damage_recover,noisy_hold,multi_target}`,
`datasets/`, full `infra/`. See [`roadmap-status.md`](.claude/rules/roadmap-status.md).

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
- **4-space indentation**, `snake_case` functions/variables, `PascalCase` classes
- Logging: `[INFO]`, `[WARN]`, `[ERROR]` prefixes matching `run_train.py` style
- Pure helpers preferred; in-place mutation only where precedent exists (`ga.py`, `world.py`)
- Config keys: lowercase `snake_case` in YAML
- All new code must include type annotations on public functions
- **Core never imports from agents** (ADR-001); agents may import from core

## Testing

- Framework: `pytest` with `pytest-cov`, `pytest-benchmark`
- Seed determinism: use `utils.set_seed()` so GA mutations replicate
- **Golden tests** in `tests/test_golden.py` are the refactor safety net —
  run them after any change touching `src/`, `core/`, or `agents/`.
  Current anchors: `TestGoldenBaselineV1` (9 @ 1e-9, CPU) +
  `TestGoldenBaselineV2` (33 @ 1e-6, CUDA)
- Integration: `python run_train.py --config configs/curriculum_smoke.yaml --train-ga --no-viz`
- Coverage: **79% total** (89% excluding `agents/train_agent/pipeline.py`
  which is 63% pending Round 4 closure decomposition). Target: ≥80% for
  any module you touch.
- Test hardening is ongoing as R&D Priority 1 — see
  [`roadmap-status.md`](.claude/rules/roadmap-status.md).

## Git & PR Conventions

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Never commit: `.venv/`, `runs/`, `__pycache__/`, `*.pyc`
- PRs: link Notion phase/sprint, list modified configs, attach IoU/fitness
  deltas and GIFs, show `pytest` + golden-test + smoke-test results
- Branch naming: `feat/<short-desc>`, `fix/<short-desc>`, `refactor/<short-desc>`

## Experiment Tracking

- `logging.run_name` in config must be descriptive for traceability
- Artifacts: `runs/<name>_<timestamp>/` → `best.pt`, `generations.csv`, `gifs/`, TensorBoard
- Clean stale TensorBoard event files before new sweeps
- **MLflow** soft-dependency landed via `c349da8`; pipeline wiring
  deferred (RNG drift risk vs baseline v2 anchors)
- **W&B** is optional for public showcase runs
- Before any new training run: satisfy the **Pre-Experiment Gate** in
  [`.claude/skills/experiment-config/SKILL.md`](.claude/skills/experiment-config/SKILL.md)
  (read Decision Doc, write hypothesis, define success criterion)

### Experiment Journal convention

Every training run with a fitness-delta hypothesis is recorded in a
dated Decision Doc. This replaces the 2026-04-12 → 2026-04-19 pattern of
experiments scattered across commits, configs, and memory with no single
log.

- **Active journal:** [`docs/DECISION_2026-04-19.md`](docs/DECISION_2026-04-19.md)
  — §2 experiments table, §3 dead ends, §4 open questions, §5 next
  experiment.
- **New entries** go in the current journal until §2 exceeds ~25 rows
  or a month has passed. Then start a successor
  `docs/DECISION_YYYY-MM-DD.md` that opens with "Supersedes
  [previous]" and carries forward §1 (baseline), §3 (dead ends), §4
  (open questions). Update the strategic memory pointer to the new
  file.
- **Workflow:** the [`experiment-loop`](.claude/skills/experiment-loop/SKILL.md)
  skill enforces the 8 steps (hypothesis → prior check → success
  criterion → config → diff → run+bench → record → decide). Step 7
  appends the journal row; step 8 updates §3/§4.
- **Read-only snapshot:** `/strategy` command
  ([`.claude/commands/strategy.md`](.claude/commands/strategy.md))
  reports the current locked baseline, dead ends, open questions, and
  next experiment in under 30 seconds. Use at the start of any session
  that might launch a training run.
- **Benchmark delta:** `scripts/multiseed_bench.py` accepts
  `--baseline-iou` (default 0.329, the locked MS-10) and prints the
  delta vs baseline in the report — use it in step 7 instead of
  restating the absolute IoU.
- **Stop-hook reminder:** a post-turn hook
  (`scripts/hooks/decision-doc-reminder.sh`) flags sessions that
  touched `configs/exp_*`, `configs/curriculum_*`, or `runs/` without
  updating the Decision Doc or strategic memory.

## Custom Agents (`.claude/agents/`)

| Agent | Purpose |
|-------|---------|
| `refactorer.md` | Decomposes monolithic code into Service-Oriented Core |
| `python-reviewer.md` | PEP 8, type safety, security review |
| `pytorch-resolver.md` | Tensor shape, CUDA, gradient error resolution |
| `tdd-guide.md` | Test-first development, Red-Green-Refactor cycles |
| `ga-optimizer.md` | GA hyperparameter tuning and algorithm improvements |
| `experiment-analyst.md` | Analyzes training runs, identifies patterns |
| `planner.md` | Implementation planning with phased breakdown |

## Custom Commands (`.claude/commands/`)

| Command | Purpose |
|---------|---------|
| `/train` | Launch GA training with config validation |
| `/benchmark` | Run multi-seed evaluation on a checkpoint |
| `/experiment-report` | Generate summary of training run |
| `/refactor-extract` | Extract module from run_train.py |
| `/test-module` | Generate tests for a specific src/ module |
| `/fitness-debug` | Debug fitness calculation for a specific model |

## Skills

Skill descriptions are TRIGGER-based — each skill's `SKILL.md` declares
the exact conditions under which it should activate.

| File(s) | Skill |
|---------|-------|
| `src/ga.py`, GA logic | `ga-optimization` |
| `src/fitness.py`, metrics | `fitness-engineering` |
| `src/model.py`, LittleLM | `pytorch-patterns` |
| `tests/**` | `tdd-workflow` |
| `configs/*.yaml`, new experiment requests | `experiment-config` |
| `run_train.py`, `core/`, `agents/` refactoring | `python-refactoring` |

## Important Constraints

- **Reproducibility**: Same seed must produce identical fitness trajectory
  ±0.001 (golden anchors: 1e-9 for v1 CPU, 1e-6 for v2 CUDA)
- **No breaking changes**: Existing configs must produce identical results
  after refactoring
- **GPU awareness**: Code must work on both CPU and CUDA; use
  `utils.select_device()`
- **Memory**: Population evaluation can be memory-intensive; batch when possible
- **Windows primary**: Development environment is Windows 11; use forward
  slashes in code paths
- **Parallel experiments**: A separate window may be running GA experiments —
  refactors are safe because Python caches modules at import, but do not
  touch `runs/` directories while experiments are live
