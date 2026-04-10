# Morphogenesis-GA — Claude Code Project Guide

This file provides guidance to Claude Code when working with this repository.

> ⚠️ **Roadmap source of truth: Notion, not this file.**
> Before planning any work, read both canonical pages:
> - [🌍 Дорожная карта]([redacted]) — 7-phase product roadmap (sprints + phases)
> - [🔬 Research & Ops Backlog]([redacted]) — parallel Kanban track (R&D spikes, reliability, infra)
>
> Both live under [🧮 Morphogenesis-GA]([redacted]).
>
> An older 5-phase "technical subcard" used to live in this file. Its items
> (NSGA-II, adaptive mutation, HPO, test hardening, PyPI/Docker/HF Space)
> are **not obsolete** — they are cross-cutting R&D and release concerns
> that overlay the Notion sprints, tracked in the Research & Ops Backlog
> above. See the [Cross-cutting tracks](#cross-cutting-tracks) section
> below for the full mapping.

---

## Project Overview

**Morphogenesis-GA** (public name: _Morphogenesis-GA_ /
internal: _Morphogenesis-GA_) is a neuroevolution platform
that evolves neural network controllers (LittleLM transformers) to govern
cellular self-organization into target morphologies. Genetic algorithms train
populations of micro-controllers; each cell reads 8 neighbors and chooses an
action (stay, differentiate A/B, divide, die). The system uses curriculum
learning to progressively evolve toward complex shapes.

**North Star Metric (per Notion):** _Scenario Success Rate (SSR)_ — the
fraction of benchmark scenarios in which the policy reaches the target,
holds it under noise, recovers from damage, and does so within a step
budget. Not just "pretty GIF" — reproducible, measurable robustness.

**Strategic framing:** the current `cross → T` curriculum is a _benchmark_,
not the final product. The product target is an **interactive sandbox**
with programmable self-organization, self-healing demos, and explainable
reports.

## Architecture (current state, post Sprint 1A refactor)

```
run_train.py                      — Thin CLI (~112 lines): parse_args + main
                                    dispatches to core/agents pipelines
src/                              — Legacy domain primitives (stable, well-tested)
  world.py                        — 2D grid simulation (World class, cell types)
  model.py                        — LittleLM transformer controller
  ga.py                           — GA engine (selection, crossover, mutation)
  simulate.py                     — Step-by-step simulation loop (93% cov)
  fitness.py                      — Multi-objective fitness metrics (97% cov)
  targets.py                      — Target shape definitions (T, cross, bitmap)
  viz.py                          — Pygame renderer + GIF recording
  benchmark.py                    — Multi-seed evaluation framework (99% cov)
  utils.py                        — Seed, device, run directory helpers
core/                             — Service-Oriented Core (Phase 2 Sprint 1A)
  services/
    simulator.py                  — run_single_simulation (was in run_train.py)
    config_loader.py              — YAML config loading & validation
    factory.py                    — prepare_world, prepare_model, stability cfg
    cleanup.py                    — Late-cleanup ratio computation
    stats.py                      — Summarization helpers
    writer.py                     — TensorBoard SummaryWriter + run dir
    visualizer.py                 — Pygame Visualizer factory
  memory/
    metrics_logger.py             — CSV/JSON/TensorBoard plot export (98% cov)
agents/                           — Business-logic pipelines
  train_agent/
    pipeline.py                   — Full train_ga (3640 lines, 21 closures;
                                    follow-up decomposition pending)
    mutation.py                   — Mutation std + stem penalty schedules
    scoring.py                    — Benchmark score composition
    t_weights.py                  — T-phase structure weight scaling
  eval_agent/
    pipeline.py                   — play_best checkpoint replay (98% cov)
configs/                          — YAML experiment configurations
runs/                             — Training artifacts (gitignored)
tests/                            — pytest suite, 223 tests, 79–89% coverage
  test_golden.py                  — Reproducibility safety net (bit-identical)
  test_fitness.py, test_simulate.py, test_benchmark.py, ...
docs/
  adr/                            — Architectural decisions (5 ADRs)
  phase0_audit_report.md          — Pre-refactor audit (2026-04-09)
```

**What does NOT yet exist** (per Notion target, see Roadmap below):
`core/brain/`, `core/domain/`, `core/workflows/` (Temporal), `core/api/`
(FastAPI routes), `frontend/` (Next.js), `benchmarks/` (formal folders),
`datasets/`, full `infra/`. These are Sprint 1B → Sprint 3 work.

## Key Commands

```bash
# Environment setup (pyproject.toml exists)
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[viz,tb,dev]"

# Smoke test (no training, no viz)
python run_train.py --config configs/baseline_T.yaml --steps 0 --no-viz

# Minimal GA training (curriculum smoke, ~30s on CPU)
python run_train.py --config configs/curriculum_smoke.yaml --train-ga --no-viz

# Full curriculum training
python run_train.py --config configs/curriculum.yaml --train-ga

# Replay best model
python run_train.py --play-best runs/<run_id>/best.pt

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
  run them after any change touching `src/`, `core/`, or `agents/`
- Integration: `python run_train.py --config configs/curriculum_smoke.yaml --train-ga --no-viz`
- Coverage: currently **79% total** (89% if you exclude `agents/train_agent/pipeline.py`
  which is 63% pending closure decomposition). Target: ≥80% for any module you touch.
- **Test hardening is NOT considered done** — continuing as a Priority 1
  item in [🔬 Research & Ops Backlog]([redacted]).
  Reproducibility audit (baseline freeze + extended golden anchors) is the
  immediate next task on that track — see [`docs/research_backlog.md`](docs/research_backlog.md).

## Git & PR Conventions

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Never commit: `.venv/`, `runs/`, `__pycache__/`, `*.pyc`
- PRs: link Notion phase/sprint, list modified configs, attach IoU/fitness
  deltas and GIFs, show `pytest` + golden-test + smoke-test results
- Branch naming: `feat/<short-desc>`, `fix/<short-desc>`, `refactor/<short-desc>`

## Experiment Tracking

- `logging.run_name` in config must be descriptive for traceability
- Artifacts: `runs/<name>_<timestamp>/` → `best.pt`, `generations.csv`, `gifs/`, TensorBoard
- Clean stale TensorBoard event files before new sweeps
- **MLflow** is the Notion-specified canonical tracker for Phase 2+ (not yet wired)
- **W&B** is optional for public showcase runs

---

## Roadmap (source of truth: Notion)

**Canonical page:** [🌍 Дорожная карта]([redacted])
(inside [🧮 Morphogenesis-GA]([redacted]))

**Methodology:** Hybrid Agile — Discovery + Architecture Gate + 2-week
Scrum sprints + Kanban for research/ops. MVP window: **4–6 weeks**.

### Phase map (7 phases)

| Phase | Name | Duration | Key Deliverables |
|---|---|---|---|
| **0** | Discovery & Reframing | 3–5 days | Vision doc v1, Product one-pager, Benchmark ladder v1, ADR-001 positioning, Licensing memo |
| **1** | Architecture & Benchmark Design | 4–5 days | Service-oriented core scaffold, API contract draft, `benchmark_spec.md`, `dataset_candidates.md` |
| **2** | Development Sprint 1 (weeks 1–2) | 2 weeks | **1A** Core Runtime (`core/services/simulation_service`) · **1B** Benchmark Base (formal `benchmarks/cross`, `benchmarks/t_shape`, deterministic evals) · **1C** Basic FastAPI (`/runs`, `/benchmarks`, `/artifacts`, `/health`) · **1D** Next.js 16 frontend shell |
| **3** | Development Sprint 2 (weeks 3–4) | 2 weeks | Damage & Recover benchmark · Metrics Layer (SSR schema) · Interactive Controls (Canvas/WebGL) · Temporal async jobs |
| **4** | Development Sprint 3 (weeks 5–6) | 2 weeks | Report Agent (LLM) · Multi-target Conditioning Lite · Shareable hosted demo · OpenTelemetry/Grafana/Loki/Sentry observability |
| **5** | QA / Validation / Demo Readiness | 3–4 days | Regression tests, Playwright UI, benchmark freeze, demo script |
| **6** | Launch | 2–3 days | Public repo + hosted demo + outreach assets |
| **7** | Post-Launch | 30–60 days | Community loop, verticalization discovery, dataset track, enterprise readiness |

### Current status (as of 2026-04-10)

We are in the middle of **Phase 2, Sprint 1**.

| Sprint | Status | Notes |
|---|---|---|
| Phase 0 Discovery | 🟡 partial | ADRs exist; vision doc + licensing memo + benchmark ladder v1 — missing |
| Phase 1 Architecture & Benchmark Design | 🟡 partial | SOC scaffold exists (`core/services/`, `agents/`); `benchmark_spec.md`, `dataset_candidates.md`, API contract draft — missing |
| **Phase 2 Sprint 1A** Core Runtime | 🟢 ~90% | `run_train.py` shrunk 4933 → 112 lines. `core/services/simulator.py`, `config_loader`, `factory`, `cleanup`, `stats`, `writer`, `visualizer` all extracted. Follow-up: decompose 21 closures inside `agents/train_agent/pipeline.py` into reusable services (`training_service`, `artifact_service` boundaries) |
| **Phase 2 Sprint 1B** Benchmark Base | 🟡 ~30% | `src/benchmark.py` exists (99% cov) but not yet formalized as `benchmarks/cross/spec.yaml` + `benchmarks/t_shape/spec.yaml` folders. `benchmark_spec.md` doc — missing. Deterministic eval registry — missing |
| **Phase 2 Sprint 1C** Basic API | ❌ 0% | No FastAPI, no `/runs`, `/benchmarks`, `/artifacts`, `/health` endpoints yet |
| **Phase 2 Sprint 1D** Frontend Shell | ❌ 0% | No Next.js 16 / React / Tailwind / PixiJS frontend |
| Phase 3 | ❌ 0% | Damage & Recover not started; Temporal not introduced |
| Phase 4 | ❌ 0% | Report Agent / multi-target / observability not started |
| Phase 5–7 | ❌ 0% | |

### What has NOT been done despite past session claims

Earlier sessions reported "Phase 1 closed" — that was in the terminology of
an obsolete 5-phase technical subcard that lived in this file. In Notion
terms, the accurate statement is: **Sprint 1A of Phase 2 is ~90% complete;
Sprint 1B is ~30%; Sprint 1C and 1D are untouched.** Phases 3–7 are all
future work.

### Target directory layout (per Notion, for orientation)

```
core/
  brain/            ← policy_compiler, conditioning, reward_specs, report_reasoner (NOT YET)
  memory/           ← postgres, redis, object_store, run_registry (only metrics_logger today)
  services/         ← simulation_service, training_service, benchmark_service,
                      artifact_service, report_service, calibration_service,
                      telemetry_service, auth_service (only ~half done)
  domain/           ← envs, policies, tasks, metrics, validators (NOT YET)
  workflows/        ← temporal_{train,eval,report}_workflow (NOT YET)
  api/              ← routes_runs, routes_benchmarks, routes_reports, routes_artifacts (NOT YET)
agents/
  sandbox_agent/    ← (we have train_agent + eval_agent instead — rename pending)
  report_agent/     ← (NOT YET)
  devtools_agent/   ← (NOT YET)
frontend/           ← Next.js 16 (NOT YET)
benchmarks/{cross,t_shape,damage_recover,noisy_hold,multi_target}/  ← (NOT YET)
datasets/           ← synthetic, real_world_candidates, calibration (NOT YET)
infra/{docker,k8s,terraform,github_actions}/  ← (partial — only .github/)
docs/{architecture,product,benchmarks,adr,investor_pack}/  ← (only adr/ today)
```

### Planning conventions

- Before starting any feature work, **cross-reference the Notion roadmap**
  and identify which Sprint/Phase the work falls under.
- When writing PR descriptions and commit messages, use Notion phase
  terminology (e.g. "Phase 2 Sprint 1B: Formalize T benchmark"), not the
  old 5-phase technical subcard.
- If a task does not cleanly map to a Notion phase, that is a signal to
  either defer it or raise it with the product owner rather than inventing
  a parallel track.
- **Reproducibility** is a non-negotiable throughout: same seed must
  produce identical fitness trajectory ±0.001. Golden tests enforce this.

---

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

Use the following skills when working on related files:

| File(s) | Skill |
|---------|-------|
| `src/ga.py`, GA logic | `ga-optimization` |
| `src/fitness.py`, metrics | `fitness-engineering` |
| `src/model.py`, LittleLM | `pytorch-patterns` |
| `tests/**` | `tdd-workflow` |
| `configs/*.yaml` | `experiment-config` |
| `run_train.py`, `core/`, `agents/` refactoring | `python-refactoring` |

## Important Constraints

- **Reproducibility**: Same seed must produce identical fitness trajectory ±0.001
- **No breaking changes**: Existing configs must produce identical results after refactoring
- **GPU awareness**: Code must work on both CPU and CUDA; use `utils.select_device()`
- **Memory**: Population evaluation can be memory-intensive; batch when possible
- **Windows primary**: Development environment is Windows 11; use forward slashes in code paths
- **Parallel experiments**: A separate window may be running GA experiments — refactors
  are safe because Python caches modules at import, but do not touch `runs/` directories
  while experiments are live

---

## Cross-cutting tracks

An earlier autogenerated version of this file contained a 5-phase
"technical subcard" roadmap. Its items (NSGA-II, adaptive mutation, HPO,
test hardening, v1.0 release infrastructure) are **NOT obsolete** — they
are real cross-cutting concerns that **overlay** the Notion 7-phase
product roadmap and run on a parallel Kanban track.

**Canonical R&D backlog:** [🔬 Research & Ops Backlog]([redacted])
**Local mirror:** [`docs/research_backlog.md`](docs/research_backlog.md)

### Why they live in a parallel track

Notion roadmap methodology explicitly calls for Kanban alongside Scrum:

> _«У тебя есть исследовательская неопределённость: надо одновременно
> развивать эксперимент и продуктовую упаковку. [...] Параллельно:
> Kanban-доска для датасетов, R&D, benchmark backlog, infra/devops.»_

The old `CLAUDE.md` framing collapsed these into sequential phases, which
doesn't match how research + product work actually interact. The Research
& Ops Backlog restores the parallel structure.

### Mapping: old 5-phase items → current home

| Old `CLAUDE.md` item | Lives in | Status |
|---|---|---|
| Phase 0: Code audit + archaeology | Done — `docs/phase0_audit_report.md`, 5 ADRs | 🟢 |
| Phase 1: Refactor `run_train.py` monolith | Notion Phase 2 **Sprint 1A Core Runtime** | 🟢 ~90% |
| Phase 2: NSGA-II multi-objective GA | R&D Backlog — Priority 2.3 (after HPO) | 🔵 blocked on HPO |
| Phase 2: Adaptive mutation std | R&D Backlog — Priority 2.1 (**next after reproducibility**) | 🔴 |
| Phase 2: Hyperparameter optimization | R&D Backlog — Priority 2.2 | 🔵 blocked on reproducibility |
| Phase 3: Test hardening ≥80% coverage | R&D Backlog — Priority 1 (ongoing) | 🟡 79%, not closed |
| Phase 3: Reproducibility audit | R&D Backlog — Priority 1 (**active now**) | 🔴 next task |
| Phase 4: v1.0 PyPI release | Notion Phase 6 Launch — Soft Launch | 🔵 |
| Phase 4: v1.0 Docker image | Notion Tech Stack + Phase 6 | 🔵 |
| Phase 4: v1.0 HF Space demo | Notion Phase 6 Launch — Shareable Demo | 🔵 |

### Immediate execution order (strict)

1. **Now** → Reproducibility audit + baseline freeze *(R&D Priority 1, 1 session)*
2. → Adaptive mutation std *(R&D Priority 2.1, 1–2 sessions)*
3. → Hyperparameter optimization with Optuna *(R&D Priority 2.2, 2–3 sessions)*
4. → NSGA-II multi-objective GA *(R&D Priority 2.3, 2–3 sessions, optional)*
5. → MLflow + seed-reproducibility CI job *(parallel, opportunistic)*
6. → Docker / PyPI / HF Space demo *(Notion Phase 6 Launch prep)*

Notion sprints 1B/1C/1D continue independently of this track. R&D work
never blocks sprint work and vice versa.

### How future sessions should reason about this

- **Sprint feature work** → look at the Notion 7-phase map
- **R&D / reliability / release infra** → look at the Research & Ops Backlog
- If a task doesn't cleanly belong to either, raise it as a question
  before inventing a third track.
- If a session starts quoting the old 5-phase numbering (e.g. "Phase 2 =
  NSGA-II"), that's a signal it's working from stale CLAUDE.md context
  and should re-read the two Notion pages before planning.
