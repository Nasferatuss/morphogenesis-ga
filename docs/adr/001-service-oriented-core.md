# ADR-001: Service-Oriented Core Architecture

**Status**: Accepted
**Date**: 2026-04-09

## Context

`run_train.py` is a 4,933-line monolith. The `train_ga()` function alone is 3,737 lines. This makes the code untestable, unreviewable, and fragile. Any change risks breaking unrelated functionality.

## Decision

Decompose into **Service-Oriented Core** pattern:

```
core/services/    — Reusable computation (GA engine, simulator, fitness)
core/brain/       — Neural network logic (LittleLM, encoding)
core/memory/      — Persistence (checkpoints, metrics, run storage)
agents/           — Business workflows (train pipeline, eval pipeline)
main.py           — Thin CLI (<200 lines)
```

**Agents import from core; core never imports from agents.**

## Rationale

- Each service is independently testable
- Curriculum logic (biggest complexity) is isolated in `agents/train_agent/`
- New agents (explore, optimize) can reuse core services
- Matches project roadmap Phase 1

## Consequences

- All existing configs must produce identical results post-refactoring
- Golden test (same seed → same fitness ±0.001) required before each extraction
- One module extracted per PR — atomic changes
