# ADR-003: Testing Strategy

**Status**: Accepted
**Date**: 2026-04-09

## Context

The project has zero tests. We need a safety net before refactoring `run_train.py` (3,737 lines in `train_ga()` alone). The project uses PyTorch with GA (no backprop), fixed-seed reproducibility, and multi-objective fitness.

## Decision

Three-tier testing approach:

### Tier 1: Unit Tests (write first)
- `src/fitness.py` — all fitness components (IoU, area, symmetry)
- `src/ga.py` — selection, crossover, mutation (seeded)
- `src/world.py` — grid operations, cell actions
- `src/targets.py` — target mask generation
- `src/model.py` — forward pass shape verification

### Tier 2: Golden Tests (write before refactoring)
- Fixed seed + config → expected fitness trajectory ±0.001
- Captures current `run_train.py` behavior as baseline
- If golden test breaks after extraction → revert

### Tier 3: Integration Tests (write during Phase 1)
- GA loop: init → evaluate → select → crossover → mutate (5 gens)
- Full pipeline: config → train → checkpoint → load → verify

## Tools

- `pytest` + `pytest-cov` (already installed)
- `pytest-benchmark` for performance regression
- Coverage target: ≥80% for any module touched

## Rationale

- Unit tests on `src/` modules are safe — these modules are already clean and small
- Golden tests protect against behavioral drift during `run_train.py` extraction
- No mocking PyTorch — use real tensors with small grid sizes for speed

## Consequences

- Tests must run in <30 seconds total (small grids, few generations)
- All tests must be deterministic (seeded)
- `tests/` directory becomes a gate for all future PRs
