# ADR-002: Fix Critical Bugs Before Refactoring

**Status**: Accepted
**Date**: 2026-04-09

## Context

Ruff analysis found 14 undefined name errors (F821) in `run_train.py`. These are real bugs that crash at runtime when late T-guard or checkpoint copy code paths are hit. We also have 46 unused variables (dead code from iterations).

## Decision

1. **Fix all 14 F821 bugs first** — before any structural refactoring
2. **Clean F841 unused variables** as a separate commit
3. **Do NOT change behavior** — only fix variable references and remove dead code
4. Each fix category is a separate commit with clear scope

## Rationale

- Refactoring code with known bugs means propagating bugs into new modules
- F821 fixes are small, low-risk, and independently verifiable
- Cleaning dead code reduces noise for subsequent refactoring
- Separate commits make it easy to revert if something breaks

## Consequences

- Need to identify correct variable names for F821 (may require reading surrounding context)
- Smoke test after each fix batch to verify no regression
