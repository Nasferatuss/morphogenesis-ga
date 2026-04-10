# ADR-004: Keep Raw YAML for Now, Migrate to Hydra in Phase 2

**Status**: Accepted
**Date**: 2026-04-09

## Context

Current config system uses raw `yaml.safe_load()` in `load_config()` (6 lines). Options considered:
- **Raw YAML** (current) — simple, no dependencies
- **Hydra** — hierarchical configs, overrides, multirun, structured validation
- **Pydantic** — validation only, no override syntax

## Decision

**Phase 1**: Keep raw YAML. Extract `load_config()` to `core/services/config_loader.py` with added validation (check required keys, value ranges).

**Phase 2**: Migrate to Hydra when implementing hyperparameter sweeps. The config structure is already Hydra-compatible (nested dicts with snake_case keys).

## Rationale

- Adding Hydra during refactoring is too many moving parts at once
- Current 6 configs work fine with raw YAML
- Validation can be added without Hydra
- Hydra migration will be natural when we add Optuna sweeps (Phase 2)

## Consequences

- Phase 1 configs remain backward-compatible
- Config validation errors will be explicit Python exceptions (not Hydra schema errors)
- Phase 2 will require config file restructuring for Hydra groups
