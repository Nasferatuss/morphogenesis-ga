# ADR-005: pyproject.toml with pip

**Status**: Accepted
**Date**: 2026-04-09

## Context

Project has no dependency manifest — deps are installed ad-hoc via `pip install`. Options:
- `requirements.txt` — simple but no dev/test separation
- `pyproject.toml` + pip — modern standard, optional groups
- `pyproject.toml` + uv — faster but adds tool dependency
- `poetry` / `pdm` — heavy for this project size

## Decision

Use `pyproject.toml` with `[project.optional-dependencies]` groups:

```toml
[project]
dependencies = ["torch", "pyyaml", "numpy", "imageio"]

[project.optional-dependencies]
viz = ["pygame", "matplotlib"]
dev = ["ruff", "mypy", "pytest", "pytest-cov", "pytest-benchmark"]
tb = ["tensorboard"]
```

Install: `pip install -e ".[dev,viz,tb]"`

Migrate to `uv` later if team grows or CI speed matters.

## Rationale

- `pyproject.toml` is PEP 621 standard
- Optional groups keep install lean for CI (skip pygame)
- No new tool dependency (pip is built-in)
- Easy migration path to uv (same pyproject.toml format)

## Consequences

- `pip install -e .` replaces ad-hoc `pip install torch pyyaml ...`
- Version pins in pyproject.toml ensure reproducible environments
- `.venv` creation stays manual (`python -m venv .venv`)
