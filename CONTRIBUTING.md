# Contributing to morphogenesis-ga

Thanks for looking. This repository is an experimental record as much as a
codebase, so two of its rules are stricter than usual: **an experiment does not
start without a written hypothesis and a numeric success bar**, and **the golden
anchors are not to be re-baselined casually**.

Everything else is ordinary.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel first
pip install -e ".[viz,tb,dev,hpo]"
```

Install the CPU torch wheel **before** the extras. Otherwise pip resolves the
CUDA build, which is what CI does and what a laptop without an NVIDIA card
should also do.

## Tests

```bash
pytest tests/ --cov=src --cov=core --cov=agents --cov-report=term-missing
python run_train.py --config configs/baseline_T.yaml --steps 0 --no-viz   # smoke, ~5 s
```

353 tests: 311 fast plus 42 golden anchors.

### The golden anchors

`tests/test_golden.py` locks exact outputs for fixed seeds at `1e-6` tolerance.
They exist to catch the class of bug that does not raise — a refactor that
silently changes what the GA explores.

If a golden test fails, the default assumption is **your change altered
behaviour**, not that the anchor is stale. Before touching an expected value:

1. Say in the PR which anchor moved and by how much.
2. Explain the mechanism — which line changed the numeric path.
3. Confirm the new value is correct rather than merely different.

Re-baselining an anchor because it is red is how a reproducibility freeze dies.
`configs/reproducibility_freeze_v1.yaml` and the specs in `benchmarks/` are what
the anchors defend.

## If your PR involves a training run

The project has a
[Pre-Experiment Gate](.claude/skills/experiment-loop/SKILL.md), written after
ten ungoverned runs in which five retested a parameter already known dead. It
is not bureaucracy, it is the fix for a mistake that actually happened.

Before spending compute:

1. **Hypothesis** — one sentence: "X will lift MS-10 IoU above *baseline*
   because *mechanism*." If the mechanism does not fit in twenty words, the
   hypothesis is too fuzzy.
2. **Prior check** — read [`docs/DECISION_2026-04-19.md`](docs/DECISION_2026-04-19.md)
   (or its successor) and [`.claude/rules/experimental-findings.md`](.claude/rules/experimental-findings.md).
   If your axis is in the dead-ends section, stop and pick one from open
   questions. If the exact config was already run, report the prior result
   instead of re-running it.
3. **Success criterion** — the numeric bar, written down **before** the run.
   For example: "MS-10 IoU > 0.349, with ≥8/10 seeds at IoU ≥ 0.30."

Afterwards, report against the bar you set — including when the answer is no.
A documented dead end is a contribution; it is what stops the next person
burning a week on the same axis.

Multi-seed results only. A single lucky seed is not evidence, and the
seed-consistency montage in the README exists because of how far apart seeds
can land.

## Housekeeping

- **Do not commit training artifacts.** `runs/*` is ignored, with a small
  explicit allowlist for the checkpoints the docs render from. Adding to that
  allowlist needs a reason in the PR.
- **Change one variable per config.** If two are needed, say why.
- `ruff check .` is CI's first job; run it before pushing.

## What a good PR looks like

- **One concern per PR.**
- **Say what breaks if you are wrong.** The commit body is for the reasoning
  and the alternative you rejected, not just what changed.
- **A test that would have failed before** for anything in `src/fitness.py`,
  `src/ga.py` or `src/simulate.py` — those three define what every number in
  the repository means.
- **English in source and commit messages.**
- **If you used an AI assistant, say so in the PR description** and confirm you
  read and ran the result yourself. Assisted code is welcome; unreviewed
  generated code is not.

## Licence

MIT — see [LICENSE](LICENSE). By contributing you agree your work ships under it.
