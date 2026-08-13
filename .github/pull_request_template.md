## What this changes

<!-- One paragraph. What is different after this PR, and why. -->

## Why this way

<!-- The reasoning, the alternative you rejected, and anything you are unsure
     about. Delete this section only if the change is genuinely obvious. -->

## How it was tested

<!-- Commands you actually ran, and what they printed. -->

- [ ] `ruff check .`
- [ ] `pytest tests/ --cov=src --cov=core --cov=agents --cov-report=term-missing`
- [ ] `python run_train.py --config configs/baseline_T.yaml --steps 0 --no-viz`

## Golden anchors

<!-- tests/test_golden.py locks exact outputs at 1e-6. A red anchor means
     behaviour changed — the default assumption is that this PR changed it,
     not that the anchor is stale. -->

- [ ] All golden tests still pass
- [ ] …or an expected value was updated, and this PR names the anchor, the
      delta, the line that changed the numeric path, and why the new value is
      correct rather than merely different

## If this involves a training run

<!-- Delete if this is pure refactoring or infrastructure.
     See the Pre-Experiment Gate in .claude/skills/experiment-loop/SKILL.md — all
     three items are written BEFORE the run, not reconstructed after it. -->

- **Hypothesis** (one sentence, with the mechanism):
- **Prior check** — checked the Decision Doc dead-ends and
  `.claude/rules/experimental-findings.md`; this axis is not already settled:
- **Success criterion**, written before the run:
- **Result against that bar** (a "no" is a valid and useful outcome):
- **Seeds** (single-seed results are not evidence):

## Housekeeping

- [ ] No training artifacts committed — `runs/*` stays ignored, apart from the
      existing docs allowlist
- [ ] One variable changed per config, or the PR says why more were needed

## AI usage

<!-- Required. "None", or which assistant and for what. Assisted code is fine;
     unreviewed generated code is not. -->

- [ ] I have read and run every line I am submitting
