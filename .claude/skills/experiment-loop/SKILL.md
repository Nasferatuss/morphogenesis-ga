# Experiment Loop Skill

## When to Use
TRIGGER when:
- user asks to "run an experiment" / "try X" / "test Y" on the T-shape,
  cross, or damage-recover target
- user asks to "launch training" against any `configs/*.yaml` with
  `--train-ga`
- user proposes a new fitness-weight config, new mutation schedule, or
  new curriculum variant on a live target
- user asks for an HPO sweep, ablation, or multi-seed comparison
- user references the Decision Doc and wants to add an entry

SKIP when the work is pure refactoring or infrastructure (no training
run on the critical path).

## Why this skill exists

From 2026-04-11 to 2026-04-19 the project ran ~10 experiments without a
governed loop. Five of them retested a parameter already proven dead
(`stem_trunk_credit` float). The root cause was no enforced gate between
"we thought of this" and "we ran training."

This skill enforces an **8-step loop**. Steps 1–3 are MANDATORY before
launching any training run. Steps 7–8 are MANDATORY after the run,
before the user can propose the next experiment. If you cannot complete
a step, ask the user — do not guess.

## The Loop

### 1. HYPOTHESIS (mandatory, before any config edit)
Write one sentence: "X will lift MS-10 IoU above [baseline] because
[mechanism]." If you cannot name the mechanism in ≤20 words, the
hypothesis is too fuzzy — refine or drop.

### 2. PRIOR CHECK (mandatory)
Open and read:
- [`docs/DECISION_2026-04-19.md`](../../../docs/DECISION_2026-04-19.md)
  (latest `docs/DECISION_*.md` if a successor exists)
- Memory files whose names match the proposed parameter (e.g.
  `project_credit_experiments_*.md` for any `stem_trunk_*` work)
- [`.claude/rules/experimental-findings.md`](../../rules/experimental-findings.md)

Verdict:
- If the proposed axis is in §3 of the Decision Doc (dead ends) → **STOP**,
  tell the user, suggest an axis from §4 (open questions) instead.
- If the exact config was already run in §2 of the Decision Doc →
  **STOP**, report the prior result, do not re-run.
- Otherwise continue.

### 3. SUCCESS CRITERION (mandatory, written BEFORE run)
Record the exact numerical bar the experiment must clear. Example:
"Success if MS-10 IoU > 0.349 (= baseline 0.329 + 0.02 delta), with
≥8/10 seeds reaching IoU ≥ 0.30." Without this, step 8 has nothing
to compare against.

### 4. CONFIG (create / modify)
- Start from the closest baseline config (usually
  `configs/curriculum_hpo_v1.yaml` or the last entry you promoted).
- Change ONE variable when possible. If multiple changes are required
  (e.g. rename + value change together), document each in step 5.
- Set `logging.run_name` to match the hypothesis (e.g.
  `exp_directional_presence_bool`).

### 5. DIFF vs baseline
Explicitly state which fields differ from the baseline config and why
each change is motivated. Show the diff to the user before running.

### 6. RUN + BENCHMARK
- `python run_train.py --config configs/<new>.yaml --train-ga --no-viz`
- On completion, **always** run
  `python scripts/multiseed_bench.py runs/<new_dir>/best.pt` (or
  `best_t.pt` / `best_t_refine.pt` as appropriate for curriculum runs).
- Training peak IoU alone is NOT sufficient (see
  `experimental-findings.md` gotcha #1).

### 7. RECORD (mandatory, after run)
Append one row to the Decision Doc §2 experiments table with:
- Date, config name, hypothesis, MS-10 mean + reach fractions,
  verdict.

If the result closes an Open Question in §4 → remove / update that
entry. If the result proves a parameter dead → add it to §3 with a
one-line "do NOT retest" note.

### 8. DECIDE (mandatory, before proposing the next experiment)
State one of:
- **Success.** Meets step-3 criterion. Propose promoting this config
  (new golden anchor, baseline bump, memory update).
- **Partial.** Beats baseline but misses delta. Propose one refinement
  with a fresh Hypothesis.
- **Failure.** Below baseline. Close the corresponding hypothesis. Do
  NOT propose a variant of the same idea — pick a different Open
  Question from §4.

## Anti-Patterns

- Launching training before step 3 is written.
- Declaring "success" from training-peak IoU without multi-seed bench.
- Running a credit-float sweep after step 2 flags it as a dead end.
- Skipping the Decision Doc row update ("I'll write it up later").
- Tweaking the success criterion after seeing the result.
- Treating training-pipeline engineering work as an experiment row —
  only runs with a live target + fitness delta belong in §2.

## Related
- [`experiment-config`](../experiment-config/SKILL.md) — config schema
  + Pre-Experiment Gate (a compressed version of steps 1–5 above)
- [`benchmark`](../../commands/benchmark.md) — the multi-seed bench
  command used in step 6
- [`/strategy`](../../commands/strategy.md) — one-shot summary of
  the current locked baseline, dead ends, and next decision point
