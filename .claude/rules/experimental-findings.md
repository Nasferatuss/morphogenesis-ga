# Experimental findings — locked

> Read these before proposing new experiments, HPO sweeps, or ablations.
> This file replaces the "Experimental findings locked in 2026-04-10"
> section that used to live in CLAUDE.md.
>
> **Live decision state** is in [`docs/DECISION_2026-04-19.md`](../../docs/DECISION_2026-04-19.md)
> (locked baseline, experiments table, dead ends, open questions, next
> experiment). When this file and the Decision Doc disagree, the
> Decision Doc wins.
>
> Supporting detail: [`docs/HANDOFF_2026-04-10.md`](../../docs/HANDOFF_2026-04-10.md),
> [`docs/leaderboard.md`](../../docs/leaderboard.md),
> [`docs/visual_diagnosis_2026-04-10.md`](../../docs/visual_diagnosis_2026-04-10.md),
> [`docs/plateau_analysis_2026-04-10.md`](../../docs/plateau_analysis_2026-04-10.md).

## Locked baseline (updated 2026-04-20, experiment #16)

- **Reproducible baseline:** `runs/exp_directional_presence_v1_20260419_210734/best_t.pt`,
  MS-10 IoU = **0.440**. Config:
  [`configs/exp_directional_presence_v1.yaml`](../../configs/exp_directional_presence_v1.yaml).
- Previous baseline (2026-04-11 → 2026-04-20): MS-10 = 0.329
  (`runs/champion_hpo_v1/best.pt`). Superseded by +0.111 (+33.7%).
- Golden anchor: still `TestGoldenBaselineV2` (0.329) until v3 anchors
  are created — pending follow-up work.
- Any new champion must beat 0.440 by ≥0.02 on multi-seed bench.
- Visual confirmation (`dump_grid.py` 5 seeds): both horizontal bar
  AND vertical trunk are built. Old structural dilemma resolved.

## Top architectural finding

`use_position: true` on `LittleLM` (from commit `28a56a1`) gives **+124%
average IoU** across all 73 historical checkpoints (positional models
avg 0.290, legacy avg 0.130). Always set it when training new models.

## Critical gotchas (learned the hard way)

1. **Training peak IoU is misleading — always multi-seed benchmark.** Use
   [`scripts/multiseed_bench.py <checkpoint>`](../../scripts/multiseed_bench.py).
   `sweep_a2_*` had training peak 0.375 but MS mean 0.251 (overfit to a
   lucky training seed). Never declare a champion from training peak alone.

2. **Do NOT "fix" varying `world_seed` in evaluator to be constant per
   generation.** Looks like a bug, is a free regulariser — locking the
   seed regressed IoU from 0.41 to 0.26.
   See [`docs/experiment_report_2026-04-10.md`](../../docs/experiment_report_2026-04-10.md) Exp 7.

3. **`t_trunk_weight` is a dead hyperparameter.** 6-config sweep proved
   `t15 ≡ t30` bit-identically. Even after directional divide actions
   merged (`bc0490a`, 2026-04-12), no recorded experiment has isolated
   a measurable effect from this weight. **Exclude from HPO search
   spaces.**

4. **Adaptive mutation alone is useless without positional encoding.**
   All 4 `ablation_adaptive_*` runs scored MS mean 0.134 — identical to
   legacy baseline. Positional encoding was the fix. Test adaptive
   mutation ONLY in combination with `use_position: true`.

5. **Champion regression is real.** Every long run (40+ gens) ends with
   final IoU lower than peak. `exp_t_long_champion_20260410_172908`
   (120 gens) peaked 0.389 @ gen 75, ended 0.342, MS mean 0.325 —
   **worse** than 40-gen variants. Don't train longer hoping to break
   the ceiling. Longer ≠ better.

6. **Float `stem_trunk_credit` is a dead hyperparameter.** Commit
   `b0dac9d` (2026-04-17) replaced float `stem_trunk_credit` with bool
   `stem_trunk_presence` after five experiments (`v6`–`v10`,
   credit=0.2/0.3/0.5/0.7) all performed far worse than the accidental
   binary-collapse bug behaviour (MS-10 ≈ 0.17–0.25 vs 0.45). GA needs a
   strong binary "any trunk cell = good" signal, not subtle gradients.
   **Exclude from HPO search spaces; always set `stem_trunk_presence:
   true` for T-shape training.**

7. **HPO breakthrough of 0.4254 was unreproducible (RNG bug).** The
   HPO v2 sweep winner (trial 17, 2026-04-11) reported MS-10 0.4254.
   After commit `1a10ac9` fixed the `run_trial` RNG seeding, the
   reproducible MS-10 collapsed to **0.329**. Postmortem:
   `docs/research/hpo_v2_report.md` § "2026-04-11 RNG postmortem". Do
   not cite 0.4254 as a performance number; use 0.329.

8. **`t_morph_bonus` was dead code before 2026-04-15.** Computed but
   never added to fitness in every experiment before `8d31bc2`. Fixing
   this is the **primary driver** of the apparent 0.339 → 0.454
   improvement — not any of the `stem_trunk_credit` values tested.

9. **Directional divide actions merged but not yet validated against
   the locked baseline.** `ACTION_DIVIDE_N/S/E/W` landed in `bc0490a`
   (2026-04-12). No multi-seed benchmark has isolated their effect
   versus pre-directional MS-10 = 0.329. This is Decision Doc Open
   Question #1.

10. **Bar-vs-trunk structural dilemma is RESOLVED (2026-04-20).**
    Experiment #16 (`exp_directional_presence_v1`) built both horizontal
    bar (row 2 type-A cells) AND vertical trunk (col 7 type-B cells)
    in all 5 dumped seeds. MS-10 = 0.440 (+33.7% over 0.329). Directional
    divide actions (`bc0490a`) + `stem_trunk_presence: true` unlock the
    trunk.

11. **New bottleneck: stem clutter.** `dump_grid.py` on exp #16 shows
    alive ≈ 95-100 cells vs target ≈ 25. The T is right; surrounding
    undifferentiated stems dominate FP penalty, capping MS-10 at ~0.45.
    Remaining fitness-side axes to attack (OQ#4 in Decision Doc):
    `stem_cleanup_multiplier`, `stem_corridor_end_scale`,
    `stem_phase_cleanup_start_frac`. Test one at a time.

12. **Hard simulation-side caps paradoxically grow what they cap
    (2026-04-21 exp #17).** Raising `die_cap_schedule[2].cap_frac:
    0.35 → 0.60` pushed MS-10 from 0.440 to 0.303 (−31%) and increased
    alive cells from ~95 to ~145-158. GA responds to "more allowed
    deaths" by spawning more stems to maintain population — the
    regulator loses the arms race because it provides no fitness
    gradient. **Prefer fitness-side levers** (weights in
    `fitness.*` block) **over simulate-side caps** (`simulate.*`
    block) when attacking structural problems. Hard caps are
    constraints; GA needs costs.

13. **Stem-specific penalties cause type-A differentiation escape-hatch
    (2026-04-21 exp #18).** Raising `stem_cleanup_multiplier: 1.5 →
    2.5` dropped alive from ~95 to ~57 (cleanup worked!) but MS-10
    fell from 0.440 to 0.378 (−14%). GA substituted stems for type-A
    cells: `alpha_fp (2.88) < stem_cost (3.20)` made `a` (out-of-target
    type-A) cheaper than `S`. Visible signature: row-3 phantom second
    bar of 12 `a` cells in every seed. **Do NOT raise stem-specific
    penalties in isolation** — couple with higher generic FP pressure
    (`late_t.fp_multiplier`, `late_t.clean_fp_weight`), or attack
    peripheral FP generically from the start.

14. **The "peripheral clutter" bottleneck is generic, not stem-specific
    (OQ#4 rescoped 2026-04-21).** What matters is "any cell outside
    target", not "stems". Attacks must be type-agnostic (e.g.
    `late_t.fp_multiplier` — hits stem, a, x equally) or combine
    stem-side and A-side pressure together. Pure stem-specific
    attacks hit Dead End #8's differentiation escape-hatch.

15. **Single-axis fitness attacks close one escape-hatch and open
    another (2026-04-22 exp #19).** `late_t.fp_multiplier: 1.2 →
    1.8` (type-agnostic) partially closed #18's type-A escape
    (phantom `a` bar 12→5 cells/seed) but stems repopulated because
    `stem_cleanup_multiplier` was reverted to 1.5. Each fitness
    weight only pushes one cell type; GA rebalances by spawning the
    uncostly type. **Pattern across #17-19:** pressure X reduces
    clutter-type-A, GA substitutes clutter-type-B.

16. **Dominant-parameter basins mask weaker levers in combined
    experiments (2026-04-23 exp #20).** When `stem_cleanup_multiplier=2.5`
    is active, changing `late_t.fp_multiplier` in range 1.2-1.8
    produces BIT-IDENTICAL GA trajectories across all 10 seeds.
    Code verified correct; the dominant fitness term (stem penalty
    ~0.3 per stem with multi=2.5) so strongly shapes individual
    ranking that secondary fp_multi perturbations (3.46 → 5.18 effective
    alpha_fp late-stage) don't flip the elite ordering → same
    selection → bit-identical children → deterministic identical
    result. **Methodological consequence:** do NOT combine a strong
    lever with a weaker one to "stack pressure". When primary lever
    dominates, secondary is silently masked. To use a weaker lever:
    test it from BASELINE (no stacking), or raise it by orders of
    magnitude (risky).

17. **Training-time IoU capability can exceed MS-10 robustness by a
    large margin (2026-04-23 exp #21).** `late_t.clean_fp_weight:
    0.08 → 0.20` drove training-best IoU to 0.56 at fp=0.0000
    repeatedly in phase_T (gens 161, 182, 227, 243), but MS-10 only
    reached 0.4296. This is 0.13 IoU worth of training-vs-eval gap.
    Confirms critical gotcha #1 with force: training peak IoU is
    not merely a "lucky seed" artefact — it can be a real capability
    of the model that simply doesn't transfer across world seeds.
    When MS-10 is stuck but training is climbing, the problem is
    **seed-generalization**, not fitness capability. No further
    fitness tuning will help at this point; must address model /
    curriculum / action space.

18. **OQ#4 (peripheral clutter cleanup) is STRUCTURALLY CLOSED
    (2026-04-23).** 5 failed experiments (#17-21) across 4 fitness
    axes (die_cap, stem_cleanup_multiplier, late_t.fp_multiplier,
    late_t.clean_fp_weight) + combinations exhausted the reasonable
    fitness-tuning space. The 0.440 baseline from exp #16
    (directional + presence) is the effective ceiling for fitness
    work alone. Breaking 0.50 requires a model-side or
    curriculum-side change (OQ#5 in Decision Doc).

19. **`phase_T_refine` regresses `best_t_refine.pt` vs `best_t.pt`
    on training fitness, but contributes to MS-10 robustness
    (REVISED 2026-04-24 after exp #22).** Initial reading (#16-21):
    `best_t.pt` outperforms `best_t_refine.pt` by 0.02-0.10 MS-10
    → conclusion "remove refine". Falsified by #22: removing refine +
    extending phase_T 150→200 gens dropped `best_t.pt` MS-10 from
    0.440 to 0.3757 (−14.6%). Mechanism: refine's "regression" in
    training fitness is actually population-level diversification
    pressure that helps generalization across world seeds. **Keep
    `phase_T_refine` in production curriculum.** `best_t.pt` is still
    the promoted artefact, but the refine phase that follows it is
    NOT optional — it shapes the population dynamics that produce
    the robust `best_t.pt` itself.

## Champion config recipe (updated 2026-04-20, exp #16 at MS-10 0.440)

- `use_position: true`
- `alpha_fp: 2.88`, `beta_fn: 1.77`, `gamma_area: 0.29` (HPO-derived
  v10 weights carried through)
- `steps: 250`
- Directional actions active (`src/world.py` post `bc0490a`)
- `stem_trunk_presence: true` (bool, never float credit)
- 3-phase `die_cap_schedule`: grow (0.20) → shape (0.40) → gentle
  clean (0.35) — *this may need to rise to attack the new stem-clutter
  bottleneck; OQ#4 candidate*
- `late_cleanup_start_frac: 0.6522`
- 4-phase curriculum: phase_cross (50) → phase_transition (16) →
  phase_T (150) → phase_T_refine (30)
- `plateau_patience_t: 90`, `plateau_stop_entire_run: true`

**Note on `best_t.pt` vs `best_t_refine.pt`:** in exp #16,
`best_t.pt` (peak of phase_T) outperformed `best_t_refine.pt` (end of
phase_T_refine) by +0.04 MS-10. `phase_T_refine` regressed the
champion — consistent with gotcha #5. Always bench both checkpoints
and promote whichever has higher MS-10.

## Convergence speed by `alpha_fp`

From 6-config sweep plateau analysis:

- `alpha_fp=2.0` → peak gen 16 (don't run > 25 gens, waste)
- `alpha_fp=2.5` → peak gen 28 (current sweet spot)
- `alpha_fp=3.0` → peak gen 38 (but −19% regression after peak)

## Reusable analysis scripts

- [`scripts/multiseed_bench.py`](../../scripts/multiseed_bench.py) —
  standalone multi-seed IoU benchmark
- [`scripts/build_leaderboard.py`](../../scripts/build_leaderboard.py) —
  ranks every `runs/*/best.pt` by MS mean
- [`scripts/dump_grid.py`](../../scripts/dump_grid.py) — ASCII
  visualisation of a model's final grid
