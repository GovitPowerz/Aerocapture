# Two-phase regime curriculum: centered high-regime anchors fine-tuned under the deployed medium objective behind a regime guard

> **Date:** 2026-10-30. **Status:** closed at Step 0 (user, 2026-10-30): the anchors hold the medium tail, nothing built (Section 10). **Issue:** #229.
> **Topic:** the TODO's "regime-matched objective schedule" (paper Section 7.3 follow-up), settled
> by a grilling session: which of the three proposed mechanisms to build, its trigger, its
> feasibility gate and its compute budget. Promotion rule recorded as ADR-0009; vocabulary in
> `CONTEXT.md` (Regime, Stacked / Centered Objective, Anchor, Regime Guard, Two-Phase Curriculum).

## 1. Motivation

Section 7.3 (`2026-06-29-objective-centering-regime-matched-design.md`, #156, #177) showed that the
deployed *stacked* objective (cubed transform, max bucket, two scenarios per individual) stalls on
the high regime at ~96% capture, and that the *centered* objective (linear, middle bucket, sixteen
scenarios) recovers the gradient: three centered Mamba_962 seeds trained from scratch under
`per_draw` (`configs/training/ou_marginal/centered/mamba_centered_s{1,2,3}.toml`, GA 256 x 16,
4000 gens, ~4 h each) beat both joint-FTC retunes on the 9M stress pool at n = 10,000. What nobody
measured is the other side: `centered_depth_v4.json` scores those seeds on the stress pool only.
Whether the centered seeds "give up the nominal sizing tail", the premise of the TODO item, is an
open question, not a finding.

## 2. Goal and success criterion

One Mamba_962 policy at **two-pool parity**:

- **Medium (deployment) pool**: the frozen 10 x 100,000 confirmatory pool under `per_draw`
  (`experiments/ou_marginal/confirmatory_marginal.py`, whose file holds the champion rows),
  CVaR99.9 paired delta (curriculum minus champion) against the three `hl_mamba_p962` champion
  seeds with a 95% CI that does not lie above zero (no significant loss), and capture parity.
- **High (stress) pool**: the 9M stress pool at n = 10,000 (`centered_depth_eval.py`), capture
  within 0.5 pt of the anchors and a CVaR95 paired delta (curriculum minus anchor) whose 95% CI
  does not lie above zero.

Constraint exceedances are reported beside every number. The strict feasibility gate is not a
criterion on the high side: no high-regime cell of either family passes it (TODO, #177).

## 3. Step 0: measure the anchors on the medium pool (gates everything)

Score `mamba_centered_s{1,2,3}` with `experiments/ou_marginal/confirmatory_marginal.py --cells`
on the frozen medium pool (`confirmatory_eval.py` defaults to the legacy noise path and pairs only
its hardcoded cells), the four high levels overridden back to the deployment presets of
`configs/training/common.toml` via `--extra-override`: `monte_carlo.atmosphere.level`,
`monte_carlo.navigation.level` and `monte_carlo.nav_filter.level` to `"medium"`,
`monte_carlo.density_perturbation.level` to `"low"`. The rows land in
`experiments/ou_marginal/confirmatory_marginal.json` next to the `hl_mamba_p962` rows they pair
with; the paper's extract keeps only its manifests' cells, so `articles/paper/data/` and its
`SHA256SUMS` stay untouched until a number is quoted. ~15 min per seed.

If the anchors already hold the medium tail, the campaign stops here: Section 7.3 gets the
two-pool table and the TODO item closes without a schedule.

## 4. Mechanism: a high-to-medium two-phase curriculum

The three mechanisms the TODO listed were rejected on the facts:

| mechanism | why not |
|---|---|
| annealed cost-transform exponent, scheduled on population capture | the trigger "capture saturates" never fires on the high regime (~5% of draws are catastrophic under any law, Section 7.3); a per-generation exponent makes `Champion.train_cost`, the gate's `val_rms` and the checkpointed `best_val_cost` non-stationary, the same incomparability the resume lesson already cost us |
| stratified curation batch (one `middle` seed + `max` seeds, n = 3-4) under `cubed` | a middle seed at ~150 m/s is 3.4e6 cubed next to a max seed at ~3000 virtual DV, 2.7e10: the anchor seed is invisible; only coherent with a linear transform, which makes it "centered with a harder seed mix" |
| medium-to-high widening curriculum with the stacked objective | its end state is the stacked-on-high objective; exp-13 (the deployed Mamba retrained on high, i.e. the same curriculum at its coarsest step) already stalled there at 96% |

The chosen mechanism runs the curriculum the other way, so that the end state is an objective
known to produce the nominal tail (the champion's) started from the only basin known to hold the
high-regime robustness (the anchors'):

- **Phase 1** is each anchor itself, finished: the `checkpoint_g04000` of
  `training_output/ou_marginal/centered/mamba_centered_sN/`.
- **Phase 2** fine-tunes from that checkpoint under the **medium regime** with the **deployed
  transform and bucket**, at the #192 fine-tune recipe (`jobs_heat_load.txt`): GA 60 x 10 (ten
  scenarios per individual, not the stacked objective's two), 2000 generations,
  `cost_transform = "cubed"`, `curation_bucket_selection = "max"`, adaptive seeds,
  `max_violation_rate = 0.0` on the medium validation pool. ~25 min per job before the guard.
- **Trigger**: fixed, at g04000. Nothing is scheduled within a phase. `campaign.sh` hardcodes the
  source checkpoint generation (`SEED_GEN=20000`) and splices it unpadded into
  `checkpoint_g${SEED_GEN}`, which matches the trainer's five-digit names only at 20000: it becomes
  `SEED_GEN=${SEED_GEN:-20000}` with every checkpoint name formatted through
  `printf 'checkpoint_g%05d'`, and the curriculum jobs file is run with `SEED_GEN=4000`.

What existing machinery already does at the switch: `_restore_seed_curator` drops the 16-wide
checkpointed seed list (width != `training_n_sims`) and re-curates; `resize_population` shrinks
256 -> 60 by keeping the lowest-cost rows; the prologue re-validates the checkpointed champion
under the new transform (the `cost_transform changed` notice); `campaign.sh` strips the copied
`rng_state` for seeds other than 1 so `--seed N` is honoured.

## 5. Regime guard (ADR-0009)

Fine-tuning on medium with the stacked transform and bucket can forget the high regime one
promotion at a time. The guard makes that visible and refuses it:

- **Config**: `[optimizer.regime_guard]`, built in `OptimizerConfig.from_dict` like
  `[optimizer.ga]`, whose dataclass kwargs reject an unknown key (Rust passes `[optimizer]`
  through as a Python-owned table, ADR-0008): `n_sims = 4000`, `tolerance = 0.05`,
  `cost_transform = "linear"`, and an `overrides` table of dot paths (the four high levels).
  Absent section: nothing changes. The guard needs the single-algorithm loop and a validation
  pool: with `algorithm = "islands"` or `validation_n_sims = 0` the section is a config error.
- **Seeds**: a new reserved stream `REGIME_GUARD_SEED_OFFSET = 12_000_000` in `training/seeds.py`
  (11M is the world-model stream), drawn through `_draw_disjoint_seeds` with the 9M stress pool's
  first 10,000 seeds (the final high score) excluded: independent streams are disjoint only
  probabilistically, ~2% odds of a shared scenario at these sizes. 4000 seeds resolve ~0.6
  capture points; RMS of the linear cost includes crashes through the virtual DV, so a capture
  loss shows up in the one number.
- **Bar**: the starting champion's (the anchor's) guard-pool RMS, computed once at prologue and
  checkpointed; a resume reads it and never re-flies it. The anchors' high validation RMS under
  linear sits at ~4076-4079 on their own 1000-seed pool, so the guard bar will be of that order.
- **Rule**, lexicographic after ADR-0005: feasible on the medium pool, then guard RMS <= bar x
  1.05, then strictly lower medium (cubed) val RMS promotes. Applied at the validation gate, the
  pre-loop champion validation and final selection. The guard is flown only for a candidate the
  first and third conditions would promote: the decision is the same conjunction, and final
  selection flies it in medium-RMS order, stopping at the first candidate that passes.
- **Refusal**: the verdict rides on the `VALIDATED` gate result as `GateResult` fields
  (`guard_rms`, `guard_bar`, `guard_pass`) next to `feasible`, not as a new `GateStatus`:
  `Champion.consider` and the trainer's validation log act on `VALIDATED` only, so a new status
  would skip the last-validated update and the record. Kept apart from `feasible`, ADR-0005's
  feasibility accounting is untouched. The refused candidate becomes last-validated (not
  re-flown), the champion stays. Every validation record in the per-gen JSONL carries the three
  columns (null where the guard was not flown), so a walk-off is a column, not a guess.
- **Cost**: one more MC pass of 4000 sims per would-promote candidate. The gate validates on most
  generations (1583 of 2000 in `hs_mamba_p962_q25`, the #192 run this recipe copies) but promotes
  a handful of times, so the phase-2 job stays at ~25-30 min; a guard flown on every validation
  would add ~6.3M sims, about an hour per job.

## 6. Cells, jobs and scoring

- `configs/training/ou_marginal/curriculum/mamba_curriculum_s{1,2,3}.toml`, in their own
  directory (`tests/test_centering_campaign_configs.py` pins the `centered/` file set to the #177
  cells), base = `../centered/mamba_centered_sN.toml`: atmosphere, navigation and nav_filter back
  to `"medium"` and density_perturbation back to `"low"` (the deployment presets),
  `n_pop = 60`, `training_n_sims = 10`, `curation_bucket_selection = "max"`,
  `cost_transform = "cubed"`, the `[optimizer.regime_guard]` section, and a run-local
  `[data] neural_network` (`training_output/ou_marginal/curriculum/mamba_curriculum_sN/best_model.json`, #170).
- `experiments/ou_marginal/jobs_curriculum.txt`:
  `curriculum/mamba_curriculum_sN|6000|N|training_output/ou_marginal/centered/mamba_centered_sN`,
  run as `SEED_GEN=4000 caffeinate -i experiments/ou_marginal/campaign.sh <jobs>` from a Terminal
  tab.
- Three runs, one per anchor, **no new baselines**: the anchors are the high-pool control
  (`centered_depth_v4.json`), the champion seeds the medium-pool control (their `per_draw` rows in
  `experiments/ou_marginal/confirmatory_marginal.json`).
- **Scoring**: medium side as in Step 0 (same script, same file, the curriculum seeds appended
  through `--cells` with no `--extra-override`: their TOML is already the deployment regime);
  high side via `centered_depth_eval.py --v5`, a cell list of the three curriculum seeds paired
  on scenario against the anchors at n = 10,000, under `per_draw`.
- Gates to keep green: `tests/test_pyo3.py`, `tests/test_run_grid.py`,
  `tests/test_thread_invariance.py` (the guard adds an evaluation path, not a physics change);
  `tests/test_seed_offsets.py` with the new offset; a Python test that an unknown
  `[optimizer.regime_guard]` key raises (`every_committed_config_parses_and_validates` does not
  see inside `[optimizer]`); a unit test that a guarded gate refuses a candidate above the bar,
  keeps it as last-validated and logs the three columns.

## 7. Compute budget (approved)

Sequential, one Terminal tab, `caffeinate -i`, `python -u`:

| step | cost |
|---|---|
| Step 0: anchors on the medium pool | 3 x ~15 min |
| Phase 2: three fine-tunes with the guard | 3 x ~25-30 min |
| Final scoring: medium 10 x 100k, high n = 10,000 | 3 x ~15 min + ~10 min |

About 3-3.5 h wall. Step 0 gates the rest.

## 8. Risks

- **The guard refuses everything.** Then the stacked transform and bucket cannot improve the medium tail
  without spending high-regime robustness at this tolerance: a finding, reported with the refusal
  counts from the JSONL. The tolerance is the one knob; the fallback mechanism is the stratified
  batch with a linear transform (Section 4, row 2), not a looser guard.
- **The medium tail does not move at 2000 gens.** The #192 recipe was sized for a ceiling
  sensitivity, not a regime change; read the medium validation curve on the sims axis before
  extending (`train.py` resumes, `--n-gen` adds generations).
- **Run variance.** Three seeds, paired CIs, nothing quoted from one run (the reference-trajectory
  lesson: single-run comparisons are not conclusive).
- **Step 0 moots the campaign.** The best outcome: no training, one table.

## 9. Non-goals

- No within-run schedule of any knob; no annealing; no mixed-regime seed draws (the regime is
  config-wide in Rust and stays so).
- No new baseline cells, no re-tune of the joint-FTC laws.
- No change to the strict medium-side ceiling; no guard on runs that do not configure one.
- Paper placement is decided from the Step 0 result, not here.

## 10. Outcome: Step 0 closed the campaign (2026-10-30)

The three anchors scored on the medium 10 x 100,000 pool (`confirmatory_marginal.py --cells`, the
four levels overridden as in Section 3; manifest `experiments/ou_marginal/centered_cells_v4.txt`,
18m52s wall for the three, not 45 min), against the three `hl_mamba_p962` champion seeds:

| cell | capture % | heat-load exceedances / 10^6 | CVaR95 | CVaR99.9 +- se | max |
|---|---|---|---|---|---|
| mamba_centered_s1 | 100.0000 | 9 | 149.7 | 170.3 +- 0.2 | 189.4 |
| mamba_centered_s2 | 100.0000 | 18 | 146.1 | 159.1 +- 0.1 | 177.9 |
| mamba_centered_s3 | 100.0000 | 208 | 143.2 | 177.5 +- 0.6 | 308.8 |
| hl_mamba_p962 (deployed) | 100.0000 | 1148 | 120.1 | 173.9 +- 2.2 | 300.4 |
| hl_mamba_p962_s2 | 99.9996 | 437 | 131.7 | 159.3 +- 0.6 | 620.8 |
| hl_mamba_p962_s3 | 100.0000 | 567 | 122.9 | 142.3 +- 0.5 | 266.5 |

Per-replicate paired CVaR99.9 delta on the same pools, anchor s_i minus champion s_i (a label
pairing, not a matched one; the seed spread is the comparison): -3.5 [-8.3, +1.4], -0.2 [-1.5, +1.0],
+35.1 [+33.5, +36.7]. Over seeds: anchors 168.9 +- 9.3, champion 158.5 +- 15.8.

Reading against Section 2: seeds 1 and 2 meet the medium criterion (CI not above zero, capture
parity), seed 3 does not; all three pay 15-30 m/s at CVaR95 (the linear transform and middle
bucket trading the body for the tail) and exceed the heat-load ceiling 5-100x less often than the
champion seeds. On the stress pool (`centered_depth_v4.json`) the same seeds already beat the
medium joint-FTC at capture parity, so seed 2 as trained is a two-pool-parity policy and the
premise of the TODO item (that the centered seeds give up the nominal tail) does not hold. The
remaining levers of a second phase, seed 3's medium tail and the CVaR95 body, are not the sizing
statistic; Sections 4-6 (mechanism, guard, cells) were not implemented, ADR-0009 is recorded as
not implemented, and the paper's Section 7.3 holds the two-pool table.

## 11. Final step

After implementation, invoke the `smart-commit` skill, telling it to take the whole git branch
into account.
