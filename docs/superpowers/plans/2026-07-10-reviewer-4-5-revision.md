# Reviewer Reports 4+5 Revision Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, serial - per
> Gregory's subagent-budget rule: one task at a time, state persisted to disk, no parallel fan-outs).
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the two 2026-07-10 reviewer reports (Reviewer_report_1.md: major revision, 12 major
comments; Reviewer_report_2.md: minor revisions) with the full funded scope: frozen confirmatory
sizing pool, state-ablation controls (reset flag + 6 retrains), sigma_run extras, and all verified
text/figure/reproducibility fixes. Venue = arXiv (keep format, Appendix C in-paper, title stays).
Code release = public repo + artifacts.

**Architecture:** Three workstreams. (1) A new confirmatory evaluation pool (10 replicates x 100k,
seed-range-disjoint from all history) requotes every headline number and kills the selection-on-test
and CVaR-from-10-obs objections. (2) Causal controls for the internal-state claim: a Rust
reset-state-per-tick eval flag (no retraining) plus window-matched and no-predicted-dv retrains
(user-launched), with pre-registered interpretation rules. (3) Verified text corrections, figure
fixes, related-work expansion, and Appendix A reproduction upgrades.

**Tech Stack:** Rust simulator + PyO3 (`aerocapture_rs`), Python analysis (`uv run`), Typst paper.

## Global Constraints

- Branch: stay on `claude/competent-bardeen-23b377`. Never push. Never `gh`.
- Compile from repo ROOT: `typst compile articles/paper/paper.typ articles/paper/paper.pdf`.
  Page-proof changed pages: `typst compile --format png --pages N articles/paper/paper.typ /tmp/proof_{p}.png`
  then view. Refresh `~/Desktop/aerocapture_paper_2026-07-08.pdf` after paper changes.
- Committed-data rule: every number quoted in `paper.typ` must regenerate from a file under
  `articles/paper/data/` (never transcribed from a terminal).
- De-tooling rule: no CLI commands, file paths, or module names in paper prose (language/library
  names and a repo URL are OK).
- Bundle pinning: every eval pins the committed-bundle model via `label:toml:bundle_key`
  (training_output can drift; dense_p515 did, 140.3 vs 128.1).
- PyO3 rebuild always from repo root: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`.
- Rust gate per Rust task: `cargo test` + `cargo fmt --check` + `cargo clippy` (or `./check_all.sh`)
  and the 6 guidance goldens bit-identical (new flags default off).
- Python gate per Python task: targeted `pytest tests/<file> -v` + `./lint_code.sh` before commit.
- Training runs are launched by Gregory (multi-day); the plan prepares configs/runners and analyzes
  results. Never run two cells of the same config TOML concurrently.
- After each task: update `articles/paper/revision_state.json` statuses, commit.
- Plain hyphens and straight quotes in all .md/.json outputs.

**Dependency order:** Tasks 1-2 first (all requotes depend on them). Tasks 5-6 (runners) handed to
Gregory EARLY so training runs in the background. Tasks 7-18 proceed during training. Task 19-20
after training completes. Tasks 21-22 close.

---

### Task 0: Revision state tracker

**Files:**
- Create: `articles/paper/revision_state.json`

**Interfaces:**
- Produces: the per-finding status file every later task updates (`status`: pending | applied |
  rejected | deferred; `note`: one line).

- [ ] **Step 1: Write the tracker** with one entry per finding. Content (complete):

```json
{
  "source": "Reviewer_report_1.md + Reviewer_report_2.md, analyzed 2026-07-10",
  "decisions": {"venue": "arxiv", "release": "public repo + artifacts",
                "experiments": ["confirmatory_pool", "state_reset", "retrain_controls", "sigma_extras"]},
  "findings": [
    {"id": "R1-1", "title": "sizing pool not selection-disjoint -> frozen confirmatory pool", "task": 2, "status": "pending"},
    {"id": "R1-2", "title": "CVaR99.9 from ~10 obs; estimator/CI/replicates", "task": 2, "status": "pending"},
    {"id": "R1-3", "title": "124.5 is 3-seed mean, not deployed artifact; estimand separation", "task": 20, "status": "pending"},
    {"id": "R1-3b", "title": "fig-1 'deployed Mamba ensemble' wording", "task": 15, "status": "pending"},
    {"id": "R1-4", "title": "conditional CVaR(dv|capture) labeling + joint failure/cost treatment", "task": 14, "status": "pending"},
    {"id": "R1-5", "title": "state-ablation controls missing", "task": 19, "status": "pending"},
    {"id": "R1-6", "title": "classical objective-equivalence documentation + no-dv ablation", "task": 19, "status": "pending"},
    {"id": "R1-6b", "title": "related work + novelty claim narrowing", "task": 17, "status": "pending"},
    {"id": "R1-7", "title": "reproducibility: equations, params, release, verification tests", "task": 18, "status": "pending"},
    {"id": "R1-8.1", "title": "monotone-transform claim wrong -> L6 objective equation", "task": 15, "status": "pending"},
    {"id": "R1-8.2", "title": "RL discussion wrong -> empirical + numbers", "task": 15, "status": "pending"},
    {"id": "R1-8.3", "title": "CMA-ES mechanism unestablished -> empirical + hypothesis", "task": 15, "status": "pending"},
    {"id": "R1-9", "title": "LSTM infeasible seed in 3-seed mean; feasibility-first", "task": 9, "status": "pending"},
    {"id": "R1-9b", "title": "113 m/s floor vs observed 104-106 -> recompute", "task": 15, "status": "pending"},
    {"id": "R1-10", "title": "dispersion model justification + high-regime table + OU/LHS questions", "task": 16, "status": "pending"},
    {"id": "R1-11", "title": "fig-1 rename occupancy envelope + constant-bank boundaries", "task": 13, "status": "pending"},
    {"id": "R1-12", "title": "compute claims: hardware/compiler/per-update detail; drop 'dominated outright'", "task": 10, "status": "pending"},
    {"id": "R1-S1", "title": "optimizer conclusions need repetitions (sigma_run extras)", "task": 19, "status": "pending"},
    {"id": "R1-S2", "title": "paired bootstrap CIs for p95/CVaR deltas", "task": 7, "status": "pending"},
    {"id": "R1-S3", "title": "scenario vs training-run uncertainty separated", "task": 20, "status": "pending"},
    {"id": "R1-S4", "title": "Wilcoxon 3e-165 -> p<1e-15 (saturated)", "task": 16, "status": "pending"},
    {"id": "R1-S5", "title": "confirmatory vs exploratory analyses distinguished", "task": 16, "status": "pending"},
    {"id": "R1-S6", "title": "fig_loss_vs_tail connector lines", "task": 12, "status": "pending"},
    {"id": "R1-P1", "title": "promotional language pass", "task": 15, "status": "pending"},
    {"id": "R1-P2", "title": "Monte Carlo attributive; 500x11 definition; heat-load unit; CVaR estimator def", "task": 16, "status": "pending"},
    {"id": "R1-P3", "title": "validation pool -> selection pool rename + query count", "task": 8, "status": "pending"},
    {"id": "R1-P4", "title": "sec-5 adaptive-integration contradiction", "task": 15, "status": "pending"},
    {"id": "R1-P5", "title": "'residual cost is irreducible scenario noise' qualified", "task": 16, "status": "pending"},
    {"id": "R1-P6", "title": "sec-7.3 'not intrinsic' softening + sigma_run upgrade", "task": 19, "status": "pending"},
    {"id": "R2-A", "title": "RAD750 / flight-compute scaling discussion", "task": 18, "status": "pending"},
    {"id": "R2-B", "title": "PPO underperformance numbers", "task": 15, "status": "pending"},
    {"id": "R2-C", "title": "CMA-ES dynamics clarification", "task": 15, "status": "pending"},
    {"id": "R2-D", "title": "V&V / simplex-monitor discussion in limitations", "task": 18, "status": "pending"},
    {"id": "R2-N1", "title": "abstract '27.6 7.6' typo", "task": null, "status": "rejected", "note": "PDF-extraction ghost; rendered PDF clean"},
    {"id": "R2-N2", "title": "eq-5 mu_ptex + parens", "task": null, "status": "rejected", "note": "ghost; source + PDF read wrap_pi(mu_prev + ...)"},
    {"id": "R2-N3", "title": "MJ/m3 in sec-2.1", "task": null, "status": "rejected", "note": "ghost; PDF says MJ/m2"},
    {"id": "R2-N4", "title": "27.6 vs 27.5 rounding consistency", "task": 16, "status": "pending"},
    {"id": "R2-N5", "title": "LSTM dagger in Table 3", "task": 9, "status": "pending"},
    {"id": "R1-P7", "title": "two-column figure sizing + color-blind-safe styling", "task": null, "status": "deferred", "note": "venue = arXiv single-column; revisit at journal submission"},
    {"id": "R1-P8", "title": "sec-8 reframed as closed-loop input sensitivity", "task": 16, "status": "pending"},
    {"id": "R1-6c", "title": "classical tuning budgets + objective equivalence stated in-paper", "task": 18, "status": "pending"}
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add articles/paper/revision_state.json
git commit -m "revision: R4/R5 finding tracker (3 R2 typos verified as PDF-extraction ghosts)"
```

---

### Task 1: Confirmatory seed pools (range-disjoint from all history)

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py` (after `make_reserved_seeds`, ~line 73)
- Create: `tests/test_confirmatory_seeds.py`

**Interfaces:**
- Produces: `make_confirmatory_pools(base_mc_seed: int, n_replicates: int = 10, n: int = 100_000) -> list[list[int]]`
  and `CONFIRM_EVAL_SEED_OFFSET = 20_000_000`. Task 2 consumes both.

**Design note (why range-disjoint):** every historical seed - reserved pools, training draws,
curation probes - comes from `default_rng(...).integers(0, 2**31, ...)`. Drawing the confirmatory
seeds from `[2**31, 2**32)` makes the pool disjoint from every scenario ever touched BY
CONSTRUCTION (no birthday-bound caveats), which is exactly the "genuinely untouched" property R1
demands. TOML integers are i64 so seeds < 2**63 parse fine; Step 4 smoke-tests the Rust side.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_confirmatory_seeds.py"""
import numpy as np

from aerocapture.training.evaluate import make_confirmatory_pools, make_reserved_seeds


def test_pools_shape_unique_and_range():
    pools = make_confirmatory_pools(42, n_replicates=3, n=500)
    assert len(pools) == 3 and all(len(p) == 500 for p in pools)
    flat = np.concatenate(pools)
    assert len(np.unique(flat)) == len(flat)          # unique within AND across replicates
    assert flat.min() >= 2**31 and flat.max() < 2**32  # range-disjoint from all history


def test_pools_deterministic():
    assert make_confirmatory_pools(42, 2, 100) == make_confirmatory_pools(42, 2, 100)


def test_disjoint_from_reserved_streams():
    flat = set(np.concatenate(make_confirmatory_pools(42, 2, 1000)).tolist())
    for offset in (1_000_000, 2_000_000, 8_000_000, 9_000_000):
        assert flat.isdisjoint(make_reserved_seeds(42, offset, 1000))
```

- [ ] **Step 2: Run to verify failure**: `uv run pytest tests/test_confirmatory_seeds.py -v`
  Expected: FAIL, `ImportError: cannot import name 'make_confirmatory_pools'`.

- [ ] **Step 3: Implement** in `evaluate.py` next to the existing offsets:

```python
CONFIRM_EVAL_SEED_OFFSET = 20_000_000  # confirmatory sizing pool RNG stream (seeds themselves live in [2^31, 2^32))


def make_confirmatory_pools(base_mc_seed: int, n_replicates: int = 10, n: int = 100_000) -> list[list[int]]:
    """Frozen confirmatory sizing pools: n_replicates independent pools of n seeds.

    Seeds are drawn WITHOUT duplicates from [2**31, 2**32) -- structurally disjoint
    from every historical pool and training/curation draw (all generated in
    [0, 2**31)). Deterministic in base_mc_seed.
    """
    rng = np.random.default_rng(base_mc_seed + CONFIRM_EVAL_SEED_OFFSET)
    total = n_replicates * n
    seeds: set[int] = set()
    while len(seeds) < total:
        seeds.update(rng.integers(2**31, 2**32, size=total - len(seeds)).tolist())
    ordered = rng.permutation(np.array(sorted(seeds), dtype=np.int64))
    return [ordered[i * n : (i + 1) * n].tolist() for i in range(n_replicates)]
```

- [ ] **Step 4: Rust seed-range smoke test** (seeds >= 2^31 must simulate):

```bash
uv run python -c "
import aerocapture_rs
r = aerocapture_rs.run('configs/test/test_ref_orig.toml', overrides={'monte_carlo.seed': 2**31 + 5, 'simulation.n_sims': 1})
print('ifinal-ok', r.final_record[:2])"
```
Expected: prints without error. If the Rust config rejects large seeds, STOP and fix the seed type
in `config.rs` (u64) before proceeding - do not fall back to the collision-prone [0, 2^31) range.

- [ ] **Step 5: Run tests**: `uv run pytest tests/test_confirmatory_seeds.py -v` -> 3 PASS.
- [ ] **Step 6: Lint + commit**

```bash
./lint_code.sh && git add src/python/aerocapture/training/evaluate.py tests/test_confirmatory_seeds.py
git commit -m "feat(eval): range-disjoint confirmatory seed pools (10x100k, [2^31,2^32))"
```

---

### Task 2: Confirmatory evaluation script + full run

**Files:**
- Create: `articles/paper/scripts/confirmatory_eval.py`
- Create (output, committed): `articles/paper/data/confirmatory_eval.json`

**Interfaces:**
- Consumes: `make_confirmatory_pools` (Task 1); `_resolve_eval_toml`, `cvar`, `bootstrap_ci` and the
  cell/bundle-pinning pattern from `articles/paper/scripts/far_tail_eval.py` (read it first; reuse
  its `_eval_one` structure with the seed list swapped in).
- Produces: `confirmatory_eval.json` with, per cell: per-replicate stats and pooled stats. Tasks 3,
  9, 14, 19, 20 consume it.

**Protocol (goes verbatim into Appendix A later):** methodology, architectures, checkpoints, and
classical tuning are frozen at the current commit; record `git rev-parse HEAD` as `freeze_commit`
in the JSON. Each cell is evaluated exactly once on 10 replicate pools of 100k scenarios
(1e6 total; CVaR99.9 averages the worst 100 per replicate, 1000 pooled). Replicate-level statistics
give design-based dispersion (t-based SE over 10 replicates) with no bootstrap-on-LHS caveat.
Paired deltas: replicates share seeds across cells, so per-replicate stat differences difference
out scenario noise; report mean +/- t-CI over the 10 replicate deltas.

- [ ] **Step 1: Write the script.** Core (mirror far_tail_eval.py; key parts):

```python
"""Frozen confirmatory sizing evaluation -- R replicates x n seeds from [2^31, 2^32).

Selection-disjoint by construction: generated AFTER all methodology/architecture/
checkpoint choices were frozen; evaluated exactly once per cell.
Usage: uv run python articles/paper/scripts/confirmatory_eval.py \
    --cells <label:toml[:bundle_key]> ... [--replicates 10] [--n 100000]
"""
# ... imports as far_tail_eval.py, plus:
from aerocapture.training.evaluate import make_confirmatory_pools

def _eval_cell(label, toml, pools, bundle_key):
    # identical model/scaffolding resolution to far_tail_eval._eval_one, then:
    reps = []
    for r, seeds in enumerate(pools):
        overrides = [{**base, "monte_carlo.seed": s} for s in seeds]
        res = aerocapture_rs.run_batch(toml_path=str(eval_toml.resolve()), overrides_list=overrides, sim_timeout_secs=5.0)
        # capture mask, x = sorted captured dv, violations: same extraction as far_tail_eval
        reps.append({"replicate": r, "n": len(seeds), "n_captured": int(cap.sum()),
                     "capture_pct": r2(100 * cap.mean()),
                     "p95": r2(np.percentile(x, 95)), "cvar95": r2(cvar(x, 0.95)),
                     "p99": r2(np.percentile(x, 99)), "cvar99": r2(cvar(x, 0.99)),
                     "p999": r2(np.percentile(x, 99.9)), "p9987": r2(np.percentile(x, 99.87)),
                     "cvar999": r2(cvar(x, 0.999)), "max": r2(x.max()),
                     "viol_pct": r2(100 * (v_hf | v_g | v_hl).mean()),
                     "heat_load_viol_pct": r2(100 * v_hl.mean())})
    pooled_x = np.sort(np.concatenate(all_captured_dv))   # keep per-replicate arrays to build this
    t95 = 2.262  # t(0.975, df=9)
    def agg(key):
        v = np.array([rp[key] for rp in reps])
        return {"mean": r2(v.mean()), "se": r2(v.std(ddof=1) / np.sqrt(len(v))),
                "ci95": [r2(v.mean() - t95 * v.std(ddof=1) / np.sqrt(len(v))),
                         r2(v.mean() + t95 * v.std(ddof=1) / np.sqrt(len(v)))]}
    return {"label": label, "replicates": reps,
            "pooled": {"n": int(sum(rp["n"] for rp in reps)),
                       "capture_pct": ..., "cvar999": r2(cvar(pooled_x, 0.999)),
                       "p9987": r2(np.percentile(pooled_x, 99.87)), "max": r2(pooled_x.max()),
                       "n_tail_obs_cvar999": int(np.ceil(0.001 * len(pooled_x)))},
            "replicate_stats": {k: agg(k) for k in ("cvar95", "cvar99", "cvar999", "p999", "max")}}
```

Also emit, at top level: `freeze_commit`, `n_replicates`, `n_per_replicate`, `seed_range`
`"[2^31, 2^32)"`, and a `paired` section computed for the pairs
(mamba_s1 vs joint_ftc), (mamba_s1 vs fnpag), (mamba_s1 vs dense515_s1), (mamba_s1 vs lstm_s1),
(joint_ftc vs fnpag): per-replicate `delta_cvar95`/`delta_cvar999`/`delta_p999` arrays plus their
`agg()` summaries. Also store, per cell, the pooled survival curve support for Task 3: `dv_sorted_sample`
(every 100th value of `pooled_x`, ~10k points) so the figure regenerates from committed data.

- [ ] **Step 2: Smoke test** (2 replicates x 200 seeds, one fast cell):

```bash
uv run python articles/paper/scripts/confirmatory_eval.py --replicates 2 --n 200 \
  --cells mamba_p962_long:configs/training/sweep/mamba_p962.toml:headline/mamba_p962
```
Expected: prints per-replicate lines, writes JSON with `replicates` arrays of length 2. Then delete
the smoke JSON (`rm articles/paper/data/confirmatory_eval.json`) so the real run starts clean.

- [ ] **Step 3: Full run - fast cells** (everything except FNPAG; minutes each, ~1 h total).
  Exact cell list (labels resolve under `training_output[/paper]/`; bundle keys under
  `articles/paper/data/runs/` - verify each key exists with `eza articles/paper/data/runs/headline/`
  before launching, and use the tail_repeats bundle keys found there for s2/s3):

```bash
uv run python articles/paper/scripts/confirmatory_eval.py --replicates 10 --n 100000 --cells \
  mamba_p962_long:configs/training/sweep/mamba_p962.toml:headline/mamba_p962 \
  paper/tail_repeats/mamba962_s2:configs/training/sweep/mamba_p962.toml:<bundle_key> \
  paper/tail_repeats/mamba962_s3:configs/training/sweep/mamba_p962.toml:<bundle_key> \
  dense_p515_ga_paper_best:configs/training/msr_aller_nn_atan2_best_paper.toml:headline/dense_p515 \
  paper/tail_repeats/dense515_s2:configs/training/sweep/dense_p515.toml:<bundle_key> \
  paper/tail_repeats/dense515_s3:configs/training/sweep/dense_p515.toml:<bundle_key> \
  lstm_p1082_long:configs/training/sweep/lstm_p1082.toml:headline/lstm_p1082 \
  paper/tail_repeats/lstm1082_s2:configs/training/sweep/lstm_p1082.toml:<bundle_key> \
  paper/tail_repeats/lstm1082_s3:configs/training/sweep/lstm_p1082.toml:<bundle_key> \
  gru_p1014_long:configs/training/sweep/gru_p1014.toml:headline/gru_p1014 \
  dense_p972_ga_paper_best:configs/training/msr_aller_nn_atan2_best_paper_1000.toml:headline/dense_p972 \
  joint_reference/ftc:configs/training/msr_aller_ftc_joint_ref_train.toml \
  joint_reference/pred_guid:configs/training/msr_aller_pred_guid_joint_ref_train.toml \
  joint_reference/energy_controller:configs/training/msr_aller_energy_controller_joint_ref_train.toml \
  ftc:configs/training/msr_aller_ftc_train.toml \
  pred_guid:configs/training/msr_aller_pred_guid_train.toml \
  energy_controller:configs/training/msr_aller_energy_controller_train.toml \
  equilibrium_glide:configs/training/msr_aller_eqglide_train.toml \
  piecewise_constant:configs/training/msr_aller_piecewise_constant_train.toml
```
(Config paths for the joint cells: read `experiments/paper/07_joint_reference.sh` and use exactly
the TOMLs it trained; same for 01_classical_baselines.sh fixed-reference cells. Do not guess.)

- [ ] **Step 4: Full run - FNPAG** (86 ms/sim x 1e6 = ~24 core-h, rayon-parallel; run in background,
  expect a few hours wall):

```bash
uv run python articles/paper/scripts/confirmatory_eval.py --replicates 10 --n 100000 --cells \
  fnpag:configs/training/msr_aller_fnpag_train.toml
```

- [ ] **Step 5: Sanity check** the deployed Mamba pooled CVaR99.9 against the development pool
  value (122.0 at n=10k): pooled value should land within a few m/s. If it moves by > ~5 m/s,
  STOP and report to Gregory before requoting anything (that magnitude of shift is itself a
  finding about selection optimism and changes how Task 20 is written).

- [ ] **Step 6: Commit** script + JSON:

```bash
git add articles/paper/scripts/confirmatory_eval.py articles/paper/data/confirmatory_eval.json
git commit -m "feat(paper): frozen confirmatory sizing pool, 10x100k range-disjoint replicates"
```

---

### Task 3: Survival-curve figure + confirmatory-fed figure updates

**Files:**
- Create: `articles/paper/scripts/fig_survival.py`; output `articles/paper/figures/fig_survival.svg`
- Modify: `articles/paper/scripts/fig_arch_tail.py`, `articles/paper/scripts/fig_classical_vs_nn.py`
  (point CVaR99.9 values at `confirmatory_eval.json` pooled/replicate stats)
- Modify: `articles/paper/paper.typ` (add fig-survival to Section 7 near tbl-perf)

**Interfaces:**
- Consumes: `confirmatory_eval.json` `dv_sorted_sample` per cell (Task 2).

- [ ] **Step 1: fig_survival.py** - log-y empirical survival `1 - F(dv)` for
  mamba_s1, dense515_s1, lstm_s1, joint_ftc, fnpag from `dv_sorted_sample` (use the shared
  `figlib.py` style; one line per scheme, direct labels, mark p95/p99.9 depths with faint
  vertical guides). Follow the style of the existing `fig_*.py` scripts.
- [ ] **Step 2:** Update `fig_arch_tail.py` and `fig_classical_vs_nn.py` to read the confirmatory
  pooled values (keep the far-tail 10k dev-pool version reproducible behind a `--dev-pool` flag
  rather than deleting it).
- [ ] **Step 3:** Regenerate the three SVGs; add to `paper.typ` after `tbl-perf`:

```typst
#fig("fig_survival.svg", [Empirical survival curves of the correction $Delta v$ on the confirmatory
pool ($10 times 100\,000$ scenarios, log scale). The network's advantage over the classical schemes
grows with tail depth; per-replicate spread at $"CVaR"_(99.9)$ is reported in @tbl-perf.], <fig-survival>)
```
- [ ] **Step 4:** Compile, page-proof the new figure page, commit.

---

### Task 4: Rust `reset_state_every_tick` eval flag + reset-control evaluation

**Files:**
- Modify: `src/rust/src/config.rs` (`TomlNeuralNetworkParams`, ~line 1188)
- Modify: `src/rust/src/data/guidance_params.rs` (carry the flag next to `NeuralNetMode`)
- Modify: `src/rust/src/gnc/guidance/dispatch.rs` (NN arm, ~line 237-246)
- Test: inline `#[cfg(test)]` in `dispatch.rs` (follow the existing test-module pattern there)

**Interfaces:**
- Produces: TOML key `[guidance.neural_network] reset_state_every_tick = true|false` (default
  false), reachable through the PyO3 override dot-path
  `"guidance.neural_network.reset_state_every_tick"`. Task 19 consumes it.

- [ ] **Step 1: Failing Rust test.** In the dispatch test module: build a small v2 GRU model
  (reuse the inline-JSON helper pattern from `data/neural/tests.rs`), create a `GuidanceState` via
  `GuidanceState::new(..., Some(&model))`, run the NN guidance arm twice with two different nav
  inputs, with the flag ON; assert the second call's output equals the output of a FRESH
  `GuidanceState` given only the second input (state was reset), and that with the flag OFF the
  two differ (state carried). Run `cargo test reset_state -p aerocapture` -> FAIL (field missing).
- [ ] **Step 2: Implement.** Config field `reset_state_every_tick: Option<bool>` in
  `TomlNeuralNetworkParams`, defaulted false into the guidance params struct; in the dispatch NN
  arm, immediately before the existing `state.nn_state.as_mut()` borrow:

```rust
if data.guidance.nn_reset_state_every_tick {
    if let (Some(st), Some(model)) = (state.nn_state.as_mut(), data.neural_net.as_ref()) {
        *st = NnState::for_model(model); // memoryless eval: fresh state every guidance tick
    }
}
```
- [ ] **Step 3:** `cargo test` (all pass, goldens bit-identical since default off), `cargo fmt`,
  `cargo clippy`. Rebuild PyO3 from repo root (Global Constraints command).
- [ ] **Step 4: Override-path check:**

```bash
uv run python -c "
import aerocapture_rs
r = aerocapture_rs.run('configs/training/sweep/mamba_p962.toml',
    overrides={'simulation.n_sims': 1, 'monte_carlo.seed': 7,
               'data.neural_network': 'articles/paper/data/runs/headline/mamba_p962/best_model.json',
               'guidance.neural_network.reset_state_every_tick': True})
print('captured', r.captured)"
```
- [ ] **Step 5: Reset-control evaluation.** Add a `--extra-override key=val` passthrough to
  `confirmatory_eval.py` (applied into `base`), then run the deployed Mamba with the flag on, on
  BOTH pools:

```bash
uv run python articles/paper/scripts/far_tail_eval.py --n-sims 10000 --cells \
  state_reset/mamba_s1:configs/training/sweep/mamba_p962.toml:headline/mamba_p962 \
  --extra-override guidance.neural_network.reset_state_every_tick=true
uv run python articles/paper/scripts/confirmatory_eval.py --replicates 10 --n 100000 --cells \
  state_reset/mamba_s1:configs/training/sweep/mamba_p962.toml:headline/mamba_p962 \
  --extra-override guidance.neural_network.reset_state_every_tick=true
```
(far_tail_eval.py needs the same small `--extra-override` addition; make it in this task.)
- [ ] **Step 6: Commit** Rust + script changes + updated JSONs:
  `git commit -m "feat(sim): reset_state_every_tick eval flag + reset-control far-tail/confirmatory eval"`

---

### Task 5: State-control training configs + runner (handoff to Gregory)

**Files:**
- Create: `configs/training/paper/window_ctrl_p970.toml`
- Create: `configs/training/paper/mamba_p962_nodv.toml`
- Create: `experiments/paper/15_state_controls.sh`

**Interfaces:**
- Produces: 6 training cells under `training_output/paper/state_controls/{window_s1..s3, nodv_s1..s3}`.
  Task 19 evaluates them.

- [ ] **Step 1: Read `configs/training/sweep/mamba_p962.toml`** and mirror its base-inheritance and
  `[network]` block structure exactly (arrays REPLACE under deep-merge: the leaf must fully
  restate `[[network.architecture]]` and `input_mask`).
- [ ] **Step 2: window_ctrl_p970.toml** - same bases, mask, normalization, scaffolding as
  mamba_p962; architecture replaced by (970 params; the explicit-history control):

```toml
# Window-matched dense control (R1 major 5): explicit 5-tick observation history,
# no learned state, ~matched budget to mamba_p962 (970 vs 962 params).
[[network.architecture]]
type = "window"
input_size = 17
n_steps = 5
[[network.architecture]]
type = "dense"
input_size = 85
output_size = 11
activation = "swish"
[[network.architecture]]
type = "dense"
input_size = 11
output_size = 2
activation = "asinh"
```
Verify the parameter count with
`uv run python -c "from aerocapture.training.config import ...; print(...)"` (use the
`NetworkConfig` param-count path the sweep generator uses; expected 970).
- [ ] **Step 3: mamba_p962_nodv.toml** - identical to mamba_p962 except `input_mask` drops 32-34
  (14 inputs: `[0, 2, 3, 5, 6, 7, 11, 12, 18, 19, 27, 28, 29, 30]`) and the encoder shrinks to
  `input_size = 14` (914 params; comment the count delta in the file). This is the
  retrain-without-predicted-dv ablation (R1 majors 5+6).
- [ ] **Step 4: 15_state_controls.sh** - copy the `run()` skeleton from
  `experiments/paper/10c_tail_sigma_repeats.sh` verbatim (skip-if-`final_selection.json`,
  resumable, Ctrl-C trap, `--sim-timeout 5`), cells:

```bash
P="training_output/paper/state_controls"
run configs/training/paper/window_ctrl_p970.toml 1 window_s1
run configs/training/paper/window_ctrl_p970.toml 2 window_s2
run configs/training/paper/window_ctrl_p970.toml 3 window_s3
run configs/training/paper/mamba_p962_nodv.toml 1 nodv_s1
run configs/training/paper/mamba_p962_nodv.toml 2 nodv_s2
run configs/training/paper/mamba_p962_nodv.toml 3 nodv_s3
```
- [ ] **Step 5: Smoke-validate both configs** load and train 1 generation:
  `uv run python -m aerocapture.training.train configs/training/paper/window_ctrl_p970.toml --n-gen 1 --no-tui --skip-report --output-dir /tmp/smoke_window` (then delete /tmp/smoke_window; same for nodv).
- [ ] **Step 6: Commit + HAND OFF.** Tell Gregory: launch `./experiments/paper/15_state_controls.sh`
  (multi-day, resumable, safe to Ctrl-C). Tasks 7-18 proceed meanwhile.

---

### Task 6: sigma_run extras runner (handoff to Gregory)

**Files:**
- Create: `experiments/paper/16_sigma_extras.sh`

**Interfaces:**
- Produces: s2/s3 repeats for the Study C decisive optimizer cells and the centered-stress Mamba.
  Task 19 evaluates them.

- [ ] **Step 1: Read** `experiments/paper/04_seed_strategy.sh`, `02_optimizer_budget.sh`, and
  `14_objective_centering.sh` to copy the EXACT config paths for: GA fixed-seed, GA rotating-seed,
  CMA-ES fixed, CMA-ES rotating (Study C cells, n=10/2000/300 regime), GA adaptive (= 02's ga_150
  cell), and the Phase-2 centered-Mamba retrain cell of 14.
- [ ] **Step 2: Write the runner** (10c skeleton again), cells under
  `training_output/paper/sigma_extras/`: `{ga_fixed,ga_rotating,ga_adaptive,cmaes_fixed,cmaes_rotating}_s{2,3}`
  plus `mamba_centered_s{2,3}`. Each optimizer cell is ~6M sims (hours); the centered retrains are
  the big ones. Order the script cheapest-first so partial completion is still useful.
- [ ] **Step 3: Commit + HAND OFF** to Gregory alongside Task 5's runner (they can share the box
  sequentially; note in the script header not to run concurrently with 15_state_controls.sh if
  cores are scarce).

---

### Task 7: Paired tail-delta CIs in the aggregator + Table 4

**Files:**
- Modify: `articles/paper/scripts/aggregate_results.py` (the paired-table builder)
- Modify: `articles/paper/paper.typ` tbl-paired (~lines 816-845)

- [ ] **Step 1:** In the paired-table function, alongside the existing `delta_mean_ci`
  (10k-resample bootstrap on the paired mean), add `delta_p95_ci` and `delta_cvar95_ci`: resample
  scenario indices WITH replacement (shared index across both schemes, preserving pairing),
  recompute `p95(A*) - p95(B*)` and `cvar(A*, .95) - cvar(B*, .95)` per resample, take the 2.5/97.5
  percentiles. 10k resamples.
- [ ] **Step 2:** Rerun the aggregator; `git diff articles/paper/data/results.json` must show ONLY
  added `*_ci` keys (no changed numbers) - that is the regression check.
- [ ] **Step 3:** In tbl-paired, keep the existing columns and extend the caption:

```typst
The CI is a $10\,000$-resample bootstrap on the paired mean difference; the tail deltas carry the
same construction ($Delta p_95$ / $Delta"CVaR"_95$ CIs: Mamba vs joint-FTC $[..., ...]$ /
$[..., ...]$; vs FNPAG $[..., ...]$ / $[..., ...]$ -- values from the paired-comparison records).
```
Fill the brackets from the regenerated `results.json` (committed-data rule). Far-tail
(CVaR99.9) difference CIs come from Task 2's replicate deltas and land in Task 20's table edit,
not here.
- [ ] **Step 4:** Compile, proof the table page, commit.

---

### Task 8: Selection-pool rename + query count

**Files:**
- Create: `articles/paper/scripts/count_validation_queries.py` (20 lines: count `validation`
  records in the headline run's `run_*.jsonl` under the mamba_p962 training dir / bundle)
- Modify: `articles/paper/paper.typ` (Section 4, Appendix A)

- [ ] **Step 1:** Count how many times the in-training gate queried the reserved n=1000 pool over
  the headline run (JSONL `validation` records). Store the count in
  `articles/paper/data/results.json` under the headline entry (rerun aggregator or write a small
  sidecar committed to data/).
- [ ] **Step 2:** Rename in prose: the reserved n=1000 in-training pool becomes the "selection
  pool" everywhere it is described as validation ("validation gate" -> "selection gate"; keep
  "validation RMS" for the loss metric but define it once as measured on the selection pool).
  In Appendix A Seed pools paragraph, state: "the selection pool was queried N times over the
  headline run; it is therefore adaptively reused and is NOT an unbiased estimate of
  generalization -- the confirmatory pool serves that role." (fill N).
- [ ] **Step 3:** Compile, proof, commit.

---

### Task 9: LSTM feasibility-first presentation

**Files:**
- Modify: `articles/paper/paper.typ` (Section 6.2 ~lines 634-641, tbl-perf ~lines 781-814,
  tbl-paired LSTM footnote)

Data (already committed in `far_tail_eval.json` + Task 2): LSTM s1 CVaR99.9 123.2 with 14.4%
heat-load violations; s2 134.2 / s3 130.1 clean -> feasible-only mean 132.1 (recompute from the
confirmatory pool when Task 2 lands; use those values).

- [ ] **Step 1: Section 6.2 edit.** After the existing asterisk paragraph, add the
  feasibility-first ranking sentence:

```typst
Under a feasibility-first rule -- rank only runs that satisfy every constraint on the sizing pool --
the LSTM's mean is taken over its two feasible seeds ($...$ m/s $"CVaR"_(99.9)$, confirmatory pool),
and the ordering Mamba $<$ LSTM $<$ dense is unchanged. We adopt that rule for the deployment
choice: the deployed cell must be feasible on every pool it was evaluated on, which the Mamba is.
```
(fill the number from `confirmatory_eval.json`).
- [ ] **Step 2: tbl-perf.** Add an `NN -- LSTM` row (three-seed mean with dagger, plus the
  feasible-only value in the caption or a second daggered footnote). Dagger note: "one of the three
  LSTM seeds exceeds the heat-load limit on $14.4%$ of the sizing pool; the bracketed value is the
  feasible-seeds-only mean." This also closes R2's Table 3 dagger request (R2-N5).
- [ ] **Step 3:** VERIFY the table structurally (column count vs cell count - the Table 4 lesson:
  count cells per row) AND on the rendered page proof. Commit.

---

### Task 10: Compute-claim hardening (data)

**Files:**
- Modify: `articles/paper/scripts/compute_benchmark.py` (record hardware/toolchain metadata +
  per-run timing spread + guidance-update count if absent)
- Modify: `articles/paper/data/compute_benchmark.json` (regenerated)
- Modify: `articles/paper/paper.typ` Appendix A Timing paragraph (~line 1064)

- [ ] **Step 1:** Read `compute_benchmark.py` + its JSON. Extend it to record:
  `sysctl -n machdep.cpu.brand_string` output, `rustc --version`, the cargo profile (release,
  lto), f64 arithmetic, the number of guidance updates per simulation (count guidance calls from
  the config cadence and sim duration: read `periods` from the mission TOML and the mean flight
  time from the headline `final_eval.parquet`), and the per-run timing distribution (p5/p50/p95
  over the 200 runs).
- [ ] **Step 2:** Re-run the benchmark ON AN IDLE BOX (ask Gregory to keep the box quiet for ~10
  minutes, or schedule with Task 5's handoff). Commit the regenerated JSON. If timings shift vs
  the quoted 3.68/2.40/1.25/86.1, update every quote from the new JSON (committed-data rule) - the
  RATIOS are what the text leans on.
- [ ] **Step 3:** Rewrite the Appendix A Timing paragraph:

```typst
*Timing.* Wall-clock per complete simulation over $200$ sequential runs of each deployed scheme on
one idle performance core of an <chip string> laptop (native code, $64$-bit floats, LTO release
build; single-threaded); per-scheme spread p5--p95 within $plus.minus ...%$. A simulation spans
$approx ...$ guidance updates, so the per-update cost of the deployed network is $approx ...$ (mu)s
including the surrounding simulation -- a loose upper bound on pure inference. These are
implementation-and-host measurements supporting the RELATIVE comparison; flight-processor timing,
worst-case execution time, and memory footprint are not established here (Section 9).
```
(fill from the JSON; keep the de-tooling rule - no tool names.)
- [ ] **Step 4:** Compile, proof, commit. (The RAD750 paragraph itself is Task 18; the "dominated
  outright" softening is Task 15.)

---

### Task 11: Heat-load unit unification (MJ/m2 everywhere)

**Files:**
- Modify: `src/python/aerocapture/training/charts.py` (heat-load chart axis kJ/m2 -> MJ/m2; keep
  the data pipeline in kJ, divide by 1e3 at plot time; check `chart_heat_load_time` and any
  constraint-limit line it draws)
- Regenerate: `articles/paper/figures/appendix/*/heat_load.svg` + card stats units via
  `uv run python articles/paper/scripts/collect_appendix.py` (re-runs 10 x 1000 sims, minutes;
  drift-check output must stay [OK] for all 10)
- Modify: `articles/paper/appendix.typ` if it prints heat-load numbers (convert to MJ/m2 with one
  decimal)

- [ ] **Step 1:** Make the axis change in charts.py; run the chart-generation pytest slice
  (`uv run pytest tests -k "heat_load" -v`) and fix any snapshot expectations.
- [ ] **Step 2:** Re-collect appendix cards; verify one card's heat-load panel shows the 25 MJ/m2
  limit line and MJ units on the axis.
- [ ] **Step 3:** rg the paper for any remaining `kJ/m` occurrence; convert. Compile, proof one
  card page, commit. (Report charts outside the paper keep kJ - scope the change to what the paper
  includes; if charts.py is shared, add a `unit_scale` parameter defaulting to legacy kJ and pass
  MJ from collect_appendix.py only.)

---

### Task 12: fig_loss_vs_tail connector lines

**Files:**
- Modify: `articles/paper/scripts/fig_loss_vs_tail.py` (~line 61)

- [ ] **Step 1:** Delete the `ax.plot(...)` per-family connector (keep markers + family colors);
  if within-family ordering needs signaling, replace with a light alpha-shaded convex span per
  family or nothing - R1 is right that lines read as trajectories.
- [ ] **Step 2:** Regenerate the SVG; update the caption's "(overall Spearman rho = 0.91)" to add
  "computed across the eleven runs pooled over families; the runs are not exchangeable across
  families, so read the coefficient as descriptive" (R1-S6 wording). Compile, proof, commit.

---

### Task 13: Corridor figure - rename + constant-bank boundaries

**Files:**
- Modify: `articles/paper/scripts/collect_corridor.py` (add `--boundaries-only` mode: two
  undispersed constant-bank runs - bank 0 deg = full lift-up / overshoot side, bank 180 deg = full
  lift-down / undershoot side - via the same piecewise config with all `bank_angle_i` fixed and
  the `nominal_flight_overrides` all-domains-off pattern from `reference.py`; APPEND
  `liftup_energy/liftup_pdyn/liftdown_energy/liftdown_pdyn` arrays to the existing
  `articles/paper/data/corridor.npz` without re-running the 1M-run histogram)
- Modify: `articles/paper/scripts/fig_corridor.py` (overlay the two boundaries as labeled dashed
  lines)
- Modify: `articles/paper/paper.typ` fig-corridor caption + the two prose mentions (~lines 180-199)

- [ ] **Step 1:** Implement + run `--boundaries-only`; verify the npz gains the four arrays and
  the histogram arrays are byte-identical (load both versions, `np.array_equal` on the old keys).
- [ ] **Step 2:** Figure overlay + caption rewrite:

```typst
Empirical trajectory-occupancy envelope of the aerocapture corridor in the (orbital energy, dynamic
pressure) plane, traced by a $1\,000\,000$-run dispersed Monte Carlo of randomized signed
piecewise-constant bank profiles (roll reversals included), with the undispersed full-lift-up and
full-lift-down constant-bank boundaries overlaid (dashed). The shaded band spans the occupied
corridor: the upper edge is the $p_(99.9)$ dynamic pressure of capturing trajectories (crash side),
the lower edge the $p_(0.5)$ of trajectories capturing below a $5000$ km apoapsis (escape side); as
quantiles of sampled profiles these are empirical, not a formal reachable set. The vehicle enters
hyperbolic ($E > 0$, right) and bleeds energy into a bound orbit ($E < 0$, left); the $200$-run
Monte Carlo ensemble of the deployed Mamba policy and its undispersed nominal (heavy line) fly well
inside the envelope.
```
Also change the two "reachable ... corridor" prose references to "occupancy envelope" (Section 2.1
and Appendix C intro). This same caption fixes R1-3b ("deployed Mamba ensemble").
- [ ] **Step 3:** Compile, proof page 2, commit.

---

### Task 14: Stress-regime joint presentation + no-failures phrasing

**Files:**
- Modify: `articles/paper/paper.typ` Sections 7.2-7.3 (~lines 760-899), fig captions fig-robust +
  fig-objcenter, abstract + tbl-perf caption (phrasing only; numbers final in Task 20)

- [ ] **Step 1:** Everywhere a tail statistic is computed over captures under differing capture
  rates (Sections 7.2, 7.3), write the metric as $"CVaR"_95 (Delta v | "capture")$ on first use
  per section and add the lexicographic reading:

```typst
Because the conditional tail can improve by failing the hardest scenarios, we read these
comparisons lexicographically -- capture probability first, conditional tail cost second -- and
never claim a win on the tail across a capture-rate deficit.
```
Audit both sections against that rule: the centered-Mamba vs joint-FTC-retrained comparison
(94.9% vs the FTC cells' rates - read the rates from
`articles/paper/data/robustness_retrain.json` / `objective_centering.json`) must be stated with
both coordinates, and any "beats" wording where capture differs gets the explicit both-axes form.
- [ ] **Step 2:** "100% capture" phrasing: in tbl-perf caption and Section 6.2, add once:

```typst
Capture rates of $100%$ are "no failures observed": with zero failures in $n$ independent
scenarios the one-sided $95%$ upper confidence bound on the failure probability is $approx 3\/n$
($3 times 10^(-4)$ at $n = 10\,000$; $3 times 10^(-6)$ at $n = 10^6$).
```
- [ ] **Step 3:** Soften Section 7.3's conclusion sentence: "the off-nominal gap is *not*
  intrinsic" -> "the off-nominal gap is not intrinsic to neural guidance on this evidence --
  single-run, $n = 1000$, capture-rate-unequal -- and Section 9 records what a sizing-grade
  version requires" (final wording adjusted again in Task 19 if the sigma_run extras land in
  time). Compile, proof, commit.

---

### Task 15: Text batch A - the verified-wrong statements

**Files:**
- Modify: `articles/paper/paper.typ` (all edits below), `articles/paper/data/` (one nominal-dv1
  sidecar)

- [ ] **Step 1: L6 objective (Section 4.2, ~line 440).** Replace:

OLD: `The transform is ranking-neutral for a deterministic argmin, but under the noisy, non-stationary objective it changes which individuals survive selection.`

NEW:
```typst
Applied per scenario before the root-mean-square aggregation over the individual's batch
(Appendix A), the cubed transform makes the per-individual objective
$ J = ( 1/n sum_(i=1)^n C_i^6 )^(1\/2) , $
a monotone function of the $L_6$ norm of the per-scenario cost vector -- a high-moment objective
that weights an individual's worst scenarios far more heavily than its typical ones. (For a single
scenario a monotone transform is ranking-neutral; across a batch it deliberately is not.) We prefer
this smooth high-moment proxy over optimizing an empirical tail quantile directly because the
deployed allocation evaluates as few as two scenarios per individual per generation -- far too few
to estimate a quantile -- while the moving batch supplies tail coverage across generations.
```
Also align the Appendix A "Cost and objective" paragraph (~line 1021): append "so the aggregate
objective is the $L_6$-norm form above" after the RMS sentence.
- [ ] **Step 2: RL paragraph (Section 5, ~lines 509-515).** Replace the sentence block
  "Policy-gradient reinforcement learning ... avoids." with:

```typst
Policy-gradient reinforcement learning (PPO @schulman2017ppo, SAC @haarnoja2018sac) does not need
a differentiable simulator -- it estimates gradients from sampled rollouts and can in principle
optimize a terminal-only reward. In practice, on this problem, it needed a dense shaped surrogate
to learn at all: we implemented and trained both algorithms with potential-based per-step shaping
aligned to the predicted correction cost plus the true terminal cost, and the best runs
underperformed the population methods by a wide margin (dense PPO $...$ m/s mean / $...$
$"CVaR"_95$ and recurrent PPO $...$/$...$, against $...$/$...$ for the population-trained dense
network on the same earlier simulator regime -- a $3$--$5 times$ gap we did not close, consistent
with the stochastic-return objective optimizing a different quantity than the deterministic
mission cost). Population search on the mission cost itself was simply the stronger tool here, so
throughout we optimize the mission cost directly.
```
Fill the numbers from `articles/paper/data/runs/legacy/` (read the legacy entries in
`results.json`; expected magnitudes: gru_ppo ~513/829, dense rl ~636/973, population atan2 base
~119/132 - verify before quoting; add a footnote that the RL cells predate two simulator fixes
and are footnoted as the legacy regime). This also closes R2-B.
- [ ] **Step 3: CMA-ES (Section 4.1, ~lines 419-429 + fig-seed caption ~line 431).** Replace "--
  because it already resamples internally through its covariance adaptation, so a moving scenario
  batch tells it nothing new (@fig-seed)." with:

```typst
-- it neither suffers under the fixed batch nor benefits from the moving one (@fig-seed). We state
the asymmetry empirically rather than mechanistically: a plausible reading is that CMA-ES's own
parameter-space sampling already decorrelates successive generations while scenario noise chiefly
perturbs its rank-based updates and step-size control -- consistent with its self-termination on
the noisy objective (Section 5) -- but our experiments were not designed to isolate that mechanism,
and we leave it as an observation.
```
And in the fig-seed caption replace "CMA-ES is essentially unchanged because it already resamples
internally." with "CMA-ES is essentially unchanged; candidate mechanisms are discussed in the
text.". Keep the following sentence about "the lever is the non-stationarity" - it is empirical
(iso-compute control) and correct. This also closes R2-C.
- [ ] **Step 4: Section 5 integration contradiction (~line 509).** Replace "produced by a
  simulator with adaptive integration, sub-tick event detection, and discrete capture/crash
  termination" with "produced by a simulator with discrete capture, crash, and atmosphere-exit
  termination, threshold-triggered phase transitions, and hard constraint limits" (true of the
  fixed-step campaign configuration; the non-differentiability argument survives intact).
- [ ] **Step 5: The floor (Section 2.2, ~line 205).** First compute the current-interface floor:
  run the deployed Mamba undispersed nominal (all dispersion domains off - reuse
  `nominal_flight_overrides`) and read its dv1; write
  `articles/paper/data/nominal_floor.json` `{"nominal_dv1_m_s": ..., "pool_min_dv1_m_s": 103.85, "pool_min_dv_total_m_s": 104.46}`
  (pool values from the appendix card stats). Then replace:

OLD: `correction cost has an irreducible floor of roughly $113$ m/s set by the nominal periapsis raise; a`
NEW: `correction cost has an irreducible floor of roughly $10X$ m/s at this entry interface, set by the periapsis raise (the 2009 study's interface put it near $113$ m/s); a`
(fill 10X from the JSON; the intro's 113 stays - it describes the 2009 result.)
- [ ] **Step 6: Promotional-language pass (R1's list).**
  - Abstract "One honest caveat remains:" -> "The main deployment caveat:"
  - ~line 632 "the recurrent state delivers a tail one can trust." -> "the recurrent state's tail
    estimate is consistent across retraining."
  - ~line 460 "the canonical optimize-the-average, blow-up-the-tail failure." -> "a clean instance
    of mean optimization degrading the worst case."
  - ~line 756 "FNPAG is dominated outright: joint-FTC matches its accuracy and the network beats
    it, both at a small fraction of its cost." -> "On accuracy and compute FNPAG is dominated --
    joint-FTC matches its accuracy and the network beats it, both at a small fraction of its cost
    -- while off-nominal robustness remains a separate axis (Section 7.3: FNPAG loses less capture
    than the network under the stress regime, though it inflates its tail more)."
  - Keep "the right optimizer"-adjacent phrasing factual: ~line 427 "the genetic algorithm becomes
    the best optimizer in the study" is a measured statement - leave it.
- [ ] **Step 7:** Compile, page-proof every touched page, commit
  (`git commit -m "paper: correct L6 objective, RL, CMA-ES, integration, floor; neutralize flagged language (R1-8, R1-9b, R1-P1/P4, R2-B/C)"`).

---

### Task 16: Text batch B - scoping, nits, and the data-split table

**Files:**
- Modify: `articles/paper/paper.typ`

- [ ] **Step 1: Data-split table** (new, end of Section 4 or Appendix A; R1's "state which
  decisions were made using each pool"):

```typst
#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    align: (left, center, center, left),
    table.hline(stroke: 0.7pt),
    table.header([*Pool*], [*n*], [*Queries*], [*Decisions taken on it*]),
    table.hline(stroke: 0.35pt),
    [Training batches], [2/gen], [every gen], [weight updates (moving, curated)],
    [Selection pool], [$1000$], [$N$ over the run], [in-training argmin promotion],
    [Development far tail], [$10\,000$], [tens], [cost transform, curation bucket, allocation, cell type, headline choice],
    [Fresh re-quote], [$1000$], [once], [none (reported only)],
    [Confirmatory sizing], [$10 times 100\,000$], [once, post-freeze], [none -- every quoted sizing number],
    [Off-nominal stress], [$1000$], [once per policy], [none (robustness probe)],
    table.hline(stroke: 0.7pt),
  ),
  caption: [Scenario-pool roles. Pools above the line influenced design choices and are reported as
  development quantities; the confirmatory pool was generated from a seed range disjoint from every
  earlier draw, after all methodology, architecture, and checkpoint choices were frozen, and each
  cell was evaluated on it exactly once.],
) <tbl-pools>
```
(fill N from Task 8; adjust wording to final pool naming.)
- [ ] **Step 2: LHS scoping fix (Section 2.3, ~line 261).** The eval pools draw ONE scenario per
  seed, so LHS never stratifies them (verified in the eval scripts). Replace "Draws are generated
  by Latin-hypercube sampling @mckay1979lhs for space-filling coverage." with "Within a
  multi-scenario batch, draws are generated by Latin-hypercube sampling @mckay1979lhs; the
  evaluation pools draw one scenario per seed and are therefore plain independent samples (which
  is what the replicate-based intervals of Section 7 assume)." (Also defuses R1's
  bootstrap-under-LHS objection - note it in the response doc.)
- [ ] **Step 3: CVaR estimator definition (Section 2.2, after the CVaR sentence ~line 219).** Add:
  "Empirically, $"CVaR"_alpha$ is the mean of the worst $ceil((1-alpha) n_"captured")$ observations
  (no interpolation); every tail statistic is reported with the number of observations it averages."
  Then ensure tbl-perf's caption states the count for its CVaR99.9 column (10 at n=10k; 1000
  pooled on the confirmatory pool - final numbers in Task 20).
- [ ] **Step 4: Nits.**
  - `rg -n "Monte-Carlo" articles/paper/paper.typ` -> replace all with "Monte Carlo" (R1 style).
  - Section 2.1 orbit line: "targets a $500 times 11$ km orbit" -> "targets a $500 times 11$ km
    (apoapsis $times$ periapsis altitude) orbit".
  - tbl-paired caption: append "Deltas are computed on unrounded per-scenario values, so a delta
    may differ from the difference of the rounded marginals in @tbl-perf by $0.1$ m/s." (R2-N4).
  - Wilcoxon column: replace the three `$3 times 10^(-165)$` entries with `$< 10^(-15)$` and adjust
    the saturation footnote to "the normal-approximation statistic saturates near sign unanimity;
    we truncate at $10^(-15)$" (R1-S4).
  - Section 8 (~line 922): "the residual correction cost is irreducible scenario noise, not a
    pocket of mishandled cases" -> "the residual correction cost shows no clustered failure mode;
    we did not attempt a per-scenario lower-bound analysis, so 'irreducible' is not claimed"
    (R1-P5).
  - Section 8 framing (R1-P8): in the opening sentence of Section 8 and the fig-ablation caption,
    name the method "a closed-loop input-sensitivity analysis (zeroing a normalized input perturbs
    every subsequent state, so deltas measure closed-loop dependence, not isolated feature
    importance)"; keep the ranking discussion but drop any "importance" wording.
  - Section 4/5 exploratory-vs-confirmatory sentence (R1-S5), one line at the top of Section 5:
    "The optimizer and shaping studies of Sections 4--5 are exploratory (single training runs
    unless stated); the confirmatory statements of this paper are the frozen-pool quantities of
    Sections 6--7."
- [ ] **Step 5:** Compile, proof touched pages, commit.

---

### Task 17: Related work + novelty narrowing (needs web verification)

**Files:**
- Modify: `articles/paper/refs.bib`, `articles/paper/paper.typ` (intro contribution 3 ~line 148,
  Section 3 FNPAG paragraph, a short related-work paragraph at the end of Section 1)

- [ ] **Step 1: Verify R1's two pointers** (WebSearch/WebFetch): DOI `10.2514/6.2021-1569`
  (two-stage optimization for aerocapture guidance) and `10.2514/6.2017-1901` (fully numerical
  guidance applied to Mars aerocapture - the FNPAG Mars-application follow-up). Add bib entries
  ONLY after confirming venue/year/authors on the AIAA landing pages (R1's links carry
  chatgpt-utm tags - verify, do not trust).
- [ ] **Step 2: Sweep for the adjacent literature** (searches: "aerocapture guidance" 2016-2026;
  "convex optimization aerocapture predictor corrector"; "machine learning aerocapture guidance
  comparison"; "reinforcement learning entry guidance"; "aerocapture neural network"). Targets:
  the 2026 NPC-vs-ML aerocapture comparison R1 alludes to (find the real citation or drop the
  concern into the response doc as unlocatable), stochastic/robust two-stage guidance, augmented
  analytical guidance, and 1-2 representative RL-entry-guidance papers. Add 4-8 verified entries
  with DOIs.
- [ ] **Step 3: Related-work paragraph** (end of Section 1, before the contributions or after
  them): 6-10 sentences situating the paper against (a) FNPAG and its Mars applications, (b)
  optimization-based and convex aerocapture guidance, (c) learning-based entry/aerocapture
  guidance, (d) what none of them report - paired-scenario far-tail correction-Deltav risk with
  co-tuned classical baselines. Then narrow contribution 3:

OLD: `*The first systematic head-to-head of neural versus predictor--corrector aerocapture guidance.*`
NEW: `*A systematic head-to-head of neural versus predictor--corrector aerocapture guidance* -- to
our knowledge the first for an MSR-class Mars aerocapture that compares them on paired dispersed
scenarios under a far-tail correction-$Delta v$ risk metric, with the classical baselines co-tuned
on the same objective.`
(adjust the qualifier to whatever the verified literature supports).
- [ ] **Step 4:** Compile (bibliography regenerates), check every new citation renders, proof,
  commit.

---

### Task 18: Appendix A expansion, vacuum-conservation test, release + R2 paragraphs

**Files:**
- Modify: `articles/paper/paper.typ` (Appendix A ~lines 1001-1068; Section 9 ~line 930-963)
- Create: `src/rust/tests/test_conservation.rs` + `tests/reference_data/vacuum_atmosphere.dat`
  (a 2-point near-zero-density table)
- Modify: `articles/paper/refs.bib` (RAD750-class reference if a citable source is found in Task
  17's sweep; otherwise state the scaling as illustrative without a citation)

- [ ] **Step 1: Vacuum conservation test (Rust).** New integration test: a config (inline or
  fixture under `tests/common/`) pointing at the near-zero-density atmosphere table, dispersions
  off, J2=J3=J4 zeroed in the `[planet]` block, fixed RK4; run via the library API with
  trajectories on; assert the trajectory energy column is constant to relative 1e-9 and recompute
  angular momentum from two state samples to the same tolerance. Follow the existing
  `src/rust/tests/` fixture patterns (read `tests/common/fixtures.rs` first). TDD: write test,
  watch it fail on the missing fixture, add fixture, pass. `cargo test conservation`.
- [ ] **Step 2: Appendix A additions** (each a compact paragraph or display equation; de-tooling
  rule applies):
  - *Correction burns.* The three-burn plan equations (read `src/rust/src/orbit/maneuver.rs`
    `compute_deltav` and transcribe faithfully): periapsis raise at apoapsis (vis-viva),
    circularization at the target radius, plane change $2 v sin(Delta i \/ 2)$; state the summed
    total is the reported $Delta v$.
  - *Non-capture virtual cost.* $C = 3000 + 1000 min(|E - E_"target"|, 50) - 500 t\/t_max$ (m/s;
    energies in MJ/kg) for crash/timeout; $10\,000 + v_infinity$ for hyperbolic escape - verify
    the constants against `sim_types.rs` / `evaluate.py` before writing.
  - *Soft penalties.* The normalized exceedance form from `compute_cost` (read `evaluate.py`).
  - *Adaptive seed curation pseudocode* (8-10 lines: probe 1000 seeds with the top individual,
    sort per-seed cost, quantile-bin, take the hardest per bin, refresh on promotion or every 2
    generations).
  - *Optimizer settings.* SBX/PM eta values and any non-default GA knobs from
    `configs/training/common.toml`.
  - *Confirmatory-pool protocol.* The Task 2 protocol paragraph: freeze commit, seed-range
    disjointness argument, replicate design, estimator, query discipline.
  - *High-regime dispersion table.* Exact numeric high presets for the four raised domains
    (read the presets in `src/rust/src/data/dispersions.rs`; small table mirroring
    tbl-dispersions' columns).
  - *Dispersion rationale sentences* (R1 major 10): one sentence each on the plus/minus 50%
    density span (conservative envelope consistent with Mars GCM seasonal/dust variability),
    independence assumptions (stated as modeling choices), OU-plus-static-bias decomposition
    (bias = profile-scale uncertainty, OU = along-track variability; they are different
    frequencies, not double-counting), OU initialization (stationary distribution - verify in
    `dispersions.rs`), and why the two wind dims stay in the draw vector (stream stability;
    inert).
  - *Validation scoping.* Replace the Appendix A "validated against the 2009 study's legacy code"
    sentence with: "The implementation is regression-validated against the 2009 study's legacy
    code -- across all $725$ time steps of a guided trajectory, $22$ of $24$ output channels are
    bit-identical (the two mismatches trace to uninitialized variables in the reference). This
    establishes code equivalence with the flight-heritage implementation, not physical validation;
    independent physics checks (an analytic gravity-gradient oracle, integrator convergence on
    analytic systems, and vacuum energy/angular-momentum conservation) run in the test suite."
    Mirror the same scoping in the intro sentence (~line 154-160, "bit-validated" in the abstract
    -> "regression-validated (bit-level)" once).
  - *Classical tuning parity (R1-6c).* A short paragraph stating, in-paper: every classical scheme
    was tuned by the same genetic algorithm on the same correction-cost objective (identical
    cost function, penalties, and virtual costs), co-optimizing its full parameter set INCLUDING
    the shared navigation-filter and actuator-shaping gains (26 parameters for FTC versus the
    network's weights plus 3 actuator-side parameters); state each side's evaluation budget
    (classical: 2000 generations x 300 individuals x 10 scenarios; network headline: the Appendix A
    allocation) and note the classical runs converged well before budget exhaustion (read the
    per-scheme JSONL plateau generation to back this; cite the joint-reference co-optimization of
    Section 7.1 as the equal-freedom counterpart to the network's co-tuned actuator parameters).
  - *Artifacts / release.* Replace the Artifacts paragraph: "Source code (simulator, training
    harness, analysis), every configuration, the deployed network weights, the committed
    per-run evaluation records behind each table, and the scripts that regenerate every figure
    and number are public at <REPO-URL> (MIT license). Every number in the tables regenerates,
    without retraining, from those records." Ask Gregory for the final public URL before filling
    it; if unavailable at execution time, put the placeholder text "public repository (URL in the
    camera-ready)" and flag it in revision_state.json as blocked-on-user.
- [ ] **Step 3: R2 paragraphs.**
  - Section 7.2 Compute, after the timing sentences: "Flight processors run one to two orders of
    magnitude slower than the laptop core measured here; at a conservative $100 times$ scaling the
    numerical predictor--corrector's $86$ ms replan would approach its own $2$ s replan period,
    while the network's $3.7$ ms forward pass stays comfortably sub-second -- the relative
    ordering, not the absolute milliseconds, is the portable result." (R2-A; keep or add a
    RAD750-class citation only if verified in Task 17.)
  - Section 9, new short paragraph (R2-D): "A flight qualification path for a stateful network
    remains open. The natural architecture is a simplex arrangement: the analytic joint-FTC --
    whose off-nominal robustness Section 7.2 established -- runs in parallel as an onboard
    monitor with authority to take over on envelope violation, while the network flies the
    nominal corridor it wins on. The deployed policy is a fixed forward pass of $962$ parameters
    in double precision with no data-dependent control flow, so worst-case execution time and
    memory are trivially bounded; verification of the *decision* behavior across the dispersion
    envelope, rather than the code, is the open problem."
- [ ] **Step 4:** `./check_all.sh` (Rust test added), compile, proof Appendix A pages, commit.

---

### Task 19: Post-training evaluation + pre-registered interpretation (AFTER Gregory's runs)

**Files:**
- Modify: `articles/paper/data/confirmatory_eval.json`, `far_tail_eval.json` (new cells)
- Modify: `articles/paper/paper.typ` Sections 6.2-6.3, 8, 9; `articles/paper/data/results.json`
  (aggregator rerun with the new cells if bundled)

**Pre-registered interpretation rules (written BEFORE the results; keep verbatim in the response
doc):**
1. If the reset-state control (Task 4) degrades the deployed Mamba's confirmatory CVaR99.9 by more
   than its replicate SE, AND neither window_ctrl_p970 nor the dense reference reaches the intact
   Mamba's tail, the causal claim stands: keep "internal state compresses the extreme tail",
   now citing the controls.
2. If the reset-state control does NOT degrade the tail, the state claim is refuted regardless of
   the retrains: Section 6.3 is rewritten to attribute the tail win to the cell's optimization
   behavior, and the abstract/conclusion drop the state mechanism. Report it plainly - that is a
   finding, not a failure.
3. If results are mixed (reset degrades but window matches Mamba), the honest claim is "temporal
   information compresses the tail; whether learned state beats an explicit history window is
   unresolved" - hypothesis language per R1's suggested wording.
4. No-dv retrain: if nodv seeds lose more median than tail, the engineered-inputs-flatten-the-bulk
   claim gains direct support; if they collapse entirely, scope the claim to "predicted-dv inputs
   are necessary for the bulk" and note the classical-fairness implication both ways.

- [ ] **Step 1:** When `15_state_controls.sh` finishes: far-tail (n=10k dev) + confirmatory eval
  of the 6 cells (window_s1..3, nodv_s1..3) with bundle collection
  (`experiments/paper/12_collect_results.sh` pattern) and violation columns.
- [ ] **Step 2:** When `16_sigma_extras.sh` finishes: deployed-DV eval of the optimizer repeats
  (n=1000 final-eval pool, matching Study C's original metric) -> sigma_run per Study C cell;
  stress-pool eval (9M offset) of mamba_centered_s2/s3 -> Section 7.3 upgraded from "single-run"
  to mean +/- range over 3 runs (and the R1-P6 softening of Task 14 revisited: if the effect
  clears sigma_run, the claim firms up; state whichever the data supports).
- [ ] **Step 3:** Apply the pre-registered rule that matched; rewrite Section 6.3's mechanism
  paragraph, Section 9's "we did not run the state-ablation control" paragraph (now run - describe
  it), Study C sentences gain "+/- sigma_run" qualifiers, and Section 5's
  optimizer-ranking hedges cite the measured scatter.
- [ ] **Step 4:** Compile, proof, commit. Update revision_state.json (R1-5, R1-6, R1-S1, R1-P6).

---

### Task 20: Headline requote + estimand separation (final numbers pass)

**Files:**
- Modify: `articles/paper/paper.typ` - abstract (~lines 69-91), intro contribution 2/3
  (~lines 140-152), Section 6.2 (~lines 607-658), Section 6.3 deployed-policy paragraph
  (~lines 690-698), Section 7.2 accuracy paragraph (~lines 746-751), tbl-perf, tbl-paired,
  conclusion (~lines 965-994)

**Rules:** every sizing number now comes from `confirmatory_eval.json`. The abstract quotes THE
DEPLOYED ARTIFACT (mamba s1) pooled CVaR99.9 with its replicate CI - not the 3-seed mean. The
3-seed mean and range live in Section 6.2 as the ACROSS-RETRAINING quantity, explicitly labeled;
scenario uncertainty (replicate CI) and training-run uncertainty (seed range) are reported side by
side and never pooled (R1-3, R1-S3).

- [ ] **Step 1: Abstract sentence** (template; fill every bracket from the JSON):

```typst
A #box[$962$-parameter] recurrent (Mamba) policy captures every one of $10^6$ confirmatory
scenarios (a $95%$ upper bound of $3 times 10^(-6)$ on its failure probability) and reaches a
far-tail $"CVaR"_(99.9)$ of #box[$<pooled>$ m/s] ($plus.minus <ci>$ across ten replicate pools;
$<lo>$--$<hi>$ across three independent retrainings).
```
Keep the 16.4 / 27.6 paired sentence but requote from the confirmatory paired deltas with their
CIs if they moved; keep the state-mechanism sentence in whatever form Task 19 fixed.
- [ ] **Step 2: Section 6.2** - replace the three-seed ordering display and per-seed lists with
  confirmatory values (per-seed CVaR99.9 + replicate CIs; the sample-max non-overlap statement
  recomputed - verify it still holds at n=1e6 per seed, where "max" is p99.9999-grade: prefer
  CVaR99.99 or p99.99 for the non-overlap claim if the raw max gets noisy). tbl-perf: CVaR99.9
  column switches to confirmatory pooled values with the dagger note updated to "(pooled over
  $10 times 10^5$; network rows: deployed seed, with the three-seed range in Section 6.2)";
  fill the previously-empty cells (PredGuid, EC, eqglide, piecewise). tbl-paired: add the
  far-tail paired-delta row block from the `paired` section (mean +/- t-CI over replicates).
- [ ] **Step 3: Cross-check pass.** `rg -n "124\.5|122\.0|139\.2|129\.2|164\.0|165\.0" articles/paper/paper.typ`
  - every hit either updated to the confirmatory value or explicitly relabeled as the development
  pool (Section 6.2 may keep the dev-pool history in one parenthetical for continuity with the
  10k-pool figures). fig_arch_tail/fig_classical_vs_nn/fig_survival captions consistent with the
  numbers shown.
- [ ] **Step 4:** Compile, proof every changed page, commit.

---

### Task 21: Final proofs, response document, docs + memory

**Files:**
- Create: `articles/paper/REVIEW_RESPONSE_R4_R5.md`
- Modify: `paper_resume.md` (handoff block), memory `project_paper_state.md`,
  `articles/paper/revision_state.json` (final statuses)

- [ ] **Step 1: Full page-proof pass.** Compile; render PNG proofs of every changed page; check
  each table structurally (cells per row vs columns) AND visually. Refresh
  `~/Desktop/aerocapture_paper_2026-07-08.pdf`.
- [ ] **Step 2: Response document.** Point-by-point over both reports: finding -> action taken
  (with paper location) or rebuttal. Must include the rebuttals with evidence: R2's three
  extraction ghosts (quote the rendered-PDF text), the LHS-bootstrap defusal (eval pools are
  one-draw-per-seed iid), the classical-tuning fairness documentation (26 co-tuned classical
  params incl. nav/actuator vs the network's 3; same cost function; budgets stated), the
  ensemble-wording clarification, and the pre-registered interpretation rules of Task 19 quoted
  verbatim with which branch the data took.
- [ ] **Step 3:** Update the `paper_resume.md` handoff block (new commits, confirmatory-pool
  protocol, state-control verdict, what remains open) and the `project_paper_state` memory file
  (same content, one paragraph). Mark every revision_state.json finding applied/rejected with a
  one-line note.
- [ ] **Step 4:** Commit.

---

### Task 22: Smart-commit close-out

- [ ] **Step 1:** Invoke the `smart-commit` skill, telling it to take the WHOLE git branch into
  account (all revision commits since `5745050`), so CLAUDE.md / README stay in sync with the new
  scripts (`confirmatory_eval.py`, the reset flag, the two runners) before the final commit.

---

## Self-review notes

- Coverage: every `revision_state.json` finding maps to a task (R2-N1..N3 rejected with evidence).
- Data dependencies marked `<...>` are defined sources (confirmatory_eval.json / legacy bundle /
  benchmark JSON), each with an explicit fill instruction - no open TBDs.
- Task 2 Step 3's `<bundle_key>` placeholders are resolved by the listed `eza` check - the bundle
  keys exist but their exact names must be read, not guessed (far_tail lesson: a bad label
  silently scores an untrained default).
- Blocked-on-user points: launch of runners (Tasks 5-6), idle box for Task 10, public repo URL
  (Task 18). Everything else executes inline, serially.
