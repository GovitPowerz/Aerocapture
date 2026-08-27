# Training Fixes & DV Chart Improvements

**Date:** 2026-03-28
**Branch:** `feature/Fix_training`

## Overview

Three TODO items addressed across two specs:
- **Spec A** — DV distribution chart improvements (visualization)
- **Spec B** — GA training hang fixes, parameter scaling, and FTC config repair

---

## Spec A — DV Distribution Charts

### A1. Split individual burn histograms into 3-row subplot

Refactor `chart_dv_individual_burns` in `charts.py` from a single overlaid histogram into a **3-row vertical subplot figure** with shared log-scale x-axis:

- **Row 1:** dv1 — periapsis correction (blue)
- **Row 2:** dv2 — circularization (orange)
- **Row 3:** dv3 — inclination plane change (green)

Each row: 25-bin histogram, individual y-axis (counts), same seaborn theme. Only the bottom row shows x-axis tick labels. The figure is still one SVG — no report template changes needed.

### A2. Unified xticklabel logic

Extract a helper function `_log10_ticks(values: ndarray, floor: float = 1.0) -> tuple[list[float], list[str]]` that:

1. Takes absolute values, clips to `floor` minimum (1.0 m/s)
2. Computes `lo = max(0, floor(log10(min)))` and `hi = ceil(log10(max))`
3. Returns decade tick positions and `10^d`-formatted label strings

Apply to:
- `chart_dv_distribution` — replace hardcoded `_LOG10_TICK_VALUES = [0.1, 1, 10, 100, 1000, 5000]`
- `chart_dv_individual_burns` — replace inline decade calculation

Update `DV_FLOOR` from 0.1 to 1.0 m/s.

### Files changed (Spec A)

| File | Change |
|------|--------|
| `src/python/aerocapture/training/charts.py` | Refactor `chart_dv_individual_burns`, extract `_log10_ticks`, update `chart_dv_distribution`, change `DV_FLOOR` |

---

## Spec B — GA Training Hang Fixes

### B1. NaN termination in Rust sim loop

**Problem:** When the GA finds extreme parameter combinations late in training, the simulation state can become NaN. Since `NaN >= max_time` is `false`, `NaN <= 0.0` is `false`, etc., **no termination check fires** and the `while term == TermReason::None` loop spins forever.

**Fix:** In `runner.rs`, add a NaN/Inf check on the state vector **before** existing termination checks, immediately after each integration step:

```rust
if state.iter().any(|x| !x.is_finite()) {
    term = TermReason::Crash;
    break;
}
```

Use `TermReason::Crash` (not a new variant) to keep things simple — the virtual DV for crashes (~15,000 m/s) already penalizes these trajectories heavily.

### B2. Per-sim wall-clock timeout in PyO3 batch path

**Problem:** Even with B1, a single pathological sim could take very long (e.g., numerically stable but physically meaningless trajectory). With Rayon parallelism, one slow sim blocks the entire batch.

**Fix:** Add an optional wall-clock timeout per simulation, threaded from PyO3 through to the Rust sim loop:

- In `runner.rs`: `run_for_api()` accepts an optional `wall_timeout: Option<std::time::Duration>`. Inside the sim loop, record `Instant::now()` at start; check `elapsed() > wall_timeout` alongside existing termination checks. Terminate with `TermReason::Timeout` if exceeded.
- In `batch.rs`: pass the timeout through to each `run_for_api()` call within the Rayon parallel iterator.
- In `lib.rs` (PyO3): expose as optional `sim_timeout_secs: Option<f64>` on `run()`, `run_mc()`, and `run_batch()`. Default: `None` (no timeout, backward compatible). Training can pass e.g. `sim_timeout_secs=30.0`.
- The timeout result gets a high virtual DV (same as simulation-time timeout), so the GA learns to avoid these parameter regions.

### B3. Log-scale parameter encoding

**Problem:** Some parameter ranges span significant ratios but use linear encoding in the binary GA. For example, `pred_guid.pdyn_threshold` spans [10, 500] (50x range).

**Fix:** Add `log_scale=True` to parameter specs in `param_spaces.py` where the range ratio exceeds ~20x. Candidates:

| Parameter | Range | Ratio | Action |
|-----------|-------|-------|--------|
| `energy_ctrl.gain` | [1e-8, 1e-5] | 1000x | Already log-scale |
| `fnpag.energy_tol` | [1e2, 1e5] | 1000x | Already log-scale |
| `pred_guid.pdyn_threshold` | [10, 500] | 50x | **Add log-scale** |

Other parameters have ranges under 20x — log-scale would add complexity for marginal benefit. The binary GA already normalizes each parameter independently to its `[0, 2^n_bit - 1]` integer range, so inter-parameter magnitude differences don't affect operators.

### B4. Fix subprocess exception syntax

**Problem:** `evaluate.py:212` has `except subprocess.TimeoutExpired, FileNotFoundError:` — Python 2 comma syntax, SyntaxError in Python 3. Ruff's formatter keeps removing the parentheses.

**Fix:**
```python
except (subprocess.TimeoutExpired, FileNotFoundError):  # fmt: skip
```

The `# fmt: skip` comment prevents ruff from reformatting the line.

### B5. Fix FTC training TOML configuration

**Problem:** `configs/training/msr_aller_ftc_train.toml` specifies `type = "ftc"` but provides **no** `[guidance.ftc]` section. The Rust parser requires all 18+ fields (capture/exit gains, density filter, lateral params, pdyn table, etc.).

**Fix:** Copy the `[guidance.ftc]` section from the working nominal config (`configs/nominal/msr_aller_ftc_nominal.toml`) into the training TOML. The GA will override the trainable subset via TOML overrides; the rest need sensible defaults in the config.

Also add `altitude_damping` and `altitude_frequency` to the FTC param space in `param_spaces.py` — they're currently missing from the trainable parameter list. Reasonable bounds: `altitude_damping` [0.3, 1.5] (same scale as `capture_damping`), `altitude_frequency` [0.01, 0.2] (same scale as `capture_frequency`).

### Files changed (Spec B)

| File | Change |
|------|--------|
| `src/rust/src/simulation/runner.rs` | Add NaN/Inf state check before termination checks |
| `src/rust/aerocapture-py/src/batch.rs` | Add optional wall-clock timeout per sim |
| `src/rust/aerocapture-py/src/lib.rs` | Expose `sim_timeout_secs` parameter |
| `src/python/aerocapture/training/evaluate.py` | Fix `except` syntax + `# fmt: skip` |
| `src/python/aerocapture/training/param_spaces.py` | Add log-scale to `pdyn_threshold`; add `altitude_damping` and `altitude_frequency` to FTC bounds |
| `src/rust/src/simulation/runner.rs` | Thread `wall_timeout` into `run_for_api()` |
| `configs/training/msr_aller_ftc_train.toml` | Add full `[guidance.ftc]` section from nominal |

---

## Out of Scope

- **Early stopping / stagnation detection** — intentionally excluded; improvements can appear after hundreds of generations of stagnation.
- **Real-valued GA (SBX/polynomial mutation)** — deferred to TODO under Training & ML. Would eliminate scale-blind bit-flip problem entirely but is a larger rewrite.
- **Adaptive mutation rates** — deferred with real-valued GA.

## Testing

- **B1 (NaN termination):** Add a Rust unit test that injects NaN into the state vector and verifies `TermReason::Crash`.
- **B2 (wall-clock timeout):** Integration test with an artificially slow config that exceeds the timeout.
- **B4 (except syntax):** Existing CI (ruff + mypy + pytest) will catch regressions.
- **B5 (FTC TOML):** Run `aerocapture_rs.run()` with the fixed FTC training config and verify it doesn't error.
- **A1/A2 (charts):** Existing chart generation tests cover SVG output; visual verification of the new subplot layout.
