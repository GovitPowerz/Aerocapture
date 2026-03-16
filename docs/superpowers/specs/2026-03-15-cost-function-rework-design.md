# Cost Function Rework: Delta-V Primary with Constraint Penalties

**Date:** 2026-03-15
**Ref:** IMPROVEMENTS.md §12.4, TODO.md line 16

## Problem

The current cost function uses orbit altitude errors (`|Δapo| + |Δperi|`) as the primary objective with delta-V as a near-irrelevant secondary term (0.01× weight). Delta-V already encodes "how far off-target is this orbit" since it's the correction burn cost to reach the target orbit. The optimizer converges to solutions where orbit errors are low but the actual fuel cost to fix the remaining orbit is not meaningfully optimized.

Additionally, heat flux and g-load are computed by the simulator but not wired into the cost function, so the optimizer has no incentive to avoid constraint violations.

## Design

### Approach: Weighted Sum with Normalized Constraint Penalties

Single scalar objective — delta-V as primary, with soft constraint penalties for g-load and heat flux exceedances, normalized by their thresholds.

### Cost Formula

**Hyperbolic (non-capture) or bogus delta-V (>1e10) — unchanged:**

```
cost = 1e6 + 1e3 × |E| - 0.1 × t
```

- Penalizes excess orbital energy, rewards longer atmospheric flight time.
- Bogus delta-V (Fortran-era artifact where maneuver computation fails → 1e30) is treated identically to non-capture — the orbit is unusable.

**Captured trajectories with valid delta-V:**

```
cost = ΔV + w_g × max((g_max - g_lim) / g_lim, 0)² + w_q × max((q_max - q_lim) / q_lim, 0)²
```

Where:
- `ΔV` = total correction delta-V (m/s), clipped to [0, 10000]
- `g_max` = peak g-load during entry (g, from final record column 17)
- `g_lim` = g-load threshold (TOML-configurable, default: 15.0 g)
- `q_max` = peak heat flux during entry (kW/m², from final record column 16)
- `q_lim` = heat flux threshold (TOML-configurable, default: 200.0 kW/m²)
- `w_g` = g-load penalty weight (TOML-configurable, default: 1000.0)
- `w_q` = heat flux penalty weight (TOML-configurable, default: 1000.0)

**Why normalized squared penalties:**
- Normalization: 10% g-load exceedance and 10% heat exceedance contribute equally regardless of absolute scales. Weights become intuitive — both operate on fractional exceedance.
- Squared: smooth gradient near the boundary (zero at limit, grows quadratically beyond). No cliff edges.
- Zero contribution when under limit — no cost for staying within constraints.

**Aggregation — unchanged:**
- Single-seed: RMS across MC sims
- Adaptive seeds: α×mean + (1-α)×CVaR (default α=0.7, CVaR=20%)

## Rust Simulator Changes

### Peak Value Tracking in runner.rs

The 52-column final record has reserved columns for peak heat flux (index 16) and peak g-load (index 17) that are currently always written as 0.0. Two values need tracking during the integration loop:

1. **Peak heat flux (column 16, kW/m²):** The heat flux rate `dflux = Cq × √ρ × V^3.05` is already computed each step for integration into `state[6]`. Track `max_heat_flux = max(max_heat_flux, dflux)` and write to column 16 converted to kW/m².

2. **Peak g-load (column 17, g):** Derive from aerodynamic acceleration magnitude: `g_load = |F_aero| / (mass × g₀)` where `g₀ = 9.81 m/s²` (matches `G0` constant in `runner.rs:23` and `data/mod.rs:160`). Track `max_load_factor = max(max_load_factor, g_load)`.

Also populate the companion columns:
- Columns 19-21: altitude at peak flux / peak g-load / peak pdyn
- Columns 22-24: time at peak flux / peak g-load / peak pdyn

**Note:** The `SimState` struct already has `max_heat_flux`, `max_load_factor`, `max_dyn_pressure` fields (initialized to 0.0), and the final record output already reads from them (lines 609-611). The only missing piece is the actual tracking updates during the integration loop — the plumbing is in place.

## TOML Configuration

New `[cost_function]` section in training TOML configs:

```toml
[cost_function]
g_load_limit = 15.0          # g (Earth g's)
heat_flux_limit = 200.0      # kW/m²
g_load_weight = 1000.0       # penalty weight on normalized squared exceedance
heat_flux_weight = 1000.0    # penalty weight on normalized squared exceedance
```

- Lives in training TOMLs only (not simulation TOMLs) — cost function is a GA concern.
- Parsed in Python only — Rust simulator doesn't need these values.
- All four values have defaults so existing configs work without changes.

## Python Changes

### evaluate.py — `compute_cost()`

```python
def compute_cost(final_conditions, *, g_load_limit=15.0,
                 heat_flux_limit=200.0, g_load_weight=1000.0,
                 heat_flux_weight=1000.0):
    energy = final_conditions[:, 7]
    ecc = final_conditions[:, 9]
    sim_time = final_conditions[:, 27]
    dv_total = final_conditions[:, 41]
    g_max = final_conditions[:, 17]
    q_max = final_conditions[:, 16]

    hyperbolic = (ecc > 1.0) | (energy > 0)
    costs = np.zeros(len(final_conditions))

    # Non-capture OR bogus delta-V: energy-based penalty
    bad = hyperbolic | (dv_total > 1e10)
    costs[bad] = 1e6 + 1e3 * np.abs(energy[bad]) - 0.1 * sim_time[bad]

    # Captured with valid delta-V
    ok = ~bad
    dv = np.clip(dv_total[ok], 0, 1e4)
    g_penalty = g_load_weight * np.maximum((g_max[ok] - g_load_limit) / g_load_limit, 0) ** 2
    q_penalty = heat_flux_weight * np.maximum((q_max[ok] - heat_flux_limit) / heat_flux_limit, 0) ** 2
    costs[ok] = dv + g_penalty + q_penalty

    return float(np.sqrt(np.mean(costs ** 2)))
```

### evaluate.py — `evaluate_chromosome()`

`evaluate_chromosome` is the main entry point called by the GA loop. It calls `compute_cost()` internally. Thread cost function params through via `**cost_kwargs`:

- `evaluate_chromosome()` gains a `cost_kwargs: dict | None = None` parameter, forwarded to `compute_cost()`.
- All callers updated: `train.py`, `population.py`, `local_search.py`.

### evaluate.py — pre-existing bug fix

Line 212 has Python 2 except syntax: `except subprocess.TimeoutExpired, FileNotFoundError:` → fix to `except (subprocess.TimeoutExpired, FileNotFoundError):`. Unrelated to this feature but blocks Python 3.14 on the subprocess fallback path.

### train.py

- Parse `[cost_function]` from TOML config with defaults via `dict.get()`.
- Build `cost_kwargs` dict, pass through to `evaluate_chromosome()` calls.
- Thread through seed pool's `evaluate_population()` calls (which call `compute_cost` via closure).

### compare_guidance.py

- Thread cost function params through to `compute_cost()` calls.

### population.py, local_search.py

- Thread `cost_kwargs` through to `evaluate_chromosome()` calls.

### Training TOMLs (6 files in configs/training/)

- Add `[cost_function]` section with default values.

## Testing

### Python Tests

- Update existing `compute_cost` tests for new formula (delta-V primary, no orbit errors).
- New tests:
  - Constraint penalties: at limit (zero penalty), below limit (zero), above limit (correct quadratic value)
  - Normalization: verify 10% exceedance on g-load equals 10% exceedance on heat flux (same contribution at equal weights)
  - Weight=0 disables a penalty term
  - Bogus delta-V (>1e10) → non-capture cost path
  - Backward compatibility: default params produce valid costs

### Rust Tests

- Integration test: run an atmospheric trajectory and verify columns 16-17 are non-zero.
- Unit test: verify peak tracking logic (heat flux max, g-load max) with known inputs.

## What's NOT Changing

- Aggregation (RMS / CVaR blend)
- Seed pool / adaptive seeds
- Hyperbolic cost formula (except bogus-ΔV now also uses it)
- GA loop structure (single-objective, roulette selection)
- Final report / visualization (already reads ΔV)
- Curriculum learning (deferred — separate concern)

## Files Changed

| File | Change |
|------|--------|
| `src/rust/src/simulation/runner.rs` | Track max heat flux and max g-load during integration, write to final record columns 16-17 and companions 19-24 |
| `src/python/aerocapture/training/evaluate.py` | Rewrite `compute_cost()`, add `cost_kwargs` to `evaluate_chromosome()`, fix Python 2 except syntax on line 212 |
| `src/python/aerocapture/training/train.py` | Parse `[cost_function]` TOML section, build and thread `cost_kwargs` through all paths |
| `src/python/aerocapture/training/compare_guidance.py` | Thread cost function params through |
| `src/python/aerocapture/training/population.py` | Thread `cost_kwargs` to `evaluate_chromosome()` calls |
| `src/python/aerocapture/training/local_search.py` | Thread `cost_kwargs` to `evaluate_chromosome()` calls |
| `configs/training/*.toml` (6 files) | Add `[cost_function]` section |
| Python tests | Update + add constraint penalty tests |
| Rust tests | Add peak tracking integration/unit tests |
