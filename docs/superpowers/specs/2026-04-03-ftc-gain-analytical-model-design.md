# FTC Gain Analytical Model Design

## Problem

The FTC capture-phase guidance computes gains from a 26-entry altitude-dependent pdyn table (`compute_gains()` in `ftc.rs`). The table covers 0-96.25 km. Above 96.25 km, the code falls back to the last entry, creating a ~1000x discontinuity in `gain_dynamic_pressure` (last entry `coeff_a = -0.0000010000` vs penultimate `coeff_a = -0.0010156255`). This causes transient bank angle spikes when the spacecraft crosses the table ceiling during initial entry and final exit.

Additionally, `gain_altitude_rate` (which is altitude-independent) never fades at high altitude, meaning the controller applies corrections where there is negligible aerodynamic authority.

The `cos_bank` clamp to [-1, 1] masks invalid acos inputs but still slams the bank angle to 0 or 180 deg transiently.

## Approach

Replace the pdyn lookup table with an analytical exponential decay model for the pressure coefficient, plus a cosine fade multiplier applied to both gains. The fade parameters are GA-tunable.

**Chosen over alternatives:**
- Density-proportional gain (couples gains to time-varying filter state -- risky for GA stability)
- Power-law with cutoff (hard zero at boundary, less physically motivated)

## Design

### Analytical Gain Model

Replace the table lookup with two formulas:

**Pressure coefficient (exponential decay):**

```
pressure_coeff(h) = pressure_coeff_base * exp(-h_km / pressure_coeff_scale_height)
```

**Cosine fade multiplier (applied to both gains):**

```
t = clamp((h_km - gain_fade_start_km) / (gain_fade_end_km - gain_fade_start_km), 0, 1)
fade = 0.5 * (1 + cos(pi * t))
```

**Full gain computation:**

```rust
let pressure_coeff = base * exp(-alt_km / scale_height);
let fade = cosine_fade(alt_km, fade_start, fade_end);

gain_altitude_rate = fade * (-2 * zeta * omega * m) / (S * Cz);
gain_dynamic_pressure = fade * (-omega^2 * m) / (pressure_coeff * S * Cz);
```

**Degenerate case:** If `gain_fade_end_km <= gain_fade_start_km`, fade = 1.0 (no fade). This prevents the GA from wasting effort on invalid combinations without requiring a repair operator.

### TOML Configuration

Replace the `pdyn_table` array in `[guidance.ftc]` with 4 scalar fields:

```toml
[guidance.ftc]
pressure_coeff_base = -0.001
pressure_coeff_scale_height = 10.0
gain_fade_start_km = 80.0
gain_fade_end_km = 100.0
```

Defaults (for backward compat when `[guidance.ftc]` exists but these fields are absent):
- `pressure_coeff_base`: -0.001
- `pressure_coeff_scale_height`: 10.0
- `gain_fade_start_km`: 80.0
- `gain_fade_end_km`: 100.0

### GA Parameter Space

Add 4 new entries to the `"ftc"` param spec in `param_spaces.py`:

```python
ParamSpec("pressure_coeff_base", -0.01, -0.0001, -0.001),
ParamSpec("pressure_coeff_scale_height", 5.0, 20.0, 10.0),
ParamSpec("gain_fade_start_km", 60.0, 90.0, 80.0),
ParamSpec("gain_fade_end_km", 85.0, 120.0, 100.0),
```

No routing prefix needed -- unprefixed params route to `[guidance.ftc]` automatically.

Bounds rationale:
- `pressure_coeff_base`: negative (matching existing `coeff_a` signs), range spans the table's variation
- `pressure_coeff_scale_height`: [5, 20] km covers sub-scale-height to super-scale-height decay rates
- `gain_fade_start_km`: [60, 90] -- below sensible atmosphere ceiling, above active flight regime
- `gain_fade_end_km`: [85, 120] -- allows GA to find optimal taper width

### Scope of Impact

- Only FTC capture-phase guidance is affected. No other guidance scheme references the pdyn table.
- The `[guidance.ftc]` section is shared for exit-phase params by all unsigned-magnitude schemes, but exit guidance does not use the pdyn table or the new analytical fields.

## Files Changed

### Rust

| File | Change |
|------|--------|
| `src/rust/src/gnc/guidance/ftc.rs` | Replace `compute_gains()` table lookup with analytical model + cosine fade |
| `src/rust/src/data/guidance_params.rs` | Remove `DynamicPressureTableEntry`, `pdyn_table` field. Add 4 new fields to `GuidanceParams` |
| `src/rust/src/config.rs` | Remove `TomlPdynEntry`, `pdyn_table` from `TomlFtcParams`. Add 4 new `f64` fields with `#[serde(default = "...")]` functions matching the defaults above |
| `src/rust/src/data/mod.rs` | Remove table-to-struct mapping. Wire new fields from TOML to `GuidanceParams` |

### Python

| File | Change |
|------|--------|
| `src/python/aerocapture/training/param_spaces.py` | Add 4 new `ParamSpec` entries to `"ftc"` |

### Config

| File | Change |
|------|--------|
| `configs/training/msr_aller_ftc_train.toml` | Remove `pdyn_table` array, add 4 scalar fields |

### Tests

| File | Change |
|------|--------|
| `src/rust/src/gnc/guidance/ftc.rs` (inline tests) | New unit tests for fade, exponential decay, degenerate cases, proptest |
| `tests/reference_data/rust_golden/` | Regenerate FTC golden file only |

## Removed Artifacts

- `DynamicPressureTableEntry` struct
- `TomlPdynEntry` struct
- `pdyn_table` field from `GuidanceParams` and `TomlFtcParams`
- 26-entry pdyn_table from `msr_aller_ftc_train.toml`
- `coeff_b` (was already unused -- `#[allow(dead_code)]` on the struct confirms this)

## Tests

### New unit tests in `ftc.rs`

1. Fade is 1.0 below `gain_fade_start_km`
2. Fade is 0.0 above `gain_fade_end_km`
3. Fade is 0.5 at midpoint (cosine symmetry)
4. Fade is monotonically decreasing between start and end
5. Degenerate: `fade_end <= fade_start` produces fade = 1.0
6. `pressure_coeff` magnitude decreases with altitude (exponential decay)
7. Both gains are zero when fade is zero
8. Proptest: gains are finite for any altitude in [0, 500] km

### Golden file regeneration

Run updated binary on the FTC test config, replace the FTC golden CSV. Other schemes' golden files are unaffected.
