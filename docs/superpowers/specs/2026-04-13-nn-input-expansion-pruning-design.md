# NN Input Expansion and Pruning Design

## Problem

The NN guidance scheme underperforms FTC despite having full trajectory coverage (capture + exit phases). Root cause: massive structural asymmetry. FTC gets a pre-optimized reference trajectory, a dedicated exit controller, lateral guidance, and a thermal limiter -- all as hardcoded structure with 26 interpretable parameters. The NN must learn equivalent behavior from 16 raw scalar inputs with 434 black-box weights and no structural scaffolding.

The goal is to surpass FTC by giving the NN an information advantage -- access to reference trajectory values and exit controller signals as inputs -- while pruning useless inputs to keep the search space lean.

## Approach

**Expand then Prune:** Add 7 new candidate inputs (4 reference trajectory, 3 exit-related) to the existing 16, producing a 23-input superset. Train a baseline network on all 23, run ablation analysis to rank input importance, prune to the top-K, retrain with the pruned set.

A TOML-configurable input mask selects which indices from the 23-element superset are fed to the network. Pruning is a config change, not a code change.

## The 23-Input Superset

All 23 inputs are always computed in Rust. The input mask selects which ones reach the network.

### Existing inputs (indices 0-15)

| Idx | Name | Normalization |
|-----|------|---------------|
| 0 | eccentricity_excess | `ecc - 1.0` |
| 1 | inclination_error | `(inc - target) * 3/5 deg` |
| 2 | radial_velocity | `2*(vr/1e3 + 1.2)/1.5 - 1` |
| 3 | orbital_energy | `-mu/(2*sma) / 6e6` |
| 4 | velocity | `(v/3e3 - 1.5) * 2` |
| 5 | accel_magnitude | `mag/20 - 1` |
| 6 | heat_flux_fraction | `frac*2 - 1` |
| 7 | heat_load_fraction | `frac*2 - 1` |
| 8 | altitude | `(alt_km - 65) / 65` |
| 9 | fpa | `fpa / 0.3` |
| 10 | latitude | `lat / (pi/2)` |
| 11 | drag_accel | `drag/50 - 1` |
| 12 | lift_accel | `lift / 10` |
| 13 | sma_error | `err / 5e5` |
| 14 | apoapsis_alt | `clamp(-10e6,10e6)/1e6 - 1` |
| 15 | bounce_flag | `flag*2 - 1` |

### New reference trajectory inputs (indices 16-19)

Interpolated from the piecewise-constant-derived reference trajectory at the current energy level.

| Idx | Name | Normalization | Notes |
|-----|------|---------------|-------|
| 16 | cos_bank_nominal | raw value (already in [-1, 1]) | What the classical controller would command |
| 17 | pdyn_nominal | `pdyn_nom / 2e3 - 1` | Reference dynamic pressure (~0-4 kPa) |
| 18 | hdot_nominal | `hdot_nom / 500` | Reference altitude rate (~-1000 to +500 m/s) |
| 19 | pdyn_error | `(pdyn_current - pdyn_nom) / 2e3` | Pre-computed tracking error |

### New exit-related inputs (indices 20-22)

All gated by `bounce_flag`: multiplied by `bounce_flag` (0 or 1) so they are exactly zero pre-bounce, avoiding garbage values.

| Idx | Name | Normalization | Notes |
|-----|------|---------------|-------|
| 20 | exit_bank_angle | `(exit_bank / pi * 2.0 - 1.0) * bounce_flag` | What exit controller would command, scaled to [-1,1]; zero pre-bounce |
| 21 | density_exit | `(log10(max(density_exit, 1e-12)) + 7.0) / 5.0 * bounce_flag` | Log-scaled (~1e-7 to 1e-2 range mapped to ~0-2); zero pre-bounce |
| 22 | ref_velocity_latched | `ref_vel / 500.0 * bounce_flag` | Latched radial velocity (~-200 to +200 m/s); zero pre-bounce |

**Note:** `guidance_phase` was considered as a candidate but dropped -- it's redundant with `bounce_flag` (index 15) since phase is always 1 pre-bounce and 2 post-bounce. Total: 23 inputs, not 24.

## Input Mask

### TOML configuration

```toml
[network]
layer_sizes = [11, 24, 2]
activations = ["tanh", "asinh"]
input_mask = [0, 1, 2, 4, 8, 9, 14, 15, 16, 19, 20]
```

- `input_mask`: list of indices into the 23-element superset
- `layer_sizes[0]` must equal `len(input_mask)` -- validated at config load time with an explicit error
- When `input_mask` is absent: defaults to `[0, 1, ..., 15]` for backward compatibility with existing 16-input models
- Validation: all indices in `0..23`, no duplicates

### Rust behavior

`nn_bank_angle()` always computes the full 23-element array, then applies the mask:

```
full_input: [f64; 23] = [ ... all 23 values ... ]
masked_input: Vec<f64> = input_mask.iter().map(|&i| full_input[i]).collect()
nn.forward(&masked_input)
```

## Ablation Analysis

### Method

Ablation on a trained 23-input network. For each input index, force it to zero during simulation and measure cost degradation.

### TOML support

```toml
[network]
ablated_input = 5  # force input index 5 to zero after construction
```

- Optional field on `NeuralNetModel`
- When set, the specified index in the full 23-element array is replaced with 0.0 before mask application
- Only used during ablation analysis, never during training or production

### Ablation script

New module: `src/python/aerocapture/training/ablation.py`

CLI: `python -m aerocapture.training.ablation <training_dir> --toml <config.toml> --n-sims 1000`

Procedure:
1. Load best model from `<training_dir>/best_model.json`
2. Run baseline MC (no ablation) -- record mean cost
3. For each index `i` in 0..22 (inclusive):
   - Set `ablated_input = i` via TOML override
   - Run N MC sims via `aerocapture_rs.run_mc()`
   - Record mean cost
4. Compute delta (ablated cost - baseline cost) per index
5. Rank by delta magnitude (high delta = important input)
6. Output:
   - Table to stdout (index, name, baseline cost, ablated cost, delta, rank)
   - JSON results file: `<training_dir>/ablation_results.json`
   - Bar chart SVG: `<training_dir>/ablation_chart.svg`

## Rust Changes

### `neural.rs`

- Expand input vector from 16 to 23 elements
- New inputs require access to `SimData` (ref trajectory interpolation) and additional nav/guidance state
- Function signature: `nn_bank_angle(nav, nn, data, planet, target_inclination, ref_velocity_latched)`
- Apply `input_mask` after constructing full 23-element array
- Apply `ablated_input` (zero override) before mask application
- Exit-related inputs (indices 20-23) multiplied by `bounce_flag` (the raw integer 0/1, before normalization)
- Speculative call to `exit::exit_guidance()` for index 21 (stateless, no side effects, cheap)

### `data/neural.rs` (NeuralNetModel)

- Add field: `input_mask: Option<Vec<usize>>`
- Add field: `ablated_input: Option<usize>`
- Validation at load time:
  - If `input_mask` is `Some`: `mask.len() == layer_sizes[0]`, all indices in `0..23`, no duplicates
  - If `ablated_input` is `Some`: index in `0..23`
- Default `input_mask`: `None` -> treated as `[0, 1, ..., 15]`

### `config.rs`

- Parse `input_mask` and `ablated_input` from `[network]` TOML section
- Wire through to `NeuralNetModel` construction

### `dispatch.rs`

- NN branch: pass `data` and `state.reference_velocity` to `nn_bank_angle()`
- Compute speculative `exit_guidance()` result for NN input (only when NN scheme is active)

## Python Changes

### `ablation.py` (new)

Standalone ablation analysis script as described above.

### No changes needed

- `param_spaces.py`: NN params are generated dynamically from architecture, `input_mask` is config-only
- `encoding.py`: `nn_param_specs_from_architecture()` already uses `layer_sizes[0]`, which reflects masked input count
- `evaluate.py`: NN training writes JSON model + TOML overrides, mask flows through TOML

## TOML Config Changes

NN training TOML gains:

```toml
[network]
layer_sizes = [23, 48, 2]
activations = ["tanh", "asinh"]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
```

After ablation pruning, updated to e.g.:

```toml
[network]
layer_sizes = [13, 28, 2]
activations = ["tanh", "asinh"]
input_mask = [0, 1, 2, 4, 6, 7, 8, 9, 14, 15, 16, 19, 20]
```

## Backward Compatibility

- Existing 16-input models: `input_mask` absent defaults to `[0..16]`. The 7 new inputs are computed but never used. No behavior change.
- Existing TOML configs without `[network] input_mask`: work unchanged.
- `ablated_input` absent: no ablation, normal operation.

## Workflow

1. Update NN training TOML: `input_mask = [0..22]`, `layer_sizes = [23, 48, 2]`
2. Train baseline 23-input network
3. Run ablation: `python -m aerocapture.training.ablation training_output/neural_network/ --toml <config> --n-sims 1000`
4. Review ranking, pick top-K inputs
5. Update TOML: `input_mask = [selected indices]`, `layer_sizes = [K, hidden, 2]`
6. Retrain with pruned inputs
7. Compare against FTC on identical MC scenarios

## Tests

### Rust

- Mask validation: correct length vs `layer_sizes[0]`, out-of-range index, duplicate index
- 23-input forward pass: verify all 23 elements computed and finite
- Mask application: verify masked vector has correct length and values
- Ablation zeroing: verify target index is zero, others unchanged
- Backward compat: no mask -> 16-input behavior identical to current
- Bounce gating: verify exit inputs are exactly zero when `bounce_flag = 0`
- Speculative exit guidance: verify it produces finite output without side effects

### Python

- Ablation script: output JSON structure, ranking correctness on synthetic data
- SVG chart generation
