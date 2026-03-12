# Neural Network Weight Initialization for GA Training

**Date**: 2026-03-12
**Status**: Approved

## Problem

The GA training pipeline initializes NN chromosomes as random binary strings that decode to weights uniformly distributed in `[-3.0, 3.0]`. This is suboptimal because:

- **Poor solution quality**: random large weights saturate `tanh` neurons and produce degenerate bank angle outputs, trapping the GA in local optima.
- **Hostile to larger architectures**: for deeper/wider nets (e.g., `[6, 64, 32, 2]`), Xavier bounds shrink to ~`[-0.18, 0.18]` — 97% of `[-3, 3]` is noise.
- **Theoretical unsoundness**: weight initialization literature (Glorot & Bengio 2010, He et al. 2015) establishes that variance should scale with layer fan-in/fan-out to preserve signal magnitude through the network.

## Solution

Activation-aware uniform weight initialization (Xavier/He/LeCun) applied to the GA's initial population. Weights are generated in float-space with proper per-layer distributions, then encoded into binary chromosomes.

### Design Decisions

- **Approach**: generate real-valued weights with proper distributions, encode to binary chromosomes (Approach 2). Avoids bit-space gymnastics while producing an exact inverse of `decode_direct()`.
- **Scheme auto-selection by activation**: `tanh`/`sigmoid`/`asinh` -> Xavier, `relu` -> He, `linear` -> LeCun. No manual configuration needed.
- **Uniform variant**: fits naturally with the bounded binary chromosome encoding. No clipping artifacts.
- **GA bounds unchanged**: `[-3.0, 3.0]` stays. Evolution is unconstrained; only initialization is smarter. Per-layer weight stats are logged to instrument for potential future adaptive bounds (data-driven, not speculative).

## Components

### 1. New module: `src/python/aerocapture/training/initialization.py`

Two public functions:

**`compute_layer_bound(fan_in: int, fan_out: int, activation: str) -> float`**

Returns the uniform limit for one layer based on the activation function:

| Activation(s)             | Scheme  | Limit formula                    |
|---------------------------|---------|----------------------------------|
| `tanh`, `sigmoid`, `asinh`| Xavier  | `sqrt(6 / (fan_in + fan_out))`   |
| `relu`                    | He      | `sqrt(6 / fan_in)`               |
| `linear`                  | LeCun   | `sqrt(3 / fan_in)`               |

**`generate_initialized_weights(layer_sizes: list[int], activations: list[str], rng: np.random.Generator) -> np.ndarray`**

- Iterates over layers, computes per-layer bounds via `compute_layer_bound()`
- Generates weights: `rng.uniform(-limit, limit, size=(fan_out, fan_in))`
- Biases: initialized to zero (standard practice)
- Returns flat weight vector in same order as `to_flat_weights()` / `decode_direct()`: all weights (row-major) then all biases, per layer

### 2. New function in `evaluate.py`: `encode_weights()`

**`encode_weights(weights: np.ndarray, config: TrainingConfig) -> np.ndarray`**

Inverse of `decode_direct()`. Converts a flat float weight vector to a binary chromosome (int8 array):

1. Clamp each weight to `[p_min, p_max]` (safety guard)
2. `int_value = round((weight - p_min) / (p_max - p_min) * (2^n_bit - 1))`
3. Convert each `int_value` to `n_bit` binary digits
4. Concatenate into flat chromosome array

### 3. Modified `create_initial_population()` in `population.py`

Current flow generates random binary chromosomes. New flow:

- **NN guidance**: each candidate is produced by `generate_initialized_weights()` -> `encode_weights()`. Every initial chromosome decodes to weights following proper per-layer distributions.
- **Non-NN guidance**: unchanged (random binary chromosomes). Branch on `config.guidance == "neural_network"`.
- **Seeding**: unchanged. When `seed_weights` is provided (from existing JSON), it overrides candidate[0] and generates mutants. A known-good model takes priority over smart initialization.

### 4. Weight stats logging

Per-generation logging of elite individual's per-layer weight statistics, appended to the existing JSONL log line:

```json
{
  "weight_stats": {
    "layer_0_w": {"min": -0.38, "max": 0.41, "std": 0.22},
    "layer_0_b": {"min": -0.01, "max": 0.03, "std": 0.01},
    "layer_1_w": {"min": -0.55, "max": 0.48, "std": 0.31},
    "layer_1_b": {"min": 0.00, "max": 0.00, "std": 0.00}
  }
}
```

Computed in the training loop (`train.py`) after evaluating each generation: decode best individual's weights, partition by layer, compute min/max/mean/std. Passed to `logger.log_generation()`.

Purpose: instrument for potential future adaptive bounds. If elite weights consistently drift to `[-3, 3]` edges, that's the signal to implement per-layer bound adaptation.

### 5. Tests

**New `tests/test_initialization.py`**:

- `test_compute_layer_bound_xavier` — tanh/sigmoid/asinh produce `sqrt(6/(fan_in+fan_out))`
- `test_compute_layer_bound_he` — relu produces `sqrt(6/fan_in)`
- `test_compute_layer_bound_lecun` — linear produces `sqrt(3/fan_in)`
- `test_encode_decode_roundtrip` — `encode_weights(decode_direct(chrom))` matches original within 1-bit quantization
- `test_generate_initialized_weights_shape` — output length matches `n_base_coef` for given architecture
- `test_generate_initialized_weights_bounds` — all weights per layer fall within `[-limit, +limit]`
- `test_generate_initialized_weights_biases_zero` — biases initialized to zero
- Hypothesis property test: for random architectures and activations, generated weights always respect per-layer bounds

**Modified/new test for population**:

- `test_initial_population_nn_uses_smart_init` — for `neural_network` guidance, initial population weights have std significantly smaller than uniform `[-3, 3]` std (~1.73)

## What's NOT Changing

- GA bounds (`p_min=-3.0`, `p_max=3.0`)
- Chromosome encoding (16-bit binary, direct encoding)
- Crossover and mutation operators
- Seeding and resume logic
- Rust simulator code
- Rich TUI dashboard
- Plotly HTML reports

## Concrete Example

For the default architecture `[6, 12, 2]` with activations `["tanh", "asinh"]`:

- **Layer 0** (6->12, tanh): Xavier limit = `sqrt(6/18)` = 0.577. Weights in `[-0.577, 0.577]`.
- **Layer 1** (12->2, asinh): Xavier limit = `sqrt(6/14)` = 0.655. Weights in `[-0.655, 0.655]`.
- **All biases**: 0.0

Compare to current: all 110 parameters uniformly in `[-3.0, 3.0]`.

For a hypothetical `[6, 64, 32, 2]` with `["relu", "tanh", "asinh"]`:

- **Layer 0** (6->64, relu): He limit = `sqrt(6/6)` = 1.0. Weights in `[-1.0, 1.0]`.
- **Layer 1** (64->32, tanh): Xavier limit = `sqrt(6/96)` = 0.25. Weights in `[-0.25, 0.25]`.
- **Layer 2** (32->2, asinh): Xavier limit = `sqrt(6/34)` = 0.42. Weights in `[-0.42, 0.42]`.
