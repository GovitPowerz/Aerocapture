# NN Weight Initialization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add activation-aware Xavier/He/LeCun uniform weight initialization to the GA's initial NN population, replacing random `[-3, 3]` generation.

**Architecture:** New `initialization.py` module with pure functions for bound computation and weight generation. These feed into the existing `encode_weights_to_chromosome()` in `population.py`. Weight stats logging added to `logger.py` / `train.py` for future adaptive-bounds instrumentation.

**Tech Stack:** Python 3.14, numpy, hypothesis (property-based tests), pytest

**Spec:** `docs/superpowers/specs/2026-03-12-nn-weight-initialization-design.md`

---

## Chunk 1: Core initialization functions

### Task 1: `compute_layer_bound()` — tests and implementation

**Files:**
- Create: `tests/test_initialization.py`
- Create: `src/python/aerocapture/training/initialization.py`

- [ ] **Step 1: Write failing tests for `compute_layer_bound`**

```python
# tests/test_initialization.py
"""Tests for NN weight initialization functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aerocapture.training.initialization import compute_layer_bound, generate_initialized_weights


class TestComputeLayerBound:
    def test_xavier_tanh(self) -> None:
        assert compute_layer_bound(6, 12, "tanh") == pytest.approx(math.sqrt(6 / 18))

    def test_xavier_sigmoid(self) -> None:
        assert compute_layer_bound(6, 12, "sigmoid") == pytest.approx(math.sqrt(6 / 18))

    def test_xavier_asinh(self) -> None:
        assert compute_layer_bound(12, 2, "asinh") == pytest.approx(math.sqrt(6 / 14))

    def test_he_relu(self) -> None:
        assert compute_layer_bound(6, 64, "relu") == pytest.approx(math.sqrt(6 / 6))

    def test_lecun_linear(self) -> None:
        assert compute_layer_bound(32, 2, "linear") == pytest.approx(math.sqrt(3 / 32))

    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            compute_layer_bound(6, 12, "swish")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_initialization.py::TestComputeLayerBound -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aerocapture.training.initialization'`

- [ ] **Step 3: Implement `compute_layer_bound`**

```python
# src/python/aerocapture/training/initialization.py
"""Activation-aware weight initialization for GA populations.

Provides Xavier (Glorot), He (Kaiming), and LeCun uniform initialization
bounds, auto-selected by activation function. Generates flat weight vectors
compatible with the GA binary chromosome encoding.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

# Activation -> scheme mapping
_XAVIER_ACTIVATIONS = frozenset({"tanh", "sigmoid", "asinh"})
_HE_ACTIVATIONS = frozenset({"relu"})
_LECUN_ACTIVATIONS = frozenset({"linear"})


def compute_layer_bound(fan_in: int, fan_out: int, activation: str) -> float:
    """Compute uniform initialization bound for a single layer.

    Auto-selects scheme based on activation:
        tanh/sigmoid/asinh -> Xavier: sqrt(6 / (fan_in + fan_out))
        relu               -> He:     sqrt(6 / fan_in)
        linear             -> LeCun:  sqrt(3 / fan_in)

    Args:
        fan_in: Number of input neurons.
        fan_out: Number of output neurons.
        activation: Activation function name.

    Returns:
        Uniform bound: weights should be drawn from U(-bound, +bound).
    """
    if activation in _XAVIER_ACTIVATIONS:
        return math.sqrt(6.0 / (fan_in + fan_out))
    if activation in _HE_ACTIVATIONS:
        return math.sqrt(6.0 / fan_in)
    if activation in _LECUN_ACTIVATIONS:
        return math.sqrt(3.0 / fan_in)
    msg = f"Unknown activation: {activation!r}. Expected one of: tanh, sigmoid, asinh, relu, linear"
    raise ValueError(msg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_initialization.py::TestComputeLayerBound -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/initialization.py tests/test_initialization.py
git commit -m "feat(training): add compute_layer_bound with Xavier/He/LeCun auto-selection"
```

---

### Task 2: `generate_initialized_weights()` — tests and implementation

**Files:**
- Modify: `tests/test_initialization.py`
- Modify: `src/python/aerocapture/training/initialization.py`

- [ ] **Step 1: Write failing tests for `generate_initialized_weights`**

Append to `tests/test_initialization.py`:

```python
class TestGenerateInitializedWeights:
    def test_shape_default_arch(self) -> None:
        """Output length matches n_base_coef for [6, 12, 2]."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], rng)
        # 6*12 + 12 + 12*2 + 2 = 110
        assert len(weights) == 110

    def test_shape_deep_arch(self) -> None:
        """Output length matches n_base_coef for [6, 64, 32, 2]."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 64, 32, 2], ["relu", "tanh", "asinh"], rng)
        # 6*64 + 64 + 64*32 + 32 + 32*2 + 2 = 384 + 64 + 2048 + 32 + 64 + 2 = 2594
        assert len(weights) == 2594

    def test_weights_within_xavier_bounds(self) -> None:
        """Layer 0 weights (tanh) fall within Xavier limits."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], rng)
        limit = math.sqrt(6 / 18)
        # Layer 0 weights: first 6*12 = 72 values
        layer0_w = weights[:72]
        assert np.all(np.abs(layer0_w) <= limit + 1e-15)

    def test_weights_within_he_bounds(self) -> None:
        """Layer 0 weights (relu) fall within He limits."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 64, 32, 2], ["relu", "tanh", "asinh"], rng)
        limit = math.sqrt(6 / 6)
        layer0_w = weights[: 6 * 64]
        assert np.all(np.abs(layer0_w) <= limit + 1e-15)

    def test_biases_are_zero(self) -> None:
        """All biases initialized to zero."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], rng)
        # Layer 0 biases: indices 72..83 (12 biases)
        assert np.all(weights[72:84] == 0.0)
        # Layer 1 biases: indices 108..109 (2 biases)
        assert np.all(weights[108:110] == 0.0)

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical weights."""
        w1 = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], np.random.default_rng(99))
        w2 = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], np.random.default_rng(99))
        np.testing.assert_array_equal(w1, w2)

    @given(data=st.data())
    @settings(max_examples=20)
    def test_property_weights_respect_bounds(self, data: st.DataObject) -> None:
        """For random architectures, all weights respect per-layer bounds."""
        n_layers = data.draw(st.integers(2, 5))
        layer_sizes = [data.draw(st.integers(2, 32)) for _ in range(n_layers)]
        activations_pool = ["tanh", "sigmoid", "asinh", "relu", "linear"]
        activations = [data.draw(st.sampled_from(activations_pool)) for _ in range(n_layers - 1)]
        rng = np.random.default_rng(42)

        weights = generate_initialized_weights(layer_sizes, activations, rng)

        # Check shape
        expected_len = sum(layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1] for i in range(n_layers - 1))
        assert len(weights) == expected_len

        # Check per-layer bounds
        idx = 0
        for i in range(n_layers - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = compute_layer_bound(fan_in, fan_out, activations[i])
            n_weights = fan_in * fan_out
            layer_w = weights[idx : idx + n_weights]
            assert np.all(np.abs(layer_w) <= limit + 1e-15), f"Layer {i} weights exceed bound {limit}"
            idx += n_weights
            # Biases should be zero
            layer_b = weights[idx : idx + fan_out]
            assert np.all(layer_b == 0.0), f"Layer {i} biases not zero"
            idx += fan_out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_initialization.py::TestGenerateInitializedWeights -v`
Expected: FAIL — `ImportError: cannot import name 'generate_initialized_weights'`

- [ ] **Step 3: Implement `generate_initialized_weights`**

Append to `src/python/aerocapture/training/initialization.py`:

```python
def generate_initialized_weights(
    layer_sizes: list[int],
    activations: list[str],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Generate a flat weight vector with per-layer initialization.

    Weight layout matches write_nn_json() / to_flat_weights():
    for each layer: weights (row-major, shape fan_out x fan_in) then biases.

    Args:
        layer_sizes: Network layer sizes, e.g. [6, 12, 2].
        activations: Activation per layer, length = len(layer_sizes) - 1.
        rng: Numpy random generator.

    Returns:
        Flat float64 array of all weights and biases.
    """
    parts: list[npt.NDArray[np.float64]] = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        limit = compute_layer_bound(fan_in, fan_out, activations[i])
        w = rng.uniform(-limit, limit, size=(fan_out, fan_in)).ravel()
        b = np.zeros(fan_out, dtype=np.float64)
        parts.append(w)
        parts.append(b)
    return np.concatenate(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_initialization.py -v`
Expected: All tests PASS (both `TestComputeLayerBound` and `TestGenerateInitializedWeights`)

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/initialization.py tests/test_initialization.py
git commit -m "feat(training): add generate_initialized_weights with per-layer bounds"
```

---

## Chunk 2: Encode roundtrip test and population integration

### Task 3: Encode/decode roundtrip test

**Files:**
- Modify: `tests/test_initialization.py`

- [ ] **Step 1: Write roundtrip test**

Append to `tests/test_initialization.py`:

```python
from aerocapture.training.evaluate import decode_direct
from aerocapture.training.population import encode_weights_to_chromosome

from tests.fixtures.factories import make_training_config


class TestEncodeDecodeRoundtrip:
    def test_initialized_weights_survive_roundtrip(self) -> None:
        """generate_initialized_weights -> encode -> decode ≈ original (within quantization)."""
        config = make_training_config("neural_network")
        rng = np.random.default_rng(42)
        original = generate_initialized_weights(
            config.network.layer_sizes, config.network.activations, rng
        )
        chrom = encode_weights_to_chromosome(original, config)
        decoded = decode_direct(chrom, config)
        # 16-bit quantization over [-3, 3]: max error = 6 / 65535 ≈ 9.15e-5
        np.testing.assert_allclose(decoded, original, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_initialization.py::TestEncodeDecodeRoundtrip -v`
Expected: PASS (uses existing functions — this is a pure integration test)

- [ ] **Step 3: Commit**

```bash
git add tests/test_initialization.py
git commit -m "test(training): add encode/decode roundtrip test for initialized weights"
```

---

### Task 4: Integrate smart init into `create_initial_population()`

**Files:**
- Modify: `src/python/aerocapture/training/population.py:77-114`
- Modify: `tests/test_initialization.py`

- [ ] **Step 1: Modify `create_initial_population()` to use smart init for NN**

In `src/python/aerocapture/training/population.py`, add import and replace the random chromosome generation for NN:

Add to imports:
```python
from aerocapture.training.initialization import generate_initialized_weights
```

Replace lines 113-114 (`candidates = rng.integers(...)`) with:
```python
    # Generate initial chromosomes
    if config.guidance_type == "neural_network" and config.ga.direct_encoding:
        # Smart initialization: per-layer Xavier/He/LeCun uniform
        candidates = np.zeros((n_candidates, chrom_len), dtype=np.int8)
        for i in range(n_candidates):
            weights = generate_initialized_weights(
                config.network.layer_sizes, config.network.activations, rng
            )
            candidates[i] = encode_weights_to_chromosome(weights, config)
    else:
        # Random binary chromosomes (non-NN schemes)
        candidates = rng.integers(0, 2, size=(n_candidates, chrom_len), dtype=np.int8)
```

The seeding block (lines 116-144) remains unchanged. When `seed_weights` is provided, slots 0..n_seeded are overwritten with seed + mutants; the remaining smart-initialized candidates stay.

- [ ] **Step 2: Write integration test that exercises the actual `create_initial_population` code path**

Append to `tests/test_initialization.py`:

```python
from unittest.mock import patch

from aerocapture.training.population import create_initial_population


class TestPopulationSmartInit:
    def test_nn_population_uses_smart_init(self) -> None:
        """create_initial_population for NN produces weights with std << 1.73 (uniform [-3,3])."""
        config = make_training_config("neural_network")
        config.ga.n_pop = 4
        rng = np.random.default_rng(42)

        # Mock evaluate_chromosome to avoid running the actual simulator
        with patch("aerocapture.training.population.evaluate_chromosome", return_value=(1.0, None)):
            with patch("aerocapture.training.population.improve_chromosome", return_value=(np.zeros(config.chrom_length, dtype=np.int8), 1.0, 0.0)):
                population, costs = create_initial_population(
                    config, np.zeros(config.network.n_coef), rng=rng, verbose=False
                )

        # Decode all chromosomes and check weight distribution
        all_decoded = np.array([decode_direct(chrom, config) for chrom in population])
        # Xavier init for [6,12,2] produces std ~0.3-0.4, far below uniform [-3,3] std ~1.73
        assert all_decoded.std() < 0.5, f"Expected std < 0.5 for Xavier init, got {all_decoded.std():.3f}"

    def test_non_nn_uses_random_init(self) -> None:
        """Non-NN guidance still uses random binary chromosomes."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_pop = 4
        rng = np.random.default_rng(42)

        with patch("aerocapture.training.population.evaluate_chromosome", return_value=(1.0, None)):
            with patch("aerocapture.training.population.improve_chromosome", return_value=(np.zeros(config.chrom_length, dtype=np.int8), 1.0, 0.0)):
                population, costs = create_initial_population(
                    config, np.zeros(config.network.n_coef), rng=rng, verbose=False
                )

        # Non-NN: random bits decode to values spread across [-3, 3]
        from aerocapture.training.evaluate import decode_params_from_chromosome
        # Just verify population was created with expected shape
        assert population.shape == (4, config.chrom_length)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_initialization.py::TestPopulationSmartInit tests/test_chromosome.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/population.py tests/test_initialization.py
git commit -m "feat(training): use Xavier/He/LeCun init for NN initial population"
```

---

## Chunk 3: Weight stats logging

### Task 5: Add `weight_stats` parameter to `log_generation()`

**Files:**
- Modify: `src/python/aerocapture/training/logger.py:40-82`
- Modify: `tests/test_training_logger.py`

- [ ] **Step 1: Write failing test for weight_stats in logger**

Add the following two methods **inside the existing `class TestTrainingLogger`** in `tests/test_training_logger.py`:

```python
    def test_weight_stats_recorded(self, logger: TrainingLogger) -> None:
        stats = {
            "layer_0_w": {"min": -0.38, "max": 0.41, "mean": 0.01, "std": 0.22},
            "layer_0_b": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
        }
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn, weight_stats=stats)
        assert logger.buffer[0]["weight_stats"] == stats
        logger.close()

    def test_weight_stats_none_by_default(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        assert "weight_stats" not in logger.buffer[0]
        logger.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_logger.py::TestTrainingLogger::test_weight_stats_recorded -v`
Expected: FAIL — `TypeError: log_generation() got an unexpected keyword argument 'weight_stats'`

- [ ] **Step 3: Add `weight_stats` parameter to `log_generation()`**

In `src/python/aerocapture/training/logger.py`, modify the `log_generation` signature (line 40) and body:

Change signature to:
```python
    def log_generation(
        self,
        generation: int,
        populations: list[npt.NDArray[np.int8]],
        costs: list[npt.NDArray[np.float64]],
        best_chromosome: npt.NDArray[np.int8],
        decode_fn: Callable[[npt.NDArray[np.int8]], dict[str, float]] | None,
        weight_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
```

After line 77 (`"config_hash": self._config_hash,`), add:
```python
        }
        if weight_stats is not None:
            record["weight_stats"] = weight_stats
```

(Remove the existing closing `}` that was on its own line and incorporate it above.)

- [ ] **Step 4: Run all logger tests to verify pass**

Run: `uv run pytest tests/test_training_logger.py -v`
Expected: All PASS (existing tests unaffected — `weight_stats` defaults to None)

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/logger.py tests/test_training_logger.py
git commit -m "feat(training): add weight_stats parameter to log_generation()"
```

---

### Task 6: Compute and pass weight stats in `train.py`

**Files:**
- Create: `src/python/aerocapture/training/weight_stats.py`
- Modify: `src/python/aerocapture/training/train.py:315-368`
- Modify: `tests/test_initialization.py`

- [ ] **Step 1: Write test for weight stats computation**

Append to `tests/test_initialization.py`:

```python
from aerocapture.training.weight_stats import compute_weight_stats


class TestComputeWeightStats:
    def test_stats_keys(self) -> None:
        """Returns per-layer weight and bias stats."""
        weights = np.zeros(110)
        stats = compute_weight_stats(weights, [6, 12, 2])
        assert "layer_0_w" in stats
        assert "layer_0_b" in stats
        assert "layer_1_w" in stats
        assert "layer_1_b" in stats

    def test_stats_values(self) -> None:
        """Stats are computed correctly for known values."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], rng)
        stats = compute_weight_stats(weights, [6, 12, 2])
        layer0_w = weights[:72]
        assert stats["layer_0_w"]["min"] == pytest.approx(float(layer0_w.min()))
        assert stats["layer_0_w"]["max"] == pytest.approx(float(layer0_w.max()))
        assert stats["layer_0_w"]["mean"] == pytest.approx(float(layer0_w.mean()))
        assert stats["layer_0_w"]["std"] == pytest.approx(float(layer0_w.std()))

    def test_zero_biases(self) -> None:
        """Zero biases produce zero stats."""
        weights = np.zeros(110)
        stats = compute_weight_stats(weights, [6, 12, 2])
        assert stats["layer_0_b"]["std"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_initialization.py::TestComputeWeightStats -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aerocapture.training.weight_stats'`

- [ ] **Step 3: Implement `compute_weight_stats`**

```python
# src/python/aerocapture/training/weight_stats.py
"""Per-layer weight statistics for GA training instrumentation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_weight_stats(
    weights: npt.NDArray[np.float64],
    layer_sizes: list[int],
) -> dict[str, dict[str, float]]:
    """Compute per-layer min/max/mean/std for weights and biases.

    Args:
        weights: Flat weight vector (same layout as write_nn_json / to_flat_weights).
        layer_sizes: Network layer sizes, e.g. [6, 12, 2].

    Returns:
        Dict with keys like "layer_0_w", "layer_0_b", each mapping to
        {"min": ..., "max": ..., "mean": ..., "std": ...}.
    """
    stats: dict[str, dict[str, float]] = {}
    idx = 0
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]

        n_w = fan_in * fan_out
        w = weights[idx : idx + n_w]
        idx += n_w
        stats[f"layer_{i}_w"] = {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "std": float(w.std()),
        }

        b = weights[idx : idx + fan_out]
        idx += fan_out
        stats[f"layer_{i}_b"] = {
            "min": float(b.min()),
            "max": float(b.max()),
            "mean": float(b.mean()),
            "std": float(b.std()),
        }

    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_initialization.py::TestComputeWeightStats -v`
Expected: All 3 PASS

- [ ] **Step 5: Wire weight stats into `train.py`'s GA loop**

In `src/python/aerocapture/training/train.py`:

Add import at top:
```python
from aerocapture.training.weight_stats import compute_weight_stats
```

Modify the `log_generation` call site (around line 362). Replace:
```python
                logger.log_generation(
                    gen + 1,
                    populations,
                    all_costs,
                    best_overall_chrom if best_overall_chrom is not None else populations[0][0],
                    decode_fn,
                )
```

With:
```python
                # Compute per-layer weight stats for NN (instrumentation for future adaptive bounds)
                ws = None
                if config.guidance_type == "neural_network" and best_overall_chrom is not None:
                    best_weights = decode_direct(best_overall_chrom, config)
                    ws = compute_weight_stats(best_weights, config.network.layer_sizes)

                logger.log_generation(
                    gen + 1,
                    populations,
                    all_costs,
                    best_overall_chrom if best_overall_chrom is not None else populations[0][0],
                    decode_fn,
                    weight_stats=ws,
                )
```

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/weight_stats.py src/python/aerocapture/training/train.py tests/test_initialization.py
git commit -m "feat(training): log per-layer weight stats for NN training runs"
```

---

## Chunk 4: Lint, type-check, final verification

### Task 7: Lint and type-check

**Files:** All modified/created files

- [ ] **Step 1: Run ruff and mypy**

Run: `./lint_code.sh`
Expected: No errors. Fix any issues that arise (unused imports, formatting, type annotations).

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -u
git commit -m "style: fix lint/type issues in initialization module"
```

(Skip if no fixes needed.)

---

### Task 8: Final integration verification

- [ ] **Step 1: Verify the complete file list matches the spec**

Files created:
- `src/python/aerocapture/training/initialization.py`
- `src/python/aerocapture/training/weight_stats.py`
- `tests/test_initialization.py`

Files modified:
- `src/python/aerocapture/training/population.py` (smart init for NN)
- `src/python/aerocapture/training/logger.py` (weight_stats param)
- `src/python/aerocapture/training/train.py` (compute + pass weight_stats)
- `tests/test_training_logger.py` (weight_stats tests)

Files NOT modified (per spec):
- No Rust code changes
- No Rich TUI changes
- No Plotly report changes
- GA bounds, chromosome encoding, crossover/mutation unchanged

- [ ] **Step 2: Run Rust tests to confirm no impact**

Run: `cd src/rust && cargo test`
Expected: All Rust tests PASS (no Rust changes made)
