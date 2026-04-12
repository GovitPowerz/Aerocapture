# pymoo Optimizer Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binary GA with a real-valued optimization framework using pymoo, supporting GA (SBX + polynomial mutation), CMA-ES, DE, and PSO algorithms via a hybrid training loop.

**Architecture:** pymoo `Algorithm` objects stepped manually via `algorithm.next()` in a custom outer loop. An `AerocaptureProblem(pymoo.Problem)` subclass encapsulates parameter decoding and batch evaluation via PyO3. All algorithms work on normalized [0, 1] float arrays, decoded to physical values at evaluation time.

**Tech Stack:** pymoo, numpy, aerocapture_rs (PyO3), existing training infrastructure (logger, display, seed_pool, corridor, report)

**Spec:** `docs/superpowers/specs/2026-04-12-pymoo-optimizer-framework-design.md`

---

## File Structure

### New files
- `src/python/aerocapture/training/optimizer.py` -- Algorithm factory + OptimizerConfig dataclass
- `src/python/aerocapture/training/problem.py` -- AerocaptureProblem(pymoo.Problem) subclass
- `src/python/aerocapture/training/encoding.py` -- Real-valued encoding/decoding (normalize, denormalize, decode to named params)
- `tests/test_encoding.py` -- Encoding roundtrip + boundary tests
- `tests/test_optimizer.py` -- Algorithm factory tests
- `tests/test_problem.py` -- Problem subclass evaluation tests
- `tests/test_hybrid_loop.py` -- Integration test for the new training loop

### Modified files
- `src/python/aerocapture/training/config.py` -- Replace GAConfig with OptimizerConfig
- `src/python/aerocapture/training/train.py` -- Replace generation loop with hybrid pymoo loop
- `src/python/aerocapture/training/evaluate.py` -- Remove binary encoding functions, keep cost + TOML patching
- `src/python/aerocapture/training/population.py` -- Rewrite for real-valued initialization
- `src/python/aerocapture/training/param_spaces.py` -- Minor: remove any n_bit references
- `src/python/aerocapture/training/seed_pool.py` -- Update evaluate_population() signature for real-valued arrays
- `src/python/aerocapture/training/logger.py` -- Update log_generation() signature for real-valued arrays
- `pyproject.toml` -- Add pymoo, remove deap
- `IMPROVEMENTS.md` -- Add Bayesian optimization and RL sections
- `configs/training/*.toml` -- Replace [ga] with [optimizer] in all training TOMLs

### Deleted files
- `src/python/aerocapture/training/local_search.py`
- `src/python/aerocapture/training/migration.py`
- `tests/test_ga_operators.py`

---

## Task 1: Add pymoo dependency, remove deap

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

In `pyproject.toml`, replace the `deap>=1.4` dependency with `pymoo>=0.6`:

```toml
dependencies = [
    "numpy>=2.4",
    "pandas>=3.0",
    "matplotlib>=3.10",
    "pymoo>=0.6",
    "scipy>=1.17.1",
    "rich>=14.3",
    "seaborn>=0.13",
    "SALib>=1.5",
    "pyarrow>=19.0",
]
```

- [ ] **Step 2: Install and verify**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv sync
```

Then verify pymoo is importable:
```bash
uv run python -c "import pymoo; print(pymoo.__version__)"
```

Expected: prints pymoo version (0.6.x)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: replace deap with pymoo for real-valued optimization"
```

---

## Task 2: Real-valued encoding module

**Files:**
- Create: `src/python/aerocapture/training/encoding.py`
- Test: `tests/test_encoding.py`

- [ ] **Step 1: Write failing tests for encoding**

Create `tests/test_encoding.py`:

```python
"""Tests for real-valued encoding/decoding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aerocapture.training.encoding import decode_normalized, encode_to_normalized, nn_param_specs_from_architecture
from aerocapture.training.param_spaces import ParamSpec


class TestLinearRoundtrip:
    """Normalized [0,1] <-> physical value roundtrip for linear params."""

    def test_midpoint(self):
        specs = [ParamSpec("x", 10.0, 20.0, 15.0)]
        physical = decode_normalized(np.array([0.5]), specs)
        assert physical["x"] == pytest.approx(15.0)

    def test_boundaries(self):
        specs = [ParamSpec("x", -5.0, 5.0, 0.0)]
        lo = decode_normalized(np.array([0.0]), specs)
        hi = decode_normalized(np.array([1.0]), specs)
        assert lo["x"] == pytest.approx(-5.0)
        assert hi["x"] == pytest.approx(5.0)

    def test_roundtrip(self):
        specs = [ParamSpec("a", 1.0, 100.0, 50.0), ParamSpec("b", -10.0, 10.0, 0.0)]
        original = {"a": 73.5, "b": -3.2}
        normalized = encode_to_normalized(original, specs)
        recovered = decode_normalized(normalized, specs)
        assert recovered["a"] == pytest.approx(73.5)
        assert recovered["b"] == pytest.approx(-3.2)


class TestLogScaleRoundtrip:
    """Normalized [0,1] <-> physical value roundtrip for log-scale params."""

    def test_midpoint_log(self):
        specs = [ParamSpec("g", 1e-8, 1e-5, 1e-6, log_scale=True)]
        physical = decode_normalized(np.array([0.5]), specs)
        # Midpoint in log10 space: 10^((-8 + -5) / 2) = 10^-6.5
        assert physical["g"] == pytest.approx(10**-6.5)

    def test_boundaries_log(self):
        specs = [ParamSpec("g", 1e-8, 1e-5, 1e-6, log_scale=True)]
        lo = decode_normalized(np.array([0.0]), specs)
        hi = decode_normalized(np.array([1.0]), specs)
        assert lo["g"] == pytest.approx(1e-8)
        assert hi["g"] == pytest.approx(1e-5)

    def test_roundtrip_log(self):
        specs = [ParamSpec("g", 1e-8, 1e-5, 1e-6, log_scale=True)]
        original = {"g": 3.7e-7}
        normalized = encode_to_normalized(original, specs)
        recovered = decode_normalized(normalized, specs)
        assert recovered["g"] == pytest.approx(3.7e-7, rel=1e-10)


class TestMixedParams:
    """Mixed linear + log-scale parameter vectors."""

    def test_multi_param_decode(self):
        specs = [
            ParamSpec("tau", 2.0, 60.0, 30.0),
            ParamSpec("gain", 1e-8, 1e-5, 1e-6, log_scale=True),
            ParamSpec("angle", -180.0, 180.0, 0.0),
        ]
        x = np.array([0.0, 1.0, 0.5])
        result = decode_normalized(x, specs)
        assert result["tau"] == pytest.approx(2.0)
        assert result["gain"] == pytest.approx(1e-5)
        assert result["angle"] == pytest.approx(0.0)

    def test_encode_defaults(self):
        specs = [
            ParamSpec("tau", 2.0, 60.0, 30.0),
            ParamSpec("gain", 1e-8, 1e-5, 1e-6, log_scale=True),
        ]
        defaults = {s.name: s.default for s in specs}
        normalized = encode_to_normalized(defaults, specs)
        assert 0.0 <= normalized[0] <= 1.0
        assert 0.0 <= normalized[1] <= 1.0
        recovered = decode_normalized(normalized, specs)
        assert recovered["tau"] == pytest.approx(30.0)
        assert recovered["gain"] == pytest.approx(1e-6, rel=1e-10)


class TestNNParamSpecs:
    """NN weight bound computation from architecture."""

    def test_layer_count(self):
        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=2.0)
        # Layer 0: 16*24 weights + 24 biases = 408
        # Layer 1: 24*2 weights + 2 biases = 50
        assert len(specs) == 458

    def test_weight_bounds_symmetric(self):
        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=2.0)
        for s in specs:
            assert s.p_min == pytest.approx(-s.p_max)
            assert s.p_max > 0.0

    def test_xavier_bound_layer0(self):
        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=1.0)
        # Xavier for tanh: sqrt(6 / (16 + 24)) = sqrt(6/40)
        expected_bound = math.sqrt(6.0 / 40.0)
        # First spec is a weight for layer 0
        assert specs[0].p_max == pytest.approx(expected_bound)

    def test_bias_bounds(self):
        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=2.0)
        # Biases for layer 0 are at indices 384..407 (after 16*24=384 weights)
        bias_spec = specs[384]
        assert "bias" in bias_spec.name
        # Bias bound = multiplier * xavier_bound
        expected = 2.0 * math.sqrt(6.0 / 40.0)
        assert bias_spec.p_max == pytest.approx(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_encoding.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'aerocapture.training.encoding'`

- [ ] **Step 3: Implement encoding module**

Create `src/python/aerocapture/training/encoding.py`:

```python
"""Real-valued encoding/decoding for optimizer parameters.

All algorithms work on normalized np.ndarray[float64] in [0, 1].
Decoding to physical values happens at evaluation time.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from aerocapture.training.initialization import compute_layer_bound
from aerocapture.training.param_spaces import ParamSpec


def decode_normalized(x: npt.NDArray[np.float64], specs: list[ParamSpec]) -> dict[str, float]:
    """Decode a normalized [0,1] vector to physical parameter values.

    Linear params:    value = p_min + x * (p_max - p_min)
    Log-scale params: value = 10^(log10(p_min) + x * (log10(p_max) - log10(p_min)))
    """
    result: dict[str, float] = {}
    for i, s in enumerate(specs):
        xi = float(x[i])
        if s.log_scale:
            log_min = math.log10(s.p_min)
            log_max = math.log10(s.p_max)
            result[s.name] = 10.0 ** (log_min + xi * (log_max - log_min))
        else:
            result[s.name] = s.p_min + xi * (s.p_max - s.p_min)
    return result


def encode_to_normalized(params: dict[str, float], specs: list[ParamSpec]) -> npt.NDArray[np.float64]:
    """Encode physical parameter values to normalized [0,1] vector."""
    x = np.empty(len(specs), dtype=np.float64)
    for i, s in enumerate(specs):
        v = params[s.name]
        if s.log_scale:
            log_min = math.log10(s.p_min)
            log_max = math.log10(s.p_max)
            x[i] = (math.log10(v) - log_min) / (log_max - log_min)
        else:
            x[i] = (v - s.p_min) / (s.p_max - s.p_min)
    return x


def decode_normalized_array(X: npt.NDArray[np.float64], specs: list[ParamSpec]) -> list[dict[str, float]]:
    """Decode a population matrix (n_pop, n_params) to a list of param dicts."""
    return [decode_normalized(X[i], specs) for i in range(X.shape[0])]


def nn_param_specs_from_architecture(
    layer_sizes: list[int],
    activations: list[str],
    bound_multiplier: float = 2.0,
) -> list[ParamSpec]:
    """Generate ParamSpec list for NN weights from architecture.

    Each weight gets bounds [-m * scale, +m * scale] where scale is the
    Xavier/He/LeCun bound for its layer and m is bound_multiplier.
    Biases use the same bounds as their layer's weights.
    """
    specs: list[ParamSpec] = []
    for layer_idx in range(len(activations)):
        fan_in = layer_sizes[layer_idx]
        fan_out = layer_sizes[layer_idx + 1]
        bound = bound_multiplier * compute_layer_bound(fan_in, fan_out, activations[layer_idx])

        for j in range(fan_out):
            for k in range(fan_in):
                specs.append(ParamSpec(f"w{layer_idx}_{j}_{k}", -bound, bound, 0.0))
        for j in range(fan_out):
            specs.append(ParamSpec(f"bias{layer_idx}_{j}", -bound, bound, 0.0))

    return specs
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_encoding.py -v
```

Expected: all 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/encoding.py tests/test_encoding.py
git commit -m "feat: add real-valued encoding module for pymoo optimizer"
```

---

## Task 3: OptimizerConfig dataclass + algorithm factory

**Files:**
- Create: `src/python/aerocapture/training/optimizer.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests for optimizer config and factory**

Create `tests/test_optimizer.py`:

```python
"""Tests for optimizer configuration and algorithm factory."""

from __future__ import annotations

import pytest

from aerocapture.training.optimizer import OptimizerConfig, GASettings, CMAESSettings, DESettings, PSOSettings, create_algorithm


class TestOptimizerConfig:
    """OptimizerConfig dataclass construction."""

    def test_default_algorithm_is_ga(self):
        cfg = OptimizerConfig()
        assert cfg.algorithm == "ga"

    def test_all_algorithms_accepted(self):
        for alg in ("ga", "cma_es", "de", "pso"):
            cfg = OptimizerConfig(algorithm=alg)
            assert cfg.algorithm == alg

    def test_invalid_algorithm_rejected(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            OptimizerConfig(algorithm="genetic_programming")

    def test_from_toml_dict_ga(self):
        d = {
            "algorithm": "ga",
            "n_pop": 80,
            "n_gen": 3000,
            "seed_pool_interval": 25,
            "ga": {"crossover_eta": 20, "mutation_eta": 25},
        }
        cfg = OptimizerConfig.from_dict(d)
        assert cfg.algorithm == "ga"
        assert cfg.n_pop == 80
        assert cfg.n_gen == 3000
        assert cfg.seed_pool_interval == 25
        assert cfg.ga.crossover_eta == 20
        assert cfg.ga.mutation_eta == 25

    def test_from_toml_dict_cma_es(self):
        d = {
            "algorithm": "cma_es",
            "n_pop": 60,
            "n_gen": 500,
            "cma_es": {"sigma0": 0.5, "restart_strategy": "bipop"},
        }
        cfg = OptimizerConfig.from_dict(d)
        assert cfg.algorithm == "cma_es"
        assert cfg.cma_es.sigma0 == 0.5
        assert cfg.cma_es.restart_strategy == "bipop"

    def test_defaults_when_subsection_missing(self):
        d = {"algorithm": "de", "n_pop": 40, "n_gen": 1000}
        cfg = OptimizerConfig.from_dict(d)
        assert cfg.de.variant == "DE/rand/1/bin"
        assert cfg.de.crossover_prob == 0.7


class TestCreateAlgorithm:
    """Algorithm factory produces correct pymoo objects."""

    def test_ga_returns_nsga2_or_ga(self):
        from pymoo.algorithms.soo.nonconvex.ga import GA

        cfg = OptimizerConfig(algorithm="ga", n_pop=20)
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg, GA)

    def test_cma_es_returns_cmaes(self):
        from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

        cfg = OptimizerConfig(algorithm="cma_es", n_pop=20)
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg, CMAES)

    def test_de_returns_de(self):
        from pymoo.algorithms.soo.nonconvex.de import DE

        cfg = OptimizerConfig(algorithm="de", n_pop=20)
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg, DE)

    def test_pso_returns_pso(self):
        from pymoo.algorithms.soo.nonconvex.pso import PSO

        cfg = OptimizerConfig(algorithm="pso", n_pop=20)
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg, PSO)

    def test_cma_es_high_dim_warns_and_falls_back(self):
        from pymoo.algorithms.soo.nonconvex.ga import GA

        cfg = OptimizerConfig(algorithm="cma_es", n_pop=20)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            alg = create_algorithm(cfg, n_params=400)
            assert len(w) == 1
            assert "high-dimensional" in str(w[0].message).lower()
        assert isinstance(alg, GA)

    def test_ga_uses_sbx_crossover(self):
        cfg = OptimizerConfig(algorithm="ga", n_pop=20)
        cfg.ga.crossover_eta = 30
        alg = create_algorithm(cfg, n_params=5)
        # pymoo GA stores operators; verify SBX is configured
        assert alg.mating.crossover.__class__.__name__ == "SBX"

    def test_ga_uses_polynomial_mutation(self):
        cfg = OptimizerConfig(algorithm="ga", n_pop=20)
        alg = create_algorithm(cfg, n_params=5)
        assert alg.mating.mutation.__class__.__name__ == "PM"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_optimizer.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement optimizer module**

Create `src/python/aerocapture/training/optimizer.py`:

```python
"""pymoo algorithm factory and optimizer configuration.

Supports GA (SBX + polynomial mutation), CMA-ES, DE, and PSO.
All algorithms operate on normalized [0, 1] search space.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

_VALID_ALGORITHMS = {"ga", "cma_es", "de", "pso"}
_CMAES_DIM_LIMIT = 200


@dataclass
class GASettings:
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    mutation_prob: float | None = None  # None = 1/n_params


@dataclass
class CMAESSettings:
    sigma0: float = 0.3
    restart_strategy: str = "ipop"


@dataclass
class DESettings:
    variant: str = "DE/rand/1/bin"
    crossover_prob: float = 0.7
    scaling_factor: float = 0.5


@dataclass
class PSOSettings:
    w: float = 0.9
    c1: float = 2.0
    c2: float = 2.0


@dataclass
class OptimizerConfig:
    algorithm: str = "ga"
    n_pop: int = 60
    n_gen: int = 2500
    seed_pool_interval: int = 50
    adaptive_seeds: bool = False
    seed_pool_cap: int = 100
    cost_alpha: float = 0.7
    cvar_percentile: int = 20
    stress_interval: int = 5
    stress_probes: int = 200
    stress_inject: int = 20
    ga: GASettings = field(default_factory=GASettings)
    cma_es: CMAESSettings = field(default_factory=CMAESSettings)
    de: DESettings = field(default_factory=DESettings)
    pso: PSOSettings = field(default_factory=PSOSettings)

    def __post_init__(self):
        if self.algorithm not in _VALID_ALGORITHMS:
            msg = f"Unknown algorithm '{self.algorithm}'. Valid: {sorted(_VALID_ALGORITHMS)}"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, d: dict) -> OptimizerConfig:
        ga = GASettings(**d.get("ga", {}))
        cma_es = CMAESSettings(**d.get("cma_es", {}))
        de = DESettings(**d.get("de", {}))
        pso = PSOSettings(**d.get("pso", {}))
        top_keys = {k: v for k, v in d.items() if k not in ("ga", "cma_es", "de", "pso")}
        return cls(**top_keys, ga=ga, cma_es=cma_es, de=de, pso=pso)


def create_algorithm(config: OptimizerConfig, n_params: int) -> Algorithm:
    """Create a pymoo Algorithm from config.

    For CMA-ES with n_params > 200, warns and falls back to GA.
    """
    alg_name = config.algorithm

    if alg_name == "cma_es" and n_params > _CMAES_DIM_LIMIT:
        warnings.warn(
            f"CMA-ES is not recommended for high-dimensional problems (n_params={n_params} > {_CMAES_DIM_LIMIT}). "
            f"Falling back to GA with SBX crossover.",
            stacklevel=2,
        )
        alg_name = "ga"

    if alg_name == "ga":
        mutation_prob = config.ga.mutation_prob if config.ga.mutation_prob is not None else 1.0 / n_params
        return GA(
            pop_size=config.n_pop,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=config.ga.crossover_eta, prob=0.9),
            mutation=PM(eta=config.ga.mutation_eta, prob=mutation_prob),
            eliminate_duplicates=True,
        )

    if alg_name == "cma_es":
        return CMAES(
            x0=0.5 * np.ones(n_params),
            sigma=config.cma_es.sigma0,
            restarts=config.cma_es.restart_strategy == "ipop" or config.cma_es.restart_strategy == "bipop",
        )

    if alg_name == "de":
        return DE(
            pop_size=config.n_pop,
            variant=config.de.variant,
            CR=config.de.crossover_prob,
            F=config.de.scaling_factor,
            sampling=FloatRandomSampling(),
        )

    if alg_name == "pso":
        return PSO(
            pop_size=config.n_pop,
            w=config.pso.w,
            c1=config.pso.c1,
            c2=config.pso.c2,
            sampling=FloatRandomSampling(),
        )

    msg = f"Unknown algorithm '{alg_name}'"
    raise ValueError(msg)
```

Note: Add `import numpy as np` at the top of the file (needed for CMA-ES x0).

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_optimizer.py -v
```

Expected: all 13 tests PASS

Note: pymoo's CMA-ES constructor parameters may differ from what's shown above. If tests fail, read pymoo's actual API:
```bash
uv run python -c "from pymoo.algorithms.soo.nonconvex.cmaes import CMAES; help(CMAES.__init__)"
```
Adjust the factory accordingly -- the test expectations (isinstance checks, class names) are what matter.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/optimizer.py tests/test_optimizer.py
git commit -m "feat: add optimizer config and pymoo algorithm factory"
```

---

## Task 4: AerocaptureProblem subclass

**Files:**
- Create: `src/python/aerocapture/training/problem.py`
- Test: `tests/test_problem.py`

- [ ] **Step 1: Write failing tests for the Problem subclass**

Create `tests/test_problem.py`:

```python
"""Tests for AerocaptureProblem pymoo integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.problem import AerocaptureProblem


def _make_specs() -> list[ParamSpec]:
    return [
        ParamSpec("tau", 2.0, 60.0, 30.0),
        ParamSpec("threshold", 0.5, 5.0, 2.0),
        ParamSpec("gain", 1e-8, 1e-5, 1e-6, log_scale=True),
    ]


class TestProblemShape:
    """Problem correctly declares dimensionality and bounds."""

    def test_n_var(self):
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        assert p.n_var == 3

    def test_bounds_zero_one(self):
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        assert np.all(p.xl == 0.0)
        assert np.all(p.xu == 1.0)

    def test_single_objective(self):
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        assert p.n_obj == 1


class TestProblemEvaluation:
    """Problem._evaluate dispatches to simulator and returns costs."""

    def test_evaluate_returns_correct_shape(self):
        specs = _make_specs()
        p = AerocaptureProblem(
            param_specs=specs,
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={"dv_threshold": 1000.0, "g_load_limit": 15.0, "heat_flux_limit": 200.0, "heat_load_limit": 25000.0, "g_load_weight": 1000.0, "heat_flux_weight": 1000.0, "heat_load_weight": 1000.0},
            scheme="equilibrium_glide",
        )
        X = np.random.default_rng(0).random((5, 3))
        out = {}
        # Mock the evaluation to return dummy costs
        with patch.object(p, "_run_batch", return_value=np.array([100.0, 200.0, 150.0, 300.0, 50.0])):
            p._evaluate(X, out)
        assert out["F"].shape == (5, 1)

    def test_evaluate_values_are_finite(self):
        specs = _make_specs()
        p = AerocaptureProblem(
            param_specs=specs,
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={"dv_threshold": 1000.0, "g_load_limit": 15.0, "heat_flux_limit": 200.0, "heat_load_limit": 25000.0, "g_load_weight": 1000.0, "heat_flux_weight": 1000.0, "heat_load_weight": 1000.0},
            scheme="equilibrium_glide",
        )
        X = np.random.default_rng(1).random((3, 3))
        out = {}
        with patch.object(p, "_run_batch", return_value=np.array([100.0, 200.0, 150.0])):
            p._evaluate(X, out)
        assert np.all(np.isfinite(out["F"]))


class TestSeedUpdate:
    """Seed update mechanism."""

    def test_update_seeds(self):
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        assert p.seeds == [42]
        p.update_seeds([1, 2, 3])
        assert p.seeds == [1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_problem.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Problem subclass**

Create `src/python/aerocapture/training/problem.py`:

```python
"""pymoo Problem subclass for aerocapture optimization.

Bridges pymoo's population-level evaluation with the Rust simulator
via PyO3 batch calls. All individuals are normalized [0, 1] arrays.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pymoo.core.problem import Problem

from aerocapture.training.encoding import decode_normalized, decode_normalized_array
from aerocapture.training.evaluate import compute_cost, write_guidance_toml, write_nn_json
from aerocapture.training.param_spaces import ParamSpec


class AerocaptureProblem(Problem):
    def __init__(
        self,
        param_specs: list[ParamSpec],
        toml_path: str | Path,
        seeds: list[int],
        cost_kwargs: dict,
        scheme: str,
        sim_timeout: float | None = None,
        nn_config: dict | None = None,
        n_sims_override: int | None = None,
    ):
        self.param_specs = param_specs
        self.toml_path = str(toml_path)
        self.seeds = list(seeds)
        self.cost_kwargs = cost_kwargs
        self.scheme = scheme
        self.sim_timeout = sim_timeout
        self.nn_config = nn_config
        self.n_sims_override = n_sims_override

        super().__init__(
            n_var=len(param_specs),
            n_obj=1,
            xl=np.zeros(len(param_specs)),
            xu=np.ones(len(param_specs)),
        )

    def update_seeds(self, seeds: list[int]) -> None:
        self.seeds = list(seeds)

    def _evaluate(self, X: npt.NDArray[np.float64], out: dict, *args, **kwargs) -> None:
        costs = self._run_batch(X)
        out["F"] = costs.reshape(-1, 1)

    def _run_batch(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate population X (n_pop, n_params) across all seeds.

        Returns cost array of shape (n_pop,).
        """
        param_dicts = decode_normalized_array(X, self.param_specs)
        n_pop = X.shape[0]

        try:
            import aerocapture_rs as _aero_rs
            return self._run_batch_pyo3(_aero_rs, param_dicts, n_pop)
        except ImportError:
            return self._run_batch_subprocess(param_dicts, n_pop)

    def _run_batch_pyo3(self, _aero_rs, param_dicts: list[dict[str, float]], n_pop: int) -> npt.NDArray[np.float64]:
        """Batch evaluation via PyO3."""
        all_costs = np.zeros((n_pop, len(self.seeds)), dtype=np.float64)

        for seed_idx, seed in enumerate(self.seeds):
            overrides_list = []
            for params in param_dicts:
                overrides = self._build_overrides(params, mc_seed=seed)
                overrides_list.append(overrides)

            results = _aero_rs.run_batch(
                self.toml_path,
                overrides_list,
                n_threads=None,
                include_trajectories=False,
                sim_timeout_secs=self.sim_timeout,
            )

            for i, final_rec in enumerate(results.final_records):
                all_costs[i, seed_idx] = compute_cost(final_rec, **self.cost_kwargs)

        # RMS across seeds
        return np.sqrt(np.mean(all_costs**2, axis=1))

    def _run_batch_subprocess(self, param_dicts: list[dict[str, float]], n_pop: int) -> npt.NDArray[np.float64]:
        """Fallback: sequential subprocess evaluation."""
        from aerocapture.training.evaluate import run_simulation

        costs = np.zeros(n_pop, dtype=np.float64)
        for i, params in enumerate(param_dicts):
            seed = self.seeds[0] if self.seeds else None
            if self.nn_config is not None:
                self._write_nn_weights(params)
            toml_path = write_guidance_toml(
                self.toml_path, self.scheme, params, mc_seed=seed, n_sims_override=self.n_sims_override,
            )
            final_cond = run_simulation_from_toml(toml_path)
            if final_cond is not None:
                costs[i] = compute_cost(final_cond, **self.cost_kwargs)
            else:
                costs[i] = 1e6
        return costs

    def _build_overrides(self, params: dict[str, float], mc_seed: int | None = None) -> dict[str, object]:
        """Build dot-path override dict for PyO3 run_batch.

        Routes parameters to TOML sections by prefix:
        - lateral.* -> guidance.lateral.*
        - exit.* -> guidance.ftc.*
        - nav.* -> navigation.*
        - thermal.* -> guidance.thermal_limiter.*
        - shaping.* -> guidance.command_shaping.*
        - unprefixed -> guidance.<scheme>.*
        """
        overrides: dict[str, object] = {}

        for name, value in params.items():
            if name.startswith("lateral."):
                key = f"guidance.lateral.{name.removeprefix('lateral.')}"
            elif name.startswith("exit."):
                key = f"guidance.ftc.{name.removeprefix('exit.')}"
            elif name.startswith("nav."):
                key = f"navigation.{name.removeprefix('nav.')}"
            elif name.startswith("thermal."):
                key = f"guidance.thermal_limiter.{name.removeprefix('thermal.')}"
            elif name.startswith("shaping."):
                key = f"guidance.command_shaping.{name.removeprefix('shaping.')}"
            else:
                key = f"guidance.{self.scheme}.{name}"
            overrides[key] = value

        if mc_seed is not None:
            overrides["monte_carlo.seed"] = mc_seed

        if self.n_sims_override is not None:
            overrides["monte_carlo.n_sims"] = self.n_sims_override

        return overrides

    def _write_nn_weights(self, params: dict[str, float]) -> None:
        """Write NN weight JSON from decoded params. Used for subprocess fallback."""
        if self.nn_config is None:
            return
        weights = np.array([params[s.name] for s in self.param_specs])
        write_nn_json(weights, self.nn_config, self.nn_config["weight_path"])
```

Note: The subprocess fallback (`_run_batch_subprocess`) references `run_simulation_from_toml` which doesn't exist yet. For now this is a placeholder -- the PyO3 path is the primary evaluation path. The subprocess fallback will be wired up when we refactor `evaluate.py` in Task 6. For testing purposes, we mock `_run_batch` directly.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_problem.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/problem.py tests/test_problem.py
git commit -m "feat: add AerocaptureProblem pymoo subclass for batch evaluation"
```

---

## Task 5: Update config.py -- replace GAConfig with OptimizerConfig

**Files:**
- Modify: `src/python/aerocapture/training/config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_config.py`

- [ ] **Step 1: Read current config.py**

Read `src/python/aerocapture/training/config.py` in full to understand the current structure.

- [ ] **Step 2: Write failing tests for OptimizerConfig integration**

In `tests/test_config.py`, delete the `TestChromLengthConsistency` class (tests `chrom_length` property which depends on `n_bit`). Add new tests:

```python
class TestOptimizerConfigFromToml:
    """OptimizerConfig parsed from training TOML."""

    def test_default_optimizer_when_section_missing(self):
        """When no [optimizer] section, defaults to GA."""
        config = make_training_config(scheme="equilibrium_glide")
        assert config.optimizer.algorithm == "ga"
        assert config.optimizer.n_pop == 60

    def test_optimizer_algorithm_from_toml(self):
        config = make_training_config(scheme="equilibrium_glide", optimizer={"algorithm": "de", "n_pop": 40})
        assert config.optimizer.algorithm == "de"
        assert config.optimizer.n_pop == 40
```

The `make_training_config` helper should be updated or created in the test fixtures to accept an `optimizer` dict.

In `tests/test_training_config.py`, delete `test_ga_config_rotate_seeds_default_false` (references `GAConfig`).

- [ ] **Step 3: Replace GAConfig with OptimizerConfig in config.py**

In `src/python/aerocapture/training/config.py`:

1. Remove the `GAConfig` dataclass (lines 57-79)
2. Remove `from aerocapture.training.config import GAConfig` if self-referenced
3. In `TrainingConfig`:
   - Replace `ga: GAConfig = field(default_factory=GAConfig)` with `optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)`
   - Add import: `from aerocapture.training.optimizer import OptimizerConfig`
   - Remove the `chrom_length` property (lines 115-117)
   - Update `n_params` to stay (it's still useful for the Problem subclass)
4. In TOML parsing logic: read `[optimizer]` section instead of `[ga]`, pass to `OptimizerConfig.from_dict()`
5. Keep `n_runs` (used by the training loop) -- move it to `OptimizerConfig` or keep at `TrainingConfig` level

- [ ] **Step 4: Run tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_config.py tests/test_training_config.py -v
```

Expected: all remaining tests PASS, deleted tests gone

- [ ] **Step 5: Fix any downstream imports**

Search for all imports of `GAConfig` or references to `config.ga`:
```bash
cd /Users/govit/Git/Govit/Aerocapture && rg "GAConfig|config\.ga\." src/python/ tests/
```

Update each reference to use `config.optimizer` instead. This will touch multiple files -- fix compilation errors but don't refactor the logic yet (that comes in Task 7).

- [ ] **Step 6: Run full test suite to check nothing else broke**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q
```

Expected: existing tests pass (some may fail due to GAConfig references -- fix those)

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/config.py tests/test_config.py tests/test_training_config.py
git add -u  # catch any other files with GAConfig fixes
git commit -m "refactor: replace GAConfig with OptimizerConfig in training config"
```

---

## Task 6: Clean up evaluate.py -- remove binary encoding, keep cost + TOML patching

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py`
- Modify: `tests/test_toml_patching.py`

- [ ] **Step 1: Read current evaluate.py**

Read `src/python/aerocapture/training/evaluate.py` to identify all functions.

- [ ] **Step 2: Remove binary encoding functions**

Delete these functions from `evaluate.py`:
- `binary_to_decimal()` (lines ~29-47)
- `decode_direct()` (lines ~285-300)
- `decode_params_from_chromosome()` (lines ~303-334)
- `perturb_network()` (lines ~50-84)
- `evaluate_chromosome()` (lines ~551-621) -- this is the main binary-era entry point, replaced by `AerocaptureProblem._evaluate()`

**Keep these functions** (still needed by Problem subclass and training loop):
- `compute_cost()` -- unchanged
- `write_guidance_toml()` -- unchanged
- `write_nn_json()` -- unchanged
- `run_simulation()` / `_run_via_pyo3()` / `_run_via_subprocess()` -- still needed for final evaluation and compare_guidance
- `log_cap()` -- used by compute_cost
- PyO3 detection (`_HAS_PYO3`, `_aero_rs`)

- [ ] **Step 3: Update imports in evaluate.py**

Remove unused imports that were only needed by deleted functions (e.g., config types used only by binary decoding).

- [ ] **Step 4: Update test_toml_patching.py**

In `tests/test_toml_patching.py`, find any tests that use `evaluate_chromosome()`, `decode_params_from_chromosome()`, or `encode_params_to_chromosome()`. Update them to use the new encoding module functions or delete them if they only test binary encoding.

Keep all tests that test `write_guidance_toml()`, `compute_cost()`, TOML patching logic, and parameter routing -- these are still valid.

- [ ] **Step 5: Run tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_toml_patching.py tests/test_cost.py -v
```

Expected: TOML patching and cost tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py tests/test_toml_patching.py
git commit -m "refactor: remove binary encoding from evaluate.py, keep cost and TOML patching"
```

---

## Task 7: Rewrite population.py for real-valued initialization

**Files:**
- Modify: `src/python/aerocapture/training/population.py`
- Modify: `tests/test_initialization.py`

- [ ] **Step 1: Write failing tests for real-valued population initialization**

Add to or replace tests in `tests/test_initialization.py`:

```python
class TestRealValuedPopulation:
    """Real-valued initial population generation."""

    def test_output_shape(self):
        from aerocapture.training.population import create_initial_population
        from aerocapture.training.param_spaces import ParamSpec

        specs = [
            ParamSpec("tau", 2.0, 60.0, 30.0),
            ParamSpec("threshold", 0.5, 5.0, 2.0),
        ]
        pop = create_initial_population(specs, n_pop=10, rng=np.random.default_rng(42))
        assert pop.shape == (10, 2)

    def test_values_in_unit_range(self):
        from aerocapture.training.population import create_initial_population
        from aerocapture.training.param_spaces import ParamSpec

        specs = [ParamSpec("x", -10.0, 10.0, 0.0)]
        pop = create_initial_population(specs, n_pop=50, rng=np.random.default_rng(42))
        assert np.all(pop >= 0.0)
        assert np.all(pop <= 1.0)

    def test_seeded_with_defaults(self):
        from aerocapture.training.population import create_initial_population
        from aerocapture.training.encoding import decode_normalized
        from aerocapture.training.param_spaces import ParamSpec

        specs = [
            ParamSpec("tau", 2.0, 60.0, 30.0),
            ParamSpec("gain", 1e-8, 1e-5, 1e-6, log_scale=True),
        ]
        pop = create_initial_population(specs, n_pop=10, rng=np.random.default_rng(42), seed_defaults=True)
        # First individual should be the encoded defaults
        decoded = decode_normalized(pop[0], specs)
        assert decoded["tau"] == pytest.approx(30.0)
        assert decoded["gain"] == pytest.approx(1e-6, rel=1e-10)

    def test_nn_population_shape(self):
        from aerocapture.training.population import create_nn_initial_population

        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        pop = create_nn_initial_population(
            layer_sizes, activations, n_pop=10,
            rng=np.random.default_rng(42), bound_multiplier=2.0,
        )
        # 16*24 + 24 + 24*2 + 2 = 458 params
        assert pop.shape == (10, 458)
        assert np.all(pop >= 0.0)
        assert np.all(pop <= 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_initialization.py::TestRealValuedPopulation -v
```

Expected: FAIL

- [ ] **Step 3: Rewrite population.py**

Replace the binary-encoding functions in `src/python/aerocapture/training/population.py` with real-valued versions:

```python
"""Real-valued population initialization for pymoo optimizer.

Generates initial populations as normalized [0, 1] arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aerocapture.training.encoding import encode_to_normalized, nn_param_specs_from_architecture
from aerocapture.training.initialization import generate_initialized_weights, compute_layer_bound
from aerocapture.training.param_spaces import ParamSpec


def create_initial_population(
    specs: list[ParamSpec],
    n_pop: int,
    rng: np.random.Generator,
    seed_defaults: bool = True,
    seed_params: dict[str, float] | None = None,
    perturbation_scale: float = 0.05,
) -> npt.NDArray[np.float64]:
    """Create initial population as normalized [0, 1] array.

    If seed_defaults=True, first individual is the encoded defaults,
    next few are perturbations around defaults.
    Rest is uniform random.
    """
    n_params = len(specs)
    pop = rng.random((n_pop, n_params))

    if seed_defaults or seed_params is not None:
        defaults = seed_params if seed_params is not None else {s.name: s.default for s in specs}
        default_normalized = encode_to_normalized(defaults, specs)
        pop[0] = default_normalized

        # Create perturbations around defaults
        n_perturb = min(n_pop // 3, 10)
        for i in range(1, 1 + n_perturb):
            noise = rng.normal(0.0, perturbation_scale, size=n_params)
            pop[i] = np.clip(default_normalized + noise, 0.0, 1.0)

    return pop


def create_nn_initial_population(
    layer_sizes: list[int],
    activations: list[str],
    n_pop: int,
    rng: np.random.Generator,
    bound_multiplier: float = 2.0,
    seed_weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Create initial population for NN weight optimization.

    Uses Xavier/He initialization scaled to [0, 1] via nn_param_specs.
    """
    specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier)
    n_params = len(specs)
    pop = np.empty((n_pop, n_params), dtype=np.float64)

    for i in range(n_pop):
        weights = generate_initialized_weights(layer_sizes, activations, rng)
        # Encode to normalized [0, 1]
        for j, s in enumerate(specs):
            pop[i, j] = (weights[j] - s.p_min) / (s.p_max - s.p_min)
            pop[i, j] = np.clip(pop[i, j], 0.0, 1.0)

    if seed_weights is not None:
        for j, s in enumerate(specs):
            pop[0, j] = np.clip((seed_weights[j] - s.p_min) / (s.p_max - s.p_min), 0.0, 1.0)

    return pop
```

- [ ] **Step 4: Delete old binary encoding functions**

Remove `encode_weights_to_chromosome()` and `encode_params_to_chromosome()` from the file. Delete the tests in `tests/test_initialization.py` that reference these:
- `TestEncodeDecodeRoundtrip.test_initialized_weights_survive_roundtrip`
- `TestPopulationSmartInit.test_nn_population_uses_smart_init`
- `TestPopulationSmartInit.test_non_nn_uses_random_init`

- [ ] **Step 5: Run tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_initialization.py -v
```

Expected: new tests PASS, deleted tests gone, kept tests (weight init, weight stats) still pass

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/population.py tests/test_initialization.py
git commit -m "refactor: rewrite population.py for real-valued initialization"
```

---

## Task 8: Update seed_pool.py and logger.py signatures

**Files:**
- Modify: `src/python/aerocapture/training/seed_pool.py`
- Modify: `src/python/aerocapture/training/logger.py`

- [ ] **Step 1: Update seed_pool.py evaluate_population() signature**

In `src/python/aerocapture/training/seed_pool.py`, the `evaluate_population()` method currently accepts `population: npt.NDArray[np.int8]` (binary chromosomes). Change to:

```python
def evaluate_population(
    self,
    population: npt.NDArray[np.float64],
    evaluator: Callable[[npt.NDArray[np.float64], int], float],
    batch_evaluator: Callable[[npt.NDArray[np.float64], list[int]], npt.NDArray[np.float64]] | None = None,
) -> npt.NDArray[np.float64]:
```

This is just a type annotation change -- the actual logic (iterate over population, call evaluator per seed) doesn't depend on the array dtype.

- [ ] **Step 2: Update logger.py log_generation() signature**

In `src/python/aerocapture/training/logger.py`, change `log_generation()`:

```python
def log_generation(
    self,
    generation: int,
    population: npt.NDArray[np.float64],
    costs: npt.NDArray[np.float64],
    best_individual: npt.NDArray[np.float64],
    decode_fn: Callable[[npt.NDArray[np.float64]], dict[str, float]] | None,
    weight_stats: dict[str, dict[str, float]] | None = None,
    mc_seed: int | None = None,
    pool_metrics: dict | None = None,
    gen_elapsed_s: float | None = None,
    gen_best_individual: npt.NDArray[np.float64] | None = None,
) -> None:
```

Key changes:
- `populations: list[npt.NDArray[np.int8]]` -> `population: npt.NDArray[np.float64]` (single population, not list of subpopulations)
- `costs: list[npt.NDArray[np.float64]]` -> `costs: npt.NDArray[np.float64]` (flat array)
- `best_chromosome` -> `best_individual`
- `gen_best_chromosome` -> `gen_best_individual`
- Update the body to work with flat arrays instead of lists of subpop arrays. Where it currently does `all_costs = np.concatenate(costs)`, just use `costs` directly. Where it does `all_pop = np.vstack(populations)`, just use `population` directly.

- [ ] **Step 3: Run seed pool and logger tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py tests/test_training_logger.py -v
```

Expected: seed pool tests pass (logic unchanged). Logger tests may need updates to pass flat arrays instead of lists -- update test fixtures accordingly.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/seed_pool.py src/python/aerocapture/training/logger.py tests/test_training_logger.py
git commit -m "refactor: update seed_pool and logger signatures for real-valued arrays"
```

---

## Task 9: Delete local_search.py, migration.py, test_ga_operators.py

**Files:**
- Delete: `src/python/aerocapture/training/local_search.py`
- Delete: `src/python/aerocapture/training/migration.py`
- Delete: `tests/test_ga_operators.py`

- [ ] **Step 1: Remove imports of deleted modules**

Search for all imports of `local_search` and `migration`:
```bash
cd /Users/govit/Git/Govit/Aerocapture && rg "from aerocapture.training.local_search|from aerocapture.training.migration|import local_search|import migration" src/python/ tests/
```

Remove all import statements and call sites found. Key locations:
- `train.py`: imports `migrate` from `migration.py` and calls it in the generation loop
- `population.py`: imports `improve_chromosome` from `local_search.py` and calls it during init
- `migration.py`: imports `improve_chromosome` from `local_search.py`

- [ ] **Step 2: Delete the files**

```bash
rm src/python/aerocapture/training/local_search.py
rm src/python/aerocapture/training/migration.py
rm tests/test_ga_operators.py
```

- [ ] **Step 3: Run test suite to verify no broken imports**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q
```

Expected: no import errors. Some tests may fail due to the train.py loop not being rewritten yet -- that's expected and will be fixed in Task 10.

- [ ] **Step 4: Commit**

```bash
git add -u  # stages deletions and modifications
git commit -m "refactor: delete local_search.py, migration.py, and binary GA operator tests"
```

---

## Task 10: Rewrite train.py -- hybrid pymoo loop

This is the largest task. It replaces the binary GA generation loop with the hybrid pymoo loop.

**Files:**
- Modify: `src/python/aerocapture/training/train.py`
- Create: `tests/test_hybrid_loop.py`

- [ ] **Step 1: Write integration test for the hybrid loop**

Create `tests/test_hybrid_loop.py`:

```python
"""Integration tests for the hybrid pymoo training loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestHybridLoopSmoke:
    """Smoke test: the training loop runs for a few generations without crashing."""

    def test_ga_runs_3_generations(self, tmp_path):
        """GA algorithm runs 3 generations and produces a result dict."""
        from aerocapture.training.train import train
        from aerocapture.training.config import TrainingConfig
        from aerocapture.training.optimizer import OptimizerConfig

        # Create a minimal config
        optimizer = OptimizerConfig(algorithm="ga", n_pop=6, n_gen=3)
        # We need to mock the actual simulation since we don't have the Rust binary in tests
        # Instead, test that the loop structure works by mocking AerocaptureProblem._run_batch
        with patch("aerocapture.training.problem.AerocaptureProblem._run_batch") as mock_batch:
            mock_batch.return_value = np.random.default_rng(0).random(6) * 1000.0
            # This test validates the loop structure compiles and runs
            # Full integration with the simulator is tested in Task 12
            pass  # Placeholder -- actual test depends on train() refactored signature

    def test_checkpoint_save_load_roundtrip(self, tmp_path):
        """Checkpoint saves and restores population correctly."""
        from aerocapture.training.train import save_checkpoint, load_checkpoint

        population = np.random.default_rng(42).random((10, 5))
        costs = np.random.default_rng(42).random(10) * 100
        best = population[np.argmin(costs)]

        save_checkpoint(
            save_dir=tmp_path,
            generation=50,
            population=population,
            costs=costs,
            best_cost=float(np.min(costs)),
            best_individual=best,
            cost_history=[100.0, 80.0, 60.0],
            rng=np.random.default_rng(42),
            algorithm_name="ga",
        )

        loaded = load_checkpoint(tmp_path)
        assert loaded is not None
        assert loaded["generation"] == 50
        assert loaded["population"].shape == (10, 5)
        assert loaded["population"].dtype == np.float64
        np.testing.assert_array_almost_equal(loaded["population"], population)
        np.testing.assert_array_almost_equal(loaded["costs"], costs)
        np.testing.assert_array_almost_equal(loaded["best_individual"], best)

    def test_interrupt_saves_checkpoint(self, tmp_path):
        """KeyboardInterrupt during training saves a checkpoint and returns cleanly."""
        # This test validates the Ctrl+C handling still works with the new loop
        pass  # Wire up after train() is refactored
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_hybrid_loop.py -v
```

Expected: FAIL (signature mismatches, missing functions)

- [ ] **Step 3: Rewrite the checkpoint functions in train.py**

Replace `save_checkpoint()` and `load_checkpoint()` with new versions that store real-valued populations:

```python
def save_checkpoint(
    save_dir: Path,
    generation: int,
    population: npt.NDArray[np.float64],
    costs: npt.NDArray[np.float64],
    best_cost: float,
    best_individual: npt.NDArray[np.float64],
    cost_history: list[float],
    rng: np.random.Generator,
    algorithm_name: str,
    seed_pool: SeedPool | None = None,
    corridor_acc: CorridorAccumulator | None = None,
) -> None:
    """Save training checkpoint (real-valued population)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Remove old checkpoints in this directory
    for old in save_dir.glob("checkpoint_*.json"):
        old.unlink()
    for old in save_dir.glob("checkpoint_*.npz"):
        old.unlink()

    tag = f"checkpoint_g{generation:05d}"
    json_path = save_dir / f"{tag}.json"
    npz_path = save_dir / f"{tag}.npz"

    # JSON metadata
    rng_state = rng.bit_generator.state
    meta = {
        "generation": generation,
        "best_cost": best_cost,
        "cost_history": cost_history,
        "algorithm": algorithm_name,
        "rng_state": _serialize_rng_state(rng_state),
    }
    if seed_pool is not None:
        meta["seed_pool"] = seed_pool.to_dict()

    json_path.write_text(json.dumps(meta, indent=2))

    # NPZ arrays
    npz_data = {
        "population": population,
        "costs": costs,
        "best_individual": best_individual,
    }
    if corridor_acc is not None:
        npz_data.update(corridor_acc.to_npz_dict())
    np.savez(npz_path, **npz_data)


def load_checkpoint(save_dir: Path) -> dict | None:
    """Load most recent checkpoint from directory."""
    json_files = sorted(save_dir.glob("checkpoint_*.json"))
    if not json_files:
        return None

    json_path = json_files[-1]
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.exists():
        return None

    meta = json.loads(json_path.read_text())
    data = np.load(npz_path)

    return {
        "generation": meta["generation"],
        "best_cost": meta["best_cost"],
        "cost_history": meta["cost_history"],
        "algorithm": meta.get("algorithm", "ga"),
        "population": data["population"],
        "costs": data["costs"],
        "best_individual": data["best_individual"],
        "rng_state": meta["rng_state"],
        "seed_pool": meta.get("seed_pool"),
    }
```

Keep the existing `_serialize_rng_state()` and `_deserialize_rng_state()` helper functions.

- [ ] **Step 4: Rewrite the main train() function**

Replace the generation loop in `train()` with the hybrid pymoo loop. This is the core change. The structure follows the spec:

```python
def train(
    config: TrainingConfig,
    seed: int | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    checkpoint_interval: int = 10,
    resume_dir: str | Path | None = None,
    no_tui: bool = False,
    corridor_acc: CorridorAccumulator | None = None,
    from_scratch: bool = False,
) -> dict:
    # 1. Setup: param specs, problem, algorithm
    specs = PARAM_SPACES[config.sim.guidance_type]  # from param_spaces.py
    problem = AerocaptureProblem(
        param_specs=specs,
        toml_path=config.sim.toml_path,
        seeds=[config.sim.mc_seed],
        cost_kwargs=build_cost_kwargs(config),
        scheme=config.sim.guidance_type,
        sim_timeout=config.sim.sim_timeout,
        nn_config=config.network.to_dict() if config.is_nn else None,
        n_sims_override=config.sim.train_n_sims,
    )

    algorithm = create_algorithm(config.optimizer, n_params=len(specs))

    # 2. Initial population
    if resume and checkpoint:
        pop_array = checkpoint["population"]
        start_gen = checkpoint["generation"]
    else:
        if config.is_nn:
            pop_array = create_nn_initial_population(...)
        else:
            pop_array = create_initial_population(specs, config.optimizer.n_pop, rng)
        start_gen = 0

    # 3. Setup pymoo algorithm with initial population
    from pymoo.operators.sampling.numpy import NumpySampling
    algorithm.setup(problem, sampling=pop_array)  # or via custom Sampling

    # 4. Seed pool setup
    seed_pool = None
    if config.optimizer.adaptive_seeds:
        seed_pool = SeedPool(base_seed=config.sim.mc_seed, max_size=config.optimizer.seed_pool_cap, ...)

    # 5. Logger + display
    logger = TrainingLogger(...)
    display = LiveDisplay(...) if not no_tui else NoopDisplay()

    # 6. Generation loop
    interrupted = False
    try:
        for gen in range(start_gen, start_gen + config.optimizer.n_gen):
            t0 = time.time()

            algorithm.next()

            pop = algorithm.pop
            X = pop.get("X")       # (n_pop, n_params) normalized
            F = pop.get("F")       # (n_pop, 1) costs
            costs = F[:, 0]
            best = algorithm.opt.get("X")[0]
            best_cost = float(algorithm.opt.get("F")[0, 0])

            gen_elapsed = time.time() - t0

            # Decode best for logging
            decode_fn = lambda x: decode_normalized(x, specs)

            logger.log_generation(gen + 1, X, costs, best, decode_fn, gen_elapsed_s=gen_elapsed)
            display.update(logger, current_run=0)

            # Corridor accumulator (piecewise_constant only)
            if corridor_acc is not None:
                _update_corridor(corridor_acc, X, specs, config, problem)

            # Seed pool checkpoint
            if seed_pool is not None and (gen + 1) % config.optimizer.seed_pool_interval == 0:
                seed_pool.add_seeds(gen)
                seed_pool.score_difficulty(...)
                seed_pool.evict_redundant()
                problem.update_seeds(seed_pool.seeds)
                # Re-evaluate population on updated seeds
                algorithm.evaluator.eval(problem, pop)

            # Stress test
            if seed_pool is not None and (gen + 1) % config.optimizer.stress_interval == 0:
                seed_pool.stress_test(...)

            # Checkpoint save
            if (gen + 1) % checkpoint_interval == 0:
                save_checkpoint(save_dir, gen + 1, X, costs, best_cost, best, ...)

    except KeyboardInterrupt:
        interrupted = True
        save_checkpoint(save_dir, gen + 1, X, costs, best_cost, best, ...)

    # 7. Post-training: final eval, report
    best_params = decode_normalized(best, specs)
    _save_best_params(best_params, config, save_dir)
    # ... final MC evaluation, PDF report generation (keep existing logic)

    return {"best_cost": best_cost, "best_params": best_params, "interrupted": interrupted}
```

This is pseudocode showing the structure -- the actual implementation needs to handle:
- Proper pymoo `Population` injection for initial sampling
- The `algorithm.evaluator.eval()` call for re-evaluation after seed pool updates
- All the existing post-training logic (final eval, best_params.json, report generation)

- [ ] **Step 5: Update CLI argument parsing**

In the `__main__` block / `parse_args()` function:
- Remove `--mutation-rate` as a top-level flag (it's now in `[optimizer.ga]`)
- Remove `--rotate-seeds` (not supported in new framework)
- Keep `--adaptive-seeds`, `--seed-pool-cap`, `--cost-alpha`, `--cvar-percentile` (move to optimizer config)
- Keep `--n-gen`, `--n-pop` (override `config.optimizer.n_gen` / `config.optimizer.n_pop`)
- Keep `--sim-timeout`, `--train-n-sims`, `--final-n-sims`, `--skip-report`, `--no-tui`

- [ ] **Step 6: Run tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_hybrid_loop.py tests/test_train_interrupt.py -v
```

Expected: checkpoint roundtrip test passes, interrupt test passes

- [ ] **Step 7: Run full test suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q
```

Fix any remaining failures from the refactor.

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_hybrid_loop.py
git add -u  # catch any other modified files
git commit -m "feat: rewrite training loop with hybrid pymoo algorithm stepping"
```

---

## Task 11: Update training TOML configs

**Files:**
- Modify: `configs/training/msr_aller_eqglide_train.toml`
- Modify: `configs/training/msr_aller_energy_controller_train.toml`
- Modify: `configs/training/msr_aller_pred_guid_train.toml`
- Modify: `configs/training/msr_aller_fnpag_train.toml`
- Modify: `configs/training/msr_aller_ftc_train.toml`
- Modify: `configs/training/msr_aller_piecewise_constant_train.toml`
- Modify: `configs/training/msr_aller_nn_train_consolidated.toml`
- Modify: any `configs/training/common.toml` if it has `[ga]`

- [ ] **Step 1: Find all training TOMLs with GA config**

```bash
cd /Users/govit/Git/Govit/Aerocapture && rg "\[ga\]|n_bit|n_subpop|mutation_rate|rotate_seeds" configs/
```

- [ ] **Step 2: Replace [ga] sections with [optimizer]**

For each training TOML, replace the `[ga]` section. Example transformation:

Before:
```toml
[ga]
n_pop = 60
n_gen = 2500
mutation_rate = 0.05
n_subpop = 1
adaptive_seeds = true
seed_pool_cap = 100
cost_alpha = 0.6
```

After:
```toml
[optimizer]
algorithm = "ga"
n_pop = 60
n_gen = 2500
adaptive_seeds = true
seed_pool_cap = 100
cost_alpha = 0.6

[optimizer.ga]
mutation_prob = 0.05
crossover_eta = 15
mutation_eta = 20
```

Note: `mutation_rate` (bit-flip rate) maps conceptually to `mutation_prob` (per-variable probability for polynomial mutation), but the scales are different. Use `null` (1/n_params) as default for most schemes -- this is pymoo's recommended default for polynomial mutation. Only set explicit values when the old config had a non-default mutation_rate.

Remove these fields that no longer exist:
- `n_bit` (binary encoding)
- `n_subpop` (island model removed)
- `migration_interval` (island model removed)
- `rotate_seeds` (replaced by adaptive_seeds)
- `n_runs` (move to top level or remove if unused)

- [ ] **Step 3: Verify TOML parsing**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run python -c "
from aerocapture.training.config import TrainingConfig
config = TrainingConfig.from_toml('configs/training/msr_aller_ftc_train.toml')
print(f'Algorithm: {config.optimizer.algorithm}')
print(f'n_pop: {config.optimizer.n_pop}')
print(f'n_gen: {config.optimizer.n_gen}')
"
```

Expected: prints correct values from the updated TOML

- [ ] **Step 4: Commit**

```bash
git add configs/training/
git commit -m "config: replace [ga] with [optimizer] in all training TOMLs"
```

---

## Task 12: Validation -- train piecewise_constant and FTC

**Files:**
- Create: `tests/test_pymoo_validation.py` (optional, can be manual)

This task validates the new framework end-to-end by training two schemes and comparing against baselines.

- [ ] **Step 1: Run piecewise_constant training (short run)**

```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run python -m aerocapture.training.train \
    configs/training/msr_aller_piecewise_constant_train.toml \
    --n-gen 50 --no-tui
```

Expected: training runs without errors, produces checkpoint and best_params.json

- [ ] **Step 2: Verify piecewise_constant output**

Check that `training_output/piecewise_constant/best_params.json` was written and contains the expected parameter names:

```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run python -c "
import json
with open('training_output/piecewise_constant/best_params.json') as f:
    params = json.load(f)
print(f'Params: {list(params.keys())}')
print(f'Count: {len(params)}')
# Should have 11 params: 10 bank angles + max_bank_acceleration
"
```

- [ ] **Step 3: Run FTC training (short run)**

```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run python -m aerocapture.training.train \
    configs/training/msr_aller_ftc_train.toml \
    --n-gen 50 --no-tui
```

Expected: training runs without errors

- [ ] **Step 4: Verify FTC output**

```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run python -c "
import json
with open('training_output/ftc/best_params.json') as f:
    params = json.load(f)
print(f'Params: {list(params.keys())}')
print(f'Count: {len(params)}')
# Should have 21 params (9 FTC + 2 nav + 6 lateral + 4 exit + 4 thermal + 1 shaping - some shared)
"
```

- [ ] **Step 5: Run full test suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q
```

Expected: all tests pass

- [ ] **Step 6: Run linter**

```bash
cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh
```

Fix any ruff or mypy issues.

- [ ] **Step 7: Commit validation results**

```bash
git add tests/
git commit -m "test: validate pymoo training loop with piecewise_constant and FTC"
```

---

## Task 13: Update IMPROVEMENTS.md with Bayesian and RL sections

**Files:**
- Modify: `IMPROVEMENTS.md`

- [ ] **Step 1: Read the current section 9 of IMPROVEMENTS.md**

Read `IMPROVEMENTS.md` to find section 9.1 and 9.2 to understand the surrounding context.

- [ ] **Step 2: Update section 9.1 to reflect completed work**

Mark section 9.1 as completed (or partially completed) -- the real-valued encoding and multi-algorithm support are now implemented. Update the text to reflect what was done.

- [ ] **Step 3: Add section 9.3 -- Bayesian Optimization**

After section 9.2, add:

```markdown
### 9.3 Bayesian optimization for low-dimensional schemes

Surrogate-model-based optimization using Gaussian Processes or Random Forests as a surrogate for the expensive MC fitness function. Promising for guidance parameter schemes (10-26 params) where each evaluation is costly.

- **Investigation**: Evaluate BoTorch (PyTorch-based, state of the art) or scikit-optimize as a pymoo-compatible backend. Key challenge: noisy fitness from MC evaluation requires noise-aware acquisition functions (e.g., noisy Expected Improvement).
- **Impact**: Could dramatically reduce the number of evaluations needed for convergence on smooth parameter landscapes, at the cost of surrogate model overhead.
```

- [ ] **Step 4: Add section 9.4 -- Reinforcement Learning for NN Training**

```markdown
### 9.4 Reinforcement learning for neural network guidance

Train the NN guidance controller as an RL policy rather than optimizing static weights via evolutionary algorithms. The simulator is already step-able (state -> action -> next state).

- **Investigation**: Wrap the Rust simulator as a Gym-compatible environment via a PyO3 step API (expose per-timestep state/action interface, not just full-trajectory evaluation). Evaluate PPO, SAC, or TD3 for continuous bank-angle control.
- **Impact**: Fundamentally different paradigm from weight optimization -- RL can learn temporal strategies that static weight optimization cannot express. Separate effort from the pymoo framework.
```

- [ ] **Step 5: Commit**

```bash
git add IMPROVEMENTS.md
git commit -m "docs: add Bayesian optimization and RL investigation sections to IMPROVEMENTS.md"
```

---

## Task 14: Update CLAUDE.md and final smart-commit

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Update the relevant sections of `CLAUDE.md` to reflect the new optimizer framework:
- In the Python Tools section, update references from "GA training pipeline" to "optimizer training pipeline"
- Update the `train.py` description to mention pymoo and the four algorithms
- Update the `config.py` description to reference `OptimizerConfig` instead of `GAConfig`
- Remove references to `local_search.py`, `migration.py`, binary encoding
- Update the GA Training & Comparison section's CLI examples to use `[optimizer]` config
- Add pymoo to the Python dependencies list

- [ ] **Step 2: Invoke smart-commit skill**

Use the `smart-commit` skill to commit all remaining changes, taking the whole git branch into account.

---

## Summary

| Task | Description | New/Modified Files | Tests |
|------|-------------|-------------------|-------|
| 1 | Add pymoo dependency | pyproject.toml | install check |
| 2 | Encoding module | encoding.py | test_encoding.py (13 tests) |
| 3 | Optimizer config + factory | optimizer.py | test_optimizer.py (13 tests) |
| 4 | Problem subclass | problem.py | test_problem.py (5 tests) |
| 5 | Replace GAConfig | config.py | test_config.py, test_training_config.py |
| 6 | Clean evaluate.py | evaluate.py | test_toml_patching.py |
| 7 | Real-valued population | population.py | test_initialization.py |
| 8 | Update seed_pool + logger | seed_pool.py, logger.py | existing tests |
| 9 | Delete old modules | local_search.py, migration.py | delete test_ga_operators.py |
| 10 | Hybrid training loop | train.py | test_hybrid_loop.py |
| 11 | Update training TOMLs | configs/training/*.toml | TOML parse check |
| 12 | Validation training | -- | PC + FTC end-to-end |
| 13 | IMPROVEMENTS.md | IMPROVEMENTS.md | -- |
| 14 | CLAUDE.md + smart-commit | CLAUDE.md | -- |
