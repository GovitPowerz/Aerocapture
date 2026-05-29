# Three-way `scaffolding` knob — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the NN-training `optimize_scaffolding` boolean with a three-valued `scaffolding = "off" | "live" | "full"` knob, where `"live"` optimizes only the 3 params that affect full_neural cost (nav ×2 + shaping) with no FTC dependency, and the choice is printed at training start.

**Architecture:** A single resolver `active_scaffolding_specs(scaffolding)` returns the active `ParamSpec` pack (`[]` / 3 / 17). Every site that hardcoded `17 if optimize_scaffolding else 0` switches to `len(pack)`. `"full"` keeps the existing FTC-seeded slab; `"live"` gets a new default-seeded slab. Pure Python-training change — the Rust runtime already ignores unknown `[guidance.neural_network]` keys.

**Tech Stack:** Python 3.14, numpy, pymoo, pytest, ruff, mypy. Spec: `docs/superpowers/specs/2026-05-29-scaffolding-three-way-knob-design.md`.

---

### Task 1: `_NN_LIVE_PARAMS` + `active_scaffolding_specs` resolver

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py` (after `_NN_SCAFFOLDING_PARAMS`, ~line 77)
- Test: `tests/test_nn_scaffolding_params.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nn_scaffolding_params.py`:

```python
def test_active_scaffolding_specs_three_way() -> None:
    from aerocapture.training.param_spaces import (
        _NN_LIVE_PARAMS,
        _NN_SCAFFOLDING_PARAMS,
        active_scaffolding_specs,
    )

    assert active_scaffolding_specs("off") == []
    assert active_scaffolding_specs("live") == _NN_LIVE_PARAMS
    assert active_scaffolding_specs("full") == _NN_SCAFFOLDING_PARAMS


def test_live_pack_is_nav_plus_shaping() -> None:
    from aerocapture.training.param_spaces import _NN_LIVE_PARAMS

    names = [s.name for s in _NN_LIVE_PARAMS]
    assert names == [
        "nav.density_filter_gain",
        "nav.density_gain_max_delta",
        "shaping.max_bank_acceleration",
    ]


def test_active_scaffolding_specs_rejects_unknown() -> None:
    import pytest

    from aerocapture.training.param_spaces import active_scaffolding_specs

    with pytest.raises(KeyError):
        active_scaffolding_specs("partial")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_nn_scaffolding_params.py::test_active_scaffolding_specs_three_way -v`
Expected: FAIL with `ImportError: cannot import name 'active_scaffolding_specs'`

- [ ] **Step 3: Add the live pack and resolver**

In `param_spaces.py`, immediately after the `_NN_SCAFFOLDING_PARAMS` list (after line 77, before the `PARAM_SPACES` dict), insert:

```python
# Live-in-full_neural scaffolding params: nav density filter feeds the NN's
# observation vector, command shaping shapes its output. These 3 have standalone
# defaults, so they can be optimized without seeding from FTC. Used for
# `scaffolding = "live"` (full_neural schemes that want nav/shaping tuned but
# don't need the FTC-only lateral/exit/thermal pack).
_NN_LIVE_PARAMS: list[ParamSpec] = [
    *_NAV_PARAMS,
    *_SHAPING_PARAMS,
]


def active_scaffolding_specs(scaffolding: str) -> list[ParamSpec]:
    """Resolve the active scaffolding ParamSpec pack for a `scaffolding` value.

    "off" -> [], "live" -> nav+shaping (3), "full" -> the 17-param FTC pack.
    Raises KeyError on any other value (caught at config load with a clearer
    message).
    """
    return {
        "off": [],
        "live": _NN_LIVE_PARAMS,
        "full": _NN_SCAFFOLDING_PARAMS,
    }[scaffolding]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_nn_scaffolding_params.py -v`
Expected: PASS (all, including the pre-existing 17-param tests)

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py tests/test_nn_scaffolding_params.py
git commit -m "feat(training): add _NN_LIVE_PARAMS + active_scaffolding_specs resolver"
```

---

### Task 2: `build_default_scaffolding_slab` (no-FTC seeding for `live`)

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (after `build_scaffolding_initial_slab`, line 337)
- Test: `tests/test_warm_start_scaffolding_seed.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_warm_start_scaffolding_seed.py`:

```python
def test_build_default_scaffolding_slab_no_file(tmp_path: Path) -> None:
    """live seeding builds the slab from ParamSpec defaults, touching no file."""
    import numpy as np

    from aerocapture.training.encoding import encode_to_normalized
    from aerocapture.training.param_spaces import _NN_LIVE_PARAMS
    from aerocapture.training.train import build_default_scaffolding_slab

    rng = np.random.default_rng(0)
    slab = build_default_scaffolding_slab(_NN_LIVE_PARAMS, n_pop=8, rng=rng, jitter=0.0)

    assert slab.shape == (8, 3)
    expected_row = encode_to_normalized({s.name: s.default for s in _NN_LIVE_PARAMS}, list(_NN_LIVE_PARAMS))
    for row in slab:
        np.testing.assert_allclose(row, expected_row)
    assert slab.min() >= 0.0 and slab.max() <= 1.0


def test_build_default_scaffolding_slab_jitter_bounds() -> None:
    import numpy as np

    from aerocapture.training.param_spaces import _NN_LIVE_PARAMS
    from aerocapture.training.train import build_default_scaffolding_slab

    rng = np.random.default_rng(1)
    slab = build_default_scaffolding_slab(_NN_LIVE_PARAMS, n_pop=100, rng=rng, jitter=0.02)
    assert slab.shape == (100, 3)
    assert slab.min() >= 0.0 and slab.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_warm_start_scaffolding_seed.py::test_build_default_scaffolding_slab_no_file -v`
Expected: FAIL with `ImportError: cannot import name 'build_default_scaffolding_slab'`

- [ ] **Step 3: Add the helper**

In `train.py`, immediately after `build_scaffolding_initial_slab` (after its `return slab` at line 337), insert:

```python
def build_default_scaffolding_slab(
    scaffolding_specs: list[ParamSpec],
    n_pop: int,
    rng: np.random.Generator,
    jitter: float = 0.02,
) -> npt.NDArray[np.float64]:
    """Seed a scaffolding slab from each spec's default (no FTC file read).

    Mirrors `build_scaffolding_initial_slab`'s shape/jitter contract but sources
    the center from `ParamSpec.default` instead of an FTC JSON. Used for
    `scaffolding = "live"`, where the params have standalone defaults and no
    FTC dependency.
    """
    from aerocapture.training.encoding import encode_to_normalized

    center = encode_to_normalized({s.name: s.default for s in scaffolding_specs}, list(scaffolding_specs))
    slab = np.tile(center, (n_pop, 1))
    if jitter > 0.0:
        slab = slab + rng.normal(0.0, jitter, size=slab.shape)
        slab = np.clip(slab, 0.0, 1.0)
    return slab
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_warm_start_scaffolding_seed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_warm_start_scaffolding_seed.py
git commit -m "feat(training): add build_default_scaffolding_slab for live seeding"
```

---

### Task 3: `NetworkConfig.scaffolding` field + validation

**Files:**
- Modify: `src/python/aerocapture/training/config.py:34` (field) and `:38` (`__post_init__`)
- Test: `tests/test_nn_optimize_scaffolding_specs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nn_optimize_scaffolding_specs.py`:

```python
def test_network_config_scaffolding_field_default() -> None:
    from aerocapture.training.config import NetworkConfig

    cfg = NetworkConfig(architecture=[{"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"}])
    assert cfg.scaffolding == "off"


def test_network_config_rejects_unknown_scaffolding() -> None:
    import pytest

    from aerocapture.training.config import NetworkConfig

    with pytest.raises(ValueError, match="scaffolding must be"):
        NetworkConfig(
            architecture=[{"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"}],
            scaffolding="partial",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_nn_optimize_scaffolding_specs.py::test_network_config_scaffolding_field_default -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'scaffolding'` (field not defined yet)

- [ ] **Step 3: Rename the field**

In `config.py`, change line 34 from:

```python
    optimize_scaffolding: bool = False
```

to:

```python
    scaffolding: str = "off"  # "off" | "live" | "full"
```

- [ ] **Step 4: Validate the value in `__post_init__`**

In `config.py`, add the validation as the FIRST lines of `__post_init__` (right after the `def __post_init__(self) -> None:` at line 38, before the `if self.architecture is not None:` block — the architecture branch returns early, so this must precede it):

```python
        if self.scaffolding not in ("off", "live", "full"):
            msg = f"scaffolding must be 'off', 'live', or 'full', got {self.scaffolding!r}"
            raise ValueError(msg)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_nn_optimize_scaffolding_specs.py::test_network_config_scaffolding_field_default tests/test_nn_optimize_scaffolding_specs.py::test_network_config_rejects_unknown_scaffolding -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/config.py tests/test_nn_optimize_scaffolding_specs.py
git commit -m "feat(config): NetworkConfig.scaffolding str field replaces optimize_scaffolding"
```

---

### Task 4: Rewire all `train.py` scaffolding sites to the resolver

This is the cohesive rename: after Task 3 the field no longer exists, so every `config.network.optimize_scaffolding` reference and every `17 if … else 0` must change together. Each edit below is exact old → new with enough context for uniqueness.

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (sites at lines 1956, 213, 376, 497, 742, 916, 957, 1802, 2209)

- [ ] **Step 1: TOML parse (line 1956-1957)**

Old:
```python
    if "optimize_scaffolding" in _gnn:
        cfg.network.optimize_scaffolding = bool(_gnn["optimize_scaffolding"])
```
New:
```python
    if "scaffolding" in _gnn:
        cfg.network.scaffolding = str(_gnn["scaffolding"])
```

- [ ] **Step 2: Resume-mismatch message (line 213)**

Old:
```python
            f"`[guidance.neural_network] optimize_scaffolding` or "
```
New:
```python
            f"`[guidance.neural_network] scaffolding` or "
```

- [ ] **Step 3: `_evaluate_pool` scaffolding pull (lines 376-386)**

Old:
```python
        if config.network.optimize_scaffolding:
            # Pull the scaffolding values from FTC's best_params.json so the
            # eval runs with the same scaffolding the chromosome will carry.
            ftc_path = Path("training_output/ftc/best_params.json")
            if ftc_path.exists():
                ftc_params = json.loads(ftc_path.read_text())
                from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

                for spec in _NN_SCAFFOLDING_PARAMS:
                    if spec.name in ftc_params:
                        decoded_params[spec.name] = float(ftc_params[spec.name])
```
New:
```python
        from aerocapture.training.param_spaces import active_scaffolding_specs

        _eval_pack = active_scaffolding_specs(config.network.scaffolding)
        if config.network.scaffolding == "full":
            # Pull the scaffolding values from FTC's best_params.json so the
            # eval runs with the same scaffolding the chromosome will carry.
            ftc_path = Path("training_output/ftc/best_params.json")
            if ftc_path.exists():
                ftc_params = json.loads(ftc_path.read_text())
                for spec in _eval_pack:
                    if spec.name in ftc_params:
                        decoded_params[spec.name] = float(ftc_params[spec.name])
        elif config.network.scaffolding == "live":
            # live tail is seeded from defaults; eval with the same.
            for spec in _eval_pack:
                decoded_params[spec.name] = float(spec.default)
```

- [ ] **Step 4: best_model write — single-algo (lines 497, 507-510)**

Old (line 497):
```python
            n_scaff = 17 if config.network.optimize_scaffolding else 0
```
New:
```python
            from aerocapture.training.param_spaces import active_scaffolding_specs

            _pack = active_scaffolding_specs(config.network.scaffolding)
            n_scaff = len(_pack)
```

Old (lines 507-510, inside `if n_scaff > 0:`):
```python
                from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

                scaff_params = decode_normalized(best_individual[n_weights:], list(_NN_SCAFFOLDING_PARAMS))
                for s in _NN_SCAFFOLDING_PARAMS:
```
New:
```python
                scaff_params = decode_normalized(best_individual[n_weights:], list(_pack))
                for s in _pack:
```

- [ ] **Step 5: param_specs assembly (lines 742-748)**

Old:
```python
        if config.network.optimize_scaffolding:
            if config.network.architecture is None:
                msg = "optimize_scaffolding=true requires v2 [[network.architecture]]; v1 layer_sizes/activations is not supported. Convert your config."
                raise ValueError(msg)
            from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

            param_specs = [*param_specs, *_NN_SCAFFOLDING_PARAMS]
```
New:
```python
        if config.network.scaffolding != "off":
            if config.network.architecture is None:
                msg = "scaffolding != 'off' requires v2 [[network.architecture]]; v1 layer_sizes/activations is not supported. Convert your config."
                raise ValueError(msg)
            from aerocapture.training.param_spaces import active_scaffolding_specs

            param_specs = [*param_specs, *active_scaffolding_specs(config.network.scaffolding)]
```

- [ ] **Step 6: initial scaffolding slab branch (lines 915-930)**

Old:
```python
            scaffolding_slab = None
            if config.network.optimize_scaffolding:
                from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

                # _NN_SCAFFOLDING_PARAMS are FTC's bounds; the seed always comes
                # from FTC's best_params.json. warm_start_from is independent --
                # it points at a behavioural-cloning source (any unsigned-magnitude
                # scheme), not at a scaffolding source.
                ftc_params_path = "training_output/ftc/best_params.json"
                scaffolding_slab = build_scaffolding_initial_slab(
                    ftc_params_path,
                    list(_NN_SCAFFOLDING_PARAMS),
                    config.optimizer.n_pop,
                    rng,
                    jitter=config.warm_start.jitter,
                )
```
New:
```python
            scaffolding_slab = None
            if config.network.scaffolding != "off":
                from aerocapture.training.param_spaces import active_scaffolding_specs

                _slab_pack = active_scaffolding_specs(config.network.scaffolding)
                if config.network.scaffolding == "full":
                    # full pack is seeded from FTC's best_params.json (FTC's bounds,
                    # FTC's optimum). warm_start_from is independent -- it points at
                    # a behavioural-cloning source, not a scaffolding source.
                    scaffolding_slab = build_scaffolding_initial_slab(
                        "training_output/ftc/best_params.json",
                        list(_slab_pack),
                        config.optimizer.n_pop,
                        rng,
                        jitter=config.warm_start.jitter,
                    )
                else:
                    # live pack: 3 params seeded from their defaults, no FTC dep.
                    scaffolding_slab = build_default_scaffolding_slab(
                        list(_slab_pack),
                        config.optimizer.n_pop,
                        rng,
                        jitter=config.warm_start.jitter,
                    )
```

- [ ] **Step 7: warm-start n_scaff (line 957)**

Old:
```python
                n_scaff = 17 if config.network.optimize_scaffolding else 0
```
New:
```python
                from aerocapture.training.param_spaces import active_scaffolding_specs

                n_scaff = len(active_scaffolding_specs(config.network.scaffolding))
```

- [ ] **Step 8: islands best_model write (lines 1802, 1816-1822)**

Old (line 1802):
```python
        n_scaff = 17 if config.network.optimize_scaffolding else 0
```
New:
```python
        from aerocapture.training.param_spaces import active_scaffolding_specs

        _pack = active_scaffolding_specs(config.network.scaffolding)
        n_scaff = len(_pack)
```

Old (lines 1816-1822):
```python
            from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS  # noqa: PLC0415

            scaff_params = decode_normalized(
                best_individual[n_weights:],
                list(_NN_SCAFFOLDING_PARAMS),
            )
            for s in _NN_SCAFFOLDING_PARAMS:
```
New:
```python
            scaff_params = decode_normalized(
                best_individual[n_weights:],
                list(_pack),
            )
            for s in _pack:
```

- [ ] **Step 9: resume/final best_model write (lines 2209, 2216-2219)**

Old (line 2209):
```python
            n_scaff = 17 if cfg.network.optimize_scaffolding else 0
```
New:
```python
            from aerocapture.training.param_spaces import active_scaffolding_specs

            _pack = active_scaffolding_specs(cfg.network.scaffolding)
            n_scaff = len(_pack)
```

Old (lines 2216-2219):
```python
                from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

                scaff_params = decode_normalized(result["best_individual"][n_weights:], list(_NN_SCAFFOLDING_PARAMS))
                for s in _NN_SCAFFOLDING_PARAMS:
```
New:
```python
                scaff_params = decode_normalized(result["best_individual"][n_weights:], list(_pack))
                for s in _pack:
```

- [ ] **Step 10: Verify train.py imports cleanly**

Run: `uv run python -c "import aerocapture.training.train"`
Expected: no output, exit 0 (no NameError / leftover `optimize_scaffolding` reference)

Run: `grep -n "optimize_scaffolding" src/python/aerocapture/training/train.py`
Expected: no matches

- [ ] **Step 11: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "refactor(training): route train.py scaffolding sites through active_scaffolding_specs"
```

---

### Task 5: `problem.py` weight-spec count via resolver

**Files:**
- Modify: `src/python/aerocapture/training/problem.py:54-60`
- Test: `tests/test_problem.py` (or wherever AerocaptureProblem is tested)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_problem.py` (create the test alongside existing problem tests; if the file does not exist, add to `tests/test_nn_optimize_scaffolding_specs.py`):

```python
def test_problem_n_nn_weight_specs_live_pack() -> None:
    """A live-scaffolding NN problem caps weights at len(param_specs) - 3."""
    import numpy as np

    from aerocapture.training.config import NetworkConfig
    from aerocapture.training.param_spaces import ParamSpec
    from aerocapture.training.problem import AerocaptureProblem

    arch = [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "tanh"}]
    net = NetworkConfig(architecture=arch, scaffolding="live")
    # 3 weight specs (placeholder) + 3 live scaffolding specs
    specs = [ParamSpec(f"w{i}", -1.0, 1.0, 0.0) for i in range(3)] + [
        ParamSpec("nav.density_filter_gain", 0.3, 1.0, 0.8),
        ParamSpec("nav.density_gain_max_delta", 0.01, 0.5, 0.1),
        ParamSpec("shaping.max_bank_acceleration", 2.0, 15.0, 5.0),
    ]
    prob = AerocaptureProblem(
        param_specs=specs,
        toml_path="configs/training/msr_aller_gru_pso_train.toml",
        scheme="neural_network",
        seeds=[0],
        cost_kwargs={},
        nn_config=net,
    )
    assert prob._n_nn_weight_specs == 3
```

(Adjust the `AerocaptureProblem(...)` kwargs to match its actual signature — check `src/python/aerocapture/training/problem.py` constructor before writing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_problem.py::test_problem_n_nn_weight_specs_live_pack -v`
Expected: FAIL — currently `_n_nn_weight_specs` subtracts `17 if opt_scaff else 0`; with `scaffolding="live"` the old `getattr(nn_config, "optimize_scaffolding", False)` is False, so it subtracts 0 and the assert (`== 3`) fails (gets 6).

- [ ] **Step 3: Rewire the count (lines 54-60)**

Old:
```python
        # NN+optimize_scaffolding: chromosome layout is [NN weights..., 17 scaffolding...].
        # _n_nn_weight_specs caps the slice that gets fed to write_nn_json so the flat
        # weight vector matches the network's actual parameter count. Without this,
        # write_nn_json gets the full chromosome and from_flat_weights_v2 errors with
        # "weight vector length mismatch".
        opt_scaff = bool(getattr(nn_config, "optimize_scaffolding", False)) if nn_config is not None else False
        self._n_nn_weight_specs = len(param_specs) - (17 if opt_scaff else 0)
```
New:
```python
        # NN scaffolding: chromosome layout is [NN weights..., scaffolding tail...].
        # _n_nn_weight_specs caps the slice fed to write_nn_json so the flat weight
        # vector matches the network's actual parameter count. Without this,
        # write_nn_json gets the full chromosome and from_flat_weights_v2 errors with
        # "weight vector length mismatch". Tail width = len(active scaffolding pack).
        from aerocapture.training.param_spaces import active_scaffolding_specs

        _scaffolding = getattr(nn_config, "scaffolding", "off") if nn_config is not None else "off"
        self._n_nn_weight_specs = len(param_specs) - len(active_scaffolding_specs(_scaffolding))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_problem.py::test_problem_n_nn_weight_specs_live_pack -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/problem.py tests/test_problem.py
git commit -m "refactor(training): problem._n_nn_weight_specs via active_scaffolding_specs"
```

---

### Task 6: `warm_start.py` cache key + 3-way scaffolding tail

**Files:**
- Modify: `src/python/aerocapture/training/warm_start.py:73` (cache key) and `:847-851` (tail)
- Test: `tests/test_warm_start_pipeline.py`

- [ ] **Step 1: Update the cache-key test**

In `tests/test_warm_start_pipeline.py`, find the existing assertions:

```python
    assert "n" in cache_key, f"cache key missing n: {cache_key}"
    assert cache_key["n"] is True
```

(these are the mangled-display form of `"optimize_scaffolding"` assertions; locate the real lines `assert "optimize_scaffolding" in cache_key` / `assert cache_key["optimize_scaffolding"] is True`)

Replace with:

```python
    assert "scaffolding" in cache_key, f"cache key missing scaffolding: {cache_key}"
    assert cache_key["scaffolding"] == "full"
```

Also update any test helper that constructs a config with `optimize_scaffolding=True` to use `scaffolding="full"` (and `optimize_scaffolding=False` → `scaffolding="off"`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_warm_start_pipeline.py -k cache -v`
Expected: FAIL — cache key still has `optimize_scaffolding`, not `scaffolding`.

- [ ] **Step 3: Update the cache key (line 73)**

Old:
```python
        "optimize_scaffolding": bool(cfg.network.optimize_scaffolding),
```
New:
```python
        "scaffolding": cfg.network.scaffolding,
```

- [ ] **Step 4: 3-way scaffolding tail (lines 847-851)**

Old:
```python
    chromo = weight_chromo
    if network.optimize_scaffolding:
        with open(scaffolding_source_path) as f:
            scaff_params = json.load(f)
        scaff_chromo = encode_to_normalized(scaff_params, list(_NN_SCAFFOLDING_PARAMS))
        chromo = np.concatenate([weight_chromo, scaff_chromo])
```
New:
```python
    chromo = weight_chromo
    if network.scaffolding != "off":
        from aerocapture.training.param_spaces import active_scaffolding_specs

        pack = active_scaffolding_specs(network.scaffolding)
        if network.scaffolding == "full":
            with open(scaffolding_source_path) as f:
                scaff_params = json.load(f)
        else:  # live: seed the 3-param tail from defaults, no FTC source needed.
            scaff_params = {s.name: s.default for s in pack}
        scaff_chromo = encode_to_normalized(scaff_params, list(pack))
        chromo = np.concatenate([weight_chromo, scaff_chromo])
```

- [ ] **Step 5: Update the docstring/comment references**

In `warm_start.py`, the `_cache_key` docstring (lines 41-43) and the comment at line 679 mention `optimize_scaffolding`. Update the prose to `scaffolding` for accuracy (no behavior change). The module-level `_INTEGER_PARAM_NAMES` (line 137) stays on `_NN_SCAFFOLDING_PARAMS` — it is a membership set of integer-typed names; live params are floats, so the full set is a harmless superset.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_warm_start_pipeline.py -v`
Expected: PASS

Run: `grep -n "optimize_scaffolding" src/python/aerocapture/training/warm_start.py`
Expected: no matches

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/warm_start.py tests/test_warm_start_pipeline.py
git commit -m "refactor(training): warm_start cache key + 3-way scaffolding tail"
```

---

### Task 7: Startup visibility print

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (in the param_specs-assembly block edited in Task 4 Step 5, gated on `verbose`)

- [ ] **Step 1: Add the print after the param_specs assembly**

In `train.py`, immediately after the `param_specs = [*param_specs, *active_scaffolding_specs(...)]` line from Task 4 Step 5 (still inside the `if config.network.scaffolding != "off":` block), append:

```python
            if verbose:
                if config.network.scaffolding == "live":
                    print("scaffolding optimization: LIVE — 3 params (nav density filter ×2, command shaping); no FTC dependency")
                else:  # full
                    print("scaffolding optimization: FULL — 17 params, seeded from training_output/ftc/best_params.json")
```

Then add the `"off"` case. Right after the `if config.network.scaffolding != "off":` block closes (same `if/elif` chain as the NN param_specs assembly), add:

```python
        elif verbose and config.guidance_type == "neural_network" and config.network.architecture is not None:
            print("scaffolding optimization: OFF — NN weights only")
```

(Confirm `verbose` is in scope here — it is the same `verbose` used by the piecewise print at line 754.)

- [ ] **Step 2: Smoke-check the print path**

Run:
```bash
uv run python -c "
from aerocapture.training.config import NetworkConfig
n = NetworkConfig(architecture=[{'type':'dense','input_size':2,'output_size':1,'activation':'tanh'}], scaffolding='live')
print(n.scaffolding)
"
```
Expected: `live`

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(training): print scaffolding optimization mode at training start"
```

---

### Task 8: Migrate configs (hard cut)

**Files:**
- Modify: `configs/training/nn_common.toml`, `msr_aller_gru_pso_train.toml`, `msr_aller_lstm_pso_train.toml`, `msr_aller_gru_pso_magonly_train.toml`, `msr_aller_nn_joint_train.toml`

- [ ] **Step 1: Drop the flag from the shared base**

In `configs/training/nn_common.toml`, remove these two lines:
```toml
[guidance.neural_network]
optimize_scaffolding = true
```
Leaving the file as just the `[guidance] type = "neural_network"` block (plus its header comment).

- [ ] **Step 2: gru_pso + lstm_pso → live**

In `configs/training/msr_aller_gru_pso_train.toml`, add (create the section if absent — gru_pso currently has no `[guidance.neural_network]` block):
```toml
[guidance.neural_network]
scaffolding = "live"
```

In `configs/training/msr_aller_lstm_pso_train.toml`, add the same block:
```toml
[guidance.neural_network]
scaffolding = "live"
```

- [ ] **Step 3: magonly + nn_joint → full**

In `configs/training/msr_aller_gru_pso_magonly_train.toml`, add `scaffolding = "full"` under the existing `[guidance.neural_network]` block (it already sets `mode = "magnitude_only"`).

In `configs/training/msr_aller_nn_joint_train.toml`, replace `optimize_scaffolding = true` with `scaffolding = "full"` under its `[guidance.neural_network]` block.

- [ ] **Step 4: Verify no config still sets the old flag**

Run: `grep -rn "optimize_scaffolding" configs/`
Expected: no matches

- [ ] **Step 5: Verify configs still load (Rust + Python)**

Run:
```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
for c in ['gru_pso','lstm_pso','gru_pso_magonly','nn_joint']:
    t = load_toml_with_bases(f'configs/training/msr_aller_{c}_train.toml')
    print(c, t['guidance']['neural_network'].get('scaffolding'))
"
```
Expected:
```
gru_pso live
lstm_pso live
gru_pso_magonly full
nn_joint full
```

- [ ] **Step 6: Commit**

```bash
git add configs/training/nn_common.toml configs/training/msr_aller_gru_pso_train.toml configs/training/msr_aller_lstm_pso_train.toml configs/training/msr_aller_gru_pso_magonly_train.toml configs/training/msr_aller_nn_joint_train.toml
git commit -m "chore(configs): migrate optimize_scaffolding -> scaffolding (live for full_neural, full for magnitude_only)"
```

---

### Task 9: `live` integration test + sweep remaining old-flag references

**Files:**
- Test: `tests/test_nn_optimize_scaffolding_specs.py`
- Modify: any remaining test files referencing `optimize_scaffolding`

- [ ] **Step 1: Add a live param_specs-assembly integration test**

Append to `tests/test_nn_optimize_scaffolding_specs.py`:

```python
def test_live_appends_three_specs_no_ftc(tmp_path: Path, monkeypatch) -> None:
    """scaffolding='live' adds exactly 3 specs and never reads the FTC file."""
    import numpy as np

    from aerocapture.training.param_spaces import active_scaffolding_specs
    from aerocapture.training.train import build_default_scaffolding_slab

    # Guard: building the live slab must not touch the FTC path.
    def _boom(*a, **k):
        raise AssertionError("live seeding must not read FTC best_params.json")

    monkeypatch.setattr("aerocapture.training.train.build_scaffolding_initial_slab", _boom)

    pack = active_scaffolding_specs("live")
    assert len(pack) == 3
    slab = build_default_scaffolding_slab(list(pack), n_pop=4, rng=np.random.default_rng(0), jitter=0.0)
    assert slab.shape == (4, 3)
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_nn_optimize_scaffolding_specs.py::test_live_appends_three_specs_no_ftc -v`
Expected: PASS

- [ ] **Step 3: Sweep every remaining test reference to the old flag**

Run: `grep -rln "optimize_scaffolding" tests/`
For each file found (expected: `test_island_model.py`, `test_warm_start_optimizer_seeding.py`, `test_nn_scaffolding_params.py`, and any others), open it and:
- replace `optimize_scaffolding=True` → `scaffolding="full"`
- replace `optimize_scaffolding=False` → `scaffolding="off"`
- replace prose/docstring mentions with `scaffolding` for accuracy

Re-run: `grep -rln "optimize_scaffolding" tests/ src/`
Expected: no matches anywhere in `tests/` or `src/`.

- [ ] **Step 4: Full Python verification**

Run: `uv run pytest tests/ -q`
Expected: all pass (no errors, no failures).

Run: `./lint_code.sh`
Expected: ruff clean + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test(training): cover live scaffolding; migrate tests off optimize_scaffolding"
```

---

### Task 10: Update CLAUDE.md + finalize via smart-commit

**Files:**
- Modify: `CLAUDE.md` (the `optimize_scaffolding` description in the `[guidance.neural_network]` paragraph)

- [ ] **Step 1: Update the CLAUDE.md prose**

In `CLAUDE.md`, find the sentence describing `optimize_scaffolding = true` (the "Three additional opt-in knobs" paragraph) and rewrite it to describe the three-valued `scaffolding = "off" | "live" | "full"` knob: `"off"` (NN weights only), `"live"` (appends the 3-param nav+shaping pack, seeded from defaults, no FTC dependency — for full_neural schemes), `"full"` (appends the 17-param `_NN_SCAFFOLDING_PARAMS`, seeded from `training_output/ftc/best_params.json` + jitter — for magnitude_only). Note that the choice is printed at training start and is now declared per-leaf (not in `nn_common.toml`).

- [ ] **Step 2: Run the smart-commit skill over the whole branch**

Invoke the `smart-commit` skill, instructing it to take the whole git branch into account (sync CLAUDE.md / README.md with the codebase, then commit anything outstanding).

- [ ] **Step 3: Final verification**

Run: `uv run pytest tests/ -q && ./lint_code.sh`
Expected: all pass, lint clean.

Run: `grep -rn "optimize_scaffolding" src/ tests/ configs/ CLAUDE.md`
Expected: no matches.

---

## Self-Review notes

- **Spec coverage:** config surface (Task 3 + 8), resolver replacing scattered `17` (Tasks 1,4,5,6), seeding split FTC vs defaults (Tasks 2,4,6), startup visibility (Task 7), cache-key + resume width (Tasks 4,6), migration hard-cut (Task 8), no-FTC live guarantee (Task 9 monkeypatch), no Rust change (verified: `TomlNeuralNetworkParams` has no `deny_unknown_fields`). All covered.
- **Type consistency:** `active_scaffolding_specs(str) -> list[ParamSpec]`, `build_default_scaffolding_slab(specs, n_pop, rng, jitter) -> ndarray`, `NetworkConfig.scaffolding: str` used consistently across all tasks.
- **report.py / compare_guidance.py:** intentionally untouched — they load `best_params.json` by file presence, so a `live` deploy (3 keys) works unchanged; only stale comments mention the boolean (cosmetic, left for smart-commit's doc sync if it touches them).
