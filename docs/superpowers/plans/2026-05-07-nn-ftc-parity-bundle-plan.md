# NN-vs-FTC Parity Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 7-25% NN-vs-FTC validation-RMS gap by removing three structural asymmetries (joint-vs-frozen scaffolding, atan2-abs output waste, no warm-start) — all gated behind TOML knobs that default off so existing trained NNs and existing configs are bit-identical.

**Architecture:** Three orthogonal fixes landed in dependency order as separate commits. (A) Add the 17 FTC scaffolding params to the NN's PSO chromosome, seeded from FTC's GA optimum + jitter. (B) New Rust `OutputParam::AcosTanh` parameterization (`bank = acos(tanh(out[0]))`, single output) gated on `magnitude_only` mode, persisted in v2 JSON model files so deploys are self-describing. (C) New PyO3 `collect_supervised` helper + Python `warm_start.py` module that runs FTC over a 4M-offset reserved seed pool, supervised-trains a PyTorch V2Policy mirror to mimic FTC, and seeds the PSO initial population from the cloned chromosome + small jitter.

**Tech Stack:** Rust 2024 edition, PyO3 for bindings, Python 3.14, PyTorch (V2Policy mirror reused from RL path), pymoo PSO/GA, pytest + proptest.

**Spec:** `docs/superpowers/specs/2026-05-07-nn-ftc-parity-bundle-design.md` (commit `beaac70`).

**Branch:** `feature/magnitude_only` (current).

---

## Task 0: TODO.md marker

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Append a parity-bundle block under the most recent in-progress entry**

```markdown
### NN-vs-FTC Parity Bundle (3 fixes, TOML opt-in) [DOING 2026-05-07 on feature/magnitude_only]
- [ ] Fix A: `_NN_SCAFFOLDING_PARAMS` + `optimize_scaffolding` knob + FTC-optimum seeding for NN initial population
- [ ] Fix B: `OutputParam::AcosTanh` enum + JSON v2 round trip + config validation (last-layer width=1, activation=tanh, mode=magnitude_only)
- [ ] Fix C: `aerocapture_rs.collect_supervised` PyO3 + `warm_start.py` PyTorch supervised pre-train + chromosome cache (4M seed offset)
- [ ] New training config `msr_aller_nn_joint_train.toml` flipping all three knobs + `train_all.sh` alias

Spec: `docs/superpowers/specs/2026-05-07-nn-ftc-parity-bundle-design.md`.
Plan: `docs/superpowers/plans/2026-05-07-nn-ftc-parity-bundle-plan.md`.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs(todo): mark NN-vs-FTC parity bundle in progress on feature/magnitude_only"
```

---

# Fix A — joint scaffolding optimization

## Task A1: Add `_NN_SCAFFOLDING_PARAMS` to `param_spaces.py`

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py` (after line 66, before `PARAM_SPACES`)
- Test: `tests/test_nn_scaffolding_params.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_nn_scaffolding_params.py`:

```python
"""Regression guard: NN scaffolding param ordering must match the deploy chromosome layout."""
from __future__ import annotations

from aerocapture.training.param_spaces import (
    _EXIT_PARAMS,
    _LATERAL_PARAMS,
    _NAV_PARAMS,
    _SHAPING_PARAMS,
    _THERMAL_LIMITER_PARAMS,
    _NN_SCAFFOLDING_PARAMS,
)


def test_nn_scaffolding_is_concatenation_in_documented_order():
    expected = [*_NAV_PARAMS, *_LATERAL_PARAMS, *_EXIT_PARAMS, *_THERMAL_LIMITER_PARAMS, *_SHAPING_PARAMS]
    assert list(_NN_SCAFFOLDING_PARAMS) == expected


def test_nn_scaffolding_has_seventeen_params():
    assert len(_NN_SCAFFOLDING_PARAMS) == 17


def test_nn_scaffolding_names_are_unique_and_prefixed():
    names = [s.name for s in _NN_SCAFFOLDING_PARAMS]
    assert len(set(names)) == len(names), "duplicate names"
    valid = ("nav.", "lateral.", "exit.", "thermal.", "shaping.")
    for name in names:
        assert name.startswith(valid), f"unexpected prefix in {name!r}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_nn_scaffolding_params.py -v
```

Expected: FAIL with `ImportError: cannot import name '_NN_SCAFFOLDING_PARAMS'`.

- [ ] **Step 3: Add the constant**

In `src/python/aerocapture/training/param_spaces.py`, after the `_NAV_PARAMS` block (around line 66):

```python
# Combined scaffolding pack used when training a neural-network scheme with
# `optimize_scaffolding = true`. Same specs FTC trains, same order. The
# routing in `problem.py::_build_overrides` already handles every prefix.
_NN_SCAFFOLDING_PARAMS: list[ParamSpec] = [
    *_NAV_PARAMS,
    *_LATERAL_PARAMS,
    *_EXIT_PARAMS,
    *_THERMAL_LIMITER_PARAMS,
    *_SHAPING_PARAMS,
]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_nn_scaffolding_params.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py tests/test_nn_scaffolding_params.py
git commit -m "feat(nn): add _NN_SCAFFOLDING_PARAMS for joint-scaffolding training"
```

---

## Task A2: Add `optimize_scaffolding` config knob to `NetworkConfig`

**Files:**
- Modify: `src/python/aerocapture/training/config.py` (the `NetworkConfig` dataclass — find via `grep -n "class NetworkConfig" src/python/aerocapture/training/config.py`)
- Modify: `src/python/aerocapture/training/train.py` (the `_net = _toml_data.get("network", {})` block around line 888-899)

- [ ] **Step 1: Add the field to `NetworkConfig`**

In `src/python/aerocapture/training/config.py`, find the `NetworkConfig` dataclass. Add a new field with a default:

```python
optimize_scaffolding: bool = False
```

- [ ] **Step 2: Wire the TOML key in `train.py`**

In `src/python/aerocapture/training/train.py`, after the existing `if "input_mask" in _net:` block (around line 896-897), and before the `if cfg.network.architecture is not None:` post-init call, add:

```python
    _gnn = _toml_data.get("guidance", {}).get("neural_network", {})
    if "optimize_scaffolding" in _gnn:
        cfg.network.optimize_scaffolding = bool(_gnn["optimize_scaffolding"])
```

- [ ] **Step 3: Sanity check**

```bash
uv run python -c "from aerocapture.training.config import NetworkConfig; n = NetworkConfig(); assert n.optimize_scaffolding is False; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/config.py src/python/aerocapture/training/train.py
git commit -m "feat(nn): add optimize_scaffolding knob to NetworkConfig + train.py TOML wiring"
```

---

## Task A3: Extend NN ParamSpec list when `optimize_scaffolding`

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (the param-spec branch around lines 315-333)
- Test: `tests/test_nn_optimize_scaffolding_specs.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_nn_optimize_scaffolding_specs.py`:

```python
"""train.py NN-branch param-spec list must include scaffolding when knob is on."""
from __future__ import annotations

from aerocapture.training.encoding import nn_param_specs_from_v2
from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS


def _toy_arch() -> list[dict]:
    return [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "swish"},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "asinh"},
    ]


def test_specs_include_scaffolding_when_knob_on():
    from pydantic import TypeAdapter
    from aerocapture.training.rl.schemas import LayerSpec

    arch = _toy_arch()
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    base_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)

    full_specs = [*base_specs, *_NN_SCAFFOLDING_PARAMS]

    assert len(full_specs) == len(base_specs) + 17
    tail_names = [s.name for s in full_specs[len(base_specs):]]
    expected_names = [s.name for s in _NN_SCAFFOLDING_PARAMS]
    assert tail_names == expected_names


def test_specs_unchanged_when_knob_off():
    from pydantic import TypeAdapter
    from aerocapture.training.rl.schemas import LayerSpec

    arch = _toy_arch()
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    base_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)
    assert len(base_specs) > 0
```

- [ ] **Step 2: Run the test (should pass — it asserts a logic-level invariant)**

```bash
uv run pytest tests/test_nn_optimize_scaffolding_specs.py -v
```

Expected: 2 PASS.

- [ ] **Step 3: Apply the same logic in `train.py`**

In `src/python/aerocapture/training/train.py`, replace the NN param-spec branch (around lines 315-331) with:

```python
    if config.guidance_type == "neural_network":
        if config.network.architecture is not None:
            from pydantic import TypeAdapter

            from aerocapture.training.rl.schemas import LayerSpec

            specs_adapter = TypeAdapter(list[LayerSpec])
            validated = specs_adapter.validate_python(config.network.architecture)
            param_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)
        else:
            param_specs = nn_param_specs_from_architecture(
                config.network.layer_sizes,
                config.network.activations,
            )

        if config.network.optimize_scaffolding:
            if config.network.architecture is None:
                msg = (
                    "optimize_scaffolding=true requires v2 [[network.architecture]]; "
                    "v1 layer_sizes/activations is not supported. Convert your config."
                )
                raise ValueError(msg)
            from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

            param_specs = [*param_specs, *_NN_SCAFFOLDING_PARAMS]
    else:
        param_specs = PARAM_SPACES[config.guidance_type]
```

- [ ] **Step 4: Re-run the test**

```bash
uv run pytest tests/test_nn_optimize_scaffolding_specs.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_nn_optimize_scaffolding_specs.py
git commit -m "feat(nn): extend ParamSpec list with scaffolding when optimize_scaffolding=true"
```

---

## Task A4: Add FTC-optimum seeding helper

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (add `build_scaffolding_initial_slab` near `build_initial_population_for_v2`)
- Test: `tests/test_warm_start_scaffolding_seed.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_warm_start_scaffolding_seed.py`:

```python
"""build_scaffolding_initial_slab seeds FTC's best_params + jitter into the chromosome."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS
from aerocapture.training.train import build_scaffolding_initial_slab


def test_seed_centers_at_ftc_optimum(tmp_path: Path):
    ftc_params = {s.name: s.default for s in _NN_SCAFFOLDING_PARAMS}
    ftc_path = tmp_path / "best_params.json"
    ftc_path.write_text(json.dumps(ftc_params))

    rng = np.random.default_rng(0)
    n_pop = 32
    slab = build_scaffolding_initial_slab(ftc_path, _NN_SCAFFOLDING_PARAMS, n_pop, rng, jitter=0.0)

    assert slab.shape == (n_pop, 17)
    from aerocapture.training.encoding import encode_to_normalized
    expected_row = encode_to_normalized(ftc_params, list(_NN_SCAFFOLDING_PARAMS))
    np.testing.assert_allclose(slab[0], expected_row, atol=1e-15)
    np.testing.assert_allclose(slab[-1], expected_row, atol=1e-15)


def test_seed_jitter_keeps_values_in_unit_box(tmp_path: Path):
    ftc_params = {s.name: s.default for s in _NN_SCAFFOLDING_PARAMS}
    ftc_path = tmp_path / "best_params.json"
    ftc_path.write_text(json.dumps(ftc_params))

    rng = np.random.default_rng(0)
    slab = build_scaffolding_initial_slab(ftc_path, _NN_SCAFFOLDING_PARAMS, 100, rng, jitter=0.02)

    assert slab.shape == (100, 17)
    assert (slab >= 0.0).all() and (slab <= 1.0).all()


def test_missing_ftc_params_fails_loud(tmp_path: Path):
    rng = np.random.default_rng(0)
    missing = tmp_path / "absent.json"
    try:
        build_scaffolding_initial_slab(missing, _NN_SCAFFOLDING_PARAMS, 4, rng, jitter=0.02)
    except FileNotFoundError as e:
        assert "absent.json" in str(e)
        return
    raise AssertionError("expected FileNotFoundError for missing FTC params")
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_warm_start_scaffolding_seed.py -v
```

Expected: FAIL with `ImportError: cannot import name 'build_scaffolding_initial_slab'`.

- [ ] **Step 3: Implement the helper**

In `src/python/aerocapture/training/train.py`, after `build_initial_population_for_v2` (around line 95):

```python
def build_scaffolding_initial_slab(
    ftc_params_path: str | Path,
    scaffolding_specs: list[ParamSpec],
    n_pop: int,
    rng: np.random.Generator,
    jitter: float = 0.02,
) -> npt.NDArray[np.float64]:
    """Seed the scaffolding slab of the PSO chromosome from FTC's GA optimum.

    Reads `<ftc_params_path>` (a JSON file with the same keys FTC writes,
    e.g. "lateral.tau", "exit.exit_pdyn_margin", ...), encodes each value
    to its [0, 1] slot via `encode_to_normalized`, replicates `n_pop`
    times, then adds `N(0, jitter)` per-individual noise clipped to [0, 1].

    Raises FileNotFoundError if `ftc_params_path` does not exist.
    Raises KeyError if any scaffolding spec name is missing from the JSON.
    """
    from aerocapture.training.encoding import encode_to_normalized

    ftc_params_path = Path(ftc_params_path)
    if not ftc_params_path.exists():
        msg = (
            f"optimize_scaffolding requires a source params file; '{ftc_params_path}' "
            f"does not exist. Run FTC training first (./train_all.sh ftc) or correct the path."
        )
        raise FileNotFoundError(msg)

    with open(ftc_params_path) as f:
        ftc_params: dict[str, float] = json.load(f)

    spec_names = {s.name for s in scaffolding_specs}
    missing = spec_names - set(ftc_params.keys())
    if missing:
        msg = (
            f"FTC params file '{ftc_params_path}' missing scaffolding keys: "
            f"{sorted(missing)}. Re-run FTC training so its best_params.json includes them."
        )
        raise KeyError(msg)

    center = encode_to_normalized(ftc_params, list(scaffolding_specs))
    slab = np.tile(center, (n_pop, 1))
    if jitter > 0.0:
        slab = slab + rng.normal(0.0, jitter, size=slab.shape)
        slab = np.clip(slab, 0.0, 1.0)
    return slab
```

- [ ] **Step 4: Re-run the test**

```bash
uv run pytest tests/test_warm_start_scaffolding_seed.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_warm_start_scaffolding_seed.py
git commit -m "feat(nn): build_scaffolding_initial_slab encodes FTC optimum + jitter"
```

---

## Task A5: Wire scaffolding seed into `build_initial_population_for_v2`

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (`build_initial_population_for_v2` around line 75 and its caller around line 467-475)

- [ ] **Step 1: Modify `build_initial_population_for_v2` signature**

Replace the existing function body in `src/python/aerocapture/training/train.py`:

```python
def build_initial_population_for_v2(
    architecture: list[dict],
    n_pop: int,
    bound_multiplier: float,
    rng: np.random.Generator,
    param_specs: list[ParamSpec],
    scaffolding_slab: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Activation-aware initial population for v2 architectures.

    When `scaffolding_slab` is provided (shape `(n_pop, n_scaffolding)`),
    appends it as the trailing slab of every individual. Used when
    `optimize_scaffolding = true` to seed scaffolding from FTC's optimum.
    """
    physical = init_v2_population(architecture, n_pop, bound_multiplier, rng)
    n_pop_actual, n_params = physical.shape
    n_scaff = 0 if scaffolding_slab is None else scaffolding_slab.shape[1]
    n_weight_specs = len(param_specs) - n_scaff
    assert n_params == n_weight_specs, (
        f"init_v2_population produced {n_params} params, ParamSpec has "
        f"{n_weight_specs} weight specs (total {len(param_specs)}, scaff {n_scaff})"
    )
    normalized = np.empty((n_pop_actual, len(param_specs)), dtype=np.float64)
    for j in range(n_weight_specs):
        s = param_specs[j]
        normalized[:, j] = np.clip((physical[:, j] - s.p_min) / (s.p_max - s.p_min), 0.0, 1.0)
    if scaffolding_slab is not None:
        normalized[:, n_weight_specs:] = scaffolding_slab
    return normalized
```

- [ ] **Step 2: Update the caller**

Replace the v2 NN initial-population branch (around line 467-475) with:

```python
        elif config.guidance_type == "neural_network" and config.network.architecture is not None:
            scaffolding_slab = None
            if config.network.optimize_scaffolding:
                from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

                ftc_params_path = (
                    _toml.get("guidance", {}).get("neural_network", {}).get("warm_start_from")
                    or "training_output/ftc/best_params.json"
                )
                scaffolding_slab = build_scaffolding_initial_slab(
                    ftc_params_path,
                    list(_NN_SCAFFOLDING_PARAMS),
                    config.optimizer.n_pop,
                    rng,
                    jitter=0.02,
                )
            pop_array = build_initial_population_for_v2(
                config.network.architecture,
                config.optimizer.n_pop,
                bound_multiplier=2.0,
                rng=rng,
                param_specs=param_specs,
                scaffolding_slab=scaffolding_slab,
            )
```

- [ ] **Step 3: Run tests to confirm no regression**

```bash
uv run pytest tests/test_warm_start_scaffolding_seed.py tests/test_nn_optimize_scaffolding_specs.py tests/test_nn_scaffolding_params.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(nn): seed scaffolding slab from FTC optimum in initial population"
```

---

## Task A6: Write `best_params.json` alongside `best_model.json` when `optimize_scaffolding`

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (`save_checkpoint` lines 146-157 and the end-of-training block lines 1076-1088)

- [ ] **Step 1: Modify `save_checkpoint` NN branch**

Replace the NN-branch in `save_checkpoint` (around lines 148-153) with:

```python
    if best_individual is not None:
        if config.guidance_type == "neural_network":
            n_scaff = 17 if config.network.optimize_scaffolding else 0
            n_weights = len(param_specs) - n_scaff
            weights = _decode_nn_weights(best_individual[:n_weights], param_specs[:n_weights])
            write_nn_json(weights, config.network, save_dir / "best_model.json", input_mask=config.network.input_mask)
            if cwd is not None:
                nn_path = Path(cwd) / config.sim.nn_param_file
                write_nn_json(weights, config.network, nn_path, input_mask=config.network.input_mask)
            if n_scaff > 0:
                from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

                scaff_params = decode_normalized(
                    best_individual[n_weights:], list(_NN_SCAFFOLDING_PARAMS)
                )
                for s in _NN_SCAFFOLDING_PARAMS:
                    if s.is_integer and s.name in scaff_params:
                        scaff_params[s.name] = int(round(scaff_params[s.name]))
                with open(save_dir / "best_params.json", "w") as fp:
                    json.dump(scaff_params, fp, indent=2)
        else:
            params = decode_normalized(best_individual, param_specs)
            with open(save_dir / "best_params.json", "w") as fp:
                json.dump(params, fp, indent=2)
```

- [ ] **Step 2: Modify the end-of-training NN block**

Find the block around line 1076-1088 in the `if __name__ == "__main__":` section. Replace with:

```python
    if result["best_individual"] is not None:
        if cfg.guidance_type == "neural_network":
            n_scaff = 17 if cfg.network.optimize_scaffolding else 0
            n_weights = len(param_specs) - n_scaff
            weights = _decode_nn_weights(result["best_individual"][:n_weights], param_specs[:n_weights])
            nn_path = Path(cwd) / cfg.sim.nn_param_file
            write_nn_json(weights, cfg.network, nn_path, input_mask=cfg.network.input_mask)
            print(f"Best weights saved to {nn_path}")
            if n_scaff > 0:
                from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

                scaff_params = decode_normalized(
                    result["best_individual"][n_weights:], list(_NN_SCAFFOLDING_PARAMS)
                )
                for s in _NN_SCAFFOLDING_PARAMS:
                    if s.is_integer and s.name in scaff_params:
                        scaff_params[s.name] = int(round(scaff_params[s.name]))
                params_path = Path(cfg.save_dir) / "best_params.json"
                with open(params_path, "w") as fp:
                    json.dump(scaff_params, fp, indent=2)
                print(f"Best scaffolding params saved to {params_path}")
        else:
            params = decode_normalized(result["best_individual"], param_specs)
            params_path = Path(cfg.save_dir) / "best_params.json"
            with open(params_path, "w") as fp:
                json.dump(params, fp, indent=2)
            print(f"Best params saved to {params_path}")
            print(f"  Params: {params}")
```

(Keep the `write_guidance_toml(...)` call below for non-NN; nothing else changes.)

- [ ] **Step 3: Smoke check**

```bash
uv run python -c "from aerocapture.training.train import save_checkpoint; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(nn): write best_params.json alongside best_model.json when optimize_scaffolding"
```

---

## Task A7: Resume shape-mismatch detection

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (helper near top + check after population restore around line 451-456)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_nn_optimize_scaffolding_specs.py`:

```python
def test_resume_with_shape_mismatch_fails_loud():
    import numpy as np

    from aerocapture.training.train import _check_resume_chromosome_shape

    saved_pop = np.zeros((4, 1266))
    try:
        _check_resume_chromosome_shape(saved_pop, expected_n_params=1283)
    except ValueError as e:
        assert "shape mismatch" in str(e).lower()
        assert "1266" in str(e) and "1283" in str(e)
        return
    raise AssertionError("expected ValueError on shape mismatch")
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_nn_optimize_scaffolding_specs.py::test_resume_with_shape_mismatch_fails_loud -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add the helper**

In `src/python/aerocapture/training/train.py`, after `_compute_fixed_seeds`:

```python
def _check_resume_chromosome_shape(
    saved_population: npt.NDArray[np.float64],
    expected_n_params: int,
) -> None:
    """Fail loudly if a resumed checkpoint's chromosome width disagrees with current ParamSpec count.

    Catches the user flipping `optimize_scaffolding` (or `output_parameterization`,
    which changes last-layer width) between training runs.
    """
    saved_n_params = saved_population.shape[1]
    if saved_n_params != expected_n_params:
        msg = (
            f"checkpoint chromosome shape mismatch: saved {saved_n_params} params, "
            f"current ParamSpec list has {expected_n_params}. This usually means "
            f"`[guidance.neural_network] optimize_scaffolding` or "
            f"`output_parameterization` was changed since the checkpoint was saved. "
            f"To resume, revert the TOML knob; to start fresh, pass --from-scratch."
        )
        raise ValueError(msg)
```

- [ ] **Step 4: Wire the check into the resume path**

After the existing `if pop_array.dtype != np.float64:` block (around line 455-456):

```python
        _check_resume_chromosome_shape(pop_array, expected_n_params=len(param_specs))
```

- [ ] **Step 5: Re-run tests**

```bash
uv run pytest tests/test_nn_optimize_scaffolding_specs.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_nn_optimize_scaffolding_specs.py
git commit -m "feat(nn): fail-loud resume detection for chromosome shape mismatch"
```

---

## Task A8: Extend `compare_guidance.py` and `report.py` to load scaffolding for NN

**Files:**
- Modify: `src/python/aerocapture/training/compare_guidance.py`
- Modify: `src/python/aerocapture/training/report.py`

- [ ] **Step 1: Locate the NN deploy path**

```bash
grep -n "best_model.json\|best_params.json" src/python/aerocapture/training/compare_guidance.py src/python/aerocapture/training/report.py | head -20
```

- [ ] **Step 2: Extend compare_guidance NN deploy**

In the section that currently routes to `best_model.json` for NN schemes, add right after the model path is computed (before the override dict is built):

```python
        scaff_path = Path(scheme_dir) / "best_params.json"
        if scaff_path.exists():
            with open(scaff_path) as f:
                scaff_params = json.load(f)
            for key, value in scaff_params.items():
                if key.startswith("lateral."):
                    overrides[f"guidance.lateral.{key.removeprefix('lateral.')}"] = value
                elif key.startswith("exit."):
                    overrides[f"guidance.ftc.{key.removeprefix('exit.')}"] = value
                elif key.startswith("nav."):
                    overrides[f"navigation.{key.removeprefix('nav.')}"] = value
                elif key.startswith("thermal."):
                    overrides[f"guidance.thermal_limiter.{key.removeprefix('thermal.')}"] = value
                elif key.startswith("shaping."):
                    overrides[f"guidance.command_shaping.{key.removeprefix('shaping.')}"] = value
                    overrides["guidance.command_shaping.enabled"] = True
```

- [ ] **Step 3: Repeat in `report.py`**

Find the equivalent NN final-eval override block in `src/python/aerocapture/training/report.py` and add the same routing.

- [ ] **Step 4: Smoke check**

```bash
uv run python -c "from aerocapture.training import compare_guidance, report; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/compare_guidance.py src/python/aerocapture/training/report.py
git commit -m "feat(nn): compare_guidance + report load NN scaffolding from best_params.json"
```

---

# Fix B — output parameterization

## Task B1: Rust `OutputParam` enum and `NeuralNetModel` field

**Files:**
- Modify: `src/rust/src/data/neural.rs` (find `pub struct NeuralNetModel`)

- [ ] **Step 1: Write failing tests**

Append to the `#[cfg(test)] mod tests` block in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn output_param_default_is_atan2_signed() {
    let p: OutputParam = OutputParam::default();
    assert_eq!(p, OutputParam::Atan2Signed);
}

#[test]
fn output_param_serde_round_trip() {
    let p = OutputParam::AcosTanh;
    let s = serde_json::to_string(&p).unwrap();
    assert_eq!(s, "\"acos_tanh\"");
    let back: OutputParam = serde_json::from_str(&s).unwrap();
    assert_eq!(back, p);

    let p2 = OutputParam::Atan2Signed;
    let s2 = serde_json::to_string(&p2).unwrap();
    assert_eq!(s2, "\"atan2_signed\"");
}
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd src/rust && cargo test --lib output_param 2>&1 | tail -20
```

Expected: FAIL — `OutputParam` not defined.

- [ ] **Step 3: Define the enum**

In `src/rust/src/data/neural.rs`, near the top after the `Activation` enum, add:

```rust
/// Output parameterization for the NN's bank-angle decoder.
///
/// `Atan2Signed` (default, backward-compatible): emits 2 outputs and
/// `bank = atan2(out[0], out[1]) ∈ (-π, π]`.
///
/// `AcosTanh`: emits 1 output through `tanh` and `bank = acos(out[0]) ∈ [0, π]`.
/// Only legal in `magnitude_only` mode (architecture validates last layer
/// `output_size = 1` with activation `tanh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputParam {
    #[default]
    Atan2Signed,
    AcosTanh,
}
```

- [ ] **Step 4: Add the field to `NeuralNetModel`**

In the `pub struct NeuralNetModel` definition, add after `ablated_input`:

```rust
    pub output_param: OutputParam,
```

- [ ] **Step 5: Update every existing constructor site**

```bash
cd src/rust && cargo build 2>&1 | grep "missing field" | sort -u
```

For every reported error, add `output_param: OutputParam::default(),` to that struct literal. Likely sites: every constructor in `data/neural.rs`, every test fixture in `gnc/guidance/neural.rs` and `gnc/guidance/dispatch.rs` and `data/nn_state.rs`.

- [ ] **Step 6: Re-run the tests**

```bash
cd src/rust && cargo test --lib output_param 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 7: Run the full Rust test suite**

```bash
cd src/rust && cargo test --lib 2>&1 | tail -5
```

Expected: all PASS (no regression).

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/data/neural.rs src/rust/src/gnc/guidance/neural.rs src/rust/src/gnc/guidance/dispatch.rs src/rust/src/data/nn_state.rs
git commit -m "feat(nn): OutputParam enum + NeuralNetModel.output_param field (default Atan2Signed)"
```

---

## Task B2: Persist `output_param` in v2 JSON save/load

**Files:**
- Modify: `src/rust/src/data/neural.rs` (`save_json` + `from_v2_json` + any other JSON constructor)

- [ ] **Step 1: Write failing tests**

Append to the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn output_param_persists_through_v2_json_round_trip() {
    let arch = vec![LayerSpec::Dense {
        input_size: 3,
        output_size: 1,
        activation: Activation::Tanh,
    }];
    let layers = vec![Layer::Dense(DenseLayer {
        w: vec![vec![0.1, 0.2, 0.3]],
        b: vec![0.4],
        activation: Activation::Tanh,
    })];
    let original = NeuralNetModel {
        architecture: arch,
        layer_sizes: vec![3, 1],
        layers,
        input_mask: None,
        ablated_input: None,
        output_param: OutputParam::AcosTanh,
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.json");
    original.save_json(path.to_str().unwrap()).unwrap();
    let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();

    assert_eq!(loaded.output_param, OutputParam::AcosTanh);
}

#[test]
fn output_param_absent_in_json_loads_as_atan2_signed() {
    let json = r#"{
        "format_version": 2,
        "architecture": [{"type": "dense", "input_size": 2, "output_size": 2, "activation": "linear"}],
        "weights": [{"w": [[0.1, 0.2], [0.3, 0.4]], "b": [0.0, 0.0]}]
    }"#;
    let m = NeuralNetModel::from_json_str(json, "<test>").unwrap();
    assert_eq!(m.output_param, OutputParam::Atan2Signed);
}
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd src/rust && cargo test --lib output_param_persists output_param_absent 2>&1 | tail -20
```

- [ ] **Step 3: Update `save_json`**

Locate `save_json` in `src/rust/src/data/neural.rs`. Find the JSON-builder block that writes `format_version`, `architecture`, `weights`, `input_mask`, `ablated_input`. Add the `output_param` key emission alongside `input_mask`:

```rust
    obj.insert("output_param".to_string(), serde_json::to_value(self.output_param).unwrap());
```

(If `save_json` uses `#[derive(Serialize)]` on a struct rather than manual JSON building, add `output_param: OutputParam` to the serialized struct with `#[serde(default)]` for back-compat reads.)

- [ ] **Step 4: Update `from_v2_json`**

After the existing extraction of `input_mask` / `ablated_input`, add:

```rust
    let output_param: OutputParam = json
        .get("output_param")
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|e| Error(format!("invalid output_param in {context}: {e}")))?
        .unwrap_or_default();
```

Then include `output_param` in the `Ok(NeuralNetModel { ... })` struct literal.

- [ ] **Step 5: Update other constructors**

```bash
cd src/rust && grep -n "input_mask:" src/rust/src/data/neural.rs | grep -v "//" | head
```

Each non-test constructor must populate `output_param`. For `from_flat_weights_v2`, accept it as an additional parameter (default `OutputParam::default()`).

- [ ] **Step 6: Re-run tests**

```bash
cd src/rust && cargo test --lib output_param 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): persist output_param in v2 JSON save/load (default Atan2Signed)"
```

---

## Task B3: Dispatch on `output_param` in `nn_bank_angle`

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs` (`nn_bank_angle` around line 142)

- [ ] **Step 1: Write the failing test**

Append to `src/rust/src/gnc/guidance/neural.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn acos_tanh_parameterization_emits_acos_of_output() {
    use crate::data::neural::{
        Activation, DenseLayer, Layer, LayerSpec, NeuralNetModel, OutputParam,
    };

    let nn = NeuralNetModel {
        architecture: vec![LayerSpec::Dense {
            input_size: 16,
            output_size: 1,
            activation: Activation::Tanh,
        }],
        layer_sizes: vec![16, 1],
        layers: vec![Layer::Dense(DenseLayer {
            w: vec![vec![0.0; 16]],
            b: vec![0.5],
            activation: Activation::Tanh,
        })],
        input_mask: None,
        ablated_input: None,
        output_param: OutputParam::AcosTanh,
    };

    let nav = test_nav();
    let data = test_sim_data();
    let planet = PlanetConfig::mars();
    let mut state = NnState::for_model(&nn);
    let bank = nn_bank_angle(&nav, &nn, &mut state, &data, &planet, 50.0_f64.to_radians(), 0.0);

    let expected = (0.5_f64).tanh().acos();
    assert!((bank - expected).abs() < 1e-12, "bank={bank} expected={expected}");
}
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd src/rust && cargo test --lib acos_tanh_parameterization 2>&1 | tail -10
```

- [ ] **Step 3: Update `nn_bank_angle`**

Find:

```rust
    let output = nn.forward(nn_state, &masked);
    output[0].atan2(output[1])
```

Replace with:

```rust
    use crate::data::neural::OutputParam;
    let output = nn.forward(nn_state, &masked);
    match nn.output_param {
        OutputParam::Atan2Signed => output[0].atan2(output[1]),
        OutputParam::AcosTanh => output[0].acos(),
    }
```

- [ ] **Step 4: Re-run the test**

```bash
cd src/rust && cargo test --lib acos_tanh_parameterization 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Run the guidance test suite**

```bash
cd src/rust && cargo test --lib gnc::guidance 2>&1 | tail -10
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs
git commit -m "feat(nn): dispatch nn_bank_angle on OutputParam (Atan2Signed | AcosTanh)"
```

---

## Task B4: Config validation for `acos_tanh`

**Files:**
- Modify: `src/rust/src/data/mod.rs` (after `neural_mode` parse around lines 423-438)
- Modify: `src/rust/src/config.rs` (add `output_parameterization` field to TOML struct backing `[guidance.neural_network]`)

- [ ] **Step 1: Add the field to the TOML struct**

```bash
grep -n "neural_network" src/rust/src/config.rs | head
```

Find the struct backing `[guidance.neural_network]` (e.g. `TomlGuidanceNeuralNetwork`). Add:

```rust
    pub output_parameterization: Option<String>,
```

- [ ] **Step 2: Add the validation in `data/mod.rs`**

After the existing `neural_mode` parsing block (around line 438):

```rust
    let output_param_toml = toml
        .guidance
        .neural_network
        .as_ref()
        .and_then(|nn| nn.output_parameterization.as_deref());
    if matches!(output_param_toml, Some("acos_tanh")) {
        if neural_mode != guidance_params::NeuralNetMode::MagnitudeOnly {
            return Err(DataError(
                "output_parameterization='acos_tanh' is only legal with mode='magnitude_only' \
                 (it cannot emit signed bank); use 'atan2_signed' for full_neural mode"
                    .to_string(),
            ));
        }
        if let Some(net) = toml.network.as_ref() {
            if let Some(arch) = net.architecture.as_ref() {
                let last = arch.last().ok_or_else(|| {
                    DataError("output_parameterization='acos_tanh' requires [[network.architecture]] entries".to_string())
                })?;
                match last {
                    TomlLayerSpec::Dense { output_size, activation, .. } => {
                        if *output_size != 1 {
                            return Err(DataError(format!(
                                "output_parameterization='acos_tanh' requires last layer output_size=1, got {output_size}"
                            )));
                        }
                        if activation.as_deref() != Some("tanh") {
                            return Err(DataError(format!(
                                "output_parameterization='acos_tanh' requires last-layer activation='tanh', got {:?}",
                                activation
                            )));
                        }
                    }
                    _ => {
                        return Err(DataError(
                            "output_parameterization='acos_tanh' requires last layer to be dense".to_string(),
                        ));
                    }
                }
            }
        }
    }
```

- [ ] **Step 3: Add tests**

Append to the `#[cfg(test)] mod tests` block (or wherever existing `mode`-parser tests live):

```rust
#[test]
fn acos_tanh_with_full_neural_mode_is_rejected() {
    let toml_str = r#"
[guidance]
type = "neural_network"

[guidance.neural_network]
mode = "full_neural"
output_parameterization = "acos_tanh"

[[network.architecture]]
type = "dense"
input_size = 21
output_size = 1
activation = "tanh"
    "#;
    let result = TomlConfig::from_str(toml_str).and_then(|t| SimData::build_from_toml(&t, /*config-stub*/));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("acos_tanh") && err.contains("magnitude_only"), "got: {err}");
}
```

(Adapt to whatever the existing test harness uses — `from_str`, `build_from_toml`, `parse_toml_for_test`, etc. Look for sibling tests of the `mode` parser to find the harness.)

Add the analogous tests for the `output_size != 1` and `activation != tanh` cases.

- [ ] **Step 4: Re-run**

```bash
cd src/rust && cargo test --lib acos_tanh_with 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/mod.rs src/rust/src/config.rs
git commit -m "feat(nn): validate acos_tanh requires magnitude_only + last-layer (1, tanh)"
```

---

## Task B5: Plumb `output_param` through `flat_weights_to_json`

**Files:**
- Modify: `src/rust/aerocapture-py/src/lib.rs` (`flat_weights_to_json`)
- Modify: `src/python/aerocapture/training/evaluate.py` (`write_nn_json`)
- Modify: `src/python/aerocapture/training/config.py` (add `output_parameterization` to `NetworkConfig`)
- Modify: `src/python/aerocapture/training/train.py` (TOML wiring + callsite)

- [ ] **Step 1: Extend the PyO3 signature**

In `src/rust/aerocapture-py/src/lib.rs`, find `flat_weights_to_json`. Add an optional `output_param: Option<String>` keyword argument (signature attribute `output_param=None`). After parsing existing inputs:

```rust
    let output_param: aerocapture::data::neural::OutputParam = match output_param.as_deref() {
        None | Some("atan2_signed") => aerocapture::data::neural::OutputParam::Atan2Signed,
        Some("acos_tanh") => aerocapture::data::neural::OutputParam::AcosTanh,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "output_param must be 'atan2_signed' or 'acos_tanh' (got {other:?})"
            )));
        }
    };
```

Pass `output_param` into `NeuralNetModel::from_flat_weights_v2(...)` (extending its signature) so it lands in the saved model.

- [ ] **Step 2: Extend `write_nn_json`**

```python
def write_nn_json(
    weights: npt.NDArray[np.float64],
    network: NetworkConfig,
    filepath: str | Path,
    input_mask: list[int] | None = None,
    output_param: str | None = None,
) -> None:
    ...
    _aero_rs.flat_weights_to_json(
        flat=weights.astype(np.float64).tolist(),
        architecture_json=json.dumps(arch),
        path=str(filepath),
        input_mask=input_mask,
        output_param=output_param,
    )
```

- [ ] **Step 3: Add `output_parameterization` to `NetworkConfig`**

In `src/python/aerocapture/training/config.py`:

```python
output_parameterization: str | None = None
```

- [ ] **Step 4: Wire TOML key in `train.py`**

Next to `optimize_scaffolding` in the `_gnn` block:

```python
    if "output_parameterization" in _gnn:
        cfg.network.output_parameterization = str(_gnn["output_parameterization"])
```

- [ ] **Step 5: Pass `output_param` at every `write_nn_json` callsite**

```bash
grep -n "write_nn_json" src/python/aerocapture/training/train.py
```

For every call, add `output_param=config.network.output_parameterization` (or `cfg.network.output_parameterization` in the main block).

- [ ] **Step 6: Rebuild PyO3**

```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
```

Expected: build succeeds.

- [ ] **Step 7: Quick smoke check**

```bash
uv run python -c "
import aerocapture_rs, json, tempfile
from pathlib import Path
arch = json.dumps([{'type': 'dense', 'input_size': 2, 'output_size': 1, 'activation': 'tanh'}])
flat = [0.1, 0.2, 0.3]
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    p = f.name
aerocapture_rs.flat_weights_to_json(flat=flat, architecture_json=arch, path=p, output_param='acos_tanh')
data = json.loads(Path(p).read_text())
assert data['output_param'] == 'acos_tanh', data
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 8: Commit**

```bash
git add src/rust/aerocapture-py/src/lib.rs src/python/aerocapture/training/evaluate.py src/python/aerocapture/training/config.py src/python/aerocapture/training/train.py
git commit -m "feat(nn): plumb output_param through flat_weights_to_json + write_nn_json"
```

---

## Task B6: Cross-language equivalence test for `acos_tanh`

**Files:**
- Modify: `tests/test_v2_rust_python_equivalence.py`

- [ ] **Step 1: Add the test**

Append:

```python
@pytest.mark.slow
def test_acos_tanh_rust_python_equivalence(tmp_path):
    """V2Policy with last-layer (1, tanh) + acos_tanh runtime → Rust nn_forward bit-equivalent."""
    import aerocapture_rs
    import numpy as np
    import torch
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.export import export_v2_policy_to_json

    arch = [
        {"type": "dense", "input_size": 8, "output_size": 16, "activation": "swish"},
        {"type": "dense", "input_size": 16, "output_size": 1, "activation": "tanh"},
    ]
    torch.manual_seed(0)
    policy = V2Policy(arch).double()
    json_path = tmp_path / "model.json"
    export_v2_policy_to_json(policy, str(json_path), output_param="acos_tanh")

    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((100, 8)).tolist()

    rust_outputs = []
    for x in inputs:
        out = aerocapture_rs.nn_forward(model_path=str(json_path), input_vec=x)
        rust_outputs.append(out)
    rust_outputs = np.asarray(rust_outputs)

    py_outputs = []
    for x in inputs:
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
            y = policy(t).cpu().numpy()
        py_outputs.append(y[0])
    py_outputs = np.asarray(py_outputs)

    diff = np.max(np.abs(rust_outputs - py_outputs))
    assert diff < 1e-10, f"max abs diff {diff} exceeds tolerance"
```

(`export_v2_policy_to_json` must accept `output_param`. If it doesn't yet, extend its signature alongside the V2Policy export path so the JSON file embeds the parameterization. The trainer's `write_nn_json` already does the analogous plumbing in B5 — replicate the same approach here.)

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_v2_rust_python_equivalence.py::test_acos_tanh_rust_python_equivalence -v -m slow
```

Expected: PASS with diff < 1e-10.

- [ ] **Step 3: Commit**

```bash
git add tests/test_v2_rust_python_equivalence.py src/python/aerocapture/training/rl/export.py
git commit -m "test(nn): cross-language equivalence for acos_tanh parameterization"
```

---

# Fix C — FTC behavioural-cloning warm-start

## Task C1: Reserved seed offset constant

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py` (constants block lines 27-30)

- [ ] **Step 1: Add the constant**

Replace the existing offsets block:

```python
# Reserved seed offsets — guarantees training, validation, final eval, RL
# training, and supervised warm-start collection never share the same RNG stream.
VALIDATION_SEED_OFFSET = 1_000_000
FINAL_EVAL_SEED_OFFSET = 2_000_000
RL_TRAINING_SEED_OFFSET = 3_000_000
WARM_START_SEED_OFFSET = 4_000_000
```

(`RL_TRAINING_SEED_OFFSET` may already exist elsewhere — search via `grep -rn "RL_TRAINING_SEED_OFFSET" src/python` and unify.)

- [ ] **Step 2: Smoke check**

```bash
uv run python -c "from aerocapture.training.evaluate import WARM_START_SEED_OFFSET; assert WARM_START_SEED_OFFSET == 4_000_000; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py
git commit -m "feat(nn): WARM_START_SEED_OFFSET = 4M (disjoint from val/final-eval/RL pools)"
```

---

## Task C2: Rust `supervised_trace` plumbing

**Files:**
- Modify: `src/rust/src/lib.rs` (`RunOutput` struct)
- Modify: `src/rust/src/simulation/runner.rs` (config + drain)
- Modify: `src/rust/src/simulation/tick.rs` (per-tick capture)
- Modify: `src/rust/src/gnc/guidance/dispatch.rs` (surface pre-lateral magnitude)

- [ ] **Step 1: Surface the pre-lateral magnitude in `GuidanceOutput`**

In `src/rust/src/gnc/guidance/dispatch.rs`, add to `GuidanceOutput` (around line 86):

```rust
    pub pre_lateral_magnitude: f64, // bank magnitude after thermal limiter, before lateral sign
```

In `guidance_step`, populate it just before lateral guidance assigns the sign (around line 246):

```rust
    out.pre_lateral_magnitude = bank_angle_longitudinal;
```

- [ ] **Step 2: Extend `RunOutput`**

In `src/rust/src/lib.rs`, add to `RunOutput`:

```rust
    /// When the runner was invoked with `collect_supervised = true`, holds
    /// per-tick (nn_input_21, bank_magnitude_post_thermal) pairs.
    pub supervised_trace: Vec<(Vec<f64>, f64)>,
```

Update any explicit constructors / `Default` impl to initialise as `Vec::new()`.

- [ ] **Step 3: Add `collect_supervised` flag to runner config**

Find the runner-config struct in `src/rust/src/simulation/runner.rs` (carries `reference_trajectory`). Add:

```rust
    pub collect_supervised: bool,
```

Default to `false` everywhere it's currently constructed.

- [ ] **Step 4: Refactor `build_nn_input` to accept mask + ablation directly**

In `src/rust/src/gnc/guidance/neural.rs`, change `build_nn_input`'s signature to take `input_mask: Option<&[usize]>` and `ablated_input: Option<usize>` instead of `&NeuralNetModel`. Update its existing caller to pass `nn.input_mask.as_deref()` and `nn.ablated_input`. (This refactor is small and single-file but enables tasks where there is no `NeuralNetModel` available — which is the supervised-trace case.)

- [ ] **Step 5: Capture per-tick in `tick.rs`**

In `src/rust/src/simulation/tick.rs`, after the `guidance_out` is computed and before `bank_angle_commanded` is consumed, add:

```rust
    if config.collect_supervised {
        // Full 21-element vector, no mask, no ablation — supervised target consumes raw vector.
        let nn_input = crate::gnc::guidance::neural::build_nn_input(
            &nav_out,
            None,                              // input_mask
            None,                              // ablated_input
            data,
            planet,
            data.target_orbit.inclination,
            state.guidance_state.reference_velocity,
        );
        state.run_state.supervised_trace.push((nn_input, guidance_out.pre_lateral_magnitude));
    }
```

(Add `supervised_trace: Vec<(Vec<f64>, f64)>` to `RunState` struct. Initialise empty in `RunState::new()`.)

- [ ] **Step 6: Drain into `RunOutput`**

In `src/rust/src/simulation/runner.rs`, after the main loop completes, if `config.collect_supervised`:

```rust
    output.supervised_trace = std::mem::take(&mut state.run_state.supervised_trace);
```

- [ ] **Step 7: Run the full Rust suite**

```bash
cd src/rust && cargo test --lib 2>&1 | tail -5
```

Expected: all PASS (no regression — `collect_supervised = false` path is identical to today).

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/lib.rs src/rust/src/simulation/runner.rs src/rust/src/simulation/tick.rs src/rust/src/gnc/guidance/dispatch.rs src/rust/src/gnc/guidance/neural.rs
git commit -m "feat(nn): supervised_trace plumbing in RunOutput (per-tick (nn_input, |bank|))"
```

---

## Task C3: PyO3 `collect_supervised` wrapper

**Files:**
- Modify: `src/rust/aerocapture-py/src/lib.rs`
- Test: `tests/test_collect_supervised.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_collect_supervised.py`:

```python
"""Smoke test for the new collect_supervised PyO3 helper."""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_collect_supervised_returns_finite_traces():
    import aerocapture_rs
    import numpy as np

    X, y = aerocapture_rs.collect_supervised(
        toml_path="configs/training/msr_aller_ftc_train.toml",
        seeds=[42],
        scheme="ftc",
    )
    X = np.asarray(X)
    y = np.asarray(y)
    assert X.ndim == 2 and X.shape[1] == 21, X.shape
    assert y.ndim == 1 and y.shape[0] == X.shape[0], (X.shape, y.shape)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert (y >= 0.0).all() and (y <= np.pi + 1e-9).all()
```

- [ ] **Step 2: Run to confirm fail**

```bash
uv run pytest tests/test_collect_supervised.py -v -m slow
```

Expected: FAIL — `aerocapture_rs.collect_supervised` does not exist.

- [ ] **Step 3: Implement the wrapper**

In `src/rust/aerocapture-py/src/lib.rs`:

```rust
#[pyfunction]
#[pyo3(signature = (toml_path, seeds, overrides=None, scheme="ftc".to_string(), sim_timeout_secs=None))]
fn collect_supervised(
    py: Python<'_>,
    toml_path: String,
    seeds: Vec<u64>,
    overrides: Option<&PyDict>,
    scheme: String,
    sim_timeout_secs: Option<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    use aerocapture::config::GuidanceType;

    let scheme = match scheme.as_str() {
        "ftc" => GuidanceType::Ftc,
        "equilibrium_glide" => GuidanceType::EquilibriumGlide,
        "energy_controller" => GuidanceType::EnergyController,
        "pred_guid" => GuidanceType::PredGuid,
        "fnpag" => GuidanceType::Fnpag,
        "piecewise_constant" => GuidanceType::PiecewiseConstant,
        other => {
            return Err(PyValueError::new_err(format!(
                "scheme must be a non-NN unsigned-magnitude scheme; got '{other}'"
            )));
        }
    };

    let mut all_x_rows: Vec<Vec<f64>> = Vec::new();
    let mut all_y: Vec<f64> = Vec::new();

    py.detach(|| {
        for seed in seeds {
            let mut cfg = build_config_from_toml(&toml_path, overrides, scheme, seed)?;
            cfg.collect_supervised = true;
            let out = aerocapture::simulation::runner::run_for_api(&cfg, sim_timeout_secs)?;
            for (x, y_val) in out.supervised_trace {
                all_x_rows.push(x);
                all_y.push(y_val);
            }
        }
        Ok::<_, PyErr>(())
    })?;

    let x_array = PyArray2::from_vec2(py, &all_x_rows)?;
    let y_array = PyArray1::from_vec(py, all_y);
    Ok((x_array.unbind(), y_array.unbind()))
}
```

(`build_config_from_toml` mirrors the existing `run_batch` config helper — match its name and signature exactly.)

- [ ] **Step 4: Register the function**

```rust
m.add_function(wrap_pyfunction!(collect_supervised, m)?)?;
```

- [ ] **Step 5: Rebuild**

```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
```

- [ ] **Step 6: Re-run the test**

```bash
uv run pytest tests/test_collect_supervised.py -v -m slow
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rust/aerocapture-py/src/lib.rs tests/test_collect_supervised.py
git commit -m "feat(nn): aerocapture_rs.collect_supervised PyO3 helper for FTC supervised data"
```

---

## Task C4: `warm_start.py` — supervised pre-train + chromosome assembly

**Files:**
- Create: `src/python/aerocapture/training/warm_start.py`
- Test: `tests/test_warm_start_pipeline.py` (new)

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_warm_start_pipeline.py`:

```python
"""End-to-end smoke for warm_start.build_warm_start_chromosome."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_build_warm_start_chromosome_returns_correctly_shaped_normalized_vector(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    ftc_params = repo_root / "training_output" / "ftc" / "best_params.json"
    if not ftc_params.exists():
        pytest.skip("FTC training output absent")

    from aerocapture.training.config import NetworkConfig, TrainingConfig
    from aerocapture.training.warm_start import build_warm_start_chromosome

    cfg = TrainingConfig()
    cfg.guidance_type = "neural_network"
    cfg.network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 21, "output_size": 8, "activation": "swish"},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(21)),
        output_parameterization="acos_tanh",
        optimize_scaffolding=False,
    )
    cfg.sim.toml_config = "configs/training/msr_aller_ftc_train.toml"
    cfg.sim.exec_dir = str(repo_root)
    cfg.save_dir = str(tmp_path / "warm")
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    chromo = build_warm_start_chromosome(
        cfg=cfg,
        n_warm_seeds=4,
        n_epochs=2,
        rng=rng,
    )
    # 21*8 + 8 + 8*1 + 1 = 185
    assert chromo.shape == (185,), chromo.shape
    assert (chromo >= 0.0).all() and (chromo <= 1.0).all()
    assert (Path(cfg.save_dir) / "warm_start_chromosome.npy").exists()
    assert (Path(cfg.save_dir) / "warm_start_cache_key.json").exists()
```

- [ ] **Step 2: Run to confirm fail**

```bash
uv run pytest tests/test_warm_start_pipeline.py -v -m slow
```

Expected: FAIL — `aerocapture.training.warm_start` does not exist.

- [ ] **Step 3: Implement the module**

Create `src/python/aerocapture/training/warm_start.py`:

```python
"""Behavioural-cloning warm-start for NN guidance training.

Runs a non-NN scheme (default FTC) over a reserved seed pool, collects
(state, |bank|) pairs via aerocapture_rs.collect_supervised, supervised
pre-trains a V2Policy mirror to mimic the cloned scheme's bank magnitude,
encodes the trained weights to a normalized [0, 1] PSO chromosome.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import NetworkConfig, TrainingConfig
from aerocapture.training.encoding import encode_to_normalized, nn_param_specs_from_v2
from aerocapture.training.evaluate import WARM_START_SEED_OFFSET, make_reserved_seeds
from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

try:
    import aerocapture_rs as _aero_rs
except ImportError as e:
    raise ImportError("warm_start requires aerocapture_rs PyO3 module") from e


def _cache_key(cfg: TrainingConfig, source_path: Path, n_warm_seeds: int, n_epochs: int) -> dict:
    return {
        "architecture": cfg.network.architecture,
        "input_mask": cfg.network.input_mask,
        "output_parameterization": cfg.network.output_parameterization or "atan2_signed",
        "source_path": str(source_path),
        "source_mtime": source_path.stat().st_mtime,
        "n_warm_seeds": n_warm_seeds,
        "n_epochs": n_epochs,
    }


def _cache_hit(save_dir: Path, expected_key: dict) -> npt.NDArray[np.float64] | None:
    chromo_path = save_dir / "warm_start_chromosome.npy"
    key_path = save_dir / "warm_start_cache_key.json"
    if not (chromo_path.exists() and key_path.exists()):
        return None
    saved_key = json.loads(key_path.read_text())
    if saved_key != expected_key:
        return None
    return np.load(chromo_path)


def _build_overrides_for_source(source_params: dict[str, float]) -> dict[str, object]:
    """Mirror problem.py::_build_overrides routing for the supervised data source."""
    overrides: dict[str, object] = {}
    for key, value in source_params.items():
        if key.startswith("lateral."):
            overrides[f"guidance.lateral.{key.removeprefix('lateral.')}"] = value
        elif key.startswith("exit."):
            overrides[f"guidance.ftc.{key.removeprefix('exit.')}"] = value
        elif key.startswith("nav."):
            overrides[f"navigation.{key.removeprefix('nav.')}"] = value
        elif key.startswith("thermal."):
            overrides[f"guidance.thermal_limiter.{key.removeprefix('thermal.')}"] = value
        elif key.startswith("shaping."):
            overrides[f"guidance.command_shaping.{key.removeprefix('shaping.')}"] = value
            overrides["guidance.command_shaping.enabled"] = True
        else:
            overrides[f"guidance.ftc.{key}"] = value
    return overrides


def _supervised_pretrain(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    network: NetworkConfig,
    n_epochs: int,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    import torch
    from torch import nn

    from aerocapture.training.rl.policy import V2Policy

    policy = V2Policy(network.architecture).double()

    output_param = network.output_parameterization or "atan2_signed"
    if output_param == "acos_tanh":
        target = np.cos(y).reshape(-1, 1)
    elif output_param == "atan2_signed":
        target = np.stack([np.sin(y), np.cos(y)], axis=1)
    else:
        raise ValueError(f"unknown output_parameterization {output_param!r}")

    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(target, dtype=torch.float64)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = X_t.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = policy(X_t[idx])
            loss = nn.functional.mse_loss(pred, y_t[idx])
            loss.backward()
            optimizer.step()
    return policy


def _policy_to_flat_weights(policy, architecture: list[dict]) -> npt.NDArray[np.float64]:
    """Extract physical weights in canonical flat order (row-major W then b per dense layer)."""
    import torch.nn as nn

    flat: list[float] = []
    layer_modules = [m for m in policy.modules() if isinstance(m, nn.Linear)]
    if len(layer_modules) != len(architecture):
        raise RuntimeError(
            f"expected {len(architecture)} Linear layers; policy has {len(layer_modules)}"
        )
    for entry, m in zip(architecture, layer_modules, strict=True):
        if entry["type"] != "dense":
            raise NotImplementedError(
                f"_policy_to_flat_weights supports dense only; got {entry['type']!r}"
            )
        w = m.weight.detach().cpu().numpy().astype(np.float64)
        b = m.bias.detach().cpu().numpy().astype(np.float64)
        flat.extend(w.ravel().tolist())
        flat.extend(b.tolist())
    return np.asarray(flat, dtype=np.float64)


def build_warm_start_chromosome(
    cfg: TrainingConfig,
    n_warm_seeds: int = 200,
    n_epochs: int = 10,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Run cfg's source scheme on n_warm_seeds, supervised-pretrain V2Policy, return chromosome."""
    if rng is None:
        rng = np.random.default_rng(0)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(
        getattr(cfg.network, "warm_start_from", None) or "training_output/ftc/best_params.json"
    )
    if not source_path.exists():
        raise FileNotFoundError(
            f"warm-start source params not found at '{source_path}'. "
            f"Run FTC training first or set warm_start_from."
        )

    cache_key = _cache_key(cfg, source_path, n_warm_seeds, n_epochs)
    cached = _cache_hit(save_dir, cache_key)
    if cached is not None:
        return cached

    with open(source_path) as f:
        source_params = json.load(f)
    overrides = _build_overrides_for_source(source_params)

    base_mc_seed = 42
    seeds = make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, n_warm_seeds)

    X_full, y = _aero_rs.collect_supervised(
        toml_path=cfg.sim.toml_config,
        seeds=seeds,
        overrides=overrides,
        scheme="ftc",
    )
    X_full = np.asarray(X_full)
    y = np.asarray(y)
    finite_mask = np.isfinite(X_full).all(axis=1) & np.isfinite(y)
    X_full = X_full[finite_mask]
    y = y[finite_mask]

    mask = cfg.network.input_mask if cfg.network.input_mask is not None else list(range(16))
    X = X_full[:, mask]

    policy = _supervised_pretrain(X, y, cfg.network, n_epochs)
    flat_weights = _policy_to_flat_weights(policy, cfg.network.architecture)

    from pydantic import TypeAdapter

    from aerocapture.training.rl.schemas import LayerSpec

    validated = TypeAdapter(list[LayerSpec]).validate_python(cfg.network.architecture)
    weight_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)
    weight_chromo = np.empty(len(weight_specs), dtype=np.float64)
    for i, s in enumerate(weight_specs):
        v = float(flat_weights[i])
        weight_chromo[i] = np.clip((v - s.p_min) / (s.p_max - s.p_min), 0.0, 1.0)

    chromo = weight_chromo
    if cfg.network.optimize_scaffolding:
        scaff_chromo = encode_to_normalized(source_params, list(_NN_SCAFFOLDING_PARAMS))
        chromo = np.concatenate([weight_chromo, scaff_chromo])

    np.save(save_dir / "warm_start_chromosome.npy", chromo)
    (save_dir / "warm_start_cache_key.json").write_text(json.dumps(cache_key, indent=2))

    return chromo
```

- [ ] **Step 4: Re-run the test**

```bash
uv run pytest tests/test_warm_start_pipeline.py -v -m slow
```

Expected: PASS (or SKIP if FTC output absent).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/warm_start.py tests/test_warm_start_pipeline.py
git commit -m "feat(nn): warm_start.py supervised pre-train + chromosome cache"
```

---

## Task C5: `warm_start_from` knob + initial-population wiring

**Files:**
- Modify: `src/python/aerocapture/training/config.py` (`NetworkConfig`)
- Modify: `src/python/aerocapture/training/train.py` (TOML wiring + initial-population branch + validation)

- [ ] **Step 1: Add the field**

In `NetworkConfig`:

```python
warm_start_from: str | None = None
```

- [ ] **Step 2: Wire the TOML key in `train.py`**

In the `_gnn` block (added in Task A2):

```python
    if "warm_start_from" in _gnn:
        cfg.network.warm_start_from = str(_gnn["warm_start_from"])
```

- [ ] **Step 3: Add validation right after the `_gnn` parse**

```python
    if cfg.network.warm_start_from is not None:
        nn_mode = _gnn.get("mode", "full_neural")
        if nn_mode != "magnitude_only":
            print(
                f"ERROR: warm_start_from is set but [guidance.neural_network] mode={nn_mode!r}. "
                f"Behavioural-cloning targets unsigned bank magnitude; only magnitude_only mode "
                f"can consume the cloned NN's output."
            )
            raise SystemExit(1)
        warm_path = Path(cfg.network.warm_start_from)
        if not warm_path.exists():
            print(f"ERROR: warm_start_from='{warm_path}' does not exist")
            raise SystemExit(1)
```

- [ ] **Step 4: Use `build_warm_start_chromosome` in initial-population branch**

In `train.py`, replace the `elif config.guidance_type == "neural_network" and config.network.architecture is not None:` block (modified in Task A5) with:

```python
        elif config.guidance_type == "neural_network" and config.network.architecture is not None:
            scaffolding_slab = None
            if config.network.optimize_scaffolding:
                from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

                ftc_params_path = (
                    config.network.warm_start_from
                    or "training_output/ftc/best_params.json"
                )
                scaffolding_slab = build_scaffolding_initial_slab(
                    ftc_params_path,
                    list(_NN_SCAFFOLDING_PARAMS),
                    config.optimizer.n_pop,
                    rng,
                    jitter=0.02,
                )

            if config.network.warm_start_from:
                from aerocapture.training.warm_start import build_warm_start_chromosome

                warm_chromo = build_warm_start_chromosome(
                    cfg=config,
                    n_warm_seeds=200,
                    n_epochs=10,
                    rng=rng,
                )
                n_pop = config.optimizer.n_pop
                pop_array = np.tile(warm_chromo, (n_pop, 1))
                n_scaff = 17 if config.network.optimize_scaffolding else 0
                n_weights = len(warm_chromo) - n_scaff
                pop_array[:, :n_weights] += rng.normal(0.0, 0.02, size=(n_pop, n_weights))
                pop_array[:, :n_weights] = np.clip(pop_array[:, :n_weights], 0.0, 1.0)
                if scaffolding_slab is not None:
                    pop_array[:, n_weights:] = scaffolding_slab
            else:
                pop_array = build_initial_population_for_v2(
                    config.network.architecture,
                    config.optimizer.n_pop,
                    bound_multiplier=2.0,
                    rng=rng,
                    param_specs=param_specs,
                    scaffolding_slab=scaffolding_slab,
                )
```

- [ ] **Step 5: Smoke check**

```bash
uv run python -c "from aerocapture.training.config import NetworkConfig; n = NetworkConfig(); assert n.warm_start_from is None; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/config.py src/python/aerocapture/training/train.py
git commit -m "feat(nn): warm_start_from knob seeds initial population from cloned chromosome"
```

---

# Final assembly

## Task F1: New training config flipping all three knobs

**Files:**
- Create: `configs/training/msr_aller_nn_joint_train.toml`

- [ ] **Step 1: Create the config**

```toml
# MSR aller -- NN training with all three parity-bundle fixes ON.
# Inherits FTC scaffolding values from training_output/ftc/best_params.json
# at training start. PSO/GA jointly tunes scaffolding alongside the network weights.
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "neural_network"

[guidance.neural_network]
mode = "magnitude_only"
optimize_scaffolding = true
output_parameterization = "acos_tanh"
warm_start_from = "training_output/ftc/best_params.json"

[network]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

[[network.architecture]]
type = "dense"
input_size = 21
output_size = 32
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 16
activation = "swish"

# Last layer: single output through tanh, fed into acos at runtime to emit [0, π] magnitude.
[[network.architecture]]
type = "dense"
input_size = 16
output_size = 1
activation = "tanh"

[data]
neural_network = "training_output/neural_network_joint/best_model.json"
results_suffix = ".train_nn_joint"
```

- [ ] **Step 2: Commit**

```bash
git add configs/training/msr_aller_nn_joint_train.toml
git commit -m "config(nn): msr_aller_nn_joint_train.toml flipping all three parity-bundle knobs"
```

---

## Task F2: `train_all.sh` alias

**Files:**
- Modify: `train_all.sh`

- [ ] **Step 1: Add the alias case**

```bash
grep -n "nn\b\|gru_pso\b" train_all.sh | head
```

Find the case statement that dispatches scheme aliases. Add:

```bash
    nn_joint|joint)
        run_one neural_network_joint configs/training/msr_aller_nn_joint_train.toml
        ;;
```

(Match the exact pattern of existing aliases — `run_one` may differ.)

- [ ] **Step 2: Smoke check**

```bash
bash train_all.sh --help 2>&1 | head -20
```

Expected: shows the new alias in usage.

- [ ] **Step 3: Commit**

```bash
git add train_all.sh
git commit -m "scripts(train_all): add nn_joint alias for msr_aller_nn_joint_train.toml"
```

---

## Task F3: Final smart-commit

- [ ] **Step 1: Invoke the smart-commit skill**

Use the `smart-commit` skill, telling it: "take the whole feature/magnitude_only branch into account; the parity-bundle landed three TOML-opt-in fixes (joint scaffolding, acos_tanh output, FTC warm-start) plus a new training config and train_all.sh alias; sync CLAUDE.md and README.md with the new knobs."

The skill handles CLAUDE.md and README.md updates and commits everything.

---

## Self-review notes

- All spec sections covered: scaffolding constant (A1), TOML knob (A2), spec extension (A3), seed helper (A4), init-pop wiring (A5), best_params writeback (A6), shape-mismatch (A7), compare/report (A8), OutputParam enum (B1), JSON serde (B2), dispatch (B3), validation (B4), PyO3 plumbing (B5), cross-language gate (B6), seed offset (C1), Rust trace (C2), PyO3 wrapper (C3), Python pipeline (C4), TOML+init wiring (C5), config (F1), alias (F2), smart-commit (F3).
- No "TBD" / "implement later" placeholders.
- Type names consistent: `OutputParam` (Rust enum), `output_param` (Rust field), `output_parameterization` (TOML key + Python field).
- File paths and commands exact.
