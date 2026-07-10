# Quantization Study Appendix (Mamba-962) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the paper's named future-work gap (paper.typ:949) with an Appendix D quantization study of the deployed Mamba-962 guidance policy: PTQ sensitivity sweep (bits x granularity x tensor-policy + 4-bit leave-one-out), two QAT-4bit training arms (champion fine-tune + matched-budget from-scratch), and honest deployment-benefit numbers (analytic memory table + real int8/int4 Rust microbenchmark).

**Architecture:** Weight-only fake-quant (round to b-bit grid, store back as f64) so the validated Rust runtime is untouched and goldens stay bit-identical. The quantizer lives in Python (`quantize.py`, ported from the unmerged `feature/quantization` branch and extended from dense-only to dense+mamba). QAT reuses the branch's two hook points: population flat-weights rounded before every `run_grid` fitness eval (`problem._run_grid_records`, the single chokepoint for per-gen fitness, validation gate, and final selection) and at deploy (`evaluate.write_nn_json`). Compute benefits are measured by a standalone criterion bench (dev-dependency only, never wired into the sim).

**Tech Stack:** Python 3.14 (numpy, pytest), PyO3 `aerocapture_rs` (soft-import pattern), Rust 2024 + criterion, TOML configs with base inheritance, Typst paper.

## Global Constraints

- Python tooling via `uv` (`uv run pytest ...`, `uv run python -m ...`). Ruff line-length 160, target py314; mypy strict — new/modified functions get full type annotations.
- `aerocapture_rs` is soft-imported (CI's pure-Python job has no PyO3 module): import it inside functions or guard with `pytest.importorskip`, never at module top-level unguarded.
- NO changes under `src/rust/src/` — the 6 guidance goldens must stay bit-identical. The bench is `src/rust/benches/` + `Cargo.toml` dev-deps only.
- ASCII output only (no em dashes, no smart quotes) in all code, configs, docs.
- Never push. Never commit to main — all work on branch `feature/quantization-mamba962` (already created, carries the spec commit). End commit messages with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Champion artifacts (read-only inputs, NEVER written to): `training_output/mamba_p962_long/{best_model.json,best_params.json,checkpoint_g20000.json,checkpoint_g20000.npz}`. Arch: Dense(17->16, swish) -> Mamba(16, d_state=12, dt_rank=1) -> Dense(16->2, asinh); 962 NN params (+3 live-scaffolding genes in the chromosome, which are NOT part of the NN weights array).
- Eval pool: `make_reserved_seeds(base_mc_seed=42, HEADLINE_REQUOTE_SEED_OFFSET=8_000_000, n)` (seed 42 = `monte_carlo.seed` from `configs/training/common.toml`). Baseline sanity anchor: champion fresh-pool quote is capture 100%, DV p50 109.69 / p95 113.81 / p99 116.01 / CVaR95 115.23 m/s (`training_output/mamba_p962_long/fresh_pool_requote.json`).
- Tail-led reporting everywhere: CVaR95/p95 lead, p50 secondary, never mean-led.
- Spec: `docs/superpowers/specs/2026-07-10-quantization-study-appendix-design.md`.

## File Structure

```
src/python/aerocapture/training/quantize.py      # ported + rewritten: quant core (JSON + flat paths), memory_footprint, sweep runner, finalists, CLI
src/python/aerocapture/training/charts_quant.py  # ported + extended: sweep curve (capture+CVaR95), LOO bars, QAT convergence overlay
src/python/aerocapture/training/config.py        # NetworkConfig qat_bits/qat_granularity/qat_tensor_policy + validate_qat
src/python/aerocapture/training/problem.py       # QAT hook in _run_grid_records
src/python/aerocapture/training/evaluate.py      # QAT hook in write_nn_json
src/python/aerocapture/training/train.py         # [network] qat_* TOML plumbing in build_training_config_from_toml
tests/test_quantize.py                           # ported + extended (mamba, policy, LOO, flat/JSON agreement, memory)
tests/test_qat_training.py                       # ported + extended (mamba accept, policy validation, e2e smoke)
configs/training/quant/mamba962_qat4_finetune.toml
configs/training/quant/mamba962_qat4_scratch.toml
src/rust/Cargo.toml                              # + criterion dev-dep + [[bench]]
src/rust/benches/quant_forward.rs                # f64/f32/w8a8/w4a8 kernels + self-checks
experiments/paper/15_quantization.sh             # campaign runner (ptq | qat | bench | finalists | collect)
docs/paper/quantization_appendix.md              # working notes (mirrors architecture_probes_appendix.md)
articles/paper/paper.typ                         # Appendix D + updated future-work sentence at line 949
articles/paper/data/quant/                       # results JSONs bundle
```

---

### Task 1: Port the PTQ core from `feature/quantization`

The branch (unmerged, diverged 2026-06-05) has three files that are new relative to main and import-clean against current main. Port them verbatim, verify green, commit. Later tasks rewrite pieces of them; porting first preserves provenance in history.

**Files:**
- Create: `src/python/aerocapture/training/quantize.py` (from branch)
- Create: `src/python/aerocapture/training/charts_quant.py` (from branch)
- Create: `tests/test_quantize.py` (from branch)

**Interfaces:**
- Produces: `_quantize_matrix(w: NDArray, n_bits: int, granularity: str) -> NDArray` (symmetric fake-quant, per_channel = one scale per output row, per_tensor = one scale; qmax = 2^(b-1)-1; zero-amax rows pass through), `quantize_dense_weights(model_json, n_bits, granularity) -> dict`, `quantize_flat_weights_batch(weights, architecture, n_bits, granularity) -> NDArray`, `run_quant_sweep(...)`, `chart_quant_sweep(results, output_path)`. Tasks 2/3/7 rewrite `quantize_dense_weights`, `quantize_flat_weights_batch`, `run_quant_sweep` respectively.

- [ ] **Step 1: Check out the three files from the branch**

```bash
git checkout feature/quantization -- \
  src/python/aerocapture/training/quantize.py \
  src/python/aerocapture/training/charts_quant.py \
  tests/test_quantize.py
```

- [ ] **Step 2: Run the ported tests**

Run: `uv run pytest tests/test_quantize.py -v`
Expected: all PASS (the pure-quant tests need no PyO3; the `@pytest.mark.slow` sweep smoke — if present in the ported file — needs `aerocapture_rs` and the old dense model; if it references a dense model path that no longer resolves, mark it for deletion in Task 7 which replaces the sweep, and deselect it here with `-m "not slow"`).

- [ ] **Step 3: Lint**

Run: `./lint_code.sh`
Expected: clean (fix any ruff/mypy drift from a month of main churn — e.g. import ordering).

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/quantize.py src/python/aerocapture/training/charts_quant.py tests/test_quantize.py
git commit -m "feat(quantize): port dense-only PTQ core from feature/quantization"
```

---

### Task 2: Generalize the JSON-path quantizer to Mamba (`quantize_model_weights`)

Replace `quantize_dense_weights` with `quantize_model_weights` supporting dense+mamba layers, a tensor policy, and a leave-one-out escape hatch. Tensor rules (from the spec): dense quantizes `w` only (bias fp); mamba quantizes `x_proj_w` + `dt_proj_w` always, `a_log` + `d_skip` only under policy `"all"`, `dt_proj_b` never (bias). 1-D tensors always get a single per-tensor scale.

**Files:**
- Modify: `src/python/aerocapture/training/quantize.py`
- Test: `tests/test_quantize.py`

**Interfaces:**
- Consumes: `_quantize_matrix`, `_layer_types` from Task 1.
- Produces:
  - `_TENSOR_POLICIES = ("all", "proj_only")`
  - `_quantizable_tensors(model_json: dict, tensor_policy: str) -> list[tuple[str, int, str, bool]]` — `(key, layer_idx, field, is_1d)` per quantizable tensor; keys are `"layer_{i}.w"` (dense) and `"layer_{i}.x_proj_w"` etc. (mamba). Raises ValueError on any layer type outside {dense, mamba}.
  - `quantize_model_weights(model_json: dict, n_bits: int, granularity: str, tensor_policy: str = "all", only_tensor: str | None = None) -> dict`
  - `_quantize_vector(v: NDArray, n_bits: int) -> NDArray`
  - `quantize_dense_weights` is DELETED (no external callers; branch never merged). `run_quant_sweep`'s internal call is retargeted to `quantize_model_weights` (full rewrite comes in Task 7).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_quantize.py`; also update the existing dense tests' imports from `quantize_dense_weights` to `quantize_model_weights` — the (model, bits, gran) call signature is unchanged — and update `test_non_dense_raises`'s match string to `"dense\\+mamba"`)

```python
def _mamba_model(rng: np.random.Generator) -> dict:
    """Dense(3->4) -> Mamba(4, d_state=2, dt_rank=1) -> Dense(4->2), random weights."""
    return {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
            {"type": "mamba", "input_size": 4, "d_state": 2, "dt_rank": 1},
            {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": rng.standard_normal((4, 3)).tolist(), "b": rng.standard_normal(4).tolist()},
            "layer_1": {
                "x_proj_w": rng.standard_normal((5, 4)).tolist(),  # (dt_rank + 2*d_state, input) = (5, 4)
                "dt_proj_w": rng.standard_normal((4, 1)).tolist(),
                "dt_proj_b": rng.standard_normal(4).tolist(),
                "a_log": rng.standard_normal((4, 2)).tolist(),
                "d_skip": rng.standard_normal(4).tolist(),
            },
            "layer_2": {"w": rng.standard_normal((2, 4)).tolist(), "b": rng.standard_normal(2).tolist()},
        },
    }


def _arrays_equal(a: object, b: object) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))


def test_mamba_all_policy_quantizes_projections_and_dynamics() -> None:
    from aerocapture.training.quantize import quantize_model_weights

    m = _mamba_model(np.random.default_rng(3))
    q = quantize_model_weights(m, 4, "per_channel", "all")
    l1, ql1 = m["weights"]["layer_1"], q["weights"]["layer_1"]
    for field in ("x_proj_w", "dt_proj_w", "a_log", "d_skip"):
        assert not _arrays_equal(l1[field], ql1[field]), f"{field} should be rounded at 4 bits"
    assert _arrays_equal(l1["dt_proj_b"], ql1["dt_proj_b"]), "dt_proj_b is a bias: never quantized"
    assert _arrays_equal(m["weights"]["layer_0"]["b"], q["weights"]["layer_0"]["b"])
    assert not _arrays_equal(m["weights"]["layer_0"]["w"], q["weights"]["layer_0"]["w"])


def test_mamba_proj_only_policy_keeps_dynamics_fp() -> None:
    from aerocapture.training.quantize import quantize_model_weights

    m = _mamba_model(np.random.default_rng(4))
    q = quantize_model_weights(m, 4, "per_channel", "proj_only")
    l1, ql1 = m["weights"]["layer_1"], q["weights"]["layer_1"]
    assert _arrays_equal(l1["a_log"], ql1["a_log"])
    assert _arrays_equal(l1["d_skip"], ql1["d_skip"])
    assert not _arrays_equal(l1["x_proj_w"], ql1["x_proj_w"])


def test_only_tensor_isolates_one_group() -> None:
    from aerocapture.training.quantize import quantize_model_weights

    m = _mamba_model(np.random.default_rng(5))
    q = quantize_model_weights(m, 4, "per_channel", "all", only_tensor="layer_1.a_log")
    assert not _arrays_equal(m["weights"]["layer_1"]["a_log"], q["weights"]["layer_1"]["a_log"])
    for i, fields in ((0, ("w", "b")), (2, ("w", "b"))):
        for f in fields:
            assert _arrays_equal(m["weights"][f"layer_{i}"][f], q["weights"][f"layer_{i}"][f])
    for f in ("x_proj_w", "dt_proj_w", "dt_proj_b", "d_skip"):
        assert _arrays_equal(m["weights"]["layer_1"][f], q["weights"]["layer_1"][f])


def test_only_tensor_unknown_key_raises() -> None:
    from aerocapture.training.quantize import quantize_model_weights

    with pytest.raises(ValueError, match="only_tensor"):
        quantize_model_weights(_mamba_model(np.random.default_rng(6)), 4, "per_channel", "all", only_tensor="layer_1.dt_proj_b")


def test_d_skip_identical_under_both_granularities() -> None:
    from aerocapture.training.quantize import quantize_model_weights

    m = _mamba_model(np.random.default_rng(7))
    qc = quantize_model_weights(m, 4, "per_channel", "all")
    qt = quantize_model_weights(m, 4, "per_tensor", "all")
    assert _arrays_equal(qc["weights"]["layer_1"]["d_skip"], qt["weights"]["layer_1"]["d_skip"])


def test_bad_tensor_policy_raises() -> None:
    from aerocapture.training.quantize import quantize_model_weights

    with pytest.raises(ValueError, match="tensor_policy"):
        quantize_model_weights(_mamba_model(np.random.default_rng(8)), 4, "per_channel", "matrices")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_quantize.py -v -m "not slow"`
Expected: new tests FAIL with `ImportError: cannot import name 'quantize_model_weights'`.

- [ ] **Step 3: Implement in `quantize.py`** (replace `quantize_dense_weights` and its docstring block; keep `_quantize_matrix`, `_layer_types`, `_GRANULARITIES` as-is)

```python
_TENSOR_POLICIES = ("all", "proj_only")


def _quantize_vector(v: npt.NDArray[np.float64], n_bits: int) -> npt.NDArray[np.float64]:
    """Symmetric fake-quant of a 1-D tensor: always a single per-tensor scale
    (per-channel on a vector would be per-element, i.e. lossless and meaningless)."""
    return _quantize_matrix(v.reshape(1, -1), n_bits, "per_tensor").reshape(-1)


def _quantizable_tensors(model_json: dict, tensor_policy: str) -> list[tuple[str, int, str, bool]]:
    """(key, layer_idx, field, is_1d) for every tensor the policy quantizes.

    dense: `w` only (biases stay fp). mamba: `x_proj_w` + `dt_proj_w` always;
    `a_log` + `d_skip` only under "all"; `dt_proj_b` never (it is a bias).
    """
    types = _layer_types(model_json)
    unsupported = sorted({t for t in types if t not in ("dense", "mamba")})
    if unsupported:
        raise ValueError(f"quantization supports dense+mamba models; found layer types {unsupported}")
    out: list[tuple[str, int, str, bool]] = []
    for i, t in enumerate(types):
        if t == "dense":
            out.append((f"layer_{i}.w", i, "w", False))
        else:
            out.append((f"layer_{i}.x_proj_w", i, "x_proj_w", False))
            out.append((f"layer_{i}.dt_proj_w", i, "dt_proj_w", False))
            if tensor_policy == "all":
                out.append((f"layer_{i}.a_log", i, "a_log", False))
                out.append((f"layer_{i}.d_skip", i, "d_skip", True))
    return out


def _validate_quant_args(n_bits: int, granularity: str, tensor_policy: str) -> None:
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2 (got {n_bits}); binary weights are out of scope")
    if granularity not in _GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r} (expected one of {_GRANULARITIES})")
    if tensor_policy not in _TENSOR_POLICIES:
        raise ValueError(f"unknown tensor_policy {tensor_policy!r} (expected one of {_TENSOR_POLICIES})")


def quantize_model_weights(
    model_json: dict,
    n_bits: int,
    granularity: str,
    tensor_policy: str = "all",
    only_tensor: str | None = None,
) -> dict:
    """Deep copy of model_json with the policy's tensors fake-quantized.

    `only_tensor` (a key from `_quantizable_tensors`, e.g. "layer_1.a_log")
    quantizes exactly that tensor group -- the leave-one-out probe. Biases,
    input_mask, normalization, output_param, architecture are never touched.
    """
    _validate_quant_args(n_bits, granularity, tensor_policy)
    targets = _quantizable_tensors(model_json, "all" if only_tensor is not None else tensor_policy)
    if only_tensor is not None:
        targets = [t for t in targets if t[0] == only_tensor]
        if not targets:
            known = [k for k, *_ in _quantizable_tensors(model_json, "all")]
            raise ValueError(f"unknown only_tensor {only_tensor!r} (expected one of {known})")
    out = copy.deepcopy(model_json)
    for _key, i, field, is_1d in targets:
        arr = np.asarray(out["weights"][f"layer_{i}"][field], dtype=np.float64)
        q = _quantize_vector(arr, n_bits) if is_1d else _quantize_matrix(arr, n_bits, granularity)
        out["weights"][f"layer_{i}"][field] = q.tolist()
    return out
```

Also retarget the one internal caller in `run_quant_sweep` (full rewrite lands in Task 7):

```python
                tmp.write_text(json.dumps(quantize_model_weights(model_json, b, gran)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quantize.py -v -m "not slow"`
Expected: PASS (including the renamed dense tests).

- [ ] **Step 5: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/quantize.py tests/test_quantize.py
git commit -m "feat(quantize): mamba-aware quantize_model_weights with tensor policy + LOO"
```

---

### Task 3: Extend the flat-path quantizer to Mamba

`quantize_flat_weights_batch` is the QAT kernel: it rounds the `(n_pop, n_w)` flat NN-weight matrix the GA searches. Extend it to mamba blocks using the canonical Rust flat order `[x_proj_w row-major, dt_proj_w row-major, dt_proj_b, a_log row-major, d_skip]` (see `impl LayerWeights for MambaLayer::to_flat` in `src/rust/src/data/neural/layers/mamba.rs:42`). The key gate is the flat/JSON agreement test: both paths must produce identical values.

**Files:**
- Modify: `src/python/aerocapture/training/quantize.py`
- Test: `tests/test_quantize.py`

**Interfaces:**
- Consumes: `_validate_quant_args` (Task 2), `resolve_mamba_dt_rank(entry: dict) -> int` from `aerocapture.training.config` (existing; resolves the `max(1, input_size // 16)` default).
- Produces: `quantize_flat_weights_batch(weights: NDArray[(n_pop, n_w)], architecture: list[dict], n_bits: int, granularity: str, tensor_policy: str = "all") -> NDArray` — Task 6's hooks and Task 5's ported tests call this 5-arg form.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_quantize.py`)

```python
_MAMBA_ARCH = [
    {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
    {"type": "mamba", "input_size": 4, "d_state": 2, "dt_rank": 1},
    {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
]
# flat widths: dense0 = 12w + 4b; mamba = 20 x_proj + 4 dt_proj_w + 4 dt_proj_b + 8 a_log + 4 d_skip; dense2 = 8w + 2b
_N_FLAT = 16 + 40 + 10  # 66


def _flat_to_model(flat: npt.NDArray[np.float64]) -> dict:
    """Slice a flat chromosome into the JSON weights layout (canonical Rust to_flat order)."""
    f = flat
    return {
        "format_version": 2,
        "architecture": [dict(e) for e in _MAMBA_ARCH],
        "weights": {
            "layer_0": {"w": f[0:12].reshape(4, 3).tolist(), "b": f[12:16].tolist()},
            "layer_1": {
                "x_proj_w": f[16:36].reshape(5, 4).tolist(),
                "dt_proj_w": f[36:40].reshape(4, 1).tolist(),
                "dt_proj_b": f[40:44].tolist(),
                "a_log": f[44:52].reshape(4, 2).tolist(),
                "d_skip": f[52:56].tolist(),
            },
            "layer_2": {"w": f[56:64].reshape(2, 4).tolist(), "b": f[64:66].tolist()},
        },
    }


@pytest.mark.parametrize("granularity", ["per_channel", "per_tensor"])
@pytest.mark.parametrize("tensor_policy", ["all", "proj_only"])
def test_flat_and_json_paths_agree(granularity: str, tensor_policy: str) -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch, quantize_model_weights

    rng = np.random.default_rng(11)
    flat = rng.standard_normal((3, _N_FLAT))
    q_flat = quantize_flat_weights_batch(flat, _MAMBA_ARCH, 4, granularity, tensor_policy)
    for row in range(3):
        q_json = quantize_model_weights(_flat_to_model(flat[row]), 4, granularity, tensor_policy)
        np.testing.assert_allclose(q_flat[row], _model_to_flat(q_json), rtol=0, atol=0)


def _model_to_flat(model: dict) -> npt.NDArray[np.float64]:
    w = model["weights"]
    parts = [
        np.asarray(w["layer_0"]["w"]).ravel(), np.asarray(w["layer_0"]["b"]).ravel(),
        np.asarray(w["layer_1"]["x_proj_w"]).ravel(), np.asarray(w["layer_1"]["dt_proj_w"]).ravel(),
        np.asarray(w["layer_1"]["dt_proj_b"]).ravel(), np.asarray(w["layer_1"]["a_log"]).ravel(),
        np.asarray(w["layer_1"]["d_skip"]).ravel(),
        np.asarray(w["layer_2"]["w"]).ravel(), np.asarray(w["layer_2"]["b"]).ravel(),
    ]
    return np.concatenate([p.astype(np.float64) for p in parts])


def test_flat_biases_and_dt_proj_b_pass_through() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    rng = np.random.default_rng(12)
    flat = rng.standard_normal((2, _N_FLAT))
    q = quantize_flat_weights_batch(flat, _MAMBA_ARCH, 3, "per_channel", "all")
    np.testing.assert_array_equal(q[:, 12:16], flat[:, 12:16])   # dense0 bias
    np.testing.assert_array_equal(q[:, 40:44], flat[:, 40:44])   # dt_proj_b
    np.testing.assert_array_equal(q[:, 64:66], flat[:, 64:66])   # dense2 bias


def test_flat_proj_only_keeps_dynamics_slabs() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    rng = np.random.default_rng(13)
    flat = rng.standard_normal((2, _N_FLAT))
    q = quantize_flat_weights_batch(flat, _MAMBA_ARCH, 3, "per_channel", "proj_only")
    np.testing.assert_array_equal(q[:, 44:52], flat[:, 44:52])   # a_log
    np.testing.assert_array_equal(q[:, 52:56], flat[:, 52:56])   # d_skip


def test_flat_scaffolding_tail_width_raises() -> None:
    """A 962+3 chromosome (live scaffolding) must never reach this function."""
    from aerocapture.training.quantize import quantize_flat_weights_batch

    flat = np.zeros((1, _N_FLAT + 3))
    with pytest.raises(ValueError, match="flat width"):
        quantize_flat_weights_batch(flat, _MAMBA_ARCH, 4, "per_channel", "all")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_quantize.py -v -m "not slow" -k "flat"`
Expected: FAIL — current `quantize_flat_weights_batch` raises `dense-only` for the mamba arch and takes 4 args.

- [ ] **Step 3: Replace `quantize_flat_weights_batch`**

```python
def quantize_flat_weights_batch(
    weights: npt.NDArray[np.float64],
    architecture: list[dict],
    n_bits: int,
    granularity: str,
    tensor_policy: str = "all",
) -> npt.NDArray[np.float64]:
    """Fake-quantize the quantizable blocks of a (n_pop, n_w) flat-weight matrix.

    Mirrors `quantize_model_weights` on the PSO/GA flat layout (dense: w then b;
    mamba: x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip -- the canonical Rust
    `LayerWeights::to_flat` order). Biases and policy-excluded slabs pass through.
    Operates on the NN-weight slab ONLY: scaffolding genes travel through
    run_grid overrides, never through this array (exact-width assert below).
    """
    from aerocapture.training.config import resolve_mamba_dt_rank

    _validate_quant_args(n_bits, granularity, tensor_policy)
    unsupported = sorted({str(e.get("type", "dense")) for e in architecture} - {"dense", "mamba"})
    if unsupported:
        raise ValueError(f"quantization supports dense+mamba architectures; found {unsupported}")

    qmax = 2 ** (n_bits - 1) - 1
    out = weights.astype(np.float64).copy()
    n_pop = out.shape[0]

    def q2d(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # block: (n_pop, rows, cols); per_channel = one scale per output row
        amax = np.max(np.abs(block), axis=2, keepdims=True) if granularity == "per_channel" else np.max(np.abs(block), axis=(1, 2), keepdims=True)
        scale = np.where(amax == 0.0, 1.0, amax / qmax)
        result: npt.NDArray[np.float64] = np.clip(np.round(block / scale), -qmax, qmax) * scale
        return result

    def q1d(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # block: (n_pop, n); 1-D tensors always take a single per-tensor scale
        amax = np.max(np.abs(block), axis=1, keepdims=True)
        scale = np.where(amax == 0.0, 1.0, amax / qmax)
        result: npt.NDArray[np.float64] = np.clip(np.round(block / scale), -qmax, qmax) * scale
        return result

    off = 0
    for e in architecture:
        t = str(e.get("type", "dense"))
        n_in = int(e["input_size"])
        if t == "dense":
            n_out = int(e["output_size"])
            wsize = n_out * n_in
            out[:, off : off + wsize] = q2d(out[:, off : off + wsize].reshape(n_pop, n_out, n_in)).reshape(n_pop, wsize)
            off += wsize + n_out  # biases fp
        else:  # mamba
            d_state = int(e["d_state"])
            dt_rank = resolve_mamba_dt_rank(e)
            rows = dt_rank + 2 * d_state
            sz = rows * n_in  # x_proj_w
            out[:, off : off + sz] = q2d(out[:, off : off + sz].reshape(n_pop, rows, n_in)).reshape(n_pop, sz)
            off += sz
            sz = n_in * dt_rank  # dt_proj_w (per_channel = per row = per element at dt_rank 1: lossless by construction)
            out[:, off : off + sz] = q2d(out[:, off : off + sz].reshape(n_pop, n_in, dt_rank)).reshape(n_pop, sz)
            off += sz
            off += n_in  # dt_proj_b: bias, fp
            sz = n_in * d_state  # a_log
            if tensor_policy == "all":
                out[:, off : off + sz] = q2d(out[:, off : off + sz].reshape(n_pop, n_in, d_state)).reshape(n_pop, sz)
            off += sz
            if tensor_policy == "all":  # d_skip
                out[:, off : off + n_in] = q1d(out[:, off : off + n_in])
            off += n_in
    if off != out.shape[1]:
        raise ValueError(f"architecture flat width {off} != weights width {out.shape[1]}")
    return out
```

- [ ] **Step 4: Run the full test file**

Run: `uv run pytest tests/test_quantize.py -v -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/quantize.py tests/test_quantize.py
git commit -m "feat(quantize): mamba flat-weight quantization + flat/JSON agreement gate"
```

---

### Task 4: Analytic memory footprint helper

Exact deployed-model bytes per (bits, granularity, policy): b-bit-packed quantized params + one f32 scale per scale-group + f32 for every fp-kept param. This produces the appendix memory table.

**Files:**
- Modify: `src/python/aerocapture/training/quantize.py`
- Test: `tests/test_quantize.py`

**Interfaces:**
- Consumes: `resolve_mamba_dt_rank`, `_validate_quant_args`.
- Produces: `memory_footprint(architecture: list[dict], n_bits: int, granularity: str, tensor_policy: str = "all") -> dict[str, int]` with keys `quant_params, fp_params, n_scales, quant_bytes, scale_bytes, fp_bytes, total_bytes, f64_baseline_bytes`. Task 7 embeds its rows in the sweep JSON.

- [ ] **Step 1: Write the failing tests** (append; champion-arch exact numbers, hand-derived in the spec's tensor table)

```python
_CHAMPION_ARCH = [
    {"type": "dense", "input_size": 17, "output_size": 16, "activation": "swish"},
    {"type": "mamba", "input_size": 16, "d_state": 12, "dt_rank": 1},
    {"type": "dense", "input_size": 16, "output_size": 2, "activation": "asinh"},
]


def test_memory_footprint_champion_int8_per_tensor_all() -> None:
    from aerocapture.training.quantize import memory_footprint

    m = memory_footprint(_CHAMPION_ARCH, 8, "per_tensor", "all")
    assert m["quant_params"] == 928 and m["fp_params"] == 34
    assert m["n_scales"] == 6  # dense0.w, x_proj_w, dt_proj_w, a_log, d_skip, dense2.w
    assert m["quant_bytes"] == 928 and m["scale_bytes"] == 24 and m["fp_bytes"] == 136
    assert m["total_bytes"] == 1088
    assert m["f64_baseline_bytes"] == 962 * 8  # 7696


def test_memory_footprint_champion_int4_per_tensor_all() -> None:
    from aerocapture.training.quantize import memory_footprint

    m = memory_footprint(_CHAMPION_ARCH, 4, "per_tensor", "all")
    assert m["quant_bytes"] == 464  # ceil(928 * 4 / 8)
    assert m["total_bytes"] == 624


def test_memory_footprint_champion_per_channel_scale_count() -> None:
    from aerocapture.training.quantize import memory_footprint

    m = memory_footprint(_CHAMPION_ARCH, 8, "per_channel", "all")
    # dense0 16 rows + x_proj 25 + dt_proj_w 16 + a_log 16 + d_skip 1 (per-tensor rule) + dense2 2
    assert m["n_scales"] == 76


def test_memory_footprint_proj_only_moves_dynamics_to_fp() -> None:
    from aerocapture.training.quantize import memory_footprint

    m = memory_footprint(_CHAMPION_ARCH, 8, "per_tensor", "proj_only")
    assert m["quant_params"] == 720 and m["fp_params"] == 242
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_quantize.py -v -k memory_footprint`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement** (`import math` at module top if absent)

```python
def memory_footprint(architecture: list[dict], n_bits: int, granularity: str, tensor_policy: str = "all") -> dict[str, int]:
    """Analytic deployed-model bytes: b-bit-packed quantized params + f32 scales + f32 fp params.

    fp-kept parameters are costed at f32 (the realistic flight deployment width),
    quantized parameters at ceil(n * b / 8) packed bytes, scales at f32 each.
    """
    from aerocapture.training.config import resolve_mamba_dt_rank

    _validate_quant_args(n_bits, granularity, tensor_policy)
    quant = fp = scales = 0
    for e in architecture:
        t = str(e.get("type", "dense"))
        n_in = int(e["input_size"])
        if t == "dense":
            n_out = int(e["output_size"])
            quant += n_out * n_in
            scales += n_out if granularity == "per_channel" else 1
            fp += n_out  # bias
        elif t == "mamba":
            d_state = int(e["d_state"])
            dt_rank = resolve_mamba_dt_rank(e)
            rows = dt_rank + 2 * d_state
            quant += rows * n_in + n_in * dt_rank  # x_proj_w + dt_proj_w
            scales += (rows + n_in) if granularity == "per_channel" else 2
            fp += n_in  # dt_proj_b
            if tensor_policy == "all":
                quant += n_in * d_state + n_in  # a_log + d_skip
                scales += (n_in if granularity == "per_channel" else 1) + 1  # a_log rows + d_skip per-tensor
            else:
                fp += n_in * d_state + n_in
        else:
            raise ValueError(f"quantization supports dense+mamba architectures; found {t!r}")
    quant_bytes = math.ceil(quant * n_bits / 8)
    return {
        "quant_params": quant,
        "fp_params": fp,
        "n_scales": scales,
        "quant_bytes": quant_bytes,
        "scale_bytes": scales * 4,
        "fp_bytes": fp * 4,
        "total_bytes": quant_bytes + scales * 4 + fp * 4,
        "f64_baseline_bytes": (quant + fp) * 8,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_quantize.py -v -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/quantize.py tests/test_quantize.py
git commit -m "feat(quantize): analytic memory_footprint table helper"
```

---

### Task 5: QAT config plumbing (NetworkConfig + validate_qat + train.py)

Port the branch's `qat_bits`/`qat_granularity` knobs onto current main's `config.py`, add `qat_tensor_policy`, relax validation to dense+mamba, and plumb the three `[network]` TOML keys through `build_training_config_from_toml` in `train.py`.

**Files:**
- Modify: `src/python/aerocapture/training/config.py` (NetworkConfig, ~line 27)
- Modify: `src/python/aerocapture/training/train.py` (`build_training_config_from_toml`, after the `input_mask` plumbing at ~line 2438)
- Create: `tests/test_qat_training.py` (port from branch + extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `NetworkConfig.qat_bits: int | None = None`, `.qat_granularity: str = "per_channel"`, `.qat_tensor_policy: str = "all"`; `validate_qat(qat_bits: int | None, qat_granularity: str, qat_tensor_policy: str, architecture: list[dict] | None) -> None` (module-level in config.py). Task 6's hooks read the three fields; Task 9's TOMLs set them.

- [ ] **Step 1: Port the branch test file, then extend it**

```bash
git checkout feature/quantization -- tests/test_qat_training.py
```

Edit the ported file: change `test_network_config_qat_non_dense_raises`'s expected match from `"dense-only"` to `"dense\\+mamba"`, and append:

```python
def test_network_config_qat_mamba_accepted() -> None:
    arch = [
        {"type": "dense", "input_size": 17, "output_size": 16, "activation": "swish"},
        {"type": "mamba", "input_size": 16, "d_state": 12},
        {"type": "dense", "input_size": 16, "output_size": 2, "activation": "asinh"},
    ]
    net = NetworkConfig(architecture=arch, qat_bits=4, qat_tensor_policy="proj_only")
    assert net.qat_bits == 4
    assert net.qat_tensor_policy == "proj_only"


def test_network_config_qat_mamba3_rejected() -> None:
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "mamba3", "input_size": 4, "d_state": 2, "dt_rank": 1, "discretization": "euler", "state_mode": "real"},
    ]
    with pytest.raises(ValueError, match="dense\\+mamba"):
        NetworkConfig(architecture=arch, qat_bits=4)


def test_network_config_qat_bad_policy_raises() -> None:
    with pytest.raises(ValueError, match="qat_tensor_policy"):
        NetworkConfig(architecture=list(_DENSE2), qat_bits=4, qat_tensor_policy="matrices")


def test_network_config_qat_policy_ignored_when_off() -> None:
    net = NetworkConfig(architecture=list(_DENSE2), qat_tensor_policy="nonsense")  # qat_bits None -> no validation
    assert net.qat_bits is None
```

Note: the two `test_qat_*_hook_*` tests in the ported file exercise Task 6's hooks — they will FAIL until Task 6 lands. Deselect them in this task's runs with `-k "not hook"`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_qat_training.py -v -k "not hook"`
Expected: FAIL — `NetworkConfig` has no `qat_bits` field.

- [ ] **Step 3: Implement in `config.py`** — add the three fields to `NetworkConfig` after `warm_start_from: str | None = None` (line 46), the validate call as the FIRST statement of `__post_init__` body after the scaffolding check, and the module-level function above `resolve_mamba_dt_rank`:

```python
    warm_start_from: str | None = None
    qat_bits: int | None = None
    qat_granularity: str = "per_channel"
    qat_tensor_policy: str = "all"
```

In `__post_init__`, right after the scaffolding ValueError block:

```python
        validate_qat(self.qat_bits, self.qat_granularity, self.qat_tensor_policy, self.architecture)
```

Module-level:

```python
def validate_qat(qat_bits: int | None, qat_granularity: str, qat_tensor_policy: str, architecture: list[dict] | None) -> None:
    """Validate QAT settings. dense+mamba only, b>=2, known granularity/policy. No-op when qat_bits is None."""
    if qat_bits is None:
        return
    if qat_bits < 2:
        raise ValueError(f"qat_bits must be >= 2 (got {qat_bits}); binary weights are out of scope")
    if qat_granularity not in ("per_channel", "per_tensor"):
        raise ValueError(f"qat_granularity must be 'per_channel' or 'per_tensor', got {qat_granularity!r}")
    if qat_tensor_policy not in ("all", "proj_only"):
        raise ValueError(f"qat_tensor_policy must be 'all' or 'proj_only', got {qat_tensor_policy!r}")
    if architecture is not None:
        unsupported = sorted({str(e.get("type", "dense")) for e in architecture} - {"dense", "mamba"})
        if unsupported:
            raise ValueError(f"qat_bits supports dense+mamba networks; found layer types {unsupported}")
```

- [ ] **Step 4: Plumb TOML keys in `train.py`** — in `build_training_config_from_toml`, right after the `if "input_mask" in _net:` block (~line 2439):

```python
    if "qat_bits" in _net:
        cfg.network.qat_bits = int(_net["qat_bits"])
    if "qat_granularity" in _net:
        cfg.network.qat_granularity = str(_net["qat_granularity"])
    if "qat_tensor_policy" in _net:
        cfg.network.qat_tensor_policy = str(_net["qat_tensor_policy"])
    # Post-hoc field assignment bypasses NetworkConfig.__post_init__, so validate here.
    from aerocapture.training.config import validate_qat

    validate_qat(cfg.network.qat_bits, cfg.network.qat_granularity, cfg.network.qat_tensor_policy, cfg.network.architecture)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_qat_training.py -v -k "not hook"`
Expected: PASS.

- [ ] **Step 6: Regression check on untouched config behavior + lint + commit**

Run: `uv run pytest tests/test_config.py tests/test_toml_to_config.py -q` (paths exist on main; if a file is named differently, run `uv run pytest tests -q -k "config"`)
Expected: PASS.

```bash
./lint_code.sh
git add src/python/aerocapture/training/config.py src/python/aerocapture/training/train.py tests/test_qat_training.py
git commit -m "feat(qat): qat_bits/qat_granularity/qat_tensor_policy knobs, dense+mamba validation"
```

---

### Task 6: QAT eval + deploy hooks

Two hook points, both from the branch, re-applied to current main's code layout: (1) `problem._run_grid_records` — the single chokepoint for per-gen fitness, the validation gate (`evaluate_individual_per_seed`), gen-0 baselines, and final selection — rounds the population weights before `run_grid`; (2) `evaluate.write_nn_json` rounds at deploy so `best_model.json` IS the quantized policy. Plus the end-to-end mamba QAT smoke.

**Files:**
- Modify: `src/python/aerocapture/training/problem.py` (`_run_grid_records`, after line 152)
- Modify: `src/python/aerocapture/training/evaluate.py` (`write_nn_json`, ~line 210)
- Test: `tests/test_qat_training.py`

**Interfaces:**
- Consumes: `quantize_flat_weights_batch` (Task 3, 5-arg), `NetworkConfig.qat_*` (Task 5).
- Produces: every eval and deploy path sees the rounded policy when `qat_bits` is set. No signature changes anywhere.

- [ ] **Step 1: Verify the two ported hook tests fail**

The ported `tests/test_qat_training.py` already contains `test_qat_eval_hook_quantizes_weights` (monkeypatches `aerocapture_rs.run_grid`, asserts the captured `weights` kwarg equals `quantize_flat_weights_batch(decoded, ...)`) and `test_qat_deploy_hook_quantizes_written_weights` (writes via `write_nn_json`, re-derives the flat vector from the JSON, asserts on-grid). Both call `quantize_flat_weights_batch(x, arch, 4, "per_channel")` — update both call sites to the 5-arg form ending in `"all"` so they pin the default policy. If `AerocaptureProblem.__init__` kwargs drifted since 2026-06-05, adapt the constructor call to the current signature (`src/python/aerocapture/training/problem.py`), changing nothing else.

Run: `uv run pytest tests/test_qat_training.py -v -k "hook"`
Expected: FAIL — hooks not yet installed (weights reach run_grid un-rounded).

- [ ] **Step 2: Install the eval hook in `problem.py`** — in `_run_grid_records`, immediately after `delta_max = getattr(nn_cfg, "delta_max", 0.35)` (line 152):

```python
            if nn_cfg.qat_bits is not None:
                from aerocapture.training.quantize import quantize_flat_weights_batch

                weights = quantize_flat_weights_batch(
                    weights, build_v2_architecture(nn_cfg), nn_cfg.qat_bits, nn_cfg.qat_granularity, nn_cfg.qat_tensor_policy
                )
```

- [ ] **Step 3: Install the deploy hook in `evaluate.py`** — in `write_nn_json`, replace the direct `flat_weights_to_json(flat=weights.astype(np.float64).tolist(), ...)` call head with:

```python
    arch = build_v2_architecture(network)
    flat = weights.astype(np.float64)
    qat_bits = getattr(network, "qat_bits", None)
    if qat_bits is not None:
        from aerocapture.training.quantize import quantize_flat_weights_batch

        flat = quantize_flat_weights_batch(flat.reshape(1, -1), arch, qat_bits, network.qat_granularity, network.qat_tensor_policy)[0]
    _aero_rs.flat_weights_to_json(
        flat=flat.tolist(),
```

(the remaining kwargs of the call are unchanged).

- [ ] **Step 4: Run the hook tests**

Run: `uv run pytest tests/test_qat_training.py -v`
Expected: PASS (needs `aerocapture_rs` built; if not built: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml` from repo root).

- [ ] **Step 5: Write the @slow end-to-end mamba QAT smoke** (append to `tests/test_qat_training.py`; `import subprocess, sys` at top)

```python
@pytest.mark.slow
def test_qat_mamba_end_to_end_deploy_idempotent(tmp_path: Path) -> None:
    """2 real GA gens with qat_bits=8 on a tiny dense->mamba->dense arch; the
    deployed best_model.json must be idempotent under re-quantization (i.e. the
    deploy writer actually rounded the weights)."""
    pytest.importorskip("aerocapture_rs")
    from aerocapture.training.quantize import quantize_model_weights

    repo = Path(__file__).resolve().parents[1]
    base = (repo / "configs/training/msr_aller_nn_atan2_train.toml").resolve()
    out_dir = tmp_path / "out"
    toml_text = f"""
base = ["{base}"]

[data]
neural_network = "{out_dir / 'best_model.json'}"
results_suffix = ".qat_smoke"

[network]
qat_bits = 8
qat_granularity = "per_channel"
qat_tensor_policy = "all"

[[network.architecture]]
type = "dense"
input_size = 17
output_size = 4
activation = "swish"

[[network.architecture]]
type = "mamba"
input_size = 4
d_state = 2

[[network.architecture]]
type = "dense"
input_size = 4
output_size = 2
activation = "asinh"

[optimizer]
algorithm = "ga"
n_pop = 4
n_gen = 2
training_n_sims = 2
validation_n_sims = 2
seed_strategy = "fixed"
"""
    cfg = tmp_path / "qat_smoke.toml"
    cfg.write_text(toml_text)
    subprocess.run(
        [sys.executable, "-m", "aerocapture.training.train", str(cfg), "--no-tui", "--skip-report", "--output-dir", str(out_dir), "--from-scratch"],
        check=True,
        cwd=repo,
        timeout=600,
    )
    model = json.loads((out_dir / "best_model.json").read_text())
    assert [e["type"] for e in model["architecture"]] == ["dense", "mamba", "dense"]
    q = quantize_model_weights(model, 8, "per_channel", "all")
    for i in range(3):
        for field in model["weights"][f"layer_{i}"]:
            np.testing.assert_allclose(
                np.asarray(model["weights"][f"layer_{i}"][field], dtype=np.float64),
                np.asarray(q["weights"][f"layer_{i}"][field], dtype=np.float64),
                rtol=0,
                atol=1e-12,
                err_msg=f"layer_{i}.{field} not on the 8-bit grid",
            )
```

- [ ] **Step 6: Run the smoke**

Run: `uv run pytest tests/test_qat_training.py -v -m slow`
Expected: PASS in ~1-3 min (4 individuals x 2 seeds x 2 gens + 2 validation sims).

- [ ] **Step 7: Full-suite regression + lint + commit**

Run: `uv run pytest tests -q -m "not slow" -x`
Expected: PASS — the hooks are no-ops when `qat_bits is None`, so nothing else may move.

```bash
./lint_code.sh
git add src/python/aerocapture/training/problem.py src/python/aerocapture/training/evaluate.py tests/test_qat_training.py
git commit -m "feat(qat): quantize population evals + deployed weights for dense+mamba QAT"
```

---

### Task 7: Sweep runner rewrite (reserved pool, scaffolding, LOO, finalists) + charts

Rewrite `run_quant_sweep` around the probe-eval methodology: reserved fresh pool, per-seed `run_batch`, pinned model path, co-trained scaffolding overrides, tail-led metrics, the 4-bit LOO pass with a pre-registered verdict rule, embedded memory rows, and a finalists mode for the n=10k re-scores. Upgrade the charts.

**Files:**
- Modify: `src/python/aerocapture/training/quantize.py` (replace `run_quant_sweep`, `_variant_metrics`, `_print_table`, `main`)
- Modify: `src/python/aerocapture/training/charts_quant.py`
- Test: `tests/test_quantize.py`

**Interfaces:**
- Consumes: `quantize_model_weights` + `_quantizable_tensors` (Task 2), `memory_footprint` (Task 4), `make_reserved_seeds` + `HEADLINE_REQUOTE_SEED_OFFSET` from `aerocapture.training.evaluate`, `_load_nn_scaffolding_overrides` + `compute_eval_summary` from `aerocapture.training.report`, `cvar95` from `aerocapture.training.experiments.probe_common`, `load_toml_with_bases` from `aerocapture.training.toml_utils`, `_load_cost_kwargs` from `aerocapture.training.ablation`.
- Produces:
  - `_score_variant(toml_path, model_path, seeds, cost_kwargs, extra_overrides, sim_timeout_secs) -> dict` (capture_rate, dv_p50, dv_p95, dv_p99, dv_cvar95, viol_pct, rms_cost)
  - `run_quant_sweep(toml_path, model_path, params_dir=None, bits=(8, 6, 4, 3, 2), granularities=("per_channel", "per_tensor"), policies=("all", "proj_only"), n_sims=1000, pool_offset=None (resolves to HEADLINE_REQUOTE_SEED_OFFSET), loo_bits=4, sim_timeout_secs=None, cost_transform="linear") -> dict` — result dict with keys `baseline, variants, loo, verdict, memory, pool, n_sims`
  - `_pick_verdict(variants: list[dict], bits: int) -> dict` (pre-registered rule: capture desc, CVaR95 asc, per_channel then all on ties)
  - `run_finalists(toml_path, entries: list[dict], n_sims=10000, pool_offset=..., sim_timeout_secs=None, cost_transform="linear") -> dict` — entry = `{"label": str, "model": str, "params_dir": str | None, "quantize": None | {"bits": int, "granularity": str, "tensor_policy": str}}`
  - Charts: `chart_quant_sweep(results, output_path)` (capture + CVaR95 vs bits, one line per (granularity, policy)), `chart_quant_loo(results, output_path)`, `chart_qat_convergence(jsonl_by_label: dict[str, list[Path]], output_path)`
  - CLI: `python -m aerocapture.training.quantize <output_dir> --toml T --model M [--params-dir D] [--n-sims N] [--bits ...] [--granularity ...] [--policies ...] [--loo-bits B|--no-loo] [--pool-offset K] [--sim-timeout S] [--finalists entries.json]`

- [ ] **Step 1: Write the failing unit tests for the pure pieces** (append to `tests/test_quantize.py`)

```python
def _variant(bits: int, gran: str, policy: str, capture: float, cvar: float) -> dict:
    return {"bits": bits, "granularity": gran, "tensor_policy": policy, "capture_rate": capture, "dv_cvar95": cvar}


def test_pick_verdict_prefers_capture_then_cvar() -> None:
    from aerocapture.training.quantize import _pick_verdict

    variants = [
        _variant(4, "per_tensor", "all", 0.99, 110.0),
        _variant(4, "per_channel", "proj_only", 1.0, 118.0),
        _variant(4, "per_channel", "all", 1.0, 116.0),
        _variant(8, "per_channel", "all", 1.0, 100.0),  # wrong bits: ignored
    ]
    v = _pick_verdict(variants, 4)
    assert (v["granularity"], v["tensor_policy"]) == ("per_channel", "all")


def test_pick_verdict_tie_breaks_toward_per_channel_all() -> None:
    from aerocapture.training.quantize import _pick_verdict

    variants = [
        _variant(4, "per_tensor", "proj_only", 1.0, 115.0),
        _variant(4, "per_channel", "all", 1.0, 115.0),
    ]
    v = _pick_verdict(variants, 4)
    assert (v["granularity"], v["tensor_policy"]) == ("per_channel", "all")


def test_pick_verdict_nan_cvar_ranks_last() -> None:
    from aerocapture.training.quantize import _pick_verdict

    variants = [
        _variant(4, "per_channel", "all", 1.0, float("nan")),
        _variant(4, "per_tensor", "all", 1.0, 120.0),
    ]
    assert _pick_verdict(variants, 4)["granularity"] == "per_tensor"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_quantize.py -v -k verdict`
Expected: FAIL with ImportError.

- [ ] **Step 3: Rewrite the runner in `quantize.py`** — delete `_variant_metrics` and the old `run_quant_sweep`/`_print_table`/`main`; the old imports of `_mean_per_sim_cost`/`_resolve_nn_path`/`is_captured`/`DV_TOTAL_RAW_INDEX` go away with them. New code:

```python
def _score_variant(
    toml_path: str,
    model_path: str | Path,
    seeds: list[int],
    cost_kwargs: dict[str, Any],
    extra_overrides: dict[str, Any],
    sim_timeout_secs: float | None,
) -> dict[str, Any]:
    """One MC batch on an explicit seed list for a pinned model; tail-led metrics."""
    import aerocapture_rs

    from aerocapture.training import charts
    from aerocapture.training.experiments.probe_common import cvar95
    from aerocapture.training.report import compute_eval_summary

    overrides = [
        {"simulation.n_sims": 1, "data.neural_network": str(model_path), "monte_carlo.seed": int(s), **extra_overrides} for s in seeds
    ]
    batch = aerocapture_rs.run_batch(toml_path, overrides, n_threads=None, include_trajectories=False, sim_timeout_secs=sim_timeout_secs)
    final = np.array(batch.final_records, dtype=np.float64)
    summary = compute_eval_summary(final, n_sims=len(seeds), cost_kwargs=cost_kwargs)
    captured = charts.is_captured(final)
    dv = np.clip(final[captured, charts._FR_DV_TOTAL], charts.DV_FLOOR, charts.DV_CAP)
    viol = max(float(c["viol_pct"]) for c in summary["constraints"].values()) if summary["constraints"] else 0.0
    return {
        "capture_rate": float(summary["capture_rate"]),
        "dv_p50": float(np.percentile(dv, 50)) if dv.size else None,
        "dv_p95": float(np.percentile(dv, 95)) if dv.size else None,
        "dv_p99": float(np.percentile(dv, 99)) if dv.size else None,
        "dv_cvar95": cvar95(dv) if dv.size else None,
        "viol_pct": viol,
        "rms_cost": float(summary["cost"]["rms"]),
    }


def _resolve_pool(toml_path: str, pool_offset: int, n_sims: int) -> tuple[list[int], dict[str, Any]]:
    from aerocapture.training.evaluate import make_reserved_seeds
    from aerocapture.training.toml_utils import load_toml_with_bases

    base_mc_seed = int(load_toml_with_bases(Path(toml_path)).get("monte_carlo", {}).get("seed", 42))
    seeds = make_reserved_seeds(base_mc_seed, pool_offset, n_sims)
    return seeds, {"base_mc_seed": base_mc_seed, "offset": pool_offset, "n": n_sims}


def _scaffolding_overrides(params_dir: str | Path | None) -> dict[str, Any]:
    if params_dir is None:
        return {}
    from aerocapture.training.report import _load_nn_scaffolding_overrides

    d = Path(params_dir)
    return dict(_load_nn_scaffolding_overrides(d, d / f"optimized_{d.name}.toml"))


def _pick_verdict(variants: list[dict[str, Any]], bits: int) -> dict[str, Any]:
    """Pre-registered QAT-cell rule: among the `bits` cells, max capture rate,
    then min CVaR95 (NaN/None last), ties break toward per_channel then all."""
    cells = [v for v in variants if v["bits"] == bits]
    if not cells:
        raise ValueError(f"no {bits}-bit cells in the sweep grid")

    def key(v: dict[str, Any]) -> tuple[float, float, int, int]:
        cvar = v.get("dv_cvar95")
        cvar_f = float(cvar) if cvar is not None and np.isfinite(cvar) else float("inf")
        return (-round(float(v["capture_rate"]), 3), cvar_f, int(v["granularity"] != "per_channel"), int(v["tensor_policy"] != "all"))

    return min(cells, key=key)


def run_quant_sweep(
    toml_path: str,
    model_path: str | Path,
    params_dir: str | Path | None = None,
    bits: tuple[int, ...] = (8, 6, 4, 3, 2),
    granularities: tuple[str, ...] = ("per_channel", "per_tensor"),
    policies: tuple[str, ...] = ("all", "proj_only"),
    n_sims: int = 1000,
    pool_offset: int | None = None,
    loo_bits: int | None = 4,
    sim_timeout_secs: float | None = None,
    cost_transform: str | None = "linear",
) -> dict[str, Any]:
    """PTQ sensitivity sweep on a reserved pool with the co-trained scaffolding applied.

    Grid: bits x granularity x tensor_policy, each scored on the SAME seeds as the
    fp baseline, so every delta is pure quantization effect. When `loo_bits` is set,
    a leave-one-out pass quantizes one tensor group at a time at that bit width
    (granularity taken from the verdict cell). Memory rows via `memory_footprint`.
    """
    from aerocapture.training.ablation import _load_cost_kwargs
    from aerocapture.training.evaluate import HEADLINE_REQUOTE_SEED_OFFSET

    offset = HEADLINE_REQUOTE_SEED_OFFSET if pool_offset is None else pool_offset
    seeds, pool = _resolve_pool(toml_path, offset, n_sims)
    scaff = _scaffolding_overrides(params_dir)
    cost_kwargs = _load_cost_kwargs(toml_path, cost_transform=cost_transform)
    model_json = json.loads(Path(model_path).read_text())

    baseline = _score_variant(toml_path, model_path, seeds, cost_kwargs, scaff, sim_timeout_secs)

    def deltas(m: dict[str, Any]) -> dict[str, Any]:
        m["delta_capture_rate"] = m["capture_rate"] - baseline["capture_rate"]
        if m["dv_cvar95"] is not None and baseline["dv_cvar95"] is not None:
            m["delta_dv_cvar95"] = m["dv_cvar95"] - baseline["dv_cvar95"]
        else:
            m["delta_dv_cvar95"] = None
        return m

    variants: list[dict[str, Any]] = []
    loo: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "quant_model.json"
        for gran in granularities:
            for policy in policies:
                for b in bits:
                    tmp.write_text(json.dumps(quantize_model_weights(model_json, b, gran, policy)))
                    m = _score_variant(toml_path, tmp, seeds, cost_kwargs, scaff, sim_timeout_secs)
                    m.update({"granularity": gran, "tensor_policy": policy, "bits": b})
                    variants.append(deltas(m))
                    print(f"  scored bits={b} gran={gran} policy={policy}: capture={m['capture_rate']:.3f}")

        verdict = _pick_verdict(variants, loo_bits) if loo_bits is not None else None
        if loo_bits is not None:
            assert verdict is not None
            for key, *_ in _quantizable_tensors(model_json, "all"):
                tmp.write_text(json.dumps(quantize_model_weights(model_json, loo_bits, verdict["granularity"], "all", only_tensor=key)))
                m = _score_variant(toml_path, tmp, seeds, cost_kwargs, scaff, sim_timeout_secs)
                m.update({"tensor": key, "bits": loo_bits, "granularity": verdict["granularity"]})
                loo.append(deltas(m))
                print(f"  scored LOO {key}: capture={m['capture_rate']:.3f}")

    memory = [
        {"bits": b, "granularity": g, "tensor_policy": p, **memory_footprint(model_json["architecture"], b, g, p)}
        for g in granularities
        for p in policies
        for b in bits
    ]
    return {
        "baseline": baseline,
        "variants": variants,
        "loo": loo,
        "verdict": verdict,
        "memory": memory,
        "pool": pool,
        "n_sims": n_sims,
        "model_path": str(model_path),
        "params_dir": str(params_dir) if params_dir is not None else None,
        "scaffolding_applied": sorted(scaff),
    }


def run_finalists(
    toml_path: str,
    entries: list[dict[str, Any]],
    n_sims: int = 10000,
    pool_offset: int | None = None,
    sim_timeout_secs: float | None = None,
    cost_transform: str | None = "linear",
) -> dict[str, Any]:
    """Deep re-score (default n=10000) of finalist models on the same reserved pool.

    Entry: {"label", "model", "params_dir", "quantize": None | {"bits", "granularity", "tensor_policy"}}.
    QAT-deployed models pass quantize=None (their best_model.json is already on-grid);
    the PTQ finalist passes the verdict cell so the champion is rounded on the fly.
    """
    from aerocapture.training.ablation import _load_cost_kwargs
    from aerocapture.training.evaluate import HEADLINE_REQUOTE_SEED_OFFSET

    offset = HEADLINE_REQUOTE_SEED_OFFSET if pool_offset is None else pool_offset
    seeds, pool = _resolve_pool(toml_path, offset, n_sims)
    cost_kwargs = _load_cost_kwargs(toml_path, cost_transform=cost_transform)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, e in enumerate(entries):
            model_path = Path(e["model"])
            if e.get("quantize") is not None:
                q = e["quantize"]
                tmp = Path(tmpdir) / f"finalist_{i}.json"
                tmp.write_text(json.dumps(quantize_model_weights(json.loads(model_path.read_text()), int(q["bits"]), str(q["granularity"]), str(q["tensor_policy"]))))
                model_path = tmp
            m = _score_variant(toml_path, model_path, seeds, cost_kwargs, _scaffolding_overrides(e.get("params_dir")), sim_timeout_secs)
            rows.append({"label": e["label"], "model": e["model"], "quantize": e.get("quantize"), **m})
            print(f"  finalist {e['label']}: capture={m['capture_rate']:.3f}")
    return {"finalists": rows, "pool": pool, "n_sims": n_sims}


def _print_table(results: dict[str, Any]) -> None:
    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.1f}"

    b = results["baseline"]
    print(f"baseline (fp): capture={b['capture_rate']:.3f}  dv_p50={fmt(b['dv_p50'])}  dv_p95={fmt(b['dv_p95'])}  cvar95={fmt(b['dv_cvar95'])}")
    print(f"{'gran':<12}{'policy':<11}{'bits':>5}{'capture':>9}{'d_cap':>9}{'dv_p50':>9}{'dv_p95':>9}{'cvar95':>9}{'d_cvar':>9}{'viol%':>7}")
    for v in results["variants"]:
        d_cvar = "-" if v["delta_dv_cvar95"] is None else f"{v['delta_dv_cvar95']:+.1f}"
        print(
            f"{v['granularity']:<12}{v['tensor_policy']:<11}{v['bits']:>5}{v['capture_rate']:>9.3f}{v['delta_capture_rate']:>+9.3f}"
            f"{fmt(v['dv_p50']):>9}{fmt(v['dv_p95']):>9}{fmt(v['dv_cvar95']):>9}{d_cvar:>9}{v['viol_pct']:>7.2f}"
        )
    for r in results["loo"]:
        print(f"LOO {r['tensor']:<22}{r['bits']:>3}b  capture={r['capture_rate']:.3f}  cvar95={fmt(r['dv_cvar95'])}  d_cvar={'-' if r['delta_dv_cvar95'] is None else f'{r['delta_dv_cvar95']:+.1f}'}")
    if results.get("verdict"):
        v = results["verdict"]
        print(f"verdict (QAT cell @ {v['bits']}b): granularity={v['granularity']} tensor_policy={v['tensor_policy']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Weight-only PTQ sweep / finalists re-score for the NN guidance policy")
    parser.add_argument("output_dir", help="directory for quantization_results.json + SVGs")
    parser.add_argument("--toml", required=True, help="training config that resolves the mission pipeline (e.g. configs/training/sweep/mamba_p962.toml)")
    parser.add_argument("--model", required=True, help="model JSON to quantize/evaluate (pinned; the TOML deploy path is never read)")
    parser.add_argument("--params-dir", default=None, help="training dir whose best_params.json carries the co-trained scaffolding overrides")
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    parser.add_argument("--granularity", nargs="+", default=["per_channel", "per_tensor"], choices=list(_GRANULARITIES))
    parser.add_argument("--policies", nargs="+", default=["all", "proj_only"], choices=list(_TENSOR_POLICIES))
    parser.add_argument("--loo-bits", type=int, default=4)
    parser.add_argument("--no-loo", action="store_true")
    parser.add_argument("--pool-offset", type=int, default=None, help="reserved-pool offset (default HEADLINE_REQUOTE_SEED_OFFSET = 8M)")
    parser.add_argument("--sim-timeout", type=float, default=None)
    parser.add_argument("--finalists", default=None, help="JSON file with finalist entries; switches to the deep re-score mode")
    parser.add_argument(
        "--cost-transform",
        default="linear",
        choices=["linear", "sqrt", "log", "squared", "cubed"],
        help="rescaling for the reported rms_cost; default linear = interpretable DV+penalties",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.finalists is not None:
        entries = json.loads(Path(args.finalists).read_text())
        results = run_finalists(args.toml, entries, n_sims=args.n_sims, pool_offset=args.pool_offset, sim_timeout_secs=args.sim_timeout, cost_transform=args.cost_transform)
        (out_dir / "finalists_results.json").write_text(json.dumps(results, indent=2))
        for r in results["finalists"]:
            cells = " ".join(f"{k}={'-' if r[k] is None else f'{r[k]:.1f}'}" for k in ("dv_p50", "dv_p95", "dv_p99", "dv_cvar95"))
            print(f"{r['label']:<28} capture={r['capture_rate']:.4f} {cells}")
        print(f"\nWrote {out_dir / 'finalists_results.json'}")
        return

    results = run_quant_sweep(
        args.toml,
        args.model,
        params_dir=args.params_dir,
        bits=tuple(args.bits),
        granularities=tuple(args.granularity),
        policies=tuple(args.policies),
        n_sims=args.n_sims,
        pool_offset=args.pool_offset,
        loo_bits=None if args.no_loo else args.loo_bits,
        sim_timeout_secs=args.sim_timeout,
        cost_transform=args.cost_transform,
    )
    (out_dir / "quantization_results.json").write_text(json.dumps(results, indent=2))

    from aerocapture.training.charts_quant import chart_quant_loo, chart_quant_sweep

    chart_quant_sweep(results, str(out_dir / "quantization_sweep.svg"))
    if results["loo"]:
        chart_quant_loo(results, str(out_dir / "quantization_loo.svg"))
    _print_table(results)
    print(f"\nWrote {out_dir / 'quantization_results.json'} and SVGs")


if __name__ == "__main__":
    main()
```

Note on imports: after this rewrite the module-top imports reduce to `argparse, copy, json, math, tempfile, Path, Any, np, npt` — everything heavier (aerocapture_rs, report, charts, ablation, evaluate, toml_utils, probe_common) is imported inside functions, preserving the soft-import property.

- [ ] **Step 4: Rewrite `charts_quant.py`**

```python
"""Quantization study charts: PTQ sweep curve, LOO bars, QAT convergence overlay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from aerocapture.training.charts import apply_theme  # noqa: E402


def chart_quant_sweep(results: dict[str, Any], output_path: str) -> None:
    """Two panels (capture rate, DV CVaR95) vs weight bits; one line per (granularity, policy)."""
    apply_theme()
    baseline = results["baseline"]
    fig, (ax_cap, ax_tail) = plt.subplots(1, 2, figsize=(14, 6))
    grans = sorted({v["granularity"] for v in results["variants"]})
    policies = sorted({v["tensor_policy"] for v in results["variants"]})
    for gran in grans:
        for policy in policies:
            rows = sorted((v for v in results["variants"] if v["granularity"] == gran and v["tensor_policy"] == policy), key=lambda r: r["bits"])
            if not rows:
                continue
            bits = [r["bits"] for r in rows]
            label = f"{gran} / {policy}"
            ax_cap.plot(bits, [r["capture_rate"] for r in rows], marker="o", label=label)
            ax_tail.plot(bits, [r["dv_cvar95"] if r["dv_cvar95"] is not None else float("nan") for r in rows], marker="o", label=label)
    ax_cap.axhline(baseline["capture_rate"], color="grey", ls="--", label="fp baseline")
    if baseline["dv_cvar95"] is not None:
        ax_tail.axhline(baseline["dv_cvar95"], color="grey", ls="--", label="fp baseline")
    for ax, ylab, title in ((ax_cap, "capture rate", "Capture rate vs bit width"), (ax_tail, "DV CVaR95 [m/s]", "Sizing tail vs bit width")):
        ax.set_xlabel("weight bits")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend()
        ax.invert_xaxis()  # degradation reads left-to-right
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def chart_quant_loo(results: dict[str, Any], output_path: str) -> None:
    """Horizontal bars: CVaR95 delta vs fp baseline when ONE tensor group is quantized."""
    apply_theme()
    rows = results["loo"]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 2))
    names = [r["tensor"] for r in rows]
    deltas = [r["delta_dv_cvar95"] if r["delta_dv_cvar95"] is not None else float("nan") for r in rows]
    colors = ["tab:red" if (d == d and d > 0) else "tab:blue" for d in deltas]
    ax.barh(names, deltas, color=colors)
    for i, r in enumerate(rows):
        if r["capture_rate"] < results["baseline"]["capture_rate"]:
            ax.annotate(f"capture {r['capture_rate']:.1%}", (0, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.set_xlabel(f"delta DV CVaR95 [m/s] at {rows[0]['bits']} bits (one tensor quantized, rest fp)")
    ax.set_title("Leave-one-out tensor sensitivity")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def chart_qat_convergence(jsonl_by_label: dict[str, list[Path]], output_path: str) -> None:
    """Best-cost convergence overlay: champion vs QAT fine-tune vs QAT from-scratch.

    Each label maps to that run's ordered `run_*.jsonl` files (concatenated)."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, paths in jsonl_by_label.items():
        gens: list[int] = []
        best: list[float] = []
        for p in paths:
            with open(p) as fh:
                for line in fh:
                    rec = json.loads(line)
                    gens.append(int(rec["generation"]))
                    best.append(float(rec["best_cost"]))
        ax.plot(gens, best, label=label, lw=1.0)
    ax.set_xlabel("generation")
    ax.set_ylabel("best training cost")
    ax.set_yscale("log")
    ax.set_title("QAT convergence vs fp champion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 5: Replace the old @slow sweep smoke** (in `tests/test_quantize.py`, delete the branch-era sweep smoke if ported, add)

```python
@pytest.mark.slow
def test_quant_sweep_smoke(tmp_path: Path) -> None:
    """Reduced end-to-end sweep on a synthetic champion-arch model: real sims, tiny grid."""
    pytest.importorskip("aerocapture_rs")
    import aerocapture_rs

    from aerocapture.training.quantize import main as quant_main

    arch = [
        {"type": "dense", "input_size": 17, "output_size": 16, "activation": "swish"},
        {"type": "mamba", "input_size": 16, "d_state": 12, "dt_rank": 1},
        {"type": "dense", "input_size": 16, "output_size": 2, "activation": "asinh"},
    ]
    rng = np.random.default_rng(0)
    model_path = tmp_path / "synthetic_962.json"
    aerocapture_rs.flat_weights_to_json(
        flat=rng.uniform(-0.5, 0.5, 962).tolist(),
        architecture_json=json.dumps(arch),
        path=str(model_path),
        input_mask=[0, 2, 3, 5, 6, 7, 11, 12, 18, 19, 27, 28, 29, 30, 32, 33, 34],
        output_param="atan2_signed",
    )
    out_dir = tmp_path / "sweep_out"
    quant_main(
        [
            str(out_dir),
            "--toml", "configs/training/sweep/mamba_p962.toml",
            "--model", str(model_path),
            "--n-sims", "3",
            "--bits", "8", "4",
            "--granularity", "per_tensor",
            "--policies", "all",
            "--loo-bits", "4",
            "--sim-timeout", "60",
        ]
    )
    results = json.loads((out_dir / "quantization_results.json").read_text())
    assert len(results["variants"]) == 2
    assert len(results["loo"]) == 6  # layer_0.w, x_proj_w, dt_proj_w, a_log, d_skip, layer_2.w
    assert results["verdict"]["bits"] == 4
    assert (out_dir / "quantization_sweep.svg").exists()
    assert (out_dir / "quantization_loo.svg").exists()
```

- [ ] **Step 6: Run everything**

Run: `uv run pytest tests/test_quantize.py -v` (from repo root so the sweep TOML resolves)
Expected: PASS, smoke takes ~1 min (17 x 3-sim batches).

- [ ] **Step 7: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/quantize.py src/python/aerocapture/training/charts_quant.py tests/test_quantize.py
git commit -m "feat(quantize): reserved-pool sweep with scaffolding + LOO + finalists + tail charts"
```

---

### Task 8: Rust criterion microbenchmark (f64 / f32 / w8a8 / w4a8)

Standalone bench for the champion arch: quantized kernels accelerate the matvec projections only; the Mamba SSM recurrence (softplus, exp, state update) stays f64 in every variant, mirroring what a fixed-point flight implementation would do. Sim source untouched.

**Files:**
- Modify: `src/rust/Cargo.toml` (criterion dev-dep + `[[bench]]`)
- Create: `src/rust/benches/quant_forward.rs`

**Interfaces:**
- Consumes: public crate API only — `aerocapture::data::neural::NeuralNetModel::from_json_str`, `aerocapture::data::nn_state::NnState::for_model`, `model.forward(&mut state, &input) -> Vec<f64>`.
- Produces: criterion benchmark ids `forward/f64_model`, `forward/f64_handrolled`, `forward/f32_handrolled`, `forward/w8a8`, `forward/w4a8`; results under `src/rust/target/criterion/forward/*/new/estimates.json` (Task 10 collects the medians).

- [ ] **Step 1: Add criterion + bench entry**

```bash
cargo add --dev criterion --manifest-path src/rust/Cargo.toml
```

Append to `src/rust/Cargo.toml`:

```toml
[[bench]]
name = "quant_forward"
harness = false
```

- [ ] **Step 2: Write `src/rust/benches/quant_forward.rs`**

```rust
//! Deployment-benefit microbenchmark for the Mamba-962 guidance head.
//!
//! Kernels: f64 (deployed `NeuralNetModel::forward`), hand-rolled f64/f32
//! references, weight-only-int8 + dynamic-int8-activation (w8a8), and packed
//! int4 weights + int8 activations (w4a8). Quantization accelerates the matvec
//! projections only; the SSM recurrence (softplus, exp, state update) stays in
//! floating point in every variant -- the honest dilution a fixed-point flight
//! implementation would see. Weights are deterministic pseudo-random: the cost
//! of a matvec does not depend on the values.
//!
//! Spec: docs/superpowers/specs/2026-07-10-quantization-study-appendix-design.md

use aerocapture::data::neural::NeuralNetModel;
use aerocapture::data::nn_state::NnState;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

// Champion shapes: Dense(17 -> 16, swish) -> Mamba(16, d_state 12, dt_rank 1) -> Dense(16 -> 2, asinh)
const N_IN: usize = 17;
const H: usize = 16;
const D_STATE: usize = 12;
const DT_RANK: usize = 1;
const XPROJ_ROWS: usize = DT_RANK + 2 * D_STATE; // 25
const N_OUT: usize = 2;

fn pseudo(seed: u64, n: usize) -> Vec<f64> {
    // Deterministic hash-noise in [-0.5, 0.5]; no rand dependency.
    (0..n)
        .map(|k| {
            let x = ((seed + k as u64 + 1) as f64 * 12.9898).sin() * 43758.5453;
            x - x.floor() - 0.5
        })
        .collect()
}

fn mat_json(v: &[f64], rows: usize, cols: usize) -> serde_json::Value {
    serde_json::Value::Array(
        (0..rows)
            .map(|r| serde_json::Value::Array((0..cols).map(|c| serde_json::json!(v[r * cols + c])).collect()))
            .collect(),
    )
}

struct Weights {
    d0_w: Vec<f64>, // (H, N_IN) row-major
    d0_b: Vec<f64>,
    x_proj: Vec<f64>, // (XPROJ_ROWS, H)
    dt_w: Vec<f64>,   // (H, DT_RANK)
    dt_b: Vec<f64>,
    a_log: Vec<f64>, // (H, D_STATE)
    d_skip: Vec<f64>,
    d2_w: Vec<f64>, // (N_OUT, H)
    d2_b: Vec<f64>,
}

fn make_weights() -> Weights {
    Weights {
        d0_w: pseudo(1, H * N_IN),
        d0_b: pseudo(2, H),
        x_proj: pseudo(3, XPROJ_ROWS * H),
        dt_w: pseudo(4, H * DT_RANK),
        dt_b: pseudo(5, H),
        a_log: pseudo(6, H * D_STATE).iter().map(|v| v + 1.0).collect(), // positive-ish, HiPPO-like
        d_skip: pseudo(7, H).iter().map(|v| v + 1.0).collect(),
        d2_w: pseudo(8, N_OUT * H),
        d2_b: pseudo(9, N_OUT),
    }
}

fn model_json(w: &Weights) -> String {
    serde_json::json!({
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": N_IN, "output_size": H, "activation": "swish"},
            {"type": "mamba", "input_size": H, "d_state": D_STATE, "dt_rank": DT_RANK},
            {"type": "dense", "input_size": H, "output_size": N_OUT, "activation": "asinh"},
        ],
        "weights": {
            "layer_0": {"w": mat_json(&w.d0_w, H, N_IN), "b": w.d0_b},
            "layer_1": {
                "x_proj_w": mat_json(&w.x_proj, XPROJ_ROWS, H),
                "dt_proj_w": mat_json(&w.dt_w, H, DT_RANK),
                "dt_proj_b": w.dt_b,
                "a_log": mat_json(&w.a_log, H, D_STATE),
                "d_skip": w.d_skip,
            },
            "layer_2": {"w": mat_json(&w.d2_w, N_OUT, H), "b": w.d2_b},
        },
    })
    .to_string()
}

fn swish(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn expm1_over_x(z: f64) -> f64 {
    if z.abs() < 1e-8 { 1.0 + z / 2.0 } else { z.exp_m1() / z }
}

/// Hand-rolled f64 full-tick forward -- validated against `NeuralNetModel::forward`.
struct HandF64 {
    w: Weights,
    h: Vec<f64>, // (H, D_STATE) row-major state
}

impl HandF64 {
    fn forward(&mut self, input: &[f64]) -> [f64; N_OUT] {
        let mut a1 = [0.0f64; H];
        for o in 0..H {
            let mut acc = self.w.d0_b[o];
            for i in 0..N_IN {
                acc += self.w.d0_w[o * N_IN + i] * input[i];
            }
            a1[o] = swish(acc);
        }
        let y1 = mamba_step_f64(&self.w, &a1, &mut self.h);
        let mut out = [0.0f64; N_OUT];
        for o in 0..N_OUT {
            let mut acc = self.w.d2_b[o];
            for i in 0..H {
                acc += self.w.d2_w[o * H + i] * y1[i];
            }
            out[o] = acc.asinh();
        }
        out
    }
}

/// f64 x_proj matvec + the shared fp SSM core (`finish_mamba`), which the
/// quantized variants reuse verbatim -- the fp-dilution the appendix reports.
fn mamba_step_f64(w: &Weights, x: &[f64; H], h: &mut [f64]) -> [f64; H] {
    let mut proj = [0.0f64; XPROJ_ROWS];
    for r in 0..XPROJ_ROWS {
        let mut acc = 0.0;
        for c in 0..H {
            acc += w.x_proj[r * H + c] * x[c];
        }
        proj[r] = acc;
    }
    finish_mamba(w, x, h, &proj)
}

/// Everything after the x_proj matvec: dt lift + softplus + ZOH recurrence + readout.
fn finish_mamba(w: &Weights, x: &[f64; H], h: &mut [f64], proj: &[f64; XPROJ_ROWS]) -> [f64; H] {
    let dt_pre = &proj[0..DT_RANK];
    let b_vec = &proj[DT_RANK..DT_RANK + D_STATE];
    let c_vec = &proj[DT_RANK + D_STATE..XPROJ_ROWS];
    let mut y = [0.0f64; H];
    for d in 0..H {
        let mut lift = w.dt_b[d];
        for r in 0..DT_RANK {
            lift += w.dt_w[d * DT_RANK + r] * dt_pre[r];
        }
        let delta = softplus(lift);
        let mut acc = 0.0;
        for n in 0..D_STATE {
            let a_dn = -w.a_log[d * D_STATE + n].exp();
            let za = delta * a_dn;
            let a_bar = za.exp();
            let b_bar = delta * b_vec[n] * expm1_over_x(za);
            let idx = d * D_STATE + n;
            h[idx] = a_bar * h[idx] + b_bar * x[d];
            acc += h[idx] * c_vec[n];
        }
        y[d] = acc + w.d_skip[d] * x[d];
    }
    y
}

/// f32 variant: same structure, all math in f32 (dtype-width reference row).
struct HandF32 {
    d0_w: Vec<f32>, d0_b: Vec<f32>, x_proj: Vec<f32>, dt_w: Vec<f32>, dt_b: Vec<f32>,
    a_log: Vec<f32>, d_skip: Vec<f32>, d2_w: Vec<f32>, d2_b: Vec<f32>, h: Vec<f32>,
}

impl HandF32 {
    fn new(w: &Weights) -> Self {
        let c = |v: &[f64]| v.iter().map(|x| *x as f32).collect::<Vec<f32>>();
        Self {
            d0_w: c(&w.d0_w), d0_b: c(&w.d0_b), x_proj: c(&w.x_proj), dt_w: c(&w.dt_w), dt_b: c(&w.dt_b),
            a_log: c(&w.a_log), d_skip: c(&w.d_skip), d2_w: c(&w.d2_w), d2_b: c(&w.d2_b), h: vec![0.0f32; H * D_STATE],
        }
    }

    fn forward(&mut self, input: &[f64]) -> [f64; N_OUT] {
        let xin: Vec<f32> = input.iter().map(|v| *v as f32).collect();
        let mut a1 = [0.0f32; H];
        for o in 0..H {
            let mut acc = self.d0_b[o];
            for i in 0..N_IN {
                acc += self.d0_w[o * N_IN + i] * xin[i];
            }
            a1[o] = acc / (1.0 + (-acc).exp());
        }
        let mut proj = [0.0f32; XPROJ_ROWS];
        for r in 0..XPROJ_ROWS {
            let mut acc = 0.0f32;
            for c in 0..H {
                acc += self.x_proj[r * H + c] * a1[c];
            }
            proj[r] = acc;
        }
        let dt_pre = &proj[0..DT_RANK];
        let b_vec = &proj[DT_RANK..DT_RANK + D_STATE];
        let c_vec = &proj[DT_RANK + D_STATE..XPROJ_ROWS];
        let mut y = [0.0f32; H];
        for d in 0..H {
            let mut lift = self.dt_b[d];
            for r in 0..DT_RANK {
                lift += self.dt_w[d * DT_RANK + r] * dt_pre[r];
            }
            let delta = lift.max(0.0) + (-lift.abs()).exp().ln_1p();
            let mut acc = 0.0f32;
            for n in 0..D_STATE {
                let za = delta * (-self.a_log[d * D_STATE + n].exp());
                let a_bar = za.exp();
                let bz = if za.abs() < 1e-4 { 1.0 + za / 2.0 } else { za.exp_m1() / za };
                let idx = d * D_STATE + n;
                self.h[idx] = a_bar * self.h[idx] + delta * b_vec[n] * bz * a1[d];
                acc += self.h[idx] * c_vec[n];
            }
            y[d] = acc + self.d_skip[d] * a1[d];
        }
        let mut out = [0.0f64; N_OUT];
        for o in 0..N_OUT {
            let mut acc = self.d2_b[o];
            for i in 0..H {
                acc += self.d2_w[o * H + i] * y[i];
            }
            out[o] = (acc as f64).asinh();
        }
        out
    }
}

/// Per-row symmetric int8 quantization of a (rows, cols) row-major matrix.
fn quant_i8(w: &[f64], rows: usize, cols: usize) -> (Vec<i8>, Vec<f64>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![1.0f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        scales[r] = s;
        for c in 0..cols {
            q[r * cols + c] = (row[c] / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

/// Per-row symmetric int4, packed two nibbles per byte (row-major, low nibble first).
fn quant_i4(w: &[f64], rows: usize, cols: usize) -> (Vec<u8>, Vec<f64>) {
    let packed_cols = cols.div_ceil(2);
    let mut q = vec![0u8; rows * packed_cols];
    let mut scales = vec![1.0f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let s = if amax == 0.0 { 1.0 } else { amax / 7.0 };
        scales[r] = s;
        for c in 0..cols {
            let v = ((row[c] / s).round().clamp(-7.0, 7.0) as i8) as u8 & 0x0F;
            let byte = &mut q[r * packed_cols + c / 2];
            if c % 2 == 0 { *byte |= v } else { *byte |= v << 4 }
        }
    }
    (q, scales)
}

/// Dynamic per-vector int8 activation quantization.
fn quant_act(x: &[f64], out: &mut [i8]) -> f64 {
    let amax = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    for (o, v) in out.iter_mut().zip(x.iter()) {
        *o = (v / s).round().clamp(-127.0, 127.0) as i8;
    }
    s
}

fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (wi, xi) in w.iter().zip(x.iter()) {
        acc += (*wi as i32) * (*xi as i32);
    }
    acc
}

fn dot_i4(packed: &[u8], x: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (k, xi) in x.iter().enumerate() {
        let byte = packed[k / 2];
        let nib = if k % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let w = ((nib << 4) as i8) >> 4; // sign-extend 4-bit
        acc += (w as i32) * (*xi as i32);
    }
    acc
}

/// w8a8 / w4a8: int projections (dense0, x_proj, dt lift folded into fp, dense2),
/// f64 SSM recurrence via `finish_mamba` -- the same fp core as the references.
struct QuantNet {
    w: Weights,
    d0_q: Vec<i8>, d0_s: Vec<f64>,
    xp_q: Vec<i8>, xp_s: Vec<f64>,
    d2_q: Vec<i8>, d2_s: Vec<f64>,
    d0_q4: Vec<u8>, d0_s4: Vec<f64>,
    xp_q4: Vec<u8>, xp_s4: Vec<f64>,
    d2_q4: Vec<u8>, d2_s4: Vec<f64>,
    h8: Vec<f64>,
    h4: Vec<f64>,
}

impl QuantNet {
    fn new(w: Weights) -> Self {
        let (d0_q, d0_s) = quant_i8(&w.d0_w, H, N_IN);
        let (xp_q, xp_s) = quant_i8(&w.x_proj, XPROJ_ROWS, H);
        let (d2_q, d2_s) = quant_i8(&w.d2_w, N_OUT, H);
        let (d0_q4, d0_s4) = quant_i4(&w.d0_w, H, N_IN);
        let (xp_q4, xp_s4) = quant_i4(&w.x_proj, XPROJ_ROWS, H);
        let (d2_q4, d2_s4) = quant_i4(&w.d2_w, N_OUT, H);
        Self {
            w, d0_q, d0_s, xp_q, xp_s, d2_q, d2_s, d0_q4, d0_s4, xp_q4, xp_s4, d2_q4, d2_s4,
            h8: vec![0.0; H * D_STATE], h4: vec![0.0; H * D_STATE],
        }
    }

    fn forward_w8a8(&mut self, input: &[f64]) -> [f64; N_OUT] {
        let mut xq = [0i8; N_IN];
        let sx = quant_act(input, &mut xq);
        let mut a1 = [0.0f64; H];
        for o in 0..H {
            let acc = dot_i8(&self.d0_q[o * N_IN..(o + 1) * N_IN], &xq) as f64;
            a1[o] = swish(acc * self.d0_s[o] * sx + self.w.d0_b[o]);
        }
        let mut a1q = [0i8; H];
        let s1 = quant_act(&a1, &mut a1q);
        let mut proj = [0.0f64; XPROJ_ROWS];
        for r in 0..XPROJ_ROWS {
            proj[r] = dot_i8(&self.xp_q[r * H..(r + 1) * H], &a1q) as f64 * self.xp_s[r] * s1;
        }
        let y1 = finish_mamba(&self.w, &a1, &mut self.h8, &proj);
        let mut y1q = [0i8; H];
        let s2 = quant_act(&y1, &mut y1q);
        let mut out = [0.0f64; N_OUT];
        for o in 0..N_OUT {
            let acc = dot_i8(&self.d2_q[o * H..(o + 1) * H], &y1q) as f64;
            out[o] = (acc * self.d2_s[o] * s2 + self.w.d2_b[o]).asinh();
        }
        out
    }

    fn forward_w4a8(&mut self, input: &[f64]) -> [f64; N_OUT] {
        let pc_in = N_IN.div_ceil(2);
        let pc_h = H.div_ceil(2);
        let mut xq = [0i8; N_IN];
        let sx = quant_act(input, &mut xq);
        let mut a1 = [0.0f64; H];
        for o in 0..H {
            let acc = dot_i4(&self.d0_q4[o * pc_in..(o + 1) * pc_in], &xq) as f64;
            a1[o] = swish(acc * self.d0_s4[o] * sx + self.w.d0_b[o]);
        }
        let mut a1q = [0i8; H];
        let s1 = quant_act(&a1, &mut a1q);
        let mut proj = [0.0f64; XPROJ_ROWS];
        for r in 0..XPROJ_ROWS {
            proj[r] = dot_i4(&self.xp_q4[r * pc_h..(r + 1) * pc_h], &a1q) as f64 * self.xp_s4[r] * s1;
        }
        let y1 = finish_mamba(&self.w, &a1, &mut self.h4, &proj);
        let mut y1q = [0i8; H];
        let s2 = quant_act(&y1, &mut y1q);
        let mut out = [0.0f64; N_OUT];
        for o in 0..N_OUT {
            let acc = dot_i4(&self.d2_q4[o * pc_h..(o + 1) * pc_h], &y1q) as f64;
            out[o] = (acc * self.d2_s4[o] * s2 + self.w.d2_b[o]).asinh();
        }
        out
    }
}

fn inputs(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|k| pseudo(100 + k as u64, N_IN)).collect()
}

fn self_check(model: &NeuralNetModel) {
    // (a) hand-rolled f64 must match the deployed forward to fp-noise level.
    let w = make_weights();
    let mut hand = HandF64 { w, h: vec![0.0; H * D_STATE] };
    let mut state = NnState::for_model(model);
    for x in inputs(200) {
        let a = model.forward(&mut state, &x);
        let b = hand.forward(&x);
        for o in 0..N_OUT {
            assert!((a[o] - b[o]).abs() < 1e-9, "handrolled f64 diverges: {} vs {}", a[o], b[o]);
        }
    }
    // (b) quantized kernels: finite and loosely sane vs f64 (activation quant adds real error).
    let mut hand2 = HandF64 { w: make_weights(), h: vec![0.0; H * D_STATE] };
    let mut qn = QuantNet::new(make_weights());
    for x in inputs(200) {
        let r = hand2.forward(&x);
        let q8 = qn.forward_w8a8(&x);
        let q4 = qn.forward_w4a8(&x);
        for o in 0..N_OUT {
            assert!(q8[o].is_finite() && q4[o].is_finite());
            assert!((q8[o] - r[o]).abs() < 0.5, "w8a8 wildly off: {} vs {}", q8[o], r[o]);
        }
    }
}

fn bench_forward(c: &mut Criterion) {
    let w = make_weights();
    let json = model_json(&w);
    let model = NeuralNetModel::from_json_str(&json, "bench_model.json").expect("model parse");
    self_check(&model);

    let xs = inputs(64);
    let mut g = c.benchmark_group("forward");

    let mut state = NnState::for_model(&model);
    let mut k = 0usize;
    g.bench_function("f64_model", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(model.forward(&mut state, black_box(&xs[k])))
        })
    });

    let mut hand = HandF64 { w: make_weights(), h: vec![0.0; H * D_STATE] };
    g.bench_function("f64_handrolled", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(hand.forward(black_box(&xs[k])))
        })
    });

    let mut hand32 = HandF32::new(&make_weights());
    g.bench_function("f32_handrolled", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(hand32.forward(black_box(&xs[k])))
        })
    });

    let mut qn = QuantNet::new(make_weights());
    g.bench_function("w8a8", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(qn.forward_w8a8(black_box(&xs[k])))
        })
    });
    g.bench_function("w4a8", |b| {
        b.iter(|| {
            k = (k + 1) % xs.len();
            black_box(qn.forward_w4a8(black_box(&xs[k])))
        })
    });
    g.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
```

NOTE for the implementer: the `d_skip_guard: ()` line inside `make_weights` is a deliberate compile error placed in this plan snippet? NO — it is a typo; the `Weights` struct has no such field. DELETE that line when writing the file (the struct literal is `d_skip: ..., d2_w: ..., d2_b: ...`).

- [ ] **Step 3: Compile-check the bench + self-checks**

```bash
cargo bench --bench quant_forward --manifest-path src/rust/Cargo.toml -- --test
```

Expected: compiles, self-checks pass (no panic), criterion runs each benchmark once in test mode. If `model.forward` visibility or module paths differ, fix the `use` lines against `src/rust/src/lib.rs` re-exports — do NOT change crate source.

- [ ] **Step 4: Rust hygiene**

```bash
cargo fmt --manifest-path src/rust/Cargo.toml
cargo clippy --benches --manifest-path src/rust/Cargo.toml -- -D warnings
cargo test --manifest-path src/rust/Cargo.toml --quiet
```

Expected: clean; the existing test suite (and therefore the goldens) untouched.

- [ ] **Step 5: Commit**

```bash
git add src/rust/Cargo.toml src/rust/Cargo.lock src/rust/benches/quant_forward.rs
git commit -m "bench(quant): f64/f32/w8a8/w4a8 forward kernels for the Mamba-962 head"
```

---

### Task 9: QAT leaf configs + campaign runner script

Two training configs inheriting the champion pipeline, and the phase-gated campaign script. The qat granularity/policy values in the configs are provisional (`per_channel`/`all`) and get overwritten with the PTQ verdict before launch (Task 10 gate).

**Files:**
- Create: `configs/training/quant/mamba962_qat4_finetune.toml`
- Create: `configs/training/quant/mamba962_qat4_scratch.toml`
- Create: `experiments/paper/15_quantization.sh` (chmod +x)

**Interfaces:**
- Consumes: qat knobs (Task 5), quantize CLI (Task 7), bench (Task 8).
- Produces: training dirs `training_output/quant/mamba962_qat4_finetune/` and `.../mamba962_qat4_scratch/`; results under `training_output/quant/ptq_sweep/` and `training_output/quant/finalists/`; collected bundle `articles/paper/data/quant/`.

- [ ] **Step 1: Write `configs/training/quant/mamba962_qat4_finetune.toml`**

```toml
# QAT-4bit FINE-TUNE arm: resume the fp champion (mamba_p962_long, GA 512 x 20k)
# with 4-bit fake-quant fitness for +3000 gens. Answers RECOVERABILITY: how much
# of the PTQ-4bit loss does adapting the weights to the grid win back?
# Launch (after copying the champion checkpoint pair -- see 15_quantization.sh):
#   uv run python -m aerocapture.training.train configs/training/quant/mamba962_qat4_finetune.toml \
#       --n-gen 3000 --output-dir training_output/quant/mamba962_qat4_finetune
# qat_granularity / qat_tensor_policy below are PROVISIONAL: overwrite with the
# PTQ verdict (training_output/quant/ptq_sweep/quantization_results.json .verdict)
# before launching. The champion dir is never written to.
base = ["../sweep/mamba_p962.toml"]

[data]
neural_network = "training_output/quant/mamba962_qat4_finetune/best_model.json"
results_suffix = ".quant_qat4_finetune"

[network]
qat_bits = 4
qat_granularity = "per_channel"
qat_tensor_policy = "all"

[optimizer]
algorithm = "ga"
n_pop = 512
n_gen = 20000
training_n_sims = 2
seed_strategy = "adaptive"
curation_bucket_selection = "max"
```

- [ ] **Step 2: Write `configs/training/quant/mamba962_qat4_scratch.toml`**

```toml
# QAT-4bit FROM-SCRATCH arm: fresh GA 512 x 20000 (budget-matched to the fp
# champion mamba_p962_long) with 4-bit fake-quant fitness from gen 0. Answers
# TRAINABILITY: does training under the 4-bit constraint reach the fp champion?
# Launch:
#   uv run python -m aerocapture.training.train configs/training/quant/mamba962_qat4_scratch.toml \
#       --output-dir training_output/quant/mamba962_qat4_scratch --from-scratch
# qat_granularity / qat_tensor_policy below are PROVISIONAL: overwrite with the
# PTQ verdict before launching.
base = ["../sweep/mamba_p962.toml"]

[data]
neural_network = "training_output/quant/mamba962_qat4_scratch/best_model.json"
results_suffix = ".quant_qat4_scratch"

[network]
qat_bits = 4
qat_granularity = "per_channel"
qat_tensor_policy = "all"

[optimizer]
algorithm = "ga"
n_pop = 512
n_gen = 20000
training_n_sims = 2
seed_strategy = "adaptive"
curation_bucket_selection = "max"
```

- [ ] **Step 3: Config-load sanity check** (both leaves must resolve and validate)

```bash
uv run python -c "
from pathlib import Path
from aerocapture.training.toml_utils import load_toml_with_bases
for p in ['configs/training/quant/mamba962_qat4_finetune.toml', 'configs/training/quant/mamba962_qat4_scratch.toml']:
    d = load_toml_with_bases(Path(p))
    net = d['network']
    assert net['qat_bits'] == 4 and len(net['architecture']) == 3, p
    assert d['optimizer']['n_pop'] == 512 and d['optimizer']['n_gen'] == 20000, p
    print(p, 'OK')
"
```

Expected: two `OK` lines.

- [ ] **Step 4: Write `experiments/paper/15_quantization.sh`**

```bash
#!/usr/bin/env bash
# Quantization study campaign (paper Appendix D). Phase-gated: run `ptq` first,
# inspect the verdict, copy it into the two QAT configs, then run the rest.
#   ./experiments/paper/15_quantization.sh ptq        # PTQ sweep + LOO on the champion (~minutes)
#   ./experiments/paper/15_quantization.sh bench      # criterion microbench (run BEFORE trainings for clean numbers)
#   ./experiments/paper/15_quantization.sh qat_finetune   # +3000 gens from the champion checkpoint (~0.5 day)
#   ./experiments/paper/15_quantization.sh qat_scratch    # GA 512 x 20000 from scratch (~2.5-3 days)
#   ./experiments/paper/15_quantization.sh finalists  # n=10000 re-score of the four finalist rows
#   ./experiments/paper/15_quantization.sh collect    # bundle JSONs into articles/paper/data/quant/
set -euo pipefail
cd "$(dirname "$0")/../.."

CHAMPION_DIR=training_output/mamba_p962_long
SWEEP_TOML=configs/training/sweep/mamba_p962.toml
QUANT_DIR=training_output/quant

case "${1:-}" in
ptq)
    uv run python -m aerocapture.training.quantize "$QUANT_DIR/ptq_sweep" \
        --toml "$SWEEP_TOML" \
        --model "$CHAMPION_DIR/best_model.json" \
        --params-dir "$CHAMPION_DIR" \
        --n-sims 1000 --loo-bits 4 --sim-timeout 120
    echo
    echo "GATE: read the verdict above; copy granularity/tensor_policy into"
    echo "configs/training/quant/mamba962_qat4_{finetune,scratch}.toml before launching QAT."
    ;;
bench)
    cargo bench --bench quant_forward --manifest-path src/rust/Cargo.toml
    ;;
qat_finetune)
    mkdir -p "$QUANT_DIR/mamba962_qat4_finetune"
    cp -n "$CHAMPION_DIR/checkpoint_g20000.json" "$QUANT_DIR/mamba962_qat4_finetune/"
    cp -n "$CHAMPION_DIR/checkpoint_g20000.npz" "$QUANT_DIR/mamba962_qat4_finetune/"
    uv run python -m aerocapture.training.train configs/training/quant/mamba962_qat4_finetune.toml \
        --n-gen 3000 --output-dir "$QUANT_DIR/mamba962_qat4_finetune" --no-tui
    ;;
qat_scratch)
    uv run python -m aerocapture.training.train configs/training/quant/mamba962_qat4_scratch.toml \
        --output-dir "$QUANT_DIR/mamba962_qat4_scratch" --from-scratch --no-tui
    ;;
finalists)
    # QAT arms pass quantize=null (their deployed best_model.json is already on-grid);
    # the PTQ finalist quantizes the champion at the verdict cell on the fly.
    uv run python - <<'PY'
import json
from pathlib import Path

verdict = json.loads(Path("training_output/quant/ptq_sweep/quantization_results.json").read_text())["verdict"]
entries = [
    {"label": "champion_fp", "model": "training_output/mamba_p962_long/best_model.json", "params_dir": "training_output/mamba_p962_long", "quantize": None},
    {"label": "ptq4_verdict", "model": "training_output/mamba_p962_long/best_model.json", "params_dir": "training_output/mamba_p962_long",
     "quantize": {"bits": 4, "granularity": verdict["granularity"], "tensor_policy": verdict["tensor_policy"]}},
    {"label": "qat4_finetune", "model": "training_output/quant/mamba962_qat4_finetune/best_model.json", "params_dir": "training_output/quant/mamba962_qat4_finetune", "quantize": None},
    {"label": "qat4_scratch", "model": "training_output/quant/mamba962_qat4_scratch/best_model.json", "params_dir": "training_output/quant/mamba962_qat4_scratch", "quantize": None},
]
Path("training_output/quant/finalists_entries.json").write_text(json.dumps(entries, indent=2))
PY
    uv run python -m aerocapture.training.quantize "$QUANT_DIR/finalists" \
        --toml "$SWEEP_TOML" \
        --model "$CHAMPION_DIR/best_model.json" \
        --n-sims 10000 --sim-timeout 120 \
        --finalists "$QUANT_DIR/finalists_entries.json"
    ;;
collect)
    mkdir -p articles/paper/data/quant
    cp "$QUANT_DIR/ptq_sweep/quantization_results.json" articles/paper/data/quant/
    cp "$QUANT_DIR/finalists/finalists_results.json" articles/paper/data/quant/
    cp "$QUANT_DIR/ptq_sweep/quantization_sweep.svg" "$QUANT_DIR/ptq_sweep/quantization_loo.svg" articles/paper/figures/ 2>/dev/null || true
    # criterion medians -> one compact JSON
    uv run python - <<'PY'
import json
from pathlib import Path

rows = {}
for d in Path("src/rust/target/criterion/forward").iterdir():
    est = d / "new" / "estimates.json"
    if est.exists():
        e = json.loads(est.read_text())
        rows[d.name] = {"median_ns": e["median"]["point_estimate"], "ci95": [e["median"]["confidence_interval"]["lower_bound"], e["median"]["confidence_interval"]["upper_bound"]]}
Path("articles/paper/data/quant/bench_forward.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
    ;;
*)
    echo "usage: $0 {ptq|bench|qat_finetune|qat_scratch|finalists|collect}" >&2
    exit 1
    ;;
esac
```

```bash
chmod +x experiments/paper/15_quantization.sh
```

- [ ] **Step 5: Dry checks + commit**

Run: `bash -n experiments/paper/15_quantization.sh && ./experiments/paper/15_quantization.sh 2>&1 | head -2`
Expected: syntax OK; usage line when called without args (exit 1 is fine).

```bash
git add configs/training/quant/ experiments/paper/15_quantization.sh
git commit -m "feat(quant): QAT-4bit finetune/scratch configs + campaign runner script"
```

---

### Task 10: Run the campaign

The compute phase. PTQ + bench are quick; the QAT arms are long-running (fine-tune ~0.5 day, scratch ~2.5-3 days on the dev machine — run them sequentially, never concurrently, to keep per-run wall time predictable). Everything below runs from repo root on the MAIN checkout (`/Users/govit/Git/Govit/Aerocapture`), where `training_output/mamba_p962_long/` lives — not from a worktree without it.

**Files:**
- Produces (untracked): `training_output/quant/ptq_sweep/quantization_results.json` + SVGs, `training_output/quant/mamba962_qat4_{finetune,scratch}/` training dirs, `training_output/quant/finalists/finalists_results.json`
- Modify: `configs/training/quant/mamba962_qat4_finetune.toml`, `configs/training/quant/mamba962_qat4_scratch.toml` (verdict values)
- Create: `articles/paper/data/quant/{quantization_results.json,finalists_results.json,bench_forward.json}`

**Interfaces:**
- Consumes: everything from Tasks 1-9.
- Produces: the numbers Task 11 writes into the appendix.

- [ ] **Step 1: PTQ sweep + LOO**

Run: `./experiments/paper/15_quantization.sh ptq`
Expected: table of 20 grid cells + 6 LOO rows + a verdict line, ~10-20 min (27 x 1000 sims at ~4 ms/sim on all cores). SANITY GATE: the baseline row must reproduce the champion quote within noise — capture 1.000, dv_p50 109.7 +/- 0.5, cvar95 115.2 +/- 1.0 (same pool, same seeds, same scaffolding). If it does not, STOP: the pool resolution or scaffolding overrides are wrong — debug against `training_output/mamba_p962_long/fresh_pool_requote.json` before burning training days.

- [ ] **Step 2: Apply the verdict to both QAT configs**

Read `verdict` from `training_output/quant/ptq_sweep/quantization_results.json`; edit `qat_granularity` and `qat_tensor_policy` in BOTH `configs/training/quant/mamba962_qat4_*.toml` to match. Commit:

```bash
git add configs/training/quant/
git commit -m "chore(quant): pin QAT cell from PTQ verdict"
```

- [ ] **Step 3: Microbenchmark (before trainings, for clean numbers)**

Run: `./experiments/paper/15_quantization.sh bench`
Expected: five `forward/*` benchmarks with ns-scale medians. Record nothing by hand — the `collect` phase reads `target/criterion`.

- [ ] **Step 4: Ticks-per-sim bridge number**

```bash
uv run python - <<'PY'
import json
import numpy as np
import aerocapture_rs
from aerocapture.training.report import _load_nn_scaffolding_overrides
from pathlib import Path

d = Path("training_output/mamba_p962_long")
scaff = _load_nn_scaffolding_overrides(d, d / "optimized_mamba_p962_long.toml")
res = aerocapture_rs.run_mc(
    "configs/training/sweep/mamba_p962.toml",
    overrides={"simulation.n_sims": 1, "monte_carlo.seed": 42, "data.neural_network": str(d / "best_model.json"), **scaff},
    include_trajectories=True,
)
traj = np.asarray(res.trajectories[0])
t_final = float(traj[-1, 7])  # trajectory column 7 = time_s
print(json.dumps({"t_final_s": t_final, "guidance_period_s": 1.0, "guidance_ticks": round(t_final / 1.0)}))
PY
```

Expected: a JSON line with `guidance_ticks` in the several-hundreds. Save it to `articles/paper/data/quant/ticks_per_sim.json` (redirect or copy-paste).

- [ ] **Step 5: QAT fine-tune arm (~0.5 day)**

Run: `./experiments/paper/15_quantization.sh qat_finetune`
Expected: resume banner from `checkpoint_g20000`, re-validation of the checkpointed best under the quantized objective (its val RMS will jump — that IS the PTQ shock being measured by the re-validation), then +3000 gens with heartbeat prints. Ends with `best_model.json`/`best_params.json` in the finetune dir.

- [ ] **Step 6: QAT from-scratch arm (~2.5-3 days, launch after Step 5 completes)**

Run: `./experiments/paper/15_quantization.sh qat_scratch`
Expected: fresh 512 x 20000 run. Monitor via the heartbeat lines; the run auto-checkpoints and auto-resumes if interrupted (re-run the same command).

- [ ] **Step 7: Finalists at n=10000**

Run: `./experiments/paper/15_quantization.sh finalists`
Expected: four rows (champion_fp, ptq4_verdict, qat4_finetune, qat4_scratch), each 10000 sims (~1-2 min each). This is the appendix headline table.

- [ ] **Step 8: QAT deployed-weights on-grid audit** (defense-in-depth before quoting numbers)

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from aerocapture.training.quantize import quantize_model_weights
import numpy as np

verdict = json.loads(Path("training_output/quant/ptq_sweep/quantization_results.json").read_text())["verdict"]
for arm in ("mamba962_qat4_finetune", "mamba962_qat4_scratch"):
    m = json.loads(Path(f"training_output/quant/{arm}/best_model.json").read_text())
    q = quantize_model_weights(m, 4, verdict["granularity"], verdict["tensor_policy"])
    for i in range(len(m["architecture"])):
        for f in m["weights"][f"layer_{i}"]:
            np.testing.assert_allclose(m["weights"][f"layer_{i}"][f], q["weights"][f"layer_{i}"][f], rtol=0, atol=1e-12)
    print(arm, "deployed weights are on the 4-bit grid: OK")
PY
```

Expected: two OK lines.

- [ ] **Step 9: Collect the data bundle + QAT convergence chart**

Run: `./experiments/paper/15_quantization.sh collect`

Then render the convergence overlay:

```bash
uv run python - <<'PY'
from pathlib import Path
from aerocapture.training.charts_quant import chart_qat_convergence

runs = {
    "fp champion (512 x 20k)": sorted(Path("training_output/mamba_p962_long").glob("run_*.jsonl")),
    "QAT-4bit fine-tune (+3k)": sorted(Path("training_output/quant/mamba962_qat4_finetune").glob("run_*.jsonl")),
    "QAT-4bit scratch (512 x 20k)": sorted(Path("training_output/quant/mamba962_qat4_scratch").glob("run_*.jsonl")),
}
chart_qat_convergence(runs, "articles/paper/figures/quant_qat_convergence.svg")
print("wrote articles/paper/figures/quant_qat_convergence.svg")
PY
```

- [ ] **Step 10: Commit the data bundle**

```bash
git add articles/paper/data/quant/ articles/paper/figures/
git commit -m "data(quant): PTQ sweep + LOO + finalists + bench numbers for Appendix D"
```

---

### Task 11: Appendix D text + paper integration

Working notes mirror `docs/paper/architecture_probes_appendix.md` (provenance block, one-line result, tables, honest-caveats section); the citable text goes into `articles/paper/paper.typ` as Appendix D; the future-work sentence at paper.typ:949 is updated to point at it. All numbers come from the Task 10 JSONs — no hand-typed values without a JSON source.

**Files:**
- Create: `docs/paper/quantization_appendix.md`
- Modify: `articles/paper/paper.typ` (new `= Appendix D: ...` section after Appendix C at ~line 1066+; sentence rewrite at ~line 949)

**Interfaces:**
- Consumes: `articles/paper/data/quant/{quantization_results.json,finalists_results.json,bench_forward.json,ticks_per_sim.json}`, `training_output/mamba_p962_long/fresh_pool_requote.json`, the probe-campaign sigma_run numbers already quoted in Appendix C.

- [ ] **Step 1: Generate the appendix tables from the JSONs** (paste-ready markdown; keeps hand-typing out)

```bash
uv run python - <<'PY'
import json
from pathlib import Path

q = json.loads(Path("articles/paper/data/quant/quantization_results.json").read_text())
f = json.loads(Path("articles/paper/data/quant/finalists_results.json").read_text())
bench = json.loads(Path("articles/paper/data/quant/bench_forward.json").read_text())

print("## Finalists (n = 10000, fresh pool offset 8M, scaffolding applied)\n")
print("| variant | capture | DV p50 | p95 | p99 | CVaR95 | viol % |")
print("|---|---|---|---|---|---|---|")
for r in f["finalists"]:
    print(f"| {r['label']} | {r['capture_rate']:.4f} | {r['dv_p50']:.1f} | {r['dv_p95']:.1f} | {r['dv_p99']:.1f} | {r['dv_cvar95']:.1f} | {r['viol_pct']:.2f} |")

print("\n## PTQ grid (n = 1000)\n")
print("| bits | granularity | policy | capture | CVaR95 | dCVaR95 |")
print("|---|---|---|---|---|---|")
for v in sorted(q["variants"], key=lambda v: (-v["bits"], v["granularity"], v["tensor_policy"])):
    d = "-" if v["delta_dv_cvar95"] is None else f"{v['delta_dv_cvar95']:+.1f}"
    c = "-" if v["dv_cvar95"] is None else f"{v['dv_cvar95']:.1f}"
    print(f"| {v['bits']} | {v['granularity']} | {v['tensor_policy']} | {v['capture_rate']:.3f} | {c} | {d} |")

print("\n## Leave-one-out at 4 bits\n")
for r in q["loo"]:
    d = "-" if r["delta_dv_cvar95"] is None else f"{r['delta_dv_cvar95']:+.2f}"
    print(f"- {r['tensor']}: capture {r['capture_rate']:.3f}, dCVaR95 {d}")

print("\n## Memory (bytes, f32 scales + f32 fp remainder)\n")
for m in q["memory"]:
    if m["granularity"] == q["verdict"]["granularity"] and m["tensor_policy"] == q["verdict"]["tensor_policy"]:
        print(f"- {m['bits']}b: {m['total_bytes']} B (quant {m['quant_bytes']} + scales {m['scale_bytes']} + fp {m['fp_bytes']}); f64 baseline {m['f64_baseline_bytes']} B")

print("\n## Bench (ns/tick, criterion medians)\n")
for k, v in sorted(bench.items()):
    print(f"- {k}: {v['median_ns']:.0f} ns")
PY
```

- [ ] **Step 2: Write `docs/paper/quantization_appendix.md`** with this skeleton, filling every `<...>` slot from Step 1's output and the JSONs (structure mirrors `architecture_probes_appendix.md`):

```markdown
# Quantization of the deployed Mamba-962 guidance head

Status: working notes for paper Appendix D; the paper appendix is the citable text.

Provenance:
- Branch `feature/quantization-mamba962`. Spec `docs/superpowers/specs/2026-07-10-quantization-study-appendix-design.md`.
- Target: `training_output/mamba_p962_long` (Dense(17->16, swish) -> Mamba(16, d_state 12) -> Dense(16->2, asinh), 962 NN params, GA 512 x 20k, scaffolding = live).
- PTQ: weight-only symmetric fake-quant, bits {8,6,4,3,2} x {per_channel, per_tensor} x {all, proj_only}, n=1000/cell; LOO at 4 bits over 6 tensor groups. Pool: HEADLINE_REQUOTE_SEED_OFFSET = 8_000_000, base seed 42, champion scaffolding applied.
- QAT: GA-in-the-loop fake-quant (population rounded before every fitness eval, deploy writer rounds `best_model.json`). Fine-tune: champion checkpoint + 3000 gens. Scratch: 512 x 20000 matched budget. Verdict cell: <bits/gran/policy>.
- Bench: `src/rust/benches/quant_forward.rs`, criterion medians on <machine>, SSM recurrence fp in all variants.

## One-line result

<fill after finalists: e.g. "Weight-only PTQ at 8/6 bits is free; 4-bit costs
<X> m/s of CVaR95, of which QAT fine-tuning recovers <Y>; the LOO fingers
<tensor> as the bottleneck; the whole head fits in <B> bytes at 4 bits.">

## Sensitivity: which tensors tolerate 4 bits

<LOO list + reading; a_log hypothesis confirmed/refuted>

## PTQ grid

<table>

## QAT arms (single runs -- deltas quoted against the probe-campaign sigma_run,
same honesty rule as Appendix C)

<finalists table + convergence-chart pointer + recoverability/trainability reading>

## Deployment benefit (the honest version)

- Memory (analytic, exact): <f64 -> int8 -> int4 rows>.
- Compute (measured, x86; SSM recurrence stays fp): <bench rows>; x <ticks>
  guidance ticks/sim -> <ms> per sim vs the paper's 3.68 ms/sim figure.
- Simulation throughput: UNCHANGED by construction -- fake-quant stores rounded
  weights back as f64 and the runtime executes the same f64 kernels. Any
  sim-time claim would be an artifact; the benefit is flight-software footprint
  and fixed-point-capable inference, not sims/s.

## Caveats

- Weight-only: activations, state, and normalization stay f64 in the accuracy study.
- Both QAT arms are single runs; sigma_run from the architecture-probe repeats
  bounds what a delta must clear.
- w8a8/w4a8 bench kernels quantize activations dynamically (compute measurement
  only); the accuracy sweep does not model activation quantization.
```

- [ ] **Step 3: Integrate into `articles/paper/paper.typ`**

(a) After the Appendix C section (starts `= Appendix C: architecture probes ...` at ~line 1066), append `= Appendix D: quantization of the deployed Mamba head` with the citable version of the working notes: one intro paragraph (motivation + method, weight-only fake-quant, pool provenance), the finalists table, the LOO reading, the memory/compute paragraph with the sim-throughput-unchanged statement, and the sigma_run caveat. Follow Appendix C's Typst idioms (same table/figure macros already used in the file).

(b) Rewrite the future-work sentence at ~line 949. Current text:

```
We have no clean campaign study of pruning or
quantizing the deployed head -- the only such cells predate the simulator fixes in this work and are
not comparable -- so deploy-size reduction of the Mamba policy is open.
```

Replace with (adjusting the second clause to the actual result):

```
Appendix D closes the quantization half of the deploy-size question with a
post-training sensitivity sweep and two quantization-aware retraining arms on
the deployed head; pruning remains open.
```

- [ ] **Step 4: Compile the paper**

Run: `typst compile articles/paper/paper.typ articles/paper/paper.pdf` (from repo root; if the repo has a build script under `articles/paper/scripts/`, use it instead)
Expected: clean compile; Appendix D renders with tables and the two figures.

- [ ] **Step 5: Commit**

```bash
git add docs/paper/quantization_appendix.md articles/paper/paper.typ articles/paper/paper.pdf
git commit -m "docs(paper): Appendix D -- quantization study of the Mamba-962 head"
```

---

### Task 12: Final verification + smart-commit

- [ ] **Step 1: Full test + lint gates**

```bash
uv run pytest tests -q -m "not slow"
uv run pytest tests/test_quantize.py tests/test_qat_training.py -v   # incl. slow, needs PyO3
./lint_code.sh
./check_all.sh
```

Expected: all green; `check_all.sh` proves the Rust side (fmt, clippy, tests, release build) is untouched by the bench addition.

- [ ] **Step 2: Update CLAUDE.md** — add a short block documenting: `quantize.py` (PTQ sweep + LOO + finalists CLI, reserved-pool + scaffolding methodology), the three `[network]` qat knobs and their two hook points, the `configs/training/quant/` arms, `experiments/paper/15_quantization.sh`, and the bench. Match the existing tool-entry style (one dense paragraph per module).

- [ ] **Step 3: Invoke the smart-commit skill**, telling it to take the whole git branch into account (per user planning rule). It reconciles CLAUDE.md/README with the branch diff and produces the final commit(s).

---

## Execution ordering note

Tasks 1-9 are code and are executable back-to-back (~1.5-2 days of work, each independently committable). Task 10 has a hard human gate after Step 1 (verdict inspection) and multi-day training waits between Steps 5-7; run it from the main checkout where the champion artifacts live. Task 11 depends on Task 10's JSONs. Task 12 closes the branch.
