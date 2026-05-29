# NN Input Behavior Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone diagnostic that runs the deployed NN over an MC ensemble, captures the per-tick 31-element candidate input vector + final DV, and renders per-input spaghetti+envelope panels (vs time and energy) colored blue/red by a DV threshold, plus a per-input saturation/separation summary table.

**Architecture:** A new `collect_nn_inputs` PyO3 function runs the configured NN scheme with the per-tick candidate-vector trace enabled (reusing the `collect_supervised` machinery, but with NO teacher override) and returns per-trajectory `{X (T,31), time, energy, dv, captured}`. A standalone Python module (`nn_input_report`, mirroring `ablation.py`) classifies trajectories by DV, computes per-input stats, and renders charts via a new `charts_nn_inputs.py`.

**Tech Stack:** Rust (PyO3, nalgebra), Python 3.14 (numpy, matplotlib/seaborn), pytest, Typst (optional).

**Spec:** `docs/superpowers/specs/2026-05-29-nn-input-report-design.md`

**Branch:** `feature/nn-input-report` (already created off `feature/nn-bank-decoders`; spec already committed).

---

## File map

- `src/rust/src/simulation/runner.rs` — widen `supervised_trace` tuple to carry `sim_time` + `energy` (3 type sites incl. `lib.rs` `RunOutput`).
- `src/rust/src/simulation/tick.rs` — populate the two new trace fields at the supervised push.
- `src/rust/src/lib.rs` — `RunOutput.supervised_trace` type widen.
- `src/rust/aerocapture-py/src/lib.rs` — new `collect_nn_inputs` pyfunction + module registration; `collect_supervised` destructure updated to ignore the 2 new fields (output unchanged).
- `src/python/aerocapture/training/nn_input_report.py` — new: collect orchestration, DV classification, per-input summary, CLI.
- `src/python/aerocapture/training/charts_nn_inputs.py` — new: binned per-class envelope helper + panel rendering.
- `tests/test_nn_input_report.py` — new: Python unit + CLI tests.
- `tests/test_collect_nn_inputs.py` — new: PyO3 behavior tests.

---

## Task 1: Rust capture — widen trace + `collect_nn_inputs`

Trace widen and the new collect fn are coupled (the widen exists to serve `collect_nn_inputs`), so they land together — no broken intermediate.

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (`supervised_trace` type, 2 sites ~155, ~448), `src/rust/src/lib.rs` (`RunOutput.supervised_trace` ~26), `src/rust/src/simulation/tick.rs` (push ~196), `src/rust/aerocapture-py/src/lib.rs` (`collect_supervised` destructure ~515, new `collect_nn_inputs`, registration ~627)
- Test: `tests/test_collect_nn_inputs.py`

- [ ] **Step 1: Write the failing Python test**

```python
# tests/test_collect_nn_inputs.py
import json

import aerocapture_rs
import numpy as np
from aerocapture.training.toml_utils import load_toml_with_bases


def _mint_zero_model(tmp_path):
    """Mint a loadable zero-weight NN matching the delta config's arch."""
    cfg = load_toml_with_bases("configs/training/msr_aller_nn_delta_train.toml")
    arch = cfg["network"]["architecture"]
    mask = cfg["network"]["input_mask"]

    def n_params(layer):  # dense only in this arch
        return layer["input_size"] * layer["output_size"] + layer["output_size"]

    flat = [0.0] * sum(n_params(l) for l in arch)
    path = str(tmp_path / "zero_model.json")
    aerocapture_rs.flat_weights_to_json(
        flat, json.dumps(arch), path, mask,
        cfg["guidance"]["neural_network"]["output_parameterization"],
        None, cfg["guidance"]["neural_network"]["delta_max"],
    )
    return path


def test_collect_nn_inputs_runs_nn_and_returns_shapes(tmp_path):
    model = _mint_zero_model(tmp_path)
    out = aerocapture_rs.collect_nn_inputs(
        "configs/training/msr_aller_nn_delta_train.toml",
        [4_000_000],
        overrides={"data.neural_network": model},
    )
    assert len(out) == 1
    r = out[0]
    assert set(r.keys()) == {"seed", "X", "time", "energy", "dv", "captured"}
    X, t, e = r["X"], r["time"], r["energy"]
    assert X.ndim == 2 and X.shape[1] == 31
    assert t.shape == (X.shape[0],) and e.shape == (X.shape[0],)
    assert np.all(np.diff(t) >= 0)  # time monotonic non-decreasing
    assert np.isfinite(X).all() and np.isfinite(t).all() and np.isfinite(e).all()
    assert isinstance(r["captured"], bool)


def test_collect_nn_inputs_rejects_non_nn_config():
    # An FTC config has guidance.type != neural_network -> must error.
    try:
        aerocapture_rs.collect_nn_inputs("configs/training/msr_aller_ftc_train.toml", [1])
    except (ValueError, RuntimeError) as ex:
        assert "neural_network" in str(ex).lower()
    else:
        raise AssertionError("expected collect_nn_inputs to reject a non-NN config")
```

- [ ] **Step 2: Run to verify it fails**

Run (from repo root):
```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml 2>&1 | tail -3
uv run pytest tests/test_collect_nn_inputs.py -x 2>&1 | tail -20
```
Expected: FAIL — `aerocapture_rs has no attribute 'collect_nn_inputs'`.

- [ ] **Step 3: Widen the supervised trace tuple**

In `src/rust/src/simulation/runner.rs` (both declarations) and `src/rust/src/lib.rs` (`RunOutput`):
```rust
    // was: Vec<(Vec<f64>, f64, f64)>   (nn_input, y_signed, prev_realized)
    // now: + sim_time + energy_mj_kg
    pub(crate) supervised_trace: Vec<(Vec<f64>, f64, f64, f64, f64)>,
```
(Apply the same 5-tuple type to all three declaration sites; the `lib.rs` one is `pub`.)

- [ ] **Step 4: Populate time + energy at the tick.rs push**

In `src/rust/src/simulation/tick.rs`, the push currently is a 3-tuple. Compute the estimated-state energy (MJ/kg, the convention the NN's energy-keyed inputs use) and push 5 fields:
```rust
            let energy_mj_kg = crate::gnc::navigation::coordinates::total_energy(
                nav_out.position_estimated[0],
                nav_out.position_estimated[1],
                nav_out.position_estimated[2],
                nav_out.velocity_estimated[0],
                nav_out.velocity_estimated[1],
                nav_out.velocity_estimated[2],
                planet,
            ) / 1e6;
            state.supervised_trace.push((
                nn_input,
                guidance_out.pre_shaper_signed,
                state.guidance_state.prev_realized_bank_for_nn,
                state.sim_time,
                energy_mj_kg,
            ));
```
(`total_energy` is already used in `gnc/guidance/neural.rs`; same signature.)

- [ ] **Step 5: Update `collect_supervised` to ignore the 2 new fields**

In `src/rust/aerocapture-py/src/lib.rs`, the extraction loop destructures the trace. Update it so `collect_supervised`'s dict output is unchanged:
```rust
        for (nn_input, bank, realized, _t, _e) in supervised_trace {
            x_rows.push(nn_input);
            y_signed.push(bank);
            prev_realized.push(realized);
        }
```
Also update the `combined_trace`/`per_seed` tuple type in `collect_supervised` to the 5-tuple element type (`Vec<(Vec<f64>, f64, f64, f64, f64)>`) so `output.supervised_trace` extends cleanly.

- [ ] **Step 6: Add `collect_nn_inputs`**

In `src/rust/aerocapture-py/src/lib.rs`, add (mirrors `collect_supervised` but runs the configured NN, captures time/energy, no teacher override):
```rust
/// Collect the deployed NN's own per-tick candidate input vectors over an MC ensemble.
///
/// Runs the CONFIGURED guidance (must be neural_network) with the per-tick
/// candidate-vector trace enabled. Unlike collect_supervised it does NOT override
/// the guidance type -- it captures the inputs the NN actually drives itself into.
///
/// Returns a list of dicts (one per seed) with keys:
///   - "seed": int
///   - "X": ndarray (T, 31) per-tick candidate inputs (full FULL_MASK)
///   - "time": ndarray (T,) sim time (s)
///   - "energy": ndarray (T,) estimated orbital energy (MJ/kg)
///   - "dv": float, total orbital-correction DV (m/s)
///   - "captured": bool
#[pyfunction]
#[pyo3(signature = (toml_path, seeds, overrides=None, sim_timeout_secs=None))]
fn collect_nn_inputs(
    py: Python<'_>,
    toml_path: String,
    seeds: Vec<u64>,
    overrides: Option<&Bound<'_, PyDict>>,
    sim_timeout_secs: Option<f64>,
) -> PyResult<Py<PyList>> {
    use aerocapture::config::GuidanceType;

    let base_overrides = extract_overrides(overrides)?;
    let wall_timeout = sim_timeout_secs.map(std::time::Duration::from_secs_f64);

    // (seed, trace, dv, captured)
    let mut per_seed: Vec<(u64, Vec<(Vec<f64>, f64, f64, f64, f64)>, f64, bool)> =
        Vec::with_capacity(seeds.len());

    py.detach(|| {
        for seed in &seeds {
            let mut seed_overrides = base_overrides.clone();
            seed_overrides.push((
                "simulation.n_sims".to_string(),
                config::OverrideValue::Int(1),
            ));
            seed_overrides.push((
                "monte_carlo.seed".to_string(),
                config::OverrideValue::Int(*seed as i64),
            ));

            let (mut sim_input, sim_data) =
                config::load_and_override(std::path::Path::new(&toml_path), &seed_overrides)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            if sim_input.guidance_type != GuidanceType::NeuralNetwork {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "collect_nn_inputs requires guidance.type = neural_network".to_string(),
                ));
            }
            sim_input.collect_supervised = true;

            let outputs = aerocapture::simulation::runner::run_for_api(
                &sim_input, &sim_data, false, wall_timeout,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Simulation error: {e}"))
            })?;
            if outputs.is_empty() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "collect_nn_inputs: run_for_api returned 0 outputs for seed {} (expected 1)",
                    seed
                )));
            }
            let mut trace: Vec<(Vec<f64>, f64, f64, f64, f64)> = Vec::new();
            let mut dv = f64::NAN;
            let mut captured = false;
            for output in outputs {
                trace.extend(output.supervised_trace);
                dv = output.final_record.get(41).copied().unwrap_or(f64::NAN);
                captured = output.captured;
            }
            per_seed.push((*seed, trace, dv, captured));
        }
        Ok::<_, PyErr>(())
    })?;

    const NN_INPUT_WIDTH: usize = 31;
    let result_list = PyList::empty(py);
    for (seed, trace, dv, captured) in per_seed {
        let n = trace.len();
        let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut time: Vec<f64> = Vec::with_capacity(n);
        let mut energy: Vec<f64> = Vec::with_capacity(n);
        for (nn_input, _bank, _realized, t, e) in trace {
            x_rows.push(nn_input);
            time.push(t);
            energy.push(e);
        }
        // Mirror collect_supervised's X construction exactly (from_vec2 + empty guard).
        let x_arr = if x_rows.is_empty() {
            numpy::PyArray2::<f64>::zeros(py, [0, NN_INPUT_WIDTH], false)
        } else {
            numpy::PyArray2::from_vec2(py, &x_rows).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build X array: {e}"))
            })?
        };
        let dict = PyDict::new(py);
        dict.set_item("seed", seed)?;
        dict.set_item("X", x_arr)?;
        dict.set_item("time", numpy::PyArray1::from_vec(py, time))?;
        dict.set_item("energy", numpy::PyArray1::from_vec(py, energy))?;
        dict.set_item("dv", dv)?;
        dict.set_item("captured", captured)?;
        result_list.append(dict)?;
    }
    Ok(result_list.into())
}
```
(Match the exact `PyArray` reshape idiom `collect_supervised` uses for `X` — read its tail ~lines 520-530 and mirror it; the snippet above assumes `from_vec(...).reshape([n, 31])`.)

Register it next to `collect_supervised`:
```rust
    m.add_function(wrap_pyfunction!(collect_nn_inputs, m)?)?;
```

- [ ] **Step 7: Rebuild + run tests**

```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml 2>&1 | tail -3
uv run pytest tests/test_collect_nn_inputs.py -x 2>&1 | tail -20
cd src/rust && cargo test -p aerocapture 2>&1 | tail -6
uv run pytest tests/test_collect_supervised.py -q 2>&1 | tail -6
```
Expected: new tests PASS; `collect_supervised` tests still PASS (output unchanged); 6 guidance goldens bit-identical.

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/simulation/runner.rs src/rust/src/lib.rs src/rust/src/simulation/tick.rs src/rust/aerocapture-py/src/lib.rs tests/test_collect_nn_inputs.py
git commit -m "feat(nn): collect_nn_inputs captures deployed NN's per-tick inputs + time/energy"
```

---

## Task 2: DV classification + per-input summary (pure functions)

**Files:**
- Create: `src/python/aerocapture/training/nn_input_report.py` (stats + classification only this task)
- Test: `tests/test_nn_input_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nn_input_report.py
import numpy as np
from aerocapture.training.nn_input_report import classify_by_dv, input_summary


def test_classify_by_dv_threshold():
    dv = np.array([100.0, 1000.0, 1500.0])
    # blue (low) = dv < threshold ; red (high) = dv >= threshold
    klass = classify_by_dv(dv, threshold=1000.0)
    assert list(klass) == [0, 1, 1]  # 0=blue(low), 1=red(high)


def test_input_summary_saturation_and_separation():
    # 2 trajectories, 3 ticks, 2 inputs. input0 saturates; input1 separates classes.
    X = [
        np.array([[0.0, -2.0], [0.5, -2.0], [2.0, -2.0]]),  # blue traj
        np.array([[0.0, 2.0], [0.5, 2.0], [2.0, 2.0]]),     # red traj
    ]
    klass = np.array([0, 1])
    rows = input_summary(X, klass, names=["a", "b"], in_mask={0, 1})
    by = {r["name"]: r for r in rows}
    # input a: |2.0|>1 on 2/6 samples
    assert abs(by["a"]["frac_out_of_range"] - 2 / 6) < 1e-9
    # input b: blue mean -2, red mean +2 -> large separation
    assert by["b"]["separation"] > by["a"]["separation"]
    assert by["a"]["in_mask"] is True
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_nn_input_report.py -x 2>&1 | tail -20`
Expected: FAIL — module/functions missing.

- [ ] **Step 3: Implement the pure functions**

```python
# src/python/aerocapture/training/nn_input_report.py
"""Standalone NN input behavior report. See
docs/superpowers/specs/2026-05-29-nn-input-report-design.md."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# class codes
BLUE_LOW_DV = 0
RED_HIGH_DV = 1


def classify_by_dv(dv: npt.NDArray[np.float64], threshold: float) -> npt.NDArray[np.int8]:
    """Blue (0) if final DV < threshold, red (1) otherwise."""
    return np.where(np.asarray(dv) < threshold, BLUE_LOW_DV, RED_HIGH_DV).astype(np.int8)


def input_summary(
    X_list: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    names: list[str],
    in_mask: set[int],
) -> list[dict]:
    """Per-input stats over all (trajectory x timestep) samples.

    Returns one dict per input with p1/p50/p99, frac_out_of_range (|v|>1),
    separation (|mean_red - mean_blue| / pooled_std), and in_mask. Sorted by
    frac_out_of_range desc, then separation desc.
    """
    n_inputs = len(names)
    blue = np.concatenate([X_list[i] for i in range(len(X_list)) if traj_class[i] == BLUE_LOW_DV], axis=0) if any(traj_class == BLUE_LOW_DV) else np.empty((0, n_inputs))
    red = np.concatenate([X_list[i] for i in range(len(X_list)) if traj_class[i] == RED_HIGH_DV], axis=0) if any(traj_class == RED_HIGH_DV) else np.empty((0, n_inputs))
    alls = np.concatenate(X_list, axis=0)
    rows: list[dict] = []
    for j in range(n_inputs):
        col = alls[:, j]
        p1, p50, p99 = np.percentile(col, [1, 50, 99])
        frac_oor = float(np.mean(np.abs(col) > 1.0))
        if blue.shape[0] and red.shape[0]:
            mb, mr = float(blue[:, j].mean()), float(red[:, j].mean())
            pooled = float(np.sqrt(0.5 * (blue[:, j].var() + red[:, j].var()))) + 1e-12
            sep = abs(mr - mb) / pooled
        else:
            sep = 0.0
        rows.append({
            "index": j, "name": names[j],
            "p1": float(p1), "p50": float(p50), "p99": float(p99),
            "frac_out_of_range": frac_oor, "separation": sep,
            "in_mask": j in in_mask,
        })
    rows.sort(key=lambda r: (r["frac_out_of_range"], r["separation"]), reverse=True)
    return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_nn_input_report.py -x 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/nn_input_report.py tests/test_nn_input_report.py
git commit -m "feat(nn): DV classification + per-input saturation/separation summary"
```

---

## Task 3: Charts — binned per-class envelope + panels

**Files:**
- Create: `src/python/aerocapture/training/charts_nn_inputs.py`
- Test: `tests/test_nn_input_report.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_nn_input_report.py
from pathlib import Path

from aerocapture.training.charts_nn_inputs import binned_band, chart_nn_input_panel


def test_binned_band_shapes_and_values():
    # x in [0,10], y = x; one class. 5 bins -> centers + p5/p95 per bin.
    x = np.linspace(0, 10, 100)
    y = x.copy()
    centers, lo, hi = binned_band(x, y, n_bins=5, lo_pct=5, hi_pct=95)
    assert centers.shape == lo.shape == hi.shape == (5,)
    assert np.all(hi >= lo)
    assert centers[0] < centers[-1]


def test_chart_nn_input_panel_writes_svg(tmp_path):
    rng = np.random.default_rng(0)
    X_list = [rng.uniform(-1.5, 1.5, size=(20, 31)) for _ in range(6)]
    time_list = [np.arange(20.0) for _ in range(6)]
    klass = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    out = tmp_path / "panel.svg"
    chart_nn_input_panel(X_list, time_list, klass, input_index=5,
                         name="accel_magnitude", in_mask=True, output=out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_nn_input_report.py -k "binned_band or panel" -x 2>&1 | tail -15`
Expected: FAIL — module/functions missing.

- [ ] **Step 3: Implement charts**

```python
# src/python/aerocapture/training/charts_nn_inputs.py
"""Per-input spaghetti + binned envelope panels for the NN input report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

COLOR_BLUE = "#1f77b4"
COLOR_RED = "#d62728"
BLUE_LOW_DV = 0
RED_HIGH_DV = 1


def binned_band(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_bins: int = 40,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    min_count: int = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Bin samples (x, y) into n_bins on x; return (centers, lo_pct, hi_pct) per
    bin. Bins with < min_count samples are dropped (NaN-free output)."""
    x = np.asarray(x); y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centers, lo, hi = [], [], []
    for b in range(n_bins):
        yb = y[idx == b]
        if yb.size >= min_count:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            lo.append(np.percentile(yb, lo_pct))
            hi.append(np.percentile(yb, hi_pct))
    return np.array(centers), np.array(lo), np.array(hi)


def _class_xy(X_list, axis_list, traj_class, input_index, cls):
    xs, ys = [], []
    for i in range(len(X_list)):
        if traj_class[i] == cls:
            xs.append(np.asarray(axis_list[i]))
            ys.append(np.asarray(X_list[i])[:, input_index])
    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def chart_nn_input_panel(
    X_list: list[npt.NDArray[np.float64]],
    axis_list: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    input_index: int,
    name: str,
    in_mask: bool,
    output: Path,
    x_label: str = "time (s)",
) -> None:
    """One panel: blue/red spaghetti + per-class p5-p95 band + +-1 guides."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    alpha = max(0.03, min(0.5, 30.0 / max(1, len(X_list))))
    for i in range(len(X_list)):
        color = COLOR_BLUE if traj_class[i] == BLUE_LOW_DV else COLOR_RED
        ax.plot(axis_list[i], np.asarray(X_list[i])[:, input_index], color=color,
                alpha=alpha, linewidth=0.5)
    for cls, color in ((BLUE_LOW_DV, COLOR_BLUE), (RED_HIGH_DV, COLOR_RED)):
        cx, cy = _class_xy(X_list, axis_list, traj_class, input_index, cls)
        c, lo, hi = binned_band(cx, cy)
        if c.size:
            ax.fill_between(c, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    ax.axhline(-1.0, color="grey", linestyle="--", linewidth=0.7)
    title = f"[{input_index}] {name}" + ("" if in_mask else "  (unused)")
    ax.set_title(title, color=("black" if in_mask else "grey"))
    ax.set_xlabel(x_label)
    ax.set_ylabel("normalized value")
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_nn_input_report.py -k "binned_band or panel" -x 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/charts_nn_inputs.py tests/test_nn_input_report.py
git commit -m "feat(nn): binned per-class envelope + spaghetti panel charts for NN inputs"
```

---

## Task 4: CLI orchestration + outputs

**Files:**
- Modify: `src/python/aerocapture/training/nn_input_report.py` (add `run_report` + `main` CLI)
- Test: `tests/test_nn_input_report.py` (append CLI smoke)

- [ ] **Step 1: Write the failing CLI smoke test**

```python
# append to tests/test_nn_input_report.py
import json as _json

import pytest


@pytest.mark.slow
def test_run_report_smoke(tmp_path):
    from aerocapture.training.nn_input_report import run_report

    model = _mint_zero_model(tmp_path)  # reuse helper from test_collect_nn_inputs
    out_dir = tmp_path / "rep"
    run_report(
        toml_path="configs/training/msr_aller_nn_delta_train.toml",
        n_sims=4,
        output_dir=out_dir,
        overrides={"data.neural_network": model},
    )
    assert (out_dir / "summary.json").exists()
    summary = _json.loads((out_dir / "summary.json").read_text())
    assert len(summary["inputs"]) == 31
    # at least one time-axis panel rendered
    assert list(out_dir.glob("nn_input_*_time.svg"))
```

> `_mint_zero_model` lives in `tests/test_collect_nn_inputs.py`; import it or duplicate the small helper into `tests/test_nn_input_report.py`. Duplicating the ~12-line helper is acceptable (test code).

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_nn_input_report.py -k run_report_smoke -x 2>&1 | tail -20`
Expected: FAIL — `run_report` missing.

- [ ] **Step 3: Implement `run_report` + `main`**

Append to `src/python/aerocapture/training/nn_input_report.py`:
```python
import argparse
import json
from pathlib import Path

import aerocapture_rs

from aerocapture.training.ablation import NN_INPUT_NAMES, _load_cost_kwargs
from aerocapture.training.charts_nn_inputs import chart_nn_input_panel
from aerocapture.training.toml_utils import load_toml_with_bases


def _resolve_mask(toml_path: str) -> set[int]:
    cfg = load_toml_with_bases(Path(toml_path))
    mask = cfg.get("network", {}).get("input_mask")
    return set(mask) if mask is not None else set(range(16))


def _default_dv_threshold(toml_path: str) -> float:
    return float(_load_cost_kwargs(toml_path).get("dv_threshold", 1000.0))


def run_report(
    toml_path: str,
    n_sims: int = 500,
    output_dir: Path | None = None,
    dv_threshold: float | None = None,
    overrides: dict | None = None,
) -> Path:
    out_dir = Path(output_dir) if output_dir else Path("nn_input_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = dv_threshold if dv_threshold is not None else _default_dv_threshold(toml_path)
    in_mask = _resolve_mask(toml_path)

    # Reserved, disjoint seed stream (offset distinct from train/val/final-eval).
    NN_INPUT_REPORT_SEED_OFFSET = 5_000_000
    seeds = [NN_INPUT_REPORT_SEED_OFFSET + i for i in range(n_sims)]
    recs = aerocapture_rs.collect_nn_inputs(toml_path, seeds, overrides=overrides)

    import numpy as np

    X_list = [r["X"] for r in recs]
    time_list = [r["time"] for r in recs]
    energy_list = [r["energy"] for r in recs]
    dv = np.array([r["dv"] for r in recs], dtype=np.float64)
    klass = classify_by_dv(dv, thr)

    rows = input_summary(X_list, klass, NN_INPUT_NAMES, in_mask)
    (out_dir / "summary.json").write_text(json.dumps(
        {"dv_threshold": thr, "n_sims": n_sims,
         "n_blue": int(np.sum(klass == BLUE_LOW_DV)),
         "n_red": int(np.sum(klass == RED_HIGH_DV)),
         "inputs": rows}, indent=2))

    for j, nm in enumerate(NN_INPUT_NAMES):
        chart_nn_input_panel(X_list, time_list, klass, j, nm, j in in_mask,
                             out_dir / f"nn_input_{j:02d}_{nm}_time.svg", x_label="time (s)")
        chart_nn_input_panel(X_list, energy_list, klass, j, nm, j in in_mask,
                             out_dir / f"nn_input_{j:02d}_{nm}_energy.svg", x_label="energy (MJ/kg)")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="NN input behavior report")
    ap.add_argument("training_dir")
    ap.add_argument("--toml", required=True)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--dv-threshold", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    out = args.output_dir or str(Path(args.training_dir) / "nn_input_report")
    run_report(args.toml, n_sims=args.n_sims, output_dir=Path(out),
               dv_threshold=args.dv_threshold)
    print(f"NN input report written to {out}")


if __name__ == "__main__":
    main()
```

> Move the `import numpy` / `import argparse` etc. to the module top if the file's lint style requires it (keep `from __future__ import annotations` first). Confirm `_load_cost_kwargs` and `NN_INPUT_NAMES` are importable from `ablation.py` (they are module-level there).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_nn_input_report.py -k run_report_smoke -x 2>&1 | tail -15`
Expected: PASS — `summary.json` with 31 inputs + time-axis SVGs.

- [ ] **Step 5: Lint + full module tests**

Run:
```bash
uv run ruff check src/python/aerocapture/training/nn_input_report.py src/python/aerocapture/training/charts_nn_inputs.py 2>&1 | tail -3
uv run mypy src/python/aerocapture/training/nn_input_report.py src/python/aerocapture/training/charts_nn_inputs.py 2>&1 | tail -3
uv run pytest tests/test_nn_input_report.py -q 2>&1 | tail -6
```
Expected: clean + green.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/nn_input_report.py tests/test_nn_input_report.py
git commit -m "feat(nn): nn_input_report CLI -- collect, classify, render panels + summary"
```

---

## Task 5: Verification + docs sync

**Files:**
- Verify only; docs via `smart-commit`.

- [ ] **Step 1: Full suites**

Run:
```bash
./check_all.sh 2>&1 | tail -12
./lint_code.sh 2>&1 | tail -8
uv run pytest tests -q 2>&1 | tail -6
```
Expected: Rust test/fmt/clippy/build clean (6 goldens bit-identical), ruff/mypy clean, pytest green.

- [ ] **Step 2: Real CLI run against a trained model (manual sanity)**

If a trained NN scheme exists (e.g. after the delta/scaledpi PSO run), run for real and eyeball:
```bash
uv run python -m aerocapture.training.nn_input_report training_output/neural_network_delta_pso \
    --toml configs/training/msr_aller_nn_delta_train.toml --n-sims 200
```
Expected: `training_output/neural_network_delta_pso/nn_input_report/` with 62 SVGs + `summary.json`; the summary's top rows should surface any saturating / high-separation inputs.

- [ ] **Step 3: Docs sync (smart-commit)**

Invoke the `smart-commit` skill, instructing it to take the whole `feature/nn-input-report` branch into account: document the new `collect_nn_inputs` PyO3 API and the `nn_input_report` CLI in `CLAUDE.md` (PyO3 Bindings section + Python Tools list + the testing inventory) and `README.md` if it lists analysis tools, then commit.

---

## Optional follow-up (NOT in this plan): Typst PDF

The spec lists an optional Typst PDF compiling the 62 SVGs + summary table. Defer unless wanted — the SVGs + `summary.json` already deliver the diagnostic. If added later: a `typst/nn_input_report.typ` template + a `_compile_pdf` helper guarded on `shutil.which("typst")`, mirroring `report.py`'s graceful-degradation pattern.

---

## Self-review notes

- **Spec coverage:** collect_nn_inputs runs the NN, not a teacher (T1); time+energy capture (T1); DV-threshold blue/red (T2/T4); 31 inputs x {time,energy} panels with +-1 guides + greyed-unused (T3/T4); per-input saturation/separation summary (T2); standalone CLI + outputs (T4); reserved seed offset 5M (T4); collect_supervised output unchanged + regression (T1). Typst PDF intentionally deferred (spec marked it optional; called out above).
- **Type consistency:** `classify_by_dv` returns int8 (0=blue,1=red) used identically in `input_summary`, `chart_nn_input_panel`, and `run_report`; trace 5-tuple `(nn_input, y_signed, prev_realized, time, energy)` consistent across runner.rs/lib.rs/tick.rs/collect_supervised/collect_nn_inputs; `binned_band` signature matches its callers.
- **Fixture:** `_mint_zero_model` derives arch/mask/knobs from the resolved delta TOML so it can't drift from the config; X is always 31-wide (FULL_MASK) regardless of the model's mask, so the shape assertions hold for any model.
