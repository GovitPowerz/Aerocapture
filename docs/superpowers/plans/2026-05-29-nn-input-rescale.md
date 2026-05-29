# NN Input Rescaling (asinh) + Periapsis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace clamps/ad-hoc linear scalings on the 5 saturating NN inputs with a signed-log `asinh(x/s)` transform (scales measured data-driven), and add `periapsis_alt` as a new input at index 31 (`NN_FULL_INPUT_SIZE` 31→32).

**Architecture:** `build_nn_input` rescales 5 inputs in place + appends periapsis; scale constants are measured once (Python, from the existing report summary + a `run_mc` trajectory pass) and baked as Rust consts. The 31→32 growth ripples through `FULL_MASK`, the PyO3 collect width, `NN_INPUT_NAMES`, the width-guard constants, and the `input_mask` validation bound. The `neural` guidance golden regenerates.

**Tech Stack:** Rust (nalgebra, PyO3), Python 3.14 (numpy), pytest, Typst (report acceptance), `typst`/`cargo`/`uv`.

**Spec:** `docs/superpowers/specs/2026-05-29-nn-input-rescale-design.md`

**Branch:** `feature/nn-input-rescale` (off `feature/nn-input-report`; spec already committed).

**Scale target refinement (deviation from spec):** the spec said `asinh(p99/s) ≈ 1.3`; that leaves ~10% of samples with `|v|>1`. To meet the spec's own <5% saturation acceptance, this plan uses **`s = max(|p1_raw|,|p99_raw|) / sinh(1.0)`** (p99 maps to ≈1.0 → ~1% saturation). Flagged for the user.

---

## File map

- `src/rust/src/gnc/guidance/neural.rs` — scale consts + asinh rescale of indices 2/3/13/14/18 + new index 31 (periapsis); `NN_FULL_INPUT_SIZE` 31→32; module-doc input table.
- `src/rust/src/simulation/tick.rs` — `FULL_MASK` `[usize; 31]` → `[usize; 32]`.
- `src/rust/aerocapture-py/src/lib.rs` — `NN_INPUT_WIDTH` const 31→32 in `collect_supervised` AND `collect_nn_inputs`.
- `src/rust/src/data/mod.rs` — `input_mask` validation upper bound `max(31, …)` → `max(32, …)`.
- `src/rust/src/data/neural.rs` — `validate_ablated_input` last-valid test 30→31 (NN_FULL_INPUT_SIZE-1).
- `src/python/aerocapture/training/ablation.py` — `NN_INPUT_NAMES` +`periapsis_alt` (→32).
- `src/python/aerocapture/training/warm_start.py` — `_CANDIDATE_INPUT_WIDTH` 31→32.
- `src/python/aerocapture/training/config.py` — `_RUNTIME_CANDIDATE_WIDTH` 31→32.
- `tests/test_ablation.py` — length assertion 31→32.
- `configs/training/msr_aller_nn_delta_train.toml`, `configs/training/msr_aller_nn_scaledpi_train.toml` — append index 31 to `input_mask`, bump first-layer `input_size` +1.
- `tests/reference_data/rust_golden/` — regenerate the `neural` golden.

---

## Task 1: Measure scales + rescale `build_nn_input` + periapsis (+ FULL_MASK / collect width)

These are coupled: the size 31→32 + new index 31 must move together with `FULL_MASK` and the PyO3 collect width or the collect path breaks. The measurement runs FIRST (on the current, pre-change code, while the deployed delta model is still valid).

**Files:** Modify `src/rust/src/gnc/guidance/neural.rs`, `src/rust/src/simulation/tick.rs`, `src/rust/aerocapture-py/src/lib.rs`. Test: inline `#[cfg(test)]` in `neural.rs` + rebuild bindings.

- [ ] **Step 1: Measure the 6 scale constants (run on the CURRENT, unmodified code)**

Run this from repo root and RECORD the 6 printed `S_*` values for Step 3:
```bash
uv run python - <<'PY'
import json, math
from pathlib import Path
import numpy as np, aerocapture_rs
from aerocapture.training.toml_utils import load_toml_with_bases

T = 1.0  # asinh target: p99 -> ~1.0 (=> ~1% saturation, under the 5% bar)
sinhT = math.sinh(T)

# (a) Invert the current scalings on the existing delta report summary (4 inputs).
summ = json.load(open("training_output/neural_network_delta_pso/nn_input_report/summary.json"))
by = {r["name"]: r for r in summ["inputs"]}
inv = {
    "radial_velocity": lambda s: ((s + 1.0) * 1.5 / 2.0 - 1.2) * 1e3,  # scaled=2*(vr/1e3+1.2)/1.5-1
    "orbital_energy":  lambda s: s * 6e6,                               # scaled=raw/6e6
    "sma_error":       lambda s: s * 5e5,                               # scaled=raw/5e5
    "hdot_nominal":    lambda s: s * 500.0,                             # scaled=raw/500
}
scales = {}
for nm, f in inv.items():
    r = by[nm]
    p = max(abs(f(r["p1"])), abs(f(r["p99"])))
    scales[nm] = p / sinhT

# (b) Reconstruct apoapsis/periapsis raw from a trajectory ensemble (delta config, current model).
cfg = load_toml_with_bases(Path("configs/training/msr_aller_nn_delta_train.toml"))
mu = float(cfg["planet"]["mu"]); Req = float(cfg["planet"]["equatorial_radius"])
res = aerocapture_rs.run_mc("configs/training/msr_aller_nn_delta_train.toml",
                            overrides={"simulation.n_sims": 200}, include_trajectories=True)
apo, per = [], []
for tr in res.trajectories:  # (T,17): 0 alt_km, 3 vel, 4 fpa_deg, 8 energy_mj_kg
    tr = np.asarray(tr)
    r = Req + tr[:, 0] * 1e3
    v = tr[:, 3]; gamma = np.radians(tr[:, 4]); E = tr[:, 8] * 1e6
    a = -mu / (2.0 * E)
    h = r * v * np.cos(gamma)
    e = np.sqrt(np.maximum(0.0, 1.0 + 2.0 * E * h * h / (mu * mu)))
    apo.append(a * (1.0 + e) - Req); per.append(a * (1.0 - e) - Req)
for nm, arr in (("apoapsis_alt", np.concatenate(apo)), ("periapsis_alt", np.concatenate(per))):
    arr = arr[np.isfinite(arr)]
    p = max(abs(np.percentile(arr, 1)), abs(np.percentile(arr, 99)))
    scales[nm] = p / sinhT

for nm in ("radial_velocity", "orbital_energy", "sma_error", "apoapsis_alt", "hdot_nominal", "periapsis_alt"):
    print(f"S_{nm.upper()} = {scales[nm]:.6e}")
PY
```
Expected: 6 lines like `S_APOAPSIS_ALT = 1.2e+07`. Record them. (If `training_output/neural_network_delta_pso/nn_input_report/summary.json` is absent, first run `python -m aerocapture.training.nn_input_report training_output/neural_network_delta_pso --toml configs/training/msr_aller_nn_delta_train.toml --n-sims 200`.)

- [ ] **Step 2: Write the failing tests** (in `neural.rs` test module)

```rust
#[test]
fn nn_full_input_size_is_32() {
    assert_eq!(NN_FULL_INPUT_SIZE, 32);
}

#[test]
fn rescaled_inputs_use_asinh_and_periapsis_present() {
    let (nav, data, planet) = /* existing neural-test fixture */;
    let mask: Vec<usize> = (0..32).collect();
    let v = build_nn_input(&nav, Some(&mask), None, &data, &planet,
        0.0, 0.0, Some(0.0), 0.3, 0.0, 0.0, 0.0);
    assert_eq!(v.len(), 32);
    assert!(v.iter().all(|x| x.is_finite()));
    // index 14 (apoapsis) is now asinh-scaled, so a huge apoapsis stays bounded:
    // the fixture's apoapsis maps through asinh -> |v[14]| is O(1), not pinned at the old ±9 clamp.
    assert!(v[14].abs() < 5.0);
    // index 31 = periapsis present + finite.
    assert!(v[31].is_finite());
}

#[test]
fn asinh_rescale_bounds_huge_values() {
    // asinh compresses: a value 100x the scale maps to ~asinh(100) ≈ 5.3, not 100.
    assert!((100.0_f64).asinh() < 6.0);
}

#[test]
fn default_mask_path_still_16() {
    let (nav, data, planet) = /* fixture */;
    let v = build_nn_input(&nav, None, None, &data, &planet,
        0.0, 0.0, Some(0.0), 0.3, 0.0, 0.0, 9.9);
    assert_eq!(v.len(), 16);
}
```
(Reuse the fixture the existing `build_nn_input` tests use — e.g. the one feeding `full_input_default_mask_is_16_and_ignores_prev_realized`.)

- [ ] **Step 3: Implement — consts + asinh rescale + periapsis + size 32**

In `neural.rs`, add the measured consts near the module top (use YOUR Step-1 numbers):
```rust
// asinh signed-log scale factors: s = max(|p1|,|p99|)/sinh(1.0), measured from a
// representative MC ensemble (see plan Task 1). asinh(raw/s) keeps the operating
// range within ~[-1,1] without clamping; tails compress.
const S_RADIAL_VELOCITY: f64 = /* S_RADIAL_VELOCITY from Step 1 */;
const S_ORBITAL_ENERGY: f64 = /* ... */;
const S_SMA_ERROR: f64 = /* ... */;
const S_APOAPSIS_ALT: f64 = /* ... */;
const S_HDOT_NOMINAL: f64 = /* ... */;
const S_PERIAPSIS_ALT: f64 = /* ... */;
```
Set `NN_FULL_INPUT_SIZE = 32` (in `data/neural.rs`). Replace the 5 assignments in `build_nn_input`:
```rust
    full_input[2] = (velocity_radial / S_RADIAL_VELOCITY).asinh();                         // radial velocity
    full_input[3] = (-mu / (2.0 * orbit.semi_major_axis) / S_ORBITAL_ENERGY).asinh();      // orbital energy
    full_input[13] = (nav.orbital_errors[0] / S_SMA_ERROR).asinh();                        // SMA error
    full_input[14] = (orbit.apoapsis_alt / S_APOAPSIS_ALT).asinh();                        // apoapsis altitude (unclamped)
    // ... (index 18 set after hdot_nominal is computed):
    full_input[18] = (hdot_nominal / S_HDOT_NOMINAL).asinh();                              // ref radial velocity
```
Append the new input after the (sin,cos) pairs (index 31):
```rust
    full_input[31] = (orbit.periapsis_alt / S_PERIAPSIS_ALT).asinh();                      // periapsis altitude
```
Update the module-doc input table (note rescaled 2/3/13/14/18 + new 31 periapsis_alt).

- [ ] **Step 4: Widen FULL_MASK + collect width (keep the crate + collect compiling)**

`tick.rs`: `const FULL_MASK: [usize; 32] = [0, 1, …, 31];`
`aerocapture-py/src/lib.rs`: bump `const NN_INPUT_WIDTH: usize = 31;` → `32` in BOTH `collect_supervised` and `collect_nn_inputs`.

- [ ] **Step 5: Build + run tests**

```bash
cd src/rust && cargo test -p aerocapture neural 2>&1 | tail -20
cd /Users/govit/Git/Govit/Aerocapture && uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml 2>&1 | tail -3
uv run pytest tests/test_collect_nn_inputs.py tests/test_collect_supervised.py -q 2>&1 | tail -6
```
Expected: new neural tests PASS; collect tests now expect width 32 — they currently assert 31 and will FAIL here. Leave them; Task 2 updates the width assertions. (If you prefer green-at-each-task, update `tests/test_collect_nn_inputs.py` / `tests/test_collect_supervised.py` X-width asserts 31→32 in THIS task and note it.) Recommended: update those two width asserts here so the suite is green.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs src/rust/src/data/neural.rs src/rust/src/simulation/tick.rs src/rust/aerocapture-py/src/lib.rs tests/test_collect_nn_inputs.py tests/test_collect_supervised.py
git commit -m "feat(nn): asinh-rescale 5 saturating inputs + add periapsis_alt (NN_FULL_INPUT_SIZE 32)"
```
Record the 6 measured constants in the commit body for provenance.

---

## Task 2: width-guard ripples + names + validation bound

**Files:** `src/python/aerocapture/training/ablation.py`, `tests/test_ablation.py`, `src/python/aerocapture/training/warm_start.py`, `src/python/aerocapture/training/config.py`, `src/rust/src/data/mod.rs`, `src/rust/src/data/neural.rs`.

- [ ] **Step 1: Failing test — NN_INPUT_NAMES length 32 + periapsis**

In `tests/test_ablation.py`, update the length assertion and add a periapsis check:
```python
def test_input_names_length() -> None:
    """32 inputs: 16 baseline + 4 ref + 1 exit-teacher + 4 lateral telemetry + 6 (sin,cos) pairs + periapsis_alt."""
    assert len(NN_INPUT_NAMES) == 32
    assert "periapsis_alt" in NN_INPUT_NAMES
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_ablation.py::test_input_names_length -x 2>&1 | tail -8`
Expected: FAIL (len 31, no periapsis_alt).

- [ ] **Step 3: Add `periapsis_alt` to NN_INPUT_NAMES**

In `ablation.py`, append after the last `(sin,cos)` entry (index 30):
```python
    "prev_realized_cos",  # 30
    "periapsis_alt",  # 31
]
```

- [ ] **Step 4: Bump the width guards + validation bound**

`warm_start.py`: `_CANDIDATE_INPUT_WIDTH = 31` → `32` (+ update its comment).
`config.py`: `_RUNTIME_CANDIDATE_WIDTH = 31` → `32`.
`data/mod.rs`: the `input_mask` upper-bound `max(31, architecture[0].input_size)` → `max(32, …)`.
`data/neural.rs`: in `validate_ablated_input_valid` test, `// index 30 is the last valid index` / `Some(30)` → `31`; and the companion out-of-range test if it hardcodes a value that is now in range.

- [ ] **Step 5: Run to verify pass**

```bash
uv run pytest tests/test_ablation.py -q 2>&1 | tail -5
uv run ruff check src/python/aerocapture/training/ablation.py src/python/aerocapture/training/warm_start.py src/python/aerocapture/training/config.py 2>&1 | tail -2
cd src/rust && cargo test -p aerocapture validate_ablated 2>&1 | tail -6
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/ablation.py tests/test_ablation.py src/python/aerocapture/training/warm_start.py src/python/aerocapture/training/config.py src/rust/src/data/mod.rs src/rust/src/data/neural.rs
git commit -m "chore(nn): widen input-vector guards to 32 + periapsis_alt name"
```

---

## Task 3: training configs (delta + scaled_pi)

**Files:** `configs/training/msr_aller_nn_delta_train.toml`, `configs/training/msr_aller_nn_scaledpi_train.toml`.

- [ ] **Step 1: Append periapsis (31) to the masks + bump first-layer input_size**

In each config: add `, 31` to the end of `input_mask` (delta goes 29→30 indices), and change the FIRST `[[network.architecture]]` block's `input_size` to equal the new mask length (delta: `29` → `30`). Leave the hidden/output layers untouched. Update the `[network]` comment to mention periapsis_alt (31). (Read each config first — scaled_pi may have a different arch; match its first-layer input_size to its new mask length.)

- [ ] **Step 2: Config-load smoke (Rust validates mask vs layer-0 size + index < 32)**

```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
for f in ['configs/training/msr_aller_nn_delta_train.toml','configs/training/msr_aller_nn_scaledpi_train.toml']:
    d = load_toml_with_bases(f)
    mask = d['network']['input_mask']; arch0 = d['network']['architecture'][0]['input_size']
    assert 31 in mask, f
    assert len(mask) == arch0, (f, len(mask), arch0)
    print(f, 'mask', len(mask), 'arch0', arch0, 'OK')
"
```
Expected: both print OK (mask length == first-layer input_size, 31 present).

- [ ] **Step 3: Commit**

```bash
git add configs/training/msr_aller_nn_delta_train.toml configs/training/msr_aller_nn_scaledpi_train.toml
git commit -m "feat(nn): add periapsis_alt (idx 31) to delta/scaledpi input masks"
```

---

## Task 4: regenerate `neural` golden + full verification

**Files:** `tests/reference_data/rust_golden/` (neural golden CSV).

- [ ] **Step 1: Identify the neural golden test + its config**

```bash
cd src/rust && rg -n "neural" tests/ | rg -i "golden|regression|\.csv" | head
ls tests/reference_data/rust_golden/ | rg -i neural
```
Find the neural golden CSV + the test config it runs (the guidance_regression harness names it).

- [ ] **Step 2: Confirm the golden test now FAILS (expected — inputs rescaled)**

```bash
cd src/rust && cargo test -p aerocapture --test e2e guidance_regression -- --test-threads=1 2>&1 | tail -15
```
Expected: the `neural` case FAILS (rescaled inputs changed NN output); the other 5 (ftc, eqglide, energy_ctrl, pred_guid, fnpag) PASS bit-identical. If any non-neural case fails, STOP and report BLOCKED.

- [ ] **Step 3: Regenerate the neural golden**

Run the release binary on the neural golden config and replace the CSV (mirror the documented "Golden File Regeneration" procedure). Determine the exact config + output path from Step 1, then:
```bash
./src/rust/target/release/aerocapture <neural_golden_config.toml>   # writes the photo/csv
# copy the produced CSV over tests/reference_data/rust_golden/<neural_golden>.csv
```
Eyeball the diff: changes must be confined to NN-driven columns (bank angle and downstream state); the input vector itself isn't in the golden. Confirm the trajectory is still physically sane (captures / behaves like before — rescaling inputs to a then-untrained-on-new-scaling model changes the bank profile, which is expected; the golden just pins the new deterministic output).

- [ ] **Step 4: Verify the golden passes + full suites**

```bash
cd src/rust && cargo test -p aerocapture --test e2e guidance_regression -- --test-threads=1 2>&1 | tail -6
cd /Users/govit/Git/Govit/Aerocapture && ./check_all.sh 2>&1 | tail -12
./lint_code.sh 2>&1 | tail -6
uv run pytest tests -q 2>&1 | tail -5
```
Expected: 6 goldens pass; Rust fmt/clippy/test/build clean (run `cargo fmt` if formatting fails); ruff/mypy clean; pytest green. (The `mc_deterministic_same_seed` e2e flake is pre-existing/parallel-cwd — verify single-threaded if it appears.)

- [ ] **Step 5: Acceptance — asinh achieves the saturation target (unit-level, model-independent)**

Add a Rust test asserting the asinh wiring meets the target at the measured p99 (replace `<p99_apoapsis>` / `<S_APOAPSIS_ALT>` with Task 1's values):
```rust
#[test]
fn apoapsis_asinh_maps_p99_near_one() {
    // s was set so asinh(p99/s)=1.0 => the measured p99 raw maps to ~1.0 (=> ~1% saturation).
    let p99_raw: f64 = /* p99 apoapsis raw from Task 1 measurement */;
    assert!(((p99_raw / S_APOAPSIS_ALT).asinh() - 1.0).abs() < 0.05);
}
```
(Empirical end-to-end saturation re-verification via `nn_input_report` is deferred until a model is retrained on the new scalings — the current deployed weights are invalid post-rescale, so a fresh MC ensemble would not be representative. The unit assertion + the construction `s=p99/sinh(1)` guarantee the target on the measured distribution.)

- [ ] **Step 6: Commit**

```bash
git add tests/reference_data/rust_golden/ src/rust/src/gnc/guidance/neural.rs
git commit -m "test(nn): regenerate neural golden for rescaled inputs + asinh target assertion"
```

---

## Task 5: docs sync (smart-commit)

- [ ] **Step 1: Sync docs**

Invoke the `smart-commit` skill, instructing it to take the whole `feature/nn-input-rescale` branch into account: update `CLAUDE.md` — the `build_nn_input` description and the candidate-input table (rescaled 2/3/13/14/18 via asinh, new index 31 periapsis_alt, `NN_FULL_INPUT_SIZE = 32`), the ablation `NN_INPUT_NAMES` count (31→32), and the input-rescaling note — then commit. README only if it documents the input set.

---

## Self-review notes

- **Spec coverage:** asinh transform + measured scales (T1); the 5 rescales + periapsis at 31 + size 32 (T1); plumbing ripples FULL_MASK/collect width/NN_INPUT_NAMES/width guards/validation bound (T1+T2); configs (T3); neural golden regen + cross-language-unaffected (T4); acceptance via construction + unit assertion (T4). All spec sections mapped.
- **Deviation flagged:** `asinh(p99/s)` target tightened from the spec's `1.3` to `1.0` so the <5% saturation acceptance is actually met (p99→1.0 ⇒ ~1% saturation). Surface to the user.
- **Type/scale consistency:** the six `S_*` consts are named identically in `build_nn_input` and the Task-4 assertion; `NN_FULL_INPUT_SIZE = 32`, `FULL_MASK [usize;32]`, `NN_INPUT_WIDTH 32`, `NN_INPUT_NAMES` len 32 all agree; mask length == first-layer input_size == 30 for delta.
- **Non-broken intermediates:** T1 keeps the crate + collect compiling (FULL_MASK + width bumped together); recommend updating the two collect X-width test asserts in T1 (noted) so the suite stays green per-task.
- **Measurement provenance:** the 6 consts are produced by T1 Step 1 (deterministic procedure) and recorded in the commit body — not placeholders.
