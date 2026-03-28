# Training Fixes & DV Chart Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix GA training hangs (NaN infinite loops + missing wall-clock timeout), repair FTC training config, improve DV chart visualizations, and add log-scale encoding for pred_guid.pdyn_threshold.

**Architecture:** The hang fix adds a NaN/Inf state check in the Rust sim loop (`run_single` in `runner.rs`) and threads an optional wall-clock timeout from PyO3 through `run_core` to `run_single`. Chart changes refactor existing DV chart functions in `charts.py` to use a shared tick helper and 3-row subplot layout.

**Tech Stack:** Rust (nalgebra, std::time), PyO3/maturin, Python (matplotlib, numpy), TOML configs

---

### Task 1: Fix subprocess exception syntax (B4)

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:212`

- [ ] **Step 1: Fix the Python 2 except syntax**

In `src/python/aerocapture/training/evaluate.py`, line 212, change:

```python
    except subprocess.TimeoutExpired, FileNotFoundError:
```

to:

```python
    except (subprocess.TimeoutExpired, FileNotFoundError):  # fmt: skip
```

The `# fmt: skip` prevents ruff from removing the parentheses.

- [ ] **Step 2: Verify lint passes**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run ruff check src/python/aerocapture/training/evaluate.py && uv run ruff format --check src/python/aerocapture/training/evaluate.py`

Expected: no errors, no formatting changes needed.

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py
git commit -m "fix: Python 3 except syntax in subprocess fallback

The comma syntax (except A, B:) is a SyntaxError in Python 3.
Added # fmt: skip to prevent ruff from removing the parentheses."
```

---

### Task 2: NaN termination in Rust sim loop (B1)

**Files:**
- Modify: `src/rust/src/simulation/runner.rs:646`

- [ ] **Step 1: Add NaN/Inf state check before termination checks**

In `src/rust/src/simulation/runner.rs`, insert the following **before** line 646 (the `// === Termination checks ===` comment), right after line 644 (`track_peak_values`):

```rust
        // NaN/Inf safety net: extreme GA parameters can blow up numerically.
        // All termination checks evaluate to false on NaN, so the loop would spin forever.
        if sim.state.iter().any(|x| !x.is_finite()) {
            term = TermReason::Crash;
            break;
        }
```

The `break` exits immediately — no point running further checks on a NaN state.

- [ ] **Step 2: Verify Rust builds and existing tests pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release && cargo test`

Expected: all existing tests pass, no compiler warnings.

- [ ] **Step 3: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/simulation/runner.rs
git commit -m "fix: terminate sim on NaN/Inf state to prevent infinite loops

Extreme GA parameter combinations can blow up numerically. Since
NaN comparisons always return false, no termination check would fire,
causing the while loop to spin forever. Now detected and terminated
as Crash with virtual DV penalty (~15k m/s)."
```

---

### Task 3: Wall-clock timeout — Rust core (B2, part 1)

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` — `run_single`, `run_core`, `run_for_api` signatures + sim loop

- [ ] **Step 1: Add `wall_timeout` parameter to `run_single`**

In `src/rust/src/simulation/runner.rs`, add `use std::time::{Duration, Instant};` at the top of the file (near other `use` statements), then change the `run_single` signature at line 401:

```rust
fn run_single(
    config: &SimInput,
    data: &SimData,
    run_state: &init::RunState,
    sim_idx: i32,
    write_photo: bool,
    wall_timeout: Option<Duration>,
) -> Result<SimResult, SimError> {
```

- [ ] **Step 2: Record start time and check in sim loop**

In `run_single`, right before the main sim loop (line ~490, before `let mut sim_time`), add:

```rust
    let wall_start = Instant::now();
```

Then in the termination checks block (after the NaN check added in Task 2, before `if altitude <= 0.0`), add:

```rust
        if let Some(timeout) = wall_timeout {
            if wall_start.elapsed() > timeout {
                term = TermReason::Timeout;
            }
        }
```

- [ ] **Step 3: Thread `wall_timeout` through `run_core`**

Change the `run_core` signature at line 111:

```rust
fn run_core(
    config: &SimInput,
    data: &SimData,
    write_photo: bool,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<SimResult>, SimError> {
```

Update both `run_single` call sites inside `run_core`:

At line ~167 (parallel path):
```rust
                let mut result = run_single(config, data, run_state, idx as i32, do_photo, wall_timeout)?;
```

At line ~192 (single path):
```rust
        let mut result = run_single(
            config,
            data,
            run_state,
            0,
            write_photo || include_trajectories,
            wall_timeout,
        )?;
```

- [ ] **Step 4: Thread through `run` (CLI) and `run_for_api`**

Update `run()` at line 205 — pass `None` (CLI has no wall timeout):
```rust
    let results = run_core(config, data, true, false, None)?;
```

Update `run_for_api()` at line 226 — add parameter and pass through:
```rust
pub fn run_for_api(
    config: &SimInput,
    data: &SimData,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<crate::RunOutput>, SimError> {
    let results = run_core(config, data, false, include_trajectories, wall_timeout)?;
```

- [ ] **Step 5: Update integration test callers**

In `src/rust/tests/e2e.rs` and `src/rust/tests/dopri45_integration.rs`, every call to `run_for_api(&cfg, &data, false)` needs a fourth argument `None`. There are 12 call sites total. Update them all:

```rust
// Before:
run_for_api(&cfg, &data, false)
// After:
run_for_api(&cfg, &data, false, None)
```

In `e2e.rs`: lines 195, 198, 219, 240, 243, 265, 289, 296.
In `dopri45_integration.rs`: lines 26, 50, 55, 109.

- [ ] **Step 6: Verify Rust builds and all tests pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test && cargo build --release`

Expected: all tests pass, release build succeeds.

- [ ] **Step 7: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/simulation/runner.rs src/rust/tests/e2e.rs src/rust/tests/dopri45_integration.rs
git commit -m "feat: add wall-clock timeout to simulation loop

Threads an optional Duration through run_for_api -> run_core ->
run_single. When elapsed wall time exceeds the timeout, the sim
terminates with TermReason::Timeout. Prevents a single pathological
sim from blocking an entire Rayon batch during GA training."
```

---

### Task 4: Wall-clock timeout — PyO3 layer (B2, part 2)

**Files:**
- Modify: `src/rust/aerocapture-py/src/lib.rs` — `run`, `run_mc`, `run_batch` signatures
- Modify: `src/rust/aerocapture-py/src/batch.rs` — `run_batch` signature + `run_for_api` call

- [ ] **Step 1: Add `sim_timeout_secs` to PyO3 `run()` function**

In `src/rust/aerocapture-py/src/lib.rs`, add the import at top:
```rust
use std::time::Duration;
```

Update the `run()` function (line 60-78):

```rust
#[pyfunction]
#[pyo3(signature = (toml_path, overrides=None, sim_timeout_secs=None))]
fn run(
    toml_path: &str,
    overrides: Option<&Bound<'_, PyDict>>,
    sim_timeout_secs: Option<f64>,
) -> PyResult<SimResult> {
    let overrides = extract_overrides(overrides)?;
    let wall_timeout = sim_timeout_secs.map(Duration::from_secs_f64);

    let (sim_input, sim_data) =
        config::load_and_override(std::path::Path::new(toml_path), &overrides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let outputs =
        aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data, false, wall_timeout)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Simulation error: {}", e))
            })?;

    let output = outputs.into_iter().next().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Simulation produced no results")
    })?;

    Ok(SimResult::from_output(output))
}
```

- [ ] **Step 2: Add `sim_timeout_secs` to PyO3 `run_mc()` function**

Update the `run_mc()` function (line 96-116):

```rust
#[pyfunction]
#[pyo3(signature = (toml_path, overrides=None, include_trajectories=false, sim_timeout_secs=None))]
fn run_mc(
    toml_path: &str,
    overrides: Option<&Bound<'_, PyDict>>,
    include_trajectories: bool,
    sim_timeout_secs: Option<f64>,
) -> PyResult<BatchResults> {
    let overrides = extract_overrides(overrides)?;
    let wall_timeout = sim_timeout_secs.map(Duration::from_secs_f64);

    let (sim_input, sim_data) =
        config::load_and_override(std::path::Path::new(toml_path), &overrides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let outputs = aerocapture::simulation::runner::run_for_api(
        &sim_input,
        &sim_data,
        include_trajectories,
        wall_timeout,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Simulation error: {}", e))
    })?;

    Ok(BatchResults::from_outputs(outputs, include_trajectories))
}
```

- [ ] **Step 3: Add `sim_timeout_secs` to PyO3 `run_batch()` and thread to batch.rs**

Update the `run_batch()` function (line 131-161):

```rust
#[pyfunction]
#[pyo3(signature = (toml_path, overrides_list, n_threads=None, include_trajectories=false, sim_timeout_secs=None))]
fn run_batch(
    toml_path: &str,
    overrides_list: &Bound<'_, PyList>,
    n_threads: Option<usize>,
    include_trajectories: bool,
    sim_timeout_secs: Option<f64>,
) -> PyResult<BatchResults> {
    let n_threads = n_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    let mut overrides_vec = Vec::new();
    for item in overrides_list.iter() {
        let dict: &Bound<'_, PyDict> = item.cast()?;
        overrides_vec.push(extract_overrides(Some(dict))?);
    }

    let wall_timeout = sim_timeout_secs.map(Duration::from_secs_f64);

    let outputs = batch::run_batch(
        std::path::Path::new(toml_path),
        overrides_vec,
        n_threads,
        include_trajectories,
        wall_timeout,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(BatchResults::from_outputs(outputs, include_trajectories))
}
```

- [ ] **Step 4: Update batch.rs to accept and pass through wall_timeout**

In `src/rust/aerocapture-py/src/batch.rs`, add the import at top:
```rust
use std::time::Duration;
```

Update the `run_batch` function signature (line 27):
```rust
pub fn run_batch(
    toml_path: &Path,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    n_threads: usize,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<RunOutput>, String> {
```

Update the `run_for_api` call inside the parallel iterator (line 77):
```rust
                let outputs = aerocapture::simulation::runner::run_for_api(
                    &sim_input,
                    &sim_data,
                    include_trajectories,
                    wall_timeout,
                )
                .map_err(|e: SimError| format!("Simulation error: {}", e))?;
```

- [ ] **Step 5: Build Rust workspace and run all Rust tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test && cargo build --release`

Expected: all tests pass, release build succeeds.

- [ ] **Step 6: Rebuild PyO3 bindings and smoke test**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`

Then test: `uv run python -c "import aerocapture_rs; print(aerocapture_rs.__version__)"`

Expected: prints `0.1.0`.

- [ ] **Step 7: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/aerocapture-py/src/lib.rs src/rust/aerocapture-py/src/batch.rs
git commit -m "feat: expose sim_timeout_secs in PyO3 run/run_mc/run_batch

Optional wall-clock timeout per simulation, threaded from Python
through to the Rust sim loop. Default None (backward compatible).
Training can pass e.g. sim_timeout_secs=30.0 to prevent one
pathological sim from blocking an entire Rayon batch."
```

---

### Task 5: Log-scale for pred_guid.pdyn_threshold (B3)

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py:55`

- [ ] **Step 1: Add log_scale=True to pdyn_threshold**

In `src/python/aerocapture/training/param_spaces.py`, line 55, change:

```python
        ParamSpec("pdyn_threshold", 10.0, 500.0, 100.0),
```

to:

```python
        ParamSpec("pdyn_threshold", 10.0, 500.0, 100.0, log_scale=True),
```

- [ ] **Step 2: Run Python tests to verify no regressions**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py
git commit -m "feat: log-scale encoding for pred_guid.pdyn_threshold

Range [10, 500] spans 50x — log-scale encoding gives the GA uniform
resolution across the full range instead of over-sampling high values."
```

---

### Task 6: Fix FTC training config + param bounds (B5)

**Files:**
- Modify: `configs/training/msr_aller_ftc_train.toml`
- Modify: `src/python/aerocapture/training/param_spaces.py:66-74`

- [ ] **Step 1: Add full [guidance.ftc] section to training TOML**

Replace the contents of `configs/training/msr_aller_ftc_train.toml` with:

```toml
# MSR outbound — FTC GA training, 10 MC sims per evaluation
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "ftc"

[guidance.ftc]
capture_damping = 0.7
capture_frequency = 0.072
capture_pdyn_margin = 1.75
altitude_damping = 0.7
altitude_frequency = 0.08
exit_velocity_threshold = 4400.0
exit_pdyn_margin = 1.75
exit_altitude_threshold = 60.0
exit_radial_vel_gain = 10.0
exit_apoapsis_threshold = 100.0
corridor_slope = 13080.458
corridor_intercept = 0.0
max_reversals = 5
security_capture = 1
security_exit = 3
density_filter_gain = 0.8
longi_activation = 1000.0
longi_inhibition = -1000.0
lateral_activation = 1.311
lateral_inhibition = 1000.0
pdyn_min = 0.0
pdyn_table = [
    { altitude =  0.0000000000, a = -0.1645497562, b = 1.4897963360 },
    { altitude = 45.6282285400, a = -0.1965988262, b = 1.3408173570 },
    { altitude = 46.3171209300, a = -0.1412271905, b = 1.2053819220 },
    { altitude = 47.1889298400, a = -0.1527424374, b = 1.0822587990 },
    { altitude = 47.9217328100, a = -0.1078032389, b = 0.9703286871 },
    { altitude = 48.8656251100, a = -0.1073334457, b = 0.8685740400 },
    { altitude = 49.7274648100, a = -0.1141791608, b = 0.7760698154 },
    { altitude = 50.4639805600, a = -0.0775047379, b = 0.6919750657 },
    { altitude = 51.4503689300, a = -0.0835505551, b = 0.6155252933 },
    { altitude = 52.2821981500, a = -0.0628220168, b = 0.5460255002 },
    { altitude = 53.2879224700, a = -0.0631744053, b = 0.4828438701 },
    { altitude = 54.1971173500, a = -0.0440526032, b = 0.4254060246 },
    { altitude = 55.3824326100, a = -0.0484318959, b = 0.3731898013 },
    { altitude = 56.3625572400, a = -0.0314772782, b = 0.3257205075 },
    { altitude = 57.7335113400, a = -0.0375532950, b = 0.2825666040 },
    { altitude = 58.7781818900, a = -0.0245224347, b = 0.2433357827 },
    { altitude = 60.2325392300, a = -0.0268146977, b = 0.2076713996 },
    { altitude = 61.4416584500, a = -0.0179607869, b = 0.1752492332 },
    { altitude = 63.0827166600, a = -0.0149100858, b = 0.1457745365 },
    { altitude = 64.8798343400, a = -0.0132187541, b = 0.1189793576 },
    { altitude = 66.7226141800, a = -0.0090497830, b = 0.0946201041 },
    { altitude = 69.1696094000, a = -0.0073162743, b = 0.0724753282 },
    { altitude = 71.9212304600, a = -0.0046864259, b = 0.0523437137 },
    { altitude = 75.8264384400, a = -0.0030096889, b = 0.0340422460 },
    { altitude = 81.3544842100, a = -0.0010156255, b = 0.0174045481 },
    { altitude = 96.2469613900, a = -0.0000010000, b = 0.0022793682 },
]

[data]
results_suffix = ".train_ftc"
```

- [ ] **Step 2: Add altitude_damping and altitude_frequency to FTC param space**

In `src/python/aerocapture/training/param_spaces.py`, change the FTC entry (lines 66-74) from:

```python
    "ftc": [
        ParamSpec("capture_damping", 0.3, 1.5, 0.7),
        ParamSpec("capture_frequency", 0.01, 0.2, 0.072),
        ParamSpec("density_filter_gain", 0.3, 1.0, 0.8),
        ParamSpec("exit_velocity_threshold", -100.0, 0.0, -20.0),
        ParamSpec("exit_radial_vel_gain", -0.1, 0.0, -0.02),
        ParamSpec("capture_pdyn_margin", 1.0, 3.0, 1.75),
        *_LATERAL_PARAMS,
    ],
```

to:

```python
    "ftc": [
        ParamSpec("capture_damping", 0.3, 1.5, 0.7),
        ParamSpec("capture_frequency", 0.01, 0.2, 0.072),
        ParamSpec("altitude_damping", 0.3, 1.5, 0.7),
        ParamSpec("altitude_frequency", 0.01, 0.2, 0.08),
        ParamSpec("density_filter_gain", 0.3, 1.0, 0.8),
        ParamSpec("exit_velocity_threshold", -100.0, 0.0, -20.0),
        ParamSpec("exit_radial_vel_gain", -0.1, 0.0, -0.02),
        ParamSpec("capture_pdyn_margin", 1.0, 3.0, 1.75),
        *_LATERAL_PARAMS,
    ],
```

- [ ] **Step 3: Verify FTC config loads without error**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml && uv run python -c "import aerocapture_rs; r = aerocapture_rs.run('configs/training/msr_aller_ftc_train.toml'); print(f'captured={r.captured}, dv_total={r.final_record[41]:.1f} m/s')"`

Expected: prints a result without error (captured status and DV value will vary).

- [ ] **Step 4: Run Python tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/training/msr_aller_ftc_train.toml src/python/aerocapture/training/param_spaces.py
git commit -m "fix: FTC training config and param space

Add full [guidance.ftc] section to training TOML (was completely
missing, causing parse error on altitude_damping). Add altitude_damping
and altitude_frequency to FTC param space for GA optimization."
```

---

### Task 7: DV chart improvements (A1 + A2)

**Files:**
- Modify: `src/python/aerocapture/training/charts.py:57-58, 167-169, 860-935`

- [ ] **Step 1: Update DV_FLOOR and remove hardcoded tick values**

In `src/python/aerocapture/training/charts.py`:

Change line 58 from:
```python
DV_FLOOR: float = 0.1
```
to:
```python
DV_FLOOR: float = 1.0
```

Remove lines 863:
```python
_LOG10_TICK_VALUES = [0.1, 1, 10, 100, 1000, 5000]
```

- [ ] **Step 2: Add `_log10_ticks` helper function**

Insert the following function right after `_clip_dv` (after line 169):

```python
def _log10_ticks(
    values: npt.NDArray[np.float64], floor: float = 1.0
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Compute snug power-of-10 tick positions and labels for log10-scaled data.

    Returns tick positions (in log10 space) and formatted string labels.
    Floor clamps the minimum to at least ``floor`` (default 1.0 m/s).
    """
    clipped = np.abs(values)
    clipped = clipped[clipped >= floor]
    if len(clipped) == 0:
        clipped = np.array([floor])
    lo = max(0, int(np.floor(np.log10(np.min(clipped)))))
    hi = int(np.ceil(np.log10(np.max(clipped))))
    if hi <= lo:
        hi = lo + 1
    tick_decades = np.arange(lo, hi + 1, dtype=float)
    tick_labels = [f"{10**d:g}" for d in tick_decades]
    return tick_decades, tick_labels
```

- [ ] **Step 3: Update `chart_dv_distribution` to use `_log10_ticks`**

Replace the `chart_dv_distribution` function (lines 866-903) with:

```python
def chart_dv_distribution(final_records: npt.NDArray[np.float64], output: Path) -> None:
    """Panel 15: Total DV histogram (log10 x) with CDF overlay and percentile markers."""
    dv = _clip_dv(final_records[:, _FR_DV_TOTAL])
    log_dv = np.log10(dv)

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    # Histogram
    ax1.hist(log_dv, bins=30, color=COLOR_CAPTURE, alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Total \u0394V (m/s)")
    ax1.set_ylabel("Count")

    # Auto-decade tick labels
    tick_pos, tick_labels = _log10_ticks(dv)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_labels)

    # CDF on secondary y-axis
    ax2 = ax1.twinx()
    sorted_log = np.sort(log_dv)
    cdf = np.arange(1, len(sorted_log) + 1) / len(sorted_log)
    ax2.plot(sorted_log, cdf, color=COLOR_MEAN, linewidth=1.5, label="CDF")
    ax2.set_ylabel("CDF")
    ax2.set_ylim(0, 1.05)

    # Percentile markers
    for pct, ls in [(5, ":"), (50, "--"), (95, "-.")]:
        val = float(np.percentile(log_dv, pct))
        ax1.axvline(val, color=COLOR_WORST, linestyle=ls, linewidth=0.8, label=f"p{pct}")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="x-small")

    ax1.set_title("Total \u0394V Distribution")
    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)
```

- [ ] **Step 4: Refactor `chart_dv_individual_burns` to 3-row subplot**

Replace the `chart_dv_individual_burns` function (lines 910-935) with:

```python
def chart_dv_individual_burns(final_records: npt.NDArray[np.float64], output: Path) -> None:
    """Panel 16: Stacked histograms for |dv1|, |dv2|, |dv3| on log10 x-axis."""
    burns = [
        (np.abs(final_records[:, _FR_DV1]), "#1f77b4", "|DV1| (periapsis)"),
        (np.abs(final_records[:, _FR_DV2]), "#ff7f0e", "|DV2| (apoapsis)"),
        (np.abs(final_records[:, _FR_DV3]), "#2ca02c", "|DV3| (inclination)"),
    ]

    # Shared tick range from all burns combined
    all_raw = np.concatenate([b[0] for b in burns])
    tick_pos, tick_labels = _log10_ticks(all_raw)

    fig, axes = plt.subplots(3, 1, figsize=(FULL_WIDTH[0], FULL_WIDTH[1] * 1.8), dpi=DPI, sharex=True)

    for ax, (raw, color, label) in zip(axes, burns):
        log_vals = np.log10(_clip_dv(raw))
        ax.hist(log_vals, bins=25, color=color, alpha=0.7, edgecolor="white", label=label)
        ax.set_ylabel("Count")
        ax.legend(fontsize="x-small", loc="upper right")

    # Only bottom axes gets x-axis labels
    axes[-1].set_xlabel("\u0394V (m/s)")
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_labels)
    axes[0].set_title("Individual Burn \u0394V")

    fig.tight_layout()
    sns.despine(fig=fig)
    _save_svg(fig, output)
```

- [ ] **Step 5: Run chart tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_charts.py -x -v`

Expected: all chart tests pass, including `test_dv_distribution` and `test_dv_individual_burns`.

- [ ] **Step 6: Run full test suite and lint**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run ruff check src/python/ && uv run ruff format --check src/python/ && uv run pytest tests/ -x -q`

Expected: all pass with no lint or formatting issues.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/charts.py
git commit -m "feat: split DV burn charts into 3-row subplot, unify xticklabels

Refactor chart_dv_individual_burns from overlaid histograms into a
3-row vertical subplot (dv1/dv2/dv3) with shared x-axis. Extract
_log10_ticks helper for auto-decade tick labels with 1 m/s floor.
Apply to both chart_dv_distribution and chart_dv_individual_burns,
replacing the hardcoded _LOG10_TICK_VALUES. Update DV_FLOOR 0.1 → 1.0."
```

---

### Task 8: Smart commit

Invoke the `smart-commit` skill, taking the whole `feature/Fix_training` branch into account.
