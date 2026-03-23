# Cap DV at 5000 m/s + Log Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clip DV values at 5000 m/s with a 0.1 m/s floor and use log-scale axes in the final evaluation report, making captured trajectory distributions readable alongside virtual DV penalties.

**Architecture:** All changes are in `final_report.py` — add two module constants (`DV_CAP`, `DV_FLOOR`), clip DV arrays before plotting/table computation, and set log scale on DV axes. Tests validate clipping behavior, log scale presence, and edge cases.

**Tech Stack:** Python (numpy, plotly), pytest

**Spec:** `docs/superpowers/specs/2026-03-23-dv-plot-cap-log-scale-design.md`

---

### Task 1: Add constants and write tests for DV clipping + log scale

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py:41-43` (add constants after `_COL_BANK_CONSUMPTION`)
- Modify: `tests/test_final_report.py` (add new test class)

- [ ] **Step 1: Write tests for DV clipping and log scale**

Add a new test class at the end of `tests/test_final_report.py`:

```python
class TestDvClippingAndLogScale:
    """Tests for DV cap at 5000 m/s + log scale (spec: 2026-03-23)."""

    def test_dv_clipping_includes_all_trajectories(self, tmp_path: Path) -> None:
        """DV histograms should include non-captured trajectories, clipped to DV_CAP."""
        from aerocapture.training.final_report import DV_CAP, DV_FLOOR, generate_final_report

        # 80 captured (DV ~ 80 m/s) + 20 hyperbolic (DV = 15000 m/s)
        arr = _make_mixed_array(n_captured=80, n_hyper=20)
        arr[80:, 41] = 15000.0  # hyperbolic: virtual DV >> DV_CAP
        eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=None)

        output = tmp_path / "report.html"
        generate_final_report(eval_data, "ftc", 50.0, output)

        # Verify clipping directly: the clipped array should have values at DV_CAP
        clipped = np.clip(arr[:, 41], DV_FLOOR, DV_CAP)
        assert clipped.max() == pytest.approx(DV_CAP)  # hyperbolic rows clipped to cap
        assert (clipped == DV_CAP).sum() == 20  # exactly the 20 hyperbolic rows
        assert clipped.min() >= DV_FLOOR  # no zeros

    def test_dv_floor_prevents_zero(self, tmp_path: Path) -> None:
        """DV values of 0.0 should be floored to DV_FLOOR for log scale safety."""
        from aerocapture.training.final_report import DV_FLOOR, generate_final_report

        arr = _make_captured_array(50)
        arr[:, 37] = 0.0  # dv1 = 0 (perfect periapsis)
        arr[:, 38] = 0.0  # dv2 = 0
        arr[:, 39] = 0.0  # dv3 = 0
        arr[:, 41] = 0.0  # total = 0
        eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=None)

        output = tmp_path / "report.html"
        # Should not raise — log(0) would break Plotly log-scale rendering
        generate_final_report(eval_data, "eqglide", 50.0, output)
        assert output.exists()
        assert output.stat().st_size > 1000

        # Verify floor is applied
        clipped = np.clip(arr[:, 41], DV_FLOOR, 5000.0)
        assert clipped.min() == pytest.approx(DV_FLOOR)

    def test_dv_axes_use_log_scale(self, tmp_path: Path) -> None:
        """DV distribution axes should be log-scaled."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(100)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "eqglide", 50.0, output)
        content = output.read_text()
        # Plotly encodes axis type in the layout JSON — check for log type
        assert '"type":"log"' in content.replace(" ", "")

    def test_all_hyperbolic_with_clipping_does_not_crash(self, tmp_path: Path) -> None:
        """100% hyperbolic trajectories should still produce a valid report."""
        from aerocapture.training.final_report import generate_final_report

        arr = _make_all_hyperbolic(50)
        arr[:, 41] = 15000.0  # all virtual DV
        eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=None)

        output = tmp_path / "report.html"
        generate_final_report(eval_data, "ftc", 50.0, output)
        assert output.exists()

    def test_dispersion_grid_uses_clipped_dv(self, tmp_path: Path) -> None:
        """Dispersion grid y-axis (DV) should be clipped at DV_CAP."""
        from aerocapture.training.final_report import DV_CAP, DV_FLOOR, generate_final_report

        arr = _make_captured_array(100)
        arr[0, 41] = 8000.0  # one outlier captured trajectory
        disp = _make_dispersions(100)
        eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=disp)

        output = tmp_path / "report.html"
        generate_final_report(eval_data, "eqglide", 50.0, output)

        # Verify clipping directly: outlier should be capped
        clipped = np.clip(arr[:, 41], DV_FLOOR, DV_CAP)
        assert clipped[0] == pytest.approx(DV_CAP)  # 8000 -> 5000
        assert clipped.max() == pytest.approx(DV_CAP)
```

- [ ] **Step 2: Add module constants to `final_report.py`**

In `src/python/aerocapture/training/final_report.py`, after line 43 (`_PERCENTILES = [5, 25, 50, 75, 95]`), add:

```python
# DV clipping for plot readability (virtual DV penalties reach 10k-20k m/s)
DV_CAP = 5000.0    # m/s — upper clip for DV values in plots and table
DV_FLOOR = 0.1     # m/s — lower clip to avoid log(0) on log-scale axes
```

- [ ] **Step 3: Run new tests to verify they fail**

Run: `uv run pytest tests/test_final_report.py::TestDvClippingAndLogScale -v`
Expected: FAIL — `DV_CAP` and `DV_FLOOR` import works, but clipping/log scale not yet applied, so assertions fail.

- [ ] **Step 4: Commit test + constants**

```bash
git add tests/test_final_report.py src/python/aerocapture/training/final_report.py
git commit -m "test: add tests for DV clipping + log scale; add DV_CAP/DV_FLOOR constants"
```

---

### Task 2: Implement DV clipping in histograms (Changes 1 & 2)

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py:240-297`

- [ ] **Step 1: Move DV arrays outside captured block and clip them**

In `generate_final_report()`, before the `if n_captured == 0:` block (line 240), add clipped DV arrays computed from ALL trajectories:

```python
    # DV arrays for all trajectories, clipped for plot readability
    dv_total = np.clip(final_array[:, _COL_DV_TOTAL], DV_FLOOR, DV_CAP)
    dv1 = np.clip(final_array[:, _COL_DV1], DV_FLOOR, DV_CAP)
    dv2 = np.clip(final_array[:, _COL_DV2], DV_FLOOR, DV_CAP)
    dv3 = np.clip(final_array[:, _COL_DV3], DV_FLOOR, DV_CAP)
```

Then in the `else` block (line 254), **remove** the old captured-only DV lines:

```python
    # DELETE these 4 lines:
    #     dv_total = cap[:, _COL_DV_TOTAL]
    #     dv1 = cap[:, _COL_DV1]
    #     dv2 = cap[:, _COL_DV2]
    #     dv3 = cap[:, _COL_DV3]
```

The DV histograms (lines 265-272) already reference `dv_total`, `dv1`, `dv2`, `dv3` — they will now use the clipped all-trajectory arrays.

Also move the DV histogram + individual burns plot calls **outside** the `if n_captured == 0` / `else` block, so they render for all trajectories even at 0% capture rate. Specifically, move lines 264-272 (the `_add_hist_cdf` call and the three `go.Histogram` traces + barmode + xaxis title) to just before `fig.update_xaxes(title_text="Orbital Error (km)", row=3, col=2)` (line 296), after the `if/else` block closes.

**Important:** Update the `n_captured == 0` annotation loop (line 242) to skip DV cells — change:
```python
for row, col in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]:
```
to:
```python
for row, col in [(2, 1), (2, 2), (3, 1), (3, 2)]:
```
Row 1 (DV histograms) will now always render data for all trajectories, so the "No captured trajectories" annotation should only appear on orbital error panels (rows 2-3).

- [ ] **Step 2: Add log scale to DV axes**

After the DV histogram calls, add:

```python
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_xaxes(type="log", row=1, col=2)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_final_report.py::TestDvClippingAndLogScale -v`
Expected: `test_dv_values_clipped_at_cap`, `test_dv_floor_prevents_zero`, `test_dv_axes_use_log_scale`, `test_all_hyperbolic_with_clipping_does_not_crash` should PASS.

- [ ] **Step 4: Run full test suite to check no regressions**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/final_report.py
git commit -m "feat: clip DV at 5000 m/s + log scale on DV histograms"
```

---

### Task 3: Clip DV in scatter plot and performance table (Changes 3 & 4)

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py:283-297` (scatter)
- Modify: `src/python/aerocapture/training/final_report.py:474` (table)

- [ ] **Step 1: Clip DV-vs-error scatter y-values and add log scale**

At line 288, change:
```python
y=cap[:, _COL_DV_TOTAL],
```
to:
```python
y=np.clip(cap[:, _COL_DV_TOTAL], DV_FLOOR, DV_CAP),
```

After line 297 (`fig.update_yaxes(title_text="Delta-V (m/s)", row=3, col=2)`), add:
```python
    fig.update_yaxes(type="log", row=3, col=2)
```

- [ ] **Step 2: Clip DV in performance table**

At line 474, change:
```python
"Correction cost \u0394V (m/s)": cap[:, _COL_DV_TOTAL],
```
to:
```python
"Correction cost \u0394V (m/s)": np.clip(final_array[:, _COL_DV_TOTAL], DV_FLOOR, DV_CAP),
```

Note: this changes from `cap` (captured-only) to `final_array` (all trajectories), both clipped.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/final_report.py
git commit -m "feat: clip DV in scatter plot and performance table"
```

---

### Task 4: Clip DV in dispersion grid (Change 5)

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py:808`

- [ ] **Step 1: Clip `cap_dv` in `_build_dispersion_grid()`**

At line 808, change:
```python
cap_dv = final_array[captured, _COL_DV_TOTAL]
```
to:
```python
cap_dv = np.clip(final_array[captured, _COL_DV_TOTAL], DV_FLOOR, DV_CAP)
```

- [ ] **Step 2: Run dispersion-specific test**

Run: `uv run pytest tests/test_final_report.py::TestDvClippingAndLogScale::test_dispersion_grid_uses_clipped_dv -v`
Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/final_report.py
git commit -m "feat: clip DV in dispersion correlation grid"
```

---

### Task 5: Lint, type-check, and final verification

**Files:**
- All modified files

- [ ] **Step 1: Run linter**

Run: `./lint_code.sh`
Expected: No errors.

- [ ] **Step 2: Run full Python test suite**

Run: `uv run pytest tests/ -v`
Expected: All ~276+ tests pass.

- [ ] **Step 3: Commit any lint fixes**

If lint produced fixes:
```bash
git add -u
git commit -m "style: lint fixes for DV clipping changes"
```

---

### Task 6: Update TODO.md and smart-commit

- [ ] **Step 1: Remove the completed TODO item**

In `TODO.md`, remove the line:
```
- [ ] 1e30 for Dv is too much in the plots, limit the correction costs to 5000 m/s and maybe use a log scale
```

- [ ] **Step 2: Invoke the `smart-commit` skill**

Use `/smart-commit` to sync docs and commit the whole branch.
