# Single-Algorithm TUI Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the single-algorithm training TUI into a colored header/Optimization/Validation/footer dashboard, fix its four rendering bugs, and additively extend `compute_eval_summary` with cost/DV min·max and DV1/DV2/DV3 burn stats.

**Architecture:** `display.py`'s `_build_panel` is replaced by four independently-testable builders composed into a Rich `Group`; alignment moves from hand-padded strings to `Table.grid`. `_format_validation_summary` is retired for a data-shaping `_validation_summary_rows` helper consumed by both the new single-algo Validation panel and the existing islands detail panels. `compute_eval_summary` (report.py) gains additive fields only.

**Tech Stack:** Python 3.14, Rich 15.0 (Panel/Table.grid/Columns/Group/Text/Live), numpy, pytest. No Rust changes.

**Spec:** `docs/superpowers/specs/2026-06-10-tui-dashboard-design.md`
**Branch:** `feature/tui-dashboard` (already created; spec committed at `4835634`).

**Conventions for every task:**
- Run all commands from the repo root: `/Users/govit/Git/Govit/Aerocapture/.claude/worktrees/strange-perlman-c3973f`. Use `uv run ...`.
- mypy strict (`disallow_untyped_defs`, tests included); ruff E,F,I,W,UP,B,SIM line-length 160. Rich is typed — annotations must be real, not `Any`-washed.
- Stage only the task's files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Per-task lint gate: `uv run ruff check <files> && uv run ruff format --check <files> && uv run mypy <files>`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/python/aerocapture/training/report.py` | Modify | `compute_eval_summary` additive extension (cost min/max, dv min/max, dv1/2/3 blocks) |
| `tests/test_eval_summary.py` | Modify | Tests for the new summary fields |
| `src/python/aerocapture/training/display.py` | Modify | Sparkline fix, `_cost_histogram`, `_rate_and_eta`, `_validation_summary_rows` (+ islands swap), dashboard builders, `algorithm` plumbing |
| `tests/test_training_display.py` | Rewrite | Unit tests per builder + render-fragment snapshot |
| `src/python/aerocapture/training/train.py` | Modify | Two one-liners: `algorithm=` arg, single-algo `set_start_gen` |

---

### Task 1: `compute_eval_summary` extension

**Files:**
- Modify: `src/python/aerocapture/training/report.py` (function at ~line 188)
- Test: `tests/test_eval_summary.py` (append)

- [ ] **Step 1.1: Write the failing tests**

Read `tests/test_eval_summary.py` first to see how existing tests build synthetic final-records matrices (reuse their fixture/helper if one exists; otherwise the code below builds rows via the `charts._FR_*` index constants). Append:

```python
class TestEvalSummaryDvComponents:
    def _records(self) -> np.ndarray:
        from aerocapture.training import charts

        n = 6
        rec = np.zeros((n, 52), dtype=np.float64)
        rec[:, charts._FR_IFINAL] = 3.0  # captured
        rec[:, charts._FR_ECC] = 0.5
        rec[:, charts._FR_DV_TOTAL] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        rec[:, charts._FR_DV1] = [-80.0, 150.0, 250.0, 330.0, 410.0, 480.0]  # negative: abs() convention
        rec[:, charts._FR_DV2] = [10.0, 30.0, 30.0, 40.0, 60.0, 80.0]
        rec[:, charts._FR_DV3] = [10.0, 20.0, 20.0, 30.0, 30.0, 40.0]
        rec[:, charts._FR_APO_ERR] = 50.0
        rec[:, charts._FR_PERI_ERR] = 5.0
        rec[:, charts._FR_INCL_ERR] = 0.1
        rec[:, charts._FR_MAX_HEAT_FLUX] = 150.0
        rec[:, charts._FR_MAX_G_LOAD] = 8.0
        rec[:, charts._FR_INTEGRATED_FLUX] = 10.0
        return rec

    def test_cost_min_max_present(self) -> None:
        from aerocapture.training.report import compute_eval_summary

        s = compute_eval_summary(self._records(), n_sims=6)
        assert s["cost"]["min"] <= s["cost"]["p50"] <= s["cost"]["p95"] <= s["cost"]["max"]

    def test_dv_min_max_present_and_clipped_consistent(self) -> None:
        from aerocapture.training.report import compute_eval_summary

        s = compute_eval_summary(self._records(), n_sims=6)
        dv = s["captured"]["dv"]
        assert dv["min"] == 100.0
        assert dv["max"] == 600.0
        assert set(dv) == {"min", "p50", "p95", "mean", "max"}

    def test_dv_components_abs_and_stats(self) -> None:
        from aerocapture.training.report import compute_eval_summary

        s = compute_eval_summary(self._records(), n_sims=6)
        dv1 = s["captured"]["dv1"]
        assert dv1["min"] == 80.0  # |-80| -> abs convention, matching chart panel 16
        assert dv1["max"] == 480.0
        dv2 = s["captured"]["dv2"]
        assert dv2["p50"] == 35.0
        for comp in ("dv1", "dv2", "dv3"):
            assert set(s["captured"][comp]) == {"min", "p50", "p95", "mean", "max"}

    def test_captured_none_when_no_captures(self) -> None:
        from aerocapture.training import charts
        from aerocapture.training.report import compute_eval_summary

        rec = self._records()
        rec[:, charts._FR_IFINAL] = 4.0  # nothing captured
        s = compute_eval_summary(rec, n_sims=6)
        assert s["captured"] is None
        assert "min" in s["cost"] and "max" in s["cost"]

    def test_existing_keys_unchanged(self) -> None:
        from aerocapture.training.report import compute_eval_summary

        s = compute_eval_summary(self._records(), n_sims=6)
        assert set(s["cost"]) == {"min", "p50", "p95", "rms", "max"}
        assert {"dv", "apoapsis", "periapsis", "inclination", "dv1", "dv2", "dv3"} <= set(s["captured"])
        for key in ("apoapsis", "periapsis", "inclination"):
            assert set(s["captured"][key]) == {"p50", "p95", "mean"}  # untouched blocks keep their shape
```

(Constant names verified: `charts._FR_ECC = 9`, `charts._FR_IFINAL = 31`; `is_captured` is `(ifinal == 3) & (ecc < 1.0)` at charts.py:105-107.)

- [ ] **Step 1.2: Run to verify they fail**

Run: `uv run pytest tests/test_eval_summary.py -q`
Expected: new tests fail with `KeyError: 'min'` / `KeyError: 'dv1'`; existing tests pass.

- [ ] **Step 1.3: Implement**

In `compute_eval_summary` (report.py):

(a) Replace the `captured_stats` block:

```python
    captured_stats: dict[str, dict[str, float]] | None = None
    if n_captured > 0:
        dv = np.clip(cap[:, charts._FR_DV_TOTAL], charts.DV_FLOOR, charts.DV_CAP)
        apo = cap[:, charts._FR_APO_ERR]
        peri = cap[:, charts._FR_PERI_ERR]
        incl = cap[:, charts._FR_INCL_ERR]

        def _spread(arr: npt.NDArray[np.float64]) -> dict[str, float]:
            return {
                "min": float(np.min(arr)),
                "p50": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
            }

        captured_stats = {
            "dv": _spread(dv),
            "apoapsis": {"p50": float(np.median(apo)), "p95": float(np.percentile(apo, 95)), "mean": float(np.mean(apo))},
            "periapsis": {"p50": float(np.median(peri)), "p95": float(np.percentile(peri, 95)), "mean": float(np.mean(peri))},
            "inclination": {"p50": float(np.median(incl)), "p95": float(np.percentile(incl, 95)), "mean": float(np.mean(incl))},
            # Terminal-maneuver burns, abs() like chart_burn_dv_histograms (charts.py:1115-1119):
            # DV1 periapsis, DV2 circularization, DV3 inclination.
            "dv1": _spread(np.abs(cap[:, charts._FR_DV1])),
            "dv2": _spread(np.abs(cap[:, charts._FR_DV2])),
            "dv3": _spread(np.abs(cap[:, charts._FR_DV3])),
        }
```

NOTE: `"dv"` gains `min`/`max` via `_spread` — its existing `p50/p95/mean` values are computed identically to before (same clipped array), so old keys are value-stable. The `dv` block previously lacked `min`/`max`; consumers reading known keys are unaffected.

(b) In the return dict, extend the cost block:

```python
        "cost": {
            "min": float(np.min(per_sim_costs)),
            "p50": float(np.median(per_sim_costs)),
            "p95": float(np.percentile(per_sim_costs, 95)),
            "rms": rms_cost,
            "max": float(np.max(per_sim_costs)),
        },
```

(c) Update the docstring's key listing to mention `cost: {min, p50, p95, rms, max}`, `captured.dv: {min, p50, p95, mean, max}`, and `captured.dv1/dv2/dv3: {min, p50, p95, mean, max}` (abs of terminal-maneuver burn components).

- [ ] **Step 1.4: Run tests**

Run: `uv run pytest tests/test_eval_summary.py tests/test_island_model.py tests/test_training_report.py -q`
Expected: all pass (islands + report consumers are additive-safe).

- [ ] **Step 1.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/report.py tests/test_eval_summary.py
git commit -m "feat(train): cost/DV min-max + DV1/DV2/DV3 burn stats in compute_eval_summary

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Display primitives — sparkline fix, histogram, ETA math

**Files:**
- Modify: `src/python/aerocapture/training/display.py`
- Test: `tests/test_training_display.py` (append a new class; the file is rewritten further in Task 4 — append here, Task 4 preserves these tests)

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/test_training_display.py`:

```python
class TestDisplayPrimitives:
    def test_sparkline_flat_series_renders_midline(self) -> None:
        from aerocapture.training.display import _sparkline

        s = _sparkline([5.0, 5.0, 5.0, 5.0])
        assert s == "▄" * 4  # ▄ midline, not blanks

    def test_sparkline_empty_and_varying(self) -> None:
        from aerocapture.training.display import _sparkline

        assert _sparkline([]) == " " * 30
        s = _sparkline([0.0, 1.0])
        assert s[0] == " " and s[-1] == "█"

    def test_cost_histogram_bins_and_caption(self) -> None:
        from aerocapture.training.display import _cost_histogram

        costs = [10.0] * 50 + [100.0] * 10 + [10000.0] * 2
        glyphs, caption = _cost_histogram(costs, bins=8)
        assert len(glyphs) == 8
        assert glyphs[0] == "█"  # densest bin -> full block
        assert "·" in glyphs  # empty bins as middle dot
        assert "log" in caption and "1e+01" in caption

    def test_cost_histogram_nonfinite_counted(self) -> None:
        from aerocapture.training.display import _cost_histogram

        glyphs, caption = _cost_histogram([10.0, 20.0, float("inf"), float("nan")], bins=4)
        assert "∞×2" in caption  # ∞×2

    def test_cost_histogram_all_nonfinite(self) -> None:
        from aerocapture.training.display import _cost_histogram

        glyphs, caption = _cost_histogram([float("inf"), float("nan")], bins=4)
        assert glyphs == ""
        assert "no finite costs" in caption and "∞×2" in caption

    def test_cost_histogram_flat(self) -> None:
        from aerocapture.training.display import _cost_histogram

        glyphs, caption = _cost_histogram([42.0, 42.0, 42.0], bins=8)
        assert glyphs[0] == "█" and set(glyphs[1:]) == {"·"}

    def test_rate_and_eta_resume_aware(self) -> None:
        from aerocapture.training.display import _rate_and_eta

        rate, remaining = _rate_and_eta(gen=740, start_gen=700, n_gen=2000, elapsed=40.0)
        assert rate == 1.0  # (740-700)/40 -- NOT 740/40
        assert remaining == 1260.0

    def test_rate_and_eta_no_progress(self) -> None:
        from aerocapture.training.display import _rate_and_eta

        rate, remaining = _rate_and_eta(gen=700, start_gen=700, n_gen=2000, elapsed=10.0)
        assert rate == 0.0 and remaining == float("inf")

    def test_rate_and_eta_overshoot_resume(self) -> None:
        from aerocapture.training.display import _rate_and_eta

        rate, remaining = _rate_and_eta(gen=2100, start_gen=2000, n_gen=2000, elapsed=50.0)
        assert remaining == 0.0  # gen > n_gen never yields negative ETA
```

- [ ] **Step 2.2: Run to verify they fail**

Run: `uv run pytest tests/test_training_display.py::TestDisplayPrimitives -q`
Expected: flat-sparkline test fails (renders spaces); `ImportError` for `_cost_histogram` / `_rate_and_eta`.

- [ ] **Step 2.3: Implement**

In `display.py`:

(a) Fix `_sparkline` (replace the function body, display.py:24-31):

```python
def _sparkline(values: list[float], width: int = 30) -> str:
    """Render a list of floats as a Unicode sparkline string."""
    if not values:
        return " " * width
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return "▄" * len(vals)  # flat series: midline, not blanks
    span = hi - lo
    return "".join(_SPARK_CHARS[min(int((v - lo) / span * 8), 8)] for v in vals)
```

(b) Add below `_format_cost`:

```python
def _cost_histogram(all_costs: list[float], bins: int = 16) -> tuple[str, str]:
    """Log-binned histogram of a population's costs as (glyphs, dim caption).

    Empty bins render as a middle dot so gaps in the distribution stay
    visible; non-finite entries (inf/NaN sim failures) are counted in the
    caption rather than binned.
    """
    import math  # noqa: PLC0415

    finite = sorted(c for c in all_costs if math.isfinite(c) and c > 0.0)
    n_nonfinite = sum(1 for c in all_costs if not math.isfinite(c))
    inf_suffix = f"  ∞×{n_nonfinite}" if n_nonfinite else ""
    if not finite:
        return "", f"no finite costs{inf_suffix}".strip()
    lo, hi = finite[0], finite[-1]
    if hi <= lo:
        return "█" + "·" * (bins - 1), f"{lo:.0e} log{inf_suffix}"
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    counts = [0] * bins
    for c in finite:
        idx = min(int((math.log10(c) - log_lo) / (log_hi - log_lo) * bins), bins - 1)
        counts[idx] += 1
    peak = max(counts)
    glyphs = "".join("·" if n == 0 else _SPARK_CHARS[max(1, min(int(n / peak * 8), 8))] for n in counts)
    return glyphs, f"{lo:.0e}→{hi:.0e} log{inf_suffix}"


def _rate_and_eta(gen: int, start_gen: int, n_gen: int, elapsed: float) -> tuple[float, float]:
    """(gens/sec, remaining seconds) — resume-aware, mirrors the islands header math."""
    rate = (gen - start_gen) / elapsed if elapsed > 0 and gen > start_gen else 0.0
    remaining_gens = max(n_gen - gen, 0)
    remaining = remaining_gens / rate if rate > 0 else float("inf")
    return rate, remaining
```

(c) In `_update_islands`, replace the inline rate/remaining computation (display.py:267-271) with:

```python
        rate, remaining = _rate_and_eta(gen, self._start_gen, n_gen, elapsed)
```

(keep the surrounding `elapsed` computation and header_text line unchanged).

- [ ] **Step 2.4: Run tests**

Run: `uv run pytest tests/test_training_display.py tests/test_island_model.py -q`
Expected: all pass (islands math is behavior-identical).

- [ ] **Step 2.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/display.py tests/test_training_display.py
git commit -m "feat(train): sparkline flat-fix + cost histogram + shared resume-aware ETA math

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `_validation_summary_rows` + islands swap

**Files:**
- Modify: `src/python/aerocapture/training/display.py` (replace `_format_validation_summary`, display.py:39-88, and its islands consumer at :341)
- Test: `tests/test_training_display.py` (append)

- [ ] **Step 3.1: Write the failing tests**

Append:

```python
def _summary_fixture() -> dict:
    block = {"min": 1.0, "p50": 2.0, "p95": 3.0, "mean": 2.2, "max": 4.0}
    return {
        "n_sims": 1000,
        "n_captured": 968,
        "capture_rate": 0.968,
        "cost": {"min": 38.1, "p50": 112.4, "p95": 387.2, "rms": 181.2, "max": 12000.0},
        "captured": {
            "dv": {"min": 62.0, "p50": 118.2, "p95": 342.0, "mean": 141.7, "max": 980.4},
            "dv1": dict(block),
            "dv2": dict(block),
            "dv3": dict(block),
            "apoapsis": {"p50": 41.2, "p95": 96.0, "mean": 50.0},
            "periapsis": {"p50": 5.0, "p95": 9.0, "mean": 6.0},
            "inclination": {"p50": 0.1, "p95": 0.3, "mean": 0.15},
        },
        "constraints": {
            "heat_flux": {"p50": 142.1, "p95": 188.4, "max": 204.9, "limit": 200.0, "viol_pct": 2.1},
            "g_load": {"p50": 6.0, "p95": 9.8, "max": 11.2, "limit": 15.0, "viol_pct": 0.0},
            "heat_load": {"p50": 9000.0, "p95": 14000.0, "max": 16000.0, "limit": None, "viol_pct": None},
        },
    }


class TestValidationSummaryRows:
    def test_full_summary_rows(self) -> None:
        from aerocapture.training.display import _validation_summary_rows

        rows = _validation_summary_rows(_summary_fixture())
        labels = [r[0] for r in rows]
        assert labels[:3] == ["Cap", "", "Cost"]
        assert "DV" in labels and "DV1" in labels and "DV2" in labels and "DV3" in labels
        assert "Apo" in labels and "Q" in labels and "G" in labels and "HL" in labels
        grid_header = next(r for r in rows if r[0] == "")
        assert grid_header[1] == ["min", "p50", "p95", "max"] and grid_header[2] == "dim"
        q_row = next(r for r in rows if r[0] == "Q")
        assert q_row[2] == "red"  # 2.1% violation
        assert any("2.1% > 200" in c for c in q_row[1])
        g_row = next(r for r in rows if r[0] == "G")
        assert g_row[2] == "dim"  # zero violations
        cost_row = next(r for r in rows if r[0] == "Cost")
        assert cost_row[2] == "yellow"  # max 12000 > 10x p95 387.2 -> outlier hint
        dv1_row = next(r for r in rows if r[0] == "DV1")
        assert dv1_row[1] == ["1.0", "2.0", "3.0", "4.0"]

    def test_captured_none_renders_placeholder(self) -> None:
        from aerocapture.training.display import _validation_summary_rows

        s = _summary_fixture()
        s["captured"] = None
        s["n_captured"] = 0
        rows = _validation_summary_rows(s)
        cap_row = rows[0]
        assert cap_row[0] == "Cap" and cap_row[2] == "red"
        dv_row = next(r for r in rows if r[0] == "DV")
        assert dv_row[1] == ["—"] and dv_row[2] == "dim"

    def test_missing_limits_render_na_style(self) -> None:
        from aerocapture.training.display import _validation_summary_rows

        rows = _validation_summary_rows(_summary_fixture())
        hl_row = next(r for r in rows if r[0] == "HL")
        assert hl_row[2] == "" and not any(">" in c for c in hl_row[1])

    def test_old_formatter_gone(self) -> None:
        import aerocapture.training.display as d

        assert not hasattr(d, "_format_validation_summary")
```

- [ ] **Step 3.2: Run to verify they fail**

Run: `uv run pytest tests/test_training_display.py::TestValidationSummaryRows -q`
Expected: `ImportError: cannot import name '_validation_summary_rows'`.

- [ ] **Step 3.3: Implement**

Replace `_format_validation_summary` (display.py:39-88) entirely with:

```python
def _validation_summary_rows(summary: dict) -> list[tuple[str, list[str], str]]:
    """Shape a `compute_eval_summary` payload into (label, cells, style) rows.

    Consumed by the single-algo Validation panel (as a Table.grid) and the
    islands per-island detail panels (as text lines). Style is a row-level
    Rich style hint: "" | "dim" | "red" | "yellow" | "green".
    """
    nan = float("nan")
    n_sims = summary.get("n_sims", 0)
    n_cap = summary.get("n_captured", 0)
    pct = 100.0 * n_cap / max(n_sims, 1)
    rows: list[tuple[str, list[str], str]] = []
    cap_style = "red" if n_cap == 0 else ("green" if pct >= 95.0 else "")
    rows.append(("Cap", [f"{n_cap}/{n_sims} ({pct:.1f}%)"], cap_style))
    rows.append(("", ["min", "p50", "p95", "max"], "dim"))

    def _grid(block: dict, fmt: str = "{:.1f}") -> list[str]:
        return [fmt.format(block.get(k, nan)) for k in ("min", "p50", "p95", "max")]

    cost = summary.get("cost", {}) or {}
    cost_style = "yellow" if cost.get("max", 0.0) > 10.0 * cost.get("p95", float("inf")) else ""
    rows.append(("Cost", _grid(cost), cost_style))
    cap_block = summary.get("captured")
    if cap_block:
        rows.append(("DV", _grid(cap_block.get("dv", {})), ""))
        for i in (1, 2, 3):
            rows.append((f"DV{i}", _grid(cap_block.get(f"dv{i}", {})), "dim"))
        apo = cap_block.get("apoapsis", {})
        rows.append(("Apo", [f"p50 {apo.get('p50', nan):.1f} · p95 {apo.get('p95', nan):.1f} km"], ""))
    else:
        rows.append(("DV", ["—"], "dim"))
    con = summary.get("constraints", {}) or {}
    for label, key, val_fmt, lim_fmt in (
        ("Q", "heat_flux", "{:.1f}", "{:.0f}"),
        ("G", "g_load", "{:.2f}", "{:.1f}"),
        ("HL", "heat_load", "{:.0f}", "{:.0f}"),
    ):
        block = con.get(key)
        if block is None:
            rows.append((label, ["n/a"], "dim"))
            continue
        cells = [f"max {val_fmt.format(block.get('max', nan))}"]
        style = ""
        if block.get("limit") is not None and block.get("viol_pct") is not None:
            cells.append(f"{block['viol_pct']:.1f}% > {lim_fmt.format(block['limit'])}")
            style = "red" if block["viol_pct"] > 0 else "dim"
        rows.append((label, cells, style))
    return rows


def _rows_to_text(summary: dict) -> Text:
    """Render summary rows as styled text lines (the islands detail panels)."""
    from rich.text import Text  # noqa: PLC0415

    text = Text(f"Validation ({summary.get('n_sims', 0)} sims)\n")
    for label, cells, style in _validation_summary_rows(summary):
        text.append(f"  {label:<5} " + "   ".join(cells) + "\n", style=style or None)
    return text
```

NOTE on the `Text` import/annotation: `_rows_to_text`'s return annotation needs `Text` available at type-time — add `from rich.text import Text` to the `TYPE_CHECKING` block at the top of display.py (the module convention is lazy runtime imports + TYPE_CHECKING names).

In `_update_islands`, replace the detail-panel loop body (display.py:336-342):

```python
        detail_panels: list[Panel] = []
        for name in ("pso", "ga", "de"):
            island_summary: dict | None = (island_records.get(name) or {}).get("val_summary")
            if not island_summary:
                continue
            detail_panels.append(Panel(_rows_to_text(island_summary), title=f"{name.upper()} validation", border_style="green"))
```

`_build_panel`'s consumer (display.py:222) still references the old function — point it at the new helper temporarily so the module stays import-clean until Task 4 replaces `_build_panel` wholesale:

```python
            lines.append(f"-- Validation detail (g{detail_src['generation']}) --")
            lines.extend(str(_rows_to_text(detail_src["validation_summary"])).splitlines())
```

(Yes, that loses per-row styling inside the old monochrome panel — it lives for exactly one task.)

- [ ] **Step 3.4: Run tests**

Run: `uv run pytest tests/test_training_display.py tests/test_island_model.py -q`
Expected: all pass.

- [ ] **Step 3.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/display.py tests/test_training_display.py
git commit -m "refactor(train): _validation_summary_rows shared by islands detail panels

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Dashboard builders

**Files:**
- Modify: `src/python/aerocapture/training/display.py` (replace `_build_panel`, constructor, `create_display`, `update`)
- Rewrite: `tests/test_training_display.py` (replace `TestLiveDisplay`; keep `TestNoopDisplay`, `TestDisplayPrimitives`, `TestValidationSummaryRows`)

- [ ] **Step 4.1: Write the failing tests**

Replace the `TestLiveDisplay` class in `tests/test_training_display.py` with:

```python
def _record(gen: int, **over: object) -> dict:
    rec: dict = {
        "generation": gen,
        "best_cost": 334.5,
        "mean_cost": 480.07,
        "worst_cost": 21300.0,
        "std_cost": 3100.0,
        "capture_rate": 1.0,
        "population_diversity": 0.42,
        "improvement": False,
        "best_params": {"gain": 1.24, "tau": 8.31, "thr": 2.05, "k4": 0.1, "k5": 0.2},
        "all_costs": [300.0 + 10.0 * i for i in range(64)],
        "gen_elapsed_s": 0.83,
        "pool_metrics": {"pool_size": 20, "last_curation_gen": 720},
    }
    rec.update(over)
    return rec


def _val_record(gen: int, promoted: bool, rms: float = 181.2) -> dict:
    return _record(
        gen,
        improvement=promoted,
        validation={"rms_cost": rms, "mean_cost": 142.9, "median_cost": 112.0, "std_cost": 80.0, "p95_cost": 355.7, "worst_cost": 900.0, "capture_rate": 0.968, "n_sims": 1000},
        validation_summary=_summary_fixture(),
    )


def _logger_with(records: list[dict]) -> MagicMock:
    logger = MagicMock()
    logger.buffer = records
    return logger


def _render(renderable: object, width: int = 120) -> str:
    from rich.console import Console

    console = Console(record=True, width=width, force_terminal=True)
    console.print(renderable)
    return console.export_text()


class TestDashboard:
    def _display(self) -> LiveDisplay:
        d = LiveDisplay(scheme="ftc", n_runs=1, n_generations=2000, algorithm="qpso")
        d.set_start_gen(700)
        return d

    def test_dashboard_renders_key_fragments(self) -> None:
        # Two validation records: the PROMOTED one (lower rms) becomes "Best",
        # the later REJECTED one is "Last" -- the Last line only renders when
        # last is not best (same convention as the old panel).
        records = (
            [_record(g) for g in range(701, 733)]
            + [_val_record(704, promoted=True, rms=168.4), _val_record(733, promoted=False, rms=181.2), _record(740)]
        )
        out = _render(self._display()._build_dashboard(_logger_with(records), current_run=0))
        assert "ftc" in out and "qpso" in out
        assert "pop 64" in out
        assert "Optimization" in out and "Validation" in out
        assert "REJECTED" in out
        assert "DV2" in out and "DV3" in out
        assert "2.1% > 200" in out
        assert "pool refresh g720" in out
        assert "gen wall 0.83s" in out
        assert "Run 1/1" not in out  # vestigial fragment removed

    def test_footer_truncates_params(self) -> None:
        out = _render(self._display()._build_footer([_record(740)]))
        assert "(+2 more)" in out  # 5 params -> 3 shown
        assert "k5" not in out

    def test_footer_suppresses_nn_params(self) -> None:
        d = LiveDisplay(scheme="neural_network", n_runs=1, n_generations=100, algorithm="pso")
        rec = _record(5, best_params={f"w_{i}": 0.1 for i in range(515)})
        out = _render(d._build_footer([rec]))
        assert "515 NN params" in out
        assert "w_0" not in out

    def test_validation_panel_placeholder_before_first_validation(self) -> None:
        out = _render(self._display()._build_validation_panel([_record(701)]))
        assert "waiting for first validation" in out

    def test_empty_buffer_renders_waiting(self) -> None:
        out = _render(self._display()._build_dashboard(_logger_with([]), current_run=0))
        assert "Waiting for first generation" in out

    def test_zero_captures_grid_placeholder(self) -> None:
        rec = _val_record(710, promoted=False)
        rec["validation_summary"]["captured"] = None
        rec["validation_summary"]["n_captured"] = 0
        out = _render(self._display()._build_validation_panel([rec]))
        assert "0/1000" in out

    def test_best_and_last_lines_render_together(self) -> None:
        # A single validation record means best == last -> only the Best line
        # shows (pre-existing convention). With two, both lines render.
        records = [_val_record(704, promoted=True, rms=168.4), _val_record(733, promoted=False, rms=181.2)]
        out = _render(self._display()._build_validation_panel(records))
        assert "Best" in out and "1.6840e+02" in out
        assert "REJECTED" in out
        single = _render(self._display()._build_validation_panel([_val_record(733, promoted=True)]))
        assert "REJECTED" not in single and "PROMOTED" not in single  # best==last suppresses the Last line

    def test_update_dispatches_dashboard(self) -> None:
        d = self._display()
        d._live = MagicMock()
        d.update(_logger_with([_record(1)]), current_run=0, island_records=None)
        assert d._live.update.called
```

Also update the imports at the top of the file (`MagicMock` already imported; add nothing else) and keep `TestNoopDisplay`, `TestDisplayPrimitives`, `TestValidationSummaryRows`, and the `create_display` noop test — but the noop test gains the new kwarg:

```python
    def test_create_display_returns_noop_in_non_tty(self) -> None:
        display = create_display(scheme="equilibrium_glide", n_runs=1, n_generations=50, enabled=False, algorithm="ga")
        assert isinstance(display, NoopDisplay)
```

- [ ] **Step 4.2: Run to verify they fail**

Run: `uv run pytest tests/test_training_display.py -q`
Expected: `AttributeError: ... no attribute '_build_dashboard'` / `TypeError: unexpected keyword argument 'algorithm'`.

- [ ] **Step 4.3: Implement**

In `display.py`:

(a) Constructor + factory:

```python
    def __init__(self, scheme: str, n_runs: int, n_generations: int, algorithm: str = "") -> None:
        self._scheme = scheme
        self._algorithm = algorithm
        self._n_runs = n_runs
        self._n_gens = n_generations
        self._live: Live | None = None
        self._start_time: float | None = None
        self._start_gen: int = 0
```

```python
def create_display(scheme: str, n_runs: int, n_generations: int, *, enabled: bool = True, algorithm: str = "") -> LiveDisplay | NoopDisplay:
    """Factory: returns LiveDisplay if enabled and terminal is interactive, else NoopDisplay."""
    if not enabled or not sys.stdout.isatty():
        return NoopDisplay()
    return LiveDisplay(scheme=scheme, n_runs=n_runs, n_generations=n_generations, algorithm=algorithm)
```

(b) Delete `_build_panel` (display.py:147-241) and add the builders. The progress line helper goes at module level next to `_rate_and_eta`:

```python
def _progress_line(gen: int, n_gen: int, width: int = 50) -> Text:
    """Styled ━/╸ progress bar line (blue filled, dim remainder, bold percent)."""
    from rich.text import Text  # noqa: PLC0415

    progress = min(max(gen / n_gen, 0.0), 1.0) if n_gen > 0 else 0.0
    filled = int(progress * width)
    t = Text()
    if filled > 0:
        t.append("━" * (filled - 1) + "╸", style="blue")
    t.append("━" * (width - filled), style="dim")
    t.append(f" {progress:.0%}", style="bold")
    return t
```

Methods on `LiveDisplay`:

```python
    def _build_header(self, gen: int, pop: int | None, elapsed: float) -> Panel:
        from rich.console import Group  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        rate, remaining = _rate_and_eta(gen, self._start_gen, self._n_gens, elapsed)
        t = Text()
        t.append(self._scheme, style="bold")
        if self._algorithm:
            t.append(" · ", style="dim")
            t.append(self._algorithm, style="bold")
        t.append("  │  Gen ", style="dim")
        t.append(str(gen), style="bold")
        t.append(f"/{self._n_gens}", style="dim")
        if pop is not None:
            t.append("  │  pop ", style="dim")
            t.append(str(pop), style="bold")
        if gen > self._start_gen:
            t.append(f"  │  elapsed {_format_duration(elapsed)}  │  {rate:.2f} gen/s  │  ETA ", style="dim")
            t.append(_format_duration(remaining), style="bold")
        return Panel(Group(t, _progress_line(gen, self._n_gens)), border_style="green")

    def _build_optimization_panel(self, buf: list[dict]) -> Panel:
        from rich.console import Group  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        latest = buf[-1]
        grid = Table.grid(padding=(0, 2))
        grid.add_column()
        grid.add_column(justify="right")
        grid.add_column()
        grid.add_row("Best", Text(_format_cost(latest["best_cost"]), style="bold"), Text(_sparkline([r["best_cost"] for r in buf]), style="cyan"))
        grid.add_row("Mean", _format_cost(latest["mean_cost"]), Text(_sparkline([r["mean_cost"] for r in buf]), style="cyan"))
        if latest.get("worst_cost") is not None:
            grid.add_row("Worst", _format_cost(latest["worst_cost"]), Text(f"σ {latest.get('std_cost', float('nan')):.1e}", style="dim"))
        grid.add_row("Capture", Text(f"{latest['capture_rate']:.0%}", style="green"), Text(_sparkline([r["capture_rate"] for r in buf]), style="green"))
        grid.add_row("Divers", f"{latest['population_diversity']:.2f}", Text(_sparkline([r["population_diversity"] for r in buf]), style="magenta"))
        all_costs = latest.get("all_costs")
        if all_costs:
            glyphs, caption = _cost_histogram(all_costs)
            grid.add_row("Pop cost", Text(glyphs, style="blue"), Text(caption, style="dim"))
        bits = []
        if latest.get("gen_elapsed_s") is not None:
            bits.append(f"gen wall {latest['gen_elapsed_s']:.2f}s")
        pool = latest.get("pool_metrics") or {}
        if pool.get("last_curation_gen") is not None:
            bits.append(f"pool refresh g{pool['last_curation_gen']}")
        body = Group(grid, Text(" · ".join(bits), style="dim")) if bits else grid
        return Panel(body, title="Optimization", border_style="cyan")

    def _build_validation_panel(self, buf: list[dict]) -> Panel:
        from rich.console import Group  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        best_val_r, last_val_r = self._scan_validation_records(buf)
        if last_val_r is None and best_val_r is None:
            placeholder = Text("waiting for first validation…\n", style="dim")
            placeholder.append("(gate fires when the gen-best individual changes)", style="dim")
            return Panel(placeholder, title="Validation", border_style="green")

        parts: list[Text | Table] = []
        if best_val_r is not None:
            bv = best_val_r["validation"]
            line = Text("Best  ", style="")
            line.append(f"RMS {_format_cost(bv['rms_cost'])}", style="green")
            line.append(f"  g{best_val_r['generation']}", style="dim")
            parts.append(line)
        if last_val_r is not None and last_val_r is not best_val_r:
            lv = last_val_r["validation"]
            outcome, style = ("PROMOTED", "green") if last_val_r.get("improvement") else ("REJECTED", "yellow")
            line = Text(f"Last  RMS {_format_cost(lv['rms_cost'])}  ")
            line.append(outcome, style=style)
            parts.append(line)

        detail_src = last_val_r if last_val_r is not None and last_val_r.get("validation_summary") else best_val_r
        title = "Validation"
        if detail_src is not None and detail_src.get("validation_summary"):
            summary = detail_src["validation_summary"]
            title = f"Validation ({summary.get('n_sims', 0)} sims · g{detail_src['generation']})"
            grid = Table.grid(padding=(0, 2))
            grid.add_column()
            for _ in range(4):
                grid.add_column(justify="right")
            for label, cells, style in _validation_summary_rows(summary):
                padded = cells + [""] * (4 - len(cells)) if len(cells) < 4 else cells
                grid.add_row(*(Text(c, style=style or "") for c in [label, *padded]))
            parts.append(grid)
        return Panel(Group(*parts), title=title, border_style="green")

    @staticmethod
    def _scan_validation_records(buf: list[dict]) -> tuple[dict | None, dict | None]:
        """(best_val_record, last_val_record) by min rms / max generation."""
        best_val_r: dict | None = None
        last_val_r: dict | None = None
        for r in buf:
            if "validation" not in r:
                continue
            if last_val_r is None or r["generation"] >= last_val_r["generation"]:
                last_val_r = r
            rms = r["validation"].get("rms_cost")
            if rms is None:
                continue
            if best_val_r is None or rms < best_val_r["validation"].get("rms_cost", float("inf")):
                best_val_r = r
        return best_val_r, last_val_r

    def _build_footer(self, buf: list[dict]) -> Text:
        from rich.text import Text  # noqa: PLC0415

        latest = buf[-1]
        gen = latest["generation"]
        t = Text(" ")
        improvements = [r["generation"] for r in buf if r.get("improvement")]
        if improvements:
            last_imp = improvements[-1]
            stag = gen - last_imp
            if stag > 0:
                t.append(f"Stagnant {stag} gens", style="yellow")
            else:
                t.append("Improved this gen", style="green")
            t.append(f" · improved g{last_imp}", style="dim")
        else:
            t.append("No improvement yet", style="yellow")
        params = latest.get("best_params")
        if params:
            if self._scheme == "neural_network":
                t.append(f" · {len(params)} NN params (best_model.json)", style="dim")
            else:
                items = list(params.items())
                preview = ", ".join(f"{k} {v:.4g}" for k, v in items[:3])
                more = f" (+{len(items) - 3} more)" if len(items) > 3 else ""
                t.append(f" · best: {preview}{more}", style="dim")
        return t

    def _build_dashboard(self, logger: TrainingLogger, current_run: int) -> ConsoleRenderable:
        import time  # noqa: PLC0415

        from rich.columns import Columns  # noqa: PLC0415
        from rich.console import Group  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        if self._start_time is None:
            self._start_time = time.monotonic()
        buf = logger.buffer
        if not buf:
            return Group(self._build_header(gen=self._start_gen, pop=None, elapsed=0.0), Text(" Waiting for first generation…", style="dim"))
        latest = buf[-1]
        gen = latest["generation"]
        pop = len(latest["all_costs"]) if latest.get("all_costs") else None
        elapsed = time.monotonic() - self._start_time
        return Group(
            self._build_header(gen, pop, elapsed),
            Columns([self._build_optimization_panel(buf), self._build_validation_panel(buf)]),
            self._build_footer(buf),
        )
```

(c) `update()` dispatch: replace the `_build_panel` call (display.py:354-355) with:

```python
        self._live.update(self._build_dashboard(logger, current_run))
```

(d) `TYPE_CHECKING` block: add `from rich.panel import Panel`, `from rich.table import Table`, `from rich.text import Text` so the method annotations resolve. Also remove the now-unused Task-3 shim in the old `_build_panel` (the whole function is deleted).

(e) mypy notes: the `parts: list[Text | Table]` union and `Group(*parts)` are typed fine in Rich 15; if mypy flags the `Text | Table` list under `Group`, annotate `parts: list[ConsoleRenderable]` instead — both are `ConsoleRenderable`. Document whichever you needed.

- [ ] **Step 4.4: Run the full display + islands net**

Run: `uv run pytest tests/test_training_display.py tests/test_island_model.py -q`
Expected: all pass.

- [ ] **Step 4.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/display.py tests/test_training_display.py
git commit -m "feat(train): single-algo TUI dashboard (header/optimization/validation/footer)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: train.py wiring + verification

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (2 one-liners)

- [ ] **Step 5.1: Wire the display**

(a) At the `create_display` call (~train.py:1137-1142), add the kwarg:

```python
    display = create_display(
        scheme=config.guidance_type,
        n_runs=1,
        n_generations=config.optimizer.n_gen,
        enabled=not no_tui and verbose,
        algorithm=config.optimizer.algorithm,
    )
```

(b) Directly after that call, add the single-algo resume wiring (mirrors the islands call at train.py:1798; `start_gen` is defined at ~1132):

```python
    display.set_start_gen(start_gen)
```

- [ ] **Step 5.2: Full verification net**

Run: `./lint_code.sh`
Then: `uv run pytest tests/test_training_display.py tests/test_eval_summary.py tests/test_island_model.py tests/test_train_interrupt.py tests/test_training_report.py tests/test_final_select.py -q`
Expected: clean lint; all tests pass.

- [ ] **Step 5.3: E2E smoke + visual snapshot**

```bash
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml \
    --algorithm qpso --n-gen 2 --n-pop 6 --no-tui --skip-report --final-n-sims 50 \
    --output-dir /tmp/tui_smoke --from-scratch && rm -rf /tmp/tui_smoke
```

Expected: completes (the `--no-tui` path proves the new kwargs don't break NoopDisplay wiring).

Then produce the visual snapshot for the report — write a throwaway `/tmp/tui_render.py` that builds a `LiveDisplay(scheme="ftc", n_runs=1, n_generations=2000, algorithm="qpso")`, calls `set_start_gen(700)`, builds ~40 synthetic records shaped like `tests/test_training_display.py::_record`/`_val_record` (import them from the test module), renders `_build_dashboard` on `Console(record=True, width=110, force_terminal=True)`, and prints `console.export_text()`. Run `uv run python /tmp/tui_render.py`, PASTE the full frame in your report, and delete the script. This is the human eyeball check on the final look.

- [ ] **Step 5.4: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(train): wire algorithm name + resume-aware ETA into the single-algo TUI

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Documentation sync + final commit (smart-commit)

- [ ] **Step 6.1: Invoke the `smart-commit` skill**

Invoke the `smart-commit` skill, telling it to take the **whole git branch** (`feature/tui-dashboard`) into account. CLAUDE.md: `display.py` bullet (dashboard layout, builders, `_validation_summary_rows`, histogram, resume-aware ETA), `compute_eval_summary` new fields wherever validation_summary is described, test-coverage note. README: the Rich TUI feature line.
