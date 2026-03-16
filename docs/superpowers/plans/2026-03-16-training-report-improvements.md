# Training Report Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix resume training so `--n-gen` means "N additional generations" and enhance reports with seed/dispersion panels and resume markers.

**Architecture:** One-line offset in `train.py` fixes resume numbering. `report.py` gets `load_run_data()` extended to return resume points from JSONL file boundaries, a dynamic grid layout, and two conditional panels (seed pool evolution, MC seed trace). No new files created.

**Tech Stack:** Python, Plotly, existing JSONL format (no changes)

**Spec:** `docs/superpowers/specs/2026-03-16-training-report-improvements-design.md`

---

## Chunk 1: Resume Continuation Fix

### Task 1: Fix generation offset on resume (`train.py`)

**Files:**
- Modify: `src/python/aerocapture/training/train.py:307-328`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training_report.py`:

```python
class TestResumeGenerationOffset:
    """Verify --n-gen means 'N additional' when resuming."""

    def test_resumed_n_gen_is_offset(self) -> None:
        """After resume from gen 100 with --n-gen 50, config.ga.n_gen should be 150."""
        from aerocapture.training.config import TrainingConfig

        config = TrainingConfig()
        config.ga.n_gen = 50  # CLI --n-gen 50
        config.ga.n_runs = 1  # CLI default

        # Simulate resume: checkpoint was at generation 100
        start_gen = 100
        resumed = {"generation": 100}

        if resumed is not None and config.ga.n_runs == 1:
            config.ga.n_gen += resumed["generation"]

        assert config.ga.n_gen == 150
        # Loop would be range(100, 150) -> gens 100..149, logged as 101..150
        loop_gens = list(range(start_gen, config.ga.n_gen))
        assert loop_gens[0] == 100
        assert loop_gens[-1] == 149
        assert len(loop_gens) == 50

    def test_no_resume_n_gen_unchanged(self) -> None:
        """Without resume, --n-gen means total generations."""
        from aerocapture.training.config import TrainingConfig

        config = TrainingConfig()
        config.ga.n_gen = 100

        resumed = None

        if resumed is not None and config.ga.n_runs == 1:
            config.ga.n_gen += resumed["generation"]

        assert config.ga.n_gen == 100

    def test_multi_run_no_offset(self) -> None:
        """With n_runs > 1, offset is not applied (would inflate subsequent runs)."""
        from aerocapture.training.config import TrainingConfig

        config = TrainingConfig()
        config.ga.n_gen = 50
        config.ga.n_runs = 3

        resumed = {"generation": 100}

        if resumed is not None and config.ga.n_runs == 1:
            config.ga.n_gen += resumed["generation"]

        assert config.ga.n_gen == 50  # Not offset
```

- [ ] **Step 2: Run test to verify it fails (or passes — this tests the logic pattern, not the actual code path)**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_training_report.py::TestResumeGenerationOffset -v`

Expected: PASS (these test the offset logic in isolation; the real fix is wiring it into `train.py`)

- [ ] **Step 3: Apply the offset in `train.py`**

In `src/python/aerocapture/training/train.py`, after line 307 (`seed_pool = SeedPool.from_dict(...)`) and before `start_gen` is used, add the offset. The block around lines 307-328 should become:

```python
            if seed_pool is not None and resumed.get("seed_pool") is not None:
                seed_pool = SeedPool.from_dict(resumed["seed_pool"])
            # Make --n-gen mean "N additional" on resume (only safe with n_runs=1,
            # which is the CLI default; with multiple runs, subsequent runs would
            # inherit the inflated n_gen and loop range(0, inflated) = too many gens)
            if config.ga.n_runs == 1:
                config.ga.n_gen += resumed["generation"]
```

This must be placed **before** line 333 (`create_display(...)`) so the TUI progress bar gets the correct total. The guard `config.ga.n_runs == 1` prevents a bug where subsequent runs in multi-run mode would inherit the inflated `n_gen` (since `gen_start=0` for non-resumed runs).

- [ ] **Step 4: Update argparse help text**

In `src/python/aerocapture/training/train.py` line 633, change:

```python
parser.add_argument("--n-gen", type=int, default=100)
```

to:

```python
parser.add_argument("--n-gen", type=int, default=100, help="Number of generations (additional when resuming)")
```

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -v --tb=short`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_training_report.py
git commit -m "fix: make --n-gen mean 'N additional' when resuming training"
```

---

## Chunk 2: Extend `load_run_data()` to Return Resume Points

### Task 2: Return resume generation numbers from JSONL file boundaries

**Files:**
- Modify: `src/python/aerocapture/training/report.py:17-37`
- Modify: `tests/test_training_report.py`

- [ ] **Step 1: Write failing tests for resume point detection**

Add to `tests/test_training_report.py`:

```python
def _write_resumed_jsonl(path: Path) -> Path:
    """Write two JSONL files simulating a resumed training run."""
    scheme_dir = path / "equilibrium_glide"
    scheme_dir.mkdir(parents=True, exist_ok=True)

    # First session: gens 1-10
    with open(scheme_dir / "run_000_20260311T120000.jsonl", "w") as f:
        for gen in range(1, 11):
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": 1e5 * (0.9 ** gen),
                "mean_cost": 3e5 * (0.9 ** gen),
                "worst_cost": 1e6 * (0.9 ** gen),
                "median_cost": 2e5 * (0.9 ** gen),
                "std_cost": 1.5e5 * (0.9 ** gen),
                "capture_rate": 0.5 + gen * 0.05,
                "population_diversity": 0.5 - gen * 0.02,
                "best_params": {"k": 0.3},
                "improvement": True,
                "scheme": "equilibrium_glide",
                "config_hash": "abc123",
            }
            f.write(json.dumps(record) + "\n")

    # Second session (resumed): gens 11-20
    with open(scheme_dir / "run_000_20260311T140000.jsonl", "w") as f:
        for gen in range(11, 21):
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T14:00:{gen:02d}Z",
                "best_cost": 1e5 * (0.9 ** gen),
                "mean_cost": 3e5 * (0.9 ** gen),
                "worst_cost": 1e6 * (0.9 ** gen),
                "median_cost": 2e5 * (0.9 ** gen),
                "std_cost": 1.5e5 * (0.9 ** gen),
                "capture_rate": 0.5 + gen * 0.025,
                "population_diversity": 0.5 - gen * 0.02,
                "best_params": {"k": 0.3},
                "improvement": gen <= 15,
                "scheme": "equilibrium_glide",
                "config_hash": "abc123",
            }
            f.write(json.dumps(record) + "\n")

    return scheme_dir


class TestResumeDetection:
    def test_detects_resume_from_file_boundaries(self, tmp_path: Path) -> None:
        scheme_dir = _write_resumed_jsonl(tmp_path)
        data, resume_gens = load_run_data(scheme_dir)
        assert len(data) == 20
        assert resume_gens == [11]  # First gen of second file

    def test_no_resume_returns_empty_list(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        data, resume_gens = load_run_data(scheme_dir)
        assert len(data) == 20
        assert resume_gens == []

    def test_multiple_resumes(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "test_scheme"
        scheme_dir.mkdir(parents=True, exist_ok=True)
        # Write 3 JSONL files: gens 1-5, 6-10, 11-15
        for file_idx, (start, end) in enumerate([(1, 6), (6, 11), (11, 16)]):
            ts = f"2026031{file_idx + 1}T120000"
            with open(scheme_dir / f"run_000_{ts}.jsonl", "w") as f:
                for gen in range(start, end):
                    record = {
                        "generation": gen,
                        "run": 0,
                        "timestamp": f"2026-03-1{file_idx + 1}T12:00:00Z",
                        "best_cost": 100.0 / gen,
                        "mean_cost": 300.0 / gen,
                        "worst_cost": 1000.0 / gen,
                        "median_cost": 200.0 / gen,
                        "std_cost": 150.0 / gen,
                        "capture_rate": 0.8,
                        "population_diversity": 0.3,
                        "best_params": {"k": 0.1},
                        "improvement": False,
                        "scheme": "test",
                        "config_hash": "xyz",
                    }
                    f.write(json.dumps(record) + "\n")
        data, resume_gens = load_run_data(scheme_dir)
        assert len(data) == 15
        assert resume_gens == [6, 11]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_training_report.py::TestResumeDetection -v`

Expected: FAIL — `load_run_data()` returns `list[dict]`, not a tuple.

- [ ] **Step 3: Update `load_run_data()` to return resume points**

In `src/python/aerocapture/training/report.py`, replace the `load_run_data` function (lines 17-37):

```python
def load_run_data(scheme_dir: Path) -> tuple[list[dict], list[int]]:
    """Load all JSONL records from a scheme directory, sorted by generation.

    Returns:
        Tuple of (records, resume_generations) where resume_generations
        contains the first generation number from each JSONL file after
        the first (i.e., where training was resumed).
    """
    # Track which file each record came from
    file_records: list[list[dict]] = []
    for jsonl_file in sorted(scheme_dir.glob("*.jsonl")):
        file_recs: list[dict] = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    file_recs.append(json.loads(line))
        if file_recs:
            file_records.append(file_recs)

    # Flatten and sort
    records: list[dict] = []
    for file_recs in file_records:
        records.extend(file_recs)
    records.sort(key=lambda r: r["generation"])

    # Deduplicate: last-writer-wins for same generation (safety net for legacy logs)
    seen: dict[int, int] = {}
    deduped: list[dict] = []
    for r in records:
        gen = r["generation"]
        if gen in seen:
            deduped[seen[gen]] = r
        else:
            seen[gen] = len(deduped)
            deduped.append(r)

    # Detect resume points: first generation of each file after the first
    resume_gens: list[int] = []
    for file_recs in file_records[1:]:
        if file_recs:
            first_gen = min(r["generation"] for r in file_recs)
            if first_gen not in resume_gens:
                resume_gens.append(first_gen)
    resume_gens.sort()

    return deduped, resume_gens
```

- [ ] **Step 4: Update all callers of `load_run_data()` to unpack the tuple**

In `src/python/aerocapture/training/report.py`:

Line 45 in `generate_single_report()`:
```python
    data, resume_gens = load_run_data(scheme_dir)
```

Line 191 in `generate_comparison_report()`:
```python
        data, resume_gens = load_run_data(scheme_dir)
```

Then after the convergence traces loop (after line 206), add resume markers to the comparison convergence panel:

```python
    # Add resume markers from all schemes to the convergence panel
    all_resume_gens: set[int] = set()
    # (Collect resume_gens during the per-scheme loop above, then after the loop:)
    for gen in sorted(all_resume_gens):
        fig.add_vline(
            x=gen, line_dash="dash", line_color="rgba(128, 128, 128, 0.5)",
            annotation_text="resumed", annotation_font_color="gray",
            row=1, col=1,
        )
```

The implementer should collect `all_resume_gens.update(resume_gens)` inside the per-scheme loop.

- [ ] **Step 5: Fix existing tests that call `load_run_data()` directly**

In `tests/test_training_report.py`, update `TestLoadRunData`:

```python
class TestLoadRunData:
    def test_loads_all_records(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        data, resume_gens = load_run_data(scheme_dir)
        assert len(data) == 20
        assert data[0]["generation"] == 1
        assert resume_gens == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "empty_scheme"
        scheme_dir.mkdir()
        data, resume_gens = load_run_data(scheme_dir)
        assert data == []
        assert resume_gens == []
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_training_report.py -v`

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/report.py tests/test_training_report.py
git commit -m "feat: extend load_run_data to detect resume points from JSONL file boundaries"
```

---

## Chunk 3: Resume Markers and Dynamic Grid Layout

### Task 3: Add resume vertical lines to all panels

**Files:**
- Modify: `src/python/aerocapture/training/report.py:40-139`

- [ ] **Step 1: Write a test for resume markers in the report HTML**

Add to `tests/test_training_report.py`:

```python
class TestResumeMarkers:
    def test_report_contains_resume_marker(self, tmp_path: Path) -> None:
        scheme_dir = _write_resumed_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "resumed" in content.lower()

    def test_report_without_resume_has_no_marker(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "resumed" not in content.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_training_report.py::TestResumeMarkers -v`

Expected: FAIL — no "resumed" text in current report.

- [ ] **Step 3: Implement resume markers and dynamic grid in `generate_single_report()`**

Rewrite `generate_single_report()` in `src/python/aerocapture/training/report.py`. The key changes:

1. Unpack `resume_gens` from `load_run_data()`
2. Detect which conditional panels are needed (check if any record has `pool_metrics` or `mc_seed`)
3. Build dynamic `subplot_titles` and `specs` arrays
4. After adding all traces, add vertical dashed lines at each resume generation on every subplot
5. Update the summary panel to include resume count

The full replacement for `generate_single_report()`:

```python
def _add_resume_markers(fig: go.Figure, resume_gens: list[int], n_rows: int, n_cols: int) -> None:
    """Add vertical dashed lines at resume points across all subplots."""
    for gen in resume_gens:
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                fig.add_vline(
                    x=gen,
                    line_dash="dash",
                    line_color="rgba(128, 128, 128, 0.5)",
                    annotation_text="resumed" if (row == 1 and col == 1) else None,
                    annotation_font_color="gray",
                    row=row,
                    col=col,
                )


def generate_single_report(scheme_dir: Path) -> None:
    """Generate a single-run HTML report from JSONL data."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data, resume_gens = load_run_data(scheme_dir)
    if not data:
        print(f"No JSONL data found in {scheme_dir}")
        return

    gens = [r["generation"] for r in data]
    best_costs = [r["best_cost"] for r in data]
    mean_costs = [r["mean_cost"] for r in data]
    worst_costs = [r["worst_cost"] for r in data]
    cap_rates = [r["capture_rate"] * 100 for r in data]
    diversities = [r["population_diversity"] for r in data]

    scheme = data[0].get("scheme", scheme_dir.name)

    # Detect conditional panels
    has_pool_metrics = any(r.get("pool_metrics") for r in data)
    has_mc_seed = any(r.get("mc_seed") is not None for r in data)

    # Build panel list: (title, specs_dict)
    panels: list[tuple[str, dict]] = [
        ("Convergence (log scale)", {}),
        ("Population Diversity vs Best Cost", {"secondary_y": True}),
        ("Capture Rate (%)", {}),
        ("Cost Distribution", {}),
        ("Parameter Evolution", {}),
    ]
    if has_pool_metrics:
        panels.append(("Seed Pool Evolution", {"secondary_y": True}))
    if has_mc_seed:
        panels.append(("MC Seed Trace", {}))
    panels.append(("Summary", {}))

    n_cols = 2
    n_rows = (len(panels) + 1) // 2
    subplot_titles = [p[0] for p in panels]
    specs = []
    for row_start in range(0, len(panels), n_cols):
        row_specs = [panels[i][1] if i < len(panels) else {} for i in range(row_start, row_start + n_cols)]
        specs.append(row_specs)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, specs=specs)

    # === Base panels (same as before) ===

    # 1. Convergence (row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=best_costs, name="Best", line={"color": "#2196F3"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=mean_costs, name="Mean", line={"color": "#FF9800", "dash": "dash"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=worst_costs, name="Worst", line={"color": "#F44336", "dash": "dot"}), row=1, col=1)
    imp_gens = [r["generation"] for r in data if r["improvement"]]
    imp_costs = [r["best_cost"] for r in data if r["improvement"]]
    fig.add_trace(go.Scatter(x=imp_gens, y=imp_costs, mode="markers", name="Improvement", marker={"color": "#4CAF50", "size": 6}), row=1, col=1)
    fig.update_yaxes(type="log", title_text="Cost", row=1, col=1)

    # 2. Diversity + best cost overlay (row=1, col=2)
    fig.add_trace(go.Scatter(x=gens, y=diversities, name="Diversity", line={"color": "#9C27B0"}), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=gens, y=best_costs, name="Best Cost", line={"color": "#2196F3", "dash": "dot"}), row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Diversity", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Best Cost", type="log", row=1, col=2, secondary_y=True)

    # 3. Capture rate (row=2, col=1)
    fig.add_trace(go.Scatter(x=gens, y=cap_rates, name="Capture %", line={"color": "#4CAF50"}, fill="tozeroy"), row=2, col=1)
    fig.update_yaxes(title_text="Capture Rate (%)", range=[0, 105], row=2, col=1)

    # 4. Cost distribution (row=2, col=2)
    n_boxes = min(10, len(data))
    step = max(1, len(data) // n_boxes)
    for i in range(0, len(data), step):
        r = data[i]
        fig.add_trace(
            go.Box(y=[r["best_cost"], r["median_cost"], r["mean_cost"], r["worst_cost"]], name=f"Gen {r['generation']}", showlegend=False),
            row=2, col=2,
        )
    fig.update_yaxes(type="log", title_text="Cost", row=2, col=2)

    # 5. Parameter evolution (row=3, col=1)
    first_params = data[0].get("best_params")
    if first_params is not None:
        for param_name in first_params:
            vals = [r["best_params"][param_name] for r in data if r.get("best_params")]
            param_gens = [r["generation"] for r in data if r.get("best_params")]
            fig.add_trace(go.Scatter(x=param_gens, y=vals, name=param_name), row=3, col=1)
    fig.update_yaxes(title_text="Parameter Value", row=3, col=1)

    # === Conditional panels ===
    # Build a position lookup: panel_positions[i] = (row, col) for 0-indexed panel i
    panel_positions = [(i // n_cols + 1, i % n_cols + 1) for i in range(len(panels))]

    # Find conditional panel positions by title
    pool_pos = next((panel_positions[i] for i, (t, _) in enumerate(panels) if t == "Seed Pool Evolution"), None)
    seed_pos = next((panel_positions[i] for i, (t, _) in enumerate(panels) if t == "MC Seed Trace"), None)
    summary_pos = next((panel_positions[i] for i, (t, _) in enumerate(panels) if t == "Summary"), None)

    if has_pool_metrics and pool_pos:
        p_row, p_col = pool_pos
        pool_gens = [r["generation"] for r in data if r.get("pool_metrics")]
        pool_sizes = [r["pool_metrics"]["pool_size"] for r in data if r.get("pool_metrics")]
        diff_mins = [r["pool_metrics"]["difficulty_min"] for r in data if r.get("pool_metrics")]
        diff_maxs = [r["pool_metrics"]["difficulty_max"] for r in data if r.get("pool_metrics")]
        fig.add_trace(go.Scatter(x=pool_gens, y=pool_sizes, name="Pool Size", line={"color": "#2196F3"}), row=p_row, col=p_col, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=pool_gens, y=diff_maxs, name="Diff. Max", line={"color": "#FF9800", "dash": "dot"}, fill=None),
            row=p_row, col=p_col, secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=pool_gens, y=diff_mins, name="Diff. Min", line={"color": "#FF9800", "dash": "dot"}, fill="tonexty"),
            row=p_row, col=p_col, secondary_y=True,
        )
        fig.update_yaxes(title_text="Pool Size", row=p_row, col=p_col, secondary_y=False)
        fig.update_yaxes(title_text="Difficulty", row=p_row, col=p_col, secondary_y=True)

    if has_mc_seed and seed_pos:
        p_row, p_col = seed_pos
        seed_gens = [r["generation"] for r in data if r.get("mc_seed") is not None]
        seed_vals = [r["mc_seed"] for r in data if r.get("mc_seed") is not None]
        fig.add_trace(go.Scatter(x=seed_gens, y=seed_vals, name="MC Seed", mode="lines+markers", line={"color": "#795548"}, marker={"size": 4}), row=p_row, col=p_col)
        fig.update_yaxes(title_text="MC Seed", row=p_row, col=p_col)

    # === Summary panel (always last) ===
    assert summary_pos is not None
    summary_row, summary_col = summary_pos

    cost_history = [r["best_cost"] for r in data]
    conv_speed = convergence_speed(cost_history)
    stag = stagnation_count(cost_history)
    config_hash = data[0].get("config_hash", "N/A")

    summary_text = (
        f"Scheme: {scheme}<br>"
        f"Final best cost: {best_costs[-1]:.4e}<br>"
        f"Total generations: {len(data)}<br>"
        f"Convergence speed (90%): gen {conv_speed}<br>"
        f"Final stagnation: {stag} gens<br>"
        f"Config hash: {config_hash}"
    )
    if resume_gens:
        summary_text += f"<br>Resume points: {len(resume_gens)}"

    # Use row/col kwargs with "x domain"/"y domain" refs to avoid manual axis
    # index calculation (secondary_y panels shift y-axis numbering, making xN/yN unreliable)
    fig.add_annotation(
        text=summary_text,
        xref="x domain", yref="y domain",
        x=0.5, y=0.5, showarrow=False, font={"size": 12}, align="left",
        row=summary_row, col=summary_col,
    )

    # === Resume markers on all panels ===
    _add_resume_markers(fig, resume_gens, n_rows, n_cols)

    fig.update_layout(height=max(1000, n_rows * 350), title_text=f"Training Report — {scheme}", showlegend=True)
    fig.update_xaxes(title_text="Generation", row=n_rows, col=1)

    output_path = scheme_dir / "report.html"
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Report saved to {output_path}")
```

Note: The `panel_positions` list provides a clean mapping from panel index to `(row, col)` using `(i // n_cols + 1, i % n_cols + 1)`. Panels are looked up by title to avoid fragile index arithmetic. The summary annotation uses `xref="paper"` / `yref="paper"` to avoid Plotly axis numbering issues caused by `secondary_y` panels consuming extra axis indices.

- [ ] **Step 4: Run tests to verify resume markers**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_training_report.py::TestResumeMarkers -v`

Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_training_report.py -v`

Expected: All tests pass (including existing tests adapted for the tuple return).

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/report.py tests/test_training_report.py
git commit -m "feat: add resume markers, seed pool panel, MC seed trace, dynamic grid layout"
```

---

## Chunk 4: Conditional Panel Tests

### Task 4: Test that conditional panels appear/disappear based on JSONL data

**Files:**
- Modify: `tests/test_training_report.py`

- [ ] **Step 1: Write tests for conditional panel rendering**

Add to `tests/test_training_report.py`:

```python
def _write_fixture_with_pool_metrics(path: Path, n_gens: int = 10) -> Path:
    """Write JSONL with pool_metrics fields (adaptive seeds)."""
    scheme_dir = path / "adaptive_scheme"
    scheme_dir.mkdir(parents=True, exist_ok=True)
    with open(scheme_dir / "run_000_20260311T120000.jsonl", "w") as f:
        for gen in range(1, n_gens + 1):
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": 1e5 * (0.9 ** gen),
                "mean_cost": 3e5 * (0.9 ** gen),
                "worst_cost": 1e6 * (0.9 ** gen),
                "median_cost": 2e5 * (0.9 ** gen),
                "std_cost": 1.5e5 * (0.9 ** gen),
                "capture_rate": 0.8,
                "population_diversity": 0.3,
                "best_params": {"k": 0.3},
                "improvement": gen <= 5,
                "scheme": "test",
                "config_hash": "abc",
                "pool_metrics": {
                    "pool_size": gen + 4,
                    "difficulty_min": 600.0 + gen * 10,
                    "difficulty_max": 800.0 + gen * 5,
                    "n_evictions": gen // 3,
                },
            }
            f.write(json.dumps(record) + "\n")
    return scheme_dir


def _write_fixture_with_mc_seed(path: Path, n_gens: int = 10) -> Path:
    """Write JSONL with mc_seed fields (rotate seeds)."""
    scheme_dir = path / "rotate_scheme"
    scheme_dir.mkdir(parents=True, exist_ok=True)
    with open(scheme_dir / "run_000_20260311T120000.jsonl", "w") as f:
        for gen in range(1, n_gens + 1):
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": 1e5 * (0.9 ** gen),
                "mean_cost": 3e5 * (0.9 ** gen),
                "worst_cost": 1e6 * (0.9 ** gen),
                "median_cost": 2e5 * (0.9 ** gen),
                "std_cost": 1.5e5 * (0.9 ** gen),
                "capture_rate": 0.8,
                "population_diversity": 0.3,
                "best_params": {"k": 0.3},
                "improvement": gen <= 5,
                "scheme": "test",
                "config_hash": "abc",
                "mc_seed": 42 + gen,
            }
            f.write(json.dumps(record) + "\n")
    return scheme_dir


class TestConditionalPanels:
    def test_pool_metrics_panel_appears(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_with_pool_metrics(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "Seed Pool" in content or "Pool Size" in content

    def test_mc_seed_panel_appears(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_with_mc_seed(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "MC Seed" in content

    def test_no_extra_panels_without_seed_data(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "Seed Pool" not in content
        assert "MC Seed" not in content
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_training_report.py::TestConditionalPanels -v`

Expected: PASS (the implementation from Task 3 should handle these).

- [ ] **Step 3: Commit**

```bash
git add tests/test_training_report.py
git commit -m "test: add conditional panel rendering tests for seed pool and MC seed"
```

---

## Chunk 5: Linting, Full Test Suite, Smart Commit

### Task 5: Final validation

**Files:**
- All modified files

- [ ] **Step 1: Run linter**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh`

Expected: Clean (no ruff or mypy errors). Fix any issues.

- [ ] **Step 2: Run full Python test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -v --tb=short`

Expected: All tests pass.

- [ ] **Step 3: Run Rust tests (unchanged but verify no breakage)**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./check_all.sh`

Expected: All pass.

- [ ] **Step 4: Smart commit**

Invoke the `smart-commit` skill, taking the whole git branch into account. This will sync CLAUDE.md and README.md with the changes and create a final commit.
