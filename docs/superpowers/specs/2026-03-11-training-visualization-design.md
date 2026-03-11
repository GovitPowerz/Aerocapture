# Training Visualization Design

**Date:** 2026-03-11
**Scope:** IMPROVEMENTS.md Section 11.3 — Live plots, convergence diagnostics, experiment tracking
**Approach:** Structured logging with Rich TUI + Plotly HTML reports (pure local, no external services)

---

## Overview

The GA training pipeline currently logs minimal console output (best cost every 5 generations) and saves checkpoint files, but provides no visualization of training dynamics. This design adds three capabilities:

1. **TrainingLogger** — structured per-generation metrics written to JSONL files
2. **Live TUI** — Rich-based terminal display with sparklines, progress bar, and stagnation warnings
3. **Post-training reports** — self-contained Plotly HTML reports for single-run analysis and cross-run comparison

---

## 1. TrainingLogger (`logger.py`)

### Responsibility

Collects per-generation metrics during training and writes them to a JSON-lines file. Single source of truth for both the live TUI and post-training reports.

### Metrics per generation

| Field | Type | Description |
|-------|------|-------------|
| `generation` | int | Generation index (1-based). `train.py` passes `gen + 1` to `log_generation`. |
| `run` | int | Run index (0-based) |
| `timestamp` | str | ISO 8601 timestamp |
| `best_cost` | float | Best cost in population |
| `mean_cost` | float | Mean cost across population |
| `worst_cost` | float | Worst cost in population |
| `median_cost` | float | Median cost |
| `std_cost` | float | Standard deviation of costs |
| `capture_rate` | float | Fraction of population with `cost < 1e6`. In `compute_cost`, non-capturing trajectories receive a cost of `1e6 + 1e3 * |energy|`, so `1e6` is the floor of the hyperbolic-branch cost, not an exact sentinel. The `< 1e6` threshold is a cost-domain proxy for physical capture (`ecc < 1 and energy < 0`). (0.0–1.0) |
| `population_diversity` | float | Mean pairwise Hamming distance of binary chromosomes, normalized 0–1 |
| `best_params` | dict \| None | Decoded parameter values from best chromosome. For non-NN schemes, this is a `dict[str, float]` with named keys from `PARAM_SPACES`. For `neural_network`, this is `None` (110 unnamed weights are not useful for display); the TUI and reports skip this field for NN runs. |
| `improvement` | bool | Whether `best_cost` improved this generation |
| `scheme` | str | Guidance scheme name |
| `config_hash` | str | Hash of the training TOML config for experiment grouping |

### File format

- Path: `training_output/<scheme>/run_<run>_<YYYYMMDDTHHMMSS>.jsonl` (e.g., `run_000_20260311T143022.jsonl`)
- Timestamp is the session start time — each `train.py` invocation creates a new file
- On resume (`--resume`), a **new** JSONL file is created (not appended to the old one). The resumed run logs generations starting from `gen_start + 1`, so there is no overlap with the previous file's generation numbers. `report.py` merges all JSONL files for the same run index by sorting on `generation`. If duplicate generation numbers are found (e.g., from a corrupted resume), the later file's records take precedence (last-writer-wins by file timestamp).
- One JSON object per line, append-only during a session
- Trivially parseable with `json.loads()` per line

### API

```python
class TrainingLogger:
    def __init__(self, scheme: str, run: int, output_dir: Path, config_hash: str) -> None: ...
    def log_generation(self, generation: int, populations: list[np.ndarray],
                       costs: list[np.ndarray], best_chromosome: np.ndarray,
                       decode_fn: Callable[[np.ndarray], dict[str, float]] | None) -> None: ...
    @property
    def buffer(self) -> list[dict]: ...  # In-memory metrics for LiveDisplay
    def close(self) -> None: ...
```

The logger computes derived metrics internally (diversity, stats) — `train.py` passes raw population data and a decode function.

**`decode_fn` contract:** A callable that takes a binary chromosome (`np.ndarray` of `np.int8`, values strictly `{0, 1}`) and returns a `dict[str, float]` of named parameter values. For non-NN schemes, `train.py` passes a lambda closing over `config`: `lambda chrom: decode_params_from_chromosome(chrom, config)`. For `neural_network`, pass `None` — the logger sets `best_params` to `None` in the JSONL record.

**Binary invariant:** `populations` arrays must contain only `{0, 1}` values (`np.int8`). The `population_diversity` function assumes this. This invariant is maintained by the GA operators (`crossover_and_mutate` uses `1 - flat[pos]` bit-flip).

### Integration with train.py

- Instantiated at run start, called once per generation
- Called **after** the tournament selection + migration step (i.e., the populations passed are the surviving elite, not the pre-tournament pool). This is the natural insertion point — after the inner subpopulation loop completes but before checkpoint saving. The logged costs and chromosomes represent the population that will seed the next generation.
- `train.py` does not compute any metrics itself — the logger owns all metric computation

### Multi-subpopulation handling

The logger receives `populations: list[np.ndarray]` (one array per subpopulation). For aggregate metrics (`population_diversity`, `cost_stats`, `capture_rate`), the logger concatenates all subpopulations via `np.vstack` / `np.concatenate` before computing. Per-subpopulation breakdowns are not logged (the aggregate view is sufficient for training diagnostics).

---

## 2. LiveDisplay (`display.py`)

### Responsibility

Real-time terminal visualization of training progress using `rich.live.Live`.

### Layout

```
+-- Equilibrium Glide . Run 1/3 . Gen 34/50 ---------------------+
| Best cost  6.79e+03  ..........sparkline..........              |
| Mean cost  2.41e+04  ..........sparkline..........              |
| Capture    92%       ..........sparkline..........              |
| Diversity  0.38      ..........sparkline..........              |
|                                                                 |
| ========================== 68%  ETA 1m 12s                      |
|                                                                 |
| Stagnant for 8 gens . Last improvement: gen 26                  |
| Best params: {k_alt: 1.23, k_hdot: 0.45, ...}                  |
+-----------------------------------------------------------------+
```

Four sparkline rows (Unicode block characters) showing trends for best cost, mean cost, capture rate, and diversity. Progress bar with ETA. Stagnation warning. Best params always visible.

### API

```python
class LiveDisplay:
    def __init__(self, scheme: str, n_runs: int, n_generations: int) -> None: ...
    def update(self, logger: TrainingLogger, current_run: int) -> None: ...
    def __enter__(self) -> LiveDisplay: ...
    def __exit__(self, *args) -> None: ...
```

### Design decisions

- Context manager wrapping `rich.live.Live`
- Reads from `TrainingLogger.buffer` — no direct coupling to training internals
- Falls back when `--no-tui` is passed or when a non-interactive terminal is auto-detected. In fallback mode, `LiveDisplay` is a no-op (context manager that does nothing), and `train.py`'s existing `verbose` print path remains unchanged. The `verbose` flag continues to control plain-text output independently of the TUI.

---

## 3. Metrics module (`metrics.py`)

### Responsibility

Pure functions for computing derived training metrics. Used by both `TrainingLogger` (during training) and `report.py` (post-hoc analysis).

### Functions

```python
def cost_stats(costs: np.ndarray) -> dict[str, float]:
    """Compute best/mean/worst/median/std cost, filtering np.inf and np.nan.
    Returns np.nan for stats when all values are non-finite."""

def population_diversity(chromosomes: np.ndarray) -> float:
    """Mean pairwise Hamming distance, normalized 0-1. Assumes binary {0,1} input."""

def convergence_speed(cost_history: list[float], threshold: float = 0.9) -> int:
    """Generation at which threshold% of final improvement was achieved."""

def stagnation_count(cost_history: list[float]) -> int:
    """Number of consecutive generations without improvement at end of history."""

def capture_rate(costs: np.ndarray, capture_threshold: float = 1e6) -> float:
    """Fraction of individuals with cost below capture threshold (default 1e6, the compute_cost sentinel)."""
```

---

## 4. Post-training reports (`report.py`)

### Responsibility

CLI tool that reads JSONL log files and generates self-contained Plotly HTML reports.

### Single-run report

Command: `uv run python -m aerocapture.training.report training_output/equilibrium_glide/`

Generates 6 figures in a single HTML file:

1. **Convergence plot** — best/mean/worst cost vs generation (log-scale Y axis, markers on improvement generations)
2. **Population diversity** — diversity vs generation, overlaid with best cost on secondary Y axis
3. **Parameter evolution** — one subplot per parameter, showing best chromosome's decoded values across generations
4. **Cost distribution** — box plots sampled every N generations (N auto-selected to avoid clutter)
5. **Capture rate** — percentage over generations
6. **Summary table** — final best cost, total generations, wall time, stagnation count, config hash

Output: `training_output/<scheme>/report.html`

### Multi-run comparison report

Command: `uv run python -m aerocapture.training.report --compare training_output/`

Generates cross-run visualizations:

1. **Cross-scheme convergence** — all schemes' best-cost curves on one plot, using existing `SCHEME_COLORS` from `plot_comparison.py`
2. **Same-scheme overlay** — multiple runs of the same scheme overlaid, legend shows config differences
3. **Final metrics table** — sortable: scheme, best cost, generations, capture rate, convergence speed

Filters: `--schemes equilibrium_glide ftc`, `--after 2026-03-01`

Output: `training_output/comparison_report.html`

---

## 5. File layout

### New files

```
src/python/aerocapture/training/
  logger.py          -- TrainingLogger (metrics collection, JSONL writing)
  display.py         -- LiveDisplay (Rich Live TUI)
  report.py          -- CLI for Plotly HTML report generation
  metrics.py         -- Pure functions: diversity, convergence speed, stagnation
```

### New dependencies

Added to `[dependency-groups]` as a `viz` group in `pyproject.toml` (consistent with how `hypothesis` lives in `dev`):

- `rich` — live TUI
- `plotly` — self-contained HTML reports

Install with: `uv sync --group viz` (or `uv sync --group viz --group dev` for development).

### Unchanged

- Checkpoint format (`.json` + `.npz`) — untouched
- `train.py` CLI args — only `--no-tui` added
- `compare_guidance.py` — untouched (final MC comparison, different purpose)
- `plot_comparison.py` — untouched

---

## 6. Data flow

```
train.py loop
  |
  |-->  TrainingLogger.log_generation(pop, costs, best_chromosome)
  |       |-- computes derived metrics (via metrics.py)
  |       |-- appends to .jsonl file
  |       |-- updates in-memory buffer
  |
  |-->  LiveDisplay.update(logger)
          |-- reads logger buffer, renders Rich Live panel

Post-training:
  report.py
  |-- reads .jsonl files (glob pattern)
  |-- builds Plotly figures
  |-- writes self-contained .html
```

---

## 7. Testing strategy

- **metrics.py** — unit tests with known chromosomes/costs, property-based tests with hypothesis
- **logger.py** — unit tests: write to temp dir, verify JSONL structure, verify metric computation
- **display.py** — minimal smoke test (instantiate, call update with mock logger, verify no crash). No visual regression tests.
- **report.py** — integration test: generate report from fixture JSONL data, verify HTML file is produced and contains expected Plotly div IDs
- **train.py integration** — one test that monkey-patches `TrainingLogger.log_generation` to record call arguments, verifying it is called once per generation after tournament selection and before checkpoint saving, with the post-tournament population
- **Edge cases** — NaN/inf costs, single-generation runs, empty populations, NN scheme (best_params=None)

---

## 8. Cleanup

### Remove legacy training data

Delete all existing files in `training_output/` (legacy checkpoints, old per-scheme outputs). These were produced before the JSONL logging system and are no longer useful. The directory structure (`training_output/<scheme>/`) will be recreated by new training runs.

### Final step

Use the `smart-commit` skill to commit all changes with updated documentation (CLAUDE.md, README.md).
