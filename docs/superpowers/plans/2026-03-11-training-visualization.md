# Training Visualization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured training metrics logging, live Rich TUI display, and Plotly HTML reports to the GA training pipeline.

**Architecture:** A `TrainingLogger` writes per-generation metrics to JSONL files. A `LiveDisplay` renders Rich sparklines in the terminal during training. A `report.py` CLI reads JSONL files and produces self-contained Plotly HTML reports. All new modules live in `src/python/aerocapture/training/`.

**Tech Stack:** Python 3.14, numpy, rich (TUI), plotly (HTML reports), pytest + hypothesis (testing)

**Spec:** `docs/superpowers/specs/2026-03-11-training-visualization-design.md`

---

## Chunk 1: Foundation — metrics.py and dependencies

### Task 1: Add `rich` and `plotly` dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add viz dependency group**

In `pyproject.toml`, add after the `dev` group:

```toml
viz = [
    "rich>=13.0",
    "plotly>=5.18",
]
```

- [ ] **Step 2: Install and verify**

Run: `uv sync --group viz --group dev`
Expected: Successful install, no conflicts.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add viz dependency group (rich, plotly)"
```

---

### Task 2: Implement `metrics.py` — pure metric functions

**Files:**
- Create: `src/python/aerocapture/training/metrics.py`
- Create: `tests/test_training_metrics.py`

- [ ] **Step 1: Write tests for `cost_stats`**

```python
"""Tests for training metrics pure functions."""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from aerocapture.training.metrics import capture_rate, convergence_speed, cost_stats, population_diversity, stagnation_count


class TestCostStats:
    def test_basic_stats(self) -> None:
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = cost_stats(costs)
        assert stats["best"] == 1.0
        assert stats["worst"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["std"] == pytest.approx(np.std(costs), abs=1e-10)

    def test_filters_inf(self) -> None:
        costs = np.array([1.0, np.inf, 3.0])
        stats = cost_stats(costs)
        assert stats["best"] == 1.0
        assert stats["worst"] == 3.0
        assert stats["mean"] == 2.0

    def test_filters_nan(self) -> None:
        costs = np.array([1.0, np.nan, 5.0])
        stats = cost_stats(costs)
        assert stats["best"] == 1.0
        assert stats["worst"] == 5.0

    def test_all_nonfinite_returns_nan(self) -> None:
        costs = np.array([np.inf, np.nan, np.inf])
        stats = cost_stats(costs)
        assert math.isnan(stats["best"])
        assert math.isnan(stats["mean"])

    def test_single_element(self) -> None:
        costs = np.array([42.0])
        stats = cost_stats(costs)
        assert stats["best"] == 42.0
        assert stats["std"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_metrics.py::TestCostStats -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aerocapture.training.metrics'`

- [ ] **Step 3: Implement `cost_stats`**

```python
"""Pure functions for computing derived training metrics.

Used by both TrainingLogger (during training) and report.py (post-hoc analysis).
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def cost_stats(costs: npt.NDArray[np.float64]) -> dict[str, float]:
    """Compute best/mean/worst/median/std cost, filtering np.inf and np.nan.

    Returns np.nan for all stats when no finite values exist.
    """
    finite = costs[np.isfinite(costs)]
    if len(finite) == 0:
        return {"best": math.nan, "worst": math.nan, "mean": math.nan, "median": math.nan, "std": math.nan}
    return {
        "best": float(np.min(finite)),
        "worst": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_metrics.py::TestCostStats -v`
Expected: All PASS.

- [ ] **Step 5: Write tests for `population_diversity`**

Add to `tests/test_training_metrics.py`:

```python
class TestPopulationDiversity:
    def test_identical_population_zero_diversity(self) -> None:
        pop = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]], dtype=np.int8)
        assert population_diversity(pop) == 0.0

    def test_maximally_diverse_pair(self) -> None:
        pop = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int8)
        assert population_diversity(pop) == 1.0

    def test_partial_diversity(self) -> None:
        pop = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int8)
        assert population_diversity(pop) == pytest.approx(0.25)

    def test_single_individual(self) -> None:
        pop = np.array([[1, 0, 1]], dtype=np.int8)
        assert population_diversity(pop) == 0.0

    @given(
        arrays(dtype=np.int8, shape=st.tuples(st.integers(2, 20), st.integers(1, 50)), elements=st.integers(0, 1)),
    )
    @settings(max_examples=50)
    def test_diversity_in_unit_range(self, pop: np.ndarray) -> None:
        d = population_diversity(pop)
        assert 0.0 <= d <= 1.0
```

- [ ] **Step 6: Run to verify they fail**

Run: `uv run pytest tests/test_training_metrics.py::TestPopulationDiversity -v`
Expected: FAIL — `ImportError`

- [ ] **Step 7: Implement `population_diversity`**

Add to `metrics.py`:

```python
def population_diversity(chromosomes: npt.NDArray[np.int8]) -> float:
    """Mean pairwise Hamming distance, normalized 0-1.

    Assumes binary {0, 1} input. For a single individual, returns 0.0.
    """
    n = len(chromosomes)
    if n < 2:
        return 0.0
    chrom_len = chromosomes.shape[1]
    # Vectorized pairwise Hamming: XOR and sum
    total_distance = 0
    n_pairs = 0
    for i in range(n):
        diffs = np.sum(chromosomes[i] != chromosomes[i + 1 :], axis=1)
        total_distance += int(np.sum(diffs))
        n_pairs += len(diffs)
    return total_distance / (n_pairs * chrom_len)
```

- [ ] **Step 8: Run to verify they pass**

Run: `uv run pytest tests/test_training_metrics.py::TestPopulationDiversity -v`
Expected: All PASS.

- [ ] **Step 9: Write tests for remaining functions**

Add to `tests/test_training_metrics.py`:

```python
class TestCaptureRate:
    def test_all_captured(self) -> None:
        costs = np.array([100.0, 200.0, 500.0])
        assert capture_rate(costs) == 1.0

    def test_none_captured(self) -> None:
        costs = np.array([1e6 + 100, 1e6 + 200, 2e6])
        assert capture_rate(costs) == 0.0

    def test_mixed(self) -> None:
        costs = np.array([100.0, 1e6 + 100, 200.0, 2e6])
        assert capture_rate(costs) == 0.5

    def test_custom_threshold(self) -> None:
        costs = np.array([10.0, 50.0, 100.0])
        assert capture_rate(costs, capture_threshold=50.0) == pytest.approx(1 / 3)


class TestConvergenceSpeed:
    def test_instant_convergence(self) -> None:
        history = [100.0, 10.0, 10.0, 10.0, 10.0]
        assert convergence_speed(history) == 1

    def test_gradual_convergence(self) -> None:
        history = [100.0, 80.0, 60.0, 40.0, 20.0, 10.0]
        # 90% of improvement (100->10 = 90) reached at cost <= 19
        # history[5] = 10 is the first <= 19
        speed = convergence_speed(history)
        assert 1 <= speed <= len(history)

    def test_no_improvement(self) -> None:
        history = [50.0, 50.0, 50.0]
        assert convergence_speed(history) == 0


class TestStagnationCount:
    def test_no_stagnation(self) -> None:
        history = [100.0, 90.0, 80.0, 70.0]
        assert stagnation_count(history) == 0

    def test_full_stagnation(self) -> None:
        history = [50.0, 50.0, 50.0, 50.0]
        assert stagnation_count(history) == 3  # 3 gens without improvement after first

    def test_trailing_stagnation(self) -> None:
        history = [100.0, 50.0, 50.0, 50.0]
        assert stagnation_count(history) == 2
```

- [ ] **Step 10: Run to verify they fail**

Run: `uv run pytest tests/test_training_metrics.py -k "CaptureRate or ConvergenceSpeed or StagnationCount" -v`
Expected: FAIL — `ImportError`

- [ ] **Step 11: Implement remaining functions**

Add to `metrics.py`:

```python
def capture_rate(costs: npt.NDArray[np.float64], capture_threshold: float = 1e6) -> float:
    """Fraction of individuals with cost below capture threshold.

    Default threshold 1e6 is the floor of the hyperbolic-branch cost
    in compute_cost (non-capturing trajectories get 1e6 + 1e3*|energy|).
    """
    return float(np.sum(costs < capture_threshold) / len(costs))


def convergence_speed(cost_history: list[float], threshold: float = 0.9) -> int:
    """Generation at which threshold% of final improvement was achieved.

    Returns 0 if no improvement occurred.
    """
    if len(cost_history) < 2:
        return 0
    initial = cost_history[0]
    final = cost_history[-1]
    total_improvement = initial - final
    if total_improvement <= 0:
        return 0
    target = initial - threshold * total_improvement
    for i, cost in enumerate(cost_history):
        if cost <= target:
            return i
    return len(cost_history) - 1


def stagnation_count(cost_history: list[float]) -> int:
    """Number of consecutive generations without improvement at end of history."""
    if len(cost_history) < 2:
        return 0
    count = 0
    best = cost_history[0]
    last_improvement = 0
    for i in range(1, len(cost_history)):
        if cost_history[i] < best:
            best = cost_history[i]
            last_improvement = i
    return len(cost_history) - 1 - last_improvement
```

- [ ] **Step 12: Run all metrics tests**

Run: `uv run pytest tests/test_training_metrics.py -v`
Expected: All PASS.

- [ ] **Step 13: Run linter**

Run: `./lint_code.sh`
Expected: No errors.

- [ ] **Step 14: Commit**

```bash
git add src/python/aerocapture/training/metrics.py tests/test_training_metrics.py
git commit -m "feat: add training metrics module (cost_stats, diversity, capture_rate, convergence)"
```

---

## Chunk 2: TrainingLogger

### Task 3: Implement `logger.py`

**Files:**
- Create: `src/python/aerocapture/training/logger.py`
- Create: `tests/test_training_logger.py`

- [ ] **Step 1: Write logger tests**

```python
"""Tests for TrainingLogger — JSONL metrics logging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aerocapture.training.logger import TrainingLogger


@pytest.fixture
def logger(tmp_path: Path) -> TrainingLogger:
    return TrainingLogger(scheme="equilibrium_glide", run=0, output_dir=tmp_path, config_hash="abc123")


def _make_populations(n_pop: int = 10, chrom_len: int = 112) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.integers(0, 2, size=(n_pop, chrom_len), dtype=np.int8)]


def _make_costs(n_pop: int = 10) -> list[np.ndarray]:
    return [np.arange(1.0, n_pop + 1, dtype=np.float64) * 100]


def _decode_fn(chrom: np.ndarray) -> dict[str, float]:
    return {"param_a": 1.0, "param_b": 2.0}


class TestTrainingLogger:
    def test_creates_jsonl_file(self, logger: TrainingLogger, tmp_path: Path) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        logger.close()
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        assert "run_000_" in jsonl_files[0].name

    def test_jsonl_record_fields(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        logger.close()
        record = logger.buffer[0]
        required_fields = {"generation", "run", "timestamp", "best_cost", "mean_cost", "worst_cost",
                           "median_cost", "std_cost", "capture_rate", "population_diversity",
                           "best_params", "improvement", "scheme", "config_hash"}
        assert required_fields.issubset(record.keys())

    def test_multiple_generations_appended(self, logger: TrainingLogger, tmp_path: Path) -> None:
        for gen in range(1, 4):
            logger.log_generation(gen, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # Must be valid JSON

    def test_buffer_matches_file(self, logger: TrainingLogger, tmp_path: Path) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        file_record = json.loads(jsonl_file.read_text().strip())
        assert logger.buffer[0]["generation"] == file_record["generation"]
        assert logger.buffer[0]["best_cost"] == file_record["best_cost"]

    def test_improvement_tracking(self, logger: TrainingLogger) -> None:
        costs_improving = [np.array([500.0, 600.0])]
        costs_worse = [np.array([700.0, 800.0])]
        pop = [np.zeros((2, 112), dtype=np.int8)]
        logger.log_generation(1, pop, costs_improving, np.zeros(112, dtype=np.int8), _decode_fn)
        logger.log_generation(2, pop, costs_worse, np.zeros(112, dtype=np.int8), _decode_fn)
        assert logger.buffer[0]["improvement"] is True  # First gen always improves (from inf)
        assert logger.buffer[1]["improvement"] is False

    def test_none_decode_fn_for_nn(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), None)
        assert logger.buffer[0]["best_params"] is None

    def test_multi_subpop_concatenation(self, logger: TrainingLogger) -> None:
        pops = _make_populations(5) + _make_populations(5)
        costs = _make_costs(5) + _make_costs(5)
        logger.log_generation(1, pops, costs, np.zeros(112, dtype=np.int8), _decode_fn)
        # Should not crash; diversity computed on concatenated pop
        assert 0.0 <= logger.buffer[0]["population_diversity"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_logger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aerocapture.training.logger'`

- [ ] **Step 3: Implement `logger.py`**

```python
"""Structured per-generation training metrics logger.

Writes one JSON-lines file per training session. Each line is a complete
record of metrics for one generation. The in-memory buffer feeds the
LiveDisplay; the file is the source of truth for post-training reports.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aerocapture.training.metrics import capture_rate, cost_stats, population_diversity

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


class TrainingLogger:
    """Collects per-generation metrics and writes them to a JSONL file."""

    def __init__(self, scheme: str, run: int, output_dir: Path, config_hash: str) -> None:
        self._scheme = scheme
        self._run = run
        self._config_hash = config_hash
        self._buffer: list[dict] = []
        self._best_cost = float("inf")

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._filepath = output_dir / f"run_{run:03d}_{timestamp}.jsonl"
        self._file = open(self._filepath, "a")  # noqa: SIM115

    def log_generation(
        self,
        generation: int,
        populations: list[npt.NDArray[np.int8]],
        costs: list[npt.NDArray[np.float64]],
        best_chromosome: npt.NDArray[np.int8],
        decode_fn: Callable[[npt.NDArray[np.int8]], dict[str, float]] | None,
    ) -> None:
        """Log metrics for one generation."""
        all_chroms = np.vstack(populations)
        all_costs = np.concatenate(costs)

        stats = cost_stats(all_costs)
        cap_rate = capture_rate(all_costs)
        diversity = population_diversity(all_chroms)

        gen_best = stats["best"]
        improved = gen_best < self._best_cost
        if improved:
            self._best_cost = gen_best

        best_params = decode_fn(best_chromosome) if decode_fn is not None else None

        record = {
            "generation": generation,
            "run": self._run,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "best_cost": gen_best,
            "mean_cost": stats["mean"],
            "worst_cost": stats["worst"],
            "median_cost": stats["median"],
            "std_cost": stats["std"],
            "capture_rate": cap_rate,
            "population_diversity": diversity,
            "best_params": best_params,
            "improvement": improved,
            "scheme": self._scheme,
            "config_hash": self._config_hash,
        }

        self._buffer.append(record)
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    @property
    def buffer(self) -> list[dict]:
        """In-memory metrics buffer for LiveDisplay."""
        return self._buffer

    def close(self) -> None:
        """Close the JSONL file."""
        self._file.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_logger.py -v`
Expected: All PASS.

- [ ] **Step 5: Run linter**

Run: `./lint_code.sh`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/logger.py tests/test_training_logger.py
git commit -m "feat: add TrainingLogger with JSONL output and in-memory buffer"
```

---

## Chunk 3: LiveDisplay

### Task 4: Implement `display.py`

**Files:**
- Create: `src/python/aerocapture/training/display.py`
- Create: `tests/test_training_display.py`

- [ ] **Step 1: Write smoke tests**

```python
"""Smoke tests for LiveDisplay — verify no crash, not visual correctness."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from aerocapture.training.display import LiveDisplay, NoopDisplay, create_display


class TestLiveDisplay:
    def test_create_display_returns_noop_in_non_tty(self) -> None:
        """In non-interactive environments (CI), create_display returns NoopDisplay."""
        display = create_display(scheme="equilibrium_glide", n_runs=1, n_generations=50, enabled=False)
        assert isinstance(display, NoopDisplay)

    def test_build_panel_does_not_crash(self) -> None:
        """Test panel building logic without opening a Live context."""
        display = LiveDisplay(scheme="equilibrium_glide", n_runs=1, n_generations=50)
        logger = MagicMock()
        logger.buffer = [
            {
                "generation": 1,
                "best_cost": 1000.0,
                "mean_cost": 5000.0,
                "capture_rate": 0.8,
                "population_diversity": 0.45,
                "improvement": True,
                "best_params": {"k_hdot_scale": 0.3},
            },
        ]
        panel = display._build_panel(logger, current_run=0)
        assert panel is not None


class TestNoopDisplay:
    def test_noop_context_manager(self) -> None:
        display = NoopDisplay()
        with display:
            display.update(MagicMock(), current_run=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_display.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `display.py`**

```python
"""Live Rich TUI display for GA training progress.

Shows sparklines for key metrics, progress bar with ETA,
stagnation warnings, and current best parameters.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from types import TracebackType

    from aerocapture.training.logger import TrainingLogger

# Sparkline characters (increasing height)
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 30) -> str:
    """Render a list of floats as a Unicode sparkline string."""
    if not values:
        return " " * width
    # Take last `width` values
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    return "".join(_SPARK_CHARS[min(int((v - lo) / span * 8), 8)] for v in vals)


def _format_cost(value: float) -> str:
    """Format cost in scientific notation."""
    return f"{value:.2e}"


class DisplayProtocol(Protocol):
    """Protocol for training display (allows NoopDisplay as substitute)."""

    def update(self, logger: TrainingLogger, current_run: int) -> None: ...
    def __enter__(self) -> DisplayProtocol: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None: ...


class NoopDisplay:
    """No-op display for non-interactive terminals or --no-tui mode."""

    def update(self, logger: TrainingLogger, current_run: int) -> None:
        pass

    def __enter__(self) -> NoopDisplay:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        pass


class LiveDisplay:
    """Rich Live TUI for training progress."""

    def __init__(self, scheme: str, n_runs: int, n_generations: int) -> None:
        self._scheme = scheme
        self._n_runs = n_runs
        self._n_gens = n_generations
        self._live = None
        self._start_time: float | None = None

    def _build_panel(self, logger: TrainingLogger, current_run: int) -> object:
        """Build a Rich Panel from logger buffer."""
        import time

        from rich.panel import Panel
        from rich.text import Text

        buf = logger.buffer
        if not buf:
            return Panel("Waiting for first generation...", title=self._scheme)

        if self._start_time is None:
            self._start_time = time.monotonic()

        latest = buf[-1]
        gen = latest["generation"]

        best_costs = [r["best_cost"] for r in buf]
        mean_costs = [r["mean_cost"] for r in buf]
        cap_rates = [r["capture_rate"] for r in buf]
        diversities = [r["population_diversity"] for r in buf]

        lines = []
        lines.append(f"Best cost  {_format_cost(latest['best_cost']):>10s}  {_sparkline(best_costs)}")
        lines.append(f"Mean cost  {_format_cost(latest['mean_cost']):>10s}  {_sparkline(mean_costs)}")
        lines.append(f"Capture    {latest['capture_rate']:>9.0%}  {_sparkline(cap_rates)}")
        lines.append(f"Diversity  {latest['population_diversity']:>9.2f}  {_sparkline(diversities)}")
        lines.append("")

        # Progress bar
        progress = gen / self._n_gens
        bar_width = 40
        filled = int(progress * bar_width)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        eta_str = ""
        if self._start_time is not None and progress > 0:
            elapsed = time.monotonic() - self._start_time
            remaining = elapsed / progress * (1 - progress)
            mins, secs = divmod(int(remaining), 60)
            eta_str = f"  ETA {mins}m {secs:02d}s"
        lines.append(f"{bar}  {progress:.0%}{eta_str}")
        lines.append("")

        # Stagnation
        improvements = [i for i, r in enumerate(buf) if r["improvement"]]
        if improvements:
            last_imp_gen = buf[improvements[-1]]["generation"]
            stag = gen - last_imp_gen
            if stag > 0:
                lines.append(f"Stagnant for {stag} gens \u00b7 Last improvement: gen {last_imp_gen}")
        else:
            lines.append("No improvement yet")

        # Best params
        params = latest.get("best_params")
        if params is not None:
            param_str = ", ".join(f"{k}: {v:.4g}" for k, v in params.items())
            lines.append(f"Best params: {{{param_str}}}")

        title = f"{self._scheme} \u00b7 Run {current_run + 1}/{self._n_runs} \u00b7 Gen {gen}/{self._n_gens}"
        return Panel(Text("\n".join(lines)), title=title)

    def update(self, logger: TrainingLogger, current_run: int) -> None:
        """Update the live display with current logger state."""
        if self._live is None:
            return
        panel = self._build_panel(logger, current_run)
        self._live.update(panel)

    def __enter__(self) -> LiveDisplay:
        from rich.live import Live

        self._live = Live(refresh_per_second=2)
        self._live.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None


def create_display(scheme: str, n_runs: int, n_generations: int, *, enabled: bool = True) -> LiveDisplay | NoopDisplay:
    """Factory: returns LiveDisplay if enabled and terminal is interactive, else NoopDisplay."""
    if not enabled or not sys.stdout.isatty():
        return NoopDisplay()
    return LiveDisplay(scheme=scheme, n_runs=n_runs, n_generations=n_generations)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_display.py -v`
Expected: All PASS.

- [ ] **Step 5: Run linter**

Run: `./lint_code.sh`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/display.py tests/test_training_display.py
git commit -m "feat: add LiveDisplay with Rich sparklines and NoopDisplay fallback"
```

---

## Chunk 4: Integrate logger + display into train.py

### Task 5: Wire TrainingLogger and LiveDisplay into train.py

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

- [ ] **Step 1: Add `--no-tui` CLI argument**

In `train.py`, in the `if __name__ == "__main__"` block, add after the `--guidance` argument:

```python
    parser.add_argument("--no-tui", action="store_true", help="Disable Rich TUI (use plain-text output)")
```

And pass it to `train()` — add `no_tui: bool = False` parameter to the `train()` function signature:

```python
def train(
    config: TrainingConfig | None = None,
    seed: int | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    checkpoint_interval: int = 10,
    resume_dir: str | Path | None = None,
    no_tui: bool = False,
) -> dict:
```

- [ ] **Step 2: Add imports and config_hash computation**

At the top of `train.py`, add:

```python
import hashlib
from collections.abc import Callable
```

(numpy and numpy.typing are already imported.)

At the top of the `train()` function body, after `save_dir.mkdir(...)`, add:

```python
    # Compute config hash for experiment grouping
    config_hash = hashlib.sha256(repr(config).encode()).hexdigest()[:12]
```

- [ ] **Step 3: Add logger + display instantiation**

After the resume block and before the main `for run in range(...)` loop, add:

```python
    from aerocapture.training.display import create_display
    from aerocapture.training.logger import TrainingLogger

    display = create_display(
        scheme=config.guidance_type,
        n_runs=config.ga.n_runs,
        n_generations=config.ga.n_gen,
        enabled=not no_tui and verbose,
    )
```

- [ ] **Step 4: Add decode_fn setup inside the run loop**

Inside the `for run in range(...)` loop, after the population initialization block and before the `for gen in range(...)` loop, add:

```python
        # Set up decode function for logger (typed for mypy disallow_untyped_defs)
        decode_fn: Callable[[npt.NDArray[np.int8]], dict[str, float]] | None
        if config.guidance_type == "neural_network":
            decode_fn = None
        else:
            def _decode(chrom: npt.NDArray[np.int8]) -> dict[str, float]:
                return decode_params_from_chromosome(chrom, config)
            decode_fn = _decode

        logger = TrainingLogger(
            scheme=config.guidance_type,
            run=run,
            output_dir=save_dir,
            config_hash=config_hash,
        )
```

- [ ] **Step 5: Add logger call and display update in generation loop**

After the migration call (line ~322) and the `gen_best_costs.append(...)` line, add:

```python
            # Log metrics
            logger.log_generation(
                gen + 1,
                populations,
                all_costs,
                best_overall_chrom if best_overall_chrom is not None else populations[0][0],
                decode_fn,
            )
            display.update(logger, current_run=run)
```

- [ ] **Step 6: Wrap run loop body in display context manager**

Wrap the `for run in range(...)` loop with:

```python
    with display:
        for run in range(start_run, config.ga.n_runs):
            ...  # existing loop body
            logger.close()
```

Add `logger.close()` at the end of each run (after `cost_history.extend(gen_best_costs)`).

- [ ] **Step 7: Pass `--no-tui` from CLI**

In the `if __name__ == "__main__"` block, update the `train()` call:

```python
    result = train(cfg, seed=args.seed, cwd=cwd, resume_dir=args.resume, no_tui=args.no_tui)
```

- [ ] **Step 8: Write train.py integration test**

Create `tests/test_training_integration.py`:

```python
"""Integration test: verify TrainingLogger is called correctly by train.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from tests.fixtures.factories import make_training_config


class TestTrainLoggerIntegration:
    def test_logger_called_once_per_generation(self, tmp_path: Path) -> None:
        """Verify log_generation is called once per gen, after tournament, before checkpoint."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 2
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.save_dir = str(tmp_path)

        call_log: list[int] = []

        original_log_gen = None

        with (
            patch("aerocapture.training.train.TrainingLogger") as MockLogger,
            patch("aerocapture.training.train.evaluate_chromosome", return_value=(100.0, None)),
            patch("aerocapture.training.train.create_initial_population") as mock_init,
        ):
            mock_logger_instance = MagicMock()
            mock_logger_instance.buffer = []
            MockLogger.return_value = mock_logger_instance

            rng = np.random.default_rng(0)
            pop = rng.integers(0, 2, size=(4, 112), dtype=np.int8)
            costs = np.array([100.0, 200.0, 300.0, 400.0])
            mock_init.return_value = (pop, costs)

            from aerocapture.training.train import train
            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

            # log_generation should be called n_gen times
            assert mock_logger_instance.log_generation.call_count == 2
            # close should be called once per run
            assert mock_logger_instance.close.call_count == 1
```

- [ ] **Step 9: Run existing tests + integration test**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All tests pass.

- [ ] **Step 10: Run linter**

Run: `./lint_code.sh`
Expected: No errors.

- [ ] **Step 11: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_training_integration.py
git commit -m "feat: integrate TrainingLogger and LiveDisplay into training loop"
```

---

## Chunk 5: Post-training reports (report.py)

### Task 6: Implement `report.py` — single-run reports

**Files:**
- Create: `src/python/aerocapture/training/report.py`
- Create: `tests/test_training_report.py`

- [ ] **Step 1: Create test fixture JSONL data**

Add a fixture factory to `tests/test_training_report.py`:

```python
"""Tests for training report generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerocapture.training.report import generate_single_report, load_run_data


def _write_fixture_jsonl(path: Path, n_gens: int = 20) -> Path:
    """Write a synthetic JSONL file for testing."""
    jsonl_path = path / "equilibrium_glide" / "run_000_20260311T120000.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    best = 1e5
    with open(jsonl_path, "w") as f:
        for gen in range(1, n_gens + 1):
            best = best * 0.9  # Improving cost
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": best,
                "mean_cost": best * 3,
                "worst_cost": best * 10,
                "median_cost": best * 2,
                "std_cost": best * 1.5,
                "capture_rate": min(0.5 + gen * 0.025, 1.0),
                "population_diversity": max(0.5 - gen * 0.02, 0.05),
                "best_params": {"k_hdot_scale": 0.3, "v_ratio_threshold": 1.1},
                "improvement": gen <= 15,
                "scheme": "equilibrium_glide",
                "config_hash": "abc123",
            }
            f.write(json.dumps(record) + "\n")
    return jsonl_path.parent


class TestLoadRunData:
    def test_loads_all_records(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        data = load_run_data(scheme_dir)
        assert len(data) == 20
        assert data[0]["generation"] == 1

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "empty_scheme"
        scheme_dir.mkdir()
        data = load_run_data(scheme_dir)
        assert data == []


class TestSingleReport:
    def test_generates_html_file(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        report_path = scheme_dir / "report.html"
        assert report_path.exists()
        content = report_path.read_text()
        assert "plotly" in content.lower()
        assert "convergence" in content.lower() or "Convergence" in content

    def test_report_contains_all_sections(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        # Check for key plot titles/div IDs
        assert "best_cost" in content or "Best" in content
        assert "diversity" in content.lower() or "Diversity" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `report.py` — data loading and single-run report**

```python
"""Generate self-contained Plotly HTML reports from training JSONL logs.

Usage:
    uv run python -m aerocapture.training.report training_output/equilibrium_glide/
    uv run python -m aerocapture.training.report --compare training_output/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aerocapture.training.metrics import convergence_speed, stagnation_count


def load_run_data(scheme_dir: Path) -> list[dict]:
    """Load all JSONL records from a scheme directory, sorted by generation."""
    records: list[dict] = []
    for jsonl_file in sorted(scheme_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    records.sort(key=lambda r: r["generation"])
    # Deduplicate: last-writer-wins for same generation
    seen: dict[int, int] = {}
    deduped: list[dict] = []
    for r in records:
        gen = r["generation"]
        if gen in seen:
            deduped[seen[gen]] = r
        else:
            seen[gen] = len(deduped)
            deduped.append(r)
    return deduped


def generate_single_report(scheme_dir: Path) -> None:
    """Generate a single-run HTML report from JSONL data."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    data = load_run_data(scheme_dir)
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

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Convergence (log scale)",
            "Population Diversity vs Best Cost",
            "Capture Rate (%)",
            "Cost Distribution",
            "Parameter Evolution",
            "Summary",
        ),
        specs=[[{}, {"secondary_y": True}], [{}, {}], [{}, {}]],
    )

    # 1. Convergence
    fig.add_trace(go.Scatter(x=gens, y=best_costs, name="Best", line={"color": "#2196F3"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=mean_costs, name="Mean", line={"color": "#FF9800", "dash": "dash"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=worst_costs, name="Worst", line={"color": "#F44336", "dash": "dot"}), row=1, col=1)
    # Mark improvement generations
    imp_gens = [r["generation"] for r in data if r["improvement"]]
    imp_costs = [r["best_cost"] for r in data if r["improvement"]]
    fig.add_trace(go.Scatter(x=imp_gens, y=imp_costs, mode="markers", name="Improvement", marker={"color": "#4CAF50", "size": 6}), row=1, col=1)
    fig.update_yaxes(type="log", title_text="Cost", row=1, col=1)

    # 2. Diversity + best cost overlay
    fig.add_trace(go.Scatter(x=gens, y=diversities, name="Diversity", line={"color": "#9C27B0"}), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=gens, y=best_costs, name="Best Cost", line={"color": "#2196F3", "dash": "dot"}), row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Diversity", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Best Cost", type="log", row=1, col=2, secondary_y=True)

    # 3. Capture rate
    fig.add_trace(go.Scatter(x=gens, y=cap_rates, name="Capture %", line={"color": "#4CAF50"}, fill="tozeroy"), row=2, col=1)
    fig.update_yaxes(title_text="Capture Rate (%)", range=[0, 105], row=2, col=1)

    # 4. Cost distribution (box plots sampled every N gens)
    n_boxes = min(10, len(data))
    step = max(1, len(data) // n_boxes)
    for i in range(0, len(data), step):
        r = data[i]
        # Use the stats we have to create a simple box representation
        fig.add_trace(go.Box(
            y=[r["best_cost"], r["median_cost"], r["mean_cost"], r["worst_cost"]],
            name=f"Gen {r['generation']}",
            showlegend=False,
        ), row=2, col=2)
    fig.update_yaxes(type="log", title_text="Cost", row=2, col=2)

    # 5. Parameter evolution
    first_params = data[0].get("best_params")
    if first_params is not None:
        for param_name in first_params:
            vals = [r["best_params"][param_name] for r in data if r.get("best_params")]
            param_gens = [r["generation"] for r in data if r.get("best_params")]
            fig.add_trace(go.Scatter(x=param_gens, y=vals, name=param_name), row=3, col=1)
    fig.update_yaxes(title_text="Parameter Value", row=3, col=1)

    # 6. Summary table
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
    fig.add_annotation(text=summary_text, xref="x6 domain", yref="y6 domain", x=0.5, y=0.5, showarrow=False, font={"size": 12}, align="left", row=3, col=2)

    fig.update_layout(height=1000, title_text=f"Training Report — {scheme}", showlegend=True)
    fig.update_xaxes(title_text="Generation", row=3, col=1)

    output_path = scheme_dir / "report.html"
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Report saved to {output_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_report.py -v`
Expected: All PASS.

- [ ] **Step 5: Run linter**

Run: `./lint_code.sh`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/report.py tests/test_training_report.py
git commit -m "feat: add single-run Plotly HTML training report generation"
```

---

### Task 7: Add comparison report and CLI

**Files:**
- Modify: `src/python/aerocapture/training/report.py`
- Modify: `tests/test_training_report.py`

- [ ] **Step 1: Write comparison report tests**

Add to `tests/test_training_report.py`:

```python
from aerocapture.training.report import generate_comparison_report


def _write_multi_scheme_fixtures(base_dir: Path) -> None:
    """Write fixture JSONL for two schemes."""
    _write_fixture_jsonl(base_dir, n_gens=10)  # equilibrium_glide
    # Add a second scheme
    ftc_dir = base_dir / "ftc"
    ftc_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = ftc_dir / "run_000_20260311T120000.jsonl"
    best = 2e5
    with open(jsonl_path, "w") as f:
        for gen in range(1, 11):
            best = best * 0.85
            record = {
                "generation": gen, "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": best, "mean_cost": best * 4,
                "worst_cost": best * 12, "median_cost": best * 2.5,
                "std_cost": best * 2, "capture_rate": 0.6 + gen * 0.04,
                "population_diversity": 0.4 - gen * 0.03,
                "best_params": {"capture_damping": 0.7},
                "improvement": gen <= 8, "scheme": "ftc", "config_hash": "def456",
            }
            f.write(json.dumps(record) + "\n")


class TestComparisonReport:
    def test_generates_comparison_html(self, tmp_path: Path) -> None:
        _write_multi_scheme_fixtures(tmp_path)
        generate_comparison_report(tmp_path)
        report_path = tmp_path / "comparison_report.html"
        assert report_path.exists()
        content = report_path.read_text()
        assert "plotly" in content.lower()

    def test_filters_by_scheme(self, tmp_path: Path) -> None:
        _write_multi_scheme_fixtures(tmp_path)
        generate_comparison_report(tmp_path, schemes=["ftc"])
        content = (tmp_path / "comparison_report.html").read_text()
        assert "ftc" in content.lower() or "FTC" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_report.py::TestComparisonReport -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement comparison report**

Add to `report.py`:

```python
_SCHEME_LABELS = {
    "ftc": "FTC",
    "neural_network": "Neural Net",
    "equilibrium_glide": "Eq. Glide",
    "energy_controller": "Energy Ctrl",
    "pred_guid": "PredGuid",
    "fnpag": "FNPAG",
}

_SCHEME_COLORS = {
    "ftc": "#2196F3",
    "neural_network": "#FF9800",
    "equilibrium_glide": "#4CAF50",
    "energy_controller": "#9C27B0",
    "pred_guid": "#F44336",
    "fnpag": "#795548",
}


def generate_comparison_report(
    base_dir: Path,
    schemes: list[str] | None = None,
    after: str | None = None,
) -> None:
    """Generate a cross-scheme comparison HTML report."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    scheme_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir() and list(d.glob("*.jsonl")))

    if schemes:
        scheme_dirs = [d for d in scheme_dirs if d.name in schemes]

    if not scheme_dirs:
        print(f"No JSONL data found in subdirectories of {base_dir}")
        return

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cross-Scheme Convergence", "Final Metrics"),
        specs=[[{}], [{}]],
        row_heights=[0.65, 0.35],
    )

    summary_rows: list[list[str]] = []

    for scheme_dir in scheme_dirs:
        scheme_name = scheme_dir.name
        data = load_run_data(scheme_dir)
        if not data:
            continue

        # Filter by date if requested
        if after:
            data = [r for r in data if r.get("timestamp", "") >= after]
            if not data:
                continue

        gens = [r["generation"] for r in data]
        best_costs = [r["best_cost"] for r in data]
        color = _SCHEME_COLORS.get(scheme_name, "#666666")
        label = _SCHEME_LABELS.get(scheme_name, scheme_name)

        fig.add_trace(go.Scatter(x=gens, y=best_costs, name=label, line={"color": color}), row=1, col=1)

        cost_history = [r["best_cost"] for r in data]
        conv = convergence_speed(cost_history)
        cap = data[-1].get("capture_rate", 0) * 100

        summary_rows.append([label, f"{best_costs[-1]:.2e}", str(len(data)), f"{cap:.0f}%", str(conv)])

    fig.update_yaxes(type="log", title_text="Best Cost", row=1, col=1)
    fig.update_xaxes(title_text="Generation", row=1, col=1)

    # Summary table
    header = ["Scheme", "Best Cost", "Generations", "Capture %", "Conv. Speed"]
    fig.add_trace(go.Table(
        header={"values": header, "fill_color": "#2196F3", "font_color": "white", "align": "center"},
        cells={"values": list(zip(*summary_rows)) if summary_rows else [[] for _ in header], "align": "center"},
    ), row=2, col=1)

    fig.update_layout(height=800, title_text="Training Comparison Report")

    output_path = base_dir / "comparison_report.html"
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Comparison report saved to {output_path}")
```

- [ ] **Step 4: Add CLI `__main__` block**

Add at the end of `report.py`:

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate training reports from JSONL logs")
    parser.add_argument("path", type=str, help="Path to scheme directory (single) or training_output/ (comparison)")
    parser.add_argument("--compare", action="store_true", help="Generate cross-scheme comparison report")
    parser.add_argument("--schemes", nargs="*", help="Filter by scheme names (comparison mode)")
    parser.add_argument("--after", type=str, default=None, help="Filter runs after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    if args.compare:
        generate_comparison_report(path, schemes=args.schemes, after=args.after)
    else:
        generate_single_report(path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all report tests**

Run: `uv run pytest tests/test_training_report.py -v`
Expected: All PASS.

- [ ] **Step 6: Run linter**

Run: `./lint_code.sh`
Expected: No errors.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/report.py tests/test_training_report.py
git commit -m "feat: add comparison report and CLI for training report generation"
```

---

## Chunk 6: Cleanup and finalization

### Task 8: Remove legacy training data

**Files:**
- Delete: contents of `training_output/`

- [ ] **Step 1: Remove legacy training data**

```bash
rm -rf training_output/*
```

- [ ] **Step 2: Ensure .gitignore covers training_output contents**

Check that `training_output/` is in `.gitignore`. If legacy files were tracked, untrack them:

```bash
git rm -r --cached training_output/ 2>/dev/null || true
```

- [ ] **Step 3: Commit**

```bash
git add -A training_output/ .gitignore
git commit -m "chore: remove legacy training data"
```

---

### Task 9: Run full test suite and lint

- [ ] **Step 1: Run all Python tests**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All tests pass.

- [ ] **Step 2: Run linter and type checker**

Run: `./lint_code.sh`
Expected: No errors.

- [ ] **Step 3: Run Rust tests**

Run: `./check_all.sh`
Expected: All pass.

---

### Task 10: Smart commit

- [ ] **Step 1: Use smart-commit skill**

Invoke the `smart-commit` skill to commit all remaining changes with updated documentation (CLAUDE.md, README.md synced with codebase state).
