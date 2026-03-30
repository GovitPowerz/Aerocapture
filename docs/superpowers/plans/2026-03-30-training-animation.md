# Training Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standalone CLI that generates GIF animations of corridor/trajectory evolution during GA training by replaying checkpoints with MC re-evaluation.

**Architecture:** Single module `animate.py` in the training package. Discovers checkpoints, reconstructs corridor state and best chromosome at each generation, re-runs MC via PyO3, renders 2x2 frames (corridor, inclination, bank angle, cost CDF), composites into GIF via matplotlib PillowWriter. Reuses `charts.py` visual language (colors, classification, spaghetti drawing).

**Tech Stack:** matplotlib (PillowWriter), aerocapture_rs (PyO3), numpy, Rich (progress bar)

---

### Task 1: Delete obsolete `plot_corridor_animation.py`

**Files:**
- Delete: `src/python/aerocapture/plotting/plot_corridor_animation.py`

- [ ] **Step 1: Delete the file**

```bash
rm src/python/aerocapture/plotting/plot_corridor_animation.py
```

- [ ] **Step 2: Verify no imports reference it**

Run: `rg plot_corridor_animation src/`
Expected: No matches

- [ ] **Step 3: Commit**

```bash
git add -u src/python/aerocapture/plotting/plot_corridor_animation.py
git commit -m "remove obsolete plot_corridor_animation.py (replaced by training.animate)"
```

---

### Task 2: Scaffold `animate.py` with checkpoint discovery and CLI

**Files:**
- Create: `src/python/aerocapture/training/animate.py`
- Create: `tests/test_animate.py`

- [ ] **Step 1: Write the test for checkpoint discovery**

```python
# tests/test_animate.py
"""Tests for training animation generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestDiscoverCheckpoints:
    @pytest.fixture()
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """Create a directory with 5 fake checkpoints."""
        d = tmp_path / "piecewise_constant"
        d.mkdir()
        for gen in [0, 10, 20, 30, 40]:
            prefix = f"checkpoint_r000_g{gen:05d}"
            meta = {"run": 0, "generation": gen, "best_cost": 100.0 - gen, "cost_history": [100.0 - g for g in range(0, gen + 1, 10)]}
            (d / f"{prefix}.json").write_text(json.dumps(meta))
            np.savez_compressed(
                d / f"{prefix}.npz",
                pop_0=np.zeros((5, 10), dtype=np.int8),
                costs_0=np.full(5, 100.0 - gen),
                n_subpops=np.array([1]),
                best_chromosome=np.zeros(10, dtype=np.int8),
            )
        return d

    def test_discovers_all_checkpoints_sorted(self, checkpoint_dir: Path) -> None:
        from aerocapture.training.animate import _discover_checkpoints

        result = _discover_checkpoints(checkpoint_dir, every=1)
        assert len(result) == 5
        assert [c["generation"] for c in result] == [0, 10, 20, 30, 40]

    def test_every_filters_checkpoints(self, checkpoint_dir: Path) -> None:
        from aerocapture.training.animate import _discover_checkpoints

        result = _discover_checkpoints(checkpoint_dir, every=2)
        # Every=2 means take every 2nd: indices 0, 2, 4 → gens 0, 20, 40
        assert len(result) == 3
        assert [c["generation"] for c in result] == [0, 20, 40]

    def test_always_includes_last_checkpoint(self, checkpoint_dir: Path) -> None:
        from aerocapture.training.animate import _discover_checkpoints

        result = _discover_checkpoints(checkpoint_dir, every=3)
        # Indices 0, 3 → gens 0, 30. Last (40) always included.
        assert result[-1]["generation"] == 40

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        from aerocapture.training.animate import _discover_checkpoints

        result = _discover_checkpoints(tmp_path, every=1)
        assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_animate.py -v`
Expected: FAIL with ImportError (module doesn't exist yet)

- [ ] **Step 3: Write checkpoint discovery and CLI scaffold**

```python
# src/python/aerocapture/training/animate.py
"""Generate GIF animations of corridor/trajectory evolution during GA training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt


def _discover_checkpoints(training_dir: Path, every: int) -> list[dict]:
    """Find and load checkpoint metadata, sorted by generation.

    Args:
        training_dir: Path to scheme training output directory.
        every: Sample every Nth checkpoint. Last checkpoint is always included.

    Returns:
        List of dicts with keys: generation, best_cost, cost_history,
        json_path, npz_path, costs (population costs array).
    """
    pattern = re.compile(r"checkpoint_r\d+_g(\d+)\.json$")
    found: list[tuple[int, Path]] = []
    for p in training_dir.glob("checkpoint_r*_g*.json"):
        m = pattern.search(p.name)
        if m:
            found.append((int(m.group(1)), p))

    found.sort(key=lambda x: x[0])
    if not found:
        return []

    # Apply --every filter, always keeping the last checkpoint
    indices = list(range(0, len(found), every))
    if indices[-1] != len(found) - 1:
        indices.append(len(found) - 1)

    result: list[dict] = []
    for idx in indices:
        gen, json_path = found[idx]
        npz_path = json_path.with_suffix(".npz")
        with open(json_path) as f:
            meta = json.load(f)

        # Load population costs from npz
        data = np.load(npz_path, allow_pickle=False)
        costs_arrays: list[npt.NDArray[np.float64]] = []
        n_subpops = int(data["n_subpops"][0]) if "n_subpops" in data else 1
        for k in range(n_subpops):
            key = f"costs_{k}"
            if key in data:
                costs_arrays.append(data[key])
        all_costs = np.concatenate(costs_arrays) if costs_arrays else np.array([])

        best_chrom = data["best_chromosome"] if "best_chromosome" in data else None

        result.append({
            "generation": gen,
            "best_cost": meta["best_cost"],
            "cost_history": meta.get("cost_history", []),
            "json_path": json_path,
            "npz_path": npz_path,
            "costs": all_costs,
            "best_chromosome": best_chrom,
            "npz_data": dict(data),
        })

    return result


def generate_animation(
    training_dir: Path,
    toml_path: Path,
    n_sims: int = 100,
    fps: int = 4,
    output: Path | None = None,
    every: int = 1,
) -> Path:
    """Generate a GIF animation of training evolution.

    Args:
        training_dir: Path to scheme training output directory.
        toml_path: Path to training TOML config.
        n_sims: Number of MC simulations per frame.
        fps: Frames per second in output GIF.
        output: Output GIF path. Defaults to training_dir/animation.gif.
        every: Use every Nth checkpoint (1 = all).

    Returns:
        Path to generated GIF file.
    """
    try:
        import aerocapture_rs as _aero_rs  # noqa: F841
    except ImportError:
        msg = (
            "aerocapture_rs (PyO3) is required for animation generation.\n"
            "Build it with: cd src/rust/aerocapture-py && maturin develop --release"
        )
        raise RuntimeError(msg)

    if output is None:
        output = training_dir / "animation.gif"

    checkpoints = _discover_checkpoints(training_dir, every)
    if not checkpoints:
        msg = f"No checkpoints found in {training_dir}"
        raise FileNotFoundError(msg)

    # TODO: Task 3 — compute axis ranges from final checkpoint
    # TODO: Task 4 — render frames and composite GIF
    raise NotImplementedError("Animation rendering not yet implemented")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GIF animation of training corridor/trajectory evolution",
    )
    parser.add_argument("training_dir", type=Path, help="Path to scheme training output directory")
    parser.add_argument("--toml", type=Path, required=True, help="Training TOML config path")
    parser.add_argument("--n-sims", type=int, default=100, help="MC simulations per frame (default: 100)")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second (default: 4)")
    parser.add_argument("--output", type=Path, default=None, help="Output GIF path (default: <training_dir>/animation.gif)")
    parser.add_argument("--every", type=int, default=1, help="Use every Nth checkpoint (default: 1 = all)")
    args = parser.parse_args()

    result = generate_animation(
        training_dir=args.training_dir,
        toml_path=args.toml,
        n_sims=args.n_sims,
        fps=args.fps,
        output=args.output,
        every=args.every,
    )
    print(f"Animation saved to {result}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_animate.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/animate.py tests/test_animate.py
git commit -m "add animate.py scaffold with checkpoint discovery and CLI"
```

---

### Task 3: Implement MC re-evaluation and axis range computation

**Files:**
- Modify: `src/python/aerocapture/training/animate.py`
- Modify: `tests/test_animate.py`

This task adds the functions that: (a) decode a checkpoint's best chromosome into TOML overrides, (b) run MC via PyO3 to get trajectories, and (c) compute global axis ranges from the final checkpoint's MC eval.

- [ ] **Step 1: Write tests for `_build_overrides` and `_compute_axis_ranges`**

Add to `tests/test_animate.py`:

```python
class TestBuildOverrides:
    def test_builds_dot_path_overrides(self) -> None:
        from aerocapture.training.animate import _build_overrides

        params = {"gain_kp": 0.5, "gain_kd": 0.1}
        overrides = _build_overrides("equilibrium_glide", params, n_sims=50)
        assert overrides["guidance.equilibrium_glide.gain_kp"] == 0.5
        assert overrides["guidance.equilibrium_glide.gain_kd"] == 0.1
        assert overrides["guidance.type"] == "equilibrium_glide"
        assert overrides["simulation.n_sims"] == 50

    def test_lateral_params_go_to_lateral_section(self) -> None:
        from aerocapture.training.animate import _build_overrides

        params = {"gain_kp": 0.5, "lateral.corridor_slope": 100.0}
        overrides = _build_overrides("equilibrium_glide", params, n_sims=50)
        assert overrides["guidance.lateral.corridor_slope"] == 100.0
        assert "guidance.equilibrium_glide.lateral.corridor_slope" not in overrides


class TestComputeAxisRanges:
    def test_returns_dict_with_expected_keys(self) -> None:
        from aerocapture.training.animate import _compute_axis_ranges

        # Create fake trajectory data (12-column format)
        rng = np.random.default_rng(42)
        trajectories = [rng.standard_normal((50, 12)) for _ in range(10)]
        costs = rng.uniform(50, 200, size=30)

        ranges = _compute_axis_ranges(trajectories, costs)
        for key in ("energy_min", "energy_max", "pdyn_min", "pdyn_max", "incl_min", "incl_max", "bank_min", "bank_max", "cost_max"):
            assert key in ranges

    def test_ranges_have_margin(self) -> None:
        from aerocapture.training.animate import _compute_axis_ranges

        # All trajectories have energy in [0, 1], pdyn in [0, 100]
        traj = np.zeros((50, 12))
        traj[:, 8] = np.linspace(0, 1, 50)  # energy
        traj[:, 9] = np.linspace(0, 100, 50)  # pdyn
        traj[:, 10] = np.linspace(-90, 90, 50)  # bank
        traj[:, 11] = np.linspace(20, 30, 50)  # inclination

        ranges = _compute_axis_ranges([traj], np.array([100.0]))
        # Ranges should be slightly wider than data (5% margin)
        assert ranges["energy_min"] < 0
        assert ranges["energy_max"] > 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_animate.py::TestBuildOverrides tests/test_animate.py::TestComputeAxisRanges -v`
Expected: FAIL (functions don't exist yet)

- [ ] **Step 3: Implement `_build_overrides` and `_compute_axis_ranges`**

Add to `animate.py`, below `_discover_checkpoints`:

```python
def _build_overrides(guidance_type: str, params: dict[str, float], n_sims: int) -> dict[str, object]:
    """Convert decoded parameters to dot-path TOML overrides for run_mc.

    Lateral params (prefixed ``lateral.``) go to ``guidance.lateral.*``.
    Scheme params go to ``guidance.<section>.*``.
    """
    from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

    section = GUIDANCE_TOML_SECTIONS[guidance_type]
    overrides: dict[str, object] = {}
    for k, v in params.items():
        if k.startswith("lateral."):
            overrides[f"guidance.{k}"] = v
        else:
            overrides[f"guidance.{section}.{k}"] = v
    overrides["guidance.type"] = guidance_type
    overrides["simulation.n_sims"] = n_sims
    return overrides


def _compute_axis_ranges(
    trajectories: list[npt.NDArray[np.float64]],
    costs: npt.NDArray[np.float64],
    margin: float = 0.05,
) -> dict[str, float]:
    """Compute fixed axis ranges from trajectories and costs.

    Adds a relative margin to each dimension so frames don't clip data.
    """
    all_energy = np.concatenate([t[:, 8] for t in trajectories])
    all_pdyn = np.concatenate([t[:, 9] for t in trajectories])
    all_bank = np.concatenate([t[:, 10] for t in trajectories])
    all_incl = np.concatenate([t[:, 11] for t in trajectories])

    def _padded(arr: npt.NDArray[np.float64]) -> tuple[float, float]:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        span = hi - lo if hi > lo else 1.0
        return lo - margin * span, hi + margin * span

    e_min, e_max = _padded(all_energy)
    p_min, p_max = _padded(all_pdyn)
    b_min, b_max = _padded(all_bank)
    i_min, i_max = _padded(all_incl)

    return {
        "energy_min": e_min,
        "energy_max": e_max,
        "pdyn_min": p_min,
        "pdyn_max": p_max,
        "bank_min": b_min,
        "bank_max": b_max,
        "incl_min": i_min,
        "incl_max": i_max,
        "cost_max": float(np.max(costs)) * 1.1,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_animate.py::TestBuildOverrides tests/test_animate.py::TestComputeAxisRanges -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/animate.py tests/test_animate.py
git commit -m "add override builder and axis range computation for animation"
```

---

### Task 4: Implement frame rendering (2x2 panels)

**Files:**
- Modify: `src/python/aerocapture/training/animate.py`
- Modify: `tests/test_animate.py`

- [ ] **Step 1: Write test for frame rendering**

Add to `tests/test_animate.py`:

```python
import matplotlib

matplotlib.use("Agg")


class TestRenderFrame:
    def test_returns_figure_with_4_axes(self) -> None:
        from aerocapture.training.animate import _render_frame

        rng = np.random.default_rng(42)
        trajectories = [rng.standard_normal((50, 12)).astype(np.float64) for _ in range(10)]
        final_records = rng.standard_normal((10, 52)).astype(np.float64)
        # Set ifinal=3 and ecc<1 for some to be "captured"
        final_records[:5, 31] = 3.0  # ifinal
        final_records[:5, 9] = 0.5  # ecc < 1
        final_records[5:, 31] = 1.0  # not captured

        costs = rng.uniform(50, 200, size=30)
        axis_ranges = {
            "energy_min": -2.0, "energy_max": 2.0,
            "pdyn_min": -2.0, "pdyn_max": 2.0,
            "incl_min": -2.0, "incl_max": 2.0,
            "bank_min": -2.0, "bank_max": 2.0,
            "cost_max": 250.0,
        }

        fig = _render_frame(
            generation=42,
            best_cost=55.0,
            capture_rate=0.8,
            trajectories=trajectories,
            final_records=final_records,
            costs=costs,
            corridor_data=None,
            axis_ranges=axis_ranges,
        )
        assert fig is not None
        axes = fig.get_axes()
        # 4 main panels + 1 twinx on cost CDF = 5 axes
        assert len(axes) == 5
        import matplotlib.pyplot as plt
        plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_animate.py::TestRenderFrame -v`
Expected: FAIL (function doesn't exist)

- [ ] **Step 3: Implement `_render_frame` and panel helpers**

Add to `animate.py`:

```python
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import from charts.py for consistent styling
from aerocapture.training.charts import (
    COLOR_CAPTURE,
    COLOR_CONSTRAINED,
    COLOR_HYPERBOLIC,
    _draw_spaghetti,
    classify_trajectories,
)


def _render_corridor_panel(
    ax: plt.Axes,
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    corridor_data: dict[str, npt.NDArray[np.float64]] | None,
    axis_ranges: dict[str, float],
) -> None:
    """Render the corridor (energy vs dynamic pressure) panel."""
    # Draw corridor zone fills if available
    if corridor_data is not None:
        e_bins = corridor_data["energy_bins"]
        crash_pdyn = corridor_data["envelope_crash_pdyn"]
        restricted_max = corridor_data["envelope_restricted_max_pdyn"]
        restricted_min = corridor_data["envelope_restricted_min_pdyn"]
        capture_pdyn = corridor_data["envelope_capture_pdyn"]

        ax.fill_between(e_bins, restricted_max, crash_pdyn, color=COLOR_HYPERBOLIC, alpha=0.15)
        ax.fill_between(e_bins, restricted_max, restricted_min, color="white", alpha=0.6)
        ax.fill_between(e_bins, restricted_min, capture_pdyn, color="#cccccc", alpha=0.3)
        ax.fill_between(e_bins, capture_pdyn, 0, color=COLOR_HYPERBOLIC, alpha=0.15)

    # col 8 = energy, col 9 = pdyn
    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=9)
    ax.set_xlim(axis_ranges["energy_min"], axis_ranges["energy_max"])
    ax.set_ylim(axis_ranges["pdyn_min"], axis_ranges["pdyn_max"])
    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Dynamic pressure (Pa)")
    ax.set_title("Corridor")


def _render_inclination_panel(
    ax: plt.Axes,
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    axis_ranges: dict[str, float],
) -> None:
    """Render the inclination (energy vs inclination) panel."""
    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=11)
    ax.set_xlim(axis_ranges["energy_min"], axis_ranges["energy_max"])
    ax.set_ylim(axis_ranges["incl_min"], axis_ranges["incl_max"])
    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Inclination (deg)")
    ax.set_title("Inclination")


def _render_bank_panel(
    ax: plt.Axes,
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    axis_ranges: dict[str, float],
) -> None:
    """Render the bank angle (energy vs bank) panel."""
    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=10)
    ax.set_xlim(axis_ranges["energy_min"], axis_ranges["energy_max"])
    ax.set_ylim(axis_ranges["bank_min"], axis_ranges["bank_max"])
    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Bank angle (deg)")
    ax.set_title("Bank angle")


def _render_cost_panel(
    ax: plt.Axes,
    costs: npt.NDArray[np.float64],
    best_cost: float,
    axis_ranges: dict[str, float],
) -> None:
    """Render the cost CDF panel (histogram + cumulative)."""
    finite = costs[np.isfinite(costs)]
    if len(finite) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.hist(finite, bins=30, density=True, alpha=0.5, color=COLOR_CAPTURE, label="Population")

    # ECDF overlay on twin axis
    ax2 = ax.twinx()
    sorted_costs = np.sort(finite)
    ecdf = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)
    ax2.plot(sorted_costs, ecdf, color=COLOR_CONSTRAINED, linewidth=1.5, label="ECDF")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("CDF")

    # Best cost marker
    ax.axvline(best_cost, color=COLOR_CAPTURE, linestyle="--", linewidth=1, label=f"Best: {best_cost:.1f}")

    ax.set_xlim(0, axis_ranges["cost_max"])
    ax.set_xlabel("Cost")
    ax.set_ylabel("Density")
    ax.set_title("Cost distribution")
    ax.legend(fontsize=7, loc="upper right")


def _render_frame(
    generation: int,
    best_cost: float,
    capture_rate: float,
    trajectories: list[npt.NDArray[np.float64]],
    final_records: npt.NDArray[np.float64],
    costs: npt.NDArray[np.float64],
    corridor_data: dict[str, npt.NDArray[np.float64]] | None,
    axis_ranges: dict[str, float],
    heat_flux_limit: float | None = None,
    g_load_limit: float | None = None,
) -> Figure:
    """Render a single animation frame (2x2 grid).

    Returns:
        Matplotlib Figure ready for GIF writer.
    """
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9, rc={"axes.facecolor": "#f5f5f5"})

    traj_class = classify_trajectories(final_records, heat_flux_limit=heat_flux_limit, g_load_limit=g_load_limit)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Generation {generation}  |  Best cost: {best_cost:.1f}  |  Capture: {capture_rate:.0%}", fontsize=13, fontweight="bold")

    _render_corridor_panel(axes[0, 0], trajectories, traj_class, corridor_data, axis_ranges)
    _render_inclination_panel(axes[0, 1], trajectories, traj_class, axis_ranges)
    _render_bank_panel(axes[1, 0], trajectories, traj_class, axis_ranges)
    _render_cost_panel(axes[1, 1], costs, best_cost, axis_ranges)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_animate.py::TestRenderFrame -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/animate.py tests/test_animate.py
git commit -m "add 2x2 frame rendering for training animation"
```

---

### Task 5: Wire up `generate_animation` — the main orchestrator

**Files:**
- Modify: `src/python/aerocapture/training/animate.py`
- Modify: `tests/test_animate.py`

- [ ] **Step 1: Write integration test for `generate_animation`**

Add to `tests/test_animate.py`:

```python
from unittest.mock import MagicMock, patch


class TestGenerateAnimation:
    def test_errors_without_pyo3(self, tmp_path: Path) -> None:
        from aerocapture.training.animate import generate_animation

        with patch.dict("sys.modules", {"aerocapture_rs": None}):
            with pytest.raises(RuntimeError, match="aerocapture_rs"):
                generate_animation(tmp_path, toml_path=tmp_path / "fake.toml")

    def test_errors_on_empty_dir(self, tmp_path: Path) -> None:
        from aerocapture.training.animate import generate_animation

        mock_aero = MagicMock()
        with patch.dict("sys.modules", {"aerocapture_rs": mock_aero}):
            with pytest.raises(FileNotFoundError, match="No checkpoints"):
                generate_animation(tmp_path, toml_path=tmp_path / "fake.toml")

    def test_generates_gif(self, tmp_path: Path) -> None:
        """End-to-end test with mocked PyO3 calls."""
        from aerocapture.training.animate import generate_animation

        # Create 2 fake checkpoints
        d = tmp_path / "scheme"
        d.mkdir()
        rng = np.random.default_rng(42)
        for gen in [0, 10]:
            prefix = f"checkpoint_r000_g{gen:05d}"
            meta = {"run": 0, "generation": gen, "best_cost": 100.0 - gen, "cost_history": [100.0]}
            (d / f"{prefix}.json").write_text(json.dumps(meta))
            np.savez_compressed(
                d / f"{prefix}.npz",
                pop_0=np.zeros((5, 80), dtype=np.int8),
                costs_0=rng.uniform(50, 150, size=5),
                n_subpops=np.array([1]),
                best_chromosome=np.zeros(80, dtype=np.int8),
            )

        # Create a fake TOML
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('[guidance]\ntype = "equilibrium_glide"\n')

        # Mock aerocapture_rs.run_mc to return fake data
        n_sims = 10
        mock_results = MagicMock()
        mock_results.final_records = rng.standard_normal((n_sims, 52)).astype(np.float64)
        # Set some as captured (ifinal=3, ecc<1)
        mock_results.final_records[:5, 31] = 3.0
        mock_results.final_records[:5, 9] = 0.5
        mock_results.final_records[5:, 31] = 1.0
        mock_results.trajectories = [rng.standard_normal((50, 12)).astype(np.float64) for _ in range(n_sims)]

        mock_aero = MagicMock()
        mock_aero.run_mc.return_value = mock_results

        with (
            patch.dict("sys.modules", {"aerocapture_rs": mock_aero}),
            patch("aerocapture.training.animate._load_pyo3", return_value=mock_aero),
            patch("aerocapture.training.animate._decode_and_build_overrides", return_value={"guidance.type": "equilibrium_glide", "simulation.n_sims": n_sims}),
        ):
            gif_path = generate_animation(d, toml_path=toml_path, n_sims=n_sims, fps=2)

        assert gif_path.exists()
        assert gif_path.suffix == ".gif"
        assert gif_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_animate.py::TestGenerateAnimation::test_generates_gif -v`
Expected: FAIL

- [ ] **Step 3: Implement `generate_animation` orchestrator**

Replace the placeholder `generate_animation` in `animate.py` with the full implementation:

```python
def _load_pyo3():  # type: ignore[no-untyped-def]
    """Import and return the aerocapture_rs module, or raise with a clear message."""
    try:
        import aerocapture_rs

        return aerocapture_rs
    except ImportError:
        msg = (
            "aerocapture_rs (PyO3) is required for animation generation.\n"
            "Build it with: cd src/rust/aerocapture-py && maturin develop --release"
        )
        raise RuntimeError(msg)


def _decode_and_build_overrides(
    best_chromosome: npt.NDArray[np.int8],
    guidance_type: str,
    toml_data: dict,
    n_sims: int,
) -> dict[str, object]:
    """Decode a checkpoint's best chromosome into TOML overrides.

    Handles both neural_network (writes JSON file, returns minimal overrides)
    and guidance-parameter schemes (returns full dot-path overrides).
    """
    if guidance_type == "neural_network":
        # NN requires writing weights to disk; return minimal overrides
        from aerocapture.training.config import TrainingConfig
        from aerocapture.training.evaluate import decode_direct, write_nn_json

        cfg = TrainingConfig()
        cfg.guidance_type = "neural_network"
        net = toml_data.get("network", {})
        if "layer_sizes" in net:
            cfg.network.layer_sizes = net["layer_sizes"]
        if "activations" in net:
            cfg.network.activations = net["activations"]
        cfg.sim.nn_param_file = toml_data.get("data", {}).get("neural_network", "data/neural_network/nn_model.json")

        weights = decode_direct(best_chromosome, cfg)
        nn_path = Path(cfg.sim.nn_param_file)
        write_nn_json(weights, cfg.network, nn_path)
        return {"simulation.n_sims": n_sims}

    from aerocapture.training.config import TrainingConfig
    from aerocapture.training.evaluate import decode_params_from_chromosome

    cfg = TrainingConfig()
    cfg.guidance_type = guidance_type
    params = decode_params_from_chromosome(best_chromosome, cfg)
    return _build_overrides(guidance_type, params, n_sims)


def _reconstruct_corridor(npz_data: dict) -> dict[str, npt.NDArray[np.float64]] | None:
    """Reconstruct corridor data from checkpoint npz, if available."""
    if "corridor_energy_bins" not in npz_data:
        return None

    from aerocapture.training.corridor import CorridorAccumulator

    acc = CorridorAccumulator.from_checkpoint(npz_data)
    return acc.to_corridor_data()


def generate_animation(
    training_dir: Path,
    toml_path: Path,
    n_sims: int = 100,
    fps: int = 4,
    output: Path | None = None,
    every: int = 1,
) -> Path:
    """Generate a GIF animation of training evolution.

    Args:
        training_dir: Path to scheme training output directory.
        toml_path: Path to training TOML config.
        n_sims: Number of MC simulations per frame.
        fps: Frames per second in output GIF.
        output: Output GIF path. Defaults to training_dir/animation.gif.
        every: Use every Nth checkpoint (1 = all).

    Returns:
        Path to generated GIF file.
    """
    aero_rs = _load_pyo3()

    if output is None:
        output = training_dir / "animation.gif"

    checkpoints = _discover_checkpoints(training_dir, every)
    if not checkpoints:
        msg = f"No checkpoints found in {training_dir}"
        raise FileNotFoundError(msg)

    # Load TOML for guidance type and constraint limits
    from aerocapture.training.toml_utils import load_toml_with_bases

    toml_data = load_toml_with_bases(toml_path)
    guidance_type = toml_data.get("guidance", {}).get("type")
    if guidance_type is None:
        msg = "TOML config must contain [guidance] type"
        raise ValueError(msg)

    constraints = toml_data.get("flight", {}).get("constraints", {})
    heat_flux_limit: float | None = constraints.get("max_heat_flux")
    g_load_limit: float | None = constraints.get("max_load_factor")

    toml_resolved = str(toml_path.resolve())

    # Step 1: Pre-compute axis ranges from the final checkpoint
    last = checkpoints[-1]
    last_overrides = _decode_and_build_overrides(last["best_chromosome"], guidance_type, toml_data, n_sims)
    last_results = aero_rs.run_mc(toml_path=toml_resolved, overrides=last_overrides, include_trajectories=True)
    all_costs_for_range = np.concatenate([c["costs"] for c in checkpoints])
    axis_ranges = _compute_axis_ranges(last_results.trajectories, all_costs_for_range)

    # Step 2: Render frames with progress bar
    try:
        from rich.progress import Progress

        progress_ctx = Progress()
    except ImportError:
        progress_ctx = None  # type: ignore[assignment]

    matplotlib.use("Agg")
    from matplotlib.animation import PillowWriter

    fig_placeholder = plt.figure()  # PillowWriter needs a figure to init
    writer = PillowWriter(fps=fps)
    writer.setup(fig_placeholder, str(output), dpi=100)
    plt.close(fig_placeholder)

    if progress_ctx is not None:
        progress_ctx.start()
        task_id = progress_ctx.add_task("Rendering frames", total=len(checkpoints))

    try:
        for ckpt in checkpoints:
            gen = ckpt["generation"]
            best_chrom = ckpt["best_chromosome"]
            if best_chrom is None:
                if progress_ctx is not None:
                    progress_ctx.advance(task_id)
                continue

            # Decode + run MC
            overrides = _decode_and_build_overrides(best_chrom, guidance_type, toml_data, n_sims)
            results = aero_rs.run_mc(toml_path=toml_resolved, overrides=overrides, include_trajectories=True)
            trajectories = results.trajectories
            final_records = results.final_records

            # Capture rate from this MC eval
            captured = (final_records[:, 31] == 3) & (final_records[:, 9] < 1.0)
            capture_rate = float(np.mean(captured))

            # Corridor from checkpoint
            corridor_data = _reconstruct_corridor(ckpt["npz_data"])

            # Render frame
            fig = _render_frame(
                generation=gen,
                best_cost=ckpt["best_cost"],
                capture_rate=capture_rate,
                trajectories=trajectories,
                final_records=final_records,
                costs=ckpt["costs"],
                corridor_data=corridor_data,
                axis_ranges=axis_ranges,
                heat_flux_limit=heat_flux_limit,
                g_load_limit=g_load_limit,
            )
            writer.grab_frame()  # grabs from current figure
            plt.close(fig)

            if progress_ctx is not None:
                progress_ctx.advance(task_id)
    finally:
        writer.finish()
        if progress_ctx is not None:
            progress_ctx.stop()

    return output
```

**Note on PillowWriter**: `PillowWriter.grab_frame()` grabs from `plt.gcf()`. Since `_render_frame` creates a new figure and we haven't closed it yet when `grab_frame()` is called, `plt.gcf()` returns our frame figure. After grabbing, we close it to avoid memory bloat.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_animate.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/animate.py tests/test_animate.py
git commit -m "wire up generate_animation orchestrator with MC re-eval and GIF output"
```

---

### Task 6: Run linter and type checker, fix issues

**Files:**
- Modify: `src/python/aerocapture/training/animate.py`
- Modify: `tests/test_animate.py`

- [ ] **Step 1: Run ruff and mypy**

Run: `./lint_code.sh`
Expected: May have import ordering, line length, or type issues to fix.

- [ ] **Step 2: Fix any issues found**

Common fixes:
- Reorder imports (ruff isort)
- Break long lines at 160 chars
- Add type annotations where mypy complains
- Fix any unused imports

- [ ] **Step 3: Run tests again after fixes**

Run: `uv run pytest tests/test_animate.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/animate.py tests/test_animate.py
git commit -m "fix lint and type issues in animate module"
```

---

### Task 7: Update TODO.md and run `smart-commit`

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Mark the TODO item as done**

In `TODO.md`, change the animation line from `- [ ]` to `- [x]`:

```
- [x] Add an animation script of entry corridors and trajectories evolution during training based on checkpoints
```

- [ ] **Step 2: Invoke `smart-commit` skill**

Use the `smart-commit` skill, telling it to take the whole git branch into account.

---

### Task 8: Request code review

- [ ] **Step 1: Invoke `requesting-code-review` skill**

Use the `requesting-code-review` skill to review the completed work on this branch.
