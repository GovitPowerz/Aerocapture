"""Training convergence chart functions — matplotlib/seaborn SVG output.

Each chart function takes training records (list of dicts from JSONL) and an
output path, then writes a self-contained SVG file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level theme
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
COLOR_BEST = "#1f77b4"
COLOR_MEAN = "#ff7f0e"
COLOR_WORST = "#d62728"
COLOR_NOMINAL_REF = "#d62728"
COLOR_NOMINAL_UNDISPERSED = "#ff7f0e"
COLOR_NOMINAL_BEST = "#2ca02c"
COLOR_CAPTURE = "#1f77b4"
COLOR_HYPERBOLIC = "#d62728"
COLOR_DIVERSITY = "#9467bd"

# ---------------------------------------------------------------------------
# Figure defaults
# ---------------------------------------------------------------------------
FULL_WIDTH: tuple[int, int] = (10, 4)
HALF_WIDTH: tuple[int, int] = (5, 4)
DPI: int = 150

# ---------------------------------------------------------------------------
# DV constants (shared with final_report.py)
# ---------------------------------------------------------------------------
DV_CAP: float = 5000.0
DV_FLOOR: float = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_svg(fig: plt.Figure, path: Path) -> None:  # type: ignore[name-defined]
    """Save figure as SVG and close it."""
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _spaghetti_alpha(n: int) -> float:
    """Return an alpha value for spaghetti plots that scales with trajectory count."""
    if n <= 0:
        return 0.2
    return max(0.02, min(0.2, 1.0 / math.sqrt(n)))


def _add_resume_markers(ax: plt.Axes, resume_gens: list[int] | None) -> None:  # type: ignore[name-defined]
    """Add vertical dashed lines at resume generations."""
    if not resume_gens:
        return
    for gen in resume_gens:
        ax.axvline(gen, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="_resume" if gen != resume_gens[0] else "Resume")


def _require_records(records: list[dict[str, Any]]) -> None:
    """Raise ValueError if records list is empty."""
    if not records:
        msg = "No training records provided"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Panel 1: Convergence — best / mean / worst cost (log y)
# ---------------------------------------------------------------------------
def chart_convergence(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None = None) -> None:
    """Panel 1: Best/mean/worst cost over generations with log-scale y-axis."""
    _require_records(records)

    gens = [r["generation"] for r in records]
    best = [r["best_cost"] for r in records]
    mean = [r["mean_cost"] for r in records]
    worst = [r["worst_cost"] for r in records]

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    ax.semilogy(gens, best, color=COLOR_BEST, linewidth=1.5, label="Best")
    ax.semilogy(gens, mean, color=COLOR_MEAN, linewidth=1.0, label="Mean")
    ax.semilogy(gens, worst, color=COLOR_WORST, linewidth=0.8, alpha=0.7, label="Worst")

    # Mark improvement points
    improvement_gens = [r["generation"] for r in records if r.get("improvement", False)]
    improvement_costs = [r["best_cost"] for r in records if r.get("improvement", False)]
    if improvement_gens:
        ax.scatter(improvement_gens, improvement_costs, color=COLOR_BEST, marker="v", s=30, zorder=5, label="Improvement")

    _add_resume_markers(ax, resume_gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost (log scale)")
    ax.set_title("Training Convergence")
    ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 2: Capture rate + constraint violation rate (dual axis)
# ---------------------------------------------------------------------------
def chart_capture_constraint_rate(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None = None) -> None:
    """Panel 2: Capture rate % and constraint violation rate on dual y-axes."""
    _require_records(records)

    gens = [r["generation"] for r in records]
    capture = [r.get("capture_rate", 0.0) * 100 for r in records]
    constraint = [r.get("constraint_violation_rate", 0.0) * 100 for r in records]

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    ax1.plot(gens, capture, color=COLOR_CAPTURE, linewidth=1.5, label="Capture rate")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Capture rate (%)", color=COLOR_CAPTURE)
    ax1.tick_params(axis="y", labelcolor=COLOR_CAPTURE)
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    ax2.plot(gens, constraint, color=COLOR_WORST, linewidth=1.0, alpha=0.8, label="Constraint violations")
    ax2.set_ylabel("Constraint violation (%)", color=COLOR_WORST)
    ax2.tick_params(axis="y", labelcolor=COLOR_WORST)
    ax2.set_ylim(-5, 105)

    _add_resume_markers(ax1, resume_gens)
    ax1.set_title("Capture & Constraint Rates")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="small", loc="lower right")

    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 3: Population diversity vs best cost (dual axis, half width)
# ---------------------------------------------------------------------------
def chart_diversity_cost(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None = None) -> None:
    """Panel 3: Population diversity and best cost on dual y-axes (half width)."""
    _require_records(records)

    gens = [r["generation"] for r in records]
    diversity = [r.get("population_diversity", 0.0) for r in records]
    best = [r["best_cost"] for r in records]

    fig, ax1 = plt.subplots(figsize=HALF_WIDTH, dpi=DPI)
    ax1.plot(gens, diversity, color=COLOR_DIVERSITY, linewidth=1.5, label="Diversity")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population diversity", color=COLOR_DIVERSITY)
    ax1.tick_params(axis="y", labelcolor=COLOR_DIVERSITY)

    ax2 = ax1.twinx()
    ax2.semilogy(gens, best, color=COLOR_BEST, linewidth=1.0, alpha=0.8, label="Best cost")
    ax2.set_ylabel("Best cost (log)", color=COLOR_BEST)
    ax2.tick_params(axis="y", labelcolor=COLOR_BEST)

    _add_resume_markers(ax1, resume_gens)
    ax1.set_title("Diversity vs Cost")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="small")

    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 4: Cost distribution box plots at sampled generations
# ---------------------------------------------------------------------------
def chart_cost_distribution(records: list[dict[str, Any]], output: Path) -> bool:
    """Panel 4: Box plots of cost distribution at sampled generations.

    Returns False if no ``all_costs`` data is available in the records.
    """
    # Filter records that have all_costs data
    with_costs = [r for r in records if r.get("all_costs")]
    if not with_costs:
        return False

    # Sample up to ~10 evenly spaced generations for readability
    n_sample = min(10, len(with_costs))
    indices = np.linspace(0, len(with_costs) - 1, n_sample, dtype=int)
    sampled = [with_costs[i] for i in indices]

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    box_data = [s["all_costs"] for s in sampled]
    positions = list(range(len(sampled)))
    labels = [str(s["generation"]) for s in sampled]

    ax.boxplot(box_data, positions=positions, tick_labels=labels, widths=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost (log scale)")
    ax.set_title("Cost Distribution Over Training")
    sns.despine(fig=fig)
    _save_svg(fig, output)
    return True


# ---------------------------------------------------------------------------
# Panel 5: Best parameter values over generations
# ---------------------------------------------------------------------------
def chart_parameter_evolution(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None = None) -> None:
    """Panel 5: Evolution of best parameter values across generations."""
    _require_records(records)

    with_params = [r for r in records if r.get("best_params")]
    if not with_params:
        # Nothing to plot but not an error — just create an empty figure
        fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
        ax.set_title("Parameter Evolution (no data)")
        sns.despine(fig=fig)
        _save_svg(fig, output)
        return

    gens = [r["generation"] for r in with_params]
    # Collect all parameter names
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in with_params:
        for k in r["best_params"]:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Normalize each parameter to [0, 1] for overlay plotting
    for key in all_keys:
        values = [r["best_params"].get(key, float("nan")) for r in with_params]
        arr = np.array(values, dtype=float)
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        normed = (arr - vmin) / (vmax - vmin) if vmax - vmin > 0 else np.full_like(arr, 0.5)
        # Store back temporarily for plotting
        for i, r in enumerate(with_params):
            r.setdefault("_normed_params", {})[key] = normed[i]

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    for key in all_keys:
        vals = [r["_normed_params"][key] for r in with_params]
        ax.plot(gens, vals, linewidth=1.0, label=key)

    # Clean up temporary data
    for r in with_params:
        r.pop("_normed_params", None)

    _add_resume_markers(ax, resume_gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized value")
    ax.set_title("Parameter Evolution")
    ax.legend(fontsize="x-small", ncol=max(1, len(all_keys) // 4), loc="upper right")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 6: Seed pool evolution (conditional)
# ---------------------------------------------------------------------------
def chart_seed_pool(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None = None) -> bool:
    """Panel 6: Seed pool size and difficulty metrics over generations.

    Returns False if no ``pool_metrics`` data is available.
    """
    with_pool = [r for r in records if r.get("pool_metrics")]
    if not with_pool:
        return False

    gens = [r["generation"] for r in with_pool]
    pool_sizes = [r["pool_metrics"].get("pool_size", 0) for r in with_pool]
    mean_difficulty = [r["pool_metrics"].get("mean_difficulty", 0.0) for r in with_pool]

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    ax1.plot(gens, pool_sizes, color=COLOR_CAPTURE, linewidth=1.5, label="Pool size")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Pool size", color=COLOR_CAPTURE)
    ax1.tick_params(axis="y", labelcolor=COLOR_CAPTURE)

    ax2 = ax1.twinx()
    ax2.plot(gens, mean_difficulty, color=COLOR_WORST, linewidth=1.0, alpha=0.8, label="Mean difficulty")
    ax2.set_ylabel("Mean difficulty", color=COLOR_WORST)
    ax2.tick_params(axis="y", labelcolor=COLOR_WORST)

    _add_resume_markers(ax1, resume_gens)
    ax1.set_title("Seed Pool Evolution")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="small")

    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)
    return True
