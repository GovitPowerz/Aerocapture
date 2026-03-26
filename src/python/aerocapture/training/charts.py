"""Training convergence chart functions — matplotlib/seaborn SVG output.

Each chart function takes training records (list of dicts from JSONL) and an
output path, then writes a self-contained SVG file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import seaborn as sns  # type: ignore[import-untyped]  # noqa: E402
from scipy import stats  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level theme
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9, rc={"axes.facecolor": "#f5f5f5"})

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
COLOR_BEST = "#1f77b4"
COLOR_MEAN = "#ff7f0e"
COLOR_WORST = "#d62728"
COLOR_NOMINAL_REF = "#d62728"
COLOR_NOMINAL_UNDISPERSED = "#ff7f0e"
COLOR_NOMINAL_BEST = "#2ca02c"
COLOR_CAPTURE = "#1f77b4"  # blue: captured, constraints OK
COLOR_CONSTRAINED = "#ff7f0e"  # orange: captured but violates heat/g-load
COLOR_HYPERBOLIC = "#d62728"  # red: crash or hyperbolic exit
COLOR_DIVERSITY = "#9467bd"

# Trajectory classification codes (used by classify_trajectories / _draw_spaghetti)
TRAJ_OK = 0  # captured + all constraints respected
TRAJ_CONSTRAINED = 1  # captured but violates at least one constraint
TRAJ_FAILED = 2  # crash, hyperbolic exit, timeout

# ---------------------------------------------------------------------------
# Figure defaults
# ---------------------------------------------------------------------------
FULL_WIDTH: tuple[int, int] = (10, 4)
HALF_WIDTH: tuple[int, int] = (5, 4)
DPI: int = 150

# ---------------------------------------------------------------------------
# DV constants (shared with report.py)
# ---------------------------------------------------------------------------
DV_CAP: float = 5000.0
DV_FLOOR: float = 0.1

# ---------------------------------------------------------------------------
# Final record column indices (52-element array)
# ---------------------------------------------------------------------------
_FR_VELOCITY = 3
_FR_FPA = 4
_FR_ECC = 9
_FR_MAX_HEAT_FLUX = 16
_FR_MAX_G_LOAD = 17
_FR_PERI_ERR = 29
_FR_APO_ERR = 30
_FR_IFINAL = 31
_FR_DV1 = 37
_FR_DV2 = 38
_FR_DV3 = 39
_FR_DV_TOTAL = 41
_FR_BANK_CONSUMPTION = 45
_FR_INCL_ERR = 46

# ---------------------------------------------------------------------------
# Dispersion field labels (24 fields)
# ---------------------------------------------------------------------------
DISPERSION_LABELS = [
    "Entry velocity",
    "Entry FPA",
    "Entry azimuth",
    "Entry altitude",
    "Density mult.",
    "Density bias",
    "Cx bias",
    "Cz bias",
    "Mass bias",
    "Ref area bias",
    "Incidence bias",
    "Nav vel err",
    "Nav FPA err",
    "Nav azimuth err",
    "Nav alt err",
    "Nav lon err",
    "Nav lat err",
    "Density filter gain",
    "Gyro bias X",
    "Gyro bias Y",
    "Gyro bias Z",
    "Wind vel",
    "Wind azimuth",
    "Reserved",
]

# ---------------------------------------------------------------------------
# Scheme colors for comparison charts
# ---------------------------------------------------------------------------
SCHEME_COLORS = {
    "ftc": "#1f77b4",
    "neural_network": "#ff7f0e",
    "equilibrium_glide": "#2ca02c",
    "energy_controller": "#9467bd",
    "pred_guid": "#d62728",
    "fnpag": "#8c564b",
    "piecewise_constant": "#e377c2",
}


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


def _add_log_minor_ticks(ax: plt.Axes, axis: Literal["x", "y"] = "y") -> None:  # type: ignore[name-defined]
    """Add light dashed horizontal (or vertical) lines for log-scale minor ticks."""
    from matplotlib.ticker import LogLocator

    target = ax.yaxis if axis == "y" else ax.xaxis
    subs: list[float] = list(range(2, 10))
    target.set_minor_locator(LogLocator(base=10.0, subs=subs, numticks=100))
    ax.grid(which="minor", axis=axis, color="#cccccc", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.grid(which="major", axis=axis, color="#aaaaaa", linestyle="-", linewidth=0.6, alpha=0.8)


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


def _clip_dv(dv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Clip DV values to [DV_FLOOR, DV_CAP] for plot readability."""
    return np.clip(dv, DV_FLOOR, DV_CAP)


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

    _add_log_minor_ticks(ax, axis="y")
    _add_resume_markers(ax, resume_gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost (log scale)")
    ax.set_title("Training Convergence")
    ax.legend(fontsize="small")
    sns.despine(fig=fig)
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

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    ax1.plot(gens, diversity, color=COLOR_DIVERSITY, linewidth=1.5, label="Diversity")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population diversity", color=COLOR_DIVERSITY)
    ax1.tick_params(axis="y", labelcolor=COLOR_DIVERSITY)

    ax2 = ax1.twinx()
    ax2.semilogy(gens, best, color=COLOR_BEST, linewidth=1.0, alpha=0.8, label="Best cost")
    ax2.set_ylabel("Best cost (log)", color=COLOR_BEST)
    ax2.tick_params(axis="y", labelcolor=COLOR_BEST)
    _add_log_minor_ticks(ax2, axis="y")

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
    _add_log_minor_ticks(ax, axis="y")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost (log scale)")
    ax.set_title("Cost Distribution Over Training")
    sns.despine(fig=fig)
    _save_svg(fig, output)
    return True


# ---------------------------------------------------------------------------
# Panel 5: Best parameter values over generations
# ---------------------------------------------------------------------------
_PARAM_DISTRIBUTION_THRESHOLD = 10


def chart_parameter_evolution(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None = None) -> None:
    """Panel 5: Evolution of best parameter values across generations.

    For schemes with <= 10 parameters (from ``best_params``), shows one line per
    parameter (normalized to [0,1]).

    For schemes with > 10 parameters, shows percentile curves (p1-p99) of the
    normalized parameter distribution.

    For NN schemes (no ``best_params``, but ``weight_stats`` present), shows
    per-layer mean/std evolution with ±1σ bands.
    """
    _require_records(records)

    with_params = [r for r in records if r.get("best_params")]
    with_weights = [r for r in records if r.get("weight_stats")]

    if not with_params and not with_weights:
        fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
        ax.set_title("Parameter Evolution (no data)")
        sns.despine(fig=fig)
        _save_svg(fig, output)
        return

    # --- NN path: use weight_stats (per-layer mean/std) ---
    if not with_params and with_weights:
        _chart_weight_stats_evolution(with_weights, output, resume_gens)
        return

    # --- Standard path: use best_params ---
    gens = [r["generation"] for r in with_params]

    # Collect all parameter names (stable order)
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in with_params:
        for k in r["best_params"]:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Build normalized matrix: (n_gens, n_params)
    raw_matrix = np.array([[r["best_params"].get(k, float("nan")) for k in all_keys] for r in with_params])
    col_min = np.nanmin(raw_matrix, axis=0)
    col_max = np.nanmax(raw_matrix, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # avoid division by zero
    normed_matrix = (raw_matrix - col_min) / col_range

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    if len(all_keys) <= _PARAM_DISTRIBUTION_THRESHOLD:
        for i, key in enumerate(all_keys):
            ax.plot(gens, normed_matrix[:, i], linewidth=1.0, label=key)
        ax.legend(fontsize="x-small", ncol=max(1, len(all_keys) // 4), loc="upper right")
        ax.set_title("Parameter Evolution")
    else:
        _plot_percentile_curves(ax, gens, normed_matrix)
        ax.set_title(f"Parameter Distribution ({len(all_keys)} params)")

    _add_resume_markers(ax, resume_gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized value")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def _plot_percentile_curves(ax: plt.Axes, gens: list[int], matrix: npt.NDArray[np.float64]) -> None:  # type: ignore[name-defined]
    """Draw percentile curves (p1-p99) with p25-p75 fill for a (n_gens, n_values) matrix."""
    percentiles = [1, 10, 25, 50, 75, 90, 99]
    pct_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#1f77b4", "#bcbd22", "#ff7f0e", "#d62728"]
    pct_widths = [0.6, 0.8, 1.0, 1.5, 1.0, 0.8, 0.6]
    pct_styles: list[str] = [":", "--", "-.", "-", "-.", "--", ":"]

    pct_data: dict[int, npt.NDArray[np.float64]] = {}
    for p in percentiles:
        pct_data[p] = np.nanpercentile(matrix, p, axis=1)

    for p, color, lw, ls in zip(percentiles, pct_colors, pct_widths, pct_styles, strict=True):
        ax.plot(gens, pct_data[p], color=color, linewidth=lw, linestyle=ls, label=f"p{p}")

    ax.fill_between(gens, pct_data[25], pct_data[75], alpha=0.15, color="#1f77b4")
    ax.legend(fontsize="x-small", ncol=4)


def _chart_weight_stats_evolution(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None) -> None:
    """NN variant: per-layer mean ± std evolution from weight_stats."""
    gens = [r["generation"] for r in records]

    # Discover layer names from first record
    layer_names = list(records[0]["weight_stats"].keys())

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, layer in enumerate(layer_names):
        color = colors[i % len(colors)]
        means = [r["weight_stats"][layer]["mean"] for r in records]
        stds = [r["weight_stats"][layer]["std"] for r in records]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(gens, means_arr, color=color, linewidth=1.0, label=layer)
        ax.fill_between(gens, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.12)

    _add_resume_markers(ax, resume_gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weight value")
    ax.set_title(f"NN Weight Evolution ({len(layer_names)} layers, mean \u00b1 1\u03c3)")
    ax.legend(fontsize="x-small", ncol=max(1, len(layer_names) // 3))
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 6: Seed pool evolution (conditional)
# ---------------------------------------------------------------------------
def chart_seed_pool(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None = None) -> bool:
    """Panel 6: Seed pool difficulty distribution over generations.

    Shows percentile curves (p1, p10, p25, p50, p75, p90, p99) of per-seed
    difficulty scores. Falls back to min/max lines if difficulty_scores arrays
    are not available in the JSONL data.

    Returns False if no ``pool_metrics`` data is available.
    """
    with_pool = [r for r in records if r.get("pool_metrics")]
    if not with_pool:
        return False

    gens = [r["generation"] for r in with_pool]
    has_scores = any(r["pool_metrics"].get("difficulty_scores") for r in with_pool)

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    if has_scores:
        percentiles = [1, 10, 25, 50, 75, 90, 99]
        pct_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#1f77b4", "#bcbd22", "#ff7f0e", "#d62728"]
        pct_widths = [0.6, 0.8, 1.0, 1.5, 1.0, 0.8, 0.6]
        pct_styles = [":", "--", "-.", "-", "-.", "--", ":"]

        pct_data: dict[int, list[float]] = {p: [] for p in percentiles}
        for r in with_pool:
            scores = r["pool_metrics"].get("difficulty_scores", [])
            if scores:
                arr = np.array(scores)
                for p in percentiles:
                    pct_data[p].append(float(np.percentile(arr, p)))
            else:
                for p in percentiles:
                    pct_data[p].append(float("nan"))

        for p, color, lw, ls in zip(percentiles, pct_colors, pct_widths, pct_styles, strict=True):
            ax.plot(gens, pct_data[p], color=color, linewidth=lw, linestyle=ls, label=f"p{p}")

        # Fill between p25-p75
        ax.fill_between(gens, pct_data[25], pct_data[75], alpha=0.15, color="#1f77b4")
    else:
        # Fallback: only min/max available
        d_min = [r["pool_metrics"].get("difficulty_min", 0.0) for r in with_pool]
        d_max = [r["pool_metrics"].get("difficulty_max", 0.0) for r in with_pool]
        ax.plot(gens, d_min, color=COLOR_CAPTURE, linewidth=1.0, label="Min difficulty")
        ax.plot(gens, d_max, color=COLOR_WORST, linewidth=1.0, label="Max difficulty")
        ax.fill_between(gens, d_min, d_max, alpha=0.15, color="#1f77b4")

    _add_resume_markers(ax, resume_gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Difficulty (cost of best individual)")
    ax.set_title("Seed Pool Difficulty Distribution")
    ax.legend(fontsize="x-small", ncol=4)
    sns.despine(fig=fig)
    _save_svg(fig, output)
    return True


# ---------------------------------------------------------------------------
# Corridor / energy helpers
# ---------------------------------------------------------------------------
def _compute_envelope(
    trajectories: list[npt.NDArray[np.float64]],
    energy_col: int,
    value_col: int,
    traj_class: npt.NDArray[np.int8],
    n_bins: int = 200,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Bin captured trajectories by energy, compute min/max value per bin.

    Returns ``(centers, min_vals, max_vals)`` with NaN for empty bins.
    """
    # Collect all captured trajectory points (OK + constrained, not failed)
    captured_indices = np.where(traj_class != TRAJ_FAILED)[0]
    if len(captured_indices) == 0:
        centers = np.full(n_bins, np.nan)
        return centers, np.full(n_bins, np.nan), np.full(n_bins, np.nan)

    all_energy: list[npt.NDArray[np.float64]] = []
    all_values: list[npt.NDArray[np.float64]] = []
    for idx in captured_indices:
        traj = trajectories[idx]
        all_energy.append(traj[:, energy_col])
        all_values.append(traj[:, value_col])

    energy = np.concatenate(all_energy)
    values = np.concatenate(all_values)

    e_min, e_max = float(np.nanmin(energy)), float(np.nanmax(energy))
    if e_min == e_max:
        centers = np.full(n_bins, e_min)
        return centers, np.full(n_bins, np.nanmin(values)), np.full(n_bins, np.nanmax(values))

    edges = np.linspace(e_min, e_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    min_vals = np.full(n_bins, np.nan)
    max_vals = np.full(n_bins, np.nan)

    bin_indices = np.digitize(energy, edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            min_vals[b] = float(np.nanmin(values[mask]))
            max_vals[b] = float(np.nanmax(values[mask]))

    return centers, min_vals, max_vals


def classify_trajectories(
    final_records: npt.NDArray[np.float64],
    heat_flux_limit: float | None = None,
    g_load_limit: float | None = None,
) -> npt.NDArray[np.int8]:
    """Classify each trajectory as OK (0), constrained (1), or failed (2).

    - OK: captured (ecc < 1) and within all constraint limits
    - Constrained: captured but exceeds heat flux or g-load limit
    - Failed: crash, hyperbolic exit, timeout
    """
    n = len(final_records)
    classification = np.full(n, TRAJ_FAILED, dtype=np.int8)

    ecc = final_records[:, _FR_ECC]
    captured = ecc < 1.0
    classification[captured] = TRAJ_OK

    # Downgrade captured trajectories that violate constraints
    if heat_flux_limit is not None:
        q_exceed = final_records[:, _FR_MAX_HEAT_FLUX] > heat_flux_limit
        classification[captured & q_exceed] = TRAJ_CONSTRAINED
    if g_load_limit is not None:
        g_exceed = final_records[:, _FR_MAX_G_LOAD] > g_load_limit
        classification[captured & g_exceed] = TRAJ_CONSTRAINED

    return classification


_TRAJ_COLORS = {TRAJ_OK: COLOR_CAPTURE, TRAJ_CONSTRAINED: COLOR_CONSTRAINED, TRAJ_FAILED: COLOR_HYPERBOLIC}


def _draw_spaghetti(
    ax: plt.Axes,  # type: ignore[name-defined]
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    x_col: int,
    y_col: int,
) -> None:
    """Draw MC spaghetti lines colored by classification: blue/orange/red."""
    alpha = _spaghetti_alpha(len(trajectories))
    for i, traj in enumerate(trajectories):
        color = _TRAJ_COLORS.get(int(traj_class[i]), COLOR_HYPERBOLIC)
        ax.plot(traj[:, x_col], traj[:, y_col], color=color, alpha=alpha, linewidth=0.5)


def _draw_nominals(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_col: int,
    y_col: int,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Overlay up to 3 nominal trajectories on *ax*."""
    if ref_nominal is not None:
        ax.plot(ref_nominal[:, x_col], ref_nominal[:, y_col], color=COLOR_NOMINAL_REF, linewidth=1.5, label="Ref nominal")
    if undispersed_nominal is not None:
        ax.plot(undispersed_nominal[:, x_col], undispersed_nominal[:, y_col], color=COLOR_NOMINAL_UNDISPERSED, linewidth=1.5, label="Undispersed")
    if best_nominal is not None:
        ax.plot(best_nominal[:, x_col], best_nominal[:, y_col], color=COLOR_NOMINAL_BEST, linewidth=1.5, label="Best MC")


# ---------------------------------------------------------------------------
# Panel 7: Energy vs pdyn (full width)
# ---------------------------------------------------------------------------
def chart_corridor_pdyn(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    corridor_data: dict[str, Any] | None = None,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 7: Energy vs dynamic pressure with optional 4-layer corridor fill."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    # Draw corridor zones if data is provided
    if corridor_data is not None:
        e_bins = corridor_data["energy_bins"]
        crash_pdyn = corridor_data["envelope_crash_pdyn"]
        restricted_max = corridor_data["envelope_restricted_max_pdyn"]
        restricted_min = corridor_data["envelope_restricted_min_pdyn"]
        capture_pdyn = corridor_data["envelope_capture_pdyn"]

        # Red zone at top (crash)
        ax.fill_between(e_bins, restricted_max, crash_pdyn, color=COLOR_WORST, alpha=0.15, label="Crash zone")
        # Grey transition above restricted
        ax.fill_between(e_bins, restricted_max, restricted_min, color="white", alpha=0.6)
        # Grey transition below restricted
        ax.fill_between(e_bins, restricted_min, capture_pdyn, color="#cccccc", alpha=0.3, label="Transition")
        # Red zone at bottom (hyperbolic)
        ax.fill_between(e_bins, capture_pdyn, 0, color=COLOR_WORST, alpha=0.15, label="Hyperbolic zone")

    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=9)
    _draw_nominals(ax, x_col=8, y_col=9, ref_nominal=ref_nominal, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Dynamic pressure (kPa)")
    ax.set_title("Corridor — Dynamic Pressure")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 8: Energy vs inclination (half width)
# ---------------------------------------------------------------------------
def chart_corridor_inclination(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 8: Energy vs inclination with captured envelope fill."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    centers, min_vals, max_vals = _compute_envelope(trajectories, energy_col=8, value_col=11, traj_class=traj_class)
    valid = ~np.isnan(min_vals)
    if np.any(valid):
        ax.fill_between(centers[valid], min_vals[valid], max_vals[valid], color=COLOR_CAPTURE, alpha=0.15, label="Captured envelope")

    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=11)
    _draw_nominals(ax, x_col=8, y_col=11, ref_nominal=ref_nominal, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Inclination (deg)")
    ax.set_title("Corridor — Inclination")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 9: Energy vs bank angle (half width)
# ---------------------------------------------------------------------------
def chart_corridor_bank(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 9: Energy vs bank angle with captured envelope fill."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    centers, min_vals, max_vals = _compute_envelope(trajectories, energy_col=8, value_col=10, traj_class=traj_class)
    valid = ~np.isnan(min_vals)
    if np.any(valid):
        ax.fill_between(centers[valid], min_vals[valid], max_vals[valid], color=COLOR_CAPTURE, alpha=0.15, label="Captured envelope")

    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=10)
    _draw_nominals(ax, x_col=8, y_col=10, ref_nominal=ref_nominal, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Bank angle (deg)")
    ax.set_title("Corridor — Bank Angle")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 10: Altitude vs time (full width)
# ---------------------------------------------------------------------------
def _draw_time_nominals(
    ax: plt.Axes,  # type: ignore[name-defined]
    y_col: int,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Overlay undispersed and best-DV trajectories on a time-domain chart."""
    if undispersed_nominal is not None:
        ax.plot(undispersed_nominal[:, 7], undispersed_nominal[:, y_col], color=COLOR_NOMINAL_UNDISPERSED, linewidth=1.5, zorder=10, label="Undispersed")
    if best_nominal is not None:
        ax.plot(best_nominal[:, 7], best_nominal[:, y_col], color=COLOR_NOMINAL_BEST, linewidth=1.5, zorder=10, label="Best MC")


def chart_altitude_time(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 10: Altitude vs time MC spaghetti with nominal overlays."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    _draw_spaghetti(ax, trajectories, traj_class, x_col=7, y_col=0)
    _draw_time_nominals(ax, y_col=0, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude vs Time")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 11: Heat flux vs time
# ---------------------------------------------------------------------------
def chart_heat_flux_time(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    limit_kw_m2: float | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 11: Heat flux vs time MC spaghetti with optional constraint line and nominals."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    _draw_spaghetti(ax, trajectories, traj_class, x_col=7, y_col=6)
    _draw_time_nominals(ax, y_col=6, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    if limit_kw_m2 is not None:
        ax.axhline(limit_kw_m2, color=COLOR_WORST, linestyle="--", linewidth=1.0, label=f"Limit ({limit_kw_m2:.0f} kW/m\u00b2)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heat flux (kW/m\u00b2)")
    ax.set_title("Heat Flux vs Time")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 12: G-load vs time
# ---------------------------------------------------------------------------
def chart_gload_time(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    limit_g: float | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 12: G-load vs time MC spaghetti with optional constraint line and nominals."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    _draw_spaghetti(ax, trajectories, traj_class, x_col=7, y_col=12)
    _draw_time_nominals(ax, y_col=12, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    if limit_g is not None:
        ax.axhline(limit_g, color=COLOR_WORST, linestyle="--", linewidth=1.0, label=f"Limit ({limit_g:.1f} g)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("G-load (g)")
    ax.set_title("G-Load vs Time")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 13: Bank angle vs time
# ---------------------------------------------------------------------------
def chart_bank_angle_time(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 13: Bank angle vs time MC spaghetti with nominal overlays."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    _draw_spaghetti(ax, trajectories, traj_class, x_col=7, y_col=10)
    _draw_time_nominals(ax, y_col=10, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bank angle (deg)")
    ax.set_title("Bank Angle vs Time")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 14: Navigation density ratio vs time
# ---------------------------------------------------------------------------
def chart_nav_density_ratio(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 14: Navigation density ratio vs time with nominals and perfect-estimate reference."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    _draw_spaghetti(ax, trajectories, traj_class, x_col=7, y_col=13)
    _draw_time_nominals(ax, y_col=13, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, label="Perfect estimate")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Nav density ratio")
    ax.set_title("Navigation Density Ratio vs Time")
    ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 15: Total DV distribution — histogram + CDF + percentile markers
# ---------------------------------------------------------------------------
_LOG10_TICK_VALUES = [0.1, 1, 10, 100, 1000, 5000]


def chart_dv_distribution(final_records: npt.NDArray[np.float64], output: Path) -> None:
    """Panel 15: Total DV histogram (log10 x) with CDF overlay and percentile markers."""
    dv = _clip_dv(final_records[:, _FR_DV_TOTAL])
    log_dv = np.log10(dv)

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    # Histogram
    ax1.hist(log_dv, bins=30, color=COLOR_CAPTURE, alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Total \u0394V (m/s)")
    ax1.set_ylabel("Count")

    # Custom log-scale tick labels
    tick_positions = [np.log10(v) for v in _LOG10_TICK_VALUES]
    tick_labels = [str(v) for v in _LOG10_TICK_VALUES]
    ax1.set_xticks(tick_positions)
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


# ---------------------------------------------------------------------------
# Panel 16: Individual burn DV histograms (overlaid, log10 x)
# ---------------------------------------------------------------------------
def chart_dv_individual_burns(final_records: npt.NDArray[np.float64], output: Path) -> None:
    """Panel 16: Overlaid histograms for |dv1|, |dv2|, |dv3| on log10 x-axis."""
    dv1 = np.log10(_clip_dv(np.abs(final_records[:, _FR_DV1])))
    dv2 = np.log10(_clip_dv(np.abs(final_records[:, _FR_DV2])))
    dv3 = np.log10(_clip_dv(np.abs(final_records[:, _FR_DV3])))

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    ax.hist(dv1, bins=25, color="#1f77b4", alpha=0.5, label="|DV1| (periapsis)")
    ax.hist(dv2, bins=25, color="#ff7f0e", alpha=0.5, label="|DV2| (apoapsis)")
    ax.hist(dv3, bins=25, color="#2ca02c", alpha=0.5, label="|DV3| (inclination)")

    # Snap x-axis to enclosing powers of 10
    all_log = np.concatenate([dv1, dv2, dv3])
    lo = np.floor(np.min(all_log))
    hi = np.ceil(np.max(all_log))
    ax.set_xlim(lo, hi)
    tick_decades = np.arange(lo, hi + 1)
    ax.set_xticks(tick_decades)
    ax.set_xticklabels([f"{10**d:g}" for d in tick_decades])

    ax.set_xlabel("\u0394V (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Individual Burn \u0394V")
    ax.legend(fontsize="x-small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 17: Entry conditions scatter (V vs FPA)
# ---------------------------------------------------------------------------
def chart_entry_conditions(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
) -> None:
    """Panel 17: Entry V vs FPA scatter — blue OK, orange constrained, red failed."""
    entry_v = np.array([traj[0, 3] for traj in trajectories])
    entry_fpa = np.array([traj[0, 4] for traj in trajectories])

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    ok = traj_class == TRAJ_OK
    constrained = traj_class == TRAJ_CONSTRAINED
    failed = traj_class == TRAJ_FAILED
    ax.scatter(entry_fpa[ok], entry_v[ok], color=COLOR_CAPTURE, s=20, label="Captured", zorder=5)
    if np.any(constrained):
        ax.scatter(entry_fpa[constrained], entry_v[constrained], color=COLOR_CONSTRAINED, s=20, label="Constrained", zorder=5)
    ax.scatter(entry_fpa[failed], entry_v[failed], color=COLOR_HYPERBOLIC, marker="x", s=30, label="Failed", zorder=5)

    ax.set_xlabel("Entry FPA (deg)")
    ax.set_ylabel("Entry velocity (m/s)")
    ax.set_title("Entry Conditions")
    ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 18: Exit conditions scatter (V vs FPA, marker size ~ log10 DV)
# ---------------------------------------------------------------------------
def chart_exit_conditions(final_records: npt.NDArray[np.float64], output: Path) -> None:
    """Panel 18: Exit V vs FPA, marker size proportional to log10(DV)."""
    exit_v = final_records[:, _FR_VELOCITY]
    exit_fpa = final_records[:, _FR_FPA]
    dv = _clip_dv(final_records[:, _FR_DV_TOTAL])
    sizes = np.log10(dv) * 10  # scale for visibility
    sizes = np.clip(sizes, 5, 100)

    ecc = final_records[:, _FR_ECC]
    captured = ecc < 1.0

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    if np.any(captured):
        ax.scatter(exit_fpa[captured], exit_v[captured], s=sizes[captured], color=COLOR_NOMINAL_BEST, alpha=0.7, label="Captured")
    if np.any(~captured):
        ax.scatter(exit_fpa[~captured], exit_v[~captured], s=sizes[~captured], color=COLOR_HYPERBOLIC, marker="x", label="Hyperbolic")

    ax.set_xlabel("Exit FPA (deg)")
    ax.set_ylabel("Exit velocity (m/s)")
    ax.set_title("Exit Conditions")
    ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 20: Dispersion correlation grid (4x6, scatter + regression)
# ---------------------------------------------------------------------------
def chart_dispersion_grid(
    final_records: npt.NDArray[np.float64],
    dispersions: npt.NDArray[np.float64],
    output: Path,
) -> None:
    """Panel 20: subplot grid — each dispersion field vs log10(DV) with linear regression."""
    n_fields = dispersions.shape[1]
    n_cols = 4
    n_rows = math.ceil(n_fields / n_cols)

    dv = _clip_dv(final_records[:, _FR_DV_TOTAL])
    log_dv = np.log10(dv)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), dpi=DPI)
    axes_flat = axes.flatten()

    for i in range(n_fields):
        ax = axes_flat[i]
        x = dispersions[:, i]
        ax.scatter(x, log_dv, s=8, alpha=0.5, color=COLOR_CAPTURE)

        # Linear regression (skip if all x values are identical — e.g. a zero-variance dispersion field)
        finite = np.isfinite(x) & np.isfinite(log_dv)
        if np.sum(finite) > 2 and np.ptp(x[finite]) > 0:
            result = stats.linregress(x[finite], log_dv[finite])
            x_range = np.array([float(np.min(x[finite])), float(np.max(x[finite]))])
            ax.plot(x_range, result.slope * x_range + result.intercept, color=COLOR_WORST, linewidth=1.0)
            label_txt = f"R\u00b2={result.rvalue**2:.2f}\np={result.pvalue:.1e}"
            ax.annotate(label_txt, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=6, verticalalignment="top")

        label = DISPERSION_LABELS[i] if i < len(DISPERSION_LABELS) else f"Field {i}"
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for i in range(n_fields, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.supylabel("log10(DV)", fontsize=9)
    fig.suptitle("Dispersion Correlation Grid", fontsize=11)
    fig.tight_layout()
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Comparison: Cross-scheme convergence (semilogy)
# ---------------------------------------------------------------------------
def chart_comparison_convergence(all_data: dict[str, list[dict[str, Any]]], output: Path) -> None:
    """Cross-scheme convergence chart — one semilogy line per guidance scheme."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    for scheme, records in all_data.items():
        if not records:
            continue
        gens = [r["generation"] for r in records]
        best = [r["best_cost"] for r in records]
        color = SCHEME_COLORS.get(scheme, "#333333")
        ax.semilogy(gens, best, color=color, linewidth=1.5, label=scheme)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best cost (log scale)")
    ax.set_title("Cross-Scheme Convergence")
    ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)
