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


def apply_theme() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9, rc={"axes.facecolor": "#f5f5f5"})


apply_theme()

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
# Trajectory column indices for (N, 17) per-timestep arrays
# Layout: alt_km(0), lon_deg(1), lat_deg(2), vel_m_s(3), fpa_deg(4),
#   heading_deg(5), heat_flux_kw_m2(6), time_s(7), energy_mj_kg(8),
#   pdyn_kpa(9), bank_angle_deg(10), inclination_deg(11), g_load_g(12),
#   nav_density_ratio(13), truth_density_kg_m3(14), heat_load_kj_m2(15),
#   density_perturbation(16)
# ---------------------------------------------------------------------------
_TC_ALT = 0
_TC_HEAT_FLUX = 6
_TC_TIME = 7
_TC_ENERGY = 8
_TC_PDYN = 9
_TC_BANK = 10
_TC_INCL = 11
_TC_GLOAD = 12
_TC_NAV_DENS = 13
_TC_HEAT_LOAD = 15

# ---------------------------------------------------------------------------
# DV constants (shared with report.py)
# ---------------------------------------------------------------------------
DV_CAP: float = 5000.0
DV_FLOOR: float = 1.0

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
_FR_INTEGRATED_FLUX = 28
_FR_BANK_CONSUMPTION = 45
_FR_INCL_ERR = 46


def is_captured(final_records: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Canonical captured definition: exited atmosphere (ifinal==3) on a bound orbit (ecc<1)."""
    result: npt.NDArray[np.bool_] = (final_records[:, _FR_IFINAL] == 3) & (final_records[:, _FR_ECC] < 1.0)
    return result


# ---------------------------------------------------------------------------
# Dispersion field labels (26 fields, matches Rust DispersionDraw::to_array() order)
# ---------------------------------------------------------------------------
DISPERSION_LABELS = [
    "Entry altitude",  # [0]  altitude (m)
    "Entry longitude",  # [1]  longitude (rad)
    "Entry latitude",  # [2]  latitude (rad)
    "Entry velocity",  # [3]  velocity (m/s)
    "Entry FPA",  # [4]  flight_path (rad)
    "Entry azimuth",  # [5]  azimuth (rad)
    "Density mult.",  # [6]  density (fractional)
    "Cx bias",  # [7]  drag_coeff (fractional)
    "Cz bias",  # [8]  lift_coeff (fractional)
    "Incidence bias",  # [9]  incidence (rad)
    "Nav alt err",  # [10] nav_altitude (m)
    "Nav lon err",  # [11] nav_longitude (rad)
    "Nav lat err",  # [12] nav_latitude (rad)
    "Nav vel err",  # [13] nav_velocity (m/s)
    "Nav FPA err",  # [14] nav_flight_path (rad)
    "Nav azimuth err",  # [15] nav_azimuth (rad)
    "Nav drag accel err",  # [16] nav_drag_accel (m/s²)
    "Mass bias",  # [17] mass (fractional)
    "Ref area bias",  # [18] ref_area (fractional)
    "Max bank rate bias",  # [19] max_bank_rate (fractional)
    "Pilot tau bias",  # [20] pilot_tau (fractional)
    "Pilot damping bias",  # [21] pilot_damping (fractional)
    "Pilot freq bias",  # [22] pilot_frequency (fractional)
    "Density filter gain",  # [23] filter_gain (absolute)
    "Wind scale",  # [24] wind_scale (multiplicative)
    "Wind direction bias",  # [25] wind_direction_bias (rad)
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


def _log10_ticks(values: npt.NDArray[np.float64], floor: float = 1.0) -> tuple[npt.NDArray[np.float64], list[str]]:
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

    # Validation gate results (sparse: only on gens where validation fired)
    val_gens = [r["generation"] for r in records if r.get("validation")]
    val_means = [r["validation"]["mean_cost"] for r in records if r.get("validation")]
    val_p95s = [r["validation"]["p95_cost"] for r in records if r.get("validation")]
    if val_gens:
        ax.semilogy(val_gens, val_means, color="#2ca02c", linewidth=1.2, marker="o", markersize=3, label="Val mean")
        ax.semilogy(val_gens, val_p95s, color="#2ca02c", linewidth=0.7, linestyle="--", alpha=0.6, label="Val p95")

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


def _get_param_bounds(records: list[dict[str, Any]], keys: list[str]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Look up ParamSpec bounds for each key. Returns (p_min, p_max) arrays or None."""
    scheme = next((r.get("scheme") for r in records if r.get("scheme")), None)
    if scheme is None or scheme == "neural_network":
        return None
    try:
        from aerocapture.training.param_spaces import PARAM_SPACES

        specs = {s.name: s for s in PARAM_SPACES[scheme]}
    except (ImportError, KeyError):  # fmt: skip
        return None
    p_min = np.array([specs[k].p_min if k in specs else np.nan for k in keys])
    p_max = np.array([specs[k].p_max if k in specs else np.nan for k in keys])
    if np.any(np.isnan(p_min)) or np.any(np.isnan(p_max)):
        return None
    return p_min, p_max


def chart_parameter_evolution(records: list[dict[str, Any]], output: Path, resume_gens: list[int] | None = None) -> None:
    """Panel 5: Evolution of generation-best parameter values across generations.

    Uses ``gen_best_params`` (current generation's best individual) when available,
    falling back to ``best_params`` (global best) for older JSONL logs.

    Normalizes against ParamSpec bounds when the scheme is known, so 0 = p_min
    and 1 = p_max. Falls back to observed min/max normalization otherwise.

    For schemes with <= 10 parameters, shows one line per parameter.
    For schemes with > 10 parameters, shows percentile curves (p1-p99).
    For NN schemes (``weight_stats``), shows per-layer mean/std evolution.
    """
    _require_records(records)

    # Prefer gen_best_params (generation best) over best_params (global best)
    params_key = "gen_best_params" if any(r.get("gen_best_params") for r in records) else "best_params"
    with_params = [r for r in records if r.get(params_key)]
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

    # --- Standard path: use gen_best_params (or best_params fallback) ---
    gens = [r["generation"] for r in with_params]

    # Collect all parameter names (stable order)
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in with_params:
        for k in r[params_key]:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    raw_matrix = np.array([[r[params_key].get(k, float("nan")) for k in all_keys] for r in with_params])

    # Normalize against ParamSpec bounds when available (0 = p_min, 1 = p_max),
    # otherwise fall back to observed min/max normalization.
    bounds = _get_param_bounds(records, all_keys)
    if bounds is not None:
        p_min, p_max = bounds
        col_range = p_max - p_min
        col_range[col_range == 0] = 1.0
        normed_matrix = (raw_matrix - p_min) / col_range
    else:
        col_min = np.nanmin(raw_matrix, axis=0)
        col_max = np.nanmax(raw_matrix, axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        normed_matrix = (raw_matrix - col_min) / col_range

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    if len(all_keys) <= _PARAM_DISTRIBUTION_THRESHOLD:
        for i, key in enumerate(all_keys):
            ax.plot(gens, normed_matrix[:, i], linewidth=1.0, label=key)
        ax.legend(fontsize="x-small", ncol=max(1, len(all_keys) // 4), loc="upper right")
        ax.set_title("Parameter Evolution (gen best)")
    else:
        _plot_percentile_curves(ax, gens, normed_matrix)
        ax.set_title(f"Parameter Distribution ({len(all_keys)} params, gen best)")

    _add_resume_markers(ax, resume_gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized value [0=p_min, 1=p_max]")
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
# Binning helper
# ---------------------------------------------------------------------------
def bin_indices(values: npt.NDArray[np.float64], edges: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Map values to 0-based bin indices for ``len(edges)-1`` bins, clamped to range.

    Inverse-edge convention matching np.digitize: bin i covers [edges[i], edges[i+1]).
    Values outside [edges[0], edges[-1]] clamp to the first/last bin.
    """
    n_bins = len(edges) - 1
    return np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)  # type: ignore[return-value]


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

    bidx = bin_indices(energy, edges)

    for b in range(n_bins):
        mask = bidx == b
        if np.any(mask):
            min_vals[b] = float(np.nanmin(values[mask]))
            max_vals[b] = float(np.nanmax(values[mask]))

    return centers, min_vals, max_vals


def classify_trajectories(
    final_records: npt.NDArray[np.float64],
    heat_flux_limit: float | None = None,
    g_load_limit: float | None = None,
    heat_load_limit: float | None = None,
) -> npt.NDArray[np.int8]:
    """Classify each trajectory as OK (0), constrained (1), or failed (2).

    - OK: captured (ecc < 1, not pending crash) and within all constraint limits
    - Constrained: captured but exceeds heat flux, g-load, or heat load limit
    - Failed: crash, hyperbolic exit, timeout, or pending crash (ifinal=4)
    """
    n = len(final_records)
    classification = np.full(n, TRAJ_FAILED, dtype=np.int8)

    # Captured = exited atmosphere (ifinal=3) on a bound orbit (ecc < 1)
    captured = is_captured(final_records)
    classification[captured] = TRAJ_OK

    # Downgrade captured trajectories that violate constraints
    constrained = np.zeros(n, dtype=bool)
    if heat_flux_limit is not None:
        constrained = constrained | (final_records[:, _FR_MAX_HEAT_FLUX] > heat_flux_limit)
    if g_load_limit is not None:
        constrained = constrained | (final_records[:, _FR_MAX_G_LOAD] > g_load_limit)
    if heat_load_limit is not None:
        hl = final_records[:, _FR_INTEGRATED_FLUX] * 1e3  # MJ/m² → kJ/m²
        constrained = constrained | (hl > heat_load_limit)
    classification[captured & constrained] = TRAJ_CONSTRAINED

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


def _corridor_panel(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path | None,
    y_col: int,
    y_label: str,
    title: str,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
    corridor_data: dict[str, Any] | None = None,
    envelope: bool = False,
    ax: plt.Axes | None = None,  # type: ignore[name-defined]
) -> None:
    """Shared builder for energy-domain (corridor) panels.

    Parameters
    ----------
    output:
        SVG output path. When ``ax`` is supplied this is ignored (caller owns the figure).
    y_col:
        Column index in the (N, 17) trajectory array for the y-axis.
    y_label, title:
        Axis label and chart title.
    corridor_data:
        Optional 4-layer corridor zone fills (used by pdyn panel).
    envelope:
        When True, draw a captured min/max envelope fill behind the spaghetti.
    ax:
        Pre-existing axes to draw into. When None a standalone figure is created and
        saved to *output*. When provided the caller owns the figure lifecycle.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    else:
        fig = None

    assert ax is not None  # always true after the branch above; quiets mypy

    if corridor_data is not None:
        e_bins = corridor_data["energy_bins"]
        crash_pdyn = corridor_data["envelope_crash_pdyn"]
        restricted_max = corridor_data["envelope_restricted_max_pdyn"]
        restricted_min = corridor_data["envelope_restricted_min_pdyn"]
        capture_pdyn = corridor_data["envelope_capture_pdyn"]

        ax.fill_between(e_bins, restricted_max, crash_pdyn, color=COLOR_WORST, alpha=0.15, label="Crash zone")
        ax.fill_between(e_bins, restricted_max, restricted_min, color="white", alpha=0.6)
        ax.fill_between(e_bins, restricted_min, capture_pdyn, color="#cccccc", alpha=0.3, label="Transition")
        ax.fill_between(e_bins, capture_pdyn, 0, color=COLOR_WORST, alpha=0.15, label="Hyperbolic zone")

    if envelope:
        centers, min_vals, max_vals = _compute_envelope(trajectories, energy_col=_TC_ENERGY, value_col=y_col, traj_class=traj_class)
        valid = ~np.isnan(min_vals)
        if np.any(valid):
            ax.fill_between(centers[valid], min_vals[valid], max_vals[valid], color=COLOR_CAPTURE, alpha=0.15, label="Captured envelope")

    _draw_spaghetti(ax, trajectories, traj_class, x_col=_TC_ENERGY, y_col=y_col)
    _draw_nominals(ax, x_col=_TC_ENERGY, y_col=y_col, ref_nominal=ref_nominal, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if standalone:
        assert fig is not None
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="small")
        sns.despine(fig=fig)
        assert output is not None
        _save_svg(fig, output)


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
    _corridor_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_PDYN,
        y_label="Dynamic pressure (kPa)",
        title="Corridor — Dynamic Pressure",
        ref_nominal=ref_nominal,
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
        corridor_data=corridor_data,
    )


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
    _corridor_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_INCL,
        y_label="Inclination (deg)",
        title="Corridor — Inclination",
        ref_nominal=ref_nominal,
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
        envelope=True,
    )


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
    _corridor_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_BANK,
        y_label="Bank angle (deg)",
        title="Corridor — Bank Angle",
        ref_nominal=ref_nominal,
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
        envelope=True,
    )


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
        ax.plot(undispersed_nominal[:, _TC_TIME], undispersed_nominal[:, y_col], color=COLOR_NOMINAL_UNDISPERSED, linewidth=1.5, zorder=10, label="Undispersed")
    if best_nominal is not None:
        ax.plot(best_nominal[:, _TC_TIME], best_nominal[:, y_col], color=COLOR_NOMINAL_BEST, linewidth=1.5, zorder=10, label="Best MC")


def _time_series_panel(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    y_col: int,
    y_label: str,
    title: str,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
    limit_value: float | None = None,
    limit_label: str | None = None,
    fixed_hline: float | None = None,
    fixed_hline_label: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Shared builder for time-domain MC spaghetti panels (x = time column).

    Parameters
    ----------
    y_col:
        Column index in the (N, 17) trajectory array for the y-axis.
    y_label, title:
        Axis label and chart title.
    limit_value, limit_label:
        Optional constraint limit horizontal line (red dashed).
    fixed_hline, fixed_hline_label:
        Optional fixed reference horizontal line (grey dashed, e.g. density ratio=1).
    """
    fig, ax = plt.subplots(figsize=figsize or FULL_WIDTH, dpi=DPI)
    _draw_spaghetti(ax, trajectories, traj_class, x_col=_TC_TIME, y_col=y_col)
    _draw_time_nominals(ax, y_col=y_col, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    if limit_value is not None:
        ax.axhline(limit_value, color=COLOR_WORST, linestyle="--", linewidth=1.0, label=limit_label or f"Limit ({limit_value})")

    if fixed_hline is not None:
        ax.axhline(fixed_hline, color="grey", linestyle="--", linewidth=1.0, label=fixed_hline_label or str(fixed_hline))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 10: Altitude vs time (full width)
# ---------------------------------------------------------------------------
def chart_altitude_time(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 10: Altitude vs time MC spaghetti with nominal overlays."""
    _time_series_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_ALT,
        y_label="Altitude (km)",
        title="Altitude vs Time",
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
    )


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
    figsize: tuple[float, float] | None = None,
) -> None:
    """Panel 11: Heat flux vs time MC spaghetti with optional constraint line and nominals."""
    limit_label = f"Limit ({limit_kw_m2:.0f} kW/m\u00b2)" if limit_kw_m2 is not None else None
    _time_series_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_HEAT_FLUX,
        y_label="Heat flux (kW/m\u00b2)",
        title="Heat Flux vs Time",
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
        limit_value=limit_kw_m2,
        limit_label=limit_label,
        figsize=figsize,
    )


# ---------------------------------------------------------------------------
# Panel 11b: Heat load vs time
# ---------------------------------------------------------------------------
def chart_heat_load_time(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    limit_kj_m2: float | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
    figsize: tuple[float, float] | None = None,
    unit: str = "kJ",
) -> None:
    """Cumulative heat load vs time MC spaghetti with optional constraint line.

    unit="MJ" rescales axis and limit to MJ/m^2 (the paper's textual unit); the
    trajectory pipeline stays in kJ/m^2 and input arrays are not mutated.
    """

    def _scaled(t: npt.NDArray[np.float64] | None) -> npt.NDArray[np.float64] | None:
        if t is None:
            return None
        t = t.copy()
        t[:, _TC_HEAT_LOAD] *= 1e-3
        return t

    limit = limit_kj_m2
    if unit == "MJ":
        trajectories = [s for s in (_scaled(t) for t in trajectories) if s is not None]
        undispersed_nominal = _scaled(undispersed_nominal)
        best_nominal = _scaled(best_nominal)
        limit = limit_kj_m2 * 1e-3 if limit_kj_m2 is not None else None
    limit_label = f"Limit ({limit:.0f} {unit}/m\u00b2)" if limit is not None else None
    _time_series_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_HEAT_LOAD,
        y_label=f"Heat load ({unit}/m\u00b2)",
        title="Cumulative Heat Load vs Time",
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
        limit_value=limit,
        limit_label=limit_label,
        figsize=figsize,
    )


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
    figsize: tuple[float, float] | None = None,
) -> None:
    """Panel 12: G-load vs time MC spaghetti with optional constraint line and nominals."""
    limit_label = f"Limit ({limit_g:.1f} g)" if limit_g is not None else None
    _time_series_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_GLOAD,
        y_label="G-load (g)",
        title="G-Load vs Time",
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
        limit_value=limit_g,
        limit_label=limit_label,
        figsize=figsize,
    )


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
    _time_series_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_BANK,
        y_label="Bank angle (deg)",
        title="Bank Angle vs Time",
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
    )


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
    _time_series_panel(
        trajectories,
        traj_class,
        output,
        y_col=_TC_NAV_DENS,
        y_label="Nav density ratio",
        title="Navigation Density Ratio vs Time",
        undispersed_nominal=undispersed_nominal,
        best_nominal=best_nominal,
        fixed_hline=1.0,
        fixed_hline_label="Perfect estimate",
    )


# ---------------------------------------------------------------------------
# Panel 14b: Objective cost distribution — histogram + CDF (all sims)
# ---------------------------------------------------------------------------
def chart_cost_objective(
    final_records: npt.NDArray[np.float64],
    output: Path,
    *,
    dv_threshold: float = 1000.0,
    g_load_limit: float = 15.0,
    heat_flux_limit: float = 200.0,
    heat_load_limit: float = 25000.0,
    g_load_weight: float = 1000.0,
    heat_flux_weight: float = 1000.0,
    heat_load_weight: float = 1000.0,
    cost_transform: str = "linear",
) -> None:
    """Objective cost histogram with CDF overlay (all sims, including crashes)."""
    from aerocapture.training.evaluate import compute_cost

    n = final_records.shape[0]
    costs = np.array(
        [
            compute_cost(
                final_records[i : i + 1],
                dv_threshold=dv_threshold,
                g_load_limit=g_load_limit,
                heat_flux_limit=heat_flux_limit,
                heat_load_limit=heat_load_limit,
                g_load_weight=g_load_weight,
                heat_flux_weight=heat_flux_weight,
                heat_load_weight=heat_load_weight,
                cost_transform=cost_transform,
            )
            for i in range(n)
        ]
    )

    log_costs = np.log10(np.maximum(costs, 1e-6))

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)

    ax1.hist(log_costs, bins=30, color=COLOR_CAPTURE, alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Objective Cost")
    ax1.set_ylabel("Count")

    tick_pos, tick_labels = _log10_ticks(np.maximum(costs, 1e-6))
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_labels)

    ax2 = ax1.twinx()
    sorted_log = np.sort(log_costs)
    cdf = np.arange(1, len(sorted_log) + 1) / len(sorted_log)
    ax2.plot(sorted_log, cdf, color=COLOR_MEAN, linewidth=1.5, label="CDF")
    ax2.set_ylabel("CDF")
    ax2.set_ylim(0, 1.05)

    for pct, ls in [(5, ":"), (50, "--"), (95, "-.")]:
        val = float(np.percentile(log_costs, pct))
        ax1.axvline(val, color=COLOR_WORST, linestyle=ls, linewidth=0.8, label=f"p{pct}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="x-small")

    ax1.set_title("Objective Cost Distribution (all sims)")
    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 15: Total DV distribution — histogram + CDF + percentile markers
# ---------------------------------------------------------------------------
def chart_dv_distribution(final_records: npt.NDArray[np.float64], output: Path) -> None:
    """Panel 15: Total DV histogram (log10 x) with CDF overlay and percentile markers. Captured only."""
    captured = is_captured(final_records)
    if not np.any(captured):
        fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
        ax.text(0.5, 0.5, "No captured trajectories", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title("Total \u0394V Distribution (captured only)")
        _save_svg(fig, output)
        return
    dv = _clip_dv(final_records[captured, _FR_DV_TOTAL])
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

    ax1.set_title("Total \u0394V Distribution (captured only)")
    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Panel 16: Individual burn DV histograms (stacked, log10 x)
# ---------------------------------------------------------------------------
def chart_dv_individual_burns(final_records: npt.NDArray[np.float64], output: Path) -> None:
    """Panel 16: 3-row subplot histograms for |dv1|, |dv2|, |dv3| on log10 x-axis. Captured only."""
    captured = is_captured(final_records)
    if not np.any(captured):
        fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
        ax.text(0.5, 0.5, "No captured trajectories", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title("Individual Burn \u0394V (captured only)")
        _save_svg(fig, output)
        return
    cap = final_records[captured]
    burns = [
        (np.abs(cap[:, _FR_DV1]), "#1f77b4", "|DV1| (periapsis)"),
        (np.abs(cap[:, _FR_DV2]), "#ff7f0e", "|DV2| (circularization)"),
        (np.abs(cap[:, _FR_DV3]), "#2ca02c", "|DV3| (inclination)"),
    ]

    # Shared tick range from all burns combined
    all_raw = np.concatenate([b[0] for b in burns])
    tick_pos, tick_labels = _log10_ticks(all_raw)

    fig, axes = plt.subplots(3, 1, figsize=(FULL_WIDTH[0], FULL_WIDTH[1] * 1.8), dpi=DPI, sharex=True)

    for ax, (raw, color, label) in zip(axes, burns, strict=True):
        log_vals = np.log10(_clip_dv(raw))
        ax.hist(log_vals, bins=25, color=color, alpha=0.7, edgecolor="white", label=label)
        ax.set_ylabel("Count")
        ax.legend(fontsize="x-small", loc="upper right")

    # Only bottom axes gets x-axis labels
    axes[-1].set_xlabel("\u0394V (m/s)")
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_labels)
    axes[0].set_title("Individual Burn \u0394V (captured only)")

    fig.tight_layout()
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

    captured = is_captured(final_records)

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    if np.any(captured):
        ax.scatter(exit_fpa[captured], exit_v[captured], s=sizes[captured], color=COLOR_CAPTURE, alpha=0.7, label="Captured")
    if np.any(~captured):
        ax.scatter(exit_fpa[~captured], exit_v[~captured], s=sizes[~captured], color=COLOR_HYPERBOLIC, marker="x", label="Failed")

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
    traj_class: npt.NDArray[np.int8] | None = None,
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

        if traj_class is not None:
            _cls_labels = {TRAJ_OK: "Captured", TRAJ_CONSTRAINED: "Constrained", TRAJ_FAILED: "Failed"}
            for cls in (TRAJ_OK, TRAJ_CONSTRAINED, TRAJ_FAILED):
                mask = traj_class == cls
                if not np.any(mask):
                    continue
                marker = "x" if cls == TRAJ_FAILED else "o"
                label = _cls_labels[cls] if i == 0 else None
                ax.scatter(x[mask], log_dv[mask], s=8, alpha=0.5, color=_TRAJ_COLORS[cls], marker=marker, label=label)
            # Regression on captured points only
            captured_mask = (traj_class == TRAJ_OK) | (traj_class == TRAJ_CONSTRAINED)
            reg_x = x[captured_mask]
            reg_y = log_dv[captured_mask]
        else:
            ax.scatter(x, log_dv, s=8, alpha=0.5, color=COLOR_CAPTURE)
            reg_x = x
            reg_y = log_dv

        # Linear regression (skip if all x values are identical — e.g. a zero-variance dispersion field)
        finite = np.isfinite(reg_x) & np.isfinite(reg_y)
        if np.sum(finite) > 2 and np.ptp(reg_x[finite]) > 0:
            result = stats.linregress(reg_x[finite], reg_y[finite])
            x_range = np.array([float(np.min(reg_x[finite])), float(np.max(reg_x[finite]))])
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
    if traj_class is not None:
        fig.legend(loc="upper right", fontsize=7, markerscale=1.5)
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


# ---------------------------------------------------------------------------
# Sensitivity Analysis charts (Part 3)
# ---------------------------------------------------------------------------
def chart_morris_scatter(morris_data: dict[str, Any], output: Path) -> None:
    """Morris mu* horizontal bar chart sorted by importance."""
    names = morris_data["names"]
    mu_star = np.asarray(morris_data["mu_star"], dtype=float)
    sigma = np.asarray(morris_data["sigma"], dtype=float)

    # Sort by mu_star descending (plot bottom-to-top so largest is at the top)
    order = np.argsort(mu_star)
    sorted_names = [names[i] for i in order]
    sorted_mu = mu_star[order]
    sorted_sigma = sigma[order]

    n = len(names)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.28)), dpi=DPI)
    y = np.arange(n)
    ax.barh(y, sorted_mu, height=0.6, color=COLOR_BEST, alpha=0.85, label="mu* (importance)")
    ax.barh(y, sorted_sigma, height=0.6, color=COLOR_MEAN, alpha=0.5, label="sigma (nonlinearity)")

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Elementary effect magnitude")
    ax.set_title("Morris Screening: Parameter Importance")
    ax.legend(fontsize=8, loc="lower right")
    sns.despine(fig=fig)
    fig.tight_layout()
    _save_svg(fig, output)


def chart_sobol_bars(sobol_data: dict[str, Any], output: Path) -> None:
    """Grouped bar chart of Sobol S1 and ST indices."""
    names = sobol_data["names"]
    s1 = np.asarray(sobol_data["S1"], dtype=float)
    st = np.asarray(sobol_data["ST"], dtype=float)
    s1_conf = np.asarray(sobol_data["S1_conf"], dtype=float)
    st_conf = np.asarray(sobol_data["ST_conf"], dtype=float)

    n = len(names)
    width = max(10, n * 0.6)
    fig, ax = plt.subplots(figsize=(width, 4), dpi=DPI)

    x = np.arange(n)
    bar_w = 0.35
    ax.bar(x - bar_w / 2, s1, bar_w, yerr=s1_conf, color=COLOR_BEST, label="S1", capsize=3)
    ax.bar(x + bar_w / 2, st, bar_w, yerr=st_conf, color=COLOR_MEAN, label="ST", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sensitivity index")
    ax.set_title("Sobol Sensitivity Indices")
    ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_sobol_heatmap(sobol_data: dict[str, Any], output: Path) -> None:
    """Heatmap of Sobol S2 interaction matrix."""
    names = sobol_data["names"]
    s2 = np.asarray(sobol_data["S2"], dtype=float)

    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    sns.heatmap(s2, annot=True, fmt=".2f", cmap="YlOrRd", square=True, xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title("Sobol S2: Parameter Interactions")
    _save_svg(fig, output)


# ---------------------------------------------------------------------------
# Generic line chart helper (used by RL report)
# ---------------------------------------------------------------------------
def save_line_chart(
    x: list[float],
    y: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Write a simple line chart SVG (seaborn-themed, full width)."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    ax.plot(x, y, linewidth=1.5, color=COLOR_BEST)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    _save_svg(fig, output_path)


# ---------------------------------------------------------------------------
# Island-model charts (Part 0 of the multi-island report)
# ---------------------------------------------------------------------------
def chart_island_convergence_overlay(
    records_by_island: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    """Overlay per-island running-min cost vs generation.

    Three colored lines (PSO=blue, GA=orange, DE=green) on a log-scale y-axis.
    The series uses validated cost when present (`validation.rms_cost`,
    promoted-best), else `best_cost` (the per-gen training argmin written by
    TrainingLogger), computing a running min per-island so the chart is
    meaningful even before any validation gate has fired. Islands with no
    finite costs are skipped.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"pso": "tab:blue", "ga": "tab:orange", "de": "tab:green"}
    plotted_any = False
    for name in ("pso", "ga", "de"):
        records = records_by_island.get(name, [])
        if not records:
            continue
        gens: list[int] = []
        running_costs: list[float] = []
        running_min = float("inf")
        for r in records:
            # Prefer the validated cost (per-island promotion metric), then
            # fall back to the per-gen training argmin (`best_cost`).
            val = r.get("validation") or {}
            candidate = val.get("rms_cost")
            if candidate is None or not np.isfinite(candidate):
                candidate = r.get("best_cost")
            if candidate is None or not np.isfinite(candidate):
                continue
            running_min = min(running_min, float(candidate))
            gens.append(r["generation"])
            running_costs.append(running_min)
        if not running_costs:
            continue
        ax.plot(
            gens,
            running_costs,
            color=colors.get(name, "k"),
            label=name.upper(),
            linewidth=1.5,
        )
        plotted_any = True

    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Running-min cost (RMS)")
    ax.set_title("Per-island convergence")
    if plotted_any:
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No finite cost data yet", ha="center", va="center", transform=ax.transAxes)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def chart_migration_timeline(
    migration_log: list[Any],
    n_gen: int,
    output: Path,
) -> None:
    """Scatter of migration events: x=generation, y=src->dst channel, color=F_migrant.

    Accepts either `MigrationEvent` dataclass instances (the in-memory form
    on `IslandModel.migration_log`) or dicts (the JSON-loaded form). Field
    access goes through `_field` so both shapes work without coupling this
    module to `island_model`.
    """

    def _field(event: Any, key: str) -> Any:
        if isinstance(event, dict):
            return event[key]
        return getattr(event, key)

    if not migration_log:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(
            0.5,
            0.5,
            "No migration events",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, max(n_gen, 1))
        fig.tight_layout()
        fig.savefig(output, format="svg")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    channels = sorted({f"{_field(e, 'src_island')}->{_field(e, 'dst_island')}" for e in migration_log})
    channel_y = {ch: i for i, ch in enumerate(channels)}
    gens = [_field(e, "gen") for e in migration_log]
    ys = [channel_y[f"{_field(e, 'src_island')}->{_field(e, 'dst_island')}"] for e in migration_log]
    fs = [_field(e, "F_migrant") for e in migration_log]

    sc = ax.scatter(gens, ys, c=fs, cmap="viridis", s=20)
    fig.colorbar(sc, ax=ax, label="F_migrant")
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_xlabel("Generation")
    ax.set_xlim(0, max(n_gen, max(gens) + 1))
    ax.set_title(f"Migration events ({len(migration_log)} total)")
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)
