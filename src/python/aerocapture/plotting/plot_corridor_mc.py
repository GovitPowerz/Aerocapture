"""Monte Carlo ensemble plots with CDF of correction cost.

Replaces MATLAB Trace_Corridor_MC.m: corridor plot with MC envelope,
time-series overlays, and correction cost CDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerocapture.io import parse_final, parse_photo
from aerocapture.plotting.corridor import draw_corridor, segment_mc_trajectories
from aerocapture.plotting.stats import empirical_cdf


def plot_corridor_mc(
    photo_mc: pd.DataFrame | str | Path,
    final_mc: pd.DataFrame | str | Path,
    photo_nominal: pd.DataFrame | str | Path | None = None,
    overshoot: np.ndarray | None = None,
    undershoot: np.ndarray | None = None,
    cost_column: int = 47,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Create Monte Carlo analysis plots.

    Args:
        photo_mc: MC photo data (all runs concatenated).
        final_mc: MC final conditions data.
        photo_nominal: Nominal trajectory photo data (optional overlay).
        overshoot: Overshoot corridor boundary array.
        undershoot: Undershoot corridor boundary array.
        cost_column: Column index in final data for correction cost (m/s).
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    if isinstance(photo_mc, (str, Path)):
        photo_mc = parse_photo(photo_mc)
    if isinstance(final_mc, (str, Path)):
        final_mc = parse_final(final_mc)
    if isinstance(photo_nominal, (str, Path)):
        photo_nominal = parse_photo(photo_nominal)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Split MC photo data into individual trajectories
    trajectories = segment_mc_trajectories(photo_mc.values)

    # Energy vs Dynamic Pressure (corridor)
    ax = axes[0, 0]
    if overshoot is not None:
        draw_corridor(ax, overshoot, undershoot)
    for traj in trajectories:
        energy = traj[:, 18] / 1e6  # energy column
        pdyn = traj[:, 19] / 1e3  # dynamic_pressure column
        ax.plot(energy, pdyn, "b-", alpha=0.1, linewidth=0.5)
    if photo_nominal is not None:
        ax.plot(photo_nominal["energy"] / 1e6, photo_nominal["dynamic_pressure"] / 1e3, "r-", linewidth=2, label="Nominal")
        ax.legend()
    ax.set_xlabel("Orbital Energy (MJ/kg)")
    ax.set_ylabel("Dynamic Pressure (kPa)")
    ax.set_xlim(-7, 5)
    ax.set_ylim(0, 2.2)

    # Energy vs Inclination
    ax = axes[0, 1]
    for traj in trajectories:
        energy = traj[:, 18] / 1e6
        incl = traj[:, 9]  # inclination column
        ax.plot(energy, incl, "b-", alpha=0.1, linewidth=0.5)
    if photo_nominal is not None:
        ax.plot(photo_nominal["energy"] / 1e6, photo_nominal["inclination"], "r-", linewidth=2)
    ax.set_xlabel("Orbital Energy (MJ/kg)")
    ax.set_ylabel("Inclination (deg)")

    # Energy vs Bank Angle
    ax = axes[1, 0]
    for traj in trajectories:
        energy = traj[:, 18] / 1e6
        bank = traj[:, 14]  # bank_angle column
        ax.plot(energy, bank, "b-", alpha=0.1, linewidth=0.5)
    if photo_nominal is not None:
        ax.plot(photo_nominal["energy"] / 1e6, photo_nominal["bank_angle"], "r-", linewidth=2)
    ax.set_xlabel("Orbital Energy (MJ/kg)")
    ax.set_ylabel("Bank Angle (deg)")

    # Correction Cost Histogram + CDF
    ax = axes[1, 1]
    cost_idx = min(cost_column, final_mc.shape[1] - 1)
    cost = final_mc.iloc[:, cost_idx].values
    # Filter out extreme values
    valid = np.isfinite(cost) & (np.abs(cost) < 1e10)
    cost_valid = cost[valid]

    if len(cost_valid) > 0:
        ax.hist(cost_valid, bins=30, density=True, alpha=0.5, color="blue", label="Histogram")
        ax2 = ax.twinx()
        xcdf, ycdf = empirical_cdf(cost_valid)
        ax2.plot(xcdf[np.isfinite(xcdf)], ycdf[np.isfinite(xcdf)], "r-", linewidth=2, label="CDF")
        ax2.set_ylabel("CDF")
        ax2.set_ylim(0, 1.05)
    ax.set_xlabel("Correction Cost (m/s)")
    ax.set_ylabel("Density")

    fig.suptitle("Monte Carlo Analysis", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
