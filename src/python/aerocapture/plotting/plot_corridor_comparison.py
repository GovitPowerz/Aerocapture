"""Two-scenario side-by-side comparison plots.

Replaces MATLAB Trace_Corridor_MC_duo.m: compares FTC vs NN guidance
side by side with corridor, inclination, bank angle, and cost CDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerocapture.io import parse_final, parse_photo
from aerocapture.plotting.corridor import draw_corridor, segment_mc_trajectories
from aerocapture.plotting.stats import empirical_cdf


def plot_corridor_comparison(
    photo_mc_a: pd.DataFrame | str | Path,
    final_mc_a: pd.DataFrame | str | Path,
    photo_mc_b: pd.DataFrame | str | Path,
    final_mc_b: pd.DataFrame | str | Path,
    label_a: str = "FTC",
    label_b: str = "NN",
    photo_nom_a: pd.DataFrame | str | Path | None = None,
    photo_nom_b: pd.DataFrame | str | Path | None = None,
    overshoot: np.ndarray | None = None,
    undershoot: np.ndarray | None = None,
    cost_column: int = 47,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Create side-by-side comparison of two guidance scenarios.

    Args:
        photo_mc_a/b: MC photo data for scenarios A and B.
        final_mc_a/b: MC final conditions for scenarios A and B.
        label_a/b: Labels for the two scenarios.
        photo_nom_a/b: Nominal trajectories (optional).
        overshoot/undershoot: Corridor boundary arrays.
        cost_column: Column index for correction cost.
        output_path: Save path for the figure.

    Returns:
        Matplotlib Figure.
    """
    if isinstance(photo_mc_a, (str, Path)):
        photo_mc_a = parse_photo(photo_mc_a)
    if isinstance(photo_mc_b, (str, Path)):
        photo_mc_b = parse_photo(photo_mc_b)
    if isinstance(final_mc_a, (str, Path)):
        final_mc_a = parse_final(final_mc_a)
    if isinstance(final_mc_b, (str, Path)):
        final_mc_b = parse_final(final_mc_b)
    if isinstance(photo_nom_a, (str, Path)):
        photo_nom_a = parse_photo(photo_nom_a)
    if isinstance(photo_nom_b, (str, Path)):
        photo_nom_b = parse_photo(photo_nom_b)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    for col, (photo_mc, final_mc, photo_nom, label) in enumerate([
        (photo_mc_a, final_mc_a, photo_nom_a, label_a),
        (photo_mc_b, final_mc_b, photo_nom_b, label_b),
    ]):
        trajectories = segment_mc_trajectories(photo_mc.values)

        # Corridor
        ax = axes[0, col]
        if overshoot is not None:
            draw_corridor(ax, overshoot, undershoot)
        for traj in trajectories:
            ax.plot(traj[:, 18] / 1e6, traj[:, 19] / 1e3, "b-", alpha=0.1, linewidth=0.5)
        if photo_nom is not None:
            ax.plot(photo_nom["energy"] / 1e6, photo_nom["dynamic_pressure"] / 1e3, "r-", linewidth=2)
        ax.set_title(f"{label} - Corridor")
        ax.set_xlabel("Energy (MJ/kg)")
        ax.set_ylabel("Dyn. Pressure (kPa)")
        ax.set_xlim(-7, 5)
        ax.set_ylim(0, 2.2)

        # Inclination
        ax = axes[1, col]
        for traj in trajectories:
            ax.plot(traj[:, 18] / 1e6, traj[:, 9], "b-", alpha=0.1, linewidth=0.5)
        if photo_nom is not None:
            ax.plot(photo_nom["energy"] / 1e6, photo_nom["inclination"], "r-", linewidth=2)
        ax.set_title(f"{label} - Inclination")
        ax.set_xlabel("Energy (MJ/kg)")
        ax.set_ylabel("Inclination (deg)")

        # Bank Angle
        ax = axes[2, col]
        for traj in trajectories:
            ax.plot(traj[:, 18] / 1e6, traj[:, 14], "b-", alpha=0.1, linewidth=0.5)
        if photo_nom is not None:
            ax.plot(photo_nom["energy"] / 1e6, photo_nom["bank_angle"], "r-", linewidth=2)
        ax.set_title(f"{label} - Bank Angle")
        ax.set_xlabel("Energy (MJ/kg)")
        ax.set_ylabel("Bank Angle (deg)")

        # Cost CDF
        ax = axes[3, col]
        cost_idx = min(cost_column, final_mc.shape[1] - 1)
        cost = final_mc.iloc[:, cost_idx].values
        valid = np.isfinite(cost) & (np.abs(cost) < 1e10)
        cost_valid = cost[valid]
        if len(cost_valid) > 0:
            ax.hist(cost_valid, bins=30, density=True, alpha=0.5, color="blue")
            ax2 = ax.twinx()
            xcdf, ycdf = empirical_cdf(cost_valid)
            ax2.plot(xcdf[np.isfinite(xcdf)], ycdf[np.isfinite(xcdf)], "r-", linewidth=2)
            ax2.set_ylim(0, 1.05)
        ax.set_title(f"{label} - Cost Distribution")
        ax.set_xlabel("Correction Cost (m/s)")

    fig.suptitle("Guidance Comparison", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
