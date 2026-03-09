"""Multi-scenario overlay plots for comparing training generations.

Replaces MATLAB Trace_Corridor_MC_multi.m: N-row subplot grid comparing
different NN generations across corridor, inclination, and cost metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aerocapture.io import parse_final, parse_photo
from aerocapture.plotting.corridor import draw_corridor, segment_mc_trajectories
from aerocapture.plotting.stats import empirical_cdf


def plot_corridor_multi(
    scenarios: list[dict],
    overshoot: np.ndarray | None = None,
    undershoot: np.ndarray | None = None,
    cost_column: int = 47,
    cost_threshold: float = 150.0,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Create multi-scenario comparison grid.

    Args:
        scenarios: List of dicts, each with keys:
            - 'label': Scenario label (e.g. 'Gen 2')
            - 'photo_mc': MC photo data (DataFrame or path)
            - 'final_mc': MC final conditions (DataFrame or path)
            - 'photo_nom': Optional nominal trajectory
        overshoot/undershoot: Corridor boundary arrays.
        cost_column: Column index for correction cost.
        cost_threshold: Cost threshold for success marking (m/s).
        output_path: Save path for the figure.

    Returns:
        Matplotlib Figure.
    """
    n = len(scenarios)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, scenario in enumerate(scenarios):
        label = scenario["label"]
        photo_mc = scenario["photo_mc"]
        final_mc = scenario["final_mc"]
        photo_nom = scenario.get("photo_nom")

        if isinstance(photo_mc, (str, Path)):
            photo_mc = parse_photo(photo_mc)
        if isinstance(final_mc, (str, Path)):
            final_mc = parse_final(final_mc)
        if isinstance(photo_nom, (str, Path)):
            photo_nom = parse_photo(photo_nom)

        trajectories = segment_mc_trajectories(photo_mc.values)

        # Corridor
        ax = axes[row, 0]
        if overshoot is not None:
            draw_corridor(ax, overshoot, undershoot)
        for traj in trajectories:
            ax.plot(traj[:, 18] / 1e6, traj[:, 19] / 1e3, "b-", alpha=0.1, linewidth=0.5)
        if photo_nom is not None:
            ax.plot(photo_nom["energy"] / 1e6, photo_nom["dynamic_pressure"] / 1e3, "r-", linewidth=2)
        ax.set_ylabel(f"{label}\nDyn. Pressure (kPa)")
        ax.set_xlim(-7, 5)
        ax.set_ylim(0, 2.2)
        if row == 0:
            ax.set_title("Corridor")
        if row == n - 1:
            ax.set_xlabel("Energy (MJ/kg)")

        # Inclination
        ax = axes[row, 1]
        for traj in trajectories:
            ax.plot(traj[:, 18] / 1e6, traj[:, 9], "b-", alpha=0.1, linewidth=0.5)
        if photo_nom is not None:
            ax.plot(photo_nom["energy"] / 1e6, photo_nom["inclination"], "r-", linewidth=2)
        if row == 0:
            ax.set_title("Inclination")
        ax.set_ylabel("Inclination (deg)")
        if row == n - 1:
            ax.set_xlabel("Energy (MJ/kg)")

        # Cost CDF (log scale)
        ax = axes[row, 2]
        cost_idx = min(cost_column, final_mc.shape[1] - 1)
        cost = final_mc.iloc[:, cost_idx].values
        valid = np.isfinite(cost) & (np.abs(cost) < 1e10)
        cost_valid = cost[valid]
        n_success = np.sum(cost_valid < cost_threshold)
        if len(cost_valid) > 0:
            xcdf, ycdf = empirical_cdf(cost_valid)
            finite = np.isfinite(xcdf)
            ax.plot(xcdf[finite], ycdf[finite], "b-", linewidth=2)
            ax.axvline(cost_threshold, color="r", linestyle="--", alpha=0.5)
            ax.set_xscale("log")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Success: {n_success}/{len(cost_valid)}" if row == 0 else f"{n_success}/{len(cost_valid)}")
        if row == n - 1:
            ax.set_xlabel("Cost (m/s)")

    fig.suptitle("Multi-Generation Comparison", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
