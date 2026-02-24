"""Animated trajectory visualization.

Replaces MATLAB Trace_Corridor_MC_Film.m: generates animation frames
showing MC trajectory evolution across NN training generations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerocapture.io import parse_final, parse_photo
from aerocapture.plotting.corridor import draw_corridor, segment_mc_trajectories
from aerocapture.plotting.stats import empirical_cdf


def plot_corridor_animation(
    frames: list[dict],
    overshoot: np.ndarray | None = None,
    undershoot: np.ndarray | None = None,
    cost_column: int = 47,
    output_dir: str | Path | None = None,
    fps: int = 2,
) -> list[plt.Figure]:
    """Generate animation frames comparing NN training generations.

    Args:
        frames: List of dicts, each with keys:
            - 'label': Frame label (e.g. 'Gen 142')
            - 'photo_mc': MC photo data (DataFrame or path)
            - 'final_mc': MC final conditions (DataFrame or path)
        overshoot/undershoot: Corridor boundary arrays.
        cost_column: Column index for correction cost.
        output_dir: Directory to save individual frame PNGs.
        fps: Frames per second (for info only, actual animation done externally).

    Returns:
        List of Matplotlib Figures (one per frame).
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for i, frame_data in enumerate(frames):
        label = frame_data["label"]
        photo_mc = frame_data["photo_mc"]
        final_mc = frame_data["final_mc"]

        if isinstance(photo_mc, (str, Path)):
            photo_mc = parse_photo(photo_mc)
        if isinstance(final_mc, (str, Path)):
            final_mc = parse_final(final_mc)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        trajectories = segment_mc_trajectories(photo_mc.values)

        # Corridor
        ax = axes[0, 0]
        if overshoot is not None:
            draw_corridor(ax, overshoot, undershoot)
        for traj in trajectories:
            ax.plot(traj[:, 18] / 1e6, traj[:, 19] / 1e3, "b-", alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Energy (MJ/kg)")
        ax.set_ylabel("Dyn. Pressure (kPa)")
        ax.set_xlim(-7, 5)
        ax.set_ylim(0, 2.2)

        # Inclination
        ax = axes[0, 1]
        for traj in trajectories:
            ax.plot(traj[:, 18] / 1e6, traj[:, 9], "b-", alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Energy (MJ/kg)")
        ax.set_ylabel("Inclination (deg)")

        # Bank angle
        ax = axes[1, 0]
        for traj in trajectories:
            ax.plot(traj[:, 18] / 1e6, traj[:, 14], "b-", alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Energy (MJ/kg)")
        ax.set_ylabel("Bank Angle (deg)")

        # Cost CDF
        ax = axes[1, 1]
        cost_idx = min(cost_column, final_mc.shape[1] - 1)
        cost = final_mc.iloc[:, cost_idx].values
        valid = np.isfinite(cost) & (np.abs(cost) < 1e10)
        cost_valid = cost[valid]
        if len(cost_valid) > 0:
            ax.hist(cost_valid, bins=30, density=True, alpha=0.5, color="blue")
            ax2 = ax.twinx()
            xcdf, ycdf = empirical_cdf(cost_valid)
            finite = np.isfinite(xcdf)
            ax2.plot(xcdf[finite], ycdf[finite], "r-", linewidth=2)
            ax2.set_ylim(0, 1.05)
        ax.set_xlabel("Cost (m/s)")

        fig.suptitle(f"Training Progress: {label}", fontsize=14)
        fig.tight_layout()

        if output_dir:
            fig.savefig(output_dir / f"frame_{i:04d}.png", dpi=100, bbox_inches="tight")

        figures.append(fig)

    return figures
