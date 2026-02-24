"""Publication-quality poster figures.

Replaces MATLAB Trace_Corridor_MC_Poster*.m: creates styled plots
with custom colors and sizing for presentations and publications.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerocapture.io import parse_photo
from aerocapture.plotting.corridor import draw_corridor, segment_mc_trajectories

# Poster color scheme (from MATLAB Poster_Orange variant)
POSTER_BG = np.array([58, 148, 213]) / 255
POSTER_TRACE = np.array([208, 100, 1]) / 255
POSTER_AXES = np.array([33, 29, 119]) / 255


def plot_corridor_poster(
    photo_mc: pd.DataFrame | str | Path,
    photo_nominal: pd.DataFrame | str | Path | None = None,
    overshoot: np.ndarray | None = None,
    undershoot: np.ndarray | None = None,
    output_prefix: str | Path | None = None,
) -> list[plt.Figure]:
    """Create publication-quality poster figures.

    Generates three separate figures:
    1. Inclination evolution
    2. Bank angle decomposition (cos/sin)
    3. Corridor plot

    Args:
        photo_mc: MC photo data.
        photo_nominal: Nominal trajectory (optional).
        overshoot/undershoot: Corridor boundary arrays.
        output_prefix: Prefix for output filenames (adds _incli.png etc).

    Returns:
        List of 3 Matplotlib Figures.
    """
    if isinstance(photo_mc, (str, Path)):
        photo_mc = parse_photo(photo_mc)
    if isinstance(photo_nominal, (str, Path)):
        photo_nominal = parse_photo(photo_nominal)

    trajectories = segment_mc_trajectories(photo_mc.values)
    figures = []

    # Figure 1: Inclination Evolution
    fig1, ax = plt.subplots(figsize=(10, 6))
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 9], color=POSTER_TRACE, alpha=0.15, linewidth=0.8)
    if photo_nominal is not None:
        ax.plot(photo_nominal["time"], photo_nominal["inclination"], color=POSTER_AXES, linewidth=2.5)
    ax.set_xlabel("Time from Entry (s)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Inclination (deg)", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=12)
    fig1.tight_layout()
    if output_prefix:
        fig1.savefig(f"{output_prefix}_incli.png", dpi=300, bbox_inches="tight")
    figures.append(fig1)

    # Figure 2: Bank Angle Decomposition
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for traj in trajectories:
        bank_rad = np.radians(traj[:, 14])
        ax1.plot(traj[:, 0], np.cos(bank_rad), color=POSTER_TRACE, alpha=0.15, linewidth=0.8)
        ax2.plot(traj[:, 0], np.sin(bank_rad), color=POSTER_TRACE, alpha=0.15, linewidth=0.8)
    if photo_nominal is not None:
        bank_rad = np.radians(photo_nominal["bank_angle"])
        ax1.plot(photo_nominal["time"], np.cos(bank_rad), color=POSTER_AXES, linewidth=2.5)
        ax2.plot(photo_nominal["time"], np.sin(bank_rad), color=POSTER_AXES, linewidth=2.5)
    ax1.set_ylabel("cos(Bank Angle)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("sin(Bank Angle)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Time from Entry (s)", fontsize=14, fontweight="bold")
    for a in [ax1, ax2]:
        a.tick_params(labelsize=12)
    fig2.tight_layout()
    if output_prefix:
        fig2.savefig(f"{output_prefix}_bank.png", dpi=300, bbox_inches="tight")
    figures.append(fig2)

    # Figure 3: Corridor Plot
    fig3, ax = plt.subplots(figsize=(10, 6))
    if overshoot is not None:
        draw_corridor(ax, overshoot, undershoot, color="0.85")
    for traj in trajectories:
        ax.plot(traj[:, 18] / 1e6, traj[:, 19] / 1e3, color=POSTER_TRACE, alpha=0.15, linewidth=0.8)
    if photo_nominal is not None:
        ax.plot(photo_nominal["energy"] / 1e6, photo_nominal["dynamic_pressure"] / 1e3, color=POSTER_AXES, linewidth=2.5)
    ax.set_xlabel("Orbital Energy (MJ/kg)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Dynamic Pressure (kPa)", fontsize=14, fontweight="bold")
    ax.set_xlim(-7, 5)
    ax.set_ylim(0, 2.2)
    ax.tick_params(labelsize=12)
    fig3.tight_layout()
    if output_prefix:
        fig3.savefig(f"{output_prefix}_corridor.png", dpi=300, bbox_inches="tight")
    figures.append(fig3)

    return figures
