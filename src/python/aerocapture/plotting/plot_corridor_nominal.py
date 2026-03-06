"""Nominal FTC vs NN comparison plots.

Replaces MATLAB Trace_Corridor_nom.m: 4 subplots showing energy vs dynamic
pressure (with corridor), energy vs inclination, energy vs bank angle.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aerocapture.io import parse_photo
from aerocapture.plotting.corridor import draw_corridor, load_corridor_boundaries


def plot_corridor_nominal(
    photo_ftc: pd.DataFrame | str | Path,
    photo_nn: pd.DataFrame | str | Path | None = None,
    overshoot_path: str | Path | None = None,
    undershoot_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Create nominal trajectory comparison plots.

    Args:
        photo_ftc: FTC trajectory photo data (DataFrame or path).
        photo_nn: NN trajectory photo data (DataFrame or path), optional.
        overshoot_path: Path to overshoot corridor boundary file.
        undershoot_path: Path to undershoot corridor boundary file.
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    if isinstance(photo_ftc, (str, Path)):
        photo_ftc = parse_photo(photo_ftc)
    if isinstance(photo_nn, (str, Path)):
        photo_nn = parse_photo(photo_nn)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Load corridor boundaries if available
    overshoot = undershoot = None
    if overshoot_path and undershoot_path:
        overshoot, undershoot = load_corridor_boundaries(overshoot_path, undershoot_path)

    # Energy vs Dynamic Pressure (corridor plot)
    ax = axes[0, 0]
    if overshoot is not None:
        draw_corridor(ax, overshoot, undershoot)
    energy_ftc = photo_ftc["energy"] / 1e6  # MJ/kg
    pdyn_ftc = photo_ftc["dynamic_pressure"] / 1e3  # kPa
    ax.plot(energy_ftc, pdyn_ftc, "r-", linewidth=1.5, label="FTC")
    if photo_nn is not None:
        energy_nn = photo_nn["energy"] / 1e6
        pdyn_nn = photo_nn["dynamic_pressure"] / 1e3
        ax.plot(energy_nn, pdyn_nn, "b-", linewidth=1.5, label="NN")
    ax.set_xlabel("Orbital Energy (MJ/kg)")
    ax.set_ylabel("Dynamic Pressure (kPa)")
    ax.legend()
    ax.set_xlim(-7, 5)
    ax.set_ylim(0, 2.2)

    # Energy vs Inclination
    ax = axes[0, 1]
    ax.plot(energy_ftc, photo_ftc["inclination"], "r-", linewidth=1.5, label="FTC")
    if photo_nn is not None:
        ax.plot(energy_nn, photo_nn["inclination"], "b-", linewidth=1.5, label="NN")
    ax.set_xlabel("Orbital Energy (MJ/kg)")
    ax.set_ylabel("Inclination (deg)")
    ax.legend()

    # Energy vs Bank Angle
    ax = axes[1, 0]
    ax.plot(energy_ftc, photo_ftc["bank_angle"], "r-", linewidth=1.5, label="FTC")
    if photo_nn is not None:
        ax.plot(energy_nn, photo_nn["bank_angle"], "b-", linewidth=1.5, label="NN")
    ax.set_xlabel("Orbital Energy (MJ/kg)")
    ax.set_ylabel("Bank Angle (deg)")
    ax.legend()

    # Time vs Altitude
    ax = axes[1, 1]
    ax.plot(photo_ftc["time"], photo_ftc["altitude"], "r-", linewidth=1.5, label="FTC")
    if photo_nn is not None:
        ax.plot(photo_nn["time"], photo_nn["altitude"], "b-", linewidth=1.5, label="NN")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.legend()

    fig.suptitle("Nominal Trajectory Comparison", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
