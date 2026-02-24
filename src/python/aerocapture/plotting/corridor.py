"""Shared corridor-drawing utilities for aerocapture visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def load_corridor_boundaries(
    overshoot_path: str | Path,
    undershoot_path: str | Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load corridor boundary data from files.

    Args:
        overshoot_path: Path to overshoot boundary file (visu.ovr_res).
        undershoot_path: Path to undershoot boundary file (visu.udr_res).

    Returns:
        (overshoot, undershoot): 2D arrays with columns [energy, pressure].
    """
    overshoot = np.loadtxt(overshoot_path)
    undershoot = np.loadtxt(undershoot_path)
    return overshoot, undershoot


def draw_corridor(
    ax: plt.Axes,
    overshoot: npt.NDArray[np.float64] | None = None,
    undershoot: npt.NDArray[np.float64] | None = None,
    color: str = "0.9",
) -> None:
    """Draw overshoot/undershoot corridor boundaries on an axes.

    Fills the constraint regions with a light color.

    Args:
        ax: Matplotlib axes to draw on.
        overshoot: Array with columns [energy (MJ/kg), pressure (kPa)].
        undershoot: Array with columns [energy (MJ/kg), pressure (kPa)].
        color: Fill color for constraint regions.
    """
    if overshoot is not None:
        ax.fill_between(overshoot[:, 0], overshoot[:, 1], 0, color=color, alpha=0.5, label="Overshoot")
    if undershoot is not None:
        ax.fill_between(undershoot[:, 0], undershoot[:, 1], 10, color=color, alpha=0.5, label="Undershoot")


def segment_mc_trajectories(photo_data: npt.NDArray[np.float64], time_col: int = 0) -> list[npt.NDArray[np.float64]]:
    """Split concatenated Monte Carlo photo data into individual trajectories.

    Detects trajectory boundaries where time decreases (restart).

    Args:
        photo_data: Full photo array with all MC runs concatenated.
        time_col: Column index for time.

    Returns:
        List of arrays, one per trajectory.
    """
    time = photo_data[:, time_col]
    # Detect restarts: time jumps backward
    restart_indices = np.where(np.diff(time) < 0)[0] + 1
    splits = np.split(photo_data, restart_indices)
    return [s for s in splits if len(s) > 0]
