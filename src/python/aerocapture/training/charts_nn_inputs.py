"""Per-input spaghetti + binned envelope panels for the NN input report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

COLOR_BLUE = "#1f77b4"
COLOR_RED = "#d62728"
BLUE_LOW_DV = 0
RED_HIGH_DV = 1


def binned_band(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_bins: int = 40,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    min_count: int = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Bin samples (x, y) into n_bins on x; return (centers, lo, hi) percentiles
    per bin. Bins with < min_count samples are dropped."""
    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        empty = np.empty(0)
        return empty, empty, empty
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centers: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for b in range(n_bins):
        yb = y[idx == b]
        if yb.size >= min_count:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            lo.append(float(np.percentile(yb, lo_pct)))
            hi.append(float(np.percentile(yb, hi_pct)))
    return np.array(centers), np.array(lo), np.array(hi)


def _class_xy(
    X_list: list[npt.NDArray[np.float64]],
    axis_list: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    input_index: int,
    cls: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    xs: list[npt.NDArray[np.float64]] = []
    ys: list[npt.NDArray[np.float64]] = []
    for i in range(len(X_list)):
        if traj_class[i] == cls:
            xs.append(np.asarray(axis_list[i]))
            ys.append(np.asarray(X_list[i])[:, input_index])
    if not xs:
        empty = np.empty(0)
        return empty, empty
    return np.concatenate(xs), np.concatenate(ys)


def chart_nn_input_panel(
    X_list: list[npt.NDArray[np.float64]],
    axis_list: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    input_index: int,
    name: str,
    in_mask: bool,
    output: Path,
    x_label: str = "time (s)",
) -> None:
    """One panel: blue/red spaghetti + per-class p5-p95 band + +-1 guides."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    alpha = max(0.03, min(0.5, 30.0 / max(1, len(X_list))))
    for i in range(len(X_list)):
        color = COLOR_BLUE if traj_class[i] == BLUE_LOW_DV else COLOR_RED
        ax.plot(axis_list[i], np.asarray(X_list[i])[:, input_index], color=color, alpha=alpha, linewidth=0.5)
    for cls, color in ((BLUE_LOW_DV, COLOR_BLUE), (RED_HIGH_DV, COLOR_RED)):
        cx, cy = _class_xy(X_list, axis_list, traj_class, input_index, cls)
        c, lo, hi = binned_band(cx, cy)
        if c.size:
            ax.fill_between(c, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    ax.axhline(-1.0, color="grey", linestyle="--", linewidth=0.7)
    title = f"[{input_index}] {name}" + ("" if in_mask else "  (unused)")
    ax.set_title(title, color=("black" if in_mask else "grey"))
    ax.set_xlabel(x_label)
    ax.set_ylabel("normalized value")
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)
