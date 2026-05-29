"""Standalone NN input behavior report. See
docs/superpowers/specs/2026-05-29-nn-input-report-design.md."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# class codes
BLUE_LOW_DV = 0
RED_HIGH_DV = 1


def classify_by_dv(dv: npt.NDArray[np.float64], threshold: float) -> npt.NDArray[np.int8]:
    """Blue (0) if final DV < threshold, red (1) otherwise."""
    return np.where(np.asarray(dv) < threshold, BLUE_LOW_DV, RED_HIGH_DV).astype(np.int8)


def input_summary(
    X_list: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    names: list[str],
    in_mask: set[int],
) -> list[dict[str, object]]:
    """Per-input stats over all (trajectory x timestep) samples.

    Returns one dict per input with index, name, p1/p50/p99, frac_out_of_range
    (fraction of samples with |value| > 1), separation
    (|mean_red - mean_blue| / pooled_std), and in_mask. Sorted by
    frac_out_of_range desc, then separation desc.
    """
    n_inputs = len(names)
    blue_parts = [X_list[i] for i in range(len(X_list)) if traj_class[i] == BLUE_LOW_DV]
    red_parts = [X_list[i] for i in range(len(X_list)) if traj_class[i] == RED_HIGH_DV]
    blue = np.concatenate(blue_parts, axis=0) if blue_parts else np.empty((0, n_inputs))
    red = np.concatenate(red_parts, axis=0) if red_parts else np.empty((0, n_inputs))
    alls = np.concatenate(list(X_list), axis=0)
    rows: list[dict[str, object]] = []
    for j in range(n_inputs):
        col = alls[:, j]
        p1, p50, p99 = (float(v) for v in np.percentile(col, [1, 50, 99]))
        frac_oor = float(np.mean(np.abs(col) > 1.0))
        if blue.shape[0] and red.shape[0]:
            mb, mr = float(blue[:, j].mean()), float(red[:, j].mean())
            pooled = float(np.sqrt(0.5 * (blue[:, j].var() + red[:, j].var()))) + 1e-12
            sep = abs(mr - mb) / pooled
        else:
            sep = 0.0
        rows.append(
            {
                "index": j,
                "name": names[j],
                "p1": p1,
                "p50": p50,
                "p99": p99,
                "frac_out_of_range": frac_oor,
                "separation": sep,
                "in_mask": j in in_mask,
            }
        )
    rows.sort(key=lambda r: (r["frac_out_of_range"], r["separation"]), reverse=True)
    return rows
