"""Statistical utilities for aerocapture analysis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def empirical_cdf(x: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute empirical CDF (step function) from samples.

    Equivalent to MATLAB cdfgov.m: produces coordinates for a step-function CDF.

    Args:
        x: 1D array of samples.

    Returns:
        (xcdf, ycdf): Arrays suitable for plotting with plt.step() or plt.plot().
        xcdf has -inf/+inf padding, ycdf ranges from 0 to 1.
    """
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = len(x)
    if n == 0:
        return np.array([-np.inf, np.inf]), np.array([0.0, 0.0])

    y = np.arange(1, n + 1, dtype=np.float64) / n

    # Remove duplicate x values (keep last occurrence)
    not_dup = np.concatenate([np.diff(x) > 0, [True]])
    x = x[not_dup]
    y = y[not_dup]

    # Create step function coordinates
    k = len(x)
    idx = np.repeat(np.arange(k), 2)
    xcdf = np.concatenate([[-np.inf], x[idx], [np.inf]])
    ycdf = np.concatenate([[0.0, 0.0], y[idx]])
    return xcdf, ycdf
