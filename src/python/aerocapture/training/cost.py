"""The per-sim optimizer cost: DV (softplus-quadratic above the knee) + constraint penalties.

`compute_cost` is the single objective every optimizer, validation gate, final
selection, report and RL terminal reward score against. `dv_cost` is C-infinity
(linear below `dv_threshold`, softplus-quadratic above); constraint exceedances
are normalized softplus penalties; `cost_transform` rescales monotonically.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aerocapture.training.metrics import apply_cost_transform
from aerocapture.training.parquet_output import (
    DV_TOTAL_RAW_INDEX,
    G_LOAD_RAW_INDEX,
    HEAT_FLUX_RAW_INDEX,
    HEAT_LOAD_RAW_INDEX,
)

# Scale for the quadratic growth above threshold. Controls how fast the
# cost grows on the non-capture side.
_DV_PENALTY_SCALE = 10000.0

# Sharpness of the softplus knee at the DV threshold. Larger = sharper wall.
# k=0.01 gives ~200 m/s transition width (captures < 500 m/s are untouched).
_DV_KNEE_SHARPNESS = 0.01

# Sharpness of the softplus knee for constraint penalties. Operates on
# normalized fractions (val-limit)/limit, so k=100 means ~1% transition.
_CONSTRAINT_KNEE_SHARPNESS = 100.0


def _softplus(x: npt.NDArray[np.float64], k: float) -> npt.NDArray[np.float64]:
    """Numerically stable softplus: ln(1 + exp(k*x)) / k.

    logaddexp is exact and warning-free on both tails -- the previous
    np.where(kx > 20, x, log1p(exp(kx))/k) still evaluated exp(kx) on the
    discarded branch, spamming RuntimeWarning overflow for kx > ~709
    (constraint fraction > 7, reachable on crashes)."""
    return np.logaddexp(0.0, k * x) / k


def dv_cost(dv: npt.NDArray[np.float64], threshold: float = 1000.0) -> npt.NDArray[np.float64]:
    """C-infinity softplus-quadratic DV cost function.

    Uses softplus to smoothly transition from linear (captures) to
    quadratic penalty (non-captures). The softplus replaces the hard
    max(0, dv-T) knee with a C-infinity smooth version, while the
    quadratic term provides strong, always-increasing gradient on the
    non-capture side.

    cost(dv) = dv + sp(dv-T) + sp(dv-T)^2 / (2*S)

    where sp(x) = ln(1 + exp(k*x)) / k  (softplus with sharpness k).

    Properties:
        - C-infinity everywhere (no kinks or discontinuities)
        - Captures nearly untouched: dv=200 -> cost=200.0, dv=500 -> cost=500.7
        - Wall at threshold: slope rises from 1.0 to 1.5 across ~200 m/s
        - Strong far gradient: slope=2.9 at dv=10000, slope=3.9 at dv=20000
        - Wide non-capture spread: dv=10000 -> 23050, dv=20000 -> 57050
    """
    dv = np.maximum(dv, 1e-6)  # safety floor
    s = _DV_PENALTY_SCALE
    x = _softplus(dv - threshold, _DV_KNEE_SHARPNESS)
    return dv + x + x**2 / (2.0 * s)


def compute_cost(
    final_conditions: npt.NDArray[np.float64],
    *,
    dv_threshold: float = 1000.0,
    g_load_limit: float = 15.0,  # fallback; overridden by [flight.constraints] via cost_kwargs
    heat_flux_limit: float = 200.0,  # fallback; overridden by [flight.constraints] via cost_kwargs
    heat_load_limit: float = 25000.0,  # fallback; overridden by [flight.constraints] via cost_kwargs
    g_load_weight: float = 10000.0,
    heat_flux_weight: float = 10000.0,
    heat_load_weight: float = 10000.0,
    cost_transform: str = "linear",
) -> float:
    """Compute RMS cost from simulation final conditions.

    Uses quadratic-penalty DV cost as the primary objective with normalized
    soft constraint penalties for g-load, heat flux, and heat load exceedances.

    All termination outcomes produce meaningful DV values from Rust:
    - Captured: real orbital correction DV
    - Hyperbolic: HYPERBOLIC_BASE (10000) + excess velocity
    - Crash/PendingCrash/Timeout: virtual_dv_non_capture = CRASH_FLOOR (3000)
      + 1000 * min(|E_orb - E_target|_MJkg, 50) - 500 * t/t_max

    Returns:
        RMS cost value. Lower is better.
    """
    dv_total = final_conditions[:, DV_TOTAL_RAW_INDEX]
    g_max = final_conditions[:, G_LOAD_RAW_INDEX]
    q_max = final_conditions[:, HEAT_FLUX_RAW_INDEX]

    costs = dv_cost(dv_total, threshold=dv_threshold)

    g_penalty = g_load_weight * _softplus((g_max - g_load_limit) / g_load_limit, _CONSTRAINT_KNEE_SHARPNESS)
    q_penalty = heat_flux_weight * _softplus((q_max - heat_flux_limit) / heat_flux_limit, _CONSTRAINT_KNEE_SHARPNESS)
    heat_load = final_conditions[:, HEAT_LOAD_RAW_INDEX] * 1e3  # MJ/m2 -> kJ/m2
    hl_penalty = heat_load_weight * _softplus((heat_load - heat_load_limit) / heat_load_limit, _CONSTRAINT_KNEE_SHARPNESS)
    costs = costs + g_penalty + q_penalty + hl_penalty

    costs = apply_cost_transform(costs, cost_transform)

    return float(np.sqrt(np.mean(costs**2)))
