"""The per-sim optimizer cost: DV (softplus-quadratic above the knee) + constraint penalties.

`per_sim_cost` is the single objective every optimizer, validation gate, final
selection, report and RL terminal reward score against; `compute_cost` is its RMS
over a pool. `dv_cost` is C-infinity (linear below `dv_threshold`,
softplus-quadratic above); constraint exceedances are normalized softplus
penalties; `cost_transform` rescales monotonically. `CostSettings` is the one
value that carries the knobs, and its field defaults are the only default table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from aerocapture.training.metrics import apply_cost_transform
from aerocapture.training.parquet_output import (
    DV_TOTAL_RAW_INDEX,
    G_LOAD_RAW_INDEX,
    HEAT_FLUX_RAW_INDEX,
    HEAT_LOAD_RAW_INDEX,
)
from aerocapture.training.toml_utils import reject_unknown_keys

COST_FUNCTION_KEYS = frozenset({"dv_threshold", "g_load_weight", "heat_flux_weight", "heat_load_weight", "cost_transform"})
# `[flight.constraints]` key -> the `CostSettings` limit it sets.
_CONSTRAINT_LIMITS = {"max_load_factor": "g_load_limit", "max_heat_flux": "heat_flux_limit", "max_heat_load": "heat_load_limit"}


@dataclass(frozen=True)
class CostSettings:
    """The cost function's knobs: `[cost_function]` plus the `[flight.constraints]` limits
    (g, kW/m^2, kJ/m^2). The defaults below are what a config that omits a key scores at."""

    dv_threshold: float = 1000.0
    g_load_limit: float = 15.0
    heat_flux_limit: float = 200.0
    heat_load_limit: float = 25000.0
    g_load_weight: float = 1000.0
    heat_flux_weight: float = 1000.0
    heat_load_weight: float = 1000.0
    cost_transform: str = "linear"

    @classmethod
    def from_toml(cls, toml_data: dict[str, Any]) -> CostSettings:
        """The one `[cost_function]` + `[flight.constraints]` reader (training, gate, report, compare).

        Rejects unknown `[cost_function]` keys: a misspelled weight used to train
        and report at the default with no message.
        """
        cost_cfg = toml_data.get("cost_function", {})
        reject_unknown_keys("cost_function", cost_cfg, COST_FUNCTION_KEYS)
        constraints = toml_data.get("flight", {}).get("constraints", {})
        values: dict[str, Any] = {key: float(cost_cfg[key]) for key in COST_FUNCTION_KEYS - {"cost_transform"} if key in cost_cfg}
        values.update({field: float(constraints[key]) for key, field in _CONSTRAINT_LIMITS.items() if key in constraints})
        if "cost_transform" in cost_cfg:
            values["cost_transform"] = str(cost_cfg["cost_transform"])
        return cls(**values)


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


def per_sim_cost(final_records: npt.NDArray[np.float64], settings: CostSettings) -> npt.NDArray[np.float64]:
    """Per-simulation cost of final records of any leading shape: `(..., FINAL_RECORD_LEN) -> (...)`.

    Quadratic-penalty DV cost plus normalized soft constraint penalties for g-load,
    heat flux and heat load exceedances, then `cost_transform`. Every operation is an
    elementwise array loop, so a cell's cost does not depend on the shape it is evaluated
    in. A single record is lifted to one row first: numpy scalar math takes `pow()` for
    `x**2` where the array loop multiplies, a last-ulp difference.

    All termination outcomes produce meaningful DV values from Rust:
    - Captured: real orbital correction DV
    - Hyperbolic: HYPERBOLIC_BASE (10000) + excess velocity
    - Crash/PendingCrash/Timeout: virtual_dv_non_capture = CRASH_FLOOR (3000)
      + 1000 * min(|E_orb - E_target|_MJkg, 50) - 500 * t/t_max
    """
    records = np.atleast_2d(final_records)
    dv_total = records[..., DV_TOTAL_RAW_INDEX]
    g_max = records[..., G_LOAD_RAW_INDEX]
    q_max = records[..., HEAT_FLUX_RAW_INDEX]

    costs = dv_cost(dv_total, threshold=settings.dv_threshold)

    g_penalty = settings.g_load_weight * _softplus((g_max - settings.g_load_limit) / settings.g_load_limit, _CONSTRAINT_KNEE_SHARPNESS)
    q_penalty = settings.heat_flux_weight * _softplus((q_max - settings.heat_flux_limit) / settings.heat_flux_limit, _CONSTRAINT_KNEE_SHARPNESS)
    heat_load = records[..., HEAT_LOAD_RAW_INDEX] * 1e3  # MJ/m2 -> kJ/m2
    hl_penalty = settings.heat_load_weight * _softplus((heat_load - settings.heat_load_limit) / settings.heat_load_limit, _CONSTRAINT_KNEE_SHARPNESS)
    costs = costs + g_penalty + q_penalty + hl_penalty

    return apply_cost_transform(costs, settings.cost_transform).reshape(final_records.shape[:-1])


def compute_cost(final_records: npt.NDArray[np.float64], settings: CostSettings) -> float:
    """RMS of `per_sim_cost` over every simulation in `final_records`. Lower is better."""
    costs = per_sim_cost(final_records, settings)
    return float(np.sqrt(np.mean(costs**2)))
