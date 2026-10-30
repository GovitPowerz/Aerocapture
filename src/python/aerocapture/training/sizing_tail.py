"""The one sizing-tail scorer: a deploy-side flight -> one tail record.

`cell_eval.evaluate_cell` flies a cell; this module turns its `CellResult` (or the
final records / parquet columns of one) into the record every report, script and
experiment writes: capture, the DV tail (p50 / p95 / p99 / p99.9, CVaR at 95 / 99 /
99.9 by ONE estimator, named on the record), the per-constraint violation rates of
ADR-0005 with `viol_pct` the union, the `[cost_function]` objective, and the
provenance a number needs to be quoted (noise regime, pool). Paired comparisons
on a shared seed pool live here too.

Rules (locked 2026-06-12, design 2026-10-30): capture = ifinal == 3 and
eccentricity < 1; every DV statistic is over the UNCLIPPED correction DV of the
captured runs and rounded to two decimals; CVaR is the mean of the worst
`cvar_tail_size` samples; the sample max is descriptive only; bootstrap CIs are
percentile CIs over seed resamples and cover eval-sampling noise only, not
training-run variance. Pure numpy + scipy: imports without `aerocapture_rs` and
without matplotlib.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import wilcoxon

from aerocapture.training.cell_eval import FR_DV_TOTAL, FR_INTEGRATED_FLUX, FR_MAX_G_LOAD, FR_MAX_HEAT_FLUX, is_captured

if TYPE_CHECKING:
    from aerocapture.training.cell_eval import CellResult

# The estimator every CVaR on a record was computed with; stored next to the number.
CVAR_ESTIMATOR = "mean of the worst max(1, round((1 - level) * n_captured)) captured-DV samples"
# ADR-0005 constraints: (name, final-record column, scale to the limit's unit, `cost_kwargs` key).
_CONSTRAINT_SPECS = (
    ("heat_flux", FR_MAX_HEAT_FLUX, 1.0, "heat_flux_limit"),
    ("g_load", FR_MAX_G_LOAD, 1.0, "g_load_limit"),
    ("heat_load", FR_INTEGRATED_FLUX, 1e3, "heat_load_limit"),  # MJ -> kJ
)
CONSTRAINTS = tuple(name for name, _, _, _ in _CONSTRAINT_SPECS)
# A `[flight.constraints]` override moves the limit the simulator flies at AND the limit the record scores against.
_LIMIT_OVERRIDES = {
    "flight.constraints.max_heat_flux": "heat_flux_limit",
    "flight.constraints.max_load_factor": "g_load_limit",
    "flight.constraints.max_heat_load": "heat_load_limit",
}

Stat = Callable[[npt.NDArray[np.float64]], float]


def _r2(v: float) -> float:
    return round(float(v), 2)


def cvar_tail_size(n: int, level: float) -> int:
    """How many of `n` samples the CVaR at `level` averages: at least one."""
    return max(1, int(round(n * (1.0 - level))))


def cvar(x: npt.NDArray[np.float64], level: float = 0.95) -> float:
    """Mean of the worst (1 - level) tail (expected shortfall), >= 1 sample; NaN on no sample."""
    if len(x) == 0:
        return float("nan")
    return float(np.sort(x)[-cvar_tail_size(len(x), level) :].mean())


def bootstrap_ci(
    x: npt.NDArray[np.float64],
    stat: Stat,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def capture_mask(ifinal: npt.NDArray[np.float64], ecc: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """The capture definition on parquet columns (`cell_eval.is_captured` on final records)."""
    return np.asarray((ifinal == 3) & (ecc < 1.0), dtype=np.bool_)


def _p(x: npt.NDArray[np.float64], q: float) -> float:
    # NaN on no sample, like `cvar`: np.percentile raises on an empty array (a side with no capture).
    return float(np.percentile(x, q)) if len(x) else float("nan")


def constraint_exceedances(
    final_records: npt.NDArray[np.float64],
    cost_kwargs: dict[str, Any] | None,
) -> dict[str, npt.NDArray[np.bool_]]:
    """Per-constraint exceedance masks over a final-records matrix: the one violation rule.

    Limits come from `cost_kwargs` (`heat_flux_limit` kW/m^2, `g_load_limit` g,
    `heat_load_limit` kJ/m^2 - the same keys `compute_cost` consumes, read from
    `[flight.constraints]` by `cost.build_cost_kwargs`, which defaults a missing one).
    A constraint without a configured limit is not assessed and is omitted.
    """
    kw = cost_kwargs or {}
    return {name: final_records[:, col] * scale > float(kw[key]) for name, col, scale, key in _CONSTRAINT_SPECS if isinstance(kw.get(key), (int, float))}


def dv_tail(dvc: npt.NDArray[np.float64], *, n_boot: int | None = None, seed: int = 0) -> dict[str, Any]:
    """The DV block of the record for the captured DV `dvc` (empty when there is none).

    `n_boot` adds the percentile-bootstrap CIs of the mean, p95, CVaR95, p99 and
    CVaR99 (at n = 1000 the far-tail CIs are wide: CVaR99 is the worst 10 samples).
    """
    if len(dvc) == 0:
        return {}
    out: dict[str, Any] = {
        "dv_mean": _r2(dvc.mean()),
        "dv_p50": _r2(_p(dvc, 50)),
        "dv_p95": _r2(_p(dvc, 95)),
        "dv_p99": _r2(_p(dvc, 99)),
        "dv_p999": _r2(_p(dvc, 99.9)),
        "dv_p9987": _r2(_p(dvc, 99.87)),  # 3-sigma, the propellant-sizing design case; ~= max below n ~ 5000
        "dv_cvar95": _r2(cvar(dvc, 0.95)),
        "dv_cvar99": _r2(cvar(dvc, 0.99)),
        "dv_cvar999": _r2(cvar(dvc, 0.999)),
        "dv_max_descriptive": _r2(dvc.max()),
    }
    if n_boot is not None:
        cis: dict[str, Stat] = {
            "dv_mean_ci": np.mean,
            "dv_p95_ci": lambda a: _p(a, 95),
            "dv_cvar95_ci": lambda a: cvar(a, 0.95),
            "dv_p99_ci": lambda a: _p(a, 99),
            "dv_cvar99_ci": lambda a: cvar(a, 0.99),
        }
        out.update({k: [_r2(v) for v in bootstrap_ci(dvc, stat, n_boot, seed)] for k, stat in cis.items()})
    return out


def run_stats(
    ifinal: npt.NDArray[np.float64],
    ecc: npt.NDArray[np.float64],
    dv: npt.NDArray[np.float64],
    *,
    n_boot: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """The capture + DV part of the record from parquet columns (a bundle's `final_eval.parquet`,
    a pool array); `tail_record` is the same on final records, with constraints and cost."""
    cap = capture_mask(ifinal, ecc)
    out: dict[str, Any] = {"n": int(len(ifinal)), "n_captured": int(cap.sum()), "capture_pct": _r2(100.0 * cap.mean())}
    if n_boot is not None:
        out["capture_pct_ci"] = [_r2(100.0 * v) for v in bootstrap_ci(cap.astype(np.float64), np.mean, n_boot, seed)]
    out.update(dv_tail(dv[cap], n_boot=n_boot, seed=seed))
    return out


def tail_record(
    final_records: npt.NDArray[np.float64],
    cost_kwargs: dict[str, Any],
    *,
    n_boot: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """One flight's final records -> the tail record (no regime / pool: see `score_cell`).

    `cost_kwargs` is `cost.build_cost_kwargs` of the TOML flown (`report.read_cost_kwargs`):
    its limits are the violation limits and its weights the `rms_cost` objective.
    `viol_pct` is the share of draws exceeding at least one limit; the per-constraint
    shares and the exact counts ride along (a 2-decimal rate reads fewer than 50 per
    10^6 as 0.00).
    """
    from aerocapture.training.cost import compute_cost  # noqa: PLC0415  (cost imports parquet_output -> pyarrow)

    cap = is_captured(final_records)
    n = int(len(final_records))
    over = constraint_exceedances(final_records, cost_kwargs)
    any_over = np.zeros(n, dtype=np.bool_)
    for m in over.values():
        any_over |= m
    out: dict[str, Any] = {"n": n, "n_captured": int(cap.sum()), "capture_pct": _r2(100.0 * cap.mean())}
    if n_boot is not None:
        out["capture_pct_ci"] = [_r2(100.0 * v) for v in bootstrap_ci(cap.astype(np.float64), np.mean, n_boot, seed)]
    out.update(dv_tail(final_records[cap, FR_DV_TOTAL], n_boot=n_boot, seed=seed))
    out.update({"viol_pct": _r2(100.0 * any_over.mean()), "viol_n": int(any_over.sum())})
    for name, m in over.items():
        out.update({f"{name}_viol_pct": _r2(100.0 * m.mean()), f"{name}_viol_n": int(m.sum())})
    out["rms_cost"] = _r2(compute_cost(final_records, **cost_kwargs))
    out["estimator"] = CVAR_ESTIMATOR
    return out


def flown_noise_seeding(result: CellResult) -> str:
    """The density-noise regime the flight ran under (ADR-0006): the override every sim
    shared, else the TOML's `[monte_carlo] noise_seeding` (the simulator default is per_draw)."""
    from aerocapture.training.toml_utils import load_toml_with_bases  # noqa: PLC0415

    regime = result.overrides.get("monte_carlo.noise_seeding")
    if regime is None:
        regime = load_toml_with_bases(result.toml_path).get("monte_carlo", {}).get("noise_seeding", "per_draw")
    return str(regime)


def flown_cost_kwargs(result: CellResult, cost_kwargs: dict[str, Any]) -> dict[str, Any]:
    """`cost_kwargs` with every `[flight.constraints]` limit the flight overrode (`heat_load_slope`'s
    ceiling legs) moved to the value flown: the record scores against the limits the simulator saw."""
    return {**cost_kwargs, **{limit: float(str(result.overrides[key])) for key, limit in _LIMIT_OVERRIDES.items() if key in result.overrides}}


def score_cell(result: CellResult, cost_kwargs: dict[str, Any], *, pool: str | None = None, n_boot: int | None = None, seed: int = 0) -> dict[str, Any]:
    """The record of one `evaluate_cell` / `fly_mc` flight, with the regime it flew under, the
    limits it flew at (a `[flight.constraints]` override moves the record's limit) and the caller's
    name for the seed pool (`"FINAL_EVAL 2M"`, `"fresh (offset 8M)"`, ...; None when the file the
    record goes into names the pool once for every record)."""
    rec = {**tail_record(result.final_records, flown_cost_kwargs(result, cost_kwargs), n_boot=n_boot, seed=seed), "noise_seeding": flown_noise_seeding(result)}
    if pool is not None:
        rec["pool"] = pool
    return rec


def paired_comparison(
    dv_a: npt.NDArray[np.float64],
    cap_a: npt.NDArray[np.bool_],
    dv_b: npt.NDArray[np.float64],
    cap_b: npt.NDArray[np.bool_],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired a-vs-b on the shared seed pool (row i = same MC scenario).

    delta = a - b on both-captured seeds: negative delta_mean = a better.
    """
    both = cap_a & cap_b
    d = dv_a[both] - dv_b[both]
    out: dict[str, Any] = {"n_pairs": int(both.sum())}
    if len(d) < 10:
        return out
    r3 = lambda v: round(float(v), 3)  # noqa: E731
    w = wilcoxon(d)
    out.update(
        delta_mean=r3(d.mean()),
        delta_mean_ci=[r3(v) for v in bootstrap_ci(d, np.mean, n_boot, seed)],
        win_rate_a=r3((d < 0).mean()),
        wilcoxon_p=float(f"{w.pvalue:.6g}"),  # six significant digits: the raw value differs in its last ulp across libm builds (x86_64 vs arm64)
    )
    return out


def paired_tail(
    dv_a: npt.NDArray[np.float64],
    cap_a: npt.NDArray[np.bool_],
    dv_b: npt.NDArray[np.float64],
    cap_b: npt.NDArray[np.bool_],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """`paired_comparison` plus the marginal TAIL deltas a - b (negative = a better) with
    scenario-resampled percentile-bootstrap CIs: the shared row index is resampled, each
    side's own capture mask is applied to the resample, and both marginals are recomputed
    (pairing preserved). Also the capture-rate delta in points (positive = a captures more).
    """
    xa, xb = dv_a[cap_a], dv_b[cap_b]
    rng = np.random.default_rng(seed)
    n = len(dv_a)
    d95 = np.empty(n_boot)
    dcv = np.empty(n_boot)
    dcap = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ra, rb = dv_a[idx][cap_a[idx]], dv_b[idx][cap_b[idx]]
        d95[i] = _p(ra, 95) - _p(rb, 95)
        dcv[i] = cvar(ra, 0.95) - cvar(rb, 0.95)
        dcap[i] = 100.0 * (cap_a[idx].mean() - cap_b[idx].mean())
    ci = lambda a: [_r2(_p(a, 2.5)), _r2(_p(a, 97.5))]  # noqa: E731
    return {
        **paired_comparison(dv_a, cap_a, dv_b, cap_b, n_boot, seed),
        "delta_p95": _r2(_p(xa, 95) - _p(xb, 95)),
        "delta_p95_ci": ci(d95),
        "delta_cvar95": _r2(cvar(xa, 0.95) - cvar(xb, 0.95)),
        "delta_cvar95_ci": ci(dcv),
        "delta_capture_pts": _r2(100.0 * (cap_a.mean() - cap_b.mean())),
        "delta_capture_pts_ci": ci(dcap),
    }
