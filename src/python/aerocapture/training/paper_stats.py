"""Pure statistical helpers for the paper (locked 2026-06-12 reporting rules).

Capture = ifinal==3 & eccentricity<1.0. Tail metrics are p99 + CVaR95 (mean of
the worst 5% of captured DV); the sample max is descriptive only. Cross-run
comparisons are PAIRED on the shared final-eval pool (same seeds, same row
order); failures in EITHER run drop the pair. Bootstrap CIs are percentile
CIs over seed resamples and cover eval-sampling noise only, NOT training-run
variance (sigma_run from the seed-repeats study covers that).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import wilcoxon


def _r2(v: float) -> float:
    return round(float(v), 2)


def cvar(x: npt.NDArray[np.float64], level: float = 0.95) -> float:
    """Mean of the worst (1-level) tail (expected shortfall), >= 1 sample."""
    k = max(1, int(round(len(x) * (1.0 - level))))
    return float(np.sort(x)[-k:].mean())


def bootstrap_ci(
    x: npt.NDArray[np.float64],
    stat: Callable[[npt.NDArray[np.float64]], float],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def capture_mask(ifinal: npt.NDArray[np.float64], ecc: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    return np.asarray((ifinal == 3) & (ecc < 1.0), dtype=np.bool_)


def run_stats(
    ifinal: npt.NDArray[np.float64],
    ecc: npt.NDArray[np.float64],
    dv: npt.NDArray[np.float64],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    cap = capture_mask(ifinal, ecc)
    dvc = dv[cap]
    out: dict[str, Any] = {"n": int(len(ifinal)), "capture_pct": _r2(100 * cap.mean())}
    if len(dvc) == 0:
        return out
    out.update(
        dv_mean=_r2(dvc.mean()),
        dv_p50=_r2(np.percentile(dvc, 50)),
        dv_p95=_r2(np.percentile(dvc, 95)),
        dv_p99=_r2(np.percentile(dvc, 99)),
        dv_p999=_r2(np.percentile(dvc, 99.9)),
        dv_cvar95=_r2(cvar(dvc, 0.95)),
        dv_cvar99=_r2(cvar(dvc, 0.99)),
        dv_max_descriptive=_r2(dvc.max()),
        dv_mean_ci=[_r2(v) for v in bootstrap_ci(dvc, np.mean, n_boot, seed)],
        dv_p95_ci=[_r2(v) for v in bootstrap_ci(dvc, lambda a: float(np.percentile(a, 95)), n_boot, seed)],
        dv_cvar95_ci=[_r2(v) for v in bootstrap_ci(dvc, lambda a: cvar(a, 0.95), n_boot, seed)],
        # Far-tail (mission-sizing) metrics with CIs. At n=1000 these are wide
        # (CVaR99 ~ worst 10 samples, p99.9 ~ 1) -- reliable only on a large pool.
        dv_p99_ci=[_r2(v) for v in bootstrap_ci(dvc, lambda a: float(np.percentile(a, 99)), n_boot, seed)],
        dv_cvar99_ci=[_r2(v) for v in bootstrap_ci(dvc, lambda a: cvar(a, 0.99), n_boot, seed)],
    )
    return out


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
        wilcoxon_p=float(w.pvalue),
    )
    return out


def actual_sims(
    records: list[dict],
    training_n_sims: int,
    validation_n_sims: int = 1000,
    curation_sample_size: int = 1000,
    curation_top_k: int = 1,
) -> dict[str, int]:
    """Reconstruct the run's ACTUAL simulation count from its JSONL records.

    training   = n_gen x n_pop x training_n_sims (selection-driving evals)
    validation = validation-gate fires x validation_n_sims
    curation   = distinct curation events x top_k x sample_size (probe sims)
    reeval     = parent re-evals on seed-change gens (n_pop x n_sims each,
                 approximated by the curation-event count; rotating-strategy
                 runs re-draw every gen, so their re-evals are already part of
                 the per-gen training term)
    """
    n_gen = len(records)
    n_pop = max((len(r.get("all_costs", [])) for r in records), default=0)
    validations = sum(1 for r in records if r.get("validation"))
    curation_gens = {r["pool_metrics"]["last_curation_gen"] for r in records if r.get("pool_metrics", {}).get("last_curation_gen") is not None}
    n_curations = len(curation_gens)
    out = {
        "n_gen": n_gen,
        "n_pop": n_pop,
        "training": n_gen * n_pop * training_n_sims,
        "validation": validations * validation_n_sims,
        "curation": n_curations * curation_top_k * curation_sample_size,
        "reeval": n_curations * n_pop * training_n_sims,
    }
    out["total"] = out["training"] + out["validation"] + out["curation"] + out["reeval"]
    return out
