"""Gen-0 validation baseline helper for warm-started chromosomes.

Persists mean / RMS / capture-rate of the bare warm-started chromosome on the
RESERVED VALIDATION seed pool, so the metrics are directly comparable to the
`Gen N validation: rms=...` line emitted by train.py's validation gate. The
caller is responsible for running the per-seed evaluation through
`problem.evaluate_individual_per_seed(warm_chromo, val_seeds)`; this module
just consumes the resulting cost array and writes the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt


def write_gen0_baseline(
    save_dir: Path,
    costs: npt.NDArray[np.float64],
    capture_rate: float,
    n_sims: int,
) -> Path:
    """Write `<save_dir>/warm_start_baseline.json` from precomputed per-seed costs.

    The cost array is the output of `problem.evaluate_individual_per_seed`,
    which already routes captures through the real DV and non-captures through
    the configured virtual-DV formula. RMS/mean/p95 use the same `compute_cost`
    convention as the rest of training, so the baseline is comparable to the
    `Best val` RMS the validation gate later reports.

    Args:
        save_dir: training output dir.
        costs: 1D array of per-seed costs, shape (n_sims,).
        capture_rate: fraction of sims that captured (real DV vs virtual DV).
        n_sims: number of validation sims actually run (== len(costs)).

    Returns:
        Path to the written JSON file.
    """
    if costs.ndim != 1 or costs.shape[0] != n_sims:
        raise ValueError(f"costs shape {costs.shape} does not match n_sims={n_sims}")
    payload = {
        "n_sims": int(n_sims),
        "n_returned": int(costs.shape[0]),
        "capture_rate": float(capture_rate),
        "rms_cost": float(np.sqrt(np.mean(costs**2))),
        "mean_cost": float(np.mean(costs)),
        "median_cost": float(np.median(costs)),
        "p95_cost": float(np.percentile(costs, 95)),
        "worst_cost": float(np.max(costs)),
    }
    out_path = save_dir / "warm_start_baseline.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
