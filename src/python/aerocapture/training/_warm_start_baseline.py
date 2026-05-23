"""Gen-0 validation baseline helper for warm-started chromosomes.

Kept as a small standalone module so train.py's warm-start integration is
one import + one function call, and the validation-MC plumbing can be tested
in isolation against a mocked aerocapture_rs.run_mc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aerocapture.training.parquet_output import DV_TOTAL_RAW_INDEX

try:
    import aerocapture_rs as _aero_rs
except ImportError as e:
    raise ImportError("warm_start baseline requires aerocapture_rs PyO3 module") from e


def write_gen0_baseline(
    save_dir: Path,
    toml_path: str,
    overrides: dict[str, Any],
    n_sims: int,
    dv_column_index: int = DV_TOTAL_RAW_INDEX,
    sim_timeout_secs: float | None = None,
) -> Path:
    """Run validation MC on a single warm-started overrides set and persist
    mean/RMS DV to `<save_dir>/warm_start_baseline.json`.

    Uses `run_mc`, which honors `simulation.n_sims` and runs N sims with
    internal seed dispersion. (The previous `run_batch` path silently kept only
    the first result per overrides entry, so the baseline was a single sample.)

    Args:
        save_dir: training output dir.
        toml_path: TOML config path.
        overrides: a single override dict containing the chromosome's decoded
            params plus `simulation.n_sims`.
        n_sims: number of validation sims. Injected as `simulation.n_sims` in
            the overrides dict.
        dv_column_index: column index of dv_total_m_s in final_records.
            Default 41 matches the raw 52-element RunOutput.final_record layout
            (BatchResults.final_records). Trimmed CSV indices are different.
        sim_timeout_secs: optional per-sim wall-clock timeout, forwarded to the
            Rust runner. Project convention is to thread this through every
            run_mc call site to guard against NaN-state hangs.

    Returns:
        Path to the written JSON file.
    """
    overrides_with_n = {**overrides, "simulation.n_sims": int(n_sims)}
    result = _aero_rs.run_mc(
        toml_path,
        overrides_with_n,
        include_trajectories=False,
        sim_timeout_secs=sim_timeout_secs,
    )
    dv_values = np.asarray(result.final_records[:, dv_column_index], dtype=np.float64)
    payload = {
        "n_sims": int(n_sims),
        "n_returned": int(dv_values.shape[0]),
        "mean": float(np.mean(dv_values)),
        "rms": float(np.sqrt(np.mean(dv_values**2))),
    }
    out_path = save_dir / "warm_start_baseline.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
