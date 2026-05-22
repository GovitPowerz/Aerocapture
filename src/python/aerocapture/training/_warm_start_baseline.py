"""Gen-0 validation baseline helper for warm-started chromosomes.

Kept as a small standalone module so train.py's warm-start integration is
one import + one function call, and the validation-MC plumbing can be tested
in isolation against a mocked aerocapture_rs.run_batch.
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
    overrides: list[dict[str, Any]],
    n_sims: int,
    dv_column_index: int = DV_TOTAL_RAW_INDEX,
) -> Path:
    """Run validation MC on a single warm-started overrides set and persist
    mean/RMS DV to `<save_dir>/warm_start_baseline.json`.

    Args:
        save_dir: training output dir.
        toml_path: TOML config path.
        overrides: a single-element list of override dicts; the chromosome's
            decoded params should already be in the dict.
        n_sims: number of validation sims.
        dv_column_index: column index of dv_total_m_s in final_records.
            Default 41 matches the raw 52-element RunOutput.final_record layout
            (BatchResults.final_records). Trimmed CSV indices are different.

    Returns:
        Path to the written JSON file.
    """
    overrides_with_n = [{**overrides[0], "simulation.n_sims": int(n_sims)}]
    result = _aero_rs.run_batch(
        toml_path,
        overrides_with_n,
        include_trajectories=False,
    )
    dv_values = np.asarray(result.final_records[:, dv_column_index], dtype=np.float64)
    payload = {
        "n_sims": int(n_sims),
        "mean": float(np.mean(dv_values)),
        "rms": float(np.sqrt(np.mean(dv_values**2))),
    }
    out_path = save_dir / "warm_start_baseline.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
