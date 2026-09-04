"""Rust CLI oracle for the PyO3-vs-subprocess bit-identity test.

Runs the release binary on a training config's TOML and parses `final.*.csv` back
into the 52-column legacy array. Test-only: production evaluation goes through
`aerocapture_rs` (run_batch / run_grid); this path exists to prove those agree
with the CLI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
from aerocapture.training.config import TrainingConfig
from aerocapture.training.evaluate import _parse_final_to_legacy_array


def run_via_subprocess(config: TrainingConfig, cwd: str | Path | None = None) -> npt.NDArray[np.float64] | None:
    """Run simulation via subprocess (legacy path)."""
    if cwd is None:
        cwd = config.sim.exec_dir
    cwd = Path(cwd)

    executable = (cwd / config.sim.executable).resolve()

    if not config.sim.toml_config:
        return None

    toml_path = (cwd / config.sim.toml_config).resolve()
    try:
        subprocess.run(
            [str(executable), str(toml_path)],
            capture_output=True,
            cwd=str(cwd.resolve()),
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):  # fmt: skip
        return None

    # Parse final conditions -- auto-detect CSV vs legacy text
    final_file = cwd / config.sim.final_file
    csv_final = Path(str(final_file) + ".csv")
    if csv_final.exists():
        final_file = csv_final
    elif not final_file.exists():
        return None

    try:
        return _parse_final_to_legacy_array(final_file)
    except Exception as e:
        print(f"Warning: could not parse final file {final_file} ({type(e).__name__}: {e})", file=sys.stderr)
        return None
