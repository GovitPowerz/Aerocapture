"""Parse Fortran initial.* Monte Carlo initial conditions into DataFrames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aerocapture.io._fortran import parse_fortran_line


def parse_initial(filepath: str | Path) -> pd.DataFrame:
    """Parse an initial.* MC initial conditions file into a DataFrame.

    Each line is one simulation run: integer sim number + D-notation floats.
    Format: 1x,i5,35(1x,d17.5)

    Args:
        filepath: Path to the initial.* file.

    Returns:
        DataFrame with columns for the initial conditions data.
    """
    filepath = Path(filepath)
    rows = []
    with open(filepath) as f:
        for line in f:
            values = parse_fortran_line(line)
            if values:
                rows.append(values)

    if not rows:
        return pd.DataFrame()

    data = np.array(rows)
    ncols = data.shape[1]
    columns = [f"col_{i}" for i in range(ncols)]
    if ncols > 0:
        columns[0] = "sim_number"
    return pd.DataFrame(data, columns=columns)
