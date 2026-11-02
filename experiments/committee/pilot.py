"""The #232 pilot table: every cell of committee_cells.txt and the deployed seed on
2000 confirmatory-range scenarios (`make_confirmatory_pools(42, 1, 2000)`, per_draw,
the scorer's 5 s wall clock), each cell on its own scaffolding (`evaluate_cell` routes
the cell dir's `best_params.json`). Writes pilot.json and prints the README table.

Usage (repo root): uv run python -u experiments/committee/pilot.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/python"))
sys.path.insert(0, str(REPO / "experiments/ou_marginal"))

from aerocapture.training.cell_eval import FR_DV_TOTAL, evaluate_cell  # noqa: E402
from aerocapture.training.seeds import make_confirmatory_pools  # noqa: E402
from aerocapture.training.sizing_tail import cvar  # noqa: E402
from confirmatory_marginal import read_manifest  # type: ignore[import-not-found]  # noqa: E402
from paired_vs_champion import CHAMPION, MANIFEST, PER_DRAW  # noqa: E402

OUT = Path(__file__).resolve().parent / "pilot.json"


def main() -> None:
    seeds = make_confirmatory_pools(42, 1, 2000)[0]
    cells = [CHAMPION, *read_manifest(MANIFEST)]
    rows: dict[str, dict[str, object]] = {}
    ref: tuple[np.ndarray, np.ndarray] | None = None
    for label, toml, model_dir in cells:
        t0 = time.time()
        res = evaluate_cell(REPO / (model_dir or f"training_output/{label}"), REPO / toml, seeds, extra_overrides=PER_DRAW, sim_timeout_secs=5.0)
        wall = time.time() - t0
        dv, cap = res.final_records[:, FR_DV_TOTAL], res.captured
        row: dict[str, object] = {
            "ms_per_sim_wall": round(1e3 * wall / len(seeds), 2),
            "capture_pct": round(100 * float(cap.mean()), 2),
            "mean": round(float(dv[cap].mean()), 1),
            "cvar95": round(float(cvar(dv[cap], 0.95)), 1),
            "max": round(float(dv[cap].max()), 0),
        }
        if ref is None:
            ref = (dv, cap)
        else:
            both = cap & ref[1]
            row["paired_mean_delta"] = round(float((dv[both] - ref[0][both]).mean()), 1)
            row["frac_better"] = round(float((dv[both] < ref[0][both]).mean()), 3)
        rows[label] = row
        paired = f"{row.get('paired_mean_delta', '')} {row.get('frac_better', '')}"
        print(f"| {label} | {row['ms_per_sim_wall']} | {row['capture_pct']}% | {row['mean']} | {row['cvar95']} | {row['max']:.0f} | {paired} |", flush=True)
    header = {"pool": "make_confirmatory_pools(42, 1, 2000)", "noise_seeding": "per_draw", "sim_timeout_secs": 5.0}
    OUT.write_text(json.dumps({**header, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
