"""Shared style + data loaders for the paper figures (fig_*.py).

Consistent seaborn theme (matches src/python/aerocapture/training/charts.py), SVG
output to articles/paper/figures/, and thin loaders for the committed data
products (results.json, far_tail_eval.json, robustness_stress.json,
compute_benchmark.json + the bundle parquets). Figures NEVER read training_output
directly -- only the committed articles/paper/data/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "articles/paper/data"
RUNS = DATA / "runs"
FIGDIR = REPO / "articles/paper/figures"

# Scheme / architecture colors (stable across figures).
C = {
    "mamba": "#1f6f3f",   # headline -- green
    "dense": "#4878cf",   # efficiency ref -- blue
    "lstm": "#6a51a3",    # co-leader -- purple
    "gru": "#999999",
    "jointftc": "#d1701f",  # best classical -- orange
    "fnpag": "#c44e52",     # accurate classical -- red
    "ftc": "#8c8c8c",
    "classical": "#c44e52",
}


def style():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95, rc={"axes.facecolor": "#f5f5f5"})


def save(fig, name: str):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{name}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")
    return out


def results() -> dict:
    return json.loads((DATA / "results.json").read_text())


def far_tail() -> dict:
    """label -> cell dict (cvar99, cvar999, p999, max, p99, ...)."""
    cells = json.loads((DATA / "far_tail_eval.json").read_text())["cells"]
    return {c["label"]: c for c in cells}


def robustness() -> list:
    return json.loads((DATA / "robustness_stress.json").read_text())["schemes"]


def compute() -> list:
    return json.loads((DATA / "compute_benchmark.json").read_text())["schemes"]
