"""Shared style + data loaders for the paper figures (fig_*.py).

One central style so every figure shares the same typography (STIX Two Text,
matching the paper body), sizes, legend/grid treatment, and palette. Figures
NEVER read training_output directly -- only the committed articles/paper/data/.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "articles/paper/data"
RUNS = DATA / "runs"
FIGDIR = REPO / "articles/paper/figures"

# Scheme / architecture colors (stable across figures).
C = {
    "mamba": "#1f6f3f",   # headline / deployed / "good" -- green
    "dense": "#4878cf",   # efficiency ref / neutral series -- blue
    "lstm": "#6a51a3",    # co-leader -- purple
    "gru": "#8c8c8c",
    "jointftc": "#d1701f",  # best classical -- orange
    "fnpag": "#c44e52",     # accurate classical -- red
    "ftc": "#8c8c8c",
    "classical": "#c44e52",
    "baseline": "#b0b0b0",  # grey: the "before" / reference bar in before-after charts
    "accent": "#d1701f",    # orange: annotations / call-outs
}

# Consistent figure sizes (inches). Uniform native widths keep on-page font
# sizes consistent after Typst scales each figure to its display width.
SIZE1 = (7.4, 4.0)      # full-width, single panel
SIZE2 = (8.4, 3.9)      # full-width, two panels side by side
SIZE_HALF = (7.8, 3.9)  # placed two-per-row in a Typst grid (displayed at ~50%)

# STIX Two Text is the paper body face; STIXGeneral / Times are metric-compatible fallbacks.
_SERIF = ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"]


def style():
    sns.set_theme(style="whitegrid", palette="muted", rc={
        "axes.facecolor": "#f5f5f5",
        "figure.facecolor": "white",
    })
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF,
        "mathtext.fontset": "stix",
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.facecolor": "white",
        "axes.edgecolor": "#bbbbbb",
        "axes.linewidth": 0.8,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.7,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


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
