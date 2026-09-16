"""Shared style + data loaders for the paper figures (fig_*.py).

One central style so every figure shares the same typography (the vendored STIX
Two Text under articles/paper/fonts/), sizes, legend/grid treatment, and palette.
Figures NEVER read training_output directly -- only the committed
articles/paper/data/. `save` writes byte-reproducible SVGs (no date stamp,
salted element ids) so `make -C articles/paper figures` can be diffed.
"""

import json
from pathlib import Path

import matplotlib as mpl

# Headless Agg everywhere: with the interactive MacOSX backend the layout pass measures
# text at the Retina device-pixel ratio, so tight_layout / bbox_inches="tight" land a
# few hundredths of a point away from a Linux run and no figure is byte-identical.
mpl.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "articles/paper/data"
RUNS = DATA / "runs"
FIGDIR = REPO / "articles/paper/figures"
FONTS = REPO / "articles/paper/fonts"

# Scheme / architecture colors (stable across figures).
C = {
    "mamba": "#1f6f3f",  # headline / deployed / "good" -- green
    "dense": "#4878cf",  # efficiency ref / neutral series -- blue
    "lstm": "#6a51a3",  # co-leader -- purple
    "gru": "#8c8c8c",
    "jointftc": "#d1701f",  # best classical -- orange
    "fnpag": "#c44e52",  # accurate classical -- red
    "ftc": "#8c8c8c",
    "classical": "#c44e52",
    "baseline": "#b0b0b0",  # grey: the "before" / reference bar in before-after charts
    "accent": "#d1701f",  # orange: annotations / call-outs
}

# Consistent figure sizes (inches). Native width ~1.35x the display width keeps
# the on-page font scale uniform: full-width figures display at ~5.5in (74% of
# 7.4in native), so half-column figures must be authored at ~3.7in to display at
# ~2.7in (73%) -- the old 7.8in-native half figures rendered their fonts at half
# the size of every other figure on the page.
SIZE1 = (7.4, 4.0)  # full-width, single panel
SIZE2 = (8.4, 3.9)  # full-width, two panels side by side
SIZE_HALF = (3.7, 3.4)  # placed two-per-row in a Typst grid (displayed at ~2.7in)

# STIX Two Text is the FIGURE face (the paper body is Typst's Libertinus Serif); it is
# vendored under articles/paper/fonts/ so glyph outlines are identical on every machine.
# The fallbacks are never reached when the vendored files load (see style()).
_SERIF = ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"]
_VENDORED_FONTS = ("STIXTwoText.ttf", "STIXTwoText-Italic.ttf")


def _register_fonts() -> None:
    """Register the vendored TTFs with matplotlib. Hard-errors when one is missing:
    a silent fallback to another serif would still render, with different bytes."""
    for name in _VENDORED_FONTS:
        path = FONTS / name
        if not path.is_file():
            raise FileNotFoundError(f"vendored figure font missing: {path}")
        fm.fontManager.addfont(str(path))


def style():
    _register_fonts()
    sns.set_theme(
        style="whitegrid",
        palette="muted",
        rc={
            "axes.facecolor": "#f5f5f5",
            "figure.facecolor": "white",
        },
    )
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _SERIF,
            "mathtext.fontset": "stix",
            # Unhinted text metrics: hinted glyph extents differ between matplotlib's
            # macOS and Linux FreeType builds by ~0.01 pt, and every tight-bbox / centred
            # label coordinate inherits the shift. Same setting matplotlib's own
            # cross-platform baseline images use.
            "text.hinting": "none",
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
        }
    )


def save(fig, name: str):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{name}.svg"
    # Byte-reproducible output: no <dc:date> stamp, and element ids hashed from a
    # fixed salt instead of uuid4 (matplotlib's default when svg.hashsalt is None).
    with mpl.rc_context({"svg.hashsalt": name}):
        fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
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
