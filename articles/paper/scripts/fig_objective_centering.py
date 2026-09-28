"""fig_objective_centering -- the regime-matched objective-shaping figure.

Deployed off-nominal performance on the 9M high-dispersion stress pool, per cell:
left = capture rate (%), right = correction-DV CVaR95 (m/s, over captured runs).
The lever sweep (stacked cubed/max/n=2 -> centered linear/middle/n=16) at n = 1000
from objective_centering.json, then the three centered-Mamba trainer seeds at
n = 10 000 from centered_depth.json (issue #156), every bar under the shared noise
path (the file's "legacy" cells) so the figure is one regime. The dashed line and
its band mark the retrained joint-FTC baseline at the same depth; whiskers are
bootstrap 95% CIs where the data carry them (CVaR95 everywhere, capture for the
depth cells). Capture and the tail are shown SEPARATELY because the levers trade
them off (the middle bucket alone drops capture); CVaR95 over captures is only
comparable among cells holding ~95% capture. Reads only committed JSON; a missing
cell or reference is an error (never a thinner figure).
"""

import json
from pathlib import Path

import figlib as fl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "articles/paper/data/objective_centering.json"
DEPTH = REPO / "articles/paper/data/centered_depth.json"
DENSE = ["stacked", "plus_sims", "plus_bucket", "plus_transform", "centered"]
SEEDS = ["mamba_centered_s1", "mamba_centered_s2", "mamba_centered_s3"]
ORDER = DENSE + SEEDS
PRETTY = {
    "stacked": "stacked\ncubed/max\nn=2",
    "plus_sims": "+sims\nn=16",
    "plus_bucket": "+bucket\nmiddle",
    "plus_transform": "+transf.\nlinear",
    "centered": "centered\ndense",
    "mamba_centered_s1": "centered\nMamba\ns1",
    "mamba_centered_s2": "centered\nMamba\ns2",
    "mamba_centered_s3": "centered\nMamba\ns3",
}
# stacked = the bad control (red); single-lever cells (blue); centered (green).
COLOR = {
    "stacked": "#C44E52",
    "plus_sims": "#4C72B0",
    "plus_bucket": "#4C72B0",
    "plus_transform": "#4C72B0",
    "centered": "#55A868",
    "mamba_centered_s1": "#55A868",
    "mamba_centered_s2": "#55A868",
    "mamba_centered_s3": "#55A868",
}


def _cells():
    """label -> stats: the dense lever cells (n = 1000) and the shared-path depth cells (n = 10 000)."""
    cells = {c["label"]: c for c in json.loads(DATA.read_text())["cells"] if c["label"] in DENSE}
    cells.update({c["label"]: c for c in json.loads(DEPTH.read_text())["cells"]["legacy"]})
    missing = [k for k in ORDER + ["jointFTC-high"] if k not in cells]
    if missing:
        raise KeyError(f"cells missing from {DATA} / {DEPTH}: {missing}")
    return cells


def _whiskers(ax, i, v, ci):
    ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)


def main():
    fl.style()
    cells = _cells()
    labels = ORDER
    ref = cells["jointFTC-high"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=fl.SIZE2)
    x = range(len(labels))
    cols = [COLOR[k] for k in labels]

    caps = [cells[k]["capture_pct"] for k in labels]
    axL.bar(x, caps, color=cols)
    for i, k in enumerate(labels):
        ci = cells[k].get("capture_pct_ci")  # the n = 1000 lever cells carry no capture CI
        if ci:
            _whiskers(axL, i, caps[i], ci)
        axL.text(i, (ci[1] if ci else caps[i]) + 0.3, f"{caps[i]:.1f}", ha="center", va="bottom", fontsize=7)
    axL.axhspan(*ref["capture_pct_ci"], color="#666666", alpha=0.12, lw=0)
    axL.axhline(ref["capture_pct"], ls="--", lw=1.0, color="#666666")
    axL.text(-0.4, ref["capture_pct_ci"][1], "joint-FTC", va="bottom", ha="left", fontsize=7, color="#666666")
    axL.set_ylim(80, 100)
    axL.set_ylabel("deployed capture rate (%)")
    axL.set_title("Off-nominal capture (9M stress pool)", fontsize=10, loc="left")
    axL.set_xticks(list(x))
    axL.set_xticklabels([PRETTY[k] for k in labels], fontsize=6.5)

    cv = [cells[k]["dv_cvar95"] for k in labels]
    axR.bar(x, cv, color=cols)
    for i, k in enumerate(labels):
        _whiskers(axR, i, cv[i], cells[k]["dv_cvar95_ci"])
        axR.text(i, cells[k]["dv_cvar95_ci"][1] + 8, f"{cv[i]:.0f}", ha="center", va="bottom", fontsize=7)
    axR.axhspan(*ref["dv_cvar95_ci"], color="#666666", alpha=0.12, lw=0)
    axR.axhline(ref["dv_cvar95"], ls="--", lw=1.0, color="#666666")
    axR.text(len(labels) - 0.5, ref["dv_cvar95_ci"][1], f" joint-FTC {ref['dv_cvar95']:.0f}", va="bottom", ha="right", fontsize=7, color="#666666")
    axR.set_ylabel("deployed CVaR$_{95}$ (m/s, over captures)")
    axR.set_title("Off-nominal correction-DV tail", fontsize=10, loc="left")
    axR.set_xticks(list(x))
    axR.set_xticklabels([PRETTY[k] for k in labels], fontsize=6.5)

    fig.tight_layout()
    fl.save(fig, "fig_objective_centering")


if __name__ == "__main__":
    main()
