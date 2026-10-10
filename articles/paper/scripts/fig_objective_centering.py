"""fig_objective_centering -- objective centering under per-scenario noise (issues #177, #180).

Deployed off-nominal performance on the 9M high-dispersion stress pool, every cell trained and
flown under per-scenario density noise (both files asserted per_draw). Left: capture rate (%).
Middle: correction-DV CVaR95 (m/s, over captured runs). Right: constraint exceedance (% of
draws over the heat-flux and heat-load limits; g-load tracks heat flux and is left to the
table). The lever sweep (stacked cubed/max/n=2 -> centered linear/middle/n=16) at n = 1000 from
objective_centering_v4.json, then the three centered-Mamba trainer seeds and the two retuned
joint-FTC baselines (medium and high regime, grey) at n = 10 000 from centered_depth_v4.json.
Whiskers are bootstrap 95% CIs where the data carry them (CVaR95 everywhere, capture for the
depth cells). The three panels are read together: the levers trade capture against the tail
(the middle bucket alone drops capture), and a tail win can be bought with heat-flux
exceedance (seed 1 against seed 3), so CVaR95 is only compared among cells holding capture
and exceedance. Reads only committed JSON; a missing cell is an error (never a thinner figure).
"""

import json
from pathlib import Path

import figlib as fl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "articles/paper/data/objective_centering_v4.json"
DEPTH = REPO / "articles/paper/data/centered_depth_v4.json"
DENSE = ["stacked", "plus_sims", "plus_bucket", "plus_transform", "centered"]
SEEDS = ["mamba_centered_s1", "mamba_centered_s2", "mamba_centered_s3"]
BASELINES = ["jointFTC-medium", "jointFTC-high"]
ORDER = DENSE + SEEDS + BASELINES
PRETTY = {
    "stacked": "stacked\ncubed/max\nn=2",
    "plus_sims": "+sims\nn=16",
    "plus_bucket": "+bucket\nmiddle",
    "plus_transform": "+transf.\nlinear",
    "centered": "centered\ndense",
    "mamba_centered_s1": "centered\nMamba s1",
    "mamba_centered_s2": "centered\nMamba s2",
    "mamba_centered_s3": "centered\nMamba s3",
    "jointFTC-medium": "joint-FTC\nmedium",
    "jointFTC-high": "joint-FTC\nhigh",
}
# stacked = the bad control (red); single-lever cells (blue); centered (green); baselines (grey).
COLOR = {
    "stacked": "#C44E52",
    "plus_sims": "#4C72B0",
    "plus_bucket": "#4C72B0",
    "plus_transform": "#4C72B0",
    "centered": "#55A868",
    "mamba_centered_s1": "#55A868",
    "mamba_centered_s2": "#55A868",
    "mamba_centered_s3": "#55A868",
    "jointFTC-medium": "#8c8c8c",
    "jointFTC-high": "#bdbdbd",
}
LABEL_SIZE = 7.5
VALUE_SIZE = 7.0


def _cells():
    """label -> stats: the dense lever cells (n = 1000) and the depth cells (n = 10 000), per_draw only."""
    lever = json.loads(DATA.read_text())
    depth = json.loads(DEPTH.read_text())
    assert lever["noise_seeding"] == "per_draw", lever["noise_seeding"]
    assert list(depth["regimes"]) == ["per_draw"] and list(depth["cells"]) == ["per_draw"], (depth["regimes"], list(depth["cells"]))
    cells = {c["label"]: c for c in lever["cells"] if c["label"] in DENSE}
    cells.update({c["label"]: c for c in depth["cells"]["per_draw"]})
    missing = [k for k in ORDER if k not in cells]
    if missing:
        raise KeyError(f"cells missing from {DATA} / {DEPTH}: {missing}")
    return cells


def _whiskers(ax, i, v, ci):
    ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)


def _separators(ax):
    ax.axvline(len(DENSE) - 0.5, color="#999999", lw=0.6, ls=":")
    ax.axvline(len(DENSE) + len(SEEDS) - 0.5, color="#999999", lw=0.6, ls=":")


def main():
    fl.style()
    cells = _cells()
    x = range(len(ORDER))
    cols = [COLOR[k] for k in ORDER]

    fig, (axL, axM, axR) = plt.subplots(3, 1, figsize=(fl.SIZE1[0], 7.2), sharex=True)

    caps = [cells[k]["capture_pct"] for k in ORDER]
    axL.bar(x, caps, color=cols)
    tops = []
    for i, k in enumerate(ORDER):
        ci = cells[k].get("capture_pct_ci")  # the n = 1000 lever cells carry no capture CI
        if ci:
            _whiskers(axL, i, caps[i], ci)
        top = ci[1] if ci else caps[i]
        tops.append(top)
        axL.text(i, top + 0.15, f"{caps[i]:.1f}", ha="center", va="bottom", fontsize=VALUE_SIZE)
    lo = min(c.get("capture_pct_ci", [c["capture_pct"]])[0] for c in cells.values())
    axL.set_ylim(int(lo) - 1, max(tops) + 1.6)
    axL.set_ylabel("deployed capture rate (%)")
    n_depth = f"{cells['jointFTC-medium']['n']:,}".replace(",", " ")
    axL.set_title(f"Off-nominal capture (9M stress pool, per-scenario noise; lever cells n = 1000, depth cells n = {n_depth})", loc="left")

    cv = [cells[k]["dv_cvar95"] for k in ORDER]
    axM.bar(x, cv, color=cols)
    for i, k in enumerate(ORDER):
        _whiskers(axM, i, cv[i], cells[k]["dv_cvar95_ci"])
        axM.text(i, cells[k]["dv_cvar95_ci"][1] + 12, f"{cv[i]:.0f}", ha="center", va="bottom", fontsize=VALUE_SIZE)
    axM.set_ylim(0, max(c["dv_cvar95_ci"][1] for c in cells.values()) * 1.1)
    axM.set_ylabel("deployed CVaR$_{95}$ (m/s, over captures)")
    axM.set_title("Off-nominal correction-DV tail", loc="left")

    w = 0.38
    hf = [cells[k]["violation_pct"]["heat_flux"] for k in ORDER]
    hl = [cells[k]["violation_pct"]["heat_load"] for k in ORDER]
    axR.bar([i - w / 2 for i in x], hf, width=w, color=cols)
    axR.bar([i + w / 2 for i in x], hl, width=w, color=cols, hatch="////", edgecolor="white", linewidth=0)
    for i in x:
        axR.text(i - w / 2, hf[i] + 0.4, f"{hf[i]:.1f}", ha="center", va="bottom", fontsize=VALUE_SIZE)
        axR.text(i + w / 2, hl[i] + 0.4, f"{hl[i]:.1f}", ha="center", va="bottom", fontsize=VALUE_SIZE)
    axR.set_ylim(0, max(hf) * 1.15)
    axR.set_ylabel("draws over the limit (%)")
    axR.set_title("Constraint exceedance", loc="left")
    axR.legend(
        handles=[Patch(facecolor="#555555", label="heat flux"), Patch(facecolor="#555555", hatch="////", edgecolor="white", label="heat load")],
        loc="upper right",
    )
    axR.set_xticks(list(x))
    axR.set_xticklabels([PRETTY[k] for k in ORDER], fontsize=LABEL_SIZE)

    for ax in (axL, axM, axR):
        _separators(ax)
    fig.tight_layout()
    fl.save(fig, "fig_objective_centering")


if __name__ == "__main__":
    main()
