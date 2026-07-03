"""fig_objective_centering -- the regime-matched objective-shaping figure.

Deployed off-nominal performance on the 9M high-dispersion stress pool, per cell:
left = capture rate (%), right = correction-DV CVaR95 (m/s, over captured runs).
The lever sweep (stacked cubed/max/n=2 -> centered linear/middle/n=16) plus the
Mamba_962 confirmation. The dashed line marks the best retrained classical
(joint-FTC). Capture and the tail are shown SEPARATELY because the levers trade
them off (the middle bucket alone drops capture); CVaR95 over captures is only
comparable among cells holding ~95% capture. Reads only committed JSON; degrades
if cells are missing.
"""

import json
from pathlib import Path

import figlib as fl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "articles/paper/data/objective_centering.json"
ROBUST = REPO / "articles/paper/data/robustness_retrain.json"
ORDER = ["stacked", "plus_sims", "plus_bucket", "plus_transform", "centered", "mamba_centered"]
PRETTY = {
    "stacked": "stacked\n(cubed/max/n=2)",
    "plus_sims": "+sims\n(n=16)",
    "plus_bucket": "+bucket\n(middle)",
    "plus_transform": "+transform\n(linear)",
    "centered": "centered\n(all, dense)",
    "mamba_centered": "centered\n(Mamba)",
}
# stacked = the bad control (red); single-lever cells (blue); centered (green).
COLOR = {
    "stacked": "#C44E52", "plus_sims": "#4C72B0", "plus_bucket": "#4C72B0",
    "plus_transform": "#4C72B0", "centered": "#55A868", "mamba_centered": "#55A868",
}


def _joint_ftc_high():
    """Best retrained classical reference (capture %, CVaR95) from the robustness eval."""
    if not ROBUST.exists():
        return None
    for s in json.loads(ROBUST.read_text()).get("schemes", []):
        if s.get("label") == "jointFTC-high":
            return s.get("capture_pct"), s.get("dv_cvar95")
    return None


def main():
    fl.style()
    cells = {c["label"]: c for c in json.loads(DATA.read_text()).get("cells", [])}
    labels = [k for k in ORDER if k in cells]
    ref = _joint_ftc_high()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=fl.SIZE2)
    x = range(len(labels))
    cols = [COLOR[k] for k in labels]

    caps = [cells[k].get("capture_pct") for k in labels]
    axL.bar(x, caps, color=cols)
    for i, v in enumerate(caps):
        axL.text(i, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    if ref:
        axL.axhline(ref[0], ls="--", lw=1.0, color="#666666")
        axL.text(len(labels) - 0.5, ref[0], " joint-FTC", va="bottom", ha="right", fontsize=7, color="#666666")
    axL.set_ylim(80, 100)
    axL.set_ylabel("deployed capture rate (%)")
    axL.set_title("Off-nominal capture (9M stress pool)", fontsize=10, loc="left")
    axL.set_xticks(list(x))
    axL.set_xticklabels([PRETTY[k] for k in labels], fontsize=7)

    cv = [cells[k].get("dv_cvar95") for k in labels]
    axR.bar(x, cv, color=cols)
    for i, v in enumerate(cv):
        if v is not None:
            axR.text(i, v + 8, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    if ref and ref[1] is not None:
        axR.axhline(ref[1], ls="--", lw=1.0, color="#666666")
        axR.text(len(labels) - 0.5, ref[1], f" joint-FTC {ref[1]:.0f}", va="bottom", ha="right", fontsize=7, color="#666666")
    axR.set_ylabel("deployed CVaR$_{95}$ (m/s, over captures)")
    axR.set_title("Off-nominal correction-DV tail", fontsize=10, loc="left")
    axR.set_xticks(list(x))
    axR.set_xticklabels([PRETTY[k] for k in labels], fontsize=7)

    fig.tight_layout()
    fl.save(fig, "fig_objective_centering")


if __name__ == "__main__":
    main()
