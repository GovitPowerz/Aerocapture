"""fig_objective_centering -- the regime-matched methodology figure.

Left: validation capture-rate vs cumulative training sims (transform-independent),
one line per cell -- the gradient-recovery evidence (stacked flat ~96%, centered
climbing). Right: deployed off-nominal CVaR95 on the 9M stress pool, per cell.
Reads only articles/paper/data/objective_centering.json; degrades if cells missing.
"""

import json
from pathlib import Path

import figlib as fl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "articles/paper/data/objective_centering.json"
ORDER = ["stacked", "plus_sims", "plus_bucket", "plus_transform", "centered"]


def main():
    fl.style()
    d = json.loads(DATA.read_text())
    conv = d.get("convergence", {})
    cells = {c["label"]: c for c in d.get("cells", [])}
    present = [k for k in ORDER if k in conv or k in cells]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for i, label in enumerate(present):
        series = conv.get(label, [])
        if series:
            xs = [p[0] / 1e6 for p in series]
            ys = [100.0 * p[1] for p in series]
            axL.plot(xs, ys, label=label, linewidth=1.4)
    axL.set_xlabel("cumulative training sims (millions)")
    axL.set_ylabel("validation capture rate (%)")
    axL.set_title("Gradient recovery under the high regime", fontsize=10, loc="left")
    axL.legend(fontsize=7, loc="lower right")

    labels = [k for k in ORDER if k in cells]
    vals = [cells[k].get("dv_cvar95") for k in labels]
    axR.bar(range(len(labels)), vals, color="#4C72B0")
    axR.set_xticks(range(len(labels)))
    axR.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axR.set_ylabel("deployed CVaR$_{95}$ on 9M pool (m/s)")
    axR.set_title("Off-nominal deployment", fontsize=10, loc="left")

    fig.tight_layout()
    fl.save(fig, "fig_objective_centering")


if __name__ == "__main__":
    main()
