"""fig_survival -- empirical survival curves on the confirmatory sizing pool.

Log-y survival 1 - F(dv) per finalist from the confirmatory pool's committed
per-cell `survival_sample` (every ~100th pooled sorted value, ~10k points per
cell at 10 x 100k). Reviewer R1 major 2: report the tail as a curve, not only
point statistics -- the network-vs-classical gap growing with depth is the
paper's thesis made visible. Data: articles/paper/data/confirmatory_eval.json.
"""

import json

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

CELLS = [  # (label in confirmatory_eval.json, display name, palette key, linestyle)
    ("mamba_p962_long", "NN Mamba (deployed)", "mamba", "-"),
    ("dense_p515_ga_paper_best", "NN dense (efficiency ref.)", "dense", "-"),
    ("lstm_p1082_long", "NN LSTM", "lstm", "-"),
    ("joint_reference/ftc", "FTC (joint reference)", "jointftc", "-"),
    ("fnpag", "FNPAG", "fnpag", "-"),
]


def main():
    fl.style()
    d = json.loads((fl.DATA / "confirmatory_eval.json").read_text())
    cells = {c["label"]: c for c in d["cells"]}

    fig, ax = plt.subplots(figsize=fl.SIZE1)
    for label, name, key, ls in CELLS:
        c = cells.get(label)
        if c is None:
            print(f"  (skip {label}: not in confirmatory_eval.json yet)")
            continue
        x = np.asarray(c["survival_sample"], dtype=float)  # sorted pooled sample
        surv = 1.0 - (np.arange(1, len(x) + 1) - 0.5) / len(x)
        ax.plot(x, surv, color=fl.C[key], ls=ls, lw=1.5, label=name)

    # the CVaR95 label anchors left of the legend box (the 5e-5 floor lifts the 0.05 line into it)
    for depth, txt, xa in ((0.05, "CVaR$_{95}$ depth", 0.70), (0.001, "CVaR$_{99.9}$ depth", 0.995)):
        ax.axhline(depth, color="#999", lw=0.7, ls=":")
        ax.annotate(txt, xy=(xa, depth), xycoords=("axes fraction", "data"),
                    fontsize=7.5, color="#666", ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_ylim(5e-5, 1.0)  # floor = the subsample's depth resolution (~0.5/10k); a higher floor
    # hides the deep tails while x still autoscales to them, leaving the right half of the plot empty
    ax.set_xlim(left=100)
    ax.set_xlabel("correction $\\Delta v$ (m/s)")
    ax.set_ylabel("survival  $1 - F(\\Delta v)$")
    ax.set_title("Confirmatory pool ($10 \\times 100\\,000$ scenarios): the gap grows with tail depth")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fl.save(fig, "fig_survival")


if __name__ == "__main__":
    main()
