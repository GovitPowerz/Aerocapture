"""fig_survival -- empirical survival curves on the 10^6 per-scenario confirmatory pool.

Log-y survival 1 - F(dv) of the correction DV per finalist, from the extract's per-cell
`survival_sample` (the pooled sorted values subsampled every ~100th order statistic, ~10k points
per cell at 10 x 100 000 scenarios, per_draw, asserted at load). Finalists: the deployed
Mamba-962, the Dense-515 and LSTM-1082 cells of the headline allocation (#173, seed 1 each) and
the retuned joint-FTC and FNPAG (#172). The y-floor is the subsample's depth resolution
(0.5 / 10k): a higher floor hides the deep tails while x still autoscales to them. The curves
show where the network-to-classical separation sits at each depth; the point statistics of
the tables are their CVaR95 and CVaR99.9 integrals. Data: articles/paper/data/confirmatory_marginal.json.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

CELLS = [  # (label in confirmatory_marginal.json, display name, palette key, linestyle)
    ("ou_marginal/hl_mamba_p962", "NN Mamba-962 (deployed)", "mamba", "-"),
    ("ou_marginal/hl_dense_p515", "NN Dense-515", "dense", "-"),
    ("ou_marginal/hl_lstm_p1082", "NN LSTM-1082", "lstm", "-"),
    ("ou_marginal/classical/ftc_joint", "joint-FTC (retuned)", "jointftc", "-"),
    ("ou_marginal/classical/fnpag", "FNPAG (retuned)", "fnpag", "-"),
]


def main():
    fl.style()
    cells = fl.marginal()

    fig, ax = plt.subplots(figsize=fl.SIZE1)
    for label, name, key, ls in CELLS:
        c = cells[label]  # KeyError = cell missing from the extract: fail, never thin the figure
        x = np.asarray(c["survival_sample"], dtype=float)  # sorted pooled sample (extract_confirmatory_marginal.SURVIVAL_CELLS)
        surv = 1.0 - (np.arange(1, len(x) + 1) - 0.5) / len(x)
        ax.plot(x, surv, color=fl.C[key], ls=ls, lw=1.5, label=name)

    # the CVaR95 label anchors left of the legend box (the 5e-5 floor lifts the 0.05 line into it)
    for depth, txt, xa in ((0.05, "CVaR$_{95}$ depth", 0.70), (0.001, "CVaR$_{99.9}$ depth", 0.995)):
        ax.axhline(depth, color="#999", lw=0.7, ls=":")
        ax.annotate(txt, xy=(xa, depth), xycoords=("axes fraction", "data"), fontsize=7.5, color="#666", ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_ylim(5e-5, 1.0)  # floor = the subsample's depth resolution (~0.5/10k)
    ax.set_xlim(left=100)
    ax.set_xlabel("correction $\\Delta v$ (m/s)")
    ax.set_ylabel("survival  $1 - F(\\Delta v)$")
    ax.set_title("$10^6$ per-scenario pool ($10 \\times 100\\,000$ scenarios per scheme): survival of the correction $\\Delta v$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fl.save(fig, "fig_survival")


if __name__ == "__main__":
    main()
