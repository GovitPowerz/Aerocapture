"""fig_output_param -- NN bank-decoder ablation.

Three full_neural bank decoders on the mission-SIZING TAIL: the default
atan2_signed head (2-output, the dense_p515 GA baseline) vs the two seam-aware
single-output variants (delta = bounded increment on the previous realized bank,
scaled_pi = wrap_to_pi(n*pi*tanh)). Bars are dv_cvar95 (primary, sizing tail) and
dv_mean (secondary). atan2 wins on both -- the wrap-seam decoders cost ~12 m/s of
CVaR95 here.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, results.json key) ordered best -> worst by CVaR95.
VARIANTS = [
    ("atan2", "optimizer_dimensionality/dense_p515_ga"),
    ("scaled_pi", "output_param/scaledpi"),
    ("delta", "output_param/delta"),
]


def main():
    fl.style()
    runs = fl.results()["runs"]

    labels = [v[0] for v in VARIANTS]
    cvar95 = np.array([runs[v[1]]["dv_cvar95"] for v in VARIANTS])
    mean = np.array([runs[v[1]]["dv_mean"] for v in VARIANTS])

    x = np.arange(len(VARIANTS))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    b1 = ax.bar(x - w / 2, cvar95, w, label="CVaR$_{95}$", color=fl.C["dense"])
    b2 = ax.bar(x + w / 2, mean, w, label="mean", color=fl.C["dense"], alpha=0.5)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    ax.axhline(cvar95[0], color=fl.C["dense"], lw=0.8, ls="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("bank decoder (full_neural head)")
    ax.set_ylabel("correction $\\Delta v$ (m/s)")
    ax.set_ylim(0, cvar95.max() * 1.18)
    ax.set_title("NN bank-decoder ablation (sizing tail)", fontsize=10, loc="left")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fl.save(fig, "fig_output_param")


if __name__ == "__main__":
    main()
