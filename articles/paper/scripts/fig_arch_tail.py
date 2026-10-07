"""fig_arch_tail -- the headline figure: the sizing tail per seed at the headline allocation.

Per-seed CVaR95 (left) and CVaR99.9 (right) of the correction DV for every v4 network cell
trained from scratch at the headline allocation (GA 512 x 2, 20 000 generations, per-scenario
density noise: the #173 `hl_*` runs), scored on the 10^6 per-scenario confirmatory pool
(10 x 100 000 scenarios per cell, per_draw, asserted at load). Mamba-962 and Dense-515 carry
three seeds (bar = seed mean), LSTM-1082, GRU-1014 and Dense-972 one. Whiskers on the right
panel are +-1.96 replicate standard errors of CVaR99.9, drawn where they exceed the marker.
The deployed cell is seed 1 of Mamba-962 (starred), fixed by the pre-registered rule before any
10^6 score existed; nothing here was selected on this pool. The retuned classical references
(joint-FTC and FNPAG, #172) are dashed lines. The sample maximum is not plotted: at 10^6
scenarios it is one extreme draw. Data: articles/paper/data/confirmatory_marginal.json.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, color key, confirmatory labels: seed 1 first). Three-seed families first, then the single seeds.
ARCHS = [
    ("Mamba-962", "mamba", ["ou_marginal/hl_mamba_p962", "ou_marginal/hl_mamba_p962_s2", "ou_marginal/hl_mamba_p962_s3"]),
    ("Dense-515", "dense", ["ou_marginal/hl_dense_p515", "ou_marginal/hl_dense_p515_s2", "ou_marginal/hl_dense_p515_s3"]),
    ("GRU-1014", "gru", ["ou_marginal/hl_gru_p1014"]),
    ("Dense-972", "dense", ["ou_marginal/hl_dense_p972"]),
    ("LSTM-1082", "lstm", ["ou_marginal/hl_lstm_p1082"]),
]
DEPLOYED = "ou_marginal/hl_mamba_p962"
# (display label, confirmatory label, color key, annotation side)
REFERENCES = [("joint-FTC", "ou_marginal/classical/ftc_joint", "jointftc", "bottom"), ("FNPAG", "ou_marginal/classical/fnpag", "fnpag", "top")]


def main():
    fl.style()
    cells = fl.marginal()

    fig, axes = plt.subplots(1, 2, figsize=fl.SIZE2, sharex=True)
    for ax, metric, title in ((axes[0], "cvar95", "CVaR$_{95}$ (m/s)"), (axes[1], "cvar999", "CVaR$_{99.9}$ (m/s)")):
        for name, label, ckey, side in REFERENCES:
            ref = cells[label]["pooled"][metric]
            ax.axhline(ref, color=fl.C[ckey], lw=1.0, ls="--", zorder=1)
            ax.annotate(f"{name} (retuned): {ref:.0f}", (0.03, ref), xycoords=("axes fraction", "data"), color=fl.C[ckey], fontsize=7.5, va=side)
        for x, (_label, ckey, labels) in enumerate(ARCHS):
            vals = np.array([cells[lb]["pooled"][metric] for lb in labels])
            for lb, v in zip(labels, vals, strict=True):
                if metric == "cvar999":
                    half = 1.96 * cells[lb]["replicate_stats"]["cvar999"]["se"]
                    ax.plot([x, x], [v - half, v + half], color=fl.C[ckey], lw=1.0, zorder=2, alpha=0.8)
                marker, size = ("*", 130) if lb == DEPLOYED else ("o", 34)
                ax.scatter([x], [v], color=fl.C[ckey], marker=marker, s=size, zorder=3, alpha=0.9, edgecolor="white", linewidth=0.6)
            if len(vals) > 1:
                ax.plot([x - 0.22, x + 0.22], [vals.mean()] * 2, color=fl.C[ckey], lw=2.4, zorder=4)
                ax.annotate(f"{vals.mean():.1f}", (x + 0.26, vals.mean()), color=fl.C[ckey], fontsize=8, va="center", fontweight="bold")
        ax.set_xticks(range(len(ARCHS)))
        ax.set_xticklabels([a[0] if len(a[2]) > 1 else f"{a[0]}\n(1 seed)" for a in ARCHS], rotation=12, fontsize=8.5)
        ax.set_ylabel(title)
        ax.margins(x=0.18, y=0.12)
    axes[0].set_title("Sizing tail per seed, headline allocation ($10^6$ per-scenario pool; $\\star$ = deployed)", fontsize=10, loc="left")
    fig.tight_layout()
    fl.save(fig, "fig_arch_tail")


if __name__ == "__main__":
    main()
