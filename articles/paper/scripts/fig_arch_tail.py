"""fig_arch_tail -- THE headline figure (study 10c).

3-seed sigma_run on the mission-SIZING TAIL for the three architectures trained
to the headline depth (n=2/512/20000 gens), far-tail n=10000. Shows that both
recurrent nets (Mamba_962, LSTM_1082) beat the dense net beyond run-to-run
variance, and that Mamba is the lowest AND tightest -- with the best classical
(joint-FTC / FNPAG ~164/165) drawn as a reference band the NN crushes.
"""

import numpy as np

import figlib as fl

# (display label, color key, [s1, s2, s3] far_tail labels) ordered best -> worst by mean CVaR99.9.
ARCHS = [
    ("Mamba-962", "mamba", ["mamba_p962_long", "paper/tail_repeats/mamba962_s2", "paper/tail_repeats/mamba962_s3"]),
    ("LSTM-1082", "lstm", ["lstm_p1082_long", "paper/tail_repeats/lstm1082_s2", "paper/tail_repeats/lstm1082_s3"]),
    ("Dense-515", "dense", ["dense_p515_ga_paper_best", "paper/tail_repeats/dense515_s2", "paper/tail_repeats/dense515_s3"]),
]
# Best classical reference (far-tail CVaR99.9), drawn as a band.
CLASSICAL = {"joint-FTC": 164.0, "FNPAG": 165.0}


def main():
    fl.style()
    ft = fl.far_tail()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    for ax, metric, title in ((axes[0], "cvar999", "CVaR$_{99.9}$ (m/s)"), (axes[1], "max", "worst case (m/s)")):
        # classical reference band (the bar the NN must beat)
        lo, hi = min(CLASSICAL.values()), max(CLASSICAL.values())
        ax.axhspan(lo, hi, color=fl.C["classical"], alpha=0.12, zorder=0)
        ax.axhline(lo, color=fl.C["classical"], lw=0.8, ls="--", zorder=1)
        ax.text(2.45, hi + 0.5, "best classical\n(joint-FTC / FNPAG)", color=fl.C["classical"],
                fontsize=7.5, ha="right", va="bottom")
        for x, (label, ckey, cells) in enumerate(ARCHS):
            vals = np.array([ft[c][metric] for c in cells])
            ax.scatter([x] * len(vals), vals, color=fl.C[ckey], s=34, zorder=3, alpha=0.85)
            ax.plot([x - 0.22, x + 0.22], [vals.mean()] * 2, color=fl.C[ckey], lw=2.4, zorder=4)
            ax.annotate(f"{vals.mean():.1f}", (x + 0.26, vals.mean()), color=fl.C[ckey],
                        fontsize=8, va="center", fontweight="bold")
        ax.set_xticks(range(len(ARCHS)))
        ax.set_xticklabels([a[0] for a in ARCHS], rotation=12)
        ax.set_ylabel(title)
        ax.margins(x=0.18)
    axes[0].set_title("Sizing tail (3-seed $\\sigma_{run}$, far-tail n=10000)", fontsize=10, loc="left")
    fig.tight_layout()
    fl.save(fig, "fig_arch_tail")


if __name__ == "__main__":
    main()
