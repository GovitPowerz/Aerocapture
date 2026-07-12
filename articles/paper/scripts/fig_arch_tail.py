"""fig_arch_tail -- THE headline figure (study 10c, confirmatory grade).

Per-seed sigma_run on the mission-sizing tail for the three architectures
trained to the headline depth, evaluated on the frozen confirmatory pool
(10 x 100k scenarios per seed; replicate CIs narrower than the markers).
Two depths -- CVaR99 (left) and CVaR99.9 (right) -- show the recurrent
advantage growing toward the sizing depth. The deployed Mamba seed is starred.
The sample maximum is deliberately NOT plotted: at 1e6 scenarios it is a
single extreme draw (the paper's section 6.2 retires it as a comparison
statistic). Best classical (joint-FTC, confirmatory) drawn as a reference line.
Data: articles/paper/data/confirmatory_eval.json.
"""

import json

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, color key, [s1, s2, s3] confirmatory labels), best -> worst by mean CVaR99.9.
ARCHS = [
    ("Mamba-962", "mamba", ["mamba_p962_long", "paper/tail_repeats/mamba962_s2", "paper/tail_repeats/mamba962_s3"]),
    ("LSTM-1082", "lstm", ["lstm_p1082_long", "paper/tail_repeats/lstm1082_s2", "paper/tail_repeats/lstm1082_s3"]),
    ("Dense-515", "dense", ["dense_p515_ga_paper_best", "paper/tail_repeats/dense515_s2", "paper/tail_repeats/dense515_s3"]),
]
DEPLOYED = "mamba_p962_long"
JOINT_FTC = "joint_reference/ftc"


def main():
    fl.style()
    cells = {c["label"]: c for c in json.loads((fl.DATA / "confirmatory_eval.json").read_text())["cells"]}

    fig, axes = plt.subplots(1, 2, figsize=fl.SIZE2, sharex=True)
    for ax, metric, title in ((axes[0], "cvar99", "CVaR$_{99}$ (m/s)"), (axes[1], "cvar999", "CVaR$_{99.9}$ (m/s)")):
        ftc = cells[JOINT_FTC]["pooled"][metric]
        ax.axhline(ftc, color=fl.C["classical"], lw=1.0, ls="--", zorder=1)
        ax.annotate(f"best classical (joint-FTC): {ftc:.0f}", (0.03, ftc), xycoords=("axes fraction", "data"),
                    color=fl.C["classical"], fontsize=7.5, va="bottom")
        for x, (_label, ckey, labels) in enumerate(ARCHS):
            vals = np.array([cells[lb]["pooled"][metric] for lb in labels])
            for lb, v in zip(labels, vals, strict=True):
                marker, size = ("*", 130) if lb == DEPLOYED else ("o", 34)
                ax.scatter([x], [v], color=fl.C[ckey], marker=marker, s=size, zorder=3,
                           alpha=0.9, edgecolor="white", linewidth=0.6)
            ax.plot([x - 0.22, x + 0.22], [vals.mean()] * 2, color=fl.C[ckey], lw=2.4, zorder=4)
            ax.annotate(f"{vals.mean():.1f}", (x + 0.26, vals.mean()), color=fl.C[ckey],
                        fontsize=8, va="center", fontweight="bold")
        ax.set_xticks(range(len(ARCHS)))
        ax.set_xticklabels([a[0] for a in ARCHS], rotation=12)
        ax.set_ylabel(title)
        ax.margins(x=0.18, y=0.12)
    axes[0].set_title("Sizing tail, per seed (confirmatory pool, $10 \\times 100\\,000$; $\\star$ = deployed)",
                      fontsize=10, loc="left")
    fig.tight_layout()
    fl.save(fig, "fig_arch_tail")


if __name__ == "__main__":
    main()
