"""fig_classical_vs_nn -- the deployability scatter (compute versus the sizing tail).

x = per-simulation compute cost (ms/sim on one idle core, log scale; data/compute_benchmark.json,
measured on the shared-path champions, whose architectures the v4 cells share, so the cost is
the cell's), y = far-tail CVaR99.9 of the correction DV on the 10^6 per-scenario confirmatory
pool (10 x 100 000 scenarios per cell, per_draw, asserted at load). Mamba-962 and Dense-515
are three-seed means of the headline-allocation runs (#173, scratch at GA 512 x 2); joint-FTC
and FNPAG are the retuned classical cells (#172); joint-FTC rides FTC's compute. Lower-left is
better: cheap and tight-tailed. Axis limits derive from the data (an arxiv-v2 proof caught
FNPAG clipped by a stale ylim). Half-column figure: point labels carry the name and the tail
value only. Data: articles/paper/data/confirmatory_marginal.json, compute_benchmark.json.
"""

import figlib as fl
import matplotlib.pyplot as plt

# (display label, color key, compute label, confirmatory labels (averaged), label offset (dx, dy) in points;
# Dense-515 labels to the left of its point, above the "better" guide)
POINTS = [
    ("Mamba-962", "mamba", "NN-mamba", ["ou_marginal/hl_mamba_p962", "ou_marginal/hl_mamba_p962_s2", "ou_marginal/hl_mamba_p962_s3"], (7, 4)),
    ("Dense-515", "dense", "NN-dense", ["ou_marginal/hl_dense_p515", "ou_marginal/hl_dense_p515_s2", "ou_marginal/hl_dense_p515_s3"], (-7, 4)),
    ("joint-FTC", "jointftc", "FTC", ["ou_marginal/classical/ftc_joint"], (7, 6)),
    ("FNPAG", "fnpag", "FNPAG", ["ou_marginal/classical/fnpag"], (-7, -14)),
]


def main():
    fl.style()
    conf = fl.marginal()
    ms = {s["label"]: s["ms_per_sim"] for s in fl.compute()}

    fig, ax = plt.subplots(figsize=fl.SIZE_HALF)
    xs, ys = [], []
    for label, ckey, bench, cells, (dx, dy) in POINTS:
        xv = ms[bench]
        yv = sum(conf[c]["pooled"]["cvar999"] for c in cells) / len(cells)
        xs.append(xv)
        ys.append(yv)
        ax.scatter([xv], [yv], color=fl.C[ckey], s=90, zorder=4, edgecolor="white", linewidth=1.0)
        ax.annotate(
            f"{label}\n{yv:.0f} m/s",
            (xv, yv),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            color=fl.C[ckey],
            fontweight="bold",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
        )

    # lower-left = better guide
    ax.annotate("better\n(cheap + tight tail)", xy=(0.04, 0.05), xycoords="axes fraction", fontsize=8, color="#555555", ha="left", va="bottom", style="italic")

    ax.set_xscale("log")
    ax.set_xlabel("compute cost (ms / sim, log scale)")
    ax.set_ylabel("far-tail CVaR$_{99.9}$ (m/s)")
    ax.set_title("Deployability: tail vs compute")
    # Limits from the data: a decade of margin either side in x, the label stack above and below in y.
    ax.set_xlim(min(xs) / 2.0, max(xs) * 2.0)
    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - 0.45 * span, max(ys) + 0.35 * span)
    fig.tight_layout()
    fl.save(fig, "fig_classical_vs_nn")


if __name__ == "__main__":
    main()
