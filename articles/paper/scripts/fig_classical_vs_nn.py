"""fig_classical_vs_nn -- the deployability scatter (compute vs sizing tail).

x = per-sim compute cost (ms/sim, LOG), y = far-tail CVaR99.9 (m/s). Lower-left is
better: cheap AND tight-tailed. The recurrent NN (Mamba-962) sits lower-left of
BOTH classical references (joint-FTC ~164, FNPAG ~165) -- it sizes a smaller
ergol margin (lower tail DV) at a fraction of FNPAG's per-sim cost, while staying
within ~3x of FTC's compute. Dense-515 is the cheapest NN but carries a fatter tail.
"""

import figlib as fl
import matplotlib.pyplot as plt

# (display label, color key, compute.label, far_tail y-value, label offset (dx, dy) in points)
# Mamba and dense y = 3-seed mean CVaR99.9 (124.5 / 139.2); classicals annotated below.
POINTS = [
    ("Mamba-962", "mamba", "NN-mamba", 124.5, (8, 6)),
    ("Dense-515", "dense", "NN-dense", None, (8, -14)),
    ("joint-FTC", "jointftc", "FTC", None, (-8, 10)),
    ("FNPAG", "fnpag", "FNPAG", None, (-10, -16)),
]


def main():
    fl.style()
    ft = fl.far_tail()
    ms = {s["label"]: s["ms_per_sim"] for s in fl.compute()}

    # far-tail CVaR99.9 per point. Mamba uses the 3-seed mean; dense the s1 cell;
    # the classical references come from the committed far_tail cells (joint-FTC
    # = joint_reference/ftc ~164, FNPAG ~165 -- the bars the NN must beat).
    # Dense-515 on the SAME 3-seed-mean basis as Mamba (and Table 3), not the
    # lucky s1 cell -- (128.11 + 139.84 + 149.61) / 3 = 139.2.
    dense515_mean = sum(
        ft[k]["cvar999"]
        for k in ("dense_p515_ga_paper_best", "paper/tail_repeats/dense515_s2", "paper/tail_repeats/dense515_s3")
    ) / 3.0
    y = {
        "Mamba-962": 124.5,
        "Dense-515": dense515_mean,                               # 139.2 (3-seed mean)
        "joint-FTC": ft["joint_reference/ftc"]["cvar999"],        # ~164
        "FNPAG": ft["fnpag"]["cvar999"],                          # ~165
    }
    # compute label -> the actual per-sim ms; joint-FTC rides FTC's compute cost.
    x = {
        "Mamba-962": ms["NN-mamba"],   # 3.68
        "Dense-515": ms["NN-dense"],   # 2.40
        "joint-FTC": ms["FTC"],        # 1.25
        "FNPAG": ms["FNPAG"],          # 86.1
    }

    fig, ax = plt.subplots(figsize=fl.SIZE_HALF)

    for label, ckey, _clbl, _yover, (dx, dy) in POINTS:
        xv, yv = x[label], y[label]
        ax.scatter([xv], [yv], color=fl.C[ckey], s=110, zorder=4,
                   edgecolor="white", linewidth=1.0)
        ax.annotate(f"{label}\n{yv:.0f} m/s @ {xv:.2f} ms",
                    (xv, yv), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=fl.C[ckey], fontweight="bold",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")

    # lower-left = better guide arrow
    ax.annotate("better\n(cheap + tight tail)", xy=(0.04, 0.06), xycoords="axes fraction",
                fontsize=8, color="#555555", ha="left", va="bottom", style="italic")

    ax.set_xscale("log")
    ax.set_xlabel("compute cost (ms / sim, log scale)")
    ax.set_ylabel("far-tail CVaR$_{99.9}$ (m/s)")
    ax.set_title("Deployability: sizing tail vs per-sim compute", fontsize=10, loc="left")
    ax.set_xlim(0.9, 130)
    ax.set_ylim(115, 175)
    fig.tight_layout()
    fl.save(fig, "fig_classical_vs_nn")


if __name__ == "__main__":
    main()
