"""fig_islands -- schematic of the 3-island heterogeneous optimizer (§5).

Three islands run complementary gradient-free operators (particle swarm,
genetic, differential evolution) on the same scenarios; every k_period
generations each island's best individuals migrate into the others,
replacing their worst. Heterogeneous search + migration = the diversity that
escapes local optima, and one population budget covers three strategies.
Schematic (no run data).
"""

import figlib as fl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BW, BH = 0.34, 0.19  # island box width/height (axis fraction)


def _island(ax, xy, color, name, sub):
    x, y = xy[0] - BW / 2, xy[1] - BH / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), BW, BH, boxstyle="round,pad=0.0,rounding_size=0.03",
        linewidth=1.4, edgecolor=color, facecolor=color + "20", zorder=3))
    ax.text(xy[0], xy[1] + 0.028, name, ha="center", va="center", fontsize=10, weight="bold", color=color, zorder=4)
    ax.text(xy[0], xy[1] - 0.036, sub, ha="center", va="center", fontsize=8, color="#444", zorder=4)


def _migrate(ax, a, b, rad):
    ax.add_patch(FancyArrowPatch(
        a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="<|-|>",
        mutation_scale=13, linewidth=1.2, color="#777",
        shrinkA=52, shrinkB=52, zorder=2))


def main():
    fl.style()
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pso = (0.5, 0.80)
    ga = (0.21, 0.34)
    de = (0.79, 0.34)

    _migrate(ax, pso, ga, 0.16)
    _migrate(ax, ga, de, 0.16)
    _migrate(ax, de, pso, 0.16)

    _island(ax, pso, fl.C["mamba"], "Particle swarm", "island 1")
    _island(ax, ga, fl.C["dense"], "Genetic", "island 2")
    _island(ax, de, fl.C["jointftc"], "Differential evolution", "island 3")

    ax.text(0.5, 0.50, "migrate\ntop-$n$", ha="center", va="center",
            fontsize=8, color="#777", style="italic", zorder=4)
    ax.text(0.5, 0.05,
            "Every $k$ generations each island's best $n$ individuals replace the worst in the others\n"
            "(swarm destinations also get fresh velocity). One budget, three complementary searches.",
            ha="center", va="center", fontsize=8, color="#333")
    fig.tight_layout()
    fl.save(fig, "fig_islands")


if __name__ == "__main__":
    main()
