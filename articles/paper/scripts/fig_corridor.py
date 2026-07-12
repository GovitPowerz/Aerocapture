"""fig_corridor -- the empirical trajectory-occupancy envelope (problem & objective, §2).

Shaded occupancy envelope between the p99.9 (upper, crash-side) and p0.5 (lower,
escape-side) dynamic-pressure quantiles traced by a dispersed randomized
piecewise-constant MC (collect_corridor.py) -- an empirical construct, not a
formal reachable set (reviewer R1-11) -- with the undispersed full-lift-up /
full-lift-down constant-bank boundary traces overlaid (dashed) and the deployed
Mamba MC ensemble + its undispersed nominal flying inside.
Data: articles/paper/data/corridor.npz.
"""

import math

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    fl.style()
    d = np.load(fl.DATA / "corridor.npz", allow_pickle=True)
    e, lo, hi = d["energy_bins"], d["lower_pdyn"], d["upper_pdyn"]
    green = fl.C["mamba"]

    fig, ax = plt.subplots(figsize=fl.SIZE1)
    ax.fill_between(e, lo, hi, color=green, alpha=0.13, lw=0, zorder=0)
    ax.plot(e, hi, color=green, lw=1.3, zorder=2)
    ax.plot(e, lo, color=green, lw=1.3, zorder=2)

    ens_e, ens_p = d["ens_energy"], d["ens_pdyn"]
    alpha = max(0.04, min(0.22, 1.5 / math.sqrt(max(len(ens_e), 1))))
    for te, tp in zip(ens_e, ens_p, strict=True):
        ax.plot(te, tp, color=green, lw=0.5, alpha=alpha, zorder=3)
    ax.plot(d["nominal_energy"], d["nominal_pdyn"], color="#111", lw=1.6, zorder=4, solid_capstyle="round")

    boundary_color = "#8a5a00"
    if "liftup_energy" in d.files:
        ax.plot(d["liftup_energy"], d["liftup_pdyn"], color=boundary_color, lw=1.2, ls=(0, (5, 3)), zorder=2)
        ax.plot(d["liftdown_energy"], d["liftdown_pdyn"], color=boundary_color, lw=1.2, ls=(0, (1, 2)), zorder=2)

    ax.axvline(0.0, color="#555", lw=0.9, ls="--", zorder=1)
    ax.set_xlim(float(e.min()), float(e.max()))
    ax.set_ylim(0, float(np.nanmax(hi)) * 1.08)
    ytop = ax.get_ylim()[1]
    ax.text(-0.3, ytop * 0.97, "bound\n(E < 0)", fontsize=8, color="#555", va="top", ha="right")
    ax.text(0.3, ytop * 0.97, "hyperbolic\n(E > 0)", fontsize=8, color="#555", va="top", ha="left")
    ax.set_xlabel("orbital energy (MJ/kg)")
    ax.set_ylabel("dynamic pressure (kPa)")
    ax.legend(handles=[
        Patch(facecolor=green, alpha=0.16, edgecolor=green, label="occupancy envelope (randomized profiles)"),
        Line2D([], [], color=boundary_color, lw=1.2, ls=(0, (5, 3)), label="full lift-up (escape side)"),
        Line2D([], [], color=boundary_color, lw=1.2, ls=(0, (1, 2)), label="full lift-down (crash side)"),
        Line2D([], [], color=green, lw=1.0, alpha=0.6, label=f"deployed Mamba MC ensemble ({len(ens_e)})"),
        Line2D([], [], color="#111", lw=1.6, label="undispersed nominal"),
    ], loc="upper left")
    fig.tight_layout()
    fl.save(fig, "fig_corridor")


if __name__ == "__main__":
    main()
