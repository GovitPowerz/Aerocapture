"""fig_corridor -- the aerocapture corridor (problem & objective, §3).

A Monte-Carlo ensemble of deployed (Mamba_962) trajectories in the (orbital energy,
dynamic pressure) plane: the spacecraft enters hyperbolic (energy > 0) and the
guidance must bleed energy through the atmosphere into a bound capture orbit
(energy < 0) WITHOUT overshooting the dynamic-pressure / heating corridor. Data:
articles/paper/data/corridor.npz (a 60-sim ensemble, downsampled).
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np


def main():
    fl.style()
    d = np.load(fl.DATA / "corridor.npz", allow_pickle=True)
    energy, pdyn, captured = d["energy"], d["pdyn"], d["captured"]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for e, p, c in zip(energy, pdyn, captured, strict=True):
        ax.plot(e, p, color=(fl.C["mamba"] if c else fl.C["fnpag"]),
                lw=0.7, alpha=0.55 if c else 0.9, zorder=2 if c else 3)
    ax.axvline(0.0, color="#444", lw=1.0, ls="--", zorder=1)
    ytop = ax.get_ylim()[1]
    ax.text(-6.1, ytop * 0.99, "captured\n(E < 0)", fontsize=8, color="#444", va="top", ha="left")
    ax.text(4.9, ytop * 0.99, "hyperbolic\n(E > 0)", fontsize=8, color="#444", va="top", ha="right")
    ax.set_xlabel("orbital energy (MJ/kg)")
    ax.set_ylabel("dynamic pressure (kPa)")
    ax.set_title("Aerocapture corridor: Mamba_962 MC ensemble", fontsize=10, loc="left")
    # legend proxies
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=fl.C["mamba"], label="captured"),
                       Line2D([], [], color=fl.C["fnpag"], label="failed")],
              frameon=True, framealpha=0.85, fontsize=8, loc="lower left")
    fig.tight_layout()
    fl.save(fig, "fig_corridor")


if __name__ == "__main__":
    main()
