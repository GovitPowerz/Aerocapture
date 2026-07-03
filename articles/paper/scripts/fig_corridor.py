"""fig_corridor -- the aerocapture corridor (problem & objective, §3).

Report-style corridor panel (cf. src/python/aerocapture/training/charts.py
chart_corridor_pdyn): a Monte-Carlo ensemble of the deployed Mamba_962 policy
in the (orbital energy, dynamic pressure) plane, drawn as classified spaghetti
over the four-layer corridor zone fills, with the undispersed nominal on top.
The vehicle enters hyperbolic (energy > 0, right) and the guidance bleeds
energy through the atmosphere into a bound capture orbit (energy < 0, left)
without leaving the dynamic-pressure / heating corridor. Data:
articles/paper/data/corridor.npz (built by collect_corridor.py).
"""

import math

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Three-way classification codes (mirror charts.py TRAJ_OK / _CONSTRAINED / _FAILED).
TRAJ_OK, TRAJ_CONSTRAINED, TRAJ_FAILED = 0, 1, 2
COL = {TRAJ_OK: fl.C["mamba"], TRAJ_CONSTRAINED: "#d1701f", TRAJ_FAILED: "#c44e52"}
CRASH = "#c44e52"  # danger zones (crash above, hyperbolic escape below)


def main():
    fl.style()
    d = np.load(fl.DATA / "corridor.npz", allow_pickle=True)
    energy, pdyn, cls = d["energy"], d["pdyn"], d["traj_class"]
    eb = d["energy_bins"]

    fig, ax = plt.subplots(figsize=(7.0, 4.1))

    # --- Four-layer corridor zone fills (behind everything) ---
    crash = d["envelope_crash_pdyn"]
    r_max = d["envelope_restricted_max_pdyn"]
    r_min = d["envelope_restricted_min_pdyn"]
    cap = d["envelope_capture_pdyn"]
    ax.fill_between(eb, r_max, crash, color=CRASH, alpha=0.12, lw=0, zorder=0)          # crash (too deep)
    ax.fill_between(eb, r_min, r_max, color=fl.C["mamba"], alpha=0.16, lw=0, zorder=0)  # restricted corridor
    ax.fill_between(eb, cap, r_min, color="#b0b0b0", alpha=0.16, lw=0, zorder=0)        # transition
    ax.fill_between(eb, 0, cap, color=CRASH, alpha=0.12, lw=0, zorder=0)                # hyperbolic (too shallow)

    # --- Classified spaghetti (alpha scales with count, like the report) ---
    alpha = max(0.04, min(0.22, 1.5 / math.sqrt(len(cls))))
    for e, p, c in zip(energy, pdyn, cls, strict=True):
        ax.plot(e, p, color=COL.get(int(c), CRASH), lw=0.6, alpha=alpha, zorder=2)

    # --- Undispersed nominal on top ---
    ax.plot(d["nominal_energy"], d["nominal_pdyn"], color="#111", lw=1.6, zorder=4, solid_capstyle="round")

    ax.axvline(0.0, color="#555", lw=0.9, ls="--", zorder=1)
    ax.set_xlim(eb.min(), eb.max())
    ax.set_ylim(0, float(np.nanmax(crash)) * 1.02)
    ytop = ax.get_ylim()[1]
    ax.text(-0.3, ytop * 0.97, "bound\n(E < 0)", fontsize=8, color="#555", va="top", ha="right")
    ax.text(0.3, ytop * 0.97, "hyperbolic\n(E > 0)", fontsize=8, color="#555", va="top", ha="left")
    ax.set_xlabel("orbital energy (MJ/kg)")
    ax.set_ylabel("dynamic pressure (kPa)")

    n_ok = int((cls == TRAJ_OK).sum())
    handles = [
        Patch(facecolor=fl.C["mamba"], alpha=0.16, label="restricted corridor"),
        Patch(facecolor=CRASH, alpha=0.12, label="crash / escape zones"),
        Line2D([], [], color=fl.C["mamba"], lw=1.2, label=f"captured ({n_ok} of {len(cls)})"),
        Line2D([], [], color="#111", lw=1.6, label="undispersed nominal"),
    ]
    if (cls == TRAJ_FAILED).any():
        handles.insert(3, Line2D([], [], color=CRASH, lw=1.2, label="failed"))
    ax.legend(handles=handles, frameon=True, framealpha=0.9, fontsize=8, loc="upper left")
    fig.tight_layout()
    fl.save(fig, "fig_corridor")


if __name__ == "__main__":
    main()
