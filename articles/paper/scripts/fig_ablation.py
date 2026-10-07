"""fig_ablation -- what the deployed v4 Mamba uses (closed-loop input sensitivity, Section 8).

Per-input cost increase when each candidate input is zeroed (ablated), ranked, for the deployed
champion ou_marginal/hl_mamba_p962 (#176: `aerocapture.training.ablation`, n = 1000 scenarios of
the config's own Monte Carlo under per-scenario noise, costs in the log transform for a clean
ranking; both recorded in the file and asserted here). Data: the bundled
runs/ou_marginal/hl_mamba_p962/ablation_results.json.
"""

import json

import figlib as fl
import matplotlib.pyplot as plt

TOP_N = 12


def main():
    fl.style()
    a = json.loads((fl.RUNS / "ou_marginal/hl_mamba_p962/ablation_results.json").read_text())
    assert a["noise_seeding"] == "per_draw" and a["cost_transform"] == "log", (a["noise_seeding"], a["cost_transform"])
    ranked = [x for x in a["ranked"] if not x.get("masked_out") and x["delta"] > 0][:TOP_N]
    ranked = ranked[::-1]  # largest at top
    names = [x["name"] for x in ranked]
    deltas = [x["delta"] for x in ranked]

    fig, ax = plt.subplots(figsize=fl.SIZE1)
    ax.barh(range(len(names)), deltas, color=fl.C["mamba"], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("cost increase when input zeroed (log-transform units)")
    ax.set_title(f"Mamba-962 (deployed) input importance (top {TOP_N}, n = {a['n_sims']}, per-scenario noise)", fontsize=10, loc="left")
    for i, d in enumerate(deltas):
        ax.annotate(f"{d:.2f}", (d, i), textcoords="offset points", xytext=(3, 0), va="center", fontsize=7.5)
    ax.margins(x=0.12)
    fig.tight_layout()
    fl.save(fig, "fig_ablation")


if __name__ == "__main__":
    main()
