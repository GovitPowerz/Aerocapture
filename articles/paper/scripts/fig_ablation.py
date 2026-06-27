"""fig_ablation -- what the deployed Mamba headline uses (interpretability, §9).

Per-input cost increase when each candidate input is zeroed (ablated), ranked, for
the deployed Mamba_962. The net leans on the engineered autoregressive/reference
inputs (eccentricity_excess, hdot_nominal, pdyn_error, predicted_dv2/3) -- the reason
internal recurrence is redundant in the BULK. Data: the bundled
headline/mamba_p962/ablation_results.json (cost transform = log for a clean ranking).
"""

import json

import figlib as fl
import matplotlib.pyplot as plt

TOP_N = 12


def main():
    fl.style()
    a = json.loads((fl.RUNS / "headline/mamba_p962/ablation_results.json").read_text())
    ranked = [x for x in a["ranked"] if not x.get("masked_out") and x["delta"] > 0][:TOP_N]
    ranked = ranked[::-1]  # largest at top
    names = [x["name"] for x in ranked]
    deltas = [x["delta"] for x in ranked]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.barh(range(len(names)), deltas, color=fl.C["mamba"], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("cost increase when input zeroed (log-transform units)")
    ax.set_title(f"Mamba-962 input importance (top {TOP_N})", fontsize=10, loc="left")
    for i, d in enumerate(deltas):
        ax.annotate(f"{d:.2f}", (d, i), textcoords="offset points", xytext=(3, 0), va="center", fontsize=7.5)
    ax.margins(x=0.12)
    fig.tight_layout()
    fl.save(fig, "fig_ablation")


if __name__ == "__main__":
    main()
