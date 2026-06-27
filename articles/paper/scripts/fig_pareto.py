"""fig_pareto -- architecture parameter-budget Pareto + dense capability floor (study 10 / 09).

Panel A: trainable params (log-x) vs the SIZING tail (dv_p99) per architecture family,
from the committed architecture_sweep bundle cells (n=2/5000/512). Panel B: the dense
capability floor (p102..515) -- no capture collapse even at 102 params; sizing-tail sweet
spot ~515. Param counts come from the manifests as DATA (never name-regex).
"""

import json

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

FAMILY_C = {"dense": fl.C["dense"], "gru": fl.C["gru"], "lstm": fl.C["lstm"],
            "mamba": fl.C["mamba"], "transformer": "#d1701f", "window": "#937860"}


def _cells():
    seen, out = set(), []
    for mf in ("manifest.json", "manifest_floor.json"):
        p = fl.REPO / "configs/training/sweep" / mf
        if not p.exists():
            continue
        for e in json.loads(p.read_text()):
            key = (e["arch"], e["params"])
            if key in seen:
                continue
            seen.add(key)
            out.append((e["arch"], e["params"]))
    return out


def _tail(arch: str, params: int):
    par = fl.RUNS / "architecture_sweep" / f"sweep_{arch}_p{params}" / "final_eval.parquet"
    if not par.exists():
        return None
    df = pq.read_table(par).to_pandas()
    cap = (df["ifinal"] == 3) & (df["eccentricity"] < 1.0)
    x = np.sort(df.loc[cap, "dv_total_m_s"].to_numpy())
    cvar95 = x[-max(1, round(len(x) * 0.05)):].mean()
    return float(np.percentile(x, 99)), float(cvar95)


def main():
    fl.style()
    from collections import defaultdict
    fam = defaultdict(list)
    for arch, params in _cells():
        t = _tail(arch, params)
        if t:
            fam[arch].append((params, t[0], t[1]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.7))
    # Panel A: Pareto -- params vs dv_p99 per family
    for arch, pts in sorted(fam.items()):
        pts = sorted(pts)
        xs = [p for p, _, _ in pts]
        ys = [p99 for _, p99, _ in pts]
        axA.plot(xs, ys, "-o", color=FAMILY_C.get(arch, "#555"), label=arch, ms=4, lw=1.4, alpha=0.85)
    axA.set_xscale("log")
    axA.set_xlabel("trainable parameters")
    axA.set_ylabel("dv$_{99}$ (m/s)")
    axA.set_title("Architecture Pareto (sweep, n=2/5000)", fontsize=10, loc="left")
    axA.legend(fontsize=7, ncol=2, frameon=False)

    # Panel B: dense capability floor (sizing tail vs params, the sweet spot)
    dense = sorted(fam["dense"])
    xs = [p for p, _, _ in dense]
    ys = [cv for _, _, cv in dense]
    axB.plot(xs, ys, "-o", color=fl.C["dense"], ms=5, lw=1.6)
    axB.set_xscale("log")
    axB.set_xlabel("dense parameters")
    axB.set_ylabel("dv CVaR$_{95}$ (m/s)")
    axB.set_title("Dense capability floor (100% capture throughout)", fontsize=10, loc="left")
    lo = min(ys)
    sweet = xs[ys.index(lo)]
    axB.annotate(f"sweet spot ~{sweet}p", (sweet, lo), textcoords="offset points", xytext=(6, -12),
                 fontsize=8, color=fl.C["dense"])

    fig.tight_layout()
    fl.save(fig, "fig_pareto")


if __name__ == "__main__":
    main()
