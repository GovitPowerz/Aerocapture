"""fig_quantization -- the post-training quantization grid of the v4 champion (issues #178, #180).

Every PTQ variant of data/quant_v4/quantization_results.json (asserted per_draw): the deployed
Mamba champion with its weights rounded to 8 / 6 / 4 / 3 / 2 bits, per-channel or per-tensor
scales, every tensor or the projections only, each flown on the fresh 8M re-quote pool at
n = 1000 under per-scenario noise. Left: capture rate (%). Right: correction-DV CVaR95 (m/s,
over captured runs; a cell that captures little has a CVaR95 over few runs, so the right panel
is read with the left). One line per granularity x policy series, the full-precision baseline
dashed, the file's verdict cell (max capture, then min CVaR95 at 4 bits) starred. The x axis
runs from 8 bits down so degradation reads left to right. Reads only committed JSON.
"""

import json

import figlib as fl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

DATA = fl.DATA / "quant_v4/quantization_results.json"
# (granularity, tensor_policy) -> (label, color, marker)
SERIES = {
    ("per_channel", "all"): ("per-channel / all", fl.C["fnpag"], "o"),
    ("per_channel", "proj_only"): ("per-channel / projections", fl.C["jointftc"], "s"),
    ("per_tensor", "all"): ("per-tensor / all", fl.C["dense"], "o"),
    ("per_tensor", "proj_only"): ("per-tensor / projections", fl.C["mamba"], "s"),
}


def main():
    fl.style()
    d = json.loads(DATA.read_text())
    assert d["noise_seeding"] == "per_draw", d["noise_seeding"]
    base = d["baseline"]
    verdict = d["verdict"]
    found = {(v["granularity"], v["tensor_policy"]) for v in d["variants"]}
    if found != set(SERIES):
        raise KeyError(f"series in {DATA}: {sorted(found)}, expected {sorted(SERIES)}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=fl.SIZE2)
    for (gran, policy), (label, color, marker) in SERIES.items():
        rows = sorted((v for v in d["variants"] if v["granularity"] == gran and v["tensor_policy"] == policy), key=lambda r: -r["bits"])
        bits = [r["bits"] for r in rows]
        axL.plot(bits, [100.0 * r["capture_rate"] for r in rows], marker=marker, ms=4.5, color=color, label=label)
        axR.plot(bits, [r["dv_cvar95"] for r in rows], marker=marker, ms=4.5, color=color, label=label)
    for ax, y in ((axL, 100.0 * base["capture_rate"]), (axR, base["dv_cvar95"])):
        ax.axhline(y, ls="--", lw=1.0, color="#666666", label="full precision")
    star = dict(marker="*", s=160, color="black", edgecolor="white", linewidth=0.8, zorder=5, label=f"verdict ({verdict['bits']} b)")
    for ax, y in ((axL, 100.0 * verdict["capture_rate"]), (axR, verdict["dv_cvar95"])):
        ax.scatter([verdict["bits"]], [y], **star)
    bits = sorted({v["bits"] for v in d["variants"]}, reverse=True)
    for ax in (axL, axR):
        ax.invert_xaxis()
        ax.set_xticks(bits)
        ax.set_xlabel("weight bits")

    axL.set_ylim(0, 105)
    axL.set_ylabel("capture rate (%)")
    axL.set_title(f"Capture under PTQ (8M re-quote pool, n = {d['n_sims']}, per-scenario)")
    axL.legend(loc="lower left")
    axR.set_yscale("log")  # the 2-bit cells reach 1344 m/s; a linear axis would flatten the 4-bit comparison around 200-300
    axR.yaxis.set_major_locator(FixedLocator([100, 150, 200, 300, 500, 1000, 1500]))
    axR.yaxis.set_minor_locator(NullLocator())
    axR.yaxis.set_major_formatter(ScalarFormatter())
    axR.set_ylabel("CVaR$_{95}$ (m/s, over captures)")
    axR.set_title("Correction-DV tail under PTQ")
    fig.tight_layout()
    fl.save(fig, "fig_quantization")


if __name__ == "__main__":
    main()
