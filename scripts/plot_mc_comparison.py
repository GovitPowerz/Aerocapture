"""Compare FTC vs NN guidance Monte Carlo results from Rust simulator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aerocapture.plotting.stats import empirical_cdf


def load_final(path: str | Path) -> np.ndarray:
    """Load a final conditions CSV file into a numpy array."""
    df = pd.read_csv(path)
    return df.to_numpy()


def main() -> None:
    base = Path("output")
    ftc = load_final(base / "final.mc100_ftc")
    nn = load_final(base / "final.mc100_nn")

    print(f"FTC: {ftc.shape[0]} sims, NN: {nn.shape[0]} sims")

    # Column mapping (0-indexed, from carltf.f xsauve):
    #  8  = energy (MJ/kg)
    # 10  = eccentricity
    # 14  = periapsis alt (km)
    # 15  = apoapsis alt (km)
    # 28  = sim time (s)
    # 30  = periapsis error (km)
    # 31  = apoapsis error (km)
    # 42  = total delta-V (m/s)
    # 32  = ifinal (termination code: 1=crash, 2=timeout, 3=exit)

    # Classify outcomes
    ftc_exit = ftc[:, 32] == 3
    nn_exit = nn[:, 32] == 3
    ftc_crash = ftc[:, 32] == 1
    nn_crash = nn[:, 32] == 1

    print(f"FTC: {ftc_exit.sum()} exit, {ftc_crash.sum()} crash, {(~ftc_exit & ~ftc_crash).sum()} timeout")
    print(f"NN:  {nn_exit.sum()} exit, {nn_crash.sum()} crash, {(~nn_exit & ~nn_crash).sum()} timeout")

    # Filter to successful exits for orbit quality comparison
    ftc_ok = ftc[ftc_exit]
    nn_ok = nn[nn_exit]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Monte Carlo Comparison: FTC vs NN Guidance (100 sims, Rust)", fontsize=14, fontweight="bold")

    colors = {"ftc": "#2563eb", "nn": "#dc2626"}

    # --- Plot 1: Apoapsis error CDF ---
    ax = axes[0, 0]
    if len(ftc_ok):
        x, y = empirical_cdf(ftc_ok[:, 31])
        ax.plot(x, y, color=colors["ftc"], linewidth=2, label=f"FTC (n={len(ftc_ok)})")
    if len(nn_ok):
        x, y = empirical_cdf(nn_ok[:, 31])
        ax.plot(x, y, color=colors["nn"], linewidth=2, label=f"NN (n={len(nn_ok)})")
    ax.set_xlabel("Apoapsis altitude error (km)")
    ax.set_ylabel("CDF")
    ax.set_title("Apoapsis Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Periapsis error CDF ---
    ax = axes[0, 1]
    if len(ftc_ok):
        x, y = empirical_cdf(ftc_ok[:, 30])
        ax.plot(x, y, color=colors["ftc"], linewidth=2, label=f"FTC (n={len(ftc_ok)})")
    if len(nn_ok):
        x, y = empirical_cdf(nn_ok[:, 30])
        ax.plot(x, y, color=colors["nn"], linewidth=2, label=f"NN (n={len(nn_ok)})")
    ax.set_xlabel("Periapsis altitude error (km)")
    ax.set_ylabel("CDF")
    ax.set_title("Periapsis Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Total delta-V CDF ---
    ax = axes[0, 2]
    if len(ftc_ok):
        dv_ftc = ftc_ok[:, 42]
        dv_ftc = dv_ftc[dv_ftc < 1e10]  # filter bogus values
        if len(dv_ftc):
            x, y = empirical_cdf(dv_ftc)
            ax.plot(x, y, color=colors["ftc"], linewidth=2, label=f"FTC (n={len(dv_ftc)})")
    if len(nn_ok):
        dv_nn = nn_ok[:, 42]
        dv_nn = dv_nn[dv_nn < 1e10]
        if len(dv_nn):
            x, y = empirical_cdf(dv_nn)
            ax.plot(x, y, color=colors["nn"], linewidth=2, label=f"NN (n={len(dv_nn)})")
    ax.set_xlabel("Total correction delta-V (m/s)")
    ax.set_ylabel("CDF")
    ax.set_title("Orbit Correction Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Apoapsis vs Periapsis scatter ---
    ax = axes[1, 0]
    if len(ftc_ok):
        ax.scatter(ftc_ok[:, 30], ftc_ok[:, 31], color=colors["ftc"], alpha=0.5, s=20, label="FTC")
    if len(nn_ok):
        ax.scatter(nn_ok[:, 30], nn_ok[:, 31], color=colors["nn"], alpha=0.5, s=20, label="NN")
    # Mark crashes
    if ftc_crash.sum():
        ax.scatter(ftc[ftc_crash, 30], ftc[ftc_crash, 31], color=colors["ftc"], marker="x", s=40, alpha=0.7, label="FTC crash")
    if nn_crash.sum():
        ax.scatter(nn[nn_crash, 30], nn[nn_crash, 31], color=colors["nn"], marker="x", s=40, alpha=0.7, label="NN crash")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Periapsis error (km)")
    ax.set_ylabel("Apoapsis error (km)")
    ax.set_title("Orbit Error Scatter")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 5: Energy CDF ---
    ax = axes[1, 1]
    x, y = empirical_cdf(ftc[:, 8])
    ax.plot(x, y, color=colors["ftc"], linewidth=2, label="FTC")
    x, y = empirical_cdf(nn[:, 8])
    ax.plot(x, y, color=colors["nn"], linewidth=2, label="NN")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Final orbital energy (MJ/kg)")
    ax.set_ylabel("CDF")
    ax.set_title("Final Energy (>0 = hyperbolic)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Summary stats table ---
    ax = axes[1, 2]
    ax.axis("off")

    def stat_row(data: np.ndarray, col: int, label: str, unit: str) -> list[str]:
        vals = data[:, col]
        vals = vals[np.isfinite(vals) & (np.abs(vals) < 1e10)]
        if len(vals) == 0:
            return [label, "N/A", "N/A", unit]
        return [label, f"{np.mean(vals):.1f}", f"{np.std(vals):.1f}", unit]

    rows = [
        ["Metric", "FTC mean", "FTC std", "NN mean", "NN std", "Unit"],
        ["Captured", f"{ftc_exit.sum()}", "/100", f"{nn_exit.sum()}", "/100", ""],
        ["Crashed", f"{ftc_crash.sum()}", "/100", f"{nn_crash.sum()}", "/100", ""],
    ]

    if len(ftc_ok) and len(nn_ok):
        for col, label, unit in [
            (31, "Apo err", "km"),
            (30, "Peri err", "km"),
            (42, "DV total", "m/s"),
            (28, "Sim time", "s"),
            (8, "Energy", "MJ/kg"),
        ]:
            fv = ftc_ok[:, col]
            nv = nn_ok[:, col]
            fv = fv[np.isfinite(fv) & (np.abs(fv) < 1e10)]
            nv = nv[np.isfinite(nv) & (np.abs(nv) < 1e10)]
            rows.append([
                label,
                f"{np.mean(fv):.1f}" if len(fv) else "N/A",
                f"{np.std(fv):.1f}" if len(fv) else "",
                f"{np.mean(nv):.1f}" if len(nv) else "N/A",
                f"{np.std(nv):.1f}" if len(nv) else "",
                unit,
            ])

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title("Summary Statistics", fontweight="bold", pad=20)

    plt.tight_layout()
    out = Path("mc_comparison_ftc_vs_nn.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
