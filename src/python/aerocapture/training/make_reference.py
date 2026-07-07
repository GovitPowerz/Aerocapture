"""Target-energy-matched constant-bank reference generator.

Generates training_output/<mission>/ref_trajectory.dat from a constant-bank
nominal whose exit energy hits the target orbit energy minus an overshoot
margin -- the legacy msr_aller.dat methodology (a 64.77 deg trim nominal that
overshoots the target by ~0.7 MJ/kg), reproduced inside the pipeline.

The bank is deliberately NOT a GA product: open-loop-optimal profiles
under-capture relative to the target and leave the reference table short of
the energies the trackers must fly through (11-segment optimum: 0.43 MJ/kg
short, FTC val RMS 9.7e6; 1-segment optimum: 1.65 MJ/kg short, 1.5e11; the
legacy energy-overshooting reference: 2.7e6).

Usage:
    uv run python -m aerocapture.training.make_reference \
        --toml configs/training/msr_aller_pc_ref_train.toml [--overshoot-mj 0.7]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Crash sentinel: far below any physical exit energy, keeps the bisection
# bracket valid when the lift-down side of the scan impacts the surface.
CRASH_ENERGY_MJ = -1e9


def target_orbit_energy_mj(toml_data: dict) -> float:
    """Target orbit specific energy (MJ/kg) from the resolved config:
    E = -mu / (2 a). `[flight.target_orbit]` is in km (altitudes); a comes from
    semi_major_axis when present, else (apoapsis + periapsis)/2 + equatorial radius."""
    planet = toml_data["planet"]
    orbit = toml_data["flight"]["target_orbit"]
    if "semi_major_axis" in orbit:
        sma = float(orbit["semi_major_axis"]) * 1e3
    else:
        sma = (float(orbit["apoapsis"]) + float(orbit["periapsis"])) / 2.0 * 1e3 + float(planet["equatorial_radius"])
    return -float(planet["mu"]) / (2.0 * sma) / 1e6


def bisect_bank_for_exit_energy(
    exit_energy_fn: Callable[[float], float],
    *,
    target_mj: float,
    lo: float,
    hi: float,
    tol_mj: float = 0.02,
    max_iter: int = 40,
) -> float:
    """Bank angle (deg) whose nominal exit energy equals `target_mj`.

    `exit_energy_fn` must be monotone decreasing in bank (more bank = less
    lift-up = deeper = more energy shed). Crash returns a very negative value,
    skip-out a positive one -- both keep the bracket valid.
    """
    e_lo = exit_energy_fn(lo)
    e_hi = exit_energy_fn(hi)
    if not (e_lo > target_mj > e_hi):
        raise ValueError(f"target {target_mj:.3f} MJ/kg not bracketed by banks [{lo:.1f}, {hi:.1f}] deg (exit energies [{e_lo:.3f}, {e_hi:.3f}])")
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        e_mid = exit_energy_fn(mid)
        if abs(e_mid - target_mj) <= tol_mj:
            return mid
        if e_mid > target_mj:
            lo = mid
        else:
            hi = mid
    return mid


def _mission_ref_path(toml_path: Path) -> Path:
    """training_output/<mission>/ref_trajectory.dat, mission derived from the
    base chain (same convention as train.py)."""
    from aerocapture.training.toml_utils import find_mission_name  # noqa: PLC0415

    mission = find_mission_name(toml_path) or toml_path.stem
    return Path("training_output") / mission / "ref_trajectory.dat"


def main(argv: list[str] | None = None) -> int:
    import aerocapture_rs  # noqa: PLC0415  # soft spot: CLI-only, keeps module importable without PyO3

    from aerocapture.training.reference import nominal_flight_overrides, ref_trajectory_array  # noqa: PLC0415  # leaf module, not the train.py re-export shim
    from aerocapture.training.toml_utils import load_toml_with_bases  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", required=True, help="piecewise-capable training TOML (e.g. msr_aller_pc_ref_train.toml)")
    parser.add_argument(
        "--overshoot-mj",
        type=float,
        default=0.7,
        help="capture margin past the target energy (0.7 matches the legacy reference and trained FTC to legacy parity; "
        "two identical-config FTC runs on the same 0.7 table spanned 2.78e6-4.08e6 val RMS, so sweep conclusions need seed repeats)",
    )
    parser.add_argument("--bank-lo", type=float, default=20.0)
    parser.add_argument("--bank-hi", type=float, default=110.0)
    parser.add_argument("--tol-mj", type=float, default=0.02)
    parser.add_argument("--output", default=None, help="override the output path (default: training_output/<mission>/ref_trajectory.dat)")
    args = parser.parse_args(argv)

    resolved = load_toml_with_bases(Path(args.toml))
    mc = resolved.get("monte_carlo", {})
    idx = aerocapture_rs.final_record_indices()
    target_e = target_orbit_energy_mj(resolved)
    aim = target_e - args.overshoot_mj
    print(f"target orbit energy {target_e:.3f} MJ/kg, aiming for exit at {aim:.3f} (overshoot {args.overshoot_mj:.2f})")

    def fly(bank_deg: float, with_traj: bool = False):  # type: ignore[no-untyped-def]
        ov = nominal_flight_overrides({}, "piecewise_constant", mc)
        ov["guidance.piecewise_constant.n_segments"] = 1
        ov["guidance.piecewise_constant.bank_angle_0"] = bank_deg
        return aerocapture_rs.run_batch(toml_path=args.toml, overrides_list=[ov], include_trajectories=with_traj, sim_timeout_secs=120)

    def exit_energy(bank_deg: float) -> float:
        rec = np.asarray(fly(bank_deg).final_records)[0]
        ifinal = int(rec[idx["ifinal"]])
        e = float(rec[idx["energy_mjkg"]])
        if ifinal in (1, 4):  # crash / pending crash
            return CRASH_ENERGY_MJ
        print(f"  bank {bank_deg:7.3f} deg -> exit energy {e:8.3f} MJ/kg (ifinal {ifinal})")
        return e

    bank = bisect_bank_for_exit_energy(exit_energy, target_mj=aim, lo=args.bank_lo, hi=args.bank_hi, tol_mj=args.tol_mj)

    batch = fly(bank, with_traj=True)
    nom = np.asarray(batch.trajectories[0])
    cos_bank = np.full(nom.shape[0], np.cos(np.radians(bank)))
    ref = ref_trajectory_array(nom, cos_bank=cos_bank)
    out = Path(args.output) if args.output else _mission_ref_path(Path(args.toml))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(out), ref, fmt="  %.16E")
    print(f"reference bank {bank:.3f} deg (cos {np.cos(np.radians(bank)):.4f})")
    print(f"written {out} ({len(ref)} pts, E {ref[:, 0].max():.3f}..{ref[:, 0].min():.3f} MJ/kg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
