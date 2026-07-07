"""Reference-trajectory generation helpers (leaf module — no train.py/problem.py imports).

Shared by train.py (end-of-piecewise ref writer, joint-ref deploy artifact),
problem.py (per-individual reference tables for the `ref_bank` chromosome gene),
and make_reference.py (standalone target-energy-matched generator).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

# Every MC dispersion domain in `dispersions.rs`. The nominal reference flight
# must turn ALL of them off — a stale subset here ships a dispersed reference
# (wind draw + OU density noise contaminated the first piecewise-generated ref).
_MC_DISPERSION_DOMAINS = (
    "initial_state",
    "atmosphere",
    "aerodynamics",
    "navigation",
    "mass",
    "vehicle",
    "pilot",
    "nav_filter",
    "wind",
    "density_perturbation",
)


def nominal_flight_overrides(best_params: dict[str, float], scheme: str, mc_config: dict) -> dict[str, object]:
    """Overrides flying `best_params` on the true undispersed nominal scenario.

    Disables every known dispersion domain plus any domain declared in the
    config's [monte_carlo] section (future-proofing — domains are dict-valued
    entries; scalar keys like seed/sampling are left alone).
    """
    from aerocapture.training.param_spaces import route_param_path  # noqa: PLC0415

    overrides: dict[str, object] = {}
    for key, value in best_params.items():
        coerced: object = int(round(value)) if key == "lateral.max_reversals" else value
        overrides[route_param_path(key, scheme)] = coerced
        if key.startswith("shaping."):
            overrides["guidance.command_shaping.enabled"] = True
    overrides["guidance.type"] = scheme
    overrides["simulation.n_sims"] = 1
    config_domains = (k for k, v in mc_config.items() if isinstance(v, dict))
    for domain in {*_MC_DISPERSION_DOMAINS, *config_domains}:
        overrides[f"monte_carlo.{domain}.level"] = "off"
    return overrides


def ref_trajectory_array(nom_traj: npt.NDArray[np.float64], cos_bank: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """7-column ref_trajectory.dat contents from a (N, 17) trajectory matrix.

    Column contract of the Rust loader (`ReferenceTrajectory::load`): energy in
    MJ/kg (the LOADER converts to J/kg — writing J/kg here shifts the energy
    axis 1e6x and collapses every runtime interpolation query into the table's
    tail), pdyn in Pa, radial velocity m/s (duplicated into the unused
    altitude_rate column), inclination rad, time s, cos(bank).

    `cos_bank` overrides the realized-bank column: the realized bank carries
    shaper sweeps through 0 deg at every reversal/transition, which a tracker
    interpolates as feedforward whipsaw — pass the COMMANDED profile instead.
    """
    vel = nom_traj[:, 3]
    fpa_rad = np.radians(nom_traj[:, 4])
    radial_vel = vel * np.sin(fpa_rad)
    energy_mj = nom_traj[:, 8]
    pdyn_pa = nom_traj[:, 9] * 1e3
    incl_rad = np.radians(nom_traj[:, 11])
    time_s = nom_traj[:, 7]
    if cos_bank is None:
        cos_bank = np.cos(np.radians(nom_traj[:, 10]))
    return np.column_stack([energy_mj, pdyn_pa, radial_vel, radial_vel, incl_rad, time_s, cos_bank])


def piecewise_commanded_cos_bank(
    energy_mj: npt.NDArray[np.float64],
    bank_angles_deg: list[float],
    *,
    energy_min_mj: float,
    energy_max_mj: float,
) -> npt.NDArray[np.float64]:
    """cos of the COMMANDED piecewise bank at each energy — mirrors the Rust
    `segment_bank_angle` lookup (piecewise_constant.rs): segment 0 at the
    highest energy, frac = (e_max - E)/(e_max - e_min), floor, clamp."""
    n = len(bank_angles_deg)
    frac = (energy_max_mj - np.asarray(energy_mj)) / (energy_max_mj - energy_min_mj)
    seg = np.clip(np.floor(frac * n).astype(int), 0, n - 1)
    result: npt.NDArray[np.float64] = np.cos(np.radians(np.asarray(bank_angles_deg)[seg]))
    return result


def generate_constant_bank_tables(
    toml_path: str,
    banks_deg: list[float],
    mc_config: dict,
    out_dir: Path,
    sim_timeout_secs: float | None = None,
) -> list[Path]:
    """Fly one undispersed 1-segment nominal per bank (a single batched run_batch
    call) and write each as `ref_bank_{i:04d}.dat` in `out_dir` (slot files —
    successive calls overwrite). A nominal that produces no trajectory (e.g. an
    immediate skip-out) writes an empty table: the Rust loader yields 0 points
    and guidance falls back to the default bank, so the individual just scores
    badly instead of crashing the batch.
    """
    import aerocapture_rs  # type: ignore[import-not-found, import-untyped]  # noqa: PLC0415

    overrides_list: list[dict[str, object]] = []
    for bank in banks_deg:
        ov = nominal_flight_overrides({}, "piecewise_constant", mc_config)
        ov["guidance.piecewise_constant.n_segments"] = 1
        ov["guidance.piecewise_constant.bank_angle_0"] = float(bank)
        overrides_list.append(ov)
    batch = aerocapture_rs.run_batch(
        toml_path=str(toml_path),
        overrides_list=overrides_list,
        include_trajectories=True,
        sim_timeout_secs=sim_timeout_secs,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, bank in enumerate(banks_deg):
        path = out_dir / f"ref_bank_{i:04d}.dat"
        traj = np.asarray(batch.trajectories[i])
        if traj.ndim == 2 and traj.shape[0] > 0:
            cos = np.full(traj.shape[0], np.cos(np.radians(float(bank))))
            np.savetxt(str(path), ref_trajectory_array(traj, cos_bank=cos), fmt="  %.16E")
        else:
            path.write_text("")
        paths.append(path)
    return paths
