"""The piecewise ref writer must emit the Rust loader's file contract:
column 0 in MJ/kg (the loader multiplies by 1e6 itself — writing J/kg shifts
the energy axis 1e6x and collapses every runtime interpolation query)."""

from __future__ import annotations

import numpy as np
from aerocapture.training.train import ref_trajectory_array


def _nom_traj(n: int = 5) -> np.ndarray:
    # 17-column trajectory matrix; only the columns the writer reads are filled:
    # 3=vel_m_s, 4=fpa_deg, 7=time_s, 8=energy_mj_kg, 9=pdyn_kpa,
    # 10=bank_angle_deg, 11=inclination_deg
    traj = np.zeros((n, 17))
    traj[:, 3] = 5687.0
    traj[:, 4] = -10.8
    traj[:, 7] = np.arange(n, dtype=float)
    traj[:, 8] = np.linspace(4.9, -5.3, n)  # MJ/kg
    traj[:, 9] = np.linspace(0.001, 1.6, n)  # kPa
    traj[:, 10] = 64.77
    traj[:, 11] = 50.0
    return traj


def test_energy_column_is_mj_per_kg() -> None:
    traj = _nom_traj()
    ref = ref_trajectory_array(traj)
    assert ref.shape == (5, 7)
    np.testing.assert_allclose(ref[:, 0], traj[:, 8])  # MJ/kg, NOT * 1e6
    assert np.abs(ref[:, 0]).max() < 100.0


def test_commanded_cos_bank_mirrors_rust_segment_lookup() -> None:
    # Mirrors `segment_bank_angle` in piecewise_constant.rs: segment 0 at the
    # highest energy, frac = (e_max - E)/(e_max - e_min), floor, clamp.
    from aerocapture.training.train import piecewise_commanded_cos_bank

    banks = [60.0, 50.0, 40.0, 30.0, 20.0, -20.0, -30.0, -40.0, -50.0, -60.0]
    e = np.array([4.5, 5.0, -6.0, -7.0, 10.0, -0.501])  # MJ/kg, e_range [-6, 5]
    cos = piecewise_commanded_cos_bank(e, banks, energy_min_mj=-6.0, energy_max_mj=5.0)
    # 4.5 -> frac 0.0454*10 -> seg 0 (60 deg); 5.0 -> seg 0; -6.0/-7.0/below -> seg 9 (-60);
    # 10.0 above e_max clamps to seg 0; -0.501 -> frac 0.50009*10 -> seg 5 (-20 deg)
    expected = np.cos(np.radians([60.0, 60.0, -60.0, -60.0, 60.0, -20.0]))
    np.testing.assert_allclose(cos, expected)


def test_commanded_cos_bank_single_segment_is_constant() -> None:
    from aerocapture.training.train import piecewise_commanded_cos_bank

    e = np.linspace(4.9, -5.4, 50)
    cos = piecewise_commanded_cos_bank(e, [64.77], energy_min_mj=-7.0, energy_max_mj=5.0)
    np.testing.assert_allclose(cos, np.cos(np.radians(64.77)))


def test_ref_array_accepts_commanded_cos_override() -> None:
    traj = _nom_traj()
    override = np.full(5, 0.5)
    ref = ref_trajectory_array(traj, cos_bank=override)
    np.testing.assert_allclose(ref[:, 6], 0.5)  # not the realized cos from col 10


def test_nominal_flight_disables_all_dispersion_domains() -> None:
    # The reference nominal must be the true undispersed trajectory. The old
    # inline override list named only 5 domains, so the reference shipped with
    # vehicle/pilot/nav_filter dispersions, a wind draw, and OU density noise.
    from aerocapture.training.train import nominal_flight_overrides

    mc = {
        "seed": 42,
        "sampling": "lhs",
        "initial_state": {"level": "medium"},
        "wind": {"level": "low"},
        "density_perturbation": {"level": "low"},
    }
    ov = nominal_flight_overrides({"bank_angle_0": -103.0, "shaping.max_bank_acceleration": 6.4}, "piecewise_constant", mc)
    for domain in (
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
    ):
        assert ov[f"monte_carlo.{domain}.level"] == "off", domain
    assert ov["guidance.piecewise_constant.bank_angle_0"] == -103.0
    assert ov["guidance.command_shaping.max_bank_acceleration"] == 6.4
    assert ov["guidance.command_shaping.enabled"] is True
    assert ov["simulation.n_sims"] == 1
    assert ov["guidance.type"] == "piecewise_constant"


def test_nominal_flight_covers_future_config_domains() -> None:
    from aerocapture.training.train import nominal_flight_overrides

    ov = nominal_flight_overrides({}, "piecewise_constant", {"some_new_domain": {"level": "high"}, "seed": 7})
    assert ov["monte_carlo.some_new_domain.level"] == "off"
    assert "monte_carlo.seed.level" not in ov


def test_pc_ref_config_resolves_single_segment_reference_only() -> None:
    from pathlib import Path

    from aerocapture.training.toml_utils import load_toml_with_bases

    repo = Path(__file__).resolve().parents[1]
    t = load_toml_with_bases(repo / "configs/training/msr_aller_pc_ref_train.toml")
    pc = t["guidance"]["piecewise_constant"]
    assert pc["n_segments"] == 1
    assert pc["reference_only"] is True
    assert t["guidance"]["type"] == "piecewise_constant"


def test_remaining_columns_match_contract() -> None:
    traj = _nom_traj()
    ref = ref_trajectory_array(traj)
    np.testing.assert_allclose(ref[:, 1], traj[:, 9] * 1e3)  # kPa -> Pa
    np.testing.assert_allclose(ref[:, 2], traj[:, 3] * np.sin(np.radians(traj[:, 4])))  # radial vel
    np.testing.assert_allclose(ref[:, 4], np.radians(traj[:, 11]))  # inclination rad
    np.testing.assert_allclose(ref[:, 5], traj[:, 7])  # time s
    np.testing.assert_allclose(ref[:, 6], np.cos(np.radians(traj[:, 10])))  # cos(bank)
