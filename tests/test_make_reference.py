"""Target-energy-matched constant-bank reference generator (variant C).

The reference bank is NOT a GA product: open-loop-optimal profiles under-capture
relative to the target orbit and leave the reference table short of the energies
the trackers must fly through. Instead, bisect the bank angle until the
undispersed nominal's exit energy hits the target orbit energy minus an
overshoot margin (the legacy msr_aller.dat overshoots by ~0.7 MJ/kg)."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.make_reference import bisect_bank_for_exit_energy, target_orbit_energy_mj


def _toml_fixture() -> dict:
    # [flight.target_orbit] is in km, as in configs/missions/mars.toml
    return {
        "planet": {"mu": 4.282829e13, "equatorial_radius": 3_393_940.0},
        "flight": {"target_orbit": {"apoapsis": 500.130, "periapsis": 11.233, "semi_major_axis": 3649.622}},
    }


def test_target_orbit_energy_matches_visviva() -> None:
    e = target_orbit_energy_mj(_toml_fixture())
    assert e == pytest.approx(-4.282829e13 / (2.0 * 3_649_622.0) / 1e6)
    assert -6.5 < e < -5.0  # MSR ballpark


def test_target_orbit_energy_falls_back_to_apo_peri() -> None:
    t = _toml_fixture()
    del t["flight"]["target_orbit"]["semi_major_axis"]
    sma = (500.130 + 11.233) / 2.0 * 1e3 + 3_393_940.0
    assert target_orbit_energy_mj(t) == pytest.approx(-4.282829e13 / (2.0 * sma) / 1e6)


def test_bisect_finds_bank_on_monotone_curve() -> None:
    # Synthetic monotone-decreasing exit energy: E(theta) = 2 - 0.1 * theta
    bank = bisect_bank_for_exit_energy(lambda b: 2.0 - 0.1 * b, target_mj=-5.0, lo=20.0, hi=110.0, tol_mj=0.001)
    assert bank == pytest.approx(70.0, abs=0.05)


def test_bisect_rejects_non_bracketing_range() -> None:
    with pytest.raises(ValueError, match="bracket"):
        bisect_bank_for_exit_energy(lambda b: 2.0 - 0.001 * b, target_mj=-5.0, lo=20.0, hi=110.0)


def test_bisect_tolerates_noisy_extremes() -> None:
    # Crash side returns a very negative sentinel, skip-out side positive energy;
    # the root must still be found between them.
    def e(b: float) -> float:
        if b > 100.0:
            return -1e9  # crash sentinel
        if b < 25.0:
            return 3.0  # hyperbolic skip-out
        return float(np.interp(b, [25.0, 100.0], [1.0, -8.0]))

    bank = bisect_bank_for_exit_energy(e, target_mj=-5.0, lo=20.0, hi=110.0, tol_mj=0.001)
    assert e(bank) == pytest.approx(-5.0, abs=0.01)
