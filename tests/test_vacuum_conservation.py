"""Vacuum two-body conservation check (paper R4/R5, reviewer R1 major 7).

With a near-zero-density atmosphere, zonal harmonics zeroed, and planet
rotation zeroed, the flown trajectory is pure two-body motion: specific
orbital energy and specific angular momentum |h| = r v cos(gamma) must be
conserved by the fixed-step Gill RK4 integrator to tight relative tolerance.
This is a physical-validation check independent of the legacy regression
(bit-identity proves code equivalence, not correct dynamics).
"""

from pathlib import Path

import numpy as np
import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")

REPO = Path(__file__).resolve().parents[1]
TOML = REPO / "configs/test/test_ref_orig.toml"
VACUUM = REPO / "tests/reference_data/vacuum_atmosphere.dat"

_TC_ALT_KM, _TC_VEL, _TC_FPA_DEG, _TC_ENERGY_MJ = 0, 3, 4, 8


def test_vacuum_two_body_conserves_energy_and_angular_momentum() -> None:
    from aerocapture.training.reference import _MC_DISPERSION_DOMAINS

    cfg = aerocapture_rs.load_config(str(TOML.resolve()))
    req = float(cfg["planet"]["equatorial_radius"])
    overrides = {
        "simulation.n_sims": 1,
        "monte_carlo.seed": 1,
        "data.atmosphere": str(VACUUM.resolve()),
        "planet.j2": 0.0,
        "planet.j3": 0.0,
        "planet.j4": 0.0,
        "planet.omega": 0.0,
        # Spherical planet: the trajectory's altitude column is measured from the
        # latitude-dependent ellipsoid, which would alias into an apparent |h|
        # drift when reconstructing r = R + alt on an oblate planet.
        "planet.polar_radius": req,
        **{f"monte_carlo.{dom}.level": "off" for dom in _MC_DISPERSION_DOMAINS},
    }
    res = aerocapture_rs.run_mc(toml_path=str(TOML.resolve()), overrides=overrides, include_trajectories=True)
    t = np.asarray(res.trajectories[0])
    assert len(t) > 50, f"expected a multi-sample trajectory, got {len(t)} rows"

    # Observed drift: ~2e-14 (energy), ~3e-15 (|h|) over 371 samples; the 1e-11
    # gates keep three orders of margin while still catching any real
    # integrator or gravity regression.
    energy = t[:, _TC_ENERGY_MJ]
    rel_e = np.abs(energy - energy[0]).max() / abs(energy[0])
    assert rel_e < 1e-11, f"vacuum energy drift {rel_e:.2e} exceeds 1e-11 relative"

    r = req + t[:, _TC_ALT_KM] * 1e3
    h = r * t[:, _TC_VEL] * np.cos(np.radians(t[:, _TC_FPA_DEG]))
    rel_h = np.abs(h - h[0]).max() / abs(h[0])
    assert rel_h < 1e-11, f"vacuum |h| drift {rel_h:.2e} exceeds 1e-11 relative"
