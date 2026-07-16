"""CPAG C0 prototype: dynamics-model unit tests (cross-checked against the Rust sim)."""

import numpy as np
import pytest
from aerocapture.cpag.model import (
    IGAMMA,
    IPSI,
    IQ,
    IR,
    ISIGMA,
    IV,
    Planet,
    apoapsis_radius,
    cos_inclination,
    entry_state,
    eom,
    eps_apoapsis,
    geodetic_altitude,
    gravity,
    inertial_energy,
    load_model,
    path_quantities,
    rk4_step,
)

TOML = "configs/nominal/msr_aller_ftc_nominal.toml"


@pytest.fixture(scope="module")
def model():  # type: ignore[no-untyped-def]
    return load_model(TOML)


@pytest.fixture(scope="module")
def mars(model):  # type: ignore[no-untyped-def]
    return model.planet


class TestGravity:
    def test_matches_potential_gradient(self, mars: Planet) -> None:
        """Port of the Rust oracle: analytic (gravtl, gravtr) must equal the
        numerical gradient of the J2-J4 geopotential."""
        re, mu = mars.req, mars.mu
        j2, j3, j4 = mars.j2, mars.j3, mars.j4

        def potential(r: float, lat: float) -> float:
            x = np.sin(lat)
            p2 = (3 * x * x - 1) / 2
            p3 = (5 * x**3 - 3 * x) / 2
            p4 = (35 * x**4 - 30 * x * x + 3) / 8
            return float((mu / r) * (1 - j2 * (re / r) ** 2 * p2 - j3 * (re / r) ** 3 * p3 - j4 * (re / r) ** 4 * p4))

        for r_mult, lat in [(1.05, 0.3), (1.2, -0.7), (1.5, 1.1)]:
            r = re * r_mult
            gravtl, gravtr = gravity(np.asarray(r), np.asarray(lat), mars)
            hr, hl = r * 1e-6, 1e-6
            dudr = (potential(r + hr, lat) - potential(r - hr, lat)) / (2 * hr)
            dudlat = (potential(r, lat + hl) - potential(r, lat - hl)) / (2 * hl)
            assert gravtr == pytest.approx(-dudr, rel=1e-5)
            assert gravtl == pytest.approx(-dudlat / r, rel=1e-5)

    def test_mars_surface_gravity_ballpark(self, mars: Planet) -> None:
        _, gravtr = gravity(np.asarray(mars.req), np.asarray(0.0), mars)
        assert gravtr == pytest.approx(3.72, rel=0.05)


class TestGeodetic:
    def test_round_trip_with_closed_form(self, mars: Planet) -> None:
        """geodetic->geocentric closed form -> ported iterative forward recovers
        the altitude within the Rust loop's 0.01 m stopping tolerance."""
        from aerocapture.cpag.studies import geodetic_to_geocentric

        for alt_km, lat_deg in [(50.0, 0.0), (80.0, 30.6), (120.0, -45.0), (10.0, 60.0)]:
            r, lat_gc = geodetic_to_geocentric(alt_km * 1e3, np.radians(lat_deg), mars.req, mars.rpol)
            alt_rt = geodetic_altitude(np.asarray(r), np.asarray(lat_gc), mars)
            assert abs(float(alt_rt) - alt_km * 1e3) < 0.02

    def test_equator_matches_spherical(self, mars: Planet) -> None:
        r = mars.req + 60e3
        alt = geodetic_altitude(np.asarray(r), np.asarray(0.0), mars)
        assert float(alt) == pytest.approx(60e3, abs=1.0)


class TestOrbitalQuantities:
    def test_circular_equatorial_apoapsis(self, mars: Planet) -> None:
        """v_rel = v_circular - omega*r eastward at the equator -> apo == r."""
        r = mars.req + 300e3
        v_inertial = np.sqrt(mars.mu / r)
        x = np.zeros(8)
        x[IR], x[IV], x[IGAMMA], x[IPSI] = r, v_inertial - mars.omega * r, 0.0, np.pi / 2
        assert float(apoapsis_radius(x, mars)) == pytest.approx(r, rel=1e-6)
        assert float(cos_inclination(x, mars)) == pytest.approx(1.0, abs=1e-9)

    def test_eps_zero_iff_apoapsis_on_target(self, model) -> None:  # type: ignore[no-untyped-def]
        """eps root coincides with apoapsis == target (Keplerian identity)."""
        mars = model.planet
        r = mars.req + 131e3
        x = np.zeros(8)
        x[IR], x[IGAMMA], x[IPSI] = r, np.radians(3.4), np.radians(60.0)
        lo, hi = 3000.0, 4600.0
        for _ in range(60):  # bisect eps(v) = 0
            x[IV] = 0.5 * (lo + hi)
            if float(eps_apoapsis(x, model)) > 0.0:
                hi = x[IV]
            else:
                lo = x[IV]
        assert float(apoapsis_radius(x, mars)) == pytest.approx(model.target_apoapsis_radius, abs=200.0)

    def test_eps_smooth_through_escape(self, model) -> None:  # type: ignore[no-untyped-def]
        """eps stays finite and monotone in v across the parabolic boundary —
        the paper's fix for the apoapsis-Jacobian singularity at v_esc."""
        mars = model.planet
        r = mars.req + 131e3
        v_esc = float(np.sqrt(2 * mars.mu / r))
        x = np.zeros(8)
        x[IR], x[IGAMMA], x[IPSI] = r, np.radians(3.0), np.radians(45.0)
        vals = []
        for dv in np.linspace(-200.0, 200.0, 41):
            x[IV] = v_esc + dv
            e = float(eps_apoapsis(x, model))
            assert np.isfinite(e)
            vals.append(e)
        diffs = np.diff(vals)
        assert np.all(diffs > 0.0)

    def test_hyperbolic_energy_positive(self, mars: Planet) -> None:
        x = np.zeros(8)
        x[IR], x[IV], x[IGAMMA], x[IPSI] = mars.req + 130e3, 5687.0, np.radians(-10.8), np.radians(38.0)
        assert float(inertial_energy(x, mars)) > 0.0
        assert float(apoapsis_radius(x, mars)) == pytest.approx(1e9)


@pytest.fixture(scope="module")
def rust_batch():  # type: ignore[no-untyped-def]
    aerocapture_rs = pytest.importorskip("aerocapture_rs")
    from pathlib import Path

    from aerocapture.training.reference import nominal_flight_overrides
    from aerocapture.training.toml_utils import load_toml_with_bases

    mc = load_toml_with_bases(Path(TOML)).get("monte_carlo", {})
    ov = nominal_flight_overrides({}, "piecewise_constant", mc)
    ov["guidance.piecewise_constant.n_segments"] = 1
    ov["guidance.piecewise_constant.bank_angle_0"] = 64.77026
    ov["monte_carlo.seed"] = 42
    return aerocapture_rs.run_batch(toml_path=TOML, overrides_list=[ov], include_trajectories=True, sim_timeout_secs=120.0)


@pytest.fixture(scope="module")
def rust_traj(rust_batch):  # type: ignore[no-untyped-def]
    return np.asarray(rust_batch.trajectories[0])


@pytest.fixture(scope="module")
def rust_final_apo_km(rust_batch):  # type: ignore[no-untyped-def]
    import aerocapture_rs

    idx = aerocapture_rs.final_record_indices()
    return float(np.asarray(rust_batch.final_records)[0, idx["apoapsis_alt_km"]])


class TestDynamicsAgainstRustSim:
    """Propagate a constant-bank profile and compare against the Rust simulator
    flying the same profile (piecewise_constant n_segments=1, dispersions off,
    perfect pilot). The prototype includes the lateral lift term, so the whole
    signed-bank EOM chain is validated: same atmosphere, gravity, rotation."""

    def test_exit_state_matches(self, model, rust_traj) -> None:  # type: ignore[no-untyped-def]
        from aerocapture.cpag.scp import ScpConfig, shoot_profile

        x0 = entry_state(TOML)
        cfg = ScpConfig(seg_dt=8.0, n_sub=8, horizon_max=800.0)  # 1 s RK4 steps like the sim
        shoot = shoot_profile(x0, np.zeros(100), model, cfg)
        assert shoot.event == "exit"
        vel_rust = float(rust_traj[-1, 3])
        t_rust = float(rust_traj[-1, 7])
        assert shoot.x_nodes[-1, IV] == pytest.approx(vel_rust, abs=6.0)
        # ~11 s later than the plant: the prototype checks exit on SPHERICAL
        # altitude (FNPAG predictor convention) while plant events use geodetic
        # (~4.3 km lower at the exit latitude). Drag-free region, so the exit
        # APOAPSIS — the quantity guidance targets — is insensitive to it.
        assert shoot.t_final == pytest.approx(t_rust, abs=20.0)
        # Heat load within 2% of the Rust cumulative value
        assert shoot.x_nodes[-1, IQ] == pytest.approx(float(rust_traj[-1, 15]), rel=0.02)

    def test_exit_apoapsis_matches_final_record(self, model, rust_final_apo_km) -> None:  # type: ignore[no-untyped-def]
        from aerocapture.cpag.scp import ScpConfig, shoot_profile

        x0 = entry_state(TOML)
        cfg = ScpConfig(seg_dt=8.0, n_sub=8, horizon_max=800.0)
        shoot = shoot_profile(x0, np.zeros(100), model, cfg)
        apo_alt_km = (float(apoapsis_radius(shoot.x_nodes[-1], model.planet)) - model.planet.req) / 1e3
        assert apo_alt_km == pytest.approx(rust_final_apo_km, abs=15.0)

    def test_path_peaks_match(self, model, rust_traj) -> None:  # type: ignore[no-untyped-def]
        from aerocapture.cpag.scp import ScpConfig, shoot_profile

        x0 = entry_state(TOML)
        cfg = ScpConfig(seg_dt=8.0, n_sub=8, horizon_max=800.0)
        shoot = shoot_profile(x0, np.zeros(100), model, cfg)
        hf, gl, _ = path_quantities(shoot.x_nodes, model)
        assert float(np.max(hf)) / 1e3 == pytest.approx(float(np.max(rust_traj[:, 6])), rel=0.02)
        assert float(np.max(gl)) == pytest.approx(float(np.max(rust_traj[:, 12])), rel=0.02)


class TestEom:
    def test_vacuum_energy_conserved(self, model) -> None:  # type: ignore[no-untyped-def]
        """Inertial point-mass energy is conserved out of the atmosphere on a
        J2-free planet (rotation terms must be exactly the frame bookkeeping,
        not spurious forces). J2-J4 stay zeroed: v^2/2 - mu/r is not the
        conserved quantity under zonal harmonics."""
        from dataclasses import replace

        spherical = replace(model, planet=replace(model.planet, j2=0.0, j3=0.0, j4=0.0))
        mars = spherical.planet
        x = np.zeros(8)
        x[IR], x[IV], x[IGAMMA], x[IPSI] = mars.req + 400e3, 3600.0, np.radians(10.0), np.radians(60.0)
        x[2] = np.radians(20.0)  # latitude
        e0 = float(inertial_energy(x, mars))
        for _ in range(600):
            x = rk4_step(x, np.asarray(0.0), 1.0, spherical)
        assert float(inertial_energy(x, mars)) == pytest.approx(e0, rel=1e-9)

    def test_bank_rate_integrates_sigma(self, model) -> None:  # type: ignore[no-untyped-def]
        x = entry_state(TOML)
        rate = np.radians(5.0)
        x1 = rk4_step(x, np.asarray(rate), 2.0, model)
        assert float(x1[ISIGMA] - x[ISIGMA]) == pytest.approx(2.0 * rate, rel=1e-12)

    def test_lift_up_raises_gamma_rate(self, model) -> None:  # type: ignore[no-untyped-def]
        x = entry_state(TOML)
        x[IR] = model.planet.req + 45e3  # dense atmosphere
        x_up, x_down = x.copy(), x.copy()
        x_up[ISIGMA], x_down[ISIGMA] = 0.0, np.pi
        d_up = eom(x_up, np.asarray(0.0), model)
        d_down = eom(x_down, np.asarray(0.0), model)
        assert float(d_up[IGAMMA]) > float(d_down[IGAMMA])
