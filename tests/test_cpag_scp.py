"""CPAG C0 prototype: SCP subproblem and replan tests."""

import numpy as np
import pytest
from aerocapture.cpag.model import N_STATES, entry_state, load_model
from aerocapture.cpag.scp import (
    ScpConfig,
    build_qp,
    canonicalize,
    linearize_segments,
    scp_replan,
    shoot_profile,
    solve_qp_clarabel,
    solve_qp_osqp,
)

TOML = "configs/nominal/msr_aller_ftc_nominal.toml"


@pytest.fixture(scope="module")
def model():  # type: ignore[no-untyped-def]
    return load_model(TOML)


@pytest.fixture(scope="module")
def small_qp(model):  # type: ignore[no-untyped-def]
    """A genuine (small) subproblem instance from a shortened entry replan."""
    cfg = ScpConfig(seg_dt=16.0, n_sub=8, horizon_max=600.0, trust_mode="box")
    x0 = entry_state(TOML)
    shoot = shoot_profile(x0, np.zeros(int(np.ceil(cfg.horizon_max / cfg.seg_dt))), model, cfg)
    a_mats, b_mats = linearize_segments(shoot, model, cfg)
    return build_qp(shoot, a_mats, b_mats, model, cfg, trust=2.0)


class TestQpConstruction:
    def test_linearization_predicts_nonlinear_step(self, model) -> None:  # type: ignore[no-untyped-def]
        """A_k dx + B_k du must predict the perturbed segment map to first order."""
        cfg = ScpConfig(seg_dt=8.0)
        x0 = entry_state(TOML)
        shoot = shoot_profile(x0, np.zeros(150), model, cfg)
        a_mats, b_mats = linearize_segments(shoot, model, cfg)
        from aerocapture.cpag.model import rk4_step
        from aerocapture.cpag.scp import _STATE_SCALE

        k = 5
        dx = 1e-3 * _STATE_SCALE
        du = 1e-3
        x_pert = shoot.x_nodes[k] + dx
        u_pert = shoot.u_nodes[k] + du
        x_next = x_pert.copy()
        for _ in range(cfg.n_sub):
            x_next = rk4_step(x_next, np.asarray(u_pert), float(shoot.dts[k]) / cfg.n_sub, model)
        pred = shoot.x_nodes[k + 1] + a_mats[k] @ dx + b_mats[k] * du
        err = np.abs(x_next - pred) / _STATE_SCALE
        assert float(np.max(err)) < 1e-4

    def test_clarabel_matches_cvxpy_reference(self, small_qp) -> None:  # type: ignore[no-untyped-def]
        """Hand-canonicalized Clarabel solve == cvxpy's independent canonicalization."""
        cvxpy = pytest.importorskip("cvxpy")
        import scipy.sparse as sp

        qp = small_qp
        n = qp.p_mat.shape[0]
        z = cvxpy.Variable(n)
        objective = 0.5 * cvxpy.quad_form(z, cvxpy.psd_wrap(sp.csc_matrix(qp.p_mat))) + qp.q @ z
        constraints = [qp.a_eq @ z == qp.b_eq, qp.a_in @ z <= qp.b_in]
        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        ref_obj = prob.solve(solver=cvxpy.CLARABEL)

        z_mine, info = solve_qp_clarabel(qp)
        my_obj = 0.5 * float(z_mine @ (qp.p_mat @ z_mine)) + float(qp.q @ z_mine)
        assert info.status.lower().startswith(("solved", "almostsolved"))
        assert my_obj == pytest.approx(float(ref_obj), rel=1e-5, abs=1e-6)

    def test_osqp_agrees_with_clarabel(self, small_qp) -> None:  # type: ignore[no-untyped-def]
        z_cl, _ = solve_qp_clarabel(small_qp)
        z_os, info = solve_qp_osqp(small_qp)
        obj_cl = 0.5 * float(z_cl @ (small_qp.p_mat @ z_cl)) + float(small_qp.q @ z_cl)
        obj_os = 0.5 * float(z_os @ (small_qp.p_mat @ z_os)) + float(small_qp.q @ z_os)
        assert obj_os == pytest.approx(obj_cl, rel=1e-3, abs=1e-4)

    def test_canonicalize_row_counts(self, small_qp) -> None:  # type: ignore[no-untyped-def]
        _, _, a_all, b_all, n_eq, soc_dims = canonicalize(small_qp)
        assert a_all.shape[0] == b_all.shape[0]
        assert n_eq == small_qp.a_eq.shape[0]
        assert soc_dims == []  # box mode

    def test_ptr_mode_builds_soc_blocks(self, model) -> None:  # type: ignore[no-untyped-def]
        cfg = ScpConfig(seg_dt=16.0, n_sub=8, horizon_max=600.0, trust_mode="ptr")
        x0 = entry_state(TOML)
        shoot = shoot_profile(x0, np.zeros(int(np.ceil(cfg.horizon_max / cfg.seg_dt))), model, cfg)
        a_mats, b_mats = linearize_segments(shoot, model, cfg)
        qp = build_qp(shoot, a_mats, b_mats, model, cfg, trust=2.0)
        m_seg = len(shoot.dts)
        assert len(qp.soc_blocks) == m_seg + 1
        assert qp.soc_blocks[0][0].shape[0] == 1 + N_STATES + 1
        assert qp.soc_blocks[-1][0].shape[0] == 1 + N_STATES  # terminal node: no du
        with pytest.raises(ValueError, match="SOC"):
            solve_qp_osqp(qp)


class TestReplan:
    @pytest.mark.slow
    def test_nominal_entry_converges_feasible(self, model) -> None:  # type: ignore[no-untyped-def]
        cfg = ScpConfig(max_iters=30, horizon_max=1200.0)
        r = scp_replan(entry_state(TOML), model, cfg)
        assert r.converged and r.feasible
        assert abs(r.apo_error_m) <= cfg.tol_apo_m
        assert r.inc_error_deg <= cfg.tol_inc_deg
        assert all(r.path_peaks[t] <= 1.01 for t in cfg.path_constraints)
        assert r.heat_load_frac <= 1.01
        assert r.event == "exit"

    @pytest.mark.slow
    def test_recoverable_crash_reference_converges(self, model) -> None:  # type: ignore[no-untyped-def]
        """From a 90-deg-held state deep in the atmosphere (crashing reference,
        recoverable per the lift-up bracket) the replan must still capture."""
        from aerocapture.cpag.model import ISIGMA

        cfg = ScpConfig(max_iters=30, horizon_max=1200.0)
        x0 = entry_state(TOML)
        x0[ISIGMA] = np.radians(90.0)
        sh = shoot_profile(x0, np.zeros(150), model, cfg)
        xm = sh.x_nodes[15].copy()
        r = scp_replan(xm, model, cfg)
        assert r.converged and r.feasible
        assert r.event == "exit"

    def test_osqp_rejected_in_ptr_mode(self, model) -> None:  # type: ignore[no-untyped-def]
        cfg = ScpConfig(trust_mode="ptr", solver="osqp")
        with pytest.raises(ValueError, match="box"):
            scp_replan(entry_state(TOML), model, cfg)
