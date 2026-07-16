"""SCP replan for the CPAG C0 prototype (paper formulation on repo dynamics).

Predictor-corrector structure per replan, following Rataczak's thesis Ch. 6 /
JGCD 2025 (see the research notes in the C0 findings doc):

1. PREDICT: integrate the nonlinear model from the current state under the
   stored bank-rate profile (ZOH segments, absolute-time grid) to atmospheric
   exit / crash / horizon. The reference is dynamically feasible by
   construction, so the delta formulation needs NO virtual control (zero
   correction is always feasible) — the paper's CPEG-inherited trick.
2. LINEARIZE: exact-discretization Jacobians of each segment map by central
   finite differences on batched RK4 (the paper uses analytic Jacobians + STM;
   same object, different derivation).
3. CORRECT: solve one convex subproblem for (dx, du) with
   - control-effort objective on the full rate profile (alpha1),
   - exact L1 penalties on the terminal energy-based apoapsis residual eps
     (alpha3) and terminal cos-inclination error (alpha2),
   - soft intermediate inclination corridor on the last 20% of nodes (lambda),
   - L1-slacked path rows: heat flux, g-load, dynamic pressure per node
     (extension: the paper carries load factor only) + terminal heat load
     (alpha5),
   - bank-rate bounds |u| <= max_bank_rate,
   - trust region: "ptr" mode = per-node SOC ||[dx_k; du_k]|| <= eta_k with
     linear penalty w_tr (the paper's penalized-trust-region, SOCP, Clarabel
     only) or "box" mode = hard |du| box + merit accept/reject (pure QP, also
     OSQP-compatible — the C1 candidate reduction).
4. Iterate until the scaled state correction is below tolerance.

Deviations from the paper, deliberate for C0 (documented in the findings doc):
free final time handled by re-timing the grid through the predictor each
iteration instead of a time-dilation parameter; ZOH bank rate instead of FOH;
L1 instead of L2 penalty on the intermediate inclination slacks; load-factor
constraint linearized directly instead of via an a^2 augmented state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from aerocapture.cpag.model import (
    IQ,
    IR,
    ISIGMA,
    N_STATES,
    UNBOUND_APOAPSIS_RADIUS_M,
    CpagModel,
    FloatArray,
    apoapsis_radius,
    cos_inclination,
    eps_apoapsis,
    path_quantities,
    rk4_step,
)

_PATH_TYPES = ("heat_flux", "g_load", "pdyn")

# Central-difference perturbation floors per state (r, lon, lat, v, gamma, psi, sigma, Q)
_FD_EPS = np.array([1.0, 1e-8, 1e-8, 1e-4, 1e-8, 1e-8, 1e-7, 1e-3], dtype=np.float64)
_FD_EPS_U = 1e-7

# Column scaling (Table 6.1 of the thesis, adapted to this mission's units):
# dr 1 km, angles 0.2 deg, dv 50 m/s, dsigma 15 deg, dQ 1000 kJ/m^2
_STATE_SCALE = np.array([1e3, 3.5e-3, 3.5e-3, 50.0, 3.5e-3, 3.5e-3, 0.26, 1e3], dtype=np.float64)
_U_SCALE = 0.05  # rad/s

_EPS_SCALE = 1e6  # eps rows reported in MJ/kg


@dataclass(frozen=True)
class ScpConfig:
    seg_dt: float = 8.0  # ZOH bank-rate segment duration (s) — N ~ 50 over a pass
    n_sub: int = 4  # RK4 substeps per segment (integration dt = seg_dt / n_sub)
    horizon_max: float = 800.0  # planning horizon cap (s)
    max_bank_rate: float = np.radians(15.0)  # |sigma_dot| bound (vehicle limit)
    # |sigma| box per node: without it the optimizer WINDS the bank through full
    # turns (each 360 deg sweep is merit-free, each linearization sees a local
    # gain through the lateral-lift term) and strands itself — observed sigma_end
    # of 729 deg on crash-side replans. The paper never hits this because its
    # references start benign; corridor-edge replans do.
    sigma_max: float = float(np.pi)
    max_iters: int = 20
    tol_dx: float = 0.02  # convergence: max scaled |dx| below this
    tol_apo_m: float = 1e4  # feasibility: nonlinear apoapsis error (FNPAG energy_tol)
    tol_inc_deg: float = 0.5  # feasibility: inclination error (mission success tol)
    alpha1: float = 1e-2  # control effort (per (rad/s)^2 s)
    alpha2: float = 5.0  # terminal cos-inclination exact penalty
    alpha3: float = 1000.0  # terminal eps exact penalty (per MJ/kg)
    alpha5: float = 1000.0  # path-constraint slack penalty (per unit fraction)
    lambda_di: float = 1000.0  # intermediate inclination slack penalty
    di_node_fraction: float = 0.8  # intermediate inclination rows on nodes >= this fraction
    # Deadband on the intermediate inclination rows (cos-space): mid-arc error below
    # the mission tolerance is free. Without it the lambda term dominates the merit
    # with physically unavoidable error, re-sampled on a re-timed grid — pure noise.
    di_deadband: float = float(np.sin(np.radians(50.0)) * np.radians(0.5))
    w_tr: float = 1e-3  # PTR eta penalty (ptr mode)
    trust_init: float = 4.0  # box mode: initial |du| box in _U_SCALE units
    trust_min: float = 0.02
    trust_max: float = 8.0
    trust_mode: str = "box"  # "box" (QP + merit accept/reject) | "ptr" (paper's SOCP, Clarabel only)
    solver: str = "clarabel"  # "clarabel" | "osqp"
    # pdyn deliberately excluded by default: the mission nominal peaks at 1.63 kPa
    # vs the 1.081 kPa config value, and the GA cost function never penalizes pdyn —
    # an unsatisfiable row just saturates its slacks and drowns the merit function.
    path_constraints: tuple[str, ...] = ("heat_flux", "g_load")
    enforce_inclination: bool = True


@dataclass
class Shoot:
    """Nonlinear propagation to exit/crash/horizon on the segment grid."""

    x_nodes: FloatArray  # (M+1, 8)
    u_nodes: FloatArray  # (M,)
    dts: FloatArray  # (M,) segment durations (last may be partial)
    event: str  # "exit" | "crash" | "horizon"
    t_final: float


@dataclass
class QpData:
    """Canonical conic program:

    min 1/2 z^T P z + q^T z
    s.t. Aeq z = beq;  Ain z <= bin;  b_soc_i - A_soc_i z in SOC(d_i)

    Variables are column-scaled by `col_scale` before the solver sees them.
    Layout: [dx (8(M+1)), du (M), nu_path (3(M+1)), nu_hl, s_eps, s_inc,
    nu_di (n_di), eta (M+1, ptr mode only)].
    """

    p_mat: sp.csc_matrix
    q: FloatArray
    a_eq: sp.csc_matrix
    b_eq: FloatArray
    a_in: sp.csc_matrix
    b_in: FloatArray
    soc_blocks: list[tuple[sp.csc_matrix, FloatArray]]
    col_scale: FloatArray
    n_seg: int


@dataclass
class SolveInfo:
    status: str
    iterations: int
    solve_time: float
    obj_val: float
    n_vars: int
    n_rows: int


@dataclass
class ScpResult:
    converged: bool
    feasible: bool
    n_iters: int
    u_profile: FloatArray
    sigma0: float
    apo_error_m: float  # nonlinear (propagated) terminal apoapsis error, inf if unbound
    eps_mj: float  # nonlinear terminal energy-apoapsis residual (MJ/kg)
    inc_error_deg: float
    event: str
    path_peaks: dict[str, float]  # peak fraction of each limit along the shoot
    heat_load_frac: float
    dx_inf_history: list[float] = field(default_factory=list)
    merit_history: list[float] = field(default_factory=list)
    solves: list[SolveInfo] = field(default_factory=list)
    wall_time: float = 0.0


def roll_to_bank_profile(x0: FloatArray, target_sigma: float, cfg: ScpConfig) -> FloatArray:
    """Bank-rate profile rolling sigma to `target_sigma` at max rate, then holding."""
    n_seg = int(np.ceil(cfg.horizon_max / cfg.seg_dt))
    u = np.zeros(n_seg)
    sig = float(x0[ISIGMA])
    for i in range(n_seg):
        step = float(np.clip(target_sigma - sig, -cfg.max_bank_rate * cfg.seg_dt, cfg.max_bank_rate * cfg.seg_dt))
        u[i] = step / cfg.seg_dt
        sig += step
    return u


def shoot_profile(x0: FloatArray, u_nodes: FloatArray, model: CpagModel, cfg: ScpConfig) -> Shoot:
    """Propagate segment-wise with substep exit/crash detection.

    Exit: spherical altitude >= exit_alt while ascending; crash: altitude <= 0
    (the FNPAG predictor's conventions). Event time by linear interpolation
    between substeps, terminal state by a partial RK4 step.
    """
    req = model.planet.req
    dt_sub = cfg.seg_dt / cfg.n_sub
    x_nodes = [x0.copy()]
    dts: list[float] = []
    t = 0.0
    x = x0.copy()
    for k in range(len(u_nodes)):
        u = np.asarray(u_nodes[k])
        node_start_t = t
        for _ in range(cfg.n_sub):
            x_new = rk4_step(x, u, dt_sub, model)
            alt_prev = float(x[IR]) - req
            alt_new = float(x_new[IR]) - req
            crash = alt_new <= 0.0
            exiting = alt_new >= model.exit_alt and float(np.sin(x_new[4])) > 0.0
            if crash or exiting:
                target = 0.0 if crash else model.exit_alt
                denom = alt_new - alt_prev
                frac = 0.5 if abs(denom) < 1e-12 else float(np.clip((target - alt_prev) / denom, 0.0, 1.0))
                dt_part = frac * dt_sub
                x_event = rk4_step(x, u, dt_part, model) if dt_part > 1e-9 else x.copy()
                t_event = t + dt_part
                dt_last = t_event - node_start_t
                if dt_last > 1e-6:
                    x_nodes.append(x_event)
                    dts.append(dt_last)
                return Shoot(
                    x_nodes=np.asarray(x_nodes),
                    u_nodes=u_nodes[: len(dts)],
                    dts=np.asarray(dts),
                    event="crash" if crash else "exit",
                    t_final=t_event,
                )
            x = x_new
            t += dt_sub
        x_nodes.append(x.copy())
        dts.append(t - node_start_t)
    return Shoot(x_nodes=np.asarray(x_nodes), u_nodes=u_nodes[: len(dts)], dts=np.asarray(dts), event="horizon", t_final=t)


def linearize_segments(shoot: Shoot, model: CpagModel, cfg: ScpConfig) -> tuple[FloatArray, FloatArray]:
    """(A (M,8,8), B (M,8)) of each segment map by central differences.

    All segments x all perturbations integrate as ONE batched RK4 chain
    (per-row dt), so the cost is a handful of vectorized eom calls.
    """
    m_seg = len(shoot.dts)
    x0s = shoot.x_nodes[:-1]
    us = shoot.u_nodes
    n_pert = 2 * N_STATES + 2
    xb = np.repeat(x0s[:, None, :], n_pert, axis=1)
    ub = np.repeat(us[:, None], n_pert, axis=1)
    for j in range(N_STATES):
        xb[:, 2 * j, j] += _FD_EPS[j]
        xb[:, 2 * j + 1, j] -= _FD_EPS[j]
    ub[:, 2 * N_STATES] += _FD_EPS_U
    ub[:, 2 * N_STATES + 1] -= _FD_EPS_U

    flat_x = xb.reshape(m_seg * n_pert, N_STATES)
    flat_u = ub.reshape(m_seg * n_pert)
    flat_dt = np.repeat(shoot.dts / cfg.n_sub, n_pert)
    for _ in range(cfg.n_sub):
        flat_x = rk4_step(flat_x, flat_u, flat_dt, model)
    xf = flat_x.reshape(m_seg, n_pert, N_STATES)

    a_mats = np.empty((m_seg, N_STATES, N_STATES))
    for j in range(N_STATES):
        a_mats[:, :, j] = (xf[:, 2 * j, :] - xf[:, 2 * j + 1, :]) / (2.0 * _FD_EPS[j])
    b_mats = (xf[:, 2 * N_STATES, :] - xf[:, 2 * N_STATES + 1, :]) / (2.0 * _FD_EPS_U)
    return a_mats, b_mats


def _fd_state_gradient(f: Callable[[FloatArray], float], x: FloatArray) -> FloatArray:
    grad = np.empty(N_STATES)
    for j in range(N_STATES):
        xp = x.copy()
        xm = x.copy()
        xp[j] += _FD_EPS[j]
        xm[j] -= _FD_EPS[j]
        grad[j] = (f(xp) - f(xm)) / (2.0 * _FD_EPS[j])
    return grad


def _path_rows(x_nodes: FloatArray, model: CpagModel) -> tuple[FloatArray, FloatArray]:
    """Normalized path values g/limit (K,3) and gradients (K,3,8) by central FD."""
    limits = np.array([model.limits.max_heat_flux, model.limits.max_g_load, model.limits.max_pdyn])
    n_pert = 2 * N_STATES
    xb = np.repeat(x_nodes[:, None, :], n_pert + 1, axis=1)
    for j in range(N_STATES):
        xb[:, 2 * j, j] += _FD_EPS[j]
        xb[:, 2 * j + 1, j] -= _FD_EPS[j]
    hf, gl, pd = path_quantities(xb, model)
    vals = np.stack([hf, gl, pd], axis=-1) / limits
    grads = np.empty((x_nodes.shape[0], 3, N_STATES))
    for j in range(N_STATES):
        grads[:, :, j] = (vals[:, 2 * j, :] - vals[:, 2 * j + 1, :]) / (2.0 * _FD_EPS[j])
    return vals[:, n_pert, :], grads


class _Rows:
    """Sparse row accumulator for one constraint block."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.r: list[int] = []
        self.c: list[int] = []
        self.v: list[float] = []
        self.b: list[float] = []

    def add(self, entries: list[tuple[int, float]], rhs: float) -> None:
        row = len(self.b)
        for c, v in entries:
            self.r.append(row)
            self.c.append(c)
            self.v.append(v)
        self.b.append(rhs)

    def matrix(self) -> tuple[sp.csc_matrix, FloatArray]:
        data = (np.asarray(self.v, dtype=np.float64), (np.asarray(self.r, dtype=np.int64), np.asarray(self.c, dtype=np.int64)))
        a = sp.csc_matrix(sp.coo_matrix(data, shape=(len(self.b), self.n)))
        return a, np.asarray(self.b)


def build_qp(shoot: Shoot, a_mats: FloatArray, b_mats: FloatArray, model: CpagModel, cfg: ScpConfig, trust: float) -> QpData:
    m_seg = len(shoot.dts)
    n_nodes = m_seg + 1
    nx = N_STATES * n_nodes
    inc_active = cfg.enforce_inclination and shoot.event != "crash"
    di_start = int(np.ceil(cfg.di_node_fraction * m_seg)) if inc_active else n_nodes
    di_nodes = list(range(di_start, m_seg)) if inc_active else []
    type_idx = [_PATH_TYPES.index(t) for t in cfg.path_constraints]
    n_types = len(type_idx)
    n_path = n_types * n_nodes
    ptr = cfg.trust_mode == "ptr"

    i_u0 = nx
    i_np0 = i_u0 + m_seg
    i_nhl = i_np0 + n_path
    i_seps = i_nhl + 1
    i_sinc = i_seps + 1
    i_di0 = i_sinc + 1
    i_eta0 = i_di0 + len(di_nodes)
    n = i_eta0 + (n_nodes if ptr else 0)

    ix = [N_STATES * k for k in range(n_nodes)]

    # ── Equalities: dx_0 = 0; dx_{k+1} = A_k dx_k + B_k du_k ──
    eq = _Rows(n)
    for j in range(N_STATES):
        eq.add([(j, 1.0)], 0.0)
    for k in range(m_seg):
        for i in range(N_STATES):
            entries = [(ix[k + 1] + i, 1.0)]
            entries += [(ix[k] + j, -float(a_mats[k, i, j])) for j in range(N_STATES) if a_mats[k, i, j] != 0.0]
            if b_mats[k, i] != 0.0:
                entries.append((i_u0 + k, -float(b_mats[k, i])))
            eq.add(entries, 0.0)

    # ── Inequalities ──
    ineq = _Rows(n)
    u_bar = shoot.u_nodes
    for k in range(m_seg):
        ineq.add([(i_u0 + k, 1.0)], cfg.max_bank_rate - float(u_bar[k]))
        ineq.add([(i_u0 + k, -1.0)], cfg.max_bank_rate + float(u_bar[k]))
        if not ptr:
            ineq.add([(i_u0 + k, 1.0)], trust * _U_SCALE)
            ineq.add([(i_u0 + k, -1.0)], trust * _U_SCALE)
    for k in range(1, n_nodes):  # node 0 is pinned by dx_0 = 0
        sig_bar = float(shoot.x_nodes[k, ISIGMA])
        ineq.add([(ix[k] + ISIGMA, 1.0)], cfg.sigma_max - sig_bar)
        ineq.add([(ix[k] + ISIGMA, -1.0)], cfg.sigma_max + sig_bar)

    if n_types:
        path_vals, path_grads = _path_rows(shoot.x_nodes, model)
        for k in range(n_nodes):
            for s_i, t in enumerate(type_idx):
                entries = [(ix[k] + j, float(path_grads[k, t, j])) for j in range(N_STATES) if path_grads[k, t, j] != 0.0]
                entries.append((i_np0 + n_types * k + s_i, -1.0))
                ineq.add(entries, 1.0 - float(path_vals[k, t]))
                ineq.add([(i_np0 + n_types * k + s_i, -1.0)], 0.0)

    # Terminal heat load (normalized by the limit)
    q_max = model.limits.max_heat_load
    q_bar = float(shoot.x_nodes[-1, IQ])
    ineq.add([(ix[m_seg] + IQ, 1.0 / q_max), (i_nhl, -1.0)], (q_max - q_bar) / q_max)
    ineq.add([(i_nhl, -1.0)], 0.0)

    # Terminal eps (energy-based apoapsis targeting), in MJ/kg
    x_n = shoot.x_nodes[-1]
    eps_bar = float(eps_apoapsis(x_n, model)) / _EPS_SCALE

    def f_eps(x: FloatArray) -> float:
        return float(eps_apoapsis(x, model)) / _EPS_SCALE

    grad_eps = _fd_state_gradient(f_eps, x_n)
    ge = [(ix[m_seg] + j, float(grad_eps[j])) for j in range(N_STATES) if grad_eps[j] != 0.0]
    ineq.add([*ge, (i_seps, -1.0)], -eps_bar)
    ineq.add([*[(c, -v) for c, v in ge], (i_seps, -1.0)], eps_bar)
    ineq.add([(i_seps, -1.0)], 0.0)

    # Terminal + intermediate inclination (cos i, frame-free)
    ci_t = float(np.cos(model.target_inclination))
    if inc_active:

        def f_ci(x: FloatArray) -> float:
            return float(cos_inclination(x, model.planet))

        ci_bar_n = f_ci(x_n) - ci_t
        grad_ci_n = _fd_state_gradient(f_ci, x_n)
        gi = [(ix[m_seg] + j, float(grad_ci_n[j])) for j in range(N_STATES) if grad_ci_n[j] != 0.0]
        ineq.add([*gi, (i_sinc, -1.0)], -ci_bar_n)
        ineq.add([*[(c, -v) for c, v in gi], (i_sinc, -1.0)], ci_bar_n)
        for idx, k in enumerate(di_nodes):
            xk = shoot.x_nodes[k]
            ci_bar_k = f_ci(xk) - ci_t
            grad_k = _fd_state_gradient(f_ci, xk)
            gk = [(ix[k] + j, float(grad_k[j])) for j in range(N_STATES) if grad_k[j] != 0.0]
            ineq.add([*gk, (i_di0 + idx, -1.0)], cfg.di_deadband - ci_bar_k)
            ineq.add([*[(c, -v) for c, v in gk], (i_di0 + idx, -1.0)], cfg.di_deadband + ci_bar_k)
            ineq.add([(i_di0 + idx, -1.0)], 0.0)
    ineq.add([(i_sinc, -1.0)], 0.0)

    # ── Column scaling ──
    col_scale = np.ones(n)
    col_scale[:nx] = np.tile(_STATE_SCALE, n_nodes)
    col_scale[i_u0 : i_u0 + m_seg] = _U_SCALE

    # ── PTR SOC blocks: || [dx_k; du_k] / scale ||_2 <= eta_k ──
    soc_blocks: list[tuple[sp.csc_matrix, FloatArray]] = []
    if ptr:
        for k in range(n_nodes):
            dim = 1 + N_STATES + (1 if k < m_seg else 0)
            rows_soc = _Rows(n)
            rows_soc.add([(i_eta0 + k, -1.0)], 0.0)
            for j in range(N_STATES):
                rows_soc.add([(ix[k] + j, -1.0)], 0.0)
            if k < m_seg:
                rows_soc.add([(i_u0 + k, -1.0)], 0.0)
            a_soc, b_soc = rows_soc.matrix()
            assert a_soc.shape[0] == dim
            soc_blocks.append((a_soc, b_soc))

    # ── Objective ──
    p_rows: list[int] = []
    p_cols: list[int] = []
    p_vals: list[float] = []
    q_vec = np.zeros(n)
    for k in range(m_seg):
        dt_k = float(shoot.dts[k])
        p_rows.append(i_u0 + k)
        p_cols.append(i_u0 + k)
        p_vals.append(2.0 * cfg.alpha1 * dt_k)
        q_vec[i_u0 + k] += 2.0 * cfg.alpha1 * dt_k * float(u_bar[k])
    q_vec[i_seps] = cfg.alpha3
    q_vec[i_sinc] = cfg.alpha2 if inc_active else 0.0
    if n_path:
        q_vec[i_np0 : i_np0 + n_path] = cfg.alpha5
    q_vec[i_nhl] = cfg.alpha5
    if di_nodes:
        q_vec[i_di0 : i_di0 + len(di_nodes)] = cfg.lambda_di
    if ptr:
        q_vec[i_eta0 : i_eta0 + n_nodes] = cfg.w_tr

    p_data = (np.asarray(p_vals, dtype=np.float64), (np.asarray(p_rows, dtype=np.int64), np.asarray(p_cols, dtype=np.int64)))
    p_mat = sp.csc_matrix(sp.coo_matrix(p_data, shape=(n, n)))
    a_eq, b_eq = eq.matrix()
    a_in, b_in = ineq.matrix()
    return QpData(p_mat=p_mat, q=q_vec, a_eq=a_eq, b_eq=b_eq, a_in=a_in, b_in=b_in, soc_blocks=soc_blocks, col_scale=col_scale, n_seg=m_seg)


def canonicalize(qp: QpData) -> tuple[sp.csc_matrix, FloatArray, sp.csc_matrix, FloatArray, int, list[int]]:
    """Column-scale and stack rows: (P, q, A, b, n_eq, soc_dims). z = D z'."""
    d = sp.diags(qp.col_scale).tocsc()
    n_eq = qp.a_eq.shape[0]
    row_scale_eq = 1.0 / np.tile(_STATE_SCALE, n_eq // N_STATES)
    r_eq = sp.diags(row_scale_eq).tocsc()
    blocks_a = [(r_eq @ qp.a_eq @ d).tocsc(), (qp.a_in @ d).tocsc()]
    blocks_b = [row_scale_eq * qp.b_eq, qp.b_in]
    soc_dims: list[int] = []
    for a_soc, b_soc in qp.soc_blocks:
        blocks_a.append((a_soc @ d).tocsc())
        blocks_b.append(b_soc)
        soc_dims.append(a_soc.shape[0])
    a_all = sp.vstack(blocks_a, format="csc")
    b_all = np.concatenate(blocks_b)
    p_s = (d @ qp.p_mat @ d).tocsc()
    q_s = qp.col_scale * qp.q
    return p_s, q_s, a_all, b_all, n_eq, soc_dims


def solve_qp_clarabel(qp: QpData) -> tuple[FloatArray, SolveInfo]:
    import clarabel  # noqa: PLC0415

    p_s, q_s, a_all, b_all, n_eq, soc_dims = canonicalize(qp)
    n_in = qp.a_in.shape[0]
    cones = [clarabel.ZeroConeT(n_eq), clarabel.NonnegativeConeT(n_in)]
    cones += [clarabel.SecondOrderConeT(d) for d in soc_dims]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(sp.triu(p_s).tocsc(), q_s, a_all, b_all, cones, settings)
    sol = solver.solve()
    z = qp.col_scale * np.asarray(sol.x)
    return z, SolveInfo(
        status=str(sol.status),
        iterations=int(sol.iterations),
        solve_time=float(sol.solve_time),
        obj_val=float(sol.obj_val),
        n_vars=int(p_s.shape[0]),
        n_rows=int(a_all.shape[0]),
    )


def solve_qp_osqp(qp: QpData) -> tuple[FloatArray, SolveInfo]:
    import osqp  # noqa: PLC0415

    if qp.soc_blocks:
        raise ValueError("OSQP cannot solve SOC blocks — use trust_mode='box'")
    p_s, q_s, a_all, b_all, n_eq, _ = canonicalize(qp)
    lower = np.concatenate([b_all[:n_eq], np.full(a_all.shape[0] - n_eq, -np.inf)])
    prob = osqp.OSQP()
    prob.setup(P=sp.triu(p_s).tocsc(), q=q_s, A=a_all, l=lower, u=b_all, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polishing=True)
    t0 = time.perf_counter()
    res = prob.solve()
    wall = time.perf_counter() - t0
    z = qp.col_scale * np.asarray(res.x, dtype=np.float64)
    return z, SolveInfo(
        status=str(res.info.status),
        iterations=int(res.info.iter),
        solve_time=min(float(res.info.solve_time + res.info.setup_time), wall),
        obj_val=float(res.info.obj_val),
        n_vars=int(p_s.shape[0]),
        n_rows=int(a_all.shape[0]),
    )


_CRASH_MERIT_OFFSET = 1e4  # any exit outranks any crash; eps still ranks within a tier


def _merit(shoot: Shoot, model: CpagModel, cfg: ScpConfig) -> tuple[float, dict[str, float], float, float, float, float]:
    """Nonlinear merit + diagnostics: (J, peaks, apo_err_m, eps_mj, inc_err_deg, hl_frac).

    Grid-size-invariant by construction (terminal values, per-node MEANS and
    PEAKS — never node sums): accept/reject compares trajectories whose node
    counts differ after re-timing, and a node-sum merit penalizes surviving
    longer, which blocks recovery from crash-truncated references.
    """
    hf, gl, pd = path_quantities(shoot.x_nodes, model)
    limits = model.limits
    peaks = {
        "heat_flux": float(np.max(hf) / limits.max_heat_flux),
        "g_load": float(np.max(gl) / limits.max_g_load),
        "pdyn": float(np.max(pd) / limits.max_pdyn),
    }
    x_n = shoot.x_nodes[-1]
    hl_frac = float(x_n[IQ] / limits.max_heat_load)
    eps_mj = float(eps_apoapsis(x_n, model)) / _EPS_SCALE
    apo = float(apoapsis_radius(x_n, model.planet))
    apo_err = float("inf") if apo >= UNBOUND_APOAPSIS_RADIUS_M else apo - model.target_apoapsis_radius
    inc = float(np.degrees(np.arccos(np.clip(cos_inclination(x_n, model.planet), -1.0, 1.0))))
    inc_err = abs(inc - float(np.degrees(model.target_inclination)))
    j = cfg.alpha3 * abs(eps_mj)
    if shoot.event == "crash":
        # Survival tier: inclination is meaningless on a spiraling-down arc and
        # fighting for it burns the lateral authority the pull-up needs.
        j += _CRASH_MERIT_OFFSET
    elif cfg.enforce_inclination:
        ci_t = float(np.cos(model.target_inclination))
        j += cfg.alpha2 * abs(float(cos_inclination(x_n, model.planet)) - ci_t)
        di_start = int(np.ceil(cfg.di_node_fraction * len(shoot.dts)))
        ci_mid = cos_inclination(shoot.x_nodes[di_start : len(shoot.dts)], model.planet)
        if ci_mid.size:
            j += cfg.lambda_di * float(np.mean(np.maximum(np.abs(ci_mid - ci_t) - cfg.di_deadband, 0.0))) * 10.0
    fracs = {"heat_flux": hf / limits.max_heat_flux, "g_load": gl / limits.max_g_load, "pdyn": pd / limits.max_pdyn}
    for t_name in cfg.path_constraints:
        j += cfg.alpha5 * (max(0.0, peaks[t_name] - 1.0) + float(np.mean(np.maximum(fracs[t_name] - 1.0, 0.0))))
    j += cfg.alpha5 * max(0.0, hl_frac - 1.0)
    return j, peaks, apo_err, eps_mj, inc_err, hl_frac


def scp_replan(
    x0: FloatArray,
    model: CpagModel,
    cfg: ScpConfig,
    u_init: FloatArray | None = None,
    collect_qp: list[QpData] | None = None,
) -> ScpResult:
    """Run one full SCP replan from state x0 (sigma0 = x0[ISIGMA])."""
    t_start = time.perf_counter()
    n_seg = int(np.ceil(cfg.horizon_max / cfg.seg_dt))
    u = np.zeros(n_seg) if u_init is None else np.clip(np.resize(np.asarray(u_init, dtype=np.float64), n_seg), -cfg.max_bank_rate, cfg.max_bank_rate)

    solve = solve_qp_clarabel if cfg.solver == "clarabel" else solve_qp_osqp
    if cfg.solver == "osqp" and cfg.trust_mode == "ptr":
        raise ValueError("OSQP requires trust_mode='box' (no SOC support)")

    result = ScpResult(
        converged=False,
        feasible=False,
        n_iters=0,
        u_profile=u.copy(),
        sigma0=float(x0[ISIGMA]),
        apo_error_m=float("inf"),
        eps_mj=float("nan"),
        inc_error_deg=float("nan"),
        event="none",
        path_peaks={},
        heat_load_frac=0.0,
    )
    trust = cfg.trust_init
    shoot = shoot_profile(x0, u, model, cfg)
    j_cur, peaks, apo_err, eps_mj, inc_err, hl_frac = _merit(shoot, model, cfg)

    # Constant-bank grid seeding: a cold replan far from the target (or on a
    # crashing reference) must otherwise climb out one trust step at a time and
    # can strand in a local optimum mid-corridor. Evaluate roll-to-constant-bank
    # profiles across the corridor (FNPAG's monotone apoapsis-vs-bank bracket as
    # an initializer; ~25 ms per shoot) and start from the best-merit reference.
    # Pure initialization; the subproblem is unchanged. In the C1 guidance loop
    # this fires only on the first call — later replans warm-start from the
    # previous profile.
    if u_init is None:
        sign = float(np.copysign(1.0, x0[ISIGMA])) if x0[ISIGMA] != 0.0 else 1.0
        for target_deg in (0.0, 45.0, 75.0, 105.0, 135.0, 180.0):
            u_cand = roll_to_bank_profile(x0, sign * np.radians(target_deg), cfg)
            shoot_cand = shoot_profile(x0, u_cand, model, cfg)
            j_cand = _merit(shoot_cand, model, cfg)[0]
            if j_cand < j_cur:
                u, shoot = u_cand, shoot_cand
                j_cur, peaks, apo_err, eps_mj, inc_err, hl_frac = _merit(shoot, model, cfg)

    for it in range(cfg.max_iters):
        result.n_iters = it + 1
        a_mats, b_mats = linearize_segments(shoot, model, cfg)
        m_seg = len(shoot.dts)
        accepted = False
        for _ in range(12 if cfg.trust_mode == "box" else 1):
            qp = build_qp(shoot, a_mats, b_mats, model, cfg, trust)
            if collect_qp is not None:
                collect_qp.append(qp)
            z, info = solve(qp)
            result.solves.append(info)
            if not info.status.lower().startswith(("solved", "optimal", "almostsolved", "solved inaccurate")):
                if cfg.trust_mode != "box":
                    break
                trust = max(trust * 0.5, cfg.trust_min)
                continue
            du = z[N_STATES * (m_seg + 1) : N_STATES * (m_seg + 1) + m_seg]
            dx = z[: N_STATES * (m_seg + 1)].reshape(m_seg + 1, N_STATES)
            dx_inf = float(np.max(np.abs(dx / _STATE_SCALE)))
            u_new = u.copy()
            u_new[:m_seg] = np.clip(u[:m_seg] + du, -cfg.max_bank_rate, cfg.max_bank_rate)
            if m_seg < n_seg:
                u_new[m_seg:] = 0.0
            shoot_new = shoot_profile(x0, u_new, model, cfg)
            j_new, peaks_new, apo_new, eps_new, inc_new, hl_new = _merit(shoot_new, model, cfg)
            if cfg.trust_mode == "box":
                # Greedy accept with noise tolerance: the QP-slack "predicted
                # merit" is not commensurate across re-timed grids, so a plain
                # descent test beats an SCvx rho ratio here.
                if j_new > j_cur * (1.0 + 1e-3) + 1e-6:
                    if trust <= cfg.trust_min * 1.001:
                        break  # at the floor a retry is a deterministic repeat
                    trust = max(trust * 0.4, cfg.trust_min)
                    continue
                trust = min(trust * 1.5, cfg.trust_max)
            accepted = True
            u, shoot = u_new, shoot_new
            j_cur, peaks, apo_err, eps_mj, inc_err, hl_frac = j_new, peaks_new, apo_new, eps_new, inc_new, hl_new
            result.dx_inf_history.append(dx_inf)
            result.merit_history.append(j_cur)
            break
        if not accepted:
            # Merit noise from grid re-timing rejects every step at a fixed
            # point; if any progress was made this IS convergence — to the
            # constrained optimum, which need not meet the target ("feasible"
            # carries that verdict; unreachable targets settle least-bad here).
            result.converged = bool(result.merit_history)
            break
        if result.dx_inf_history and result.dx_inf_history[-1] < cfg.tol_dx:
            result.converged = True
            break
        # Merit-stagnation convergence: the iteration has settled (re-shoot grid
        # re-timing sets a dx noise floor, and saturated least-bad optima never
        # reach tol_dx). Target attainment is reported separately via `feasible`.
        if len(result.merit_history) >= 3 and abs(result.merit_history[-1] - result.merit_history[-3]) <= 0.005 * max(1.0, result.merit_history[-3]):
            result.converged = True
            break

    result.u_profile = u
    result.apo_error_m = apo_err
    result.eps_mj = eps_mj
    result.inc_error_deg = inc_err
    result.event = shoot.event
    result.path_peaks = peaks
    result.heat_load_frac = hl_frac
    slop = 1.01
    result.feasible = (
        np.isfinite(apo_err)
        and abs(apo_err) <= cfg.tol_apo_m
        and (not cfg.enforce_inclination or inc_err <= cfg.tol_inc_deg)
        and all(peaks[t] <= slop for t in cfg.path_constraints)
        and hl_frac <= slop
        and shoot.event == "exit"
    )
    result.wall_time = time.perf_counter() - t_start
    return result
