"""Embedded-solver benchmark on real CPAG subproblem instances.

Captures the actual QP/SOCP instances an SCP replan generates (cold entry
replan = largest, mid-pass = typical, late = smallest), then times each solver
on the canonicalized matrices alone — the same matrices a Rust C1
implementation would hand to the embedded solver, so scipy assembly cost is
excluded from the timing.

Solvers under test:
  clarabel (interior point, pure Rust — solves both the box-QP and the paper's
  PTR-SOCP variants), osqp (ADMM, C — box-QP only, cold + warm-started).
Timing is the solver-reported solve time (Clarabel includes its setup; OSQP
setup is measured separately and warm re-solves reuse the factorization —
the replan-to-replan regime).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from aerocapture.cpag.model import entry_state, load_model
from aerocapture.cpag.scp import QpData, ScpConfig, canonicalize, scp_replan


def capture_instances(toml_path: str, seg_dt: float, trust_mode: str) -> list[tuple[str, QpData]]:
    """Run one entry replan and keep first / middle / last subproblem instances."""
    model = load_model(toml_path)
    cfg = ScpConfig(seg_dt=seg_dt, trust_mode=trust_mode, max_iters=30, horizon_max=1200.0)
    collected: list[QpData] = []
    scp_replan(entry_state(toml_path), model, cfg, collect_qp=collected)
    if not collected:
        return []
    picks = {0: "first", len(collected) // 2: "mid", len(collected) - 1: "last"}
    return [(f"{trust_mode}_dt{seg_dt:g}_{name}", collected[i]) for i, name in picks.items()]


def bench_clarabel(qp: QpData, repeats: int) -> dict[str, Any]:
    import clarabel  # noqa: PLC0415

    p_s, q_s, a_all, b_all, n_eq, soc_dims = canonicalize(qp)
    n_in = qp.a_in.shape[0]
    cones = [clarabel.ZeroConeT(n_eq), clarabel.NonnegativeConeT(n_in)]
    cones += [clarabel.SecondOrderConeT(d) for d in soc_dims]
    p_triu = sp.triu(p_s).tocsc()
    times: list[float] = []
    iters: list[int] = []
    status = ""
    for _ in range(repeats):
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(p_triu, q_s, a_all, b_all, cones, settings)
        t0 = time.perf_counter()
        sol = solver.solve()
        times.append(time.perf_counter() - t0)
        iters.append(int(sol.iterations))
        status = str(sol.status)
    return _stats("clarabel", times, iters, status, p_s.shape[0], a_all.shape[0])


def bench_osqp(qp: QpData, repeats: int) -> dict[str, Any]:
    import osqp  # noqa: PLC0415

    if qp.soc_blocks:
        return {"solver": "osqp", "status": "skipped (SOC)"}
    p_s, q_s, a_all, b_all, n_eq, _ = canonicalize(qp)
    lower = np.concatenate([b_all[:n_eq], np.full(a_all.shape[0] - n_eq, -np.inf)])
    p_triu = sp.triu(p_s).tocsc()

    cold_times: list[float] = []
    cold_iters: list[int] = []
    status = ""
    for _ in range(repeats):
        prob = osqp.OSQP()
        prob.setup(P=p_triu, q=q_s, A=a_all, l=lower, u=b_all, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polishing=False)
        t0 = time.perf_counter()
        res = prob.solve()
        cold_times.append(time.perf_counter() - t0)
        cold_iters.append(int(res.info.iter))
        status = str(res.info.status)
    out = _stats("osqp_cold", cold_times, cold_iters, status, p_s.shape[0], a_all.shape[0])

    # Warm regime: keep the factorization, perturb q by 1% (replan-to-replan)
    prob = osqp.OSQP()
    prob.setup(P=p_triu, q=q_s, A=a_all, l=lower, u=b_all, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polishing=False)
    prob.solve()
    rng = np.random.default_rng(0)
    warm_times: list[float] = []
    warm_iters: list[int] = []
    for _ in range(repeats):
        prob.update(q=q_s * (1.0 + 0.01 * rng.standard_normal(q_s.shape[0])))
        t0 = time.perf_counter()
        res = prob.solve()
        warm_times.append(time.perf_counter() - t0)
        warm_iters.append(int(res.info.iter))
    warm = _stats("osqp_warm", warm_times, warm_iters, str(res.info.status), p_s.shape[0], a_all.shape[0])
    out["warm"] = warm
    return out


def _stats(solver: str, times: list[float], iters: list[int], status: str, n_vars: int, n_rows: int) -> dict[str, Any]:
    t = np.asarray(times)
    return {
        "solver": solver,
        "status": status,
        "n_vars": n_vars,
        "n_rows": n_rows,
        "time_ms_p50": float(np.percentile(t, 50) * 1e3),
        "time_ms_p95": float(np.percentile(t, 95) * 1e3),
        "iters_p50": float(np.percentile(iters, 50)),
        "iters_max": int(np.max(iters)),
    }


def run_bench(
    toml_path: str = "configs/nominal/msr_aller_ftc_nominal.toml",
    out_dir: str | Path = "training_output/cpag_c0",
    repeats: int = 25,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for seg_dt in (8.0, 4.0):
        for mode in ("box", "ptr"):
            for name, qp in capture_instances(toml_path, seg_dt, mode):
                entry: dict[str, Any] = {"instance": name, "n_seg": qp.n_seg}
                entry["clarabel"] = bench_clarabel(qp, repeats)
                if mode == "box":
                    entry["osqp"] = bench_osqp(qp, repeats)
                results.append(entry)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "solver_bench.json").write_text(json.dumps(results, indent=1))
    return results
