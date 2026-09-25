"""FNPAG-style planning through a learned model or the true plant (#113).

The planner is FNPAG's corrector (src/rust/src/gnc/guidance/fnpag.rs): bisect a constant bank
magnitude so the predicted exit apoapsis hits the target, re-solved every replan period, with the
roll sign from a port of the lateral reversal law (src/rust/src/gnc/guidance/lateral.rs) at
FNPAG's deployed gains. Only the predictor differs between arms.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wm_model
import wm_plant

N_BISECT_STEPS = 8  # fnpag.rs: 8 halvings + 2 endpoints + 1 midpoint = 11 predictions per replan


@dataclass(frozen=True)
class LateralParams:
    tau: float  # s, projection horizon of the inclination error
    threshold: float  # rad
    min_reversal_interval: float  # s
    activation: float  # J/kg, upper edge of the energy window
    inhibition: float  # J/kg, lower edge
    max_reversals: int

    @classmethod
    def from_params(cls, p: dict[str, float]) -> LateralParams:
        """From a best_params.json's `lateral.*` genes (TOML units: deg, MJ/kg; max_reversals rounds like deploy_overrides)."""
        return cls(
            tau=p["lateral.tau"],
            threshold=np.deg2rad(p["lateral.threshold"]),
            min_reversal_interval=p["lateral.min_reversal_interval"],
            activation=p["lateral.lateral_activation"] * 1e6,
            inhibition=p["lateral.lateral_inhibition"] * 1e6,
            max_reversals=int(round(p["lateral.max_reversals"])),
        )


class Lateral:
    """One flight's roll sign: reverse when the inclination error projected tau seconds ahead exceeds the threshold."""

    def __init__(self, params: LateralParams) -> None:
        self.params = params
        self.sign = 1.0
        self.n_reversals = 0
        self.last_reversal_time = -np.inf
        self.prev_error: float | None = None
        self.prev_time = 0.0

    def update(self, error: float, energy: float, bank_magnitude: float, t: float) -> bool:
        """`error` = target - current inclination (rad), `energy` in J/kg; returns True on a reversal."""
        p = self.params
        if p.tau <= 0.0:
            return False
        if energy > p.activation or energy < p.inhibition or abs(bank_magnitude) < 1e-10 or abs(abs(bank_magnitude) - np.pi) < 1e-10:
            self.prev_error = None
            return False
        if self.prev_error is None:
            self.prev_error, self.prev_time = error, t
            return False
        dt = t - self.prev_time
        rate = (error - self.prev_error) / dt if dt > 1e-12 else 0.0
        self.prev_error, self.prev_time = error, t
        projected = error + rate * p.tau
        if abs(projected) <= p.threshold or self.n_reversals >= p.max_reversals or t - self.last_reversal_time < p.min_reversal_interval:
            return False
        desired = -1.0 if projected > 0.0 else 1.0
        if desired * self.sign >= 0.0:
            return False
        self.sign = desired
        self.n_reversals += 1
        self.last_reversal_time = t
        return True


def bisect_bank(residual: Callable[[np.ndarray, np.ndarray], np.ndarray], lo: np.ndarray, hi: np.ndarray, tol: float) -> np.ndarray:
    """FNPAG's corrector, vectorized over flights: the bank in [lo, hi] zeroing residual = apoapsis - target.

    `residual(rows, banks)` predicts for the flights `rows` only. Apoapsis falls with bank, so a
    non-positive residual at `lo` commands `lo` (least dissipation) and a non-negative one at `hi`
    commands `hi`; otherwise 8 halvings, stopping early per flight once |residual| < tol. Like
    fnpag.rs, the midpoint is only predicted for flights whose bracket straddles the target.
    """
    n = len(lo)
    every = np.arange(n)
    f_lo, f_hi = residual(every, lo), residual(every, hi)
    a, b = lo.copy(), hi.copy()
    mid = 0.5 * (a + b)
    active = (f_lo > 0.0) & (f_hi < 0.0)
    f_mid = np.full(n, np.nan)
    rows = np.flatnonzero(active)
    if len(rows):
        f_mid[rows] = residual(rows, mid[rows])
    for _ in range(N_BISECT_STEPS):
        active &= np.abs(f_mid) >= tol
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        a[rows] = np.where(f_mid[rows] > 0.0, mid[rows], a[rows])
        b[rows] = np.where(f_mid[rows] <= 0.0, mid[rows], b[rows])
        mid[rows] = 0.5 * (a[rows] + b[rows])
        f_mid[rows] = residual(rows, mid[rows])
    return np.where(f_lo <= 0.0, lo, np.where(f_hi >= 0.0, hi, mid))


# Feature indices: candidate inputs 1 inclination error (deg, current - target), 4 velocity (m/s),
# 8 altitude (km), 9 fpa (rad), 14 apoapsis altitude (m), 15 bounce flag, 31 periapsis altitude (m),
# then aux energy (MJ/kg) and asinh(dv / 100) x3 (wm_model.features).
INCL_ERR, VEL, ALT, FPA, APO, BOUNCE, PERI, ENERGY, DV = 1, 4, 8, 9, 14, 15, 31, 35, slice(37, 40)
UNBOUND_APOAPSIS_M = 1e9  # fnpag.rs UNBOUND_APOAPSIS_RADIUS_M: an unbound orbit reads as apoapsis at infinity
ROLLOUT_CAP = 1000  # ticks a model rollout may take to reach a predicted exit or crash


class Readout:
    """Termination and outcome of predicted trajectories.

    Exit and impact mirror the plant's events (events.rs). The re-descent clause of `crashed` does
    not: the plant and fnpag.rs keep integrating a post-bounce dip, this readout ends the rollout
    there and scores it as a crash.
    """

    def __init__(self, norm: list[dict[str, object]], exit_alt_km: float) -> None:
        self.norm, self.exit_alt_km = norm, exit_alt_km

    def raw(self, x: np.ndarray, j: int) -> np.ndarray:
        return wm_plant.raw_input(x[..., j], self.norm[j])

    def crashed(self, x: np.ndarray) -> np.ndarray:
        """Below the surface, or re-descending inside the atmosphere after the bounce."""
        alt, bounced = self.raw(x, ALT), self.raw(x, BOUNCE) > 0.5
        return np.asarray((alt <= 0.0) | (bounced & (self.raw(x, FPA) < 0.0) & (alt < self.exit_alt_km)))

    def exited(self, x: np.ndarray) -> np.ndarray:
        return np.asarray((self.raw(x, BOUNCE) > 0.5) & (self.raw(x, ALT) >= self.exit_alt_km))

    def terminated(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.crashed(x) | self.exited(x))

    def terminal(self, traj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(B, H, 42) trajectories -> (terminal step, crashed, terminal state (B, 42)); unterminated rows end at H - 1."""
        term = self.terminated(traj)
        step = np.where(term.any(axis=1), term.argmax(axis=1), traj.shape[1] - 1)
        last = traj[np.arange(len(traj)), step]
        return step, self.crashed(last), last

    def apoapsis(self, traj: np.ndarray) -> np.ndarray:
        """Predicted exit apoapsis altitude (m): 0 on a crash, UNBOUND_APOAPSIS_M on an unbound orbit or a diverged (non-finite) rollout.

        fnpag.rs's osc_apoapsis_radius reads a nan / infinite apoapsis as unbound, so the corrector adds bank.
        """
        _, crashed, last = self.terminal(traj)
        raw = self.raw(last, APO)
        apo = np.where((last[:, ENERGY] >= 0.0) | ~np.isfinite(raw), UNBOUND_APOAPSIS_M, raw)
        return np.asarray(np.where(crashed, 0.0, apo))

    def dv(self, traj: np.ndarray) -> np.ndarray:
        """Predicted correction delta-v at termination (m/s): inf on a crash or a diverged (nan) rollout."""
        _, crashed, last = self.terminal(traj)
        dv = (100.0 * np.sinh(last[:, DV])).sum(axis=1)
        return np.asarray(np.where(crashed | np.isnan(dv), np.inf, dv))


def apoapsis_m(ifinal: np.ndarray, ecc: np.ndarray, apoapsis_km: np.ndarray) -> np.ndarray:
    """True exit apoapsis altitude (m) from final records, with the Readout conventions (crash 0, unbound UNBOUND_APOAPSIS_M)."""
    return np.asarray(np.where(ifinal == 1, 0.0, np.where(ecc >= 1.0, UNBOUND_APOAPSIS_M, apoapsis_km * 1e3)))


def plant_apoapsis(eps: list[wm_plant.Episode]) -> np.ndarray:
    return apoapsis_m(np.array([e.ifinal for e in eps]), np.array([e.ecc for e in eps]), np.array([e.apoapsis_km for e in eps]))


@dataclass(frozen=True)
class PlannerConfig:
    target_apoapsis_m: float
    bank_min: float  # rad
    bank_max_high: float  # rad, above BANK_LIMIT_SWITCH_ALT_KM
    bank_max_low: float  # rad
    tol_m: float  # early-exit tolerance on the apoapsis residual
    replan_every: int  # ticks
    lateral: LateralParams

    @classmethod
    def from_fnpag(cls, p: dict[str, float], target_apoapsis_km: float, replan_every: int) -> PlannerConfig:
        """FNPAG's deployed gains (best_params.json); the energy_tol gene is the corrector's apoapsis tolerance in m."""
        return cls(
            target_apoapsis_m=target_apoapsis_km * 1e3,
            bank_min=np.deg2rad(p["bank_min_deg"]),
            bank_max_high=np.deg2rad(p["bank_max_high_deg"]),
            bank_max_low=np.deg2rad(p["bank_max_low_deg"]),
            tol_m=p["energy_tol"],
            replan_every=replan_every,
            lateral=LateralParams.from_params(p),
        )


BANK_LIMIT_SWITCH_ALT_KM = 50.0  # fnpag.rs BANK_LIMIT_SWITCH_ALTITUDE_M


class ModelArm:
    """Predicts exit apoapsis by free-running a learned model from each flight's filtered state."""

    def __init__(self, pred: wm_model.Predictor, n: int, readout: Readout) -> None:
        self.pred, self.readout = pred, readout
        self.h = np.zeros((n, wm_model.HIDDEN), np.float32)
        self.x: np.ndarray | None = None

    def observe(self, x: np.ndarray, applied: np.ndarray, live: np.ndarray) -> None:
        """x = the step's state x_k, applied = a_k: fold (x_{k-1}, a_k) into the recurrent state."""
        if self.pred.recurrent and self.x is not None and live.any():
            h, _ = self.pred.step(self.h[live], self.x[live], applied[live])
            assert h is not None
            self.h[live] = h
        self.x = x.copy()

    def apoapsis(self, idx: np.ndarray, banks: np.ndarray) -> np.ndarray:
        assert self.x is not None
        traj = self.pred.free_run(self.h[idx], self.x[idx], np.repeat(banks[:, None], ROLLOUT_CAP, axis=1), stop=self.readout.terminated)
        return self.readout.apoapsis(traj)


class OracleArm:
    """Predicts exit apoapsis by re-flying the true plant from t = 0 (same seed, the applied banks, then the candidate).

    Clairvoyant: the replay reproduces the flight's future density and navigation noise exactly.
    """

    def __init__(self, toml: str | Path, seeds: list[int], stub: Path) -> None:
        self.toml, self.seeds, self.stub = toml, seeds, stub
        self.plant: wm_plant.Plant | None = None

    def observe(self, x: np.ndarray, applied: np.ndarray, live: np.ndarray) -> None:
        pass

    def apoapsis(self, idx: np.ndarray, banks: np.ndarray) -> np.ndarray:
        assert self.plant is not None
        rows = np.concatenate([self.plant.applied_actions(idx), banks[:, None].astype(np.float32)], axis=1)
        return plant_apoapsis(wm_plant.fly(self.toml, [self.seeds[i] for i in idx], rows, self.stub))


def run_mpc(
    toml: str | Path, seeds: list[int], cfg: PlannerConfig, arm: ModelArm | OracleArm, readout: Readout, stub: Path
) -> tuple[list[wm_plant.Episode], dict[str, float]]:
    """Fly every seed closed loop: replan the bank magnitude every cfg.replan_every ticks, roll sign every tick.

    The bank chosen on step k's observation (nav at tick k + 1) is a_{k+1}, flown at tick k + 1 as deployed.
    """
    plant = wm_plant.Plant(toml, seeds, stub)
    if isinstance(arm, OracleArm):
        arm.plant = plant
    n = len(seeds)
    lateral = [Lateral(cfg.lateral) for _ in range(n)]
    mag = np.full(n, 0.5 * (cfg.bank_min + cfg.bank_max_high))
    act = mag.copy()
    plan_s, n_plans, k = 0.0, 0, 0
    while not plant.done.all():
        was_live = ~plant.done
        obs, aux = plant.step(act)
        arm.observe(wm_model.features(obs, aux), act.astype(np.float32), was_live)
        live = np.flatnonzero(~plant.done)
        alt = readout.raw(obs, ALT)
        hi = np.where(alt < BANK_LIMIT_SWITCH_ALT_KM, cfg.bank_max_low, cfg.bank_max_high)
        if k % cfg.replan_every == 0 and len(live):
            t0 = time.perf_counter()
            sign = np.array([lateral[i].sign for i in live])

            def residual(rows: np.ndarray, b: np.ndarray, live: np.ndarray = live, sign: np.ndarray = sign) -> np.ndarray:
                return arm.apoapsis(live[rows], sign[rows] * b) - cfg.target_apoapsis_m

            mag[live] = bisect_bank(residual, np.full(len(live), cfg.bank_min), hi[live], cfg.tol_m)
            plan_s += time.perf_counter() - t0
            n_plans += len(live)
            if k % (cfg.replan_every * 20) == 0:
                print(f"    tick {k:4d}: {len(live)} flying, {1e3 * plan_s / n_plans:.2f} ms per flight-replan")
        mag = np.clip(mag, cfg.bank_min, hi)
        incl_err = readout.raw(obs, INCL_ERR)
        for i in live:
            lateral[i].update(-np.deg2rad(incl_err[i]), float(aux[i, 0]), mag[i], k + 1.0)
        act = np.array([lat.sign for lat in lateral]) * mag
        k += 1
    return [plant.episode(i) for i in range(n)], {"plan_wall_s": plan_s, "flight_replans": n_plans, "ms_per_flight_replan": 1e3 * plan_s / max(n_plans, 1)}
