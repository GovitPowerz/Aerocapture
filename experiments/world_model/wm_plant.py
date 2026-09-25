"""The aerocapture plant as the world model sees it (#113): BatchedSimulation with an injected bank.

Contract of the seam (asserted in tests/test_world_model.py): the injected bank replaces the
deployed NN's output, so it is gated and command-shaped like one, and the NN telemetry inputs
(prev_bank_signed and its sin/cos) record the shaped command. `step(a_k)` flies tick k and returns
the navigation at tick k + 1, the input a deployed NN reads when choosing a_{k+1}. Row k of an
Episode is that post-a_k observation, so the world model predicts row k + 1 from rows <= k and
a_{k+1}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import aerocapture_rs
import numpy as np
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parents[2]
STUB_MODEL = REPO / "training_output/world_model/stub_model.json"
N_INPUTS = 35
FLY_BATCH = 1000  # flights per BatchedSimulation in `fly`
FR = aerocapture_rs.final_record_indices()

# Behavior policy: piecewise-constant signed bank. The per-episode base magnitude sits on the
# capture corridor (constant banks below ~60 deg skip out, above ~75 deg crash on this plant);
# each segment perturbs it, draws a random sign and a log-uniform duration; an OU jitter rides on top.
_BASE_BANK_DEG = (40.0, 100.0)
_SEGMENT_SIGMA_DEG = 25.0
_SEGMENT_TICKS = (5.0, 200.0)
_JITTER_SIGMA_DEG, _JITTER_TAU_S = 3.0, 10.0
_BEHAVIOR_STREAM = 113


def bank_schedule(seed: int, n_ticks: int) -> np.ndarray:
    """The behavior policy's signed bank (rad) per 1 s guidance tick, deterministic in the MC seed."""
    rng = np.random.default_rng([seed, _BEHAVIOR_STREAM])
    base = rng.uniform(*_BASE_BANK_DEG)
    out = np.empty(n_ticks)
    t = 0
    while t < n_ticks:
        d = int(np.exp(rng.uniform(np.log(_SEGMENT_TICKS[0]), np.log(_SEGMENT_TICKS[1]))))
        mag = np.clip(base + _SEGMENT_SIGMA_DEG * rng.standard_normal(), 0.0, 180.0)
        out[t : t + d] = rng.choice([-1.0, 1.0]) * np.deg2rad(mag)
        t += d
    phi = 1.0 - 1.0 / _JITTER_TAU_S
    jitter = lfilter([np.deg2rad(_JITTER_SIGMA_DEG) * np.sqrt(2.0 / _JITTER_TAU_S)], [1.0, -phi], rng.standard_normal(n_ticks))
    return np.asarray(np.clip(out + jitter, -np.pi, np.pi))


def write_stub_model(toml: str | Path, path: Path = STUB_MODEL) -> None:
    """A zero-weight dense 35 -> 2 model: only sets the observation width to all 35 candidate inputs.

    Embeds the config's resolved normalization so the observation is self-describing.
    """
    norm = aerocapture_rs.load_config(str(toml))["network"]["normalization"]
    arch = [{"type": "dense", "input_size": N_INPUTS, "output_size": 2, "activation": "linear"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    aerocapture_rs.flat_weights_to_json(
        [0.0] * (N_INPUTS * 2 + 2), json.dumps(arch), str(path), list(range(N_INPUTS)), "atan2_signed", None, None, json.dumps(norm)
    )


def load_normalization(stub: Path = STUB_MODEL) -> list[dict[str, object]]:
    norm: list[dict[str, object]] = json.loads(stub.read_text())["normalization"]
    return norm


def raw_input(x: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    """Invert one input's norm = transform((raw - center) / scale) (a tanh input clips at +-1)."""
    v = x.astype(np.float64)
    match spec["transform"]:
        case "asinh":
            v = np.sinh(v)
        case "tanh":
            v = np.arctanh(np.clip(v, -1.0 + 1e-7, 1.0 - 1e-7))
    return np.asarray(v * float(spec["scale"]) + float(spec["center"]))  # type: ignore[arg-type]


@dataclass
class Episode:
    """One flight. Row k = the observation and aux after flying bank a_k (nav at tick k + 1), and a_k."""

    seed: int
    obs: np.ndarray  # (T, 35) float32
    aux: np.ndarray  # (T, 7) float32: energy J/kg, pdyn Pa, predicted dv1..3 m/s, heat flux / load fractions
    actions: np.ndarray  # (T,) float32, rad
    ifinal: int
    captured: bool
    dv_m_s: float
    apoapsis_km: float  # true final osculating apoapsis altitude
    ecc: float
    final_record: np.ndarray  # (52,) aerocapture_rs.final_record_indices() columns

    def __len__(self) -> int:
        return len(self.actions)


class Plant:
    """N flights stepped in lockstep; a finished flight freezes at its terminal state (`auto_reset=False`)."""

    def __init__(self, toml: str | Path, seeds: list[int], stub: Path = STUB_MODEL) -> None:
        self.seeds = [int(s) for s in seeds]
        n = len(self.seeds)
        self.env = aerocapture_rs.BatchedSimulation(str(toml), n, {"data.neural_network": str(stub)}, auto_reset=False)
        self.env.reset(np.asarray(self.seeds, dtype=np.int64))
        self.done = np.zeros(n, dtype=bool)
        self.length = np.zeros(n, dtype=np.int64)
        self._obs: list[np.ndarray] = []
        self._aux: list[np.ndarray] = []
        self._act: list[np.ndarray] = []
        self._info: list[dict[str, object] | None] = [None] * n

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance every live flight one tick; returns this step's (obs, aux), the terminal rows for finished flights."""
        obs, _, done, info, aux = self.env.step(actions.astype(np.float32))
        live = ~self.done
        for i in np.flatnonzero(done & live):
            self._info[i] = info[i]
        self._obs.append(obs)
        self._aux.append(aux)
        self._act.append(actions.astype(np.float32))
        self.length[live] += 1
        self.done |= done
        return obs, aux

    def applied_actions(self, idx: np.ndarray) -> np.ndarray:
        """(len(idx), steps so far) banks applied to flights idx, as the env received them (float32)."""
        return np.asarray(np.stack(self._act, axis=1)[idx])

    def history(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flight i's (obs, aux, actions) so far."""
        n = self.length[i]
        return np.stack([o[i] for o in self._obs[:n]]), np.stack([a[i] for a in self._aux[:n]]), np.array([a[i] for a in self._act[:n]])

    def episode(self, i: int) -> Episode:
        info = self._info[i]
        assert info is not None, "flight has not finished"
        fr = info["final_record"]
        obs, aux, act = self.history(i)
        return Episode(
            seed=self.seeds[i],
            obs=obs,
            aux=aux,
            actions=act,
            ifinal=int(info["ifinal"]),  # type: ignore[call-overload]
            captured=bool(info["captured"]),
            dv_m_s=float(info["dv_m_s"]),  # type: ignore[arg-type]
            apoapsis_km=float(fr[FR["apoapsis_alt_km"]]),  # type: ignore[index]
            ecc=float(fr[FR["ecc"]]),  # type: ignore[index]
            final_record=np.asarray(fr, dtype=np.float64),
        )


def fly(toml: str | Path, seeds: list[int], actions: np.ndarray, stub: Path = STUB_MODEL) -> list[Episode]:
    """Fly seed i open loop on actions[i] (the last bank is held past the row's end) until every flight ends."""
    out: list[Episode] = []
    for lo in range(0, len(seeds), FLY_BATCH):
        plant = Plant(toml, list(seeds[lo : lo + FLY_BATCH]), stub)
        acts = actions[lo : lo + FLY_BATCH]
        k = 0
        while not plant.done.all():
            plant.step(acts[:, min(k, acts.shape[1] - 1)])
            k += 1
        out.extend(plant.episode(i) for i in range(len(plant.seeds)))
    return out
