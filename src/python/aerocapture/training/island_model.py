"""Three-island PSO/GA/DE evolutionary trainer with episodic migration.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm


@dataclass
class MigrationEvent:
    gen: int
    src_island: str
    dst_island: str
    slot_idx: int
    F_migrant: float
    F_displaced: float


@dataclass
class Island:
    """One algorithm-population pair in the 3-island model.

    `algorithm` is a pymoo Algorithm (PSO/GA/DE) whose `.pop` Population is
    mutated in-place by migration.
    """

    name: str
    algorithm: Any  # pymoo Algorithm or test stand-in (must expose .pop)
    last_validated_individual: npt.NDArray[np.float64] | None = None
    best_overall_individual: npt.NDArray[np.float64] | None = None
    best_overall_cost: float = float("inf")
    best_val_cost: float = float("inf")
    stagnation_counter: int = 0


def migrate(
    islands: list[Island],
    k_top: int,
    current_gen: int,
    rng: np.random.Generator,
    velocity_scale: float = 0.05,
) -> list[MigrationEvent]:
    """Apply full-mesh top-k -> worst-(k * (n_islands - 1)) migration in-place.

    Snapshots emigrants from every island BEFORE any in-place replacement
    (so a destination's incoming pool is not corrupted by its own outgoing).
    For PSO destinations, also resets V / pbest / pbest_F for the new slots
    via `inject_into_pso`.
    """
    # 1. Snapshot top-k emigrants from each island under current F.
    emigrants: dict[str, list[tuple[npt.NDArray[np.float64], float]]] = {}
    for src in islands:
        F_src = src.algorithm.pop.get("F").flatten()
        top_idx = np.argsort(F_src, kind="stable")[:k_top]
        emigrants[src.name] = [
            (src.algorithm.pop[int(i)].X.copy(), float(F_src[int(i)]))
            for i in top_idx
        ]

    # 2. For each destination, apply replacements from all other islands.
    events: list[MigrationEvent] = []
    for dst in islands:
        incoming: list[tuple[npt.NDArray[np.float64], float, str]] = []
        for src in islands:
            if src.name == dst.name:
                continue
            for X_em, F_em in emigrants[src.name]:
                incoming.append((X_em, F_em, src.name))

        n_incoming = len(incoming)
        F_dst = dst.algorithm.pop.get("F").flatten()
        worst_slots = np.argsort(F_dst, kind="stable")[-n_incoming:]

        for slot_i, (X_new, F_new, src_name) in zip(worst_slots, incoming, strict=True):
            slot = int(slot_i)
            F_displaced = float(F_dst[slot])
            dst.algorithm.pop[slot].X = X_new
            dst.algorithm.pop[slot].F = np.array([F_new])

            if isinstance(dst.algorithm, PSO):
                inject_into_pso(
                    dst.algorithm,
                    slot=slot,
                    X=X_new,
                    F=F_new,
                    velocity_scale=velocity_scale,
                    rng=rng,
                )

            events.append(
                MigrationEvent(
                    gen=current_gen,
                    src_island=src_name,
                    dst_island=dst.name,
                    slot_idx=slot,
                    F_migrant=F_new,
                    F_displaced=F_displaced,
                )
            )

    return events


def inject_into_pso(
    algorithm: Algorithm,
    slot: int,
    X: npt.NDArray[np.float64],
    F: float,
    velocity_scale: float,
    rng: np.random.Generator,
) -> None:
    """Write a fresh velocity and reset pbest/pbest_F for a migrant PSO slot.

    Zero velocity is a trap: a collapsed swarm's gbest will pull the migrant
    in within 2-3 ticks. A small uniform velocity gives the migrant a few
    ticks to evaluate its neighborhood and (if better than gbest) become the
    new attractor itself.
    """
    n_params = X.shape[0]
    V = algorithm.pop.get("V")
    pbest = algorithm.pop.get("pbest")
    pbest_F = algorithm.pop.get("pbest_F")

    V[slot] = rng.uniform(-velocity_scale, velocity_scale, size=n_params)
    pbest[slot] = X.copy()
    pbest_F[slot] = np.array([F])

    algorithm.pop.set("V", V)
    algorithm.pop.set("pbest", pbest)
    algorithm.pop.set("pbest_F", pbest_F)
