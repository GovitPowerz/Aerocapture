"""Three-island PSO/GA/DE evolutionary trainer with episodic migration.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm

from aerocapture.training.optimizer import OptimizerConfig, create_algorithm


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
    For PSO destinations, also resets V and the personal-best slot
    (pop[slot].X / .F) via `inject_into_pso`.
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
    """Place a migrant into a PSO slot: fresh velocity + pbest + current swarm pos.

    pymoo PSO maintains two Population objects:
    - `algorithm.pop` is the PERSONAL-BEST population (`pop[i].X` is particle i's pbest).
    - `algorithm.particles` is the CURRENT swarm position (identical to `pop` on
      iteration 1, divergent after).

    We must update both so the migrant lands at the slot in both reference frames.
    Velocity gets a small uniform random kick: zero velocity is a trap because a
    collapsed swarm's gbest will pull the migrant in within 2-3 ticks; the kick
    gives 2-3 ticks of independent motion for the migrant to assert its own F.
    """
    n_params = X.shape[0]
    particles = getattr(algorithm, "particles", algorithm.pop)

    # Velocity injection.
    V = particles.get("V")
    V[slot] = rng.uniform(-velocity_scale, velocity_scale, size=n_params)
    particles.set("V", V)

    # Current swarm position (particles[slot]).
    particles[slot].X = X.copy()
    particles[slot].F = np.array([F])

    # Personal best (pop[slot] in pymoo PSO).
    algorithm.pop[slot].X = X.copy()
    algorithm.pop[slot].F = np.array([F])


_ISLAND_NAMES = ("pso", "ga", "de")


def _build_island(
    name: str,
    config: OptimizerConfig,
    n_params: int,
) -> Island:
    sub_config = deepcopy(config)
    sub_config.algorithm = name
    algorithm = create_algorithm(sub_config, n_params=n_params)
    return Island(name=name, algorithm=algorithm)


class IslandModel:
    """Owns 3 islands (PSO, GA, DE) and the migration / validation / final-eval flow."""

    def __init__(
        self,
        config: OptimizerConfig,
        problem: Any,
        n_params: int,
        validation_seeds: list[int],
        final_eval_seeds: list[int],
        base_mc_seed: int,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.problem = problem
        self.n_params = n_params
        self.validation_seeds = validation_seeds
        self.final_eval_seeds = final_eval_seeds
        self.base_mc_seed = base_mc_seed
        self.rng = rng
        self.islands: list[Island] = [
            _build_island(name, config, n_params) for name in _ISLAND_NAMES
        ]
        self.migration_log: list[MigrationEvent] = []

    def final_eval(self) -> list[dict[str, Any]]:
        """Re-evaluate each island's best_overall on the reserved final-eval seeds.

        Returns one record per island that had a validated best, sorted by rms ascending.
        The lowest-rms record is the winner.
        """
        results: list[dict[str, Any]] = []
        for island in self.islands:
            if island.best_overall_individual is None:
                continue
            costs = self.problem.evaluate_individual_per_seed(
                island.best_overall_individual,
                self.final_eval_seeds,
            )
            rms = float(np.sqrt(np.mean(costs**2)))
            results.append({
                "island": island.name,
                "X": island.best_overall_individual.copy(),
                "rms": rms,
                "mean": float(np.mean(costs)),
                "p95": float(np.percentile(costs, 95)),
                "capture_rate": float(np.mean(np.asarray(costs) < 1000.0)),
                "n_sims": len(self.final_eval_seeds),
            })
        results.sort(key=lambda r: r["rms"])
        return results
