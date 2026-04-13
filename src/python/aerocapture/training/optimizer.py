from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

_VALID_ALGORITHMS = ("ga", "cma_es", "de", "pso")
_CMAES_MAX_PARAMS = 200


@dataclass
class GASettings:
    crossover_eta: float = 3.0
    mutation_eta: float = 5.0
    mutation_prob: float | None = 0.15


@dataclass
class CMAESSettings:
    sigma0: float = 0.3
    restart_strategy: str = "ipop"


@dataclass
class DESettings:
    variant: str = "DE/rand/1/bin"
    crossover_prob: float = 0.8
    scaling_factor: float = 0.6


@dataclass
class PSOSettings:
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5


@dataclass
class OptimizerConfig:
    algorithm: str = "ga"
    n_pop: int = 60
    n_gen: int = 2500
    seed_pool_interval: int = 50
    adaptive_seeds: bool = False
    seed_pool_cap: int = 100
    cost_alpha: float = 0.7
    cvar_percentile: int = 20
    stress_interval: int = 5
    stress_probes: int = 200
    stress_inject: int = 20
    training_n_sims: int = 1
    validation_n_sims: int = 1000
    validation_interval: int = 50
    ga: GASettings = field(default_factory=GASettings)
    cma_es: CMAESSettings = field(default_factory=CMAESSettings)
    de: DESettings = field(default_factory=DESettings)
    pso: PSOSettings = field(default_factory=PSOSettings)

    def __post_init__(self) -> None:
        if self.algorithm not in _VALID_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'. Must be one of: {_VALID_ALGORITHMS}")
        if self.validation_interval <= 0:
            raise ValueError(f"validation_interval must be > 0, got {self.validation_interval}")

    @classmethod
    def from_dict(cls, d: dict) -> OptimizerConfig:
        ga = GASettings(**d["ga"]) if "ga" in d else GASettings()
        cma_es = CMAESSettings(**d["cma_es"]) if "cma_es" in d else CMAESSettings()
        de = DESettings(**d["de"]) if "de" in d else DESettings()
        pso = PSOSettings(**d["pso"]) if "pso" in d else PSOSettings()

        top_level = {k: v for k, v in d.items() if k not in ("ga", "cma_es", "de", "pso")}
        return cls(**top_level, ga=ga, cma_es=cma_es, de=de, pso=pso)


def create_algorithm(config: OptimizerConfig, n_params: int) -> Algorithm:
    algorithm = config.algorithm

    if algorithm == "cma_es" and n_params > _CMAES_MAX_PARAMS:
        warnings.warn(
            f"CMA-ES is not recommended for n_params={n_params} > {_CMAES_MAX_PARAMS}. Falling back to GA.",
            UserWarning,
            stacklevel=2,
        )
        algorithm = "ga"

    if algorithm == "ga":
        ga = config.ga
        mut_prob = ga.mutation_prob if ga.mutation_prob is not None else 1.0 / n_params
        return GA(
            pop_size=config.n_pop,
            crossover=SBX(eta=ga.crossover_eta),
            mutation=PM(eta=ga.mutation_eta, prob=mut_prob),
        )

    if algorithm == "cma_es":
        cma = config.cma_es
        return CMAES(
            pop_size=config.n_pop,
            sigma=cma.sigma0,
            restarts=1 if cma.restart_strategy == "ipop" else 0,
        )

    if algorithm == "de":
        de = config.de
        return DE(
            pop_size=config.n_pop,
            variant=de.variant,
            CR=de.crossover_prob,
            F=de.scaling_factor,
        )

    if algorithm == "pso":
        pso = config.pso
        return PSO(
            pop_size=config.n_pop,
            w=pso.w,
            c1=pso.c1,
            c2=pso.c2,
        )

    raise ValueError(f"Unhandled algorithm: {algorithm}")  # unreachable
