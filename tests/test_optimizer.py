import warnings

import pytest
from aerocapture.training.optimizer import (
    DESettings,
    GASettings,
    OptimizerConfig,
    PSOSettings,
    create_algorithm,
)
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM


class TestOptimizerConfig:
    def test_default_algorithm_is_ga(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.algorithm == "ga"

    def test_all_algorithms_accepted(self) -> None:
        for algo in ("ga", "cma_es", "de", "pso"):
            cfg = OptimizerConfig(algorithm=algo)
            assert cfg.algorithm == algo

    def test_invalid_algorithm_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            OptimizerConfig(algorithm="bees")

    def test_validation_interval_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="validation_interval must be > 0"):
            OptimizerConfig(validation_interval=0)

    def test_default_fields(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.n_pop == 60
        assert cfg.n_gen == 2500
        assert cfg.seed_pool_interval == 50
        assert cfg.adaptive_seeds is False
        assert cfg.seed_pool_cap == 100
        assert cfg.cost_alpha == 0.7
        assert cfg.cvar_percentile == 20
        assert cfg.stress_interval == 5
        assert cfg.stress_probes == 200
        assert cfg.stress_inject == 20
        assert cfg.training_n_sims == 1
        assert cfg.validation_n_sims == 1000
        assert cfg.validation_interval == 50

    def test_from_toml_dict_ga(self) -> None:
        d = {
            "algorithm": "ga",
            "n_pop": 80,
            "ga": {"crossover_eta": 10.0, "mutation_eta": 25.0},
        }
        cfg = OptimizerConfig.from_dict(d)
        assert cfg.algorithm == "ga"
        assert cfg.n_pop == 80
        assert cfg.ga.crossover_eta == 10.0
        assert cfg.ga.mutation_eta == 25.0

    def test_from_toml_dict_cma_es(self) -> None:
        d = {
            "algorithm": "cma_es",
            "cma_es": {"sigma0": 0.5, "restart_strategy": "bipop"},
        }
        cfg = OptimizerConfig.from_dict(d)
        assert cfg.algorithm == "cma_es"
        assert cfg.cma_es.sigma0 == 0.5
        assert cfg.cma_es.restart_strategy == "bipop"

    def test_defaults_when_subsection_missing(self) -> None:
        cfg = OptimizerConfig.from_dict({"algorithm": "de"})
        assert isinstance(cfg.de, DESettings)
        assert cfg.de.variant == "DE/rand/1/bin"
        assert isinstance(cfg.ga, GASettings)
        assert isinstance(cfg.pso, PSOSettings)

    def test_default_training_n_sims(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.training_n_sims == 1

    def test_default_validation_n_sims(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.validation_n_sims == 1000

    def test_default_validation_interval(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.validation_interval == 50

    def test_from_dict_training_n_sims(self) -> None:
        d = {"algorithm": "ga", "training_n_sims": 20}
        cfg = OptimizerConfig.from_dict(d)
        assert cfg.training_n_sims == 20

    def test_from_dict_validation_fields(self) -> None:
        d = {"algorithm": "ga", "validation_n_sims": 500, "validation_interval": 25}
        cfg = OptimizerConfig.from_dict(d)
        assert cfg.validation_n_sims == 500
        assert cfg.validation_interval == 25


class TestCreateAlgorithm:
    def test_ga_returns_ga(self) -> None:
        cfg = OptimizerConfig(algorithm="ga")
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg, GA)

    def test_cma_es_returns_cmaes(self) -> None:
        cfg = OptimizerConfig(algorithm="cma_es")
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg, CMAES)

    def test_de_returns_de(self) -> None:
        cfg = OptimizerConfig(algorithm="de")
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg, DE)

    def test_pso_returns_pso(self) -> None:
        cfg = OptimizerConfig(algorithm="pso")
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg, PSO)

    def test_cma_es_high_dim_warns_and_falls_back(self) -> None:
        cfg = OptimizerConfig(algorithm="cma_es")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            alg = create_algorithm(cfg, n_params=2500)
        assert isinstance(alg, GA)
        assert any("Falling back to GA" in str(w.message) for w in caught)

    def test_ga_uses_sbx_crossover(self) -> None:
        cfg = OptimizerConfig(algorithm="ga")
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg.mating.crossover, SBX)

    def test_ga_uses_polynomial_mutation(self) -> None:
        cfg = OptimizerConfig(algorithm="ga")
        alg = create_algorithm(cfg, n_params=10)
        assert isinstance(alg.mating.mutation, PM)

    def test_ga_mutation_prob_default(self) -> None:
        cfg = OptimizerConfig(algorithm="ga")
        alg = create_algorithm(cfg, n_params=20)
        assert alg.mating.mutation.prob.value == pytest.approx(0.15)

    def test_ga_explicit_mutation_prob(self) -> None:
        cfg = OptimizerConfig(algorithm="ga", ga=GASettings(mutation_prob=0.05))
        alg = create_algorithm(cfg, n_params=20)
        assert alg.mating.mutation.prob.value == pytest.approx(0.05)

    def test_ga_uses_configured_eta(self) -> None:
        cfg = OptimizerConfig(algorithm="ga", ga=GASettings(crossover_eta=25.0, mutation_eta=30.0))
        alg = create_algorithm(cfg, n_params=5)
        assert alg.mating.crossover.eta.value == 25.0
        assert alg.mating.mutation.eta.value == 30.0

    def test_pop_size_propagated(self) -> None:
        cfg = OptimizerConfig(algorithm="ga", n_pop=42)
        alg = create_algorithm(cfg, n_params=5)
        assert alg.pop_size == 42
