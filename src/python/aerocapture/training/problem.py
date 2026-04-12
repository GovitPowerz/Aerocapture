"""pymoo Problem subclass bridging population-level evaluation with the Rust simulator."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pymoo.core.problem import Problem

from aerocapture.training.encoding import decode_normalized_array
from aerocapture.training.evaluate import compute_cost
from aerocapture.training.param_spaces import ParamSpec

try:
    import aerocapture_rs as _aero_rs  # type: ignore[import-not-found, import-untyped]

    _HAS_PYO3 = True
except ImportError:
    _aero_rs = None  # type: ignore[assignment]
    _HAS_PYO3 = False


class AerocaptureProblem(Problem):
    """pymoo Problem for aerocapture guidance parameter optimization.

    Works on normalized [0, 1] decision variables; decoding to physical
    values happens inside _evaluate via decode_normalized_array.
    """

    def __init__(
        self,
        param_specs: list[ParamSpec],
        toml_path: str,
        seeds: list[int],
        cost_kwargs: dict[str, float],
        scheme: str,
        sim_timeout: float | None = None,
        nn_config: object | None = None,
        n_sims_override: int | None = None,
    ) -> None:
        super().__init__(n_var=len(param_specs), n_obj=1, xl=0.0, xu=1.0)
        self.param_specs = param_specs
        self.toml_path = toml_path
        self.seeds = seeds
        self.cost_kwargs = cost_kwargs
        self.scheme = scheme
        self.sim_timeout = sim_timeout
        self.nn_config = nn_config
        self.n_sims_override = n_sims_override

    def update_seeds(self, seeds: list[int]) -> None:
        self.seeds = seeds

    def _evaluate(self, X: npt.NDArray[np.float64], out: dict, *args, **kwargs) -> None:  # type: ignore[override]
        costs = self._run_batch(X)
        out["F"] = costs.reshape(-1, 1)

    def _run_batch(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Decode population, run simulator for each seed, aggregate costs via RMS."""
        n_pop = X.shape[0]
        param_dicts = decode_normalized_array(X, self.param_specs)

        if _HAS_PYO3 and _aero_rs is not None:
            return self._run_batch_pyo3(_aero_rs, param_dicts, n_pop)

        raise NotImplementedError("PyO3 aerocapture_rs module is required for batch evaluation")

    def _run_batch_pyo3(
        self,
        aero_rs: object,
        param_dicts: list[dict[str, float]],
        n_pop: int,
    ) -> npt.NDArray[np.float64]:
        """Evaluate population via PyO3 run_batch, one call per seed, aggregate by RMS."""
        seed_costs: list[npt.NDArray[np.float64]] = []

        for seed in self.seeds:
            overrides_list = [self._build_overrides(p, mc_seed=seed) for p in param_dicts]
            result = aero_rs.run_batch(  # type: ignore[union-attr]
                self.toml_path,
                overrides_list,
                n_threads=None,
                include_trajectories=False,
                sim_timeout_secs=self.sim_timeout,
            )
            final_records = result.final_records  # list of (52,) arrays
            per_run_costs = np.array(
                [compute_cost(fr.reshape(1, 52), **self.cost_kwargs) for fr in final_records],
                dtype=np.float64,
            )
            seed_costs.append(per_run_costs)

        # RMS across seeds for each individual
        stacked = np.stack(seed_costs, axis=0)  # (n_seeds, n_pop)
        return np.sqrt(np.mean(stacked**2, axis=0))

    def _build_overrides(
        self,
        params: dict[str, float],
        mc_seed: int | None = None,
    ) -> dict[str, object]:
        """Route param dict to TOML dot-path overrides."""
        overrides: dict[str, object] = {}

        for key, value in params.items():
            if key.startswith("lateral."):
                overrides[f"guidance.lateral.{key.removeprefix('lateral.')}"] = value
            elif key.startswith("exit."):
                overrides[f"guidance.ftc.{key.removeprefix('exit.')}"] = value
            elif key.startswith("nav."):
                overrides[f"navigation.{key.removeprefix('nav.')}"] = value
            elif key.startswith("thermal."):
                overrides[f"guidance.thermal_limiter.{key.removeprefix('thermal.')}"] = value
            elif key.startswith("shaping."):
                overrides[f"guidance.command_shaping.{key.removeprefix('shaping.')}"] = value
            else:
                overrides[f"guidance.{self.scheme}.{key}"] = value

        if mc_seed is not None:
            overrides["monte_carlo.seed"] = mc_seed

        if self.n_sims_override is not None:
            overrides["monte_carlo.n_sims"] = self.n_sims_override

        return overrides
