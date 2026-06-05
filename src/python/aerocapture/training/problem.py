"""pymoo Problem subclass bridging population-level evaluation with the Rust simulator."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from pymoo.core.problem import Problem

from aerocapture.training.encoding import decode_normalized_array
from aerocapture.training.evaluate import compute_cost
from aerocapture.training.param_spaces import SCAFFOLDING_PREFIXES, ParamSpec, active_scaffolding_specs, route_param_path
from aerocapture.training.parquet_output import FINAL_RECORD_LEN

_MAX_CONSECUTIVE_EVAL_FAILURES = 5

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
        cost_kwargs: dict[str, Any],
        scheme: str,
        sim_timeout: float | None = None,
        nn_config: object | None = None,
    ) -> None:
        super().__init__(n_var=len(param_specs), n_obj=1, xl=0.0, xu=1.0)
        self.param_specs = param_specs
        self.toml_path = toml_path
        self.seeds = seeds
        self.cost_kwargs = cost_kwargs
        self.scheme = scheme
        self.sim_timeout = sim_timeout
        self.nn_config = nn_config
        self._integer_params = {s.name for s in param_specs if s.is_integer}
        self._consecutive_eval_failures = 0

        # NN scaffolding: chromosome layout is [NN weights..., scaffolding tail...].
        # _n_nn_weight_specs caps the NN-weight slice so run_grid gets a weights
        # matrix with exactly the right column count. Tail width = len(active scaffolding pack).
        _scaffolding = getattr(nn_config, "scaffolding", "off") if nn_config is not None else "off"
        self._n_nn_weight_specs = len(param_specs) - len(active_scaffolding_specs(_scaffolding))

    def update_seeds(self, seeds: list[int]) -> None:
        self.seeds = seeds

    def _evaluate(self, X: npt.NDArray[np.float64], out: dict, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        try:
            costs = self._run_batch(X)
            self._consecutive_eval_failures = 0
        except Exception as e:
            self._consecutive_eval_failures = getattr(self, "_consecutive_eval_failures", 0) + 1
            print(
                f"  [problem] batch eval failed ({type(e).__name__}: {e}); penalizing 1e9 (consecutive failures: {self._consecutive_eval_failures})",
                file=sys.stderr,
            )
            if self._consecutive_eval_failures >= _MAX_CONSECUTIVE_EVAL_FAILURES:
                raise RuntimeError(f"{self._consecutive_eval_failures} consecutive batch-eval failures; aborting (last: {type(e).__name__}: {e})") from e
            costs = np.full(X.shape[0], 1e9)
        out["F"] = costs.reshape(-1, 1)

    def _run_batch(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Decode population, run simulator for each seed, aggregate costs via RMS."""
        if _HAS_PYO3 and _aero_rs is not None:
            return self._run_batch_pyo3(X)
        raise NotImplementedError("PyO3 aerocapture_rs module is required for batch evaluation")

    def _run_batch_pyo3(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the population via one GIL-releasing run_grid call (all seeds),
        aggregating costs by RMS across the seed axis (bit-identical to the old
        per-seed run_batch loop)."""
        n_pop = X.shape[0]
        grid = self._run_grid_records(X, self.seeds)  # (n_pop, n_seeds, rec)
        n_seeds = len(self.seeds)
        costs = np.empty((n_pop, n_seeds), dtype=np.float64)
        for i in range(n_pop):
            for k in range(n_seeds):
                costs[i, k] = compute_cost(grid[i, k].reshape(1, FINAL_RECORD_LEN), **self.cost_kwargs)
        rms: npt.NDArray[np.float64] = np.sqrt(np.mean(costs**2, axis=1))
        return rms

    def _run_grid_records(
        self,
        X_rows: npt.NDArray[np.float64],
        seeds: list[int],
    ) -> npt.NDArray[np.float64]:
        """Shared kernel: run_grid for an arbitrary n_pop of rows and a seed list.

        Returns (n_pop, n_seeds, FINAL_RECORD_LEN) float64 array.
        Handles both NN (in-memory weights) and non-NN (overrides-only) schemes.
        """
        assert _HAS_PYO3 and _aero_rs is not None
        param_dicts = decode_normalized_array(X_rows, self.param_specs)
        overrides_list = [self._build_grid_overrides(p) for p in param_dicts]

        weights = None
        architecture_json = None
        input_mask = None
        output_param = None
        scaled_pi_n = None
        delta_max = None
        if self.scheme == "neural_network" and self.nn_config is not None:
            from aerocapture.training.config import NetworkConfig
            from aerocapture.training.evaluate import build_v2_architecture

            nn_cfg = self.nn_config
            assert isinstance(nn_cfg, NetworkConfig)
            n_w = self._n_nn_weight_specs
            lo = np.array([self.param_specs[j].p_min for j in range(n_w)], dtype=np.float64)
            hi = np.array([self.param_specs[j].p_max for j in range(n_w)], dtype=np.float64)
            weights = (lo + X_rows[:, :n_w] * (hi - lo)).astype(np.float64)
            architecture_json = json.dumps(build_v2_architecture(nn_cfg))
            input_mask = nn_cfg.input_mask
            output_param = nn_cfg.output_parameterization
            scaled_pi_n = getattr(nn_cfg, "scaled_pi_n", 1.0)
            delta_max = getattr(nn_cfg, "delta_max", 0.35)

        grid = np.asarray(
            _aero_rs.run_grid(  # type: ignore[union-attr, attr-defined]
                self.toml_path,
                overrides_list,
                [int(s) for s in seeds],
                weights=weights,
                architecture_json=architecture_json,
                input_mask=input_mask,
                output_param=output_param,
                scaled_pi_n=scaled_pi_n,
                delta_max=delta_max,
                n_threads=None,
                sim_timeout_secs=self.sim_timeout,
            ),
            dtype=np.float64,
        )  # (n_pop, n_seeds, FINAL_RECORD_LEN)
        return grid

    def evaluate_individual_per_seed(
        self,
        x: npt.NDArray[np.float64],
        seeds: list[int],
    ) -> npt.NDArray[np.float64]:
        """Evaluate a single individual on a list of seeds, returning per-seed costs.

        Args:
            x: Normalized [0,1] individual vector (n_params,).
            seeds: List of MC seeds to evaluate on.

        Returns:
            1D array of costs (n_seeds,).
        """
        grid = self._run_grid_records(x.reshape(1, -1), seeds)  # (1, n_seeds, rec)
        records = grid[0]  # (n_seeds, FINAL_RECORD_LEN)
        costs = np.array(
            [compute_cost(records[k].reshape(1, FINAL_RECORD_LEN), **self.cost_kwargs) for k in range(len(seeds))],
            dtype=np.float64,
        )
        return costs

    def evaluate_individual_records_per_seed(
        self,
        x: npt.NDArray[np.float64],
        seeds: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Evaluate a single individual on a list of seeds, returning per-seed
        costs AND the full (n_seeds, 52) `final_records` matrix.

        Same compute path as `evaluate_individual_per_seed`, but exposes the
        raw records so downstream consumers (e.g. report.print_eval_summary)
        can derive DV / apoapsis / heat-flux statistics without re-running
        the MC.
        """
        grid = self._run_grid_records(x.reshape(1, -1), seeds)  # (1, n_seeds, rec)
        final_records = grid[0]  # (n_seeds, FINAL_RECORD_LEN)
        costs = np.array(
            [compute_cost(final_records[k].reshape(1, FINAL_RECORD_LEN), **self.cost_kwargs) for k in range(len(seeds))],
            dtype=np.float64,
        )
        return costs, final_records

    def _build_overrides(
        self,
        params: dict[str, float],
        mc_seed: int | None = None,
    ) -> dict[str, object]:
        """Route param dict to TOML dot-path overrides.

        For neural_network, NN weight keys (anything not matching one of the
        scaffolding routing prefixes) are skipped here; they are carried in-memory
        via _run_grid_records. The whitelist must be a positive
        match against the routing-prefix set, not a heuristic startswith()
        check -- v2 stateful-layer weight names (b_ih*, x_proj_w*, a_log*,
        ln1_gamma*, b_q*, ...) don't all start with "w" or "bias" and would
        leak as phantom `guidance.neural_network.<weight>` overrides if the
        skip filter were a denylist.
        """
        overrides: dict[str, object] = {}

        # For NN schemes, anything that isn't a scaffolding param (lateral,
        # exit, nav, thermal, shaping) is an NN weight and skipped — carried
        # in-memory by run_grid.
        skip_nn_weights = self.scheme == "neural_network" and self.nn_config is not None

        for key, value in params.items():
            if skip_nn_weights and not key.startswith(SCAFFOLDING_PREFIXES):
                continue
            # Round integer-typed params so Rust TOML parser accepts them
            if key in self._integer_params:
                value = int(round(value))
            overrides[route_param_path(key, self.scheme)] = value

        if mc_seed is not None:
            overrides["monte_carlo.seed"] = mc_seed

        # Always n_sims=1: each run_batch call evaluates one individual on one seed.
        overrides["simulation.n_sims"] = 1

        return overrides

    def _build_grid_overrides(self, params: dict[str, float]) -> dict[str, object]:
        """Route param dict to TOML dot-path overrides for run_grid (no seed /
        n_sims keys -- run_grid owns the seed axis and runs one sim per cell)."""
        overrides: dict[str, object] = {}
        skip_nn_weights = self.scheme == "neural_network" and self.nn_config is not None
        for key, value in params.items():
            if skip_nn_weights and not key.startswith(SCAFFOLDING_PREFIXES):
                continue
            if key in self._integer_params:
                value = int(round(value))
            overrides[route_param_path(key, self.scheme)] = value
        return overrides
