"""Chromosome evaluation glue: the validation gate and the NN / guidance parameter writers.

Seed pools live in `seeds.py`, the cost function in `cost.py`, the TOML writer in
`toml_utils.py`; this module keeps what turns a decoded individual into something the
simulator can run (`write_nn_json`, `write_guidance_toml`) and the in-training
`run_validation_gate`.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import NetworkConfig

try:
    import aerocapture_rs as _aero_rs  # type: ignore[import-not-found, import-untyped]

    _HAS_PYO3 = True
except ImportError:
    _aero_rs = None  # type: ignore[assignment]
    _HAS_PYO3 = False


class GateStatus(Enum):
    """Outcome of the guarded validation-gate selection."""

    SKIP_ALL_INF = "skip_all_inf"  # no finite training cost -> do not select/promote
    SKIP_UNCHANGED = "skip_unchanged"  # argmin identical to last validated -> no re-validate
    VALIDATED = "validated"  # ran validation MC; `promoted` says whether to swap best


@dataclass
class GateResult:
    """Result of `run_validation_gate`. Callers apply their own state updates
    (stagnation counters, result-dict shaping, no-validation fallback)."""

    status: GateStatus
    argmin_cost: float
    individual: npt.NDArray[np.float64] | None = None
    val_costs: npt.NDArray[np.float64] | None = None
    val_records: npt.NDArray[np.float64] | None = None
    val_rms: float | None = None
    promoted: bool = False


class _SupportsPerSeedEval(Protocol):
    def evaluate_individual_records_per_seed(
        self,
        x: npt.NDArray[np.float64],
        seeds: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


def run_validation_gate(
    X: npt.NDArray[np.float64],
    F: npt.NDArray[np.float64],
    last_validated: npt.NDArray[np.float64] | None,
    best_val_cost: float,
    problem: _SupportsPerSeedEval,
    val_seeds: list[int],
) -> GateResult:
    """Guarded gen-best selection + identity-trigger validation, shared by the
    single-algorithm loop and the islands trainer.

    Decides three things and nothing else:
      1. SKIP_ALL_INF when every training cost is non-finite (the guard -- a bare
         `np.argmin` on an all-inf array returns 0, which would promote whatever
         junk chromosome sits at pop[0]).
      2. SKIP_UNCHANGED when the guarded argmin matches `last_validated` (no point
         re-running the same individual through the validation MC).
      3. VALIDATED otherwise: runs the validation MC and reports `val_rms` plus
         whether it beats `best_val_cost` (the promotion boolean).

    Selection uses `nanargmin(where(isfinite, f, inf))`, which also skips NaN-cost
    rows: in a mixed finite+NaN population the finite minimum is selected, not the
    NaN-indexed individual that bare `np.argmin` would have returned.  This matches
    the islands trainer's already-NaN-safe selection and closes a latent single-algo
    bug where a Rust sim leaking a NaN state (e.g. no sim_timeout_secs) could
    otherwise cause a NaN-cost individual to be validated and promoted.

    It does NOT manage stagnation counters, build result/summary dicts, or run the
    no-validation fallback -- those stay caller-side so each caller's observable
    behavior (result-dict shape, logging payload) is unchanged.
    """
    f = np.asarray(F).reshape(-1)
    if not np.any(np.isfinite(f)):
        return GateResult(status=GateStatus.SKIP_ALL_INF, argmin_cost=float("inf"))

    idx = int(np.nanargmin(np.where(np.isfinite(f), f, np.inf)))
    individual = X[idx].copy()
    argmin_cost = float(f[idx])

    if last_validated is not None and np.array_equal(individual, last_validated):
        return GateResult(status=GateStatus.SKIP_UNCHANGED, argmin_cost=argmin_cost, individual=individual)

    val_costs, val_records = problem.evaluate_individual_records_per_seed(individual, val_seeds)
    val_rms = float(np.sqrt(np.mean(val_costs**2)))
    return GateResult(
        status=GateStatus.VALIDATED,
        argmin_cost=argmin_cost,
        individual=individual,
        val_costs=val_costs,
        val_records=val_records,
        val_rms=val_rms,
        promoted=val_rms < best_val_cost,
    )


def build_v2_architecture(network: NetworkConfig) -> list[dict[str, object]]:
    """The v2 LayerSpec list for a NetworkConfig: the explicit architecture when
    present, else a dense chain synthesized from layer_sizes/activations. Single
    source of truth for the architecture JSON used by write_nn_json and run_grid.
    """
    if network.architecture is not None:
        return [dict(entry) for entry in network.architecture]
    arch: list[dict[str, object]] = []
    for i in range(len(network.layer_sizes) - 1):
        arch.append(
            {
                "type": "dense",
                "input_size": network.layer_sizes[i],
                "output_size": network.layer_sizes[i + 1],
                "activation": network.activations[i],
            }
        )
    return arch


def write_nn_json(
    weights: npt.NDArray[np.float64],
    network: NetworkConfig,
    filepath: str | Path,
    input_mask: list[int] | None = None,
    output_param: str | None = None,
    normalization: list[dict] | None = None,
) -> None:
    """Write PSO chromosome weights as v2 NN JSON via the Rust LayerWeights trait.

    Routes through `aerocapture_rs.flat_weights_to_json` so the Rust side is the
    single source of truth for weight serialization (closes Phase 0 review
    carry-over #2). Legacy dense-only `NetworkConfig` is translated into a v2
    architecture list before the call.

    When `normalization` is provided (the config's `[network.normalization]`
    override, a list of NN_FULL_INPUT_SIZE `{transform, scale, center}` dicts),
    it is embedded into the written model so the deployed JSON is self-describing
    and matches the scales the model trained under. When None, the model keeps
    the baked `DEFAULT_NORMALIZATION` (backward-compatible).
    """
    if not _HAS_PYO3 or _aero_rs is None:
        raise RuntimeError(
            "write_nn_json now requires the aerocapture_rs PyO3 module. "
            "Build it with `maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`."
        )

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    arch = build_v2_architecture(network)
    flat = weights.astype(np.float64)
    qat_bits = getattr(network, "qat_bits", None)
    if qat_bits is not None:
        from aerocapture.training.quantize import quantize_flat_weights_batch

        flat = quantize_flat_weights_batch(flat.reshape(1, -1), arch, qat_bits, network.qat_granularity, network.qat_tensor_policy)[0]
    _aero_rs.flat_weights_to_json(
        flat=flat.tolist(),
        architecture_json=json.dumps(arch),
        path=str(filepath),
        input_mask=input_mask,
        output_param=output_param,
        scaled_pi_n=getattr(network, "scaled_pi_n", 1.0),
        delta_max=getattr(network, "delta_max", 0.35),
        normalization_json=json.dumps(normalization) if normalization is not None else None,
    )


def _parse_final_to_legacy_array(filepath: Path) -> npt.NDArray[np.float64] | None:
    """Parse a final conditions CSV file, returning 0-based 52-column array.

    Maps named CSV columns to their xsauve indices (0-based, no sim_number prefix).
    """
    import pandas as pd

    from aerocapture.io.parse_final import CSV_TO_LEGACY_INDEX

    df = pd.read_csv(filepath)
    if df.empty:
        return None
    n = len(df)
    result = np.zeros((n, 52))
    for col_name, legacy_idx in CSV_TO_LEGACY_INDEX.items():
        if col_name in df.columns:
            result[:, legacy_idx] = df[col_name].to_numpy()
    return result


def write_guidance_toml(
    base_toml_path: str | Path,
    guidance_type: str,
    params: dict[str, float],
    output_path: str | Path | None = None,
    mc_seed: int | None = None,
    n_sims_override: int | None = None,
) -> Path:
    """Patch a base TOML config with optimized guidance parameters.

    Reads the base TOML, adds/overwrites the [guidance.<section>] with
    the provided parameter values, and writes to output_path (or a temp file).

    Returns:
        Path to the written TOML file.
    """
    from aerocapture.training.deploy_overrides import overrides_from_params
    from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS
    from aerocapture.training.toml_utils import load_toml_with_bases, set_dot_path, write_toml

    base_toml_path = Path(base_toml_path)
    toml_data = load_toml_with_bases(base_toml_path)

    # Set the guidance type
    toml_data.setdefault("guidance", {})["type"] = guidance_type

    # The scheme's own section always exists in the written file, even when
    # every tuned param is a scaffolding one (legacy contract of this writer).
    toml_data["guidance"].setdefault(GUIDANCE_TOML_SECTIONS[guidance_type], {})
    for dot_path, value in overrides_from_params(params, guidance_type).items():
        set_dot_path(toml_data, dot_path, value)

    if mc_seed is not None:
        toml_data.setdefault("monte_carlo", {})["seed"] = mc_seed

    if n_sims_override is not None:
        toml_data.setdefault("simulation", {})["n_sims"] = n_sims_override

    # Write TOML (minimal writer -- machine-consumed only)
    if output_path is None:
        fd, path_str = tempfile.mkstemp(suffix=".toml", prefix="guidance_")
        output_path = Path(path_str)
        import os

        os.close(fd)
    else:
        output_path = Path(output_path)

    write_toml(toml_data, output_path)
    return output_path
