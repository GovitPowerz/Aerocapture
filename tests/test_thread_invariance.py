"""run_grid thread-count invariance: the (n_pop, n_seeds, LEN) grid is byte-identical
whatever the Rayon pool size. Extends ADR-0004's bit-identity claim (run_grid ==
per-seed run_batch) to the parallel axis: work stealing may run the cells in any
order on any thread, and no cell may see another's state."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.layer_schema import layer_n_params
from aerocapture.training.toml_utils import load_toml_with_bases

aero = pytest.importorskip("aerocapture_rs")

EQGLIDE_TOML = "configs/training/msr_aller_eqglide_train.toml"
FNPAG_TOML = "configs/training/msr_aller_fnpag_train.toml"
MAMBA_TOML = "configs/training/sweep/mamba_p962.toml"
DENSE_TOML = "configs/training/sweep/dense_p515.toml"
MAMBA_MODEL = "articles/paper/data/runs/headline/mamba_p962/best_model.json"

SEEDS = list(range(7_200_000, 7_200_000 + 6))
# 1 is the reference; 3 makes cells straddle threads unevenly; None is the global
# pool, the call the training loop actually makes.
THREAD_COUNTS: list[int | None] = [3, os.cpu_count() or 1, None]

CASES: dict[str, tuple[str, list[dict[str, object]]]] = {
    # common.toml: dispersions + Gauss-Markov density perturbation + EKF noise streams.
    "eqglide_fixed_step": (EQGLIDE_TOML, [{"guidance.equilibrium_glide.k_hdot_scale": k} for k in (0.30, 0.35, 0.40)]),
    "eqglide_adaptive_events": (
        EQGLIDE_TOML,
        [{"guidance.equilibrium_glide.k_hdot_scale": k, "integration.mode": "adaptive"} for k in (0.30, 0.40)],
    ),
    "fnpag_predictor": (FNPAG_TOML, [{"navigation.density_filter_gain": g} for g in (0.3, 0.6)]),
    "mamba_stateful_nn": (
        MAMBA_TOML,
        [{"data.neural_network": MAMBA_MODEL, "guidance.command_shaping.max_bank_acceleration": a} for a in (8.0, 12.0)],
    ),
}


def _assert_thread_invariant(toml: str, overrides_list: list[dict[str, object]], **nn_kwargs: object) -> None:
    ref = np.asarray(aero.run_grid(toml, overrides_list, SEEDS, n_threads=1, **nn_kwargs), dtype=np.float64)
    assert ref.shape == (len(overrides_list), len(SEEDS), int(aero.FINAL_RECORD_LEN))
    # Non-degenerate: every cell flew its own trajectory, so a pool that swapped or
    # shared cells could not hide behind identical rows.
    rows = ref.reshape(-1, ref.shape[-1])
    assert len({r.tobytes() for r in rows}) == rows.shape[0]
    for n in THREAD_COUNTS:
        got = np.asarray(aero.run_grid(toml, overrides_list, SEEDS, n_threads=n, **nn_kwargs), dtype=np.float64)
        assert got.tobytes() == ref.tobytes(), f"run_grid differs at n_threads={n} vs n_threads=1"


@pytest.mark.parametrize("case", list(CASES))
def test_run_grid_byte_identical_across_thread_counts(case: str) -> None:
    toml, overrides_list = CASES[case]
    _assert_thread_invariant(toml, overrides_list)


def test_in_memory_nn_weights_byte_identical_across_thread_counts() -> None:
    # The training hot path: per-individual weights injected in-memory, models built
    # inside the parallel SimData construction (problem.py::_run_grid_records).
    network = load_toml_with_bases(Path(DENSE_TOML))["network"]
    arch = network["architecture"]
    n_w = sum(layer_n_params(layer) for layer in arch)
    weights = np.random.default_rng(114).standard_normal((3, n_w)) * 0.1
    _assert_thread_invariant(
        DENSE_TOML,
        [{} for _ in range(weights.shape[0])],
        weights=weights,
        architecture_json=json.dumps(arch),
        input_mask=network["input_mask"],
        output_param="atan2_signed",
    )
