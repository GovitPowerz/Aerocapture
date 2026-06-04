"""run_grid bit-identity gates (Phase 4)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

aero = pytest.importorskip("aerocapture_rs")

EQGLIDE_TOML = "configs/training/msr_aller_eqglide_train.toml"


def _per_seed_run_batch(toml: str, overrides_list, seeds, *, nn_paths=None):
    """OLD path: loop seeds, run_batch per seed, return (n_pop, n_seeds, LEN)."""
    grid = []
    for seed in seeds:
        per = []
        for i, ovr in enumerate(overrides_list):
            o = dict(ovr)
            o["monte_carlo.seed"] = int(seed)
            o["simulation.n_sims"] = 1
            if nn_paths is not None:
                o["data.neural_network"] = str(nn_paths[i])
            per.append(o)
        res = aero.run_batch(toml, per, n_threads=None, include_trajectories=False)
        grid.append(np.asarray(res.final_records, dtype=np.float64))  # (n_pop, LEN)
    arr = np.stack(grid, axis=0)  # (n_seeds, n_pop, LEN)
    return np.transpose(arr, (1, 0, 2))  # (n_pop, n_seeds, LEN)


class TestRunGridSeedDrawIdentity:
    def test_non_nn_grid_bit_identical(self) -> None:
        # eqglide inherits common.toml -> dispersions + density_perturbation ON,
        # exercising the static draw AND the per-sim GM-RNG stream.
        seeds = list(range(7_000_000, 7_000_000 + 5))
        overrides_list = [
            {"guidance.equilibrium_glide.k_hdot_scale": 0.30},
            {"guidance.equilibrium_glide.k_hdot_scale": 0.35},
            {"guidance.equilibrium_glide.k_hdot_scale": 0.40},
        ]
        old = _per_seed_run_batch(EQGLIDE_TOML, overrides_list, seeds)
        new = np.asarray(aero.run_grid(EQGLIDE_TOML, overrides_list, seeds, n_threads=None))
        assert new.shape == old.shape
        np.testing.assert_array_equal(new, old)


class TestRunGridNnInMemory:
    def test_in_memory_weights_match_temp_json(self) -> None:
        # Self-contained: full-width (no mask) single Dense(35->2) full_neural NN
        # on a dispersion-enabled NN config. OLD writes temp JSON per individual;
        # NEW injects the same flat weights in-memory via run_grid.
        NN_INPUT = int(aero.NN_FULL_INPUT_SIZE)  # 35
        # Full-width mask: the runtime's *default* mask is the legacy [0..16], so a
        # 35-input Dense with no mask would be fed only 16 values (panic in BOTH
        # paths). An explicit all-35 mask routes every candidate input to the Dense
        # layer and is embedded in the model identically by both paths.
        full_mask = list(range(NN_INPUT))
        arch = [
            {"type": "dense", "input_size": NN_INPUT, "output_size": 2, "activation": "linear"}
        ]
        arch_json = json.dumps(arch)
        n_w = NN_INPUT * 2 + 2

        # Minimal dispersion-enabled NN config. No [network] section: atan2_signed
        # needs no architecture constraint, the runtime model comes from the temp
        # JSON (OLD, with the mask baked in by flat_weights_to_json) or the injected
        # weights (NEW, with the mask passed to run_grid), so the model is used
        # verbatim in both paths. common.toml supplies dispersions + density_perturbation.
        cfg = (
            'base = ["../missions/mars.toml", "common.toml"]\n'
            "[guidance]\n"
            'type = "neural_network"\n'
            "[guidance.neural_network]\n"
            'mode = "full_neural"\n'
        )
        cfg_dir = Path("configs/training")  # so relative base paths resolve
        with tempfile.NamedTemporaryFile("w", suffix=".toml", dir=cfg_dir, delete=False) as f:
            f.write(cfg)
            cfg_path = f.name

        rng = np.random.default_rng(123)
        n_pop = 3
        weights = (rng.standard_normal((n_pop, n_w)) * 0.1).astype(np.float64)
        seeds = list(range(7_100_000, 7_100_000 + 4))
        overrides_list = [{} for _ in range(n_pop)]

        nn_paths = []
        try:
            for i in range(n_pop):
                fd, p = tempfile.mkstemp(suffix=".json", prefix=f"nn_grid_{i}_")
                os.close(fd)
                aero.flat_weights_to_json(
                    flat=weights[i].tolist(),
                    architecture_json=arch_json,
                    path=p,
                    input_mask=full_mask,
                    output_param="atan2_signed",
                )
                nn_paths.append(p)

            old = _per_seed_run_batch(cfg_path, overrides_list, seeds, nn_paths=nn_paths)
            new = np.asarray(
                aero.run_grid(
                    cfg_path,
                    overrides_list,
                    seeds,
                    weights=weights,
                    architecture_json=arch_json,
                    input_mask=full_mask,
                    output_param="atan2_signed",
                    n_threads=None,
                )
            )
            assert new.shape == old.shape == (n_pop, len(seeds), int(aero.FINAL_RECORD_LEN))
            np.testing.assert_array_equal(new, old)
        finally:
            for p in nn_paths:
                Path(p).unlink(missing_ok=True)
            Path(cfg_path).unlink(missing_ok=True)
