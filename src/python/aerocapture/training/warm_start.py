"""Behavioural-cloning warm-start for NN guidance training.

Runs a non-NN scheme (default FTC) over a reserved seed pool, collects
(state, |bank|) pairs via aerocapture_rs.collect_supervised, supervised
pre-trains a V2Policy mirror to mimic the cloned scheme's bank magnitude,
encodes the trained weights to a normalized [0, 1] PSO chromosome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import NetworkConfig, TrainingConfig
from aerocapture.training.encoding import encode_to_normalized, nn_param_specs_from_v2
from aerocapture.training.evaluate import WARM_START_SEED_OFFSET, make_reserved_seeds
from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

if TYPE_CHECKING:
    from aerocapture.training.rl.policy import V2Policy

try:
    import aerocapture_rs as _aero_rs
except ImportError as e:
    raise ImportError("warm_start requires aerocapture_rs PyO3 module") from e


def _cache_key(cfg: TrainingConfig, source_path: Path, n_warm_seeds: int, n_epochs: int) -> dict:
    # `optimize_scaffolding` and `toml_config` MUST be in the key:
    # - `optimize_scaffolding` flips the cached chromosome width (NN weights
    #   alone vs NN weights + 17 scaffolding slots). Caching across the flip
    #   silently corrupts the initial population in train.py.
    # - `toml_config` drives the supervised dataset (mission, dispersions,
    #   constraint limits). Different TOMLs with the same architecture would
    #   otherwise collide on the cache.
    return {
        "architecture": cfg.network.architecture,
        "input_mask": cfg.network.input_mask,
        "output_parameterization": cfg.network.output_parameterization or "atan2_signed",
        "optimize_scaffolding": bool(cfg.network.optimize_scaffolding),
        "toml_config": str(cfg.sim.toml_config) if cfg.sim.toml_config else None,
        "source_path": str(source_path),
        "source_mtime": source_path.stat().st_mtime,
        "n_warm_seeds": n_warm_seeds,
        "n_epochs": n_epochs,
    }


def _cache_hit(save_dir: Path, expected_key: dict) -> npt.NDArray[np.float64] | None:
    chromo_path = save_dir / "warm_start_chromosome.npy"
    key_path = save_dir / "warm_start_cache_key.json"
    if not (chromo_path.exists() and key_path.exists()):
        return None
    saved_key = json.loads(key_path.read_text())
    if saved_key != expected_key:
        return None
    return np.asarray(np.load(chromo_path), dtype=np.float64)


_INTEGER_PARAM_NAMES: frozenset[str] = frozenset(s.name for s in _NN_SCAFFOLDING_PARAMS if s.is_integer)


def _build_overrides_for_source(source_params: dict[str, float]) -> dict[str, object]:
    """Mirror problem.py::_build_overrides routing for the supervised data source."""
    overrides: dict[str, object] = {}
    for key, value in source_params.items():
        # Round integer-typed params so the Rust TOML parser accepts them (same as problem.py)
        coerced: object = int(round(value)) if key in _INTEGER_PARAM_NAMES else value
        if key.startswith("lateral."):
            overrides[f"guidance.lateral.{key.removeprefix('lateral.')}"] = coerced
        elif key.startswith("exit."):
            overrides[f"guidance.ftc.{key.removeprefix('exit.')}"] = coerced
        elif key.startswith("nav."):
            overrides[f"navigation.{key.removeprefix('nav.')}"] = coerced
        elif key.startswith("thermal."):
            overrides[f"guidance.thermal_limiter.{key.removeprefix('thermal.')}"] = coerced
        elif key.startswith("shaping."):
            overrides[f"guidance.command_shaping.{key.removeprefix('shaping.')}"] = coerced
            overrides["guidance.command_shaping.enabled"] = True
        else:
            overrides[f"guidance.ftc.{key}"] = coerced
    return overrides


def _supervised_pretrain(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    network: NetworkConfig,
    n_epochs: int,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> V2Policy:
    import torch
    from pydantic import TypeAdapter
    from torch import nn

    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import LayerSpec

    validated = TypeAdapter(list[LayerSpec]).validate_python(network.architecture)
    policy = V2Policy(architecture=validated, input_mask=network.input_mask).double()

    output_param = network.output_parameterization or "atan2_signed"
    if output_param == "acos_tanh":
        target = np.cos(y).reshape(-1, 1)
    elif output_param == "atan2_signed":
        target = np.stack([np.sin(y), np.cos(y)], axis=1)
    else:
        raise ValueError(f"unknown output_parameterization {output_param!r}")

    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(target, dtype=torch.float64)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = X_t.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_batch = X_t[idx]
            # V2Policy needs a per-batch zero state (dense layers use None state,
            # so this is free — no allocations beyond the list itself).
            state = policy.new_state(batch_size=x_batch.shape[0], device=x_batch.device)
            optimizer.zero_grad()
            pred, _ = policy(x_batch, state)
            loss = nn.functional.mse_loss(pred, y_t[idx])
            loss.backward()
            optimizer.step()
    return policy


def _policy_to_flat_weights_v2(policy: V2Policy, architecture: list[dict]) -> npt.NDArray[np.float64]:
    """Extract physical weights from a V2Policy in canonical flat order.

    Dispatches per-layer via each layer module's `to_flat()` method (which
    mirrors Rust `LayerWeights::to_flat` for that variant). Concatenates the
    per-layer flat slabs in architecture order.

    Window contributes an empty slab (zero trainable params); the v2 chromosome
    width is the sum across non-empty layers.
    """
    parts: list[npt.NDArray[np.float64]] = []
    for i, (entry, layer_module) in enumerate(zip(architecture, policy.layers, strict=True)):
        if not hasattr(layer_module, "to_flat"):
            raise RuntimeError(
                f"layer {i} ({entry.get('type', '?')}) has no to_flat() method; "
                "ensure the layer module mirrors Rust LayerWeights::to_flat"
            )
        parts.append(np.asarray(layer_module.to_flat(), dtype=np.float64))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


def _select_best_teacher_per_seed(
    results_by_scheme: dict[str, list[dict]],
) -> list[dict]:
    """Across schemes, pick the captured trajectory with the lowest DV per seed.

    Returns a list of dicts with the original (seed, X, y_signed, dv, captured)
    fields plus a "scheme" field naming the winner. Seeds where no scheme
    captures are dropped (warm-start should teach winning behavior).
    """
    all_seeds: set[int] = set()
    for results in results_by_scheme.values():
        for r in results:
            all_seeds.add(r["seed"])

    selected: list[dict] = []
    for seed in sorted(all_seeds):
        candidates: list[tuple[str, dict]] = []
        for scheme, results in results_by_scheme.items():
            for r in results:
                if r["seed"] == seed and r["captured"]:
                    candidates.append((scheme, r))
        if not candidates:
            continue
        scheme, r = min(candidates, key=lambda sr: float(sr[1]["dv"]))
        selected.append({"scheme": scheme, **r})
    return selected


def build_warm_start_chromosome(
    cfg: TrainingConfig,
    base_mc_seed: int,
    n_warm_seeds: int = 200,
    n_epochs: int = 10,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Run cfg's source scheme on n_warm_seeds, supervised-pretrain V2Policy, return chromosome.

    `base_mc_seed` MUST be the resolved value train.py uses for the
    validation/final-eval pools (i.e. `monte_carlo.seed` or 42 if absent).
    Drawing warm-start seeds from a different base would break the
    disjointness contract with those reserved pools.
    """
    raise NotImplementedError(
        "warm_start.build_warm_start_chromosome is in transit: collect_supervised "
        "now returns list[dict] (post Task 1 of the warm-start-all-archs plan); "
        "the multi-supervisor + BPTT rewrite lands in Task 11. See "
        "docs/superpowers/plans/2026-05-22-warm-start-all-archs-plan.md."
    )
    if rng is None:
        rng = np.random.default_rng(0)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(getattr(cfg.network, "warm_start_from", None) or "training_output/ftc/best_params.json")
    if not source_path.exists():
        raise FileNotFoundError(f"warm-start source params not found at '{source_path}'. Run FTC training first or set warm_start_from.")

    cache_key = _cache_key(cfg, source_path, n_warm_seeds, n_epochs)
    cached = _cache_hit(save_dir, cache_key)
    if cached is not None:
        return cached

    with open(source_path) as f:
        source_params = json.load(f)
    overrides = _build_overrides_for_source(source_params)

    seeds = make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, n_warm_seeds)

    X_full, y = _aero_rs.collect_supervised(
        toml_path=cfg.sim.toml_config,
        seeds=seeds,
        overrides=overrides,
        scheme="ftc",
    )
    X_full = np.asarray(X_full)
    y = np.asarray(y)
    finite_mask = np.isfinite(X_full).all(axis=1) & np.isfinite(y)
    X_full = X_full[finite_mask]
    y = y[finite_mask]

    mask = cfg.network.input_mask if cfg.network.input_mask is not None else list(range(16))
    X = X_full[:, mask]

    if cfg.network.architecture is None:
        raise ValueError("warm-start requires cfg.network.architecture; got None")
    architecture = cfg.network.architecture

    policy = _supervised_pretrain(X, y, cfg.network, n_epochs)
    flat_weights = _policy_to_flat_weights_v2(policy, architecture)

    from pydantic import TypeAdapter

    from aerocapture.training.rl.schemas import LayerSpec

    validated = TypeAdapter(list[LayerSpec]).validate_python(architecture)
    weight_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)
    weight_chromo = np.empty(len(weight_specs), dtype=np.float64)
    n_clipped = 0
    for i, s in enumerate(weight_specs):
        v = float(flat_weights[i])
        normalized = (v - s.p_min) / (s.p_max - s.p_min)
        if normalized < 0.0 or normalized > 1.0:
            n_clipped += 1
        weight_chromo[i] = np.clip(normalized, 0.0, 1.0)

    # The PSO chromosome bounds are 2× Xavier; Adam-trained weights routinely drift
    # past that, especially on the last layer. Heavy clipping means the warm-started
    # population starts piled at chromosome boundaries — defeating the warm-start.
    # Log so the user can react (widen bound_multiplier, fewer epochs, lower LR).
    clip_rate = n_clipped / max(len(weight_specs), 1)
    if clip_rate > 0.05:
        print(
            f"  [warm_start] WARNING: {n_clipped}/{len(weight_specs)} weights "
            f"({100 * clip_rate:.1f}%) clipped to chromosome bounds. "
            f"Consider widening bound_multiplier or reducing n_epochs/lr."
        )
    elif n_clipped > 0:
        print(f"  [warm_start] {n_clipped}/{len(weight_specs)} weights clipped ({100 * clip_rate:.2f}%).")

    chromo = weight_chromo
    if cfg.network.optimize_scaffolding:
        scaff_chromo = encode_to_normalized(source_params, list(_NN_SCAFFOLDING_PARAMS))
        chromo = np.concatenate([weight_chromo, scaff_chromo])

    np.save(save_dir / "warm_start_chromosome.npy", chromo)
    (save_dir / "warm_start_cache_key.json").write_text(json.dumps(cache_key, indent=2))

    return chromo
