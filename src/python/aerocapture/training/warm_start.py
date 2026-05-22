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


def _cache_key(
    cfg: TrainingConfig,
    resolved_paths: dict[str, Path],
    scaffolding_source_path: Path,
    mode: str,
) -> dict:
    # `optimize_scaffolding` and `toml_config` MUST be in the key:
    # - `optimize_scaffolding` flips the cached chromosome width (NN weights
    #   alone vs NN weights + 17 scaffolding slots). Caching across the flip
    #   silently corrupts the initial population in train.py.
    # - `toml_config` drives the supervised dataset (mission, dispersions,
    #   constraint limits). Different TOMLs with the same architecture would
    #   otherwise collide on the cache.
    #
    # Note: scaffolding_source_{path,mtime} are tracked even though train.py
    # overwrites the cached scaffolding tail with build_scaffolding_initial_slab.
    # Intentional: conservative cache invalidation; FTC retraining is rare.
    return {
        "architecture": cfg.network.architecture,
        "input_mask": cfg.network.input_mask,
        "output_parameterization": cfg.network.output_parameterization or "atan2_signed",
        "optimize_scaffolding": bool(cfg.network.optimize_scaffolding),
        "toml_config": str(cfg.sim.toml_config) if cfg.sim.toml_config else None,
        "supervisor_schemes": sorted(cfg.warm_start.supervisor_schemes),
        "supervisor_params": {
            scheme: {"path": str(p), "mtime": p.stat().st_mtime}
            for scheme, p in sorted(resolved_paths.items())
        },
        "scaffolding_source_path": str(scaffolding_source_path),
        "scaffolding_source_mtime": scaffolding_source_path.stat().st_mtime,
        "n_warm_seeds": cfg.warm_start.n_warm_seeds,
        "n_epochs": cfg.warm_start.n_epochs,
        "bptt_length": cfg.warm_start.bptt_length,
        "bound_multiplier": cfg.warm_start.bound_multiplier,
        "mode": mode,
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


def _chunked_bptt_train(
    trajectories: list[dict],
    network: NetworkConfig,
    bptt_length: int,
    n_epochs: int,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[V2Policy, list[float], int]:
    """Chunked truncated-BPTT supervised pretraining (windowed variant).

    Each trajectory is split into `bptt_length`-sized chunks; per-chunk forward
    is via `V2Policy.forward_seq_means`. Hidden state is zero-initialized at the
    start of every chunk (no cross-chunk state carry) - chunks are shuffled and
    batched across trajectories, so per-trajectory state continuity would not
    align row-wise inside a minibatch. The cold-start bias this introduces in
    recurrent layers is absorbed by downstream GA/PSO fine-tuning, which runs
    against the actual Rust runtime that carries state correctly across the
    full trajectory.

    Loss is MSE between the predicted output parameterization (cos(y) for
    acos_tanh, (sin,cos) for atan2_signed) and the target. For magnitude_only
    mode, callers pre-process `y_signed -> abs(y_signed)`.
    """
    import torch
    from pydantic import TypeAdapter
    from torch import nn

    from aerocapture.training.rl.layers.transformer import TransformerLayer
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import LayerSpec

    if network.architecture is None:
        raise ValueError("_chunked_bptt_train requires a v2 architecture (network.architecture is None)")

    validated_arch = TypeAdapter(list[LayerSpec]).validate_python(network.architecture)
    policy = V2Policy(architecture=validated_arch, input_mask=network.input_mask).double()

    # Validate bptt_length <= n_seq for any Transformer layer
    for i, layer in enumerate(policy.layers):
        if isinstance(layer, TransformerLayer) and bptt_length > layer.n_seq:
            raise ValueError(f"bptt_length={bptt_length} > layer {i} Transformer n_seq={layer.n_seq}; reduce bptt_length or increase n_seq")

    output_param = network.output_parameterization or "atan2_signed"
    input_mask = network.input_mask if network.input_mask is not None else list(range(21))

    # Build chunks: list of (X_chunk[T_c, input_dim], y_chunk[T_c]) per trajectory.
    chunks: list[tuple[np.ndarray, np.ndarray, int]] = []  # (X, y, traj_id)
    for tid, traj in enumerate(trajectories):
        X = np.asarray(traj["X"])[:, input_mask]
        y = np.asarray(traj["y_signed"])
        # Drop non-finite rows
        finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[finite]
        y = y[finite]
        T = X.shape[0]
        # Slice into bptt_length chunks; trailing partial chunk dropped (clean BPTT)
        n_chunks = T // bptt_length
        for c in range(n_chunks):
            s = c * bptt_length
            e = s + bptt_length
            chunks.append((X[s:e], y[s:e], tid))

    if not chunks:
        raise RuntimeError("no usable BPTT chunks; check bptt_length vs trajectory lengths")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses: list[float] = []

    for _epoch in range(n_epochs):
        # Shuffle chunks; minibatch as the chunk-batch dim
        order = rng.permutation(len(chunks))
        # Group into minibatches of up to 32 chunks; each minibatch is forwarded together.
        # Different trajectories' chunks can be batched freely because we re-init state per chunk.
        chunk_batch_size = min(32, len(chunks))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), chunk_batch_size):
            batch_idx = order[start : start + chunk_batch_size]
            X_batch = np.stack([chunks[i][0] for i in batch_idx], axis=0)  # (B, T, in)
            y_batch = np.stack([chunks[i][1] for i in batch_idx], axis=0)  # (B, T)
            # Time-major
            obs_seq = torch.tensor(X_batch.transpose(1, 0, 2), dtype=torch.float64)  # (T, B, in)
            y_t = torch.tensor(y_batch.transpose(1, 0), dtype=torch.float64)  # (T, B)

            B = obs_seq.shape[1]
            state_0 = policy.new_state(batch_size=B, device=None)
            dones = torch.zeros(obs_seq.shape[0], B, dtype=torch.bool)  # no dones within a chunk

            optimizer.zero_grad()
            means = policy.forward_seq_means(obs_seq, state_0, dones)  # (T, B, out_dim)

            if output_param == "acos_tanh":
                pred = torch.tanh(means[..., 0])  # (T, B)
                target = torch.cos(y_t)  # (T, B)
                loss = nn.functional.mse_loss(pred, target)
            elif output_param == "atan2_signed":
                # means: (T, B, 2). Target = (sin(y), cos(y)).
                target = torch.stack([torch.sin(y_t), torch.cos(y_t)], dim=-1)
                loss = nn.functional.mse_loss(means, target)
            else:
                raise ValueError(f"unknown output_parameterization {output_param!r}")

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return policy, losses, len(chunks)


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
            raise RuntimeError(f"layer {i} ({entry.get('type', '?')}) has no to_flat() method; ensure the layer module mirrors Rust LayerWeights::to_flat")
        parts.append(np.asarray(layer_module.to_flat(), dtype=np.float64))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


def _select_best_teacher_per_seed(
    results_by_scheme: dict[str, list[dict]],
) -> list[dict]:
    """Across schemes, pick the captured trajectory with the lowest DV per seed.

    Returns a list of dicts with the original (seed, X, y_signed, dv, captured)
    fields plus a "scheme" field naming the winner. Seeds where no scheme
    captures are dropped (warm-start should teach winning behavior). Ties on
    DV broken by scheme iteration order in `results_by_scheme`.
    """
    best: dict[int, tuple[str, dict]] = {}
    for scheme, results in results_by_scheme.items():
        for r in results:
            if not r["captured"]:
                continue
            seed = int(r["seed"])
            if seed not in best or float(r["dv"]) < float(best[seed][1]["dv"]):
                best[seed] = (scheme, r)

    return [{"scheme": scheme, **r} for seed, (scheme, r) in sorted(best.items())]


def build_warm_start_chromosome(
    cfg: TrainingConfig,
    base_mc_seed: int,
) -> npt.NDArray[np.float64]:
    """Multi-supervisor warm-start: collect per-seed best teacher, chunked-BPTT, encode.

    Configuration is fully read from cfg.warm_start (supervisor_schemes,
    bptt_length, n_warm_seeds, n_epochs, bound_multiplier, params_paths)
    and cfg.network (architecture, input_mask, output_parameterization,
    optimize_scaffolding, warm_start_from for the scaffolding source).

    `base_mc_seed` MUST be the resolved value train.py uses for
    validation/final-eval pools so warm-start seeds are disjoint
    (`WARM_START_SEED_OFFSET = 4M`).
    """
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ws = cfg.warm_start
    network = cfg.network
    mode = _resolve_nn_mode(cfg)

    # 1. Resolve supervisor paths
    resolved_paths: dict[str, Path] = {}
    for scheme in ws.supervisor_schemes:
        override = ws.params_paths.get(scheme)
        path = Path(override) if override else Path(f"training_output/{scheme}/best_params.json")
        if not path.exists():
            raise FileNotFoundError(
                f"warm-start supervisor '{scheme}' params not found at '{path}'. "
                f"Train {scheme} first or set [warm_start.params_paths].{scheme}."
            )
        resolved_paths[scheme] = path

    # Scaffolding source (for the 17-slot tail when optimize_scaffolding)
    scaffolding_source_path = Path(network.warm_start_from) if network.warm_start_from else resolved_paths[ws.supervisor_schemes[0]]
    if not scaffolding_source_path.exists():
        raise FileNotFoundError(f"scaffolding source params not found at '{scaffolding_source_path}'")

    # 2. Cache check
    cache_key = _cache_key(cfg, resolved_paths, scaffolding_source_path, mode)
    cached = _cache_hit(save_dir, cache_key)
    if cached is not None:
        return cached

    # 3. Collect per scheme
    seeds = make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, ws.n_warm_seeds)
    results_by_scheme: dict[str, list[dict]] = {}
    for scheme, path in resolved_paths.items():
        with open(path) as f:
            source_params = json.load(f)
        overrides = _build_overrides_for_source(source_params)
        results_by_scheme[scheme] = _aero_rs.collect_supervised(
            toml_path=cfg.sim.toml_config,
            seeds=seeds,
            overrides=overrides,
            scheme=scheme,
        )

    # 4. Pick best per seed
    selected = _select_best_teacher_per_seed(results_by_scheme)
    min_corpus = max(20, ws.n_warm_seeds // 4)
    if len(selected) < min_corpus:
        raise RuntimeError(
            f"warm-start corpus too small: {len(selected)} captures across {ws.n_warm_seeds} seeds "
            f"(threshold {min_corpus}). Widen MC dispersions, check the TOML, or revise supervisor_schemes."
        )

    # 5. Magnitude_only mode: derive |y| Python-side
    for traj in selected:
        if mode == "magnitude_only":
            traj["y_signed"] = np.abs(traj["y_signed"])

    # 6. Chunked-BPTT supervised pretraining
    policy, losses, n_chunks = _chunked_bptt_train(
        trajectories=selected,
        network=network,
        bptt_length=ws.bptt_length,
        n_epochs=ws.n_epochs,
    )
    (save_dir / "warm_start_loss.json").write_text(
        json.dumps(
            [{"epoch": i, "mean_mse": float(loss), "n_chunks": n_chunks} for i, loss in enumerate(losses)],
            indent=2,
        )
    )
    print(f"  [warm_start] supervised MSE: {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} epochs")

    # 7. Extract flat weights and encode to normalized chromosome at warm-start bound_multiplier
    assert network.architecture is not None  # validated by _chunked_bptt_train
    flat_weights = _policy_to_flat_weights_v2(policy, network.architecture)
    from pydantic import TypeAdapter

    from aerocapture.training.rl.schemas import LayerSpec

    validated_arch = TypeAdapter(list[LayerSpec]).validate_python(network.architecture)
    weight_specs = nn_param_specs_from_v2(validated_arch, bound_multiplier=ws.bound_multiplier)

    # Safety guard (per Task 7 code-quality review): zero-param layers must be
    # skipped consistently by nn_param_specs_from_v2 and _policy_to_flat_weights_v2.
    assert len(flat_weights) == len(weight_specs), (
        f"flat_weights length ({len(flat_weights)}) != weight_specs length ({len(weight_specs)}); "
        "zero-param layers (Window) must be skipped consistently in both encoders"
    )

    weight_chromo = np.empty(len(weight_specs), dtype=np.float64)
    n_clipped = 0
    for i, s in enumerate(weight_specs):
        v = float(flat_weights[i])
        normalized = (v - s.p_min) / (s.p_max - s.p_min)
        if normalized < 0.0 or normalized > 1.0:
            n_clipped += 1
        weight_chromo[i] = np.clip(normalized, 0.0, 1.0)

    clip_rate = n_clipped / max(len(weight_specs), 1)
    if clip_rate > 0.05:
        raise RuntimeError(
            f"warm-start clip rate {100 * clip_rate:.1f}% ({n_clipped}/{len(weight_specs)}) exceeds 5% threshold. "
            "Widen [warm_start] bound_multiplier, reduce n_epochs, or lower lr."
        )
    elif n_clipped > 0:
        print(f"  [warm_start] {n_clipped}/{len(weight_specs)} weights clipped ({100 * clip_rate:.2f}%).")

    chromo = weight_chromo
    if network.optimize_scaffolding:
        with open(scaffolding_source_path) as f:
            scaff_params = json.load(f)
        scaff_chromo = encode_to_normalized(scaff_params, list(_NN_SCAFFOLDING_PARAMS))
        chromo = np.concatenate([weight_chromo, scaff_chromo])

    np.save(save_dir / "warm_start_chromosome.npy", chromo)
    (save_dir / "warm_start_cache_key.json").write_text(json.dumps(cache_key, indent=2))
    return chromo


def _resolve_nn_mode(cfg: TrainingConfig) -> str:
    """Read [guidance.neural_network] mode from the resolved TOML; default 'full_neural'.

    Uses load_toml_with_bases so the key is honored when set in a parent base TOML.
    """
    if cfg.sim.toml_config is None:
        return "full_neural"
    from aerocapture.training.toml_utils import load_toml_with_bases

    doc = load_toml_with_bases(Path(cfg.sim.toml_config))
    return str(doc.get("guidance", {}).get("neural_network", {}).get("mode", "full_neural"))
