"""Behavioural-cloning warm-start for NN guidance training.

Runs a non-NN scheme (default FTC) over a reserved seed pool, collects
(state, |bank|) pairs via aerocapture_rs.collect_supervised, supervised
pre-trains a V2Policy mirror to mimic the cloned scheme's bank magnitude,
encodes the trained weights to a normalized [0, 1] PSO chromosome.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import AdamConfig, NetworkConfig, TrainingConfig
from aerocapture.training.encoding import encode_to_normalized, nn_param_specs_from_v2
from aerocapture.training.evaluate import WARM_START_SEED_OFFSET, make_reserved_seeds
from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS, route_param_path

if TYPE_CHECKING:
    import torch

    from aerocapture.training.rl.policy import V2Policy

try:
    import aerocapture_rs as _aero_rs
except ImportError as e:
    raise ImportError("warm_start requires aerocapture_rs PyO3 module") from e


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def encode_supervised_target(
    output_param: str,
    y: torch.Tensor,
    prev_realized: torch.Tensor | None,
    scaled_pi_n: float,
    delta_max: float,
) -> torch.Tensor:
    """Per-decoder supervised target read directly from the tanh head (means[...,0])."""
    import torch

    if output_param == "scaled_pi":
        return torch.clamp(y / (scaled_pi_n * math.pi), -1.0, 1.0)
    if output_param == "delta":
        assert prev_realized is not None, "delta decoder requires prev_realized"
        diff = _wrap_to_pi(y - prev_realized)
        return torch.clamp(diff / delta_max, -1.0, 1.0)
    raise ValueError(f"encode_supervised_target: {output_param!r} is not delta/scaled_pi")


def _cache_key(
    cfg: TrainingConfig,
    resolved_paths: dict[str, Path],
    scaffolding_source_path: Path,
    mode: str,
    base_mc_seed: int,
) -> dict:
    # `scaffolding` and `toml_config` MUST be in the key:
    # - `scaffolding` flips the cached chromosome width (off→NN weights only,
    #   live→NN weights + 3 nav/shaping slots, full→NN weights + 17 slots).
    #   Caching across a mode change silently corrupts the initial population in train.py.
    # - `toml_config` drives the supervised dataset (mission, dispersions,
    #   constraint limits). Different TOMLs with the same architecture would
    #   otherwise collide on the cache.
    #
    # Note: scaffolding_source_{path,mtime} are tracked even though train.py
    # rebuilds the scaffolding tail from the source ("full":
    # build_scaffolding_initial_slab; "live": build_default_scaffolding_slab) --
    # the source mtime is still tracked for conservative cache invalidation.
    # Track mtime of the leaf TOML so in-place edits invalidate the cache.
    # In-place edits change `[monte_carlo]`, `[flight.constraints]`,
    # `[onboard_atmosphere]`, `[navigation]`, etc., all of which affect the
    # supervised dataset. Base TOMLs are not separately tracked -- the leaf's
    # mtime is the only one Python sees from `cfg.sim.toml_config`. Users who
    # edit a base TOML without touching the leaf can `touch` the leaf to force
    # a cache miss.
    toml_path_str = str(cfg.sim.toml_config) if cfg.sim.toml_config else None
    toml_mtime: float | None = None
    if toml_path_str:
        toml_path_p = Path(toml_path_str)
        if toml_path_p.exists():
            toml_mtime = toml_path_p.stat().st_mtime

    # Use list ordering (not sorted) so reordering supervisor_schemes invalidates
    # the cache. _select_best_teacher_per_seed's tie-breaking depends on
    # list order, and so does the scaffolding_source_path fallback.
    return {
        "architecture": cfg.network.architecture,
        "input_mask": cfg.network.input_mask,
        "output_parameterization": cfg.network.output_parameterization or "atan2_signed",
        "scaffolding": cfg.network.scaffolding,
        "toml_config": toml_path_str,
        "toml_config_mtime": toml_mtime,
        "supervisor_schemes": list(cfg.warm_start.supervisor_schemes),
        "supervisor_params": {scheme: {"path": str(p), "mtime": p.stat().st_mtime} for scheme, p in sorted(resolved_paths.items())},
        "scaffolding_source_path": str(scaffolding_source_path),
        "scaffolding_source_mtime": scaffolding_source_path.stat().st_mtime,
        "n_warm_seeds": cfg.warm_start.n_warm_seeds,
        "n_epochs": cfg.warm_start.n_epochs,
        "bptt_length": cfg.warm_start.bptt_length,
        "minibatch_size": cfg.warm_start.minibatch_size,
        "bound_multiplier": cfg.warm_start.bound_multiplier,
        "adaptive_bounds": bool(cfg.warm_start.adaptive_bounds),
        "adam": {
            "lr": cfg.warm_start.adam.lr,
            "beta1": cfg.warm_start.adam.beta1,
            "beta2": cfg.warm_start.adam.beta2,
            "eps": cfg.warm_start.adam.eps,
            "weight_decay": cfg.warm_start.adam.weight_decay,
            "amsgrad": bool(cfg.warm_start.adam.amsgrad),
        },
        "mode": mode,
        # The supervised dataset is `make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, n)`.
        # Different base_mc_seed produces a different seed pool, so cache hits
        # must be gated on it -- otherwise rerunning with a different
        # monte_carlo.seed silently reuses the previous chromosome.
        "base_mc_seed": int(base_mc_seed),
    }


def _cache_hit(save_dir: Path, expected_key: dict) -> tuple[npt.NDArray[np.float64], list | None] | None:
    """Return (chromosome, weight_specs | None) on cache hit, else None.

    `weight_specs` is None when the cache predates adaptive-bounds persistence;
    callers should rebuild the specs via `nn_param_specs_from_v2` in that case.
    """
    chromo_path = save_dir / "warm_start_chromosome.npy"
    key_path = save_dir / "warm_start_cache_key.json"
    if not (chromo_path.exists() and key_path.exists()):
        return None
    saved_key = json.loads(key_path.read_text())
    if saved_key != expected_key:
        return None
    chromo = np.asarray(np.load(chromo_path), dtype=np.float64)
    weight_specs: list | None = None
    bounds_path = save_dir / "warm_start_bounds.json"
    if bounds_path.exists():
        from aerocapture.training.param_spaces import ParamSpec

        raw = json.loads(bounds_path.read_text())
        weight_specs = [
            ParamSpec(
                name=str(e["name"]),
                p_min=float(e["p_min"]),
                p_max=float(e["p_max"]),
                default=float(e.get("default", 0.0)),
                log_scale=bool(e.get("log_scale", False)),
                is_integer=bool(e.get("is_integer", False)),
            )
            for e in raw
        ]
    return chromo, weight_specs


_INTEGER_PARAM_NAMES: frozenset[str] = frozenset(s.name for s in _NN_SCAFFOLDING_PARAMS if s.is_integer)


def _build_overrides_for_source(source_params: dict[str, float], scheme: str) -> dict[str, object]:
    """Mirror problem.py::_build_overrides routing for the supervised data source.

    `scheme` is the supervisor scheme name (e.g. "ftc", "equilibrium_glide").
    Unprefixed keys -- the scheme's primary parameters -- route to
    `guidance.{scheme}.*` so each supervisor sees its OWN best_params.json values
    rather than having them silently dropped under `guidance.ftc.*` (which would
    leave non-FTC supervisors running with TOML-default parameters).
    """
    overrides: dict[str, object] = {}
    for key, value in source_params.items():
        # Round integer-typed params so the Rust TOML parser accepts them (same as problem.py)
        coerced: object = int(round(value)) if key in _INTEGER_PARAM_NAMES else value
        # exit.* routes to [guidance.ftc.*] for all schemes: the shared exit-phase
        # controller (gnc/guidance/exit.rs) reads those keys regardless of supervisor.
        overrides[route_param_path(key, scheme)] = coerced
        if key.startswith("shaping."):
            overrides["guidance.command_shaping.enabled"] = True
    return overrides


def _seed_policy_init(policy: object, architecture: list, bound_multiplier: float, rng: np.random.Generator) -> None:
    """Overwrite V2Policy parameter init with `init_v2_population` centers
    for ALL layer types (Dense, GRU, LSTM, Window, Transformer, Mamba).

    Without this step:
      - Mamba's a_log/dt_proj_b/d_skip start at zero (degenerate fixed point:
        A=-exp(0)=-1 uniform, no input projection, no skip).
      - LSTM forget-bias slot misses the Jozefowicz +1.0 lift.
      - Dense/GRU/Transformer keep torch's narrow uniform(-1/sqrt(H), +1/sqrt(H))
        defaults instead of the Xavier × bound_multiplier (typically × 4) range
        that the from-scratch PSO path uses -- so warm-start and PSO occupy
        different init basins.

    This helper draws one population row from init_v2_population (which holds
    the canonical activation-aware init for every layer type) and writes each
    layer's slab into the corresponding torch module via per-type from_flat.
    The slab layout must match each layer's `to_flat()` order (which mirrors
    Rust LayerWeights::to_flat).

    Init RNG is sourced from a fresh sub-rng so that warm-start architecture
    width does NOT couple to the per-epoch chunk-shuffle order downstream.
    """
    import torch

    from aerocapture.training.config import _layer_n_params
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.layers import DenseLayer, GruLayer, LstmLayer, MambaLayer, TransformerLayer, WindowLayer

    # Sub-rng decouples init draws from outer rng's downstream uses (chunk
    # shuffle); architecture-width changes don't affect shuffle reproducibility.
    init_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    flat_pop = init_v2_population(architecture, n_pop=1, bound_multiplier=bound_multiplier, rng=init_rng)
    flat = flat_pop[0]

    def _copy(param: torch.Tensor, src: np.ndarray) -> None:
        # Accepts any Tensor with .copy_ (nn.Parameter and nn.Linear.weight/bias both qualify).
        param.copy_(torch.from_numpy(np.ascontiguousarray(src)).to(param.dtype))

    cursor = 0
    for module, entry in zip(policy.layers, architecture, strict=True):  # type: ignore[attr-defined]
        n = _layer_n_params(entry)
        slab = flat[cursor : cursor + n]
        cursor += n
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        ltype = entry["type"]
        with torch.no_grad():
            if ltype == "dense" and isinstance(module, DenseLayer):
                fan_in = int(entry["input_size"])
                fan_out = int(entry["output_size"])
                n_w = fan_out * fan_in
                _copy(module.linear.weight, slab[:n_w].reshape(fan_out, fan_in))
                _copy(module.linear.bias, slab[n_w : n_w + fan_out])
            elif ltype == "gru" and isinstance(module, GruLayer):
                fan_in = int(entry["input_size"])
                hidden = int(entry["hidden_size"])
                three_h = 3 * hidden
                n_w_ih = three_h * fan_in
                n_w_hh = three_h * hidden
                c = 0
                _copy(module.weight_ih, slab[c : c + n_w_ih].reshape(three_h, fan_in))
                c += n_w_ih
                _copy(module.weight_hh, slab[c : c + n_w_hh].reshape(three_h, hidden))
                c += n_w_hh
                _copy(module.bias_ih, slab[c : c + three_h])
                c += three_h
                _copy(module.bias_hh, slab[c : c + three_h])
            elif ltype == "lstm" and isinstance(module, LstmLayer):
                fan_in = int(entry["input_size"])
                hidden = int(entry["hidden_size"])
                four_h = 4 * hidden
                n_w_ih = four_h * fan_in
                n_w_hh = four_h * hidden
                c = 0
                _copy(module.weight_ih, slab[c : c + n_w_ih].reshape(four_h, fan_in))
                c += n_w_ih
                _copy(module.weight_hh, slab[c : c + n_w_hh].reshape(four_h, hidden))
                c += n_w_hh
                # Write the FULL bias_ih AND bias_hh from the slab. init_v2_population
                # already sets bias_ih's forget slot to 1.0 + noise and bias_hh's
                # forget slot to ~0 -- preserving the Jozefowicz signal which would
                # otherwise be diluted by torch's uniform(-1/sqrt(H), +1/sqrt(H))
                # defaults on bias_hh.
                _copy(module.bias_ih, slab[c : c + four_h])
                c += four_h
                _copy(module.bias_hh, slab[c : c + four_h])
            elif ltype == "window" and isinstance(module, WindowLayer):
                # Zero trainable params; slab is empty by construction.
                assert n == 0 and slab.size == 0, f"window slab expected 0-width, got {slab.size}"
            elif ltype == "transformer" and isinstance(module, TransformerLayer):
                # Transformer flat order (matches to_flat in layers/transformer.py:155
                # and Rust LayerWeights<TransformerLayer>::to_flat):
                #   w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
                #   w_ffn1, b_ffn1, w_ffn2, b_ffn2,
                #   ln1_gamma, ln1_beta, ln2_gamma, ln2_beta
                d_model = int(entry["d_model"])
                d_ffn = int(entry["d_ffn"])
                c = 0
                # Q/K/V/O projections: each is (d_model x d_model) + (d_model,) bias.
                for linear in (module.w_q, module.w_k, module.w_v, module.w_o):
                    n_w = d_model * d_model
                    _copy(linear.weight, slab[c : c + n_w].reshape(d_model, d_model))
                    c += n_w
                    _copy(linear.bias, slab[c : c + d_model])
                    c += d_model
                # FFN1: (d_ffn x d_model) + (d_ffn,)
                n_ffn1_w = d_ffn * d_model
                _copy(module.w_ffn1.weight, slab[c : c + n_ffn1_w].reshape(d_ffn, d_model))
                c += n_ffn1_w
                _copy(module.w_ffn1.bias, slab[c : c + d_ffn])
                c += d_ffn
                # FFN2: (d_model x d_ffn) + (d_model,)
                n_ffn2_w = d_model * d_ffn
                _copy(module.w_ffn2.weight, slab[c : c + n_ffn2_w].reshape(d_model, d_ffn))
                c += n_ffn2_w
                _copy(module.w_ffn2.bias, slab[c : c + d_model])
                c += d_model
                # Layer norms: (gamma, beta) x 2, each (d_model,)
                _copy(module.ln1_gamma, slab[c : c + d_model])
                c += d_model
                _copy(module.ln1_beta, slab[c : c + d_model])
                c += d_model
                _copy(module.ln2_gamma, slab[c : c + d_model])
                c += d_model
                _copy(module.ln2_beta, slab[c : c + d_model])
                # PE offsets are recomputed lazily on forward; no post-write step needed.
            elif ltype == "mamba" and isinstance(module, MambaLayer):
                d_inner = int(entry["input_size"])
                d_state = int(entry["d_state"])
                dt_rank = int(entry["dt_rank"])
                c = 0
                n_xp = (dt_rank + 2 * d_state) * d_inner
                _copy(module.x_proj_w, slab[c : c + n_xp].reshape(dt_rank + 2 * d_state, d_inner))
                c += n_xp
                n_dw = d_inner * dt_rank
                _copy(module.dt_proj_w, slab[c : c + n_dw].reshape(d_inner, dt_rank))
                c += n_dw
                _copy(module.dt_proj_b, slab[c : c + d_inner])
                c += d_inner
                n_al = d_inner * d_state
                _copy(module.a_log, slab[c : c + n_al].reshape(d_inner, d_state))
                c += n_al
                _copy(module.d_skip, slab[c : c + d_inner])
            else:
                raise ValueError(f"_seed_policy_init: unknown layer type {ltype!r} or module/spec mismatch")


def _chunked_bptt_train(
    trajectories: list[dict],
    network: NetworkConfig,
    bptt_length: int,
    n_epochs: int,
    bound_multiplier: float = 4.0,
    minibatch_size: int = 128,
    adam: AdamConfig | None = None,
    seed: int = 0,
    eval_callback: Callable[[int, V2Policy], None] | None = None,
    eval_interval: int = 0,
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

    `adam` overrides the Adam hyperparameters; defaults match
    `torch.optim.Adam`'s defaults (lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0, amsgrad=False).
    """
    import torch
    from pydantic import TypeAdapter
    from torch import nn

    from aerocapture.training.rl.layers.transformer import TransformerLayer
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import LayerSpec

    if network.architecture is None:
        raise ValueError("_chunked_bptt_train requires a v2 architecture (network.architecture is None)")

    # Make policy init reproducible across cache-miss reruns. Without this,
    # `nn.Linear.reset_parameters` and GRU/LSTM internal gate inits draw from
    # torch's global RNG, which depends on whatever else has run in the
    # interpreter -- so two cache-miss rebuilds with identical TOML produce
    # different `warm_start_chromosome.npy` outputs.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    validated_arch = TypeAdapter(list[LayerSpec]).validate_python(network.architecture)
    policy = V2Policy(architecture=validated_arch, input_mask=network.input_mask).double()

    # Mirror the activation-aware init that init_v2_population applies to the
    # PSO from-scratch path. Without this, Mamba layers start at all-zero
    # (degenerate fixed point) and the LSTM forget bias never gets the
    # Jozefowicz +1.0 lift.
    _seed_policy_init(policy, list(validated_arch), bound_multiplier, rng)

    # Validate bptt_length <= n_seq for any Transformer layer
    for i, layer in enumerate(policy.layers):
        if isinstance(layer, TransformerLayer) and bptt_length > layer.n_seq:
            raise ValueError(f"bptt_length={bptt_length} > layer {i} Transformer n_seq={layer.n_seq}; reduce bptt_length or increase n_seq")

    output_param = network.output_parameterization or "atan2_signed"
    # Default to architecture[0].input_size when input_mask is absent. Rust
    # `collect_supervised` emits X with shape (T, NN_FULL_INPUT_SIZE) (the
    # FULL_MASK / build_nn_input contract). If the first layer wants more
    # inputs than the candidate vector, the silent first-N slice would
    # IndexError on the column select below, so reject explicitly here.
    from aerocapture.training.config import _RUNTIME_CANDIDATE_WIDTH as _CANDIDATE_INPUT_WIDTH

    arch_first_in = int(network.architecture[0]["input_size"])
    if arch_first_in > _CANDIDATE_INPUT_WIDTH:
        raise ValueError(
            f"architecture[0].input_size={arch_first_in} exceeds the {_CANDIDATE_INPUT_WIDTH}-wide "
            f"supervised candidate vector; either pick a smaller first layer or extend Rust build_nn_input."
        )
    input_mask = network.input_mask if network.input_mask is not None else list(range(arch_first_in))

    # Build chunks. Per-trajectory windows of bptt_length; trailing partial
    # chunk dropped. Collected into two flat lists so we can pre-stack into
    # contiguous tensors once and reuse them across all epochs.
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    pr_list: list[np.ndarray] = []
    n_short = 0
    for traj in trajectories:
        X = np.asarray(traj["X"])[:, input_mask]
        y = np.asarray(traj["y_signed"])
        # `prev_realized` (Task 6) is only consumed by the `delta` decoder; for
        # all other decoders it is unused. Fall back to zeros (same shape as y)
        # so trajectory dicts that omit it (e.g. legacy mocks) still work.
        if "prev_realized" in traj:
            pr = np.asarray(traj["prev_realized"])
        elif output_param == "delta":
            raise ValueError("delta warm-start requires 'prev_realized' in the supervised trajectory; collect_supervised must emit it")
        else:
            pr = np.zeros_like(y)
        # Drop non-finite rows
        finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[finite]
        y = y[finite]
        pr = pr[finite]
        T = X.shape[0]
        n_chunks = T // bptt_length
        if n_chunks == 0:
            n_short += 1
        for c in range(n_chunks):
            s = c * bptt_length
            e = s + bptt_length
            X_list.append(X[s:e])
            y_list.append(y[s:e])
            pr_list.append(pr[s:e])

    if not X_list:
        raise RuntimeError(f"no usable BPTT chunks; bptt_length={bptt_length} but all {len(trajectories)} trajectories have T < bptt_length")

    if n_short > 0:
        print(f"  [warm_start] WARNING: {n_short}/{len(trajectories)} supervisor trajectories shorter than bptt_length={bptt_length}; dropped from corpus")

    n_chunks_total = len(X_list)

    # Pre-stack ALL chunks into time-major tensors once. Cost: O(n_chunks * T * in)
    # memory + a single copy. Pays off vs per-minibatch np.stack + torch.tensor
    # (the previous hot path) by ~1-2 orders of magnitude on long training runs
    # because every epoch then becomes pure indexing + matmul.
    obs_all = torch.from_numpy(np.ascontiguousarray(np.stack(X_list, axis=0).transpose(1, 0, 2))).to(torch.float64)  # (T, N, in)
    y_all = torch.from_numpy(np.ascontiguousarray(np.stack(y_list, axis=0).transpose(1, 0))).to(torch.float64)  # (T, N)
    pr_all = torch.from_numpy(np.ascontiguousarray(np.stack(pr_list, axis=0).transpose(1, 0))).to(torch.float64)  # (T, N)
    # dones is identical across all chunks (no done within a chunk by construction),
    # so we build the (T, max_minibatch) tensor once and slice per minibatch.
    effective_minibatch_size = max(1, min(minibatch_size, n_chunks_total))
    dones_max = torch.zeros(bptt_length, effective_minibatch_size, dtype=torch.bool)

    adam_cfg = adam if adam is not None else AdamConfig()
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=adam_cfg.lr,
        betas=(adam_cfg.beta1, adam_cfg.beta2),
        eps=adam_cfg.eps,
        weight_decay=adam_cfg.weight_decay,
        amsgrad=adam_cfg.amsgrad,
    )
    losses: list[float] = []

    epoch_width = len(str(n_epochs))  # zero-pad index for clean column alignment
    print(
        f"  [warm_start] supervised pretrain: {n_chunks_total} chunks, {n_epochs} epochs, bptt_length={bptt_length}, minibatch_size={effective_minibatch_size}"
    )
    for epoch in range(n_epochs):
        # Shuffle chunks; minibatch as the chunk-batch dim. Different
        # trajectories' chunks can be batched freely because we re-init state
        # per chunk.
        order_np = rng.permutation(n_chunks_total)
        order = torch.from_numpy(order_np.astype(np.int64))
        epoch_loss = 0.0
        n_batches = 0
        epoch_t0 = time.monotonic()
        for start in range(0, n_chunks_total, effective_minibatch_size):
            batch_idx = order[start : start + effective_minibatch_size]
            obs_seq = obs_all.index_select(1, batch_idx)  # (T, B, in) -- no copy when contiguous along axis 0
            y_t = y_all.index_select(1, batch_idx)  # (T, B)
            pr_t = pr_all.index_select(1, batch_idx)  # (T, B)

            B = obs_seq.shape[1]
            state_0 = policy.new_state(batch_size=B, device=None)
            # Trailing minibatch may be smaller than effective_minibatch_size;
            # slice the pre-built dones tensor to match.
            dones = dones_max[:, :B] if effective_minibatch_size != B else dones_max

            optimizer.zero_grad()
            means = policy.forward_seq_means(obs_seq, state_0, dones)  # (T, B, out_dim)

            if output_param == "acos_tanh":
                # V2Policy's last layer is required to have activation="tanh"
                # (validated at config load), so means[..., 0] is already in
                # [-1, 1]. Do NOT apply tanh again here -- the runtime decoder
                # is `bank = acos(tanh(out[0]))` only because the network's
                # *raw* linear-output is wrapped in tanh at the layer boundary.
                pred = means[..., 0]  # (T, B), already tanh-activated
                target = torch.cos(y_t)  # (T, B)
                loss = nn.functional.mse_loss(pred, target)
            elif output_param == "atan2_signed":
                # means: (T, B, 2). Target = (sin(y), cos(y)).
                target = torch.stack([torch.sin(y_t), torch.cos(y_t)], dim=-1)
                loss = nn.functional.mse_loss(means, target)
            elif output_param in ("scaled_pi", "delta"):
                pred = means[..., 0]  # tanh head, already in [-1, 1]
                target = encode_supervised_target(
                    output_param,
                    y_t,
                    prev_realized=(pr_t if output_param == "delta" else None),
                    scaled_pi_n=network.scaled_pi_n,
                    delta_max=network.delta_max,
                )
                loss = nn.functional.mse_loss(pred, target)
            else:
                raise ValueError(f"unknown output_parameterization {output_param!r}")

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        mean_mse = epoch_loss / max(n_batches, 1)
        losses.append(mean_mse)
        epoch_dt = time.monotonic() - epoch_t0
        # Convergence delta vs previous epoch (skipped on epoch 0). Use relative
        # change so the magnitude is interpretable across loss scales. Emoji
        # marks: green check on loss decrease, red cross on increase.
        if epoch == 0:
            trend = "          "
        else:
            prev = losses[epoch - 1]
            if prev > 0.0:
                rel = (mean_mse - prev) / prev * 100.0
                marker = "✅" if rel < 0 else "❌"
                trend = f"  {marker} {abs(rel):5.1f}%"
            else:
                trend = "          "
        print(f"  [warm_start] epoch {epoch + 1:>{epoch_width}}/{n_epochs}: MSE = {mean_mse:.4e}{trend}  ({epoch_dt:5.1f}s)")

        # Periodic in-training evaluation hook. Caller (train.py) injects the
        # MC machinery so this function stays free of `problem` / `val_seeds`
        # plumbing. Fires every `eval_interval` epochs AND on the last epoch
        # so the final state is always reported.
        if eval_callback is not None and eval_interval > 0 and ((epoch + 1) % eval_interval == 0 or (epoch + 1) == n_epochs):
            try:
                eval_callback(epoch + 1, policy)
            except Exception as e:
                # Best-effort: a failed eval must not abort training.
                print(f"  [warm_start] WARNING: in-training eval at epoch {epoch + 1} failed: {type(e).__name__}: {e}")

    return policy, losses, n_chunks_total


def _policy_to_flat_weights_v2(policy: V2Policy, architecture: list[dict]) -> npt.NDArray[np.float64]:
    """Extract physical weights from a V2Policy in canonical flat order.

    Dispatches per-layer via each layer module's `to_flat()` method (which
    mirrors Rust `LayerWeights::to_flat` for that variant). Concatenates the
    per-layer flat slabs in architecture order.

    Window contributes an empty slab (zero trainable params); the v2 chromosome
    width is the sum across non-empty layers.
    """
    parts: list[npt.NDArray[np.float64]] = []
    # nn.ModuleList items are typed as `Module | Tensor`; our layer modules all
    # implement `to_flat()` per the LayerWeights mirror contract, but mypy
    # cannot prove that statically.
    for i, (entry, layer_module) in enumerate(zip(architecture, policy.layers, strict=True)):  # type: ignore[union-attr]
        if not hasattr(layer_module, "to_flat"):
            raise RuntimeError(f"layer {i} ({entry.get('type', '?')}) has no to_flat() method; ensure the layer module mirrors Rust LayerWeights::to_flat")
        parts.append(np.asarray(layer_module.to_flat(), dtype=np.float64))  # type: ignore[operator]
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


def _adaptive_layer_slab_specs(
    weight_specs: list,
    flat_weights: np.ndarray,
    architecture: list,
) -> list:
    """Replace each NN-weight ParamSpec with per-layer-slab data-driven bounds.

    For each layer slab in `architecture`:
      slab_bound = max(2.0 * max(|slab|), original_slab_max_abs_bound)

    All ParamSpecs in that slab get symmetric `(-slab_bound, +slab_bound)`
    around 0 (replacing the Xavier × bound_multiplier centers/asymmetric
    structure). The floor at the original Xavier bound prevents the slab
    from shrinking below the activation-aware scale when Adam barely
    moved the weights.

    Zero-param layers (Window) contribute no entries; the cursor stays in
    sync via `_layer_n_params`.

    Trades the per-parameter Xavier structure (which mattered for LSTM
    forget-bias-2, Mamba dt_proj_b, Transformer LN gamma) for guaranteed
    zero-clipping at chromosome encoding. The chromosome's centers shift
    accordingly; encoded `normalized = (v - p_min) / (p_max - p_min)` is
    always inside [0, 1] by construction.
    """
    from aerocapture.training.config import _layer_n_params
    from aerocapture.training.param_spaces import ParamSpec

    new_specs: list = []
    cursor = 0
    for entry in architecture:
        n = _layer_n_params(entry)
        if n == 0:
            continue
        slab = flat_weights[cursor : cursor + n]
        slab_specs = weight_specs[cursor : cursor + n]
        # Adaptive: 2x max-abs over the slab.
        adaptive_bound = 2.0 * float(np.abs(slab).max()) if slab.size else 0.0
        # Floor at the original max-abs bound across the slab (so trivially
        # all-zero slabs don't collapse to a degenerate zero-width range,
        # and PSO keeps the original Xavier room).
        original_floor = max((max(abs(s.p_min), abs(s.p_max)) for s in slab_specs), default=0.0)
        slab_bound = max(adaptive_bound, original_floor)
        for s in slab_specs:
            new_specs.append(
                ParamSpec(
                    name=s.name,
                    p_min=-slab_bound,
                    p_max=+slab_bound,
                    default=s.default,
                    log_scale=s.log_scale,
                    is_integer=s.is_integer,
                )
            )
        cursor += n
    assert cursor == len(weight_specs), f"adaptive bound walk did not consume all weight_specs ({cursor} vs {len(weight_specs)})"
    return new_specs


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
    eval_callback: Callable[[int, V2Policy], None] | None = None,
) -> tuple[npt.NDArray[np.float64], list]:
    """Multi-supervisor warm-start: collect per-seed best teacher, chunked-BPTT, encode.

    Configuration is fully read from cfg.warm_start (supervisor_schemes,
    bptt_length, n_warm_seeds, n_epochs, bound_multiplier, adaptive_bounds,
    params_paths) and cfg.network (architecture, input_mask,
    output_parameterization, scaffolding, warm_start_from for the
    scaffolding source).

    `base_mc_seed` MUST be the resolved value train.py uses for
    validation/final-eval pools so warm-start seeds are disjoint
    (`WARM_START_SEED_OFFSET = 4M`).

    Returns:
        Tuple of (chromosome, weight_specs) where:
          - chromosome: normalized [0, 1] vector of length n_weights (+ len(pack)
            slots when scaffolding != "off": 3 for "live", 17 for "full").
            Row-0 anchor for the PSO/GA/DE initial population.
          - weight_specs: the ParamSpec list for the NN-WEIGHT slab (no
            scaffolding). When `cfg.warm_start.adaptive_bounds` is True
            (default), these carry per-layer-slab adaptive bounds derived
            from the trained values; otherwise they carry the original
            Xavier × bound_multiplier bounds. Caller must propagate these
            into the outer param_specs so PSO decode matches the encoding.
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
                f"warm-start supervisor '{scheme}' params not found at '{path}'. Train {scheme} first or set [warm_start.params_paths].{scheme}."
            )
        resolved_paths[scheme] = path

    # Scaffolding source (for the tail when scaffolding != "off")
    scaffolding_source_path = Path(network.warm_start_from) if network.warm_start_from else resolved_paths[ws.supervisor_schemes[0]]
    if not scaffolding_source_path.exists():
        raise FileNotFoundError(f"scaffolding source params not found at '{scaffolding_source_path}'")

    # 2. Cache check
    cache_key = _cache_key(cfg, resolved_paths, scaffolding_source_path, mode, base_mc_seed)
    cached = _cache_hit(save_dir, cache_key)
    if cached is not None:
        chromo, cached_specs = cached
        if cached_specs is not None:
            return chromo, cached_specs
        # Cache predates adaptive-bounds persistence: rebuild specs at the
        # configured bound_multiplier so the chromosome decode matches.
        from pydantic import TypeAdapter

        from aerocapture.training.rl.schemas import LayerSpec

        assert network.architecture is not None
        validated_arch_cached = TypeAdapter(list[LayerSpec]).validate_python(network.architecture)
        return chromo, nn_param_specs_from_v2(validated_arch_cached, bound_multiplier=ws.bound_multiplier)

    # 3. Collect per scheme. Thread sim_timeout_secs through so a NaN-state
    # supervisor sim can't hang the warm-start pipeline indefinitely (project
    # convention: every run_mc-equivalent path passes the timeout).
    seeds = make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, ws.n_warm_seeds)
    results_by_scheme: dict[str, list[dict]] = {}
    for scheme, path in resolved_paths.items():
        with open(path) as f:
            source_params = json.load(f)
        overrides = _build_overrides_for_source(source_params, scheme)
        results_by_scheme[scheme] = _aero_rs.collect_supervised(
            toml_path=cfg.sim.toml_config,
            seeds=seeds,
            overrides=overrides,
            scheme=scheme,
            sim_timeout_secs=cfg.sim.sim_timeout_secs,
        )

    # 4. Pick best per seed
    selected = _select_best_teacher_per_seed(results_by_scheme)
    min_corpus = max(1, ws.n_warm_seeds // 4)
    if len(selected) < min_corpus:
        raise RuntimeError(
            f"warm-start corpus too small: {len(selected)} captures across {ws.n_warm_seeds} seeds "
            f"(threshold {min_corpus}). Widen MC dispersions, check the TOML, or revise supervisor_schemes."
        )

    # Persist per-supervisor selection counts (and per-supervisor capture stats)
    # so warm_start_report can show which scheme dominated the teaching corpus.
    # Without this, an under-performing supervisor pool (e.g. all FTC wins
    # because eqglide / fnpag never captured) is invisible.
    selection_counts: dict[str, int] = dict.fromkeys(resolved_paths, 0)
    for traj in selected:
        selection_counts[str(traj["scheme"])] += 1
    per_scheme_stats: dict[str, dict] = {}
    for scheme, results in results_by_scheme.items():
        n_total = len(results)
        n_captured = sum(1 for r in results if r["captured"])
        captured_dvs = [float(r["dv"]) for r in results if r["captured"] and np.isfinite(r["dv"])]
        per_scheme_stats[scheme] = {
            "n_supervised": n_total,
            "n_captured": n_captured,
            "capture_rate": n_captured / max(n_total, 1),
            "n_selected": selection_counts.get(scheme, 0),
            "mean_dv_captured": float(np.mean(captured_dvs)) if captured_dvs else None,
            "median_dv_captured": float(np.median(captured_dvs)) if captured_dvs else None,
        }
    (save_dir / "warm_start_selection.json").write_text(
        json.dumps(
            {
                "n_warm_seeds": int(ws.n_warm_seeds),
                "n_selected_total": len(selected),
                "min_corpus_required": min_corpus,
                "per_scheme": per_scheme_stats,
            },
            indent=2,
        )
    )

    # 5. Magnitude_only mode: collapse sign Python-side so the supervised
    # target matches the runtime decoder. Under magnitude_only deploy, the
    # NN's output is .abs()'d in dispatch.rs and routed through lateral
    # guidance for sign re-selection -- so the warm-start target should be
    # unsigned. Under full_neural deploy, no lateral guidance runs at
    # runtime, so the signed target (the supervisor's lateral-chosen sign)
    # is exactly what the NN must learn to emit.
    for traj in selected:
        if mode == "magnitude_only":
            traj["y_signed"] = np.abs(traj["y_signed"])

    # 6. Chunked-BPTT supervised pretraining. Seed is derived from base_mc_seed
    # xor'd with WARM_START_SEED_OFFSET so different `monte_carlo.seed` values
    # produce both different supervised datasets AND different Adam init/shuffle
    # trajectories (previously hardcoded seed=0 made Adam deterministic).
    bptt_seed = int(base_mc_seed) ^ int(WARM_START_SEED_OFFSET)
    policy, losses, n_chunks = _chunked_bptt_train(
        trajectories=selected,
        network=network,
        bptt_length=ws.bptt_length,
        n_epochs=ws.n_epochs,
        bound_multiplier=ws.bound_multiplier,
        minibatch_size=ws.minibatch_size,
        adam=ws.adam,
        seed=bptt_seed,
        eval_callback=eval_callback,
        eval_interval=ws.eval_interval,
    )
    (save_dir / "warm_start_loss.json").write_text(
        json.dumps(
            [{"epoch": i, "mean_mse": float(loss), "n_chunks": n_chunks} for i, loss in enumerate(losses)],
            indent=2,
        )
    )
    # End-of-pretrain summary line (also captured by the per-epoch log + warm_start_loss.json)
    if not losses:
        # n_epochs == 0: caller wants chromosome encoding without any supervised training.
        print("  [warm_start] supervised MSE: (n_epochs=0, no epochs run)")
    elif len(losses) > 1 and losses[0] > 0.0:
        reduction = (losses[0] - losses[-1]) / losses[0] * 100.0
        print(f"  [warm_start] supervised MSE {losses[0]:.4e} -> {losses[-1]:.4e}  ({reduction:+.1f}%)")
    else:
        print(f"  [warm_start] supervised MSE: {losses[-1]:.4e}")

    # 7. Extract flat weights and encode to normalized chromosome.
    assert network.architecture is not None  # validated by _chunked_bptt_train
    flat_weights = _policy_to_flat_weights_v2(policy, network.architecture)
    from pydantic import TypeAdapter

    from aerocapture.training.rl.schemas import LayerSpec

    validated_arch = TypeAdapter(list[LayerSpec]).validate_python(network.architecture)
    base_specs = nn_param_specs_from_v2(validated_arch, bound_multiplier=ws.bound_multiplier)

    # Safety guard (per Task 7 code-quality review): zero-param layers must be
    # skipped consistently by nn_param_specs_from_v2 and _policy_to_flat_weights_v2.
    assert len(flat_weights) == len(base_specs), (
        f"flat_weights length ({len(flat_weights)}) != weight_specs length ({len(base_specs)}); "
        "zero-param layers (Window) must be skipped consistently in both encoders"
    )

    # adaptive_bounds (default True): per-layer-slab 2x max-abs bounds floored at
    # the Xavier × bound_multiplier half-width. By construction every trained
    # value lies inside its slab's [-bound, +bound] range, so encoding never clips.
    weight_specs = _adaptive_layer_slab_specs(base_specs, flat_weights, list(validated_arch)) if ws.adaptive_bounds else base_specs

    weight_chromo = np.empty(len(weight_specs), dtype=np.float64)
    n_clipped = 0
    for i, s in enumerate(weight_specs):
        v = float(flat_weights[i])
        normalized = (v - s.p_min) / (s.p_max - s.p_min)
        if normalized < 0.0 or normalized > 1.0:
            n_clipped += 1
        weight_chromo[i] = np.clip(normalized, 0.0, 1.0)

    clip_rate = n_clipped / max(len(weight_specs), 1)
    if ws.adaptive_bounds:
        # Adaptive bounds guarantee 0% clipping by construction; anything > 0 is a bug.
        assert clip_rate == 0.0, f"adaptive_bounds produced {n_clipped} clipped weights -- internal bug, bounds should always contain trained values"
    elif clip_rate > 0.05:
        raise RuntimeError(
            f"warm-start clip rate {100 * clip_rate:.1f}% ({n_clipped}/{len(weight_specs)}) exceeds 5% threshold. "
            "Set [warm_start] adaptive_bounds = true (default), or widen bound_multiplier, reduce n_epochs, or lower lr."
        )
    elif n_clipped > 0:
        print(f"  [warm_start] {n_clipped}/{len(weight_specs)} weights clipped ({100 * clip_rate:.2f}%).")

    chromo = weight_chromo
    if network.scaffolding != "off":
        from aerocapture.training.param_spaces import active_scaffolding_specs

        pack = active_scaffolding_specs(network.scaffolding)
        if network.scaffolding == "full":
            with open(scaffolding_source_path) as f:
                scaff_params = json.load(f)
        else:  # live: seed the 3-param tail from defaults, no FTC source needed.
            scaff_params = {s.name: s.default for s in pack}
        scaff_chromo = encode_to_normalized(scaff_params, list(pack))
        chromo = np.concatenate([weight_chromo, scaff_chromo])

    np.save(save_dir / "warm_start_chromosome.npy", chromo)
    (save_dir / "warm_start_cache_key.json").write_text(json.dumps(cache_key, indent=2))
    # Persist the bounds so resume / cache-hit returns the same specs the
    # chromosome was encoded under.
    (save_dir / "warm_start_bounds.json").write_text(
        json.dumps(
            [
                {"name": s.name, "p_min": s.p_min, "p_max": s.p_max, "default": s.default, "log_scale": s.log_scale, "is_integer": s.is_integer}
                for s in weight_specs
            ],
            indent=2,
        )
    )
    return chromo, weight_specs


def _resolve_nn_mode(cfg: TrainingConfig) -> str:
    """Read [guidance.neural_network] mode from the resolved TOML; default 'full_neural'.

    Uses load_toml_with_bases so the key is honored when set in a parent base TOML.
    """
    if cfg.sim.toml_config is None:
        return "full_neural"
    from aerocapture.training.toml_utils import load_toml_with_bases

    doc = load_toml_with_bases(Path(cfg.sim.toml_config))
    return str(doc.get("guidance", {}).get("neural_network", {}).get("mode", "full_neural"))
