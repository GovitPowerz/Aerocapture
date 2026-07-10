"""Weight-only post-training quantization (PTQ) sweep for the dense+Mamba NN guidance policy.

Rounds weight matrices to a symmetric b-bit grid (per-channel or per-tensor)
and stores the rounded values back as f64 -- "fake quant". The Rust runtime then runs
its normal f64 matmul, so this measures the accuracy impact of b-bit *weights* exactly
for a weight-only scheme, with no integer kernel. Biases, activations, hidden state,
and input normalization stay fp64. Dense and Mamba layers supported via tensor policy.

Spec: docs/superpowers/specs/2026-06-05-dense-nn-ptq-sweep-design.md
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_GRANULARITIES = ("per_channel", "per_tensor")
_TENSOR_POLICIES = ("all", "proj_only")


def _layer_types(model_json: dict) -> list[str]:
    """Return per-layer type strings for v2 (tagged list) or v1 (dense-only) JSON."""
    arch = model_json.get("architecture")
    if isinstance(arch, list):  # v2: tagged-layer list
        return [str(layer.get("type", "dense")) for layer in arch]
    if isinstance(arch, dict):  # v1: {"layers": [...], "activations": [...]} -> all dense
        return ["dense"] * len(arch.get("activations", []))
    raise ValueError("model JSON has no recognizable 'architecture'")


def _quantize_matrix(w: npt.NDArray[np.float64], n_bits: int, granularity: str) -> npt.NDArray[np.float64]:
    """Symmetric fake-quant of one weight matrix [n_out, n_in]. Rows are output channels."""
    qmax = 2 ** (n_bits - 1) - 1  # symmetric: drop the extra -2^(b-1) level so 0 maps to 0
    # per_channel: one scale per output row; per_tensor: one scale for the whole layer
    amax = np.max(np.abs(w), axis=1, keepdims=True) if granularity == "per_channel" else np.max(np.abs(w))
    scale = np.where(amax == 0.0, 1.0, amax / qmax)  # 0-group -> scale 1.0 (leaves zeros, no div0)
    q = np.clip(np.round(w / scale), -qmax, qmax)
    result: npt.NDArray[np.float64] = q * scale
    return result


def _quantize_vector(v: npt.NDArray[np.float64], n_bits: int) -> npt.NDArray[np.float64]:
    """Symmetric fake-quant of a 1-D tensor: always a single per-tensor scale
    (per-channel on a vector would be per-element, i.e. lossless and meaningless)."""
    return _quantize_matrix(v.reshape(1, -1), n_bits, "per_tensor").reshape(-1)


def _quantizable_tensors(model_json: dict, tensor_policy: str) -> list[tuple[str, int, str, bool]]:
    """(key, layer_idx, field, is_1d) for every tensor the policy quantizes.

    dense: `w` only (biases stay fp). mamba: `x_proj_w` + `dt_proj_w` always;
    `a_log` + `d_skip` only under "all"; `dt_proj_b` never (it is a bias).
    """
    types = _layer_types(model_json)
    unsupported = sorted({t for t in types if t not in ("dense", "mamba")})
    if unsupported:
        raise ValueError(f"quantization supports dense+mamba models; found layer types {unsupported}")
    out: list[tuple[str, int, str, bool]] = []
    for i, t in enumerate(types):
        if t == "dense":
            out.append((f"layer_{i}.w", i, "w", False))
        else:
            out.append((f"layer_{i}.x_proj_w", i, "x_proj_w", False))
            out.append((f"layer_{i}.dt_proj_w", i, "dt_proj_w", False))
            if tensor_policy == "all":
                out.append((f"layer_{i}.a_log", i, "a_log", False))
                out.append((f"layer_{i}.d_skip", i, "d_skip", True))
    return out


def _validate_quant_args(n_bits: int, granularity: str, tensor_policy: str) -> None:
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2 (got {n_bits}); binary weights are out of scope")
    if granularity not in _GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r} (expected one of {_GRANULARITIES})")
    if tensor_policy not in _TENSOR_POLICIES:
        raise ValueError(f"unknown tensor_policy {tensor_policy!r} (expected one of {_TENSOR_POLICIES})")


def memory_footprint(architecture: list[dict], n_bits: int, granularity: str, tensor_policy: str = "all") -> dict[str, int]:
    """Analytic deployed-model bytes: b-bit-packed quantized params + f32 scales + f32 fp params.

    fp-kept parameters are costed at f32 (the realistic flight deployment width),
    quantized parameters at ceil(n * b / 8) packed bytes, scales at f32 each.
    """
    from aerocapture.training.config import resolve_mamba_dt_rank

    _validate_quant_args(n_bits, granularity, tensor_policy)
    quant = fp = scales = 0
    for e in architecture:
        t = str(e.get("type", "dense"))
        n_in = int(e["input_size"])
        if t == "dense":
            n_out = int(e["output_size"])
            quant += n_out * n_in
            scales += n_out if granularity == "per_channel" else 1
            fp += n_out  # bias
        elif t == "mamba":
            d_state = int(e["d_state"])
            dt_rank = resolve_mamba_dt_rank(e)
            rows = dt_rank + 2 * d_state
            quant += rows * n_in + n_in * dt_rank  # x_proj_w + dt_proj_w
            scales += (rows + n_in) if granularity == "per_channel" else 2
            fp += n_in  # dt_proj_b
            if tensor_policy == "all":
                quant += n_in * d_state + n_in  # a_log + d_skip
                scales += (n_in if granularity == "per_channel" else 1) + 1  # a_log rows + d_skip per-tensor
            else:
                fp += n_in * d_state + n_in
        else:
            raise ValueError(f"quantization supports dense+mamba architectures; found {t!r}")
    quant_bytes = math.ceil(quant * n_bits / 8)
    return {
        "quant_params": quant,
        "fp_params": fp,
        "n_scales": scales,
        "quant_bytes": quant_bytes,
        "scale_bytes": scales * 4,
        "fp_bytes": fp * 4,
        "total_bytes": quant_bytes + scales * 4 + fp * 4,
        "f64_baseline_bytes": (quant + fp) * 8,
    }


def quantize_model_weights(
    model_json: dict,
    n_bits: int,
    granularity: str,
    tensor_policy: str = "all",
    only_tensor: str | None = None,
) -> dict:
    """Deep copy of model_json with the policy's tensors fake-quantized.

    `only_tensor` (a key from `_quantizable_tensors`, e.g. "layer_1.a_log")
    quantizes exactly that tensor group -- the leave-one-out probe. Biases,
    input_mask, normalization, output_param, architecture are never touched.
    """
    _validate_quant_args(n_bits, granularity, tensor_policy)
    targets = _quantizable_tensors(model_json, "all" if only_tensor is not None else tensor_policy)
    if only_tensor is not None:
        targets = [t for t in targets if t[0] == only_tensor]
        if not targets:
            known = [k for k, *_ in _quantizable_tensors(model_json, "all")]
            raise ValueError(f"unknown only_tensor {only_tensor!r} (expected one of {known})")
    out = copy.deepcopy(model_json)
    for _key, i, field, is_1d in targets:
        arr = np.asarray(out["weights"][f"layer_{i}"][field], dtype=np.float64)
        q = _quantize_vector(arr, n_bits) if is_1d else _quantize_matrix(arr, n_bits, granularity)
        out["weights"][f"layer_{i}"][field] = q.tolist()
    return out


def quantize_flat_weights_batch(
    weights: npt.NDArray[np.float64],
    architecture: list[dict],
    n_bits: int,
    granularity: str,
    tensor_policy: str = "all",
) -> npt.NDArray[np.float64]:
    """Fake-quantize the quantizable blocks of a (n_pop, n_w) flat-weight matrix.

    Mirrors `quantize_model_weights` on the PSO/GA flat layout (dense: w then b;
    mamba: x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip -- the canonical Rust
    `LayerWeights::to_flat` order). Biases and policy-excluded slabs pass through.
    Operates on the NN-weight slab ONLY: scaffolding genes travel through
    run_grid overrides, never through this array (exact-width assert below).
    """
    from aerocapture.training.config import resolve_mamba_dt_rank

    _validate_quant_args(n_bits, granularity, tensor_policy)
    unsupported = sorted({str(e.get("type", "dense")) for e in architecture} - {"dense", "mamba"})
    if unsupported:
        raise ValueError(f"quantization supports dense+mamba architectures; found {unsupported}")

    qmax = 2 ** (n_bits - 1) - 1
    out = weights.astype(np.float64).copy()
    n_pop = out.shape[0]

    def q2d(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # block: (n_pop, rows, cols); per_channel = one scale per output row
        amax = np.max(np.abs(block), axis=2, keepdims=True) if granularity == "per_channel" else np.max(np.abs(block), axis=(1, 2), keepdims=True)
        scale = np.where(amax == 0.0, 1.0, amax / qmax)
        result: npt.NDArray[np.float64] = np.clip(np.round(block / scale), -qmax, qmax) * scale
        return result

    def q1d(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # block: (n_pop, n); 1-D tensors always take a single per-tensor scale
        amax = np.max(np.abs(block), axis=1, keepdims=True)
        scale = np.where(amax == 0.0, 1.0, amax / qmax)
        result: npt.NDArray[np.float64] = np.clip(np.round(block / scale), -qmax, qmax) * scale
        return result

    off = 0
    for e in architecture:
        t = str(e.get("type", "dense"))
        n_in = int(e["input_size"])
        if t == "dense":
            n_out = int(e["output_size"])
            wsize = n_out * n_in
            out[:, off : off + wsize] = q2d(out[:, off : off + wsize].reshape(n_pop, n_out, n_in)).reshape(n_pop, wsize)
            off += wsize + n_out  # biases fp
        else:  # mamba
            d_state = int(e["d_state"])
            dt_rank = resolve_mamba_dt_rank(e)
            rows = dt_rank + 2 * d_state
            sz = rows * n_in  # x_proj_w
            out[:, off : off + sz] = q2d(out[:, off : off + sz].reshape(n_pop, rows, n_in)).reshape(n_pop, sz)
            off += sz
            sz = n_in * dt_rank  # dt_proj_w (per_channel = per row = per element at dt_rank 1: lossless by construction)
            out[:, off : off + sz] = q2d(out[:, off : off + sz].reshape(n_pop, n_in, dt_rank)).reshape(n_pop, sz)
            off += sz
            off += n_in  # dt_proj_b: bias, fp
            sz = n_in * d_state  # a_log
            if tensor_policy == "all":
                out[:, off : off + sz] = q2d(out[:, off : off + sz].reshape(n_pop, n_in, d_state)).reshape(n_pop, sz)
            off += sz
            if tensor_policy == "all":  # d_skip
                out[:, off : off + n_in] = q1d(out[:, off : off + n_in])
            off += n_in
    if off != out.shape[1]:
        raise ValueError(f"architecture flat width {off} != weights width {out.shape[1]}")
    return out


def _score_variant(
    toml_path: str,
    model_path: str | Path,
    seeds: list[int],
    cost_kwargs: dict[str, Any],
    extra_overrides: dict[str, Any],
    sim_timeout_secs: float | None,
) -> dict[str, Any]:
    """One MC batch on an explicit seed list for a pinned model; tail-led metrics."""
    import aerocapture_rs

    from aerocapture.training import charts
    from aerocapture.training.experiments.probe_common import cvar95
    from aerocapture.training.report import compute_eval_summary

    overrides = [{"simulation.n_sims": 1, "data.neural_network": str(model_path), "monte_carlo.seed": int(s), **extra_overrides} for s in seeds]
    batch = aerocapture_rs.run_batch(toml_path, overrides, n_threads=None, include_trajectories=False, sim_timeout_secs=sim_timeout_secs)
    final = np.array(batch.final_records, dtype=np.float64)
    summary = compute_eval_summary(final, n_sims=len(seeds), cost_kwargs=cost_kwargs)
    captured = charts.is_captured(final)
    dv = np.clip(final[captured, charts._FR_DV_TOTAL], charts.DV_FLOOR, charts.DV_CAP)
    viol = max(float(c["viol_pct"]) for c in summary["constraints"].values()) if summary["constraints"] else 0.0
    return {
        "capture_rate": float(summary["capture_rate"]),
        "dv_p50": float(np.percentile(dv, 50)) if dv.size else None,
        "dv_p95": float(np.percentile(dv, 95)) if dv.size else None,
        "dv_p99": float(np.percentile(dv, 99)) if dv.size else None,
        "dv_cvar95": cvar95(dv) if dv.size else None,
        "viol_pct": viol,
        "rms_cost": float(summary["cost"]["rms"]),
    }


def _resolve_pool(toml_path: str, pool_offset: int, n_sims: int) -> tuple[list[int], dict[str, Any]]:
    from aerocapture.training.evaluate import make_reserved_seeds
    from aerocapture.training.toml_utils import load_toml_with_bases

    base_mc_seed = int(load_toml_with_bases(Path(toml_path)).get("monte_carlo", {}).get("seed", 42))
    seeds = make_reserved_seeds(base_mc_seed, pool_offset, n_sims)
    return seeds, {"base_mc_seed": base_mc_seed, "offset": pool_offset, "n": n_sims}


def _scaffolding_overrides(params_dir: str | Path | None) -> dict[str, Any]:
    if params_dir is None:
        return {}
    from aerocapture.training.report import _load_nn_scaffolding_overrides

    d = Path(params_dir)
    return dict(_load_nn_scaffolding_overrides(d, d / f"optimized_{d.name}.toml"))


def _pick_verdict(variants: list[dict[str, Any]], bits: int) -> dict[str, Any]:
    """Pre-registered QAT-cell rule: among the `bits` cells, max capture rate,
    then min CVaR95 (NaN/None last), ties break toward per_channel then all."""
    cells = [v for v in variants if v["bits"] == bits]
    if not cells:
        raise ValueError(f"no {bits}-bit cells in the sweep grid")

    def key(v: dict[str, Any]) -> tuple[float, float, int, int]:
        cvar = v.get("dv_cvar95")
        cvar_f = float(cvar) if cvar is not None and np.isfinite(cvar) else float("inf")
        return (-round(float(v["capture_rate"]), 3), cvar_f, int(v["granularity"] != "per_channel"), int(v["tensor_policy"] != "all"))

    return min(cells, key=key)


def run_quant_sweep(
    toml_path: str,
    model_path: str | Path,
    params_dir: str | Path | None = None,
    bits: tuple[int, ...] = (8, 6, 4, 3, 2),
    granularities: tuple[str, ...] = ("per_channel", "per_tensor"),
    policies: tuple[str, ...] = ("all", "proj_only"),
    n_sims: int = 1000,
    pool_offset: int | None = None,
    loo_bits: int | None = 4,
    sim_timeout_secs: float | None = None,
    cost_transform: str | None = "linear",
) -> dict[str, Any]:
    """PTQ sensitivity sweep on a reserved pool with the co-trained scaffolding applied.

    Grid: bits x granularity x tensor_policy, each scored on the SAME seeds as the
    fp baseline, so every delta is pure quantization effect. When `loo_bits` is set,
    a leave-one-out pass quantizes one tensor group at a time at that bit width
    (granularity taken from the verdict cell). Memory rows via `memory_footprint`.
    """
    from aerocapture.training.ablation import _load_cost_kwargs
    from aerocapture.training.evaluate import HEADLINE_REQUOTE_SEED_OFFSET

    offset = HEADLINE_REQUOTE_SEED_OFFSET if pool_offset is None else pool_offset
    seeds, pool = _resolve_pool(toml_path, offset, n_sims)
    scaff = _scaffolding_overrides(params_dir)
    cost_kwargs = _load_cost_kwargs(toml_path, cost_transform=cost_transform)
    model_json = json.loads(Path(model_path).read_text())

    baseline = _score_variant(toml_path, model_path, seeds, cost_kwargs, scaff, sim_timeout_secs)

    def deltas(m: dict[str, Any]) -> dict[str, Any]:
        m["delta_capture_rate"] = m["capture_rate"] - baseline["capture_rate"]
        if m["dv_cvar95"] is not None and baseline["dv_cvar95"] is not None:
            m["delta_dv_cvar95"] = m["dv_cvar95"] - baseline["dv_cvar95"]
        else:
            m["delta_dv_cvar95"] = None
        return m

    variants: list[dict[str, Any]] = []
    loo: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "quant_model.json"
        for gran in granularities:
            for policy in policies:
                for b in bits:
                    tmp.write_text(json.dumps(quantize_model_weights(model_json, b, gran, policy)))
                    m = _score_variant(toml_path, tmp, seeds, cost_kwargs, scaff, sim_timeout_secs)
                    m.update({"granularity": gran, "tensor_policy": policy, "bits": b})
                    variants.append(deltas(m))
                    print(f"  scored bits={b} gran={gran} policy={policy}: capture={m['capture_rate']:.3f}")

        verdict = _pick_verdict(variants, loo_bits) if loo_bits is not None else None
        if loo_bits is not None:
            assert verdict is not None
            for key, *_ in _quantizable_tensors(model_json, "all"):
                tmp.write_text(json.dumps(quantize_model_weights(model_json, loo_bits, verdict["granularity"], "all", only_tensor=key)))
                m = _score_variant(toml_path, tmp, seeds, cost_kwargs, scaff, sim_timeout_secs)
                m.update({"tensor": key, "bits": loo_bits, "granularity": verdict["granularity"]})
                loo.append(deltas(m))
                print(f"  scored LOO {key}: capture={m['capture_rate']:.3f}")

    memory = [
        {"bits": b, "granularity": g, "tensor_policy": p, **memory_footprint(model_json["architecture"], b, g, p)}
        for g in granularities
        for p in policies
        for b in bits
    ]
    return {
        "baseline": baseline,
        "variants": variants,
        "loo": loo,
        "verdict": verdict,
        "memory": memory,
        "pool": pool,
        "n_sims": n_sims,
        "model_path": str(model_path),
        "params_dir": str(params_dir) if params_dir is not None else None,
        "scaffolding_applied": sorted(scaff),
    }


def run_finalists(
    toml_path: str,
    entries: list[dict[str, Any]],
    n_sims: int = 10000,
    pool_offset: int | None = None,
    sim_timeout_secs: float | None = None,
    cost_transform: str | None = "linear",
) -> dict[str, Any]:
    """Deep re-score (default n=10000) of finalist models on the same reserved pool.

    Entry: {"label", "model", "params_dir", "quantize": None | {"bits", "granularity", "tensor_policy"}}.
    QAT-deployed models pass quantize=None (their best_model.json is already on-grid);
    the PTQ finalist passes the verdict cell so the champion is rounded on the fly.
    """
    from aerocapture.training.ablation import _load_cost_kwargs
    from aerocapture.training.evaluate import HEADLINE_REQUOTE_SEED_OFFSET

    offset = HEADLINE_REQUOTE_SEED_OFFSET if pool_offset is None else pool_offset
    seeds, pool = _resolve_pool(toml_path, offset, n_sims)
    cost_kwargs = _load_cost_kwargs(toml_path, cost_transform=cost_transform)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, e in enumerate(entries):
            model_path = Path(e["model"])
            if e.get("quantize") is not None:
                q = e["quantize"]
                tmp = Path(tmpdir) / f"finalist_{i}.json"
                tmp.write_text(
                    json.dumps(quantize_model_weights(json.loads(model_path.read_text()), int(q["bits"]), str(q["granularity"]), str(q["tensor_policy"])))
                )
                model_path = tmp
            m = _score_variant(toml_path, model_path, seeds, cost_kwargs, _scaffolding_overrides(e.get("params_dir")), sim_timeout_secs)
            rows.append({"label": e["label"], "model": e["model"], "quantize": e.get("quantize"), **m})
            print(f"  finalist {e['label']}: capture={m['capture_rate']:.3f}")
    return {"finalists": rows, "pool": pool, "n_sims": n_sims}


def _print_table(results: dict[str, Any]) -> None:
    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.1f}"

    b = results["baseline"]
    print(f"baseline (fp): capture={b['capture_rate']:.3f}  dv_p50={fmt(b['dv_p50'])}  dv_p95={fmt(b['dv_p95'])}  cvar95={fmt(b['dv_cvar95'])}")
    print(f"{'gran':<12}{'policy':<11}{'bits':>5}{'capture':>9}{'d_cap':>9}{'dv_p50':>9}{'dv_p95':>9}{'cvar95':>9}{'d_cvar':>9}{'viol%':>7}")
    for v in results["variants"]:
        d_cvar = "-" if v["delta_dv_cvar95"] is None else f"{v['delta_dv_cvar95']:+.1f}"
        print(
            f"{v['granularity']:<12}{v['tensor_policy']:<11}{v['bits']:>5}{v['capture_rate']:>9.3f}{v['delta_capture_rate']:>+9.3f}"
            f"{fmt(v['dv_p50']):>9}{fmt(v['dv_p95']):>9}{fmt(v['dv_cvar95']):>9}{d_cvar:>9}{v['viol_pct']:>7.2f}"
        )
    for r in results["loo"]:
        d_cvar = "-" if r["delta_dv_cvar95"] is None else f"{r['delta_dv_cvar95']:+.1f}"
        print(f"LOO {r['tensor']:<22}{r['bits']:>3}b  capture={r['capture_rate']:.3f}  cvar95={fmt(r['dv_cvar95'])}  d_cvar={d_cvar}")
    if results.get("verdict"):
        v = results["verdict"]
        print(f"verdict (QAT cell @ {v['bits']}b): granularity={v['granularity']} tensor_policy={v['tensor_policy']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Weight-only PTQ sweep / finalists re-score for the NN guidance policy")
    parser.add_argument("output_dir", help="directory for quantization_results.json + SVGs")
    parser.add_argument("--toml", required=True, help="training config that resolves the mission pipeline (e.g. configs/training/sweep/mamba_p962.toml)")
    parser.add_argument("--model", required=True, help="model JSON to quantize/evaluate (pinned; the TOML deploy path is never read)")
    parser.add_argument("--params-dir", default=None, help="training dir whose best_params.json carries the co-trained scaffolding overrides")
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    parser.add_argument("--granularity", nargs="+", default=["per_channel", "per_tensor"], choices=list(_GRANULARITIES))
    parser.add_argument("--policies", nargs="+", default=["all", "proj_only"], choices=list(_TENSOR_POLICIES))
    parser.add_argument("--loo-bits", type=int, default=4)
    parser.add_argument("--no-loo", action="store_true")
    parser.add_argument("--pool-offset", type=int, default=None, help="reserved-pool offset (default HEADLINE_REQUOTE_SEED_OFFSET = 8M)")
    parser.add_argument("--sim-timeout", type=float, default=None)
    parser.add_argument("--finalists", default=None, help="JSON file with finalist entries; switches to the deep re-score mode")
    parser.add_argument(
        "--cost-transform",
        default="linear",
        choices=["linear", "sqrt", "log", "squared", "cubed"],
        help="rescaling for the reported rms_cost; default linear = interpretable DV+penalties",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.finalists is not None:
        entries = json.loads(Path(args.finalists).read_text())
        results = run_finalists(
            args.toml, entries, n_sims=args.n_sims, pool_offset=args.pool_offset, sim_timeout_secs=args.sim_timeout, cost_transform=args.cost_transform
        )
        (out_dir / "finalists_results.json").write_text(json.dumps(results, indent=2))
        for r in results["finalists"]:
            cells = " ".join(f"{k}={'-' if r[k] is None else f'{r[k]:.1f}'}" for k in ("dv_p50", "dv_p95", "dv_p99", "dv_cvar95"))
            print(f"{r['label']:<28} capture={r['capture_rate']:.4f} {cells}")
        print(f"\nWrote {out_dir / 'finalists_results.json'}")
        return

    results = run_quant_sweep(
        args.toml,
        args.model,
        params_dir=args.params_dir,
        bits=tuple(args.bits),
        granularities=tuple(args.granularity),
        policies=tuple(args.policies),
        n_sims=args.n_sims,
        pool_offset=args.pool_offset,
        loo_bits=None if args.no_loo else args.loo_bits,
        sim_timeout_secs=args.sim_timeout,
        cost_transform=args.cost_transform,
    )
    (out_dir / "quantization_results.json").write_text(json.dumps(results, indent=2))

    from aerocapture.training.charts_quant import chart_quant_loo, chart_quant_sweep

    chart_quant_sweep(results, str(out_dir / "quantization_sweep.svg"))
    if results["loo"]:
        chart_quant_loo(results, str(out_dir / "quantization_loo.svg"))
    _print_table(results)
    print(f"\nWrote {out_dir / 'quantization_results.json'} and SVGs")


if __name__ == "__main__":
    main()
