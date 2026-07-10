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
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aerocapture.training.ablation import _load_cost_kwargs, _mean_per_sim_cost, _resolve_nn_path
from aerocapture.training.charts import is_captured
from aerocapture.training.parquet_output import DV_TOTAL_RAW_INDEX

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


def _variant_metrics(final_records: npt.NDArray[np.float64], cost_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Capture rate, mean training cost, and captured-only DV percentiles for one MC batch."""
    captured = is_captured(final_records)
    capture_rate = float(np.mean(captured))
    mean_cost = _mean_per_sim_cost(final_records, cost_kwargs)
    if np.any(captured):
        dv = final_records[captured, DV_TOTAL_RAW_INDEX]
        dv_p50: float | None = float(np.percentile(dv, 50))
        dv_p95: float | None = float(np.percentile(dv, 95))
    else:
        dv_p50 = None
        dv_p95 = None
    return {"capture_rate": capture_rate, "mean_cost": mean_cost, "dv_p50": dv_p50, "dv_p95": dv_p95}


def run_quant_sweep(
    toml_path: str,
    bits: tuple[int, ...] = (8, 6, 4, 3, 2),
    granularities: tuple[str, ...] = ("per_channel", "per_tensor"),
    n_sims: int = 1000,
    sim_timeout_secs: float | None = None,
    cost_transform: str | None = None,
) -> dict[str, Any]:
    """Sweep weight-only PTQ over (granularity, bits); evaluate each on the same MC seeds.

    monte_carlo.seed is left to the config so every variant sees identical dispersions;
    each delta vs the fp baseline is therefore pure quantization effect.
    """
    import aerocapture_rs

    nn_path = _resolve_nn_path(toml_path)
    model_json = json.loads(nn_path.read_text())
    cost_kwargs = _load_cost_kwargs(toml_path, cost_transform=cost_transform)
    common = {"simulation.n_sims": n_sims}

    baseline_res = aerocapture_rs.run_mc(toml_path, overrides=common, sim_timeout_secs=sim_timeout_secs)
    baseline = _variant_metrics(baseline_res.final_records, cost_kwargs)

    variants: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "quant_model.json"
        for gran in granularities:
            for b in bits:
                tmp.write_text(json.dumps(quantize_model_weights(model_json, b, gran)))
                overrides = {**common, "data.neural_network": str(tmp)}
                res = aerocapture_rs.run_mc(toml_path, overrides=overrides, sim_timeout_secs=sim_timeout_secs)
                m = _variant_metrics(res.final_records, cost_kwargs)
                m["granularity"] = gran
                m["bits"] = b
                m["delta_capture_rate"] = m["capture_rate"] - baseline["capture_rate"]
                m["delta_mean_cost"] = m["mean_cost"] - baseline["mean_cost"]
                variants.append(m)

    return {
        "baseline": baseline,
        "variants": variants,
        "n_sims": n_sims,
        "bits": list(bits),
        "granularities": list(granularities),
        "model_path": str(nn_path),
    }


def _print_table(results: dict[str, Any]) -> None:
    b = results["baseline"]
    bp50 = "-" if b["dv_p50"] is None else f"{b['dv_p50']:.1f}"
    bp95 = "-" if b["dv_p95"] is None else f"{b['dv_p95']:.1f}"
    print(f"baseline (fp): capture={b['capture_rate']:.3f}  mean_cost={b['mean_cost']:.4g}  dv_p50={bp50}  dv_p95={bp95}")
    print(f"{'gran':<12}{'bits':>5}{'capture':>9}{'d_cap':>9}{'mean_cost':>14}{'d_cost':>14}{'dv_p50':>9}{'dv_p95':>9}")
    for v in results["variants"]:
        p50 = "-" if v["dv_p50"] is None else f"{v['dv_p50']:.1f}"
        p95 = "-" if v["dv_p95"] is None else f"{v['dv_p95']:.1f}"
        print(
            f"{v['granularity']:<12}{v['bits']:>5}{v['capture_rate']:>9.3f}{v['delta_capture_rate']:>+9.3f}"
            f"{v['mean_cost']:>14.4g}{v['delta_mean_cost']:>+14.4g}{p50:>9}{p95:>9}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Weight-only PTQ sweep for the dense NN guidance policy")
    parser.add_argument("output_dir", help="directory for quantization_results.json + quantization_sweep.svg")
    parser.add_argument("--toml", required=True, help="training/nominal config whose data.neural_network is the dense model")
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    parser.add_argument("--granularity", nargs="+", default=["per_channel", "per_tensor"], choices=list(_GRANULARITIES))
    parser.add_argument("--sim-timeout", type=float, default=None)
    parser.add_argument(
        "--cost-transform",
        default="linear",
        choices=["linear", "sqrt", "log", "squared", "cubed"],
        help="rescaling for the reported mean_cost; default linear = interpretable DV+penalties "
        "(overrides the config's optimization-shaping transform, which can blow mean_cost up to ~1e9)",
    )
    args = parser.parse_args(argv)

    results = run_quant_sweep(
        args.toml,
        bits=tuple(args.bits),
        granularities=tuple(args.granularity),
        n_sims=args.n_sims,
        sim_timeout_secs=args.sim_timeout,
        cost_transform=args.cost_transform,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quantization_results.json").write_text(json.dumps(results, indent=2))

    from aerocapture.training.charts_quant import chart_quant_sweep

    chart_quant_sweep(results, str(out_dir / "quantization_sweep.svg"))
    _print_table(results)
    print(f"\nWrote {out_dir / 'quantization_results.json'} and {out_dir / 'quantization_sweep.svg'}")


if __name__ == "__main__":
    main()
