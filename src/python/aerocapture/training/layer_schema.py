"""Per-layer tensor schema: named tensors and shapes in canonical flat order.

Rust owns the table (`tensor_table!` in src/rust/src/data/neural/layers/*.rs,
exported as `aerocapture_rs.layer_schema`); it is at once the PSO flat order,
the JSON weight keys, and the parameter count. Python reads it here instead of
mirroring it. `_fallback_layer_schema` is the pure-Python copy for the
extension-less CI job, asserted equal to Rust for every layer type by
tests/test_layer_schema_drift.py.
"""

from __future__ import annotations

import json
import math
from functools import cache
from typing import Any

try:
    import aerocapture_rs as _aero_rs  # type: ignore[import-not-found, import-untyped]
except ImportError:
    _aero_rs = None

Schema = list[tuple[str, tuple[int, ...]]]


def resolve_mamba_dt_rank(entry: Any) -> int:
    """Resolve the Mamba `dt_rank` field, applying the paper default when absent.

    Accepts either a plain dict (pre-pydantic TOML entry) or a pydantic `MambaSpec`.
    Returns `max(1, input_size // 16)` when `dt_rank` is None / missing, matching
    `MambaSpec._resolve_and_validate_dt_rank` and Rust `TomlLayerSpec::to_layer_spec`.
    """
    input_size = int(entry["input_size"]) if isinstance(entry, dict) else int(entry.input_size)
    raw = entry.get("dt_rank") if isinstance(entry, dict) else getattr(entry, "dt_rank", None)
    return int(raw) if raw is not None else max(1, input_size // 16)


def layer_entry_dict(entry: Any) -> dict[str, Any]:
    """A v2 layer entry (dict or pydantic spec) as a plain dict with Mamba `dt_rank` resolved."""
    d: dict[str, Any] = dict(entry.model_dump()) if hasattr(entry, "model_dump") else dict(entry)
    if d["type"] in ("mamba", "mamba3"):
        d["dt_rank"] = resolve_mamba_dt_rank(d)
    return d


def layer_schema(entry: Any) -> Schema:
    """`[(name, shape), ...]` for one v2 layer entry; shape `()` scalar, `(n,)`, `(rows, cols)`."""
    return _cached_schema(json.dumps(layer_entry_dict(entry), sort_keys=True))


def layer_n_params(entry: Any) -> int:
    return sum(math.prod(shape) for _, shape in layer_schema(entry))


@cache
def _cached_schema(entry_json: str) -> Schema:
    if _aero_rs is not None:
        return [(name, tuple(shape)) for name, shape in _aero_rs.layer_schema(entry_json)]
    return _fallback_layer_schema(json.loads(entry_json))


def _fallback_layer_schema(e: dict[str, Any]) -> Schema:
    """Pure-Python mirror of the Rust tensor tables (no extension available)."""
    t = e["type"]
    if t == "dense":
        i, o = int(e["input_size"]), int(e["output_size"])
        return [("w", (o, i)), ("b", (o,))]
    if t in ("gru", "lstm"):
        i, h = int(e["input_size"]), int(e["hidden_size"])
        g = 3 * h if t == "gru" else 4 * h
        return [("weight_ih", (g, i)), ("weight_hh", (g, h)), ("bias_ih", (g,)), ("bias_hh", (g,))]
    if t == "window":
        return []
    if t == "transformer":
        d, f = int(e["d_model"]), int(e["d_ffn"])
        proj: Schema = [(w, (d, d)) for w in ("w_q", "w_k", "w_v", "w_o")]
        out: Schema = []
        for (w, ws), bias in zip(proj, ("b_q", "b_k", "b_v", "b_o"), strict=True):
            out += [(w, ws), (bias, (d,))]
        out += [("w_ffn1", (f, d)), ("b_ffn1", (f,)), ("w_ffn2", (d, f)), ("b_ffn2", (d,))]
        out += [(ln, (d,)) for ln in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta")]
        return out
    if t in ("mamba", "mamba3"):
        i, n, r = int(e["input_size"]), int(e["d_state"]), resolve_mamba_dt_rank(e)
        s: Schema = [("x_proj_w", (r + 2 * n, i)), ("dt_proj_w", (i, r)), ("dt_proj_b", (i,)), ("a_log", (i, n))]
        if t == "mamba3" and e.get("state_mode", "real") == "complex":
            s.append(("a_imag", (i, n)))
        if t == "mamba3" and e.get("discretization", "euler") == "trapezoidal":
            s.append(("lambda_logit", (i,)))
        s.append(("d_skip", (i,)))
        return s
    if t == "cfc":
        i, h, b = int(e["input_size"]), int(e["hidden_size"]), int(e["backbone_units"])
        s = [("w_bb", (b, i + h)), ("b_bb", (b,))]
        for head in ("ff1", "ff2", "ta", "tb"):
            s += [(f"w_{head}", (h, b)), (f"b_{head}", (h,))]
        return s
    if t == "slstm":
        i, h = int(e["input_size"]), int(e["hidden_size"])
        return [("weight_ih", (4 * h, i)), ("weight_hh", (4 * h, h)), ("bias", (4 * h,))]
    if t == "mlstm":
        i, h = int(e["input_size"]), int(e["hidden_size"])
        s = []
        for p in ("q", "k", "v", "o"):
            s += [(f"w_{p}", (h, i)), (f"b_{p}", (h,))]
        return s + [("w_i", (i,)), ("b_i", ()), ("w_f", (i,)), ("b_f", ())]
    raise ValueError(f"Unknown v2 layer type: {t!r}")
