"""One row per NN layer type: the table behind `test_nn_pso_smoke.py` (PSO
serialization round-trip) and `test_nn_equivalence.py` (Rust <-> torch mirror).

Adding a layer type is one `ArchCase` row here; `test_archs_cover_every_layer_type`
fails until the row exists. Tensor names and shapes come from `layer_schema`
(the Python view of the Rust tensor table), never from a test. Init seeds, init
ranges and tolerances are the ones the retired per-type gate files used: they
keep each gate away from saturation, and the tolerance is per type (1e-14 for
mamba, 1e-12 for the probe types / mamba3, 1e-10 elsewhere). The input
sequences are drawn from one seed for every row (standard normal, as before).

Not collected by pytest (no `test_` prefix). Must not import `aerocapture_rs`
at module level: the pure-Python CI job imports it through the test modules'
`importorskip`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aerocapture.training.layer_schema import layer_entry_dict, layer_schema
from aerocapture.training.torch_mirror.layers.cfc import CfcLayer
from aerocapture.training.torch_mirror.layers.dense import DenseLayer
from aerocapture.training.torch_mirror.layers.gru import GruLayer
from aerocapture.training.torch_mirror.layers.lstm import LstmLayer
from aerocapture.training.torch_mirror.layers.mamba import MambaLayer
from aerocapture.training.torch_mirror.layers.mamba3 import Mamba3Layer
from aerocapture.training.torch_mirror.layers.mlstm import MlstmLayer
from aerocapture.training.torch_mirror.layers.slstm import SlstmLayer
from aerocapture.training.torch_mirror.layers.transformer import TransformerLayer
from aerocapture.training.torch_mirror.layers.window import WindowLayer
from torch import Tensor, nn

Entry = dict[str, Any]


@dataclass(frozen=True)
class TrainCase:
    """A full-size architecture + training TOML for the 2-generation PSO `train()` smoke."""

    arch: list[Entry]
    toml: str
    n_inputs: int


@dataclass(frozen=True)
class ArchCase:
    name: str
    arch: list[Entry]  # reduced stack; the layer under test sits between two Dense layers (Window leads)
    n_params: int  # hand-computed oracle, independent of layer_n_params
    tol: float  # cross-language max-abs-diff gate
    mirror: Callable[[], list[nn.Module]]  # seeded, initialized f64 torch modules matching `arch`
    slow: bool = True  # marks this row's 100-step equivalence test
    stateful: bool = True  # False for a stack with no state (dense only)
    train: TrainCase | None = None

    @property
    def n_inputs(self) -> int:
        first = self.arch[0]
        return int(first["input_size"]) if "input_size" in first else int(first["d_model"])


def _uniform(t: Tensor, lo: float, hi: float) -> None:
    torch.nn.init.uniform_(t, lo, hi)


def dense(i: int, o: int, act: str) -> DenseLayer:
    return DenseLayer(input_size=i, output_size=o, activation=act).double()


def init_dense(layers: list[DenseLayer], w: float, b: float) -> None:
    for d in layers:
        _uniform(d.linear.weight, -w, w)
        _uniform(d.linear.bias, -b, b)


def _mirror_dense() -> list[nn.Module]:
    torch.manual_seed(42)
    d0, d1 = dense(5, 8, "tanh"), dense(8, 2, "linear")
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.1)
    return [d0, d1]


def _mirror_gru() -> list[nn.Module]:
    torch.manual_seed(42)
    d0, gru, d1 = dense(4, 8, "tanh"), GruLayer(input_size=8, hidden_size=8).double(), dense(8, 2, "linear")
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.3)
        for p in gru.parameters():
            _uniform(p, -0.3, 0.3)
    return [d0, gru, d1]


def _mirror_lstm() -> list[nn.Module]:
    torch.manual_seed(1337)
    d0, lstm, d1 = dense(4, 8, "tanh"), LstmLayer(input_size=8, hidden_size=4).double(), dense(4, 2, "linear")
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.3)
        for p in lstm.parameters():
            _uniform(p, -0.3, 0.3)
        # Forget bias (gate order i/f/g/o) toward "remember" so the cell state accumulates and the forget gate is exercised.
        h = lstm.hidden_size
        lstm.bias_ih[h : 2 * h] = 1.0 + 0.1 * torch.randn(h, dtype=torch.float64)
    return [d0, lstm, d1]


def _mirror_window() -> list[nn.Module]:
    rng = np.random.default_rng(2026)
    window, d0, d1 = WindowLayer(input_size=4, n_steps=4).double(), dense(16, 4, "tanh"), dense(4, 2, "linear")
    with torch.no_grad():
        for d in (d0, d1):
            d.linear.weight.copy_(torch.tensor(rng.normal(0.0, 0.3, tuple(d.linear.weight.shape)), dtype=torch.float64))
            d.linear.bias.copy_(torch.tensor(rng.normal(0.0, 0.1, tuple(d.linear.bias.shape)), dtype=torch.float64))
    return [window, d0, d1]


def _mirror_transformer() -> list[nn.Module]:
    torch.manual_seed(0)
    d0 = dense(8, 16, "linear")
    tr = TransformerLayer(d_model=16, n_heads=2, d_ffn=32, n_seq=8).double()
    d1 = dense(16, 2, "linear")
    # Small range keeps outputs bounded and avoids softmax saturation that could mask drift.
    with torch.no_grad():
        for lin in (d0.linear, tr.w_q, tr.w_k, tr.w_v, tr.w_o, tr.w_ffn1, tr.w_ffn2, d1.linear):
            _uniform(lin.weight, -0.1, 0.1)
            _uniform(lin.bias, -0.05, 0.05)
        _uniform(tr.ln1_gamma, 0.9, 1.1)
        _uniform(tr.ln1_beta, -0.05, 0.05)
        _uniform(tr.ln2_gamma, 0.9, 1.1)
        _uniform(tr.ln2_beta, -0.05, 0.05)
    return [d0, tr, d1]


def init_mamba_core(m: MambaLayer | Mamba3Layer) -> None:
    """The retired mamba gates' init: dt_proj_b centered for softplus, a_log in the HiPPO-ish range."""
    _uniform(m.x_proj_w, -0.3, 0.3)
    _uniform(m.dt_proj_w, -0.3, 0.3)
    _uniform(m.dt_proj_b, -0.5, 0.5)
    _uniform(m.a_log, 0.0, 2.0)
    m.d_skip.fill_(1.0)


def _mirror_mamba() -> list[nn.Module]:
    torch.manual_seed(0)
    d0, m, d1 = dense(4, 8, "tanh"), MambaLayer(input_size=8, d_state=4, dt_rank=2).double(), dense(8, 2, "linear")
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.3)
        init_mamba_core(m)
    return [d0, m, d1]


def _mirror_mamba3(trapezoidal: bool, complex_mode: bool) -> Callable[[], list[nn.Module]]:
    def build() -> list[nn.Module]:
        torch.manual_seed(0)
        d0 = dense(4, 8, "tanh")
        m = Mamba3Layer(input_size=8, d_state=4, dt_rank=2, trapezoidal=trapezoidal, complex=complex_mode).double()
        d1 = dense(8, 2, "linear")
        with torch.no_grad():
            init_dense([d0, d1], 0.3, 0.3)
            init_mamba_core(m)
            if m.a_imag is not None:
                _uniform(m.a_imag, -1.5, 1.5)  # rotation frequency
            if m.lambda_logit is not None:
                _uniform(m.lambda_logit, -2.0, 6.0)  # sweeps euler..trapezoidal
        return [d0, m, d1]

    return build


def _mirror_cfc() -> list[nn.Module]:
    torch.manual_seed(0)
    d0, cfc, d1 = dense(4, 8, "tanh"), CfcLayer(input_size=8, hidden_size=6, backbone_units=5).double(), dense(6, 2, "linear")
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.3)
        for p in cfc.parameters():
            _uniform(p, -0.6, 0.6)
    return [d0, cfc, d1]


def _mirror_slstm() -> list[nn.Module]:
    torch.manual_seed(0)
    d0, s, d1 = dense(4, 8, "tanh"), SlstmLayer(input_size=8, hidden_size=6).double(), dense(6, 2, "linear")
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.3)
        _uniform(s.weight_ih, -0.6, 0.6)
        _uniform(s.weight_hh, -0.6, 0.6)
        _uniform(s.bias, 0.0, 2.0)
    return [d0, s, d1]


def _mirror_mlstm() -> list[nn.Module]:
    torch.manual_seed(0)
    d0, m, d1 = dense(4, 8, "tanh"), MlstmLayer(input_size=8, hidden_size=6).double(), dense(6, 2, "linear")
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.3)
        for mat in (m.w_q, m.w_k, m.w_v, m.w_o, m.w_i, m.w_f):
            _uniform(mat, -0.5, 0.5)
        for vec in (m.b_q, m.b_k, m.b_v, m.b_o):
            _uniform(vec, -0.5, 0.5)
        m.b_i.fill_(0.3)
        m.b_f.fill_(2.0)  # forget gate toward remember (exp gating)
    return [d0, m, d1]


def _d(i: int, o: int, act: str) -> Entry:
    return {"type": "dense", "input_size": i, "output_size": o, "activation": act}


def _mamba3_row(disc: str, sm: str) -> ArchCase:
    extra = (32 if sm == "complex" else 0) + (8 if disc == "trapezoidal" else 0)  # a_imag 8x4, lambda_logit 8
    return ArchCase(
        name=f"mamba3_{disc}_{sm}",
        arch=[_d(4, 8, "tanh"), {"type": "mamba3", "input_size": 8, "d_state": 4, "dt_rank": 2, "discretization": disc, "state_mode": sm}, _d(8, 2, "linear")],
        n_params=202 + extra,
        tol=1e-12,  # complex adds a multiply vs the real mamba path's 1e-14; observed ~1e-14
        mirror=_mirror_mamba3(disc == "trapezoidal", sm == "complex"),
    )


_ROWS: list[ArchCase] = [
    # dense: 5*8+8 + 8*2+2 = 48 + 18
    ArchCase("dense", [_d(5, 8, "tanh"), _d(8, 2, "linear")], 66, 1e-10, _mirror_dense, slow=False, stateful=False),
    # gru(8, 8): 3*8*8 + 3*8*8 + 6*8 = 432; dense 40 + 18
    ArchCase(
        "gru",
        [_d(4, 8, "tanh"), {"type": "gru", "input_size": 8, "hidden_size": 8}, _d(8, 2, "linear")],
        490,
        1e-10,
        _mirror_gru,
        slow=False,
        train=TrainCase(
            arch=[_d(25, 8, "tanh"), {"type": "gru", "input_size": 8, "hidden_size": 8}, _d(8, 2, "linear")],
            toml="configs/training/msr_aller_gru_pso_train.toml",
            n_inputs=25,
        ),
    ),
    # lstm(8, 4): 4*4*8 + 4*4*4 + 2*16 = 224; dense 40 + 10
    ArchCase(
        "lstm",
        [_d(4, 8, "tanh"), {"type": "lstm", "input_size": 8, "hidden_size": 4}, _d(4, 2, "linear")],
        274,
        1e-10,
        _mirror_lstm,
        slow=False,
        train=TrainCase(
            arch=[_d(21, 8, "tanh"), {"type": "lstm", "input_size": 8, "hidden_size": 8}, _d(8, 2, "linear")],
            toml="configs/training/msr_aller_lstm_pso_train.toml",
            n_inputs=21,
        ),
    ),
    # window: 0; dense 16*4+4 = 68 + 10
    ArchCase("window", [{"type": "window", "input_size": 4, "n_steps": 4}, _d(16, 4, "tanh"), _d(4, 2, "linear")], 78, 1e-10, _mirror_window, slow=False),
    # transformer(16, 2, 32, 8): 4*(16*16+16) + (16*32+32) + (32*16+16) + 4*16 = 1088 + 544 + 528 + 64 = 2224; dense 144 + 34
    ArchCase(
        "transformer",
        [_d(8, 16, "linear"), {"type": "transformer", "d_model": 16, "n_heads": 2, "d_ffn": 32, "n_seq": 8}, _d(16, 2, "linear")],
        2402,
        1e-10,
        _mirror_transformer,
    ),
    # mamba(8, 4, 2): 8*(3*4 + 2*2 + 2) = 144; dense 40 + 18
    ArchCase("mamba", [_d(4, 8, "tanh"), {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2}, _d(8, 2, "linear")], 202, 1e-14, _mirror_mamba),
    _mamba3_row("euler", "real"),
    _mamba3_row("trapezoidal", "real"),
    _mamba3_row("euler", "complex"),
    _mamba3_row("trapezoidal", "complex"),
    # cfc(8, 6, 5): backbone 5*14+5 = 75, four heads 4*(6*5+6) = 144 -> 219; dense 40 + 14
    ArchCase("cfc", [_d(4, 8, "tanh"), {"type": "cfc", "input_size": 8, "hidden_size": 6, "backbone_units": 5}, _d(6, 2, "linear")], 273, 1e-12, _mirror_cfc),
    # slstm(8, 6): 24*8 + 24*6 + 24 = 360; dense 40 + 14
    ArchCase("slstm", [_d(4, 8, "tanh"), {"type": "slstm", "input_size": 8, "hidden_size": 6}, _d(6, 2, "linear")], 414, 1e-12, _mirror_slstm),
    # mlstm(8, 6): 4*(6*8+6) = 216 + w_i 8 + b_i 1 + w_f 8 + b_f 1 = 234; dense 40 + 14
    ArchCase("mlstm", [_d(4, 8, "tanh"), {"type": "mlstm", "input_size": 8, "hidden_size": 6}, _d(6, 2, "linear")], 288, 1e-12, _mirror_mlstm),
]
assert len({c.name for c in _ROWS}) == len(_ROWS), "duplicate ArchCase name: a copy-pasted row would silently replace its sibling"
ARCHS: dict[str, ArchCase] = {c.name: c for c in _ROWS}


def _tensor_for(layer: nn.Module, name: str) -> Tensor:
    """The mirror tensor behind a tensor-table name: nn.Parameter mirrors expose the name
    directly; Dense keeps `linear`; the Transformer wraps each `w_x` in an nn.Linear whose bias is `b_x`."""
    if isinstance(layer, DenseLayer):
        return layer.linear.weight if name == "w" else layer.linear.bias
    attr = getattr(layer, name, None)
    if isinstance(attr, Tensor):
        return attr
    if isinstance(attr, nn.Linear):
        return attr.weight
    if name.startswith("b_"):
        lin = getattr(layer, "w_" + name[2:], None)
        if isinstance(lin, nn.Linear):
            return lin.bias
    raise AssertionError(f"{type(layer).__name__} has no tensor for schema name {name!r}")


def mirror_weights(layer: nn.Module, entry: Entry) -> dict[str, Any]:
    """`{tensor-table name: nested list}` for one layer, shapes checked against `layer_schema`."""
    out: dict[str, Any] = {}
    for name, shape in layer_schema(entry):
        t = _tensor_for(layer, name)
        assert tuple(t.shape) == tuple(shape), f"{entry['type']}.{name}: mirror {tuple(t.shape)} vs schema {shape}"
        out[name] = t.detach().tolist()
    return out


def write_model_json(path: Path, arch: list[Entry], modules: list[nn.Module]) -> Path:
    """A v2 model JSON assembled from the mirror modules (zero-param layers get no weights entry)."""
    weights = {f"layer_{i}": mirror_weights(m, e) for i, (e, m) in enumerate(zip(arch, modules, strict=True)) if layer_schema(e)}
    path.write_text(json.dumps({"format_version": 2, "architecture": [layer_entry_dict(e) for e in arch], "weights": weights}))
    return path


def _new_state(m: nn.Module) -> Any:
    if isinstance(m, DenseLayer):
        return None
    if isinstance(m, (CfcLayer, SlstmLayer, MlstmLayer, Mamba3Layer)):
        return m.new_state()
    if isinstance(m, MambaLayer):
        return m.new_state(batch_size=1)[0]
    return m.new_state(batch_size=1)  # type: ignore[operator]  # gru / lstm / window / transformer


def _step(m: nn.Module, x: Tensor, state: Any) -> tuple[Tensor, Any]:
    """One tick through one mirror, hiding the three step protocols (unbatched, mamba's
    sliced state, batched `(1, d)` rows). Dense accepts either shape."""
    if isinstance(m, (CfcLayer, SlstmLayer, MlstmLayer, Mamba3Layer)):
        return m.forward_unbatched(x.reshape(-1), state)
    if isinstance(m, (DenseLayer, MambaLayer)):
        y, new_state = m(x, state)
    else:
        y, new_state = m(x if x.ndim == 2 else x.unsqueeze(0), state)
    return y, new_state


def run_mirror(modules: list[nn.Module], inputs: np.ndarray) -> np.ndarray:
    """Thread one state per layer through `inputs` `(T, d_in)`; returns the `(T, 2)` outputs."""
    states = [_new_state(m) for m in modules]
    for m in modules:
        m.eval()
    out = np.empty((inputs.shape[0], 2), dtype=np.float64)
    with torch.no_grad():
        for t in range(inputs.shape[0]):
            x = torch.tensor(inputs[t], dtype=torch.float64)
            for i, m in enumerate(modules):
                x, states[i] = _step(m, x, states[i])
            out[t] = x.reshape(-1).numpy()
    return out
