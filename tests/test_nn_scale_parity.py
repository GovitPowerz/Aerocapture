"""Parity guard: Rust neural.rs scale consts must match calibrate_inputs CURRENT_TRANSFORMS,
and the candidate-vector width must agree across Rust + the Python copies."""

import re
from pathlib import Path

from aerocapture.training.ablation import NN_INPUT_NAMES
from aerocapture.training.calibrate_inputs import (
    _AFFINE_CONST_NAME,
    _ASINH_CONST_NAME,
    CURRENT_TRANSFORMS,
)
from aerocapture.training.config import _RUNTIME_CANDIDATE_WIDTH

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NEURAL_RS = _REPO_ROOT / "src/rust/src/gnc/guidance/neural.rs"
_DATA_RS = _REPO_ROOT / "src/rust/src/data/neural.rs"


def _parse_rust_f64_consts(path: Path) -> dict[str, float]:
    text = path.read_text()
    out: dict[str, float] = {}
    for m in re.finditer(r"const\s+(\w+):\s*f64\s*=\s*([0-9eE.+\-]+)\s*;", text):
        out[m.group(1)] = float(m.group(2))
    return out


def test_rust_asinh_scales_match_current_transforms() -> None:
    consts = _parse_rust_f64_consts(_NEURAL_RS)
    for idx, transform in CURRENT_TRANSFORMS.items():
        if transform[0] != "asinh":
            continue
        name = _ASINH_CONST_NAME[idx]
        assert name in consts, f"Rust const {name} (idx {idx}) not found in neural.rs"
        assert abs(consts[name] - transform[1]) <= 1e-6 * abs(transform[1]), f"idx {idx} ({name}): Rust {consts[name]} != CURRENT_TRANSFORMS {transform[1]}"


def test_rust_affine_consts_match_current_transforms() -> None:
    consts = _parse_rust_f64_consts(_NEURAL_RS)
    for idx, transform in CURRENT_TRANSFORMS.items():
        if transform[0] != "affine_ch":
            continue
        cn, hn = _AFFINE_CONST_NAME[idx]
        _, center, half = transform
        assert abs(consts[cn] - center) <= 1e-6 * abs(center), f"idx {idx} center {cn}"
        assert abs(consts[hn] - half) <= 1e-6 * abs(half), f"idx {idx} half {hn}"


def test_candidate_width_agreement() -> None:
    # NN_FULL_INPUT_SIZE (Rust) == _RUNTIME_CANDIDATE_WIDTH (config.py) == len(NN_INPUT_NAMES)
    m = re.search(r"NN_FULL_INPUT_SIZE:\s*usize\s*=\s*(\d+)", _DATA_RS.read_text())
    assert m, "NN_FULL_INPUT_SIZE not found in data/neural.rs"
    rust_width = int(m.group(1))
    assert rust_width == _RUNTIME_CANDIDATE_WIDTH == len(NN_INPUT_NAMES), (
        f"width mismatch: Rust={rust_width}, config={_RUNTIME_CANDIDATE_WIDTH}, names={len(NN_INPUT_NAMES)}"
    )


def test_dv_sentinel_norm_is_1_5() -> None:
    consts = _parse_rust_f64_consts(_NEURAL_RS)
    assert consts["DV_SENTINEL_NORM"] == 1.5
