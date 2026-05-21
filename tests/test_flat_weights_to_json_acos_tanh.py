"""Tests for flat_weights_to_json's output_param plumbing (PSO production path)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


def test_flat_weights_to_json_embeds_acos_tanh_into_v2_file() -> None:
    arch = json.dumps([{"type": "dense", "input_size": 2, "output_size": 1, "activation": "tanh"}])
    flat = [0.1, 0.2, 0.3]  # 2*1 weights + 1 bias
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        aerocapture_rs.flat_weights_to_json(
            flat=flat,
            architecture_json=arch,
            path=path,
            output_param="acos_tanh",
        )
        data = json.loads(Path(path).read_text())
        assert data["output_param"] == "acos_tanh", f"expected acos_tanh in JSON, got {data}"
        assert data["format_version"] == 2
    finally:
        Path(path).unlink(missing_ok=True)


def test_flat_weights_to_json_embeds_atan2_signed_when_requested() -> None:
    arch = json.dumps([{"type": "dense", "input_size": 2, "output_size": 2, "activation": "asinh"}])
    flat = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 2*2 weights + 2 biases
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        aerocapture_rs.flat_weights_to_json(
            flat=flat,
            architecture_json=arch,
            path=path,
            output_param="atan2_signed",
        )
        data = json.loads(Path(path).read_text())
        # atan2_signed is the default; serde may either omit the field or
        # write it explicitly. Both round-trip to Atan2Signed at load time.
        assert data.get("output_param", "atan2_signed") == "atan2_signed"
    finally:
        Path(path).unlink(missing_ok=True)


def test_flat_weights_to_json_default_omits_or_writes_atan2_signed() -> None:
    arch = json.dumps([{"type": "dense", "input_size": 2, "output_size": 2, "activation": "asinh"}])
    flat = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        # No output_param argument — should default to Atan2Signed
        aerocapture_rs.flat_weights_to_json(flat=flat, architecture_json=arch, path=path)
        data = json.loads(Path(path).read_text())
        # Either omitted (loads as default) or "atan2_signed"
        assert data.get("output_param", "atan2_signed") == "atan2_signed"
    finally:
        Path(path).unlink(missing_ok=True)


def test_flat_weights_to_json_rejects_invalid_output_param() -> None:
    arch = json.dumps([{"type": "dense", "input_size": 2, "output_size": 1, "activation": "tanh"}])
    flat = [0.1, 0.2, 0.3]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(Exception, match="(?i)output_param|atan2_signed|acos_tanh"):
            aerocapture_rs.flat_weights_to_json(
                flat=flat,
                architecture_json=arch,
                path=path,
                output_param="banana",
            )
        with pytest.raises(Exception, match="(?i)output_param|atan2_signed|acos_tanh"):
            aerocapture_rs.flat_weights_to_json(
                flat=flat,
                architecture_json=arch,
                path=path,
                output_param="AcosTanh",  # case-sensitive: should reject CamelCase
            )
    finally:
        Path(path).unlink(missing_ok=True)


def test_flat_weights_to_json_acos_tanh_rejects_non_tanh_activation() -> None:
    """Catches the case where a PSO trainer accidentally produces a v2 JSON
    with output_param=acos_tanh but the last layer's activation is not tanh.
    The Rust validator catches this at from_flat_weights_v2 (added via fix #2)."""
    arch = json.dumps([{"type": "dense", "input_size": 2, "output_size": 1, "activation": "linear"}])
    flat = [0.1, 0.2, 0.3]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(Exception, match="(?i)tanh|acos"):
            aerocapture_rs.flat_weights_to_json(
                flat=flat,
                architecture_json=arch,
                path=path,
                output_param="acos_tanh",
            )
    finally:
        Path(path).unlink(missing_ok=True)
