"""Tests for PyTorch V2Policy Mamba export path (Phase 4a seam for Phase 4b)."""

from __future__ import annotations

import pytest
import torch
from aerocapture.training.rl.layers.mamba import MambaLayer
from aerocapture.training.rl.schemas import DenseSpec, MambaSpec


def test_export_v2_mamba_layer_emits_flat_keys():
    """Export a hand-constructed MambaLayer, verify the JSON v2 weights dict
    has the 5 flat Mamba keys at layer level (not nested).
    """
    from aerocapture.training.rl.export import _serialize_mamba_layer

    m = MambaLayer(input_size=4, d_state=2, dt_rank=1)
    m.double()
    with torch.no_grad():
        m.x_proj_w.normal_(0, 0.1)
        m.dt_proj_w.normal_(0, 0.1)
        m.dt_proj_b.normal_(0, 0.1)
        m.a_log.normal_(0, 0.1)
        m.d_skip.normal_(0, 0.1)

    weights_dict = _serialize_mamba_layer(m)
    # Flat at layer level -- NOT nested under further sub-dicts.
    assert set(weights_dict.keys()) == {"x_proj_w", "dt_proj_w", "dt_proj_b", "a_log", "d_skip"}
    # Shapes
    assert len(weights_dict["x_proj_w"]) == 5  # dt_rank + 2*d_state = 1 + 4 = 5
    assert len(weights_dict["x_proj_w"][0]) == 4  # input_size
    assert len(weights_dict["dt_proj_w"]) == 4  # input_size
    assert len(weights_dict["dt_proj_w"][0]) == 1  # dt_rank
    assert len(weights_dict["dt_proj_b"]) == 4  # input_size
    assert len(weights_dict["a_log"]) == 4  # input_size
    assert len(weights_dict["a_log"][0]) == 2  # d_state
    assert len(weights_dict["d_skip"]) == 4


def test_obs_norm_bake_in_rejects_mamba_as_layer_zero():
    """Phase 0 invariant: obs-normalizer bake-in is only safe into a Dense layer 0.
    Mamba's x_proj + softplus + A = -exp(a_log) nonlinearity means absorbing an
    affine input transform isn't closed-form.
    """
    from aerocapture.training.rl.export import _check_obs_norm_bake_compatibility

    arch = [
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]

    with pytest.raises(NotImplementedError, match="Mamba"):
        _check_obs_norm_bake_compatibility(arch, obs_normalizer_active=True)
