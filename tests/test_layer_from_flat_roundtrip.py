"""Per-module to_flat ∘ from_flat == identity tests.

For each of the 6 layer types, build a module with random weights, compute
slab = m.to_flat(), create a fresh same-shape module, call m2.from_flat(slab),
and assert:
  1. np.array_equal(m2.to_flat(), slab)  -- byte-for-byte flat round-trip
  2. all parameter tensors are equal between m and m2
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _check_params_equal(m1: torch.nn.Module, m2: torch.nn.Module) -> None:
    sd1 = dict(m1.named_parameters())
    sd2 = dict(m2.named_parameters())
    assert set(sd1) == set(sd2), f"parameter name mismatch: {set(sd1)} vs {set(sd2)}"
    for name, p1 in sd1.items():
        p2 = sd2[name]
        assert torch.equal(p1, p2), f"parameter {name!r} differs after from_flat"


class TestDenseFromFlat:
    def test_roundtrip(self) -> None:
        from aerocapture.training.rl.layers.dense import DenseLayer

        rng = np.random.default_rng(42)
        m = DenseLayer(8, 4, "tanh").double()
        # Fill with random values so the round-trip is non-trivial.
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.from_numpy(rng.standard_normal(p.shape)))
        slab = m.to_flat()

        m2 = DenseLayer(8, 4, "tanh").double()
        m2.from_flat(slab)

        assert np.array_equal(m2.to_flat(), slab)
        _check_params_equal(m, m2)

    def test_flat_length(self) -> None:
        from aerocapture.training.rl.layers.dense import DenseLayer

        m = DenseLayer(5, 3, "linear").double()
        assert len(m.to_flat()) == 5 * 3 + 3  # 18


class TestGruFromFlat:
    def test_roundtrip(self) -> None:
        from aerocapture.training.rl.layers.gru import GruLayer

        rng = np.random.default_rng(7)
        m = GruLayer(6, 4).double()
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.from_numpy(rng.standard_normal(p.shape)))
        slab = m.to_flat()

        m2 = GruLayer(6, 4).double()
        m2.from_flat(slab)

        assert np.array_equal(m2.to_flat(), slab)
        _check_params_equal(m, m2)

    def test_flat_length(self) -> None:
        from aerocapture.training.rl.layers.gru import GruLayer

        m = GruLayer(6, 4).double()
        # 3*4*6 + 3*4*4 + 3*4 + 3*4 = 72 + 48 + 12 + 12 = 144
        assert len(m.to_flat()) == 3 * 4 * 6 + 3 * 4 * 4 + 2 * 3 * 4


class TestLstmFromFlat:
    def test_roundtrip(self) -> None:
        from aerocapture.training.rl.layers.lstm import LstmLayer

        rng = np.random.default_rng(13)
        m = LstmLayer(6, 4).double()
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.from_numpy(rng.standard_normal(p.shape)))
        slab = m.to_flat()

        m2 = LstmLayer(6, 4).double()
        m2.from_flat(slab)

        assert np.array_equal(m2.to_flat(), slab)
        _check_params_equal(m, m2)

    def test_flat_length(self) -> None:
        from aerocapture.training.rl.layers.lstm import LstmLayer

        m = LstmLayer(6, 4).double()
        # 4*4*6 + 4*4*4 + 4*4 + 4*4 = 96 + 64 + 16 + 16 = 192
        assert len(m.to_flat()) == 4 * 4 * 6 + 4 * 4 * 4 + 2 * 4 * 4


class TestWindowFromFlat:
    def test_empty_slab_noop(self) -> None:
        from aerocapture.training.rl.layers.window import WindowLayer

        m = WindowLayer(4, 3)
        slab = m.to_flat()
        assert slab.size == 0

        m2 = WindowLayer(4, 3)
        m2.from_flat(slab)  # must not raise
        assert m2.to_flat().size == 0

    def test_rejects_nonempty_slab(self) -> None:
        from aerocapture.training.rl.layers.window import WindowLayer

        m = WindowLayer(4, 3)
        with pytest.raises(AssertionError):
            m.from_flat(np.array([1.0, 2.0]))


class TestTransformerFromFlat:
    def test_roundtrip(self) -> None:
        from aerocapture.training.rl.layers.transformer import TransformerLayer

        rng = np.random.default_rng(99)
        m = TransformerLayer(d_model=8, n_heads=2, d_ffn=16, n_seq=4).double()
        with torch.no_grad():
            for _name, p in m.named_parameters():
                p.copy_(torch.from_numpy(rng.standard_normal(p.shape)))
        slab = m.to_flat()

        m2 = TransformerLayer(d_model=8, n_heads=2, d_ffn=16, n_seq=4).double()
        m2.from_flat(slab)

        assert np.array_equal(m2.to_flat(), slab)
        _check_params_equal(m, m2)

    def test_flat_length(self) -> None:
        from aerocapture.training.rl.layers.transformer import TransformerLayer

        d, f = 8, 16
        m = TransformerLayer(d_model=d, n_heads=2, d_ffn=f, n_seq=4).double()
        # 4 projections of (d*d + d) + 2 FFN layers + 4 LN vectors
        expected = 4 * (d * d + d) + (f * d + f) + (d * f + d) + 4 * d
        assert len(m.to_flat()) == expected


class TestMambaFromFlat:
    def test_roundtrip(self) -> None:
        from aerocapture.training.rl.layers.mamba import MambaLayer

        rng = np.random.default_rng(55)
        m = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.from_numpy(rng.standard_normal(p.shape)))
        slab = m.to_flat()

        m2 = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
        m2.from_flat(slab)

        assert np.array_equal(m2.to_flat(), slab)
        _check_params_equal(m, m2)

    def test_flat_length(self) -> None:
        from aerocapture.training.rl.layers.mamba import MambaLayer

        d, n, r = 8, 4, 2
        m = MambaLayer(input_size=d, d_state=n, dt_rank=r).double()
        # x_proj_w: (r + 2*n)*d, dt_proj_w: d*r, dt_proj_b: d, a_log: d*n, d_skip: d
        expected = (r + 2 * n) * d + d * r + d + d * n + d
        assert len(m.to_flat()) == expected
