"""Torch-only unit tests for the Mamba3Layer mirror (flat round-trip + finiteness)."""

from __future__ import annotations

import numpy as np
import torch
from aerocapture.training.rl.layers.mamba3 import Mamba3Layer


def test_flat_roundtrip_all_flags() -> None:
    for trap in (False, True):
        for cplx in (False, True):
            m = Mamba3Layer(4, 3, 2, trap, cplx).double()
            n = len(m.to_flat())
            slab = np.linspace(-0.5, 0.5, n)
            m.from_flat(slab)
            assert np.array_equal(m.to_flat(), slab), (trap, cplx)


def test_forward_finite_all_flags() -> None:
    for trap in (False, True):
        for cplx in (False, True):
            m = Mamba3Layer(3, 4, 1, trap, cplx).double()
            m.from_flat(np.linspace(-0.3, 0.3, len(m.to_flat())))
            st = m.new_state()
            y = torch.zeros(3, dtype=torch.float64)
            for t in range(10):
                x = torch.tensor([0.1 * (d + t) for d in range(3)], dtype=torch.float64)
                y, st = m.forward_unbatched(x, st)
            assert torch.isfinite(y).all(), (trap, cplx)
