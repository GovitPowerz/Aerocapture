"""Tests for return and observation normalizers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aerocapture.training.rl.normalizers import ObsNormalizer, ReturnNormalizer  # noqa: E402


class TestReturnNormalizer:
    def test_warmup_returns_unscaled(self) -> None:
        norm = ReturnNormalizer(warmup_episodes=10)
        for _ in range(5):
            norm.update_episode_return(-500.0)
        raw = np.array([-400.0, -600.0], dtype=np.float64)
        out = norm.normalize(raw)
        np.testing.assert_array_equal(out, raw)

    def test_post_warmup_scales_by_std(self) -> None:
        norm = ReturnNormalizer(warmup_episodes=2)
        norm.update_episode_return(-100.0)
        norm.update_episode_return(-300.0)
        # mean=-200, var=10000, std=100
        raw = np.array([-200.0], dtype=np.float64)
        out = norm.normalize(raw)
        assert abs(out[0] - (-200.0 / 100.0)) < 0.1

    def test_checkpoint_roundtrip(self) -> None:
        norm = ReturnNormalizer(warmup_episodes=2)
        for v in [-100.0, -200.0, -300.0]:
            norm.update_episode_return(v)
        state = norm.state_dict()
        norm2 = ReturnNormalizer(warmup_episodes=2)
        norm2.load_state_dict(state)
        raw = np.array([-250.0], dtype=np.float64)
        np.testing.assert_allclose(norm.normalize(raw), norm2.normalize(raw))


class TestObsNormalizer:
    def test_normalize_shape_preserved(self) -> None:
        norm = ObsNormalizer(obs_dim=4, warmup_steps=0)
        obs = np.ones((8, 4), dtype=np.float32)
        norm.update(obs)
        out = norm.normalize(obs)
        assert out.shape == (8, 4)
        assert out.dtype == np.float32

    def test_normalize_zero_mean_unit_var(self) -> None:
        rng = np.random.default_rng(42)
        norm = ObsNormalizer(obs_dim=3, warmup_steps=0)
        for _ in range(100):
            obs = rng.standard_normal((64, 3)).astype(np.float32) * 10 + 5
            norm.update(obs)
        obs = rng.standard_normal((64, 3)).astype(np.float32) * 10 + 5
        out = norm.normalize(obs)
        assert abs(np.mean(out)) < 2.0

    def test_clip_bounds(self) -> None:
        norm = ObsNormalizer(obs_dim=2, warmup_steps=0, clip=5.0)
        norm.update(np.array([[0.0, 0.0]], dtype=np.float32))
        extreme = np.array([[1e6, -1e6]], dtype=np.float32)
        out = norm.normalize(extreme)
        assert np.all(out <= 5.0)
        assert np.all(out >= -5.0)

    def test_bake_into_linear_layer(self) -> None:
        norm = ObsNormalizer(obs_dim=4, warmup_steps=0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            norm.update(rng.standard_normal((32, 4)).astype(np.float32) * 10 + 5)
        linear = torch.nn.Linear(4, 8)
        torch.manual_seed(0)
        torch.nn.init.normal_(linear.weight)
        torch.nn.init.normal_(linear.bias)
        w_orig = linear.weight.data.clone()
        b_orig = linear.bias.data.clone()
        norm.bake_into_linear(linear)
        raw = torch.from_numpy(rng.standard_normal((16, 4)).astype(np.float32) * 10 + 5)
        normalized = torch.from_numpy(norm.normalize(raw.numpy()))
        out_baked = linear(raw)
        out_manual = torch.nn.functional.linear(normalized, w_orig, b_orig)
        torch.testing.assert_close(out_baked, out_manual, atol=1e-4, rtol=1e-4)

    def test_checkpoint_roundtrip(self) -> None:
        norm = ObsNormalizer(obs_dim=3, warmup_steps=0)
        norm.update(np.ones((10, 3), dtype=np.float32) * 5)
        state = norm.state_dict()
        norm2 = ObsNormalizer(obs_dim=3, warmup_steps=0)
        norm2.load_state_dict(state)
        obs = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        np.testing.assert_allclose(norm.normalize(obs), norm2.normalize(obs))
