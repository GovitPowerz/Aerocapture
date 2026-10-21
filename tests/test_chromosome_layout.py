"""The chromosome layout as one value (#222): `[weight slab | params tail | ref_bank]`.

The layout is the only width check, the only splice and the only decode; the three
constructors agreeing on one run is `tests/test_training_config.py::TestRunParamSpecs`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.chromosome_layout import ChromosomeLayout
from aerocapture.training.param_spaces import _NN_LIVE_PARAMS, PARAM_SPACES, ParamSpec

REPO = Path(__file__).resolve().parents[1]
ARCH = [
    {"type": "dense", "input_size": 17, "output_size": 4, "activation": "swish"},
    {"type": "dense", "input_size": 4, "output_size": 2, "activation": "asinh"},
]
N_WEIGHTS = 17 * 4 + 4 + 4 * 2 + 2


@pytest.fixture()
def toml_path(tmp_path: Path) -> Path:
    from tests.fixtures.factories import leaf_toml_with_architecture

    p = tmp_path / "config.toml"
    p.write_text(leaf_toml_with_architecture(REPO / "configs/training/msr_aller_nn_atan2_train.toml", ARCH))
    return p


def _sidecar(run_dir: Path) -> list[ParamSpec]:
    bounds = [ParamSpec(f"w_{i}", -7.0 - i, 7.0 + i, 0.0) for i in range(N_WEIGHTS)]
    (run_dir / "warm_start_bounds.json").write_text(json.dumps([{"name": s.name, "p_min": s.p_min, "p_max": s.p_max} for s in bounds]))
    return bounds


class TestSidecar:
    def test_with_sidecar_replaces_the_slab_and_keeps_the_tail(self, tmp_path: Path, toml_path: Path) -> None:
        from aerocapture.training.training_config import _setup_param_specs, build_training_config_from_toml

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        bounds = _sidecar(run_dir)
        config, toml_data = build_training_config_from_toml(str(toml_path))
        init = ChromosomeLayout.from_specs(_setup_param_specs(config, toml_data, verbose=False), config.guidance_type)
        resumed = init.with_sidecar(run_dir)

        assert resumed.n_weights == N_WEIGHTS and resumed.width == N_WEIGHTS + 3
        assert list(resumed.weights) == bounds and init.weights != resumed.weights
        assert list(resumed.params) == list(_NN_LIVE_PARAMS) and resumed.ref_bank is None
        lo, hi = resumed.slab_bounds()
        assert lo[1] == -8.0 and hi[1] == 8.0

    def test_without_a_sidecar_resume_is_the_init_layout(self, tmp_path: Path, toml_path: Path) -> None:
        from aerocapture.training.training_config import _setup_param_specs, build_training_config_from_toml

        config, toml_data = build_training_config_from_toml(str(toml_path))
        init = ChromosomeLayout.from_specs(_setup_param_specs(config, toml_data, verbose=False), config.guidance_type)
        assert init.with_sidecar(tmp_path) == init


class TestFromSpecs:
    def test_classical_scheme_has_no_slab(self) -> None:
        specs = [*PARAM_SPACES["ftc"], ParamSpec("ref_bank", 55.0, 80.0, 68.0)]
        layout = ChromosomeLayout.from_specs(specs, "ftc")
        assert layout.weights == ()
        assert list(layout.params) == list(PARAM_SPACES["ftc"])  # FTC's own lateral./exit. genes are not scaffolding
        assert layout.ref_bank is not None and layout.ref_bank.name == "ref_bank"
        assert layout.specs == specs

    def test_specs_out_of_layout_order_are_refused(self) -> None:
        specs = [ParamSpec("nav.density_filter_gain", 0.3, 1.0, 0.8), ParamSpec("w0", -1.0, 1.0, 0.0)]
        with pytest.raises(ValueError, match="laid out"):
            ChromosomeLayout.from_specs(specs, "neural_network")


class TestWidthAndSplice:
    def test_one_width_message(self) -> None:
        layout = ChromosomeLayout.from_specs([ParamSpec("w0", -1.0, 1.0, 0.0), ParamSpec("w1", -1.0, 1.0, 0.0)], "neural_network")
        with pytest.raises(ValueError, match=r"chromosome width 3 != 2 param specs \(shape mismatch: wrong TOML\?\)"):
            layout.check_width(3, "wrong TOML?")

    @pytest.mark.parametrize("delta", [-1, 1])
    def test_bounds_not_spanning_the_slab_is_an_error(self, delta: int) -> None:
        layout = ChromosomeLayout.from_specs([ParamSpec(f"w{i}", -1.0, 1.0, 0.0) for i in range(4)] + list(_NN_LIVE_PARAMS), "neural_network")
        bounds = [ParamSpec(f"b{i}", -2.0, 2.0, 0.0) for i in range(4 + delta)]
        with pytest.raises(SystemExit, match="warm_start_bounds.json has"):
            layout.with_weight_bounds(bounds, source="warm_start_bounds.json")


class TestDecode:
    def test_splits_weights_params_and_ref_bank(self) -> None:
        specs = [ParamSpec("w0", -2.0, 2.0, 0.0), ParamSpec("w1", 0.0, 10.0, 0.0), ParamSpec("nav.density_filter_gain", 0.0, 1.0, 0.8)]
        layout = ChromosomeLayout.from_specs(specs, "neural_network")
        d = layout.decode(np.array([0.5, 0.1, 0.25]), "test")
        assert d.weights is not None and d.weights.tolist() == [0.0, 1.0]
        assert d.params == {"nav.density_filter_gain": 0.25}

    def test_classical_decode_keeps_ref_bank_in_params(self) -> None:
        specs = [ParamSpec("gain", 0.0, 2.0, 1.0), ParamSpec("ref_bank", 50.0, 70.0, 60.0)]
        d = ChromosomeLayout.from_specs(specs, "equilibrium_glide").decode(np.array([0.5, 0.5]), "test")
        assert d.weights is None
        assert d.params == {"gain": 1.0, "ref_bank": 60.0}

    def test_width_is_checked_before_decoding(self) -> None:
        layout = ChromosomeLayout.from_specs([ParamSpec("w0", -1.0, 1.0, 0.0)], "neural_network")
        with pytest.raises(ValueError, match="chromosome width 2 != 1"):
            layout.decode(np.zeros(2), "test")


class TestProblemReadsTheLayout:
    def test_slab_bounds_are_the_first_n_weights_specs(self) -> None:
        from aerocapture.training.config import NetworkConfig
        from aerocapture.training.problem import AerocaptureProblem

        specs = [ParamSpec(f"w{i}", -1.0 - i, 1.0 + i, 0.0) for i in range(3)] + list(_NN_LIVE_PARAMS)
        prob = AerocaptureProblem(
            param_specs=specs,
            toml_path="x.toml",
            seeds=[0],
            cost_kwargs={},
            scheme="neural_network",
            nn_config=NetworkConfig(architecture=[{"type": "dense", "input_size": 2, "output_size": 2, "activation": "tanh"}], scaffolding="live"),
        )
        assert prob.layout.n_weights == 3
        lo, hi = prob.layout.slab_bounds()
        assert lo.tolist() == [-1.0, -2.0, -3.0] and hi.tolist() == [1.0, 2.0, 3.0]
        assert prob.param_specs == specs

    def test_rebinding_param_specs_rebuilds_the_layout(self) -> None:
        from aerocapture.training.problem import AerocaptureProblem

        prob = AerocaptureProblem(param_specs=list(PARAM_SPACES["ftc"]), toml_path="x.toml", seeds=[0], cost_kwargs={}, scheme="ftc")
        assert prob._joint_ref_bank is False
        prob.param_specs = [*PARAM_SPACES["ftc"], ParamSpec("ref_bank", 55.0, 80.0, 68.0)]
        assert prob._joint_ref_bank is True and prob.layout.ref_bank is not None
