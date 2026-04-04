"""Tests for write_guidance_toml TOML patching roundtrip.

For each non-NN scheme, decodes a mid-range chromosome, patches the
training TOML, parses it back with tomllib, and asserts the guidance
section is present and contains the expected keys.
"""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path

import pytest
from aerocapture.training.evaluate import decode_params_from_chromosome, patch_toml_mc_seed, write_guidance_toml
from aerocapture.training.param_spaces import PARAM_SPACES

from tests.fixtures.factories import make_chromosome, make_training_config

ROOT = Path(__file__).resolve().parent.parent

TRAINING_CONFIGS: dict[str, Path] = {
    "equilibrium_glide": ROOT / "configs" / "training" / "msr_aller_eqglide_train.toml",
    "energy_controller": ROOT / "configs" / "training" / "msr_aller_energy_controller_train.toml",
    "pred_guid": ROOT / "configs" / "training" / "msr_aller_pred_guid_train.toml",
    "fnpag": ROOT / "configs" / "training" / "msr_aller_fnpag_train.toml",
    "ftc": ROOT / "configs" / "training" / "msr_aller_ftc_train.toml",
}


@pytest.mark.parametrize("scheme", list(TRAINING_CONFIGS.keys()))
def test_toml_roundtrip_guidance_section_exists(scheme: str, tmp_path: Path) -> None:
    """Decode mid-range chromosome → write TOML → parse back → guidance section exists."""
    config = make_training_config(scheme)
    specs = PARAM_SPACES[scheme]
    chrom_len = len(specs) * config.ga.n_bit
    chrom = make_chromosome(chrom_len, strategy="mid")

    params = decode_params_from_chromosome(chrom, config)

    base_toml = TRAINING_CONFIGS[scheme]
    out_path = tmp_path / f"{scheme}_patched.toml"
    written = write_guidance_toml(base_toml, scheme, params, output_path=out_path)

    with open(written, "rb") as f:
        parsed = tomllib.load(f)

    assert "guidance" in parsed, f"scheme={scheme}: 'guidance' section missing from patched TOML"


@pytest.mark.parametrize("scheme", list(TRAINING_CONFIGS.keys()))
def test_toml_roundtrip_params_present(scheme: str, tmp_path: Path) -> None:
    """All decoded parameter keys appear inside the guidance sub-section."""
    config = make_training_config(scheme)
    specs = PARAM_SPACES[scheme]
    chrom_len = len(specs) * config.ga.n_bit
    chrom = make_chromosome(chrom_len, strategy="mid")

    params = decode_params_from_chromosome(chrom, config)

    base_toml = TRAINING_CONFIGS[scheme]
    out_path = tmp_path / f"{scheme}_params.toml"
    written = write_guidance_toml(base_toml, scheme, params, output_path=out_path)

    with open(written, "rb") as f:
        parsed = tomllib.load(f)

    # Navigate into guidance sub-section
    from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

    section_name = GUIDANCE_TOML_SECTIONS[scheme]
    guidance = parsed.get("guidance", {})
    sub = guidance.get(section_name, {})
    lateral_sub = guidance.get("lateral", {})

    exit_sub = guidance.get("ftc", {})
    thermal_sub = guidance.get("thermal_limiter", {})
    nav_sub = parsed.get("navigation", {})

    for name in params:
        if name.startswith("lateral."):
            bare = name.removeprefix("lateral.")
            assert bare in lateral_sub, f"scheme={scheme}: lateral param '{bare}' missing from guidance.lateral"
        elif name.startswith("exit."):
            bare = name.removeprefix("exit.")
            assert bare in exit_sub, f"scheme={scheme}: exit param '{bare}' missing from guidance.ftc"
        elif name.startswith("nav."):
            bare = name.removeprefix("nav.")
            assert bare in nav_sub, f"scheme={scheme}: nav param '{bare}' missing from navigation"
        elif name.startswith("thermal."):
            bare = name.removeprefix("thermal.")
            assert bare in thermal_sub, f"scheme={scheme}: thermal param '{bare}' missing from guidance.thermal_limiter"
        elif name.startswith("shaping."):
            shaping_sub = guidance.get("command_shaping", {})
            bare = name.removeprefix("shaping.")
            assert bare in shaping_sub, f"scheme={scheme}: shaping param '{bare}' missing from guidance.command_shaping"
        else:
            assert name in sub, f"scheme={scheme}: param '{name}' missing from guidance.{section_name}"


@pytest.mark.parametrize("scheme", list(TRAINING_CONFIGS.keys()))
def test_toml_roundtrip_values_close(scheme: str, tmp_path: Path) -> None:
    """Values written to TOML round-trip to within float repr precision."""
    config = make_training_config(scheme)
    specs = PARAM_SPACES[scheme]
    chrom_len = len(specs) * config.ga.n_bit
    chrom = make_chromosome(chrom_len, strategy="mid")

    params = decode_params_from_chromosome(chrom, config)

    base_toml = TRAINING_CONFIGS[scheme]
    out_path = tmp_path / f"{scheme}_values.toml"
    written = write_guidance_toml(base_toml, scheme, params, output_path=out_path)

    with open(written, "rb") as f:
        parsed = tomllib.load(f)

    from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

    section_name = GUIDANCE_TOML_SECTIONS[scheme]
    sub = parsed["guidance"][section_name]
    lateral_sub = parsed["guidance"].get("lateral", {})
    exit_sub = parsed["guidance"].get("ftc", {})
    thermal_sub = parsed["guidance"].get("thermal_limiter", {})
    nav_sub = parsed.get("navigation", {})

    for name, expected in params.items():
        if name.startswith("lateral."):
            bare = name.removeprefix("lateral.")
            actual = lateral_sub[bare]
            # max_reversals is rounded to int before writing
            if bare == "max_reversals":
                expected = float(int(round(expected)))
        elif name.startswith("exit."):
            bare = name.removeprefix("exit.")
            actual = exit_sub[bare]
        elif name.startswith("nav."):
            bare = name.removeprefix("nav.")
            actual = nav_sub[bare]
        elif name.startswith("thermal."):
            bare = name.removeprefix("thermal.")
            actual = thermal_sub[bare]
        elif name.startswith("shaping."):
            shaping_sub = parsed["guidance"].get("command_shaping", {})
            bare = name.removeprefix("shaping.")
            actual = shaping_sub[bare]
        else:
            actual = sub[name]
        assert abs(actual - expected) <= 1e-9 * max(abs(expected), 1.0), f"scheme={scheme} param={name}: written={expected}, read back={actual}"


@pytest.mark.parametrize("scheme", list(TRAINING_CONFIGS.keys()))
def test_toml_roundtrip_temp_file(scheme: str) -> None:
    """write_guidance_toml without output_path creates a valid temp file."""
    config = make_training_config(scheme)
    specs = PARAM_SPACES[scheme]
    chrom_len = len(specs) * config.ga.n_bit
    chrom = make_chromosome(chrom_len, strategy="zeros")

    params = decode_params_from_chromosome(chrom, config)
    base_toml = TRAINING_CONFIGS[scheme]

    written = write_guidance_toml(base_toml, scheme, params)
    try:
        with open(written, "rb") as f:
            parsed = tomllib.load(f)
        assert "guidance" in parsed
    finally:
        written.unlink(missing_ok=True)


class TestPatchTomlMcSeed:
    def test_overrides_seed(self, tmp_path: Path) -> None:
        base = tmp_path / "base.toml"
        base.write_text('[monte_carlo]\nseed = 42\n\n[mission]\ntype = "aerocapture"\n')
        patched = patch_toml_mc_seed(base, 99)
        try:
            with open(patched, "rb") as f:
                data = tomllib.load(f)
            assert data["monte_carlo"]["seed"] == 99
            assert data["mission"]["type"] == "aerocapture"
        finally:
            patched.unlink(missing_ok=True)

    def test_adds_seed_when_missing(self, tmp_path: Path) -> None:
        base = tmp_path / "base.toml"
        base.write_text('[mission]\ntype = "aerocapture"\n')
        patched = patch_toml_mc_seed(base, 55)
        try:
            with open(patched, "rb") as f:
                data = tomllib.load(f)
            assert data["monte_carlo"]["seed"] == 55
        finally:
            patched.unlink(missing_ok=True)


class TestEvaluateChromosomeMcSeed:
    def test_mc_seed_param_exists(self) -> None:
        from aerocapture.training.evaluate import evaluate_chromosome

        sig = inspect.signature(evaluate_chromosome)
        assert "mc_seed" in sig.parameters


class TestWriteGuidanceTomlMcSeed:
    def test_mc_seed_composed_into_patched_toml(self, tmp_path: Path) -> None:
        base = tmp_path / "base.toml"
        base.write_text(
            '[mission]\ntype = "aerocapture"\n\n[monte_carlo]\nseed = 42\n\n'
            '[guidance]\ntype = "equilibrium_glide"\n\n[guidance.equilibrium_glide]\nk_hdot = 1.0\n'
        )
        patched = write_guidance_toml(base, "equilibrium_glide", {"k_hdot": 2.0}, mc_seed=99)
        try:
            with open(patched, "rb") as f:
                data = tomllib.load(f)
            assert data["monte_carlo"]["seed"] == 99
            assert data["guidance"]["equilibrium_glide"]["k_hdot"] == 2.0
        finally:
            patched.unlink(missing_ok=True)

    def test_no_mc_seed_preserves_original(self, tmp_path: Path) -> None:
        base = tmp_path / "base.toml"
        base.write_text(
            '[mission]\ntype = "aerocapture"\n\n[monte_carlo]\nseed = 42\n\n'
            '[guidance]\ntype = "equilibrium_glide"\n\n[guidance.equilibrium_glide]\nk_hdot = 1.0\n'
        )
        patched = write_guidance_toml(base, "equilibrium_glide", {"k_hdot": 2.0})
        try:
            with open(patched, "rb") as f:
                data = tomllib.load(f)
            assert data["monte_carlo"]["seed"] == 42
        finally:
            patched.unlink(missing_ok=True)


class TestThermalLimiterParams:
    """Thermal limiter params present in unsigned-magnitude schemes and route correctly."""

    UNSIGNED_SCHEMES = ["equilibrium_glide", "energy_controller", "pred_guid", "fnpag", "ftc"]

    @pytest.mark.parametrize("scheme", UNSIGNED_SCHEMES)
    def test_thermal_params_in_param_space(self, scheme: str) -> None:
        """All unsigned-magnitude schemes include thermal limiter params."""
        specs = PARAM_SPACES[scheme]
        thermal_names = {s.name for s in specs if s.name.startswith("thermal.")}
        expected = {
            "thermal.heat_flux_activation",
            "thermal.heat_load_activation",
            "thermal.heat_flux_ramp_exponent",
            "thermal.heat_load_ramp_exponent",
        }
        assert thermal_names == expected, f"scheme={scheme}: thermal params mismatch: {thermal_names}"

    def test_piecewise_constant_has_no_thermal_params(self) -> None:
        """Piecewise constant should NOT have thermal limiter params."""
        specs = PARAM_SPACES["piecewise_constant"]
        thermal_names = [s.name for s in specs if s.name.startswith("thermal.")]
        assert thermal_names == [], f"piecewise_constant should not have thermal params: {thermal_names}"

    @pytest.mark.parametrize("scheme", UNSIGNED_SCHEMES)
    def test_thermal_params_route_to_toml_section(self, scheme: str, tmp_path: Path) -> None:
        """thermal.* params end up in [guidance.thermal_limiter] in the patched TOML."""
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chrom_len = len(specs) * config.ga.n_bit
        chrom = make_chromosome(chrom_len, strategy="mid")

        params = decode_params_from_chromosome(chrom, config)
        base_toml = TRAINING_CONFIGS[scheme]
        out_path = tmp_path / f"{scheme}_thermal.toml"
        written = write_guidance_toml(base_toml, scheme, params, output_path=out_path)

        with open(written, "rb") as f:
            parsed = tomllib.load(f)

        thermal_section = parsed.get("guidance", {}).get("thermal_limiter", {})
        assert "heat_flux_activation" in thermal_section, f"scheme={scheme}: heat_flux_activation missing"
        assert "heat_load_activation" in thermal_section, f"scheme={scheme}: heat_load_activation missing"
        assert "heat_flux_ramp_exponent" in thermal_section, f"scheme={scheme}: heat_flux_ramp_exponent missing"
        assert "heat_load_ramp_exponent" in thermal_section, f"scheme={scheme}: heat_load_ramp_exponent missing"


class TestExitParamsSafety:
    """Non-FTC schemes get density filter params routed to [navigation]."""

    NON_FTC_SCHEMES = ["equilibrium_glide", "energy_controller", "pred_guid", "fnpag"]

    @pytest.mark.parametrize("scheme", NON_FTC_SCHEMES)
    def test_density_filter_params_written_for_non_ftc(self, scheme: str, tmp_path: Path) -> None:
        """Non-FTC schemes must have density_filter_gain and density_gain_max_delta in [navigation]."""
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chrom_len = len(specs) * config.ga.n_bit
        chrom = make_chromosome(chrom_len, strategy="mid")

        params = decode_params_from_chromosome(chrom, config)
        base_toml = TRAINING_CONFIGS[scheme]
        out_path = tmp_path / f"{scheme}_nav_params.toml"
        written = write_guidance_toml(base_toml, scheme, params, output_path=out_path)

        with open(written, "rb") as f:
            parsed = tomllib.load(f)

        nav_section = parsed.get("navigation", {})
        assert "density_filter_gain" in nav_section, f"scheme={scheme}: density_filter_gain missing from navigation"
        assert 0.3 <= nav_section["density_filter_gain"] <= 1.0, f"scheme={scheme}: density_filter_gain out of bounds: {nav_section['density_filter_gain']}"
        assert "density_gain_max_delta" in nav_section, f"scheme={scheme}: density_gain_max_delta missing from navigation"
        assert 0.01 <= nav_section["density_gain_max_delta"] <= 0.5, (
            f"scheme={scheme}: density_gain_max_delta out of bounds: {nav_section['density_gain_max_delta']}"
        )

    @pytest.mark.parametrize("scheme", NON_FTC_SCHEMES)
    def test_exit_altitude_threshold_in_param_space(self, scheme: str) -> None:
        """All unsigned-magnitude schemes include exit_altitude_threshold as a GA param."""
        specs = PARAM_SPACES[scheme]
        exit_names = {s.name for s in specs if s.name.startswith("exit.")}
        assert "exit.exit_altitude_threshold" in exit_names, f"scheme={scheme}: exit_altitude_threshold missing from param space"
