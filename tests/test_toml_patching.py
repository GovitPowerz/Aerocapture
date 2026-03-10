"""Tests for write_guidance_toml TOML patching roundtrip.

For each non-NN scheme, decodes a mid-range chromosome, patches the
training TOML, parses it back with tomllib, and asserts the guidance
section is present and contains the expected keys.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from aerocapture.training.evaluate import decode_params_from_chromosome, write_guidance_toml
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

    for name in params:
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

    for name, expected in params.items():
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
