"""Gate for the external physics cross-check (issue #112, docs/validation.md).

The slow tests re-fly the AMAT-matched cases and the published Mars entry corridor through run_batch
and hold them to AMAT's frozen outputs within the tolerances stated in `physics_crosscheck`. AMAT is
never imported: amat_oracle.py ran it once and froze its numbers.
"""

from typing import Any

import pytest
from aerocapture import physics_crosscheck as crosscheck


@pytest.fixture(scope="module")
def frozen() -> dict[str, Any]:
    return crosscheck.load_frozen()


def test_frozen_outputs_come_from_the_committed_oracle(frozen: dict[str, Any]) -> None:
    assert crosscheck.sha256(crosscheck.REPO / frozen["generator"]) == frozen["generator_sha256"], "amat_oracle.py changed: rerun it"
    doc = (crosscheck.REPO / "docs/validation.md").read_text()
    assert f"AMAT {frozen['amat']['version']}" in doc
    assert str(crosscheck.FROZEN.relative_to(crosscheck.REPO)) in doc


@pytest.mark.slow
def test_validation_doc_quotes_the_current_tables(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("aerocapture_rs")
    crosscheck.main()
    doc = (crosscheck.REPO / "docs/validation.md").read_text()
    stale = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| ") and line not in doc]
    assert not stale, "docs/validation.md is stale; paste `python -m aerocapture.physics_crosscheck`:\n" + "\n".join(stale)


@pytest.mark.slow
@pytest.mark.parametrize("block", ["matched", "published"])
def test_configs_still_state_the_inputs_amat_flew(frozen: dict[str, Any], block: str) -> None:
    pytest.importorskip("aerocapture_rs")
    assert crosscheck.input_drift(frozen[block]["inputs"]) == []


def _report(rows: list[crosscheck.Row]) -> str:
    return "\n".join(f"{r.case} {r.label}: ours {r.ours} vs {r.amat} (tolerance {r.tolerance.value}: {r.tolerance.reason})" for r in rows)


@pytest.mark.slow
def test_matched_cases_agree_with_amat(frozen: dict[str, Any]) -> None:
    pytest.importorskip("aerocapture_rs")
    ours = crosscheck.fly_matched(frozen)
    assert {k: v["outcome"] for k, v in ours.items()} == {k: v["outcome"] for k, v in frozen["matched"]["cases"].items()}
    rows = crosscheck.matched_rows(ours, frozen)
    n_exit_only = sum(q.exit_only for q in crosscheck.QUANTITIES.values())
    assert len(rows) == 3 * len(crosscheck.QUANTITIES) - n_exit_only  # two exits, one crash
    failed = [r for r in rows if not r.ok]
    assert not failed, _report(failed)


@pytest.mark.slow
def test_banked_residual_is_amats_heading_regularization(frozen: dict[str, Any]) -> None:
    pytest.importorskip("aerocapture_rs")
    failed = [r for r in crosscheck.diagnostic_rows(frozen) if not r.ok]
    assert not failed, _report(failed)


@pytest.mark.slow
def test_lift_modulation_corridor_agrees_with_amat(frozen: dict[str, Any]) -> None:
    pytest.importorskip("aerocapture_rs")
    rows = crosscheck.corridor_rows("lift modulation", crosscheck.matched_corridor(frozen), frozen["matched"]["corridor"], crosscheck.EFPA_TOLERANCE)
    failed = [r for r in rows if not r.ok]
    assert not failed, _report(failed)


@pytest.mark.slow
def test_published_mars_corridor_is_reproduced(frozen: dict[str, Any]) -> None:
    pytest.importorskip("aerocapture_rs")
    pub = frozen["published"]
    rows = crosscheck.corridor_rows("published", crosscheck.published_corridor(frozen), pub["published"], crosscheck.PUBLISHED_TOLERANCE)
    failed = [r for r in rows if not r.ok]
    assert not failed, _report(failed)
