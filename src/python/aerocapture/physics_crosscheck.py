"""Our side of the external physics cross-check against AMAT (issue #112, docs/validation.md).

AMAT flew the cases once in a scratch venv (experiments/external_validation/amat_oracle.py) and its
outputs are frozen in amat_frozen.json next to that script; AMAT is never imported here. This module
re-flies the same cases through `run_batch`, compares each quantity with its stated tolerance, and
prints the tables quoted in docs/validation.md:

    uv run python -m aerocapture.physics_crosscheck

"Validation" elsewhere in this repo is the reserved seed pool; here it means physics validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[3]
FROZEN = REPO / "experiments/external_validation/amat_frozen.json"
EFPA_TOL_DEG = 1e-6
# Our adaptive DOPRI45 at a tolerance where its own error vanishes, with exit located by event.
CONVERGED_INTEGRATION: dict[str, object] = {"integration.mode": "adaptive", "integration.rtol": 1e-12, "integration.max_dt": 0.1}
OUTCOMES = {1: "crash", 2: "timeout", 3: "exit", 4: "crash"}  # finalize.rs ifinal_for; 4 = pending crash


@dataclass(frozen=True)
class Tolerance:
    value: float
    relative: bool
    reason: str

    def bound(self, reference: float) -> float:
        return self.value * abs(reference) if self.relative else self.value


@dataclass(frozen=True)
class Quantity:
    label: str
    record_key: str  # our final-record column (`final_record_indices`)
    amat_key: str  # the frozen AMAT case field
    tolerance: Tolerance
    exit_only: bool = False
    amat_native: bool = False  # compare with AMAT's own heating law rather than ours on AMAT's trajectory


# Each tolerance is the budget of a named difference between the two tools (docs/validation.md,
# "Tolerances"), not a fit to the observed gap.
SAMPLING = "our peak is sampled at 1 s ticks, AMAT's at 0.1 s"
SUTTON_GRAVES = "Sutton-Graves v^3 vs our v^3.05, nose radius matched at entry speed: (v0/v)^0.05"
QUANTITIES: dict[str, Quantity] = {
    "exit_energy": Quantity(
        "Exit inertial energy (MJ/kg)",
        "energy_mjkg",
        "exit_energy_mj_kg",
        Tolerance(1e-3, True, "AMAT's cos(gamma) + 0.01 heading regularization (diagnosed below); at full lift also our 1 s RK4 step (5e-6)"),
        exit_only=True,
    ),
    "apoapsis": Quantity(
        "Apoapsis altitude (km)",
        "apoapsis_alt_km",
        "apoapsis_alt_km",
        Tolerance(5e-3, True, "the exit-energy gap, amplified by d(apoapsis)/d(energy)"),
        exit_only=True,
    ),
    "periapsis": Quantity(
        "Periapsis altitude of the pass (km)",
        "bounce_alt_km",
        "min_alt_km",
        Tolerance(0.02, False, "ours is the 1 s tick where the flight-path angle turns positive"),
        exit_only=True,
    ),
    "peak_heat_flux": Quantity("Peak heat flux, our law (kW/m2)", "heat_flux_kw_m2", "peak_heat_flux_kw_m2", Tolerance(1e-3, True, SAMPLING)),
    "peak_g": Quantity("Peak aerodynamic load (g)", "g_load", "peak_g", Tolerance(1e-3, True, SAMPLING)),
    "heat_load": Quantity(
        "Heat load, our law (MJ/m2)",
        "heat_load_mjm2",
        "heat_load_mj_m2",
        Tolerance(1e-3, True, "our integral stops at the terminating tick, AMAT's trapezoid at its last sample"),
    ),
    "flight_time": Quantity(
        "Atmospheric flight time (s)",
        "sim_time_s",
        "flight_time_s",
        Tolerance(1.0, False, "fixed-step termination is quantized to 1 s and the record carries the tick's start time (#141)"),
    ),
    "sg_peak_heat_flux": Quantity(
        "Peak heat flux, AMAT's Sutton-Graves (kW/m2)", "heat_flux_kw_m2", "peak_heat_flux_kw_m2", Tolerance(0.02, True, SUTTON_GRAVES), amat_native=True
    ),
    "sg_heat_load": Quantity(
        "Heat load, AMAT's Sutton-Graves (MJ/m2)", "heat_load_mjm2", "heat_load_mj_m2", Tolerance(0.02, True, SUTTON_GRAVES), amat_native=True
    ),
}
# The banked-case residual attributed to AMAT's heading regularization: with it removed from AMAT
# and our integration converged, the two tools must agree to integrator precision.
DIAGNOSTIC_TOLERANCES: dict[str, Tolerance] = {
    "exit_energy": Tolerance(1e-6, True, "converged integrators, identical models"),
    "apoapsis": Tolerance(1e-5, True, "converged integrators, identical models"),
    "flight_time": Tolerance(0.01, False, "exit located by event (ours) and by interpolation between 0.1 s samples (AMAT)"),
}
EFPA_TOLERANCE = Tolerance(0.01, False, "AMAT's bisection resolves 1e-4 deg, ours 1e-6 deg; full lift up/down has no lateral force")
PUBLISHED_TOLERANCE = Tolerance(0.1, False, "atmosphere table: ours (MarsGram 3.8) vs AMAT's Mars-GRAM mean; AMAT itself moves up to 0.06 deg on the swap")
BOUNDS = ("overshoot_efpa_deg", "undershoot_efpa_deg", "width_deg")


@dataclass(frozen=True)
class Row:
    case: str
    label: str
    ours: float
    amat: float
    tolerance: Tolerance

    @property
    def diff(self) -> float:
        return self.ours - self.amat

    @property
    def ok(self) -> bool:
        return abs(self.diff) <= self.tolerance.bound(self.amat)


def load_frozen() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(FROZEN.read_text())
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_drift(inputs: dict[str, Any]) -> list[str]:
    """Every input AMAT flew, compared with the Rust-resolved config it was read from; [] = no drift."""
    import aerocapture_rs  # noqa: PLC0415

    cfg = aerocapture_rs.load_config(str(REPO / inputs["config"]))
    stated: dict[str, Any] = {
        **{f"planet.{k}": cfg["planet"][k] for k in inputs["planet"]},
        **{f"entry.{k}": cfg["entry"][k] for k in inputs["entry"]},
        **{f"vehicle.{k}": cfg["vehicle"][k] for k in inputs["vehicle"]},
        "ca": [p["ca"] for p in cfg["aerodynamics"]["points"]],
        "cn": [p["cn"] for p in cfg["aerodynamics"]["points"]],
        "incidence": cfg["incidence"]["angles"],
        "exit_altitude_km": cfg["flight"]["final_conditions"]["altitude"],
        "target_apoapsis_km": cfg["flight"]["target_orbit"]["apoapsis"],
        "wind": cfg["flight"]["wind"],
        "atmosphere": {"path": cfg["data"]["atmosphere"], "sha256": sha256(REPO / cfg["data"]["atmosphere"])},
    }
    aero = inputs["aerodynamics"]
    expected: dict[str, Any] = {
        **{f"{s}.{k}": v for s in ("planet", "entry", "vehicle") for k, v in inputs[s].items()},
        "ca": [aero["ca"]] * len(stated["ca"]),
        "cn": [aero["cn"]] * len(stated["cn"]),
        "incidence": [aero["aoa_deg"]] * len(stated["incidence"]),
        **{k: inputs[k] for k in ("exit_altitude_km", "target_apoapsis_km", "wind", "atmosphere")},
    }
    return [f"{k}: config {stated[k]!r} != frozen {expected[k]!r}" for k in expected if stated[k] != expected[k]]


def _fly(toml: str, overrides_list: list[dict[str, object]]) -> npt.NDArray[np.float64]:
    import aerocapture_rs  # noqa: PLC0415

    records: npt.NDArray[np.float64] = np.asarray(aerocapture_rs.run_batch(toml_path=str(REPO / toml), overrides_list=overrides_list).final_records)
    return records


def fly_matched(frozen: dict[str, Any], integration: dict[str, object] | None = None) -> dict[str, dict[str, Any]]:
    """Our outcome and QUANTITIES for each constant-bank case (default fixed-step integration unless overridden)."""
    import aerocapture_rs  # noqa: PLC0415

    idx = aerocapture_rs.final_record_indices()
    block = frozen["matched"]
    cases = block["cases"]
    records = _fly(block["inputs"]["config"], [{**(integration or {}), "guidance.reference_bank_angle": c["bank_deg"]} for c in cases.values()])
    out: dict[str, dict[str, Any]] = {}
    for name, fr in zip(cases, records, strict=True):
        outcome = OUTCOMES[int(fr[idx["ifinal"]])]
        values = {k: float(fr[idx[q.record_key]]) for k, q in QUANTITIES.items() if outcome == "exit" or not q.exit_only}
        out[name] = {"outcome": outcome, **values}
    return out


def matched_rows(ours: dict[str, dict[str, Any]], frozen: dict[str, Any]) -> list[Row]:
    rows = []
    for name, amat in frozen["matched"]["cases"].items():
        for k, q in QUANTITIES.items():
            if q.exit_only and amat["outcome"] != "exit":
                continue
            ref = amat["amat_native"][q.amat_key] if q.amat_native else amat[q.amat_key]
            rows.append(Row(name, q.label, ours[name][k], ref, q.tolerance))
    return rows


def diagnostic_rows(frozen: dict[str, Any]) -> list[Row]:
    ours = fly_matched(frozen, CONVERGED_INTEGRATION)["mid_bank"]
    amat = frozen["matched"]["diagnostics"]["mid_bank_exact_heading"]
    return [Row("mid_bank", QUANTITIES[k].label, ours[k], amat[QUANTITIES[k].amat_key], tol) for k, tol in DIAGNOSTIC_TOLERANCES.items()]


def _undershoots(toml: str, overrides: dict[str, object], target_km: float) -> bool:
    """The side of AMAT's hitsTargetApoapsis2: True when the flight falls short of the target apoapsis
    (crash, timeout, or a bounded orbit below it), False when it reaches it or escapes."""
    import aerocapture_rs  # noqa: PLC0415

    idx = aerocapture_rs.final_record_indices()
    fr = _fly(toml, [overrides])[0]
    if OUTCOMES[int(fr[idx["ifinal"]])] != "exit":
        return True
    if fr[idx["energy_mjkg"]] >= 0.0:
        return False
    return bool(fr[idx["apoapsis_alt_km"]] < target_km)


def efpa_bound(toml: str, overrides: dict[str, object], target_km: float, steep: float, shallow: float) -> float:
    """Bisection on the entry flight-path angle for the flight that just reaches `target_km` apoapsis."""
    assert _undershoots(toml, {**overrides, "entry.flight_path_angle": steep}, target_km), f"{steep} deg does not undershoot"
    assert not _undershoots(toml, {**overrides, "entry.flight_path_angle": shallow}, target_km), f"{shallow} deg does not overshoot"
    while shallow - steep > EFPA_TOL_DEG:
        mid = 0.5 * (steep + shallow)
        if _undershoots(toml, {**overrides, "entry.flight_path_angle": mid}, target_km):
            steep = mid
        else:
            shallow = mid
    return 0.5 * (steep + shallow)


def matched_corridor(frozen: dict[str, Any]) -> dict[str, float]:
    """Lift-modulation entry corridor at the matched entry state: overshoot = full lift-down, undershoot = full lift-up."""
    toml = frozen["matched"]["inputs"]["config"]
    target = frozen["matched"]["corridor"]["target_apoapsis_km"]
    os_ = efpa_bound(toml, {"guidance.reference_bank_angle": 180.0}, target, -14.0, -8.0)
    us = efpa_bound(toml, {"guidance.reference_bank_angle": 0.0}, target, -14.0, -8.0)
    return {"overshoot_efpa_deg": os_, "undershoot_efpa_deg": us, "width_deg": os_ - us}


def published_corridor(frozen: dict[str, Any]) -> dict[str, float]:
    """Drag-modulation entry corridor of the published case: overshoot = beta1 throughout, undershoot = beta2 = ratio * beta1."""
    pub = frozen["published"]
    toml = pub["inputs"]["config"]
    target = pub["inputs"]["target_apoapsis_km"]
    beta2_area = pub["inputs"]["vehicle"]["reference_area"] / pub["beta_ratio"]
    os_ = efpa_bound(toml, {}, target, -20.0, -5.0)
    us = efpa_bound(toml, {"vehicle.reference_area": beta2_area}, target, -20.0, -5.0)
    return {"overshoot_efpa_deg": os_, "undershoot_efpa_deg": us, "width_deg": os_ - us}


def corridor_rows(case: str, ours: dict[str, float], reference: dict[str, float], tolerance: Tolerance) -> list[Row]:
    return [Row(case, b, ours[b], reference[b], tolerance) for b in BOUNDS]


def _fmt(x: float) -> str:
    return f"{x:.6g}" if abs(x) >= 1e-3 or x == 0.0 else f"{x:.2e}"


def _row_line(r: Row) -> str:
    tol = f"{r.tolerance.value:g}{' rel' if r.tolerance.relative else ''}"
    rel = f" ({r.diff / r.amat:+.1e})" if r.tolerance.relative else ""
    return f"| {r.case} | {r.label} | {_fmt(r.ours)} | {_fmt(r.amat)} | {_fmt(r.diff)}{rel} | {tol} | {'yes' if r.ok else '**NO**'} | {r.tolerance.reason} |"


def main() -> None:
    frozen = load_frozen()
    for block in ("matched", "published"):
        drift = input_drift(frozen[block]["inputs"])
        if drift:
            raise SystemExit(f"{block}: config drifted from the inputs AMAT flew; rerun the oracle:\n" + "\n".join(drift))
    amat = frozen["amat"]
    print(f"AMAT {amat['version']} (source {amat['source_commit'][:12]}), frozen in {FROZEN.relative_to(REPO)}\n")

    ours = fly_matched(frozen)
    header = "| Case | Quantity | Ours | AMAT | Ours - AMAT (rel) | Tolerance | Within | Explanation |\n|---|---|---|---|---|---|---|---|"
    print(header)
    for name, amat_case in frozen["matched"]["cases"].items():
        same = ours[name]["outcome"] == amat_case["outcome"]
        print(f"| {name} | Outcome | {ours[name]['outcome']} | {amat_case['outcome']} | | | {'yes' if same else '**NO**'} | |")
    for r in matched_rows(ours, frozen):
        print(_row_line(r))

    print("\nmid_bank, AMAT without its heading regularization vs ours with converged integration:\n")
    print(header.replace("| Ours | AMAT |", "| Ours, DOPRI45 rtol 1e-12 | AMAT, exact heading |"))
    for r in diagnostic_rows(frozen):
        print(_row_line(r))

    print("\n" + header.replace("| Case | Quantity |", "| Corridor | Bound (deg) |"))
    for r in corridor_rows("lift modulation", matched_corridor(frozen), frozen["matched"]["corridor"], EFPA_TOLERANCE):
        print(_row_line(r))

    pub = frozen["published"]
    ours_pub = published_corridor(frozen)
    print(f"\n{pub['citation']}")
    print("\n| Bound (deg) | Published | AMAT, notebook settings | AMAT, our Mars | Ours | Ours - published | Tolerance | Within |")
    print("|---|---|---|---|---|---|---|---|")
    for r in corridor_rows("published", ours_pub, pub["published"], PUBLISHED_TOLERANCE):
        a, b = pub["amat_gram_mean_table"][r.label], pub["amat_our_mars"][r.label]
        print(f"| {r.label} | {r.amat:g} | {a:.4f} | {b:.4f} | {r.ours:.4f} | {r.diff:+.4f} | {r.tolerance.value:g} | {'yes' if r.ok else '**NO**'} |")


if __name__ == "__main__":
    main()
