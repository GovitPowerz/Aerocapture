"""The champion record: one training run's (or one island's) current best and its
promotion state, with every transition written once (#221).

A `Champion` is *trusted* when its validation RMS is finite (validated feasible under
the current ceiling) and *untrusted* otherwise: re-validation found it infeasible (e.g.
after a ceiling or limit change) or its RMS non-finite, it is the fresh-start gen-0
argmin not yet validated, or there is no validation pool. The decision rules stay where they are
(`evaluate.run_validation_gate`, `final_select.select_final_individual`, ADR-0002 /
ADR-0005); this module applies their results. Feasibility and violation rates are
in-memory only: every resume with a validation pool re-validates (`revalidate`).

Wire format is exactly the pre-record keys, nothing new on disk: single-algo
`best_cost` / `best_val_cost` (json) + `best_individual` (npz); islands
`best_overall_individual` / `best_overall_cost` / `best_val_cost` /
`last_validated_individual` (pickled island state). A finite `best_val_cost` loads
as trusted, inf as untrusted, a missing individual as an empty record. A leaf like
`checkpoint.py`: no trainer import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from aerocapture.training.evaluate import GateResult, GateStatus, _SupportsPerSeedEval, constraint_violation_rates, format_violation_rates, is_feasible


@dataclass
class KnownCandidate:
    """A pre-scored final-selection candidate (a trusted champion); never re-simulated.

    `provenance` is structured, not just display text: the islands call sites
    use the grammar "<island>:champion" / "<island>:last_gen[i]" and parse the
    island name back out via split(":", 1). Island names must not contain ":".
    """

    x: npt.NDArray[np.float64]
    provenance: str
    val_rms: float


@dataclass
class Champion:
    """The individual a run currently holds as its best, and its promotion state."""

    individual: npt.NDArray[np.float64] | None = None
    train_cost: float = float("inf")  # training cost at promotion (resume-incomparability rule: restored verbatim)
    val_rms: float = float("inf")  # validation RMS; inf when untrusted
    last_validated: npt.NDArray[np.float64] | None = None  # identity trigger of the validation gate
    violation_rates: dict[str, float] | None = None  # in-memory only

    @property
    def trusted(self) -> bool:
        return self.individual is not None and bool(np.isfinite(self.val_rms))

    # -- transitions -------------------------------------------------------

    def consider(self, gate: GateResult) -> bool:
        """Apply a validation-gate result: a validated candidate becomes last-validated;
        it displaces the champion only when feasible with strictly lower RMS
        (`gate.promoted`, ADR-0002 / ADR-0005). Returns True on promotion."""
        if gate.status is not GateStatus.VALIDATED:
            return False
        assert gate.individual is not None and gate.val_rms is not None
        self.last_validated = gate.individual
        if not gate.promoted:
            return False
        self.individual = gate.individual.copy()
        self.train_cost = gate.argmin_cost
        self.val_rms = gate.val_rms
        self.violation_rates = gate.violation_rates
        return True

    def reanchor(self, val_rms: float, violation_rates: dict[str, float] | None, ceiling: float) -> bool:
        """Re-anchor under the current cost kwargs and ceiling: feasible anchors a finite
        `val_rms`, infeasible (or a non-finite RMS, which no `<` could ever beat) sets it to
        inf; the individual becomes last-validated either way. Returns the feasibility verdict."""
        assert self.individual is not None
        feasible = is_feasible(violation_rates, ceiling)
        self.val_rms = val_rms if feasible and np.isfinite(val_rms) else float("inf")
        self.violation_rates = violation_rates
        self.last_validated = self.individual.copy()
        return feasible

    def adopt_unvalidated(self, individual: npt.NDArray[np.float64], train_cost: float) -> None:
        """The no-validation fallback (and the fresh-start gen-0 argmin): an untrusted champion."""
        self.individual = individual
        self.train_cost = train_cost
        self.val_rms = float("inf")
        self.violation_rates = None

    def adopt_selection(self, individual: npt.NDArray[np.float64], val_rms: float, train_cost: float | None) -> None:
        """The final-selection write-back; `None` keeps the recorded training cost."""
        self.individual = individual.copy()
        self.val_rms = float(val_rms)
        if train_cost is not None:
            self.train_cost = train_cost

    # -- presentation to final selection -----------------------------------

    def known(self, label: str = "") -> KnownCandidate | None:
        """A trusted champion as a pre-scored candidate; None otherwise."""
        if not self.trusted:
            return None
        assert self.individual is not None
        return KnownCandidate(x=np.asarray(self.individual, dtype=np.float64), provenance=f"{label}champion", val_rms=float(self.val_rms))

    def untrusted_row(self, label: str = "") -> tuple[npt.NDArray[np.float64], str] | None:
        """An untrusted champion as an ordinary candidate row with the `[infeasible]` provenance; None otherwise."""
        if self.individual is None or self.trusted:
            return None
        return np.asarray(self.individual, dtype=np.float64), f"{label}champion[infeasible]"

    # -- wire format (today's keys; nothing new on disk) --------------------

    @classmethod
    def from_single(cls, meta: Mapping[str, Any], individual: npt.NDArray[np.float64] | None) -> Champion:
        """Single-algo checkpoint: `best_cost` / `best_val_cost` from the json meta (or the
        `load_checkpoint` dict), `best_individual` from the npz. The single format does not
        persist the last-validated individual; the prologue re-validates on resume. Without
        an individual the record is empty but keeps `best_cost`, so a write-back preserves it."""
        if individual is None:
            return cls(train_cost=float(meta.get("best_cost", float("inf"))))
        return cls(
            individual=individual,
            train_cost=float(meta.get("best_cost", float("inf"))),
            val_rms=float(meta.get("best_val_cost", float("inf"))),
            last_validated=individual.copy(),
        )

    def single_meta(self) -> dict[str, float]:
        return {"best_cost": self.train_cost, "best_val_cost": self.val_rms}

    @classmethod
    def from_island_state(cls, state: Mapping[str, Any]) -> Champion:
        individual = state.get("best_overall_individual")
        return cls(
            individual=individual,
            train_cost=float(state.get("best_overall_cost", float("inf"))),
            val_rms=float(state.get("best_val_cost", float("inf"))),
            last_validated=state.get("last_validated_individual"),
        )

    def island_state(self) -> dict[str, Any]:
        return {
            "last_validated_individual": self.last_validated,
            "best_overall_individual": self.individual,
            "best_overall_cost": self.train_cost,
            "best_val_cost": self.val_rms,
        }


def revalidate(
    champion: Champion,
    problem: _SupportsPerSeedEval,
    val_seeds: list[int],
    cost_kwargs: dict[str, Any] | None,
    ceiling: float,
    verbose: bool,
    label: str = "",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
    """Fly the champion on the validation pool and `reanchor` it (the prologue /
    `revalidate_each` / retro CLI block). Returns the per-seed costs and records (the
    caller builds the run-log payload from them) and the feasibility verdict. An
    infeasible champion is announced when `verbose`; it stays last-validated but no
    longer anchors the promotion bar."""
    assert champion.individual is not None
    val_costs, val_records = problem.evaluate_individual_records_per_seed(champion.individual, val_seeds)
    rms = float(np.sqrt(np.mean(val_costs**2)))
    rates = constraint_violation_rates(val_records, cost_kwargs)
    feasible = champion.reanchor(rms, rates, ceiling)
    if not feasible and verbose:
        print(
            f"  {label}champion re-validated: INFEASIBLE on the validation pool ({format_violation_rates(rates)} > ceiling {ceiling:.3%}); "
            f"untrusted, competes as a candidate (rms was {rms:.4g})"
        )
    return val_costs, val_records, feasible
