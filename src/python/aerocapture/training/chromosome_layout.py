"""The chromosome layout as one value per run (#222).

A chromosome is `[NN weight slab | scaffolding tail | ref_bank]` for `neural_network` and
`[scheme table | ref_bank]` for the classical schemes. The slab's bounds may be replaced by
the warm-start sidecar (`<run_dir>/warm_start_bounds.json`), and every holder of the spec
list must see the replacement: before this module three code paths spliced one aliased
list in three styles, five sites recomputed the slab boundary by subtraction and six sites
checked a chromosome's width with six messages, so a splice that missed one holder
silently flew a different network (#196, #203).

`ChromosomeLayout` is the only splice (`with_weight_bounds`), the only boundary
(`n_weights`), the only slab bounds (`slab_bounds`, what `run_grid` decodes under) and the
only decode of a chromosome into (weights, params, ref_bank). The width check is
`param_spaces.check_chromosome_width`, shared with `encoding.decode_normalized`.

Three constructors of one run's layout: fresh start (`from_specs` on the trainer's spec
list, then `with_weight_bounds` with the warm-start chromosome's specs), resume
(`with_sidecar` on the run dir), offline rebuild (`training_config.run_param_specs`).
`warm_start` is the producer of the sidecar, not a consumer. A leaf: `param_spaces` and
`encoding` only (`warm_start` lazily, for the sidecar reader).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.encoding import _decode_nn_weights, decode_normalized
from aerocapture.training.param_spaces import SCAFFOLDING_PREFIXES, ParamSpec, check_chromosome_width

REF_BANK = "ref_bank"


@dataclass(frozen=True)
class DecodedChromosome:
    """A chromosome decoded through its layout: the physical NN weights (None for a
    classical scheme) and the params dict (scaffolding tail or scheme table, plus
    `ref_bank` when the run optimizes the reference jointly)."""

    weights: npt.NDArray[np.float64] | None
    params: dict[str, float]


@dataclass(frozen=True)
class ChromosomeLayout:
    weights: tuple[ParamSpec, ...]
    params: tuple[ParamSpec, ...]
    ref_bank: ParamSpec | None

    @classmethod
    def from_specs(cls, specs: Sequence[ParamSpec], scheme: str) -> ChromosomeLayout:
        """Partition a spec list by gene name: for `neural_network` the scaffolding genes are
        the `SCAFFOLDING_PREFIXES` names (the rule `_build_grid_overrides` routes by) and
        every other gene is a weight; a classical scheme has no slab (FTC's own `lateral.` /
        `exit.` genes are its table, not scaffolding). Refuses a list not laid out as
        `[weights | params | ref_bank]`."""
        ref = next((s for s in specs if s.name == REF_BANK), None)
        if scheme == "neural_network":
            weights = tuple(s for s in specs if s.name != REF_BANK and not s.name.startswith(SCAFFOLDING_PREFIXES))
            params = tuple(s for s in specs if s.name.startswith(SCAFFOLDING_PREFIXES))
        else:
            weights = ()
            params = tuple(s for s in specs if s.name != REF_BANK)
        layout = cls(weights, params, ref)
        if layout.specs != list(specs):
            raise ValueError(f"chromosome specs are not laid out as [weights | params | ref_bank]: {[s.name for s in specs]}")
        return layout

    @property
    def specs(self) -> list[ParamSpec]:
        """The full spec list, the chromosome vocabulary pymoo and the decoders speak."""
        return [*self.weights, *self.tail]

    @property
    def tail(self) -> list[ParamSpec]:
        """Everything after the slab: the params and, when present, `ref_bank`."""
        return [*self.params, *([self.ref_bank] if self.ref_bank is not None else [])]

    @property
    def n_weights(self) -> int:
        return len(self.weights)

    @property
    def width(self) -> int:
        return len(self.weights) + len(self.tail)

    def check_width(self, n_actual: int, context: str) -> None:
        check_chromosome_width(n_actual, self.width, context)

    def with_weight_bounds(self, bounds: Sequence[ParamSpec], source: str) -> ChromosomeLayout:
        """The slab under the bounds the population was encoded with (the warm-start
        chromosome's specs, or the sidecar); the tail is untouched. Exits when `bounds`
        does not span the slab exactly: one short would land an init bound on the last
        weight, one long a weight bound on a scaffolding gene (wrong TOML for this run)."""
        if len(bounds) != self.n_weights:
            raise SystemExit(f"ERROR: {source} has {len(bounds)} specs but the config's weight slab has {self.n_weights} params")
        return replace(self, weights=tuple(bounds))

    def with_sidecar(self, run_dir: Path, announce: bool = False) -> ChromosomeLayout:
        """The layout under `<run_dir>/warm_start_bounds.json` when the run recorded one
        (decoding a warm-started population under rebuilt init bounds silently corrupts
        the weights), else this layout."""
        from aerocapture.training.warm_start import load_warm_start_bounds  # noqa: PLC0415

        bounds = load_warm_start_bounds(run_dir)
        if bounds is None:
            return self
        if announce:
            print(f"  Overlaid {len(bounds)} weight-spec bounds from warm_start_bounds.json")
        return self.with_weight_bounds(bounds, source="warm_start_bounds.json")

    def slab_bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """(lo, hi) of the weight slab, the arrays `run_grid` decodes the population under."""
        lo = np.array([s.p_min for s in self.weights], dtype=np.float64)
        hi = np.array([s.p_max for s in self.weights], dtype=np.float64)
        return lo, hi

    def decode(self, x: npt.NDArray[np.float64], context: str) -> DecodedChromosome:
        """Decode one normalized chromosome; `context` names the caller in the width error."""
        x = np.asarray(x, dtype=np.float64)
        self.check_width(x.shape[0], context)
        n_w = self.n_weights
        weights = _decode_nn_weights(x[:n_w], list(self.weights)) if self.weights else None
        return DecodedChromosome(weights, decode_normalized(x[n_w:], self.tail))
