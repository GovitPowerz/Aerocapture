"""The scaffolding deploy rule: `best_params.json` -> finished TOML dot-path overrides.

Turning a tuned parameter dict into overrides that `aerocapture_rs.run_batch` /
`run_grid` accept needs three things, not one: the prefix -> section routing
(`route_param_path`), the rule that a tuned `shaping.*` parameter implies
`guidance.command_shaping.enabled = true`, and integer coercion for the params
whose `ParamSpec.is_integer` is set (the Rust TOML parser rejects `4.0` for an
integer field). This module is the only place that knows all three; every
deploy-side consumer (report, compare, demo, reference generation, warm-start
supervisors, animation, the paper scripts) calls it instead of re-deriving it.

Training-side note: `AerocaptureProblem._build_grid_overrides` routes with
`route_param_path` and its own `ParamSpec.is_integer` set and deliberately does
NOT add the enable flag -- a `[guidance.command_shaping]` section created by a
`max_bank_acceleration` override is enabled by default on the Rust side, so the
grid overrides stay byte-identical to the run_grid bit-identity gate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS, PARAM_SPACES, SCAFFOLDING_PREFIXES, route_param_path

# Every ParamSpec flagged is_integer across all schemes + the NN scaffolding pack.
# `lateral.max_reversals` today; extend a ParamSpec, not this module, to add one.
INTEGER_PARAM_NAMES: frozenset[str] = frozenset(spec.name for specs in (*PARAM_SPACES.values(), _NN_SCAFFOLDING_PARAMS) for spec in specs if spec.is_integer)

_SHAPING_ENABLED_KEY = "guidance.command_shaping.enabled"


def overrides_from_params(params: Mapping[str, object], scheme: str, *, scaffolding_only: bool = False) -> dict[str, object]:
    """Finished dot-path override dict for `params` deployed on `scheme`.

    - prefixed keys (`lateral.` / `exit.` / `nav.` / `thermal.` / `shaping.`) route to
      their fixed sections; unprefixed keys route to `guidance.<scheme>.<key>`
    - any `shaping.*` key sets `guidance.command_shaping.enabled = True`
    - `INTEGER_PARAM_NAMES` values are rounded to int
    - `ref_bank` (the joint-reference gene) is skipped: it is not a guidance TOML
      key and deploys as the cell's own reference table instead
    - `scaffolding_only=True` keeps prefixed keys only (a `best_params.json`
      written next to an NN model carries just the scaffolding pack)
    """
    overrides: dict[str, object] = {}
    for key, value in params.items():
        if key == "ref_bank":
            continue
        if scaffolding_only and not key.startswith(SCAFFOLDING_PREFIXES):
            continue
        coerced: object = int(round(float(value))) if key in INTEGER_PARAM_NAMES else value  # type: ignore[arg-type]
        overrides[route_param_path(key, scheme)] = coerced
        if key.startswith("shaping."):
            overrides[_SHAPING_ENABLED_KEY] = True
    return overrides


def load_scaffolding_overrides(cell_dir: Path) -> dict[str, object]:
    """Overrides for the scaffolding pack in `<cell_dir>/best_params.json`; `{}` if absent.

    NN cells trained with `scaffolding = "live" | "full"` write `best_model.json`
    plus `best_params.json`; with `scaffolding = "off"` the params file is absent.
    """
    scaff_path = cell_dir / "best_params.json"
    if not scaff_path.exists():
        return {}
    params: dict[str, object] = json.loads(scaff_path.read_text())
    return overrides_from_params(params, "", scaffolding_only=True)


def resolve_eval_toml(toml_path: Path, scheme_dir: Path) -> tuple[Path, dict[str, object]]:
    """The TOML to evaluate a cell with, plus its scaffolding overrides.

    Non-NN schemes bake their tuned params into `optimized_<scheme>.toml`, which
    wins when present (the file is named after the guidance type, which differs
    from the dir name for custom --output-dir runs, hence the single-match glob
    fallback) and needs no overrides. Otherwise the base TOML is used with the
    cell's `best_params.json` scaffolding routed as overrides.
    """
    optimized = scheme_dir / f"optimized_{scheme_dir.name}.toml"
    if not optimized.exists():
        candidates = sorted(scheme_dir.glob("optimized_*.toml"))
        if len(candidates) == 1:
            optimized = candidates[0]
    if optimized.exists():
        return optimized, {}
    return toml_path, load_scaffolding_overrides(scheme_dir)
