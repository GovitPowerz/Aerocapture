"""TOML config loading with base inheritance resolution."""

import tomllib
from pathlib import Path


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay into base. Tables merge recursively; scalars/arrays replace."""
    result = dict(base)
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_toml_with_bases(path: Path, *, _visited: frozenset[Path] | None = None) -> dict:
    """Load a TOML file, recursively resolving ``base`` references.

    ``base`` can be a single string or array of strings, resolved relative
    to the declaring file's directory. Cycle detection via canonical paths.
    """
    path = Path(path).resolve()
    if _visited is None:
        _visited = frozenset()

    if path in _visited:
        msg = f"Cycle detected: '{path}' was already visited"
        raise ValueError(msg)

    _visited = _visited | {path}

    with open(path, "rb") as f:
        data = tomllib.load(f)

    base_refs = data.pop("base", None)
    if base_refs is None:
        return data

    if isinstance(base_refs, str):
        base_refs = [base_refs]

    base_dir = path.parent
    merged: dict = {}
    for ref in base_refs:
        base_path = (base_dir / ref).resolve()
        base_data = load_toml_with_bases(base_path, _visited=_visited)
        merged = _deep_merge(merged, base_data)

    return _deep_merge(merged, data)


def find_mission_name(toml_path: Path) -> str | None:
    """Mission name (stem of the first missions/ base) reachable through the
    base chain, depth-first. Shallow scans miss it for nested leaf configs
    (e.g. a config whose only base is another training leaf)."""
    import tomllib  # noqa: PLC0415

    def walk(path: Path) -> str | None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        bases = raw.get("base", [])
        if isinstance(bases, str):
            bases = [bases]
        for b in bases:
            if "missions/" in b:
                return Path(b).stem
        for b in bases:
            found = walk((path.parent / b).resolve())
            if found is not None:
                return found
        return None

    return walk(Path(toml_path).resolve())
