"""TOML config loading with base inheritance resolution."""

import tomllib
from io import TextIOWrapper
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


def set_dot_path(data: dict, dot_path: str, value: object) -> None:
    """Assign `value` at a dot-separated key path, creating intermediate tables."""
    parts = dot_path.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def write_toml(data: dict, path: Path) -> None:
    """Minimal TOML writer for nested dicts with scalar/list values."""
    with open(path, "w") as f:
        _write_toml_section(f, data, prefix="")


def _write_toml_section(f: TextIOWrapper, data: dict, prefix: str) -> None:
    """Recursively write TOML sections."""
    # First pass: write scalar/list values at this level
    for key, value in data.items():
        if not isinstance(value, dict):
            f.write(f"{key} = {_toml_value(value)}\n")

    # Second pass: write array-of-tables
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            full_key = f"{prefix}{key}" if prefix else key
            for item in value:
                f.write(f"\n[[{full_key}]]\n")
                _write_toml_section(f, item, prefix=f"{full_key}.")

    # Third pass: write subsections
    for key, value in data.items():
        if isinstance(value, dict):
            full_key = f"{prefix}{key}" if prefix else key
            # Write section header if this dict has scalar values
            scalars = {k: v for k, v in value.items() if not isinstance(v, dict) and not _is_table_array(v)}
            if scalars:
                f.write(f"\n[{full_key}]\n")
                for sk, sv in scalars.items():
                    if isinstance(sv, list) and sv and isinstance(sv[0], dict):
                        continue  # handled separately
                    f.write(f"{sk} = {_toml_value(sv)}\n")
            # Write array-of-tables within this section
            for sk, sv in value.items():
                if isinstance(sv, list) and sv and isinstance(sv[0], dict):
                    aot_key = f"{full_key}.{sk}"
                    for item in sv:
                        f.write(f"\n[[{aot_key}]]\n")
                        for ik, iv in item.items():
                            f.write(f"{ik} = {_toml_value(iv)}\n")
            # Recurse into nested dicts
            for sk, sv in value.items():
                if isinstance(sv, dict):
                    _write_toml_section(f, {sk: sv}, prefix=f"{full_key}.")


def _is_table_array(value: object) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], dict)


def _toml_value(value: object) -> str:
    """Format a Python value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    # Coerce numpy scalar floats (np.float64, np.float32, etc.) to plain Python float
    # before formatting; repr(np.float64(...)) produces invalid TOML like "np.float64(1e-07)".
    import numbers

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return repr(float(value))
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            # Inline table array -- should be handled as [[section]]
            items = []
            for item in value:
                fields = ", ".join(f"{k} = {_toml_value(v)}" for k, v in item.items())
                items.append(f"{{ {fields} }}")
            return f"[{', '.join(items)}]"
        return f"[{', '.join(_toml_value(v) for v in value)}]"
    return str(value)
