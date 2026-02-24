"""Shared utilities for parsing Fortran output files."""

from __future__ import annotations

import re

# Matches Fortran floating-point values including:
# - Standard D/E notation: 0.1234D+02, -1.5E-03
# - Missing-D notation (format overflow): 0.97310-149, 0.34428-309
# - Plain numbers: 123, 1.5, -0.3
# The key pattern matches digits.digits followed by [+-]digits (implicit D)
_FORTRAN_TOKEN_RE = re.compile(
    r"[+-]?"  # optional sign
    r"(?:"
    r"\d+\.\d*[DEde][+-]?\d+"  # 1.23D+04 or 1.23E-05
    r"|\d+\.\d+[+-]\d+"  # 1.23456-309 (missing D, implicit exponent)
    r"|\d+\.\d*"  # 1.23 (plain decimal)
    r"|\d+"  # 123 (integer)
    r")"
)


def parse_fortran_float(s: str) -> float:
    """Parse a Fortran D-notation or implicit-exponent float to Python float.

    Handles: '0.1234D+02', '0.97310-149', '1.5', '123'
    """
    # Replace D/d notation with E
    s = s.replace("D", "E").replace("d", "e")
    # Handle missing exponent letter: if we have digits.digits[+-]digits with no E
    # e.g. '0.97310-149' -> need to insert E
    if "E" not in s and "e" not in s:
        match = re.match(r"([+-]?\d+\.\d+)([+-]\d+)$", s)
        if match:
            s = match.group(1) + "E" + match.group(2)
    return float(s)


def parse_fortran_line(line: str) -> list[float]:
    """Parse a line of Fortran output into a list of floats."""
    tokens = _FORTRAN_TOKEN_RE.findall(line)
    return [parse_fortran_float(t) for t in tokens]
