"""
QuickDraw stroke-sequence capability suite.

This package provides utilities to parse QuickDraw NDJSON, normalize strokes
into a canonical point sequence representation, apply deterministic transforms,
and run evaluation suites against pluggable model adapters.
"""

__all__ = [
    "data",
    "repr",
    "transforms",
    "eval",
    "adapters",
]
