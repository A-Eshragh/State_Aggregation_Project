"""
Solver subpackage.

Avoid importing optional solver backends (e.g. Gurobi) at import-time.
Import concrete solvers directly from their modules as needed.
"""

__all__ = []

