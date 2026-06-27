"""Shared type aliases for :mod:`tinyns`.

The aliases in this module intentionally stay lightweight so the public API can
accept NumPy arrays, JAX arrays, and array-like inputs without committing to a
single concrete array class too early in the project.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias

ArrayLike: TypeAlias = Any
"""An object that can be interpreted as an array by NumPy or JAX."""

PRNGKeyLike: TypeAlias = Any
"""A JAX pseudo-random number generator key or compatible key-like object."""

LogLikelihood: TypeAlias = Callable[[ArrayLike], float]
"""Callable that evaluates the log likelihood at a point in parameter space."""

PriorTransform: TypeAlias = Callable[[ArrayLike], ArrayLike]
"""Callable that maps a point from the unit cube into parameter space."""


class SupportsRun(Protocol):
    """Protocol for sampler-like objects that expose a ``run`` method."""

    def run(self, key: PRNGKeyLike, *, dlogz: float = 0.1) -> Any:
        """Run a sampler and return a result object."""
