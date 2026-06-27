"""Small numerical helpers used by the sampler implementation.

These helpers are deliberately minimal for the initial scaffold. They provide
stable log-space operations that are useful for nested-sampling bookkeeping and
are easy to validate independently before the sampler itself is implemented.
"""

from __future__ import annotations

import numpy as np

from tinyns.types import ArrayLike


def logsumexp(values: ArrayLike) -> float:
    """Return ``log(sum(exp(values)))`` using a numerically stable algorithm.

    Parameters
    ----------
    values:
        One-dimensional or array-like collection of log-space values.

    Returns
    -------
    float
        The log of the summed exponentials. Empty inputs return ``-inf``.
    """

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("-inf")

    maximum = np.max(array)
    if np.isneginf(maximum):
        return float("-inf")

    return float(maximum + np.log(np.sum(np.exp(array - maximum))))


def logdiffexp(left: float, right: float) -> float:
    """Return ``log(exp(left) - exp(right))`` safely.

    Parameters
    ----------
    left:
        Logarithm of the larger positive value.
    right:
        Logarithm of the value to subtract.

    Raises
    ------
    ValueError
        If ``right`` is greater than ``left`` because the real-valued result
        would be negative.
    """

    if right > left:
        raise ValueError("logdiffexp requires right <= left")
    if right == left:
        return float("-inf")
    return float(left + np.log1p(-np.exp(right - left)))
