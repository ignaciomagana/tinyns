"""Small log-space helpers for the result object: weights, ESS, resampling.

They compute in NumPy float64 on the host, whatever JAX's precision: a
result's ``logwt`` is float64, and normalizing it in float32 loses weight once
``|logwt|`` is large (at ``|logwt| ~ 1e6`` the float32 weights are off by about
3%). On the host they also compile nothing, so a new result (of a new sample
count) costs no compilation. Only :func:`systematic_resample` draws from a JAX
key, its start offset.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from jax import random

ArrayLike = Any


def as_key(key):
    """Return ``key`` as a JAX PRNG key: an int seed becomes ``PRNGKey(seed)``."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return random.PRNGKey(int(key))
    return key


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

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("-inf")
    if np.any(np.isnan(array)):
        return float("nan")
    if np.any(np.isposinf(array)):
        return float("inf")

    maximum = np.max(array)
    if np.isneginf(maximum):
        return float("-inf")

    return float(maximum + np.log(np.sum(np.exp(array - maximum))))


def logdiffexp(a: float, b: float) -> float:
    """Return ``log(exp(a) - exp(b))`` safely.

    The real-valued result is defined only when ``a > b``.
    """

    if b >= a:
        raise ValueError("logdiffexp requires a > b")
    return float(a + np.log1p(-np.exp(b - a)))


def normalize_log_weights(logw: ArrayLike) -> np.ndarray:
    """Return log weights normalized to sum to one in probability space
    (NumPy float64).

    Empty, NaN, and all-negative-infinity inputs have no defined probability
    distribution and raise ValueError. If one or more entries are positive
    infinity, probability is shared uniformly among those entries.
    """

    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        raise ValueError("log weights must not be empty")
    if np.any(np.isnan(logw)):
        raise ValueError("log weights must not contain NaN")

    positive_infinity = np.isposinf(logw)
    if np.any(positive_infinity):
        count = np.count_nonzero(positive_infinity)
        return np.where(positive_infinity, -math.log(count), -np.inf)

    maximum = np.max(logw)
    if np.isneginf(maximum):
        raise ValueError("log weights must contain at least one finite value")
    return logw - (maximum + np.log(np.sum(np.exp(logw - maximum))))


def effective_sample_size_from_log_weights(logw: ArrayLike) -> float:
    """Return the effective sample size implied by unnormalized log weights."""

    weights = np.exp(normalize_log_weights(logw))
    return float(1.0 / np.sum(np.square(weights)))


def systematic_resample(key, logw: ArrayLike, n: int) -> np.ndarray:
    """Draw ``n`` systematic-resampling indices from log weights.

    ``key`` is a PRNG key or an int seed; it draws the offset ``U`` of the
    ``n`` evenly spaced positions ``(U + i) / n`` (in float32, so the indices
    do not depend on the x64 flag). Index ``j`` comes ``floor(n w_j)`` or
    ``ceil(n w_j)`` times, and the indices are in increasing order: they are
    not independent draws, and a prefix of them is not a sample of the
    weights (:meth:`tinyns.NestedSamplingResult.resample_equal` permutes
    them). An index of zero weight is never drawn.
    """

    cdf = np.cumsum(np.exp(normalize_log_weights(logw)))
    cdf /= cdf[-1]  # exactly 1 at the end, so no position lies beyond it
    start = float(random.uniform(as_key(key), (), np.float32))
    positions = (start + np.arange(n)) / n
    return np.searchsorted(cdf, positions, side="right")
