"""Small numerical helpers used by the sampler implementation.

These helpers are deliberately minimal for the initial scaffold. They provide
stable log-space operations that are useful for nested-sampling bookkeeping and
are easy to validate independently before the sampler itself is implemented.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

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

    array = jnp.asarray(values, dtype=float)
    if array.size == 0:
        return float("-inf")

    maximum = jnp.max(array)
    if bool(jnp.isneginf(maximum)):
        return float("-inf")

    return float(maximum + jnp.log(jnp.sum(jnp.exp(array - maximum))))


def logdiffexp(a: float, b: float):
    """Return ``log(exp(a) - exp(b))`` safely.

    The real-valued result is defined only when ``a > b``.
    """

    if b >= a:
        raise ValueError("logdiffexp requires a > b")
    return a + jnp.log1p(-jnp.exp(b - a))


def normalize_log_weights(logw: ArrayLike):
    """Return log weights normalized to sum to one in probability space."""

    logw = jnp.asarray(logw, dtype=float)
    maximum = jnp.max(logw)
    return logw - (maximum + jnp.log(jnp.sum(jnp.exp(logw - maximum))))


def effective_sample_size_from_log_weights(logw: ArrayLike):
    """Return the effective sample size implied by unnormalized log weights."""

    normalized = normalize_log_weights(logw)
    weights = jnp.exp(normalized)
    return 1.0 / jnp.sum(jnp.square(weights))


def systematic_resample(key, logw: ArrayLike, n: int):
    """Draw ``n`` systematic-resampling indices from log weights."""

    normalized = normalize_log_weights(logw)
    weights = jnp.exp(normalized)
    cdf = jnp.cumsum(weights)
    start = random.uniform(key, shape=()) / n
    positions = start + jnp.arange(n) / n
    return jnp.searchsorted(cdf, positions, side="left")


def reflect_unit_cube(u: ArrayLike):
    """Reflect arbitrary coordinates into ``[0, 1]`` with mirror boundaries."""

    period_position = jnp.mod(u, 2.0)
    return jnp.where(period_position <= 1.0, period_position, 2.0 - period_position)
