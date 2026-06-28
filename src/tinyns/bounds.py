"""Internal single-ellipsoid bounds in unit-cube coordinates.

These utilities are small geometry building blocks for bounded random-walk
sampler development. They operate on unit-cube coordinates ``u`` and are not
wired into nested sampling by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
from jax import random

from tinyns.types import ArrayLike


@dataclass(frozen=True)
class SingleEllipsoidBound:
    """A single ellipsoid represented by an affine unit-ball transform."""

    center: jnp.ndarray
    chol: jnp.ndarray
    inv_chol: jnp.ndarray
    enlargement: float
    log_volume: float
    ndim: int


def unit_ball_log_volume(ndim: int) -> float:
    """Return the log volume of the ``ndim``-dimensional unit ball."""

    if ndim < 1:
        raise ValueError("ndim must be positive")
    half_dim = 0.5 * ndim
    return half_dim * math.log(math.pi) - math.lgamma(half_dim + 1.0)


def build_single_ellipsoid_bound(
    live_u: ArrayLike,
    *,
    enlargement: float = 1.25,
    jitter: float = 1e-6,
) -> SingleEllipsoidBound:
    """Build a single ellipsoid containing live points in unit-cube space."""

    if enlargement <= 0.0:
        raise ValueError("enlargement must be positive")
    if jitter <= 0.0:
        raise ValueError("jitter must be positive")

    live_u = jnp.asarray(live_u, dtype=float)
    if live_u.ndim != 2:
        raise ValueError("live_u must have shape (nlive, ndim)")

    nlive, ndim = live_u.shape
    if nlive < 1 or ndim < 1:
        raise ValueError("live_u must have shape (nlive, ndim) with positive sizes")

    center = jnp.mean(live_u, axis=0)
    centered = live_u - center
    cov = centered.T @ centered / max(nlive - 1, 1)
    cov = cov + jitter * jnp.eye(ndim)
    cov_chol = jnp.linalg.cholesky(cov)
    inv_cov_chol = jnp.linalg.inv(cov_chol)
    whitened = centered @ inv_cov_chol.T
    r2 = jnp.sum(jnp.square(whitened), axis=-1)
    rmax = jnp.sqrt(jnp.max(r2))
    scale = enlargement * jnp.maximum(rmax, 1.0)
    chol = scale * cov_chol
    inv_chol = jnp.linalg.inv(chol)
    sign, logabsdet = jnp.linalg.slogdet(chol)
    if not bool(sign > 0):
        raise ValueError("ellipsoid transform must have positive determinant")
    log_volume = unit_ball_log_volume(ndim) + float(logabsdet)
    return SingleEllipsoidBound(
        center=center,
        chol=chol,
        inv_chol=inv_chol,
        enlargement=float(enlargement),
        log_volume=log_volume,
        ndim=int(ndim),
    )


def contains_single_ellipsoid(bound: SingleEllipsoidBound, u: ArrayLike) -> jnp.ndarray:
    """Return whether one point or a batch of points lies inside ``bound``."""

    u = jnp.asarray(u, dtype=float)
    if u.shape[-1:] != (bound.ndim,):
        raise ValueError("u must have shape (ndim,) or (..., ndim)")
    delta = u - bound.center
    x = delta @ bound.inv_chol.T
    return jnp.sum(jnp.square(x), axis=-1) <= 1.0 + 1e-10


def sample_single_ellipsoid(key, bound: SingleEllipsoidBound, n: int):
    """Draw ``n`` samples uniformly from the raw ellipsoid."""

    if n < 0:
        raise ValueError("n must be non-negative")
    key_dir, key_r = random.split(key)
    z = random.normal(key_dir, shape=(n, bound.ndim))
    norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
    direction = z / jnp.maximum(norm, jnp.finfo(z.dtype).tiny)
    radius = random.uniform(key_r, shape=(n, 1)) ** (1.0 / bound.ndim)
    x = radius * direction
    return bound.center + x @ bound.chol.T


def in_unit_cube(u: ArrayLike) -> jnp.ndarray:
    """Return whether all coordinates of one point or batch are in ``[0, 1]``."""

    u = jnp.asarray(u, dtype=float)
    return jnp.all((u >= 0.0) & (u <= 1.0), axis=-1)
