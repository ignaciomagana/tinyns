"""Internal ellipsoid bounds in unit-cube coordinates.

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


@dataclass(frozen=True)
class MultiEllipsoidBound:
    """A deterministic union of single ellipsoid bounds."""

    ellipsoids: tuple[SingleEllipsoidBound, ...]
    log_volumes: jnp.ndarray
    log_total_volume: float
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


def _logsumexp_pair(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _principal_axis(points: jnp.ndarray, jitter: float) -> jnp.ndarray:
    centered = points - jnp.mean(points, axis=0)
    denom = max(int(points.shape[0]) - 1, 1)
    cov = (centered.T @ centered) / denom
    cov = cov + jitter * jnp.eye(points.shape[1])
    _evals, evecs = jnp.linalg.eigh(cov)
    return evecs[:, -1]


def _split_cluster(points: jnp.ndarray, *, min_points: int, jitter: float):
    axis = _principal_axis(points, jitter)
    projection = points @ axis
    order = jnp.argsort(projection)
    mid = int(points.shape[0]) // 2
    left = points[order[:mid]]
    right = points[order[mid:]]
    if left.shape[0] < min_points or right.shape[0] < min_points:
        return None
    return left, right


def build_multi_ellipsoid_bound(
    live_u: ArrayLike,
    *,
    enlargement: float = 1.25,
    jitter: float = 1e-6,
    max_ellipsoids: int = 32,
    min_points: int | None = None,
    split_threshold: float = 0.9,
) -> MultiEllipsoidBound:
    """Build a greedy PCA/median-split multiellipsoid bound.

    The splitter is deterministic: repeatedly try the largest-volume cluster
    and accept a median split along its principal axis only when child volumes
    reduce total volume by ``split_threshold``.
    """

    live_u = jnp.asarray(live_u, dtype=float)
    if live_u.ndim != 2:
        raise ValueError("live_u must have shape (nlive, ndim)")
    nlive, ndim = live_u.shape
    if nlive < 1 or ndim < 1:
        raise ValueError("live_u must have shape (nlive, ndim) with positive sizes")
    if max_ellipsoids <= 0:
        raise ValueError("max_ellipsoids must be a positive integer")
    if min_points is None:
        min_points = max(2 * int(ndim) + 2, 16)
    if min_points <= 0:
        raise ValueError("min_points must be a positive integer or None")
    if split_threshold <= 0.0:
        raise ValueError("split_threshold must be positive")

    clusters = [
        (
            live_u,
            build_single_ellipsoid_bound(
                live_u, enlargement=enlargement, jitter=jitter
            ),
        )
    ]
    blocked: set[int] = set()
    while len(clusters) < max_ellipsoids:
        candidates = [
            (idx, bound.log_volume)
            for idx, (points, bound) in enumerate(clusters)
            if idx not in blocked and int(points.shape[0]) >= 2 * min_points
        ]
        if not candidates:
            break
        split_idx = max(candidates, key=lambda item: item[1])[0]
        points, parent = clusters[split_idx]
        split = _split_cluster(points, min_points=min_points, jitter=jitter)
        if split is None:
            blocked.add(split_idx)
            continue
        left, right = split
        left_bound = build_single_ellipsoid_bound(
            left, enlargement=enlargement, jitter=jitter
        )
        right_bound = build_single_ellipsoid_bound(
            right, enlargement=enlargement, jitter=jitter
        )
        child_log_volume = _logsumexp_pair(
            left_bound.log_volume, right_bound.log_volume
        )
        if child_log_volume < parent.log_volume + math.log(split_threshold):
            clusters[split_idx] = (left, left_bound)
            clusters.append((right, right_bound))
            blocked = set()
        else:
            blocked.add(split_idx)

    ellipsoids = tuple(bound for _points, bound in clusters)
    log_volumes = jnp.asarray([bound.log_volume for bound in ellipsoids], dtype=float)
    log_total_volume = float(
        jnp.max(log_volumes)
        + jnp.log(jnp.sum(jnp.exp(log_volumes - jnp.max(log_volumes))))
    )
    return MultiEllipsoidBound(
        ellipsoids=ellipsoids,
        log_volumes=log_volumes,
        log_total_volume=log_total_volume,
        ndim=int(ndim),
    )


def contains_multi_ellipsoid(bound: MultiEllipsoidBound, u: ArrayLike) -> jnp.ndarray:
    """Return whether points lie inside at least one ellipsoid in ``bound``."""

    return count_containing_ellipsoids(bound, u) > 0


def count_containing_ellipsoids(
    bound: MultiEllipsoidBound, u: ArrayLike
) -> jnp.ndarray:
    """Count how many ellipsoids contain each point."""

    u = jnp.asarray(u, dtype=float)
    if u.shape[-1:] != (bound.ndim,):
        raise ValueError("u must have shape (ndim,) or (..., ndim)")
    counts = jnp.zeros(u.shape[:-1], dtype=int)
    for ellipsoid in bound.ellipsoids:
        counts = counts + contains_single_ellipsoid(ellipsoid, u).astype(int)
    return counts


def sample_multi_ellipsoid(key, bound: MultiEllipsoidBound, n: int):
    """Draw raw samples from volume-weighted ellipsoids and return indices."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if not bound.ellipsoids:
        raise ValueError("bound must contain at least one ellipsoid")
    key_idx, *sample_keys = random.split(key, len(bound.ellipsoids) + 1)
    probs = jnp.exp(bound.log_volumes - bound.log_total_volume)
    indices = random.choice(key_idx, len(bound.ellipsoids), shape=(n,), p=probs)
    per_ellipsoid = [
        sample_single_ellipsoid(sample_key, ellipsoid, n)
        for sample_key, ellipsoid in zip(sample_keys, bound.ellipsoids, strict=True)
    ]
    stacked = jnp.stack(per_ellipsoid, axis=0)
    return stacked[indices, jnp.arange(n)], indices


def sample_multi_ellipsoid_corrected(
    key,
    bound: MultiEllipsoidBound,
    n: int,
    *,
    max_draws_multiplier: int = 10,
):
    """Draw approximately uniform samples from an overlapping ellipsoid union."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if max_draws_multiplier <= 0:
        raise ValueError("max_draws_multiplier must be positive")
    if n == 0:
        return jnp.empty((0, bound.ndim)), jnp.empty((0,), dtype=int), 0, 0
    budget = max(n, int(n) * int(max_draws_multiplier))
    accepted_samples = []
    accepted_indices = []
    draws = 0
    overlap_rejections = 0
    new_key = key
    while len(accepted_samples) < n and draws < budget:
        batch = min(n, budget - draws)
        new_key, draw_key, accept_key = random.split(new_key, 3)
        candidates, indices = sample_multi_ellipsoid(draw_key, bound, batch)
        counts = count_containing_ellipsoids(bound, candidates)
        accept_prob = 1.0 / jnp.maximum(counts, 1)
        accepted = random.uniform(accept_key, shape=(batch,)) < accept_prob
        draws += int(batch)
        overlap_rejections += int(jnp.sum(~accepted))
        for candidate, index, is_accepted in zip(
            candidates, indices, accepted, strict=True
        ):
            if bool(is_accepted):
                accepted_samples.append(candidate)
                accepted_indices.append(index)
                if len(accepted_samples) == n:
                    break
    if not accepted_samples:
        return (
            jnp.empty((0, bound.ndim)),
            jnp.empty((0,), dtype=int),
            draws,
            overlap_rejections,
        )
    return (
        jnp.stack(accepted_samples),
        jnp.asarray(accepted_indices, dtype=int),
        draws,
        overlap_rejections,
    )


def in_unit_cube(u: ArrayLike) -> jnp.ndarray:
    """Return whether all coordinates of one point or batch are in ``[0, 1]``."""

    u = jnp.asarray(u, dtype=float)
    return jnp.all((u >= 0.0) & (u <= 1.0), axis=-1)
