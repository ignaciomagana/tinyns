"""Internal ellipsoid bounds in unit-cube coordinates.

These utilities are small geometry building blocks for bounded random-walk
sampler development. They operate on unit-cube coordinates ``u`` and are not
wired into nested sampling by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
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


@dataclass(frozen=True)
class JaxEllipsoidBound:
    """Padded array-backed ellipsoid union for future JAX kernels."""

    centers: jnp.ndarray
    chols: jnp.ndarray
    inv_chols: jnp.ndarray
    log_volumes: jnp.ndarray
    active: jnp.ndarray
    n_active: int
    ndim: int
    max_ellipsoids: int
    log_total_volume: float


def unit_ball_log_volume(ndim: int) -> float:
    """Return the log volume of the ``ndim``-dimensional unit ball."""

    if ndim < 1:
        raise ValueError("ndim must be positive")
    half_dim = 0.5 * ndim
    return half_dim * math.log(math.pi) - math.lgamma(half_dim + 1.0)


def _host_covariance_factor(points, *, jitter: float):
    """Return robust NumPy covariance geometry for host-side bound builders."""

    points_np = np.asarray(points, dtype=float)
    if points_np.ndim != 2:
        raise ValueError("points must have shape (npoints, ndim)")
    npoints, ndim = points_np.shape
    if npoints < 1 or ndim < 1:
        raise ValueError("points must have positive shape")

    center = np.mean(points_np, axis=0)
    centered = points_np - center
    denom = max(npoints - 1, 1)
    cov = (centered.T @ centered) / denom
    cov = 0.5 * (cov + cov.T)

    eye = np.eye(ndim, dtype=cov.dtype)
    base_jitter = float(jitter)
    chol = None
    for scale in (1.0, 10.0, 100.0, 1000.0, 1.0e4, 1.0e5, 1.0e6):
        try:
            cov_jittered = cov + base_jitter * scale * eye
            chol = np.linalg.cholesky(cov_jittered)
            break
        except np.linalg.LinAlgError:
            continue
    if chol is None:
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, base_jitter)
        chol = evecs @ np.diag(np.sqrt(evals))

    inv_chol = np.linalg.pinv(chol)
    whitened = centered @ inv_chol.T
    r2 = np.sum(whitened * whitened, axis=-1)
    rmax = float(np.sqrt(np.max(r2))) if r2.size else 0.0
    singular_values = np.linalg.svd(chol, compute_uv=False)
    singular_values = np.maximum(singular_values, np.finfo(float).tiny)
    logabsdet = float(np.sum(np.log(singular_values)))
    return center, chol, inv_chol, logabsdet, whitened, rmax


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

    live_np = np.asarray(live_u, dtype=float)
    if live_np.ndim != 2:
        raise ValueError("live_u must have shape (nlive, ndim)")

    nlive, ndim = live_np.shape
    if nlive < 1 or ndim < 1:
        raise ValueError("live_u must have shape (nlive, ndim) with positive sizes")

    center, cov_chol, _inv_cov_chol, _logabsdet, _whitened, rmax = (
        _host_covariance_factor(live_np, jitter=jitter)
    )
    scale = float(enlargement) * max(float(rmax), 1.0)
    chol_np = scale * cov_chol
    inv_chol_np = np.linalg.pinv(chol_np)
    singular_values = np.linalg.svd(chol_np, compute_uv=False)
    singular_values = np.maximum(singular_values, np.finfo(float).tiny)
    logabsdet = float(np.sum(np.log(singular_values)))
    log_volume = unit_ball_log_volume(ndim) + logabsdet
    return SingleEllipsoidBound(
        center=jnp.asarray(center),
        chol=jnp.asarray(chol_np),
        inv_chol=jnp.asarray(inv_chol_np),
        enlargement=float(enlargement),
        log_volume=float(log_volume),
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


def as_jax_ellipsoid_bound(
    bound: SingleEllipsoidBound | MultiEllipsoidBound | JaxEllipsoidBound,
    *,
    max_ellipsoids: int | None = None,
) -> JaxEllipsoidBound:
    """Convert an ellipsoid bound to a padded array-backed representation."""

    if isinstance(bound, JaxEllipsoidBound):
        if max_ellipsoids is None or max_ellipsoids == bound.max_ellipsoids:
            return bound
        if max_ellipsoids < bound.n_active:
            raise ValueError("max_ellipsoids cannot be smaller than active ellipsoids")
        raise ValueError("max_ellipsoids is incompatible with existing padding")

    if isinstance(bound, SingleEllipsoidBound):
        ellipsoids = (bound,)
        log_total_volume = float(bound.log_volume)
        ndim = int(bound.ndim)
    elif isinstance(bound, MultiEllipsoidBound):
        ellipsoids = bound.ellipsoids
        log_total_volume = float(bound.log_total_volume)
        ndim = int(bound.ndim)
    else:
        raise TypeError("bound must be a SingleEllipsoidBound or MultiEllipsoidBound")

    n_ellipsoids = len(ellipsoids)
    if n_ellipsoids < 1:
        raise ValueError("bound must contain at least one ellipsoid")
    if max_ellipsoids is None:
        max_ellipsoids = n_ellipsoids
    max_ellipsoids = int(max_ellipsoids)
    if max_ellipsoids < n_ellipsoids:
        raise ValueError("max_ellipsoids cannot be smaller than number of ellipsoids")

    centers = jnp.zeros((max_ellipsoids, ndim), dtype=float)
    chols = jnp.broadcast_to(jnp.eye(ndim), (max_ellipsoids, ndim, ndim)).astype(float)
    inv_chols = jnp.broadcast_to(jnp.eye(ndim), (max_ellipsoids, ndim, ndim)).astype(
        float
    )
    log_volumes = jnp.full((max_ellipsoids,), -jnp.inf, dtype=float)
    active = jnp.arange(max_ellipsoids) < n_ellipsoids

    active_centers = jnp.stack([ellipsoid.center for ellipsoid in ellipsoids])
    active_chols = jnp.stack([ellipsoid.chol for ellipsoid in ellipsoids])
    active_inv_chols = jnp.stack([ellipsoid.inv_chol for ellipsoid in ellipsoids])
    active_log_volumes = jnp.asarray(
        [ellipsoid.log_volume for ellipsoid in ellipsoids], dtype=float
    )
    centers = centers.at[:n_ellipsoids].set(active_centers)
    chols = chols.at[:n_ellipsoids].set(active_chols)
    inv_chols = inv_chols.at[:n_ellipsoids].set(active_inv_chols)
    log_volumes = log_volumes.at[:n_ellipsoids].set(active_log_volumes)

    return JaxEllipsoidBound(
        centers=centers,
        chols=chols,
        inv_chols=inv_chols,
        log_volumes=log_volumes,
        active=active,
        n_active=int(n_ellipsoids),
        ndim=ndim,
        max_ellipsoids=max_ellipsoids,
        log_total_volume=log_total_volume,
    )


def count_containing_jax_ellipsoids(
    bound: JaxEllipsoidBound,
    u: ArrayLike,
) -> jnp.ndarray:
    """Count padded JAX ellipsoids that contain one point or a batch."""

    u = jnp.asarray(u, dtype=float)
    if u.shape[-1:] != (bound.ndim,):
        raise ValueError("u must have shape (ndim,) or (..., ndim)")
    delta = u[..., None, :] - bound.centers
    x = jnp.einsum("...ed,ekd->...ek", delta, bound.inv_chols)
    contained = jnp.sum(jnp.square(x), axis=-1) <= 1.0 + 1e-10
    contained = contained & bound.active
    return jnp.sum(contained.astype(int), axis=-1)


def contains_jax_ellipsoid_bound(
    bound: JaxEllipsoidBound,
    u: ArrayLike,
) -> jnp.ndarray:
    """Return whether points lie inside at least one active JAX ellipsoid."""

    return count_containing_jax_ellipsoids(bound, u) > 0


def jax_bound_volume_probs(bound: JaxEllipsoidBound) -> jnp.ndarray:
    """Return normalized volume probabilities over padded ellipsoid rows."""

    probs = jnp.exp(bound.log_volumes - bound.log_total_volume)
    probs = jnp.where(bound.active, probs, 0.0)
    total = jnp.sum(probs)
    valid_total = jnp.isfinite(total) & (total > 0.0)
    safe_total = jnp.where(valid_total, total, 1.0)
    return jnp.where(valid_total, probs / safe_total, jnp.zeros_like(probs))


def _sample_unit_ball(key, n: int, ndim: int) -> jnp.ndarray:
    """Draw ``n`` points uniformly from the ``ndim``-dimensional unit ball."""

    key_dir, key_r = random.split(key)
    z = random.normal(key_dir, shape=(n, ndim))
    norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
    direction = z / jnp.maximum(norm, jnp.finfo(z.dtype).tiny)
    radius = random.uniform(key_r, shape=(n, 1)) ** (1.0 / ndim)
    return radius * direction


def sample_jax_ellipsoid_bound(key, bound: JaxEllipsoidBound, n: int):
    """Draw raw samples from volume-weighted active JAX ellipsoids.

    Returns ``(samples, indices)`` where ``indices`` identifies the active
    padded ellipsoid row used for each sample. This low-level helper samples
    ellipsoids by volume and does not correct for overlap between ellipsoids.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    key_idx, key_x = random.split(key)
    probs = jax_bound_volume_probs(bound)
    indices = random.choice(key_idx, bound.max_ellipsoids, shape=(n,), p=probs)
    x = _sample_unit_ball(key_x, n, bound.ndim)
    center = bound.centers[indices]
    chol = bound.chols[indices]
    samples = center + jnp.einsum("nd,nkd->nk", x, chol)
    return samples, indices


def sample_jax_ellipsoid_bound_corrected(
    key,
    bound: JaxEllipsoidBound,
    n: int,
    *,
    max_draws_multiplier: int = 10,
):
    """Draw approximately uniform samples from an overlapping JAX ellipsoid union.

    Candidates are drawn from volume-weighted ellipsoids and accepted with
    probability ``1 / count``, where ``count`` is the number of active
    ellipsoids containing the candidate. If the draw budget is exhausted, this
    returns fewer than ``n`` samples instead of failing.
    """

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
    while draws < budget and sum(sample.shape[0] for sample in accepted_samples) < n:
        batch = min(n, budget - draws)
        new_key, draw_key, accept_key = random.split(new_key, 3)
        candidates, indices = sample_jax_ellipsoid_bound(draw_key, bound, batch)
        counts = count_containing_jax_ellipsoids(bound, candidates)
        accept_prob = 1.0 / jnp.maximum(counts, 1)
        accepted = random.uniform(accept_key, shape=(batch,)) < accept_prob
        accepted_samples.append(candidates[accepted])
        accepted_indices.append(indices[accepted])
        draws += int(batch)
        overlap_rejections += int(jnp.sum(~accepted))

    if not accepted_samples:
        return (
            jnp.empty((0, bound.ndim)),
            jnp.empty((0,), dtype=int),
            draws,
            overlap_rejections,
        )

    samples = jnp.concatenate(accepted_samples, axis=0)[:n]
    indices = jnp.concatenate(accepted_indices, axis=0)[:n]
    return samples, indices, draws, overlap_rejections


def jax_in_unit_cube(u: ArrayLike) -> jnp.ndarray:
    """JAX-native unit-cube membership helper for one point or a batch."""

    return in_unit_cube(u)


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
    points_np = np.asarray(points, dtype=float)
    centered = points_np - np.mean(points_np, axis=0)
    denom = max(points_np.shape[0] - 1, 1)
    cov = (centered.T @ centered) / denom
    cov = 0.5 * (cov + cov.T)
    cov = cov + float(jitter) * np.eye(points_np.shape[1])
    try:
        _evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, -1]
    except np.linalg.LinAlgError:
        axis = np.zeros(points_np.shape[1])
        axis[0] = 1.0
    axis = axis / max(np.linalg.norm(axis), np.finfo(float).tiny)
    return jnp.asarray(axis)


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
