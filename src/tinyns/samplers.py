"""Replacement samplers for :mod:`tinyns`."""

from __future__ import annotations

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from tinyns.bounds import (
    JaxEllipsoidBound,
    MultiEllipsoidBound,
    as_jax_ellipsoid_bound,
    count_containing_jax_ellipsoids,
    in_unit_cube,
    sample_jax_ellipsoid_bound,
    sample_jax_ellipsoid_bound_corrected,
    sample_single_ellipsoid,
)
from tinyns.math import reflect_unit_cube
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike


class _IdentityCacheKey:
    """Hash an otherwise unhashable object by identity for internal caches."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _IdentityCacheKey) and self.value is other.value
        )


def _cacheable_callable(value):
    try:
        hash(value)
    except TypeError:
        return _IdentityCacheKey(value)
    return value


def _unwrap_cacheable_callable(value):
    return value.value if isinstance(value, _IdentityCacheKey) else value


def _is_dynamic_leaf(leaf) -> bool:
    return isinstance(leaf, (jax.Array, np.ndarray))


def _partition_callable(fn):
    """Split a pytree callable into ``(dynamic, static)`` (eqx.partition-style).

    ``dynamic`` is the tuple of array leaves (``jax.Array`` / ``np.ndarray``) of
    ``jax.tree_util.tree_flatten(fn)``; ``static`` holds the treedef and every
    other leaf. The compiled fast-path kernels take ``dynamic`` as explicit jit
    arguments and rebuild ``fn`` inside the trace with
    :func:`_combine_callable`, so large data are not embedded as constants.

    A plain function or closure flattens to ``[fn]``: ``dynamic`` is empty and
    :func:`_combine_callable` returns ``fn`` itself (closure semantics).

    Only the fast-path jit boundaries use this split: the unbounded rwalk
    kernel (:func:`_make_rwalk_jax_kernel_cached`, hence
    :func:`draw_constrained_rwalk_jax`), the block kernel and the initial
    live-point pass in :mod:`tinyns.run`. The bounded and fused-bounded
    kernels, the ``replacement_chain_schedule`` adaptive kernels, the python
    kernel and prior sampling keep closure semantics: pytree callables work
    there, but their arrays are embedded in the compiled programs as constants.
    """
    leaves, treedef = jax.tree_util.tree_flatten(fn)
    mask = tuple(_is_dynamic_leaf(leaf) for leaf in leaves)
    dynamic = tuple(leaf for leaf in leaves if _is_dynamic_leaf(leaf))
    static_leaves = tuple(None if _is_dynamic_leaf(leaf) else leaf for leaf in leaves)
    return dynamic, (treedef, mask, static_leaves)


def _combine_callable(dynamic, static):
    """Rebuild the callable split by :func:`_partition_callable`."""
    treedef, mask, static_leaves = static
    dynamic = tuple(dynamic)
    if len(dynamic) != sum(mask):
        raise ValueError(
            f"expected {sum(mask)} dynamic callable leaves, got {len(dynamic)}"
        )
    it = iter(dynamic)
    leaves = [
        next(it) if is_dyn else leaf
        for leaf, is_dyn in zip(static_leaves, mask, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _callable_leaves(loglike, prior_transform):
    """Return the dynamic leaves of ``loglike`` then ``prior_transform``."""
    return _partition_callable(loglike)[0] + _partition_callable(prior_transform)[0]


def _validate_theta_shape(theta, ndim: int):
    theta = jnp.asarray(theta)
    if ndim == 1 and theta.shape == ():
        theta = theta.reshape((1,))
    if theta.shape != (ndim,):
        raise ValueError(f"prior_transform must return shape ({ndim},)")
    return theta


def _evaluate_jax_prior_batch(
    prior_transform,
    u_batch,
    ndim: int,
    *,
    jax_vectorized: bool,
):
    """Evaluate a JAX prior transform and normalize its batch shape."""
    u_batch = jnp.asarray(u_batch)
    if u_batch.ndim != 2 or u_batch.shape[1] != ndim:
        raise ValueError(f"u_batch must have shape (batch, {ndim})")
    batch_size = int(u_batch.shape[0])

    if jax_vectorized:
        theta_batch = jnp.asarray(prior_transform(u_batch))
        if theta_batch.shape != (batch_size, ndim):
            raise ValueError(
                "jax_vectorized prior_transform must return shape "
                f"({batch_size}, {ndim}); got {theta_batch.shape}"
            )
        return theta_batch

    theta_batch = jnp.asarray(jax.vmap(prior_transform)(u_batch))
    if ndim == 1 and theta_batch.shape == (batch_size,):
        theta_batch = theta_batch.reshape((batch_size, 1))
    if theta_batch.shape != (batch_size, ndim):
        raise ValueError(f"prior_transform must return shape ({ndim},)")
    return theta_batch


def _evaluate_jax_batch(
    loglike,
    prior_transform,
    u_batch,
    ndim,
    *,
    jax_vectorized: bool,
):
    """Evaluate JAX prior/likelihood functions on a unit-cube batch."""
    u_batch = jnp.asarray(u_batch)
    if u_batch.ndim != 2 or u_batch.shape[1] != ndim:
        raise ValueError(f"u_batch must have shape (batch, {ndim})")
    batch_size = int(u_batch.shape[0])
    theta_batch = _evaluate_jax_prior_batch(
        prior_transform,
        u_batch,
        ndim,
        jax_vectorized=jax_vectorized,
    )

    if jax_vectorized:
        logl_batch = jnp.asarray(loglike(theta_batch))
        if logl_batch.shape != (batch_size,):
            raise ValueError(
                "jax_vectorized loglike must return shape "
                f"({batch_size},); got {logl_batch.shape}"
            )
        return theta_batch, logl_batch

    logl_batch = jnp.asarray(jax.vmap(loglike)(theta_batch))
    if logl_batch.shape != (batch_size,):
        raise ValueError("loglike must return a scalar")
    return theta_batch, logl_batch


def _validate_min_accepts(min_accepts: int) -> None:
    if (
        not isinstance(min_accepts, int)
        or isinstance(min_accepts, bool)
        or min_accepts <= 0
    ):
        raise ValueError("min_accepts must be a positive integer")


def _bound_info(bound, bound_draws, unit_cube_survivors, ncall, overlap_rejections):
    log_volume = (
        bound.log_total_volume
        if isinstance(bound, MultiEllipsoidBound)
        else bound.log_volume
    )
    return {
        "bound_draws": bound_draws,
        "bound_unit_cube_survivors": unit_cube_survivors,
        "bound_loglike_evals": ncall,
        "bound_acceptance": ncall / bound_draws if bound_draws else 0.0,
        "bound_unit_cube_acceptance": (
            unit_cube_survivors / bound_draws if bound_draws else 0.0
        ),
        "bound_nellipsoids": len(getattr(bound, "ellipsoids", (bound,))),
        "bound_log_total_volume": log_volume,
        "bound_overlap_rejections": overlap_rejections,
    }


def draw_constrained_single_bound(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    bound,
    ndim: int,
    *,
    max_attempts: int = 10_000,
    batch_size: int = 128,
    overlap_correction: bool = True,
):
    """Draw a constrained replacement from an ellipsoid bound.

    Raw ellipsoid draws are first clipped to the unit cube. Only unit-cube
    survivors are transformed and counted as likelihood calls.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if getattr(bound, "ndim", ndim) != ndim:
        raise ValueError("bound dimensionality must match ndim")

    is_multi = isinstance(bound, MultiEllipsoidBound)
    jax_bound = as_jax_ellipsoid_bound(bound) if is_multi else None

    best_u = None
    best_theta = None
    best_logl = -math.inf
    ncall = 0
    bound_draws = 0
    overlap_rejections = 0
    unit_cube_survivors = 0
    new_key = key

    while ncall < max_attempts:
        new_key, draw_key = random.split(new_key)
        if is_multi:
            if overlap_correction:
                u_batch, _idx, draws, overlap = sample_jax_ellipsoid_bound_corrected(
                    draw_key, jax_bound, batch_size
                )
                bound_draws += int(draws)
                overlap_rejections += int(overlap)
            else:
                u_batch, _idx = sample_jax_ellipsoid_bound(
                    draw_key, jax_bound, batch_size
                )
                bound_draws += int(batch_size)
        else:
            u_batch = sample_single_ellipsoid(draw_key, bound, batch_size)
            bound_draws += int(batch_size)
        inside = in_unit_cube(u_batch)
        for u in u_batch[inside]:
            if ncall >= max_attempts:
                break
            unit_cube_survivors += 1
            theta = _validate_theta_shape(prior_transform(u), ndim)
            logl = float(loglike(theta))
            ncall += 1
            if best_u is None or logl > best_logl:
                best_u = u
                best_theta = theta
                best_logl = logl
            if logl >= logl_min:
                info = _bound_info(
                    bound, bound_draws, unit_cube_survivors, ncall, overlap_rejections
                )
                return new_key, u, theta, logl, ncall, True, info

    if best_u is None:
        best_u = jnp.full((ndim,), 0.5)
        best_theta = _validate_theta_shape(prior_transform(best_u), ndim)
        best_logl = float(loglike(best_theta))
        ncall += 1
        unit_cube_survivors += 1
    info = _bound_info(
        bound, bound_draws, unit_cube_survivors, ncall, overlap_rejections
    )
    info["bound_acceptance"] = 0.0
    return new_key, best_u, best_theta, best_logl, ncall, False, info


def draw_constrained_single_bound_jax(
    key,
    loglike,
    prior_transform,
    logl_min,
    bound,
    ndim: int,
    *,
    batch_size: int = 128,
    max_batches: int = 100,
    jax_vectorized: bool = False,
):
    """Draw a constrained replacement from one ellipsoid using JAX batches.

    Candidate generation, prior transforms, and likelihood calls are performed
    in fixed-size batches. This implementation evaluates the likelihood for
    every candidate in each attempted batch, including candidates outside the
    unit cube, then masks outside-cube likelihoods to ``-inf`` for acceptance
    and fallback selection. Therefore ``ncall`` and
    ``info["bound_seed_loglike_evals"]`` count all batch candidates evaluated.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_batches <= 0:
        raise ValueError("max_batches must be a positive integer")

    jax_bound = as_jax_ellipsoid_bound(bound)
    if jax_bound.ndim != ndim:
        raise ValueError("bound dimensionality must match ndim")
    if jax_bound.n_active != 1:
        raise ValueError(
            "draw_constrained_single_bound_jax requires one active ellipsoid"
        )
    if (
        not isinstance(bound, JaxEllipsoidBound)
        and getattr(bound, "ndim", ndim) != ndim
    ):
        raise ValueError("bound dimensionality must match ndim")

    best_u = None
    best_theta = None
    best_logl = -jnp.inf
    ncall = 0
    bound_draws = 0
    unit_cube_survivors = 0
    batches = 0
    new_key = key

    for _ in range(max_batches):
        batches += 1
        new_key, draw_key, select_key = random.split(new_key, 3)
        u_batch, _idx = sample_jax_ellipsoid_bound(draw_key, jax_bound, batch_size)
        if u_batch.shape != (batch_size, ndim):
            raise ValueError(f"ellipsoid draw must return shape ({batch_size}, {ndim})")

        inside = in_unit_cube(u_batch)
        safe_u_batch = jnp.where(inside[:, None], u_batch, 0.5)
        theta_batch, logl_batch = _evaluate_jax_batch(
            loglike,
            prior_transform,
            safe_u_batch,
            ndim,
            jax_vectorized=jax_vectorized,
        )

        ncall += int(batch_size)
        bound_draws += int(batch_size)
        unit_cube_survivors += int(jnp.sum(inside))
        masked_logl = jnp.where(inside, logl_batch, -jnp.inf)

        if bool(jnp.any(inside)):
            batch_best_idx = int(jnp.argmax(masked_logl))
            batch_best_logl = masked_logl[batch_best_idx]
            if best_u is None or bool(batch_best_logl > best_logl):
                best_u = u_batch[batch_best_idx]
                best_theta = theta_batch[batch_best_idx]
                best_logl = batch_best_logl

        accepted_mask = masked_logl >= logl_min
        n_accepted = int(jnp.sum(accepted_mask))
        if n_accepted > 0:
            accepted_indices = jnp.nonzero(
                accepted_mask, size=batch_size, fill_value=0
            )[0]
            chosen_rank = random.randint(select_key, (), 0, n_accepted)
            chosen_idx = accepted_indices[chosen_rank]
            info = {
                "bound_seed_draws": bound_draws,
                "bound_seed_loglike_evals": ncall,
                "bound_seed_batches": batches,
                "bound_seed_unit_cube_acceptance": unit_cube_survivors / bound_draws
                if bound_draws
                else 0.0,
            }
            return (
                new_key,
                u_batch[chosen_idx],
                theta_batch[chosen_idx],
                float(logl_batch[chosen_idx]),
                ncall,
                True,
                info,
            )

    if best_u is None:
        best_u = jnp.full((ndim,), 0.5)
        best_theta = _validate_theta_shape(prior_transform(best_u), ndim)
        best_logl = loglike(best_theta)
        ncall += 1
    info = {
        "bound_seed_draws": bound_draws,
        "bound_seed_loglike_evals": ncall,
        "bound_seed_batches": batches,
        "bound_seed_unit_cube_acceptance": unit_cube_survivors / bound_draws
        if bound_draws
        else 0.0,
    }
    return new_key, best_u, best_theta, float(best_logl), ncall, False, info


def draw_constrained_multi_bound_jax(
    key,
    loglike,
    prior_transform,
    logl_min,
    bound,
    ndim: int,
    *,
    batch_size: int = 128,
    max_batches: int = 100,
    overlap_correction: bool = True,
    jax_vectorized: bool = False,
):
    """Draw a constrained replacement from a JAX multiellipsoid bound.

    Candidates are drawn from volume-weighted active ellipsoids. When requested,
    overlap correction keeps each candidate with probability ``1 / count``, where
    ``count`` is the number of active ellipsoids containing it. Accepted
    likelihood-constrained candidates are selected uniformly at random within the
    first successful batch rather than by likelihood rank.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_batches <= 0:
        raise ValueError("max_batches must be a positive integer")

    jax_bound = as_jax_ellipsoid_bound(bound)
    if jax_bound.ndim != ndim:
        raise ValueError("bound dimensionality must match ndim")
    if jax_bound.n_active < 1:
        raise ValueError("bound must contain at least one active ellipsoid")
    if (
        not isinstance(bound, JaxEllipsoidBound)
        and getattr(bound, "ndim", ndim) != ndim
    ):
        raise ValueError("bound dimensionality must match ndim")

    best_u = None
    best_theta = None
    best_logl = -jnp.inf
    ncall = 0
    bound_draws = 0
    unit_cube_survivors = 0
    overlap_rejections = 0
    batches = 0
    new_key = key

    for _ in range(max_batches):
        batches += 1
        new_key, draw_key, overlap_key, select_key = random.split(new_key, 4)
        u_batch, idx_batch = sample_jax_ellipsoid_bound(draw_key, jax_bound, batch_size)
        if u_batch.shape != (batch_size, ndim):
            raise ValueError(f"ellipsoid draw must return shape ({batch_size}, {ndim})")

        overlap_mask = jnp.ones((batch_size,), dtype=bool)
        if overlap_correction:
            counts = count_containing_jax_ellipsoids(jax_bound, u_batch)
            overlap_prob = 1.0 / jnp.maximum(counts, 1)
            overlap_mask = (
                random.uniform(overlap_key, shape=(batch_size,)) < overlap_prob
            )
            overlap_rejections += int(jnp.sum(~overlap_mask))

        inside = in_unit_cube(u_batch)
        usable = overlap_mask & inside
        safe_u_batch = jnp.where(usable[:, None], u_batch, 0.5)
        theta_batch, logl_batch = _evaluate_jax_batch(
            loglike,
            prior_transform,
            safe_u_batch,
            ndim,
            jax_vectorized=jax_vectorized,
        )

        ncall += int(batch_size)
        bound_draws += int(batch_size)
        unit_cube_survivors += int(jnp.sum(usable))
        masked_logl = jnp.where(usable, logl_batch, -jnp.inf)

        if bool(jnp.any(usable)):
            batch_best_idx = int(jnp.argmax(masked_logl))
            batch_best_logl = masked_logl[batch_best_idx]
            if best_u is None or bool(batch_best_logl > best_logl):
                best_u = u_batch[batch_best_idx]
                best_theta = theta_batch[batch_best_idx]
                best_logl = batch_best_logl

        accepted_mask = masked_logl >= logl_min
        n_accepted = int(jnp.sum(accepted_mask))
        if n_accepted > 0:
            accepted_indices = jnp.nonzero(
                accepted_mask, size=batch_size, fill_value=0
            )[0]
            chosen_rank = random.randint(select_key, (), 0, n_accepted)
            chosen_idx = accepted_indices[chosen_rank]
            info = _bound_seed_info(
                bound_draws,
                ncall,
                batches,
                unit_cube_survivors,
                overlap_rejections,
                jax_bound.n_active,
            )
            info["bound_seed_ellipsoid_index"] = int(idx_batch[chosen_idx])
            return (
                new_key,
                u_batch[chosen_idx],
                theta_batch[chosen_idx],
                float(logl_batch[chosen_idx]),
                ncall,
                True,
                info,
            )

    if best_u is None:
        best_u = jnp.full((ndim,), 0.5)
        best_theta = _validate_theta_shape(prior_transform(best_u), ndim)
        best_logl = loglike(best_theta)
        ncall += 1
    info = _bound_seed_info(
        bound_draws,
        ncall,
        batches,
        unit_cube_survivors,
        overlap_rejections,
        jax_bound.n_active,
    )
    return new_key, best_u, best_theta, float(best_logl), ncall, False, info


def _bound_seed_info(
    bound_draws,
    ncall,
    batches,
    unit_cube_survivors,
    overlap_rejections,
    nellipsoids,
):
    return {
        "bound_seed_draws": bound_draws,
        "bound_seed_loglike_evals": ncall,
        "bound_seed_batches": batches,
        "bound_seed_unit_cube_acceptance": unit_cube_survivors / bound_draws
        if bound_draws
        else 0.0,
        "bound_seed_overlap_rejections": overlap_rejections,
        "bound_seed_nellipsoids": nellipsoids,
    }


def draw_constrained_prior(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    ndim: int,
    *,
    max_attempts: int = 10_000,
):
    """Draw a point from the prior subject to a likelihood constraint.

    Points are drawn uniformly from the unit cube, transformed through
    ``prior_transform``, and accepted once ``loglike(theta) >= logl_min``. If no
    attempted point satisfies the constraint, the best attempted point is
    returned with ``accepted`` set to ``False``.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")

    best_u = None
    best_theta = None
    best_logl = -math.inf

    new_key = key
    for ncall in range(1, max_attempts + 1):
        new_key, draw_key = random.split(new_key)
        u = random.uniform(draw_key, shape=(ndim,))
        theta = _validate_theta_shape(prior_transform(u), ndim)
        logl = float(loglike(theta))

        if best_u is None or logl > best_logl:
            best_u = u
            best_theta = theta
            best_logl = logl

        if logl >= logl_min:
            return new_key, u, theta, logl, ncall, True

    return new_key, best_u, best_theta, best_logl, max_attempts, False


def draw_constrained_prior_vectorized(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    ndim: int,
    *,
    batch_size: int = 128,
    max_attempts: int = 10_000,
):
    """Draw a constrained prior replacement using batched proposals.

    Proposals are drawn from the unit cube in batches. The first point in a
    batch with ``loglike(theta) >= logl_min`` is accepted. If no proposal is
    accepted after at least ``max_attempts`` likelihood evaluations, the best
    attempted point is returned with ``accepted`` set to ``False``.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")

    best_u = None
    best_theta = None
    best_logl = -math.inf
    ncall = 0
    new_key = key

    while ncall < max_attempts:
        new_key, draw_key = random.split(new_key)
        u_batch = random.uniform(draw_key, shape=(batch_size, ndim))
        theta_batch = jnp.asarray(prior_transform(u_batch))
        if theta_batch.shape != (batch_size, ndim):
            raise ValueError(
                f"vectorized prior_transform must return shape ({batch_size}, {ndim})"
            )
        logl_batch = jnp.asarray(loglike(theta_batch), dtype=float)
        try:
            logl_batch = logl_batch.reshape((batch_size,))
        except TypeError as exc:
            raise ValueError(
                f"vectorized loglike must return {batch_size} values"
            ) from exc

        ncall += batch_size

        batch_best_idx = int(jnp.argmax(logl_batch))
        batch_best_logl = float(logl_batch[batch_best_idx])
        if best_u is None or batch_best_logl > best_logl:
            best_u = u_batch[batch_best_idx]
            best_theta = theta_batch[batch_best_idx]
            best_logl = batch_best_logl

        accepted_mask = logl_batch >= logl_min
        if bool(jnp.any(accepted_mask)):
            accept_idx = int(jnp.argmax(accepted_mask))
            return (
                new_key,
                u_batch[accept_idx],
                theta_batch[accept_idx],
                float(logl_batch[accept_idx]),
                ncall,
                True,
            )

    return new_key, best_u, best_theta, best_logl, ncall, False


RWALK_PROPOSALS = ("isotropic", "live-cov")


def live_cov_cholesky(live_u):
    """Return the Cholesky factor of the live-point covariance (unit cube).

    A small relative jitter keeps near-degenerate live sets factorable; if the
    factorization still fails, the per-axis standard deviations are used.
    """
    live_u = jnp.asarray(live_u)
    nlive, ndim = live_u.shape
    centered = live_u - jnp.mean(live_u, axis=0)
    cov = centered.T @ centered / max(nlive - 1, 1)
    mean_var = jnp.maximum(jnp.trace(cov) / ndim, jnp.finfo(cov.dtype).tiny)
    jitter = 10.0 * jnp.finfo(cov.dtype).eps * mean_var
    cov = cov + jitter * jnp.eye(ndim, dtype=cov.dtype)
    chol = jnp.linalg.cholesky(cov)
    fallback = jnp.diag(jnp.sqrt(jnp.diag(cov)))
    return jnp.where(jnp.all(jnp.isfinite(chol)), chol, fallback)


@lru_cache(maxsize=32)
def _make_rwalk_jax_kernel_cached(
    loglike,
    prior_transform,
    ndim: int,
    walks: int,
    replacement_chains: int,
    jax_vectorized: bool,
    proposal: str = "isotropic",
):
    """Return a cached compiled retrying constrained rwalk kernel.

    This is the single internal rwalk kernel factory. It is imported directly
    by :mod:`tinyns.run` (for the per-iteration and block paths) and wrapped by
    :func:`draw_constrained_rwalk_jax`. The returned jitted ``kernel`` always
    produces an eight-element tuple, in this order:

    1. ``key`` -- the advanced PRNG key.
    2. ``new_u`` -- the accepted replacement point in unit-cube coordinates.
    3. ``new_theta`` -- the prior-transformed replacement point.
    4. ``new_logl`` -- the log-likelihood of ``new_theta``.
    5. ``ncall`` -- total likelihood evaluations spent (batches x walks x
       ``replacement_chains``).
    6. ``accepted`` -- whether an in-constraint move was found before the batch
       budget was exhausted.
    7. ``accepted_move_count`` -- accepted proposals summed over all chains and
       batches (rwalk-acceptance numerator).
    8. ``total_proposal_count`` -- total proposals attempted; equals ``ncall``
       and is the rwalk-acceptance denominator.

    ``proposal="isotropic"`` steps by ``step_scale * N(0, I)`` in the unit cube
    and reflects at the cube faces. ``proposal="live-cov"`` steps by
    ``step_scale * L N(0, I)`` with ``L`` the Cholesky factor of the live-point
    covariance, so the step follows the contracting, correlated live set; moves
    that leave the cube are rejected (reflection is not symmetric for
    correlated steps). Its chains start from live points strictly above
    ``logl_min`` whenever any exist.

    ``loglike`` and ``prior_transform`` may be JAX pytrees (for example
    ``jax.tree_util.Partial(fn, data)``). Their array leaves are then trailing
    arguments of ``kernel``: call it as ``kernel(key, ..., max_batches,
    *_callable_leaves(loglike, prior_transform))`` and the callables are rebuilt
    inside the trace, so the arrays are jit arguments rather than baked-in
    constants. Called without them, ``kernel`` closes over the callables
    (closure semantics). Plain functions and closures have no array leaves,
    take no trailing arguments and are traced exactly as before.
    """

    if proposal not in RWALK_PROPOSALS:
        raise ValueError(f"proposal must be one of {RWALK_PROPOSALS}")
    live_cov = proposal == "live-cov"
    loglike_closure = _unwrap_cacheable_callable(loglike)
    prior_closure = _unwrap_cacheable_callable(prior_transform)
    loglike_dynamic, loglike_static = _partition_callable(loglike_closure)
    prior_dynamic, prior_static = _partition_callable(prior_closure)
    nloglike_leaves = len(loglike_dynamic)
    nprior_leaves = len(prior_dynamic)
    # Only the static halves are kept; the array leaves arrive as arguments.
    del loglike_dynamic, prior_dynamic

    @jax.jit
    def kernel(
        key,
        logl_min,
        live_u,
        live_logl,
        step_scale,
        min_accepts,
        max_batches,
        *callable_leaves,
    ):
        # No trailing leaves: closure semantics (the arrays become constants).
        if callable_leaves and len(callable_leaves) != (
            nloglike_leaves + nprior_leaves
        ):
            raise ValueError(
                f"kernel expects {nloglike_leaves + nprior_leaves} callable "
                f"array leaves, got {len(callable_leaves)}"
            )
        loglike = (
            _combine_callable(callable_leaves[:nloglike_leaves], loglike_static)
            if callable_leaves and nloglike_leaves
            else loglike_closure
        )
        prior_transform = (
            _combine_callable(callable_leaves[nloglike_leaves:], prior_static)
            if callable_leaves and nprior_leaves
            else prior_closure
        )
        nlive = live_u.shape[0]
        chol = live_cov_cholesky(live_u) if live_cov else None
        if live_cov:
            # Never restart a chain from the point being replaced when others exist.
            above = live_logl > logl_min
            seed_logits = jnp.where(
                jnp.any(above), jnp.where(above, 0.0, -jnp.inf), 0.0
            )
        template_u = live_u[0]
        template_theta = _evaluate_jax_prior_batch(
            prior_transform,
            template_u[None, :],
            ndim,
            jax_vectorized=jax_vectorized,
        )[0]
        initial_best_logl = jnp.asarray(-jnp.inf, dtype=live_logl.dtype)
        initial_ncall = jnp.asarray(0, dtype=jnp.int32)
        initial_done = jnp.asarray(False)
        initial_batch_index = jnp.asarray(0, dtype=jnp.int32)
        initial_accepted_move_count = jnp.asarray(0, dtype=jnp.int32)
        batch_ncall = jnp.asarray(walks * replacement_chains, dtype=jnp.int32)

        def cond(state):
            return (~state[2]) & (state[11] < max_batches)

        def body(state):
            (
                key,
                ncall,
                _done,
                _accepted,
                out_u,
                out_theta,
                out_logl,
                best_u,
                best_theta,
                best_logl,
                accepted_move_count,
                batch_index,
            ) = state

            key, seed_key = random.split(key)
            if live_cov:
                seed_idx = random.categorical(
                    seed_key, seed_logits, shape=(replacement_chains,)
                )
            else:
                seed_idx = random.randint(
                    seed_key, shape=(replacement_chains,), minval=0, maxval=nlive
                )
            current_u = live_u[seed_idx]
            current_theta = _evaluate_jax_prior_batch(
                prior_transform,
                current_u,
                ndim,
                jax_vectorized=jax_vectorized,
            )
            current_logl = live_logl[seed_idx]
            attempt_best_u = current_u
            attempt_best_theta = current_theta
            attempt_best_logl = jnp.full(
                (replacement_chains,), -jnp.inf, live_logl.dtype
            )
            accepted_moves = jnp.zeros((replacement_chains,), dtype=jnp.int32)

            def one_step(carry, _):
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                ) = carry
                key, proposal_key = random.split(key)
                z = random.normal(proposal_key, shape=(replacement_chains, ndim))
                if live_cov:
                    u_raw = current_u + step_scale * (z @ chol.T)
                    in_cube = jnp.all((u_raw >= 0.0) & (u_raw <= 1.0), axis=1)
                    u_prop = jnp.clip(u_raw, 0.0, 1.0)
                else:
                    u_prop = reflect_unit_cube(current_u + step_scale * z)
                theta_prop, logl_prop = _evaluate_jax_batch(
                    loglike,
                    prior_transform,
                    u_prop,
                    ndim,
                    jax_vectorized=jax_vectorized,
                )

                if live_cov:
                    # Out-of-cube moves never count as the fallback best point.
                    logl_prop = jnp.where(in_cube, logl_prop, -jnp.inf)
                is_best = logl_prop > attempt_best_logl
                attempt_best_u = jnp.where(is_best[:, None], u_prop, attempt_best_u)
                attempt_best_theta = jnp.where(
                    is_best[:, None], theta_prop, attempt_best_theta
                )
                attempt_best_logl = jnp.where(is_best, logl_prop, attempt_best_logl)

                accept = logl_prop >= logl_min
                if live_cov:
                    # Out-of-cube moves are rejected even when logl_min is -inf.
                    accept = accept & in_cube
                current_u = jnp.where(accept[:, None], u_prop, current_u)
                current_theta = jnp.where(accept[:, None], theta_prop, current_theta)
                current_logl = jnp.where(accept, logl_prop, current_logl)
                accepted_moves = accepted_moves + accept.astype(jnp.int32)
                return (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                ), None

            (
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                ),
                _,
            ) = lax.scan(
                one_step,
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                ),
                xs=None,
                length=walks,
            )

            batch_best_idx = jnp.argmax(attempt_best_logl)
            batch_best_logl = attempt_best_logl[batch_best_idx]
            is_global_best = batch_best_logl > best_logl
            best_u = jnp.where(is_global_best, attempt_best_u[batch_best_idx], best_u)
            best_theta = jnp.where(
                is_global_best, attempt_best_theta[batch_best_idx], best_theta
            )
            best_logl = jnp.where(is_global_best, batch_best_logl, best_logl)

            success_mask = accepted_moves >= min_accepts
            any_success = jnp.any(success_mask)
            key, select_key = random.split(key)
            selection_scores = jnp.where(
                success_mask, random.uniform(select_key, (replacement_chains,)), -1.0
            )
            selected_idx = jnp.argmax(selection_scores)
            out_u = jnp.where(any_success, current_u[selected_idx], out_u)
            out_theta = jnp.where(any_success, current_theta[selected_idx], out_theta)
            out_logl = jnp.where(any_success, current_logl[selected_idx], out_logl)
            # dtype-pinned accumulation: under JAX_ENABLE_X64 the bare
            # jnp.sum promotes to int64 while the while_loop carry was
            # initialized int32, which aborts the kernel with a carry-type
            # mismatch (state[10] int32 vs int64).
            ncall = ncall + jnp.asarray(batch_ncall, dtype=ncall.dtype)
            accepted_move_count = accepted_move_count + jnp.sum(
                accepted_moves, dtype=accepted_move_count.dtype
            )
            batch_index = batch_index + jnp.asarray(1, dtype=jnp.int32)
            return (
                key,
                ncall,
                any_success,
                any_success,
                out_u,
                out_theta,
                out_logl,
                best_u,
                best_theta,
                best_logl,
                accepted_move_count,
                batch_index,
            )

        (
            key,
            ncall,
            done,
            accepted,
            out_u,
            out_theta,
            out_logl,
            best_u,
            best_theta,
            best_logl,
            accepted_move_count,
            _batch_index,
        ) = lax.while_loop(
            cond,
            body,
            (
                key,
                initial_ncall,
                initial_done,
                initial_done,
                template_u,
                template_theta,
                initial_best_logl,
                template_u,
                template_theta,
                initial_best_logl,
                initial_accepted_move_count,
                initial_batch_index,
            ),
        )
        new_u = jnp.where(done, out_u, best_u)
        new_theta = jnp.where(done, out_theta, best_theta)
        new_logl = jnp.where(done, out_logl, best_logl)
        return (
            key,
            new_u,
            new_theta,
            new_logl,
            ncall,
            accepted,
            accepted_move_count,
            ncall,
        )

    return kernel


def _make_rwalk_jax_kernel(
    loglike,
    prior_transform,
    ndim: int,
    walks: int,
    replacement_chains: int,
    jax_vectorized: bool,
    proposal: str = "isotropic",
):
    """Return a cached rwalk kernel for hashable or unhashable callables."""
    return _make_rwalk_jax_kernel_cached(
        _cacheable_callable(loglike),
        _cacheable_callable(prior_transform),
        ndim,
        walks,
        replacement_chains,
        jax_vectorized,
        proposal,
    )


_make_rwalk_jax_kernel.cache_clear = _make_rwalk_jax_kernel_cached.cache_clear
_make_rwalk_jax_kernel.cache_info = _make_rwalk_jax_kernel_cached.cache_info


def _validate_replacement_chain_schedule(
    replacement_chain_schedule, walks: int, max_attempts: int
) -> tuple[int, ...]:
    """Validate and normalize an adaptive replacement-chain schedule."""
    if replacement_chain_schedule is None:
        raise ValueError("replacement_chain_schedule must not be None here")
    try:
        schedule = tuple(replacement_chain_schedule)
    except TypeError as exc:
        raise ValueError(
            "replacement_chain_schedule must be a non-empty sequence "
            "of positive integers"
        ) from exc
    if not schedule:
        raise ValueError(
            "replacement_chain_schedule must be a non-empty sequence "
            "of positive integers"
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in schedule
    ):
        raise ValueError(
            "replacement_chain_schedule must be a non-empty sequence "
            "of positive integers"
        )
    if max(schedule) * int(walks) > int(max_attempts):
        raise ValueError(
            "max_attempts must be at least max(replacement_chain_schedule) * walks"
        )
    return schedule


def draw_constrained_rwalk_jax(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    live_u,
    live_logl,
    ndim: int,
    *,
    walks: int = 25,
    step_scale: float = 0.1,
    max_attempts: int = 10_000,
    min_accepts: int = 1,
    replacement_chains: int = 1,
    jax_vectorized: bool = False,
    return_info: bool = False,
    proposal: str = "isotropic",
):
    """Draw a constrained replacement with a compiled JAX rwalk kernel.

    Array leaves of pytree ``loglike`` / ``prior_transform`` callables (see
    :func:`_partition_callable`) are passed to the compiled kernel as arguments.
    ``np.ndarray`` leaves are copied to the device on every call; use
    ``jax.Array`` leaves to avoid that.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if walks <= 0:
        raise ValueError("walks must be a positive integer")
    if step_scale <= 0:
        raise ValueError("step_scale must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    _validate_min_accepts(min_accepts)
    if (
        not isinstance(replacement_chains, int)
        or isinstance(replacement_chains, bool)
        or replacement_chains <= 0
    ):
        raise ValueError("replacement_chains must be a positive integer")
    batch_ncall = int(walks) * int(replacement_chains)
    if batch_ncall > max_attempts:
        raise ValueError("max_attempts must be at least walks * replacement_chains")

    live_u = jnp.asarray(live_u)
    live_logl = jnp.asarray(live_logl)
    if live_u.ndim != 2 or live_u.shape[1] != ndim:
        raise ValueError(f"live_u must have shape (nlive, {ndim})")
    nlive = live_u.shape[0]
    if nlive <= 0:
        raise ValueError("live_u must contain at least one live point")
    if live_logl.shape != (nlive,):
        raise ValueError(f"live_logl must have shape ({nlive},)")

    kernel = _make_rwalk_jax_kernel(
        loglike,
        prior_transform,
        int(ndim),
        int(walks),
        int(replacement_chains),
        bool(jax_vectorized),
        str(proposal),
    )
    max_batches = max_attempts // batch_ncall
    (
        new_key,
        new_u,
        new_theta,
        new_logl,
        ncall,
        accepted,
        accepted_move_count,
        total_proposal_count,
    ) = kernel(
        key,
        jnp.asarray(logl_min),
        live_u,
        live_logl,
        jnp.asarray(step_scale),
        jnp.asarray(min_accepts),
        jnp.asarray(max_batches, dtype=jnp.int32),
        *_callable_leaves(loglike, prior_transform),
    )
    if return_info:
        batches = int(math.ceil(int(ncall) / batch_ncall))
        observed_acceptance = (
            float(accepted_move_count) / float(total_proposal_count)
            if int(total_proposal_count) > 0
            else 0.0
        )
        info = {
            "replacement_batches": batches,
            "replacement_chains_used": int(replacement_chains) * batches,
            "replacement_chain_usage_counts": {str(int(replacement_chains)): batches},
            "accepted_move_count": int(accepted_move_count),
            "total_proposal_count": int(total_proposal_count),
            "observed_rwalk_acceptance": observed_acceptance,
            "accepted_rwalk_moves": int(accepted_move_count),
            "total_rwalk_proposals": int(total_proposal_count),
            "rwalk_acceptance": observed_acceptance,
        }
        return (
            new_key,
            new_u,
            new_theta,
            float(new_logl),
            int(ncall),
            bool(accepted),
            info,
        )
    return new_key, new_u, new_theta, float(new_logl), int(ncall), bool(accepted)


def _draw_constrained_bound_rwalk_jax(
    key,
    loglike,
    prior_transform,
    logl_min,
    ndim: int,
    *,
    seed_draw,
    walks: int,
    step_scale: float,
    max_attempts: int,
    min_accepts: int,
    replacement_chains: int,
    replacement_chain_schedule,
    jax_vectorized: bool,
):
    """Run JAX rwalk chains from a bound-drawn seed.

    Shared implementation for the single- and multi-bound rwalk wrappers. The
    only per-variant difference is ``seed_draw``, a callable ``seed_draw(key)``
    that returns the seven-tuple produced by the corresponding bound seed draw
    (``key, seed_u, seed_theta, seed_logl, seed_ncall, seed_accepted,
    seed_info``).
    """
    seed_result = seed_draw(key)
    (
        key,
        seed_u,
        seed_theta,
        seed_logl,
        seed_ncall,
        seed_accepted,
        seed_info,
    ) = seed_result
    if not seed_accepted:
        info = {
            **seed_info,
            "rwalk_kernel_calls": 0,
            "replacement_batches": 0,
            "replacement_chains_used": 0,
            "replacement_chain_usage_counts": (
                {str(int(c)): 0 for c in replacement_chain_schedule}
                if replacement_chain_schedule is not None
                else {str(int(replacement_chains)): 0}
            ),
            "accepted_move_count": 0,
            "total_proposal_count": 0,
            "observed_rwalk_acceptance": 0.0,
            "accepted_rwalk_moves": 0,
            "total_rwalk_proposals": 0,
            "rwalk_acceptance": 0.0,
        }
        return key, seed_u, seed_theta, seed_logl, int(seed_ncall), False, info

    if replacement_chain_schedule is None:
        rwalk_result = draw_constrained_rwalk_jax(
            key,
            loglike,
            prior_transform,
            logl_min,
            jnp.asarray(seed_u).reshape((1, ndim)),
            jnp.asarray([seed_logl]),
            ndim,
            walks=walks,
            step_scale=step_scale,
            max_attempts=max_attempts,
            min_accepts=min_accepts,
            replacement_chains=replacement_chains,
            jax_vectorized=jax_vectorized,
            return_info=True,
        )
        (
            key,
            new_u,
            new_theta,
            new_logl,
            rwalk_ncall,
            accepted,
            rwalk_info,
        ) = rwalk_result
        batch_ncall = int(walks) * int(replacement_chains)
        replacement_batches = int(math.ceil(int(rwalk_ncall) / batch_ncall))
        rwalk_info = {
            "replacement_batches": replacement_batches,
            "replacement_chains_used": int(replacement_chains) * replacement_batches,
            "replacement_chain_usage_counts": {
                str(int(replacement_chains)): replacement_batches
            },
            **rwalk_info,
        }
    else:
        rwalk_result = draw_constrained_rwalk_jax_adaptive_from_seed(
            key,
            loglike,
            prior_transform,
            logl_min,
            seed_u,
            seed_logl,
            ndim,
            walks=walks,
            step_scale=step_scale,
            max_attempts=max_attempts,
            min_accepts=min_accepts,
            replacement_chain_schedule=replacement_chain_schedule,
            jax_vectorized=jax_vectorized,
        )
        (
            key,
            new_u,
            new_theta,
            new_logl,
            rwalk_ncall,
            accepted,
            rwalk_info,
        ) = rwalk_result
    info = {
        **seed_info,
        "rwalk_kernel_calls": int(rwalk_ncall),
        **rwalk_info,
    }
    return (
        key,
        new_u,
        new_theta,
        new_logl,
        int(seed_ncall) + int(rwalk_ncall),
        accepted,
        info,
    )


def draw_constrained_single_bound_rwalk_jax(
    key,
    loglike,
    prior_transform,
    logl_min,
    bound,
    ndim: int,
    *,
    walks: int = 25,
    step_scale: float = 0.1,
    max_attempts: int = 10_000,
    min_accepts: int = 1,
    replacement_chains: int = 1,
    replacement_chain_schedule=None,
    bound_batch_size: int = 128,
    bound_max_batches: int = 100,
    jax_vectorized: bool = False,
):
    """Draw a single-bound seed and run JAX rwalk chains from that seed."""

    def seed_draw(seed_key):
        return draw_constrained_single_bound_jax(
            seed_key,
            loglike,
            prior_transform,
            logl_min,
            bound,
            ndim,
            batch_size=bound_batch_size,
            max_batches=bound_max_batches,
            jax_vectorized=jax_vectorized,
        )

    return _draw_constrained_bound_rwalk_jax(
        key,
        loglike,
        prior_transform,
        logl_min,
        ndim,
        seed_draw=seed_draw,
        walks=walks,
        step_scale=step_scale,
        max_attempts=max_attempts,
        min_accepts=min_accepts,
        replacement_chains=replacement_chains,
        replacement_chain_schedule=replacement_chain_schedule,
        jax_vectorized=jax_vectorized,
    )


def draw_constrained_multi_bound_rwalk_jax(
    key,
    loglike,
    prior_transform,
    logl_min,
    bound,
    ndim: int,
    *,
    walks: int = 25,
    step_scale: float = 0.1,
    max_attempts: int = 10_000,
    min_accepts: int = 1,
    replacement_chains: int = 1,
    replacement_chain_schedule=None,
    bound_batch_size: int = 128,
    bound_max_batches: int = 100,
    overlap_correction: bool = True,
    jax_vectorized: bool = False,
):
    """Draw a multi-bound seed and run JAX rwalk chains from that seed.

    The multi-ellipsoid bound is used only for the constrained seed draw. The
    subsequent rwalk proposal remains the standard JAX rwalk kernel, optionally
    with a caller-provided proposal covariance Cholesky factor.
    """
    jax_bound = as_jax_ellipsoid_bound(bound)

    def seed_draw(seed_key):
        return draw_constrained_multi_bound_jax(
            seed_key,
            loglike,
            prior_transform,
            logl_min,
            jax_bound,
            ndim,
            batch_size=bound_batch_size,
            max_batches=bound_max_batches,
            overlap_correction=overlap_correction,
            jax_vectorized=jax_vectorized,
        )

    return _draw_constrained_bound_rwalk_jax(
        key,
        loglike,
        prior_transform,
        logl_min,
        ndim,
        seed_draw=seed_draw,
        walks=walks,
        step_scale=step_scale,
        max_attempts=max_attempts,
        min_accepts=min_accepts,
        replacement_chains=replacement_chains,
        replacement_chain_schedule=replacement_chain_schedule,
        jax_vectorized=jax_vectorized,
    )

def draw_constrained_rwalk_jax_adaptive(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    live_u,
    live_logl,
    ndim: int,
    *,
    walks: int = 25,
    step_scale: float = 0.1,
    max_attempts: int = 10_000,
    min_accepts: int = 1,
    replacement_chain_schedule=(1, 4, 16, 64),
    jax_vectorized: bool = False,
):
    """Draw a constrained JAX rwalk replacement with adaptive batch retries."""
    schedule = _validate_replacement_chain_schedule(
        replacement_chain_schedule, walks, max_attempts
    )
    total_ncall = 0
    accepted_move_count = 0
    total_proposal_count = 0
    best_u = None
    best_theta = None
    best_logl = -float("inf")
    batches = 0
    chains_used = 0
    usage_counts = {int(c): 0 for c in schedule}

    def try_batch(key, c):
        return draw_constrained_rwalk_jax(
            key,
            loglike,
            prior_transform,
            logl_min,
            live_u,
            live_logl,
            ndim,
            walks=walks,
            step_scale=step_scale,
            max_attempts=int(walks) * int(c),
            min_accepts=min_accepts,
            replacement_chains=int(c),
            jax_vectorized=jax_vectorized,
            return_info=True,
        )

    c = schedule[-1]
    stage_index = 0
    while True:
        if stage_index < len(schedule):
            c = schedule[stage_index]
            stage_index += 1
        batch_budget = int(walks) * int(c)
        if total_ncall + batch_budget > int(max_attempts):
            break
        (
            key,
            candidate_u,
            candidate_theta,
            candidate_logl,
            ncall,
            accepted,
            batch_info,
        ) = try_batch(key, c)
        total_ncall += int(ncall)
        accepted_move_count += int(batch_info["accepted_move_count"])
        total_proposal_count += int(batch_info["total_proposal_count"])
        batches += 1
        chains_used += int(c)
        usage_counts[int(c)] = usage_counts.get(int(c), 0) + 1
        if float(candidate_logl) > best_logl or best_u is None:
            best_u = candidate_u
            best_theta = candidate_theta
            best_logl = float(candidate_logl)
        observed_acceptance = (
            float(accepted_move_count) / float(total_proposal_count)
            if total_proposal_count > 0
            else 0.0
        )
        info = {
            "replacement_batches": batches,
            "replacement_chains_used": chains_used,
            "replacement_last_chain_count": int(c),
            "replacement_chain_usage_counts": {
                str(k): v for k, v in usage_counts.items()
            },
            "accepted_move_count": int(accepted_move_count),
            "total_proposal_count": int(total_proposal_count),
            "observed_rwalk_acceptance": observed_acceptance,
            "accepted_rwalk_moves": int(accepted_move_count),
            "total_rwalk_proposals": int(total_proposal_count),
            "rwalk_acceptance": observed_acceptance,
        }
        if accepted:
            return (
                key,
                candidate_u,
                candidate_theta,
                candidate_logl,
                total_ncall,
                True,
                info,
            )

    if best_u is None:
        raise RuntimeError("adaptive replacement schedule made no attempts")
    observed_acceptance = (
        float(accepted_move_count) / float(total_proposal_count)
        if total_proposal_count > 0
        else 0.0
    )
    info = {
        "replacement_batches": batches,
        "replacement_chains_used": chains_used,
        "replacement_last_chain_count": int(c),
        "replacement_chain_usage_counts": {str(k): v for k, v in usage_counts.items()},
        "accepted_move_count": int(accepted_move_count),
        "total_proposal_count": int(total_proposal_count),
        "observed_rwalk_acceptance": observed_acceptance,
        "accepted_rwalk_moves": int(accepted_move_count),
        "total_rwalk_proposals": int(total_proposal_count),
        "rwalk_acceptance": observed_acceptance,
    }
    return key, best_u, best_theta, best_logl, total_ncall, False, info



def draw_constrained_rwalk_jax_adaptive_from_seed(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    seed_u,
    seed_logl,
    ndim: int,
    *,
    walks: int = 25,
    step_scale: float = 0.1,
    max_attempts: int = 10_000,
    min_accepts: int = 1,
    replacement_chain_schedule=(1, 4, 16, 64),
    jax_vectorized: bool = False,
):
    """Run adaptive JAX rwalk replacement batches from one fixed seed."""
    seed_u = jnp.asarray(seed_u)
    if seed_u.shape != (ndim,):
        raise ValueError(f"seed_u must have shape ({ndim},)")
    live_u = seed_u.reshape((1, ndim))
    live_logl = jnp.asarray([seed_logl])
    return draw_constrained_rwalk_jax_adaptive(
        key,
        loglike,
        prior_transform,
        logl_min,
        live_u,
        live_logl,
        ndim,
        walks=walks,
        step_scale=step_scale,
        max_attempts=max_attempts,
        min_accepts=min_accepts,
        replacement_chain_schedule=replacement_chain_schedule,
        jax_vectorized=jax_vectorized,
    )

def draw_constrained_rwalk(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    live_u,
    live_logl,
    ndim: int,
    *,
    walks: int = 25,
    step_scale: float = 0.1,
    max_attempts: int = 10_000,
    min_accepts: int = 1,
):
    """Draw a constrained replacement with a reflected random walk.

    A live point is chosen as the seed, then ``walks`` reflected Gaussian
    transition proposals are attempted per replacement attempt. The copied live
    seed does not count toward ``min_accepts``; ``min_accepts`` is a minimum
    accepted-move sanity check after the full update length, not a chain length.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if walks <= 0:
        raise ValueError("walks must be a positive integer")
    if step_scale <= 0:
        raise ValueError("step_scale must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    _validate_min_accepts(min_accepts)

    live_u = jnp.asarray(live_u)
    live_logl = jnp.asarray(live_logl)
    if live_u.ndim != 2 or live_u.shape[1] != ndim:
        raise ValueError(f"live_u must have shape (nlive, {ndim})")
    nlive = live_u.shape[0]
    if nlive <= 0:
        raise ValueError("live_u must contain at least one live point")
    if live_logl.shape != (nlive,):
        raise ValueError(f"live_logl must have shape ({nlive},)")

    best_u = None
    best_theta = None
    best_logl = -math.inf
    ncall = 0
    new_key = key

    while ncall < max_attempts:
        new_key, seed_key = random.split(new_key)
        seed_idx = int(random.randint(seed_key, shape=(), minval=0, maxval=nlive))
        current_u = live_u[seed_idx]
        current_theta = _validate_theta_shape(prior_transform(current_u), ndim)
        current_logl = float(live_logl[seed_idx])
        accepted_moves = 0

        for _ in range(walks):
            if ncall >= max_attempts:
                break
            new_key, proposal_key = random.split(new_key)
            step = step_scale * random.normal(proposal_key, shape=(ndim,))
            u_prop = reflect_unit_cube(current_u + step)
            theta_prop = _validate_theta_shape(prior_transform(u_prop), ndim)
            logl_prop = float(loglike(theta_prop))
            ncall += 1

            if best_u is None or logl_prop > best_logl:
                best_u = u_prop
                best_theta = theta_prop
                best_logl = logl_prop

            if logl_prop >= logl_min:
                current_u = u_prop
                current_theta = theta_prop
                current_logl = logl_prop
                accepted_moves += 1

        if accepted_moves >= min_accepts:
            return new_key, current_u, current_theta, current_logl, ncall, True

    return new_key, best_u, best_theta, best_logl, ncall, False
