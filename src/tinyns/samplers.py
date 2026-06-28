"""Replacement samplers for :mod:`tinyns`."""

from __future__ import annotations

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import lax, random

from tinyns.bounds import (
    MultiEllipsoidBound,
    in_unit_cube,
    sample_multi_ellipsoid,
    sample_multi_ellipsoid_corrected,
    sample_single_ellipsoid,
)
from tinyns.math import reflect_unit_cube
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike


def _validate_theta_shape(theta, ndim: int):
    theta = jnp.asarray(theta)
    if ndim == 1 and theta.shape == ():
        theta = theta.reshape((1,))
    if theta.shape != (ndim,):
        raise ValueError(f"prior_transform must return shape ({ndim},)")
    return theta


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
        if isinstance(bound, MultiEllipsoidBound):
            if overlap_correction:
                u_batch, _idx, draws, overlap = sample_multi_ellipsoid_corrected(
                    draw_key, bound, batch_size
                )
                bound_draws += int(draws)
                overlap_rejections += int(overlap)
            else:
                u_batch, _idx = sample_multi_ellipsoid(draw_key, bound, batch_size)
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


def draw_constrained_prior(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    ndim: int,
    *,
    vectorized: bool = False,
    max_attempts: int = 10_000,
):
    """Draw a point from the prior subject to a likelihood constraint.

    Points are drawn uniformly from the unit cube, transformed through
    ``prior_transform``, and accepted once ``loglike(theta) >= logl_min``. If no
    attempted point satisfies the constraint, the best attempted point is
    returned with ``accepted`` set to ``False``.
    """
    if vectorized:
        raise NotImplementedError(
            "draw_constrained_prior currently supports vectorized=False only"
        )
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


def draw_constrained_slice(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    live_u,
    live_logl,
    ndim: int,
    *,
    slices: int = 5,
    slice_steps: int = 10,
    step_scale: float = 0.1,
    max_attempts: int = 10_000,
    min_accepts: int = 1,
):
    """Draw a constrained replacement with coordinate-wise slice updates.

    A live point is chosen as the seed. ``slices`` is the number of coordinate
    update attempts per replacement attempt. The copied live seed does not count
    toward ``min_accepts``; ``min_accepts`` is a minimum accepted-move sanity
    check after the full update length, not a chain length. Proposals are
    reflected into the unit cube before evaluating ``prior_transform`` and
    ``loglike``.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if slices <= 0:
        raise ValueError("slices must be a positive integer")
    if slice_steps <= 0:
        raise ValueError("slice_steps must be a positive integer")
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

        for _ in range(slices):
            if ncall >= max_attempts:
                break
            new_key, axis_key, bracket_key = random.split(new_key, 3)
            axis = int(random.randint(axis_key, shape=(), minval=0, maxval=ndim))
            x = current_u[axis]
            r = random.uniform(bracket_key, shape=())
            left = x - r * step_scale
            right = left + step_scale

            for _ in range(slice_steps):
                if ncall >= max_attempts:
                    break
                new_key, proposal_key = random.split(new_key)
                x_prop = left + random.uniform(proposal_key, shape=()) * (right - left)
                u_prop = reflect_unit_cube(current_u.at[axis].set(x_prop))
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
                    # Continue with the next coordinate update after a
                    # successful constrained move.
                    break

                if bool(x_prop < x):
                    left = x_prop
                else:
                    right = x_prop

        if accepted_moves >= min_accepts:
            return new_key, current_u, current_theta, current_logl, ncall, True

    return new_key, best_u, best_theta, best_logl, ncall, False


def draw_constrained_rslice(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    logl_min: float,
    live_u,
    live_logl,
    ndim: int,
    *,
    slices: int = 5,
    slice_steps: int = 10,
    step_scale: float = 0.1,
    max_attempts: int = 10_000,
    min_accepts: int = 1,
):
    """Draw a constrained replacement with random-direction slice updates.

    A live point is chosen as the seed. ``slices`` is the number of
    random-direction update attempts per replacement attempt. The copied live
    seed does not count toward ``min_accepts``; ``min_accepts`` is a minimum
    accepted-move sanity check after the full update length, not a chain length.
    Proposals are reflected into the unit cube before evaluating
    ``prior_transform`` and ``loglike``.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if slices <= 0:
        raise ValueError("slices must be a positive integer")
    if slice_steps <= 0:
        raise ValueError("slice_steps must be a positive integer")
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

        for _ in range(slices):
            if ncall >= max_attempts:
                break
            new_key, direction_key, bracket_key = random.split(new_key, 3)
            direction = random.normal(direction_key, shape=(ndim,))
            norm = float(jnp.linalg.norm(direction))
            if not math.isfinite(norm) or norm <= 0.0:
                continue
            direction = direction / norm

            r = random.uniform(bracket_key, shape=())
            left = -r * step_scale
            right = left + step_scale

            for _ in range(slice_steps):
                if ncall >= max_attempts:
                    break
                new_key, proposal_key = random.split(new_key)
                alpha = left + random.uniform(proposal_key, shape=()) * (right - left)
                u_prop = reflect_unit_cube(current_u + alpha * direction)
                theta_prop = _validate_theta_shape(prior_transform(u_prop), ndim)
                logl_prop = float(loglike(theta_prop))
                ncall += 1

                if logl_prop > best_logl:
                    best_u = u_prop
                    best_theta = theta_prop
                    best_logl = logl_prop

                if logl_prop >= logl_min:
                    current_u = u_prop
                    current_theta = theta_prop
                    current_logl = logl_prop
                    accepted_moves += 1
                    # Continue with the next random-direction update after a
                    # successful constrained move.
                    break

                if bool(alpha < 0):
                    left = alpha
                else:
                    right = alpha

        if accepted_moves >= min_accepts:
            return new_key, current_u, current_theta, current_logl, ncall, True

    return new_key, best_u, best_theta, best_logl, ncall, False


@lru_cache(maxsize=32)
def _make_rwalk_jax_kernel(
    loglike, prior_transform, ndim: int, walks: int, replacement_chains: int
):
    """Return a cached compiled retrying constrained rwalk kernel."""

    @jax.jit
    def kernel(
        key,
        logl_min,
        live_u,
        live_logl,
        step_scale,
        min_accepts,
        max_batches,
        proposal_chol,
    ):
        nlive = live_u.shape[0]
        template_u = live_u[0]
        template_theta = jnp.asarray(prior_transform(template_u))
        initial_best_logl = jnp.asarray(-jnp.inf, dtype=live_logl.dtype)
        initial_ncall = jnp.asarray(0, dtype=jnp.int32)
        initial_done = jnp.asarray(False)
        initial_batch_index = jnp.asarray(0, dtype=jnp.int32)
        batch_ncall = jnp.asarray(walks * replacement_chains, dtype=jnp.int32)

        def cond(state):
            return (~state[2]) & (state[10] < max_batches)

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
                batch_index,
            ) = state

            key, seed_key = random.split(key)
            seed_idx = random.randint(
                seed_key, shape=(replacement_chains,), minval=0, maxval=nlive
            )
            current_u = live_u[seed_idx]
            current_theta = jax.vmap(prior_transform)(current_u)
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
                step = step_scale * (z @ proposal_chol.T)
                u_prop = reflect_unit_cube(current_u + step)
                theta_prop = jax.vmap(prior_transform)(u_prop)
                logl_prop = jax.vmap(loglike)(theta_prop)

                is_best = logl_prop > attempt_best_logl
                attempt_best_u = jnp.where(is_best[:, None], u_prop, attempt_best_u)
                attempt_best_theta = jnp.where(
                    is_best[:, None], theta_prop, attempt_best_theta
                )
                attempt_best_logl = jnp.where(is_best, logl_prop, attempt_best_logl)

                accept = logl_prop >= logl_min
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
            ncall = ncall + batch_ncall
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
                initial_batch_index,
            ),
        )
        new_u = jnp.where(done, out_u, best_u)
        new_theta = jnp.where(done, out_theta, best_theta)
        new_logl = jnp.where(done, out_logl, best_logl)
        return key, new_u, new_theta, new_logl, ncall, accepted

    return kernel


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
    proposal_chol=None,
):
    """Draw a constrained replacement with a compiled JAX rwalk kernel."""
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
    if proposal_chol is None:
        proposal_chol = jnp.eye(ndim, dtype=live_u.dtype)
    else:
        proposal_chol = jnp.asarray(proposal_chol, dtype=live_u.dtype)
        if proposal_chol.shape != (ndim, ndim):
            raise ValueError(f"proposal_chol must have shape ({ndim}, {ndim})")

    kernel = _make_rwalk_jax_kernel(
        loglike, prior_transform, int(ndim), int(walks), int(replacement_chains)
    )
    max_batches = max_attempts // batch_ncall
    new_key, new_u, new_theta, new_logl, ncall, accepted = kernel(
        key,
        jnp.asarray(logl_min),
        live_u,
        live_logl,
        jnp.asarray(step_scale),
        jnp.asarray(min_accepts),
        jnp.asarray(max_batches, dtype=jnp.int32),
        proposal_chol,
    )
    return new_key, new_u, new_theta, float(new_logl), int(ncall), bool(accepted)


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
    proposal_chol=None,
):
    """Draw a constrained JAX rwalk replacement with adaptive batch retries."""
    schedule = _validate_replacement_chain_schedule(
        replacement_chain_schedule, walks, max_attempts
    )
    total_ncall = 0
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
            proposal_chol=proposal_chol,
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
        key, candidate_u, candidate_theta, candidate_logl, ncall, accepted = try_batch(
            key, c
        )
        total_ncall += int(ncall)
        batches += 1
        chains_used += int(c)
        usage_counts[int(c)] = usage_counts.get(int(c), 0) + 1
        if float(candidate_logl) > best_logl or best_u is None:
            best_u = candidate_u
            best_theta = candidate_theta
            best_logl = float(candidate_logl)
        info = {
            "replacement_batches": batches,
            "replacement_chains_used": chains_used,
            "replacement_last_chain_count": int(c),
            "replacement_chain_usage_counts": {
                str(k): v for k, v in usage_counts.items()
            },
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
    info = {
        "replacement_batches": batches,
        "replacement_chains_used": chains_used,
        "replacement_last_chain_count": int(c),
        "replacement_chain_usage_counts": {str(k): v for k, v in usage_counts.items()},
    }
    return key, best_u, best_theta, best_logl, total_ncall, False, info


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
