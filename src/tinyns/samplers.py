"""Replacement samplers for :mod:`tinyns`."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import random

from tinyns.math import reflect_unit_cube
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike


def _validate_theta_shape(theta, ndim: int):
    theta = jnp.asarray(theta)
    if ndim == 1 and theta.shape == ():
        theta = theta.reshape((1,))
    if theta.shape != (ndim,):
        raise ValueError(f"prior_transform must return shape ({ndim},)")
    return theta


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
                "vectorized prior_transform must return shape "
                f"({batch_size}, {ndim})"
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
):
    """Draw a constrained replacement with coordinate-wise slice updates.

    A live point is chosen as the seed. The copied seed does not count as a
    successful replacement; at least one coordinate proposal must satisfy the
    likelihood constraint. Proposals are reflected into the unit cube before
    evaluating ``prior_transform`` and ``loglike``.
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

    live_u = jnp.asarray(live_u)
    live_logl = jnp.asarray(live_logl)
    if live_u.ndim != 2 or live_u.shape[1] != ndim:
        raise ValueError(f"live_u must have shape (nlive, {ndim})")
    nlive = live_u.shape[0]
    if nlive <= 0:
        raise ValueError("live_u must contain at least one live point")
    if live_logl.shape != (nlive,):
        raise ValueError(f"live_logl must have shape ({nlive},)")

    new_key, seed_key = random.split(key)
    seed_idx = int(random.randint(seed_key, shape=(), minval=0, maxval=nlive))
    current_u = live_u[seed_idx]
    current_theta = _validate_theta_shape(prior_transform(current_u), ndim)
    current_logl = float(live_logl[seed_idx])

    best_u = None
    best_theta = None
    best_logl = -math.inf
    ncall = 0

    for _ in range(max_attempts):
        moved = False
        for _ in range(slices):
            new_key, axis_key, bracket_key = random.split(new_key, 3)
            axis = int(random.randint(axis_key, shape=(), minval=0, maxval=ndim))
            x = current_u[axis]
            r = random.uniform(bracket_key, shape=())
            left = x - r * step_scale
            right = left + step_scale

            for _ in range(slice_steps):
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
                    moved = True
                    break

                if bool(x_prop < x):
                    left = x_prop
                else:
                    right = x_prop

        if moved:
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
):
    """Draw a constrained replacement with random-direction slice updates.

    A live point is chosen as the seed. The copied seed does not count as a
    successful replacement; at least one random-direction proposal must satisfy
    the likelihood constraint. Proposals are reflected into the unit cube before
    evaluating ``prior_transform`` and ``loglike``.
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

    live_u = jnp.asarray(live_u)
    live_logl = jnp.asarray(live_logl)
    if live_u.ndim != 2 or live_u.shape[1] != ndim:
        raise ValueError(f"live_u must have shape (nlive, {ndim})")
    nlive = live_u.shape[0]
    if nlive <= 0:
        raise ValueError("live_u must contain at least one live point")
    if live_logl.shape != (nlive,):
        raise ValueError(f"live_logl must have shape ({nlive},)")

    new_key, seed_key = random.split(key)
    seed_idx = int(random.randint(seed_key, shape=(), minval=0, maxval=nlive))
    current_u = live_u[seed_idx]
    current_theta = _validate_theta_shape(prior_transform(current_u), ndim)
    current_logl = float(live_logl[seed_idx])

    best_u = current_u
    best_theta = current_theta
    best_logl = current_logl
    ncall = 0

    for _ in range(max_attempts):
        moved = False
        for _ in range(slices):
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
                    moved = True
                    break

                if bool(alpha < 0):
                    left = alpha
                else:
                    right = alpha

        if moved:
            return new_key, current_u, current_theta, current_logl, ncall, True

    return new_key, best_u, best_theta, best_logl, ncall, False


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
):
    """Draw a constrained replacement with a reflected random walk.

    A live point is chosen as the seed, then Gaussian proposals are reflected
    into the unit cube. The copied seed does not count as an accepted
    replacement; at least one proposal must satisfy the likelihood constraint.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if walks <= 0:
        raise ValueError("walks must be a positive integer")
    if step_scale <= 0:
        raise ValueError("step_scale must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")

    live_u = jnp.asarray(live_u)
    live_logl = jnp.asarray(live_logl)
    if live_u.ndim != 2 or live_u.shape[1] != ndim:
        raise ValueError(f"live_u must have shape (nlive, {ndim})")
    nlive = live_u.shape[0]
    if nlive <= 0:
        raise ValueError("live_u must contain at least one live point")
    if live_logl.shape != (nlive,):
        raise ValueError(f"live_logl must have shape ({nlive},)")

    new_key, seed_key = random.split(key)
    seed_idx = int(random.randint(seed_key, shape=(), minval=0, maxval=nlive))
    current_u = live_u[seed_idx]
    current_theta = _validate_theta_shape(prior_transform(current_u), ndim)
    current_logl = float(live_logl[seed_idx])

    best_u = None
    best_theta = None
    best_logl = -math.inf
    ncall = 0

    for _ in range(max_attempts):
        moved = False
        for _ in range(walks):
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
                moved = True

        if moved:
            return new_key, current_u, current_theta, current_logl, ncall, True

    return new_key, best_u, best_theta, best_logl, ncall, False
