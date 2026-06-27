"""Replacement samplers for :mod:`tinyns`."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import random

from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike


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
        theta = jnp.asarray(prior_transform(u))
        logl = float(loglike(theta))

        if best_u is None or logl > best_logl:
            best_u = u
            best_theta = theta
            best_logl = logl

        if logl >= logl_min:
            return new_key, u, theta, logl, ncall, True

    return new_key, best_u, best_theta, best_logl, max_attempts, False
