from __future__ import annotations

import math

import jax.numpy as jnp
import pytest
from jax import random

from tinyns.samplers import draw_constrained_prior, draw_constrained_rwalk


def gaussian_loglike(theta):
    return -0.5 * jnp.sum(theta**2)


def identity_prior_transform(u):
    return u


def test_draw_constrained_prior_accepts_immediately_with_unbounded_threshold() -> None:
    ndim = 3

    _, u, theta, logl, ncall, accepted = draw_constrained_prior(
        random.PRNGKey(0),
        gaussian_loglike,
        identity_prior_transform,
        -math.inf,
        ndim,
    )

    assert accepted is True
    assert u.shape == (ndim,)
    assert theta.shape == (ndim,)
    assert math.isfinite(logl)
    assert ncall > 0
    assert ncall == 1


def test_draw_constrained_prior_returns_best_after_impossible_threshold() -> None:
    max_attempts = 5

    _, u, theta, logl, ncall, accepted = draw_constrained_prior(
        random.PRNGKey(1),
        gaussian_loglike,
        identity_prior_transform,
        math.inf,
        2,
        max_attempts=max_attempts,
    )

    assert accepted is False
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert math.isfinite(logl)
    assert ncall == max_attempts


def test_draw_constrained_prior_rejects_vectorized_mode() -> None:
    with pytest.raises(NotImplementedError, match="vectorized=False only"):
        draw_constrained_prior(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            2,
            vectorized=True,
        )


def test_draw_constrained_rwalk_accepts_loose_threshold() -> None:
    ndim = 3
    live_u = jnp.array(
        [
            [0.2, 0.3, 0.4],
            [0.4, 0.5, 0.6],
            [0.6, 0.7, 0.8],
        ]
    )
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])
    logl_min = -10.0

    _, u, theta, logl, ncall, accepted = draw_constrained_rwalk(
        random.PRNGKey(2),
        gaussian_loglike,
        identity_prior_transform,
        logl_min,
        live_u,
        live_logl,
        ndim,
        walks=10,
        step_scale=0.05,
        max_attempts=20,
    )

    assert accepted is True
    assert u.shape == (ndim,)
    assert theta.shape == (ndim,)
    assert logl >= logl_min
    assert ncall > 0


def test_draw_constrained_rwalk_returns_best_after_impossible_threshold() -> None:
    ndim = 2
    live_u = jnp.array([[0.25, 0.25], [0.75, 0.75]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])
    max_attempts = 4
    walks = 3

    _, u, theta, logl, ncall, accepted = draw_constrained_rwalk(
        random.PRNGKey(3),
        gaussian_loglike,
        identity_prior_transform,
        math.inf,
        live_u,
        live_logl,
        ndim,
        walks=walks,
        step_scale=0.1,
        max_attempts=max_attempts,
    )

    assert accepted is False
    assert u.shape == (ndim,)
    assert theta.shape == (ndim,)
    assert math.isfinite(logl)
    assert ncall == max_attempts * walks


def test_draw_constrained_rwalk_rejects_invalid_parameters_and_shapes() -> None:
    live_u = jnp.ones((3, 2)) * 0.5
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    with pytest.raises(ValueError, match="walks"):
        draw_constrained_rwalk(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            live_logl,
            2,
            walks=0,
        )

    with pytest.raises(ValueError, match="step_scale"):
        draw_constrained_rwalk(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            live_logl,
            2,
            step_scale=0.0,
        )

    with pytest.raises(ValueError, match="live_u"):
        draw_constrained_rwalk(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            jnp.ones((3, 3)),
            live_logl,
            2,
        )

    with pytest.raises(ValueError, match="live_logl"):
        draw_constrained_rwalk(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            jnp.ones((2,)),
            2,
        )
