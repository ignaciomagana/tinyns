from __future__ import annotations

import math

import jax.numpy as jnp
from jax import random

from tinyns.run import run_static_nested


def test_constant_likelihood_unit_cube_logz_close_to_zero() -> None:
    result = run_static_nested(
        random.PRNGKey(0),
        lambda theta: 0.0,
        lambda u: u,
        ndim=2,
        nlive=50,
        dlogz=0.01,
        maxiter=500,
    )

    assert abs(result.logz) < 0.05
    assert jnp.isfinite(result.logz)
    assert result.samples.shape[1:] == (2,)


def test_gaussian_likelihood_uniform_prior_logz_close_to_inverse_width() -> None:
    def loglike(theta):
        return float(-0.5 * theta[0] ** 2 - 0.5 * math.log(2.0 * math.pi))

    def prior_transform(u):
        return 20.0 * u - 10.0

    result = run_static_nested(
        random.PRNGKey(1),
        loglike,
        prior_transform,
        ndim=1,
        nlive=100,
        dlogz=0.05,
        maxiter=2_000,
    )

    assert abs(result.logz - (-math.log(20.0))) < 0.5
    assert jnp.isfinite(result.logz)


def test_result_shapes_finite_logz_and_equal_resampling() -> None:
    result = run_static_nested(
        random.PRNGKey(2),
        lambda theta: float(-jnp.sum(theta**2)),
        lambda u: 2.0 * u - 1.0,
        ndim=3,
        nlive=40,
        dlogz=0.1,
        maxiter=500,
    )

    assert result.samples.shape == result.samples_u.shape
    assert result.samples.shape[1:] == (3,)
    assert result.logl.shape == (result.samples.shape[0],)
    assert result.logwt.shape == (result.samples.shape[0],)
    assert jnp.isfinite(result.logz)
    assert result.resample_equal(random.PRNGKey(3), n=10).shape == (10, 3)


def test_failure_to_replace_returns_result_with_live_contribution() -> None:
    result = run_static_nested(
        random.PRNGKey(4),
        lambda theta: float(theta[0]),
        lambda u: u,
        ndim=1,
        nlive=3,
        dlogz=0.0,
        maxiter=10,
        max_attempts=1,
    )

    assert result.success is False
    assert "max_attempts=1" in result.message
    assert result.ncall > result.nlive
    assert result.samples.shape[1:] == (1,)
    assert result.logwt.shape == (result.samples.shape[0],)
    assert jnp.isfinite(result.logz)


def test_scalar_prior_transform_for_one_dimension_keeps_matrix_shape() -> None:
    result = run_static_nested(
        random.PRNGKey(5),
        lambda theta: float(-(theta[0] ** 2)),
        lambda u: u[0],
        ndim=1,
        nlive=5,
        maxiter=0,
    )

    assert result.samples_u.shape == (5, 1)
    assert result.samples.shape == (5, 1)
    assert result.logl.shape == (5,)
    assert result.logwt.shape == (5,)
