from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from validation.targets import (
    available_targets,
    gaussian_1d,
    get_target,
    heavy_gaussian_2d,
)


def test_available_targets_contains_expected_names() -> None:
    assert set(available_targets()) >= {
        "constant2d",
        "gaussian1d",
        "gaussian2d",
        "correlated_gaussian2d",
        "banana2d",
        "ring2d",
        "eggbox2d",
        "heavy_gaussian2d",
    }


def test_get_target_gaussian2d_has_expected_metadata() -> None:
    target = get_target("gaussian2d")
    assert target.ndim == 2
    assert target.expected_logz is not None
    assert np.isfinite(target.expected_logz)


def test_gaussian_1d_prior_transform_returns_vector_shape() -> None:
    target = gaussian_1d()
    theta = np.asarray(target.prior_transform(np.array([0.5])))
    assert theta.shape == (1,)


def test_gaussian_2d_loglike_is_finite() -> None:
    target = get_target("gaussian2d")
    assert np.isfinite(float(target.loglike(np.array([0.0, 0.0]))))


def test_correlated_gaussian_covariance_has_expected_off_diagonal() -> None:
    target = get_target("correlated_gaussian2d")
    assert target.expected_cov is not None
    assert target.expected_cov[0, 1] == pytest.approx(0.8)
    assert target.expected_cov[1, 0] == pytest.approx(0.8)


def test_ring_2d_target_has_expected_geometry() -> None:
    target = get_target("ring2d")

    assert target.ndim == 2
    np.testing.assert_allclose(
        np.asarray(target.prior_transform(jnp.array([0.5, 0.5]))),
        np.array([0.0, 0.0]),
    )

    loglike_on_ring = float(target.loglike(jnp.array([1.0, 0.0])))
    loglike_at_center = float(target.loglike(jnp.array([0.0, 0.0])))

    assert loglike_on_ring == pytest.approx(0.0)
    assert loglike_at_center < loglike_on_ring


def test_heavy_gaussian_2d_constructs_with_finite_scalar_loglike() -> None:
    target = heavy_gaussian_2d(work_size=100)

    theta = target.prior_transform(jnp.array([0.5, 0.5]))
    loglike = target.loglike(jnp.zeros(2))

    assert target.name == "heavy_gaussian2d"
    assert target.ndim == 2
    assert np.asarray(theta).shape == (2,)
    assert np.asarray(loglike).shape == ()
    assert np.isfinite(float(loglike))


def test_unknown_target_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_target("does-not-exist")
