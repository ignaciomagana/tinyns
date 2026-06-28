from __future__ import annotations

import math

import jax.numpy as jnp
import pytest
from jax import random

from tinyns.bounds import (
    build_single_ellipsoid_bound,
    contains_single_ellipsoid,
    in_unit_cube,
    sample_single_ellipsoid,
    unit_ball_log_volume,
)


def _live_points(nlive: int = 64, ndim: int = 3):
    key = random.PRNGKey(0)
    return 0.5 + 0.1 * random.normal(key, shape=(nlive, ndim))


def test_build_single_ellipsoid_bound_shapes() -> None:
    live_u = _live_points(32, 4)

    bound = build_single_ellipsoid_bound(live_u)

    assert bound.center.shape == (4,)
    assert bound.chol.shape == (4, 4)
    assert bound.inv_chol.shape == (4, 4)
    assert bound.ndim == 4
    assert bound.enlargement == pytest.approx(1.25)
    assert math.isfinite(bound.log_volume)


def test_build_single_ellipsoid_bound_contains_live_points() -> None:
    live_u = _live_points()

    bound = build_single_ellipsoid_bound(live_u)

    assert bool(jnp.all(contains_single_ellipsoid(bound, live_u)))


def test_contains_single_ellipsoid_accepts_single_point_and_batch() -> None:
    live_u = _live_points()
    bound = build_single_ellipsoid_bound(live_u)

    single = contains_single_ellipsoid(bound, live_u[0])
    batch = contains_single_ellipsoid(bound, live_u[:3])

    assert single.shape == ()
    assert bool(single)
    assert batch.shape == (3,)
    assert bool(jnp.all(batch))


def test_sample_single_ellipsoid_shape() -> None:
    bound = build_single_ellipsoid_bound(_live_points())

    samples = sample_single_ellipsoid(random.PRNGKey(1), bound, 11)

    assert samples.shape == (11, 3)


def test_sample_single_ellipsoid_points_are_inside_bound() -> None:
    bound = build_single_ellipsoid_bound(_live_points())

    samples = sample_single_ellipsoid(random.PRNGKey(2), bound, 50)
    inside = contains_single_ellipsoid(bound, samples)

    assert float(jnp.mean(inside)) > 0.95


def test_in_unit_cube() -> None:
    assert bool(in_unit_cube(jnp.array([0.0, 0.5, 1.0])))
    assert not bool(in_unit_cube(jnp.array([-0.1, 0.5, 1.0])))
    mask = in_unit_cube(jnp.array([[0.1, 0.9], [1.1, 0.5]]))
    assert mask.tolist() == [True, False]


def test_invalid_live_array_shapes_raise_value_error() -> None:
    with pytest.raises(ValueError, match="live_u"):
        build_single_ellipsoid_bound(jnp.ones(3))
    with pytest.raises(ValueError, match="positive sizes"):
        build_single_ellipsoid_bound(jnp.ones((0, 2)))


def test_invalid_enlargement_or_jitter_raise_value_error() -> None:
    live_u = _live_points()
    with pytest.raises(ValueError, match="enlargement"):
        build_single_ellipsoid_bound(live_u, enlargement=0.0)
    with pytest.raises(ValueError, match="jitter"):
        build_single_ellipsoid_bound(live_u, jitter=0.0)


def test_unit_ball_log_volume_is_finite_for_common_dimensions() -> None:
    for ndim in (1, 2, 10):
        assert math.isfinite(unit_ball_log_volume(ndim))
