from __future__ import annotations

import math

import jax.numpy as jnp
import pytest
from jax import random

from tinyns.bounds import (
    JaxEllipsoidBound,
    as_jax_ellipsoid_bound,
    build_multi_ellipsoid_bound,
    build_single_ellipsoid_bound,
    contains_jax_ellipsoid_bound,
    contains_single_ellipsoid,
    count_containing_ellipsoids,
    count_containing_jax_ellipsoids,
    in_unit_cube,
    jax_bound_volume_probs,
    sample_single_ellipsoid,
    unit_ball_log_volume,
)


def _live_points(nlive: int = 64, ndim: int = 3):
    key = random.PRNGKey(0)
    return 0.5 + 0.1 * random.normal(key, shape=(nlive, ndim))


def _clustered_live_points() -> jnp.ndarray:
    return jnp.concatenate(
        [
            0.25 + 0.03 * random.normal(random.PRNGKey(10), shape=(24, 2)),
            0.75 + 0.03 * random.normal(random.PRNGKey(11), shape=(24, 2)),
        ],
        axis=0,
    )


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


def test_multi_ellipsoid_bound_geometry_smoke() -> None:
    live_u = _clustered_live_points()
    from tinyns.bounds import (
        contains_multi_ellipsoid,
        sample_multi_ellipsoid,
    )

    bound = build_multi_ellipsoid_bound(
        live_u, max_ellipsoids=4, min_points=8, split_threshold=0.99
    )
    samples, indices = sample_multi_ellipsoid(random.PRNGKey(12), bound, 13)
    counts = count_containing_ellipsoids(bound, live_u)

    assert 1 <= len(bound.ellipsoids) <= 4
    assert bool(jnp.all(contains_multi_ellipsoid(bound, live_u)))
    assert samples.shape == (13, 2)
    assert indices.shape == (13,)
    assert bool(jnp.all(counts > 0))
    assert math.isfinite(bound.log_total_volume)


def test_as_jax_ellipsoid_bound_converts_single_bound() -> None:
    single = build_single_ellipsoid_bound(_live_points(32, 3))

    bound = as_jax_ellipsoid_bound(single)

    assert isinstance(bound, JaxEllipsoidBound)
    assert bound.centers.shape == (1, 3)
    assert bound.chols.shape == (1, 3, 3)
    assert bound.inv_chols.shape == (1, 3, 3)
    assert bound.log_volumes.shape == (1,)
    assert bound.active.tolist() == [True]
    assert bound.n_active == 1
    assert bound.ndim == 3
    assert bound.max_ellipsoids == 1
    assert bound.log_total_volume == pytest.approx(single.log_volume)


def test_as_jax_ellipsoid_bound_converts_multi_bound() -> None:
    multi = build_multi_ellipsoid_bound(
        _clustered_live_points(),
        max_ellipsoids=4,
        min_points=8,
        split_threshold=0.99,
    )

    bound = as_jax_ellipsoid_bound(multi, max_ellipsoids=5)

    assert bound.centers.shape == (5, 2)
    assert bound.chols.shape == (5, 2, 2)
    assert bound.inv_chols.shape == (5, 2, 2)
    assert bound.log_volumes.shape == (5,)
    assert bound.active.shape == (5,)
    assert bound.n_active == len(multi.ellipsoids)
    assert bound.ndim == 2
    assert bound.max_ellipsoids == 5
    assert bound.log_total_volume == pytest.approx(multi.log_total_volume)


def test_jax_ellipsoid_bound_padding_is_inactive_and_finite_where_appropriate():
    single = build_single_ellipsoid_bound(_live_points(32, 2))

    bound = as_jax_ellipsoid_bound(single, max_ellipsoids=3)

    assert bound.active.tolist() == [True, False, False]
    assert jnp.all(bound.centers[1:] == 0.0)
    assert jnp.all(bound.chols[1:] == jnp.eye(2))
    assert jnp.all(bound.inv_chols[1:] == jnp.eye(2))
    assert jnp.all(jnp.isfinite(bound.centers[1:]))
    assert jnp.all(jnp.isfinite(bound.chols[1:]))
    assert jnp.all(jnp.isfinite(bound.inv_chols[1:]))
    assert jnp.all(jnp.isneginf(bound.log_volumes[1:]))


def test_contains_jax_ellipsoid_bound_agrees_with_single_bound() -> None:
    live_u = _live_points(32, 3)
    single = build_single_ellipsoid_bound(live_u)
    bound = as_jax_ellipsoid_bound(single)

    assert bool(contains_jax_ellipsoid_bound(bound, live_u[0]))
    assert jnp.array_equal(
        contains_jax_ellipsoid_bound(bound, live_u[:5]),
        contains_single_ellipsoid(single, live_u[:5]),
    )


def test_count_containing_jax_ellipsoids_agrees_with_multi_bound() -> None:
    live_u = _clustered_live_points()
    multi = build_multi_ellipsoid_bound(
        live_u, max_ellipsoids=4, min_points=8, split_threshold=0.99
    )
    bound = as_jax_ellipsoid_bound(multi, max_ellipsoids=6)

    assert jnp.array_equal(
        count_containing_jax_ellipsoids(bound, live_u),
        count_containing_ellipsoids(multi, live_u),
    )


def test_jax_bound_volume_probs_sum_to_one_over_active_entries() -> None:
    multi = build_multi_ellipsoid_bound(
        _clustered_live_points(),
        max_ellipsoids=4,
        min_points=8,
        split_threshold=0.99,
    )
    bound = as_jax_ellipsoid_bound(multi, max_ellipsoids=6)

    probs = jax_bound_volume_probs(bound)

    assert probs.shape == (6,)
    assert jnp.sum(probs[: bound.n_active]) == pytest.approx(1.0)
    assert jnp.all(probs[bound.n_active :] == 0.0)


def test_as_jax_ellipsoid_bound_invalid_max_ellipsoids_raises() -> None:
    multi = build_multi_ellipsoid_bound(
        _clustered_live_points(),
        max_ellipsoids=4,
        min_points=8,
        split_threshold=0.99,
    )

    with pytest.raises(ValueError, match="max_ellipsoids"):
        as_jax_ellipsoid_bound(multi, max_ellipsoids=len(multi.ellipsoids) - 1)


def test_jax_ellipsoid_bound_shapes_are_stable_for_common_dimensions() -> None:
    for ndim in (1, 2, 10):
        single = build_single_ellipsoid_bound(_live_points(32, ndim))
        bound = as_jax_ellipsoid_bound(single, max_ellipsoids=4)

        assert bound.centers.shape == (4, ndim)
        assert bound.chols.shape == (4, ndim, ndim)
        assert bound.inv_chols.shape == (4, ndim, ndim)
        assert bound.log_volumes.shape == (4,)
        assert bound.active.shape == (4,)
