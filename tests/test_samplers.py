from __future__ import annotations

import math

import jax.numpy as jnp
import pytest
from jax import random

from tinyns.samplers import (
    draw_constrained_prior,
    draw_constrained_prior_vectorized,
    draw_constrained_rslice,
    draw_constrained_rwalk,
    draw_constrained_slice,
)


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


def test_draw_constrained_slice_accepts_loose_threshold() -> None:
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

    _, u, theta, logl, ncall, accepted = draw_constrained_slice(
        random.PRNGKey(12),
        gaussian_loglike,
        identity_prior_transform,
        logl_min,
        live_u,
        live_logl,
        ndim,
        slices=4,
        slice_steps=5,
        step_scale=0.05,
        max_attempts=10,
    )

    assert accepted is True
    assert u.shape == (ndim,)
    assert theta.shape == (ndim,)
    assert logl >= logl_min
    assert ncall > 0


def test_draw_constrained_slice_returns_best_after_impossible_threshold() -> None:
    ndim = 2
    live_u = jnp.array([[0.25, 0.25], [0.75, 0.75]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])
    max_attempts = 3
    slices = 2
    slice_steps = 4

    _, u, theta, logl, ncall, accepted = draw_constrained_slice(
        random.PRNGKey(13),
        gaussian_loglike,
        identity_prior_transform,
        math.inf,
        live_u,
        live_logl,
        ndim,
        slices=slices,
        slice_steps=slice_steps,
        step_scale=0.1,
        max_attempts=max_attempts,
    )

    assert accepted is False
    assert u.shape == (ndim,)
    assert theta.shape == (ndim,)
    assert math.isfinite(logl)
    assert ncall == max_attempts


def test_draw_constrained_slice_rejects_invalid_parameters_and_shapes() -> None:
    live_u = jnp.ones((3, 2)) * 0.5
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    with pytest.raises(ValueError, match="slices"):
        draw_constrained_slice(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            live_logl,
            2,
            slices=0,
        )

    with pytest.raises(ValueError, match="slice_steps"):
        draw_constrained_slice(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            live_logl,
            2,
            slice_steps=0,
        )

    with pytest.raises(ValueError, match="step_scale"):
        draw_constrained_slice(
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
        draw_constrained_slice(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            jnp.ones((3, 3)),
            live_logl,
            2,
        )

    with pytest.raises(ValueError, match="live_logl"):
        draw_constrained_slice(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            jnp.ones((2,)),
            2,
        )

    with pytest.raises(ValueError, match="min_accepts"):
        draw_constrained_slice(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            live_logl,
            2,
            min_accepts=0,
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
    assert ncall == max_attempts


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

    with pytest.raises(ValueError, match="min_accepts"):
        draw_constrained_rwalk(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            live_logl,
            2,
            min_accepts=0,
        )


def test_draw_constrained_prior_vectorized_accepts_easy_threshold() -> None:
    ndim = 2

    _, u, theta, logl, ncall, accepted = draw_constrained_prior_vectorized(
        random.PRNGKey(9),
        lambda theta_batch: -jnp.sum(theta_batch**2, axis=1),
        lambda u_batch: u_batch,
        -10.0,
        ndim,
        batch_size=4,
    )

    assert accepted is True
    assert u.shape == (ndim,)
    assert theta.shape == (ndim,)
    assert logl >= -10.0
    assert ncall == 4


def test_draw_constrained_prior_vectorized_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        draw_constrained_prior_vectorized(
            random.PRNGKey(10),
            lambda theta_batch: jnp.zeros((theta_batch.shape[0],)),
            lambda u_batch: u_batch,
            -math.inf,
            2,
            batch_size=0,
        )


def test_draw_constrained_prior_vectorized_rejects_wrong_prior_shape() -> None:
    with pytest.raises(ValueError, match="prior_transform"):
        draw_constrained_prior_vectorized(
            random.PRNGKey(11),
            lambda theta_batch: jnp.zeros((theta_batch.shape[0],)),
            lambda u_batch: u_batch[:, 0],
            -math.inf,
            2,
            batch_size=3,
        )


def test_draw_constrained_rslice_accepts_loose_threshold() -> None:
    ndim = 3
    live_u = jnp.array([[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])
    logl_min = -10.0

    _, u, theta, logl, ncall, accepted = draw_constrained_rslice(
        random.PRNGKey(22),
        gaussian_loglike,
        identity_prior_transform,
        logl_min,
        live_u,
        live_logl,
        ndim,
        slices=4,
        slice_steps=5,
        step_scale=0.05,
        max_attempts=10,
    )

    assert accepted is True
    assert u.shape == (ndim,)
    assert theta.shape == (ndim,)
    assert logl >= logl_min
    assert ncall > 0


def test_draw_constrained_rslice_returns_best_after_impossible_threshold() -> None:
    ndim = 2
    live_u = jnp.array([[0.25, 0.25], [0.75, 0.75]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])
    max_attempts = 3
    slices = 2
    slice_steps = 4

    _, u, theta, logl, ncall, accepted = draw_constrained_rslice(
        random.PRNGKey(23),
        gaussian_loglike,
        identity_prior_transform,
        math.inf,
        live_u,
        live_logl,
        ndim,
        slices=slices,
        slice_steps=slice_steps,
        step_scale=0.1,
        max_attempts=max_attempts,
    )

    assert accepted is False
    assert u.shape == (ndim,)
    assert theta.shape == (ndim,)
    assert math.isfinite(logl)
    assert ncall == max_attempts


def test_draw_constrained_rslice_rejects_invalid_parameters_and_shapes() -> None:
    live_u = jnp.ones((3, 2)) * 0.5
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    invalid_kwargs = [
        {"slices": 0},
        {"slice_steps": 0},
        {"step_scale": 0.0},
        {"max_attempts": 0},
        {"min_accepts": 0},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            draw_constrained_rslice(
                random.PRNGKey(0),
                gaussian_loglike,
                identity_prior_transform,
                -math.inf,
                live_u,
                live_logl,
                2,
                **kwargs,
            )

    with pytest.raises(ValueError, match="live_u"):
        draw_constrained_rslice(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            jnp.ones((3, 3)),
            live_logl,
            2,
        )

    with pytest.raises(ValueError, match="live_logl"):
        draw_constrained_rslice(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            live_u,
            jnp.ones((2,)),
            2,
        )


@pytest.mark.parametrize(
    "draw, kwargs",
    [
        (draw_constrained_slice, {"slices": 3, "slice_steps": 4, "step_scale": 0.05}),
        (draw_constrained_rslice, {"slices": 3, "slice_steps": 4, "step_scale": 0.05}),
    ],
)
def test_slice_samplers_min_accepts_three_accept_on_easy_target(draw, kwargs) -> None:
    ndim = 2
    live_u = jnp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    *_, ncall, accepted = draw(
        random.PRNGKey(103),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        live_u,
        live_logl,
        ndim,
        max_attempts=50,
        min_accepts=3,
        **kwargs,
    )

    assert accepted is True
    assert ncall >= 3


@pytest.mark.parametrize(
    "draw, kwargs",
    [
        (draw_constrained_rwalk, {"walks": 10, "step_scale": 0.05}),
        (draw_constrained_slice, {"slices": 4, "slice_steps": 5, "step_scale": 0.05}),
        (draw_constrained_rslice, {"slices": 4, "slice_steps": 5, "step_scale": 0.05}),
    ],
)
def test_min_accepts_three_accepts_and_costs_at_least_one(draw, kwargs) -> None:
    ndim = 2
    live_u = jnp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])
    args = (
        random.PRNGKey(101),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        live_u,
        live_logl,
        ndim,
    )

    *_, ncall_one, accepted_one = draw(*args, max_attempts=50, min_accepts=1, **kwargs)
    *_, ncall_three, accepted_three = draw(
        *args, max_attempts=50, min_accepts=3, **kwargs
    )

    assert accepted_one is True
    assert accepted_three is True
    assert ncall_three >= ncall_one


@pytest.mark.parametrize(
    "draw, kwargs",
    [
        (draw_constrained_rwalk, {"walks": 10, "step_scale": 0.05}),
        (draw_constrained_slice, {"slices": 4, "slice_steps": 5, "step_scale": 0.05}),
        (draw_constrained_rslice, {"slices": 4, "slice_steps": 5, "step_scale": 0.05}),
    ],
)
def test_min_accepts_impossible_budget_returns_unaccepted(draw, kwargs) -> None:
    ndim = 2
    live_u = jnp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    *_, ncall, accepted = draw(
        random.PRNGKey(102),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        live_u,
        live_logl,
        ndim,
        max_attempts=1,
        min_accepts=3,
        **kwargs,
    )

    assert accepted is False
    assert ncall == 1


def test_draw_constrained_rwalk_walks_are_full_update_length() -> None:
    ndim = 2
    live_u = jnp.array([[0.4, 0.5], [0.6, 0.5]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    *_, ncall, accepted = draw_constrained_rwalk(
        random.PRNGKey(100),
        gaussian_loglike,
        identity_prior_transform,
        -100.0,
        live_u,
        live_logl,
        ndim,
        walks=5,
        step_scale=0.01,
        max_attempts=20,
        min_accepts=1,
    )

    assert accepted is True
    assert ncall >= 5


def test_draw_constrained_rwalk_single_walk_still_works() -> None:
    ndim = 2
    live_u = jnp.array([[0.4, 0.5], [0.6, 0.5]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    *_, ncall, accepted = draw_constrained_rwalk(
        random.PRNGKey(101),
        gaussian_loglike,
        identity_prior_transform,
        -100.0,
        live_u,
        live_logl,
        ndim,
        walks=1,
        step_scale=0.01,
        max_attempts=20,
        min_accepts=1,
    )

    assert accepted is True
    assert ncall == 1


def test_draw_constrained_slice_slices_are_full_update_length_when_possible() -> None:
    ndim = 2
    live_u = jnp.array([[0.4, 0.5], [0.6, 0.5]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    *_, ncall, accepted = draw_constrained_slice(
        random.PRNGKey(102),
        gaussian_loglike,
        identity_prior_transform,
        -100.0,
        live_u,
        live_logl,
        ndim,
        slices=5,
        slice_steps=3,
        step_scale=0.01,
        max_attempts=20,
        min_accepts=1,
    )

    assert accepted is True
    assert ncall > 1


def test_draw_constrained_rslice_slices_are_full_update_length_when_possible() -> None:
    ndim = 2
    live_u = jnp.array([[0.4, 0.5], [0.6, 0.5]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    *_, ncall, accepted = draw_constrained_rslice(
        random.PRNGKey(103),
        gaussian_loglike,
        identity_prior_transform,
        -100.0,
        live_u,
        live_logl,
        ndim,
        slices=5,
        slice_steps=3,
        step_scale=0.01,
        max_attempts=20,
        min_accepts=1,
    )

    assert accepted is True
    assert ncall > 1


def test_local_sampler_max_attempts_is_respected() -> None:
    ndim = 2
    live_u = jnp.array([[0.4, 0.5], [0.6, 0.5]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    for sampler, kwargs in [
        (draw_constrained_rwalk, {"walks": 5, "step_scale": 0.01}),
        (draw_constrained_slice, {"slices": 5, "slice_steps": 3, "step_scale": 0.01}),
        (draw_constrained_rslice, {"slices": 5, "slice_steps": 3, "step_scale": 0.01}),
    ]:
        *_, ncall, accepted = sampler(
            random.PRNGKey(104),
            gaussian_loglike,
            identity_prior_transform,
            math.inf,
            live_u,
            live_logl,
            ndim,
            max_attempts=2,
            min_accepts=1,
            **kwargs,
        )
        assert accepted is False
        assert ncall == 2
