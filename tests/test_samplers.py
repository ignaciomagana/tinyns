from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from jax import random

from tinyns.bounds import build_single_ellipsoid_bound
from tinyns.samplers import (
    draw_constrained_prior,
    draw_constrained_prior_vectorized,
    draw_constrained_rwalk,
    draw_constrained_single_bound,
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




def test_local_sampler_max_attempts_is_respected() -> None:
    ndim = 2
    live_u = jnp.array([[0.4, 0.5], [0.6, 0.5]])
    live_logl = jnp.array([gaussian_loglike(u) for u in live_u])

    for sampler, kwargs in [
        (draw_constrained_rwalk, {"walks": 5, "step_scale": 0.01}),
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


def test_draw_constrained_rwalk_jax_counts_walks_on_easy_target() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    walks = 5
    live_u = jnp.full((4, 2), 0.5)
    live_logl = jnp.zeros(4)

    _, _, _, logl, ncall, accepted = draw_constrained_rwalk_jax(
        random.PRNGKey(123),
        gaussian_loglike,
        identity_prior_transform,
        -math.inf,
        live_u,
        live_logl,
        2,
        walks=walks,
        step_scale=0.01,
        max_attempts=100,
    )

    assert ncall == walks
    assert accepted is True
    assert math.isfinite(logl)


def test_draw_constrained_rwalk_jax_can_return_move_acceptance_info() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    walks = 5
    replacement_chains = 4
    live_u = jnp.full((4, 2), 0.5)
    live_logl = jnp.zeros(4)

    *_, ncall, accepted, info = draw_constrained_rwalk_jax(
        random.PRNGKey(124),
        gaussian_loglike,
        identity_prior_transform,
        -math.inf,
        live_u,
        live_logl,
        2,
        walks=walks,
        replacement_chains=replacement_chains,
        step_scale=0.01,
        max_attempts=100,
        return_info=True,
    )

    assert accepted is True
    assert ncall == walks * replacement_chains
    assert info["total_rwalk_proposals"] == ncall
    assert info["accepted_rwalk_moves"] <= info["total_rwalk_proposals"]
    assert 0.0 <= info["rwalk_acceptance"] <= 1.0


def test_draw_constrained_rwalk_jax_retries_until_chain_succeeds() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    walks = 3
    live_u = jnp.asarray([[0.1], [0.95]])
    live_logl = live_u[:, 0]

    _, _, _, logl, ncall, accepted = draw_constrained_rwalk_jax(
        random.PRNGKey(5),
        lambda theta: theta[0],
        identity_prior_transform,
        0.9,
        live_u,
        live_logl,
        1,
        walks=walks,
        step_scale=1e-6,
        max_attempts=30,
    )

    assert accepted is True
    assert ncall >= walks
    assert ncall % walks == 0
    assert logl >= 0.9


def test_draw_constrained_rwalk_jax_exhausts_full_walk_batches() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    walks = 5
    max_attempts = 12
    live_u = jnp.full((4, 2), 0.5)
    live_logl = jnp.zeros(4)

    _, u, theta, logl, ncall, accepted = draw_constrained_rwalk_jax(
        random.PRNGKey(321),
        gaussian_loglike,
        identity_prior_transform,
        math.inf,
        live_u,
        live_logl,
        2,
        walks=walks,
        step_scale=0.01,
        max_attempts=max_attempts,
    )

    assert accepted is False
    assert ncall == (max_attempts // walks) * walks
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert math.isfinite(logl)


def test_draw_constrained_rwalk_jax_rejects_invalid_min_accepts() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    with pytest.raises(ValueError, match="min_accepts"):
        draw_constrained_rwalk_jax(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            jnp.full((2, 2), 0.5),
            jnp.zeros(2),
            2,
            min_accepts=0,
        )


def test_draw_constrained_rwalk_jax_batched_chains_first_batch_succeeds() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    _, _, _, logl, ncall, accepted = draw_constrained_rwalk_jax(
        random.PRNGKey(42),
        gaussian_loglike,
        identity_prior_transform,
        -math.inf,
        jnp.full((8, 2), 0.5),
        jnp.zeros(8),
        2,
        walks=5,
        replacement_chains=4,
        step_scale=0.01,
        max_attempts=100,
    )

    assert accepted is True
    assert ncall == 20
    assert math.isfinite(logl)


def test_draw_constrained_rwalk_jax_batched_chains_exhausts_max_attempts() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    _, _, _, logl, ncall, accepted = draw_constrained_rwalk_jax(
        random.PRNGKey(43),
        gaussian_loglike,
        identity_prior_transform,
        math.inf,
        jnp.full((8, 2), 0.5),
        jnp.zeros(8),
        2,
        walks=5,
        replacement_chains=4,
        step_scale=0.01,
        max_attempts=100,
    )

    assert accepted is False
    assert ncall == 100
    assert math.isfinite(logl)


@pytest.mark.parametrize("replacement_chains", [0, True])
def test_draw_constrained_rwalk_jax_rejects_invalid_replacement_chains(
    replacement_chains,
) -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    with pytest.raises(ValueError, match="replacement_chains"):
        draw_constrained_rwalk_jax(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            jnp.full((2, 2), 0.5),
            jnp.zeros(2),
            2,
            replacement_chains=replacement_chains,
        )


def test_draw_constrained_rwalk_jax_rejects_batch_larger_than_max_attempts() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    with pytest.raises(ValueError, match=r"walks \* replacement_chains"):
        draw_constrained_rwalk_jax(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -math.inf,
            jnp.full((2, 2), 0.5),
            jnp.zeros(2),
            2,
            walks=5,
            replacement_chains=4,
            max_attempts=19,
        )


def test_draw_constrained_rwalk_jax_adaptive_accepts_schedule() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax_adaptive

    _, _, _, _logl, ncall, accepted, info = draw_constrained_rwalk_jax_adaptive(
        random.PRNGKey(0),
        gaussian_loglike,
        identity_prior_transform,
        -1.0,
        jnp.array([[0.5]]),
        jnp.array([0.0]),
        1,
        walks=2,
        max_attempts=32,
        replacement_chain_schedule=(1, 4, 16),
    )
    assert accepted
    assert ncall == 2
    assert info["replacement_chains_used"] == 1


@pytest.mark.parametrize("schedule", [(), (0,), (-1,), (True,)])
def test_draw_constrained_rwalk_jax_adaptive_rejects_invalid_schedule(schedule) -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax_adaptive

    with pytest.raises(ValueError, match="replacement_chain_schedule"):
        draw_constrained_rwalk_jax_adaptive(
            random.PRNGKey(0),
            gaussian_loglike,
            identity_prior_transform,
            -1.0,
            jnp.array([[0.5]]),
            jnp.array([0.0]),
            1,
            walks=2,
            max_attempts=32,
            replacement_chain_schedule=schedule,
        )


def test_draw_constrained_rwalk_jax_adaptive_exhausts_schedule_budget() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax_adaptive

    _, _, _, _logl, ncall, accepted, info = draw_constrained_rwalk_jax_adaptive(
        random.PRNGKey(0),
        gaussian_loglike,
        identity_prior_transform,
        1.0,
        jnp.array([[0.5]]),
        jnp.array([0.0]),
        1,
        walks=2,
        max_attempts=10,
        replacement_chain_schedule=(1, 4),
    )
    assert not accepted
    assert ncall == 10
    assert info["replacement_chains_used"] == 5


def test_draw_constrained_single_bound_returns_finite_point() -> None:
    live_u = jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6], [0.5, 0.5]])
    bound = build_single_ellipsoid_bound(live_u)

    _, u, theta, logl, ncall, accepted, info = draw_constrained_single_bound(
        random.PRNGKey(987),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        bound,
        2,
        max_attempts=20,
        batch_size=8,
    )

    assert accepted
    assert ncall >= 1
    assert jnp.all(jnp.isfinite(u))
    assert jnp.all(jnp.isfinite(theta))
    assert jnp.isfinite(logl)
    assert info["bound_draws"] >= info["bound_loglike_evals"]


def test_draw_constrained_single_bound_jax_accepts_single_bound() -> None:
    from tinyns.samplers import draw_constrained_single_bound_jax

    live_u = jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6], [0.5, 0.5]])
    bound = build_single_ellipsoid_bound(live_u)

    _, u, theta, logl, ncall, accepted, info = draw_constrained_single_bound_jax(
        random.PRNGKey(1001),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        bound,
        2,
        batch_size=16,
        max_batches=4,
    )

    assert accepted is True
    assert ncall == info["bound_seed_loglike_evals"]
    assert ncall % 16 == 0
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert jnp.isfinite(logl)
    assert set(info) == {
        "bound_seed_draws",
        "bound_seed_loglike_evals",
        "bound_seed_batches",
        "bound_seed_unit_cube_acceptance",
    }


def test_draw_constrained_single_bound_jax_accepts_jax_bound() -> None:
    from tinyns.bounds import as_jax_ellipsoid_bound
    from tinyns.samplers import draw_constrained_single_bound_jax

    live_u = jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6], [0.5, 0.5]])
    bound = as_jax_ellipsoid_bound(build_single_ellipsoid_bound(live_u))

    _, _u, _theta, logl, ncall, accepted, info = draw_constrained_single_bound_jax(
        random.PRNGKey(1002),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        bound,
        2,
        batch_size=8,
        max_batches=4,
    )

    assert accepted is True
    assert ncall == info["bound_seed_loglike_evals"]
    assert jnp.isfinite(logl)


def test_draw_constrained_single_bound_rwalk_jax_shapes_and_info() -> None:
    from tinyns.samplers import draw_constrained_single_bound_rwalk_jax

    live_u = jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6], [0.5, 0.5]])
    bound = build_single_ellipsoid_bound(live_u)

    _, u, theta, logl, ncall, accepted, info = (
        draw_constrained_single_bound_rwalk_jax(
            random.PRNGKey(10021),
            gaussian_loglike,
            identity_prior_transform,
            -10.0,
            bound,
            2,
            walks=3,
            step_scale=0.01,
            replacement_chains=2,
            max_attempts=30,
            bound_batch_size=8,
            bound_max_batches=4,
        )
    )

    assert accepted is True
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert jnp.isfinite(logl)
    assert ncall == info["bound_seed_loglike_evals"] + info["rwalk_kernel_calls"]
    assert info["replacement_batches"] >= 1
    assert info["replacement_chains_used"] >= 2
    assert info["replacement_chain_usage_counts"]["2"] >= 1


def test_draw_single_bound_rwalk_jax_impossible_seed_fails_cleanly() -> None:
    from tinyns.samplers import draw_constrained_single_bound_rwalk_jax

    live_u = jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6], [0.5, 0.5]])
    bound = build_single_ellipsoid_bound(live_u)

    _, u, theta, logl, ncall, accepted, info = (
        draw_constrained_single_bound_rwalk_jax(
            random.PRNGKey(10022),
            gaussian_loglike,
            identity_prior_transform,
            1.0,
            bound,
            2,
            walks=3,
            max_attempts=30,
            replacement_chains=2,
            bound_batch_size=8,
            bound_max_batches=2,
        )
    )

    assert accepted is False
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert jnp.isfinite(logl)
    assert ncall == info["bound_seed_loglike_evals"]
    assert info["rwalk_kernel_calls"] == 0
    assert info["replacement_batches"] == 0


def test_draw_constrained_single_bound_jax_impossible_threshold_returns_best() -> None:
    from tinyns.samplers import draw_constrained_single_bound_jax

    live_u = jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6], [0.5, 0.5]])
    bound = build_single_ellipsoid_bound(live_u)

    _, u, theta, logl, ncall, accepted, info = draw_constrained_single_bound_jax(
        random.PRNGKey(1003),
        gaussian_loglike,
        identity_prior_transform,
        1.0,
        bound,
        2,
        batch_size=8,
        max_batches=3,
    )

    assert accepted is False
    assert ncall == 24
    assert info["bound_seed_batches"] == 3
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert logl <= 0.0


def test_draw_constrained_single_bound_jax_random_selection_not_argmax_only() -> None:
    from tinyns.samplers import draw_constrained_single_bound_jax

    live_u = jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6], [0.5, 0.5]])
    bound = build_single_ellipsoid_bound(live_u)

    argmax_count = 0
    runs = 16
    for seed in range(runs):
        _, u, _theta, logl, _ncall, accepted, _info = draw_constrained_single_bound_jax(
            random.PRNGKey(seed),
            gaussian_loglike,
            identity_prior_transform,
            -10.0,
            bound,
            2,
            batch_size=32,
            max_batches=1,
        )
        assert accepted is True
        # If the implementation always chose the argmax accepted point, the
        # returned point would equal the best log likelihood in the same batch.
        _, replay_key, _select_key = random.split(random.PRNGKey(seed), 3)
        from tinyns.bounds import (
            as_jax_ellipsoid_bound,
            in_unit_cube,
            sample_jax_ellipsoid_bound,
        )

        batch, _ = sample_jax_ellipsoid_bound(
            replay_key, as_jax_ellipsoid_bound(bound), 32
        )
        masked = jnp.where(
            in_unit_cube(batch), jax.vmap(gaussian_loglike)(batch), -jnp.inf
        )
        is_argmax = jnp.allclose(u, batch[jnp.argmax(masked)]) and jnp.isclose(
            logl, jnp.max(masked)
        )
        argmax_count += bool(is_argmax)

    assert argmax_count < runs


def test_draw_constrained_single_bound_jax_rejects_bad_shapes() -> None:
    from tinyns.samplers import draw_constrained_single_bound_jax

    live_u = jnp.asarray([[0.2, 0.3], [0.4, 0.7], [0.8, 0.6], [0.5, 0.5]])
    bound = build_single_ellipsoid_bound(live_u)

    with pytest.raises(ValueError, match="prior_transform"):
        draw_constrained_single_bound_jax(
            random.PRNGKey(1004),
            gaussian_loglike,
            lambda u: u[0],
            -10.0,
            bound,
            2,
            batch_size=4,
            max_batches=1,
        )

    with pytest.raises(ValueError, match="loglike"):
        draw_constrained_single_bound_jax(
            random.PRNGKey(1005),
            lambda theta: theta,
            identity_prior_transform,
            -10.0,
            bound,
            2,
            batch_size=4,
            max_batches=1,
        )


def _two_ellipsoid_bound():
    from tinyns.bounds import MultiEllipsoidBound

    left = build_single_ellipsoid_bound(
        jnp.asarray([[0.15, 0.20], [0.20, 0.25], [0.25, 0.20], [0.20, 0.15]])
    )
    right = build_single_ellipsoid_bound(
        jnp.asarray([[0.70, 0.75], [0.75, 0.80], [0.80, 0.75], [0.75, 0.70]])
    )
    log_volumes = jnp.asarray([left.log_volume, right.log_volume])
    log_total_volume = float(
        jnp.max(log_volumes)
        + jnp.log(jnp.sum(jnp.exp(log_volumes - jnp.max(log_volumes))))
    )
    return MultiEllipsoidBound((left, right), log_volumes, log_total_volume, 2)


def test_draw_constrained_multi_bound_rwalk_jax_shapes_and_info() -> None:
    from tinyns.samplers import draw_constrained_multi_bound_rwalk_jax

    _, u, theta, logl, ncall, accepted, info = draw_constrained_multi_bound_rwalk_jax(
        random.PRNGKey(1100),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        _two_ellipsoid_bound(),
        2,
        walks=2,
        replacement_chains=2,
        bound_batch_size=8,
        bound_max_batches=4,
    )

    assert accepted is True
    assert ncall == info["bound_seed_loglike_evals"] + info["rwalk_kernel_calls"]
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert jnp.isfinite(logl)
    assert info["bound_seed_nellipsoids"] == 2
    assert "bound_seed_overlap_rejections" in info
    assert "bound_seed_ellipsoid_index" in info


def test_draw_constrained_multi_bound_jax_accepts_multi_bound() -> None:
    from tinyns.samplers import draw_constrained_multi_bound_jax

    bound = _two_ellipsoid_bound()

    _, u, theta, logl, ncall, accepted, info = draw_constrained_multi_bound_jax(
        random.PRNGKey(1101),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        bound,
        2,
        batch_size=16,
        max_batches=4,
    )

    assert accepted is True
    assert ncall == info["bound_seed_loglike_evals"]
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert jnp.isfinite(logl)
    assert info["bound_seed_nellipsoids"] == 2
    assert "bound_seed_overlap_rejections" in info


def test_draw_constrained_multi_bound_jax_accepts_jax_bound() -> None:
    from tinyns.bounds import as_jax_ellipsoid_bound
    from tinyns.samplers import draw_constrained_multi_bound_jax

    bound = as_jax_ellipsoid_bound(_two_ellipsoid_bound())

    _, _u, _theta, logl, ncall, accepted, info = draw_constrained_multi_bound_jax(
        random.PRNGKey(1102),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        bound,
        2,
        batch_size=8,
        max_batches=4,
    )

    assert accepted is True
    assert ncall == info["bound_seed_loglike_evals"]
    assert jnp.isfinite(logl)
    assert info["bound_seed_nellipsoids"] == 2


@pytest.mark.parametrize("overlap_correction", [True, False])
def test_draw_constrained_multi_bound_jax_overlap_modes_run(overlap_correction) -> None:
    from tinyns.samplers import draw_constrained_multi_bound_jax

    _, _, _, logl, _, accepted, info = draw_constrained_multi_bound_jax(
        random.PRNGKey(1103),
        gaussian_loglike,
        identity_prior_transform,
        -10.0,
        _two_ellipsoid_bound(),
        2,
        batch_size=8,
        max_batches=4,
        overlap_correction=overlap_correction,
    )

    assert accepted is True
    assert jnp.isfinite(logl)
    assert info["bound_seed_overlap_rejections"] >= 0
    assert info["bound_seed_nellipsoids"] == 2


def test_draw_constrained_multi_bound_jax_impossible_threshold_returns_best() -> None:
    from tinyns.samplers import draw_constrained_multi_bound_jax

    _, u, theta, logl, ncall, accepted, info = draw_constrained_multi_bound_jax(
        random.PRNGKey(1104),
        gaussian_loglike,
        identity_prior_transform,
        1.0,
        _two_ellipsoid_bound(),
        2,
        batch_size=8,
        max_batches=3,
    )

    assert accepted is False
    assert ncall == 24
    assert info["bound_seed_batches"] == 3
    assert u.shape == (2,)
    assert theta.shape == (2,)
    assert logl <= 0.0


def test_evaluate_jax_batch_scalar_functions_use_vmap() -> None:
    from tinyns.samplers import _evaluate_jax_batch

    u_batch = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])
    theta, logl = _evaluate_jax_batch(
        gaussian_loglike,
        identity_prior_transform,
        u_batch,
        2,
        jax_vectorized=False,
    )

    assert theta.shape == (2, 2)
    assert logl.shape == (2,)
    assert jnp.allclose(theta, u_batch)


def test_evaluate_jax_batch_vectorized_functions() -> None:
    from tinyns.samplers import _evaluate_jax_batch

    def prior_batch(u):
        return 2.0 * u - 1.0

    def loglike_batch(theta):
        return -jnp.sum(theta**2, axis=1)

    theta, logl = _evaluate_jax_batch(
        loglike_batch,
        prior_batch,
        jnp.asarray([[0.25, 0.5], [0.75, 0.5]]),
        2,
        jax_vectorized=True,
    )

    assert theta.shape == (2, 2)
    assert logl.shape == (2,)
    assert jnp.allclose(logl, jnp.asarray([-0.25, -0.25]))


def test_evaluate_jax_batch_vectorized_shape_errors() -> None:
    from tinyns.samplers import _evaluate_jax_batch

    u_batch = jnp.ones((3, 2))
    with pytest.raises(ValueError, match="jax_vectorized prior_transform"):
        _evaluate_jax_batch(
            gaussian_loglike,
            lambda u: u[0],
            u_batch,
            2,
            jax_vectorized=True,
        )

    with pytest.raises(ValueError, match="jax_vectorized loglike"):
        _evaluate_jax_batch(
            lambda theta: theta,
            lambda u: u,
            u_batch,
            2,
            jax_vectorized=True,
        )


def _nan_outside_prior_transform(u):
    u = jnp.asarray(u)
    outside = jnp.any((u < 0.0) | (u > 1.0))
    return jnp.where(outside, jnp.full_like(u, jnp.nan), u)


def _large_test_bound(ndim=2):
    from tinyns.bounds import SingleEllipsoidBound

    center = jnp.full((ndim,), 0.5)
    chol = jnp.eye(ndim) * 5.0
    inv_chol = jnp.linalg.inv(chol)
    return SingleEllipsoidBound(
        center=center,
        chol=chol,
        inv_chol=inv_chol,
        enlargement=1.0,
        log_volume=0.0,
        ndim=ndim,
    )


def test_draw_constrained_single_bound_jax_masks_outside_before_prior_transform():
    from tinyns.samplers import draw_constrained_single_bound_jax

    _, u, theta, logl, _ncall, accepted, _info = draw_constrained_single_bound_jax(
        random.PRNGKey(5001),
        gaussian_loglike,
        _nan_outside_prior_transform,
        -10.0,
        _large_test_bound(),
        2,
        batch_size=64,
        max_batches=8,
    )

    assert accepted is True
    assert bool(jnp.all((u >= 0.0) & (u <= 1.0)))
    assert bool(jnp.all((theta >= 0.0) & (theta <= 1.0)))
    assert jnp.isfinite(logl)


def test_draw_constrained_single_bound_jax_failure_fallback_is_finite_and_safe():
    from tinyns.samplers import draw_constrained_single_bound_jax

    _, u, theta, logl, _ncall, accepted, _info = draw_constrained_single_bound_jax(
        random.PRNGKey(5002),
        gaussian_loglike,
        _nan_outside_prior_transform,
        jnp.inf,
        _large_test_bound(),
        2,
        batch_size=1,
        max_batches=1,
    )

    assert accepted is False
    assert bool(jnp.all((u >= 0.0) & (u <= 1.0)))
    assert bool(jnp.all((theta >= 0.0) & (theta <= 1.0)))
    assert jnp.isfinite(logl)


def test_draw_constrained_multi_bound_jax_masks_outside_before_prior_transform():
    from tinyns.bounds import MultiEllipsoidBound
    from tinyns.samplers import draw_constrained_multi_bound_jax

    ellipsoid = _large_test_bound()
    bound = MultiEllipsoidBound(
        ellipsoids=(ellipsoid, ellipsoid),
        log_volumes=jnp.asarray([0.0, 0.0]),
        log_total_volume=float(jnp.log(2.0)),
        ndim=2,
    )

    _, u, theta, logl, _ncall, accepted, _info = draw_constrained_multi_bound_jax(
        random.PRNGKey(5003),
        gaussian_loglike,
        _nan_outside_prior_transform,
        -10.0,
        bound,
        2,
        batch_size=64,
        max_batches=8,
    )

    assert accepted is True
    assert bool(jnp.all((u >= 0.0) & (u <= 1.0)))
    assert bool(jnp.all((theta >= 0.0) & (theta <= 1.0)))
    assert jnp.isfinite(logl)


def test_draw_constrained_multi_bound_jax_failure_fallback_is_finite_and_safe():
    from tinyns.bounds import MultiEllipsoidBound
    from tinyns.samplers import draw_constrained_multi_bound_jax

    ellipsoid = _large_test_bound()
    bound = MultiEllipsoidBound(
        ellipsoids=(ellipsoid, ellipsoid),
        log_volumes=jnp.asarray([0.0, 0.0]),
        log_total_volume=float(jnp.log(2.0)),
        ndim=2,
    )

    _, u, theta, logl, _ncall, accepted, _info = draw_constrained_multi_bound_jax(
        random.PRNGKey(5004),
        gaussian_loglike,
        _nan_outside_prior_transform,
        jnp.inf,
        bound,
        2,
        batch_size=1,
        max_batches=1,
    )

    assert accepted is False
    assert bool(jnp.all((u >= 0.0) & (u <= 1.0)))
    assert bool(jnp.all((theta >= 0.0) & (theta <= 1.0)))
    assert jnp.isfinite(logl)
