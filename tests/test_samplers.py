from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from jax import random

from tinyns.samplers import (
    draw_constrained_prior,
    draw_constrained_prior_vectorized,
    draw_constrained_rwalk,
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


def test_draw_constrained_prior_rejects_vectorized_kwarg() -> None:
    with pytest.raises(TypeError):
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
            min_accepts=-1,
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


def test_draw_constrained_rwalk_jax_keeps_vectorized_prior_calls_batched() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    def prior_batch(u):
        if u.ndim != 2:
            raise ValueError("batch-only prior received a non-batch input")
        return 2.0 * u - 1.0

    def loglike_batch(theta):
        if theta.ndim != 2:
            raise ValueError("batch-only likelihood received a non-batch input")
        return -jnp.sum(theta**2, axis=1)

    live_u = jnp.asarray(((0.2, 0.3), (0.4, 0.7), (0.8, 0.6), (0.5, 0.5)))
    live_logl = loglike_batch(prior_batch(live_u))
    _, new_u, new_theta, new_logl, ncall, accepted = (
        draw_constrained_rwalk_jax(
            random.PRNGKey(125),
            loglike_batch,
            prior_batch,
            -math.inf,
            live_u,
            live_logl,
            2,
            walks=2,
            replacement_chains=2,
            max_attempts=8,
            jax_vectorized=True,
        )
    )

    assert accepted is True
    assert ncall == 4
    assert new_u.shape == (2,)
    assert new_theta.shape == (2,)
    assert jnp.isfinite(new_logl)


def test_draw_constrained_rwalk_jax_normalizes_scalar_one_dimensional_prior() -> None:
    from tinyns.samplers import draw_constrained_rwalk_jax

    def scalar_prior(u):
        return 2.0 * u[0] - 1.0

    def vector_loglike(theta):
        return -(theta[0] ** 2)

    live_u = jnp.asarray(((0.2,), (0.4,), (0.6,), (0.8,)))
    live_logl = -(2.0 * live_u[:, 0] - 1.0) ** 2
    _, new_u, new_theta, new_logl, ncall, accepted = (
        draw_constrained_rwalk_jax(
            random.PRNGKey(126),
            vector_loglike,
            scalar_prior,
            -math.inf,
            live_u,
            live_logl,
            1,
            walks=2,
            replacement_chains=2,
            max_attempts=8,
        )
    )

    assert accepted is True
    assert ncall == 4
    assert new_u.shape == (1,)
    assert new_theta.shape == (1,)
    assert jnp.isfinite(new_logl)


def test_draw_constrained_rwalk_jax_caches_unhashable_callable_instances() -> None:
    from tinyns.samplers import (
        _make_rwalk_jax_kernel,
        draw_constrained_rwalk_jax,
    )

    class UnhashablePrior:
        __hash__ = None

        def __call__(self, u):
            return 2.0 * u - 1.0

    class UnhashableLogLike:
        __hash__ = None

        def __call__(self, theta):
            return -jnp.sum(theta**2)

    prior = UnhashablePrior()
    loglike = UnhashableLogLike()
    _make_rwalk_jax_kernel.cache_clear()
    first_kernel = _make_rwalk_jax_kernel(loglike, prior, 2, 2, 1, False)
    second_kernel = _make_rwalk_jax_kernel(loglike, prior, 2, 2, 1, False)

    assert first_kernel is second_kernel
    live_u = jnp.asarray(((0.2, 0.3), (0.4, 0.7), (0.8, 0.6), (0.5, 0.5)))
    live_logl = jax.vmap(lambda u: loglike(prior(u)))(live_u)
    _, new_u, new_theta, new_logl, ncall, accepted = (
        draw_constrained_rwalk_jax(
            random.PRNGKey(127),
            loglike,
            prior,
            -math.inf,
            live_u,
            live_logl,
            2,
            walks=2,
            max_attempts=4,
        )
    )

    assert accepted is True
    assert ncall == 2
    assert new_u.shape == (2,)
    assert new_theta.shape == (2,)
    assert jnp.isfinite(new_logl)


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
            min_accepts=-1,
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


def test_rwalk_jax_unmoved_chain_returns_seed_by_default() -> None:
    """With min_accepts=0 a chain that cannot move is kept as a copy of its seed."""
    import jax.numpy as jnp
    from jax import random

    from tinyns.samplers import draw_constrained_rwalk_jax

    live_u = jnp.asarray([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
    live_logl = jnp.asarray([0.0, 1.0, 2.0])

    def loglike(theta):
        # only the exact live points are inside the constraint
        hit = jnp.any(jnp.all(jnp.isclose(theta, live_u), axis=1))
        return jnp.where(hit, jnp.sum(theta) * 2.0 - 0.8 * 2.0 + 2.0 - 1.2 * 0.0, -10.0)

    def prior_transform(u):
        return u

    kwargs = dict(walks=6, step_scale=0.3, max_attempts=60, proposal="live-cov")
    out = draw_constrained_rwalk_jax(
        random.PRNGKey(0), loglike, prior_transform, 0.5, live_u, live_logl, 2,
        return_info=True, **kwargs,
    )
    _, new_u, _, new_logl, _, accepted, info = out
    assert accepted is True
    assert info["accepted_rwalk_moves"] == 0
    assert bool(jnp.any(jnp.all(new_u == live_u[1:], axis=1)))  # a seed above logl_min
    assert new_logl > 0.5

    # the old rule (min_accepts=1) keeps retrying and reports failure
    out_old = draw_constrained_rwalk_jax(
        random.PRNGKey(0), loglike, prior_transform, 0.5, live_u, live_logl, 2,
        min_accepts=1, **kwargs,
    )
    assert out_old[5] is False
