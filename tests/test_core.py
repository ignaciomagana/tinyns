"""The functional core: ``Config``, ``init`` and ``step`` (``tinyns.core``)."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from tinyns import NestedSampler, core
from tinyns.run import _update_scale

NDIM, NLIVE, BLOCK = 3, 40, 8


def loglike(theta):
    return -0.5 * jnp.sum(((theta - 0.2) / 0.3) ** 2)


def prior_transform(u):
    return 2.0 * u - 1.0


def make_config(**kwargs):
    kwargs.setdefault("block_size", BLOCK)
    kwargs.setdefault("cluster_swap", False)
    return core.Config(NDIM, NLIVE, **kwargs)


def test_config_resolves_defaults_and_is_hashable() -> None:
    cfg = core.Config(NDIM, NLIVE)
    assert cfg.walks == 25 and cfg.replacement_chains == 1
    assert cfg.block_size == 32 and cfg.cluster_swap is True
    assert core.Config(1, 10).walks == 12
    assert core.Config(5, 10, replacement_chains=2).cluster_swap is False
    assert cfg.max_batches == 10_000 // 25
    assert hash(cfg) == hash(core.Config(NDIM, NLIVE, walks=25, cluster_swap=True))
    tail = dataclasses.replace(cfg, block_size=7)
    assert tail.block_size == 7 and tail.walks == cfg.walks
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.nlive = 3


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"ndim": 0}, "ndim"),
        ({"nlive": 0}, "nlive"),
        ({"walks": 0}, "walks"),
        ({"block_size": 1.5}, "block_size"),
        ({"replacement_chains": True}, "replacement_chains"),
    ],
)
def test_config_rejects_invalid_values(kwargs, match) -> None:
    args = {"ndim": NDIM, "nlive": NLIVE, **kwargs}
    with pytest.raises(ValueError, match=match):
        core.Config(**args)
    with pytest.raises(NotImplementedError, match="cluster_swap"):
        core.Config(NDIM, NLIVE, replacement_chains=2, cluster_swap=True)


def test_init_and_step_shapes_and_pytrees() -> None:
    cfg = make_config()
    state = core.init(random.PRNGKey(0), loglike, prior_transform, cfg)
    assert state.u.shape == state.theta.shape == (NLIVE, NDIM)
    assert state.logl.shape == state.birth.shape == (NLIVE,)
    assert bool(jnp.all(jnp.isinf(state.birth)))
    assert int(state.it) == 0 and int(state.ncall) == NLIVE
    assert not bool(state.failed)
    np.testing.assert_allclose(state.theta, prior_transform(state.u))

    new_state, dead = core.step(state, loglike, prior_transform, cfg)
    for before, after in zip(state, new_state, strict=True):
        assert jnp.shape(before) == jnp.shape(after)
    assert dead.swaps is None
    for name in ("u", "theta"):
        assert getattr(dead, name).shape == (BLOCK, NDIM)
    for name in core.Dead._fields[2:-1]:
        assert getattr(dead, name).shape == (BLOCK,), name
    assert int(new_state.it) == BLOCK
    assert int(new_state.ncall) == NLIVE + int(dead.ncall.sum())
    assert bool(dead.valid.all()) and not bool(new_state.failed)
    assert bool(jnp.all(dead.logl[1:] >= dead.logl[:-1]))  # contours rise
    assert bool(jnp.all(new_state.logl >= dead.logl[-1]))
    # Every replacement was born above the contour it replaced.
    born = new_state.birth > -jnp.inf
    assert 0 < int(born.sum()) <= BLOCK
    assert bool(jnp.all(new_state.logl[born] >= new_state.birth[born]))
    assert float(new_state.logx) == pytest.approx(-BLOCK / NLIVE)

    for tree in (state, new_state, dead):
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(rebuilt) is type(tree)
        for a, b in zip(jax.tree_util.tree_leaves(rebuilt), leaves, strict=True):
            np.testing.assert_array_equal(a, b)
    # Dead without the swap has no leaf for it.
    assert len(jax.tree_util.tree_leaves(dead)) == len(core.Dead._fields) - 1


def test_step_does_not_advance_a_failed_state() -> None:
    cfg = make_config()
    state = core.init(random.PRNGKey(1), loglike, prior_transform, cfg)
    state = state._replace(failed=jnp.asarray(True))
    new_state, dead = core.step(state, loglike, prior_transform, cfg)
    assert not bool(dead.valid.any()) and int(dead.ncall.sum()) == 0
    np.testing.assert_array_equal(new_state.key, state.key)
    np.testing.assert_array_equal(new_state.u, state.u)
    assert int(new_state.it) == 0 and bool(new_state.failed)


def test_repeated_steps_reproduce_nested_sampler_run() -> None:
    nblocks = 3
    key = random.PRNGKey(5)
    result = NestedSampler(
        loglike,
        prior_transform,
        NDIM,
        nlive=NLIVE,
        block_size=BLOCK,
        cluster_swap=False,
    ).run(key, maxiter=nblocks * BLOCK, dlogz=0.0)

    cfg = make_config()
    state = core.init(key, loglike, prior_transform, cfg)
    scale, blocks = core._INITIAL_SCALE, []
    for _ in range(nblocks):
        # The host loop adapts the step scale between blocks.
        state, dead = core.step(
            state._replace(scale=jnp.asarray(scale)), loglike, prior_transform, cfg
        )
        blocks.append(dead)
        scale = _update_scale(scale, int(dead.moves.sum()) / int(dead.proposals.sum()))

    niter = nblocks * BLOCK
    for name, field in (
        ("samples_u", "u"),
        ("samples", "theta"),
        ("logl", "logl"),
        ("logwt", "logwt"),
    ):
        dead_rows = np.concatenate([getattr(dead, field) for dead in blocks])
        np.testing.assert_array_equal(getattr(result, name)[:niter], dead_rows)
    np.testing.assert_array_equal(result.samples_u[niter:], state.u)
    np.testing.assert_array_equal(result.logl[niter:], state.logl)
    assert result.ncall == int(state.ncall)
    assert result.metadata["rwalk_scale_final"] == scale
    insertion = np.concatenate([dead.insertion for dead in blocks])
    np.testing.assert_array_equal(result.metadata["insertion_indices"], insertion)
    assert result.metadata["final_logz_dead"] == float(state.logz)
    assert result.metadata["final_delta_logz"] == core.remaining_dlogz(state)
    assert math.isfinite(core.remaining_dlogz(state))
