"""Thin modes: the hop's eligibility rule.

Once a mode is separated, only the hop brings points into it, through an
eligible frame of its cluster. A frame takes part from 3 members whatever its
volume share: the 1.0 rule, a volume share that predicts ``2 d`` live points
and more than ``d`` members, left out the thin clusters that need it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from tinyns import core, modes
from tinyns.core import Config


def _old_rule(logdet, count, active, nlive, d):
    """The 1.0 rule: a volume share that predicts ``2 d`` live points and
    more than ``d`` members; the most populated cluster always."""
    logv = jnp.where(active, logdet, -jnp.inf)
    share = jnp.exp(logv - logsumexp(logv))
    eligible = active & (nlive * share >= 2.0 * d) & (count > d)
    largest = jnp.argmax(jnp.where(active, count, -1.0))
    return eligible.at[largest].set(active[largest])


def _union_rule(logdet, count, active, nlive, d):
    """The union rule of the population study: more than ``d`` members, and
    the volume share or at least ``2 d`` members."""
    logv = jnp.where(active, logdet, -jnp.inf)
    share = jnp.exp(logv - logsumexp(logv))
    big = (nlive * share >= 2.0 * d) | (count >= 2.0 * d)
    eligible = active & big & (count > d)
    largest = jnp.argmax(jnp.where(active, count, -1.0))
    return eligible.at[largest].set(active[largest])


def test_eligibility_counts_members_not_volume() -> None:
    """From 3 members (2 in one dimension) whatever the volume share; the
    most populated cluster always. Every frame that the 1.0 rule or the union
    rule (volume share or ``2 d`` members) admits is admitted."""
    on = jnp.asarray([True] * 5 + [False] * 3)

    def eligible(count, d, active=on):
        count = jnp.asarray(count + [0.0] * (modes.C_MAX - len(count)))
        return np.asarray(modes._eligible(count, active, d)).tolist()

    assert eligible([600.0, 3.0, 2.0, 19.0, 1.0], 18)[:5] == [
        True, True, False, True, False
    ]
    assert eligible([2.0, 1.0, 0.0, 0.0, 0.0], 18)[:2] == [True, False]
    assert eligible([50.0, 2.0, 1.0, 0.0, 0.0], 1)[:3] == [True, True, False]
    assert eligible([50.0, 9.0], 4, jnp.asarray([True, False] + [False] * 6)) == [
        True
    ] + [False] * 7

    rng = np.random.default_rng(0)
    for d in (1, 2, 4, 10, 18, 32):
        for _ in range(200):
            active = jnp.asarray(rng.uniform(size=modes.C_MAX) < 0.7)
            count = jnp.asarray(np.floor(rng.uniform(0, 3 * d + 6, modes.C_MAX)))
            count = jnp.where(active, count, 0.0)
            logdet = jnp.asarray(rng.normal(0.0, 3.0 * d, modes.C_MAX))
            nlive = int(rng.integers(50, 4000))
            new = np.asarray(modes._eligible(count, active, d))
            for rule in (_old_rule, _union_rule):
                old = np.asarray(rule(logdet, count, active, nlive, d))
                assert not np.any(old & ~new), (d, rule.__name__)


def _blob_state(m=300, d=4, thin=18, seed=0):
    """A live set of ``m`` points in ``d`` dimensions: a wide Gaussian and
    ``thin`` points of a narrow one far from it (slots 0 to ``thin - 1``,
    spread over the folds)."""
    rng = np.random.default_rng(seed)
    u = 0.35 + 0.04 * rng.standard_normal((m, d))
    u[:thin] = 0.75 + 0.004 * rng.standard_normal((thin, d))
    cfg = Config(d, m)
    state = core.init(0, _bowl, lambda v: v, cfg)
    u = jnp.asarray(u, state.u.dtype)
    return state._replace(u=u, logl=jax.vmap(_bowl)(u)), cfg


def _bowl(x):
    return -jnp.sum(x * x)


def _thin_slots(state, thin):
    """The slot of the thin points in each clustering (they must share
    one)."""
    label = np.asarray(state.label)
    fold = np.arange(label.shape[1]) % label.shape[0]
    slots = []
    for j in range(label.shape[0]):
        mine = label[j, :thin][fold[:thin] != j]
        assert len(set(mine.tolist())) == 1, mine
        slots.append(int(mine[0]))
    return slots


def test_a_thin_cluster_is_eligible() -> None:
    """A narrow group of 18 of 300 live points in 4-D, 12 outside each fold:
    the clustering splits it off, and its volume share predicts far fewer
    than ``2 d = 8`` live points. The 1.0 rule left it out of the hop (so
    nothing but the walk kept it populated); it takes part now."""
    state, cfg = _blob_state()
    state = core._recluster(state)
    slots = _thin_slots(state, 18)
    for j, c in enumerate(slots):
        assert float(state.mode_count[j, c]) == 12
        logv = jnp.where(state.mode_active[j], state.mode_logdet[j], -jnp.inf)
        share = float(jnp.exp(logv[c] - logsumexp(logv)))
        assert cfg.nlive * share < 2 * cfg.ndim  # the old rule's premise
        assert bool(state.mode_eligible[j, c]), j
        assert int(jnp.sum(state.mode_eligible[j])) == 2
