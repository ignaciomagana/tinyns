"""Thin modes: the hop's eligibility rule and the frames kept for it
(:func:`tinyns.modes.keep_frames`).

Once a mode is separated, only the hop brings points into it, through an
eligible frame of its cluster. A frame takes part from 3 members whatever its
volume share, and a slot whose cluster falls below 3 members keeps its last
frame for the hop for ``GHOST_RECLUSTERS`` reclusters. The unit tests follow
a kept frame through its life; the invariance test checks that kept frames
leave the constrained prior invariant, and its control that the test sees a
kept frame that depends on the chain's own fold. ``pytest -m slow`` runs the
gate: the 18-D two-mode target whose 20% mode the 1.0 rule lost.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.scipy.special import logsumexp

from tinyns import NestedSampler, core, modes
from tinyns.core import Config

D = 3
CENTERS = np.array([[0.30, 0.35, 0.40], [0.72, 0.62, 0.55]])
RADII = np.array([0.16, 0.11])
MINOR_FRACTION = RADII[1] ** D / np.sum(RADII**D)  # 0.2453
WALKS, SCALE = 40, 0.3


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


def _frames(points, labels):
    st = modes.stats(jnp.asarray(points), jnp.asarray(labels, jnp.int32), modes.C_MAX)
    return modes.frames(st, hop=True)


def test_keep_frames_life_of_a_kept_frame() -> None:
    """A slot whose cluster fell below 3 members keeps its last frame for
    ``GHOST_RECLUSTERS`` reclusters (eligible, not walking), then shows its
    own frame and is not kept again; one that holds 3 members again is an
    ordinary frame; a slot never used keeps nothing."""
    rng = np.random.default_rng(1)
    big = 0.3 + 0.05 * rng.standard_normal((40, D))
    small = 0.7 + 0.01 * rng.standard_normal((6, D))
    before = _frames(np.concatenate([big, small]), [0] * 40 + [1] * 6)
    one = _frames(np.concatenate([big, small[:1]]), [0] * 40 + [1])
    four = _frames(np.concatenate([big, small[:4]]), [0] * 40 + [1] * 4)
    assert bool(before.eligible[1]) and not bool(one.eligible[1])
    assert bool(four.eligible[1])
    ghost = jnp.zeros(modes.C_MAX, jnp.int32)

    frames, ghost = modes.keep_frames(before, ghost, one)
    assert np.asarray(ghost).tolist() == [0, modes.GHOST_RECLUSTERS] + [0] * 6
    for name in ("mu", "chol", "ichol", "logdet"):
        mine = getattr(frames, name)
        np.testing.assert_array_equal(mine[1], getattr(before, name)[1])
        np.testing.assert_array_equal(mine[0], getattr(one, name)[0])
    assert bool(frames.active[1]) and bool(frames.eligible[1])
    assert float(frames.count[1]) == 1 and not bool(modes.walk_frames(frames)[1])
    assert int(jnp.sum(frames.active)) == 2  # the empty slots stay empty
    assert bool(modes.hop_enabled(frames))
    kept = frames
    for left in range(modes.GHOST_RECLUSTERS - 1, 0, -1):
        frames, ghost = modes.keep_frames(frames, ghost, one)
        assert int(ghost[1]) == left
        np.testing.assert_array_equal(frames.mu[1], kept.mu[1])
        assert bool(frames.eligible[1])
    frames, ghost = modes.keep_frames(frames, ghost, one)  # it ends
    assert int(ghost[1]) == 0
    np.testing.assert_array_equal(frames.mu[1], one.mu[1])
    assert not bool(frames.eligible[1]) and not bool(modes.hop_enabled(frames))
    frames, ghost = modes.keep_frames(frames, ghost, one)  # not kept again
    assert int(ghost[1]) == 0 and not bool(frames.eligible[1])

    # Three members again: an ordinary frame, which may be kept later.
    frames, ghost = modes.keep_frames(before, jnp.zeros_like(ghost), one)
    frames, ghost = modes.keep_frames(frames, ghost, one)
    frames, ghost = modes.keep_frames(frames, ghost, four)
    assert int(ghost[1]) == 0
    np.testing.assert_array_equal(frames.mu[1], four.mu[1])
    frames, ghost = modes.keep_frames(frames, ghost, one)
    assert int(ghost[1]) == modes.GHOST_RECLUSTERS
    np.testing.assert_array_equal(frames.mu[1], four.mu[1])


def test_recluster_keeps_the_frame_of_a_cluster_that_died() -> None:
    """Through the sampler's recluster: the thin cluster of
    :func:`test_a_thin_cluster_is_eligible` dies (its slots are refilled in
    the wide one, whose label they take), and each clustering keeps its
    frame for the hop. The frame is not a walk frame and not a cluster of
    the recorded history; a point at its mean takes its label; it ends after
    ``GHOST_RECLUSTERS`` reclusters, and three members of each clustering
    make it an ordinary frame again."""
    state, cfg = _blob_state()
    state = core._recluster(state)
    slots = _thin_slots(state, 18)
    alive = state
    rng = np.random.default_rng(2)
    refill = 0.35 + 0.04 * rng.standard_normal((18, cfg.ndim))
    lab = np.asarray(state.label)[:, 18:]
    wide = [int(row[row >= 0][0]) for row in lab]  # the wide cluster's slot
    label = state.label.at[:, :18].set(
        jnp.where(state.label[:, :18] >= 0, jnp.asarray(wide, jnp.int32)[:, None], -1)
    )
    u = state.u.at[:18].set(refill.astype(state.u.dtype))
    dead = state._replace(u=u, logl=jax.vmap(_bowl)(u), label=label)
    state = core._recluster(dead)
    for j, c in enumerate(slots):
        assert int(state.mode_ghost[j, c]) == modes.GHOST_RECLUSTERS
        assert float(state.mode_count[j, c]) == 0
        np.testing.assert_array_equal(state.mode_mu[j, c], alive.mode_mu[j, c])
        np.testing.assert_array_equal(state.mode_chol[j, c], alive.mode_chol[j, c])
        assert bool(state.mode_active[j, c]) and bool(state.mode_eligible[j, c])
        frames = jax.tree_util.tree_map(lambda a, j=j: a[j], core._frames(state))
        assert not bool(modes.walk_frames(frames)[c])
        assert int(modes.nearest(frames, state.mode_mu[j, c][None])[0]) == c
    assert int(core._kernel_level(state, cfg)) > 0  # the hop reads it

    _, rows = core._step(state, _bowl, lambda v: v, cfg)
    assert np.asarray(rows.nclusters).tolist() == [1] * cfg.num_delete
    assert np.asarray(rows.neligible).tolist() == [2] * cfg.num_delete

    kept = state
    for left in range(modes.GHOST_RECLUSTERS - 1, 0, -1):
        state = core._recluster(state)
        for j, c in enumerate(slots):
            assert int(state.mode_ghost[j, c]) == left
            np.testing.assert_array_equal(state.mode_mu[j, c], kept.mode_mu[j, c])
        if left == 4:
            back = state
    state = core._recluster(state)
    for j, c in enumerate(slots):
        assert int(state.mode_ghost[j, c]) == 0
        assert not bool(state.mode_active[j, c])

    # Nine points back where the thin cluster was, labelled with its slot in
    # every clustering but their own fold's (as the new points of a hop
    # into the kept frame are): six members in each clustering.
    near = 0.75 + 0.004 * rng.standard_normal((9, cfg.ndim))
    label = back.label.at[:, 20:29].set(
        jnp.where(back.label[:, 20:29] >= 0, jnp.asarray(slots, jnp.int32)[:, None], -1)
    )
    back = back._replace(u=back.u.at[20:29].set(near.astype(back.u.dtype)),
                         label=label)
    back = core._recluster(back)
    for j, c in enumerate(slots):
        assert int(back.mode_ghost[j, c]) == 0
        assert float(back.mode_count[j, c]) == 6
        assert bool(back.mode_eligible[j, c])
        frames = jax.tree_util.tree_map(lambda a, j=j: a[j], core._frames(back))
        assert bool(modes.walk_frames(frames)[c])


def test_a_kept_slot_that_holds_points_is_a_cluster() -> None:
    """Eight narrow groups fill the ``C_MAX = 8`` slots; then one group falls
    to 2 points per clustering, below the 3 a frame needs, and its slot
    keeps its frame. The slot still holds points, so no split can take it:
    the recorded history counts 8 clusters (``mode_slots_full`` sees the
    slots full, as 1.0 did) and 8 eligible frames (1.0 left the group's out
    of the hop)."""
    d, per = 4, 36
    rng = np.random.default_rng(0)
    centers = rng.uniform(0.15, 0.85, (8, d))
    while min(np.linalg.norm(a - b) for i, a in enumerate(centers)
              for b in centers[i + 1:]) < 0.2:
        centers = rng.uniform(0.15, 0.85, (8, d))
    u = np.concatenate([c + 0.01 * rng.standard_normal((per, d)) for c in centers])
    group = np.repeat(np.arange(8), per)
    cfg = Config(d, 8 * per, 10, 20)
    state = core.init(0, _bowl, lambda v: v, cfg)
    x = jnp.asarray(u, state.u.dtype)
    state = core._recluster(state._replace(u=x, logl=jax.vmap(_bowl)(x)))
    assert np.asarray(jnp.sum(state.mode_active, axis=1)).tolist() == [8, 8, 8]

    # Group 7 keeps one point per fold; the others join group 0 and take its
    # label in every clustering but their own fold's.
    fold = np.arange(cfg.nlive) % cfg._folds
    seven = np.flatnonzero(group == 7)
    stay = [seven[fold[seven] == f][0] for f in range(cfg._folds)]
    move = np.setdiff1d(seven, stay)
    label = np.asarray(state.label).copy()
    for j in range(cfg._folds):
        zero = label[j, np.flatnonzero((group == 0) & (fold != j))[0]]
        label[j, move] = np.where(fold[move] == j, -1, zero)
    u[move] = centers[0] + 0.01 * rng.standard_normal((len(move), d))
    x = jnp.asarray(u, state.u.dtype)
    state = core._recluster(state._replace(
        u=x, logl=jax.vmap(_bowl)(x), label=jnp.asarray(label, jnp.int32)
    ))
    _, rows = core._step(state, _bowl, lambda v: v, cfg)
    assert int(rows.nclusters[0]) == modes.C_MAX
    assert int(rows.neligible[0]) == modes.C_MAX
    kept = np.asarray(state.mode_ghost) > 0
    assert kept.sum(axis=1).tolist() == [1, 1, 1]
    assert np.asarray(state.mode_count)[kept].tolist() == [2.0, 2.0, 2.0]


def region_loglike(u):
    """0 inside either ball, -1 outside; the contour is L* = -0.5."""
    r2 = jnp.sum((u[None, :] - jnp.asarray(CENTERS)) ** 2, axis=1)
    return jnp.where(jnp.any(r2 <= jnp.asarray(RADII) ** 2), 0.0, -1.0)


def uniform_region(rng, n, minor=MINOR_FRACTION):
    ball = rng.choice(2, size=n, p=[1 - minor, minor])
    z = rng.standard_normal((n, D))
    z *= (rng.uniform(size=(n, 1)) ** (1 / D)) / np.linalg.norm(z, axis=1)[:, None]
    return CENTERS[ball] + RADII[ball][:, None] * z


def _kept_minor_frame(dtype):
    """Frames of a two-ball live set whose minor ball has died out: the
    major ball's cluster, and the minor ball's frame kept from the recluster
    before; the live covariance of the points left."""
    rng = np.random.default_rng(4)
    before = jnp.asarray(uniform_region(rng, 60), dtype)
    labels, st = modes.recluster(before, jnp.zeros(60, jnp.int32))
    assert int(jnp.sum(st.count > 0)) == 2, "the live set must split"
    fit = modes.frames(st, hop=True)
    minor = int(labels[int(np.argmin(
        np.sum((np.asarray(before) - CENTERS[1]) ** 2, axis=1)
    ))])
    after = jnp.asarray(uniform_region(rng, 60, minor=0.0), dtype)
    new = modes.frames(modes.stats(after, modes.nearest(fit, after), modes.C_MAX),
                       hop=True)
    frames, ghost = modes.keep_frames(fit, jnp.zeros(modes.C_MAX, jnp.int32), new)
    assert int(ghost[minor]) == modes.GHOST_RECLUSTERS
    assert int(jnp.sum(ghost > 0)) == 1 and int(jnp.sum(frames.eligible)) == 2
    assert int(jnp.sum(modes.walk_frames(frames))) == 1
    return frames, new, core._live_chol(after, jnp.ones(60, bool))


def _ball_chains(n, frames_of, seed):
    """Chains from ``n`` seeds uniform in the two balls with the frames
    ``frames_of(seed)``: the z-score of the end points' minor fraction
    against the minor ball's volume share, and the fraction of chains that
    changed ball."""
    dtype = jnp.result_type(float)
    rng = np.random.default_rng(seed)
    seeds = jnp.asarray(uniform_region(rng, n), dtype)

    def chain(key, s):
        frames, chol = frames_of(s)
        u = core._chain(
            key, s, jnp.zeros((), dtype), jnp.asarray(-0.5, dtype), chol,
            jnp.asarray(SCALE, dtype), region_loglike, lambda v: v, WALKS, False,
            frames, True, True,
        )[0]
        return jnp.sum((u - CENTERS[1]) ** 2) <= RADII[1] ** 2

    end = np.asarray(jax.jit(jax.vmap(chain))(random.split(random.PRNGKey(seed), n),
                                              seeds))
    start = np.sum((np.asarray(seeds) - CENTERS[1]) ** 2, axis=1) <= RADII[1] ** 2
    z = (end.mean() - MINOR_FRACTION) / math.sqrt(
        MINOR_FRACTION * (1 - MINOR_FRACTION) / n
    )
    return z, np.mean(start != end)


def test_hop_with_a_kept_frame_keeps_the_constrained_prior() -> None:
    """The minor ball has no live point left outside the chains' fold, and
    only its kept frame lets the hop reach it. Chains seeded uniformly in
    both balls end uniformly in both, and many change ball. The control
    keeps the frame only for chains seeded in the minor ball, as a frame
    kept while the chain's own fold still holds points there would be: the
    hop then carries chains out of the minor ball and none into it (z of -58
    in float32 and x64, against -0.6 with the frame kept for every chain)."""
    dtype = jnp.result_type(float)
    frames, without, chol = _kept_minor_frame(dtype)
    n = 20_000
    z, moved = _ball_chains(n, lambda s: (frames, chol), seed=5)
    assert abs(z) < 4.0, z
    assert moved > 0.1, moved

    def own_fold(s):  # the control
        mine = jnp.sum((s - jnp.asarray(CENTERS[1], dtype)) ** 2) <= RADII[1] ** 2
        pick = jax.tree_util.tree_map(
            lambda a, b: jnp.where(mine, a, b), frames, without
        )
        return pick, chol

    z, _ = _ball_chains(n, own_fold, seed=5)
    assert z < -10.0, z


def two_gaussians(theta):
    """Masses 0.75 and 0.25, widths 0.03 and 0.012: the light mode's peak
    is 1.65 higher."""

    def log_normal(center, sigma):
        z = (theta - center) / sigma
        return -0.5 * jnp.sum(z * z) - D * math.log(sigma * math.sqrt(2 * math.pi))

    return jnp.logaddexp(
        math.log(0.75) + log_normal(jnp.asarray(CENTERS[0]), 0.03),
        math.log(0.25) + log_normal(jnp.asarray(CENTERS[1]), 0.012),
    )


def test_a_run_keeps_the_frame_of_a_mode_that_dies() -> None:
    """In a run on two Gaussians the heavier, lower mode dies out while the
    other's posterior bulk is still ahead, and its frame is kept for the
    hop: the recorded history then shows more eligible frames than clusters,
    which the 1.0 rule never did. A unimodal run keeps no frame."""
    cfg = Config(D, 200, 20, 20)
    result = core.run(3, two_gaussians, lambda u: u, cfg)
    history = result.metadata["mode_history"]
    assert any(eligible > clusters for _, clusters, eligible in history), history
    minor = np.asarray(result.samples)[:, 0] > 0.5
    w = np.asarray(result.weights())
    assert 0.1 < w[minor].sum() / w.sum() < 0.45  # truth 0.25

    def gaussian(theta):
        return -0.5 * jnp.sum(((theta - 0.5) / 0.05) ** 2)

    result = core.run(3, gaussian, lambda u: u, Config(D, 200, 20, 20))
    history = result.metadata["mode_history"]
    assert all(eligible <= clusters for _, clusters, eligible in history), history


@pytest.mark.slow
@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the gate is in x64")
def test_a_thin_18d_mode_is_kept() -> None:
    """The gate: ``build_two_mode(18, 0.2)`` of ``bench.targets`` (shape M:
    the 20% mode's volume is 0.035 of the main one's), 32 batched runs at
    nlive 1000 in x64. The 20% mode is split off at deaths 23,000 to 30,000
    with 27 to 47 live points and lives as a cluster of 5 to 30 for 15 to 20
    e-folds; the 1.0 rule left such a cluster out of the hop, and the mode
    drifted to none. A run fails if it lost the mode (an oracle mass below
    0.02: the target's responsibility averaged over the posterior) or got
    its mass more than 0.5 off in logit. On these keys (CPU, JAX 0.10.2) the
    1.0 rule failed 9 runs (6 lost) and this code none; over 128 runs (keys
    10 to 13) the 1.0 rule lost 11 and was off in 7 more, against 1 lost and
    none off here. (In float32 these keys lost 1 run and put none off; the
    gate was set from the x64 runs.)"""
    from bench.targets import build_two_mode, mixture_target

    target = mixture_target("plainM_d18", build_two_mode(18, 0.2))
    sampler = NestedSampler(target.loglike, target.prior_transform, 18)
    responsibility = jax.jit(jax.vmap(target.responsibility))
    truth = target.mode_mass[1]
    keys = jax.random.split(jax.random.key(10), 32)
    masses, dlogz = [], []
    for result in sampler.run(keys):
        share = np.asarray(responsibility(result.samples))[:, 1]
        masses.append(float(result.weights() @ share / result.weights().sum()))
        dlogz.append(float(result.logz) / float(result.logzerr))
    masses = np.array(masses)
    logit = np.log(masses / (1 - masses)) - math.log(truth / (1 - truth))
    failed = (masses < 0.1 * truth) | (np.abs(logit) > 0.5)
    print(f"failed {int(failed.sum())} of 32 (lost {int(np.sum(masses < 0.02))}); "
          f"masses {np.round(masses, 4).tolist()}; logZ / logzerr "
          f"{np.round(dlogz, 2).tolist()}")
    assert failed.sum() <= 2, np.round(masses, 4).tolist()
