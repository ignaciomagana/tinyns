"""The full-JAX core: Config, init, step, the evidence bookkeeping, the driver."""

from __future__ import annotations

import dataclasses
import math
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import tinyns
from tinyns import STATUS, Config, NestedSampler, core, finalise, init, step
from tinyns.result import _evidence

FLOAT = jnp.result_type(float)


def gauss(center, sigma):
    """Isotropic normal density (normalized over R^d)."""

    def loglike(x):
        z = (x - center) / sigma
        norm = x.shape[0] * math.log(sigma * math.sqrt(2 * math.pi))
        return -0.5 * jnp.sum(z * z) - norm

    return loglike


def identity(u):
    return u


# --- the float64 reference of the bookkeeping -------------------------------


def reference_evidence(logl, nlive_i):
    """Sample by sample, plain Python floats.

    Sample i shrinks log X by 1 / n_i and takes X_{i-1} - X_i; the last sample
    takes all of X_{N-2}. Var(log Z) is first order in the shrinkages: log t_i
    has variance 1 / n_i^2 and moves log Z by (Z_{>i} - L_i X_i) / Z, with
    Z_{>i} the evidence of the samples after i (the last sample: 0).
    """
    logz, log_x = -math.inf, 0.0
    logwts, log_xs = [], []
    for i, (ll, n) in enumerate(zip(logl, nlive_i, strict=True)):
        log_x_new = log_x - 1.0 / n
        if i == len(logl) - 1:
            width = math.exp(log_x)
        else:
            width = math.exp(log_x) - math.exp(log_x_new)
        logwt = ll + math.log(width) if ll > -math.inf else -math.inf
        logwts.append(logwt)
        log_xs.append(log_x_new)
        if logwt > -math.inf:
            logz = float(np.logaddexp(logz, logwt))
        log_x = log_x_new
    var, after = 0.0, 0.0  # after: Z_{>i} / Z
    for i in reversed(range(len(logl))):
        if i < len(logl) - 1:
            own = math.exp(logl[i] + log_xs[i] - logz) if logl[i] > -math.inf else 0.0
            var += ((after - own) / nlive_i[i]) ** 2
        if logwts[i] > -math.inf:
            after += math.exp(logwts[i] - logz)
    return np.array(logwts), logz, math.sqrt(var)


FOLDS = Config(1)._folds


def fake_run(dead_logl, live_logl, m, k, nlive=None, **fields):
    """A final State and Dead rows (numpy) with the given likelihoods; the
    dead rows' live counts are ``nlive`` (default: ``m - j``), and ``fields``
    override fields of the state."""
    dead_logl = np.asarray(dead_logl, np.float64).reshape(-1, k)
    steps = dead_logl.shape[0]
    rng = np.random.default_rng(0)
    state = core.State(
        key=np.zeros(2, np.uint32),
        u=rng.uniform(size=(m, 1)),
        logl=np.asarray(live_logl, np.float64),
        logl_birth=np.full(m, -np.inf),
        it=np.int32(steps),
        logz=np.float64(0.0),
        log_x_ties=np.float64(0.0),
        lstar_prev=np.float64(-np.inf),
        tie_run=np.int32(0),
        log_scale=np.float64(0.0),
        ncall=np.int32(0),
        ncall_valid=np.int32(0),
        status=np.int32(core.CONVERGED),
        label=np.zeros((FOLDS, m), np.int32),
        **core._empty_frames(1, FOLDS, FLOAT),
    )._replace(**fields)
    if nlive is None:
        nlive = np.tile(m - np.arange(k, dtype=np.int32), (steps, 1))
    zeros = np.zeros((steps, k), np.int32)
    dead = core.Dead(
        u=rng.uniform(size=(steps, k, 1)),
        logl=dead_logl,
        logl_birth=np.full((steps, k), -np.inf),
        insertion=zeros,
        moves=np.ones((steps, k), np.int32),
        label=zeros,
        hops=zeros,
        hop_tries=zeros,
        nclusters=zeros,
        neligible=zeros,
        nlive=np.asarray(nlive, np.int32).reshape(steps, k),
        nslots=np.full((steps, k), m - k + 1, np.int32),
    )
    return state, dead


@pytest.mark.parametrize("k", [1, 3, 10])
def test_constant_likelihood_bookkeeping_is_exact(k) -> None:
    m, c = 20, -1.7
    cfg = Config(1, m, num_delete=k)
    state, dead = fake_run(np.full(6 * k, c), np.full(m, c), m, k)
    result = finalise(state, dead, cfg)
    logwt, logz, logzerr = reference_evidence(result.logl, result.nlive_i)
    assert result.logz == pytest.approx(c, abs=1e-12)  # the widths sum to 1
    assert abs(result.logz - logz) < 1e-6
    assert abs(result.logzerr - logzerr) < 1e-6
    np.testing.assert_allclose(result.logwt, logwt, rtol=0, atol=1e-6)
    expected = np.concatenate([np.tile(m - np.arange(k), 6), m - np.arange(m)])
    np.testing.assert_array_equal(result.nlive_i, expected)


def test_live_points_tied_at_the_last_contour_count_on_from_the_plateau() -> None:
    """An indicator likelihood (0 on part of the cube, -inf elsewhere) that
    ended on its plateau: 3 steps of k = 4 tied deaths at -inf, and the r = 2
    points still tied there stay live (the step that would delete them finds
    nothing above L* = 0). They die at counts 8, 7, after the 12 dead rows at
    20..9, not at 20, 19 (logZ was log((m - kT)(m - r) / m^2) for the
    truth log((m - kT - r) / m), up to 1.5 nats high at the defaults). The
    q = 14 tied deaths leave the volume X_q, held by the 18 points at L = 1,
    so Z = X_q exactly, and the shrinkages of the tied deaths alone carry
    the error: logzerr^2 = sum 1 / n^2 over their counts (it was 0)."""
    m, k = 20, 4
    cfg = Config(1, m, num_delete=k)
    counts = m - np.arange(12)  # the FHS counts of the 12 tied dead rows
    live = np.concatenate([[-np.inf, -np.inf], np.zeros(m - 2)])
    state, dead = fake_run(
        np.full(12, -np.inf), live, m, k, nlive=counts,
        lstar_prev=np.float64(-np.inf), tie_run=np.int32(3),
    )
    result = finalise(state, dead, cfg)
    tied = np.arange(m, m - 14, -1)
    np.testing.assert_array_equal(
        result.nlive_i, np.concatenate([tied, m - 2 - np.arange(m - 2)])
    )
    assert result.logz == pytest.approx(-np.sum(1.0 / tied), abs=1e-12)
    assert result.logzerr == pytest.approx(math.sqrt(np.sum(1.0 / tied**2)), rel=1e-9)
    logwt, logz, logzerr = reference_evidence(result.logl, result.nlive_i)
    assert abs(result.logz - logz) < 1e-9 and abs(result.logzerr - logzerr) < 1e-9
    # Before the first step (tie_run 0) nothing counts as tied, whatever the
    # live likelihoods.
    state, dead = fake_run(np.zeros(0), live, m, k, it=np.int32(0))
    np.testing.assert_array_equal(finalise(state, dead, cfg).nlive_i, m - np.arange(m))


def test_logzerr_is_the_first_order_volume_spread() -> None:
    """logzerr^2 = sum_j (S_j / n_j)^2, S_j = d logZ / d log t_j: the
    shrinkages of tied deaths below the posterior mass (zero weight) count,
    and it is never negative. The sum of information increments gave 0 on an
    indicator likelihood (clipped from a negative sum) while logZ scattered
    by the binomial sd sqrt((1 - V) / (m V))."""
    m, q = 400, 300  # 300 tied deaths at -inf, 100 points left on the plateau
    logl = np.concatenate([np.full(q, -np.inf), np.zeros(m - q)])
    nlive = np.concatenate([m - np.arange(q), m - q - np.arange(m - q)])
    _, logz, logzerr = _evidence(logl, nlive)
    n = m - np.arange(q)
    assert logz == pytest.approx(-np.sum(1.0 / n), abs=1e-12)
    assert logzerr == pytest.approx(math.sqrt(np.sum(1.0 / n**2)), rel=1e-12)
    assert logzerr == pytest.approx(math.sqrt((1 - 0.25) / (m * 0.25)), rel=0.01)
    # The same deaths with a finite plateau below the mass: L = -30 there.
    _, _, err_low = _evidence(np.where(logl == -np.inf, -30.0, 0.0), nlive)
    assert err_low == pytest.approx(logzerr, rel=1e-6)
    # Without ties, at a constant count (an ideal run, deterministic volumes),
    # it is close to Skilling's sqrt(H / n).
    m = 500
    log_x = -np.arange(1, 20 * m + 1) / m
    logl = -0.5 * np.exp(2 * log_x / 3) / 1e-2  # a 3-D Gaussian of sd 0.1
    logwt, logz, logzerr = _evidence(logl, np.full(logl.size, m))
    p = np.exp(logwt - logz)
    info = float(np.sum(p * (logl - logz)))
    assert logzerr == pytest.approx(math.sqrt(info / m), rel=0.05)


@pytest.mark.parametrize("k", [1, 10])
def test_gaussian_1d_bookkeeping_matches_reference_and_truth(k) -> None:
    sampler = NestedSampler(gauss(0.0, 1.0), lambda u: 20.0 * u - 10.0, 1, 200,
                            num_delete=k)
    logzs, errs = [], []
    for seed in range(4):
        result = sampler.run(seed)
        logwt, logz, logzerr = reference_evidence(result.logl, result.nlive_i)
        assert abs(result.logz - logz) < 1e-6
        assert abs(result.logzerr - logzerr) < 1e-6
        np.testing.assert_allclose(result.logwt, logwt, rtol=0, atol=1e-6)
        logzs.append(result.logz)
        errs.append(result.logzerr)
    # Z = 1/20; four runs: the mean is within 3 of its standard errors.
    assert abs(np.mean(logzs) + math.log(20.0)) < 3 * np.mean(errs) / 2 + 0.02
    # logzerr is close to sqrt(H / m); the final live points (counts m..1)
    # move it by some 10% in one dimension.
    info = result.information()
    assert result.logzerr == pytest.approx(math.sqrt(info / 200), rel=0.2)


def test_log_widths_and_shrink_constants() -> None:
    cfg = Config(2, 10, num_delete=3)
    assert cfg.log_shrink == pytest.approx(1 / 10 + 1 / 9 + 1 / 8)
    x = np.exp(-np.cumsum([1 / 10, 1 / 9, 1 / 8]))
    widths = np.diff(np.concatenate([[1.0], x])) * -1
    np.testing.assert_allclose(np.exp(cfg._log_widths()), widths, rtol=1e-12)


# --- Config ------------------------------------------------------------------


def test_config_resolves_defaults_and_is_hashable() -> None:
    cfg = Config(3)
    assert (cfg.nlive, cfg.num_delete, cfg.walks) == (1000, 100, 25)
    assert Config(10, 500).walks == 60
    # walks = max(25, 6 ndim, ndim^2 // 6), whatever nlive is
    assert Config(32).walks == 192 and Config(36, 500).walks == 216
    assert (Config(48).walks, Config(64, 2000).walks) == (384, 682)
    assert Config(64, walks=10).walks == 10
    assert Config(3, 15).num_delete == 1
    assert hash(cfg) == hash(Config(3, 1000, 100, 25)) and cfg == Config(3)
    with pytest.raises(AttributeError):
        cfg.nlive = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"ndim": 0}, ValueError),
        ({"nlive": 1}, ValueError),
        ({"num_delete": 0}, ValueError),
        ({"num_delete": 6}, ValueError),  # > nlive // 2
        ({"walks": 0}, ValueError),
        ({"nlive": 10.0}, TypeError),
        ({"walks": True}, TypeError),
    ],
)
def test_config_rejects_invalid_values(kwargs, error) -> None:
    with pytest.raises(error):
        Config(**{"ndim": 2, "nlive": 10, **kwargs})


def test_config_warns_when_the_walk_covariances_cannot_span_ndim() -> None:
    """A fold's walk covariance is fitted to the other two folds' points,
    about 2/3 of nlive: below 1.5 (ndim + 1) they span fewer than ndim
    directions (on a 32-D Gaussian logZ came out 9 nats high at nlive 40)."""
    with pytest.warns(UserWarning, match="nlive=40 is too few for ndim=32"):
        Config(32, 40)
    with pytest.warns(UserWarning, match="below 5 live points"):
        Config(2, 4, num_delete=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Config(32, 50)  # 1.5 * 33 = 49.5
        Config(2, 5)


def test_the_nlive_warning_names_the_users_line() -> None:
    """The warning points at the line that built the ``Config`` or the
    ``NestedSampler``, not at tinyns or the stdlib (it named
    ``tinyns/api.py`` for a ``NestedSampler`` and ``dataclasses.py`` for a
    ``Config`` re-made with ``dataclasses.replace``)."""
    for build in (
        lambda: Config(32, 40),
        lambda: dataclasses.replace(Config(32, 100), nlive=40),
        lambda: NestedSampler(gauss(0.5, 0.1), identity, 32, nlive=40),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build()
        assert [os.path.basename(w.filename) for w in caught] == ["test_core.py"]


def test_mode_tree_counts_the_splits_it_did_not_record() -> None:
    """``metadata["mode_tree"]`` records the parents of the first
    ``TREE_SIZE - 1`` cluster ids of each clustering; ``mode_tree_dropped``
    counts the others (a float32 run whose clustering churned made 3306
    splits and recorded 1533, with no sign of it)."""
    from types import SimpleNamespace

    from tinyns.modes import TREE_SIZE

    result = NestedSampler(gauss(0.5, 0.1), identity, 2, nlive=40).run(0)
    assert result.metadata["mode_tree_dropped"] == 0
    state = SimpleNamespace(
        mode_tree=np.zeros((3, TREE_SIZE), np.int32),
        mode_next=np.array([TREE_SIZE + 88, 3, TREE_SIZE], np.int32),
    )
    assert len(core._mode_tree(state)) == 2 * (TREE_SIZE - 1) + 2
    assert core._mode_tree_dropped(state) == 88


# --- init and step -----------------------------------------------------------


def test_init_and_step_shapes_dtypes_and_order() -> None:
    cfg = Config(3, 40, num_delete=4, walks=10)
    loglike = gauss(0.5, 0.2)
    state = init(0, loglike, identity, cfg)
    assert state.u.shape == (40, 3) and state.u.dtype == FLOAT
    assert state.logl.dtype == FLOAT and state.logz.dtype == FLOAT
    assert state.it.dtype == jnp.int32 and int(state.ncall) == 40
    new, dead = step(state, loglike, identity, cfg)
    assert jax.tree_util.tree_structure(new) == jax.tree_util.tree_structure(state)
    for a, b in zip(jax.tree_util.tree_leaves(new), jax.tree_util.tree_leaves(state),
                    strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
    assert dead.u.shape == (4, 3) and dead.insertion.dtype == jnp.int32
    # The k lowest, by increasing likelihood; L* is the highest of them.
    np.testing.assert_array_equal(dead.logl, np.sort(state.logl)[:4])
    lstar = float(dead.logl[-1])
    assert np.all(np.asarray(new.logl) > lstar)
    born = np.asarray(new.logl_birth) == lstar
    assert born.sum() == 4
    assert int(new.it) == 1 and int(new.status) == core.RUNNING
    insertion = np.asarray(dead.insertion)
    assert np.all((insertion >= 0) & (insertion <= 36))
    # k chains of walks steps, every proposal evaluated (vmapped).
    assert int(new.ncall) == 40 + 4 * 10
    assert int(new.ncall_valid) <= int(new.ncall)


def test_seeds_are_distinct_when_possible() -> None:
    """With an impossible walk (scale ~ 0) every chain returns its seed, so the
    new points are the seeds: live points above L* of the fold of the slot
    they fill, distinct unless that fold has too few."""
    cfg = Config(2, 30, num_delete=10, walks=3)
    loglike = gauss(0.5, 0.2)
    state = init(1, loglike, identity, cfg)
    state = state._replace(log_scale=jnp.asarray(-80.0, FLOAT))
    new, dead = step(state, loglike, identity, cfg)
    u, logl = np.asarray(state.u), np.asarray(state.logl)
    above = logl > float(dead.logl[-1])
    filled = np.flatnonzero(np.asarray(new.logl_birth) > -np.inf)
    seeds = np.array([int(np.flatnonzero((u == v).all(axis=1))[0])
                      for v in np.asarray(new.u)[filled]])
    assert np.all(above[seeds])
    fold = np.arange(30) % cfg._folds
    for j in range(cfg._folds):
        mine = seeds[fold[filled] == j]
        assert np.all(fold[mine] == j)
        assert len(set(mine.tolist())) == min(len(mine), int(above[fold == j].sum()))


@pytest.mark.parametrize("k", [1, 6])
def test_seeds_and_walk_covariance_are_cross_fitted(monkeypatch, k) -> None:
    """Each new point is seeded from a live point of its own fold (the fold of
    the dead slot it fills), and the walk covariance of the chains seeded in
    fold j comes from the points outside fold j above L*. With an impossible
    walk the new points are the seeds."""
    masks = []
    live_chol = core._live_chol

    def spy(u, mask):
        masks.append(np.asarray(mask))
        return live_chol(u, mask)

    monkeypatch.setattr(core, "_live_chol", spy)
    cfg = Config(2, 30, num_delete=k, walks=3)
    loglike = gauss(0.5, 0.2)
    state = init(1, loglike, identity, cfg)
    state = state._replace(log_scale=jnp.asarray(-80.0, FLOAT))
    new, dead = core._step(state, loglike, identity, cfg)  # eager: the spy runs
    u = np.asarray(state.u)
    above = np.asarray(state.logl) > float(dead.logl[-1])
    fold = np.arange(30) % cfg._folds
    assert len(masks) == cfg._folds
    for j, mask in enumerate(masks):
        np.testing.assert_array_equal(mask, above & (fold != j))
    filled = np.flatnonzero(np.asarray(new.logl_birth) > -np.inf)
    seeds = [int(np.flatnonzero((u == v).all(axis=1))[0])
             for v in np.asarray(new.u)[filled]]
    assert len(set(seeds)) == k and all(above[seeds])
    np.testing.assert_array_equal(fold[seeds], fold[filled])


@pytest.mark.parametrize("k", [1, 12])
def test_each_chain_walks_with_the_kernel_of_its_seeds_fold(monkeypatch, k) -> None:
    """The chain seeded in fold j walks with the covariance fitted outside
    fold j. Here fold j's factor moves coordinate j alone and the likelihood
    reads the last coordinate only, so the one walk step is always accepted
    and the coordinate a new point moved along names the kernel its chain
    read."""
    d, m = 4, 60
    cfg = Config(d, m, num_delete=k, walks=1)
    folds = cfg._folds
    fold = np.arange(m) % folds
    assert folds < d

    def marker(u, mask):  # the fold left out of the mask is the kernel's
        j = jnp.argmin(jnp.stack([jnp.sum(mask & (fold == i)) for i in range(folds)]))
        e = jnp.eye(d, dtype=u.dtype)[j]
        return 1e-2 * e[:, None] * e[None, :]

    monkeypatch.setattr(core, "_live_chol", marker)

    def loglike(x):
        return -((x[-1] - 0.5) ** 2)

    # Points in [0.2, 0.8]^d: no step of the walk leaves the cube.
    u = jnp.asarray(np.random.default_rng(1).uniform(0.2, 0.8, (m, d)), FLOAT)
    state = init(0, loglike, identity, cfg)
    state = state._replace(u=u, logl=jax.vmap(loglike)(u))
    new, _ = core._step(state, loglike, identity, cfg)
    u = np.asarray(u)
    filled = np.flatnonzero(np.asarray(new.logl_birth) > -np.inf)
    assert len(filled) == k
    read = []  # (the fold of the seed, the coordinate its chain moved)
    for slot in filled:
        same = u == np.asarray(new.u)[slot]
        seed = np.flatnonzero(same.sum(axis=1) == d - 1)
        assert len(seed) == 1  # all coordinates but the moved one agree
        assert fold[seed[0]] == fold[slot]
        read.append((int(fold[seed[0]]), int(np.flatnonzero(~same[seed[0]])[0])))
    assert all(j == moved for j, moved in read), read
    if k > 1:
        assert {j for j, _ in read} == set(range(folds))


def test_seeds_fall_back_when_a_fold_runs_short() -> None:
    """A fold with too few points above L* reuses them; a fold with none
    borrows from the others (two folds: the parity of the slot)."""
    above = jnp.asarray([True, False, True, False, False, False, True, False])
    worst = jnp.asarray([1, 3, 5, 4])  # three odd slots, one even
    seeds = np.asarray(core._seeds(
        random.PRNGKey(0), random.PRNGKey(1), above, worst, 2, FLOAT
    ))
    assert seeds[3] in (0, 2, 6)  # the even slot: an even point above L*
    assert all(above[np.asarray(seeds)])  # odd slots: none above, any fold
    above = jnp.asarray([True, True, True, False, False, False, True, False])
    seeds = np.asarray(core._seeds(
        random.PRNGKey(0), random.PRNGKey(1), above, worst, 2, FLOAT
    ))
    assert set(seeds[:3].tolist()) == {1}  # the only odd point, reused
    # Seeds drawn with replacement are uniform on the fold's points above L*
    # (here the borrowed ones: slots 0, 2 and 6 for the three odd deaths).
    above = jnp.asarray([True, False, True, False, False, False, True, False])
    draw = jax.jit(jax.vmap(
        lambda key: core._seeds(random.PRNGKey(0), key, above, worst, 2, FLOAT)
    ))
    seeds = np.asarray(draw(random.split(random.PRNGKey(2), 3000)))[:, :3]
    counts = np.array([(seeds == slot).sum() for slot in (0, 2, 6)])
    assert counts.sum() == seeds.size
    assert np.all(np.abs(counts - 3000) < 4 * np.sqrt(9000 * (1 / 3) * (2 / 3)))


def test_walk_mask_falls_back_when_too_few_points_remain() -> None:
    """Fewer than ndim + 1 points of the other folds above L*: every point of
    the other folds (still cross-fitted, of full rank); all live points only
    when even those are too few."""
    above = jnp.asarray([True] * 6 + [False] * 4)
    other = jnp.asarray([True, False] * 5)  # 3 even points above L*, 5 in all
    np.testing.assert_array_equal(
        core._walk_mask(above, other, 2), np.asarray(above & other)
    )
    np.testing.assert_array_equal(core._walk_mask(above, other, 3), np.asarray(other))
    np.testing.assert_array_equal(core._walk_mask(above, other, 4), np.asarray(other))
    np.testing.assert_array_equal(core._walk_mask(above, other, 5), np.ones(10, bool))
    # A step on a nearly exhausted live set still runs: 2 points above L*,
    # both seeds, in 2-D.
    with pytest.warns(UserWarning, match="too few for ndim=2"):
        cfg = Config(2, 4, num_delete=2, walks=5)
    loglike = gauss(0.5, 0.2)
    state = init(0, loglike, identity, cfg)
    new, _ = step(state, loglike, identity, cfg)
    assert np.all(np.isfinite(np.asarray(new.u))) and int(new.status) == core.RUNNING


def test_out_of_cube_proposals_are_skipped_at_k1_and_counted_at_k_gt_1() -> None:
    loglike = gauss(0.02, 0.05)  # at a corner: many proposals leave the cube
    for k in (1, 8):
        cfg = Config(2, 40, num_delete=k, walks=12)
        state = init(2, loglike, identity, cfg)
        steps = 0
        for _ in range(64 // k):
            state, _ = step(state, loglike, identity, cfg)
            steps += 1
        ncall, valid = int(state.ncall) - 40, int(state.ncall_valid) - 40
        if k == 1:  # lax.cond: an out-of-cube proposal costs nothing
            assert ncall == valid < steps * 12
        else:  # vmapped: every lane is evaluated; ncall_valid counts those
            # passing the likelihood-free tests (here the in-cube ones)
            assert ncall == steps * k * 12 and valid < ncall


def test_nan_likelihood_counts_as_minus_infinity() -> None:
    base = gauss(0.5, 0.05)

    def loglike(x):
        return jnp.where(x[0] < 0.2, jnp.nan, base(x))

    result = NestedSampler(loglike, identity, 2, 100, num_delete=10).run(3)
    assert np.all(~np.isnan(result.logl))
    assert abs(result.logz) < 4 * result.logzerr + 0.05  # the density integrates to 1


def test_scale_adapts_and_is_clamped() -> None:
    cfg = Config(2, 40, num_delete=4, walks=10)
    flat = gauss(0.5, 30.0)  # nearly flat: in-cube moves are accepted
    state = init(0, flat, identity, cfg)
    new, _ = step(state, flat, identity, cfg)
    assert float(new.log_scale) > float(state.log_scale)
    top = state._replace(log_scale=jnp.asarray(math.log(10.0), FLOAT))
    new, _ = step(top, flat, identity, Config(2, 40, num_delete=4, walks=10))
    assert float(new.log_scale) <= math.log(10.0) + 1e-6


# --- plateaus and termination ------------------------------------------------


def test_constant_likelihood_is_a_plateau_with_exact_evidence() -> None:
    result = NestedSampler(lambda x: -2.5, identity, 2, 30, num_delete=3).run(0)
    assert result.metadata["status"] == "plateau" and result.success
    assert result.niter == 0
    assert result.logz == pytest.approx(-2.5, abs=1e-6)
    assert result.logzerr < 1e-12  # the widths sum to 1 exactly


@pytest.mark.parametrize("value", [-jnp.inf, jnp.nan])
def test_no_finite_likelihood_is_not_a_success(value) -> None:
    """No initial point has a finite likelihood: step 0 finds nothing above
    L* = -inf. It was a plateau with success=True and logz = -inf, and
    summary(), print(result) and diagnostics() then raised."""
    sampler = NestedSampler(lambda x: value * x[0], identity, 2, 50, num_delete=5)
    with pytest.warns(RuntimeWarning, match="no live point has a finite likelihood"):
        result = sampler.run(0)
    assert not result.success and result.metadata["status"] == "invalid"
    assert result.logz == -math.inf and result.niter == 0
    assert "raise nlive" in result.message
    assert "success: False" in str(result) and result.message in result.summary()
    diagnostics = result.diagnostics()
    assert diagnostics["posterior_ess"] == 0.0 and result.posterior_ess() == 0.0
    assert math.isnan(diagnostics["live_weight_fraction"])
    assert diagnostics["warnings"][0] == result.message
    with pytest.raises(ValueError, match="no sample has a positive posterior weight"):
        result.resample_equal(0)
    with pytest.raises(ValueError, match="no sample has a positive posterior weight"):
        result.weights()


def spike(x):
    """+inf on a disc of area 0.03 at the peak of a Gaussian."""
    r2 = jnp.sum((x - 0.5) ** 2)
    return jnp.where(r2 < 0.01, jnp.inf, -0.5 * r2 / 0.1**2)


def test_plus_infinite_likelihood_stops_the_run() -> None:
    """A likelihood of +inf (a float32 overflow, say) never dies: the run went
    on to a plateau of +inf points and reported success with logz = +inf.
    The first step after which a +inf point is live now stops it."""
    cfg = Config(2, 60, num_delete=6, walks=10)
    with pytest.warns(RuntimeWarning, match=r"loglike returned \+inf"):
        result = core.run(1, spike, identity, cfg)
    assert not result.success and result.metadata["status"] == "invalid"
    assert result.logz == math.inf and "+inf" in result.message
    # Stopped at once: at most the initial +inf points and one step's births.
    n_inf = int(np.sum(np.isposinf(result.logl)))
    at_init = int(np.sum(np.isposinf(init(1, spike, identity, cfg).logl)))
    assert 1 <= n_inf <= at_init + 6
    warnings = result.diagnostics()["warnings"]
    assert warnings[0] == result.message
    # No advice on dlogz or nlive, which do not mend a likelihood of +inf.
    stopping = ("consider tighter dlogz", "final-live", "final_delta_logz")
    assert not any(text in w for w in warnings for text in stopping), warnings
    assert "success: False" in str(result)
    # The functional core: step sets the status "invalid", its rows are valid.
    state = init(1, spike, identity, cfg)
    while STATUS[int(state.status)] == "running":
        state, dead = step(state, spike, identity, cfg)
    assert STATUS[int(state.status)] == "invalid" and int(state.it) == result.niter // 6
    assert np.any(np.isposinf(state.logl)) and not np.any(np.isposinf(dead.logl))


def test_an_invalid_lane_stops_alone() -> None:
    """One lane of a batch without a finite likelihood and one with a +inf
    region stop as invalid; the run does not raise and the other lane
    converges as it would alone."""

    def loglike(data, x):  # data: (offset, +inf disc on or off)
        r2 = jnp.sum((x - 0.5) ** 2)
        inf = (r2 < 0.01) & (data[1] > 0)
        return jnp.where(inf, jnp.inf, data[0] - 0.5 * r2 / 0.1**2)

    data = jnp.asarray([[0.0, 0.0], [-jnp.inf, 0.0], [0.0, 1.0]])
    sampler = NestedSampler(jax.tree_util.Partial(loglike, data), identity, 2, 60,
                            num_delete=6, walks=10)
    with pytest.warns(RuntimeWarning) as record:
        ok, nowhere, inf = sampler.run(jax.random.split(jax.random.key(2), 3),
                                       batched_data=True)
    assert ok.success and ok.metadata["status"] == "converged"
    assert abs(ok.logz - math.log(2 * math.pi * 0.01)) < 4 * ok.logzerr + 0.05
    assert nowhere.metadata["status"] == "invalid" and nowhere.logz == -math.inf
    assert inf.metadata["status"] == "invalid" and inf.logz == math.inf
    assert not nowhere.success and not inf.success
    messages = {str(w.message) for w in record}
    assert {nowhere.message, inf.message} <= messages


@pytest.mark.parametrize("value", [-jnp.inf, jnp.nan, jnp.inf])
def test_a_live_set_without_a_finite_likelihood_is_invalid_at_once(
    value, tmp_path, capsys
) -> None:
    """Every live point at -inf (or NaN) or at +inf: nothing lies above L*,
    and the step set ``"plateau"``. The functional core, the progress line
    and the checkpoint called the run a plateau stop; only ``finalise``
    said ``"invalid"``."""
    cfg = Config(2, 50, num_delete=5)

    def loglike(x):
        return value + 0.0 * x[0]

    state, _ = step(init(0, loglike, identity, cfg), loglike, identity, cfg)
    assert STATUS[int(state.status)] == "invalid" and int(state.it) == 0
    path = tmp_path / "run.npz"
    with pytest.warns(RuntimeWarning):
        result = core.run(0, loglike, identity, cfg, progress=True, checkpoint=path)
    assert result.metadata["status"] == "invalid" and result.niter == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith("[invalid]")
    with np.load(path) as data:
        assert STATUS[int(data["state/status"])] == "invalid"


def test_the_progress_line_counts_a_lane_without_a_finite_likelihood_as_invalid(
    capsys,
) -> None:
    """The batched progress line ended ``[converged=2 plateau=1]`` over
    results that were converged, invalid and converged."""

    def loglike(data, x):
        return data[0] - 0.5 * jnp.sum((x - 0.5) ** 2) / 0.1**2

    data = jnp.asarray([[0.0], [-jnp.inf], [0.0]])
    sampler = NestedSampler(jax.tree_util.Partial(loglike, data), identity, 2, 50,
                            num_delete=5)
    with pytest.warns(RuntimeWarning, match="no live point has a finite likelihood"):
        results = sampler.run(jax.random.split(jax.random.key(0), 3),
                              batched_data=True, progress=True)
    assert [r.metadata["status"] for r in results] == [
        "converged", "invalid", "converged"
    ]
    lines = capsys.readouterr().out.splitlines()
    assert not any("plateau" in line for line in lines), lines
    assert lines[-1].endswith("[converged=2 invalid=1]"), lines[-1]


def test_runs_under_jax_debug_nans() -> None:
    """The contour of the last step starts at -inf (it was a NaN sentinel,
    which made every run fail in init under jax_debug_nans), and the
    information no longer forms 0 * -inf at a likelihood of -inf (which made
    diagnostics() fail there)."""
    cfg = Config(2, 30, num_delete=3, walks=10)
    loglike = gauss(0.5, 0.1)

    def disc(x):  # 0 on a disc of area 0.25, -inf elsewhere
        return jnp.where(jnp.sum((x - 0.5) ** 2) < 0.25 / math.pi, 0.0, -jnp.inf)

    with jax.debug_nans(True):
        state = init(0, loglike, identity, cfg)
        assert float(state.lstar_prev) == -math.inf and int(state.tie_run) == 0
        result = core.run(0, loglike, identity, cfg)
        assert result.diagnostics()["information"] > 0.0
        indicator = core.run(0, disc, identity, cfg)
        information = indicator.diagnostics()["information"]
    assert result.success and abs(result.logz) < 4 * result.logzerr + 0.05
    assert indicator.metadata["status"] == "plateau"
    # All the weight is at L = 1: H = log(1 / Z).
    assert information == pytest.approx(-indicator.logz, rel=1e-4)


def test_flat_top_likelihood_stops_on_the_plateau() -> None:
    def flat_top(x):
        r2 = jnp.sum((x - 0.5) ** 2)
        return -100.0 * jnp.maximum(r2 - 0.01, 0.0)

    truth = math.log(2 * math.pi * 0.01)  # pi r^2 plus the Gaussian skirt
    sampler = NestedSampler(flat_top, identity, 2, 100, num_delete=10)
    close = 0
    for seed in range(1, 6):
        result = sampler.run(seed, dlogz=0.0, maxiter=200_000)
        assert result.metadata["status"] == "plateau"
        # The step that found no point above L* = 0 was abandoned: fewer than
        # k live points lie below the plateau.
        assert np.sum(result.logl[result.niter :] < 0.0) < 10
        close += abs(result.logz - truth) < 4 * result.logzerr + 0.05
    # The first-order logzerr matches the scatter here ((logz - truth) /
    # logzerr has sd 1.02 and 1.07 over two sets of 64 runs; it was 1.6-1.7
    # with the sum of information increments, with rare 6-8 sigma runs), and
    # a majority of five must pass.
    assert close >= 3


def test_termination_by_dlogz_maxiter_and_maxcall() -> None:
    sampler = NestedSampler(gauss(0.5, 0.1), identity, 2, 50, num_delete=5)
    done = sampler.run(0, dlogz=0.5)
    assert done.metadata["status"] == "converged" and done.success
    assert done.metadata["final_delta_logz"] < 0.5

    capped = sampler.run(0, maxiter=137)
    assert capped.metadata["status"] == "maxiter" and not capped.success
    assert capped.niter == 135
    assert "maxiter" in capped.message
    assert sampler.run(0, maxiter=0).niter == 0

    walks = sampler.config.walks
    calls = sampler.run(0, maxcall=3000)
    assert calls.metadata["status"] == "maxcall"
    assert 3000 <= calls.ncall < 3000 + 5 * walks
    assert sampler.run(0, maxcall=10).niter == 0  # init alone spends 50


def test_run_is_deterministic_per_key() -> None:
    sampler = NestedSampler(gauss(0.5, 0.1), identity, 2, 50, num_delete=5)
    a, b = sampler.run(4), sampler.run(jax.random.PRNGKey(4))
    assert a.logz == b.logz and a.ncall == b.ncall
    np.testing.assert_array_equal(a.samples_u, b.samples_u)
    assert sampler.run(5).logz != a.logz


# --- k = 1 against k > 1 -----------------------------------------------------


def correlated_gaussian(ndim=3, rho=0.8, scale=0.05):
    cov = scale**2 * ((1 - rho) * np.eye(ndim) + rho * np.ones((ndim, ndim)))
    prec = jnp.asarray(np.linalg.inv(cov))
    norm = -0.5 * (ndim * math.log(2 * math.pi) + np.linalg.slogdet(cov)[1])

    def loglike(x):
        z = x - 0.5
        return norm - 0.5 * z @ prec @ z

    return loglike


def test_k1_and_k_gt_1_agree_statistically() -> None:
    loglike = correlated_gaussian()
    stats = {}
    for k in (1, 20):
        sampler = NestedSampler(loglike, identity, 3, 200, num_delete=k)
        runs = [sampler.run(seed) for seed in range(6)]
        logz = np.array([r.logz for r in runs])
        err = np.mean([r.logzerr for r in runs])
        stats[k] = (logz.mean(), err / math.sqrt(len(runs)))
        assert abs(logz.mean()) < 3.5 * stats[k][1] + 0.02  # truth 0
        assert runs[0].insertion_test()["pvalue"] > 1e-4
    diff = stats[1][0] - stats[20][0]
    assert abs(diff) < 3.5 * math.hypot(stats[1][1], stats[20][1]) + 0.02


# --- the functional core -----------------------------------------------------


def test_functional_loop_ranks_match_the_insertion_replay() -> None:
    cfg = Config(2, 40, num_delete=5, walks=15)
    loglike = gauss(0.5, 0.1)
    state = init(7, loglike, identity, cfg)
    rows = []
    while float(core.delta_logz(state, cfg)) > 0.1:
        state, dead = step(state, loglike, identity, cfg)
        rows.append(dead)
    dead = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *rows)
    result = finalise(state, dead, cfg, prior_transform=lambda u: 2 * u)
    assert result.niter == 5 * len(rows) and result.num_delete == 5
    np.testing.assert_allclose(result.samples, 2 * np.asarray(result.samples_u))
    # The result holds the recorded ranks; without ties the replay of the
    # births gives them too (a step with an unmoved chain's copy, tied with
    # its seed, may differ: the recorded rank takes a random place there).
    np.testing.assert_array_equal(
        result.insertion_indices(), np.asarray(dead.insertion).reshape(-1)
    )
    replay = np.sort(result._births[1].reshape(-1, 5), axis=1)
    same = np.all(replay == np.sort(np.asarray(dead.insertion), axis=1), axis=1)
    assert np.mean(same) > 0.95
    assert abs(result.logz) < 4 * result.logzerr + 0.05


def test_finalise_reports_convergence_of_a_hand_written_loop() -> None:
    cfg = Config(2, 40, num_delete=5, walks=15)
    loglike = gauss(0.5, 0.1)
    state = init(7, loglike, identity, cfg)
    rows = []
    ncall = int(state.ncall)
    while tinyns.delta_logz(state, cfg) >= 0.1:
        before = state.ncall
        state, dead = step(state, loglike, identity, cfg)
        rows.append(dead)
        ncall += int(state.ncall - before)
    assert tinyns.STATUS[int(state.status)] == "running"  # step leaves it
    dead = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *rows)

    plain = finalise(state, dead, cfg)
    assert not plain.success and plain.message == "running"
    assert "dlogz" not in plain.metadata

    done = finalise(state, dead, cfg, dlogz=0.1, ncall=ncall)
    assert done.success and done.message.startswith("converged")
    assert done.metadata["status"] == "converged" and done.metadata["dlogz"] == 0.1
    assert done.ncall == ncall == plain.ncall
    assert done.logz == plain.logz

    # Stopped early (the dlogz was not reached): still "running".
    early = finalise(state, dead, cfg, dlogz=1e-6)
    assert not early.success and early.metadata["status"] == "running"
    with pytest.raises(ValueError, match="dlogz"):
        finalise(state, dead, cfg, dlogz=-1.0)
    # The rows must be those of the state's steps.
    with pytest.raises(ValueError, match="state has done"):
        finalise(state, jax.tree_util.tree_map(lambda x: x[:-1], dead), cfg)

    # A status the loop or the driver set is kept: dlogz never overrides it.
    for code in (core.MAXITER, core.MAXCALL, core.PLATEAU, core.INVALID):
        forced = state._replace(status=jnp.asarray(code, jnp.int32))
        result = finalise(forced, dead, cfg, dlogz=0.1)
        assert result.metadata["status"] == tinyns.STATUS[code]
        assert result.success == (code == core.PLATEAU)

    # The driver's result agrees with the hand-written loop's.
    driver = core.run(7, loglike, identity, cfg, dlogz=0.1)
    assert driver.niter == done.niter and driver.ncall == done.ncall
    np.testing.assert_allclose(driver.logz, done.logz, rtol=1e-6)


def test_host_call_count_survives_int32_wraparound() -> None:
    """The documented pattern: differences of State.ncall stay right when the
    int32 counter wraps."""
    before = jnp.asarray(2**31 - 10, jnp.int32)
    after = before + jnp.asarray(25, jnp.int32)  # wraps
    assert int(after) < 0 and int(after - before) == 25


def test_final_prior_pass_blocks_follow_num_delete() -> None:
    # A step maps num_delete rows through the prior at once; the final pass
    # took 65536 rows whatever the run, and a heavy prior that sampled fine
    # ran out of memory there.
    assert core._transform_block(1) == core._transform_block(1000) == 1024
    assert core._transform_block(1025) == 2048
    assert core._transform_block(5000) == 8192


def spy_transform_kernel(monkeypatch, fits):
    """Record the rows of each prior-pass call; raise XLA's out-of-memory error
    for blocks with more rows than ``fits``."""
    blocks = []
    real = core._transform_kernel

    def kernel_of(spec):
        kernel = real(spec)

        def call(u, *leaves):
            blocks.append(len(u))
            if len(u) > fits:
                raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory allocating")
            return kernel(u, *leaves)

        return call

    monkeypatch.setattr(core, "_transform_kernel", kernel_of)
    return blocks


def test_final_prior_pass_halves_its_block_out_of_memory(monkeypatch) -> None:
    cfg = Config(1, 40, num_delete=4)
    state, dead = fake_run(np.linspace(-9.0, -1.0, 80), np.zeros(40), 40, 4)
    state = state._replace(u=state.u.astype(FLOAT))  # the dtype of a real run

    def prior(u):
        return jnp.sin(3.0 * u) - u

    expected = finalise(state, dead, cfg, prior_transform=prior).samples
    assert expected.shape == (120, 1)
    blocks = spy_transform_kernel(monkeypatch, fits=64)
    result = finalise(state, dead, cfg, prior_transform=prior)
    # Halved down to 64 rows, then two blocks of 64 (the second one padded).
    assert blocks == [1024, 512, 256, 128, 64, 64]
    np.testing.assert_array_equal(result.samples, expected)

    spy_transform_kernel(monkeypatch, fits=0)
    with pytest.raises(MemoryError, match="even one sample at a time"):
        finalise(state, dead, cfg, prior_transform=prior)


def test_public_key_arguments_accept_int_seeds() -> None:
    cfg = Config(2, 20, num_delete=2, walks=5)
    loglike = gauss(0.5, 0.1)
    a = init(3, loglike, identity, cfg)
    b = init(np.int64(3), loglike, identity, cfg)
    c = init(jax.random.PRNGKey(3), loglike, identity, cfg)
    np.testing.assert_array_equal(a.u, b.u)
    np.testing.assert_array_equal(a.u, c.u)
    r1 = core.run(3, loglike, identity, cfg, maxiter=40)
    r2 = core.run(jax.random.PRNGKey(3), loglike, identity, cfg, maxiter=40)
    np.testing.assert_array_equal(r1.samples_u, r2.samples_u)
    np.testing.assert_array_equal(
        r1.resample_equal(5, n=7), r1.resample_equal(jax.random.PRNGKey(5), n=7)
    )
    with pytest.raises(TypeError):
        init(True, loglike, identity, cfg)  # a bool is not a seed


def test_chunk_length_and_limits_are_traced() -> None:
    cfg = Config(2, 20, num_delete=2, walks=5)
    loglike = gauss(0.5, 0.1)
    from tinyns.callables import _split_callables

    (ll, ll_spec), (pt, pt_spec) = _split_callables(loglike, identity, 2)
    kernel = core._chunk_kernel(ll_spec, pt_spec, cfg, 16)
    state = init(0, loglike, identity, cfg)
    i32 = jnp.int32
    for n_steps, dlogz, maxiter in ((1, 0.1, 100), (3, 0.5, 7), (16, 0.0, 9)):
        state, rows, count, _ = kernel(
            state, i32(n_steps), jnp.asarray(dlogz, FLOAT), i32(maxiter), i32(10**6)
        )
    assert kernel._cache_size() == 1
    assert int(state.status) == core.MAXITER and int(state.it) == 9
    assert rows.u.shape == (16, 2, 2) and int(count) == 5


def test_x64_flag_sets_the_state_dtypes() -> None:
    cfg = Config(2, 10, num_delete=1, walks=5)
    state = init(0, gauss(0.5, 0.1), identity, cfg)
    expected = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    assert state.u.dtype == expected and state.logl.dtype == expected
    result = NestedSampler(gauss(0.5, 0.1), identity, 2, 30).run(0)
    assert result.metadata["x64"] is bool(jax.config.jax_enable_x64)
    assert np.asarray(result.logwt).dtype == np.float64
