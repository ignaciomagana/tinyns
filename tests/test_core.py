"""The full-JAX core: Config, init, step, the evidence bookkeeping, the driver."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from tinyns import Config, NestedSampler, core, finalise, init, step

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
    """Sample by sample (dynesty-style running Z and H), plain Python floats.

    Sample i shrinks log X by 1 / n_i and takes X_{i-1} - X_i; the last sample
    takes all of X_{N-2}. Var(log Z) accumulates dH_i / n_i.
    """
    logz, h, var, log_x = -math.inf, 0.0, 0.0, 0.0
    logwts = []
    for i, (ll, n) in enumerate(zip(logl, nlive_i, strict=True)):
        log_x_new = log_x - 1.0 / n
        if i == len(logl) - 1:
            width = math.exp(log_x)
        else:
            width = math.exp(log_x) - math.exp(log_x_new)
        logwt = ll + math.log(width) if ll > -math.inf else -math.inf
        logwts.append(logwt)
        if logwt > -math.inf:
            logz_new = np.logaddexp(logz, logwt)
            old = 0.0 if logz == -math.inf else math.exp(logz - logz_new) * (h + logz)
            h_new = math.exp(logwt - logz_new) * ll + old - logz_new
            var += (h_new - h) / n
            h, logz = h_new, logz_new
        log_x = log_x_new
    return np.array(logwts), float(logz), math.sqrt(max(var, 0.0))


FOLDS = Config(1)._folds


def fake_run(dead_logl, live_logl, m, k):
    """A final State and Dead rows (numpy) with the given likelihoods."""
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
        log_scale=np.float64(0.0),
        ncall=np.int32(0),
        ncall_valid=np.int32(0),
        status=np.int32(core.CONVERGED),
        label=np.zeros((FOLDS, m), np.int32),
        **core._empty_frames(1, FOLDS, np.float64),
    )
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


def test_walk_mask_falls_back_when_too_few_points_remain() -> None:
    above = jnp.asarray([True] * 6 + [False] * 4)
    other = jnp.asarray([True, False] * 5)  # 3 even points above L*
    seeds = jnp.asarray([1, 3])
    np.testing.assert_array_equal(
        core._walk_mask(above, other, seeds, 2), np.asarray(above & other)
    )
    held_out = np.asarray(above).copy()
    held_out[[1, 3]] = False  # 4 points: the step's seeds left out
    np.testing.assert_array_equal(core._walk_mask(above, other, seeds, 3), held_out)
    np.testing.assert_array_equal(core._walk_mask(above, other, seeds, 4), above)
    # A step on a nearly exhausted live set still runs: 2 points above L*,
    # both seeds, in 2-D.
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
        else:  # vmapped: every lane is evaluated, the in-cube ones counted apart
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
    # logzerr understates the scatter on a plateau ((logz - truth) / logzerr
    # has sd ~1.6, with rare 6-8 sigma runs), so a majority of five must pass.
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


def test_functional_loop_matches_the_insertion_replay() -> None:
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
    replay = result.insertion_indices().reshape(-1, 5)
    np.testing.assert_array_equal(
        np.sort(replay, axis=1), np.sort(np.asarray(dead.insertion), axis=1)
    )
    assert abs(result.logz) < 4 * result.logzerr + 0.05


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
