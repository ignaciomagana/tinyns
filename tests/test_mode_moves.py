"""The inter-mode hop (:mod:`tinyns.modes`): exactness and integration.

The invariance test is the gate of the hop: on a fixed two-mode live set,
chains started from seeds drawn uniformly in the constrained region must end
in each mode in proportion to its volume. The seed is one of the points the
frames were fitted to, so the test fails unless the leave-one-out refit
removes it (the control without it is biased). ``pytest -m slow`` runs it
with 10^5 chains.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from tinyns import core, modes
from tinyns.core import Config

D = 3
CENTERS = np.array([[0.30, 0.35, 0.40], [0.72, 0.62, 0.55]])
RADII = np.array([0.16, 0.11])
MINOR_FRACTION = RADII[1] ** D / np.sum(RADII**D)  # 0.2453
NFIT = 40  # live points the frames are fitted to (the seed is one of them)
WALKS = 40  # four hop steps per chain
SCALE = 0.3


def region_loglike(u):
    """0 inside either ball, -1 outside; the contour is L* = -0.5."""
    r2 = jnp.sum((u[None, :] - jnp.asarray(CENTERS)) ** 2, axis=1)
    return jnp.where(jnp.any(r2 <= jnp.asarray(RADII) ** 2), 0.0, -1.0)


def uniform_region(rng, n):
    ball = rng.choice(2, size=n, p=[1 - MINOR_FRACTION, MINOR_FRACTION])
    z = rng.standard_normal((n, D))
    z *= (rng.uniform(size=(n, 1)) ** (1 / D)) / np.linalg.norm(z, axis=1)[:, None]
    return CENTERS[ball] + RADII[ball][:, None] * z


def invariance(nchains: int, loo: bool = True, seed: int = 0):
    """Fraction of chain end points in the minor ball, and the hop counts."""
    dtype = jnp.result_type(float)
    rng = np.random.default_rng(seed)
    others = jnp.asarray(uniform_region(rng, NFIT - 1), dtype)
    labels, st = modes.recluster(others, jnp.zeros(NFIT - 1, jnp.int32))
    assert int(jnp.sum(st.count > 0)) == 2, "the fixed live set must split"
    fit_others = modes.frames(st, NFIT)
    chol = core._live_chol(others, jnp.ones(NFIT - 1, bool))

    def one(key, s):
        label = modes.nearest(fit_others, s[None])[0]
        all_stats = modes.stats(
            jnp.concatenate([others, s[None]]),
            jnp.concatenate([labels, label[None]]),
            modes.C_MAX,
        )
        fr = modes.chain_frames(all_stats, s, label, loo, NFIT)
        u, _, _, _, _, hops, tries = core._chain(
            key, s, jnp.zeros((), dtype), jnp.asarray(-0.5, dtype), chol,
            jnp.asarray(SCALE, dtype), region_loglike, lambda v: v, WALKS, False,
            fr,
        )
        return jnp.sum((u - CENTERS[1]) ** 2) <= RADII[1] ** 2, hops, tries

    batch = jax.jit(jax.vmap(one))
    minor = hops = tries = 0
    key = random.PRNGKey(seed)
    for start in range(0, nchains, 10_000):
        n = min(10_000, nchains - start)
        key, sub = random.split(key)
        seeds = jnp.asarray(uniform_region(rng, n), dtype)
        m, h, t = batch(random.split(sub, n), seeds)
        minor += int(jnp.sum(m))
        hops += int(jnp.sum(h))
        tries += int(jnp.sum(t))
    return minor / nchains, hops, tries


def z_score(fraction, n):
    return (fraction - MINOR_FRACTION) / math.sqrt(
        MINOR_FRACTION * (1 - MINOR_FRACTION) / n
    )


def test_hop_keeps_the_constrained_prior() -> None:
    n = 20_000
    fraction, hops, tries = invariance(n)
    assert tries == n * (WALKS // modes.HOP_EVERY)  # two eligible frames
    assert hops > 0.02 * tries
    assert abs(z_score(fraction, n)) < 4.0


@pytest.mark.slow
def test_hop_keeps_the_constrained_prior_1e5() -> None:
    """The gate: 10^5 chains, within 3 sigma of the volume fraction."""
    n = 100_000
    fraction, hops, tries = invariance(n, seed=1)
    z = z_score(fraction, n)
    print(f"minor fraction {fraction:.4f} (volume {MINOR_FRACTION:.4f}) "
          f"z={z:+.2f} hop acceptance {hops / tries:.3f}")
    assert abs(z) < 3.0


@pytest.mark.slow
def test_without_leave_one_out_the_hop_is_biased() -> None:
    """The control: frames that contain the seed bias the mode fractions, so
    the invariance test has the power to see a missing downdate."""
    n = 100_000
    fraction, _, _ = invariance(n, loo=False, seed=1)
    z = z_score(fraction, n)
    print(f"without LOO: minor fraction {fraction:.4f} z={z:+.2f}")
    assert abs(z) > 3.0


def test_downdate_matches_a_refit() -> None:
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.uniform(size=(30, 4)))
    labels = jnp.asarray(rng.integers(0, 3, 30), jnp.int32)
    full = modes.stats(x, labels, modes.C_MAX)
    i = 7
    down = modes.downdate(full, x[i], labels[i], True)
    refit = modes.stats(x, labels.at[i].set(-1), modes.C_MAX)
    tol = 1e-4 if x.dtype == jnp.float32 else 1e-10
    for a, b in zip(down, refit, strict=True):
        np.testing.assert_allclose(a, b, atol=tol)
    same = modes.downdate(full, x[i], labels[i], False)
    for a, b in zip(same, full, strict=True):
        np.testing.assert_array_equal(a, b)


def two_gaussians(theta):
    def log_normal(center, sigma):
        z = (theta - center) / sigma
        return -0.5 * jnp.sum(z * z) - D * math.log(sigma * math.sqrt(2 * math.pi))

    return jnp.logaddexp(
        math.log(0.75) + log_normal(jnp.asarray(CENTERS[0]), 0.03),
        math.log(0.25) + log_normal(jnp.asarray(CENTERS[1]), 0.02),
    )


def test_run_hops_between_modes() -> None:
    cfg = Config(D, 200, 20, 20)
    result = core.run(3, two_gaussians, lambda u: u, cfg)
    assert abs(result.logz) < 5 * result.logzerr + 0.2  # truth 0
    meta = result.metadata
    assert meta["hop_tries"] > 0 and meta["hops"] > 0
    assert max(c for _, c, _ in meta["mode_history"]) >= 2
    minor = np.asarray(result.samples)[:, 0] > 0.5
    w = np.asarray(result.weights())
    assert 0.1 < w[minor].sum() / w.sum() < 0.45  # truth 0.25


def test_without_the_hop_the_clustering_still_runs() -> None:
    """``_hop=False`` (the comparison arm N of the bake-off): random-walk
    steps only, but the labels are still recorded for modes()."""
    cfg = Config(D, 200, 20, 20, _hop=False)
    result = core.run(3, two_gaussians, lambda u: u, cfg)
    meta = result.metadata
    assert meta["hop_tries"] == 0 and meta["hop_acceptance"] is None
    assert max(c for _, c, _ in meta["mode_history"]) >= 2
    assert len(result.labels) == len(result.logl)
    assert len(result.modes()) == 2
    assert Config(D, 40, 4, 10).recluster_every == 3
    with pytest.raises(TypeError, match="_hop"):
        Config(D, 40, 4, 10, _hop="N")


def test_one_chain_per_step_hops() -> None:
    cfg = Config(D, 100, 1, 20)
    result = core.run(5, two_gaussians, lambda u: u, cfg, maxiter=3000)
    assert result.metadata["hop_tries"] > 0


def test_step_reclusters_on_schedule() -> None:
    cfg = Config(D, 40, 4, 10)
    state = core.init(1, two_gaussians, lambda u: u, cfg)
    state, _ = core.step(state, two_gaussians, lambda u: u, cfg)
    assert bool(jnp.all(state.mode_count[1:] == 0)) and state.mode_count[0] == 40
    assert int(jnp.sum(state.fitted)) == 40 - 4  # the new points are not fitted


def test_batched_run_matches_single_runs_with_the_hop() -> None:
    """As in test_batch: equal to roundoff in float64; in float32 a roundoff
    difference can flip a Metropolis decision, so only the start must agree."""
    cfg = Config(D, 60, 6, 10)
    keys = random.split(random.PRNGKey(2), 3)
    batch = core.run(keys, two_gaussians, lambda u: u, cfg, maxiter=1200)
    for key, result in zip(keys, batch, strict=True):
        alone = core.run(key, two_gaussians, lambda u: u, cfg, maxiter=1200)
        if jax.config.jax_enable_x64:
            assert (alone.niter, alone.ncall) == (result.niter, result.ncall)
            np.testing.assert_allclose(alone.samples_u, result.samples_u, atol=1e-9)
            assert alone.metadata["mode_history"] == result.metadata["mode_history"]
        else:
            np.testing.assert_allclose(
                alone.logl[:120], result.logl[:120], rtol=1e-4, atol=1e-4
            )
        assert result.metadata["hop_tries"] > 0


def test_checkpoint_round_trip_with_cluster_state(tmp_path, monkeypatch) -> None:
    cfg = Config(D, 60, 6, 10)
    path = tmp_path / "run.npz"
    whole = core.run(4, two_gaussians, lambda u: u, cfg)
    # Stop after a few steps, then resume from the checkpoint.
    monkeypatch.setattr(core, "_CHECKPOINT_SECONDS", 0.0)
    part = core.run(4, two_gaussians, lambda u: u, cfg, maxiter=300, checkpoint=path)
    assert part.niter == 300
    resumed = core.run(4, two_gaussians, lambda u: u, cfg, checkpoint=path)
    assert resumed.metadata["resumed"]
    np.testing.assert_array_equal(resumed.logl, whole.logl)
    assert resumed.metadata["mode_history"] == whole.metadata["mode_history"]
    other = Config(D, 60, 6, 10, _hop=False)
    with pytest.raises(ValueError, match="_hop"):
        core.run(4, two_gaussians, lambda u: u, other, checkpoint=path)
