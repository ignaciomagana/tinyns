"""``tinyns.compare``: runs of one problem flag the run that lost a mode."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

import tinyns
from tinyns import NestedSampler, NestedSamplingResult, compare
from tinyns.result import (
    COMPARE_LOST_SHARE,
    COMPARE_LOST_SHARE_UNTRACKED,
    COMPARE_MIN_MASS,
    COMPARE_SIGMA,
)

CENTERS = (jnp.array([0.3, 0.4, 0.5]), jnp.array([0.75, 0.6, 0.45]))
SIGMA = 0.03
W_MINOR = 0.25


def two_modes(theta):
    def log_normal(center):
        z = (theta - center) / SIGMA
        return -0.5 * jnp.sum(z * z) - 3 * math.log(SIGMA * math.sqrt(2 * math.pi))

    return jnp.logaddexp(
        math.log(1 - W_MINOR) + log_normal(CENTERS[0]),
        math.log(W_MINOR) + log_normal(CENTERS[1]),
    )


@pytest.fixture(scope="module")
def batch():
    """Four runs that keep both modes (minor mass 0.25)."""
    keys = jax.random.split(jax.random.key(0), 4)
    return NestedSampler(two_modes, lambda u: u, 3, nlive=400).run(keys)


def minor_scaled(result, scale):
    """``result`` with the posterior mass of its minor mode times ``scale``
    (0: a run that lost the mode), and logz to match."""
    logwt = np.asarray(result.logwt, dtype=float).copy()
    minor = np.asarray(result.samples_u)[:, 0] > 0.55
    with np.errstate(divide="ignore"):
        logwt[minor] += math.log(scale) if scale > 0 else -np.inf
    return dataclasses.replace(
        result, logwt=logwt, logz=float(np.logaddexp.reduce(logwt))
    )


def built(logz, logzerr=0.1, ndim=2, **fields):
    """A one-mode result built by hand (no labels)."""
    u = np.random.default_rng(0).uniform(size=(30, ndim))
    logl = np.linspace(-10.0, 0.0, 30)
    nlive_i = np.full(30, 10)
    nlive_i[-10:] = np.arange(10, 0, -1)
    logwt = logl - np.arange(1, 31) / 10
    logwt += logz - np.logaddexp.reduce(logwt)
    values = dict(
        samples_u=u, samples=u, logl=logl, logwt=logwt,
        logl_birth=np.full(30, -np.inf), nlive_i=nlive_i, logz=logz,
        logzerr=logzerr, ncall=100, niter=20, nlive=10, num_delete=1, ndim=ndim,
        message="converged", metadata={"status": "converged"},
    )
    return NestedSamplingResult(**{**values, **fields})


def test_compare_is_public() -> None:
    assert "compare" in tinyns.__all__ and tinyns.compare is compare
    assert "Comparison" in tinyns.__all__
    assert (COMPARE_SIGMA, COMPARE_MIN_MASS) == (4.0, 0.05)
    assert (COMPARE_LOST_SHARE, COMPARE_LOST_SHARE_UNTRACKED) == (0.1, 0.01)


def test_runs_that_kept_every_mode_are_not_flagged(batch) -> None:
    report = compare(batch)
    assert report.flagged == []
    assert report.best == int(np.argmax([r.logz for r in batch]))
    for i, row in enumerate(report.runs):
        assert row["run"] == i and row["compared"] and not row["flagged"]
        assert row["reason"] == "" and row["missing_modes"] == []
        assert row["nmodes"] == 2 and row["nlive"] == 400
        assert row["logz"] == float(batch[i].logz)
        assert 0 <= row["logz_shortfall"] < COMPARE_SIGMA
    assert report.runs[report.best]["logz_shortfall"] == 0
    assert str(report) == report.summary()
    assert report.summary().startswith("4 runs, 4 compared: 0 flagged")


def test_a_run_that_lost_a_mode_is_flagged(batch) -> None:
    """The run that lost the 0.25 mode lists no such mode and its logZ is
    only log(0.75) = -0.29 low, about 1.6 combined logzerr: the mode test
    flags it, from where the other runs found the mode."""
    lost = minor_scaled(batch[0], 0.0)
    assert len(lost.modes()) == 1
    report = compare([*batch[1:], lost])
    assert report.flagged == [3]
    row = report.runs[3]
    assert "it lacks a mode that 3 other runs found (mode 1 of run " in row["reason"]
    assert sorted(item["run"] for item in row["missing_modes"]) == [0, 1, 2]
    for item in row["missing_modes"]:
        assert item["mode"] == 1
        assert item["mass"] == pytest.approx(W_MINOR, abs=0.05)
        assert item["mass"] == batch[1 + item["run"]].modes()[1]["mass"]
        assert item["mass_here"] == pytest.approx(0.0, abs=1e-6)
    assert "run 3: " + row["reason"] in report.summary()


def test_a_run_with_its_modes_merged_is_not_flagged(batch) -> None:
    """Its samples hold the minor mode's mass where the other runs found it,
    whatever its own modes(): without labels it reports one mode."""
    merged = dataclasses.replace(batch[0], labels=None)
    assert len(merged.modes()) == 1
    assert compare([merged, *batch[1:]]).flagged == []


def test_a_mode_counts_as_lost_below_a_tenth_of_its_mass(batch) -> None:
    for scale, lacks in ((0.2, False), (0.02, True)):
        run = minor_scaled(batch[0], scale)
        report = compare([*batch[1:], run])
        here = run.weights()[np.asarray(run.samples_u)[:, 0] > 0.55].sum()
        assert (here < COMPARE_LOST_SHARE * W_MINOR) == lacks
        assert bool(report.runs[3]["missing_modes"]) == lacks
        for item in report.runs[3]["missing_modes"]:
            assert item["mass_here"] == pytest.approx(here, rel=0.05)


def test_an_untracked_mode_counts_as_lost_below_a_hundredth(batch) -> None:
    """The hop does not balance a mode that the clustering did not separate,
    and its mass scatters from run to run as without the hop (the peaks of
    the eggbox up to 26-fold): a run with 4% of the mass another run found
    there holds such a mode, one with 0.4% lacks it."""
    refs = [dataclasses.replace(result) for result in batch[1:]]
    for ref in refs:
        ref._mode_fit[0][1]["tracked"] = False  # as if pass 4 had cut it out
    for scale, lacks in ((0.03, False), (0.003, True)):
        run = minor_scaled(batch[0], scale)
        report = compare([*refs, run])
        here = run.weights()[np.asarray(run.samples_u)[:, 0] > 0.55].sum()
        assert (here < COMPARE_LOST_SHARE_UNTRACKED * W_MINOR) == lacks
        assert bool(report.runs[3]["missing_modes"]) == lacks
        assert compare([*batch[1:], run]).runs[3]["missing_modes"]  # tracked


def test_the_logz_test_counts_combined_logzerr() -> None:
    runs = [built(0.0), built(-0.5), built(-2.0, logzerr=0.2), built(0.3)]
    report = compare(runs)
    assert report.best == 3
    shortfall = [row["logz_shortfall"] for row in report.runs]
    assert shortfall == pytest.approx(
        [0.3 / math.hypot(0.1, 0.1), 0.8 / math.hypot(0.1, 0.1),
         2.3 / math.hypot(0.1, 0.2), 0.0]
    )
    assert report.flagged == [1, 2]
    assert report.runs[2]["reason"] == (
        "logz -2 lies 10.3 combined logzerr below the 0.3 of run 3"
    )
    assert all(row["nmodes"] == 1 and not row["missing_modes"] for row in report.runs)


def test_runs_that_did_not_succeed_are_flagged_and_not_compared() -> None:
    maxcall = "stopped at maxcall before reaching dlogz"
    stopped = built(5.0, success=False, message=maxcall)
    empty = built(0.0, success=False, message="no live point has a finite likelihood")
    empty = dataclasses.replace(
        empty, logwt=np.full(30, -np.inf), logz=-math.inf, logzerr=math.nan
    )
    report = compare([built(0.0), stopped, built(-0.1, logzerr=0.05), empty])
    assert report.best == 0  # not the stopped run, despite its logz
    assert report.flagged == [1, 3]
    assert report.runs[1]["reason"] == "not compared: " + maxcall
    assert report.runs[3]["reason"].startswith("not compared: no live point")
    assert [row["compared"] for row in report.runs] == [True, False, True, False]
    assert report.runs[1]["logz_shortfall"] is None
    hand = dataclasses.replace(built(0.0), logwt=np.full(30, -np.inf), logz=-math.inf)
    assert compare([built(0.0), hand]).runs[1]["reason"] == (
        "not compared: logz or logzerr is not finite"
    )
    # A logzerr that is not finite would make every shortfall infinite (or
    # none) if that run held the highest logZ.
    report = compare([built(0.0), built(1.0, logzerr=math.nan), built(-0.1)])
    assert report.best == 0 and report.flagged == [1]
    assert report.runs[1]["reason"] == "not compared: logz or logzerr is not finite"
    assert compare([stopped]).best is None


def test_one_run_and_runs_of_other_nlive() -> None:
    report = compare([built(0.0)])
    assert report.flagged == [] and report.best == 0
    assert report.summary().startswith("1 run: nothing to compare it with")
    wide = built(-0.2, nlive=40)
    report = compare([built(0.0), wide])
    assert report.flagged == [] and report.runs[1]["nlive"] == 40


def test_compare_refuses_what_is_not_one_problem() -> None:
    with pytest.raises(TypeError, match="list of results"):
        compare(built(0.0))
    with pytest.raises(TypeError, match="NestedSamplingResult"):
        compare([built(0.0), {"logz": 0.0}])
    with pytest.raises(ValueError, match="at least one"):
        compare([])
    with pytest.raises(ValueError, match="ndim 2, 3"):
        compare([built(0.0), built(0.0, ndim=3)])
    lane = built(0.0, metadata={"status": "converged", "batched_data": True})
    with pytest.raises(ValueError, match="batched_data=True"):
        compare([built(0.0), lane])
    assert compare(iter([built(0.0), built(0.0)])).flagged == []


def rot(d, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return q * np.sign(np.diag(r))


# The 4-D mixture of the lost-mode study (masses 0.6, 0.3 and 0.1, widths
# 0.25, 0.5 and 0.75, rotated, in [-12, 12]^4): the 0.6 mode holds about 6 of
# 500 live points when it separates and is lost in about 1 of 4 runs.
MIX_W = np.array([0.6, 0.3, 0.1])
MIX_MU = np.array([[-5, -5, 0, 0], [5, -4, 1, 0], [0, 5, -1, 1]], dtype=float)
MIX_CHOL = np.linalg.cholesky([
    rot(4, 10 + i) @ np.diag((s * np.roll(np.linspace(0.6, 1.4, 4), i)) ** 2)
    @ rot(4, 10 + i).T
    for i, s in enumerate((0.25, 0.5, 0.75))
])
MIX_CONST = (np.log(MIX_W) - np.log(np.diagonal(MIX_CHOL, axis1=1, axis2=2)).sum(1)
             - 2 * math.log(2 * math.pi))


def mix3(x):
    z = jax.vmap(lambda m, c: jax.scipy.linalg.solve_triangular(c, x - m, lower=True))(
        jnp.asarray(MIX_MU), jnp.asarray(MIX_CHOL)
    )
    return logsumexp(jnp.asarray(MIX_CONST) - 0.5 * jnp.sum(z * z, axis=1))


def mix3_lost(result):
    """Whether the run kept less than a tenth of the mass of a component (the
    mass of the samples whose likeliest component it is)."""
    x = np.asarray(result.samples, dtype=float)
    score = [MIX_CONST[i] - 0.5 * np.sum(np.linalg.solve(MIX_CHOL[i], (x - MIX_MU[i]).T)
                                         ** 2, axis=0) for i in range(3)]
    best = np.argmax(score, axis=0)
    mass = [result.weights()[best == i].sum() for i in range(3)]
    return bool(np.any(np.asarray(mass) < 0.1 * MIX_W))


def flagged_runs(results):
    """compare() on ``results``: every run that lost a mode is flagged if
    another run kept every mode, and a run that lacks a mode of another run
    did lose one; returns the number of runs lost and of the others flagged
    (by the logZ test)."""
    lost = [mix3_lost(r) for r in results]
    report = compare(results)
    for row, gone in zip(report.runs, lost, strict=True):
        assert gone or not row["missing_modes"], (row, lost)
        assert row["flagged"] or not gone or all(lost), (row, lost)
    return sum(lost), sum(row["flagged"] and not gone
                          for row, gone in zip(report.runs, lost, strict=True))


def test_real_lost_modes_are_flagged() -> None:
    """Eight runs of the study's 4-D mixture at nlive 300, where the 0.6 mode
    was lost in 14 of 32 runs (x64 off and on), in two batches of four."""
    sampler = NestedSampler(mix3, lambda u: -12 + 24 * u, 4, nlive=300)
    results = sampler.run(jax.random.split(jax.random.key(0), 8))
    for i in (0, 4):
        flagged_runs(results[i : i + 4])


@pytest.mark.slow
def test_lost_modes_are_flagged_in_batches_of_four() -> None:
    """64 runs of the 4-D mixture at nlive 500 (the 0.6 mode lost in about 1
    of 4) in batches of four: every lost run is flagged where a run of its
    batch kept the mode, and at most one other run is flagged."""
    sampler = NestedSampler(mix3, lambda u: -12 + 24 * u, 4, nlive=500)
    results = sampler.run(jax.random.split(jax.random.key(1), 64))
    lost, kept_flagged = np.sum(
        [flagged_runs(results[i : i + 4]) for i in range(0, 64, 4)], axis=0
    )
    assert lost >= 2 and kept_flagged <= 1, (lost, kept_flagged)
