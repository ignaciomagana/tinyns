"""Result containers returned by :class:`tinyns.NestedSampler`."""

from __future__ import annotations

import bisect
import copy
import itertools
import json
import math
import operator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from jax import random

from tinyns.checkpoint import _atomic_savez
from tinyns.math import (
    as_key,
    effective_sample_size_from_log_weights,
    normalize_log_weights,
    systematic_resample,
)
from tinyns.modes import C_MAX, split_threshold

ArrayLike = Any

_RESULT_NPZ_FORMAT_VERSION = "tinyns-result-npz-v4"
_RESULT_NPZ_ARRAYS = ("samples_u", "samples", "logl", "logwt", "logl_birth", "nlive_i")
# Optional arrays (``None`` in a result built by hand), saved when present.
_RESULT_NPZ_OPTIONAL = ("labels", "insertion_ranks", "insertion_slots")
_RESULT_NPZ_SCALARS = {
    "logz": float,
    "logzerr": float,
    "ncall": int,
    "niter": int,
    "nlive": int,
    "num_delete": int,
    "ndim": int,
    "success": bool,
    "message": str,
}
# A mode whose live count falls below this many points per dimension between
# its isolation and its posterior bulk is flagged unresolved (raise nlive). A
# covariance-adapted walk needs about that many points to learn a mode's
# shape: on one Gaussian with the shape of the minor mode of the 18-D
# bake-off target, 1.7, 3.3 and 6.7 points per dimension left logZ 4.35,
# 0.21 and 0.03 too high, and the bake-off cells whose minor mode held fewer
# than 5 per dimension kept a weight bias or a large scatter.
UNRESOLVED_PER_DIM = 5.0
# Two clusters are one mode unless their live points leave a gap of this many
# within-cluster standard deviations along the discriminant direction at each
# of their posterior medians. The split test also cuts one curved mode (a
# banana) into linear pieces, which touch: gaps of -0.5 to 1.1 there, against
# 22 to 32 between the separated modes of the Gaussian validation mixtures
# and 5 to 20 between those of LogGamma (10-D, 500 to 1000 live points).
# Before that, a cluster with at most ndim distinct live points at its own
# posterior median is set aside: it spans no volume.
SEPARATION_SIGMA = 3.0
# A mode is cut where its live points leave a slab across a coordinate axis
# of the unit cube empty (:func:`_empty_slab`): the width of the slab times
# the density of the SLAB_NEIGHBOURS (at least ndim + 1) live points on its
# sparser side must reach SLAB_SCORE. Live points are uniform in the cube
# above the likelihood contour, so a connected region leaves no such slab:
# the largest score was 21.5 over 768 runs (64 each of Gaussians in 2-D and
# 10-D, bananas in 5-D and 10-D, the 4-D Rosenbrock and the 10-D funnel, at
# nlive 500 and 1000), against at least 56 between the modes of 12 Gaussians
# in 3-D, 250 for the 2-D LogGamma target and 1600 for the eggbox.
SLAB_SCORE = 30.0
SLAB_NEIGHBOURS = 8
# Two modes are one if the likelihood dips by less than SADDLE_DEPTH between
# them and the bottom of the dip lies above the posterior median likelihood
# of at least one of them (:func:`_apart`). The split test also cuts lumps
# off one mode whose contours are a core with thin arms, and the lumps leave
# gaps in the live points too: dips of 0.0 to 1.8 between such lumps and
# their mode (needle_d18 of ``bench.targets``, 16 runs; an 18-D posterior
# with a funnel neck, 3 runs), against at least 5.7 between the separate
# modes of LogGamma (2-D and 10-D) and 40 to 200 for the Gaussian mixtures,
# the product of Cauchy factors and the eggbox (nlive 400 to 1000, 132
# runs). In few dimensions no depth tells both apart: two equal Gaussians D
# sigma apart dip by D^2/8 - log 2 in any dimension (1.84 at 4.5 sigma), and
# pairs 3.5 to 5 sigma apart in 2-D and 4.5 to 5 in 3-D, which the
# clustering separates, were one mode in all 96 runs (nlive 500). But the
# bulk of a mode lies close to its peak there (a Gaussian's posterior median
# 0.69 below it in 2-D, 1.18 in 3-D and 8.7 in 18-D), and such a dip runs
# below the bulk of both: those pairs are now two modes in the 95 runs whose
# clustering found two, while the dips between the lumps of needle_d18
# bottom out 7.8 to 9.0 above the bulk (8 runs) and a 2-D pair 3 sigma apart
# stays one mode (0.08 to 0.16 above). With both tests, pass 6 keeps equal
# Gaussians apart from about 2.5 sigma in 1-D (a dip of 0.13: two modes in
# 32 of 32 runs at nlive 200 and 500, one at 2.2 sigma), 3.3 in 2-D, 3.9 in
# 3-D, 4.8 in 5-D and 5.2 in 6-D, and from 5.4 (the depth) on in 7-D and up,
# but only where passes 1 to 5 separated them: at 5 sigma in 5-D they did in
# 7 of 32 runs. Nor does it help a close mode of less mass: the clustering
# leaves it in the labels it split the heavier one from, which also hold
# both modes' points from before the split, and passes 2 and 3 merge the
# two. In 2-D a mode of mass 0.1 or 0.2 at 4.5 to 5.5 sigma was one mode
# with the other in 70 of 72 runs (a 0.2 mode at 5 sigma was kept apart in
# 2 of 16), and a 0.1 mode at 6.5 sigma in 18 of 32 (nlive 500).
SADDLE_DEPTH = 3.0
SADDLE_BINS = 16
# The sampler's clustering separated a mode if at least SEPARATED_SHARE of
# the mode's posterior mass carries recorded labels (clusters) that put at
# least PURE_LABEL of their own mass in that mode.
PURE_LABEL, SEPARATED_SHARE = 0.9, 0.5
# The hop balanced the modes only in part (a warning) when, after they
# separated and while two of them held live points, the clustering held
# LUMPS_WARN clusters per mode or more on average over the posterior mass,
# when a clustering had fewer than two eligible clusters for HOP_OFF_WARN of
# that mass or more, or when over the whole run the hop accepted less than
# HOP_ACCEPT_WARN of its proposals. Measured on the targets of ``bench``
# (nlive 200 to 2000, 330 runs, averages up to the end of the run): the
# needle (d >= 10) and Cauchy targets held 1.54 to 3.7 clusters per mode and
# an 18-D posterior whose modes have a funnel neck 2.5 to 3.6. On those
# three the hop cut the run-to-run scatter of the minor mass in logit only
# 1.6 to 3 times (needle_d18: 0.57 to 0.19), against 16 to 19 times on the
# plain targets (0.37 to 0.019 at 18-D), and it accepted at least 0.19 of
# its proposals on every target but the 18-D posterior (0.6 to 0.8%). Once
# one mode is left the hop is off by necessity; averaged up to the end, that
# stretch fired the warning on correct runs whose heaviest mode is also the
# tallest (hop off for 0.47 to 0.68 of the mass: all 16 runs of a 4-D
# mixture of masses 0.7 and 0.3 and widths 0.5 and 1, 13 of 16 of two 2-D
# Gaussians of masses 0.8 and 0.2). Up to the death of the second-last mode
# (CPU, nlive 400 to 1000, about 400 runs of two- and three-mode Gaussian
# mixtures in 2 to 18 dimensions, LogGamma and the needle and Cauchy
# targets), the mixtures and LogGamma held at most 1.13 clusters per mode,
# against 2.0 to 3.6 for the needle and 1.8 to 2.5 for Cauchy targets; the
# hop was off for at most 0.35 of the mass on the mixtures and LogGamma at
# nlive 1000, but for 0.5 to 0.88 in 4 of 8 loggamma_d10 runs and 1 of 8
# runs of a 10-D plain two-mode target at nlive 500, and it accepted at
# least 0.2 of its proposals. A run that lost a mode, or the narrow core of
# one, ends like a correct run with one mode alive, and the warning does not
# flag it: up to the end of the run the hop was off for 0.51 to 0.61 of the
# mass on 6 such runs (2 of 32 of a 4-D mixture of masses 0.6, 0.3 and 0.1
# lost the 0.6 mode; 3 of 8 and 1 of 8 of a 4-D and a 6-D pair of masses 0.8
# and 0.2, each mode a narrow core in a wide needle, lost most of the minor
# core), against 0.38 to 0.84 on correct runs, so the old average flagged
# them by accident.
LUMPS_WARN, HOP_OFF_WARN, HOP_ACCEPT_WARN = 1.5, 0.5, 0.02
# The nearest-neighbour search of modes() takes its distances in blocks of
# about this many coordinate differences (float64: 32 MB, at any nlive).
_NEAREST_BLOCK = 2**22


def _jsonable(value):
    """``json.dumps`` fallback: NumPy and JAX values as lists, others as text."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return str(value)
    return str(value) if array.dtype == object else array.tolist()


def _npz_scalar(value):
    """Return a Python scalar from a NumPy value loaded from ``np.load``."""

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected scalar value in result .npz file")
    return array.reshape(()).item()


def _information(logwt, logl, logz: float) -> float:
    """Return the information ``H`` in float64 (NaN when undefined)."""
    if not math.isfinite(logz):
        return math.nan
    with np.errstate(over="ignore"):
        weights = np.exp(np.asarray(logwt, dtype=np.float64) - logz)
    logl = np.asarray(logl, dtype=np.float64)
    finite = np.isfinite(logl)
    if not np.all(np.isfinite(weights)) or np.any((weights > 0.0) & ~finite):
        return math.nan
    contributing = (weights > 0.0) & finite
    # Mask before multiplying: 0 * -inf would be NaN.
    information = np.sum(weights * np.where(contributing, logl - logz, 0.0))
    return max(float(information), 0.0)


def _ridge(cov):
    """Add a relative ridge of ``1e-12`` times the mean variance."""
    ndim = cov.shape[-1]
    scale = max(float(np.trace(cov)) / ndim, 1e-300)
    return cov + 1e-12 * scale * np.eye(ndim)


def _fisher(x, side) -> float:
    """Fisher separation ``J = dm^T Sw^-1 dm`` of ``x[~side]`` and ``x[side]``."""
    a, b = x[~side], x[side]
    dm = b.mean(0) - a.mean(0)
    a, b = a - a.mean(0), b - b.mean(0)
    within = _ridge((a.T @ a + b.T @ b) / max(len(x) - 2, 1))
    return float(dm @ np.linalg.solve(within, dm))


def _mahalanobis2(x, point) -> float:
    """Squared Mahalanobis distance of ``point`` to the rows ``x`` (their mean
    and covariance); ``inf`` for fewer than two rows."""
    if len(x) < 2:
        return math.inf
    dev = x - x.mean(0)
    cov = _ridge(dev.T @ dev / (len(x) - 1))
    dm = point - x.mean(0)
    return float(dm @ np.linalg.solve(cov, dm))


def _nearest_mode(u, weights, label, k):
    """Hard-EM classification of the rows ``u`` among ``k`` modes.

    Each mode's frame is the posterior-weighted mean and covariance of its
    samples, shrunk toward the pooled one with a weight of ``ndim`` effective
    samples; a row goes to the mode of least ``r^2 + 2 log det L - 2 log
    mass``.
    """
    ndim = u.shape[1]
    fits, pooled, total = [], np.zeros((ndim, ndim)), 0.0
    for c in range(k):
        w = weights * (label == c)
        mass = w.sum()
        mu = w @ u / mass
        dev = u - mu
        cov = (dev * w[:, None]).T @ dev / mass
        ess = mass * mass / np.sum(w * w)
        fits.append((mass, mu, cov, ess))
        pooled += cov * ess
        total += ess
    pooled /= total
    score = []
    for mass, mu, cov, ess in fits:
        rho = ndim / (ess + ndim)
        chol = np.linalg.cholesky(_ridge((1 - rho) * cov + rho * pooled))
        z = np.linalg.solve(chol, (u - mu).T)
        logdet = np.sum(np.log(np.diag(chol)))
        score.append(np.sum(z * z, axis=0) + 2 * logdet - 2 * math.log(mass))
    return np.argmin(np.asarray(score), axis=0)


def _nearest_other(z, rows) -> np.ndarray:
    """The nearest other row of ``z`` (Euclidean) to each of the rows
    ``rows``, taken in blocks of about ``_NEAREST_BLOCK`` distances, so
    that the memory does not grow as ``len(rows) * len(z)``."""
    nearest = np.zeros(len(rows), dtype=int)
    block = max(1, _NEAREST_BLOCK // max(z.size, 1))
    for start in range(0, len(rows), block):
        i = rows[start : start + block]
        dist = np.sum((z[i][:, None, :] - z[None, :, :]) ** 2, axis=-1)
        dist[np.arange(len(i)), i] = np.inf
        nearest[start : start + block] = np.argmin(dist, axis=1)
    return nearest


def _discriminant(x, side):
    """The Fisher discriminant ``w`` of ``x[~side]`` and ``x[side]``, scaled to
    a unit within-set standard deviation, and the ends ``(lo, hi)`` of the
    empty stretch between the two sets along it: ``lo = max x[~side] @ w``
    and ``hi = min x[side] @ w`` (``hi < lo`` if they overlap; ``w = 0`` and
    no stretch for sets without any spread).

    Repeated rows count once: an unmoved chain returns a copy of its seed, and
    copies would shrink the within-set scatter without adding information."""
    x, first = np.unique(x, axis=0, return_index=True)  # copies of a seed: once
    side = np.asarray(side)[first]
    a, b = x[~side], x[side]
    dev = np.concatenate([a - a.mean(0), b - b.mean(0)])
    within = _ridge(dev.T @ dev / max(len(x) - 2, 1))
    with np.errstate(all="ignore"):
        w = np.linalg.solve(within, b.mean(0) - a.mean(0))
        w = w / math.sqrt(max(w @ within @ w, 1e-300))
    if not np.all(np.isfinite(w)):  # no spread within the sets: no scale
        return np.zeros(x.shape[1]), 0.0, 0.0
    p = x @ w
    return w, float(p[~side].max()), float(p[side].min())


def _gap(x, side) -> float:
    """Empty stretch between ``x[~side]`` and ``x[side]`` along the Fisher
    discriminant, in within-set standard deviations (negative if they
    overlap; see :func:`_discriminant`)."""
    side = np.asarray(side)
    if side.all() or not side.any():
        return math.inf
    _, lo, hi = _discriminant(x, side)
    return hi - lo


def _empty_slab(x, m: int) -> tuple[float, int, float]:
    """The emptiest slab across a coordinate axis among the rows ``x``:
    ``(score, axis, cut)``.

    Along each axis, every stretch between two neighbouring rows with at
    least ``m`` rows on either side is scored by its width times ``(m - 1) /
    span``, the density of the ``m`` rows next to it on its sparser side
    (``span``: the stretch they cover). For rows drawn from a density without
    holes the score is of order ``log(len(x))``; ``cut`` is the middle of the
    best stretch. The score is 0 for fewer than ``2 m`` rows."""
    n, ndim = x.shape
    best = (0.0, -1, 0.0)
    if n < 2 * m:
        return best
    i = np.arange(m - 1, n - m)
    for axis in range(ndim):
        p = np.sort(x[:, axis])
        span = np.maximum(p[i] - p[i - m + 1], p[i + m] - p[i + 1])
        score = (p[i + 1] - p[i]) * (m - 1) / np.maximum(span, 1e-300)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (float(score[j]), axis, 0.5 * float(p[i[j]] + p[i[j] + 1]))
    return best


def _slab_parts(live, x, m: int):
    """Cut the rows ``x`` wherever the rows ``live`` leave an empty slab
    (:func:`_empty_slab` with a score of ``SLAB_SCORE``), again and again;
    returns the part of every row of ``x`` (0, 1, ...)."""
    part = np.zeros(len(x), dtype=int)
    todo, count = [(live, np.arange(len(x)))], 0
    while todo:
        points, rows = todo.pop()
        score, axis, cut = _empty_slab(points, m)
        if score < SLAB_SCORE:
            part[rows] = count
            count += 1
            continue
        low, below = points[:, axis] <= cut, x[rows, axis] <= cut
        todo += [(points[low], rows[below]), (points[~low], rows[~below])]
    return part


def _weighted_median(x, weights) -> float:
    """The median of ``x`` under ``weights``."""
    order = np.argsort(x)
    cw = np.cumsum(weights[order])
    return float(x[order][min(np.searchsorted(cw, 0.5 * cw[-1]), len(x) - 1)])


def _apart(u, logl, weights, a, b) -> bool:
    """Whether the samples ``a`` and ``b`` are separate modes: the likelihood
    dips between them (:func:`_saddle`) by at least ``SADDLE_DEPTH``, or to
    below the posterior median likelihood of both."""
    depth, saddle = _saddle(u, logl, weights, a, b)
    bulk = min(_weighted_median(logl[mask], weights[mask]) for mask in (a, b))
    return depth >= SADDLE_DEPTH or saddle < bulk


def _saddle(u, logl, weights, a, b) -> tuple[float, float]:
    """How far the likelihood dips between the samples ``a`` and ``b``, and
    the likelihood at the bottom of the dip: ``(depth, saddle)``.

    Along the Fisher discriminant of the two sets (posterior-weighted
    covariances), the stretch between their weighted medians is cut into
    ``SADDLE_BINS`` bins, and each bin's profile is the highest likelihood
    of a sample of ``a`` or ``b`` in it. ``saddle`` is the lowest profile and
    ``depth`` the lower of the two end bins' profiles minus ``saddle``
    (``inf`` and ``-inf`` if a bin is empty; ``(0, inf)``, no dip, for sets
    with the same median). Dead points cover every contour, so between
    separate modes the bins hold only the samples of low contours.
    """

    def moments(mask):
        w = weights[mask] / weights[mask].sum()
        mean = w @ u[mask]
        dev = u[mask] - mean
        return mean, (dev * w[:, None]).T @ dev

    (ma, ca), (mb, cb) = moments(a), moments(b)
    within = _ridge(0.5 * (ca + cb))
    w = np.linalg.solve(within, mb - ma)
    w /= math.sqrt(max(float(w @ within @ w), 1e-300))
    pa = _weighted_median(u[a] @ w, weights[a])
    pb = _weighted_median(u[b] @ w, weights[b])
    if not pb > pa:
        return 0.0, math.inf
    both = a | b
    at = np.floor((u[both] @ w - pa) / (pb - pa) * SADDLE_BINS).astype(int)
    inside = (at >= 0) & (at <= SADDLE_BINS)  # the median of b: the last bin
    profile = np.full(SADDLE_BINS, -np.inf)
    np.maximum.at(profile, np.minimum(at[inside], SADDLE_BINS - 1), logl[both][inside])
    saddle = float(profile.min())
    return float(min(profile[0], profile[-1]) - saddle), saddle


def _kolmogorov_pvalue(n: int, ks: float) -> float:
    """The asymptotic Kolmogorov tail of a KS distance ``ks`` of ``n`` values
    (Stephens' small-sample correction)."""
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * ks
    if lam < 0.2:  # the series converges slowly here; the tail is 1 - 1e-11
        return 1.0
    terms = (2 * (-1) ** (k - 1) * math.exp(-2 * (k * lam) ** 2) for k in range(1, 101))
    return min(max(sum(terms), 0.0), 1.0)


def _ks_uniform(values) -> tuple[float, float]:
    """KS distance of ``values`` from the uniform law on ``[0, 1]``, p-value."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = len(x)
    i = np.arange(1, n + 1)
    ks = float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    return ks, _kolmogorov_pvalue(n, ks)


def _rank_pit(ranks, nslots) -> np.ndarray:
    """``(rank + U) / nslots``: uniform on ``[0, 1)`` when each rank is uniform
    on ``0..nslots-1`` (a randomized PIT, so rows with different slot counts
    pool; ``U`` is uniform from a fixed seed, so the test is reproducible)."""
    ranks = np.asarray(ranks, dtype=np.float64)
    jitter = np.random.default_rng(0).random(ranks.shape)
    return (ranks + jitter) / np.asarray(nslots, dtype=np.float64)


@dataclass(frozen=True)
class LogZBootstrap:
    """Simulated-weights (jittered) log-evidence realizations.

    Produced by :meth:`NestedSamplingResult.logz_bootstrap`. ``logzerr`` is the
    sample standard deviation of the log-evidence realizations: the
    prior-volume-path uncertainty of a single run, not any sampling bias. The
    percentiles capture the (typically left-skewed) shape that a single
    Gaussian ``sqrt(H/nlive)`` cannot.
    """

    logz_mean: float
    logzerr: float
    logz_median: float
    logz_p16: float
    logz_p84: float
    n_realizations: int
    samples: np.ndarray


def _np_logsumexp(values, axis):
    """Return a stable ``log(sum(exp(values)))`` reduction over ``axis``."""

    values = np.asarray(values, dtype=float)
    vmax = np.max(values, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(vmax), vmax, 0.0)
    total = np.sum(np.exp(values - safe), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = safe + np.log(total)
    out = np.where(total > 0.0, out, -np.inf)
    return np.squeeze(out, axis=axis)


def _log_weights(logl, nlive_i, log_t=None):
    """Return the log posterior weights of samples in death order.

    Sample ``i`` dies with ``nlive_i[i]`` live points and shrinks the prior
    volume by ``t_i``: ``log X_i = sum_{j<=i} log t_j``, with the expected
    ``log t_i = -1 / nlive_i[i]`` unless ``log_t`` (any leading shape, the
    last axis over the samples) is given. Sample ``i`` takes the width
    ``X_{i-1} - X_i`` (``X_{-1} = 1``) and the last sample takes all the
    volume left, ``X_{N-2}``, so a constant likelihood ``L`` gives ``Z = L``.
    """
    logl = np.asarray(logl, dtype=np.float64)
    if log_t is None:
        log_t = -1.0 / np.asarray(nlive_i, dtype=np.float64)
    log_t = np.asarray(log_t, dtype=np.float64)
    log_x = np.cumsum(log_t, axis=-1)
    log_prev = np.concatenate([np.zeros_like(log_x[..., :1]), log_x[..., :-1]], -1)
    with np.errstate(divide="ignore"):
        log_width = log_prev + np.log(-np.expm1(log_t))
    log_width[..., -1] = log_prev[..., -1]
    return log_width + logl


def _evidence(logl, nlive_i):
    """Return ``(logwt, logz, logzerr)`` in float64 (see :func:`_log_weights`).

    ``logzerr`` is the spread of ``logz`` over the prior-volume path, to
    first order: the shrinkage of sample ``j`` has ``Var(log t_j) = 1 /
    n_j^2`` (``t_j ~ Beta(n_j, 1)``, ``n_j = nlive_i[j]``) and moves ``logz``
    by ``S_j = d logz / d log t_j = sum_{i>j} p_i - L_j X_j / Z`` (``p_i``
    the normalized weights; ``S = 0`` for the last sample, which takes all
    the volume left), so ``logzerr^2 = sum_j (S_j / n_j)^2``. At a constant
    live count ``n`` it is close to Skilling's ``sqrt(H / n)``. Unlike a sum
    of information increments it keeps the volume uncertainty of the deaths
    on a plateau below the posterior mass (zero-weight deaths, ``S_j``
    near 1) and is never negative; the deaths of a plateau that holds the
    remaining mass (``S_j = 0``) add none, since their widths sum to the
    volume left exactly. With few live points above a plateau the spread of
    ``logz`` is skewed, which :meth:`NestedSamplingResult.logz_bootstrap`
    (it draws the shrinkages) shows.
    """
    logl = np.asarray(logl, dtype=np.float64)
    n = np.asarray(nlive_i, dtype=np.float64)
    logwt = _log_weights(logl, n)
    logz = float(_np_logsumexp(logwt, axis=0))
    if not math.isfinite(logz):
        return logwt, logz, math.nan
    p = np.exp(logwt - logz)
    after = np.concatenate([np.cumsum(p[::-1])[::-1][1:], [0.0]])  # sum_{i>j} p_i
    with np.errstate(divide="ignore"):
        own = np.exp(logl + np.cumsum(-1.0 / n) - logz)  # L_j X_j / Z; 0 at -inf
    s = after - own
    s[-1] = 0.0
    return logwt, logz, math.sqrt(float(np.sum((s / n) ** 2)))


@dataclass(kw_only=True)
class NestedSamplingResult:
    """Container for completed nested-sampling outputs.

    The samples are the ``niter`` dead points in the order they died, then
    the final live points by increasing likelihood. ``num_delete`` points die
    per step, so the live counts at the deaths (``nlive_i``) cycle through
    ``nlive, nlive - 1, ..., nlive - num_delete + 1`` and end with
    ``nlive, ..., 1`` over the final live points. On a likelihood plateau
    (live points tied at one likelihood) the tied deaths count at
    ``nlive, nlive - 1, ...`` across the steps they take (Fowlie, Handley and
    Su 2021), so ``nlive_i`` is lower there, and so it is for the final live
    points still tied at the last step's contour (a run that ended on a
    plateau, or stopped during one).
    """

    samples_u: ArrayLike
    """Posterior samples in unit-cube coordinates."""

    samples: ArrayLike
    """Posterior samples in parameter-space coordinates."""

    logl: ArrayLike
    """Log-likelihood values associated with ``samples``."""

    logwt: ArrayLike
    """Unnormalized log posterior weights associated with ``samples``."""

    logl_birth: ArrayLike
    """Likelihood contour each sample was born above (``-inf`` for the initial
    live points), aligned with ``logl``."""

    nlive_i: ArrayLike
    """Number of live points when each sample died (dynesty's ``samples_n``),
    counting plateaus as above."""

    labels: ArrayLike | None = None
    """Cluster slot of each sample when it died (the final live points: at the
    end of the run), from the on-device clustering (:mod:`tinyns.modes`);
    :meth:`modes` reads it. ``None`` for a result built by hand."""

    insertion_ranks: ArrayLike | None = None
    """In-sampler insertion rank of every point born during the run (``niter``
    values; row ``i`` is a new point of the step of dead row ``i``): the
    number of surviving live points strictly above the step's contour with
    a likelihood below the new point's, plus a uniform place among those
    with exactly its likelihood. ``None`` for a result built by hand
    (:meth:`insertion_indices` then replays the births)."""

    insertion_slots: ArrayLike | None = None
    """The number of possible ranks of each of :attr:`insertion_ranks`: the
    survivors strictly above the contour, plus one (``nlive - num_delete +
    1`` but on a plateau)."""

    logz: float
    """Estimated log evidence."""

    logzerr: float
    """Estimated uncertainty on ``logz``."""

    ncall: int
    """Number of likelihood calls performed."""

    niter: int
    """Number of iterations (dead points)."""

    nlive: int
    """Number of live points used by the sampler."""

    num_delete: int
    """Live points deleted and replaced per step."""

    ndim: int
    """Number of sampled dimensions."""

    success: bool = True
    """True if the run converged (``dlogz``) or ended on a likelihood plateau;
    False if it stopped at ``maxiter`` or ``maxcall``, or if ``logz`` is not
    finite (status ``"invalid"``: a likelihood of +inf, or no finite
    likelihood found)."""

    message: str = ""
    """Human-readable status of the run."""

    metadata: dict[str, Any] | None = None
    """Run settings and telemetry: ``status``, ``walks``, ``dlogz``,
    ``final_delta_logz``, ``acceptance`` (of the walk steps), ``scale``,
    ``ncall_valid`` (in-cube likelihood calls), ``x64``, the hop counters,
    ``wall_time_s``, ``compile_s``, ``sampling_s``, ``resumed`` and more (the
    CHANGELOG lists the keys)."""

    def _no_weight(self) -> bool:
        """True if no sample has a positive posterior weight (``logz = -inf``:
        no finite likelihood was found)."""
        logwt = np.asarray(self.logwt, dtype=float)
        return logwt.size > 0 and not np.any(logwt > -np.inf)

    def _check_weight(self) -> None:
        if self._no_weight():
            raise ValueError(
                "no sample has a positive posterior weight (logz is -inf)"
                + (f": {self.message}" if self.message else "")
            )

    def log_weights(self) -> np.ndarray:
        """Return posterior weights normalized in log space.

        This and the other weight statistics compute in NumPy float64 on the
        host (:mod:`tinyns.math`), whatever JAX's precision.
        """

        self._check_weight()
        return normalize_log_weights(self.logwt)

    def weights(self) -> np.ndarray:
        """Return normalized linear posterior weights (NumPy float64)."""

        return np.exp(self.log_weights())

    def posterior_ess(self) -> float:
        """Return the effective sample size of the posterior weights (0 when
        no sample has a positive weight)."""

        if self._no_weight():
            return 0.0
        return effective_sample_size_from_log_weights(self.logwt)

    def max_weight_fraction(self) -> float:
        """Return the largest normalized posterior weight fraction."""

        weights = self.weights()
        if weights.size == 0:
            return 0.0
        return float(np.max(weights))

    def posterior_weight_entropy(self) -> float:
        """Return the Shannon entropy of normalized posterior weights."""

        weights = self.weights()
        if weights.size == 0:
            return 0.0
        positive = weights[weights > 0.0]
        return float(-np.sum(positive * np.log(positive)))

    def posterior_weight_entropy_fraction(self) -> float:
        """Return posterior weight entropy as a fraction of equal-weight entropy."""

        n = int(np.size(self.logwt))
        if n <= 1:
            return 0.0
        fraction = self.posterior_weight_entropy() / math.log(n)
        return float(np.clip(fraction, 0.0, 1.0))

    def live_weight_fraction(self) -> float:
        """Return the posterior weight fraction in the final live points."""

        return float(np.sum(self.weights()[int(self.niter) :]))

    def dead_weight_fraction(self) -> float:
        """Return the posterior weight fraction in the dead points."""

        return float(np.clip(1.0 - self.live_weight_fraction(), 0.0, 1.0))

    @cached_property
    def _births(self) -> tuple[np.ndarray, np.ndarray]:
        """Birth iteration of every sample (-1: initial live point), and the
        insertion index of every point born during the run.

        Rebuilt from ``logl_birth``: the ``num_delete`` points born at a step
        share its contour, the likelihood of the last point that died there,
        so sorting the births orders them by step. A point born at the step
        whose deaths are ``i - num_delete + 1 .. i`` has birth iteration
        ``i``. Replaying the deaths and births on a sorted list of the live
        likelihoods gives each new point's rank among the survivors.
        """
        logl = np.asarray(self.logl, dtype=float)
        order = np.argsort(np.asarray(self.logl_birth, dtype=float), kind="stable")
        k, niter = int(self.num_delete), int(self.niter)
        n0 = len(logl) - niter  # never born during the run
        born = np.full(len(logl), -1)
        insertion = np.zeros(niter, dtype=int)
        values = logl.tolist()
        live = sorted(values[i] for i in order[:n0])
        births = order[n0:].tolist()
        for step in range(niter // k):
            del live[:k]  # the step's deaths
            group = births[step * k : (step + 1) * k]
            for j, i in enumerate(group):
                born[i] = (step + 1) * k - 1
                insertion[step * k + j] = bisect.bisect_right(live, values[i])
            for i in group:
                bisect.insort(live, values[i])
        return born, insertion

    def insertion_indices(self) -> np.ndarray:
        """Return the insertion index of every point born during the run.

        The index is the number of the surviving live points of its step
        strictly above the step's contour (``nlive - num_delete`` of them
        without ties) with a likelihood below the new point's (ties: a
        uniform place among them), so it is
        uniform on ``0..insertion_slots-1`` for a correct constrained sampler
        (the new points of one step are ranked against the survivors only,
        not against each other). These are the ranks the sampler recorded
        (:attr:`insertion_ranks`); a result built without them replays the
        births from ``logl_birth``, which is ambiguous on a likelihood
        plateau.
        """

        if self.insertion_ranks is not None:
            return np.asarray(self.insertion_ranks, dtype=int)
        return self._births[1]

    def _insertion_slots(self) -> np.ndarray:
        if self.insertion_slots is not None:
            return np.asarray(self.insertion_slots, dtype=int)
        nslots = int(self.nlive) - int(self.num_delete) + 1
        return np.full(int(self.niter), nslots)

    def insertion_test(self, windows: int = 3) -> dict[str, Any]:
        """Return a Kolmogorov-Smirnov test of the insertion indices.

        Each index (:meth:`insertion_indices`) is mapped to ``(rank + U) /
        nslots``, with ``nslots`` its number of possible ranks
        (:attr:`insertion_slots`) and ``U`` uniform (from a fixed seed), which
        is uniform on ``[0, 1)`` for a correct sampler even when the slot
        counts differ (a likelihood plateau). These values are tested against
        the uniform law over the whole run and in ``windows`` equal
        stretches of it, so that a bias confined to part of the run (the
        narrow posterior bulk, say) is not diluted by the rest. Returns the
        pooled ``n``, ``ks`` distance and ``pvalue``, and the same per
        window (with its ``start`` and ``stop`` iteration) under
        ``"windows"``.

        With ``num_delete > 1`` the ranks of one step share its survivors,
        so they are not independent and the p-values are somewhat
        anti-conservative (on 2-D Gaussians with ``num_delete=50``, 5-12% of
        runs fall below 0.05 instead of 5%).
        """

        if int(windows) < 1:
            raise ValueError("windows must be a positive integer")
        ranks = self.insertion_indices()
        values = _rank_pit(ranks, self._insertion_slots())

        def test(start, stop):
            ks, pvalue = math.nan, math.nan
            if stop > start:
                ks, pvalue = _ks_uniform(values[start:stop])
            return {"start": start, "stop": stop, "n": stop - start, "ks": ks,
                    "pvalue": pvalue}

        edges = np.linspace(0, len(ranks), int(windows) + 1).astype(int).tolist()
        out = test(0, len(ranks))
        out["windows"] = [test(a, b) for a, b in itertools.pairwise(edges)]
        return out

    def modes(self) -> list[dict[str, Any]]:
        """Return the posterior modes, heaviest first, with their mass scatter.

        Post-processing of the run's samples: it reads the cluster labels the
        run recorded (:attr:`labels`: the id of each sample's cluster when it
        died, in one of the cross-fitted clusterings that drive the
        inter-mode hop), runs no likelihood and compiles nothing. A label is
        not a mode: the clustering's split test also cuts a curved mode into
        touching pieces, each clustering covers every mode with labels of its
        own, a cluster that is split later held several modes until then, and
        a lattice of many modes is never split at all. The modes are found in
        six passes:

        1. a label carrying less than one effective posterior sample, or with
           at most ``ndim`` distinct live points at its own posterior median
           (they span no volume, so no gap next to them can be measured), is
           set aside;
        2. two labels *touch* if both have live points at the later of their
           two posterior medians and these leave a gap of less than
           ``SEPARATION_SIGMA`` within-cluster standard deviations along
           their discriminant direction, there or at the earlier median; they
           are *apart* if they leave a gap wherever both have live points.
           Touching labels are merged, with three exceptions. A cluster is
           left out if it touches two labels that are apart and its own live
           points leave a gap between the two (a cluster of another
           clustering that never split them), or if the clusters it was split
           into ended up apart (``metadata["mode_tree"]`` records what was
           split into what). And a label that does not live to see the
           other's median only *leans* on it: modes that are separate in the
           posterior bulk are still connected at a low likelihood contour,
           so an early label joins the one mode it leans on, and none if it
           leans on several;
        3. a mode with fewer than ``UNRESOLVED_PER_DIM * ndim`` live points
           at its posterior median joins another if the live points of the
           labels left out fill the gap between the two, and a mode found by
           one clustering only is left out if no live point of another fold
           has one of its points as nearest neighbour (the descendants of a
           stuck chain stay in one fold);
        4. a mode is cut wherever its live points, at its posterior median,
           leave a slab across a coordinate axis of the unit cube empty
           (``SLAB_SCORE``): this finds the modes the sampler's clustering
           did not separate, such as the 18 of an eggbox, when planes across
           the axes of the cube separate them;
        5. every sample is relabelled by its nearest mode (hard-EM
           classification with each mode's posterior-weighted mean and
           covariance), so that a mode has its points from before it was
           split off and those of the labels left out;
        6. two modes are one if the likelihood of their samples dips by less
           than ``SADDLE_DEPTH`` along the line between them (the highest
           likelihood in each of ``SADDLE_BINS`` stretches of their
           discriminant direction) and the bottom of the dip lies above the
           posterior median likelihood of at least one of them
           (:func:`_apart`): the split test also cuts lumps off a mode whose
           contours have thin arms or a funnel neck, and their live points
           leave gaps as separate modes do, but the dead points show no
           saddle between them. In few dimensions the bulk of a mode lies
           close to its peak, and a shallow dip below the bulk of both
           separates two modes (equal Gaussians from about 2.5 standard
           deviations apart in 1-D, 3.3 in 2-D, 3.9 in 3-D, against 5.4
           from the depth alone), if the passes before separated them. A
           close mode of less mass is mostly not separated before: the
           clustering leaves it in the labels it split the heavier one from
           (in 2-D a mode of mass 0.1 or 0.2 was one mode with the other in
           nearly all runs at 4.5 to 5.5 standard deviations apart, and a
           0.1 mode in about half of them at 6.5). A lump joined to its mode
           only along a curved path (a funnel neck) is not found this way.

        Each mode's live count ``n(t)`` is rebuilt from the birth and death
        iterations. The mode is isolated from the first iteration at which
        its live points and those of every other mode pass the split test's
        threshold (:func:`tinyns.modes.split_threshold`; if they never do
        before the earlier of the two posterior medians, there is no drift
        term and ``isolation_iteration`` is that median). From then on
        random-walk chains cannot leave it, so without the hop ``n`` does a
        random walk (the urn) and the mode's mass scatters from seed to seed
        by ``urn_sd`` in ``logit(mass)``: ``1/n + 1/(N - n)`` at isolation,
        plus ``2 (1 - g) / (N^2 g)`` per iteration up to the mode's posterior
        median and ``2 g / (N^2 (1 - g))`` up to the rest's, with ``g = n /
        N``. The hop removes most of that drift (about 7x less scatter in the
        v1 bake-off), so ``urn_sd`` is an upper bound for a mode the sampler
        tracked. It treats the rest as one urn, which holds for two modes or
        modes of one shape: without the hop, three modes of different widths
        in 2 to 4 dimensions scattered 1.5 to 4 times more than ``urn_sd``.

        The live slots form ``metadata["folds"]`` cross-fitted folds, and a
        point and all its descendants stay in one fold, so each fold samples
        the modes like a run of its own. ``fold_mass`` is the mode's mass
        among each fold's samples and ``fold_sd`` the standard deviation of
        its logit over the folds: a measurement, within the run, of how much
        the mass scatters. Where the hop balances the modes the folds agree
        (``fold_sd`` 0.04 to 0.1 on the Gaussian mixtures of ``bench``);
        where it does not, each fold drifts on its own (0.4 on the 18-D
        needle target, 1 to 1.6 on an 18-D posterior whose modes have a
        funnel neck). Every fold weighs its samples with the run's prior
        volumes, so ``fold_sd`` misses their error, which moves the masses of
        all folds together (:meth:`_volume_variance`): it vanishes for modes
        of one height and grows with the prior volume between the bulk of the
        mode and that of the rest, as when the heaviest mode is also the
        narrowest. ``run_sd = sqrt(fold_sd^2 / folds + V)``, with ``V`` that
        error's variance, is the run-to-run scatter of the mass in logit,
        measured within the run. On a 4-D mixture of masses 0.75, 0.2 and
        0.05 and widths 0.25, 0.5 and 0.75 (16 runs at nlive 1000) the masses
        scattered by 0.056, 0.053 and 0.079, against a median ``run_sd`` of
        0.053, 0.046 and 0.065 and ``fold_sd / sqrt(3)`` of 0.015, 0.012 and
        0.027; for modes of one height ``run_sd`` is ``fold_sd / sqrt(3)``.
        Three folds give two degrees of freedom, so one run's value is good
        to a factor of about 2. That holds for runs that kept every mode: a
        run that lost a mode, or most of one, lies far outside its
        ``run_sd``, and the warnings of :meth:`diagnostics` need not say so.
        On 16 other seeds of the mixture above, one run kept 0.47 of the
        0.75 mode (logZ 9 ``logzerr`` low), and the masses scattered by
        0.31, 0.25 and 0.27; without it by 0.069, 0.058 and 0.101, against
        a median ``run_sd`` of 0.053, 0.048 and 0.062.

        Per mode: ``mass``, ``urn_sd``, ``fold_mass``, ``fold_sd``,
        ``run_sd`` (``None`` without folds), ``min_live`` (the smallest ``n``
        between isolation and the mode's median), ``isolation_iteration``,
        ``extinction_iteration`` (the first iteration from which the mode
        holds no live point; ``niter`` for a mode that holds some of the
        final live points), ``unresolved`` (``min_live`` below
        ``UNRESOLVED_PER_DIM * ndim``: the mass is not reliable, raise
        ``nlive``) and ``tracked``: False for a mode that the sampler's
        clustering did not separate from its neighbours, that is, one that
        pass 4 cut out of a larger one while less than ``SEPARATED_SHARE`` of
        its mass carries labels that put ``PURE_LABEL`` of their own mass in
        it. The hop did not balance such a mode, its mass scatters as
        without the hop (``run_sd`` measures by how much: each fold drifts
        on its own), and the count of such modes is a lower bound (modes
        with fewer than ``SLAB_NEIGHBOURS`` live points, or that no
        axis-aligned plane separates, stay merged). :meth:`diagnostics` also
        reports for how much of the run the clustering's slots were full
        (``mode_slots_full``: no further mode could be split off then) and
        for how much of the posterior mass the hop was off while two modes
        held live points (``mode_hop_off``). A mode lost before the end
        leaves no trace in its own run, and neither does one that never held
        more than about ``ndim`` live points per fold. A unimodal run (or a
        result without :attr:`labels`, or without posterior weight) gives one
        mode of mass 1, flagged ``unresolved`` if ``nlive`` is below
        ``UNRESOLVED_PER_DIM * ndim``: too few live points for a
        covariance-adapted walk to learn the shape of a mode (a near-Gaussian
        one can do with fewer: on a 32-D Gaussian logZ was 9 nats high at
        nlive 40, 1 at 64 and right at 100).
        ``metadata["mode_tree"]`` holds the parents of the first
        ``TREE_SIZE - 1`` cluster ids of each clustering
        (:data:`tinyns.modes.TREE_SIZE`), and ``metadata["mode_tree_dropped"]``
        counts the clusters made past them, whose parents are not recorded:
        pass 2 cannot tell that the clusters they were split from held
        several modes. The modes are worked out once per result, and each
        call returns a copy: :meth:`diagnostics`, :meth:`summary` and
        ``print(result)`` reuse them, so a result must not be changed in place
        after any of these (``dataclasses.replace`` gives a new one, whose
        modes are worked out afresh).
        """

        return copy.deepcopy(self._mode_list)

    @cached_property
    def _mode_list(self) -> list[dict[str, Any]]:
        """:meth:`modes`, worked out once per result."""
        fold = self._sample_folds()
        folds = None if fold is None else int(fold.max()) + 1
        nlive, niter = int(self.nlive), int(self.niter)
        one = [{"mass": 1.0, "urn_sd": 0.0,
                "fold_mass": None if folds is None else [1.0] * folds,
                "fold_sd": None if folds is None else 0.0,
                "run_sd": None if folds is None else 0.0,
                "min_live": nlive, "isolation_iteration": 0,
                "extinction_iteration": niter,
                "unresolved": bool(nlive < UNRESOLVED_PER_DIM * int(self.ndim)),
                "tracked": True}]
        if niter == 0 or self.labels is None or self._no_weight():
            return one
        try:
            return self._modes(one)
        except np.linalg.LinAlgError:  # a degenerate set of samples
            return one

    def _mode_samples(self):
        """Passes 1 to 6 of :meth:`modes`: the mode of every sample and, for
        each mode, whether pass 4 left it whole; ``None`` for one mode."""
        ndim, weights = int(self.ndim), self._weights
        u = np.asarray(self.samples_u, dtype=float)
        ids, label = np.unique(np.asarray(self.labels), return_inverse=True)
        median, alive = self._median, self._alive

        # 1. Labels without posterior mass, or with at most ndim distinct
        # live points at their posterior median, are set aside (-1).
        small = np.bincount(label, weights) * self.posterior_ess() < weights.sum()
        for c in np.flatnonzero(~small):
            mine = label == c
            small[c] = len(np.unique(u[alive(median(mine)) & mine], axis=0)) <= ndim
        label = np.where(small[label], -1, label)
        # 2, 3. Labels that cover one mode are merged.
        label = self._merge_labels(label, ids, u, median, alive)
        if label.max() < 0:
            label = np.zeros_like(label)
        # 4. A mode is cut where its live points leave an empty slab.
        cut, tracked = np.full(len(label), -1), []
        for c in range(int(label.max()) + 1):
            mine = label == c
            live = np.unique(u[alive(median(mine)) & mine], axis=0)
            part = _slab_parts(live, u[mine], max(SLAB_NEIGHBOURS, ndim + 1))
            cut[mine] = len(tracked) + part
            tracked += [part.max() == 0] * (int(part.max()) + 1)
        if len(tracked) < 2:
            return None
        # 5. Relabel every sample by its nearest mode, so that n(t) counts the
        # points of a mode also where the run's clustering had merged it.
        kept, label = np.unique(
            _nearest_mode(u, weights, cut, len(tracked)), return_inverse=True
        )
        # 6. Modes that the likelihood does not dip between are one mode.
        logl, group = np.asarray(self.logl, dtype=float), np.arange(len(kept))
        for a, b in itertools.combinations(range(len(kept)), 2):
            if not _apart(u, logl, weights, label == a, label == b):
                group[group == group[b]] = group[a]
        whole = [all(tracked[c] for c in kept[group == g]) for g in np.unique(group)]
        label = np.unique(group[label], return_inverse=True)[1]
        if len(whole) < 2:
            return None
        return label, whole

    @cached_property
    def _weights(self) -> np.ndarray:
        """:meth:`weights`, computed once for :meth:`modes`."""
        return self.weights()

    def _median(self, mask) -> int:
        """The posterior median iteration of the samples ``mask``."""
        cw = np.cumsum(self._weights * mask)
        return min(int(np.searchsorted(cw, 0.5 * cw[-1])), int(self.niter) - 1)

    def _alive(self, t):
        """The samples that are live points at iteration ``t``."""
        born = self._births[0]
        return (born < t) & (np.arange(len(born)) >= t)

    def _modes(self, one: list) -> list[dict[str, Any]]:
        found = self._mode_samples()
        if found is None:
            return one
        label, whole = found
        k = len(whole)
        nlive, niter, ndim = int(self.nlive), int(self.niter), int(self.ndim)
        weights, u = self._weights, np.asarray(self.samples_u, dtype=float)
        cumulative = np.cumsum(weights)
        born = self._births[0]
        median, alive = self._median, self._alive
        masks = [label == c for c in range(k)]
        pure = self._separated(label, k)
        tracked = [a or b for a, b in zip(whole, pure, strict=True)]
        threshold = split_threshold(ndim)
        out = []
        for c, mine in enumerate(masks):
            net = np.bincount(born[mine & (born >= 0)], minlength=niter) - mine[:niter]
            n = np.sum(mine & (born < 0)) + np.concatenate([[0], np.cumsum(net)[:-1]])

            def isolated(t, c=c, mine=mine):
                """Every other mode with live points passes the split test."""
                live = alive(t)
                others = [
                    live & other
                    for b, other in enumerate(masks)
                    if b != c and np.any(live & other)
                ]
                return bool(others) and np.any(live & mine) and all(
                    _fisher(u[pair], mine[pair]) >= threshold
                    for pair in (other | (live & mine) for other in others)
                )

            stop, stop_rest = median(mine), median(~mine)
            # Isolation precedes both bulks; one side may be gone after its own.
            lo = hi = min(stop, stop_rest)
            if isolated(hi):  # the first isolated iteration, by bisection
                lo = 0
                while lo < hi:
                    mid = (lo + hi) // 2
                    lo, hi = (lo, mid) if isolated(mid) else (mid + 1, hi)
            g = np.clip(n / nlive, 0.5 / nlive, 1 - 0.5 / nlive)
            variance = (
                1 / max(n[lo], 0.5)
                + 1 / max(nlive - n[lo], 0.5)
                + np.sum((2 * (1 - g) / (nlive**2 * g))[lo:stop])
                + np.sum((2 * g / (nlive**2 * (1 - g)))[lo:stop_rest])
            )
            min_live = int(n[lo : max(stop, lo + 1)].min())
            fold_mass, fold_sd = self._fold_spread(mine)
            run_sd = None
            if fold_sd is not None:
                volume = self._volume_variance(mine)
                run_sd = math.sqrt(fold_sd**2 / len(fold_mass) + volume)
            living = np.flatnonzero(n > 0)
            out.append({
                "mass": float(weights[mine].sum() / cumulative[-1]),
                "urn_sd": float(math.sqrt(variance)),
                "fold_mass": fold_mass,
                "fold_sd": fold_sd,
                "run_sd": run_sd,
                "min_live": min_live,
                "isolation_iteration": int(lo),
                "extinction_iteration": int(living[-1]) + 1 if len(living) else 0,
                "unresolved": bool(min_live < UNRESOLVED_PER_DIM * ndim),
                "tracked": tracked[c],
            })
        return sorted(out, key=lambda mode: -mode["mass"])

    def _separated(self, label, k: int) -> list[bool]:
        """Whether the recorded labels separate each of the ``k`` modes
        (``label``: the mode of every sample): at least ``SEPARATED_SHARE``
        of the mode's posterior mass carries labels that put at least
        ``PURE_LABEL`` of their mass in it. Pass 4 of :meth:`modes` may cut
        out a mode that the clustering did separate, when passes 2 and 3
        did not assemble its labels."""
        raw = np.unique(np.asarray(self.labels), return_inverse=True)[1]
        held = np.zeros((raw.max() + 1, k))
        np.add.at(held, (raw, label), self._weights)
        pure = held >= PURE_LABEL * held.sum(axis=1, keepdims=True)
        share = np.sum(held * pure, axis=0) / np.maximum(held.sum(axis=0), 1e-300)
        return [bool(x >= SEPARATED_SHARE) for x in share]

    def _sample_folds(self):
        """The cross-fitted fold of every sample (``None`` without
        ``metadata["folds"]``). A label is ``id * folds + j`` with ``j`` the
        clustering of the next fold, ``(fold + 1) mod folds``, and a point
        and all its descendants stay in one fold."""
        folds = (self.metadata or {}).get("folds")
        if not folds or self.labels is None:
            return None
        return (np.asarray(self.labels) % folds - 1) % folds

    def _fold_spread(self, mine) -> tuple[list[float] | None, float | None]:
        """The mass of the samples ``mine`` among each fold's samples, and the
        standard deviation of its logit over the folds (``(None, None)``
        without folds). The logit takes the masses no closer to 0 or 1 than
        half a live point of a fold."""
        fold = self._sample_folds()
        if fold is None:
            return None, None
        folds, weights = int(fold.max()) + 1, self._weights
        mass = np.array([
            weights[mine & (fold == f)].sum() / max(weights[fold == f].sum(), 1e-300)
            for f in range(folds)
        ])
        half = 0.5 * folds / int(self.nlive)
        g = np.clip(mass, half, 1.0 - half)
        return mass.tolist(), float(np.std(np.log(g / (1.0 - g)), ddof=1))

    def _volume_variance(self, mine) -> float:
        """The variance of the logit of the mass of the samples ``mine`` from
        the prior volumes, which every fold shares (first order in the
        ``Beta(n_i, 1)`` shrinkages of :meth:`logz_bootstrap`). An error ``e``
        in the log shrinkage of death ``i`` scales the weights of every later
        sample by ``exp(e)``, which moves the logit by ``e (F_rest - F)``,
        with ``F`` and ``F_rest`` the shares of the mass of the mode and of
        the rest in the samples up to ``i``; ``e`` has variance ``1 /
        n_i^2``. It vanishes where the mode and the rest die together (modes
        of one height) and grows with the prior volume between their bulks.
        The widths of single samples are left out: their scatter is that of
        the posterior samples, which the folds measure."""
        weights = self._weights
        mode, rest = np.cumsum(weights * mine), np.cumsum(weights * ~mine)
        gap = rest / max(rest[-1], 1e-300) - mode / max(mode[-1], 1e-300)
        return float(np.sum((gap / np.asarray(self.nlive_i, dtype=float)) ** 2))

    def _merge_labels(self, label, ids, u, median, alive):
        """Passes 2 and 3 of :meth:`modes`: the mode of every sample (-1:
        none).

        ``label`` holds the labels ``0..len(ids)-1`` (-1: set aside) and
        ``ids`` the recorded label each stands for; ``median(mask)`` is the
        posterior median iteration of the samples ``mask`` and ``alive(t)``
        the samples that are live at iteration ``t``.
        """
        k = len(ids)
        metadata = self.metadata or {}
        rows = [np.flatnonzero(label == c) for c in range(k)]
        kept = [c for c in range(k) if len(rows[c])]
        born = self._births[0]
        when = {c: median(label == c) for c in kept}

        def points(c, t):
            i = rows[c]
            return u[i[(born[i] < t) & (i >= t)]]

        def gap(a, b, t):
            xa, xb = points(a, t), points(b, t)
            if not len(xa) or not len(xb):
                return None
            side = np.arange(len(xa) + len(xb)) >= len(xa)
            return _gap(np.concatenate([xa, xb]), side)

        # touch[a, b]: both labels have live points at the later of their two
        # medians, and leave no gap there or at the earlier one. leans[a]:
        # the labels b of a later median that a does not live to see and
        # touches at its own. apart[a, b]: a gap wherever both have live
        # points.
        touch = np.zeros((k, k), bool)
        apart = np.zeros((k, k), bool)
        leans: dict[int, list[int]] = {c: [] for c in kept}
        for a, b in itertools.combinations(kept, 2):
            early, late = (a, b) if when[a] <= when[b] else (b, a)
            first, last = gap(a, b, when[early]), gap(a, b, when[late])
            gaps = [g for g in (first, last) if g is not None]
            if not gaps:
                continue
            if min(gaps) > SEPARATION_SIGMA:
                apart[a, b] = apart[b, a] = True
            elif last is None:
                leans[early].append(late)
            else:
                touch[a, b] = touch[b, a] = True

        slot = {int(raw): c for c, raw in enumerate(ids)}
        children: dict[int, list[int]] = {}
        for child, parent in metadata.get("mode_tree") or []:
            children.setdefault(int(parent), []).append(int(child))
        composite = np.zeros(k, bool)
        group = np.arange(k)

        def find(c):
            while group[c] != c:
                group[c] = group[group[c]]
                c = group[c]
            return c

        def heirs(raw):
            """The label ``raw`` if it stands for part of a mode, else the
            nearest descendants that do."""
            c = slot.get(raw)
            if c is not None and len(rows[c]) and not composite[c]:
                return [c]
            return [h for child in children.get(raw, []) for h in heirs(child)]

        def tainted(raw):
            """Whether the label ``raw`` or one of its descendants belongs to
            no mode."""
            c = slot.get(raw)
            if c is not None and composite[c]:
                return True
            return any(tainted(child) for child in children.get(raw, []))

        def spans(q, a, b):
            """Whether the live points of ``q`` lie on both sides of the gap
            between ``a`` and ``b`` (more than ``ndim`` on either) and leave
            a gap themselves."""
            for t in {when[q], when[a], when[b]}:
                xa, xb, xq = points(a, t), points(b, t), points(q, t)
                if not (len(xa) and len(xb) and len(xq)):
                    continue
                side = np.arange(len(xa) + len(xb)) >= len(xa)
                w, lo, hi = _discriminant(np.concatenate([xa, xb]), side)
                xq = np.unique(xq, axis=0)
                over = xq @ w > 0.5 * (lo + hi)
                if (
                    min(over.sum(), len(over) - over.sum()) > xq.shape[1]
                    and _gap(xq, over) > SEPARATION_SIGMA
                ):
                    return True
            return False

        # A cluster belongs to no mode if it touches (or leans on) two
        # labels that are apart and its own live points leave a gap between
        # the two: a cluster of another clustering that covers both.
        for q in kept:
            near = [c for c in kept if touch[q, c] or c in leans[q] or q in leans[c]]
            composite[q] = any(
                apart[a, b] and spans(q, a, b)
                for a, b in itertools.combinations(near, 2)
            )
        # Nor does it if the clusters it was split into ended in separate
        # modes. Touching labels are merged from the last cluster made to
        # the first, so that the parts of a cluster are merged before it.
        whole: list[int] = []
        for q in sorted(kept, key=lambda c: -rows[c][0]):
            below = children.get(int(ids[q]), [])
            parts = [h for child in below for h in heirs(child)]
            composite[q] |= any(tainted(child) for child in below) or any(
                apart[a, b] and find(a) != find(b)
                for a, b in itertools.combinations(parts, 2)
            )
            if not composite[q]:
                for c in whole:
                    if touch[q, c]:
                        group[find(c)] = find(q)
                whole.append(q)
        # A group of labels is a mode, or the early part of the one mode its
        # labels lean on; it belongs to none if they lean on several. The
        # groups are taken by decreasing time of their last median.
        last: dict[int, int] = {}
        for c in whole:
            last[find(c)] = max(last.get(find(c), -1), when[c])
        mode: dict[int, int] = {}
        for r in sorted(last, key=lambda r: -last[r]):
            targets = {
                mode[find(b)]
                for c in whole
                if find(c) == r
                for b in leans[c]
                if not composite[b] and find(b) in mode and find(b) != r
            }
            if not targets:
                mode[r] = len(mode)
            else:
                targets -= {-1}
                mode[r] = targets.pop() if len(targets) == 1 else -1
        of_label = np.full(k + 1, -1)
        for c in whole:
            of_label[c] = mode[find(c)]

        # 3. A mode with fewer than UNRESOLVED_PER_DIM * ndim live points at
        # its posterior median is part of another if the live points of the
        # labels that belong to no mode fill the gap between the two.
        def live_points(m):
            """The distinct live points at the median of mode ``m``, the mode
            of each (-1: none) and its row among the samples."""
            mine = of_label[label] == m
            live = alive(median(mine))
            x, first = np.unique(u[live], axis=0, return_index=True)
            return x, of_label[label][live][first], np.flatnonzero(live)[first]

        found = sorted(set(of_label[:k]) - {-1})
        joint = {m: m for m in found}
        for m in found:
            x, owner, _ = live_points(m)
            small = np.sum(owner == m) < UNRESOLVED_PER_DIM * u.shape[1]
            for rival in found if small else []:
                pair = (owner == m) | (owner == rival)
                if rival != m and np.any(owner == rival):
                    w, lo, hi = _discriminant(x[pair], owner[pair] == rival)
                    p, mid = x[owner < 0] @ w, 0.5 * (lo + hi)
                    lo, hi = max([lo, *p[p <= mid]]), min([hi, *p[p > mid]])
                    if hi - lo <= SEPARATION_SIGMA:
                        low, high = sorted((joint[m], joint[rival]))
                        joint = {a: low if b == high else b for a, b in joint.items()}
        of_label = np.array([joint.get(m, -1) for m in of_label])
        # A mode whose labels are all of one clustering holds points of one
        # fold only. It is left out if no live point of another fold has one
        # of them as nearest neighbour: a point and its descendants stay in
        # one fold, so these are the descendants of a stuck chain.
        folds = metadata.get("folds")
        for m in sorted(set(joint.values())) if folds else []:
            members = np.flatnonzero(of_label[:k] == m)
            if len({int(ids[c]) % folds for c in members}) > 1:
                continue
            x, owner, at = live_points(m)
            other = np.asarray(self.labels)[at] % folds != int(ids[members[0]]) % folds
            dev = x - x.mean(0)
            chol = np.linalg.cholesky(_ridge(dev.T @ dev / max(len(x) - 1, 1)))
            z = np.linalg.solve(chol, dev.T).T
            nearest = _nearest_other(z, np.flatnonzero(other))
            if not (other.any() and np.any(owner[nearest] == m)):
                of_label[members] = -1
        return np.unique(of_label, return_inverse=True)[1][label] - 1

    def _mode_slots(self) -> tuple[int | None, float | None]:
        """The most clusters one of the run's clusterings held at a time, and
        the fraction of the iterations at which all ``C_MAX`` slots were in
        use (``metadata["mode_history"]``; ``(None, None)`` without it)."""
        history = (self.metadata or {}).get("mode_history")
        if not history or int(self.niter) == 0:
            return None, None
        starts = [int(row[0]) for row in history] + [int(self.niter)]
        full = sum(
            stop - start
            for row, start, stop in zip(history, starts, starts[1:], strict=False)
            if row[1] >= C_MAX
        )
        return max(int(row[1]) for row in history), full / int(self.niter)

    def _history_mean(self, start: int, stop: int, value) -> float | None:
        """The posterior-weighted mean of ``value(clusters, eligible)`` over
        the dead points of iterations ``start`` to ``stop - 1``, from the
        record of the step at which each died (``metadata["mode_history"]``:
        the most clusters and the fewest eligible ones of a clustering);
        ``None`` without that record."""
        history = (self.metadata or {}).get("mode_history")
        niter = int(self.niter)
        if not history or niter == 0:
            return None
        start = min(max(int(start), 0), niter - 1)
        stop = min(max(int(stop), start + 1), niter)
        starts = [int(row[0]) for row in history]
        at = np.maximum(np.searchsorted(starts, np.arange(niter), side="right") - 1, 0)
        values = np.array([float(value(int(row[1]), int(row[2]))) for row in history])
        weights = self._weights[start:stop]
        return float(weights @ values[at][start:stop] / max(weights.sum(), 1e-300))

    def _hop_window(self, modes) -> tuple[int, int]:
        """The iterations ``[start, stop)`` during which the hop had modes to
        balance: from the first isolation of one of ``modes`` to the last
        iteration at which two tracked modes (two modes, if fewer are
        tracked) still held live points. After that, one mode is left and
        the hop is off by necessity; where the heaviest mode is also the
        tallest, that stretch holds most of the posterior mass."""
        start = min(mode["isolation_iteration"] for mode in modes)
        ends = [mode["extinction_iteration"] for mode in modes if mode["tracked"]]
        if len(ends) < 2:
            ends = [mode["extinction_iteration"] for mode in modes]
        stop = sorted(ends)[-2] if len(ends) > 1 else int(self.niter)
        return start, stop

    def _hop_off(self, start: int, stop: int) -> float | None:
        """The fraction of the posterior mass of iterations ``start`` to
        ``stop - 1`` during which a clustering had fewer than two eligible
        clusters, so that the hop was off in its fold."""
        return self._history_mean(start, stop, lambda clusters, eligible: eligible < 2)

    def _lumps(self, start: int, stop: int, nmodes: int) -> float | None:
        """The posterior-weighted mean number of clusters per mode, of the
        clustering with the most, over iterations ``start`` to ``stop - 1``."""
        return self._history_mean(
            start, stop, lambda clusters, eligible: clusters / nmodes
        )

    def _mode_warnings(self, modes) -> list[str]:
        """The warnings of :meth:`diagnostics` about ``modes``."""
        warnings = [
            f"mode {i} (mass {mode['mass']:.3g}) is unresolved: it held "
            f"{mode['min_live']} live points; raise nlive"
            for i, mode in enumerate(modes)
            if mode["unresolved"] and len(modes) > 1
        ]
        if len(modes) == 1 and modes[0]["unresolved"]:
            least = math.ceil(UNRESOLVED_PER_DIM * int(self.ndim))
            warnings.append(
                f"mode 0 (mass 1) is unresolved: the run held {modes[0]['min_live']} "
                f"live points, fewer than the {least} ({UNRESOLVED_PER_DIM:g} * "
                "ndim) a covariance-adapted walk may need to learn the shape of "
                "a mode, and logZ may be biased beyond logzerr (a near-Gaussian "
                "mode can do with fewer); raise nlive"
            )
        most, full = self._mode_slots()
        loose = sum(not mode["tracked"] for mode in modes)
        if loose:
            held = "" if most is None else f" (it held at most {most} clusters)"
            warnings.append(
                f"{loose} of the {len(modes)} modes were not separated by the "
                f"sampler's clustering{held}: the inter-mode hop did not balance "
                "them, so their masses scatter from run to run as without the "
                "hop (by about run_sd), and there may be more modes than were "
                "found"
            )
        warnings += self._hop_warnings(modes)
        if full and len(modes) > 1:
            text = (
                f"the {C_MAX} cluster slots were full for {100 * full:.0f}% of "
                "the run"
            )
            if most is not None and most > len(modes):
                text += (
                    f", with {most} clusters for the {len(modes)} modes found: "
                    "the clustering cut modes into pieces, and while the slots "
                    "were full it could split off no further mode"
                )
            else:
                text += (
                    f": there may be more modes than the {len(modes)} found, and "
                    "the mode masses are not reliable"
                )
            warnings.append(text)
        return warnings

    def _hop_warnings(self, modes) -> list[str]:
        """The warning of :meth:`diagnostics` when the hop balanced two or
        more modes that the clustering separated only in part: while two of
        them held live points after they separated (:meth:`_hop_window`),
        the clustering held at least ``LUMPS_WARN`` clusters per mode on
        average over the posterior mass (it cut modes into lumps, and the
        hop's ellipsoids cover such modes poorly), or the hop was off for at
        least ``HOP_OFF_WARN`` of that mass, or, over the whole run, it
        accepted less than ``HOP_ACCEPT_WARN`` of its proposals. The warning
        names only the causes that passed their threshold. The masses then
        scatter from run to run more than where the hop works, and the spread
        of the folds' masses measures by how much (``run_sd``)."""
        if sum(mode["tracked"] for mode in modes) < 2:
            return []  # the warning about untracked modes says it
        start, stop = self._hop_window(modes)
        lumps = self._lumps(start, stop, len(modes))
        off = self._hop_off(start, stop)
        acceptance = (self.metadata or {}).get("hop_acceptance")
        lumpy = lumps is not None and lumps >= LUMPS_WARN
        idle = off is not None and off >= HOP_OFF_WARN
        rare = acceptance is not None and acceptance < HOP_ACCEPT_WARN
        parts = []
        if lumpy:
            parts.append(
                f"the sampler's clustering held {lumps:.2g} clusters per mode "
                "on average (it cut modes into lumps)"
            )
        if idle:
            parts.append(
                "the inter-mode hop was off (fewer than two eligible clusters) "
                f"for {100 * off:.0f}% of the posterior mass"
            )
        if parts:
            parts[0] = (
                "after the modes separated and while two of them held live "
                "points, " + parts[0]
            )
        if rare:  # the acceptance is counted over the whole run
            hop = "it" if idle else "the inter-mode hop"
            parts.append(
                f"over the run {hop} accepted {100 * acceptance:.2g}% of its "
                "proposals"
            )
        if not parts:
            return []
        text = (
            ", ".join(parts[:-1])
            + (", and " if len(parts) > 1 else "") + parts[-1]
            + ": the hop balanced the modes only in part, so their masses may "
            "scatter from run to run more than where it works"
        )
        spread = [
            (i, mode) for i, mode in enumerate(modes)
            if i and mode["fold_sd"] is not None
        ]
        if spread:
            folds = len(spread[0][1]["fold_mass"])
            text += f". The {folds} cross-fitted folds gave " + "; ".join(
                f"mode {i} the masses "
                + ", ".join(f"{m:.3g}" for m in mode["fold_mass"])
                + f" (fold_sd {mode['fold_sd']:.2f} in logit: a run-to-run "
                f"scatter, run_sd, of about {mode['run_sd']:.2g})"
                for i, mode in spread
            )
        return [text]

    def information(self) -> float:
        """Return the nested-sampling information from posterior weights."""

        return float(_information(self.logwt, self.logl, float(self.logz)))

    def logz_bootstrap(
        self, n_realizations: int = 256, seed: int = 0
    ) -> LogZBootstrap:
        """Return simulated-weights (jittered) log-evidence realizations.

        The analytic ``logzerr`` is a Gaussian approximation of the
        prior-volume path uncertainty. This estimator instead draws every
        shrinkage ``t_i`` from its ``Beta(n_i, 1)`` law (``log t_i = log(U_i) /
        n_i`` with ``U_i`` uniform and ``n_i = nlive_i[i]``, final live points
        included), rebuilds the weights over the stored likelihoods with the
        sampler's convention (:func:`_log_weights`) and recomputes ``logz`` for
        ``n_realizations`` independent volume paths. It is pure
        post-processing: no likelihood is evaluated.

        The returned ``logzerr`` (the realization standard deviation) captures
        the skewed path uncertainty. It does not capture sampling bias (e.g.
        under-decorrelated walks), so it is a lower bound on the true
        single-run uncertainty; a biased insertion-rank distribution can still
        inflate seed-to-seed scatter beyond this estimate.
        """

        if int(n_realizations) < 1:
            raise ValueError("n_realizations must be at least 1")
        nlive_i = np.asarray(self.nlive_i, dtype=float)
        if nlive_i.size == 0 or np.any(nlive_i <= 0):
            raise ValueError("nlive_i must hold positive live counts")

        rng = np.random.default_rng(seed)
        uniforms = rng.random((int(n_realizations), nlive_i.size))
        with np.errstate(divide="ignore"):
            log_t = np.log(uniforms) / nlive_i
        samples = _np_logsumexp(_log_weights(self.logl, nlive_i, log_t), axis=1)

        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            nan = math.nan
            return LogZBootstrap(nan, nan, nan, nan, nan, int(n_realizations), samples)
        return LogZBootstrap(
            logz_mean=float(np.mean(finite)),
            logzerr=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            logz_median=float(np.median(finite)),
            logz_p16=float(np.percentile(finite, 16.0)),
            logz_p84=float(np.percentile(finite, 84.0)),
            n_realizations=int(n_realizations),
            samples=samples,
        )

    def diagnostics(self) -> dict[str, object]:
        """Return run diagnostics and warnings as a plain dictionary.

        Besides the weight statistics it holds the pooled insertion-test
        p-value (``insertion_pvalue``; :meth:`insertion_test` has the
        windows), the posterior ``modes`` (:meth:`modes`),
        ``mode_slots_full``, the fraction of the run during which every
        cluster slot of the sampler's clustering was in use, two averages
        over the posterior mass after the modes separated and while two of
        them held live points (the whole run for one mode): ``mode_lumps``,
        the number of clusters per mode found (of the clustering with the
        most), and ``mode_hop_off``, the fraction during which a clustering
        had fewer than two eligible clusters (the hop was off in its fold),
        and ``hop_acceptance`` (``None`` for a result without those
        records).
        The warnings name each unresolved mode (a single mode too) and the
        modes the clustering did not separate (their number is a lower
        bound); they say when the hop balanced the modes only in part
        (``mode_lumps`` at least ``LUMPS_WARN``, ``mode_hop_off`` at least
        ``HOP_OFF_WARN`` or ``hop_acceptance`` below ``HOP_ACCEPT_WARN``;
        each cause is named only when it passed its threshold), with the
        folds' masses and the ``run_sd`` of each mode, and when the slots were
        full in a run of several modes.

        A run that ended on a likelihood plateau (``metadata["status"]`` is
        ``"plateau"``) stopped where it should: the final live points are
        the plateau and carry its posterior mass whatever ``dlogz``, so the
        checks of the final live weight and of ``final_delta_logz`` are
        skipped and one note gives that mass (more live points can find a
        higher region that none of them reached). The checks of the final
        live weight are skipped on an ``"invalid"`` run too, which neither
        ``dlogz`` nor ``nlive`` mends. A result without posterior weight
        (``logz = -inf``) has ``posterior_ess`` 0 and NaN weight fractions.
        """

        metadata = {} if self.metadata is None else self.metadata
        plateau = metadata.get("status") == "plateau"
        # The final live weight says nothing of the stopping rule on a
        # plateau (it is the plateau's) or at a likelihood of +inf.
        check_stop = metadata.get("status") not in ("plateau", "invalid")
        weighted = not self._no_weight()
        posterior_ess = self.posterior_ess()
        nposterior = int(np.size(self.logwt))
        max_weight_fraction = entropy_fraction = math.nan
        live_weight_fraction = dead_weight_fraction = math.nan
        if weighted:
            max_weight_fraction = self.max_weight_fraction()
            entropy_fraction = self.posterior_weight_entropy_fraction()
            live_weight_fraction = self.live_weight_fraction()
            dead_weight_fraction = self.dead_weight_fraction()
        insertion = self.insertion_test()
        modes = self.modes()
        start, stop = self._hop_window(modes)
        dlogz = metadata.get("dlogz")
        final_delta_logz = metadata.get("final_delta_logz")
        acceptance = metadata.get("acceptance")
        warnings: list[str] = []

        if not self.success:
            warnings.append(self.message)
        if posterior_ess < 100.0:
            warnings.append("low posterior ESS")
        if max_weight_fraction > 0.1:
            warnings.append("posterior dominated by a small number of weighted samples")
        if entropy_fraction < 0.5:
            warnings.append("low posterior weight entropy")
        if live_weight_fraction > 0.5 and check_stop:
            warnings.append(
                "final live points carry most posterior weight; consider tighter "
                "dlogz or more live points"
            )
        if (
            live_weight_fraction > 0.25
            and dlogz is not None
            and dlogz >= 0.1
            and check_stop
        ):
            warnings.append(
                "large final-live weight fraction; evidence may be sensitive to "
                "stopping"
            )
        if (
            self.success
            and check_stop
            and None not in (final_delta_logz, dlogz)
            and final_delta_logz > dlogz
        ):
            warnings.append("successful run has final_delta_logz above requested dlogz")
        if plateau and weighted:
            warnings.append(self._plateau_note())
        if not math.isfinite(float(self.logzerr)):
            warnings.append("logzerr is not finite")
        if acceptance is not None and acceptance < 0.01:
            warnings.append("low replacement acceptance")
        # Bonferroni over the pooled test and the windows.
        pvalues = [insertion["pvalue"]] + [w["pvalue"] for w in insertion["windows"]]
        if insertion["n"] >= 20 and min(pvalues) * len(pvalues) < 0.01:
            warnings.append(
                "insertion indices look non-uniform; constrained sampler may be "
                "biased or poorly mixed"
            )
        warnings.extend(self._mode_warnings(modes))
        if nposterior < self.nlive + 10:
            warnings.append("very few dead points")

        return {
            "success": self.success,
            "message": self.message,
            "logz": float(self.logz),
            "logzerr": float(self.logzerr),
            "information": self.information(),
            "niter": int(self.niter),
            "ncall": int(self.ncall),
            "nlive": int(self.nlive),
            "num_delete": int(self.num_delete),
            "ndim": int(self.ndim),
            "nposterior": nposterior,
            "posterior_ess": posterior_ess,
            "max_weight_fraction": max_weight_fraction,
            "posterior_weight_entropy_fraction": entropy_fraction,
            "live_weight_fraction": live_weight_fraction,
            "dead_weight_fraction": dead_weight_fraction,
            "final_delta_logz": final_delta_logz,
            "acceptance": acceptance,
            "insertion_pvalue": insertion["pvalue"],
            "modes": modes,
            "mode_slots_full": self._mode_slots()[1],
            "mode_lumps": self._lumps(start, stop, len(modes)) if weighted else None,
            "mode_hop_off": self._hop_off(start, stop) if weighted else None,
            "hop_acceptance": metadata.get("hop_acceptance"),
            "warnings": warnings,
        }

    def _plateau_note(self) -> str:
        """The note of :meth:`diagnostics` on a run that ended on a plateau:
        the posterior mass of the final live points tied at the highest
        likelihood (the plateau)."""
        niter = int(self.niter)
        live = np.asarray(self.logl, dtype=float)[niter:]
        top = live == live.max() if live.size else live.astype(bool)
        mass = float(np.sum(self._weights[niter:][top]))
        return (
            f"the run ended on a likelihood plateau, which holds {mass:.3g} of "
            "the posterior in the final live points on it (a tighter dlogz does "
            "not change that; raise nlive if a higher region could have been "
            "missed)"
        )

    def resample_equal(self, key, n: int | None = None) -> np.ndarray:
        """Return ``n`` equally weighted posterior samples (systematic resampling).

        ``key`` is a JAX PRNG key or an int seed; ``n`` defaults to the
        posterior effective sample size. Sample ``i`` comes ``floor(n w_i)``
        or ``ceil(n w_i)`` times (:func:`tinyns.math.systematic_resample`), so
        the samples are not independent draws, and they are returned in a
        random order: in the order of ``samples`` (by increasing likelihood)
        a prefix or a split of them would not be a posterior sample. Returns a
        NumPy array.
        """

        if n is None:
            n = max(1, int(self.posterior_ess()))
        try:
            if isinstance(n, bool) or getattr(n, "dtype", None) == np.bool_:
                raise TypeError
            n = operator.index(n)  # an int, NumPy or JAX integer; not 5.0
        except TypeError:
            raise TypeError("n must be an integer") from None
        if n < 1:
            raise ValueError("n must be at least 1")
        self._check_weight()

        key_start, key_order = random.split(as_key(key))
        indices = systematic_resample(key_start, self.logwt, n)
        seed = np.asarray(random.key_data(key_order))
        order = np.random.default_rng(seed).permutation(n)
        return np.asarray(self.samples)[indices[order]]

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the result.

        With more than one posterior mode it ends with a per-mode table
        (:meth:`modes`) and the warnings about the modes; the columns are the
        mode's ``mass``, three standard deviations of its logit, and
        ``min_live``. ``run sd`` (``run_sd``) is the run-to-run scatter of the
        mass, measured within the run; ``fold sd`` (``fold_sd``) its spread
        over the cross-fitted folds; ``sd w/o hop`` (``urn_sd``) the scatter of
        a run without the inter-mode hop, from the live counts (for two modes;
        with three or more of different shapes it is too low). The warning
        about an unresolved mode is printed for a single mode too.
        """

        insertion = self.insertion_test()
        lines = [
            f"logz: {self.logz} +/- {self.logzerr}",
            f"niter: {self.niter}  ncall: {self.ncall}  nlive: {self.nlive}  "
            f"num_delete: {self.num_delete}  ndim: {self.ndim}",
            f"posterior ESS: {self.posterior_ess():.1f}",
            f"insertion KS p-value: {insertion['pvalue']:.3g} (windows: "
            + ", ".join(f"{w['pvalue']:.3g}" for w in insertion["windows"])
            + ")",
            f"success: {self.success}",
            f"message: {self.message}",
        ]
        modes = self.modes()
        warnings = self._mode_warnings(modes)
        if len(modes) > 1:
            lines.append("mode      mass    run sd   fold sd  sd w/o hop  min live")
            for i, mode in enumerate(modes):
                flag = "  unresolved: raise nlive" if mode["unresolved"] else ""
                flag += "" if mode["tracked"] else "  not tracked"
                run_sd, fold_sd = (
                    "       -" if mode[name] is None else f"{mode[name]:8.2f}"
                    for name in ("run_sd", "fold_sd")
                )
                lines.append(
                    f"{i:4d}  {mode['mass']:8.4f}  {run_sd}  {fold_sd}  "
                    f"{mode['urn_sd']:10.2f}  {mode['min_live']:8d}{flag}"
                )
            warnings = [text for text in warnings if "is unresolved" not in text]
        lines += ["warning: " + text for text in warnings]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def save_npz(self, path) -> None:
        """Save the final result to a compressed NumPy ``.npz`` file (with
        :attr:`labels` and the insertion ranks when present).

        The file is ``path`` itself (no ``.npz`` is appended; a symbolic
        link is written through), written atomically like a checkpoint
        (:mod:`tinyns.checkpoint`). ``path`` may also be a writable binary
        file object.
        """

        optional = {
            name: np.asarray(getattr(self, name))
            for name in _RESULT_NPZ_OPTIONAL
            if getattr(self, name) is not None
        }
        arrays = {
            **{name: np.asarray(getattr(self, name)) for name in _RESULT_NPZ_ARRAYS},
            **optional,
            **{
                name: np.asarray(kind(getattr(self, name)))
                for name, kind in _RESULT_NPZ_SCALARS.items()
            },
            "metadata_json": np.asarray(json.dumps(self.metadata, default=_jsonable)),
            "format_version": np.asarray(_RESULT_NPZ_FORMAT_VERSION),
        }
        if hasattr(path, "write"):
            np.savez_compressed(path, **arrays)
        else:
            _atomic_savez(path, arrays, compressed=True)

    @classmethod
    def load_npz(cls, path) -> NestedSamplingResult:
        """Load a result previously written by :meth:`save_npz`.

        The arrays are NumPy arrays of the dtypes saved, whatever JAX's
        precision: a result saved under x64 keeps its float64 values.
        """

        with np.load(path) as data:
            required = {*_RESULT_NPZ_ARRAYS, *_RESULT_NPZ_SCALARS}
            required |= {"metadata_json", "format_version"}
            missing = sorted(required - set(data.files))
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"missing required result .npz keys: {joined}")

            format_version = str(_npz_scalar(data["format_version"]))
            if format_version != _RESULT_NPZ_FORMAT_VERSION:
                raise ValueError(
                    "unknown result .npz format_version: "
                    f"{format_version!r}; expected {_RESULT_NPZ_FORMAT_VERSION!r}"
                )
            return cls(
                **{name: np.asarray(data[name]) for name in _RESULT_NPZ_ARRAYS},
                **{
                    name: np.asarray(data[name]) if name in data.files else None
                    for name in _RESULT_NPZ_OPTIONAL
                },
                **{
                    name: kind(_npz_scalar(data[name]))
                    for name, kind in _RESULT_NPZ_SCALARS.items()
                },
                metadata=json.loads(str(_npz_scalar(data["metadata_json"]))),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python dictionary representation of the result
        (the fields :meth:`save_npz` writes; ``labels`` and the insertion
        ranks may be ``None``)."""

        out = {name: getattr(self, name) for name in _RESULT_NPZ_ARRAYS}
        out.update({name: getattr(self, name) for name in _RESULT_NPZ_OPTIONAL})
        out.update({name: getattr(self, name) for name in _RESULT_NPZ_SCALARS})
        out["metadata"] = None if self.metadata is None else dict(self.metadata)
        return out

    def to_numpy(self) -> dict[str, object]:
        """Return a plain dictionary with array fields converted to NumPy arrays."""

        out = {name: np.asarray(getattr(self, name)) for name in _RESULT_NPZ_ARRAYS}
        for name in _RESULT_NPZ_OPTIONAL:
            value = getattr(self, name)
            out[name] = None if value is None else np.asarray(value)
        out.update(
            {
                name: kind(getattr(self, name))
                for name, kind in _RESULT_NPZ_SCALARS.items()
            }
        )
        out["metadata"] = None if self.metadata is None else dict(self.metadata)
        return out

    def to_dynesty_dict(self) -> dict[str, object]:
        """Return a lightweight dynesty-compatibility dictionary.

        This is not a full dynesty ``Results`` object, only a lightweight
        compatibility dict using dynesty-like keys where tinyns has matching
        fields (``eff`` is dynesty's ``100 * niter / ncall``).
        """

        return {
            "samples": np.asarray(self.samples),
            "samples_u": np.asarray(self.samples_u),
            "logl": np.asarray(self.logl),
            "logwt": np.asarray(self.logwt),
            "logz": float(self.logz),
            "logzerr": float(self.logzerr),
            "ncall": int(self.ncall),
            "niter": int(self.niter),
            "nlive": int(self.nlive),
            "samples_n": np.asarray(self.nlive_i),
            "eff": 100.0 * int(self.niter) / max(int(self.ncall), 1),
        }
