"""Result containers returned by :class:`tinyns.NestedSampler`."""

from __future__ import annotations

import bisect
import itertools
import json
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax.numpy as jnp
import numpy as np

from tinyns.math import (
    effective_sample_size_from_log_weights,
    normalize_log_weights,
    systematic_resample,
)
from tinyns.modes import C_MAX, split_threshold

ArrayLike = Any

_RESULT_NPZ_FORMAT_VERSION = "tinyns-result-npz-v3"
_RESULT_NPZ_ARRAYS = ("samples_u", "samples", "logl", "logwt", "logl_birth", "nlive_i")
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


def _information(logwt, logl, logz: float):
    """Return the information ``H`` as a JAX scalar (NaN when undefined)."""
    weights = jnp.exp(jnp.asarray(logwt) - logz)
    logl = jnp.asarray(logl)
    finite = jnp.isfinite(logl)
    if (
        not math.isfinite(logz)
        or not bool(jnp.all(jnp.isfinite(weights)))
        or bool(jnp.any((weights > 0.0) & ~finite))
    ):
        return jnp.asarray(math.nan)
    contributing = (weights > 0.0) & finite
    information = jnp.sum(jnp.where(contributing, weights * (logl - logz), 0.0))
    return jnp.maximum(information, 0.0)


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


def _ks_uniform(ranks, nslots: int) -> tuple[float, float]:
    """KS distance of ranks from the uniform law on ``0..nslots-1``, p-value.

    The p-value is the asymptotic Kolmogorov tail (Stephens' small-sample
    correction), conservative for a discrete law.
    """
    n = len(ranks)
    ecdf = np.cumsum(np.bincount(ranks, minlength=nslots)[:nslots]) / n
    ks = float(np.max(np.abs(ecdf - np.arange(1, nslots + 1) / nslots)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * ks
    if lam < 0.2:  # the series converges slowly here; the tail is 1 - 1e-11
        return ks, 1.0
    terms = (2 * (-1) ** (k - 1) * math.exp(-2 * (k * lam) ** 2) for k in range(1, 101))
    return ks, min(max(sum(terms), 0.0), 1.0)


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

    ``logzerr = sqrt(sum_i dH_i / n_i)``, with ``dH_i`` the increment of the
    running information at sample ``i`` and ``n_i = nlive_i[i]``; at a
    constant live count ``n`` it is Skilling's ``sqrt(H / n)``.
    """
    logl = np.asarray(logl, dtype=np.float64)
    logwt = _log_weights(logl, nlive_i)
    logz = float(_np_logsumexp(logwt, axis=0))
    if not math.isfinite(logz):
        return logwt, logz, math.nan
    p = np.exp(logwt - logz)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.cumsum(p)
        a = np.cumsum(np.where(p > 0.0, p * logl, 0.0))
        h = np.where(c > 0.0, a / c - np.log(c) - logz, 0.0)
    var = np.sum(np.diff(h, prepend=0.0) / np.asarray(nlive_i, dtype=np.float64))
    return logwt, logz, math.sqrt(max(float(var), 0.0))


@dataclass(kw_only=True)
class NestedSamplingResult:
    """Container for completed nested-sampling outputs.

    The samples are the ``niter`` dead points in the order they died, then
    the final live points by increasing likelihood. ``num_delete`` points die
    per step, so the live counts at the deaths (``nlive_i``) cycle through
    ``nlive, nlive - 1, ..., nlive - num_delete + 1`` and end with
    ``nlive, ..., 1`` over the final live points.
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
    """Number of live points when each sample died (dynesty's ``samples_n``)."""

    labels: ArrayLike | None = None
    """Cluster slot of each sample when it died (the final live points: at the
    end of the run), from the on-device clustering (:mod:`tinyns.modes`);
    :meth:`modes` reads it. ``None`` for a result built by hand."""

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
    False if it stopped at ``maxiter`` or ``maxcall``."""

    message: str = ""
    """Human-readable status of the run."""

    metadata: dict[str, Any] | None = None
    """Run settings and telemetry: ``status``, ``walks``, ``dlogz``,
    ``final_delta_logz``, ``acceptance`` (of the walk steps), ``scale``,
    ``ncall_valid`` (in-cube likelihood calls), ``x64``, the hop counters,
    ``wall_time_s``, ``compile_s``, ``sampling_s``, ``resumed`` and more (the
    CHANGELOG lists the keys)."""

    def log_weights(self):
        """Return posterior weights normalized in log space."""

        return normalize_log_weights(self.logwt)

    def weights(self):
        """Return normalized linear posterior weights."""

        return jnp.exp(self.log_weights())

    def posterior_ess(self) -> float:
        """Return the effective sample size of the posterior weights."""

        return float(effective_sample_size_from_log_weights(self.logwt))

    def max_weight_fraction(self) -> float:
        """Return the largest normalized posterior weight fraction."""

        weights = self.weights()
        if weights.size == 0:
            return 0.0
        return float(jnp.max(weights))

    def posterior_weight_entropy(self) -> float:
        """Return the Shannon entropy of normalized posterior weights."""

        weights = self.weights()
        if weights.size == 0:
            return 0.0
        positive = weights > 0.0
        return float(-jnp.sum(weights[positive] * jnp.log(weights[positive])))

    def posterior_weight_entropy_fraction(self) -> float:
        """Return posterior weight entropy as a fraction of equal-weight entropy."""

        n = int(jnp.asarray(self.logwt).size)
        if n <= 1:
            return 0.0
        fraction = self.posterior_weight_entropy() / float(jnp.log(n))
        return float(jnp.clip(fraction, 0.0, 1.0))

    def live_weight_fraction(self) -> float:
        """Return the posterior weight fraction in the final live points."""

        return float(jnp.sum(self.weights()[int(self.niter) :]))

    def dead_weight_fraction(self) -> float:
        """Return the posterior weight fraction in the dead points."""

        return float(jnp.clip(1.0 - self.live_weight_fraction(), 0.0, 1.0))

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

        The index is the number of the ``nlive - num_delete`` surviving live
        points of its step with a likelihood at or below the new point's, so
        it is uniform on ``0..nlive-num_delete`` for a correct constrained
        sampler (the new points of one step are ranked against the survivors
        only, not against each other).
        """

        return self._births[1]

    def insertion_test(self, windows: int = 3) -> dict[str, Any]:
        """Return a Kolmogorov-Smirnov test of the insertion indices.

        The indices (:meth:`insertion_indices`) are tested against the uniform
        law on ``0..nlive-num_delete`` over the whole run and in ``windows`` equal
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

        def test(start, stop):
            ks, pvalue = math.nan, math.nan
            if stop > start:
                nslots = int(self.nlive) - int(self.num_delete) + 1
                ks, pvalue = _ks_uniform(ranks[start:stop], nslots)
            return {"start": start, "stop": stop, "n": stop - start, "ks": ks,
                    "pvalue": pvalue}

        edges = np.linspace(0, len(ranks), int(windows) + 1).astype(int).tolist()
        out = test(0, len(ranks))
        out["windows"] = [test(a, b) for a, b in itertools.pairwise(edges)]
        return out

    def modes(self) -> list[dict[str, Any]]:
        """Return the posterior modes, heaviest first, with their urn error bar.

        Post-processing of the run's samples: it reads the cluster labels the
        run recorded (:attr:`labels`: the id of each sample's cluster when it
        died, in one of the cross-fitted clusterings that drive the
        inter-mode hop), runs no likelihood and compiles nothing. A label is
        not a mode: the clustering's split test also cuts a curved mode into
        touching pieces, each clustering covers every mode with labels of its
        own, a cluster that is split later held several modes until then, and
        a lattice of many modes is never split at all. The modes are found in
        five passes:

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
           split off and those of the labels left out.

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
        tracked.

        Per mode: ``mass``, ``urn_sd``, ``isolation_iteration``,
        ``min_live`` (the smallest ``n`` between isolation and the mode's
        median), ``unresolved`` (``min_live`` below ``UNRESOLVED_PER_DIM *
        ndim``: the mass is not reliable, raise ``nlive``) and ``tracked``:
        False for a mode that pass 4 cut out of a larger one. The sampler's
        clustering did not separate such a mode from its neighbours, so the
        hop did not balance it, its mass scatters by about ``urn_sd``, and
        the count of such modes is a lower bound (modes with fewer than
        ``SLAB_NEIGHBOURS`` live points, or that no axis-aligned plane
        separates, stay merged). :meth:`diagnostics` also reports for how
        much of the run the clustering's slots were full
        (``mode_slots_full``); then too there may be more modes than are
        returned. A mode lost before the end leaves no trace in its own run,
        and neither does one that never held more than about ``ndim`` live
        points per fold. A unimodal run (or a result without :attr:`labels`)
        gives one mode of mass 1.
        """

        one = [{"mass": 1.0, "urn_sd": 0.0, "min_live": int(self.nlive),
                "isolation_iteration": 0, "unresolved": False, "tracked": True}]
        if int(self.niter) == 0 or self.labels is None:
            return one
        try:
            return self._modes(one)
        except np.linalg.LinAlgError:  # a degenerate set of samples
            return one

    def _modes(self, one: list) -> list[dict[str, Any]]:
        nlive, niter, ndim = int(self.nlive), int(self.niter), int(self.ndim)
        weights = np.asarray(self.weights(), dtype=float)
        u = np.asarray(self.samples_u, dtype=float)
        ids, label = np.unique(np.asarray(self.labels), return_inverse=True)
        cumulative = np.cumsum(weights)
        born, index = self._births[0], np.arange(len(u))

        def median(mask):
            cw = np.cumsum(weights * mask)
            return min(int(np.searchsorted(cw, 0.5 * cw[-1])), niter - 1)

        def alive(t):
            return (born < t) & (index >= t)

        # 1. Labels without posterior mass, or with at most ndim distinct
        # live points at their posterior median, are set aside (-1).
        small = np.bincount(label, weights) * self.posterior_ess() < cumulative[-1]
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
            return one
        # 5. Relabel every sample by its nearest mode, so that n(t) counts the
        # points of a mode also where the run's clustering had merged it.
        kept, label = np.unique(
            _nearest_mode(u, weights, cut, len(tracked)), return_inverse=True
        )
        masks = [label == c for c in range(len(kept))]
        if len(masks) < 2:
            return one
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
            out.append({
                "mass": float(weights[mine].sum() / cumulative[-1]),
                "urn_sd": float(math.sqrt(variance)),
                "min_live": min_live,
                "isolation_iteration": int(lo),
                "unresolved": bool(min_live < UNRESOLVED_PER_DIM * ndim),
                "tracked": bool(tracked[kept[c]]),
            })
        return sorted(out, key=lambda mode: -mode["mass"])

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
            dist = np.sum((z[other][:, None, :] - z[None, :, :]) ** 2, axis=-1)
            dist[np.arange(other.sum()), np.flatnonzero(other)] = np.inf
            if not (other.any() and np.any(owner[np.argmin(dist, axis=1)] == m)):
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

    def _mode_warnings(self, modes) -> list[str]:
        """The warnings of :meth:`diagnostics` about ``modes``."""
        warnings = [
            f"mode {i} (mass {mode['mass']:.3g}) is unresolved: it held "
            f"{mode['min_live']} live points; raise nlive"
            for i, mode in enumerate(modes)
            if mode["unresolved"]
        ]
        most, full = self._mode_slots()
        loose = sum(not mode["tracked"] for mode in modes)
        if loose:
            held = "" if most is None else f" (it held at most {most} clusters)"
            warnings.append(
                f"{loose} of the {len(modes)} modes were not separated by the "
                f"sampler's clustering{held}: the inter-mode hop did not balance "
                "them, so their masses scatter from run to run by about urn_sd, "
                "and there may be more modes than were found"
            )
        if full and len(modes) > 1:
            warnings.append(
                f"the {C_MAX} cluster slots were full for {100 * full:.0f}% of "
                f"the run: there may be more modes than the {len(modes)} found, "
                "and the mode masses are not reliable"
            )
        return warnings

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
        windows), the posterior ``modes`` (:meth:`modes`) and
        ``mode_slots_full``, the fraction of the run during which every
        cluster slot of the sampler's clustering was in use (``None`` for a
        result without that record). The warnings name each unresolved mode,
        the modes the clustering did not separate, and a run of several
        modes whose slots were full: in the last two cases the number of
        modes is a lower bound.
        """

        metadata = {} if self.metadata is None else self.metadata
        posterior_ess = self.posterior_ess()
        nposterior = int(jnp.asarray(self.logwt).size)
        max_weight_fraction = self.max_weight_fraction()
        entropy_fraction = self.posterior_weight_entropy_fraction()
        live_weight_fraction = self.live_weight_fraction()
        insertion = self.insertion_test()
        modes = self.modes()
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
        if live_weight_fraction > 0.5:
            warnings.append(
                "final live points carry most posterior weight; consider tighter "
                "dlogz or more live points"
            )
        if live_weight_fraction > 0.25 and dlogz is not None and dlogz >= 0.1:
            warnings.append(
                "large final-live weight fraction; evidence may be sensitive to "
                "stopping"
            )
        if (
            self.success
            and None not in (final_delta_logz, dlogz)
            and final_delta_logz > dlogz
        ):
            warnings.append("successful run has final_delta_logz above requested dlogz")
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
            "dead_weight_fraction": self.dead_weight_fraction(),
            "final_delta_logz": final_delta_logz,
            "acceptance": acceptance,
            "insertion_pvalue": insertion["pvalue"],
            "modes": modes,
            "mode_slots_full": self._mode_slots()[1],
            "warnings": warnings,
        }

    def resample_equal(self, key, n: int | None = None):
        """Return ``n`` equally weighted posterior samples (systematic resampling).

        ``key`` is a JAX PRNG key or an int seed; ``n`` defaults to the
        posterior effective sample size.
        """

        if n is None:
            n = max(1, int(self.posterior_ess()))
        if n < 1:
            raise ValueError("n must be at least 1")

        indices = systematic_resample(key, self.logwt, n)
        return jnp.asarray(self.samples)[indices]

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the result.

        With more than one posterior mode it ends with a per-mode table
        (:meth:`modes`) and the warnings about the mode count.
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
        if len(modes) > 1:
            lines.append("mode      mass  logit sd  min live")
            for i, mode in enumerate(modes):
                flag = "  unresolved: raise nlive" if mode["unresolved"] else ""
                flag += "" if mode["tracked"] else "  not tracked"
                lines.append(
                    f"{i:4d}  {mode['mass']:8.4f}  {mode['urn_sd']:8.2f}  "
                    f"{mode['min_live']:8d}{flag}"
                )
            lines += [
                "warning: " + text
                for text in self._mode_warnings(modes)
                if "is unresolved" not in text
            ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def save_npz(self, path) -> None:
        """Save the final result to a compressed NumPy ``.npz`` file (with
        :attr:`labels` when present)."""

        labels = {} if self.labels is None else {"labels": np.asarray(self.labels)}
        np.savez_compressed(
            path,
            **{name: np.asarray(getattr(self, name)) for name in _RESULT_NPZ_ARRAYS},
            **labels,
            **{
                name: np.asarray(kind(getattr(self, name)))
                for name, kind in _RESULT_NPZ_SCALARS.items()
            },
            metadata_json=np.asarray(json.dumps(self.metadata, default=_jsonable)),
            format_version=np.asarray(_RESULT_NPZ_FORMAT_VERSION),
        )

    @classmethod
    def load_npz(cls, path) -> NestedSamplingResult:
        """Load a result previously written by :meth:`save_npz`."""

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
                **{name: jnp.asarray(data[name]) for name in _RESULT_NPZ_ARRAYS},
                labels=np.asarray(data["labels"]) if "labels" in data.files else None,
                **{
                    name: kind(_npz_scalar(data[name]))
                    for name, kind in _RESULT_NPZ_SCALARS.items()
                },
                metadata=json.loads(str(_npz_scalar(data["metadata_json"]))),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python dictionary representation of the result
        (the fields :meth:`save_npz` writes; ``labels`` may be ``None``)."""

        out = {name: getattr(self, name) for name in _RESULT_NPZ_ARRAYS}
        out["labels"] = self.labels
        out.update({name: getattr(self, name) for name in _RESULT_NPZ_SCALARS})
        out["metadata"] = None if self.metadata is None else dict(self.metadata)
        return out

    def to_numpy(self) -> dict[str, object]:
        """Return a plain dictionary with array fields converted to NumPy arrays."""

        out = {name: np.asarray(getattr(self, name)) for name in _RESULT_NPZ_ARRAYS}
        out["labels"] = None if self.labels is None else np.asarray(self.labels)
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
