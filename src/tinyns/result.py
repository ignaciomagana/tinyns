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
from tinyns.modes import split_threshold

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
# Two clusters are one mode unless, at each of their posterior medians, their
# live points leave a gap of this many within-cluster standard deviations
# along the discriminant direction. The split test also cuts one curved mode
# (a banana) into linear pieces, which touch: gaps of -0.5 to 1.1 there,
# against 22 to 32 between the separated modes of the validation targets.
# Before that, a cluster with at most ndim distinct live points at its own
# posterior median is dropped: it spans no volume, and the stuck chains of a
# banana's tip leave such clumps of near-copies 7 to 50 deviations out.
SEPARATION_SIGMA = 3.0


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


def _gap(x, side) -> float:
    """Empty stretch between ``x[~side]`` and ``x[side]`` along the Fisher
    discriminant, in within-set standard deviations (negative if they overlap).

    Repeated rows count once: an unmoved chain returns a copy of its seed, and
    copies would shrink the within-set scatter without adding information."""
    x, first = np.unique(x, axis=0, return_index=True)  # copies of a seed: once
    side = np.asarray(side)[first]
    a, b = x[~side], x[side]
    if not len(a) or not len(b):
        return math.inf
    dev = np.concatenate([a - a.mean(0), b - b.mean(0)])
    within = _ridge(dev.T @ dev / max(len(x) - 2, 1))
    w = np.linalg.solve(within, b.mean(0) - a.mean(0))
    p = x @ w / math.sqrt(max(w @ within @ w, 1e-300))
    return float(p[side].min() - p[~side].max())


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

        Pure post-processing of the cluster labels the run recorded
        (:attr:`labels`: each sample's cluster when it died, from the
        on-device clustering that drives the inter-mode hop), so it runs no
        clustering and compiles nothing, and its modes are the clusters the
        sampler balanced. The clustering's split test also cuts a curved mode
        into touching pieces, its slots are reused over a run, and a slot
        holds different points at different times, so the labels become modes
        in four passes:

        1. a label carrying less than one effective posterior sample (mass
           times the posterior ESS below 1, e.g. a piece split off early in
           the run and merged back) joins the heaviest label;
        2. a label with at most ``ndim`` distinct live points at its own
           posterior median joins the label nearest to it then (Mahalanobis
           distance of its mean to the other labels' live points): so few
           points span no volume, and the gap next to them cannot tell a
           separate mode from a thinly sampled tail, such as the clump of
           near-copies that stuck chains leave in the tip of a banana. A real
           mode that small would be ``unresolved`` anyway;
        3. two labels count as separate modes only if their live points leave
           a gap of ``SEPARATION_SIGMA`` within-cluster standard deviations
           along the discriminant direction at each of the two posterior
           medians (where both have live points); touching labels are merged,
           one pair at a time, until every pair is separated. Pieces of a
           curved mode that look apart at one time touch at the other;
        4. every sample is relabelled by its nearest mode (hard-EM
           classification with each mode's posterior-weighted mean and
           covariance), so that a mode keeps its points where the run's
           clustering had them in another slot (before it split the mode
           off, or after a merge near the end of the run).

        Each mode's live count ``n(t)`` is rebuilt from the birth and death
        iterations. The mode is isolated from the first iteration at which
        its live points and the others pass the split test's threshold
        (:func:`tinyns.modes.split_threshold`; if they never do before the
        earlier of the two posterior medians, there is no drift term and
        ``isolation_iteration`` is that median). From then on random-walk
        chains cannot leave it, so without the hop ``n`` does a random walk
        (the urn) and the mode's mass scatters from seed to seed by
        ``urn_sd`` in ``logit(mass)``: ``1/n + 1/(N - n)`` at isolation,
        plus ``2 (1 - g) / (N^2 g)`` per iteration up to the mode's posterior
        median and ``2 g / (N^2 (1 - g))`` up to the rest's, with ``g = n /
        N``. The hop removes most of that drift (about 7x less scatter in the
        v1 bake-off), so ``urn_sd`` is an upper bound. ``min_live`` is the
        smallest ``n`` between isolation and the mode's median; below
        ``UNRESOLVED_PER_DIM * ndim`` the mode is ``unresolved`` and its mass
        is not reliable: raise ``nlive``. A mode lost before the end leaves no
        trace in its own run. A unimodal run (or a result without
        :attr:`labels`) gives one mode of mass 1.
        """

        one = [{"mass": 1.0, "urn_sd": 0.0, "min_live": int(self.nlive),
                "isolation_iteration": 0, "unresolved": False}]
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
        label = np.unique(np.asarray(self.labels), return_inverse=True)[1]
        k = int(label.max()) + 1 if label.size else 0
        if k < 2:
            return one
        cumulative = np.cumsum(weights)
        born, index = self._births[0], np.arange(len(u))

        def median(mask):
            cw = np.cumsum(weights * mask)
            return min(int(np.searchsorted(cw, 0.5 * cw[-1])), niter - 1)

        def alive(t):
            return (born < t) & (index >= t)

        def compact(label):
            return np.unique(label, return_inverse=True)[1]

        # 1. Labels without posterior mass join the heaviest.
        mass = np.bincount(label, weights, minlength=k)
        light = mass * self.posterior_ess() < cumulative[-1]
        label = compact(np.where(light[label], np.argmax(mass), label))
        k = int(label.max()) + 1
        # 2. Labels with at most ndim distinct live points at their posterior
        # median join the nearest other label (the Mahalanobis distance to
        # its live points at its own median: at the small label's median it
        # may be down to a few points, which span no covariance).
        when = [median(label == c) for c in range(k)]
        tiny = [
            len(np.unique(u[alive(when[c]) & (label == c)], axis=0)) <= ndim
            for c in range(k)
        ]
        if k - sum(tiny) < 2:
            return one
        root = np.arange(k)
        for c in np.flatnonzero(tiny):
            live = alive(when[c])
            mine = u[live & (label == c)]
            centre = mine.mean(0) if len(mine) else u[label == c].mean(0)
            dist = [
                math.inf
                if tiny[b]
                else _mahalanobis2(u[alive(when[b]) & (label == b)], centre)
                for b in range(k)
            ]
            heaviest = max((b for b in range(k) if not tiny[b]), key=lambda b: mass[b])
            root[c] = int(np.argmin(dist)) if math.isfinite(min(dist)) else heaviest
        label = compact(root[label])
        k = int(label.max()) + 1
        # 3. Merge labels that touch (pieces of one mode), one pair at a time,
        # until every pair is separated.
        def separated(a, b, label):
            gaps = []
            for t in sorted({median(label == a), median(label == b)}):
                pair = alive(t) & ((label == a) | (label == b))
                side = label[pair] == b
                if side.any() and not side.all():
                    gaps.append(_gap(u[pair], side))
            return bool(gaps) and min(gaps) > SEPARATION_SIGMA

        merged = True
        while merged and k > 1:
            merged = False
            for a, b in itertools.combinations(range(k), 2):
                if not separated(a, b, label):
                    label = compact(np.where(label == b, a, label))
                    k, merged = k - 1, True
                    break
        if k < 2:
            return one
        # 4. Relabel every sample by its nearest mode, so that n(t) counts the
        # points of a mode also where the run's clustering had merged it.
        label = compact(_nearest_mode(u, weights, label, k))
        k = int(label.max()) + 1
        if k < 2:
            return one
        threshold = split_threshold(ndim)
        out = []
        for mine in (label == c for c in range(k)):
            net = np.bincount(born[mine & (born >= 0)], minlength=niter) - mine[:niter]
            n = np.sum(mine & (born < 0)) + np.concatenate([[0], np.cumsum(net)[:-1]])

            def isolated(t, mine=mine):
                live = alive(t)
                side = mine[live]
                return 0 < side.sum() < side.size and (
                    _fisher(u[live], side) >= threshold
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
            })
        return sorted(out, key=lambda mode: -mode["mass"])

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
        windows) and the posterior ``modes`` (:meth:`modes`).
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
        for i, mode in enumerate(modes):
            if mode["unresolved"]:
                warnings.append(
                    f"mode {i} (mass {mode['mass']:.3g}) is unresolved: it held "
                    f"{mode['min_live']} live points; raise nlive"
                )
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
            "warnings": warnings,
        }

    def resample_equal(self, key, n: int | None = None):
        """Return ``n`` equally weighted posterior samples (systematic resampling).

        ``key`` is a JAX PRNG key (not an int seed); ``n`` defaults to the
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
        (:meth:`modes`).
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
                lines.append(
                    f"{i:4d}  {mode['mass']:8.4f}  {mode['urn_sd']:8.2f}  "
                    f"{mode['min_live']:8d}{flag}"
                )
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
        """Return a plain Python dictionary representation of the result."""

        out = {name: getattr(self, name) for name in _RESULT_NPZ_ARRAYS}
        out.update({name: getattr(self, name) for name in _RESULT_NPZ_SCALARS})
        out["metadata"] = None if self.metadata is None else dict(self.metadata)
        return out

    def to_numpy(self) -> dict[str, object]:
        """Return a plain dictionary with array fields converted to NumPy arrays."""

        out = {name: np.asarray(getattr(self, name)) for name in _RESULT_NPZ_ARRAYS}
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
