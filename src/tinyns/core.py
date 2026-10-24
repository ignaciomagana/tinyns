"""The full-JAX core of :mod:`tinyns`: ``Config``, ``State``, ``init``, ``step``.

One :func:`step` deletes the ``k = cfg.num_delete`` lowest live points and
replaces them with the ends of ``k`` fixed-length Metropolis chains (a random
walk in the covariance of the current point's cluster, with an inter-mode hop
every 10th step; :mod:`tinyns.modes`), seeded from live points above the new
contour ``L*`` (the highest deleted likelihood). The evidence bookkeeping
treats the ``k`` deaths of a step as consecutive deaths at live counts
``m, m-1, ..., m-k+1`` (the birth/death contours of Fowlie, Handley and Su),
so the expected log prior volume after ``it`` steps is ``-it * S`` with
``S = sum_j 1 / (m - j)``, less the extra compression of exact likelihood
ties (plateaus; :func:`_tie_counts`), which :class:`State` carries apart
(``log_x_ties``). Each dead row records its live count (``Dead.nlive``).
:func:`finalise` recomputes the weights, the evidence and its error on the
host in float64 from those counts.

:func:`run` is the driver: a jitted ``lax.while_loop`` runs up to ``n_steps``
steps into a preallocated buffer of dead rows, and the host syncs once per
chunk, prints progress and writes checkpoints (:mod:`tinyns.checkpoint`). A
batch of keys runs as one program that vmaps :func:`step` inside the loop.
``loglike`` and ``prior_transform`` may be pytree callables
(``jax.tree_util.Partial(fn, data)``): their array leaves are jit arguments
and the compiled kernels are cached on their structure
(:mod:`tinyns.callables`), so datasets of one shape share one compile.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.special import logsumexp

from tinyns import checkpoint as _checkpoint
from tinyns import modes as _modes
from tinyns.callables import (
    _device_leaves,
    _kernel_cache,
    _partition_callable,
    _split_callable,
    _split_callables,
)
from tinyns.math import as_key
from tinyns.result import NestedSamplingResult, _evidence

# Run status codes (State.status): ``STATUS[code]`` is the name of ``code``.
RUNNING, CONVERGED, MAXITER, MAXCALL, PLATEAU, INVALID = range(6)
STATUS = ("running", "converged", "maxiter", "maxcall", "plateau", "invalid")
_MESSAGES = (
    "running",
    "converged: the live points hold less than dlogz of the evidence",
    "stopped at maxiter before reaching dlogz",
    "stopped at maxcall before reaching dlogz",
    "likelihood plateau: no live point lies above the deleted ones",
    "loglike returned +inf at a live point (an overflow? write it in log "
    "space or enable x64): logz is +inf and the run stopped there",
)
# The message of an INVALID result whose logz is -inf.
_NO_FINITE = (
    "no live point has a finite likelihood (-inf or NaN wherever sampled): "
    "check loglike, or raise nlive if its support is tiny"
)

# Step-scale adaptation: the walk step is ``exp(log_scale) * L z`` with ``L``
# the Cholesky factor of the live covariance. After every step the log scale
# moves by ``rate * clip(acceptance - 0.25, -0.5, 0.5)``, with
# ``rate = 0.5 * min(1, k / 32)`` (a step of few chains measures the
# acceptance noisily).
_TARGET_ACCEPT = 0.25
_INITIAL_SCALE = 0.5
_MIN_SCALE = 1e-3
_MAX_SCALE = 10.0

# Driver: a chunk holds at most this many e-folds of prior volume (its dead
# buffer is preallocated at that size) and the host aims for chunks of about
# this many seconds, growing them at most 4x per chunk from one step. Chunk
# lengths may follow the wall clock: the random stream depends on the step
# count only. A checkpoint is written at a chunk boundary at most this often.
_CHUNK_EFOLDS = 4.0
_CHUNK_SECONDS = 45.0
_CHECKPOINT_SECONDS = 600.0
# On resume, the number of live points (of highest likelihood) and of dead
# points (spread over the death order) of each lane that are evaluated again
# (:func:`_check_resume`).
_RESUME_CHECKS = 16
_INT32_MAX = 2**31 - 1
# The frames a chain looks up at every step when no clustering has more in use
# (kernel 1 of :func:`_kernel_level`).
_FEW = 2


def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _user_stacklevel() -> int:
    """The ``stacklevel`` at which a warning issued by the caller of this
    function names the first frame outside tinyns and ``dataclasses``: the
    user's line, whether they built a :class:`Config`, re-made one with
    ``dataclasses.replace``, built a ``NestedSampler`` or ran one. Python
    shows a warning once per line it names, so a fixed level inside tinyns
    would show it once per session."""
    own = ("tinyns", "dataclasses")
    frame, level = sys._getframe(1), 1
    while frame and frame.f_globals.get("__name__", "").split(".")[0] in own:
        frame, level = frame.f_back, level + 1
    return level


def default_walks(ndim: int) -> int:
    """Default chain length: ``max(25, 6 ndim, ndim^2 // 6)``.

    That is 6 steps per dimension up to 36 dimensions (25 below 5), and
    ``ndim / 6`` per dimension beyond (384 at 48, 682 at 64). Early
    in a run, while the prior box still cuts the likelihood contours, the
    likelihood rank along a chain decorrelates in 2 to 3 ``ndim`` steps (about
    9 later on), and in high dimensions ``6 ndim`` steps leave a small upward
    logZ bias from that phase. This default keeps the bias below the logZ
    scatter on correlated Gaussians from 2 to 64 dimensions with 250 to 2000
    live points, for posteriors that are roughly Gaussian in the unit cube. A
    nonlinear prior transform that presses most coordinates against faces of
    the cube needs more: ``ndtri_d6`` in ``bench/validate.py`` (a normal
    prior, the data 3 prior sd off centre in all 6 coordinates) needs 4 times
    as many, and a stronger conflict can need more still, so check that
    doubling ``walks`` moves logZ by less than ``logzerr``. On a 10-D
    Rosenbrock it leaves logZ 0.05 to 0.09 low at ``nlive=1000`` and
    unbiased from 2000.
    """
    return max(25, 6 * ndim, ndim * ndim // 6)


@dataclasses.dataclass(frozen=True)
class Config:
    """Static sampler configuration (hashable: part of the compile cache key).

    ``nlive`` (``m``) live points; ``num_delete`` (``k``) points deleted and
    replaced per step, default ``max(1, nlive // 10)``, at most ``nlive // 2``;
    ``walks`` steps per replacement chain, default :func:`default_walks`
    (``max(25, 6 ndim, ndim^2 // 6)``).

    The live slots form ``K = _folds`` fixed folds (slot index mod ``K``);
    every :attr:`recluster_every` steps the points outside each fold are
    clustered, and a replacement chain seeded in a fold is driven by the
    points outside it alone (cross-fitting, :mod:`tinyns.modes`). Every
    ``HOP_EVERY``-th chain step is the inter-mode hop; the other steps are a
    random walk in the covariance of the cluster of the current point.
    Private switches, for comparisons such as ``bench/bakeoff/``:
    ``_hop=False`` replaces the hop by walk steps (the clustering still
    runs), ``_local=False`` makes the walk use the live covariance (of the
    points outside the fold) everywhere, and ``_folds`` sets ``K``.
    """

    ndim: int
    nlive: int = 1000
    num_delete: int | None = None
    walks: int | None = None
    _hop: bool = True
    _local: bool = True
    _folds: int = 3

    def __post_init__(self):
        ndim = _check_int("ndim", self.ndim, 1)
        nlive = _check_int("nlive", self.nlive, 2)
        k = max(1, nlive // 10) if self.num_delete is None else self.num_delete
        k = _check_int("num_delete", k, 1)
        if k > nlive // 2:
            raise ValueError(f"num_delete must be at most nlive // 2 = {nlive // 2}")
        walks = default_walks(ndim) if self.walks is None else self.walks
        walks = _check_int("walks", walks, 1)
        for name in ("_hop", "_local"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        folds = _check_int("_folds", self._folds, 2)
        # Each fold's walk covariance is fitted to the other folds' points,
        # about (K - 1) / K of nlive, which span fewer than ndim directions
        # below this: on a 32-D Gaussian nlive 40 left logZ 9 nats high.
        least = folds / (folds - 1) * (ndim + 1)
        if nlive < least:
            warnings.warn(
                f"nlive={nlive} is too few for ndim={ndim}: below "
                f"{math.ceil(least)} live points the walk covariances, fitted to "
                f"{folds - 1} of the {folds} folds, cannot span every "
                "direction, and logZ can be biased far beyond logzerr; use at "
                "least 5 * ndim",
                UserWarning,
                stacklevel=_user_stacklevel(),
            )
        for name, value in (
            ("ndim", ndim),
            ("nlive", nlive),
            ("num_delete", k),
            ("walks", walks),
            ("_folds", folds),
        ):
            object.__setattr__(self, name, value)

    @property
    def recluster_every(self) -> int:
        """Steps between reclusterings: ``max(1, round(m / (4 k)))``, a quarter
        of an e-fold."""
        return max(1, int(self.nlive / (4 * self.num_delete) + 0.5))

    @property
    def log_shrink(self) -> float:
        """``S = sum_{j<k} 1 / (m - j)``: the expected e-folds of one step."""
        return math.fsum(1.0 / (self.nlive - j) for j in range(self.num_delete))

    def _log_widths(self) -> np.ndarray:
        """Log prior-volume widths of the ``k`` deaths of step 0 (float64).

        Death ``j`` takes ``X_{j-1} - X_j`` with ``log X_j = -sum_{i<=j}
        1 / (m - i)``; step ``it`` multiplies every width by ``exp(-it S)``.
        """
        n = self.nlive - np.arange(self.num_delete, dtype=np.float64)
        log_x_prev = -np.concatenate([[0.0], np.cumsum(1.0 / n)[:-1]])
        return log_x_prev + np.log(-np.expm1(-1.0 / n))


class State(NamedTuple):
    """The live set and the run scalars, threaded through :func:`step`.

    ``ncall`` counts the likelihood evaluations the algorithm makes and
    ``ncall_valid`` the proposals that pass the likelihood-free tests (inside
    the open unit cube, and the Hastings test of a local-walk or hop step).
    With ``num_delete > 1`` the chains are vmapped and every proposal is
    evaluated (an out-of-cube one at the chain's point) and counted. With
    ``num_delete=1`` the other proposals skip the likelihood (``lax.cond``)
    and are not counted, so ``ncall == ncall_valid``; but in a batched
    :func:`run`, which vmaps the step over the runs, the ``cond`` becomes a
    select that evaluates them too (an out-of-cube one at the chain's point),
    uncounted. ``ncall`` thus counts what the run alone evaluates, in a batch
    as well, and ``maxcall`` applies to it. Under ``jax.vmap(step)`` the
    ``lax.switch`` over the three chain kernels also gets a batched index and
    runs all three, which triples the evaluations (and the recluster ``cond``
    computes a recluster at every step); batch runs with :func:`run`, which
    passes one unbatched kernel level and recluster predicate. ``ncall`` and
    ``ncall_valid`` are int32, and :func:`run` drains them into host integers
    after every chunk. A hand-written loop that may exceed ``2**31`` calls
    must count on the host too (``ncall += int(new.ncall - old.ncall)`` stays
    right when the counter wraps) and pass the total to :func:`finalise`.
    ``logz`` is the running evidence of the dead points, used only to stop;
    :func:`finalise` recomputes it.

    The log prior volume after the step is ``-it * S + log_x_ties``
    (:func:`_log_x`): ``-it * S`` is recomputed from ``it`` (no running sum
    to lose precision in float32) and ``log_x_ties`` is the extra
    compression of the deaths tied with an earlier step's contour, exactly 0
    on a run without ties. ``lstar_prev`` is the contour of the last step
    (-inf before the first, with ``it = 0``) and ``tie_run`` the number of
    consecutive steps, ending with the last, whose contour equals it
    (:func:`_tie_counts`).
    """

    key: Any  # PRNG key
    u: Any  # (m, d) live points in the open unit cube
    logl: Any  # (m,)
    logl_birth: Any  # (m,) contour each live point was born above; -inf at init
    it: Any  # int32 steps done
    logz: Any  # running log evidence of the dead points
    log_x_ties: Any  # extra log prior volume removed by tied deaths (<= 0)
    lstar_prev: Any  # contour of the last step; -inf before the first
    tie_run: Any  # int32 consecutive steps, ending with the last, at lstar_prev
    log_scale: Any  # log of the walk-step multiplier
    ncall: Any  # int32 likelihood evaluations
    ncall_valid: Any  # int32 proposals passing the likelihood-free tests
    status: Any  # int32: RUNNING, CONVERGED, MAXITER, MAXCALL, PLATEAU, INVALID
    # Mode tracking (tinyns.modes). The live slots form K folds (slot index
    # mod K); clustering j is of the points outside fold j. label[j, i] is
    # point i's cluster slot in clustering j (-1 when i is in fold j), and
    # the mode_* fields are the frames of the C_MAX slots of each clustering
    # (tinyns.modes.Frames), fitted at the last recluster.
    label: Any  # (K, m) int32
    mode_mu: Any  # (K, C, d)
    mode_chol: Any  # (K, C, d, d)
    mode_ichol: Any  # (K, C, d, d)
    mode_logdet: Any  # (K, C)
    mode_count: Any  # (K, C)
    mode_active: Any  # (K, C) bool
    mode_eligible: Any  # (K, C) bool
    # The cluster id of each slot, the next free id of each clustering, and
    # the parent of every id (-1: none; the first TREE_SIZE ids). The two
    # parts of a split take new ids, with the id of the cluster they split as
    # parent, so an id names one cluster between the split that made it and
    # the split that ends it.
    mode_uid: Any  # (K, C) int32
    mode_next: Any  # (K,) int32
    mode_tree: Any  # (K, TREE_SIZE) int32


class Dead(NamedTuple):
    """The ``k`` rows of one step: deleted points by increasing likelihood.

    ``nlive[j]`` is the live count at death ``j`` (:func:`_tie_counts`):
    ``m - j`` but on a likelihood plateau.

    ``insertion[j]``, ``nslots[j]``, ``moves[j]``, ``hops[j]`` and
    ``hop_tries[j]`` belong to the ``j``-th new point (not to dead point
    ``j``): its rank among the surviving live points strictly above ``L*``
    (survivors with exactly its likelihood in a random order with it;
    uniform on ``0..nslots[j]-1`` for a correct constrained sampler;
    ``nslots = m - k + 1`` without ties), the accepted moves of its chain (0:
    the chain returned its seed), and the accepted and attempted hops among
    them.
    ``label`` is each dead point's cluster label when it died (the id of its
    cluster in clustering ``j``, the next fold's; :func:`_reported_label`), and
    ``nclusters`` / ``neligible`` the step's number of clusters (the largest
    over the clusterings) and of frames taking part in the hop (the
    smallest), the same in every row.
    """

    u: Any  # (k, d)
    logl: Any  # (k,)
    logl_birth: Any  # (k,)
    insertion: Any  # (k,) int32
    moves: Any  # (k,) int32
    label: Any  # (k,) int32
    hops: Any  # (k,) int32
    hop_tries: Any  # (k,) int32
    nclusters: Any  # (k,) int32
    neligible: Any  # (k,) int32
    nlive: Any  # (k,) int32
    nslots: Any  # (k,) int32


def _loglike_u(loglike, prior_transform, u, dtype):
    """``loglike(prior_transform(u))`` as a ``dtype`` scalar; NaN becomes -inf
    (+inf is kept: :func:`_step` stops the run on it)."""
    logl = jnp.asarray(loglike(prior_transform(u)))
    if logl.shape != ():
        raise ValueError(f"loglike must return a scalar, got shape {logl.shape}")
    logl = logl.astype(dtype)
    return jnp.where(jnp.isnan(logl), -jnp.inf, logl)


def _live_chol(u, mask):
    """Cholesky factor of the covariance of the live points ``u[mask]``.

    It is computed in standardized coordinates (the correlation matrix, plus a
    relative jitter of ``10 eps``) and rescaled by the per-axis standard
    deviations, so that a narrow axis is not swamped by the jitter of a wide
    one in float32. If the factorization still fails, the diagonal is used.
    """
    m, d = u.shape
    w = mask.astype(u.dtype)
    n = jnp.maximum(jnp.sum(w), 1.0)
    x = (u - _modes.matmul(w, u)[None] / n) * w[:, None]
    dof = jnp.maximum(n - 1.0, 1.0)
    std = jnp.sqrt(jnp.sum(x * x, axis=0) / dof)
    z = x / jnp.where(std > 0, std, 1.0)[None]
    corr = _modes.matmul(z.T, z) / dof
    corr = corr + 10.0 * jnp.finfo(u.dtype).eps * jnp.eye(d, dtype=u.dtype)
    chol = jnp.linalg.cholesky(corr)
    chol = jnp.where(jnp.all(jnp.isfinite(chol)), chol, jnp.eye(d, dtype=u.dtype))
    return std[:, None] * chol


def _walk_mask(above, other, ndim):
    """The live points whose covariance shapes the walk of the chains seeded
    in one fold: the points ``other`` (outside the fold) above ``L*``.

    None of them is a chain's seed or (by the seeding rule of :func:`_step`)
    an ancestor of one, so the proposal depends only on the other folds, and
    (for independent live points) given them a seed is a uniform draw
    inside the contour. A Metropolis walk with a proposal fixed in advance
    leaves that uniform distribution invariant, so the chain's end point is
    exactly uniform inside the contour for any number of steps. A seed that
    also shaped its own proposal pulled its chain inward by an amount of
    order ``ndim / nlive``, which overestimated logZ unless the walks were
    long.

    When fewer than ``ndim + 1`` of them lie above ``L*`` (at the start of a
    run whose likelihood is ``-inf`` or tied on most of the prior, or on a
    nearly exhausted live set), every point outside the fold is used, those
    at or below ``L*`` too: the covariance stays cross-fitted and of full
    rank, and spans the region the few points above ``L*`` were drawn from.
    (Falling back to the points above ``L*`` of every fold, the seeds among
    them, kept the chains near the few finite points of a run's start, and
    logZ came out several ``logzerr`` high in 10 and 20 dimensions.) Only
    when the live set is too small for that (``nlive`` below about ``1.5
    (ndim + 1)``) are all live points used.

    The rule counts points, not directions: on an extreme support (a slab 1%
    of the prior wide in 10-D) the ``ndim + 1`` or more points above ``L*``
    of the next step can be copies of a few (a chain that did not move
    returns its seed), and the covariance is rank-deficient for that step.
    Testing its conditioning instead of the count left logZ unchanged.
    """
    mask = jnp.where(jnp.sum(above & other) >= ndim + 1, above & other, other)
    return jnp.where(jnp.sum(mask) >= ndim + 1, mask, jnp.ones_like(mask))


def _hop_steps(walks: int, n_hops: int) -> np.ndarray:
    """The steps of a chain that are hops: every ``HOP_EVERY``-th (static)."""
    at_hop = np.zeros(walks, bool)
    at_hop[_modes.HOP_EVERY - 1 : n_hops * _modes.HOP_EVERY : _modes.HOP_EVERY] = True
    return at_hop


def _step_keys(key, walks: int, hop: bool):
    """The keys of a chain: ``(keys, hop_keys)``.

    ``keys[i]`` are the two keys of step ``i`` (its normal vector and its
    Hastings test). A hop step first splits its key in two, one for the walk
    step it becomes when the hop is off and one, in ``hop_keys``, for the
    hop (``None`` for a chain without hop steps).
    """
    n_hops = walks // _modes.HOP_EVERY if hop else 0
    keys = random.split(key, walks)
    hop_keys = None
    if n_hops > 0:
        at_hop = _hop_steps(walks, n_hops)
        pair = jax.vmap(random.split)(keys[at_hop])
        keys = keys.at[np.flatnonzero(at_hop)].set(pair[:, 0])
        hop_keys = pair[:, 1]
    return jax.vmap(random.split)(keys), hop_keys


def _hop_draws(hop_keys, frames: _modes.Frames, fold=None):
    """The proposals of all the hop steps of a chain and the uniforms of
    their tests, ``(points, uniforms)``.

    A hop proposal does not depend on the chain's point (an independence
    sampler), so the chain draws them before it runs
    (:func:`tinyns.modes.hop_proposal`) and a hop step looks one up.
    ``frames`` are those of the chain's kernel, or with ``fold`` those of
    every fold.
    """
    at = () if fold is None else (fold,)

    def draw(key):
        return _modes.hop_proposal(
            key, frames.eligible[at], frames.logdet[at],
            lambda c: (frames.mu[(*at, c)], frames.chol[(*at, c)]),
        )

    return jax.vmap(draw)(hop_keys)


def _chain(
    key, u, logl, lstar, chol, scale, loglike, prior_transform, walks, skip,
    frames, hop=True, local=False, fold=None, plain=False, near=None, draws=None,
):
    """A fixed-length constrained Metropolis chain from ``(u, logl)``.

    Proposals outside the open unit cube or at or below ``lstar`` are
    rejected; an unmoved chain returns its seed. With ``skip`` (one unbatched
    chain) a proposal that fails the likelihood-free tests skips the
    likelihood (``lax.cond``; a vmap over runs turns it into a select, which
    evaluates the proposal but does not count it); otherwise it is evaluated
    and masked. Wherever one outside the cube is evaluated, it is evaluated
    at the chain's point, so the callables only ever see points of the open
    cube.

    The steps are the random walk ``u + s L z``, except that with ``hop``
    every ``HOP_EVERY``-th step is the inter-mode hop from the chain's
    ``frames`` (:mod:`tinyns.modes`): a schedule fixed in advance, so the
    chain composes invariant kernels in a state-independent order. A hop step
    random-walks when the frames hold fewer than two eligible clusters.

    ``L`` is ``chol`` (the live covariance), or with ``local`` the Cholesky
    factor of the cluster of the current point when the frames hold at least
    two walk clusters (:func:`tinyns.modes.walk_frames`): the label ``c(x)``
    is the nearest walk frame and the proposal ``N(y | x, s^2 S_c(x))`` is not
    symmetric, so ``y`` is accepted with the Hastings ratio
    ``N(x | y, s^2 S_c(y)) / N(y | x, s^2 S_c(x))`` (exact for fixed frames).

    ``frames`` and ``chol`` are those of the chain's kernel, or with ``fold``
    (the chain's fold, under ``vmap``) those of every fold, with a leading
    fold axis. The chain carries the whitened offsets ``L_c^-1 (x - mu_c)``
    of its point to the frames of its fold (and their squared lengths), so a
    step whitens one point, the proposal: against the frames of every fold
    in one product (the frames are shared by the chains; gathering each
    chain's own factors would move ``C_MAX d^2`` numbers per chain and step),
    keeping its fold's. The label, the Hastings ratio and the hop's count of
    ellipsoids are read from the two sets of offsets.

    ``near``: the frames in use alone (:func:`_few`), which the chain then
    looks up instead of ``frames``; the labels are positions in ``near``.
    ``plain``: the frames hold fewer than two walk clusters and fewer than
    two eligible ones, in every fold. The chain is then the symmetric walk
    in ``chol`` at every step, and nothing of the frames is computed. The
    three kernels make the same draws from the same keys and return the same
    chain (to roundoff) when the frames allow the cheaper one.

    ``draws``: the chain's :func:`_step_keys` and :func:`_hop_draws`; made
    from ``key`` when not given (``key`` is not read otherwise).

    Returns ``(u, logl, moves, ncall, ncall_valid, hops, hop_tries)``.
    """
    dtype = logl.dtype
    d = u.shape[0]
    n_hops = walks // _modes.HOP_EVERY if hop else 0
    tracked = (local or n_hops > 0) and not plain
    local = local and tracked

    def evaluate(v):
        return _loglike_u(loglike, prior_transform, v, dtype)

    near = frames if near is None else near
    if fold is None:
        mine = near

        def live_step(z):
            return _modes.matmul(chol, z)

        def frame_chol(c):
            return near.chol[c]

        def offsets(v):
            return jnp.einsum(
                "cij,cj->ci", near.ichol, v[None] - near.mu, precision=_modes.PRECISION
            )

    else:
        if tracked:
            mine = _modes.Frames(*(
                None if name in ("chol", "ichol") else getattr(near, name)[fold]
                for name in _modes.Frames._fields
            ))

        def live_step(z):
            return jnp.einsum("fij,j->fi", chol, z, precision=_modes.PRECISION)[fold]

        def frame_chol(c):
            return near.chol[fold, c]

        def offsets(v):
            y = jnp.einsum(
                "fcij,fcj->fci", near.ichol, v[None, None] - near.mu,
                precision=_modes.PRECISION,
            )
            return y[fold]

    if draws is None:
        keys, hop_keys = _step_keys(key, walks, hop)
        hops = (None, None)
        if tracked and n_hops > 0:
            hops = _hop_draws(hop_keys, frames, fold)
        draws = (keys, *hops)
    keys, hop_points, hop_uniforms = draws
    at_hop = _hop_steps(walks, n_hops)
    hop_index = np.cumsum(at_hop) - at_hop  # the number of the hop at a hop step

    if tracked:
        walkable = _modes.walk_frames(mine)
        use_local = (jnp.sum(walkable) >= 2) if local else jnp.asarray(False)

        def locate(v):
            """The offsets of ``v`` to the frames, their squared lengths and
            the nearest walk frame."""
            y = offsets(v)
            r2 = jnp.sum(y * y, axis=-1)
            c = jnp.argmin(jnp.where(walkable, r2, jnp.inf)).astype(jnp.int32)
            return c, y, r2

        place = locate(u)
    else:
        place = (jnp.zeros((), jnp.int32),) + (jnp.zeros((), dtype),) * 2

    if tracked and n_hops > 0:
        enabled = _modes.hop_enabled(mine)

    def step(carry, xs):
        (k_z, k_u), is_hop, index = xs
        u, logl, c, y, r2, moves, nev, nvalid, hops, tries = carry
        z = random.normal(k_z, u.shape, u.dtype)
        move = live_step(z)
        if local:
            move = jnp.where(use_local, _modes.matmul(frame_chol(c), z), move)
        prop = u + scale * move
        hop = jnp.asarray(False)
        if tracked and n_hops > 0:
            hop = is_hop & enabled
            prop = jnp.where(hop, hop_points[index], prop)
        c_new, y_new, r2_new = c, y, r2
        if tracked:  # whiten the proposal, once
            c_new, y_new, r2_new = locate(prop)
        ok = jnp.asarray(True)
        if local:
            # U < N(x | y, s^2 S_c(y)) / N(y | x, s^2 S_c(x)), with
            # L_c(y)^-1 (x - y) / s from the two points' offsets.
            back = (y[c_new] - y_new[c_new]) / scale
            log_ratio = (
                0.5 * (jnp.sum(z * z) - jnp.sum(back * back))
                + mine.logdet[c] - mine.logdet[c_new]
            )
            ok = ~use_local | (jnp.log(random.uniform(k_u, (), dtype)) < log_ratio)
        if tracked and n_hops > 0:
            h_ok = _modes.hop_accept(hop_uniforms[index], mine.eligible, r2_new, r2, d)
            ok = jnp.where(hop, h_ok, ok)

        # The open cube: the faces have zero measure, and a transform such
        # as ndtri(u) or -log(u) is infinite on them. A proposal outside is
        # evaluated (vmapped) at the chain's point instead, and masked.
        inside = jnp.all((prop > 0.0) & (prop < 1.0))
        call = inside & ok
        at = jnp.where(inside, prop, u)
        if skip:
            new = lax.cond(
                call, evaluate, lambda v: jnp.asarray(-jnp.inf, dtype), at
            )
            nev = nev + call.astype(jnp.int32)
        else:
            new = jnp.where(call, evaluate(at), -jnp.inf)
            nev = nev + 1
        accept = call & (new > lstar)
        u = jnp.where(accept, prop, u)
        logl = jnp.where(accept, new, logl)
        c = jnp.where(accept, c_new, c)
        y = jnp.where(accept, y_new, y)
        r2 = jnp.where(accept, r2_new, r2)
        moves = moves + accept.astype(jnp.int32)
        hops = hops + (accept & hop).astype(jnp.int32)
        tries = tries + hop.astype(jnp.int32)
        nvalid = nvalid + call.astype(jnp.int32)
        return (u, logl, c, y, r2, moves, nev, nvalid, hops, tries), None

    zero = jnp.zeros((), jnp.int32)
    carry = (u, logl, *place, zero, zero, zero, zero, zero)
    xs = (keys, jnp.asarray(at_hop), jnp.asarray(hop_index, jnp.int32))
    (u, logl, _, _, _, *counts), _ = lax.scan(step, carry, xs)
    return (u, logl, *counts)


def _init(key, loglike, prior_transform, cfg: Config) -> State:
    m, d, k = cfg.nlive, cfg.ndim, cfg.num_delete
    dtype = jnp.result_type(float)
    key, sub = random.split(key)
    u = random.uniform(sub, (m, d), dtype)
    # A draw of exactly 0 (one in 2^23 per coordinate in float32) moves into
    # the open cube, to half the spacing of the draws.
    u = jnp.where(u > 0, u, jnp.finfo(dtype).eps / 2)
    logl = lax.map(
        lambda v: _loglike_u(loglike, prior_transform, v, dtype), u, batch_size=k
    )
    i32 = jnp.int32
    folds = cfg._folds
    return State(
        key=key,
        u=u,
        logl=logl,
        logl_birth=jnp.full((m,), -jnp.inf, dtype),
        it=jnp.zeros((), i32),
        logz=jnp.asarray(-jnp.inf, dtype),
        log_x_ties=jnp.zeros((), dtype),
        lstar_prev=jnp.asarray(-jnp.inf, dtype),
        tie_run=jnp.zeros((), i32),
        log_scale=jnp.asarray(math.log(_INITIAL_SCALE), dtype),
        ncall=jnp.asarray(m, i32),
        ncall_valid=jnp.asarray(m, i32),
        status=jnp.asarray(RUNNING, i32),
        label=jnp.where(
            _fold(m, folds)[None, :] == jnp.arange(folds, dtype=i32)[:, None], -1, 0
        ).astype(i32),
        **_empty_frames(d, folds, dtype),
    )


def _empty_frames(ndim: int, folds: int, dtype) -> dict:
    """The ``mode_*`` fields of a :class:`State` before the first recluster."""
    empty = _modes.empty_frames(ndim, dtype)
    out = {
        "mode_" + name: jnp.stack([x] * folds)
        for name, x in empty._asdict().items()
    }
    out["mode_uid"] = jnp.zeros((folds, _modes.C_MAX), jnp.int32)
    out["mode_next"] = jnp.ones((folds,), jnp.int32)
    out["mode_tree"] = jnp.full((folds, _modes.TREE_SIZE), -1, jnp.int32)
    return out


def _fold(m: int, folds: int):
    """The fold of each live slot: its index mod ``folds``."""
    return jnp.arange(m, dtype=jnp.int32) % folds


def _complements(m: int, folds: int) -> np.ndarray:
    """``(folds, n)`` slot indices outside each fold (static), padded with -1."""
    idx = [np.flatnonzero(np.arange(m) % folds != j) for j in range(folds)]
    n = max(len(i) for i in idx)
    return np.stack([np.concatenate([i, np.full(n - len(i), -1)]) for i in idx])


def _frames(state: State) -> _modes.Frames:
    """The cluster frames of every clustering (a leading fold axis; frames
    ``j`` are fitted to the points outside fold ``j``)."""
    return _modes.Frames(*(
        getattr(state, "mode_" + name) for name in _modes.Frames._fields
    ))


def _reported_label(label, uid, slot_fold, folds: int):
    """The label recorded for :meth:`~tinyns.NestedSamplingResult.modes`: the
    id of a point's cluster in the clustering of the next fold, ``j = (fold +
    1) mod folds`` (one that contains the point), as ``id * folds + j``
    (``uid``: the ids of the slots, :attr:`State.mode_uid`). A cluster's id
    ends when it is split, so a label never names a cluster and, later, a
    part of it; ``modes()`` merges the labels that cover one mode."""
    j = (slot_fold + 1) % folds
    xp = np if isinstance(label, np.ndarray) else jnp
    slot = xp.take_along_axis(label, j[None, :], 0)[0]
    return uid[j, slot] * folds + j


def _recluster(state: State, batched: bool = False) -> State:
    """Recluster the points outside each fold (one lane per fold,
    :func:`tinyns.modes.recluster_lanes`): every live point gets a fresh label
    in every clustering but its own fold's, and the frames are refitted (they
    stay fixed until the next recluster, so they are factorized here, once).
    ``batched``: the state has a leading lane axis.
    """
    u, label = state.u, state.label
    if not batched:
        u, label = u[None], label[None]
    lanes, m, d = u.shape
    folds = label.shape[1]
    comp = _complements(m, folds)  # (folds, n)
    valid = jnp.asarray(comp >= 0)
    at = jnp.asarray(np.maximum(comp, 0))
    sub_u = u[:, at]  # (lanes, folds, n, d)
    sub_lab = jnp.take_along_axis(label, at[None], 2)  # (lanes, folds, n)
    sub_lab = jnp.where(valid[None], sub_lab, -1)
    n = comp.shape[1]
    lab, st, fresh, origin = _modes.recluster_lanes(
        sub_u.reshape(lanes * folds, n, d), sub_lab.reshape(lanes * folds, n)
    )
    # The parts of the clusters split in this call take new ids, and the id
    # of the cluster they were split from is recorded as their parent (ids
    # past TREE_SIZE are dropped).
    old, nxt, tree = state.mode_uid, state.mode_next, state.mode_tree
    if not batched:
        old, nxt, tree = old[None], nxt[None], tree[None]
    shape = (lanes, folds, _modes.C_MAX)
    fresh, origin = fresh.reshape(shape), origin.reshape(shape)
    rank = jnp.cumsum(fresh, axis=-1, dtype=jnp.int32) - 1
    uid = jnp.where(fresh, nxt[..., None] + rank, old)
    nxt = nxt + jnp.sum(fresh, axis=-1, dtype=jnp.int32)
    parent = jnp.take_along_axis(old, origin, axis=-1)
    entry = jnp.where(fresh, uid, _modes.TREE_SIZE)  # out of range: dropped
    tree = jax.vmap(jax.vmap(lambda t, i, p: t.at[i].set(p, mode="drop")))(
        tree, entry, parent
    )
    lab = jnp.where(valid[None], lab.reshape(lanes, folds, n), -1)
    # Scatter back (padding goes to a dummy column m, dropped).
    cols = jnp.where(valid, at, m)
    label = jnp.full((lanes, folds, m + 1), -1, jnp.int32)
    label = label.at[:, jnp.arange(folds)[:, None], cols].set(lab)[..., :m]
    frames = jax.vmap(lambda st: _modes.frames(st, m))(st)
    new = {
        "mode_" + name: x.reshape(lanes, folds, *x.shape[1:])
        for name, x in frames._asdict().items()
    }
    new.update(label=label, mode_uid=uid, mode_next=nxt, mode_tree=tree)
    if not batched:
        new = {name: x[0] for name, x in new.items()}
    return state._replace(**new)


def _recluster_due(state: State, cfg: Config):
    return state.it % cfg.recluster_every == 0


def _maybe_recluster(state: State, cfg: Config, batched: bool = False) -> State:
    """Recluster before a step whose index is a multiple of
    ``cfg.recluster_every``.

    The predicate of the ``lax.cond`` is unbatched: for a batch of runs
    (``batched``, a leading lane axis) the clustering runs (vmapped) when any
    running lane is due and is kept only in those lanes, so each lane
    reclusters exactly when it would alone.
    """
    if not batched:
        return lax.cond(_recluster_due(state, cfg), _recluster, lambda s: s, state)
    due = _recluster_due(state, cfg) & (state.status == RUNNING)
    return lax.cond(
        jnp.any(due),
        lambda s: _select(due, _recluster(s, batched=True), s),
        lambda s: s,
        state,
    )


def _seeds(key_seed, key_fill, above, worst, folds: int, dtype):
    """The seed of each new point: a distinct live point above ``L*`` of the
    fold of the dead slot it fills (Gumbel top-k within the fold), drawn with
    replacement only when that fold has too few (one uniform per new point:
    the rank of its seed among the fold's points above ``L*``), and from the
    whole live set only when it has none. A live point and its descendants
    thus stay in one fold.

    The draws with replacement read the points above ``L*`` in slot order
    from one sort of each fold, so a step costs ``O(m log m)``, not ``O(k
    m)`` (a ``(k, m)`` mask took 28 of a 44 ms step at nlive 1e4 on a CPU,
    and 7.6 GiB per run at nlive 1e5)."""
    m, k = above.shape[0], worst.shape[0]
    fold = _fold(m, folds)
    score = jnp.where(above, random.gumbel(key_seed, (m,), dtype), -jnp.inf)
    dead_fold = fold[worst]
    rank = jnp.cumsum(jax.nn.one_hot(dead_fold, folds, dtype=jnp.int32), 0) - 1
    rank = jnp.take_along_axis(rank, dead_fold[:, None], 1)[:, 0]
    tops = jnp.stack([
        lax.top_k(jnp.where(fold == j, score, -jnp.inf), k)[1] for j in range(folds)
    ])
    # Row j of a (folds, n) view holds slots j, j + folds, ... (padded at the
    # end), so its points above L* come first, in slot order, after a stable
    # sort of ~above; position p in row j is slot j + folds * p.
    n = -(-m // folds)
    by_fold = jnp.concatenate([above, jnp.zeros((n * folds - m,), bool)])
    by_fold = by_fold.reshape(n, folds).T
    first = jnp.argsort(~by_fold, axis=1, stable=True).astype(jnp.int32)
    anywhere = jnp.argsort(~above, stable=True).astype(jnp.int32)
    n_fold = jnp.sum(by_fold, axis=1, dtype=jnp.int32)
    has = n_fold[dead_fold] > 0
    count = jnp.where(has, n_fold[dead_fold], jnp.sum(above, dtype=jnp.int32))
    pick = random.uniform(key_fill, (k,), dtype) * count.astype(dtype)
    pick = jnp.maximum(jnp.minimum(pick.astype(jnp.int32), count - 1), 0)
    fill = jnp.where(
        has, dead_fold + folds * first[dead_fold, jnp.minimum(pick, n - 1)],
        anywhere[pick],
    )
    return jnp.where(rank < n_fold[dead_fold], tops[dead_fold, rank], fill)


def _used(frames: _modes.Frames, cfg: Config):
    """The frames the moves of ``cfg`` can read: the walk clusters and the
    eligible ones."""
    used = jnp.zeros_like(frames.active)
    if cfg._local:
        used |= _modes.walk_frames(frames)
    if cfg._hop and cfg.walks >= _modes.HOP_EVERY:
        used |= frames.eligible
    return used


def _kernel_level(state: State, cfg: Config):
    """Which chain kernel a step needs: 0, 1 or 2 (an unbatched int32, the
    largest over the clusterings and, for a batch of runs, over the lanes).

    0: no clustering holds two walk clusters or two eligible ones, so the
    chains are symmetric walks in the live covariance and read no frame
    (:func:`_chain`, ``plain``). 1: no clustering has more than ``_FEW``
    frames in use, and the chains look up those alone. 2: all ``C_MAX``.
    """
    frames = _frames(state)
    multi = jnp.asarray(False)
    if cfg._local:
        multi |= jnp.any(jnp.sum(_modes.walk_frames(frames), axis=-1) >= 2)
    if cfg._hop and cfg.walks >= _modes.HOP_EVERY:
        multi |= jnp.any(jnp.sum(frames.eligible, axis=-1) >= 2)
    few = jnp.all(jnp.sum(_used(frames, cfg), axis=-1) <= _FEW)
    return jnp.where(multi, jnp.where(few, 1, 2), 0).astype(jnp.int32)


def _few(frames: _modes.Frames, cfg: Config) -> _modes.Frames:
    """The first ``_FEW`` frames in use of every clustering (in slot order)."""
    order = jnp.argsort(~_used(frames, cfg), axis=-1, stable=True)[:, :_FEW]
    return _modes.Frames(*(
        jnp.take_along_axis(a, order.reshape(order.shape + (1,) * (a.ndim - 2)), 1)
        for a in frames
    ))


def _log_x(state: State, cfg: Config):
    """Log prior volume after ``state.it`` steps: ``-it * S + log_x_ties``."""
    dtype = state.logl.dtype
    log_x = -state.it.astype(dtype) * jnp.asarray(cfg.log_shrink, dtype)
    return log_x + state.log_x_ties


def _tie_counts(dead_logl, state: State, cfg: Config):
    """The live count at each death of a step, and which deaths are tied.

    New points land strictly above ``L*``, so when live points tie at one
    likelihood (``L = 0`` or NaN on part of the prior, a finite plateau, a
    floor) the tied points left behind die in later steps, below every point
    born since. Following Fowlie, Handley and Su (2021, plateaus), the ``q``
    deaths of a plateau count at ``m, m-1, ..., m-q+1``: a point born above
    the tie is not live for the tied deaths still to come. A death equal to
    the previous step's contour (``lstar_prev``) is tied, and the
    ``tie_run`` steps at that contour have taken ``k`` deaths each, so death
    ``j`` counts at ``m - j - k * tie_run``. Any other death counts at
    ``m - j``, as does every death of the first step (``lstar_prev`` is -inf
    there, a sentinel and not a contour).
    """
    m, k = cfg.nlive, cfg.num_delete
    tied = (dead_logl == state.lstar_prev) & (state.it > 0)
    base = jnp.arange(m, m - k, -1, dtype=jnp.int32)
    nlive = jnp.where(tied, base - k * state.tie_run, base)
    return jnp.maximum(nlive, 1), tied


def _step(state: State, loglike, prior_transform, cfg: Config, level=None):
    """One step. ``level`` (:func:`_kernel_level`) selects the chains' kernel
    through a ``lax.switch``; a batch of runs passes one unbatched value (the
    largest of its lanes), so that the ``switch`` stays a branch."""
    m, k, walks = cfg.nlive, cfg.num_delete, cfg.walks
    dtype = state.logl.dtype
    # The k lowest points, by increasing likelihood; L* is the highest of them.
    _, worst = lax.top_k(-state.logl, k)
    dead_logl = state.logl[worst]
    lstar = dead_logl[-1]
    above = state.logl > lstar
    n_above = jnp.sum(above, dtype=jnp.int32)
    folds = cfg._folds
    fold = _fold(m, folds)
    key, k_seed, k_fill, k_chain = random.split(state.key, 4)
    seeds = _seeds(k_seed, k_fill, above, worst, folds, dtype)

    # Cross-fitting: the kernel of a chain seeded in fold j (the walk
    # covariance, the cluster frames of the hop and of the local walk) is
    # built from the points outside fold j alone.
    chols = jnp.stack([
        _live_chol(state.u, _walk_mask(above, fold != j, cfg.ndim))
        for j in range(folds)
    ])
    frames = _frames(state)  # frames[j]: fitted outside fold j
    scale = jnp.exp(state.log_scale)

    hopping = cfg._hop and walks >= _modes.HOP_EVERY
    switching = cfg._local or hopping
    if switching and level is None:
        level = _kernel_level(state, cfg)
    # What the chains draw before they run: the keys of their steps and, if
    # any frame is in use, their hop proposals (the same for every kernel).
    chain_keys = k_chain[None] if k == 1 else random.split(k_chain, k)
    chain_fold = fold[seeds]
    step_keys, hop_keys = jax.vmap(lambda key: _step_keys(key, walks, cfg._hop))(
        chain_keys
    )
    hop_draws = (None, None)
    if hopping:
        n_hops = walks // _modes.HOP_EVERY
        hop_draws = lax.cond(
            level > 0,
            lambda: jax.vmap(lambda keys, j: _hop_draws(keys, frames, j))(
                hop_keys, chain_fold
            ),
            lambda: (
                jnp.zeros((k, n_hops, cfg.ndim), dtype),
                jnp.zeros((k, n_hops), dtype),
            ),
        )

    def chains(kernel):
        """The step's chains with kernel 0, 1 or 2 (:func:`_kernel_level`)."""
        near = _few(frames, cfg) if kernel == 1 else frames

        def chain(i, j, draws, lone=False):
            """The chain seeded at live point ``i`` of fold ``j``; ``lone``
            (one unbatched chain): its fold's kernel is sliced out."""
            mine = (chols, frames, near, j)
            if lone:
                mine = (*jax.tree_util.tree_map(lambda a: a[j], mine[:3]), None)
            return _chain(
                None, state.u[i], state.logl[i], lstar, mine[0], scale, loglike,
                prior_transform, walks, lone, mine[1], cfg._hop, cfg._local,
                mine[3], kernel == 0, mine[2], draws,
            )

        def run():
            draws = (step_keys, *hop_draws)
            if k == 1:  # no vmap: the cond skips rejected proposals (in one run)
                first = jax.tree_util.tree_map(lambda a: a[0], draws)
                out = chain(seeds[0], chain_fold[0], first, True)
                return tuple(x[None] for x in out)
            return jax.vmap(chain)(seeds, chain_fold, draws)

        return run

    if switching:
        out = lax.switch(level, [chains(i) for i in range(3)])
    else:
        out = chains(0)()
    new_u, new_logl, moves, nev, nvalid, hops, hop_tries = out

    # A new point's rank among the survivors strictly above L* (points tied
    # at L* are below every new point), out of n_above + 1 slots. Survivors
    # with exactly its likelihood (a plateau above L*, the seed of an unmoved
    # chain) are in a random order with it: the rank takes a uniform place
    # among them, from a key folded off k_seed (the other draws are as they
    # were). The survivors' likelihoods are sorted once (the others padded
    # with +inf, which only the count of equals could see). The two binary
    # searches are unrolled: JAX's default runs each as a loop of log2(nlive)
    # sequential steps, which a GPU pays for at every step (a CPU step takes
    # the same time either way).
    ranked = jnp.sort(jnp.where(above, state.logl, jnp.inf))
    below = jnp.searchsorted(ranked, new_logl, side="left", method="scan_unrolled")
    upto = jnp.searchsorted(ranked, new_logl, side="right", method="scan_unrolled")
    below, upto = below.astype(jnp.int32), upto.astype(jnp.int32)
    same = jnp.minimum(upto, n_above) - below
    place = random.uniform(random.fold_in(k_seed, 1), (k,), dtype)
    place = jnp.minimum((place * (same + 1).astype(dtype)).astype(jnp.int32), same)
    insertion = below + place
    nlive, tied = _tie_counts(dead_logl, state, cfg)
    log_x = _log_x(state, cfg)
    # The weights: the constant widths of cfg._log_widths() without ties,
    # else the widths of the live counts (as finalise computes them).
    log_t = -1.0 / nlive.astype(dtype)
    log_prev = log_x + (jnp.cumsum(log_t) - log_t)
    logwt = jnp.where(
        jnp.any(tied),
        dead_logl + log_prev + jnp.log(-jnp.expm1(log_t)),
        dead_logl + jnp.asarray(cfg._log_widths(), dtype) + log_x,
    )
    plain_t = jnp.asarray(-1.0 / (m - np.arange(k)), dtype)
    log_x_ties = state.log_x_ties + jnp.sum(jnp.where(tied, log_t - plain_t, 0.0))
    # The step scale adapts to the random-walk acceptance alone.
    walk_moves = jnp.sum(moves - hops).astype(dtype)
    walk_tries = jnp.maximum(k * walks - jnp.sum(hop_tries), 1).astype(dtype)
    acceptance = walk_moves / walk_tries
    rate = 0.5 * min(1.0, k / 32)
    log_scale = jnp.clip(
        state.log_scale + rate * jnp.clip(acceptance - _TARGET_ACCEPT, -0.5, 0.5),
        math.log(_MIN_SCALE),
        math.log(_MAX_SCALE),
    )
    ncall = state.ncall + jnp.sum(nev, dtype=jnp.int32)
    ncall_valid = state.ncall_valid + jnp.sum(nvalid, dtype=jnp.int32)
    new = state._replace(
        key=key,
        u=state.u.at[worst].set(new_u),
        logl=state.logl.at[worst].set(new_logl),
        logl_birth=state.logl_birth.at[worst].set(lstar),
        it=state.it + 1,
        logz=jnp.logaddexp(state.logz, logsumexp(logwt)),
        log_x_ties=log_x_ties,
        lstar_prev=lstar,
        tie_run=jnp.where(lstar == state.lstar_prev, state.tie_run + 1, 1).astype(
            jnp.int32
        ),
        log_scale=log_scale.astype(dtype),
        ncall=ncall,
        ncall_valid=ncall_valid,
        status=state.status,
    )
    # A new point takes the label of the nearest frame of every clustering
    # but its own fold's; it is not in the cluster statistics until the next
    # recluster.
    i32 = jnp.int32
    dead_fold = fold[worst]
    nearest = jax.vmap(  # over clusterings j: (folds, k)
        lambda fr: _modes.nearest(fr, new_u)
    )(frames)
    clustering = jnp.arange(folds, dtype=i32)
    nearest = jnp.where(clustering[:, None] == dead_fold[None, :], -1, nearest)
    new = new._replace(label=state.label.at[:, worst].set(nearest))
    n_clusters = jnp.max(jnp.sum(frames.active, axis=1))
    n_eligible = jnp.min(jnp.sum(frames.eligible, axis=1))
    dead = Dead(
        state.u[worst],
        dead_logl,
        state.logl_birth[worst],
        insertion,
        moves,
        label=_reported_label(
            state.label[:, worst], state.mode_uid, dead_fold, folds
        ),
        hops=hops,
        hop_tries=hop_tries,
        nclusters=jnp.full((k,), n_clusters, i32),
        neligible=jnp.full((k,), n_eligible, i32),
        nlive=nlive,
        nslots=jnp.full((k,), n_above + 1, i32),
    )
    # A plateau (nothing above L*) leaves the live set as it was. At L* =
    # -inf no live point has a finite likelihood, and at L* = +inf the live
    # points above the deleted ones are all +inf: neither run is valid.
    stuck = state._replace(
        status=jnp.where(jnp.isfinite(lstar), PLATEAU, INVALID).astype(jnp.int32)
    )
    new = new._replace(**{
        name: jnp.where(n_above == 0, getattr(stuck, name), getattr(new, name))
        for name in State._fields
        if name not in ("key", "ncall", "ncall_valid")
    })
    # A live point with a likelihood of +inf never dies: it is not valid, and
    # the first step after which one is live (from init or new) sets INVALID.
    invalid = (n_above > 0) & jnp.any(jnp.isposinf(new.logl))
    status = jnp.where(invalid, jnp.asarray(INVALID, jnp.int32), new.status)
    return new._replace(status=status), dead


def delta_logz(state: State, cfg: Config):
    """``log(Z + X max L_live) - log Z`` of ``state`` (``inf`` before any death).

    The evidence the live points could still add, in log units: the driver
    stops when it falls below ``dlogz``.
    """
    log_x = _log_x(state, cfg)
    finite = jnp.isfinite(state.logz)
    logz = jnp.where(finite, state.logz, 0.0)
    remain = jnp.logaddexp(logz, log_x + jnp.max(state.logl)) - logz
    return jnp.where(finite, remain, jnp.inf)


def _terminate(state: State, cfg: Config, dlogz, max_steps, call_budget):
    """Set ``state.status`` from dlogz, the step limit and the call budget."""
    status = jnp.select(
        [
            state.status != RUNNING,
            delta_logz(state, cfg) < dlogz,
            state.it >= max_steps,
            state.ncall >= call_budget,
        ],
        [state.status, CONVERGED, MAXITER, MAXCALL],
        RUNNING,
    )
    return state._replace(status=status.astype(jnp.int32))


def _rebuild(loglike_spec, prior_spec, leaves):
    n = loglike_spec.nleaves
    return loglike_spec.rebuild(leaves[:n]), prior_spec.rebuild(leaves[n:])


def _select(mask, new, old):
    """Per lane, ``new`` where ``mask`` else ``old`` (``mask`` has shape ``(B,)``).

    Typed PRNG keys are selected through their key data.
    """

    def pick(a, b):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            impl = random.key_impl(a)
            data = pick(random.key_data(a), random.key_data(b))
            return random.wrap_key_data(data, impl=impl)
        return jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 1)), a, b)

    return jax.tree_util.tree_map(pick, new, old)


@_kernel_cache
def _init_kernel(loglike_spec, prior_spec, cfg: Config, axes=None):
    """Jitted :func:`_init`; ``axes`` (a batched run) vmaps it over the keys,
    with ``axes[i]`` the batch axis (``0`` or ``None``) of callable leaf ``i``."""

    def lane(key, leaves):
        return _init(key, *_rebuild(loglike_spec, prior_spec, leaves), cfg)

    def kernel(key, *leaves):
        if axes is None:
            return lane(key, leaves)
        return jax.vmap(lane, in_axes=(0, axes))(key, leaves)

    return jax.jit(kernel)


@_kernel_cache
def _logl_kernel(loglike_spec, prior_spec, cfg: Config, axes=None):
    """Jitted ``loglike(prior_transform(u))`` of the rows of ``u``, ``(n, d)``,
    in batches of ``num_delete`` like :func:`_init`; ``axes`` (a batched run;
    see :func:`_init_kernel`) vmaps it over a leading lane axis of ``u``."""

    def lane(u, leaves):
        loglike, prior_transform = _rebuild(loglike_spec, prior_spec, leaves)
        return lax.map(
            lambda v: _loglike_u(loglike, prior_transform, v, u.dtype), u,
            batch_size=min(cfg.num_delete, u.shape[0]),
        )

    def kernel(u, *leaves):
        if axes is None:
            return lane(u, leaves)
        return jax.vmap(lane, in_axes=(0, axes))(u, leaves)

    return jax.jit(kernel)


@_kernel_cache
def _step_kernel(loglike_spec, prior_spec, cfg: Config):
    def kernel(state, *leaves):
        state = _maybe_recluster(state, cfg)
        return _step(state, *_rebuild(loglike_spec, prior_spec, leaves), cfg)

    return jax.jit(kernel)


@_kernel_cache
def _chunk_kernel(loglike_spec, prior_spec, cfg: Config, capacity: int, axes=None):
    """Jitted chunk: up to ``min(n_steps, capacity)`` steps into a dead buffer.

    ``n_steps``, ``dlogz``, ``max_steps`` and ``call_budget`` are traced, so
    the driver changes them without recompiling. Returns the state, the
    buffer of ``capacity`` steps of :class:`Dead` rows, the number of valid
    rows and the dlogz remainder.

    With ``axes`` (a batched run; see :func:`_init_kernel`) every state leaf
    and ``call_budget`` carry a leading lane axis. The ``while_loop`` is not
    vmapped: its body vmaps :func:`_step` over the lanes, so the loop
    predicate and any future unbatched ``lax.cond`` stay scalar. The loop runs
    while any lane runs; a lane that has stopped is frozen (``jnp.where``), and
    the valid rows of each lane are a prefix of the buffer of length
    ``count[lane]``. Mode tracking reclusters before a step through an
    unbatched ``lax.cond`` (:func:`_maybe_recluster`).
    """
    def lane_step(state, leaves, level=None):
        return _step(state, *_rebuild(loglike_spec, prior_spec, leaves), cfg, level)

    def kernel(state, n_steps, dlogz, max_steps, call_budget, *leaves):
        i32 = jnp.int32

        def terminate(state, budget):
            return _terminate(state, cfg, dlogz, max_steps, budget)

        if axes is None:
            lanes = ()

            def step_all(state):
                return lane_step(_maybe_recluster(state, cfg), leaves)

        else:
            lanes = state.it.shape
            terminate = jax.vmap(terminate)

            def step_all(state):
                state = _maybe_recluster(state, cfg, batched=True)
                level = _kernel_level(state, cfg)  # unbatched: of all lanes
                new, dead = jax.vmap(lane_step, in_axes=(0, axes, None))(
                    state, leaves, level
                )
                return _select(state.status == RUNNING, new, state), dead

        rows = jax.tree_util.tree_map(
            lambda x: jnp.zeros((capacity, *lanes, *x.shape), x.dtype),
            _dead_template(cfg, state.u.dtype),
        )
        limit = jnp.minimum(n_steps, capacity)

        def cond(carry):
            state, _, _, i = carry
            return (i < limit) & jnp.any(state.status == RUNNING)

        def body(carry):
            state, rows, count, i = carry
            it = state.it
            state, dead = step_all(state)
            rows = jax.tree_util.tree_map(
                lambda r, x: lax.dynamic_update_index_in_dim(r, x, i, 0), rows, dead
            )
            # The rows are valid when the step counted (not on a plateau, and
            # not for a lane that had stopped).
            count = count + (state.it - it).astype(i32)
            return terminate(state, call_budget), rows, count, i + 1

        start = terminate(state, call_budget)
        carry = (start, rows, jnp.zeros(lanes, i32), jnp.int32(0))
        state, rows, count, _ = lax.while_loop(cond, body, carry)
        if axes is None:
            return state, rows, count, delta_logz(state, cfg)
        return state, rows, count, jax.vmap(lambda s: delta_logz(s, cfg))(state)

    return jax.jit(kernel)


def init(key, loglike, prior_transform, cfg: Config) -> State:
    """Draw ``cfg.nlive`` uniform points, evaluate them, return the first state.

    ``key`` is a PRNG key or an int seed. The points are evaluated with
    ``lax.map`` in batches of ``cfg.num_delete``, the batch a step uses.
    ``prior_transform`` must return one array (``TypeError`` otherwise).
    """
    _check_prior_transform(prior_transform, cfg.ndim)
    (ll_dyn, ll_spec), (pt_dyn, pt_spec) = _split_callables(
        loglike, prior_transform, cfg.ndim
    )
    return _init_kernel(ll_spec, pt_spec, cfg, None)(
        as_key(key), *ll_dyn, *pt_dyn
    )


def step(state: State, loglike, prior_transform, cfg: Config) -> tuple[State, Dead]:
    """Delete the ``k`` lowest live points and replace them; return the new
    state and the step's :class:`Dead` rows.

    It does not test dlogz: loop while :func:`delta_logz` is at or above
    your ``dlogz`` and the status is ``"running"``, and pass that ``dlogz``
    to :func:`finalise`. On a plateau (no live point above the deleted ones)
    the live points and the run's scalars (``it``, ``logz``, ...) are
    returned as they were, the rows are not valid, and ``status`` becomes
    ``PLATEAU`` (``STATUS[int(state.status)] == "plateau"``), or ``INVALID``
    (``"invalid"``) where L*, the highest deleted likelihood, is not finite:
    no live point has a finite likelihood, or the others are +inf. Such a
    step still runs its chains: the key advances, and ``ncall`` and
    ``ncall_valid`` count their likelihood calls, so a ``lax.scan`` chunk
    that goes on after a plateau calls the likelihood at each of its
    remaining steps (keep the chunks short for a costly likelihood). A step
    after which a live point has a likelihood of +inf sets ``INVALID`` too
    (its rows are valid). Otherwise ``status`` is left as it is
    (``RUNNING`` after :func:`init`). So the rows of the first
    ``int(state.it)`` steps are the valid ones::

        rows = []
        while delta_logz(state, cfg) >= dlogz:
            state, dead = step(state, loglike, prior_transform, cfg)
            rows.append(dead)
            if STATUS[int(state.status)] != "running":  # plateau or invalid
                break
        dead = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs)[: int(state.it)], *rows
        )

    Before the step, the live points are reclustered if a recluster is due
    (``cfg.recluster_every``).

    Pass the same ``loglike`` and ``prior_transform`` objects at every call.
    The compiled step is cached on a pytree callable's structure (such as
    ``jax.tree_util.Partial(fn, data)``) and on any other callable's identity
    (:mod:`tinyns.callables`), so a new function, closure or bound method
    compiles it again (a new pytree of the same structure does not): seconds
    per call in an eager loop whose compiled step takes milliseconds. A bound
    method is a new object at every access: store ``ll = model.loglike``
    once, or use a module-level function or a pytree callable with array
    leaves. A stored method, like any closure, keeps the values it was traced
    with: after ``model.mu = 5`` store it again (or put ``mu`` in a pytree
    callable's leaves). A step inside ``lax.scan``, or in one jitted
    function called repeatedly, is traced once and is not affected. A Python
    loop under ``jax.jit`` that passes a new object at every step traces
    each step anew, though the jitted function still compiles once.
    """
    (ll_dyn, ll_spec), (pt_dyn, pt_spec) = _split_callables(
        loglike, prior_transform, cfg.ndim
    )
    return _step_kernel(ll_spec, pt_spec, cfg)(state, *ll_dyn, *pt_dyn)


def finalise(
    state: State,
    dead: Dead,
    cfg: Config,
    *,
    prior_transform=None,
    dlogz: float | None = None,
    ncall: int | None = None,
    metadata: dict | None = None,
) -> NestedSamplingResult:
    """Build the :class:`NestedSamplingResult` from the final state and the rows.

    ``dead`` holds the valid rows of the ``state.it`` steps in order, with
    any leading shape (``(steps, k)`` or flat; other row counts raise
    ``ValueError``: drop the rows of a step on a plateau, see :func:`step`).
    Each dead row dies at its recorded live count (``dead.nlive``: ``m - j``
    for death ``j`` of a step, fewer on a likelihood plateau) and the final
    live points are appended by increasing likelihood at live counts ``m,
    m-1, ..., 1``, but those still tied at the last step's contour (a run
    that stopped on a plateau, or during one), which count on from the
    plateau's last dead row as :func:`_tie_counts` does: ``m - j - k *
    tie_run``. The weights, ``logz`` and ``logzerr`` (first order in the
    prior-volume shrinkages) are recomputed in float64
    (:func:`tinyns.result._evidence`). ``result.labels`` holds each
    sample's cluster label when it died (the final live points: their last
    labels), which :meth:`~tinyns.NestedSamplingResult.modes` reads, and
    ``result.insertion_ranks`` / ``insertion_slots`` the in-sampler insertion
    ranks (``dead.insertion``, ``dead.nslots``). ``prior_transform`` maps
    the samples to parameter space (without it ``samples`` are the unit-cube
    points).

    ``result.success`` and ``result.message`` are read from ``state.status``.
    :func:`step` leaves it ``RUNNING`` (or sets ``PLATEAU`` or ``INVALID``),
    so after a hand-written loop pass the ``dlogz`` the loop stopped at: a
    running state with ``delta_logz(state, cfg) < dlogz`` (the driver's rule)
    is reported as converged. Without ``dlogz``, or above it, the result has
    ``success=False`` and the message ``"running"``. A result whose ``logz``
    is not finite has the status ``"invalid"`` and ``success=False``,
    whatever stopped the run, with a message (also issued as a
    ``RuntimeWarning``) for ``logz = +inf`` (a likelihood of +inf) or
    ``-inf`` (no finite likelihood was found).

    ``ncall`` overrides ``state.ncall``, an int32 that wraps after ``2**31``
    calls: the driver drains it, and a long hand-written loop passes its own
    host count (see :class:`State`).
    """
    state, dead = jax.device_get((state, dead))
    if dlogz is not None:
        dlogz = float(dlogz)
        if not dlogz >= 0.0:
            raise ValueError("dlogz must be non-negative")
    m, d, k = cfg.nlive, cfg.ndim, cfg.num_delete
    dead_logl = np.asarray(dead.logl, np.float64).reshape(-1)
    niter = dead_logl.size
    if niter != int(state.it) * k:
        raise ValueError(
            f"{niter} dead rows, but the state has done {int(state.it)} steps of "
            f"{k}: pass the rows of the first state.it steps (a step on a "
            "plateau returns rows that are not valid)"
        )
    order = np.argsort(np.asarray(state.logl), kind="stable")
    u = np.concatenate([np.asarray(dead.u).reshape(-1, d), np.asarray(state.u)[order]])
    logl = np.concatenate([dead_logl, np.asarray(state.logl, np.float64)[order]])
    logl_birth = np.concatenate([
        np.asarray(dead.logl_birth, np.float64).reshape(-1),
        np.asarray(state.logl_birth, np.float64)[order],
    ])
    live_label = _reported_label(
        np.asarray(state.label), np.asarray(state.mode_uid),
        np.arange(m) % cfg._folds, cfg._folds,
    )
    labels = np.concatenate([
        np.asarray(dead.label).reshape(-1), live_label[order]
    ]).astype(np.int32)
    # The live points still tied at the last step's contour (a prefix: every
    # live point is at or above it) die after the tie_run steps of k tied
    # deaths there (_tie_counts; tie_run is 0 before the first step).
    live_n = m - np.arange(m)
    tied = logl[niter:] == float(state.lstar_prev)
    live_n = np.maximum(np.where(tied, live_n - k * int(state.tie_run), live_n), 1)
    nlive_i = np.concatenate([np.asarray(dead.nlive, np.int64).reshape(-1), live_n])
    logwt, logz, logzerr = _evidence(logl, nlive_i)

    log_x = -(niter // k) * cfg.log_shrink + float(state.log_x_ties)
    logz_dead = float(np.logaddexp.reduce(logwt[:niter])) if niter else -math.inf
    remain = (
        float(np.logaddexp(logz_dead, log_x + logl[niter:].max()) - logz_dead)
        if math.isfinite(logz_dead)
        else math.inf
    )
    moves = np.asarray(dead.moves).reshape(-1)
    hops = np.asarray(dead.hops).reshape(-1)
    hop_tries = np.asarray(dead.hop_tries).reshape(-1)
    walk_tries = niter * cfg.walks - int(hop_tries.sum())
    status = int(state.status)
    if status == RUNNING and dlogz is not None and delta_logz(state, cfg) < dlogz:
        status = CONVERGED
    message = _MESSAGES[status]
    if not math.isfinite(logz):
        status = INVALID
        message = _MESSAGES[INVALID] if logz > 0 else _NO_FINITE
        warnings.warn(message, RuntimeWarning, stacklevel=_user_stacklevel())
    samples = u
    if prior_transform is not None:
        samples = _transform(prior_transform, u, state.u.dtype, _transform_block(k))
    info = {
        "status": STATUS[status],
        "num_delete": k,
        "walks": cfg.walks,
        "final_delta_logz": remain,
        "acceptance": (
            float((moves - hops).sum() / walk_tries) if walk_tries else None
        ),
        "unmoved_fraction": float(np.mean(moves == 0)) if niter else None,
        "scale": float(np.exp(state.log_scale)),
        "ncall_valid": int(state.ncall_valid),
        "x64": bool(jax.config.jax_enable_x64),
        **({} if dlogz is None else {"dlogz": dlogz}),
        **_mode_info(dead, k, hops, hop_tries),
        "mode_tree": _mode_tree(state),
        "mode_tree_dropped": _mode_tree_dropped(state),
        "folds": cfg._folds,
    }
    return NestedSamplingResult(
        samples_u=u,
        samples=samples,
        logl=logl,
        logwt=logwt,
        logl_birth=logl_birth,
        nlive_i=nlive_i,
        labels=labels,
        insertion_ranks=np.asarray(dead.insertion, np.int64).reshape(-1),
        insertion_slots=np.asarray(dead.nslots, np.int64).reshape(-1),
        logz=logz,
        logzerr=logzerr,
        ncall=int(state.ncall) if ncall is None else int(ncall),
        niter=niter,
        nlive=m,
        num_delete=k,
        ndim=d,
        success=status in (CONVERGED, PLATEAU),
        message=message,
        metadata={**info, **(metadata or {})},
    )


# The final prior pass of :func:`finalise` maps the samples in blocks of the
# power of two at or above num_delete (a step maps num_delete rows at once),
# and at least this many rows. A block that runs out of device memory is
# halved (:func:`_transform`).
_TRANSFORM_BLOCK = 1024


def _transform_block(num_delete: int) -> int:
    """Rows per call of the final prior pass of a run of ``num_delete``."""
    return max(_TRANSFORM_BLOCK, 1 << (num_delete - 1).bit_length())


def _check_theta(theta) -> None:
    """Raise ``TypeError`` unless ``theta`` (the output of ``prior_transform``
    or its shape) is one array; its shape is free (a scalar at ``ndim=1``,
    derived parameters)."""
    if not isinstance(theta, (jax.Array, jax.ShapeDtypeStruct)):
        raise TypeError(
            f"prior_transform must return one array, got {type(theta).__name__} "
            f"{jax.tree_util.tree_structure(theta)}: stack the parameters into "
            "a vector (jnp.stack) and unpack them in loglike"
        )


def _check_prior_transform(prior_transform, ndim: int) -> None:
    """Check that ``prior_transform`` returns one array, before a run samples
    (``finalise`` could not map the samples otherwise). A prior that cannot
    be evaluated abstractly here is left to the compiled kernels."""
    u = jax.ShapeDtypeStruct((int(ndim),), jnp.result_type(float))
    try:
        theta = jax.eval_shape(prior_transform, u)
    except Exception:  # noqa: BLE001 - the kernels raise their own error
        return
    _check_theta(theta)


@_kernel_cache
def _transform_kernel(prior_spec):
    def kernel(u, *leaves):
        return jax.vmap(prior_spec.rebuild(leaves))(u)

    return jax.jit(kernel)


def _transform(prior_transform, u: np.ndarray, dtype, block: int) -> np.ndarray:
    """``prior_transform`` of every row of ``u``, as a ``(len(u), -1)`` array.

    The rows go through one compiled kernel in blocks of ``block`` rows (the
    last one padded), so the lanes of a batch, whose sample counts differ,
    share one compile. A block that runs out of device memory is halved and
    the pass goes on from it; ``MemoryError`` if one row does not fit.
    """
    n, d = u.shape
    leaves, spec = _split_callable(
        prior_transform, (d, np.dtype(dtype).name),
        lambda: jax.ShapeDtypeStruct((d,), dtype),
    )
    kernel = _transform_kernel(spec)
    u = np.asarray(u, dtype)
    out = []
    start = 0
    while start < n:
        rows = min(block, n - start)
        chunk = u[start : start + rows]
        if rows < block:  # the last block, padded
            chunk = np.concatenate([chunk, np.full((block - rows, d), 0.5, dtype)])
        try:
            theta = kernel(chunk, *leaves)
            _check_theta(theta)
            out.append(np.asarray(theta)[:rows].reshape(rows, -1))
        except Exception as err:
            if "RESOURCE_EXHAUSTED" not in str(err):
                raise
            if block == 1:
                raise MemoryError(
                    "out of device memory in the final prior_transform pass "
                    "of finalise, even one sample at a time"
                ) from err
            block //= 2
            continue
        start += rows
    return np.concatenate(out) if out else np.zeros((0, d), dtype)


def _mode_tree(state: State) -> list:
    """``[label, parent label]`` of every cluster made by a split (labels as
    :func:`_reported_label` records them), for ``result.metadata``: the ids
    below ``TREE_SIZE`` of each clustering (:func:`_mode_tree_dropped` counts
    the others)."""
    tree, nxt = np.asarray(state.mode_tree), np.asarray(state.mode_next)
    folds = tree.shape[0]
    return [
        [int(uid * folds + j), int(tree[j, uid] * folds + j)]
        for j in range(folds)
        for uid in range(1, min(int(nxt[j]), tree.shape[1]))
    ]


def _mode_tree_dropped(state: State) -> int:
    """The number of clusters made by a split whose parent ``mode_tree``
    did not record: ``TREE_SIZE - 1`` ids per clustering are recorded."""
    nxt = np.asarray(state.mode_next)
    return int(np.maximum(nxt - _modes.TREE_SIZE, 0).sum())


def _mode_info(dead: Dead, k: int, hops, hop_tries) -> dict:
    """Mode-tracking telemetry for ``result.metadata``.

    ``hop_acceptance`` is the accepted fraction of the inter-mode moves that
    were tried, and ``mode_history`` lists ``[niter, clusters, eligible]``
    at the start and at every step where either count changed.
    """
    nclusters = np.asarray(dead.nclusters).reshape(-1, k)[:, 0]
    neligible = np.asarray(dead.neligible).reshape(-1, k)[:, 0]
    history = []
    pairs = zip(nclusters.tolist(), neligible.tolist(), strict=True)
    for step, pair in enumerate(pairs):
        if not history or history[-1][1:] != list(pair):
            history.append([step * k, *pair])
    tries = int(hop_tries.sum())
    return {
        "hops": int(hops.sum()),
        "hop_tries": tries,
        "hop_acceptance": float(hops.sum() / tries) if tries else None,
        "mode_history": history,
    }


def _key_batch(key):
    """Return ``(key, batch)``: the key as an array and its lane count.

    ``batch`` is ``None`` for one key (an int seed, a raw ``uint32`` key or a
    typed key) and ``B`` for a batch of ``B`` keys (a leading axis).
    """
    key = as_key(key)
    if isinstance(key, np.ndarray):
        key = jnp.asarray(key)
    if not isinstance(key, jax.Array):
        raise TypeError("key must be a PRNG key, a batch of keys or an int seed")
    typed = jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
    if not typed and key.dtype != jnp.uint32:
        raise TypeError(
            f"key must be a PRNG key, a batch of keys or an int seed, got an "
            f"array of {key.dtype}"
        )
    extra = key.ndim - (0 if typed else 1)
    if extra == 0:
        return key, None
    if extra == 1 and key.shape[0] >= 1:
        return key, int(key.shape[0])
    raise ValueError(f"key must be one key or a 1-D batch of keys, got {key.shape}")


def _lane_axes(fn, nleaves: int, batched_data: bool) -> tuple:
    """vmap axes of the ``nleaves`` dynamic leaves of ``fn`` in a batched run.

    With ``batched_data`` the array leaves of a pytree callable carry a lane
    axis; hoisted closure constants never do.
    """
    batched = batched_data and bool(_partition_callable(fn)[0])
    return (0 if batched else None,) * nleaves


def _lane_callable(fn, lane: int):
    """``fn`` with every array leaf replaced by its slice ``lane``."""
    return jax.tree_util.tree_map(
        lambda x: x[lane] if isinstance(x, (jax.Array, np.ndarray)) else x, fn
    )


def _check_resume(path, state: State, dead: list, cfg: Config, kernel,
                  leaves) -> None:
    """Raise ``ValueError`` unless ``loglike(prior_transform(u))`` gives back
    the likelihoods stored in a checkpoint, in every lane with its own data:
    those of the ``_RESUME_CHECKS`` live points of highest likelihood and of
    as many dead points (``dead``, one :class:`Dead` per lane), spread evenly
    over the death order (the same count in every lane).

    The callables are not stored in a checkpoint: this refuses one written
    for another likelihood, prior or dataset, which would otherwise be
    resumed (a finished one returned as it is). The dead points see a change
    the live ones do not: a wider constraint of a plateau whose live points
    all lie in the old one, or a change where the run has been. A change at
    none of the points checked is not seen. ``kernel`` is the run's
    :func:`_logl_kernel`. It compiles apart from the run's kernels, so the
    values may differ in the last bits: finite ones match to a relative
    ``sqrt(eps)`` of the float dtype, ``|new - old| <= sqrt(eps) (1 + |old|)``,
    and infinite ones (a NaN is stored as ``-inf``) exactly. In float32 a
    change smaller than that (``3.5e-4 |logL|``) is not seen: possibly a new
    noise realisation of the same model, from about ``1e5`` data points (the
    change goes like ``sqrt(|logL|)``; at ``1e5`` it was still 1.2 times the
    tolerance). Setting ``_RESUME_CHECKS = 0`` skips the check, for a
    likelihood that does not give back its own values to that tolerance.
    """
    n = min(_RESUME_CHECKS, cfg.nlive)
    if n == 0:
        return
    batched = state.u.ndim == 3
    u, logl = np.asarray(state.u), np.asarray(state.logl)
    if not batched:  # one lane: give it a lane axis
        u, logl = u[None], logl[None]
    top = np.argsort(-logl, axis=1, kind="stable")[:, :n]
    u = np.take_along_axis(u, top[..., None], axis=1)
    logl = np.take_along_axis(logl, top, axis=1)
    m = min(_RESUME_CHECKS, *(rows.logl.size for rows in dead))
    if m:
        dead_u, dead_logl = [], []
        for rows in dead:
            i = np.linspace(0, rows.logl.size - 1, m).round().astype(int)
            dead_u.append(np.asarray(rows.u).reshape(-1, u.shape[-1])[i])
            dead_logl.append(np.asarray(rows.logl).reshape(-1)[i])
        u = np.concatenate([u, np.stack(dead_u)], axis=1)
        logl = np.concatenate([logl, np.stack(dead_logl)], axis=1)
    old = logl.astype(np.float64)
    new = kernel(jnp.asarray(u if batched else u[0]), *leaves)
    new = np.asarray(new, np.float64).reshape(old.shape)
    rtol = math.sqrt(np.finfo(state.u.dtype).eps)
    finite = np.isfinite(old)
    with np.errstate(invalid="ignore"):  # inf - inf
        diff = np.abs(new - old) / (1.0 + np.abs(old))
    bad = ~np.where(finite, diff <= rtol, new == old)
    if np.any(bad):
        lane, i = np.argwhere(bad)[0]
        where = f" (lane {lane})" if batched else ""
        how = (
            f"a relative difference of {float(diff[lane, i]):.3g} where "
            f"{rtol:.3g} is allowed" if finite[lane, i]
            else "an infinite value must come back exactly"
        )
        raise ValueError(
            f"{path}: this loglike and prior_transform do not give back the "
            f"likelihoods stored in the checkpoint{where}: "
            f"{float(new[lane, i])!r} against {float(old[lane, i])!r} stored "
            f"at a {'live' if i < n else 'dead'} point ({how}), at "
            f"{int(bad[lane].sum())} of the {n} live and {m} dead points "
            "checked. A checkpoint of another likelihood, prior or dataset is "
            "not resumed: use one checkpoint path per problem. If this is the "
            "same problem and its likelihood does not give back its own "
            "values, tinyns.core._RESUME_CHECKS = 0 skips this check"
        )


def _dead_template(cfg: Config, dtype) -> Dead:
    """The rows of one step as zeros."""
    k, d = cfg.num_delete, cfg.ndim
    i32 = np.int32
    return Dead(
        np.zeros((k, d), dtype),
        np.zeros((k,), dtype),
        np.zeros((k,), dtype),
        *(np.zeros((k,), i32),) * 9,
    )


def _empty_dead(cfg: Config, dtype) -> Dead:
    return jax.tree_util.tree_map(
        lambda x: np.zeros((0, *x.shape), x.dtype), _dead_template(cfg, dtype)
    )


def _concat(rows: list) -> Dead:
    return jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *rows)


def run(
    key,
    loglike,
    prior_transform,
    cfg: Config,
    *,
    dlogz: float = 0.1,
    maxiter: int | None = None,
    maxcall: int | None = None,
    progress: bool = False,
    checkpoint=None,
    batched_data: bool = False,
):
    """Run nested sampling to termination; return the result.

    Stops when ``delta_logz < dlogz``, before ``maxiter`` dead points would be
    exceeded, once ``maxcall`` likelihood evaluations are reached (checked
    after each step), on a plateau, or at a live likelihood of +inf or with
    no finite one (status ``"invalid"``, ``success=False``; a lane of a batch
    stops alone). Chunks of steps run on the device; the host syncs once per
    chunk, sizes the next one to about 45 s and, with ``progress``, prints
    one line per chunk.

    ``checkpoint`` is a path: if the file exists the run resumes from it
    (refusing a different config, x64 flag, float dtype, key or batch, and
    live or dead points whose likelihoods ``loglike`` and ``prior_transform``
    do not give back: :func:`_check_resume`), and the run writes it
    atomically at a chunk boundary at most every ``_CHECKPOINT_SECONDS`` and
    at the end. A path that cannot be written raises ``OSError`` before
    anything compiles; a symbolic link (or ``..``) in it is resolved once.
    Each step splits ``state.key``, so the random stream depends on the step
    count only, never on where the chunks end: on a CPU a resumed run is
    bit-identical to an uninterrupted one on the same JAX build (a checkpoint
    of another build or device warns). On a GPU in float32, where a fixed
    seed is not bit-reproducible between processes, a run resumed in a new
    process agrees with an uninterrupted one in distribution. A status of
    ``converged``, ``maxiter`` or ``maxcall`` in the checkpoint is re-tested
    against this call's limits, so a run stopped by ``maxiter`` continues.

    A batch of keys (a leading axis of ``B``) runs ``B`` independent runs in
    one compiled program and returns a list of ``B`` results. With
    ``batched_data`` the array leaves of the pytree callables carry the same
    leading lane axis (one dataset per lane).
    """
    dlogz = float(dlogz)
    if not dlogz >= 0.0:
        raise ValueError("dlogz must be non-negative")
    if maxiter is not None:
        maxiter = _check_int("maxiter", maxiter, 0)
    if maxcall is not None:
        maxcall = _check_int("maxcall", maxcall, 0)
    t0 = time.perf_counter()
    key, batch = _key_batch(key)
    if batched_data and batch is None:
        raise ValueError("batched_data=True needs a batch of keys")
    if checkpoint is not None:  # the file read and written: links resolved
        checkpoint = _checkpoint.check_writable(checkpoint)
    loglike = _device_leaves(loglike)
    prior_transform = _device_leaves(prior_transform)
    (ll_dyn, ll_spec), (pt_dyn, pt_spec) = _split_callables(
        loglike, prior_transform, cfg.ndim
    )
    leaves = (*ll_dyn, *pt_dyn)
    axes = None
    if batch is not None:
        axes = _lane_axes(loglike, len(ll_dyn), batched_data) + _lane_axes(
            prior_transform, len(pt_dyn), batched_data
        )
        if batched_data and 0 not in axes:
            raise ValueError(
                "batched_data=True needs pytree callables with array leaves"
            )
        for leaf, axis in zip(leaves, axes, strict=True):
            if axis == 0 and (leaf.ndim == 0 or leaf.shape[0] != batch):
                hint = (
                    " (a NumPy float scalar or 0-d array is an array leaf too:"
                    " pass a value shared by all runs as a Python float)"
                    if leaf.ndim == 0
                    else ""
                )
                raise ValueError(
                    f"batched_data: every array leaf needs a leading axis of "
                    f"{batch} lanes, got shape {leaf.shape}{hint}"
                )
    _check_prior_transform(
        _lane_callable(prior_transform, 0) if batched_data else prior_transform,
        cfg.ndim,
    )
    lanes = 1 if batch is None else batch
    dtype = jnp.result_type(float)

    resumed = checkpoint is not None and os.path.exists(checkpoint)
    if resumed:
        ckpt = _checkpoint.load(
            checkpoint,
            config=dataclasses.asdict(cfg),
            batch=batch,
            batched_data=batched_data,
            key=key,
        )
        _check_resume(
            checkpoint, ckpt.state, ckpt.dead, cfg,
            _logl_kernel(ll_spec, pt_spec, cfg, axes), leaves,
        )
        state = ckpt.state
        status = np.asarray(state.status)
        stopped = np.isin(status, (CONVERGED, MAXITER, MAXCALL))  # re-tested
        status = np.where(stopped, RUNNING, status)
        state = state._replace(status=jnp.asarray(status, jnp.int32))
        rows_by_lane = [[dead] for dead in ckpt.dead]
        ncall, ncall_valid = list(ckpt.ncall), list(ckpt.ncall_valid)
        wall0, compile_s, nchunks = ckpt.wall_time_s, ckpt.compile_s, ckpt.chunks
        sampling_s = ckpt.sampling_s
        mixed_build = ckpt.mixed_build
    else:
        state = _init_kernel(ll_spec, pt_spec, cfg, axes)(key, *leaves)
        rows_by_lane = [[_empty_dead(cfg, dtype)] for _ in range(lanes)]
        ncall, ncall_valid = [0] * lanes, [0] * lanes
        wall0 = compile_s = sampling_s = 0.0
        nchunks = 0
        mixed_build = False

    capacity = max(1, math.ceil(_CHUNK_EFOLDS / cfg.log_shrink))
    kernel = _chunk_kernel(ll_spec, pt_spec, cfg, capacity, axes)
    k = cfg.num_delete
    i32 = jnp.int32
    max_steps = _INT32_MAX if maxiter is None else min(maxiter // k, _INT32_MAX)
    limits = (jnp.asarray(dlogz, dtype), jnp.asarray(max_steps, i32))

    def budget():
        left = [
            _INT32_MAX if maxcall is None else min(max(maxcall - n, 0), _INT32_MAX)
            for n in ncall
        ]
        return jnp.asarray(left[0] if batch is None else left, i32)

    def save():
        for rows in rows_by_lane:
            rows[:] = [_concat(rows)]
        _checkpoint.save(
            checkpoint,
            _checkpoint.Checkpoint(
                state=state,
                dead=[rows[0] for rows in rows_by_lane],
                init_key=key,
                ncall=ncall,
                ncall_valid=ncall_valid,
                wall_time_s=wall0 + time.perf_counter() - t0,
                compile_s=compile_s,
                sampling_s=sampling_s,
                chunks=nchunks,
                config=dataclasses.asdict(cfg),
                batch=batch,
                batched_data=batched_data,
                mixed_build=mixed_build,
            ),
        )

    # A zero-step chunk compiles the kernel, so compile_s is measured apart.
    tick = time.perf_counter()
    zero_steps = jnp.asarray(0, i32)
    jax.block_until_ready(kernel(state, zero_steps, *limits, budget(), *leaves))
    compile_s += time.perf_counter() - tick
    last_save = time.perf_counter()
    n_steps = 1
    while True:
        tick = time.perf_counter()
        state, rows, count, remain = kernel(
            state, jnp.asarray(n_steps, i32), *limits, budget(), *leaves
        )
        count, calls, valid, status, remain, logz, it, log_scale, rows = (
            jax.device_get((
                count, state.ncall, state.ncall_valid, state.status, remain,
                state.logz, state.it, state.log_scale, rows,
            ))
        )
        elapsed = time.perf_counter() - tick
        sampling_s += elapsed
        nchunks += 1
        if batch is None:  # one lane: give every output a lane axis
            count, calls, valid, status, remain, logz, it, log_scale = (
                np.asarray(x)[None]
                for x in (count, calls, valid, status, remain, logz, it, log_scale)
            )
            rows = jax.tree_util.tree_map(lambda r: r[:, None], rows)
        status = np.array(status, np.int32)
        for lane in range(lanes):
            ncall[lane] += int(calls[lane])
            ncall_valid[lane] += int(valid[lane])
            n = int(count[lane])
            # A copy: a view would keep the chunk's whole buffer (every lane,
            # the full capacity) alive until the rows are concatenated.
            rows_by_lane[lane].append(
                jax.tree_util.tree_map(lambda r, n=n, b=lane: np.array(r[:n, b]), rows)
            )
            if status[lane] == MAXCALL and (maxcall is None or ncall[lane] < maxcall):
                status[lane] = RUNNING  # only the int32 window of the budget was used
        zero = jnp.zeros_like(state.ncall)
        state = state._replace(
            ncall=zero,
            ncall_valid=zero,
            status=jnp.asarray(status[0] if batch is None else status),
        )
        done = not np.any(status == RUNNING)
        if progress:
            moves = sum(int(rows.moves[: count[b], b].sum()) for b in range(lanes))
            nrows = int(count.sum())
            print(
                _progress_line(
                    cfg, batch, status, it, logz, remain, ncall,
                    float(calls.sum()) / max(elapsed, 1e-9),
                    moves / (nrows * k * cfg.walks) if nrows else float("nan"),
                    np.exp(np.asarray(log_scale, np.float64)),
                ),
                flush=True,
            )
        if checkpoint is not None and (
            done or time.perf_counter() - last_save >= _CHECKPOINT_SECONDS
        ):
            save()
            last_save = time.perf_counter()
        if done:
            break
        per_step = elapsed / max(int(count.max()), 1)
        n_steps = int(min(capacity, 4 * n_steps, max(1.0, _CHUNK_SECONDS / per_step)))

    wall_time_s = wall0 + time.perf_counter() - t0
    results = []
    # The lanes are sliced on the host (finalise does not read the key).
    state = jax.device_get(state._replace(key=None))
    for lane in range(lanes):
        lane_state = state
        lane_prior = prior_transform
        if batch is not None:
            lane_state = jax.tree_util.tree_map(lambda x, b=lane: x[b], state)
            if batched_data:
                lane_prior = _lane_callable(prior_transform, lane)
        results.append(
            finalise(
                lane_state,
                _concat(rows_by_lane[lane]),
                cfg,
                prior_transform=lane_prior,
                ncall=ncall[lane],
                metadata={
                    "ncall_valid": ncall_valid[lane],
                    "dlogz": dlogz,
                    "maxiter": maxiter,
                    "maxcall": maxcall,
                    "chunks": nchunks,
                    "wall_time_s": wall_time_s,
                    "compile_s": compile_s,
                    "sampling_s": sampling_s,
                    "resumed": resumed,
                    "batched_data": batched_data,
                },
            )
        )
    return results[0] if batch is None else results


def _progress_line(cfg, batch, status, it, logz, remain, ncall, rate, acc, scale):
    """One progress line: iteration, logz, dlogz, calls, calls/s, acceptance
    and step scale. For a batch: the number of running lanes, the range of
    the iteration, the largest dlogz and the median scale over the running
    lanes alone (a stopped lane is frozen), and the number of stopped lanes
    in each status."""
    k = cfg.num_delete
    if batch is None:
        return (
            f"tinyns: niter={int(it[0]) * k} logz={float(logz[0]):.4f} "
            f"dlogz={float(remain[0]):.3g} ncall={ncall[0]} calls/s={rate:.3g} "
            f"acc={acc:.3f} scale={float(scale[0]):.3g} [{STATUS[int(status[0])]}]"
        )
    running = status == RUNNING
    line = f"tinyns: lanes running={int(running.sum())}/{batch} "
    if running.any():
        it = it[running]
        line += (
            f"niter={int(it.min()) * k}..{int(it.max()) * k} "
            f"dlogz<={float(np.max(remain[running])):.3g} "
        )
    line += f"ncall={sum(ncall)} calls/s={rate:.3g} acc={acc:.3f}"
    if running.any():
        line += f" scale={float(np.median(scale[running])):.3g}"
    stopped = np.bincount(status[~running], minlength=len(STATUS))
    if stopped.any():
        line += " [" + " ".join(
            f"{name}={n}" for name, n in zip(STATUS, stopped, strict=True) if n
        ) + "]"
    return line
