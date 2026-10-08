"""On-device mode tracking, the inter-mode hop and the local random walk.

Why: a replacement chain starts from a live point and cannot cross between
separated modes, so the number of live points in each mode does a random walk
(a Polya urn) and the mode weights drift from seed to seed. The hop below is
exact for the constrained prior and restores the balance: the flux between
two modes vanishes when their populations are in proportion to their
volumes, and the frames only set how fast the populations relax. In the v1
bake-off (``bench/bakeoff/``) it cut the seed-to-seed logit sd of minor-mode
masses about 7x at the same number of likelihood calls. Within a mode, a walk
in the covariance of all live points mixes poorly when the modes differ in
shape (that covariance is the largest mode's, plus the offsets between
modes), so the walk uses the covariance of the current point's cluster.

**Folds (cross-fitting).** The live slots form ``K = Config._folds`` fixed
folds (slot index mod ``K``, default 3). Each new point fills a dead slot and
is seeded from a live point above ``L*`` of the same fold, so a point and all
its descendants stay in one fold. Clustering ``j`` is of the points outside
fold ``j``, and the kernel of a chain seeded in fold ``j`` (the hop's frames,
the walk's cluster frames and the live covariance the walk falls back to) is
built from those points alone. This generalizes the two-ensemble split of
emcee's parallel stretch move (Foreman-Mackey et al. 2013, after Goodman and
Weare 2010), which moves each half of the walkers with proposals built from
the other half; ``K = 3`` fits every frame to two thirds of the live points.

**Clustering** (:func:`recluster`, pure JAX, static shapes, ``C_MAX`` slots per
clustering, the ``K`` clusterings as lanes of one call). Every
``Config.recluster_every`` steps (about a quarter of an e-fold) each
clustering is refreshed: warm-started hard EM from the current labels, merges
of cluster pairs whose Fisher separation ``J = dm^T Sw^-1 dm`` fell below half
the split threshold, then top-down splits. A split of a cluster is 2-means in
the cluster's whitened coordinates from four starts (farthest points, and the
best one-dimensional cuts along the independent-component directions),
refined by hard EM; the best ``J`` (deflated for small clusters) is accepted
if it is at least ``max(25, 1.5 (d + 2))`` and both parts have at least 3
points. A single convex mode gives ``J`` of 10 to 16 however it is cut. Each
cluster has a frame: its mean and the Cholesky factor of its covariance shrunk
toward the pooled within-cluster covariance (prior weight ``0.1 d`` points).
A frame is *eligible* for the hop if its volume share predicts at least ``2
d`` live points (``m V_c / sum V >= 2 d``) and it has more than ``d``
members; the most populated cluster always is. A frame *walks*
(:func:`walk_frames`) if it has at least ``WALK_MIN_POINTS = 3`` members (a
small mode keeps its own covariance, shrunk toward the pooled one, until it
is nearly gone). The frames are factorized once, at the recluster, and kept
in the state until the next one; new live points take the label of the
nearest frame of each clustering until then. The labels are also the record that
:meth:`tinyns.NestedSamplingResult.modes` reads: the id of a point's cluster
in the clustering of the next fold (which contains it) when the point died.
A slot's cluster takes a new id when a split writes it (both parts of a
split do; a merge keeps the id of the slot that absorbs the other), so an id
never names a cluster and, later, one part of it; ``modes()`` merges the
labels of different clusterings and times that cover one mode.

**The hop.** Every ``HOP_EVERY``-th step of a chain (``P_HOP = 1 /
HOP_EVERY``) is an independence Metropolis-Hastings step from the uniform law
on the union of the eligible frames' ellipsoids ``|L^-1 (x - mu)|^2 <= d +
2``: a frame is chosen in proportion to its volume and a point drawn
uniformly in its ellipsoid, so the proposal density ``q(x)`` is proportional
to the number of ellipsoids containing ``x``. The proposal ``x'`` is accepted
iff it is in the cube, above ``L*`` and ``U < q(x) / q(x')``; a start outside
every ellipsoid (``q(x) = 0``) never moves. With fewer than two eligible
frames the hop step is a walk step. The proposal does not depend on the
chain's point, so a chain draws those of all its hop steps before it runs
(:func:`hop_proposal`). The private ``Config._hop=False`` turns the hop off
(the clustering still runs), for comparisons.

**The local walk.** The other steps propose ``y = x + s L_c(x) z``, with
``c(x)`` the nearest walking frame (Mahalanobis distance) and ``L_c`` its
Cholesky factor; ``s`` is the global step scale, adapted to a walk acceptance
of 1/4. The proposal is not symmetric when ``c(y) != c(x)``, so ``y`` is
accepted iff it is in the cube, above ``L*`` and ``U < N(x | y, s^2 S_c(y)) /
N(y | x, s^2 S_c(x))``. With fewer than two walking frames the walk uses the
live covariance of the points outside the fold (symmetric). The private
``Config._local=False`` always uses that covariance, for comparisons.

**Cost.** A chain carries the whitened offsets ``L_c^-1 (x - mu_c)`` of its
point to the frames, so a step whitens one point, the proposal, and reads the
label, the Hastings ratio and the hop's count of ellipsoids from the two sets
of offsets. A step of the sampler picks the cheapest of three chain kernels
that its frames allow (:func:`tinyns.core._kernel_level`; they return the
same chains): plain symmetric walks while no clustering has two walk clusters
or two eligible ones (every unimodal run), a kernel that looks up the two
frames in use when no clustering has more, and one that looks up all
``C_MAX``. The split search of a recluster runs in rounds, one cluster of
every clustering per round, as many as the clustering with the most clusters
needs.

**Why the chains are exact.** For independent live points, given the points
outside fold ``j`` (and the dead points), a seed drawn from the points of fold
``j`` above ``L*`` is uniform in the constrained region ``{L > L*}``. A kernel
that leaves that uniform law invariant and depends only on the points outside
fold ``j`` returns a point with the same law, for any number of steps. The hop
and the local walk each leave it invariant for fixed frames
(Metropolis-Hastings with the exact ratio of proposal densities, restricted
to the cube and ``{L > L*}``), the schedule of hop and walk steps is fixed in
advance (a composition of invariant kernels), and the frames and covariances
do not change during a chain. Nothing of the seed's fold enters its kernel:
not the seed, not the clustering of the other points (fold ``j`` is left out
of it), and not the seed's ancestors, which are in fold ``j``. The restoring
force of the hop on the points of fold ``j`` (how well the frames cover each
mode) depends on the populations of the other folds, not on fold ``j``'s.
What remains is common to every MCMC-driven nested sampler: a new point is
correlated with its seed, and the step scale adapts on past chains.

The v1 hop refitted the seed's cluster without the seed (a rank-one downdate
of the statistics of all live points) and walked in the covariance of all
live points but the step's seeds. On sepW_d18 at nlive 2000 that left the
minor mode 1.25 +- 0.16% light. The downdate's known gap, the seed's influence
on the other points' labels, was not the cause: re-clustering a fixed two-mode
live set with and without the seed (4.4e5 chains, sepW and connW at d = 10
and 18, nlive 500 and 2000) changed no other label. The cause was mixing
within the minor mode: there the global walk accepted 0.08 of its steps at d
= 18 (0.26 in the main mode; 0.06 against 0.26 at d = 32), so chains returned
near-copies of their seeds, which sat inside frames fitted to their parents
and hopped out too readily. The local walk accepts 0.3 to 0.4 in either mode
and removes the bias (+0.2 +- 0.2% with the downdate, 0.0 +- 0.2% with two
folds). The folds also take the seed's ancestors out of the walk covariance:
on sepW_d32 at nlive 500 the seed-free covariance of all live points left
logZ 0.24 +- 0.05 high, with or without the hop, and cross-fitted kernels
0.04 +- 0.04.

A mode with fewer than about ``5 d`` live points cannot be sampled reliably by
any covariance-adapted walk: on one Gaussian with the shape of sepW_d18's
minor mode, nlive 30, 60 and 120 give logZ 4.35 +- 0.07, 0.21 +- 0.05 and
0.03 +- 0.04 too high. A minor mode that small (sepW_d18 at nlive 500: 30 to
70 points; the d = 32 targets) keeps a bias of a few percent whatever the
moves; see the known limitations in the CHANGELOG.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

C_MAX = 8  # cluster slots (static shapes)
TREE_SIZE = 512  # cluster ids of a clustering whose parent is recorded
HOP_EVERY = 10  # every 10th chain step is the hop: P_HOP = 0.1
SPLIT_J_ABS, SPLIT_J_PER_DIM, MERGE_FRACTION = 25.0, 1.5, 0.5
MIN_SPLIT_POINTS = 3  # each side of a split needs this many points
WALK_MIN_POINTS = 3  # a cluster gives the walk its covariance from this size
SHRINK = 0.1  # frame covariances: prior weight 0.1 d points on the pooled one
ELIGIBLE_PER_DIM = 2.0  # a frame hops if its volume share predicts 2 d points
ELL_R2_PER_DIM = 1.0  # hop ellipsoids: Mahalanobis^2 <= (d + 2) * this
LLOYD_ITERS = 30  # 2-means iterations of a split
SPLIT_EM_ITERS = 8  # hard-EM iterations refining a split
REFINE_ITERS = 2  # warm-started hard-EM iterations of a recluster


class Stats(NamedTuple):
    """Sufficient statistics of the ``C_MAX`` cluster slots (count 0: empty)."""

    mu: jax.Array  # (C, d)
    scat: jax.Array  # (C, d, d) scatter matrices about mu
    count: jax.Array  # (C,) float


class Frames(NamedTuple):
    """Cluster frames: mean, Cholesky factor of the shrunk covariance, its
    inverse, log det of the factor (log volume up to a constant), and flags."""

    mu: jax.Array  # (C, d)
    chol: jax.Array  # (C, d, d)
    ichol: jax.Array  # (C, d, d)
    logdet: jax.Array  # (C,)
    count: jax.Array  # (C,)
    active: jax.Array  # (C,) bool: a non-empty slot with a valid factor
    eligible: jax.Array  # (C,) bool: takes part in the moves


def split_threshold(ndim: int) -> float:
    """The Fisher separation a split needs: ``max(25, 1.5 (d + 2))``."""
    return max(SPLIT_J_ABS, SPLIT_J_PER_DIM * (ndim + 2))


def _deflate(j, n, d):
    """``J`` scaled by ``(n - d - 3) / (n - 2)``, for ``n`` points in ``d`` dims.

    The within-cluster covariance of ``n`` points has ``n - 2`` degrees of
    freedom, and the separation of the best cut of a single mode grows by
    about ``(n - 2) / (n - d - 3)`` (inverse-Wishart mean) when it is
    estimated from few points. Without this factor a cluster of 16 to 20
    points at d = 4 (a small mode) was cut in two in a quarter of snapshots.
    It is 0.96 for 500 points at d = 18.
    """
    return j * jnp.maximum(n - d - 3.0, 0.0) / jnp.maximum(n - 2.0, 1.0)


def empty_stats(ndim: int, dtype) -> Stats:
    return Stats(
        jnp.zeros((C_MAX, ndim), dtype),
        jnp.zeros((C_MAX, ndim, ndim), dtype),
        jnp.zeros((C_MAX,), dtype),
    )


def _ridge(cov):
    """Add a relative ridge of ``10 eps`` times the mean variance."""
    d = cov.shape[-1]
    eps = 10.0 * jnp.finfo(cov.dtype).eps
    scale = jnp.trace(cov, axis1=-2, axis2=-1) / d
    scale = jnp.where(scale > 0, scale, jnp.finfo(cov.dtype).tiny)
    return cov + (eps * scale)[..., None, None] * jnp.eye(d, dtype=cov.dtype)


def _cholesky(cov):
    """Cholesky factor and a flag; the identity where the factorization fails."""
    chol = jnp.linalg.cholesky(cov)
    ok = jnp.all(jnp.isfinite(chol), axis=(-2, -1))
    eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
    return jnp.where(ok[..., None, None], chol, eye), ok


def stats(x, labels, nslots: int) -> Stats:
    """Mean, scatter and count of ``x[labels == c]`` for ``c < nslots``.

    Rows with a label outside ``0..nslots-1`` (e.g. ``-1``) are left out. The
    scatter is taken about the cluster mean (no ``m x d x d`` intermediate).
    """
    member = (labels[None, :] == jnp.arange(nslots)[:, None]).astype(x.dtype)
    count = jnp.sum(member, axis=1)
    mu = (member @ x) / jnp.maximum(count, 1.0)[:, None]
    dev = (x[None, :, :] - mu[:, None, :]) * member[:, :, None]  # (K, m, d)
    scat = jnp.einsum("kmi,kmj->kij", dev, dev)
    return Stats(mu, scat, count)


def frames(st: Stats, nlive: int | None = None) -> Frames:
    """Frames of the clusters in ``st`` (``nlive``: eligibility, else none).

    Each covariance is ``(1 - rho) S / (n - 1) + rho kappa P`` with ``S`` the
    scatter, ``P`` the pooled within-cluster covariance, ``kappa`` its scale
    match ``tr(P^-1 S) / (d (n - 1))`` and ``rho = 0.1 d / (n - 1 + 0.1 d)``;
    a cluster of one point takes ``P``.
    """
    mu, scat, count = st
    d = mu.shape[-1]
    dtype = mu.dtype
    nonempty = count > 0.5
    n_clusters = jnp.sum(nonempty).astype(dtype)
    within = jnp.maximum(jnp.sum(count) - n_clusters, 1.0)
    pooled = _ridge(jnp.sum(scat, axis=0) / within)
    pchol, _ = _cholesky(pooled)
    eye = jnp.eye(d, dtype=dtype)
    pinv = jax.scipy.linalg.cho_solve((pchol, True), eye)
    dof = jnp.maximum(count - 1.0, 1.0)
    kappa = jnp.maximum(jnp.einsum("ij,kij->k", pinv, scat) / (d * dof), 1e-6)
    rho = SHRINK * d / (dof + SHRINK * d)
    cov = (1.0 - rho)[:, None, None] * scat / dof[:, None, None] + (rho * kappa)[
        :, None, None
    ] * pooled
    cov = jnp.where((count > 1.5)[:, None, None], cov, pooled)
    chol, ok = _cholesky(_ridge(cov))
    active = nonempty & ok
    ichol = solve_triangular(chol, jnp.broadcast_to(eye, chol.shape), lower=True)
    logdet = jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    if nlive is None:
        eligible = jnp.zeros_like(active)
    else:
        eligible = _eligible(logdet, count, active, nlive, d)
    return Frames(mu, chol, ichol, logdet, count, active, eligible)


def empty_frames(ndim: int, dtype) -> Frames:
    """Frames of ``C_MAX`` empty slots (none active)."""
    eye = jnp.broadcast_to(jnp.eye(ndim, dtype=dtype), (C_MAX, ndim, ndim))
    zero = jnp.zeros((C_MAX,), dtype)
    off = jnp.zeros((C_MAX,), bool)
    return Frames(jnp.zeros((C_MAX, ndim), dtype), eye, eye, zero, zero, off, off)


def _eligible(logdet, count, active, nlive, d):
    """Volume share predicts ``2 d`` live points, more than ``d`` members; the
    most populated cluster always takes part."""
    logv = jnp.where(active, logdet, -jnp.inf)
    share = jnp.exp(logv - logsumexp(logv))
    eligible = active & (nlive * share >= ELIGIBLE_PER_DIM * d) & (count > d)
    largest = jnp.argmax(jnp.where(active, count, -1.0))
    return eligible.at[largest].set(active[largest])


def mahalanobis(fr: Frames, x):
    """Squared Mahalanobis distance of ``x`` to every frame (``inf`` if
    inactive) and the whitened offsets ``L^-1 (x - mu)``, shape ``(C, d)``."""
    y = jnp.einsum("cij,cj->ci", fr.ichol, x[None, :] - fr.mu)
    r2 = jnp.sum(y * y, axis=-1)
    return jnp.where(fr.active, r2, jnp.inf), y


def nearest(fr: Frames, x):
    """Label of the nearest active frame of each row of ``x`` (Mahalanobis)."""
    r2 = jax.vmap(lambda v: mahalanobis(fr, v)[0])(x)
    return jnp.argmin(r2, axis=-1).astype(jnp.int32)


# ----------------------------------------------------------------- the hop


def walk_frames(fr: Frames):
    """The frames that give the random walk its covariance: active, with at
    least ``WALK_MIN_POINTS`` members."""
    return fr.active & (fr.count >= WALK_MIN_POINTS)


def hop_enabled(fr: Frames):
    """The hop needs two eligible frames; otherwise its steps random-walk."""
    return jnp.sum(fr.eligible) >= 2


def hop_proposal(key, eligible, logdet, frame):
    """Draw a hop proposal: ``(point, uniform)``.

    ``point`` is uniform in the ellipsoid ``|L^-1 (x - mu)|^2 <= d + 2`` of an
    eligible frame chosen in proportion to its volume (``eligible`` and
    ``logdet`` of the ``C_MAX`` frames; ``frame(c)`` returns ``(mu, L)`` of
    frame ``c``), and ``uniform`` is the draw of its Metropolis-Hastings test
    (:func:`hop_accept`). The proposal does not depend on the chain's point.
    One normal vector and three uniforms: the frame by inversion of the
    volumes' distribution, the radius ``U^(1/d)`` and the test.
    """
    k_dir, k_uni = random.split(key)
    dtype = logdet.dtype
    uni = lax.optimization_barrier(random.uniform(k_uni, (3,), dtype))
    logv = jnp.where(eligible, logdet, -jnp.inf)
    cdf = jnp.cumsum(jnp.where(eligible, jnp.exp(logv - jnp.max(logv)), 0.0))
    c = jnp.sum(cdf <= uni[0] * cdf[-1])  # the first frame with cdf above
    last = eligible.shape[0] - 1 - jnp.argmax(eligible[::-1])
    mu, chol = frame(jnp.minimum(c, last))
    d = mu.shape[0]
    # The barrier keeps XLA from fusing the generator into the norm: on a
    # GPU that fusion compiled for 40 to 90 s at d = 32 and 64.
    z = lax.optimization_barrier(random.normal(k_dir, (d,), dtype))
    ball = z / jnp.linalg.norm(z) * uni[1] ** (1.0 / d)
    point = mu + math.sqrt(ELL_R2_PER_DIM * (d + 2.0)) * (chol @ ball)
    return point, uni[2]


def hop_accept(uniform, eligible, r2_new, r2_old, d: int):
    """The hop's Metropolis-Hastings test ``U q(new) < q(old)`` for the
    uniform draw ``U``.

    The proposal density ``q`` is proportional to the number of eligible
    ellipsoids that hold a point, read from its squared Mahalanobis
    distances ``r2`` to the frames; a proposal in no ellipsoid (it cannot be
    drawn) is refused. The caller also requires the proposal in the cube and
    above ``L*``.
    """
    r2max = ELL_R2_PER_DIM * (d + 2.0)
    n_new = jnp.sum(eligible & (r2_new <= r2max))
    n_old = jnp.sum(eligible & (r2_old <= r2max))
    return (uniform * n_new < n_old) & (n_new >= 1)


# ------------------------------------------------------------ clustering


def _assign(fr: Frames, x, keep, whitened: bool = False):
    """Hard-EM classification: ``argmin r^2 + 2 log det L - 2 log weight``.

    Rows with ``keep`` false get label -1. ``whitened``: ``x`` is of order one
    (the whitened coordinates of a cluster), and ``L^-1 (x - mu)`` is taken
    as ``L^-1 x - L^-1 mu``, one product with ``x`` for all the clusters. (In
    the unit cube a late cluster is a tiny box far from the origin, and that
    difference would lose its digits in float32.)
    """
    total = jnp.maximum(jnp.sum(fr.count), 1.0)
    log_weight = jnp.log(jnp.maximum(fr.count, 1e-30) / total)
    offset = 2.0 * fr.logdet - 2.0 * log_weight  # (C,)
    if whitened:
        z = jnp.einsum("kij,mj->kmi", fr.ichol, x)
        z = z - jnp.einsum("kij,kj->ki", fr.ichol, fr.mu)[:, None, :]
    else:
        dev = x[None, :, :] - fr.mu[:, None, :]  # (K, m, d)
        z = jnp.einsum("kij,kmj->kmi", fr.ichol, dev)
    score = jnp.sum(z * z, axis=2) + offset[:, None]
    score = jnp.where(fr.active[:, None], score, jnp.inf)
    return jnp.where(keep, jnp.argmin(score, axis=0), -1).astype(jnp.int32)


def _until_fixed(step, state, iters: int):
    """Apply ``step`` until its output repeats, at most ``iters`` times (the
    same result as exactly ``iters`` times: a fixed point stays fixed)."""

    def cond(carry):
        i, _, changed = carry
        return (i < iters) & changed

    def body(carry):
        i, state, _ = carry
        new = step(state)
        return i + 1, new, jnp.any(new != state)

    return lax.while_loop(cond, body, (0, state, jnp.asarray(True)))[1]


def _refine(x, labels, nslots: int, iters: int, whitened: bool = False):
    """Hard EM over ``nslots`` clusters: refit and reassign, up to ``iters``
    times (until the labels repeat).

    Rows labelled -1 stay out. Slots that empty out stay empty. ``whitened``:
    see :func:`_assign`.
    """
    keep = labels >= 0

    def step(labels):
        return _assign(frames(stats(x, labels, nslots)), x, keep, whitened)

    return _until_fixed(step, labels, iters)


def _fisher(x, w, part):
    """Fisher separation of the rows ``w`` split by ``part`` (both float 0/1)."""
    wa, wb = w * (1.0 - part), w * part
    na, nb = jnp.sum(wa), jnp.sum(wb)
    mua = (wa @ x) / jnp.maximum(na, 1.0)
    mub = (wb @ x) / jnp.maximum(nb, 1.0)
    dev = (x - jnp.where(part[:, None] > 0.5, mub, mua)) * w[:, None]
    within = _ridge(dev.T @ dev / jnp.maximum(na + nb - 2.0, 1.0))
    dm = mub - mua
    chol, ok = _cholesky(within)
    z = solve_triangular(chol, dm, lower=True)
    return jnp.where(ok, jnp.sum(z * z), 0.0)


def _best_cut(p, mask):
    """The threshold on the projections ``p`` (rows ``mask``) with the largest
    one-dimensional Fisher ratio, both sides of at least ``MIN_SPLIT_POINTS``;
    returns ``(ratio, threshold)`` (the upper side is ``p > threshold``)."""
    m = p.shape[0]
    n = jnp.sum(mask).astype(p.dtype)
    key = jnp.sort(jnp.where(mask, p, jnp.inf))
    vals = jnp.where(jnp.isfinite(key), key, 0.0)
    left = jnp.arange(1, m + 1, dtype=p.dtype)
    s1, s2 = jnp.cumsum(vals), jnp.cumsum(vals * vals)
    right = n - left
    mean_l = s1 / left
    mean_r = (s1[-1] - s1) / jnp.maximum(right, 1.0)
    ss = (s2 - left * mean_l**2) + (s2[-1] - s2 - right * mean_r**2)
    ratio = (mean_r - mean_l) ** 2 / jnp.maximum(ss / jnp.maximum(n - 2.0, 1.0), 1e-30)
    ok = (left >= MIN_SPLIT_POINTS) & (right >= MIN_SPLIT_POINTS)
    best = jnp.argmax(jnp.where(ok, ratio, -1.0))
    return jnp.where(ok[best], ratio[best], 0.0), key[best]


def _split(x, mask):
    """Best two-way split of the rows ``mask`` of ``x``: ``(J, part)``.

    ``J`` is the Fisher separation scaled by :func:`_deflate`; it is 0 when
    the cluster is too small (fewer than ``max(6, 4 d)`` points) or no start
    gives two parts of at least ``MIN_SPLIT_POINTS``. The split is 2-means in
    the cluster's whitened coordinates, refined by hard EM (which isolates a
    small mode that 2-means would rather share with half the bulk), from four
    starts: the (mean, farthest point) and (farthest point, point farthest
    from it) centre pairs of the measured prototype, and the best
    one-dimensional cuts (Fisher ratio) along the two best of ``d + 2``
    directions: the eigenvectors of ``E[|y|^2 y y^T]`` (the independent
    components, among them any direction along which the modes are
    separated) and the two farthest-point directions. A farthest point is
    often a tail point of the bulk, which then gets halved.
    """
    m, d = x.shape
    dtype = x.dtype
    w = mask.astype(dtype)
    n = jnp.sum(w)
    big = n >= max(2 * MIN_SPLIT_POINTS, 4 * d)
    mean = (w @ x) / jnp.maximum(n, 1.0)
    xc = (x - mean) * w[:, None]
    chol, _ = _cholesky(_ridge(xc.T @ xc / jnp.maximum(n - 1.0, 1.0)))
    ichol = solve_triangular(chol, jnp.eye(d, dtype=dtype), lower=True)
    y = xc @ ichol.T  # whitened; 0 outside
    r2 = jnp.sum(y * y, axis=1)
    far = y[jnp.argmax(jnp.where(mask, r2, -jnp.inf))]
    far2 = y[jnp.argmax(jnp.where(mask, jnp.sum((y - far) ** 2, axis=1), -jnp.inf))]
    total = w @ y

    def nearer(c0, c1):
        """The rows nearer to ``c1`` than to ``c0``: ``|y - c1|^2 < |y -
        c0|^2``, as one product with ``y``."""
        return 2.0 * (y @ (c0 - c1)) < jnp.sum(c0 * c0) - jnp.sum(c1 * c1)

    kurt = (y * (w * r2)[:, None]).T @ y / jnp.maximum(n, 1.0)
    _, vecs = jnp.linalg.eigh(kurt)

    def unit(v):
        return v / jnp.maximum(jnp.linalg.norm(v), 1e-30)

    dirs = jnp.concatenate([vecs.T, unit(far)[None], unit(far2 - far)[None]])
    ratios, cuts = jax.vmap(lambda v: _best_cut(y @ v, mask))(dirs)
    _, top = lax.top_k(ratios, 2)
    starts = jnp.stack([
        nearer(jnp.zeros((d,), dtype), far),
        nearer(far, far2),
        y @ dirs[top[0]] > cuts[top[0]],
        y @ dirs[top[1]] > cuts[top[1]],
    ]) & mask

    def one(part):
        def lloyd(part):
            weight = part.astype(dtype)
            n1 = jnp.sum(weight)
            sum1 = weight @ y
            c1 = sum1 / jnp.maximum(n1, 1.0)
            c0 = (total - sum1) / jnp.maximum(n - n1, 1.0)
            return nearer(c0, c1) & mask

        part = _until_fixed(lloyd, part, LLOYD_ITERS)
        labels = jnp.where(mask, part.astype(jnp.int32), -1)
        labels = _refine(y, labels, 2, SPLIT_EM_ITERS, whitened=True)
        part = (labels == 1).astype(dtype)
        n1 = jnp.sum(part)
        ok = (jnp.minimum(n1, n - n1) >= MIN_SPLIT_POINTS) & big
        return jnp.where(ok, _deflate(_fisher(y, w, part), n, d), 0.0), labels == 1

    js, parts = jax.vmap(one)(starts)
    best = jnp.argmax(js)
    return js[best], parts[best]


def _pair_separation(st: Stats):
    """``(C, C)`` Fisher separation of every pair of clusters (``inf`` on the
    diagonal and for empty slots)."""
    mu, scat, count = st
    nonempty = count > 0.5

    def pair(a, b):
        dof = jnp.maximum(count[a] + count[b] - 2.0, 1.0)
        within = _ridge((scat[a] + scat[b]) / dof)
        chol, ok = _cholesky(within)
        z = solve_triangular(chol, mu[b] - mu[a], lower=True)
        valid = ok & nonempty[a] & nonempty[b] & (a != b)
        j = _deflate(jnp.sum(z * z), count[a] + count[b], mu.shape[-1])
        return jnp.where(valid, j, jnp.inf)

    idx = jnp.arange(C_MAX)
    return jax.vmap(lambda a: jax.vmap(lambda b: pair(a, b))(idx))(idx)


def _merge(u, labels):
    """Merge the pair of lowest ``J`` while it is below half the threshold."""
    threshold = split_threshold(u.shape[1])

    def cond(carry):
        return jnp.min(carry[1]) < MERGE_FRACTION * threshold

    def body(carry):
        labels, j = carry
        flat = jnp.argmin(j)
        a = jnp.minimum(flat // C_MAX, flat % C_MAX)
        b = jnp.maximum(flat // C_MAX, flat % C_MAX)
        labels = jnp.where(labels == b, a, labels).astype(jnp.int32)
        return labels, _pair_separation(stats(u, labels, C_MAX))

    j0 = _pair_separation(stats(u, labels, C_MAX))
    return lax.while_loop(cond, body, (labels, j0))[0]


def _split_all(u, labels):
    """Split every lane's clusters top-down: ``u`` ``(B, m, d)``, ``labels``
    ``(B, m)``; returns the labels, ``fresh`` ``(B, C_MAX)``, the slots a
    split wrote (both parts of every split applied), and ``origin`` ``(B,
    C_MAX)``, the slot that held each slot's points before the splits.

    In each lane: the best split of every cluster that can split (at least
    ``max(6, 4 d)`` members) is searched; while the best of them passes the
    threshold and a slot is free it is applied, and the two clusters it made
    are searched in turn.

    One loop serves all lanes. An iteration applies the due split of every
    lane that has no search pending, then searches one cluster in every lane
    that has one pending (:func:`_split`, vmapped over the lanes). The loop's
    predicate and the ``lax.cond`` around the search are unbatched, so the
    search runs as many times as the busiest lane needs: the number of
    clusters of the lane with the most, plus two per split. (Searching two
    clusters of a lane at once was no faster on a GPU and compiled slower.)
    """
    lanes, m, d = u.shape
    threshold = split_threshold(d)
    min_size = max(2 * MIN_SPLIT_POINTS, 4 * d)
    slots = jnp.arange(C_MAX)
    lane = jnp.arange(lanes)

    def sizes(labels):
        return jnp.sum(labels[:, None, :] == slots[None, :, None], axis=-1)

    def due(labels, js, todo):
        """Lanes whose best split passes, with a free slot and no search
        pending."""
        full = jnp.all(sizes(labels) > 0, axis=1)
        return (jnp.max(js, axis=1) >= threshold) & ~full & ~jnp.any(todo, axis=1)

    def cond(carry):
        labels, js, _, todo, _, _ = carry
        return jnp.any(todo) | jnp.any(due(labels, js, todo))

    def body(carry):
        labels, js, parts, todo, fresh, origin = carry
        # Apply the best split where it is due. The part that moves to the
        # first free slot is the one without the cluster's first row, so the
        # slots do not depend on the orientation of the cut (the sign of an
        # eigenvector, which batched and single solvers can choose apart).
        go = due(labels, js, todo)
        c = jnp.argmax(js, axis=1)
        free = jnp.argmin(sizes(labels) > 0, axis=1)
        member = labels == c[:, None]
        part = parts[lane, c]
        first = jnp.take_along_axis(part, jnp.argmax(member, axis=1)[:, None], 1)
        part = jnp.where(first, member & ~part, part) & go[:, None]
        labels = jnp.where(part, free[:, None], labels).astype(jnp.int32)
        changed = go[:, None] & (
            (slots[None, :] == c[:, None]) | (slots[None, :] == free[:, None])
        )
        js = jnp.where(changed, 0.0, js)
        fresh = fresh | changed
        moved = go[:, None] & (slots[None, :] == free[:, None])
        origin = jnp.where(moved, origin[lane, c][:, None], origin)
        todo = todo | (changed & (sizes(labels) >= min_size))
        # Search the first pending cluster of every lane that has one.
        has = jnp.any(todo, axis=1)
        pick = jnp.argmax(todo, axis=1)
        target = jnp.where(has, pick, C_MAX)  # C_MAX: no row has this label

        def search(_):
            return jax.vmap(lambda u, lab, c: _split(u, lab == c))(u, labels, target)

        def skip(_):
            return jnp.zeros((lanes,), u.dtype), jnp.zeros((lanes, m), bool)

        j, p = lax.cond(jnp.any(has), search, skip, None)
        done = has[:, None] & (slots[None, :] == pick[:, None])
        js = jnp.where(done, j[:, None], js)
        parts = jnp.where(done[:, :, None], p[:, None, :], parts)
        return labels, js, parts, todo & ~done, fresh, origin

    carry = (
        labels,
        jnp.zeros((lanes, C_MAX), u.dtype),
        jnp.zeros((lanes, C_MAX, m), bool),
        sizes(labels) >= min_size,
        jnp.zeros((lanes, C_MAX), bool),
        jnp.broadcast_to(slots, (lanes, C_MAX)),
    )
    out = lax.while_loop(cond, body, carry)
    return out[0], out[4], out[5]


def recluster_lanes(u, labels):
    """:func:`recluster` for a batch of independent live sets: ``u`` of shape
    ``(B, m, d)`` and ``labels`` ``(B, m)``; returns batched labels, stats,
    ``fresh`` ``(B, C_MAX)``, the slots that hold a part of a cluster this
    call split, and ``origin`` ``(B, C_MAX)``, the slot that held that
    cluster. (The caller gives the fresh slots new cluster ids, with the id of
    ``origin`` as their parent; a merge leaves the id of the slot that
    absorbs the other.) Rows labelled ``-1`` (padding) stay out of every
    cluster.

    Call it with the lane axis explicit (not under ``vmap``): the split search
    then skips the slots that no lane occupies.
    """
    labels = jnp.where(labels < 0, -1, jnp.clip(labels, 0, C_MAX - 1))
    labels = labels.astype(jnp.int32)
    labels = jax.vmap(lambda u, lab: _refine(u, lab, C_MAX, REFINE_ITERS))(u, labels)
    occupied = jnp.any(labels[:, None, :] == jnp.arange(C_MAX)[None, :, None], axis=-1)
    labels = lax.cond(  # nothing to merge unless a lane holds two clusters
        jnp.any(jnp.sum(occupied, axis=1) >= 2),
        lambda labels: jax.vmap(_merge)(u, labels),
        lambda labels: labels,
        labels,
    )
    labels, fresh, origin = _split_all(u, labels)
    st = jax.vmap(lambda u, lab: stats(u, lab, C_MAX))(u, labels)
    return labels, st, fresh, origin


def recluster(u, labels):
    """Relabel the live points ``u`` starting from ``labels``; return
    ``(labels, stats)`` (labels in ``0..C_MAX-1``, ``-1`` rows stay out;
    slots keep their ids).

    Hard EM from the current labels, then merges (the pair of lowest ``J``,
    while it is below half the split threshold), then splits (the best split
    of any cluster, while one passes the threshold and a slot is free).
    """
    labels, st, _, _ = recluster_lanes(u[None], labels[None])
    return labels[0], Stats(*(x[0] for x in st))
