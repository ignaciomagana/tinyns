"""On-device mode tracking and the inter-mode moves of the replacement chains.

Why: a replacement chain starts from a live point and cannot cross between
separated modes, so the number of live points in each mode does a random walk
(a Polya urn) and the mode weights drift from seed to seed. An inter-mode
move that is exact for the constrained prior restores the balance: the flux
between two modes vanishes when their populations are in proportion to their
volumes, and the frames below only set how fast the populations relax.

**Clustering** (:func:`recluster`, pure JAX, static shapes, ``C_MAX`` slots).
Every ``Config.recluster_every`` steps (about a quarter of an e-fold) the live
points are relabelled: warm-started hard EM from the current labels, merges of
cluster pairs whose Fisher separation ``J = dm^T Sw^-1 dm`` fell below half
the split threshold, then top-down splits. A split of a cluster is 2-means in
the cluster's whitened coordinates from four starts (farthest points, and the
best one-dimensional cuts along the independent-component directions),
refined by hard EM; the best ``J`` (deflated for small clusters) is accepted
if it is at least ``max(25, 1.5 (d + 2))`` and both parts have at least 3
points. A single convex mode gives ``J`` of 10 to 16 however it is cut.
Each cluster has a frame: its mean and the Cholesky factor of its covariance
shrunk toward the pooled within-cluster covariance (prior weight ``0.1 d``
points). A frame is *eligible* for the moves if its volume share predicts
at least ``2 d`` live points (``m V_c / sum V >= 2 d``) and it has more than
``d`` members; the most populated cluster always is. A rule on the live
count instead would switch the moves on only while a cluster is
over-populated and so drain it.

**Moves** (the private static ``Config._mode``). Every ``HOP_EVERY``-th step of
a chain is the arm's move (``P_HOP = 1 / HOP_EVERY``), the others the global
live-covariance random walk; with fewer than two eligible frames the move step
is a random-walk step. The arms:

``N``
    the random walk only.
``B_ell``
    independence Metropolis-Hastings from the uniform law on the union of the
    eligible frames' ellipsoids ``|L^-1 (x - mu)|^2 <= d + 2`` (a frame chosen
    in proportion to its volume), so ``q(x)`` is proportional to the number of
    ellipsoids containing ``x``; accepted iff ``x'`` is in the cube, above
    ``L*`` and ``U < q(x) / q(x')``.
``B_t``
    the same with a Student-t mixture (``nu = 4``, covariance ``1.5^2 L L^T``,
    weights proportional to the frame volume), which has full support.
``C``
    the affine swap ``u' = mu_b + L_b L_a^-1 (u - mu_a)``, with ``a`` the
    nearest frame of ``u`` (Mahalanobis) and ``b`` uniform among the other
    eligible frames; accepted iff ``u'`` is in the cube, its nearest frame is
    ``b``, ``U < det L_b / det L_a`` and ``L(u') > L*``.
``BC``
    the move steps alternate between ``B_t`` and ``C``.

**Why the moves are exact.** Given the other live points (and the dead
points), a chain's seed is uniform in the constrained region ``{L > L*}``. A
kernel that leaves that uniform law invariant and does not depend on the seed
returns a point with the same law, for any number of steps. Each move above
leaves it invariant for *fixed* frames: B is independence MH with the
Hastings ratio ``q(x) / q(x')``; C is a deterministic involution (``T_ab`` and
``T_ba`` are inverse maps with constant Jacobian ``det L_b / det L_a``) whose
forward and reverse selection probabilities are equal (``1 / (n_eligible -
1)``, both ends nearest to their own frame), so the MH-Green ratio is the
Jacobian. The schedule of move and walk steps is fixed in advance (a
composition of invariant kernels) and the frames do not change during a
chain. What remains is that the frames must not depend on the seed. They are
fitted at the last recluster from the live points of that time, which may
include the seed. If so (the per-point ``fitted`` flag), the chain uses frames
refitted without it: the mean, scatter matrix and count of the seed's cluster
get a rank-one downdate, and everything derived from them (the pooled
covariance the frames are shrunk toward, every frame's Cholesky factor, the
volume shares and the eligibility) is recomputed from the downdated
statistics (:func:`chain_frames`). With ``k`` chains per step each chain
removes only its own seed. A seed born after the last recluster is in no
frame's statistics and the frames are used as they are. The partition of the
other points is taken as given; their labels came out of a clustering that
saw the seed, a dependence of one point in ``m`` on hard-EM labels that the
downdate does not remove (as in any population-adapted proposal, the live
points are also not exactly independent). Without the downdate a seed always
sits inside the frame fitted to it while the reverse flux covers only the
frame's true volume, and small clusters drain.

New live points take the label of their nearest frame (``fitted`` false);
:func:`recluster` relabels everything and marks every live point fitted.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

ARMS = ("N", "B_ell", "B_t", "C", "BC")
C_MAX = 8  # cluster slots (static shapes)
HOP_EVERY = 10  # every 10th chain step is the arm's move: P_HOP = 0.1
SPLIT_J_ABS, SPLIT_J_PER_DIM, MERGE_FRACTION = 25.0, 1.5, 0.5
MIN_SPLIT_POINTS = 3  # each side of a split needs this many points
SHRINK = 0.1  # frame covariances: prior weight 0.1 d points on the pooled one
ELIGIBLE_PER_DIM = 2.0  # a frame moves if its volume share predicts 2 d points
ELL_R2_PER_DIM = 1.0  # B_ell ellipsoids: Mahalanobis^2 <= (d + 2) * this
T_NU, T_SCALE = 4.0, 1.5  # B_t components: Student-t, nu = 4, scale x 1.5
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


def _eligible(logdet, count, active, nlive, d):
    """Volume share predicts ``2 d`` live points, more than ``d`` members; the
    most populated cluster always takes part."""
    logv = jnp.where(active, logdet, -jnp.inf)
    share = jnp.exp(logv - logsumexp(logv))
    eligible = active & (nlive * share >= ELIGIBLE_PER_DIM * d) & (count > d)
    largest = jnp.argmax(jnp.where(active, count, -1.0))
    return eligible.at[largest].set(active[largest])


def downdate(st: Stats, x, label, weight) -> Stats:
    """``st`` with point ``x`` removed from cluster ``label`` (if ``weight``).

    ``weight`` is 1 (the point is in the statistics) or 0 (unchanged).
    """
    mu, scat, count = st
    a = label
    n = count[a]
    w = jnp.asarray(weight, mu.dtype) * (n > 0.5)
    left = n - w
    dx = x - mu[a]
    mu_a = mu[a] - w * dx / jnp.maximum(left, 1.0)
    scat_a = scat[a] - (w * n / jnp.maximum(left, 1.0)) * jnp.outer(dx, dx)
    empty = left < 0.5
    mu_a = jnp.where(empty, 0.0, mu_a)
    scat_a = jnp.where(empty, 0.0, scat_a)
    return Stats(mu.at[a].set(mu_a), scat.at[a].set(scat_a), count.at[a].set(left))


def chain_frames(st: Stats, seed_u, seed_label, seed_fitted, nlive: int) -> Frames:
    """The frames of one chain: refitted without its seed if the seed was fitted."""
    return frames(downdate(st, seed_u, seed_label, seed_fitted), nlive)


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


# --------------------------------------------------------------- the moves


def move_enabled(fr: Frames):
    """The move needs two eligible frames; otherwise its steps random-walk."""
    return jnp.sum(fr.eligible) >= 2


def propose(arm: str, key, u, fr: Frames, index):
    """Propose the arm's move from ``u``; return ``(u_new, ok)``.

    ``ok`` holds every acceptance test that needs no likelihood (the
    Metropolis-Hastings ratio and, for the swap, the nearest-frame test); the
    caller also requires ``u_new`` in the cube and above ``L*``. ``index`` is
    the move's index within the chain (``BC`` alternates on it).
    """
    if arm == "B_ell":
        return _propose_ellipsoid(key, u, fr)
    if arm == "B_t":
        return _propose_student(key, u, fr)
    if arm == "C":
        return _propose_swap(key, u, fr)
    if arm == "BC":
        k_t, k_s = random.split(key)
        u_t, ok_t = _propose_student(k_t, u, fr)
        u_s, ok_s = _propose_swap(k_s, u, fr)
        swap = index % 2 == 1
        return jnp.where(swap, u_s, u_t), jnp.where(swap, ok_s, ok_t)
    raise ValueError(f"unknown arm {arm!r}")


def _volume_logits(fr: Frames):
    return jnp.where(fr.eligible, fr.logdet, -jnp.inf)


def _unit_ball(key, d, dtype):
    k_dir, k_rad = random.split(key)
    z = random.normal(k_dir, (d,), dtype)
    return z / jnp.linalg.norm(z) * random.uniform(k_rad, (), dtype) ** (1.0 / d)


def _propose_ellipsoid(key, u, fr: Frames):
    d = u.shape[0]
    r2max = ELL_R2_PER_DIM * (d + 2.0)
    k_c, k_x, k_acc = random.split(key, 3)
    c = random.categorical(k_c, _volume_logits(fr))
    new = fr.mu[c] + math.sqrt(r2max) * (fr.chol[c] @ _unit_ball(k_x, d, u.dtype))

    def n_in(x):
        return jnp.sum(fr.eligible & (mahalanobis(fr, x)[0] <= r2max))

    n_new, n_old = n_in(new), n_in(u)
    accept = random.uniform(k_acc, (), u.dtype) * n_new < n_old
    return new, accept & (n_new >= 1)


def _log_student_mixture(fr: Frames, x):
    """``log q(x)`` up to a constant: the frame weights (proportional to the
    volume) cancel the frames' normalizations, leaving the kernels' sum."""
    d = x.shape[0]
    r2 = mahalanobis(fr, x)[0] / T_SCALE**2
    terms = -0.5 * (T_NU + d) * jnp.log1p(r2 / T_NU)
    return logsumexp(jnp.where(fr.eligible, terms, -jnp.inf))


def _propose_student(key, u, fr: Frames):
    d = u.shape[0]
    k_c, k_z, k_g, k_acc = random.split(key, 4)
    c = random.categorical(k_c, _volume_logits(fr))
    z = random.normal(k_z, (d,), u.dtype)
    chi2 = 2.0 * random.gamma(k_g, 0.5 * T_NU, (), u.dtype)
    new = fr.mu[c] + T_SCALE * jnp.sqrt(T_NU / chi2) * (fr.chol[c] @ z)
    log_ratio = _log_student_mixture(fr, u) - _log_student_mixture(fr, new)
    accept = jnp.log(random.uniform(k_acc, (), u.dtype)) < log_ratio
    return new, accept


def _propose_swap(key, u, fr: Frames):
    r_pick, r_acc = random.uniform(key, (2,), u.dtype)
    dist, y = mahalanobis(fr, u)
    a = jnp.argmin(dist)
    others = fr.eligible & (jnp.arange(C_MAX) != a)
    n_others = jnp.sum(others)
    # b is uniform among the other eligible frames. The reverse move starts
    # nearest to b and picks a among as many, so only the Jacobian is left.
    pick = jnp.minimum(jnp.floor(r_pick * n_others).astype(jnp.int32), n_others - 1)
    b = jnp.argmax(jnp.cumsum(others) == pick + 1)
    new = fr.mu[b] + fr.chol[b] @ y[a]
    accept = (
        fr.eligible[a]
        & (n_others > 0)
        & (jnp.argmin(mahalanobis(fr, new)[0]) == b)
        & (jnp.log(r_acc) < fr.logdet[b] - fr.logdet[a])
    )
    return new, accept


# ------------------------------------------------------------ clustering


def _assign(fr: Frames, x, keep):
    """Hard-EM classification: ``argmin r^2 + 2 log det L - 2 log weight``.

    Rows with ``keep`` false get label -1.
    """
    total = jnp.maximum(jnp.sum(fr.count), 1.0)
    log_weight = jnp.log(jnp.maximum(fr.count, 1e-30) / total)
    offset = 2.0 * fr.logdet - 2.0 * log_weight  # (C,)
    dev = x[None, :, :] - fr.mu[:, None, :]  # (K, m, d)
    z = solve_triangular(fr.chol, jnp.swapaxes(dev, 1, 2), lower=True)  # (K, d, m)
    score = jnp.sum(z * z, axis=1) + offset[:, None]
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


def _refine(x, labels, nslots: int, iters: int):
    """Hard EM over ``nslots`` clusters: refit and reassign, up to ``iters``
    times (until the labels repeat).

    Rows labelled -1 stay out. Slots that empty out stay empty.
    """
    keep = labels >= 0

    def step(labels):
        return _assign(frames(stats(x, labels, nslots)), x, keep)

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
    y = solve_triangular(chol, xc.T, lower=True).T  # whitened; 0 outside
    r2 = jnp.sum(y * y, axis=1)
    far = y[jnp.argmax(jnp.where(mask, r2, -jnp.inf))]
    far2 = y[jnp.argmax(jnp.where(mask, jnp.sum((y - far) ** 2, axis=1), -jnp.inf))]

    def nearer(c0, c1):
        return jnp.sum((y - c1) ** 2, axis=1) < jnp.sum((y - c0) ** 2, axis=1)

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
            c1 = (weight @ y) / jnp.maximum(n1, 1.0)
            c0 = ((w - weight) @ y) / jnp.maximum(n - n1, 1.0)
            return nearer(c0, c1) & mask

        part = _until_fixed(lloyd, part, LLOYD_ITERS)
        labels = jnp.where(mask, part.astype(jnp.int32), -1)
        labels = _refine(y, labels, 2, SPLIT_EM_ITERS)
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


def _split_slots(u, labels):
    """The best split of every slot of every lane: ``(J, part)``, shapes
    ``(B, C)`` and ``(B, C, m)``.

    The slots run in sequence, and a slot that no lane could split (fewer
    than ``max(6, 4 d)`` members in every lane, e.g. an empty slot) is
    skipped by a ``lax.cond`` whose predicate is unbatched: the search costs
    the occupied slots only, also for a batch of runs.
    """
    lanes, m, d = u.shape
    size = jnp.sum(labels[:, None, :] == jnp.arange(C_MAX)[None, :, None], axis=-1)
    need = jnp.any(size >= max(2 * MIN_SPLIT_POINTS, 4 * d), axis=0)  # (C,)

    def slot(xs):
        c, go = xs

        def search(_):
            return jax.vmap(lambda u, lab: _split(u, lab == c))(u, labels)

        def skip(_):
            return jnp.zeros((lanes,), u.dtype), jnp.zeros((lanes, m), bool)

        return lax.cond(go, search, skip, None)

    js, parts = lax.map(slot, (jnp.arange(C_MAX), need))
    return jnp.swapaxes(js, 0, 1), jnp.swapaxes(parts, 0, 1)


def _split_loop(u, labels, js, parts):
    """Apply the best split while one passes the threshold and a slot is free;
    the two clusters a split changes are searched again."""
    threshold = split_threshold(u.shape[1])

    def occupied(labels):
        return jnp.any(labels[None, :] == jnp.arange(C_MAX)[:, None], axis=1)

    def cond(carry):
        labels, js, _ = carry
        return (jnp.max(js) >= threshold) & ~jnp.all(occupied(labels))

    def body(carry):
        labels, js, parts = carry
        c = jnp.argmax(js)
        slot = jnp.argmin(occupied(labels))  # the first free slot
        labels = jnp.where(parts[c], slot, labels).astype(jnp.int32)
        pair = jnp.stack([c, slot])
        new_j, new_parts = jax.vmap(lambda s: _split(u, labels == s))(pair)
        return labels, js.at[pair].set(new_j), parts.at[pair].set(new_parts)

    return lax.while_loop(cond, body, (labels, js, parts))[0]


def recluster_lanes(u, labels):
    """:func:`recluster` for a batch of independent live sets: ``u`` of shape
    ``(B, m, d)`` and ``labels`` ``(B, m)``; returns batched labels and stats.

    Call it with the lane axis explicit (not under ``vmap``): the split search
    then skips the slots that no lane occupies.
    """
    labels = jnp.clip(labels, 0, C_MAX - 1).astype(jnp.int32)
    labels = jax.vmap(lambda u, lab: _refine(u, lab, C_MAX, REFINE_ITERS))(u, labels)
    labels = jax.vmap(_merge)(u, labels)
    js, parts = _split_slots(u, labels)
    labels = jax.vmap(_split_loop)(u, labels, js, parts)
    return labels, jax.vmap(lambda u, lab: stats(u, lab, C_MAX))(u, labels)


def recluster(u, labels):
    """Relabel the live points ``u`` starting from ``labels``; return
    ``(labels, stats)`` (labels in ``0..C_MAX-1``; slots keep their ids).

    Hard EM from the current labels, then merges (the pair of lowest ``J``,
    while it is below half the split threshold), then splits (the best split
    of any cluster, while one passes the threshold and a slot is free).
    """
    labels, st = recluster_lanes(u[None], labels[None])
    return labels[0], Stats(*(x[0] for x in st))
