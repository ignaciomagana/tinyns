"""Cluster tracking and the affine swap move for multimodal posteriors.

Replacement chains start from live points and cannot cross between separated
modes, so the number of live points in each mode does a random walk and the
mode weights scatter from seed to seed. Two pieces fix that:

* :class:`ClusterTracker` (numpy, on the host between blocks) clusters the
  live points, tracks the clusters across blocks and fits a frame (mean and
  Cholesky factor of a shrunk covariance) to each.
* :func:`swap_step` (JAX, inside a chain) proposes
  ``u' = mu_b + L_b L_a^-1 (u - mu_a)``: the point moves from its own cluster
  ``a`` to the same place in another cluster ``b``.

The map is an involution with a constant Jacobian. Accepting it with
probability ``min(1, det L_b / det L_a)``, when ``u'`` is in the cube, above
the likelihood threshold and nearest to frame ``b``, leaves the constrained
prior invariant whatever the frames are. The flux between two modes then
balances when their populations are in proportion to their volumes; the
frames only set how fast the populations relax.

Three rules keep the move unbiased: the frame of the seed's cluster is refit
without the seed (:func:`loo_frames`), a cluster takes part only if its volume
share, not its live count, predicts enough live points (``ELIGIBLE_PER_DIM``),
and the rwalk steps keep the global live covariance.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

MAX_CLUSTERS = 3  # frames are padded to this many clusters (static shapes)
SWAP_PROB = 0.1  # fraction of chain steps that propose a swap
RECLUSTER_ITERS = 128  # iterations between cluster updates (at most nlive / 4)
# A split is accepted when its Fisher separation J = dm^T Sw^-1 dm is at least
# max(25, 1.5 (d + 2)). A single convex mode gives J of 10 to 16 however it is
# cut. Two clusters are merged again when J falls below half of that.
SPLIT_J_ABS, SPLIT_J_PER_DIM, MERGE_FRACTION = 25.0, 1.5, 0.5
MIN_SPLIT_POINTS = 3  # each side of a split needs this many points
# Frame covariances are shrunk toward the pooled within-cluster covariance
# with a prior weight of 0.1 d points. More shrinkage inflates the frame of a
# small cluster and kills the swap acceptance.
SHRINK = 0.1
# A cluster swaps only if its volume share predicts at least 3 d live points.
# The test must not use the cluster's live count: that would switch the swap
# on only while the cluster is over-populated, and so drain it.
ELIGIBLE_PER_DIM = 3.0


# --- kernel side (JAX) ---


def _distances(u, frames):
    """Squared Mahalanobis distance of ``u`` to each frame, and ``L^-1 (u - mu)``."""
    y = jnp.einsum("kij,kj->ki", frames["ichol"], u - frames["mu"])
    return jnp.where(frames["active"], jnp.sum(y * y, axis=1), jnp.inf), y


def loo_frames(clusters, seed_idx, seed_u):
    """Return the frames with the seed's cluster refit without the seed.

    A frame fitted to a set that contains the chain's seed always has the seed
    inside it, while the reverse swap only covers the frame's true volume, so
    small clusters would be drained. The mean and scatter matrix of the seed's
    cluster get a rank-one downdate and the same shrinkage as on the host (the
    shrinkage target, the pooled covariance, is not downdated). A seed born
    after the last cluster update has label -1 and is in no cluster's
    statistics: the frames are returned as they are.
    """
    ndim = seed_u.shape[0]
    eye = jnp.eye(ndim, dtype=seed_u.dtype)
    label = clusters["labels"][seed_idx]
    k = jnp.maximum(label, 0)
    n_left = clusters["count"][k] - 1.0
    dx = seed_u - clusters["mu"][k]
    mu = clusters["mu"][k] - dx / jnp.maximum(n_left, 1.0)
    scat = clusters["scat"][k] - (n_left + 1.0) / jnp.maximum(n_left, 1.0) * jnp.outer(
        dx, dx
    )
    dof = jnp.maximum(n_left - 1.0, 1.0)
    kappa = jnp.maximum(jnp.trace(clusters["pinv"] @ scat) / (ndim * dof), 1e-6)
    rho = SHRINK * ndim / (n_left - 1.0 + SHRINK * ndim)
    cov = jnp.where(
        n_left > 1.5,
        (1.0 - rho) * scat / dof + rho * kappa * clusters["pooled"],
        clusters["pooled"],
    )
    cov = cov + 10.0 * jnp.finfo(cov.dtype).eps * jnp.trace(cov) / ndim * eye
    chol = jnp.linalg.cholesky(cov)
    refit = (label >= 0) & (n_left > 0.5) & jnp.all(jnp.isfinite(chol))
    chol = jnp.where(refit, chol, eye)
    new = {
        "mu": mu,
        "chol": chol,
        "ichol": jax.scipy.linalg.solve_triangular(chol, eye, lower=True),
        "logdet": jnp.sum(jnp.log(jnp.diagonal(chol))),
    }
    frames = {
        name: clusters[name].at[k].set(jnp.where(refit, value, clusters[name][k]))
        for name, value in new.items()
    }
    # A cluster that cannot be refit (the seed was its only point, or the
    # factorization failed) is switched off for this chain.
    frames["active"] = (
        clusters["active"].at[k].set(clusters["active"][k] & (refit | (label < 0)))
    )
    frames["eligible"] = clusters["eligible"]
    return frames


def swap_step(key, u, frames):
    """Propose the affine swap of ``u``; return ``(u_new, ok, is_swap)``.

    ``is_swap`` says whether this chain step is a swap at all (probability
    ``SWAP_PROB``, independent of the state, so the chain is a fixed mixture
    of the rwalk and swap kernels). ``ok`` holds the tests that need no
    likelihood call; the caller still requires ``u_new`` to be in the cube and
    at or above the likelihood threshold.
    """
    r_type, r_pick, r_accept = random.uniform(key, (3,), dtype=u.dtype)
    dist, y = _distances(u, frames)
    a = jnp.argmin(dist)  # the cluster of u: its nearest frame
    can_swap = frames["active"] & frames["eligible"]
    others = can_swap & (jnp.arange(dist.shape[0]) != a)
    nothers = jnp.sum(others)
    # b is uniform among the other eligible clusters. The reverse move starts
    # in b and picks a among as many others, so the selection probabilities
    # cancel and only the Jacobian det L_b / det L_a is left.
    pick = jnp.minimum(jnp.floor(r_pick * nothers).astype(nothers.dtype), nothers - 1)
    b = jnp.argmax(jnp.cumsum(others) == pick + 1)
    u_new = frames["mu"][b] + frames["chol"][b] @ y[a]
    log_jacobian = frames["logdet"][b] - frames["logdet"][a]
    ok = (
        can_swap[a]
        & (nothers > 0)
        # u_new must belong to b, so that the reverse move proposes u.
        & (jnp.argmin(_distances(u_new, frames)[0]) == b)
        & (jnp.log(r_accept) < log_jacobian)
    )
    return u_new, ok, r_type < SWAP_PROB


# --- host side (numpy) ---


def _ridge(cov):
    ndim = len(cov)
    return cov + 1e-12 * max(np.trace(cov) / ndim, 1e-300) * np.eye(ndim)


def _logsumexp(x):
    return float(np.logaddexp.reduce(x))  # -inf for no points


def _fit(u, labels, k):
    """Fit ``k`` clusters to their points (``labels == c``; -1 is left out).

    Returns each cluster's mean, scatter matrix, count, and the Cholesky
    factor of its covariance shrunk toward the scale-matched pooled
    within-cluster covariance.
    """
    ndim = u.shape[1]
    mu, scat, count = np.zeros((k, ndim)), np.zeros((k, ndim, ndim)), np.zeros(k)
    for c in range(k):
        members = u[labels == c]
        count[c] = len(members)
        if len(members):
            mu[c] = members.mean(0)
            scat[c] = (members - mu[c]).T @ (members - mu[c])
    pooled = _ridge(scat.sum(0) / max(count.sum() - k, 1))
    pinv = np.linalg.inv(pooled)
    chol = np.empty_like(scat)
    for c in range(k):
        cov = pooled
        if count[c] > 1:
            dof = count[c] - 1
            kappa = max(np.trace(pinv @ scat[c]) / (ndim * dof), 1e-6)
            rho = SHRINK * ndim / (dof + SHRINK * ndim)
            cov = _ridge((1 - rho) * scat[c] / dof + rho * kappa * pooled)
        chol[c] = np.linalg.cholesky(cov)
    logdet = np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(1)
    return {
        "mu": mu,
        "scat": scat,
        "count": count,
        "chol": chol,
        "logdet": logdet,
        "pooled": pooled,
        "pinv": pinv,
    }


def _assign(fit, x):
    """Return the best cluster of each row of ``x`` (hard-EM classification)."""
    score = np.full((len(fit["count"]), len(x)), np.inf)
    total = fit["count"].sum()
    for c, n in enumerate(fit["count"]):
        if n:
            z = np.linalg.solve(fit["chol"][c], (x - fit["mu"][c]).T)
            score[c] = (z * z).sum(0) + 2 * fit["logdet"][c] - 2 * np.log(n / total)
    return np.argmin(score, axis=0)


def _refine(u, labels, k, iters):
    """Hard EM: refit the clusters and reassign every point, ``iters`` times."""
    for _ in range(iters):
        new = _assign(_fit(u, labels, k), u)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def _fisher(x, part):
    """Fisher separation ``J = dm^T Sw^-1 dm`` of a two-way partition."""
    a, b = x[~part], x[part]
    dm = b.mean(0) - a.mean(0)
    a, b = a - a.mean(0), b - b.mean(0)
    within = _ridge((a.T @ a + b.T @ b) / max(len(x) - 2, 1))
    return float(dm @ np.linalg.solve(within, dm))


def _split(x):
    """Try to split one cluster; return ``(mask of the new part or None, J)``.

    Two-means in whitened coordinates from a farthest-point start, refined by
    hard EM. Plain k-means would rather halve the bulk than isolate a small
    mode, and without the EM step a small mode is often missed.
    """
    n, ndim = x.shape
    if n < max(2 * MIN_SPLIT_POINTS, 4 * ndim):
        return None, 0.0
    centered = x - x.mean(0)
    chol = np.linalg.cholesky(_ridge(centered.T @ centered / (n - 1)))
    y = np.linalg.solve(chol, centered.T).T
    far = y[np.argmax((y * y).sum(1))]
    far2 = y[np.argmax(((y - far) ** 2).sum(1))]
    best, best_j = None, 0.0
    for c0, c1 in ((np.zeros(ndim), far), (far, far2)):
        part = None
        for _ in range(30):
            new = ((y - c1) ** 2).sum(1) < ((y - c0) ** 2).sum(1)
            if new.all() or not new.any():
                part = None
                break
            if part is not None and np.array_equal(new, part):
                break
            part = new
            c0, c1 = y[~part].mean(0), y[part].mean(0)
        if part is None:
            continue
        part = _refine(y, part.astype(int), 2, 8) == 1
        if min(part.sum(), n - part.sum()) < MIN_SPLIT_POINTS:
            continue
        j = _fisher(y, part)
        if j > best_j:
            best, best_j = part, j
    if best_j < max(SPLIT_J_PER_DIM * (ndim + 2), SPLIT_J_ABS):
        return None, best_j
    return best, best_j


class ClusterTracker:
    """Clusters of the live points, tracked across blocks (host side, numpy).

    Nothing here touches the PRNG stream. ``log`` is JSON-serializable and,
    with ``arrays()``, restores the tracker from a checkpoint.
    """

    def __init__(self, nlive, arrays=None, log=None):
        self.every = min(RECLUSTER_ITERS, max(int(nlive) // 4, 1))
        self.labels = self.u = self.fit = self._frames = None
        self.eligible = np.zeros(1, dtype=bool)
        self.log = log or {
            "iteration": 0,  # of the last update
            "ids": [0],  # persistent id of each current cluster
            "next_id": 1,
            "counts": [],  # [iteration, number of clusters], at changes only
            "iters": [],  # iteration of every update
            "modes": {},  # per cluster id: populations and evidence per update
            "swap": [0, 0],  # accepted, proposed
        }
        if arrays is not None:
            self.labels = np.asarray(arrays["labels"])
            self.u = np.asarray(arrays["u"])
            self._refit()

    def arrays(self):
        """Return the arrays a checkpoint needs besides ``log``."""
        return None if self.labels is None else {"labels": self.labels, "u": self.u}

    def frames(self, live_u, iteration, dead_u, dead_logwt):
        """Return the cluster frames for the next block, or ``None``.

        Re-clusters when due. ``None`` means fewer than two eligible clusters:
        the block then runs the plain rwalk kernel. Points replaced since the
        last update get label -1.
        """
        last = self.log["iteration"]
        if self.labels is None or iteration - last >= self.every:
            self._update(
                np.asarray(live_u, dtype=float),
                int(iteration),
                dead_u[last:iteration],
                dead_logwt[last:iteration],
            )
        if self._frames is None:
            return None
        live_u = np.asarray(live_u)
        frames = {
            name: value.astype(live_u.dtype) if value.dtype.kind == "f" else value
            for name, value in self._frames.items()
        }
        unchanged = (live_u == self.u).all(1)
        frames["labels"] = np.where(unchanged, self.labels, -1).astype(np.int32)
        return frames

    def _update(self, u, iteration, dead_u, dead_logwt):
        log = self.log
        if self.labels is not None:
            self._account(dead_u, dead_logwt)
        try:
            labels = self._track(u)
        except np.linalg.LinAlgError:  # degenerate live set: back to one cluster
            labels = np.zeros(len(u), dtype=int)
            del log["ids"][1:]
        self.labels, self.u = labels, u
        log["iteration"] = iteration
        self._refit()
        k = len(log["ids"])
        if not log["counts"] or log["counts"][-1][1] != k:
            log["counts"].append([iteration, k])
        log["iters"].append(iteration)
        population = np.bincount(labels, minlength=k)
        for c, cid in enumerate(log["ids"]):
            mode = log["modes"].setdefault(
                str(cid),
                {"start": len(log["iters"]) - 1, "n": [], "swap": [], "logz": []},
            )
            mode["n"].append(int(population[c]))
            mode["swap"].append(bool(self.eligible[c] and self._frames is not None))

    def _track(self, u):
        """Relabel the live points: warm-started hard EM, merges, then splits."""
        n, ndim = u.shape
        ids = self.log["ids"]
        threshold = max(SPLIT_J_PER_DIM * (ndim + 2), SPLIT_J_ABS)
        if self.labels is None or len(ids) == 1:
            labels = np.zeros(n, dtype=int)
        else:
            # Points replaced since the last update are labelled by the EM pass.
            labels = np.where((u == self.u).all(1), self.labels, -1)
            labels = self._compact(_refine(u, labels, len(ids), 2))
            merged = True
            while merged and len(ids) > 1:
                merged = False
                for a, b in itertools.combinations(range(len(ids)), 2):
                    pair = (labels == a) | (labels == b)
                    if _fisher(u[pair], labels[pair] == b) < MERGE_FRACTION * threshold:
                        labels[labels == b] = a
                        labels = self._compact(labels)
                        merged = True
                        break
        tried = set()
        while len(ids) < MAX_CLUSTERS:
            best = None
            for c in set(range(len(ids))) - tried:
                members = np.flatnonzero(labels == c)
                part, j = _split(u[members])
                if part is None:
                    tried.add(c)
                elif best is None or j > best[0]:
                    best = (j, c, members[part])
            if best is None:
                break
            labels[best[2]] = len(ids)
            ids.append(self.log["next_id"])
            self.log["next_id"] += 1
        return labels

    def _compact(self, labels):
        """Drop empty clusters; the others keep their order, and so their ids."""
        ids = self.log["ids"]
        kept = np.flatnonzero(np.bincount(labels, minlength=len(ids)))
        ids[:] = [ids[c] for c in kept]
        return np.searchsorted(kept, labels)

    def _refit(self):
        """Fit the frames; they go to the kernel once two clusters are eligible."""
        k = len(self.log["ids"])
        self.fit = self._frames = None
        self.eligible = np.zeros(k, dtype=bool)
        if k < 2:
            return
        n, ndim = self.u.shape
        fit = self.fit = _fit(self.u, self.labels, k)
        share = np.exp(fit["logdet"] - _logsumexp(fit["logdet"]))  # of the volume
        eligible = (n * share >= ELIGIBLE_PER_DIM * ndim) & (fit["count"] > ndim)
        eligible[np.argmax(fit["count"])] = True  # the bulk always takes part
        self.eligible = eligible
        if eligible.sum() < 2:
            return

        def pad(x):
            out = np.zeros((MAX_CLUSTERS,) + x.shape[1:], dtype=x.dtype)
            out[:k] = x
            return out

        chol = np.tile(np.eye(ndim), (MAX_CLUSTERS, 1, 1))
        chol[:k] = fit["chol"]
        self._frames = {
            "mu": pad(fit["mu"]),
            "scat": pad(fit["scat"]),
            "count": pad(fit["count"]),
            "chol": chol,
            "ichol": np.linalg.inv(chol),
            "logdet": pad(fit["logdet"]),
            "active": pad(np.ones(k, dtype=bool)),
            "eligible": pad(eligible),
            "pooled": fit["pooled"],
            "pinv": fit["pinv"],
        }

    def _account(self, dead_u, dead_logwt):
        """Credit the evidence of the points that died since the last update."""
        which = np.zeros(len(dead_u), dtype=int)
        if self.fit is not None and len(dead_u):
            which = _assign(self.fit, dead_u)
        for c, cid in enumerate(self.log["ids"]):
            self.log["modes"][str(cid)]["logz"].append(
                _logsumexp(dead_logwt[which == c])
            )

    def summary(self, dead_u, dead_logwt, live_u, live_logwt):
        """Return the cluster telemetry for ``result.metadata``.

        ``cluster_modes`` lists every cluster that ever shared the live set
        with another one: its posterior ``mass``, its smallest population
        between its detection and its posterior median, and ``urn_logit_sd``,
        the scatter of ``logit(mass)`` that the random walk of its population
        would cause without the swap, from the run's own populations over the
        same stretch (section 7 of the multimodal study). ``swap_fraction`` is
        the part of that stretch during which the cluster could swap; near 1
        the actual scatter is several times smaller than ``urn_logit_sd``.
        """
        log, nlive = self.log, len(live_u)
        out = {
            "cluster_count_history": [list(pair) for pair in log["counts"]],
            "cluster_min_population": None,
            "cluster_swap_accepts": int(log["swap"][0]),
            "cluster_swap_proposals": int(log["swap"][1]),
            "cluster_modes": [],
        }
        if not log["iters"]:  # never clustered (e.g. resumed at the end)
            return out
        edges = log["iters"] + [len(dead_logwt)]
        logz = _logsumexp(np.concatenate([dead_logwt, live_logwt]))
        # Mass per update interval: the last one also holds the live points.
        tail_logwt = np.concatenate([dead_logwt[edges[-2] :], live_logwt])
        total = [_logsumexp(dead_logwt[a:b]) for a, b in itertools.pairwise(edges)]
        total = np.exp(np.array(total[:-1] + [_logsumexp(tail_logwt)]) - logz)
        tail = np.zeros(len(tail_logwt), dtype=int)
        if self.fit is not None:
            tail = _assign(self.fit, np.concatenate([dead_u[edges[-2] :], live_u]))
        tail_logz = {cid: _logsumexp(tail_logwt[tail == c]) for c, cid in
                     enumerate(log["ids"])}
        modes = []
        for cid, mode in log["modes"].items():
            n = np.array(mode["n"], dtype=float)
            alive = len(mode["logz"]) < len(n)  # still a cluster at the end
            own = mode["logz"] + ([tail_logz[int(cid)]] if alive else [])
            own = np.exp(np.array(own) - logz)
            shared = (n > 0) & (n < nlive)  # another cluster exists
            if not shared.any():
                continue
            first = int(np.argmax(shared))
            stop = first + 1 + int(np.searchsorted(np.cumsum(own[first:]),
                                                   0.5 * own[first:].sum()))
            other = total[mode["start"] :][: len(n)] - own
            stop_other = int(np.searchsorted(np.cumsum(other), 0.5 * other.sum())) + 1
            g = np.clip(n / nlive, 0.5 / nlive, 1 - 0.5 / nlive)
            steps = np.diff(edges)[mode["start"] :][: len(n)]
            rate = np.where(shared, 2.0 * steps / nlive**2, 0.0)
            variance = (
                1.0 / n[first]
                + 1.0 / (nlive - n[first])
                + (rate * (1 - g) / g)[first:stop].sum()
                + (rate * g / (1 - g))[first:stop_other].sum()
            )
            modes.append(
                {
                    "id": int(cid),
                    "first_iteration": int(edges[mode["start"] + first]),
                    "mass": float(own.sum()),
                    "min_population": int(n[first:stop].min()),
                    "urn_logit_sd": float(np.sqrt(variance)),
                    "swap_fraction": float(np.mean(mode["swap"][first:stop])),
                }
            )
        out["cluster_modes"] = modes
        out["cluster_min_population"] = min(
            (m["min_population"] for m in modes), default=None
        )
        return out
