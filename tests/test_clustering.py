"""On-device clustering (:func:`tinyns.modes.recluster`) on live-point snapshots.

The gates of the split test: at most 1% false splits of unimodal snapshots
(uniform in an ellipsoid, Gaussian draws, a banana) at d = 4 to 32, and at
least 80% detection of an 18- to 50-point minor mode at d = 18. The
snapshots are uniform in the iso-likelihood ellipsoids of the bench's
``sepM`` two-mode target (as in the multimodality prototype). Each snapshot
is clustered twice, the second time warm-started from the first, as a run
does at successive reclusters. ``pytest -m slow`` runs the full grids.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bench.targets import build_two_mode

from tinyns import modes


@jax.jit
def cluster_twice(u):
    """Cluster a batch of snapshots ``(B, m, d)`` from scratch, then again
    from the first labels."""
    labels, _ = modes.recluster_lanes(u, jnp.zeros(u.shape[:2], jnp.int32))
    return modes.recluster_lanes(u, labels)[0]


def nclusters(labels) -> np.ndarray:
    return np.array([len(np.unique(row)) for row in np.asarray(labels)])


def unit_ball(rng, n, d):
    z = rng.standard_normal((n, d))
    return z / np.linalg.norm(z, axis=1)[:, None] * rng.uniform(size=(n, 1)) ** (1 / d)


def unimodal(shape, rng, n, d):
    """``n`` points of one mode, scaled into the unit cube."""
    mix = np.eye(d) + rng.standard_normal((d, d)) / math.sqrt(d)
    if shape == "ellipsoid":
        y = unit_ball(rng, n, d) @ mix.T
    elif shape == "gauss":
        y = rng.standard_normal((n, d)) @ mix.T
    elif shape == "banana":  # uniform in a curved ellipsoid (volume preserving)
        y = unit_ball(rng, n, d) * 3.0
        y[:, 1] += 0.3 * y[:, 0] ** 2
    else:
        raise ValueError(shape)
    return 0.5 + 0.01 * y


def two_mode_snapshot(rng, d, n, n_minor, r2_main):
    """Uniform in the main mode's ellipsoid ``r^2 <= r2_main`` and the minor
    mode's ellipsoid at the same likelihood (``sepM``; isolation at
    ``r^2 = 400`` for d >= 10)."""
    t = build_two_mode(d, 0.06, "sep", "M")
    (s1, s2), (w1, w2) = t["covs"], t["weights"]
    a1 = math.log(w1) - 0.5 * np.linalg.slogdet(s1)[1]
    a2 = math.log(w2) - 0.5 * np.linalg.slogdet(s2)[1]
    r2_minor = r2_main + 2 * (a2 - a1)
    x1 = t["means"][0] + math.sqrt(r2_main) * unit_ball(rng, n - n_minor, d) @ (
        np.linalg.cholesky(s1).T
    )
    x2 = t["means"][1] + math.sqrt(r2_minor) * unit_ball(rng, n_minor, d) @ (
        np.linalg.cholesky(s2).T
    )
    minor = np.r_[np.zeros(n - n_minor, bool), np.ones(n_minor, bool)]
    return np.vstack([x1, x2]), minor


def detected(labels, minor, tolerance=0.01) -> bool:
    """Two clusters, and at most ``tolerance`` of the points on the wrong side."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2:
        return False
    wrong = sum(
        min(np.sum(minor[labels == c]), np.sum(~minor[labels == c]))
        for c in np.unique(labels)
    )
    return wrong <= tolerance * len(labels)


def false_split_rate(shape, d, n, count, seed):
    rng = np.random.default_rng(seed)
    x = np.stack([unimodal(shape, rng, n, d) for _ in range(count)])
    return np.mean(nclusters(cluster_twice(jnp.asarray(x))) > 1)


def detection_rate(d, n, n_minor, r2_main, count, seed):
    rng = np.random.default_rng(seed)
    snaps = [two_mode_snapshot(rng, d, n, n_minor, r2_main) for _ in range(count)]
    labels = cluster_twice(jnp.asarray(np.stack([s[0] for s in snaps])))
    return np.mean([detected(lab, s[1]) for lab, s in zip(labels, snaps, strict=True)])


@pytest.mark.parametrize("d", [4, 18])
def test_unimodal_snapshots_do_not_split(d) -> None:
    for shape in ("ellipsoid", "gauss", "banana"):
        assert false_split_rate(shape, d, 500, 10, seed=d) == 0.0, shape


def test_minor_mode_is_detected_at_18d() -> None:
    assert detection_rate(18, 500, 25, 60.0, 10, seed=0) >= 0.8


def test_far_tail_point_does_not_hide_a_comparable_mode() -> None:
    """A 25% mode next to a heavy-tailed bulk: the farthest point is a tail
    point of the bulk; a cut along an independent-component direction still
    finds the mode (a few t tails of the bulk go with it)."""
    rng = np.random.default_rng(5)
    bulk = rng.standard_t(3, size=(375, 3)) * 0.02 + 0.3
    minor = rng.standard_normal((125, 3)) * 0.02 + np.array([0.45, 0.3, 0.3])
    labels = cluster_twice(jnp.asarray(np.vstack([bulk, minor])[None]))[0]
    truth = np.r_[np.zeros(375, bool), np.ones(125, bool)]
    assert detected(labels, truth, tolerance=0.03)


def test_labels_and_statistics_agree() -> None:
    rng = np.random.default_rng(6)
    x, _ = two_mode_snapshot(rng, 10, 300, 40, 60.0)
    labels, st = modes.recluster(jnp.asarray(x), jnp.zeros(300, jnp.int32))
    ref = modes.stats(jnp.asarray(x), labels, modes.C_MAX)
    for a, b in zip(st, ref, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-9)
    assert sorted(np.unique(labels).tolist()) == [0, 1]


@pytest.mark.slow
def test_false_split_rate_grid() -> None:
    """The PR 3 gate: at most 1% false splits over all unimodal snapshots."""
    rates = {}
    for n in (500, 2000):
        for d in (4, 10, 18, 32):
            for shape in ("ellipsoid", "gauss", "banana"):
                rates[n, d, shape] = false_split_rate(shape, d, n, 100, seed=d + n)
    for key, rate in rates.items():
        print("false splits", key, rate)
    assert np.mean(list(rates.values())) <= 0.01
    assert max(rates.values()) <= 0.03


@pytest.mark.slow
def test_detection_grid_18d() -> None:
    """The PR 3 gate: at least 80% detection of 18- to 50-point minor modes at
    d = 18, past the pinch-off of the two contours (r^2 = 120 and 60)."""
    for r2_main in (120.0, 60.0):
        for n_minor in (18, 30, 50):
            rate = detection_rate(18, 500, n_minor, r2_main, 100, seed=n_minor)
            print(f"detection r2={r2_main} n_minor={n_minor}: {rate:.2f}")
            assert rate >= 0.8
