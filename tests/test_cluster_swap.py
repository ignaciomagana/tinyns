"""The affine swap move between tracked clusters (``cluster_swap``)."""

import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from tests.helpers import run_ns

from tinyns import NestedSampler, checkpoint, clusters, core
from tinyns.callables import _callable_specs
from tinyns.core import _chain_kernel

# --- the swap kernel leaves the constrained prior invariant ---

# Two disjoint ellipses; the constrained region is their union.
CENTERS = np.array([[0.3, 0.5], [0.75, 0.45]])
AXES = np.array([[0.14, 0.07], [0.07, 0.05]])
ANGLES = np.array([0.4, -0.9])


def _rotations():
    c, s = np.cos(ANGLES), np.sin(ANGLES)
    pairs = zip(c, s, strict=True)
    return np.stack([np.array([[ci, -si], [si, ci]]) for ci, si in pairs])


def _ellipse_radius2(u):
    """Squared normalized radius of ``u`` in each ellipse, shape (..., 2)."""
    rot = jnp.asarray(_rotations())
    x = jnp.einsum("kji,...kj->...ki", rot, u[..., None, :] - jnp.asarray(CENTERS))
    return jnp.sum((x / jnp.asarray(AXES)) ** 2, axis=-1)


def _region_logl(u):
    return jnp.where(jnp.min(_ellipse_radius2(u)) <= 1.0, 0.0, -1.0)


def _identity(u):
    return u


def _uniform_in(k, n, rng):
    r = np.sqrt(rng.uniform(size=n))
    phi = rng.uniform(0, 2 * np.pi, size=n)
    x = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1) * AXES[k]
    return CENTERS[k] + x @ _rotations()[k].T


def _frames(nlive):
    """Deliberately imperfect frames (inflated, rotated); none holds the walker."""
    chol = np.tile(np.eye(2), (clusters.MAX_CLUSTERS, 1, 1))
    for k, (scale, tilt) in enumerate(((1.2, 0.2), (0.9, -0.3))):
        c, s = np.cos(ANGLES[k] + tilt), np.sin(ANGLES[k] + tilt)
        rot = np.array([[c, -s], [s, c]])
        cov = scale**2 * rot @ np.diag(AXES[k] ** 2 / 4) @ rot.T
        chol[k] = np.linalg.cholesky(cov)
    pad = clusters.MAX_CLUSTERS - 2
    return {
        "mu": np.vstack([CENTERS + 0.01, np.zeros((pad, 2))]),
        "scat": np.zeros((clusters.MAX_CLUSTERS, 2, 2)),
        "count": np.zeros(clusters.MAX_CLUSTERS),
        "chol": chol,
        "ichol": np.linalg.inv(chol),
        "logdet": np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(1),
        "active": np.arange(clusters.MAX_CLUSTERS) < 2,
        "eligible": np.arange(clusters.MAX_CLUSTERS) < 2,
        "pooled": np.eye(2),
        "pinv": np.eye(2),
        "labels": np.full(nlive, -1, dtype=np.int32),
    }


def test_swap_kernel_relaxes_to_the_volume_share() -> None:
    """Walkers started in one ellipse end up spread by volume, uniform within.

    Each walker is the only live point above the threshold, so it is the seed
    of every chain; a fixed reference set supplies the rwalk covariance. The
    rwalk alone cannot cross the gap, so the populations are set by the swap.
    """
    rng = np.random.default_rng(0)
    nwalkers, nsteps = 2000, 12
    reference = np.vstack([_uniform_in(0, 150, rng), _uniform_in(1, 50, rng)])
    nlive = len(reference) + 1
    kernel = _chain_kernel(
        *_callable_specs(_region_logl, _identity, 2),
        2,
        10,
        1,
        True,
    )
    frames = jax.tree_util.tree_map(jnp.asarray, _frames(nlive))
    live_logl = jnp.concatenate([jnp.zeros(1), -jnp.ones(nlive - 1)])

    def move(key, walker, reference, frames):
        live_u = jnp.concatenate([walker[None, :], reference])
        out = kernel(key, -0.5, live_u, live_logl, 0.5, jnp.int32(1), frames)
        return out[1], out[8]

    move = jax.jit(jax.vmap(move, in_axes=(0, 0, None, None)))
    walkers = jnp.asarray(_uniform_in(0, nwalkers, rng))
    keys = random.split(random.PRNGKey(1), nsteps)
    swaps = np.zeros(2, dtype=int)
    for key in keys:
        walkers, counts = move(
            random.split(key, nwalkers), walkers, jnp.asarray(reference), frames
        )
        swaps += np.asarray(counts).sum(0)

    r2 = np.asarray(_ellipse_radius2(walkers))
    assert (r2.min(axis=1) <= 1.0).all()
    in_b = r2[:, 1] <= 1.0
    share_b = AXES[1].prod() / AXES.prod(axis=1).sum()
    sd = math.sqrt(share_b * (1 - share_b) / nwalkers)
    assert swaps[0] > 0.02 * swaps[1] > 0
    assert abs(in_b.mean() - share_b) < 4 * sd
    # Uniform within each ellipse: half of the area lies inside radius^2 1/2.
    for k, members in enumerate((~in_b, in_b)):
        inner = (r2[members, k] <= 0.5).mean()
        assert abs(inner - 0.5) < 4 * math.sqrt(0.25 / members.sum())


def test_leave_one_out_frame_matches_a_refit_without_the_seed() -> None:
    rng = np.random.default_rng(2)
    u = np.vstack([_uniform_in(0, 40, rng), _uniform_in(1, 20, rng)])
    labels = np.repeat([0, 1], [40, 20])
    fit = clusters._fit(u, labels, 2)
    frames = {
        "mu": fit["mu"],
        "scat": fit["scat"],
        "count": fit["count"],
        "chol": fit["chol"],
        "ichol": np.linalg.inv(fit["chol"]),
        "logdet": fit["logdet"],
        "active": np.ones(2, dtype=bool),
        "eligible": np.ones(2, dtype=bool),
        "pooled": fit["pooled"],
        "pinv": fit["pinv"],
        "labels": labels.astype(np.int32),
    }
    seed = 45  # in cluster 1
    out = clusters.loo_frames(
        jax.tree_util.tree_map(jnp.asarray, frames), seed, jnp.asarray(u[seed])
    )
    rest = np.delete(u[labels == 1], seed - 40, axis=0)
    n, ndim = len(rest), 2
    scat = (rest - rest.mean(0)).T @ (rest - rest.mean(0))
    kappa = np.trace(fit["pinv"] @ scat) / (ndim * (n - 1))
    rho = clusters.SHRINK * ndim / (n - 1 + clusters.SHRINK * ndim)
    cov = (1 - rho) * scat / (n - 1) + rho * kappa * fit["pooled"]
    np.testing.assert_allclose(out["mu"][1], rest.mean(0), rtol=1e-6)
    np.testing.assert_allclose(out["chol"][1], np.linalg.cholesky(cov), rtol=1e-4)
    np.testing.assert_allclose(out["mu"][0], fit["mu"][0], rtol=1e-6)  # untouched


# --- unimodal runs are bit-identical to v0.2.4 ---

COV4 = 0.01 * (0.6 ** np.abs(np.subtract.outer(np.arange(4), np.arange(4))))
P4 = jnp.asarray(np.linalg.inv(COV4))


def gauss4(theta):
    x = theta - 0.4
    return -0.5 * x @ P4 @ x


def banana3(theta):
    x = (theta - 0.5) / 0.1
    return -0.5 * (x[0] ** 2 + (x[1] - 0.8 * x[0] ** 2) ** 2 + x[2] ** 2)


def plain2(theta):
    return -0.5 * jnp.sum(((theta - 0.2) / 0.1) ** 2)


# Fixed-seed fingerprints on CPU with jax 0.4.34, keyed by JAX_ENABLE_X64:
# (logz.hex(), ncall, sha256[:16] of samples, samples_u and logl). x64 off: tinyns
# v0.2.4 (94cc24a); x64 on: recorded on the v0.3 loop branch, which
# tools/ab_bitwise.py shows bit-identical to main on the same machine.
V024_FINGERPRINTS = {
    "gauss4": (
        (gauss4, 4, 100, {}),
        {
            False: (
                "-0x1.7ed5a40000000p+2",
                18499,
                "0d3ac595642d43dc",
                "0d3ac595642d43dc",
                "527c11c43b505dc5",
            ),
            True: (
                "-0x1.8db480534bbfcp+2",
                19259,
                "d7db932053d51877",
                "d7db932053d51877",
                "74431bed060b3c6f",
            ),
        },
    ),
    "banana3": (
        (banana3, 3, 80, {}),
        {
            False: (
                "-0x1.f4de840000000p+1",
                12064,
                "2cd71abefd88accc",
                "2cd71abefd88accc",
                "71e38f42e35d51e5",
            ),
            True: (
                "-0x1.e9fb15f2c48e8p+1",
                11930,
                "7c2d6ac03eaeb775",
                "7c2d6ac03eaeb775",
                "0c70c57b59b60e77",
            ),
        },
    ),
    "plain2_b8": (
        (plain2, 2, 40, {"block_size": 8}),
        {
            False: (
                "-0x1.7c3fac0000000p+1",
                4372,
                "a6b37cb3c0686527",
                "a6b37cb3c0686527",
                "cd01b397cd9ac307",
            ),
            True: (
                "-0x1.58d2919c035a8p+1",
                4349,
                "2c756a59110a85cf",
                "2c756a59110a85cf",
                "b0305a370c1d94e4",
            ),
        },
    ),
}

def fingerprint(result):
    def digest(x):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]

    return (
        float(result.logz).hex(),
        result.ncall,
        digest(result.samples),
        digest(result.samples_u),
        digest(result.logl),
    )


@pytest.mark.skipif(
    jax.__version__ != "0.4.34" or jax.default_backend() != "cpu",
    reason="fingerprints were recorded with jax 0.4.34 on CPU",
)
@pytest.mark.parametrize("case", sorted(V024_FINGERPRINTS))
def test_unimodal_runs_match_v024_bit_for_bit(case, monkeypatch) -> None:
    (loglike, ndim, nlive, kwargs), expected = V024_FINGERPRINTS[case]
    swap_flags = []
    step = core.step

    def spy(*args, extras=None, **kw):
        swap_flags.append(extras is not None)
        return step(*args, extras=extras, **kw)

    monkeypatch.setattr(core, "step", spy)
    result = run_ns(
        random.PRNGKey(11), loglike, lambda u: u, ndim, nlive, **kwargs
    )
    assert result.metadata["cluster_swap"] is True
    assert set(swap_flags) == {False}  # the swap kernel is never built
    assert fingerprint(result) == expected[bool(jax.config.jax_enable_x64)]


# --- two separated modes ---

SIGMAS = (0.03, 0.02)
CENTERS3 = (0.3, 0.7)
W_MINOR = 0.25


def two_modes(theta):
    def log_normal(center, sigma):
        z = (theta - center) / sigma
        return -0.5 * jnp.sum(z * z) - 3 * jnp.log(sigma * math.sqrt(2 * math.pi))

    return jnp.logaddexp(
        math.log(1 - W_MINOR) + log_normal(CENTERS3[0], SIGMAS[0]),
        math.log(W_MINOR) + log_normal(CENTERS3[1], SIGMAS[1]),
    )


def test_two_mode_run_finds_and_weighs_the_minor_mode() -> None:
    result = NestedSampler(two_modes, lambda u: u, 3, nlive=200).run(5)
    md = result.metadata
    assert md["cluster_swap_accepts"] > 0
    assert max(k for _, k in md["cluster_count_history"]) >= 2
    weights = np.asarray(result.weights())
    minor = np.asarray(result.samples)[:, 0] > 0.5
    mass = weights[minor].sum() / weights.sum()
    assert 0.15 < mass < 0.37  # truth 0.25
    assert abs(result.logz) < 4 * result.logzerr + 0.1  # truth 0
    modes = result.modes()
    assert len(modes) == 2
    assert sum(m["mass"] for m in modes) == pytest.approx(1.0)
    assert modes[1]["mass"] == pytest.approx(mass, abs=0.02)
    assert all(0 < m["urn_sd"] < 1 for m in modes)
    assert not any(m["unresolved"] for m in modes)
    assert modes[1]["min_live"] >= 3 * 3


def narrow_minor4(theta):
    """4-D: a 25% mode half as wide as the main one, so 1/16 of its volume."""

    def log_normal(center, sigma):
        z = (theta - center) / sigma
        return -0.5 * jnp.sum(z * z) - 4 * jnp.log(sigma * math.sqrt(2 * math.pi))

    return jnp.logaddexp(
        math.log(0.75) + log_normal(0.35, 0.03), math.log(0.25) + log_normal(0.7, 0.015)
    )


def test_modes_flag_an_under_populated_mode() -> None:
    """With 120 live points the narrow mode holds a handful when it pinches off
    (3 * ndim = 12 are needed). A run either loses it (no trace) or flags it."""
    flagged = 0
    for seed in range(4):
        result = NestedSampler(narrow_minor4, lambda u: u, 4, nlive=120).run(seed)
        modes = result.modes()
        assert sum(m["mass"] for m in modes) == pytest.approx(1.0)
        if len(modes) == 1:  # the mode was lost
            assert modes[0]["unresolved"] is False
            continue
        major, minor = modes
        assert not major["unresolved"]
        assert minor["unresolved"] is (minor["min_live"] < 12)
        assert 0.0 < minor["urn_sd"] < 5.0
        assert minor["urn_sd"] == pytest.approx(major["urn_sd"])  # symmetric
        if minor["unresolved"]:
            flagged += 1
            assert "unresolved: raise nlive" in result.summary()
            assert any("unresolved" in w for w in result.diagnostics()["warnings"])
    assert flagged >= 1


def test_unimodal_run_reports_one_mode() -> None:
    result = run_ns(3, gauss4, lambda u: u, 4, 100)
    assert result.modes() == [
        {
            "mass": 1.0,
            "urn_sd": 0.0,
            "min_live": 100,
            "isolation_iteration": 0,
            "unresolved": False,
        }
    ]
    assert "logit sd" not in result.summary()


def test_two_mode_resume_with_swap_matches_uninterrupted(tmp_path) -> None:
    path = tmp_path / "swap.checkpoint.npz"
    sampler = NestedSampler(two_modes, lambda u: u, 3, nlive=200)
    full = sampler.run(5, maxiter=1600, dlogz=0.0)
    sampler.run(5, maxiter=1024, dlogz=0.0, checkpoint_path=path)
    ckpt = checkpoint.load(path)
    assert ckpt.config["cluster_swap"] is True
    assert len(ckpt.ext["clusters"]["log"]["ids"]) == 2  # resumed with the swap on
    resumed = sampler.resume(path, maxiter=1600, dlogz=0.0)

    assert resumed.metadata["cluster_swap_accepts"] > 0
    assert resumed.metadata["cluster_count_history"] == (
        full.metadata["cluster_count_history"]
    )
    assert resumed.modes() == full.modes()
    assert resumed.logz == full.logz
    np.testing.assert_array_equal(resumed.samples_u, full.samples_u)
    np.testing.assert_array_equal(resumed.logl, full.logl)
    assert resumed.ncall == full.ncall


# --- API ---


def test_cluster_swap_needs_a_single_replacement_chain() -> None:
    with pytest.raises(NotImplementedError, match="cluster_swap"):
        NestedSampler(
            plain2, lambda u: u, 2, nlive=20, cluster_swap=True, replacement_chains=2
        )
    result = run_ns(
        0, plain2, lambda u: u, 2, 20, maxiter=40, replacement_chains=2
    )
    assert result.metadata["cluster_swap"] is False
    assert not any(key.startswith("cluster_swap_") for key in result.metadata)


def test_checkpoint_without_cluster_swap_is_rejected(tmp_path) -> None:
    import json

    path = tmp_path / "old.checkpoint.npz"
    NestedSampler(plain2, lambda u: u, 2, nlive=20, cluster_swap=False).run(
        3, maxiter=64, dlogz=0.0, checkpoint_path=path
    )
    with np.load(path) as data:
        fields = dict(data)
    config = json.loads(str(fields["config_json"]))
    assert config.pop("cluster_swap") is False
    fields["config_json"] = np.asarray(json.dumps(config))
    np.savez(path, **fields)

    for cluster_swap in (None, False):
        with pytest.raises(ValueError, match="cluster_swap"):
            NestedSampler(
                plain2, lambda u: u, 2, nlive=20, cluster_swap=cluster_swap
            ).resume(path, maxiter=96)
