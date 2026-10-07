"""Benchmark targets with known evidence and, for mixtures, exact mode masses.

Every target has a uniform box prior ``[lo, hi]^d``, so the prior transform is
affine and is available both as JAX (``prior_transform``) and numpy
(``prior_transform_np``). ``loglike`` is a JAX function of one physical point.
``logz`` is the truth the summaries compare against; ``logz_source`` says where
it comes from:

``analytic``
    closed form (Gaussians, mixtures, LogGamma).
``quadrature``
    a deterministic numerical integral, computed by the function named in
    ``meta["logz_method"]`` and cached as a constant here (Rosenbrock, eggbox,
    funnel). ``python -m bench.targets --check`` recomputes every cached value.

For multimodal targets ``responsibility(x)`` returns the oracle probability of
each mode at ``x`` and ``mode_mass`` holds the exact posterior masses
``E_post[responsibility]``. Every sampler's mode masses are measured with this
same function, applied to its weighted (or equal-weight) samples.

Names: ``gauss_d{d}``, ``rosen_d{d}``, ``funnel_d{d}``, ``loggamma_d{d}``,
``eggbox_d2``, ``sepW_d{d}``, ``sepM_d{d}``, ``connW_d{d}`` and ``mix3_d10``;
``sepWtw_d{d}`` is ``sepW`` with a banana-twisted main mode.
``BENCH_TARGETS`` lists the ones in the standard sweep.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class Target:
    name: str
    ndim: int
    loglike: Callable
    lo: np.ndarray
    hi: np.ndarray
    logz: float | None
    logz_source: str
    mode_mass: tuple[float, ...] | None = None
    responsibility: Callable | None = None
    meta: dict = field(default_factory=dict)

    def prior_transform(self, u):
        import jax.numpy as jnp

        return jnp.asarray(self.lo) + jnp.asarray(self.hi - self.lo) * u

    def prior_transform_np(self, u):
        return self.lo + (self.hi - self.lo) * np.asarray(u)

    @property
    def log_prior_volume(self) -> float:
        return float(np.sum(np.log(self.hi - self.lo)))


def _norm_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# ---------------------------------------------------------------- Gaussians
GAUSS_HALF_WIDTH = 10.0


def gaussian(d: int) -> Target:
    """Correlated anisotropic Gaussian in the box [-10, 10]^d.

    Covariance ``Q diag(s^2) Q^T`` with ``s`` log-spaced over [0.1, 1] and a
    random rotation ``Q`` (fixed seed per d); mean uniform in [-1, 1]^d. The
    likelihood is a normalised density, so ``logZ = -d log 20``. Every marginal
    sd is <= 1 and every mean is within 1 of the origin, so the box edge is at
    least 9 marginal sd away: the truncated mass is < d * 2.3e-19.
    """
    rng = np.random.default_rng(1000 + d)
    s = np.geomspace(0.1, 1.0, d) if d > 1 else np.ones(1)
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    q = q * np.sign(np.diag(r))
    cov = (q * s**2) @ q.T
    mu = rng.uniform(-1.0, 1.0, d)
    prec = np.linalg.inv(cov)
    const = -0.5 * (d * LOG_2PI + np.linalg.slogdet(cov)[1])
    marg = np.sqrt(np.diag(cov))
    trunc = float(
        np.sum(
            [
                math.erfc((GAUSS_HALF_WIDTH - abs(m)) / (sd * math.sqrt(2)))
                for m, sd in zip(mu, marg, strict=True)
            ]
        )
    )

    import jax.numpy as jnp

    P, m = jnp.asarray(prec), jnp.asarray(mu)

    def loglike(x):
        y = x - m
        return const - 0.5 * y @ P @ y

    w = 2.0 * GAUSS_HALF_WIDTH
    return Target(
        name=f"gauss_d{d}",
        ndim=d,
        loglike=loglike,
        lo=np.full(d, -GAUSS_HALF_WIDTH),
        hi=np.full(d, GAUSS_HALF_WIDTH),
        logz=-d * math.log(w),
        logz_source="analytic",
        meta=dict(
            mean=mu.tolist(),
            sd_eigen=[float(s.min()), float(s.max())],
            truncated_mass_bound=trunc,
        ),
    )


# --------------------------------------------------------------- Rosenbrock
ROSEN_HALF_WIDTH = 5.0


def rosenbrock_logz_quadrature(d: int, n: int = 20001, chunk: int = 512) -> float:
    """Evidence of the chained Rosenbrock target by 1-D transfer-operator quadrature.

    The integrand is ``prod_i exp(-(1-x_i)^2) exp(-100 (x_{i+1} - x_i^2)^2)``,
    a Markov chain in one coordinate, so with ``v_1 = 1`` and
    ``v_{i+1}(y) = int v_i(x) exp(-(1-x)^2 - 100 (y - x^2)^2) dx`` over [-5, 5],
    ``Z * 10^d = int v_d(y) dy``. Trapezoid rule on an ``n``-point grid
    (spacing 5e-4 for n = 20001, against the narrowest kernel width 0.07).
    Converged to < 1e-9 in log Z (n = 10001 vs 20001 at d = 10; d = 2 also
    agrees with a brute-force 2-D grid to 4e-9).
    """
    x = np.linspace(-ROSEN_HALF_WIDTH, ROSEN_HALF_WIDTH, n)
    h = x[1] - x[0]
    tw = np.full(n, h)
    tw[0] = tw[-1] = 0.5 * h
    v = np.ones(n)
    logscale = 0.0
    a = np.exp(-((1.0 - x) ** 2)) * tw  # quadrature weight in x times the x factor
    x2 = x**2
    for _ in range(d - 1):
        new = np.empty(n)
        va = v * a
        for s in range(0, n, chunk):
            y = x[s : s + chunk, None]
            new[s : s + chunk] = np.exp(-100.0 * (y - x2[None, :]) ** 2) @ va
        m = new.max()
        logscale += math.log(m)
        v = new / m
    return logscale + math.log(np.sum(v * tw)) - d * math.log(2 * ROSEN_HALF_WIDTH)


# rosenbrock_logz_quadrature(d) with n = 20001 (see --check).
ROSEN_LOGZ = {2: -5.8041324, 4: -15.1016907, 10: -43.1083552}


def rosenbrock(d: int) -> Target:
    """Chained Rosenbrock on [-5, 5]^d.

    ``lnL = -sum_i [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]``, unnormalised.
    """
    import jax.numpy as jnp

    def loglike(x):
        return -jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    logz = ROSEN_LOGZ.get(d)
    return Target(
        name=f"rosen_d{d}",
        ndim=d,
        loglike=loglike,
        lo=np.full(d, -ROSEN_HALF_WIDTH),
        hi=np.full(d, ROSEN_HALF_WIDTH),
        logz=logz if logz is not None else rosenbrock_logz_quadrature(d),
        logz_source="quadrature",
        meta=dict(logz_method="rosenbrock_logz_quadrature"),
    )


# ------------------------------------------------------------------- funnel
FUNNEL_V, FUNNEL_X, FUNNEL_SIGMA_V = 9.0, 20.0, 3.0


def funnel_logz_quadrature(d: int, n: int = 200001) -> float:
    """Neal's funnel evidence: 1-D quadrature over v of the truncated x-marginals.

    ``Z * V = int_{-9}^{9} N(v; 0, 3) [erf(20 / (sqrt(2) e^{v/2}))]^{d-1} dv``.
    """
    from math import erf

    v = np.linspace(-FUNNEL_V, FUNNEL_V, n)
    sd = np.exp(v / 2.0)
    erf_v = np.vectorize(erf)
    f = (
        np.exp(-0.5 * (v / FUNNEL_SIGMA_V) ** 2)
        / (FUNNEL_SIGMA_V * math.sqrt(2 * math.pi))
        * erf_v(FUNNEL_X / (math.sqrt(2.0) * sd)) ** (d - 1)
    )
    integral = np.trapezoid(f, v)
    logv = math.log(2 * FUNNEL_V) + (d - 1) * math.log(2 * FUNNEL_X)
    return math.log(integral) - logv


FUNNEL_LOGZ = {10: -36.1491652}


def funnel(d: int) -> Target:
    """Neal's funnel: v ~ N(0, 3^2), x_i | v ~ N(0, e^v).

    Prior box: v in [-9, 9], each x_i in [-20, 20].

    The likelihood is the normalised funnel density; the box truncates the wide
    mouth (sd 20 at v = 6), which the quadrature includes exactly.
    """
    import jax.numpy as jnp

    def loglike(x):
        v, z = x[0], x[1:]
        lv = -0.5 * (v / FUNNEL_SIGMA_V) ** 2 - math.log(FUNNEL_SIGMA_V) - 0.5 * LOG_2PI
        lz = -0.5 * jnp.sum(z**2) * jnp.exp(-v) - 0.5 * (d - 1) * (v + LOG_2PI)
        return lv + lz

    lo = np.r_[-FUNNEL_V, np.full(d - 1, -FUNNEL_X)]
    logz = FUNNEL_LOGZ.get(d)
    return Target(
        name=f"funnel_d{d}",
        ndim=d,
        loglike=loglike,
        lo=lo,
        hi=-lo,
        logz=logz if logz is not None else funnel_logz_quadrature(d),
        logz_source="quadrature",
        meta=dict(logz_method="funnel_logz_quadrature"),
    )


# ----------------------------------------------------------------- LogGamma
LG_SCALE = 1.0 / 30.0


def _loggamma_mass(loc: float) -> float:
    """Mass in [0, 1] of LogGamma(c=1, loc, scale 1/30): CDF(y) = 1 - exp(-e^y)."""

    def cdf(x):
        return -math.expm1(-math.exp((x - loc) / LG_SCALE))

    return cdf(1.0) - cdf(0.0)


def _normal_mass(loc: float) -> float:
    return _norm_cdf((1.0 - loc) / LG_SCALE) - _norm_cdf(-loc / LG_SCALE)


def loggamma(d: int) -> Target:
    """The LogGamma target (Beaujean & Caldwell 2013; Buchner 2016; Lange 2023).

    Unit-cube prior. Coordinate 1 is an equal mixture of LogGamma(c=1) densities
    at 1/3 and 2/3, coordinate 2 an equal mixture of normals at 1/3 and 2/3, all
    with scale 1/30. Coordinates 3..(d+2)/2 are LogGamma at 2/3, the rest normal
    at 2/3. Each factor is a normalised density, so logZ = 0 up to the box
    truncation; the exact value (about -4.5e-5 per LogGamma factor, from the
    heavy left tail) is computed from the CDFs. Four modes (component of
    coordinate 1 x component of coordinate 2), each of mass ~1/4.
    """
    if d < 2:
        raise ValueError("loggamma needs d >= 2")
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    n_lg = (d + 2) // 2 - 2  # coordinates 3..(d+2)/2
    lg_idx = np.arange(2, 2 + n_lg)
    nm_idx = np.arange(2 + n_lg, d)
    c = LG_SCALE
    log_half = math.log(0.5)

    def lg_logpdf(x, loc):
        y = (x - loc) / c
        return y - jnp.exp(y) - math.log(c)

    def nm_logpdf(x, loc):
        y = (x - loc) / c
        return -0.5 * y**2 - math.log(c) - 0.5 * LOG_2PI

    def comps(x):
        a = jnp.stack([lg_logpdf(x[0], 1 / 3), lg_logpdf(x[0], 2 / 3)]) + log_half
        b = jnp.stack([nm_logpdf(x[1], 1 / 3), nm_logpdf(x[1], 2 / 3)]) + log_half
        rest = jnp.sum(lg_logpdf(x[lg_idx], 2 / 3)) + jnp.sum(
            nm_logpdf(x[nm_idx], 2 / 3)
        )
        return a, b, rest

    def loglike(x):
        a, b, rest = comps(x)
        return logsumexp(a) + logsumexp(b) + rest

    def responsibility(x):
        a, b, _ = comps(x)
        ra = jnp.exp(a - logsumexp(a))
        rb = jnp.exp(b - logsumexp(b))
        return jnp.outer(ra, rb).reshape(-1)  # modes (a1b1, a1b2, a2b1, a2b2)

    ma = np.array([_loggamma_mass(1 / 3), _loggamma_mass(2 / 3)]) * 0.5
    mb = np.array([_normal_mass(1 / 3), _normal_mass(2 / 3)]) * 0.5
    logz = (
        math.log(ma.sum())
        + math.log(mb.sum())
        + n_lg * math.log(_loggamma_mass(2 / 3))
        + len(nm_idx) * math.log(_normal_mass(2 / 3))
    )
    masses = np.outer(ma / ma.sum(), mb / mb.sum()).reshape(-1)
    return Target(
        name=f"loggamma_d{d}",
        ndim=d,
        loglike=loglike,
        lo=np.zeros(d),
        hi=np.ones(d),
        logz=logz,
        logz_source="analytic",
        mode_mass=tuple(float(m) for m in masses),
        responsibility=responsibility,
        meta=dict(n_loggamma_tail=int(n_lg), n_normal_tail=int(len(nm_idx))),
    )


# ------------------------------------------------------------------- eggbox
EGG_WIDTH = 10.0 * math.pi


def eggbox_logz_quadrature(n: int = 4001) -> float:
    """Eggbox evidence by the trapezoid rule on an n x n grid (spectrally
    accurate for this smooth function; peak width ~0.1 against spacing 0.008)."""
    x = np.linspace(0.0, EGG_WIDTH, n)
    h = x[1] - x[0]
    tw = np.full(n, h)
    tw[0] = tw[-1] = 0.5 * h
    cx = np.cos(x / 2.0)
    logl = (2.0 + np.outer(cx, cx)) ** 5
    m = logl.max()
    z = tw @ np.exp(logl - m) @ tw
    return m + math.log(z) - 2.0 * math.log(EGG_WIDTH)


EGG_LOGZ = 235.8559403


def eggbox(d: int = 2) -> Target:
    """``lnL = (2 + cos(x/2) cos(y/2))^5`` on [0, 10 pi]^2 (Feroz et al. 2009).

    Reference logZ = 235.856 by quadrature (Feroz et al. quote 235.88).
    18 modes; no mode masses are scored.
    """
    if d != 2:
        raise ValueError("eggbox is 2-D")
    import jax.numpy as jnp

    def loglike(x):
        return (2.0 + jnp.cos(x[0] / 2.0) * jnp.cos(x[1] / 2.0)) ** 5

    return Target(
        name="eggbox_d2",
        ndim=2,
        loglike=loglike,
        lo=np.zeros(2),
        hi=np.full(2, EGG_WIDTH),
        logz=EGG_LOGZ,
        logz_source="quadrature",
        meta=dict(logz_method="eggbox_logz_quadrature"),
    )


# ------------------------------------------------------- Gaussian mixtures
# Ported from tinyns_h100_2026-09-30/multimodal_prototype/mm_targets.py
# (``build``), the targets of the multimodality prototype and the v1 bake-off;
# d = 32 added with the d = 18 minor-mode shapes.
SIG_MIN, SIG_MAX, RHO = 0.002, 0.0632, 0.8
SHAPES = {
    "W": {4: (0.7, 3), 10: (0.8, 7), 18: (0.9, 15), 32: (0.9, 15)},
    "M": {4: (0.5, 3), 10: (0.6, 7), 18: (0.8, 15), 32: (0.8, 15)},
    "S": {4: (0.5, 3), 10: (0.5, 7), 18: (0.7, 15), 32: (0.7, 15)},
    "X": {4: (0.5, 3), 10: (0.5, 8), 18: (0.5, 14), 32: (0.5, 14)},
}
OFF = {"sep": 10.0, "conn": 4.0}


def build_two_mode(d, w2, variant="sep", shape="M", seed=0, twist=0.0):
    """Two-component mixture in the unit cube (verbatim port of mm_targets.build).

    main mode: sigmas log-spaced over [0.002, 0.0632] (shuffled), AR(1) rho 0.8
    in natural order, mean 0.5. minor mode: sigmas = fac * main in ``n_narrow``
    dims, AR(1) |rho| 0.8 over a different ordering, offset along the 2 (d = 4)
    or 3 smallest-sigma dims with total norm OFF[variant] marginal sigmas
    ('sep' 10, i.e. 5.8 sigma per dim at d >= 10; 'conn' 4).

    ``twist`` bends the main mode into a banana with the prototype's
    volume-preserving shear: its density is N1 evaluated at
    ``y_j - twist * sigma_j * (y_i / sigma_i)^2`` (``i`` the widest dim, ``j``
    the narrowest dim that is not offset), so logZ and the mode masses are
    unchanged.
    """
    rng = np.random.default_rng(seed)
    sig = np.geomspace(SIG_MIN, SIG_MAX, d)
    rng.shuffle(sig)
    idx = np.arange(d)
    corr1 = RHO ** np.abs(np.subtract.outer(idx, idx))
    S1 = corr1 * np.outer(sig, sig)
    mu1 = np.full(d, 0.5)

    fac_val, n_narrow = SHAPES[shape][d]
    order = np.argsort(sig)
    n_off = 2 if d <= 4 else 3
    off_dims = order[:n_off]
    fac = np.ones(d)
    narrow_dims = rng.permutation(d)[:n_narrow]
    fac[narrow_dims] = fac_val
    sig2 = sig * fac
    perm = rng.permutation(d)
    pos = np.empty(d, int)
    pos[perm] = idx
    sign = rng.choice([-1.0, 1.0], size=d)
    corr2 = RHO ** np.abs(np.subtract.outer(pos, pos)) * np.outer(sign, sign)
    S2 = corr2 * np.outer(sig2, sig2)

    delta = np.zeros(d)
    delta[off_dims] = sig[off_dims] * np.array([1.0, -1.0, 1.0])[:n_off]
    delta *= OFF[variant] / math.sqrt(n_off)
    mu2 = mu1 + delta
    return dict(
        d=d,
        weights=[1.0 - w2, w2],
        means=[mu1, mu2],
        covs=[S1, S2],
        sigs=[sig, sig2],
        variant=variant,
        shape=shape,
        vol_ratio=fac_val**n_narrow,
        twist=float(twist),
        twist_dims=(int(order[-1]), int(order[n_off])),
    )


def build_three_mode(d=10, weights=(0.7, 0.2, 0.1), seed=0):
    """Three-component version: the main mode and minor mode of
    ``build_two_mode(d, ., 'sep', 'W')``, plus a third mode with its own narrowed
    dims and orientation, offset by 10 marginal sigmas along the next three
    smallest-sigma dims (so modes 2 and 3 are ~14 sigma apart)."""
    t = build_two_mode(d, weights[1], "sep", "W", seed=seed)
    rng = np.random.default_rng(seed + 1)
    sig = t["sigs"][0]
    order = np.argsort(sig)
    off_dims = order[3:6]
    fac_val, n_narrow = SHAPES["W"][d]
    fac = np.ones(d)
    fac[rng.permutation(d)[:n_narrow]] = fac_val
    sig3 = sig * fac
    idx = np.arange(d)
    perm = rng.permutation(d)
    pos = np.empty(d, int)
    pos[perm] = idx
    sign = rng.choice([-1.0, 1.0], size=d)
    corr3 = RHO ** np.abs(np.subtract.outer(pos, pos)) * np.outer(sign, sign)
    S3 = corr3 * np.outer(sig3, sig3)
    delta = np.zeros(d)
    delta[off_dims] = (
        sig[off_dims] * np.array([-1.0, 1.0, 1.0]) * OFF["sep"] / math.sqrt(3)
    )
    t["weights"] = list(weights)
    t["means"].append(t["means"][0] + delta)
    t["covs"].append(S3)
    t["sigs"].append(sig3)
    return t


def mixture_target(name: str, t: dict) -> Target:
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    d = t["d"]
    w = np.asarray(t["weights"], float)
    precs = np.stack([np.linalg.inv(S) for S in t["covs"]])
    consts = np.array(
        [
            math.log(wk) - 0.5 * (d * LOG_2PI + np.linalg.slogdet(S)[1])
            for wk, S in zip(w, t["covs"], strict=True)
        ]
    )
    means = np.stack(t["means"])
    P, M, A = jnp.asarray(precs), jnp.asarray(means), jnp.asarray(consts)

    # mass of each component outside the unit cube (union bound over dims)
    def outside(mu, sg):
        z = np.minimum(mu, 1 - mu) / (np.asarray(sg) * math.sqrt(2))
        return sum(math.erfc(v) for v in z)

    trunc = sum(w[k] * outside(means[k], t["sigs"][k]) for k in range(len(w)))

    twist = float(t.get("twist", 0.0))
    ti, tj = t.get("twist_dims", (0, 1))
    si, sj = float(t["sigs"][0][ti]), float(t["sigs"][0][tj])

    def comps(x):
        y = x[None, :] - M
        if twist:  # banana shear of the main mode (component 0) only
            y = y.at[0, tj].add(-twist * sj * (y[0, ti] / si) ** 2)
        return A - 0.5 * jnp.einsum("ki,kij,kj->k", y, P, y)

    def loglike(x):
        return logsumexp(comps(x))

    def responsibility(x):
        c = comps(x)
        return jnp.exp(c - logsumexp(c))

    return Target(
        name=name,
        ndim=d,
        loglike=loglike,
        lo=np.zeros(d),
        hi=np.ones(d),
        logz=0.0,
        logz_source="analytic",
        mode_mass=tuple(float(x) for x in w),
        responsibility=responsibility,
        meta=dict(
            variant=t.get("variant"),
            shape=t.get("shape"),
            vol_ratio=t.get("vol_ratio"),
            twist=t.get("twist", 0.0),
            truncated_mass_bound=float(trunc),
        ),
    )


# ----------------------------------------------------------------- registry
MINOR_WEIGHT = 0.06
TWIST = 0.3  # the banana-twisted variants, e.g. ``sepWtw_d10`` (prototype value)

BENCH_TARGETS = (
    [f"gauss_d{d}" for d in (2, 8, 16, 32, 64)]
    + ["rosen_d2", "rosen_d10", "funnel_d10"]
    + [f"loggamma_d{d}" for d in (2, 10, 30)]
    + ["eggbox_d2"]
    + [f"sepW_d{d}" for d in (4, 10, 18, 32)]
    + [f"connW_d{d}" for d in (4, 10, 18, 32)]
    + [f"sepM_d{d}" for d in (10, 18, 32)]
    + ["mix3_d10"]
)

_FAMILIES = {
    "gauss": gaussian,
    "rosen": rosenbrock,
    "funnel": funnel,
    "loggamma": loggamma,
    "eggbox": eggbox,
}


def _split(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"([A-Za-z0-9]+?)_d(\d+)", name)
    if m is None:
        raise ValueError(f"bad target name {name!r}; expected <family>_d<ndim>")
    return m.group(1), int(m.group(2))


def mixture_spec(name: str) -> dict | None:
    """The component weights, means and covariances of a mixture target."""
    fam, d = _split(name)
    mm = re.fullmatch(r"(sep|conn)([WMSX])(tw)?", fam)
    if mm:
        twist = TWIST if mm.group(3) else 0.0
        return build_two_mode(d, MINOR_WEIGHT, mm.group(1), mm.group(2), twist=twist)
    if fam == "mix3":
        return build_three_mode(d)
    return None


def get_target(name: str) -> Target:
    fam, d = _split(name)
    if fam in _FAMILIES:
        return _FAMILIES[fam](d)
    spec = mixture_spec(name)
    if spec is not None:
        return mixture_target(name, spec)
    raise ValueError(f"unknown target family {fam!r}")


def _check() -> None:
    """Recompute every cached quadrature constant."""
    for d, v in ROSEN_LOGZ.items():
        print(
            f"rosen_d{d}: cached {v:.7f} quadrature {rosenbrock_logz_quadrature(d):.7f}"
        )
    for d, v in FUNNEL_LOGZ.items():
        print(f"funnel_d{d}: cached {v:.7f} quadrature {funnel_logz_quadrature(d):.7f}")
    print(f"eggbox_d2: cached {EGG_LOGZ:.7f} quadrature {eggbox_logz_quadrature():.7f}")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
    else:
        for n in BENCH_TARGETS:
            t = get_target(n)
            print(
                f"{n:14s} d={t.ndim:3d} logZ={t.logz:12.6f} ({t.logz_source})"
                f" modes={t.mode_mass}"
            )
