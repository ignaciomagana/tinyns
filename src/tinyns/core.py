"""The full-JAX core of :mod:`tinyns`: ``Config``, ``State``, ``init``, ``step``.

One :func:`step` deletes the ``k = cfg.num_delete`` lowest live points and
replaces them with the ends of ``k`` fixed-length constrained random walks,
seeded from distinct live points above the new contour ``L*`` (the highest
deleted likelihood). The evidence bookkeeping treats the ``k`` deaths of a
step as consecutive deaths at live counts ``m, m-1, ..., m-k+1`` (the
birth/death contours of Fowlie, Handley and Su), so the expected log prior
volume after ``it`` steps is ``-it * S`` with ``S = sum_j 1 / (m - j)``.
:func:`finalise` recomputes the weights, the evidence and its error on the
host in float64.

:func:`run` is the driver: a jitted ``lax.while_loop`` runs up to ``n_steps``
steps into a preallocated buffer of dead rows, and the host syncs once per
chunk. ``loglike`` and ``prior_transform`` may be pytree callables
(``jax.tree_util.Partial(fn, data)``): their array leaves are jit arguments
and the compiled kernels are cached on their structure
(:mod:`tinyns.callables`), so datasets of one shape share one compile.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.special import logsumexp

from tinyns.callables import _device_leaves, _kernel_cache, _split_callables
from tinyns.result import NestedSamplingResult, _evidence

# Run status codes (State.status).
RUNNING, CONVERGED, MAXITER, MAXCALL, PLATEAU = range(5)
STATUS = ("running", "converged", "maxiter", "maxcall", "plateau")
_MESSAGES = (
    "running",
    "converged: the live points hold less than dlogz of the evidence",
    "stopped at maxiter before reaching dlogz",
    "stopped at maxcall before reaching dlogz",
    "likelihood plateau: no live point lies above the deleted ones",
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
# this many seconds, growing them at most 4x per chunk from one step.
_CHUNK_EFOLDS = 4.0
_CHUNK_SECONDS = 45.0
_INT32_MAX = 2**31 - 1


def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


@dataclasses.dataclass(frozen=True)
class Config:
    """Static sampler configuration (hashable: part of the compile cache key).

    ``nlive`` (``m``) live points; ``num_delete`` (``k``) points deleted and
    replaced per step, default ``max(1, nlive // 10)``, at most ``nlive // 2``;
    ``walks`` steps per replacement chain, default ``max(25, 6 * ndim)``.
    """

    ndim: int
    nlive: int = 1000
    num_delete: int | None = None
    walks: int | None = None

    def __post_init__(self):
        ndim = _check_int("ndim", self.ndim, 1)
        nlive = _check_int("nlive", self.nlive, 2)
        k = max(1, nlive // 10) if self.num_delete is None else self.num_delete
        k = _check_int("num_delete", k, 1)
        if k > nlive // 2:
            raise ValueError(f"num_delete must be at most nlive // 2 = {nlive // 2}")
        walks = max(25, 6 * ndim) if self.walks is None else self.walks
        walks = _check_int("walks", walks, 1)
        for name, value in (
            ("ndim", ndim),
            ("nlive", nlive),
            ("num_delete", k),
            ("walks", walks),
        ):
            object.__setattr__(self, name, value)

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

    ``ncall`` counts the likelihood evaluations (out-of-cube proposals of a
    vmapped step are evaluated, clipped, and counted) and ``ncall_valid`` the
    in-cube ones; both are int32, and :func:`run` drains them into host
    integers after every chunk. ``logz`` is the running evidence of the dead
    points, used only to stop; :func:`finalise` recomputes it.
    """

    key: Any  # PRNG key
    u: Any  # (m, d) live points in the unit cube
    logl: Any  # (m,)
    logl_birth: Any  # (m,) contour each live point was born above; -inf at init
    it: Any  # int32 steps done
    logz: Any  # running log evidence of the dead points
    log_scale: Any  # log of the walk-step multiplier
    ncall: Any  # int32 likelihood evaluations
    ncall_valid: Any  # int32 in-cube likelihood evaluations
    status: Any  # int32: RUNNING, CONVERGED, MAXITER, MAXCALL or PLATEAU


class Dead(NamedTuple):
    """The ``k`` rows of one step: deleted points by increasing likelihood.

    ``insertion[j]`` and ``moves[j]`` belong to the ``j``-th new point (not to
    dead point ``j``): its rank among the ``m - k`` surviving live points
    (uniform on ``0..m-k`` for a correct constrained sampler) and the accepted
    moves of its chain (0: the chain returned its seed).
    """

    u: Any  # (k, d)
    logl: Any  # (k,)
    logl_birth: Any  # (k,)
    insertion: Any  # (k,) int32
    moves: Any  # (k,) int32


def _as_key(key):
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return random.PRNGKey(int(key))
    return key


def _loglike_u(loglike, prior_transform, u, dtype):
    """``loglike(prior_transform(u))`` as a ``dtype`` scalar; NaN becomes -inf."""
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
    x = (u - (w @ u) / n) * w[:, None]
    dof = jnp.maximum(n - 1.0, 1.0)
    std = jnp.sqrt(jnp.sum(x * x, axis=0) / dof)
    z = x / jnp.where(std > 0, std, 1.0)
    corr = z.T @ z / dof + 10.0 * jnp.finfo(u.dtype).eps * jnp.eye(d, dtype=u.dtype)
    chol = jnp.linalg.cholesky(corr)
    chol = jnp.where(jnp.all(jnp.isfinite(chol)), chol, jnp.eye(d, dtype=u.dtype))
    return std[:, None] * chol


def _propose(key, u, chol, scale):
    """One chain proposal: the live-covariance random walk ``u + s L z``.

    This is the move hook of the chain kernel. The walk is symmetric, so the
    chain accepts any in-cube proposal above ``L*``; an inter-mode move (a
    static mixture with the walk, PR 3) adds its Hastings ratio here.
    """
    return u + scale * (chol @ random.normal(key, u.shape, u.dtype))


def _chain(key, u, logl, lstar, chol, scale, loglike, prior_transform, walks, skip):
    """A fixed-length constrained Metropolis chain from ``(u, logl)``.

    Proposals outside the unit cube or at or below ``lstar`` are rejected; an
    unmoved chain returns its seed. With ``skip`` (one unbatched chain) an
    out-of-cube proposal skips the likelihood (``lax.cond``); otherwise the
    clipped proposal is evaluated and masked. Returns ``(u, logl, moves,
    ncall, ncall_valid)``.
    """
    dtype = logl.dtype

    def evaluate(v):
        return _loglike_u(loglike, prior_transform, v, dtype)

    def body(carry, key):
        u, logl, moves, nev, nvalid = carry
        prop = _propose(key, u, chol, scale)
        inside = jnp.all((prop >= 0.0) & (prop <= 1.0))
        clipped = jnp.clip(prop, 0.0, 1.0)
        if skip:
            new = lax.cond(
                inside, evaluate, lambda v: jnp.asarray(-jnp.inf, dtype), clipped
            )
            nev = nev + inside.astype(jnp.int32)
        else:
            new = jnp.where(inside, evaluate(clipped), -jnp.inf)
            nev = nev + 1
        accept = inside & (new > lstar)
        u = jnp.where(accept, prop, u)
        logl = jnp.where(accept, new, logl)
        moves = moves + accept.astype(jnp.int32)
        return (u, logl, moves, nev, nvalid + inside.astype(jnp.int32)), None

    zero = jnp.zeros((), jnp.int32)
    carry, _ = lax.scan(body, (u, logl, zero, zero, zero), random.split(key, walks))
    return carry


def _init(key, loglike, prior_transform, cfg: Config) -> State:
    m, d, k = cfg.nlive, cfg.ndim, cfg.num_delete
    dtype = jnp.result_type(float)
    key, sub = random.split(key)
    u = random.uniform(sub, (m, d), dtype)
    logl = lax.map(
        lambda v: _loglike_u(loglike, prior_transform, v, dtype), u, batch_size=k
    )
    i32 = jnp.int32
    return State(
        key=key,
        u=u,
        logl=logl,
        logl_birth=jnp.full((m,), -jnp.inf, dtype),
        it=jnp.zeros((), i32),
        logz=jnp.asarray(-jnp.inf, dtype),
        log_scale=jnp.asarray(math.log(_INITIAL_SCALE), dtype),
        ncall=jnp.asarray(m, i32),
        ncall_valid=jnp.asarray(m, i32),
        status=jnp.asarray(RUNNING, i32),
    )


def _step(state: State, loglike, prior_transform, cfg: Config):
    m, k, walks = cfg.nlive, cfg.num_delete, cfg.walks
    dtype = state.logl.dtype
    # The k lowest points, by increasing likelihood; L* is the highest of them.
    _, worst = lax.top_k(-state.logl, k)
    dead_logl = state.logl[worst]
    lstar = dead_logl[-1]
    survivor = jnp.ones((m,), bool).at[worst].set(False)
    above = state.logl > lstar
    n_above = jnp.sum(above, dtype=jnp.int32)

    # Seeds: distinct live points above L* (Gumbel top-k), with replacement
    # only when fewer than k exist.
    key, k_seed, k_fill, k_chain = random.split(state.key, 4)
    score = jnp.where(above, random.gumbel(k_seed, (m,), dtype), -jnp.inf)
    _, top = lax.top_k(score, k)
    fill = random.categorical(k_fill, jnp.where(above, 0.0, -jnp.inf), shape=(k,))
    seeds = jnp.where(jnp.arange(k) < n_above, top, fill)

    chol = _live_chol(state.u, above)
    scale = jnp.exp(state.log_scale)
    chain_args = (lstar, chol, scale, loglike, prior_transform, walks)
    if k == 1:  # no vmap: the cond really skips out-of-cube proposals
        out = _chain(k_chain, state.u[seeds[0]], state.logl[seeds[0]],
                     *chain_args, True)
        new_u, new_logl, moves, nev, nvalid = (x[None] for x in out)
    else:
        new_u, new_logl, moves, nev, nvalid = jax.vmap(
            lambda key, u, logl: _chain(key, u, logl, *chain_args, False)
        )(random.split(k_chain, k), state.u[seeds], state.logl[seeds])

    insertion = jnp.sum(
        survivor & (state.logl <= new_logl[:, None]), axis=1, dtype=jnp.int32
    )
    log_x = -state.it.astype(dtype) * jnp.asarray(cfg.log_shrink, dtype)
    logwt = dead_logl + jnp.asarray(cfg._log_widths(), dtype) + log_x
    acceptance = jnp.sum(moves).astype(dtype) / (k * walks)
    rate = 0.5 * min(1.0, k / 32)
    log_scale = jnp.clip(
        state.log_scale + rate * jnp.clip(acceptance - _TARGET_ACCEPT, -0.5, 0.5),
        math.log(_MIN_SCALE),
        math.log(_MAX_SCALE),
    )
    ncall = state.ncall + jnp.sum(nev, dtype=jnp.int32)
    ncall_valid = state.ncall_valid + jnp.sum(nvalid, dtype=jnp.int32)
    new = State(
        key=key,
        u=state.u.at[worst].set(new_u),
        logl=state.logl.at[worst].set(new_logl),
        logl_birth=state.logl_birth.at[worst].set(lstar),
        it=state.it + 1,
        logz=jnp.logaddexp(state.logz, logsumexp(logwt)),
        log_scale=log_scale.astype(dtype),
        ncall=ncall,
        ncall_valid=ncall_valid,
        status=state.status,
    )
    # A plateau (nothing above L*) leaves the live set as it was.
    stuck = state._replace(status=jnp.asarray(PLATEAU, jnp.int32))
    new = new._replace(**{
        name: jnp.where(n_above == 0, getattr(stuck, name), getattr(new, name))
        for name in ("u", "logl", "logl_birth", "it", "logz", "log_scale", "status")
    })
    dead = Dead(state.u[worst], dead_logl, state.logl_birth[worst], insertion, moves)
    return new, dead


def delta_logz(state: State, cfg: Config):
    """``log(Z + X max L_live) - log Z`` of ``state`` (``inf`` before any death)."""
    dtype = state.logl.dtype
    log_x = -state.it.astype(dtype) * jnp.asarray(cfg.log_shrink, dtype)
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


@_kernel_cache
def _init_kernel(loglike_spec, prior_spec, cfg: Config):
    def kernel(key, *leaves):
        return _init(key, *_rebuild(loglike_spec, prior_spec, leaves), cfg)

    return jax.jit(kernel)


@_kernel_cache
def _step_kernel(loglike_spec, prior_spec, cfg: Config):
    def kernel(state, *leaves):
        return _step(state, *_rebuild(loglike_spec, prior_spec, leaves), cfg)

    return jax.jit(kernel)


@_kernel_cache
def _chunk_kernel(loglike_spec, prior_spec, cfg: Config, capacity: int):
    """Jitted chunk: up to ``min(n_steps, capacity)`` steps into a dead buffer.

    ``n_steps``, ``dlogz``, ``max_steps`` and ``call_budget`` are traced, so
    the driver changes them without recompiling. Returns the state, the
    buffer of ``capacity`` steps of :class:`Dead` rows, the number of steps
    filled and the dlogz remainder.
    """
    d, k = cfg.ndim, cfg.num_delete

    def kernel(state, n_steps, dlogz, max_steps, call_budget, *leaves):
        loglike, prior_transform = _rebuild(loglike_spec, prior_spec, leaves)
        dtype, i32 = state.logl.dtype, jnp.int32
        rows = Dead(
            jnp.zeros((capacity, k, d), state.u.dtype),
            jnp.zeros((capacity, k), dtype),
            jnp.zeros((capacity, k), dtype),
            jnp.zeros((capacity, k), i32),
            jnp.zeros((capacity, k), i32),
        )
        limit = jnp.minimum(n_steps, capacity)

        def cond(carry):
            state, _, i = carry
            return (i < limit) & (state.status == RUNNING)

        def body(carry):
            state, rows, i = carry
            state, dead = _step(state, loglike, prior_transform, cfg)
            rows = jax.tree_util.tree_map(
                lambda r, x: lax.dynamic_update_index_in_dim(r, x, i, 0), rows, dead
            )
            i = i + (state.status != PLATEAU).astype(i32)
            return _terminate(state, cfg, dlogz, max_steps, call_budget), rows, i

        start = _terminate(state, cfg, dlogz, max_steps, call_budget)
        state, rows, count = lax.while_loop(cond, body, (start, rows, jnp.int32(0)))
        return state, rows, count, delta_logz(state, cfg)

    return jax.jit(kernel)


def init(key, loglike, prior_transform, cfg: Config) -> State:
    """Draw ``cfg.nlive`` uniform points, evaluate them, return the first state.

    ``key`` is a PRNG key or an int seed. The points are evaluated with
    ``lax.map`` in batches of ``cfg.num_delete``, the batch a step uses.
    """
    (ll_dyn, ll_spec), (pt_dyn, pt_spec) = _split_callables(
        loglike, prior_transform, cfg.ndim
    )
    return _init_kernel(ll_spec, pt_spec, cfg)(_as_key(key), *ll_dyn, *pt_dyn)


def step(state: State, loglike, prior_transform, cfg: Config) -> tuple[State, Dead]:
    """Delete the ``k`` lowest live points and replace them; return the new
    state and the step's :class:`Dead` rows.

    It does not test termination (see :func:`delta_logz`). On a plateau (no
    live point above the deleted ones) the live set is unchanged, the rows are
    not valid and ``status`` becomes ``PLATEAU``.
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
    ncall: int | None = None,
    metadata: dict | None = None,
) -> NestedSamplingResult:
    """Build the :class:`NestedSamplingResult` from the final state and the rows.

    ``dead`` holds the valid rows of every step in order, with any leading
    shape (``(steps, k)`` or flat). The final live points are appended by
    increasing likelihood at live counts ``m, m-1, ..., 1``; the weights,
    ``logz`` and ``logzerr = sqrt(sum_i dH_i / n_i)`` are recomputed in
    float64 (:func:`tinyns.result._evidence`). ``prior_transform`` maps the
    samples to parameter space (without it ``samples`` are the unit-cube
    points); ``ncall`` overrides ``state.ncall`` (the driver drains it).
    """
    state, dead = jax.device_get((state, dead))
    m, d, k = cfg.nlive, cfg.ndim, cfg.num_delete
    dead_logl = np.asarray(dead.logl, np.float64).reshape(-1)
    niter = dead_logl.size
    if niter % k:
        raise ValueError(f"{niter} dead rows are not whole steps of {k}")
    order = np.argsort(np.asarray(state.logl), kind="stable")
    u = np.concatenate([np.asarray(dead.u).reshape(-1, d), np.asarray(state.u)[order]])
    logl = np.concatenate([dead_logl, np.asarray(state.logl, np.float64)[order]])
    logl_birth = np.concatenate([
        np.asarray(dead.logl_birth, np.float64).reshape(-1),
        np.asarray(state.logl_birth, np.float64)[order],
    ])
    nlive_i = np.concatenate([np.tile(m - np.arange(k), niter // k), m - np.arange(m)])
    logwt, logz, logzerr = _evidence(logl, nlive_i)

    log_x = -(niter // k) * cfg.log_shrink
    logz_dead = float(np.logaddexp.reduce(logwt[:niter])) if niter else -math.inf
    remain = (
        float(np.logaddexp(logz_dead, log_x + logl[niter:].max()) - logz_dead)
        if math.isfinite(logz_dead)
        else math.inf
    )
    moves = np.asarray(dead.moves).reshape(-1)
    status = int(state.status)
    samples = u
    if prior_transform is not None:
        theta = jax.vmap(prior_transform)(jnp.asarray(u, state.u.dtype))
        samples = np.asarray(theta).reshape(len(u), -1)
    info = {
        "status": STATUS[status],
        "num_delete": k,
        "walks": cfg.walks,
        "final_delta_logz": remain,
        "acceptance": float(moves.sum() / (niter * cfg.walks)) if niter else None,
        "unmoved_fraction": float(np.mean(moves == 0)) if niter else None,
        "scale": float(np.exp(state.log_scale)),
        "ncall_valid": int(state.ncall_valid),
        "x64": bool(jax.config.jax_enable_x64),
    }
    return NestedSamplingResult(
        samples_u=u,
        samples=samples,
        logl=logl,
        logwt=logwt,
        logl_birth=logl_birth,
        nlive_i=nlive_i,
        logz=logz,
        logzerr=logzerr,
        ncall=int(state.ncall) if ncall is None else int(ncall),
        niter=niter,
        nlive=m,
        num_delete=k,
        ndim=d,
        success=status in (CONVERGED, PLATEAU),
        message=_MESSAGES[status],
        metadata={**info, **(metadata or {})},
    )


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
) -> NestedSamplingResult:
    """Run nested sampling to termination; return the result.

    Stops when ``delta_logz < dlogz``, before ``maxiter`` dead points would be
    exceeded, once ``maxcall`` likelihood evaluations are reached (checked
    after each step) or on a plateau. Chunks of steps run on the device; the
    host syncs once per chunk and sizes the next one to about 45 s.
    """
    dlogz = float(dlogz)
    if not dlogz >= 0.0:
        raise ValueError("dlogz must be non-negative")
    if maxiter is not None:
        maxiter = _check_int("maxiter", maxiter, 0)
    if maxcall is not None:
        maxcall = _check_int("maxcall", maxcall, 0)
    t0 = time.perf_counter()
    loglike = _device_leaves(loglike)
    prior_transform = _device_leaves(prior_transform)
    (ll_dyn, ll_spec), (pt_dyn, pt_spec) = _split_callables(
        loglike, prior_transform, cfg.ndim
    )
    leaves = (*ll_dyn, *pt_dyn)
    state = _init_kernel(ll_spec, pt_spec, cfg)(_as_key(key), *leaves)
    capacity = max(1, math.ceil(_CHUNK_EFOLDS / cfg.log_shrink))
    kernel = _chunk_kernel(ll_spec, pt_spec, cfg, capacity)
    k = cfg.num_delete
    max_steps = _INT32_MAX if maxiter is None else min(maxiter // k, _INT32_MAX)
    i32, dtype = jnp.int32, state.logl.dtype
    zero = jnp.zeros((), i32)
    ncall = ncall_valid = nchunks = 0
    chunks, n_steps = [], 1
    while True:
        budget = _INT32_MAX if maxcall is None else min(maxcall - ncall, _INT32_MAX)
        tick = time.perf_counter()
        state, rows, count, remain = kernel(
            state,
            jnp.asarray(n_steps, i32),
            jnp.asarray(dlogz, dtype),
            jnp.asarray(max_steps, i32),
            jnp.asarray(max(budget, 0), i32),
            *leaves,
        )
        count, calls, valid, status, remain, logz, rows = jax.device_get(
            (count, state.ncall, state.ncall_valid, state.status, remain, state.logz,
             rows)
        )
        elapsed = time.perf_counter() - tick
        nchunks += 1
        ncall += int(calls)
        ncall_valid += int(valid)
        state = state._replace(ncall=zero, ncall_valid=zero)
        if status == MAXCALL and (maxcall is None or ncall < maxcall):
            status = RUNNING  # only the int32 window of the budget was used up
            state = state._replace(status=jnp.asarray(RUNNING, i32))
        chunks.append(jax.tree_util.tree_map(lambda r, c=int(count): r[:c], rows))
        if progress:
            ndead = sum(c.logl.shape[0] for c in chunks) * k
            print(
                f"tinyns: ndead={ndead} ncall={ncall} logz={float(logz):.4f} "
                f"dlogz={float(remain):.3g} chunk={int(count)} steps "
                f"{elapsed:.2f}s [{STATUS[int(status)]}]",
                flush=True,
            )
        if status != RUNNING:
            break
        per_step = elapsed / max(int(count), 1)
        n_steps = int(min(capacity, 4 * n_steps, max(1.0, _CHUNK_SECONDS / per_step)))

    dead = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *chunks)
    return finalise(
        state,
        dead,
        cfg,
        prior_transform=prior_transform,
        ncall=ncall,
        metadata={
            "ncall_valid": ncall_valid,
            "dlogz": dlogz,
            "maxiter": maxiter,
            "maxcall": maxcall,
            "chunks": nchunks,
            "wall_time": time.perf_counter() - t0,
        },
    )
