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
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.special import logsumexp

from tinyns import checkpoint as _checkpoint
from tinyns.callables import (
    _device_leaves,
    _kernel_cache,
    _partition_callable,
    _split_callables,
)
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
# this many seconds, growing them at most 4x per chunk from one step. Chunk
# lengths may follow the wall clock: the random stream depends on the step
# count only. A checkpoint is written at a chunk boundary at most this often.
_CHUNK_EFOLDS = 4.0
_CHUNK_SECONDS = 45.0
_CHECKPOINT_SECONDS = 600.0
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
def _step_kernel(loglike_spec, prior_spec, cfg: Config):
    def kernel(state, *leaves):
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
    ``count[lane]``.
    """
    d, k = cfg.ndim, cfg.num_delete

    def lane_step(state, leaves):
        return _step(state, *_rebuild(loglike_spec, prior_spec, leaves), cfg)

    def kernel(state, n_steps, dlogz, max_steps, call_budget, *leaves):
        dtype, i32 = state.logl.dtype, jnp.int32

        def terminate(state, budget):
            return _terminate(state, cfg, dlogz, max_steps, budget)

        if axes is None:
            lanes = ()

            def step_all(state):
                return lane_step(state, leaves)

        else:
            lanes = state.it.shape
            terminate = jax.vmap(terminate)

            def step_all(state):
                new, dead = jax.vmap(lane_step, in_axes=(0, axes))(state, leaves)
                return _select(state.status == RUNNING, new, state), dead

        rows = Dead(
            jnp.zeros((capacity, *lanes, k, d), state.u.dtype),
            jnp.zeros((capacity, *lanes, k), dtype),
            jnp.zeros((capacity, *lanes, k), dtype),
            jnp.zeros((capacity, *lanes, k), i32),
            jnp.zeros((capacity, *lanes, k), i32),
        )
        limit = jnp.minimum(n_steps, capacity)

        def cond(carry):
            state, _, _, i = carry
            return (i < limit) & jnp.any(state.status == RUNNING)

        def body(carry):
            state, rows, count, i = carry
            running = state.status == RUNNING
            state, dead = step_all(state)
            rows = jax.tree_util.tree_map(
                lambda r, x: lax.dynamic_update_index_in_dim(r, x, i, 0), rows, dead
            )
            count = count + (running & (state.status != PLATEAU)).astype(i32)
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
    """
    (ll_dyn, ll_spec), (pt_dyn, pt_spec) = _split_callables(
        loglike, prior_transform, cfg.ndim
    )
    return _init_kernel(ll_spec, pt_spec, cfg, None)(
        _as_key(key), *ll_dyn, *pt_dyn
    )


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


def _key_batch(key):
    """Return ``(key, batch)``: the key as an array and its lane count.

    ``batch`` is ``None`` for one key (an int seed, a raw ``uint32`` key or a
    typed key) and ``B`` for a batch of ``B`` keys (a leading axis).
    """
    key = _as_key(key)
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


def _empty_dead(cfg: Config, dtype) -> Dead:
    k, d = cfg.num_delete, cfg.ndim
    return Dead(
        np.zeros((0, k, d), dtype),
        np.zeros((0, k), dtype),
        np.zeros((0, k), dtype),
        np.zeros((0, k), np.int32),
        np.zeros((0, k), np.int32),
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
    after each step) or on a plateau. Chunks of steps run on the device; the
    host syncs once per chunk, sizes the next one to about 45 s and, with
    ``progress``, prints one line per chunk.

    ``checkpoint`` is a path: if the file exists the run resumes from it
    (refusing a different config, x64 flag, float dtype, key or batch), and
    the run writes it atomically at a chunk boundary at most every
    ``_CHECKPOINT_SECONDS`` and at the end. A resumed run is bit-identical to
    an uninterrupted one: each step splits ``state.key``, so the random stream
    depends on the step count only, never on where the chunks end. A status of
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
                raise ValueError(
                    f"batched_data: every array leaf needs a leading axis of "
                    f"{batch} lanes, got shape {leaf.shape}"
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
        state = ckpt.state
        status = np.asarray(state.status)
        stopped = np.isin(status, (CONVERGED, MAXITER, MAXCALL))  # re-tested
        status = np.where(stopped, RUNNING, status)
        state = state._replace(status=jnp.asarray(status, jnp.int32))
        rows_by_lane = [[dead] for dead in ckpt.dead]
        ncall, ncall_valid = list(ckpt.ncall), list(ckpt.ncall_valid)
        wall0, compile_s, nchunks = ckpt.wall_time_s, ckpt.compile_s, ckpt.chunks
    else:
        state = _init_kernel(ll_spec, pt_spec, cfg, axes)(key, *leaves)
        rows_by_lane = [[_empty_dead(cfg, dtype)] for _ in range(lanes)]
        ncall, ncall_valid = [0] * lanes, [0] * lanes
        wall0 = compile_s = 0.0
        nchunks = 0

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
                chunks=nchunks,
                config=dataclasses.asdict(cfg),
                batch=batch,
                batched_data=batched_data,
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
            rows_by_lane[lane].append(
                jax.tree_util.tree_map(lambda r, n=n, b=lane: r[:n, b], rows)
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
                    "resumed": resumed,
                },
            )
        )
    return results[0] if batch is None else results


def _progress_line(cfg, batch, status, it, logz, remain, ncall, rate, acc, scale):
    """One progress line: iteration, logz, dlogz, calls, calls/s, acceptance
    and step scale (for a batch: the running lanes and the range of each)."""
    k = cfg.num_delete
    if batch is None:
        return (
            f"tinyns: niter={int(it[0]) * k} logz={float(logz[0]):.4f} "
            f"dlogz={float(remain[0]):.3g} ncall={ncall[0]} calls/s={rate:.3g} "
            f"acc={acc:.3f} scale={float(scale[0]):.3g} [{STATUS[int(status[0])]}]"
        )
    running = status == RUNNING
    return (
        f"tinyns: lanes running={int(running.sum())}/{batch} "
        f"niter={int(it.min()) * k}..{int(it.max()) * k} "
        f"dlogz<={float(np.max(remain)):.3g} ncall={sum(ncall)} "
        f"calls/s={rate:.3g} acc={acc:.3f} scale={float(np.median(scale)):.3g}"
    )
