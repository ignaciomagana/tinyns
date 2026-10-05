"""Static nested-sampling implementation for :mod:`tinyns`."""

from __future__ import annotations

import functools
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.special import logsumexp

from tinyns.clusters import ClusterTracker
from tinyns.result import NestedSamplingResult
from tinyns.samplers import (
    _callable_leaves,
    _callable_specs,
    _evaluate_jax_batch,
    _make_rwalk_jax_kernel_cached,
    _split_callables,
)
from tinyns.state import NestedRunState, save_checkpoint_npz
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike

# The live-cov step is ``scale * L z``. The scale starts at 0.5 and, after
# every block, moves in log space toward 25% move acceptance. It is a
# dimensionless O(1) factor (the step already follows the contracting live
# set), so the update can chase it fast.
_INITIAL_SCALE = 0.5
_TARGET_ACCEPT = 0.25
_SCALE_RATE = 0.5
_MIN_SCALE = 1e-3
_MAX_SCALE = 10.0
# Likelihood-call budget of one replacement: a batch of ``walks`` steps on
# each of ``replacement_chains`` chains is retried until one chain ends above
# the threshold, at most ``max(1, 10_000 // (walks * replacement_chains))``
# times. Chains start strictly above the threshold, so only a likelihood
# plateau exhausts it.
_MAX_REPLACEMENT_CALLS = 10_000


def _positive_int(name: str, value) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_options(
    ndim: int,
    *,
    walks: int | None,
    replacement_chains: int,
    block_size: int,
    cluster_swap: bool | None,
) -> dict:
    """Validate the sampler options and resolve their ``None`` defaults.

    ``walks`` defaults to ``max(25, 6 * ndim)`` (12 for ``ndim=1``).
    ``cluster_swap`` defaults to True with one replacement chain, the only
    configuration that supports it, and to False otherwise; an explicit True
    with more chains raises.
    """

    if walks is None:
        # 1-D needs far fewer walks for unbiased logZ (validated to 10);
        # from 2-D on, max(25, 6 * ndim) (see CHANGELOG, v0.2.0 / v0.2.1).
        walks = 12 if int(ndim) == 1 else max(25, 6 * int(ndim))
    _positive_int("walks", walks)
    _positive_int("replacement_chains", replacement_chains)
    _positive_int("block_size", block_size)
    if cluster_swap is None:
        cluster_swap = replacement_chains == 1
    elif cluster_swap and replacement_chains != 1:
        raise NotImplementedError("cluster_swap=True needs replacement_chains=1")
    return {
        "walks": walks,
        "replacement_chains": replacement_chains,
        "block_size": block_size,
        "cluster_swap": bool(cluster_swap),
    }


def _evaluate_live_points_jax(
    loglike, prior_transform, live_u, ndim, *, chunk_size, callable_leaves=None
):
    """Evaluate the initial live set in one compiled pass.

    Points go through ``lax.map`` ``chunk_size`` at a time (vmapped within a
    chunk), so device memory is bounded by the same batch the rwalk kernel uses.
    Array leaves of pytree callables and large closure constants are jit
    arguments, not constants; ``callable_leaves`` (default: taken from the
    callables) lets the caller pass leaves it has already placed on the device.
    """
    nlive = int(live_u.shape[0])
    chunk_size = max(1, min(int(chunk_size), nlive))
    nchunks = -(-nlive // chunk_size)
    pad = nchunks * chunk_size - nlive
    padded = jnp.concatenate(
        [live_u, jnp.broadcast_to(live_u[:1], (pad, ndim))], axis=0
    ).reshape((nchunks, chunk_size, ndim))

    (loglike_dynamic, loglike_spec), (prior_dynamic, prior_spec) = (
        _split_callables(loglike, prior_transform, ndim)
    )
    if callable_leaves is None:
        callable_leaves = tuple(loglike_dynamic) + tuple(prior_dynamic)
    evaluate = _make_live_points_kernel(loglike_spec, prior_spec, int(ndim))
    theta, logl = evaluate(padded, *callable_leaves)
    theta = theta.reshape((nchunks * chunk_size,) + theta.shape[2:])[:nlive]
    return theta, logl.reshape((-1,))[:nlive]


@functools.lru_cache(maxsize=32)
def _make_live_points_kernel(loglike_spec, prior_spec, ndim: int):
    """Return the cached jitted pass of :func:`_evaluate_live_points_jax`."""
    nloglike_leaves = loglike_spec.nleaves

    def evaluate(u, *leaves):
        loglike_fn = loglike_spec.rebuild(leaves[:nloglike_leaves])
        prior_fn = prior_spec.rebuild(leaves[nloglike_leaves:])

        def evaluate_chunk(u_chunk):
            return _evaluate_jax_batch(loglike_fn, prior_fn, u_chunk, ndim)

        return lax.map(evaluate_chunk, u)

    return jax.jit(evaluate)


def _remaining_delta_logz(logz_dead, logx_final, live_logl):
    """Return the live-evidence remainder in log-evidence units."""

    if math.isinf(float(logz_dead)) and float(logz_dead) < 0.0:
        return math.inf
    logz_remain = float(logx_final) + float(jnp.max(live_logl))
    return float(jnp.logaddexp(logz_dead, logz_remain) - logz_dead)



def _logzerr_diagnostics(
    logwt,
    logl,
    logz: float,
    nlive: int,
    nlive_final: int,
) -> tuple[float, dict[str, object]]:
    """Return ``logzerr`` and diagnostics for nonfinite inputs."""

    logwt = jnp.asarray(logwt)
    logl = jnp.asarray(logl)
    npoints = int(logwt.size)
    nlive_final = max(0, min(int(nlive_final), npoints))
    ndead = npoints - nlive_final

    finite_logl = jnp.isfinite(logl)
    finite_logwt = jnp.isfinite(logwt)
    finite_pair = finite_logl & finite_logwt
    diagnostics: dict[str, object] = {
        "logzerr_status": "ok",
        "information_H": math.nan,
        "n_nonfinite_logl": int(jnp.sum(~finite_logl)),
        "n_nonfinite_logwt": int(jnp.sum(~finite_logwt)),
        "n_nonfinite_weights": 0,
        "n_dead_finite": int(jnp.sum(finite_pair[:ndead])),
        "n_live_finite": int(jnp.sum(finite_pair[ndead:])),
    }

    if nlive <= 0:
        diagnostics["logzerr_status"] = "invalid_nlive"
        return math.nan, diagnostics
    if not math.isfinite(float(logz)):
        diagnostics["logzerr_status"] = "nonfinite_logz"
        return math.nan, diagnostics

    raw_weights = jnp.exp(logwt - logz)
    finite_weights = jnp.isfinite(raw_weights)
    diagnostics["n_nonfinite_weights"] = int(jnp.sum(~finite_weights))
    if diagnostics["n_nonfinite_weights"]:
        diagnostics["logzerr_status"] = "nonfinite_posterior_weights"
        return math.nan, diagnostics

    contributing = (raw_weights > 0.0) & finite_logl
    if bool(jnp.any((raw_weights > 0.0) & ~finite_logl)):
        diagnostics["logzerr_status"] = "nonfinite_weighted_logl"
        return math.nan, diagnostics
    if not bool(jnp.any(contributing)):
        diagnostics["logzerr_status"] = "no_finite_weighted_samples"
        diagnostics["information_H"] = 0.0
        return math.nan, diagnostics

    information = jnp.sum(
        jnp.where(contributing, raw_weights * (logl - logz), 0.0)
    )
    information = jnp.maximum(information, 0.0)
    diagnostics["information_H"] = float(information)
    if not math.isfinite(float(information)):
        diagnostics["logzerr_status"] = "nonfinite_information_H"
        return math.nan, diagnostics

    return float(jnp.sqrt(information / nlive)), diagnostics



def _update_scale(current_scale: float, observed_accept: float) -> float:
    """Return the log-space adapted live-cov step scale, clamped to bounds."""
    log_scale = math.log(float(current_scale))
    delta = _SCALE_RATE * float(np.clip(observed_accept - _TARGET_ACCEPT, -0.5, 0.5))
    new_scale = math.exp(log_scale + delta)
    return float(np.clip(new_scale, _MIN_SCALE, _MAX_SCALE))


@functools.lru_cache(maxsize=32)
def _make_static_jax_rwalk_block_kernel_cached(
    loglike_spec,
    prior_spec,
    ndim: int,
    walks: int,
    replacement_chains: int,
    block_size: int,
    cluster_swap: bool = False,
):
    """Return a cached jitted block kernel: ``block_size`` NS iterations.

    The cache is keyed on the callables' specs (see
    :class:`tinyns.samplers._CallableSpec`), so pytree callables that differ
    only in their array leaves share one compiled kernel. The returned
    ``block_kernel`` takes the array leaves of pytree callables and the large
    constants of closures (``*_callable_leaves(loglike, prior_transform,
    ndim)``) as trailing arguments and threads them to the rwalk kernel, so
    they are jit arguments rather than compiled-in constants. Callables with
    neither take no trailing arguments.

    After a failed replacement the remaining iterations of the block are
    skipped: they neither touch the live set nor advance the key.

    With ``cluster_swap=True`` the chains also propose affine swaps between
    clusters (see :mod:`tinyns.clusters`): ``block_kernel`` takes the cluster
    frames before the callable leaves and returns one more output, the
    per-iteration ``[accepted swaps, proposed swaps]``. A replaced live point
    leaves its cluster's statistics (its label becomes -1).
    """

    ndim = int(ndim)
    walks = int(walks)
    replacement_chains = int(replacement_chains)
    block_size = int(block_size)
    batch_ncall = walks * replacement_chains
    rwalk_kernel = _make_rwalk_jax_kernel_cached(
        loglike_spec,
        prior_spec,
        ndim,
        walks,
        replacement_chains,
        cluster_swap,
    )

    def block_kernel(
        key,
        live_u,
        live_theta,
        live_logl,
        logz_dead,
        start_iteration,
        nlive,
        scale,
        max_batches,
        *callable_leaves,
    ):
        if cluster_swap:
            clusters, *callable_leaves = callable_leaves
            labels = [clusters["labels"]]
        else:
            labels = []

        def one_iteration(carry, offset):
            key, live_u, live_theta, live_logl, logz_dead, active, *labels = carry

            def run_iteration(state):
                key, live_u, live_theta, live_logl, logz_dead, *labels = state
                worst = jnp.argmin(live_logl)
                dead_u = live_u[worst]
                dead_theta = live_theta[worst]
                logl_worst = live_logl[worst]
                iteration = jnp.asarray(start_iteration, dtype=jnp.int32) + offset
                logx_prev = -iteration / nlive
                logx_new = -(iteration + 1) / nlive
                logwidth = logx_prev + jnp.log1p(
                    -jnp.exp(logx_new - logx_prev)
                )
                logwt = logwidth + logl_worst
                logz_dead = jnp.logaddexp(logz_dead, logwt)

                (
                    key,
                    new_u,
                    new_theta,
                    new_logl,
                    replacement_ncall,
                    accepted,
                    accepted_move_count,
                    total_proposal_count,
                    *swap_counts,
                ) = rwalk_kernel(
                    key,
                    logl_worst,
                    live_u,
                    live_logl,
                    jnp.asarray(scale),
                    jnp.asarray(max_batches, dtype=jnp.int32),
                    *([{**clusters, "labels": labels[0]}] if cluster_swap else []),
                    *callable_leaves,
                )
                if cluster_swap:
                    labels = [labels[0].at[worst].set(-1)]
                replacement_batches_used = (
                    total_proposal_count
                    + jnp.asarray(batch_ncall - 1, dtype=jnp.int32)
                ) // jnp.asarray(batch_ncall, dtype=jnp.int32)
                insertion_index = (
                    jnp.sum(live_logl <= new_logl) - (logl_worst <= new_logl)
                ).astype(jnp.int32)
                live_u = jnp.where(accepted, live_u.at[worst].set(new_u), live_u)
                live_theta = jnp.where(
                    accepted, live_theta.at[worst].set(new_theta), live_theta
                )
                live_logl = jnp.where(
                    accepted, live_logl.at[worst].set(new_logl), live_logl
                )
                return (
                    key,
                    live_u,
                    live_theta,
                    live_logl,
                    logz_dead,
                    accepted,
                    *labels,
                ), (
                    dead_u,
                    dead_theta,
                    logl_worst,
                    logwt,
                    replacement_ncall,
                    insertion_index,
                    replacement_batches_used,
                    accepted,
                    accepted_move_count,
                    total_proposal_count,
                    *swap_counts,
                )

            def skip_iteration(state):
                key, live_u, live_theta, live_logl, logz_dead, *labels = state
                worst = jnp.argmin(live_logl)
                dead_u = live_u[worst]
                dead_theta = live_theta[worst]
                logl_worst = live_logl[worst]
                zero = jnp.asarray(0, dtype=jnp.int32)
                return (
                    key,
                    live_u,
                    live_theta,
                    live_logl,
                    logz_dead,
                    jnp.asarray(False),
                    *labels,
                ), (
                    dead_u,
                    dead_theta,
                    logl_worst,
                    jnp.asarray(-jnp.inf, dtype=live_logl.dtype),
                    zero,
                    zero,
                    zero,
                    jnp.asarray(False),
                    zero,
                    zero,
                    *([jnp.zeros(2, dtype=jnp.int32)] if cluster_swap else []),
                )

            return lax.cond(
                active,
                run_iteration,
                skip_iteration,
                (key, live_u, live_theta, live_logl, logz_dead, *labels),
            )

        (
            (
                new_key,
                new_live_u,
                new_live_theta,
                new_live_logl,
                logz_dead_new,
                *_,
            ),
            block,
        ) = lax.scan(
            one_iteration,
            (
                key,
                live_u,
                live_theta,
                live_logl,
                jnp.asarray(logz_dead),
                jnp.asarray(True),
                *labels,
            ),
            jnp.arange(block_size, dtype=jnp.int32),
        )
        logx_final_new = -(
            jnp.asarray(start_iteration, dtype=jnp.float32)
            + jnp.asarray(block_size, dtype=jnp.float32)
        ) / jnp.asarray(nlive, dtype=jnp.float32)
        return (
            new_key,
            new_live_u,
            new_live_theta,
            new_live_logl,
            *block,
            logz_dead_new,
            logx_final_new,
        )

    return jax.jit(block_kernel)


def _make_static_jax_rwalk_block_kernel(
    loglike,
    prior_transform,
    ndim: int,
    walks: int,
    replacement_chains: int,
    block_size: int,
    cluster_swap: bool = False,
):
    """Return the cached block kernel for these callables' specs."""
    return _make_static_jax_rwalk_block_kernel_cached(
        *_callable_specs(loglike, prior_transform, ndim),
        ndim,
        walks,
        replacement_chains,
        block_size,
        cluster_swap,
    )


_make_static_jax_rwalk_block_kernel.cache_clear = (
    _make_static_jax_rwalk_block_kernel_cached.cache_clear
)
_make_static_jax_rwalk_block_kernel.cache_info = (
    _make_static_jax_rwalk_block_kernel_cached.cache_info
)


def _make_run_state(
    *,
    iteration: int,
    logz: float,
    dlogz: float,
    ncall: int,
    logl_min: float,
    logl_live_max: float,
    nlive: int,
    ndim: int,
    replacement_ncall: list[int],
    replacement_failures: int,
    replacement_batches: list[int] | None = None,
    replacement_chains: int | None = None,
    walks: int | None = None,
    calls_per_s: float | None = None,
) -> dict[str, object]:
    if replacement_ncall:
        replacement_mean_ncall_so_far = float(
            sum(replacement_ncall) / len(replacement_ncall)
        )
    else:
        replacement_mean_ncall_so_far = None
    if replacement_batches:
        replacement_mean_batches_so_far = float(
            sum(replacement_batches) / len(replacement_batches)
        )
        replacement_max_batches_so_far = int(max(replacement_batches))
    else:
        replacement_mean_batches_so_far = None
        replacement_max_batches_so_far = None
    return {
        "iter": int(iteration),
        "logz": float(logz),
        "dlogz": float(dlogz),
        "ncall": int(ncall),
        "calls_per_s": calls_per_s,
        "logl_min": float(logl_min),
        "logl_live_max": float(logl_live_max),
        "nlive": int(nlive),
        "ndim": int(ndim),
        "replacement_mean_ncall_so_far": replacement_mean_ncall_so_far,
        "replacement_mean_batches_so_far": replacement_mean_batches_so_far,
        "replacement_max_batches_so_far": replacement_max_batches_so_far,
        "replacement_chains": replacement_chains,
        "walks": walks,
        "replacement_failures": int(replacement_failures),
    }


def _format_progress_line(state: dict[str, object]) -> str:
    """Format one dependency-free progress line for a run state."""

    rate = state.get("calls_per_s")
    rate_text = "n/a" if rate is None else f"{float(rate):.3g}"
    repl = state.get("replacement_mean_ncall_so_far")
    repl_text = "n/a" if repl is None else f"{float(repl):.1f}"
    batches = state.get("replacement_mean_batches_so_far")
    batches_text = "n/a" if batches is None else f"{float(batches):.2f}"
    return (
        f"iter={int(state['iter']):05d} "
        f"logz={float(state['logz']):.3f} "
        f"dlogz={float(state['dlogz']):.3f} "
        f"ncall={int(state['ncall'])} "
        f"calls/s={rate_text} "
        f"logl_min={float(state['logl_min']):.3g} "
        f"logl_live_max={float(state['logl_live_max']):.3g} "
        f"repl_ncall={repl_text} "
        f"repl_batches={batches_text}"
    )


class _ProgressPrinter:
    """Print dependency-free progress updates without ANSI escape sequences."""

    def __init__(self) -> None:
        self._last_len = 0

    def print(self, line: str, *, final: bool = False) -> None:
        padding = " " * max(0, self._last_len - len(line))
        end = "\n" if final else "\r"
        print("\r" + line + padding, end=end, flush=True)
        self._last_len = 0 if final else len(line)



def run_static_nested(
    key: PRNGKeyLike,
    loglike: LogLikelihood,
    prior_transform: PriorTransform,
    ndim: int,
    nlive: int,
    *,
    dlogz: float = 0.1,
    maxiter: int | None = None,
    progress: bool = False,
    progress_interval: int = 100,
    callback=None,
    callback_interval: int = 100,
    walks: int | None = None,
    replacement_chains: int = 1,
    block_size: int = 32,
    cluster_swap: bool | None = None,
    initial_state: NestedRunState | None = None,
    checkpoint_path=None,
    checkpoint_interval: int = 100,
):
    """Run static nested sampling with the live-cov rwalk.

    ``loglike`` and ``prior_transform`` must be JAX-traceable scalar functions
    (one point in, one point or one value out); they are ``jax.vmap``-ped
    where needed. Every ``block_size`` iterations run as one jitted block.
    ``walks`` defaults to ``max(25, 6 * ndim)`` (12 for ``ndim=1``).

    ``cluster_swap`` (default: on with one replacement chain) tracks clusters
    of the live points between blocks and, once two clusters are large
    enough, lets the chains swap between them so that mode weights do not
    drift with the seed (see :mod:`tinyns.clusters`). Pass
    ``cluster_swap=False`` to opt out.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if nlive <= 0:
        raise ValueError("nlive must be a positive integer")
    options = _resolve_options(
        ndim,
        walks=walks,
        replacement_chains=replacement_chains,
        block_size=block_size,
        cluster_swap=cluster_swap,
    )
    walks = options["walks"]
    cluster_swap = options["cluster_swap"]
    if progress_interval <= 0:
        raise ValueError("progress_interval must be a positive integer")
    if callback_interval <= 0:
        raise ValueError("callback_interval must be a positive integer")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    if checkpoint_path is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")
    if maxiter is not None and (
        not isinstance(maxiter, int)
        or isinstance(maxiter, bool)
        or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer")
    if maxiter is None:
        maxiter = 10_000 * ndim

    config = {"ndim": int(ndim), "nlive": int(nlive), **options}
    batch_ncall = int(walks) * int(replacement_chains)
    max_batches = max(1, _MAX_REPLACEMENT_CALLS // batch_ncall)
    checkpoint_path_str = (
        None if checkpoint_path is None else os.fspath(checkpoint_path)
    )
    resumed_from_checkpoint = initial_state is not None
    restored_telemetry = (
        dict(getattr(initial_state, "telemetry", {}) or {})
        if initial_state is not None
        else {}
    )

    # Wall-time telemetry: the first block carries the compiles, so
    # throughput is measured from its end.
    sampling_start = time.perf_counter()
    first_block = None  # (seconds since sampling_start, ncall) after block 1

    # Array leaves of pytree callables and large closure constants, placed on
    # the device once per run and passed to the compiled kernels as arguments
    # (empty for callables with neither).
    callable_leaves = tuple(
        jnp.asarray(leaf) for leaf in _callable_leaves(loglike, prior_transform, ndim)
    )

    if initial_state is None:
        key = random.PRNGKey(int(key)) if isinstance(key, int) else key
        key, init_key = random.split(key)
        live_u = random.uniform(init_key, shape=(nlive, ndim))
        live_theta, live_logl = _evaluate_live_points_jax(
            loglike,
            prior_transform,
            live_u,
            ndim,
            chunk_size=replacement_chains,
            callable_leaves=callable_leaves,
        )
        ncall = nlive
        logz_dead = -math.inf
        replacement_ncall = []
        insertion_indices = []
        replacement_failures = 0
        replacement_batches = []
        logx_final = 0.0
        iteration = 0
    else:
        key = initial_state.key
        live_u = initial_state.live_u
        live_theta = initial_state.live_theta
        live_logl = initial_state.live_logl
        checkpoint_dead_count = len(initial_state.dead_logl)
        if not (
            len(initial_state.dead_u)
            == len(initial_state.dead_theta)
            == checkpoint_dead_count
            == len(initial_state.dead_logwt)
        ):
            raise ValueError("checkpoint dead point arrays have inconsistent lengths")
        logz_dead = float(initial_state.logz_dead)
        logx_final = float(initial_state.logx_final)
        ncall = int(initial_state.ncall)
        replacement_ncall = list(initial_state.replacement_ncall)
        insertion_indices = list(initial_state.insertion_indices)
        replacement_failures = int(initial_state.replacement_failures)
        replacement_batches = [
            int(value)
            for value in restored_telemetry.get("replacement_batches", [])
        ]
        iteration = int(initial_state.iteration)
        if maxiter < iteration:
            raise ValueError(
                f"maxiter={maxiter} is smaller than checkpoint iteration={iteration}"
            )
        if checkpoint_dead_count != iteration:
            raise ValueError(
                "checkpoint dead point count must match checkpoint iteration; "
                f"got {checkpoint_dead_count} dead points and iteration={iteration}"
            )
    success = True
    message = "converged"
    stopped_by_callback = False

    dead_u_storage = np.empty((maxiter, ndim), dtype=np.asarray(live_u).dtype)
    dead_theta_storage = np.empty((maxiter, ndim), dtype=np.asarray(live_theta).dtype)
    dead_logl_storage = np.empty((maxiter,), dtype=np.asarray(live_logl).dtype)
    dead_logwt_storage = np.empty((maxiter,), dtype=np.asarray(live_logl).dtype)
    if initial_state is not None and iteration:
        dead_u_storage[:iteration] = np.asarray(jnp.stack(initial_state.dead_u))
        dead_theta_storage[:iteration] = np.asarray(jnp.stack(initial_state.dead_theta))
        dead_logl_storage[:iteration] = np.asarray(initial_state.dead_logl)
        dead_logwt_storage[:iteration] = np.asarray(initial_state.dead_logwt)

    initial_iteration = iteration
    final_delta_logz = math.inf
    terminated_after_partial_block_failure = False
    partial_block_failure_delta_logz = None
    partial_block_failure_offset = None
    partial_block_failure_message = None
    progress_printer = _ProgressPrinter() if progress else None
    rwalk_accepted_move_history = [
        int(value)
        for value in restored_telemetry.get("rwalk_accepted_move_history", [])
    ]
    rwalk_proposal_history = [
        int(value)
        for value in restored_telemetry.get("rwalk_proposal_history", [])
    ]
    scale = _INITIAL_SCALE
    if initial_state is not None:
        restored_scale = getattr(initial_state, "scale", None)
        if restored_scale is not None and math.isfinite(float(restored_scale)):
            scale = float(restored_scale)
    adaptive_scale_history = [
        float(value)
        for value in restored_telemetry.get("adaptive_scale_history", [])
    ]
    if not adaptive_scale_history or adaptive_scale_history[-1] != scale:
        adaptive_scale_history.append(scale)
    adaptive_accept_history = [
        float(value)
        for value in restored_telemetry.get("adaptive_accept_history", [])
    ]
    adaptive_updates = int(restored_telemetry.get("adaptive_updates", 0))
    # Clusters are tracked on the host between blocks; the tracker never
    # touches the PRNG stream, and the swap kernel only runs (and is only
    # compiled) once two clusters can swap.
    tracker = None
    cluster_host_s = 0.0
    if cluster_swap:
        stored = getattr(initial_state, "clusters", None) or {}
        tracker = ClusterTracker(nlive, stored.get("arrays"), stored.get("log"))

    def update_scale(observed_accept: float) -> None:
        nonlocal scale, adaptive_updates
        if not math.isfinite(float(observed_accept)):
            return
        adaptive_accept_history.append(float(observed_accept))
        scale = _update_scale(scale, float(observed_accept))
        adaptive_scale_history.append(scale)
        adaptive_updates += 1

    def current_state() -> NestedRunState:
        return NestedRunState(
            key=key,
            live_u=live_u,
            live_theta=live_theta,
            live_logl=live_logl,
            dead_u=[jnp.asarray(point) for point in dead_u_storage[:iteration]],
            dead_theta=[jnp.asarray(point) for point in dead_theta_storage[:iteration]],
            dead_logl=[float(x) for x in dead_logl_storage[:iteration]],
            dead_logwt=[float(x) for x in dead_logwt_storage[:iteration]],
            logz_dead=logz_dead,
            logx_final=logx_final,
            ncall=ncall,
            replacement_ncall=replacement_ncall,
            insertion_indices=insertion_indices,
            replacement_failures=replacement_failures,
            iteration=iteration,
            success=success,
            message=message,
            stopped_by_callback=stopped_by_callback,
            scale=scale,
            clusters=(
                None
                if tracker is None
                else {"log": tracker.log, "arrays": tracker.arrays()}
            ),
            telemetry={
                "replacement_batches": list(replacement_batches),
                "rwalk_accepted_move_history": list(
                    rwalk_accepted_move_history
                ),
                "rwalk_proposal_history": list(rwalk_proposal_history),
                "adaptive_scale_history": list(adaptive_scale_history),
                "adaptive_accept_history": list(adaptive_accept_history),
                "adaptive_updates": int(adaptive_updates),
            },
        )

    last_checkpoint_iteration = initial_iteration

    def maybe_checkpoint(*, final: bool = False) -> None:
        nonlocal last_checkpoint_iteration
        if checkpoint_path_str is None:
            return
        if final or (iteration - last_checkpoint_iteration) >= checkpoint_interval:
            save_checkpoint_npz(checkpoint_path_str, current_state(), config)
            last_checkpoint_iteration = iteration

    def build_state(
        *,
        iteration: int,
        logz: float,
        dlogz: float,
        ncall: int,
        logl_min: float,
        logl_live_max: float,
    ) -> dict[str, object]:
        """Assemble a run-state dict, closing over the constant config args.

        Called once per finished block; the first call marks the end of the
        first block for the wall-time telemetry.
        """
        nonlocal first_block
        elapsed = time.perf_counter() - sampling_start
        if first_block is None:
            first_block = (elapsed, ncall)
        calls = ncall - first_block[1]
        seconds = elapsed - first_block[0]
        return _make_run_state(
            iteration=iteration,
            logz=logz,
            dlogz=dlogz,
            ncall=ncall,
            logl_min=logl_min,
            logl_live_max=logl_live_max,
            nlive=nlive,
            ndim=ndim,
            replacement_ncall=replacement_ncall,
            replacement_failures=replacement_failures,
            replacement_batches=replacement_batches,
            replacement_chains=replacement_chains,
            walks=walks,
            calls_per_s=calls / seconds if calls > 0 and seconds > 0 else None,
        )

    # A resumed run may already be terminal: converged, or checkpointed at
    # iteration >= maxiter without converging. Label it here so the loop is
    # not entered and the result does not carry the neutral
    # `success=True, message="converged"` initial values.
    resumed_terminal = False
    if iteration > 0:
        delta_logz = _remaining_delta_logz(logz_dead, logx_final, live_logl)
        final_delta_logz = delta_logz
        if delta_logz < dlogz:
            success = True
            message = "converged"
            maybe_checkpoint(final=True)
            resumed_terminal = True
        elif iteration >= maxiter:
            success = False
            message = f"maxiter={maxiter} reached"
            maybe_checkpoint(final=True)
            resumed_terminal = True

    while not resumed_terminal and iteration < maxiter:
        if iteration > 0:
            delta_logz = _remaining_delta_logz(logz_dead, logx_final, live_logl)
            final_delta_logz = delta_logz
            if delta_logz < dlogz:
                success = True
                message = "converged"
                maybe_checkpoint(final=True)
                break
        block_size_now = min(int(block_size), maxiter - iteration)
        logz_dead_before_block = logz_dead
        frames = None
        if tracker is not None:
            host_start = time.perf_counter()
            frames = tracker.frames(
                live_u, iteration, dead_u_storage, dead_logwt_storage
            )
            cluster_host_s += time.perf_counter() - host_start
        block_kernel = _make_static_jax_rwalk_block_kernel(
            loglike,
            prior_transform,
            int(ndim),
            int(walks),
            int(replacement_chains),
            int(block_size_now),
            frames is not None,
        )
        (
            key,
            live_u,
            live_theta,
            live_logl,
            dead_u_block,
            dead_theta_block,
            dead_logl_block,
            dead_logwt_block,
            replacement_ncall_block,
            insertion_indices_block,
            replacement_batches_block,
            accepted_block,
            accepted_move_count_block,
            total_proposal_count_block,
            *swap_block,
            logz_dead,
            logx_final,
        ) = block_kernel(
            key,
            live_u,
            live_theta,
            live_logl,
            jnp.asarray(logz_dead),
            jnp.asarray(iteration, dtype=jnp.int32),
            jnp.asarray(nlive, dtype=jnp.int32),
            jnp.asarray(scale),
            jnp.asarray(max_batches, dtype=jnp.int32),
            *([] if frames is None else [frames]),
            *callable_leaves,
        )
        block_start = iteration
        failed_offsets = np.flatnonzero(~np.asarray(accepted_block))
        if failed_offsets.size:
            # The block kernel left the live set and the key as they were
            # after the last successful replacement; keep the dead points
            # before the failed one.
            replacement_failures += 1
            success = False
            partial_block_failure_offset = int(failed_offsets[0])
            message = (
                "replacement failed at iteration "
                f"{block_start + partial_block_failure_offset + 1}: no chain "
                f"ended above the likelihood threshold in {max_batches} "
                f"batches of {walks} steps x {replacement_chains} chains"
            )
            partial_block_failure_message = message
            block_size_now = partial_block_failure_offset
            ncall += int(
                np.asarray(replacement_ncall_block)[partial_block_failure_offset]
            )
            logz_dead = float(logz_dead_before_block)
            for logwt_value in np.asarray(dead_logwt_block[:block_size_now]):
                logz_dead = float(jnp.logaddexp(logz_dead, float(logwt_value)))
            logx_final = -(block_start + block_size_now) / int(nlive)
        block_stop = block_start + block_size_now
        dead_u_block = dead_u_block[:block_size_now]
        dead_theta_block = dead_theta_block[:block_size_now]
        dead_logl_block = dead_logl_block[:block_size_now]
        dead_logwt_block = dead_logwt_block[:block_size_now]
        replacement_ncall_block = replacement_ncall_block[:block_size_now]
        insertion_indices_block = insertion_indices_block[:block_size_now]
        replacement_batches_block = replacement_batches_block[:block_size_now]
        accepted_move_count_block = accepted_move_count_block[:block_size_now]
        total_proposal_count_block = total_proposal_count_block[:block_size_now]
        if swap_block:
            # The rwalk telemetry and the scale adaptation count rwalk steps
            # only; swaps are counted on their own.
            swaps = np.asarray(swap_block[0])[:block_size_now]
            accepted_move_count_block = (
                np.asarray(accepted_move_count_block) - swaps[:, 0]
            )
            total_proposal_count_block = (
                np.asarray(total_proposal_count_block) - swaps[:, 1]
            )
            tracker.log["swap"] = [
                int(a + b)
                for a, b in zip(tracker.log["swap"], swaps.sum(0), strict=True)
            ]
        dead_u_storage[block_start:block_stop] = np.asarray(dead_u_block)
        dead_theta_storage[block_start:block_stop] = np.asarray(dead_theta_block)
        dead_logl_storage[block_start:block_stop] = np.asarray(dead_logl_block)
        dead_logwt_storage[block_start:block_stop] = np.asarray(dead_logwt_block)
        block_ncalls = [int(x) for x in np.asarray(replacement_ncall_block)]
        replacement_ncall.extend(block_ncalls)
        insertion_indices.extend(int(x) for x in np.asarray(insertion_indices_block))
        ncall += int(sum(block_ncalls))
        replacement_batches.extend(
            int(x) for x in np.asarray(replacement_batches_block)
        )
        rwalk_accepted_move_history.extend(
            int(x) for x in np.asarray(accepted_move_count_block)
        )
        rwalk_proposal_history.extend(
            int(x) for x in np.asarray(total_proposal_count_block)
        )
        total_moves = int(np.sum(np.asarray(accepted_move_count_block)))
        total_proposals = int(np.sum(np.asarray(total_proposal_count_block)))
        if total_proposals > 0:
            update_scale(total_moves / total_proposals)
        iteration = block_stop
        maybe_checkpoint()
        if failed_offsets.size:
            delta_logz = _remaining_delta_logz(logz_dead, logx_final, live_logl)
            final_delta_logz = delta_logz
            partial_block_failure_delta_logz = float(delta_logz)
            if iteration > 0 and delta_logz < dlogz:
                success = True
                message = "converged after partial block before replacement failure"
                terminated_after_partial_block_failure = True
            maybe_checkpoint(final=True)
            break

        delta_logz = _remaining_delta_logz(logz_dead, logx_final, live_logl)
        final_delta_logz = delta_logz
        final_iteration = delta_logz < dlogz or iteration == maxiter
        if iteration == maxiter and delta_logz >= dlogz:
            success = False
            message = f"maxiter={maxiter} reached"
        state = build_state(
            iteration=iteration,
            logz=logz_dead,
            dlogz=delta_logz,
            ncall=ncall,
            logl_min=float(dead_logl_block[-1]),
            logl_live_max=float(jnp.max(live_logl)),
        )
        if callback is not None and (
            iteration == 1 or iteration % callback_interval == 0 or final_iteration
        ):
            if callback(state) is False:
                success = False
                message = "stopped by callback"
                stopped_by_callback = True
                final_iteration = True
        if progress_printer is not None and (
            iteration == 1 or iteration % progress_interval == 0 or final_iteration
        ):
            progress_printer.print(
                _format_progress_line(state), final=final_iteration
            )
        if final_iteration:
            maybe_checkpoint(final=True)
            break

    wall_time_s = time.perf_counter() - sampling_start
    compile_s = None if first_block is None else first_block[0]
    calls_after_first_block = 0 if first_block is None else ncall - first_block[1]
    mean_ms_per_call = (
        1000.0 * (wall_time_s - compile_s) / calls_after_first_block
        if calls_after_first_block > 0
        else None
    )

    live_logwt = logx_final - math.log(nlive) + live_logl

    dead_u_arr = jnp.asarray(dead_u_storage[:iteration])
    dead_theta_arr = jnp.asarray(dead_theta_storage[:iteration])
    dead_logl_arr = jnp.asarray(dead_logl_storage[:iteration])
    dead_logwt_arr = jnp.asarray(dead_logwt_storage[:iteration])
    if iteration:
        samples_u = jnp.concatenate([dead_u_arr, live_u], axis=0)
        samples = jnp.concatenate([dead_theta_arr, live_theta], axis=0)
        logl = jnp.concatenate([dead_logl_arr, live_logl], axis=0)
        logwt = jnp.concatenate([dead_logwt_arr, live_logwt], axis=0)
    else:
        samples_u = live_u
        samples = live_theta
        logl = live_logl
        logwt = live_logwt

    niter = int(iteration)
    nlive_final = int(live_logl.size)
    nposterior = int(logwt.size)
    logz = float(logsumexp(logwt))
    logzerr, logzerr_diagnostics = _logzerr_diagnostics(
        logwt, logl, logz, nlive, nlive_final
    )
    if replacement_ncall:
        mean_replacement_ncall = float(sum(replacement_ncall) / len(replacement_ncall))
        max_replacement_ncall = int(max(replacement_ncall))
        replacement_acceptance_proxy = (
            1.0 / mean_replacement_ncall
            if math.isfinite(mean_replacement_ncall) and mean_replacement_ncall > 0.0
            else 0.0
        )
    else:
        mean_replacement_ncall = 0.0
        max_replacement_ncall = 0
        replacement_acceptance_proxy = 0.0
    accepted_rwalk_moves = int(sum(rwalk_accepted_move_history))
    total_rwalk_proposals = int(sum(rwalk_proposal_history))
    rwalk_acceptance = (
        accepted_rwalk_moves / total_rwalk_proposals
        if total_rwalk_proposals > 0
        else None
    )

    cluster_metadata = {"cluster_swap": bool(cluster_swap)}
    if tracker is not None:
        host_start = time.perf_counter()
        cluster_metadata.update(
            tracker.summary(
                dead_u_storage[:iteration],
                dead_logwt_storage[:iteration],
                np.asarray(live_u),
                np.asarray(live_logwt),
            )
        )
        cluster_metadata["cluster_host_s"] = (
            cluster_host_s + time.perf_counter() - host_start
        )

    return NestedSamplingResult(
        samples_u=samples_u,
        samples=samples,
        logl=logl,
        logwt=logwt,
        logz=logz,
        logzerr=logzerr,
        ncall=ncall,
        nlive=nlive,
        ndim=ndim,
        success=success,
        message=message,
        metadata={
            "rwalk_scale_initial": _INITIAL_SCALE,
            "rwalk_scale_final": float(scale),
            "rwalk_scale_min_seen": float(min(adaptive_scale_history)),
            "rwalk_scale_max_seen": float(max(adaptive_scale_history)),
            "rwalk_scale_mean": float(
                sum(adaptive_scale_history) / len(adaptive_scale_history)
            ),
            "rwalk_adaptation_updates": int(adaptive_updates),
            "rwalk_observed_accept_mean": (
                float(sum(adaptive_accept_history) / len(adaptive_accept_history))
                if adaptive_accept_history
                else 0.0
            ),
            **cluster_metadata,
            # Wall time of this call (a resume counts only its own part); the
            # first block includes the compiles, so the per-call cost skips it.
            "wall_time_s": float(wall_time_s),
            "compile_s": compile_s,
            "mean_ms_per_call": mean_ms_per_call,
            "block_size": int(block_size),
            "dlogz": dlogz,
            "maxiter": maxiter,
            "niter": niter,
            "ndead": niter,
            "nlive_final": nlive_final,
            "nposterior": nposterior,
            **logzerr_diagnostics,
            "final_delta_logz": float(final_delta_logz),
            "final_logx": float(logx_final),
            "final_logz_dead": float(logz_dead),
            "final_logl_live_max": float(jnp.max(live_logl)),
            "walks": walks,
            "replacement_chains": replacement_chains,
            "replacement_batch_ncall": batch_ncall,
            "replacement_ncall": replacement_ncall,
            "insertion_indices": jnp.asarray(insertion_indices, dtype=int),
            "insertion_index_nslots": nlive,
            "insertion_index_nlive": nlive - 1,
            "replacement_failures": int(replacement_failures),
            "terminated_after_partial_block_failure": bool(
                terminated_after_partial_block_failure
            ),
            "partial_block_failure_delta_logz": partial_block_failure_delta_logz,
            "partial_block_failure_offset": partial_block_failure_offset,
            "partial_block_failure_message": partial_block_failure_message,
            "mean_replacement_ncall": mean_replacement_ncall,
            "max_replacement_ncall": max_replacement_ncall,
            "mean_replacement_batches": (
                float(sum(replacement_batches) / len(replacement_batches))
                if replacement_batches
                else 0.0
            ),
            "max_replacement_batches": int(max(replacement_batches, default=0)),
            "replacement_acceptance_proxy": replacement_acceptance_proxy,
            "accepted_rwalk_moves": accepted_rwalk_moves,
            "total_rwalk_proposals": total_rwalk_proposals,
            "rwalk_acceptance": rwalk_acceptance,
            "mean_rwalk_acceptance": rwalk_acceptance,
            "progress_interval": progress_interval,
            "callback_interval": callback_interval,
            "stopped_by_callback": bool(stopped_by_callback),
            "checkpoint_path": checkpoint_path_str,
            "checkpoint_interval": (
                checkpoint_interval if checkpoint_path_str is not None else None
            ),
            "resumed_from_checkpoint": bool(resumed_from_checkpoint),
            "initial_iteration": int(initial_iteration),
            "final_iteration": int(iteration),
        },
    )
