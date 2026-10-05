"""The pure-JAX core of :mod:`tinyns`: ``Config``, ``State``, ``init``, ``step``.

One :func:`step` is one jitted block of ``block_size`` nested-sampling
iterations. Each iteration kills the worst live point and replaces it with the
end of a live-cov random walk started from a live point above it (the chain
kernel, :func:`_chain_kernel`). ``loglike`` and ``prior_transform`` are passed
to every call and may be pytree callables: their array leaves (or a closure's
large constants) are jit arguments, and the compiled kernels are cached on
their structure (:mod:`tinyns.callables`).

The host loop (:func:`tinyns.loop.run`) owns everything between blocks:
termination, the step-scale adaptation, telemetry, checkpoints and the cluster
tracker of the swap move.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random

from tinyns.callables import _kernel_cache, _split_callables
from tinyns.clusters import loo_frames, swap_step

# The live-cov step is ``scale * L z``; the scale starts here and the host loop
# adapts it after every block (see tinyns.loop).
_INITIAL_SCALE = 0.5
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


@dataclasses.dataclass(frozen=True)
class Config:
    """Static sampler configuration (hashable; part of the compile cache key).

    ``walks`` defaults to ``max(25, 6 * ndim)`` (12 for ``ndim=1``).
    ``cluster_swap`` defaults to True with one replacement chain, the only
    configuration that supports it, and to False otherwise; an explicit True
    with more chains raises. It switches the host-side cluster tracker on;
    :func:`step` runs the swap move whenever it is given ``extras``.
    """

    ndim: int
    nlive: int
    walks: int | None = None
    replacement_chains: int = 1
    block_size: int = 32
    cluster_swap: bool | None = None

    def __post_init__(self):
        if self.ndim <= 0:
            raise ValueError("ndim must be a positive integer")
        if self.nlive <= 0:
            raise ValueError("nlive must be a positive integer")
        ndim = int(self.ndim)
        walks = self.walks
        if walks is None:
            # 1-D needs far fewer walks for unbiased logZ (validated to 10);
            # from 2-D on, max(25, 6 * ndim) (see CHANGELOG, v0.2.0 / v0.2.1).
            walks = 12 if ndim == 1 else max(25, 6 * ndim)
        _positive_int("walks", walks)
        _positive_int("replacement_chains", self.replacement_chains)
        _positive_int("block_size", self.block_size)
        cluster_swap = self.cluster_swap
        if cluster_swap is None:
            cluster_swap = self.replacement_chains == 1
        elif cluster_swap and self.replacement_chains != 1:
            raise NotImplementedError("cluster_swap=True needs replacement_chains=1")
        for name, value in (
            ("ndim", ndim),
            ("nlive", int(self.nlive)),
            ("walks", walks),
            ("cluster_swap", bool(cluster_swap)),
        ):
            object.__setattr__(self, name, value)

    @property
    def max_batches(self) -> int:
        """Chain batches one replacement may use before it fails."""
        return max(1, _MAX_REPLACEMENT_CALLS // (self.walks * self.replacement_chains))


class State(NamedTuple):
    """The live set and the run scalars, threaded through :func:`step`.

    ``logz`` and ``logx`` are the host loop's: after a failed replacement it
    recomputes both on the host (and may hold them, like ``scale``, as Python
    floats, which enter the jit exactly as ``jnp.asarray(value)`` would).
    """

    key: Any  # PRNG key, split in the v0.2.4 order
    u: Any  # (nlive, ndim) live points in the unit cube
    theta: Any  # (nlive, ndim) transformed live points
    logl: Any  # (nlive,)
    birth: Any  # (nlive,) contour each live point was born above; -inf at init
    logz: Any  # log evidence of the dead points
    logx: Any  # float32 log prior volume of the live set after the last block
    it: Any  # int32 iterations done
    scale: Any  # live-cov step multiplier
    ncall: Any  # int32 likelihood calls (out-of-cube proposals are not counted)
    failed: Any  # bool: a replacement found no point above the threshold


class Dead(NamedTuple):
    """One block of dead rows (leading axis ``block_size``).

    Rows from the first failed replacement on have ``valid`` False; the failed
    row still counts its likelihood calls in ``ncall``.
    """

    u: Any
    theta: Any
    logl: Any
    logwt: Any
    ncall: Any  # likelihood calls of the replacement
    insertion: Any  # insertion rank of the new point among the other live points
    batches: Any  # chain batches used
    valid: Any  # the replacement succeeded
    moves: Any  # accepted chain steps
    proposals: Any  # proposed chain steps
    birth: Any  # contour the dead point was born above
    swaps: Any = None  # [accepted, proposed] swap steps, with the swap move only


def _evaluate_jax_prior_batch(prior_transform, u_batch, ndim: int):
    """Evaluate a scalar JAX prior transform on a batch (``jax.vmap``)."""
    u_batch = jnp.asarray(u_batch)
    if u_batch.ndim != 2 or u_batch.shape[1] != ndim:
        raise ValueError(f"u_batch must have shape (batch, {ndim})")
    nbatch = int(u_batch.shape[0])
    theta_batch = jnp.asarray(jax.vmap(prior_transform)(u_batch))
    if ndim == 1 and theta_batch.shape == (nbatch,):
        theta_batch = theta_batch.reshape((nbatch, 1))
    if theta_batch.shape != (nbatch, ndim):
        raise ValueError(f"prior_transform must return shape ({ndim},)")
    return theta_batch


def _evaluate_jax_batch(loglike, prior_transform, u_batch, ndim):
    """Evaluate scalar JAX prior/likelihood functions on a unit-cube batch."""
    u_batch = jnp.asarray(u_batch)
    if u_batch.ndim != 2 or u_batch.shape[1] != ndim:
        raise ValueError(f"u_batch must have shape (batch, {ndim})")
    nbatch = int(u_batch.shape[0])
    theta_batch = _evaluate_jax_prior_batch(prior_transform, u_batch, ndim)
    logl_batch = jnp.asarray(jax.vmap(loglike)(theta_batch))
    if logl_batch.shape != (nbatch,):
        raise ValueError("loglike must return a scalar")
    return theta_batch, logl_batch


def live_cov_cholesky(live_u):
    """Return the Cholesky factor of the live-point covariance (unit cube).

    A small relative jitter keeps near-degenerate live sets factorable; if the
    factorization still fails, the per-axis standard deviations are used.
    """
    live_u = jnp.asarray(live_u)
    nlive, ndim = live_u.shape
    centered = live_u - jnp.mean(live_u, axis=0)
    cov = centered.T @ centered / max(nlive - 1, 1)
    mean_var = jnp.maximum(jnp.trace(cov) / ndim, jnp.finfo(cov.dtype).tiny)
    jitter = 10.0 * jnp.finfo(cov.dtype).eps * mean_var
    cov = cov + jitter * jnp.eye(ndim, dtype=cov.dtype)
    chol = jnp.linalg.cholesky(cov)
    fallback = jnp.diag(jnp.sqrt(jnp.diag(cov)))
    return jnp.where(jnp.all(jnp.isfinite(chol)), chol, fallback)


@_kernel_cache
def _chain_kernel(
    loglike_spec,
    prior_spec,
    ndim: int,
    walks: int,
    replacement_chains: int,
    cluster_swap: bool = False,
):
    """Return the cached jitted live-cov chain: one replacement point.

    :func:`step` calls it once per iteration. The returned jitted ``kernel``
    always produces an eight-element tuple, in this order:

    1. ``key`` -- the advanced PRNG key.
    2. ``new_u`` -- the accepted replacement point in unit-cube coordinates.
    3. ``new_theta`` -- the prior-transformed replacement point.
    4. ``new_logl`` -- the log-likelihood of ``new_theta``.
    5. ``ncall`` -- likelihood evaluations actually made. This is batches x
       walks x ``replacement_chains``, except that a single chain skips (and
       does not count) proposals that leave the unit cube.
    6. ``accepted`` -- whether a chain ended inside the constraint before the
       ``max_batches`` budget was exhausted.
    7. ``accepted_move_count`` -- accepted proposals summed over all chains and
       batches (rwalk-acceptance numerator).
    8. ``total_proposal_count`` -- total proposals attempted (batches x walks x
       ``replacement_chains``), the rwalk-acceptance denominator.

    Each step is ``scale * L N(0, I)`` with ``L`` the Cholesky factor of the
    live-point covariance, so the step follows the contracting, correlated
    live set. Moves that leave the cube are rejected. Chains start from live
    points strictly above ``logl_min`` whenever any exist. A batch runs
    ``replacement_chains`` chains of ``walks`` steps and keeps one chain that
    ends inside the constraint; an unmoved chain is kept as a copy of its seed
    (its seed is strictly above ``logl_min``). Only if no chain qualifies does
    the kernel run another batch, up to ``max_batches``.

    The callables arrive as specs (see :mod:`tinyns.callables`), so the cache
    holds no data. Their array leaves (pytree callables) or large constants
    (closures) are trailing arguments of ``kernel``: call it as
    ``kernel(key, ..., max_batches, *_callable_leaves(loglike,
    prior_transform, ndim))`` and the callables are rebuilt inside the trace,
    so the arrays are jit arguments rather than baked-in constants. Callables
    with neither take no trailing arguments.

    ``cluster_swap=True`` (single chain only) makes a fraction of the chain
    steps affine swaps between cluster frames (see :mod:`tinyns.clusters`).
    The kernel then takes the cluster frames as an extra argument before the
    callable leaves and returns a ninth output, ``[accepted swaps, proposed
    swaps]``; the swap steps are included in ``accepted_move_count`` and
    ``total_proposal_count``.
    """

    # With one chain, out-of-cube proposals skip the likelihood (lax.cond);
    # under vmap a cond would evaluate both branches.
    skip_out_of_cube = replacement_chains == 1
    if cluster_swap and not skip_out_of_cube:
        raise ValueError("cluster_swap needs a single replacement chain")
    nloglike_leaves = loglike_spec.nleaves
    nleaves = loglike_spec.nleaves + prior_spec.nleaves

    @jax.jit
    def kernel(
        key,
        logl_min,
        live_u,
        live_logl,
        scale,
        max_batches,
        *callable_leaves,
    ):
        if cluster_swap:
            clusters, *callable_leaves = callable_leaves
        if len(callable_leaves) != nleaves:
            raise ValueError(
                f"kernel expects {nleaves} callable array leaves, "
                f"got {len(callable_leaves)}"
            )
        loglike = loglike_spec.rebuild(callable_leaves[:nloglike_leaves])
        prior_transform = prior_spec.rebuild(callable_leaves[nloglike_leaves:])
        chol = live_cov_cholesky(live_u)
        # Never restart a chain from the point being replaced when others exist.
        above = live_logl > logl_min
        seed_logits = jnp.where(jnp.any(above), jnp.where(above, 0.0, -jnp.inf), 0.0)
        template_u = live_u[0]
        template_theta = _evaluate_jax_prior_batch(
            prior_transform, template_u[None, :], ndim
        )[0]
        initial_best_logl = jnp.asarray(-jnp.inf, dtype=live_logl.dtype)
        initial_ncall = jnp.asarray(0, dtype=jnp.int32)
        initial_done = jnp.asarray(False)
        initial_batch_index = jnp.asarray(0, dtype=jnp.int32)
        initial_accepted_move_count = jnp.asarray(0, dtype=jnp.int32)
        batch_ncall = jnp.asarray(walks * replacement_chains, dtype=jnp.int32)

        def cond(state):
            return (~state[2]) & (state[11] < max_batches)

        def body(state):
            (
                key,
                ncall,
                _done,
                _accepted,
                out_u,
                out_theta,
                out_logl,
                best_u,
                best_theta,
                best_logl,
                accepted_move_count,
                batch_index,
                n_evals,
                *swap_counts,
            ) = state

            key, seed_key = random.split(key)
            seed_idx = random.categorical(
                seed_key, seed_logits, shape=(replacement_chains,)
            )
            current_u = live_u[seed_idx]
            current_theta = _evaluate_jax_prior_batch(prior_transform, current_u, ndim)
            current_logl = live_logl[seed_idx]
            if cluster_swap:
                frames = loo_frames(clusters, seed_idx[0], current_u[0])
            attempt_best_u = current_u
            attempt_best_theta = current_theta
            attempt_best_logl = jnp.full(
                (replacement_chains,), -jnp.inf, live_logl.dtype
            )
            accepted_moves = jnp.zeros((replacement_chains,), dtype=jnp.int32)
            batch_evals = jnp.asarray(0, dtype=jnp.int32)

            def one_step(carry, _):
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                    batch_evals,
                ) = carry
                key, proposal_key = random.split(key)
                z = random.normal(proposal_key, shape=(replacement_chains, ndim))
                u_raw = current_u + scale * (z @ chol.T)
                if cluster_swap:
                    swap_u, swap_ok, is_swap = swap_step(
                        random.fold_in(proposal_key, 1), current_u[0], frames
                    )
                    u_raw = jnp.where(is_swap, swap_u[None, :], u_raw)
                in_cube = jnp.all((u_raw >= 0.0) & (u_raw <= 1.0), axis=1)
                if cluster_swap:
                    # A swap that fails its likelihood-free tests is rejected
                    # without a call, like an out-of-cube step.
                    in_cube = in_cube & (swap_ok | ~is_swap)
                u_prop = jnp.clip(u_raw, 0.0, 1.0)
                if skip_out_of_cube:

                    def evaluate(u):
                        theta, logl = _evaluate_jax_batch(
                            loglike, prior_transform, u, ndim
                        )
                        return (
                            theta.astype(current_theta.dtype),
                            logl.astype(current_logl.dtype),
                        )

                    def skip(u):
                        del u
                        return current_theta, jnp.full_like(current_logl, -jnp.inf)

                    theta_prop, logl_prop = lax.cond(in_cube[0], evaluate, skip, u_prop)
                    batch_evals = batch_evals + in_cube[0].astype(jnp.int32)
                else:
                    theta_prop, logl_prop = _evaluate_jax_batch(
                        loglike, prior_transform, u_prop, ndim
                    )
                    batch_evals = batch_evals + jnp.asarray(
                        replacement_chains, dtype=jnp.int32
                    )

                # Out-of-cube moves never count as the fallback best point.
                logl_prop = jnp.where(in_cube, logl_prop, -jnp.inf)
                is_best = logl_prop > attempt_best_logl
                attempt_best_u = jnp.where(is_best[:, None], u_prop, attempt_best_u)
                attempt_best_theta = jnp.where(
                    is_best[:, None], theta_prop, attempt_best_theta
                )
                attempt_best_logl = jnp.where(is_best, logl_prop, attempt_best_logl)

                # Out-of-cube moves are rejected even when logl_min is -inf.
                accept = (logl_prop >= logl_min) & in_cube
                current_u = jnp.where(accept[:, None], u_prop, current_u)
                current_theta = jnp.where(accept[:, None], theta_prop, current_theta)
                current_logl = jnp.where(accept, logl_prop, current_logl)
                accepted_moves = accepted_moves + accept.astype(jnp.int32)
                return (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                    batch_evals,
                ), (jnp.stack([is_swap & accept[0], is_swap]) if cluster_swap else None)

            (
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                    batch_evals,
                ),
                swap_steps,
            ) = lax.scan(
                one_step,
                (
                    key,
                    current_u,
                    current_theta,
                    current_logl,
                    attempt_best_u,
                    attempt_best_theta,
                    attempt_best_logl,
                    accepted_moves,
                    batch_evals,
                ),
                xs=None,
                length=walks,
            )

            batch_best_idx = jnp.argmax(attempt_best_logl)
            batch_best_logl = attempt_best_logl[batch_best_idx]
            is_global_best = batch_best_logl > best_logl
            best_u = jnp.where(is_global_best, attempt_best_u[batch_best_idx], best_u)
            best_theta = jnp.where(
                is_global_best, attempt_best_theta[batch_best_idx], best_theta
            )
            best_logl = jnp.where(is_global_best, batch_best_logl, best_logl)

            # A chain succeeds if it ends inside the constraint. An unmoved
            # chain is kept as a copy of its seed, provided the seed is
            # strictly above logl_min; discarding unmoved chains would
            # under-sample hard-to-move regions.
            success_mask = jnp.where(
                accepted_moves > 0, current_logl >= logl_min, current_logl > logl_min
            )
            any_success = jnp.any(success_mask)
            key, select_key = random.split(key)
            selection_scores = jnp.where(
                success_mask, random.uniform(select_key, (replacement_chains,)), -1.0
            )
            selected_idx = jnp.argmax(selection_scores)
            out_u = jnp.where(any_success, current_u[selected_idx], out_u)
            out_theta = jnp.where(any_success, current_theta[selected_idx], out_theta)
            out_logl = jnp.where(any_success, current_logl[selected_idx], out_logl)
            # dtype-pinned accumulation: under JAX_ENABLE_X64 the bare
            # jnp.sum promotes to int64 while the while_loop carry was
            # initialized int32, which aborts the kernel with a carry-type
            # mismatch (state[10] int32 vs int64).
            ncall = ncall + jnp.asarray(batch_ncall, dtype=ncall.dtype)
            accepted_move_count = accepted_move_count + jnp.sum(
                accepted_moves, dtype=accepted_move_count.dtype
            )
            batch_index = batch_index + jnp.asarray(1, dtype=jnp.int32)
            n_evals = n_evals + batch_evals.astype(n_evals.dtype)
            if cluster_swap:
                swap_counts = [
                    swap_counts[0] + jnp.sum(swap_steps, axis=0, dtype=jnp.int32)
                ]
            return (
                key,
                ncall,
                any_success,
                any_success,
                out_u,
                out_theta,
                out_logl,
                best_u,
                best_theta,
                best_logl,
                accepted_move_count,
                batch_index,
                n_evals,
                *swap_counts,
            )

        (
            key,
            ncall,
            done,
            accepted,
            out_u,
            out_theta,
            out_logl,
            best_u,
            best_theta,
            best_logl,
            accepted_move_count,
            _batch_index,
            n_evals,
            *swap_counts,
        ) = lax.while_loop(
            cond,
            body,
            (
                key,
                initial_ncall,
                initial_done,
                initial_done,
                template_u,
                template_theta,
                initial_best_logl,
                template_u,
                template_theta,
                initial_best_logl,
                initial_accepted_move_count,
                initial_batch_index,
                jnp.asarray(0, dtype=jnp.int32),
                *([jnp.zeros(2, dtype=jnp.int32)] if cluster_swap else []),
            ),
        )
        new_u = jnp.where(done, out_u, best_u)
        new_theta = jnp.where(done, out_theta, best_theta)
        new_logl = jnp.where(done, out_logl, best_logl)
        return (
            key,
            new_u,
            new_theta,
            new_logl,
            n_evals,
            accepted,
            accepted_move_count,
            ncall,
            *swap_counts,
        )

    return kernel


@_kernel_cache
def _step_kernel(
    loglike_spec,
    prior_spec,
    ndim: int,
    walks: int,
    replacement_chains: int,
    block_size: int,
):
    """Return the cached jitted block of :func:`step`: ``block_size`` iterations.

    The returned ``block(state, nlive, max_batches, n_active, extras,
    *callable_leaves)`` returns ``(state, dead)``. ``nlive``, ``max_batches``
    and ``n_active`` are int32 arrays (traced, not constants), so the maxiter
    tail reuses the compiled block. The callables' array leaves and large
    constants (``*_callable_leaves(loglike, prior_transform, ndim)``) are
    threaded to the chain kernel, so they are jit arguments rather than
    compiled-in constants.

    After a failed replacement, and from offset ``n_active`` on, the remaining
    iterations of the block are skipped: they neither touch the live set nor
    advance the key. The rows of skipped iterations have ``valid`` False; only
    a failed replacement sets ``state.failed``.

    With ``extras`` (the cluster frames; see :mod:`tinyns.clusters`) the chains
    also propose affine swaps between clusters and ``dead.swaps`` holds the
    per-iteration ``[accepted swaps, proposed swaps]``. A replaced live point
    leaves its cluster's statistics (its label becomes -1). ``extras=None``
    traces the swap-free program.
    """

    batch_ncall = walks * replacement_chains

    def block_kernel(state, nlive, max_batches, n_active, extras, *callable_leaves):
        cluster_swap = extras is not None
        rwalk_kernel = _chain_kernel(
            loglike_spec,
            prior_spec,
            ndim,
            walks,
            replacement_chains,
            cluster_swap,
        )
        start_iteration = state.it
        scale = state.scale
        if cluster_swap:
            clusters = extras
            labels = [clusters["labels"]]
        else:
            labels = []

        def one_iteration(carry, offset):
            (
                key,
                live_u,
                live_theta,
                live_logl,
                live_birth,
                logz_dead,
                active,
                *labels,
            ) = carry

            def run_iteration(operand):
                key, live_u, live_theta, live_logl, live_birth, logz_dead, *labels = (
                    operand
                )
                worst = jnp.argmin(live_logl)
                dead_u = live_u[worst]
                dead_theta = live_theta[worst]
                logl_worst = live_logl[worst]
                dead_birth = live_birth[worst]
                iteration = jnp.asarray(start_iteration, dtype=jnp.int32) + offset
                logx_prev = -iteration / nlive
                logx_new = -(iteration + 1) / nlive
                logwidth = logx_prev + jnp.log1p(-jnp.exp(logx_new - logx_prev))
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
                    total_proposal_count + jnp.asarray(batch_ncall - 1, dtype=jnp.int32)
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
                live_birth = jnp.where(
                    accepted, live_birth.at[worst].set(logl_worst), live_birth
                )
                return (
                    key,
                    live_u,
                    live_theta,
                    live_logl,
                    live_birth,
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
                    dead_birth,
                    *swap_counts,
                )

            def skip_iteration(operand):
                key, live_u, live_theta, live_logl, live_birth, logz_dead, *labels = (
                    operand
                )
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
                    live_birth,
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
                    live_birth[worst],
                    *([jnp.zeros(2, dtype=jnp.int32)] if cluster_swap else []),
                )

            in_block = offset < n_active
            carry, row = lax.cond(
                active & in_block,
                run_iteration,
                skip_iteration,
                (key, live_u, live_theta, live_logl, live_birth, logz_dead, *labels),
            )
            # Iterations past n_active leave the failure flag alone.
            active = jnp.where(in_block, carry[6], active)
            return (*carry[:6], active, *carry[7:]), row

        (
            (
                new_key,
                new_live_u,
                new_live_theta,
                new_live_logl,
                new_live_birth,
                logz_dead_new,
                active,
                *_,
            ),
            block,
        ) = lax.scan(
            one_iteration,
            (
                state.key,
                state.u,
                state.theta,
                state.logl,
                state.birth,
                jnp.asarray(state.logz),
                jnp.logical_not(state.failed),
                *labels,
            ),
            jnp.arange(block_size, dtype=jnp.int32),
        )
        logx_final_new = -(
            jnp.asarray(start_iteration, dtype=jnp.float32)
            + jnp.asarray(n_active, dtype=jnp.float32)
        ) / jnp.asarray(nlive, dtype=jnp.float32)
        dead = Dead(*block[:11], swaps=block[11] if cluster_swap else None)
        new_state = State(
            key=new_key,
            u=new_live_u,
            theta=new_live_theta,
            logl=new_live_logl,
            birth=new_live_birth,
            logz=logz_dead_new,
            logx=logx_final_new,
            it=start_iteration + jnp.sum(dead.valid, dtype=jnp.int32),
            scale=scale,
            ncall=state.ncall + jnp.sum(dead.ncall, dtype=jnp.int32),
            failed=jnp.logical_not(active),
        )
        return new_state, dead

    return jax.jit(block_kernel)


@_kernel_cache
def _init_kernel(loglike_spec, prior_spec, ndim: int):
    """Return the cached jitted live-point pass of :func:`init`."""
    nloglike_leaves = loglike_spec.nleaves

    def evaluate(u, *leaves):
        loglike_fn = loglike_spec.rebuild(leaves[:nloglike_leaves])
        prior_fn = prior_spec.rebuild(leaves[nloglike_leaves:])

        def evaluate_chunk(u_chunk):
            return _evaluate_jax_batch(loglike_fn, prior_fn, u_chunk, ndim)

        return lax.map(evaluate_chunk, u)

    return jax.jit(evaluate)


def init(key, loglike, prior_transform, cfg: Config) -> State:
    """Draw and evaluate the initial live set; return the first :class:`State`.

    ``key`` is a PRNG key or an int seed. The live points are evaluated in one
    compiled pass, ``replacement_chains`` points at a time (vmapped within a
    chunk), so device memory is bounded by the batch the chains use.
    """
    key = random.PRNGKey(int(key)) if isinstance(key, int) else key
    key, init_key = random.split(key)
    nlive, ndim = cfg.nlive, cfg.ndim
    live_u = random.uniform(init_key, shape=(nlive, ndim))

    chunk_size = max(1, min(int(cfg.replacement_chains), nlive))
    nchunks = -(-nlive // chunk_size)
    pad = nchunks * chunk_size - nlive
    padded = jnp.concatenate(
        [live_u, jnp.broadcast_to(live_u[:1], (pad, ndim))], axis=0
    ).reshape((nchunks, chunk_size, ndim))
    (loglike_dynamic, loglike_spec), (prior_dynamic, prior_spec) = _split_callables(
        loglike, prior_transform, ndim
    )
    evaluate = _init_kernel(loglike_spec, prior_spec, ndim)
    theta, logl = evaluate(padded, *loglike_dynamic, *prior_dynamic)
    theta = theta.reshape((nchunks * chunk_size,) + theta.shape[2:])[:nlive]
    logl = logl.reshape((-1,))[:nlive]
    return State(
        key=key,
        u=live_u,
        theta=theta,
        logl=logl,
        birth=jnp.full(logl.shape, -jnp.inf, dtype=logl.dtype),
        logz=jnp.asarray(-math.inf),
        logx=jnp.asarray(0.0, dtype=jnp.float32),
        it=jnp.asarray(0, dtype=jnp.int32),
        scale=jnp.asarray(_INITIAL_SCALE),
        ncall=jnp.asarray(nlive, dtype=jnp.int32),
        failed=jnp.asarray(False),
    )


def step(
    state: State, loglike, prior_transform, cfg: Config, extras=None, *, n_active=None
) -> tuple[State, Dead]:
    """Run one jitted block of ``cfg.block_size`` iterations.

    Returns the new state and the block's :class:`Dead` rows. ``extras`` are
    the cluster frames of the swap move (host tracker, :mod:`tinyns.clusters`);
    ``None`` runs the swap-free program. ``n_active`` (default: all) runs only
    the first ``n_active`` iterations, in the same compiled program; the other
    rows have ``valid`` False. The step scale is ``state.scale``; the host loop
    adapts it between blocks.
    """
    if n_active is None:
        n_active = cfg.block_size
    (loglike_dynamic, loglike_spec), (prior_dynamic, prior_spec) = _split_callables(
        loglike, prior_transform, cfg.ndim
    )
    block = _step_kernel(
        loglike_spec,
        prior_spec,
        cfg.ndim,
        cfg.walks,
        cfg.replacement_chains,
        cfg.block_size,
    )
    return block(
        state,
        jnp.asarray(cfg.nlive, dtype=jnp.int32),
        jnp.asarray(cfg.max_batches, dtype=jnp.int32),
        jnp.asarray(n_active, dtype=jnp.int32),
        extras,
        *loglike_dynamic,
        *prior_dynamic,
    )


def remaining_dlogz(state: State) -> float:
    """Return the live-evidence remainder ``log(Z_dead + X L_max) - log Z_dead``.

    Host-side float64 arithmetic, as in v0.2.4: the loop stops once this falls
    below ``dlogz``.
    """
    logz_dead = state.logz
    if math.isinf(float(logz_dead)) and float(logz_dead) < 0.0:
        return math.inf
    logz_remain = float(state.logx) + float(jnp.max(state.logl))
    return float(jnp.logaddexp(logz_dead, logz_remain) - logz_dead)
