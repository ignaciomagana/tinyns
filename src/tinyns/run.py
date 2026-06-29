"""Static nested-sampling implementation for :mod:`tinyns`."""

from __future__ import annotations

import math
import os
import time

import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.special import logsumexp

from tinyns.bounds import (
    as_jax_ellipsoid_bound,
    build_multi_ellipsoid_bound,
    build_single_ellipsoid_bound,
)
from tinyns.math import logdiffexp
from tinyns.result import NestedSamplingResult
from tinyns.samplers import (
    _evaluate_jax_batch,
    _make_rwalk_jax_adaptive_kernel,
    _make_rwalk_jax_kernel,
    draw_constrained_multi_bound_jax,
    draw_constrained_multi_bound_rwalk_jax,
    draw_constrained_prior,
    draw_constrained_prior_vectorized,
    draw_constrained_rslice,
    draw_constrained_rwalk,
    draw_constrained_rwalk_jax,
    draw_constrained_rwalk_jax_adaptive,
    draw_constrained_single_bound,
    draw_constrained_single_bound_jax,
    draw_constrained_single_bound_rwalk_jax,
    draw_constrained_slice,
)
from tinyns.state import NestedRunState, save_checkpoint_npz
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike


def _as_points(array, ndim: int):
    """Return an array with trailing dimension ``ndim`` for sample points."""

    array = jnp.asarray(array)
    if ndim == 1 and array.ndim == 1:
        return array.reshape((-1, 1))
    if array.shape[-1:] != (ndim,):
        raise ValueError(f"point arrays must have trailing shape ({ndim},)")
    return array


def _as_point(array, ndim: int):
    """Return a single point with shape ``(ndim,)``."""

    array = jnp.asarray(array)
    if ndim == 1 and array.shape == ():
        return array.reshape((1,))
    if array.shape != (ndim,):
        raise ValueError(f"point must have shape ({ndim},)")
    return array


def _evaluate_live_points(loglike, theta_live, *, vectorized: bool):
    if vectorized:
        logl = jnp.asarray(loglike(theta_live), dtype=float).reshape((-1,))
        expected_shape = (theta_live.shape[0],)
        if logl.shape != expected_shape:
            raise ValueError(
                "vectorized loglike must return one value per live point; "
                f"expected shape {expected_shape}, got {logl.shape}"
            )
        return logl
    return jnp.asarray([float(loglike(theta)) for theta in theta_live], dtype=float)


def _transform_live_points(prior_transform, u_live, ndim: int, *, vectorized: bool):
    if vectorized:
        return _as_points(prior_transform(u_live), ndim)
    return jnp.stack([_as_point(prior_transform(u), ndim) for u in u_live])


def _logzerr(logwt, logl, logz: float, nlive: int) -> float:
    log_weights = jnp.asarray(logwt) - logz
    weights = jnp.exp(log_weights)
    information = jnp.sum(weights * (jnp.asarray(logl) - logz))
    return float(jnp.sqrt(jnp.maximum(information / nlive, 0.0)))


def _run_static_jax_rwalk_block(
    key,
    live_u,
    live_theta,
    live_logl,
    logz_dead,
    start_iteration,
    nlive,
    loglike,
    prior_transform,
    ndim,
    *,
    block_size,
    walks,
    step_scale,
    replacement_chains,
    replacement_chain_schedule=None,
    max_attempts,
    min_accepts,
):
    """Run an experimental fixed-size block of unbounded JAX rwalk iterations.

    Convergence is intentionally checked only between blocks by the Python
    driver, so an experimental block-mode run may overshoot ``dlogz`` by up to
    ``block_size - 1`` nested iterations.
    """

    if replacement_chain_schedule is None:
        batch_ncall = int(walks) * int(replacement_chains)
        max_batches = int(max_attempts) // batch_ncall
        rwalk_kernel = _make_rwalk_jax_kernel(
            loglike,
            prior_transform,
            int(ndim),
            int(walks),
            int(replacement_chains),
            False,
        )
        adaptive_rwalk_kernel = None
    else:
        replacement_chain_schedule = tuple(int(c) for c in replacement_chain_schedule)
        rwalk_kernel = None
        adaptive_rwalk_kernel = _make_rwalk_jax_adaptive_kernel(
            loglike,
            prior_transform,
            int(ndim),
            int(walks),
            replacement_chain_schedule,
        )
    proposal_chol = jnp.eye(int(ndim), dtype=jnp.asarray(live_u).dtype)

    def one_iteration(carry, offset):
        key, live_u, live_theta, live_logl, logz_dead = carry
        worst = jnp.argmin(live_logl)
        dead_u = live_u[worst]
        dead_theta = live_theta[worst]
        logl_worst = live_logl[worst]
        iteration = jnp.asarray(start_iteration, dtype=jnp.int32) + offset
        logx_prev = -iteration / nlive
        logx_new = -(iteration + 1) / nlive
        logwidth = logx_prev + jnp.log1p(-jnp.exp(logx_new - logx_prev))
        logwt = logwidth + logl_worst
        logz_dead = jnp.logaddexp(logz_dead, logwt)

        if adaptive_rwalk_kernel is None:
            (
                key,
                new_u,
                new_theta,
                new_logl,
                replacement_ncall,
                accepted,
            ) = rwalk_kernel(
                key,
                logl_worst,
                live_u,
                live_logl,
                jnp.asarray(step_scale),
                jnp.asarray(min_accepts),
                jnp.asarray(max_batches, dtype=jnp.int32),
                proposal_chol,
            )
            replacement_batches_used = (
                replacement_ncall + jnp.asarray(batch_ncall - 1, dtype=jnp.int32)
            ) // jnp.asarray(batch_ncall, dtype=jnp.int32)
            replacement_chains_used = (
                replacement_batches_used
                * jnp.asarray(replacement_chains, dtype=jnp.int32)
            )
        else:
            (
                key,
                new_u,
                new_theta,
                new_logl,
                replacement_ncall,
                accepted,
                replacement_batches_used,
                replacement_chains_used,
                _last_chain_count,
            ) = adaptive_rwalk_kernel(
                key,
                logl_worst,
                live_u,
                live_logl,
                jnp.asarray(step_scale),
                jnp.asarray(min_accepts),
                jnp.asarray(max_attempts, dtype=jnp.int32),
                proposal_chol,
            )
        insertion_index = (
            jnp.sum(live_logl <= new_logl) - (logl_worst <= new_logl)
        ).astype(jnp.int32)
        live_u = jnp.where(accepted, live_u.at[worst].set(new_u), live_u)
        live_theta = jnp.where(
            accepted, live_theta.at[worst].set(new_theta), live_theta
        )
        live_logl = jnp.where(accepted, live_logl.at[worst].set(new_logl), live_logl)
        return (key, live_u, live_theta, live_logl, logz_dead), (
            dead_u,
            dead_theta,
            logl_worst,
            logwt,
            replacement_ncall,
            insertion_index,
            replacement_batches_used,
            replacement_chains_used,
            accepted,
        )

    (
        new_key,
        new_live_u,
        new_live_theta,
        new_live_logl,
        logz_dead_new,
    ), block = lax.scan(
        one_iteration,
        (key, live_u, live_theta, live_logl, jnp.asarray(logz_dead)),
        jnp.arange(int(block_size), dtype=jnp.int32),
    )
    (
        dead_u_block,
        dead_theta_block,
        dead_logl_block,
        dead_logwt_block,
        replacement_ncall_block,
        insertion_indices_block,
        replacement_batches_block,
        replacement_chains_used_block,
        accepted_block,
    ) = block
    logx_final_new = -(int(start_iteration) + int(block_size)) / int(nlive)
    return (
        new_key,
        new_live_u,
        new_live_theta,
        new_live_logl,
        dead_u_block,
        dead_theta_block,
        dead_logl_block,
        dead_logwt_block,
        replacement_ncall_block,
        insertion_indices_block,
        replacement_batches_block,
        replacement_chains_used_block,
        accepted_block,
        logz_dead_new,
        logx_final_new,
    )


def _run_static_jax_bounded_rwalk_block(
    key,
    live_u,
    live_theta,
    live_logl,
    logz_dead,
    start_iteration,
    nlive,
    loglike,
    prior_transform,
    ndim,
    *,
    jax_bound,
    bound_kind,
    block_size,
    walks,
    step_scale,
    replacement_chains,
    replacement_chain_schedule=None,
    max_attempts,
    min_accepts,
    bound_batch_size,
    bound_max_batches,
    overlap_correction,
    proposal_chol=None,
    jax_vectorized: bool = False,
):
    """Run an experimental bounded rwalk block with one fixed JAX bound.

    The ellipsoid arrays are reused for every replacement in the block.  The
    Python driver intentionally rebuilds bounds only between blocks according
    to ``bound_update_interval``, so this experimental path can use a slightly
    stale bound inside a block.
    """

    fused_draw = (
        draw_constrained_multi_bound_rwalk_jax
        if bound_kind == "multi"
        else draw_constrained_single_bound_rwalk_jax
    )
    carry_key = key
    dead_u_values = []
    dead_theta_values = []
    dead_logl_values = []
    dead_logwt_values = []
    replacement_ncall_values = []
    insertion_index_values = []
    replacement_batch_values = []
    replacement_chain_values = []
    bound_seed_call_values = []
    bound_seed_batch_values = []
    rwalk_kernel_call_values = []
    bound_draw_values = []
    bound_eval_values = []
    bound_unit_cube_acceptance_values = []
    bound_overlap_rejection_values = []

    for offset in range(int(block_size)):
        worst = int(jnp.argmin(live_logl))
        dead_u = live_u[worst]
        dead_theta = live_theta[worst]
        logl_worst = float(live_logl[worst])
        iteration = int(start_iteration) + offset
        logx_prev = -iteration / int(nlive)
        logx_new = -(iteration + 1) / int(nlive)
        logwidth = logdiffexp(logx_prev, logx_new)
        logwt = float(logwidth + logl_worst)
        logz_dead = float(jnp.logaddexp(logz_dead, logwt))

        result = fused_draw(
            carry_key,
            loglike,
            prior_transform,
            logl_worst,
            jax_bound,
            ndim,
            walks=walks,
            step_scale=step_scale,
            max_attempts=max_attempts,
            min_accepts=min_accepts,
            replacement_chains=replacement_chains,
            replacement_chain_schedule=replacement_chain_schedule,
            bound_batch_size=bound_batch_size,
            bound_max_batches=bound_max_batches,
            proposal_chol=proposal_chol,
            jax_vectorized=jax_vectorized,
            **(
                {"overlap_correction": overlap_correction}
                if bound_kind == "multi"
                else {}
            ),
        )
        carry_key, new_u, new_theta, new_logl, calls, accepted, info = result
        if not bool(accepted):
            raise RuntimeError("bounded JAX rwalk block failed to draw a replacement")
        insertion_index = int(
            jnp.sum(live_logl <= new_logl) - (live_logl[worst] <= new_logl)
        )
        live_u = live_u.at[worst].set(new_u)
        live_theta = live_theta.at[worst].set(new_theta)
        live_logl = live_logl.at[worst].set(new_logl)

        dead_u_values.append(dead_u)
        dead_theta_values.append(dead_theta)
        dead_logl_values.append(logl_worst)
        dead_logwt_values.append(logwt)
        replacement_ncall_values.append(int(calls))
        insertion_index_values.append(insertion_index)
        replacement_batch_values.append(int(info["replacement_batches"]))
        replacement_chain_values.append(int(info["replacement_chains_used"]))
        bound_seed_call_values.append(int(info.get("bound_seed_loglike_evals", 0)))
        bound_seed_batch_values.append(int(info.get("bound_seed_batches", 0)))
        rwalk_kernel_call_values.append(int(info.get("rwalk_kernel_calls", 0)))
        bound_draw_values.append(int(info.get("bound_seed_draws", 0)))
        bound_eval_values.append(int(info.get("bound_seed_loglike_evals", 0)))
        bound_unit_cube_acceptance_values.append(
            float(info.get("bound_seed_unit_cube_acceptance", 0.0))
        )
        bound_overlap_rejection_values.append(
            int(info.get("bound_seed_overlap_rejections", 0))
        )

    logx_final_new = -(int(start_iteration) + int(block_size)) / int(nlive)
    return (
        carry_key,
        live_u,
        live_theta,
        live_logl,
        jnp.stack(dead_u_values),
        jnp.stack(dead_theta_values),
        jnp.asarray(dead_logl_values),
        jnp.asarray(dead_logwt_values),
        jnp.asarray(replacement_ncall_values, dtype=int),
        jnp.asarray(insertion_index_values, dtype=int),
        jnp.asarray(replacement_batch_values, dtype=int),
        jnp.asarray(replacement_chain_values, dtype=int),
        jnp.ones((int(block_size),), dtype=bool),
        float(logz_dead),
        float(logx_final_new),
        bound_seed_call_values,
        bound_seed_batch_values,
        rwalk_kernel_call_values,
        bound_draw_values,
        bound_eval_values,
        bound_unit_cube_acceptance_values,
        bound_overlap_rejection_values,
    )


def _make_run_state(
    *,
    iteration: int,
    logz: float,
    dlogz: float,
    ncall: int,
    logl_min: float,
    logl_live_max: float,
    sample: str,
    nlive: int,
    ndim: int,
    replacement_ncall: list[int],
    replacement_failures: int,
    replacement_batches: list[int] | None = None,
    replacement_chains_used: list[int] | None = None,
    replacement_chain_usage_counts: dict[str, int] | None = None,
    replacement_chain_schedule=None,
    replacement_chains: int | None = None,
    kernel: str | None = None,
    walks: int | None = None,
    bound: str = "none",
    bound_draws: list[int] | None = None,
    bound_unit_cube_acceptance: list[float] | None = None,
    bound_nellipsoids: list[int] | None = None,
    rwalk_seed: str = "live",
    bound_seed_kernel: str = "python",
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
    if replacement_chains_used:
        replacement_mean_chains_used_so_far = float(
            sum(replacement_chains_used) / len(replacement_chains_used)
        )
        replacement_max_chains_used_so_far = int(max(replacement_chains_used))
    else:
        replacement_mean_chains_used_so_far = None
        replacement_max_chains_used_so_far = None
    replacement_chain_usage_counts_so_far = dict(replacement_chain_usage_counts or {})
    mean_bound_draws_so_far = (
        float(sum(bound_draws) / len(bound_draws)) if bound_draws else None
    )
    mean_bound_unit_cube_acceptance_so_far = (
        float(sum(bound_unit_cube_acceptance) / len(bound_unit_cube_acceptance))
        if bound_unit_cube_acceptance
        else None
    )
    mean_bound_nellipsoids_so_far = (
        float(sum(bound_nellipsoids) / len(bound_nellipsoids))
        if bound_nellipsoids
        else None
    )
    adaptive_replacement_chains = replacement_chain_schedule is not None
    return {
        "iter": int(iteration),
        "logz": float(logz),
        "dlogz": float(dlogz),
        "ncall": int(ncall),
        "logl_min": float(logl_min),
        "logl_live_max": float(logl_live_max),
        "sample": str(sample),
        "nlive": int(nlive),
        "ndim": int(ndim),
        "replacement_mean_ncall_so_far": replacement_mean_ncall_so_far,
        "replacement_mean_batches_so_far": replacement_mean_batches_so_far,
        "replacement_max_batches_so_far": replacement_max_batches_so_far,
        "replacement_mean_chains_used_so_far": replacement_mean_chains_used_so_far,
        "replacement_max_chains_used_so_far": replacement_max_chains_used_so_far,
        "replacement_chain_usage_counts_so_far": replacement_chain_usage_counts_so_far,
        "adaptive_replacement_chains": adaptive_replacement_chains,
        "replacement_chains": replacement_chains,
        "replacement_chain_schedule": replacement_chain_schedule,
        "kernel": kernel,
        "walks": walks,
        "replacement_failures": int(replacement_failures),
        "bound": bound,
        "rwalk_seed": rwalk_seed,
        "bound_seed_kernel": bound_seed_kernel,
        "mean_bound_draws_so_far": mean_bound_draws_so_far,
        "mean_bound_unit_cube_acceptance_so_far": (
            mean_bound_unit_cube_acceptance_so_far
        ),
        "mean_bound_nellipsoids_so_far": mean_bound_nellipsoids_so_far,
    }


def _format_progress_line(state: dict[str, object]) -> str:
    """Format one dependency-free progress line for a run state."""

    repl = state.get("replacement_mean_ncall_so_far")
    repl_text = "n/a" if repl is None else f"{float(repl):.1f}"
    batches = state.get("replacement_mean_batches_so_far")
    batches_text = "n/a" if batches is None else f"{float(batches):.2f}"
    chains = state.get("replacement_mean_chains_used_so_far")
    chains_text = "n/a" if chains is None else f"{float(chains):.1f}"
    usage_counts = state.get("replacement_chain_usage_counts_so_far") or {}
    usage_text = ""
    if state.get("adaptive_replacement_chains") and usage_counts:
        nonzero = [
            (str(chain_count), int(count))
            for chain_count, count in usage_counts.items()
            if int(count) > 0
        ]
        nonzero.sort(key=lambda item: int(item[0]))
        usage = ",".join(f"{chain_count}:{count}" for chain_count, count in nonzero[:4])
        if len(nonzero) > 4:
            usage += ",..."
        usage_text = f" usage={usage}" if usage else ""
    bound_text = ""
    if state.get("bound", "none") != "none":
        bdraw = state.get("mean_bound_draws_so_far")
        bacc = state.get("mean_bound_unit_cube_acceptance_so_far")
        bdraw_text = "n/a" if bdraw is None else f"{float(bdraw):.1f}"
        bacc_text = "n/a" if bacc is None else f"{float(bacc):.3f}"
        if state.get("bound") == "multi":
            nell = state.get("mean_bound_nellipsoids_so_far")
            nell_text = "n/a" if nell is None else f"{float(nell):.1f}"
            seed_text = state.get("rwalk_seed", "live")
            seed_kernel_text = (
                f" seed_kernel={state['bound_seed_kernel']}"
                if state.get("bound_seed_kernel", "python") != "python"
                else ""
            )
            bound_text = (
                f" bound=multi seed={seed_text}{seed_kernel_text} "
                f"nell={nell_text} bacc={bacc_text}"
            )
        else:
            seed_text = state.get("rwalk_seed", "live")
            seed_kernel_text = (
                f" seed_kernel={state['bound_seed_kernel']}"
                if state.get("bound_seed_kernel", "python") != "python"
                else ""
            )
            bound_text = (
                f" bound={state['bound']} seed={seed_text}{seed_kernel_text} "
                f"bdraw={bdraw_text} bacc={bacc_text}"
            )
    return (
        f"iter={int(state['iter']):05d} "
        f"logz={float(state['logz']):.3f} "
        f"dlogz={float(state['dlogz']):.3f} "
        f"ncall={int(state['ncall'])} "
        f"logl_min={float(state['logl_min']):.3g} "
        f"logl_live_max={float(state['logl_live_max']):.3g} "
        f"repl_ncall={repl_text} "
        f"repl_batches={batches_text} "
        f"repl_chains={chains_text}"
        f"{usage_text} "
        f"sample={state['sample']}"
        f"{bound_text}"
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
    sample: str = "prior",
    kernel: str = "python",
    vectorized: bool = False,
    max_attempts: int = 10_000,
    progress: bool = False,
    progress_interval: int = 100,
    callback=None,
    callback_interval: int = 100,
    batch_size: int = 128,
    walks: int = 25,
    step_scale: float = 0.1,
    slices: int = 5,
    slice_steps: int = 10,
    min_accepts: int = 1,
    replacement_chains: int = 1,
    replacement_chain_schedule=None,
    rwalk_proposal: str = "isotropic",
    rwalk_cov_jitter: float = 1e-6,
    bound: str = "none",
    bound_enlargement: float = 1.25,
    bound_update_interval: int = 1,
    bound_jitter: float = 1e-6,
    bound_max_draws: int | None = None,
    bound_rebuild_on_failure: bool = False,
    bound_failure_rebuild_threshold: int = 1,
    multi_bound_max_ellipsoids: int = 32,
    multi_bound_min_points: int | None = None,
    multi_bound_split_threshold: float = 0.9,
    multi_bound_enlargement: float | None = None,
    multi_bound_overlap_correction: bool = True,
    rwalk_seed: str = "live",
    rwalk_seed_fallback: bool = True,
    bound_seed_kernel: str = "python",
    allow_unused_bound: bool = False,
    fused_bound_rwalk: bool = False,
    jax_vectorized: bool = False,
    jax_block_size: int = 1,
    initial_state: NestedRunState | None = None,
    checkpoint_path=None,
    checkpoint_interval: int = 100,
):
    """Run a simple static nested-sampling loop.

    The replacement ``sample`` strategy may be ``"prior"``, ``"rwalk"``,
    ``"slice"``, or ``"rslice"``.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if nlive <= 0:
        raise ValueError("nlive must be a positive integer")
    if sample not in {"prior", "rwalk", "slice", "rslice", "bound"}:
        raise ValueError(
            "sample must be one of {'prior', 'rwalk', 'slice', 'rslice', 'bound'}"
        )
    if kernel not in {"python", "jax"}:
        raise ValueError("kernel must be one of {'python', 'jax'}")
    if bound not in {"none", "single", "multi"}:
        raise ValueError("bound must be one of {'none', 'single', 'multi'}")
    if bound_update_interval <= 0:
        raise ValueError("bound_update_interval must be a positive integer")
    if bound_enlargement <= 0:
        raise ValueError("bound_enlargement must be positive")
    if bound_jitter <= 0:
        raise ValueError("bound_jitter must be positive")
    if bound_max_draws is not None and bound_max_draws <= 0:
        raise ValueError("bound_max_draws must be positive or None")
    if (
        not isinstance(bound_failure_rebuild_threshold, int)
        or isinstance(bound_failure_rebuild_threshold, bool)
        or bound_failure_rebuild_threshold <= 0
    ):
        raise ValueError("bound_failure_rebuild_threshold must be a positive integer")
    if rwalk_seed not in {"live", "bound"}:
        raise ValueError("rwalk_seed must be one of {'live', 'bound'}")
    if bound_seed_kernel not in {"python", "jax"}:
        raise ValueError("bound_seed_kernel must be one of {'python', 'jax'}")
    if fused_bound_rwalk and not (
        sample == "rwalk"
        and kernel == "jax"
        and bound in {"single", "multi"}
        and rwalk_seed == "bound"
    ):
        raise NotImplementedError(
            "fused_bound_rwalk=True is supported only for sample='rwalk', "
            "kernel='jax', bound in {'single', 'multi'}, and rwalk_seed='bound'"
        )
    if bound_seed_kernel == "jax" and not (
        sample == "rwalk"
        and kernel == "jax"
        and bound in {"single", "multi"}
        and rwalk_seed == "bound"
    ):
        raise NotImplementedError(
            "bound_seed_kernel='jax' is supported only for sample='rwalk', "
            "kernel='jax', bound in {'single', 'multi'}, and "
            "rwalk_seed='bound'"
        )
    if sample == "bound" and bound not in {"single", "multi"}:
        raise ValueError('sample="bound" requires bound="single" or bound="multi"')
    if (
        sample == "rwalk"
        and bound in {"single", "multi"}
        and rwalk_seed == "live"
        and not allow_unused_bound
    ):
        raise ValueError(
            "bound='single' or bound='multi' with sample='rwalk' requires "
            "rwalk_seed='bound'. Otherwise the bound is built but not used. "
            "Set rwalk_seed='bound' for bounded rwalk, or bound='none' "
            "for ordinary live-seeded rwalk."
        )
    if multi_bound_max_ellipsoids <= 0:
        raise ValueError("multi_bound_max_ellipsoids must be positive")
    if multi_bound_min_points is not None and multi_bound_min_points <= 0:
        raise ValueError("multi_bound_min_points must be positive or None")
    if multi_bound_split_threshold <= 0.0:
        raise ValueError("multi_bound_split_threshold must be positive")
    if multi_bound_enlargement is not None and multi_bound_enlargement <= 0.0:
        raise ValueError("multi_bound_enlargement must be positive or None")
    if rwalk_proposal not in {"isotropic", "live-cov"}:
        raise ValueError("rwalk_proposal must be one of {'isotropic', 'live-cov'}")
    if rwalk_cov_jitter <= 0:
        raise ValueError("rwalk_cov_jitter must be positive")
    if rwalk_proposal == "live-cov" and not (sample == "rwalk" and kernel == "jax"):
        raise NotImplementedError(
            'rwalk_proposal="live-cov" is currently supported only with '
            'sample="rwalk", kernel="jax"'
        )
    if kernel == "jax" and sample not in {"rwalk"}:
        raise NotImplementedError(
            'kernel="jax" is currently only supported with sample="rwalk"'
        )
    if sample == "rwalk" and vectorized:
        raise NotImplementedError(
            "vectorized rwalk is not implemented yet; use vectorized=False "
            'with sample="rwalk"'
        )
    if sample == "slice" and vectorized:
        raise NotImplementedError(
            "vectorized slice sampling is not implemented yet; use vectorized=False "
            'with sample="slice"'
        )
    if sample == "rslice" and vectorized:
        raise NotImplementedError(
            "vectorized rslice sampling is not implemented yet; use vectorized=False "
            'with sample="rslice"'
        )
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be a positive integer")
    if callback_interval <= 0:
        raise ValueError("callback_interval must be a positive integer")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    if checkpoint_path is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if slices <= 0:
        raise ValueError("slices must be a positive integer")
    if slice_steps <= 0:
        raise ValueError("slice_steps must be a positive integer")
    if (
        not isinstance(min_accepts, int)
        or isinstance(min_accepts, bool)
        or min_accepts <= 0
    ):
        raise ValueError("min_accepts must be a positive integer")
    if (
        not isinstance(jax_block_size, int)
        or isinstance(jax_block_size, bool)
        or jax_block_size <= 0
    ):
        raise ValueError("jax_block_size must be a positive integer")
    if jax_block_size > 1:
        unbounded_block = sample == "rwalk" and kernel == "jax" and bound == "none"
        bounded_block = (
            sample == "rwalk"
            and kernel == "jax"
            and bound in {"single", "multi"}
            and rwalk_seed == "bound"
            and bound_seed_kernel == "jax"
            and fused_bound_rwalk
        )
        if not (unbounded_block or bounded_block):
            raise NotImplementedError(
                "jax_block_size > 1 is experimental and currently supported only "
                "for unbounded sample='rwalk', kernel='jax' or fixed-bound "
                "block mode with bound in {'single', 'multi'}, rwalk_seed='bound', "
                "bound_seed_kernel='jax', and fused_bound_rwalk=True"
            )
        if jax_vectorized:
            raise NotImplementedError(
                "jax_block_size > 1 does not support jax_vectorized"
            )
    if (
        not isinstance(replacement_chains, int)
        or isinstance(replacement_chains, bool)
        or replacement_chains <= 0
    ):
        raise ValueError("replacement_chains must be a positive integer")
    if replacement_chains != 1 and not (sample == "rwalk" and kernel == "jax"):
        raise NotImplementedError(
            "replacement_chains is currently supported only for "
            "sample='rwalk', kernel='jax'"
        )
    if (
        sample == "rwalk"
        and kernel == "jax"
        and int(walks) * int(replacement_chains) > int(max_attempts)
    ):
        raise ValueError("max_attempts must be at least walks * replacement_chains")
    if replacement_chain_schedule is not None:
        if not (sample == "rwalk" and kernel == "jax"):
            raise NotImplementedError(
                "replacement_chain_schedule is currently supported only for "
                "sample='rwalk', kernel='jax'"
            )
        try:
            replacement_chain_schedule = tuple(replacement_chain_schedule)
        except TypeError as exc:
            raise ValueError(
                "replacement_chain_schedule must be a non-empty sequence "
                "of positive integers"
            ) from exc
        if not replacement_chain_schedule or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in replacement_chain_schedule
        ):
            raise ValueError(
                "replacement_chain_schedule must be a non-empty sequence "
                "of positive integers"
            )
        if max(replacement_chain_schedule) * int(walks) > int(max_attempts):
            raise ValueError(
                "max_attempts must be at least max(replacement_chain_schedule) * walks"
            )
    if maxiter is None:
        maxiter = 10_000 * ndim
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative")

    config = {
        "ndim": int(ndim),
        "nlive": int(nlive),
        "sample": str(sample),
        "kernel": str(kernel),
        "vectorized": bool(vectorized),
        "max_attempts": int(max_attempts),
        "batch_size": int(batch_size),
        "walks": int(walks),
        "step_scale": float(step_scale),
        "slices": int(slices),
        "slice_steps": int(slice_steps),
        "min_accepts": int(min_accepts),
        "replacement_chains": int(replacement_chains),
        "replacement_chain_schedule": (
            None
            if replacement_chain_schedule is None
            else list(replacement_chain_schedule)
        ),
        "bound": str(bound),
        "bound_enlargement": float(bound_enlargement),
        "bound_update_interval": int(bound_update_interval),
        "bound_jitter": float(bound_jitter),
        "bound_max_draws": bound_max_draws,
        "bound_rebuild_on_failure": bool(bound_rebuild_on_failure),
        "bound_failure_rebuild_threshold": int(bound_failure_rebuild_threshold),
        "multi_bound_max_ellipsoids": int(multi_bound_max_ellipsoids),
        "multi_bound_min_points": multi_bound_min_points,
        "multi_bound_split_threshold": float(multi_bound_split_threshold),
        "multi_bound_enlargement": multi_bound_enlargement,
        "multi_bound_overlap_correction": bool(multi_bound_overlap_correction),
        "rwalk_seed": str(rwalk_seed),
        "rwalk_seed_fallback": bool(rwalk_seed_fallback),
        "bound_seed_kernel": "jax" if fused_bound_rwalk else str(bound_seed_kernel),
        "allow_unused_bound": bool(allow_unused_bound),
        "fused_bound_rwalk": bool(fused_bound_rwalk),
        "jax_vectorized": bool(jax_vectorized),
        "jax_block_size": int(jax_block_size),
    }
    checkpoint_path_str = (
        None if checkpoint_path is None else os.fspath(checkpoint_path)
    )
    resumed_from_checkpoint = initial_state is not None

    if initial_state is None:
        key = random.PRNGKey(int(key)) if isinstance(key, int) else key
        key, init_key = random.split(key)
        live_u = random.uniform(init_key, shape=(nlive, ndim))
        if kernel == "jax" and jax_vectorized:
            live_theta, live_logl = _evaluate_jax_batch(
                loglike,
                prior_transform,
                live_u,
                ndim,
                jax_vectorized=True,
            )
        else:
            live_theta = _transform_live_points(
                prior_transform, live_u, ndim, vectorized=vectorized
            )
            live_logl = _evaluate_live_points(
                loglike, live_theta, vectorized=vectorized
            )
        ncall = nlive
        logz_dead = -math.inf
        replacement_ncall = []
        insertion_indices = []
        replacement_failures = 0
        replacement_batches = []
        replacement_chains_used = []
        replacement_chain_usage_counts = {}
        success = True
        message = "converged"
        stopped_by_callback = False
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
        replacement_batches = []
        replacement_chains_used = []
        replacement_chain_usage_counts = {}
        success = True
        message = "converged"
        stopped_by_callback = False
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
    progress_printer = _ProgressPrinter() if progress else None
    current_bound = None
    bound_updates = 0
    bound_draw_history = []
    bound_eval_history = []
    bound_unit_cube_acceptance_history = []
    bound_build_time_history = []
    bound_log_volume_history = []
    bound_nellipsoid_history = []
    bound_overlap_rejection_history = []
    bound_seed_call_history = []
    bound_seed_batch_history = []
    rwalk_kernel_call_history = []
    consecutive_bound_failures = 0
    force_bound_rebuild = False
    bound_forced_rebuilds = 0

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
        )

    def maybe_checkpoint(*, final: bool = False) -> None:
        if checkpoint_path_str is None:
            return
        if final or iteration == 1 or iteration % checkpoint_interval == 0:
            save_checkpoint_npz(checkpoint_path_str, current_state(), config)

    if jax_block_size > 1:
        while iteration < maxiter:
            block_size_now = min(int(jax_block_size), maxiter - iteration)
            logz_dead_before_block = logz_dead
            block_extra = None
            if bound in {"single", "multi"}:
                if (
                    current_bound is None
                    or iteration % bound_update_interval == 0
                    or force_bound_rebuild
                ):
                    build_start = time.perf_counter()
                    if bound == "multi":
                        current_bound = build_multi_ellipsoid_bound(
                            live_u,
                            enlargement=multi_bound_enlargement
                            or bound_enlargement,
                            jitter=bound_jitter,
                            max_ellipsoids=multi_bound_max_ellipsoids,
                            min_points=multi_bound_min_points,
                            split_threshold=multi_bound_split_threshold,
                        )
                    else:
                        current_bound = build_single_ellipsoid_bound(
                            live_u,
                            enlargement=bound_enlargement,
                            jitter=bound_jitter,
                        )
                    bound_build_time_history.append(time.perf_counter() - build_start)
                    bound_log_volume_history.append(
                        float(
                            current_bound.log_total_volume
                            if hasattr(current_bound, "log_total_volume")
                            else current_bound.log_volume
                        )
                    )
                    bound_nellipsoid_history.append(
                        int(len(getattr(current_bound, "ellipsoids", (current_bound,))))
                    )
                    bound_updates += 1
                    if force_bound_rebuild:
                        bound_forced_rebuilds += 1
                        force_bound_rebuild = False
                seed_limit = bound_max_draws or max_attempts
                result = _run_static_jax_bounded_rwalk_block(
                    key,
                    live_u,
                    live_theta,
                    live_logl,
                    logz_dead,
                    iteration,
                    nlive,
                    loglike,
                    prior_transform,
                    ndim,
                    jax_bound=as_jax_ellipsoid_bound(current_bound),
                    bound_kind=bound,
                    block_size=block_size_now,
                    walks=walks,
                    step_scale=step_scale,
                    replacement_chains=replacement_chains,
                    replacement_chain_schedule=replacement_chain_schedule,
                    max_attempts=max_attempts,
                    min_accepts=min_accepts,
                    bound_batch_size=batch_size,
                    bound_max_batches=int(math.ceil(seed_limit / batch_size)),
                    overlap_correction=multi_bound_overlap_correction,
                    jax_vectorized=jax_vectorized,
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
                    replacement_chains_used_block,
                    accepted_block,
                    logz_dead,
                    logx_final,
                    *block_extra,
                ) = result
            else:
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
                    replacement_chains_used_block,
                    accepted_block,
                    logz_dead,
                    logx_final,
                ) = _run_static_jax_rwalk_block(
                    key,
                    live_u,
                    live_theta,
                    live_logl,
                    logz_dead,
                    iteration,
                    nlive,
                    loglike,
                    prior_transform,
                    ndim,
                    block_size=block_size_now,
                    walks=walks,
                    step_scale=step_scale,
                    replacement_chains=replacement_chains,
                    replacement_chain_schedule=replacement_chain_schedule,
                    max_attempts=max_attempts,
                    min_accepts=min_accepts,
                )
            block_start = iteration
            block_accepted = [bool(x) for x in np.asarray(accepted_block)]
            failed_offsets = [idx for idx, ok in enumerate(block_accepted) if not ok]
            if failed_offsets:
                replacement_failures += 1
                success = False
                message = (
                    f"max_attempts={max_attempts} hit during constrained rwalk draw"
                )
                block_size_now = failed_offsets[0]
            block_stop = iteration + block_size_now
            if failed_offsets:
                logz_dead = float(logz_dead_before_block)
                for logwt_value in np.asarray(dead_logwt_block[:block_size_now]):
                    logz_dead = float(jnp.logaddexp(logz_dead, float(logwt_value)))
                logx_final = -block_stop / int(nlive)
            dead_u_block = dead_u_block[:block_size_now]
            dead_theta_block = dead_theta_block[:block_size_now]
            dead_logl_block = dead_logl_block[:block_size_now]
            dead_logwt_block = dead_logwt_block[:block_size_now]
            replacement_ncall_block = replacement_ncall_block[:block_size_now]
            insertion_indices_block = insertion_indices_block[:block_size_now]
            replacement_batches_block = replacement_batches_block[:block_size_now]
            replacement_chains_used_block = replacement_chains_used_block[
                :block_size_now
            ]
            dead_u_storage[block_start:block_stop] = np.asarray(dead_u_block)
            dead_theta_storage[block_start:block_stop] = np.asarray(dead_theta_block)
            dead_logl_storage[block_start:block_stop] = np.asarray(dead_logl_block)
            dead_logwt_storage[block_start:block_stop] = np.asarray(dead_logwt_block)
            block_ncalls = [int(x) for x in np.asarray(replacement_ncall_block)]
            replacement_ncall.extend(block_ncalls)
            insertion_indices.extend(
                int(x) for x in np.asarray(insertion_indices_block)
            )
            ncall += int(sum(block_ncalls))
            block_batches = [int(x) for x in np.asarray(replacement_batches_block)]
            block_chains_used = [
                int(x) for x in np.asarray(replacement_chains_used_block)
            ]
            replacement_batches.extend(block_batches)
            replacement_chains_used.extend(block_chains_used)
            if block_extra is not None:
                (
                    block_bound_seed_calls,
                    block_bound_seed_batches,
                    block_rwalk_kernel_calls,
                    block_bound_draws,
                    block_bound_evals,
                    block_bound_unit_cube_acceptance,
                    block_bound_overlap_rejections,
                ) = block_extra
                bound_seed_call_history.extend(block_bound_seed_calls)
                bound_seed_batch_history.extend(block_bound_seed_batches)
                rwalk_kernel_call_history.extend(block_rwalk_kernel_calls)
                bound_draw_history.extend(block_bound_draws)
                bound_eval_history.extend(block_bound_evals)
                bound_unit_cube_acceptance_history.extend(
                    block_bound_unit_cube_acceptance
                )
                bound_overlap_rejection_history.extend(
                    block_bound_overlap_rejections
                )
            if replacement_chain_schedule is None:
                chain_count = str(int(replacement_chains))
                replacement_chain_usage_counts[chain_count] = (
                    replacement_chain_usage_counts.get(chain_count, 0)
                    + sum(block_batches)
                )
            else:
                schedule = [int(c) for c in replacement_chain_schedule]
                for batches_used in block_batches:
                    for batch_index in range(batches_used):
                        chain_count = str(
                            schedule[batch_index]
                            if batch_index < len(schedule)
                            else schedule[-1]
                        )
                        replacement_chain_usage_counts[chain_count] = (
                            replacement_chain_usage_counts.get(chain_count, 0) + 1
                        )
            iteration = block_stop
            maybe_checkpoint()
            if failed_offsets:
                break

            logz_remain = float(logx_final) + float(jnp.max(live_logl))
            delta_logz = float(jnp.logaddexp(logz_dead, logz_remain) - logz_dead)
            final_delta_logz = delta_logz
            final_iteration = delta_logz < dlogz or iteration == maxiter
            if iteration == maxiter and delta_logz >= dlogz:
                success = False
                message = f"maxiter={maxiter} reached"
            state = _make_run_state(
                iteration=iteration,
                logz=logz_dead,
                dlogz=delta_logz,
                ncall=ncall,
                logl_min=float(dead_logl_block[-1]),
                logl_live_max=float(jnp.max(live_logl)),
                sample=sample,
                nlive=nlive,
                ndim=ndim,
                replacement_ncall=replacement_ncall,
                replacement_failures=replacement_failures,
                replacement_batches=replacement_batches,
                replacement_chains_used=replacement_chains_used,
                replacement_chain_usage_counts=replacement_chain_usage_counts,
                replacement_chain_schedule=replacement_chain_schedule,
                replacement_chains=replacement_chains,
                kernel=kernel,
                walks=walks,
                bound=bound,
                bound_draws=bound_draw_history,
                bound_unit_cube_acceptance=bound_unit_cube_acceptance_history,
                bound_nellipsoids=bound_nellipsoid_history,
                rwalk_seed=rwalk_seed,
                bound_seed_kernel=bound_seed_kernel,
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
    else:
        for i in range(iteration, maxiter):
            worst = int(jnp.argmin(live_logl))
            logl_worst = float(live_logl[worst])
            logx_prev = -i / nlive
            logx_new = -(i + 1) / nlive
            logwidth = logdiffexp(logx_prev, logx_new)
            logwt = float(logwidth + logl_worst)

            dead_u_storage[i] = np.asarray(live_u[worst])
            dead_theta_storage[i] = np.asarray(live_theta[worst])
            dead_logl_storage[i] = logl_worst
            dead_logwt_storage[i] = logwt
            logz_dead = float(jnp.logaddexp(logz_dead, logwt))
            logx_final = logx_new

            replacement_info = None
            rebuild_bound = bound in {"single", "multi"} and (
                current_bound is None
                or i % bound_update_interval == 0
                or force_bound_rebuild
            )
            if rebuild_bound:
                build_start = time.perf_counter()
                if bound == "multi":
                    current_bound = build_multi_ellipsoid_bound(
                        live_u,
                        enlargement=multi_bound_enlargement or bound_enlargement,
                        jitter=bound_jitter,
                        max_ellipsoids=multi_bound_max_ellipsoids,
                        min_points=multi_bound_min_points,
                        split_threshold=multi_bound_split_threshold,
                    )
                else:
                    current_bound = build_single_ellipsoid_bound(
                        live_u, enlargement=bound_enlargement, jitter=bound_jitter
                    )
                bound_build_time_history.append(time.perf_counter() - build_start)
                bound_log_volume_history.append(
                    float(
                        current_bound.log_total_volume
                        if hasattr(current_bound, "log_total_volume")
                        else current_bound.log_volume
                    )
                )
                bound_nellipsoid_history.append(
                    int(len(getattr(current_bound, "ellipsoids", (current_bound,))))
                )
                bound_updates += 1
                if force_bound_rebuild:
                    bound_forced_rebuilds += 1
                    force_bound_rebuild = False

            bound_failure = False
            bound_success = False
            if sample == "bound":
                (
                    key,
                    new_u,
                    new_theta,
                    new_logl,
                    calls,
                    accepted,
                    replacement_info,
                ) = draw_constrained_single_bound(
                    key,
                    loglike,
                    prior_transform,
                    logl_worst,
                    current_bound,
                    ndim,
                    batch_size=batch_size,
                    max_attempts=bound_max_draws or max_attempts,
                    overlap_correction=multi_bound_overlap_correction,
                )
                bound_success = bool(accepted)
                bound_failure = not accepted
            elif sample == "prior":
                if vectorized:
                    (
                        key,
                        new_u,
                        new_theta,
                        new_logl,
                        calls,
                        accepted,
                    ) = draw_constrained_prior_vectorized(
                        key,
                        loglike,
                        prior_transform,
                        logl_worst,
                        ndim,
                        batch_size=batch_size,
                        max_attempts=max_attempts,
                    )
                else:
                    (
                        key,
                        new_u,
                        new_theta,
                        new_logl,
                        calls,
                        accepted,
                    ) = draw_constrained_prior(
                        key,
                        loglike,
                        prior_transform,
                        logl_worst,
                        ndim,
                        vectorized=False,
                        max_attempts=max_attempts,
                    )
            elif sample == "rwalk":
                rwalk_draw = (
                    draw_constrained_rwalk_jax_adaptive
                    if kernel == "jax" and replacement_chain_schedule is not None
                    else (
                        draw_constrained_rwalk_jax
                        if kernel == "jax"
                        else draw_constrained_rwalk
                    )
                )
                proposal_chol = None
                if kernel == "jax" and rwalk_proposal == "live-cov":
                    centered = live_u - jnp.mean(live_u, axis=0)
                    denom = max(int(live_u.shape[0]) - 1, 1)
                    cov = (centered.T @ centered) / denom
                    cov = cov + jnp.asarray(
                        rwalk_cov_jitter, dtype=live_u.dtype
                    ) * jnp.eye(ndim, dtype=live_u.dtype)
                    proposal_chol = jnp.linalg.cholesky(cov)
                seed_live_u = live_u
                seed_live_logl = live_logl
                if fused_bound_rwalk:
                    seed_limit = bound_max_draws or max_attempts
                    fused_draw = (
                        draw_constrained_multi_bound_rwalk_jax
                        if bound == "multi"
                        else draw_constrained_single_bound_rwalk_jax
                    )
                    draw_result = fused_draw(
                        key,
                        loglike,
                        prior_transform,
                        logl_worst,
                        current_bound,
                        ndim,
                        walks=walks,
                        step_scale=step_scale,
                        max_attempts=max_attempts,
                        min_accepts=min_accepts,
                        replacement_chains=replacement_chains,
                        replacement_chain_schedule=replacement_chain_schedule,
                        bound_batch_size=batch_size,
                        bound_max_batches=int(math.ceil(seed_limit / batch_size)),
                        proposal_chol=proposal_chol,
                        jax_vectorized=jax_vectorized,
                        **(
                            {"overlap_correction": multi_bound_overlap_correction}
                            if bound == "multi"
                            else {}
                        ),
                    )
                    replacement_info = draw_result[6]
                    bound_success = bool(draw_result[5])
                    bound_failure = not bool(draw_result[5])
                    bound_seed_call_history.append(
                        int(replacement_info.get("bound_seed_loglike_evals", 0))
                    )
                    bound_seed_batch_history.append(
                        int(replacement_info.get("bound_seed_batches", 0))
                    )
                    rwalk_kernel_call_history.append(
                        int(replacement_info.get("rwalk_kernel_calls", 0))
                    )
                elif bound in {"single", "multi"} and rwalk_seed == "bound":
                    seed_draw = (
                        draw_constrained_single_bound_jax
                        if bound_seed_kernel == "jax" and bound == "single"
                        else (
                            draw_constrained_multi_bound_jax
                            if bound_seed_kernel == "jax"
                            else draw_constrained_single_bound
                        )
                    )
                    seed_limit = bound_max_draws or max_attempts
                    seed_kwargs = {"batch_size": batch_size}
                    if bound_seed_kernel == "jax":
                        seed_kwargs["max_batches"] = int(
                            math.ceil(seed_limit / batch_size)
                        )
                        seed_kwargs["jax_vectorized"] = jax_vectorized
                        if bound == "multi":
                            seed_kwargs["overlap_correction"] = (
                                multi_bound_overlap_correction
                            )
                    else:
                        seed_kwargs["max_attempts"] = seed_limit
                        seed_kwargs["overlap_correction"] = (
                            multi_bound_overlap_correction
                        )
                    seed_result = seed_draw(
                        key,
                        loglike,
                        prior_transform,
                        logl_worst,
                        current_bound,
                        ndim,
                        **seed_kwargs,
                    )
                    (
                        key,
                        seed_u,
                        _seed_theta,
                        seed_logl,
                        seed_calls,
                        seed_accepted,
                        seed_info,
                    ) = seed_result
                    replacement_info = {
                        f"bound_seed_{key}": value for key, value in seed_info.items()
                    }
                    if seed_accepted:
                        bound_success = True
                        seed_live_u = jnp.asarray(seed_u).reshape((1, ndim))
                        seed_live_logl = jnp.asarray([seed_logl], dtype=live_logl.dtype)
                    else:
                        bound_failure = True
                    if not seed_accepted and not rwalk_seed_fallback:
                        new_u, new_theta, new_logl, calls, accepted = (
                            seed_u,
                            _seed_theta,
                            seed_logl,
                            seed_calls,
                            False,
                        )
                        draw_result = None
                        bound_seed_call_history.append(int(seed_calls))
                        bound_seed_batch_history.append(
                            int(seed_info.get("bound_seed_batches", 0))
                        )
                    elif not seed_accepted:
                        seed_calls = 0
                else:
                    seed_calls = 0
                    seed_accepted = True
                if not (
                    fused_bound_rwalk
                    or (
                        bound in {"single", "multi"}
                        and rwalk_seed == "bound"
                        and not seed_accepted
                        and not rwalk_seed_fallback
                    )
                ):
                    draw_result = rwalk_draw(
                        key,
                        loglike,
                        prior_transform,
                        logl_worst,
                        seed_live_u,
                        seed_live_logl,
                        ndim,
                        walks=walks,
                        step_scale=step_scale,
                        max_attempts=max_attempts,
                        min_accepts=min_accepts,
                        **(
                            {
                                "proposal_chol": proposal_chol,
                                "jax_vectorized": jax_vectorized,
                            }
                            if kernel == "jax"
                            else {}
                        ),
                        **(
                            {"replacement_chain_schedule": replacement_chain_schedule}
                            if kernel == "jax"
                            and replacement_chain_schedule is not None
                            else (
                                {"replacement_chains": replacement_chains}
                                if kernel == "jax"
                                else {}
                            )
                        ),
                    )
                    if seed_calls:
                        rwalk_kernel_calls = int(draw_result[4])
                        rwalk_kernel_call_history.append(rwalk_kernel_calls)
                        bound_seed_call_history.append(int(seed_calls))
                        bound_seed_batch_history.append(
                            int(seed_info.get("bound_seed_batches", 0))
                        )
                        if len(draw_result) == 7:
                            draw_result = (
                                *draw_result[:4],
                                draw_result[4] + seed_calls,
                                draw_result[5],
                                {**replacement_info, **draw_result[6]},
                            )
                        else:
                            draw_result = (
                                *draw_result[:4],
                                draw_result[4] + seed_calls,
                                draw_result[5],
                            )
                if draw_result is not None and len(draw_result) == 7:
                    (
                        key,
                        new_u,
                        new_theta,
                        new_logl,
                        calls,
                        accepted,
                        replacement_info,
                    ) = draw_result
                    replacement_batches.append(
                        int(replacement_info["replacement_batches"])
                    )
                    replacement_chains_used.append(
                        int(replacement_info["replacement_chains_used"])
                    )
                    for chain_count, count in replacement_info[
                        "replacement_chain_usage_counts"
                    ].items():
                        replacement_chain_usage_counts[chain_count] = (
                            replacement_chain_usage_counts.get(chain_count, 0)
                            + int(count)
                        )
                elif draw_result is not None:
                    key, new_u, new_theta, new_logl, calls, accepted = draw_result
                    if kernel == "jax":
                        batch_ncall = int(walks) * int(replacement_chains)
                        batches_used = int(math.ceil(int(calls) / batch_ncall))
                        chains_used = int(replacement_chains) * batches_used
                        replacement_batches.append(batches_used)
                        replacement_chains_used.append(chains_used)
                        chain_count = str(int(replacement_chains))
                        replacement_chain_usage_counts[chain_count] = (
                            replacement_chain_usage_counts.get(chain_count, 0)
                            + batches_used
                        )
                    else:
                        replacement_batches.append(1)
                        replacement_chains_used.append(1)
            elif sample == "slice":
                key, new_u, new_theta, new_logl, calls, accepted = (
                    draw_constrained_slice(
                        key,
                        loglike,
                        prior_transform,
                        logl_worst,
                        live_u,
                        live_logl,
                        ndim,
                        slices=slices,
                        slice_steps=slice_steps,
                        step_scale=step_scale,
                        max_attempts=max_attempts,
                        min_accepts=min_accepts,
                    )
                )
            else:
                key, new_u, new_theta, new_logl, calls, accepted = (
                    draw_constrained_rslice(
                        key,
                        loglike,
                        prior_transform,
                        logl_worst,
                        live_u,
                        live_logl,
                        ndim,
                        slices=slices,
                        slice_steps=slice_steps,
                        step_scale=step_scale,
                        max_attempts=max_attempts,
                        min_accepts=min_accepts,
                    )
                )
            if bound_rebuild_on_failure and bound in {"single", "multi"}:
                if bound_failure:
                    consecutive_bound_failures += 1
                    if consecutive_bound_failures >= bound_failure_rebuild_threshold:
                        force_bound_rebuild = True
                        consecutive_bound_failures = 0
                elif bound_success:
                    consecutive_bound_failures = 0

            if replacement_info is not None:
                bound_draw_history.append(
                    int(
                        replacement_info.get(
                            "bound_draws",
                            replacement_info.get(
                                "bound_seed_bound_draws",
                                replacement_info.get("bound_seed_draws", 0),
                            ),
                        )
                    )
                )
                bound_eval_history.append(
                    int(
                        replacement_info.get(
                            "bound_loglike_evals",
                            replacement_info.get(
                                "bound_seed_bound_loglike_evals",
                                replacement_info.get("bound_seed_loglike_evals", 0),
                            ),
                        )
                    )
                )
                bound_unit_cube_acceptance_history.append(
                    float(
                        replacement_info.get(
                            "bound_unit_cube_acceptance",
                            replacement_info.get(
                                "bound_seed_bound_unit_cube_acceptance",
                                replacement_info.get(
                                    "bound_seed_unit_cube_acceptance", 0.0
                                ),
                            ),
                        )
                    )
                )
                bound_overlap_rejection_history.append(
                    int(
                        replacement_info.get(
                            "bound_overlap_rejections",
                            replacement_info.get(
                                "bound_seed_bound_overlap_rejections",
                                replacement_info.get(
                                    "bound_seed_overlap_rejections", 0
                                ),
                            ),
                        )
                    )
                )
            ncall += calls
            replacement_ncall.append(int(calls))
            iteration = i + 1
            if not accepted:
                replacement_failures += 1
                success = False
                message = (
                    f"max_attempts={max_attempts} hit during constrained prior draw"
                )
                logz_remain = logx_new + float(jnp.max(live_logl))
                delta_logz = float(jnp.logaddexp(logz_dead, logz_remain) - logz_dead)
                final_delta_logz = delta_logz
                state = _make_run_state(
                    iteration=i + 1,
                    logz=logz_dead,
                    dlogz=delta_logz,
                    ncall=ncall,
                    logl_min=logl_worst,
                    logl_live_max=float(jnp.max(live_logl)),
                    sample=sample,
                    nlive=nlive,
                    ndim=ndim,
                    replacement_ncall=replacement_ncall,
                    replacement_failures=replacement_failures,
                    replacement_batches=replacement_batches,
                    replacement_chains_used=replacement_chains_used,
                    replacement_chain_usage_counts=replacement_chain_usage_counts,
                    replacement_chain_schedule=replacement_chain_schedule,
                    replacement_chains=replacement_chains,
                    kernel=kernel,
                    walks=walks,
                    bound=bound,
                    bound_draws=bound_draw_history,
                    bound_unit_cube_acceptance=bound_unit_cube_acceptance_history,
                    bound_nellipsoids=bound_nellipsoid_history,
                    rwalk_seed=rwalk_seed,
                    bound_seed_kernel=bound_seed_kernel,
                )
                if callback is not None and (
                    i + 1 == 1 or (i + 1) % callback_interval == 0
                ):
                    if callback(state) is False:
                        message = "stopped by callback"
                        stopped_by_callback = True
                if progress_printer is not None:
                    progress_printer.print(_format_progress_line(state), final=True)
                maybe_checkpoint(final=True)
                break

            other_live_logl = jnp.delete(live_logl, worst)
            insertion_index = int(
                jnp.searchsorted(jnp.sort(other_live_logl), new_logl, side="right")
            )
            insertion_indices.append(insertion_index)

            live_u = live_u.at[worst].set(new_u)
            live_theta = live_theta.at[worst].set(new_theta)
            live_logl = live_logl.at[worst].set(new_logl)
            maybe_checkpoint()

            logz_remain = logx_new + float(jnp.max(live_logl))
            delta_logz = float(jnp.logaddexp(logz_dead, logz_remain) - logz_dead)
            final_delta_logz = delta_logz
            final_iteration = delta_logz < dlogz or i + 1 == maxiter
            if i + 1 == maxiter and delta_logz >= dlogz:
                success = False
                message = f"maxiter={maxiter} reached"
            state = _make_run_state(
                iteration=i + 1,
                logz=logz_dead,
                dlogz=delta_logz,
                ncall=ncall,
                logl_min=logl_worst,
                logl_live_max=float(jnp.max(live_logl)),
                sample=sample,
                nlive=nlive,
                ndim=ndim,
                replacement_ncall=replacement_ncall,
                replacement_failures=replacement_failures,
                replacement_batches=replacement_batches,
                replacement_chains_used=replacement_chains_used,
                replacement_chain_usage_counts=replacement_chain_usage_counts,
                replacement_chain_schedule=replacement_chain_schedule,
                replacement_chains=replacement_chains,
                kernel=kernel,
                walks=walks,
                bound=bound,
                bound_draws=bound_draw_history,
                bound_unit_cube_acceptance=bound_unit_cube_acceptance_history,
                bound_nellipsoids=bound_nellipsoid_history,
                rwalk_seed=rwalk_seed,
                bound_seed_kernel=bound_seed_kernel,
            )
            if callback is not None and (
                i + 1 == 1 or (i + 1) % callback_interval == 0 or final_iteration
            ):
                if callback(state) is False:
                    success = False
                    message = "stopped by callback"
                    stopped_by_callback = True
                    final_iteration = True
            if progress_printer is not None and (
                i + 1 == 1 or (i + 1) % progress_interval == 0 or final_iteration
            ):
                progress_printer.print(
                    _format_progress_line(state), final=final_iteration
                )
            if final_iteration:
                maybe_checkpoint(final=True)
                break
        else:
            success = False
            message = f"maxiter={maxiter} reached"
            logx_final = -maxiter / nlive
            iteration = maxiter
            maybe_checkpoint(final=True)

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

    logz = float(logsumexp(logwt))
    logzerr = _logzerr(logwt, logl, logz, nlive)
    niter = int(iteration)
    nlive_final = int(live_logl.size)
    nposterior = int(logwt.size)
    replacement_initial_batch_ncall = int(walks) * int(replacement_chains)
    replacement_max_batch_ncall = replacement_initial_batch_ncall
    replacement_batch_ncall = replacement_initial_batch_ncall
    if replacement_chain_schedule is not None:
        replacement_initial_batch_ncall = int(walks) * int(
            replacement_chain_schedule[0]
        )
        replacement_max_batch_ncall = int(walks) * int(replacement_chain_schedule[-1])
        replacement_batch_ncall = replacement_max_batch_ncall
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
    bound_log_volume = None
    if current_bound is not None:
        bound_log_volume = float(
            current_bound.log_total_volume
            if hasattr(current_bound, "log_total_volume")
            else current_bound.log_volume
        )
    bound_build_count = len(bound_build_time_history)
    bound_build_time_total = float(sum(bound_build_time_history))
    bound_log_volume_final = (
        float(bound_log_volume_history[-1]) if bound_log_volume_history else None
    )
    bound_nellipsoids_final = (
        int(bound_nellipsoid_history[-1]) if bound_nellipsoid_history else None
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
            "sample": sample,
            "kernel": kernel,
            "jax_block_size": int(jax_block_size),
            "jax_block_mode": bool(jax_block_size > 1),
            "jax_block_bound_fixed": bool(jax_block_size > 1 and bound != "none"),
            "dlogz": dlogz,
            "maxiter": maxiter,
            "niter": niter,
            "ndead": niter,
            "nlive_final": nlive_final,
            "nposterior": nposterior,
            "final_delta_logz": float(final_delta_logz),
            "final_logx": float(logx_final),
            "final_logz_dead": float(logz_dead),
            "final_logl_live_max": float(jnp.max(live_logl)),
            "walks": walks,
            "step_scale": step_scale,
            "slices": slices,
            "slice_steps": slice_steps,
            "min_accepts": min_accepts,
            "rwalk_proposal": rwalk_proposal,
            "rwalk_cov_jitter": rwalk_cov_jitter,
            "replacement_chains": replacement_chains,
            "replacement_chain_schedule": (
                None
                if replacement_chain_schedule is None
                else list(replacement_chain_schedule)
            ),
            "adaptive_replacement_chains": replacement_chain_schedule is not None,
            "replacement_batch_ncall": replacement_batch_ncall,
            "replacement_initial_batch_ncall": replacement_initial_batch_ncall,
            "replacement_max_batch_ncall": replacement_max_batch_ncall,
            "batch_size": batch_size,
            "replacement_ncall": replacement_ncall,
            "insertion_indices": jnp.asarray(insertion_indices, dtype=int),
            "insertion_index_nslots": nlive,
            "insertion_index_nlive": nlive - 1,
            "replacement_failures": int(replacement_failures),
            "mean_replacement_ncall": mean_replacement_ncall,
            "max_replacement_ncall": max_replacement_ncall,
            "mean_replacement_batches": (
                float(sum(replacement_batches) / len(replacement_batches))
                if replacement_batches
                else 0.0
            ),
            "max_replacement_batches": int(max(replacement_batches, default=0)),
            "mean_replacement_chains_used": (
                float(sum(replacement_chains_used) / len(replacement_chains_used))
                if replacement_chains_used
                else 0.0
            ),
            "max_replacement_chains_used": int(max(replacement_chains_used, default=0)),
            "replacement_chain_usage_counts": replacement_chain_usage_counts,
            "replacement_acceptance_proxy": replacement_acceptance_proxy,
            "bound": bound,
            "bound_enlargement": bound_enlargement,
            "bound_update_interval": bound_update_interval,
            "bound_jitter": bound_jitter,
            "bound_max_draws": bound_max_draws,
            "bound_forced_rebuilds": int(bound_forced_rebuilds),
            "bound_rebuild_on_failure": bool(bound_rebuild_on_failure),
            "bound_failure_rebuild_threshold": int(bound_failure_rebuild_threshold),
            "multi_bound_max_ellipsoids": multi_bound_max_ellipsoids,
            "multi_bound_min_points": multi_bound_min_points,
            "multi_bound_split_threshold": multi_bound_split_threshold,
            "multi_bound_overlap_correction": multi_bound_overlap_correction,
            "bound_updates": bound_updates,
            "bound_build_time_total": bound_build_time_total,
            "bound_build_time_mean": (
                bound_build_time_total / bound_build_count if bound_build_count else 0.0
            ),
            "bound_build_time_max": (
                float(max(bound_build_time_history))
                if bound_build_time_history
                else 0.0
            ),
            "bound_build_count": bound_build_count,
            "bound_log_volume": bound_log_volume,
            "bound_log_volume_final": bound_log_volume_final,
            "bound_log_volume_mean": (
                float(sum(bound_log_volume_history) / len(bound_log_volume_history))
                if bound_log_volume_history
                else None
            ),
            "bound_log_volume_min": (
                float(min(bound_log_volume_history))
                if bound_log_volume_history
                else None
            ),
            "bound_log_volume_max": (
                float(max(bound_log_volume_history))
                if bound_log_volume_history
                else None
            ),
            "bound_nellipsoids_mean": (
                float(sum(bound_nellipsoid_history) / len(bound_nellipsoid_history))
                if bound_nellipsoid_history
                else None
            ),
            "bound_nellipsoids_max": (
                int(max(bound_nellipsoid_history, default=0))
                if bound_nellipsoid_history
                else None
            ),
            "bound_nellipsoids_final": bound_nellipsoids_final,
            "bound_seed_nellipsoids": (
                int(max(bound_nellipsoid_history, default=0))
                if bound_nellipsoid_history
                else None
            ),
            "bound_seed_overlap_rejections": (
                int(sum(bound_overlap_rejection_history))
                if bound_overlap_rejection_history
                else None
            ),
            "mean_bound_draws": (
                float(sum(bound_draw_history) / len(bound_draw_history))
                if bound_draw_history
                else None
            ),
            "max_bound_draws": (
                int(max(bound_draw_history, default=0)) if bound_draw_history else None
            ),
            "mean_bound_loglike_evals": (
                float(sum(bound_eval_history) / len(bound_eval_history))
                if bound_eval_history
                else None
            ),
            "mean_bound_unit_cube_acceptance": (
                float(
                    sum(bound_unit_cube_acceptance_history)
                    / len(bound_unit_cube_acceptance_history)
                )
                if bound_unit_cube_acceptance_history
                else None
            ),
            "rwalk_seed": rwalk_seed,
            "rwalk_seed_fallback": rwalk_seed_fallback,
            "bound_seed_kernel": "jax" if fused_bound_rwalk else bound_seed_kernel,
            "allow_unused_bound": bool(allow_unused_bound),
            "fused_bound_rwalk": bool(fused_bound_rwalk),
            "jax_vectorized": bool(jax_vectorized),
            "bounded_rwalk": bool(
                sample == "rwalk" and bound != "none" and rwalk_seed == "bound"
            ),
            "mean_bound_seed_calls": (
                float(sum(bound_seed_call_history) / len(bound_seed_call_history))
                if bound_seed_call_history
                else None
            ),
            "max_bound_seed_calls": (
                int(max(bound_seed_call_history, default=0))
                if bound_seed_call_history
                else None
            ),
            "mean_bound_seed_batches": (
                float(sum(bound_seed_batch_history) / len(bound_seed_batch_history))
                if bound_seed_batch_history
                else None
            ),
            "max_bound_seed_batches": (
                int(max(bound_seed_batch_history, default=0))
                if bound_seed_batch_history
                else None
            ),
            "mean_rwalk_kernel_calls": (
                float(sum(rwalk_kernel_call_history) / len(rwalk_kernel_call_history))
                if rwalk_kernel_call_history
                else None
            ),
            "mean_total_replacement_calls": mean_replacement_ncall,
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
