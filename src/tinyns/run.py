"""Static nested-sampling implementation for :mod:`tinyns`."""

from __future__ import annotations

import math
import os

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp

from tinyns.math import logdiffexp
from tinyns.result import NestedSamplingResult
from tinyns.samplers import (
    draw_constrained_prior,
    draw_constrained_prior_vectorized,
    draw_constrained_rslice,
    draw_constrained_rwalk,
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
) -> dict[str, object]:
    if replacement_ncall:
        replacement_mean_ncall_so_far = float(
            sum(replacement_ncall) / len(replacement_ncall)
        )
    else:
        replacement_mean_ncall_so_far = None
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
        "replacement_failures": int(replacement_failures),
    }


def _format_progress_line(state: dict[str, object]) -> str:
    """Format one dependency-free progress line for a run state."""

    repl = state.get("replacement_mean_ncall_so_far")
    repl_text = "n/a" if repl is None else f"{float(repl):.1f}"
    return (
        f"iter={int(state['iter']):05d} "
        f"logz={float(state['logz']):.3f} "
        f"dlogz={float(state['dlogz']):.3f} "
        f"ncall={int(state['ncall'])} "
        f"logl_min={float(state['logl_min']):.3g} "
        f"logl_live_max={float(state['logl_live_max']):.3g} "
        f"repl_ncall={repl_text} "
        f"sample={state['sample']}"
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
    if sample not in {"prior", "rwalk", "slice", "rslice"}:
        raise ValueError("sample must be one of {'prior', 'rwalk', 'slice', 'rslice'}")
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
    if maxiter is None:
        maxiter = 10_000 * ndim
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative")

    config = {
        "ndim": int(ndim),
        "nlive": int(nlive),
        "sample": str(sample),
        "vectorized": bool(vectorized),
        "max_attempts": int(max_attempts),
        "batch_size": int(batch_size),
        "walks": int(walks),
        "step_scale": float(step_scale),
        "slices": int(slices),
        "slice_steps": int(slice_steps),
        "min_accepts": int(min_accepts),
    }
    checkpoint_path_str = (
        None if checkpoint_path is None else os.fspath(checkpoint_path)
    )
    resumed_from_checkpoint = initial_state is not None

    if initial_state is None:
        key = random.PRNGKey(int(key)) if isinstance(key, int) else key
        key, init_key = random.split(key)
        live_u = random.uniform(init_key, shape=(nlive, ndim))
        live_theta = _transform_live_points(
            prior_transform, live_u, ndim, vectorized=vectorized
        )
        live_logl = _evaluate_live_points(loglike, live_theta, vectorized=vectorized)
        ncall = nlive
        dead_u = []
        dead_theta = []
        dead_logl = []
        dead_logwt = []
        logz_dead = -math.inf
        replacement_ncall = []
        insertion_indices = []
        replacement_failures = 0
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
        dead_u = list(initial_state.dead_u)
        dead_theta = list(initial_state.dead_theta)
        dead_logl = list(initial_state.dead_logl)
        dead_logwt = list(initial_state.dead_logwt)
        logz_dead = float(initial_state.logz_dead)
        logx_final = float(initial_state.logx_final)
        ncall = int(initial_state.ncall)
        replacement_ncall = list(initial_state.replacement_ncall)
        insertion_indices = list(initial_state.insertion_indices)
        replacement_failures = int(initial_state.replacement_failures)
        success = True
        message = "converged"
        stopped_by_callback = False
        iteration = int(initial_state.iteration)
        if maxiter < iteration:
            raise ValueError(
                f"maxiter={maxiter} is smaller than checkpoint iteration={iteration}"
            )

    initial_iteration = iteration
    final_delta_logz = math.inf
    progress_printer = _ProgressPrinter() if progress else None

    def current_state() -> NestedRunState:
        return NestedRunState(
            key=key,
            live_u=live_u,
            live_theta=live_theta,
            live_logl=live_logl,
            dead_u=dead_u,
            dead_theta=dead_theta,
            dead_logl=dead_logl,
            dead_logwt=dead_logwt,
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

    for i in range(iteration, maxiter):
        worst = int(jnp.argmin(live_logl))
        logl_worst = float(live_logl[worst])
        logx_prev = -i / nlive
        logx_new = -(i + 1) / nlive
        logwidth = logdiffexp(logx_prev, logx_new)
        logwt = float(logwidth + logl_worst)

        dead_u.append(live_u[worst])
        dead_theta.append(live_theta[worst])
        dead_logl.append(logl_worst)
        dead_logwt.append(logwt)
        logz_dead = float(jnp.logaddexp(logz_dead, logwt))
        logx_final = logx_new

        if sample == "prior":
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
            key, new_u, new_theta, new_logl, calls, accepted = draw_constrained_rwalk(
                key,
                loglike,
                prior_transform,
                logl_worst,
                live_u,
                live_logl,
                ndim,
                walks=walks,
                step_scale=step_scale,
                max_attempts=max_attempts,
                min_accepts=min_accepts,
            )
        elif sample == "slice":
            key, new_u, new_theta, new_logl, calls, accepted = draw_constrained_slice(
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
        else:
            key, new_u, new_theta, new_logl, calls, accepted = draw_constrained_rslice(
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
        ncall += calls
        replacement_ncall.append(int(calls))
        iteration = i + 1
        if not accepted:
            replacement_failures += 1
            success = False
            message = f"max_attempts={max_attempts} hit during constrained prior draw"
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
            progress_printer.print(_format_progress_line(state), final=final_iteration)
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

    if dead_u:
        samples_u = jnp.concatenate([jnp.stack(dead_u), live_u], axis=0)
        samples = jnp.concatenate([jnp.stack(dead_theta), live_theta], axis=0)
        logl = jnp.concatenate([jnp.asarray(dead_logl), live_logl], axis=0)
        logwt = jnp.concatenate([jnp.asarray(dead_logwt), live_logwt], axis=0)
    else:
        samples_u = live_u
        samples = live_theta
        logl = live_logl
        logwt = live_logwt

    logz = float(logsumexp(logwt))
    logzerr = _logzerr(logwt, logl, logz, nlive)
    niter = len(dead_logl)
    nlive_final = int(live_logl.size)
    nposterior = int(logwt.size)
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
            "batch_size": batch_size,
            "replacement_ncall": replacement_ncall,
            "insertion_indices": jnp.asarray(insertion_indices, dtype=int),
            "insertion_index_nslots": nlive,
            "insertion_index_nlive": nlive - 1,
            "replacement_failures": int(replacement_failures),
            "mean_replacement_ncall": mean_replacement_ncall,
            "max_replacement_ncall": max_replacement_ncall,
            "replacement_acceptance_proxy": replacement_acceptance_proxy,
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
