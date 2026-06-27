"""Static nested-sampling implementation for :mod:`tinyns`."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp

from tinyns.math import logdiffexp
from tinyns.result import NestedSamplingResult
from tinyns.samplers import (
    draw_constrained_prior,
    draw_constrained_prior_vectorized,
    draw_constrained_rwalk,
    draw_constrained_slice,
)
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
    batch_size: int = 128,
    walks: int = 25,
    step_scale: float = 0.1,
    slices: int = 5,
    slice_steps: int = 10,
):
    """Run a simple static nested-sampling loop.

    The replacement ``sample`` strategy may be ``"prior"``, ``"rwalk"``, or
    ``"slice"``.
    """
    if ndim <= 0:
        raise ValueError("ndim must be a positive integer")
    if nlive <= 0:
        raise ValueError("nlive must be a positive integer")
    if sample not in {"prior", "rwalk", "slice"}:
        raise ValueError("sample must be one of {'prior', 'rwalk', 'slice'}")
    if sample == "rwalk" and vectorized:
        raise NotImplementedError(
            'vectorized rwalk is not implemented yet; use vectorized=False '
            'with sample="rwalk"'
        )
    if sample == "slice" and vectorized:
        raise NotImplementedError(
            'vectorized slice sampling is not implemented yet; use vectorized=False '
            'with sample="slice"'
        )
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if slices <= 0:
        raise ValueError("slices must be a positive integer")
    if slice_steps <= 0:
        raise ValueError("slice_steps must be a positive integer")
    if maxiter is None:
        maxiter = 10_000 * ndim
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative")

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
    logx_final = 0.0

    for i in range(maxiter):
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
            )
        else:
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
            )
        ncall += calls
        replacement_ncall.append(int(calls))
        if not accepted:
            replacement_failures += 1
            success = False
            message = f"max_attempts={max_attempts} hit during constrained prior draw"
            break

        other_live_logl = jnp.delete(live_logl, worst)
        insertion_index = int(
            jnp.searchsorted(jnp.sort(other_live_logl), new_logl, side="right")
        )
        insertion_indices.append(insertion_index)

        live_u = live_u.at[worst].set(new_u)
        live_theta = live_theta.at[worst].set(new_theta)
        live_logl = live_logl.at[worst].set(new_logl)

        logz_remain = logx_new + float(jnp.max(live_logl))
        delta_logz = float(jnp.logaddexp(logz_dead, logz_remain) - logz_dead)
        if progress and (i + 1) % 100 == 0:
            print(
                f"iter={i + 1} logz={logz_dead:.6g} "
                f"dlogz={delta_logz:.6g} ncall={ncall}"
            )
        if delta_logz < dlogz:
            break
    else:
        success = False
        message = f"maxiter={maxiter} reached"
        logx_final = -maxiter / nlive

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
            "walks": walks,
            "step_scale": step_scale,
            "slices": slices,
            "slice_steps": slice_steps,
            "batch_size": batch_size,
            "replacement_ncall": replacement_ncall,
            "insertion_indices": jnp.asarray(insertion_indices, dtype=int),
            "insertion_index_nslots": nlive,
            "insertion_index_nlive": nlive - 1,
            "replacement_failures": int(replacement_failures),
            "mean_replacement_ncall": mean_replacement_ncall,
            "max_replacement_ncall": max_replacement_ncall,
            "replacement_acceptance_proxy": replacement_acceptance_proxy,
        },
    )
