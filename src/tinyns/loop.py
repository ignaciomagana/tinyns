"""The host loop of :mod:`tinyns`: :func:`run` drives :func:`tinyns.core.step`.

Everything between two jitted blocks happens here: termination, the step-scale
adaptation, the progress line and the callback, the checkpoint cadence, the
wall-time telemetry, the cluster tracker of the swap move, the handling of a
failed replacement, and the :class:`~tinyns.result.NestedSamplingResult`.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from tinyns import core
from tinyns.callables import _device_leaves
from tinyns.clusters import ClusterTracker
from tinyns.result import NestedSamplingResult, _logzerr_diagnostics
from tinyns.state import NestedRunState, save_checkpoint_npz

# The live-cov step is ``scale * L z``. The scale starts at 0.5 and, after
# every block, moves in log space toward 25% move acceptance. It is a
# dimensionless O(1) factor (the step already follows the contracting live
# set), so the update can chase it fast.
_TARGET_ACCEPT = 0.25
_SCALE_RATE = 0.5
_MIN_SCALE = 1e-3
_MAX_SCALE = 10.0


def _update_scale(current_scale: float, observed_accept: float) -> float:
    """Return the log-space adapted live-cov step scale, clamped to bounds."""
    log_scale = math.log(float(current_scale))
    delta = _SCALE_RATE * float(np.clip(observed_accept - _TARGET_ACCEPT, -0.5, 0.5))
    new_scale = math.exp(log_scale + delta)
    return float(np.clip(new_scale, _MIN_SCALE, _MAX_SCALE))


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


class _Rows:
    """Dead rows on the host: numpy columns that grow by doubling."""

    def __init__(self, **columns):  # name=(dtype, trailing shape)
        self.n = 0
        self._columns = {
            name: np.empty((0, *shape), dtype=dtype)
            for name, (dtype, shape) in columns.items()
        }

    def extend(self, **values) -> None:
        k = len(next(iter(values.values())))
        for name, value in values.items():
            column = self._columns[name]
            if self.n + k > len(column):
                grown = np.empty(
                    (max(2 * len(column), self.n + k, 64),) + column.shape[1:],
                    dtype=column.dtype,
                )
                grown[: self.n] = column[: self.n]
                self._columns[name] = column = grown
            column[self.n : self.n + k] = value
        self.n += k

    def __getitem__(self, name):
        return self._columns[name][: self.n]


def _check_options(
    maxiter,
    progress_interval,
    callback,
    callback_interval,
    checkpoint_path,
    checkpoint_interval,
) -> None:
    if progress_interval <= 0:
        raise ValueError("progress_interval must be a positive integer")
    if callback_interval <= 0:
        raise ValueError("callback_interval must be a positive integer")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    if checkpoint_path is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")
    if maxiter is not None and (
        not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer")


def _resumed_state(resume: NestedRunState, maxiter: int) -> core.State:
    """Rebuild the :class:`core.State` of a checkpoint, after sanity checks."""
    ndead = len(resume.dead_logl)
    if (
        not len(resume.dead_u)
        == len(resume.dead_theta)
        == ndead
        == len(resume.dead_logwt)
    ):
        raise ValueError("checkpoint dead point arrays have inconsistent lengths")
    iteration = int(resume.iteration)
    if maxiter < iteration:
        raise ValueError(
            f"maxiter={maxiter} is smaller than checkpoint iteration={iteration}"
        )
    if ndead != iteration:
        raise ValueError(
            "checkpoint dead point count must match checkpoint iteration; "
            f"got {ndead} dead points and iteration={iteration}"
        )
    live_logl = resume.live_logl
    birth = resume.live_birth
    return core.State(
        key=resume.key,
        u=resume.live_u,
        theta=resume.live_theta,
        logl=live_logl,
        birth=(
            jnp.full_like(live_logl, jnp.nan)
            if birth is None
            else jnp.asarray(birth, dtype=live_logl.dtype)
        ),
        # Host floats, as checkpointed; see core.State.
        logz=float(resume.logz_dead),
        logx=float(resume.logx_final),
        it=jnp.asarray(iteration, dtype=jnp.int32),
        scale=None,  # the host scale, set by the loop
        ncall=jnp.asarray(int(resume.ncall), dtype=jnp.int32),
        failed=jnp.asarray(False),
    )


def run(
    cfg: core.Config,
    loglike,
    prior_transform,
    key,
    *,
    dlogz: float = 0.1,
    maxiter: int | None = None,
    progress: bool = False,
    progress_interval: int = 100,
    callback=None,
    callback_interval: int = 100,
    checkpoint_path=None,
    checkpoint_interval: int = 100,
    resume_state: NestedRunState | None = None,
) -> NestedSamplingResult:
    """Run static nested sampling block by block; return the result.

    ``key`` (a PRNG key or an int seed) starts a new run; with
    ``resume_state`` (a loaded checkpoint) the run continues from it instead
    and ``key`` is ignored. The run stops once the live-evidence remainder
    falls below ``dlogz``, at ``maxiter`` iterations (default
    ``10_000 * ndim``), when ``callback`` returns ``False``, or after a failed
    replacement. Progress and callbacks fire on block boundaries, at
    iteration 1, every ``*_interval`` iterations and at the end; a checkpoint
    is written once at least ``checkpoint_interval`` iterations have passed
    since the last one, and at the end.
    """
    _check_options(
        maxiter,
        progress_interval,
        callback,
        callback_interval,
        checkpoint_path,
        checkpoint_interval,
    )
    ndim, nlive, walks = cfg.ndim, cfg.nlive, cfg.walks
    chains, block_size = cfg.replacement_chains, cfg.block_size
    if maxiter is None:
        maxiter = 10_000 * ndim
    config = dataclasses.asdict(cfg)
    checkpoint_path = None if checkpoint_path is None else os.fspath(checkpoint_path)
    restored = dict(resume_state.telemetry or {}) if resume_state is not None else {}

    # Wall-time telemetry: the first block carries the compiles, so
    # throughput is measured from its end.
    start = time.perf_counter()
    first_block = None  # (seconds since start, ncall) after block 1

    # Array leaves of pytree callables are jit arguments of the compiled
    # kernels; numpy leaves are placed on the device once per run.
    loglike = _device_leaves(loglike)
    prior_transform = _device_leaves(prior_transform)

    if resume_state is None:
        state = core.init(key, loglike, prior_transform, cfg)
        ncall, failures, iteration = nlive, 0, 0
    else:
        state = _resumed_state(resume_state, maxiter)
        ncall = int(resume_state.ncall)
        failures = int(resume_state.replacement_failures)
        iteration = int(resume_state.iteration)
    real = np.asarray(state.logl).dtype
    rows = _Rows(
        u=(np.asarray(state.u).dtype, (ndim,)),
        theta=(np.asarray(state.theta).dtype, (ndim,)),
        logl=(real, ()),
        logwt=(real, ()),
        birth=(real, ()),
        ncall=(np.int64, ()),
        insertion=(np.int64, ()),
        batches=(np.int64, ()),
    )
    if resume_state is not None and iteration:
        dead_birth = resume_state.dead_birth
        rows.extend(
            u=np.asarray(resume_state.dead_u),
            theta=np.asarray(resume_state.dead_theta),
            logl=np.asarray(resume_state.dead_logl),
            logwt=np.asarray(resume_state.dead_logwt),
            birth=np.full(iteration, np.nan) if dead_birth is None else dead_birth,
            ncall=np.asarray(resume_state.replacement_ncall, dtype=np.int64),
            insertion=np.asarray(resume_state.insertion_indices, dtype=np.int64),
            # Unknown without the telemetry: counted as one batch each.
            batches=restored.get("replacement_batches", np.ones(iteration, np.int64)),
        )

    initial_iteration = iteration
    success, message, stopped_by_callback = True, "converged", False
    final_delta_logz = math.inf
    partial_failure = {"offset": None, "delta_logz": None, "message": None}
    terminated_after_partial_failure = False
    printer = _ProgressPrinter() if progress else None
    rwalk_moves = int(restored.get("rwalk_moves", 0))
    rwalk_proposals = int(restored.get("rwalk_proposals", 0))
    scale = core._INITIAL_SCALE
    if resume_state is not None and resume_state.scale is not None:
        if math.isfinite(float(resume_state.scale)):
            scale = float(resume_state.scale)
    scale_history = [float(x) for x in restored.get("adaptive_scale_history", [])]
    if not scale_history or scale_history[-1] != scale:
        scale_history.append(scale)
    accept_history = [float(x) for x in restored.get("adaptive_accept_history", [])]
    # Clusters are tracked on the host between blocks; the tracker never
    # touches the PRNG stream, and the swap kernel only runs (and is only
    # compiled) once two clusters can swap.
    tracker = None
    cluster_host_s = 0.0
    if cfg.cluster_swap:
        stored = (resume_state.clusters if resume_state is not None else None) or {}
        tracker = ClusterTracker(nlive, stored.get("arrays"), stored.get("log"))

    def checkpoint_state() -> NestedRunState:
        return NestedRunState(
            key=state.key,
            live_u=state.u,
            live_theta=state.theta,
            live_logl=state.logl,
            live_birth=state.birth,
            dead_u=rows["u"],
            dead_theta=rows["theta"],
            dead_logl=rows["logl"],
            dead_logwt=rows["logwt"],
            dead_birth=rows["birth"],
            logz_dead=state.logz,
            logx_final=state.logx,
            ncall=ncall,
            replacement_ncall=rows["ncall"],
            insertion_indices=rows["insertion"],
            replacement_failures=failures,
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
                "replacement_batches": rows["batches"].tolist(),
                "rwalk_moves": rwalk_moves,
                "rwalk_proposals": rwalk_proposals,
                "adaptive_scale_history": list(scale_history),
                "adaptive_accept_history": list(accept_history),
            },
        )

    last_checkpoint = initial_iteration

    def maybe_checkpoint(*, final: bool = False) -> None:
        nonlocal last_checkpoint
        if checkpoint_path is None:
            return
        if final or iteration - last_checkpoint >= checkpoint_interval:
            save_checkpoint_npz(checkpoint_path, checkpoint_state(), config)
            last_checkpoint = iteration

    def run_state(delta_logz: float, logl_min: float) -> dict[str, object]:
        """The callback/progress dict; the first call ends the first block."""
        nonlocal first_block
        elapsed = time.perf_counter() - start
        if first_block is None:
            first_block = (elapsed, ncall)
        calls = ncall - first_block[1]
        seconds = elapsed - first_block[0]
        n = rows.n
        batches = rows["batches"]
        return {
            "iter": int(iteration),
            "logz": float(state.logz),
            "dlogz": float(delta_logz),
            "ncall": int(ncall),
            "calls_per_s": calls / seconds if calls > 0 and seconds > 0 else None,
            "logl_min": float(logl_min),
            "logl_live_max": float(jnp.max(state.logl)),
            "nlive": int(nlive),
            "ndim": int(ndim),
            "replacement_mean_ncall_so_far": (
                float(int(rows["ncall"].sum()) / n) if n else None
            ),
            "replacement_mean_batches_so_far": (
                float(int(batches.sum()) / n) if n else None
            ),
            "replacement_max_batches_so_far": int(batches.max()) if n else None,
            "replacement_chains": chains,
            "walks": walks,
            "replacement_failures": int(failures),
        }

    # A resumed run may already be terminal: converged, or checkpointed at
    # iteration >= maxiter without converging.
    done = False
    if iteration > 0:
        final_delta_logz = core.remaining_dlogz(state)
        if final_delta_logz >= dlogz and iteration >= maxiter:
            success, message = False, f"maxiter={maxiter} reached"
        done = final_delta_logz < dlogz or iteration >= maxiter
        if done:
            maybe_checkpoint(final=True)

    while not done:
        # One compiled block serves every block: the maxiter tail runs the
        # same program with fewer active iterations.
        n_active = min(block_size, maxiter - iteration)
        logz_before = state.logz
        frames = None
        if tracker is not None:
            host_start = time.perf_counter()
            frames = tracker.frames(state.u, iteration, rows["u"], rows["logwt"])
            cluster_host_s += time.perf_counter() - host_start
        state, dead = core.step(
            state._replace(scale=jnp.asarray(scale)),
            loglike,
            prior_transform,
            cfg,
            extras=frames,
            n_active=n_active,
        )
        block = jax.tree_util.tree_map(
            lambda x, n=n_active: x[:n], jax.device_get(dead)
        )
        failed = np.flatnonzero(~block.valid)
        if failed.size:
            # The step left the live set and the key as they were after the
            # last successful replacement; keep the dead points before the
            # failed one.
            failures += 1
            success = False
            offset = int(failed[0])
            message = (
                f"replacement failed at iteration {iteration + offset + 1}: no "
                f"chain ended above the likelihood threshold in {cfg.max_batches} "
                f"batches of {walks} steps x {chains} chains"
            )
            partial_failure["offset"] = offset
            partial_failure["message"] = message
            ncall += int(block.ncall[offset])
            logz_dead = float(logz_before)
            for logwt in block.logwt[:offset]:
                logz_dead = float(jnp.logaddexp(logz_dead, float(logwt)))
            state = state._replace(
                logz=logz_dead, logx=-(iteration + offset) / int(nlive)
            )
            block = jax.tree_util.tree_map(lambda x, n=offset: x[:n], block)
        moves, proposals = block.moves, block.proposals
        if block.swaps is not None:
            # The rwalk telemetry and the scale adaptation count rwalk steps
            # only; swaps are counted on their own.
            moves = moves - block.swaps[:, 0]
            proposals = proposals - block.swaps[:, 1]
            tracker.log["swap"] = [
                int(a + b)
                for a, b in zip(tracker.log["swap"], block.swaps.sum(0), strict=True)
            ]
        rows.extend(
            u=block.u,
            theta=block.theta,
            logl=block.logl,
            logwt=block.logwt,
            birth=block.birth,
            ncall=block.ncall,
            insertion=block.insertion,
            batches=block.batches,
        )
        ncall += int(sum(int(x) for x in block.ncall))
        block_moves, block_proposals = int(np.sum(moves)), int(np.sum(proposals))
        rwalk_moves += block_moves
        rwalk_proposals += block_proposals
        if block_proposals > 0:
            observed = block_moves / block_proposals
            if math.isfinite(observed):
                accept_history.append(observed)
                scale = _update_scale(scale, observed)
                scale_history.append(scale)
        iteration = rows.n
        maybe_checkpoint()
        final_delta_logz = core.remaining_dlogz(state)
        if failed.size:
            partial_failure["delta_logz"] = float(final_delta_logz)
            if iteration > 0 and final_delta_logz < dlogz:
                success = True
                message = "converged after partial block before replacement failure"
                terminated_after_partial_failure = True
            maybe_checkpoint(final=True)
            break

        done = final_delta_logz < dlogz or iteration == maxiter
        if iteration == maxiter and final_delta_logz >= dlogz:
            success, message = False, f"maxiter={maxiter} reached"
        report = run_state(final_delta_logz, block.logl[-1])
        if callback is not None and (
            iteration == 1 or iteration % callback_interval == 0 or done
        ):
            if callback(report) is False:
                success, message = False, "stopped by callback"
                stopped_by_callback = done = True
        if printer is not None and (
            iteration == 1 or iteration % progress_interval == 0 or done
        ):
            printer.print(_format_progress_line(report), final=done)
        if done:
            maybe_checkpoint(final=True)

    wall_time_s = time.perf_counter() - start
    compile_s = None if first_block is None else first_block[0]
    calls_after_first_block = 0 if first_block is None else ncall - first_block[1]
    mean_ms_per_call = (
        1000.0 * (wall_time_s - compile_s) / calls_after_first_block
        if calls_after_first_block > 0
        else None
    )

    live_u, live_logl = state.u, state.logl
    live_logwt = state.logx - math.log(nlive) + live_logl
    live = (live_u, state.theta, live_logl, live_logwt, state.birth)
    if iteration:
        dead = [jnp.asarray(rows[k]) for k in ("u", "theta", "logl", "logwt", "birth")]
        samples_u, samples, logl, logwt, logl_birth = (
            jnp.concatenate([d, x], axis=0) for d, x in zip(dead, live, strict=True)
        )
    else:
        samples_u, samples, logl, logwt, logl_birth = live

    nlive_final = int(live_logl.size)
    logz = float(logsumexp(logwt))
    logzerr, logzerr_diagnostics = _logzerr_diagnostics(
        logwt, logl, logz, nlive, nlive_final
    )
    replacement_ncall = rows["ncall"].tolist()
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
    replacement_batches = rows["batches"].tolist()
    rwalk_acceptance = rwalk_moves / rwalk_proposals if rwalk_proposals > 0 else None

    cluster_metadata = {"cluster_swap": bool(cfg.cluster_swap)}
    if tracker is not None:
        host_start = time.perf_counter()
        cluster_metadata.update(
            tracker.summary(
                rows["u"], rows["logwt"], np.asarray(live_u), np.asarray(live_logwt)
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
        logl_birth=logl_birth,
        metadata={
            "rwalk_scale_initial": core._INITIAL_SCALE,
            "rwalk_scale_final": float(scale),
            "rwalk_scale_min_seen": float(min(scale_history)),
            "rwalk_scale_max_seen": float(max(scale_history)),
            "rwalk_scale_mean": float(sum(scale_history) / len(scale_history)),
            "rwalk_adaptation_updates": len(accept_history),
            "rwalk_observed_accept_mean": (
                float(sum(accept_history) / len(accept_history))
                if accept_history
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
            "niter": int(iteration),
            "ndead": int(iteration),
            "nlive_final": nlive_final,
            "nposterior": int(logwt.size),
            **logzerr_diagnostics,
            "final_delta_logz": float(final_delta_logz),
            "final_logx": float(state.logx),
            "final_logz_dead": float(state.logz),
            "final_logl_live_max": float(jnp.max(live_logl)),
            "walks": walks,
            "replacement_chains": chains,
            "replacement_batch_ncall": int(walks) * int(chains),
            "replacement_ncall": replacement_ncall,
            "insertion_indices": jnp.asarray(rows["insertion"], dtype=int),
            "insertion_index_nslots": nlive,
            "insertion_index_nlive": nlive - 1,
            "replacement_failures": int(failures),
            "terminated_after_partial_block_failure": terminated_after_partial_failure,
            "partial_block_failure_delta_logz": partial_failure["delta_logz"],
            "partial_block_failure_offset": partial_failure["offset"],
            "partial_block_failure_message": partial_failure["message"],
            "mean_replacement_ncall": mean_replacement_ncall,
            "max_replacement_ncall": max_replacement_ncall,
            "mean_replacement_batches": (
                float(sum(replacement_batches) / len(replacement_batches))
                if replacement_batches
                else 0.0
            ),
            "max_replacement_batches": int(max(replacement_batches, default=0)),
            "replacement_acceptance_proxy": replacement_acceptance_proxy,
            "accepted_rwalk_moves": rwalk_moves,
            "total_rwalk_proposals": rwalk_proposals,
            "rwalk_acceptance": rwalk_acceptance,
            "mean_rwalk_acceptance": rwalk_acceptance,
            "progress_interval": progress_interval,
            "callback_interval": callback_interval,
            "stopped_by_callback": bool(stopped_by_callback),
            "checkpoint_path": checkpoint_path,
            "checkpoint_interval": (
                checkpoint_interval if checkpoint_path is not None else None
            ),
            "resumed_from_checkpoint": resume_state is not None,
            "initial_iteration": int(initial_iteration),
            "final_iteration": int(iteration),
        },
    )
