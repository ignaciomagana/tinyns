"""The host loop of :mod:`tinyns`: :func:`run` drives :func:`tinyns.core.step`.

Everything between two jitted blocks happens here: termination, the step-scale
adaptation, the progress line and the callback, the checkpoint cadence, the
wall-time telemetry, the host hook of the cluster swap
(:class:`~tinyns.clusters.ClusterTracker`), the handling of a failed
replacement, and the :class:`~tinyns.result.NestedSamplingResult`.
"""

from __future__ import annotations

import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from tinyns import checkpoint, core
from tinyns.callables import _device_leaves
from tinyns.clusters import ClusterTracker
from tinyns.result import NestedSamplingResult, _information

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

    def columns(self) -> dict:
        return {name: self[name] for name in self._columns}


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
    resume: checkpoint.Checkpoint | None = None,
) -> NestedSamplingResult:
    """Run static nested sampling block by block; return the result.

    ``key`` (a PRNG key or an int seed) starts a new run; with ``resume`` (a
    loaded :class:`~tinyns.checkpoint.Checkpoint`) the run continues from it
    instead and ``key`` is ignored. The run stops once the live-evidence remainder
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
    checkpoint_path = None if checkpoint_path is None else os.fspath(checkpoint_path)

    # Wall-time telemetry: the first block carries the compiles, so
    # throughput is measured from its end.
    start = time.perf_counter()
    first_block = None  # (seconds since start, ncall) after block 1

    # Array leaves of pytree callables are jit arguments of the compiled
    # kernels; numpy leaves are placed on the device once per run.
    loglike = _device_leaves(loglike)
    prior_transform = _device_leaves(prior_transform)

    failures = 0
    if resume is None:
        state = core.init(key, loglike, prior_transform, cfg)
        ncall, iteration, scale = nlive, 0, core._INITIAL_SCALE
        telemetry = {"rwalk_moves": 0, "rwalk_proposals": 0}
    else:
        state, telemetry = resume.state, resume.telemetry
        iteration, scale = int(state.it), float(state.scale)
        if bool(state.failed):
            raise ValueError(
                "cannot resume a checkpoint saved after a replacement failure"
            )
        if len(resume.dead["logl"]) != iteration:
            raise ValueError(
                "checkpoint dead point count must match checkpoint iteration; got "
                f"{len(resume.dead['logl'])} dead points and iteration={iteration}"
            )
        if maxiter < iteration:
            raise ValueError(
                f"maxiter={maxiter} is smaller than checkpoint iteration={iteration}"
            )
        # The host count (an int32 State.ncall may wrap): every likelihood
        # call of a run without a failed replacement is in the dead rows.
        ncall = nlive + int(resume.dead["ncall"].sum())
    real = np.asarray(state.logl).dtype
    rows = _Rows(
        u=(np.asarray(state.u).dtype, (ndim,)),
        theta=(np.asarray(state.theta).dtype, (ndim,)),
        logl=(real, ()),
        logwt=(real, ()),
        birth=(real, ()),
        ncall=(np.int64, ()),
        batches=(np.int64, ()),
    )
    if resume is not None:
        rows.extend(**{name: resume.dead[name] for name in rows.columns()})

    initial_iteration = iteration
    success, message = True, "converged"
    final_delta_logz = math.inf
    printer = _ProgressPrinter() if progress else None
    rwalk_moves = int(telemetry["rwalk_moves"])
    rwalk_proposals = int(telemetry["rwalk_proposals"])
    # Wall-time telemetry accumulates over resumes: the checkpoint holds the
    # time, compile time and timed calls of the calls before this one.
    timed_before = {
        name: telemetry.get(name, 0)
        for name in ("wall_time_s", "compile_s", "timed_ncall")
    }
    # The cluster swap's host hook (tinyns.clusters): its extras switch the
    # swap move on in core.step once two clusters can swap.
    hook = ClusterTracker(nlive) if cfg.cluster_swap else None
    if hook is not None and resume is not None:
        hook.load_state_dict(resume.ext.get("clusters"))

    last_checkpoint = initial_iteration

    def timing() -> dict[str, float]:
        """Wall time, compile time and calls timed after the first block,
        summed over this call and the ones it resumed from. Until the first
        block ends, all of this call's time counts as compile time."""
        elapsed = time.perf_counter() - start
        compile_s, calls = (elapsed, 0) if first_block is None else (
            first_block[0], ncall - first_block[1]
        )
        return {
            "wall_time_s": timed_before["wall_time_s"] + elapsed,
            "compile_s": timed_before["compile_s"] + compile_s,
            "timed_ncall": timed_before["timed_ncall"] + calls,
        }

    def maybe_checkpoint(*, final: bool = False) -> None:
        nonlocal last_checkpoint
        if checkpoint_path is None:
            return
        if final or iteration - last_checkpoint >= checkpoint_interval:
            checkpoint.save(
                checkpoint_path,
                state._replace(scale=scale),  # the host scale of the next block
                rows.columns(),
                cfg,
                {
                    "rwalk_moves": rwalk_moves,
                    "rwalk_proposals": rwalk_proposals,
                    **timing(),
                },
                ext=None if hook is None else {"clusters": hook.state_dict()},
            )
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
        extras = None if hook is None else hook.before_block(state, rows)
        state, dead = core.step(
            state._replace(scale=jnp.asarray(scale)),
            loglike,
            prior_transform,
            cfg,
            extras=extras,
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
            ncall += int(block.ncall[offset])
            logz_dead = float(logz_before)
            for logwt in block.logwt[:offset]:
                logz_dead = float(jnp.logaddexp(logz_dead, float(logwt)))
            state = state._replace(
                logz=logz_dead, logx=-(iteration + offset) / int(nlive)
            )
            block = jax.tree_util.tree_map(lambda x, n=offset: x[:n], block)
        if hook is not None:  # takes the swap steps out of moves and proposals
            block = hook.after_block(state, block)
        rows.extend(
            u=block.u,
            theta=block.theta,
            logl=block.logl,
            logwt=block.logwt,
            birth=block.birth,
            ncall=block.ncall,
            batches=block.batches,
        )
        ncall += int(sum(int(x) for x in block.ncall))
        block_moves = int(np.sum(block.moves))
        block_proposals = int(np.sum(block.proposals))
        rwalk_moves += block_moves
        rwalk_proposals += block_proposals
        if block_proposals > 0:
            observed = block_moves / block_proposals
            if math.isfinite(observed):
                scale = _update_scale(scale, observed)
        iteration = rows.n
        maybe_checkpoint()
        final_delta_logz = core.remaining_dlogz(state)
        if failed.size:
            if iteration > 0 and final_delta_logz < dlogz:
                success = True
                message = "converged after partial block before replacement failure"
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
                done = True
        if printer is not None and (
            iteration == 1 or iteration % progress_interval == 0 or done
        ):
            printer.print(_format_progress_line(report), final=done)
        if done:
            maybe_checkpoint(final=True)

    times = timing()
    mean_ms_per_call = (
        1000.0 * (times["wall_time_s"] - times["compile_s"]) / times["timed_ncall"]
        if times["timed_ncall"] > 0
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

    logz = float(logsumexp(logwt))
    logzerr = float(jnp.sqrt(_information(logwt, logl, logz) / nlive))
    metadata = {
        "walks": walks,
        "replacement_chains": chains,
        "block_size": int(block_size),
        "cluster_swap": bool(cfg.cluster_swap),
        "dlogz": dlogz,
        "acceptance": rwalk_moves / rwalk_proposals if rwalk_proposals else None,
        "scale_initial": core._INITIAL_SCALE,
        "scale_final": float(scale),
        # Summed over the calls of a resumed run. The first block of each call
        # includes the compiles, so the per-call cost skips it.
        "wall_time_s": float(times["wall_time_s"]),
        "compile_s": float(times["compile_s"]),
        "mean_ms_per_call": mean_ms_per_call,
        "final_delta_logz": float(final_delta_logz),
        "resumed": resume is not None,
    }
    if hook is not None:
        metadata.update(hook.summary())

    return NestedSamplingResult(
        samples_u=samples_u,
        samples=samples,
        logl=logl,
        logwt=logwt,
        logl_birth=logl_birth,
        logz=logz,
        logzerr=logzerr,
        ncall=ncall,
        niter=int(iteration),
        nlive=nlive,
        ndim=ndim,
        success=success,
        message=message,
        metadata=metadata,
    )
