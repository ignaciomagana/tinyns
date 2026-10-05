"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

import inspect
import warnings
from typing import Any

from tinyns.result import NestedSamplingResult
from tinyns.run import _resolve_defaults, run_static_nested
from tinyns.samplers import RWALK_PROPOSALS
from tinyns.state import load_checkpoint_npz
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike

# Sampler options forwarded from ``**kwargs`` to :func:`run_static_nested`, with
# their defaults taken from its signature so the two entry points cannot drift.
# ``None`` defaults (``kernel``, ``walks``, ``step_scale``, ``rwalk_proposal``,
# ``jax_block_size``) are resolved per sampler by ``_resolve_defaults``. Unknown
# keys are still stored (dynesty drop-in compatibility) but trigger a warning so
# typos and unsupported options are not silently ignored.
_OPTION_NAMES = (
    "kernel",
    "walks",
    "step_scale",
    "batch_size",
    "min_accepts",
    "replacement_chains",
    "replacement_chain_schedule",
    "rwalk_proposal",
    "jax_vectorized",
    "jax_block_size",
    "rwalk_adaptive_step_scale",
    "rwalk_target_accept",
    "cluster_swap",
)
_RUN_PARAMETERS = inspect.signature(run_static_nested).parameters
_OPTION_DEFAULTS = {name: _RUN_PARAMETERS[name].default for name in _OPTION_NAMES}
_KNOWN_KWARGS = frozenset(_OPTION_NAMES)

# Values assumed for keys missing from checkpoints written by older versions.
# These are the historical defaults and must not follow later default changes.
_OLD_CHECKPOINT_DEFAULTS = {
    "min_accepts": 1,
    "replacement_chains": 1,
    "rwalk_proposal": "isotropic",
    "jax_vectorized": False,
    "jax_block_size": 1,
    "rwalk_adaptive_step_scale": False,
    "rwalk_target_accept": 0.25,
    "cluster_swap": False,
}


class NestedSampler:
    """Tiny dynesty-style facade over the static nested sampler.

    Parameters
    ----------
    loglike:
        Callable accepting a point in parameter space and returning its log
        likelihood. It may be a JAX pytree callable such as
        ``jax.tree_util.Partial(loglike_fn, data)``: on the fast path its array
        leaves are passed to the compiled kernels as arguments rather than
        embedded as constants, which helps when ``data`` is large.
    prior_transform:
        Callable mapping a unit-cube point to parameter space. May be a pytree
        callable, like ``loglike``.
    ndim:
        Number of model dimensions. Must be positive.
    nlive:
        Number of live points to use. Must be positive.
    vectorized:
        Whether ``loglike`` and ``prior_transform`` accept batches of points
        (``sample="prior"`` only).
    sample:
        Sampling strategy: ``"rwalk"`` (default) or ``"prior"``.
    max_attempts:
        Cap on likelihood calls per constrained replacement draw. ``None``
        resolves to ``max(10_000, walks * replacement_chains)``.
    **kwargs:
        Additional sampler options, with the defaults of
        :func:`~tinyns.run_static_nested`. By default ``sample="rwalk"`` runs
        the fast path: ``kernel="jax"``, ``rwalk_proposal="live-cov"``,
        ``jax_block_size=32``, ``walks=max(25, 6 * ndim)`` (12 for ``ndim=1``)
        and an initial ``step_scale=0.5`` that is adapted toward
        ``rwalk_target_accept``. On that path ``cluster_swap=True`` lets the
        chains swap between tracked clusters of the live points, which keeps
        the weights of separated modes from drifting; pass
        ``cluster_swap=False`` to opt out.
        Where live-cov is unsupported (``kernel="python"`` or a
        ``replacement_chain_schedule``) the defaults fall back to
        ``rwalk_proposal="isotropic"``, ``step_scale=0.1`` and
        ``jax_block_size=1``. ``sample="prior"`` defaults to ``kernel="python"``.
        ``jax_vectorized=True`` declares that JAX replacement kernels should call
        ``prior_transform`` and ``loglike`` on explicit batches instead of using
        ``jax.vmap`` around scalar callables.
    """

    def __init__(
        self,
        loglike: LogLikelihood,
        prior_transform: PriorTransform,
        ndim: int,
        nlive: int = 500,
        *,
        vectorized: bool = False,
        sample: str = "rwalk",
        max_attempts: int | None = 10_000,
        **kwargs: Any,
    ):
        if ndim <= 0:
            raise ValueError("ndim must be a positive integer")
        if nlive <= 0:
            raise ValueError("nlive must be a positive integer")
        if sample not in {"prior", "rwalk"}:
            raise ValueError("sample must be one of {'prior', 'rwalk'}")
        options = {
            **_OPTION_DEFAULTS,
            **{name: value for name, value in kwargs.items() if name in _KNOWN_KWARGS},
        }
        options.update(_resolve_defaults(ndim, sample, options))
        kernel = options["kernel"]
        if kernel not in {"python", "jax"}:
            raise ValueError("kernel must be one of {'python', 'jax'}")
        if not callable(loglike):
            raise TypeError("loglike must be callable")
        if not callable(prior_transform):
            raise TypeError("prior_transform must be callable")

        self.loglike = loglike
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.nlive = nlive
        self.vectorized = vectorized
        self.sample = sample
        self.kernel = kernel
        if max_attempts is None:
            max_attempts = max(
                10_000, int(options["walks"]) * int(options["replacement_chains"])
            )
        self.max_attempts = max_attempts

        replacement_chains = options["replacement_chains"]
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
        replacement_chain_schedule = options["replacement_chain_schedule"]
        if replacement_chain_schedule is not None and not (
            sample == "rwalk" and kernel == "jax"
        ):
            raise NotImplementedError(
                "replacement_chain_schedule is currently supported only for "
                "sample='rwalk', kernel='jax'"
            )
        rwalk_proposal = options["rwalk_proposal"]
        if rwalk_proposal not in RWALK_PROPOSALS:
            raise ValueError(f"rwalk_proposal must be one of {RWALK_PROPOSALS}")
        if rwalk_proposal == "live-cov" and not (
            sample == "rwalk" and kernel == "jax" and replacement_chain_schedule is None
        ):
            raise NotImplementedError(
                "rwalk_proposal='live-cov' is supported only for sample='rwalk', "
                "kernel='jax' and a fixed replacement_chains"
            )
        if bool(options["rwalk_adaptive_step_scale"]) and not (
            sample == "rwalk" and kernel == "jax"
        ):
            raise ValueError(
                "rwalk_adaptive_step_scale=True is supported only for "
                "sample='rwalk', kernel='jax'"
            )
        if not (0.0 < float(options["rwalk_target_accept"]) < 1.0):
            raise ValueError("rwalk_target_accept must be between 0 and 1")
        jax_block_size = options["jax_block_size"]
        if (
            not isinstance(jax_block_size, int)
            or isinstance(jax_block_size, bool)
            or jax_block_size <= 0
        ):
            raise ValueError("jax_block_size must be a positive integer")
        if jax_block_size > 1:
            if not (sample == "rwalk" and kernel == "jax"):
                raise NotImplementedError(
                    "jax_block_size > 1 is supported only for sample='rwalk', "
                    "kernel='jax'"
                )
            if replacement_chain_schedule is not None:
                raise ValueError(
                    "replacement_chain_schedule is not supported with "
                    "jax_block_size > 1; use jax_block_size=1 for adaptive "
                    "replacement-chain schedules"
                )
        unknown = sorted(set(kwargs) - _KNOWN_KWARGS)
        if unknown:
            warnings.warn(
                "NestedSampler received unknown keyword arguments (ignored): "
                + ", ".join(unknown),
                stacklevel=2,
            )
        self.kwargs = dict(kwargs)
        # Resolved options, forwarded to run_static_nested and checkpointed.
        self._options = options

    def run(
        self,
        key: PRNGKeyLike,
        *,
        dlogz: float = 0.1,
        maxiter: int | None = None,
        progress: bool = False,
        progress_interval: int = 100,
        callback=None,
        callback_interval: int = 100,
        checkpoint_path=None,
        checkpoint_interval: int = 100,
    ) -> NestedSamplingResult:
        """Run nested sampling and return a :class:`NestedSamplingResult`."""

        return run_static_nested(
            key,
            self.loglike,
            self.prior_transform,
            self.ndim,
            self.nlive,
            dlogz=dlogz,
            maxiter=maxiter,
            sample=self.sample,
            vectorized=self.vectorized,
            max_attempts=self.max_attempts,
            progress=progress,
            progress_interval=progress_interval,
            callback=callback,
            callback_interval=callback_interval,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            **self._options,
        )

    def _checkpoint_config(self) -> dict[str, object]:
        options = self._options
        schedule = options["replacement_chain_schedule"]
        return {
            "ndim": int(self.ndim),
            "nlive": int(self.nlive),
            "sample": str(self.sample),
            "kernel": str(self.kernel),
            "vectorized": bool(self.vectorized),
            "max_attempts": int(self.max_attempts),
            "batch_size": int(options["batch_size"]),
            "walks": int(options["walks"]),
            "step_scale": float(options["step_scale"]),
            "min_accepts": int(options["min_accepts"]),
            "replacement_chains": int(options["replacement_chains"]),
            "rwalk_proposal": str(options["rwalk_proposal"]),
            "replacement_chain_schedule": None if schedule is None else list(schedule),
            "jax_vectorized": bool(options["jax_vectorized"]),
            "jax_block_size": int(options["jax_block_size"]),
            # live-cov always adapts its step scale (see run_static_nested).
            "rwalk_adaptive_step_scale": bool(
                options["rwalk_adaptive_step_scale"]
                or options["rwalk_proposal"] == "live-cov"
            ),
            "rwalk_target_accept": float(options["rwalk_target_accept"]),
            "cluster_swap": bool(options["cluster_swap"]),
        }

    def _validate_checkpoint_config(self, checkpoint_config: dict) -> None:
        current = self._checkpoint_config()
        if "kernel" not in checkpoint_config:
            checkpoint_config = {**checkpoint_config, "kernel": "python"}
        if checkpoint_config.get("kernel") not in {"python", "jax"}:
            raise ValueError(
                f"checkpoint kernel={checkpoint_config.get('kernel')!r} is invalid"
            )
        for name, current_value in current.items():
            checkpoint_value = checkpoint_config.get(
                name, _OLD_CHECKPOINT_DEFAULTS.get(name)
            )
            if checkpoint_value != current_value:
                raise ValueError(
                    f"checkpoint {name}={checkpoint_value!r} is not "
                    f"compatible with sampler {name}={current_value!r}"
                )

    def resume(
        self,
        checkpoint_path,
        *,
        dlogz: float = 0.1,
        maxiter: int | None = None,
        progress: bool = False,
        progress_interval: int = 100,
        callback=None,
        callback_interval: int = 100,
        checkpoint_path_out=None,
        checkpoint_interval: int = 100,
    ) -> NestedSamplingResult:
        """Resume nested sampling from an active checkpoint ``.npz`` file."""

        state, checkpoint_config = load_checkpoint_npz(checkpoint_path)
        self._validate_checkpoint_config(checkpoint_config)
        if not state.success and "max_attempts" in state.message:
            raise ValueError(
                "cannot resume checkpoint saved after replacement failure: "
                f"{state.message}"
            )
        output_path = (
            checkpoint_path if checkpoint_path_out is None else checkpoint_path_out
        )
        return run_static_nested(
            state.key,
            self.loglike,
            self.prior_transform,
            self.ndim,
            self.nlive,
            dlogz=dlogz,
            maxiter=maxiter,
            sample=self.sample,
            vectorized=self.vectorized,
            max_attempts=self.max_attempts,
            progress=progress,
            progress_interval=progress_interval,
            callback=callback,
            callback_interval=callback_interval,
            checkpoint_path=output_path,
            checkpoint_interval=checkpoint_interval,
            initial_state=state,
            **self._options,
        )
