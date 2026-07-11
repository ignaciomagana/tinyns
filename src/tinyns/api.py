"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

import warnings
from typing import Any

from tinyns.result import NestedSamplingResult
from tinyns.run import run_static_nested
from tinyns.state import load_checkpoint_npz
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike

# Every keyword argument consumed from ``**kwargs`` anywhere in this module.
# Unknown keys are still stored (dynesty drop-in compatibility) but trigger a
# warning so typos and unsupported options are not silently ignored.
_KNOWN_KWARGS = frozenset(
    {
        "kernel",
        "bound",
        "rwalk_seed",
        "bound_seed_kernel",
        "replacement_chains",
        "replacement_chain_schedule",
        "rwalk_adaptive_step_scale",
        "rwalk_target_accept",
        "fused_bound_rwalk",
        "jax_block_size",
        "walks",
        "step_scale",
        "batch_size",
        "min_accepts",
        "rwalk_proposal",
        "bound_enlargement",
        "bound_update_interval",
        "bound_jitter",
        "bound_max_draws",
        "multi_bound_max_ellipsoids",
        "multi_bound_min_points",
        "multi_bound_split_threshold",
        "multi_bound_enlargement",
        "multi_bound_overlap_correction",
        "rwalk_seed_fallback",
        "allow_unused_bound",
        "bound_rebuild_on_failure",
        "bound_failure_rebuild_threshold",
        "jax_vectorized",
    }
)


class NestedSampler:
    """Tiny dynesty-style facade over the static nested sampler.

    Parameters
    ----------
    loglike:
        Callable accepting a point in parameter space and returning its log
        likelihood.
    prior_transform:
        Callable mapping a unit-cube point to parameter space.
    ndim:
        Number of model dimensions. Must be positive.
    nlive:
        Number of live points to use. Must be positive.
    vectorized:
        Whether ``loglike`` and ``prior_transform`` accept batches of points.
    sample:
        Sampling strategy. ``"prior"`` and ``"rwalk"`` are currently supported.
    max_attempts:
        Cap on rejection attempts per constrained prior draw.
    **kwargs:
        Additional sampler options. ``walks``, ``step_scale``, and
        ``min_accepts`` are used by ``sample="rwalk"``. ``kernel`` may be
        ``"python"`` (default) or experimental ``"jax"`` for ``sample="rwalk"``.
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
        sample: str = "prior",
        max_attempts: int = 10_000,
        **kwargs: Any,
    ):
        if ndim <= 0:
            raise ValueError("ndim must be a positive integer")
        if nlive <= 0:
            raise ValueError("nlive must be a positive integer")
        if sample not in {"prior", "rwalk"}:
            raise ValueError("sample must be one of {'prior', 'rwalk'}")
        kernel = kwargs.get("kernel", "python")
        if kernel not in {"python", "jax"}:
            raise ValueError("kernel must be one of {'python', 'jax'}")
        bound = kwargs.get("bound", "none")
        if bound not in {"none", "single", "multi"}:
            raise ValueError("bound must be one of {'none', 'single', 'multi'}")
        rwalk_seed = kwargs.get("rwalk_seed", "live")
        if rwalk_seed not in {"live", "bound"}:
            raise ValueError("rwalk_seed must be one of {'live', 'bound'}")
        bound_seed_kernel = kwargs.get("bound_seed_kernel", "python")
        if bound_seed_kernel not in {"python", "jax"}:
            raise ValueError("bound_seed_kernel must be one of {'python', 'jax'}")
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
        self.max_attempts = max_attempts

        replacement_chains = kwargs.get("replacement_chains", 1)
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
        replacement_chain_schedule = kwargs.get("replacement_chain_schedule")
        if replacement_chain_schedule is not None and not (
            sample == "rwalk" and kernel == "jax"
        ):
            raise NotImplementedError(
                "replacement_chain_schedule is currently supported only for "
                "sample='rwalk', kernel='jax'"
            )
        if kwargs.get("rwalk_proposal", "isotropic") != "isotropic":
            raise ValueError(
                "rwalk_proposal='live-cov' has been removed; only "
                "rwalk_proposal='isotropic' is supported"
            )
        if bool(kwargs.get("rwalk_adaptive_step_scale", False)) and not (
            sample == "rwalk" and kernel == "jax"
        ):
            raise ValueError(
                "rwalk_adaptive_step_scale=True is supported only for "
                "sample='rwalk', kernel='jax'"
            )
        if not (0.0 < float(kwargs.get("rwalk_target_accept", 0.25)) < 1.0):
            raise ValueError("rwalk_target_accept must be between 0 and 1")
        fused_bound_rwalk = bool(kwargs.get("fused_bound_rwalk", False))
        jax_block_size = kwargs.get("jax_block_size", 1)
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
                    "block mode with bound in {'single', 'multi'}, "
                    "rwalk_seed='bound', bound_seed_kernel='jax', and "
                    "fused_bound_rwalk=True"
                )
            if replacement_chain_schedule is not None:
                raise ValueError(
                    "replacement_chain_schedule is not supported with "
                    "jax_block_size > 1; use jax_block_size=1 for adaptive "
                    "replacement-chain schedules"
                )
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
        unknown = sorted(set(kwargs) - _KNOWN_KWARGS)
        if unknown:
            warnings.warn(
                "NestedSampler received unknown keyword arguments (ignored): "
                + ", ".join(unknown),
                stacklevel=2,
            )
        self.kwargs = dict(kwargs)

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
            kernel=self.kernel,
            vectorized=self.vectorized,
            max_attempts=self.max_attempts,
            progress=progress,
            progress_interval=progress_interval,
            callback=callback,
            callback_interval=callback_interval,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            walks=self.kwargs.get("walks", 25),
            step_scale=self.kwargs.get("step_scale", 0.1),
            batch_size=self.kwargs.get("batch_size", 128),
            min_accepts=self.kwargs.get("min_accepts", 1),
            replacement_chains=self.kwargs.get("replacement_chains", 1),
            replacement_chain_schedule=self.kwargs.get("replacement_chain_schedule"),
            rwalk_proposal=self.kwargs.get("rwalk_proposal", "isotropic"),
            bound=self.kwargs.get("bound", "none"),
            bound_enlargement=self.kwargs.get("bound_enlargement", 1.25),
            bound_update_interval=self.kwargs.get("bound_update_interval", 1),
            bound_jitter=self.kwargs.get("bound_jitter", 1e-6),
            bound_max_draws=self.kwargs.get("bound_max_draws"),
            multi_bound_max_ellipsoids=self.kwargs.get(
                "multi_bound_max_ellipsoids", 32
            ),
            multi_bound_min_points=self.kwargs.get("multi_bound_min_points"),
            multi_bound_split_threshold=self.kwargs.get(
                "multi_bound_split_threshold", 0.9
            ),
            multi_bound_enlargement=self.kwargs.get("multi_bound_enlargement"),
            multi_bound_overlap_correction=self.kwargs.get(
                "multi_bound_overlap_correction", True
            ),
            rwalk_seed=self.kwargs.get("rwalk_seed", "live"),
            rwalk_seed_fallback=self.kwargs.get("rwalk_seed_fallback", True),
            bound_seed_kernel=self.kwargs.get("bound_seed_kernel", "python"),
            allow_unused_bound=self.kwargs.get("allow_unused_bound", False),
            fused_bound_rwalk=self.kwargs.get("fused_bound_rwalk", False),
            bound_rebuild_on_failure=self.kwargs.get("bound_rebuild_on_failure", False),
            bound_failure_rebuild_threshold=self.kwargs.get(
                "bound_failure_rebuild_threshold", 1
            ),
            jax_vectorized=self.kwargs.get("jax_vectorized", False),
            jax_block_size=self.kwargs.get("jax_block_size", 1),
            rwalk_adaptive_step_scale=self.kwargs.get(
                "rwalk_adaptive_step_scale", False
            ),
            rwalk_target_accept=self.kwargs.get("rwalk_target_accept", 0.25),
        )

    def _checkpoint_config(self) -> dict[str, object]:
        return {
            "ndim": int(self.ndim),
            "nlive": int(self.nlive),
            "sample": str(self.sample),
            "kernel": str(self.kernel),
            "vectorized": bool(self.vectorized),
            "max_attempts": int(self.max_attempts),
            "batch_size": int(self.kwargs.get("batch_size", 128)),
            "walks": int(self.kwargs.get("walks", 25)),
            "step_scale": float(self.kwargs.get("step_scale", 0.1)),
            "min_accepts": int(self.kwargs.get("min_accepts", 1)),
            "replacement_chains": int(self.kwargs.get("replacement_chains", 1)),
            "rwalk_proposal": str(self.kwargs.get("rwalk_proposal", "isotropic")),
            "replacement_chain_schedule": (
                None
                if self.kwargs.get("replacement_chain_schedule") is None
                else list(self.kwargs.get("replacement_chain_schedule"))
            ),
            "bound": str(self.kwargs.get("bound", "none")),
            "bound_enlargement": float(self.kwargs.get("bound_enlargement", 1.25)),
            "bound_update_interval": int(self.kwargs.get("bound_update_interval", 1)),
            "bound_jitter": float(self.kwargs.get("bound_jitter", 1e-6)),
            "bound_max_draws": self.kwargs.get("bound_max_draws"),
            "multi_bound_max_ellipsoids": int(
                self.kwargs.get("multi_bound_max_ellipsoids", 32)
            ),
            "multi_bound_min_points": self.kwargs.get("multi_bound_min_points"),
            "multi_bound_split_threshold": float(
                self.kwargs.get("multi_bound_split_threshold", 0.9)
            ),
            "multi_bound_enlargement": self.kwargs.get("multi_bound_enlargement"),
            "multi_bound_overlap_correction": bool(
                self.kwargs.get("multi_bound_overlap_correction", True)
            ),
            "rwalk_seed": str(self.kwargs.get("rwalk_seed", "live")),
            "rwalk_seed_fallback": bool(self.kwargs.get("rwalk_seed_fallback", True)),
            "bound_seed_kernel": (
                "jax"
                if self.kwargs.get("fused_bound_rwalk", False)
                else str(self.kwargs.get("bound_seed_kernel", "python"))
            ),
            "allow_unused_bound": bool(self.kwargs.get("allow_unused_bound", False)),
            "fused_bound_rwalk": bool(self.kwargs.get("fused_bound_rwalk", False)),
            "bound_rebuild_on_failure": bool(
                self.kwargs.get("bound_rebuild_on_failure", False)
            ),
            "bound_failure_rebuild_threshold": int(
                self.kwargs.get("bound_failure_rebuild_threshold", 1)
            ),
            "jax_vectorized": bool(self.kwargs.get("jax_vectorized", False)),
            "jax_block_size": int(self.kwargs.get("jax_block_size", 1)),
            "rwalk_adaptive_step_scale": bool(
                self.kwargs.get("rwalk_adaptive_step_scale", False)
            ),
            "rwalk_target_accept": float(
                self.kwargs.get("rwalk_target_accept", 0.25)
            ),
        }

    def _validate_checkpoint_config(self, checkpoint_config: dict) -> None:
        current = self._checkpoint_config()
        if "kernel" not in checkpoint_config:
            checkpoint_config = {**checkpoint_config, "kernel": "python"}
        if checkpoint_config.get("kernel") not in {"python", "jax"}:
            raise ValueError(
                f"checkpoint kernel={checkpoint_config.get('kernel')!r} is invalid"
            )
        for name in ("ndim", "nlive", "sample", "kernel", "vectorized"):
            if checkpoint_config.get(name) != current[name]:
                raise ValueError(
                    f"checkpoint {name}={checkpoint_config.get(name)!r} is not "
                    f"compatible with sampler {name}={current[name]!r}"
                )
        for name in (
            "max_attempts",
            "batch_size",
            "walks",
            "step_scale",
            "min_accepts",
            "replacement_chains",
            "rwalk_proposal",
            "replacement_chain_schedule",
            "bound",
            "bound_enlargement",
            "bound_update_interval",
            "bound_jitter",
            "bound_max_draws",
            "multi_bound_max_ellipsoids",
            "multi_bound_min_points",
            "multi_bound_split_threshold",
            "multi_bound_enlargement",
            "multi_bound_overlap_correction",
            "rwalk_seed",
            "rwalk_seed_fallback",
            "bound_seed_kernel",
            "allow_unused_bound",
            "fused_bound_rwalk",
            "bound_rebuild_on_failure",
            "bound_failure_rebuild_threshold",
            "jax_vectorized",
            "jax_block_size",
            "rwalk_adaptive_step_scale",
            "rwalk_target_accept",
        ):
            default_values = {
                "min_accepts": 1,
                "replacement_chains": 1,
                "rwalk_proposal": "isotropic",
                "bound": "none",
                "bound_enlargement": 1.25,
                "bound_update_interval": 1,
                "bound_jitter": 1e-6,
                "bound_max_draws": None,
                "multi_bound_max_ellipsoids": 32,
                "multi_bound_min_points": None,
                "multi_bound_split_threshold": 0.9,
                "multi_bound_enlargement": None,
                "multi_bound_overlap_correction": True,
                "rwalk_seed": "live",
                "rwalk_seed_fallback": True,
                "bound_seed_kernel": "python",
                "allow_unused_bound": False,
                "fused_bound_rwalk": False,
                "bound_rebuild_on_failure": False,
                "bound_failure_rebuild_threshold": 1,
                "jax_vectorized": False,
                "jax_block_size": 1,
                "rwalk_adaptive_step_scale": False,
                "rwalk_target_accept": 0.25,
            }
            checkpoint_value = checkpoint_config.get(name, default_values.get(name))
            if checkpoint_value != current[name]:
                raise ValueError(
                    f"checkpoint {name}={checkpoint_value!r} is not "
                    f"compatible with sampler {name}={current[name]!r}"
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
            kernel=self.kernel,
            vectorized=self.vectorized,
            max_attempts=self.max_attempts,
            progress=progress,
            progress_interval=progress_interval,
            callback=callback,
            callback_interval=callback_interval,
            checkpoint_path=output_path,
            checkpoint_interval=checkpoint_interval,
            initial_state=state,
            walks=self.kwargs.get("walks", 25),
            step_scale=self.kwargs.get("step_scale", 0.1),
            batch_size=self.kwargs.get("batch_size", 128),
            min_accepts=self.kwargs.get("min_accepts", 1),
            replacement_chains=self.kwargs.get("replacement_chains", 1),
            replacement_chain_schedule=self.kwargs.get("replacement_chain_schedule"),
            rwalk_proposal=self.kwargs.get("rwalk_proposal", "isotropic"),
            bound=self.kwargs.get("bound", "none"),
            bound_enlargement=self.kwargs.get("bound_enlargement", 1.25),
            bound_update_interval=self.kwargs.get("bound_update_interval", 1),
            bound_jitter=self.kwargs.get("bound_jitter", 1e-6),
            bound_max_draws=self.kwargs.get("bound_max_draws"),
            multi_bound_max_ellipsoids=self.kwargs.get(
                "multi_bound_max_ellipsoids", 32
            ),
            multi_bound_min_points=self.kwargs.get("multi_bound_min_points"),
            multi_bound_split_threshold=self.kwargs.get(
                "multi_bound_split_threshold", 0.9
            ),
            multi_bound_enlargement=self.kwargs.get("multi_bound_enlargement"),
            multi_bound_overlap_correction=self.kwargs.get(
                "multi_bound_overlap_correction", True
            ),
            rwalk_seed=self.kwargs.get("rwalk_seed", "live"),
            rwalk_seed_fallback=self.kwargs.get("rwalk_seed_fallback", True),
            bound_seed_kernel=self.kwargs.get("bound_seed_kernel", "python"),
            allow_unused_bound=self.kwargs.get("allow_unused_bound", False),
            fused_bound_rwalk=self.kwargs.get("fused_bound_rwalk", False),
            bound_rebuild_on_failure=self.kwargs.get("bound_rebuild_on_failure", False),
            bound_failure_rebuild_threshold=self.kwargs.get(
                "bound_failure_rebuild_threshold", 1
            ),
            jax_vectorized=self.kwargs.get("jax_vectorized", False),
            jax_block_size=self.kwargs.get("jax_block_size", 1),
            rwalk_adaptive_step_scale=self.kwargs.get(
                "rwalk_adaptive_step_scale", False
            ),
            rwalk_target_accept=self.kwargs.get("rwalk_target_accept", 0.25),
        )
