"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

from typing import Any

from tinyns.result import NestedSamplingResult
from tinyns.run import run_static_nested
from tinyns.state import load_checkpoint_npz
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike


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
        Sampling strategy. ``"prior"``, ``"rwalk"``, ``"slice"``, and ``"rslice"`` are
        currently supported.
    max_attempts:
        Cap on rejection attempts per constrained prior draw.
    **kwargs:
        Additional sampler options. ``walks`` and ``step_scale`` are used by
        ``sample="rwalk"``; ``slices``, ``slice_steps``, and ``step_scale`` are
        used by ``sample="slice"`` and ``sample="rslice"``.
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
        if sample not in {"prior", "rwalk", "slice", "rslice"}:
            raise ValueError(
                "sample must be one of {'prior', 'rwalk', 'slice', 'rslice'}"
            )
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
        self.max_attempts = max_attempts
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
            slices=self.kwargs.get("slices", 5),
            slice_steps=self.kwargs.get("slice_steps", 10),
        )

    def _checkpoint_config(self) -> dict[str, object]:
        return {
            "ndim": int(self.ndim),
            "nlive": int(self.nlive),
            "sample": str(self.sample),
            "vectorized": bool(self.vectorized),
            "max_attempts": int(self.max_attempts),
            "batch_size": int(self.kwargs.get("batch_size", 128)),
            "walks": int(self.kwargs.get("walks", 25)),
            "step_scale": float(self.kwargs.get("step_scale", 0.1)),
            "slices": int(self.kwargs.get("slices", 5)),
            "slice_steps": int(self.kwargs.get("slice_steps", 10)),
        }

    def _validate_checkpoint_config(self, checkpoint_config: dict) -> None:
        current = self._checkpoint_config()
        for name in ("ndim", "nlive", "sample", "vectorized"):
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
            "slices",
            "slice_steps",
        ):
            if checkpoint_config.get(name) != current[name]:
                raise ValueError(
                    f"checkpoint {name}={checkpoint_config.get(name)!r} is not "
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
            slices=self.kwargs.get("slices", 5),
            slice_steps=self.kwargs.get("slice_steps", 10),
        )
