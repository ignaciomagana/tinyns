"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

from typing import Any

from tinyns.result import NestedSamplingResult
from tinyns.run import run_static_nested
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
        Sampling strategy. ``"prior"``, ``"rwalk"``, and ``"slice"`` are currently supported.
    max_attempts:
        Cap on rejection attempts per constrained prior draw.
    **kwargs:
        Additional sampler options. ``walks`` and ``step_scale`` are used by
        ``sample="rwalk"``; ``slices``, ``slice_steps``, and ``step_scale`` are
        used by ``sample="slice"``.
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
        if sample not in {"prior", "rwalk", "slice"}:
            raise ValueError("sample must be one of {'prior', 'rwalk', 'slice'}")
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
            walks=self.kwargs.get("walks", 25),
            step_scale=self.kwargs.get("step_scale", 0.1),
            batch_size=self.kwargs.get("batch_size", 128),
            slices=self.kwargs.get("slices", 5),
            slice_steps=self.kwargs.get("slice_steps", 10),
        )
