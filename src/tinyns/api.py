"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

from typing import Any

from tinyns.result import NestedSamplingResult
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike


class NestedSampler:
    """Tiny dynesty-style nested sampler facade.

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
        Number of live points to use when the sampler is implemented. Must be
        positive.
    vectorized:
        Whether ``loglike`` and ``prior_transform`` accept batches of points.
    sample:
        Sampling strategy. Only ``"prior"`` is accepted by this API shell.
    max_attempts:
        Future cap on rejection attempts per constrained prior draw.
    **kwargs:
        Additional sampler options reserved for future implementations.

    Notes
    -----
    The actual nested-sampling algorithm is intentionally not implemented in
    this scaffold. ``run`` currently raises :class:`NotImplementedError`.
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
        if sample not in {"prior"}:
            raise ValueError("sample must currently be one of {'prior'}")
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
        """Run nested sampling and return a :class:`NestedSamplingResult`.

        Parameters
        ----------
        key:
            JAX PRNG key used to drive stochastic sampler operations.
        dlogz:
            Target remaining-evidence stopping criterion.
        maxiter:
            Optional maximum number of nested-sampling iterations.
        progress:
            Whether to display progress information when implemented.
        """

        _ = (key, dlogz, maxiter, progress)
        raise NotImplementedError("NestedSampler.run is not implemented yet")
