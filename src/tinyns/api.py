"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

from dataclasses import dataclass

from tinyns.result import NestedSamplerResult
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike


@dataclass(frozen=True)
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
        Number of model dimensions.
    nlive:
        Number of live points to use when the sampler is implemented.

    Notes
    -----
    The actual nested-sampling algorithm is intentionally not implemented in
    this scaffold. ``run`` currently raises :class:`NotImplementedError`.
    """

    loglike: LogLikelihood
    prior_transform: PriorTransform
    ndim: int
    nlive: int = 500

    def __post_init__(self) -> None:
        """Validate basic sampler configuration."""

        if self.ndim <= 0:
            raise ValueError("ndim must be a positive integer")
        if self.nlive <= 0:
            raise ValueError("nlive must be a positive integer")
        if not callable(self.loglike):
            raise TypeError("loglike must be callable")
        if not callable(self.prior_transform):
            raise TypeError("prior_transform must be callable")

    def run(self, key: PRNGKeyLike, *, dlogz: float = 0.1) -> NestedSamplerResult:
        """Run nested sampling and return a :class:`NestedSamplerResult`.

        Parameters
        ----------
        key:
            JAX PRNG key used to drive stochastic sampler operations.
        dlogz:
            Target remaining-evidence stopping criterion.
        """

        _ = key
        if dlogz <= 0:
            raise ValueError("dlogz must be positive")
        raise NotImplementedError("NestedSampler.run is not implemented yet")
