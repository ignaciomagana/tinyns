"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

from tinyns import core
from tinyns.core import Config
from tinyns.result import NestedSamplingResult


class NestedSampler:
    """Nested sampler with ``num_delete`` replacements per step, all in JAX.

    Parameters
    ----------
    loglike:
        JAX-traceable function of one point in parameter space returning its
        log likelihood (NaN counts as ``-inf``). It may be a JAX pytree
        callable such as ``jax.tree_util.Partial(loglike_fn, data)``: its array
        leaves are passed to the compiled kernels as arguments, so datasets of
        one shape share one compiled program.
    prior_transform:
        JAX-traceable function mapping one unit-cube point to parameter space.
        May be a pytree callable, like ``loglike``.
    ndim:
        Number of dimensions.
    nlive:
        Number of live points (at least 2).
    num_delete:
        Live points deleted and replaced per step, default
        ``max(1, nlive // 10)``, at most ``nlive // 2``. Its replacement chains
        run in parallel (vmapped); ``num_delete=1`` runs one unbatched chain
        that skips the likelihood of out-of-cube proposals, for expensive
        likelihoods.
    walks:
        Steps per replacement chain, default ``max(25, 6 * ndim)``.
    """

    def __init__(
        self,
        loglike,
        prior_transform,
        ndim: int,
        nlive: int = 1000,
        *,
        num_delete: int | None = None,
        walks: int | None = None,
    ):
        if not callable(loglike):
            raise TypeError("loglike must be callable")
        if not callable(prior_transform):
            raise TypeError("prior_transform must be callable")
        self.loglike = loglike
        self.prior_transform = prior_transform
        self.config = Config(ndim, nlive, num_delete, walks)

    def run(
        self,
        key,
        *,
        dlogz: float = 0.1,
        maxiter: int | None = None,
        maxcall: int | None = None,
        progress: bool = False,
    ) -> NestedSamplingResult:
        """Run nested sampling from ``key`` (a PRNG key or an int seed).

        Stops once the live points hold less than ``dlogz`` of the evidence,
        before more than ``maxiter`` dead points, once ``maxcall`` likelihood
        evaluations are reached (checked after each step), or on a likelihood
        plateau. ``progress`` prints one line per chunk of steps.
        """
        return core.run(
            key,
            self.loglike,
            self.prior_transform,
            self.config,
            dlogz=dlogz,
            maxiter=maxiter,
            maxcall=maxcall,
            progress=progress,
        )
