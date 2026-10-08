"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

from tinyns import core
from tinyns.core import Config
from tinyns.result import NestedSamplingResult


class NestedSampler:
    """Nested sampler with ``num_delete`` replacements per step, all in JAX.

    Each step deletes the ``num_delete`` lowest live points and replaces them
    with the ends of that many parallel Metropolis chains of ``walks`` steps:
    a random walk in the covariance of the current point's cluster, with an
    inter-mode hop every 10th step (:mod:`tinyns.modes`).

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
        Steps per replacement chain, default ``max(25, 6 * ndim, ndim**2 // 6)``
        (:func:`tinyns.core.default_walks`): 6 steps per dimension up to 36
        dimensions, ``ndim / 6`` per dimension beyond. Calibrated on correlated
        Gaussians up to 64 dimensions; strongly curved targets need more (a
        10-D Rosenbrock valley needs 12 to 25 ``* ndim`` for an honest
        ``logzerr``). If in doubt, rerun with twice the walks and compare.
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
        checkpoint=None,
        batched_data: bool = False,
    ) -> NestedSamplingResult | list[NestedSamplingResult]:
        """Run nested sampling from ``key`` (a PRNG key or an int seed).

        Stops once the live points hold less than ``dlogz`` of the evidence,
        before more than ``maxiter`` dead points, once ``maxcall`` likelihood
        evaluations are reached (checked after each step), or on a likelihood
        plateau. ``progress`` prints one line per chunk of steps: dead points,
        ``logz``, the dlogz remainder, calls, calls per second, the chain
        acceptance and the step scale.

        ``checkpoint`` is a file path. If the file exists, the run resumes
        from it; a checkpoint written with a different config, x64 flag,
        float dtype, key or batch is refused (``ValueError``). The run writes
        it atomically at a chunk boundary at most every 10 minutes and at the
        end. A resumed run is bit-identical to an uninterrupted one, and
        ``metadata["wall_time_s"]`` adds up the time of every session. A
        checkpoint stopped by ``maxiter``, ``maxcall`` or ``dlogz`` continues
        under this call's limits. The callables are not stored or checked.

        ``key`` may be a batch of keys (``jax.random.split(key, B)``): the
        ``B`` runs go through one compiled program (the step is vmapped over
        the runs) and a list of ``B`` results is returned. Each is a draw from
        the same distribution as the run of its key alone but is not
        bit-identical to it (vmap changes the summation order; in float64 the
        two agree to roundoff). With ``batched_data=True`` every array leaf of
        a pytree ``loglike`` or ``prior_transform`` (e.g.
        ``jax.tree_util.Partial(fn, data)`` with ``data`` stacked over runs)
        carries the same leading axis of ``B``: run ``i`` sees slice ``i``.
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
            checkpoint=checkpoint,
            batched_data=batched_data,
        )
