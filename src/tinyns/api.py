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
        log likelihood (NaN counts as ``-inf``; ``+inf`` is not valid: the run
        stops with ``success=False`` and the status ``"invalid"``). It may be
        a JAX pytree callable such as ``jax.tree_util.Partial(loglike_fn,
        data)``: its array leaves are passed to the compiled kernels as
        arguments, so datasets of one shape share one compiled program.
    prior_transform:
        JAX-traceable function mapping one unit-cube point to parameter space.
        It is only called on the open cube ``(0, 1)^ndim``, never on a face,
        so transforms such as ``ndtri(u)`` or ``-log(u)`` stay finite. May be
        a pytree callable, like ``loglike``.
    ndim:
        Number of dimensions.
    nlive:
        Number of live points (at least 2).
    num_delete:
        Live points deleted and replaced per step, default
        ``max(1, nlive // 10)``, at most ``nlive // 2``. Its replacement chains
        run in parallel (vmapped) and evaluate every proposal (an out-of-cube
        one at the chain's point); ``num_delete=1`` runs one unbatched
        chain that skips the likelihood of a proposal rejected without it
        (outside the cube, or by a local-walk or hop Hastings test), for
        expensive likelihoods. In a batch of runs that chain is vmapped over
        the runs, and it evaluates those proposals too; ``ncall`` still counts
        only what the run evaluates alone, and ``maxcall`` limits that count.
    walks:
        Steps per replacement chain, default ``max(25, 6 * ndim, ndim**2 // 6)``
        (:func:`tinyns.core.default_walks`): 6 steps per dimension up to 36
        dimensions, ``ndim / 6`` per dimension beyond. Calibrated on correlated
        Gaussians up to 64 dimensions; on a 10-D Rosenbrock logZ is 0.05 to
        0.09 low at ``nlive=1000`` and unbiased from 2000. If in doubt, rerun
        with twice the walks and compare.
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
        evaluations are reached (checked after each step), on a likelihood
        plateau, or at a live likelihood of ``+inf`` or with no finite one
        (status ``"invalid"``, ``success=False``; a lane of a batch stops
        alone). ``progress`` prints one line per chunk of steps: dead points,
        ``logz``, the dlogz remainder, calls, calls per second, the chain
        acceptance and the step scale. For a batch of runs it gives the
        number running, the range of dead points, the largest dlogz remainder
        and the median step scale over the running runs (left out once none
        runs), the calls of all runs, calls per second and the acceptance,
        and the number of stopped runs in each status.

        ``checkpoint`` is a file path. If the file exists, the run resumes
        from it; a checkpoint written with a different config, x64 flag,
        float dtype, key or batch is refused (``ValueError``). The callables
        are not stored: on resume the 16 live points of highest likelihood
        and 16 dead points spread over the run (of each run of a batch) are
        evaluated again, and a checkpoint whose stored likelihoods they do
        not give back (another likelihood, prior or dataset) is refused too,
        so use one checkpoint path per problem. They must match to a relative
        ``sqrt(eps)`` of the float dtype, and infinite values exactly. In
        float32 a change below ``3.5e-4 |logL|`` is not seen (from about
        ``1e5`` data points possibly a new noise realisation of the same
        model), nor one at none of the points checked. A path that cannot
        be written raises ``OSError`` before the run starts. The run writes
        the file atomically at a chunk boundary at most every 10 minutes and
        at the end. The random stream depends on the step count only: on a
        CPU a resumed run is bit-identical to an uninterrupted one on the same
        JAX build (a checkpoint of another build or device warns), and on a
        GPU in float32, where a fixed seed is not bit-reproducible between
        processes, the two agree in distribution.
        ``metadata["wall_time_s"]`` adds up the time of every session. A
        checkpoint stopped by ``maxiter``, ``maxcall`` or ``dlogz`` continues
        under this call's limits. A symbolic link at the path is written
        through.

        Running out of device memory raises ``MemoryError`` with what to
        lower: ``num_delete``, or the number of keys of a batched run.

        ``key`` may be a batch of keys (``jax.random.split(key, B)``): the
        ``B`` runs go through one compiled program (the step is vmapped over
        the runs) and a list of ``B`` results is returned. Each is a draw from
        the same distribution as the run of its key alone but is not
        bit-identical to it (vmap changes the summation order; in float64 the
        two agree to roundoff). The batch computes every run until the last
        one stops (a stopped run is frozen, not skipped), so the device
        evaluates about ``B`` times the likelihood calls of the longest run;
        each result's ``ncall`` counts its own run's calls. With
        ``batched_data=True`` every array leaf of a pytree ``loglike`` or
        ``prior_transform`` (e.g. ``jax.tree_util.Partial(fn, data)`` with
        ``data`` stacked over runs) carries the same leading axis of ``B``:
        run ``i`` sees slice ``i``.
        """
        try:
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
        except Exception as err:
            if "RESOURCE_EXHAUSTED" not in str(err):
                raise
            k = self.config.num_delete
            batch = core._key_batch(key)[1] or 1
            what = (
                f"num_delete={k} likelihoods at once"
                if k > 1
                else "one likelihood (num_delete=1)"
            )
            hints = []
            if k > 1:
                hints.append(f"lower num_delete, e.g. num_delete={max(1, k // 4)}")
            if batch > 1:
                what += f" in each of {batch} runs"
                hints.append(f"run fewer than {batch} keys per call")
            hint = " or ".join(hints) or (
                "num_delete cannot go lower: reduce the memory of one likelihood call"
            )
            raise MemoryError(
                f"out of device memory: a step evaluates {what}; {hint}"
            ) from err
