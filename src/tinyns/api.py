"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

import difflib
from typing import Any

from tinyns import checkpoint, loop
from tinyns.core import Config
from tinyns.result import NestedSamplingResult
from tinyns.types import LogLikelihood, PriorTransform, PRNGKeyLike

_KNOWN_KWARGS = frozenset(("walks", "replacement_chains", "block_size", "cluster_swap"))


class NestedSampler:
    """Static nested sampler with a live-cov random walk in jitted blocks.

    Parameters
    ----------
    loglike:
        JAX-traceable function of one point in parameter space that returns
        its log likelihood (it is ``jax.vmap``-ped where needed). It may be a
        JAX pytree callable such as ``jax.tree_util.Partial(loglike_fn,
        data)``: its array leaves are passed to the compiled kernels as
        arguments rather than embedded as constants, which helps when
        ``data`` is large.
    prior_transform:
        JAX-traceable function mapping one unit-cube point to parameter space.
        May be a pytree callable, like ``loglike``.
    ndim:
        Number of model dimensions. Must be positive.
    nlive:
        Number of live points to use. Must be positive.
    walks:
        rwalk steps per replacement chain. ``None`` resolves to
        ``max(25, 6 * ndim)`` (12 for ``ndim=1``).
    replacement_chains:
        Chains run in parallel per replacement (one is kept).
    block_size:
        Nested-sampling iterations per jitted block.
    cluster_swap:
        Let the chains swap between tracked clusters of the live points, which
        keeps the weights of separated modes from drifting. ``None`` turns it
        on with one replacement chain, the only configuration that supports
        it. Any other keyword raises ``TypeError``.
    """

    def __init__(
        self,
        loglike: LogLikelihood,
        prior_transform: PriorTransform,
        ndim: int,
        nlive: int = 500,
        *,
        walks: int | None = None,
        replacement_chains: int = 1,
        block_size: int = 32,
        cluster_swap: bool | None = None,
        **kwargs: Any,
    ):
        for name in sorted(kwargs):
            close = difflib.get_close_matches(name, _KNOWN_KWARGS, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise TypeError(
                f"NestedSampler got an unexpected keyword argument {name!r}{hint}"
            )
        if ndim <= 0:
            raise ValueError("ndim must be a positive integer")
        if nlive <= 0:
            raise ValueError("nlive must be a positive integer")
        if not callable(loglike):
            raise TypeError("loglike must be callable")
        if not callable(prior_transform):
            raise TypeError("prior_transform must be callable")

        self.loglike = loglike
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.nlive = nlive
        # The resolved options; checkpointed and checked on resume.
        self._config = Config(
            ndim,
            nlive,
            walks=walks,
            replacement_chains=replacement_chains,
            block_size=block_size,
            cluster_swap=cluster_swap,
        )

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

        return loop.run(
            self._config,
            self.loglike,
            self.prior_transform,
            key,
            dlogz=dlogz,
            maxiter=maxiter,
            progress=progress,
            progress_interval=progress_interval,
            callback=callback,
            callback_interval=callback_interval,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
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
        """Resume nested sampling from a ``tinyns-ckpt-2`` checkpoint file.

        The checkpoint's ``ndim``, ``nlive``, ``walks``, ``replacement_chains``,
        ``block_size`` and ``cluster_swap`` must match this sampler's, or a
        ``ValueError`` names the key that differs. The callables are not saved
        and cannot be checked: pass the ``loglike`` and ``prior_transform`` the
        run started with. A checkpoint saved after a replacement failure cannot
        be resumed, and files written before tinyns v0.3 are not read.
        """

        ckpt = checkpoint.load(checkpoint_path)
        checkpoint.check_config(ckpt.config, self._config)
        output_path = (
            checkpoint_path if checkpoint_path_out is None else checkpoint_path_out
        )
        return loop.run(
            self._config,
            self.loglike,
            self.prior_transform,
            None,
            dlogz=dlogz,
            maxiter=maxiter,
            progress=progress,
            progress_interval=progress_interval,
            callback=callback,
            callback_interval=callback_interval,
            checkpoint_path=output_path,
            checkpoint_interval=checkpoint_interval,
            resume=ckpt,
        )
