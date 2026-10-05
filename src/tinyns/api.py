"""Public sampler API for :mod:`tinyns`."""

from __future__ import annotations

import difflib
from typing import Any

from tinyns.result import NestedSamplingResult
from tinyns.run import _resolve_options, run_static_nested
from tinyns.state import load_checkpoint_npz
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
        # Resolved options, forwarded to run_static_nested and checkpointed.
        self._options = _resolve_options(
            ndim,
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

        return run_static_nested(
            key,
            self.loglike,
            self.prior_transform,
            self.ndim,
            self.nlive,
            dlogz=dlogz,
            maxiter=maxiter,
            progress=progress,
            progress_interval=progress_interval,
            callback=callback,
            callback_interval=callback_interval,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            **self._options,
        )

    def _checkpoint_config(self) -> dict[str, object]:
        return {"ndim": int(self.ndim), "nlive": int(self.nlive), **self._options}

    def _validate_checkpoint_config(self, checkpoint_config: dict) -> None:
        for name, current_value in self._checkpoint_config().items():
            checkpoint_value = checkpoint_config.get(name)
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
        if not state.success and state.replacement_failures:
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
            progress=progress,
            progress_interval=progress_interval,
            callback=callback,
            callback_interval=callback_interval,
            checkpoint_path=output_path,
            checkpoint_interval=checkpoint_interval,
            initial_state=state,
            **self._options,
        )
