"""Result containers returned by :class:`tinyns.NestedSampler`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tinyns.math import (
    effective_sample_size_from_log_weights,
    normalize_log_weights,
    systematic_resample,
)
from tinyns.types import ArrayLike


@dataclass
class NestedSamplingResult:
    """Container for completed nested-sampling outputs."""

    samples_u: ArrayLike
    """Posterior samples in unit-cube coordinates."""

    samples: ArrayLike
    """Posterior samples in parameter-space coordinates."""

    logl: ArrayLike
    """Log-likelihood values associated with ``samples``."""

    logwt: ArrayLike
    """Unnormalized log posterior weights associated with ``samples``."""

    logz: float
    """Estimated log evidence."""

    logzerr: float
    """Estimated uncertainty on ``logz``."""

    ncall: int
    """Number of likelihood calls performed."""

    nlive: int
    """Number of live points used by the sampler."""

    ndim: int
    """Number of sampled dimensions."""

    success: bool = True
    """Whether the sampler completed successfully."""

    message: str = ""
    """Optional human-readable sampler status message."""

    metadata: dict[str, Any] | None = None
    """Additional implementation-specific metadata."""

    def log_weights(self):
        """Return posterior weights normalized in log space."""

        return normalize_log_weights(self.logwt)

    def weights(self):
        """Return normalized linear posterior weights."""

        return jnp.exp(self.log_weights())

    def posterior_ess(self) -> float:
        """Return the effective sample size of the posterior weights."""

        return float(effective_sample_size_from_log_weights(self.logwt))

    def resample_equal(self, key, n: int | None = None):
        """Return equally weighted posterior samples using systematic resampling."""

        if n is None:
            n = max(1, int(self.posterior_ess()))
        if n < 1:
            raise ValueError("n must be at least 1")

        indices = systematic_resample(key, self.logwt, n)
        return jnp.asarray(self.samples)[indices]

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the result."""

        return "\n".join(
            [
                f"logz: {self.logz}",
                f"logzerr: {self.logzerr}",
                f"ncall: {self.ncall}",
                f"nlive: {self.nlive}",
                f"ndim: {self.ndim}",
                f"posterior ESS: {self.posterior_ess()}",
                f"success: {self.success}",
                f"message: {self.message}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python dictionary representation of the result."""

        return {
            "samples_u": self.samples_u,
            "samples": self.samples,
            "logl": self.logl,
            "logwt": self.logwt,
            "logz": self.logz,
            "logzerr": self.logzerr,
            "ncall": self.ncall,
            "nlive": self.nlive,
            "ndim": self.ndim,
            "success": self.success,
            "message": self.message,
            "metadata": None if self.metadata is None else dict(self.metadata),
        }
