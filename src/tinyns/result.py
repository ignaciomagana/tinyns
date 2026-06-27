"""Result containers returned by :class:`tinyns.NestedSampler`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tinyns.types import ArrayLike


@dataclass(frozen=True)
class NestedSamplerResult:
    """Container for nested-sampling outputs.

    The real sampler is not implemented yet, but defining the result shape now
    gives downstream code a stable object to type against and lets tests lock in
    the intended user-facing contract.
    """

    samples: ArrayLike
    """Dead/live point samples in parameter space."""

    logl: ArrayLike
    """Log-likelihood values associated with ``samples``."""

    logwt: ArrayLike
    """Log posterior weights associated with ``samples``."""

    logz: float
    """Estimated log evidence."""

    logzerr: float
    """Estimated uncertainty on ``logz``."""

    niter: int
    """Number of sampler iterations performed."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional implementation-specific metadata."""

    def asdict(self) -> dict[str, Any]:
        """Return a shallow dictionary representation of the result."""

        return {
            "samples": self.samples,
            "logl": self.logl,
            "logwt": self.logwt,
            "logz": self.logz,
            "logzerr": self.logzerr,
            "niter": self.niter,
            "metadata": dict(self.metadata),
        }

    @property
    def nsamples(self) -> int:
        """Return the number of sample rows in ``samples``."""

        return int(np.asarray(self.samples).shape[0])
