"""Result containers returned by :class:`tinyns.NestedSampler`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

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

    def insertion_indices(self):
        """Return recorded live-point insertion indices, if available."""

        metadata = {} if self.metadata is None else self.metadata
        return jnp.asarray(metadata.get("insertion_indices", []), dtype=int)

    def information(self) -> float:
        """Return the nested-sampling information from posterior weights."""

        information = float(
            jnp.sum(self.weights() * (jnp.asarray(self.logl) - self.logz))
        )
        if information < 0.0:
            return 0.0
        return information

    def diagnostics(self) -> dict[str, object]:
        """Return lightweight run diagnostics as a plain dictionary."""

        metadata = {} if self.metadata is None else self.metadata
        posterior_ess = self.posterior_ess()
        nposterior = int(jnp.asarray(self.logwt).size)
        warnings: list[str] = []

        if not self.success:
            warnings.append(self.message)
        if posterior_ess < 100.0:
            warnings.append("low posterior ESS")

        replacement_failures = metadata.get("replacement_failures")
        if replacement_failures is not None and replacement_failures > 0:
            warnings.append("replacement failures occurred")

        replacement_acceptance_proxy = metadata.get("replacement_acceptance_proxy")
        if (
            replacement_acceptance_proxy is not None
            and replacement_acceptance_proxy < 0.01
        ):
            warnings.append("low replacement acceptance")

        insertion_indices = self.insertion_indices()
        insertion_index_nslots = metadata.get(
            "insertion_index_nslots",
            metadata.get("insertion_index_nlive", self.nlive - 1),
        )
        if insertion_indices.size >= 20 and insertion_index_nslots > 0:
            normalized_ranks = (insertion_indices + 0.5) / insertion_index_nslots
            mean_normalized_rank = float(jnp.mean(normalized_ranks))
            if mean_normalized_rank < 0.35 or mean_normalized_rank > 0.65:
                warnings.append(
                    "insertion indices look non-uniform; constrained sampler may be "
                    "biased or poorly mixed"
                )

        if nposterior < self.nlive + 10:
            warnings.append("very few dead points")

        diagnostics: dict[str, object] = {
            "success": self.success,
            "message": self.message,
            "logz": float(self.logz),
            "logzerr": float(self.logzerr),
            "information": self.information(),
            "posterior_ess": posterior_ess,
            "ncall": int(self.ncall),
            "nlive": int(self.nlive),
            "ndim": int(self.ndim),
            "nposterior": nposterior,
            "warnings": warnings,
        }

        if "replacement_mean_ncall" in metadata:
            diagnostics["replacement_mean_ncall"] = metadata[
                "replacement_mean_ncall"
            ]
        elif "mean_replacement_ncall" in metadata:
            diagnostics["replacement_mean_ncall"] = metadata[
                "mean_replacement_ncall"
            ]
        if replacement_failures is not None:
            diagnostics["replacement_failures"] = replacement_failures

        return diagnostics

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

        lines = [
            f"logz: {self.logz}",
            f"logzerr: {self.logzerr}",
            f"ncall: {self.ncall}",
            f"nlive: {self.nlive}",
            f"ndim: {self.ndim}",
            f"posterior ESS: {self.posterior_ess()}",
        ]
        metadata = {} if self.metadata is None else self.metadata
        if "mean_replacement_ncall" in metadata:
            lines.append(
                f"replacement mean ncall: {metadata['mean_replacement_ncall']}"
            )
        if "replacement_failures" in metadata:
            lines.append(f"replacement failures: {metadata['replacement_failures']}")
        lines.extend(
            [
                f"success: {self.success}",
                f"message: {self.message}",
            ]
        )
        return "\n".join(lines)

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

    def to_numpy(self) -> dict[str, object]:
        """Return a plain dictionary with array fields converted to NumPy arrays."""

        return {
            "samples_u": np.asarray(self.samples_u),
            "samples": np.asarray(self.samples),
            "logl": np.asarray(self.logl),
            "logwt": np.asarray(self.logwt),
            "logz": float(self.logz),
            "logzerr": float(self.logzerr),
            "ncall": int(self.ncall),
            "nlive": int(self.nlive),
            "ndim": int(self.ndim),
            "success": bool(self.success),
            "message": str(self.message),
            "metadata": None if self.metadata is None else dict(self.metadata),
        }

    def to_dynesty_dict(self) -> dict[str, object]:
        """Return a lightweight dynesty-compatibility dictionary.

        This is not a full dynesty ``Results`` object, only a lightweight
        compatibility dict using dynesty-like keys where tinyns has matching
        fields.
        """

        result = {
            "samples": np.asarray(self.samples),
            "samples_u": np.asarray(self.samples_u),
            "logl": np.asarray(self.logl),
            "logwt": np.asarray(self.logwt),
            "logz": float(self.logz),
            "logzerr": float(self.logzerr),
            "ncall": int(self.ncall),
            "nlive": int(self.nlive),
        }
        metadata = {} if self.metadata is None else self.metadata
        if "replacement_acceptance_proxy" in metadata:
            result["eff"] = metadata["replacement_acceptance_proxy"]
        return result
