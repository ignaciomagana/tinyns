"""Result containers returned by :class:`tinyns.NestedSampler`."""

from __future__ import annotations

import json
import math
import numbers
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

_RESULT_NPZ_FORMAT_VERSION = "tinyns-result-npz-v1"
_RESULT_NPZ_REQUIRED_KEYS = {
    "samples_u",
    "samples",
    "logl",
    "logwt",
    "logz",
    "logzerr",
    "ncall",
    "nlive",
    "ndim",
    "success",
    "message",
    "metadata_json",
    "format_version",
}


def _metadata_to_jsonable(metadata):
    """Return metadata converted to JSON-compatible Python values."""

    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return {
            str(key): _metadata_to_jsonable(value)
            for key, value in metadata.items()
        }
    if isinstance(metadata, list):
        return [_metadata_to_jsonable(value) for value in metadata]
    if isinstance(metadata, (str, bool, int, float)) or metadata is None:
        return metadata
    if isinstance(metadata, np.generic):
        return metadata.item()

    try:
        array = np.asarray(metadata)
    except (TypeError, ValueError):
        return str(metadata)

    if array.dtype == object:
        return str(metadata)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (str, bool, int, float)) or scalar is None:
            return scalar
        return str(scalar)
    return array.tolist()


def _metadata_from_jsonable(metadata):
    """Return metadata loaded from its JSON-compatible representation."""

    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return {
            str(key): _metadata_from_jsonable(value)
            for key, value in metadata.items()
        }
    if isinstance(metadata, list):
        return [_metadata_from_jsonable(value) for value in metadata]
    if isinstance(metadata, (str, bool, int, float)) or metadata is None:
        return metadata
    return str(metadata)


def _npz_scalar(value):
    """Return a Python scalar from a NumPy value loaded from ``np.load``."""

    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    raise ValueError("expected scalar value in result .npz file")


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

    def max_weight_fraction(self) -> float:
        """Return the largest normalized posterior weight fraction."""

        weights = self.weights()
        if weights.size == 0:
            return 0.0
        return float(jnp.max(weights))

    def posterior_weight_entropy(self) -> float:
        """Return the Shannon entropy of normalized posterior weights."""

        weights = self.weights()
        if weights.size == 0:
            return 0.0
        positive = weights > 0.0
        return float(-jnp.sum(weights[positive] * jnp.log(weights[positive])))

    def posterior_weight_entropy_fraction(self) -> float:
        """Return posterior weight entropy as a fraction of equal-weight entropy."""

        n = int(jnp.asarray(self.logwt).size)
        if n <= 1:
            return 0.0
        fraction = self.posterior_weight_entropy() / float(jnp.log(n))
        return float(jnp.clip(fraction, 0.0, 1.0))

    def live_weight_fraction(self) -> float:
        """Return posterior weight fraction in final live points.

        The sampler records final live points as the last ``metadata["nlive_final"]``
        weighted samples. If this metadata is missing or invalid, return ``0.0``
        so diagnostics remain simple and conservative.
        """

        nlive_final = self._valid_nlive_final()
        if nlive_final is None:
            return 0.0
        return float(jnp.sum(self.weights()[-nlive_final:]))

    def dead_weight_fraction(self) -> float:
        """Return posterior weight fraction in dead points."""

        if self._valid_nlive_final() is None:
            return 0.0
        return float(jnp.clip(1.0 - self.live_weight_fraction(), 0.0, 1.0))

    def _valid_nlive_final(self) -> int | None:
        """Return valid final-live count metadata, or ``None`` if invalid."""

        metadata = {} if self.metadata is None else self.metadata
        nlive_final = metadata.get("nlive_final")
        nweights = int(jnp.asarray(self.logwt).size)
        if (
            nlive_final is None
            or isinstance(nlive_final, bool)
            or not isinstance(nlive_final, numbers.Integral)
            or nlive_final <= 0
            or nlive_final > nweights
        ):
            return None
        return int(nlive_final)

    def insertion_indices(self):
        """Return recorded live-point insertion indices, if available."""

        metadata = {} if self.metadata is None else self.metadata
        return jnp.asarray(metadata.get("insertion_indices", []), dtype=int)

    def information(self) -> float:
        """Return the nested-sampling information from posterior weights."""

        logl = jnp.asarray(self.logl)
        logwt = jnp.asarray(self.logwt)
        weights = jnp.exp(logwt - self.logz)
        if bool(jnp.any(~jnp.isfinite(weights))):
            return math.nan
        contributing = (weights > 0.0) & jnp.isfinite(weights) & jnp.isfinite(logl)
        if bool(jnp.any((weights > 0.0) & ~jnp.isfinite(logl))):
            return math.nan
        if not bool(jnp.any(contributing)):
            return 0.0
        information = float(
            jnp.sum(jnp.where(contributing, weights * (logl - self.logz), 0.0))
        )
        if not math.isfinite(information):
            return math.nan
        if information < 0.0:
            return 0.0
        return information

    def diagnostics(self) -> dict[str, object]:
        """Return lightweight run diagnostics as a plain dictionary."""

        metadata = {} if self.metadata is None else self.metadata
        posterior_ess = self.posterior_ess()
        nposterior = int(jnp.asarray(self.logwt).size)
        max_weight_fraction = self.max_weight_fraction()
        posterior_weight_entropy = self.posterior_weight_entropy()
        entropy_fraction = self.posterior_weight_entropy_fraction()
        live_weight_fraction = self.live_weight_fraction()
        dead_weight_fraction = self.dead_weight_fraction()
        warnings: list[str] = []

        if not self.success:
            warnings.append(self.message)
        if posterior_ess < 100.0:
            warnings.append("low posterior ESS")
        if max_weight_fraction > 0.1:
            warnings.append("posterior dominated by a small number of weighted samples")
        if entropy_fraction < 0.5:
            warnings.append("low posterior weight entropy")
        if live_weight_fraction > 0.5:
            warnings.append(
                "final live points carry most posterior weight; consider tighter "
                "dlogz or more live points"
            )
        requested_dlogz = metadata.get("dlogz")
        try:
            requested_dlogz_float = float(requested_dlogz)
        except (TypeError, ValueError):
            requested_dlogz_float = None
        if (
            live_weight_fraction > 0.25
            and requested_dlogz_float is not None
            and requested_dlogz_float >= 0.1
        ):
            warnings.append(
                "large final-live weight fraction; evidence may be sensitive to "
                "stopping"
            )

        final_delta_logz = metadata.get("final_delta_logz")
        try:
            final_delta_logz_float = float(final_delta_logz)
        except (TypeError, ValueError):
            final_delta_logz_float = None
        if (
            self.success
            and final_delta_logz_float is not None
            and requested_dlogz_float is not None
            and final_delta_logz_float > requested_dlogz_float
        ):
            warnings.append("successful run has final_delta_logz above requested dlogz")

        replacement_failures = metadata.get("replacement_failures")
        if replacement_failures is not None and replacement_failures > 0:
            warnings.append("replacement failures occurred")

        logzerr_status = metadata.get("logzerr_status")
        if logzerr_status is None:
            logzerr_status = "ok" if math.isfinite(float(self.logzerr)) else "unknown"
        if logzerr_status != "ok":
            warnings.append(f"logzerr estimate unavailable: {logzerr_status}")

        replacement_chains = int(metadata.get("replacement_chains", 1) or 1)
        replacement_batch_ncall = metadata.get("replacement_batch_ncall")
        if replacement_batch_ncall is None:
            walks = metadata.get("walks")
            if walks is not None:
                replacement_batch_ncall = int(walks) * replacement_chains
        if replacement_batch_ncall is not None:
            replacement_batch_ncall = int(replacement_batch_ncall)

        mean_replacement_ncall = metadata.get(
            "replacement_mean_ncall", metadata.get("mean_replacement_ncall")
        )
        max_replacement_ncall = metadata.get("max_replacement_ncall")
        mean_replacement_batches = None
        max_replacement_batches = None
        if replacement_batch_ncall is not None and replacement_batch_ncall > 0:
            if mean_replacement_ncall is not None:
                mean_replacement_batches = (
                    float(mean_replacement_ncall) / replacement_batch_ncall
                )
            if max_replacement_ncall is not None:
                max_replacement_batches = (
                    float(max_replacement_ncall) / replacement_batch_ncall
                )

        replacement_acceptance_proxy = metadata.get("replacement_acceptance_proxy")
        if replacement_chains == 1:
            if (
                replacement_acceptance_proxy is not None
                and replacement_acceptance_proxy < 0.01
            ):
                warnings.append("low replacement acceptance")
        elif (
            mean_replacement_batches is not None
            and mean_replacement_batches > 2.0
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
            rank_count = int(insertion_indices.size)
            insertion_rank_mean_z = (mean_normalized_rank - 0.5) / math.sqrt(
                (1.0 / 12.0) / rank_count
            )
            insertion_rank_std_ratio = float(jnp.std(normalized_ranks, ddof=1)) / (
                1.0 / math.sqrt(12.0)
            )
            if mean_normalized_rank < 0.35 or mean_normalized_rank > 0.65:
                warnings.append(
                    "insertion indices look non-uniform; constrained sampler may be "
                    "biased or poorly mixed"
                )

        if nposterior < self.nlive + 10:
            warnings.append("very few dead points")

        has_insertion_rank_stats = (
            insertion_indices.size >= 20 and insertion_index_nslots > 0
        )

        diagnostics: dict[str, object] = {
            "success": self.success,
            "message": self.message,
            "logz": float(self.logz),
            "logzerr": float(self.logzerr),
            "information": metadata.get("information_H", self.information()),
            "logzerr_status": logzerr_status,
            "information_H": metadata.get("information_H"),
            "n_nonfinite_logl": metadata.get("n_nonfinite_logl"),
            "n_nonfinite_logwt": metadata.get("n_nonfinite_logwt"),
            "n_nonfinite_weights": metadata.get("n_nonfinite_weights"),
            "n_dead_finite": metadata.get("n_dead_finite"),
            "n_live_finite": metadata.get("n_live_finite"),
            "posterior_ess": posterior_ess,
            "max_weight_fraction": max_weight_fraction,
            "posterior_weight_entropy": posterior_weight_entropy,
            "posterior_weight_entropy_fraction": entropy_fraction,
            "live_weight_fraction": live_weight_fraction,
            "dead_weight_fraction": dead_weight_fraction,
            "ncall": int(self.ncall),
            "nlive": int(self.nlive),
            "ndim": int(self.ndim),
            "nposterior": nposterior,
            "warnings": warnings,
            "final_delta_logz": final_delta_logz,
            "insertion_rank_mean_z": (
                insertion_rank_mean_z if has_insertion_rank_stats else None
            ),
            "insertion_rank_std_ratio": (
                insertion_rank_std_ratio if has_insertion_rank_stats else None
            ),
            "final_logx": metadata.get("final_logx"),
            "final_logz_dead": metadata.get("final_logz_dead"),
            "final_logl_live_max": metadata.get("final_logl_live_max"),
        }

        if "niter" in metadata:
            diagnostics["niter"] = metadata["niter"]
        if "ndead" in metadata:
            diagnostics["ndead"] = metadata["ndead"]
        diagnostics["replacement_chains"] = replacement_chains
        if replacement_batch_ncall is not None:
            diagnostics["replacement_batch_ncall"] = replacement_batch_ncall
        if mean_replacement_ncall is not None:
            diagnostics["replacement_mean_ncall"] = mean_replacement_ncall
        if mean_replacement_batches is not None:
            diagnostics["replacement_mean_batches"] = mean_replacement_batches
        if max_replacement_batches is not None:
            diagnostics["replacement_max_batches"] = max_replacement_batches
        if replacement_failures is not None:
            diagnostics["replacement_failures"] = replacement_failures
        for name in (
            "accepted_rwalk_moves",
            "total_rwalk_proposals",
            "rwalk_acceptance",
            "mean_rwalk_acceptance",
        ):
            if name in metadata:
                diagnostics[name] = metadata[name]

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

    def save_npz(self, path) -> None:
        """Save the final result to a compressed NumPy ``.npz`` file."""

        metadata_json = json.dumps(_metadata_to_jsonable(self.metadata))
        np.savez_compressed(
            path,
            samples_u=np.asarray(self.samples_u),
            samples=np.asarray(self.samples),
            logl=np.asarray(self.logl),
            logwt=np.asarray(self.logwt),
            logz=np.asarray(float(self.logz)),
            logzerr=np.asarray(float(self.logzerr)),
            ncall=np.asarray(int(self.ncall)),
            nlive=np.asarray(int(self.nlive)),
            ndim=np.asarray(int(self.ndim)),
            success=np.asarray(bool(self.success)),
            message=np.asarray(str(self.message)),
            metadata_json=np.asarray(metadata_json),
            format_version=np.asarray(_RESULT_NPZ_FORMAT_VERSION),
        )

    @classmethod
    def load_npz(cls, path) -> NestedSamplingResult:
        """Load a result previously written by :meth:`save_npz`."""

        with np.load(path) as data:
            keys = set(data.files)
            missing = sorted(_RESULT_NPZ_REQUIRED_KEYS - keys)
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"missing required result .npz keys: {joined}")

            format_version = str(_npz_scalar(data["format_version"]))
            if format_version != _RESULT_NPZ_FORMAT_VERSION:
                raise ValueError(
                    "unknown result .npz format_version: "
                    f"{format_version!r}; expected {_RESULT_NPZ_FORMAT_VERSION!r}"
                )

            metadata_json = str(_npz_scalar(data["metadata_json"]))
            metadata = _metadata_from_jsonable(json.loads(metadata_json))

            return cls(
                samples_u=jnp.asarray(data["samples_u"]),
                samples=jnp.asarray(data["samples"]),
                logl=jnp.asarray(data["logl"]),
                logwt=jnp.asarray(data["logwt"]),
                logz=float(_npz_scalar(data["logz"])),
                logzerr=float(_npz_scalar(data["logzerr"])),
                ncall=int(_npz_scalar(data["ncall"])),
                nlive=int(_npz_scalar(data["nlive"])),
                ndim=int(_npz_scalar(data["ndim"])),
                success=bool(_npz_scalar(data["success"])),
                message=str(_npz_scalar(data["message"])),
                metadata=metadata,
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
