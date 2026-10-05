"""Private active run-state and checkpoint helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

_CHECKPOINT_NPZ_FORMAT_VERSION = "tinyns-checkpoint-npz-v1"
_CHECKPOINT_REQUIRED_KEYS = {
    "format_version",
    "key",
    "live_u",
    "live_theta",
    "live_logl",
    "dead_u",
    "dead_theta",
    "dead_logl",
    "dead_logwt",
    "logz_dead",
    "logx_final",
    "ncall",
    "replacement_ncall",
    "insertion_indices",
    "replacement_failures",
    "iteration",
    "success",
    "message",
    "stopped_by_callback",
    "config_json",
}


@dataclass
class NestedRunState:
    key: Any
    live_u: Any
    live_theta: Any
    live_logl: Any
    dead_u: list[Any]
    dead_theta: list[Any]
    dead_logl: list[float]
    dead_logwt: list[float]
    logz_dead: float
    logx_final: float
    ncall: int
    replacement_ncall: list[int]
    insertion_indices: list[int]
    replacement_failures: int
    iteration: int
    success: bool
    message: str
    stopped_by_callback: bool = False
    scale: float | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    # Cluster tracker state (``cluster_swap``): {"log": dict, "arrays": dict
    # with "labels" and "u", or None}.
    clusters: dict[str, Any] | None = None


def _npz_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    raise ValueError("expected scalar value in checkpoint .npz file")


def _stack_points(points: list[Any], ndim: int):
    if not points:
        return np.empty((0, ndim), dtype=float)
    return np.asarray(jnp.stack(points))


def save_checkpoint_npz(path, state: NestedRunState, config: dict) -> None:
    """Atomically save an active static nested-sampling run checkpoint."""

    path = os.fspath(path)
    tmp_path = f"{path}.tmp"
    ndim = int(config["ndim"])
    with open(tmp_path, "wb") as file:
        np.savez_compressed(
            file,
            format_version=np.asarray(_CHECKPOINT_NPZ_FORMAT_VERSION),
            key=np.asarray(state.key),
            live_u=np.asarray(state.live_u),
            live_theta=np.asarray(state.live_theta),
            live_logl=np.asarray(state.live_logl),
            dead_u=_stack_points(state.dead_u, ndim),
            dead_theta=_stack_points(state.dead_theta, ndim),
            dead_logl=np.asarray(state.dead_logl, dtype=float),
            dead_logwt=np.asarray(state.dead_logwt, dtype=float),
            logz_dead=np.asarray(float(state.logz_dead)),
            logx_final=np.asarray(float(state.logx_final)),
            ncall=np.asarray(int(state.ncall)),
            replacement_ncall=np.asarray(state.replacement_ncall, dtype=int),
            insertion_indices=np.asarray(state.insertion_indices, dtype=int),
            replacement_failures=np.asarray(int(state.replacement_failures)),
            iteration=np.asarray(int(state.iteration)),
            success=np.asarray(bool(state.success)),
            message=np.asarray(str(state.message)),
            stopped_by_callback=np.asarray(bool(state.stopped_by_callback)),
            scale=np.asarray(
                float(state.scale) if state.scale is not None else float("nan")
            ),
            telemetry_json=np.asarray(json.dumps(state.telemetry, sort_keys=True)),
            config_json=np.asarray(json.dumps(config, sort_keys=True)),
            **_cluster_fields(state.clusters),
        )
    os.replace(tmp_path, path)


def _cluster_fields(clusters) -> dict:
    """Optional checkpoint fields for the cluster tracker (no version bump)."""
    if clusters is None:
        return {}
    fields = {"clusters_json": np.asarray(json.dumps(clusters["log"]))}
    if clusters["arrays"] is not None:
        fields["cluster_labels"] = np.asarray(clusters["arrays"]["labels"])
        fields["cluster_u"] = np.asarray(clusters["arrays"]["u"])
    return fields


def load_checkpoint_npz(path) -> tuple[NestedRunState, dict]:
    """Load an active static nested-sampling run checkpoint."""

    with np.load(path) as data:
        missing = sorted(_CHECKPOINT_REQUIRED_KEYS - set(data.files))
        if missing:
            raise ValueError(
                "missing required checkpoint .npz keys: " + ", ".join(missing)
            )
        format_version = str(_npz_scalar(data["format_version"]))
        if format_version != _CHECKPOINT_NPZ_FORMAT_VERSION:
            raise ValueError(
                "unknown checkpoint .npz format_version: "
                f"{format_version!r}; expected {_CHECKPOINT_NPZ_FORMAT_VERSION!r}"
            )
        config = json.loads(str(_npz_scalar(data["config_json"])))
        scale = None
        if "scale" in data.files:
            raw_scale = float(_npz_scalar(data["scale"]))
            if np.isfinite(raw_scale):
                scale = raw_scale
        telemetry = {}
        if "telemetry_json" in data.files:
            telemetry = json.loads(str(_npz_scalar(data["telemetry_json"])))
        clusters = None
        if "clusters_json" in data.files:
            arrays = None
            if "cluster_labels" in data.files:
                arrays = {
                    "labels": np.asarray(data["cluster_labels"]),
                    "u": np.asarray(data["cluster_u"]),
                }
            clusters = {
                "log": json.loads(str(_npz_scalar(data["clusters_json"]))),
                "arrays": arrays,
            }
        dead_u = [jnp.asarray(point) for point in np.asarray(data["dead_u"])]
        dead_theta = [jnp.asarray(point) for point in np.asarray(data["dead_theta"])]
        state = NestedRunState(
            key=jnp.asarray(data["key"]),
            live_u=jnp.asarray(data["live_u"]),
            live_theta=jnp.asarray(data["live_theta"]),
            live_logl=jnp.asarray(data["live_logl"]),
            dead_u=dead_u,
            dead_theta=dead_theta,
            dead_logl=[float(x) for x in np.asarray(data["dead_logl"])],
            dead_logwt=[float(x) for x in np.asarray(data["dead_logwt"])],
            logz_dead=float(_npz_scalar(data["logz_dead"])),
            logx_final=float(_npz_scalar(data["logx_final"])),
            ncall=int(_npz_scalar(data["ncall"])),
            replacement_ncall=[int(x) for x in np.asarray(data["replacement_ncall"])],
            insertion_indices=[int(x) for x in np.asarray(data["insertion_indices"])],
            replacement_failures=int(_npz_scalar(data["replacement_failures"])),
            iteration=int(_npz_scalar(data["iteration"])),
            success=bool(_npz_scalar(data["success"])),
            message=str(_npz_scalar(data["message"])),
            stopped_by_callback=bool(_npz_scalar(data["stopped_by_callback"])),
            scale=scale,
            telemetry=telemetry,
            clusters=clusters,
        )
        return state, config
