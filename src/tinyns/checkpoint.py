"""Checkpoint files of :mod:`tinyns`, format ``tinyns-ckpt-2``.

A checkpoint is one ``.npz`` file, written atomically (a temporary file, then
``os.replace``). Its entries:

- ``format``: ``"tinyns-ckpt-2"``;
- ``state/<field>``: the :class:`tinyns.core.State` leaves. ``state/key`` holds
  the raw ``uint32`` key data and ``state/key_impl`` the PRNG implementation of
  a typed key (``""`` for a legacy ``uint32`` key), so both round-trip exactly.
  ``state/scale`` is the host loop's step scale (float64);
- ``dead/<column>``: the dead rows so far (``u``, ``theta``, ``logl``,
  ``logwt``, ``birth``, ``ncall``, ``batches``);
- ``config_json``: the resolved :class:`tinyns.core.Config` (``ndim``,
  ``nlive``, ``walks``, ``replacement_chains``, ``block_size``,
  ``cluster_swap``) plus ``tinyns_version``;
- ``telemetry_json``: the host loop's running telemetry (``rwalk_moves``,
  ``rwalk_proposals``, and ``wall_time_s``, ``compile_s`` and ``timed_ncall``
  summed over the run so far, so the wall time accumulates over resumes);
- ``ext/<hook>/<name>``: a host hook's ``state_dict()``; arrays as they are,
  anything else as JSON under ``<name>.json``.

Files of any other format (v1 included) are not read. The callables are not
saved, so a resume cannot check them: pass the same ``loglike`` and
``prior_transform``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tinyns import core

FORMAT = "tinyns-ckpt-2"


class Checkpoint(NamedTuple):
    """A loaded checkpoint: everything :func:`tinyns.loop.run` resumes from."""

    state: core.State  # jax arrays; ``scale`` is the host float
    dead: dict[str, np.ndarray]  # dead-row columns
    config: dict[str, Any]
    telemetry: dict[str, Any]
    ext: dict[str, dict[str, Any]]  # hook name -> state_dict


def _key_impl(key) -> str:
    """The PRNG implementation name of a typed key."""
    spec = jax.random.key_impl(key)
    return getattr(getattr(spec, "_impl", spec), "name", str(spec))


def save(path, state: core.State, dead, cfg: core.Config, telemetry, ext=None):
    """Write a checkpoint atomically; ``ext`` maps hook names to state dicts."""
    from tinyns import __version__

    key, impl = state.key, ""
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        key, impl = jax.random.key_data(key), _key_impl(key)
    entries = {"format": np.asarray(FORMAT), "state/key_impl": np.asarray(impl)}
    for name, value in state._replace(key=key)._asdict().items():
        entries[f"state/{name}"] = np.asarray(value)
    for name, value in dead.items():
        entries[f"dead/{name}"] = np.asarray(value)
    config = {**dataclasses.asdict(cfg), "tinyns_version": __version__}
    entries["config_json"] = np.asarray(json.dumps(config))
    entries["telemetry_json"] = np.asarray(json.dumps(telemetry))
    for hook, state_dict in (ext or {}).items():
        for name, value in state_dict.items():
            if isinstance(value, np.ndarray | jax.Array):
                entries[f"ext/{hook}/{name}"] = np.asarray(value)
            else:
                entries[f"ext/{hook}/{name}.json"] = np.asarray(json.dumps(value))
    path = os.fspath(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as file:
        np.savez_compressed(file, **entries)
    os.replace(tmp_path, path)


def load(path) -> Checkpoint:
    """Read a checkpoint; raise ``ValueError`` for any other file format."""
    with np.load(path) as data:
        entries = {name: data[name] for name in data.files}
    if "format" not in entries or str(entries["format"]) != FORMAT:
        raise ValueError(
            f"not a {FORMAT} file: {os.fspath(path)!r} (checkpoints written "
            "before tinyns v0.3 are not read)"
        )
    leaves = {name: entries[f"state/{name}"] for name in core.State._fields}
    impl = str(entries["state/key_impl"])
    key = leaves.pop("key")
    key = jax.random.wrap_key_data(key, impl=impl) if impl else jnp.asarray(key)
    scale = float(leaves.pop("scale"))
    state = core.State(
        key=key, scale=scale, **{k: jnp.asarray(v) for k, v in leaves.items()}
    )
    dead, ext = {}, {}
    for name, value in entries.items():
        if name.startswith("dead/"):
            dead[name[5:]] = value
        elif name.startswith("ext/"):
            _, hook, field = name.split("/", 2)
            if field.endswith(".json"):
                field, value = field[:-5], json.loads(str(value))
            ext.setdefault(hook, {})[field] = value
    return Checkpoint(
        state=state,
        dead=dead,
        config=json.loads(str(entries["config_json"])),
        telemetry=json.loads(str(entries["telemetry_json"])),
        ext=ext,
    )


def check_config(config, cfg: core.Config) -> None:
    """Refuse a checkpoint whose config differs from ``cfg``, naming the key.

    Only the six :class:`tinyns.core.Config` keys are compared; the callables
    cannot be checked.
    """
    for name, value in dataclasses.asdict(cfg).items():
        if config.get(name) != value:
            raise ValueError(
                f"checkpoint {name}={config.get(name)!r} is not compatible "
                f"with sampler {name}={value!r}"
            )
