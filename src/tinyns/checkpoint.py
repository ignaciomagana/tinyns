"""Checkpoints of the :func:`tinyns.core.run` driver (format ``tinyns-ckpt-7``).

A checkpoint is one ``.npz`` file, written atomically: a temporary file in the
same directory, ``fsync``, then ``os.replace``, so a crash leaves either the
previous checkpoint or the new one, never a torn file. It holds

- ``state/<field>``: the :class:`tinyns.core.State` leaves (with a leading
  lane axis for a batched run), including the mode-tracking fields (the
  cluster labels, frames and cluster ids of each clustering) and the
  plateau bookkeeping (``log_x_ties``, ``lstar_prev``, ``tie_run``). The
  PRNG key is stored as its raw ``uint32`` key data, with ``state/key_impl``
  naming the implementation of a typed key (``""`` for a raw ``uint32``
  key), so both kinds round-trip exactly;
- ``init_key`` (and ``init_key_impl``): the key the run started from, so a
  resume with a different key is refused;
- ``dead/<lane>/<column>``: the dead rows of each lane so far, ``(steps, k,
  ...)``, one entry per :class:`tinyns.core.Dead` column (with the live
  count ``nlive`` and insertion slots ``nslots`` of every row);
- ``meta_json``: the format, the tinyns version, the resolved ``Config``, the
  x64 flag and float dtype, the lane count and ``batched_data``, the host call
  counters (Python ints, one per lane), the accumulated ``wall_time_s``,
  ``compile_s`` and ``sampling_s``, the number of chunks, the JAX build
  and device the run is bit-identical on (:func:`_environment`), and
  ``mixed_build``, true once the run has been resumed under another one.

Files of any other format are refused; there is no reader for older formats.
A different JAX build or device is not refused but warned about: the resumed
run is a valid run, but it may not be bit-identical to an uninterrupted one.
The warning is repeated at every later resume (``mixed_build``), although the
checkpoints written after it record the new build. The file keeps the mode of
the checkpoint it replaces (a new one gets the mode of any new file, ``0o666``
less the umask), so a ``chmod`` that shares it with a group survives the
rewrites. A symbolic link at the path is followed, as ``open`` does.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import json
import os
import secrets
import stat
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax import random

FORMAT = "tinyns-ckpt-7"


def _is_typed_key(key) -> bool:
    dtype = getattr(key, "dtype", None)
    return dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)


# The PRNG implementations a typed key may use, by the name ``jax.random.key``
# takes (``str(jax.random.key_impl(key))`` is not that name on every version).
_KEY_IMPLS = ("threefry2x32", "rbg", "unsafe_rbg")


def _impl_name(key) -> str:
    impl = random.key_impl(key)
    if isinstance(impl, str):
        return impl
    for name in _KEY_IMPLS:
        if random.key_impl(random.key(0, impl=name)) == impl:
            return name
    raise ValueError(f"cannot checkpoint a key of PRNG implementation {impl!r}")


def key_to_numpy(key) -> tuple[np.ndarray, str]:
    """Return ``(uint32 key data, impl)``; ``impl`` is ``""`` for a raw key."""
    if _is_typed_key(key):
        return np.asarray(random.key_data(key)), _impl_name(key)
    return np.asarray(key), ""


def key_from_numpy(data, impl: str):
    """Inverse of :func:`key_to_numpy`."""
    data = jnp.asarray(np.asarray(data, np.uint32))
    return random.wrap_key_data(data, impl=impl) if impl else data


@dataclasses.dataclass
class Checkpoint:
    """The contents of a checkpoint, with host (numpy) arrays.

    ``state`` is a :class:`tinyns.core.State` whose key is a JAX key; ``dead``
    holds one :class:`tinyns.core.Dead` of numpy rows per lane (a single lane
    for an unbatched run); ``ncall`` and ``ncall_valid`` are per-lane Python
    ints; ``batch`` is the lane count of a batched run (``None`` unbatched);
    ``mixed_build`` is true once the run has been resumed under another
    tinyns version, JAX build or device.
    """

    state: Any
    dead: list
    init_key: Any
    ncall: list
    ncall_valid: list
    wall_time_s: float
    compile_s: float
    sampling_s: float
    chunks: int
    config: dict
    batch: int | None
    batched_data: bool
    mixed_build: bool = False


def _dtype_name() -> str:
    return jnp.dtype(jnp.result_type(float)).name


def _device(array):
    """The device that ``array`` (a state leaf) is on; the first device for a
    host array."""
    devices = getattr(array, "devices", None)
    return next(iter(devices())) if callable(devices) else jax.devices()[0]


def _environment(device) -> dict:
    """The JAX build and device that a resume is bit-identical on: the JAX
    and jaxlib versions, ``jax_threefry_partitionable`` (it changes the random
    stream; its default changed in JAX 0.5), and the platform and kind of
    ``device``, the one the run's state is on."""
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "threefry_partitionable": getattr(
            jax.config, "jax_threefry_partitionable", None
        ),
        "backend": device.platform,
        "device_kind": device.device_kind,
    }


def save(path, ckpt: Checkpoint) -> None:
    """Write ``ckpt`` to ``path`` atomically (temporary file, fsync, replace)."""
    from tinyns import __version__

    arrays = {}
    state = ckpt.state
    for name, value in state._asdict().items():
        if name == "key":
            data, impl = key_to_numpy(value)
            arrays["state/key"] = data
            arrays["state/key_impl"] = np.asarray(impl)
        else:
            arrays[f"state/{name}"] = np.asarray(value)
    data, impl = key_to_numpy(ckpt.init_key)
    arrays["init_key"] = data
    arrays["init_key_impl"] = np.asarray(impl)
    for lane, dead in enumerate(ckpt.dead):
        for name, value in dead._asdict().items():
            arrays[f"dead/{lane}/{name}"] = np.asarray(value)
    meta = {
        "format": FORMAT,
        "tinyns_version": __version__,
        "config": ckpt.config,
        "x64": bool(jax.config.jax_enable_x64),
        "dtype": _dtype_name(),
        "batch": ckpt.batch,
        "batched_data": bool(ckpt.batched_data),
        "ncall": [int(n) for n in ckpt.ncall],
        "ncall_valid": [int(n) for n in ckpt.ncall_valid],
        "wall_time_s": float(ckpt.wall_time_s),
        "compile_s": float(ckpt.compile_s),
        "sampling_s": float(ckpt.sampling_s),
        "chunks": int(ckpt.chunks),
        **_environment(_device(state.u)),
        "mixed_build": bool(ckpt.mixed_build),
    }
    arrays["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))
    _atomic_savez(path, arrays)


def _file_path(path) -> str:
    """``path`` with its symbolic links resolved, so that a link is written
    through (replacing the link itself would leave its target stale).
    Raises ``IsADirectoryError`` when it names no file: it is empty, ends in a
    separator or is a directory (the parent directory was written to, or the
    run failed at its last save)."""
    path = os.fspath(path)
    if not os.path.basename(path) or os.path.isdir(path):
        raise IsADirectoryError(
            errno.EISDIR,
            f"{path!r} names no file (it is empty, ends in a separator or is a "
            "directory)",
        )
    return os.path.realpath(path)


def _temp_file(path: str) -> tuple[int, str]:
    """Create a new file next to ``path`` for writing; return ``(fd, name)``.

    It is created like any new file (mode ``0o666`` less the umask, where
    ``mkstemp`` would make it ``0o600``) and then takes the mode of ``path``
    if that exists, so that replacing ``path`` keeps its mode.
    """
    directory = os.path.dirname(os.path.abspath(path))
    prefix = os.path.join(directory, f".{os.path.basename(path)}.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        tmp = f"{prefix}{secrets.token_hex(4)}.tmp"
        try:
            fd = os.open(tmp, flags, 0o666)
        except FileExistsError:
            continue
        with contextlib.suppress(OSError):  # no file to replace yet
            os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode))
        return fd, tmp
    raise FileExistsError(f"no unused temporary file name next to {path}")


def check_writable(path) -> None:
    """Raise ``OSError`` naming ``path`` unless a file can be written next to
    it, as :func:`save` does: a run checks its checkpoint path before it
    compiles and samples, not at its first save. A temporary file is created
    and removed (``os.access`` is unreliable on network file systems)."""
    path = os.fspath(path)
    target = _file_path(path)
    try:
        fd, tmp = _temp_file(target)
    except OSError as exc:
        directory = os.path.dirname(target)
        raise type(exc)(
            exc.errno,
            f"cannot write the checkpoint {path} in the directory {directory}: "
            f"{exc.strerror or exc}",
        ) from exc
    os.close(fd)
    os.unlink(tmp)


def _atomic_savez(path, arrays: dict, compressed: bool = False) -> None:
    """Write ``arrays`` to the ``.npz`` file ``path`` (exactly that name):
    a temporary file in the same directory, ``fsync``, then ``os.replace``,
    so that a crash leaves the old file or the new one. The file keeps the
    mode of the file it replaces (:func:`_temp_file`), and a symbolic link at
    ``path`` is written through (:func:`_file_path`)."""
    path = _file_path(path)
    directory = os.path.dirname(path)
    fd, tmp = _temp_file(path)
    try:
        with os.fdopen(fd, "wb") as f:
            (np.savez_compressed if compressed else np.savez)(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    with contextlib.suppress(OSError):  # make the rename durable (POSIX)
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def load(path, *, config: dict, batch: int | None, batched_data: bool, key):
    """Read the checkpoint at ``path`` for a run of ``config`` from ``key``.

    Raises ``ValueError`` naming the mismatch when the file is not a
    ``tinyns-ckpt-7`` checkpoint, or when its config, x64 flag, float dtype,
    lane count, ``batched_data`` or starting key differ from this run's.
    Warns when it was written by another tinyns version, JAX build or device
    (:func:`_environment`), or was resumed under another before
    (``mixed_build``): the resumed run may not be bit-identical.
    Returns a :class:`Checkpoint`.
    """
    from tinyns import __version__
    from tinyns.core import Dead, State

    path = os.fspath(path)
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is refused
        raise ValueError(f"{path} is not a {FORMAT} checkpoint: {exc}") from exc
    if not isinstance(data, np.lib.npyio.NpzFile):  # a .npy file
        raise ValueError(f"{path} is not a {FORMAT} checkpoint (not an .npz file)")
    with data:
        if "meta_json" not in data.files:
            raise ValueError(f"{path} is not a {FORMAT} checkpoint")
        meta = json.loads(str(data["meta_json"][()]))
        if meta.get("format") != FORMAT:
            raise ValueError(
                f"{path} is not a {FORMAT} checkpoint (format "
                f"{meta.get('format')!r}); older formats are not read"
            )
        _check(path, "x64 flag", meta["x64"], bool(jax.config.jax_enable_x64))
        _check(path, "float dtype", meta["dtype"], _dtype_name())
        diffs = [
            f"{name}={meta['config'].get(name)!r} (checkpoint) != {value!r}"
            for name, value in config.items()
            if meta["config"].get(name) != value
        ]
        diffs += [
            f"{name}={value!r} (checkpoint) is not in this config"
            for name, value in meta["config"].items()
            if name not in config
        ]
        if diffs:
            raise ValueError(f"{path}: config mismatch: " + "; ".join(diffs))
        _check(path, "batch of keys", meta["batch"], batch)
        _check(path, "batched_data", meta["batched_data"], bool(batched_data))
        key_data, key_impl = key_to_numpy(key)
        if str(data["init_key_impl"][()]) != key_impl or not np.array_equal(
            data["init_key"], key_data
        ):
            raise ValueError(
                f"{path}: the checkpoint was started from a different key; pass "
                "the original key or a new checkpoint path"
            )
        fields = {}
        for name in State._fields:
            if name == "key":
                fields[name] = key_from_numpy(
                    data["state/key"], str(data["state/key_impl"][()])
                )
            else:
                fields[name] = jnp.asarray(data[f"state/{name}"])
        now = {"tinyns_version": __version__, **_environment(_device(fields["u"]))}
        changed = [
            name for name, value in now.items() if meta.get(name, value) != value
        ]
        if changed:
            warnings.warn(
                f"{path} was written with "
                + ", ".join(f"{name}={meta[name]!r}" for name in changed)
                + "; this run has "
                + ", ".join(f"{name}={now[name]!r}" for name in changed)
                + ": the resumed run may not be bit-identical to an "
                "uninterrupted one",
                stacklevel=3,
            )
        elif meta.get("mixed_build"):
            warnings.warn(
                f"{path} was resumed under another tinyns version, JAX build "
                "or device before: the run may not be bit-identical to an "
                "uninterrupted one",
                stacklevel=3,
            )
        lanes = 1 if batch is None else batch
        dead = [
            Dead(**{
                name: np.asarray(data[f"dead/{lane}/{name}"])
                for name in Dead._fields
            })
            for lane in range(lanes)
        ]
    return Checkpoint(
        state=State(**fields),
        dead=dead,
        init_key=key,
        ncall=[int(n) for n in meta["ncall"]],
        ncall_valid=[int(n) for n in meta["ncall_valid"]],
        wall_time_s=float(meta["wall_time_s"]),
        compile_s=float(meta["compile_s"]),
        sampling_s=float(meta["sampling_s"]),
        chunks=int(meta["chunks"]),
        config=meta["config"],
        batch=meta["batch"],
        batched_data=bool(meta["batched_data"]),
        mixed_build=bool(changed or meta.get("mixed_build")),
    )


def _check(path, what, saved, current) -> None:
    if saved != current:
        raise ValueError(
            f"{path}: {what} mismatch: {saved!r} (checkpoint) != {current!r} "
            "(this run)"
        )
