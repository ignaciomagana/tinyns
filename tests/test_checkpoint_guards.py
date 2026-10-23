"""Checkpoint guards: the path is checked before the run, the JAX build is
recorded and a change warns, the file keeps its mode, and odd inputs fail
early instead of at the last save."""

from __future__ import annotations

import json
import os
import re
import stat
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinyns import Config, NestedSampler, core


def gauss(x):
    return -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2)


def identity(u):
    return u


def test_an_unwritable_checkpoint_path_fails_before_the_run(tmp_path) -> None:
    """The first save came after the first chunk of 10 minutes or at the end:
    a short run computed everything, then failed."""
    calls = []

    def loglike(x):
        calls.append(1)  # traced: the run has started
        return gauss(x)

    sampler = NestedSampler(loglike, identity, 2, 50, num_delete=5)
    path = tmp_path / "missing" / "run.npz"
    with pytest.raises(FileNotFoundError, match=re.escape(str(path))):
        sampler.run(0, checkpoint=path)
    assert not calls and not path.parent.exists()
    if os.name == "posix" and os.geteuid() != 0:  # root writes anywhere
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        with pytest.raises(PermissionError, match="cannot write the checkpoint"):
            sampler.run(0, checkpoint=locked / "run.npz")
        assert not calls and not os.listdir(locked)


def test_the_jax_build_is_recorded_and_a_change_warns(tmp_path) -> None:
    path = tmp_path / "run.npz"
    sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    sampler.run(0, maxiter=100, checkpoint=path)

    def meta():
        with np.load(path) as data:
            return dict(data), json.loads(str(data["meta_json"][()]))

    _, written = meta()
    assert written["jax_version"] == jax.__version__
    assert written["backend"] == jax.default_backend()
    for name in ("jaxlib_version", "threefry_partitionable", "device_kind"):
        assert name in written
    with warnings.catch_warnings(record=True) as caught:  # same build: no warning
        warnings.simplefilter("always")
        sampler.run(0, maxiter=150, checkpoint=path)
    assert not [w for w in caught if "bit-identical" in str(w.message)]

    arrays, written = meta()
    written["jax_version"] = "0.4.0"
    written["threefry_partitionable"] = not written["threefry_partitionable"]
    arrays["meta_json"] = np.asarray(json.dumps(written))
    np.savez(path, **arrays)
    with pytest.warns(UserWarning, match=r"jax_version='0\.4\.0', threefry_part"
                      r".*may not be bit-identical"):
        result = sampler.run(0, checkpoint=path)
    assert result.metadata["resumed"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_a_checkpoint_keeps_its_file_mode(tmp_path) -> None:
    """A checkpoint was always 0600 (``mkstemp``), and a rewrite undid a
    ``chmod``, so group members could not read or resume it."""
    umask = os.umask(0o022)
    try:
        plain = tmp_path / "plain"
        plain.write_bytes(b"")
        path = tmp_path / "run.npz"
        sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
        sampler.run(0, maxiter=50, checkpoint=path)
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == stat.S_IMODE(plain.stat().st_mode)  # as any new file
        path.chmod(0o640)
        sampler.run(0, maxiter=100, checkpoint=path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    finally:
        os.umask(umask)


def test_a_npy_file_is_not_a_checkpoint(tmp_path) -> None:
    path = tmp_path / "samples.npy"
    np.save(path, np.arange(3.0))
    with pytest.raises(ValueError, match="not a tinyns-ckpt-7 checkpoint"):
        NestedSampler(gauss, identity, 2, 50, num_delete=5).run(0, checkpoint=path)


def test_numpy_int_folds_are_stored_as_an_int(tmp_path) -> None:
    """``_folds=np.int64(3)`` was accepted and then failed the JSON of the
    first save, at the end of a short run."""
    cfg = Config(2, 50, num_delete=5, _folds=np.int64(3))
    assert type(cfg._folds) is int and cfg == Config(2, 50, num_delete=5)
    result = core.run(0, gauss, identity, cfg, maxiter=100,
                      checkpoint=tmp_path / "run.npz")
    assert result.niter == 100
