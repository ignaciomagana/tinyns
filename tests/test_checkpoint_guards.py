"""Checkpoint guards: the file keeps its mode."""

from __future__ import annotations

import os
import stat

import jax.numpy as jnp
import pytest

from tinyns import NestedSampler


def gauss(x):
    return -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2)


def identity(u):
    return u


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
