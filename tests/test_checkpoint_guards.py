"""Checkpoint guards: a checkpoint of another problem is refused (some of its
live and dead points are evaluated again), the path is checked before the run
and resolved once, the JAX build is recorded and a change warns, the file
keeps its mode and a link at the path is written through, and odd inputs fail
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
from jax.tree_util import Partial

from tinyns import Config, NestedSampler, core


def gauss(x):
    return -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2)


def identity(u):
    return u


def loglike_data(data, theta):
    return -0.5 * jnp.sum((data - theta[0]) ** 2)


def prior(u):
    return -5.0 + 10.0 * u


def test_a_finished_checkpoint_of_other_data_is_refused(tmp_path) -> None:
    """A loop over datasets that reuses one path and key got the first
    dataset's result for every dataset."""
    path = tmp_path / "run.npz"
    first = NestedSampler(Partial(loglike_data, jnp.full((20,), -2.0)), prior, 1, 100)
    done = first.run(0, checkpoint=path)
    other = NestedSampler(Partial(loglike_data, jnp.zeros(20)), prior, 1, 100)
    with pytest.raises(ValueError, match="use one checkpoint path per problem"):
        other.run(0, checkpoint=path)
    again = first.run(0, checkpoint=path)  # the same problem comes back
    assert again.metadata["resumed"] and again.logz == done.logz


@pytest.mark.parametrize("change", ["loglike", "prior"])
def test_an_unfinished_checkpoint_of_another_model_is_refused(tmp_path,
                                                              change) -> None:
    path = tmp_path / "run.npz"
    NestedSampler(gauss, identity, 2, 50, num_delete=5).run(
        0, maxiter=100, checkpoint=path
    )
    loglike = (lambda x: gauss(x) / 4.0) if change == "loglike" else gauss
    prior_transform = (lambda u: 0.9 * u) if change == "prior" else identity
    sampler = NestedSampler(loglike, prior_transform, 2, 50, num_delete=5)
    with pytest.raises(ValueError, match="do not give back the likelihoods"):
        sampler.run(0, checkpoint=path)


def test_a_batched_checkpoint_is_checked_lane_by_lane(tmp_path) -> None:
    path = tmp_path / "run.npz"
    data = jnp.stack([jnp.full((20,), mean) for mean in (-2.0, 0.0, 2.0)])
    keys = jax.random.split(jax.random.key(1), 3)
    sampler = NestedSampler(Partial(loglike_data, data), prior, 1, 50, num_delete=5)
    sampler.run(keys, batched_data=True, maxiter=100, checkpoint=path)
    changed = NestedSampler(
        Partial(loglike_data, data.at[2].add(0.5)), prior, 1, 50, num_delete=5
    )
    with pytest.raises(ValueError, match=r"stored in the checkpoint \(lane 2\)"):
        changed.run(keys, batched_data=True, checkpoint=path)
    results = sampler.run(keys, batched_data=True, checkpoint=path)
    assert all(result.metadata["resumed"] for result in results)


def test_an_infinite_stored_likelihood_must_come_back_exactly(tmp_path) -> None:
    """The relative tolerance was infinite at a stored ``-inf``, so any value
    matched it: a likelihood that gave NaN everywhere, fixed and rerun on the
    same path, returned the finished ``logz=-inf`` run, and a model that is
    finite where the old one was ``-inf`` resumed an early checkpoint."""
    path = tmp_path / "nan.npz"
    NestedSampler(lambda x: jnp.nan * x[0], identity, 2, 50, num_delete=5).run(
        0, checkpoint=path
    )
    fixed = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    with pytest.raises(ValueError, match="an infinite value must come back exactly"):
        fixed.run(0, checkpoint=path)

    def sliver(x):
        return jnp.where(x[0] < 0.1, gauss(x), -jnp.inf)

    path = tmp_path / "sliver.npz"
    sampler = NestedSampler(sliver, identity, 2, 40, num_delete=4)
    sampler.run(0, maxiter=4, checkpoint=path)
    with np.load(path) as data:
        assert 0 < np.isfinite(data["state/logl"]).sum() < core._RESUME_CHECKS
    widened = NestedSampler(gauss, identity, 2, 40, num_delete=4)
    with pytest.raises(ValueError, match="an infinite value must come back exactly"):
        widened.run(0, checkpoint=path)
    assert sampler.run(0, maxiter=8, checkpoint=path).metadata["resumed"]


def test_a_change_the_live_points_do_not_see_is_refused(tmp_path) -> None:
    """Only the live points of highest likelihood were evaluated again: a
    finished plateau run, whose live points all lie in the box, resumed with a
    wider box was returned as it was (its dead points come back finite)."""

    def box(half):
        def loglike(x):
            return jnp.where(jnp.all(jnp.abs(x - 0.5) < half), 0.0, -jnp.inf)

        return loglike

    path = tmp_path / "box.npz"
    sampler = NestedSampler(box(0.2), identity, 2, 50, num_delete=5)
    done = sampler.run(0, checkpoint=path)
    assert done.metadata["status"] == "plateau" and np.isfinite(done.logz)
    wider = NestedSampler(box(0.45), identity, 2, 50, num_delete=5)
    with pytest.raises(ValueError, match="stored at a dead point"):
        wider.run(0, checkpoint=path)
    again = sampler.run(0, checkpoint=path)  # the same box comes back
    assert again.metadata["resumed"] and again.logz == done.logz


def test_the_refusal_gives_the_relative_difference(tmp_path) -> None:
    path = tmp_path / "run.npz"
    NestedSampler(gauss, identity, 2, 50, num_delete=5).run(
        0, maxiter=100, checkpoint=path
    )
    shifted = NestedSampler(lambda x: gauss(x) - 1.0, identity, 2, 50, num_delete=5)
    with pytest.raises(ValueError) as info:
        shifted.run(0, checkpoint=path)
    found = re.search(r"a relative difference of (\S+) where (\S+) is allowed",
                      str(info.value))
    eps = np.finfo(jnp.result_type(float)).eps
    assert float(found[1]) > 1e-3 and float(found[2]) == pytest.approx(
        np.sqrt(eps), rel=1e-2
    )
    assert "tinyns.core._RESUME_CHECKS = 0 skips this check" in str(info.value)


def test_a_likelihood_equal_up_to_roundoff_resumes(tmp_path) -> None:
    """The live points are evaluated by a kernel compiled apart from the run's,
    which may differ in the last bits: a few ulps are not a mismatch."""
    path = tmp_path / "run.npz"
    NestedSampler(gauss, identity, 2, 50, num_delete=5).run(
        0, maxiter=100, checkpoint=path
    )

    def same(x):
        z = (x - 0.5) / 0.1
        return -0.5 * jnp.dot(z, z) * (1.0 + 4.0 * jnp.finfo(z.dtype).eps)

    result = NestedSampler(same, identity, 2, 50, num_delete=5).run(
        0, checkpoint=path
    )
    assert result.metadata["resumed"]


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


def test_a_path_that_names_no_file_fails_before_the_run(tmp_path,
                                                        monkeypatch) -> None:
    """For ``""`` or ``"missing/"`` the probe wrote in the parent directory
    and passed: the run sampled to the end and failed at its last save."""
    calls = []

    def loglike(x):
        calls.append(1)  # traced: the run has started
        return gauss(x)

    sampler = NestedSampler(loglike, identity, 2, 50, num_delete=5)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    for path in ("", "missing/", "out", "out/"):
        with pytest.raises(IsADirectoryError, match="names no file"):
            sampler.run(0, checkpoint=path)
    assert not calls
    assert os.listdir(tmp_path) == ["out"] and not os.listdir(tmp_path / "out")


def test_a_path_through_a_missing_directory_is_resolved_once(tmp_path,
                                                             monkeypatch) -> None:
    """The probe tested the path as given but wrote where it resolves to:
    ``"missing/.."`` (the current directory) passed and the run failed at its
    last save, and ``"missing/../run.npz"`` did not exist for the resume test
    but was saved as ``run.npz``, so the run never resumed."""
    calls = []

    def loglike(x):
        calls.append(1)  # traced: the run has started
        return gauss(x)

    sampler = NestedSampler(loglike, identity, 2, 50, num_delete=5)
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    with pytest.raises(IsADirectoryError, match="names no file"):
        sampler.run(0, checkpoint="missing/..")
    assert not calls and os.listdir(tmp_path) == ["sub"]
    sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    sampler.run(0, maxiter=50, checkpoint="missing/../run.npz")
    assert os.listdir(tmp_path / "sub") == ["run.npz"]
    result = sampler.run(0, maxiter=100, checkpoint="missing/../run.npz")
    assert result.metadata["resumed"] and result.niter == 100


@pytest.mark.skipif(os.name != "posix", reason="symbolic links")
def test_a_checkpoint_behind_a_symbolic_link_is_written_through(tmp_path) -> None:
    """The atomic write replaced a link at the path with a file, so the linked
    checkpoint (in a shared directory, say) was never updated."""
    shared = tmp_path / "shared"
    shared.mkdir()
    link = tmp_path / "run.npz"
    link.symlink_to(shared / "run.npz")  # dangling until the first save
    sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    sampler.run(0, maxiter=50, checkpoint=link)
    assert link.is_symlink() and os.listdir(shared) == ["run.npz"]
    result = sampler.run(0, maxiter=100, checkpoint=link)
    assert result.metadata["resumed"] and link.is_symlink()
    assert os.listdir(shared) == ["run.npz"]
    with np.load(shared / "run.npz") as data:
        assert json.loads(str(data["meta_json"][()]))["ncall"] == [result.ncall]


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
    assert written["mixed_build"] is False
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
        result = sampler.run(0, maxiter=200, checkpoint=path)
    assert result.metadata["resumed"]
    # The next save records this build, but the run stays mixed: it warns on.
    _, written = meta()
    assert written["jax_version"] == jax.__version__ and written["mixed_build"]
    with pytest.warns(UserWarning, match=r"was resumed under another tinyns "
                      r"version, JAX build or device before"):
        sampler.run(0, checkpoint=path)


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
