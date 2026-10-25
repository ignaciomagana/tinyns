"""Checkpoints (``tinyns-ckpt-8``): kill and resume, refusals, atomic writes,
progress lines and the accumulated wall time."""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial

from tinyns import NestedSampler, core
from tinyns import checkpoint as ckpt

TIMING = {"wall_time_s", "compile_s", "sampling_s", "chunks", "resumed"}


def gauss(x):
    return -0.5 * jnp.sum(((x - 0.5) / 0.1) ** 2)


def identity(u):
    return u


def assert_same_result(a, b) -> None:
    """Bit-identical samples, weights, evidence, counts and metadata."""
    for name in ("samples_u", "samples", "logl", "logwt", "logl_birth", "nlive_i"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), name)
    for name in ("logz", "logzerr", "ncall", "niter", "success", "message"):
        assert getattr(a, name) == getattr(b, name), name
    np.testing.assert_array_equal(a.insertion_indices(), b.insertion_indices())
    drop = TIMING | {"maxiter"}
    assert {k: v for k, v in a.metadata.items() if k not in drop} == {
        k: v for k, v in b.metadata.items() if k not in drop
    }


class Crash(Exception):
    pass


def crash_after(monkeypatch, n_saves: int) -> list:
    """Checkpoint after every chunk and raise after the ``n_saves``-th write."""
    monkeypatch.setattr(core, "_CHECKPOINT_SECONDS", 0.0)
    saves = []
    save = ckpt.save

    def save_then_crash(path, c):
        save(path, c)
        saves.append(int(np.max(np.asarray(c.state.it))))
        if len(saves) == n_saves:
            raise Crash

    monkeypatch.setattr(ckpt, "save", save_then_crash)
    return saves


@pytest.mark.parametrize("num_delete", [1, 5])
@pytest.mark.parametrize("typed", [False, True])
def test_kill_and_resume_is_bit_identical(tmp_path, monkeypatch, num_delete,
                                          typed) -> None:
    sampler = NestedSampler(gauss, lambda u: 2.0 * u, 2, 50, num_delete=num_delete)
    key = jax.random.key(3) if typed else 3
    full = sampler.run(key)
    path = tmp_path / "run.npz"

    # A crash in the middle of the run, after the third checkpoint.
    with monkeypatch.context() as patch:
        saves = crash_after(patch, 3)
        with pytest.raises(Crash):
            sampler.run(key, checkpoint=path)
    assert 0 < saves[-1] * num_delete < full.niter
    # Resume with a different chunk schedule (one step per chunk): chunk edges
    # do not touch the random stream.
    with monkeypatch.context() as patch:
        patch.setattr(core, "_CHUNK_SECONDS", 0.0)
        resumed = sampler.run(key, checkpoint=path)
    assert resumed.metadata["resumed"] and not full.metadata["resumed"]
    assert resumed.metadata["chunks"] > full.metadata["chunks"]
    assert_same_result(resumed, full)

    # A run stopped by maxiter continues to the same end.
    os.remove(path)
    stopped = sampler.run(key, maxiter=10 * num_delete, checkpoint=path)
    assert stopped.metadata["status"] == "maxiter"
    assert_same_result(sampler.run(key, checkpoint=path), full)
    # The finished checkpoint returns the same result without new steps.
    again = sampler.run(key, checkpoint=path)
    assert_same_result(again, full)


def capped_gauss(mu, sigma, top, x):
    return jnp.minimum(-0.5 * jnp.sum(((x - mu) / sigma) ** 2), top)


def scaled(width, u):
    return width * u


def test_batched_data_resume_with_lanes_in_mixed_states(tmp_path,
                                                        monkeypatch) -> None:
    """A batched_data run killed while one lane has converged, one has ended
    on a plateau and one still runs resumes bit-identically, lane by lane
    (the likelihood and the prior both carry per-lane data)."""
    mu = jnp.asarray([[0.3, 0.3], [0.5, 0.6], [0.7, 0.4]])
    sigma = jnp.asarray([0.003, 0.1, 0.03])
    top = jnp.asarray([0.0, 0.0, -2.0])  # lane 2: a flat top, so a plateau
    loglike = Partial(capped_gauss, mu, sigma[:, None], top)
    prior = Partial(scaled, jnp.asarray([[1.0], [2.0], [1.5]]))
    sampler = NestedSampler(loglike, prior, 2, 40, num_delete=4)
    keys = jax.random.split(jax.random.key(11), 3)
    full = sampler.run(keys, batched_data=True)
    assert [r.metadata["status"] for r in full] == [
        "converged", "converged", "plateau"
    ]
    path = tmp_path / "batch.npz"

    # Checkpoint after every one-step chunk; crash at the first mixed state.
    seen = []
    with monkeypatch.context() as patch:
        patch.setattr(core, "_CHECKPOINT_SECONDS", 0.0)
        patch.setattr(core, "_CHUNK_SECONDS", 0.0)
        save = ckpt.save

        def save_then_crash(p, c):
            save(p, c)
            status = set(np.asarray(c.state.status).tolist())
            seen.append(status)
            if status == {core.RUNNING, core.CONVERGED, core.PLATEAU}:
                raise Crash

        patch.setattr(ckpt, "save", save_then_crash)
        with pytest.raises(Crash):
            sampler.run(keys, checkpoint=path, batched_data=True)
    assert len(seen) > 1
    resumed = sampler.run(keys, checkpoint=path, batched_data=True)
    for a, b in zip(resumed, full, strict=True):
        assert a.metadata["resumed"]
        assert_same_result(a, b)
        np.testing.assert_array_equal(a.labels, b.labels)
        np.testing.assert_array_equal(a.insertion_ranks, b.insertion_ranks)


def test_maxcall_is_counted_across_resumes(tmp_path) -> None:
    sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    full = sampler.run(0, maxcall=3000)
    path = tmp_path / "run.npz"
    sampler.run(0, maxcall=1500, checkpoint=path)
    resumed = sampler.run(0, maxcall=3000, checkpoint=path)
    assert resumed.metadata["status"] == "maxcall"
    assert_same_result(resumed, full)


def test_wall_time_accumulates_across_resumes(tmp_path) -> None:
    sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    path = tmp_path / "run.npz"
    first = sampler.run(0, maxiter=100, checkpoint=path)
    wall = first.metadata["wall_time_s"]
    assert wall > first.metadata["compile_s"] > 0
    with np.load(path) as data:
        meta = json.loads(str(data["meta_json"][()]))
    assert meta["wall_time_s"] <= wall
    second = sampler.run(0, checkpoint=path)
    md = second.metadata
    assert md["wall_time_s"] > meta["wall_time_s"]
    assert md["compile_s"] > meta["compile_s"]
    assert md["chunks"] > first.metadata["chunks"]


def test_mismatches_are_refused(tmp_path) -> None:
    path = tmp_path / "run.npz"
    NestedSampler(gauss, identity, 2, 50, num_delete=5).run(0, maxiter=50,
                                                            checkpoint=path)
    cases = [
        (NestedSampler(gauss, identity, 2, 60, num_delete=5), 0, "nlive=50"),
        (NestedSampler(gauss, identity, 2, 50, num_delete=5, walks=7), 0, "walks"),
        (NestedSampler(gauss, identity, 3, 50, num_delete=5), 0, "ndim"),
        (NestedSampler(gauss, identity, 2, 50, num_delete=5), 1, "different key"),
        (NestedSampler(gauss, identity, 2, 50, num_delete=5),
         jax.random.split(jax.random.PRNGKey(0), 2), "batch"),
    ]
    for sampler, key, match in cases:
        with pytest.raises(ValueError, match=match):
            sampler.run(key, checkpoint=path)

    sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    x64 = bool(jax.config.jax_enable_x64)
    for field, value, match in (
        ("x64", not x64, "x64 flag"),
        ("dtype", "float16", "float dtype"),
        # 1.0 wrote tinyns-ckpt-7, without the frames kept for the hop
        ("format", "tinyns-ckpt-7", r"ckpt-8 checkpoint \(format 'tinyns-ckpt-7'"),
    ):
        with np.load(path) as data:
            arrays = dict(data)
        meta = json.loads(str(arrays["meta_json"][()]))
        meta[field] = value
        arrays["meta_json"] = np.asarray(json.dumps(meta))
        bad = tmp_path / f"bad_{field}.npz"
        np.savez(bad, **arrays)
        with pytest.raises(ValueError, match=match):
            sampler.run(0, checkpoint=bad)

    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="not a tinyns-ckpt-8"):
        sampler.run(0, checkpoint=junk)


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch) -> None:
    sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    path = tmp_path / "run.npz"
    sampler.run(0, maxiter=50, checkpoint=path)
    before = path.read_bytes()
    assert sorted(os.listdir(tmp_path)) == ["run.npz"]

    def torn_savez(f, **arrays):
        f.write(b"PK\x03\x04 half a zip file")
        raise OSError("disk full")

    monkeypatch.setattr(ckpt.np, "savez", torn_savez)
    with pytest.raises(OSError, match="disk full"):
        sampler.run(0, maxiter=100, checkpoint=path)
    assert path.read_bytes() == before  # the old checkpoint survives intact
    assert sorted(os.listdir(tmp_path)) == ["run.npz"]  # no temporary file left
    monkeypatch.undo()
    assert sampler.run(0, maxiter=100, checkpoint=path).niter == 100


@pytest.mark.parametrize(
    "key",
    [jax.random.PRNGKey(5), jax.random.key(5), jax.random.key(5, impl="rbg"),
     jax.random.split(jax.random.key(5), 3)],
)
def test_keys_round_trip(key) -> None:
    data, impl = ckpt.key_to_numpy(key)
    assert data.dtype == np.uint32
    back = ckpt.key_from_numpy(data, impl)
    assert back.dtype == key.dtype and back.shape == key.shape
    np.testing.assert_array_equal(ckpt.key_to_numpy(back)[0], data)
    flat = back.reshape(-1) if impl else back.reshape(-1, data.shape[-1])
    orig = key.reshape(-1) if impl else key.reshape(-1, data.shape[-1])
    draw = jax.vmap(lambda k: jax.random.uniform(k, (3,)))
    np.testing.assert_array_equal(draw(flat), draw(orig))


def test_progress_prints_one_line_per_chunk(capsys) -> None:
    sampler = NestedSampler(gauss, identity, 2, 50, num_delete=5)
    result = sampler.run(0, progress=True)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == result.metadata["chunks"]
    for line in lines:
        for field in ("niter=", "logz=", "dlogz=", "ncall=", "calls/s=", "acc=",
                      "scale="):
            assert field in line
    assert lines[-1].endswith("[converged]")
    assert f"niter={result.niter} " in lines[-1]
    assert f"ncall={result.ncall} " in lines[-1]

    results = sampler.run(jax.random.split(jax.random.PRNGKey(0), 3), progress=True)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == results[0].metadata["chunks"]
    assert "lanes running=0/3" in lines[-1]
    assert f"ncall={sum(r.ncall for r in results)} " in lines[-1]
    assert lines[-1].endswith("[converged=3]")


def test_batched_progress_reads_the_running_lanes(capsys) -> None:
    """The batched line takes niter, dlogz and scale over the running lanes:
    a lane that stopped (here a flat likelihood, a plateau at step 0) does
    not pin them. The last line counts the lanes in each status."""
    flat = Partial(capped_gauss, jnp.full((2, 2), 0.5), 0.1, jnp.asarray([0.0, -1e9]))
    sampler = NestedSampler(flat, identity, 2, 50, num_delete=5)
    results = sampler.run(jax.random.split(jax.random.key(0), 2), progress=True,
                          batched_data=True)
    assert [r.metadata["status"] for r in results] == ["converged", "plateau"]
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) > 2
    for line in lines[:-1]:
        assert "lanes running=1/2" in line and line.endswith(" [plateau=1]")
        assert "niter=0.." not in line and "inf" not in line
    ncall = sum(r.ncall for r in results)
    assert lines[-1].startswith(f"tinyns: lanes running=0/2 ncall={ncall} ")
    assert lines[-1].endswith(" [converged=1 plateau=1]")
    assert "niter=" not in lines[-1] and "scale=" not in lines[-1]

    status = np.asarray([core.CONVERGED, core.RUNNING, core.PLATEAU, core.RUNNING])
    line = core._progress_line(
        sampler.config, 4, status, np.asarray([60, 59, 0, 40]),
        np.zeros(4), np.asarray([0.09, 0.5, np.inf, 1.25]), [1, 2, 3, 4], 1.0,
        0.25, np.asarray([9.0, 1.0, 9.0, 2.0]),
    )
    assert line == (
        "tinyns: lanes running=2/4 niter=200..295 dlogz<=1.25 ncall=10 "
        "calls/s=1 acc=0.250 scale=1.5 [converged=1 plateau=1]"
    )
