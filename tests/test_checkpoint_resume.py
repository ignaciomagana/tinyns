import json
import math
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tinyns.loop as loop_mod
from tinyns import NestedSampler, checkpoint, core


def loglike(theta):
    theta = jnp.asarray(theta)
    return -0.5 * jnp.sum(((theta - 0.5) / 0.2) ** 2)


def prior_transform(u):
    return jnp.asarray(u)


def make_sampler(**kwargs):
    # block_size=1 checkpoints after every iteration and compiles one kernel.
    options = {"ndim": 2, "nlive": 20, "walks": 5, "block_size": 1}
    options.update(kwargs)
    return NestedSampler(loglike, prior_transform, **options)


def host_ncall(ckpt):
    return int(ckpt.config["nlive"]) + int(ckpt.dead["ncall"].sum())


def _rewrite(path, update):
    """Rewrite the raw entries of an ``.npz`` file through ``update(entries)``."""
    with np.load(path) as data:
        entries = {name: data[name] for name in data.files}
    update(entries)
    with open(path, "wb") as file:
        np.savez_compressed(file, **entries)


@pytest.fixture(params=[False, True], ids=["x64off", "x64on"])
def x64(request):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param)
    yield request.param
    jax.config.update("jax_enable_x64", previous)


# --- the file ---


def test_checkpoint_file_is_created(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    result = make_sampler().run(
        1, maxiter=3, checkpoint_path=path, checkpoint_interval=1
    )
    assert path.exists()
    assert not (tmp_path / "run.checkpoint.npz.tmp").exists()  # atomic write
    assert math.isfinite(result.logz)


def test_checkpoint_layout(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(walks=3).run(19, maxiter=40, dlogz=0.0, checkpoint_path=path)

    with np.load(path) as data:
        files = set(data.files)
        assert str(data["format"]) == "tinyns-ckpt-2"
        assert data["state/key"].dtype == np.uint32
        assert str(data["state/key_impl"]) == ""  # an int seed: a legacy key
        assert data["state/scale"].dtype == np.float64
    columns = ("u", "theta", "logl", "logwt", "birth", "ncall", "insertion")
    assert files == {
        "format",
        "config_json",
        "telemetry_json",
        "state/key_impl",
        *(f"state/{name}" for name in core.State._fields),
        *(f"dead/{name}" for name in (*columns, "batches")),
        "ext/clusters/log.json",
        "ext/clusters/labels",
        "ext/clusters/u",
    }

    ckpt = checkpoint.load(path)
    assert ckpt.config == {
        "ndim": 2,
        "nlive": 20,
        "walks": 3,
        "replacement_chains": 1,
        "block_size": 1,
        "cluster_swap": True,
        "tinyns_version": __import__("tinyns").__version__,
    }
    assert set(ckpt.telemetry) == {
        "rwalk_moves",
        "rwalk_proposals",
        "scale_history",
        "accept_history",
    }
    assert int(ckpt.state.it) == 40
    assert all(len(ckpt.dead[name]) == 40 for name in ckpt.dead)
    assert ckpt.dead["u"].shape == (40, 2)
    assert set(ckpt.ext["clusters"]) == {"log", "labels", "u"}
    assert isinstance(ckpt.state.scale, float)
    assert ckpt.state.scale == ckpt.telemetry["scale_history"][-1]


def test_checkpoint_without_cluster_swap_has_no_ext(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(cluster_swap=False).run(2, maxiter=4, checkpoint_path=path)

    ckpt = checkpoint.load(path)
    assert ckpt.config["cluster_swap"] is False
    assert ckpt.ext == {}


@pytest.mark.parametrize("make_key", [jax.random.key, jax.random.PRNGKey])
def test_typed_and_legacy_keys_round_trip(tmp_path, make_key):
    path = tmp_path / "run.checkpoint.npz"
    sampler = make_sampler(block_size=4)
    key = make_key(5)
    full = sampler.run(key, maxiter=16, dlogz=0.0)
    sampler.run(key, maxiter=8, dlogz=0.0, checkpoint_path=path)

    ckpt = checkpoint.load(path)
    typed = jnp.issubdtype(ckpt.state.key.dtype, jax.dtypes.prng_key)
    assert typed is jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
    checkpoint.save(path, ckpt.state, ckpt.dead, sampler._config, ckpt.telemetry)
    again = checkpoint.load(path)
    assert again.state.key.dtype == ckpt.state.key.dtype
    data = jax.random.key_data if typed else np.asarray
    np.testing.assert_array_equal(data(again.state.key), data(ckpt.state.key))

    resumed = sampler.resume(path, maxiter=16, dlogz=0.0)
    np.testing.assert_array_equal(resumed.samples_u, full.samples_u)
    assert resumed.logz == full.logz


# --- refusals ---


def test_v1_checkpoint_is_refused(tmp_path):
    path = tmp_path / "v1.checkpoint.npz"
    np.savez(
        path,
        format_version=np.asarray("tinyns-checkpoint-npz-v1"),
        key=np.zeros(2, dtype=np.uint32),
        live_u=np.zeros((20, 2)),
        config_json=np.asarray(json.dumps({"ndim": 2, "nlive": 20})),
    )

    with pytest.raises(ValueError, match="not a tinyns-ckpt-2 file"):
        checkpoint.load(path)
    with pytest.raises(ValueError, match="not a tinyns-ckpt-2 file"):
        make_sampler().resume(path, maxiter=3)


def test_unknown_format_is_refused(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(5, maxiter=2, checkpoint_path=path)
    _rewrite(path, lambda e: e.update(format=np.asarray("tinyns-ckpt-3")))

    with pytest.raises(ValueError, match="not a tinyns-ckpt-2 file"):
        make_sampler().resume(path, maxiter=3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"ndim": 3}, "ndim"),
        ({"nlive": 25}, "nlive"),
        ({"walks": 6}, "walks"),
        ({"replacement_chains": 2}, "replacement_chains"),
        ({"block_size": 2}, "block_size"),
        ({"cluster_swap": False}, "cluster_swap"),
    ],
)
def test_incompatible_checkpoint_config_raises(tmp_path, kwargs, match):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(4, maxiter=2, checkpoint_path=path)

    with pytest.raises(ValueError, match=f"checkpoint {match}=.*sampler {match}="):
        make_sampler(**kwargs).resume(path, maxiter=3)


def test_checkpoint_missing_config_key_is_rejected(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(walks=3).run(18, maxiter=1, dlogz=0.0, checkpoint_path=path)

    def drop_chains(entries):
        config = json.loads(str(entries["config_json"]))
        config.pop("replacement_chains")
        entries["config_json"] = np.asarray(json.dumps(config))

    _rewrite(path, drop_chains)

    with pytest.raises(ValueError, match="checkpoint replacement_chains=None"):
        make_sampler(walks=3).resume(path, maxiter=2)


def test_resume_rejects_maxiter_smaller_than_checkpoint_iteration(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(9, maxiter=4, dlogz=0.0, checkpoint_path=path)

    with pytest.raises(ValueError, match="maxiter.*checkpoint iteration"):
        make_sampler().resume(path, maxiter=2, dlogz=0.0)


def test_resume_rejects_inconsistent_dead_count(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(13, maxiter=3, dlogz=0.0, checkpoint_path=path)
    ckpt = checkpoint.load(path)
    ckpt = ckpt._replace(dead={k: v[:-1] for k, v in ckpt.dead.items()})

    with pytest.raises(ValueError, match="dead point count.*iteration"):
        loop_mod.run(
            core.Config(2, 20, walks=5, block_size=1),
            loglike,
            prior_transform,
            None,
            dlogz=0.0,
            maxiter=5,
            resume=ckpt,
        )


def test_resume_rejects_checkpoint_after_replacement_failure(tmp_path):
    path = tmp_path / "failed.checkpoint.npz"

    # No point is admissible above a NaN threshold.
    failed_sampler = NestedSampler(
        lambda theta: jnp.nan,
        prior_transform,
        ndim=1,
        nlive=3,
        walks=10,
        cluster_swap=False,
    )
    result = failed_sampler.run(0, maxiter=10, dlogz=0.0, checkpoint_path=path)

    assert path.exists()
    assert result.success is False
    assert "replacement failed" in result.message
    assert bool(checkpoint.load(path).state.failed)
    with pytest.raises(ValueError, match="replacement failure"):
        failed_sampler.resume(path, maxiter=10, dlogz=0.0)


# --- resume ---


def test_resume_produces_valid_result(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(2, maxiter=3, checkpoint_path=path, checkpoint_interval=1)
    ckpt = checkpoint.load(path)

    result = make_sampler().resume(path, maxiter=6)

    assert math.isfinite(result.logz)
    assert result.metadata["resumed_from_checkpoint"] is True
    assert result.metadata["initial_iteration"] == int(ckpt.state.it)
    assert result.metadata["final_iteration"] > int(ckpt.state.it)
    assert result.ncall > host_ncall(ckpt)


def test_checkpoint_interval_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_interval"):
        make_sampler().run(
            7,
            maxiter=2,
            checkpoint_path=tmp_path / "run.checkpoint.npz",
            checkpoint_interval=0,
        )


def test_checkpoint_path_out_works(tmp_path):
    path_a = tmp_path / "a.checkpoint.npz"
    path_b = tmp_path / "b.checkpoint.npz"
    make_sampler().run(8, maxiter=2, checkpoint_path=path_a)

    make_sampler().resume(path_a, maxiter=4, checkpoint_path_out=path_b)

    assert path_b.exists()
    assert int(checkpoint.load(path_a).state.it) == 2
    assert int(checkpoint.load(path_b).state.it) == 4


def test_resume_matches_uninterrupted_run(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    sampler = make_sampler()

    full = sampler.run(10, maxiter=8, dlogz=0.0)
    sampler.run(10, maxiter=4, dlogz=0.0, checkpoint_path=path)
    resumed = sampler.resume(path, maxiter=8, dlogz=0.0)

    _assert_same_run(resumed, full)


def test_resume_preserves_cumulative_rwalk_telemetry(tmp_path):
    path = tmp_path / "rwalk_telemetry.checkpoint.npz"
    sampler = make_sampler(walks=2, replacement_chains=2)

    full = sampler.run(100, maxiter=8, dlogz=0.0)
    sampler.run(
        100,
        maxiter=4,
        dlogz=0.0,
        checkpoint_path=path,
        checkpoint_interval=1,
    )
    ckpt = checkpoint.load(path)
    resumed = sampler.resume(path, maxiter=8, dlogz=0.0)

    assert len(ckpt.dead["batches"]) == 4
    assert ckpt.telemetry["rwalk_proposals"] > 0
    assert len(ckpt.telemetry["scale_history"]) == 5  # initial + one per block
    _assert_same_run(resumed, full)


def test_block_mode_writes_intermediate_checkpoints(tmp_path, monkeypatch):
    saves = []
    original_save = checkpoint.save

    def recording_save(path, state, *args, **kwargs):
        saves.append(int(state.it))
        return original_save(path, state, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "save", recording_save)

    path = tmp_path / "block.checkpoint.npz"
    # Block advances iteration by 8; with interval 10 the cadence saves at
    # 16, 32, 48, ... An `iteration % interval == 0` cadence would not save
    # until iteration 40 (block 32 / interval 100 example: not until 800).
    result = make_sampler(block_size=8).run(
        21, maxiter=400, dlogz=0.1, checkpoint_path=path, checkpoint_interval=10
    )

    final_iteration = result.metadata["final_iteration"]
    assert final_iteration > 16
    nonfinal_iterations = {it for it in saves if it < final_iteration}
    assert nonfinal_iterations, "expected intermediate block-mode checkpoints"
    assert 16 in nonfinal_iterations
    assert all(it % 8 == 0 for it in nonfinal_iterations)
    assert result.success is True


@pytest.mark.parametrize("block_size", [1, 8])
def test_resume_of_converged_run_reports_success(tmp_path, block_size):
    path = tmp_path / "converged_block.checkpoint.npz"
    sampler = make_sampler(block_size=block_size)

    first = sampler.run(31, dlogz=0.5, checkpoint_path=path, checkpoint_interval=8)
    assert first.success is True
    assert "converged" in first.message

    resumed = sampler.resume(path, dlogz=0.5)

    assert resumed.success is True
    assert "converged" in resumed.message
    assert resumed.metadata["resumed_from_checkpoint"] is True


def test_adapted_scale_restored_on_resume(tmp_path):
    path = tmp_path / "adaptive.checkpoint.npz"
    sampler = make_sampler()

    first = sampler.run(40, dlogz=0.5, checkpoint_path=path, checkpoint_interval=5)
    assert first.success is True
    assert checkpoint.load(path).state.scale == first.metadata["rwalk_scale_final"]
    # Force a distinctive scale, clearly different from the initial 0.5.
    _rewrite(path, lambda e: e.update({"state/scale": np.asarray(0.037)}))

    resumed = sampler.resume(path, dlogz=0.5)

    assert resumed.success is True
    # Already converged on resume, so no further adaptation runs: the reported
    # final scale is the restored checkpoint value.
    assert resumed.metadata["rwalk_scale_final"] == 0.037
    assert resumed.metadata["rwalk_scale_initial"] == 0.5


@pytest.mark.parametrize("block_size", [1, 4])
def test_resume_at_maxiter_without_convergence_reports_maxiter(tmp_path, block_size):
    path = tmp_path / "maxiter.checkpoint.npz"
    sampler = make_sampler(block_size=block_size)

    first = sampler.run(51, maxiter=4, dlogz=0.0, checkpoint_path=path)
    assert first.success is False
    assert "maxiter" in first.message

    resumed = sampler.resume(path, maxiter=4, dlogz=0.1)

    assert resumed.success is False
    assert "maxiter" in resumed.message
    assert "converged" not in resumed.message


def test_live_cov_resume_continues_with_adapted_scale(tmp_path):
    path = tmp_path / "live_cov.checkpoint.npz"
    sampler = make_sampler(walks=8, block_size=4)
    partial = sampler.run(43, maxiter=40, dlogz=0.5, checkpoint_path=path)
    assert checkpoint.load(path).state.scale != pytest.approx(0.5)

    resumed = sampler.resume(path, dlogz=0.5)
    assert resumed.success is True
    assert len(resumed.logl) > len(partial.logl)


def test_default_kwargs_checkpoint_round_trip(tmp_path):
    path = tmp_path / "defaults.checkpoint.npz"

    def default_sampler():
        return NestedSampler(loglike, prior_transform, ndim=2, nlive=30)

    full = default_sampler().run(44, maxiter=128, dlogz=0.0)
    default_sampler().run(
        44, maxiter=64, dlogz=0.0, checkpoint_path=path, checkpoint_interval=32
    )
    config = checkpoint.load(path).config

    # The resolved defaults are recorded, not the None placeholders.
    assert config.pop("tinyns_version")
    assert config == {
        "ndim": 2,
        "nlive": 30,
        "walks": 25,
        "replacement_chains": 1,
        "block_size": 32,
        "cluster_swap": True,
    }

    resumed = default_sampler().resume(path, maxiter=128, dlogz=0.0)
    _assert_same_run(resumed, full)


# --- kill and resume: bit-identical to an uninterrupted run ---

# Keys that legitimately differ between a resumed and an uninterrupted run.
_RUN_SPECIFIC = {
    "wall_time_s",
    "compile_s",
    "mean_ms_per_call",
    "cluster_host_s",
    "checkpoint_path",
    "checkpoint_interval",
    "resumed_from_checkpoint",
    "initial_iteration",
}


def _assert_same_run(a, b):
    for name in ("samples_u", "samples", "logl", "logwt", "logl_birth"):
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert x.dtype == y.dtype, name
        np.testing.assert_array_equal(x, y, err_msg=name)
    assert a.logz == b.logz
    assert a.logzerr == b.logzerr
    assert a.ncall == b.ncall
    assert (a.success, a.message) == (b.success, b.message)
    assert set(a.metadata) == set(b.metadata)
    for key in sorted(set(a.metadata) - _RUN_SPECIFIC):
        x, y = jax.device_get((a.metadata[key], b.metadata[key]))
        np.testing.assert_equal(x, y, err_msg=key)


def two_modes(x):
    a = -0.5 * jnp.sum(((x - 0.3) / 0.04) ** 2)
    b = -0.5 * jnp.sum(((x - 0.7) / 0.04) ** 2)
    return jnp.logaddexp(math.log(0.7) + a, math.log(0.3) + b)


def test_kill_and_resume_is_bit_identical(tmp_path, monkeypatch, x64):
    sampler = NestedSampler(two_modes, prior_transform, 2, nlive=80, block_size=16)
    full = sampler.run(7)
    md = full.metadata
    assert md["cluster_swap"] is True
    assert md["cluster_swap_accepts"] > 0
    assert max(k for _, k in md["cluster_count_history"]) >= 2

    # Checkpoint after every block and keep a copy of each: a kill at any
    # block boundary leaves one of these files behind.
    original_save = checkpoint.save

    def keep_every_save(path, state, *args, **kwargs):
        original_save(path, state, *args, **kwargs)
        shutil.copy(path, tmp_path / f"it{int(state.it)}.npz")

    monkeypatch.setattr(checkpoint, "save", keep_every_save)
    path = tmp_path / "run.npz"
    with_saves = sampler.run(7, checkpoint_path=path, checkpoint_interval=1)
    monkeypatch.undo()
    _assert_same_run(with_saves, full)  # writing checkpoints changes nothing

    final = md["niter"]
    boundaries = sorted(int(p.stem[2:]) for p in tmp_path.glob("it*.npz"))
    assert boundaries == list(range(16, final, 16)) + [final]
    nclusters = {
        it: len(checkpoint.load(tmp_path / f"it{it}.npz").ext["clusters"]["log"]["ids"])
        for it in boundaries
    }
    # Kill with one cluster, just before and after the first split, at the
    # most clusters, at the last swapping boundary, and one block from the end.
    swapping = [it for it in boundaries if nclusters[it] >= 2]
    most = max(boundaries, key=nclusters.get)
    kills = {16, swapping[0] - 16, swapping[0], most, swapping[-1], boundaries[-2]}
    kills = sorted(kills)
    assert nclusters[16] == 1 and len(kills) >= 5

    for it in kills:
        resumed = sampler.resume(
            tmp_path / f"it{it}.npz", checkpoint_path_out=tmp_path / "out.npz"
        )
        assert resumed.metadata["initial_iteration"] == it
        _assert_same_run(resumed, full)

    # Killed twice: resume to the next kill, then to the end.
    first, second = swapping[0], swapping[-1]
    out = tmp_path / "chain.npz"
    sampler.resume(tmp_path / f"it{first}.npz", maxiter=second, checkpoint_path_out=out)
    assert int(checkpoint.load(out).state.it) == second
    _assert_same_run(sampler.resume(out), full)
