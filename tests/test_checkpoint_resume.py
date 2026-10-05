import json
import math

import jax.numpy as jnp
import numpy as np
import pytest

import tinyns.run as run_mod
from tinyns import NestedSampler
from tinyns.run import run_static_nested
from tinyns.state import load_checkpoint_npz


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


def test_checkpoint_file_is_created(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    result = make_sampler().run(
        1, maxiter=3, checkpoint_path=path, checkpoint_interval=1
    )
    assert path.exists()
    assert math.isfinite(result.logz)


def test_resume_produces_valid_result(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(2, maxiter=3, checkpoint_path=path, checkpoint_interval=1)
    checkpoint_state, _ = load_checkpoint_npz(path)

    result = make_sampler().resume(path, maxiter=6)

    assert math.isfinite(result.logz)
    assert result.metadata["resumed_from_checkpoint"] is True
    assert result.metadata["initial_iteration"] == checkpoint_state.iteration
    assert result.metadata["final_iteration"] > checkpoint_state.iteration


def test_resume_does_not_reinitialize(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(3, maxiter=2, checkpoint_path=path, checkpoint_interval=1)
    checkpoint_state, _ = load_checkpoint_npz(path)

    result = make_sampler().resume(path, maxiter=5)

    assert result.metadata["ndead"] >= checkpoint_state.iteration
    assert result.ncall > checkpoint_state.ncall


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

    with pytest.raises(ValueError, match=match):
        make_sampler(**kwargs).resume(path, maxiter=3)


def test_bad_checkpoint_format_version_raises(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    bad_path = tmp_path / "bad.checkpoint.npz"
    make_sampler().run(5, maxiter=2, checkpoint_path=path)
    with np.load(path) as data:
        values = {name: data[name] for name in data.files}
    values["format_version"] = np.asarray("not-a-tinyns-checkpoint")
    np.savez(bad_path, **values)

    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint_npz(bad_path)


def test_missing_checkpoint_field_raises(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    bad_path = tmp_path / "missing.checkpoint.npz"
    make_sampler().run(6, maxiter=2, checkpoint_path=path)
    with np.load(path) as data:
        values = {name: data[name] for name in data.files if name != "key"}
    np.savez(bad_path, **values)

    with pytest.raises(ValueError, match="missing required checkpoint"):
        load_checkpoint_npz(bad_path)


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


def test_resume_rejects_maxiter_smaller_than_checkpoint_iteration(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(9, maxiter=4, dlogz=0.0, checkpoint_path=path)

    with pytest.raises(ValueError, match="maxiter.*checkpoint iteration"):
        make_sampler().resume(path, maxiter=2, dlogz=0.0)


def test_resume_matches_uninterrupted_run(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    sampler = make_sampler()

    full = sampler.run(10, maxiter=8, dlogz=0.0)
    sampler.run(10, maxiter=4, dlogz=0.0, checkpoint_path=path)
    resumed = sampler.resume(path, maxiter=8, dlogz=0.0)

    np.testing.assert_allclose(resumed.samples_u, full.samples_u)
    np.testing.assert_allclose(resumed.samples, full.samples)
    np.testing.assert_allclose(resumed.logl, full.logl)
    np.testing.assert_allclose(resumed.logwt, full.logwt)
    assert resumed.ncall == full.ncall
    assert resumed.logz == full.logz


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
    state, _ = load_checkpoint_npz(path)
    resumed = sampler.resume(path, maxiter=8, dlogz=0.0)

    assert len(state.telemetry["replacement_batches"]) == 4
    assert len(state.telemetry["rwalk_proposal_history"]) == 4
    assert resumed.ncall == full.ncall
    assert resumed.metadata["replacement_ncall"] == full.metadata[
        "replacement_ncall"
    ]
    for key in (
        "mean_replacement_batches",
        "max_replacement_batches",
        "accepted_rwalk_moves",
        "total_rwalk_proposals",
        "rwalk_adaptation_updates",
    ):
        assert resumed.metadata[key] == full.metadata[key]
    for key in (
        "rwalk_acceptance",
        "rwalk_scale_final",
        "rwalk_scale_min_seen",
        "rwalk_scale_max_seen",
        "rwalk_scale_mean",
        "rwalk_observed_accept_mean",
    ):
        assert resumed.metadata[key] == pytest.approx(full.metadata[key])


def test_resume_without_telemetry_payload_uses_empty_history_defaults(tmp_path):
    path = tmp_path / "telemetry_full.checkpoint.npz"
    stripped = tmp_path / "telemetry_stripped.checkpoint.npz"
    make_sampler().run(102, maxiter=2, dlogz=0.0, checkpoint_path=path)

    with np.load(path) as data:
        values = {
            name: data[name] for name in data.files if name != "telemetry_json"
        }
    np.savez(stripped, **values)

    state, _ = load_checkpoint_npz(stripped)
    result = make_sampler().resume(stripped, maxiter=3, dlogz=0.0)

    assert state.telemetry == {}
    assert result.metadata["resumed_from_checkpoint"] is True
    assert math.isfinite(result.logz)


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
    state, _ = load_checkpoint_npz(path)
    assert state.success is False
    assert state.replacement_failures == 1
    with pytest.raises(ValueError, match="replacement failure"):
        failed_sampler.resume(path, maxiter=10, dlogz=0.0)


def test_checkpoint_created_after_preallocation_can_be_loaded(tmp_path):
    path = tmp_path / "run.checkpoint.npz"

    make_sampler().run(12, maxiter=3, dlogz=0.0, checkpoint_path=path)
    state, config = load_checkpoint_npz(path)

    assert state.iteration == len(state.dead_logl)
    assert len(state.dead_u) == state.iteration
    assert config["ndim"] == 2


def test_resume_rejects_inconsistent_dead_count(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler().run(13, maxiter=3, dlogz=0.0, checkpoint_path=path)
    state, _ = load_checkpoint_npz(path)
    state.dead_u = state.dead_u[:-1]
    state.dead_theta = state.dead_theta[:-1]
    state.dead_logl = state.dead_logl[:-1]
    state.dead_logwt = state.dead_logwt[:-1]

    with pytest.raises(ValueError, match="dead point count.*iteration"):
        run_static_nested(
            state.key,
            loglike,
            prior_transform,
            ndim=2,
            nlive=20,
            dlogz=0.0,
            maxiter=5,
            initial_state=state,
        )


def _rewrite_checkpoint_config(path, update):
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    config = json.loads(str(arrays["config_json"].item()))
    update(config)
    arrays["config_json"] = np.asarray(json.dumps(config, sort_keys=True))
    with open(path, "wb") as file:
        np.savez_compressed(file, **arrays)


def test_checkpoint_config_includes_replacement_chains(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(walks=3, replacement_chains=2).run(
        16, maxiter=1, dlogz=0.0, checkpoint_path=path
    )

    _, config = load_checkpoint_npz(path)

    assert config["replacement_chains"] == 2


def test_resume_rejects_replacement_chains_mismatch(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(walks=3, replacement_chains=2).run(
        17, maxiter=1, dlogz=0.0, checkpoint_path=path
    )

    with pytest.raises(ValueError, match="replacement_chains"):
        make_sampler(walks=3).resume(path, maxiter=2)


def test_checkpoint_missing_config_key_is_rejected(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(walks=3).run(18, maxiter=1, dlogz=0.0, checkpoint_path=path)
    _rewrite_checkpoint_config(path, lambda config: config.pop("replacement_chains"))

    with pytest.raises(ValueError, match="checkpoint replacement_chains=None"):
        make_sampler(walks=3).resume(path, maxiter=2)


def test_checkpoint_config_and_telemetry_keys(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(walks=3).run(19, maxiter=40, dlogz=0.0, checkpoint_path=path)

    state, config = load_checkpoint_npz(path)
    assert config == {
        "ndim": 2,
        "nlive": 20,
        "walks": 3,
        "replacement_chains": 1,
        "block_size": 1,
        "cluster_swap": True,
    }
    assert set(state.telemetry) == {
        "replacement_batches",
        "rwalk_accepted_move_history",
        "rwalk_proposal_history",
        "adaptive_scale_history",
        "adaptive_accept_history",
        "adaptive_updates",
    }


def _rewrite_checkpoint_array(path, name, value):
    with np.load(path) as data:
        arrays = {n: data[n] for n in data.files}
    arrays[name] = np.asarray(value)
    with open(path, "wb") as file:
        np.savez_compressed(file, **arrays)


def test_block_mode_writes_intermediate_checkpoints(tmp_path, monkeypatch):
    saves = []
    original_save = run_mod.save_checkpoint_npz

    def recording_save(path, state, config):
        saves.append((int(state.iteration), bool(state.success)))
        return original_save(path, state, config)

    monkeypatch.setattr(run_mod, "save_checkpoint_npz", recording_save)

    path = tmp_path / "block.checkpoint.npz"
    # Block advances iteration by 8; with interval 10 the new cadence saves at
    # 16, 32, 48, ... The old `iteration % interval == 0` cadence would not save
    # until iteration 40 (block 32 / interval 100 example: not until 800).
    result = make_sampler(block_size=8).run(
        21, maxiter=400, dlogz=0.1, checkpoint_path=path, checkpoint_interval=10
    )

    final_iteration = result.metadata["final_iteration"]
    assert final_iteration > 16
    nonfinal_iterations = {it for (it, _s) in saves if it < final_iteration}
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

    state, _ = load_checkpoint_npz(path)
    assert state.scale == pytest.approx(first.metadata["rwalk_scale_final"])
    # Force a distinctive scale, clearly different from the initial 0.5.
    _rewrite_checkpoint_array(path, "scale", 0.037)

    resumed = sampler.resume(path, dlogz=0.5)

    assert resumed.success is True
    # Already converged on resume, so no further adaptation runs: the reported
    # final scale is the restored checkpoint value.
    assert resumed.metadata["rwalk_scale_final"] == pytest.approx(0.037)
    assert resumed.metadata["rwalk_scale_initial"] == 0.5


def test_resume_without_scale_field_falls_back_to_initial_scale(tmp_path):
    path = tmp_path / "adaptive_full.checkpoint.npz"
    stripped = tmp_path / "adaptive_stripped.checkpoint.npz"
    sampler = make_sampler()

    sampler.run(41, dlogz=0.5, checkpoint_path=path, checkpoint_interval=5)

    with np.load(path) as data:
        values = {name: data[name] for name in data.files if name != "scale"}
    np.savez(stripped, **values)

    state, _ = load_checkpoint_npz(stripped)
    assert state.scale is None

    resumed = sampler.resume(stripped, dlogz=0.5)

    assert resumed.success is True
    assert resumed.metadata["rwalk_scale_final"] == 0.5


def test_resume_at_maxiter_without_convergence_reports_maxiter_per_iteration(
    tmp_path, monkeypatch
):
    captured = []
    original_save = run_mod.save_checkpoint_npz

    def recording_save(path, state, config):
        captured.append(state)
        return original_save(path, state, config)

    monkeypatch.setattr(run_mod, "save_checkpoint_npz", recording_save)

    path = tmp_path / "midrun.checkpoint.npz"
    make_sampler().run(
        50, maxiter=6, dlogz=0.0, checkpoint_path=path, checkpoint_interval=1
    )

    # A mid-run snapshot: iteration 3, far from converged, carrying the neutral
    # in-progress labels (success=True, message="converged").
    state = next(s for s in captured if s.iteration == 3)
    assert state.success is True

    result = run_static_nested(
        state.key,
        loglike,
        prior_transform,
        ndim=2,
        nlive=20,
        dlogz=0.1,
        maxiter=3,
        initial_state=state,
    )

    assert result.success is False
    assert "maxiter" in result.message
    assert "converged" not in result.message


def test_resume_at_maxiter_without_convergence_reports_maxiter_block(tmp_path):
    path = tmp_path / "maxiter_block.checkpoint.npz"
    sampler = make_sampler(block_size=4)

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
    state, _ = load_checkpoint_npz(path)
    assert state.scale is not None
    assert state.scale != pytest.approx(0.5)

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
    _, config = load_checkpoint_npz(path)

    # The resolved defaults are recorded, not the None placeholders.
    assert config == {
        "ndim": 2,
        "nlive": 30,
        "walks": 25,
        "replacement_chains": 1,
        "block_size": 32,
        "cluster_swap": True,
    }

    resumed = default_sampler().resume(path, maxiter=128, dlogz=0.0)

    np.testing.assert_allclose(resumed.samples_u, full.samples_u)
    np.testing.assert_allclose(resumed.logl, full.logl)
    assert resumed.ncall == full.ncall
    assert resumed.logz == pytest.approx(full.logz)

