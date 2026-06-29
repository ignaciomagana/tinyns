import json
import math

import jax.numpy as jnp
import numpy as np
import pytest

from tinyns import NestedSampler
from tinyns.run import run_static_nested
from tinyns.state import load_checkpoint_npz


def loglike(theta):
    theta = jnp.asarray(theta)
    return -0.5 * jnp.sum(((theta - 0.5) / 0.2) ** 2)


def prior_transform(u):
    return jnp.asarray(u)


def make_sampler(**kwargs):
    options = {"ndim": 2, "nlive": 20, "sample": "prior"}
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
        ({"sample": "rwalk"}, "sample"),
        ({"step_scale": 0.2}, "step_scale"),
        ({"min_accepts": 2}, "min_accepts"),
    ],
)
def test_incompatible_checkpoint_config_raises(tmp_path, kwargs, match):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(step_scale=0.1).run(4, maxiter=2, checkpoint_path=path)

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


def test_checkpoint_with_min_accepts_two_resumes_with_matching_config(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    sampler = make_sampler(sample="rwalk", walks=3, min_accepts=2)

    sampler.run(11, maxiter=2, dlogz=0.0, checkpoint_path=path)
    result = sampler.resume(path, maxiter=4, dlogz=0.0)

    assert math.isfinite(result.logz)
    assert result.metadata["min_accepts"] == 2
    assert result.metadata["resumed_from_checkpoint"] is True


def test_resume_rejects_checkpoint_after_replacement_failure(tmp_path):
    path = tmp_path / "failed.checkpoint.npz"

    def increasing_loglike(theta):
        return float(jnp.asarray(theta)[0])

    failed_sampler = NestedSampler(
        increasing_loglike,
        prior_transform,
        ndim=1,
        nlive=1,
        max_attempts=1,
    )
    result = failed_sampler.run(0, maxiter=10, dlogz=0.0, checkpoint_path=path)

    assert path.exists()
    assert result.success is False
    assert "max_attempts" in result.message
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


def test_checkpoint_kernel_mismatch_raises(tmp_path):
    path = tmp_path / "jax.checkpoint.npz"
    make_sampler(sample="rwalk", kernel="jax", walks=3, step_scale=0.05).run(
        14, maxiter=2, dlogz=0.0, checkpoint_path=path
    )

    with pytest.raises(ValueError, match="kernel"):
        make_sampler(sample="rwalk", kernel="python", walks=3, step_scale=0.05).resume(
            path, maxiter=3
        )


def test_checkpoint_missing_kernel_defaults_to_python(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    old_path = tmp_path / "old.checkpoint.npz"
    make_sampler(kernel="python").run(15, maxiter=2, checkpoint_path=path)
    with np.load(path) as data:
        values = {name: data[name] for name in data.files}
    config = json.loads(str(values["config_json"].item()))
    config.pop("kernel", None)
    values["config_json"] = np.asarray(json.dumps(config, sort_keys=True))
    np.savez(old_path, **values)

    result = make_sampler(kernel="python").resume(old_path, maxiter=3)

    assert result.metadata["kernel"] == "python"


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
    make_sampler(sample="rwalk", kernel="jax", walks=3, replacement_chains=2).run(
        16, maxiter=1, dlogz=0.0, checkpoint_path=path
    )

    _, config = load_checkpoint_npz(path)

    assert config["replacement_chains"] == 2


def test_resume_rejects_replacement_chains_mismatch(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(sample="rwalk", kernel="jax", walks=3, replacement_chains=2).run(
        17, maxiter=1, dlogz=0.0, checkpoint_path=path
    )

    with pytest.raises(ValueError, match="replacement_chains"):
        make_sampler(sample="rwalk", kernel="jax", walks=3).resume(path, maxiter=2)


def test_old_checkpoint_missing_replacement_chains_defaults_to_one(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(sample="rwalk", walks=3).run(
        18, maxiter=1, dlogz=0.0, checkpoint_path=path
    )
    _rewrite_checkpoint_config(path, lambda config: config.pop("replacement_chains"))

    result = make_sampler(sample="rwalk", walks=3).resume(path, maxiter=2)

    assert result.metadata["replacement_chains"] == 1


def test_checkpoint_config_includes_bound_settings(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(bound="single", rwalk_seed="live").run(
        19, maxiter=1, dlogz=0.0, checkpoint_path=path
    )

    _, config = load_checkpoint_npz(path)
    assert config["bound"] == "single"
    assert config["bound_enlargement"] == 1.25
    assert config["bound_update_interval"] == 1
    assert config["bound_jitter"] == 1e-6
    assert config["bound_max_draws"] is None
    assert config["bound_rebuild_on_failure"] is False
    assert config["bound_failure_rebuild_threshold"] == 1
    assert config["multi_bound_max_ellipsoids"] == 32
    assert config["multi_bound_min_points"] is None
    assert config["multi_bound_split_threshold"] == 0.9
    assert config["multi_bound_overlap_correction"] is True
    assert config["rwalk_seed"] == "live"


def test_checkpoint_config_includes_multi_bound_settings(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(
        bound="multi",
        sample="bound",
        multi_bound_max_ellipsoids=4,
        multi_bound_min_points=8,
        multi_bound_split_threshold=0.95,
        multi_bound_overlap_correction=True,
    ).run(20, maxiter=1, dlogz=0.0, checkpoint_path=path)

    _, config = load_checkpoint_npz(path)
    assert config["bound"] == "multi"
    assert config["multi_bound_max_ellipsoids"] == 4
    assert config["multi_bound_min_points"] == 8
    assert config["multi_bound_split_threshold"] == 0.95
    assert config["multi_bound_overlap_correction"] is True


def test_checkpoint_config_validates_jax_vectorized(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(sample="rwalk", kernel="jax", jax_vectorized=False).run(
        101, maxiter=1, dlogz=0.0, checkpoint_path=path
    )

    with pytest.raises(ValueError, match="jax_vectorized"):
        make_sampler(sample="rwalk", kernel="jax", jax_vectorized=True).resume(
            path, maxiter=2
        )


def test_checkpoint_config_validates_bound_rebuild_policy(tmp_path):
    path = tmp_path / "run.checkpoint.npz"
    make_sampler(
        bound="single",
        bound_rebuild_on_failure=True,
        bound_failure_rebuild_threshold=2,
    ).run(102, maxiter=1, dlogz=0.0, checkpoint_path=path)

    _, config = load_checkpoint_npz(path)
    assert config["bound_rebuild_on_failure"] is True
    assert config["bound_failure_rebuild_threshold"] == 2
    with pytest.raises(ValueError, match="bound_rebuild_on_failure"):
        make_sampler(bound="single", bound_rebuild_on_failure=False).resume(
            path, maxiter=2
        )
    with pytest.raises(ValueError, match="bound_failure_rebuild_threshold"):
        make_sampler(
            bound="single",
            bound_rebuild_on_failure=True,
            bound_failure_rebuild_threshold=1,
        ).resume(path, maxiter=2)
