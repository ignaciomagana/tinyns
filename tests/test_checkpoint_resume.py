import math

import jax.numpy as jnp
import numpy as np
import pytest

from tinyns import NestedSampler
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
