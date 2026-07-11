from __future__ import annotations

import math

import jax.numpy as jnp
import pytest
from jax import random

import tinyns.run as run_mod
from tinyns.run import run_static_nested


def test_constant_likelihood_unit_cube_logz_close_to_zero() -> None:
    result = run_static_nested(
        random.PRNGKey(0),
        lambda theta: 0.0,
        lambda u: u,
        ndim=2,
        nlive=50,
        dlogz=0.01,
        maxiter=500,
    )

    assert abs(result.logz) < 0.05
    assert jnp.isfinite(result.logz)
    assert result.samples.shape[1:] == (2,)


def test_gaussian_likelihood_uniform_prior_logz_close_to_inverse_width() -> None:
    def loglike(theta):
        return float(-0.5 * theta[0] ** 2 - 0.5 * math.log(2.0 * math.pi))

    def prior_transform(u):
        return 20.0 * u - 10.0

    result = run_static_nested(
        random.PRNGKey(1),
        loglike,
        prior_transform,
        ndim=1,
        nlive=100,
        dlogz=0.05,
        maxiter=2_000,
    )

    assert abs(result.logz - (-math.log(20.0))) < 0.5
    assert jnp.isfinite(result.logz)


def test_result_shapes_finite_logz_and_equal_resampling() -> None:
    result = run_static_nested(
        random.PRNGKey(2),
        lambda theta: float(-jnp.sum(theta**2)),
        lambda u: 2.0 * u - 1.0,
        ndim=3,
        nlive=40,
        dlogz=0.1,
        maxiter=500,
    )

    assert result.samples.shape == result.samples_u.shape
    assert result.samples.shape[1:] == (3,)
    assert result.logl.shape == (result.samples.shape[0],)
    assert result.logwt.shape == (result.samples.shape[0],)
    assert jnp.isfinite(result.logz)
    assert result.resample_equal(random.PRNGKey(3), n=10).shape == (10, 3)


def test_static_nested_result_counts_match_metadata() -> None:
    result = run_static_nested(
        random.PRNGKey(23),
        lambda theta: float(-jnp.sum(theta**2)),
        lambda u: u,
        ndim=2,
        nlive=12,
        dlogz=0.0,
        maxiter=5,
    )

    assert result.samples_u.shape[0] == (
        result.metadata["ndead"] + result.metadata["nlive_final"]
    )
    assert result.logwt.shape[0] == result.samples.shape[0]


def test_static_nested_maxiter_zero_raises() -> None:
    with pytest.raises(ValueError, match="maxiter must be a positive integer"):
        run_static_nested(
            random.PRNGKey(24),
            lambda theta: float(-jnp.sum(theta**2)),
            lambda u: u,
            ndim=2,
            nlive=7,
            maxiter=0,
        )


def test_failure_to_replace_returns_result_with_live_contribution() -> None:
    result = run_static_nested(
        random.PRNGKey(4),
        lambda theta: float(theta[0]),
        lambda u: u,
        ndim=1,
        nlive=3,
        dlogz=0.0,
        maxiter=10,
        max_attempts=1,
    )

    assert result.success is False
    assert "max_attempts=1" in result.message
    assert result.ncall > result.nlive
    assert result.samples.shape[1:] == (1,)
    assert result.logwt.shape == (result.samples.shape[0],)
    assert jnp.isfinite(result.logz)


def test_scalar_prior_transform_for_one_dimension_keeps_matrix_shape() -> None:
    result = run_static_nested(
        random.PRNGKey(5),
        lambda theta: float(-(theta[0] ** 2)),
        lambda u: u[0],
        ndim=1,
        nlive=5,
        maxiter=2,
    )

    assert result.samples_u.ndim == 2 and result.samples_u.shape[1] == 1
    assert result.samples.ndim == 2 and result.samples.shape[1] == 1
    assert result.logl.ndim == 1
    assert result.logwt.ndim == 1
    assert result.metadata["replacement_failures"] == 0
    assert result.metadata["nlive_final"] == result.nlive
    assert result.metadata["nposterior"] == result.metadata["niter"] + result.nlive


def test_static_nested_rwalk_gaussian_returns_finite_logz() -> None:
    def loglike(theta):
        return float(-0.5 * theta[0] ** 2 - 0.5 * math.log(2.0 * math.pi))

    def prior_transform(u):
        return 20.0 * u - 10.0

    result = run_static_nested(
        random.PRNGKey(6),
        loglike,
        prior_transform,
        ndim=1,
        nlive=40,
        dlogz=0.1,
        maxiter=300,
        sample="rwalk",
        walks=5,
        step_scale=0.2,
    )

    assert jnp.isfinite(result.logz)
    assert result.metadata["sample"] == "rwalk"
    assert result.metadata["walks"] == 5
    assert result.metadata["step_scale"] == 0.2


def test_replacement_stats_metadata_after_normal_run() -> None:
    result = run_static_nested(
        random.PRNGKey(7),
        lambda theta: float(-jnp.sum(theta**2)),
        lambda u: u,
        ndim=2,
        nlive=10,
        dlogz=0.1,
        maxiter=20,
    )

    metadata = result.metadata
    assert set(
        [
            "replacement_ncall",
            "replacement_failures",
            "mean_replacement_ncall",
            "max_replacement_ncall",
            "replacement_acceptance_proxy",
            "niter",
            "ndead",
            "nlive_final",
            "nposterior",
        ]
    ).issubset(metadata)
    assert len(metadata["replacement_ncall"]) > 0
    assert metadata["niter"] == len(metadata["replacement_ncall"])
    assert metadata["ndead"] == metadata["niter"]
    assert metadata["nlive_final"] == result.nlive
    assert metadata["nposterior"] == result.logwt.size
    assert metadata["replacement_failures"] == 0
    assert metadata["mean_replacement_ncall"] > 0.0
    assert metadata["max_replacement_ncall"] >= 1
    assert (
        metadata["replacement_acceptance_proxy"]
        == 1.0 / metadata["mean_replacement_ncall"]
    )


def test_insertion_indices_metadata_after_normal_run() -> None:
    result = run_static_nested(
        random.PRNGKey(70),
        lambda theta: float(-jnp.sum(theta**2)),
        lambda u: u,
        ndim=2,
        nlive=10,
        dlogz=0.1,
        maxiter=20,
    )

    metadata = result.metadata
    insertion_indices = metadata["insertion_indices"]
    assert metadata["insertion_index_nslots"] == result.nlive
    assert metadata["insertion_index_nlive"] == result.nlive - 1
    assert insertion_indices.shape == (len(metadata["replacement_ncall"]),)
    assert insertion_indices.size > 0
    assert bool(jnp.all(insertion_indices >= 0))
    assert bool(jnp.all(insertion_indices < metadata["insertion_index_nslots"]))
    assert bool(jnp.all(insertion_indices <= metadata["insertion_index_nlive"]))


def test_failure_to_replace_increments_replacement_failures() -> None:
    result = run_static_nested(
        random.PRNGKey(8),
        lambda theta: float(theta[0]),
        lambda u: u,
        ndim=1,
        nlive=3,
        dlogz=0.0,
        maxiter=10,
        max_attempts=1,
    )

    assert result.success is False
    assert result.metadata["replacement_failures"] == 1
    assert result.metadata["replacement_ncall"][-1] == 1


def test_vectorized_prior_constant_likelihood_returns_finite_logz() -> None:
    result = run_static_nested(
        random.PRNGKey(9),
        lambda theta_batch: jnp.zeros((theta_batch.shape[0],)),
        lambda u_batch: u_batch,
        ndim=2,
        nlive=20,
        dlogz=0.1,
        maxiter=50,
        vectorized=True,
        batch_size=8,
    )

    assert jnp.isfinite(result.logz)
    assert result.samples.shape[1:] == (2,)
    assert all(ncall == 8 for ncall in result.metadata["replacement_ncall"])


def test_vectorized_prior_1d_gaussian_returns_finite_logz() -> None:
    def loglike(theta_batch):
        theta = theta_batch[:, 0]
        return -0.5 * theta**2 - 0.5 * math.log(2.0 * math.pi)

    result = run_static_nested(
        random.PRNGKey(10),
        loglike,
        lambda u_batch: 20.0 * u_batch - 10.0,
        ndim=1,
        nlive=30,
        dlogz=0.2,
        maxiter=100,
        vectorized=True,
        batch_size=6,
    )

    assert jnp.isfinite(result.logz)
    assert result.samples.shape[1:] == (1,)
    assert result.metadata["batch_size"] == 6


def test_vectorized_loglike_correct_initial_shape_passes() -> None:
    result = run_static_nested(
        random.PRNGKey(11),
        lambda theta_batch: -jnp.sum(theta_batch**2, axis=1),
        lambda u_batch: u_batch,
        ndim=2,
        nlive=7,
        maxiter=2,
        vectorized=True,
    )

    assert result.logl.ndim == 1
    assert math.isfinite(result.logz)


def test_vectorized_loglike_scalar_initial_shape_raises() -> None:
    with pytest.raises(ValueError, match="one value per live point"):
        run_static_nested(
            random.PRNGKey(12),
            lambda theta_batch: 0.0,
            lambda u_batch: u_batch,
            ndim=2,
            nlive=7,
            maxiter=1,
            vectorized=True,
        )


def test_vectorized_loglike_wrong_initial_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"expected shape \(7,\), got \(6,\)"):
        run_static_nested(
            random.PRNGKey(13),
            lambda theta_batch: jnp.zeros((theta_batch.shape[0] - 1,)),
            lambda u_batch: u_batch,
            ndim=2,
            nlive=7,
            maxiter=1,
            vectorized=True,
        )


def test_progress_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="progress_interval"):
        run_static_nested(
            random.PRNGKey(20),
            lambda theta: 0.0,
            lambda u: u,
            ndim=1,
            nlive=5,
            maxiter=1,
            progress_interval=0,
        )


def test_callback_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="callback_interval"):
        run_static_nested(
            random.PRNGKey(21),
            lambda theta: 0.0,
            lambda u: u,
            ndim=1,
            nlive=5,
            maxiter=1,
            callback_interval=0,
        )


def test_callback_must_be_callable() -> None:
    with pytest.raises(TypeError, match="callback"):
        run_static_nested(
            random.PRNGKey(22),
            lambda theta: 0.0,
            lambda u: u,
            ndim=1,
            nlive=5,
            maxiter=1,
            callback="not callable",
        )


def test_callback_is_called_during_short_run() -> None:
    states = []

    result = run_static_nested(
        random.PRNGKey(23),
        lambda theta: 0.0,
        lambda u: u,
        ndim=1,
        nlive=5,
        maxiter=3,
        callback=states.append,
        callback_interval=1,
    )

    assert jnp.isfinite(result.logz)
    assert states
    assert {"iter", "logz", "dlogz", "ncall", "sample"}.issubset(states[0])


def test_callback_can_stop_run_gracefully() -> None:
    def callback(state):
        return False if state["iter"] >= 2 else None

    result = run_static_nested(
        random.PRNGKey(24),
        lambda theta: 0.0,
        lambda u: u,
        ndim=1,
        nlive=5,
        maxiter=10,
        callback=callback,
        callback_interval=1,
    )

    assert result.success is False
    assert result.message == "stopped by callback"
    assert result.metadata["stopped_by_callback"] is True
    assert jnp.isfinite(result.logz)
    assert result.samples.shape[0] > 0


def test_progress_true_does_not_crash(capsys) -> None:
    run_static_nested(
        random.PRNGKey(25),
        lambda theta: 0.0,
        lambda u: u,
        ndim=1,
        nlive=5,
        maxiter=2,
        progress=True,
        progress_interval=1,
    )

    captured = capsys.readouterr()
    assert "iter=" in captured.out
    assert "logz=" in captured.out
    assert "sample=" in captured.out
    assert "\x1b" not in captured.out
    assert "[K" not in captured.out


def test_format_progress_line_contains_core_fields() -> None:
    from tinyns.run import _format_progress_line

    line = _format_progress_line(
        {
            "iter": 1,
            "logz": -5.0,
            "dlogz": 0.1,
            "ncall": 10,
            "logl_min": -1.0,
            "logl_live_max": 2.0,
            "replacement_mean_ncall_so_far": 3.0,
            "sample": "slice",
        }
    )

    assert isinstance(line, str)
    assert "iter=" in line
    assert "logz=" in line
    assert "dlogz=" in line
    assert "repl_batches=" in line
    assert "repl_chains=" in line


def test_progress_printer_pads_shorter_final_line(capsys) -> None:
    from tinyns.run import _ProgressPrinter

    printer = _ProgressPrinter()
    printer.print("sample=longer-name", final=False)
    printer.print("sample=x", final=True)

    captured = capsys.readouterr()
    assert "sample=x" in captured.out
    padded_short_line = "sample=x" + " " * (len("sample=longer-name") - len("sample=x"))
    assert padded_short_line in captured.out
    assert "\x1b" not in captured.out
    assert "[K" not in captured.out


def _jax_loglike(theta):
    return -0.5 * jnp.sum(((theta - 0.5) / 0.1) ** 2)


def _jax_prior_transform(u):
    return u


def _standard_gaussian_2d_loglike(theta):
    return -0.5 * jnp.sum(theta**2) - math.log(2.0 * math.pi)


def _wide_box_prior_transform(u):
    return 10.0 * u - 5.0


def test_nested_sampler_rwalk_jax_runs_and_records_kernel() -> None:
    from tinyns import NestedSampler

    sampler = NestedSampler(
        _jax_loglike,
        _jax_prior_transform,
        ndim=2,
        nlive=25,
        sample="rwalk",
        kernel="jax",
        walks=5,
        step_scale=0.05,
    )
    result = sampler.run(random.PRNGKey(0), dlogz=10.0)

    assert result.success is True
    assert math.isfinite(result.logz)
    assert result.metadata["kernel"] == "jax"
    assert result.metadata["jax_block_size"] == 1
    assert result.metadata["jax_block_mode"] is False
    assert result.metadata["jax_block_impl"] is None
    assert result.metadata["fused_bound_rwalk_impl"] is None


def test_nested_sampler_rwalk_jax_block_size_one_matches_existing_path() -> None:
    base_kwargs = dict(
        sample="rwalk",
        kernel="jax",
        walks=3,
        step_scale=0.05,
        maxiter=5,
        max_attempts=60,
    )
    existing = run_static_nested(
        random.PRNGKey(10),
        _jax_loglike,
        _jax_prior_transform,
        2,
        15,
        **base_kwargs,
    )
    block_one = run_static_nested(
        random.PRNGKey(10),
        _jax_loglike,
        _jax_prior_transform,
        2,
        15,
        jax_block_size=1,
        **base_kwargs,
    )

    assert jnp.allclose(block_one.samples_u, existing.samples_u)
    assert jnp.allclose(block_one.logl, existing.logl)
    assert jnp.allclose(block_one.logwt, existing.logwt)
    assert block_one.metadata["jax_block_mode"] is False


def test_nested_sampler_rwalk_jax_block_size_five_runs_and_shapes() -> None:
    result = run_static_nested(
        random.PRNGKey(11),
        _jax_loglike,
        _jax_prior_transform,
        2,
        20,
        sample="rwalk",
        kernel="jax",
        walks=3,
        step_scale=0.05,
        maxiter=10,
        max_attempts=60,
        jax_block_size=5,
    )

    assert result.success is False
    assert math.isfinite(result.logz)
    assert result.samples_u.shape == (result.metadata["nposterior"], 2)
    assert result.samples.shape == (result.metadata["nposterior"], 2)
    assert result.logl.shape == (result.metadata["nposterior"],)
    assert result.logwt.shape == (result.metadata["nposterior"],)
    assert len(result.metadata["replacement_ncall"]) == result.metadata["niter"]
    assert result.metadata["insertion_indices"].shape == (result.metadata["niter"],)
    assert result.metadata["jax_block_size"] == 5
    assert result.metadata["jax_block_mode"] is True
    assert result.metadata["jax_block_impl"] == "lax-scan-unbounded"
    assert result.metadata["jax_block_cached"] is True
    assert result.metadata["jax_block_kernel"] == "fixed-rwalk-cached"
    assert result.metadata["total_rwalk_proposals"] == sum(
        result.metadata["replacement_ncall"]
    )
    assert (
        0
        <= result.metadata["accepted_rwalk_moves"]
        <= result.metadata["total_rwalk_proposals"]
    )
    assert 0.0 <= result.metadata["rwalk_acceptance"] <= 1.0


def test_recommended_rwalk_jax_isotropic_cached_block_b32_records_metadata() -> None:
    from tinyns import NestedSampler

    sampler = NestedSampler(
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        nlive=50,
        sample="rwalk",
        kernel="jax",
        walks=5,
        replacement_chains=1,
        rwalk_proposal="isotropic",
        jax_block_size=32,
    )
    result = sampler.run(random.PRNGKey(112), dlogz=0.5, maxiter=300)

    assert result.success is True
    assert result.metadata["sample"] == "rwalk"
    assert result.metadata["kernel"] == "jax"
    assert result.metadata["rwalk_proposal"] == "isotropic"
    assert result.metadata["jax_block_mode"] is True
    assert result.metadata["jax_block_cached"] is True
    assert result.metadata["jax_block_kernel"] == "fixed-rwalk-cached"
    assert result.metadata["jax_block_size"] == 32
    assert result.metadata["jax_block_bound_fixed"] is False
    assert result.metadata["replacement_failures"] == 0
    assert math.isfinite(result.logz)
    assert result.ncall > 0
    assert result.metadata["niter"] > 0
    assert result.metadata.get("rwalk_adaptive_step_scale") is False
    assert "rwalk_adaptation_updates" not in result.metadata


def test_nested_sampler_rwalk_jax_cached_b32_block_close_to_non_block() -> None:
    base_kwargs = dict(
        sample="rwalk",
        kernel="jax",
        walks=5,
        replacement_chains=1,
        nlive=50,
        dlogz=0.5,
        maxiter=300,
        rwalk_proposal="isotropic",
    )
    non_block = run_static_nested(
        random.PRNGKey(113),
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        jax_block_size=1,
        **base_kwargs,
    )
    block = run_static_nested(
        random.PRNGKey(113),
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        jax_block_size=32,
        **base_kwargs,
    )

    assert math.isfinite(non_block.logz)
    assert math.isfinite(block.logz)
    assert non_block.metadata["replacement_failures"] == 0
    assert block.metadata["replacement_failures"] == 0
    tolerance = max(0.5, 3.0 * max(float(block.logzerr), float(non_block.logzerr)))
    assert abs(float(block.logz) - float(non_block.logz)) < tolerance
    assert block.metadata["jax_block_cached"] is True
    assert block.metadata["jax_block_kernel"] == "fixed-rwalk-cached"


def test_nested_sampler_rwalk_jax_cached_block_ring2d_no_failures() -> None:
    from validation.targets import get_target

    target = get_target("ring2d")
    result = run_static_nested(
        random.PRNGKey(114),
        target.loglike,
        target.prior_transform,
        target.ndim,
        30,
        sample="rwalk",
        kernel="jax",
        walks=5,
        replacement_chains=1,
        rwalk_proposal="isotropic",
        dlogz=2.0,
        maxiter=96,
        max_attempts=200,
        jax_block_size=16,
    )

    assert math.isfinite(result.logz)
    assert result.metadata["replacement_failures"] == 0
    assert result.ncall > 0
    assert result.metadata["jax_block_cached"] is True
    assert result.metadata["jax_block_kernel"] == "fixed-rwalk-cached"


def test_schedule_with_block_size_gt_one_raises() -> None:
    from tinyns import NestedSampler

    with pytest.raises(
        ValueError, match="replacement_chain_schedule is not supported"
    ):
        NestedSampler(
            _jax_loglike,
            _jax_prior_transform,
            ndim=2,
            nlive=20,
            sample="rwalk",
            kernel="jax",
            walks=3,
            replacement_chain_schedule=(1, 2, 4),
            jax_block_size=4,
        )

    with pytest.raises(
        ValueError, match="replacement_chain_schedule is not supported"
    ):
        run_static_nested(
            random.PRNGKey(14),
            _jax_loglike,
            _jax_prior_transform,
            2,
            20,
            sample="rwalk",
            kernel="jax",
            walks=3,
            max_attempts=24,
            replacement_chain_schedule=(1, 2, 4),
            jax_block_size=4,
        )


def test_live_cov_proposal_rejected() -> None:
    from tinyns import NestedSampler

    with pytest.raises(ValueError, match="live-cov"):
        NestedSampler(
            _jax_loglike,
            _jax_prior_transform,
            ndim=2,
            nlive=20,
            sample="rwalk",
            kernel="jax",
            rwalk_proposal="live-cov",
        )

    with pytest.raises(ValueError, match="live-cov"):
        run_static_nested(
            random.PRNGKey(15),
            _jax_loglike,
            _jax_prior_transform,
            2,
            20,
            sample="rwalk",
            kernel="jax",
            rwalk_proposal="live-cov",
        )


def test_static_nested_multi_bound_fused_rwalk_chain_telemetry_regression() -> None:
    result = run_static_nested(
        random.PRNGKey(2102),
        lambda theta: -0.5 * jnp.sum(theta**2),
        lambda u: 2.0 * u - 1.0,
        ndim=2,
        nlive=20,
        sample="rwalk",
        kernel="jax",
        bound="multi",
        rwalk_seed="bound",
        bound_seed_kernel="jax",
        fused_bound_rwalk=True,
        walks=5,
        replacement_chains=1,
        batch_size=128,
        multi_bound_max_ellipsoids=4,
        multi_bound_min_points=8,
        maxiter=5,
        dlogz=10.0,
    )

    metadata = result.metadata
    assert metadata["mean_replacement_chains_used"] == pytest.approx(1.0)
    assert metadata["mean_bound_seed_calls"] == pytest.approx(128.0)
    assert metadata["mean_rwalk_kernel_calls"] == pytest.approx(5.0)
    assert metadata["total_rwalk_proposals"] == pytest.approx(
        metadata["mean_rwalk_kernel_calls"] * metadata["niter"]
    )
    assert metadata["total_rwalk_proposals"] < sum(metadata["replacement_ncall"])
    assert 0 <= metadata["accepted_rwalk_moves"] <= metadata["total_rwalk_proposals"]


def test_static_nested_bound_seed_kernel_jax_invalid_combinations_raise() -> None:
    with pytest.raises(NotImplementedError, match="bound_seed_kernel='jax'"):
        run_static_nested(
            random.PRNGKey(111),
            lambda theta: -0.5 * jnp.sum(theta**2),
            lambda u: u,
            ndim=2,
            nlive=10,
            sample="prior",
            bound_seed_kernel="jax",
        )

    with pytest.raises(ValueError, match="bound_seed_kernel"):
        run_static_nested(
            random.PRNGKey(112),
            lambda theta: -0.5 * jnp.sum(theta**2),
            lambda u: u,
            ndim=2,
            nlive=10,
            bound_seed_kernel="bad",
        )


def test_static_nested_unbounded_rwalk_metadata_unchanged_shape() -> None:
    result = run_static_nested(
        random.PRNGKey(107),
        lambda theta: -0.5 * jnp.sum(theta**2),
        lambda u: 2.0 * u - 1.0,
        ndim=2,
        nlive=20,
        sample="rwalk",
        kernel="jax",
        bound="none",
        maxiter=2,
        dlogz=0.0,
    )

    assert result.metadata["bound"] == "none"
    assert result.metadata["bounded_rwalk"] is False
    assert result.metadata["bound_build_count"] == 0
    assert result.metadata["bound_build_time_total"] == 0.0
    assert result.metadata["bound_build_time_mean"] == 0.0
    assert result.metadata["bound_build_time_max"] == 0.0
    assert result.metadata["bound_log_volume_final"] is None
    assert result.metadata["bound_log_volume_mean"] is None
    assert result.metadata["bound_log_volume_min"] is None
    assert result.metadata["bound_log_volume_max"] is None
    assert result.metadata["bound_nellipsoids_mean"] is None
    assert result.metadata["bound_nellipsoids_max"] is None
    assert result.metadata["bound_nellipsoids_final"] is None
    assert result.metadata["mean_bound_seed_calls"] is None


def test_static_nested_invalid_multi_bound_options_raise() -> None:
    with pytest.raises(ValueError, match="multi_bound_max_ellipsoids"):
        run_static_nested(
            random.PRNGKey(32),
            lambda theta: 0.0,
            lambda u: u,
            ndim=2,
            nlive=10,
            bound="multi",
            multi_bound_max_ellipsoids=0,
        )


def test_jax_vectorized_metadata_default_false() -> None:
    result = run_static_nested(
        random.PRNGKey(4400),
        lambda theta: 0.0,
        lambda u: u,
        ndim=2,
        nlive=12,
        dlogz=10.0,
        maxiter=1,
    )

    assert result.metadata["jax_vectorized"] is False


def test_jax_bounded_seed_draw_supports_vectorized_functions() -> None:
    def prior_batch(u):
        return 2.0 * u - 1.0

    def loglike_batch(theta):
        return -jnp.sum(theta**2, axis=1)

    result = run_static_nested(
        random.PRNGKey(4401),
        loglike_batch,
        prior_batch,
        ndim=2,
        nlive=24,
        dlogz=10.0,
        maxiter=2,
        sample="rwalk",
        kernel="jax",
        bound="single",
        rwalk_seed="bound",
        bound_seed_kernel="jax",
        walks=2,
        batch_size=8,
        max_attempts=32,
        jax_vectorized=True,
    )

    assert result.metadata["jax_vectorized"] is True
    assert result.metadata["bound_seed_kernel"] == "jax"
    assert jnp.isfinite(result.logz)


def test_jax_block_rwalk_failure_propagates_replacement_failure() -> None:
    result = run_static_nested(
        random.PRNGKey(1234),
        _jax_loglike,
        _jax_prior_transform,
        2,
        12,
        sample="rwalk",
        kernel="jax",
        walks=1,
        min_accepts=2,
        replacement_chains=1,
        max_attempts=1,
        maxiter=4,
        dlogz=0.0,
        jax_block_size=4,
    )

    assert result.success is False
    assert result.metadata["replacement_failures"] == 1
    assert "max_attempts=1" in result.message


def test_jax_block_partial_failure_after_convergence_reports_success(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_mod,
        "_make_static_jax_rwalk_block_kernel",
        _partial_failure_block_kernel_20tuple(
            accepted_prefix=1, replacement_ncall=(1, 1, 1, 1)
        ),
    )

    result = run_static_nested(
        random.PRNGKey(1236),
        lambda theta: 0.0,
        lambda u: u,
        2,
        12,
        sample="rwalk",
        kernel="jax",
        maxiter=4,
        dlogz=10.0,
        jax_block_size=4,
    )

    assert result.success is True
    assert "converged" in result.message
    assert result.metadata["replacement_failures"] == 1
    assert result.metadata["terminated_after_partial_block_failure"] is True
    assert result.metadata["partial_block_failure_offset"] == 1
    assert (
        result.metadata["partial_block_failure_delta_logz"] < result.metadata["dlogz"]
    )
    assert result.metadata["final_delta_logz"] < result.metadata["dlogz"]


def test_jax_block_partial_failure_before_convergence_remains_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_mod,
        "_make_static_jax_rwalk_block_kernel",
        _partial_failure_block_kernel_20tuple(
            accepted_prefix=1, replacement_ncall=(1, 1, 1, 1)
        ),
    )

    result = run_static_nested(
        random.PRNGKey(1237),
        lambda theta: 0.0,
        lambda u: u,
        2,
        12,
        sample="rwalk",
        kernel="jax",
        walks=1,
        replacement_chains=1,
        # max_attempts == walks * replacement_chains disables the rescue ladder,
        # preserving the partial-failure-remains-failure behavior.
        max_attempts=1,
        maxiter=4,
        dlogz=0.0,
        jax_block_size=4,
    )

    assert result.success is False
    assert "max_attempts" in result.message
    assert result.metadata["replacement_failures"] == 1
    assert result.metadata["terminated_after_partial_block_failure"] is False
    assert result.metadata["partial_block_failure_offset"] == 1
    assert (
        result.metadata["partial_block_failure_delta_logz"] >= result.metadata["dlogz"]
    )
    assert result.metadata["final_delta_logz"] >= result.metadata["dlogz"]


def test_jax_block_rwalk_rescue_succeeds_after_normal_failure(monkeypatch) -> None:
    def make_kernel(*_args, **_kwargs):
        def kernel(
            key,
            live_u,
            live_theta,
            live_logl,
            logz_dead,
            start_iteration,
            nlive,
            *_rest,
        ):
            block_size = 2
            worst = int(jnp.argmin(live_logl))
            dead_u = jnp.repeat(live_u[worst][None, :], block_size, axis=0)
            dead_theta = jnp.repeat(live_theta[worst][None, :], block_size, axis=0)
            dead_logl = jnp.repeat(live_logl[worst][None], block_size, axis=0)
            offsets = jnp.arange(block_size)
            iterations = start_iteration + offsets
            logx_prev = -iterations / nlive
            logx_new = -(iterations + 1) / nlive
            dead_logwt = (
                logx_prev + jnp.log1p(-jnp.exp(logx_new - logx_prev)) + live_logl[worst]
            )
            return (
                key,
                live_u,
                live_theta,
                live_logl,
                dead_u,
                dead_theta,
                dead_logl,
                dead_logwt,
                jnp.full((block_size,), 2, dtype=jnp.int32),
                jnp.zeros((block_size,), dtype=jnp.int32),
                jnp.ones((block_size,), dtype=jnp.int32),
                jnp.ones((block_size,), dtype=jnp.int32),
                jnp.zeros((block_size,), dtype=bool),
                jnp.zeros((block_size,), dtype=jnp.int32),
                jnp.full((block_size,), 2, dtype=jnp.int32),
                dead_u,
                dead_theta,
                dead_logl,
                logz_dead,
                -(start_iteration + block_size) / nlive,
            )

        return kernel

    rescue_calls = {"count": 0}

    def rescue_draw(
        key, _loglike, _prior_transform, logl_min, live_u, _live_logl, ndim, **_kwargs
    ):
        rescue_calls["count"] += 1
        new_u = jnp.clip(live_u[0] + 0.01, 0.0, 1.0)
        if rescue_calls["count"] == 1:
            return (
                key,
                new_u,
                new_u,
                jnp.asarray(logl_min),
                3,
                False,
                {
                    "replacement_batches": 1,
                    "replacement_chains_used": 1,
                    "replacement_chain_usage_counts": {"1": 1},
                    "accepted_rwalk_moves": 1,
                    "total_rwalk_proposals": 3,
                },
            )
        return (
            key,
            new_u,
            new_u,
            jnp.asarray(logl_min + 1.0),
            5,
            True,
            {
                "replacement_batches": 1,
                "replacement_chains_used": 2,
                "replacement_chain_usage_counts": {"2": 1},
                "accepted_rwalk_moves": 2,
                "total_rwalk_proposals": 5,
            },
        )

    monkeypatch.setattr(run_mod, "_make_static_jax_rwalk_block_kernel", make_kernel)
    monkeypatch.setattr(run_mod, "draw_constrained_rwalk_jax", rescue_draw)

    result = run_static_nested(
        random.PRNGKey(2001),
        lambda theta: 0.0,
        lambda u: u,
        2,
        12,
        sample="rwalk",
        kernel="jax",
        walks=1,
        replacement_chains=1,
        max_attempts=2,
        maxiter=5,
        dlogz=10.0,
        jax_block_size=2,
    )

    assert result.success is True
    assert result.metadata["replacement_failures"] == 1
    assert result.metadata["replacement_rescue_used"] is True
    assert result.metadata["replacement_rescue_attempts"] == 2
    assert result.metadata["replacement_rescue_successes"] == 1
    assert result.metadata["replacement_rescue_failures"] == 0
    assert result.metadata["replacement_rescue_ncall"] == 8
    assert result.metadata["replacement_rescue_stage_counts"] == {
        "1": 1,
        "2": 1,
        "3": 0,
        "4": 0,
    }
    assert result.metadata["replacement_ncall"] == [10]
    assert result.metadata["replacement_chain_usage_counts"]["1"] == 1
    assert result.metadata["replacement_chain_usage_counts"]["2"] == 1
    assert result.metadata["accepted_rwalk_moves"] == 3
    assert result.metadata["total_rwalk_proposals"] == 10
    assert result.metadata["rwalk_acceptance"] == pytest.approx(0.3)
    # ncall accounts for both the failed offset's calls (2) and the full rescue
    # ladder's calls (stage 1: 3 + stage 2: 5 = 8): nlive + 2 + 8 == nlive + 10.
    failed_offset_calls = 2
    rescue_calls_total = 3 + 5
    assert result.ncall == result.nlive + failed_offset_calls + rescue_calls_total


def test_jax_block_rwalk_rescue_fails_before_convergence(monkeypatch) -> None:
    def rescue_draw(
        key, _loglike, _prior_transform, logl_min, live_u, _live_logl, ndim, **_kwargs
    ):
        return key, live_u[0], live_u[0], jnp.asarray(logl_min), 4, False

    monkeypatch.setattr(run_mod, "draw_constrained_rwalk_jax", rescue_draw)

    result = run_static_nested(
        random.PRNGKey(2002),
        _jax_loglike,
        _jax_prior_transform,
        2,
        12,
        sample="rwalk",
        kernel="jax",
        walks=1,
        min_accepts=3,
        replacement_chains=1,
        max_attempts=2,
        maxiter=4,
        dlogz=0.0,
        jax_block_size=4,
    )

    assert result.success is False
    assert "max_attempts" in result.message
    assert result.metadata["replacement_rescue_used"] is True
    assert result.metadata["replacement_rescue_failures"] == 1
    assert result.metadata["replacement_rescue_attempts"] == 4
    assert result.metadata["replacement_rescue_ncall"] == 16
    assert result.metadata["final_delta_logz"] >= result.metadata["dlogz"]


def _partial_failure_block_kernel_20tuple(
    *, accepted_prefix: int, replacement_ncall
):
    replacement_ncall = tuple(int(x) for x in replacement_ncall)
    block_size = len(replacement_ncall)

    def make_kernel(*_args, **_kwargs):
        def kernel(
            key,
            live_u,
            live_theta,
            live_logl,
            logz_dead,
            start_iteration,
            nlive,
            *_rest,
        ):
            worst = int(jnp.argmin(live_logl))
            dead_u = jnp.repeat(live_u[worst][None, :], block_size, axis=0)
            dead_theta = jnp.repeat(live_theta[worst][None, :], block_size, axis=0)
            dead_logl = jnp.repeat(live_logl[worst][None], block_size, axis=0)
            offsets = jnp.arange(block_size)
            iterations = start_iteration + offsets
            logx_prev = -iterations / nlive
            logx_new = -(iterations + 1) / nlive
            dead_logwt = (
                logx_prev + jnp.log1p(-jnp.exp(logx_new - logx_prev)) + live_logl[worst]
            )
            ncall_block = jnp.asarray(replacement_ncall, dtype=jnp.int32)
            accepted = offsets < accepted_prefix
            return (
                key,
                live_u,
                live_theta,
                live_logl,
                dead_u,
                dead_theta,
                dead_logl,
                dead_logwt,
                ncall_block,
                jnp.zeros((block_size,), dtype=jnp.int32),
                jnp.ones((block_size,), dtype=jnp.int32),
                jnp.ones((block_size,), dtype=jnp.int32),
                accepted,
                jnp.ones((block_size,), dtype=jnp.int32),
                ncall_block,
                dead_u,
                dead_theta,
                dead_logl,
                logz_dead,
                -(start_iteration + block_size) / nlive,
            )

        return kernel

    return make_kernel


def test_jax_block_ncall_counts_failed_offset_when_rescue_skipped(monkeypatch) -> None:
    # Prefix offsets 0, 1 succeed (3 + 3 calls); offset 2 fails (9 calls).
    monkeypatch.setattr(
        run_mod,
        "_make_static_jax_rwalk_block_kernel",
        _partial_failure_block_kernel_20tuple(
            accepted_prefix=2, replacement_ncall=(3, 3, 9, 3)
        ),
    )

    result = run_static_nested(
        random.PRNGKey(4242),
        lambda theta: 0.0,
        lambda u: u,
        2,
        12,
        sample="rwalk",
        kernel="jax",
        walks=1,
        replacement_chains=1,
        # max_attempts == walks * replacement_chains disables the rescue ladder.
        max_attempts=1,
        maxiter=8,
        dlogz=0.0,
        jax_block_size=4,
    )

    assert result.success is False
    assert result.metadata["replacement_failures"] == 1
    assert result.metadata["partial_block_failure_offset"] == 2
    assert result.metadata["replacement_rescue_used"] is False
    # nlive initial evals + successful prefix (3 + 3) + failed offset (9).
    successful_prefix_calls = 3 + 3
    failed_offset_calls = 9
    assert result.ncall == result.nlive + successful_prefix_calls + failed_offset_calls


def test_jax_bounded_block_rwalk_failure_returns_failed_result() -> None:
    result = run_static_nested(
        random.PRNGKey(1235),
        _jax_loglike,
        _jax_prior_transform,
        2,
        12,
        sample="rwalk",
        kernel="jax",
        bound="single",
        rwalk_seed="bound",
        bound_seed_kernel="jax",
        fused_bound_rwalk=True,
        walks=1,
        min_accepts=2,
        replacement_chains=1,
        max_attempts=1,
        maxiter=4,
        dlogz=0.0,
        jax_block_size=4,
        batch_size=2,
    )

    assert result.success is False
    assert result.metadata["replacement_failures"] > 0
    assert "bounded JAX rwalk" in result.message
    assert "max_attempts=1" in result.message
    assert result.metadata["jax_block_impl"] == "python-loop-fixed-bound"


def test_update_adaptive_step_scale_direction_and_clamps() -> None:
    from tinyns.run import _update_adaptive_step_scale

    assert _update_adaptive_step_scale(0.1, 0.05, 0.25, 0.1, 1e-4, 0.5) < 0.1
    assert _update_adaptive_step_scale(0.1, 0.75, 0.25, 0.1, 1e-4, 0.5) > 0.1
    assert _update_adaptive_step_scale(
        0.1, 0.25, 0.25, 0.1, 1e-4, 0.5
    ) == pytest.approx(0.1)
    assert _update_adaptive_step_scale(
        1e-4, 0.0, 1.0, 10.0, 1e-4, 0.5
    ) == pytest.approx(1e-4)
    assert _update_adaptive_step_scale(0.5, 1.0, 0.0, 10.0, 1e-4, 0.5) == pytest.approx(
        0.5
    )


def test_rwalk_adaptive_step_scale_uses_low_move_acceptance_to_shrink(
    monkeypatch,
) -> None:
    import tinyns.run as run_mod
    from tinyns import NestedSampler

    def fake_draw(
        key, loglike, prior_transform, logl_min, live_u, live_logl, ndim, **kwargs
    ):
        new_u = live_u[0]
        new_theta = prior_transform(new_u)
        info = {
            "replacement_batches": 1,
            "replacement_chains_used": 1,
            "replacement_chain_usage_counts": {"1": 1},
            "accepted_move_count": 1,
            "total_proposal_count": 20,
            "observed_rwalk_acceptance": 0.05,
        }
        return key, new_u, new_theta, float(loglike(new_theta)), 20, True, info

    monkeypatch.setattr(run_mod, "draw_constrained_rwalk_jax", fake_draw)
    sampler = NestedSampler(
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        nlive=16,
        sample="rwalk",
        kernel="jax",
        walks=5,
        step_scale=0.1,
        jax_block_size=1,
        rwalk_adaptive_step_scale=True,
        rwalk_target_accept=0.25,
    )
    result = sampler.run(random.PRNGKey(123), dlogz=10.0, maxiter=3)

    assert result.metadata["rwalk_effective_step_scale_final"] < 0.1
    assert result.metadata["rwalk_observed_accept_mean"] == pytest.approx(0.05)


def test_rwalk_adaptive_step_scale_uses_high_move_acceptance_to_grow(
    monkeypatch,
) -> None:
    import tinyns.run as run_mod
    from tinyns import NestedSampler

    def fake_draw(
        key, loglike, prior_transform, logl_min, live_u, live_logl, ndim, **kwargs
    ):
        new_u = live_u[0]
        new_theta = prior_transform(new_u)
        info = {
            "replacement_batches": 1,
            "replacement_chains_used": 1,
            "replacement_chain_usage_counts": {"1": 1},
            "accepted_move_count": 15,
            "total_proposal_count": 20,
            "observed_rwalk_acceptance": 0.75,
        }
        return key, new_u, new_theta, float(loglike(new_theta)), 20, True, info

    monkeypatch.setattr(run_mod, "draw_constrained_rwalk_jax", fake_draw)
    sampler = NestedSampler(
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        nlive=16,
        sample="rwalk",
        kernel="jax",
        walks=5,
        step_scale=0.1,
        jax_block_size=1,
        rwalk_adaptive_step_scale=True,
        rwalk_target_accept=0.25,
    )
    result = sampler.run(random.PRNGKey(124), dlogz=10.0, maxiter=3)

    assert result.metadata["rwalk_effective_step_scale_final"] > 0.1
    assert result.metadata["rwalk_observed_accept_mean"] == pytest.approx(0.75)


def test_nested_sampler_rwalk_jax_adaptive_step_scale_records_metadata() -> None:
    from tinyns import NestedSampler

    sampler = NestedSampler(
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        nlive=32,
        sample="rwalk",
        kernel="jax",
        walks=5,
        step_scale=0.05,
        jax_block_size=4,
        rwalk_adaptive_step_scale=True,
        rwalk_target_accept=0.25,
    )
    result = sampler.run(random.PRNGKey(314), dlogz=2.0, maxiter=120)

    assert result.success is True
    assert result.metadata["rwalk_adaptive_step_scale"] is True
    assert result.metadata["rwalk_target_accept"] == pytest.approx(0.25)
    assert math.isfinite(result.metadata["rwalk_effective_step_scale_final"])
    assert math.isfinite(result.metadata["rwalk_effective_step_scale_min_seen"])
    assert math.isfinite(result.metadata["rwalk_effective_step_scale_max_seen"])
    assert math.isfinite(result.metadata["rwalk_effective_step_scale_mean"])
    assert result.metadata["rwalk_adaptation_updates"] > 0
    assert math.isfinite(result.metadata["rwalk_observed_accept_mean"])
    assert result.metadata["rwalk_observed_accept_source"] == "move_acceptance"
