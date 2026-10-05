from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import tinyns.run as run_mod
from tinyns.run import run_static_nested


def _sum_squares(theta):
    return -jnp.sum(theta**2)


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
        return -0.5 * theta[0] ** 2 - 0.5 * math.log(2.0 * math.pi)

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
        _sum_squares,
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
        _sum_squares,
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
            _sum_squares,
            lambda u: u,
            ndim=2,
            nlive=7,
            maxiter=0,
        )


def test_replacement_failure_stops_the_run_with_a_message() -> None:
    # No point is admissible above a NaN threshold: the first replacement
    # exhausts its batches and the run stops.
    walks = 10
    result = run_static_nested(
        random.PRNGKey(8),
        lambda theta: jnp.nan,
        lambda u: u,
        ndim=1,
        nlive=3,
        walks=walks,
        dlogz=0.0,
        maxiter=10,
        block_size=4,
        cluster_swap=False,
    )

    assert result.success is False
    assert result.message.startswith("replacement failed at iteration 1")
    assert result.metadata["replacement_failures"] == 1
    assert result.metadata["partial_block_failure_offset"] == 0
    assert result.metadata["niter"] == 0
    max_batches = 10_000 // walks
    assert result.metadata["replacement_batch_ncall"] == walks
    # the failed replacement's calls are counted
    assert result.nlive < result.ncall <= result.nlive + max_batches * walks
    assert result.samples.shape == (3, 1)
    assert result.logwt.shape == (3,)


def test_scalar_prior_transform_for_one_dimension_keeps_matrix_shape() -> None:
    result = run_static_nested(
        random.PRNGKey(5),
        lambda theta: -(theta[0] ** 2),
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


def test_replacement_stats_metadata_after_normal_run() -> None:
    result = run_static_nested(
        random.PRNGKey(7),
        _sum_squares,
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
        _sum_squares,
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


def _scripted_rwalk_kernel(new_logl, accepted=None):
    """Fake rwalk kernel factory: replacement ``key`` gets ``new_logl[key]``.

    The key is an int32 counter, advanced by one per replacement.
    """
    new_logl = jnp.asarray(new_logl, dtype=float)
    if accepted is None:
        accepted = [True] * int(new_logl.size)
    accepted = jnp.asarray(accepted)

    def make_rwalk_kernel(*_args, **_kwargs):
        def kernel(key, logl_min, live_u, _live_logl, _scale, _max_batches):
            new_u = jnp.full(live_u.shape[1:], (key + 3).astype(live_u.dtype) / 10)
            ok = accepted[key]
            return (
                key + 1,
                new_u,
                new_u,
                new_logl[key],
                jnp.asarray(1, dtype=jnp.int32),
                ok,
                ok.astype(jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
            )

        return kernel

    return make_rwalk_kernel


def _run_block_kernel(live_logl, block_size):
    live_logl = jnp.asarray(live_logl, dtype=float)
    live_u = jnp.tile(live_logl[:, None] / 100.0, (1, 2))
    run_mod._make_static_jax_rwalk_block_kernel.cache_clear()
    kernel = run_mod._make_static_jax_rwalk_block_kernel(
        lambda theta: theta[0], lambda u: u, 2, 1, 1, block_size
    )
    try:
        return kernel(
            jnp.asarray(0, dtype=jnp.int32),
            live_u,
            live_u,
            live_logl,
            jnp.asarray(-jnp.inf),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(live_logl.size, dtype=jnp.int32),
            jnp.asarray(0.1),
            jnp.asarray(1, dtype=jnp.int32),
        )
    finally:
        run_mod._make_static_jax_rwalk_block_kernel.cache_clear()


def test_block_insertion_ranks_match_bruteforce_reference(monkeypatch) -> None:
    """Insertion ranks equal the count of surviving live points at or below
    each replacement, for a scripted, varied rank sequence."""
    initial_live_logl = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # Each scripted likelihood is above the worst it replaces.
    scripted_new_logl = [0.5, 6.5, 1.5, 4.5, 100.0, 3.5]
    fake = _scripted_rwalk_kernel(scripted_new_logl)
    monkeypatch.setattr(run_mod, "_make_rwalk_jax_kernel_cached", fake)
    result = _run_block_kernel(initial_live_logl, len(scripted_new_logl))
    got = [int(x) for x in np.asarray(result[9])]

    live = list(initial_live_logl)
    expected = []
    for new_logl in scripted_new_logl:
        worst = min(range(len(live)), key=lambda j: live[j])
        others = live[:worst] + live[worst + 1 :]
        expected.append(sum(1 for value in others if value <= new_logl))
        live[worst] = new_logl

    assert got == expected
    assert len(set(got)) >= 4  # varied, non-zero middle ranks


def test_block_stops_scanning_after_first_failed_replacement(monkeypatch) -> None:
    monkeypatch.setattr(
        run_mod,
        "_make_rwalk_jax_kernel_cached",
        _scripted_rwalk_kernel([1.0, 2.0, 3.0, 4.0], [True, False, True, True]),
    )
    result = _run_block_kernel([0.0, 1.0], 4)

    assert int(result[0]) == 2  # the skipped iterations do not advance the key
    assert jnp.asarray(result[11]).tolist() == [True, False, False, False]
    assert jnp.asarray(result[8]).tolist() == [1, 1, 0, 0]  # ncall
    assert jnp.asarray(result[13]).tolist() == [1, 1, 0, 0]  # proposals
    # The live set holds the first replacement only.
    assert jnp.allclose(
        jnp.sort(jnp.asarray(result[1]), axis=0),
        jnp.asarray(((0.01, 0.01), (0.3, 0.3))),
    )
    assert jnp.asarray(result[3]).tolist() == [1.0, 1.0]


def _fake_block_kernel(*, accepted_prefix: int, replacement_ncall, moves=None):
    """Factory for a fake block kernel with the real output layout."""
    replacement_ncall = tuple(int(x) for x in replacement_ncall)
    block_size = len(replacement_ncall)
    moves = replacement_ncall if moves is None else tuple(moves)
    calls = []

    def make_kernel(*args, **_kwargs):
        def kernel(
            key,
            live_u,
            live_theta,
            live_logl,
            logz_dead,
            start_iteration,
            nlive,
            scale,
            *_rest,
        ):
            calls.append(float(scale))
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
                offsets < accepted_prefix,
                jnp.asarray(moves, dtype=jnp.int32),
                ncall_block,
                logz_dead,
                -(start_iteration + block_size) / nlive,
            )

        return kernel

    make_kernel.scales = calls
    return make_kernel


def test_block_partial_failure_after_convergence_reports_success(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_mod,
        "_make_static_jax_rwalk_block_kernel",
        _fake_block_kernel(accepted_prefix=1, replacement_ncall=(1, 1, 1, 1)),
    )

    result = run_static_nested(
        random.PRNGKey(1236),
        lambda theta: 0.0,
        lambda u: u,
        2,
        12,
        maxiter=4,
        dlogz=10.0,
        block_size=4,
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


def test_block_partial_failure_before_convergence_remains_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_mod,
        "_make_static_jax_rwalk_block_kernel",
        _fake_block_kernel(accepted_prefix=1, replacement_ncall=(1, 1, 1, 1)),
    )

    result = run_static_nested(
        random.PRNGKey(1237),
        lambda theta: 0.0,
        lambda u: u,
        2,
        12,
        walks=1,
        maxiter=4,
        dlogz=0.0,
        block_size=4,
    )

    assert result.success is False
    assert result.message.startswith("replacement failed at iteration 2")
    assert result.metadata["partial_block_failure_message"] == result.message
    assert result.metadata["replacement_failures"] == 1
    assert result.metadata["terminated_after_partial_block_failure"] is False
    assert result.metadata["partial_block_failure_offset"] == 1
    assert result.metadata["niter"] == 1
    assert (
        result.metadata["partial_block_failure_delta_logz"] >= result.metadata["dlogz"]
    )
    assert result.metadata["final_delta_logz"] >= result.metadata["dlogz"]


def test_block_ncall_counts_failed_offset(monkeypatch) -> None:
    # Prefix offsets 0, 1 succeed (3 + 3 calls); offset 2 fails (9 calls).
    monkeypatch.setattr(
        run_mod,
        "_make_static_jax_rwalk_block_kernel",
        _fake_block_kernel(accepted_prefix=2, replacement_ncall=(3, 3, 9, 3)),
    )

    result = run_static_nested(
        random.PRNGKey(4242),
        lambda theta: 0.0,
        lambda u: u,
        2,
        12,
        walks=1,
        maxiter=8,
        dlogz=0.0,
        block_size=4,
    )

    assert result.success is False
    assert result.metadata["replacement_failures"] == 1
    assert result.metadata["partial_block_failure_offset"] == 2
    assert result.metadata["replacement_ncall"] == [3, 3]
    # nlive initial evals + successful prefix (3 + 3) + failed offset (9).
    assert result.ncall == result.nlive + 3 + 3 + 9


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
    assert {"iter", "logz", "dlogz", "ncall", "walks", "calls_per_s"}.issubset(
        states[0]
    )


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
        block_size=1,
    )

    assert result.success is False
    assert result.message == "stopped by callback"
    assert result.metadata["stopped_by_callback"] is True
    assert result.metadata["niter"] == 2
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
    assert "repl_batches=" in captured.out
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
        }
    )

    assert isinstance(line, str)
    assert "iter=" in line
    assert "logz=" in line
    assert "dlogz=" in line
    assert "repl_ncall=3.0" in line
    assert "repl_batches=n/a" in line


def test_progress_printer_pads_shorter_final_line(capsys) -> None:
    from tinyns.run import _ProgressPrinter

    printer = _ProgressPrinter()
    printer.print("iter=longer-line", final=False)
    printer.print("iter=x", final=True)

    captured = capsys.readouterr()
    assert "iter=x" in captured.out
    padded_short_line = "iter=x" + " " * (len("iter=longer-line") - len("iter=x"))
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


def test_nested_sampler_block_size_one_runs() -> None:
    from tinyns import NestedSampler

    sampler = NestedSampler(
        _jax_loglike, _jax_prior_transform, ndim=2, nlive=25, walks=5, block_size=1
    )
    result = sampler.run(random.PRNGKey(0), dlogz=10.0)

    assert result.success is True
    assert math.isfinite(result.logz)
    assert result.metadata["block_size"] == 1
    # one scale update per block, so per iteration
    assert result.metadata["rwalk_adaptation_updates"] == result.metadata["niter"]


def test_block_size_five_runs_and_shapes() -> None:
    result = run_static_nested(
        random.PRNGKey(11),
        _jax_loglike,
        _jax_prior_transform,
        2,
        20,
        walks=3,
        maxiter=10,
        block_size=5,
    )

    assert result.success is False
    assert result.message == "maxiter=10 reached"
    assert math.isfinite(result.logz)
    assert result.samples_u.shape == (result.metadata["nposterior"], 2)
    assert result.samples.shape == (result.metadata["nposterior"], 2)
    assert result.logl.shape == (result.metadata["nposterior"],)
    assert result.logwt.shape == (result.metadata["nposterior"],)
    assert len(result.metadata["replacement_ncall"]) == result.metadata["niter"]
    assert result.metadata["insertion_indices"].shape == (result.metadata["niter"],)
    assert result.metadata["block_size"] == 5
    assert result.metadata["rwalk_adaptation_updates"] == 2
    # A single chain skips the likelihood for out-of-cube proposals.
    assert result.metadata["total_rwalk_proposals"] >= sum(
        result.metadata["replacement_ncall"]
    )
    assert (
        0
        <= result.metadata["accepted_rwalk_moves"]
        <= result.metadata["total_rwalk_proposals"]
    )
    assert 0.0 <= result.metadata["rwalk_acceptance"] <= 1.0


def test_default_block_run_records_metadata() -> None:
    from tinyns import NestedSampler

    sampler = NestedSampler(
        _standard_gaussian_2d_loglike, _wide_box_prior_transform, ndim=2, nlive=50
    )
    result = sampler.run(random.PRNGKey(112), dlogz=0.5, maxiter=300)

    metadata = result.metadata
    assert result.success is True
    assert metadata["block_size"] == 32
    assert metadata["walks"] == 25
    assert metadata["replacement_chains"] == 1
    assert metadata["replacement_batch_ncall"] == 25
    assert metadata["replacement_failures"] == 0
    assert math.isfinite(result.logz)
    assert result.ncall > 0
    assert metadata["niter"] > 0
    assert metadata["rwalk_scale_initial"] == 0.5
    assert metadata["rwalk_adaptation_updates"] == -(-metadata["niter"] // 32)
    for name in ("final", "min_seen", "max_seen", "mean"):
        assert math.isfinite(metadata[f"rwalk_scale_{name}"])
    assert math.isfinite(metadata["rwalk_observed_accept_mean"])
    assert metadata["wall_time_s"] > 0.0
    assert metadata["compile_s"] is not None


def test_block_size_one_and_32_agree_within_errors() -> None:
    kwargs = dict(walks=5, nlive=50, dlogz=0.5, maxiter=300)
    one = run_static_nested(
        random.PRNGKey(113),
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        block_size=1,
        **kwargs,
    )
    block = run_static_nested(
        random.PRNGKey(113),
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        block_size=32,
        **kwargs,
    )

    assert one.metadata["replacement_failures"] == 0
    assert block.metadata["replacement_failures"] == 0
    tolerance = max(0.5, 3.0 * max(float(block.logzerr), float(one.logzerr)))
    assert abs(float(block.logz) - float(one.logz)) < tolerance


def test_block_ring2d_no_failures() -> None:
    from validation.targets import get_target

    target = get_target("ring2d")
    result = run_static_nested(
        random.PRNGKey(114),
        target.loglike,
        target.prior_transform,
        target.ndim,
        30,
        walks=5,
        dlogz=2.0,
        maxiter=96,
        block_size=16,
    )

    assert math.isfinite(result.logz)
    assert result.metadata["replacement_failures"] == 0
    assert result.ncall > 0


def test_live_cov_cholesky_handles_degenerate_live_set() -> None:
    from tinyns.samplers import live_cov_cholesky

    live_u = jnp.tile(jnp.asarray([[0.3, 0.7, 0.5]]), (8, 1))
    chol = live_cov_cholesky(live_u)
    assert chol.shape == (3, 3)
    assert bool(jnp.all(jnp.isfinite(chol)))

    rng = np.random.default_rng(0)
    points = rng.uniform(size=(400, 2))
    chol = live_cov_cholesky(jnp.asarray(points))
    # float32 (CI default) tolerance: the helper adds a ~1e-6 relative diagonal
    # jitter, and the off-diagonal covariance of uniform points is near zero.
    np.testing.assert_allclose(chol @ chol.T, np.cov(points.T), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("block_size", [1, 8])
def test_live_cov_rwalk_recovers_correlated_gaussian_evidence(block_size) -> None:
    """A narrow, strongly correlated 2D Gaussian under a wide box prior."""
    width = 20.0
    cov = np.array([[1.0, 0.95 * 0.05], [0.95 * 0.05, 0.05**2]])
    prec = jnp.asarray(np.linalg.inv(cov))
    norm = -0.5 * (2 * math.log(2 * math.pi) + math.log(np.linalg.det(cov)))

    def loglike(theta):
        return norm - 0.5 * theta @ prec @ theta

    def prior_transform(u):
        return -0.5 * width + width * u

    result = run_static_nested(
        random.PRNGKey(3),
        loglike,
        prior_transform,
        2,
        200,
        walks=25,
        block_size=block_size,
    )
    assert result.success
    assert abs(result.logz - (-2 * math.log(width))) < 5 * result.logzerr
    assert 0.1 < result.metadata["rwalk_acceptance"] < 0.5


@pytest.mark.parametrize("cluster_swap", [False, True])
def test_block_size_one_supports_cluster_swap_setting(cluster_swap) -> None:
    result = run_static_nested(
        random.PRNGKey(31),
        _jax_loglike,
        _jax_prior_transform,
        2,
        40,
        block_size=1,
        cluster_swap=cluster_swap,
        maxiter=50,
        dlogz=0.0,
    )

    assert result.metadata["cluster_swap"] is cluster_swap
    assert result.metadata["niter"] == 50
    assert result.metadata["replacement_failures"] == 0


def test_static_nested_metadata_has_no_removed_option_keys() -> None:
    states = []
    result = run_static_nested(
        random.PRNGKey(107),
        lambda theta: -0.5 * jnp.sum(theta**2),
        lambda u: 2.0 * u - 1.0,
        ndim=2,
        nlive=20,
        maxiter=2,
        dlogz=0.0,
        callback=states.append,
    )

    removed = {
        "sample",
        "kernel",
        "rwalk_proposal",
        "min_accepts",
        "max_attempts",
        "batch_size",
        "jax_vectorized",
        "rwalk_target_accept",
        "mean_total_replacement_calls",
    }
    removed_parts = (
        "bound",
        "step_scale",
        "schedule",
        "rescue",
        "chains_used",
        "chain_usage",
        "jax_block",
    )
    assert result.metadata["niter"] == 2
    assert states
    for names in (result.metadata, states[-1]):
        stale = [
            n for n in names if n in removed or any(r in n for r in removed_parts)
        ]
        assert not stale


def test_block_rwalk_supports_unhashable_callable_instances() -> None:
    class UnhashablePrior:
        __hash__ = None

        def __call__(self, u):
            return 2.0 * u - 1.0

    class UnhashableLogLike:
        __hash__ = None

        def __call__(self, theta):
            return -jnp.sum(theta**2)

    result = run_static_nested(
        random.PRNGKey(12341),
        UnhashableLogLike(),
        UnhashablePrior(),
        2,
        12,
        walks=1,
        maxiter=2,
        dlogz=10.0,
        block_size=2,
    )

    assert result.success is True
    assert result.metadata["niter"] == 2
    assert jnp.isfinite(result.logz)


def test_static_jax_block_kernel_cache_is_bounded() -> None:
    def prior_transform(u):
        return u

    run_mod._make_static_jax_rwalk_block_kernel.cache_clear()
    run_mod._make_rwalk_jax_kernel_cached.cache_clear()
    try:
        for offset in range(40):
            def loglike(theta, offset=offset):
                return -jnp.sum(theta**2) + offset

            run_mod._make_static_jax_rwalk_block_kernel(
                loglike,
                prior_transform,
                2,
                1,
                1,
                2,
            )

        cache_info = run_mod._make_static_jax_rwalk_block_kernel.cache_info()
        assert cache_info.maxsize == 32
        assert cache_info.currsize == 32
        assert cache_info.misses == 40
    finally:
        run_mod._make_static_jax_rwalk_block_kernel.cache_clear()
        run_mod._make_rwalk_jax_kernel_cached.cache_clear()


def test_update_scale_direction_and_clamps() -> None:
    from tinyns.run import _update_scale

    assert _update_scale(0.5, 0.05) < 0.5
    assert _update_scale(0.5, 0.75) > 0.5
    assert _update_scale(0.5, 0.25) == pytest.approx(0.5)
    # log-space step of rate * clip(accept - 0.25, -0.5, 0.5)
    assert _update_scale(0.5, 1.0) == pytest.approx(0.5 * math.exp(0.25))
    assert _update_scale(1e-3, 0.0) == pytest.approx(1e-3)
    assert _update_scale(10.0, 1.0) == pytest.approx(10.0)


@pytest.mark.parametrize("moves, grows", [(1, False), (15, True)])
def test_scale_adapts_to_block_move_acceptance(monkeypatch, moves, grows) -> None:
    fake = _fake_block_kernel(
        accepted_prefix=4, replacement_ncall=(20, 20, 20, 20), moves=(moves,) * 4
    )
    monkeypatch.setattr(run_mod, "_make_static_jax_rwalk_block_kernel", fake)
    result = run_static_nested(
        random.PRNGKey(123),
        _standard_gaussian_2d_loglike,
        _wide_box_prior_transform,
        ndim=2,
        nlive=16,
        walks=5,
        dlogz=0.0,
        maxiter=8,
        block_size=4,
    )

    metadata = result.metadata
    assert fake.scales[0] == 0.5  # the initial scale
    assert len(fake.scales) == 2
    assert (fake.scales[1] > 0.5) is grows
    assert (metadata["rwalk_scale_final"] > 0.5) is grows
    assert metadata["rwalk_adaptation_updates"] == 2
    assert metadata["rwalk_observed_accept_mean"] == pytest.approx(moves / 20)


def test_live_cov_single_chain_skips_out_of_cube_evaluations() -> None:
    """ncall counts real likelihood calls; out-of-cube proposals are not evaluated."""
    import jax

    calls = []

    def loglike(theta):
        jax.debug.callback(lambda: calls.append(1))
        # mass near the cube corner, so many proposals leave the cube
        return -0.5 * jnp.sum(((theta - 0.05) / 0.05) ** 2)

    def prior_transform(u):
        return u

    result = run_static_nested(
        random.PRNGKey(7), loglike, prior_transform, 3, 100, walks=25, maxiter=640
    )
    jax.effects_barrier()
    metadata = result.metadata
    assert len(calls) == result.ncall
    assert result.ncall < 100 + metadata["total_rwalk_proposals"]
