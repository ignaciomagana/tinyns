from __future__ import annotations

import itertools
import math

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from tinyns.result import NestedSamplingResult


def make_result() -> NestedSamplingResult:
    """Two live points, two iterations: -3 and -2 die and are replaced by -1
    and -0.5, each inserted above the one other live point."""
    return NestedSamplingResult(
        samples_u=jnp.array([[0.1, 0.2], [0.3, 0.1], [0.5, 0.6], [0.7, 0.4]]),
        samples=jnp.arange(8, dtype=float).reshape(4, 2),
        logl=jnp.array([-3.0, -2.0, -1.0, -0.5]),
        logwt=jnp.array([-4.0, -2.0, -1.0, -0.25]),
        logl_birth=jnp.array([-jnp.inf, -jnp.inf, -3.0, -2.0]),
        nlive_i=np.array([2, 2, 2, 1]),
        labels=np.array([0, 0, 0, 0], np.int32),
        logz=-0.1,
        logzerr=0.01,
        ncall=10,
        niter=2,
        nlive=2,
        num_delete=1,
        ndim=2,
        message="ok",
        metadata={"status": "complete"},
    )


def test_weights_sum_to_one() -> None:
    result = make_result()

    assert np.isclose(float(jnp.sum(result.weights())), 1.0)


def test_resample_equal_returns_requested_shape() -> None:
    result = make_result()

    samples = result.resample_equal(random.PRNGKey(0), n=3)

    assert samples.shape == (3, result.ndim)


def test_posterior_ess_is_positive() -> None:
    result = make_result()

    assert result.posterior_ess() > 0.0


def test_summary_contains_logz() -> None:
    result = make_result()

    summary = result.summary()

    assert isinstance(summary, str)
    assert "logz" in summary


def test_to_dict_contains_expected_keys() -> None:
    result = make_result()

    data = result.to_dict()

    assert set(data) == {
        "samples_u",
        "samples",
        "logl",
        "logwt",
        "logl_birth",
        "nlive_i",
        "logz",
        "logzerr",
        "ncall",
        "niter",
        "nlive",
        "num_delete",
        "ndim",
        "success",
        "message",
        "metadata",
    }
    assert data["metadata"] == {"status": "complete"}
    assert data["metadata"] is not result.metadata


def test_to_numpy_converts_arrays_and_preserves_to_dict_behavior() -> None:
    result = make_result()

    numpy_data = result.to_numpy()
    dict_data = result.to_dict()

    assert isinstance(numpy_data["samples"], np.ndarray)
    assert isinstance(numpy_data["samples_u"], np.ndarray)
    assert isinstance(numpy_data["logl"], np.ndarray)
    assert isinstance(numpy_data["logwt"], np.ndarray)
    assert isinstance(numpy_data["logz"], float)
    assert isinstance(numpy_data["ncall"], int)
    assert isinstance(numpy_data["logl_birth"], np.ndarray)
    assert numpy_data["niter"] == 2
    assert numpy_data["metadata"] == {"status": "complete"}
    assert dict_data["samples"] is result.samples


def test_to_dynesty_dict_contains_lightweight_compatibility_keys() -> None:
    result = make_result()

    data = result.to_dynesty_dict()

    assert {"samples", "logl", "logwt", "logz"}.issubset(data)
    assert isinstance(data["samples"], np.ndarray)
    assert isinstance(data["samples_u"], np.ndarray)
    assert data["logz"] == result.logz
    assert data["logzerr"] == result.logzerr
    assert data["ncall"] == result.ncall
    assert data["nlive"] == result.nlive
    assert data["niter"] == 2
    assert data["eff"] == 100.0 * 2 / 10  # dynesty's 100 niter / ncall


def test_str_is_the_summary() -> None:
    result = make_result()

    assert str(result) == result.summary()
    assert "niter: 2" in str(result)
    assert "mode" not in str(result)  # the table only for several modes


def test_information_is_finite_and_non_negative() -> None:
    result = make_result()

    information = result.information()

    assert np.isfinite(information)
    assert information >= 0.0


def test_information_ignores_zero_weight_nonfinite_likelihood() -> None:
    result = make_result()
    result.logz = 0.0
    result.logwt = jnp.array([0.0, -jnp.inf])
    result.logl = jnp.array([0.0, -jnp.inf])

    assert result.information() == 0.0


def _small_gaussian_run(num_delete=None):
    from tinyns import NestedSampler

    def loglike(theta):
        return -0.5 * jnp.sum(theta**2) - math.log(2.0 * math.pi)

    sampler = NestedSampler(
        loglike, lambda u: 4.0 * u - 2.0, 2, 50, num_delete=num_delete
    )
    return sampler.run(random.PRNGKey(0), dlogz=0.3)


def test_logz_bootstrap_reconstruction_matches_sampler() -> None:
    from tinyns.result import _log_weights, _np_logsumexp

    for k in (1, 5):
        result = _small_gaussian_run(k)
        # Feeding the expected shrinkage log t_i = -1 / n_i must reproduce the
        # sampler's own logz, confirming the weight convention matches.
        log_t = -1.0 / np.asarray(result.nlive_i, dtype=float)[None, :]
        logwt = _log_weights(result.logl, result.nlive_i, log_t)
        assert abs(float(_np_logsumexp(logwt, axis=1)[0]) - result.logz) < 1e-12


def test_logz_bootstrap_is_deterministic_and_sane() -> None:
    result = _small_gaussian_run()

    first = result.logz_bootstrap(n_realizations=512, seed=0)
    repeat = result.logz_bootstrap(n_realizations=512, seed=0)
    other = result.logz_bootstrap(n_realizations=512, seed=1)

    assert first.n_realizations == 512
    assert first.samples.shape == (512,)
    assert np.array_equal(first.samples, repeat.samples)
    assert not np.array_equal(first.samples, other.samples)

    assert math.isfinite(first.logzerr) and first.logzerr > 0.0
    assert first.logz_p16 <= first.logz_median <= first.logz_p84
    # Mean realization sits near the reported logz.
    assert abs(first.logz_mean - float(result.logz)) < 5.0 * first.logzerr
    # Volume-path uncertainty is the same order as the analytic sqrt(H/nlive).
    analytic = math.sqrt(max(result.information(), 0.0) / result.nlive)
    assert 0.25 * analytic < first.logzerr < 4.0 * analytic


def test_logz_bootstrap_input_validation() -> None:
    result = make_result()
    with np.testing.assert_raises(ValueError):
        result.logz_bootstrap(n_realizations=0)

    result.nlive_i = np.array([2, 2, 0, 1])
    with np.testing.assert_raises(ValueError):
        result.logz_bootstrap(n_realizations=8)


def test_information_is_nan_for_weighted_nonfinite_likelihood() -> None:
    from tinyns.result import _information

    assert math.isnan(float(_information(jnp.array([0.0]), jnp.array([jnp.nan]), 0.0)))
    information = _information(
        jnp.array([0.0, -jnp.inf]), jnp.array([0.0, -jnp.inf]), 0.0
    )
    assert float(information) == 0.0


def test_diagnostics_returns_plain_dict_with_warning_list() -> None:
    result = make_result()

    diagnostics = result.diagnostics()

    assert isinstance(diagnostics, dict)
    assert isinstance(diagnostics["warnings"], list)
    assert diagnostics["information"] == result.information()
    assert diagnostics["nposterior"] == 4
    assert diagnostics["niter"] == 2
    assert diagnostics["modes"] == result.modes()
    assert diagnostics["insertion_pvalue"] == result.insertion_test()["pvalue"]
    for name in (
        "max_weight_fraction",
        "posterior_weight_entropy_fraction",
        "live_weight_fraction",
        "dead_weight_fraction",
        "final_delta_logz",
        "acceptance",
    ):
        assert name in diagnostics


def test_diagnostics_warns_on_nonfinite_logzerr_and_low_acceptance() -> None:
    result = make_result()
    result.logzerr = math.nan
    result.metadata = {"acceptance": 0.001}

    warnings = result.diagnostics()["warnings"]

    assert "logzerr is not finite" in warnings
    assert "low replacement acceptance" in warnings


def test_max_weight_fraction_returns_expected_value() -> None:
    result = make_result()
    result.logwt = jnp.log(jnp.array([0.2, 0.3, 0.5]))

    assert np.isclose(result.max_weight_fraction(), 0.5)


def test_posterior_weight_entropy_is_finite_and_positive() -> None:
    result = make_result()

    entropy = result.posterior_weight_entropy()

    assert np.isfinite(entropy)
    assert entropy > 0.0


def test_posterior_weight_entropy_fraction_near_one_for_equal_weights() -> None:
    result = make_result()
    result.logwt = jnp.zeros(10)

    assert np.isclose(result.posterior_weight_entropy_fraction(), 1.0)


def test_degenerate_weights_have_high_max_weight_and_low_entropy_fraction() -> None:
    result = make_result()
    result.logwt = jnp.concatenate([jnp.array([0.0]), jnp.full(99, -1000.0)])

    assert result.max_weight_fraction() > 0.9
    assert result.posterior_weight_entropy_fraction() < 0.1


def test_live_weight_fraction_counts_the_samples_after_niter() -> None:
    result = make_result()
    result.logwt = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))

    assert np.isclose(result.live_weight_fraction(), 0.7)


def test_dead_and_live_weight_fractions_sum_to_one() -> None:
    result = make_result()
    result.logwt = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))

    total_weight_fraction = (
        result.dead_weight_fraction() + result.live_weight_fraction()
    )

    assert np.isclose(total_weight_fraction, 1.0)


def test_diagnostics_high_live_weight_triggers_warning() -> None:
    result = make_result()
    result.logwt = jnp.log(jnp.array([0.1, 0.1, 0.4, 0.4]))
    result.metadata = {"dlogz": 0.1}

    diagnostics = result.diagnostics()

    assert (
        "final live points carry most posterior weight; consider tighter dlogz or "
        "more live points"
    ) in diagnostics["warnings"]
    assert (
        "large final-live weight fraction; evidence may be sensitive to stopping"
    ) in diagnostics["warnings"]


def test_diagnostics_high_max_weight_triggers_warning() -> None:
    result = make_result()
    result.logwt = jnp.log(jnp.array([0.85, 0.05, 0.05, 0.05]))

    diagnostics = result.diagnostics()

    assert (
        "posterior dominated by a small number of weighted samples"
        in diagnostics["warnings"]
    )


def test_diagnostics_success_with_high_final_delta_logz_triggers_warning() -> None:
    result = make_result()
    result.metadata = {"dlogz": 0.1, "final_delta_logz": 0.2}

    diagnostics = result.diagnostics()

    assert (
        "successful run has final_delta_logz above requested dlogz"
        in diagnostics["warnings"]
    )


def test_diagnostics_low_ess_triggers_warning() -> None:
    result = NestedSamplingResult(
        samples_u=jnp.zeros((101, 1)),
        samples=jnp.zeros((101, 1)),
        logl=jnp.zeros(101),
        logwt=jnp.concatenate([jnp.array([0.0]), jnp.full(100, -1000.0)]),
        logl_birth=jnp.full(101, -jnp.inf),
        nlive_i=101 - np.arange(101),
        logz=0.0,
        logzerr=0.1,
        ncall=101,
        niter=0,
        nlive=101,
        num_delete=1,
        ndim=1,
    )

    diagnostics = result.diagnostics()

    assert "low posterior ESS" in diagnostics["warnings"]


def test_insertion_indices_are_rebuilt_from_the_birth_contours() -> None:
    result = make_result()

    assert result.insertion_indices().tolist() == [1, 1]


def _replayed_result(ranks, nlive, k=1):
    """A result whose births insert each new point at the given ranks among
    the survivors of its step (``k`` deaths and births per step)."""
    rng = np.random.default_rng(0)
    live = sorted(rng.uniform(0.0, 1.0, nlive).tolist())
    logl, births = [], {}
    initial = list(live)
    for step in range(len(ranks) // k):
        dead = live[:k]
        del live[:k]
        logl += dead
        survivors = list(live)
        for rank in ranks[step * k : (step + 1) * k]:
            lo = dead[-1] if rank == 0 else survivors[rank - 1]
            hi = survivors[rank] if rank < len(survivors) else lo + 1.0
            new = lo + (hi - lo) * rng.uniform(0.01, 0.99)
            births[new] = dead[-1]
            live.append(new)
        live.sort()
    birth = [births.get(x, -np.inf) for x in logl + live]
    assert sum(b == -np.inf for b in birth) == len(initial)
    n = len(logl) + nlive
    nlive_i = np.concatenate(
        [np.tile(nlive - np.arange(k), len(logl) // k), nlive - np.arange(nlive)]
    )
    return NestedSamplingResult(
        samples_u=jnp.asarray(rng.uniform(size=(n, 1))),
        samples=jnp.zeros((n, 1)),
        logl=np.asarray(logl + live),  # float64: no ties after many births
        logwt=jnp.zeros(n),
        logl_birth=np.asarray(birth),
        nlive_i=nlive_i,
        logz=0.0,
        logzerr=0.1,
        ncall=n,
        niter=len(logl),
        nlive=nlive,
        num_delete=k,
        ndim=1,
    )


def test_insertion_replay_with_several_deaths_per_step() -> None:
    nlive, k = 12, 4
    rng = np.random.default_rng(3)
    ranks = rng.integers(0, nlive - k + 1, 400)
    result = _replayed_result(ranks, nlive, k)
    rebuilt = result.insertion_indices().reshape(-1, k)
    np.testing.assert_array_equal(
        np.sort(rebuilt, axis=1), np.sort(ranks.reshape(-1, k), axis=1)
    )
    born = result._births[0]
    assert set(born[born >= 0].tolist()) == set(range(k - 1, 400, k))
    assert result.insertion_test()["pvalue"] > 0.01
    top = np.full_like(ranks, nlive - k)  # always above every survivor
    assert _replayed_result(top, nlive, k).insertion_test()["pvalue"] < 1e-6


def test_insertion_test_passes_uniform_ranks_and_flags_biased_ones() -> None:
    nlive = 10
    uniform = _replayed_result(np.random.default_rng(1).integers(0, nlive, 600), nlive)
    np.testing.assert_array_equal(
        uniform.insertion_indices(),
        np.random.default_rng(1).integers(0, nlive, 600),
    )
    test = uniform.insertion_test(windows=3)
    assert test["n"] == 600 and test["pvalue"] > 0.01
    assert [(w["start"], w["stop"]) for w in test["windows"]] == [
        (0, 200),
        (200, 400),
        (400, 600),
    ]
    warning = (
        "insertion indices look non-uniform; constrained sampler may be biased or "
        "poorly mixed"
    )
    assert warning not in uniform.diagnostics()["warnings"]

    # Uniform for two thirds of the run, then always inserted at the bottom:
    # the last window sees it at once, the pooled test more weakly.
    ranks = np.concatenate([np.tile(np.arange(nlive), 40), np.zeros(200, dtype=int)])
    biased = _replayed_result(ranks, nlive).insertion_test(windows=3)
    assert biased["windows"][2]["pvalue"] < 1e-6
    assert biased["windows"][0]["pvalue"] > 0.5
    assert warning in _replayed_result(ranks, nlive).diagnostics()["warnings"]
    with np.testing.assert_raises(ValueError):
        uniform.insertion_test(windows=0)


def test_ks_pvalue_matches_the_kolmogorov_law() -> None:
    from tinyns.result import _ks_uniform

    assert _ks_uniform(np.arange(10), 10) == (0.0, 1.0)
    ks, pvalue = _ks_uniform(np.zeros(100, dtype=int), 10)
    assert ks == 0.9 and pvalue < 1e-20
    # 227 of 400 ranks in the lower of two slots: D = 0.0675, at the 5%
    # critical value 1.358 of the Kolmogorov law.
    ks, pvalue = _ks_uniform(np.repeat([0, 1], [227, 173]), 2)
    assert ks == 0.0675
    assert abs(pvalue - 0.05) < 0.002


def test_result_npz_round_trip(tmp_path) -> None:
    result = make_result()
    result.success = False
    result.message = "stopped"
    path = tmp_path / "result.npz"

    result.save_npz(path)
    loaded = NestedSamplingResult.load_npz(path)

    np.testing.assert_allclose(
        np.asarray(loaded.samples_u), np.asarray(result.samples_u)
    )
    np.testing.assert_allclose(np.asarray(loaded.samples), np.asarray(result.samples))
    np.testing.assert_allclose(np.asarray(loaded.logl), np.asarray(result.logl))
    np.testing.assert_allclose(np.asarray(loaded.logwt), np.asarray(result.logwt))
    assert loaded.logz == result.logz
    assert loaded.logzerr == result.logzerr
    assert loaded.ncall == result.ncall
    assert loaded.niter == result.niter
    np.testing.assert_array_equal(loaded.logl_birth, result.logl_birth)
    np.testing.assert_array_equal(loaded.nlive_i, result.nlive_i)
    np.testing.assert_array_equal(loaded.labels, result.labels)
    assert loaded.nlive == result.nlive
    assert loaded.num_delete == result.num_delete
    assert loaded.ndim == result.ndim
    assert loaded.success == result.success
    assert loaded.message == result.message


def test_result_npz_round_trip_metadata_jsonable(tmp_path) -> None:
    result = make_result()
    result.metadata = {
        "name": "demo",
        "count": np.int64(3),
        "scale": np.float64(1.5),
        "ok": np.bool_(True),
        "items": [1, "two", False],
        "nested": {"value": jnp.asarray(2.0)},
        "numpy_array": np.array([1, 2, 3]),
        "jax_array": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "unsupported": object(),
    }
    path = tmp_path / "metadata.npz"

    result.save_npz(path)
    loaded = NestedSamplingResult.load_npz(path)

    assert loaded.metadata is not None
    assert loaded.metadata["name"] == "demo"
    assert loaded.metadata["count"] == 3
    assert loaded.metadata["scale"] == 1.5
    assert loaded.metadata["ok"] is True
    assert loaded.metadata["items"] == [1, "two", False]
    assert loaded.metadata["nested"] == {"value": 2.0}
    assert loaded.metadata["numpy_array"] == [1, 2, 3]
    assert loaded.metadata["jax_array"] == [[1.0, 2.0], [3.0, 4.0]]
    assert isinstance(loaded.metadata["unsupported"], str)


def test_result_npz_without_metadata_round_trips(tmp_path) -> None:
    result = make_result()
    result.metadata = None
    path = tmp_path / "none.npz"
    result.save_npz(path)

    assert NestedSamplingResult.load_npz(path).metadata is None


def test_loaded_result_npz_still_behaves_like_result(tmp_path) -> None:
    result = make_result()
    path = tmp_path / "result.npz"
    result.save_npz(path)

    loaded = NestedSamplingResult.load_npz(path)

    assert np.isclose(float(jnp.sum(loaded.weights())), 1.0)
    assert loaded.posterior_ess() > 0.0
    assert isinstance(loaded.diagnostics(), dict)
    assert loaded.resample_equal(random.PRNGKey(0), n=3).shape == (3, result.ndim)
    assert isinstance(loaded.summary(), str)
    assert isinstance(loaded.to_numpy()["samples"], np.ndarray)
    assert isinstance(loaded.to_dynesty_dict()["samples"], np.ndarray)


def test_result_npz_bad_format_version_raises(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        samples_u=np.zeros((1, 1)),
        samples=np.zeros((1, 1)),
        logl=np.zeros(1),
        logwt=np.zeros(1),
        logl_birth=np.zeros(1),
        nlive_i=np.ones(1),
        num_delete=1,
        logz=0.0,
        niter=0,
        logzerr=0.0,
        ncall=1,
        nlive=1,
        ndim=1,
        success=True,
        message="",
        metadata_json="null",
        format_version="bad-version",
    )

    with np.testing.assert_raises(ValueError):
        NestedSamplingResult.load_npz(path)


def test_result_npz_missing_required_key_raises(tmp_path) -> None:
    path = tmp_path / "missing.npz"
    np.savez_compressed(path, format_version="tinyns-result-npz-v3")

    with np.testing.assert_raises(ValueError):
        NestedSamplingResult.load_npz(path)


def test_split_pieces_of_a_curved_ridge_touch_and_separate_modes_do_not() -> None:
    """The split test cuts one curved mode into pieces; modes() merges them
    because neighbouring pieces leave no gap along their discriminant."""
    from tinyns import modes
    from tinyns.result import SEPARATION_SIGMA, _gap

    rng = np.random.default_rng(2)
    ridge = [rng.normal(0.0, 2.0, 500)]
    for _ in range(4):  # a 5-D Rosenbrock ridge
        ridge.append(0.5 * ridge[-1] ** 2 - 1.0 + 0.5 * rng.normal(size=500))
    x = np.stack(ridge, axis=1)
    labels, _ = modes.recluster(jnp.asarray(x), jnp.zeros(500, jnp.int32))
    labels = np.unique(np.asarray(labels), return_inverse=True)[1]
    k = int(labels.max()) + 1
    assert k >= 2  # the split test does cut the ridge
    touching = []
    for a, b in itertools.combinations(range(k), 2):
        pair = (labels == a) | (labels == b)
        if _gap(x[pair], labels[pair] == b) <= SEPARATION_SIGMA:
            touching.append((a, b))
    assert len(touching) >= k - 1
    assert {c for edge in touching for c in edge} == set(range(k))

    blobs = np.concatenate(
        [rng.normal(size=(400, 5)), rng.normal(size=(30, 5)) + [12, 0, 0, 0, 0]]
    )
    side = np.arange(430) >= 400
    assert _gap(blobs, side) > 2 * SEPARATION_SIGMA
    # Copies of one point (an unmoved chain returns its seed) count once.
    copies = np.concatenate([blobs, np.repeat(blobs[400:401], 25, axis=0)])
    with_copies = _gap(copies, np.concatenate([side, np.ones(25, bool)]))
    assert with_copies == pytest.approx(_gap(blobs, side))

