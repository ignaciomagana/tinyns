from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random

from tinyns.result import NestedSamplingResult


def make_result() -> NestedSamplingResult:
    return NestedSamplingResult(
        samples_u=jnp.zeros((4, 2)),
        samples=jnp.arange(8, dtype=float).reshape(4, 2),
        logl=jnp.array([-3.0, -2.0, -1.0, -0.5]),
        logwt=jnp.array([-4.0, -2.0, -1.0, -0.25]),
        logz=-0.1,
        logzerr=0.01,
        ncall=10,
        nlive=4,
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
        "logz",
        "logzerr",
        "ncall",
        "nlive",
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
    assert numpy_data["metadata"] == {"status": "complete"}
    assert dict_data["samples"] is result.samples


def test_to_dynesty_dict_contains_lightweight_compatibility_keys() -> None:
    result = make_result()
    result.metadata = {"replacement_acceptance_proxy": 0.25}

    data = result.to_dynesty_dict()

    assert {"samples", "logl", "logwt", "logz"}.issubset(data)
    assert isinstance(data["samples"], np.ndarray)
    assert isinstance(data["samples_u"], np.ndarray)
    assert data["logz"] == result.logz
    assert data["logzerr"] == result.logzerr
    assert data["ncall"] == result.ncall
    assert data["nlive"] == result.nlive
    assert data["eff"] == 0.25


def test_to_dynesty_dict_omits_eff_without_acceptance_proxy() -> None:
    result = make_result()

    data = result.to_dynesty_dict()

    assert "eff" not in data


def test_summary_contains_replacement_information_when_present() -> None:
    result = make_result()
    result.metadata = {
        "mean_replacement_ncall": 2.5,
        "replacement_failures": 1,
    }

    summary = result.summary()

    assert "replacement mean ncall: 2.5" in summary
    assert "replacement failures: 1" in summary


def test_information_is_finite_and_non_negative() -> None:
    result = make_result()

    information = result.information()

    assert np.isfinite(information)
    assert information >= 0.0


def test_diagnostics_returns_plain_dict_with_warning_list() -> None:
    result = make_result()

    diagnostics = result.diagnostics()

    assert isinstance(diagnostics, dict)
    assert isinstance(diagnostics["warnings"], list)
    assert diagnostics["information"] == result.information()
    assert diagnostics["nposterior"] == 4
    assert "max_weight_fraction" in diagnostics
    assert "posterior_weight_entropy" in diagnostics
    assert "posterior_weight_entropy_fraction" in diagnostics
    assert "live_weight_fraction" in diagnostics
    assert "dead_weight_fraction" in diagnostics
    assert "final_delta_logz" in diagnostics
    assert "final_logx" in diagnostics
    assert "final_logz_dead" in diagnostics
    assert "final_logl_live_max" in diagnostics


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


def test_live_weight_fraction_uses_nlive_final_metadata() -> None:
    result = make_result()
    result.logwt = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))
    result.metadata = {"nlive_final": 2}

    assert np.isclose(result.live_weight_fraction(), 0.7)


def test_dead_and_live_weight_fractions_sum_to_one_with_valid_metadata() -> None:
    result = make_result()
    result.logwt = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))
    result.metadata = {"nlive_final": 2}

    total_weight_fraction = (
        result.dead_weight_fraction() + result.live_weight_fraction()
    )

    assert np.isclose(total_weight_fraction, 1.0)


def test_diagnostics_high_live_weight_triggers_warning() -> None:
    result = make_result()
    result.logwt = jnp.log(jnp.array([0.1, 0.1, 0.4, 0.4]))
    result.metadata = {"nlive_final": 2, "dlogz": 0.1}

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


def test_diagnostics_includes_iteration_counts_when_present() -> None:
    result = make_result()
    result.metadata = {"niter": 12, "ndead": 12}

    diagnostics = result.diagnostics()

    assert diagnostics["niter"] == 12
    assert diagnostics["ndead"] == 12


def test_diagnostics_low_ess_triggers_warning() -> None:
    result = NestedSamplingResult(
        samples_u=jnp.zeros((101, 1)),
        samples=jnp.zeros((101, 1)),
        logl=jnp.zeros(101),
        logwt=jnp.concatenate([jnp.array([0.0]), jnp.full(100, -1000.0)]),
        logz=0.0,
        logzerr=0.1,
        ncall=101,
        nlive=10,
        ndim=1,
    )

    diagnostics = result.diagnostics()

    assert "low posterior ESS" in diagnostics["warnings"]


def test_insertion_indices_returns_array() -> None:
    result = make_result()
    result.metadata = {"insertion_indices": [0, 2, 1]}

    insertion_indices = result.insertion_indices()

    assert isinstance(insertion_indices, jnp.ndarray)
    assert insertion_indices.tolist() == [0, 2, 1]


def test_insertion_indices_missing_metadata_returns_empty_array() -> None:
    result = make_result()

    insertion_indices = result.insertion_indices()

    assert isinstance(insertion_indices, jnp.ndarray)
    assert insertion_indices.shape == (0,)

def test_diagnostics_insertion_indices_use_slot_count_for_normalization() -> None:
    result = make_result()
    result.nlive = 2
    result.metadata = {
        "insertion_indices": jnp.tile(jnp.arange(2), 10),
        "insertion_index_nslots": 2,
        "insertion_index_nlive": 1,
    }

    diagnostics = result.diagnostics()

    assert (
        "insertion indices look non-uniform; constrained sampler may be biased or "
        "poorly mixed"
    ) not in diagnostics["warnings"]


def test_diagnostics_prefers_insertion_index_nslots_when_present() -> None:
    result = make_result()
    result.metadata = {
        "insertion_indices": jnp.tile(jnp.array([0, 1]), 10),
        "insertion_index_nslots": 2,
        "insertion_index_nlive": 100,
    }

    diagnostics = result.diagnostics()

    assert (
        "insertion indices look non-uniform; constrained sampler may be biased or "
        "poorly mixed"
    ) not in diagnostics["warnings"]


def test_diagnostics_supports_legacy_insertion_index_nlive_only() -> None:
    result = make_result()
    result.metadata = {
        "insertion_indices": jnp.tile(jnp.arange(2), 10),
        "insertion_index_nlive": 1,
    }

    diagnostics = result.diagnostics()

    assert isinstance(diagnostics, dict)
    assert isinstance(diagnostics["warnings"], list)

def test_diagnostics_bad_insertion_indices_triggers_warning() -> None:
    result = make_result()
    result.metadata = {
        "insertion_indices": jnp.zeros(20, dtype=int),
        "insertion_index_nlive": 10,
    }

    diagnostics = result.diagnostics()

    assert (
        "insertion indices look non-uniform; constrained sampler may be biased or "
        "poorly mixed"
    ) in diagnostics["warnings"]


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
    assert loaded.nlive == result.nlive
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
        logz=0.0,
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
    np.savez_compressed(path, format_version="tinyns-result-npz-v1")

    with np.testing.assert_raises(ValueError):
        NestedSamplingResult.load_npz(path)
