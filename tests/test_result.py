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
