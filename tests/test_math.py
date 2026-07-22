from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from tinyns.math import (
    effective_sample_size_from_log_weights,
    logdiffexp,
    logsumexp,
    normalize_log_weights,
    reflect_unit_cube,
    systematic_resample,
)


def test_logsumexp_matches_numpy_for_simple_values() -> None:
    values = np.array([-3.0, -2.0, -1.0])

    assert logsumexp(values) == pytest.approx(np.log(np.sum(np.exp(values))))


def test_logsumexp_handles_empty_input() -> None:
    assert logsumexp([]) == -math.inf


def test_logsumexp_handles_all_negative_infinity() -> None:
    assert logsumexp([-math.inf, -math.inf]) == -math.inf


def test_logsumexp_handles_positive_infinity() -> None:
    assert logsumexp([-math.inf, 0.0, math.inf]) == math.inf


def test_logsumexp_propagates_nan() -> None:
    assert math.isnan(logsumexp([0.0, math.nan, math.inf]))


def test_logdiffexp_matches_direct_computation() -> None:
    a = math.log(5.0)
    b = math.log(2.0)

    assert logdiffexp(a, b) == pytest.approx(math.log(3.0))


def test_logdiffexp_requires_strictly_larger_left_value() -> None:
    with pytest.raises(ValueError, match="a > b"):
        logdiffexp(1.0, 1.0)


def test_normalize_log_weights_sums_to_one_in_probability_space() -> None:
    normalized = normalize_log_weights(jnp.array([-2.0, -1.0, -0.5]))

    assert jnp.sum(jnp.exp(normalized)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("logw", "match"),
    [
        ([], "must not be empty"),
        ([-math.inf, -math.inf], "at least one finite"),
        ([0.0, math.nan], "NaN"),
    ],
)
def test_normalize_log_weights_rejects_undefined_distributions(logw, match) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_log_weights(logw)


def test_normalize_log_weights_shares_mass_across_positive_infinities() -> None:
    normalized = normalize_log_weights(
        jnp.asarray((math.inf, 1.0, math.inf, -math.inf))
    )

    assert normalized[0] == pytest.approx(-math.log(2.0))
    assert normalized[2] == pytest.approx(-math.log(2.0))
    assert jnp.isneginf(normalized[1])
    assert jnp.isneginf(normalized[3])
    assert jnp.sum(jnp.exp(normalized)) == pytest.approx(1.0)


def test_effective_sample_size_is_n_for_equal_weights() -> None:
    logw = jnp.zeros(4)

    assert effective_sample_size_from_log_weights(logw) == pytest.approx(4.0)


def test_effective_sample_size_counts_positive_infinite_weights() -> None:
    logw = jnp.asarray((math.inf, 0.0, math.inf))

    assert effective_sample_size_from_log_weights(logw) == pytest.approx(2.0)


def test_systematic_resample_returns_integer_indices_of_requested_shape() -> None:
    indices = systematic_resample(random.PRNGKey(0), jnp.zeros(5), 7)

    assert indices.shape == (7,)
    assert jnp.issubdtype(indices.dtype, jnp.integer)


def test_systematic_resample_uses_only_positive_infinite_weights() -> None:
    indices = systematic_resample(
        random.PRNGKey(1), jnp.asarray((math.inf, 0.0, math.inf)), 20
    )

    assert jnp.all((indices == 0) | (indices == 2))


def test_reflect_unit_cube_maps_values_into_unit_interval() -> None:
    reflected = reflect_unit_cube(jnp.array([-1.25, -0.25, 0.25, 1.25, 2.25]))

    assert jnp.all(reflected >= 0.0)
    assert jnp.all(reflected <= 1.0)
