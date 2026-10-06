"""``NestedSamplingResult.modes()`` on sampler runs (post-processing only)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from tinyns import NestedSampler

CENTERS = (jnp.array([0.3, 0.4, 0.5]), jnp.array([0.75, 0.6, 0.45]))
SIGMA = 0.03
W_MINOR = 0.25


def two_modes(theta):
    def log_normal(center):
        z = (theta - center) / SIGMA
        return -0.5 * jnp.sum(z * z) - 3 * math.log(SIGMA * math.sqrt(2 * math.pi))

    return jnp.logaddexp(
        math.log(1 - W_MINOR) + log_normal(CENTERS[0]),
        math.log(W_MINOR) + log_normal(CENTERS[1]),
    )


def test_two_mode_run_finds_both_modes() -> None:
    result = NestedSampler(two_modes, lambda u: u, 3, nlive=400).run(5)
    weights = np.asarray(result.weights())
    minor = np.asarray(result.samples)[:, 0] > 0.55
    mass = weights[minor].sum() / weights.sum()
    assert 0.08 < mass < 0.5  # truth 0.25; the urn drifts without a mode move
    assert abs(result.logz) < 4 * result.logzerr + 0.1  # truth 0
    modes = result.modes()
    assert len(modes) == 2
    assert sum(m["mass"] for m in modes) == pytest.approx(1.0)
    assert modes[1]["mass"] == pytest.approx(mass, abs=0.02)
    assert all(0 < m["urn_sd"] < 1 for m in modes)


def banana5(theta):
    y = (theta - 0.5) * 10.0
    ridge = (y[1:] - 0.5 * y[:-1] ** 2 + 1.0) ** 2
    return -0.5 * (y[0] ** 2 / 4.0 + jnp.sum(ridge) / 0.25)


def test_curved_unimodal_run_reports_one_mode() -> None:
    result = NestedSampler(banana5, lambda u: u, 5, nlive=250).run(0)
    assert len(result.modes()) == 1
    assert not any("mode" in w for w in result.diagnostics()["warnings"])


def test_unimodal_run_reports_one_mode() -> None:
    def gauss4(theta):
        return -0.5 * jnp.sum(((theta - 0.5) / 0.1) ** 2)

    result = NestedSampler(gauss4, lambda u: u, 4, nlive=100).run(3)
    assert result.modes() == [
        {
            "mass": 1.0,
            "urn_sd": 0.0,
            "min_live": 100,
            "isolation_iteration": 0,
            "unresolved": False,
        }
    ]
    assert "logit sd" not in result.summary()
