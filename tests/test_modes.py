"""``NestedSamplingResult.modes()`` on sampler runs: post-processing of the
cluster labels the run recorded."""

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


def test_two_mode_runs_find_both_modes() -> None:
    """Both modes in every run (96 of 96 batched runs of this target at nlive
    400, x64 off and on, minor mass 0.25 +- 0.012; the v0.3 heuristic missed
    11). Before modes() measured a small label's distance to the others at
    their own posterior medians, 3 of the 192 reported one mode: the
    clustering had cut three points off the minor mode in its last steps."""
    for seed in range(5):
        result = NestedSampler(two_modes, lambda u: u, 3, nlive=400).run(seed)
        weights = np.asarray(result.weights())
        minor = np.asarray(result.samples)[:, 0] > 0.55
        mass = weights[minor].sum() / weights.sum()
        assert 0.15 < mass < 0.35  # truth 0.25; the hop keeps the urn balanced
        assert abs(result.logz) < 4 * result.logzerr + 0.1  # truth 0
        modes = result.modes()
        assert len(modes) == 2
        assert sum(m["mass"] for m in modes) == pytest.approx(1.0)
        assert modes[1]["mass"] == pytest.approx(mass, abs=0.005)
        assert all(0 < m["urn_sd"] < 1 for m in modes)
        assert not any(m["unresolved"] for m in modes)


def test_unresolved_flags_a_mode_below_five_points_per_dimension() -> None:
    """``unresolved`` is ``min_live < 5 ndim``: at nlive 60 the 25% mode of
    the 3-D target holds about 15 live points, on either side of the
    threshold from seed to seed; at nlive 400 it holds about 100."""
    from tinyns import result as result_module

    assert result_module.UNRESOLVED_PER_DIM == 5.0
    flags = []
    for seed in range(6):
        result = NestedSampler(two_modes, lambda u: u, 3, nlive=60).run(seed)
        for mode in result.modes():
            assert mode["unresolved"] == (mode["min_live"] < 15)
            flags.append(mode["unresolved"])
        warned = sum("unresolved" in w for w in result.diagnostics()["warnings"])
        assert warned == sum(m["unresolved"] for m in result.modes())
    assert any(flags)


def banana5(theta):
    y = (theta - 0.5) * 10.0
    ridge = (y[1:] - 0.5 * y[:-1] ** 2 + 1.0) ** 2
    return -0.5 * (y[0] ** 2 / 4.0 + jnp.sum(ridge) / 0.25)


def test_curved_unimodal_runs_report_one_mode() -> None:
    """The split test cuts a 5-D banana into pieces, and stuck chains leave
    clumps of near-copies in its tips; modes() must merge or drop them all.
    No run in 192 (nlive 500 and 1000, x64 off and on) reported a second
    mode."""
    ones = 0
    for seed in range(5):
        result = NestedSampler(banana5, lambda u: u, 5, nlive=500).run(seed)
        warnings = result.diagnostics()["warnings"]
        ones += len(result.modes()) == 1 and not any("mode" in w for w in warnings)
    assert ones >= 4


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


def test_modes_do_not_depend_on_slot_ids_or_reuse() -> None:
    """The run's labels are cluster slots: a mode's slot id is arbitrary, a
    slot freed early can hold other points later, and a mode's first points
    die in the bulk's slot before the clustering splits it off. modes()
    relabels every sample by its nearest mode, so the masses do not change."""
    result = NestedSampler(two_modes, lambda u: u, 3, nlive=400).run(0)
    modes = result.modes()
    assert len(modes) == 2
    labels = np.asarray(result.labels)
    minor = np.asarray(result.samples)[:, 0] > 0.55
    early = np.flatnonzero(minor & np.isin(labels, labels[~minor]))
    assert len(early) > 0  # minor points that died before the split
    ids = labels.max() + 2
    labels = (labels + 3) % ids  # other slot ids
    labels[: result.niter // 4] = ids  # a slot used in the prior phase only
    result.labels = labels
    relabelled = result.modes()
    assert len(relabelled) == 2
    assert relabelled[1]["mass"] == pytest.approx(modes[1]["mass"], abs=1e-6)
    assert relabelled[1]["min_live"] == modes[1]["min_live"]
