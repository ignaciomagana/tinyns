"""``NestedSamplingResult.modes()`` on sampler runs: post-processing of the
cluster labels the run recorded."""

from __future__ import annotations

import dataclasses
import math

import jax
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
            "tracked": True,
        }
    ]
    assert "logit sd" not in result.summary()
    diagnostics = result.diagnostics()
    assert diagnostics["mode_slots_full"] == 0.0
    assert not any("mode" in w for w in diagnostics["warnings"])


@pytest.fixture(scope="module")
def two_mode_run():
    return NestedSampler(two_modes, lambda u: u, 3, nlive=400).run(0)


def relabelled(result, labels, tree=()):
    """``result`` with other cluster labels and the split tree ``tree``."""
    metadata = {**result.metadata, "mode_tree": [list(pair) for pair in tree]}
    return dataclasses.replace(result, labels=np.asarray(labels), metadata=metadata)


def quantile_iteration(result, q):
    return int(np.searchsorted(np.cumsum(np.asarray(result.weights())), q))


def test_the_run_records_cluster_ids_and_their_parents(two_mode_run) -> None:
    """A label is ``id * folds + clustering``; the two parts of a split take
    new ids and ``mode_tree`` records the id they were split from, so a
    label never names a cluster and, later, a part of it."""
    result = two_mode_run
    folds = result.metadata["folds"]
    labels = np.asarray(result.labels)
    tree = dict(map(tuple, result.metadata["mode_tree"]))
    assert folds == 3 and set(labels[:folds].tolist()) <= set(range(folds))
    assert tree
    assert all(child % folds == parent % folds for child, parent in tree.items())
    assert all(parent < child for child, parent in tree.items())
    minor = np.asarray(result.samples)[:, 0] > 0.55
    for j in range(folds):  # every clustering split the two modes apart
        split = [c for c, parent in tree.items() if parent == j]
        assert len(split) == 2
        sides = [minor[labels == c].mean() for c in split]
        assert min(sides) < 0.02 and max(sides) > 0.98
        # the root's samples all died before those of its parts
        first = min(np.flatnonzero(np.isin(labels, split)))
        assert np.flatnonzero(labels == j).max() < first
    modes = result.modes()
    assert len(modes) == 2 and all(m["tracked"] for m in modes)


def test_modes_survive_late_and_missing_splits(two_mode_run) -> None:
    """The LogGamma failure: clusterings that split two modes late, each at
    its own time, or not at all. The label of the unsplit cluster holds both
    modes; modes() leaves it out because its parts end up apart
    (``mode_tree``), or because its own live points leave a gap between the
    two modes another clustering found."""
    result = two_mode_run
    folds = result.metadata["folds"]
    fold = np.asarray(result.labels) % folds
    minor = np.asarray(result.samples)[:, 0] > 0.55
    weights = np.asarray(result.weights())
    truth = weights[minor].sum()
    index = np.arange(len(fold))
    part = np.where(minor, 2, 1) * folds + fold  # ids 1 (main) and 2 (minor)
    tree = [(uid * folds + j, j) for j in range(folds) for uid in (1, 2)]

    # each clustering splits in the posterior bulk, at its own iteration
    late = np.array([quantile_iteration(result, q) for q in (0.3, 0.5, 0.7)])
    labels = np.where(index < late[fold], fold, part)
    modes = relabelled(result, labels, tree).modes()
    assert len(modes) == 2 and all(m["tracked"] for m in modes)
    assert modes[1]["mass"] == pytest.approx(truth, abs=0.01)

    # one clustering never splits; the others split early
    early = quantile_iteration(result, 0.02)
    labels = np.where((index < early) | (fold == 2), fold, part)
    modes = relabelled(result, labels, tree[:4]).modes()
    assert len(modes) == 2
    assert modes[1]["mass"] == pytest.approx(truth, abs=0.01)

    # no clustering splits: the empty slab between the modes still shows
    modes = relabelled(result, fold).modes()
    assert len(modes) == 2 and not any(m["tracked"] for m in modes)
    assert modes[1]["mass"] == pytest.approx(truth, abs=0.01)


def test_pieces_of_one_mode_are_merged() -> None:
    """Clusters that cut one mode into touching pieces, in every clustering
    at another place and time, are one mode."""

    def gauss3(theta):
        return -0.5 * jnp.sum(((theta - 0.5) / 0.05) ** 2)

    result = NestedSampler(gauss3, lambda u: u, 3, nlive=300).run(1)
    folds = result.metadata["folds"]
    fold = np.asarray(result.labels) % folds
    x = np.asarray(result.samples)
    index = np.arange(len(fold))
    cuts = np.array([0.48, 0.5, 0.53])[fold]
    when = np.array([quantile_iteration(result, q) for q in (0.05, 0.3, 0.6)])[fold]
    part = np.where(x[:, 0] > cuts, 2, 1) * folds + fold
    tree = [(uid * folds + j, j) for j in range(folds) for uid in (1, 2)]
    modes = relabelled(result, np.where(index < when, fold, part), tree).modes()
    assert len(modes) == 1 and modes[0]["tracked"]


def test_empty_slabs_cut_a_lattice_of_modes() -> None:
    from tinyns.result import SLAB_SCORE, _empty_slab, _slab_parts

    rng = np.random.default_rng(0)
    blob = rng.uniform(-1.0, 1.0, (4000, 2))
    blob = blob[np.sum(blob**2, axis=1) < 1.0][:600] * 0.05
    # one convex region: no slab, whatever its shape
    gauss = rng.normal(0.5, 0.05, (600, 2))
    for points in (blob + 0.5, (blob * [8.0, 1.0]) + 0.5, gauss):
        assert _empty_slab(points, 8)[0] < 0.5 * SLAB_SCORE
        assert _slab_parts(points, points, 8).max() == 0
    # six regions on a 3 x 2 lattice, of 100 points each
    centres = np.array([[x, y] for x in (0.2, 0.5, 0.8) for y in (0.3, 0.7)])
    lattice = np.concatenate(
        [blob[i * 100 : (i + 1) * 100] + c for i, c in enumerate(centres)]
    )
    score, axis, cut = _empty_slab(lattice, 8)
    assert score > 3 * SLAB_SCORE
    assert (axis, round(cut, 1)) in ((0, 0.3), (0, 0.6), (1, 0.5))
    part = _slab_parts(lattice, lattice, 8)
    assert part.max() == 5
    assert all(len(np.unique(part[i * 100 : (i + 1) * 100])) == 1 for i in range(6))
    # too few points on one side of a slab: no cut
    assert _slab_parts(lattice[:107], lattice[:107], 8).max() == 0


def loggamma2(theta):
    """Four modes of mass 1/4 (the 2-D LogGamma target of ``bench.targets``)."""
    lg = (theta[0] - jnp.array([1 / 3, 2 / 3])) * 30.0
    nm = (theta[1] - jnp.array([1 / 3, 2 / 3])) * 30.0
    first = jax.scipy.special.logsumexp(lg - jnp.exp(lg))
    return first + jax.scipy.special.logsumexp(-0.5 * nm**2)


def test_four_modes_are_counted() -> None:
    for seed in range(3):
        result = NestedSampler(loggamma2, lambda u: u, 2, nlive=400).run(seed)
        modes = result.modes()
        assert len(modes) == 4
        assert all(0.17 < m["mass"] < 0.33 and not m["unresolved"] for m in modes)
        assert not any("modes" in w for w in result.diagnostics()["warnings"])


def eggbox(theta):
    x = theta * 10.0 * math.pi
    return (2.0 + jnp.cos(x[0] / 2.0) * jnp.cos(x[1] / 2.0)) ** 5


def test_more_modes_than_the_clustering_separates_are_flagged() -> None:
    """The eggbox has 18 modes, which no two-way split of the sampler's
    clustering separates. modes() counts the ones its live points still hold
    apart (15 to 18 of them in 64 runs at nlive 1000) and marks them as not
    tracked; diagnostics() and summary() say that the count is a lower bound
    and that the masses were not balanced."""
    result = NestedSampler(eggbox, lambda u: u, 2, nlive=1000).run(0)
    modes = result.modes()
    assert 14 <= len(modes) <= 18
    assert not any(m["tracked"] for m in modes)
    assert sum(m["mass"] for m in modes) == pytest.approx(1.0)
    # The run is not the same on every CPU in float32, so the bounds hold for
    # any seed. The heaviest mode is one 8% peak with its scatter, or two that
    # stayed merged: 0.10 to 0.20 over 21 runs. The typical urn_sd is that of
    # a mode of 20 to 70 live points left unbalanced (median 0.33 to 0.55 over
    # those runs); a single mode cut late has only its 1 / n term (0.10 and up).
    assert 0.04 < modes[0]["mass"] < 0.3
    urn_sd = sorted(m["urn_sd"] for m in modes)
    assert urn_sd[0] > 0.05 and urn_sd[len(urn_sd) // 2] > 0.25
    warnings = result.diagnostics()["warnings"]
    assert sum("were not separated by the sampler's" in w for w in warnings) == 1
    assert "not tracked" in result.summary() and "warning: " in result.summary()


def test_full_cluster_slots_are_reported(two_mode_run) -> None:
    """``mode_slots_full`` is the fraction of the run with every slot in
    use; with several modes it is a warning that there may be more."""
    from tinyns.modes import C_MAX

    result = two_mode_run
    assert result.diagnostics()["mode_slots_full"] == 0.0
    history = [[0, 1, 1], [result.niter // 2, C_MAX, 2]]
    metadata = {**result.metadata, "mode_history": history}
    full = dataclasses.replace(result, metadata=metadata)
    diagnostics = full.diagnostics()
    assert diagnostics["mode_slots_full"] == pytest.approx(0.5, abs=0.01)
    text = f"the {C_MAX} cluster slots were full for 50%"
    assert any(text in w for w in diagnostics["warnings"])
    assert "cluster slots were full" in full.summary()
    bare = dataclasses.replace(result, metadata=None)
    assert bare.diagnostics()["mode_slots_full"] is None and len(bare.modes()) == 2
