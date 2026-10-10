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
            "fold_mass": [1.0, 1.0, 1.0],
            "fold_sd": 0.0,
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
    use. With more clusters than modes the slots held pieces of the modes
    found; only with as many modes as slots may there be more."""
    from tinyns.modes import C_MAX

    result = two_mode_run
    assert result.diagnostics()["mode_slots_full"] == 0.0
    history = [[0, 1, 1], [result.niter // 2, C_MAX, 2]]
    metadata = {**result.metadata, "mode_history": history}
    full = dataclasses.replace(result, metadata=metadata)
    diagnostics = full.diagnostics()
    assert diagnostics["mode_slots_full"] == pytest.approx(0.5, abs=0.01)
    text = f"the {C_MAX} cluster slots were full for 50%"
    slots = [w for w in diagnostics["warnings"] if text in w]
    assert len(slots) == 1 and f"{C_MAX} clusters for the 2 modes" in slots[0]
    assert "there may be more modes" not in slots[0]
    assert "cluster slots were full" in full.summary()
    many = full._mode_warnings([full.modes()[1]] * C_MAX)
    assert any(text in w and "there may be more modes" in w for w in many)
    bare = dataclasses.replace(result, metadata=None)
    assert bare.diagnostics()["mode_slots_full"] is None and len(bare.modes()) == 2
    assert bare.modes()[0]["fold_sd"] is None


def test_folds_measure_the_scatter_of_the_mode_masses(two_mode_run) -> None:
    """A point and its descendants stay in one fold (its label names the
    clustering of the next fold), so each fold's samples give the modes'
    masses of a run of their own. Where the hop balances the modes the
    folds agree: ``fold_sd`` was 0.03 to 0.16 over 32 runs of a 10-D
    two-mode mixture at nlive 300 and 500, and 0.04 on average at 18-D."""
    result = two_mode_run
    folds = result.metadata["folds"]
    fold = (np.asarray(result.labels) % folds - 1) % folds
    weights = np.asarray(result.weights())
    minor = np.asarray(result.samples)[:, 0] > 0.55
    modes = result.modes()
    assert len(modes) == 2
    for f in range(folds):
        own = weights[fold == f].sum()
        assert modes[1]["fold_mass"][f] == pytest.approx(
            weights[minor & (fold == f)].sum() / own, abs=0.01
        )
        assert sum(m["fold_mass"][f] for m in modes) == pytest.approx(1.0)
    assert np.mean(modes[1]["fold_mass"]) == pytest.approx(modes[1]["mass"], abs=0.02)
    assert modes[0]["fold_sd"] == pytest.approx(modes[1]["fold_sd"])
    assert 0.0 < modes[1]["fold_sd"] < 0.3
    assert "fold sd" in result.summary()


def test_hop_warning_reads_the_recorded_history(two_mode_run) -> None:
    """The warning that the hop balanced the modes only in part comes from
    the run's record: the clustering held many clusters per mode (lumps),
    the hop was off (fewer than two eligible clusters), or it was rarely
    accepted. It quotes the folds' masses."""
    from tinyns import result as result_module

    result = two_mode_run
    diagnostics = result.diagnostics()
    assert diagnostics["mode_lumps"] < result_module.LUMPS_WARN
    assert diagnostics["mode_hop_off"] < result_module.HOP_OFF_WARN
    assert diagnostics["hop_acceptance"] > result_module.HOP_ACCEPT_WARN
    assert not any("inter-mode hop" in w for w in diagnostics["warnings"])
    modes = result.modes()
    late = min(m["isolation_iteration"] for m in modes) + 1
    cases = {
        "held 2.5 clusters per mode": ([[0, 1, 1], [late, 5, 3]], 0.5),
        "was off (fewer than two eligible clusters) for 100%": (
            [[0, 1, 1], [late, 2, 1]], 0.5
        ),
        "accepted 0.1% of its proposals": (result.metadata["mode_history"], 0.001),
    }
    for text, (history, acceptance) in cases.items():
        metadata = {
            **result.metadata, "mode_history": history, "hop_acceptance": acceptance
        }
        warned = dataclasses.replace(result, metadata=metadata).diagnostics()
        hop = [w for w in warned["warnings"] if "only in part" in w]
        assert len(hop) == 1 and text in hop[0], text
        assert "cross-fitted folds gave mode 1 the masses" in hop[0]


def test_tracked_when_the_clustering_separated_the_modes(two_mode_run) -> None:
    """A mode that only the empty-slab pass found is still ``tracked`` when
    the run's labels separated it: here every sample has a label of its own,
    so the labels are all set aside, but each holds one mode only."""
    result = two_mode_run
    folds = result.metadata["folds"]
    own = np.arange(len(result.labels)) * folds + np.asarray(result.labels) % folds
    modes = relabelled(result, own).modes()
    assert len(modes) == 2 and all(m["tracked"] for m in modes)
    warnings = relabelled(result, own).diagnostics()["warnings"]
    assert not any("not separated" in w for w in warnings)


def test_saddle_depth_tells_lumps_from_modes() -> None:
    """Dead points cover every contour, so the likelihood of the samples
    between two separate modes dips; between a mode and a lump of it, it
    does not."""
    from tinyns.result import SADDLE_DEPTH, _saddle_depth

    rng = np.random.default_rng(0)
    d = 6
    centre = np.zeros(d)
    centre[0] = 10.0
    prior = rng.uniform(-5.0, 15.0, (4000, d))  # early samples: anywhere
    post = rng.normal(size=(4000, d))
    post[3000:] += centre  # a second mode of a third of the mass
    u = np.concatenate([prior, post])
    r2, r2c = np.sum(u**2, axis=1), np.sum((u - centre) ** 2, axis=1)
    logl = np.logaddexp(-0.5 * r2, np.log(1 / 3) - 0.5 * r2c)
    weights = np.where(np.arange(len(u)) < len(prior), 1e-6, 1.0)
    near = r2 < r2c
    assert _saddle_depth(u, logl, weights, near, ~near) > 4 * SADDLE_DEPTH
    one = near & (r2 < 30)  # a single mode, cut in two
    lump = u[:, 1] >= 1.0
    assert _saddle_depth(u, logl, weights, one & ~lump, one & lump) < 1.0


def test_modes_with_thin_arms_are_reported_with_their_scatter() -> None:
    """The needle target of ``bench.targets`` (each mode a core with two
    thin arms) against the same two modes without arms. The clustering cuts
    the needle's modes into lumps and the hop balances them only in part,
    which the warning says. The plain modes are not cut into lumps; the
    warning may still say that the hop was off, which happens on some
    trajectories (on JAX 0.4.31 the hop never ran in 2 of 8 runs at nlive
    500, and the minor mode was lost), but never that the modes were lumps."""
    from bench.targets import build_needle, build_two_mode, mixture_target

    from tinyns import result as result_module

    for spec, arms in ((build_needle(10), True), (build_two_mode(10, 0.2), False)):
        target = mixture_target("needle" if arms else "plain", spec)
        sampler = NestedSampler(target.loglike, target.prior_transform, 10, nlive=500)
        diagnostics = sampler.run(0).diagnostics()
        modes = diagnostics["modes"]
        warned = [w for w in diagnostics["warnings"] if "only in part" in w]
        if arms:
            assert len(modes) >= 2
            assert sum(m["mass"] for m in modes[1:]) == pytest.approx(0.2, abs=0.1)
            assert diagnostics["mode_lumps"] > 1.5 and len(warned) == 1
            assert "it cut modes into lumps" in warned[0]
        else:
            assert diagnostics["mode_lumps"] < 1.2
            assert not any("slots were full" in w for w in diagnostics["warnings"])
            acceptance = diagnostics["hop_acceptance"]
            off = diagnostics["mode_hop_off"] >= result_module.HOP_OFF_WARN or (
                acceptance is not None and acceptance < result_module.HOP_ACCEPT_WARN
            )
            assert off or not warned


@pytest.mark.slow
def test_modes_that_are_not_convex_18d() -> None:
    """The gate of the needle and Cauchy targets of ``bench.targets`` against
    the same two modes without arms (18-D, nlive 1000, 8 batched runs each).
    Over 16 single runs each: the needle reported 2 modes and the warning
    every time (minor mass 0.198, logit sd 0.19 over the runs; 0.57 without
    the hop), the Cauchy target 2 modes in 14 runs and the warning in all 14
    (0.192, sd 0.32; truth 0.203), the plain modes 2 and no warning (sd
    0.019). In float32 two of these 8 batched plain runs report one mode,
    as before this check existed."""
    from bench.targets import build_two_mode, get_target, mixture_target

    keys = jax.random.split(jax.random.key(0), 8)
    plain = mixture_target("plain_d18", build_two_mode(18, 0.2))
    for target in (get_target("needle_d18"), get_target("cauchy_d18"), plain):
        sampler = NestedSampler(target.loglike, target.prior_transform, 18)
        found, warned, minor = 0, 0, []
        for result in sampler.run(keys):
            diagnostics = result.diagnostics()
            modes = diagnostics["modes"]
            if len(modes) == 2:
                found += 1
                minor.append(modes[1]["mass"])
            warned += any("only in part" in w for w in diagnostics["warnings"])
        logit = np.log(np.array(minor) / (1 - np.array(minor)))
        print(target.name, found, warned, np.mean(minor), np.std(logit))
        truth = target.mode_mass[1]
        assert abs(np.mean(minor) - truth) < 0.05
        if target is plain:
            assert found >= 6 and warned == 0 and np.std(logit) < 0.1
        else:
            assert found >= 6 and warned >= found
