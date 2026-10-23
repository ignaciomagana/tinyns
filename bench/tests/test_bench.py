"""Harness tests: target truths, every installed adapter on a 2-D Gaussian,
run.py, the head-to-head report and the validation gate."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from bench import adapters
from bench.targets import (
    BENCH_TARGETS,
    EGG_LOGZ,
    FUNNEL_LOGZ,
    ROSEN_LOGZ,
    _trapezoid,
    eggbox_logz_quadrature,
    funnel_logz_quadrature,
    get_target,
    rosenbrock_logz_quadrature,
)


def _grid_logz(target, n=1501):
    """Brute-force 2-D trapezoid evidence."""
    import jax

    xs = [np.linspace(lo, hi, n) for lo, hi in zip(target.lo, target.hi, strict=True)]
    X, Y = np.meshgrid(*xs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    ll = np.asarray(jax.jit(jax.vmap(target.loglike))(pts), float).reshape(n, n)
    m = ll.max()
    z = _trapezoid(_trapezoid(np.exp(ll - m), xs[1], axis=1), xs[0])
    return m + math.log(z) - target.log_prior_volume


def test_every_bench_target_builds():
    import jax
    from bench.validate import CASES

    for name in sorted(set(BENCH_TARGETS) | set(CASES)):
        t = get_target(name)
        x = t.prior_transform_np(np.full(t.ndim, 0.5))
        assert np.isfinite(float(jax.jit(t.loglike)(x))), name
        assert t.logz is not None and np.isfinite(t.logz), name
        if t.mode_mass is not None:
            assert abs(sum(t.mode_mass) - 1.0) < 1e-9, name
            r = np.asarray(t.responsibility(x))
            assert r.shape == (len(t.mode_mass),) and abs(r.sum() - 1.0) < 1e-5, name


@pytest.mark.parametrize("name", ["gauss_d2", "loggamma_d2", "rosen_d2"])
def test_2d_logz_matches_grid(name):
    t = get_target(name)
    assert abs(_grid_logz(t, n=3001) - t.logz) < 2e-3


def test_cached_quadratures():
    assert abs(rosenbrock_logz_quadrature(2, n=4001) - ROSEN_LOGZ[2]) < 1e-5
    assert abs(funnel_logz_quadrature(10, n=20001) - FUNNEL_LOGZ[10]) < 1e-5
    assert abs(eggbox_logz_quadrature(1001) - EGG_LOGZ) < 1e-5


def test_rosenbrock_lobe_mass():
    """The 10-D Rosenbrock target has a second lobe at x_1 = -1 behind a thin
    bridge (the gate accepts one mode or two); at d = 4 and 2 the arm at
    x_1 < 0 is part of the one mode."""
    from bench.targets import rosenbrock_x1_mass

    assert rosenbrock_x1_mass(10, -5.0, 0.0, n=4001) == pytest.approx(0.0252, abs=2e-4)
    assert rosenbrock_x1_mass(10, -0.3, 0.3, n=4001) == pytest.approx(0.0083, abs=2e-4)
    assert rosenbrock_x1_mass(4, -5.0, 0.0, n=4001) == pytest.approx(0.0626, abs=2e-4)
    assert rosenbrock_x1_mass(4, -0.3, 0.3, n=4001) == pytest.approx(0.0676, abs=2e-4)
    assert rosenbrock_x1_mass(2, -5.0, 5.0, n=2001) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", ["sepW_d4", "connW_d10", "mix3_d10", "needle_d10"])
def test_mixture_mode_masses_are_exact(name):
    """E_post[responsibility] = mixture weights, by exact draws from the mixture
    (the needle's modes are two components each)."""
    import jax
    from bench.targets import mixture_spec

    t = get_target(name)
    d = t.ndim
    spec = mixture_spec(name)
    rng = np.random.default_rng(1)
    n = 20000
    comp = rng.choice(len(spec["weights"]), size=n, p=spec["weights"])
    x = np.empty((n, d))
    for k, (mu, cov) in enumerate(zip(spec["means"], spec["covs"], strict=True)):
        sel = comp == k
        x[sel] = rng.multivariate_normal(mu, cov, size=sel.sum())
    r = np.asarray(jax.vmap(t.responsibility)(x), float)
    mean, se = r.mean(axis=0), r.std(axis=0) / math.sqrt(n)
    assert np.all(np.abs(mean - np.array(t.mode_mass)) < 5 * se + 1e-4)


def test_cauchy_evidence_and_mode_masses_are_exact():
    """Exact draws from the two products of Cauchy and normal factors: the
    fraction inside the box is the evidence times the box volume, and the
    responsibilities of the draws inside average to the mode masses."""
    import jax

    t = get_target("cauchy_d10")
    d, half = t.ndim, t.hi[0]
    rng = np.random.default_rng(2)
    n = 40000
    minor = rng.random(n) < 0.2
    scale = np.where(minor, 0.8, 1.0)[:, None]
    z = np.concatenate(
        [rng.standard_cauchy((n, 4)), rng.standard_normal((n, d - 4))], axis=1
    )
    x = z * scale
    x[minor, 4:7] += 12.0 / math.sqrt(3)
    inside = np.all(np.abs(x) < half, axis=1)
    p = math.exp(t.logz + t.log_prior_volume)
    assert abs(inside.mean() - p) < 5 * math.sqrt(p * (1 - p) / n)
    r = np.asarray(jax.vmap(t.responsibility)(x[inside]), float)
    se = r.std(axis=0) / math.sqrt(len(r))
    assert np.all(np.abs(r.mean(axis=0) - np.array(t.mode_mass)) < 5 * se + 1e-4)


def test_mixture_targets_have_four_dimensions():
    for name in ("sepW_d6", "sepWtw_d8", "needle_d6", "connMbox_d20"):
        with pytest.raises(ValueError, match=r"d in \[4, 10, 18, 32\]"):
            get_target(name)


def test_box_mixture_is_the_cube_mixture_in_other_coordinates():
    """``sepWbox_d4`` has the prior box ``BOX``, and its likelihood and
    responsibility at x are those of ``sepW_d4`` at u = (x - lo) / (hi - lo)."""
    import jax
    from bench.targets import BOX, mixture_spec

    cube, box = get_target("sepW_d4"), get_target("sepWbox_d4")
    assert np.all(box.lo == BOX[0]) and np.all(box.hi == BOX[1])
    assert box.logz == cube.logz == 0.0 and box.mode_mass == cube.mode_mass
    spec = mixture_spec("sepW_d4")
    rng = np.random.default_rng(0)
    u = np.concatenate([rng.multivariate_normal(mu, cov, 32)
                        for mu, cov in zip(spec["means"], spec["covs"], strict=True)])
    x = box.prior_transform_np(u)
    for f in ("loglike", "responsibility"):
        a = np.asarray(jax.vmap(getattr(cube, f))(u), float)
        b = np.asarray(jax.vmap(getattr(box, f))(x), float)
        assert np.allclose(a, b, rtol=1e-4, atol=1e-4), f


def test_ndtri_target_logz_matches_quadrature():
    """The coordinates are independent: the 1-D evidence by quadrature of the
    target's own likelihood in the unit cube, ndtri included."""
    import jax

    one = get_target("ndtri_d1")
    u = np.linspace(0.0, 1.0, 400001)[1:-1]
    ll = np.asarray(jax.jit(jax.vmap(one.loglike))(u[:, None]), float)
    m = ll.max()
    assert abs(m + math.log(_trapezoid(np.exp(ll - m), u)) - one.logz) < 1e-4
    assert get_target("ndtri_d6").logz == pytest.approx(6 * one.logz, rel=1e-12)


DEFAULT_SPECS = sorted(adapters.REGISTRY)


@pytest.mark.parametrize("name", DEFAULT_SPECS)
def test_adapter_on_2d_gaussian(name):
    reason = adapters.unavailable_reason(name)
    if reason:
        pytest.skip(reason)
    t = get_target("gauss_d2")
    opts = {"n_live": "200"} if name == "nautilus" else {}
    out = adapters.load(name).run(
        t, 0, dict(nlive=100, dlogz=0.1, variant="default", opts=opts)
    )
    for key in ("samples", "logz", "ncall", "wall_s", "sampler_version"):
        assert key in out
    assert out["samples"].shape[1] == 2 and out["ncall"] > 0
    tol = 5 * max(out["logzerr"] or 0.3, 0.2)
    assert abs(out["logz"] - t.logz) < tol, (out["logz"], t.logz, out["logzerr"])


def _any_installed():
    for name in ("tinyns_v1", "tinyns_v02", "blackjax_nss"):
        if adapters.unavailable_reason(name) is None:
            return name
    pytest.skip("no JAX sampler installed")


def test_run_cli_writes_schema_records_and_summary(tmp_path):
    from bench import run, summarize

    name = _any_installed()
    out = tmp_path / "r.jsonl"
    assert (
        run.main(
            [
                "--sampler",
                name,
                "--target",
                "sepW_d4",
                "--seeds",
                "0",
                "--nlive",
                "100",
                "--out",
                str(out),
                "--timeout",
                "600",
            ]
        )
        == 0
    )
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 1
    rec = recs[0]
    assert set(run.RECORD_KEYS) <= set(rec)
    assert rec["schema"] == "tinyns-bench-1" and rec["status"] == "ok", rec["error"]
    assert len(rec["mode_mass"]) == 2 and abs(sum(rec["mode_mass"]) - 1) < 1e-6
    md, rows = summarize.report(summarize.load([out]))
    assert len(rows) == 1 and rows[0]["n_ok"] == 1 and rows[0]["lost"] is not None
    assert f"| sepW_d4 | {name} | {rows[0]['hw']} | 100 | 1/1 |" in md


def test_run_cli_records_timeouts(tmp_path):
    from bench import run

    name = _any_installed()
    out = tmp_path / "r.jsonl"
    run.main(
        [
            "--sampler",
            name,
            "--target",
            "gauss_d2",
            "--seeds",
            "0",
            "--out",
            str(out),
            "--timeout",
            "0.01",
        ]
    )
    rec = json.loads(out.read_text())
    assert rec["status"] == "timeout" and rec["logz"] is None


def test_parse_seeds():
    from bench.run import parse_seeds

    assert parse_seeds("0-3,7") == [0, 1, 2, 3, 7]


def test_bakeoff_runner_and_summary(tmp_path) -> None:
    """One tiny cell per arm (N and B_ell), then the decision tables."""
    import jax
    from bench.bakeoff import emit, summarize
    from bench.bakeoff import run as bakeoff

    out = tmp_path / "bakeoff.jsonl"
    for arm in ("N", "B_ell"):
        bakeoff.main([
            "--target", "sepW_d4", "--arm", arm, "--nlive", "100", "--k", "10",
            "--seeds", "0-1", "--out", str(out),
            "--x64" if jax.config.jax_enable_x64 else "--no-x64",
        ])
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 4 and {r["arm"] for r in recs} == {"N", "B_ell"}
    for r in recs:
        assert abs(sum(r["mode_mass"]) - 1) < 1e-6
        assert len(r["minor_live"]) == 1 and r["isolation_niter"][0] > 0
    assert all(r["hop_tries"] > 0 for r in recs if r["arm"] == "B_ell")
    assert all(r["hop_tries"] == 0 for r in recs if r["arm"] == "N")
    assert all(abs(sum(m["mass"] for m in r["modes"]) - 1) < 1e-6 for r in recs)
    report = tmp_path / "report.md"
    summarize.main([str(out), "--out", str(report)])
    text = report.read_text()
    assert "## Decision" in text and "| sepW_d4 | 100 | 10 | B_ell |" in text
    assert len(list(emit.cells())) == 66


def _record(sampler, target, seed, dz, *, err=0.1, ncall=1000, wall=10.0, gpu=None,
            status="ok", mass=None, opts=None, truth_mass=None):
    """A minimal ``tinyns-bench-1`` record."""
    ok = status == "ok"
    return dict(
        schema="tinyns-bench-1", sampler=sampler, target=target, seed=seed,
        ndim=int(target.rsplit("_d", 1)[1]), status=status,
        truth=dict(logz=0.0, mode_mass=truth_mass),
        config=dict(cli=dict(nlive=500, dlogz=0.1, opts=opts or {})),
        hw=dict(gpu=gpu, cpu_model="cpu", cpu_threads=4),
        logz=dz if ok else None, logzerr=err if ok else None,
        ncall=ncall if ok else None, wall_s=wall, compile_s=None,
        mode_mass=mass if ok else None,
    )


def test_report_merges_files_and_ranks_the_reference(tmp_path):
    from bench import summarize

    rng = np.random.default_rng(0)
    truth = [0.94, 0.06]
    gpu, cpu = [], []
    for seed in range(20):
        noise = 0.1 * rng.normal(size=4)
        gpu.append(_record("ref", "gauss_d2", seed, noise[0], gpu="NVIDIA H100 80GB"))
        gpu.append(_record("ref", "sepW_d4", seed, noise[1], gpu="NVIDIA H100 80GB",
                           mass=[0.94, 0.06], truth_mass=truth))
        # twice the calls and five times the wall time, on a CPU
        cpu.append(_record("slow", "gauss_d2", seed, noise[2], ncall=2000, wall=50.0))
        # loses the minor mode
        cpu.append(_record("slow", "sepW_d4", seed, noise[3], mass=[1.0, 0.0],
                           truth_mass=truth))
        # a biased sampler, one with no error bar, one that times out, one with --opt
        cpu.append(_record("biased", "gauss_d2", seed, 1.0 + noise[0]))
        cpu.append(_record("noerr", "gauss_d2", seed, 0.01 * noise[1], err=None))
        cpu.append(_record("stuck", "gauss_d2", seed, 0.0, status="timeout", wall=99.0))
        cpu.append(_record("slow", "gauss_d2", seed, noise[1], opts={"walks": "9"}))
    files = []
    for name, recs in (("gpu", gpu), ("cpu", cpu)):
        files.append(tmp_path / f"{name}.jsonl")
        files[-1].write_text("".join(json.dumps(r) + "\n" for r in recs))

    groups = summarize.load(files)
    rows = {(r["sampler"], r["target"]): r for r in summarize.summarize(groups)}
    assert len(rows) == 8
    assert rows[("ref", "gauss_d2")]["hw"] == "GPU H100"
    assert rows[("slow", "gauss_d2")]["hw"] == "CPU 4c"
    assert rows[("ref", "gauss_d2")]["accurate"]
    assert rows[("ref", "gauss_d2")]["in3"] == 1
    assert rows[("biased", "gauss_d2")]["why_not"] == "Z"
    assert rows[("slow", "sepW_d4")]["why_not"] == "M"
    assert rows[("slow", "sepW_d4")]["lost"] == 1.0
    assert rows[("stuck", "gauss_d2")]["why_not"] == "T"
    assert rows[("stuck", "gauss_d2")]["wall_is_timeout"]
    assert "scat_over_err" not in rows[("noerr", "gauss_d2")]
    assert rows[("noerr", "gauss_d2")]["accurate"]
    assert ("slow [walks=9]", "gauss_d2") in rows

    ratios, ranks = summarize.headline(list(rows.values()), "ref")
    slow = next(x for x in ratios if (x["sampler"], x["family"]) == ("slow", "gauss"))
    assert slow["common"] == 1
    assert slow["ncall"] == pytest.approx(2.0) and slow["wall_s"] == pytest.approx(5.0)
    a, b = rows[("ref", "gauss_d2")], rows[("slow", "gauss_d2")]
    assert slow["ncall_eq"] == pytest.approx(2.0 * (b["rms"] / a["rms"]) ** 2)
    lost = next(x for x in ratios if (x["sampler"], x["family"]) == ("slow", "sepW"))
    assert lost["common"] == 0 and lost["ncall"] is None
    gauss = next(r for r in ranks if r["family"] == "gauss")
    # only "noerr" (the same calls at a hundredth of the rms) is cheaper at equal rms
    assert gauss["rank_ncall"] == 2 and gauss["of"] == 4 and gauss["accurate"] == 1

    out, csv_path, json_path = (tmp_path / n for n in ("r.md", "c.csv", "c.json"))
    args = [str(f) for f in files] + ["--out", str(out), "--csv", str(csv_path)]
    summarize.main(args + ["--json", str(json_path), "--reference", "ref"])
    md = out.read_text()
    assert "| gauss | slow | CPU 4c | 1/1 | 2 | 5 |" in md
    assert "| gauss_d2 | stuck | CPU 4c | 500 | 0/20 (20T 0E) |" in md and ">99" in md
    assert len(csv_path.read_text().splitlines()) == 9
    assert len(json.loads(json_path.read_text())["cells"]) == 8
    # without the reference the report still comes out, with a note
    md, _ = summarize.report(groups)
    assert "No `tinyns_v1` records" in md and "## Cells" in md


def test_report_times_a_cold_start_and_filters_on_the_commit(tmp_path):
    from bench import summarize

    new, old = "a" * 40, "b" * 40
    recs = []
    for seed in range(6):
        # the first run compiles (20 s of 30); the others load the cache (1 s of 11)
        r = _record("tinyns_v1", "gauss_d2", seed, 0.01 * seed, gpu="NVIDIA H100 80GB",
                    wall=30.0 if seed == 0 else 11.0)
        r.update(compile_s=20.0 if seed == 0 else 1.0, git_sha=new,
                 ts=f"2026-01-01T00:00:{seed:02d}",
                 sampler_version="tinyns 1.0.0 (/some/where/tinyns)")
        recs.append(r)
        # the same sampler at an older commit, and its k1 variant: biased by 5
        for spec in ("tinyns_v1", "tinyns_v1:k1"):
            stale = _record(spec, "gauss_d8", seed, 5.0, gpu="NVIDIA H100 80GB")
            recs.append(dict(stale, git_sha=old, ts="2025-01-01T00:00:00"))
        # another sampler is not filtered, whatever its commit; a CPU cell
        # has no cold start
        recs.append(dict(_record("other", "gauss_d2", seed, 0.0), git_sha=old))
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))

    rows = summarize.summarize(summarize.load([path]))
    assert len(rows) == 4
    dropped = {}
    groups = summarize.load([path], summarize.parse_sha(["tinyns_v1=aaaaaaa"]), dropped)
    assert dropped == {"tinyns_v1": 6, "tinyns_v1:k1": 6}
    rows = {r["sampler"]: r for r in summarize.summarize(groups)}
    assert set(rows) == {"tinyns_v1", "other"}
    v1 = rows["tinyns_v1"]
    assert v1["wall_s"] == 11.0 and v1["compile_s"] == 1.0
    assert v1["run_s"] == 10.0 and v1["first_s"] == 30.0
    assert rows["other"]["first_s"] is None and rows["other"]["run_s"] is None

    md, _ = summarize.report(groups, dropped=dropped)
    assert "Left out by `--sha`: 6 records of `tinyns_v1` from" in md
    # the samplers table drops the import path from the version
    assert "| tinyns_v1 | tinyns 1.0.0 | aaaaaaa | GPU H100 | 1 | 6 | 0 | 0 |" in md
    assert "/some/where" not in md
    assert "| 11 | 1.0 | 10.0 | 30 | yes |" in md
    with pytest.raises(SystemExit):
        summarize.parse_sha(["tinyns_v1"])
    out = tmp_path / "r.md"
    summarize.main([str(path), "--sha", "tinyns_v1=" + new[:7], "--out", str(out)])
    assert "gauss_d8" not in out.read_text()


def test_report_keeps_cells_of_different_nlive_apart(tmp_path):
    from bench import summarize

    rng = np.random.default_rng(1)
    recs = []
    for seed in range(20):
        noise = 0.1 * rng.normal(size=3)
        recs.append(_record("ref", "gauss_d2", seed, noise[0]))
        # the reference at a larger nlive: twice the calls, and biased
        big = _record("ref", "gauss_d2", seed, 1.0 + noise[1], ncall=2000)
        big["config"]["cli"]["nlive"] = 4000
        recs.append(big)
        recs.append(_record("other", "gauss_d2", seed, noise[2], ncall=3000))
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows = summarize.summarize(summarize.load([path]))

    table = summarize.accurate_markdown(rows)
    assert "| ref (nlive 500) | 1/1 |" in table
    assert "| ref (nlive 4000) | 0/1 |" in table
    assert "| other | 1/1 |" in table

    # every reference cell pairs with the competitor: the biased one is not common
    ratios, ranks = summarize.headline(rows, "ref")
    assert ranks[0]["cells"] == 2 and ranks[0]["accurate"] == 1
    assert ratios[0]["ran"] == 2 and ratios[0]["common"] == 1
    ratios, ranks = summarize.headline(rows, "ref", nlive={500})
    assert ranks[0]["cells"] == 1 and ratios[0]["ran"] == 1
    assert ratios[0]["ncall"] == pytest.approx(3.0)

    out = tmp_path / "r.md"
    args = [str(path), "--reference", "ref", "--out", str(out)]
    summarize.main(args + ["--reference-nlive", "500"])
    md = out.read_text()
    assert "Only the `ref` cells at nlive 500 enter this section." in md
    assert "| gauss | 1/1 | " in md and "| gauss | 1/2 | " not in md
    assert "| gauss_d2 | ref | CPU 4c | 4000 | 20/20 |" in md  # still a cell


def test_validation_gate_criteria():
    from bench import validate

    good = dict(dz_mean=0.02, dz_se=0.02, scat_over_err=1.0, seeds=16,
                not_converged=0, unresolved=0)
    assert validate.judge(good) == []
    assert validate.judge(dict(good, dz_mean=0.07)) == ["bias"]
    assert validate.judge(dict(good, scat_over_err=2.5)) == ["scatter"]
    assert validate.judge(dict(good, scat_over_err=0.4)) == ["scatter"]
    # 1.5 is outside [0.75, 1.3] but within the noise of 16 seeds, not of 400
    assert validate.judge(dict(good, scat_over_err=1.5)) == []
    assert validate.judge(dict(good, scat_over_err=1.5, seeds=400)) == ["scatter"]
    assert validate.judge(dict(good, unresolved=1)) == ["flags"]
    assert validate.judge(dict(good, not_converged=1)) == ["flags"]
    assert validate.judge(dict(good, post_t=3.9, post_bound=4.4)) == []
    assert validate.judge(dict(good, post_t=4.5, post_bound=4.4)) == ["posterior"]
    modes = dict(good, lost=0.0, logit_z=1.0, logit_sd=0.1)
    assert validate.judge(modes) == []
    for bad in (dict(lost=0.1), dict(logit_z=3.5), dict(logit_sd=0.5),
                dict(max_logit_sd=0.08)):
        assert validate.judge(dict(modes, **bad)) == ["modes"]
    # the hop: tried and accepting in at least 90% of the runs
    hop = dict(modes, hop_acceptance=[0.7] * 15 + [None])
    assert validate.judge(hop) == []
    two_low = [0.7] * 14 + [0.01, None]
    assert validate.judge(dict(hop, hop_acceptance=two_low)) == ["hop"]
    # the mode count: at most 10% of the runs may be wrong
    counted = dict(good, modes_min=4, modes_max=4, nmodes=[4] * 15 + [3],
                   lower_bound=[False] * 16)
    assert validate.judge(counted) == []
    assert validate.judge(dict(counted, nmodes=[4] * 14 + [3, 5])) == ["count"]
    # fewer modes than the truth pass only with the lower-bound flag
    egg = dict(good, modes_min=8, modes_max=18, nmodes=[18] * 8 + [15] * 8,
               lower_bound=[False] * 8 + [True] * 8, small_mode=True, unresolved=9)
    assert validate.judge(egg) == []
    assert validate.judge(dict(egg, lower_bound=[False] * 16)) == ["count"]
    assert validate.judge(dict(egg, nmodes=[18] * 8 + [19] * 8)) == ["count"]
    # one mode or two are both right for rosen_d10, with no flag needed
    lobe = dict(good, modes_min=1, modes_max=2, nmodes=[1] * 12 + [2] * 4,
                lower_bound=[False] * 16, small_mode=True, unresolved=4)
    assert validate.judge(lobe) == []
    assert validate.judge(dict(lobe, small_mode=False)) == ["flags"]
    # every case names a target, and the quick tier is a subset of the full one
    for name, case in validate.CASES.items():
        assert get_target(name).logz is not None and case.full is not None
        assert case.quick is None or case.quick <= case.full
        assert 1 <= case.modes[0] <= case.modes[1]


def test_gate_eggbox_floor_admits_a_correct_count():
    """At nlive 500 a correct sampler reports 4 to 18 eggbox modes, flagged as
    a lower bound below 18 (960 runs); the floor stays above 2, which
    catches a sampler that reports one or two modes."""
    from bench import validate

    lo, hi = validate.CASES["eggbox_d2"][3][:2]
    good = dict(dz_mean=0.0, dz_se=0.02, scat_over_err=1.0, seeds=16,
                not_converged=0, unresolved=9, small_mode=True)
    egg = dict(good, modes_min=lo, modes_max=hi, nmodes=[14] * 14 + [4, 7],
               lower_bound=[True] * 16)
    assert validate.judge(egg) == []
    assert validate.judge(dict(egg, nmodes=[14] * 14 + [2, 1])) == ["count"]


def test_gate_catches_a_disabled_hop():
    """With the hop off the sepW minor-mode logit sd is 0.25 to 0.37 at the
    gate's nlive, against 0.02 to 0.07 with it: under the accuracy floor of
    0.4, but over the case's own limit (scaled by 1 / sqrt(nlive) when
    --nlive overrides the case)."""
    from bench import validate

    row = dict(dz_mean=0.0, dz_se=0.02, scat_over_err=1.0, seeds=16,
               not_converged=0, unresolved=0, lost=0.0, logit_z=1.0,
               max_logit_sd=0.12)
    assert validate.judge(dict(row, logit_sd=0.048)) == []
    assert validate.judge(dict(row, logit_sd=0.25)) == ["modes"]
    assert validate.judge(dict(row, logit_sd=0.25, hop_acceptance=[None] * 16)) == [
        "modes", "hop"
    ]
    for name in ("sepW_d10", "sepW_d18", "sepWbox_d4"):
        assert 0.05 < validate.CASES[name].logit_sd <= 0.15, name


def _x64_flag():
    import jax

    return "--x64" if jax.config.jax_enable_x64 else "--no-x64"


def _fake_run_case(ran, dz_se=0.1):
    """A stand-in for ``validate.run_case`` that records (case, nlive)."""

    def run_case(name, nlive, *args, **kwargs):
        ran.append((name, nlive))
        n = 4
        return dict(case=name, ndim=1, nlive=nlive, walks=1, seeds=n, wall_s=0.0,
                    ncall=1.0, logz=[0.0] * n, logzerr=[1.0] * n, not_converged=0,
                    unresolved=0, nmodes=[1] * n, lower_bound=[False] * n,
                    dz_mean=0.0, dz_se=dz_se, scat_over_err=1.0, failed=[])

    return run_case


def test_gate_overrides_keep_to_the_tier(monkeypatch, capsys):
    """--nlive changes the cases' nlive, not the tier's cases; --cases runs a
    case outside the tier (at its full-tier nlive), and warns when an nlive
    leaves a mode too few live points to be resolved."""
    from bench import validate

    ran = []
    monkeypatch.setattr(validate, "run_case", _fake_run_case(ran))
    assert validate.main(["--tier", "quick", "--nlive", "300", _x64_flag()]) == 0
    quick = [name for name, case in validate.CASES.items() if case[0] is not None]
    assert ran == [(name, 300) for name in quick]
    assert "nlive 300 in every case" in capsys.readouterr().out
    ran.clear()
    validate.main(["--cases", "sepW_d18", _x64_flag()])
    assert ran == [("sepW_d18", validate.CASES["sepW_d18"][1])]
    ran.clear()
    validate.main(["--cases", "sepW_d10", "--nlive", "500", _x64_flag()])
    assert ran == [("sepW_d10", 500)]
    assert "expect a FAIL on flags" in capsys.readouterr().err


def test_gate_degenerate_inputs_and_the_record(monkeypatch, capsys, tmp_path):
    """Identical runs (se 0) print no z instead of dividing by 0; one seed is
    rejected before any run; the JSON is written before the table, with the
    commit the gate ran at."""
    from bench import run, validate

    real = validate.run_case
    with pytest.raises(ValueError, match="at least 2 seeds"):
        real("gauss_d2", 100, seeds=1)
    ran = []
    monkeypatch.setattr(validate, "run_case", _fake_run_case(ran, dz_se=0.0))
    out = tmp_path / "gate.json"
    assert validate.main(["--cases", "gauss_d2", _x64_flag(), "--json", str(out)]) == 0
    table = capsys.readouterr().out
    assert "| +0.000 ± 0.000 | - |" in table
    meta = json.loads(out.read_text())
    assert meta["git_sha"] == run.git_sha() and meta["seeds"] == 16
    assert f"at {run.git_sha() or 'an unknown commit'}," in table
    with pytest.raises(SystemExit):
        validate.main(["--cases", "gauss_d2", "--seeds", "1", _x64_flag()])
    assert ran == [("gauss_d2", 500)]

    def broken(rows):
        raise RuntimeError("the table")

    monkeypatch.setattr(validate, "to_markdown", broken)
    out.unlink()
    with pytest.raises(RuntimeError):
        validate.main(["--cases", "gauss_d2", _x64_flag(), "--json", str(out)])
    assert json.loads(out.read_text())["cases"][0]["case"] == "gauss_d2"


def test_git_sha_flags_a_modified_sampler(tmp_path, monkeypatch):
    """``-dirty`` when a tracked file under src, bench or pyproject.toml
    changed; untracked files and other files do not count."""
    import shutil
    import subprocess

    from bench import run

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    files = ("src/tinyns/core.py", "bench/run.py", "pyproject.toml", "README.md")
    for f in files:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text("x = 1\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
           "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "c"], check=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    monkeypatch.setattr(run, "ROOT", repo)
    assert run.git_sha() == head
    for f in files[:3]:
        (repo / f).write_text("x = 2\n")
        assert run.git_sha() == head + "-dirty", f
        subprocess.run(git + ["checkout", "-q", "--", f], check=True)
    (repo / "README.md").write_text("x = 2\n")
    (repo / "bench" / "results.jsonl").write_text("{}\n")
    assert run.git_sha() == head
    monkeypatch.setattr(run, "ROOT", tmp_path)  # not a checkout
    assert run.git_sha() is None


def test_report_commit_filter_keeps_modified_builds_apart(tmp_path):
    """--sha SAMPLER=SHA keeps the records written at SHA from a clean
    checkout; SHA-dirty keeps those of a modified one."""
    from bench import summarize

    sha = "a" * 40
    recs = [dict(_record("tinyns_v1", "gauss_d2", seed, 0.0), git_sha=sha)
            for seed in range(3)]
    recs += [dict(_record("tinyns_v1", "gauss_d2", seed, 0.0), git_sha=sha + "-dirty")
             for seed in range(3, 5)]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    for want, kept in (("aaaaaaa", 3), ("aaaaaaa-dirty", 2), ("b", 0)):
        dropped = {}
        groups = summarize.load([path], {"tinyns_v1": want}, dropped)
        assert sum(len(g) for g in groups.values()) == kept, want
        assert dropped.get("tinyns_v1", 0) == 5 - kept, want
    md, _ = summarize.report(summarize.load([path]))
    assert "| aaaaaaa, aaaaaaa-dirty |" in md


def test_gate_posterior_criterion():
    """Exact draws from a gate Gaussian pass; the same draws left in the unit
    cube (the prior transform not applied) fail by far. The bound uses
    Student's t."""
    from bench import validate

    t = get_target("gauss_d10")
    mean, cov = np.array(t.meta["mean"]), np.array(t.meta["cov"])
    rng = np.random.default_rng(0)
    runs = [rng.multivariate_normal(mean, cov, 4000) for _ in range(16)]
    logwt = np.zeros(4000)

    def stats(samples):
        offsets = [validate.posterior_offsets(x, logwt, mean, cov) for x in samples]
        return validate.posterior_stats(*zip(*offsets, strict=True))

    good = stats(runs)
    assert good["post_t"] < good["post_bound"]
    assert stats([(x - t.lo) / (t.hi - t.lo) for x in runs])["post_t"] > 100
    assert validate.t_tail(2.131450, 15) == pytest.approx(0.05, abs=1e-5)
    assert validate.t_tail(2.656145, 63) == pytest.approx(0.01, abs=1e-5)
    assert validate.t_bound(1, 10**6) == pytest.approx(3.0, abs=1e-3)
    assert good["post_bound"] == pytest.approx(validate.t_bound(20, 15))


def test_validation_gate_runs_on_tiny_settings(tmp_path, capsys, monkeypatch):
    """The quick tier's code path: batched seeds, the table, the JSON, the exit
    code; a mixture's mode and hop columns, and a hop switched off."""
    import functools

    from bench import validate

    import tinyns.api

    if adapters.unavailable_reason("tinyns_v1"):
        pytest.skip("tinyns v1 is not installed")
    out = tmp_path / "gate.json"
    args = ["--cases", "gauss_d2", "--nlive", "100", "--seeds", "3"]
    code = validate.main(args + [_x64_flag(), "--json", str(out)])
    table = capsys.readouterr().out
    case = json.loads(out.read_text())["cases"][0]
    assert code == (1 if case["failed"] else 0)
    assert len(case["logz"]) == 3 and case["nlive"] == 100 and case["walks"] == 25
    assert "| gauss_d2 | 100 | 25 | 3 |" in table and "cases pass" in table
    assert abs(case["dz_mean"]) < 1.0 and case["unresolved"] == 0
    assert case["nmodes"] == [1, 1, 1] and "| 1 (3/3 ok) |" in table
    assert np.shape(case["post_mean"]) == (3, 2) and case["post_bound"] > 3
    assert f"| {case['post_t']:.1f} [{case['post_bound']:.1f}] |" in table

    row = validate.run_case("sepW_d4", 200, seeds=3)  # a mixture: the mode columns
    assert len(row["mode_mass"]) == 3 and 0.0 <= row["lost"] <= 1.0
    assert all(abs(sum(m) - 1.0) < 1e-6 for m in row["mode_mass"])
    assert "| sepW_d4 | 200 |" in validate.to_markdown([row])
    assert len(row["hop_acceptance"]) == 3 and "hop" not in row["failed"]
    no_hop = functools.partial(tinyns.api.Config, _hop=False)
    monkeypatch.setattr(tinyns.api, "Config", no_hop)
    row = validate.run_case("sepW_d4", 200, seeds=3)
    assert row["hop_acceptance"] == [None] * 3 and "hop" in row["failed"]
    assert "| off (3/3 low) |" in validate.to_markdown([row])
