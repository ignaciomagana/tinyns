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
    z = np.trapezoid(np.trapezoid(np.exp(ll - m), xs[1], axis=1), xs[0])
    return m + math.log(z) - target.log_prior_volume


def test_every_bench_target_builds():
    import jax

    for name in BENCH_TARGETS:
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


@pytest.mark.parametrize("name", ["sepW_d4", "connW_d10", "mix3_d10"])
def test_mixture_mode_masses_are_exact(name):
    """E_post[responsibility] = mixture weights, by exact draws from the mixture."""
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
    modes = dict(good, lost=0.0, logit_z=1.0, logit_sd=0.1)
    assert validate.judge(modes) == []
    for bad in (dict(lost=0.1), dict(logit_z=3.5), dict(logit_sd=0.5)):
        assert validate.judge(dict(modes, **bad)) == ["modes"]
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
    for name, (quick, full, _, modes) in validate.CASES.items():
        assert get_target(name).logz is not None and full is not None
        assert quick is None or quick <= full
        assert 1 <= modes[0] <= modes[1]


def test_validation_gate_runs_on_tiny_settings(tmp_path, capsys):
    """The quick tier's code path: batched seeds, the table, the JSON, the exit code."""
    import jax
    from bench import validate

    if adapters.unavailable_reason("tinyns_v1"):
        pytest.skip("tinyns v1 is not installed")
    x64 = "--x64" if jax.config.jax_enable_x64 else "--no-x64"
    out = tmp_path / "gate.json"
    args = ["--cases", "gauss_d2", "--nlive", "100", "--seeds", "3"]
    code = validate.main(args + [x64, "--json", str(out)])
    table = capsys.readouterr().out
    case = json.loads(out.read_text())["cases"][0]
    assert code == (1 if case["failed"] else 0)
    assert len(case["logz"]) == 3 and case["nlive"] == 100 and case["walks"] == 25
    assert "| gauss_d2 | 100 | 25 | 3 |" in table and "cases pass" in table
    assert abs(case["dz_mean"]) < 1.0 and case["unresolved"] == 0
    assert case["nmodes"] == [1, 1, 1] and "| 1 (3/3 ok) |" in table

    row = validate.run_case("sepW_d4", 200, seeds=3)  # a mixture: the mode columns
    assert len(row["mode_mass"]) == 3 and 0.0 <= row["lost"] <= 1.0
    assert all(abs(sum(m) - 1.0) < 1e-6 for m in row["mode_mass"])
    assert "| sepW_d4 | 200 |" in validate.to_markdown([row])
