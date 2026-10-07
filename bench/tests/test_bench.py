"""Harness tests: target truths, every installed adapter on a 2-D Gaussian, run.py."""

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
    md = summarize.to_markdown(summarize.summarize(summarize.load([out])))
    assert "| sepW_d4 | " + name + " | 100 | 1/1 |" in md


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
