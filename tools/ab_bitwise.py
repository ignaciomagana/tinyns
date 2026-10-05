"""Same-machine A/B check: does this checkout reproduce a base commit bit for bit?

Fixed-seed fingerprints differ between machines (CPU, BLAS and XLA builds), so a
restructuring change is checked against its base commit on the same machine
instead of against stored golden files. The base is checked out into a temporary
git worktree, every case runs once per side in a fresh subprocess, and every
output array and count must be identical.

    python tools/ab_bitwise.py --base origin/main
    python tools/ab_bitwise.py --base origin/main --cases g1 mm4 --x64 on

The runner adapts to the API of the side it runs under (``jax_block_size`` was
renamed ``block_size``), so the same case list works across the v0.3 series.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# name: (target, nlive, options); options may hold walks, chains, block, maxiter,
# resume_at (iteration of a mid-run checkpoint) and pytree.
CASES = {
    "g1": ("gauss1", 100, {}),
    "c5": ("corr5", 200, {}),
    "a13": ("aniso13", 200, {"maxiter": 2500}),
    "mm4": ("twomode4", 300, {}),
    "chains4": ("corr5", 200, {"chains": 4}),
    "blk7": ("corr5", 200, {"block": 7}),
    "maxiter": ("gauss1", 100, {"maxiter": 150}),
    "resume": ("corr5", 200, {"resume_at": 640}),
    "pytree": ("corr5_pytree", 200, {}),
}


def _targets():
    import jax
    import jax.numpy as jnp

    def gauss(ndim, sigma, rho=0.0, perm=None):
        cov = np.full((ndim, ndim), rho) + (1.0 - rho) * np.eye(ndim)
        s = np.asarray(sigma, dtype=float) * np.ones(ndim)
        cov = cov * np.outer(s, s)
        if perm is not None:
            cov = cov[np.ix_(perm, perm)]
        prec = jnp.asarray(np.linalg.inv(cov))
        logdet = float(np.linalg.slogdet(cov)[1])

        def loglike(x):
            d = x - 0.5
            return -0.5 * d @ prec @ d - 0.5 * logdet - 0.5 * ndim * jnp.log(2 * jnp.pi)

        return loglike

    def identity(u):
        return u

    def twomode(x):
        a = -0.5 * jnp.sum(((x - 0.35) / 0.03) ** 2)
        b = (
            -0.5 * jnp.sum(((x - 0.70) / 0.025) ** 2)
            + jnp.log(0.3)
            - 4 * jnp.log(0.025 / 0.03)
        )
        return jnp.logaddexp(a, b) - 4 * jnp.log(0.03 * jnp.sqrt(2 * jnp.pi))

    def corr5_data(x, prec):
        d = x - 0.5
        return -0.5 * d @ prec @ d

    prec5 = jnp.asarray(np.linalg.inv(0.01 * (np.full((5, 5), 0.6) + 0.4 * np.eye(5))))
    sig13 = np.logspace(np.log10(0.004), np.log10(0.08), 13)
    return {
        "gauss1": (gauss(1, 0.05), identity, 1),
        "corr5": (gauss(5, 0.1, rho=0.6), identity, 5),
        "aniso13": (
            gauss(13, sig13, rho=0.5, perm=np.random.default_rng(3).permutation(13)),
            identity,
            13,
        ),
        "twomode4": (twomode, identity, 4),
        "corr5_pytree": (jax.tree_util.Partial(corr5_data, prec=prec5), identity, 5),
    }


def _fingerprint(res) -> dict:
    out = {}
    for name in ("samples_u", "samples", "logl", "logwt"):
        out[name] = np.asarray(getattr(res, name))
    for name in ("logz", "logzerr", "ncall", "nlive"):
        out[name] = np.asarray(getattr(res, name))
    meta = res.metadata or {}
    for name in (
        "replacement_rescue_attempts",
        "cluster_swap_proposals",
        "cluster_swap_accepts",
    ):
        if name in meta:
            out["meta_" + name] = np.asarray(meta[name])
    return out


def run_cases(names, out_path):
    """Run cases under whichever tinyns is importable; write fingerprints."""

    import jax

    import tinyns

    expect = os.environ.get("AB_EXPECT_SRC")
    if expect and not Path(tinyns.__file__).resolve().is_relative_to(Path(expect)):
        raise SystemExit(f"imported {tinyns.__file__}, expected a copy under {expect}")
    sig = inspect.signature(tinyns.NestedSampler.__init__).parameters
    block_kw = "block_size" if "block_size" in sig else "jax_block_size"
    targets = _targets()
    results, info = (
        {},
        {"tinyns": tinyns.__version__, "x64": bool(jax.config.jax_enable_x64)},
    )
    for name in names:
        target, nlive, opt = CASES[name]
        loglike, prior_transform, ndim = targets[target]
        kw = {block_kw: opt.get("block", 32)}
        if "chains" in opt:
            kw["replacement_chains"] = opt["chains"]
        if "walks" in opt:
            kw["walks"] = opt["walks"]
        ns = tinyns.NestedSampler(loglike, prior_transform, ndim, nlive=nlive, **kw)
        key = jax.random.PRNGKey(17)
        if "resume_at" in opt:
            with tempfile.TemporaryDirectory() as tmp:
                ck = os.path.join(tmp, "ck.npz")
                ns.run(
                    key,
                    maxiter=opt["resume_at"],
                    checkpoint_path=ck,
                    checkpoint_interval=10**9,
                )
                res = ns.resume(ck)
        else:
            res = ns.run(key, maxiter=opt.get("maxiter"))
        fp = _fingerprint(res)
        rescue = int(fp.get("meta_replacement_rescue_attempts", 0))
        if rescue:
            raise SystemExit(
                f"case {name}: replacement_rescue_attempts = {rescue}, expected 0"
            )
        for k, v in fp.items():
            results[f"{name}/{k}"] = v
        info[name] = {
            "logz": float(fp["logz"]),
            "ncall": int(fp["ncall"]),
            "n": int(fp["logl"].size),
        }
    np.savez(out_path, **results)
    Path(out_path).with_suffix(".json").write_text(json.dumps(info))


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    keys = sorted(set(a.files) | set(b.files))
    bad, notes = [], []
    for k in keys:
        if k not in a.files or k not in b.files:
            # A metadata key may legitimately appear or go away between versions.
            (notes if "/meta_" in k else bad).append((k, "missing on one side"))
        elif a[k].shape != b[k].shape:
            bad.append((k, f"shape {a[k].shape} vs {b[k].shape}"))
        elif not np.array_equal(a[k], b[k], equal_nan=True):
            diff = (
                np.nanmax(np.abs(a[k].astype(float) - b[k].astype(float)))
                if a[k].size
                else 0.0
            )
            bad.append((k, f"max abs diff {diff:.3g}"))
    return keys, bad, notes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default="origin/main", help="git ref to compare against")
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--x64", choices=("on", "off", "both"), default="both")
    ap.add_argument("--run-cases", metavar="OUT", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.run_cases:
        run_cases(a.cases, a.run_cases)
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="tinyns-ab-"))
    base_dir = tmp / "base"
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "worktree",
            "add",
            "--detach",
            "-q",
            str(base_dir),
            a.base,
        ],
        check=True,
    )
    failed = False
    try:
        for x64 in ("on", "off") if a.x64 == "both" else (a.x64,):
            outs = {}
            for side, root in (("base", base_dir), ("head", REPO)):
                env = dict(
                    os.environ,
                    PYTHONPATH=str(root / "src"),
                    AB_EXPECT_SRC=str((root / "src").resolve()),
                    JAX_PLATFORMS="cpu",
                    JAX_ENABLE_X64="1" if x64 == "on" else "0",
                )
                outs[side] = tmp / f"{side}_x64{x64}.npz"
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--run-cases",
                        str(outs[side]),
                        "--cases",
                        *a.cases,
                    ],
                    env=env,
                    check=True,
                )
            keys, bad, notes = compare(outs["base"], outs["head"])
            info = json.loads(outs["head"].with_suffix(".json").read_text())
            print(f"x64={x64}: {len(keys) - len(bad)}/{len(keys)} arrays identical")
            for name in a.cases:
                row = info[name]
                print(
                    f"  {name:8s} logz {row['logz']:+.6f}"
                    f"  ncall {row['ncall']:8d}  n {row['n']}"
                )
            for k, why in bad:
                print(f"  MISMATCH {k}: {why}")
            for k, why in notes:
                print(f"  note {k}: {why}")
            failed |= bool(bad)
    finally:
        subprocess.run(
            ["git", "-C", str(REPO), "worktree", "remove", "--force", str(base_dir)],
            check=False,
        )
        shutil.rmtree(tmp, ignore_errors=True)
    print(
        ("FAIL: outputs differ from " if failed else "OK: bit-identical to ") + a.base
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
