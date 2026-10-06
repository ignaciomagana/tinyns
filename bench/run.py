"""Run one sampler on one target for a list of seeds; append JSONL records.

    python bench/run.py --sampler dynesty:rwalk100 --target sepW_d10 \
        --seeds 0-39 --nlive 500 --out results.jsonl --timeout 3600

Each seed runs in a fresh subprocess (spawned, so the JAX backend and x64 flag
are set before JAX is imported, and a hung run can be killed at ``--timeout``).
The child computes the mode masses with the target's oracle responsibility
function on the sampler's weighted samples, the same function for every
sampler, and one ``tinyns-bench-1`` record per seed is appended to ``--out``
under ``flock``. The record keys are listed in ``RECORD_KEYS``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "tinyns-bench-1"
RECORD_KEYS = (
    "schema",
    "sampler",
    "sampler_version",
    "git_sha",
    "target",
    "ndim",
    "seed",
    "truth",
    "config",
    "hw",
    "jax",
    "x64",
    "status",
    "error",
    "wall_s",
    "compile_s",
    "ncall",
    "ncall_valid",
    "logz",
    "logzerr",
    "mode_mass",
    "n_samples",
    "ess",
    "ts",
)


def parse_seeds(text: str) -> list[int]:
    """'0-19', '0,3,7' or '5'."""
    seeds: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        seeds.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return seeds


def parse_opts(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep:
            raise SystemExit(f"--opt expects key=value, got {item!r}")
        out[k] = v
    return out


def git_sha() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "bench"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _finite(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    line = json.dumps(_finite(record), default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def weighted_summary(target, samples, logwt):
    """(mode masses or None, Kish ESS) from samples and log weights."""
    import jax
    import numpy as np

    n = len(samples)
    if logwt is None:
        w = np.full(n, 1.0 / n)
    else:
        lw = np.asarray(logwt, float)
        lw = np.where(np.isfinite(lw), lw, -np.inf)
        w = np.exp(lw - lw.max())
        w /= w.sum()
    ess = float(1.0 / np.sum(w**2))
    if target.responsibility is None:
        return None, ess
    resp = jax.jit(jax.vmap(target.responsibility))
    x = np.asarray(samples, float)
    keep = w > 0
    x, w = x[keep], w[keep]
    mass = np.zeros(len(target.mode_mass))
    for s in range(0, len(x), 65536):
        r = np.asarray(resp(x[s : s + 65536]), float)
        mass += w[s : s + 65536] @ np.nan_to_num(r)
    return [float(m) for m in mass], ess


def _child(conn, spec, target_name, seed, cfg, env):
    os.environ.update(env)
    try:
        import jax

        from bench import adapters
        from bench.targets import get_target

        name, variant = adapters.parse_spec(spec)
        reason = adapters.unavailable_reason(name)
        if reason:
            raise adapters.AdapterUnavailable(reason)
        target = get_target(target_name)
        out = adapters.load(name).run(target, seed, dict(cfg, variant=variant))
        mode_mass, ess = weighted_summary(target, out["samples"], out.get("logwt"))
        dev = jax.devices()[0]
        conn.send(
            dict(
                status="ok",
                error=None,
                sampler_version=out["sampler_version"],
                config=dict(out.get("config") or {}, variant=variant),
                wall_s=out["wall_s"],
                compile_s=out.get("compile_s"),
                ncall=out["ncall"],
                ncall_valid=out.get("ncall_valid"),
                logz=out["logz"],
                logzerr=out.get("logzerr"),
                mode_mass=mode_mass,
                n_samples=int(len(out["samples"])),
                ess=ess,
                jax=jax.__version__,
                x64=bool(jax.config.jax_enable_x64),
                platform=dev.platform,
                gpu=None if dev.platform == "cpu" else dev.device_kind,
            )
        )
    except Exception as err:  # noqa: BLE001 - every failure becomes a record
        unavailable = type(err).__name__ == "AdapterUnavailable"
        conn.send(
            dict(
                status="unavailable" if unavailable else "error",
                error=f"{type(err).__name__}: {err}\n{traceback.format_exc()[-4000:]}",
            )
        )
    finally:
        conn.close()


def run_one(spec, target_name, seed, cfg, env, timeout):
    """Run one seed in a spawned child; returns the child's dict plus timing."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    t0 = time.perf_counter()
    proc = ctx.Process(target=_child, args=(child, spec, target_name, seed, cfg, env))
    proc.start()
    child.close()
    if parent.poll(timeout):
        try:
            msg = parent.recv()
        except EOFError:
            proc.join(10)
            msg = dict(status="error", error=f"child died (exit code {proc.exitcode})")
    else:
        msg = dict(status="timeout", error=f"no result after {timeout:.0f} s")
    if proc.is_alive():
        proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join()
    msg.setdefault("wall_s", time.perf_counter() - t0)
    return msg


def cpu_model() -> str | None:
    """The CPU model name from /proc/cpuinfo (Linux), else platform.processor()."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or None


def make_record(args, spec, target, seed, cfg, msg, sha, device):
    hw = dict(
        host=socket.gethostname(),
        gpu=msg.get("gpu"),
        platform=msg.get("platform", device),
        cpu_model=cpu_model(),
        cpu_threads=len(os.sched_getaffinity(0)),
        exclusive=bool(args.exclusive),
    )
    if os.environ.get("SLURM_JOB_ID"):
        hw["slurm_job"] = os.environ.get("SLURM_JOB_ID")
        hw["slurm_partition"] = os.environ.get("SLURM_JOB_PARTITION")
    rec = dict(
        schema=SCHEMA,
        sampler=spec,
        sampler_version=msg.get("sampler_version"),
        git_sha=sha,
        target=target.name,
        ndim=target.ndim,
        seed=seed,
        truth=dict(
            logz=target.logz,
            logz_source=target.logz_source,
            mode_mass=list(target.mode_mass) if target.mode_mass else None,
        ),
        config=dict(msg.get("config") or {}, cli=cfg),
        hw=hw,
        jax=msg.get("jax"),
        x64=msg.get("x64", args.x64),
        status=msg["status"],
        error=msg.get("error"),
        wall_s=msg.get("wall_s"),
        compile_s=msg.get("compile_s"),
        ncall=msg.get("ncall"),
        ncall_valid=msg.get("ncall_valid"),
        logz=msg.get("logz"),
        logzerr=msg.get("logzerr"),
        mode_mass=msg.get("mode_mass"),
        n_samples=msg.get("n_samples"),
        ess=msg.get("ess"),
        ts=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    if args.tag:
        rec["tag"] = args.tag
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sampler", required=True, help="name or name:variant")
    p.add_argument("--target", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per seed")
    p.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="auto: the sampler's default (adapters.REGISTRY)",
    )
    p.add_argument(
        "--exclusive",
        action="store_true",
        help="record that the run had the machine (or its cores) to itself",
    )
    p.add_argument(
        "--opt", action="append", default=[], help="sampler option key=value"
    )
    p.add_argument(
        "--jax-cache",
        default=None,
        help="persistent JAX compilation cache dir (compile_s then "
        "measures cache loads after the first seed)",
    )
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)

    from bench import adapters
    from bench.targets import get_target

    name, _ = adapters.parse_spec(args.sampler)
    target = get_target(args.target)
    device = adapters.device(name) if args.device == "auto" else args.device
    env = {"JAX_ENABLE_X64": "1" if args.x64 else "0"}
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    if args.jax_cache:
        env["JAX_COMPILATION_CACHE_DIR"] = args.jax_cache
    cfg = dict(nlive=args.nlive, dlogz=args.dlogz, opts=parse_opts(args.opt))
    sha = git_sha()
    for seed in parse_seeds(args.seeds):
        msg = run_one(args.sampler, args.target, seed, cfg, env, args.timeout)
        if msg["status"] == "unavailable":  # not a run: nothing is recorded
            print(msg["error"].splitlines()[0], file=sys.stderr)
            return 2
        rec = make_record(args, args.sampler, target, seed, cfg, msg, sha, device)
        append_jsonl(args.out, rec)
        dz = None if rec["logz"] is None else round(rec["logz"] - target.logz, 3)
        err = None if rec["logzerr"] is None else round(rec["logzerr"], 3)
        modes = rec["mode_mass"] and [round(m, 3) for m in rec["mode_mass"]]
        print(
            f"{args.sampler} {args.target} seed={seed} {rec['status']} dlogz={dz} "
            f"err={err} ncall={rec['ncall']} wall={rec['wall_s']:.1f}s modes={modes}",
            flush=True,
        )
        if rec["status"] == "error":
            print(rec["error"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
