"""Lightweight static nested-sampling benchmarks for tinyns."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from validation.targets import get_target  # noqa: E402

from tinyns import NestedSampler  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def compute_rates(
    niter: int | None, ncall: int, seconds: float
) -> tuple[float | None, float | None]:
    """Return iterations/sec and likelihood calls/sec, or ``None`` for zero time."""

    if seconds == 0.0:
        return None, None
    return (None if niter is None else niter / seconds), ncall / seconds


def _mean(values: list[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return grouped benchmark summaries by ``(target, sampler)``."""

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        key = (
            row["target"],
            row["sampler"],
            row.get("kernel", "python"),
            int(row.get("replacement_chains", 1)),
        )
        grouped[key].append(row)

    summaries = []
    for (target, sampler, kernel, replacement_chains), rows in sorted(grouped.items()):
        summaries.append(
            {
                "target": target,
                "sampler": sampler,
                "kernel": kernel,
                "replacement_chains": replacement_chains,
                "nruns": len(rows),
                "mean_seconds": _mean([row.get("seconds") for row in rows]),
                "mean_iter_per_s": _mean(
                    [row.get("iterations_per_second") for row in rows]
                ),
                "mean_ncall_per_s": _mean(
                    [row.get("likelihood_calls_per_second") for row in rows]
                ),
                "mean_ncall": _mean([row.get("ncall") for row in rows]),
                "mean_repl_ncall": _mean(
                    [row.get("mean_replacement_ncall") for row in rows]
                ),
                "success_fraction": sum(bool(row.get("success")) for row in rows)
                / len(rows),
            }
        )
    return summaries


def build_payload(
    results: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return JSON-output payload for benchmark results."""

    return {"results": results, "summaries": summaries}


def _sampler_kwargs(sampler_name: str, args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "sample": sampler_name,
        "nlive": args.nlive,
        "max_attempts": args.max_attempts,
        "step_scale": args.step_scale,
        "min_accepts": args.min_accepts,
        "kernel": args.kernel,
        "replacement_chains": args.replacement_chains,
    }
    if sampler_name == "rwalk":
        kwargs["walks"] = args.walks
    elif sampler_name in {"slice", "rslice"}:
        kwargs["slices"] = args.slices
        kwargs["slice_steps"] = args.slice_steps
    return kwargs


def run_one(
    target_name: str, sampler_name: str, seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    """Run one benchmark case and return a metrics row."""

    target = get_target(target_name)
    sampler = NestedSampler(
        target.loglike,
        target.prior_transform,
        ndim=target.ndim,
        **_sampler_kwargs(sampler_name, args),
    )

    start = time.perf_counter()
    result = sampler.run(
        jax.random.PRNGKey(seed),
        dlogz=args.dlogz,
        maxiter=args.maxiter,
        progress=args.progress,
    )
    seconds = time.perf_counter() - start

    metadata = {} if result.metadata is None else result.metadata
    diagnostics = result.diagnostics()
    niter = diagnostics.get("niter", diagnostics.get("ndead"))
    if niter is not None:
        niter = int(niter)
    iter_per_s, ncall_per_s = compute_rates(niter, int(result.ncall), seconds)
    warnings = diagnostics.get("warnings", [])

    return {
        "target": target_name,
        "sampler": sampler_name,
        "kernel": args.kernel,
        "seed": seed,
        "nlive": args.nlive,
        "ndim": target.ndim,
        "dlogz": args.dlogz,
        "maxiter": args.maxiter,
        "walks": args.walks,
        "slices": args.slices,
        "slice_steps": args.slice_steps,
        "step_scale": args.step_scale,
        "min_accepts": args.min_accepts,
        "replacement_chains": args.replacement_chains,
        "seconds": seconds,
        "ncall": int(result.ncall),
        "niter": niter,
        "ndead": diagnostics.get("ndead"),
        "iterations_per_second": iter_per_s,
        "likelihood_calls_per_second": ncall_per_s,
        "mean_replacement_ncall": float(metadata.get("mean_replacement_ncall", 0.0)),
        "max_replacement_ncall": int(metadata.get("max_replacement_ncall", 0)),
        "replacement_failures": int(metadata.get("replacement_failures", 0)),
        "logz": float(result.logz),
        "logzerr": float(result.logzerr),
        "success": bool(result.success),
        "message": str(result.message),
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def _fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def print_results(results: list[dict[str, Any]]) -> None:
    print(
        "target sampler kernel chains seed seconds niter ncall iter/s "
        "ncall/s repl_ncall logz logzerr success warnings"
    )
    for row in results:
        print(
            f"{row['target']} {row['sampler']} {row['kernel']} "
            f"{row['replacement_chains']} {row['seed']} "
            f"{_fmt(row['seconds'])} {_fmt(row['niter'])} {row['ncall']} "
            f"{_fmt(row['iterations_per_second'])} "
            f"{_fmt(row['likelihood_calls_per_second'])} "
            f"{_fmt(row['mean_replacement_ncall'])} {_fmt(row['logz'])} "
            f"{_fmt(row['logzerr'])} {row['success']} {row['warning_count']}"
        )


def print_summaries(summaries: list[dict[str, Any]]) -> None:
    print()
    print(
        "target sampler kernel chains nruns mean_seconds mean_iter_per_s "
        "mean_ncall_per_s mean_ncall mean_repl_ncall success_fraction"
    )
    for row in summaries:
        print(
            f"{row['target']} {row['sampler']} {row['kernel']} "
            f"{row['replacement_chains']} {row['nruns']} "
            f"{_fmt(row['mean_seconds'])} {_fmt(row['mean_iter_per_s'])} "
            f"{_fmt(row['mean_ncall_per_s'])} {_fmt(row['mean_ncall'])} "
            f"{_fmt(row['mean_repl_ncall'])} {_fmt(row['success_fraction'])}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="+", default=["gaussian2d", "correlated_gaussian2d"]
    )
    parser.add_argument("--samplers", nargs="+", default=["rwalk", "slice", "rslice"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--nlive", type=int, default=200)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--walks", type=int, default=25)
    parser.add_argument("--slices", type=int, default=5)
    parser.add_argument("--slice-steps", type=int, default=10)
    parser.add_argument("--step-scale", type=float, default=0.1)
    parser.add_argument("--min-accepts", type=int, default=1)
    parser.add_argument("--replacement-chains", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--kernel", choices=["python", "jax"], default="python")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = [
        run_one(target_name, sampler_name, seed, args)
        for target_name in args.targets
        for sampler_name in args.samplers
        for seed in args.seeds
    ]
    summaries = summarize_results(results)
    print_results(results)
    print_summaries(summaries)

    if args.output is not None:
        payload = build_payload(results, summaries)
        Path(args.output).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


if __name__ == "__main__":
    main()
