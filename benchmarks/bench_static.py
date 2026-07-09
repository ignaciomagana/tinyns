"""Lightweight static nested-sampling benchmarks for tinyns."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from copy import copy
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
    """Return grouped benchmark summaries, excluding warmup rows."""

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row.get("warmup", False):
            continue
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
                "mean_repl_batches": _mean(
                    [
                        row.get("mean_replacement_batches", row.get("repl_batches"))
                        for row in rows
                    ]
                ),
                "max_repl_batches": max(
                    (
                        row.get("max_replacement_batches", row.get("max_repl_batches"))
                        for row in rows
                        if row.get(
                            "max_replacement_batches", row.get("max_repl_batches")
                        )
                        is not None
                    ),
                    default=None,
                ),
                "mean_max_repl_batches": _mean(
                    [row.get("max_repl_batches") for row in rows]
                ),
                "success_fraction": sum(bool(row.get("success")) for row in rows)
                / len(rows),
            }
        )
    baselines = {
        (row["target"], row["sampler"], row["kernel"]): row
        for row in summaries
        if row["replacement_chains"] == 1
    }
    for row in summaries:
        baseline = baselines.get((row["target"], row["sampler"], row["kernel"]))
        row["relative_speedup_vs_chains1"] = None
        row["relative_iter_s_vs_chains1"] = None
        if baseline is not None:
            seconds = row.get("mean_seconds")
            base_seconds = baseline.get("mean_seconds")
            if seconds not in (None, 0) and base_seconds is not None:
                row["relative_speedup_vs_chains1"] = base_seconds / seconds
            base_iter_s = baseline.get("mean_iter_per_s")
            iter_s = row.get("mean_iter_per_s")
            if base_iter_s not in (None, 0) and iter_s is not None:
                row["relative_iter_s_vs_chains1"] = iter_s / base_iter_s
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
        "replacement_chain_schedule": args.replacement_chain_schedule,
        "rwalk_proposal": args.rwalk_proposal,
        "bound": args.bound,
        "bound_enlargement": args.bound_enlargement,
        "bound_update_interval": args.bound_update_interval,
        "bound_jitter": args.bound_jitter,
        "multi_bound_max_ellipsoids": args.multi_bound_max_ellipsoids,
        "multi_bound_min_points": args.multi_bound_min_points,
        "multi_bound_split_threshold": args.multi_bound_split_threshold,
        "multi_bound_overlap_correction": args.multi_bound_overlap_correction,
        "rwalk_seed": args.rwalk_seed,
        "bound_seed_kernel": args.bound_seed_kernel,
        "allow_unused_bound": args.allow_unused_bound,
    }
    if sampler_name == "rwalk":
        kwargs["walks"] = args.walks
    return kwargs


def run_one(
    target_name: str,
    sampler_name: str,
    seed: int,
    args: argparse.Namespace,
    *,
    warmup: bool = False,
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
    replacement_batch_ncall = int(
        diagnostics.get("replacement_batch_ncall")
        or metadata.get("replacement_batch_ncall")
        or (args.walks * args.replacement_chains)
    )
    mean_replacement_ncall = float(metadata.get("mean_replacement_ncall", 0.0))
    max_replacement_ncall = int(metadata.get("max_replacement_ncall", 0))
    repl_batches = metadata.get("mean_replacement_batches")
    if repl_batches is None:
        repl_batches = (
            mean_replacement_ncall / replacement_batch_ncall
            if replacement_batch_ncall > 0
            else None
        )
    max_repl_batches = metadata.get("max_replacement_batches")
    if max_repl_batches is None:
        max_repl_batches = (
            max_replacement_ncall / replacement_batch_ncall
            if replacement_batch_ncall > 0
            else None
        )

    return {
        "target": target_name,
        "sampler": sampler_name,
        "kernel": args.kernel,
        "bound": args.bound,
        "rwalk_seed": args.rwalk_seed,
        "bound_seed_kernel": args.bound_seed_kernel,
        "allow_unused_bound": args.allow_unused_bound,
        "bound_update_interval": metadata.get(
            "bound_update_interval", args.bound_update_interval
        ),
        "bound_build_time_total": metadata.get("bound_build_time_total"),
        "bound_build_time_mean": metadata.get("bound_build_time_mean"),
        "bound_build_time_max": metadata.get("bound_build_time_max"),
        "bound_build_count": metadata.get("bound_build_count"),
        "bound_log_volume_final": metadata.get("bound_log_volume_final"),
        "bound_log_volume_mean": metadata.get("bound_log_volume_mean"),
        "bound_log_volume_min": metadata.get("bound_log_volume_min"),
        "bound_log_volume_max": metadata.get("bound_log_volume_max"),
        "bound_nellipsoids_mean": metadata.get("bound_nellipsoids_mean"),
        "bound_nellipsoids_max": metadata.get("bound_nellipsoids_max"),
        "bound_nellipsoids_final": metadata.get("bound_nellipsoids_final"),
        "seed": seed,
        "nlive": args.nlive,
        "ndim": target.ndim,
        "dlogz": args.dlogz,
        "maxiter": args.maxiter,
        "walks": args.walks,
        "step_scale": args.step_scale,
        "min_accepts": args.min_accepts,
        "replacement_chains": args.replacement_chains,
        "adaptive_replacement_chains": bool(
            args.replacement_chain_schedule is not None
        ),
        "replacement_chain_schedule": args.replacement_chain_schedule,
        "replacement_batch_ncall": replacement_batch_ncall,
        "replacement_initial_batch_ncall": metadata.get(
            "replacement_initial_batch_ncall"
        ),
        "replacement_max_batch_ncall": metadata.get("replacement_max_batch_ncall"),
        "repl_batches": repl_batches,
        "max_repl_batches": max_repl_batches,
        "mean_replacement_batches": repl_batches,
        "max_replacement_batches": max_repl_batches,
        "seconds": seconds,
        "ncall": int(result.ncall),
        "niter": niter,
        "ndead": diagnostics.get("ndead"),
        "iterations_per_second": iter_per_s,
        "likelihood_calls_per_second": ncall_per_s,
        "mean_replacement_ncall": mean_replacement_ncall,
        "max_replacement_ncall": max_replacement_ncall,
        "mean_replacement_chains_used": metadata.get("mean_replacement_chains_used"),
        "max_replacement_chains_used": metadata.get("max_replacement_chains_used"),
        "replacement_chain_usage_counts": metadata.get(
            "replacement_chain_usage_counts"
        ),
        "replacement_failures": int(metadata.get("replacement_failures", 0)),
        "logz": float(result.logz),
        "logzerr": float(result.logzerr),
        "success": bool(result.success),
        "message": str(result.message),
        "warning_count": len(warnings),
        "warnings": warnings,
        "warmup": warmup,
    }


def _fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def print_results(results: list[dict[str, Any]]) -> None:
    print(
        "target sampler kernel replacement_chains seed seconds niter ncall iter/s "
        "ncall/s repl_ncall repl_batches max_repl_batches mean_repl_chains "
        "max_repl_chains usage adaptive initial_batch max_batch logz logzerr "
        "success warnings"
    )
    for row in results:
        print(
            f"{row['target']} {row['sampler']} {row['kernel']} "
            f"{row['replacement_chains']} {row['seed']} "
            f"{_fmt(row['seconds'])} {_fmt(row['niter'])} {row['ncall']} "
            f"{_fmt(row['iterations_per_second'])} "
            f"{_fmt(row['likelihood_calls_per_second'])} "
            f"{_fmt(row['mean_replacement_ncall'])} {_fmt(row['repl_batches'])} "
            f"{_fmt(row['max_repl_batches'])} "
            f"{_fmt(row.get('mean_replacement_chains_used'))} "
            f"{_fmt(row.get('max_replacement_chains_used'))} "
            f"{row.get('replacement_chain_usage_counts')} "
            f"{row.get('adaptive_replacement_chains')} "
            f"{_fmt(row.get('replacement_initial_batch_ncall'))} "
            f"{_fmt(row.get('replacement_max_batch_ncall'))} "
            f"{_fmt(row['logz'])} {_fmt(row['logzerr'])} "
            f"{row['success']} {row['warning_count']}"
        )


def print_summaries(summaries: list[dict[str, Any]]) -> None:
    print()
    print(
        "target sampler kernel replacement_chains nruns mean_seconds mean_iter_per_s "
        "mean_ncall_per_s mean_ncall mean_repl_ncall mean_repl_batches "
        "max_repl_batches success_fraction relative_speedup_vs_chains1 "
        "relative_iter_s_vs_chains1"
    )
    for row in summaries:
        print(
            f"{row['target']} {row['sampler']} {row['kernel']} "
            f"{row['replacement_chains']} {row['nruns']} "
            f"{_fmt(row['mean_seconds'])} {_fmt(row['mean_iter_per_s'])} "
            f"{_fmt(row['mean_ncall_per_s'])} {_fmt(row['mean_ncall'])} "
            f"{_fmt(row['mean_repl_ncall'])} {_fmt(row['mean_repl_batches'])} "
            f"{_fmt(row['max_repl_batches'])} {_fmt(row['success_fraction'])} "
            f"{_fmt(row['relative_speedup_vs_chains1'])} "
            f"{_fmt(row['relative_iter_s_vs_chains1'])}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="+", default=["gaussian2d", "correlated_gaussian2d"]
    )
    parser.add_argument(
        "--samplers", nargs="+", choices=["prior", "rwalk"], default=["prior", "rwalk"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--nlive", type=int, default=200)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--walks", type=int, default=25)
    parser.add_argument("--step-scale", type=float, default=0.1)
    parser.add_argument("--min-accepts", type=int, default=1)
    parser.add_argument("--replacement-chains", type=int, default=1)
    parser.add_argument(
        "--rwalk-proposal", choices=["isotropic"], default="isotropic"
    )
    parser.add_argument("--bound", choices=["none", "single", "multi"], default="none")
    parser.add_argument("--bound-enlargement", type=float, default=1.25)
    parser.add_argument("--bound-update-interval", type=int, default=1)
    parser.add_argument("--bound-jitter", type=float, default=1e-6)
    parser.add_argument("--multi-bound-max-ellipsoids", type=int, default=32)
    parser.add_argument("--multi-bound-min-points", type=int, default=None)
    parser.add_argument("--multi-bound-split-threshold", type=float, default=0.9)
    parser.add_argument(
        "--multi-bound-overlap-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rwalk-seed", choices=["live", "bound"], default="live")
    parser.add_argument(
        "--bound-seed-kernel", choices=["python", "jax"], default="python"
    )
    parser.add_argument("--allow-unused-bound", action="store_true")
    parser.add_argument("--replacement-chains-grid", nargs="+", type=int, default=None)
    parser.add_argument(
        "--replacement-chain-schedule", nargs="+", type=int, default=None
    )
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--discard-warmup", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument(
        "--auto-max-attempts",
        action="store_true",
        help=(
            "Benchmark convenience for JAX rwalk chain sweeps: raise "
            "--max-attempts to at least four replacement batches when needed."
        ),
    )
    parser.add_argument("--kernel", choices=["python", "jax"], default="python")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def validate_benchmark_args(args: argparse.Namespace) -> None:
    """Validate benchmark-only argument combinations before running cases."""

    replacement_chains_values = (
        [args.replacement_chains]
        if args.replacement_chain_schedule is not None
        else args.replacement_chains_grid
        if args.replacement_chains_grid is not None
        else [args.replacement_chains]
    )
    max_replacement_chains = max(
        args.replacement_chain_schedule
        if args.replacement_chain_schedule is not None
        else replacement_chains_values
    )
    required_max_attempts = args.walks * max_replacement_chains

    if (
        args.bound in {"single", "multi"}
        and "rwalk" in args.samplers
        and args.rwalk_seed == "live"
        and not args.allow_unused_bound
    ):
        raise ValueError(
            "--bound single or --bound multi with --samplers rwalk requires "
            "--rwalk-seed bound. Otherwise the bound is built but not used. "
            "Use --allow-unused-bound to benchmark live-seeded rwalk with "
            "bound overhead."
        )

    if args.kernel == "jax" and "rwalk" in args.samplers:
        if args.auto_max_attempts:
            args.max_attempts = max(args.max_attempts, 4 * required_max_attempts)
        elif args.max_attempts < required_max_attempts:
            raise ValueError(
                "--max-attempts must be at least walks * max(replacement_chains). "
                f"Got max_attempts={args.max_attempts}, walks={args.walks}, "
                f"max replacement_chains={max_replacement_chains}, "
                f"required={required_max_attempts}. "
                f"Try --max-attempts {required_max_attempts} or larger."
            )


def _args_for_replacement_chains(
    args: argparse.Namespace, replacement_chains: int
) -> argparse.Namespace:
    case_args = copy(args)
    case_args.replacement_chains = replacement_chains
    return case_args


def _run_benchmark_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    replacement_chains_values = (
        [args.replacement_chains]
        if args.replacement_chain_schedule is not None
        else args.replacement_chains_grid
        if args.replacement_chains_grid is not None
        else [args.replacement_chains]
    )
    results: list[dict[str, Any]] = []
    for target_name in args.targets:
        for sampler_name in args.samplers:
            for replacement_chains in replacement_chains_values:
                case_args = _args_for_replacement_chains(args, replacement_chains)
                warmup_rows = [
                    run_one(
                        target_name,
                        sampler_name,
                        -(warmup_index + 1),
                        case_args,
                        warmup=True,
                    )
                    for warmup_index in range(args.warmup_runs)
                ]
                if not args.discard_warmup:
                    results.extend(warmup_rows)
                results.extend(
                    run_one(target_name, sampler_name, seed, case_args, warmup=False)
                    for seed in args.seeds
                )
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_benchmark_args(args)
    results = _run_benchmark_grid(args)
    summaries = summarize_results(results)
    print_results(results)
    print_summaries(summaries)

    if args.output is not None:
        payload = build_payload(results, summaries)
        Path(args.output).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


if __name__ == "__main__":
    main()
