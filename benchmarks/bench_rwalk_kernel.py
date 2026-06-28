"""Microbenchmark the JAX random-walk constrained replacement kernel."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from validation.targets import get_target, heavy_gaussian_2d  # noqa: E402

from tinyns.samplers import (  # noqa: E402
    draw_constrained_rwalk_jax,
    draw_constrained_rwalk_jax_adaptive,
)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="+", default=["gaussian2d", "correlated_gaussian2d"]
    )
    parser.add_argument(
        "--replacement-chains-grid",
        nargs="+",
        type=int,
        default=[1, 4, 16, 64, 256, 1024],
    )
    parser.add_argument(
        "--replacement-chain-schedule", nargs="+", type=int, default=None
    )
    parser.add_argument("--walks", type=int, default=25)
    parser.add_argument("--nlive", type=int, default=200)
    parser.add_argument("--n-replacements", type=int, default=1000)
    parser.add_argument("--warmup-replacements", type=int, default=100)
    parser.add_argument("--step-scale", type=float, default=0.1)
    parser.add_argument("--min-accepts", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=102400)
    parser.add_argument("--work-size", type=int, default=100_000)
    parser.add_argument("--constraint-quantile", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="bench_rwalk_kernel.json")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    max_replacement_chains = max(
        args.replacement_chain_schedule
        if args.replacement_chain_schedule is not None
        else args.replacement_chains_grid
    )
    required_max_attempts = args.walks * max_replacement_chains
    if args.max_attempts < required_max_attempts:
        raise ValueError(
            "--max-attempts must be at least walks * max(replacement_chains_grid). "
            f"Got max_attempts={args.max_attempts}, walks={args.walks}, "
            f"max replacement_chains={max_replacement_chains}, "
            f"required={required_max_attempts}. "
            f"Try --max-attempts {required_max_attempts} or larger."
        )
    if not 0.0 <= args.constraint_quantile <= 1.0:
        raise ValueError("--constraint-quantile must be between 0 and 1")


def make_target(target_name: str, work_size: int):
    if target_name == "heavy_gaussian2d":
        return heavy_gaussian_2d(work_size=work_size)
    return get_target(target_name)


def build_live_state(
    target_name: str,
    key: jax.Array,
    nlive: int,
    constraint_quantile: float,
    work_size: int,
):
    target = make_target(target_name, work_size)
    live_u = jax.random.uniform(key, (nlive, target.ndim))
    live_theta = jax.vmap(target.prior_transform)(live_u)
    live_logl = jax.vmap(target.loglike)(live_theta)
    logl_min = jnp.quantile(live_logl, constraint_quantile)
    jax.block_until_ready((live_u, live_theta, live_logl, logl_min))
    return target, live_u, live_logl, float(logl_min)


def run_one(
    target_name: str, replacement_chains: int, seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    live_key, run_key = jax.random.split(jax.random.PRNGKey(seed))
    target, live_u, live_logl, logl_min = build_live_state(
        target_name, live_key, args.nlive, args.constraint_quantile, args.work_size
    )

    for _ in range(args.warmup_replacements):
        run_key, *_ = (
            draw_constrained_rwalk_jax_adaptive
            if args.replacement_chain_schedule is not None
            else draw_constrained_rwalk_jax
        )(
            run_key,
            target.loglike,
            target.prior_transform,
            logl_min,
            live_u,
            live_logl,
            target.ndim,
            walks=args.walks,
            step_scale=args.step_scale,
            max_attempts=args.max_attempts,
            min_accepts=args.min_accepts,
            **(
                {"replacement_chain_schedule": args.replacement_chain_schedule}
                if args.replacement_chain_schedule is not None
                else {"replacement_chains": replacement_chains}
            ),
        )

    ncall_values: list[int] = []
    success_count = 0
    start = time.perf_counter()
    for _ in range(args.n_replacements):
        draw_result = (
            draw_constrained_rwalk_jax_adaptive
            if args.replacement_chain_schedule is not None
            else draw_constrained_rwalk_jax
        )(
            run_key,
            target.loglike,
            target.prior_transform,
            logl_min,
            live_u,
            live_logl,
            target.ndim,
            walks=args.walks,
            step_scale=args.step_scale,
            max_attempts=args.max_attempts,
            min_accepts=args.min_accepts,
            **(
                {"replacement_chain_schedule": args.replacement_chain_schedule}
                if args.replacement_chain_schedule is not None
                else {"replacement_chains": replacement_chains}
            ),
        )
        run_key, _new_u, _new_theta, _new_logl, ncall, accepted = draw_result[:6]
        ncall_values.append(ncall)
        success_count += int(accepted)
    seconds = time.perf_counter() - start

    total_ncall = sum(ncall_values)
    mean_ncall = total_ncall / args.n_replacements if args.n_replacements else 0.0
    max_ncall = max(ncall_values, default=0)
    replacement_batch_ncall = args.walks * replacement_chains
    return {
        "target": target_name,
        "kernel": "jax",
        "replacement_chains": replacement_chains,
        "adaptive_replacement_chains": bool(
            args.replacement_chain_schedule is not None
        ),
        "replacement_chain_schedule": args.replacement_chain_schedule,
        "walks": args.walks,
        "replacement_batch_ncall": replacement_batch_ncall,
        "nlive": args.nlive,
        "n_replacements": args.n_replacements,
        "warmup_replacements": args.warmup_replacements,
        "constraint_quantile": args.constraint_quantile,
        "work_size": args.work_size if target_name == "heavy_gaussian2d" else None,
        "seconds": seconds,
        "replacements_per_second": args.n_replacements / seconds if seconds else None,
        "scalar_likelihood_calls_per_second": total_ncall / seconds
        if seconds
        else None,
        "mean_replacement_ncall": mean_ncall,
        "mean_replacement_batches": mean_ncall / replacement_batch_ncall,
        "max_replacement_batches": max_ncall / replacement_batch_ncall,
        "mean_replacement_chains_used": mean_ncall / args.walks if args.walks else None,
        "max_replacement_chains_used": max_ncall / args.walks if args.walks else None,
        "replacement_chain_usage_counts": None,
        "success_fraction": success_count / args.n_replacements
        if args.n_replacements
        else 0.0,
    }


def print_results(rows: list[dict[str, Any]]) -> None:
    print(
        "target kernel replacement_chains walks nlive n_replacements seconds "
        "replacements/s scalar_ncall/s mean_ncall mean_batches max_batches "
        "success_fraction"
    )
    for row in rows:
        print(
            f"{row['target']} {row['kernel']} {row['replacement_chains']} "
            f"{row['walks']} {row['nlive']} {row['n_replacements']} "
            f"{row['seconds']:.3g} {row['replacements_per_second']:.3g} "
            f"{row['scalar_likelihood_calls_per_second']:.3g} "
            f"{row['mean_replacement_ncall']:.3g} "
            f"{row['mean_replacement_batches']:.3g} "
            f"{row['max_replacement_batches']:.3g} {row['success_fraction']:.3g}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    rows = [
        run_one(target_name, replacement_chains, args.seed, args)
        for target_name in args.targets
        for replacement_chains in (
            [args.replacement_chains_grid[0]]
            if args.replacement_chain_schedule is not None
            else args.replacement_chains_grid
        )
    ]
    print_results(rows)
    if args.output is not None:
        Path(args.output).write_text(
            json.dumps(_jsonable({"results": rows}), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
