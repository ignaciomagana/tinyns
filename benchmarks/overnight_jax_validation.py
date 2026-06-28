"""Opt-in overnight validation/benchmark harness for fast JAX rwalk configs.

Defaults are intentionally tiny so an accidental invocation is safe.  Use larger
``--nlive``, stricter ``--dlogz``, more ``--seeds``, and ``--include-bounds`` /
``--include-block`` for overnight diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from validation.targets import available_targets, get_target  # noqa: E402

from tinyns import NestedSampler  # noqa: E402

DEFAULT_TARGETS = [
    "gaussian2d",
    "correlated_gaussian2d",
    "banana2d",
    "ring2d",
    "eggbox2d",
]
OPTIONAL_10D_TARGETS = ["gaussian10d", "correlated_gaussian10d"]
EXPECTED_KEYS = [
    "target",
    "config_name",
    "seed",
    "seconds",
    "ncall",
    "niter",
    "logz",
    "logzerr",
    "final_delta_logz",
    "replacement_failures",
    "mean_replacement_ncall",
    "mean_replacement_chains_used",
    "mean_bound_seed_calls",
    "mean_rwalk_kernel_calls",
    "bound_nellipsoids_mean",
    "bound_build_time_total",
]


@dataclass(frozen=True)
class Config:
    name: str
    kwargs: dict[str, Any]
    needs_bounds: bool = False
    needs_block: bool = False


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


def available_default_targets() -> list[str]:
    names = available_targets()
    return [name for name in DEFAULT_TARGETS if name in names]


def overnight_targets() -> list[str]:
    names = available_targets()
    targets = available_default_targets()
    targets.extend(name for name in OPTIONAL_10D_TARGETS if name in names)
    return targets


def build_configs(args: argparse.Namespace) -> list[Config]:
    common = {
        "sample": "rwalk",
        "kernel": "jax",
        "walks": args.walks,
        "max_attempts": args.max_attempts,
        "step_scale": args.step_scale,
        "min_accepts": args.min_accepts,
        "replacement_chains": args.replacement_chains,
        "bound_update_interval": args.bound_update_interval,
    }
    configs = [
        Config("unbounded_isotropic_rwalk", {**common, "bound": "none"}),
        Config(
            "unbounded_live_cov_rwalk",
            {**common, "bound": "none", "rwalk_proposal": "live-cov"},
        ),
        Config(
            "adaptive_rwalk",
            {
                **common,
                "bound": "none",
                "replacement_chain_schedule": args.replacement_chain_schedule,
            },
        ),
    ]
    if args.include_bounds:
        bounded = {
            **common,
            "rwalk_seed": "bound",
            "bound_seed_kernel": "jax",
            "allow_unused_bound": True,
        }
        configs.extend(
            [
                Config(
                    "single_bound_bounded_rwalk",
                    {**bounded, "bound": "single"},
                    True,
                ),
                Config(
                    "multi_bound_bounded_rwalk",
                    {**bounded, "bound": "multi"},
                    True,
                ),
                Config(
                    "fused_bounded_rwalk",
                    {**bounded, "bound": "multi", "fused_bound_rwalk": True},
                    True,
                ),
            ]
        )
    if args.include_block:
        configs.append(
            Config(
                "block_jax_rwalk",
                {**common, "bound": "none", "jax_block_size": args.jax_block_size},
                needs_block=True,
            )
        )
    return configs


def run_one(
    target_name: str, config: Config, seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    target = get_target(target_name)
    sampler = NestedSampler(
        target.loglike,
        target.prior_transform,
        ndim=target.ndim,
        nlive=args.nlive,
        **config.kwargs,
    )
    start = time.perf_counter()
    result = sampler.run(
        jax.random.PRNGKey(seed),
        dlogz=args.dlogz,
        maxiter=args.maxiter,
        progress=args.progress,
    )
    seconds = time.perf_counter() - start
    diagnostics = result.diagnostics()
    metadata = {} if result.metadata is None else result.metadata
    niter = diagnostics.get("niter", diagnostics.get("ndead"))
    row = {
        "target": target_name,
        "config_name": config.name,
        "seed": seed,
        "seconds": float(seconds),
        "ncall": int(result.ncall),
        "niter": None if niter is None else int(niter),
        "logz": float(result.logz),
        "logzerr": float(result.logzerr),
        "expected_logz": target.expected_logz,
        "analytic_logz": target.expected_logz,
        "final_delta_logz": diagnostics.get("final_delta_logz"),
        "replacement_failures": int(
            metadata.get(
                "replacement_failures", diagnostics.get("replacement_failures", 0)
            )
            or 0
        ),
        "mean_replacement_ncall": metadata.get(
            "mean_replacement_ncall", diagnostics.get("replacement_mean_ncall")
        ),
        "mean_replacement_chains_used": metadata.get("mean_replacement_chains_used"),
        "mean_bound_seed_calls": metadata.get("mean_bound_seed_calls"),
        "mean_rwalk_kernel_calls": metadata.get("mean_rwalk_kernel_calls"),
        "bound_nellipsoids_mean": metadata.get("bound_nellipsoids_mean"),
        "bound_build_time_total": metadata.get("bound_build_time_total"),
        "success": bool(result.success),
        "message": str(result.message),
        "ndim": target.ndim,
        "nlive": args.nlive,
        "dlogz": args.dlogz,
        "maxiter": args.maxiter,
        "config": config.kwargs,
    }
    for key in EXPECTED_KEYS:
        row.setdefault(key, None)
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=available_default_targets())
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--nlive", type=int, default=25)
    parser.add_argument("--dlogz", type=float, default=10.0)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--output", type=str, default="overnight_jax_validation.json")
    parser.add_argument("--include-bounds", action="store_true")
    parser.add_argument("--include-block", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="Use explicit tiny smoke settings."
    )
    parser.add_argument("--walks", type=int, default=5)
    parser.add_argument("--step-scale", type=float, default=0.1)
    parser.add_argument("--min-accepts", type=int, default=1)
    parser.add_argument("--replacement-chains", type=int, default=1)
    parser.add_argument(
        "--replacement-chain-schedule", nargs="+", type=int, default=[1, 2, 4]
    )
    parser.add_argument("--bound-update-interval", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--jax-block-size", type=int, default=4)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        quick = copy(args)
        quick.targets = args.targets[:1]
        quick.seeds = args.seeds[:1]
        quick.nlive = min(args.nlive, 20)
        quick.dlogz = max(args.dlogz, 10.0)
        quick.maxiter = min(args.maxiter if args.maxiter is not None else 5, 5)
        return quick
    return args


def main(argv: list[str] | None = None) -> None:
    args = apply_quick_defaults(parse_args(argv))
    rows = [
        run_one(target_name, config, seed, args)
        for target_name in args.targets
        for config in build_configs(args)
        for seed in args.seeds
    ]
    Path(args.output).write_text(json.dumps(_jsonable(rows), indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
