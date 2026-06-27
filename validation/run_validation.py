"""Run repeated tinyns validation jobs on analytic and stress targets."""

from __future__ import annotations

import argparse
import json
import sys
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


def _posterior_moments(result, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ess = result.posterior_ess()
    n = min(5000, max(1000, int(ess)))
    samples = np.asarray(result.resample_equal(jax.random.PRNGKey(seed + 10_000), n=n))
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples, rowvar=False)
    if result.ndim == 1:
        sample_cov = np.asarray(sample_cov).reshape((1, 1))
    return sample_mean, sample_cov


def run_one(target_name: str, sampler_name: str, seed: int, args) -> dict[str, Any]:
    target = get_target(target_name)
    kwargs = {
        "sample": sampler_name,
        "nlive": args.nlive,
        "max_attempts": args.max_attempts,
        "step_scale": args.step_scale,
    }
    if sampler_name == "rwalk":
        kwargs["walks"] = args.walks
    elif sampler_name == "slice":
        kwargs["slices"] = args.slices
        kwargs["slice_steps"] = args.slice_steps

    sampler = NestedSampler(
        target.loglike,
        target.prior_transform,
        ndim=target.ndim,
        **kwargs,
    )
    result = sampler.run(
        jax.random.PRNGKey(seed), dlogz=args.dlogz, maxiter=args.maxiter
    )
    diagnostics = result.diagnostics()
    sample_mean, sample_cov = _posterior_moments(result, seed)

    logz_error = None
    abs_logz_error = None
    z_score = None
    if target.expected_logz is not None:
        logz_error = float(result.logz - target.expected_logz)
        abs_logz_error = abs(logz_error)
        if result.logzerr > 0.0 and np.isfinite(result.logzerr):
            z_score = logz_error / result.logzerr

    mean_error_norm = None
    if target.expected_mean is not None:
        mean_error_norm = float(np.linalg.norm(sample_mean - target.expected_mean))

    cov_error_frobenius = None
    if target.expected_cov is not None:
        cov_error_frobenius = float(np.linalg.norm(sample_cov - target.expected_cov))

    return {
        "target": target_name,
        "sampler": sampler_name,
        "seed": seed,
        "ndim": target.ndim,
        "logz": float(result.logz),
        "logzerr": float(result.logzerr),
        "expected_logz": target.expected_logz,
        "logz_error": logz_error,
        "abs_logz_error": abs_logz_error,
        "z_score": z_score,
        "ncall": int(result.ncall),
        "niter": diagnostics.get("niter"),
        "ndead": diagnostics.get("ndead"),
        "posterior_ess": float(result.posterior_ess()),
        "success": bool(result.success),
        "message": str(result.message),
        "warnings": diagnostics.get("warnings", []),
        "replacement_mean_ncall": diagnostics.get("replacement_mean_ncall", 0.0),
        "replacement_failures": diagnostics.get("replacement_failures", 0),
        "sample_mean": sample_mean,
        "sample_cov": sample_cov,
        "mean_error_norm": mean_error_norm,
        "cov_error_frobenius": cov_error_frobenius,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["gaussian1d", "gaussian2d", "correlated_gaussian2d"],
    )
    parser.add_argument("--samplers", nargs="+", default=["prior", "rwalk", "slice"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--nlive", type=int, default=200)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--walks", type=int, default=25)
    parser.add_argument("--slices", type=int, default=5)
    parser.add_argument("--slice-steps", type=int, default=10)
    parser.add_argument("--step-scale", type=float, default=0.1)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for target_name in args.targets:
        for sampler_name in args.samplers:
            for seed in args.seeds:
                run = run_one(target_name, sampler_name, seed, args)
                results.append(run)
                print(
                    f"target={target_name} sampler={sampler_name} seed={seed} "
                    f"logz={run['logz']:.3g} err={run['logz_error']} "
                    f"logzerr={run['logzerr']:.3g} z={run['z_score']} "
                    f"ncall={run['ncall']} ess={run['posterior_ess']:.0f} "
                    f"success={run['success']} warnings={len(run['warnings'])}"
                )

    if args.output is not None:
        payload = {"config": vars(args), "results": results}
        Path(args.output).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


if __name__ == "__main__":
    main()
