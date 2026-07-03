"""Synthetic 10D GW-like stress test for the recommended B32 JAX rwalk path.

This is intentionally not a production gravitational-wave likelihood. It is a
self-contained geometry stress test with GW-flavored degeneracies: mass-ratio
curvature, bounded spin parameters, distance/inclination degeneracy, a
banana-like sky mode plus a mirror mode, wrapped phase/polarization structure,
and spin/mass-ratio coupling.

The target is much harder than the included 2D validation problems. A failed run
is useful sampler-diagnostic information, not a scientific posterior recovery.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tinyns import NestedSampler  # noqa: E402

PARAMETER_NAMES = (
    "mchirp",
    "q",
    "chi_eff",
    "chi_p",
    "distance",
    "cos_iota",
    "ra",
    "dec",
    "phase",
    "psi",
)

TRUTH = {
    "mchirp": 36.0,
    "q": 0.62,
    "chi_eff": 0.08,
    "chi_p": 0.34,
    "distance": 850.0,
    "cos_iota": 0.42,
    "ra": 1.25,
    "dec": -0.35,
    "phase": 2.1,
    "psi": 0.7,
}
TRUTH_VECTOR = np.array([TRUTH[name] for name in PARAMETER_NAMES], dtype=float)


def _wrap_periodic(x, period):
    return jnp.mod(x + 0.5 * period, period) - 0.5 * period


def prior_transform(u):
    """Map the 10D unit cube to simple GW-like physical coordinates."""

    u = jnp.asarray(u)
    mchirp = 18.0 + 62.0 * u[0]
    q = u[1]
    chi_eff = -0.8 + 1.6 * u[2]
    chi_p = u[3]
    distance = 150.0 + 1850.0 * u[4]
    cos_iota = -1.0 + 2.0 * u[5]
    ra = 2.0 * jnp.pi * u[6]
    dec = jnp.arcsin(-1.0 + 2.0 * u[7])
    phase = 2.0 * jnp.pi * u[8]
    psi = jnp.pi * u[9]
    return jnp.array(
        [mchirp, q, chi_eff, chi_p, distance, cos_iota, ra, dec, phase, psi]
    )


def loglike(theta):
    """Unnormalized synthetic GW-like log likelihood."""

    (
        mchirp,
        q,
        chi_eff,
        chi_p,
        distance,
        cos_iota,
        ra,
        dec,
        phase,
        psi,
    ) = theta

    # Curved chirp-mass / mass-ratio ridge.
    dq = q - TRUTH["q"]
    mchirp_ridge = TRUTH["mchirp"] + 9.0 * dq**2 - 1.4 * dq
    logl_mass = -0.5 * (((mchirp - mchirp_ridge) / 0.38) ** 2 + (dq / 0.10) ** 2)

    # Bounded spin structure with spin-mass-ratio coupling.
    chi_eff_ridge = TRUTH["chi_eff"] + 0.55 * dq - 0.35 * (chi_p - TRUTH["chi_p"])
    chi_p_ridge = TRUTH["chi_p"] + 0.16 * dq**2 - 0.05 * dq
    logl_spin = -0.5 * (
        ((chi_eff - chi_eff_ridge) / 0.055) ** 2 + ((chi_p - chi_p_ridge) / 0.060) ** 2
    )

    # Distance/inclination amplitude degeneracy.
    amp = (1.0 + cos_iota**2) / (2.0 * distance)
    amp_truth = (1.0 + TRUTH["cos_iota"] ** 2) / (2.0 * TRUTH["distance"])
    log_amp_resid = (jnp.log(amp) - jnp.log(amp_truth)) / 0.040
    weak_distance_resid = (distance - TRUTH["distance"]) / 650.0
    logl_amp = -0.5 * (log_amp_resid**2 + weak_distance_resid**2)

    # Sky banana with a mirror mode.
    dra_primary = _wrap_periodic(ra - TRUTH["ra"], 2.0 * jnp.pi)
    dec_primary = TRUTH["dec"] + 0.23 * jnp.sin(2.0 * dra_primary)
    primary = -0.5 * ((dra_primary / 0.18) ** 2 + ((dec - dec_primary) / 0.08) ** 2)

    mirror_ra = _wrap_periodic(ra - (TRUTH["ra"] + jnp.pi), 2.0 * jnp.pi)
    mirror_dec = -TRUTH["dec"] + 0.18 * jnp.sin(2.0 * mirror_ra + 0.4)
    mirror = -0.5 * ((mirror_ra / 0.22) ** 2 + ((dec - mirror_dec) / 0.09) ** 2) - 0.45

    logl_sky = jax.nn.logsumexp(jnp.array([primary, mirror]))

    # Wrapped phase/polarization degeneracy.
    dpsi = _wrap_periodic(psi - TRUTH["psi"], jnp.pi)
    dphase = _wrap_periodic(
        phase - TRUTH["phase"] - 2.0 * dpsi,
        2.0 * jnp.pi,
    )
    logl_phase = -0.5 * ((dphase / 0.13) ** 2 + (dpsi / 0.35) ** 2)

    # One weak cross-term tying sky and amplitude together.
    cross = _wrap_periodic(ra - TRUTH["ra"], 2.0 * jnp.pi)
    cos_iota_ridge = TRUTH["cos_iota"] + 0.10 * jnp.sin(cross)
    logl_cross = -0.5 * ((cos_iota - cos_iota_ridge) / 0.35) ** 2

    return logl_mass + logl_spin + logl_amp + logl_sky + logl_phase + logl_cross


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_sampler(args: argparse.Namespace) -> NestedSampler:
    return NestedSampler(
        loglike,
        prior_transform,
        ndim=len(PARAMETER_NAMES),
        nlive=args.nlive,
        sample="rwalk",
        kernel="jax",
        walks=args.walks,
        replacement_chains=args.replacement_chains,
        rwalk_proposal="isotropic",
        step_scale=args.step_scale,
        min_accepts=args.min_accepts,
        max_attempts=args.max_attempts,
        jax_block_size=args.jax_block_size,
        rwalk_adaptive_step_scale=args.rwalk_adaptive_step_scale,
        rwalk_target_accept=args.rwalk_target_accept,
    )


def _downsample(samples: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if samples.shape[0] <= max_points:
        return samples
    rng = np.random.default_rng(seed)
    indices = rng.choice(samples.shape[0], size=max_points, replace=False)
    return samples[np.sort(indices)]


def _make_corner_plot(
    samples: np.ndarray,
    output_path: Path,
    *,
    max_points: int,
    seed: int,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    samples = _downsample(samples, max_points=max_points, seed=seed)
    ndim = len(PARAMETER_NAMES)
    fig, axes = plt.subplots(ndim, ndim, figsize=(1.55 * ndim, 1.55 * ndim))

    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue

            if row == col:
                ax.hist(samples[:, col], bins=40, density=True, histtype="step")
                ax.axvline(TRUTH_VECTOR[col], linewidth=1)
            else:
                ax.scatter(
                    samples[:, col],
                    samples[:, row],
                    s=1.0,
                    alpha=0.25,
                    rasterized=True,
                )
                ax.axvline(TRUTH_VECTOR[col], linewidth=0.8)
                ax.axhline(TRUTH_VECTOR[row], linewidth=0.8)

            if row == ndim - 1:
                ax.set_xlabel(PARAMETER_NAMES[col], fontsize=8)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(PARAMETER_NAMES[row], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _make_focus_plots(
    samples: np.ndarray,
    output_path: Path,
    *,
    max_points: int,
    seed: int,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    samples = _downsample(samples, max_points=max_points, seed=seed)
    pairs = [
        (0, 1, "mchirp/q banana"),
        (4, 5, "distance/inclination"),
        (6, 7, "sky banana/mirror"),
        (2, 3, "spin coupling"),
        (8, 9, "phase/polarization"),
    ]

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.0 * len(pairs), 3.4))
    for ax, (xidx, yidx, pair_title) in zip(axes, pairs, strict=True):
        ax.scatter(
            samples[:, xidx],
            samples[:, yidx],
            s=2.0,
            alpha=0.30,
            rasterized=True,
        )
        ax.axvline(TRUTH_VECTOR[xidx], linewidth=0.8)
        ax.axhline(TRUTH_VECTOR[yidx], linewidth=0.8)
        ax.set_xlabel(PARAMETER_NAMES[xidx])
        ax.set_ylabel(PARAMETER_NAMES[yidx])
        ax.set_title(pair_title)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _maybe_make_plots(
    result,
    seed: int,
    args: argparse.Namespace,
    seed_dir: Path,
    diagnostics: dict[str, Any],
) -> list[str]:
    if args.no_plots:
        return []
    if not result.success and not args.plot_failed:
        return []

    try:
        posterior = result.resample_equal(
            jax.random.PRNGKey(args.plot_seed + seed),
            n=args.n_plot_samples,
        )
        samples = np.asarray(posterior)
        status = "failed" if not result.success else "success"
        title = f"GW-like 10D {status}, seed={seed}"
        corner_path = seed_dir / "posterior_corner.png"
        focus_path = seed_dir / "posterior_focus.png"
        _make_corner_plot(
            samples,
            corner_path,
            max_points=args.max_plot_points,
            seed=args.plot_seed + seed,
            title=title,
        )
        _make_focus_plots(
            samples,
            focus_path,
            max_points=args.max_plot_points,
            seed=args.plot_seed + seed + 1000,
            title=title,
        )
        return [str(corner_path), str(focus_path)]
    except ImportError:
        diagnostics.setdefault("warnings", []).append(
            "matplotlib is not installed; skipping plots"
        )
        return []


def _run_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    seed_dir = args.output_dir / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    sampler = _make_sampler(args)
    start = time.perf_counter()
    result = sampler.run(
        jax.random.PRNGKey(seed),
        dlogz=args.dlogz,
        maxiter=args.maxiter,
        progress=args.progress,
    )
    seconds = time.perf_counter() - start

    diagnostics = dict(result.diagnostics())
    diagnostics["seconds"] = seconds
    diagnostics["seed"] = seed
    diagnostics["output_dir"] = str(seed_dir)
    diagnostics["plots"] = _maybe_make_plots(result, seed, args, seed_dir, diagnostics)

    result.save_npz(seed_dir / "result.npz")
    (seed_dir / "diagnostics.json").write_text(
        json.dumps(_jsonable(diagnostics), indent=2) + "\n",
        encoding="utf-8",
    )

    print(result.summary())
    print(json.dumps(_jsonable(diagnostics), indent=2))
    if not result.success and not args.plot_failed and not args.no_plots:
        print("Skipping failed-run plots. Re-run with --plot-failed to force them.")

    return diagnostics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--nlive", type=int, default=1000)
    parser.add_argument("--dlogz", type=float, default=0.5)
    parser.add_argument("--maxiter", type=int, default=80000)
    parser.add_argument("--walks", type=int, default=20)
    parser.add_argument("--replacement-chains", type=int, default=8)
    parser.add_argument("--step-scale", type=float, default=0.03)
    parser.add_argument("--min-accepts", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--jax-block-size", type=int, default=32)
    parser.add_argument("--rwalk-adaptive-step-scale", action="store_true")
    parser.add_argument("--rwalk-target-accept", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=Path("gw_like_10d_results"))
    parser.add_argument("--n-plot-samples", type=int, default=3000)
    parser.add_argument("--max-plot-points", type=int, default=3000)
    parser.add_argument("--plot-seed", type=int, default=1234)
    parser.add_argument(
        "--plot-failed",
        action="store_true",
        help=(
            "Also make posterior plots for unsuccessful runs. These plots are "
            "diagnostics only and should not be interpreted scientifically."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_attempts < args.walks * args.replacement_chains:
        raise ValueError(
            "--max-attempts must be at least --walks * --replacement-chains "
            "for one complete batched rwalk replacement attempt"
        )

    rows = [_run_seed(seed, args) for seed in args.seeds]
    payload = {
        "config": {
            "seeds": args.seeds,
            "nlive": args.nlive,
            "dlogz": args.dlogz,
            "maxiter": args.maxiter,
            "walks": args.walks,
            "replacement_chains": args.replacement_chains,
            "step_scale": args.step_scale,
            "min_accepts": args.min_accepts,
            "max_attempts": args.max_attempts,
            "jax_block_size": args.jax_block_size,
            "rwalk_adaptive_step_scale": args.rwalk_adaptive_step_scale,
            "rwalk_target_accept": args.rwalk_target_accept,
        },
        "truth": dict(TRUTH),
        "parameter_names": PARAMETER_NAMES,
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )

    successes = sum(bool(row.get("success")) for row in rows)
    failures = len(rows) - successes
    replacement_failures = sum(
        int(row.get("replacement_failures", 0) or 0) for row in rows
    )
    print(
        f"summary: success={successes}/{len(rows)} "
        f"failures={failures} replacement_failures={replacement_failures}"
    )


if __name__ == "__main__":
    main()
