# tinyns

`tinyns` is a tiny, dynesty-style nested sampler for JAX-friendly likelihoods.
The core public API is deliberately small: provide `loglike`,
`prior_transform`, and `ndim`, then call `NestedSampler(...).run(key)`.

TinyNS is not many samplers; it is one excellent tiny static nested sampler,
plus reference baselines. TinyNS deliberately keeps the sampler surface small.
The main optimized path is static nested sampling with JAX random-walk
replacement. Other samplers are kept as reference baselines or experimental
research knobs, not as equally supported production paths.

## Install

From source:

```bash
git clone <repo-url>
cd tinyns
python -m pip install .
```

Editable development install:

```bash
git clone <repo-url>
cd tinyns
python -m pip install -e '.[dev]'
```

## Minimal working example

```python
import jax
import jax.numpy as jnp
from tinyns import NestedSampler


def prior_transform(u):
    return -10.0 + 20.0 * u


def loglike(theta):
    return -0.5 * theta[0] ** 2 - 0.5 * jnp.log(2 * jnp.pi)


key = jax.random.PRNGKey(0)
sampler = NestedSampler(loglike, prior_transform, ndim=1, nlive=200)
result = sampler.run(key, dlogz=0.1)

print(result.summary())
```

The defaults run the recommended fast path (JAX `rwalk` with live-cov
proposals and cached blocks of 32 iterations; see below), so `loglike` and
`prior_transform` must be JAX-traceable. For plain Python or NumPy functions,
pass `kernel="python"`.

## Minimal API

| API | Purpose |
| --- | --- |
| `NestedSampler(loglike, prior_transform, ndim, ...)` | Dynesty-style sampler facade for static nested sampling. |
| `NestedSamplingResult` | Result container with samples, weights, evidence estimates, status, and metadata. |
| `result.summary()` | Human-readable run summary. |
| `result.diagnostics()` | Plain-dict diagnostics, including ESS, call counts, and warnings. |
| `result.resample_equal(key, n=None)` | Equally weighted posterior samples via systematic resampling. |
| `result.to_numpy()` | Plain dictionary with array fields converted to NumPy arrays. |
| `result.to_dynesty_dict()` | Lightweight dynesty-compatible dictionary using matching tinyns fields. |

`NestedSampler` accepts many optional keyword arguments (`sample`, `kernel`,
`bound`, `walks`, `rwalk_proposal`, and others documented below). An unknown
keyword argument is not an error: it is still stored, preserving dynesty
drop-in compatibility, but `NestedSampler` emits a `UserWarning` naming the
unrecognized keyword(s) so typos or unsupported options are not silently
ignored.

## Saving and loading results

Final results can be saved without extra dependencies using NumPy `.npz`:

```python
from tinyns import NestedSamplingResult

result.save_npz("run.npz")
loaded = NestedSamplingResult.load_npz("run.npz")
```

The `.npz` file stores weighted samples, evidence estimates, status, and
JSON-serialized metadata. Equal-weight posterior samples are not stored because
they can be regenerated with `resample_equal`.

HDF5 is not part of the core package to keep dependencies minimal. If needed,
HDF5 support can be added later as an optional extra.

## Progress and callbacks

`tinyns` has dependency-free progress reporting:

```python
result = sampler.run(key, progress=True, progress_interval=50)
```

For custom logging or early stopping, pass a callback:

```python
def callback(state):
    print(state["iter"], state["logz"], state["dlogz"])
    if state["iter"] > 1000:
        return False


result = sampler.run(key, callback=callback, callback_interval=25)
```

Returning `False` from the callback stops the run gracefully and returns a
partial `NestedSamplingResult`.

## Sampler recommendations

TinyNS is intentionally organized around one recommended fast path, with
reference baselines and experimental research knobs separated from that primary
route. The fast path is the default. It should still be revalidated for new
target geometries.

| Tier | Options | Status |
| --- | --- | --- |
| Recommended fast path (default) | `sample="rwalk"`, `kernel="jax"`, `rwalk_proposal="live-cov"`, `walks=max(25, 6 * ndim)` (12 in 1-D), `replacement_chains=1`, `jax_block_size=32` | Likelihood calls per iteration stay at `walks` on correlated Gaussians, d = 2..18 |
| Isotropic block path | `rwalk_proposal="isotropic"`, `walks=5`, `jax_block_size=32` | Validated on the included 2D benchmark targets; cost per iteration grows geometrically at d >= 4 |
| Reference baseline | `sample="rwalk"`, `kernel="python"` | Simple CPU/Python correctness/debug baseline |
| Reference baseline | `sample="prior"` | Conceptual brute-force constrained-prior baseline |
| Experimental | bounds / fused bounds / bounded block | Useful research direction; not production-ready |
| Experimental | adaptive replacement-chain schedules | Useful tuning knob; not the main recommended path |

Removed: slice/random-slice samplers were removed to keep TinyNS small (use dynesty for slice-based external comparisons). An earlier live-cov proposal that reflected moves at the unit-cube faces, and its `rwalk_cov_jitter` option, were also removed; the current `rwalk_proposal="live-cov"` rejects such moves instead.

### Recommended fast JAX rwalk path

The defaults are the fast path:

```python
from tinyns import NestedSampler

sampler = NestedSampler(loglike, prior_transform, ndim)
result = sampler.run(key, dlogz=0.1)
```

This is `sample="rwalk"`, `kernel="jax"`, `rwalk_proposal="live-cov"`,
`walks=max(25, 6 * ndim)` (12 in 1-D), `step_scale=0.5`, `replacement_chains=1` and
`jax_block_size=32`. Each live-cov step is `step_scale * L @ z`, where `L` is
the Cholesky factor of the live-point covariance in the unit cube and `z` is a
standard normal vector. Moves that leave the unit cube are rejected. The step
follows the contracting, correlated live set, and `step_scale` is only the
initial value: live-cov always adapts it toward `rwalk_target_accept` (0.25).

The old default, a fixed isotropic unit-cube step (`step_scale=0.1`), does not
follow the contracting live set: its acceptance collapsed, and the cost per
iteration grew geometrically at d >= 4. On anisotropic correlated Gaussians, d = 2..18,
live-cov keeps the likelihood calls per iteration flat at `walks`. Unbiased
evidence needs `walks` of about 5-6 x `ndim`: `walks=25` biased logZ high by
+0.5 nats at 13D and +1.4 nats at 18D, while `walks >= 5 * ndim` was within
about 0.2 nats, with seed scatter matching the reported `logzerr`. Hence the
default `walks=max(25, 6 * ndim)`. In 1-D, 10 walks were already unbiased
(20 seeds, bias 0.2 x `logzerr`), so 1-D defaults to 12; from 2-D on, 15 walks
still biased logZ on Gaussian and banana targets.

Live-cov is supported only for unbounded JAX rwalk with a fixed
`replacement_chains`. If you choose `kernel="python"`, a bound, or a
`replacement_chain_schedule` and leave `rwalk_proposal` unset, the proposal
falls back to `"isotropic"` with `step_scale=0.1` and `jax_block_size=1`.
Passing `rwalk_proposal="live-cov"` explicitly in those combinations raises
`NotImplementedError`. `sample="prior"` defaults to `kernel="python"`.

With `kernel="jax"`, the initial live points are evaluated in one compiled
pass.

Large arrays used by `loglike` or `prior_transform` are passed to the compiled
kernels as arguments instead of being embedded as constants (faster compiles,
less memory). A closure gets this automatically: tinyns traces it once, and
every captured array with at least 4096 elements, `jax.Array` or `np.ndarray`
and including operands of a nested `jax.jit` call, becomes a kernel argument,
while smaller constants stay embedded, so a hoisted closure compiles the same
program as the equivalent pytree callable. If tracing the closure fails, it
keeps closure semantics. Arrays captured inside the body of
an inner `jax.jit` function (rather than passed to it) stay embedded.

A pytree callable is the explicit form, for example
`jax.tree_util.Partial(loglike_fn, data)` with `loglike_fn(data, theta)`, or an
equinox-style module or registered dataclass with `__call__`: its `jax.Array` /
`np.ndarray` leaves become kernel arguments and everything else stays static.
Use `jax.Array` leaves so the data move to the device once. Both forms apply to
the fast path (unbounded JAX rwalk, block and per-iteration modes, the rescue
ladder and the initial live-point pass, without `jax_vectorized`); the bounded,
fused-bounded and `replacement_chain_schedule` kernels still close over the
callables.

For a campaign over mock datasets in one process, pass
`jax.tree_util.Partial(loglike_fn, data_i)`: pytree callables with the same
structure and same-shaped leaves reuse one compiled kernel, while each closure
compiles its own. tinyns keeps no reference to a finished run's data, so a
dataset is freed once you drop it. `result.metadata` reports `wall_time_s`,
`compile_s` (time to the first block) and `mean_ms_per_call` (after it).

Separate processes (array jobs, one dataset each) can share compiles through
JAX's persistent cache: call
`jax.config.update("jax_compilation_cache_dir", "/path/to/cache")` before the
first run, with the same path in every process (compiles under 1 s are not stored).

`jax_block_size > 1` batches several nested-sampling replacement iterations
into one cached, jitted JAX block. This reduces Python/JAX dispatch overhead,
which usually gives a large speedup for cheap or moderately expensive JAX
likelihoods. For very expensive likelihoods, the speedup may be smaller because
likelihood cost dominates dispatch overhead. Convergence is checked between
blocks, not after every individual nested iteration, so a run may overshoot the
requested `dlogz` threshold by up to roughly `jax_block_size - 1` iterations.

`jax_block_size=32` (the default for unbounded JAX rwalk) is the fastest
validated block size. Use `jax_block_size=16` if you want a more conservative
block size with slightly less convergence overshoot. Use `jax_block_size=1` for
the most conservative behavior, which disables block mode. This recommendation
is based on current validation on the included benchmark targets; it is not a
proof for all likelihoods.


### Bound update interval

For `bound="multi"`, rebuilding every iteration can be expensive. Use
`bound_update_interval` to reuse a bound for multiple nested-sampling
iterations. Larger intervals reduce Python/bound-building overhead but can make
bounds stale. Validate evidence before relying on results.


### Batched JAX replacement chains

For GPU-native likelihoods, `rwalk+jax` can run several independent replacement chains in parallel:

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    walks=25,
    replacement_chains=16,
)
```

Here `walks` is the length of each chain, while `replacement_chains` is the number of independent chains run in parallel per replacement batch. This can improve wall time on GPU by evaluating many proposals in parallel.

The replacement remains valid only if a successful chain is selected without favoring higher-likelihood endpoints. `tinyns` selects randomly among successful chains.

> Warning: Increasing `replacement_chains` increases likelihood evaluations per replacement attempt. It is useful only when the likelihood benefits from batched/device parallelism.

For batched JAX chains, `ncall` counts scalar likelihood evaluations, not wall-clock-equivalent work. A replacement with `walks=25` and `replacement_chains=16` costs 400 scalar likelihood evaluations, but those chains are evaluated in parallel on device. Use wall time and replacement batch counts when judging batched performance.

For JAX rwalk, `repl_ncall` is scalar likelihood calls per replacement. In fixed chain mode, `repl_chains` reports the effective number of parallel chains used per replacement. In adaptive mode, `usage=...` reports how often each stage in the replacement chain schedule was used. Use these diagnostics to choose the smallest chain count or schedule that avoids retry tails.


### Adaptive JAX replacement-chain schedules

Fixed `replacement_chains` runs the same number of independent chains for every replacement. This can waste work when most chains succeed.

For JAX rwalk, `tinyns` can instead start with a small batch and escalate only if needed:

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    walks=25,
    replacement_chain_schedule=(1, 4, 16, 64, 256),
)
```

Use this when replacement difficulty varies across the nested-sampling run. The sampler returns as soon as any stage succeeds and randomly selects among successful chains in that stage. This avoids always paying for large batches.

`replacement_chain_schedule` requires `jax_block_size=1` (the default when a
schedule is given). It is rejected with a clear error when combined with `jax_block_size > 1`: the
cached block kernel does not implement schedule-aware early return, so it
cannot benefit from a schedule and would otherwise silently overstate the
savings. Use the host-level per-iteration path (`jax_block_size=1`) for
adaptive replacement-chain schedules.

> Warning: Adaptive schedules do not replace validation. Check evidence calibration and insertion-rank diagnostics on representative targets.

For non-JAX likelihoods, or when debugging sampler behavior, use `kernel="python"` with `sample="rwalk"`.

`kernel="jax"` currently supports `sample="rwalk"` only. The top-level nested-sampling loop remains in Python; only the constrained replacement kernel is compiled.

For local constrained samplers, step-count parameters are decorrelation lengths:

- `walks`: number of random-walk proposals per replacement attempt for `rwalk` (default `max(25, 6 * ndim)`, 12 in 1-D)
- `min_accepts`: accepted rwalk moves a chain must make before it is kept (default `0`: no retry; see below)

The sampler does not return merely after the first accepted local move; it runs the requested local update length.

`sample="prior"` supports vectorized replacement proposals with
`vectorized=True`; the full nested-sampling loop remains a small Python loop.
Vectorized `rwalk` replacement sampling is not implemented yet, so
`vectorized=True` needs an explicit `sample="prior"`.

### Bounding

`tinyns` supports `bound="none"` by default. Bounds are experimental modifiers for rwalk, not a separate public sampler mode. Use `sample="rwalk"` with `bound="single"` or `bound="multi"` and `rwalk_seed="bound"`. Bounds are built in unit-cube coordinates from the live points and enlarged by `bound_enlargement`.

The only currently recommended fast path is unbounded JAX rwalk with live-cov proposals and cached block mode. Bounded rwalk uses isotropic proposals. Bounded/fused-bounded paths remain experimental and require target-specific validation.

Bounding is experimental. Validate evidence and insertion-rank diagnostics on representative targets before relying on it for scientific results.


### Experimental adaptive rwalk step scale

Live-cov always adapts its step scale. For isotropic proposals, `rwalk_adaptive_step_scale=True` is an explicitly experimental JAX-only rwalk option that adapts the isotropic proposal scale from constrained-replacement acceptance telemetry. It defaults to off, with a fixed `step_scale=0.1`.

This is intended for hard-target diagnostics where a fixed rwalk scale is a poor compromise. It is not a dynesty replacement, does not add slice/rslice or `sample="bound"`, and is not a substitute for checking insertion-rank diagnostics, replacement diagnostics, and seed/config stability on the target.

### Bounded rwalk

For experimental dynesty-style bounded rwalk, use both a bound and bound seeding:

```python
sampler = NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    bound="multi",
    rwalk_seed="bound",
    rwalk_proposal="isotropic",
    walks=5,
    replacement_chains=16,
)
```

Setting `bound="multi"` alone does not define a bounded rwalk transition unless `rwalk_seed="bound"` is also enabled. `tinyns` raises a clear error for `bound != "none"` with live-seeded rwalk unless `allow_unused_bound=True`. Use `allow_unused_bound=True` only when you intentionally want to build bounds for diagnostics or overhead measurements while keeping ordinary live-seeded rwalk.

`fused_bound_rwalk=True` currently means the bounded seed draw and rwalk transition are exposed as one replacement path and share accounting. It is not yet a single compiled seed+rwalk kernel. A future implementation may replace this wrapper fusion with a true single-dispatch JAX kernel.

For `bound="none"`, `jax_block_size=32` uses a cached JAX `lax.scan` over several nested-sampling iterations and is the recommended fast path described above. For bounded rwalk, block mode remains experimental: the current mode reuses a fixed bound across a Python-level block and is mainly a stepping stone toward a fully compiled bounded block kernel.

### Multiellipsoid bounding

`bound="multi"` is an experimental dynesty-style union-of-ellipsoids bound. It recursively splits the live points using a dependency-free PCA/median split and samples from the volume-weighted union of ellipsoids with overlap correction.

An experimental bounded/fused candidate configuration for separate validation is:

```python
NestedSampler(
    loglike,
    prior_transform,
    ndim,
    sample="rwalk",
    kernel="jax",
    bound="multi",
    rwalk_seed="bound",
    rwalk_proposal="isotropic",
    walks=5,
    replacement_chains=16,
)
```

This mode is experimental and is not the recommended fast path. Check evidence calibration, insertion-rank diagnostics, and seed stability before using it for science.

### JAX bound representation

`tinyns` keeps Python-friendly bound objects for readability, but also provides an internal padded `JaxEllipsoidBound` representation. The padded representation is used as a bridge toward fast JAX replacement kernels and should not change public sampling behavior.

## Current validation status

The live-cov default was checked on anisotropic correlated Gaussians, d = 2..18 (see the fast-path section above). The isotropic block path (`sample="rwalk"`, `kernel="jax"`, `rwalk_proposal="isotropic"`, `walks=5`, `replacement_chains=1`, `jax_block_size=32`) has been checked with repeated-seed validation on the included benchmark targets:

- `gaussian2d`
- `correlated_gaussian2d`
- `ring2d`
- `banana2d`
- `eggbox2d`

The analytic Gaussian targets show good evidence calibration in the current validation suite, and qualitative targets show acceptable insertion-rank diagnostics. TinyNS focuses on one optimized static nested-sampling path, JAX rwalk with cached block mode, plus small reference baselines. Bounds are tracked separately as experimental. Users should still validate on their own target geometry before relying on evidence values.

### `min_accepts`

The default is `min_accepts=0`: each replacement runs one chain of `walks`
proposals and keeps wherever it ends. A chain that never moved returns a copy
of its seed (a live point strictly above the threshold); that happens in about
1% of replacements at 25 walks and under 0.1% at 108.

`min_accepts >= 1` restores the old rule: discard chains with fewer accepted
moves and retry. That rule is biased. Chains fail to move more often in regions
the proposal fits poorly (a minor mode, a narrow ridge), so retrying
under-samples them. On two-mode targets with a true 6% minor mode, the old
default `min_accepts=1` gave 2.9% at 4-D (logZ bias +0.13 nats) and 1.7% at
10-D; the new default gives 6.3% and 6.7%.

## Multimodal posteriors

Replacement chains start from live points and cannot cross between separated
modes. Each mode's live-point count then does a random walk, so the mode
weights scatter from seed to seed: on a real 18-D posterior with a 16% minor
mode, two runs gave 7% and 32%. Neither `logzerr` nor the insertion ranks show
this.

On the fast path, tinyns tracks clusters of the live points between blocks.
Once two clusters each hold enough of the live volume, a tenth of the chain
steps become affine swaps `u' = mu_b + L_b L_a^-1 (u - mu_a)`: the point moves
from its cluster `a` to the same relative place in another cluster `b`. The
swap is accepted with the Metropolis factor `det L_b / det L_a`. This is exact
for any cluster frames. It pulls the populations toward the modes' true
volumes, while the rwalk keeps the global live covariance.
`cluster_swap=False` turns it off.

On two-mode Gaussian targets with a 6% minor mode (nlive 500, 20 seeds per
cell), the seed scatter of the minor mode's mass in logit units went from
0.46 to 0.15 at 4-D, 0.83 to 0.16 at 10-D and 1.28 to 0.58 at 18-D. The mean
stayed on the truth (0.060, 0.057 and 0.070 +/- 0.009), and logZ did not
change. Runs made 2% to 5% fewer likelihood calls. The host-side clustering
costs about 12 ms per update at 13-D, one update every 128 iterations.

Limits:

- A minor mode needs its volume share to predict at least about 3 x `ndim`
  live points (`nlive * V_minor / V_total >= 3 * ndim`); below that the swap
  stays off and the weights drift as before. Raise `nlive`.
- The frames are ellipsoids. Curved or truncated modes lower the swap
  acceptance, and the result moves back toward the drift, but stays valid.
- A unimodal run does not switch the swap on (the clustering made no false
  split in our tests) and is then bit-identical to v0.2.4.

If the weight of a small or non-ellipsoidal mode matters, split the prior into
one region per mode and run each, or run several seeds and compare.
`result.metadata["cluster_modes"]` lists each detected mode's mass, its
smallest live count between detection and its posterior median, and
`urn_logit_sd`, the seed scatter of `logit(mass)` that the drift would cause
without the swap; `cluster_swap_accepts` and `cluster_swap_proposals` count
the swaps.

## Design philosophy

- **Tiny:** keep dependencies and abstractions minimal.
- **Dynesty-style:** expose familiar `loglike + prior_transform + ndim` entry
  points and lightweight dynesty-compatible exports.
- **JAX-friendly:** support JAX arrays, PRNG keys, and JAX-compatible callbacks
  without requiring users to define model objects.
- **Correctness and diagnostics first:** prefer clear bookkeeping, tests, and
  warnings over premature speedups.

## Limitations

`tinyns` is an early (v0.2) implementation. It has been validated against
dynesty and analytic evidences on 1-18 dimensional targets, including
gravitational-wave population likelihoods. It is intentionally not a
probabilistic programming language.

- Static nested sampling only.
- No dynamic nested sampling.
- The default `kernel="jax"` needs JAX-traceable `loglike` and
  `prior_transform`; use `kernel="python"` otherwise.
- Live-cov proposals need unbounded JAX rwalk with a fixed
  `replacement_chains`; bounds, `kernel="python"` and replacement-chain
  schedules use isotropic proposals.
- Unbiased evidence needs `walks` of about 5-6 x `ndim`, so likelihood calls
  per iteration grow linearly with dimension.
- Strongly curved posteriors need more walks than that default. On a 10-D
  Rosenbrock-like target, `walks=60` (6 x `ndim`) biased logZ by +0.26 nats,
  and about 1000 walks were needed. If a new model's posterior may be strongly
  curved, check logZ against a run with several times more `walks`.
- The reported `logzerr` (`sqrt(H / nlive)`, and `logz_bootstrap()`) covers
  the prior-volume path. It does not cover residual chain correlation. In 20-seed
  ensembles, the seed-to-seed scatter matched `logzerr` in 1-D (ratio 0.94) and
  was about 1.3x `logzerr` at 13-D with the default walks (1.1x with twice the
  walks). At high dimension, treat `logzerr` as up to about 25% optimistic, or
  raise `walks`.
- Mode weights of separated modes are reliable only when each mode holds at
  least about 3 x `ndim` live points (see Multimodal posteriors).
- Multiellipsoid bounding is experimental.
- No full vectorized `rwalk` replacement sampler.
- Not a PPL; users provide functions, not model objects.
- Replacement attempts are capped by `max_attempts`; hitting the cap returns
  `success=False` with a partial result rather than raising during the run.

## Additional examples

### Primary

- `examples/gaussian_2d_rwalk_jax_block.py`: recommended fast path with the defaults (JAX `rwalk`, live-cov proposals, cached block mode with `jax_block_size=32`).

### Reference

- `examples/gaussian_2d.py`: 2D Gaussian with the default sampler.
- `examples/gaussian_2d_rwalk.py`: reflected isotropic Python random-walk constrained sampling.
- `examples/gaussian_2d_rwalk_jax.py`: JAX-native random-walk replacement without block mode (`jax_block_size=1`).

### Utility

- `examples/progress_and_callback.py`: dependency-free progress and callbacks.
- `examples/vectorized_gaussian_2d.py`: vectorized prior-rejection proposals.
- `examples/checkpoint_resume.py`: checkpoint and resume with the rwalk baseline.

### Experimental

- `examples/banana_2d.py` and `examples/eggbox_2d.py`: qualitative target demos that can exercise non-primary sampler options for experimentation.
- `examples/repeated_gaussian_evidence.py` and `examples/save_load_result.py`: workflow/diagnostic demos for coverage or reproducibility checks.

## Validation

For repeated-seed validation on analytic targets:

```bash
python validation/run_validation.py --output validation_results.json
python validation/summarize_validation.py validation_results.json
```

The validation harness is intended to catch calibration and reliability issues,
not to be a formal speed benchmark.
The validation summary includes heuristic calibration warnings such as repeated
large evidence z-scores, high final-live weight fraction, and concentrated
posterior weights.

## Checkpoint and resume

`run(...)` and `resume(...)` both accept `maxiter`, the maximum number of
nested-sampling replacement iterations. It defaults to `10_000 * ndim` and, if
given explicitly, must be a positive integer (`maxiter >= 1`). A run still
stops early once `dlogz` is satisfied (or a callback returns `False`)
regardless of `maxiter`.

Long static nested-sampling runs can save active checkpoints:

```python
result = sampler.run(
    key,
    checkpoint_path="run.checkpoint.npz",
    checkpoint_interval=100,
)
```

Checkpoints are written every `checkpoint_interval` nested-sampling
iterations, counting elapsed iterations since the last checkpoint (or since
the run/resume start) rather than an iteration-modulo check. This applies in
both per-iteration mode and block mode (`jax_block_size > 1`): a run with, for
example, `jax_block_size=32` and the default `checkpoint_interval=100` still
checkpoints roughly every 100 iterations rather than only after the first
block boundary that happens to land on a multiple of 100.

A checkpoint stores the active live points, accumulated dead points, PRNG key,
iteration counters, and sampler metadata. It does not serialize user functions.
To resume, reconstruct the same sampler and call:

```python
result = sampler.resume("run.checkpoint.npz")
```

The sampler configuration must match the checkpoint. Checkpoints record the
resolved `walks`, `step_scale`, `rwalk_proposal` and `jax_block_size`, so a
sampler built with the same (or default) arguments resumes it. A checkpoint
written by an earlier version with default arguments records the old defaults
(`sample="prior"`, `kernel="python"`, `rwalk_proposal="isotropic"`, ...);
pass those values explicitly to resume it. Checkpoints are distinct
from final result files saved with `result.save_npz(...)`. Checkpoints use NumPy
`.npz` files to avoid extra dependencies. Checkpoint/resume is intended for
static nested sampling; dynamic nested sampling is not implemented yet.
