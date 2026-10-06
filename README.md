# tinyns

`tinyns` is a tiny, dynesty-style nested sampler for JAX-friendly likelihoods.
The core public API is deliberately small: provide `loglike`,
`prior_transform`, and `ndim`, then call `NestedSampler(...).run(key)`.

TinyNS is not many samplers; it is one tiny static nested sampler: a
random-walk replacement that follows the live-point covariance, run in jitted
JAX blocks. The sampler surface is deliberately small.

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

`loglike` and `prior_transform` must be JAX-traceable functions of one point:
tinyns `jax.vmap`s and `jax.jit`s them.

## Minimal API

| API | Purpose |
| --- | --- |
| `NestedSampler(loglike, prior_transform, ndim, ...)` | Static nested sampler (see the keyword arguments below). |
| `NestedSamplingResult` | Result container with samples, weights, evidence estimates, status, and metadata. |
| `result.summary()`, `print(result)` | Human-readable run summary, with a per-mode table when the posterior has several modes. |
| `result.diagnostics()` | Plain-dict diagnostics, including ESS, call counts, the modes, and warnings. |
| `result.modes()` | Per-mode mass, its seed-to-seed error bar and an `unresolved` flag (see Multimodal posteriors). |
| `result.insertion_test(windows=3)` | Kolmogorov-Smirnov test of the insertion ranks, pooled and per stretch of the run. |
| `result.resample_equal(key, n=None)` | Equally weighted posterior samples via systematic resampling. |
| `result.to_numpy()` | Plain dictionary with array fields converted to NumPy arrays. |
| `result.to_dynesty_dict()` | Lightweight dynesty-compatible dictionary using matching tinyns fields. |

The full constructor is

```python
NestedSampler(
    loglike,
    prior_transform,
    ndim,
    nlive=500,
    *,
    walks=None,  # None -> max(25, 6 * ndim); 12 if ndim == 1
    replacement_chains=1,  # chains run per replacement; one is kept
    block_size=32,  # nested-sampling iterations per jitted block
    cluster_swap=None,  # None -> on when replacement_chains == 1
)
```

Any other keyword argument raises `TypeError`, with the closest known name
suggested when there is one.

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

## The sampler

TinyNS has one sampler: static nested sampling with a random-walk replacement
that runs entirely in JAX.

```python
from tinyns import NestedSampler

sampler = NestedSampler(loglike, prior_transform, ndim)
result = sampler.run(key, dlogz=0.1)
```

Each replacement runs `replacement_chains` chains of `walks` steps from live
points strictly above the likelihood threshold and keeps one chain that ends
inside the constraint. A chain that never moved is kept as a copy of its seed:
discarding unmoved chains and retrying would under-sample regions where the
proposal fits poorly. Each step is `scale * L @ z`, where `L` is the Cholesky
factor of the live-point covariance in the unit cube and `z` is a standard
normal vector. Moves that leave the unit cube are rejected; with one chain they
do not call the likelihood. The step follows the contracting, correlated live
set. `scale` starts at 0.5 and adapts after every block toward 25% move
acceptance.

On anisotropic correlated Gaussians, d = 2..18, the likelihood calls per
iteration stay flat at about `walks`. Unbiased evidence needs `walks` of about
5-6 x `ndim`: `walks=25` biased logZ high by +0.5 nats at 13D and +1.4 nats at
18D, while `walks >= 5 * ndim` was within about 0.2 nats, with seed scatter
matching the reported `logzerr`. Hence the default `walks=max(25, 6 * ndim)`.
In 1-D, 10 walks were already unbiased (20 seeds, bias 0.2 x `logzerr`), so
1-D defaults to 12; from 2-D on, 15 walks still biased logZ on Gaussian and
banana targets.

The initial live points are evaluated in one compiled pass.

### Data in the likelihood

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
Use `jax.Array` leaves so the data move to the device once.

For a campaign over mock datasets in one process, pass
`jax.tree_util.Partial(loglike_fn, data_i)`: pytree callables with the same
structure and same-shaped leaves reuse one compiled kernel, while each closure
compiles its own. tinyns keeps no reference to a finished run's data, so a
dataset is freed once you drop it. `result.metadata` reports `wall_time_s`,
`compile_s` (time to the first block) and `mean_ms_per_call` (after it),
summed over a run and its resumes.

Separate processes (array jobs, one dataset each) can share compiles through
JAX's persistent cache: call
`jax.config.update("jax_compilation_cache_dir", "/path/to/cache")` before the
first run, with the same path in every process (compiles under 1 s are not stored).

### Block size

Every `block_size` nested-sampling iterations run as one jitted block, which
removes the Python/JAX dispatch overhead per iteration. This usually gives a
large speedup for cheap or moderately expensive likelihoods; for very
expensive likelihoods the likelihood cost dominates and the speedup is
smaller. Convergence is checked, and the step scale adapted, between blocks,
so a run may overshoot the requested `dlogz` threshold by up to
`block_size - 1` iterations. `block_size=32` (the default) is the fastest
validated block size; smaller blocks overshoot less, and `block_size=1`
checks after every iteration.

### Batched replacement chains

For GPU-native likelihoods, several independent replacement chains can run in
parallel:

```python
sampler = NestedSampler(loglike, prior_transform, ndim, replacement_chains=16)
```

Here `walks` is the length of each chain, while `replacement_chains` is the
number of independent chains run in parallel per replacement batch. This can
improve wall time on GPU by evaluating many proposals in parallel.

The replacement remains valid only if a successful chain is selected without
favoring higher-likelihood endpoints. `tinyns` selects randomly among
successful chains.

> Warning: Increasing `replacement_chains` increases likelihood evaluations per replacement. It is useful only when the likelihood benefits from batched/device parallelism.

With several chains, `ncall` counts scalar likelihood evaluations, not
wall-clock-equivalent work, and every proposal is evaluated, in or out of the
cube. A replacement with `walks=25` and `replacement_chains=16` costs 400
scalar likelihood evaluations, but those chains are evaluated in parallel on
device. Use wall time and replacement batch counts when judging batched
performance. The cluster swap (below) needs a single chain.

If no chain ends inside the constraint, the replacement runs another batch,
up to `10_000 // (walks * replacement_chains)` batches (at least one). Chains
start strictly above the threshold, so this happens only on a likelihood
plateau. When every batch fails, the run stops with `success=False` and a
message, and returns the dead and live points so far.

## Current validation status

The default was checked on anisotropic correlated Gaussians, d = 2..18 (see
above), and against dynesty and analytic evidences on 1-18 dimensional
targets. The validation harness (`validation/`) runs repeated seeds on the
included targets:

- `gaussian2d`
- `correlated_gaussian2d`
- `ring2d`
- `banana2d`
- `eggbox2d`

Users should still validate on their own target geometry before relying on
evidence values.

## Multimodal posteriors

Replacement chains start from live points and cannot cross between separated
modes. Each mode's live-point count then does a random walk, so the mode
weights scatter from seed to seed: on a real 18-D posterior with a 16% minor
mode, two runs gave 7% and 32%. Neither `logzerr` nor the insertion ranks show
this.

With one replacement chain (the default), tinyns tracks clusters of the live points between blocks.
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

- A minor mode needs its volume share to predict at least about 2 x `ndim`
  live points (`nlive * V_minor / V_total >= 2 * ndim`); below that the swap
  stays off and the weights drift as before. Raise `nlive`: the mode is
  reliably resolved once it holds about 3 x `ndim` live points when it is
  first detected.
- The frames are ellipsoids. Curved or truncated modes lower the swap
  acceptance, and the result moves back toward the drift, but stays valid.
- A unimodal run that the clustering does not split is bit-identical to
  v0.2.4. Real posteriors are sometimes split (3 of 10 seeds on a 13-D
  spectral-siren mock); the swap is exact either way and those runs agree with
  v0.2.4 within `logzerr`.

If the weight of a small or non-ellipsoidal mode matters, split the prior into
one region per mode and run each, or run several seeds and compare.
`result.modes()` lists each mode found by clustering the posterior after the
run (with the swap on or off): its `mass`, `min_live`, its smallest live count
between its isolation and its posterior median, and `urn_sd`, the seed scatter
of `logit(mass)` that the drift causes without the swap (an upper bound with
it). A mode with `min_live < 3 * ndim` is flagged `unresolved`: raise `nlive`.
A mode lost before the end of a run leaves no trace in it.
`result.metadata["cluster_swap_accepts"]` and `["cluster_swap_proposals"]`
count the swaps.

## Design philosophy

- **Tiny:** keep dependencies and abstractions minimal.
- **Dynesty-style:** expose familiar `loglike + prior_transform + ndim` entry
  points and lightweight dynesty-compatible exports.
- **JAX-friendly:** support JAX arrays, PRNG keys, and JAX-compatible callbacks
  without requiring users to define model objects.
- **Correctness and diagnostics first:** prefer clear bookkeeping, tests, and
  warnings over premature speedups.

## Limitations

`tinyns` is an early (v0.3) implementation. It has been validated against
dynesty and analytic evidences on 1-18 dimensional targets, including
gravitational-wave population likelihoods. It is intentionally not a
probabilistic programming language.

- Static nested sampling only.
- No dynamic nested sampling.
- `loglike` and `prior_transform` must be JAX-traceable scalar functions of
  one point.
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
- Not a PPL; users provide functions, not model objects.
- A replacement that finds no admissible point (a likelihood plateau) stops
  the run with `success=False` and a partial result rather than raising.

## Additional examples

### Primary

- `examples/gaussian_2d_rwalk_jax_block.py`: the defaults on a 2D Gaussian.

### Reference

- `examples/gaussian_1d.py`, `examples/gaussian_2d.py` and `examples/constant.py`: analytic evidences.

### Utility

- `examples/progress_and_callback.py`: dependency-free progress and callbacks.
- `examples/checkpoint_resume.py`: checkpoint and resume.
- `examples/save_load_result.py`: save and load a result.

### Qualitative targets

- `examples/banana_2d.py` and `examples/eggbox_2d.py`: curved and multimodal demos.
- `examples/repeated_gaussian_evidence.py`: repeated seeds against the analytic evidence.

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
the run/resume start) rather than an iteration-modulo check, at block
boundaries: a run with `block_size=32` and the default
`checkpoint_interval=100` checkpoints every 128 iterations rather than only at
a block boundary that happens to land on a multiple of 100. A run also writes
a final checkpoint when it ends.

A checkpoint is one NumPy `.npz` file (format `tinyns-ckpt-2`), written
atomically: the live set and the run scalars (including the PRNG key, as raw
key data, so typed and legacy keys both round-trip), the dead points so far,
the resolved sampler configuration, the step-scale and acceptance telemetry,
and the cluster tracker's state. It does not serialize user functions.
To resume, reconstruct the same sampler and call:

```python
result = sampler.resume("run.checkpoint.npz")
```

The sampler configuration must match the checkpoint. Checkpoints record
`ndim`, `nlive` and the resolved `walks`, `replacement_chains`, `block_size`
and `cluster_swap`, so a sampler built with the same (or default) arguments
resumes it, and a mismatch raises `ValueError` naming the key. The
`loglike` and `prior_transform` cannot be checked: resuming with different
callables silently continues the run with them. A run killed and resumed at
any block boundary is bit-identical to an uninterrupted one. A checkpoint
saved after a replacement failure cannot be resumed. Checkpoints written
before v0.3 are not read (`ValueError: not a tinyns-ckpt-2 file`), and there
is no converter. Checkpoints are distinct from final result files saved with
`result.save_npz(...)`. Checkpoint/resume is intended for static nested
sampling; dynamic nested sampling is not implemented yet.
