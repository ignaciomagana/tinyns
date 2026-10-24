# tinyns

`tinyns` is a tiny, general-purpose nested sampler written entirely in JAX, in
the spirit of `emcee` and `tinygp`. Give it a log likelihood, a prior transform
and the number of dimensions; it returns the evidence and weighted posterior
samples. The whole run executes on the device (GPU or CPU), it handles
multimodal posteriors inside the sampler, and it is not tied to any application.

## Install

`tinyns` needs Python 3.10+ and JAX 0.4.31+. It is not on PyPI yet; install
the 1.0.0 release from GitHub:

```bash
python -m pip install "git+https://github.com/ignaciomagana/tinyns.git@v1.0.0"
```

Without `@v1.0.0` you get the development version (`main`). For a GPU,
install the matching JAX build first (see the JAX install guide). For
development: clone, `python -m pip install -e '.[dev]'`, then `make test`.

## Quickstart

```python
import jax
import jax.numpy as jnp
from tinyns import NestedSampler


def prior_transform(u):  # unit cube -> parameters: uniform on [-10, 10]^2
    return -10.0 + 20.0 * u


def loglike(theta):  # one point -> log likelihood
    return -0.5 * jnp.sum(theta**2) - jnp.log(2 * jnp.pi)


sampler = NestedSampler(loglike, prior_transform, ndim=2)
result = sampler.run(jax.random.key(0))
print(result.summary())  # logz is -log(400) = -5.99 within logzerr
```

`loglike` and `prior_transform` are JAX-traceable functions of one point
(tinyns vmaps them). `prior_transform` returns one array (of any shape:
derived parameters stay in `result.samples`; a dict or tuple raises
`TypeError`) and only sees the open unit cube `(0, 1)^ndim`, so `ndtri(u)` or
`-log(u)` stay finite. A NaN likelihood counts as `-inf`, and a region where
the likelihood is `-inf` or NaN (outside a model's support, say) counts as
prior volume with zero likelihood: `logz` is the evidence over the full
prior. A likelihood of `+inf` is not valid: the run stops once a live point
has one, with `success=False`, the status `"invalid"`, `logz = +inf` and a
`RuntimeWarning` (usually a float32 overflow: write the likelihood in log
space or enable x64). Exact likelihood ties (a zero-likelihood region, a
finite plateau, a floor) are handled as plateaus (see How it works). The whole
API is `NestedSampler(loglike, prior_transform, ndim, nlive, *, num_delete,
walks)` and `sampler.run(key, *, dlogz=0.1, maxiter, maxcall, progress,
checkpoint, batched_data)`. `key` is a PRNG key or an int seed, here and
wherever tinyns takes a key. The run stops when the live points hold less than
`dlogz` of the evidence, at `maxiter` dead points or `maxcall` likelihood
calls, or on a likelihood plateau. `progress=True` prints one line per chunk
of steps.

## The result

```python
from tinyns import NestedSamplingResult

print(result.logz, result.logzerr)  # log evidence and its error
samples, weights = result.samples, result.weights()  # weighted posterior
equal = result.resample_equal(1, n=2000)  # equal weights; key or int seed
print(result.modes())  # modes: mass, its scatter, smallest live count, flags
print(result.insertion_test()["pvalue"])  # insertion-rank uniformity test
print(result.diagnostics()["warnings"])  # the checks, as a list of warnings
result.save_npz("result.npz")
result = NestedSamplingResult.load_npz("result.npz")
```

The arrays (`samples`, `samples_u`, `logl`, `logwt`, `logl_birth`, `nlive_i`,
`labels`) hold the dead points in order, then the final live points;
`insertion_ranks` holds the rank of each new point among the survivors of its
step and `insertion_slots` its number of possible ranks, which
`insertion_test()` reads. `metadata` has the settings and timings; `to_dict()`
and `to_numpy()` return every field, and `to_dynesty_dict()` gives
dynesty-style keys. The weights are computed in NumPy float64,
`resample_equal` returns systematic draws in a random order, `save_npz(path)`
writes exactly `path`, atomically, and `load_npz` keeps the saved dtypes.

A run that found no finite likelihood (a bug in `loglike`, or a support
smaller than about `1 / nlive` of the prior) ends with `success=False`, the
status `"invalid"`, `logz = -inf` and a `RuntimeWarning`; `summary()` and
`diagnostics()` still work, and the weights and `resample_equal()` raise a
`ValueError`. After a run that ended on a likelihood plateau, `diagnostics()`
skips the checks of the final live weight and of `final_delta_logz` (the
final live points are the plateau) and notes the plateau's posterior mass.

## How it works

Each step deletes the `k = num_delete` lowest live points and replaces them
with the end points of `k` Metropolis chains of fixed length `walks`, run in
parallel (vmapped). Each chain starts from a live point above the new
likelihood contour `L*`; proposals that leave the open unit cube or fall to
`L*` or below are rejected, and a chain that never moves returns its seed.

- **Local random walk.** The live points are clustered on the device every
  quarter e-fold of prior volume. A walk step is `x + s L z`, with `L` the
  Cholesky factor of the covariance of the cluster of `x`, and the exact
  Hastings correction when the proposal lands in another cluster.
- **Hop.** Every 10th chain step is an independence Metropolis-Hastings
  proposal from the union of ellipsoids fitted to the clusters that are large
  enough to be eligible (with fewer than two, the hop is off). It moves
  points between modes and keeps the mode weights right.
- **Cross-fitting.** The live slots form three folds. A chain seeded in one
  fold uses clusters and covariances fitted to the other two only (as emcee
  moves each half of its walkers with the other half).
- **Evidence.** The `k` deaths of a step are counted at live counts
  `m, ..., m-k+1` (birth/death contours; Fowlie, Handley & Su); `logz` and
  `logzerr` are recomputed on the host in float64. `logzerr` is the
  first-order spread of `logz` over the prior-volume path (every shrinkage
  `t ~ Beta(n, 1)`), tied deaths included: `sd(logz) / logzerr` was 1.00 to
  1.28 on 2-D Gaussians and 1.02 to 1.07 on a flat-topped likelihood (128
  runs). The `logzerr` figures of the sweeps and benchmarks below used the
  sum of information increments it replaced; without ties the first-order
  `logzerr` is 3% smaller (10-D) to 11% larger (2-D).
- **Plateaus.** New points land strictly above `L*`, so live points tied at
  one likelihood (a zero-likelihood region, a finite plateau, a floor) are
  left behind and die over several steps. Following Fowlie, Handley & Su
  (2021), the `q` tied deaths count at `m, m-1, ..., m-q+1`: a point born
  above the tie is not live for the tied deaths still to come. The sampler
  carries this in its state, so the running evidence, the stopping rule and
  `result.nlive_i` all count it, and it ranks each new point among the
  survivors strictly above `L*` for the insertion test. On a 5-D Gaussian
  with zero likelihood on half the cube, logZ was 0.17 to 0.19 too high
  without it (at every `nlive` and `num_delete`) and is now unbiased
  (-0.007 ± 0.018 over 48 runs at nlive 500). On a gravitational-wave
  population likelihood that is zero on 52% of its prior, 6 runs gave
  -192.89 against a truth of -193.15; recounted, they give -193.11 ± 0.08.
  The final live points still tied at the last contour (a run that ends on an
  indicator or step likelihood) count on from the tied deaths too: logZ of a
  uniform likelihood on a disc of area 0.25 (2-D, defaults) was 0.13 too high
  without it and is +0.009 ± 0.010 (32 runs).
- **Driver.** A compiled `while_loop` runs chunks of steps on the device; the
  host syncs once per chunk (progress, checkpoints, stopping).

**Exactness.** A chain's kernel is built without its own fold, so it does not
depend on the chain's seed or its ancestors, and both moves leave the uniform
law on `{L > L*}` invariant. The end point therefore has that law for any
chain length. What remains is common to every MCMC-driven nested sampler: a
new point is correlated with its seed (hence `walks`), and the step scale
adapts on past chains.

**Speed.** Measured on one NVIDIA H100 in float32, on a 13-D correlated
Gaussian run to `dlogz = 0.1` with the default settings of each version (v1
before the fixes of its release review, which were not timed on the GPU):

| live points | v1, compiled | v1, first run | v0.2.5, compiled | v0.2.5, first run |
|---|---|---|---|---|
| 1000 | 0.93 s | 9.8 s | 47 s | 53 s |
| 4000 | 1.15 s | 10.3 s | 227 s | 233 s |

"Compiled" is a second run in the same process; "first run" includes the
compilation (about 8 s for v1). v1 is 50x faster at 1000 live points and 200x
at 4000 once compiled, and 5x and 23x on a first run. With `num_delete`
proportional to `nlive` the number of steps does not depend on `nlive`, so for
a cheap likelihood the run time on a GPU is nearly flat in `nlive`. With
`num_delete=1` the chains run one at a time and the gain is small: 35 s and
136 s compiled, 1.3x and 1.7x faster than v0.2.5. A two-mode target in 10
dimensions took 0.91 s compiled against 0.66 s for a 10-D Gaussian (15 to 20%
more time per likelihood call with two modes and the hop active); the Gaussian
pays for the mode tracking too (see Multimodal posteriors).

## Multimodal posteriors

A random walk cannot cross between separated modes, so without help the number
of live points in each mode drifts and the mode weights scatter from seed to
seed. With the hop, that scatter was about 7x smaller than with chains seeded
from live points alone, at the same number of likelihood calls (Gaussian
mixtures of two and three modes in 4 to 32 dimensions).

A mode needs about `5 * ndim` live points for its own walk to mix. In the
validation cells where every mode held that many, the weights showed no bias
within the measurement error. Below it a mode's weight is biased or scatters
widely, and the mode can be lost; `result.modes()` (and `summary()`) flag it
`unresolved: raise nlive`. The mode tracking is a fixed cost per step, which
unimodal runs pay too: about 6% of the run time at 1 ms per likelihood call,
but about 40% of a compiled run of the 13-D Gaussian of Speed (0.93 s against
0.56 s before the mode tracking).

`result.modes()` counts the modes from the clusters the run recorded, and
gives each mode's mass, its run-to-run scatter measured within the run
(`run_sd`), the spread of its mass over the folds (`fold_sd`), the scatter to
expect without the hop (`urn_sd`: right for two modes or modes of one shape;
three modes of different widths scattered 1.5 to 4 times more) and two flags.

- `fold_sd`: the standard deviation, in logit, of the mode's mass over the
  three cross-fitted folds (`fold_mass`). A point and its descendants stay in
  one fold, so each fold samples the modes like a run of its own, but every
  fold weighs its samples with the same prior volumes, so `fold_sd` misses
  their error. Where the hop balances the modes the folds agree: `fold_sd`
  was 0.04 on average on a two-mode mixture in 18-D, 0.1 on `sepW_d10` and
  `sepW_d18`.
- `run_sd`: `sqrt(fold_sd**2 / 3 + V)`, with `V` the error of the prior
  volumes that the folds share, which dominates when the heaviest mode is also
  the narrowest (a 4-D mixture of masses 0.75, 0.2 and 0.05: logit scatter
  0.056, 0.053 and 0.079, median `run_sd` 0.053, 0.046 and 0.065,
  `fold_sd / sqrt(3)` 0.015, 0.012 and 0.027). One run's value is good to a
  factor of about 2, if the run kept every mode.
- `unresolved`: the mode held fewer than `5 * ndim` live points (a unimodal
  run too: a 32-D Gaussian was 9 nats high at nlive 40 and right at 100).
- `tracked`: `False` if the sampler's clustering never separated the mode from
  its neighbours. The hop then did not balance it, its mass scatters as
  without the hop (by about `run_sd`: each fold drifts on its own), and the
  number of such modes is a lower bound. An eggbox is the example: its 18
  modes sit on a lattice that no two-way split separates, and `modes()` finds
  13 to 18 of them from the gaps between their live points (nlive 1000, 192
  runs), none tracked.

`diagnostics()` and `summary()` turn both flags into warnings. They also warn
when the hop balanced the modes only in part: after the modes separated and
while two of them held live points, the clustering held 1.5 clusters per mode
or more on average (it cut the modes into lumps) or the hop was off (fewer
than two clusters eligible to propose from) for half of the posterior mass or
more, or, over the whole run, the hop accepted less than 2% of its proposals.
That warning quotes the fold masses and `run_sd` of each mode but the
heaviest.
And they say when the clustering's 8 slots were full during a run with
several modes: no further mode could be split off then.

## Choosing `walks`

The default, `max(25, 6 * ndim, ndim**2 // 6)`, keeps the logZ bias below its
scatter on correlated Gaussians from 2 to 64 dimensions, which holds for
posteriors that are roughly Gaussian in the unit cube. A nonlinear prior
transform that presses most coordinates against faces of the cube needs
more: with a normal prior and the data 3 prior sd off centre in all 6
coordinates (`ndtri_d6` in `bench/validate.py`), the default 36 walks left
logZ 0.24 +- 0.08 low with a scatter 3.5 times `logzerr` (64 runs), and it
took 144 (4 times the default) to remove both. The default was enough for a
curved target once there were enough live points: the logZ of a 10-D
Rosenbrock (60 walks) was 0.16 +- 0.05 low at `nlive=500`, 0.05 +- 0.04 at
1000 (0.09 +- 0.04 in 32 more runs in float32) and unbiased from 2000, with a
scatter of 1.0 to 1.1 times `logzerr` (40 runs each).

Too few walks bias logZ and inflate its scatter, which the insertion test
barely sees (an unmoved chain returns its seed, and the copies keep the ranks
uniform). `diagnostics()` warns when 30% of the chains or more did not move
(`metadata["unmoved_fraction"]`; at most 7% on the targets of `bench` at the
default walks and `nlive=500`): there logZ scattered 1.2 to 2.9 times
`logzerr` on 5-D and 10-D Gaussians. It catches gross shortfalls only. If in
doubt, rerun with twice the walks and check that logZ moves by less than
`logzerr` (that check flagged every bad run of `ndtri_d6`).

## Choosing `nlive` and `num_delete`

The defaults are `nlive=1000` and `num_delete = nlive // 10`. They come from a
sweep of 362 cells of 40 runs each on an H100: `nlive` from 250 to 4000,
`num_delete / nlive` from 0.05 to 0.5, on Gaussians (2 to 64 dimensions),
two- and three-mode mixtures (10 to 32), a 10-D Rosenbrock, a 10-D funnel and
the eggbox.

**`nlive`.** `logzerr` falls as `1 / sqrt(nlive)` and the likelihood calls
grow in proportion to it: at 1000, `logzerr` is 0.17, 0.31 and 0.44 on 10-,
32- and 64-D Gaussians after 2, 18 and 125 million calls. On a GPU with a
cheap likelihood the run time hardly depends on `nlive` (0.71 s at 1000,
0.85 s at 4000 on a 10-D Gaussian), so raise it freely there. Raise it when:

- *a small mode matters.* A mode needs `5 * ndim` live points to be weighted
  correctly, twice that if it is much thinner than the main mode. For a mode
  holding a fraction `f` of the posterior that is `nlive` of about
  `5 * ndim / f` to `10 * ndim / f`. A 6% mode was weighted without bias and
  never lost from `nlive` 500, 1000 and 2000 at 10, 18 and 32 dimensions
  (1000, 2000 and 4000 for a thin one, still lost in 1 run of 40 at 32). At
  half of that the thin mode was lost in 5 to 40% of the runs.
- *`modes()` flags a mode unresolved.* The flag is a reason to raise `nlive`,
  but its absence is not proof. A mode that is lost leaves no trace in its own
  run: every run that lost the 6% mode reported one mode and no warning. At
  `nlive=1000` the flag was raised in 7 of the 57 runs (of 360) whose minor
  weight was lost or off by more than 0.4 in the logit, all of them at 18 and
  32 dimensions. If a small mode matters, run again at twice the `nlive` and
  compare the modes.
- *the posterior has many modes.* Each of the eggbox's two smallest peaks (2%
  of the posterior) fell below a tenth of its weight in 32% of the runs at
  `nlive=500`, 15% at 1000, 4% at 2000 and none at 4000; its 8% peaks were
  kept from 1000. Its logZ was right at every `nlive`, and its peak weights
  scatter as `1 / sqrt(nlive)` (see Limitations).
- *the target is curved.* The logZ of the 10-D Rosenbrock was 0.16 too low at
  500, 0.05 at 1000 and unbiased from 2000 (at the default `walks`).
- *`ndim` is above about `nlive / 10`.* At `nlive=250` logZ was 0.2 to 0.4 too
  high at 32 and 64 dimensions; 1000 was unbiased up to 64.
- *`NestedSampler` warns that `nlive` is too few for `ndim`* (below
  `1.5 * (ndim + 1)`, where the walk covariances of the folds cannot span
  every direction): a 32-D Gaussian was 16 to 19 nats high at nlive 20, with
  `logzerr` under 1.

**`num_delete`.** For a given number of likelihood calls the accuracy of logZ
does not depend on it: a larger share shrinks the volume faster per death and
loses as much in effective live points. It sets the run time instead (a 10-D
Gaussian at `nlive=1000`, compiled: 1.1, 0.7, 0.5 and 0.3 s at a share of 0.05,
0.1, 0.25 and 0.5; 27 s at `num_delete=1`). Stay at `nlive // 10` unless you
have a reason:

- Above it the diagnostics degrade. On the eggbox logZ scattered 1.5 to 2.5
  times `logzerr` at shares of 0.25 and 0.5 (0.9 to 1.2 at 0.1), a mode near
  the `5 * ndim` limit was lost more often, and the insertion test rejected
  13% and 30% of correct runs at the 5% level (6% at 0.1 and below).
- `num_delete=1` is for a likelihood that already fills the device in one
  call. A single run then skips the likelihood of a proposal rejected without
  it (outside the cube, or by a hop's or local walk's Hastings test), which
  saved 8 to 17% of the calls, and its `logzerr` is 2% smaller; a batch of
  runs evaluates those proposals anyway, uncounted (5 to 17% more than
  `ncall`).
  If the device can evaluate several likelihoods at once, the default is
  faster by the batch size.

**Memory** was no constraint on these targets: 40 batched runs at 64
dimensions peaked at 4 GiB of GPU memory with `nlive=4000` (7 GiB at a share
of 0.5), and one run holds `niter * ndim` samples on the host (740 thousand
dead points at 64-D, `nlive=4000`). A heavy likelihood is different (see GPU
notes).

## GPU notes

- **Precision.** tinyns runs in JAX's default precision, float32. Set
  `JAX_ENABLE_X64=1` when `|loglike|` is large (float32 resolves about 7
  digits) or the posterior is very narrow in the unit cube. On an A100 or
  H100, float32 matrix products run in TF32 by default, about 3 digits (on
  the H100 a 512 x 512 product had a relative error of 2.9e-4, against 1.1e-6
  at `"highest"`). tinyns's own products ask for full precision; for a
  likelihood built from `@`, `jnp.dot` or `einsum` set
  `jax.config.update("jax_default_matmul_precision", "highest")` or use x64.
- **Compile cache.** `jax.config.update("jax_compilation_cache_dir", path)`
  lets separate processes reuse compiled programs.
- **Memory.** A step evaluates `num_delete` likelihoods at once (times the
  batch size of a batched run), and so does the start of a run. For a heavy
  likelihood the default does not fit: a gravitational-wave population
  likelihood over 259 events and 4096 samples per event asked for 67 GB at
  `num_delete=200`, and a 13-D mock of it 78 GB at 100, on an 80 GB H100.
  Running out raises `MemoryError`; lower `num_delete` (25 and 100 fitted
  there), or run fewer keys per call in a batched run. The posterior does not
  depend on it. The final prior map takes blocks of at least 1024 rows and
  halves one that does not fit.
- **Reproducibility.** On a GPU in float32 a fixed seed is not bit-reproducible
  between processes: five identical runs of a 32-D mixture gave four different
  `logz`. On a CPU it is. The results agree in distribution.

## Checkpoints

```python
result = sampler.run(jax.random.key(0), checkpoint="run.checkpoint.npz")
```

The run writes `run.checkpoint.npz` atomically at most every 10 minutes and
at the end, keeping the mode of the file it replaces; a path that cannot be
written raises `OSError` before the run starts. If the file exists, the run
resumes from it. On a CPU the result is bit-identical to an uninterrupted run
on the same JAX version (a checkpoint of another JAX build or device resumes
with a warning); on a GPU in float32, where a fixed seed is not
bit-reproducible between processes, the two agree in distribution. A run
stopped by `maxiter` or `maxcall` continues when called again with a larger
limit.

A checkpoint of a different configuration, key or precision is refused, and
so is one of a different likelihood, prior or dataset: on resume 16 live and
16 dead points are evaluated again (in each run of a batch, with its data),
and they must give back the stored likelihoods to a relative `sqrt(eps)`. In
float32 that misses a change below about `3.5e-4 |logL|` (from about `1e5`
data points, possibly a new noise realisation), so use one checkpoint path
per problem, for example one per dataset of a mock or SBC loop.

## Batched runs and SBC

```python
from jax.tree_util import Partial


def loglike_data(data, theta):  # the data are the first argument
    return -0.5 * jnp.sum((data - theta[0]) ** 2)


data = jax.random.normal(jax.random.key(2), (16, 20))  # 16 datasets
like = Partial(loglike_data, data)
sampler = NestedSampler(like, prior_transform, ndim=1, nlive=200)
results = sampler.run(jax.random.split(jax.random.key(3), 16), batched_data=True)
```

A batch of keys runs that many independent runs in one compiled program and
returns a list of results. With `batched_data=True` every array of a pytree
callable carries a leading batch axis: one dataset per run, for
simulation-based calibration and mock campaigns. An array that every run
shares (a lookup table) needs that axis too: tile it, or let `fn` capture it,
which embeds it in the compiled program. A batched run agrees with the
run of its key alone in distribution, not bit for bit. A stopped run is
frozen, not skipped, so the device evaluates about `B` times the calls of the
longest run: batch runs of similar length.

## Pytree callables

```python
for one in data[:2]:  # same shapes: compiled once
    like = Partial(loglike_data, one)
    print(NestedSampler(like, prior_transform, ndim=1, nlive=200).run(0).logz)
```

Pass data as `jax.tree_util.Partial(fn, data)` or any pytree callable. Its
arrays, NumPy float scalars such as `sigmas[i]` included, become (traced)
arguments of the compiled program, so datasets of one shape share one
compilation. Python scalars and NumPy integers and booleans are part of the
program, so wrap a per-dataset Python float in `jnp.asarray`. An array that
`fn` captures instead of taking it as an argument is embedded in the program
and stays alive while it is cached. A plain function or closure is cached by
identity, and a bound method (`model.loglike`) is a new object at each
access, so each `NestedSampler(model.loglike, ...)` compiles again: store it
once (`ll = model.loglike`).

## The functional core

```python
from tinyns import STATUS, Config, delta_logz, finalise, init, step

cfg = Config(ndim=2, nlive=200)
state = init(0, loglike, prior_transform, cfg)
rows = []
while delta_logz(state, cfg) >= 0.1:  # evidence left in the live points
    state, dead = step(state, loglike, prior_transform, cfg)
    rows.append(dead)
    if STATUS[int(state.status)] != "running":  # a plateau, or +inf
        break
# The rows of the first state.it steps are valid (a step on a plateau leaves
# the state as it was and returns rows that are not).
dead = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs)[: int(state.it)], *rows)
result = finalise(state, dead, cfg, prior_transform=prior_transform, dlogz=0.1)
```

`State` and `Dead` are pytrees of arrays, so `step` works under `jax.jit`,
`lax.scan` and `jax.vmap` (under `jax.vmap` it runs all three chain kernels
and a recluster at every step: batch runs with `run`). Pass the same
`loglike` and `prior_transform` function objects at every `step`: a new
function, closure or bound method compiles it again (a pytree callable is
keyed on its structure). `step` does not test termination; the loop does.
Given the loop's `dlogz`, `finalise` applies the stopping rule of `run` and
reports `converged` (without it: `success=False`, `running`).
`tinyns.STATUS[int(state.status)]` names the status, which `step` changes only
on a likelihood plateau (`"plateau"`) or at a likelihood of `+inf`
(`"invalid"`): stop there. `finalise` takes the rows of exactly the first
`state.it` steps. `state.ncall` is an int32: a loop that may pass `2**31`
likelihood calls counts them on the host and gives `finalise(...,
ncall=total)`, as `examples/functional_core.py` does.

`examples/` has five scripts, each under a minute on a CPU: `quickstart.py`,
`multimodal.py`, `checkpoint_resume.py`, `sbc_batched.py` and
`functional_core.py`. The test suite runs them and the snippets above.

## Limitations

- **Small modes.** A mode with fewer than about `5 * ndim` live points is not
  sampled reliably. `modes()` flags it; raise `nlive`.
- **Modes with thin arms or a funnel neck.** When a mode's contours are a core
  with thin arms, or narrow into a neck (a scale parameter pressed against its
  prior edge and coupled to a location), its live points are heavy-tailed
  along a few directions. The clustering cuts each mode into lumps, and the
  hop's ellipsoids cover the modes poorly. On the 18-D `needle_d18` target of
  `bench` the hop cut the run-to-run logit scatter of the minor mass from 0.57
  to 0.19 (16 runs at nlive 1000), against 0.37 to 0.019 for the same modes
  without arms; on `cauchy_d18` from 0.53 to 0.32. On an 18-D posterior with
  a funnel neck it accepted 0.6% of its proposals and gave no gain: the
  minor mass scattered by 0.25 over 3 runs, against 0.30 for v0.2.5, and the
  three folds' masses differed by a logit sd of 1.0 to 1.6 within each run.
  The mode masses then scatter as for a sampler without the hop; logZ is
  unaffected (`needle_d18`: +0.03 +- 0.08 from the truth). `diagnostics()`
  warns, and `run_sd` gives the scatter. `modes()` merges lumps that the
  likelihood does not dip between, but a lump joined to its mode only along a
  curved path, such as a funnel neck, can be reported as a small mode of its
  own (unresolved; 1 of the 3 runs above), and the clustering's labels may
  also be merged into one mode (2 of 16 `cauchy_d18` runs).
- **Small modes in high dimensions go unreported.** At 32 dimensions and the
  default `nlive`, the clustering did not separate a 6% mode (about 60 live
  points, below `5 * ndim`) in 32 to 36 of 40 runs of `sepW_d32`. `modes()`
  then reports one mode and no warning. The samples still carry the mode at
  its weight (logit scatter 0.14 to 0.17, never lost), so the posterior and
  logZ are right; only the mode count is wrong. At 18 dimensions the same
  mode was missed in 0 of 40 runs with x64 and 2 of 40 in float32. If the
  number of modes matters, raise `nlive` until every mode holds `5 * ndim`
  live points.
- **Close modes.** `modes()` reports two modes as one when the likelihood dips
  by less than 3 between them and the bottom of the dip lies above the
  posterior median likelihood of at least one of them: two equal Gaussians
  closer than about 3.3 standard deviations in 2-D, 4.8 in 5-D and 5.4 from
  7-D on. A close mode of less mass merges further out (in 2-D, a 0.1 or 0.2
  mode at 4.5 to 5.5 standard deviations in 70 of 72 runs), with no warning.
  The samples, logZ and the total mass are unaffected.
- **Separated rings in two dimensions.** The clustering's split test cannot
  separate two well-separated rings in 2-D (its statistic is at most 24.5,
  against a threshold of 25): on the Feroz-Hobson 2-D shells most runs warn,
  the hop barely balances the rings, and 1 run in 32 reported one mode
  without a warning. logZ is unaffected.
- **Many modes.** The clustering has 8 slots and splits two ways, so it does
  not separate more modes than that, or modes on a lattice (an eggbox). The
  evidence is still right, but the hop does not balance those modes: their
  weights scatter from seed to seed (logit sd 0.4 for the eggbox's 8% modes
  at nlive 1000, 0.6 for its 4% modes) and a 2% mode was lost (below a tenth of its
  weight) in 15% of the runs. `modes()` marks such modes `tracked: False`, and counts only the ones
  that planes across the axes of the unit cube separate.
- **Curved targets.** A 10-D Rosenbrock is biased low at small `nlive`: by
  0.16 at `nlive=500` and 0.05 to 0.09 at 1000, unbiased from 2000 (see
  Choosing `walks`).
- **Posteriors pressed against the faces of the unit cube** by a nonlinear
  prior transform need more `walks`: at the default, logZ comes out low and
  `logzerr` 2 to 3.5 times too small (see Choosing `walks`).
- **Expensive likelihoods.** Parallel chains do not reduce the number of
  likelihood calls, and the proposals a single `num_delete=1` run skips
  (outside the cube, or rejected by a Hastings test) are evaluated with
  `num_delete > 1` and in a batch of runs. When one call already fills the
  device, use `num_delete=1` in a single run.
- **float32.** Likelihood differences below about `1e-7 * |loglike|` are lost
  (about `1e-3` for a likelihood built from matrix products on an A100 or
  H100 at JAX's default matmul precision; see GPU notes); enable x64. On two-
  and three-mode mixtures from 10 to 32 dimensions (40 runs each at the
  default `nlive`, H100) float32 and x64 gave the same logZ and mode weights,
  except where a mode held fewer than `5 * ndim` live points (`sepM_d18`:
  minor-mode logit scatter 0.54 in float32, 0.16 in x64).
- **JAX debug flags.** tinyns runs under `jax_numpy_rank_promotion="raise"`,
  `jax_numpy_dtype_promotion="strict"` and `jax_debug_nans` (which does not
  catch a NaN from `loglike`: the sampler maps it to `-inf`), but not under
  `jax_debug_infs` (`-inf` is a legitimate likelihood) or under
  `jax_debug_key_reuse` with a typed key (an int seed or a
  `jax.random.PRNGKey` works).
- **Insertion test.** Its p-values are honest up to `num_delete = nlive // 10`
  and anti-conservative above (see Choosing `nlive` and `num_delete`).
- **Finite plateaus with few points above them.** When a plateau holds nearly
  all the live points, the fraction of prior volume above it is estimated
  from the few that are not, and the estimate is noisy: with 5 to 10 of 500
  live points above a plateau (`L = -5` on half of the 5-D cube above), logZ
  was +0.06 ± 0.04 from the truth at `num_delete=25` (96 runs) and +0.03 ±
  0.07 at 1 (24 runs), with a scatter of 0.41 and 0.35. `logzerr` counts the
  uncertainty of that fraction (0.35 on average), but the spread is skewed,
  which `logz_bootstrap()` shows. Raise `nlive` there. A likelihood that is
  `-inf` on all but a small part of the prior is the same case at the start
  of a run: with 1 to 10 of 500 initial points finite in a 10-D box, logZ came
  out within -0.10 to +0.08 of the truth (half a `logzerr`) once that volume
  noise is removed. A zero-likelihood region with a fair share of the live
  points above it is unbiased, with the right `logzerr`, and so is a
  likelihood that ends on a plateau.
- **Compile cache.** A process keeps up to 8 compiled configurations.

## Benchmarks

tinyns v1 was run against tinyns 0.x, BlackJAX NSS, JAXNS, dynesty, UltraNest
and Nautilus on 24 targets with a known evidence: 20 seeds per target, 40 on
the mixtures. `bench/RESULTS.md` has every cell, the settings and the
validation gate; `bench/README.md` describes the harness. The competitors ran
at nlive 500 or their own defaults (JAXNS with 500 chains, Nautilus with its
2000 live points, UltraNest with MLFriends up to d = 10 and its slice sampler
above). The times are not like for like: tinyns, BlackJAX and JAXNS ran on one
H100, the others on 4 CPU cores with the likelihood called from Python. The
tinyns rows predate the fixes of its release review.

Evidence. Each cell is the logZ bias ± its standard error; likelihood calls;
time (compiled for tinyns and BlackJAX, with a fixed cost of about 9 s still
in BlackJAX's; with cached programs for JAXNS):

| sampler | on | `gauss_d16` | `gauss_d64` | `rosen_d10` |
|---|---|---|---|---|
| **tinyns v1**, nlive 500 | H100 | -0.05 ± 0.06; 2.4e6; 2.1 s | +0.21 ± 0.13; 6.2e7; 19 s | -0.22 ± 0.06; 1.3e6; 2.6 s |
| **tinyns v1**, nlive 1000 (default) | H100 | -0.06 ± 0.05; 4.8e6; 2.2 s | -0.04 ± 0.12; 1.3e8; 20 s | -0.06 ± 0.05; 2.6e6; 2.7 s |
| tinyns 0.x | H100 | -0.02 ± 0.05; 2.2e6; 42 s | +0.91 ± 0.12; 3.2e7; 513 s | -0.38 ± 0.10; 1.2e6; 31 s |
| BlackJAX NSS | H100 | -0.02 ± 0.06; 3.8e6; 13 s | +3.27 ± 0.16; 5.6e7; 77 s | -0.06 ± 0.15; 2.0e6; 11 s |
| JAXNS | H100 | +0.07 ± 0.07; 1.1e7; 67 s | +0.06 ± 0.10; 1.7e8; 360 s | -0.17 ± 0.20; 7.1e6; 56 s |
| dynesty | 4 cores | +0.24 ± 0.06; 8.9e5; 133 s | +7.43 ± 0.12; 3.2e7; 4082 s | -0.03 ± 0.19; 6.3e5; 96 s |
| UltraNest | 4 cores | +0.17 ± 0.09; 3.4e6; 460 s | +1.57 ± 0.14; 7.5e7; 10951 s | timed out (>14405 s) |
| Nautilus | 4 cores | -0.07 ± 0.00; 1.6e5; 1160 s | timed out (>14405 s) | -0.04 ± 0.00; 1.6e5; 2104 s |

Mixtures with a 6% minor mode (`sepM`: a narrow one; `mix3`: three modes of
weight 0.7, 0.2 and 0.1). Each cell is the sd over seeds of the logit of a
minor mode's weight; the seeds that lost a mode; calls; time:

| sampler | on | `sepW_d10` | `sepW_d18` | `sepM_d18` | `sepM_d32` | `mix3_d10` |
|---|---|---|---|---|---|---|
| **tinyns v1**, nlive 500 | H100 | 0.14; 0%; 1.2e6; 1.9 s | 0.40; 0%; 3.8e6; 3.5 s | 1.08; 68%; 3.8e6; 3.3 s | 1.48; 70%; 1.2e7; 6.5 s | 0.06; 0%; 1.2e6; 2.0 s |
| **tinyns v1**, nlive 1000 (default) | H100 | 0.06; 0%; 2.4e6; 2.0 s | 0.06; 0%; 7.5e6; 3.6 s | 0.40; 20%; 7.5e6; 4.0 s | 0.85; 55%; 2.3e7; 8.1 s | 0.05; 0%; 2.4e6; 2.1 s |
| **tinyns v1**, nlive 2000 (*) | H100 | 0.03; 0%; 4.8e6; 2.1 s | 0.04; 0%; 1.5e7; 3.9 s | 0.03; 0%; 1.5e7; 4.1 s | 0.34; 28%; 4.6e7; 9.2 s | 0.03; 0%; 4.8e6; 2.3 s |
| **tinyns v1**, nlive 4000 (*) | H100 | 0.02; 0%; 9.6e6; 2.4 s | 0.02; 0%; 3.0e7; 4.4 s | 0.02; 0%; 3.0e7; 4.4 s | 0.08; 2%; 9.3e7; 10 s | 0.02; 0%; 9.6e6; 2.6 s |
| tinyns 0.x | H100 | 0.13; 0%; 9.8e5; 33 s | 0.27; 0%; 3.1e6; 80 s | 0.82; 57%; 3.1e6; 69 s | 0.89; 65%; 9.4e6; 196 s | 0.08; 0%; 9.8e5; 32 s |
| BlackJAX NSS | H100 | 0.65; 10%; 1.9e6; 12 s | 0.81; 10%; 5.9e6; 17 s | 1.37; 57%; 6.0e6; 17 s | 0.58; 60%; 1.8e7; 33 s | 0.87; 12%; 1.9e6; 12 s |
| JAXNS | H100 | 0.82; 5%; 7.3e6; 55 s | 0.97; 30%; 2.3e7; 98 s | 1.39; 72%; 2.4e7; 99 s | -; 98%; 7.2e7; 196 s | 1.02; 25%; 7.4e6; 54 s |
| dynesty | 4 cores | 0.87; 0%; 5.9e5; 91 s | 0.59; 2%; 1.3e6; 204 s | 1.17; 52%; 1.3e6; 200 s | 0.53; 40%; 1.1e7; 1372 s | 0.93; 10%; 5.9e5; 91 s |
| UltraNest | 4 cores | 0.07; 0%; 1.4e7; 1837 s; 5 of 40 timed out | 0.80; 5%; 5.0e6; 738 s | 0.87; 62%; 4.8e6; 724 s | 0.93; 70%; 1.4e7; 2195 s | timed out (>14405 s) |
| Nautilus | 4 cores | 0.03; 0%; 1.4e5; 961 s | 0.08; 0%; 2.6e5; 3481 s | -; 100%; 2.4e5; 1993 s | -; 100%; 4.8e5; 5671 s | 0.02; 0%; 1.5e5; 1507 s |

(*) tinyns only, with more live points than any competitor ran: not a
comparison.

- **Accuracy.** tinyns v1 is accurate on 18 of the 24 targets at nlive 500 and
  on 22 at 1000, and its error bars are honest (sd / logzerr from 0.69 to
  1.32, with the earlier `logzerr`). At 500 it is low by 0.22 on `rosen_d10`
  and loses the minor mode of the 32-D mixtures in 25 to 30% of the runs.
- **Time.** A compiled run takes 1 to 20 s and a first run 9 to 31 s, against
  9 to 77 s for BlackJAX NSS and 16 to 360 s for JAXNS. About 9 s of each
  BlackJAX run is a fixed cost, which on CPU is the first-call compilation of
  its adapter's eager ops (fixed since; the H100 split is not measured). Past
  it, BlackJAX's time grows 4.2 times as fast with the calls as tinyns's.
- **Calls.** tinyns does not need the fewest. Nautilus needs 1.7 to 24 times
  fewer for an rms 2 to 28 times smaller, and dynesty and UltraNest need 1.5
  to 13 times fewer below d = 10 on unimodal targets: with an expensive
  likelihood they are the better choice where they are accurate. Nautilus
  timed out at d = 64 and lost the minor mode on `sepM` and at d = 32.
- **Mode weights.** On the two-mode `sepW` and `connW` up to d = 18, tinyns's
  weights scatter by 0.08 to 0.47 at nlive 500 and 0.06 to 0.08 at 1000, with
  one run in 480 losing the mode. At d = 10 and 18 BlackJAX, JAXNS and dynesty
  scatter by 0.4 to 1.2 at 500 and lose the mode in up to 30% of the runs.
  Nautilus and UltraNest's MLFriends are as tight as tinyns; MLFriends timed
  out on `mix3_d10`.
- **`sepM`.** At nlive 500 and 1000 tinyns is accurate on 1 of the 6 `sepM`
  cells (`sepM_d10` at 1000), and no competitor is accurate on any at these
  settings. tinyns recovers the mode once it holds about `5 * ndim` live
  points: `sepM_d18` works at nlive 2000, for 4 times the calls of nlive 500
  (1.5e7) and 0.8 s more time, and `sepM_d32` at 4000 (it still loses the
  mode in 28% of the runs at 2000), for 8 times the calls (9.3e7) and 3.8 s
  more. At nlive 4000 all 12 mixtures are accurate.
- **Known gaps.** Deferred reruns would favour the competitors: Nautilus with
  a pool and a larger `n_live` (at 8000 it resolves `sepM_d10`), UltraNest's
  slice sampler at d = 10 (it finishes where MLFriends timed out), chain
  lengths matched to tinyns's at d >= 30 (dynesty's `rwalk` is unbiased on
  `gauss_d32` with 192 steps), competitor rows at nlive 1000 and above, and
  the H100 cells again with the adapters fixed after the sweep (BlackJAX's
  times hold a fixed cost of about 9 s, which on CPU is the compilation of
  its eager ops, and its stop ran past `dlogz`, +19% calls on `gauss_d2`; the
  CPU samplers' walls hold a compile per batch size). JAXNS and UltraNest
  stop later than `dlogz = 0.1` in low dimensions (JAXNS: about 1.5 times the
  calls at d = 2). `bench/RESULTS.md` states each correction.

## References and license

tinyns builds on nested sampling (Skilling 2006), the evidence bookkeeping
and insertion-rank test of Fowlie, Handley & Su, and the split-ensemble idea
of emcee (Foreman-Mackey et al. 2013). `CHANGELOG.md` lists the v1 changes and
what was removed from v0.x. MIT license (`LICENSE`).
