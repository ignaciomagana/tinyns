# tinyns

`tinyns` is a tiny, general-purpose nested sampler written entirely in JAX, in
the spirit of `emcee` and `tinygp`. Give it a log likelihood, a prior transform
and the number of dimensions; it returns the evidence and weighted posterior
samples. The whole run executes on the device (GPU or CPU), it handles
multimodal posteriors inside the sampler, and it is not tied to any application.

## Install

`tinyns` needs Python 3.10+ and JAX 0.4.31+. It is not on PyPI yet:

```bash
python -m pip install git+https://github.com/ignaciomagana/tinyns.git
```

For a GPU, install the matching JAX build first (see the JAX install guide).
For development: clone, `python -m pip install -e '.[dev]'`, then `make test`.

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
(tinyns vmaps them); a NaN likelihood counts as `-inf`. The whole API is
`NestedSampler(loglike, prior_transform, ndim, nlive, *, num_delete, walks)`
and `sampler.run(key, *, dlogz=0.1, maxiter, maxcall, progress, checkpoint,
batched_data)`. `key` is a PRNG key or an int seed, here and wherever tinyns
takes a key. The run stops when the live points hold less than `dlogz` of the
evidence, at `maxiter` dead points or `maxcall` likelihood calls, or on a
likelihood plateau. `progress=True` prints one line per chunk of steps.

## The result

```python
from tinyns import NestedSamplingResult

print(result.logz, result.logzerr)  # log evidence and its error
samples, weights = result.samples, result.weights()  # weighted posterior
equal = result.resample_equal(1, n=2000)  # equal weights; key or int seed
print(result.modes())  # posterior modes: mass, smallest live count, flags
print(result.insertion_test()["pvalue"])  # insertion-rank uniformity test
print(result.diagnostics()["warnings"])  # the checks, as a list of warnings
result.save_npz("result.npz")
result = NestedSamplingResult.load_npz("result.npz")
```

The arrays (`samples`, `samples_u`, `logl`, `logwt`, `logl_birth`, `nlive_i`,
`labels`) hold the dead points in order, then the final live points.
`metadata` has the settings and timings; `to_dict()` and `to_numpy()` return
every field, and `to_dynesty_dict()` gives dynesty-style keys.

## How it works

Each step deletes the `k = num_delete` lowest live points and replaces them
with the end points of `k` Metropolis chains of fixed length `walks`, run in
parallel (vmapped). Each chain starts from a live point above the new
likelihood contour `L*`; proposals that leave the unit cube or fall to `L*` or
below are rejected, and a chain that never moves returns its seed.

- **Local random walk.** The live points are clustered on the device every
  quarter e-fold of prior volume. A walk step is `x + s L z`, with `L` the
  Cholesky factor of the covariance of the cluster of `x`, and the exact
  Hastings correction when the proposal lands in another cluster.
- **Hop.** Every 10th chain step is an independence Metropolis-Hastings
  proposal from the union of ellipsoids fitted to the clusters. It moves
  points between modes and keeps the mode weights right.
- **Cross-fitting.** The live slots form three folds. A chain seeded in one
  fold uses clusters and covariances fitted to the other two only (as emcee
  moves each half of its walkers with the other half).
- **Evidence.** The `k` deaths of a step are counted at live counts
  `m, ..., m-k+1` (birth/death contours; Fowlie, Handley & Su); `logz` and
  `logzerr` are recomputed on the host in float64.
- **Driver.** A compiled `while_loop` runs chunks of steps on the device; the
  host syncs once per chunk (progress, checkpoints, stopping).

**Exactness.** A chain's kernel is built without its own fold, so it does not
depend on the chain's seed or its ancestors, and both moves leave the uniform
law on `{L > L*}` invariant. The end point therefore has that law for any
chain length. What remains is common to every MCMC-driven nested sampler: a
new point is correlated with its seed (hence `walks`), and the step scale
adapts on past chains.

**Speed.** Measured on one NVIDIA H100 in float32, on a 13-D correlated
Gaussian run to `dlogz = 0.1` with the default settings of each version:

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
dimensions took 0.91 s compiled against 0.66 s for a 10-D Gaussian; with
clustering and hops active, the time per likelihood call was 15 to 20% higher.

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
`unresolved: raise nlive`. The mode tracking added about 6% to the run time at
1 ms per likelihood call, and adds nothing per chain step while a run has one
cluster.

## Choosing `walks`

The default, `max(25, 6 * ndim, ndim**2 // 6)`, keeps the logZ bias below its
scatter on correlated Gaussians from 2 to 64 dimensions. Strongly curved
targets need longer chains: a 10-D Rosenbrock valley needs 12 to 25 `* ndim`.
If in doubt, rerun with twice the walks and check that logZ moves by less than
`logzerr`.

## Choosing nlive / num_delete

<!-- PLACEHOLDER: the defaults-sweep PR replaces this section. -->
The current defaults are `nlive=1000` and `num_delete = nlive // 10` (at most
`nlive // 2`). Guidance from the defaults sweep will go here. Until then:
`logzerr` falls as `1 / sqrt(nlive)`, every mode needs `5 * ndim` live points,
and `num_delete=1` suits expensive likelihoods (see Limitations).

## GPU notes

- **Precision.** tinyns runs in JAX's default precision, float32. Set
  `JAX_ENABLE_X64=1` when `|loglike|` is large (float32 resolves about 7
  digits) or the posterior is very narrow in the unit cube.
- **Compile cache.** `jax.config.update("jax_compilation_cache_dir", path)`
  lets separate processes reuse compiled programs.
- **Memory.** A step evaluates `num_delete` likelihoods at once (times the
  batch size of a batched run). If that does not fit, lower `num_delete`.

## Checkpoints

```python
result = sampler.run(jax.random.key(0), checkpoint="run.npz")
```

The run writes `run.npz` atomically at most every 10 minutes and at the end.
If the file exists, the run resumes from it, and the result is bit-identical
to an uninterrupted run (tested on CPU). A checkpoint of a different
configuration, key or precision is refused. A run stopped by `maxiter` or
`maxcall` continues when called again with a larger limit.

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
simulation-based calibration and mock campaigns. A batched run agrees with the
run of its key alone in distribution, not bit for bit.

## Pytree callables

```python
for one in data[:2]:  # same shapes: compiled once
    like = Partial(loglike_data, one)
    print(NestedSampler(like, prior_transform, ndim=1, nlive=200).run(0).logz)
```

Pass data as `jax.tree_util.Partial(fn, data)` or any pytree callable. Its
arrays become arguments of the compiled program, so datasets of one shape
share one compilation.

## The functional core

```python
from tinyns import Config, delta_logz, finalise, init, step

cfg = Config(ndim=2, nlive=200)
state = init(0, loglike, prior_transform, cfg)
rows = []
while delta_logz(state, cfg) >= 0.1:  # evidence left in the live points
    state, dead = step(state, loglike, prior_transform, cfg)
    rows.append(dead)
dead = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)
result = finalise(state, dead, cfg, prior_transform=prior_transform, dlogz=0.1)
```

`State` and `Dead` are pytrees of arrays, so `step` works under `jax.jit`,
`lax.scan` and `jax.vmap`. `step` does not test termination; the loop does.
Given the loop's `dlogz`, `finalise` applies the stopping rule of `run` and
reports `converged` (without it: `success=False`, `running`).
`tinyns.STATUS[int(state.status)]` names the status, which `step` changes only
on a likelihood plateau (`"plateau"`: stop there). `state.ncall` is an int32:
a loop that may pass `2**31` likelihood calls counts them on the host and
gives `finalise(..., ncall=total)`, as `examples/functional_core.py` does.

`examples/` has five scripts, each under a minute on a CPU: `quickstart.py`,
`multimodal.py`, `checkpoint_resume.py`, `sbc_batched.py` and
`functional_core.py`. The test suite runs them and the snippets above.

## Limitations

- **Small modes.** A mode with fewer than about `5 * ndim` live points is not
  sampled reliably. `modes()` flags it; raise `nlive`.
- **Many modes.** The clustering has 8 slots. Mode weights are not validated
  for more separated modes than that (an eggbox).
- **Curved targets** need more `walks` than the default.
- **Expensive likelihoods.** Parallel chains do not reduce the number of
  likelihood calls, and with `num_delete > 1` out-of-cube proposals are
  evaluated too. When one call already fills the device, use `num_delete=1`.
- **float32.** Likelihood differences below about `1e-7 * |loglike|` are lost;
  enable x64.
- **Insertion test.** With `num_delete > 1` its p-values are mildly
  anti-conservative.
- **Compile cache.** A process keeps up to 8 compiled configurations.

## Benchmarks

`bench/README.md` describes the harness that runs tinyns, dynesty, UltraNest,
Nautilus, BlackJAX and JAXNS on targets with known evidence.

<!-- PLACEHOLDER: head-to-head table, to be filled at release. -->
The head-to-head table will be added here at release.

## References and license

tinyns builds on nested sampling (Skilling 2006), the evidence bookkeeping
and insertion-rank test of Fowlie, Handley & Su, and the split-ensemble idea
of emcee (Foreman-Mackey et al. 2013). `CHANGELOG.md` lists the v1 changes and
what was removed from v0.x. MIT license (`LICENSE`).
