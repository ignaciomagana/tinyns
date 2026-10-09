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
`unresolved: raise nlive`. The mode tracking is a fixed cost per step: about 6%
of the run time at 1 ms per likelihood call. For likelihoods that take
microseconds it is a larger share (15 to 21% more per call on a 10-D two-mode
target than on a 10-D Gaussian; see Speed).

`result.modes()` counts the modes from the clusters the run recorded, and
gives each mode's mass, the seed-to-seed scatter to expect (`urn_sd`) and two
flags.

- `unresolved`: the mode held fewer than `5 * ndim` live points.
- `tracked`: `False` if the sampler's clustering never separated the mode from
  its neighbours. The hop then did not balance it, its mass scatters by about
  `urn_sd`, and the number of such modes is a lower bound. An eggbox is the
  example: its 18 modes sit on a lattice that no two-way split separates, and
  `modes()` finds 15 to 18 of them from the gaps between their live points
  (nlive 1000), none tracked.

`diagnostics()` and `summary()` turn both flags into warnings, and warn when
the clustering's 8 slots were full during a run with several modes.

## Choosing `walks`

The default, `max(25, 6 * ndim, ndim**2 // 6)`, keeps the logZ bias below its
scatter on correlated Gaussians from 2 to 64 dimensions. It was also enough
for a curved target once there were enough live points: with the default
`walks` (60) and `num_delete = nlive // 10`, the logZ of a 10-D Rosenbrock was
0.16 +- 0.05 low at `nlive=500` and within 0.05 +- 0.04 from 1000, with a
scatter of 1.0 to 1.1 times `logzerr` (40 runs each). If in doubt, rerun with
twice the walks and check that logZ moves by less than `logzerr`.

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
  call. It evaluates no out-of-cube proposal, which saved 8 to 17% of the
  calls, and its `logzerr` is 2% smaller. If the device can evaluate several
  likelihoods at once, the default is faster by the batch size.

**Memory** was no constraint: 40 batched runs at 64 dimensions peaked at 4 GiB
of GPU memory with `nlive=4000` (7 GiB at a share of 0.5), and one run holds
`niter * ndim` samples on the host (740 thousand dead points at 64-D,
`nlive=4000`).

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
- **Many modes.** The clustering has 8 slots and splits two ways, so it does
  not separate more modes than that, or modes on a lattice (an eggbox). The
  evidence is still right, but the hop does not balance those modes: their
  weights scatter from seed to seed (logit sd 0.4 for the eggbox's 8% modes
  at nlive 1000, 0.6 for its 4% modes) and a 2% mode was lost (below a tenth of its
  weight) in 15% of the runs. `modes()` marks such modes `tracked: False`, and counts only the ones
  that planes across the axes of the unit cube separate.
- **Curved targets.** A 10-D Rosenbrock is biased low below `nlive=1000` (see
  Choosing `walks`).
- **Expensive likelihoods.** Parallel chains do not reduce the number of
  likelihood calls, and with `num_delete > 1` out-of-cube proposals are
  evaluated too. When one call already fills the device, use `num_delete=1`.
- **float32.** Likelihood differences below about `1e-7 * |loglike|` are lost;
  enable x64.
- **Insertion test.** Its p-values are honest up to `num_delete = nlive // 10`
  and anti-conservative above (see Choosing `nlive` and `num_delete`).
- **Compile cache.** A process keeps up to 8 compiled configurations.

## Benchmarks

tinyns v1 was run against tinyns 0.x, BlackJAX NSS, JAXNS, dynesty, UltraNest
and Nautilus on 24 targets with a known evidence: 20 seeds per target, 40 on
the mixtures. `bench/RESULTS.md` has every cell, the settings and the
validation gate; `bench/README.md` describes the harness. The competitors ran
at nlive 500 or their own defaults (JAXNS with 500 chains, Nautilus with its
2000 live points, UltraNest with MLFriends up to d = 10 and its slice sampler
above). The times are not like for like: tinyns, BlackJAX and JAXNS ran on one
H100, the others on 4 CPU cores with the likelihood called from Python.

Evidence. Each cell is the logZ bias ± its standard error; likelihood calls;
time (compiled for tinyns and BlackJAX, with cached programs for JAXNS):

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

| sampler | on | `sepW_d10` | `sepW_d18` | `sepM_d18` | `mix3_d10` |
|---|---|---|---|---|---|
| **tinyns v1**, nlive 500 | H100 | 0.14; 0%; 1.2e6; 1.9 s | 0.40; 0%; 3.8e6; 3.5 s | 1.08; 68%; 3.8e6; 3.3 s | 0.06; 0%; 1.2e6; 2.0 s |
| **tinyns v1**, nlive 1000 (default) | H100 | 0.06; 0%; 2.4e6; 2.0 s | 0.06; 0%; 7.5e6; 3.6 s | 0.40; 20%; 7.5e6; 4.0 s | 0.05; 0%; 2.4e6; 2.1 s |
| **tinyns v1**, nlive 2000 (*) | H100 | 0.03; 0%; 4.8e6; 2.1 s | 0.04; 0%; 1.5e7; 3.9 s | 0.03; 0%; 1.5e7; 4.1 s | not run |
| tinyns 0.x | H100 | 0.13; 0%; 9.8e5; 33 s | 0.27; 0%; 3.1e6; 80 s | not run | not run |
| BlackJAX NSS | H100 | 0.65; 10%; 1.9e6; 12 s | 0.81; 10%; 5.9e6; 17 s | 1.37; 57%; 6.0e6; 17 s | 0.87; 12%; 1.9e6; 12 s |
| JAXNS | H100 | 0.82; 5%; 7.3e6; 55 s | 0.97; 30%; 2.3e7; 98 s | not run | not run |
| dynesty | 4 cores | 0.87; 0%; 5.9e5; 91 s | 0.59; 2%; 1.3e6; 204 s | 1.17; 52%; 1.3e6; 200 s | 0.93; 10%; 5.9e5; 91 s |
| UltraNest | 4 cores | 0.07; 0%; 1.4e7; 1837 s; 5 of 40 timed out | 0.80; 5%; 5.0e6; 738 s | 0.87; 62%; 4.8e6; 724 s | timed out (>14405 s) |
| Nautilus | 4 cores | 0.03; 0%; 1.4e5; 961 s | 0.08; 0%; 2.6e5; 3481 s | -; 100%; 2.4e5; 1993 s | 0.02; 0%; 1.5e5; 1507 s |

(*) More live points than any competitor ran: not a comparison.

- **Accuracy.** tinyns v1 is accurate on 18 of the 24 targets at nlive 500 and
  on 22 at 1000, and its error bars are honest (sd / logzerr from 0.69 to
  1.32). At 500 it is low by 0.22 on `rosen_d10` and loses the minor mode of
  the 32-D mixtures in 25 to 30% of the runs.
- **Time.** A compiled run takes 1 to 20 s and a first run 9 to 31 s, against
  9 to 77 s compiled for BlackJAX NSS and 16 to 360 s for JAXNS.
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
  points: `sepM_d18` works at nlive 2000 (25 seeds), for 4 times the calls of
  nlive 500 (1.5e7) and 0.8 s more time. `sepM_d32` fails at 1000 and has not
  run above.
- **Known gaps.** Deferred reruns would favour the competitors: Nautilus with
  a pool and a larger `n_live` (at 8000 it resolves `sepM_d10`), UltraNest's
  slice sampler at d = 10 (it finishes where MLFriends timed out), chain
  lengths matched to tinyns's at d >= 30 (dynesty's `rwalk` is unbiased on
  `gauss_d32` with 192 steps), and competitor rows at nlive 1000 and above.
  tinyns 0.x and JAXNS have not run `connW`, `sepM` and `mix3`; a GPU outage
  cut tinyns's nlive 2000 rows short and its 4000 rows did not run.

## References and license

tinyns builds on nested sampling (Skilling 2006), the evidence bookkeeping
and insertion-rank test of Fowlie, Handley & Su, and the split-ensemble idea
of emcee (Foreman-Mackey et al. 2013). `CHANGELOG.md` lists the v1 changes and
what was removed from v0.x. MIT license (`LICENSE`).
