# tinyns

`tinyns` is a tiny nested sampler written entirely in JAX, in the spirit of
`emcee` and `tinygp`: give it a log likelihood, a prior transform and the
number of dimensions, and it returns the evidence and weighted posterior
samples. Every step runs on the device; the host syncs once per chunk of steps.

This is the v1 development branch: the API below is a hard break from v0.x
(no compatibility shims, no old-checkpoint readers).

## Install

```bash
git clone <repo-url>
cd tinyns
python -m pip install -e '.[dev]'
```

## Example

```python
import jax.numpy as jnp
from tinyns import NestedSampler


def prior_transform(u):
    return -10.0 + 20.0 * u


def loglike(theta):
    return -0.5 * jnp.sum(theta**2) - jnp.log(2 * jnp.pi)


result = NestedSampler(loglike, prior_transform, ndim=2).run(0)
print(result.summary())
```

`loglike` and `prior_transform` are JAX-traceable functions of one point. A
NaN likelihood counts as `-inf`.

## API

```python
NestedSampler(loglike, prior_transform, ndim, nlive=1000, *,
              num_delete=None,  # None -> max(1, nlive // 10), at most nlive // 2
              walks=None)       # None -> max(25, 6 * ndim, ndim**2 // 6)
sampler.run(key, *, dlogz=0.1, maxiter=None, maxcall=None, progress=False,
            checkpoint=None, batched_data=False)
```

Any other keyword raises `TypeError`. `key` is a PRNG key or an int seed. The
run stops when the live points hold less than `dlogz` of the evidence, before
more than `maxiter` dead points, once `maxcall` likelihood calls are reached,
or on a likelihood plateau (no live point above the deleted ones).
`progress=True` prints one line per chunk of steps.

The functional core is exported too: `Config`, `init(key, loglike,
prior_transform, cfg)`, `step(state, loglike, prior_transform, cfg) -> (state,
dead)` and `finalise(state, dead, cfg, prior_transform=...)`. `State` and
`Dead` are pytrees of arrays, so a loop over `step` can be jitted or vmapped.

## How it works

Each step deletes the `k = num_delete` lowest live points and replaces them
with the ends of `k` fixed-length constrained random walks (vmapped), seeded
from distinct live points above the new contour `L*`. A walk step is
`u + s L z`, with `L` the Cholesky factor of the live-point covariance; steps
that leave the unit cube or fall to `L*` or below are rejected, and a chain
that never moves returns its seed. The step multiplier `s` adapts toward 25%
acceptance inside every step. With `num_delete=1` the single chain is not
vmapped, so an out-of-cube proposal skips the likelihood: use it for expensive
likelihoods.

### Multimodal posteriors

A random-walk chain cannot cross between separated modes, so without help the
number of live points in each mode drifts from step to step (a Polya urn) and
the mode weights scatter from seed to seed. tinyns clusters the live points on
the device every quarter e-fold (`tinyns/modes.py`: hard EM, merges and Fisher
splits in 8 cluster slots) and makes every 10th chain step an inter-mode
*hop*: an independence Metropolis-Hastings step from the union of the
clusters' moment-matched ellipsoids. The other steps walk in the covariance of
the current point's cluster, with the exact Metropolis-Hastings correction for
the change of covariance between clusters, so a mode shaped unlike the
largest one still mixes. Both moves are exact for the constrained prior: the
live slots form three folds, each new point is seeded from its own fold, and
its chain's clusters and covariances are fitted to the other two folds only
(as emcee moves each half of its walkers with the other half). In the v1
bake-off the hop cut the seed-to-seed scatter of minor-mode weights about 7x
at the same number of likelihood calls. `result.modes()` reports each mode's
mass, an urn error bar and its smallest live count. A mode needs about
`5 * ndim` live points for its own walk to mix; below that its weight keeps a
bias of a few percent (see the CHANGELOG), and `modes()` flags it unresolved:
raise `nlive`.

### Choosing `walks`

The default is `walks = max(25, 6 * ndim, ndim**2 // 6)`
(`tinyns.core.default_walks`): 6 steps per dimension up to 36 dimensions, then
`ndim / 6` per dimension (384 steps at 48, 682 at 64). The proposal covariance
of a step is built from the live points above `L*` other than the chains'
seeds, so a chain's proposals do not depend on where it starts and its end
point is uniform inside the contour however short the walk. (With the seed
included, its pull on the covariance biased logZ upward by an amount that grew
with `ndim` and shrank with `nlive`.) What remains is mixing: early in a run,
while the prior box still cuts the likelihood contours, the likelihood rank
along a chain takes 2 to 3 `ndim` steps to decorrelate (about 9 steps later
on). Above about 48 dimensions `6 * ndim` steps are too few for that phase:
they left +0.36 nats at 64 dimensions. The default keeps the logZ bias below
its scatter on correlated Gaussians from 2 to 64 dimensions with 250 to 2000
live points.

Strongly curved targets need longer walks than their dimension suggests. A
10-D Rosenbrock valley needs 12 to 25 `* ndim`: at the default (60) logZ
scatters 2.3 times more than `logzerr` says. Neal's funnel in 10-D is fine
at the default. If in doubt, rerun with twice the walks and check that logZ
moves by less than `logzerr`.

The `k` deaths of a step are counted at live counts `m, m-1, ..., m-k+1`
(Fowlie, Handley & Su), so the expected log prior volume after `it` steps is
`-it * sum_j 1/(m - j)`; the final live points die at counts `m, ..., 1`. The
weights, `logz` and `logzerr = sqrt(sum_i dH_i / n_i)` are recomputed on the
host in float64. `result.nlive_i` holds the live count at every death.

## The result

`NestedSamplingResult` holds `samples`, `samples_u`, `logl`, `logwt`,
`logl_birth`, `nlive_i`, `labels` (each sample's cluster when it died),
`logz`, `logzerr`, `ncall`, `niter`, `nlive`,
`num_delete` and `metadata`. Methods: `summary()`, `diagnostics()`,
`insertion_test()` (ranks of the new points among the `nlive - num_delete`
survivors of their step), `modes()`, `logz_bootstrap()`, `resample_equal()`,
`save_npz()`/`load_npz()`, `to_numpy()` and `to_dynesty_dict()`.

## Checkpoints

`run(key, checkpoint="run.npz")` writes one `.npz` file atomically at most
every 10 minutes and at the end. If the file exists, the run resumes from it,
and the result is bit-identical to an uninterrupted run (each step splits the
PRNG key it carries, so the random stream depends on the step count only, not
on where the chunks end). A checkpoint written with a different config, key,
x64 flag or float dtype is refused. A run stopped by `maxiter` or `maxcall`
continues when called again with a larger limit; `metadata["wall_time_s"]`
adds up every session.

## Batches of runs

`run(jax.random.split(key, B))` runs `B` independent runs in one compiled
program and returns a list of `B` results; with `batched_data=True` the
arrays of a `Partial` likelihood carry a leading axis of `B`, one dataset per
run (simulation-based calibration, mock campaigns). The step is vmapped over
the runs inside the loop, and a finished run waits, frozen, for the others.
Each run is a draw from the same distribution as its separate run but is not
bit-identical to it, since vmap changes XLA's summation order: in float64 the
two agree to roundoff, in float32 they can part late in the run.

## Data in the likelihood

Pass data as a pytree callable, `jax.tree_util.Partial(loglike_fn, data)`:
its arrays become arguments of the compiled kernels, and datasets of one
shape share one compiled program, which suits mock-data campaigns. A closure's
captured arrays of at least 4096 elements are hoisted to arguments as well.
Use `jax.config.update("jax_compilation_cache_dir", path)` to share compiles
across processes.

## float32 and float64

tinyns runs in the default JAX precision. Enable `JAX_ENABLE_X64` when the
log likelihood is large in magnitude (float32 resolves about 7 digits) or
the posterior is very narrow in the unit cube.
