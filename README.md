# tinyns

`tinyns` is a tiny nested sampler written entirely in JAX, in the spirit of
`emcee` and `tinygp`: give it a log likelihood, a prior transform and the
number of dimensions, and it returns the evidence and weighted posterior
samples. Every step runs on the device; the host syncs once per chunk of steps.

This is the v1 development branch: the API below is a hard break from v0.x
(no compatibility shims, no old-checkpoint readers). Checkpoints, batched
keys and multimodal moves are being added in the next pull requests.

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
              walks=None)       # None -> max(25, 6 * ndim)
sampler.run(key, *, dlogz=0.1, maxiter=None, maxcall=None, progress=False)
```

Any other keyword raises `TypeError`. `key` is a PRNG key or an int seed. The
run stops when the live points hold less than `dlogz` of the evidence, before
more than `maxiter` dead points, once `maxcall` likelihood calls are reached,
or on a likelihood plateau (no live point above the deleted ones).

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

The `k` deaths of a step are counted at live counts `m, m-1, ..., m-k+1`
(Fowlie, Handley & Su), so the expected log prior volume after `it` steps is
`-it * sum_j 1/(m - j)`; the final live points die at counts `m, ..., 1`. The
weights, `logz` and `logzerr = sqrt(sum_i dH_i / n_i)` are recomputed on the
host in float64. `result.nlive_i` holds the live count at every death.

## The result

`NestedSamplingResult` holds `samples`, `samples_u`, `logl`, `logwt`,
`logl_birth`, `nlive_i`, `logz`, `logzerr`, `ncall`, `niter`, `nlive`,
`num_delete` and `metadata`. Methods: `summary()`, `diagnostics()`,
`insertion_test()` (ranks of the new points among the `nlive - num_delete`
survivors of their step), `modes()`, `logz_bootstrap()`, `resample_equal()`,
`save_npz()`/`load_npz()`, `to_numpy()` and `to_dynesty_dict()`.

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
