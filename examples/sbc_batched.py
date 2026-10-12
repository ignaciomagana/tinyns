"""A mock-data campaign in one call: simulation-based calibration (SBC).

Model: ``y_i ~ Normal(mu, sigma)`` for 30 observations, with uniform priors
on ``mu`` and ``log sigma``. For each of ``B`` lanes we draw the parameters
from the prior, simulate a dataset, and sample its posterior. The data enter
the likelihood as a pytree callable, ``jax.tree_util.Partial(loglike, data)``,
with one dataset per lane, and ``run(keys, batched_data=True)`` runs all the
lanes in one compiled program. If the posteriors are calibrated, the true
value falls inside the central 50% (90%) credible interval in about 50% (90%)
of the lanes.
"""

import jax
import jax.numpy as jnp

from tinyns import NestedSampler

B = 64  # datasets
NOBS = 30
LOW = jnp.array([-5.0, -1.0])  # mu, log sigma
HIGH = jnp.array([5.0, 1.0])


def prior_transform(u):
    """Map the unit square to the uniform prior on (mu, log sigma)."""
    return LOW + (HIGH - LOW) * u


def loglike(data, theta):
    """Gaussian log likelihood of one dataset."""
    mu, log_sigma = theta
    z = (data - mu) * jnp.exp(-log_sigma)
    return -0.5 * jnp.sum(z**2) - data.size * (log_sigma + 0.5 * jnp.log(2 * jnp.pi))


def main():
    key_truth, key_noise, key_run, key_draw = jax.random.split(jax.random.key(0), 4)
    truth = prior_transform(jax.random.uniform(key_truth, (B, 2)))
    noise = jax.random.normal(key_noise, (B, NOBS))
    data = truth[:, :1] + jnp.exp(truth[:, 1:]) * noise  # (B, NOBS)

    sampler = NestedSampler(
        jax.tree_util.Partial(loglike, data), prior_transform, ndim=2, nlive=400
    )
    results = sampler.run(jax.random.split(key_run, B), batched_data=True)

    # Posterior quantile of the truth in every lane (uniform if calibrated).
    quantiles = []
    for i, result in enumerate(results):
        draws = result.resample_equal(jax.random.fold_in(key_draw, i), n=1000)
        quantiles.append(jnp.mean(draws < truth[i], axis=0))
    quantiles = jnp.stack(quantiles)  # (B, 2)

    print(f"{B} runs, {sum(r.ncall for r in results)} likelihood calls in all")
    print(f"compile {results[0].metadata['compile_s']:.1f} s, "
          f"sampling {results[0].metadata['sampling_s']:.1f} s")
    for level in (0.5, 0.9):
        inside = jnp.abs(quantiles - 0.5) < level / 2
        cover = jnp.mean(inside, axis=0)
        print(f"truth inside the {level:.0%} interval: "
              f"mu {cover[0]:.2f}, log sigma {cover[1]:.2f}")
    return results, quantiles


if __name__ == "__main__":
    main()
