import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tinyns import NestedSampler  # noqa: E402


def prior_transform(u):
    # Replace with your own prior transform.
    return -5.0 + 10.0 * u


def loglike(theta):
    # Replace this placeholder with your expensive JAX likelihood.
    # Keep it JAX-compatible if using kernel="jax".
    return -0.5 * jnp.sum(theta**2) - 0.5 * theta.shape[0] * jnp.log(2.0 * jnp.pi)


def main():
    ndim = 2
    key = jax.random.PRNGKey(0)

    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim,
        nlive=500,
        sample="rwalk",
        kernel="jax",
        walks=5,
        replacement_chains=1,
        rwalk_proposal="isotropic",
        jax_block_size=32,
    )

    start = time.perf_counter()
    result = sampler.run(key, dlogz=0.1)
    seconds = time.perf_counter() - start

    print(result.summary())
    print(result.diagnostics())
    print(f"seconds={seconds:.3f}")


if __name__ == "__main__":
    main()
