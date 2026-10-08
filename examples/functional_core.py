"""Drive the sampler by hand with the functional core.

``NestedSampler.run`` is a driver around four pieces: a static ``Config``,
``init`` (the first live set), ``step`` (delete the ``num_delete`` lowest live
points and replace them) and ``finalise`` (weights, evidence and the result
object). ``State`` and the ``Dead`` rows of a step are pytrees of arrays, so
``step`` composes with ``jax.jit``, ``lax.scan`` and ``jax.vmap``. Use the
pieces when you need your own loop: a custom stopping rule, your own
checkpointing, or nested sampling inside a larger JAX program.
"""

import math

import jax
import jax.numpy as jnp

from tinyns import Config, finalise, init, step
from tinyns.core import CONVERGED, delta_logz

NDIM = 3
DLOGZ = 0.1
STEPS_PER_CHUNK = 20
TRUE_LOGZ = -NDIM * math.log(20.0)


def prior_transform(u):
    """Map the unit cube to the uniform prior on [-10, 10]^3."""
    return -10.0 + 20.0 * u


def loglike(theta):
    """Normalized standard normal log likelihood."""
    return -0.5 * jnp.sum(theta**2) - 0.5 * NDIM * jnp.log(2 * jnp.pi)


def main():
    cfg = Config(ndim=NDIM, nlive=500, num_delete=50)

    @jax.jit
    def chunk(state):
        """Run a fixed number of steps on the device; return their dead rows."""

        def body(state, _):
            return step(state, loglike, prior_transform, cfg)

        return jax.lax.scan(body, state, None, length=STEPS_PER_CHUNK)

    state = init(jax.random.key(0), loglike, prior_transform, cfg)
    chunks = []
    # delta_logz is the evidence the live points could still add, in log
    # units. A likelihood plateau would need a check of state.status too.
    while float(delta_logz(state, cfg)) >= DLOGZ:
        state, dead = chunk(state)  # leaves of shape (STEPS_PER_CHUNK, k, ...)
        chunks.append(dead)
        print(
            f"steps {int(state.it):4d}  running logz {float(state.logz):8.4f}  "
            f"remaining {float(delta_logz(state, cfg)):.3g}"
        )

    dead = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *chunks)
    # The status is the driver's business: step never sets "converged".
    state = state._replace(status=jnp.asarray(CONVERGED, jnp.int32))
    result = finalise(state, dead, cfg, prior_transform=prior_transform)

    print(result.summary())
    print(f"true logZ: {TRUE_LOGZ:.4f}")
    return result


if __name__ == "__main__":
    main()
