"""The specimen: one attention block, and the exact arrays it runs on.

Sixteen rows, thirty-two channels, no batch axis. Three projections, one
scaled dot product, one softmax, one more dot product. That is the smallest
program that still contains everything the site teaches at the compiler
floor: matmuls that get their operands transposed for free, a constant that
folds, and a softmax that arrives as nine primitives instead of one.

Nothing else in this directory decides anything about the program. The
capture script imports this module and writes down what the machine makes of
it, level by level.
"""

import jax
import jax.numpy as jnp

ROWS = 16
CHANNELS = 32
SEED = 0


def block(x, wq, wk, wv):
    q, k, v = x @ wq, x @ wk, x @ wv
    s = (q @ k.T) / jnp.sqrt(jnp.float32(q.shape[-1]))
    return jax.nn.softmax(s, axis=-1) @ v


def inputs():
    """The four arrays the specimen is captured on, from one fixed seed.

    Values do not change any artifact below the source: the trace and every
    level under it depend on shapes and dtypes only. They are seeded anyway
    so that running the block for its numbers gives the same numbers twice.
    """
    kx, kq, kk, kv = jax.random.split(jax.random.PRNGKey(SEED), 4)
    x = jax.random.normal(kx, (ROWS, CHANNELS), jnp.float32)
    wq = jax.random.normal(kq, (CHANNELS, CHANNELS), jnp.float32)
    wk = jax.random.normal(kk, (CHANNELS, CHANNELS), jnp.float32)
    wv = jax.random.normal(kv, (CHANNELS, CHANNELS), jnp.float32)
    return x, wq, wk, wv
