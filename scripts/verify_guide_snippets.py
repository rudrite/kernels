# Verifies every code snippet destined for the chapter 06 Pallas guide, in
# interpret mode. A snippet that does not pass here does not go on the site.
#
#   uv run --with 'jax==0.4.38' python scripts/verify_guide_snippets.py
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

print("jax", jax.__version__)
MS = getattr(pltpu, "MemorySpace", None) or getattr(pltpu, "TPUMemorySpace")


# ---- snippet: scratch accumulator (f32 accumulation for a bf16 output) ----
def matmul_scratch_kernel(a_ref, b_ref, o_ref, acc_ref):
    k = pl.program_id(2)
    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)
    @pl.when(k == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)

def matmul_scratch(a, b, bm=128, bn=128, bk=128):
    m, k = a.shape
    _, n = b.shape
    return pl.pallas_call(
        matmul_scratch_kernel,
        grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
                  pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        interpret=True,
    )(a, b)

a = jax.random.normal(jax.random.key(0), (256, 256), jnp.float32)
b = jax.random.normal(jax.random.key(1), (256, 256), jnp.float32)
err = float(jnp.abs(matmul_scratch(a, b) - a @ b).max())
print("scratch accumulator:", err)
assert err < 1e-3

# ---- snippet: scalar prefetch (a data-driven index map) ----
def permute_kernel(perm_ref, x_ref, o_ref):
    o_ref[...] = x_ref[...]

def permute_blocks(x, perm, bm=64):
    m, d = x.shape
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(m // bm,),
        in_specs=[pl.BlockSpec((bm, d), lambda i, perm: (perm[i], 0))],
        out_specs=pl.BlockSpec((bm, d), lambda i, perm: (i, 0)),
    )
    return pl.pallas_call(
        permute_kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True,
    )(perm, x)

x = jax.random.normal(jax.random.key(0), (256, 128), jnp.float32)
perm = jnp.array([3, 0, 2, 1], jnp.int32)
got = permute_blocks(x, perm)
want = jnp.concatenate([x[192:256], x[0:64], x[128:192], x[64:128]])
err = float(jnp.abs(got - want).max())
print("scalar prefetch:", err)
assert err == 0.0

# ---- snippet: input_output_aliases (update a buffer in place) ----
def scale_rows_kernel(x_ref, s_ref, o_ref):
    o_ref[...] = x_ref[...] * s_ref[0]

def scale_rows_inplace(x, s, rows=64):
    n, d = x.shape
    return pl.pallas_call(
        scale_rows_kernel,
        grid=(n // rows,),
        in_specs=[pl.BlockSpec((rows, d), lambda i: (i, 0)),
                  pl.BlockSpec(memory_space=MS.SMEM)],
        out_specs=pl.BlockSpec((rows, d), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        input_output_aliases={0: 0},
        interpret=True,
    )(x, s)

x = jnp.ones((256, 128), jnp.float32)
got = scale_rows_inplace(x, jnp.array([3.0], jnp.float32))
err = float(jnp.abs(got - 3.0).max())
print("input_output_aliases:", err)
assert err == 0.0

# ---- snippet: the edge-block mask (array not divisible by the block) ----
def sum_ragged_kernel(x_ref, o_ref, *, n_valid, rows):
    i = pl.program_id(0)
    x = x_ref[...]
    idx = i * rows + jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    x = jnp.where(idx < n_valid, x, 0.0)
    @pl.when(i == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)
    o_ref[...] += jnp.sum(x, axis=0, keepdims=True)

def sum_valid_rows(x, n_valid, rows=64):
    n, d = x.shape
    return pl.pallas_call(
        functools.partial(sum_ragged_kernel, n_valid=n_valid, rows=rows),
        grid=(n // rows,),
        in_specs=[pl.BlockSpec((rows, d), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, d), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((1, d), x.dtype),
        interpret=True,
    )(x)

x = jnp.ones((256, 128), jnp.float32)
got = sum_valid_rows(x, n_valid=200)
err = float(jnp.abs(got - 200.0).max())
print("edge mask:", err)
assert err == 0.0

# ---- snippet: manual DMA (ANY-space input, explicit copy, semaphore) ----
def dma_kernel(x_hbm_ref, o_ref, scratch_ref, sem):
    copy = pltpu.make_async_copy(x_hbm_ref, scratch_ref, sem)
    copy.start()
    copy.wait()
    o_ref[...] = scratch_ref[...] * 2

def double_via_dma(x):
    return pl.pallas_call(
        dma_kernel,
        in_specs=[pl.BlockSpec(memory_space=MS.ANY)],
        out_specs=pl.BlockSpec(x.shape, lambda: (0, 0)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        scratch_shapes=[pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA],
        interpret=True,
    )(x)

x = jax.random.normal(jax.random.key(0), (128, 128), jnp.float32)
err = float(jnp.abs(double_via_dma(x) - x * 2).max())
print("manual dma:", err)
assert err == 0.0

print("ALL GUIDE SNIPPETS VERIFIED under jax", jax.__version__)
