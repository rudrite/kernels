// CodeWalk data for the row-softmax kernel. Code copied verbatim from
// labs/stage-1/lab-1.3-fused-softmax-layernorm.ipynb (softmax_kernel and
// its pallas_call wrapper, minus the notebook's own check() call below it).
export const walk = {
  id: 'softmax',
  title: "row softmax: one kernel, one grid axis",
  lang: 'python' as const,
  code: "def softmax_kernel(x_ref, o_ref):\n    x = x_ref[...].astype(jnp.float32)\n    m = jnp.max(x, axis=-1, keepdims=True)\n    e = jnp.exp(x - m)\n    o_ref[...] = (e / jnp.sum(e, axis=-1, keepdims=True)).astype(o_ref.dtype)\n\ndef softmax(x, rows_per_block=8):\n    n, d = x.shape\n    return pl.pallas_call(\n        softmax_kernel,\n        grid=(n // rows_per_block,),\n        in_specs=[pl.BlockSpec((rows_per_block, d), lambda i: (i, 0))],\n        out_specs=pl.BlockSpec((rows_per_block, d), lambda i: (i, 0)),\n        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),\n        interpret=INTERP,\n    )(x)",
  steps: [
    { lines: [0, 1], note: "softmax_kernel takes just two refs, one input block and one output block, no accumulator to carry across grid steps the way the matmul kernel needed. Casting x to float32 up front is the same move as in the flash kernels: the exponential a couple of lines down needs full precision even if the data arrives and leaves in a lower-precision dtype." },
    { lines: [2, 2], note: "m holds the maximum value in each row of this block. Subtracting a row's own max before exponentiating is the standard trick for numerically stable softmax: without it, a large enough input value would overflow when exponentiated, and this row's max is guaranteed to make the largest entry come out as exp(0) = 1." },
    { lines: [3, 3], note: "e is the exponentiated, max-shifted row. This is the algorithm's core arithmetic, one call to jnp.exp over the whole block at once, no loop needed because a softmax row block, unlike a flash-attention key/value axis, is small enough to sit in VMEM whole." },
    { lines: [4, 4], note: "The output is e divided by the sum of e across each row, turning the exponentials into a proper probability distribution that sums to one, then written into o_ref cast back to the output dtype. There is no accumulator state here because this kernel sees a complete row block in one call; unlike flash attention, there is nothing to stream and nothing to carry between grid steps." },
    { lines: [6, 8], note: "softmax() is the entry point a caller invokes directly. It reads n and d, the row and column counts of the input, and takes one tunable parameter, rows_per_block, which sets how many rows each grid step handles." },
    { lines: [9, 11], note: "grid=(n // rows_per_block,) is a one-dimensional schedule: the only thing being tiled is rows, in chunks of rows_per_block. The single BlockSpec in in_specs confirms that shape, (rows_per_block, d): every grid step gets some rows but always every column, because softmax normalizes across a whole row and can't be split mid-row without losing that context." },
    { lines: [12, 15], note: "out_specs mirrors the input tiling exactly, out_shape and interpret set the result's shape, dtype, and execution mode, and (x) is the call itself. Compared to the matmul or flash kernels, this schedule is about as simple as Pallas gets: one axis, one BlockSpec shape reused for input and output, nothing to prefetch or double-buffer." },
  ],
}
