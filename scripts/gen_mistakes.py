# Generates the mistake museum: six real Pallas/JAX failures, each run on
# CPU in interpret mode inside a try/except so the captured error text is
# the genuine exception JAX raised, never invented. Run with a pinned jax;
# output is committed so site builds never need python.
#
#   uv run --with jax python scripts/gen_mistakes.py
import inspect
import json
import os
import pathlib
import textwrap

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

MAX_ERROR_CHARS = 1200


# ---------------- case 1: BlockSpec index_map returns the wrong rank ----------------

def broken_index_map_rank():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    def index_map(i):
        return (i,)  # bug: a (2, 128) block over a 2D array needs two entries

    x = jnp.ones((8, 128), dtype=jnp.float32)
    return pl.pallas_call(
        kernel,
        grid=(4,),
        in_specs=[pl.BlockSpec((2, 128), index_map)],
        out_specs=pl.BlockSpec((2, 128), index_map),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


def fixed_index_map_rank():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    def index_map(i):
        return (i, 0)  # one entry per block dimension

    x = jnp.ones((8, 128), dtype=jnp.float32)
    return pl.pallas_call(
        kernel,
        grid=(4,),
        in_specs=[pl.BlockSpec((2, 128), index_map)],
        out_specs=pl.BlockSpec((2, 128), index_map),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


# ---------------- case 2: out_shape disagrees with what the kernel writes ----------------

def broken_out_shape():
    def kernel(x_ref, o_ref):
        # writes a (8, 256) value into a ref the caller sized as (8, 128)
        o_ref[...] = jnp.concatenate([x_ref[...], x_ref[...]], axis=1)

    x = jnp.ones((8, 128), dtype=jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


def fixed_out_shape():
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.concatenate([x_ref[...], x_ref[...]], axis=1)

    x = jnp.ones((8, 128), dtype=jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 256), jnp.float32),  # matches the write
        interpret=True,
    )(x)


# ---------------- case 3: fori_loop carry structure mismatch ----------------

def broken_carry_structure():
    def kernel(x_ref, o_ref):
        def body(i, carry):
            acc, count = carry
            return acc + x_ref[i, :], count + 1, acc  # bug: returns 3 values, carry is 2

        acc, count, _ = jax.lax.fori_loop(0, 8, body, (jnp.zeros((128,), jnp.float32), 0))
        o_ref[...] = acc

    x = jnp.ones((8, 128), dtype=jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        interpret=True,
    )(x)


def fixed_carry_structure():
    def kernel(x_ref, o_ref):
        def body(i, carry):
            acc, count = carry
            return acc + x_ref[i, :], count + 1  # same structure in and out

        acc, count = jax.lax.fori_loop(0, 8, body, (jnp.zeros((128,), jnp.float32), 0))
        o_ref[...] = acc

    x = jnp.ones((8, 128), dtype=jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        interpret=True,
    )(x)


# ---------------- case 4: reading a ref with wrong-rank indexing ----------------

def broken_ref_rank_indexing():
    def kernel(x_ref, o_ref):
        # x_ref is a 2D block; a three-index read assumes a rank it doesn't have
        o_ref[...] = x_ref[:, :, 0] * 2.0

    x = jnp.ones((8, 128), dtype=jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


def fixed_ref_rank_indexing():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[:, :] * 2.0  # two indices for a 2D ref

    x = jnp.ones((8, 128), dtype=jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


# ---------------- case 5: index_map written for the wrong grid rank ----------------
# A grid that doesn't evenly divide the array is the classic way this bug
# shows up, but under jax 0.4.38's interpret mode the ragged remainder is
# padded with NaN rather than raised, so it doesn't produce real error text.
# This is the same root mistake from the angle that does raise: the
# index_map's signature no longer matches the grid it was written for, which
# is exactly what happens when someone reshapes the grid to handle a
# remainder and forgets to update every index_map that reads it.

def broken_grid_index_map_rank():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    def index_map(i, j):
        return (i, j)

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    return pl.pallas_call(
        kernel,
        grid=(2,),  # declared a 1D grid...
        in_specs=[pl.BlockSpec((4, 128), index_map)],  # ...but index_map wants two grid axes
        out_specs=pl.BlockSpec((4, 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


def fixed_grid_index_map_rank():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0

    def index_map(i):
        return (i, 0)  # matches the declared 1D grid

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    return pl.pallas_call(
        kernel,
        grid=(2,),
        in_specs=[pl.BlockSpec((4, 128), index_map)],
        out_specs=pl.BlockSpec((4, 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


# ---------------- case 6: dtype mismatch on accumulation ----------------

def broken_dtype_accumulation():
    def kernel(x_ref, o_ref):
        # x_ref is bf16; o_ref is f32. Storing the bf16 sum straight into
        # the f32 ref, with no astype.
        o_ref[...] = x_ref[...] + 1.0

    x = jnp.ones((8, 128), dtype=jnp.bfloat16)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


def fixed_dtype_accumulation():
    def kernel(x_ref, o_ref):
        o_ref[...] = (x_ref[...] + 1.0).astype(jnp.float32)

    x = jnp.ones((8, 128), dtype=jnp.bfloat16)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(x)


CASES = [
    dict(id="blockspec-index-rank", title="BlockSpec index_map returns the wrong rank",
         broken=broken_index_map_rank, fixed=fixed_index_map_rank),
    dict(id="out-shape-mismatch", title="out_shape disagrees with what the kernel writes",
         broken=broken_out_shape, fixed=fixed_out_shape),
    dict(id="carry-structure-mismatch", title="fori_loop carry structure mismatch",
         broken=broken_carry_structure, fixed=fixed_carry_structure),
    dict(id="wrong-rank-indexing", title="reading a ref with wrong-rank indexing",
         broken=broken_ref_rank_indexing, fixed=fixed_ref_rank_indexing),
    dict(id="grid-index-map-rank", title="index_map written for the wrong grid rank",
         broken=broken_grid_index_map_rank, fixed=fixed_grid_index_map_rank),
    dict(id="dtype-accumulation-mismatch", title="dtype mismatch on accumulation",
         broken=broken_dtype_accumulation, fixed=fixed_dtype_accumulation),
]


def source_of(fn) -> str:
    return textwrap.dedent(inspect.getsource(fn)).rstrip()


entries = []
for case in CASES:
    broken, fixed = case["broken"], case["fixed"]

    try:
        broken()
        raise RuntimeError(
            f"case {case['id']!r} was supposed to raise but ran clean; "
            "pick a different real failure for this class"
        )
    except RuntimeError:
        raise
    except Exception as e:  # the genuine captured error, never fabricated
        error_text = str(e)[:MAX_ERROR_CHARS]

    fixed()  # must run clean; a fix that still raises is not a fix

    entries.append({
        "id": case["id"],
        "title": case["title"],
        "snippet": source_of(broken),
        "error": error_text,
        "fix": source_of(fixed),
        "why": "",
    })

out_path = pathlib.Path(__file__).resolve().parent.parent / "site/src/data/mistakes.json"
out_path.write_text(json.dumps(entries, indent=1))
print(f"wrote {out_path.name}: {len(entries)} entries")
for e in entries:
    print(f"  {e['id']}: {e['error'].splitlines()[0]}")
