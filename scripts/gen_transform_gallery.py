# Generates the transform gallery for chapter 02: one small function shown
# as a jaxpr under each transform that changes its shape. Committed output;
# site builds never need python.
#
#   uv run --with 'jax==0.4.38' python scripts/gen_transform_gallery.py
import json
import os
import pathlib
from functools import partial

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

B = jnp.bfloat16
S = jax.ShapeDtypeStruct


def dense(x, w, b):
    return jnp.tanh(x @ w + b)


x = S((32, 64), B)
w = S((64, 64), B)
b = S((64,), B)

mesh = Mesh(np.array(jax.devices()), ("x",))
sharded_dense = partial(shard_map, mesh=mesh,
                        in_specs=(P("x", None), P(None, None), P(None)),
                        out_specs=P("x", None))(dense)

VARIANTS = [
    dict(id="plain", title="the function itself", note="one matmul, one bias broadcast, one tanh: the baseline every transform rewrites",
         fn=dense, args=[x, w, b]),
    dict(id="grad", title="under jax.grad", note="the forward is joined by its transpose: find tanh's derivative (1 - tanh^2) as mul equations",
         fn=jax.grad(lambda x, w, b: jnp.sum(dense(x, w, b)), argnums=1), args=[x, w, b]),
    dict(id="vmap", title="under jax.vmap", note="every equation gains a batch dimension; nothing else changes, which is the whole point of vmap",
         fn=jax.vmap(dense, in_axes=(0, None, None)), args=[S((8, 32, 64), B), w, b]),
    dict(id="scan", title="under lax.scan", note="the body becomes one nested jaxpr run length times: read the carry in, the carry out",
         fn=lambda x, ws, b: jax.lax.scan(lambda h, w: (dense(h, w, b), None), x, ws)[0],
         args=[x, S((4, 64, 64), B), b]),
    dict(id="remat", title="under jax.checkpoint", note="a remat wrapper marks the body for recompute in the backward pass; the body itself is unchanged",
         fn=jax.grad(lambda x, w, b: jnp.sum(jax.checkpoint(dense)(x, w, b)), argnums=1), args=[x, w, b]),
    dict(id="shard_map", title="under shard_map", note="the per-device program: same equations, per-shard shapes, and no collective because none is needed",
         fn=sharded_dense, args=[x, w, b]),
]

gallery = []
for spec in VARIANTS:
    closed = jax.make_jaxpr(spec["fn"])(*spec["args"])
    gallery.append({
        "id": spec["id"],
        "title": spec["title"],
        "note": spec["note"],
        "jaxpr": str(closed).splitlines(),
    })
    print(f"{spec['id']}: {len(gallery[-1]['jaxpr'])} lines")

out = pathlib.Path(__file__).resolve().parent.parent / "site/src/data/transform-gallery.json"
meta = {"jax": jax.__version__, "generated": "gen_transform_gallery.py", "source_fn": "dense(x, w, b) = tanh(x @ w + b)"}
out.write_text(json.dumps({"meta": meta, "variants": gallery}, indent=1))
print(f"wrote {out.name}")
