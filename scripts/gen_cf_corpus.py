# Generates the control-flow corpus for chapter 03: lax.scan and lax.cond
# lowered to StableHLO, where they become stablehlo.while and stablehlo.if.
# Committed output; the chapter embeds excerpts with this file as provenance.
#
#   uv run --with 'jax==0.4.38' python scripts/gen_cf_corpus.py
import inspect
import json
import pathlib

import jax
import jax.numpy as jnp

F = jnp.float32
S = jax.ShapeDtypeStruct


def running_mean(xs):
    def step(carry, x):
        total, n = carry
        return (total + x, n + 1.0), (total + x) / (n + 1.0)
    (_, _), means = jax.lax.scan(step, (jnp.zeros(xs.shape[1]), 0.0), xs)
    return means


def clipped_update(x, g, lr):
    return jax.lax.cond(
        jnp.linalg.norm(g) > 1.0,
        lambda: x - lr * g / jnp.linalg.norm(g),
        lambda: x - lr * g,
    )


PROGRAMS = [
    dict(id="scan", title="lax.scan becomes stablehlo.while", fn=running_mean,
         note="the carry is explicit loop state: find the tuple threading through the while region",
         args=[S((8, 16), F)]),
    dict(id="cond", title="lax.cond becomes stablehlo.if", fn=clipped_update,
         note="both branches exist as regions; the predicate picks at runtime, not at trace time",
         args=[S((16,), F), S((16,), F), S((), F)]),
]

corpus = []
for spec in PROGRAMS:
    corpus.append({
        "id": spec["id"],
        "title": spec["title"],
        "note": spec["note"],
        "source": inspect.getsource(spec["fn"]).rstrip().splitlines(),
        "jaxpr": str(jax.make_jaxpr(spec["fn"])(*spec["args"])).splitlines(),
        "stablehlo": jax.jit(spec["fn"]).lower(*spec["args"]).as_text().splitlines(),
    })
    print(f"{spec['id']}: {len(corpus[-1]['stablehlo'])} hlo lines")

out = pathlib.Path(__file__).resolve().parent.parent / "site/src/data/cf-corpus.json"
meta = {"jax": jax.__version__, "generated": "gen_cf_corpus.py"}
out.write_text(json.dumps({"meta": meta, "programs": corpus}, indent=1))
print(f"wrote {out.name}")
