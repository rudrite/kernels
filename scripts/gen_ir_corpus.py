# Generates the IR corpus that drives the reading gym: real programs, real
# jaxprs, real StableHLO, plus structured per-equation facts extracted from
# the jaxpr objects themselves (never parsed from text). Run with a pinned
# jax; output is committed so site builds never need python.
#
#   uv run --with jax python scripts/gen_ir_corpus.py
import inspect
import json
import os
import pathlib

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

B = jnp.bfloat16
F = jnp.float32


# ---------------- programs ----------------

def mlp_block(x, w1, b1, w2, b2):
    h = jax.nn.gelu(x @ w1 + b1)
    return h @ w2 + b2

def layernorm(x, g, b):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + 1e-6) * g + b

def rmsnorm(x, g):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + 1e-6) * g

def softmax_rows(x):
    return jax.nn.softmax(x, axis=-1)

def logsumexp_rows(x):
    m = jnp.max(x, axis=-1, keepdims=True)
    return m + jnp.log(jnp.sum(jnp.exp(x - m), axis=-1, keepdims=True))

def attention(q, k, v):
    s = q @ k.T
    m = jnp.max(s, axis=-1, keepdims=True)
    p = jnp.exp(s - m)
    return (p / jnp.sum(p, axis=-1, keepdims=True)) @ v

def causal_attention(q, k, v):
    s = q @ k.T
    n = s.shape[-1]
    mask = jnp.tril(jnp.ones((n, n), bool))
    s = jnp.where(mask, s, -jnp.inf)
    return jax.nn.softmax(s, axis=-1) @ v

def batched_mha(q, k, v):
    # [batch, heads, seq, dim] scaled dot-product attention via einsum
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1]).astype(q.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1), v)

def gqa_scores(q, k):
    # grouped queries: 8 q heads share each of 2 kv heads
    qg = q.reshape(2, 4, *q.shape[1:])
    return jnp.einsum("ghqd,gkd->ghqk", qg, k)

def rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def cumsum_scan(x):
    return jax.lax.associative_scan(jnp.add, x, axis=-1)

def remat_mlp(x, w1, w2):
    f = jax.checkpoint(lambda y: jax.nn.gelu(y @ w1))
    return f(x) @ w2

def moe_dispatch(tokens, routing, expert_w):
    # tokens routed to 4 experts by argmax, computed densely with masking
    onehot = jax.nn.one_hot(jnp.argmax(routing, axis=-1), 4, dtype=tokens.dtype)
    per_expert = jnp.einsum("te,td->etd", onehot, tokens)
    out = jnp.einsum("etd,edf->etf", per_expert, expert_w)
    return jnp.einsum("te,etf->tf", onehot, out)

def attention_grads(q, k, v):
    def loss(q, k, v):
        return jnp.sum(attention(q, k, v))
    return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

def int8_dequant_matmul(x, wq, scale):
    return x @ (wq.astype(jnp.bfloat16) * scale)


mesh = Mesh(np.array(jax.devices()), ("x",))

@partial(shard_map, mesh=mesh, in_specs=(P("x", None), P(None, None)), out_specs=P("x", None))
def sharded_matmul_psum(x, w):
    return x @ w

@partial(shard_map, mesh=mesh, in_specs=(P(None, "x"), P("x", None)), out_specs=P(None, None))
def contracted_matmul_psum(x, w):
    return jax.lax.psum(x @ w, "x")


S = jax.ShapeDtypeStruct
PROGRAMS = [
    dict(id="mlp", title="MLP block", note="the transformer FFN: two matmuls around a GELU; watch bias broadcasts appear",
         fn=mlp_block, args=[S((64, 128), B), S((128, 512), B), S((512,), B), S((512, 128), B), S((128,), B)]),
    dict(id="layernorm", title="LayerNorm", note="mean, variance, rsqrt, two broadcasts: the elementwise workhorse",
         fn=layernorm, args=[S((64, 128), B), S((128,), B), S((128,), B)]),
    dict(id="rmsnorm", title="RMSNorm", note="LayerNorm minus the mean subtraction; compare their jaxprs line by line",
         fn=rmsnorm, args=[S((64, 128), B), S((128,), B)]),
    dict(id="softmax", title="softmax over rows", note="max, subtract, exp, sum, divide: five primitives and one upcast",
         fn=softmax_rows, args=[S((64, 128), B)]),
    dict(id="logsumexp", title="log-sum-exp", note="the numerically stable form: same skeleton as softmax, one log at the end",
         fn=logsumexp_rows, args=[S((64, 128), B)]),
    dict(id="attention", title="naive attention", note="the track's running example: two dot_generals bracketing a softmax",
         fn=attention, args=[S((128, 64), B)] * 3),
    dict(id="causal", title="causal attention", note="a where against -inf before the softmax: find the select and the constant",
         fn=causal_attention, args=[S((128, 64), B)] * 3),
    dict(id="mha", title="batched multi-head attention", note="einsums become dot_generals with batch dimensions: decode all four groups",
         fn=batched_mha, args=[S((2, 4, 64, 32), B)] * 3),
    dict(id="gqa", title="grouped-query scores", note="a reshape then an einsum with an unshared axis: GQA in two lines",
         fn=gqa_scores, args=[S((8, 64, 32), B), S((2, 64, 32), B)]),
    dict(id="rope", title="rotary embedding", note="split, multiply, concatenate: position becomes rotation",
         fn=rope, args=[S((64, 128), B), S((64, 64), B), S((64, 64), B)]),
    dict(id="scan", title="associative cumulative sum", note="associative_scan unrolls into a log-depth tree: count the adds",
         fn=cumsum_scan, args=[S((8, 128), F)]),
    dict(id="remat", title="checkpointed MLP", note="jax.checkpoint leaves a remat marker: find what will be recomputed",
         fn=remat_mlp, args=[S((64, 128), B), S((128, 512), B), S((512, 128), B)]),
    dict(id="moe", title="dense MoE dispatch", note="argmax routing, one-hot dispatch, three einsums: the ragged problem, densely",
         fn=moe_dispatch, args=[S((64, 32), B), S((64, 4), F), S((4, 32, 32), B)]),
    dict(id="grads", title="attention gradients", note="jax.grad of the running example: find D = rowsum(dO * O) hiding in the transposes",
         fn=attention_grads, args=[S((32, 16), F)] * 3),
    dict(id="dequant", title="int8 dequantize matmul", note="cast and scale before the dot: quantized serving in miniature",
         fn=int8_dequant_matmul, args=[S((64, 128), B), S((128, 256), jnp.int8), S((256,), B)]),
    dict(id="sharded", title="sharded matmul (data parallel)", note="under shard_map: the per-device program, no collective needed",
         fn=sharded_matmul_psum, args=[S((64, 32), F), S((32, 16), F)]),
    dict(id="psum", title="contracted matmul + psum", note="the contraction is sharded, so a psum appears: the collective as a primitive",
         fn=contracted_matmul_psum, args=[S((16, 64), F), S((64, 16), F)]),
]


def aval_info(v):
    return {"dtype": str(v.aval.dtype), "shape": list(v.aval.shape)}


def src_line_of(eq, fn_file, fn_name, fn_start, n_lines):
    """Relative source line (0-based) of the user code that made this eqn,
    read from jax's own source_info; None when it cannot be pinned."""
    tb = getattr(eq.source_info, "traceback", None)
    if tb is None:
        return None
    try:
        frames = tb.frames
    except AttributeError:
        return None
    for fr in frames:
        if fr.file_name == fn_file and fr.function_name == fn_name:
            rel = fr.line_num - fn_start
            if 0 <= rel < n_lines:
                return rel
    return None


def walk_eqns(jaxpr, out, src_ctx=None, depth=0):
    for eq in jaxpr.eqns:
        entry = {
            "op": eq.primitive.name,
            "inputs": [aval_info(v) for v in eq.invars if hasattr(v, "aval")],
            "outputs": [aval_info(v) for v in eq.outvars],
        }
        if src_ctx is not None:
            entry["src_line"] = src_line_of(eq, *src_ctx)
        if eq.primitive.name == "dot_general":
            dn = eq.params["dimension_numbers"]
            entry["dimension_numbers"] = [[list(dn[0][0]), list(dn[0][1])], [list(dn[1][0]), list(dn[1][1])]]
        out.append(entry)
        # descend into called jaxprs (remat, shard_map, custom transforms)
        for p in eq.params.values():
            if hasattr(p, "jaxpr"):
                walk_eqns(p.jaxpr, out, src_ctx, depth + 1)


corpus = []
for spec in PROGRAMS:
    fn, args = spec["fn"], spec["args"]
    closed = jax.make_jaxpr(fn)(*args)
    stablehlo = jax.jit(fn).lower(*args).as_text()
    eqns = []
    src_ctx = None
    try:
        src_lines_raw, fn_start = inspect.getsourcelines(fn)
        src_ctx = (inspect.getsourcefile(fn), fn.__name__, fn_start, len(src_lines_raw))
    except (OSError, TypeError):
        pass
    walk_eqns(closed.jaxpr, eqns, src_ctx)
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = "(wrapped: see the shard_map definition in scripts/gen_ir_corpus.py)"
    corpus.append({
        "id": spec["id"],
        "title": spec["title"],
        "note": spec["note"],
        "source": source.rstrip().splitlines(),
        "jaxpr": str(closed).splitlines(),
        "stablehlo": stablehlo.splitlines(),
        "eqns": eqns,
    })

dots = []
for prog in corpus:
    for eq in prog["eqns"]:
        if eq["op"] == "dot_general" and len(eq["inputs"]) >= 2:
            dots.append({
                "program": prog["id"],
                "lhs": eq["inputs"][0]["shape"],
                "rhs": eq["inputs"][1]["shape"],
                "out": eq["outputs"][0]["shape"],
                "dimension_numbers": eq["dimension_numbers"],
            })

out_path = pathlib.Path(__file__).resolve().parent.parent / "site/src/data/ir-corpus.json"
meta = {"jax": jax.__version__, "generated": "gen_ir_corpus.py", "programs": len(corpus), "dot_instances": len(dots)}
out_path.write_text(json.dumps({"meta": meta, "programs": corpus, "dots": dots}, indent=1))
print(f"wrote {out_path.name}: {len(corpus)} programs, {len(dots)} dot_general instances, "
      f"{sum(len(p['eqns']) for p in corpus)} eqns")
