# Generates the Mosaic corpus for the gym: real Pallas kernels captured at
# the layer below pallas_call. For each kernel we record its source, the
# kernel jaxpr, and the Mosaic (tpu dialect) module, all taken from Pallas's
# own debug output during lowering. No TPU needed: jax.export cross-lowers
# for the tpu platform on any machine, and the Mosaic passes ship in jaxlib.
#
# The jaxpr-to-Mosaic line pairing is computed here, mechanically, by
# order-preserving matching over op families. Anything the pairer cannot
# place with confidence stays unmapped rather than guessed.
#
#   uv run --with 'jax==0.4.38' --with absl-py python scripts/gen_mosaic_corpus.py
import contextlib
import inspect
import io
import json
import pathlib
import re

import jax
import jax.export
import jax.numpy as jnp
from jax.experimental import pallas as pl

B = jnp.bfloat16
F = jnp.float32
S = jax.ShapeDtypeStruct


# ---------------- kernels ----------------
# Each is a track kernel: LAB 1.1's elementwise add, LAB 1.2's tiled matmul,
# LAB 1.3's fused row softmax. Bodies match the labs; debug=True is the only
# addition.

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=S((256, 256), B),
        debug=True,
    )(x, y)


def matmul_kernel(a_ref, b_ref, o_ref):
    k = pl.program_id(2)
    @pl.when(k == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)
    o_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)

def matmul(a, b, bm=256, bn=256, bk=256):
    m, k = a.shape
    _, n = b.shape
    return pl.pallas_call(
        matmul_kernel,
        grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
                  pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=S((m, n), B),
        debug=True,
    )(a, b)


def softmax_kernel(x_ref, o_ref):
    x = x_ref[...].astype(jnp.float32)
    m = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x - m)
    o_ref[...] = (e / jnp.sum(e, axis=-1, keepdims=True)).astype(o_ref.dtype)

def softmax(x):
    return pl.pallas_call(
        softmax_kernel,
        out_shape=S(x.shape, B),
        debug=True,
    )(x)


KERNELS = [
    dict(id="add", title="elementwise add (LAB 1.1)", kernel=add_kernel, entry=add,
         note="the smallest possible read: two loads, one VPU add, one store; start here",
         args=[S((256, 256), B), S((256, 256), B)]),
    dict(id="matmul", title="tiled matmul (LAB 1.2)", kernel=matmul_kernel, entry=matmul,
         note="the running kernel: pl.when becomes a real branch, jnp.dot becomes tpu.matmul",
         args=[S((512, 512), B), S((512, 512), B)]),
    dict(id="softmax", title="fused row softmax (LAB 1.3)", kernel=softmax_kernel, entry=softmax,
         note="the VPU story: an upcast, a row reduction, a transcendental, all lane-wise",
         args=[S((256, 512), B)]),
]


# ---------------- capture ----------------

JAXPR_HDR = "The kernel jaxpr for pallas_call"
MOSAIC_HDR = "The Mosaic module for pallas_call"


def capture(entry, args):
    """Run the tpu cross-lowering and return (jaxpr_lines, mosaic_lines),
    with the debug headers (which carry absolute paths) dropped."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jax.export.export(jax.jit(entry), platforms=["tpu"])(*args)
    text = buf.getvalue()
    j_at = text.index(JAXPR_HDR)
    m_at = text.index(MOSAIC_HDR)
    jaxpr_block = text[j_at:m_at].splitlines()[1:]
    mosaic_block = text[m_at:].splitlines()[1:]
    strip = lambda lines: [ln.rstrip() for ln in lines if ln.strip()]
    return strip(jaxpr_block), strip(mosaic_block)


# ---------------- jaxpr side: eqn extraction ----------------

def jaxpr_eqns(lines):
    """Yield (op, [line indices]) per top-level equation, in program order.
    A top-level line (exactly 4-space indent) starts a new equation unless
    it is a closer (`] k l`); deeper-indented lines, including whole nested
    jaxprs under cond/scan, belong to the equation above them."""
    eqns = []
    current = None
    for i, ln in enumerate(lines):
        if not ln.startswith("    "):
            continue  # header / footer
        top_level = not ln.startswith("     ")
        closer = ln.lstrip().startswith(("]", ")"))
        if top_level and not closer:
            if " <- " in ln:
                _, rhs = ln.split(" <- ", 1)
                op = "load" if re.match(r"\s*\w+\[", rhs) else "store"
            elif " = " in ln:
                m = re.search(r"= (\w+)", ln)
                op = m.group(1) if m else None
            else:
                op = ln.strip().split("[")[0].split(" ")[0]  # bare call: cond, scan
            current = (op, [i])
            eqns.append(current)
        elif current is not None:
            current[1].append(i)
    return [(op, idx) for op, idx in eqns if op]


# ---------------- mosaic side: op extraction ----------------

MOSAIC_OP_RE = re.compile(r"(?:= |^\s*)(tpu\.\w+|vector\.\w+|arith\.\w+|math\.\w+|scf\.\w+|func\.\w+)")


def mosaic_ops(lines):
    """(op token, line index) per op line, in order. multi_reduction carries
    its kind so max and sum reductions stay distinguishable."""
    ops = []
    for i, ln in enumerate(lines):
        m = MOSAIC_OP_RE.search(ln)
        if not m:
            continue
        tok = m.group(1)
        if tok == "vector.multi_reduction":
            kind = re.search(r"<(\w+)>", ln)
            tok = f"multi_reduction:{kind.group(1)}" if kind else tok
        ops.append((tok, i))
    return ops


# jaxpr op family -> acceptable mosaic op tokens, in the order lowering
# emits them; pairing walks both lists forward and never backtracks
PAIR: dict[str, set[str]] = {
    "load": {"vector.load"},
    "store": {"tpu.vector_store", "vector.store"},
    "dot_general": {"tpu.matmul"},
    "convert_element_type": {"arith.truncf", "arith.extf", "arith.extui", "arith.sitofp", "arith.fptosi"},
    "exp": {"math.exp"},
    "eq": {"arith.cmpi", "arith.cmpf"},
    "add": {"arith.addf", "arith.addi"},
    "sub": {"arith.subf", "arith.subi"},
    "mul": {"arith.mulf", "arith.muli"},
    "div": {"arith.divf"},
    "max": {"arith.maximumf"},
    "reduce_max": {"multi_reduction:maximumf", "multi_reduction:maxf", "multi_reduction:maxnumf"},
    "reduce_sum": {"multi_reduction:add"},
    "broadcast_in_dim": {"vector.broadcast", "tpu.broadcast", "vector.shape_cast"},
    "cond": {"scf.if"},
}


def region_end(lines, start):
    """Index of the line closing the brace region opened at `start`."""
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0:
            return i
    return start


def pair_eqns(eqns, mops, mosaic_lines):
    """Order-preserving pairing: for each jaxpr eqn walk the mosaic op list
    forward for the first acceptable candidate. A cond matched to scf.if
    claims the whole brace region (its branches lowered in place), and the
    cursor skips past it so inner ops never mis-pair with later equations.
    Unmatched eqns are skipped, never guessed."""
    groups = []
    cursor = 0
    for op, jlines in eqns:
        accept = PAIR.get(op)
        if not accept:
            continue
        for probe in range(cursor, len(mops)):
            tok, mline = mops[probe]
            if tok in accept:
                if tok == "scf.if":
                    end = region_end(mosaic_lines, mline)
                    groups.append({"op": op, "jaxpr": jlines, "mosaic": list(range(mline, end + 1))})
                    cursor = probe
                    while cursor < len(mops) and mops[cursor][1] <= end:
                        cursor += 1
                else:
                    groups.append({"op": op, "jaxpr": jlines, "mosaic": [mline]})
                    cursor = probe + 1
                break
    return groups


# ---------------- source side: conservative token scan ----------------

SRC_TOKENS = {
    "dot_general": "jnp.dot",
    "exp": "jnp.exp",
    "reduce_max": "jnp.max",
    "reduce_sum": "jnp.sum",
    "cond": "pl.when",
}


def attach_src(groups, src_lines, kernel_fn, jaxpr_lines):
    """Attach a kernel-source line to a group only when exactly one line
    carries the op's token. Loads match on the ref parameter name, resolved
    through the jaxpr header (invar order = parameter order)."""
    params = [p for p in inspect.signature(kernel_fn).parameters]
    invars = re.findall(r"(\w+):MemRef", jaxpr_lines[0]) if jaxpr_lines else []
    var_to_param = dict(zip(invars, params))
    for g in groups:
        if g["op"] != "load":
            continue
        rhs = jaxpr_lines[g["jaxpr"][0]].split(" <- ")[1].strip()
        pname = var_to_param.get(rhs.split("[")[0])
        if not pname:
            continue
        hits = [i for i, ln in enumerate(src_lines) if f"{pname}[" in ln]
        if len(hits) == 1:
            g["src"] = hits
    for g in groups:
        tok = SRC_TOKENS.get(g["op"])
        if not tok:
            continue
        hits = [i for i, ln in enumerate(src_lines) if tok in ln and tok + "s" not in ln]
        if len(hits) == 1:
            g["src"] = hits
    out_ref = params[-1] if params else None
    if out_ref:
        store_hits = [i for i, ln in enumerate(src_lines) if re.search(rf"{out_ref}\[[^\]]*\]\s*[+]?=", ln)]
        if len(store_hits) > 1:  # an accumulating kernel: the += line is the top-level store
            store_hits = [i for i in store_hits if "+=" in src_lines[i]]
        stores = [g for g in groups if g["op"] == "store"]
        if len(store_hits) == 1 and len(stores) == 1:
            stores[0]["src"] = store_hits
    return groups


# ---------------- main ----------------

corpus = []
for spec in KERNELS:
    jaxpr_lines, mosaic_lines = capture(spec["entry"], spec["args"])
    src = inspect.getsource(spec["kernel"]).rstrip().splitlines()
    groups = pair_eqns(jaxpr_eqns(jaxpr_lines), mosaic_ops(mosaic_lines), mosaic_lines)
    attach_src(groups, src, spec["kernel"], jaxpr_lines)
    corpus.append({
        "id": spec["id"],
        "title": spec["title"],
        "note": spec["note"],
        "source": src,
        "jaxpr": jaxpr_lines,
        "mosaic": mosaic_lines,
        "groups": groups,
    })
    print(f"{spec['id']}: {len(jaxpr_lines)} jaxpr lines, {len(mosaic_lines)} mosaic lines, "
          f"{len(groups)} paired groups, {sum(1 for g in groups if 'src' in g)} with source")

out_path = pathlib.Path(__file__).resolve().parent.parent / "site/src/data/mosaic-corpus.json"
meta = {
    "jax": jax.__version__,
    "generated": "gen_mosaic_corpus.py",
    "method": "pallas_call debug=True under jax.export cross-lowering for tpu",
    "kernels": len(corpus),
}
out_path.write_text(json.dumps({"meta": meta, "kernels": corpus}, indent=1))
print(f"wrote {out_path.name}")
