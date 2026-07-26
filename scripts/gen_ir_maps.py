# Generates a mechanical jaxpr-to-StableHLO line mapping for every program in
# the IR corpus. Pure text walking, no jax import needed: equations are found
# by bracket-balance over the jaxpr's own printed lines, ops are found by the
# `%x = ` def pattern over the StableHLO's own printed lines, and the two are
# aligned in execution order through a small op-name correspondence table.
#
# Where alignment is not certain, the mapper skips rather than guesses: an
# op absent from the table, an op whose match text disagrees on the output
# shape/dtype the two dumps both print, or an op with no plausible partner
# left in the stream all come out unmapped. Two known ambiguities are called
# out explicitly below (see FORCE_SKIP_AFTER and the recursion note).
#
#   python3 scripts/gen_ir_maps.py
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "site/src/data/ir-corpus.json"
OUT_PATH = ROOT / "site/src/data/ir-maps.json"

# ---------------- jaxpr side: recursive equation parser ----------------
#
# An equation start looks like "  f:bf16[64,512] = dot_general[" (leading
# whitespace, a bound name, a colon-led type, then "="). Continuation lines
# (params, nested "jaxpr={...}" bodies) never match that shape themselves,
# so a plain bracket-balance walk from the start line finds exactly the
# equation's own span, nested content included, with no special-casing per
# primitive.
EQ_START_RE = re.compile(r"^\s+\w+:.*=")


def bracket_delta(line: str) -> int:
    return line.count("[") - line.count("]")


def extract_op(line: str) -> str:
    after = line.split("=", 1)[1].strip()
    m = re.match(r"\w+", after)
    return m.group(0) if m else ""


def parse_equations(lines: list[str], lo: int, hi: int, recurse: bool) -> list[dict]:
    """Flat, in-order list of {op, lines} for equations in lines[lo:hi].

    With recurse=True, any equation whose span has continuation lines is
    also searched for equations nested inside it (a pjit/shard_map/remat2
    body), inserted right after their parent in execution order. This is
    what lets a wrapper's own inline computation (remat2 in this corpus)
    still get matched against the HLO it really produces, instead of
    leaving every equation after it desynced from the HLO stream.
    """
    groups = []
    i = lo
    while i < hi:
        line = lines[i]
        if not EQ_START_RE.match(line):
            i += 1
            continue
        start = i
        bal = bracket_delta(line)
        end = i
        while bal > 0:
            end += 1
            bal += bracket_delta(lines[end])
        op = extract_op(line)
        groups.append({"op": op, "lines": list(range(start, end + 1))})
        if recurse and end > start:
            groups.extend(parse_equations(lines, start + 1, end, True))
        i = end + 1
    return groups


def parse_jaxpr_type(line: str) -> tuple[str, tuple[str, ...]] | None:
    """dtype/shape of an equation's (first, if multi-output) bound name."""
    head = line.strip().split("=", 1)[0]
    first = head.split()[0] if head.split() else ""
    m = re.match(r"\w+:(\w+)(\[([^\]]*)\])?$", first)
    if not m:
        return None
    dtype = DTYPE_ALIAS.get(m.group(1), m.group(1))
    dims_str = m.group(3)
    dims = tuple(d.strip() for d in dims_str.split(",") if d.strip()) if dims_str else ()
    return dtype, dims


# ---------------- stablehlo side: atom stream over @main only ----------------
#
# pjit/shard_map compile to separate `func.func private @callee` blocks
# called via `%x = call @callee(...)`, never inlined at the call site (JAX
# 0.4.38 / this corpus). Scanning only the public @main body means those
# callees' internal ops (which reuse names like broadcast_in_dim, convert,
# add that also appear for real, later, top-level equations) never enter
# the stream at all, so they can't be mismatched onto an unrelated equation.
# `call @callee` lines are kept as inert atoms: they never equal any
# correspondence-table name, so they're harmlessly skipped over.
OP_LINE_RE = re.compile(r"%\w+ = ")
DTYPE_ALIAS = {"bool": "i1"}


def main_body_lines(hlo: list[str]) -> tuple[int, list[str]]:
    start = next(i for i, l in enumerate(hlo) if "func.func public @main" in l)
    depth = 0
    end = start
    for i in range(start, len(hlo)):
        depth += hlo[i].count("{") - hlo[i].count("}")
        if depth == 0:
            end = i
            break
    return start, hlo[start + 1 : end]


def build_atoms(hlo: list[str]) -> list[dict]:
    start, body = main_body_lines(hlo)
    atoms = []
    for offset, line in enumerate(body):
        if not OP_LINE_RE.search(line):
            continue
        orig = start + 1 + offset
        if "stablehlo.constant" in line:
            atoms.append({"kind": "constant", "line": orig})
            continue
        m = re.search(r"=\s*stablehlo\.(\w+)", line)
        atoms.append({"kind": "op", "name": m.group(1) if m else None, "line": orig, "text": line})
    return atoms


def parse_hlo_type(text: str) -> tuple[str, tuple[str, ...]] | None:
    matches = re.findall(r"tensor<([^>]*)>", text)
    if not matches:
        return None
    parts = matches[-1].split("x")
    dtype = DTYPE_ALIAS.get(parts[-1], parts[-1])
    return dtype, tuple(parts[:-1])


# ---------------- op-name correspondence table ----------------
#
# The compound entries (reduce_max, reduce_sum) match a StableHLO `reduce`
# line whose text names the reducer it applies; simple entries match an
# exact StableHLO op name. Everything absent here (argmax, eq/ge/gt,
# integer_pow, pjit, pbroadcast, psum2, remat2 itself, select_n, shard_map,
# split, square, stop_gradient) has a lowering this table can't state
# without guessing at compiler internals, so it's left unmapped.
CORRESPONDENCE = {
    "dot_general": {"hlo": "dot_general"},
    "transpose": {"hlo": "transpose"},
    "exp": {"hlo": "exponential"},
    "sub": {"hlo": "subtract"},
    "div": {"hlo": "divide"},
    "broadcast_in_dim": {"hlo": "broadcast_in_dim"},
    "convert_element_type": {"hlo": "convert"},
    "reduce_max": {"hlo": "reduce", "requires_substr": "maximum"},
    "reduce_sum": {"hlo": "reduce", "requires_substr": "add"},
    # unambiguous same-computation renames/identities, safe to add mechanically
    "add": {"hlo": "add"},
    "add_any": {"hlo": "add"},
    "mul": {"hlo": "multiply"},
    "tanh": {"hlo": "tanh"},
    "sqrt": {"hlo": "sqrt"},
    "rsqrt": {"hlo": "rsqrt"},
    "iota": {"hlo": "iota"},
    "log": {"hlo": "log"},
    "reshape": {"hlo": "reshape"},
    "pad": {"hlo": "pad"},
    "slice": {"hlo": "slice"},
    "concatenate": {"hlo": "concatenate"},
    "neg": {"hlo": "negate"},
    "max": {"hlo": "maximum"},
}

# integer_pow[y=N] lowers to a chain of `multiply` ops (verified against the
# real dumps for mlp and remat's checkpointed gelu). Since it isn't in the
# table above it's always skipped on its own, but the very next equation, if
# it's a real `mul`, would otherwise adjacency-match one of integer_pow's
# own multiplies (same op name, same shape - the shape check below can't
# tell them apart either). Observed only for a `mul` immediately following;
# force that one case to skip too rather than show a wrong line.
FORCE_SKIP_AFTER = {("integer_pow", "multiply")}


def align(groups: list[dict], jaxpr: list[str], atoms: list[dict]) -> tuple[list[dict], dict]:
    stats = {"not_in_table": 0, "force_skip": 0, "no_match": 0, "shape_mismatch": 0, "mapped": 0}
    out = []
    ptr = 0
    for idx, g in enumerate(groups):
        op = g["op"]
        rule = CORRESPONDENCE.get(op)
        if rule is None:
            stats["not_in_table"] += 1
            continue
        if idx > 0 and (groups[idx - 1]["op"], rule["hlo"]) in FORCE_SKIP_AFTER:
            stats["force_skip"] += 1
            continue

        buffer: list[int] = []
        i = ptr
        matched = None
        while i < len(atoms):
            atom = atoms[i]
            if atom["kind"] == "constant":
                buffer.append(atom["line"])
                i += 1
                continue
            if atom["name"] == rule["hlo"] and ("requires_substr" not in rule or rule["requires_substr"] in atom["text"]):
                matched = atom
                i += 1
                break
            buffer = []
            i += 1
        if matched is None:
            stats["no_match"] += 1
            continue

        jaxpr_type = parse_jaxpr_type(jaxpr[g["lines"][0]])
        hlo_type = parse_hlo_type(matched["text"])
        if jaxpr_type is None or hlo_type != jaxpr_type:
            stats["shape_mismatch"] += 1
            continue

        ptr = i
        out.append({"op": op, "idx": idx, "jaxpr": g["lines"], "hlo": buffer + [matched["line"]]})
        stats["mapped"] += 1
    return out, stats


# Programs whose fully-recursive equation count diverges from the corpus's
# own `eqns` (a flattened equation list generated straight from the jaxpr
# object, not from text). Both known divergences are wrapper primitives
# that `eqns` doesn't expand but this script's parser does: remat2 (its
# body is genuinely inline in the HLO, so expanding it is what lets the
# ops after it stay in sync - see FORCE_SKIP_AFTER's comment) and
# shard_map (its body compiles to an excluded private function, so
# expanding it is harmless, just extra groups that will not find a match).
KNOWN_EQNS_DIVERGENCE = {"remat", "sharded", "psum"}


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text())
    maps: dict[str, dict] = {}
    total_mapped = 0
    total_skipped = 0
    print(f"{'program':<12} {'groups':>7} {'mapped':>7} {'no-table':>9} {'no-match':>9} {'shape':>6} {'force':>6}")
    for program in corpus["programs"]:
        pid = program["id"]
        jaxpr = program["jaxpr"]
        full = parse_equations(jaxpr, 0, len(jaxpr), True)

        eqn_ops = [e["op"] for e in program["eqns"]]
        full_ops = [g["op"] for g in full]
        if pid not in KNOWN_EQNS_DIVERGENCE and full_ops != eqn_ops:
            raise AssertionError(f"{pid}: recursive jaxpr parse disagrees with eqns\n  parsed={full_ops}\n  eqns  ={eqn_ops}")

        atoms = build_atoms(program["stablehlo"])
        groups, stats = align(full, jaxpr, atoms)
        # source lines ride along only where the text parse and the corpus's
        # own eqn walk are known to agree index-for-index
        src_ok = pid not in KNOWN_EQNS_DIVERGENCE
        eqns = program["eqns"]
        out_groups = []
        for g in groups:
            entry = {"jaxpr": g["jaxpr"], "hlo": g["hlo"], "op": g["op"]}
            if src_ok:
                src_line = eqns[g["idx"]].get("src_line")
                if src_line is not None:
                    entry["src"] = [src_line]
            out_groups.append(entry)
        maps[pid] = {"groups": out_groups}

        skipped = stats["not_in_table"] + stats["force_skip"] + stats["no_match"] + stats["shape_mismatch"]
        total_mapped += stats["mapped"]
        total_skipped += skipped
        print(
            f"{pid:<12} {len(full):>7} {stats['mapped']:>7} {stats['not_in_table']:>9} "
            f"{stats['no_match']:>9} {stats['shape_mismatch']:>6} {stats['force_skip']:>6}"
        )

    OUT_PATH.write_text(json.dumps(maps, indent=2) + "\n")
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")
    print(f"total: {total_mapped} groups mapped, {total_skipped} equations skipped as ambiguous")


if __name__ == "__main__":
    main()
