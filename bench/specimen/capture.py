#!/usr/bin/env python3
"""Capture the specimen at every level this machine can reach.

    python3 bench/specimen/capture.py

Writes bench/specimen/artifacts/ and bench/specimen/manifest.json. The run
clears the artifact directory first, so a second run leaves exactly what the
first one did, and the site reads the committed files at build time rather
than compiling anything itself.

Seven artifacts come out of one process on the CPU backend: the source, the
jaxpr, the StableHLO module, the HLO before and after the backend pipeline,
and the LLVM IR the CPU backend hands to LLVM before and after LLVM's own
passes. The levels this machine cannot reach are listed in the manifest as
pending, with the capture that will fill each one. Nothing is synthesized.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
MANIFEST = HERE / "manifest.json"
REPO = HERE.parent.parent
COMMAND = "python3 bench/specimen/capture.py"

# The dump flag has to be in the environment before jax picks a backend, so it
# goes in before the import rather than next to the compile it belongs to.
DUMP = Path(tempfile.mkdtemp(prefix="specimen-xla-"))
os.environ["XLA_FLAGS"] = f"--xla_dump_to={DUMP}"

import jax  # noqa: E402
import jaxlib  # noqa: E402

import specimen  # noqa: E402

# Instruction metadata names the file and line the compile was driven from,
# which describes the machine that ran it rather than the program. op_name
# stays, because it is the thread back to the jaxpr.
SOURCE_TAG = re.compile(r'\s*source_file="[^"]*"(\s+source_line=\d+)?')
ABSOLUTE_PATH = re.compile(r"(?<![\w.])/(?:Users|home|private|var|tmp|opt|usr)/[\w./+-]+")

SCRUB_NOTE = (
    "HLO instruction metadata keeps op_name and drops source_file and source_line, "
    "because those name the machine that ran the compile rather than the program. "
    "The LLVM IR and the StableHLO module carry no paths and are written verbatim. "
    "Nothing else is edited."
)

PENDING = [
    {
        "id": "torch-lazy-ir",
        "level": 1,
        "lane": "torch",
        "title": "the torch_xla lazy trace and its seam calls",
        "pending": (
            "A Colab TPU runtime runs the same block under torch_xla and pastes back "
            "_XLAC._get_xla_tensors_hlo plus the PJRT calls torch_xla.debug.metrics counts, "
            "the way the reader-driven labs already hand numbers back."
        ),
    },
    {
        "id": "triton-ttir",
        "level": 4,
        "lane": "gpu",
        "title": "the Triton kernel, TTIR",
        "pending": (
            "A cited public capture pinned to its source commit; filled by paste-in, "
            "carried through re-runs by the preservation block above."
        ),
    },
    {
        "id": "triton-ttgir",
        "level": 4,
        "lane": "gpu",
        "title": "the Triton kernel, TTGIR",
        "pending": (
            "A cited public capture pinned to its source commit; filled by paste-in, "
            "carried through re-runs by the preservation block above."
        ),
    },
    {
        "id": "ptx",
        "level": 5,
        "lane": "gpu",
        "title": "PTX",
        "pending": (
            "A cited public capture: the .ptx file --xla_dump_to leaves on a CUDA machine, "
            "pinned to the CUDA toolkit that emitted it."
        ),
    },
    {
        "id": "sass",
        "level": 5,
        "lane": "gpu",
        "title": "SASS",
        "pending": (
            "A cited public capture: nvdisasm over the cubin from the same CUDA run, "
            "pinned to the GPU generation, since SASS is one generation wide."
        ),
    },
]


def scrub(text: str) -> str:
    cleaned = SOURCE_TAG.sub("", text)
    leftover = ABSOLUTE_PATH.search(cleaned)
    if leftover:
        raise SystemExit(f"capture: an absolute path survived the scrub: {leftover.group(0)}")
    return cleaned


def dumped(pattern: str) -> list[Path]:
    return sorted(p for p in DUMP.iterdir() if pattern in p.name)


def join_parts(paths: list[Path]) -> str:
    """XLA splits the CPU module across parallel-compilation parts. They are
    separate LLVM modules, so they are kept apart by a header rather than
    concatenated into something that never existed."""
    if not paths:
        raise SystemExit("capture: the CPU backend dumped no LLVM IR")
    if len(paths) == 1:
        return paths[0].read_text()
    chunks = []
    for i, path in enumerate(paths):
        chunks.append(f"; ==== module part {i:02d} of {len(paths):02d} ====\n{path.read_text()}")
    return "\n".join(chunks)


def write(name: str, text: str) -> Path:
    path = ARTIFACTS / name
    path.write_text(text if text.endswith("\n") else text + "\n")
    return path


def record(path: Path, level: int, lane: str, title: str, lang: str, produced_by: str) -> dict:
    text = path.read_text()
    return {
        "id": path.stem,
        "level": level,
        "lane": lane,
        "title": title,
        "file": str(path.relative_to(REPO)),
        "lang": lang,
        "lines": len(text.splitlines()),
        "bytes": len(text.encode()),
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "produced_by": produced_by,
    }


def main() -> None:
    # Paste-back rows (filled outside this script, like the Colab torch trace)
    # survive a re-run: their manifest entries and artifact bytes are carried
    # over instead of being reset to the pending placeholder.
    pasted: dict[str, tuple[dict, bytes]] = {}
    pending_ids = {entry["id"] for entry in PENDING}
    if MANIFEST.exists():
        for entry in json.loads(MANIFEST.read_text()).get("levels", []):
            if entry["id"] in pending_ids and entry.get("status") == "captured":
                src = ARTIFACTS / Path(entry["file"]).name
                if src.exists():
                    pasted[entry["id"]] = (entry, src.read_bytes())

    shutil.rmtree(ARTIFACTS, ignore_errors=True)
    ARTIFACTS.mkdir(parents=True)
    for entry, data in pasted.values():
        (ARTIFACTS / Path(entry["file"]).name).write_bytes(data)

    x, wq, wk, wv = specimen.inputs()
    lowered = jax.jit(specimen.block).lower(x, wq, wk, wv)
    compiled = lowered.compile()
    result = compiled(x, wq, wk, wv)

    levels = [
        record(
            write("source.py", (HERE / "specimen.py").read_text()),
            0,
            "the program",
            "the program you wrote",
            "python",
            "bench/specimen/specimen.py, copied verbatim",
        ),
        record(
            write("jaxpr.txt", str(jax.make_jaxpr(specimen.block)(x, wq, wk, wv))),
            1,
            "jax",
            "the jaxpr",
            "jaxpr",
            "jax.make_jaxpr(block)(x, wq, wk, wv)",
        ),
        record(
            write("stablehlo.mlir", lowered.as_text()),
            2,
            "waist",
            "StableHLO, as the program leaves JAX",
            "mlir",
            "jax.jit(block).lower(x, wq, wk, wv).as_text()",
        ),
    ]

    before = dumped("jit_block.before_optimizations.txt")
    after = dumped("jit_block.cpu_after_optimizations.txt")
    if not before or not after:
        raise SystemExit("capture: the HLO dump pair is missing; check XLA_FLAGS")
    levels += [
        record(
            write("hlo-before-optimizations.txt", scrub(before[0].read_text())),
            3,
            "waist",
            "HLO, before the backend pipeline",
            "hlo",
            "XLA_FLAGS=--xla_dump_to=... · jit_block.before_optimizations.txt",
        ),
        record(
            write("hlo-after-optimizations.txt", scrub(after[0].read_text())),
            3,
            "waist",
            "HLO, fused and scheduled",
            "hlo",
            "XLA_FLAGS=--xla_dump_to=... · jit_block.cpu_after_optimizations.txt",
        ),
        record(
            write("llvm-ir-no-opt.ll", scrub(join_parts(dumped("jit_block.ir-no-opt")))),
            5,
            "cpu",
            "LLVM IR, as XLA emits it",
            "llvm",
            "XLA_FLAGS=--xla_dump_to=... · jit_block.ir-no-opt*.ll",
        ),
        record(
            write("llvm-ir-with-opt.ll", scrub(join_parts(dumped("jit_block.ir-with-opt")))),
            5,
            "cpu",
            "LLVM IR, after LLVM's own passes",
            "llvm",
            "XLA_FLAGS=--xla_dump_to=... · jit_block.ir-with-opt*.ll",
        ),
    ]
    for entry in levels:
        entry["status"] = "captured"
    for entry in PENDING:
        entry["status"] = "pending"
    # a pasted-back row keeps its own manifest entry over the placeholder
    tail = [pasted[e["id"]][0] if e["id"] in pasted else e for e in PENDING]

    manifest = {
        "specimen": {
            "id": "attention-block",
            "title": "one attention block, sixteen rows, no batch axis",
            "source": "bench/specimen/specimen.py",
            "shapes": "x f32[16,32], three weights f32[32,32]",
            "seed": specimen.SEED,
            "checksum": f"the block sums to {float(result.sum()):.6f} on this run",
        },
        "capture": {
            "command": COMMAND,
            "cwd": "the repo root",
            "backend": str(jax.devices()[0].platform),
            "devices": [str(d) for d in jax.devices()],
            "platform": platform.platform(),
            "versions": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
            },
            "scrub": SCRUB_NOTE,
        },
        # Descent order, so the manifest reads the way the map does. The sort
        # is stable, which keeps a pending level under the captures it sits with.
        "levels": sorted(levels + tail, key=lambda entry: entry["level"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    shutil.rmtree(DUMP, ignore_errors=True)
    print(f"specimen: {len(levels)} artifacts captured, {len(PENDING)} levels pending")
    for entry in levels:
        print(f"  {entry['file']:52s} {entry['lines']:5d} lines  {entry['bytes']:7d} bytes")
    print(f"manifest: {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
