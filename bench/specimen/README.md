# The specimen

One program, written down at every level the machine under it can reach.
`specimen.py` holds the program: an attention block sixteen rows wide, three
projections, one softmax, no batch axis. `capture.py` runs it once and writes
what each stage of the descent made of it into `artifacts/`, with
`manifest.json` recording the versions, the command, and what was edited.

Regenerate everything:

    python3 bench/specimen/capture.py

The run needs jax and jaxlib on the CPU backend, nothing else. It clears
`artifacts/` first, so re-running leaves exactly what the previous run did,
and a diff on the committed files is the check that the capture still
reproduces.

Seven artifacts come out of one process: the source, the jaxpr, the StableHLO
module, the HLO on both sides of the backend pipeline, and the LLVM IR on
both sides of LLVM's own passes. Four levels stay empty and say so in the
manifest, each with the capture that will fill it: the torch_xla lazy trace
and its seam calls, the Triton dialects, PTX, and SASS. None of those runs on
a CPU, so none of them is written here.

HLO instruction metadata is the one place the dumps name the machine that ran
the compile. `op_name` is kept, `source_file` and `source_line` come off, and
the script fails the run if any absolute path survives that pass. Everything
else is verbatim.

The site reads `manifest.json` and `artifacts/` at build time; nothing is
compiled during a build. LAB·L1 walks the same descent by hand in a notebook,
which is where a reader meets it first.
