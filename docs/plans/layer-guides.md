# Layer guides: every chapter becomes a full tutorial

Goal: each layer chapter (/l/<id>) teaches its layer to mastery on its own,
at the depth of the official guides and past them. The one-path structure
stays; the chapters get deeper, and every new claim rides a verified code
snippet or a generated dump, never a retyped one.

Yardsticks: jax's Pallas quickstart and TPU guides, "Understanding jaxprs",
the StableHLO spec, XLA's fusion docs. The bar is: a reader who finishes a
chapter needs no external guide for that layer's daily work.

Rules: snippets verified in interpret mode or cross-lowered before commit;
provenance stated; museum and gym cross-linked where the concept bites.

## 0. Audit (2026-07-26)

Strong already: reading drills (corpus x-ray, decoder, oracle), rooflines,
pipeline model, algorithm/schedule split, flash derivation chain, Mosaic
reading (chapter 07 + GYM·05), real measured numbers, real captured errors.

Missing, by layer:
- pallas: memory spaces/SMEM, scratch, scalar prefetch, dimension_semantics,
  input_output_aliases, manual DMA + semaphores, pl.ds/edge rules,
  interpret-mode limits
- jaxpr: structural grammar (invars/constvars/outvars), transforms' shapes
  (grad/vmap/scan/cond as jaxpr rewrites), sub-jaxprs
- stablehlo: types, control flow ops, custom_call, versioning story
- xla: systematic optimized-HLO reading, fusion kinds, layouts, memory report
- tpu: SMEM/scalar core placement in the story
- ici: reduce_scatter / all_to_all / collective_permute vocabulary

## 1. Pallas full guide (chapter 06): first, biggest gap

- [x] BlockSpecs precisely: index maps, block-to-array math, whole-array
      specs, edge behavior
- [x] Memory spaces: VMEM, SMEM, ANY; what lives where and why
- [x] Scratch buffers: pltpu.VMEM scratch_shapes, when accumulators need one
- [x] dimension_semantics: parallel vs arbitrary, what it unlocks
- [x] Dynamic slicing: pl.ds inside kernels, the masked-edge pattern
- [x] Scalar prefetch: PrefetchScalarGridSpec, data-dependent schedules
      (the splash/RPA mechanism, taught small)
- [x] Manual DMA: make_async_copy, DMA semaphores, double-buffer by hand
- [x] input_output_aliases and multiple outputs
- [x] The debugging toolkit: interpret=True and its limits, debug=True,
      the museum as an error index
- [x] One language, three backends: the GPU paths (Triton backend and
      Mosaic GPU) at awareness level; what transfers (algorithm/schedule
      split, refs/blocks, the algebra, interpret mode) and what does not
      (sequential pipeline vs thousands of blocks, VMEM vs shared memory,
      auto-pipelining vs warp choreography); where the lowerings live in
      the jax repo; why this track goes deep on TPU
- [x] Every snippet verified (interpret or cross-lowered) by scripts/
      verify_guide_snippets.py, run in CI-adjacent local loop

## 2. jaxpr grammar guide (chapter 02)

- [x] The anatomy: invars, constvars, eqns, outvars; ClosedJaxpr
- [x] Transforms as jaxpr rewrites: one function shown under jit, grad,
      vmap, scan (generated dumps, corpus-style)
- [x] Sub-jaxprs: cond branches, scan bodies, remat; how to read nesting
- [x] Params: what lives in eqn.params and when it matters

## 3. StableHLO systematics (chapter 03)

- [x] The type system: tensor types, dynamism, tuples/tokens
- [x] Control flow: stablehlo.while / if / case, generated from lax
- [x] custom_call as the escape hatch (and how pallas_call appears)
- [x] Why "Stable": versioning, serialization, the portability contract

## 4. Reading optimized HLO (chapter 04)

- [x] Fusion kinds with real dumps: loop, input, output fusion
- [x] Layouts: {1,0} notation, when layout changes are inserted
- [x] The memory report: reading the vmem breakdown (museum exhibit 07 tie-in)
- [x] xla_dump flags: getting and navigating the artifacts
- [x] The same compiler, other machines: XLA:GPU at awareness level; fully
      open in openxla/xla but partly a dispatcher (cuBLAS, cuDNN, Triton
      codegen, NCCL) where XLA:TPU compiles everything itself; the fusion
      ceiling and the algorithm-failure lesson are backend-independent
      (flash exists on GPU for the same reason); every stack ends in a
      vendor black box, LLO/libtpu on TPU and ptxas/SASS on GPU, only the
      boundary moves (ties into chapter 07's last-readable-layer story)

## 5. Mosaic deepening (chapter 07), partially done 2026-07-26

- [x] Open/closed boundary, real module excerpt, GYM·05, LAB·1.4 cell
- [x] Layout/tiling vocabulary: what the (8,128) lattice looks like in
      types, sublanes and lanes named precisely

## 6. The machines + ICI (chapters 08, 09)

- [x] SMEM and the scalar core placed in the block diagram story
- [x] GPU architecture counterpart section in chapter 08: SMs, warps,
      tensor cores, the memory hierarchy (HBM, L2, shared, registers),
      and the two latency-hiding philosophies: TPU pipelines DMAs over
      a software-managed scratchpad, GPU oversubscribes warps. This is
      the setup for why kernels can win on both machines, and it makes
      the pipelining chapter a statement about machine models rather
      than a TPU quirk
- [x] Collective vocabulary: reduce_scatter, all_to_all, collective_permute,
      each as ring movements with cost formulas

Order: 1 → 2 → 3 → 4 → 6 → 5 → then 2nd passes. One commit per chapter guide.

## Detailed budgets and the competitive inventory (2026-07-26)

What the existing material out there contains, and what each of our
chapters builds past it. The standing differentiators nothing else has
as a package: one path with gates, hover-synced corpora generated never
retyped, drills with streaks, a museum of real captured errors, measured
numbers with provenance, and derivations written to be read.

Reference inventory:
- jax Pallas docs: quickstart (add/matmul, grid, BlockSpecs), TPU notes
  (memory spaces, dimension_semantics, scalar prefetch, pipelining,
  distributed), Mosaic GPU notes, design rationale. Static prose; no
  interactivity, no measured numbers, no error taxonomy.
- Understanding Jaxprs: the grammar reference, small static examples.
- Autodidax: internals by construction; teaches building, not reading.
- StableHLO spec (openxla.org): complete opset + versioning contract;
  a reference, not a tutorial.
- XLA docs: architecture overviews and op semantics; nearly nothing on
  reading real dumps.
- The scaling book: rooflines, TPU architecture, sharding math. Superb
  theory; stops before kernels and IRs.
- GPU-mode / Triton tutorials: the GPU-side analogues.
- Tokamax / maxtext / jax kernels: production code, no pedagogy.

Per-chapter build budgets (prose is new teaching words, all snippets
verified before commit):

- ch02 jaxpr (~2,500 words): the anatomy (invars, constvars, eqns,
  outvars, ClosedJaxpr); a transform gallery, one function shown under
  jit / grad / vmap / scan / remat / shard_map, generated by a new
  gen_transform_gallery.py; sub-jaxpr reading; eqn params reference.
  New gym drill: name the transform from the jaxpr. Past the official
  doc via the gallery and the drills.
- ch03 stablehlo (~2,500 words): the type system; control flow corpus
  (lax.scan and lax.cond lowered, stablehlo.while / if read line by
  line); custom_call as the escape hatch, showing pallas_call become
  tpu_custom_call, which bridges straight into chapter 06; the
  versioning contract and why the name says Stable. Corpus gains the
  control-flow programs.
- ch04 xla (~3,000 words): a reading guide for optimized HLO with a
  fusion taxonomy over real dumps; a before/after fusion x-ray (new
  instrument: the same program pre- and post-optimization, hover-
  synced); layouts; the vmem memory report walked line by line
  (museum exhibit 07 as the specimen); xla_dump navigation; the GPU
  dispatcher contrast section already planned.
- ch06 pallas (~6,000 words, the flagship): every checklist item in
  section 1 plus the GPU backends section; a new
  scripts/verify_guide_snippets.py so every snippet on the page runs
  in interpret mode in the local loop before commit; power-day case
  studies inline with their measured numbers; a possible LAB·1.5 for
  scratch, SMEM, and scalar prefetch hands-on. Strict superset of the
  official quickstart plus TPU notes, with receipts.
- ch07 mosaic (~800 words added): layout and tiling vocabulary,
  sublanes and lanes named precisely, reading layout attributes.
- ch08 the machines (~2,000 words): TPU deepening (SMEM, scalar core)
  plus the GPU architecture counterpart above; extend the memory
  hierarchy exhibit with the GPU column.
- ch09 ici (~1,500 words): the collective vocabulary with cost
  formulas; reduce_scatter as the reversed ring in the trace engine
  if the choreography generalizes cleanly.
- ch01 source (~600 words): closures, consts, and what tracing
  forbids. Lowest priority.

Total: roughly 19,000 words of new teaching prose, four generators,
two to three new instruments, one snippet-verification harness, and
two new drills, landing one chapter per commit.

## Appendix: the GPU IR ladder (for a future GPU wing, not this track)

jaxpr and StableHLO are shared with the TPU path and already drilled.
The GPU-specific drills, if a wing ever earns its place:
- HLO with the GPU backend: fusion kinds plus custom-call dispatch
  (cuBLAS, cuDNN, Triton gemm fusions); more dispatch table, less
  compiled program
- Triton IR (tt/ttg dialects) and the Mosaic GPU dialect: the GYM·05
  analogue; both print their modules, so the corpus pipeline transfers
- PTX primer + SASS awareness: PTX is open and documented (one layer
  deeper than TPU allows); SASS behind closed ptxas is the true box
- Error vocabulary: register pressure and occupancy via ptxas -v and
  Nsight Compute; GPUs run badly where TPUs refuse to compile


## The mastery matrix (2026-07-27)

The sure-shot path check: every subject has teaching, drilling, doing,
and a measured proof. Chapter numbers, gym stations, labs, and gates:

- jaxpr: ch02 guide + transform gallery · GYM 02/03/06 + the corpus ·
  LAB 2.1 · 276 source-pinned eqns in the corpus
- StableHLO: ch03 guide + control-flow corpus · GYM 01/04 · LAB 2.1 ·
  the custom_call bridge taught with the museum receipt
- XLA and optimized HLO: ch04 guide · GYM 03/07 · LAB 2.2 · gate 02
  closed at 1.2% with the fusion x-ray as the exhibit
- Pallas: ch06 flagship guide (all features, verified snippets) ·
  EX 02/07/09 + walks · LAB 1.1 to 1.4 + power day · gates 01 and 03
  passed on measured retunes and pinned differentials
- Mosaic: ch07 guide + boundary drawn exactly · GYM 05 · LAB 1.4's
  debug cell · the captured module corpus
- The machines + XProf: ch08 guide at ten sections (architecture both
  chips, capture, the three accounts, pane mastery, operator habits) ·
  EX 01/06 + GYM 08 · LAB 0.1 and LAB 2.3 · gates 00 and 02
- Kernels: stage 3 chapter + flash/causal/splash walks · EX 03/08 ·
  LAB 3.1 to 3.4 · gate 03 passed, power rows as the wins
- ICI and distributed: ch09 guide · EX 04/10/11 · LAB 4.1/4.2 verified
  off-chip · gate 04 awaits a multi-chip slice
- The capstone: reader-owned gate with the three criteria, runnable
  starting points, and the venues; tracked through mastery items

Known open edges, honestly: gate 04 hardware; no gym drill for the
distributed vocabulary yet (the stage instruments carry it); second
passes on guides as jax versions move.
