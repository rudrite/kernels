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

- [ ] BlockSpecs precisely: index maps, block-to-array math, whole-array
      specs, edge behavior
- [ ] Memory spaces: VMEM, SMEM, ANY; what lives where and why
- [ ] Scratch buffers: pltpu.VMEM scratch_shapes, when accumulators need one
- [ ] dimension_semantics: parallel vs arbitrary, what it unlocks
- [ ] Dynamic slicing: pl.ds inside kernels, the masked-edge pattern
- [ ] Scalar prefetch: PrefetchScalarGridSpec, data-dependent schedules
      (the splash/RPA mechanism, taught small)
- [ ] Manual DMA: make_async_copy, DMA semaphores, double-buffer by hand
- [ ] input_output_aliases and multiple outputs
- [ ] The debugging toolkit: interpret=True and its limits, debug=True,
      the museum as an error index
- [ ] One language, three backends: the GPU paths (Triton backend and
      Mosaic GPU) at awareness level; what transfers (algorithm/schedule
      split, refs/blocks, the algebra, interpret mode) and what does not
      (sequential pipeline vs thousands of blocks, VMEM vs shared memory,
      auto-pipelining vs warp choreography); where the lowerings live in
      the jax repo; why this track goes deep on TPU
- [ ] Every snippet verified (interpret or cross-lowered) by scripts/
      verify_guide_snippets.py, run in CI-adjacent local loop

## 2. jaxpr grammar guide (chapter 02)

- [ ] The anatomy: invars, constvars, eqns, outvars; ClosedJaxpr
- [ ] Transforms as jaxpr rewrites: one function shown under jit, grad,
      vmap, scan (generated dumps, corpus-style)
- [ ] Sub-jaxprs: cond branches, scan bodies, remat; how to read nesting
- [ ] Params: what lives in eqn.params and when it matters

## 3. StableHLO systematics (chapter 03)

- [ ] The type system: tensor types, dynamism, tuples/tokens
- [ ] Control flow: stablehlo.while / if / case, generated from lax
- [ ] custom_call as the escape hatch (and how pallas_call appears)
- [ ] Why "Stable": versioning, serialization, the portability contract

## 4. Reading optimized HLO (chapter 04)

- [ ] Fusion kinds with real dumps: loop, input, output fusion
- [ ] Layouts: {1,0} notation, when layout changes are inserted
- [ ] The memory report: reading the vmem breakdown (museum exhibit 07 tie-in)
- [ ] xla_dump flags: getting and navigating the artifacts

## 5. Mosaic deepening (chapter 07), partially done 2026-07-26

- [x] Open/closed boundary, real module excerpt, GYM·05, LAB·1.4 cell
- [ ] Layout/tiling vocabulary: what the (8,128) lattice looks like in
      types, sublanes and lanes named precisely

## 6. Chip + ICI vocabulary (chapters 08, 09)

- [ ] SMEM and the scalar core placed in the block diagram story
- [ ] Collective vocabulary: reduce_scatter, all_to_all, collective_permute,
      each as ring movements with cost formulas

Order: 1 → 2 → 3 → 4 → 5 → 6. One commit per chapter guide.
