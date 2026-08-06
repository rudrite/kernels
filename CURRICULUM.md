# TPU Kernel Curriculum

Purpose: take one engineer (expert in the JAX distributed runtime: sharding, collectives, GSPMD, XProf; new to single-chip kernel land) from zero Pallas to shipping a production-grade TPU kernel upstream, in 14 weeks part-time. The bar is mastery, not completion: by the end, any production Pallas kernel should read like prose. Every stage ends with a verifiable checkpoint, and every stage's material is delivered on kernels.rudrite.com: the site is the course. Do not advance a stage until its checkpoint passes.

Course shape on the site: each chapter is a unit carrying an ordered run of lessons, each ending in a check, with the drills, walks, labs, and gates paced by the mastery ledger.

Hardware ladder: `interpret=True` on any machine (Stages 1-3 logic), Colab/Kaggle v5e (Stages 1-3 timings), TRC or internal capacity for a v5e-8+ slice (Stages 4-5).

Prior-knowledge credit: sharding, meshes, collectives semantics, XProf navigation, and jit/pytrees are assumed known and are not re-taught. The curriculum teaches what happens below the jaxpr.

---

## Stage 0: The machine (week 1)

Goal: given an op and its shapes, predict from first principles whether it is compute-bound or memory-bound on a given TPU generation, and estimate its ceiling latency.

### Read
- [JAX scaling book](https://jax-ml.github.io/scaling-book/): the "All about TPUs" and "Rooflines" chapters in full. This is the backbone text; take the chip-constant tables (HBM bandwidth, FLOP/s, VMEM size, ICI bandwidth per generation) from here rather than from memory or blog posts.
- [Google Cloud TPU system architecture docs](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm): skim for the v5e/v6e specifics.

### Internalize
- The compute units: MXU (systolic matmul array) vs VPU (elementwise vector unit), and which ops land on which.
- The memory hierarchy: HBM, VMEM, SMEM, registers; why VMEM residency is the whole game.
- Arithmetic intensity and the roofline: FLOPs / bytes-moved against the chip's FLOPs-per-byte ratio.
- The tiling lattice: why the last two dims want multiples of (8, 128), and how dtype changes it (bf16 packs (16, 128)).
- ICI vs DCN, and what a "slice" physically is. This connects to collectives knowledge you already have.

### Exercises
1. For five ops (large matmul, skinny matmul, elementwise add, softmax over a long axis, embedding lookup), compute arithmetic intensity by hand and classify compute- vs memory-bound per chip generation.
2. Estimate the step-time floor for a 7B-parameter forward pass at batch 8, seq 4096 on one v5e chip from the constants alone.
3. Open an XProf trace of a small model on Colab TPU and reconcile three of your estimates against measured op times.

### Checkpoint (verify)
Latency predictions for the five ops land within 2x of measured, and you can explain every miss.

### Delivered on the site
Chapter 08 and the stage 0 instruments: the roofline playground and the memory hierarchy at real widths.

---

## Stage 1: Pallas fundamentals (weeks 2-4)

Goal: write, benchmark, and profile correct single-chip Pallas kernels; read production kernels fluently.

### Week 2: Mechanics
- Read: [Pallas quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html), then the [Pallas TPU pages](https://docs.jax.dev/en/latest/pallas/tpu/) (grids and BlockSpecs, pipelining, TPU details).
- Concepts: `pallas_call`, `BlockSpec` index maps, grid iteration order, refs vs arrays, `interpret=True`, TPU grid steps as a software pipeline (not GPU-style parallel blocks), dimension semantics (`parallel` vs `arbitrary`).
- Exercises, in order: (1) elementwise add and scale; (2) transpose; (3) tiled matmul with a VMEM accumulator carried across the K grid dimension.

### Week 3: Fusion and reductions
- Concepts: row reductions inside a block, fused chains (matmul + bias + activation), masked epilogues for shapes that do not divide the block, dtype staging, SMEM for scalars.
- Exercises: (4) fused row-softmax (single load per block, no HBM intermediate); (5) LayerNorm forward; (6) matmul + GELU fused, compared against the XLA-fused chain.

### Week 4: Pipelining, measurement, profiling
- Concepts: automatic double-buffering from the grid pipeline, when to size blocks for overlap, `pl.when`, cost of transcendentals on VPU.
- Benchmarking discipline: warmup, `block_until_ready`, median-of-N, fixed clocks assumption stated; record chip generation with every number. Prefer JAXBench's harness and Tokamax's benchmarking utilities over hand-rolled timing loops; write thin glue, not infrastructure.
- Profile your own kernels in XProf; find the pipeline bubbles.
- Read as literature (line by line, annotating "algorithm or schedule?" in the margin): one Tokamax kernel ([openxla/tokamax](https://github.com/openxla/tokamax)) and the [JAX Pallas ops directory](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu).

### Checkpoint (verify)
- Tiled matmul within 15% of XLA's `jnp.dot` for 4096x4096x4096 bf16 on v5e.
- Fused softmax beats the unfused XLA chain at rows of 32k+.
- You can state, for every line of your matmul kernel, whether it is algorithm or schedule.

### Delivered on the site
The stage 1 chapter, the chapter 06 full guide, and the guided walks, with the algorithm/schedule split color-coded throughout.

---

## Stage 2: The IR stack (week 5, interleave with late Stage 1)

Goal: read every representation your code passes through, and find XLA's fusion decisions and their limits with your own eyes.

### Ladder
1. jaxpr: `jax.make_jaxpr` on your Stage 1 reference functions; eqns, invars, closed jaxprs.
2. StableHLO: `jax.jit(f).lower(args).as_text()`; map each jaxpr eqn to its StableHLO op; learn `dot_general` dimension_numbers cold (reading any tensor IR fluently lives or dies on this).
3. Optimized HLO: `.compile().as_text()`; identify fusion boundaries; find where naive attention materializes the S matrix to HBM. That spill is the reason flash attention exists; see it yourself, not in a blog post.
4. MLIR, conceptual minimum only: what a dialect is, ops/regions/attributes; StableHLO and Mosaic are dialects; skim the [StableHLO spec](https://openxla.org/stablehlo/spec) op list in one sitting. No MLIR is written anywhere in this track.

### Exercises
1. Trace 5-line naive attention; produce a table: JAX source line, jaxpr eqn, StableHLO op, fused-HLO location.
2. Annotate the optimized HLO dump of naive attention at seq 8192: mark every fusion, mark the HBM spill, estimate its bytes, and confirm the estimate in XProf.
3. Pull one real operator from MaxText (the production model library) and repeat the trace on it; catalog what the production jaxpr contains that your toy version does not (sharding constraints, remat markers, dtype casts).

### Checkpoint (verify)
The annotated dump exists and your spill-size estimate matches the profiler within 20%.

### Delivered on the site
The chapter 02 to 04 and 07 guides plus the x-ray instruments: one program held open at every layer with hover sync. Profiling lives here too: LAB·2.3 captures and reads a real trace, GYM·08 renders it op by op, and chapter 08's guide teaches the capture, the three cost accounts, and the timeline reading.

---

## Stage 3: The priesthood kernels (weeks 6-8)

Goal: own the two axes that make the famous kernels hard: algorithmic restructuring and data-dependent iteration.

### Week 6: Flash attention, derived then built
- On paper, from a blank page: derive online softmax. Start from the two-pass definition, introduce the running (m, l) pair, prove the rescaling identity, then extend to the (m, l, acc) triple with the second matmul folded in. Then prove the combine operation is associative and commutative (this is the softmax monoid: the algebraic fact that makes blocked, streaming, and parallel attention correct; the derivation lives on the site as the stage 3 chapter and walks).
- Implement single-chip flash attention forward in Pallas from your own derivation, without looking at any reference implementation.
- Only then, diff yours against Tokamax's and [Splash attention in the JAX repo](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention). Every difference is a lesson: catalog each one as algorithm, schedule, or feature.
- Also run Tokamax's own selection layer (`implementation=None`) across a shape sweep and record which implementation it picks per shape; understanding what the selector knows, and where it is thin, is part of mastering the ecosystem you build on.

### Week 7: Backward, and Splash
- Derive the flash backward on paper: why the forward saves (m, l) or the logsumexp, the D = rowsum(dO * O) identity, blockwise recomputation of P, and the separate dQ vs dK/dV loop structures.
- Wire your forward + a reference backward with `jax.custom_vjp`; differential-test values and grads against the XLA reference.
- Read Splash's mask machinery: how a block-sparse mask summary becomes the grid, and what scalar prefetch is doing. Implement a toy: attention that skips fully-masked blocks given a static block mask, and show the speedup scales with sparsity.

### Week 8: Ragged, paged, MoE
- Read the [Ragged Paged Attention paper](https://arxiv.org/pdf/2604.15464), then its kernel source in [tpu-inference](https://github.com/vllm-project/tpu-inference). Focus: how the page table and sequence lengths (data) become loop structure via scalar prefetch.
- Read one MoE dispatch/grouped-matmul kernel (Tokamax or megablocks lineage). Map it to the same pattern: tokens grouped by expert is ragged iteration.
- Implement a toy ragged kernel: rowwise softmax over variable-length rows given a lengths array, no wasted work past each row's end.

### Checkpoint (verify)
- Your flash forward is within 1.3x of the reference implementations at seq 8192 on v5e.
- Differential tests pass: 1e-3 forward, 1e-2 grads, bf16.
- You can explain every line of Splash attention and state which of the two axes each part serves.

### Delivered on the site
The stage 3 chapter, the streaming-attention instruments, and the flash, causal, and splash walks; the power bench carries the measured wins.

---

## Stage 4: Distributed Pallas (weeks 9-10)

Goal: ground your existing collectives knowledge in the mechanism: remote DMA + semaphores.

### Read
- [Distributed Computing in Pallas for TPUs](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html): the RDMA model, `make_async_remote_copy`, send/recv semaphores, barriers, ring topologies from mesh slices.

### Exercises
1. On a v5e-8 slice: implement ring all-gather in Pallas. Validate output against `jax.lax.all_gather`.
2. Profile it: show the per-hop transfer overlapping the copy of the previous chunk. Then break it on purpose (reorder a semaphore wait) under interpret mode to see what a deadlock looks like.
3. Compose Stage 3 with the ring: a toy ring attention step where KV blocks arrive by remote DMA instead of HBM DMA, or a close reading of a reference if the slice budget is tight. Observe that the online-softmax algebra is indifferent to where a block came from: that indifference is exactly what makes ring attention work.

### Checkpoint (verify)
Ring all-gather matches the collective bitwise and the profile shows compute/comm overlap.

### Delivered on the site
The chapter 09 guide and the ring instruments: the same schedule with longer arrows, cost formulas worked to link numbers.

---

## Stage 5: Capstone, a kernel shipped upstream (weeks 11-14)

Goal: prove the previous ten weeks against production standards by building one kernel end to end and putting it in front of the community.

Pick one, sized to close a real gap:
- An attention variant with no fast public TPU kernel (for example, a mask family Splash does not cover), forward and backward.
- A block-size autotuning pass for an existing Tokamax kernel, with measured wins across a shape sweep.
- A JAXBench entry: an optimized kernel for one of the operators where published baselines trail the expert bound.

Requirements, non-negotiable:
- Forward and backward (`custom_vjp`), differential-tested at 1e-3 forward and 1e-2 grads in bf16, corner cases enumerated.
- Benchmarked against both the XLA floor and the closest hand-tuned ceiling, on stated hardware, recorded in `bench/`.
- Sent upstream: a PR to Tokamax or JAXBench (or a standalone release plus write-up if upstream declines), with the review discussion linked from the site.

### Checkpoint (verify)
The kernel is correct under the differential suite, beats XLA decisively at its target shapes, lands within striking distance of the hand-tuned ceiling, and the upstream submission exists in public.

### Delivered in public
The upstream PR itself: derivation, schedule decisions with measured consequences, the benchmark record, and the review thread. The submission is the finale; the site links it.

---

## Operating rules

1. Order within a stage is prescriptive; derivations come before implementations, implementations before reading references.
2. Every benchmark number recorded in this repo states: chip generation, dtype, shapes, and measurement method.
3. The site delivers each stage's teaching; work the stage through its chapters, guides, drills, and labs in order.
4. Weekly time expectation: 8-12 focused hours. If a stage overruns by more than a week, slow down; never cut scope from the checkpoint.
5. Internal resources (Mosaic/Pallas docs, Tokamax design notes, the JAXBench context pack) supersede the public links above where available.
6. Leverage the open ecosystem before building anything. Tokamax is the performance ceiling and the tooling shelf; JAXBench is the benchmark harness and workload source; MaxText is where real operators come from; the JAX Pallas ops and tpu-inference kernels are the reference literature. Custom infrastructure is a last resort; thin glue over these is the default.
7. Upstream presence starts in week 2, not week 11. When source reading turns up a bug, a doc gap, or a missing benchmark, file the issue or the small PR the same week. By the capstone, the maintainers reviewing your kernel should already know your name.
