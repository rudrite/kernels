# Plan: from course skeleton to mastery-grade teaching

The bar, in the founder's words: by the end, read any jaxpr or StableHLO file fluently; kernel building intuitive; full Pallas mastery. The current site is a coherent course skeleton with instruments. Fluency does not come from more prose; it comes from drills over a broad corpus, interactive manipulation of every concept, and annotated exposure to production code. That is what this plan builds, in five phases, each shippable alone.

Principles carried from the existing system: instruments never fabricate (every dump generated, every drill answer computed), notebooks and generators stay the single sources of truth, quantized motion, copper = algorithm / steel = schedule everywhere, no backend (progress in localStorage).

## Phase 1: the reading gym (targets "read any jaxpr / any StableHLO")

Fluency is corpus breadth times active recall. Build both:

- **The corpus generator** (`scripts/gen-ir-corpus.py`): 25-30 real programs dumped to JSON at authoring time with jax pinned: norms, GELU chains, einsum variants, attention (MHA/GQA/MQA), scans, remat, custom_vjp pairs, sharded programs under shard_map, MoE dispatch. Every dump paired: source, jaxpr, StableHLO. Nothing hand-typed, ever.
- **The x-ray, generalized**: EX·05 gains a program selector over the corpus. Hover-sync mappings generated mechanically (jaxpr eqn ↔ StableHLO op by variable lineage) instead of hand-authored.
- **Drill islands** (predict-then-reveal, one tap, streaks tracked locally):
  - `dot_general` decoder: shown real dimension_numbers, pick contracting/batch axes; 10-in-a-row unlocks the next drill.
  - Shape oracle: shown an eqn with inputs, type the output shape before reveal.
  - Spot-the-decision: find the upcast / the broadcast / the remat marker in an unfamiliar dump, against the clock.
- **Op reference cards**: hover any op name anywhere on the site for its one-paragraph card (generated set covering every op the corpus uses).

Acceptance: a reader who finishes the gym can open a MaxText operator dump they have never seen and narrate it. Measured by the capstone drill: an unseen program, full narration checklist.

## Phase 2: the Pallas mastery system (targets "kernel building intuitive")

- **BlockSpec sandbox**: edit block shapes and index maps in a form; live grid animation, VMEM budget meter, (8, 128) lattice check with the exact Mosaic-style error text on violation. This is the legality arithmetic as a toy, and it makes schedule intuition manipulable instead of narrated.
- **CodeWalk island**: step-through annotated code (line-range highlighting plus margin notes per step, Code-Hike style but Astro-native over Shiki output). Every kernel in the track gets a walk: the six stage-1 kernels, flash forward, the backward, causal-skip.
- **Trace-to-code sync**: click a choreographer frame, the kernel line responsible lights; click a line, the frames it produces play. One mapping file per kernel, generated from the trace generators.
- **The mistake museum**: the eight failures every Pallas learner hits (lattice violation, VMEM overflow, wrong index map, accumulator on the wrong grid axis, interpret-vs-compiled drift, dynamic-slice pitfalls, fori_loop carry shape mismatch, masked-edge off-by-one), each as failing snippet + real error text + fix + why. Errors reproduced for real, captured once, checked in.

Acceptance: a reader can go from a blank cell to a correct novel fused kernel (given the math) without consulting references, using only what the site taught. The stage 1 gate criteria already measure the performance half.

## Phase 3: the production-kernel corpus (breadth for intuition)

Annotated reads, as CodeWalks, of the kernels that define the craft: Tokamax flash attention, Splash's mask machinery and scalar prefetch, ragged paged attention (paper + kernel), a MoE grouped matmul, one quantized kernel. Each walk marks every line algorithm or schedule, and ends with the diff table against the track's own build of the same idea. Licensing: excerpts within fair bounds with links, or walks driven off the reader's own checkout via line anchors.

## Phase 4: the assessment spine (makes the path self-verifying)

- Per-layer and per-stage "can you" checklists with local progress; the spine and map render personal state alongside the public gate state.
- Stage entry checks: stage 2 opens its gate panel only after the reading-gym streaks; stage 3 after the sandbox produces a legal nontrivial schedule. Soft gates (skippable, stated), because the site has no accounts and honesty beats enforcement.
- The bench compare exhibit: paste a lab's results blob, see your chip plotted against the published records.

## Phase 5: distributed depth

- Mesh visualizer: chips, rings, per-link bandwidth from chips.json; scrub a collective hop by hop.
- Semaphore timeline: the ring all-gather's sends/waits as a timeline; a toggle reorders one wait and steps into the deadlock, visually.
- Ring attention CodeWalk synced to EX·04.

## Status (2026-07-26)

Built and live: P1 gym (corpus of 17 programs, dot decoder, shape oracle, corpus reader, 33 op cards), P2 sandbox + four CodeWalks + the mistake museum (six real captured errors), P4 checklists + bench compare, P5 semaphore timeline. Remaining: x-ray generalized over the corpus with mechanical mappings, trace-to-code sync, the production-kernel walks (P3), the mesh visualizer, spot-the-decision drill, soft stage-entry checks.

## Sequencing and estimates

P1 first: it targets the stated bar most directly and its corpus generator underpins P2's walks and P3. Rough solo-effort: P1 2-3 weeks, P2 3-4, P3 2 (content-heavy), P4 1, P5 2. Each phase lands as its own set of commits with the loop discipline already in use (build, screenshot-read, fix, commit).

Layer pages and stage chapters keep growing alongside (each layer page gains exercises and a second worked dump as its drills arrive); prose grows where a drill needs setup, never as filler.
