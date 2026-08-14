# The torch_xla / PJRT boundary: working mastery

Status: planned (2026-08-14), research phase in flight. Purpose is double:
first, working-level command of the boundary for upcoming hands-on torch_xla
work; second, the site arcs that the study produces as a by-product. The
site half enters under the P5 standing rule, one telling per fact.

## What mastery means here

Able to answer, from source, for any tensor operation in a torch_xla program:
where it is right now (traced IR node, HLO, StableHLO, PjRtBuffer), what event
moves it to the next stage, and which interface call crosses each boundary.
Specifically:

- The frontend: lazy tensor tracing, the IR graph, graph cut points
  (mark_step / sync, early-sync forcers), lowering to HLO, the compilation
  cache and what invalidates it.
- The modes: lazy step-graph mode, eager mode (how it rides the same
  machinery), torch.compile with the openxla backend (dynamo capture vs lazy
  retrace), and when each wins.
- The seam: the ComputationClient interface and its two implementations
  (PjRt, Ifrt); which frontend action bottoms out in which client call.
- PJRT proper: PjRtClient / Device / MemorySpace / Buffer / LoadedExecutable /
  Compiler, the async future model, the C API and plugin ABI, what Compile
  accepts and what Execute returns.
- IFRT above it: Array + Sharding vs per-device Buffers, the pjrt_ifrt
  adapter (chapter 14's walk), the proxy.

## Phase R: source-grounded maps (in flight)

Three research passes over shallow clones of pytorch/xla and openxla/xla,
each producing a notes file (local scratch, not tracked) with every claim
cited to a path, plus a 15-file reading order:

- [x] R1 · frontend: tracing → lowering → compile → modes. Landed
      2026-08-14 against pytorch/xla 41398bf (still master's tip; the branch
      has been quiet ~4 months). Headline verdicts to reconcile in Phase 2:
      TorchTPU (announced 2026-04-22) is slated to replace PyTorch/XLA with
      an eager-first PrivateUse1 backend, leaving XLA as a torch.compile
      backend via StableHLO, so the lazy frontend is now the legacy design;
      eager mode is the same tracing machinery with per-op sync (same
      lowering, same cache), not a separate path; interior IR-node hashes
      exclude shapes while leaf data nodes fold them in; print and sync
      compile the same graph twice by design (force_ltc_data enters the
      hash); the dynamo path caches by graph hash with a hard failure on
      LRU eviction; the seam is crossed at 13 named ComputationClient
      call sites.
- [x] R2 · runtime seam: ComputationClient, both clients, plugin loading,
      transfers, threading. Landed 2026-08-14 against pytorch/xla 41398bf.
      Headline verdicts to reconcile in Phase 2: the IFRT client is compiled
      but switched off in runtime.cpp (a hard-coded false, not a flag), and
      SPMD-only with a dozen unimplemented methods; no in-tree GPU runtime
      remains, GPU means an out-of-tree plugin; buffer donation travels
      inside the HLO as buffer-donor annotations, never in ExecuteOptions;
      collectives lower into the graph and never cross the seam at runtime.
- [x] R3 · PJRT + IFRT interfaces, C API/ABI, plugin ecosystem. Landed
      2026-08-14 against openxla/xla a6c8e17. Headline verdicts to
      reconcile in Phase 2: Compile no longer loads (Compile returns
      PjRtExecutable, CompileAndLoad returns the loaded form); buffers moved
      from device-addressed to memory-space-addressed APIs (CopyToDevice is
      gone from C++); PJRT and IFRT now share one future type; the C API
      version gate enforces a stated 12-week forward window and StableHLO
      is serialized at min(framework, plugin) version; a minimum plugin is
      a PjRtClient subclass plus PJRT_Client_Create plus GetPjrtApi, and
      libtpu loads as an ordinary plugin; IFRT's array is an ArraySpec plus
      positionally-matched per-device buffers, and its README states the
      single-host-to-thousands rationale outright.

Pin the commits inspected; site quotes carry those commit ids, matching the
walk convention already in `site/src/data/walks/`.

## Phase 1: the study loop (personal, drives everything)

- [ ] Read R1-R3 reading orders against the clones, in seam-outward order:
      computation_client.h first, then PjRtComputationClient, then the
      frontend that calls it, then pjrt_client.h beneath it
- [ ] Trace one training step end to end on the CPU plugin: metrics report,
      IR dump, HLO dump, and the exact seam calls, reconciled against the R2
      call-flow table
- [ ] Repeat the trace in eager mode and under torch.compile; diff the three
      (what got captured, what recompiled, what hit the cache)
- [ ] Exercise the boundary directly: a small script driving
      ComputationClient calls; confirm the mock-plugin walk's C surface
      against a real plugin load
- [ ] Write up deltas against the site's current telling (chapter 10
      bridges, xla:pjrt, xla:ifrt, chapter 14 interfaces): what the study
      showed that the site does not yet teach

## Phase 2: the site arcs the study earns

Owning units, per the audit rule (deepen, never duplicate):

- `pt:bridges` (chapter 10): the deep arc. Lazy tensor engine, cut points
  and sync semantics, eager mode and its compile escape, the dynamo/openxla
  path, and the seam lesson that names the ComputationClient surface. This
  is the biggest gap today; the chapter tells the story at survey depth.
- `xla:pjrt` and `xla:ifrt`: reconcile anything the study contradicts or
  deepens; cross-link the seam lesson instead of retelling PJRT.
- Chapter 14 (`xla:interfaces`): candidate third walk, PjRtComputationClient
  function by function, completing the set (mock plugin below, IFRT adapter
  beside, the torch client above). Decide after R2 lands whether the walk
  earns its place or duplicates the seam lesson.
- Labs: a boundary-trace notebook (CPU plugin, metrics + dumps) if the
  Phase 1 trace proves capturable in notebook form.

## Work items

- [x] Phase R complete and reconciled 2026-08-14: the site-facing audit
      verified 24 claims and corrected 3 (chapter 1 CompileAndLoad naming,
      ReshardArrays not metadata-only, the array.h RTTI line); R1 and R2
      agree on the seam
- [x] Phase 1 encoded as LAB·P5 "The boundary trace" (labs/pytorch/): the
      three-mode trace with the seam table as its reading frame. The Colab
      TPU run and its paste-backs are the reader-driven half, still to run
- [x] Phase 2 complete 2026-08-14: the pt:bridges arc (five lessons with
      the first pt: lesson route), the xla reconcile pass applied, the
      chapter-14 third walk shipped (torch-client, nine steps over one
      round trip, coverage-checked verbatim), and LAB·P5 carded on the
      bridges chapter
- [x] §P5 grade recorded; CURRICULUM.md enumerates no lessons. Plan
      complete 2026-08-14 except the reader-driven Colab paste-backs
