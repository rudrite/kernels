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

- [ ] R1 · frontend: tracing → lowering → compile → modes
- [ ] R2 · runtime seam: ComputationClient, both clients, plugin loading,
      transfers, threading
- [ ] R3 · PJRT + IFRT interfaces, C API/ABI, plugin ecosystem

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

- [ ] Phase R complete, notes reconciled against each other (the seam is
      described by R1 from above and R2 from within; they must agree)
- [ ] Phase 1 study loop run on this machine, artifacts kept locally
- [ ] Phase 2 arcs authored per the §P3 protocol (checks, voice, provenance,
      tests + build green, browser check, one commit per unit)
- [ ] CURRICULUM.md and §P5 grades updated with whatever ships
