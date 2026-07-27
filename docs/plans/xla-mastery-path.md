# The XLA mastery path

Third path on the site, at `/xla`: how a StableHLO program becomes a running
executable, and how the runtime stack above the compiler (PJRT, IFRT, McJAX,
Pathways) drives it. Built with the same production system as the JAX path:
a blueprint with fixed facts and verified artifacts first, prose written to
the house voice second, presentation and diagrams owned separately.

## The bar

- Facts are anchored to the public XLA source tree; every cited source path
  is verified to exist in a current checkout before it ships.
- Runnable artifacts are real: the pass pipeline shown is a real dump from a
  real compile on this machine, the optimized HLO is the compiler's own
  output, provenance stated.
- Readings are public only: openxla.org, the GitHub source tree, the
  Pathways and GSPMD papers, JAX runtime docs. A vitest enforces the
  required reading set, chapter structure, and the voice rules.
- Closed layers stay honestly closed: Pathways is taught from the paper and
  the in-tree IFRT proxy surface, and the chapter says so.

## The curriculum (fourteen chapters)

Part i, the compiler:

1. `pjrt`: the seam every framework talks to; the C ABI and plugins
2. `ingestion`: StableHLO in, internal HLO out; the multi-IR layering
3. `hlo`: the IR itself; modules, computations, dataflow, aliasing
4. `pipeline`: the pass pipeline, and reading `--xla_dump_to` output
5. `fusion`: the fusion policy, priority fusion, where fusion stops
6. `layout-memory`: layout assignment and buffer assignment
7. `spmd`: the SPMD partitioner; sharding propagation semantics
8. `collectives`: communicator semantics, async execution, host offload
9. `codegen`: CPU thunks, GPU emitters, the closed TPU floor
10. `autotuning`: the autotuning cache, cost analysis, profiling hooks

Part ii, the runtime stack above:

11. `ifrt`: the array and client abstraction above PJRT
12. `mcjax`: how stock JAX binds the stack; dispatch and the caches
13. `pathways`: single-controller execution; what swapping the backend means
14. `capstone`: a plugin skeleton, a pass traced end to end, or a fusion
    decision taught from real dumps

## Work items

- [x] Source checkout pulled; anchor paths verified against it
- [x] Verified artifacts captured (pass list, optimized HLO) with provenance
- [x] `site/src/data/xla/track.test.ts` failing first
- [x] Blueprint authored; prose written; assembled into
      `site/src/data/xla/track.ts`
- [x] Diagrams: the stack, the pipeline, the runtime ladder, the controllers
- [x] `/xla` landing + `/xla/<chapter>` pages; registry integration
      (progress, strip, search, home card)
- [x] Mastery ledger wired to the XLA gym stations
- [x] vitest green, build green, browser-checked

## Later phases

- XLA-specific drills (name the pass, predict the fusion) with auto rules
- Real-chip captures for the museum (OOM, layout) alongside XLA labs
- The PyTorch path
