# kernels

Rudrite's accelerator wing: four public, lab-driven mastery courses for the stack that runs machine learning on TPUs, built while learning it, in the open. Numbers count only when a real chip measured them, and the ones that turned out to be wrong stay published with their corrections.

Site: **live** at [kernels.rudrite.com](https://kernels.rudrite.com) · Cloudflare Pages, deploys on every push to main

## The four paths

| path | what it teaches | chapters |
|---|---|---|
| [kernels](https://kernels.rudrite.com/) | down the stack to the machine: Pallas, the IR layers, flash attention derived, distributed kernels | 15 |
| [jax](https://kernels.rudrite.com/jax) | across the language: tracing, the jit cache, autodiff, vmap, pytrees, sharding, the training run | 12 |
| [xla](https://kernels.rudrite.com/xla) | through the compiler: PJRT, HLO, the pass pipeline, fusion, SPMD, codegen, then IFRT and Pathways | 14 |
| [pytorch](https://kernels.rudrite.com/pytorch) | the eager world and its bridges: storage and strides, the tape, dynamo, then torch on TPU | 12 |

Each chapter carries a diagram, its assigned readings, and tracked mastery work. The JAX, XLA, and PyTorch paths cite every page of their framework's official tutorial series and go past it; a test enforces that.

## The workshop

- **The gym** ([/gym](https://kernels.rudrite.com/gym)): one floor per path, 13 drill stations. Every answer is computed from generated dumps or real runs, never written by hand. Streaks live in your browser.
- **The museum** ([/mistakes](https://kernels.rudrite.com/mistakes)): one wing per path, 27 exhibits. Each is a failure that was reproduced, with its verbatim error and a fix that was run.
- **Instruments**: 18 live surfaces, from the roofline playground to the jit cache key, the view explorer, and the pass pipeline stepped from a real dump.
- **Lessons**: 50 lessons nested inside the chapters that own them, each ending in a check and carrying mastery work that rolls up into the chapter's ledger: the machine, the profiler, the fabric, and the Pallas language in the kernel path; the compiler internals in the XLA path, down to the symbol a PJRT plugin exports.

## The repo

- `CURRICULUM.md`: the kernel track's 14-week plan. Six stages, each gated by a measurable checkpoint.
- `labs/`: 30 runnable notebooks. Open directly in Colab from GitHub, then save a copy to keep your work. The 18 kernel labs run off-chip (interpret mode, simulated devices); the 12 path labs cover JAX, XLA, and PyTorch, and the four that need real hardware say so.
- `bench/`: 39 records, every number the site shows with its provenance (chip, dtype, shapes, date, commit). One row reads RETRACTED, because a number published and later disproved belongs in the same table as the ones that held.
- `docs/plans/`: how each path was built, and what remains.
- `site/`: kernels.rudrite.com, the presentation layer. The repo stands alone without it.

Status: all four paths are live with labs, drills, exhibits, and instruments. Kernel-track gates 0 through 3 have passed on a v6e-1; gate 4 awaits a slice. Every lab has been executed end to end, ten of them here and the TPU ones on a Colab v6e.
