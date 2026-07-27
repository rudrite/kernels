# The JAX mastery path

The site grows from one path to a family of paths. The TPU kernel track keeps
its routes and its identity untouched. JAX mastery becomes a sibling path at
`/jax/`, built in depth now. PyTorch mastery is reserved at `/pytorch/` and
built later; nothing ships there in this phase.

## Why a sibling, not a merge

The kernel track descends the stack toward hardware. JAX mastery moves across
the language: transformations, state, sharding, and the training loop. The two
share a progress model and a design language, not a spine. Forcing JAX
chapters into the descent would bend both curricula, so each path owns its
routes, its chapter order, and its landing page.

## Decisions

- Kernel-track routes (`/`, `/l/<id>`, `/s/<id>`, `/gym`, `/mistakes`,
  `/bench`) do not move. Existing localStorage keys keep their meaning.
- The JAX path lives at `/jax/` (landing) and `/jax/<chapter>` (twelve
  chapters). Chapter keys in the progress stores are `jax:<id>`.
- PyTorch is reserved at `/pytorch/`; the only mention this phase is a
  "reserved" line on surfaces that list paths.
- Progress stays browser-only. `ProgressKit` learns one thing: path scope.
  `data-path-progress` and `data-continue` read an optional path id
  (`kernels` when absent, so every existing page keeps working unchanged).
- The mastery ledger for JAX chapters ships in the JAX track module and is
  merged with the kernel ledger in one place (`data/mastery-index.ts`).
  JAX mastery items are manual only in this phase: the path has no gym
  drills or Colab labs yet, and the ledger does not pretend otherwise.
- Code snippets on JAX pages follow the site's honesty rule. Part i
  snippets run on CPU JAX before publishing. Snippets that need current
  sharding APIs or accelerators state their provenance in the caption and
  get verified in the Colab lab pass (follow-up phase).

## The curriculum (twelve chapters)

Part i, the model: how JAX thinks.

1. `arrays`: immutability, `.at`, dtype promotion, async dispatch
2. `tracing`: the trace, `make_jaxpr`, what tracing takes away
3. `jit`: the cache, static arguments, recompilation, AOT
4. `autodiff`: grad, vjp/jvp, `custom_vjp`, `stop_gradient`, remat
5. `vmap`: batching semantics, axis specs, per-sample gradients
6. `control-flow`: `cond`, `while_loop`, `scan` as the workhorse
7. `pytrees`: leaves and structure, `tree_map`, custom nodes
8. `random`: explicit keys, `split`, `fold_in`, keys under `vmap`

Part ii, the practice: state, scale, and the training run.

9. `state`: the functional state pattern, optax, Flax NNX, orbax
10. `sharding`: `Mesh`, `NamedSharding`, jit as GSPMD, `shard_map`
11. `performance`: honest benchmarking, profiling, recompilation traps
12. `training-run`: the capstone loop, everything composed end to end

## Work items

- [x] Plan written (this file)
- [x] `site/src/data/jax/track.test.ts`: failing first: structure,
      mastery coverage, prose bans, link shapes
- [x] `site/src/data/jax/track.ts`: chapters, prose, code, readings,
      work blocks, mastery ledger
- [x] Part i snippets verified on CPU JAX; failures fixed before landing
- [x] `site/src/data/mastery-index.ts`: merged ledger
- [x] `ProgressKit` path scoping (visited regex, scoped progress/continue)
- [x] `PathStrip` renders the JAX strip on `/jax/*` pages
- [x] `MasteryWork` reads the merged ledger
- [x] `site/src/pages/jax/index.astro`: the landing
- [x] `site/src/pages/jax/[id].astro`: the chapter template
- [x] `Base.astro` nav gains the JAX link; search index gains JAX entries
- [x] Home page lists the paths (kernels live, JAX live, PyTorch reserved)
- [x] `vitest run` green, `astro build` green, output read

## Later phases (not this one)

- JAX gym drills (pytree structure oracle, axis-spec drill, recompile
  spotter) and auto rules for them
- Colab labs per chapter with mark-as-run wiring
- Instruments (trace explorer, sharding visualizer)
- The PyTorch mastery path at `/pytorch/`
