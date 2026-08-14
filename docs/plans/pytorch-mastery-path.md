# The PyTorch mastery path

Fourth path, at `/pytorch`: the eager world and its compilers. Same
production system as the JAX and XLA paths: blueprint with locally verified
snippets first, prose to the house voice second, presentation owned
separately. This phase also ships the PyTorch museum wing (captured), the
PyTorch gym floor with the path's first original station, and an elevation
pass adding diagrams across all three new paths.

## The identity

The kernel path descends the stack, the JAX path moves across the language,
the XLA path goes through the compiler. The PyTorch path is the other
philosophy: state is the model, the tape is dynamic, and compilation is
opt-in. Chapters teach it on its own terms and name the contrasts (mutation
vs threading, guard-and-recapture vs trace-once) instead of pretending the
frameworks agree.

## The curriculum (twelve chapters)

Part i, the model:

1. `tensors`: storage, strides, views, in-place semantics
2. `autograd`: the dynamic tape, grad_fn, leaves, custom Functions
3. `modules`: nn.Module state, parameters and buffers, state_dict
4. `the-loop`: optimizer, schedulers, checkpoint and resume
5. `data`: Dataset, DataLoader, workers, the input pipeline
6. `dynamo`: torch.compile capture, guards, graph breaks
7. `graphs`: aot_autograd, ATen and FX, torch.export

Part ii, the practice, aimed at TPU (this site's home ground):

8. `distributed`: c10d, DDP, FSDP, the device mesh; the vocabulary the
   TPU chapters contrast against
9. `performance`: the profiler, AMP, memory, honest benchmarks
10. `bridges`: how torch reaches TPU; torch_xla's lazy tensors and
    torchax's torch-ops-on-jax, both converging on StableHLO and the
    XLA path's compiler
11. `tpu-practice`: SPMD sharding for torch models, profiling on TPU,
    Pallas kernels called from torch, the serving stack (tpu-inference)
12. `training-run`: the capstone loop on a TPU via the bridge, resumable

## The bar (same as before)

- Snippets verified on this machine (torch 2.2.2, CPU, the last Intel-mac
  wheel); printed values are real; where current APIs differ, captions say
  so. Superset rule enforced by test over the official tutorial series.
- Museum wing: only captured failures, errors verbatim, fixes proven.
- Gym floor: the stride oracle, a new station whose corpus is generated
  from live torch runs; plus a real distributed and compile drill later.

## Work items

- [x] Official tutorial URLs verified; torch 2.2.2 probed
- [x] `site/src/data/pytorch/track.test.ts` failing first
- [x] Snippets + 6 museum exhibits + stride-oracle corpus captured
- [x] Blueprint authored; prose written; assembled into
      `site/src/data/pytorch/track.ts`
- [x] Diagrams: the tape, the compile stack, DDP vs FSDP, tensor/storage
- [x] Elevation: new diagrams for JAX and XLA chapters that lacked them
- [x] `/pytorch` landing + chapter pages; registry integration
- [x] `/gym/pytorch` (stride oracle live) + `/mistakes/pytorch`; hub and
      home cards flip from reserved to open
- [x] Mastery streak items wired to the stride oracle
- [x] vitest green, build green, browser-checked

## The labs-and-stations phase (same day)

- [x] Eleven lab notebooks authored (labs/jax, labs/xla, labs/pytorch),
      lab cards on their chapters, mark-as-run wired as labs auto-rules
- [x] Guard-or-break station (GYM·10): eight scenarios verdicted by
      dynamo's own counters, streak wired into the dynamo chapter
- [x] Name-the-collective station (GYM·11): five collectives captured
      from a real world-of-4 gloo run, streak wired into the pytorch
      distributed chapter and the xla collectives chapter
- [x] Real-chip captures closed 2026-08-14, all from one v6e-1 session set:
      LAB·X2's museum exhibit reproduced exactly (same byte counts as the
      2026-07-27 capture), LAB·P4's capstone numbers landed (kill at step 99
      loss 0.01456, first resumed step 0.01437, curve continuous) and now
      ship as the lab's reference block, and LAB·P5 gained its full
      reference run the day it was authored
