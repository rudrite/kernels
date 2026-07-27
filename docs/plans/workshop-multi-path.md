# The workshop goes multi-path

The gym and the museum were built kernel-first. With three live paths they
become families: a hub per facility, one sub-page per path, and every
station or exhibit assigned to the path whose skill it drills.

## Decisions

- `/gym` and `/mistakes` become hubs (cards per path, live progress where it
  exists). Kernel content moves to `/gym/kernels` and `/mistakes/kernels`
  unchanged; anchors keep their ids, so old deep links land one hop away.
- Stations are assigned by the level of the skill, not duplicated blindly:
  jax gets the shape oracle, name-the-transform, and the corpus x-ray; xla
  gets spot-the-decision, the fusion x-ray, and the timeline x-ray. Streak
  stores stay shared: the skill is the same skill wherever it is drilled.
- The JAX and XLA mastery ledgers gain auto-checked streak items now that
  their gym pages exist; the kernel ledger's hrefs move to `/gym/kernels`.
- New museum wings hold only captured failures: every jax exhibit ran and
  failed on this machine (verbatim errors, fixes proven by running them);
  the xla wing opens with the SPMD-seam failures capturable on host
  devices, and says plainly that compile-side exhibits need real chips.
- PyTorch shows as reserved on both hubs, consistent with the home page.

## Work items

- [x] `/gym` hub; kernel gym at `/gym/kernels`; jax and xla gym pages
- [x] `/mistakes` hub; kernel museum at `/mistakes/kernels`
- [x] jax + xla mistakes captured by script, verbatim, fixes proven
- [x] Every internal `/gym#` and `/mistakes#` reference updated
- [x] JAX/XLA mastery auto rules wired; jax track test updated to allow them
- [x] Search index covers the new pages and wings
- [x] vitest green, build green, browser-checked
