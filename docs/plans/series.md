# The series shelf

The four paths teach in ordered chapters, and a chapter is one page. Depth has nowhere to
live: a subject like the Pallas language design, XLA's internals at plugin-writing level,
or the hardware itself needs an ordered run of full pages, not more paragraphs in a
chapter. The founder asked for exactly that (Pallas in depth, TPU and GPU architecture
with animations, XLA down to writing a libtpu plugin), and for the site to be
restructured so the shape generalizes.

## The structural decision

One new URL family, `/series/<series>/<page>`, with an index per series and a catalog at
`/series`. Not a new top level per series, for two reasons the survey made concrete:

- Chrome registration is manual in six places (nav, OG slug, PathStrip, ProgressKit,
  mastery-index, search-index). A single `series` prefix is registered once; every later
  series is data only.
- `/xla` and `/pallas`-adjacent names collide or confuse against the existing path URLs.
  A series deepens a path; it does not compete with it. Every series page names the path
  chapters it goes past and links back.

The site already contains the primitive four times: `lib/path.ts` plus `PathPager` for
the kernel path, and three inline byte-identical copies in the sibling `[id].astro` pages
and `track.ts` files. The new `lib/series.ts` + `SeriesPager.astro` implement the shape
once for series. Migrating the four paths onto it is deliberate follow-up debt, tracked
here, not done in this change; the paths keep working untouched.

## Data model

`site/src/data/series/<id>.ts`, JSON-shaped TS like the sibling tracks:

- `Series { id, title, lede, deepens: {label, href}[], pages: SeriesPage[] }`
- `SeriesPage { id, num, title, lede, goal, sections, readings }`
- `sections` reuse the house grammar: `{ h, ps }` with `'>> '` pull quotes, optional
  `code`, optional `diagram` (id into a per-series diagrams registry), optional
  `exhibit` (id into the instrument roster, rendered with provenance like EX pages).

Registered in `site/src/data/series/index.ts` (the one list, like `lib/path.ts`).
Mastery keys claim the `sr:` prefix: `sr:<series>:<page>`, merged into `mastery-index`.

## Animations

House policy holds: a diagram is a figure; motion is an instrument. The hardware series
ships new frame-stepped React islands in the Choreographer idiom (quantized steps, no
easing, autoplay opt-in), each with a real reduced-motion guard
(`matchMedia('(prefers-reduced-motion: reduce)')`), which Choreographer's comment
promises and its code omits; that gap gets fixed in the same idiom, separate commit.

Candidates, trimmed to what earns its place: a systolic-array stepper (operands
streaming through the MXU grid, partial sums on the wavefront), a torus-hop animator
(2D/3D, wraparound, per-generation link speeds, computed hop cost), an SM issue stepper
(subpartitions, warps, tensor core tiles), a fabric comparison (the same collective on a
torus and on a switched tree, byte counts computed from link constants).

Chip constants extend `chips.json` (more TPU generations) and add `gpus.json`, both with
the `source`/`retrieved` provenance contract; every number on these pages traces to the
scaling book or vendor docs, and the roofline instrument inherits the new chips for free.

## The series (initial shelf)

1. **the machine** (~6 pages): the TPU chip (MXU as systolic array, VPU, VMEM/SMEM,
   scalar core, SparseCore); generations v3 to 7x and what each changed; the TPU fabric
   (ICI, torus shapes, wraparound rules, DCN, hosts); the GPU chip (SM anatomy, SIMT vs
   SIMD, TMEM); the GPU fabric (NVLink/NVSwitch, NVL72, the InfiniBand fat tree); two
   machines one job (the mapping table, why kernels differ).
2. **pallas, the language** (~6 pages): why a kernel language lives inside JAX; refs and
   mutation in a functional language; BlockSpec and the index map; the grid and the
   pipeline; the scalar world; from tracing to machine code.
3. **xla from the inside** (~7 pages): PJRT as the boundary, at
   write-your-own-plugin level; HLO and its invariants; the pass pipeline from a real
   dump; fusion; SPMD; codegen and the backend seam; IFRT and Pathways above.

Sources are fetched and recorded in the working notes: scaling-book TPU and GPU chapters
(constants, topologies, the GPU-to-TPU mapping table), the Pallas design doc, the PJRT
integration guide (symbol names, discovery, versioning). Prose is voice-gated; numbers
carry provenance; every page names what it deepens.

## Steps

- [ ] `lib/series.ts` + `SeriesPager.astro` + page templates
      (`pages/series/index.astro`, `[series]/index.astro`, `[series]/[id].astro`)
- [ ] Chrome registration, once: nav link, OG slug (falls back to home card until
      `gen-og.mjs` learns the slug, tracked below), PathStrip neutral state for
      `/series`, ProgressKit segment, `sr:` mastery prefix, search-index entries
- [ ] Structure tests: `data/series/series.test.ts` in the shape of the track tests
- [ ] `chips.json` gains v4p/v5p/7x rows; `gpus.json` lands with A100/H100/B200 rows
- [ ] Series 1: the machine, with the new instruments
- [ ] Series 2: pallas, the language
- [ ] Series 3: xla from the inside
- [ ] Reduced-motion guard added to the stepper idiom and retrofitted to Choreographer
- [ ] Follow-up debt, tracked not done: migrate the four paths onto `lib/series.ts`;
      teach `gen-og.mjs` the series slug; the dead `#exercises` anchor in the mastery
      ledgers
