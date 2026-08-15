# The descent revamp: one story, one spine

Status: decided (2026-08-14, same day); decisions recorded in the last
section, phases M1+ cleared to build. Supersedes the IA framing of
`true-mastery-courses.md` where they conflict; the arc model (unit → lessons →
practice → prove) survives unchanged, because it works.

## The diagnosis, stated bluntly

The site owns four courses, three workshop surfaces, walks, labs, instruments,
and (per the three new floor plans) is about to own the compiler floor, the
silicon floor, and the runtime boundary. Every piece is good; the whole reads
as a pile. A new reader cannot answer "where am I, what is below me, and why
does the next unit exist" without holding the courses in their head the way
the authors do. Course-first IA hides the one thing this site knows that
nothing else on the internet teaches in one place: it is all one vertical
stack, and every unit is a floor or a seam of it.

## The thesis: the stack becomes the site

One program descends from the code you wrote to electrons in a die. That
descent is the site's actual subject, so it becomes the site's actual IA.

The map is an hourglass:

```
  PyTorch          JAX               (L0 · the program you wrote)
     \              /
   FX/lazy IR   jaxpr                (L1 · the trace)
        \        /
        StableHLO                    (L2 · the portable graph)
     ~~~ the seam: PJRT ~~~          (where the program changes owners)
           HLO                       (L3 · the compiler)
         /      \
  Triton/GPU   Pallas/Mosaic/TPU     (L4 · the kernel)
      |            |
   PTX/SASS       LLO                (L5 · machine code)
      |            |
   SM/HBM      TensorCore/HBM        (L6 · the silicon)
      |            |
   NVLink         ICI                (L7 · the pod)
```

Two frameworks converge into a narrow waist, cross one seam, and diverge into
two machines. Every existing unit and every planned lesson has a cell on this
map: a level, and a side (or the waist). The map is navigation, not content;
one-fact-one-home is untouched.

## The five moves

1. **The map is the home page and the constant companion.** Home renders the
   hourglass live: every cell shows its units and the reader's state. Every
   lesson page carries a miniature of it, current cell lit, so "where am I in
   the stack" is always one glance away. The old scroll-as-descent idea
   becomes the literal front door. Courses do not die: they become named
   routes drawn across the map (the kernel path descends the TPU side, the
   XLA path lives at the waist and below, the PyTorch path descends the left
   edge to the seam). URLs do not change; the map is a layer over the
   existing pages, so no renumbering and no redirect debt.

2. **One specimen program, every floor.** Capture one small training step
   (one attention block, real code, both frameworks) and dump it at every
   level of the map: source, jaxpr and FX, StableHLO, optimized HLO, a
   Pallas and a Triton kernel from inside it, PTX and SASS, plus the seam
   calls it makes (from the R2 trace). Every unit opens with its slice of
   the specimen: "this is the same program you saw upstairs, at this floor."
   Continuity of example is the single biggest teach-better move available;
   today every chapter introduces a fresh toy. The capture pipeline is a
   checked-in script per the provenance rule, and the compiler-floor and
   silicon-floor capture passes fold into it.

3. **Reading the real thing becomes the site-wide verb.** The walk player
   generalizes into an artifact explorer: scrub vertically through the
   specimen's representations, expand any level into the full dump, step
   through the walks that already exist where they exist. The map drill
   ("here is an unlabeled artifact, place it") becomes the site's signature
   game, wired as a gym station over captured artifacts only.

4. **Depth is the progress model.** Mastery rolls up per cell, and the
   reader's headline number is their depth gauge: the deepest level at which
   they can read fluently, per side. Course completion bars remain inside
   route pages, but the identity the site sells is "I read to L5 on the TPU
   side," which is both truer and more motivating than four percentages.

5. **The workshop dissolves into the map.** Gym stations, museum exhibits,
   and bench records pin to cells (the arcs already point this way). The
   Workshop nav item retires; nav becomes: the Map · Routes · GitHub.

## The UI question, faced honestly

ADR 0001 kept the paper register a week ago, and the founder has called the
result underwhelming twice since. The recommendation is to supersede ADR 0001
and migrate chrome to Brutalist Studio Black v2, the canonical Rudrite
language: hard 2px borders, offset shadows, dark and light both shipped,
display-scale level numerals. The hourglass map wants exactly that visual
weight, and the academy exemplar proves the language at production quality.
The reading column inside lessons keeps its serif and generous measure
(long-form reading is still the product); everything around it gets the
brutalist chrome. This carries `docs/decisions/0003-supersede-paper.md` plus
the brand README note in the same commit. The alternative, a third register
invented fresh, is more work for less coherence and is not recommended.

## What does not change

The arc model and its checks. One home per fact. URLs. The firewall, the
voice gate, provenance on every number. The four tracks' content, which
becomes route content. All current tests keep passing throughout; the map is
additive data (`cell: { level, side }` per unit) plus new surfaces.

## Phases

- **M0** · this proposal; founder picked, see the decisions section. Done.
- **M1, the map.** Shipped 2026-08-14: cells on every unit, the hourglass as
  the front door, mini-maps, routes as overlays, eleven invariant tests.
  Purely additive; the old home content survives below the map until M3.
- **M2, the specimen.** One attention step captured at every level, both
  frameworks, and surfaced through the site.
  1. *Capture pipeline.* A checked-in script (`bench/specimen/`) that emits
     every artifact reproducible on CPU: source, jaxpr, FX, StableHLO,
     optimized HLO, LLVM IR, plus the torch_xla seam trace when run on a
     Colab TPU (paste-back, like the existing reader-driven labs). GPU-only
     stages (Triton stack, SASS) enter as cited captures with pinned
     provenance, never synthesized. The artifact-descent lab authored under
     the compiler-floor plan is the pipeline's seed; M2 promotes it from
     notebook to committed artifacts with a regeneration script.
  2. *Specimen slices.* Each unit on the map opens with its slice of the one
     program ("your program at this floor"), a short excerpt linking into
     the full artifact. One-fact-one-home holds: the slice quotes, the
     lesson teaches.
  3. *Artifact explorer v1.* The walk player generalized: scrub vertically
     through the specimen's levels, expand any level to the full dump. One
     component, fed by the captured artifacts.
- **M3, depth.** The progress model and the surfaces that ride it. Shipped
  2026-08-15: the depth gauge, the place-the-artifact station, the workshop
  fold with its three-item nav, and the home reconcile. Two notes against
  what the steps below asked for. No URL was retired, so `_redirects` stays
  untouched. And the bench rows from the jax, xla and pytorch labs carry a
  stage number no chapter claims, so the map lists the kernel-stage groups
  only, while the bench page keeps every row.
  1. Mastery re-rollup by map cell; the depth gauge ("reads to L5, TPU
     side") as the headline number on home and profile surfaces; course
     bars remain inside route pages.
  2. The place-the-artifact station: an unlabeled artifact, the reader
     places it on the map; corpus drawn only from the specimen captures and
     lesson-quoted dumps (the drill the compiler-floor plan deferred here).
  3. Workshop fold: gym stations, museum exhibits, and bench records pin to
     cells; the Workshop nav item retires; nav becomes Map · Routes ·
     GitHub. `_redirects` maps the retired URLs.
  4. Home reconcile: the old scroll-descent bands and path cards merge into
     the map surface so the stack is told once.
- **M4, the register.** The founder-directed new theme: brutalist base,
  retro-fitted morphisms, techish, aiming for beautiful.
  1. *Concept boards first.* Two or three full-page concepts (home + one
     lesson page each), built as static comps, presented for founder pick.
     Nothing migrates before a pick.
  2. On pick: `docs/decisions/0003-<name>.md` superseding ADR 0001, brand
     README deviation note in the same commit, tokens and primitives, then
     surface-by-surface migration.
  3. Exit gate: three widths, light and dark, reduced motion, zero
     regressions on the map and instrument surfaces.
- Ordering: M2 before M3 (the station and the slices need the captures);
  M4 can run beside either once concepts are picked. Remaining boundary-plan
  items (the reconcile pass, the third walk, the labs) finish independently
  and feed M2's capture inventory.

## Decisions (founder, 2026-08-14)

1. **IA**: stack-first; courses become routes over the hourglass map.
2. **Register**: neither plain Studio Black v2 nor a paper evolution. A new
   brutalist-based theme, retro-fitted morphisms on a techish base, aiming
   for genuinely beautiful UI. This deviates from the canonical register, so
   M4 opens with concept boards for founder sign-off, and the shipping
   commit carries the superseding ADR (0003) plus the brand README deviation
   note, per the cross-project rule.
   **Resolved 2026-08-14, second pick:** from three boards (Phosphor,
   Faceplate, Platen) the founder chose **Phosphor**: the site as an
   instrument readout, hard brutalist blocks, aperture-grille tint on
   chrome, bloom spent only on the accent and on measured numerals, the
   hourglass as a lit spine. Dark mode ships as part of the direction (the
   board designed it as a full design, not a filter), so ADR 0001's
   no-dark-mode scope falls with the rest of it. M4a migrates tokens,
   primitives, home with the map spine, and the lesson template; M4b
   sweeps the remaining surfaces.
3. **Specimen scope**: both frameworks and both machine sides from the start.
4. **Workshop fold**: dissolve into the map in M3; nav goes to three items.
