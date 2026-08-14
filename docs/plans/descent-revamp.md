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

- **M0** · this proposal; founder picks the decisions below.
- **M1, the map.** Cell coordinates on every unit, the map component, home
  replaced, mini-map on lesson pages, routes as overlays. Purely additive.
- **M2, the specimen.** Capture pipeline (folds in the compiler-floor and
  silicon-floor capture passes), specimen slices at the top of each unit,
  artifact explorer v1 from the walk player.
- **M3, depth.** Progress re-rollup by cell, depth gauge, place-the-artifact
  station, workshop fold, nav to three items.
- **M4, the register.** UI migration behind the superseding ADR, browser
  verified at three widths, both modes, reduced motion.
- The three floor content plans proceed in parallel; their lessons land in
  cells from day one.

## Decisions (founder, 2026-08-14)

1. **IA**: stack-first; courses become routes over the hourglass map.
2. **Register**: neither plain Studio Black v2 nor a paper evolution. A new
   brutalist-based theme, retro-fitted morphisms on a techish base, aiming
   for genuinely beautiful UI. This deviates from the canonical register, so
   M4 opens with concept boards for founder sign-off, and the shipping
   commit carries the superseding ADR (0003) plus the brand README deviation
   note, per the cross-project rule.
3. **Specimen scope**: both frameworks and both machine sides from the start.
4. **Workshop fold**: dissolve into the map in M3; nav goes to three items.
