# True mastery courses: the site, replanned

Founder direction, 2026-08-07, five directives in sequence: fold the series into the
courses; no second surface teaching the same thing; the existing paths should teach
better and more verbosely; they should be true mastery courses, not high-level
run-throughs; one article per topic is not a teaching model; and the whole site plus
its UI needs replanning, because the current look is underwhelming.

This plan supersedes the phase-1 merge in the earlier revision of this file (which
would have produced exactly the "one long article per topic" shape the founder
rejected) and the shelf design in `series.md`.

## 1. Honest diagnosis

**Teaching.** A chapter today is one article. The practice surfaces are real and good
(gym drills with streaks, runnable labs, guided walks, 18 instruments, the museum) but
they sit beside the narrative as destinations, not inside it as steps. Depth arrived
yesterday as a separate shelf, which was the wrong shape: a second place to learn a
subject the course already owned. Nothing today sequences a topic as an arc that ends
in demonstrated competence.

**IA.** Nine top-nav items. The kernel course spans `/`, `/l/`, `/s/` with numbering
coupled across them. The workshop surfaces read as parallel products. The shelf added a
fourth taxonomy in a day.

**UI.** The site runs a paper/copper register that predates the brand ruling. Brutalist
Studio Black v2 has been the canonical Rudrite-wide design language since 2026-04-29
(brand README; academy is the reference exemplar), and kernels never migrated. Flat
panels, small type presence, no dark mode, no depth. The founder's "underwhelming" has
a sanctioned answer already written down.

## 2. The teaching model: course → unit → lesson arc

A **unit** is one topic taught to competence (today's chapters are the unit list). A
unit is not an article; it is an ordered arc:

1. **Orient.** Why this exists, where it sits in the stack, what you arrive with, what
   you leave able to do, how long it takes. The unit hub page.
2. **Teach.** Two to six focused lessons, each one idea, each 10 to 25 minutes. Prose
   in the house voice, verbose where the idea needs it. Evidence inline at the moment
   of need: code that ran, captures quoted verbatim, tables with provenance,
   instruments embedded in the lesson that raises their question. Every lesson ends
   with a check: two or three questions or a micro-task with a revealable answer.
3. **Practice.** Drills, labs, and walks are steps in the arc with their place in the
   order, not sidebar links. A drill step names its streak target; a lab step is the
   notebook; a walk step is the guided reading.
4. **Prove.** The unit checklist ("can you..."), auto-verified where the site can
   check (streaks, labs run, checklist), manual where it cannot. Stages keep their
   measured gates.
5. **Return.** The museum's failures and the gym's streaks are the spaced-review layer;
   units link their exhibits as "come back when it breaks."

Rules that hold everywhere: one canonical home per fact (two tellings of one mechanism
is a bug); every claim carries evidence; every unit has at least one applied item;
lesson kinds reuse the primitives the site already has (concept, walk, lab, drill
session, instrument) so this is sequencing work first and authoring work second.

**Where yesterday's series content lands:** the pages become lessons inside their
owning units. machine → units `l:tpu`, `l:ici`, `s:machine` (GPU contrast material).
pallas → units `l:pallas`, `s:pallas`, `l:mosaic`. xla-internals → the seven xla units
it deepens (`pjrt`, `hlo`, `pipeline`, `fusion`, `spmd`, `codegen`, `ifrt`). The shelf,
its nav item, and the `sr:` namespace retire; `_redirects` maps every old URL to the
lesson that absorbed it.

## 3. The IA

- **URLs.** Units keep today's chapter URLs and become hubs; lessons nest under them:
  `/l/tpu` (hub) with `/l/tpu/<lesson>`, `/s/pallas/<lesson>`, `/xla/pjrt/<lesson>`.
  No course renames, no chapter renumbering, redirects only for retired series URLs.
- **Nav.** Six items: Kernels · JAX · XLA · PyTorch · Workshop · GitHub. The Workshop
  page fronts gym, museum, and bench. Home becomes the catalog: four course tiles with
  live motifs, a continue-where-you-left block, and the bench receipts strip.
- **Progress.** Lesson mastery rolls up to unit, unit to course. Streaks, labs, and
  checklist mechanics unchanged. The strip and spine keep working at unit granularity;
  hubs show per-lesson state.

## 4. The UI: adopt Brutalist Studio Black v2

Migrate the chrome to the canonical language (brand README, promotion of 2026-04-29;
primitive contract as written there):

- Neutral surfaces (`#fafafa` light, `#131316` dark, both modes shipped), 2px hard
  borders, opaque offset shadows at 3/4/6/8px, hover lift by translate plus shadow
  grow, `rounded-[4px]` max, no blur, no gradients on chrome, no soft shadows.
  Reduced motion collapses animation, never slows it.
- Primitives mirrored from the academy exemplar as Astro/CSS: tile, button, chip,
  divider. Every card surface (unit tiles, lesson list rows, readings, labs, drill
  stations, museum exhibits, bench records) becomes a brutalist tile.
- **Chrome accent = the brand accent token; copper and steel become semantic-only.**
  Copper keeps meaning algorithm, steel keeps meaning schedule, green pass, red fail,
  but only inside diagrams, instruments, and prose highlights. Buttons, links, active
  states, and progress chrome use `brand-*`. No palette-deviation ADR needed this way.
- Type: VT323 stays lockup-only. IBM Plex Serif stays for lesson narrative (long-form
  reading is the product; the academy register's mono-caption discipline applies to
  chrome labels). Type scale gets a real ramp: hubs and the catalog carry display-size
  numerals and titles; today's timid hierarchy is the main "flat" feeling.
- Instruments keep their dark stage panels (they read as equipment), reframed with the
  2px border + offset shadow so they sit in the new chrome instead of floating.
- The home scroll-as-descent is worth keeping as an idea (it is the course's thesis in
  scroll form) and rebuilding in the new language with the catalog on top.

## 5. Phases

- **P0.** This plan, plus the decisions below. Nothing ships until the founder picks.
- **P1, IA scaffold.** Unit/lesson primitive (yesterday's series machinery becomes it:
  same data grammar, new nesting); series pages migrated to lessons under owning
  units; shelf and `sr:` retired; redirects; progress rollups; workshop consolidation;
  nav to six items. Tests move with it.
- **P2, UI migration.** Tokens, primitives, chrome surfaces, both modes, home catalog,
  unit hub template, lesson template. Browser-verified at three widths, light and
  dark, reduced motion checked.
- **P3, lesson arcs for kernels + xla.** Unit by unit: split the existing article into
  focused lessons, place practice steps into the order, write the checks, reconcile
  duplicates with the absorbed series lessons, deepen where the material is thin.
  Order by learner impact: part i layers first.

  The working protocol, per unit (this is the loop's contract):
  1. Read the unit's current article, its guide if it has one, its absorbed lessons,
     and its mastery ledger. Decide the arc: 2 to 6 lessons, each one idea, ordered.
  2. Author or restructure the lessons in `site/src/data/lessons/`. Every lesson ends
     with a check (2 or 3 questions with revealable answers; the first P3 iteration
     adds the `check` field to the Lesson type and template). Reconcile duplicates:
     one telling per fact, the hub keeps orientation, the lessons keep depth.
  3. Sequence practice into the hub: the unit's drills, labs, and walks appear as
     ordered mastery items with auto rules where checkable.
  4. Voice per `docs/VOICE.md`; facts verified (run snippets or quote verbatim from
     corpora); numbers carry provenance.
  5. `npm test` and `npm run build` green (build runs the firewall: never a blocked
     word in files or commit messages), unit checked in a browser, one conventional
     commit per unit, pushed. Tick the unit below in the same commit.

  The worklist, in order:
  - [x] P3 bootstrap: `check` field on lessons + template rendering + test invariant
  - [x] l:source · l:jaxpr · l:stablehlo (the IR descent, one arc each)
  - [x] l:xla · l:gap (the compiler wall and the escape; the gap stays a single-article unit by design, one bridging idea, and gains an applied ledger item)
  - [x] l:pallas · l:mosaic (absorbed lessons reconciled; the pallas guide sections on practice depth are reserved for the s:pallas arc and the hub now renders only the unclaimed remainder)
  - [x] l:tpu · l:ici (absorbed machine lessons reconciled; both guides fully told in lessons: profiling arc under tpu, collectives arc under ici)
  - [x] s:machine (GPU contrast lessons with checks) · s:pallas (the three absorbed
        lessons claim the pallas guide's reserved practice sections; the language guide
        is now fully told across the two pallas units)
  - [ ] s:ir · s:kernels · s:distributed · s:capstone
  - [ ] xla:pjrt · xla:ingestion · xla:hlo · xla:pipeline
  - [ ] xla:fusion · xla:layout-memory · xla:spmd · xla:collectives
  - [ ] xla:codegen · xla:autotuning · xla:ifrt · xla:mcjax · xla:pathways ·
        xla:capstone
  - [ ] Close: README and CURRICULUM reflect the arcs; this file's grades updated;
        the loop reports and stops
- **P4.** The same pass for jax and pytorch.
- **P5.** Continuous deepening against the model, chapter grades tracked in this file.

## 6. Decisions (founder, 2026-08-07)

1. UI: **keep the paper register and refresh it** (type scale, depth, hierarchy).
   This is a deviation from the canonical Brutalist Studio Black v2, so it carries
   `docs/decisions/0001-paper-register.md` and a brand README update in the same
   commit. Section 4 above is amended accordingly: the primitive contract does not
   apply here; the refresh raises presence within the paper language.
2. Accent: **brand token for chrome; copper and steel semantic-only** in diagrams,
   instruments, and prose highlights.
3. Phase order: **IA scaffold → UI → lesson arcs**, as listed.
