# ADR 0003: the register is Phosphor, and dark mode ships with it

Date: 2026-08-14 · Status: accepted (founder decision) · Supersedes ADR 0001; keeps the
accent-token reasoning of ADR 0002

## Context

ADR 0001 kept kernels on its paper register and ruled out dark mode for that pass. The
founder called the live result underwhelming twice, and the descent revamp
(`docs/plans/descent-revamp.md`, phase M4, decision 2) directed a new register: brutalist
base, retro-fitted, aiming at genuinely beautiful UI rather than at the canonical
Brutalist Studio Black v2. That decision required concept boards before any migration.

Three full boards were built, each with four plates (home and one lesson page, in dark and
in light): Phosphor, Faceplate, Platen. On 2026-08-14 the founder picked **Phosphor**: the
site as an instrument readout, hard brutalist blocks, an aperture-grille tint on chrome
plates, bloom spent only on the accent and on numerals that carry a measurement, and the
hourglass map drawn as a lit spine.

## Decision

1. kernels migrates to the Phosphor register. The paper register falls with ADR 0001. The
   ground, plate, edge and ink scales come from the board's token sheet, a warm ink scale
   rather than the canonical neutral grey scale.
2. Dark mode ships. The board designed each mode as a full design rather than as a filter
   over the other, so ADR 0001's "no dark mode in this pass" clause falls with the rest of
   it. There is no mode toggle yet; the site follows `prefers-color-scheme`, and a stored
   preference can be added later without a new palette.
3. Two chrome effects enter that the brutalist primitive contract does not have. Bloom is a
   `text-shadow` (and, on lit bars and indicators, a `box-shadow`) applied to the accent and
   to measured numerals only; body prose never glows, and bloom resolves to `none` in light
   mode, where the offset stamp does the separating work instead. The aperture-grille tint
   is a 1px-in-3px repeating gradient at about 2% opacity, allowed on chrome plates and
   instrument panels and forbidden inside the reading column.
4. What the brand keeps, unchanged: one accent token pair (`--accent`, `--accent-deep`) in
   the terracotta family, so a future accent change stays a three-line edit (ADR 0002's
   reasoning survives intact); VT323 on the wordmark and nothing else; the voxel mark never
   on pure white, with the Dark-Ember variant of the mark rendered on dark plates by the
   brand README's role-for-role hex map; IBM Plex Serif carrying the reading column at a
   64ch measure, because long-form reading is the product. Copper and steel stay semantic
   (algorithm and schedule) and stay out of interactive chrome, per ADR 0001 clause 2.
5. A display grotesque joins the family for headlines, route titles and section heads. It is
   self-hosted like the other three faces. Serif and mono roles are otherwise untouched.

## Consequences

- The brand README's migration-cadence line for kernels currently records the paper-register
  deviation and points at ADR 0001 and 0002. That line has to be rewritten to record the
  Phosphor deviation, dark mode, and the two new chrome effects, pointing here. The brand
  repo sits outside this repo's worktree, so the edit lands with the merge of this change,
  not in this commit. Until it lands, the brand README understates what kernels ships.
- This commit is M4a: the token sheet with both modes, the shared primitives (plate, bloom,
  grille, section head, panel frame), the home page with the map as a lit spine, and the
  lesson template with its rails, checks, code frames and provenance captions. The site is
  visually mixed until M4b, which sweeps the chapter pages, the workshop, the gym, the
  museum and the bench through the same primitives.
- Instrument panels stay dark in both modes. They are the site's stage surfaces and they
  keep their own ink scale; what changes is the frame around them.
- A future register change supersedes this ADR with a new one. This file does not get edited.
