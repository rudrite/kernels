# The voice contract

All reader-facing prose in this repo follows one voice: a person explaining, not a spec
defining. That covers chapter prose in the four `track.ts` files, the series pages,
guides JSON, walks, mistake-exhibit `why` text, page and component copy in `site/src/`,
README.md, and CURRICULUM.md.

Two layers enforce it:

1. **The mechanical gate.** `scripts/voice.mjs` runs in `npm test` and `npm run build`
   (so deploys refuse a hit). It bans em-dashes and spaced dashes, punchline hooks,
   report scaffolds, bold-label stamps, and a banned idiom list. A hit is a defect, the
   same class as a failing test. If you add a prose surface outside the scanned roots,
   add it to `PROSE_ROOTS` in the same commit.
2. **The judgment layer**, which no regex catches. Before shipping any new or edited
   prose, walk four rules:
   - Term after intuition: the concrete moment first, the name second. Never open
     "Term: definition".
   - No colon-led templates or repeated entry shapes across sibling items; vary how
     entries begin.
   - One idea per sentence; a short sentence for the punch, a longer one for reasoning.
   - Hedges become human asides, not spec clauses.
   Read the paragraph aloud. If it reads like a spec defining, rewrite it before it
   ships.

Untouchables in any prose pass: verbatim error text, code, measured numbers and their
provenance, ids and anchors, generated corpora. Reword around facts, never through them.
Prose already at the bar is left alone; churn is a defect.

Page `<title>` separators use " · ", never a dash.
