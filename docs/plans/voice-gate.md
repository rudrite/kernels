# Voice gate

The site's prose is bound to the global humanized-writing rules, but nothing in this repo
enforces them. Three of the four track test files ban em-dashes and nothing else; guides,
walks, mistakes, page copy, and the README carry no check at all. This plan adds the
mechanical gate and the contract that carries the judgment layer.

The gate catches what a regex can catch. The declarative register the founder flagged
(term-before-intuition violations, colon-led templates, uniform rhythm) is not regexable;
that lives in the contract as a per-edit checklist, and in review passes.

- [x] `scripts/voice.mjs`: scans authored prose sources (site/src data, pages, components,
      layouts, exhibits, README.md, CURRICULUM.md) for mechanical violations: em-dash,
      spaced double-hyphen or en-dash used as a dash, punchline hooks, report scaffolds,
      banned idiom and reassurance shapes, bold-label stamps. Generated corpora are
      excluded by name; they are machine dumps, not prose.
- [x] Wired into `site` test and build scripts, after the existing steps, so vitest runs
      and Cloudflare deploys both refuse a violation.
- [x] `docs/VOICE.md` carries the voice contract: the surface inventory, the gate, and
      the four-rule judgment checklist every prose edit walks before shipping.
- [x] The gate fails on a planted violation and passes on the current tree (or the tree is
      fixed until it does).
- [x] Notebook prose scanned (closed 2026-08-15): `scripts/voice.mjs` now
      parses `labs/**/*.ipynb`, judges markdown cells only, and strips code
      fences and inline code spans before the dash rules run, since those
      quote commands and error text verbatim. The first run found twelve
      real hits in four older notebooks (one idiom, eleven bold-label
      stamps), fixed in the same commit.
