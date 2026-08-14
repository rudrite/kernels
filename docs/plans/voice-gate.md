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
- [ ] Notebook prose is unscanned (flagged 2026-08-14 during the floor-labs
      review): `labs/**/*.ipynb` markdown cells never pass through the gate.
      All notebooks are clean of em-dashes today (checked by grep), so this
      is hardening, not a live defect. The fix is a markdown-cell extractor
      in `scripts/voice.mjs`, not a raw-JSON scan, because code cells
      legitimately carry `--flag` strings the dash rules would misread.
