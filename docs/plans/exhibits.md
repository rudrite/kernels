# Plan: the exhibit engine and full site build-out

The loop, per iteration: build → `npm run build` (types + firewall) → screenshot affected pages at 1440 and 390 → read the screenshots, fix defects → atomic commit. Repeat until the site covers the whole track, then a final whole-site audit.

Architecture: exhibits play traces. A trace is a JSON-serializable list of frames computed by a pure generator from real block/schedule parameters; the renderer is one React island. Instruments never show fabricated numbers: chip constants come from the scaling book (fetched 2026-07-26, cited in the data file); kernel state appears as formulas, not invented values.

- [x] Iteration A: engine + matmul. Fixed after screenshot read: DMA edges rerouted through a free lane below tile rows (were crossing tiles), SVG height computed from content, HBM header added.
- [x] Iteration B: roofline. Fixed after screenshot read: PFLOP tier added to the axis formatter. Hand-checked: v5e ridge 240 F/B, N=2048 matmul 682.7 F/B compute-bound, 87.2 µs floor.
- [x] Iteration C: reuse proof. One generator, `remote` flag swaps HBM DMA for ICI transport (dashed edges, dashed tiles); renderer unchanged between flash and ring, as designed.
- [x] Iteration D: audit. All 8 routes 200; console 0 errors 0 warnings; mobile fixed after screenshot read (min-width + own scroller, SVGs no longer shrink to illegibility); firewall clean over files, dist, history.

## E2E completion pass

- [ ] Labs pipeline: `labs/stage-0/lab-0.1-rooflines.ipynb` (real, runnable, tagged cells), `scripts/extract-labs.mjs` (ipynb → generated JSON, gitignored, runs before astro build), notebook cells rendered on stage pages, LabCard honest states (Colab link activates when repo is public)
- [ ] IR x-ray (EX·05, Stage 2): real jaxpr + StableHLO generated with jax 0.4.38 (cpu; both IRs backend-independent), hover-synced three-column view; optimized HLO deliberately left as the stage's TPU lab exercise
- [ ] Trace generator unit tests (vitest): frame counts, slot alternation, transport swap invariant between flash and ring
- [ ] OG images in title-block style for all 8 pages + meta tags
- [ ] Deploy rig: GitHub Pages workflow (test + build + firewall gate), CNAME kernels.rudrite.com; remote/push left manual pending the open-source release check
- [ ] Loop: build → screenshots read → fix → commit per concern

Out of scope (later): predict-then-reveal, paste-back compare.
