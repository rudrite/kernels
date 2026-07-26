# Plan: kernels.rudrite.com site scaffold

Goal: the full site UI for the curriculum. Static Astro, no client JS in v1 (islands arrive with the first interactive exhibit). Design system: datasheet vernacular; mono carries structure, serif carries thought; paper ground with dark instrument insets; copper (algorithm) and steel (schedule) as the only meaningful hues plus pass/fail. Signature element: the stack-map home, where stages pin to the layer of the compilation stack they teach and the XLA-to-Pallas gap is drawn empty.

Scope guard: no engine references anywhere; instruments never show fabricated data ("awaiting bench data" until real records exist); scrub hooks land before site code.

- [x] Scrub rails: `.githooks/commit-msg` (rejects AI attribution), `scripts/firewall.mjs` (denylist scan of tracked files + dist + history), `core.hooksPath` set
- [x] `.gitignore` (node_modules, dist, .astro)
- [x] Astro scaffold: `site/package.json`, `astro.config.mjs`, `tsconfig.json` (strict)
- [x] Tokens + global CSS: palette, type roles (Plex Serif prose, Plex Mono structure, VT323 lockup only), focus states, reduced-motion
- [x] `src/data/track.ts`: typed model of the six stages (layer pinning, labs, gates, artifacts), single source for all pages
- [x] Brand: static voxel mark in `public/` (xmlns required for img loading), `Lockup.astro` (`rudrite | kernels`)
- [x] Components: `TitleBlock`, `StackMap` (signature), `GateStamp`, `LabCard`, `Spine`
- [x] Pages: `index` (map home), `s/[id]` (six stage pages), `bench` (empty honest state from `bench/results.json`)
- [x] Verify: `astro build` clean, firewall clean, browser screenshots (desktop + mobile) read; fixed: missing svg xmlns broke the mark
- [x] Commit (clean message, no attribution), plan checkboxes ticked

Out of scope (later): interactive exhibits (roofline playground, choreographer), notebook extraction pipeline, dark mode, deploy workflow, OG images.
