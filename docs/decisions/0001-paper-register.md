# ADR 0001: kernels stays on the paper register, with the brand accent

Date: 2026-08-07 · Status: accepted (founder decision)

## Context

Brutalist Studio Black v2 has been the canonical Rudrite-wide design language since
2026-04-29 (brand README), and kernels.rudrite.com was built after that ruling on its
own paper/copper register without a recorded deviation. During the true-mastery replan
(docs/plans/true-mastery-courses.md) the founder reviewed both options.

## Decision

1. kernels keeps the paper register: warm paper surfaces, ink rules, dark instrument
   panels, IBM Plex Serif for narrative and IBM Plex Mono for structure. The refresh
   raises type scale, hierarchy, and depth within that language rather than migrating
   to the brutalist primitive contract.
2. Chrome accent moves to the brand accent (indigo `#4f46dc`, deep `#423ac0`, with a
   brighter `#7d76ec` for dark panels). Copper and steel become semantic-only: copper
   means algorithm and steel means schedule inside diagrams, instruments, pull quotes,
   goal blocks, designators, and prose highlights; neither is interactive chrome.
3. No dark mode in this pass; the instrument panels remain the site's dark islands.

## Consequences

- The brand README's migration-cadence list records kernels as a founder-approved
  paper-register deviation pointing at this ADR (updated with this change).
- The voxel mark and wordmark rules are unaffected; the lockup keeps its canonical
  rendering and never sits on pure white (the paper is `#f4f5f2`).
- A future migration to Brutalist Studio Black v2 supersedes this ADR with a new one;
  this file does not get edited.
