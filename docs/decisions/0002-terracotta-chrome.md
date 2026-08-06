# ADR 0002: chrome accent is terracotta, not the brand token

Date: 2026-08-07 · Status: accepted (founder decision) · Supersedes the accent clause
of ADR 0001

## Context

ADR 0001 moved interactive chrome to the brand accent (indigo) while keeping copper
semantic. Seen live, the founder judged the indigo foreign to the site ("blue looks
odd") and directed a return to copper.

## Decision

The chrome accent tokens (`--accent`, `--accent-deep`, `--accent-bright`) point at the
terracotta values (`#c88a70` / `#a86e55`). Chrome and the algorithm hue share a family
by intent: on this site the interface itself reads as part of the instrument. The
token separation stays in the code, so any future accent change is a three-line edit.
Everything else in ADR 0001 stands: paper register, the presence layer, copper and
steel as the meaning hues, no dark mode this pass.

## Consequences

- The brand README's kernels deviation line now records the terracotta chrome accent
  (updated with this change).
- The wordmark's middle-i accent and the marks are untouched, as always.
