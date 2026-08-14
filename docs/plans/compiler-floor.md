# The compiler floor: IR and hardware vocabulary arcs

Status: planned (2026-08-14). Enters under the P5 standing rule in
`true-mastery-courses.md`: new material lands as lessons inside owning units,
one telling per fact.

## The goal

The reader can place any name from the compiler and hardware zoo on a two-axis
map (abstraction level vertical, ecosystem horizontal) and can read real dumps.
The mastery criterion is concrete: handed an unlabeled IR or assembly dump,
name the dialect or ISA, the level, and what transformation came just before
and after it.

The vocabulary in scope: SSA, LLVM, MLIR, dialects and lowering, HLO, MHLO,
CHLO, StableHLO, VHLO, Triton, Mosaic, LLO, ISA, microarchitecture, virtual
ISA, PTX, SASS, VLIW, superscalar, SIMD, SIMT, systolic array.

## The audit: term by term, where it lives today

| Term | Current home | Verdict |
| --- | --- | --- |
| StableHLO | `l:stablehlo`, `xla:ingestion` | taught; cross-link only |
| HLO | `xla:hlo`, `l:xla` | taught; cross-link only |
| Mosaic | `l:mosaic` | taught; gains the LLO edge below |
| MXU / systolic array | `l:tpu` | taught |
| SSA | none | new lesson |
| LLVM (IR + pipeline) | brushed in `xla:codegen` | new lesson |
| MLIR (dialects, lowering) | implicit in `l:stablehlo` | new lesson |
| MHLO / CHLO / VHLO | none | new lesson |
| Triton internals (TTIR/TTGIR) | named in `xla:autotuning` | new lesson |
| LLO | none | new lesson |
| ISA / µarch / PTX / SASS | none | new lesson |
| VLIW / superscalar / SIMD / SIMT | fragments in `s:machine` | new lesson |

## Placement: no new units, no renumbering

Every new lesson has a natural owning unit already on the site. The IR descent
(`l:source` → `l:jaxpr` → `l:stablehlo` → `l:xla` → `l:gap`) absorbs the
compiler-theory layer; the machine units absorb the hardware layer; the XLA
course absorbs the backend layer.

1. `l:jaxpr` · **SSA, one name one definition.** A jaxpr is already SSA; teach
   the property there, where the reader first meets an IR that has it. Phi
   nodes via the scan/cond contrast.
2. `l:xla` · **The one-level world (LLVM).** What LLVM IR is, reading
   practice, why one low level cannot see tensor structure. Motivates the
   wall the unit already teaches.
3. `l:xla` · **Levels as a first-class idea (MLIR).** Dialects, progressive
   lowering, legalization; the same computation read in three dialects.
4. `l:stablehlo` · **The HLO family tree.** HLO existed, MLIR arrived so MHLO
   bridged in, portability demanded StableHLO, stability demanded VHLO,
   frontend fidelity demanded CHLO. Taught as history because the names only
   make sense as history.
5. `s:machine` · **The contract: ISA, microarchitecture, PTX, SASS.** ISA as
   interface, µarch as implementation, virtual ISA, ptxas, reading SASS.
6. `s:machine` · **Who schedules: VLIW, superscalar, SIMD, SIMT.** One
   question (who decides what runs in parallel, and when) sorting all four;
   warps and divergence; why dense linear algebra suits static scheduling.
7. `l:tpu` (or `l:mosaic`, decide during authoring by where the guide
   sections sit) · **The TPU as a VLIW machine, and LLO.** The private floor,
   told honestly to the verify-boundary per the pathways precedent.
8. `xla:codegen` · **Triton end to end.** TTIR → TTGIR → LLVM IR → PTX, the
   four dumps of one kernel, and where XLA:GPU uses Triton vs its own
   emitters.

The parallel worth naming across lessons 5 and 7: PTX is to SASS what the
public Mosaic level is to LLO, a stable public layer over a private
per-generation floor.

## Practice and proof

- Lab: **one program, every artifact.** One small jit step dumped at every
  level (`lower().as_text()`, `--xla_dump_to`, a Pallas kernel's Mosaic dump,
  a Triton kernel's four dumps, `nvdisasm` where a GPU exists). The lab is the
  provenance for every lesson claim.
- Drill (optional, judge during authoring): a name-the-dialect station over
  `ir-corpus.json` style snippets, streak wired to the descent units. Only if
  the corpus can be captured from real runs; no synthetic dumps.
- Checks: every lesson ends with 2-3 revealable questions, per the standing
  arc rules.

## Work items

Per-unit protocol as in `true-mastery-courses.md` §P3 (author, reconcile
duplicates, voice per `docs/VOICE.md`, verified snippets with provenance,
tests and build green, browser check, one conventional commit per unit).

- [x] Capture pass, folded into each lesson: every artifact either ran on
      this machine and is quoted verbatim, or is cited at a pinned commit
      or URL; nothing synthetic shipped
- [x] Lessons 1-4 (the descent layer: SSA, LLVM, MLIR, HLO family)
- [x] Lessons 5-6 (the machine layer: ISA contract, scheduling styles)
- [x] Lesson 7 (TPU VLIW + LLO edge)
- [x] Lesson 8 (Triton in xla:codegen)
- [ ] Lab notebook + lab cards on the owning units
- [ ] Optional drill station, or a recorded decision not to
- [ ] CURRICULUM.md and unit hubs reflect the new lessons; grades noted in
      `true-mastery-courses.md` §P5
