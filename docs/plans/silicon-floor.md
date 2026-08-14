# The silicon floor: TPU and GPU hardware architecture arcs

Status: planned (2026-08-14), provenance sweep in flight. Enters under the P5
standing rule in `true-mastery-courses.md`: lessons inside owning units, one
telling per fact. Companion to `compiler-floor.md`: that plan teaches who
schedules, this one teaches what is being scheduled.

## The goal

Two abilities, both checkable:

1. Given a chip's spec sheet, re-derive its headline numbers from parts:
   peak FLOPs from matrix-unit dimensions × count × clock, bandwidth from
   HBM stacks × per-stack rate, and explain every gap against the datasheet.
2. Given a published die shot or block diagram, name every block and say
   which software behavior it explains.

This turns the stage-0 chip constants from givens into consequences; the
roofline numbers the track already teaches become arithmetic over parts.

## The three floors the lessons cover

- **Silicon economics.** Wafer and die, defect-driven yield, the reticle
  limit, chiplets as the escape (yield, reticle wall, mixed process nodes),
  2.5D packaging and the interposer, HBM as stacked DRAM beside the compute
  die, SRAM vs DRAM as the tradeoff under the whole memory hierarchy.
- **On-die architecture.** GPU: the SM as the repeated unit (warp
  schedulers, register file, shared memory, matrix units), L2, HBM
  controllers. TPU: the TensorCore complex (MXU, VPU, VMEM, scalar unit and
  sequencer), MegaCore fusion, SparseCore as the third processor type for
  embedding-shaped work. The organizing contrast: ~130 small
  hardware-scheduled cores vs 1-2 huge compiler-scheduled ones.
- **Systems.** Scale-up vs scale-out: ICI and OCS-wired torus slices on the
  TPU side, NVLink/NVSwitch domains and the rack-scale machine on the GPU
  side, hosts and PCIe between.

Two naming traps get explicit call-outs: NVIDIA's Tensor Core (a unit inside
an SM) vs Google's TensorCore (nearly the whole chip), and NVIDIA structured
sparsity (a Tensor Core mode) vs Google SparseCore (a separate processor).

## The audit: where it lives today

| Topic | Current home | Verdict |
| --- | --- | --- |
| MXU / VPU / VMEM / lattice | `l:tpu` | taught; deepen with sequencer, SMEM, MegaCore |
| ICI, collectives cost model | `l:ici` | taught; absorbs OCS + torus depth |
| GPU contrast (SIMT, SMs at survey depth) | `s:machine` | taught; gains the three lessons below |
| Die, yield, reticle, chiplets, packaging, HBM | none | new lesson |
| Inside the SM | survey only | new lesson |
| NVLink domain / scale-up vs scale-out | fragments | new lesson |
| TensorCore complex + MegaCore + name collision | partial | new lesson |
| SparseCore | none | new lesson |
| Spec-sheet derivation capstone | none | new lab |

## Placement: no new units

1. `s:machine` · **The die and the reticle.** Wafer to die, yield vs area,
   the reticle limit, chiplets, interposer packaging, HBM stacks, SRAM vs
   DRAM. Blackwell's dual-die join as the worked example.
2. `s:machine` · **Inside the SM.** Warp schedulers, register file, shared
   memory, matrix units, occupancy as a register/memory budget. The SIMT
   scheduling lesson gets its hardware body here; cross-link, never retell.
3. `s:machine` · **The scale-up domain.** NVLink/NVSwitch, the rack-scale
   memory domain, where InfiniBand takes over; contrast with ICI held to one
   paragraph and a pointer at `l:ici`.
4. `l:tpu` · **The TensorCore complex.** Sequencer and VLIW issue, SMEM,
   scalar unit, MegaCore, and the name-collision box. The VLIW lesson from
   `compiler-floor.md` lands its silicon here.
5. `l:tpu` · **SparseCore.** Why embedding lookups defy a systolic array,
   the dataflow design near HBM, and the general principle: a workload class
   that defies the execution model gets a sibling core, not a bigger core.
6. `l:ici` · **The optical patch panel.** OCS, torus slices, wiring shapes,
   repair around failures; deepens the unit's existing collectives story.

## Practice and proof

- Lab: **spec sheet from parts.** One TPU generation and one GPU generation,
  every headline number derived from unit counts and clocks, reconciled
  against the datasheet, every gap explained in a sentence.
- Applied item: **label the die shot.** Published H100 and TPU die shots,
  block by block, with a revealable key.
- Checks: 2-3 revealable questions per lesson, per the standing arc rules.

## Provenance

The sweep collects, with a source URL per figure: the TPU ISCA papers (v1,
v4, the OCS paper), Hot Chips decks, NVIDIA architecture whitepapers (A100,
H100, Blackwell), HBM generation specs, reticle-limit and packaging figures,
and the scaling book's tables. TPU internals past the public record get the
LLO treatment: say where the record ends and stop. No number enters a lesson
without its source; the sweep's notes file is local scratch, the citations
ship in the lessons.

## Work items

Per-unit protocol as in `true-mastery-courses.md` §P3.

- [ ] Provenance sweep reconciled; every planned figure has a source or the
      claim is cut
- [ ] Lessons 1-3 (`s:machine`)
- [ ] Lessons 4-5 (`l:tpu`)
- [ ] Lesson 6 (`l:ici`)
- [ ] Spec-sheet lab + die-shot applied item wired into the owning units
- [ ] CURRICULUM.md and unit hubs reflect the new lessons; grades noted in
      `true-mastery-courses.md` §P5
