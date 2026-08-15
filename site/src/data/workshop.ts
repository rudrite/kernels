// The workshop, folded into the map. A gym station, a museum wing and a
// group of bench records all belong to a floor of the stack, and the unit
// that assigns each one already says which floor that is. So the pin here is
// a unit key, never a second copy of a cell: move a chapter on the map and
// its equipment moves with it (docs/plans/descent-revamp.md, phase M3
// step 3).
//
// The equipment keeps living on its own pages. What this module adds is the
// one line the map needs to list it under the right cell, which is why the
// labels are short and the descriptions stay where the surfaces are.
import { UNIT_CELLS, type Cell } from './map'
import { STAGES } from './track'
import mistakes from './mistakes.json'
import jaxMistakes from './jax-mistakes.json'
import xlaMistakes from './xla-mistakes.json'
import pytorchMistakes from './pytorch-mistakes.json'
import results from '../../../bench/results.json'

export type WorkshopKind = 'gym' | 'museum' | 'bench'

export interface WorkshopEntry {
  kind: WorkshopKind
  /** What the surface calls it, short enough to sit under a cell's units. */
  label: string
  /** A count or a designator the surface already prints. */
  meta: string
  href: string
  /** The unit that assigns this equipment. */
  unit: string
}

interface Pinned extends WorkshopEntry {
  cell: Cell
}

// Stations, each pinned to the unit whose mastery work assigns it. The
// floors host more stations than they own: a drill counts on every floor
// that drills it, and the map lists it once, where the ledger asks for it.
const STATIONS: WorkshopEntry[] = [
  { kind: 'gym', label: 'the dot_general decoder', meta: 'GYM·01', href: '/gym/kernels#drills', unit: 'l:stablehlo' },
  { kind: 'gym', label: 'the shape oracle', meta: 'GYM·02', href: '/gym/kernels#drills', unit: 'l:jaxpr' },
  { kind: 'gym', label: 'spot the decision', meta: 'GYM·03', href: '/gym/kernels#drills', unit: 'l:xla' },
  { kind: 'gym', label: 'the corpus x-ray', meta: 'GYM·04', href: '/gym/kernels#xray', unit: 'l:jaxpr' },
  { kind: 'gym', label: 'the mosaic x-ray', meta: 'GYM·05', href: '/gym/kernels#mosaic', unit: 'l:mosaic' },
  { kind: 'gym', label: 'name the transform', meta: 'GYM·06', href: '/gym/kernels#drills', unit: 'l:jaxpr' },
  { kind: 'gym', label: 'the fusion x-ray', meta: 'GYM·07', href: '/gym/kernels#fusion', unit: 'l:xla' },
  { kind: 'gym', label: 'the timeline x-ray', meta: 'GYM·08', href: '/gym/kernels#timeline', unit: 'l:tpu' },
  { kind: 'gym', label: 'the stride oracle', meta: 'GYM·09', href: '/gym/pytorch#drills', unit: 'pt:tensors' },
  { kind: 'gym', label: 'guard or break', meta: 'GYM·10', href: '/gym/pytorch#drills', unit: 'pt:dynamo' },
  { kind: 'gym', label: 'name the collective', meta: 'GYM·11', href: '/gym/pytorch#collectives', unit: 'pt:distributed' },
  { kind: 'gym', label: 'which pipeline filed this pass?', meta: 'GYM·12', href: '/gym/xla#drills', unit: 'xla:pipeline' },
  { kind: 'gym', label: 'count the fusions', meta: 'GYM·13', href: '/gym/xla#drills', unit: 'xla:fusion' },
  { kind: 'gym', label: 'place the artifact', meta: 'GYM·14', href: '/gym/kernels#place', unit: 'l:stablehlo' },
]

// One wing per path, pinned to the unit that teaches the rule its exhibits
// break. The kernel and pytorch pins are the ones their ledgers already
// make; the jax and xla wings have no ledger link, so the pin is the
// chapter the wing's subject belongs to.
const WINGS: WorkshopEntry[] = [
  { kind: 'museum', label: 'Pallas failures', meta: `${mistakes.length} exhibits`, href: '/mistakes/kernels', unit: 's:pallas' },
  { kind: 'museum', label: 'JAX failures', meta: `${jaxMistakes.length} exhibits`, href: '/mistakes/jax', unit: 'jax:tracing' },
  { kind: 'museum', label: 'SPMD-seam failures', meta: `${xlaMistakes.length} exhibits`, href: '/mistakes/xla', unit: 'xla:spmd' },
  { kind: 'museum', label: 'The tape defends itself', meta: `${pytorchMistakes.length} exhibits`, href: '/mistakes/pytorch', unit: 'pt:autograd' },
]

interface BenchRow {
  stage: number
}

// Records group by the stage that measured them, and a stage sits where its
// layer sits. The rows from the jax, xla and pytorch labs carry a stage
// number no chapter claims, so the map does not list them and the bench
// keeps every row it always had.
const RECORDS: WorkshopEntry[] = STAGES.map((stage) => ({
  stage,
  rows: (results as BenchRow[]).filter((r) => r.stage === stage.num).length,
}))
  .filter((g) => g.rows > 0)
  .map((g) => ({
    kind: 'bench' as const,
    label: 'what this floor measured',
    meta: `${g.rows} records`,
    href: '/bench#records',
    unit: `s:${g.stage.id}`,
  }))

/** Every piece of equipment the map lists, with the cell it lands in. */
export const WORKSHOP: Pinned[] = [...STATIONS, ...WINGS, ...RECORDS].map((e) => {
  const cell = UNIT_CELLS[e.unit]
  if (!cell) throw new Error(`workshop: ${e.label} pins to ${e.unit}, which no unit on the map claims`)
  return { ...e, cell }
})

/** What the map should print under one cell, stations first. */
export const workshopAt = (level: number, side: string): Pinned[] =>
  WORKSHOP.filter((e) => e.cell.level === level && e.cell.side === side)
