// The map: one hourglass, eight levels, and a cell for every unit on the
// site. Two frameworks converge into the StableHLO waist, cross the PJRT
// seam under it, and diverge again into two machines. The map is navigation
// only; it links to units and never retells them, so one home per fact is
// untouched (docs/plans/descent-revamp.md, phase M1).
//
// Titles, hrefs and order come from the path registries. This module adds
// exactly one thing: where each unit sits.
import { LAYERS, STAGES } from './track'
import { PATH } from '../lib/path'
import { JAX_PATH } from './jax/track'
import { XLA_PATH } from './xla/track'
import { PT_PATH } from './pytorch/track'

export type MapSide = 'torch' | 'jax' | 'waist' | 'gpu' | 'tpu'

export interface Cell {
  /** 0 at the source, 7 in the pod. */
  level: number
  side: MapSide
}

export interface MapLevel {
  level: number
  /** What the program has become by this floor. */
  title: string
  left: { side: MapSide; label: string }
  right: { side: MapSide; label: string }
  /** The shared column, where both sides say the same thing. */
  waist?: string
  /** A seam band drawn below this row. */
  seam?: string
}

export const LEVELS: MapLevel[] = [
  {
    level: 0,
    title: 'the program you wrote',
    left: { side: 'torch', label: 'PyTorch' },
    right: { side: 'jax', label: 'JAX' },
    waist: 'both frameworks',
  },
  {
    level: 1,
    title: 'the trace',
    left: { side: 'torch', label: 'FX / lazy IR' },
    right: { side: 'jax', label: 'jaxpr' },
  },
  {
    level: 2,
    title: 'the portable graph',
    left: { side: 'torch', label: '' },
    right: { side: 'jax', label: '' },
    waist: 'StableHLO',
    seam: 'the seam · PJRT · the program changes owners here',
  },
  {
    level: 3,
    title: 'the compiler',
    left: { side: 'torch', label: '' },
    right: { side: 'jax', label: '' },
    waist: 'HLO',
  },
  {
    level: 4,
    title: 'the kernel',
    left: { side: 'gpu', label: 'Triton / GPU' },
    right: { side: 'tpu', label: 'Pallas / Mosaic' },
    waist: 'the gap',
  },
  {
    level: 5,
    title: 'machine code',
    left: { side: 'gpu', label: 'PTX / SASS' },
    right: { side: 'tpu', label: 'LLO' },
  },
  {
    level: 6,
    title: 'the silicon',
    left: { side: 'gpu', label: 'SM / HBM' },
    right: { side: 'tpu', label: 'TensorCore / HBM' },
  },
  {
    level: 7,
    title: 'the pod',
    left: { side: 'gpu', label: 'NVLink' },
    right: { side: 'tpu', label: 'ICI' },
  },
]

export const levelAt = (level: number): MapLevel | undefined => LEVELS.find((l) => l.level === level)

// The kernel path's layers, which the whole map is drawn from. The chapter
// that covers both frameworks at once sits in the shared column.
const LAYER_CELLS: Record<string, Cell> = {
  source: { level: 0, side: 'waist' },
  jaxpr: { level: 1, side: 'jax' },
  stablehlo: { level: 2, side: 'waist' },
  xla: { level: 3, side: 'waist' },
  gap: { level: 4, side: 'waist' },
  pallas: { level: 4, side: 'tpu' },
  mosaic: { level: 5, side: 'tpu' },
  tpu: { level: 6, side: 'tpu' },
  ici: { level: 7, side: 'tpu' },
}

// The JAX and PyTorch paths teach the top two floors: what you write, and
// what capture turns it into. Two chapters land lower, where their subject
// actually runs.
const JAX_CELLS: Record<string, Cell> = {
  arrays: { level: 0, side: 'jax' },
  tracing: { level: 1, side: 'jax' },
  jit: { level: 1, side: 'jax' },
  autodiff: { level: 1, side: 'jax' },
  vmap: { level: 1, side: 'jax' },
  'control-flow': { level: 1, side: 'jax' },
  pytrees: { level: 0, side: 'jax' },
  random: { level: 0, side: 'jax' },
  state: { level: 0, side: 'jax' },
  sharding: { level: 0, side: 'jax' },
  performance: { level: 0, side: 'jax' },
  'training-run': { level: 0, side: 'jax' },
}

const PT_CELLS: Record<string, Cell> = {
  tensors: { level: 0, side: 'torch' },
  autograd: { level: 0, side: 'torch' },
  modules: { level: 0, side: 'torch' },
  'the-loop': { level: 0, side: 'torch' },
  data: { level: 0, side: 'torch' },
  dynamo: { level: 1, side: 'torch' },
  graphs: { level: 1, side: 'torch' },
  distributed: { level: 0, side: 'torch' },
  performance: { level: 0, side: 'torch' },
  bridges: { level: 1, side: 'torch' },
  'tpu-practice': { level: 6, side: 'tpu' },
  'training-run': { level: 0, side: 'torch' },
}

// The XLA path owns the waist. Its runtime chapters sit at the portable
// graph, on the seam itself; the passes that rewrite HLO sit one floor
// down; the emitters sit where the two machines part.
const XLA_CELLS: Record<string, Cell> = {
  pjrt: { level: 2, side: 'waist' },
  ingestion: { level: 2, side: 'waist' },
  hlo: { level: 3, side: 'waist' },
  pipeline: { level: 3, side: 'waist' },
  fusion: { level: 3, side: 'waist' },
  'layout-memory': { level: 3, side: 'waist' },
  spmd: { level: 3, side: 'waist' },
  collectives: { level: 3, side: 'waist' },
  codegen: { level: 4, side: 'waist' },
  autotuning: { level: 3, side: 'waist' },
  ifrt: { level: 2, side: 'waist' },
  mcjax: { level: 2, side: 'waist' },
  pathways: { level: 2, side: 'waist' },
  interfaces: { level: 2, side: 'waist' },
  capstone: { level: 3, side: 'waist' },
}

const prefixed = (prefix: string, cells: Record<string, Cell>): Record<string, Cell> =>
  Object.fromEntries(Object.entries(cells).map(([id, cell]) => [`${prefix}:${id}`, cell]))

/** Every unit key on the site, with the one cell it belongs to. */
export const UNIT_CELLS: Record<string, Cell> = {
  ...prefixed('l', LAYER_CELLS),
  // A stage sits where its layer sits. track.ts already pins the two
  // together, so the pin is read here, never restated.
  ...Object.fromEntries(STAGES.map((s) => [`s:${s.id}`, LAYER_CELLS[s.layer]!])),
  ...prefixed('jax', JAX_CELLS),
  ...prefixed('xla', XLA_CELLS),
  ...prefixed('pt', PT_CELLS),
}

/** Every occupied cell, in descent order, with the unit keys that fill it. */
export interface CellUnits {
  level: number
  side: MapSide
  keys: string[]
}

/** One side of the descent, floor by floor: the progress model's columns. */
export interface Gauge {
  side: MapSide
  /** How the gauge names this side to a reader. */
  label: string
  /** Top floor first, each carrying the unit keys a reader has to finish. */
  floors: { level: number; keys: string[] }[]
}

const SIDE_LABELS: Record<MapSide, string> = {
  torch: 'PyTorch side',
  jax: 'JAX side',
  waist: 'the waist',
  gpu: 'GPU side',
  tpu: 'TPU side',
}

export type RouteId = 'kernels' | 'jax' | 'xla' | 'pytorch'

export interface MapUnit {
  key: string
  href: string
  title: string
  num: number
  route: RouteId
  cell: Cell
}

interface RouteSpec {
  id: RouteId
  label: string
  href: string
  /** What ProgressKit calls this path; the kernel track's value is empty. */
  progress: string
  note: string
  chapters: { key: string; href: string; num: number; title: string }[]
}

const ROUTE_SPECS: RouteSpec[] = [
  {
    id: 'kernels',
    label: 'TPU kernels',
    href: '/',
    progress: '',
    note: 'the layers of the stack in order, then the stages that build on them',
    chapters: PATH,
  },
  {
    id: 'jax',
    label: 'JAX mastery',
    href: '/jax',
    progress: 'jax',
    note: 'the JAX language, from arrays to a training run',
    chapters: JAX_PATH,
  },
  {
    id: 'xla',
    label: 'XLA mastery',
    href: '/xla',
    progress: 'xla',
    note: 'the compiler at the waist, and the runtime seam above it',
    chapters: XLA_PATH,
  },
  {
    id: 'pytorch',
    label: 'PyTorch mastery',
    href: '/pytorch',
    progress: 'pytorch',
    note: 'eager PyTorch, then the bridges that land it on a TPU',
    chapters: PT_PATH,
  },
]

export interface Route {
  id: RouteId
  label: string
  href: string
  progress: string
  note: string
  /** In course order. */
  units: MapUnit[]
  /** The cells the route passes through, first visit first. */
  cells: Cell[]
}

export const ROUTES: Route[] = ROUTE_SPECS.map((spec) => {
  const units: MapUnit[] = spec.chapters.map((c) => ({
    key: c.key,
    href: c.href,
    title: c.title,
    num: c.num,
    route: spec.id,
    cell: UNIT_CELLS[c.key]!,
  }))
  const cells: Cell[] = []
  for (const u of units) {
    if (!cells.some((c) => c.level === u.cell.level && c.side === u.cell.side)) cells.push(u.cell)
  }
  return { id: spec.id, label: spec.label, href: spec.href, progress: spec.progress, note: spec.note, units, cells }
})

/** Every unit on the map, in route order. */
export const MAP_UNITS: MapUnit[] = ROUTES.flatMap((r) => r.units)

/** The units in one cell, routes in nav order and chapters in course order. */
export const unitsAt = (level: number, side: MapSide): MapUnit[] =>
  MAP_UNITS.filter((u) => u.cell.level === level && u.cell.side === side)

/** The cell a unit or one of its lessons belongs to; lessons inherit. */
export const cellOf = (key: string): Cell | undefined =>
  UNIT_CELLS[key] ?? UNIT_CELLS[key.split(':').slice(0, 2).join(':')]

/** The routes that pass through a cell, for the overlay. */
export const routesAt = (level: number, side: MapSide): RouteId[] =>
  ROUTES.filter((r) => r.cells.some((c) => c.level === level && c.side === side)).map((r) => r.id)

/** Every cell that holds units, top floor first, waist before the right side. */
export const CELL_UNITS: CellUnits[] = LEVELS.flatMap((level) =>
  (['torch', 'waist', 'jax', 'gpu', 'tpu'] as MapSide[])
    .map((side) => ({ level: level.level, side, keys: unitsAt(level.level, side).map((u) => u.key) }))
    .filter((c) => c.keys.length > 0),
)

// A descent runs down one side and through the waist, because the waist is
// where both frameworks meet and where both machines come from. So a side's
// column is its own cells plus the shared ones, and a floor is everything the
// reader has to finish to say they read that far down this side.
const gaugeFor = (side: MapSide): Gauge => ({
  side,
  label: SIDE_LABELS[side],
  floors: LEVELS.map((level) => ({
    level: level.level,
    keys: [...unitsAt(level.level, 'waist'), ...unitsAt(level.level, side)].map((u) => u.key),
  })).filter((f) => f.keys.length > 0),
})

/** The sides the depth gauge reads: the ones that carry units of their own. */
export const GAUGES: Gauge[] = (['torch', 'jax', 'gpu', 'tpu'] as MapSide[])
  .filter((side) => MAP_UNITS.some((u) => u.cell.side === side))
  .map(gaugeFor)

/** Layer ids in map order, so the map and the descent agree. */
/** Floors with no unit of their own whose subject the site teaches from a
    unit pinned elsewhere. The cell links the lessons instead of standing
    empty; navigation only, the lesson keeps the one telling. */
export interface CrossListing {
  title: string
  href: string
}
export const CROSS_LISTINGS: Record<string, CrossListing[]> = {
  '4/gpu': [{ title: 'Triton, end to end', href: '/xla/codegen/triton-end-to-end' }],
  '5/gpu': [{ title: 'The ISA contract: PTX and SASS', href: '/s/machine/the-isa-contract' }],
  '6/gpu': [
    { title: 'The GPU chip', href: '/s/machine/gpu-chip' },
    { title: 'Inside the SM, multiplied out', href: '/s/machine/inside-the-sm' },
    { title: 'The die and the reticle', href: '/s/machine/die-and-reticle' },
  ],
  '7/gpu': [
    { title: 'The scale-up domain', href: '/s/machine/scale-up' },
    { title: 'The GPU fabric', href: '/l/ici/gpu-fabric' },
  ],
}
export const crossListingsAt = (level: number, side: MapSide): CrossListing[] =>
  CROSS_LISTINGS[`${level}/${side}`] ?? []

export const LAYER_IDS: string[] = LAYERS.map((l) => l.id)
