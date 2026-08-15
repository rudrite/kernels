// The place-the-artifact corpus: dumps this repo already owns, each carrying
// the cell of the hourglass it came out of. Nothing here is written by hand
// and nothing is synthesized. Every excerpt is the opening lines of a
// committed specimen artifact or of a generated corpus the lessons already
// quote, verbatim, and an artifact whose floor the map does not draw stays
// out rather than being filed somewhere plausible: the CPU LLVM IR pair is
// real output of a real run, and no cell on this hourglass claims it.
//
// The map is imported here, not in the drill: this module runs at build time
// and hands the island a small table, so the chapter prose behind the map
// never reaches the browser (docs/plans/descent-revamp.md, phase M3 step 2).
import { CELL_UNITS, LEVELS, type MapSide } from './map'
import { SPECIMEN, SPECIMEN_LEVELS } from './specimen'
import irCorpus from './ir-corpus.json'
import cfCorpus from './cf-corpus.json'
import hloPairs from './hlo-pairs.json'
import mosaicCorpus from './mosaic-corpus.json'

export interface PlaceItem {
  id: string
  /** What the artifact is, told only once the reader has answered. */
  title: string
  /** What produced it, in the register the exhibits use. */
  provenance: string
  /** The excerpt, verbatim. */
  lines: string[]
  level: number
  side: MapSide
}

/** A cell the reader can point at: every occupied cell of the hourglass. */
export interface PlaceCell {
  level: number
  side: MapSide
  /** The column head the map prints for this cell. */
  label: string
}

const EXCERPT_LINES = 12

// A capture writes a banner over some dumps and the specimen source opens
// with a docstring naming the floors it visits, either of which would answer
// the question before it was asked. The excerpt starts at the artifact
// itself; the lines it does start at are untouched.
const artifactStart = (lines: string[]): string[] => {
  const banner = lines.findIndex((l) => /^==.*==\s*$/.test(l))
  if (banner === 0) return lines.slice(1)
  if (lines[0]?.startsWith('"""')) {
    const close = lines.findIndex((l, i) => i > 0 && l.includes('"""'))
    if (close > 0) return lines.slice(close + 1).filter((l, i) => !(i === 0 && l.trim() === ''))
  }
  return lines
}

const excerpt = (lines: string[]): string[] => artifactStart(lines).slice(0, EXCERPT_LINES)

// Which column of the map each specimen lane belongs to. A lane the map has
// no column for resolves to null and its artifacts stay out of the drill.
const LANE_SIDE: Record<string, MapSide | null> = {
  'the program': 'waist',
  jax: 'jax',
  torch: 'torch',
  waist: 'waist',
  gpu: 'gpu',
  cpu: null,
}

const specimenItems: PlaceItem[] = SPECIMEN_LEVELS.filter((l) => l.status === 'captured')
  .filter((l) => LANE_SIDE[l.lane])
  .map((l) => ({
    id: `specimen-${l.id}`,
    title: `the specimen at L${l.level}, ${l.title}`,
    provenance: `${l.file} · ${SPECIMEN.capture.command} on ${SPECIMEN.capture.backend}`,
    lines: excerpt(l.text.split('\n')),
    level: l.level,
    side: LANE_SIDE[l.lane]!,
  }))

interface CorpusProgram {
  id: string
  title: string
  jaxpr: string[]
  stablehlo: string[]
}

const traced = (source: string, generator: string, programs: CorpusProgram[]): PlaceItem[] =>
  programs.flatMap((p) => [
    {
      id: `${source}-${p.id}-jaxpr`,
      title: `${p.title}, traced to a jaxpr`,
      provenance: `${source} · ${generator}`,
      lines: excerpt(p.jaxpr),
      level: 1,
      side: 'jax' as MapSide,
    },
    {
      id: `${source}-${p.id}-stablehlo`,
      title: `${p.title}, lowered to StableHLO`,
      provenance: `${source} · ${generator}`,
      lines: excerpt(p.stablehlo),
      level: 2,
      side: 'waist' as MapSide,
    },
  ])

const ir = irCorpus as { meta: { generated: string; jax: string }; programs: CorpusProgram[] }
const cf = cfCorpus as { meta: { generated: string; jax: string }; programs: CorpusProgram[] }
const hlo = hloPairs as {
  meta: { generated: string; chip: string }
  programs: { id: string; title: string; unopt: string[]; opt: string[] }[]
}
const mosaic = mosaicCorpus as {
  meta: { generated: string; jax: string }
  kernels: { id: string; title: string; source: string[] }[]
}

const hloItems: PlaceItem[] = hlo.programs.flatMap((p) => [
  {
    id: `hlo-${p.id}-unopt`,
    title: `${p.title}, the StableHLO the compiler was handed`,
    provenance: `ir-corpus pairs · ${hlo.meta.generated} on ${hlo.meta.chip}`,
    lines: excerpt(p.unopt),
    level: 2,
    side: 'waist' as MapSide,
  },
  {
    id: `hlo-${p.id}-opt`,
    title: `${p.title}, the optimized HLO the compiler decided on`,
    provenance: `ir-corpus pairs · ${hlo.meta.generated} on ${hlo.meta.chip}`,
    lines: excerpt(p.opt),
    level: 3,
    side: 'waist' as MapSide,
  },
])

// A Pallas kernel is Python, and telling it from the program at the top of
// the map is the point of having both in here: refs and grids instead of
// arrays and returns.
const pallasItems: PlaceItem[] = mosaic.kernels.map((k) => ({
  id: `mosaic-${k.id}-source`,
  title: `${k.title}, the Pallas kernel`,
  provenance: `mosaic corpus · ${k.title.includes('LAB') ? 'a track kernel' : 'a corpus kernel'} · ${mosaic.meta.generated}`,
  lines: excerpt(k.source),
  level: 4,
  side: 'tpu' as MapSide,
}))

/** Every artifact the drill can hand a reader, in descent order. */
export const PLACE_ITEMS: PlaceItem[] = [
  ...specimenItems,
  ...traced('ir corpus', `${ir.meta.generated} · jax ${ir.meta.jax}`, ir.programs),
  ...traced('control-flow corpus', `${cf.meta.generated} · jax ${cf.meta.jax}`, cf.programs),
  ...hloItems,
  ...pallasItems,
].sort((a, b) => a.level - b.level)

const labelFor = (level: number, side: MapSide): string => {
  const row = LEVELS.find((l) => l.level === level)!
  if (side === 'waist') return row.waist ?? row.title
  return (row.left.side === side ? row.left.label : row.right.label) || row.title
}

/** The cells the picker offers: what the map draws, occupied cells only. */
export const PLACE_CELLS: PlaceCell[] = CELL_UNITS.map((c) => ({
  level: c.level,
  side: c.side,
  label: labelFor(c.level, c.side),
}))
