// The specimen slices: which floors of the map open with a piece of the one
// program, and which piece. A unit's cell gives a level and a side; a slice
// exists for that pair only when the capture actually reached it, so the
// GPU floors and the two machine floors below the kernel show nothing rather
// than something plausible (docs/plans/descent-revamp.md, phase M2 step 2).
//
// Every excerpt is a contiguous run of a committed artifact, addressed by
// line number and never retyped. The framing sentence orients the reader and
// stops there: the artifact is explained on /specimen and taught in the
// lesson, once, in one place.
import { cellOf, type MapSide } from './map'
import { GRAMMAR, SPECIMEN, SPECIMEN_LEVELS, type SpecimenLevel } from './specimen'

export interface Slice {
  level: number
  side: MapSide
  /** The specimen level this floor quotes, by manifest id. */
  artifact: string
  /** First and last line of the run, 1-based and inclusive. */
  from: number
  to: number
  /** One sentence, placing the excerpt. It does not read the artifact. */
  framing: string
}

// One entry per occupied floor the capture reached. Floors sharing a slice
// share it because they share a program, not because the excerpt was close
// enough: the waist units all watch the same module get rewritten.
export const SLICES: Slice[] = [
  {
    level: 0,
    side: 'waist',
    artifact: 'source',
    from: 17,
    to: 25,
    framing: 'The stack starts here, with one attention block on sixteen rows and no batch axis.',
  },
  {
    level: 0,
    side: 'jax',
    artifact: 'source',
    from: 17,
    to: 25,
    framing: 'Every jaxpr, lowering and dump the JAX chapters read starts as this block.',
  },
  {
    level: 1,
    side: 'jax',
    artifact: 'jaxpr',
    from: 19,
    to: 28,
    framing: 'Tracing turns that block into this, with one call to jax.nn.softmax already spread across several primitives.',
  },
  {
    level: 1,
    side: 'torch',
    artifact: 'torch-lazy-ir',
    from: 9,
    to: 16,
    framing: 'Under torch_xla the same block becomes a lazy graph, one node per aten call, captured on a TPU runtime.',
  },
  {
    level: 2,
    side: 'waist',
    artifact: 'stablehlo',
    from: 3,
    to: 11,
    framing: 'This is what leaves JAX and crosses the seam, the form every chapter at the waist works on.',
  },
  {
    level: 3,
    side: 'waist',
    artifact: 'hlo-after-optimizations',
    from: 45,
    to: 53,
    framing: 'After the backend pipeline runs, the same block arrives fused and scheduled.',
  },
]

export interface ResolvedSlice extends Slice {
  /** The manifest entry the excerpt was cut from. */
  entry: SpecimenLevel
  /** The excerpt itself, verbatim. */
  text: string
  /** Shiki grammar, or 'mlir' where the site tints by hand. */
  lang: string
  /** The artifact file, its line range, and the versions that produced it. */
  provenance: string
  /** Deep link into the explorer, on the level this excerpt came from. */
  href: string
}

const versionsOf = (entry: SpecimenLevel): string =>
  Object.entries(entry.versions ?? SPECIMEN.capture.versions)
    .map(([name, v]) => `${name} ${v}`)
    .join(', ')

const resolve = (slice: Slice): ResolvedSlice => {
  const entry = SPECIMEN_LEVELS.find((l) => l.id === slice.artifact)
  if (!entry || entry.status !== 'captured') {
    throw new Error(`slices: ${slice.artifact} is not a captured specimen level`)
  }
  return {
    ...slice,
    entry,
    text: entry.text.replace(/\n$/, '').split('\n').slice(slice.from - 1, slice.to).join('\n'),
    lang: entry.lang === 'mlir' ? 'mlir' : GRAMMAR[entry.lang!] ?? 'txt',
    provenance: `${entry.file} · lines ${slice.from} to ${slice.to} of ${entry.lines} · ${versionsOf(entry)}`,
    href: `/specimen#axp-${entry.id}`,
  }
}

/** The slice a unit or one of its lessons opens with, where one is honest. */
export const sliceFor = (unitKey: string): ResolvedSlice | undefined => {
  const cell = cellOf(unitKey)
  if (!cell) return undefined
  const slice = SLICES.find((s) => s.level === cell.level && s.side === cell.side)
  return slice ? resolve(slice) : undefined
}

/** Every slice, resolved, for the tests and for anything that wants the set. */
export const RESOLVED_SLICES: ResolvedSlice[] = SLICES.map(resolve)
