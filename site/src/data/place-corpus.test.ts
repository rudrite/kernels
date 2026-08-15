// The drill's one rule is that every excerpt it shows came out of a machine.
// So the checks here are about provenance and about the join with the map:
// each excerpt has to be a contiguous run of lines from a file this repo
// committed or a corpus it generated, and each answer has to be a cell the
// hourglass actually draws with units in it. Invent a dump, file an artifact
// on a floor that does not exist, or let the picker drift away from the map,
// and this is what notices.
import { describe, expect, it } from 'vitest'
import { PLACE_CELLS, PLACE_ITEMS } from './place-corpus'
import { CELL_UNITS, LEVELS, levelAt } from './map'
import { SPECIMEN_LEVELS } from './specimen'
import irCorpus from './ir-corpus.json'
import cfCorpus from './cf-corpus.json'
import hloPairs from './hlo-pairs.json'
import mosaicCorpus from './mosaic-corpus.json'

// Every block of machine output this repo holds, flattened. Nothing in the
// corpus may fall outside it.
const SOURCE_BLOCKS: string[][] = [
  ...SPECIMEN_LEVELS.filter((l) => l.status === 'captured').map((l) => l.text.split('\n')),
  ...(irCorpus as { programs: { jaxpr: string[]; stablehlo: string[] }[] }).programs.flatMap((p) => [p.jaxpr, p.stablehlo]),
  ...(cfCorpus as { programs: { jaxpr: string[]; stablehlo: string[] }[] }).programs.flatMap((p) => [p.jaxpr, p.stablehlo]),
  ...(hloPairs as { programs: { unopt: string[]; opt: string[] }[] }).programs.flatMap((p) => [p.unopt, p.opt]),
  ...(mosaicCorpus as { kernels: { source: string[] }[] }).kernels.map((k) => k.source),
]

const runsIn = (block: string[], lines: string[]): boolean =>
  block.some((_, i) => lines.every((l, j) => block[i + j] === l))

describe('the place-the-artifact corpus', () => {
  it('shows only lines a machine wrote, verbatim and in order', () => {
    expect(PLACE_ITEMS.length).toBeGreaterThan(20)
    for (const item of PLACE_ITEMS) {
      expect(item.lines.length, item.id).toBeGreaterThan(0)
      expect(SOURCE_BLOCKS.some((b) => runsIn(b, item.lines)), item.id).toBe(true)
    }
  })

  it('says what produced every excerpt', () => {
    for (const item of PLACE_ITEMS) {
      expect(item.provenance.length, item.id).toBeGreaterThan(10)
      expect(item.title.length, item.id).toBeGreaterThan(4)
    }
  })

  it('answers with a cell the hourglass draws and fills', () => {
    for (const item of PLACE_ITEMS) {
      const level = levelAt(item.level)
      expect(level, item.id).toBeDefined()
      expect([level!.left.side, level!.right.side, 'waist'], item.id).toContain(item.side)
      expect(
        CELL_UNITS.some((c) => c.level === item.level && c.side === item.side),
        item.id,
      ).toBe(true)
    }
  })

  it('leaves out the artifacts no cell claims, rather than filing them somewhere close', () => {
    const cpu = SPECIMEN_LEVELS.filter((l) => l.lane === 'cpu').map((l) => l.id)
    expect(cpu.length).toBeGreaterThan(0)
    for (const id of cpu) expect(PLACE_ITEMS.map((i) => i.id)).not.toContain(`specimen-${id}`)
  })

  it('spans the descent rather than drilling one floor', () => {
    const cells = new Set(PLACE_ITEMS.map((i) => `${i.level}:${i.side}`))
    expect(cells.size).toBeGreaterThanOrEqual(5)
    expect(new Set(PLACE_ITEMS.map((i) => i.side)).size).toBeGreaterThanOrEqual(3)
  })
})

describe('the picker', () => {
  it('offers exactly the occupied cells of the map, in descent order', () => {
    expect(PLACE_CELLS.map((c) => `${c.level}:${c.side}`)).toEqual(CELL_UNITS.map((c) => `${c.level}:${c.side}`))
  })

  it('labels each cell the way the map heads it', () => {
    for (const cell of PLACE_CELLS) {
      const row = LEVELS.find((l) => l.level === cell.level)!
      const expected =
        cell.side === 'waist'
          ? row.waist ?? row.title
          : (row.left.side === cell.side ? row.left.label : row.right.label) || row.title
      expect(cell.label, `${cell.level}/${cell.side}`).toBe(expected)
      expect(cell.label.length, `${cell.level}/${cell.side}`).toBeGreaterThan(0)
    }
  })

  it('offers more cells than the corpus can ask about, so a guess costs something', () => {
    const answered = new Set(PLACE_ITEMS.map((i) => `${i.level}:${i.side}`))
    expect(PLACE_CELLS.length).toBeGreaterThan(answered.size)
  })
})
