// A slice is a quote, so the only way it can go wrong is by quoting
// something the machine did not write. These checks byte-compare every
// excerpt against the committed artifact it claims, refuse a level the
// manifest has not captured, and hold the framing sentences to the same
// voice bans the gate runs over prose. Widen a range past the end of a file,
// point a floor at a pending capture, or slip a dash into a framing line,
// and this is what notices.
import { describe, expect, it } from 'vitest'
import { RESOLVED_SLICES, SLICES, sliceFor } from './slices'
import { CELL_UNITS, MAP_UNITS, levelAt } from './map'
import { SPECIMEN_LEVELS } from './specimen'

const captured = new Map(SPECIMEN_LEVELS.filter((l) => l.status === 'captured').map((l) => [l.id, l]))

describe('the specimen slices', () => {
  it('quotes a contiguous run of the committed artifact, byte for byte', () => {
    expect(RESOLVED_SLICES.length).toBeGreaterThan(0)
    for (const slice of RESOLVED_SLICES) {
      const file = captured.get(slice.artifact)!.text.replace(/\n$/, '').split('\n')
      expect(slice.from, slice.artifact).toBeGreaterThanOrEqual(1)
      expect(slice.to, slice.artifact).toBeLessThanOrEqual(file.length)
      expect(slice.from, slice.artifact).toBeLessThanOrEqual(slice.to)
      expect(slice.text.split('\n'), slice.artifact).toEqual(file.slice(slice.from - 1, slice.to))
    }
  })

  it('keeps every excerpt short enough to orient and too short to teach', () => {
    for (const slice of RESOLVED_SLICES) {
      const lines = slice.text.split('\n').length
      expect(lines, slice.artifact).toBeGreaterThanOrEqual(6)
      expect(lines, slice.artifact).toBeLessThanOrEqual(10)
    }
  })

  it('names a level the manifest captured, never a pending one', () => {
    for (const slice of SLICES) {
      const entry = SPECIMEN_LEVELS.find((l) => l.id === slice.artifact)
      expect(entry, slice.artifact).toBeDefined()
      expect(entry!.status, slice.artifact).toBe('captured')
      expect(entry!.level, slice.artifact).toBe(slice.level)
    }
  })

  it('sits on a floor the hourglass draws and fills with units', () => {
    for (const slice of SLICES) {
      expect(levelAt(slice.level), slice.artifact).toBeDefined()
      expect(
        CELL_UNITS.some((c) => c.level === slice.level && c.side === slice.side),
        `${slice.level}/${slice.side}`,
      ).toBe(true)
    }
  })

  it('claims one floor once, so units on a floor cannot disagree', () => {
    const cells = SLICES.map((s) => `${s.level}:${s.side}`)
    expect(new Set(cells).size).toBe(cells.length)
  })

  it('links into the explorer at the level it quotes, and says what produced it', () => {
    for (const slice of RESOLVED_SLICES) {
      expect(slice.href).toBe(`/specimen#axp-${slice.artifact}`)
      expect(slice.provenance, slice.artifact).toContain(slice.entry.file!)
      expect(slice.provenance, slice.artifact).toMatch(/\d+\.\d+\.\d+/)
    }
  })

  it('frames each excerpt in one sentence, in the voice the gate enforces', () => {
    for (const slice of SLICES) {
      const line = slice.framing
      expect(line.length, slice.artifact).toBeGreaterThan(40)
      expect(line.length, slice.artifact).toBeLessThan(160)
      expect(line.endsWith('.'), slice.artifact).toBe(true)
      expect(line, slice.artifact).not.toMatch(/—|\s--\s|\s–\s/)
      expect(line, slice.artifact).not.toMatch(/here'?s the (catch|kicker|magic|trick|secret)|the whole trick/i)
      expect(line, slice.artifact).not.toMatch(/^[A-Z][\w ]{0,20}:/)
    }
  })

  it('varies how the sentences open, rather than stamping one shape', () => {
    const openers = SLICES.map((s) => s.framing.split(' ').slice(0, 2).join(' ').toLowerCase())
    expect(new Set(openers).size).toBe(openers.length)
  })
})

describe('the lookup a chapter template calls', () => {
  it('gives a chapter the slice of its own cell, and nothing else', () => {
    for (const unit of MAP_UNITS) {
      const slice = sliceFor(unit.key)
      const expected = SLICES.find((s) => s.level === unit.cell.level && s.side === unit.cell.side)
      expect(slice?.artifact, unit.key).toBe(expected?.artifact)
    }
  })

  it('stays silent on the floors the capture never reached', () => {
    const pendingFloors = MAP_UNITS.filter((u) => !SLICES.some((s) => s.level === u.cell.level && s.side === u.cell.side))
    expect(pendingFloors.length).toBeGreaterThan(0)
    for (const unit of pendingFloors) expect(sliceFor(unit.key), unit.key).toBeUndefined()
    // The GPU side's rows are cited neighbor captures, not the specimen
    // program, so no unit on that side may show a "your program" slice.
    for (const unit of MAP_UNITS.filter((u) => u.cell.side === 'gpu')) {
      expect(sliceFor(unit.key), unit.key).toBeUndefined()
    }
  })

  it('lets a lesson inherit its chapter slice, the way the map does', () => {
    expect(sliceFor('jax:tracing')?.artifact).toBe('jaxpr')
    expect(sliceFor('jax:tracing:1')?.artifact).toBe('jaxpr')
  })

  it('is wired into all five chapter templates, under the mastery panel', () => {
    const templates = import.meta.glob('../pages/*/[[]id[]].astro', { query: '?raw', import: 'default', eager: true })
    const sources = Object.entries(templates) as [string, string][]
    expect(sources.length).toBe(5)
    for (const [path, src] of sources) {
      expect(src, path).toMatch(/<MasteryWork[^>]*\/>\s*\n\s*<SpecimenSlice unit=/)
    }
  })
})
