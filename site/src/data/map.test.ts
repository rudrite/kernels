// The map claims a cell for every unit on the site. Nothing else notices
// when a new chapter lands without one, or when a cell points at a level and
// side the hourglass does not have, so this does.
import { describe, expect, it } from 'vitest'
import { CELL_UNITS, CROSS_LISTINGS, GAUGES, LEVELS, MAP_UNITS, ROUTES, UNIT_CELLS, cellOf, levelAt, unitsAt, type MapSide } from './map'
import { STAGES } from './track'
import { PATH } from '../lib/path'
import { JAX_PATH } from './jax/track'
import { XLA_PATH } from './xla/track'
import { PT_PATH } from './pytorch/track'
import { ALL_UNIT_LESSONS, lessonHref, lessonKey } from './lessons'

const ALL_KEYS = [...PATH, ...JAX_PATH, ...XLA_PATH, ...PT_PATH].map((c) => c.key)

describe('the hourglass', () => {
  it('runs from the source to the pod, one entry per level', () => {
    expect(LEVELS).toHaveLength(8)
    LEVELS.forEach((l, i) => expect(l.level).toBe(i))
  })

  it('converges to a waist and diverges into two machines', () => {
    expect(LEVELS.filter((l) => l.waist).map((l) => l.level)).toEqual([0, 2, 3, 4])
    expect(LEVELS.filter((l) => l.seam).map((l) => l.level)).toEqual([2])
    for (const l of LEVELS) {
      expect([l.left.side, l.right.side], String(l.level)).toEqual(
        l.level <= 3 ? ['torch', 'jax'] : ['gpu', 'tpu'],
      )
    }
  })
})

describe('cells', () => {
  it('gives every unit across the four courses exactly one cell', () => {
    for (const key of ALL_KEYS) expect(UNIT_CELLS[key], key).toBeDefined()
    expect(Object.keys(UNIT_CELLS).sort()).toEqual([...ALL_KEYS].sort())
  })

  it('resolves every cell to a level and a side that level has', () => {
    for (const [key, cell] of Object.entries(UNIT_CELLS)) {
      const level = levelAt(cell.level)
      expect(level, key).toBeDefined()
      expect([level!.left.side, level!.right.side, 'waist'], key).toContain(cell.side)
    }
  })

  it('pins each stage where its layer sits, the pin track.ts already made', () => {
    for (const s of STAGES) expect(UNIT_CELLS[`s:${s.id}`], s.id).toEqual(UNIT_CELLS[`l:${s.layer}`])
  })

  it("lands every lesson in its unit's cell", () => {
    for (const u of ALL_UNIT_LESSONS) {
      for (const l of u.lessons) {
        expect(cellOf(lessonKey(u.unit, l.id)), `${u.unit}/${l.id}`).toEqual(UNIT_CELLS[u.unit])
      }
    }
  })

  it('finds every unit again through the cell it claims', () => {
    for (const u of MAP_UNITS) {
      expect(unitsAt(u.cell.level, u.cell.side).map((x) => x.key), u.key).toContain(u.key)
    }
  })
})

describe('routes', () => {
  it('draws all four courses, each over occupied cells only', () => {
    expect(ROUTES.map((r) => r.id)).toEqual(['kernels', 'jax', 'xla', 'pytorch'])
    for (const r of ROUTES) {
      expect(r.units.length, r.id).toBeGreaterThan(0)
      expect(r.cells.length, r.id).toBeGreaterThan(0)
      expect(r.href, r.id).toMatch(/^\//)
      for (const c of r.cells) expect(unitsAt(c.level, c.side).length, `${r.id} ${c.level}/${c.side}`).toBeGreaterThan(0)
    }
  })

  it('keeps the course order the path registries define', () => {
    const kernels = ROUTES.find((r) => r.id === 'kernels')!
    expect(kernels.units.map((u) => u.key)).toEqual(PATH.map((c) => c.key))
    expect(kernels.units.map((u) => u.href)).toEqual(PATH.map((c) => c.href))
  })

  it('visits each cell once per route', () => {
    for (const r of ROUTES) {
      const seen = r.cells.map((c) => `${c.level}/${c.side}`)
      expect(new Set(seen).size, r.id).toBe(seen.length)
    }
  })
})

describe('the depth gauge', () => {
  it('rolls every unit into exactly one cell, and no cell into thin air', () => {
    const rolled = CELL_UNITS.flatMap((c) => c.keys)
    expect(rolled.sort()).toEqual(MAP_UNITS.map((u) => u.key).sort())
    for (const cell of CELL_UNITS) expect(cell.keys.length, `${cell.level}/${cell.side}`).toBeGreaterThan(0)
  })

  it('lists the cells in descent order', () => {
    const levels = CELL_UNITS.map((c) => c.level)
    expect([...levels].sort((a, b) => a - b)).toEqual(levels)
  })

  it('gauges every side that carries units of its own, and none that does not', () => {
    const occupied = [...new Set(MAP_UNITS.map((u) => u.cell.side))].filter((s) => s !== 'waist')
    expect(GAUGES.map((g) => g.side).sort()).toEqual(occupied.sort())
    for (const g of GAUGES) expect(g.label.length, g.side).toBeGreaterThan(0)
  })

  it('walks each side down through the waist, one floor at a time', () => {
    for (const g of GAUGES) {
      const levels = g.floors.map((f) => f.level)
      expect([...levels].sort((a, b) => a - b), g.side).toEqual(levels)
      expect(new Set(levels).size, g.side).toBe(levels.length)
      for (const floor of g.floors) {
        const expected = [...unitsAt(floor.level, 'waist'), ...unitsAt(floor.level, g.side)].map((u) => u.key)
        expect(floor.keys, `${g.side} L${floor.level}`).toEqual(expected)
        expect(floor.keys.length, `${g.side} L${floor.level}`).toBeGreaterThan(0)
      }
    }
  })

  it('reaches the deepest floor its side occupies', () => {
    for (const g of GAUGES) {
      const deepest = Math.max(...MAP_UNITS.filter((u) => u.cell.side === g.side).map((u) => u.cell.level))
      expect(g.floors.map((f) => f.level), g.side).toContain(deepest)
    }
  })
})

describe('the voice rules', () => {
  it('carries no em-dashes or double-hyphen dashes in prose', () => {
    const prose = [
      ...LEVELS.flatMap((l) => [l.title, l.left.label, l.right.label, l.waist ?? '', l.seam ?? '']),
      ...ROUTES.flatMap((r) => [r.label, r.note]),
    ]
    for (const text of prose) {
      expect(text.includes('—'), text).toBe(false)
      expect(/\s--\s/.test(text), text).toBe(false)
    }
  })
})

describe('the cross-listings', () => {
  it('resolve to real lessons on cells that hold no unit', () => {
    for (const [cell, listings] of Object.entries(CROSS_LISTINGS)) {
      const [levelStr, side] = cell.split('/') as [string, MapSide]
      const level = Number(levelStr)
      expect(unitsAt(level, side).length, cell).toBe(0)
      for (const listing of listings) {
        const hit = ALL_UNIT_LESSONS.some((u) =>
          u.lessons.some((l) => lessonHref(u.unit, l.id) === listing.href),
        )
        expect(hit, `${cell} -> ${listing.href}`).toBe(true)
      }
    }
  })
})
