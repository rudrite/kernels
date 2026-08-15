// The map lists the workshop, and a list that points at a station nobody
// hosts, an anchor nobody renders, or a title the page has since changed is
// worse than no list. So every entry is checked against the page it claims,
// against the map's cells, and against the counts the surfaces already know.
import { describe, expect, it } from 'vitest'
import { WORKSHOP, workshopAt } from './workshop'
import { CELL_UNITS, UNIT_CELLS, levelAt } from './map'
import { STAGES } from './track'
import mistakes from './mistakes.json'
import results from '../../../bench/results.json'

const pageSources = import.meta.glob('../pages/**/*.astro', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

const pageFor = (href: string): string | undefined => {
  const path = href.split('#')[0]!
  const key = Object.keys(pageSources).find((p) => p.endsWith(`../pages${path}.astro`) || p.endsWith(`../pages${path}/index.astro`))
  return key ? pageSources[key] : undefined
}

describe('the workshop on the map', () => {
  it('pins every entry to a unit the map places', () => {
    expect(WORKSHOP.length).toBeGreaterThan(15)
    for (const e of WORKSHOP) {
      expect(UNIT_CELLS[e.unit], e.label).toBeDefined()
      expect(e.cell, e.label).toEqual(UNIT_CELLS[e.unit])
      const level = levelAt(e.cell.level)
      expect([level!.left.side, level!.right.side, 'waist'], e.label).toContain(e.cell.side)
      expect(CELL_UNITS.some((c) => c.level === e.cell.level && c.side === e.cell.side), e.label).toBe(true)
    }
  })

  it('lands every entry in a cell the map draws with units in it', () => {
    for (const e of WORKSHOP) {
      expect(workshopAt(e.cell.level, e.cell.side).map((x) => x.label), e.label).toContain(e.label)
    }
  })

  it('links pages that exist, at anchors they render', () => {
    for (const e of WORKSHOP) {
      const page = pageFor(e.href)
      expect(page, `${e.label} → ${e.href}`).toBeDefined()
      const anchor = e.href.split('#')[1]
      if (anchor) expect(page!.includes(`id="${anchor}"`), `${e.label} → ${e.href}`).toBe(true)
    }
  })

  it('calls each station what its floor calls it', () => {
    for (const e of WORKSHOP.filter((x) => x.kind === 'gym')) {
      const page = pageFor(e.href)!
      expect(page.includes(`title="${e.label}"`), e.label).toBe(true)
      expect(page.includes(`designator="${e.meta}"`), e.meta).toBe(true)
    }
  })

  it('lists every station once, and the museum wing titles the museum uses', () => {
    const designators = WORKSHOP.filter((e) => e.kind === 'gym').map((e) => e.meta)
    expect(new Set(designators).size).toBe(designators.length)
    const museumIndex = pageFor('/mistakes')!
    for (const e of WORKSHOP.filter((x) => x.kind === 'museum')) {
      expect(museumIndex.includes(e.label), e.label).toBe(true)
    }
  })

  it('counts exhibits and records off the data, never off a caption', () => {
    const kernelsWing = WORKSHOP.find((e) => e.href === '/mistakes/kernels')!
    expect(kernelsWing.meta).toBe(`${mistakes.length} exhibits`)
    for (const e of WORKSHOP.filter((x) => x.kind === 'bench')) {
      const stage = STAGES.find((s) => `s:${s.id}` === e.unit)!
      const rows = (results as { stage: number }[]).filter((r) => r.stage === stage.num).length
      expect(e.meta, e.unit).toBe(`${rows} records`)
      expect(rows).toBeGreaterThan(0)
    }
  })

  it('carries no em-dashes or double-hyphen dashes in its labels', () => {
    for (const e of WORKSHOP) {
      expect(e.label.includes('—'), e.label).toBe(false)
      expect(/\s--\s/.test(e.label), e.label).toBe(false)
    }
  })
})
