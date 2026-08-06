import { describe, expect, it } from 'vitest'
import { ALL_SERIES, SR_MASTERY } from './index'
import { SERIES_DIAGRAMS } from './diagrams'

// Every prose field the series pages render; code and table cells are data.
const proseOf = () => {
  const out: string[] = []
  for (const s of ALL_SERIES) {
    out.push(s.title, s.lede)
    for (const p of s.pages) {
      out.push(p.title, p.lede, p.goal)
      for (const sec of p.sections) out.push(sec.h, ...sec.ps)
      for (const r of p.readings) out.push(r.label, r.note)
    }
  }
  return out
}

describe('the series shelf structure', () => {
  it('has unique kebab-case series ids and page ids, pages numbered in order', () => {
    const sids = ALL_SERIES.map((s) => s.id)
    expect(new Set(sids).size).toBe(sids.length)
    for (const s of ALL_SERIES) {
      expect(s.id).toMatch(/^[a-z][a-z-]*$/)
      const ids = s.pages.map((p) => p.id)
      expect(new Set(ids).size, s.id).toBe(ids.length)
      s.pages.forEach((p, i) => {
        expect(p.id, `${s.id}/${p.id}`).toMatch(/^[a-z][a-z-]*$/)
        expect(p.num, `${s.id}/${p.id}`).toBe(i + 1)
      })
    }
  })

  it('teaches in depth: lede, goal, three-plus sections, real paragraphs', () => {
    for (const s of ALL_SERIES) {
      expect(s.lede.length, s.id).toBeGreaterThan(30)
      expect(s.deepens.length, s.id).toBeGreaterThanOrEqual(1)
      for (const d of s.deepens) expect(d.href, s.id).toMatch(/^\//)
      for (const p of s.pages) {
        expect(p.lede.length, p.id).toBeGreaterThan(30)
        expect(p.goal.length, p.id).toBeGreaterThan(40)
        expect(p.sections.length, p.id).toBeGreaterThanOrEqual(3)
        for (const sec of p.sections) {
          expect(sec.ps.length, `${p.id} · ${sec.h}`).toBeGreaterThanOrEqual(1)
          for (const par of sec.ps) {
            if (par.startsWith('>> ')) continue // pull quotes are short on purpose
            expect(par.length, `${p.id} · ${sec.h}`).toBeGreaterThan(60)
          }
        }
      }
    }
  })

  it('cites at least two readings per page, all https', () => {
    for (const s of ALL_SERIES)
      for (const p of s.pages) {
        expect(p.readings.length, p.id).toBeGreaterThanOrEqual(2)
        for (const r of p.readings) expect(r.url, p.id).toMatch(/^https:\/\//)
      }
  })

  it('keeps tables rectangular and slots resolvable', () => {
    const exhibitIds = new Set<string>()
    const astro = import.meta.glob('../../pages/series/**/*.astro', { query: '?raw', import: 'default', eager: true })
    for (const raw of Object.values(astro) as string[]) {
      for (const m of raw.matchAll(/^ {2}(\w+): \{\n {4}d: 'EX/gm)) exhibitIds.add(m[1]!)
    }
    for (const s of ALL_SERIES)
      for (const p of s.pages)
        for (const sec of p.sections) {
          if (sec.table) for (const row of sec.table.rows) expect(row.length, `${p.id} · ${sec.h}`).toBe(sec.table.cols.length)
          if (sec.diagram) expect(SERIES_DIAGRAMS[sec.diagram], `${p.id} · ${sec.h} · ${sec.diagram}`).toBeDefined()
          if (sec.exhibit) expect(exhibitIds.has(sec.exhibit), `${p.id} · ${sec.h} · ${sec.exhibit}`).toBe(true)
        }
  })

  it('covers every page in the mastery ledger with a read item first', () => {
    for (const s of ALL_SERIES)
      for (const p of s.pages) {
        const items = SR_MASTERY[`sr:${s.id}:${p.id}`]
        expect(items, p.id).toBeDefined()
        expect(items![0]!.id, p.id).toBe('read')
      }
  })

  it('anchors every work-item href to a real section slug or absolute path', () => {
    const slug = (h: string) => h.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
    for (const s of ALL_SERIES)
      for (const p of s.pages) {
        const slugs = new Set(p.sections.map((sec) => slug(sec.h)))
        for (const w of p.work ?? []) {
          if (!w.href) continue
          if (w.href.startsWith('#')) expect(slugs.has(w.href.slice(1)), `${p.id} · ${w.id} · ${w.href}`).toBe(true)
          else expect(w.href, `${p.id} · ${w.id}`).toMatch(/^(\/|https:\/\/)/)
        }
      }
  })
})

describe('the series voice rules', () => {
  it('carries no em-dashes or double-hyphen dashes in prose', () => {
    for (const text of proseOf()) {
      expect(text.includes('—'), text).toBe(false)
      expect(/\s--\s/.test(text), text).toBe(false)
    }
  })
})
