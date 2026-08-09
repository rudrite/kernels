import { describe, expect, it } from 'vitest'
import { ALL_UNIT_LESSONS, LESSON_MASTERY, lessonKey } from './index'
import { LESSON_DIAGRAMS } from './diagrams'
import { PATH } from '../../lib/path'
import { XLA_PATH } from '../xla/track'

// Every prose field the lesson pages render; code and table cells are data.
const proseOf = () => {
  const out: string[] = []
  for (const u of ALL_UNIT_LESSONS)
    for (const l of u.lessons) {
      out.push(l.title, l.lede, l.goal)
      for (const sec of l.sections) out.push(sec.h, ...sec.ps)
      for (const r of l.readings) out.push(r.label, r.note)
    }
  return out
}

describe('the lesson layer structure', () => {
  it('homes every unit key in a real course chapter', () => {
    const chapterKeys = new Set([...PATH.map((c) => c.key), ...XLA_PATH.map((c) => c.key)])
    for (const u of ALL_UNIT_LESSONS) expect(chapterKeys.has(u.unit), u.unit).toBe(true)
  })

  it('keeps unit keys unique and lesson ids kebab, unique, and numbered in order', () => {
    const units = ALL_UNIT_LESSONS.map((u) => u.unit)
    expect(new Set(units).size).toBe(units.length)
    for (const u of ALL_UNIT_LESSONS) {
      const ids = u.lessons.map((l) => l.id)
      expect(new Set(ids).size, u.unit).toBe(ids.length)
      u.lessons.forEach((l, i) => {
        expect(l.id, `${u.unit}/${l.id}`).toMatch(/^[a-z][a-z-]*$/)
        expect(l.num, `${u.unit}/${l.id}`).toBe(i + 1)
        // a lesson id must not shadow its unit's URL segment
        expect(l.id, `${u.unit}/${l.id}`).not.toBe(u.unit.split(':').pop())
      })
    }
  })

  it('teaches in depth: lede, goal, and a real body (own sections or a guide slice)', () => {
    for (const u of ALL_UNIT_LESSONS)
      for (const l of u.lessons) {
        expect(l.lede.length, l.id).toBeGreaterThan(30)
        expect(l.goal.length, l.id).toBeGreaterThan(40)
        if (l.guide) expect(l.guide.sections.length, l.id).toBeGreaterThanOrEqual(1)
        else expect(l.sections.length, l.id).toBeGreaterThanOrEqual(3)
        for (const sec of l.sections) {
          expect(sec.ps.length, `${l.id} · ${sec.h}`).toBeGreaterThanOrEqual(1)
          for (const par of sec.ps) {
            if (par.startsWith('>> ')) continue // pull quotes are short on purpose
            expect(par.length, `${l.id} · ${sec.h}`).toBeGreaterThan(60)
          }
        }
      }
  })

  it('slices guides without losing a section: valid indices, one claim per section, covered guides whole', () => {
    const guides = import.meta.glob('../guides/*.json', { eager: true }) as Record<
      string,
      { default: { sections: unknown[] } }
    >
    // claims are global: a guide's sections may be told across units, but
    // each section is told exactly once
    const claimedBy = new Map<string, Set<number>>()
    for (const u of ALL_UNIT_LESSONS)
      for (const l of u.lessons) {
        if (!l.guide) continue
        const g = guides[`../guides/${l.guide.id}.json`]
        expect(g, `${u.unit}/${l.id} · guide ${l.guide.id}`).toBeDefined()
        const claimed = claimedBy.get(l.guide.id) ?? new Set<number>()
        claimedBy.set(l.guide.id, claimed)
        expect(l.guide.sections.length, `${u.unit}/${l.id}`).toBeGreaterThanOrEqual(1)
        for (const i of l.guide.sections) {
          expect(i, `${u.unit}/${l.id} · section ${i}`).toBeLessThan(g!.default.sections.length)
          expect(claimed.has(i), `${u.unit}/${l.id} · section ${i} claimed twice`).toBe(false)
          claimed.add(i)
        }
      }
    // coversGuide on a unit asserts its own layer's guide is fully told in
    // lessons somewhere; the hub then renders no inline remainder
    for (const u of ALL_UNIT_LESSONS) {
      if (!u.coversGuide) continue
      const layerId = u.unit.split(':').pop()!
      const total = guides[`../guides/${layerId}.json`]
      expect(total, `${u.unit} covers a guide that does not exist`).toBeDefined()
      const claimed = claimedBy.get(layerId) ?? new Set<number>()
      expect([...claimed].sort((a, b) => a - b), `${u.unit} covers ${layerId}`).toEqual(
        Array.from({ length: total!.default.sections.length }, (_, i) => i),
      )
    }
  })

  it('cites at least two readings per lesson, all https', () => {
    for (const u of ALL_UNIT_LESSONS)
      for (const l of u.lessons) {
        expect(l.readings.length, l.id).toBeGreaterThanOrEqual(2)
        for (const r of l.readings) expect(r.url, l.id).toMatch(/^https:\/\//)
      }
  })

  it('keeps tables rectangular and slots resolvable', () => {
    const exhibitIds = new Set<string>()
    const astro = import.meta.glob('../../components/LessonPage.astro', { query: '?raw', import: 'default', eager: true })
    for (const raw of Object.values(astro) as string[]) {
      for (const m of raw.matchAll(/^ {2}(\w+): \{\n {4}d: 'EX/gm)) exhibitIds.add(m[1]!)
    }
    expect(exhibitIds.size).toBeGreaterThanOrEqual(3)
    for (const u of ALL_UNIT_LESSONS)
      for (const l of u.lessons)
        for (const sec of l.sections) {
          if (sec.table) for (const row of sec.table.rows) expect(row.length, `${l.id} · ${sec.h}`).toBe(sec.table.cols.length)
          if (sec.diagram) expect(LESSON_DIAGRAMS[sec.diagram], `${l.id} · ${sec.h} · ${sec.diagram}`).toBeDefined()
          if (sec.exhibit) expect(exhibitIds.has(sec.exhibit), `${l.id} · ${sec.h} · ${sec.exhibit}`).toBe(true)
        }
  })

  it('folds hold more than their excerpts and link with absolute paths', () => {
    for (const u of ALL_UNIT_LESSONS)
      for (const l of u.lessons)
        for (const sec of l.sections) {
          const full = sec.code?.full
          if (!full) continue
          const where = `${l.id} · ${sec.h}`
          expect(full.text.split('\n').length, where).toBeGreaterThan(sec.code!.text.split('\n').length)
          if (full.href) expect(full.href, where).toMatch(/^\//)
        }
  })

  it('keeps checks well-formed: two or three per lesson, real questions, real answers', () => {
    for (const u of ALL_UNIT_LESSONS)
      for (const l of u.lessons) {
        if (!l.check) continue
        expect(l.check.length, l.id).toBeGreaterThanOrEqual(2)
        expect(l.check.length, l.id).toBeLessThanOrEqual(3)
        for (const c of l.check) {
          expect(c.q.length, l.id).toBeGreaterThan(20)
          expect(c.q.trim().endsWith('?'), `${l.id} · ${c.q}`).toBe(true)
          expect(c.a.length, l.id).toBeGreaterThan(40)
        }
      }
  })

  it('covers every lesson in the mastery ledger with a read item first', () => {
    for (const u of ALL_UNIT_LESSONS)
      for (const l of u.lessons) {
        const items = LESSON_MASTERY[lessonKey(u.unit, l.id)]
        expect(items, l.id).toBeDefined()
        expect(items![0]!.id, l.id).toBe('read')
      }
  })

  it('anchors every work-item href to a real section slug or absolute path', () => {
    const slug = (h: string) => h.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
    for (const u of ALL_UNIT_LESSONS)
      for (const l of u.lessons) {
        // template-emitted anchors exist on every lesson page
        const slugs = new Set([...l.sections.map((sec) => slug(sec.h)), 'check', 'readings'])
        for (const w of l.work ?? []) {
          if (!w.href) continue
          if (w.href.startsWith('#')) expect(slugs.has(w.href.slice(1)), `${l.id} · ${w.id} · ${w.href}`).toBe(true)
          else expect(w.href, `${l.id} · ${w.id}`).toMatch(/^(\/|https:\/\/)/)
        }
      }
  })

  it('prose no longer speaks in shelf language', () => {
    for (const text of proseOf()) {
      expect(/this series|series page|\bpage \d\b/i.test(text), text).toBe(false)
    }
  })
})

describe('the lesson voice rules', () => {
  it('carries no em-dashes or double-hyphen dashes in prose', () => {
    for (const text of proseOf()) {
      expect(text.includes('—'), text).toBe(false)
      expect(/\s--\s/.test(text), text).toBe(false)
    }
  })
})
