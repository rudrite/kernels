import { describe, expect, it } from 'vitest'
import { XLA_CHAPTERS, XLA_MASTERY, XLA_PATH } from './track'

const proseOf = () => {
  const out: string[] = []
  for (const c of XLA_CHAPTERS) {
    out.push(c.title, c.lede, c.goal)
    for (const s of c.sections) {
      out.push(s.h, ...s.ps)
      if (s.code) out.push(s.code.caption)
    }
    for (const r of c.readings) out.push(r.label, r.note)
  }
  for (const items of Object.values(XLA_MASTERY)) for (const i of items) out.push(i.label)
  return out
}

describe('the XLA path structure', () => {
  it('has fifteen chapters, numbered in order, split i/ii at chapter 11', () => {
    expect(XLA_CHAPTERS).toHaveLength(15)
    XLA_CHAPTERS.forEach((c, i) => {
      expect(c.num).toBe(i + 1)
      expect(c.part).toBe(c.num <= 10 ? 'i' : 'ii')
    })
  })

  it('has unique kebab-case ids and derived /xla/ hrefs', () => {
    const ids = XLA_CHAPTERS.map((c) => c.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const id of ids) expect(id).toMatch(/^[a-z][a-z-]*$/)
    expect(XLA_PATH).toHaveLength(15)
    XLA_PATH.forEach((p, i) => {
      expect(p.key).toBe(`xla:${XLA_CHAPTERS[i]!.id}`)
      expect(p.href).toBe(`/xla/${XLA_CHAPTERS[i]!.id}`)
    })
  })

  it('teaches in depth: goal, three-plus sections, real paragraphs', () => {
    for (const c of XLA_CHAPTERS) {
      expect(c.lede.length, c.id).toBeGreaterThan(30)
      expect(c.goal.length, c.id).toBeGreaterThan(40)
      expect(c.sections.length, c.id).toBeGreaterThanOrEqual(3)
      for (const s of c.sections) {
        expect(s.ps.length, `${c.id} · ${s.h}`).toBeGreaterThanOrEqual(1)
        for (const p of s.ps) {
          if (p.startsWith('>> ')) continue // pull quotes are short on purpose
          expect(p.length, `${c.id} · ${s.h}`).toBeGreaterThan(60)
        }
      }
    }
  })

  it('cites at least two public readings per chapter, all https', () => {
    for (const c of XLA_CHAPTERS) {
      expect(c.readings.length, c.id).toBeGreaterThanOrEqual(2)
      for (const r of c.readings) expect(r.url, c.id).toMatch(/^https:\/\//)
    }
  })
})

describe('the required public source set', () => {
  // The canon this path is grounded in; all public, all verified reachable
  // when the path was built. The internal study notes behind the blueprint
  // are not citable and must never appear.
  const REQUIRED = [
    'https://openxla.org/xla/architecture',
    'https://openxla.org/xla/pjrt',
    'https://openxla.org/xla/pjrt/pjrt_integration',
    'https://openxla.org/xla/tools',
    'https://openxla.org/xla/operation_semantics',
    'https://openxla.org/xla/custom_call',
    'https://openxla.org/stablehlo/spec',
    'https://arxiv.org/abs/2203.12533',
    'https://arxiv.org/abs/2105.04663',
    'https://docs.jax.dev/en/latest/multi_process.html',
    'https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro',
    'https://github.com/openxla/xla/tree/main/xla/python/ifrt',
    'https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h',
  ]

  it('cites every required source in some chapter', () => {
    const cited = new Set(XLA_CHAPTERS.flatMap((c) => c.readings.map((r) => r.url)))
    for (const url of REQUIRED) expect(cited.has(url), url).toBe(true)
  })

  it('never references internal tooling or private paths', () => {
    // assembled so this test file itself never carries the banned strings
    const banned = new RegExp(['carto' + 'graph', '\\/Users\\/'].join('|'), 'i')
    for (const text of proseOf()) {
      expect(banned.test(text), text).toBe(false)
    }
  })
})

describe('the XLA mastery ledger', () => {
  it('covers every chapter with a read item first', () => {
    for (const c of XLA_CHAPTERS) {
      const items = XLA_MASTERY[`xla:${c.id}`]
      expect(items, c.id).toBeDefined()
      expect(items!.length, c.id).toBeGreaterThanOrEqual(2)
      expect(items![0]!.id, c.id).toBe('read')
    }
  })

  it('auto rules are streak or labs only, hrefs resolve to real shapes', () => {
    for (const [key, items] of Object.entries(XLA_MASTERY)) {
      for (const item of items) {
        if (item.auto) expect(["streak", "labs"].includes(item.auto.type), `${key} · ${item.id}`).toBe(true)
        if (item.href) expect(item.href, `${key} · ${item.id}`).toMatch(/^(\/|#|https:\/\/)/)
      }
    }
  })
})

describe('the voice rules', () => {
  it('carries no em-dashes or double-hyphen dashes in prose', () => {
    for (const text of proseOf()) {
      expect(text.includes('—'), text).toBe(false)
      expect(/\s--\s/.test(text), text).toBe(false)
    }
  })
})
