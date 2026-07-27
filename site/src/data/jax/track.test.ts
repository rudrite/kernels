import { describe, expect, it } from 'vitest'
import { JAX_CHAPTERS, JAX_MASTERY, JAX_PATH } from './track'

// Every prose field the pages render; code text is exempt (it is code).
const proseOf = () => {
  const out: string[] = []
  for (const c of JAX_CHAPTERS) {
    out.push(c.title, c.lede, c.goal)
    for (const s of c.sections) {
      out.push(s.h, ...s.ps)
      if (s.code) out.push(s.code.caption)
    }
    for (const r of c.readings) out.push(r.label, r.note)
  }
  for (const items of Object.values(JAX_MASTERY)) for (const i of items) out.push(i.label)
  return out
}

describe('the JAX path structure', () => {
  it('has twelve chapters, numbered in order, split i/ii at chapter 9', () => {
    expect(JAX_CHAPTERS).toHaveLength(12)
    JAX_CHAPTERS.forEach((c, i) => {
      expect(c.num).toBe(i + 1)
      expect(c.part).toBe(c.num <= 8 ? 'i' : 'ii')
    })
  })

  it('has unique kebab-case ids and derived /jax/ hrefs', () => {
    const ids = JAX_CHAPTERS.map((c) => c.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const id of ids) expect(id).toMatch(/^[a-z][a-z-]*$/)
    expect(JAX_PATH).toHaveLength(12)
    JAX_PATH.forEach((p, i) => {
      expect(p.key).toBe(`jax:${JAX_CHAPTERS[i]!.id}`)
      expect(p.href).toBe(`/jax/${JAX_CHAPTERS[i]!.id}`)
    })
  })

  it('teaches in depth: every chapter has a goal, three-plus sections, real paragraphs', () => {
    for (const c of JAX_CHAPTERS) {
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

  it('cites at least two readings per chapter, all https', () => {
    for (const c of JAX_CHAPTERS) {
      expect(c.readings.length, c.id).toBeGreaterThanOrEqual(2)
      for (const r of c.readings) expect(r.url, c.id).toMatch(/^https:\/\//)
    }
  })
})

describe('the JAX mastery ledger', () => {
  it('covers every chapter under its jax: key with a read item first', () => {
    for (const c of JAX_CHAPTERS) {
      const items = JAX_MASTERY[`jax:${c.id}`]
      expect(items, c.id).toBeDefined()
      expect(items!.length, c.id).toBeGreaterThanOrEqual(2)
      expect(items![0]!.id, c.id).toBe('read')
    }
  })

  it('auto rules are streak or labs only, no dead hrefs', () => {
    for (const [key, items] of Object.entries(JAX_MASTERY)) {
      for (const item of items) {
        if (item.auto) expect(["streak", "labs"].includes(item.auto.type), `${key} · ${item.id}`).toBe(true)
        if (item.href) expect(item.href, `${key} · ${item.id}`).toMatch(/^(\/|#|https:\/\/)/)
      }
    }
  })
})

describe('superset of the official tutorials', () => {
  // The docs.jax.dev tutorial sequence, plus the advanced guides it points
  // into. The path must cite every one of these somewhere; our chapters go
  // past them, never under them.
  const REQUIRED = [
    'https://docs.jax.dev/en/latest/quickstart.html',
    'https://docs.jax.dev/en/latest/key-concepts.html',
    'https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html',
    'https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html',
    'https://docs.jax.dev/en/latest/jit-compilation.html',
    'https://docs.jax.dev/en/latest/aot.html',
    'https://docs.jax.dev/en/latest/automatic-differentiation.html',
    'https://docs.jax.dev/en/latest/advanced-autodiff.html',
    'https://docs.jax.dev/en/latest/gradient-checkpointing.html',
    'https://docs.jax.dev/en/latest/automatic-vectorization.html',
    'https://docs.jax.dev/en/latest/control-flow.html',
    'https://docs.jax.dev/en/latest/working-with-pytrees.html',
    'https://docs.jax.dev/en/latest/random-numbers.html',
    'https://docs.jax.dev/en/latest/stateful-computations.html',
    'https://docs.jax.dev/en/latest/sharded-computation.html',
    'https://docs.jax.dev/en/latest/notebooks/shard_map.html',
    'https://docs.jax.dev/en/latest/debugging.html',
    'https://docs.jax.dev/en/latest/external-callbacks.html',
    'https://docs.jax.dev/en/latest/profiling.html',
    'https://docs.jax.dev/en/latest/multi_process.html',
    'https://docs.jax.dev/en/latest/faq.html#benchmarking-jax-code',
  ]

  it('cites every official tutorial page in some chapter', () => {
    const cited = new Set(JAX_CHAPTERS.flatMap((c) => c.readings.map((r) => r.url)))
    for (const url of REQUIRED) expect(cited.has(url), url).toBe(true)
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
