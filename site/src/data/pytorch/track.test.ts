import { describe, expect, it } from 'vitest'
import { PT_CHAPTERS, PT_MASTERY, PT_PATH } from './track'

const proseOf = () => {
  const out: string[] = []
  for (const c of PT_CHAPTERS) {
    out.push(c.title, c.lede, c.goal)
    for (const s of c.sections) {
      out.push(s.h, ...s.ps)
      if (s.code) out.push(s.code.caption)
    }
    for (const r of c.readings) out.push(r.label, r.note)
  }
  for (const items of Object.values(PT_MASTERY)) for (const i of items) out.push(i.label)
  return out
}

describe('the PyTorch path structure', () => {
  it('has twelve chapters, numbered in order, split i/ii at chapter 8', () => {
    expect(PT_CHAPTERS).toHaveLength(12)
    PT_CHAPTERS.forEach((c, i) => {
      expect(c.num).toBe(i + 1)
      expect(c.part).toBe(c.num <= 7 ? 'i' : 'ii')
    })
  })

  it('has unique kebab-case ids and derived /pytorch/ hrefs', () => {
    const ids = PT_CHAPTERS.map((c) => c.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const id of ids) expect(id).toMatch(/^[a-z][a-z-]*$/)
    expect(PT_PATH).toHaveLength(12)
    PT_PATH.forEach((p, i) => {
      expect(p.key).toBe(`pt:${PT_CHAPTERS[i]!.id}`)
      expect(p.href).toBe(`/pytorch/${PT_CHAPTERS[i]!.id}`)
    })
  })

  it('teaches in depth: goal, three-plus sections, real paragraphs', () => {
    for (const c of PT_CHAPTERS) {
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
    for (const c of PT_CHAPTERS) {
      expect(c.readings.length, c.id).toBeGreaterThanOrEqual(2)
      for (const r of c.readings) expect(r.url, c.id).toMatch(/^https:\/\//)
    }
  })
})

describe('superset of the official tutorials', () => {
  // The docs.pytorch.org Learn-the-Basics series plus the intermediate
  // tutorials and notes the path deepens. All verified reachable when the
  // path was built; every one must be assigned somewhere.
  const REQUIRED = [
    'https://docs.pytorch.org/tutorials/beginner/basics/intro.html',
    'https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html',
    'https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html',
    'https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html',
    'https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html',
    'https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html',
    'https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html',
    'https://docs.pytorch.org/docs/stable/notes/autograd.html',
    'https://docs.pytorch.org/docs/stable/notes/extending.html',
    'https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html',
    'https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html',
    'https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html',
    'https://docs.pytorch.org/docs/stable/export.html',
    'https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html',
    'https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html',
    'https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html',
    'https://docs.pytorch.org/docs/stable/distributed.html',
    'https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html',
    'https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html',
    'https://blog.ezyang.com/2019/05/pytorch-internals/',
    // the TPU bridge canon: this path lands on the site's home ground
    'https://github.com/pytorch/xla',
    'https://github.com/pytorch/xla/tree/master/torchax',
    'https://docs.pytorch.org/xla/',
    'https://cloud.google.com/tpu/docs/run-calculation-pytorch',
    'https://github.com/vllm-project/tpu-inference',
  ]

  it('cites every required page in some chapter', () => {
    const cited = new Set(PT_CHAPTERS.flatMap((c) => c.readings.map((r) => r.url)))
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

describe('the PyTorch mastery ledger', () => {
  it('covers every chapter with a read item first', () => {
    for (const c of PT_CHAPTERS) {
      const items = PT_MASTERY[`pt:${c.id}`]
      expect(items, c.id).toBeDefined()
      expect(items!.length, c.id).toBeGreaterThanOrEqual(2)
      expect(items![0]!.id, c.id).toBe('read')
    }
  })

  it('auto rules are streak-only and hrefs resolve to real shapes', () => {
    for (const [key, items] of Object.entries(PT_MASTERY)) {
      for (const item of items) {
        if (item.auto) expect(item.auto.type, `${key} · ${item.id}`).toBe('streak')
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
