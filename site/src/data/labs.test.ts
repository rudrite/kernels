// The lab cards on chapter pages name notebooks by designator, and the
// notebooks are the source of truth. Nothing else notices when the two
// drift, so this does: rename a notebook and its card's Colab link would
// 404 in silence.
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import labs from './labs.generated.json'

// Resolved from this file, not the working directory, so the suite passes
// wherever vitest is invoked from.
const SITE = fileURLToPath(new URL('../..', import.meta.url))
const REPO = fileURLToPath(new URL('../../..', import.meta.url))

const PAGES = ['jax', 'xla', 'pytorch'].map((p) => join(SITE, 'src/pages', p, '[id].astro'))
const generated = labs as Record<string, { file: string; title: string }>

/** Designators the chapter pages hand to LabCard, with the notebook path each claims. */
const carded = PAGES.flatMap((page) => {
  const src = readFileSync(page, 'utf8')
  return [...src.matchAll(/lab\('(LAB·[^']+)',\s*'([^']*)',\s*'[^']*',\s*'[^']*',\s*'([^']+)'\)/g)].map((m) => ({
    page: page.replace(SITE, ''),
    designator: m[1]!,
    title: m[2]!,
    file: m[3]!,
  }))
})

describe('lab cards and notebooks', () => {
  it('cards every lab the three new paths ship', () => {
    expect(carded.length).toBe(12)
  })

  it('names a designator the notebooks actually register', () => {
    for (const c of carded) {
      expect(generated[c.designator], `${c.page} cards ${c.designator}`).toBeDefined()
    }
  })

  it('points at the file that notebook lives in', () => {
    for (const c of carded) {
      expect(generated[c.designator]!.file, c.designator).toBe(c.file)
    }
  })

  it('agrees with the notebook about its title', () => {
    for (const c of carded) {
      expect(generated[c.designator]!.title, c.designator).toBe(c.title)
    }
  })

  it('leaves no notebook in the new paths uncarded', () => {
    const onDisk = ['jax', 'xla', 'pytorch'].flatMap((d) =>
      readdirSync(join(REPO, 'labs', d)).filter((f) => f.endsWith('.ipynb')).map((f) => `labs/${d}/${f}`),
    )
    const cardedFiles = new Set(carded.map((c) => c.file))
    for (const f of onDisk) expect(cardedFiles.has(f), `${f} has no card`).toBe(true)
  })
})
