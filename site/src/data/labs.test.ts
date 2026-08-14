// The lab cards on chapter pages name notebooks by designator, and the
// notebooks are the source of truth. Nothing else notices when the two
// drift, so this does: rename a notebook and its card's Colab link would
// 404 in silence.
//
// Sources are loaded through Vite rather than the filesystem, so the suite
// needs no node types and no working directory of its own.
import { describe, expect, it } from 'vitest'
import labs from './labs.generated.json'

const PATHS = ['jax', 'xla', 'pytorch', 'l']
const generated = labs as Record<string, { file: string; title: string }>

const pageSources = import.meta.glob('../pages/**/*.astro', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

/** Designators the chapter pages hand to LabCard, with what each claims. */
const carded = Object.entries(pageSources)
  .filter(([p]) => PATHS.some((d) => p.includes(`/pages/${d}/`)) && p.includes('[id].astro'))
  .flatMap(([page, src]) =>
    [...src.matchAll(/lab\('(LAB·[^']+)',\s*'([^']*)',\s*'[^']*',\s*'[^']*',\s*'([^']+)'\)/g)].map((m) => ({
      page,
      designator: m[1]!,
      title: m[2]!,
      file: m[3]!,
    })),
  )

/** Every notebook the three new paths ship, per the notebooks themselves. */
const shipped = Object.entries(generated).filter(([, l]) =>
  PATHS.some((d) => l.file.startsWith(`labs/${d}/`)),
)

describe('lab cards and notebooks', () => {
  it('cards every lab the three new paths ship', () => {
    expect(carded.length).toBe(shipped.length)
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
    const cardedFiles = new Set(carded.map((c) => c.file))
    for (const [designator, lab] of shipped) {
      expect(cardedFiles.has(lab.file), `${designator} (${lab.file}) has no card`).toBe(true)
    }
  })
})
