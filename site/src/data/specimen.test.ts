// The specimen is committed output, not something the site can regenerate,
// so the checks here are about the join between the capture and the page: a
// manifest that still parses, an artifact behind every level that claims to
// have one, and an explorer showing exactly the levels the manifest lists.
// Delete an artifact file or quietly drop a pending level from the page and
// this is what notices.
import { describe, expect, it } from 'vitest'
import { GRAMMAR, SPECIMEN, SPECIMEN_LEVELS } from './specimen'
import { LEVELS } from './map'

const captured = SPECIMEN_LEVELS.filter((l) => l.status === 'captured')
const pending = SPECIMEN_LEVELS.filter((l) => l.status === 'pending')

const pageSource = Object.values(
  import.meta.glob('../pages/specimen.astro', { query: '?raw', import: 'default', eager: true }),
)[0] as string

describe('the specimen manifest', () => {
  it('names the command and the versions that produced it', () => {
    expect(SPECIMEN.capture.command).toBe('python3 bench/specimen/capture.py')
    expect(SPECIMEN.capture.versions.jax).toMatch(/^\d+\.\d+\.\d+$/)
    expect(SPECIMEN.capture.versions.jaxlib).toMatch(/^\d+\.\d+\.\d+$/)
    expect(SPECIMEN.capture.versions.python).toMatch(/^\d+\.\d+\.\d+$/)
    expect(SPECIMEN.capture.scrub.length).toBeGreaterThan(40)
  })

  it('carries no absolute path, which the firewall would block anyway', () => {
    const all = JSON.stringify(SPECIMEN) + SPECIMEN_LEVELS.map((l) => l.text).join('\n')
    expect(all).not.toMatch(/(?<![\w.])\/(Users|home|private|var|tmp|opt|usr)\//)
  })

  it('descends: every level sits on a floor the hourglass has', () => {
    const floors = LEVELS.map((l) => l.level)
    for (const level of SPECIMEN_LEVELS) expect(floors, level.id).toContain(level.level)
    const seen = SPECIMEN_LEVELS.map((l) => l.level)
    expect([...seen].sort((a, b) => a - b)).toEqual(seen)
  })
})

describe('the captured artifacts', () => {
  it('has a file on disk behind every captured level, and it is not empty', () => {
    expect(captured.length).toBeGreaterThan(0)
    for (const level of captured) {
      expect(level.file, level.id).toMatch(/^bench\/specimen\/artifacts\//)
      expect(level.text.trim().length, level.id).toBeGreaterThan(0)
    }
  })

  it('agrees with the manifest about how long each artifact is', () => {
    for (const level of captured) {
      expect(level.text.replace(/\n$/, '').split('\n').length, level.id).toBe(level.lines)
    }
  })

  it('renders every artifact language through a grammar the site has', () => {
    for (const level of captured) {
      expect(level.lang === 'mlir' || GRAMMAR[level.lang!] !== undefined, level.id).toBe(true)
    }
  })
})

describe('the pending levels', () => {
  it('claims no file and says what will produce each one', () => {
    expect(pending.length).toBeGreaterThan(0)
    for (const level of pending) {
      expect(level.file, level.id).toBeUndefined()
      expect(level.text, level.id).toBe('')
      expect(level.pending!.length, level.id).toBeGreaterThan(40)
    }
  })
})

describe('the explorer', () => {
  it('shows the manifest levels, all of them, in the order they were captured', () => {
    expect(SPECIMEN_LEVELS.map((l) => l.id)).toEqual(SPECIMEN.levels.map((l) => l.id))
  })

  it('hands the whole list to the explorer rather than a filtered one', () => {
    expect(pageSource).toMatch(/<ArtifactExplorer[^>]*levels=\{SPECIMEN_LEVELS\}/)
  })
})
