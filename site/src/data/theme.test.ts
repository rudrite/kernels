// The dark tokens live twice: once under the prefers-color-scheme media
// query (guarded so a stored light choice wins) and once under an explicit
// data-theme choice. A drifted pair paints two different darks depending on
// how the reader arrived at the mode, so the two bodies must stay identical.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync(new URL('../styles/global.css', import.meta.url), 'utf8')

const body = (open: string): string => {
  const i = css.indexOf(open)
  expect(i, open).toBeGreaterThan(-1)
  const start = css.indexOf('{', i) + 1
  let depth = 1
  let j = start
  while (depth > 0 && j < css.length) {
    if (css[j] === '{') depth++
    if (css[j] === '}') depth--
    j++
  }
  return css.slice(start, j - 1).trim()
}

describe('the scheme toggle', () => {
  it('keeps the two dark token bodies byte-identical', () => {
    expect(body(":root[data-theme='dark']")).toBe(body(":root:not([data-theme='light'])"))
  })

  it('guards the media block so an explicit light choice wins', () => {
    const media = css.indexOf('@media (prefers-color-scheme: dark)')
    const guard = css.indexOf(":root:not([data-theme='light'])", media)
    expect(guard).toBeGreaterThan(media)
    expect(guard).toBeLessThan(css.indexOf(':root[data-theme=', media))
  })
})
