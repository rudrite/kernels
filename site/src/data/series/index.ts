// The series shelf: ordered runs of full pages that take one subject past
// its path chapter. A series deepens a path; it never replaces one. Pages
// declare what they deepen and link back. The shape mirrors the sibling
// tracks (JSON-shaped TS, '>> ' pull quotes) so authoring carries over.
import type { WorkItem } from '../mastery'
import { MACHINE_SERIES } from './machine'
import { PALLAS_SERIES } from './pallas'

export interface SeriesCode {
  caption: string
  text: string
  /** Highlighter grammar; 'mlir' renders through MlirCode, not Shiki. */
  lang?: 'python' | 'haskell' | 'c' | 'mlir' | 'text'
}
export interface SeriesTable {
  caption: string
  cols: string[]
  rows: string[][]
}
export interface SeriesSection {
  h: string
  /** Paragraphs; a string starting '>> ' renders as a pull quote. */
  ps: string[]
  code?: SeriesCode
  table?: SeriesTable
  /** Id into SERIES_DIAGRAMS, rendered after the paragraphs. */
  diagram?: string
  /** Id into the page template's instrument roster, rendered after the diagram. */
  exhibit?: string
}
export interface SeriesReading {
  label: string
  url: string
  note: string
}
export interface SeriesPage {
  id: string
  num: number
  title: string
  lede: string
  goal: string
  sections: SeriesSection[]
  readings: SeriesReading[]
  /** Mastery work past the derived read item. */
  work?: WorkItem[]
}
export interface Series {
  id: string
  title: string
  lede: string
  /** The path chapters this series goes past; every page links back. */
  deepens: { label: string; href: string }[]
  pages: SeriesPage[]
}

export const ALL_SERIES: Series[] = [MACHINE_SERIES, PALLAS_SERIES]

// Every series page is mastery work under the 'sr:' prefix: the derived
// read item first, then whatever the page adds.
export const SR_MASTERY: Record<string, WorkItem[]> = Object.fromEntries(
  ALL_SERIES.flatMap((s) =>
    s.pages.map((p) => [
      `sr:${s.id}:${p.id}`,
      [{ id: 'read', label: `read the page: ${p.title}` }, ...(p.work ?? [])],
    ]),
  ),
)
