// Order and next/previous for series pages, derived from the one registry.
// Same contract as lib/path.ts: nothing else may define what comes next
// inside a series.
import { ALL_SERIES, type Series, type SeriesPage } from '../data/series'

export interface SeriesPageRef {
  key: string
  href: string
  num: number
  title: string
  part: 'i'
}

export const seriesById = (id: string): Series => {
  const s = ALL_SERIES.find((s) => s.id === id)
  if (!s) throw new Error(`unknown series: ${id}`)
  return s
}

export const seriesRefs = (s: Series): SeriesPageRef[] =>
  s.pages.map((p) => ({
    key: `sr:${s.id}:${p.id}`,
    href: `/series/${s.id}/${p.id}`,
    num: p.num,
    title: p.title,
    part: 'i' as const,
  }))

export const seriesPageAt = (
  seriesId: string,
  pageId: string,
): { series: Series; current: SeriesPage; prev?: SeriesPage; next?: SeriesPage } => {
  const series = seriesById(seriesId)
  const idx = series.pages.findIndex((p) => p.id === pageId)
  if (idx === -1) throw new Error(`unknown series page: ${seriesId}/${pageId}`)
  return { series, current: series.pages[idx]!, prev: series.pages[idx - 1], next: series.pages[idx + 1] }
}
