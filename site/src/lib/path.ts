// The single source of the path's order. Every next/previous control on the
// site derives from this list; nothing else may define what comes next.
import { LAYERS, STAGES } from '../data/track'

export interface PathChapter {
  key: string
  href: string
  num: number
  title: string
  part: 'i' | 'ii'
}

export const PATH: PathChapter[] = [
  ...LAYERS.map((l, i) => ({
    key: `l:${l.id}`,
    href: `/l/${l.id}`,
    num: i + 1,
    title: l.label,
    part: 'i' as const,
  })),
  ...STAGES.map((s, i) => ({
    key: `s:${s.id}`,
    href: `/s/${s.id}`,
    num: LAYERS.length + i + 1,
    title: `stage ${s.num} · ${s.title}`,
    part: 'ii' as const,
  })),
]

export const chapterAt = (key: string): { current: PathChapter; prev?: PathChapter; next?: PathChapter } => {
  const idx = PATH.findIndex((c) => c.key === key)
  if (idx === -1) throw new Error(`unknown path chapter: ${key}`)
  return { current: PATH[idx]!, prev: PATH[idx - 1], next: PATH[idx + 1] }
}
