// Lessons: the depth layer of a unit. A unit is a course chapter (its key is
// the chapter key: 'l:tpu', 's:pallas', 'xla:pjrt'); its lessons are ordered
// full pages nested under the chapter's URL. One canonical home per fact: a
// lesson belongs to exactly one unit, and the shelf that once held these
// pages is gone (docs/plans/true-mastery-courses.md).
import type { WorkItem } from '../mastery'
import { COMPILER_WALL_LESSONS } from './compiler-wall'
import { IR_DESCENT_LESSONS } from './ir-descent'
import { MACHINE_LESSONS } from './machine'
import { PALLAS_LESSONS } from './pallas'
import { TORCH_BRIDGE_LESSONS } from './torch-bridge'
import { XLA_LESSONS } from './xla-internals'

export interface LessonCode {
  caption: string
  text: string
  /** Highlighter grammar; 'mlir' renders through MlirCode, not Shiki. */
  lang?: 'python' | 'haskell' | 'c' | 'mlir' | 'text'
  /** The whole capture the excerpt was cut from, folded shut below the
      panel; href points at a page that works the same capture live. */
  full?: { text: string; label?: string; href?: string; hrefLabel?: string }
}
export interface LessonTable {
  caption: string
  cols: string[]
  rows: string[][]
}
export interface LessonSection {
  h: string
  /** Paragraphs; a string starting '>> ' renders as a pull quote. */
  ps: string[]
  code?: LessonCode
  table?: LessonTable
  /** Id into LESSON_DIAGRAMS, rendered after the paragraphs. */
  diagram?: string
  /** Id into the lesson template's instrument roster, rendered after the diagram. */
  exhibit?: string
}
export interface LessonReading {
  label: string
  url: string
  note: string
}
/** An end-of-lesson check: a question with a revealable answer. */
export interface LessonCheck {
  q: string
  a: string
}
export interface Lesson {
  id: string
  num: number
  title: string
  lede: string
  goal: string
  /** Authored sections; may be empty when the lesson slices a guide. */
  sections: LessonSection[]
  readings: LessonReading[]
  /** Two or three questions the reader answers before moving on. */
  check?: LessonCheck[]
  /** Mastery work past the derived read item. */
  work?: WorkItem[]
  /** A slice of a layer guide: the body lives there, told once. */
  guide?: { id: string; sections: number[] }
}
export interface UnitLessons {
  /** The owning chapter's key: 'l:<id>' | 's:<id>' | 'xla:<id>'. */
  unit: string
  /** True when the lessons carry the unit's whole guide between them,
      so the hub stops rendering the guide inline. */
  coversGuide?: boolean
  lessons: Lesson[]
}

export const ALL_UNIT_LESSONS: UnitLessons[] = [
  ...IR_DESCENT_LESSONS,
  ...COMPILER_WALL_LESSONS,
  ...MACHINE_LESSONS,
  ...PALLAS_LESSONS,
  ...XLA_LESSONS,
  ...TORCH_BRIDGE_LESSONS,
]

/** Guide sections already told inside lessons, across every unit. */
export const claimedGuideSections = (guideId: string): Set<number> => {
  const claimed = new Set<number>()
  for (const u of ALL_UNIT_LESSONS)
    for (const l of u.lessons)
      if (l.guide?.id === guideId) for (const i of l.guide.sections) claimed.add(i)
  return claimed
}

export const lessonKey = (unit: string, lessonId: string) => `${unit}:${lessonId}`

/** '/l/tpu/tpu-chip' from ('l:tpu', 'tpu-chip'); mirrors the chapter URL rules.
    The pytorch course's chapter URLs live under /pytorch while its unit keys use
    the pt: prefix, so that one prefix maps. */
export const lessonHref = (unit: string, lessonId: string) => {
  const [prefix, id] = unit.split(':') as [string, string]
  const seg = prefix === 'pt' ? 'pytorch' : prefix
  return `/${seg}/${id}/${lessonId}`
}

export const lessonsOf = (unit: string): Lesson[] =>
  ALL_UNIT_LESSONS.find((u) => u.unit === unit)?.lessons ?? []

export const lessonAt = (
  unit: string,
  lessonId: string,
): { lessons: Lesson[]; current: Lesson; prev?: Lesson; next?: Lesson } => {
  const lessons = lessonsOf(unit)
  const idx = lessons.findIndex((l) => l.id === lessonId)
  if (idx === -1) throw new Error(`unknown lesson: ${unit}/${lessonId}`)
  return { lessons, current: lessons[idx]!, prev: lessons[idx - 1], next: lessons[idx + 1] }
}

// Each lesson is mastery work under its own key: the derived read item first,
// then whatever the lesson adds.
export const LESSON_MASTERY: Record<string, WorkItem[]> = Object.fromEntries(
  ALL_UNIT_LESSONS.flatMap((u) =>
    u.lessons.map((l) => [
      lessonKey(u.unit, l.id),
      [{ id: 'read', label: `read the lesson: ${l.title}` }, ...(l.work ?? [])],
    ]),
  ),
)

// The rollup: every unit's ledger gains one auto item per lesson, complete
// when the lesson's own work is complete.
export const unitLessonItems = (unit: string): WorkItem[] =>
  lessonsOf(unit).map((l) => ({
    id: `lesson-${l.id}`,
    label: `lesson ${String(l.num).padStart(2, '0')} · ${l.title}`,
    href: lessonHref(unit, l.id),
    auto: { type: 'work' as const, key: lessonKey(unit, l.id) },
  }))
