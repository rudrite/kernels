// Lessons: the depth layer of a unit. A unit is a course chapter (its key is
// the chapter key: 'l:tpu', 's:pallas', 'xla:pjrt'); its lessons are ordered
// full pages nested under the chapter's URL. One canonical home per fact: a
// lesson belongs to exactly one unit, and the shelf that once held these
// pages is gone (docs/plans/true-mastery-courses.md).
import type { WorkItem } from '../mastery'
import { MACHINE_LESSONS } from './machine'
import { PALLAS_LESSONS } from './pallas'
import { XLA_LESSONS } from './xla-internals'

export interface LessonCode {
  caption: string
  text: string
  /** Highlighter grammar; 'mlir' renders through MlirCode, not Shiki. */
  lang?: 'python' | 'haskell' | 'c' | 'mlir' | 'text'
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
export interface Lesson {
  id: string
  num: number
  title: string
  lede: string
  goal: string
  sections: LessonSection[]
  readings: LessonReading[]
  /** Mastery work past the derived read item. */
  work?: WorkItem[]
}
export interface UnitLessons {
  /** The owning chapter's key: 'l:<id>' | 's:<id>' | 'xla:<id>'. */
  unit: string
  lessons: Lesson[]
}

export const ALL_UNIT_LESSONS: UnitLessons[] = [
  ...MACHINE_LESSONS,
  ...PALLAS_LESSONS,
  ...XLA_LESSONS,
]

export const lessonKey = (unit: string, lessonId: string) => `${unit}:${lessonId}`

/** '/l/tpu/tpu-chip' from ('l:tpu', 'tpu-chip'); mirrors the chapter URL rules. */
export const lessonHref = (unit: string, lessonId: string) => {
  const [prefix, id] = unit.split(':') as [string, string]
  return `/${prefix}/${id}/${lessonId}`
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
