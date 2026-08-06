// One ledger across every path, merged for the components that paint
// progress. Chapter keys never collide by construction ('l:'/'s:' vs
// 'jax:' vs 'xla:'; lesson keys extend their unit's key). Units with
// lessons gain one auto item per lesson, complete when the lesson's own
// work completes: the rollup.
import { MASTERY, type WorkItem } from './mastery'
import { JAX_MASTERY } from './jax/track'
import { XLA_MASTERY } from './xla/track'
import { PT_MASTERY } from './pytorch/track'
import { ALL_UNIT_LESSONS, LESSON_MASTERY, unitLessonItems } from './lessons'

export const ALL_MASTERY: Record<string, WorkItem[]> = (() => {
  const merged: Record<string, WorkItem[]> = {
    ...MASTERY,
    ...JAX_MASTERY,
    ...XLA_MASTERY,
    ...PT_MASTERY,
    ...LESSON_MASTERY,
  }
  for (const u of ALL_UNIT_LESSONS) {
    merged[u.unit] = [...(merged[u.unit] ?? []), ...unitLessonItems(u.unit)]
  }
  return merged
})()
