// One ledger across every path, merged for the components that paint
// progress. Chapter keys never collide by construction ('l:'/'s:' vs
// 'jax:' vs 'xla:').
import { MASTERY, type WorkItem } from './mastery'
import { JAX_MASTERY } from './jax/track'
import { XLA_MASTERY } from './xla/track'

export const ALL_MASTERY: Record<string, WorkItem[]> = { ...MASTERY, ...JAX_MASTERY, ...XLA_MASTERY }
