// One ledger across every path: the kernel track's mastery record and the
// JAX path's, merged for the components that paint progress. Chapter keys
// never collide by construction ('l:'/'s:' vs 'jax:').
import { MASTERY, type WorkItem } from './mastery'
import { JAX_MASTERY } from './jax/track'

export const ALL_MASTERY: Record<string, WorkItem[]> = { ...MASTERY, ...JAX_MASTERY }
