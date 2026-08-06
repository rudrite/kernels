// One check, shared by every instrument that can play on its own. Stepping
// by hand always works; only self-advancing playback respects the setting.
export const prefersReducedMotion = (): boolean =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
