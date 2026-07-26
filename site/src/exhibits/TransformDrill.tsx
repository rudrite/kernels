// Name the transform: a gallery variant's jaxpr appears with its loudest
// primitive names redacted, and the drill is recognizing which transform
// produced it from shapes and structure alone. Dumps come from
// gen_transform_gallery.py; the redaction is presentational and labeled.
import { useEffect, useState } from 'react'
import gallery from '../data/transform-gallery.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#9aa1a8'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.transform.streak'
const STREAK_TO_PASS = 5
const MAX_LINES = 16

interface Variant {
  id: string
  title: string
  jaxpr: string[]
}
const VARIANTS = (gallery as { variants: Variant[] }).variants

const OPTIONS: { id: string; label: string }[] = [
  { id: 'plain', label: 'the plain function' },
  { id: 'grad', label: 'jax.grad' },
  { id: 'vmap', label: 'jax.vmap' },
  { id: 'scan', label: 'lax.scan' },
  { id: 'remat', label: 'jax.checkpoint' },
  { id: 'shard_map', label: 'shard_map' },
]

// redact the giveaway primitive names; everything else is the real dump
const redact = (line: string) =>
  line.replace(/\b(scan|remat2|shard_map|pbroadcast|psum)\b/g, '█████')

export default function TransformDrill() {
  const [qi, setQi] = useState(0)
  const [picked, setPicked] = useState<string | null>(null)
  const [streak, setStreak] = useState(0)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    setQi(Math.floor(Math.random() * VARIANTS.length))
    setHydrated(true)
  }, [])
  useEffect(() => {
    if (!hydrated) return
    window.localStorage.setItem(STREAK_KEY, String(streak))
    window.dispatchEvent(new Event('storage'))
  }, [streak, hydrated])

  const variant = VARIANTS[qi]!
  const lines = variant.jaxpr.slice(0, MAX_LINES)
  const truncated = variant.jaxpr.length > MAX_LINES

  const answer = (id: string) => {
    if (picked !== null) return
    setPicked(id)
    setStreak(id === variant.id ? streak + 1 : 0)
  }
  const next = () => {
    setPicked(null)
    let n = Math.floor(Math.random() * VARIANTS.length)
    if (n === qi) n = (n + 1) % VARIANTS.length
    setQi(n)
  }

  const passing = streak >= STREAK_TO_PASS

  return (
    <div className="tdrill">
      <div className="head">
        <span className="prompt">the same function, traced under one transform. Which one?</span>
        <span className="streak" style={{ color: passing ? PASS : PANEL_MUTE }}>
          streak {streak}{passing ? ' ✓' : ` / ${STREAK_TO_PASS}`}
        </span>
      </div>
      <pre>
        {lines.map((l, i) => (
          <span key={i} className="ln">{redact(l) || ' '}{'\n'}</span>
        ))}
        {truncated && <span className="more">… {variant.jaxpr.length - MAX_LINES} more lines</span>}
      </pre>
      <div className="opts">
        {OPTIONS.map((o) => {
          const state = picked === null ? '' : o.id === variant.id ? 'right' : o.id === picked ? 'wrong' : 'dim'
          return (
            <button key={o.id} type="button" className={`opt ${state}`} onClick={() => answer(o.id)}>
              {o.label}
            </button>
          )
        })}
      </div>
      <div className="foot" aria-live="polite">
        {picked === null ? (
          <span className="hint">redacted blocks hide primitive names; shapes and structure are the tell</span>
        ) : (
          <>
            <span className="verdict" style={{ color: picked === variant.id ? PASS : FAIL }}>
              {picked === variant.id ? 'right' : 'not this one'} · {variant.title}
            </span>
            <button type="button" className="next" onClick={next}>next →</button>
          </>
        )}
      </div>

      <style>{`
        .tdrill { font-family: 'IBM Plex Mono', monospace; }
        .tdrill .head { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; padding-bottom: 0.625rem; flex-wrap: wrap; }
        .tdrill .prompt { font-size: 0.8125rem; color: ${PANEL_INK}; }
        .tdrill .streak { font-size: 0.6875rem; letter-spacing: 0.06em; }
        .tdrill pre { margin: 0; border: 1px solid ${PANEL_RULE}; padding: 0.625rem 0.75rem; overflow-x: auto; max-height: 20rem; overflow-y: auto; font-size: 0.75rem; line-height: 1.65; color: ${PANEL_INK}; }
        .tdrill .more { color: ${PANEL_MUTE}; }
        .tdrill .opts { display: flex; gap: 0.5rem; flex-wrap: wrap; padding-top: 0.75rem; }
        .tdrill .opt { font-family: inherit; font-size: 0.75rem; background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; padding: 0.375rem 0.75rem; cursor: pointer; }
        .tdrill .opt:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .tdrill .opt.right { border-color: ${PASS}; color: ${PASS}; }
        .tdrill .opt.wrong { border-color: ${FAIL}; color: ${FAIL}; }
        .tdrill .opt.dim { opacity: 0.4; }
        .tdrill .foot { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding-top: 0.625rem; min-height: 2rem; flex-wrap: wrap; }
        .tdrill .hint, .tdrill .verdict { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .tdrill .next { font-family: inherit; font-size: 0.75rem; background: transparent; color: ${COPPER}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; padding: 0.3rem 0.75rem; cursor: pointer; }
        .tdrill .next:hover { border-color: ${COPPER}; }
      `}</style>
    </div>
  )
}
