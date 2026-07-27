// The fusion-count drill: a small program, and one question, how many
// fusion computations the optimized module ends up with. Counting is the
// question rather than "did these two ops fuse", because a count is
// unambiguous in the dump while a merge is a judgement about which ops the
// backend happened to wrap. Every answer here was compiled and counted.
import { useEffect, useState } from 'react'
import corpus from '../data/xla/fusion-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.fusioncount.streak'
const STREAK_TO_PASS = 5

interface Item {
  title: string
  code: string
  fusions: number
  bodies: string[][]
  why: string
}

const DATA = corpus as { jax: string; backend: string; items: Item[] }
const ITEMS = DATA.items
const CHOICES = [0, 1, 2, 3]

function pickItem(exclude?: Item): Item {
  let item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  while (ITEMS.length > 1 && item === exclude) item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  return item
}

export default function CountTheFusions() {
  const [item, setItem] = useState(ITEMS[0]!)
  const [picked, setPicked] = useState<number | null>(null)
  const [streak, setStreak] = useState(0)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    setItem(pickItem())
    const raw = window.localStorage.getItem(STREAK_KEY)
    const n = raw === null ? 0 : Number(raw)
    setStreak(Number.isFinite(n) && n >= 0 ? n : 0)
    setHydrated(true)
  }, [])

  useEffect(() => {
    if (!hydrated) return
    window.localStorage.setItem(STREAK_KEY, String(streak))
  }, [streak, hydrated])

  const answer = (n: number) => {
    if (picked !== null) return
    setPicked(n)
    setStreak((s) => (n === item.fusions ? s + 1 : 0))
  }

  const advance = () => {
    setItem(pickItem(item))
    setPicked(null)
  }

  const passing = streak >= STREAK_TO_PASS

  return (
    <div className={`countfusions${passing ? ' passing' : ''}`}>
      <div className="setup">
        <div className="prog">
          <span className="k">{item.title}</span>
          <code className="code">{item.code}</code>
        </div>
        <div className="meta">
          <span className="streak">streak {streak}</span>
          <span className="prov">compiled and counted</span>
        </div>
      </div>

      <p className="prompt">how many fusion computations does the optimized module hold?</p>

      <div className="choices" role="group" aria-label="fusion count">
        {CHOICES.map((n) => {
          const revealed = picked !== null
          const state = !revealed ? '' : n === item.fusions ? 'correct' : picked === n ? 'wrong' : 'dim'
          return (
            <button key={n} type="button" className={`choice ${state}`} onClick={() => answer(n)} disabled={revealed}>
              {n === 3 ? '3 or more' : n}
            </button>
          )
        })}
      </div>

      {picked !== null && (
        <div className="reveal" aria-live="polite">
          <p className={picked === item.fusions ? 'ok' : 'no'}>{item.why}</p>
          {item.bodies.length > 0 && (
            <div className="bodies">
              <span className="k">what each fusion holds</span>
              {item.bodies.map((b, i) => (
                <code key={i} className="body">
                  {b.join(' · ')}
                </code>
              ))}
            </div>
          )}
          <button type="button" className="advance" onClick={advance}>
            next program →
          </button>
        </div>
      )}

      <p className="foot">
        counted on {DATA.backend}, jax {DATA.jax} · a different backend draws these boundaries differently, which is the point of
        reading your own dump rather than trusting a rule
      </p>

      <style>{`
        .countfusions { font-family: 'IBM Plex Mono', monospace; border: 1px solid transparent; border-radius: 2px; }
        .countfusions.passing { border-color: ${PASS}; }
        .setup { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .prog { display: flex; flex-direction: column; gap: 0.35rem; }
        .k { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .code { font-size: 0.875rem; color: ${COPPER}; }
        .meta { display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .meta .streak { color: ${COPPER}; font-size: 0.75rem; }
        .prompt { margin: 0.75rem 0 0.625rem; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .choices { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .choice { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.875rem; padding: 0.4rem 1rem; cursor: pointer; }
        .choice:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .choice:disabled { cursor: default; }
        .choice.correct { border-color: ${PASS}; color: ${PASS}; }
        .choice.wrong { border-color: ${FAIL}; color: ${FAIL}; }
        .choice.dim { opacity: 0.5; }
        .reveal { display: flex; flex-direction: column; gap: 0.625rem; padding-top: 0.875rem; }
        .reveal p { margin: 0; font-size: 0.8125rem; line-height: 1.65; max-width: 68ch; }
        .reveal .ok { color: ${PASS}; }
        .reveal .no { color: ${FAIL}; }
        .bodies { display: flex; flex-direction: column; gap: 0.3rem; }
        .body { font-size: 0.75rem; color: ${PANEL_INK}; border-left: 2px solid ${COPPER}; padding-left: 0.6rem; }
        .advance { align-self: flex-start; background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.75rem; cursor: pointer; }
        .advance:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .foot { margin-top: 0.875rem; padding-top: 0.625rem; border-top: 1px solid ${PANEL_RULE}; font-size: 0.6875rem; color: ${PANEL_MUTE}; line-height: 1.6; }
      `}</style>
    </div>
  )
}
