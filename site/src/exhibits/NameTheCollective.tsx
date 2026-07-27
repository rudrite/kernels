// The name-the-collective drill: four ranks' tensors before and after one
// collective, captured from a real four-process gloo run (torch 2.2.2,
// CPU). One question: which collective ran? The same skill serves the
// xla path's collectives chapter and the pytorch path's distributed one.
import { useEffect, useMemo, useState } from 'react'
import corpus from '../data/pytorch/collective-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.collective.streak'
const STREAK_TO_PASS = 5

interface Item {
  op: string
  world: number
  before: number[][]
  after: number[][]
}

const ITEMS = (corpus as { torch: string; backend: string; items: Item[] }).items
const ALL_OPS = ITEMS.map((i) => i.op)

function shuffle<T>(items: T[]): T[] {
  const arr = [...items]
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    const t = arr[i]!
    arr[i] = arr[j]!
    arr[j] = t
  }
  return arr
}

const fmt = (xs: number[]) => `[${xs.map((v) => (Number.isInteger(v) ? v : v.toFixed(1))).join(', ')}]`

function pickItem(exclude?: Item): Item {
  let item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  while (ITEMS.length > 1 && item === exclude) item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  return item
}

export default function NameTheCollective() {
  const [item, setItem] = useState(ITEMS[0]!)
  const [picked, setPicked] = useState<string | null>(null)
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

  const options = useMemo(() => (hydrated ? shuffle(ALL_OPS) : ALL_OPS), [item, hydrated])

  const answer = (op: string) => {
    if (picked !== null) return
    setPicked(op)
    setStreak((s) => (op === item.op ? s + 1 : 0))
  }

  const advance = () => {
    setItem(pickItem(item))
    setPicked(null)
  }

  const passing = streak >= STREAK_TO_PASS

  return (
    <div className={`namecollective${passing ? ' passing' : ''}`}>
      <div className="setup">
        <table className="ranks">
          <thead>
            <tr><th>rank</th><th>before</th><th>after</th></tr>
          </thead>
          <tbody>
            {item.before.map((b, r) => (
              <tr key={r}>
                <td className="rk">{r}</td>
                <td>{fmt(b)}</td>
                <td className="af">{fmt(item.after[r]!)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="meta">
          <span className="streak">streak {streak}</span>
          <span className="prov">world {item.world} · gloo · real run</span>
        </div>
      </div>

      <p className="prompt">which collective ran?</p>

      <div className="choices" role="group" aria-label="name the collective">
        {options.map((op) => {
          const revealed = picked !== null
          const state = !revealed ? '' : op === item.op ? 'correct' : picked === op ? 'wrong' : 'dim'
          return (
            <button key={op} type="button" className={`choice ${state}`} onClick={() => answer(op)} disabled={revealed}>
              {op}
            </button>
          )
        })}
      </div>

      {picked !== null && (
        <div className="reveal" aria-live="polite">
          <span className={picked === item.op ? 'ok' : 'no'}>
            {picked === item.op ? 'correct' : `not quite: this was ${item.op}`}
          </span>
          <button type="button" className="advance" onClick={advance}>
            next run →
          </button>
        </div>
      )}

      <style>{`
        .namecollective { font-family: 'IBM Plex Mono', monospace; border: 1px solid transparent; border-radius: 2px; overflow-x: auto; }
        .namecollective.passing { border-color: ${PASS}; }
        .setup { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .ranks { border-collapse: collapse; font-size: 0.75rem; }
        .ranks th { text-align: left; font-weight: 400; font-size: 0.625rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; padding: 0 0.875rem 0.375rem 0; }
        .ranks td { padding: 0.15rem 0.875rem 0.15rem 0; color: ${PANEL_INK}; white-space: nowrap; }
        .ranks .rk { color: ${COPPER}; }
        .ranks .af { color: ${PANEL_INK}; }
        .meta { display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .meta .streak { color: ${COPPER}; font-size: 0.75rem; }
        .prompt { margin: 0.75rem 0 0.625rem; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .choices { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .choice { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.8125rem; padding: 0.45rem 0.75rem; cursor: pointer; }
        .choice:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .choice:disabled { cursor: default; }
        .choice.correct { border-color: ${PASS}; color: ${PASS}; }
        .choice.wrong { border-color: ${FAIL}; color: ${FAIL}; }
        .choice.dim { opacity: 0.5; }
        .reveal { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding-top: 0.75rem; font-size: 0.8125rem; flex-wrap: wrap; }
        .reveal .ok { color: ${PASS}; }
        .reveal .no { color: ${FAIL}; }
        .reveal .advance { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.75rem; cursor: pointer; }
        .reveal .advance:hover { border-color: ${COPPER}; color: ${COPPER}; }
      `}</style>
    </div>
  )
}
