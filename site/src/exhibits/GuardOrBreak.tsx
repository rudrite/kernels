// The guard-or-break drill: one torch.compile scenario, run for real, and
// one question: did the calls end in one graph, a recompile, or a graph
// break? Every verdict was measured with dynamo's own counters (torch
// 2.2.2, Python 3.11, CPU); the reveal shows the counts.
import { useEffect, useState } from 'react'
import corpus from '../data/pytorch/guard-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.guard.streak'
const STREAK_TO_PASS = 5

interface Item {
  id: string
  code: string
  graphs: number
  breaks: number
  verdict: 'one-graph' | 'recompile' | 'graph-break'
}

const ITEMS = (corpus as { torch: string; items: Item[] }).items

const CHOICES: { verdict: Item['verdict']; label: string; sub: string }[] = [
  { verdict: 'one-graph', label: 'one graph', sub: 'guards held; the cache served every call' },
  { verdict: 'recompile', label: 'recompile', sub: 'a guard missed; dynamo captured again' },
  { verdict: 'graph-break', label: 'graph break', sub: 'Python ran in the middle; pieces stitched around it' },
]

function pickItem(exclude?: Item): Item {
  let item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  while (ITEMS.length > 1 && item === exclude) item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  return item
}

export default function GuardOrBreak() {
  const [item, setItem] = useState(ITEMS[0]!)
  const [picked, setPicked] = useState<Item['verdict'] | null>(null)
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

  const answer = (v: Item['verdict']) => {
    if (picked !== null) return
    setPicked(v)
    setStreak((s) => (v === item.verdict ? s + 1 : 0))
  }

  const advance = () => {
    setItem(pickItem(item))
    setPicked(null)
  }

  const passing = streak >= STREAK_TO_PASS

  return (
    <div className={`guardbreak${passing ? ' passing' : ''}`}>
      <div className="setup">
        <pre className="code">{item.code}</pre>
        <div className="meta">
          <span className="streak">streak {streak}</span>
          <span className="prov">torch 2.2.2 · measured</span>
        </div>
      </div>

      <p className="prompt">what did dynamo do?</p>

      <div className="choices" role="group" aria-label="one graph, recompile, or graph break">
        {CHOICES.map((c) => {
          const revealed = picked !== null
          const state = !revealed ? '' : c.verdict === item.verdict ? 'correct' : picked === c.verdict ? 'wrong' : 'dim'
          return (
            <button key={c.verdict} type="button" className={`choice ${state}`} onClick={() => answer(c.verdict)} disabled={revealed}>
              <span className="line">{c.label}</span>
              <span className="sub">{c.sub}</span>
            </button>
          )
        })}
      </div>

      {picked !== null && (
        <div className="reveal" aria-live="polite">
          <span className={picked === item.verdict ? 'ok' : 'no'}>
            {picked === item.verdict ? 'correct' : 'not quite'} · measured: {item.graphs} graph{item.graphs === 1 ? '' : 's'}, {item.breaks} break{item.breaks === 1 ? '' : 's'}
          </span>
          <button type="button" className="advance" onClick={advance}>
            next scenario →
          </button>
        </div>
      )}

      <style>{`
        .guardbreak { font-family: 'IBM Plex Mono', monospace; border: 1px solid transparent; border-radius: 2px; overflow-x: auto; }
        .guardbreak.passing { border-color: ${PASS}; }
        .setup { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .code { margin: 0; font-size: 0.78125rem; line-height: 1.65; color: ${PANEL_INK}; white-space: pre; }
        .meta { display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .meta .streak { color: ${COPPER}; font-size: 0.75rem; }
        .prompt { margin: 0.75rem 0 0.625rem; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 0.5rem; }
        .choice { display: flex; flex-direction: column; gap: 0.25rem; background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.8125rem; padding: 0.5rem 0.625rem; text-align: left; cursor: pointer; }
        .choice .sub { font-size: 0.6875rem; color: ${PANEL_MUTE}; line-height: 1.5; }
        .choice:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .choice:disabled { cursor: default; }
        .choice.correct { border-color: ${PASS}; color: ${PASS}; }
        .choice.correct .sub { color: ${PASS}; }
        .choice.wrong { border-color: ${FAIL}; color: ${FAIL}; }
        .choice.wrong .sub { color: ${FAIL}; }
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
