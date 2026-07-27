// The pass-origin drill: one pass name from a real dump, and one question,
// which pipeline filed it. Every name here was read out of a dump this
// course produced, the CPU side on the course machine and the TPU side on
// a Colab v6e. The answer is about these two dumps, not about every build
// of XLA that ever shipped, and the prose says so.
import { useEffect, useState } from 'react'
import corpus from '../data/xla/pass-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.pass.streak'
const STREAK_TO_PASS = 5

interface Item {
  name: string
  origin: 'cpu' | 'tpu' | 'both'
  why: string
}

const DATA = corpus as { cpu: string; tpu: string; items: Item[] }
const ITEMS = DATA.items

const CHOICES: { origin: Item['origin']; label: string; sub: string }[] = [
  { origin: 'cpu', label: 'the CPU dump', sub: 'only in the CPU pipeline' },
  { origin: 'tpu', label: 'the TPU dump', sub: 'only in the TPU pipeline' },
  { origin: 'both', label: 'both dumps', sub: 'the same name in each' },
]

function pickItem(exclude?: Item): Item {
  let item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  while (ITEMS.length > 1 && item === exclude) item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  return item
}

export default function NameThePass() {
  const [item, setItem] = useState(ITEMS[0]!)
  const [picked, setPicked] = useState<Item['origin'] | null>(null)
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

  const answer = (o: Item['origin']) => {
    if (picked !== null) return
    setPicked(o)
    setStreak((s) => (o === item.origin ? s + 1 : 0))
  }

  const advance = () => {
    setItem(pickItem(item))
    setPicked(null)
  }

  const passing = streak >= STREAK_TO_PASS

  return (
    <div className={`namepass${passing ? ' passing' : ''}`}>
      <div className="setup">
        <div className="pass">
          <span className="k">pass</span>
          <code className="v">{item.name}</code>
        </div>
        <div className="meta">
          <span className="streak">streak {streak}</span>
          <span className="prov">two real dumps</span>
        </div>
      </div>

      <p className="prompt">which pipeline filed this one?</p>

      <div className="choices" role="group" aria-label="cpu, tpu, or both">
        {CHOICES.map((c) => {
          const revealed = picked !== null
          const state = !revealed ? '' : c.origin === item.origin ? 'correct' : picked === c.origin ? 'wrong' : 'dim'
          return (
            <button key={c.origin} type="button" className={`choice ${state}`} onClick={() => answer(c.origin)} disabled={revealed}>
              <span className="line">{c.label}</span>
              <span className="sub">{c.sub}</span>
            </button>
          )
        })}
      </div>

      {picked !== null && (
        <div className="reveal" aria-live="polite">
          <span className={picked === item.origin ? 'ok' : 'no'}>{item.why}</span>
          <button type="button" className="advance" onClick={advance}>
            next pass →
          </button>
        </div>
      )}

      <p className="foot">
        cpu: {DATA.cpu} · tpu: {DATA.tpu}
      </p>

      <style>{`
        .namepass { font-family: 'IBM Plex Mono', monospace; border: 1px solid transparent; border-radius: 2px; overflow-x: auto; }
        .namepass.passing { border-color: ${PASS}; }
        .setup { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .pass { display: flex; flex-direction: column; gap: 0.3rem; }
        .pass .k { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .pass .v { font-size: 1rem; color: ${COPPER}; }
        .meta { display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .meta .streak { color: ${COPPER}; font-size: 0.75rem; }
        .prompt { margin: 0.75rem 0 0.625rem; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 0.5rem; }
        .choice { display: flex; flex-direction: column; gap: 0.25rem; background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.8125rem; padding: 0.5rem 0.625rem; text-align: left; cursor: pointer; }
        .choice .sub { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .choice:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .choice:disabled { cursor: default; }
        .choice.correct { border-color: ${PASS}; color: ${PASS}; }
        .choice.correct .sub { color: ${PASS}; }
        .choice.wrong { border-color: ${FAIL}; color: ${FAIL}; }
        .choice.wrong .sub { color: ${FAIL}; }
        .choice.dim { opacity: 0.5; }
        .reveal { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; padding-top: 0.75rem; font-size: 0.78125rem; line-height: 1.6; flex-wrap: wrap; }
        .reveal .ok { color: ${PASS}; max-width: 44rem; }
        .reveal .no { color: ${FAIL}; max-width: 44rem; }
        .reveal .advance { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.75rem; cursor: pointer; white-space: nowrap; }
        .reveal .advance:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .foot { margin-top: 0.875rem; padding-top: 0.625rem; border-top: 1px solid ${PANEL_RULE}; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
      `}</style>
    </div>
  )
}
