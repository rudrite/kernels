// The stride-oracle drill: one real view chain, run on real torch, and one
// question: what (shape, stride, contiguous) lands on y. The correct answer
// is read straight off the corpus (torch 2.2.2, CPU); the wrong ones are
// the same tensor through plausible misreadings (the contiguous layout's
// strides, a shape/stride swap of axes, the flipped contiguity), never
// invented numbers.
import { useEffect, useMemo, useState } from 'react'
import corpus from '../data/pytorch/stride-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.stride.streak'
const STREAK_TO_PASS = 5

interface Item {
  code: string
  shape: number[]
  stride: number[]
  contiguous: boolean
}

const ITEMS = (corpus as { torch: string; items: Item[] }).items

interface Answer {
  shape: number[]
  stride: number[]
  contiguous: boolean
}

const eq = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i])
const sameAnswer = (a: Answer, b: Answer) =>
  eq(a.shape, b.shape) && eq(a.stride, b.stride) && a.contiguous === b.contiguous

/** Row-major strides for a shape: what the tensor would have if contiguous. */
function contiguousStrides(shape: number[]): number[] {
  const out = new Array<number>(shape.length)
  let acc = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    out[i] = acc
    acc *= shape[i]!
  }
  return out
}

function rotate(xs: number[]): number[] {
  return xs.length < 2 ? [...xs, 1] : [...xs.slice(1), xs[0]!]
}

interface Choice extends Answer {
  correct: boolean
}

function buildChoices(item: Item, randomize: boolean): Choice[] {
  const correct: Answer = { shape: item.shape, stride: item.stride, contiguous: item.contiguous }
  const raw: Answer[] = [
    correct,
    // the classic misread: assume the layout is whatever the shape implies
    { shape: item.shape, stride: contiguousStrides(item.shape), contiguous: true },
    // axes remembered in the wrong order
    { shape: rotate(item.shape), stride: rotate(item.stride), contiguous: item.contiguous },
    // right map, wrong verdict
    { shape: item.shape, stride: item.stride, contiguous: !item.contiguous },
  ]
  const seen: Answer[] = []
  const deduped = raw.map((cand) => {
    let cur = cand
    let guard = 0
    while (seen.some((s) => sameAnswer(s, cur)) && guard < 6) {
      cur = { ...cur, stride: [...cur.stride.slice(0, -1), (cur.stride.at(-1) ?? 1) + 1] }
      guard++
    }
    seen.push(cur)
    return cur
  })
  const withFlags = deduped.map((c) => ({ ...c, correct: sameAnswer(c, correct) }))
  return randomize ? shuffle(withFlags) : withFlags
}

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

const fmt = (xs: number[]) => `(${xs.join(', ')})`

function pickItem(exclude?: Item): Item {
  let item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  while (ITEMS.length > 1 && item === exclude) item = ITEMS[Math.floor(Math.random() * ITEMS.length)]!
  return item
}

export default function StrideOracle() {
  // First render sticks to the corpus's first item, unshuffled, so server
  // markup and the pre-hydration client pass agree; randomness arrives in
  // the mount effect.
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

  const choices = useMemo(() => buildChoices(item, hydrated), [item, hydrated])

  const answer = (i: number) => {
    if (picked !== null) return
    setPicked(i)
    setStreak((s) => (choices[i]!.correct ? s + 1 : 0))
  }

  const advance = () => {
    setItem(pickItem(item))
    setPicked(null)
  }

  const passing = streak >= STREAK_TO_PASS

  return (
    <div className={`strideoracle${passing ? ' passing' : ''}`}>
      <div className="setup">
        <pre className="code">{item.code}</pre>
        <div className="meta">
          <span className="streak">streak {streak}</span>
          <span className="prov">torch 2.2.2 · cpu</span>
        </div>
      </div>

      <p className="prompt">what is y: shape, stride, contiguous?</p>

      <div className="choices" role="group" aria-label="shape, stride, and contiguity, 4 choices">
        {choices.map((c, i) => {
          const chosen = picked === i
          const revealed = picked !== null
          const state = !revealed ? '' : c.correct ? 'correct' : chosen ? 'wrong' : 'dim'
          return (
            <button
              key={i}
              type="button"
              className={`choice ${state}`}
              onClick={() => answer(i)}
              disabled={revealed}
              aria-pressed={chosen}
            >
              <span className="line">shape {fmt(c.shape)}</span>
              <span className="line">stride {fmt(c.stride)}</span>
              <span className="flag">{c.contiguous ? 'contiguous' : 'non-contiguous'}</span>
            </button>
          )
        })}
      </div>

      {picked !== null && (
        <div className="reveal" aria-live="polite">
          <span className={choices[picked]!.correct ? 'ok' : 'no'}>
            {choices[picked]!.correct ? 'correct' : 'not quite'}
          </span>
          <button type="button" className="advance" onClick={advance}>
            next chain →
          </button>
        </div>
      )}

      <style>{`
        .strideoracle { font-family: 'IBM Plex Mono', monospace; border: 1px solid transparent; border-radius: 2px; overflow-x: auto; }
        .strideoracle.passing { border-color: ${PASS}; }

        .setup { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .code { margin: 0; font-size: 0.8125rem; line-height: 1.7; color: ${PANEL_INK}; white-space: pre; }
        .meta { display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .meta .streak { color: ${COPPER}; font-size: 0.75rem; }

        .prompt { margin: 0.75rem 0 0.625rem; font-size: 0.8125rem; color: ${PANEL_INK}; }

        .choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 0.5rem; }
        .choice { display: flex; flex-direction: column; gap: 0.2rem; background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.78125rem; padding: 0.5rem 0.625rem; text-align: left; cursor: pointer; }
        .choice .flag { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .choice:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .choice:hover:not(:disabled) .flag { color: ${COPPER}; }
        .choice:disabled { cursor: default; }
        .choice.correct { border-color: ${PASS}; color: ${PASS}; }
        .choice.correct .flag { color: ${PASS}; }
        .choice.wrong { border-color: ${FAIL}; color: ${FAIL}; }
        .choice.wrong .flag { color: ${FAIL}; }
        .choice.dim { opacity: 0.5; }

        .reveal { display: flex; justify-content: space-between; align-items: center; padding-top: 0.75rem; font-size: 0.8125rem; }
        .reveal .ok { color: ${PASS}; }
        .reveal .no { color: ${FAIL}; }
        .reveal .advance { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.75rem; cursor: pointer; }
        .reveal .advance:hover { border-color: ${COPPER}; color: ${COPPER}; }
      `}</style>
    </div>
  )
}
