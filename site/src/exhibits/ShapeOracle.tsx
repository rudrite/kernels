// The shape-oracle drill: one real equation from a real jaxpr, its inputs'
// dtypes and shapes, and a single question: what shape and dtype lands on
// the output. The correct answer is read straight off the corpus; the three
// wrong ones are the same real output run through a plausible misreading
// (kept a reduced axis, transposed it, or got the dtype wrong), not
// invented numbers.
import { useEffect, useMemo, useState } from 'react'
import corpus from '../data/ir-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.shape.streak'
const STREAK_TO_PASS = 10
const DTYPES = ['bfloat16', 'float32', 'int32', 'int8', 'bool']

interface TensorSpec {
  dtype: string
  shape: number[]
}

interface Eqn {
  op: string
  inputs: TensorSpec[]
  outputs: TensorSpec[]
}

interface Program {
  id: string
  eqns: Eqn[]
}

const PROGRAMS = corpus.programs as Program[]

// Only equations with exactly one output are eligible: the game asks for
// "the output", which only means one thing when there is one.
const POOL: { program: string; eqn: Eqn }[] = PROGRAMS.flatMap((p) =>
  p.eqns.filter((e) => e.outputs.length === 1 && e.inputs.length >= 1).map((eqn) => ({ program: p.id, eqn })),
)

function arraysEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i])
}

// A rotation by one position: the same dims, reordered, as if the axes had
// been transposed. Returns null when the shape is too symmetric to notice
// (rank < 2, or every rotation lands back on the same sequence).
function rotateLeft(shape: number[]): number[] | null {
  if (shape.length < 2) return null
  const rotated = [...shape.slice(1), shape[0]!]
  return arraysEqual(rotated, shape) ? null : rotated
}

function transposedVariant(shape: number[]): number[] {
  return rotateLeft(shape) ?? [1, ...shape]
}

function keepdimsVariant(shape: number[]): number[] {
  return [...shape, 1]
}

interface Pair {
  shape: number[]
  dtype: string
}

function pairEqual(a: Pair, b: Pair): boolean {
  return a.dtype === b.dtype && arraysEqual(a.shape, b.shape)
}

interface Choice extends Pair {
  correct: boolean
}

// `randomize` stays false for the very first render (server-rendered markup
// and the client's pre-hydration pass must agree exactly); a mount effect
// flips it true so the wrong-dtype pick and choice order only randomize
// client-side, after hydration.
function buildChoices(spec: TensorSpec, randomize: boolean): Choice[] {
  const correct: Pair = { shape: spec.shape, dtype: spec.dtype }
  const otherDtypes = DTYPES.filter((d) => d !== spec.dtype)
  const wrongDtype = (randomize ? otherDtypes[Math.floor(Math.random() * otherDtypes.length)] : otherDtypes[0]) ?? spec.dtype

  const raw: Pair[] = [
    correct,
    { shape: keepdimsVariant(spec.shape), dtype: spec.dtype },
    { shape: transposedVariant(spec.shape), dtype: spec.dtype },
    { shape: spec.shape, dtype: wrongDtype },
  ]

  // Two variants can coincide (a symmetric shape's "transpose" is itself,
  // say). When that happens, extend the colliding candidate with one more
  // real value (a trailing 1) until it reads as its own option.
  const seen: Pair[] = []
  const deduped = raw.map((cand) => {
    let cur = cand
    let guard = 0
    while (seen.some((s) => pairEqual(s, cur)) && guard < 6) {
      cur = { shape: [...cur.shape, 1], dtype: cur.dtype }
      guard++
    }
    seen.push(cur)
    return cur
  })

  const withFlags = deduped.map((c) => ({ ...c, correct: pairEqual(c, correct) }))
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

function formatShape(shape: number[]): string {
  return shape.length === 0 ? 'scalar []' : `[${shape.join(', ')}]`
}

function pickEntry(exclude?: (typeof POOL)[number]) {
  let entry = POOL[Math.floor(Math.random() * POOL.length)]!
  while (POOL.length > 1 && entry === exclude) entry = POOL[Math.floor(Math.random() * POOL.length)]!
  return entry
}

export default function ShapeOracle() {
  // Starts on the corpus's first eligible equation, unshuffled, so the
  // first render matches the server-rendered markup exactly; a mount
  // effect below swaps in a real random equation once hydration is safe.
  const [entry, setEntry] = useState(POOL[0]!)
  const [picked, setPicked] = useState<number | null>(null)
  const [streak, setStreak] = useState(0)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    setEntry(pickEntry())
    const raw = window.localStorage.getItem(STREAK_KEY)
    const n = raw === null ? 0 : Number(raw)
    setStreak(Number.isFinite(n) && n >= 0 ? n : 0)
    setHydrated(true)
  }, [])

  useEffect(() => {
    if (!hydrated) return
    window.localStorage.setItem(STREAK_KEY, String(streak))
  }, [streak, hydrated])

  const choices = useMemo(() => buildChoices(entry.eqn.outputs[0]!, hydrated), [entry, hydrated])

  const answer = (i: number) => {
    if (picked !== null) return
    setPicked(i)
    setStreak((s) => (choices[i]!.correct ? s + 1 : 0))
  }

  const advance = () => {
    setEntry(pickEntry(entry))
    setPicked(null)
  }

  const passing = streak >= STREAK_TO_PASS

  return (
    <div className={`shapeoracle${passing ? ' passing' : ''}`}>
      <div className="setup">
        <div className="op">
          <span className="k">op</span>
          <span className="v">{entry.eqn.op}</span>
        </div>
        <ul className="inputs">
          {entry.eqn.inputs.map((inp, i) => (
            <li key={i}>
              <span className="dtype">{inp.dtype}</span>
              <span className="shape">{formatShape(inp.shape)}</span>
            </li>
          ))}
        </ul>
        <div className="meta">
          <span className="streak">streak {streak}</span>
          <span className="prog">program: {entry.program}</span>
        </div>
      </div>

      <p className="prompt">what shape and dtype comes out?</p>

      <div className="choices" role="group" aria-label="output shape and dtype, 4 choices">
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
              <span className="shape">{formatShape(c.shape)}</span>
              <span className="dtype">{c.dtype}</span>
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
            next equation →
          </button>
        </div>
      )}

      <style>{`
        .shapeoracle { font-family: 'IBM Plex Mono', monospace; border: 1px solid transparent; border-radius: 2px; }
        .shapeoracle.passing { border-color: ${PASS}; }
        .shapeoracle { overflow-x: auto; }

        .setup { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .op { display: flex; gap: 0.6rem; align-items: baseline; }
        .op .k { font-size: 0.625rem; letter-spacing: 0.08em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .op .v { font-size: 0.9375rem; color: ${COPPER}; }

        .inputs { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; }
        .inputs li { display: flex; gap: 0.5rem; font-size: 0.75rem; }
        .inputs .dtype { color: ${PANEL_MUTE}; }
        .inputs .shape { color: ${PANEL_INK}; }

        .meta { display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .meta .streak { color: ${COPPER}; font-size: 0.75rem; }

        .prompt { margin: 0.75rem 0 0.625rem; font-size: 0.8125rem; color: ${PANEL_INK}; }

        .choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap: 0.5rem; }
        .choice { display: flex; flex-direction: column; gap: 0.2rem; background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.8125rem; padding: 0.5rem 0.625rem; text-align: left; cursor: pointer; }
        .choice .dtype { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .choice:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .choice:hover:not(:disabled) .dtype { color: ${COPPER}; }
        .choice:disabled { cursor: default; }
        .choice.correct { border-color: ${PASS}; color: ${PASS}; }
        .choice.correct .dtype { color: ${PASS}; }
        .choice.wrong { border-color: ${FAIL}; color: ${FAIL}; }
        .choice.wrong .dtype { color: ${FAIL}; }
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
