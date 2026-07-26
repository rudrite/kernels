// Spot the decision: a full jaxpr, and one question. Somewhere in this
// program the framework made a call the Python source never wrote out loud:
// widened a dtype for a stable reduction, expanded a bias into a broadcast,
// or marked a block for recomputation instead of storing it. Find that line.
// The target line and its explanation are both computed from this program's
// own eqns (real dtypes, real shapes); nothing here is invented to make the
// round work.
import { useEffect, useMemo, useState } from 'react'
import corpus from '../data/ir-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.spot.streak'
const STREAK_TO_PASS = 10

interface EqnIO {
  dtype: string
  shape: number[]
}
interface Eqn {
  op: string
  inputs: EqnIO[]
  outputs: EqnIO[]
}
interface Program {
  id: string
  title: string
  note: string
  source: string[]
  jaxpr: string[]
  eqns: Eqn[]
}
const PROGRAMS = corpus.programs as Program[]

// Bit widths for the dtypes this corpus actually uses. An "upcast" is any
// convert_element_type whose output is wider than its input by this table;
// nothing here guesses at dtypes the corpus doesn't emit.
const DTYPE_BITS: Record<string, number> = {
  bool: 1,
  int8: 8,
  bfloat16: 16,
  float16: 16,
  int32: 32,
  float32: 32,
}

type Category = 'upcast' | 'broadcast' | 'remat'

const CATEGORY_PHRASE: Record<Category, string> = {
  upcast: 'upcasts a dtype',
  broadcast: 'makes a hidden broadcast explicit',
  remat: 'marks recomputation',
}

interface RoundSpec {
  program: Program
  category: Category
  targetLine: number
  /** The fact that makes the explanation sentence honest: real dtypes, a real shape, or the real op token. */
  detail: string
}

// The Nth occurrence of an op in the jaxpr text and the Nth eqn of that op
// share the same order, because both are read off the same trace in program
// order (verified against every program in the corpus: the counts always
// match). So the line for a given eqn is just its ordinal position among
// same-op text lines.
function linesForOp(jaxpr: string[], op: string): number[] {
  const marker = `${op}[`
  const lines: number[] = []
  jaxpr.forEach((line, i) => {
    if (line.includes(marker)) lines.push(i)
  })
  return lines
}

function upcastRound(program: Program): RoundSpec | null {
  const textLines = linesForOp(program.jaxpr, 'convert_element_type')
  const eqns = program.eqns.filter((e) => e.op === 'convert_element_type')
  for (let i = 0; i < eqns.length && i < textLines.length; i++) {
    const eqn = eqns[i]!
    const inBits = DTYPE_BITS[eqn.inputs[0]?.dtype ?? ''] ?? 0
    const outBits = DTYPE_BITS[eqn.outputs[0]?.dtype ?? ''] ?? 0
    if (outBits > inBits) {
      return {
        program,
        category: 'upcast',
        targetLine: textLines[i]!,
        detail: `${eqn.inputs[0]!.dtype} to ${eqn.outputs[0]!.dtype}`,
      }
    }
  }
  return null
}

function broadcastRound(program: Program): RoundSpec | null {
  const textLines = linesForOp(program.jaxpr, 'broadcast_in_dim')
  const eqn = program.eqns.find((e) => e.op === 'broadcast_in_dim')
  if (textLines.length === 0 || !eqn) return null
  return {
    program,
    category: 'broadcast',
    targetLine: textLines[0]!,
    detail: `[${eqn.outputs[0]!.shape.join(', ')}]`,
  }
}

function rematRound(program: Program): RoundSpec | null {
  const line = program.jaxpr.findIndex((l) => l.includes('remat'))
  if (line === -1) return null
  const match = /\bremat\w*/.exec(program.jaxpr[line]!)
  return {
    program,
    category: 'remat',
    targetLine: line,
    detail: match ? match[0] : 'remat2',
  }
}

// Every (program, category) pair the corpus actually supports. Programs with
// none of the three markers (gqa, rope, scan, sharded, psum, as of this
// corpus) simply contribute no rounds.
const ALL_ROUNDS: RoundSpec[] = PROGRAMS.flatMap((program) => {
  const rounds = [upcastRound(program), broadcastRound(program), rematRound(program)]
  return rounds.filter((r): r is RoundSpec => r !== null)
})

function explainSentence(round: RoundSpec): string {
  const ln = round.targetLine + 1
  if (round.category === 'upcast') {
    return `Line ${ln}: convert_element_type carries the value from ${round.detail}, precision the source function never wrote.`
  }
  if (round.category === 'broadcast') {
    return `Line ${ln}: broadcast_in_dim expands into shape ${round.detail}, the broadcast the Python line left implicit.`
  }
  return `Line ${ln}: ${round.detail} marks this block for recomputation, trading a stored activation for a second forward pass.`
}

function pickRound(exclude?: RoundSpec): RoundSpec {
  let r = ALL_ROUNDS[Math.floor(Math.random() * ALL_ROUNDS.length)]!
  while (ALL_ROUNDS.length > 1 && r.program.id === exclude?.program.id && r.category === exclude?.category) {
    r = ALL_ROUNDS[Math.floor(Math.random() * ALL_ROUNDS.length)]!
  }
  return r
}

export default function SpotDecision() {
  // Starts on the corpus's first round so the first render matches the
  // server-rendered markup exactly; a mount effect swaps in a real random
  // round and reads the streak once hydration is safe.
  const [round, setRound] = useState<RoundSpec>(ALL_ROUNDS[0]!)
  const [picked, setPicked] = useState<number | null>(null)
  const [streak, setStreak] = useState(0)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    setRound(pickRound())
    const raw = window.localStorage.getItem(STREAK_KEY)
    const n = raw === null ? 0 : Number(raw)
    setStreak(Number.isFinite(n) && n >= 0 ? n : 0)
    setHydrated(true)
  }, [])

  useEffect(() => {
    if (!hydrated) return
    window.localStorage.setItem(STREAK_KEY, String(streak))
  }, [streak, hydrated])

  const explanation = useMemo(() => explainSentence(round), [round])
  const passing = streak >= STREAK_TO_PASS

  const answer = (line: number) => {
    if (picked !== null) return // one click per round
    setPicked(line)
    setStreak((s) => (line === round.targetLine ? s + 1 : 0))
  }

  const next = () => {
    setRound(pickRound(round))
    setPicked(null)
  }

  const lineState = (i: number): string => {
    if (picked === null) return ''
    if (i === round.targetLine) return 'correct'
    if (i === picked) return 'wrong'
    return ''
  }

  return (
    <div className={`spotdecision${passing ? ' passing' : ''}`}>
      <div className="setup">
        <span className="title">{round.program.title}</span>
        <span className="streak">streak {streak}</span>
      </div>

      <p className="prompt">
        click the first line where the framework <span className="copper">{CATEGORY_PHRASE[round.category]}</span>
      </p>

      <div className="jaxpr" role="list" aria-label={`jaxpr for ${round.program.title}, click a line`}>
        {round.program.jaxpr.map((line, i) => (
          <button
            key={i}
            type="button"
            role="listitem"
            className={`ln ${lineState(i)}`}
            onClick={() => answer(i)}
            disabled={picked !== null}
          >
            <span className="num">{i + 1}</span>
            <span className="code">{line || ' '}</span>
          </button>
        ))}
      </div>

      {picked !== null && (
        <div className="reveal" aria-live="polite">
          <div className="verdict">
            <span className={picked === round.targetLine ? 'ok' : 'no'}>
              {picked === round.targetLine ? 'correct' : 'not quite'}
            </span>
            <span className="explain">{explanation}</span>
          </div>
          <button type="button" className="advance" onClick={next}>
            next round →
          </button>
        </div>
      )}

      <p className="prog">program: {round.program.id}</p>

      <style>{`
        .spotdecision { font-family: 'IBM Plex Mono', monospace; border: 1px solid transparent; border-radius: 2px; padding: 0.125rem; }
        .spotdecision.passing { border-color: ${PASS}; }

        .spotdecision .setup { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; padding-bottom: 0.5rem; }
        .spotdecision .title { font-size: 0.8125rem; color: ${PANEL_INK}; }
        .spotdecision .streak { font-size: 0.75rem; color: ${COPPER}; flex: 0 0 auto; }

        .spotdecision .prompt { margin: 0 0 0.625rem; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .spotdecision .prompt .copper { color: ${COPPER}; }

        .spotdecision .jaxpr { background: #101215; border: 1px solid ${PANEL_RULE}; border-radius: 2px; max-height: 22rem; overflow-y: auto; overflow-x: auto; }
        .spotdecision .ln { display: flex; gap: 0.75rem; width: 100%; min-width: max-content; text-align: left; background: transparent; color: ${PANEL_MUTE}; border: none; border-left: 3px solid transparent; font: inherit; font-size: 0.6875rem; line-height: 1.6; padding: 0 0.75rem; cursor: pointer; }
        .spotdecision .ln:hover:not(:disabled) { color: ${PANEL_INK}; background: #16191d; }
        .spotdecision .ln:disabled { cursor: default; }
        .spotdecision .ln .num { flex: 0 0 auto; width: 2rem; text-align: right; color: ${PANEL_MUTE}; opacity: 0.7; }
        .spotdecision .ln .code { white-space: pre; }
        .spotdecision .ln.correct { border-left-color: ${PASS}; color: ${PASS}; background: #0f1a15; }
        .spotdecision .ln.correct .num { color: ${PASS}; }
        .spotdecision .ln.wrong { border-left-color: ${FAIL}; color: ${FAIL}; background: #1a1211; }
        .spotdecision .ln.wrong .num { color: ${FAIL}; }

        .spotdecision .reveal { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; padding-top: 0.75rem; font-size: 0.8125rem; }
        .spotdecision .verdict { display: flex; gap: 0.625rem; align-items: baseline; flex-wrap: wrap; min-width: 0; }
        .spotdecision .verdict .ok { color: ${PASS}; flex: 0 0 auto; }
        .spotdecision .verdict .no { color: ${FAIL}; flex: 0 0 auto; }
        .spotdecision .verdict .explain { color: ${PANEL_INK}; }
        .spotdecision .advance { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.75rem; cursor: pointer; flex: 0 0 auto; }
        .spotdecision .advance:hover { border-color: ${COPPER}; color: ${COPPER}; }

        .spotdecision .prog { margin: 0.625rem 0 0; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
      `}</style>
    </div>
  )
}
