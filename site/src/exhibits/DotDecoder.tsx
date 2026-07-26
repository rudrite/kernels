// The dot-decoder drill: read a real dot_general call, exactly the shapes
// and dimension_numbers XLA prints, and answer two questions about it:
// which lhs axes get contracted, and what shape falls out the other side.
// Every choice, right or wrong, is assembled from that instance's own
// lhs/rhs/out numbers; nothing is invented to make a distractor tempting.
import { useEffect, useMemo, useState } from 'react'
import corpus from '../data/ir-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.dot.streak'
const STREAK_TO_PASS = 10

type AxisPair = [number[], number[]]

interface DotInstance {
  program: string
  lhs: number[]
  rhs: number[]
  out: number[]
  dimension_numbers: [AxisPair, AxisPair]
}

const DOTS = corpus.dots as DotInstance[]

// array helpers shared by both questions

function arraysEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i])
}

function dedupeAgainst(candidates: number[][], taken: number[][]): number[][] {
  const seen = [...taken]
  const out: number[][] = []
  for (const cand of candidates) {
    if (seen.some((s) => arraysEqual(s, cand))) continue
    out.push(cand)
    seen.push(cand)
  }
  return out
}

// Rotations, a reversal, and adjacent swaps of a real array: always enough
// distinct reorderings to fill 3 distractors at the ranks (2-4) this corpus
// uses. If a rank is symmetric enough that every reordering matches (e.g. a
// square [128, 128]), the last resort pads the array with a trailing 1:
// still built from real values, just extended past the point of ambiguity.
function fallbackVariants(base: number[]): number[][] {
  const pool: number[][] = []
  for (let i = 1; i < base.length; i++) pool.push([...base.slice(i), ...base.slice(0, i)])
  pool.push([...base].reverse())
  for (let i = 0; i < base.length - 1; i++) {
    const v = [...base]
    const a = v[i]!
    v[i] = v[i + 1]!
    v[i + 1] = a
    pool.push(v)
  }
  pool.push([1, ...base])
  pool.push([...base, 1])
  return pool
}

function pickThreeDistinct(seeds: number[][], base: number[], correct: number[]): number[][] {
  const fromSeeds = dedupeAgainst(seeds, [correct])
  if (fromSeeds.length >= 3) return fromSeeds.slice(0, 3)
  const more = dedupeAgainst(fallbackVariants(base), [correct, ...fromSeeds])
  return [...fromSeeds, ...more].slice(0, 3)
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

interface Choice {
  arr: number[]
  correct: boolean
}

// Q1 distractors: read the rhs's contracting axes instead of the lhs's
// (swap axes), fold the batch axes into the contracting set (forget batch),
// reverse the axis order, or mistake the free axes for the contracting ones.
// `randomizeOrder` stays false for the very first render (server-rendered
// markup and the client's pre-hydration pass must agree exactly); a mount
// effect flips it true so the choice order only becomes random client-side.
function axisChoices(dot: DotInstance, randomizeOrder: boolean): Choice[] {
  const [[lhsContract, rhsContract], [lhsBatch]] = dot.dimension_numbers
  const rank = dot.lhs.length
  const free = Array.from({ length: rank }, (_, i) => i).filter(
    (i) => !lhsContract.includes(i) && !lhsBatch.includes(i),
  )
  const seeds: number[][] = [
    rhsContract,
    [...lhsContract, ...lhsBatch],
    [...lhsContract].reverse(),
    free,
    lhsBatch,
    lhsContract.map((i) => (i + 1) % Math.max(rank, 1)),
  ]
  const base = lhsContract.length > 0 ? lhsContract : [0]
  const distractors = pickThreeDistinct(seeds, base, lhsContract)
  const ordered = [lhsContract, ...distractors]
  return (randomizeOrder ? shuffle(ordered) : ordered).map((arr) => ({ arr, correct: arraysEqual(arr, lhsContract) }))
}

// Q2 distractors: swap which operand's free axes come first, treat the
// batch axes as ordinary free axes instead of pulling them to the front, or
// keep the batch prefix but swap the two free segments.
function shapeChoices(dot: DotInstance, randomizeOrder: boolean): Choice[] {
  const [[lhsContract, rhsContract], [lhsBatch, rhsBatch]] = dot.dimension_numbers
  const freeLhs = dot.lhs.filter((_, i) => !lhsContract.includes(i) && !lhsBatch.includes(i))
  const freeRhs = dot.rhs.filter((_, i) => !rhsContract.includes(i) && !rhsBatch.includes(i))
  const batch = dot.lhs.filter((_, i) => lhsBatch.includes(i))
  const freeLhsAll = dot.lhs.filter((_, i) => !lhsContract.includes(i))
  const freeRhsAll = dot.rhs.filter((_, i) => !rhsContract.includes(i))

  const seeds: number[][] = [
    [...freeRhs, ...freeLhs, ...batch],
    [...freeLhsAll, ...freeRhsAll],
    [...batch, ...freeRhs, ...freeLhs],
  ]
  const distractors = pickThreeDistinct(seeds, dot.out, dot.out)
  const ordered = [dot.out, ...distractors]
  return (randomizeOrder ? shuffle(ordered) : ordered).map((arr) => ({ arr, correct: arraysEqual(arr, dot.out) }))
}

function formatAxes(axes: number[]): string {
  if (axes.length === 0) return 'none (outer product)'
  if (axes.length === 1) return `axis ${axes[0]}`
  return `axes ${axes.join(', ')}`
}

function formatShape(shape: number[]): string {
  return `[${shape.join(', ')}]`
}

function formatDN(dn: [AxisPair, AxisPair]): string {
  const [[lc, rc], [lb, rb]] = dn
  const a = (xs: number[]) => `[${xs.join(', ')}]`
  return `((${a(lc)}, ${a(rc)}), (${a(lb)}, ${a(rb)}))`
}

function pickDot(exclude?: DotInstance): DotInstance {
  let d = DOTS[Math.floor(Math.random() * DOTS.length)]!
  while (DOTS.length > 1 && d === exclude) d = DOTS[Math.floor(Math.random() * DOTS.length)]!
  return d
}

type Stage = 'contract' | 'shape'

export default function DotDecoder() {
  // Starts on the corpus's first instance, unshuffled, so the first render
  // matches the server-rendered markup exactly; a mount effect below swaps
  // in a real random instance and choice order once hydration is safe.
  const [dot, setDot] = useState<DotInstance>(DOTS[0]!)
  const [stage, setStage] = useState<Stage>('contract')
  const [picked, setPicked] = useState<number | null>(null)
  const [streak, setStreak] = useState(0)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    setDot(pickDot())
    const raw = window.localStorage.getItem(STREAK_KEY)
    const n = raw === null ? 0 : Number(raw)
    setStreak(Number.isFinite(n) && n >= 0 ? n : 0)
    setHydrated(true)
  }, [])

  useEffect(() => {
    if (!hydrated) return
    window.localStorage.setItem(STREAK_KEY, String(streak))
  }, [streak, hydrated])

  const axisOptions = useMemo(() => axisChoices(dot, hydrated), [dot, hydrated])
  const shapeOptions = useMemo(() => shapeChoices(dot, hydrated), [dot, hydrated])
  const options = stage === 'contract' ? axisOptions : shapeOptions
  const format = stage === 'contract' ? formatAxes : formatShape

  const answer = (i: number) => {
    if (picked !== null) return // one tap per question
    setPicked(i)
    setStreak((s) => (options[i]!.correct ? s + 1 : 0))
  }

  const advance = () => {
    if (stage === 'contract') {
      setStage('shape')
      setPicked(null)
      return
    }
    setDot(pickDot(dot))
    setStage('contract')
    setPicked(null)
  }

  const passing = streak >= STREAK_TO_PASS

  return (
    <div className={`dotdecoder${passing ? ' passing' : ''}`}>
      <div className="setup">
        <dl className="shapes">
          <div>
            <dt>lhs</dt>
            <dd>{formatShape(dot.lhs)}</dd>
          </div>
          <div>
            <dt>rhs</dt>
            <dd>{formatShape(dot.rhs)}</dd>
          </div>
          <div className="dn">
            <dt>dimension_numbers</dt>
            <dd>{formatDN(dot.dimension_numbers)}</dd>
          </div>
        </dl>
        <div className="meta">
          <span className="streak">streak {streak}</span>
          <span className="prog">program: {dot.program}</span>
        </div>
      </div>

      <p className="prompt">
        <span className="step">{stage === 'contract' ? '1 of 2' : '2 of 2'}</span>
        {stage === 'contract' ? 'which lhs axes are contracted?' : 'what shape falls out the other side?'}
      </p>

      <div
        className="choices"
        role="group"
        aria-label={stage === 'contract' ? 'lhs contracting axes, 4 choices' : 'output shape, 4 choices'}
      >
        {options.map((opt, i) => {
          const chosen = picked === i
          const revealed = picked !== null
          const state = !revealed ? '' : opt.correct ? 'correct' : chosen ? 'wrong' : 'dim'
          return (
            <button
              key={i}
              type="button"
              className={`choice ${state}`}
              onClick={() => answer(i)}
              disabled={revealed}
              aria-pressed={chosen}
            >
              {format(opt.arr)}
            </button>
          )
        })}
      </div>

      {picked !== null && (
        <div className="reveal" aria-live="polite">
          <span className={options[picked]!.correct ? 'ok' : 'no'}>
            {options[picked]!.correct ? 'correct' : 'not quite'}
          </span>
          <button type="button" className="advance" onClick={advance}>
            {stage === 'contract' ? 'next question →' : 'next instance →'}
          </button>
        </div>
      )}

      <style>{`
        .dotdecoder { font-family: 'IBM Plex Mono', monospace; border: 1px solid transparent; border-radius: 2px; }
        .dotdecoder.passing { border-color: ${PASS}; }
        .dotdecoder { overflow-x: auto; }

        .setup { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .shapes { display: flex; flex-direction: column; gap: 0.3rem; margin: 0; min-width: 0; }
        .shapes > div { display: flex; gap: 0.6rem; align-items: baseline; }
        .shapes dt { flex: 0 0 auto; font-size: 0.625rem; letter-spacing: 0.08em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .shapes dd { margin: 0; font-size: 0.8125rem; color: ${PANEL_INK}; white-space: nowrap; }
        .shapes .dn dd { white-space: pre; }

        .meta { display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .meta .streak { color: ${COPPER}; font-size: 0.75rem; }

        .prompt { display: flex; gap: 0.75rem; align-items: baseline; margin: 0.75rem 0 0.625rem; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .prompt .step { font-size: 0.625rem; letter-spacing: 0.08em; color: ${PANEL_MUTE}; }

        .choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: 0.5rem; }
        .choice { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.8125rem; padding: 0.5rem 0.625rem; text-align: left; cursor: pointer; }
        .choice:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .choice:disabled { cursor: default; }
        .choice.correct { border-color: ${PASS}; color: ${PASS}; }
        .choice.wrong { border-color: ${FAIL}; color: ${FAIL}; }
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
