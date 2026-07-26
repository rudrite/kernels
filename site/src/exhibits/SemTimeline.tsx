// Remote-DMA semaphore ordering on a 4-chip ring, teaching the deadlock class
// that shows up when a wait gets issued ahead of the send it's paired with.
//
// Everything here is derived from one explicit dependency graph, not from
// per-frame scripting. Two rules hold for a correct ring all-gather:
//   - a chip's wait can't resolve before its left neighbor has sent that
//     hop's data
//   - a chip's send for hop > 1 can't fire before that same chip's own
//     previous-hop wait resolved (there's nothing received yet to relay)
// The 'reorder one wait' toggle simulates a real bug: chip 2 issues its
// hop-1 wait before its hop-1 send. Two honest consequences follow, both
// added as graph edges rather than asserted:
//   - chip 2's own send now has to wait for its own wait, because a chip's
//     instructions issue in program order
//   - chip 1's send lands in chip 2's single relay buffer; it can't land
//     safely until chip 2 has drained that buffer with its own send, which
//    : thanks to the reorder: chip 2 hasn't done yet
// Those three ops form a real cycle. Every other blocked op on the timeline
// is a chip that transitively depends on one of them; the reachability is
// computed, not hand-placed.
import { useState } from 'react'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const IDLE_EDGE = '#4a525a'
const FAIL = '#c4574a'

const N_CHIPS = 4
const HOPS = [1, 2, 3] as const
type Hop = (typeof HOPS)[number]
type Lane = 'send' | 'wait'

interface Op {
  chip: number
  lane: Lane
  hop: Hop
}

const key = (chip: number, lane: Lane, hop: number): string => `${lane}-${chip}-${hop}`

const OPS: Op[] = []
for (let chip = 0; chip < N_CHIPS; chip++) {
  for (const hop of HOPS) {
    OPS.push({ chip, lane: 'send', hop })
    OPS.push({ chip, lane: 'wait', hop })
  }
}

/** The dependency graph: op key -> keys of ops it requires to have fired first. */
function buildDeps(reordered: boolean): Record<string, string[]> {
  const deps: Record<string, string[]> = {}
  for (const op of OPS) deps[key(op.chip, op.lane, op.hop)] = []
  for (let chip = 0; chip < N_CHIPS; chip++) {
    for (const hop of HOPS) {
      if (hop > 1) deps[key(chip, 'send', hop)]!.push(key(chip, 'wait', (hop - 1) as Hop))
      deps[key(chip, 'wait', hop)]!.push(key((chip - 1 + N_CHIPS) % N_CHIPS, 'send', hop))
    }
  }
  if (reordered) {
    // chip 2's send now needs its own (already-blocked) wait: program order, flipped
    deps[key(2, 'send', 1)]!.push(key(2, 'wait', 1))
    // chip 1's send into chip 2's relay buffer needs chip 2 to have drained it first
    deps[key(1, 'send', 1)]!.push(key(2, 'send', 1))
  }
  return deps
}

/** Longest-path depth of each op in an acyclic graph: its earliest possible tick. */
function computeTicks(deps: Record<string, string[]>): Record<string, number> {
  const tick: Record<string, number> = {}
  const depthOf = (node: string): number => {
    if (node in tick) return tick[node]!
    const ds = deps[node]!
    const d = ds.length === 0 ? 0 : 1 + Math.max(...ds.map(depthOf))
    tick[node] = d
    return d
  }
  for (const k of Object.keys(deps)) depthOf(k)
  return tick
}

/** First cycle found (DFS, white/gray/black), returned as the closed walk that found it. */
function findCycle(deps: Record<string, string[]>): string[] | null {
  const WHITE = 0
  const GRAY = 1
  const BLACK = 2
  const color: Record<string, number> = {}
  for (const k of Object.keys(deps)) color[k] = WHITE
  const path: string[] = []
  const visit = (node: string): string[] | null => {
    color[node] = GRAY
    path.push(node)
    for (const dep of deps[node]!) {
      if (color[dep] === GRAY) {
        const idx = path.indexOf(dep)
        return [...path.slice(idx), dep]
      }
      if (color[dep] === WHITE) {
        const found = visit(dep)
        if (found) return found
      }
    }
    path.pop()
    color[node] = BLACK
    return null
  }
  for (const k of Object.keys(deps)) {
    if (color[k] === WHITE) {
      const found = visit(k)
      if (found) return found
    }
  }
  return null
}

/** Every op that transitively requires a cycle member: doomed to never fire. */
function findDoomed(deps: Record<string, string[]>, cycle: string[]): Set<string> {
  const requirers: Record<string, string[]> = {}
  for (const k of Object.keys(deps)) requirers[k] = []
  for (const [node, ds] of Object.entries(deps)) for (const d of ds) requirers[d]!.push(node)
  const doomed = new Set(cycle)
  const stack = [...cycle]
  while (stack.length > 0) {
    const n = stack.pop()!
    for (const r of requirers[n]!) {
      if (!doomed.has(r)) {
        doomed.add(r)
        stack.push(r)
      }
    }
  }
  return doomed
}

const describeOp = (k: string): string => {
  const [lane, chip, hop] = k.split('-')
  return `chip ${chip} hop-${hop} ${lane}`
}

// The reference schedule (unreordered) fixes each op's tick on the timeline :
// that axis doesn't move when the toggle flips; only which cells light up does.
const NORMAL_DEPS = buildDeps(false)
const REORDERED_DEPS = buildDeps(true)
const REF_TICK = computeTicks(NORMAL_DEPS)
const CYCLE = findCycle(REORDERED_DEPS)
if (!CYCLE) throw new Error('SemTimeline: the reordered dependency graph is expected to contain a cycle')
const DOOMED = findDoomed(REORDERED_DEPS, CYCLE)
const MAX_STEP = Math.max(...Object.values(REF_TICK))
const CYCLE_CAPTION = `deadlock: ${CYCLE.map(describeOp).join(', which waits on ')}: the ring never turns.`

type CellState = 'pending' | 'fired' | 'blocked'

function cellState(k: string, reordered: boolean, step: number): CellState {
  if (step < REF_TICK[k]!) return 'pending'
  return reordered && DOOMED.has(k) ? 'blocked' : 'fired'
}

function stepCaption(step: number, reordered: boolean): string {
  const due = OPS.filter((op) => REF_TICK[key(op.chip, op.lane, op.hop)] === step)
  const bits = due.map((op) => {
    const state = cellState(key(op.chip, op.lane, op.hop), reordered, step)
    const tag = state === 'blocked' ? 'blocked' : 'fires'
    return `chip ${op.chip} ${op.lane} (hop ${op.hop}) ${tag}`
  })
  return `tick ${step}: ${bits.join(' · ')}`
}

const LEFT_W = 128
const COL_W = 132
const ROW_H = 30
const CHIP_GAP = 12
const TOP = 30
const W = LEFT_W + HOPS.length * COL_W + 16
const H = TOP + N_CHIPS * (2 * ROW_H) + (N_CHIPS - 1) * CHIP_GAP + 14

function rowY(chip: number, lane: Lane): number {
  return TOP + chip * (2 * ROW_H + CHIP_GAP) + (lane === 'send' ? 0 : ROW_H)
}

export default function SemTimeline() {
  const [reordered, setReordered] = useState(false)
  const [step, setStep] = useState(0)

  const deadlocked = reordered && step >= REF_TICK[key(2, 'wait', 1)]!

  return (
    <div className={`sem-timeline${deadlocked ? ' deadlocked' : ''}`}>
      <div className="row controls">
        <button
          type="button"
          className={reordered ? 'toggle active' : 'toggle'}
          aria-pressed={reordered}
          onClick={() => {
            setReordered((r) => !r)
            setStep(0)
          }}
        >
          reorder one wait
        </button>
        <span className="hint">chip 2: hop-1 wait issued before hop-1 send</span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Semaphore timeline, ${reordered ? 'reordered' : 'normal'} configuration, tick ${step} of ${MAX_STEP}: ${stepCaption(step, reordered)}`}
      >
        {HOPS.map((hop, hi) => (
          <text key={hop} x={LEFT_W + hi * COL_W + COL_W / 2} y={TOP - 12} textAnchor="middle" fill={PANEL_MUTE} fontSize={10.5} letterSpacing={0.8} fontFamily="inherit">
            HOP {hop}
          </text>
        ))}

        {Array.from({ length: N_CHIPS }, (_, chip) => (
          <g key={chip}>
            <text x={0} y={rowY(chip, 'send') + ROW_H / 2 + 4} fill={PANEL_INK} fontSize={11} fontFamily="inherit">
              chip {chip}
            </text>
            <text x={0} y={rowY(chip, 'send') + ROW_H + ROW_H / 2 + 4} fill={PANEL_MUTE} fontSize={9.5} fontFamily="inherit">
              → {(chip + 1) % N_CHIPS} / ← {(chip - 1 + N_CHIPS) % N_CHIPS}
            </text>
            {(['send', 'wait'] as Lane[]).map((lane) => (
              <text key={lane} x={40} y={rowY(chip, lane) + ROW_H / 2 + 4} fill={PANEL_MUTE} fontSize={9.5} letterSpacing={0.5} fontFamily="inherit">
                {lane}
              </text>
            ))}
            {HOPS.map((hop, hi) => {
              const cx = LEFT_W + hi * COL_W + COL_W / 2
              return (
                <g key={hop}>
                  {(['send', 'wait'] as Lane[]).map((lane) => {
                    const k = key(chip, lane, hop)
                    const state = cellState(k, reordered, step)
                    const cy = rowY(chip, lane) + ROW_H / 2
                    const fill = state === 'fired' ? (lane === 'send' ? STEEL : COPPER) : 'transparent'
                    const stroke = state === 'blocked' ? FAIL : state === 'fired' ? (lane === 'send' ? STEEL : COPPER) : IDLE_EDGE
                    const dashed = lane === 'wait' && state !== 'fired'
                    return (
                      <g key={lane}>
                        <rect
                          x={cx - 44}
                          y={cy - 10}
                          width={88}
                          height={20}
                          rx={2}
                          fill={fill}
                          fillOpacity={state === 'fired' ? 0.22 : 1}
                          stroke={stroke}
                          strokeWidth={state === 'blocked' ? 1.5 : 1}
                          strokeDasharray={dashed ? '3 2' : undefined}
                        />
                        <text x={cx} y={cy + 3.5} textAnchor="middle" fill={state === 'pending' ? PANEL_MUTE : PANEL_INK} fontSize={9.5} fontFamily="inherit">
                          {state === 'blocked' ? 'blocked' : `t${REF_TICK[k]}`}
                        </text>
                        {reordered && chip === 2 && hop === 1 && (
                          <text x={cx + 38} y={cy - 12} textAnchor="end" fill={PANEL_MUTE} fontSize={8.5} fontFamily="inherit">
                            {lane === 'wait' ? 'issued 1st' : 'issued 2nd'}
                          </text>
                        )}
                      </g>
                    )
                  })}
                </g>
              )
            })}
          </g>
        ))}
      </svg>

      <p className="caption" aria-live="polite">
        <span className="fno">
          {String(step).padStart(2, '0')}/{String(MAX_STEP).padStart(2, '0')}
        </span>
        {stepCaption(step, reordered)}
      </p>

      {deadlocked && <p className="deadlock-caption">{CYCLE_CAPTION}</p>}

      <div className="controls step">
        <button onClick={() => setStep(0)} disabled={step === 0} aria-label="Back to start">
          |«
        </button>
        <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0} aria-label="Previous step">
          «
        </button>
        <button onClick={() => setStep((s) => Math.min(MAX_STEP, s + 1))} disabled={step === MAX_STEP} aria-label="Next step">
          »
        </button>
        <input
          type="range"
          min={0}
          max={MAX_STEP}
          value={step}
          onChange={(e) => setStep(Number(e.target.value))}
          aria-label="Scrub through ticks"
        />
      </div>

      <div className="legend">
        <span><i className="sw send" /> send fired</span>
        <span><i className="sw wait" /> wait fired</span>
        <span><i className="sw blocked" /> blocked (never resolves)</span>
      </div>

      <style>{`
        .sem-timeline { font-family: 'IBM Plex Mono', monospace; overflow-x: auto; }
        .sem-timeline .row.controls { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; padding-bottom: 0.75rem; }
        .sem-timeline .toggle { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.35rem 0.7rem; cursor: pointer; }
        .sem-timeline .toggle:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .sem-timeline .toggle.active { border-color: ${FAIL}; color: ${FAIL}; }
        .sem-timeline .hint { font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.02em; }
        .sem-timeline svg { width: 100%; min-width: 520px; height: auto; display: block; }
        .sem-timeline.deadlocked { border: 1px solid ${FAIL}; border-radius: 2px; padding: 0.75rem; margin: -0.75rem; }
        .sem-timeline .caption { font-size: 0.8125rem; color: ${PANEL_INK}; padding: 0.625rem 0.125rem 0; min-height: 2.4em; margin: 0; }
        .sem-timeline .fno { color: ${PANEL_MUTE}; margin-right: 0.75em; }
        .sem-timeline .deadlock-caption { font-size: 0.8125rem; color: ${FAIL}; margin: 0.4rem 0.125rem 0; }
        .sem-timeline .controls.step { display: flex; gap: 0.5rem; align-items: center; padding: 0.625rem 0.125rem 0.5rem; }
        .sem-timeline .controls.step button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.25rem 0.625rem; cursor: pointer; }
        .sem-timeline .controls.step button:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .sem-timeline .controls.step button:disabled { opacity: 0.35; cursor: default; }
        .sem-timeline .controls.step input[type='range'] { flex: 1; accent-color: ${COPPER}; }
        .sem-timeline .legend { display: flex; gap: 1.25rem; flex-wrap: wrap; font-size: 0.6875rem; color: ${PANEL_MUTE}; padding-top: 0.25rem; }
        .sem-timeline .legend .sw { display: inline-block; width: 9px; height: 9px; margin-right: 0.4em; border-radius: 1px; vertical-align: middle; }
        .sem-timeline .legend .sw.send { background: ${STEEL}; }
        .sem-timeline .legend .sw.wait { background: ${COPPER}; }
        .sem-timeline .legend .sw.blocked { background: transparent; border: 1.5px solid ${FAIL}; }
      `}</style>
    </div>
  )
}
