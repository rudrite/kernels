// The pass pipeline, stepped. Every entry is a real dump file from one
// compile of this course's attention program, so the order, the pipeline
// names, and the repeats are what the compiler actually did rather than a
// tidied-up version of it. Layout assignment is marked because chapter 04
// calls it the watershed, and the repeated names are the fixpoint loops.
import { useState } from 'react'
import corpus from '../data/xla/pipeline-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'

interface Step {
  n: number
  pipeline: string
  after: string
  before: string
}

const DATA = corpus as { jax: string; backend: string; program: string; steps: Step[] }
const STEPS = DATA.steps

const WATERSHED = STEPS.findIndex((s) => s.before === 'layout-assignment')

/** How many earlier steps ran under this same pipeline name. */
const repeatOf = (i: number) => STEPS.slice(0, i).filter((s) => s.pipeline === STEPS[i]!.pipeline).length

export default function PipelineStepper() {
  const [at, setAt] = useState(0)
  const step = STEPS[at]!
  const side = WATERSHED === -1 ? 'unknown' : at < WATERSHED ? 'before' : at === WATERSHED ? 'at' : 'after'
  const repeat = repeatOf(at)

  return (
    <div className="stepper">
      <div className="head">
        <div className="pos">
          <span className="k">step</span>
          <span className="v">
            {String(step.n).padStart(4, '0')} <span className="of">of {STEPS.length}</span>
          </span>
        </div>
        <div className="controls">
          <button type="button" onClick={() => setAt((i) => Math.max(0, i - 1))} disabled={at === 0}>
            ←
          </button>
          <input
            type="range"
            min={0}
            max={STEPS.length - 1}
            value={at}
            onChange={(e) => setAt(Number(e.target.value))}
            aria-label="pass step"
          />
          <button type="button" onClick={() => setAt((i) => Math.min(STEPS.length - 1, i + 1))} disabled={at === STEPS.length - 1}>
            →
          </button>
        </div>
      </div>

      <div className="track" aria-hidden="true">
        {STEPS.map((s, i) => (
          <span
            key={s.n}
            className={`tick${i === at ? ' here' : ''}${i === WATERSHED ? ' watershed' : ''}`}
            title={`${s.pipeline}: ${s.before}`}
            onClick={() => setAt(i)}
          />
        ))}
      </div>

      <div className="detail">
        <div className="row">
          <span className="k">pipeline</span>
          <code className="v">{step.pipeline}</code>
          {repeat > 0 && <span className="tag">rerun · {repeat} earlier step{repeat === 1 ? '' : 's'} under this name</span>}
        </div>
        <div className="row">
          <span className="k">just finished</span>
          <code className="v mute">{step.after}</code>
        </div>
        <div className="row">
          <span className="k">about to run</span>
          <code className="v">{step.before}</code>
          {step.before === 'layout-assignment' && <span className="tag hot">the watershed: layouts bind here</span>}
        </div>
        <p className="note">
          {side === 'before'
            ? 'still layout-free: the module can be rewritten without minding how anything sits in memory'
            : side === 'at'
              ? 'every shape is about to acquire a layout, and later passes have to respect it'
              : 'layout-bound: from here the passes work with memory order fixed, which is why fusion and copy insertion live on this side'}
        </p>
      </div>

      <p className="foot">
        {DATA.steps.length} steps from one compile of {DATA.program} on {DATA.backend}, jax {DATA.jax} · a TPU compile of the
        same program files 88 steps under different names, which chapter 04 shows
      </p>

      <style>{`
        .stepper { font-family: 'IBM Plex Mono', monospace; }
        .head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
        .k { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .pos { display: flex; flex-direction: column; gap: 0.2rem; }
        .pos .v { font-size: 1rem; color: ${COPPER}; }
        .pos .of { color: ${PANEL_MUTE}; font-size: 0.75rem; }
        .controls { display: flex; align-items: center; gap: 0.5rem; }
        .controls button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.8125rem; padding: 0.2rem 0.6rem; cursor: pointer; }
        .controls button:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .controls button:disabled { opacity: 0.4; cursor: default; }
        .controls input { width: 12rem; accent-color: ${COPPER}; }

        .track { display: flex; gap: 2px; margin: 0.75rem 0; }
        .tick { flex: 1; height: 0.75rem; background: #1c1f24; border: 1px solid ${PANEL_RULE}; border-radius: 1px; cursor: pointer; }
        .tick.watershed { background: #2a2119; border-color: ${COPPER}; }
        .tick.here { background: ${COPPER}; border-color: ${COPPER}; }

        .detail { display: flex; flex-direction: column; gap: 0.5rem; padding-top: 0.5rem; border-top: 1px solid ${PANEL_RULE}; }
        .row { display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap; }
        .row .v { font-size: 0.8125rem; color: ${PANEL_INK}; }
        .row .v.mute { color: ${PANEL_MUTE}; }
        .tag { font-size: 0.6875rem; color: ${PANEL_MUTE}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; padding: 0.1rem 0.4rem; }
        .tag.hot { color: ${PASS}; border-color: ${PASS}; }
        .note { margin: 0.25rem 0 0; font-size: 0.75rem; color: ${PANEL_MUTE}; line-height: 1.6; max-width: 62ch; }
        .foot { margin-top: 0.875rem; padding-top: 0.625rem; border-top: 1px solid ${PANEL_RULE}; font-size: 0.6875rem; color: ${PANEL_MUTE}; line-height: 1.6; }
      `}</style>
    </div>
  )
}
