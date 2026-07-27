// Pick a sharding for each operand of a matmul and see what the partitioner
// inserted. Every combination here was compiled on eight devices and the
// collectives were read out of the module, so this shows what GSPMD chose
// rather than what a rule of thumb predicts. The two disagree more than you
// would expect, which is the reason the instrument exists.
import { useState } from 'react'
import corpus from '../data/xla/sharding-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'

interface Item {
  a: string
  b: string
  out: string
  collectives: string[]
}

const DATA = corpus as {
  jax: string
  devices: number
  mesh: string
  shapes: string
  specs: Record<string, string>
  items: Item[]
}

const ITEMS = DATA.items
const NAMES = Object.keys(DATA.specs)

export default function CollectivePicker() {
  const [at, setAt] = useState(0)
  const item = ITEMS[at]!
  const free = item.collectives.length === 0

  return (
    <div className="picker">
      <div className="cases">
        <span className="k">compiled combinations</span>
        <div className="row">
          {ITEMS.map((it, i) => (
            <button key={i} type="button" className={i === at ? 'on' : ''} onClick={() => setAt(i)}>
              {i + 1}
            </button>
          ))}
        </div>
      </div>

      <div className="board">
        <div className="operands">
          <div className="operand">
            <span className="k">left operand</span>
            <span className="v">{item.a}</span>
            <code className="spec">{DATA.specs[item.a]}</code>
          </div>
          <span className="times">x</span>
          <div className="operand">
            <span className="k">right operand</span>
            <span className="v">{item.b}</span>
            <code className="spec">{DATA.specs[item.b]}</code>
          </div>
          <span className="times">=</span>
          <div className="operand">
            <span className="k">asked-for output</span>
            <span className="v">{item.out}</span>
            <code className="spec">{DATA.specs[item.out]}</code>
          </div>
        </div>

        <div className={`result${free ? ' free' : ''}`}>
          <span className="k">what the partitioner inserted</span>
          {free ? (
            <p className="none">
              nothing. The shardings line up with the contraction, so every device already holds what it needs and the
              matmul runs with no communication at all.
            </p>
          ) : (
            <ul className="colls">
              {item.collectives.map((c) => (
                <li key={c}>
                  <code>{c}</code>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <p className="note">
        Worth noticing across all six: not one of them inserted an `all-reduce`. Sharding the contracted dimension is the
        textbook case for reducing partial results, and this partitioner preferred to gather the operand instead. That is a
        choice, made per program and per backend, which is why the module is the only place the answer actually lives.
      </p>

      <p className="foot">
        {DATA.items.length} combinations compiled on {DATA.devices} host-platform devices, mesh {DATA.mesh}, {DATA.shapes},
        jax {DATA.jax} · a TPU mesh may choose differently, and LAB·X3 is how you check
      </p>

      <style>{`
        .picker { font-family: 'IBM Plex Mono', monospace; }
        .k { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .cases { display: flex; flex-direction: column; gap: 0.35rem; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .row { display: flex; gap: 0.375rem; flex-wrap: wrap; }
        .row button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.25rem 0.7rem; cursor: pointer; }
        .row button:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .row button.on { border-color: ${COPPER}; color: ${COPPER}; background: #1c1f24; }

        .board { display: flex; flex-direction: column; gap: 1rem; padding: 0.875rem 0; }
        .operands { display: flex; align-items: flex-end; gap: 0.75rem; flex-wrap: wrap; }
        .operand { display: flex; flex-direction: column; gap: 0.2rem; min-width: 9rem; }
        .operand .v { font-size: 0.8125rem; color: ${PANEL_INK}; }
        .operand .spec { font-size: 0.6875rem; color: ${COPPER}; }
        .times { color: ${PANEL_MUTE}; padding-bottom: 0.2rem; }

        .result { display: flex; flex-direction: column; gap: 0.4rem; border-left: 3px solid ${COPPER}; padding-left: 0.875rem; }
        .result.free { border-left-color: ${PASS}; }
        .colls { list-style: none; margin: 0; padding: 0; display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .colls code { font-size: 0.875rem; color: ${COPPER}; border: 1px solid ${COPPER}; border-radius: 2px; padding: 0.2rem 0.55rem; }
        .none { margin: 0; font-size: 0.8125rem; color: ${PASS}; line-height: 1.6; max-width: 64ch; }

        .note { font-size: 0.78125rem; color: ${PANEL_INK}; line-height: 1.65; max-width: 70ch; border-top: 1px solid ${PANEL_RULE}; padding-top: 0.75rem; }
        .foot { margin-top: 0.5rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; line-height: 1.6; }
      `}</style>
    </div>
  )
}
