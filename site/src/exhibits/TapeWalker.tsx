// The tape, walked node by node. Each chain here was built by running the
// expression on torch and following grad_fn through next_functions, so the
// node names and the branching are what autograd actually recorded rather
// than a drawing of what it probably does.
import { useState } from 'react'
import corpus from '../data/pytorch/tape-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'

interface TapeNode {
  name: string
  depth: number
  children: string[]
  is_leaf_accumulator: boolean
}

interface Item {
  code: string
  title: string
  nodes: TapeNode[]
  why: string
  grad_fn: string
}

const DATA = corpus as { torch: string; items: Item[] }
const ITEMS = DATA.items

export default function TapeWalker() {
  const [at, setAt] = useState(0)
  const [step, setStep] = useState(0)
  const item = ITEMS[at]!
  const visible = item.nodes.slice(0, step + 1)

  const choose = (i: number) => {
    setAt(i)
    setStep(0)
  }

  return (
    <div className="tape">
      <div className="pick">
        <span className="k">expression</span>
        <div className="row">
          {ITEMS.map((it, i) => (
            <button key={it.code} type="button" className={i === at ? 'on' : ''} onClick={() => choose(i)}>
              {it.code}
            </button>
          ))}
        </div>
      </div>

      <div className="head">
        <div>
          <span className="k">y.grad_fn</span>
          <code className="v">{item.grad_fn}</code>
        </div>
        <div className="controls">
          <button type="button" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
            ←
          </button>
          <span className="pos">
            {step + 1} / {item.nodes.length}
          </span>
          <button
            type="button"
            onClick={() => setStep((s) => Math.min(item.nodes.length - 1, s + 1))}
            disabled={step === item.nodes.length - 1}
          >
            walk backward →
          </button>
        </div>
      </div>

      <ol className="nodes">
        {visible.map((n, i) => (
          <li key={`${n.name}-${i}`} className={`${n.is_leaf_accumulator ? 'acc' : ''}${i === step ? ' here' : ''}`}>
            <span className="depth" aria-hidden="true">
              {'·'.repeat(n.depth)}
            </span>
            <code className="name">{n.name}</code>
            {n.is_leaf_accumulator ? (
              <span className="tag">this one writes into a leaf's .grad</span>
            ) : n.children.length > 1 ? (
              <span className="tag branch">branches into {n.children.length}</span>
            ) : n.children.length === 0 ? (
              <span className="tag">the end of this branch</span>
            ) : null}
          </li>
        ))}
      </ol>

      {step === item.nodes.length - 1 && <p className="why">{item.why}</p>}

      <p className="foot">
        chains recorded on torch {DATA.torch} by following grad_fn through next_functions · backward walks these in this
        order and frees them as it goes, which is why a second call finds nothing
      </p>

      <style>{`
        .tape { font-family: 'IBM Plex Mono', monospace; }
        .k { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .pick { display: flex; flex-direction: column; gap: 0.35rem; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .row { display: flex; gap: 0.375rem; flex-wrap: wrap; }
        .row button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.71875rem; padding: 0.28rem 0.6rem; cursor: pointer; }
        .row button:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .row button.on { border-color: ${COPPER}; color: ${COPPER}; background: #1c1f24; }

        .head { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; padding: 0.75rem 0; }
        .head .v { display: block; font-size: 0.875rem; color: ${COPPER}; margin-top: 0.2rem; }
        .controls { display: flex; align-items: center; gap: 0.5rem; }
        .controls button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.25rem 0.65rem; cursor: pointer; }
        .controls button:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .controls button:disabled { opacity: 0.4; cursor: default; }
        .pos { font-size: 0.75rem; color: ${PANEL_MUTE}; }

        .nodes { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; }
        .nodes li { display: flex; align-items: baseline; gap: 0.6rem; padding: 0.3rem 0.5rem; border-left: 2px solid ${PANEL_RULE}; }
        .nodes li.here { border-left-color: ${COPPER}; background: #1a1d21; }
        .nodes li.acc { border-left-color: ${PASS}; }
        .depth { color: ${PANEL_RULE}; letter-spacing: 0.3em; }
        .name { font-size: 0.8125rem; color: ${PANEL_INK}; }
        .nodes li.acc .name { color: ${PASS}; }
        .tag { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .tag.branch { color: ${COPPER}; }

        .why { margin: 0.875rem 0 0; font-size: 0.8125rem; color: ${PANEL_INK}; line-height: 1.65; max-width: 70ch; border-left: 3px solid ${COPPER}; padding-left: 0.875rem; }
        .foot { margin-top: 0.875rem; padding-top: 0.625rem; border-top: 1px solid ${PANEL_RULE}; font-size: 0.6875rem; color: ${PANEL_MUTE}; line-height: 1.6; }
      `}</style>
    </div>
  )
}
