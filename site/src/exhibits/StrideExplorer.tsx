// The view explorer: build a chain of views and watch the map change while
// the storage underneath does not. The arithmetic lives in lib/strides.ts,
// which is checked against 48 chains executed on real torch, so what you
// see here is what torch would print.
import { useMemo, useState } from 'react'
import { applyAll, baseView, isContiguous, fmt, type Op } from '../lib/strides'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const BASES: { label: string; shape: number[] }[] = [
  { label: '(3, 4)', shape: [3, 4] },
  { label: '(4, 8)', shape: [4, 8] },
  { label: '(2, 3, 4)', shape: [2, 3, 4] },
]

const OPS: { label: string; op: Op; needs2d?: boolean }[] = [
  { label: '.t()', op: { kind: 't' }, needs2d: true },
  { label: '.transpose(0, 1)', op: { kind: 'transpose', a: 0, b: 1 } },
  { label: '.unsqueeze(0)', op: { kind: 'unsqueeze', dim: 0 } },
  { label: '[1:]', op: { kind: 'sliceRows', start: 1 } },
  { label: '[:, ::2]', op: { kind: 'strideCols', step: 2 } },
  { label: '.flatten()', op: { kind: 'flatten' } },
  { label: '.contiguous()', op: { kind: 'contiguous' } },
]

/** Every storage index this view reads, in the order it reads them. */
function visited(shape: number[], stride: number[], offset: number): number[] {
  const out: number[] = []
  const walk = (dim: number, at: number) => {
    if (dim === shape.length) {
      out.push(at)
      return
    }
    for (let i = 0; i < shape[dim]!; i++) walk(dim + 1, at + i * stride[dim]!)
  }
  walk(0, offset)
  return out
}

export default function StrideExplorer() {
  const [base, setBase] = useState(BASES[0]!)
  const [chain, setChain] = useState<{ label: string; op: Op }[]>([])

  const result = useMemo(() => {
    try {
      return { view: applyAll(base.shape, chain.map((c) => c.op)), error: null as string | null }
    } catch (e) {
      return { view: null, error: (e as Error).message }
    }
  }, [base, chain])

  const storageSize = base.shape.reduce((a, b) => a * b, 1)
  const touched = result.view ? new Set(visited(result.view.shape, result.view.stride, result.view.offset)) : new Set<number>()
  const order = result.view ? visited(result.view.shape, result.view.stride, result.view.offset) : []
  const code = `x = torch.arange(${storageSize}.).reshape${base.label.replace(/ /g, '')}\ny = x${chain.map((c) => c.label).join('')}`

  const contiguous = result.view ? isContiguous(result.view) : false
  const copied = result.view && chain.some((c) => c.op.kind === 'contiguous' || c.op.kind === 'flatten')

  return (
    <div className="strideexp">
      <div className="controls">
        <div className="group">
          <span className="k">start from</span>
          <div className="row">
            {BASES.map((b) => (
              <button
                key={b.label}
                type="button"
                className={base.label === b.label ? 'on' : ''}
                onClick={() => {
                  setBase(b)
                  setChain([])
                }}
              >
                {b.label}
              </button>
            ))}
          </div>
        </div>
        <div className="group">
          <span className="k">then apply</span>
          <div className="row">
            {OPS.map((o) => (
              <button key={o.label} type="button" onClick={() => setChain((c) => [...c, { label: o.label, op: o.op }])}>
                {o.label}
              </button>
            ))}
            <button type="button" className="clear" onClick={() => setChain([])} disabled={chain.length === 0}>
              clear
            </button>
          </div>
        </div>
      </div>

      <pre className="code">{code}</pre>

      {result.error ? (
        <p className="err">torch would refuse this: {result.error}</p>
      ) : (
        <div className="readout">
          <div className="facts">
            <div>
              <span className="k">shape</span>
              <span className="v">{fmt(result.view!.shape)}</span>
            </div>
            <div>
              <span className="k">stride</span>
              <span className="v">{fmt(result.view!.stride)}</span>
            </div>
            <div>
              <span className="k">offset</span>
              <span className="v">{result.view!.offset}</span>
            </div>
            <div>
              <span className="k">contiguous</span>
              <span className={contiguous ? 'v yes' : 'v no'}>{String(contiguous)}</span>
            </div>
          </div>

          <div className="storage">
            <span className="k">
              the storage {copied ? '(a copy: this chain materialized a new one)' : '(untouched: every step above only rewrote the map)'}
            </span>
            <div className="cells" style={{ gridTemplateColumns: `repeat(${Math.min(12, storageSize)}, 1fr)` }}>
              {Array.from({ length: storageSize }, (_, i) => (
                <span key={i} className={touched.has(i) ? 'cell on' : 'cell'} title={`storage[${i}]`}>
                  {i}
                </span>
              ))}
            </div>
            <span className="note">
              read order: {order.slice(0, 14).join(', ')}
              {order.length > 14 ? ' …' : ''}
            </span>
          </div>
        </div>
      )}

      <style>{`
        .strideexp { font-family: 'IBM Plex Mono', monospace; }
        .controls { display: flex; flex-direction: column; gap: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .group { display: flex; flex-direction: column; gap: 0.35rem; }
        .k { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .row { display: flex; gap: 0.375rem; flex-wrap: wrap; }
        .row button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.6rem; cursor: pointer; }
        .row button:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .row button.on { border-color: ${COPPER}; color: ${COPPER}; background: #1c1f24; }
        .row button.clear { color: ${PANEL_MUTE}; }
        .row button:disabled { opacity: 0.4; cursor: default; }

        .code { margin: 0.75rem 0; font-size: 0.8125rem; line-height: 1.7; color: ${PANEL_INK}; white-space: pre-wrap; }
        .err { font-size: 0.8125rem; color: ${FAIL}; }

        .readout { display: flex; flex-direction: column; gap: 0.875rem; }
        .facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr)); gap: 0.625rem; }
        .facts > div { display: flex; flex-direction: column; gap: 0.2rem; }
        .facts .v { font-size: 0.875rem; color: ${COPPER}; }
        .facts .v.yes { color: ${PASS}; }
        .facts .v.no { color: ${PANEL_INK}; }

        .storage { display: flex; flex-direction: column; gap: 0.4rem; }
        .cells { display: grid; gap: 3px; }
        .cell { display: flex; align-items: center; justify-content: center; height: 1.6rem; font-size: 0.625rem; color: ${PANEL_RULE}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; }
        .cell.on { color: ${COPPER}; border-color: ${COPPER}; background: #1c1f24; }
        .note { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
      `}</style>
    </div>
  )
}
