// The IR x-ray: one program shown at three layers, hover-synced. The dumps
// are real (generated with jax; provenance on the panel), and both shown IRs
// are backend-independent. The optimized-HLO layer is deliberately absent:
// producing and reading it on a real TPU is Stage 2's lab.
import { useState } from 'react'
import ir from '../data/ir-attention.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'

interface Group {
  id: string
  note: string
  src: number[]
  jaxpr: number[]
  hlo: number[]
}
const GROUPS = ir.groups as Group[]

type Col = 'src' | 'jaxpr' | 'hlo'
const COLS: { key: Col; label: string; lines: string[] }[] = [
  { key: 'src', label: 'JAX source', lines: ir.source },
  { key: 'jaxpr', label: 'jaxpr', lines: ir.jaxpr },
  { key: 'hlo', label: 'StableHLO', lines: ir.stablehlo },
]

export default function IRXray() {
  const [active, setActive] = useState<string | null>(null)
  const groupOf = (col: Col, line: number) => GROUPS.find((g) => g[col].includes(line))?.id ?? null
  const activeGroup = GROUPS.find((g) => g.id === active)

  return (
    <div className="xray">
      <div className="cols">
        {COLS.map((col) => (
          <div className="col" key={col.key}>
            <div className="col-head">{col.label}</div>
            <pre onMouseLeave={() => setActive(null)}>
              {col.lines.map((line, n) => {
                const g = groupOf(col.key, n)
                const hot = g !== null && g === active
                return (
                  <button
                    key={n}
                    type="button"
                    className={`ln ${hot ? 'hot' : ''} ${g ? 'mapped' : ''}`}
                    onMouseEnter={() => setActive(g)}
                    onFocus={() => setActive(g)}
                    tabIndex={g ? 0 : -1}
                  >
                    {line || ' '}
                  </button>
                )
              })}
            </pre>
          </div>
        ))}
      </div>
      <p className="note" aria-live="polite">
        {activeGroup ? `${activeGroup.id} · ${activeGroup.note}` : 'hover or tab through any line: the same computation lights up at every layer'}
      </p>

      <style>{`
        .xray { font-family: 'IBM Plex Mono', monospace; }
        .xray .cols { display: grid; grid-template-columns: minmax(180px, 0.7fr) 1fr 1.3fr; gap: 1px; background: ${PANEL_RULE}; }
        .xray .col { background: #101215; min-width: 0; }
        .xray .col-head { font-size: 0.6875rem; letter-spacing: 0.12em; text-transform: uppercase; color: ${PANEL_MUTE}; padding: 0.5rem 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .xray pre { margin: 0; padding: 0.5rem 0; overflow-x: auto; }
        .xray .ln { display: block; width: 100%; text-align: left; background: transparent; border: none; border-left: 3px solid transparent; color: ${PANEL_MUTE}; font: inherit; font-size: 0.6875rem; line-height: 1.6; padding: 0 0.75rem; white-space: pre; cursor: default; }
        .xray .ln.mapped { color: ${PANEL_INK}; cursor: pointer; }
        .xray .ln.hot { background: #20150f; border-left-color: ${COPPER}; color: ${COPPER}; }
        .xray .note { font-size: 0.75rem; color: ${PANEL_INK}; padding: 0.625rem 0.125rem 0; margin: 0; min-height: 2.5em; }
        @media (max-width: 900px) { .xray .cols { grid-template-columns: 1fr; } }
      `}</style>
    </div>
  )
}
