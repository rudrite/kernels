// The corpus x-ray: IRXray generalized over every program in the reading
// gym's corpus. Pick a program, hover or tab through either column, and
// the equation it belongs to lights up at the other layer too. The line
// mapping itself is not hand-authored: scripts/gen_ir_maps.py walks the
// jaxpr and StableHLO text mechanically and writes data/ir-maps.json,
// skipping anything it can't align with confidence rather than guessing.
import { useState } from 'react'
import corpus from '../data/ir-corpus.json'
import irMaps from '../data/ir-maps.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#9aa1a8'
const PANEL_RULE = '#23272c'

interface Program {
  id: string
  title: string
  note: string
  source: string[]
  jaxpr: string[]
  stablehlo: string[]
}
const PROGRAMS = corpus.programs as Program[]

interface MapGroup {
  op: string
  jaxpr: number[]
  hlo: number[]
}
const MAPS = irMaps as Record<string, { groups: MapGroup[] }>

type Col = 'jaxpr' | 'hlo'

export default function CorpusXray() {
  const [programId, setProgramId] = useState(PROGRAMS[0]!.id)
  const [active, setActive] = useState<number | null>(null)

  const program = PROGRAMS.find((p) => p.id === programId)!
  const groups = MAPS[programId]?.groups ?? []
  const activeGroup = active !== null ? groups[active] : undefined

  const changeProgram = (id: string) => {
    setProgramId(id)
    setActive(null)
  }

  const groupOf = (col: Col, line: number) => groups.findIndex((g) => g[col].includes(line))

  const mappedJaxprLines = new Set(groups.flatMap((g) => g.jaxpr)).size
  const mappedHloLines = new Set(groups.flatMap((g) => g.hlo)).size

  const renderCol = (col: Col, label: string, lines: string[]) => (
    <div className="col">
      <div className="col-head">{label}</div>
      <pre onMouseLeave={() => setActive(null)}>
        {lines.map((line, n) => {
          const g = groupOf(col, n)
          const mapped = g !== -1
          const hot = mapped && g === active
          return (
            <button
              key={n}
              type="button"
              className={`ln ${hot ? 'hot' : ''} ${mapped ? 'mapped' : ''}`}
              onMouseEnter={() => mapped && setActive(g)}
              onFocus={() => mapped && setActive(g)}
              tabIndex={mapped ? 0 : -1}
            >
              {line || ' '}
            </button>
          )
        })}
      </pre>
    </div>
  )

  return (
    <div className="cxray">
      <div className="picker">
        <label>
          program
          <select value={programId} onChange={(e) => changeProgram(e.target.value)}>
            {PROGRAMS.map((p) => (
              <option key={p.id} value={p.id}>{p.title}</option>
            ))}
          </select>
        </label>
        <span className="stat">
          {groups.length} mapped equations · {mappedJaxprLines} of {program.jaxpr.length} jaxpr lines ·{' '}
          {mappedHloLines} of {program.stablehlo.length} StableHLO lines
        </span>
      </div>

      <div className="cols">
        {renderCol('jaxpr', 'jaxpr', program.jaxpr)}
        {renderCol('hlo', 'StableHLO', program.stablehlo)}
      </div>

      <p className="note" aria-live="polite">
        {activeGroup
          ? `${activeGroup.op} · ${activeGroup.jaxpr.length} jaxpr line(s) ↔ ${activeGroup.hlo.length} StableHLO line(s)`
          : 'mapping computed mechanically; unmapped lines are conservative skips, not absences'}
      </p>

      <style>{`
        .cxray { font-family: 'IBM Plex Mono', monospace; }

        .cxray .picker { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; padding-bottom: 0.75rem; }
        .cxray .picker label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.06em; }
        .cxray .picker select { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.4rem; max-width: 20rem; }
        .cxray .picker .stat { font-size: 0.6875rem; color: ${PANEL_MUTE}; }

        .cxray .cols { display: grid; grid-template-columns: 1fr 1.3fr; gap: 1px; background: ${PANEL_RULE}; }
        .cxray .col { background: #101215; min-width: 0; }
        .cxray .col-head { font-size: 0.625rem; letter-spacing: 0.12em; text-transform: uppercase; color: ${PANEL_MUTE}; padding: 0.5rem 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .cxray pre { margin: 0; padding: 0.5rem 0; overflow-x: auto; max-height: 26rem; overflow-y: auto; }
        .cxray .ln { display: block; width: 100%; text-align: left; background: transparent; border: none; border-left: 3px solid transparent; color: ${PANEL_MUTE}; font: inherit; font-size: 0.6875rem; line-height: 1.6; padding: 0 0.75rem; white-space: pre; cursor: default; }
        .cxray .ln.mapped { color: ${PANEL_INK}; cursor: pointer; }
        .cxray .ln.hot { background: #20150f; border-left-color: ${COPPER}; color: ${COPPER}; }

        .cxray .note { font-size: 0.75rem; color: ${PANEL_INK}; padding: 0.625rem 0.125rem 0; margin: 0; min-height: 2.5em; }

        @media (max-width: 900px) { .cxray .cols { grid-template-columns: 1fr; } .cxray pre { max-height: 18rem; } }
      `}</style>
    </div>
  )
}
