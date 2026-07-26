// The corpus x-ray: IRXray generalized over every program in the reading
// gym's corpus. Pick a program, hover or tab through either column, and
// the equation it belongs to lights up at the other layer too. The line
// mapping itself is not hand-authored: scripts/gen_ir_maps.py walks the
// jaxpr and StableHLO text mechanically and writes data/ir-maps.json,
// skipping anything it can't align with confidence rather than guessing.
import { useEffect, useRef, useState, type ReactNode } from 'react'
import corpus from '../data/ir-corpus.json'
import irMaps from '../data/ir-maps.json'
import opCards from '../data/op-cards.json'

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
  /** Source line(s), recorded by the generator from jax's own source_info. */
  src?: number[]
}
const MAPS = irMaps as Record<string, { groups: MapGroup[] }>

type Col = 'src' | 'jaxpr' | 'hlo'

const CARDS = opCards as Record<string, string>
// StableHLO op spellings back to the card names the reference uses
const HLO_ALIAS: Record<string, string> = {
  multiply: 'mul', subtract: 'sub', divide: 'div', exponential: 'exp',
  maximum: 'max', convert: 'convert_element_type', select: 'select_n',
  negate: 'sub', power: 'mul', reduce: 'reduce',
}

// deterministic syntax tint: same regex on server and client, so hydration
// stays exact; colors follow the vesper family the site's code blocks use
const TOKEN_RE = /(\btensor<[^>]+>|\b(?:bf16|f32|f16|i32|i8|u8|u32|bool)\[[^\]]*\]|\bstablehlo\.\w+|\bfunc\.func\b|\bdef\b|\breturn\b|\blambda\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g
const tint = (line: string): ReactNode[] => {
  const parts = line.split(TOKEN_RE)
  return parts.map((part, i) => {
    if (i % 2 === 0) return part
    let color = '#c9cdd1'
    if (part.startsWith('tensor<') || /^\w+\[/.test(part)) color = '#8fa8bb'
    else if (part.startsWith('stablehlo.') || part === 'func.func') color = '#e8b088'
    else if (part === 'def' || part === 'return' || part === 'lambda') color = '#e8b088'
    else color = '#a8c4b0'
    return <span key={i} style={{ color }}>{part}</span>
  })
}

const cardFor = (op: string): string | undefined => {
  const key = CARDS[op] ? op : HLO_ALIAS[op]
  const text = key ? CARDS[key] : undefined
  return text ? text.split('. ')[0] + '.' : undefined
}

// one honest sentence for any line, mapped or not
const explainLine = (col: Col, line: string): string => {
  const t = line.trim()
  if (!t) return ''
  if (col === 'src') return 'your code: hover it to light the IR it traced to; the mapping rides on jax\u2019s own source records'
  if (t.startsWith('module')) return 'module wrapper: attributes for the whole compiled program, owned by no source line'
  if (t.startsWith('func.func')) return 'the entry function: your inputs, arrived as tensors with explicit shapes and dtypes'
  if (t.startsWith('return') || t === '}' || t === '{') return 'scaffolding: closes the function and hands back the outputs'
  if (t.startsWith('{ lambda')) return 'the jaxpr header: the traced inputs, each with its shape and dtype'
  if (t.startsWith('in (')) return 'the jaxpr outputs'
  if (t.includes('stablehlo.constant')) return 'a constant the compiler materialized: a Python literal, a coefficient from a rewrite (GELU\u2019s tanh form makes several), or a reduction\u2019s init value'
  const hloOp = t.match(/stablehlo\.(\w+)/)?.[1]
  const jaxprOp = t.match(/=\s*(\w+)/)?.[1]
  const op = hloOp ?? jaxprOp
  if (op) {
    const card = cardFor(op)
    if (card) return `${op}: ${card}`
    return `${op}: no reference card yet; unmapped lines here are scaffolding or lowerings the mapper skips rather than guesses`
  }
  return 'a continuation line: parameters of the equation above it'
}

export default function CorpusXray() {
  const [programId, setProgramId] = useState(PROGRAMS[0]!.id)
  const [active, setActive] = useState<number | null>(null)
  const [lineInfo, setLineInfo] = useState<string | null>(null)
  const hoveredCol = useRef<Col | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  // bring the counterpart lines into view when a group lights up, without
  // yanking the column the reader's pointer is in
  useEffect(() => {
    if (active === null || !rootRef.current) return
    for (const pre of rootRef.current.querySelectorAll('pre')) {
      const col = pre.getAttribute('data-col') as Col | null
      if (col && col === hoveredCol.current) continue
      const hot = pre.querySelector('.hot')
      if (hot) hot.scrollIntoView({ block: 'nearest' })
    }
  }, [active])

  const program = PROGRAMS.find((p) => p.id === programId)!
  const groups = MAPS[programId]?.groups ?? []
  const activeGroup = active !== null ? groups[active] : undefined

  const changeProgram = (id: string) => {
    setProgramId(id)
    setActive(null)
    setLineInfo(null)
  }

  const groupOf = (col: Col, line: number) => groups.findIndex((g) => (g[col] ?? []).includes(line))

  const mappedJaxprLines = new Set(groups.flatMap((g) => g.jaxpr)).size
  const mappedHloLines = new Set(groups.flatMap((g) => g.hlo)).size
  const hasSource = !program.source[0]?.startsWith('(wrapped') && groups.some((g) => g.src)

  const renderCol = (col: Col, label: string, lines: string[]) => (
    <div className="col">
      <div className="col-head">{label}</div>
      <pre data-col={col} onMouseLeave={() => { setActive(null); setLineInfo(null); hoveredCol.current = null }}>
        {lines.map((line, n) => {
          const g = groupOf(col, n)
          const mapped = g !== -1
          const hot = mapped && g === active
          return (
            <button
              key={n}
              type="button"
              className={`ln ${hot ? 'hot' : ''} ${mapped ? 'mapped' : ''}`}
              onMouseEnter={() => { hoveredCol.current = col; if (mapped) setActive(g); else setActive(null); setLineInfo(explainLine(col, line)) }}
              onFocus={() => { if (mapped) setActive(g); setLineInfo(explainLine(col, line)) }}
              tabIndex={mapped ? 0 : -1}
            >
              {line ? tint(line) : ' '}
            </button>
          )
        })}
      </pre>
    </div>
  )

  return (
    <div className="cxray" ref={rootRef}>
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
      <p className="prognote">{program.note}</p>

      <div className={`cols ${hasSource ? 'three' : ''}`}>
        {hasSource && renderCol('src', 'source', program.source)}
        {renderCol('jaxpr', 'jaxpr', program.jaxpr)}
        {renderCol('hlo', 'StableHLO', program.stablehlo)}
      </div>

      <p className="legend">
        <span className="lg-bright">bright</span> lines are mapped equations · <span className="lg-dim">dim</span> lines
        are scaffolding, compiler-made constants and broadcasts, or alignments the mapper skips rather than guesses
      </p>
      <p className="note" aria-live="polite">
        {activeGroup
          ? `${activeGroup.op}${activeGroup.src ? ` · from source line ${activeGroup.src[0]! + 1}` : ''} · ${activeGroup.jaxpr.length} jaxpr line(s) ↔ ${activeGroup.hlo.length} StableHLO line(s)${cardFor(activeGroup.op) ? ` · ${cardFor(activeGroup.op)}` : ''}`
          : lineInfo ?? 'hover any line, bright or dim: it explains itself here · source lines recorded by jax itself, IR mapping computed mechanically'}
      </p>

      <style>{`
        .cxray { font-family: 'IBM Plex Mono', monospace; }

        .cxray .picker { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; padding-bottom: 0.75rem; }
        .cxray .picker label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.06em; }
        .cxray .picker select { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.4rem; max-width: 20rem; }
        .cxray .picker .stat { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .cxray .prognote { margin: 0 0 0.75rem; font-size: 0.75rem; color: ${PANEL_MUTE}; }

        .cxray .cols { display: grid; grid-template-columns: 1fr 1.3fr; gap: 1px; background: ${PANEL_RULE}; }
        .cxray .cols.three { grid-template-columns: minmax(10rem, 0.55fr) 1fr 1.3fr; }
        .cxray .col { background: #101215; min-width: 0; }
        .cxray .col-head { font-size: 0.625rem; letter-spacing: 0.12em; text-transform: uppercase; color: ${PANEL_MUTE}; padding: 0.5rem 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .cxray pre { margin: 0; padding: 0.5rem 0; overflow-x: auto; max-height: 30rem; overflow-y: auto; scroll-behavior: auto; }
        .cxray .ln { display: block; width: 100%; text-align: left; background: transparent; border: none; border-left: 3px solid transparent; color: ${PANEL_MUTE}; font: inherit; font-size: 0.8125rem; line-height: 1.7; padding: 0 0.75rem; white-space: pre; cursor: default; opacity: 0.5; }
        .cxray .ln.mapped { color: ${PANEL_INK}; cursor: pointer; opacity: 1; }
        .cxray .ln.hot { background: #20150f; border-left-color: ${COPPER}; }
        .cxray .ln.hot, .cxray .ln.hot span { color: ${COPPER} !important; }

        .cxray .legend { font-size: 0.6875rem; color: ${PANEL_MUTE}; padding: 0.625rem 0.125rem 0; margin: 0; }
        .cxray .legend .lg-bright { color: ${PANEL_INK}; }
        .cxray .legend .lg-dim { opacity: 0.7; }
        .cxray .note { font-size: 0.75rem; color: ${PANEL_INK}; padding: 0.375rem 0.125rem 0; margin: 0; min-height: 2.5em; }

        @media (max-width: 900px) { .cxray .cols, .cxray .cols.three { grid-template-columns: 1fr; } .cxray pre { max-height: 18rem; } }
      `}</style>
    </div>
  )
}
