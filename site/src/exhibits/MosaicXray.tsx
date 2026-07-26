// The mosaic x-ray: the corpus x-ray's grammar, one layer down. Pick a
// kernel, hover any column, and see your Pallas line, the kernel jaxpr it
// traced to, and the Mosaic (tpu dialect) op it lowered to light up as one.
// Modules come from Pallas's own debug output (scripts/gen_mosaic_corpus.py);
// the pairing is computed there by order-preserving op matching, and lines
// the pairer cannot place stay dim rather than guessed.
import { useEffect, useRef, useState, type ReactNode } from 'react'
import corpus from '../data/mosaic-corpus.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#9aa1a8'
const PANEL_RULE = '#23272c'

interface Group {
  op: string
  jaxpr: number[]
  mosaic: number[]
  src?: number[]
}
interface Kernel {
  id: string
  title: string
  note: string
  source: string[]
  jaxpr: string[]
  mosaic: string[]
  groups: Group[]
}
const KERNELS = (corpus as { kernels: Kernel[] }).kernels
const META = (corpus as { meta: { jax: string } }).meta

type Col = 'src' | 'jaxpr' | 'mosaic'

// deterministic syntax tint, same regex server and client
const TOKEN_RE = /(\bmemref<[^>]+>|\bvector<[^>]+>|#tpu\.\w+(?:<\w+>)?|\b(?:bf16|f32|f16|i32|i64|i8|i1|u8|u32|bool|index)\[[^\]]*\]|\b(?:tpu|vector|arith|math|scf|func)\.\w+|\bdef\b|\breturn\b|\blambda\b|\bmodule\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g
const tint = (line: string): ReactNode[] => {
  const parts = line.split(TOKEN_RE)
  return parts.map((part, i) => {
    if (i % 2 === 0) return part
    let color = '#c9cdd1'
    if (part.startsWith('memref<') || part.startsWith('vector<') || /^\w+\[/.test(part)) color = '#8fa8bb'
    else if (part.startsWith('#tpu.')) color = '#b48ead'
    else if (/^(tpu|vector|arith|math|scf|func)\./.test(part) || ['def', 'return', 'lambda', 'module'].includes(part)) color = '#e8b088'
    else color = '#a8c4b0'
    return <span key={i} style={{ color }}>{part}</span>
  })
}

// one honest sentence for any Mosaic line; the vocabulary here is the same
// vocabulary the museum's lattice and VMEM errors speak
const MOSAIC_EXPLAIN: Array<[RegExp, string]> = [
  [/^module @/, 'the module wrapper: one pallas_call becomes one MLIR module, named after your kernel function'],
  [/dimension_semantics|iteration_bounds|window_params/, 'the schedule, compiled: grid bounds, per-axis semantics, and one window per BlockSpec whose transform function IS your index map'],
  [/^func\.func @main/, 'the kernel entry: grid indices arrive as scalars, every ref as a memref already resident in VMEM; Mosaic never touches HBM'],
  [/^func\.func @transform/, 'a BlockSpec index map, compiled to a function: grid indices in, block coordinates out'],
  [/arith\.constant dense/, 'a vector constant materialized whole: a reduction identity or an init value'],
  [/arith\.constant/, 'a scalar constant: an index, a zero, a comparison operand'],
  [/vector\.load/, 'a whole block read from VMEM into a vector value: a ref[...] read on your side'],
  [/tpu\.vector_store|vector\.store/, 'the block written back to VMEM: your ref[...] = assignment'],
  [/tpu\.matmul/, 'the MXU op itself: your dot, with dimension_numbers carried down intact from dot_general'],
  [/tpu\.iota|tpu\.\w*rotate/, 'a tpu dialect data-movement op: lane and sublane choreography on the (8, 128) grid'],
  [/vector\.multi_reduction <max/, 'a reduction across vector dimensions: your jnp.max over the row axis, lane-wise on the VPU'],
  [/vector\.multi_reduction <add/, 'a reduction across vector dimensions: your jnp.sum over the row axis, lane-wise on the VPU'],
  [/vector\.broadcast|vector\.shape_cast/, 'shape plumbing: a broadcast or rank change so operand shapes agree, usually from keepdims'],
  [/math\.exp/, 'a VPU transcendental: exp evaluated lane-wise across the (8, 128) grid'],
  [/math\.\w+/, 'a VPU transcendental, evaluated lane-wise'],
  [/arith\.truncf/, 'a downcast: your astype back to bf16, now a hardware truncate'],
  [/arith\.extf/, 'an upcast: your astype to f32, widened lane-wise before the math'],
  [/arith\.extui|arith\.sitofp|arith\.fptosi/, 'an integer or float conversion the lowering inserted'],
  [/arith\.cmpi|arith\.cmpf/, 'a comparison: the pl.when predicate against the grid index'],
  [/scf\.if/, 'structured control flow: pl.when compiled to a real branch; both branches exist in the module'],
  [/scf\.for/, 'structured control flow: your fori_loop, still a real loop at this layer'],
  [/arith\.(addf|addi|subf|subi|mulf|muli|divf|maximumf|minimumf)/, 'VPU arithmetic: elementwise across the vector, one op per jaxpr equation'],
  [/^return|^\}|^\{|^\s*\} else \{/, 'scaffolding: closes a region or hands back results'],
]

const explainLine = (col: Col, line: string): string => {
  const t = line.trim()
  if (!t) return ''
  if (col === 'src') return 'your Pallas kernel: hover it to light what it became one and two layers down'
  if (col === 'jaxpr') {
    if (t.startsWith('{ lambda')) return 'the kernel jaxpr header: refs arrive as MemRefs, not arrays; the kernel reads and writes them in place'
    if (t.startsWith('in (')) return 'the kernel jaxpr outputs: none, because a kernel writes refs instead of returning values'
    const op = t.match(/=\s*(\w+)/)?.[1] ?? (t.includes(' <- ') ? 'a ref move' : null)
    if (op) return `${op}: hover a mapped line to see its Mosaic counterpart`
    return 'a continuation line: parameters of the equation above it'
  }
  for (const [re, text] of MOSAIC_EXPLAIN) {
    if (re.test(t)) return text
  }
  return 'a tpu dialect line the pairer left unmapped: real, but not attributable to one equation without guessing'
}

export default function MosaicXray() {
  const [kernelId, setKernelId] = useState(KERNELS[0]!.id)
  const [active, setActive] = useState<number | null>(null)
  const [lineInfo, setLineInfo] = useState<string | null>(null)
  const hoveredCol = useRef<Col | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (active === null || !rootRef.current) return
    for (const pre of rootRef.current.querySelectorAll('pre')) {
      const col = pre.getAttribute('data-col') as Col | null
      if (col && col === hoveredCol.current) continue
      const hot = pre.querySelector('.hot')
      if (hot) hot.scrollIntoView({ block: 'nearest' })
    }
  }, [active])

  const kernel = KERNELS.find((k) => k.id === kernelId)!
  const groups = kernel.groups
  const activeGroup = active !== null ? groups[active] : undefined

  const changeKernel = (id: string) => {
    setKernelId(id)
    setActive(null)
    setLineInfo(null)
  }

  const groupOf = (col: Col, line: number) => groups.findIndex((g) => (g[col] ?? []).includes(line))

  const mappedJaxpr = new Set(groups.flatMap((g) => g.jaxpr)).size
  const mappedMosaic = new Set(groups.flatMap((g) => g.mosaic)).size

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
    <div className="mxray" ref={rootRef}>
      <div className="picker">
        <label>
          kernel
          <select value={kernelId} onChange={(e) => changeKernel(e.target.value)}>
            {KERNELS.map((k) => (
              <option key={k.id} value={k.id}>{k.title}</option>
            ))}
          </select>
        </label>
        <span className="stat">
          {groups.length} paired groups · {mappedJaxpr} of {kernel.jaxpr.length} jaxpr lines ·{' '}
          {mappedMosaic} of {kernel.mosaic.length} Mosaic lines
        </span>
      </div>
      <p className="kernnote">{kernel.note}</p>

      <div className="cols">
        {renderCol('src', 'Pallas kernel', kernel.source)}
        {renderCol('jaxpr', 'kernel jaxpr', kernel.jaxpr)}
        {renderCol('mosaic', 'Mosaic module (tpu dialect)', kernel.mosaic)}
      </div>

      <p className="legend">
        <span className="lg-bright">bright</span> lines are paired · <span className="lg-dim">dim</span> lines still
        explain themselves on hover: constants, index plumbing, and pairings the script skips rather than guesses
      </p>
      <p className="note" aria-live="polite">
        {activeGroup
          ? `${activeGroup.op}${activeGroup.src ? ` · from kernel line ${activeGroup.src[0]! + 1}` : ''} · ${activeGroup.jaxpr.length} jaxpr line(s) ↔ ${activeGroup.mosaic.length} Mosaic line(s)`
          : lineInfo ?? `hover any line, bright or dim: it explains itself here · captured from Pallas debug output on jax ${META.jax}, paired mechanically`}
      </p>

      <style>{`
        .mxray { font-family: 'IBM Plex Mono', monospace; }

        .mxray .picker { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; padding-bottom: 0.75rem; }
        .mxray .picker label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.06em; }
        .mxray .picker select { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.4rem; max-width: 20rem; }
        .mxray .picker .stat { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .mxray .kernnote { margin: 0 0 0.75rem; font-size: 0.75rem; color: ${PANEL_MUTE}; }

        .mxray .cols { display: grid; grid-template-columns: minmax(12rem, 0.7fr) 1fr 1.4fr; gap: 1px; background: ${PANEL_RULE}; }
        .mxray .col { background: #101215; min-width: 0; }
        .mxray .col-head { font-size: 0.625rem; letter-spacing: 0.12em; text-transform: uppercase; color: ${PANEL_MUTE}; padding: 0.5rem 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .mxray pre { margin: 0; padding: 0.5rem 0; overflow-x: auto; max-height: 30rem; overflow-y: auto; scroll-behavior: auto; }
        .mxray .ln { display: block; width: 100%; text-align: left; background: transparent; border: none; border-left: 3px solid transparent; color: ${PANEL_MUTE}; font: inherit; font-size: 0.8125rem; line-height: 1.7; padding: 0 0.75rem; white-space: pre; cursor: default; opacity: 0.5; }
        .mxray .ln.mapped { color: ${PANEL_INK}; cursor: pointer; opacity: 1; }
        .mxray .ln.hot { background: #20150f; border-left-color: ${COPPER}; }
        .mxray .ln.hot, .mxray .ln.hot span { color: ${COPPER} !important; }

        .mxray .legend { font-size: 0.6875rem; color: ${PANEL_MUTE}; padding: 0.625rem 0.125rem 0; margin: 0; }
        .mxray .legend .lg-bright { color: ${PANEL_INK}; }
        .mxray .legend .lg-dim { opacity: 0.7; }
        .mxray .note { font-size: 0.75rem; color: ${PANEL_INK}; padding: 0.375rem 0.125rem 0; margin: 0; min-height: 2.5em; }

        @media (max-width: 900px) { .mxray .cols { grid-template-columns: 1fr; } .mxray pre { max-height: 18rem; } }
      `}</style>
    </div>
  )
}
