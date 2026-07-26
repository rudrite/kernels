// The fusion x-ray: the same program at two moments, side by side. Left,
// the StableHLO XLA received; right, the optimized HLO it decided on, with
// its fusion ops lit and the spill-carrying lines flagged. Dumps captured
// on a real TPU by labs/capture-hlo.ipynb; nothing retyped, no line pairing
// invented: fusion rewrites wholesale, so each side explains itself.
import { useState, type ReactNode } from 'react'
import pairs from '../data/hlo-pairs.json'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#9aa1a8'
const PANEL_RULE = '#23272c'
const FAIL = '#c4574a'

interface Program {
  id: string
  title: string
  note: string
  unopt: string[]
  opt: string[]
}
const PROGRAMS = (pairs as { programs: Program[] }).programs
const META = (pairs as { meta: { chip: string; jax: string } }).meta

// the big intermediate each program's spill rides in, for flagging
const SPILL_TOKEN: Record<string, string> = {
  attention: 'bf16[1024,1024]',
  mlp: 'bf16[256,2048]',
  softmax: 'bf16[1024,512]',
}

// deterministic syntax tint, same regex server and client; vesper family
const TOKEN_RE = /(\btensor<[^>]+>|\b(?:bf16|f32|f16|s32|i32|u32|u8|pred)\[[^\]]*\]|\bstablehlo\.\w+|\bfunc\.func\b|kind=k\w+|%[\w.]+|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g
const tint = (line: string): ReactNode[] => {
  const parts = line.split(TOKEN_RE)
  return parts.map((part, i) => {
    if (i % 2 === 0) return part
    let color = '#c9cdd1'
    if (part.startsWith('tensor<') || /^\w+\[/.test(part)) color = '#8fa8bb'
    else if (part.startsWith('stablehlo.') || part === 'func.func' || part.startsWith('kind=')) color = '#e8b088'
    else if (part.startsWith('%')) color = '#b7bfc7'
    else color = '#a8c4b0'
    return <span key={i} style={{ color }}>{part}</span>
  })
}

const classify = (line: string): { cls: string; note: string } => {
  const t = line.trim()
  if (!t) return { cls: 'dim', note: '' }
  if (/^HloModule/.test(t)) return { cls: 'dim', note: 'the module header: entry layout and config, owned by no op' }
  if (/= fusion\(| fusion\(/.test(t)) {
    const kind = t.match(/kind=(\w+)/)?.[1]
    return {
      cls: 'fus',
      note: `a fusion op${kind ? ` (${kind})` : ''}: several of your ops compiled into one kernel so intermediates stay in fast memory`,
    }
  }
  if (/%fused_computation|^%region|^\}/.test(t)) return { cls: 'dim', note: 'a fused computation body or region: the ops living inside a fusion, listed once each' }
  if (/custom-call/.test(t)) return { cls: 'hot', note: 'a custom call: work dispatched outside HLO; on TPU this is how pallas_call and library ops ride through' }
  if (/= copy\(|copy-start|copy-done/.test(t)) return { cls: 'hot', note: 'a copy the compiler inserted, usually a layout change: this line is bandwidth spent on data arrangement' }
  if (/convolution|= dot\(/.test(t)) return { cls: 'hot', note: 'the matmul itself (TPU spells small dots as convolutions); fusions feed it, it does not fuse into them' }
  if (/parameter\(\d+\)/.test(t)) return { cls: 'dim', note: 'an input arriving, with its layout {minor-to-major:tile} stated inline' }
  if (/^ROOT/.test(t)) return { cls: 'hot', note: 'the computation\'s result: read its shape and layout to see what this whole block produces' }
  if (/file_location|parent_frame|^\d+ \{/.test(t)) return { cls: 'dim', note: 'debug frame bookkeeping the dump carries at the end; no machine code corresponds to it' }
  if (/stablehlo\./.test(t)) return { cls: '', note: 'one op as XLA received it: still one-to-one with your traced program' }
  if (/reduce|exponential|divide|subtract|maximum|broadcast|convert/.test(t)) return { cls: '', note: 'an elementwise or reduction op: a candidate for fusion on the other side' }
  return { cls: 'dim', note: 'scaffolding or bookkeeping' }
}

export default function FusionXray() {
  const [programId, setProgramId] = useState(PROGRAMS[0]!.id)
  const [note, setNote] = useState<string | null>(null)

  const program = PROGRAMS.find((p) => p.id === programId)!
  const spillToken = SPILL_TOKEN[programId]
  const fusions = program.opt.filter((l) => / fusion\(|= fusion\(/.test(l)).length
  const spills = spillToken ? program.opt.filter((l) => l.includes(spillToken) && /fusion|convolution|ROOT/.test(l)).length : 0

  const renderCol = (label: string, lines: string[], isOpt: boolean) => (
    <div className="col">
      <div className="col-head">{label}</div>
      <pre onMouseLeave={() => setNote(null)}>
        {lines.map((line, n) => {
          const { cls, note: lineNote } = isOpt ? classify(line) : { cls: '', note: classify(line).note }
          const spill = isOpt && spillToken && line.includes(spillToken)
          return (
            <span
              key={n}
              className={`ln ${cls} ${spill ? 'spill' : ''}`}
              onMouseEnter={() => setNote(spill ? `${lineNote} · carries ${spillToken}: the spill, visible in the plan` : lineNote)}
            >
              {line ? tint(line) : ' '}
              {'\n'}
            </span>
          )
        })}
      </pre>
    </div>
  )

  return (
    <div className="fxray">
      <div className="picker">
        <label>
          program
          <select value={programId} onChange={(e) => { setProgramId(e.target.value); setNote(null) }}>
            {PROGRAMS.map((p) => (
              <option key={p.id} value={p.id}>{p.title}</option>
            ))}
          </select>
        </label>
        <span className="stat">
          {program.unopt.length} lines in → {program.opt.length} lines decided · {fusions} fusion ops
          {spills > 0 && ` · ${spills} carry ${spillToken}`}
        </span>
      </div>
      <p className="prognote">{program.note}</p>

      <div className="cols">
        {renderCol('what XLA received (StableHLO)', program.unopt, false)}
        {renderCol('what XLA decided (optimized HLO)', program.opt, true)}
      </div>

      <p className="legend">
        <span className="lg-fus">copper</span> lines are fusion ops · <span className="lg-spill">underlined</span>{' '}
        lines carry the program’s big intermediate · dim lines are regions, parameters, and bookkeeping
      </p>
      <p className="note" aria-live="polite">
        {note || `hover any line on either side · captured on ${META.chip}, jax ${META.jax}, by labs/capture-hlo.ipynb`}
      </p>

      <style>{`
        .fxray { font-family: 'IBM Plex Mono', monospace; }
        .fxray .picker { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; padding-bottom: 0.75rem; }
        .fxray .picker label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.06em; }
        .fxray .picker select { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.4rem; max-width: 20rem; }
        .fxray .picker .stat { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .fxray .prognote { margin: 0 0 0.75rem; font-size: 0.75rem; color: ${PANEL_MUTE}; }

        .fxray .cols { display: grid; grid-template-columns: 1fr 1.35fr; gap: 1px; background: ${PANEL_RULE}; }
        .fxray .col { background: #101215; min-width: 0; }
        .fxray .col-head { font-size: 0.625rem; letter-spacing: 0.12em; text-transform: uppercase; color: ${PANEL_MUTE}; padding: 0.5rem 0.75rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .fxray pre { margin: 0; padding: 0.5rem 0.75rem; overflow: auto; max-height: 28rem; font-size: 0.75rem; line-height: 1.7; }
        .fxray .ln { display: block; white-space: pre; color: ${PANEL_MUTE}; opacity: 0.85; cursor: default; }
        .fxray .ln.dim { opacity: 0.45; }
        .fxray .ln.dim span { color: inherit !important; }
        .fxray .ln.hot { color: ${PANEL_INK}; opacity: 1; }
        .fxray .ln.fus, .fxray .ln.fus span { color: ${COPPER} !important; opacity: 1; }
        .fxray .ln.spill { text-decoration: underline; text-decoration-color: ${FAIL}; text-underline-offset: 3px; }

        .fxray .legend { font-size: 0.6875rem; color: ${PANEL_MUTE}; padding: 0.625rem 0.125rem 0; margin: 0; }
        .fxray .legend .lg-fus { color: ${COPPER}; }
        .fxray .legend .lg-spill { text-decoration: underline; text-decoration-color: ${FAIL}; text-underline-offset: 3px; }
        .fxray .note { font-size: 0.75rem; color: ${PANEL_INK}; padding: 0.375rem 0.125rem 0; margin: 0; min-height: 2.5em; }

        @media (max-width: 900px) { .fxray .cols { grid-template-columns: 1fr; } .fxray pre { max-height: 16rem; } }
      `}</style>
    </div>
  )
}
