// The timeline x-ray: a real XProf device timeline, rendered readable. One
// bar per op, scaled to hardware time per iteration; hover explains what
// each op is, click opens its full HLO text. The data is the captured
// aggregation from labs/profile-bytes.ipynb, verbatim: the same run that
// found XLA's own online-softmax kernel and closed gate 02 at 1.2%.
import { useState } from 'react'
import timeline from '../data/timeline.json'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#9aa1a8'
const PANEL_RULE = '#23272c'

interface Row {
  id: string
  op: string
  kind: 'jit' | 'custom' | 'fusion' | 'copy' | 'counter'
  count: number
  total_us: number
  us_per_iter: number
  full: string
  note: string
}
const ROWS = (timeline as { rows: Row[] }).rows
const META = (timeline as { meta: { chip: string; jax: string; iters: number; program: string } }).meta
const MAX_US = Math.max(...ROWS.map((r) => r.us_per_iter))

const KIND_COLOR: Record<Row['kind'], string> = {
  jit: PANEL_INK,
  custom: COPPER,
  fusion: COPPER,
  copy: STEEL,
  counter: '#4a525a',
}
const KIND_LABEL: Record<Row['kind'], string> = {
  jit: 'the envelope',
  custom: 'custom call',
  fusion: 'fusion',
  copy: 'async copy',
  counter: 'counters',
}

export default function TimelineXray() {
  const [note, setNote] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)

  return (
    <div className="txray">
      <div className="head">
        <span className="stat">
          {META.program} · {META.iters} iterations · per-op hardware time from the device plane
        </span>
      </div>

      <div className="rows" onMouseLeave={() => setNote(null)}>
        {ROWS.map((r) => {
          const pct = MAX_US > 0 ? (r.us_per_iter / MAX_US) * 100 : 0
          const color = KIND_COLOR[r.kind]
          return (
            <div key={r.id} className="rowwrap">
              <button
                type="button"
                className={`row ${r.kind}`}
                onMouseEnter={() => setNote(r.note)}
                onFocus={() => setNote(r.note)}
                onClick={() => setOpen(open === r.id ? null : r.id)}
              >
                <span className="opname">{r.op}</span>
                <span className="barzone">
                  <span className="bar" style={{ width: `${Math.max(pct, 0.5)}%`, background: color, opacity: r.kind === 'jit' ? 0.25 : 0.85 }} />
                  <span className="us">{r.us_per_iter.toFixed(1)} µs</span>
                </span>
                <span className="kindtag" style={{ color }}>{KIND_LABEL[r.kind]}</span>
                <span className="cnt">×{r.count}</span>
              </button>
              {open === r.id && <pre className="full">{r.full}</pre>}
            </div>
          )
        })}
      </div>

      <p className="reading">
        The reading: the envelope is 217.6 µs and so is fusion + online-softmax alone, which means
        the 89.6 µs copy hid completely behind compute. And the second-largest op is a kernel nobody
        in this program wrote: the compiler dispatched its own.
      </p>
      <p className="note" aria-live="polite">
        {note ?? `hover an op: it explains itself · click for its full HLO text · captured on ${META.chip}, jax ${META.jax}`}
      </p>

      <style>{`
        .txray { font-family: 'IBM Plex Mono', monospace; }
        .txray .head { padding-bottom: 0.75rem; }
        .txray .stat { font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.03em; }

        .txray .rows { border: 1px solid ${PANEL_RULE}; }
        .txray .rowwrap { border-bottom: 1px solid ${PANEL_RULE}; }
        .txray .rowwrap:last-child { border-bottom: none; }
        .txray .row {
          display: grid; grid-template-columns: minmax(11rem, 0.6fr) 1fr 6.5rem 3rem;
          gap: 0.75rem; align-items: center; width: 100%;
          background: transparent; border: none; padding: 0.5rem 0.75rem;
          font: inherit; cursor: pointer; text-align: left;
        }
        .txray .row:hover { background: #16181c; }
        .txray .opname { color: ${PANEL_INK}; font-size: 0.75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .txray .row.counter .opname { color: ${PANEL_MUTE}; opacity: 0.7; }
        .txray .barzone { display: flex; align-items: center; gap: 0.5rem; min-width: 0; }
        .txray .bar { display: block; height: 0.625rem; border-radius: 1px; }
        .txray .us { font-size: 0.6875rem; color: ${PANEL_MUTE}; white-space: nowrap; }
        .txray .kindtag { font-size: 0.625rem; letter-spacing: 0.08em; text-transform: uppercase; }
        .txray .cnt { font-size: 0.6875rem; color: ${PANEL_MUTE}; text-align: right; }
        .txray .full { margin: 0; padding: 0.625rem 0.75rem; background: #0c0e10; color: ${PANEL_MUTE}; font-size: 0.6875rem; line-height: 1.6; white-space: pre-wrap; word-break: break-all; border-top: 1px dashed ${PANEL_RULE}; }

        .txray .reading { font-size: 0.75rem; color: ${PANEL_INK}; padding: 0.75rem 0.125rem 0; margin: 0; max-width: 70ch; line-height: 1.6; }
        .txray .note { font-size: 0.75rem; color: ${PANEL_MUTE}; padding: 0.375rem 0.125rem 0; margin: 0; min-height: 2.5em; }

        @media (max-width: 760px) {
          .txray .row { grid-template-columns: 1fr; gap: 0.25rem; }
          .txray .kindtag, .txray .cnt { display: none; }
        }
      `}</style>
    </div>
  )
}
