// One stepper, two panes: a compact schedule readout on the left, the
// kernel source on the right. Stepping through the trace's frames moves
// both at once, so the reader sees which lines of code produced the slot
// occupancy and compute state they're looking at. No SVG diagram here on
// purpose: Choreographer already owns that job and owns its own state, so
// this component re-implements only the minimal shared-state slice it
// needs (frame caption, slot table, code highlight) rather than embedding
// or forking Choreographer.
import { useEffect, useState } from 'react'
import type { TraceDoc } from './types'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PANEL_BG = '#101215'
const STEP_BG = '#20150f'

interface Props {
  trace: TraceDoc
  /** Pre-highlighted HTML per source line, from lib/highlight.ts. lines[i] is source line i + 1. */
  lines: string[]
  /** 1-indexed, inclusive source line range each trace frame is about. frameLines[i] pairs with trace.frames[i]. */
  frameLines: [number, number][]
}

export default function SyncedWalk({ trace, lines, frameLines }: Props) {
  const frameCount = trace.frames.length
  const [i, setI] = useState(0)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return
      if (e.key === 'j') setI((x) => Math.min(frameCount - 1, x + 1))
      else if (e.key === 'k') setI((x) => Math.max(0, x - 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [frameCount])

  if (frameCount === 0) return null

  const frame = trace.frames[i]!
  const range = frameLines[i]
  const [from, to] = range ?? [0, -1] // no range for this frame: dim every line, highlight none

  return (
    <div className="syncedwalk panel">
      <div className="sw-head">
        <span className="sw-title">{trace.title}</span>
      </div>

      <div className="sw-body">
        <div className="sw-schedule">
          <p className="sw-caption" aria-live="polite">{frame.caption}</p>

          <dl className="sw-slots">
            {trace.slotIds.map((s) => {
              const occupant = frame.slots[s]
              const isAcc = s === 'acc'
              return (
                <div className="sw-row" key={s}>
                  <dt>{s}</dt>
                  <dd className={occupant ? (isAcc ? 'copper' : 'ink') : 'mute'}>{occupant ?? 'empty'}</dd>
                </div>
              )
            })}
            {trace.stateIds.map((sid) => (
              <div className="sw-row" key={sid}>
                <dt>{sid}</dt>
                <dd className="copper">{frame.state?.[sid] ?? '·'}</dd>
              </div>
            ))}
            <div className="sw-row">
              <dt>transfers</dt>
              <dd className={frame.dma.length ? 'steel' : 'mute'}>
                {frame.dma.length ? frame.dma.map((d) => `${d.tile} → ${d.toSlot}`).join(', ') : 'none'}
              </dd>
            </div>
            <div className="sw-row">
              <dt>compute</dt>
              <dd className={frame.compute ? 'copper' : 'mute'}>
                {frame.compute ? `${frame.compute.unit} · ${frame.compute.expr}` : 'idle'}
              </dd>
            </div>
          </dl>
        </div>

        <div className="sw-code">
          <pre>
            {lines.map((html, idx) => {
              const lineNo = idx + 1
              const inFrame = lineNo >= from && lineNo <= to
              return (
                <div key={lineNo} className={inFrame ? 'sw-line active' : 'sw-line dim'}>
                  <span className="sw-lineno">{lineNo}</span>
                  {/* html is our own build-time shiki output (site/src/lib/highlight.ts),
                      never user-supplied, so injecting it directly is safe. */}
                  <span className="sw-tokens" dangerouslySetInnerHTML={{ __html: html }} />
                </div>
              )
            })}
          </pre>
        </div>
      </div>

      <p className="sw-sync">the schedule step on the left is these lines on the right</p>

      <div className="sw-controls">
        <button onClick={() => setI((x) => Math.max(0, x - 1))} disabled={i === 0} aria-label="Previous frame">
          « prev
        </button>
        <span className="sw-fno">
          {String(i + 1).padStart(2, '0')}/{String(frameCount).padStart(2, '0')}
        </span>
        <button onClick={() => setI((x) => Math.min(frameCount - 1, x + 1))} disabled={i === frameCount - 1} aria-label="Next frame">
          next »
        </button>
      </div>

      <style>{`
        .syncedwalk { font-family: 'IBM Plex Mono', monospace; }

        .syncedwalk .sw-head { padding: 0.625rem 0.875rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .syncedwalk .sw-title { font-size: 0.75rem; letter-spacing: 0.06em; color: ${PANEL_MUTE}; }

        .syncedwalk .sw-body { display: grid; grid-template-columns: minmax(13rem, 1fr) minmax(16rem, 1.4fr); gap: 0; }
        @media (max-width: 34rem) { .syncedwalk .sw-body { grid-template-columns: 1fr; } }

        .syncedwalk .sw-schedule { padding: 0.875rem; border-right: 1px solid ${PANEL_RULE}; min-width: 0; }
        @media (max-width: 34rem) { .syncedwalk .sw-schedule { border-right: none; border-bottom: 1px solid ${PANEL_RULE}; } }

        .syncedwalk .sw-caption { margin: 0 0 0.75rem; font-size: 0.8125rem; line-height: 1.5; color: ${PANEL_INK}; min-height: 2.4em; }

        .syncedwalk .sw-slots { margin: 0; display: flex; flex-direction: column; gap: 0.375rem; }
        .syncedwalk .sw-row { display: flex; justify-content: space-between; gap: 0.75rem; font-size: 0.75rem; }
        .syncedwalk .sw-row dt { color: ${PANEL_MUTE}; letter-spacing: 0.04em; flex: 0 0 auto; }
        .syncedwalk .sw-row dd { margin: 0; text-align: right; word-break: break-word; }
        .syncedwalk .sw-row dd.ink { color: ${PANEL_INK}; }
        .syncedwalk .sw-row dd.copper { color: ${COPPER}; }
        .syncedwalk .sw-row dd.steel { color: ${STEEL}; }
        .syncedwalk .sw-row dd.mute { color: ${PANEL_MUTE}; }

        .syncedwalk .sw-code { overflow-x: auto; overflow-y: auto; max-height: 26rem; background: ${PANEL_BG}; }
        .syncedwalk pre { margin: 0; padding: 0.5rem 0; }
        .syncedwalk .sw-line {
          display: flex;
          align-items: flex-start;
          gap: 0.875rem;
          padding: 0 0.875rem;
          border-left: 2px solid transparent;
          white-space: pre;
          opacity: 1;
        }
        .syncedwalk .sw-line.dim { opacity: 0.45; }
        .syncedwalk .sw-line.active { background: ${STEP_BG}; border-left-color: ${COPPER}; opacity: 1; }
        .syncedwalk .sw-lineno { flex: 0 0 auto; width: 2.25rem; text-align: right; color: ${PANEL_MUTE}; font-size: 0.75rem; user-select: none; }
        .syncedwalk .sw-tokens { font-size: 0.8125rem; line-height: 1.6; }

        .syncedwalk .sw-sync { margin: 0; padding: 0.625rem 0.875rem; font-size: 0.75rem; color: ${PANEL_MUTE}; border-top: 1px solid ${PANEL_RULE}; }

        .syncedwalk .sw-controls { display: flex; align-items: center; gap: 0.75rem; padding: 0 0.875rem 0.875rem; }
        .syncedwalk .sw-fno { color: ${PANEL_MUTE}; font-size: 0.75rem; }
        .syncedwalk button {
          background: transparent;
          color: ${PANEL_INK};
          border: 1px solid ${PANEL_RULE};
          border-radius: 2px;
          font-family: inherit;
          font-size: 0.75rem;
          padding: 0.25rem 0.625rem;
          cursor: pointer;
        }
        .syncedwalk button:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .syncedwalk button:disabled { opacity: 0.35; cursor: default; }
      `}</style>
    </div>
  )
}
