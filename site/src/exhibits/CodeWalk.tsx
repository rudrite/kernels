// Line-by-line code walkthrough: a fixed code panel on top, one step's note
// underneath. Each step names the line range it's about; CodeWalk lights
// those lines and dims everything else so the reader's eye lands where the
// explanation is pointing. No autoplay: the reader drives every step.
import { useEffect, useRef, useState } from 'react'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const STEP_BG = '#20150f'

interface Step {
  /** 1-indexed, inclusive source line range this step is about. */
  lines: [number, number]
  note: string
}

interface Props {
  /** Pre-highlighted HTML per source line, from lib/highlight.ts. lines[i] is source line i + 1. */
  lines: string[]
  steps: Step[]
  title: string
}

export default function CodeWalk({ lines, steps, title }: Props) {
  const [i, setI] = useState(0)
  const codeRef = useRef<HTMLDivElement>(null)

  // Long walks scroll inside .cw-code; each step centers its own lines.
  // Positions come from getBoundingClientRect measured against the container
  // (offsetTop would measure against a distant positioned ancestor and pin
  // the scroll to the bottom), and only the container scrolls, never the page.
  useEffect(() => {
    const container = codeRef.current
    if (!container) return
    const actives = container.querySelectorAll<HTMLElement>('.cw-line.active')
    if (actives.length === 0) return
    const cRect = container.getBoundingClientRect()
    const first = actives[0]!.getBoundingClientRect()
    const last = actives[actives.length - 1]!.getBoundingClientRect()
    const mid = (first.top + last.bottom) / 2 - cRect.top + container.scrollTop
    container.scrollTo({ top: Math.max(0, mid - container.clientHeight / 2), behavior: 'smooth' })
  }, [i])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return
      if (e.key === 'j') setI((x) => Math.min(steps.length - 1, x + 1))
      else if (e.key === 'k') setI((x) => Math.max(0, x - 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [steps.length])

  if (steps.length === 0) return null

  const step = steps[i]!
  const [from, to] = step.lines

  return (
    <div className="codewalk panel">
      <div className="cw-head">
        <span className="cw-title">{title}</span>
      </div>

      <div className="cw-code" ref={codeRef}>
        <pre>
          {lines.map((html, idx) => {
            const lineNo = idx + 1
            const inStep = lineNo >= from && lineNo <= to
            return (
              <div key={lineNo} className={inStep ? 'cw-line active' : 'cw-line dim'}>
                <span className="cw-lineno">{lineNo}</span>
                {/* html is our own build-time shiki output (site/src/lib/highlight.ts),
                    never user-supplied, so injecting it directly is safe. */}
                <span className="cw-tokens" dangerouslySetInnerHTML={{ __html: html }} />
              </div>
            )
          })}
        </pre>
      </div>

      <p className="cw-caption" aria-live="polite">
        <span className="cw-fno">
          {String(i + 1).padStart(2, '0')}/{String(steps.length).padStart(2, '0')}
        </span>
        {step.note}
      </p>

      <div className="cw-controls">
        <button onClick={() => setI((x) => Math.max(0, x - 1))} disabled={i === 0} aria-label="Previous step">
          « prev
        </button>
        <button onClick={() => setI((x) => Math.min(steps.length - 1, x + 1))} disabled={i === steps.length - 1} aria-label="Next step">
          next »
        </button>
      </div>

      <style>{`
        .codewalk { font-family: 'IBM Plex Mono', monospace; }
        .codewalk .cw-head { padding: 0.625rem 0.875rem; border-bottom: 1px solid ${PANEL_RULE}; }
        .codewalk .cw-title { font-size: 0.75rem; letter-spacing: 0.06em; color: ${PANEL_MUTE}; }
        .codewalk .cw-code { overflow-x: auto; overflow-y: auto; max-height: 26rem; }
        .codewalk pre { margin: 0; padding: 0.5rem 0; }
        .codewalk .cw-line {
          display: flex;
          align-items: flex-start;
          gap: 0.875rem;
          padding: 0 0.875rem;
          border-left: 2px solid transparent;
          white-space: pre;
          opacity: 1;
        }
        .codewalk .cw-line.dim { opacity: 0.45; }
        .codewalk .cw-line.active { background: ${STEP_BG}; border-left-color: ${COPPER}; opacity: 1; }
        .codewalk .cw-lineno { flex: 0 0 auto; width: 2.25rem; text-align: right; color: ${PANEL_MUTE}; font-size: 0.75rem; user-select: none; }
        .codewalk .cw-tokens { font-size: 0.8125rem; line-height: 1.6; }
        .codewalk .cw-caption {
          margin: 0;
          padding: 0.75rem 0.875rem;
          min-height: 3.2em;
          font-size: 0.8125rem;
          line-height: 1.5;
          color: ${PANEL_INK};
          border-top: 1px solid ${PANEL_RULE};
        }
        .codewalk .cw-fno { color: ${PANEL_MUTE}; margin-right: 0.75em; }
        .codewalk .cw-controls { display: flex; gap: 0.5rem; padding: 0 0.875rem 0.875rem; }
        .codewalk button {
          background: transparent;
          color: ${PANEL_INK};
          border: 1px solid ${PANEL_RULE};
          border-radius: 2px;
          font-family: inherit;
          font-size: 0.75rem;
          padding: 0.25rem 0.625rem;
          cursor: pointer;
        }
        .codewalk button:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .codewalk button:disabled { opacity: 0.35; cursor: default; }
      `}</style>
    </div>
  )
}
