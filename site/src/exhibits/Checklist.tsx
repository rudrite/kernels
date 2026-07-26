// A quiet mono checklist. State persists to localStorage under
// 'checklist.' + id so a reader's progress survives a reload. The only
// color used is the pass green on a checked mark; everything else stays
// grayscale, matching the panel's restraint.
import { useEffect, useState } from 'react'

const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'

interface Props {
  id: string
  items: string[]
}

function isBooleanArray(v: unknown, length: number): v is boolean[] {
  return Array.isArray(v) && v.length === length && v.every((x) => typeof x === 'boolean')
}

export default function Checklist({ id, items }: Props) {
  const storageKey = `checklist.${id}`
  // Starts unchecked so the first client render matches the server-rendered
  // HTML; the stored state (if any) loads a moment later in an effect.
  const [checked, setChecked] = useState<boolean[]>(() => items.map(() => false))
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    const raw = window.localStorage.getItem(storageKey)
    if (raw) {
      try {
        const parsed: unknown = JSON.parse(raw)
        if (isBooleanArray(parsed, items.length)) setChecked(parsed)
      } catch {
        // malformed storage, start from unchecked
      }
    }
    setHydrated(true)
    // storageKey and items.length are stable for a given instance of this
    // checklist; re-running on every render would just replay the same read.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!hydrated) return
    window.localStorage.setItem(storageKey, JSON.stringify(checked))
  }, [checked, hydrated, storageKey])

  const toggle = (i: number) => {
    setChecked((cur) => cur.map((v, idx) => (idx === i ? !v : v)))
  }

  const n = checked.filter(Boolean).length
  const m = items.length

  return (
    <div className="checklist panel">
      <div className="head">
        <span className="count">
          {n}/{m}
        </span>
      </div>
      <ul>
        {items.map((label, i) => {
          const inputId = `${id}-${i}`
          const isChecked = checked[i] ?? false
          return (
            <li key={inputId}>
              <label htmlFor={inputId}>
                <input id={inputId} type="checkbox" checked={isChecked} onChange={() => toggle(i)} />
                <span className={`mark${isChecked ? ' on' : ''}`}>{isChecked ? '✓' : '·'}</span>
                <span className="label">{label}</span>
              </label>
            </li>
          )
        })}
      </ul>

      <style>{`
        .checklist { font-family: 'IBM Plex Mono', monospace; padding: 0.875rem; }
        .checklist .head { display: flex; justify-content: flex-end; padding-bottom: 0.5rem; border-bottom: 1px solid ${PANEL_RULE}; margin-bottom: 0.5rem; }
        .checklist .count { font-size: 0.75rem; color: ${PANEL_MUTE}; letter-spacing: 0.04em; }
        .checklist ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.375rem; }
        .checklist label { position: relative; display: flex; align-items: baseline; gap: 0.625rem; cursor: pointer; font-size: 0.8125rem; }
        .checklist input[type='checkbox'] { position: absolute; opacity: 0; width: 1px; height: 1px; }
        .checklist input:focus-visible ~ .mark { outline: 1px solid ${PANEL_INK}; outline-offset: 2px; }
        .checklist .mark { display: inline-block; width: 1em; text-align: center; color: ${PANEL_MUTE}; }
        .checklist .mark.on { color: ${PASS}; }
        .checklist .label { color: ${PANEL_MUTE}; }
        .checklist input:checked ~ .label { color: ${PANEL_INK}; }
        .checklist label:hover .label { color: ${PANEL_INK}; }
      `}</style>
    </div>
  )
}
