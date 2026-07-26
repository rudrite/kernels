// The command palette: cmd/ctrl-k (or the header button) opens it, type to
// filter everything on the site, arrows plus enter to go. Chapters rank
// first so the palette doubles as the chapter picker.
import { useEffect, useMemo, useRef, useState } from 'react'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#9aa1a8'
const PANEL_RULE = '#23272c'

export interface PaletteEntry {
  kind: string
  title: string
  href: string
  hint: string
  keywords: string
}

const KIND_ORDER = ['chapter', 'page', 'instrument', 'drill', 'walk', 'lab', 'mistake', 'op']

interface Props {
  index: PaletteEntry[]
}

export default function SearchPalette({ index }: Props) {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const [sel, setSel] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      } else if (e.key === 'Escape') {
        setOpen(false)
      }
    }
    const onOpen = () => setOpen(true)
    window.addEventListener('keydown', onKey)
    window.addEventListener('open-palette', onOpen)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('open-palette', onOpen)
    }
  }, [])

  useEffect(() => {
    if (open) {
      setQ('')
      setSel(0)
      // the input mounts in the same tick the overlay opens
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  const results = useMemo(() => {
    const needle = q.trim().toLowerCase()
    const pool = needle
      ? index.filter((e) => `${e.title} ${e.keywords} ${e.hint}`.toLowerCase().includes(needle))
      : index.filter((e) => e.kind === 'chapter')
    return [...pool]
      .sort((a, b) => KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind))
      .slice(0, 12)
  }, [q, index])

  useEffect(() => setSel(0), [q])

  const go = (href: string) => {
    setOpen(false)
    location.href = href
  }

  if (!open) return null
  return (
    <div className="pal-overlay" onClick={() => setOpen(false)} role="dialog" aria-modal="true" aria-label="Search the site">
      <div className="pal" onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(results.length - 1, s + 1)) }
            else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(0, s - 1)) }
            else if (e.key === 'Enter' && results[sel]) go(results[sel]!.href)
          }}
          placeholder="search chapters, instruments, labs, ops…"
          aria-label="Search"
        />
        <ol>
          {results.map((r, i) => (
            <li key={`${r.href}-${r.title}`}>
              <button type="button" className={i === sel ? 'hot' : ''} onMouseEnter={() => setSel(i)} onClick={() => go(r.href)}>
                <span className="rt">{r.title}</span>
                <span className="rh">{r.hint}</span>
              </button>
            </li>
          ))}
          {results.length === 0 && <li className="none">nothing matches</li>}
        </ol>
        <div className="foot">↑↓ move · enter go · esc close</div>
      </div>

      <style>{`
        .pal-overlay { position: fixed; inset: 0; background: rgba(10, 11, 13, 0.55); z-index: 90; display: flex; justify-content: center; padding-top: 12vh; }
        .pal { width: min(90vw, 34rem); height: fit-content; background: #101215; border: 1px solid ${PANEL_RULE}; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; box-shadow: 0 12px 48px rgba(0,0,0,0.5); }
        .pal input { width: 100%; box-sizing: border-box; background: transparent; border: none; border-bottom: 1px solid ${PANEL_RULE}; color: ${PANEL_INK}; font: inherit; font-size: 0.9375rem; padding: 0.875rem 1rem; outline: none; }
        .pal input::placeholder { color: #7e868d; }
        .pal ol { list-style: none; margin: 0; padding: 0.375rem 0; max-height: 50vh; overflow-y: auto; }
        .pal button { display: flex; justify-content: space-between; gap: 1rem; width: 100%; background: transparent; border: none; border-left: 3px solid transparent; color: ${PANEL_INK}; font: inherit; font-size: 0.8125rem; padding: 0.5rem 1rem; cursor: pointer; text-align: left; }
        .pal button.hot { background: #1a1d21; border-left-color: ${COPPER}; }
        .pal .rt { color: ${PANEL_INK}; }
        .pal button.hot .rt { color: ${COPPER}; }
        .pal .rh { color: ${PANEL_MUTE}; font-size: 0.6875rem; white-space: nowrap; align-self: center; }
        .pal .none { padding: 0.75rem 1rem; color: ${PANEL_MUTE}; font-size: 0.8125rem; }
        .pal .foot { border-top: 1px solid ${PANEL_RULE}; padding: 0.5rem 1rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
      `}</style>
    </div>
  )
}
