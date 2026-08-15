// Place the artifact: an excerpt of a real dump arrives unlabeled, and the
// drill is putting it back on the hourglass, level and side. The corpus is
// built at build time from committed artifacts and generated corpora
// (data/place-corpus.ts) and handed in as props, so the map's chapter data
// never ships to the browser.
import { useEffect, useState } from 'react'
import type { PlaceCell, PlaceItem } from '../data/place-corpus'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#9aa1a8'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

const STREAK_KEY = 'gym.place.streak'
const STREAK_TO_PASS = 5

const LEFT: string[] = ['torch', 'gpu']
const RIGHT: string[] = ['jax', 'tpu']

const cellKey = (level: number, side: string) => `${level}:${side}`

interface Props {
  items: PlaceItem[]
  cells: PlaceCell[]
}

export default function PlaceIt({ items, cells }: Props) {
  const [qi, setQi] = useState(0)
  const [picked, setPicked] = useState<string | null>(null)
  const [streak, setStreak] = useState(0)
  const [hydrated, setHydrated] = useState(false)

  // One cell at a time, then one artifact from it: sampling the corpus flat
  // would ask about a jaxpr most of the time, since most of it is jaxprs.
  const draw = (avoid?: number) => {
    const cellsWithItems = [...new Set(items.map((i) => cellKey(i.level, i.side)))]
    const cell = cellsWithItems[Math.floor(Math.random() * cellsWithItems.length)]!
    const pool = items.map((it, i) => i).filter((i) => cellKey(items[i]!.level, items[i]!.side) === cell && i !== avoid)
    return pool[Math.floor(Math.random() * pool.length)] ?? 0
  }

  useEffect(() => {
    setQi(draw())
    setHydrated(true)
  }, [])
  useEffect(() => {
    if (!hydrated) return
    window.localStorage.setItem(STREAK_KEY, String(streak))
    window.dispatchEvent(new Event('storage'))
  }, [streak, hydrated])

  const item = items[qi]!
  const answer = cellKey(item.level, item.side)
  const levels = [...new Set(cells.map((c) => c.level))].sort((a, b) => a - b)

  const place = (key: string) => {
    if (picked !== null) return
    setPicked(key)
    setStreak(key === answer ? streak + 1 : 0)
  }
  const next = () => {
    setPicked(null)
    setQi(draw(qi))
  }

  const passing = streak >= STREAK_TO_PASS
  const right = picked === answer

  return (
    <div className="place">
      <div className="head">
        <span className="prompt">an artifact, no label. Which floor of the map wrote it, and which side?</span>
        <span className="streak" style={{ color: passing ? PASS : PANEL_MUTE }}>
          streak {streak}
          {passing ? ' ✓' : ` / ${STREAK_TO_PASS}`}
        </span>
      </div>
      <pre>
        {item.lines.map((l, i) => (
          <span key={i} className="ln">
            {l || ' '}
            {'\n'}
          </span>
        ))}
      </pre>
      <div className="grid" role="group" aria-label="The hourglass, one button per cell">
        {levels.map((level) => (
          <div key={level} className="row">
            <span className="lv">L{level}</span>
            {[LEFT, ['waist'], RIGHT].map((group, gi) => {
              const cell = cells.find((c) => c.level === level && group.includes(c.side))
              if (!cell) return <span key={gi} className="slot empty" />
              const key = cellKey(cell.level, cell.side)
              const state =
                picked === null ? '' : key === answer ? 'right' : key === picked ? 'wrong' : 'dim'
              return (
                <button key={gi} type="button" className={`slot cell ${state}`} onClick={() => place(key)}>
                  {cell.label}
                </button>
              )
            })}
          </div>
        ))}
      </div>
      <div className="foot" aria-live="polite">
        {picked === null ? (
          <span className="hint">the vocabulary is the tell: refs and grids, dimension_numbers, tiled layouts, device data</span>
        ) : (
          <>
            <span className="told">
              <span className="verdict" style={{ color: right ? PASS : FAIL }}>
                {right ? 'placed' : `it belongs at L${item.level}`} · {item.title}
              </span>
              <span className="prov">{item.provenance}</span>
            </span>
            <button type="button" className="next" onClick={next}>
              next →
            </button>
          </>
        )}
      </div>

      <style>{`
        .place { font-family: 'IBM Plex Mono', monospace; }
        .place .head { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; padding-bottom: 0.625rem; flex-wrap: wrap; }
        .place .prompt { font-size: 0.8125rem; color: ${PANEL_INK}; }
        .place .streak { font-size: 0.6875rem; letter-spacing: 0.06em; }
        .place pre { margin: 0; border: 1px solid ${PANEL_RULE}; padding: 0.625rem 0.75rem; overflow-x: auto; max-height: 18rem; overflow-y: auto; font-size: 0.75rem; line-height: 1.65; color: ${PANEL_INK}; }
        .place .grid { display: flex; flex-direction: column; gap: 0.3rem; padding-top: 0.75rem; }
        .place .row { display: grid; grid-template-columns: 2.25rem repeat(3, minmax(0, 1fr)); gap: 0.3rem; align-items: stretch; }
        .place .lv { font-size: 0.6875rem; color: ${PANEL_MUTE}; align-self: center; }
        .place .slot { min-height: 1.85rem; }
        .place .slot.empty { border: 1px dashed ${PANEL_RULE}; opacity: 0.3; }
        .place .cell { font-family: inherit; font-size: 0.6875rem; background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; padding: 0.3rem 0.5rem; cursor: pointer; text-align: left; }
        .place .cell:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .place .cell.right { border-color: ${PASS}; color: ${PASS}; }
        .place .cell.wrong { border-color: ${FAIL}; color: ${FAIL}; }
        .place .cell.dim { opacity: 0.4; }
        .place .foot { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding-top: 0.625rem; min-height: 2rem; flex-wrap: wrap; }
        .place .hint, .place .verdict, .place .prov { font-size: 0.6875rem; color: ${PANEL_MUTE}; }
        .place .told { display: flex; flex-direction: column; gap: 0.15rem; flex: 1; min-width: 0; }
        .place .verdict { line-height: 1.5; }
        .place .prov { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .place .next { font-family: inherit; font-size: 0.75rem; background: transparent; color: ${COPPER}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; padding: 0.3rem 0.75rem; cursor: pointer; }
        .place .next:hover { border-color: ${COPPER}; }
        @media (max-width: 640px) {
          .place .row { grid-template-columns: 2rem repeat(3, minmax(0, 1fr)); }
          .place .cell { font-size: 0.625rem; padding: 0.3rem 0.35rem; }
        }
      `}</style>
    </div>
  )
}
