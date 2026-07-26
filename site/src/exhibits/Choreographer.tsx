// The site's core instrument: plays a TraceDoc as a stepped SVG diagram of
// HBM tiles, DMA edges, VMEM slots, carried state, and the compute unit.
// Motion is quantized (discrete frames, no easing); autoplay never starts
// itself and is disabled entirely under prefers-reduced-motion.
import { useEffect, useMemo, useRef, useState } from 'react'
import type { TraceDoc } from './types'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'

const TILE = 34
const GAP = 8

interface Props {
  trace: TraceDoc
}

export default function Choreographer({ trace }: Props) {
  const [i, setI] = useState(0)
  const [playing, setPlaying] = useState(false)
  const reduced = useRef(false)

  useEffect(() => {
    reduced.current = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  }, [])

  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => {
      setI((cur) => {
        if (cur >= trace.frames.length - 1) {
          setPlaying(false)
          return cur
        }
        return cur + 1
      })
    }, 900)
    return () => clearInterval(t)
  }, [playing, trace.frames.length])

  const frame = trace.frames[i]!

  const groups = useMemo(() => {
    const seen: string[] = []
    for (const t of trace.hbmTiles) if (!seen.includes(t.group)) seen.push(t.group)
    return seen
  }, [trace])

  // layout: HBM (or ICI) groups stacked on the left; slots, state, unit on the right
  const groupCols = Math.max(...trace.hbmTiles.map((t) => t.col)) + 1
  const leftW = groupCols * (TILE + GAP) + 70
  const rightX = leftW + 90
  const slotX = rightX
  const width = rightX + 260
  const groupH = TILE + 34
  const leftH = groups.length * groupH + 30
  const unitY = 16 + trace.slotIds.length * (TILE + GAP) + trace.stateIds.length * 22 + (trace.stateIds.length ? 42 : 8)
  const height = Math.max(leftH, unitY + 30) + 14

  const tilePos = (id: string) => {
    const t = trace.hbmTiles.find((x) => x.id === id)
    if (!t) return null
    const gi = groups.indexOf(t.group)
    return { x: 60 + t.col * (TILE + GAP), y: 16 + gi * groupH + 18 }
  }
  const slotPos = (id: string) => {
    const si = trace.slotIds.indexOf(id)
    return { x: slotX, y: 16 + si * (TILE + GAP) }
  }

  return (
    <div className="choreo">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${trace.title}, frame ${i + 1} of ${trace.frames.length}: ${frame.caption}`}>
        <text x={8} y={10} fill={PANEL_MUTE} fontSize={9} letterSpacing={1} fontFamily="inherit">
          HBM
        </text>
        {/* HBM groups */}
        {groups.map((g, gi) => {
          const remote = trace.remoteGroups.includes(g)
          return (
            <g key={g}>
              <text x={8} y={16 + gi * groupH + 12} fill={remote ? STEEL : PANEL_MUTE} fontSize={9} letterSpacing={1} fontFamily="inherit">
                {remote ? `${g} · ICI` : g}
              </text>
              {trace.hbmTiles
                .filter((t) => t.group === g)
                .map((t) => {
                  const p = tilePos(t.id)!
                  const hot = frame.reads.includes(t.id)
                  const dead = trace.deadTiles?.includes(t.id) ?? false
                  return (
                    <g key={t.id} opacity={dead ? 0.45 : 1}>
                      <rect x={p.x} y={p.y} width={TILE} height={TILE} fill={hot ? '#1a1d21' : 'transparent'} stroke={hot ? STEEL : PANEL_RULE} strokeWidth={hot ? 1.5 : 1} strokeDasharray={remote ? '3 2' : undefined} />
                      <text x={p.x + TILE / 2} y={p.y + TILE / 2 + 3} textAnchor="middle" fill={hot ? PANEL_INK : PANEL_MUTE} fontSize={10} fontFamily="inherit">
                        {t.id}
                      </text>
                      {dead && <line x1={p.x + 4} y1={p.y + TILE - 4} x2={p.x + TILE - 4} y2={p.y + 4} stroke={PANEL_MUTE} strokeWidth={1.5} />}
                    </g>
                  )
                })}
            </g>
          )
        })}

        {/* DMA edges: drop below the tile row into a free lane, run right past
            the grid, then rise to the destination slot. Never crosses tiles. */}
        {frame.dma.map((d) => {
          const from = tilePos(d.tile)
          const to = slotPos(d.toSlot)
          if (!from || !to) return null
          const x1 = from.x + TILE / 2
          const y1 = from.y + TILE
          const laneY = y1 + 10
          const busX = leftW + 24
          const x2 = to.x - 6
          const y2 = to.y + TILE / 2
          return (
            <g key={`${d.tile}-${d.toSlot}`}>
              <path
                d={`M ${x1} ${y1} L ${x1} ${laneY} L ${busX} ${laneY} L ${busX} ${y2} L ${x2} ${y2}`}
                fill="none"
                stroke={STEEL}
                strokeWidth={1.5}
                strokeDasharray={d.kind === 'ici' ? '4 3' : undefined}
              />
              <path d={`M ${x2} ${y2} l -6 -4 l 0 8 z`} fill={STEEL} />
            </g>
          )
        })}

        {/* VMEM slots */}
        <text x={slotX} y={10} fill={PANEL_MUTE} fontSize={9} letterSpacing={1} fontFamily="inherit">
          VMEM
        </text>
        {trace.slotIds.map((s) => {
          const p = slotPos(s)
          const occupant = frame.slots[s]
          const isAcc = s === 'acc'
          return (
            <g key={s}>
              <rect x={p.x} y={p.y} width={isAcc ? 190 : 150} height={TILE} fill={occupant ? '#1a1d21' : 'transparent'} stroke={occupant ? (isAcc ? COPPER : PANEL_INK) : PANEL_RULE} strokeWidth={1} />
              <text x={p.x + 8} y={p.y + TILE / 2 + 3} fill={PANEL_MUTE} fontSize={9} fontFamily="inherit">
                {s}
              </text>
              <text x={p.x + 42} y={p.y + TILE / 2 + 3} fill={occupant ? (isAcc ? COPPER : PANEL_INK) : PANEL_RULE} fontSize={10} fontFamily="inherit">
                {occupant ?? 'empty'}
              </text>
            </g>
          )
        })}

        {/* carried state */}
        {trace.stateIds.length > 0 && (
          <g>
            <text x={slotX} y={16 + trace.slotIds.length * (TILE + GAP) + 8} fill={PANEL_MUTE} fontSize={9} letterSpacing={1} fontFamily="inherit">
              CARRIED STATE
            </text>
            {trace.stateIds.map((sid, si) => (
              <text key={sid} x={slotX} y={16 + trace.slotIds.length * (TILE + GAP) + 28 + si * 20} fill={COPPER} fontSize={10.5} fontFamily="inherit">
                {sid} : {frame.state?.[sid] ?? '·'}
              </text>
            ))}
          </g>
        )}

        {/* compute unit */}
        {(() => {
          const y = unitY
          const active = frame.compute
          return (
            <g>
              <rect x={slotX} y={y} width={190} height={30} fill={active ? '#20150f' : 'transparent'} stroke={active ? COPPER : PANEL_RULE} strokeWidth={active ? 1.5 : 1} />
              <text x={slotX + 8} y={y + 19} fill={active ? COPPER : PANEL_MUTE} fontSize={10.5} fontFamily="inherit">
                {active ? `${active.unit} · ${active.expr}` : 'MXU · idle'}
              </text>
            </g>
          )
        })()}
      </svg>

      <p className="caption" aria-live="polite">
        <span className="fno">
          {String(i + 1).padStart(2, '0')}/{String(trace.frames.length).padStart(2, '0')}
        </span>
        {frame.caption}
      </p>

      <div className="controls">
        <button onClick={() => { setPlaying(false); setI(0) }} disabled={i === 0} aria-label="Back to start">|«</button>
        <button onClick={() => { setPlaying(false); setI((x) => Math.max(0, x - 1)) }} disabled={i === 0} aria-label="Previous step">«</button>
        <button
          onClick={() => {
            if (reduced.current) setI((x) => Math.min(trace.frames.length - 1, x + 1))
            else setPlaying((p) => !p)
          }}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? '❚❚' : '▶'}
        </button>
        <button onClick={() => { setPlaying(false); setI((x) => Math.min(trace.frames.length - 1, x + 1)) }} disabled={i === trace.frames.length - 1} aria-label="Next step">»</button>
        <input
          type="range"
          min={0}
          max={trace.frames.length - 1}
          value={i}
          onChange={(e) => { setPlaying(false); setI(Number(e.target.value)) }}
          aria-label="Scrub through frames"
        />
      </div>

      <style>{`
        .choreo { overflow-x: auto; }
        .choreo svg { width: 100%; min-width: 620px; height: auto; display: block; font-family: 'IBM Plex Mono', monospace; }
        .choreo .caption { font-size: 0.75rem; color: ${PANEL_INK}; padding: 0.625rem 0.875rem 0; min-height: 3.2em; margin: 0; }
        .choreo .fno { color: ${PANEL_MUTE}; margin-right: 0.75em; }
        .choreo .controls { display: flex; gap: 0.5rem; align-items: center; padding: 0.625rem 0.875rem 0.875rem; }
        .choreo button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.25rem 0.625rem; cursor: pointer; }
        .choreo button:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .choreo button:disabled { opacity: 0.35; cursor: default; }
        .choreo input[type='range'] { flex: 1; accent-color: ${COPPER}; }
      `}</style>
    </div>
  )
}
