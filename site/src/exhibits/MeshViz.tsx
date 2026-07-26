// The mesh walk: a 4x2 chip mesh (the shape of a v5e-8 slice), ICI links
// drawn between physical neighbors, and a hop-by-hop stepper over a ring
// collective run independently along each row's x-axis. Every number on the
// panel comes from chips.json's v5e ICI bandwidth and one stated example
// shard size; nothing else is invented.
import { useState } from 'react'
import chipsData from '../data/chips.json'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#9aa1a8'
const PANEL_RULE = '#23272c'
const BG = '#101215'

interface ChipSpec {
  id: string
  label: string
  iciBytesPerSecOneWay: number
}
const CHIPS = chipsData.chips as ChipSpec[]
const V5E = CHIPS.find((c) => c.id === 'v5e')!

const COLS = 4
const ROWS = 2
const RING = COLS // ring length for the x-axis collective (one ring per row)
const HOPS = RING - 1 // hops for a ring all-gather / reduce-scatter to finish
const SHARD_BYTES = 32e6 // fixed example: 32 MB, SI bytes, matching chips.json's own convention

type Collective = 'ag' | 'rs'

const W = 680
const H = 340
const COL_X = [70, 250, 430, 610]
const ROW_Y = [110, 230]
const R = 18
const ARC = 70 // how far the wrap curve bows out from its row

const gbps = (bytesPerSec: number) => `${(bytesPerSec / 1e9).toFixed(1)} GB/s`
const ms = (sec: number) => `${(sec * 1e3).toFixed(3)} ms`

// All-gather: at hop t a chip holds every shard it has received so far, plus
// its own. Shard k arrives at column c on the hop where t equals (c - k) mod
// RING, since every chip forwards to (col + 1) mod RING each hop.
function agHeld(col: number, hop: number): Set<number> {
  const held = new Set<number>()
  for (let t = 0; t <= hop; t++) held.add((((col - t) % RING) + RING) % RING)
  return held
}
function agNewlyArrived(col: number, hop: number): number | null {
  if (hop < 1) return null
  return (((col - hop) % RING) + RING) % RING
}

// Reduce-scatter runs the identical edge schedule in a different role:
// every chip's running accumulator gains exactly one folded-in contribution
// per hop, reaching all RING contributions (fully reduced) after HOPS hops.
function rsContributions(hop: number): number {
  return Math.min(hop + 1, RING)
}

interface ChipNode {
  id: number
  col: number
  row: number
}
const MESH: ChipNode[] = Array.from({ length: ROWS * COLS }, (_, id) => ({
  id,
  col: id % COLS,
  row: Math.floor(id / COLS),
}))

interface Edge {
  key: string
  kind: 'row' | 'wrap' | 'col'
  row: 0 | 1
  x1: number
  y1: number
  x2: number
  y2: number
  active: boolean
  labelX: number
  labelY: number
}

function buildEdges(hop: number): Edge[] {
  const edges: Edge[] = []
  const ringActive = hop >= 1 // every ring edge fires on every hop of a ring collective

  for (let row = 0; row < ROWS; row++) {
    const rowIdx = row as 0 | 1
    const y = ROW_Y[row]!
    for (let c = 0; c < COLS - 1; c++) {
      edges.push({
        key: `row${row}-${c}`,
        kind: 'row',
        row: rowIdx,
        x1: COL_X[c]!,
        y1: y,
        x2: COL_X[c + 1]!,
        y2: y,
        active: ringActive,
        labelX: (COL_X[c]! + COL_X[c + 1]!) / 2,
        labelY: row === 0 ? y - 10 : y + 18,
      })
    }
    // Torus wrap: column 3 back to column 0. Only the x-axis has a ring
    // longer than 2, so this is the only wrap edge the mesh needs; a y-axis
    // wrap with just 2 rows would retrace the direct link, so it is left
    // undrawn rather than duplicated.
    edges.push({
      key: `wrap${row}`,
      kind: 'wrap',
      row: rowIdx,
      x1: COL_X[3]!,
      y1: y,
      x2: COL_X[0]!,
      y2: y,
      active: ringActive,
      labelX: (COL_X[3]! + COL_X[0]!) / 2,
      labelY: row === 0 ? y - ARC - 8 : y + ARC + 16,
    })
  }
  for (let c = 0; c < COLS; c++) {
    edges.push({
      key: `col${c}`,
      kind: 'col',
      row: 0,
      x1: COL_X[c]!,
      y1: ROW_Y[0]! + R,
      x2: COL_X[c]!,
      y2: ROW_Y[1]! - R,
      active: false, // idle for an x-axis ring collective
      labelX: COL_X[c]! + 9,
      labelY: (ROW_Y[0]! + ROW_Y[1]!) / 2,
    })
  }
  return edges
}

function wrapPath(e: Edge): string {
  const midX = (e.x1 + e.x2) / 2
  const bow = e.row === 0 ? e.y1 - ARC : e.y1 + ARC
  return `M ${e.x1} ${e.y1} Q ${midX} ${bow} ${e.x2} ${e.y2}`
}

export default function MeshViz() {
  const [collective, setCollective] = useState<Collective>('ag')
  const [hop, setHop] = useState(0)

  const changeCollective = (next: Collective) => {
    setCollective(next)
    setHop(0)
  }

  const edges = buildEdges(hop)
  const linkBw = V5E.iciBytesPerSecOneWay
  const timePerHopSec = SHARD_BYTES / linkBw
  const bytesThisHop = hop >= 1 ? SHARD_BYTES : 0
  const cumulativeBytes = hop * SHARD_BYTES
  const cumulativeSec = hop * timePerHopSec
  const totalSec = HOPS * timePerHopSec

  return (
    <div className="meshviz">
      <div className="row controls">
        <label>
          collective
          <select value={collective} onChange={(e) => changeCollective(e.target.value as Collective)}>
            <option value="ag">all-gather (ring over x)</option>
            <option value="rs">reduce-scatter (ring over x)</option>
          </select>
        </label>
        <div className="stepper" role="group" aria-label="hop stepper">
          <button type="button" onClick={() => setHop((h) => Math.max(0, h - 1))} disabled={hop === 0} aria-label="previous hop">
            ← hop
          </button>
          <span className="hopread">hop {hop} of {HOPS}</span>
          <button type="button" onClick={() => setHop((h) => Math.min(HOPS, h + 1))} disabled={hop === HOPS} aria-label="next hop">
            hop →
          </button>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`4 by 2 chip mesh, ${collective === 'ag' ? 'all-gather' : 'reduce-scatter'} ring over x, hop ${hop} of ${HOPS}`}
      >
        {edges.map((e) => {
          const dim = e.kind === 'col'
          const stroke = dim ? PANEL_RULE : e.active ? STEEL : PANEL_RULE
          const labelColor = dim ? PANEL_MUTE : e.active ? STEEL : PANEL_MUTE
          const shared = { stroke, strokeWidth: e.active ? 2 : 1, opacity: dim ? 0.4 : 1 }
          return (
            <g key={e.key}>
              {e.kind === 'wrap' ? (
                <path d={wrapPath(e)} fill="none" {...shared} />
              ) : (
                <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} strokeDasharray={dim ? '3 3' : undefined} {...shared} />
              )}
              <text
                x={e.labelX}
                y={e.labelY}
                textAnchor={dim ? 'start' : 'middle'}
                fill={labelColor}
                fontSize={10.5}
                fontFamily="inherit"
                opacity={dim ? 0.7 : 1}
              >
                {gbps(linkBw)}
              </text>
            </g>
          )
        })}

        {MESH.map((chip) => {
          const x = COL_X[chip.col]!
          const y = ROW_Y[chip.row]!
          return (
            <g key={chip.id}>
              <circle cx={x} cy={y} r={R} fill={BG} stroke={PANEL_INK} strokeWidth={1.5} />
              <text x={x} y={y + 4} textAnchor="middle" fill={PANEL_INK} fontSize={11} fontFamily="inherit">
                c{chip.id}
              </text>
            </g>
          )
        })}
      </svg>

      <div className="cells" role="group" aria-label="payload accumulated per chip">
        {MESH.map((chip) => {
          const held = collective === 'ag' ? agHeld(chip.col, hop) : null
          const newly = collective === 'ag' ? agNewlyArrived(chip.col, hop) : null
          const contributions = collective === 'rs' ? rsContributions(hop) : 0
          const filled = collective === 'ag' ? (held?.size ?? 0) : contributions
          return (
            <div
              className="chipcells"
              key={chip.id}
              aria-label={`chip c${chip.id}: ${filled} of ${RING} ${collective === 'ag' ? 'shards held' : 'contributions folded in'}`}
            >
              <span className="chiplabel">c{chip.id}</span>
              <div className="stack">
                {Array.from({ length: RING }, (_, cellIdx) => {
                  let state: 'empty' | 'old' | 'new' = 'empty'
                  if (collective === 'ag' && held) {
                    if (held.has(cellIdx)) state = cellIdx === newly ? 'new' : 'old'
                  } else if (collective === 'rs') {
                    if (cellIdx < contributions) state = cellIdx === contributions - 1 && hop >= 1 ? 'new' : 'old'
                  }
                  return <span key={cellIdx} className={`cell ${state}`} />
                })}
              </div>
            </div>
          )
        })}
      </div>

      <p className="caption">
        {collective === 'ag'
          ? `all-gather, ring over x: at hop ${hop}, each chip holds ${hop + 1} of ${RING} shards.`
          : `reduce-scatter, ring over x: at hop ${hop}, each chip's running accumulator carries ${Math.min(hop + 1, RING)} of ${RING} folded-in contributions.`}
        {' '}Fixed example: a {(SHARD_BYTES / 1e6).toFixed(0)} MB shard ({SHARD_BYTES.toLocaleString('en-US')} bytes) moves over every active link each hop.
      </p>

      <dl className="readout">
        <div>
          <dt>link bandwidth</dt>
          <dd>{gbps(linkBw)} one-way ({V5E.label} ICI, chips.json's only ICI field, used for every link shown)</dd>
        </div>
        <div>
          <dt>bytes this hop</dt>
          <dd>{hop >= 1 ? `${(bytesThisHop / 1e6).toFixed(0)} MB / link` : '0 MB, initial state'}</dd>
        </div>
        <div>
          <dt>time this hop</dt>
          <dd className="copper">{hop >= 1 ? ms(timePerHopSec) : 'n/a, no hop taken yet'}</dd>
        </div>
        <div>
          <dt>elapsed</dt>
          <dd>{ms(cumulativeSec)} ({(cumulativeBytes / 1e6).toFixed(0)} MB moved / link)</dd>
        </div>
        <div>
          <dt>full collective, {HOPS} hops</dt>
          <dd>{ms(totalSec)}</dd>
        </div>
      </dl>

      <style>{`
        .meshviz { font-family: 'IBM Plex Mono', monospace; }
        .meshviz .row { display: flex; gap: 1.25rem; flex-wrap: wrap; align-items: flex-end; padding-bottom: 0.75rem; }
        .meshviz label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.06em; }
        .meshviz select { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.4rem; }

        .meshviz .stepper { display: flex; align-items: center; gap: 0.6rem; }
        .meshviz .stepper button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.35rem 0.6rem; cursor: pointer; }
        .meshviz .stepper button:hover:not(:disabled) { border-color: ${COPPER}; color: ${COPPER}; }
        .meshviz .stepper button:disabled { opacity: 0.35; cursor: default; }
        .meshviz .hopread { font-size: 0.75rem; color: ${PANEL_INK}; min-width: 7rem; text-align: center; }

        .meshviz { overflow-x: auto; }
        .meshviz svg { width: 100%; min-width: 600px; height: auto; display: block; }

        .meshviz .cells { display: flex; gap: 0.6rem; flex-wrap: wrap; padding-top: 0.75rem; }
        .meshviz .chipcells { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; }
        .meshviz .chiplabel { font-size: 0.6875rem; letter-spacing: 0.06em; color: ${PANEL_MUTE}; }
        .meshviz .stack { display: flex; flex-direction: column-reverse; gap: 2px; }
        .meshviz .cell { width: 22px; height: 8px; border: 1px solid ${PANEL_RULE}; border-radius: 1px; background: transparent; }
        .meshviz .cell.old { background: ${COPPER}; border-color: ${COPPER}; opacity: 0.45; }
        .meshviz .cell.new { background: ${COPPER}; border-color: ${COPPER}; opacity: 1; }

        .meshviz .caption { margin: 0.75rem 0 0; font-size: 0.8125rem; color: ${PANEL_INK}; line-height: 1.5; }

        .meshviz .readout { display: flex; gap: 1.75rem; flex-wrap: wrap; padding-top: 0.75rem; margin: 0; }
        .meshviz dt { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .meshviz dd { margin: 0.15rem 0 0; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .meshviz dd.copper { color: ${COPPER}; }
      `}</style>
    </div>
  )
}
