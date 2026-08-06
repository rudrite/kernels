// A weight-stationary systolic array, stepped one cycle at a time: weights
// parked in the cells, activation diagonals marching in from the left,
// partial sums draining down. The grid is 6x6 for legibility; the readout
// scales the active fraction to the real array of the chosen generation.
// Motion is quantized; autoplay is opt-in and off under reduced motion.
import { useEffect, useState } from 'react'
import { prefersReducedMotion } from './reduced-motion'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'

const P = 6 // grid shown
const K = 12 // activation depth streamed through
const LAST = 2 * (P - 1) + K - 1 // final cycle with any cell active

const CHIPS = [
  { id: 'v5e', label: '128x128 (v5e)', dim: 128 },
  { id: 'v6e', label: '256x256 (v6e)', dim: 256 },
]

const CELL = 30
const GAP = 5

export default function SystolicArray() {
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [chip, setChip] = useState(CHIPS[0]!)
  const reduced = prefersReducedMotion()

  useEffect(() => {
    if (!playing) return
    const iv = setInterval(() => {
      setT((cur) => {
        if (cur >= LAST) {
          setPlaying(false)
          return cur
        }
        return cur + 1
      })
    }, 450)
    return () => clearInterval(iv)
  }, [playing])

  const active = (r: number, c: number) => t - r - c >= 0 && t - r - c < K
  const activeCount = Array.from({ length: P * P }).filter((_, i) => active(Math.floor(i / P), i % P)).length
  const allActive = activeCount === P * P
  const phase = t > LAST - 1 ? 'done' : allActive ? 'steady state' : t < 2 * (P - 1) ? 'fill' : 'drain'
  const macsNow = Math.round((activeCount / (P * P)) * chip.dim * chip.dim)

  const gridW = P * (CELL + GAP) - GAP
  const width = gridW + 62 + 190
  const height = gridW + 92

  return (
    <div className="systolic">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Systolic array, cycle ${t} of ${LAST}: ${phase}, ${activeCount} of ${P * P} cells active`}
      >
        <text x={64} y={12} fill={PANEL_MUTE} fontSize={10.5} letterSpacing={1}>
          activations enter →
        </text>
        {/* incoming diagonals, left edge */}
        {Array.from({ length: P }).map((_, r) => {
          const arriving = t - r >= 0 && t - r < K
          return (
            <line
              key={`in${r}`}
              x1={40}
              y1={26 + r * (CELL + GAP) + CELL / 2}
              x2={58}
              y2={26 + r * (CELL + GAP) + CELL / 2}
              stroke={arriving ? COPPER : PANEL_RULE}
              strokeWidth={arriving ? 2 : 1}
            />
          )
        })}
        {/* the array */}
        {Array.from({ length: P }).map((_, r) =>
          Array.from({ length: P }).map((_, c) => {
            const on = active(r, c)
            return (
              <rect
                key={`${r}.${c}`}
                x={62 + c * (CELL + GAP)}
                y={26 + r * (CELL + GAP)}
                width={CELL}
                height={CELL}
                rx={2}
                fill={on ? COPPER : '#181b1f'}
                fillOpacity={on ? 0.85 : 1}
                stroke={on ? COPPER : PANEL_RULE}
              />
            )
          }),
        )}
        {/* partial sums draining, bottom edge */}
        {Array.from({ length: P }).map((_, c) => {
          const draining = active(P - 1, c)
          return (
            <line
              key={`out${c}`}
              x1={62 + c * (CELL + GAP) + CELL / 2}
              y1={26 + gridW + 4}
              x2={62 + c * (CELL + GAP) + CELL / 2}
              y2={26 + gridW + 20}
              stroke={draining ? STEEL : PANEL_RULE}
              strokeWidth={draining ? 2 : 1}
            />
          )
        })}
        <text x={62} y={26 + gridW + 36} fill={PANEL_MUTE} fontSize={10.5} letterSpacing={1}>
          ↓ partial sums drain
        </text>
        {/* readout */}
        <text x={62 + gridW + 16} y={40} fill={PANEL_INK} fontSize={12} fontFamily="inherit">
          cycle {t}
        </text>
        <text x={62 + gridW + 16} y={60} fill={phase === 'steady state' ? COPPER : PANEL_MUTE} fontSize={11}>
          {phase}
        </text>
        <text x={62 + gridW + 16} y={84} fill={PANEL_INK} fontSize={11}>
          {activeCount}/{P * P} cells
        </text>
        <text x={62 + gridW + 16} y={104} fill={PANEL_MUTE} fontSize={10.5}>
          scaled to {chip.label}:
        </text>
        <text x={62 + gridW + 16} y={122} fill={PANEL_INK} fontSize={11}>
          {macsNow.toLocaleString('en-US')} MACs this cycle
        </text>
      </svg>
      <div className="row">
        <button onClick={() => setT((c) => Math.max(0, c - 1))}>← cycle</button>
        <button onClick={() => setT((c) => Math.min(LAST, c + 1))}>cycle →</button>
        {!reduced && (
          <button onClick={() => (playing ? setPlaying(false) : (t >= LAST && setT(0), setPlaying(true)))}>
            {playing ? 'pause' : 'play'}
          </button>
        )}
        <button onClick={() => (setPlaying(false), setT(0))}>reset</button>
        <span className="spacer" />
        {CHIPS.map((c) => (
          <button key={c.id} className={c.id === chip.id ? 'on' : ''} onClick={() => setChip(c)}>
            {c.label}
          </button>
        ))}
      </div>
      <style>{`
        .systolic svg { width: 100%; height: auto; min-width: 420px; display: block; }
        .systolic { overflow-x: auto; }
        .systolic .row { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.625rem; flex-wrap: wrap; }
        .systolic .spacer { flex: 1; }
        .systolic button {
          font-family: inherit; font-size: 0.6875rem; letter-spacing: 0.04em;
          background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE};
          border-radius: 2px; padding: 0.3rem 0.6rem; cursor: pointer;
        }
        .systolic button:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .systolic button.on { border-color: ${COPPER}; color: ${COPPER}; }
      `}</style>
    </div>
  )
}
