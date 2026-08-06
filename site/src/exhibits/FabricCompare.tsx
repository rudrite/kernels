// Two fabrics, one question: how far away is everybody else? Pick a source
// on the torus and every chip shows its hop distance; on the switched node
// every GPU is one switch crossing away by construction. Egress figures are
// the per-device numbers from the scaling book's TPU and GPU chapters.
import { useState } from 'react'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'

const N = 4
const CELL = 30
const GAP = 10
const PAD = 24

const TPUS = [
  { id: 'v5e', label: 'v5e', egress: 4 * 4.5e10 },
  { id: 'v6e', label: 'v6e', egress: 4 * 9.0e10 },
]
const GPUS = [
  { id: 'h100', label: 'H100', egress: 4.5e11 },
  { id: 'b200', label: 'B200', egress: 9.0e11 },
]

const dist = (a: number, b: number) => {
  const d = Math.abs(a - b)
  return Math.min(d, N - d)
}

export default function FabricCompare() {
  const [src, setSrc] = useState<[number, number]>([0, 0])
  const [tpu, setTpu] = useState(TPUS[1]!)
  const [gpu, setGpu] = useState(GPUS[0]!)

  const gridW = N * (CELL + GAP) - GAP
  const leftW = PAD * 2 + gridW
  const rightX = leftW + 40
  const width = rightX + 320
  const height = PAD * 2 + gridW + 46

  const cx = (x: number) => PAD + x * (CELL + GAP) + CELL / 2
  const cy = (y: number) => PAD + 14 + y * (CELL + GAP) + CELL / 2

  const maxHops = N / 2 + N / 2
  const shade = (h: number) => (h === 0 ? COPPER : ['#a8785f', '#7d5f50', '#544741', '#3a3634'][h - 1] ?? '#3a3634')

  const gpuY = (i: number) => PAD + 20 + i * 34
  return (
    <div className="fabric">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`A 4x4 torus where the farthest chip is ${maxHops} hops from the source, next to an 8-GPU switched node where every GPU is one switch crossing away`}
      >
        <text x={PAD} y={16} fill={PANEL_MUTE} fontSize={10.5} letterSpacing={1}>
          TPU · 4x4 torus (wrap on)
        </text>
        {Array.from({ length: N }).map((_, y) =>
          Array.from({ length: N }).map((_, x) => {
            const h = dist(x, src[0]) + dist(y, src[1])
            return (
              <g key={`${x}.${y}`}>
                <rect
                  x={cx(x) - CELL / 2}
                  y={cy(y) - CELL / 2}
                  width={CELL}
                  height={CELL}
                  rx={2}
                  fill={shade(h)}
                  stroke={h === 0 ? COPPER : PANEL_RULE}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setSrc([x, y])}
                />
                <text x={cx(x)} y={cy(y) + 4} fill={h <= 1 ? '#16181b' : PANEL_MUTE} fontSize={11} textAnchor="middle">
                  {h}
                </text>
              </g>
            )
          }),
        )}
        <text x={PAD} y={height - 10} fill={PANEL_MUTE} fontSize={10.5}>
          numbers are hops from the copper chip · click to move it
        </text>

        <text x={rightX} y={16} fill={PANEL_MUTE} fontSize={10.5} letterSpacing={1}>
          GPU · 8-GPU switched node
        </text>
        {/* switch bar */}
        <rect x={rightX + 150} y={PAD + 12} width={26} height={8 * 34 - 14} rx={2} fill="#181b1f" stroke={STEEL} />
        <text x={rightX + 163} y={PAD + 8 + 4 * 34} fill={STEEL} fontSize={10} textAnchor="middle" transform={`rotate(90 ${rightX + 163} ${PAD + 8 + 4 * 34})`}>
          NVSwitch
        </text>
        {Array.from({ length: 8 }).map((_, i) => (
          <g key={`g${i}`}>
            <rect x={rightX} y={gpuY(i)} width={64} height={22} rx={2} fill={i === 0 ? COPPER : '#181b1f'} stroke={i === 0 ? COPPER : PANEL_RULE} />
            <text x={rightX + 32} y={gpuY(i) + 15} fill={i === 0 ? '#16181b' : PANEL_MUTE} fontSize={10} textAnchor="middle">
              gpu{i}
            </text>
            <line x1={rightX + 64} y1={gpuY(i) + 11} x2={rightX + 150} y2={gpuY(i) + 11} stroke={i === 0 ? STEEL : PANEL_RULE} strokeWidth={i === 0 ? 2 : 1} />
          </g>
        ))}
        <text x={rightX + 196} y={PAD + 32} fill={PANEL_INK} fontSize={11}>
          every pair: 1 crossing
        </text>
        <text x={rightX + 196} y={PAD + 56} fill={PANEL_MUTE} fontSize={10.5}>
          torus worst pair: {maxHops} hops
        </text>
        <text x={rightX + 196} y={PAD + 88} fill={PANEL_MUTE} fontSize={10.5}>
          per-device egress
        </text>
        <text x={rightX + 196} y={PAD + 106} fill={COPPER} fontSize={11}>
          {tpu.label}: {tpu.egress.toExponential(1)} B/s
        </text>
        <text x={rightX + 196} y={PAD + 124} fill={STEEL} fontSize={11}>
          {gpu.label}: {gpu.egress.toExponential(1)} B/s
        </text>
      </svg>
      <div className="row">
        {TPUS.map((c) => (
          <button key={c.id} className={c.id === tpu.id ? 'on' : ''} onClick={() => setTpu(c)}>
            {c.label}
          </button>
        ))}
        <span className="spacer" />
        {GPUS.map((c) => (
          <button key={c.id} className={c.id === gpu.id ? 'on' : ''} onClick={() => setGpu(c)}>
            {c.label}
          </button>
        ))}
      </div>
      <style>{`
        .fabric svg { width: 100%; height: auto; min-width: 520px; display: block; }
        .fabric { overflow-x: auto; }
        .fabric .row { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.625rem; flex-wrap: wrap; }
        .fabric .spacer { flex: 1; }
        .fabric button {
          font-family: inherit; font-size: 0.6875rem; letter-spacing: 0.04em;
          background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE};
          border-radius: 2px; padding: 0.3rem 0.6rem; cursor: pointer;
        }
        .fabric button:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .fabric button.on { border-color: ${COPPER}; color: ${COPPER}; }
      `}</style>
    </div>
  )
}
