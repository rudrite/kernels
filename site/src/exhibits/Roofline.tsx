// The roofline playground: pick a chip and an op, drag the shape, watch the
// dot cross the ridge. Every number on this instrument is computed from the
// chip constants in data/chips.json (source cited on the panel) and the op's
// own FLOP/byte formulas. Nothing is invented.
import { useMemo, useState } from 'react'
import chipsData from '../data/chips.json'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'

interface Chip {
  id: string
  label: string
  bf16FlopsPerSec: number
  hbmBytesPerSec: number
}
const CHIPS = chipsData.chips as Chip[]

interface Op {
  id: string
  label: string
  /** n is the slider value; returns FLOPs and minimal HBM bytes for bf16. */
  cost: (n: number) => { flops: number; bytes: number; shape: string }
  min: number
  max: number
  step: number
}

const OPS: Op[] = [
  {
    id: 'matmul',
    label: 'square matmul N×N×N',
    min: 256,
    max: 16384,
    step: 256,
    cost: (n) => ({ flops: 2 * n ** 3, bytes: 2 * 3 * n ** 2, shape: `N = ${n}` }),
  },
  {
    id: 'add',
    label: 'elementwise add, N² tensors',
    min: 256,
    max: 16384,
    step: 256,
    cost: (n) => ({ flops: n ** 2, bytes: 2 * 3 * n ** 2, shape: `N = ${n}` }),
  },
  {
    id: 'softmax',
    label: 'softmax over rows, N×N',
    min: 256,
    max: 16384,
    step: 256,
    // ~5 elementwise passes worth of FLOPs; reads and writes the matrix once
    cost: (n) => ({ flops: 5 * n ** 2, bytes: 2 * 2 * n ** 2, shape: `N = ${n}` }),
  },
]

const W = 640
const H = 380
const PAD = { l: 64, r: 20, t: 16, b: 44 }
const X_MIN = 0.1
const X_MAX = 4096
const Y_MIN = 0.05e12
const Y_MAX = 2e15

const xPos = (v: number) => PAD.l + ((Math.log10(v) - Math.log10(X_MIN)) / (Math.log10(X_MAX) - Math.log10(X_MIN))) * (W - PAD.l - PAD.r)
const yPos = (v: number) => H - PAD.b - ((Math.log10(v) - Math.log10(Y_MIN)) / (Math.log10(Y_MAX) - Math.log10(Y_MIN))) * (H - PAD.t - PAD.b)

const fmt = (v: number, unit: string) => {
  if (v >= 1e15) return `${(v / 1e15).toPrecision(3)} P${unit}`
  if (v >= 1e12) return `${(v / 1e12).toPrecision(3)} T${unit}`
  if (v >= 1e9) return `${(v / 1e9).toPrecision(3)} G${unit}`
  if (v >= 1e6) return `${(v / 1e6).toPrecision(3)} M${unit}`
  return `${v.toPrecision(3)} ${unit}`
}

export default function Roofline() {
  const [chipId, setChipId] = useState(CHIPS[0]!.id)
  const [opId, setOpId] = useState(OPS[0]!.id)
  const [n, setN] = useState(2048)

  const chip = CHIPS.find((c) => c.id === chipId)!
  const op = OPS.find((o) => o.id === opId)!

  const { flops, bytes, shape } = op.cost(n)
  const intensity = flops / bytes
  const ridge = chip.bf16FlopsPerSec / chip.hbmBytesPerSec
  const attainable = Math.min(chip.bf16FlopsPerSec, chip.hbmBytesPerSec * intensity)
  const bound = intensity < ridge ? 'memory-bound' : 'compute-bound'
  const floorSec = Math.max(flops / chip.bf16FlopsPerSec, bytes / chip.hbmBytesPerSec)

  const roofPath = useMemo(() => {
    const pts: string[] = []
    for (let px = 0; px <= 100; px++) {
      const x = 10 ** (Math.log10(X_MIN) + (px / 100) * (Math.log10(X_MAX) - Math.log10(X_MIN)))
      const y = Math.min(chip.bf16FlopsPerSec, chip.hbmBytesPerSec * x)
      pts.push(`${px === 0 ? 'M' : 'L'} ${xPos(x).toFixed(1)} ${yPos(y).toFixed(1)}`)
    }
    return pts.join(' ')
  }, [chip])

  const dotX = xPos(Math.min(Math.max(intensity, X_MIN), X_MAX))
  const dotY = yPos(Math.min(Math.max(attainable, Y_MIN), Y_MAX))

  return (
    <div className="roofline">
      <div className="row">
        <label>
          chip
          <select value={chipId} onChange={(e) => setChipId(e.target.value)}>
            {CHIPS.map((c) => (
              <option key={c.id} value={c.id}>{c.label}</option>
            ))}
          </select>
        </label>
        <label>
          op
          <select value={opId} onChange={(e) => setOpId(e.target.value)}>
            {OPS.map((o) => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
        </label>
        <label className="grow">
          {shape}
          <input type="range" min={op.min} max={op.max} step={op.step} value={n} onChange={(e) => setN(Number(e.target.value))} aria-label="Shape size N" />
        </label>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Roofline for ${chip.label}: ${op.label} at ${shape} is ${bound}`}>
        {[0.1, 1, 10, 100, 1000].map((v) => (
          <g key={v}>
            <line x1={xPos(v)} y1={PAD.t} x2={xPos(v)} y2={H - PAD.b} stroke={PANEL_RULE} strokeWidth={1} />
            <text x={xPos(v)} y={H - PAD.b + 16} textAnchor="middle" fill={PANEL_MUTE} fontSize={10.5} fontFamily="inherit">{v}</text>
          </g>
        ))}
        {[1e11, 1e12, 1e13, 1e14, 1e15].map((v) => (
          <g key={v}>
            <line x1={PAD.l} y1={yPos(v)} x2={W - PAD.r} y2={yPos(v)} stroke={PANEL_RULE} strokeWidth={1} />
            <text x={PAD.l - 6} y={yPos(v) + 3} textAnchor="end" fill={PANEL_MUTE} fontSize={10.5} fontFamily="inherit">{fmt(v, 'F/s')}</text>
          </g>
        ))}
        <text x={(W + PAD.l - PAD.r) / 2} y={H - 8} textAnchor="middle" fill={PANEL_MUTE} fontSize={10.5} letterSpacing={0.8} fontFamily="inherit">
          ARITHMETIC INTENSITY · FLOPs PER HBM BYTE
        </text>

        <path d={roofPath} fill="none" stroke={PANEL_INK} strokeWidth={1.5} />
        <line x1={xPos(ridge)} y1={PAD.t} x2={xPos(ridge)} y2={H - PAD.b} stroke={STEEL} strokeWidth={1} strokeDasharray="4 3" />
        <text x={xPos(ridge) + 5} y={PAD.t + 12} fill={STEEL} fontSize={10.5} fontFamily="inherit">
          ridge {ridge.toFixed(0)} F/B
        </text>

        <circle cx={dotX} cy={dotY} r={5} fill={COPPER} />
        <circle cx={dotX} cy={dotY} r={9} fill="none" stroke={COPPER} strokeWidth={1} opacity={0.5} />
      </svg>

      <dl className="readout">
        <div><dt>intensity</dt><dd>{intensity.toFixed(1)} FLOPs/byte</dd></div>
        <div><dt>verdict</dt><dd className={bound === 'compute-bound' ? 'copper' : 'steel'}>{bound}</dd></div>
        <div><dt>attainable</dt><dd>{fmt(attainable, 'FLOP/s')}</dd></div>
        <div><dt>floor latency</dt><dd>{floorSec >= 1e-3 ? `${(floorSec * 1e3).toPrecision(3)} ms` : `${(floorSec * 1e6).toPrecision(3)} µs`}</dd></div>
      </dl>

      <style>{`
        .roofline { font-family: 'IBM Plex Mono', monospace; }
        .roofline .row { display: flex; gap: 1rem; flex-wrap: wrap; padding-bottom: 0.75rem; }
        .roofline label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.06em; }
        .roofline label.grow { flex: 1; min-width: 10rem; }
        .roofline select { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.4rem; }
        .roofline input[type='range'] { accent-color: ${COPPER}; }
        .roofline { overflow-x: auto; }
        .roofline svg { width: 100%; min-width: 560px; height: auto; display: block; }
        .roofline .readout { display: flex; gap: 1.75rem; flex-wrap: wrap; padding-top: 0.75rem; margin: 0; }
        .roofline dt { font-size: 0.625rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .roofline dd { margin: 0.15rem 0 0; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .roofline dd.copper { color: ${COPPER}; }
        .roofline dd.steel { color: ${STEEL}; }
      `}</style>
    </div>
  )
}
