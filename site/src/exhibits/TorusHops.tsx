// The torus, hop by hop: pick two chips on a 2D torus and read the shortest
// path with and without the wraparound links. The 8x8 grid stands for a full
// 16x16 axis pair; the arithmetic shown is exact for the grid drawn, and the
// link constants are the per-generation ICI figures from the scaling book.
import { useState } from 'react'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'

const N = 8
const CELL = 26
const GAP = 7
const PAD = 34

const GENS = [
  { id: 'v5e', label: 'v5e · 4.5e10 B/s per link', linkOneWay: 4.5e10 },
  { id: 'v6e', label: 'v6e · 9.0e10 B/s per link', linkOneWay: 9.0e10 },
]

const axisDist = (a: number, b: number, wrap: boolean) => {
  const d = Math.abs(a - b)
  return wrap ? Math.min(d, N - d) : d
}

export default function TorusHops() {
  const [a, setA] = useState<[number, number]>([1, 1])
  const [b, setB] = useState<[number, number]>([6, 5])
  const [picking, setPicking] = useState<'a' | 'b'>('b')
  const [wrap, setWrap] = useState(true)
  const [gen, setGen] = useState(GENS[1]!)

  const dx = axisDist(a[0], b[0], wrap)
  const dy = axisDist(a[1], b[1], wrap)
  const hops = dx + dy
  const worst = wrap ? N / 2 + N / 2 : 2 * (N - 1)

  // The drawn route: x first, then y, taking the shorter direction per axis.
  const step = (from: number, to: number) => {
    if (from === to) return 0
    const fwd = (to - from + N) % N
    const back = (from - to + N) % N
    if (!wrap) return to > from ? 1 : -1
    return fwd <= back ? 1 : -1
  }
  const route: [number, number][] = [[a[0], a[1]]]
  {
    let [x, y] = a
    while (x !== b[0]) {
      x = (x + step(x, b[0]) + N) % N
      route.push([x, y])
    }
    while (y !== b[1]) {
      y = (y + step(y, b[1]) + N) % N
      route.push([x, y])
    }
  }
  const onRoute = (x: number, y: number) => route.some(([rx, ry]) => rx === x && ry === y)

  const cx = (x: number) => PAD + x * (CELL + GAP) + CELL / 2
  const cy = (y: number) => PAD + y * (CELL + GAP) + CELL / 2
  const size = PAD * 2 + N * (CELL + GAP) - GAP

  const pick = (x: number, y: number) => {
    if (picking === 'a') {
      setA([x, y])
      setPicking('b')
    } else {
      setB([x, y])
      setPicking('a')
    }
  }

  return (
    <div className="torus">
      <svg
        viewBox={`0 0 ${size + 210} ${size}`}
        role="img"
        aria-label={`Torus with wraparound ${wrap ? 'on' : 'off'}: shortest path between the chosen chips is ${hops} hops`}
      >
        {/* wraparound arcs, drawn faint on the outer edges */}
        {wrap &&
          Array.from({ length: N }).map((_, i) => (
            <g key={`wrap${i}`} stroke={PANEL_RULE} fill="none">
              <path d={`M ${cx(0) - CELL / 2} ${cy(i)} C ${PAD - 26} ${cy(i)}, ${PAD - 26} ${cy(i)}, ${cx(N - 1) + CELL / 2} ${cy(i)}`} opacity={0.35} />
              <path d={`M ${cx(i)} ${cy(0) - CELL / 2} C ${cx(i)} ${PAD - 26}, ${cx(i)} ${PAD - 26}, ${cx(i)} ${cy(N - 1) + CELL / 2}`} opacity={0.35} />
            </g>
          ))}
        {/* neighbor links */}
        {Array.from({ length: N }).map((_, y) =>
          Array.from({ length: N }).map((_, x) => (
            <g key={`ln${x}.${y}`} stroke={PANEL_RULE}>
              {x < N - 1 && <line x1={cx(x) + CELL / 2} y1={cy(y)} x2={cx(x + 1) - CELL / 2} y2={cy(y)} />}
              {y < N - 1 && <line x1={cx(x)} y1={cy(y) + CELL / 2} x2={cx(x)} y2={cy(y + 1) - CELL / 2} />}
            </g>
          )),
        )}
        {/* route */}
        {route.slice(1).map(([x, y], i) => {
          const [px, py] = route[i]!
          const wrapped = Math.abs(x - px) > 1 || Math.abs(y - py) > 1
          return (
            <line
              key={`r${i}`}
              x1={cx(px)}
              y1={cy(py)}
              x2={cx(x)}
              y2={cy(y)}
              stroke={COPPER}
              strokeWidth={2}
              strokeDasharray={wrapped ? '3 4' : undefined}
              opacity={wrapped ? 0.7 : 1}
            />
          )
        })}
        {/* chips */}
        {Array.from({ length: N }).map((_, y) =>
          Array.from({ length: N }).map((_, x) => {
            const isA = x === a[0] && y === a[1]
            const isB = x === b[0] && y === b[1]
            return (
              <rect
                key={`c${x}.${y}`}
                x={cx(x) - CELL / 2}
                y={cy(y) - CELL / 2}
                width={CELL}
                height={CELL}
                rx={2}
                fill={isA || isB ? COPPER : onRoute(x, y) ? '#2a2018' : '#181b1f'}
                stroke={isA || isB ? COPPER : onRoute(x, y) ? COPPER : PANEL_RULE}
                strokeWidth={isA || isB ? 2 : 1}
                style={{ cursor: 'pointer' }}
                onClick={() => pick(x, y)}
              />
            )
          }),
        )}
        {/* readout */}
        <text x={size + 8} y={PAD + 6} fill={PANEL_INK} fontSize={12}>
          {hops} hops
        </text>
        <text x={size + 8} y={PAD + 26} fill={PANEL_MUTE} fontSize={10.5}>
          {dx} on x · {dy} on y
        </text>
        <text x={size + 8} y={PAD + 54} fill={PANEL_MUTE} fontSize={10.5}>
          worst pair on this grid:
        </text>
        <text x={size + 8} y={PAD + 72} fill={PANEL_INK} fontSize={11}>
          {worst} hops {wrap ? 'with wrap' : 'without wrap'}
        </text>
        <text x={size + 8} y={PAD + 100} fill={PANEL_MUTE} fontSize={10.5}>
          each hop re-sends over one
        </text>
        <text x={size + 8} y={PAD + 118} fill={STEEL} fontSize={11}>
          {gen.linkOneWay.toExponential(1)} B/s link
        </text>
        <text x={size + 8} y={PAD + 146} fill={PANEL_MUTE} fontSize={10.5}>
          click chips to move the pair
        </text>
      </svg>
      <div className="row">
        <button className={wrap ? 'on' : ''} onClick={() => setWrap(true)}>wraparound on</button>
        <button className={!wrap ? 'on' : ''} onClick={() => setWrap(false)}>wraparound off</button>
        <span className="spacer" />
        {GENS.map((g) => (
          <button key={g.id} className={g.id === gen.id ? 'on' : ''} onClick={() => setGen(g)}>
            {g.label}
          </button>
        ))}
      </div>
      <style>{`
        .torus svg { width: 100%; height: auto; min-width: 460px; display: block; }
        .torus { overflow-x: auto; }
        .torus .row { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.625rem; flex-wrap: wrap; }
        .torus .spacer { flex: 1; }
        .torus button {
          font-family: inherit; font-size: 0.6875rem; letter-spacing: 0.04em;
          background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE};
          border-radius: 2px; padding: 0.3rem 0.6rem; cursor: pointer;
        }
        .torus button:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .torus button.on { border-color: ${COPPER}; color: ${COPPER}; }
      `}</style>
    </div>
  )
}
