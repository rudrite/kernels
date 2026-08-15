// Label the die, in the honest two-part shape. The left panel is a schematic
// drawn after Figure 6 of the H100 whitepaper (full GH100 die), with the
// vendor's own block names hidden behind a picker; the counts in the key are
// the whitepaper's. The right panel is a real photograph of silicon, and it is
// deliberately a different chip: no open-licensed H100 die photo exists, so it
// shows the consumer GP102 (CC0, Fritzchens Fritz), captioned as what it is.
// Nothing on the photograph is labeled, because labeling an unlabeled
// photograph would be inventing a floorplan.
import { useState } from 'react'

const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#3a4048'
const ACCENT = '#d98c5f'
const PASS = '#58b98b'
const FAIL = '#d4685a'

interface Region {
  id: string
  label: string
  key: string
  x: number
  y: number
  w: number
  h: number
}

// The six block families of the full GH100 die, after whitepaper Figure 6.
const REGIONS: Region[] = [
  { id: 'pcie', label: 'PCIe Gen 5 host interface', key: 'the one strip that talks to the host; everything else talks to memory or peers', x: 30, y: 8, w: 300, h: 22 },
  { id: 'gpc', label: 'one GPC (9 TPCs, 18 SMs on the full die)', key: '8 of these hold the full die’s 144 SMs; an SXM5 part ships 132 enabled', x: 96, y: 40, w: 80, h: 96 },
  { id: 'l2', label: 'an L2 cache partition', key: 'two partitions, 60 MB on the full die, 50 MB enabled on SXM5, shared by every GPC', x: 96, y: 142, w: 168, h: 26 },
  { id: 'hbm', label: 'one HBM3 site and its PHY', key: '6 sites on the die; SXM5 enables 5 of them for 80 GB at 3.35 TB/s', x: 8, y: 40, w: 20, h: 230 },
  { id: 'sm', label: 'one SM', key: 'the repeated unit: 4 subpartitions, 4 Tensor Cores, 256 KB registers, 256 KB L1/SMEM', x: 100, y: 46, w: 16, h: 20 },
  { id: 'nvlink', label: 'the NVLink 4 links', key: '18 links at 25 GB/s per direction each: 900 GB/s per GPU into the scale-up domain', x: 30, y: 276, w: 300, h: 22 },
]

const CHOICES = [...REGIONS].sort((a, b) => a.label.localeCompare(b.label)).map((r) => r.label)

export default function DieLabeler() {
  const [picks, setPicks] = useState<Record<string, string>>({})
  const [checked, setChecked] = useState(false)

  const right = REGIONS.filter((r) => picks[r.id] === r.label).length
  const done = REGIONS.every((r) => picks[r.id])

  const numeral = (i: number) => String(i + 1).padStart(2, '0')

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', fontFamily: 'var(--mono)', color: PANEL_INK }}>
      <div style={{ flex: '1 1 340px', minWidth: 300 }}>
        <svg viewBox="0 0 360 306" style={{ width: '100%', display: 'block' }} aria-label="Schematic of the GH100 die after whitepaper Figure 6">
          <rect x="2" y="2" width="356" height="302" fill="none" stroke={PANEL_RULE} strokeWidth="2" />
          {/* HBM columns, three sites a side */}
          {[0, 1, 2].map((i) => (
            <g key={`hl${i}`}>
              <rect x={8} y={40 + i * 78} width={20} height={72} fill="none" stroke={PANEL_RULE} />
              <rect x={332} y={40 + i * 78} width={20} height={72} fill="none" stroke={PANEL_RULE} />
            </g>
          ))}
          {/* GPC grid, two rows of four */}
          {[0, 1, 2, 3].map((c) =>
            [0, 1].map((r) => (
              <rect key={`g${c}${r}`} x={40 + c * 72} y={40 + r * 132} width={64} height={96} fill="none" stroke={PANEL_RULE} />
            )),
          )}
          {/* the SM grid inside each GPC, suggested not drawn to count */}
          {[0, 1, 2, 3].map((c) =>
            [0, 1].map((r) =>
              [0, 1, 2].map((i) => (
                <line key={`s${c}${r}${i}`} x1={40 + c * 72} y1={64 + r * 132 + i * 24} x2={104 + c * 72} y2={64 + r * 132 + i * 24} stroke={PANEL_RULE} strokeWidth="0.5" />
              )),
            ),
          )}
          {/* L2 band across the middle, two partitions */}
          <rect x={40} y={142} width={136} height={26} fill="none" stroke={PANEL_RULE} />
          <rect x={184} y={142} width={136} height={26} fill="none" stroke={PANEL_RULE} />
          {/* PCIe top, NVLink bottom */}
          <rect x={30} y={8} width={300} height={22} fill="none" stroke={PANEL_RULE} />
          <rect x={30} y={276} width={300} height={22} fill="none" stroke={PANEL_RULE} />
          {/* numbered hotspots */}
          {REGIONS.map((r, i) => {
            const ok = checked && picks[r.id] === r.label
            const bad = checked && picks[r.id] && picks[r.id] !== r.label
            return (
              <g key={r.id}>
                <rect x={r.x} y={r.y} width={r.w} height={r.h} fill={ok ? 'rgba(88,185,139,0.18)' : bad ? 'rgba(212,104,90,0.18)' : 'rgba(217,140,95,0.10)'} stroke={ok ? PASS : bad ? FAIL : ACCENT} strokeWidth="1.5" />
                <circle cx={r.x + 10} cy={r.y + 10} r={8} fill="#17191d" stroke={ACCENT} />
                <text x={r.x + 10} y={r.y + 13} textAnchor="middle" fontSize="8" fill={ACCENT}>{numeral(i)}</text>
              </g>
            )
          })}
        </svg>
        <p style={{ fontSize: '0.7rem', color: PANEL_MUTE, margin: '0.5rem 0 0' }}>
          schematic drawn after NVIDIA H100 whitepaper Figure 6 (full GH100 die); block proportions indicative, counts as published
        </p>
      </div>

      <div style={{ flex: '1 1 260px', minWidth: 240 }}>
        {REGIONS.map((r, i) => (
          <div key={r.id} style={{ marginBottom: '0.5rem' }}>
            <label style={{ display: 'block', fontSize: '0.7rem', color: PANEL_MUTE }}>
              region {numeral(i)}
              <select
                value={picks[r.id] ?? ''}
                onChange={(e) => {
                  setPicks({ ...picks, [r.id]: e.target.value })
                  setChecked(false)
                }}
                style={{ display: 'block', width: '100%', marginTop: 2, background: '#17191d', color: PANEL_INK, border: `1px solid ${PANEL_RULE}`, borderRadius: 3, padding: '0.3rem', fontFamily: 'var(--mono)', fontSize: '0.72rem' }}
              >
                <option value="" disabled>
                  pick the block
                </option>
                {CHOICES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            {checked && picks[r.id] && (
              <p style={{ fontSize: '0.68rem', margin: '0.25rem 0 0', color: picks[r.id] === r.label ? PASS : FAIL }}>
                {picks[r.id] === r.label ? r.key : `no: this one is ${r.label}`}
              </p>
            )}
          </div>
        ))}
        <button
          type="button"
          onClick={() => setChecked(true)}
          disabled={!done}
          style={{ background: '#17191d', color: done ? ACCENT : PANEL_MUTE, border: `1px solid ${done ? ACCENT : PANEL_RULE}`, borderRadius: 3, padding: '0.4rem 0.8rem', fontFamily: 'var(--mono)', fontSize: '0.72rem', cursor: done ? 'pointer' : 'default' }}
        >
          check the six
        </button>
        {checked && (
          <span style={{ marginLeft: '0.6rem', fontSize: '0.75rem', color: right === REGIONS.length ? PASS : PANEL_INK }}>
            {right} / {REGIONS.length}
          </span>
        )}
      </div>

      <figure style={{ flex: '1 1 300px', minWidth: 260, margin: 0 }}>
        <img src="/die-gp102.jpg" alt="Bare-die photograph of the NVIDIA GP102 (GTX 1080 Ti), top metal layer" loading="lazy" style={{ width: '100%', display: 'block', border: `1px solid ${PANEL_RULE}` }} />
        <figcaption style={{ fontSize: '0.7rem', color: PANEL_MUTE, marginTop: '0.4rem', lineHeight: 1.5 }}>
          what silicon actually looks like: the bare GP102 die (GTX 1080 Ti, Pascal, a different chip and three generations older than the schematic), top metal layer after delidding, photographed by Fritzchens Fritz, CC0 1.0, via Wikimedia Commons. The glowing perimeter is the memory PHY and I/O ring; the compute floorplan sits inside it. No open-licensed H100 or TPU die photograph exists, and an unlabeled photograph stays unlabeled here.
        </figcaption>
      </figure>
    </div>
  )
}
