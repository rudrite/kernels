// Teaches BlockSpec legality by letting the reader break it. Every number on
// this panel is computed from the array shape, the block shape, the dtype's
// tiling rule, and the TPU v5e VMEM budget in data/chips.json. Nothing here
// is invented: illegal combinations produce a diagnostic built from the same
// arithmetic the verdict is computed from, not a canned string.
import { useMemo, useState } from 'react'
import chipsData from '../data/chips.json'

const COPPER = '#c88a70'
const STEEL = '#7f98ab'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

interface Chip {
  id: string
  vmemBytes: number | null
}
const V5E = (chipsData.chips as Chip[]).find((c) => c.id === 'v5e')!
const VMEM_BUDGET = V5E.vmemBytes!

type Dtype = 'bf16' | 'f32' | 'int8'
const DTYPE_BYTES: Record<Dtype, number> = { bf16: 2, f32: 4, int8: 1 }
// Minimum sublane (second-minor block dim) granularity per dtype. The lane
// (minor dim) granularity is fixed at 128 regardless of dtype.
const SUBLANE_MULT: Record<Dtype, number> = { f32: 8, bf16: 16, int8: 32 }
const LANE_MULT = 128

const DIM_OPTIONS = [512, 1024, 2048, 4096, 8192]
const BLOCK_OPTIONS = [8, 16, 32, 64, 128, 256, 512, 1024]

const fmtBytes = (v: number): string => {
  if (v >= 1e9) return `${(v / 1e9).toPrecision(3)} GB`
  if (v >= 1e6) return `${(v / 1e6).toPrecision(3)} MB`
  if (v >= 1e3) return `${(v / 1e3).toPrecision(3)} KB`
  return `${v} B`
}

// Grid is drawn schematically: at most this many rows/cols get their own
// cell, however large the real grid is. The real grid dims are always
// printed alongside it, so nothing is misrepresented, just abbreviated.
const MAX_SHOWN = 12

// locale-independent thousands grouping: the locale-dependent formatter
// differs between the build server and the reader's browser, breaking hydration
const fmt = (n: number): string => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',')

export default function BlockSpecSandbox() {
  const [M, setM] = useState(2048)
  const [N, setN] = useState(2048)
  const [dtype, setDtype] = useState<Dtype>('bf16')
  const [bm, setBm] = useState(256)
  const [bn, setBn] = useState(256)

  const gridM = Math.ceil(M / bm)
  const gridN = Math.ceil(N / bn)

  const sublaneMult = SUBLANE_MULT[dtype]
  const bmLegal = bm % sublaneMult === 0 || bm >= M
  const bnLegal = bn % LANE_MULT === 0 || bn >= N

  const bytesPerElem = DTYPE_BYTES[dtype]
  const bytesPerBlock = bm * bn * bytesPerElem
  const workingSet = bytesPerBlock * 3 // 3 blocks in flight: current, prefetch, drain
  const vmemFits = workingSet <= VMEM_BUDGET
  const vmemPct = Math.min((workingSet / VMEM_BUDGET) * 100, 100)

  const legal = bmLegal && bnLegal && vmemFits

  const diagnostics = useMemo(() => {
    const lines: string[] = []
    if (!bmLegal) {
      lines.push(
        `mosaic: bm=${bm} is not a multiple of the ${dtype} sublane granularity (${sublaneMult}) and does not cover the full array dim (M=${M}). smallest legal fix: bm=${sublaneMult}.`
      )
    }
    if (!bnLegal) {
      lines.push(
        `mosaic: bn=${bn} is not a multiple of the ${LANE_MULT}-lane register width and does not cover the full array dim (N=${N}). smallest legal fix: bn=${LANE_MULT}.`
      )
    }
    if (!vmemFits) {
      const perBlockBudget = Math.floor(VMEM_BUDGET / 3)
      lines.push(
        `mosaic: working set 3 × ${fmtBytes(bytesPerBlock)} = ${fmtBytes(workingSet)} exceeds the v5e VMEM budget of ${fmtBytes(VMEM_BUDGET)}. smallest legal fix: shrink the block so bytes/block ≤ ${fmtBytes(perBlockBudget)}.`
      )
    }
    return lines
  }, [bm, bn, dtype, M, N, sublaneMult, bmLegal, bnLegal, vmemFits, bytesPerBlock, workingSet])

  const shownRows = Math.min(gridM, MAX_SHOWN)
  const shownCols = Math.min(gridN, MAX_SHOWN)
  const truncated = gridM > MAX_SHOWN || gridN > MAX_SHOWN
  const boxW = 300
  const boxH = 220
  const cellW = boxW / shownCols
  const cellH = boxH / shownRows
  const originX = 46
  const originY = 14

  return (
    <div className="bss">
      <div className="row">
        <label>
          array M
          <select value={M} onChange={(e) => setM(Number(e.target.value))}>
            {DIM_OPTIONS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          array N
          <select value={N} onChange={(e) => setN(Number(e.target.value))}>
            {DIM_OPTIONS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          dtype
          <select value={dtype} onChange={(e) => setDtype(e.target.value as Dtype)}>
            <option value="bf16">bf16</option>
            <option value="f32">f32</option>
            <option value="int8">int8</option>
          </select>
        </label>
        <label>
          block bm
          <select value={bm} onChange={(e) => setBm(Number(e.target.value))}>
            {BLOCK_OPTIONS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          block bn
          <select value={bn} onChange={(e) => setBn(Number(e.target.value))}>
            {BLOCK_OPTIONS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
      </div>

      <p className={`verdict ${legal ? 'pass' : 'fail'}`}>
        {legal
          ? `legal: BlockSpec(${bm}, ${bn}) tiles the (${M}, ${N}) ${dtype} array`
          : `illegal: BlockSpec(${bm}, ${bn}) violates ${diagnostics.length} rule${diagnostics.length > 1 ? 's' : ''} on the (${M}, ${N}) ${dtype} array`}
      </p>

      <svg
        viewBox={`0 0 ${originX + boxW + 20} ${originY + boxH + 34}`}
        role="img"
        aria-label={`Array carved into a ${gridM} by ${gridN} grid of ${bm} by ${bn} blocks, first block highlighted`}
      >
        <text x={0} y={originY + boxH / 2} fill={PANEL_MUTE} fontSize={10.5} fontFamily="inherit" transform={`rotate(-90 10 ${originY + boxH / 2})`} textAnchor="middle">
          M = {M}
        </text>
        <text x={originX + boxW / 2} y={originY + boxH + 14} fill={PANEL_MUTE} fontSize={10.5} fontFamily="inherit" textAnchor="middle">
          N = {N}
        </text>
        {Array.from({ length: shownRows }).map((_, r) =>
          Array.from({ length: shownCols }).map((_, c) => {
            const first = r === 0 && c === 0
            return (
              <rect
                key={`${r}-${c}`}
                x={originX + c * cellW}
                y={originY + r * cellH}
                width={cellW}
                height={cellH}
                fill={first ? '#141c22' : 'transparent'}
                stroke={first ? STEEL : PANEL_RULE}
                strokeWidth={first ? 1.5 : 1}
              />
            )
          })
        )}
        <text x={originX + cellW / 2} y={originY + cellH / 2 + 4} fill={STEEL} fontSize={9.5} fontFamily="inherit" textAnchor="middle">
          0,0
        </text>
      </svg>
      <p className="gridnote">
        {gridM} × {gridN} = {fmt(gridM * gridN)} blocks launched
        {truncated ? ` (first ${shownRows} × ${shownCols} shown)` : ''}
      </p>

      <dl className="readout">
        <div><dt>grid</dt><dd>{gridM} × {gridN}</dd></div>
        <div><dt>sublane rule</dt><dd className={bmLegal ? 'pass' : 'fail'}>bm % {sublaneMult} {bmLegal ? '== 0' : '!= 0'}</dd></div>
        <div><dt>lane rule</dt><dd className={bnLegal ? 'pass' : 'fail'}>bn % {LANE_MULT} {bnLegal ? '== 0' : '!= 0'}</dd></div>
        <div><dt>bytes/block</dt><dd>{fmtBytes(bytesPerBlock)}</dd></div>
      </dl>

      <div className="meter">
        <div className="meter-label">
          <span>VMEM working set (3 blocks in flight)</span>
          <span className={vmemFits ? 'pass' : 'fail'}>{fmtBytes(workingSet)} / {fmtBytes(VMEM_BUDGET)}</span>
        </div>
        <div className="meter-track">
          <div className="meter-fill" style={{ width: `${vmemPct}%`, background: vmemFits ? PASS : FAIL }} />
        </div>
      </div>

      {diagnostics.length > 0 && (
        <div className="diag">
          {diagnostics.map((line) => (
            <p key={line} className="diag-line">{line}</p>
          ))}
          <p className="diag-caption">
            The message text above is our phrasing, written to teach the rule. The numbers in it
            (block sizes, byte counts, the {fmtBytes(VMEM_BUDGET)} budget) come from the array
            shape you set and the TPU v5e constants in data/chips.json, not from a real Mosaic run.
          </p>
        </div>
      )}

      <p className="provenance-line">
        VMEM budget {fmt(VMEM_BUDGET)} B for TPU v5e, source: {chipsData.source} (retrieved {chipsData.retrieved}).
      </p>

      <style>{`
        .bss { font-family: 'IBM Plex Mono', monospace; }
        .bss .row { display: flex; gap: 1rem; flex-wrap: wrap; padding-bottom: 0.75rem; }
        .bss label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; letter-spacing: 0.06em; }
        .bss select { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.4rem; }

        .bss .verdict { margin: 0 0 0.75rem; font-size: 0.8125rem; }
        .bss .verdict.pass { color: ${PASS}; }
        .bss .verdict.fail { color: ${FAIL}; }

        .bss { overflow-x: auto; }
        .bss svg { width: 100%; max-width: 380px; height: auto; display: block; }
        .bss .gridnote { margin: 0.4rem 0 0; font-size: 0.75rem; color: ${PANEL_MUTE}; }

        .bss .readout { display: flex; gap: 1.75rem; flex-wrap: wrap; padding-top: 0.875rem; margin: 0; }
        .bss dt { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .bss dd { margin: 0.15rem 0 0; font-size: 0.8125rem; color: ${PANEL_INK}; }
        .bss dd.pass { color: ${PASS}; }
        .bss dd.fail { color: ${FAIL}; }

        .bss .meter { padding-top: 0.875rem; }
        .bss .meter-label { display: flex; justify-content: space-between; font-size: 0.75rem; color: ${PANEL_MUTE}; margin-bottom: 0.3rem; }
        .bss .meter-label .pass { color: ${PASS}; }
        .bss .meter-label .fail { color: ${FAIL}; }
        .bss .meter-track { height: 8px; border: 1px solid ${PANEL_RULE}; border-radius: 1px; overflow: hidden; }
        .bss .meter-fill { height: 100%; }

        .bss .diag { margin-top: 0.875rem; border: 1px solid ${FAIL}; border-radius: 2px; padding: 0.75rem 0.875rem; background: #1a1210; }
        .bss .diag-line { margin: 0; color: ${FAIL}; font-size: 0.75rem; line-height: 1.5; }
        .bss .diag-line + .diag-line { margin-top: 0.4rem; }
        .bss .diag-caption { margin: 0.6rem 0 0; color: ${PANEL_MUTE}; font-size: 0.6875rem; line-height: 1.5; font-style: italic; }

        .bss .provenance-line { margin-top: 0.75rem; font-size: 0.6875rem; color: ${PANEL_MUTE}; }
      `}</style>
    </div>
  )
}
