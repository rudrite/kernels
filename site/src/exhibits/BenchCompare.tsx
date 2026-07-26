// A paste-in comparator for the lab notebooks' results blob. Every lab on
// this site prints one JSON blob whose "results" is a dict of named
// measurements; paste it here and each entry lines up against the
// published record it corresponds to. Parsed locally; nothing typed here
// leaves the browser.
import { useMemo, useState } from 'react'
import publishedRecords from '../../../bench/results.json'

const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const COPPER = '#c88a70'
const FAIL = '#c4574a'

interface PublishedRecord {
  id: string
  kernel: string
  metric: string
  value: string
  chip: string
}
const RECORDS = publishedRecords as PublishedRecord[]

// blob keys the notebooks emit, mapped to the bench row that records the
// same measurement on this site's reference chip
const KEY_TO_RECORD: Record<string, string> = {
  'lab0.1/matmul NxNxN': 'M-001',
  'lab0.1/matmul skinny 8xNxN': 'M-002',
  'lab0.1/elementwise add': 'M-003',
  'lab0.1/softmax rows': 'M-004',
  'lab0.1/reduce_sum': 'M-005',
  'gate01/matmul': 'M-006',
  'gate01/softmax': 'M-007',
  'gate02/spill': 'M-008',
  'gate03/flash': 'M-009',
  'gate03/grads_bf16': 'M-010',
  'lab3.4/causal_scaling': 'M-011',
  'power/matmul_tuned': 'M-012',
  'power/softmax_tuned': 'M-013',
  'power/flash_scaling': 'M-014',
  'power/windowed': 'M-015',
  'power/ragged': 'M-016',
  'power/int8_decode': 'M-017',
  'gate02/byte_confirm': 'M-018',
  'gate03/forward': 'M-019',
  'gate03/grads': 'M-020',
  'gate02/profiled': 'M-021',
  'gate02/achieved': 'M-023',
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

// one line of honest text for any value shape a blob carries
function flatten(v: unknown, depth = 0): string {
  if (typeof v === 'number') {
    if (v !== 0 && Math.abs(v) < 0.001) return v.toExponential(2)
    return Number.isInteger(v) ? String(v) : String(Math.round(v * 10000) / 10000)
  }
  if (typeof v === 'string') return v.length > 90 ? v.slice(0, 90) + '…' : v
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (v === null || v === undefined) return '·'
  if (Array.isArray(v)) {
    const parts = v.map((x) => flatten(x, depth + 1))
    const joined = parts.join('  |  ')
    return joined.length > 220 ? joined.slice(0, 220) + '…' : joined
  }
  if (isRecord(v)) {
    const parts = Object.entries(v)
      .filter(([k]) => k !== 'op' && k !== 'fusion_ops' && k !== 'method')
      .map(([k, val]) => `${k} ${flatten(val, depth + 1)}`)
    const joined = parts.join(' · ')
    return joined.length > 220 && depth > 0 ? joined.slice(0, 220) + '…' : joined
  }
  return String(v)
}

interface ParsedBlob {
  chip: string | null
  notebook: string | null
  entries: { key: string; text: string }[]
}
type ParseOutcome = { ok: true; data: ParsedBlob } | { ok: false; error: string }

function parseBlob(text: string): ParseOutcome {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch (err) {
    const detail = err instanceof Error ? err.message : 'invalid JSON'
    return { ok: false, error: `could not parse JSON: ${detail}` }
  }
  if (!isRecord(raw)) return { ok: false, error: 'expected the JSON object a lab’s final cell prints' }

  const results = raw.results
  const entries: { key: string; text: string }[] = []
  if (isRecord(results)) {
    for (const [key, value] of Object.entries(results)) entries.push({ key, text: flatten(value) })
  } else if (Array.isArray(results)) {
    // the oldest blob shape: a list of {op, ...} rows
    for (const [i, item] of results.entries()) {
      const key = isRecord(item) && typeof item.op === 'string' ? item.op : `row ${i + 1}`
      entries.push({ key, text: flatten(item) })
    }
  } else {
    return { ok: false, error: 'no "results" on the blob; paste the whole JSON the final cell printed' }
  }

  if (isRecord(raw.museum_captures)) {
    for (const [key, value] of Object.entries(raw.museum_captures)) {
      entries.push({ key: `museum/${key}`, text: flatten(String(value).slice(0, 120)) })
    }
  }

  return {
    ok: true,
    data: {
      chip: typeof raw.chip === 'string' ? raw.chip : null,
      notebook: typeof raw.notebook === 'string' ? raw.notebook : null,
      entries,
    },
  }
}

function findPublished(key: string): PublishedRecord | null {
  const id = KEY_TO_RECORD[key]
  if (id) return RECORDS.find((r) => r.id === id) ?? null
  const needle = key.split('/').pop()?.trim().toLowerCase() ?? ''
  if (needle.length < 4) return null
  let best: PublishedRecord | null = null
  for (const rec of RECORDS) {
    const hay = rec.kernel.toLowerCase()
    if (hay.includes(needle)) {
      if (!best || rec.kernel.length < best.kernel.length) best = rec
    }
  }
  return best
}

const PLACEHOLDER = `{
 "chip": "TPU v6 lite",
 "notebook": "gate-close",
 "results": {
  "gate03/forward": { "f32_max_err": 3.34e-06, "f32_bar_1e3": true }
 }
}`

export default function BenchCompare() {
  const [text, setText] = useState('')
  const trimmed = text.trim()
  const parsed = useMemo(() => (trimmed ? parseBlob(trimmed) : null), [trimmed])

  return (
    <div className="benchcompare panel">
      <label className="paste-label" htmlFor="bench-input">
        paste the results blob printed by a lab notebook's final cell
      </label>
      <textarea
        id="bench-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={PLACEHOLDER}
        spellCheck={false}
        rows={7}
      />

      {parsed === null && (
        <p className="hint">
          waiting for a results blob ·{' '}
          <button type="button" className="try" onClick={() => setText(PLACEHOLDER)}>
            try the example
          </button>
        </p>
      )}
      {parsed !== null && !parsed.ok && <p className="error">{parsed.error}</p>}

      {parsed !== null && parsed.ok && (
        <>
          <div className="meta">
            <span>chip: {parsed.data.chip ?? '·'}</span>
            <span>notebook: {parsed.data.notebook ?? '·'}</span>
            <span>{parsed.data.entries.length} entries</span>
          </div>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>measurement</th>
                  <th>your run</th>
                  <th>the published record</th>
                </tr>
              </thead>
              <tbody>
                {parsed.data.entries.map((e) => {
                  const rec = findPublished(e.key)
                  return (
                    <tr key={e.key}>
                      <td className="keycell">{e.key}</td>
                      <td>{e.text || '·'}</td>
                      <td>
                        {rec ? (
                          <span className="pub">
                            <span className="pubid">{rec.id}</span> {rec.value}
                            <span className="pubchip"> · {rec.chip}</span>
                          </span>
                        ) : (
                          <span className="nopub">no published record</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      <p className="provenance">Parsed in your browser. Nothing you paste here is sent anywhere.</p>

      <style>{`
        .benchcompare { font-family: 'IBM Plex Mono', monospace; padding: 0.875rem; }
        .benchcompare .paste-label { display: block; font-size: 0.75rem; letter-spacing: 0.06em; color: ${PANEL_MUTE}; margin-bottom: 0.4rem; }
        .benchcompare textarea {
          width: 100%; box-sizing: border-box; resize: vertical;
          background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE};
          border-radius: 2px; font-family: inherit; font-size: 0.8125rem; line-height: 1.6;
          padding: 0.5rem 0.625rem;
        }
        .benchcompare textarea::placeholder { color: #7e868d; opacity: 1; }
        .benchcompare .hint { font-size: 0.8125rem; color: #b9bec3; margin-top: 0.625rem; }
        .benchcompare .try { background: transparent; border: none; padding: 0; color: ${COPPER}; font: inherit; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
        .benchcompare .error { font-size: 0.75rem; color: ${FAIL}; margin-top: 0.625rem; }
        .benchcompare .meta { display: flex; gap: 1.25rem; font-size: 0.75rem; color: ${PANEL_MUTE}; margin-top: 0.75rem; flex-wrap: wrap; }
        .benchcompare .tablewrap { overflow-x: auto; margin-top: 0.625rem; }
        .benchcompare table { width: 100%; border-collapse: collapse; font-size: 0.75rem; min-width: 40rem; }
        .benchcompare th { text-align: left; font-weight: 500; letter-spacing: 0.04em; color: ${PANEL_MUTE}; padding: 0.4rem 0.75rem 0.4rem 0; border-bottom: 1px solid ${PANEL_RULE}; }
        .benchcompare td { padding: 0.45rem 0.75rem 0.45rem 0; border-bottom: 1px solid ${PANEL_RULE}; color: ${PANEL_INK}; vertical-align: top; line-height: 1.55; }
        .benchcompare .keycell { color: ${COPPER}; white-space: nowrap; }
        .benchcompare .pub { color: ${PANEL_INK}; }
        .benchcompare .pubid { color: ${PANEL_MUTE}; }
        .benchcompare .pubchip { color: ${PANEL_MUTE}; }
        .benchcompare .nopub { color: ${PANEL_MUTE}; }
        .benchcompare .provenance { font-size: 0.6875rem; color: ${PANEL_MUTE}; margin-top: 0.75rem; padding-top: 0.625rem; border-top: 1px solid ${PANEL_RULE}; }
      `}</style>
    </div>
  )
}
