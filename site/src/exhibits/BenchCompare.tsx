// A paste-in comparator for the lab notebooks' results blob. It parses
// whatever JSON the reader drops in, tolerantly, and lines each entry up
// against the published records in bench/results.json by matching on the
// kernel name. Every number in the table traces to one of those two
// sources: the pasted blob, or the published record. Nothing is invented,
// and nothing typed here leaves the browser.
import { useMemo, useState } from 'react'
import publishedRecords from '../../../bench/results.json'

const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const FAIL = '#c4574a'

interface PublishedRecord {
  id: string
  stage: number
  kernel: string
  metric: string
  value: string
  baseline: string
  chip: string
  dtype: string
  shapes: string
  date: string
  commit: string
}

const RECORDS = publishedRecords as PublishedRecord[]

interface UserRow {
  label: string
  measuredUs: number | null
  predictedUs: number | null
  ratio: number | null
}

interface ParsedBlob {
  lab: string | null
  chip: string | null
  n: number | null
  rows: UserRow[]
}

type ParseOutcome = { ok: true; data: ParsedBlob } | { ok: false; error: string }

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function asString(v: unknown): string | null {
  return typeof v === 'string' ? v : null
}

function asNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

/** Parses the notebook's results blob. Assumes non-empty, trimmed input. */
function parseBlob(text: string): ParseOutcome {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch (err) {
    const detail = err instanceof Error ? err.message : 'invalid JSON'
    return { ok: false, error: `could not parse JSON: ${detail}` }
  }

  if (!isRecord(raw)) return { ok: false, error: 'expected a JSON object with a "results" array' }
  if (!Array.isArray(raw.results)) return { ok: false, error: 'expected a "results" array on the blob' }

  const rows: UserRow[] = raw.results.map((item, i) => {
    if (!isRecord(item)) return { label: `row ${i + 1}`, measuredUs: null, predictedUs: null, ratio: null }
    const label = asString(item.op) ?? asString(item.name) ?? `row ${i + 1}`
    const measuredUs = asNumber(item.measured_us)
    const predictedUs = asNumber(item.predicted_us)
    const givenRatio = asNumber(item.ratio)
    // The ratio field is optional in the blob; if it's missing but both
    // timings are present, deriving it from the reader's own two numbers
    // is still real math, not a fabricated value.
    const ratio = givenRatio ?? (measuredUs !== null && predictedUs !== null && predictedUs !== 0 ? measuredUs / predictedUs : null)
    return { label, measuredUs, predictedUs, ratio }
  })

  return {
    ok: true,
    data: { lab: asString(raw.lab), chip: asString(raw.chip), n: asNumber(raw.n), rows },
  }
}

/** Best substring match on kernel name; ties broken by the longer, more specific kernel string. */
function findPublished(label: string): PublishedRecord | null {
  const needle = label.trim().toLowerCase()
  if (!needle) return null
  let best: PublishedRecord | null = null
  for (const rec of RECORDS) {
    const hay = rec.kernel.toLowerCase()
    if (hay.includes(needle) || needle.includes(hay)) {
      if (!best || rec.kernel.length > best.kernel.length) best = rec
    }
  }
  return best
}

/** Extracts a microsecond figure from a published value like "697.7 µs", or null if the value isn't in that unit. */
function publishedMicroseconds(rec: PublishedRecord): number | null {
  const m = rec.value.match(/^([\d.]+)\s*µs$/)
  return m ? Number(m[1]) : null
}

const fmtUs = (v: number | null) => (v !== null ? `${v.toFixed(1)} µs` : '·')
const fmtRatio = (v: number | null) => (v !== null ? v.toFixed(2) : '·')

const PLACEHOLDER = `{
  "lab": "3.2",
  "chip": "v5e",
  "n": 4096,
  "results": [
    { "op": "matmul", "measured_us": 710.2, "predicted_us": 697.7, "ratio": 1.02 }
  ]
}`

export default function BenchCompare() {
  const [text, setText] = useState('')
  const trimmed = text.trim()
  const parsed = useMemo(() => (trimmed ? parseBlob(trimmed) : null), [trimmed])

  return (
    <div className="benchcompare panel">
      <label className="paste-label" htmlFor="bench-input">
        paste the results blob printed by a lab notebook
      </label>
      <textarea
        id="bench-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={PLACEHOLDER}
        spellCheck={false}
        rows={6}
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
            <span>lab {parsed.data.lab ?? '·'}</span>
            <span>chip {parsed.data.chip ?? '·'}</span>
            <span>n {parsed.data.n ?? '·'}</span>
          </div>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>kernel</th>
                  <th>measured</th>
                  <th>ratio</th>
                  <th>published</th>
                </tr>
              </thead>
              <tbody>
                {parsed.data.rows.map((row, i) => {
                  const rec = findPublished(row.label)
                  const pubUs = rec ? publishedMicroseconds(rec) : null
                  const comparable = pubUs !== null && row.measuredUs !== null

                  return (
                    <tr key={`${row.label}-${i}`}>
                      <td>{row.label}</td>
                      <td>{fmtUs(row.measuredUs)}</td>
                      <td>{fmtRatio(row.ratio)}</td>
                      <td>
                        {rec ? (
                          <span className="pub">
                            {rec.value} · {rec.metric} · {rec.chip}
                            {comparable && (
                              <span className="pubratio">
                                {' '}
                                · yours/published {(row.measuredUs! / pubUs!).toFixed(2)}
                              </span>
                            )}
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
          border-radius: 2px; font-family: inherit; font-size: 0.8125rem; line-height: 1.6; padding: 0.625rem 0.75rem;
          padding: 0.5rem 0.625rem;
        }
        .benchcompare textarea::placeholder { color: #7e868d; opacity: 1; }
        .benchcompare .hint { font-size: 0.8125rem; color: #b9bec3; margin-top: 0.625rem; }
        .benchcompare .try { background: transparent; border: none; padding: 0; color: #c88a70; font: inherit; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
        .benchcompare .error { font-size: 0.75rem; color: ${FAIL}; margin-top: 0.625rem; }
        .benchcompare .meta { display: flex; gap: 1.25rem; font-size: 0.75rem; color: ${PANEL_MUTE}; margin-top: 0.75rem; }
        .benchcompare .tablewrap { overflow-x: auto; margin-top: 0.625rem; }
        .benchcompare table { width: 100%; border-collapse: collapse; font-size: 0.75rem; min-width: 34rem; }
        .benchcompare th { text-align: left; font-weight: 500; letter-spacing: 0.04em; color: ${PANEL_MUTE}; padding: 0.4rem 0.75rem 0.4rem 0; border-bottom: 1px solid ${PANEL_RULE}; }
        .benchcompare td { padding: 0.4rem 0.75rem 0.4rem 0; border-bottom: 1px solid ${PANEL_RULE}; color: ${PANEL_INK}; vertical-align: top; }
        .benchcompare .pub { color: ${PANEL_INK}; }
        .benchcompare .pubratio { color: ${PANEL_MUTE}; }
        .benchcompare .nopub { color: ${PANEL_MUTE}; }
        .benchcompare .provenance { font-size: 0.6875rem; color: ${PANEL_MUTE}; margin-top: 0.75rem; padding-top: 0.625rem; border-top: 1px solid ${PANEL_RULE}; }
      `}</style>
    </div>
  )
}
