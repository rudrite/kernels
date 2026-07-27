// The jit cache, made turnable. Chapter 3 says a recompile is never
// mysterious because the key is knowable, so this hands you the four parts
// of the key and lets you change them one at a time. The cache below is a
// real cache: it stores the keys you have already produced and tells you
// whether this call is a hit or a compile, exactly as the chapter describes.
import { useState } from 'react'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'

type Dtype = 'float32' | 'bfloat16' | 'int32'
type Structure = 'a dict of two arrays' | 'a tuple of two arrays' | 'one array'

interface Call {
  fn: string
  structure: Structure
  shape: string
  dtype: Dtype
  k: number
}

const START: Call = { fn: 'step', structure: 'a dict of two arrays', shape: '(8, 128)', dtype: 'float32', k: 3 }

/** The key jit would build for this call: everything that is not a value. */
const keyOf = (c: Call) => `${c.fn} · ${c.structure} · ${c.shape} · ${c.dtype} · k=${c.k}`

const SHAPES = ['(8, 128)', '(16, 128)', '(8, 256)']
const DTYPES: Dtype[] = ['float32', 'bfloat16', 'int32']
const STRUCTURES: Structure[] = ['a dict of two arrays', 'a tuple of two arrays', 'one array']

export default function CacheKey() {
  const [call, setCall] = useState<Call>(START)
  const [seen, setSeen] = useState<string[]>([keyOf(START)])
  const [last, setLast] = useState<{ key: string; hit: boolean } | null>(null)

  const key = keyOf(call)
  const wouldHit = seen.includes(key)

  const run = () => {
    setLast({ key, hit: wouldHit })
    if (!wouldHit) setSeen((s) => [...s, key])
  }

  const reset = () => {
    setSeen([])
    setLast(null)
  }

  const set = <K extends keyof Call>(k: K, v: Call[K]) => setCall((c) => ({ ...c, [k]: v }))

  return (
    <div className="cachekey">
      <div className="grid">
        <div className="controls">
          <label>
            <span className="k">the function object</span>
            <div className="row">
              <button type="button" className={call.fn === 'step' ? 'on' : ''} onClick={() => set('fn', 'step')}>
                step
              </button>
              <button
                type="button"
                className={call.fn.startsWith('lambda') ? 'on' : ''}
                onClick={() => set('fn', `lambda #${Math.floor(Math.random() * 900 + 100)}`)}
              >
                a fresh lambda
              </button>
            </div>
            <span className="note">a new function object is a new key, which is the churn nobody sees</span>
          </label>

          <label>
            <span className="k">pytree structure of the arguments</span>
            <div className="row">
              {STRUCTURES.map((s) => (
                <button key={s} type="button" className={call.structure === s ? 'on' : ''} onClick={() => set('structure', s)}>
                  {s.replace('a ', '').replace(' of two arrays', '')}
                </button>
              ))}
            </div>
          </label>

          <label>
            <span className="k">leaf shape</span>
            <div className="row">
              {SHAPES.map((s) => (
                <button key={s} type="button" className={call.shape === s ? 'on' : ''} onClick={() => set('shape', s)}>
                  {s}
                </button>
              ))}
            </div>
          </label>

          <label>
            <span className="k">leaf dtype</span>
            <div className="row">
              {DTYPES.map((d) => (
                <button key={d} type="button" className={call.dtype === d ? 'on' : ''} onClick={() => set('dtype', d)}>
                  {d}
                </button>
              ))}
            </div>
          </label>

          <label>
            <span className="k">static argument k</span>
            <div className="row">
              {[3, 5, 8].map((n) => (
                <button key={n} type="button" className={call.k === n ? 'on' : ''} onClick={() => set('k', n)}>
                  k={n}
                </button>
              ))}
            </div>
            <span className="note">static means baked into the key, so every value compiles its own executable</span>
          </label>
        </div>

        <div className="readout">
          <span className="k">the key this call builds</span>
          <code className="key">{key}</code>
          <div className="verdict">
            <span className={wouldHit ? 'hit' : 'miss'}>{wouldHit ? 'cache hit' : 'trace and compile'}</span>
            <button type="button" className="go" onClick={run}>
              call it →
            </button>
          </div>
          {last && (
            <p className="last" aria-live="polite">
              last call: {last.hit ? 'served from the cache' : 'compiled, and the key was added'}
            </p>
          )}
          <span className="k cached">executables held ({seen.length})</span>
          <ul className="seen">
            {seen.length === 0 && <li className="empty">none yet</li>}
            {seen.map((s) => (
              <li key={s} className={s === key ? 'current' : ''}>
                {s}
              </li>
            ))}
          </ul>
          <button type="button" className="reset" onClick={reset}>
            clear the cache
          </button>
        </div>
      </div>

      <style>{`
        .cachekey { font-family: 'IBM Plex Mono', monospace; }
        .grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.5rem; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

        .controls { display: flex; flex-direction: column; gap: 0.875rem; }
        label { display: flex; flex-direction: column; gap: 0.35rem; }
        .k { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .note { font-size: 0.6875rem; color: ${PANEL_MUTE}; line-height: 1.5; }
        .row { display: flex; gap: 0.375rem; flex-wrap: wrap; }
        .row button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.3rem 0.6rem; cursor: pointer; }
        .row button:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .row button.on { border-color: ${COPPER}; color: ${COPPER}; background: #1c1f24; }

        .readout { display: flex; flex-direction: column; gap: 0.5rem; }
        .key { display: block; font-size: 0.78125rem; color: ${COPPER}; line-height: 1.6; word-break: break-word; }
        .verdict { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding-top: 0.25rem; }
        .verdict .hit { color: ${PASS}; font-size: 0.875rem; }
        .verdict .miss { color: ${COPPER}; font-size: 0.875rem; }
        .go { background: transparent; color: ${PANEL_INK}; border: 1px solid ${COPPER}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.35rem 0.75rem; cursor: pointer; }
        .go:hover { color: ${COPPER}; }
        .last { font-size: 0.6875rem; color: ${PANEL_MUTE}; margin: 0; }
        .cached { margin-top: 0.375rem; }
        .seen { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; }
        .seen li { font-size: 0.6875rem; color: ${PANEL_MUTE}; line-height: 1.5; word-break: break-word; }
        .seen li.current { color: ${PASS}; }
        .seen li.empty { color: ${PANEL_RULE}; }
        .reset { align-self: flex-start; margin-top: 0.375rem; background: transparent; color: ${PANEL_MUTE}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.6875rem; padding: 0.25rem 0.6rem; cursor: pointer; }
        .reset:hover { border-color: ${COPPER}; color: ${COPPER}; }
      `}</style>
    </div>
  )
}
