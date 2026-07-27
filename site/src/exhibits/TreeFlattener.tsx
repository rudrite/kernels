// Flatten a params tree the way every transformation does. The rules are
// small enough to run here exactly: dicts flatten by sorted key, lists and
// tuples by position, None is an empty subtree rather than a leaf, and the
// structure comes back out untouched. Change the shape and watch what the
// treedef records, which is the thing grad has to hand you back.
import { useState } from 'react'

const COPPER = '#c88a70'
const PANEL_INK = '#d6d9dc'
const PANEL_MUTE = '#878e94'
const PANEL_RULE = '#23272c'
const PASS = '#4da37a'
const FAIL = '#c4574a'

type Node =
  | { kind: 'leaf'; name: string }
  | { kind: 'none' }
  | { kind: 'dict'; entries: [string, Node][] }
  | { kind: 'list'; items: Node[] }
  | { kind: 'tuple'; items: Node[] }

const START: Node = {
  kind: 'dict',
  entries: [
    ['dense', { kind: 'dict', entries: [['w', { kind: 'leaf', name: 'w' }], ['b', { kind: 'leaf', name: 'b' }]] }],
    ['scale', { kind: 'leaf', name: 'scale' }],
  ],
}

const VARIANTS: { label: string; node: Node; note: string }[] = [
  { label: 'a dict of dicts', node: START, note: 'the idiomatic params tree' },
  {
    label: 'the same leaves in a list',
    node: { kind: 'list', items: [{ kind: 'leaf', name: 'w' }, { kind: 'leaf', name: 'b' }, { kind: 'leaf', name: 'scale' }] },
    note: 'same leaves, different structure: tree.map across the two would refuse',
  },
  {
    label: 'a tuple instead of a list',
    node: { kind: 'tuple', items: [{ kind: 'leaf', name: 'w' }, { kind: 'leaf', name: 'b' }, { kind: 'leaf', name: 'scale' }] },
    note: 'a tuple and a list are different structures, even holding identical leaves',
  },
  {
    label: 'with a None branch',
    node: {
      kind: 'dict',
      entries: [
        ['dense', { kind: 'dict', entries: [['w', { kind: 'leaf', name: 'w' }], ['b', { kind: 'none' }]] }],
        ['scale', { kind: 'leaf', name: 'scale' }],
      ],
    },
    note: 'None is an empty subtree, so it contributes no leaf while still shaping the treedef',
  },
]

/** Leaves in the order a transformation sees them, and the treedef beside it. */
function flatten(n: Node): { leaves: string[]; treedef: string } {
  switch (n.kind) {
    case 'leaf':
      return { leaves: [n.name], treedef: '*' }
    case 'none':
      return { leaves: [], treedef: 'None' }
    case 'dict': {
      const sorted = [...n.entries].sort(([a], [b]) => (a < b ? -1 : 1))
      const parts = sorted.map(([k, v]) => {
        const f = flatten(v)
        return { k, ...f }
      })
      return {
        leaves: parts.flatMap((p) => p.leaves),
        treedef: `{${parts.map((p) => `'${p.k}': ${p.treedef}`).join(', ')}}`,
      }
    }
    case 'list':
    case 'tuple': {
      const parts = n.items.map(flatten)
      const inner = parts.map((p) => p.treedef).join(', ')
      return {
        leaves: parts.flatMap((p) => p.leaves),
        treedef: n.kind === 'list' ? `[${inner}]` : `(${inner})`,
      }
    }
  }
}

export default function TreeFlattener() {
  const [at, setAt] = useState(0)
  const [other, setOther] = useState(1)
  const v = VARIANTS[at]!
  const f = flatten(v.node)
  const g = flatten(VARIANTS[other]!.node)
  const compatible = f.treedef === g.treedef

  return (
    <div className="flattener">
      <div className="pick">
        <span className="k">the tree</span>
        <div className="row">
          {VARIANTS.map((x, i) => (
            <button key={x.label} type="button" className={i === at ? 'on' : ''} onClick={() => setAt(i)}>
              {x.label}
            </button>
          ))}
        </div>
      </div>

      <div className="out">
        <div className="col">
          <span className="k">leaves, in order</span>
          <div className="leaves">
            {f.leaves.length === 0 && <span className="empty">none</span>}
            {f.leaves.map((l, i) => (
              <span key={l} className="leaf">
                <span className="idx">{i}</span>
                {l}
              </span>
            ))}
          </div>
          <span className="note">{v.note}</span>
        </div>
        <div className="col">
          <span className="k">treedef</span>
          <code className="treedef">{f.treedef}</code>
          <span className="note">
            grad hands this exact structure back, which is why params can be any shape you like as long as the shape is
            stable
          </span>
        </div>
      </div>

      <div className="compare">
        <span className="k">tree.map against</span>
        <div className="row">
          {VARIANTS.map((x, i) => (
            <button key={x.label} type="button" className={i === other ? 'on' : ''} onClick={() => setOther(i)}>
              {x.label}
            </button>
          ))}
        </div>
        <p className={compatible ? 'ok' : 'no'}>
          {compatible
            ? 'same treedef: tree.map walks them together, leaf by leaf'
            : 'different treedefs: tree.map refuses, and the error names the first place they disagree'}
        </p>
      </div>

      <style>{`
        .flattener { font-family: 'IBM Plex Mono', monospace; }
        .k { font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase; color: ${PANEL_MUTE}; }
        .pick, .compare { display: flex; flex-direction: column; gap: 0.35rem; }
        .row { display: flex; gap: 0.375rem; flex-wrap: wrap; }
        .row button { background: transparent; color: ${PANEL_INK}; border: 1px solid ${PANEL_RULE}; border-radius: 2px; font-family: inherit; font-size: 0.75rem; padding: 0.28rem 0.6rem; cursor: pointer; }
        .row button:hover { border-color: ${COPPER}; color: ${COPPER}; }
        .row button.on { border-color: ${COPPER}; color: ${COPPER}; background: #1c1f24; }

        .out { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1.25rem; padding: 0.875rem 0; margin: 0.75rem 0; border-top: 1px solid ${PANEL_RULE}; border-bottom: 1px solid ${PANEL_RULE}; }
        .col { display: flex; flex-direction: column; gap: 0.4rem; }
        .leaves { display: flex; gap: 0.4rem; flex-wrap: wrap; }
        .leaf { display: inline-flex; align-items: baseline; gap: 0.35rem; font-size: 0.8125rem; color: ${COPPER}; border: 1px solid ${COPPER}; border-radius: 2px; padding: 0.15rem 0.5rem; }
        .leaf .idx { font-size: 0.625rem; color: ${PANEL_MUTE}; }
        .empty { font-size: 0.8125rem; color: ${PANEL_RULE}; }
        .treedef { font-size: 0.8125rem; color: ${PANEL_INK}; word-break: break-word; }
        .note { font-size: 0.6875rem; color: ${PANEL_MUTE}; line-height: 1.55; }

        .compare p { margin: 0.5rem 0 0; font-size: 0.78125rem; line-height: 1.6; }
        .compare .ok { color: ${PASS}; }
        .compare .no { color: ${FAIL}; }
      `}</style>
    </div>
  )
}
