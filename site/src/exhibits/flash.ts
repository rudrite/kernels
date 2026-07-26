// Streaming attention trace: one query block resident in VMEM, KV blocks
// streamed through a double-buffered pair while the carried (m, l, acc)
// state rescales per step. ringTrace plays the identical schedule with KV
// blocks arriving over ICI from the neighbor chip instead of local HBM:
// the algebra does not care where a block came from.
import type { TraceDoc, TraceFrame, DmaEdge } from './types'

function streamingAttentionTrace(kvBlocks: number, remote: boolean): TraceDoc {
  const kind = remote ? ('ici' as const) : ('hbm' as const)
  const tiles = [{ id: 'Q', group: 'Q', row: 0, col: 0 }]
  for (let j = 0; j < kvBlocks; j++) {
    tiles.push({ id: `K${j}`, group: 'K', row: 0, col: j })
    tiles.push({ id: `V${j}`, group: 'V', row: 0, col: j })
  }

  const slotOf = (j: number) => (j % 2 === 0 ? ['kA', 'vA'] : ['kB', 'vB'])
  const occ: Record<string, string | null> = { q: null, kA: null, vA: null, kB: null, vB: null }
  const frames: TraceFrame[] = []
  const src = remote ? 'the neighbor chip' : 'HBM'

  frames.push({
    caption: `prologue · Q pins resident in VMEM; K0, V0 stream in from ${src}`,
    reads: ['Q', 'K0', 'V0'],
    slots: { ...occ, q: null },
    compute: null,
    state: { m: '−inf', l: '0', acc: '0' },
    dma: [
      { tile: 'Q', toSlot: 'q', kind: 'hbm' },
      { tile: 'K0', toSlot: 'kA', kind },
      { tile: 'V0', toSlot: 'vA', kind },
    ],
  })

  occ['q'] = 'Q (resident)'
  for (let j = 0; j < kvBlocks; j++) {
    const [sk, sv] = slotOf(j)
    occ[sk!] = `K${j}`
    occ[sv!] = `V${j}`
    const next = j + 1
    const prefetch: DmaEdge[] =
      next < kvBlocks
        ? [
            { tile: `K${next}`, toSlot: slotOf(next)[0]!, kind },
            { tile: `V${next}`, toSlot: slotOf(next)[1]!, kind },
          ]
        : []
    frames.push({
      caption: remote
        ? `step ${j + 1} · compute on shard ${j} while shard ${next < kvBlocks ? next : '·'} rides the ring; state rescales exactly`
        : `step ${j + 1} · one pass over K${j}, V${j}: rescale (m, l, acc), never materializing scores to HBM`,
      reads: prefetch.map((p) => p.tile),
      slots: { ...occ },
      compute: { unit: 'MXU', expr: `S = Q·K${j}ᵀ, then acc` },
      state: {
        m: `max(m, rowmax S${j})`,
        l: `l·e^(m₋−m) + rowsum e^(S${j}−m)`,
        acc: `acc·e^(m₋−m) + e^(S${j}−m)·V${j}`,
      },
      dma: prefetch,
    })
  }

  frames.push({
    caption: `epilogue · O = acc / l flushes to HBM: full attention, one streaming pass, no score matrix ever stored`,
    reads: [],
    slots: { q: 'Q (resident)', kA: null, vA: null, kB: null, vB: null },
    compute: { unit: 'VPU', expr: 'O = acc / l' },
    state: { m: 'final', l: 'final', acc: 'O → HBM' },
    dma: [],
  })

  return {
    id: remote ? 'ring' : 'flash',
    title: remote ? 'ring attention · same schedule, longer arrows' : 'streaming attention, one pass',
    chip: 'schedule shape, chip-independent',
    slotIds: ['q', 'kA', 'vA', 'kB', 'vB'],
    stateIds: ['m', 'l', 'acc'],
    hbmTiles: tiles,
    remoteGroups: remote ? ['K', 'V'] : [],
    frames,
    provenance: remote
      ? 'identical frames to EX·03 with KV transport swapped from HBM DMA to remote DMA over ICI · state formulas unchanged'
      : 'frames follow the online-softmax schedule; carried state shown as the exact update formulas, never invented values',
  }
}

export const flashTrace = (kvBlocks: number): TraceDoc => streamingAttentionTrace(kvBlocks, false)
export const ringTrace = (kvBlocks: number): TraceDoc => streamingAttentionTrace(kvBlocks, true)
