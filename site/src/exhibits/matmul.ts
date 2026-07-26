// Double-buffered tiled matmul trace: one output block of C = A·B, streaming
// K-blocks of A and B through two VMEM slot pairs while the MXU consumes the
// pair loaded on the previous grid step. This is the schedule Pallas's grid
// pipeline emits for the same BlockSpecs.
import type { TraceDoc, TraceFrame } from './types'

export function matmulTrace(kBlocks: number, pipelined = true): TraceDoc {
  if (!pipelined) return naiveMatmulTrace(kBlocks)
  return pipelinedMatmulTrace(kBlocks)
}

// The pipeline turned off: load, then compute, strictly alternating. The MXU
// idles during every transfer; the contrast with the pipelined schedule is
// the entire argument for double buffering.
function naiveMatmulTrace(kBlocks: number): TraceDoc {
  const tiles = []
  for (let k = 0; k < kBlocks; k++) {
    tiles.push({ id: `A${k}`, group: 'A', row: 0, col: k })
    tiles.push({ id: `B${k}`, group: 'B', row: 0, col: k })
  }
  const frames: TraceFrame[] = []
  const occ: Record<string, string | null> = { a: null, b: null, acc: 'C (zeros)' }
  for (let k = 0; k < kBlocks; k++) {
    frames.push({
      caption: `load ${k + 1} · DMA A${k}, B${k} into VMEM; the MXU sits idle the whole transfer`,
      reads: [`A${k}`, `B${k}`],
      slots: { ...occ },
      compute: null,
      dma: [
        { tile: `A${k}`, toSlot: 'a', kind: 'hbm' },
        { tile: `B${k}`, toSlot: 'b', kind: 'hbm' },
      ],
    })
    occ['a'] = `A${k}`
    occ['b'] = `B${k}`
    occ['acc'] = `C += A·B (k ≤ ${k})`
    frames.push({
      caption: `compute ${k + 1} · MXU consumes A${k}·B${k}; the DMA engines sit idle the whole time`,
      reads: [],
      slots: { ...occ },
      compute: { unit: 'MXU', expr: `acc += A${k} · B${k}` },
      dma: [],
    })
  }
  frames.push({
    caption: 'epilogue · every step paid full transfer time plus full compute time, serially',
    reads: [],
    slots: { a: null, b: null, acc: 'C → HBM' },
    compute: null,
    dma: [],
  })
  return {
    id: 'matmul-naive',
    title: 'the same matmul with the pipeline off',
    chip: 'schedule shape, chip-independent',
    slotIds: ['a', 'b', 'acc'],
    stateIds: [],
    hbmTiles: tiles,
    remoteGroups: [],
    frames,
    provenance: 'the depth-1 schedule: compare frame count and MXU idle time against EX·02 above · no invented values',
  }
}

function pipelinedMatmulTrace(kBlocks: number): TraceDoc {
  const tiles = []
  for (let k = 0; k < kBlocks; k++) {
    tiles.push({ id: `A${k}`, group: 'A', row: 0, col: k })
    tiles.push({ id: `B${k}`, group: 'B', row: 0, col: k })
  }

  const frames: TraceFrame[] = []
  const slotOf = (k: number) => (k % 2 === 0 ? ['a0', 'b0'] : ['a1', 'b1'])
  const occupancy: Record<string, string | null> = { a0: null, b0: null, a1: null, b1: null, acc: 'C (zeros)' }

  frames.push({
    caption: `grid step 0 · DMA A0, B0 into the first slot pair; the MXU has no data yet and waits`,
    reads: ['A0', 'B0'],
    slots: { ...occupancy },
    compute: null,
    dma: [
      { tile: 'A0', toSlot: 'a0', kind: 'hbm' },
      { tile: 'B0', toSlot: 'b0', kind: 'hbm' },
    ],
  })

  for (let k = 0; k < kBlocks; k++) {
    const [sa, sb] = slotOf(k)
    occupancy[sa!] = `A${k}`
    occupancy[sb!] = `B${k}`
    const next = k + 1
    const prefetch =
      next < kBlocks
        ? [
            { tile: `A${next}`, toSlot: slotOf(next)[0]!, kind: 'hbm' as const },
            { tile: `B${next}`, toSlot: slotOf(next)[1]!, kind: 'hbm' as const },
          ]
        : []
    occupancy['acc'] = `C += A·B (k ≤ ${k})`
    frames.push({
      caption:
        next < kBlocks
          ? `grid step ${k + 1} · MXU consumes A${k}·B${k} while DMA prefetches A${next}, B${next} into the other pair`
          : `grid step ${k + 1} · MXU consumes the final pair A${k}·B${k}; nothing left to prefetch`,
      reads: prefetch.map((p) => p.tile),
      slots: { ...occupancy },
      compute: { unit: 'MXU', expr: `acc += A${k} · B${k}` },
      dma: prefetch,
    })
  }

  frames.push({
    caption: `epilogue · the accumulator flushes to HBM as the C output block; slots retire`,
    reads: [],
    slots: { a0: null, b0: null, a1: null, b1: null, acc: 'C → HBM' },
    compute: null,
    dma: [],
  })

  return {
    id: 'matmul',
    title: 'tiled matmul, double-buffered',
    chip: 'schedule shape, chip-independent',
    slotIds: ['a0', 'b0', 'a1', 'b1', 'acc'],
    stateIds: [],
    hbmTiles: tiles,
    remoteGroups: [],
    frames,
    provenance: 'frames computed from the BlockSpec schedule (K-blocked, depth-2 pipeline) · no invented values; compute is shown as expressions',
  }
}
