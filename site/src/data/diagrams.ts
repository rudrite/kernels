// Diagram specs for BlockDiagram: copper is compute, steel is memory,
// dashed is control or absence. Every number on a diagram is a chip
// constant or a labeled spec-sheet figure; captions state provenance.

export interface DiagramBox {
  k: 'box'
  x: number
  y: number
  w: number
  h: number
  t: string
  s?: string
  tone?: 'copper' | 'steel' | 'ink' | 'mute'
  dash?: boolean
}
export interface DiagramLine {
  k: 'line'
  x1: number
  y1: number
  x2: number
  y2: number
  t?: string
  lx?: number
  ly?: number
  tone?: 'copper' | 'steel' | 'ink' | 'mute'
  w?: number
  dash?: boolean
  a?: 'end' | 'both' | 'none'
}
export interface DiagramText {
  k: 'text'
  x: number
  y: number
  t: string
  tone?: 'copper' | 'steel' | 'ink' | 'mute'
  size?: number
  anchor?: 'start' | 'middle' | 'end'
}
export type DiagramEl = DiagramBox | DiagramLine | DiagramText

export interface DiagramSpec {
  w: number
  h: number
  alt: string
  caption: string
  els: DiagramEl[]
}

export const DIAGRAMS: Record<string, DiagramSpec> = {
  'tpu-chip': {
    w: 720,
    h: 340,
    alt: 'Block diagram of a TPU chip: HBM feeding VMEM over the memory bus, VMEM feeding the MXU and VPU, with SMEM and the scalar core controlling DMA, and ICI links leaving the chip',
    caption: 'one TPU chip, v5e numbers from the chip table: everything the roofline needs is on this picture',
    els: [
      { k: 'box', x: 24, y: 60, w: 120, h: 200, t: 'HBM', s: '16 GB\nfar, wide', tone: 'steel' },
      { k: 'line', x1: 144, y1: 160, x2: 240, y2: 160, w: 8, tone: 'steel', a: 'both', t: '8.2e11 B/s', ly: 148 },
      { k: 'box', x: 240, y: 84, w: 132, h: 152, t: 'VMEM', s: '~128 MiB\nsoftware managed\nyour schedule stages it', tone: 'steel' },
      { k: 'line', x1: 372, y1: 122, x2: 458, y2: 122, w: 3, tone: 'copper', a: 'end' },
      { k: 'line', x1: 372, y1: 198, x2: 458, y2: 198, w: 3, tone: 'copper', a: 'end' },
      { k: 'box', x: 458, y: 70, w: 226, h: 100, t: 'MXU', s: '128x128 systolic array\nmatmuls only · 1.97e14 bf16 FLOP/s', tone: 'copper' },
      { k: 'box', x: 458, y: 190, w: 226, h: 100, t: 'VPU', s: '(8, 128) lanes\neverything elementwise', tone: 'copper' },
      { k: 'box', x: 240, y: 262, w: 132, h: 44, t: 'SMEM + scalar core', s: 'control flow, DMA issue', tone: 'mute' },
      { k: 'line', x1: 306, y1: 262, x2: 306, y2: 236, dash: true, a: 'end', tone: 'mute' },
      { k: 'line', x1: 240, y1: 284, x2: 192, y2: 284, dash: true, tone: 'mute' },
      { k: 'line', x1: 192, y1: 284, x2: 192, y2: 168, dash: true, a: 'end', tone: 'mute', t: 'DMA descriptors', lx: 178, ly: 230 },
      { k: 'box', x: 24, y: 284, w: 120, h: 36, t: 'ICI ×4', s: '4.5e10 B/s each way', tone: 'mute' },
      { k: 'line', x1: 24, y1: 302, x2: 4, y2: 302, a: 'end', tone: 'mute' },
      { k: 'text', x: 700, y: 330, t: 'copper = compute · steel = memory · dashed = control', anchor: 'end', size: 9.5 },
    ],
  },

  'gpu-sm': {
    w: 720,
    h: 360,
    alt: 'Block diagram of an H100-class GPU: HBM3 feeding a large L2, feeding one of 132 streaming multiprocessors, shown with warp schedulers, tensor cores, shared memory, and the register file',
    caption: 'an H100-class GPU by the public spec sheet (numbers approximate): one SM shown, and there are 132 of them',
    els: [
      { k: 'box', x: 24, y: 70, w: 110, h: 220, t: 'HBM3', s: '~3.35e12 B/s', tone: 'steel' },
      { k: 'line', x1: 134, y1: 180, x2: 210, y2: 180, w: 8, tone: 'steel', a: 'both' },
      { k: 'box', x: 210, y: 100, w: 88, h: 160, t: 'L2', s: '~50 MB\nshared by all SMs', tone: 'steel' },
      { k: 'line', x1: 298, y1: 180, x2: 356, y2: 180, w: 4, tone: 'steel', a: 'both' },
      { k: 'box', x: 380, y: 44, w: 300, h: 284, t: '', tone: 'mute', dash: true },
      { k: 'box', x: 368, y: 56, w: 300, h: 284, t: '', tone: 'mute', dash: true },
      { k: 'box', x: 356, y: 68, w: 300, h: 284, t: '', tone: 'ink' },
      { k: 'text', x: 370, y: 90, t: 'one SM · × 132 on the chip', tone: 'ink', size: 12 },
      { k: 'box', x: 372, y: 104, w: 268, h: 44, t: 'warp schedulers', s: 'pick a ready warp every cycle', tone: 'mute' },
      { k: 'box', x: 372, y: 162, w: 126, h: 64, t: 'tensor cores', s: '×4 · the MXU analogue', tone: 'copper' },
      { k: 'box', x: 514, y: 162, w: 126, h: 64, t: 'shared memory', s: 'up to ~228 KB\nthe VMEM analogue, tiny', tone: 'steel' },
      { k: 'box', x: 372, y: 240, w: 268, h: 44, t: 'register file', s: 'pressure here caps resident warps', tone: 'mute' },
      { k: 'text', x: 372, y: 316, t: 'occupancy = how many warps fit = the latency-hiding budget', tone: 'copper', size: 10 },
      { k: 'text', x: 700, y: 350, t: 'ridge ~295 F/B vs 240 (v5e) and 575 (v6e)', anchor: 'end', size: 9.5 },
    ],
  },

  'latency-two-ways': {
    w: 720,
    h: 300,
    alt: 'Two timelines: the TPU overlapping DMA transfers behind compute steps in one sequential pipeline, and the GPU switching among four resident warps whenever one stalls on memory',
    caption: 'the same goal, two mechanisms: hide memory latency behind compute, or behind other warps',
    els: [
      { k: 'text', x: 24, y: 34, t: 'TPU · one core, DMAs pipelined behind compute', tone: 'ink', size: 12 },
      { k: 'text', x: 24, y: 62, t: 'compute', size: 10, anchor: 'start' },
      { k: 'box', x: 100, y: 46, w: 170, h: 26, t: 'step k', tone: 'copper' },
      { k: 'box', x: 274, y: 46, w: 170, h: 26, t: 'step k+1', tone: 'copper' },
      { k: 'box', x: 448, y: 46, w: 170, h: 26, t: 'step k+2', tone: 'copper' },
      { k: 'text', x: 24, y: 100, t: 'DMA', size: 10 },
      { k: 'box', x: 100, y: 84, w: 170, h: 26, t: 'blocks for k+1', tone: 'steel' },
      { k: 'box', x: 274, y: 84, w: 170, h: 26, t: 'blocks for k+2', tone: 'steel' },
      { k: 'box', x: 448, y: 84, w: 170, h: 26, t: 'blocks for k+3', tone: 'steel' },
      { k: 'line', x1: 100, y1: 126, x2: 618, y2: 126, a: 'end', tone: 'mute', t: 'time' },

      { k: 'text', x: 24, y: 168, t: 'GPU · many resident warps, scheduler switches on stall', tone: 'ink', size: 12 },
      { k: 'text', x: 24, y: 196, t: 'warp 0', size: 10 },
      { k: 'box', x: 100, y: 180, w: 120, h: 22, t: 'exec', tone: 'copper' },
      { k: 'box', x: 220, y: 180, w: 150, h: 22, t: 'stalled on memory', tone: 'mute', dash: true },
      { k: 'box', x: 370, y: 180, w: 120, h: 22, t: 'exec', tone: 'copper' },
      { k: 'text', x: 24, y: 228, t: 'warp 1', size: 10 },
      { k: 'box', x: 100, y: 212, w: 120, h: 22, t: 'stalled', tone: 'mute', dash: true },
      { k: 'box', x: 220, y: 212, w: 150, h: 22, t: 'exec', tone: 'copper' },
      { k: 'box', x: 370, y: 212, w: 120, h: 22, t: 'stalled', tone: 'mute', dash: true },
      { k: 'text', x: 24, y: 260, t: 'warp 2', size: 10 },
      { k: 'box', x: 100, y: 244, w: 250, h: 22, t: 'stalled on memory', tone: 'mute', dash: true },
      { k: 'box', x: 490, y: 212, w: 128, h: 22, t: 'exec', tone: 'copper' },
      { k: 'box', x: 350, y: 244, w: 268, h: 22, t: 'exec', tone: 'copper' },
      { k: 'line', x1: 100, y1: 284, x2: 618, y2: 284, a: 'end', tone: 'mute', t: 'time' },
    ],
  },

  'memory-spaces': {
    w: 720,
    h: 280,
    alt: 'The four places a Pallas ref can live: HBM outside the kernel, VMEM blocks staged by BlockSpecs, SMEM scalars, scratch owned by the kernel, and an ANY ref waiting for a manual DMA',
    caption: 'where refs live during one pallas_call: staged blocks, scalars, kernel-owned scratch, and the ANY ref you move yourself',
    els: [
      { k: 'box', x: 24, y: 40, w: 150, h: 210, t: 'HBM', s: 'full arrays', tone: 'steel' },
      { k: 'box', x: 246, y: 28, w: 450, h: 234, t: '', tone: 'mute', dash: true },
      { k: 'text', x: 260, y: 50, t: 'one kernel invocation · VMEM', tone: 'ink', size: 12 },
      { k: 'box', x: 262, y: 66, w: 128, h: 58, t: 'block in', s: 'staged by BlockSpec', tone: 'steel' },
      { k: 'box', x: 262, y: 136, w: 128, h: 58, t: 'block out', s: 'written back after', tone: 'steel' },
      { k: 'line', x1: 174, y1: 96, x2: 262, y2: 96, w: 4, tone: 'steel', a: 'end', t: 'pipeline DMA' },
      { k: 'line', x1: 262, y1: 166, x2: 174, y2: 166, w: 4, tone: 'steel', a: 'end' },
      { k: 'box', x: 412, y: 66, w: 128, h: 58, t: 'scratch', s: 'scratch_shapes\nf32 accumulator lives here', tone: 'copper' },
      { k: 'box', x: 412, y: 136, w: 128, h: 58, t: 'SMEM', s: 'lengths, flags, indices', tone: 'mute' },
      { k: 'box', x: 562, y: 66, w: 118, h: 128, t: 'ANY ref', s: 'unplaced until\nyour make_async_copy', tone: 'mute', dash: true },
      { k: 'line', x1: 99, y1: 220, x2: 562, y2: 152, dash: true, a: 'end', tone: 'copper', t: 'manual DMA + semaphore', lx: 330, ly: 226 },
      { k: 'text', x: 260, y: 246, t: 'persists across grid steps: scratch · re-staged per step: blocks', size: 9.5 },
    ],
  },

  'ici-ring': {
    w: 720,
    h: 300,
    alt: 'Eight TPU chips connected in a bidirectional ring over ICI links, with one hop labeled with its bandwidth and a dashed chord showing there is no direct path across the ring',
    caption: 'a v5e ring: every collective is hops on this picture; an all-gather is N−1 of them, bidirectional halves the time',
    els: [
      { k: 'box', x: 320, y: 30, w: 80, h: 40, t: 'chip 0', tone: 'copper' },
      { k: 'box', x: 490, y: 62, w: 80, h: 40, t: 'chip 1', tone: 'copper' },
      { k: 'box', x: 560, y: 148, w: 80, h: 40, t: 'chip 2', tone: 'copper' },
      { k: 'box', x: 490, y: 234, w: 80, h: 40, t: 'chip 3', tone: 'copper' },
      { k: 'box', x: 320, y: 258, w: 80, h: 40, t: 'chip 4', tone: 'copper' },
      { k: 'box', x: 150, y: 234, w: 80, h: 40, t: 'chip 5', tone: 'copper' },
      { k: 'box', x: 84, y: 148, w: 80, h: 40, t: 'chip 6', tone: 'copper' },
      { k: 'box', x: 150, y: 62, w: 80, h: 40, t: 'chip 7', tone: 'copper' },
      { k: 'line', x1: 400, y1: 56, x2: 490, y2: 76, a: 'both', tone: 'steel', w: 2.5, t: '4.5e10 B/s each way', lx: 470, ly: 44 },
      { k: 'line', x1: 552, y1: 102, x2: 578, y2: 148, a: 'both', tone: 'steel', w: 2.5 },
      { k: 'line', x1: 578, y1: 188, x2: 552, y2: 234, a: 'both', tone: 'steel', w: 2.5 },
      { k: 'line', x1: 490, y1: 262, x2: 400, y2: 282, a: 'both', tone: 'steel', w: 2.5 },
      { k: 'line', x1: 320, y1: 282, x2: 230, y2: 262, a: 'both', tone: 'steel', w: 2.5 },
      { k: 'line', x1: 168, y1: 234, x2: 142, y2: 188, a: 'both', tone: 'steel', w: 2.5 },
      { k: 'line', x1: 142, y1: 148, x2: 168, y2: 102, a: 'both', tone: 'steel', w: 2.5 },
      { k: 'line', x1: 230, y1: 56, x2: 320, y2: 44, a: 'both', tone: 'steel', w: 2.5 },
      { k: 'line', x1: 230, y1: 96, x2: 490, y2: 240, dash: true, tone: 'mute', t: 'no direct path: hops only', lx: 360, ly: 178 },
    ],
  },

  'blockspec-mapping': {
    w: 720,
    h: 300,
    alt: 'A 4 by 4 blocked array with one block highlighted, the index map returning block coordinates, and the block staged into VMEM with its element range',
    caption: 'the BlockSpec contract: the index map returns block coordinates, Pallas multiplies by the block shape to find elements',
    els: [
      { k: 'text', x: 24, y: 34, t: 'array in HBM · (512, 512) · block_shape (128, 128)', tone: 'ink', size: 12 },
      { k: 'box', x: 24, y: 50, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 79, y: 50, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 134, y: 50, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 189, y: 50, w: 55, h: 42, t: '', tone: 'mute' },
      { k: 'box', x: 24, y: 92, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 79, y: 92, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 134, y: 92, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 189, y: 92, w: 55, h: 42, t: '', tone: 'mute' },
      { k: 'box', x: 24, y: 134, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 79, y: 134, w: 55, h: 42, t: '(2,1)', tone: 'copper' }, { k: 'box', x: 134, y: 134, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 189, y: 134, w: 55, h: 42, t: '', tone: 'mute' },
      { k: 'box', x: 24, y: 176, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 79, y: 176, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 134, y: 176, w: 55, h: 42, t: '', tone: 'mute' }, { k: 'box', x: 189, y: 176, w: 55, h: 42, t: '', tone: 'mute' },
      { k: 'text', x: 24, y: 244, t: 'block coordinates, not element offsets', size: 9.5 },
      { k: 'box', x: 296, y: 96, w: 180, h: 110, t: 'index_map(i, j)', s: 'grid step (2, 1)\nreturns (2, 1)\n× block_shape (128, 128)', tone: 'ink' },
      { k: 'line', x1: 134, y1: 155, x2: 296, y2: 155, a: 'end', tone: 'copper' },
      { k: 'line', x1: 476, y1: 151, x2: 540, y2: 151, a: 'end', tone: 'steel', w: 4, t: 'pipeline DMA' },
      { k: 'box', x: 540, y: 108, w: 156, h: 86, t: 'block in VMEM', s: 'elements\n[256:384, 128:256]', tone: 'steel' },
    ],
  },

  'jaxpr-anatomy': {
    w: 720,
    h: 310,
    alt: 'The four zones of a jaxpr: invars at the top, constvars from closed-over values, the equations in single-assignment form, and the outvars at the bottom',
    caption: 'the anatomy of every jaxpr you will ever read: four zones, always in this order',
    els: [
      { k: 'box', x: 150, y: 30, w: 420, h: 40, t: 'lambda ; a:bf16[32,64]  b:bf16[64,64]', tone: 'steel' },
      { k: 'text', x: 590, y: 54, t: 'invars: the arguments', size: 10 },
      { k: 'box', x: 150, y: 78, w: 420, h: 36, t: 'c:bf16[64]', tone: 'steel', dash: true },
      { k: 'text', x: 590, y: 100, t: 'constvars: closed over', size: 10 },
      { k: 'box', x: 150, y: 126, w: 420, h: 34, t: 'd = dot_general[...] a b', tone: 'copper' },
      { k: 'box', x: 150, y: 164, w: 420, h: 34, t: 'e = add d c', tone: 'copper' },
      { k: 'box', x: 150, y: 202, w: 420, h: 34, t: 'f = tanh e', tone: 'copper' },
      { k: 'text', x: 590, y: 172, t: 'eqns: one primitive', size: 10 },
      { k: 'text', x: 590, y: 184, t: 'each, SSA form', size: 10 },
      { k: 'box', x: 150, y: 248, w: 420, h: 36, t: 'in (f,)', tone: 'steel' },
      { k: 'text', x: 590, y: 270, t: 'outvars: the results', size: 10 },
      { k: 'line', x1: 130, y1: 50, x2: 130, y2: 266, a: 'end', tone: 'mute', dash: true, t: 'read top to bottom', lx: 60, ly: 160 },
    ],
  },

  'lowering-ladder': {
    w: 720,
    h: 280,
    alt: 'The lowering ladder: JAX python through jaxpr, PyTorch through torchax, both converging on StableHLO, which feeds the XLA backend and machine code',
    caption: 'the ladder, and the convergence point: by StableHLO the framework identity is gone',
    els: [
      { k: 'box', x: 24, y: 40, w: 150, h: 52, t: 'JAX python', s: 'jnp on arrays', tone: 'ink' },
      { k: 'box', x: 24, y: 180, w: 150, h: 52, t: 'PyTorch', s: 'ATen ops', tone: 'ink' },
      { k: 'box', x: 226, y: 40, w: 130, h: 52, t: 'jaxpr', s: 'traced, SSA', tone: 'copper' },
      { k: 'line', x1: 174, y1: 66, x2: 226, y2: 66, a: 'end', tone: 'mute', t: 'trace' },
      { k: 'box', x: 408, y: 106, w: 150, h: 60, t: 'StableHLO', s: 'versioned · portable\nthe shared vocabulary', tone: 'steel' },
      { k: 'line', x1: 356, y1: 66, x2: 408, y2: 122, a: 'end', tone: 'mute', t: 'lower' },
      { k: 'line', x1: 174, y1: 206, x2: 408, y2: 152, a: 'end', tone: 'mute', t: 'torchax / TorchTPU', lx: 280, ly: 200 },
      { k: 'box', x: 606, y: 42, w: 90, h: 52, t: 'XLA:TPU', tone: 'ink' },
      { k: 'box', x: 606, y: 178, w: 90, h: 52, t: 'XLA:GPU', tone: 'ink' },
      { k: 'line', x1: 558, y1: 122, x2: 606, y2: 70, a: 'end', tone: 'mute' },
      { k: 'line', x1: 558, y1: 152, x2: 606, y2: 202, a: 'end', tone: 'mute' },
    ],
  },

  'fusion-before-after': {
    w: 720,
    h: 330,
    alt: 'The same softmax-between-matmuls program before and after the fusion pass: five elementwise and reduction ops collapse into one fusion, while the two dots stay outside it',
    caption: 'what the fusion pass does to naive attention: the chain fuses, the dots do not, and the score matrix between them still spills',
    els: [
      { k: 'text', x: 24, y: 34, t: 'received', tone: 'ink', size: 12 },
      { k: 'box', x: 24, y: 48, w: 120, h: 34, t: 'dot_general', tone: 'copper' },
      { k: 'box', x: 24, y: 92, w: 120, h: 28, t: 'reduce max', tone: 'mute' },
      { k: 'box', x: 24, y: 124, w: 120, h: 28, t: 'subtract', tone: 'mute' },
      { k: 'box', x: 24, y: 156, w: 120, h: 28, t: 'exponential', tone: 'mute' },
      { k: 'box', x: 24, y: 188, w: 120, h: 28, t: 'reduce sum', tone: 'mute' },
      { k: 'box', x: 24, y: 220, w: 120, h: 28, t: 'divide', tone: 'mute' },
      { k: 'box', x: 24, y: 262, w: 120, h: 34, t: 'dot_general', tone: 'copper' },
      { k: 'line', x1: 84, y1: 82, x2: 84, y2: 262, a: 'end', tone: 'mute' },
      { k: 'line', x1: 220, y1: 170, x2: 320, y2: 170, a: 'end', tone: 'copper', w: 2.5, t: 'fusion pass' },
      { k: 'text', x: 380, y: 34, t: 'decided', tone: 'ink', size: 12 },
      { k: 'box', x: 380, y: 48, w: 140, h: 36, t: 'dot_general', tone: 'copper' },
      { k: 'box', x: 360, y: 100, w: 320, h: 150, t: '', tone: 'copper', dash: true },
      { k: 'text', x: 376, y: 122, t: 'one fusion (kLoop)', tone: 'copper', size: 11 },
      { k: 'box', x: 376, y: 136, w: 90, h: 26, t: 'max', tone: 'mute' },
      { k: 'box', x: 376, y: 168, w: 90, h: 26, t: 'sub', tone: 'mute' },
      { k: 'box', x: 376, y: 200, w: 90, h: 26, t: 'exp', tone: 'mute' },
      { k: 'box', x: 486, y: 136, w: 90, h: 26, t: 'sum', tone: 'mute' },
      { k: 'box', x: 486, y: 168, w: 90, h: 26, t: 'div', tone: 'mute' },
      { k: 'text', x: 486, y: 218, t: 'intermediates never', size: 9.5 },
      { k: 'text', x: 486, y: 230, t: 'touch HBM', size: 9.5 },
      { k: 'box', x: 380, y: 266, w: 140, h: 36, t: 'dot_general', tone: 'copper' },
      { k: 'line', x1: 450, y1: 84, x2: 450, y2: 100, a: 'end', tone: 'steel', w: 3, t: 'S spills here', lx: 560, ly: 96 },
      { k: 'line', x1: 450, y1: 250, x2: 450, y2: 266, a: 'end', tone: 'steel', w: 3 },
    ],
  },

  'mosaic-boundary': {
    w: 720,
    h: 240,
    alt: 'The lowering path from pallas_call through the kernel jaxpr and the Mosaic module, then a frontier line, then LLO inside libtpu marked closed',
    caption: 'the boundary, drawn exactly: everything left of the line prints with debug=True; right of it is closed inside libtpu',
    els: [
      { k: 'box', x: 24, y: 80, w: 130, h: 60, t: 'pallas_call', s: 'your kernel body', tone: 'copper' },
      { k: 'line', x1: 154, y1: 110, x2: 200, y2: 110, a: 'end', tone: 'mute' },
      { k: 'box', x: 200, y: 80, w: 130, h: 60, t: 'kernel jaxpr', s: 'traced over refs', tone: 'copper' },
      { k: 'line', x1: 330, y1: 110, x2: 376, y2: 110, a: 'end', tone: 'mute' },
      { k: 'box', x: 376, y: 80, w: 150, h: 60, t: 'Mosaic module', s: 'tpu dialect · open\nin jaxlib/mosaic', tone: 'steel' },
      { k: 'text', x: 300, y: 170, t: 'debug=True prints both', tone: 'copper', size: 10, anchor: 'middle' },
      { k: 'line', x1: 200, y1: 152, x2: 380, y2: 152, tone: 'mute', dash: true },
      { k: 'line', x1: 556, y1: 30, x2: 556, y2: 210, tone: 'copper', dash: true, t: 'the last readable layer', lx: 556, ly: 24 },
      { k: 'line', x1: 526, y1: 110, x2: 572, y2: 110, a: 'end', tone: 'mute' },
      { k: 'box', x: 572, y: 80, w: 124, h: 60, t: 'LLO', s: 'libtpu · closed', tone: 'mute', dash: true },
    ],
  },

  'tracing': {
    w: 720,
    h: 260,
    alt: 'A python function with a closed-over array runs once with tracers and produces a jaxpr whose equations record the primitives and whose consts carry the closed-over value',
    caption: 'what jit actually does: run your python once with tracers, keep the recording',
    els: [
      { k: 'box', x: 24, y: 60, w: 170, h: 70, t: 'your python fn', s: 'control flow, prints,\njnp on arrays', tone: 'ink' },
      { k: 'box', x: 24, y: 150, w: 170, h: 44, t: 'closure: W', s: 'a numpy array outside', tone: 'mute', dash: true },
      { k: 'line', x1: 194, y1: 112, x2: 300, y2: 112, a: 'end', tone: 'copper', w: 2.5, t: 'runs ONCE, with tracers' },
      { k: 'box', x: 300, y: 82, w: 130, h: 60, t: 'trace', s: 'records every\nprimitive it meets', tone: 'copper' },
      { k: 'line', x1: 430, y1: 112, x2: 512, y2: 112, a: 'end', tone: 'mute' },
      { k: 'box', x: 512, y: 48, w: 184, h: 90, t: 'jaxpr · eqns', s: 'shapes frozen\nbranches: only the taken one', tone: 'steel' },
      { k: 'box', x: 512, y: 152, w: 184, h: 44, t: 'consts ← W', s: 'closed-over values ride along', tone: 'steel', dash: true },
      { k: 'line', x1: 194, y1: 172, x2: 512, y2: 172, a: 'end', tone: 'mute', dash: true },
      { k: 'text', x: 300, y: 232, t: 'side effects fire once, at trace time · a new shape means a new trace', size: 10 },
    ],
  },
}
