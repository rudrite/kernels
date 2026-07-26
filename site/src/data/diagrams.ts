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

  'grid-revisit': {
    w: 720,
    h: 310,
    alt: 'The sequence of grid steps for a three-axis matmul grid, with the three steps sharing i and j pointing at the same output block, initialized at k equals zero and accumulated after',
    caption: 'why the K axis is "arbitrary": three grid steps in a row own the same output block, and order is the correctness',
    els: [
      { k: 'text', x: 24, y: 34, t: 'grid (2, 2, 3) · steps in order', tone: 'ink', size: 12 },
      { k: 'box', x: 24, y: 48, w: 150, h: 26, t: '(i=0, j=0, k=0)', tone: 'copper' },
      { k: 'box', x: 24, y: 80, w: 150, h: 26, t: '(i=0, j=0, k=1)', tone: 'copper' },
      { k: 'box', x: 24, y: 112, w: 150, h: 26, t: '(i=0, j=0, k=2)', tone: 'copper' },
      { k: 'box', x: 24, y: 152, w: 150, h: 26, t: '(i=0, j=1, k=0)', tone: 'mute' },
      { k: 'box', x: 24, y: 184, w: 150, h: 26, t: '(i=0, j=1, k=1)', tone: 'mute' },
      { k: 'box', x: 24, y: 216, w: 150, h: 26, t: '…', tone: 'mute' },
      { k: 'line', x1: 174, y1: 61, x2: 400, y2: 118, a: 'end', tone: 'copper', t: 'k=0: initialize', lx: 300, ly: 62 },
      { k: 'line', x1: 174, y1: 93, x2: 400, y2: 130, a: 'end', tone: 'copper', t: 'k=1: accumulate', lx: 306, ly: 96 },
      { k: 'line', x1: 174, y1: 125, x2: 400, y2: 142, a: 'end', tone: 'copper', t: 'k=2: accumulate', lx: 306, ly: 156 },
      { k: 'box', x: 400, y: 96, w: 190, h: 76, t: 'output block C(0, 0)', s: 'the same VMEM block,\nrevisited three times', tone: 'steel' },
      { k: 'line', x1: 174, y1: 165, x2: 400, y2: 210, a: 'end', tone: 'mute', dash: true },
      { k: 'box', x: 400, y: 190, w: 190, h: 46, t: 'C(0, 1)', s: 'a different block: j moved', tone: 'mute' },
      { k: 'text', x: 24, y: 282, t: 'i, j: "parallel", steps independent · k: "arbitrary", order is the algorithm', tone: 'copper', size: 10.5 },
    ],
  },

  'scalar-prefetch': {
    w: 720,
    h: 280,
    alt: 'Scalar arrays land in SMEM before the grid pipeline starts, and each grid step reads the prefetched permutation to decide which block its index map fetches',
    caption: 'scalar prefetch: the scalars land before the pipeline starts, so the index maps themselves can read data',
    els: [
      { k: 'box', x: 24, y: 44, w: 190, h: 60, t: 'perm → SMEM', s: '[3, 0, 2, 1]\nbefore any grid step', tone: 'steel' },
      { k: 'line', x1: 119, y1: 104, x2: 119, y2: 140, a: 'end', tone: 'steel', w: 2.5 },
      { k: 'text', x: 24, y: 132, t: 'then the pipeline starts', size: 10 },
      { k: 'box', x: 24, y: 148, w: 158, h: 62, t: 'step 0', s: 'index map reads perm[0]\nfetches block 3', tone: 'copper' },
      { k: 'box', x: 196, y: 148, w: 158, h: 62, t: 'step 1', s: 'reads perm[1]\nfetches block 0', tone: 'copper' },
      { k: 'box', x: 368, y: 148, w: 158, h: 62, t: 'step 2', s: 'reads perm[2]\nfetches block 2', tone: 'copper' },
      { k: 'box', x: 540, y: 148, w: 158, h: 62, t: 'step 3', s: 'reads perm[3]\nfetches block 1', tone: 'copper' },
      { k: 'line', x1: 214, y1: 74, x2: 620, y2: 74, dash: true, tone: 'mute' },
      { k: 'line', x1: 620, y1: 74, x2: 620, y2: 148, dash: true, a: 'end', tone: 'mute', t: 'visible to every index map', lx: 480, ly: 66 },
      { k: 'text', x: 24, y: 246, t: 'the schedule read data: this is the splash and ragged-attention mechanism', tone: 'copper', size: 10.5 },
    ],
  },

  'dma-semaphore': {
    w: 720,
    h: 280,
    alt: 'Three lanes showing a manual DMA: the kernel starts the copy and keeps computing, the DMA engine transfers in flight, the semaphore signals at completion, and the wait returns when the signal has landed',
    caption: 'a manual DMA, the transaction the automatic pipeline writes for you: start, overlap, signal, wait',
    els: [
      { k: 'text', x: 24, y: 40, t: 'kernel', size: 10 },
      { k: 'box', x: 100, y: 24, w: 110, h: 28, t: 'copy.start()', tone: 'copper' },
      { k: 'box', x: 214, y: 24, w: 240, h: 28, t: 'compute on other data', tone: 'copper' },
      { k: 'box', x: 458, y: 24, w: 110, h: 28, t: 'copy.wait()', tone: 'copper' },
      { k: 'box', x: 572, y: 24, w: 124, h: 28, t: 'use scratch', tone: 'copper' },
      { k: 'text', x: 24, y: 116, t: 'DMA engine', size: 10 },
      { k: 'box', x: 130, y: 100, w: 260, h: 28, t: 'transfer in flight', tone: 'steel' },
      { k: 'line', x1: 155, y1: 52, x2: 155, y2: 100, a: 'end', tone: 'mute', dash: true },
      { k: 'text', x: 24, y: 192, t: 'semaphore', size: 10 },
      { k: 'line', x1: 390, y1: 128, x2: 390, y2: 176, a: 'end', tone: 'copper', w: 2, t: 'signal at completion', lx: 420, ly: 160 },
      { k: 'box', x: 340, y: 176, w: 100, h: 26, t: 'count +1', tone: 'mute' },
      { k: 'line', x1: 440, y1: 189, x2: 513, y2: 189, tone: 'mute', dash: true },
      { k: 'line', x1: 513, y1: 189, x2: 513, y2: 52, a: 'end', tone: 'mute', dash: true, t: 'wait consumes it', lx: 580, ly: 120 },
      { k: 'text', x: 24, y: 250, t: 'the whole grid pipeline is this transaction, written for you per block · remote DMA aims it at a neighbor chip', size: 10 },
    ],
  },

  'all-gather-steps': {
    w: 720,
    h: 300,
    alt: 'Four chips before and after a ring all-gather: at the start each chip holds only its own shard on the diagonal, and after N minus one hops every chip holds all four shards',
    caption: 'a ring all-gather on four chips: N−1 hops, each chip forwarding the shard it just received',
    els: [
      { k: 'text', x: 24, y: 34, t: 'before · each chip has its shard', tone: 'ink', size: 12 },
      { k: 'text', x: 30, y: 66, t: 'chip 0', size: 9.5 }, { k: 'text', x: 30, y: 98, t: 'chip 1', size: 9.5 },
      { k: 'text', x: 30, y: 130, t: 'chip 2', size: 9.5 }, { k: 'text', x: 30, y: 162, t: 'chip 3', size: 9.5 },
      { k: 'box', x: 90, y: 50, w: 44, h: 24, t: 'A', tone: 'copper' }, { k: 'box', x: 138, y: 50, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 186, y: 50, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 234, y: 50, w: 44, h: 24, t: '', tone: 'mute', dash: true },
      { k: 'box', x: 90, y: 82, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 138, y: 82, w: 44, h: 24, t: 'B', tone: 'copper' }, { k: 'box', x: 186, y: 82, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 234, y: 82, w: 44, h: 24, t: '', tone: 'mute', dash: true },
      { k: 'box', x: 90, y: 114, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 138, y: 114, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 186, y: 114, w: 44, h: 24, t: 'C', tone: 'copper' }, { k: 'box', x: 234, y: 114, w: 44, h: 24, t: '', tone: 'mute', dash: true },
      { k: 'box', x: 90, y: 146, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 138, y: 146, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 186, y: 146, w: 44, h: 24, t: '', tone: 'mute', dash: true }, { k: 'box', x: 234, y: 146, w: 44, h: 24, t: 'D', tone: 'copper' },
      { k: 'line', x1: 320, y1: 110, x2: 420, y2: 110, a: 'end', tone: 'steel', w: 3, t: '3 hops around the ring', lx: 370, ly: 96 },
      { k: 'text', x: 440, y: 34, t: 'after · everyone has everything', tone: 'ink', size: 12 },
      { k: 'box', x: 446, y: 50, w: 44, h: 24, t: 'A', tone: 'steel' }, { k: 'box', x: 494, y: 50, w: 44, h: 24, t: 'B', tone: 'steel' }, { k: 'box', x: 542, y: 50, w: 44, h: 24, t: 'C', tone: 'steel' }, { k: 'box', x: 590, y: 50, w: 44, h: 24, t: 'D', tone: 'steel' },
      { k: 'box', x: 446, y: 82, w: 44, h: 24, t: 'A', tone: 'steel' }, { k: 'box', x: 494, y: 82, w: 44, h: 24, t: 'B', tone: 'steel' }, { k: 'box', x: 542, y: 82, w: 44, h: 24, t: 'C', tone: 'steel' }, { k: 'box', x: 590, y: 82, w: 44, h: 24, t: 'D', tone: 'steel' },
      { k: 'box', x: 446, y: 114, w: 44, h: 24, t: 'A', tone: 'steel' }, { k: 'box', x: 494, y: 114, w: 44, h: 24, t: 'B', tone: 'steel' }, { k: 'box', x: 542, y: 114, w: 44, h: 24, t: 'C', tone: 'steel' }, { k: 'box', x: 590, y: 114, w: 44, h: 24, t: 'D', tone: 'steel' },
      { k: 'box', x: 446, y: 146, w: 44, h: 24, t: 'A', tone: 'steel' }, { k: 'box', x: 494, y: 146, w: 44, h: 24, t: 'B', tone: 'steel' }, { k: 'box', x: 542, y: 146, w: 44, h: 24, t: 'C', tone: 'steel' }, { k: 'box', x: 590, y: 146, w: 44, h: 24, t: 'D', tone: 'steel' },
      { k: 'text', x: 24, y: 222, t: 'hop h: chip n sends the shard it received at hop h−1 to chip n+1', size: 10 },
      { k: 'text', x: 24, y: 244, t: 'time ≈ (N−1)/N × bytes / link bandwidth · bidirectional rings halve it · reduce-scatter is this ring with an add at each hop', size: 10 },
    ],
  },

  'while-region': {
    w: 720,
    h: 280,
    alt: 'A stablehlo.while op holding two regions, the condition and the body, with the carry tuple entering, looping from body back through condition, and leaving when the condition is false',
    caption: 'control flow as regions: the carry is explicit loop state, and both regions live inside the op',
    els: [
      { k: 'line', x1: 40, y1: 130, x2: 120, y2: 130, a: 'end', tone: 'steel', w: 2.5, t: 'init carry', lx: 74, ly: 118 },
      { k: 'box', x: 120, y: 40, w: 470, h: 200, t: '', tone: 'ink' },
      { k: 'text', x: 138, y: 66, t: 'stablehlo.while · carry: (total, n)', tone: 'ink', size: 12 },
      { k: 'box', x: 144, y: 84, w: 190, h: 64, t: 'cond region', s: 'n < 8 → i1', tone: 'mute' },
      { k: 'box', x: 376, y: 84, w: 190, h: 64, t: 'body region', s: 'new carry = step(carry)', tone: 'copper' },
      { k: 'line', x1: 334, y1: 116, x2: 376, y2: 116, a: 'end', tone: 'mute', t: 'true' },
      { k: 'line', x1: 471, y1: 148, x2: 471, y2: 190, tone: 'copper' },
      { k: 'line', x1: 471, y1: 190, x2: 239, y2: 190, tone: 'copper' },
      { k: 'line', x1: 239, y1: 190, x2: 239, y2: 148, a: 'end', tone: 'copper', t: 'carry loops back', lx: 355, ly: 208 },
      { k: 'line', x1: 590, y1: 116, x2: 680, y2: 116, a: 'end', tone: 'steel', w: 2.5, t: 'false: final carry', lx: 636, ly: 104 },
      { k: 'text', x: 138, y: 262, t: 'lax.scan lowers to exactly this, with the stacked outputs threaded through the carry', size: 10 },
    ],
  },

  'layout-minor': {
    w: 720,
    h: 300,
    alt: 'A two by four array shown once as values and twice as memory strips: minor-to-major one-zero puts rows contiguous in memory, zero-one puts columns contiguous',
    caption: 'what {1,0} means: which axis is contiguous in memory, and why a consumer wanting the other order costs a copy',
    els: [
      { k: 'text', x: 24, y: 34, t: 'the array · shape (2, 4)', tone: 'ink', size: 12 },
      { k: 'box', x: 24, y: 48, w: 60, h: 32, t: 'a00', tone: 'mute' }, { k: 'box', x: 84, y: 48, w: 60, h: 32, t: 'a01', tone: 'mute' }, { k: 'box', x: 144, y: 48, w: 60, h: 32, t: 'a02', tone: 'mute' }, { k: 'box', x: 204, y: 48, w: 60, h: 32, t: 'a03', tone: 'mute' },
      { k: 'box', x: 24, y: 80, w: 60, h: 32, t: 'a10', tone: 'mute' }, { k: 'box', x: 84, y: 80, w: 60, h: 32, t: 'a11', tone: 'mute' }, { k: 'box', x: 144, y: 80, w: 60, h: 32, t: 'a12', tone: 'mute' }, { k: 'box', x: 204, y: 80, w: 60, h: 32, t: 'a13', tone: 'mute' },
      { k: 'text', x: 24, y: 156, t: '{1,0} · axis 1 minor: rows contiguous', tone: 'copper', size: 11 },
      { k: 'box', x: 24, y: 166, w: 54, h: 26, t: 'a00', tone: 'copper' }, { k: 'box', x: 78, y: 166, w: 54, h: 26, t: 'a01', tone: 'copper' }, { k: 'box', x: 132, y: 166, w: 54, h: 26, t: 'a02', tone: 'copper' }, { k: 'box', x: 186, y: 166, w: 54, h: 26, t: 'a03', tone: 'copper' }, { k: 'box', x: 240, y: 166, w: 54, h: 26, t: 'a10', tone: 'steel' }, { k: 'box', x: 294, y: 166, w: 54, h: 26, t: 'a11', tone: 'steel' }, { k: 'box', x: 348, y: 166, w: 54, h: 26, t: 'a12', tone: 'steel' }, { k: 'box', x: 402, y: 166, w: 54, h: 26, t: 'a13', tone: 'steel' },
      { k: 'text', x: 24, y: 226, t: '{0,1} · axis 0 minor: columns contiguous', tone: 'copper', size: 11 },
      { k: 'box', x: 24, y: 236, w: 54, h: 26, t: 'a00', tone: 'copper' }, { k: 'box', x: 78, y: 236, w: 54, h: 26, t: 'a10', tone: 'steel' }, { k: 'box', x: 132, y: 236, w: 54, h: 26, t: 'a01', tone: 'copper' }, { k: 'box', x: 186, y: 236, w: 54, h: 26, t: 'a11', tone: 'steel' }, { k: 'box', x: 240, y: 236, w: 54, h: 26, t: 'a02', tone: 'copper' }, { k: 'box', x: 294, y: 236, w: 54, h: 26, t: 'a12', tone: 'steel' }, { k: 'box', x: 348, y: 236, w: 54, h: 26, t: 'a03', tone: 'copper' }, { k: 'box', x: 402, y: 236, w: 54, h: 26, t: 'a13', tone: 'steel' },
      { k: 'text', x: 480, y: 186, t: 'same values, different walk:', size: 10 },
      { k: 'text', x: 480, y: 200, t: 'a transpose can be free (relabel)', size: 10 },
      { k: 'text', x: 480, y: 214, t: 'or a copy (rearrange), and the', size: 10 },
      { k: 'text', x: 480, y: 228, t: 'dump shows which one you got', size: 10 },
    ],
  },

  'sublane-lane': {
    w: 720,
    h: 280,
    alt: 'One vector register drawn as eight sublane rows by 128 lanes, with the packing rule that bf16 fits sixteen rows and int8 thirty-two into the same physical tile',
    caption: 'the register tile the whole lattice comes from: 8 sublanes by 128 lanes, packed denser as dtypes shrink',
    els: [
      { k: 'text', x: 190, y: 40, t: '128 lanes →', size: 10.5 },
      { k: 'box', x: 110, y: 52, w: 440, h: 22, t: '', tone: 'copper' },
      { k: 'box', x: 110, y: 74, w: 440, h: 22, t: '', tone: 'mute' },
      { k: 'box', x: 110, y: 96, w: 440, h: 22, t: '', tone: 'mute' },
      { k: 'box', x: 110, y: 118, w: 440, h: 22, t: '', tone: 'mute' },
      { k: 'box', x: 110, y: 140, w: 440, h: 22, t: '', tone: 'mute' },
      { k: 'box', x: 110, y: 162, w: 440, h: 22, t: '', tone: 'mute' },
      { k: 'box', x: 110, y: 184, w: 440, h: 22, t: '', tone: 'mute' },
      { k: 'box', x: 110, y: 206, w: 440, h: 22, t: '', tone: 'mute' },
      { k: 'line', x1: 96, y1: 52, x2: 96, y2: 228, a: 'both', tone: 'mute', t: '8 sublanes', lx: 52, ly: 140 },
      { k: 'text', x: 240, y: 146, t: 'one f32 register · (8, 128)', tone: 'ink', size: 12 },
      { k: 'text', x: 580, y: 92, t: 'bf16: two per slot', size: 10 },
      { k: 'text', x: 580, y: 106, t: '→ (16, 128)', tone: 'copper', size: 10 },
      { k: 'text', x: 580, y: 132, t: 'int8: four per slot', size: 10 },
      { k: 'text', x: 580, y: 146, t: '→ (32, 128)', tone: 'copper', size: 10 },
      { k: 'text', x: 110, y: 256, t: 'block shapes must tile this, which is the sentence the lattice error quotes back at you', size: 10 },
    ],
  },
}
