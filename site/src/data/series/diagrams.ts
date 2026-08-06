// Diagram registry for series pages. Copper is compute, steel is memory,
// dashed is control or absence, same as everywhere. The tpu-chip figure is
// the layer guide's, shared by reference; the GPU figures are drawn from
// the scaling book's Hopper description.
import { DIAGRAMS, type DiagramSpec } from '../diagrams'

export const SERIES_DIAGRAMS: Record<string, DiagramSpec> = {
  'tpu-chip': DIAGRAMS['tpu-chip']!,
  'sm-anatomy': {
    w: 720,
    h: 330,
    alt: 'One H100 streaming multiprocessor: four identical subpartitions, each with a warp scheduler, a tensor core, 32 CUDA cores, and a 16k-register file, over 256 kB of shared SMEM',
    caption: 'one H100 SM · four subpartitions over one shared SMEM · counts from jax-ml.github.io/scaling-book/gpus/',
    els: [
      { k: 'box', x: 16, y: 16, w: 688, h: 250, t: '', tone: 'ink' },
      { k: 'text', x: 28, y: 36, t: 'streaming multiprocessor (1 of 132)', tone: 'ink', size: 11 },
      ...[0, 1, 2, 3].flatMap((i): DiagramSpec['els'] => [
        { k: 'box', x: 32 + i * 168, y: 52, w: 152, h: 44, t: 'warp scheduler', s: '32 threads/issue', tone: 'mute' },
        { k: 'box', x: 32 + i * 168, y: 104, w: 152, h: 44, t: 'tensor core', s: '~8x8x8 matmul/cycle', tone: 'copper' },
        { k: 'box', x: 32 + i * 168, y: 156, w: 152, h: 34, t: '32 fp32 CUDA cores', tone: 'copper' },
        { k: 'box', x: 32 + i * 168, y: 198, w: 152, h: 34, t: '16k 32-bit registers', tone: 'steel' },
      ]),
      { k: 'box', x: 32, y: 276, w: 656, h: 38, t: 'SMEM / L1 · 256 kB, shared across the four subpartitions', tone: 'steel' },
      { k: 'line', x1: 360, y1: 266, x2: 360, y2: 276, a: 'both', tone: 'mute' },
    ],
  },
  'nvlink-node': {
    w: 720,
    h: 300,
    alt: 'An 8-GPU H100 node: every GPU wired to four NVSwitches, giving 450 GB/s of egress per GPU and one switch crossing between any pair',
    caption: 'one H100 node · 8 GPUs, 4 NVSwitches, 18 NVLink-4 links per GPU at 25 GB/s each · jax-ml.github.io/scaling-book/gpus/',
    els: [
      ...[0, 1, 2, 3, 4, 5, 6, 7].map((i): DiagramSpec['els'][number] => ({
        k: 'box', x: 24 + i * 86, y: 30, w: 72, h: 44, t: `gpu ${i}`, tone: 'copper',
      })),
      ...[0, 1, 2, 3].map((i): DiagramSpec['els'][number] => ({
        k: 'box', x: 110 + i * 130, y: 170, w: 110, h: 40, t: `NVSwitch ${i}`, tone: 'steel',
      })),
      ...[0, 1, 2, 3, 4, 5, 6, 7].flatMap((g): DiagramSpec['els'] =>
        [0, 3].map((s): DiagramSpec['els'][number] => ({
          k: 'line', x1: 60 + g * 86, y1: 74, x2: 165 + s * 130, y2: 170, tone: 'mute',
        })),
      ),
      { k: 'text', x: 24, y: 240, t: 'every GPU reaches every switch; two of eight fan-out lines drawn per GPU', tone: 'mute', size: 10 },
      { k: 'text', x: 24, y: 262, t: 'per-GPU egress: 18 links x 25 GB/s = 450 GB/s full duplex', tone: 'ink', size: 11 },
      { k: 'text', x: 24, y: 284, t: 'any pair of GPUs: one switch crossing, no multi-hop forwarding', tone: 'ink', size: 11 },
    ],
  },
}
