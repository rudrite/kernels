// Diagram registry for lesson pages. Copper is compute, steel is memory,
// dashed is control or absence, same as everywhere. The tpu-chip figure is
// the layer guide's, shared by reference; the GPU figures are drawn from
// the scaling book's Hopper description.
import { DIAGRAMS, type DiagramSpec } from '../diagrams'

export const LESSON_DIAGRAMS: Record<string, DiagramSpec> = {
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
  },  'blackwell-package': {
    w: 720,
    h: 300,
    alt: 'One B200 package: two reticle-limited dies side by side, joined by a 10 TB/s NV-HBI link, with four HBM3e stacks on each side, all on one interposer',
    caption: 'one B200 package · two reticle-limited dies joined by NV-HBI, eight HBM3e stacks · NVIDIA Blackwell architecture technical brief and the Blackwell Ultra developer blog; per-die area is not published',
    els: [
      { k: 'box', x: 16, y: 40, w: 688, h: 178, t: '', tone: 'mute', dash: true },
      { k: 'text', x: 28, y: 60, t: 'silicon interposer (CoWoS) · TSMC certified 5.5x reticle-size interposers in 2025', tone: 'mute', size: 10 },
      ...[0, 1, 2, 3].map((i): DiagramSpec['els'][number] => ({
        k: 'box', x: 28 + i * 52, y: 76, w: 48, h: 120, t: 'HBM3e', s: '24 GB\n1.0 TB/s', tone: 'steel',
      })),
      { k: 'box', x: 246, y: 76, w: 86, h: 120, t: 'die 0', s: 'at the\nreticle limit', tone: 'copper' },
      { k: 'line', x1: 332, y1: 136, x2: 392, y2: 136, w: 8, tone: 'copper', a: 'both' },
      { k: 'text', x: 362, y: 124, t: 'NV-HBI 10 TB/s', tone: 'ink', size: 10, anchor: 'middle' },
      { k: 'box', x: 392, y: 76, w: 86, h: 120, t: 'die 1', s: 'at the\nreticle limit', tone: 'copper' },
      ...[0, 1, 2, 3].map((i): DiagramSpec['els'][number] => ({
        k: 'box', x: 494 + i * 52, y: 76, w: 48, h: 120, t: 'HBM3e', s: '24 GB\n1.0 TB/s', tone: 'steel',
      })),
      { k: 'text', x: 28, y: 244, t: '208 B transistors on TSMC 4NP, both dies together; NVIDIA publishes no per-die mm2', tone: 'ink', size: 11 },
      { k: 'text', x: 28, y: 266, t: '8 stacks x 24 GB = 192 GB at 8 TB/s; 10 TB/s on NV-HBI, direction unspecified', tone: 'ink', size: 11 },
    ],
  },  sparsecore: {
    w: 720,
    h: 320,
    alt: 'One TPU v4 SparseCore: sixteen compute tiles, each holding a Fetch Unit, an 8-wide scVPU and a Flush Unit over its own HBM channel, all working out of 2.5 MiB of Spmem, with five cross-channel units reaching across all sixteen banks',
    caption: 'one of the four SparseCores on a TPU v4 chip · units and sizes from Jouppi et al., ISCA 2023, section 3.5 and Figure 7',
    els: [
      { k: 'box', x: 152, y: 44, w: 548, h: 232, t: '', tone: 'ink' },
      { k: 'text', x: 164, y: 64, t: 'SparseCore · 1 of 4 on a v4 chip · dataflow, not an array', tone: 'ink', size: 11 },
      { k: 'box', x: 20, y: 92, w: 112, h: 98, t: 'HBM', s: '16 channels\none per tile', tone: 'steel' },
      { k: 'line', x1: 132, y1: 107, x2: 166, y2: 107, a: 'end', tone: 'steel', t: 'fetch', ly: 100 },
      { k: 'line', x1: 166, y1: 175, x2: 132, y2: 175, a: 'end', tone: 'steel', t: 'flush, backward pass', ly: 190 },
      ...[0, 1, 2, 3].flatMap((i) => [
        { k: 'box' as const, x: 166 + i * 128, y: 92, w: 112, h: 30, t: 'Fetch Unit', tone: 'steel' as const },
        { k: 'box' as const, x: 166 + i * 128, y: 126, w: 112, h: 30, t: 'scVPU · 8-wide', tone: 'copper' as const },
        { k: 'box' as const, x: 166 + i * 128, y: 160, w: 112, h: 30, t: 'Flush Unit', tone: 'steel' as const },
      ]),
      { k: 'text', x: 166, y: 86, t: 'tile 1', tone: 'mute', size: 9.5 },
      { k: 'text', x: 550, y: 86, t: 'tile 16', tone: 'mute', size: 9.5 },
      { k: 'box', x: 166, y: 200, w: 520, h: 30, t: 'Spmem · 2.5 MiB per SparseCore, one slice per tile', tone: 'steel' },
      { k: 'box', x: 166, y: 236, w: 520, h: 30, t: '5 cross-channel units · across all 16 banks · CISC-like, data-dependent runtime', tone: 'copper' },
      { k: 'text', x: 700, y: 300, t: 'copper = compute · steel = memory', anchor: 'end', size: 9.5 },
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
