// Diagram specs for the PyTorch path, rendered by the shared BlockDiagram:
// copper is the artifact under discussion, steel is machinery, dashed is
// closed, control, or a break. Facts restate chapter facts.
import type { DiagramSpec } from '../diagrams'

export const PT_DIAGRAMS: Record<string, DiagramSpec> = {
  'pt-tensor': {
    w: 720,
    h: 260,
    alt: 'Two tensors sharing one storage: x with shape (3,4) and stride (4,1), and its transpose view with shape (4,3) and stride (1,4), both pointing into the same twelve-element buffer',
    caption: 'two tensors, one storage: the strides are the map, and t() only rewrote the map',
    els: [
      { k: 'box', x: 40, y: 30, w: 220, h: 74, t: 'x', s: 'shape (3, 4)\nstride (4, 1) · contiguous', tone: 'copper' },
      { k: 'box', x: 460, y: 30, w: 220, h: 74, t: 'x.t()', s: 'shape (4, 3)\nstride (1, 4) · a view', tone: 'copper' },
      { k: 'box', x: 40, y: 160, w: 640, h: 56, t: 'storage', s: 'one flat buffer of 12 floats · nobody copied anything', tone: 'steel' },
      { k: 'line', x1: 150, y1: 104, x2: 220, y2: 160, a: 'end', tone: 'steel' },
      { k: 'line', x1: 570, y1: 104, x2: 500, y2: 160, a: 'end', tone: 'steel' },
      { k: 'text', x: 360, y: 140, t: 'same bytes, two maps', size: 10, tone: 'mute' },
      { k: 'text', x: 700, y: 250, t: 'copper = tensors (the maps) · steel = the memory', anchor: 'end', size: 9.5 },
    ],
  },

  'pt-tape': {
    w: 720,
    h: 260,
    alt: 'The dynamic tape: x flows through mul and sum to the loss; each op appended a grad_fn node; backward walks the nodes in reverse, fills x.grad, and frees the graph',
    caption: 'the tape: built by running, consumed by backward, freed unless you ask otherwise',
    els: [
      { k: 'box', x: 30, y: 40, w: 110, h: 56, t: 'x', s: 'leaf\nrequires_grad', tone: 'steel' },
      { k: 'line', x1: 140, y1: 68, x2: 210, y2: 68, a: 'end', tone: 'copper' },
      { k: 'box', x: 210, y: 40, w: 120, h: 56, t: 'x * x', tone: 'copper' },
      { k: 'line', x1: 330, y1: 68, x2: 400, y2: 68, a: 'end', tone: 'copper' },
      { k: 'box', x: 400, y: 40, w: 120, h: 56, t: 'sum', tone: 'copper' },
      { k: 'line', x1: 520, y1: 68, x2: 590, y2: 68, a: 'end', tone: 'copper', t: 'forward', lx: 555, ly: 56 },
      { k: 'box', x: 590, y: 40, w: 100, h: 56, t: 'y', s: 'scalar loss', tone: 'steel' },
      { k: 'box', x: 400, y: 140, w: 150, h: 48, t: 'SumBackward0', s: 'y.grad_fn', tone: 'mute' },
      { k: 'box', x: 210, y: 140, w: 150, h: 48, t: 'MulBackward0', s: 'saved: x', tone: 'mute' },
      { k: 'line', x1: 460, y1: 96, x2: 470, y2: 140, a: 'end', tone: 'mute', dash: true },
      { k: 'line', x1: 270, y1: 96, x2: 280, y2: 140, a: 'end', tone: 'mute', dash: true },
      { k: 'line', x1: 400, y1: 164, x2: 360, y2: 164, a: 'end', tone: 'steel' },
      { k: 'line', x1: 210, y1: 164, x2: 100, y2: 164, tone: 'steel' },
      { k: 'line', x1: 100, y1: 164, x2: 88, y2: 96, a: 'end', tone: 'steel', t: 'backward fills x.grad', lx: 150, ly: 210 },
      { k: 'text', x: 700, y: 250, t: 'the graph frees itself as backward passes: the museum has the receipt', anchor: 'end', size: 9.5 },
    ],
  },

  'pt-compile': {
    w: 720,
    h: 300,
    alt: 'The torch.compile stack: dynamo captures bytecode into an FX graph behind guards; aot_autograd traces forward and backward; inductor emits code; a guard miss recaptures; a graph break falls back to Python around the break',
    caption: 'guarded capture: same guards hit the cache, a new shape recaptures, a break stitches Python back in',
    els: [
      { k: 'box', x: 24, y: 40, w: 140, h: 64, t: 'your function', s: 'Python bytecode\nside effects and all', tone: 'mute' },
      { k: 'line', x1: 164, y1: 72, x2: 232, y2: 72, a: 'end', tone: 'steel' },
      { k: 'box', x: 232, y: 40, w: 150, h: 64, t: 'dynamo', s: 'captures tensor ops\ninstalls guards', tone: 'copper' },
      { k: 'line', x1: 382, y1: 72, x2: 450, y2: 72, a: 'end', tone: 'steel' },
      { k: 'box', x: 450, y: 40, w: 130, h: 64, t: 'FX graph', s: 'ATen calls', tone: 'copper' },
      { k: 'line', x1: 580, y1: 72, x2: 636, y2: 72, a: 'end', tone: 'steel' },
      { k: 'box', x: 636, y: 26, w: 74, h: 44, t: 'aot', s: 'fw + bw', tone: 'steel' },
      { k: 'box', x: 636, y: 82, w: 74, h: 44, t: 'inductor', tone: 'steel' },
      { k: 'box', x: 232, y: 150, w: 220, h: 56, t: 'the guard set', s: 'shapes · dtypes · types · closures', tone: 'copper' },
      { k: 'line', x1: 307, y1: 104, x2: 307, y2: 150, a: 'end', tone: 'mute', dash: true },
      { k: 'text', x: 480, y: 172, t: 'hit: reuse · miss: capture again', size: 10, tone: 'mute' },
      { k: 'box', x: 24, y: 226, w: 428, h: 50, t: 'graph break', s: 'data-dependent Python runs eagerly; compiled pieces stitched around it', tone: 'mute', dash: true },
      { k: 'line', x1: 300, y1: 206, x2: 240, y2: 226, a: 'end', tone: 'mute', dash: true },
      { k: 'text', x: 700, y: 292, t: 'contrast with the jax path: capture is per guard set, and Python survives', anchor: 'end', size: 9.5 },
    ],
  },

  'pt-bridges': {
    w: 720,
    h: 300,
    alt: 'Two bridges from a torch model to the TPU: torch_xla records lazy tensors and materializes at a sync point; torchax maps torch ops onto jax; both converge on StableHLO, then XLA, then the TPU',
    caption: 'two bridges, one destination: below StableHLO you are on the xla path',
    els: [
      { k: 'box', x: 250, y: 20, w: 220, h: 54, t: 'your torch model', s: 'nn.Module · the ch 1-7 world', tone: 'mute' },
      { k: 'line', x1: 320, y1: 74, x2: 200, y2: 110, a: 'end', tone: 'steel' },
      { k: 'line', x1: 400, y1: 74, x2: 520, y2: 110, a: 'end', tone: 'steel' },
      { k: 'box', x: 60, y: 110, w: 280, h: 64, t: 'torch_xla', s: 'lazy tensors record ops\na sync point materializes', tone: 'copper' },
      { k: 'box', x: 380, y: 110, w: 280, h: 64, t: 'torchax', s: 'torch ops mapped onto jax\na torch tensor backed by a jax array', tone: 'copper' },
      { k: 'line', x1: 200, y1: 174, x2: 320, y2: 212, a: 'end', tone: 'steel' },
      { k: 'line', x1: 520, y1: 174, x2: 400, y2: 212, a: 'end', tone: 'steel' },
      { k: 'box', x: 250, y: 212, w: 220, h: 48, t: 'StableHLO', s: 'the framework identity ends here', tone: 'steel' },
      { k: 'line', x1: 470, y1: 236, x2: 560, y2: 236, a: 'end', tone: 'steel', t: 'XLA · TPU', lx: 610, ly: 226 },
      { k: 'text', x: 700, y: 290, t: 'everything below the steel box: the xla path, then the kernel path', anchor: 'end', size: 9.5 },
    ],
  },
}
