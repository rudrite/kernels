// Diagram specs for the JAX path, rendered by the same BlockDiagram the
// kernel path uses: copper is the artifact under discussion, steel is
// machinery, dashed is control or a miss. Facts on these diagrams restate
// chapter facts; nothing appears here that the prose does not teach.
import type { DiagramSpec } from '../diagrams'

export const JAX_DIAGRAMS: Record<string, DiagramSpec> = {
  'jax-machine': {
    w: 720,
    h: 300,
    alt: 'The JAX machine: a Python function is traced into a jaxpr; grad and vmap map jaxpr to jaxpr; jit hands the jaxpr to XLA, which compiles one executable for the device',
    caption: 'the machine of the whole path: one recording, many transformations of it',
    els: [
      { k: 'box', x: 24, y: 120, w: 150, h: 76, t: 'your function', s: 'Python\nruns once, at trace time', tone: 'mute' },
      { k: 'line', x1: 174, y1: 158, x2: 246, y2: 158, a: 'end', tone: 'steel', t: 'trace', ly: 146 },
      { k: 'box', x: 246, y: 110, w: 160, h: 96, t: 'jaxpr', s: 'the recording\nshapes and dtypes,\nno values', tone: 'copper' },
      { k: 'box', x: 226, y: 20, w: 96, h: 48, t: 'grad', tone: 'steel' },
      { k: 'box', x: 338, y: 20, w: 96, h: 48, t: 'vmap', tone: 'steel' },
      { k: 'line', x1: 296, y1: 110, x2: 274, y2: 68, a: 'end', tone: 'steel' },
      { k: 'line', x1: 364, y1: 68, x2: 356, y2: 110, a: 'end', tone: 'steel', t: 'jaxpr in, jaxpr out', lx: 512, ly: 46 },
      { k: 'line', x1: 406, y1: 158, x2: 478, y2: 158, a: 'end', tone: 'steel', t: 'jit', ly: 146 },
      { k: 'box', x: 478, y: 110, w: 122, h: 96, t: 'XLA', s: 'fuse, schedule,\ncompile', tone: 'steel' },
      { k: 'line', x1: 600, y1: 158, x2: 636, y2: 158, a: 'end', tone: 'steel' },
      { k: 'box', x: 636, y: 110, w: 76, h: 96, t: 'devices', s: 'TPU\nGPU\nCPU', tone: 'ink' },
      { k: 'text', x: 700, y: 290, t: 'copper = the artifact · steel = machinery', anchor: 'end', size: 9.5 },
    ],
  },

  'jax-cache': {
    w: 720,
    h: 250,
    alt: 'The jit cache: a call is reduced to a key of function identity, pytree structure, leaf shapes and dtypes, and static values; a hit reuses the executable in microseconds, a miss traces and compiles again',
    caption: 'every call becomes this key; a recompile means one of its four parts changed',
    els: [
      { k: 'box', x: 24, y: 84, w: 130, h: 72, t: 'f(x, k)', s: 'a call arrives', tone: 'mute' },
      { k: 'line', x1: 154, y1: 120, x2: 226, y2: 120, a: 'end', tone: 'steel' },
      { k: 'box', x: 226, y: 60, w: 200, h: 122, t: 'the key', s: 'function identity\npytree structure\nshape + dtype per leaf\nstatic argument values', tone: 'copper' },
      { k: 'line', x1: 426, y1: 96, x2: 512, y2: 72, a: 'end', tone: 'steel', t: 'hit', lx: 466, ly: 70 },
      { k: 'box', x: 512, y: 40, w: 184, h: 58, t: 'cached executable', s: 'dispatch in microseconds', tone: 'steel' },
      { k: 'line', x1: 426, y1: 150, x2: 512, y2: 172, a: 'end', tone: 'mute', dash: true, t: 'miss', lx: 466, ly: 176 },
      { k: 'box', x: 512, y: 144, w: 184, h: 58, t: 'trace + compile', s: 'milliseconds to minutes', tone: 'mute', dash: true },
      { k: 'text', x: 700, y: 240, t: 'dashed = the path steady state must never take', anchor: 'end', size: 9.5 },
    ],
  },

  'jax-reverse': {
    w: 720,
    h: 270,
    alt: 'Reverse-mode autodiff: the forward pass runs left to right storing one residual per operation; the backward pass runs right to left consuming them; remat replaces stored residuals with recomputation',
    caption: 'where the memory goes in reverse mode: one residual per op, consumed right to left',
    els: [
      { k: 'box', x: 24, y: 40, w: 120, h: 56, t: 'op 1', tone: 'copper' },
      { k: 'box', x: 204, y: 40, w: 120, h: 56, t: 'op 2', tone: 'copper' },
      { k: 'box', x: 384, y: 40, w: 120, h: 56, t: 'op 3', tone: 'copper' },
      { k: 'box', x: 564, y: 40, w: 132, h: 56, t: 'loss', s: 'scalar out', tone: 'copper' },
      { k: 'line', x1: 144, y1: 68, x2: 204, y2: 68, a: 'end', tone: 'copper' },
      { k: 'line', x1: 324, y1: 68, x2: 384, y2: 68, a: 'end', tone: 'copper' },
      { k: 'line', x1: 504, y1: 68, x2: 564, y2: 68, a: 'end', tone: 'copper', t: 'forward', lx: 534, ly: 56 },
      { k: 'box', x: 24, y: 128, w: 120, h: 44, t: 'residual', tone: 'steel' },
      { k: 'box', x: 204, y: 128, w: 120, h: 44, t: 'residual', tone: 'steel' },
      { k: 'box', x: 384, y: 128, w: 120, h: 44, t: 'residual', s: 'the memory bill', tone: 'steel' },
      { k: 'line', x1: 84, y1: 96, x2: 84, y2: 128, a: 'end', tone: 'steel' },
      { k: 'line', x1: 264, y1: 96, x2: 264, y2: 128, a: 'end', tone: 'steel' },
      { k: 'line', x1: 444, y1: 96, x2: 444, y2: 128, a: 'end', tone: 'steel' },
      { k: 'line', x1: 630, y1: 96, x2: 630, y2: 210, a: 'end', tone: 'mute' },
      { k: 'line', x1: 630, y1: 210, x2: 96, y2: 210, a: 'end', tone: 'mute', t: 'backward: cotangents consume residuals, right to left', lx: 372, ly: 232 },
      { k: 'text', x: 700, y: 260, t: 'jax.checkpoint: drop the steel row, recompute it instead', anchor: 'end', size: 9.5 },
    ],
  },

  'jax-sharding-ladder': {
    w: 720,
    h: 280,
    alt: 'Three levels of sharding control: jit with NamedSharding lets GSPMD insert collectives automatically; with_sharding_constraint pins intermediate layouts; shard_map hands each device its local block and you call the collectives yourself',
    caption: 'the ladder of control: climb down only when the level above fights you',
    els: [
      { k: 'box', x: 130, y: 24, w: 470, h: 62, t: 'jit + NamedSharding', s: 'global math · GSPMD inserts the collectives', tone: 'copper' },
      { k: 'box', x: 130, y: 108, w: 470, h: 62, t: 'with_sharding_constraint', s: 'same program · you pin the layouts propagation got wrong', tone: 'steel' },
      { k: 'box', x: 130, y: 192, w: 470, h: 62, t: 'shard_map', s: 'per-device blocks · you call psum, ppermute yourself', tone: 'ink' },
      { k: 'line', x1: 80, y1: 40, x2: 80, y2: 240, a: 'end', tone: 'mute', t: 'more control,\nmore responsibility', lx: 40, ly: 136 },
      { k: 'text', x: 640, y: 55, t: 'start here', tone: 'copper', size: 10 },
      { k: 'text', x: 700, y: 272, t: 'below this ladder: the kernel path (collectives as code)', anchor: 'end', size: 9.5 },
    ],
  },
}
