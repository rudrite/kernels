// The descent's depth system, shared site-wide: every chapter inherits the
// shade of its stratum on the home page, so internal pages carry the same
// light-above-the-gap, dark-below-it grammar.
import { LAYERS, STAGES } from '../data/track'

export interface Depth {
  cls: string
  dark: boolean
}

const LAYER_DEPTH: Depth[] = [
  { cls: 'depth-0', dark: false },
  { cls: 'depth-1', dark: false },
  { cls: 'depth-2', dark: false },
  { cls: 'depth-3', dark: false },
  { cls: 'depth-gap', dark: false },
  { cls: 'depth-5', dark: true },
  { cls: 'depth-6', dark: true },
  { cls: 'depth-7', dark: true },
  { cls: 'depth-8', dark: true },
]

export const depthFor = (key: string): Depth => {
  const [kind, id] = key.split(':')
  if (kind === 'l') {
    const idx = LAYERS.findIndex((l) => l.id === id)
    return LAYER_DEPTH[idx] ?? LAYER_DEPTH[0]!
  }
  const stage = STAGES.find((s) => s.id === id)
  const idx = LAYERS.findIndex((l) => l.id === stage?.layer)
  return LAYER_DEPTH[idx] ?? LAYER_DEPTH[0]!
}
