// The site-wide search index, built at compile time from the same sources
// the pages render. One entry per reachable thing; the palette filters it.
import { PATH } from './path'
import { LAYER_DOCS } from '../data/layers'
import { STAGES } from '../data/track'
import mistakes from '../data/mistakes.json'
import opCards from '../data/op-cards.json'

export interface SearchEntry {
  kind: 'chapter' | 'instrument' | 'lab' | 'walk' | 'mistake' | 'drill' | 'op' | 'page'
  title: string
  href: string
  hint: string
  /** Extra text the filter matches beyond the title. */
  keywords: string
}

const pad = (n: number) => String(n).padStart(2, '0')

const chapters: SearchEntry[] = PATH.map((c) => {
  const layerDoc = LAYER_DOCS.find((d) => `l:${d.id}` === c.key)
  const stage = STAGES.find((s) => `s:${s.id}` === c.key)
  return {
    kind: 'chapter',
    title: `ch ${pad(c.num)} · ${c.title}`,
    href: c.href,
    hint: c.part === 'i' ? 'part i · the descent' : 'part ii · the track',
    keywords: layerDoc?.lede ?? stage?.goal ?? '',
  }
})

const instruments: SearchEntry[] = [
  { t: 'EX·01 the roofline playground', h: '/s/machine#instruments', k: 'intensity ridge memory-bound compute-bound' },
  { t: 'EX·06 the memory hierarchy', h: '/s/machine#instruments', k: 'hbm vmem mxu vpu widths' },
  { t: 'EX·02 tiled matmul, double-buffered', h: '/s/pallas#instruments', k: 'pipeline dma prefetch grid' },
  { t: 'EX·07 the matmul with the pipeline off', h: '/s/pallas#instruments', k: 'idle mxu serial transfer' },
  { t: 'EX·09 the BlockSpec sandbox', h: '/s/pallas#instruments', k: 'legality lattice vmem budget blocks' },
  { t: 'EX·12 the synced walk', h: '/s/pallas#instruments', k: 'trace to code frames lines' },
  { t: 'EX·05 the IR x-ray', h: '/s/ir#instruments', k: 'jaxpr stablehlo hover sync attention' },
  { t: 'EX·03 streaming attention', h: '/s/kernels#instruments', k: 'flash online softmax carried state' },
  { t: 'EX·08 causal skip', h: '/s/kernels#instruments', k: 'mask loop bound blocks never loaded' },
  { t: 'EX·04 ring attention', h: '/s/distributed#instruments', k: 'ici remote dma longer arrows' },
  { t: 'EX·10 the semaphore timeline', h: '/s/distributed#instruments', k: 'deadlock send wait ring' },
  { t: 'EX·11 the mesh, hop by hop', h: '/s/distributed#instruments', k: 'torus links bandwidth collective' },
].map((x) => ({ kind: 'instrument' as const, title: x.t, href: x.h, hint: 'instrument', keywords: x.k }))

const labs: SearchEntry[] = STAGES.flatMap((s) =>
  s.labs.map((lab) => ({
    kind: 'lab' as const,
    title: `${lab.designator} · ${lab.title}`,
    href: `/s/${s.id}#labs`,
    hint: `lab · ${lab.hardware}`,
    keywords: `colab notebook stage ${s.num}`,
  })),
)

const walks: SearchEntry[] = [
  { t: 'walk · tiled matmul', h: '/s/pallas#walks' },
  { t: 'walk · fused softmax', h: '/s/pallas#walks' },
  { t: 'walk · flash attention, blind build', h: '/s/kernels#walks' },
  { t: 'walk · causal flash', h: '/s/kernels#walks' },
  { t: 'walk · splash mask machinery (production)', h: '/s/kernels#walks' },
].map((x) => ({ kind: 'walk' as const, title: x.t, href: x.h, hint: 'guided walk', keywords: 'code walk step lines' }))

const mistakeEntries: SearchEntry[] = (mistakes as { id: string; title: string }[]).map((m) => ({
  kind: 'mistake',
  title: `mistake · ${m.title}`,
  href: `/mistakes#${m.id}`,
  hint: 'the museum',
  keywords: 'error failure fix pallas',
}))

const drills: SearchEntry[] = [
  { t: 'GYM·01 the dot_general decoder', k: 'dimension numbers contracting axes' },
  { t: 'GYM·02 the shape oracle', k: 'predict output shape dtype' },
  { t: 'GYM·03 spot the decision', k: 'upcast broadcast remat find line' },
  { t: 'GYM·04 the corpus x-ray', k: 'hover sync any program' },
  { t: 'GYM·05 the mosaic x-ray', k: 'pallas kernel tpu dialect mosaic module debug' },
  { t: 'GYM·06 name the transform', k: 'grad vmap scan remat shard_map jaxpr quiz' },
].map((x) => ({ kind: 'drill' as const, title: x.t, href: '/gym#drills', hint: 'the gym', keywords: x.k }))

const ops: SearchEntry[] = Object.keys(opCards as Record<string, string>).map((op) => ({
  kind: 'op',
  title: op,
  href: '/gym#ops',
  hint: 'op reference',
  keywords: 'primitive op card',
}))

const pages: SearchEntry[] = [
  { t: 'the path', h: '/', k: 'home chapters map' },
  { t: 'the gym', h: '/gym', k: 'drills corpus reference' },
  { t: 'the mistake museum', h: '/mistakes', k: 'errors failures' },
  { t: 'the bench', h: '/bench', k: 'numbers provenance records compare' },
].map((x) => ({ kind: 'page' as const, title: x.t, href: x.h, hint: 'page', keywords: x.k }))

export const SEARCH_INDEX: SearchEntry[] = [
  ...chapters,
  ...instruments,
  ...labs,
  ...walks,
  ...mistakeEntries,
  ...drills,
  ...ops,
  ...pages,
]
