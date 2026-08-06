// The site-wide search index, built at compile time from the same sources
// the pages render. One entry per reachable thing; the palette filters it.
import { PATH } from './path'
import { LAYER_DOCS } from '../data/layers'
import { STAGES } from '../data/track'
import { JAX_CHAPTERS } from '../data/jax/track'
import { XLA_CHAPTERS } from '../data/xla/track'
import { PT_CHAPTERS } from '../data/pytorch/track'
import { ALL_UNIT_LESSONS, lessonHref } from '../data/lessons'
import jaxMistakes from '../data/jax-mistakes.json'
import xlaMistakes from '../data/xla-mistakes.json'
import pytorchMistakes from '../data/pytorch-mistakes.json'
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
  { t: 'EX·19 the systolic array, cycle by cycle', h: '/l/tpu/tpu-chip', k: 'mxu weights wavefront fill drain' },
  { t: 'EX·20 the torus, hop by hop', h: '/l/ici/tpu-fabric', k: 'ici wraparound hops link bandwidth' },
  { t: 'EX·21 two fabrics, one question', h: '/l/ici/gpu-fabric', k: 'nvlink nvswitch torus egress' },
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
  href: `/mistakes/kernels#${m.id}`,
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
  { t: 'GYM·07 the fusion x-ray', k: 'before after optimized hlo fusion spill' },
  { t: 'GYM·08 the timeline x-ray', k: 'xprof profile device plane online-softmax copies hidden' },
].map((x) => ({ kind: 'drill' as const, title: x.t, href: '/gym/kernels#drills', hint: 'the gym', keywords: x.k }))

const ops: SearchEntry[] = Object.keys(opCards as Record<string, string>).map((op) => ({
  kind: 'op',
  title: op,
  href: '/gym/kernels#ops',
  hint: 'op reference',
  keywords: 'primitive op card',
}))

const jaxChapters: SearchEntry[] = JAX_CHAPTERS.map((c) => ({
  kind: 'chapter',
  title: `jax ${pad(c.num)} · ${c.title}`,
  href: `/jax/${c.id}`,
  hint: c.part === 'i' ? 'jax path · part i, the model' : 'jax path · part ii, the practice',
  keywords: `jax ${c.lede}`,
}))

const xlaChapters: SearchEntry[] = XLA_CHAPTERS.map((c) => ({
  kind: 'chapter',
  title: `xla ${pad(c.num)} · ${c.title}`,
  href: `/xla/${c.id}`,
  hint: c.part === 'i' ? 'xla path · part i, the compiler' : 'xla path · part ii, the runtime',
  keywords: `xla ${c.lede}`,
}))

const ptChapters: SearchEntry[] = PT_CHAPTERS.map((c) => ({
  kind: 'chapter',
  title: `pt ${pad(c.num)} · ${c.title}`,
  href: `/pytorch/${c.id}`,
  hint: c.part === 'i' ? 'pytorch path · part i, the model' : 'pytorch path · part ii, the practice',
  keywords: `pytorch torch ${c.lede}`,
}))

const newWingMistakes: SearchEntry[] = [
  ...(jaxMistakes as { id: string; title: string }[]).map((m) => ({ wing: 'jax', m })),
  ...(xlaMistakes as { id: string; title: string }[]).map((m) => ({ wing: 'xla', m })),
  ...(pytorchMistakes as { id: string; title: string }[]).map((m) => ({ wing: 'pytorch', m })),
].map(({ wing, m }) => ({
  kind: 'mistake' as const,
  title: `mistake · ${m.title}`,
  href: `/mistakes/${wing}#${m.id}`,
  hint: `the museum · ${wing} wing`,
  keywords: `error failure fix ${wing}`,
}))

const lessonEntries: SearchEntry[] = ALL_UNIT_LESSONS.flatMap((u) =>
  u.lessons.map((l) => ({
    kind: 'chapter' as const,
    title: `lesson · ${l.title}`,
    href: lessonHref(u.unit, l.id),
    hint: u.unit.startsWith('xla:') ? 'the xla path · lesson' : 'the kernel path · lesson',
    keywords: `${u.unit} ${l.lede}`,
  })),
)

const pages: SearchEntry[] = [
  { t: 'the path', h: '/', k: 'home chapters map kernels' },
  { t: 'the jax path', h: '/jax', k: 'jax mastery course transformations' },
  { t: 'the xla path', h: '/xla', k: 'xla mastery course compiler pjrt ifrt pathways' },
  { t: 'the pytorch path', h: '/pytorch', k: 'pytorch mastery course torch tpu torchax' },
  { t: 'the gym', h: '/gym', k: 'drills corpus reference floors' },
  { t: 'the gym · kernels floor', h: '/gym/kernels', k: 'drills corpus ops mosaic' },
  { t: 'the gym · jax floor', h: '/gym/jax', k: 'shape oracle transform corpus x-ray' },
  { t: 'the gym · xla floor', h: '/gym/xla', k: 'spot decision fusion timeline profile' },
  { t: 'the gym · pytorch floor', h: '/gym/pytorch', k: 'stride oracle views contiguity' },
  { t: 'the mistake museum', h: '/mistakes', k: 'errors failures wings' },
  { t: 'the museum · kernels wing', h: '/mistakes/kernels', k: 'pallas errors failures' },
  { t: 'the museum · jax wing', h: '/mistakes/jax', k: 'jax errors tracer immutable' },
  { t: 'the museum · xla wing', h: '/mistakes/xla', k: 'sharding spmd mesh errors' },
  { t: 'the museum · pytorch wing', h: '/mistakes/pytorch', k: 'torch autograd inplace errors' },
  { t: 'the bench', h: '/bench', k: 'numbers provenance records compare' },
].map((x) => ({ kind: 'page' as const, title: x.t, href: x.h, hint: 'page', keywords: x.k }))

export const SEARCH_INDEX: SearchEntry[] = [
  ...chapters,
  ...jaxChapters,
  ...xlaChapters,
  ...ptChapters,
  ...lessonEntries,
  ...newWingMistakes,
  ...instruments,
  ...labs,
  ...walks,
  ...mistakeEntries,
  ...drills,
  ...ops,
  ...pages,
]
