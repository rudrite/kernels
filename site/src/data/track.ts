// Single source of truth for the track's structure. Prose depth lives in
// CURRICULUM.md at the repo root; this model drives every page of the site.

export type GateState = 'queued' | 'active' | 'passed' | 'failed'

export interface Lab {
  designator: string
  title: string
  time: string
  hardware: 'any machine (interpret mode)' | 'Colab TPU'
  /** Colab URL once the repo is public; null renders the honest pending state. */
  colab: string | null
  /** Repo path of the notebook once it exists. */
  notebook?: string
}

export interface Gate {
  state: GateState
  criteria: string[]
  /** Measured results, same order as criteria; null until real records exist. */
  measured: (string | null)[]
}

export interface Stage {
  id: string
  num: number
  title: string
  weeks: string
  /** Which layer of the stack this stage is pinned to on the map. */
  layer: string
  goal: string
  build: string[]
  labs: Lab[]
  gate: Gate
  artifact: string
}

export const STAGES: Stage[] = [
  {
    id: 'machine',
    num: 0,
    title: 'The machine',
    weeks: 'week 1',
    layer: 'tpu',
    goal:
      'Given an op and its shapes, predict from first principles whether it is compute-bound or memory-bound on a given TPU generation, and estimate its ceiling latency.',
    build: [
      'Roofline estimates for five ops, by hand, reconciled against XProf measurements',
      'A step-time floor estimate for a 7B forward pass from chip constants alone',
      'The working vocabulary: MXU, VPU, VMEM, HBM, the (8, 128) tiling lattice, ICI',
    ],
    labs: [
      {
        designator: 'LAB·0.1',
        title: 'Rooflines by hand',
        time: '~2 h',
        hardware: 'Colab TPU',
        colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-0/lab-0.1-rooflines.ipynb',
        notebook: 'labs/stage-0/lab-0.1-rooflines.ipynb',
      },
    ],
    gate: {
      state: 'active',
      criteria: ['Latency predictions for five ops within 2x of measured, every miss explained'],
      measured: ['predictions computed (see bench); measured half pending a TPU run'],
    },
    artifact: 'Explainer: how a TPU actually spends its time',
  },
  {
    id: 'pallas',
    num: 1,
    title: 'Pallas fundamentals',
    weeks: 'weeks 2-4',
    layer: 'pallas',
    goal:
      'Write, benchmark, and profile correct single-chip Pallas kernels; read production kernels fluently.',
    build: [
      'Six kernels in order: add, transpose, tiled matmul, fused softmax, LayerNorm, matmul+GELU',
      'A benchmarking habit: warmup, block_until_ready, median-of-N, chip stated with every number',
      'Annotated readings of production kernels, every line marked algorithm or schedule',
    ],
    labs: [
      { designator: 'LAB·1.1', title: 'First kernels: add, transpose', time: '~2 h', hardware: 'any machine (interpret mode)', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-1/lab-1.1-first-kernels.ipynb', notebook: 'labs/stage-1/lab-1.1-first-kernels.ipynb' },
      { designator: 'LAB·1.2', title: 'Tiled matmul with a carried accumulator', time: '~4 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-1/lab-1.2-tiled-matmul.ipynb', notebook: 'labs/stage-1/lab-1.2-tiled-matmul.ipynb' },
      { designator: 'LAB·1.3', title: 'Fused softmax and LayerNorm', time: '~4 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-1/lab-1.3-fused-softmax-layernorm.ipynb', notebook: 'labs/stage-1/lab-1.3-fused-softmax-layernorm.ipynb' },
      { designator: 'LAB·1.4', title: 'Pipelining and the profile', time: '~3 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-1/lab-1.4-pipelining-profile.ipynb', notebook: 'labs/stage-1/lab-1.4-pipelining-profile.ipynb' },
    ],
    gate: {
      state: 'queued',
      criteria: [
        'Tiled matmul within 15% of XLA for 4096³ bf16',
        'Fused softmax beats the unfused XLA chain at rows of 32k+',
      ],
      measured: ['est. ~1.1x XLA at 4096³ (pending run)', 'est. beats unfused chain (pending run)'],
    },
    artifact: 'Explainer: Pallas from zero, with the algorithm/schedule split color-coded',
  },
  {
    id: 'ir',
    num: 2,
    title: 'The IR stack',
    weeks: 'week 5',
    layer: 'stablehlo',
    goal:
      'Read every representation your code passes through, and find the fusion decisions and their limits with your own eyes.',
    build: [
      'A table mapping one program across jaxpr, StableHLO, and optimized HLO',
      'An annotated HLO dump of naive attention: every fusion marked, the HBM spill found and sized',
      'A trace of a real production operator, cataloged against the toy version',
    ],
    labs: [
      { designator: 'LAB·2.1', title: 'The lowering ladder, traced', time: '~3 h', hardware: 'any machine (interpret mode)', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-2/lab-2.1-lowering-ladder.ipynb', notebook: 'labs/stage-2/lab-2.1-lowering-ladder.ipynb' },
      { designator: 'LAB·2.2', title: 'Finding the spill in naive attention', time: '~3 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-2/lab-2.2-finding-the-spill.ipynb', notebook: 'labs/stage-2/lab-2.2-finding-the-spill.ipynb' },
    ],
    gate: {
      state: 'queued',
      criteria: ['Spill-size estimate from the HLO dump matches the profiler within 20%'],
      measured: ['computed: 268 MB round-trip at seq 8192 bf16; profiler confirm pending'],
    },
    artifact: 'Explainer: what your JAX becomes, one program shown at every layer',
  },
  {
    id: 'kernels',
    num: 3,
    title: 'The priesthood kernels',
    weeks: 'weeks 6-8',
    layer: 'gap',
    goal:
      'Own the two axes that make the famous kernels hard: algorithmic restructuring and data-dependent iteration.',
    build: [
      'Online softmax derived on paper from a blank page, associativity proven',
      'Flash attention forward, written from the derivation before reading any reference',
      'The backward pass, derived and wired with custom_vjp, differential-tested',
      'Toy block-sparse and ragged kernels; deep reads of Splash, ragged paged attention, MoE dispatch',
    ],
    labs: [
      { designator: 'LAB·3.1', title: 'Deriving online softmax', time: '~4 h', hardware: 'any machine (interpret mode)', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-3/lab-3.1-deriving-online-softmax.ipynb', notebook: 'labs/stage-3/lab-3.1-deriving-online-softmax.ipynb' },
      { designator: 'LAB·3.2', title: 'Flash attention, blind build', time: '~6 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-3/lab-3.2-flash-attention-blind.ipynb', notebook: 'labs/stage-3/lab-3.2-flash-attention-blind.ipynb' },
      { designator: 'LAB·3.3', title: 'The backward pass', time: '~6 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-3/lab-3.3-backward-pass.ipynb', notebook: 'labs/stage-3/lab-3.3-backward-pass.ipynb' },
      { designator: 'LAB·3.4', title: 'Skipping blocks: masks as loop structure', time: '~4 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-3/lab-3.4-masks-as-loop-structure.ipynb', notebook: 'labs/stage-3/lab-3.4-masks-as-loop-structure.ipynb' },
    ],
    gate: {
      state: 'queued',
      criteria: [
        'Own flash forward within 1.3x of reference implementations at seq 8192',
        'Differential tests green: 1e-3 forward, 1e-2 grads, bf16',
      ],
      measured: ['est. within 1.3x (pending run)', 'cpu interpret: green at 1e-4 fwd / 1e-5 grads f32; bf16 on-chip pending'],
    },
    artifact: 'Explainers: flash attention derived, and kernels where the data shapes the loop',
  },
  {
    id: 'distributed',
    num: 4,
    title: 'Distributed Pallas',
    weeks: 'weeks 9-10',
    layer: 'ici',
    goal:
      'Ground collectives knowledge in the mechanism: remote DMA and semaphores, compute hiding communication.',
    build: [
      'Ring all-gather from raw remote copies, validated against the collective',
      'A deliberate deadlock, observed and explained',
      'A toy ring attention step: the streaming algebra is indifferent to where a block came from',
    ],
    labs: [
      { designator: 'LAB·4.1', title: 'Ring all-gather from remote DMAs', time: '~5 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-4/lab-4.1-ring-all-gather.ipynb', notebook: 'labs/stage-4/lab-4.1-ring-all-gather.ipynb' },
      { designator: 'LAB·4.2', title: 'Ring attention, composed', time: '~5 h', hardware: 'Colab TPU', colab: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-4/lab-4.2-ring-attention.ipynb', notebook: 'labs/stage-4/lab-4.2-ring-attention.ipynb' },
    ],
    gate: {
      state: 'queued',
      criteria: ['Ring all-gather matches lax.all_gather bitwise, overlap visible in the profile'],
      measured: ['exact on 8 simulated devices; slice profile pending'],
    },
    artifact: 'Explainer: a collective is just a kernel',
  },
  {
    id: 'capstone',
    num: 5,
    title: 'Capstone: shipped upstream',
    weeks: 'weeks 11-14',
    layer: 'pallas',
    goal:
      'Prove the track against production standards: one kernel, end to end, in front of the community.',
    build: [
      'One kernel closing a real gap: an uncovered attention variant, an autotuning pass, or a benchmark entry',
      'Forward and backward, differential-tested, corner cases enumerated',
      'Benchmarked against the XLA floor and the hand-tuned ceiling, recorded in bench/',
      'Sent upstream, review discussion linked from this site',
    ],
    labs: [],
    gate: {
      state: 'queued',
      criteria: [
        'Correct under the differential suite',
        'Beats XLA decisively at target shapes, within striking distance of the hand-tuned ceiling',
        'The upstream submission exists in public',
      ],
      measured: [null, null, null],
    },
    artifact: 'The capstone write-up: derivation, schedule decisions, full benchmark record, what review changed',
  },
]

/** The lowering path, top to bottom. The map is the stack. */
export interface Layer {
  id: string
  label: string
  desc: string
  kind: 'layer' | 'gap' | 'ici'
}

export const LAYERS: Layer[] = [
  { id: 'source', label: 'JAX / PyTorch', desc: 'Math on whole arrays. No notion of hardware.', kind: 'layer' },
  { id: 'jaxpr', label: 'jaxpr', desc: 'The traced program: a flat list of primitive ops.', kind: 'layer' },
  { id: 'stablehlo', label: 'StableHLO', desc: 'The portable tensor IR every framework lowers into.', kind: 'layer' },
  { id: 'xla', label: 'XLA', desc: 'Fuses along dataflow edges. Cannot restructure an algorithm.', kind: 'layer' },
  {
    id: 'gap',
    label: 'the gap',
    desc: 'XLA stops at fusion. Pallas asks for everything by hand. Between them: nothing. Stage 3 teaches you to cross on foot.',
    kind: 'gap',
  },
  { id: 'pallas', label: 'Pallas', desc: 'Hand choreography: BlockSpecs, grids, VMEM residency, DMA.', kind: 'layer' },
  { id: 'mosaic', label: 'Mosaic → LLO', desc: 'The TPU backend. Kernels compile here; you never touch it.', kind: 'layer' },
  { id: 'tpu', label: 'TPU chip', desc: 'MXU, VPU, VMEM, HBM. Where all the time is actually spent.', kind: 'layer' },
  { id: 'ici', label: 'ICI ⇄ chips', desc: 'The inter-chip links. Collectives are kernels here too.', kind: 'ici' },
]

export const stagesForLayer = (layerId: string): Stage[] => STAGES.filter((s) => s.layer === layerId)

export const TAGLINE = 'kernels, derived'
