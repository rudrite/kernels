// Single source of truth for the track's structure and chapter content.
// CURRICULUM.md at the repo root is the repo-native view of the same track;
// the two are kept in step deliberately.

export type GateState = 'queued' | 'active' | 'passed' | 'failed'

export interface Lab {
  designator: string
  title: string
  time: string
  hardware: 'any machine (interpret mode)' | 'Colab TPU'
  colab: string | null
  notebook?: string
}

export interface Gate {
  state: GateState
  criteria: string[]
  measured: (string | null)[]
}

export interface Reading {
  label: string
  url: string
  note: string
}

export interface WorkBlock {
  label: string
  items: string[]
}

export interface Stage {
  id: string
  num: number
  title: string
  weeks: string
  layer: string
  goal: string
  /** The chapter: named sections of teaching prose; '>>' entries render as pull quotes. */
  chapter: { h: string; ps: string[] }[]
  build: string[]
  readings: Reading[]
  work: WorkBlock[]
  labs: Lab[]
  gate: Gate
  artifact: string
}

const COLAB = 'https://colab.research.google.com/github/rudrite/kernels/blob/main/'

export const STAGES: Stage[] = [
  {
    id: 'machine',
    num: 0,
    title: 'The machine',
    weeks: 'week 1',
    layer: 'tpu',
    goal:
      'Given an op and its shapes, predict from first principles whether it is compute-bound or memory-bound on a given TPU generation, and estimate its ceiling latency.',
    chapter: [
      { h: 'Two speeds, one ridge', ps: [
        'A chip has two speeds: how fast it computes and how fast it feeds. A TPU v5e can do `1.97e14 bf16 FLOPs per second`, but its HBM can only deliver `8.2e11 bytes per second`. Divide the two and you get the ridge: about `240 FLOPs per byte`. Any op that does fewer FLOPs per byte it moves is **memory-bound**, and no amount of compute cleverness will speed it up; any op above the ridge is **compute-bound**, and no amount of bandwidth will. This one division explains most TPU performance mysteries you will ever meet.',
      ] },
      { h: 'The units and the scratchpad', ps: [
        'The compute lives in two units. The MXU is a systolic array (`128x128` on most generations, `256x256` on v6e) built for matmuls; the VPU is a vector unit that handles everything elementwise. Between them and HBM sits VMEM, roughly `128 MiB` of software-managed scratchpad on v5e. Data must be staged in VMEM before compute can touch it, and almost all kernel engineering is choreography of that staging. EX·06 below shows the hierarchy with its real widths.',
      ] },
      { h: 'The lattice rule', ps: [
        'One lattice rule to memorize now, because every kernel in this track obeys it: TPU tiles want their last two dimensions in multiples of `(8, 128)`, and packing doubles the sublane count for narrower dtypes (bf16 packs `(16, 128)`). Blocks that violate the lattice compile badly or not at all.',
      ] },
      { h: 'Predict, then measure', ps: [
        'The habit this stage installs: predict before you measure, every time. Work the arithmetic in EX·01, commit to a number, then run the lab on a real chip and explain every miss. An engineer who predicts within 2x and can name the reason for each miss understands the machine; one who only measures is collecting trivia.',
        '>> Predict before you measure, every time.',
      ] },
    ],
    build: [
      'Roofline estimates for five ops, by hand, reconciled against XProf measurements',
      'A step-time floor estimate for a 7B forward pass from chip constants alone',
      'The working vocabulary: MXU, VPU, VMEM, HBM, the (8, 128) tiling lattice, ICI',
    ],
    readings: [
      { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the backbone text; take chip constants from its tables, not from memory' },
      { label: 'Scaling book · Rooflines', url: 'https://jax-ml.github.io/scaling-book/roofline/', note: 'the reasoning framework this whole stage drills' },
      { label: 'Cloud TPU system architecture', url: 'https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm', note: 'skim for v5e/v6e specifics' },
    ],
    work: [
      {
        label: 'week 1',
        items: [
          'Read both scaling-book chapters in full before touching a notebook',
          'By hand: classify five ops (large matmul, skinny matmul, elementwise add, long-axis softmax, embedding lookup) as compute- or memory-bound per chip generation',
          'Estimate the step-time floor for a 7B forward pass at batch 8, seq 4096 on one v5e from constants alone',
          'Run LAB·0.1 on a Colab TPU and reconcile three estimates against XProf; explain every miss in a sentence',
        ],
      },
    ],
    labs: [
      { designator: 'LAB·0.1', title: 'Rooflines by hand', time: '~2 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-0/lab-0.1-rooflines.ipynb', notebook: 'labs/stage-0/lab-0.1-rooflines.ipynb' },
    ],
    gate: {
      state: 'active',
      criteria: ['Latency predictions for five ops within 2x of measured, every miss explained'],
      measured: [
        'measured on v6e-1 (jax 0.11): big matmul 362.5 µs vs 149.4 µs floor (2.4x); the four µs-scale ops sit 3.6x to 7.7x over floor, dominated by a per-launch overhead the roofline does not model. The remaining work of this gate is writing those explanations up.',
      ],
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
    chapter: [
      { h: 'Refs, blocks, and the grid', ps: [
        'A Pallas kernel is a function over refs: windows into arrays that the runtime has already staged into VMEM. You never write loads or stores against HBM; you write what one grid step does to its window, and two other objects decide everything else. The `BlockSpec` says how an array is carved into blocks and which block a given grid position sees (its index map). The grid says how many steps run and in what order. Keeping "what the step computes" separate from "which data the step sees" in your head from day one is the discipline everything later builds on; the site colors the first copper and the second steel everywhere.',
      ] },
      { h: 'The grid is a pipeline', ps: [
        'The part nobody tells you early enough: on TPU, the grid is **a software pipeline, not a batch of parallel threads**. Step k computes on its blocks while the runtime streams step k+1\'s blocks in behind it. You get double buffering without writing a DMA, but only if your blocks are sized so transfer time hides under compute time. EX·02 plays the pipelined schedule; EX·07 plays the same matmul with the pipeline off, so you can watch the MXU idle between loads. The difference between those two animations is most of single-chip performance engineering.',
      ] },
      { h: 'The carried accumulator', ps: [
        'The other load-bearing pattern is **the carried accumulator**. A reduction axis in the grid means the same output block is revisited once per step along that axis, initialized on the first visit and accumulated into after. Matmul carries C over the K axis here; flash attention will carry `(m, l, acc)` over KV blocks in stage 3. Same shape, bigger algebra.',
      ] },
      { h: 'Reading the layer below', ps: [
        'When a kernel misbehaves, drop one layer and read what Pallas actually built. `pallas_call(..., debug=True)` prints two artifacts at lowering time: the kernel jaxpr (your body, traced against refs) and the Mosaic module, the MLIR the TPU backend receives. Everything you wrote is still visible in it: BlockSpec index maps become `@transform` functions, `pl.when` becomes an `scf.if`, `jnp.dot` becomes `tpu.matmul`, and every vector type wears the shape the `(8, 128)` lattice demands. Chapter 07 walks the layer in depth, the GYM·05 instrument holds three of this stage\'s kernels open at it with hover sync, and LAB·1.4 has you print your own.',
        'The reason to build the habit now: the two hardest exhibits in the museum, the VMEM overflow and the lattice violation, are this layer speaking. An engineer who has read a Mosaic module recognizes "window allocation" as a BlockSpec made physical and "divisible by 8 and 128" as the register tile. One who has not sees noise.',
      ] },
      { h: 'Numbers carry provenance', ps: [
        'And the habit: every measured number states its chip, dtype, shapes, and method (warmup, `block_until_ready`, median of N). Numbers without provenance are noise, and the bench page refuses them.',
      ] },
    ],
    build: [
      'Six kernels in order: add, transpose, tiled matmul, fused softmax, LayerNorm, matmul+GELU',
      'A benchmarking habit: warmup, block_until_ready, median-of-N, chip stated with every number',
      'Annotated readings of production kernels, every line marked algorithm or schedule',
    ],
    readings: [
      { label: 'Pallas quickstart', url: 'https://docs.jax.dev/en/latest/pallas/quickstart.html', note: 'refs, BlockSpecs, grids; read before LAB·1.1' },
      { label: 'Pallas on TPU: pipelining', url: 'https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html', note: 'why the grid is a pipeline; read before week 4' },
      { label: 'Tokamax', url: 'https://github.com/openxla/tokamax', note: 'production kernels as literature; pick one and annotate every line algorithm-or-schedule' },
      { label: 'JAX Pallas TPU ops', url: 'https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu', note: 'the reference kernel library you will diff against in stage 3' },
    ],
    work: [
      {
        label: 'week 2 · mechanics',
        items: [
          'LAB·1.1: add and transpose; internalize refs, index maps, and interpret mode',
          'LAB·1.2 first half: the tiled matmul with a carried accumulator, correctness only',
          'Answer in writing: why is K the innermost grid dim, and what breaks if it is outermost?',
        ],
      },
      {
        label: 'week 3 · fusion and reductions',
        items: [
          'LAB·1.3: fused softmax and LayerNorm; one HBM round-trip each',
          'Blocked rows, unblocked features: write down why the reduction axis must fit the block',
          'Read one Tokamax kernel line by line, margin-marking algorithm vs schedule',
        ],
      },
      {
        label: 'week 4 · pipelining and measurement',
        items: [
          'LAB·1.2 second half: the block-size sweep against XLA on a TPU runtime',
          'LAB·1.4: profile the matmul, find the pipeline in the trace, starve it on purpose',
          'Predict the VMEM budget line from stage 0 constants and confirm where compilation actually fails',
        ],
      },
    ],
    labs: [
      { designator: 'LAB·1.1', title: 'First kernels: add, transpose', time: '~2 h', hardware: 'any machine (interpret mode)', colab: COLAB + 'labs/stage-1/lab-1.1-first-kernels.ipynb', notebook: 'labs/stage-1/lab-1.1-first-kernels.ipynb' },
      { designator: 'LAB·1.2', title: 'Tiled matmul with a carried accumulator', time: '~4 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-1/lab-1.2-tiled-matmul.ipynb', notebook: 'labs/stage-1/lab-1.2-tiled-matmul.ipynb' },
      { designator: 'LAB·1.3', title: 'Fused softmax and LayerNorm', time: '~4 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-1/lab-1.3-fused-softmax-layernorm.ipynb', notebook: 'labs/stage-1/lab-1.3-fused-softmax-layernorm.ipynb' },
      { designator: 'LAB·1.4', title: 'Pipelining and the profile', time: '~3 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-1/lab-1.4-pipelining-profile.ipynb', notebook: 'labs/stage-1/lab-1.4-pipelining-profile.ipynb' },
    ],
    gate: {
      state: 'queued',
      criteria: [
        'Tiled matmul within 15% of XLA for 4096³ bf16',
        'Fused softmax beats the unfused XLA chain at rows of 32k+',
      ],
      measured: [
        'measured on v6e-1: the first schedule ran 1.65x behind XLA; retuned with dimension_semantics and a (2048, 1024, 512) block, 384.6 µs vs 353.0 µs, 1.09x. Inside the bar, and the tuning IS the lesson.',
        'measured on v6e-1: rows=1024 blocks give 222.7 µs vs 379.5 µs unfused, and beat XLA\'s own fusion (267.6 µs). The rows=64 schedule that lost by 27% was the same kernel; only the schedule changed.',
      ],
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
    chapter: [
      { h: 'What your code becomes', ps: [
        'Your Python is not what runs. JAX traces it into a jaxpr (a flat list of primitive ops), lowers that to StableHLO (the portable tensor IR every framework converges on), and hands it to XLA, which makes the performance decisions and emits code for the backend. Each layer is readable, and reading them is a skill worth drilling until it is boring, because the IR states everything Python hides: broadcasts made explicit, dtypes chosen for accumulators, the exact contraction dimensions of every matmul. EX·05 below holds one attention program open at three layers at once; hover any line and its counterparts light up.',
      ] },
      { h: 'Fusion, and its exact limit', ps: [
        'XLA\'s big move is fusion: merging adjacent ops so intermediates stay in fast memory instead of round-tripping through HBM. Fusion is powerful and it has a precise limit: XLA fuses along dataflow edges, but it cannot change the algorithm. Naive softmax needs the row max before any exponential, so the S = QK^T score matrix is computed, spilled, and re-read no matter how well XLA fuses around it. At `seq 8192` in bf16 that spill is `134 MB` written and read back: `268 MB` of HBM traffic that exists purely because the algorithm is multi-pass.',
      ] },
      { h: 'An algorithm failure', ps: [
        'That sentence deserves its own paragraph, because this whole track pivots on it: the spill is not a fusion failure, it is an algorithm failure. No pass in the compiler can fix it; a theorem can, and stage 3 proves that theorem. This stage\'s job is to see the spill in the compiler\'s own output so you never again confuse "the compiler fused it" with "the memory traffic is gone."',
        '>> The spill is not a fusion failure. It is an algorithm failure.',
      ] },
      { h: 'The fluency drill', ps: [
        'The fluency drill: `dot_general` `dimension_numbers`, decoded on sight. Which axes contract, which batch. It feels pedantic for a week and then every IR you ever read becomes legible.',
      ] },
    ],
    build: [
      'A table mapping one program across jaxpr, StableHLO, and optimized HLO',
      'An annotated HLO dump of naive attention: every fusion marked, the HBM spill found and sized',
      'A trace of a real production operator, cataloged against the toy version',
    ],
    readings: [
      { label: 'StableHLO spec', url: 'https://openxla.org/stablehlo/spec', note: 'skim the op list in one sitting; it is smaller than you fear' },
      { label: 'MaxText', url: 'https://github.com/AI-Hypercomputer/maxtext', note: 'pull one real operator and trace it; production jaxprs teach what toys cannot' },
    ],
    work: [
      {
        label: 'week 5',
        items: [
          'LAB·2.1: trace one function through jaxpr, StableHLO, and optimized HLO; build the mapping table',
          'Drill dot_general dimension_numbers until decoding is instant',
          'LAB·2.2: predict the naive-attention spill in bytes, then find it in the TPU dump and confirm within 20% in the profiler',
          'Trace a MaxText operator and catalog what the production jaxpr contains that your toy does not (remat markers, sharding constraints, dtype casts)',
        ],
      },
    ],
    labs: [
      { designator: 'LAB·2.1', title: 'The lowering ladder, traced', time: '~3 h', hardware: 'any machine (interpret mode)', colab: COLAB + 'labs/stage-2/lab-2.1-lowering-ladder.ipynb', notebook: 'labs/stage-2/lab-2.1-lowering-ladder.ipynb' },
      { designator: 'LAB·2.2', title: 'Finding the spill in naive attention', time: '~3 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-2/lab-2.2-finding-the-spill.ipynb', notebook: 'labs/stage-2/lab-2.2-finding-the-spill.ipynb' },
    ],
    gate: {
      state: 'queued',
      criteria: ['Spill-size estimate from the HLO dump matches the profiler within 20%'],
      measured: [
        'measured on v6e-1: 3 fusion ops in the compiled HLO carry the full bf16[8192,8192]; naive attention runs 414.4 µs at seq 8192. The profiler byte-count confirm is still open.',
      ],
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
    chapter: [
      { h: 'The two axes of hard', ps: [
        'Every kernel people speak of with reverence (flash, Splash, ragged paged attention, fused MoE) is hard for one or both of exactly two reasons. Axis one: **algorithmic restructuring**, turning a multi-pass computation into a streaming one, which is a theorem, not a schedule. Axis two: **data-dependent iteration**, where the loop structure itself depends on runtime data: a sparsity mask, a page table, routing decisions. Master both axes on their simplest instances and the priesthood kernels stop being magic; that is this stage.',
      ] },
      { h: 'The softmax monoid', ps: [
        'The central result of axis one is online softmax. The naive form needs the row max before the exponentials, hence the multi-pass shape and stage 2\'s spill. The streaming form carries a triple `(m, l, acc)`: running max, running rescaled denominator, running rescaled output, and folds each new block in with an exact rescale, since `$e^{s-m_{\\text{new}}} = e^{s-m_{\\text{old}}} \\cdot e^{m_{\\text{old}}-m_{\\text{new}}}$`. The deep fact you prove in LAB·3.1 is that the combine operation on these triples is associative and commutative. That single proof is why blocked, streaming, parallel, and (in stage 4) distributed attention are all simultaneously correct. EX·03 plays the schedule with the state updates written as exact formulas.',
        '>> One associativity proof makes blocked, streaming, and distributed attention correct at once.',
      ] },
      { h: 'The backward, streamed', ps: [
        'A fast forward with a slow backward is useless for training, so the backward gets its own week. Two ideas carry it: the identity `$D = \\mathrm{rowsum}(dO \\odot O)$` collapses the softmax Jacobian into something streamable, and recomputing P blockwise from the saved `(m, l)` beats storing it, trading cheap FLOPs for expensive bytes. You wire your forward to a streaming backward with `custom_vjp` and let `jax.grad` referee every step.',
      ] },
      { h: 'Masks become loop structure', ps: [
        'Axis two starts with the observation that a causal mask is not something you add to scores; it is blocks you never visit. In EX·08 the kv loop bound depends on the query block index, so blocks past the diagonal are never loaded at all: the mask became loop structure, and the speedup scales with sparsity. Splash attention is this idea industrialized (arbitrary block masks via scalar prefetch), and ragged paged attention is the same axis with a page table as the data. Read both after you have built the toy; the diffs are the syllabus.',
      ] },
      { h: 'Derive before you read', ps: [
        'One rule for the whole stage: derivation before code, your own build before any reference. The blind-build discipline is not ceremony; reading Splash first would rob you of the exact struggle that makes its choices legible.',
      ] },
    ],
    build: [
      'Online softmax derived on paper from a blank page, associativity proven',
      'Flash attention forward, written from the derivation before reading any reference',
      'The backward pass, derived and wired with custom_vjp, differential-tested',
      'Toy block-sparse and ragged kernels; deep reads of Splash, ragged paged attention, MoE dispatch',
    ],
    readings: [
      { label: 'Splash attention source', url: 'https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention', note: 'only after your blind build passes; diff yours against it and catalog every difference' },
      { label: 'Ragged Paged Attention paper', url: 'https://arxiv.org/pdf/2604.15464', note: 'axis two at production scale: the page table becomes the loop' },
      { label: 'tpu-inference kernels', url: 'https://github.com/vllm-project/tpu-inference', note: 'the serving kernels in production; week 8 reading' },
    ],
    work: [
      {
        label: 'week 6 · flash, derived then built',
        items: [
          'On paper, from blank page: two-pass softmax to the `(m, l)` pair to the `(m, l, acc)` triple; prove the combine associative and commutative',
          'LAB·3.1: check the derivation numerically, including 20 random reduction bracketings',
          'LAB·3.2: the blind build; no references until check passes, then diff against Splash and Tokamax',
        ],
      },
      {
        label: 'week 7 · the backward',
        items: [
          'Derive dQ, dK, dV on paper; find the rowsum identity yourself before confirming it',
          'LAB·3.3: wire custom_vjp, differential-test values and grads against jax.grad',
          'Upgrade the whole-array backward to a blocked streaming pass; corner cases: fully-masked row, length-1 sequence',
        ],
      },
      {
        label: 'week 8 · data shapes the loop',
        items: [
          'LAB·3.4: causal flash where the loop bound is the mask; measure the sparsity scaling',
          'Extend to an arbitrary block mask passed as data; you have now rebuilt Splash\'s core idea',
          'Read the RPA paper and kernel: watch the page table and lengths become loop structure via scalar prefetch',
        ],
      },
    ],
    labs: [
      { designator: 'LAB·3.1', title: 'Deriving online softmax', time: '~4 h', hardware: 'any machine (interpret mode)', colab: COLAB + 'labs/stage-3/lab-3.1-deriving-online-softmax.ipynb', notebook: 'labs/stage-3/lab-3.1-deriving-online-softmax.ipynb' },
      { designator: 'LAB·3.2', title: 'Flash attention, blind build', time: '~6 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-3/lab-3.2-flash-attention-blind.ipynb', notebook: 'labs/stage-3/lab-3.2-flash-attention-blind.ipynb' },
      { designator: 'LAB·3.3', title: 'The backward pass', time: '~6 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-3/lab-3.3-backward-pass.ipynb', notebook: 'labs/stage-3/lab-3.3-backward-pass.ipynb' },
      { designator: 'LAB·3.4', title: 'Skipping blocks: masks as loop structure', time: '~4 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-3/lab-3.4-masks-as-loop-structure.ipynb', notebook: 'labs/stage-3/lab-3.4-masks-as-loop-structure.ipynb' },
    ],
    gate: {
      state: 'queued',
      criteria: [
        'Own flash forward within 1.3x of reference implementations at seq 8192',
        'Differential tests green: 1e-3 forward, 1e-2 grads, bf16',
      ],
      measured: [
        'measured on v6e-1: tuned flash (512, 1024 blocks) beats naive at every length, 1.05x at seq 4k growing to 1.35x at 32k, and beats jax.nn.dot_product_attention 4.15x at 32k. The windowed variant, where the mask is a loop bound, runs 4.87x ahead of XLA\'s masked attention at 16k.',
        'measured on v6e-1, all bf16: forward max |flash − naive| 0.254 (dominated by naive computing its softmax in bf16); grads max diff q 0.163 · k 0.203 · v 0.0 against autodiff. Tightening these against f32 references is open gate work.',
      ],
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
    chapter: [
      { h: 'Collectives are kernels', ps: [
        'A collective is not a primitive; it is a kernel somebody wrote. On a TPU pod, chips along a mesh axis form physical rings, and an all-gather is what it looks like: N−1 hops where every chip pushes one shard to its neighbor. The push is the TPU\'s native distributed move: an async remote DMA that writes directly into the neighbor\'s VMEM and signals a semaphore, leaving the MXU free the whole time. Once you have built one collective from these parts, the opaque names (all-gather, reduce-scatter, all-to-all) resolve into loop shapes you can reason about like any other schedule.',
      ] },
      { h: 'Overlap is the economics', ps: [
        'The economics that make this stage matter at scale: at frontier size, the dominant tax is not slow kernels but **exposed communication**, the fraction of each step where compute units idle waiting on the interconnect. The cure is overlap: transfer shard t+1 while computing on shard t, which is exactly the double-buffering pattern from stage 1 with the arrows crossing chips. EX·04 shows it: the same streaming attention schedule as EX·03 with the KV transport swapped from HBM DMA to remote DMA. Same algebra, longer arrows.',
      ] },
      { h: 'Deadlock, met safely', ps: [
        'With semaphores comes the one genuinely new failure mode of this stage: deadlock. A send waiting on a recv that waits on the send. LAB·4.1 has you build a deadlock on purpose in interpret mode, because meeting this failure for the first time in a real training run is a bad week, and meeting it in a toy is an afternoon.',
      ] },
      { h: 'Composition closes the loop', ps: [
        'The composition payoff closes the loop on the whole track: ring attention is LAB·3.1\'s monoid folded over shards arriving via LAB·4.1\'s ring, and nothing new needs deriving because associativity was proven once. When compositions keep coming out correct because one theorem was proven early, you have learned the track\'s deepest lesson: the famous kernels are few theorems wearing many schedules.',
        '>> The famous kernels are few theorems wearing many schedules.',
      ] },
    ],
    build: [
      'Ring all-gather from raw remote copies, validated against the collective',
      'A deliberate deadlock, observed and explained',
      'A toy ring attention step: the streaming algebra is indifferent to where a block came from',
    ],
    readings: [
      { label: 'Distributed computing in Pallas for TPUs', url: 'https://docs.jax.dev/en/latest/pallas/tpu/distributed.html', note: 'the RDMA model, semaphores, and the ring all-gather this stage rebuilds' },
    ],
    work: [
      {
        label: 'week 9 · the ring',
        items: [
          'LAB·4.1: ring all-gather as ppermute hops, validated exactly against lax.all_gather',
          'On a real slice: the same loop as Pallas remote DMAs, bitwise validated, overlap found in the profile',
          'Break it on purpose: reorder one semaphore wait under interpret mode and study the deadlock',
        ],
      },
      {
        label: 'week 10 · the composition',
        items: [
          'LAB·4.2: ring attention from the stage 3 monoid plus the ring; verify equal to full attention',
          'On a slice: swap hops for remote DMAs, put the flash kernel inside the loop, profile the hop hiding under compute',
          'Note what you did not do: re-derive anything; write the paragraph explaining why causal ring attention is free',
        ],
      },
    ],
    labs: [
      { designator: 'LAB·4.1', title: 'Ring all-gather from remote DMAs', time: '~5 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-4/lab-4.1-ring-all-gather.ipynb', notebook: 'labs/stage-4/lab-4.1-ring-all-gather.ipynb' },
      { designator: 'LAB·4.2', title: 'Ring attention, composed', time: '~5 h', hardware: 'Colab TPU', colab: COLAB + 'labs/stage-4/lab-4.2-ring-attention.ipynb', notebook: 'labs/stage-4/lab-4.2-ring-attention.ipynb' },
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
    chapter: [
      { h: 'A different referee', ps: [
        'Everything before this was practice under your own referee. The capstone changes the referee: a kernel is done when it is correct under a differential suite, competitive against both the XLA floor and the hand-tuned ceiling, and reviewed by people who maintain kernels for a living. **Upstream review is the only test that cannot be gamed**, and what it changes about your code is part of the artifact.',
      ] },
      { h: 'Pick a real gap', ps: [
        'Pick a gap that is real: an attention variant no fast public TPU kernel covers, a block-size autotuning pass for an existing Tokamax kernel with measured wins across a shape sweep, or a JAXBench entry on an operator where published baselines trail the expert bound. Sized right, this is four weeks: one derivation, one forward, one backward, one review cycle.',
      ] },
      { h: 'The write-up is the artifact', ps: [
        'The write-up matters as much as the merge. Derivation, schedule decisions with their measured consequences, the full benchmark record with provenance, and what review changed: that document is the proof that the fourteen weeks produced an engineer, and it closes the track in public, where it started.',
      ] },
    ],
    build: [
      'One kernel closing a real gap: an uncovered attention variant, an autotuning pass, or a benchmark entry',
      'Forward and backward, differential-tested, corner cases enumerated',
      'Benchmarked against the XLA floor and the hand-tuned ceiling, recorded in bench/',
      'Sent upstream, review discussion linked from this site',
    ],
    readings: [
      { label: 'JAXBench', url: 'https://arxiv.org/abs/2607.20466', note: 'the benchmark and the gaps its baselines leave open; capstone target candidates live here' },
      { label: 'Tokamax', url: 'https://github.com/openxla/tokamax', note: 'the natural upstream venue; its contribution surface is the autotuning option' },
    ],
    work: [
      {
        label: 'weeks 11-14',
        items: [
          'Week 11: pick the gap, write the derivation and the plan, open the conversation upstream early',
          'Week 12: forward, differential-tested; first benchmark sweep against floor and ceiling',
          'Week 13: backward via custom_vjp; corner cases enumerated; bench records land in the repo',
          'Week 14: the upstream PR, the review cycle, and the capstone write-up as the track finale',
        ],
      },
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
