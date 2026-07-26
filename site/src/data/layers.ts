// The layer tutorials: one in-depth chapter per layer of the stack, ordered
// as a descent. Each teaches what the next layer needs. Code excerpts marked
// fromIR render real generated dumps (data/ir-attention.json), never retyped.

export interface LayerCode {
  caption: string
  /** Literal code text, or a reference into the real IR dumps. */
  text?: string
  fromIR?: 'source' | 'jaxpr' | 'stablehlo'
  lines?: [number, number]
  /** Highlighter language for literal text; defaults to python. */
  lang?: 'python' | 'mlir'
}

export interface LayerSection {
  h: string
  ps: string[]
  code?: LayerCode
}

export interface LayerDoc {
  id: string
  title: string
  lede: string
  sections: LayerSection[]
  /** Stage that drills this layer hands-on. */
  drills?: { stage: string; label: string }
}

export const LAYER_DOCS: LayerDoc[] = [
  {
    id: 'source',
    title: 'JAX / PyTorch',
    lede: 'Where you write math on whole arrays, and what tracing takes away from you.',
    sections: [
      {
        h: 'what this layer is',
        ps: [
          'Everything starts as framework code: `jnp` operations on whole arrays, module trees, Python control flow. The essential fact about this layer is that it is not what runs. When you call `jax.jit`, JAX executes your Python once with tracer objects instead of numbers, records every primitive operation the tracers pass through, and keeps the recording. **Your function becomes data.**',
          'Tracing explains most early JAX surprises. Python control flow that branches on values disappears into whichever branch the trace took. Shapes are fixed at trace time, which is why a new shape triggers a recompile. Print statements run once, at trace time, not per step. None of these are quirks; they are what "your function becomes data" means in practice.',
          'PyTorch reaches this same stack through a different door: TorchTPU and torchax map ATen operations into StableHLO, two layers below. By the time either framework reaches the compiler, the framework identity is gone. That shared convergence point is why everything you learn descending this stack applies to both.',
        ],
        code: {
          caption: 'the running example, used at every layer of this descent: naive attention',
          fromIR: 'source',
        },
      },
      {
        h: 'what to take down the stack',
        ps: [
          'Carry one question downward: which of these Python lines will still be visible three layers below, and in what form? The matmuls survive. The softmax dissolves into five primitives. The broadcasting that Python hid becomes explicit. The next layer, the jaxpr, is where you first see all of that written down.',
        ],
      },
    ],
    drills: { stage: 'ir', label: 'Stage 2 traces real programs through this layer' },
  },
  {
    id: 'jaxpr',
    title: 'jaxpr',
    lede: 'The traced program: every primitive op written down, every hidden decision made visible.',
    sections: [
      {
        h: 'reading a jaxpr',
        ps: [
          'A jaxpr is the recording tracing produced: a flat list of equations in single-assignment form. Each line binds one fresh variable to one primitive applied to earlier variables, with every shape and dtype stated. There is no control flow left, no module structure, no Python. It reads like assembly for tensor math, and it is the most honest description of your program you can get without leaving Python: print it with `jax.make_jaxpr(fn)(*args)`.',
          'Here is our attention function as a real jaxpr. Things Python never showed you are now explicit: `k.T` became a transpose equation, the matmuls became `dot_general` with their contraction dimensions spelled out, and the broadcast that aligned the row max for subtraction became a `broadcast_in_dim` with stated shapes.',
        ],
        code: {
          caption: 'the real jaxpr of naive attention (generated, not retyped): first half',
          fromIR: 'jaxpr',
          lines: [0, 13],
        },
      },
      {
        h: 'the detail worth staring at',
        ps: [
          'Look at the `convert_element_type` pair around the `reduce_sum` in the second half: JAX upcasts the bf16 summation to f32 and converts back. Nobody wrote that. It is a numerics decision the framework made for you, and the jaxpr is the first place it becomes visible. Learning to read IRs is learning to notice decisions like this one, because one layer further down, you will be the one making them.',
        ],
        code: {
          caption: 'second half: the f32 upcast around the sum, then the normalize and output matmul',
          fromIR: 'jaxpr',
          lines: [14, 27],
        },
      },
    ],
    drills: { stage: 'ir', label: 'Stage 2 drills jaxpr reading until it is boring' },
  },
  {
    id: 'stablehlo',
    title: 'StableHLO',
    lede: 'The portable tensor IR: where JAX, PyTorch, and every tool downstream agree on vocabulary.',
    sections: [
      {
        h: 'what changes from the jaxpr',
        ps: [
          'StableHLO is the jaxpr\'s information re-expressed in MLIR form, versioned and stable so that different tools and framework versions can exchange it. Same ops, same shapes, now written as a module with a function, in the syntax the whole compiler ecosystem shares. Print it with `jax.jit(fn).lower(*args).as_text()`.',
          'The word MLIR will follow you from here down, so pin it now: MLIR is not a language, it is a framework for defining IRs (called dialects) and transformations over them. StableHLO is a dialect. Mosaic, two layers down, is a dialect. When you read that something is "an MLIR pass," it means one bounded rewrite applied to one of these dialects.',
        ],
        code: {
          caption: 'the same attention, real StableHLO: note dot_general\'s contracting_dims, stated inline',
          fromIR: 'stablehlo',
          lines: [1, 9],
        },
      },
      {
        h: 'the fluency drill',
        ps: [
          'One skill separates people who can read tensor IRs from people who squint at them: decoding `dot_general` `dimension_numbers` on sight. Which axes contract, which axes batch. Drill it here where the notation is friendly, because every matmul you ever chase through a compiler dump is one of these.',
          'A discipline this layer rewards: when something surprises you in a StableHLO dump, resist the urge to guess. The dump is complete; the answer is in it. The EX·05 instrument on Stage 2 holds this exact program open at three layers with hover sync, which makes the correspondence physical.',
        ],
      },
    ],
    drills: { stage: 'ir', label: 'Stage 2, with the EX·05 x-ray as the companion instrument' },
  },
  {
    id: 'xla',
    title: 'XLA',
    lede: 'The compiler: brilliant at fusion, structurally unable to change your algorithm.',
    sections: [
      {
        h: 'what XLA does with your program',
        ps: [
          'XLA takes StableHLO and makes the performance decisions: which ops fuse into one kernel, what layouts arrays get, how work schedules across the chip. Its central optimization is fusion: merging adjacent elementwise chains, reductions, and data movement so intermediates stay in fast memory instead of round-tripping through HBM. On most programs it is very good, which is why the honest bar for any hand-written kernel is "beats XLA," not "beats naive."',
          'Print what it actually decided with `jax.jit(fn).lower(*args).compile().as_text()`. The dump is backend-specific: run it on the TPU runtime to see TPU decisions. Inside you will find fusion ops with their operands: those are the compiler\'s actual choices, not documentation about them.',
        ],
      },
      {
        h: 'the ceiling, precisely',
        ps: [
          'Fusion has an exact limit: XLA merges along dataflow edges, but it **cannot apply algebraic identities**. It cannot notice that a two-pass softmax could become a one-pass streaming computation, because that is a theorem about exponentials, not a graph transformation. So in naive attention, the seq x seq score matrix is computed, written to HBM, and read back, no matter how well everything around it fuses. At `seq 8192` in bf16 that is `134 MB` written and read: `268 MB` of traffic that exists because the algorithm is multi-pass.',
          'Say the conclusion precisely, because the whole stack turns on it: the spill is not a fusion failure, **it is an algorithm failure**. No compiler pass fixes it. A theorem fixes it, and a human currently has to apply that theorem by hand, one layer down, on the other side of the gap.',
          'One update from this site\'s own bench: XLA:TPU now pattern-matches this exact program and dispatches a hand-written online-softmax custom kernel; the profiler shows it by name, and GYM·08 holds the trace. That is recognition of a famous shape, not derivation of the rewrite: perturb the pattern and the ceiling returns. Chapter 05 states what follows.',
        ],
      },
    ],
    drills: { stage: 'ir', label: 'Stage 2\'s LAB·2.2 hunts the spill in XLA\'s own output' },
  },
  {
    id: 'gap',
    title: 'the gap',
    lede: 'Between the compiler\'s ceiling and the kernel language\'s floor, there is no middle layer.',
    sections: [
      {
        h: 'the two modes',
        ps: [
          'Above the gap: XLA, where the compiler decides everything and you cannot intervene. Below it: Pallas, where you decide everything: block shapes, memory residency, pipeline depth, all of it, tangled together with your math. There is no dial between the modes. Either you accept the compiler\'s ceiling, or you take over completely.',
          'This is why the famous kernels exist as hand-written artifacts maintained by a small number of specialists. Flash attention is the online-softmax theorem plus expert memory choreography, welded together in one file. Every kernel like it is a crossing of this gap on foot: someone applied an algebraic identity the compiler could not, then hand-scheduled the result.',
          'The track teaches you to make that crossing yourself. Stage 2 shows you the ceiling from above, in the compiler\'s own dumps. Stage 3 hands you the theorem and has you build the crossing: derive online softmax, prove its combine associative, then write the streaming kernel the compiler could not produce. After you have crossed once, the priesthood kernels read as engineering rather than magic.',
          'A field note from this site\'s own profiler, and it sharpens the claim rather than weakening it: on current XLA:TPU, the device timeline shows the compiler recognizing naive attention and dispatching its own hand-written online-softmax kernel (GYM·08 has the trace). Read the distinction exactly. Pattern recognition ships crossings humans already built; it does not derive new ones. For the shapes the matcher knows, the compiler now carries you over the gap. For everything else, and for every variant one mutation away, the gap is where it always was.',
        ],
      },
    ],
    drills: { stage: 'kernels', label: 'Stage 3 is the crossing' },
  },
  {
    id: 'pallas',
    title: 'Pallas',
    lede: 'The kernel language: you write what one grid step does; BlockSpecs decide what it sees.',
    sections: [
      {
        h: 'the mental model',
        ps: [
          'A Pallas kernel is a Python function over refs: windows into arrays that the runtime has already staged into VMEM. You never load from HBM yourself. The `BlockSpec` for each array says how it is carved into blocks and which block a given grid position sees, via an index map. The grid says how many steps run. Three objects, and the separation between them is the discipline this whole site color-codes: the kernel body is algorithm (copper), the specs and grid are schedule (steel).',
        ],
        code: {
          caption: 'the complete mental model in one kernel: LAB·1.2\'s tiled matmul',
          text: `def matmul_kernel(a_ref, b_ref, o_ref):      # algorithm
    k = pl.program_id(2)
    @pl.when(k == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)
    o_ref[...] += jnp.dot(a_ref[...], b_ref[...])

pl.pallas_call(                               # schedule
    matmul_kernel,
    grid=(M // bm, N // bn, K // bk),
    in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
    out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
    out_shape=jax.ShapeDtypeStruct((M, N), dtype),
)`,
        },
      },
      {
        h: 'the pipeline you get for free, conditionally',
        ps: [
          'On TPU the grid is a software pipeline: while step k computes, the runtime streams step k+1\'s blocks into the other half of a double buffer. You wrote no DMA, yet transfers overlap compute. The condition: your blocks must be sized so transfer time hides under compute time, and the whole working set (roughly three blocks in flight per array) must fit VMEM. Stage 1\'s EX·02 and EX·07 instruments play the pipeline on and off so the difference is visible rather than asserted.',
          'Two practical notes that save days: `interpret=True` runs kernel logic anywhere, including your laptop, which is how every lab in this track verifies before touching a chip; and the `(8, 128)` tiling lattice from the chip layer governs your block shapes, so violations fail at compile time with errors that make sense only after you know the lattice exists.',
        ],
      },
    ],
    drills: { stage: 'pallas', label: 'Stage 1 is three weeks inside this layer' },
  },
  {
    id: 'mosaic',
    title: 'Mosaic → LLO',
    lede: 'The TPU backend: Mosaic you can read, LLO you cannot, and the boundary between them drawn exactly.',
    sections: [
      {
        h: 'what happens below pallas_call',
        ps: [
          'Pallas does not execute your kernel; it lowers it to Mosaic, an MLIR dialect for TPU kernels. Mosaic maps your block operations onto the chip\'s real units: matmuls onto the MXU\'s systolic array, elementwise work onto the VPU\'s lanes, and everything onto the `(8, 128)` register tiling. Below Mosaic sits LLO, the TPU\'s near-assembly, where operations become VLIW bundles the sequencer issues.',
          'The two halves have different visibility, and knowing where the line falls matters. Mosaic\'s front half is open: the `tpu` dialect and much of its lowering live in the jax repo under `jaxlib/mosaic`, and any kernel prints its own module with `pallas_call(..., debug=True)`. LLO is not: it ships closed inside `libtpu`, and the public pipeline ends where Mosaic hands the module over. So the honest division of labor is: you write Pallas, you read Mosaic, and you never see LLO.',
          'Reading Mosaic is worth the hour it takes, because everything you wrote is still recognizable, just compiled. Your BlockSpec index maps become literal functions in the module. Your `pl.when` becomes a real `scf.if` with both branches present. Your `jnp.dot` becomes `tpu.matmul` with the dimension numbers carried down intact. The GYM·05 instrument holds three track kernels open at this layer with hover sync, the same way GYM·04 does for the layers above.',
          'You will also read Mosaic\'s complaints. When a block shape violates the lattice, when a dynamic slice defeats vectorization, when VMEM overflows: the error surfaces from this layer, in this layer\'s vocabulary of layouts and vector shapes. The reason this track teaches the machine (stage 0) before the kernel language (stage 1) is exactly so those errors read as information rather than noise.',
        ],
        code: {
          caption: 'the running matmul kernel at this layer (captured from Pallas debug output, jax 0.4.38): the dot is now the MXU op, and a BlockSpec index map is now a function',
          lang: 'mlir',
          text: `%6 = tpu.matmul %4, %5, %cst {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>} : vector<256x256xbf16>, vector<256x256xbf16>, vector<256x256xf32> -> vector<256x256xf32>
%7 = arith.truncf %6 : vector<256x256xf32> to vector<256x256xbf16>

// later in the same module: lambda i, j, kk: (kk, j), compiled
func.func @transform_1(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  return %arg2, %arg1 : i32, i32
}`,
        },
      },
      {
        h: 'the design fact to carry down',
        ps: [
          'One thing this layer explains: why TPU kernel scheduling is tractable at all. TPU grid steps are a sequential pipeline, not thousands of independent thread blocks, so questions like "will this transfer hide under this compute" have answers you can reason about from constants. The whole roofline discipline of stage 0 works because the hardware below is this predictable.',
        ],
      },
    ],
    drills: { stage: 'pallas', label: 'Stage 1\'s LAB·1.4 provokes and reads this layer\'s limits' },
  },
  {
    id: 'tpu',
    title: 'The machines',
    lede: 'Two compute units, one scratchpad, one far memory. Every performance question resolves here.',
    sections: [
      {
        h: 'the four numbers',
        ps: [
          'A TPU v5e: `1.97e14 bf16 FLOPs per second` in the MXU, `8.2e11 bytes per second` from HBM, roughly `128 MiB` of VMEM scratchpad, `16 GB` of HBM. Every kernel decision in this track is arithmetic over these four numbers. The ratio of the first two is the ridge (about `240 FLOPs per byte`): ops below it are memory-bound, ops above it compute-bound, and no cleverness moves an op across the ridge except changing its algorithm.',
          'The MXU is a `128x128` systolic array (`256x256` on v6e): matmuls only. The VPU handles everything elementwise in `(8, 128)` lanes, which is where the tiling lattice that haunts every layer above actually comes from. VMEM is software-managed: nothing is cached for you; what is resident is what your schedule staged. Stage 0\'s EX·06 draws the hierarchy to these real widths, and EX·01 lets you drag op shapes across the roofline.',
          'The habit the whole track builds on this layer: predict from the constants before measuring, then explain every miss. The constants are public, the arithmetic is short, and an engineer who cannot predict a latency floor from them is guessing everywhere above.',
        ],
      },
    ],
    drills: { stage: 'machine', label: 'Stage 0 lives on this layer' },
  },
  {
    id: 'ici',
    title: 'ICI ⇄ chips',
    lede: 'The inter-chip links: where collectives stop being primitives and become kernels again.',
    sections: [
      {
        h: 'the physical picture',
        ps: [
          'Chips in a TPU pod connect to their neighbors over ICI links (`4.5e10 bytes per second` each way per link on v5e), and slices along a mesh axis form physical rings. Every collective you have ever called resolves to movements over these links: an all-gather is N−1 hops around the ring, each chip pushing one shard to its neighbor; a reduce-scatter is the same ring with an add at each hop.',
          'The TPU\'s native distributed operation is the async remote DMA: a chip pushes a buffer directly into a neighbor\'s VMEM and signals a semaphore, while its compute units keep working. That last clause is the entire economics of distributed kernels: transfers that hide behind compute are free, and transfers that do not are the dominant tax at scale. The craft is arranging the overlap.',
          'The deep continuity with everything above: **sharding decides where data lives, and a kernel is always a per-device program** on its local shard. The streaming-attention algebra from stage 3 does not care whether a KV block arrived from local HBM or from the neighbor chip, which is why ring attention (stage 4) is a composition, not a new derivation. EX·04 shows it: the same schedule as EX·03, longer arrows.',
        ],
      },
    ],
    drills: { stage: 'distributed', label: 'Stage 4 builds a collective from raw remote DMAs' },
  },
]

export const layerDoc = (id: string): LayerDoc | undefined => LAYER_DOCS.find((l) => l.id === id)
