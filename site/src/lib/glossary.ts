// The site-wide glossary: every term here gets a dotted underline and a
// hover popover (definition plus the chapter that teaches it) wherever it
// appears in prose rendered through Rich. First occurrence per paragraph
// only, so pages read highlighted rather than noisy.

export interface GlossaryTerm {
  /** Exact-case pattern; longest terms are matched first. */
  t: string
  /** One-sentence definition, plain text only. */
  d: string
  /** The chapter that teaches it. */
  href: string
  /** Extra regex alternates (plurals, spellings). */
  alt?: string[]
}

export const GLOSSARY: GlossaryTerm[] = [
  { t: 'input_output_aliases', d: 'Donates an input buffer to an output so the kernel updates it in place instead of allocating a copy.', href: '/l/pallas' },
  { t: 'dimension_semantics', d: 'A per-grid-axis promise to the compiler: "parallel" means steps can reorder, "arbitrary" means order matters, as on an accumulation axis.', href: '/l/pallas' },
  { t: 'scratch_shapes', d: 'Requests per-invocation workspace (VMEM buffers, semaphores) that belongs to the kernel itself, not to any input or output, and persists across grid steps.', href: '/l/pallas' },
  { t: 'scalar prefetch', d: 'Scalar arrays land in SMEM before the pipeline starts, so index maps themselves can read data. The mechanism behind splash and ragged attention.', href: '/l/pallas' },
  { t: 'streaming multiprocessor', d: 'One of a GPU’s many cores; each keeps many warps resident and switches among them to hide memory latency.', href: '/l/tpu', alt: ['streaming multiprocessors'] },
  { t: 'register pressure', d: 'How many registers each GPU thread claims; it caps how many warps fit on a core, and with them the latency-hiding budget.', href: '/l/tpu' },
  { t: 'dot_general', d: 'The general matmul primitive. Its dimension_numbers state which axes contract and which batch; decoding them on sight is the core IR-reading skill.', href: '/l/stablehlo' },
  { t: 'custom_call', d: 'HLO’s escape hatch for work outside the op set; a pallas_call rides through the compiler as one, carrying its Mosaic payload.', href: '/l/stablehlo', alt: ['custom-call'] },
  { t: 'pallas_call', d: 'The entry point that runs a kernel body over a grid, with BlockSpecs deciding what each step sees.', href: '/l/pallas' },
  { t: 'interpret=True', d: 'Runs kernel logic on any machine for correctness work. It ignores memory spaces and never sees compile-time errors like lattice or VMEM violations.', href: '/l/pallas' },
  { t: 'index map', d: 'The BlockSpec function that returns block coordinates (not element offsets) for each grid step; Pallas multiplies by the block shape to find elements.', href: '/l/pallas', alt: ['index maps', 'index_map'] },
  { t: 'BlockSpec', d: 'How one array is carved for the grid: a block shape plus an index map saying which block each grid step sees.', href: '/l/pallas', alt: ['BlockSpecs'] },
  { t: 'remote DMA', d: 'A chip pushes a buffer straight into a neighbor’s memory and signals a semaphore, while its compute keeps working. The native distributed operation.', href: '/l/ici', alt: ['remote DMAs'] },
  { t: 'semaphore', d: 'The counter a DMA signals on completion and a kernel waits on; the synchronization primitive under every transfer.', href: '/l/ici', alt: ['semaphores'] },
  { t: 'StableHLO', d: 'The portable, versioned tensor IR that JAX and PyTorch both lower into; chapter 03 reads it line by line.', href: '/l/stablehlo' },
  { t: 'jaxpr', d: 'The traced program: one equation per primitive in single-assignment form, every shape and dtype stated.', href: '/l/jaxpr', alt: ['jaxprs'] },
  { t: 'Mosaic', d: 'The MLIR dialect Pallas lowers to, and the last layer of the TPU stack you can read; only LLO below it is closed.', href: '/l/mosaic' },
  { t: 'fusion', d: 'Several ops compiled into one kernel so intermediates stay in fast memory instead of round-tripping through HBM. XLA’s central optimization, with an exact limit.', href: '/l/xla', alt: ['fusions'] },
  { t: 'roofline', d: 'The floor model: latency is at least the larger of FLOPs over peak compute and bytes over bandwidth. Predict first, measure second.', href: '/l/tpu', alt: ['rooflines'] },
  { t: 'ridge', d: 'The FLOP-per-byte ratio where an op flips from memory-bound to compute-bound: about 240 on v5e, about 575 on v6e.', href: '/l/tpu' },
  { t: 'occupancy', d: 'How many warps stay resident on a GPU core; the currency of GPU latency hiding, the way pipeline depth is on TPU.', href: '/l/tpu' },
  { t: 'VMEM', d: 'The TPU’s software-managed vector scratchpad, about 128 MiB. Blocks must be staged here before compute touches them; what is resident is what your schedule staged.', href: '/l/tpu' },
  { t: 'SMEM', d: 'Scalar memory: lengths, flags, and indices live here, feeding control flow without ever entering the vector datapath.', href: '/l/tpu' },
  { t: 'HBM', d: 'The chip’s main memory: large, far, and the resource memory-bound ops spend. 8.2e11 bytes per second on v5e, 1.6e12 on v6e.', href: '/l/tpu' },
  { t: 'MXU', d: 'The systolic matmul array: 128x128 on v5e, 256x256 on v6e. Matmuls only; everything else is the VPU’s job.', href: '/l/tpu' },
  { t: 'VPU', d: 'The vector unit for elementwise work, organized as (8, 128) lanes; the origin of the tiling lattice every layer above obeys.', href: '/l/tpu' },
  { t: 'ICI', d: 'The inter-chip links (4.5e10 bytes per second each way per link on v5e); every collective resolves to hops over these.', href: '/l/ici' },
  { t: 'LLO', d: 'The TPU’s near-assembly, closed inside libtpu. The readable world ends one layer above, at Mosaic.', href: '/l/tpu/vliw-bundles-and-llo' },
  { t: 'XLA', d: 'The compiler: brilliant at fusing along dataflow edges, structurally unable to change your algorithm. That gap is why kernels exist.', href: '/l/xla' },
  { t: 'DMA', d: 'An asynchronous copy between memories that runs while compute continues; the grid pipeline is DMAs the runtime writes for you.', href: '/l/pallas', alt: ['DMAs'] },
  { t: 'ANY', d: 'The memory space that tells the compiler not to place a ref at all: a promise that the kernel will move the data itself with a manual DMA.', href: '/l/pallas' },
  { t: 'warp', d: '32 GPU threads scheduled as one unit; the GPU hides latency by switching among resident warps rather than by pipelining a scratchpad.', href: '/l/tpu', alt: ['warps'] },
  { t: 'pl.ds', d: 'A dynamic slice of a ref inside the kernel: reads only the chunk, at an offset that can depend on runtime values.', href: '/l/pallas' },
]
