// Pallas lessons: the language in depth, homed in the pallas layer chapter
// (the design argument and refs), the pallas stage (the practice of carves,
// grids, and scalars), and the mosaic layer (the lowering). Repo captures
// quoted are jax 0.4.38 with debug=True; authored snippets ran under
// interpret mode before shipping.
import type { UnitLessons } from './index'

export const PALLAS_LESSONS: UnitLessons[] = [
  {
    unit: 'l:pallas',
    lessons: [
      {
        id: 'why-a-kernel-language',
        num: 1,
        work: [
          { id: 'restate', label: 'read the design doc motivation end to end and restate the Triton argument in two sentences' },
        ],
        title: 'Why a kernel language lives inside JAX',
        lede: 'XLA already had an escape hatch. It was written in C++, and almost nobody could use it.',
        goal:
          'Argue from the design document why a tile-level kernel language belongs inside JAX rather than beside it, and say what Pallas refuses to compile and why the refusal is a position rather than a gap.',
        sections: [
          {
            h: 'the hatch with a C++ door',
            ps: [
              'XLA compiles most machine-learning programs well, and then some program hits its limit. The documented way out has been `CustomCall`: hand the compiler an opaque function you wrote yourself. The design document is blunt about what that costs. Writing one means writing C++, and on GPU it means learning CUDA, which the authors call arguably too low-level for many machine learning GPU kernels, matrix multiplication among them. Their reading is that even expert users have trouble implementing an efficient matmul or multi-headed attention that way.',
              'There is a second motive listed right next to the first, and it is about time rather than difficulty. Advances in systems research take a while to land inside XLA, and people want to run ahead of the compiler. A hand-written kernel is how you run ahead; later the compiler absorbs whatever the kernel proved out. So the escape hatch is not only for the cases XLA will never cover. It is also for the cases XLA has not covered yet.',
              'The kernel path already had you writing Pallas by week two, at /s/pallas, without ever asking why the language exists in this shape. That question is this lesson.',
            ],
          },
          {
            h: "Triton's lesson, and the wall at the TPU",
            ps: [
              'Triton changed the terms of the argument by showing that a kernel could be written as array programming instead of thread programming. You describe what happens to a tile, the compiler handles the parallelism underneath, and the result is good enough that Triton became the primary code generation route for `torch.compile` through Torch Inductor. The design document treats this as settled evidence, not as a competitor to be argued with.',
              'The obvious next move would be to point Triton at a TPU, and the document says exactly why that does not work. Triton exposes a TPU-like model already, programs written for tiles of arrays in L1 cache, but it is specialized enough to GPU that it cannot be compiled directly for TPU. The example given is atomics: Triton offers atomic operations meant to handle parallel writes, and parallel writes are not what a TPU does. So the tile-level model needed a front end one level above Triton, abstract enough that a single kernel could reach two very different machines.',
            ],
          },
          {
            h: 'JAX was already the front end',
            ps: [
              'Having ruled out the C++ hatch and the Triton port, the document asks its own question and answers it in one sentence. The open question, as written: is JAX a good fit for a kernel language at all?',
              '>> Triton demonstrates that an array programming language can be practical for writing GPU kernels and JAX is just that.',
              'That line, from the Pallas design document, is the whole argument compressed. JAX is a mature tracing front end for numerical computing, its users already write NumPy-style array code, and its transformations are the reason people use it. Pallas is then described as three extensions and nothing more: `Ref` types so you can talk about memory, a handful of new primitives like `program_id`, and `pallas_call` to run the body over a grid.',
              'Reusing tracing buys something an AST-parsing front end cannot offer. Your kernel is ordinary Python at trace time, so closures, higher-order functions, and any templating you can express in Python all work, and the document says outright that this makes Pallas far more amenable to templating than Triton. A kernel factory is a function that returns a function. Nothing in the toolchain needs to know it happened.',
            ],
            code: {
              caption: 'templating by closure; both kernels run under interpret=True (verified, jax 0.4.38)',
              lang: 'python',
              text: "def make_eltwise_add(eltwise):\n    def kernel(x_ref, y_ref, o_ref):\n        o_ref[...] = eltwise(x_ref[...] + y_ref[...])\n    return kernel\n\ndoubled = make_eltwise_add(lambda v: v * 2)\nexponentiated = make_eltwise_add(jnp.exp)   # two kernels, one source",
            },
          },
          {
            h: 'the primitives it will not lower',
            ps: [
              'Pallas accepts a subset of JAX primitives, and the design document names two exclusions with their reasons attached. On `conv_general`: convolution usually is not offered as a primitive in the underlying hardware. On `gather/scatter`: the underlying compiler may not support noncontiguous memory reads and writes. Both reasons point at the machine, not at the implementation calendar.',
              'Read as a gap, that list looks like work someone has not gotten to. Read as a position, it says something firmer: a kernel language whose substrate is tiles and contiguous moves should not offer you an operation that hides an arbitrary access pattern. The gather you actually need does not disappear, it changes form. It becomes an index map fed by prefetched scalars, which is a schedule you can look at and cost, and the scalar-world lesson at /s/pallas/scalar-world is that mechanism in full.',
              'The TPU reference applies the same position at finer grain. Integer reductions are unsupported. Elementwise operations carry a published cost ranking where `jnp.sin` and `jnp.cos` sit in the expensive tier and `jnp.exp` in the middle. Loop primitives get fully unrolled during compilation, so a large trip count is a compile-time problem. None of that reads like a feature backlog. It reads like a language declining to pretend the hardware is uniform.',
            ],
          },
          {
            h: 'what the narrowness buys',
            ps: [
              'A language that refuses things can promise things. The TPU page states the promise plainly: while the features are experimental, a kernel accepted by the compiler must return the expected results, and if your outputs look wrong the instruction is to compare against a run with `interpret=True` and file a bug report. Correctness is not negotiable in exchange for the experimental label; expressiveness is.',
              'The cost is that everything the compiler used to decide is now yours to state. Which bytes are resident, in what order, in which memory space, with which axis of the grid reused. The next five pages are those decisions one at a time, and each of them is a decision only because the language declined to guess.',
            ],
          },
        ],
        readings: [
          {
            label: 'Pallas design document',
            url: 'https://docs.jax.dev/en/latest/pallas/design/design.html',
            note: 'the argument this lesson is built on; read the motivation section first',
          },
          {
            label: 'Writing TPU kernels with Pallas',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/details.html',
            note: 'the accepted-means-correct promise, and the op cost table behind it',
          },
          {
            label: 'Triton',
            url: 'https://triton-lang.org/main/index.html',
            note: 'the lineage; skim enough to see what a tile-level GPU language looks like',
          },
        ],
      },
      {
        id: 'memory-spaces-and-scratch',
        num: 2,
        title: 'Memory spaces and scratch',
        lede: 'Four places a ref can live, and one of them is a promise rather than a placement. Scratch is the fifth thing: memory that belongs to no input at all.',
        goal: 'Choose a memory space per argument deliberately, and reach for scratch with the right dtype when partial sums must survive a grid axis.',
        sections: [],
        guide: { id: 'pallas', sections: [1] },
        readings: [
          { label: 'TPU pipelining: memory spaces', url: 'https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html', note: 'the enum-to-hardware mapping behind this lesson' },
          { label: 'Scalar prefetch and block-sparse computation', url: 'https://docs.jax.dev/en/latest/pallas/tpu/sparse.html', note: 'where SMEM refs start carrying schedules' },
        ],
        check: [
          {
            q: 'Who copies a VMEM-staged block in, and when?',
            a: 'The automatic pipeline: the compiler copies the block your BlockSpec describes into on-chip vector memory before the kernel body runs, overlapped against the neighboring steps.',
          },
          {
            q: 'What is scratch memory tied to, and what is the classic use?',
            a: 'Nothing: it belongs to no input or output. Requested through scratch_shapes, it lives for the whole invocation, and the classic use is an f32 accumulator carried across a reduction axis.',
          },
          {
            q: 'Why accumulate in f32 scratch when the data is bf16?',
            a: 'Accumulating directly in bf16 rounds every partial sum to bf16 before the next add, and the rounding compounds. Ordinary floating point behavior, not a Pallas quirk.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'refs',
        num: 3,
        work: [
          { id: 'jaxpr', label: 'print a kernel jaxpr of your own and count the loads against your mental tally' },
        ],
        title: 'Refs, and why mutation entered a functional language',
        lede: 'JAX spent years teaching you that arrays are values. A kernel needs memory you can write into, and Refs are how both stay true.',
        goal:
          'Predict, for any kernel argument, what type the body sees, which memory space it lives in, which operations on it will refuse to trace, and what an aliased buffer means for your write order.',
        sections: [
          {
            h: 'a thing you cannot pass anywhere',
            ps: [
              'Hand a `Ref` to `jnp.exp` and it will not trace. The design document states the rule as a property of the type: refs cannot be passed into the usual set of JAX primitives without being read from first. Read one and you get a JAX `Array` back. Write into one and what you write must be an `Array`. Those two sentences are the entire contract, and every kernel you have written obeys them whether or not you noticed.',
              'A read is not notation. On TPU it means loading into vector registers, on GPU into the lowest level of the hierarchy, and a write is the same move in reverse. So `x_ref[...] + y_ref[...]` is two loads and an add, in that order, and the kernel jaxpr shows all three.',
              'This site\'s own capture makes it visible. The binders are memrefs, not arrays, and each `<-` is a real transfer between memory and registers.',
            ],
            code: {
              caption: 'the add kernel\'s jaxpr, captured on this repo with debug=True (jax 0.4.38)',
              lang: 'haskell',
              text: "{ lambda ; a:MemRef<None>{bfloat16[256,256]} b:MemRef<None>{bfloat16[256,256]} c:MemRef<None>{bfloat16[256,256]}. let\n    d:bf16[256,256] <- a[:,:]\n    e:bf16[256,256] <- b[:,:]\n    f:bf16[256,256] = add d e\n    c[:,:] <- f\n  in () }",
            },
          },
          {
            h: 'Refs were not invented for kernels',
            ps: [
              'The design document is careful about the provenance here: refs are not a Pallas-specific concept, they were introduced to JAX to represent stateful computations, and Pallas leverages them for kernels that operate on mutable memory. That matters for how you should read the restriction. A ref refusing to enter `jnp.exp` is not a rule Pallas bolted onto JAX. It is what the type has always meant.',
              'The practical consequence is that a kernel body has no hidden loads. Every trip between memory and registers is a term in the jaxpr, which is why counting loads in a printed kernel is a legitimate way to reason about a body that feels slower than it should. Read the same ref twice in a body and you get two loads, because you asked for two.',
            ],
          },
          {
            h: 'choosing a memory space is choosing who moves the data',
            ps: [
              'The memory-spaces lesson before this one covers VMEM as the default, SMEM for scalars, and ANY as the promise that you will move the data yourself. The detail the guide leaves implicit is what ANY forbids. A buffer in the `ANY` space cannot be dereferenced with ordinary indexing at all: `x_ref[...]` on it is not a slow read, it is not a read. You copy into a VMEM or SMEM buffer first with `pltpu.sync_copy` or `pltpu.async_copy`, and only then does indexing mean anything.',
              '`ANY` is also documented as a hint rather than an address. It tells the compiler the memory space is unconstrained, and in most cases XLA will place the buffer in HBM. You get a promise about who is responsible, not a guarantee about where the bytes sit.',
            ],
            table: {
              caption: 'the four spaces a TPU kernel argument can occupy',
              cols: ['Pallas enum', 'TPU memory space', 'kind', 'what it means for you'],
              rows: [
                ['`pl.ANY`', 'HBM, usually', 'DRAM', 'unplaced; you copy it in before you can index it'],
                ['`pltpu.VMEM`', 'VMEM', 'SRAM', 'the default; the pipeline stages your block here'],
                ['`pltpu.SMEM`', 'SMEM', 'SRAM', 'scalar loads and stores only; where decisions are made'],
                ['`pltpu.SEMAPHORE`', 'semaphore', 'SRAM', 'barriers and async tracking, allocated like scratch'],
              ],
            },
          },
          {
            h: 'where the scalar line actually falls',
            ps: [
              'The TPU reference draws the boundary by rank, not by size. Every 0D array is stored in scalar registers and its operations run on the scalar core. Everything else runs on the vector core, and the document says so explicitly for the case people get wrong: even a single-element array of rank 1 or more goes to the vector unit.',
              'That rule has a price tag attached, because all vector computation is padded up to the tile. Adding two 1x1 arrays costs what adding two 8x128 arrays costs. So a sequence length kept as a shape-`()` scalar in SMEM is a scalar-core decision; the same number wrapped in a length-1 vector is a full tile of vector work to compute one comparison. The shape you chose picked the processor.',
            ],
          },
          {
            h: 'aliasing is a promise the compiler cannot check',
            ps: [
              '`input_output_aliases` tells `pallas_call` that an output may reuse an input\'s buffer, which is donation at kernel granularity. The aliasing lesson later in this arc covers the mechanics. What it does not spell out is the ordering consequence: once the two share memory, a write to the output ref is a write to the input, so within a grid step the order of your reads and writes is now semantics. Read the aliased input after writing the output and you read what you just wrote.',
              'The sparse-kernel guide uses aliasing for something other than saving an allocation, and it is worth stealing. In its block-sparse matmul, some output blocks are never visited by the grid at all, so their buffer would hold whatever was there. It passes an array of zeros in and aliases it onto the output, which makes "never visited" mean zero instead of uninitialized. Aliasing as an initialization strategy, not a memory optimization.',
            ],
            code: {
              caption: 'from the scalar-prefetch guide\'s DSD matmul; the fifth argument is an array of zeros',
              lang: 'python',
              text: "kernel = pl.pallas_call(\n  dsd_kernel,\n  grid_spec=grid_spec,\n  out_shape=out_shape,\n  # We use input-output aliases to zero-out o_ref for blocks that we never\n  # visit. By passing in an array of zeros we avoid having o_ref start with\n  # uninitialized values.\n  input_output_aliases={4: 0},  # Map zeros to o_ref.\n)",
            },
          },
        ],
        readings: [
          {
            label: 'Pallas design: reference types',
            url: 'https://docs.jax.dev/en/latest/pallas/design/design.html',
            note: 'where the read-gives-an-Array rule is stated as a property of the type',
          },
          {
            label: 'TPU pipelining: memory spaces',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html',
            note: 'the enum-to-hardware table, and what ANY forbids',
          },
          {
            label: 'Writing TPU kernels with Pallas',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/details.html',
            note: 'computation placement: rank decides scalar core or vector core',
          },
        ],
      },
      {
        id: 'manual-dma',
        num: 4,
        title: 'Manual DMA and semaphores',
        lede: 'The automatic pipeline is make_async_copy and semaphores, done for you. This lesson does it by hand, and names the bug you sign up for.',
        goal: 'Issue and wait an async copy correctly, put compute in the window between them, and say why a missed wait is silent on hardware and invisible in interpret mode.',
        sections: [],
        guide: { id: 'pallas', sections: [5] },
        readings: [
          { label: 'Pallas TPU pipelining', url: 'https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html', note: 'the automatic version of everything this lesson does by hand' },
          { label: 'Pallas TPU details', url: 'https://docs.jax.dev/en/latest/pallas/tpu/details.html', note: 'the memory-space rules a manual copy must respect' },
        ],
        check: [
          {
            q: 'When is a manual DMA justified at all?',
            a: 'When the automatic pipeline cannot express the access pattern, an irregular gather the BlockSpec grammar has no vocabulary for. Not as a casual optimization.',
          },
          {
            q: 'What happens if you skip the semaphore wait, and why is it not a compile error?',
            a: 'The kernel reads whatever bytes already sat in that memory before the copy landed. Shapes and dtypes all check out, so nothing objects at compile time, and interpret mode cannot catch it because it has no memory spaces.',
          },
          {
            q: 'How does this mechanism relate to multi-chip communication?',
            a: 'It is the same mechanism: a remote DMA over ICI is the same make_async_copy-and-semaphore pattern with a destination on a neighboring chip.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'aliasing-and-debugging',
        num: 5,
        title: 'Aliasing, and the debugging toolkit',
        lede: 'Aliasing changes what happens to a buffer, not what your kernel looks like. That is exactly why the two debug flags sit beside it.',
        goal: 'Declare input_output_aliases correctly, and run the interpret-then-debug workflow in the order that matches the failure you have.',
        sections: [],
        guide: { id: 'pallas', sections: [6] },
        readings: [
          { label: 'Pallas quickstart', url: 'https://docs.jax.dev/en/latest/pallas/quickstart.html', note: 'interpret mode, in the official introduction' },
          { label: 'Pallas TPU details', url: 'https://docs.jax.dev/en/latest/pallas/tpu/details.html', note: 'what the real lowering path checks that interpret mode skips' },
        ],
        check: [
          {
            q: 'What does input_output_aliases change, and what does it leave alone?',
            a: 'The output reuses the input\'s buffer, so the caller\'s memory is written in place. The kernel logic and shapes stay exactly as written, which is why an aliasing mistake corrupts data instead of failing loudly.',
          },
          {
            q: 'A kernel is green under interpret=True and fails to lower on hardware. Which flag next, and why?',
            a: 'debug=True. It runs the real lowering path and prints the jaxpr and the Mosaic module, which is where that class of failure lives; interpret mode already vindicated the algebra.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'one-language-three-backends',
        num: 6,
        title: 'One language, three backends',
        lede: 'The algorithm you wrote was never chip-specific. The schedule was, and this lesson names exactly which half transfers.',
        goal: 'Say what carries from Pallas on TPU to Mosaic GPU and interpret mode, and what must be rebuilt, precisely enough to plan a port.',
        sections: [],
        guide: { id: 'pallas', sections: [7] },
        readings: [
          { label: 'Pallas documentation', url: 'https://docs.jax.dev/en/latest/pallas/index.html', note: 'the backend surface as it stands today' },
          { label: 'Pallas design document', url: 'https://docs.jax.dev/en/latest/pallas/design/design.html', note: 'why one front end was the goal from the start' },
        ],
        check: [
          {
            q: 'What transfers across the three backends, and what does not?',
            a: 'The algorithm half transfers: the algebra in the body and the split between computation and scheduling. The execution model does not: the TPU\'s sequential software pipeline against the GPU\'s warps, and VMEM against GPU shared memory.',
          },
          {
            q: 'Why does a TPU have no warp-like unit?',
            a: 'A TPU core was never a bundle of threads. The grid runs as a sequential software pipeline with parallelism coming from the wide units and the overlap, so there is no thread group for a warp to name.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
    ],
  },
  {
    unit: 's:pallas',
    lessons: [
      {
        id: 'blockspec',
        num: 1,
        work: [
          { id: 'carve', label: 'design a carve for a (1024, 1024) bf16 matmul on v5e: lattice, budget, reuse, write order' },
        ],
        title: 'BlockSpec and the index map',
        lede: 'A carve is not a slicing convenience. It is the contract that decides which bytes are resident on which step, and what garbage sits past the edge.',
        goal:
          'Design a carve for a given kernel and shape: satisfy the lattice and the VMEM budget, exploit revisits, and predict exactly what each grid step sees, including the last one.',
        sections: [
          {
            h: 'the carve, stated exactly',
            ps: [
              'The layer guide at /l/pallas already makes the central correction: an index map returns block coordinates, not element offsets, and Pallas multiplies by the block shape to get the address. The reference states the full rule as executable code, and two details in it are easy to miss.',
              'First, the multiplication is unconditional and per axis, so every axis of the returned tuple is scaled by that axis\'s block size. Second, there is an assertion: at least one element of the block must be within bounds. A start index past the end of the array is an error, but a block that starts inside and runs off the end is legal and common.',
            ],
            code: {
              caption: 'the reference\'s own account of what a grid step reads (jax docs, Grids and BlockSpecs)',
              lang: 'python',
              text: "block_indices = x_spec.index_map(*invocation_indices)\nelem_indices = []\nfor x_size, block_size, block_idx in zip(x_shape, x_spec.block_shape, block_indices):\n    start_idx = block_idx * block_size\n    # At least one element of the block must be within bounds\n    assert start_idx < x_size\n    elem_indices.append(slice(start_idx, start_idx + block_size))",
            },
          },
          {
            h: 'the last block is a promise about garbage',
            ps: [
              'When the block shape does not divide the array evenly, the final iteration on that axis still receives a full-shape block. Out-of-bounds elements are padded on input and discarded on output, and the reference is explicit that the padding values are unspecified and you should assume they are garbage.',
              'Interpret mode pads with NaN for floating-point values so you have a chance to spot the access, and the same paragraph says this behavior should not be depended upon. Both halves matter. The NaN is a debugging affordance, so treat a NaN in interpret mode as a real finding; and never write a kernel whose correctness needs the padding to be any particular value.',
              'That is why a mask built from an iota against the array\'s true bound is not defensive style. It is the only thing standing between a reduction and unspecified bytes. The museum keeps this family of rank mistakes, an index map handing back one coordinate for a two-axis block among them, at /mistakes/kernels.',
            ],
          },
          {
            h: 'None axes, whole-array specs, and the default map',
            ps: [
              'A `None` entry in `block_shape` behaves as the value 1, except that the axis is squeezed out of the ref the kernel body sees. You can write `pl.Squeezed()` for the same thing. A three-axis array carved with `(None, 2)` on its last two axes hands the body a rank-1 ref, which is usually what you wanted when you wrote a leading batch axis into the grid.',
              'Two defaults save typing and are worth knowing by name. `block_shape=None` means the whole array shape. `index_map=None` means the map that returns zeros for every axis regardless of grid position, which is the "stage it once, everybody reads the same copy" spec written the short way.',
              'There is also a second indexing mode most kernels never touch. With `pl.Element(block_size)`, the values your index map returns are used directly as array indices with no scaling, and you may declare virtual low-high padding per dimension as though the array were padded on input. Element mode is TPU only. Reach for it when the natural expression of your access is an element offset and the block arithmetic is fighting you.',
            ],
            code: {
              caption: 'a None axis is squeezed out of the body\'s ref (jax docs, Grids and BlockSpecs)',
              lang: 'python',
              text: "def kernel(o_ref):\n    assert o_ref.shape == (2,)          # (None, 2) arrived as rank 1\n    o_ref[...] = jnp.full((2,), 10 * pl.program_id(1) + pl.program_id(0))\n\npl.pallas_call(kernel,\n               jax.ShapeDtypeStruct((3, 4), dtype=np.int32),\n               out_specs=pl.BlockSpec((None, 2), lambda i, j: (i, j)),\n               grid=(3, 2), interpret=True)()",
            },
          },
          {
            h: 'what makes a carve good',
            ps: [
              'Three constraints decide a block shape, and they pull in different directions. The lattice is the hard one: the last two dimensions must be divisible by 8 and 128, or equal the array\'s dimensions exactly. Rank-1 blocks have their own rule that the path never states, namely that the block dimension must equal the array dimension, or be a multiple of 1024, or be a power of two and at least `128 * (32 / bitwidth(dtype))`.',
              'The budget is the soft one. Larger windows generally give better hardware utilization, so the pull is upward, and the ceiling is that a window plus the space for spilled vector registers can exceed VMEM. What you get then is a low-level compiler error about memory, which is the museum\'s VMEM overflow exhibit and the reason a budget line belongs in your notes before you compile.',
              'Reuse is the one people forget, because it is a property of the grid order rather than the block shape. When two lexicographically consecutive grid indices map to the same slice of an input, the HBM transfer for the second one is skipped, since the data is already there. Your grid order therefore sets how many transfers your carve costs, and reordering axes can change traffic without changing a single block dimension.',
            ],
            table: {
              caption: 'the three constraints and how each announces itself',
              cols: ['constraint', 'rule', 'how it fails'],
              rows: [
                ['lattice', 'last two dims divisible by 8 and 128, or equal the array dims', 'compile error naming both numbers'],
                ['rank-1 blocks', 'equal the array dim, or a multiple of 1024, or a power of two at least `128 * (32 / bitwidth)`', 'compile error at lowering'],
                ['VMEM budget', 'window plus spilled registers must fit', 'low-level out-of-memory from the backend'],
                ['reuse', 'consecutive steps on the same slice skip the transfer', 'no error at all, just traffic you paid for'],
              ],
            },
          },
          {
            h: 'writes have an ordering rule that reads do not',
            ps: [
              'Because the TPU grid runs sequentially, several invocations may write the same slice of the output with no risk of a race. The reference attaches one condition to that permission: all invocations that write a particular slice must be consecutive. Break the run and you are outside what the backend guarantees.',
              'This is where the matmul convention comes from, and it is a correctness argument rather than a performance one. Some prefix of the grid axes varies the output slice; the remaining suffix leaves the output window fixed. A reduction axis leaves the output fixed, so it has to be last, and the output ref then works as an accumulator across it. Put K first and the writes to a given output block are no longer consecutive.',
              'The general statement is stricter still: when multiple invocations write to the same elements of the output, the result is platform dependent. Nothing about your shapes or dtypes is invalid, so nothing warns you.',
            ],
          },
        ],
        readings: [
          {
            label: 'Grids and BlockSpecs',
            url: 'https://docs.jax.dev/en/latest/pallas/grid_blockspec.html',
            note: 'the exact slice semantics, None axes, and the Element indexing mode',
          },
          {
            label: 'TPU kernels: noteworthy properties and restrictions',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/details.html',
            note: 'the consecutive-write rule and the revisit skip, in the backend\'s own words',
          },
        ],
      },
      {
        id: 'grid-and-pipeline',
        num: 2,
        work: [
          { id: 'reroll', label: 're-derive the double-buffered loop from the unrolled form without looking' },
        ],
        title: 'The grid and the pipeline',
        lede: 'The grid promises that every point runs. Every other thing you rely on comes from the backend underneath it.',
        goal:
          'Separate what pallas_call guarantees from what the TPU backend happens to do, read a grid as a double-buffered schedule, and diagnose a stalled pipeline from block sizes and grid order.',
        sections: [
          {
            h: 'the guarantee, and what it withholds',
            ps: [
              'The kernel path teaches the grid as a software pipeline, at /s/pallas, and that teaching is true of the TPU backend. It is not what `pallas_call` promises. The design document says so in a single sentence, and the sentence is worth carrying around.',
              '>> pallas_call also provides no guarantees on the order of loop iterations over the iteration space, just that every member of the iteration space will be looped over.',
              'That line is from the Pallas design document, and the paragraph after it says where the order actually comes from: compilers like Triton and Mosaic have more specific operational semantics associated with the grid. Portability lives in the front end. Order lives in the backend.',
              'On TPU the backend is specific indeed. The reference calls TPUs highly sequential machines and says the grid is generally not processed in parallel but sequentially, in lexicographic order. So a kernel that depends on order is correct here and is not portable, and you were told which of those two you were buying.',
            ],
          },
          {
            h: 'parallel and arbitrary, as permission',
            ps: [
              '`dimension_semantics` is often described as a hint about your intent. It is closer to permission. Some TPU chips carry two TensorCores behind one device, each with its own VMEM, VREGs, SMEM, SREGs and compute units, sharing HBM. Using both means breaking the sequential grid guarantee, and the annotation is how you say which axis may be broken.',
              'The rule of thumb the reference gives is mechanical: an axis is parallel unless the output window does not vary along it. Which is why the annotation always reads as some parallel axes followed by some arbitrary ones, and why a reduction axis is never parallel. Mark it wrong and you get a wrong answer rather than an error, because nothing about the shapes became invalid.',
              'Two caveats are worth carrying. Partitioning across two cores often gives close to 2x, and can give much less when the per-step cost varies, since one core can be handed all the expensive steps and the other idles waiting. And the guide notes megacore as currently a v4 and v5p feature, where supplying the annotation elsewhere is a no-op, while omitting it entirely leaves a second core unused. This site\'s own matmul retune moved the annotation and the block shape in the same step, so that measurement does not isolate either one.',
            ],
            code: {
              caption: 'the annotation as the pipelining guide writes it today; jax 0.4.x spelled it TPUCompilerParams',
              lang: 'python',
              text: "pl.pallas_call(\n    add_matrices_kernel,\n    out_shape=jax.ShapeDtypeStruct.like(x),\n    in_specs=[block_spec, block_spec],\n    out_specs=block_spec,\n    grid=(2,),\n    compiler_params=pltpu.CompilerParams(\n        dimension_semantics=(\"parallel\",)),\n)(x, y)",
            },
          },
          {
            h: 'how a grid becomes double buffering',
            ps: [
              'The derivation in the software pipelining tutorial is short enough to hold in your head, and holding it is what turns "the grid is a pipeline" from a slogan into something you can predict from. Start with a loop that copies in, computes, copies out. Split each copy into a start and a wait so asynchrony is expressible. Give the staging buffer two slots so iteration i can compute out of one while i+1 fills the other. Push each copy-out wait as late as it can go. Re-roll the loop.',
              'What falls out has a prologue that starts the first copy, a steady state where every iteration issues the next input copy before waiting on the current one, and an epilogue that drains the last write. The alternation between slots is the `i % 2` you see in the pseudocode, and the reason the default buffer count is two.',
            ],
            code: {
              caption: 'the re-rolled pipeline, from the software pipelining tutorial',
              lang: 'python',
              text: "# Prologue\ncopy_in_start(A[0], X[0])\n\n# Main loop\nfor i in range(N):\n  cur_slot = i % 2\n  next_slot = (i + 1) % 2\n\n  if i+1 < N:\n    copy_in_start(A[i+1], X[next_slot])\n\n  copy_in_wait(X[cur_slot])\n  Y[cur_slot] = X[cur_slot] + 1\n  copy_out_start(Y[cur_slot], A[i])\n\n  if i > 0:\n    copy_out_wait(Y[next_slot])\n\n# Epilogue\ncopy_out_wait(Y[1])",
            },
          },
          {
            h: 'the schedule, visible in the module',
            ps: [
              'The design document says `BlockSpec`s can be converted into pipeline schedules, and on this repo you can read the conversion instead of taking it on faith. Lowering the tiled matmul with `debug=True` puts the whole schedule into one attribute dictionary on the Mosaic function: the grid arrives as `iteration_bounds`, each spec arrives as a `window_params` entry carrying its `window_bounds` and a `transform_indices` function, and `dimension_semantics` rides in beside them.',
              'Everything you wrote outside the kernel body is in that line, and nothing else is. The body became instructions; the carve became metadata that the pipeline emitter reads.',
            ],
            code: {
              caption: 'one line of the tiled matmul\'s Mosaic module, captured on this repo (jax 0.4.38)',
              lang: 'mlir',
              text: "func.func @main(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<256x256xbf16, #tpu.memory_space<vmem>>, %arg4: memref<256x256xbf16, #tpu.memory_space<vmem>>, %arg5: memref<256x256xbf16, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<arbitrary>, #tpu.dimension_semantics<arbitrary>, #tpu.dimension_semantics<arbitrary>], iteration_bounds = array<i64: 2, 2, 2>, scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, window_params = [{transform_indices = @transform_0, window_bounds = array<i64: 256, 256>}, {transform_indices = @transform_1, window_bounds = array<i64: 256, 256>}, {transform_indices = @transform_2, window_bounds = array<i64: 256, 256>}]}",
            },
          },
          {
            h: 'two buffers is a default, not a law',
            ps: [
              'Buffer count is per argument. Pass `pl.Buffered(buffer_count=n)` as a `BlockSpec`\'s `pipeline_mode` and that input or output gets n slots instead of two, which is what you want when one operand\'s transfer is much longer than a single step of compute. The same object turns on lookahead prefetch with `use_lookahead=True`.',
              '`pltpu.emit_pipeline` moves the whole mechanism inside the kernel body. Instead of one pipeline created at kernel entry, you construct pipelines where you need them, which is how a nested schedule gets written: an outer pipeline moving data between chips, an inner one moving it between HBM and VMEM. It also carries the features that only exist at that level, dynamic block shapes among them.',
            ],
          },
          {
            h: 'when the pipeline stalls',
            ps: [
              'A pipeline hides transfer under compute, so it stalls whenever a step has more transfer than compute to hide it behind. The site plays this as an instrument: EX·07 runs the same matmul as EX·02 with the overlap removed, and the MXU sitting idle between loads is what the stall looks like when you can see it. The number that decides which animation you are living in is bytes per step over flops per step, against the chip\'s ridge.',
              'Three other stalls have nothing to do with block size. A grid with very few steps pays its prologue and epilogue in full, since neither has a partner to hide behind, and a short grid is mostly prologue and epilogue. A grid reordered so that consecutive steps no longer touch the same input slice loses the skipped transfers that the previous order was quietly getting. And a partitioned axis whose steps cost wildly different amounts leaves one core idle while the other finishes.',
              'The diagnosis is the same in every case and it lives in the profile: name the envelope, sum the compute ops inside it, and the difference is unhidden transfer time. The /l/tpu chapter walks that reading against a real capture, and the number it produces is the only honest answer to whether your pipeline is working.',
            ],
          },
        ],
        readings: [
          {
            label: 'Pallas design: executing kernels with pallas_call',
            url: 'https://docs.jax.dev/en/latest/pallas/design/design.html',
            note: 'the no-guarantee sentence, in context, with the backends named after it',
          },
          {
            label: 'TPU pipelining',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html',
            note: 'multiple buffering, lookahead, emit_pipeline, and megacore partitioning',
          },
          {
            label: 'Software pipelining',
            url: 'https://docs.jax.dev/en/latest/pallas/pipelining.html',
            note: 'the double-buffer derivation, unrolled then re-rolled',
          },
        ],
      },
      {
        id: 'scalar-world',
        num: 3,
        work: [
          { id: 'prefetch', label: 'write a prefetch map that repeats an index and explain which transfers disappear' },
        ],
        title: 'The scalar world',
        lede: 'A kernel that decides has to keep the number it decides on somewhere, and a vector register is the wrong place.',
        goal:
          'Write kernels whose block selection and control flow come from data: place scalars in SMEM, prefetch the index arrays a schedule depends on, and say precisely which work a branch can skip and which it cannot.',
        sections: [
          {
            h: 'the memory that answers questions',
            ps: [
              'SMEM is small, low-latency, randomly addressable, and reads or writes 32 bits per instruction. Set that against the 4 KiB granularity of a VMEM transaction and the trade is clear: SMEM moves almost nothing per instruction and has no alignment requirement to satisfy, which is exactly the shape of a single sequence length or loop bound. The reference\'s rule of thumb is short. Any data used to perform control-flow decisions should be placed in SMEM.',
              'The chapter at /l/tpu puts the scalar core beside the MXU and the VPU and explains why the usual two-block diagram is incomplete. This page is the programming model that unit exposes: what you may put in front of it, when it runs relative to everything else, and what it is allowed to decide.',
            ],
          },
          {
            h: 'prefetch is an ordering guarantee, not a copy',
            ps: [
              'Every index map so far took grid coordinates and nothing else, which is what lets the pipeline evaluate it before any real data exists. `PrefetchScalarGridSpec` widens the input to that function, and the widening is the whole feature. With `num_scalar_prefetch=n`, the first n arguments are placed in SMEM with no `BlockSpec` of their own, and every subsequent spec\'s index map receives those SMEM refs after the grid indices.',
              'The reason this needs an API rather than an ordinary read is scheduling, not syntax. The index map runs in order to decide which copy to issue. If the value it reads were itself staged by the pipeline, the decision would depend on a transfer that the decision was supposed to schedule. Prefetch resolves that by landing the scalars before the pipeline\'s first step exists, so by the time any index map runs, the numbers are already there.',
              'Three orderings follow from that and they are easy to transpose. The index map takes grid indices first, then prefetch refs. The kernel body takes prefetch refs first, then inputs, then outputs, then scratch. The caller passes prefetch arguments before the real ones.',
            ],
            code: {
              caption: 'the three signatures, schematic, from the scalar prefetch guide',
              lang: 'text',
              text: "def index_map(*grid_indices, *prefetch_refs):\n    ...\n\ndef kernel(*prefetch_refs, *input_refs, *output_refs, *scratch_refs):\n    ...\n\nkernel = pl.pallas_call(...)\nresult = kernel(*prefetch_args, *input_args)",
            },
          },
          {
            h: 'the prefetch map',
            ps: [
              'Once an index map can read data, a schedule becomes something you compute. The pattern the sparse guide builds is a prefetch map: an array with one entry per grid position holding the index of the next non-zero block, computed in ordinary JAX outside the kernel and passed in as a prefetch argument. The index map then returns that entry instead of a coordinate derived from the grid.',
              'What this buys is not a faster loop, it is a smaller iteration space. The block-sparse matmul in that guide runs a grid of `(N // blk_N, num_blocks)` where the second axis walks the non-zero blocks of a compressed representation, so blocks that are entirely zero are never a grid point at all. The guide notes in passing that the grid size itself does not have to be static.',
            ],
            code: {
              caption: 'the index map reads the map instead of the grid (scalar prefetch guide)',
              lang: 'python',
              text: "def mask_index_map(prefetch_map, i, j, ...):\n  next_nonzero_block = prefetch_map[i, j]\n  return (next_nonzero_block, 0, 0)",
            },
          },
          {
            h: 'kernels that decide',
            ps: [
              'A sparse grid breaks an assumption the dense one gave you for free: consecutive steps no longer reliably share an output block. So the kernel has to work out for itself when a new accumulation starts and when the running total is finished. The DSD kernel does exactly that by comparing its block index against its neighbors in the prefetched array, zeroing the accumulator when the block changed and flushing to the output when it is about to change again. Both decisions read only SMEM.',
              'Now the distinction that governs everything in this lesson. A branch in the body decides what happens to a block; it cannot decide whether the block arrives. The copy was issued by the pipeline emitter from the index map, before your body ran. Wrapping compute in `pl.when` saves the compute and pays the transfer anyway.',
              '>> The index map decides what moves. A branch in the body only decides what happens to it once it has arrived.',
              'Which gives the actual technique for skipping a block: make the index map not ask for it. Point a step at the same block its predecessor used, and the backend skips the transfer entirely, because consecutive steps on the same slice reuse what is already resident. A prefetch map that repeats an index is a schedule that repeats no work.',
            ],
            code: {
              caption: 'the accumulator flush, from the guide\'s block-sparse matmul',
              lang: 'python',
              text: "blk_idx = pl.program_id(1)\nis_start = blk_idx == 0\nchanged_blocks = (idxs_i_ref[blk_idx] != idxs_i_ref[jnp.maximum(blk_idx-1, 0)])\n@pl.when(is_start | changed_blocks)\ndef _():\n  accum_scratch[...] = jnp.zeros_like(accum_scratch)\naccum_scratch[...] += jnp.dot(x_ref[0, :, :], y_ref[...], preferred_element_type=jnp.float32)\n\nnext_block_change = (idxs_i_ref[blk_idx] != idxs_i_ref[jnp.minimum(blk_idx+1, num_blocks)])\nis_end = blk_idx == (num_blocks - 1)\n@pl.when(is_end | next_block_change)\ndef _():\n  o_ref[...] = accum_scratch[...].astype(o_ref.dtype)",
            },
          },
          {
            h: 'what may be data, and what may not',
            ps: [
              'Ragged work is a fixed tile whose count and placement are data. Block indices can be data, through the index map. Loop bounds inside the body can be data, through a scalar in SMEM. The grid size can be data. With `pltpu.emit_pipeline` even the block extent can move, using `pl.BoundedSlice` in the block shape and `pl.ds` in the index map, so consecutive steps copy differently sized chunks.',
              'What cannot be data is the tile the vector registers are laid out for. That is the lattice from the blockspec lesson, and no amount of prefetching relaxes it. So the ragged kernel you write is not a kernel with a variable shape; it is a kernel with a fixed shape whose schedule was computed at run time.',
              'One cost to plan for, since the guide names it: the prefetch step runs before the main pipeline begins, so scalar arrays large enough to matter push out the start of real work. Page tables and sequence lengths are the intended size. This mechanism earns its place when the alternative schedule cannot be written at all, not when it would merely have been slightly clumsier.',
            ],
            code: {
              caption: 'dynamic block extents inside emit_pipeline (TPU pipelining guide)',
              lang: 'python',
              text: "def index_map(i):\n    start = slices_smem[i, 0]\n    size = slices_smem[i, 1] - slices_smem[i, 0]\n    return (pl.ds(start, size), 0)\n\nblock_spec = pl.BlockSpec(block_shape=(pl.BoundedSlice(8), 128),\n                          index_map=index_map)",
            },
          },
        ],
        readings: [
          {
            label: 'Scalar prefetch and block-sparse computation',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/sparse.html',
            note: 'the signatures, the prefetch map, and the DSD kernel this lesson reads',
          },
          {
            label: 'TPU kernels: placing operands in SMEM',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/details.html',
            note: 'why 32-bit random access is the right shape for a control decision',
          },
          {
            label: 'TPU pipelining: dynamic block shapes',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html',
            note: 'where a block extent becomes data: BoundedSlice inside emit_pipeline',
          },
        ],
      },
    ],
  },
  {
    unit: 'l:mosaic',
    coversGuide: true,
    lessons: [
      {
        id: 'to-machine-code',
        num: 1,
        work: [
          { id: 'interpret', label: 'run one kernel in interpret mode and name the two failures it cannot see' },
        ],
        title: 'From tracing to machine code',
        lede: 'One kernel body, three destinations: a TPU backend, a GPU backend, and a scan that runs on your laptop.',
        goal:
          'Follow a kernel from traced jaxpr to Mosaic MLIR, and predict what vmap, grad, and interpret mode each do to the same source.',
        sections: [
          {
            h: 'the jaxpr names no chip',
            ps: [
              'Tracing a kernel body against refs produces a jaxpr like any other, with two differences you can see at a glance: the binders are memrefs rather than arrays, and loads and stores appear as their own terms. The path had you print one in LAB·1.4 and read it in chapter 07. The claim worth adding here is about what is absent from it.',
              'Nothing in that jaxpr mentions a TPU. No tile shape, no memory space, no pipeline. It is a body plus its memory traffic, stated in the same primitives an ordinary JAX program uses, which is what makes the same source reachable by three different compilers.',
              '>> The jaxpr names no chip. Every backend decision happens below it.',
            ],
          },
          {
            h: 'Mosaic: standard dialects, then LLO',
            ps: [
              'Mosaic consumes mostly standard-dialect MLIR and emits LLO to be compiled for the TPU. Pallas gets there by translating JAX primitives into MLIR, mostly the `vector` and `arith` dialects, and by converting the `BlockSpec`s into the pipeline schedules the grid lesson at /s/pallas/grid-and-pipeline read out of the module\'s attributes.',
              'Read one of this repo\'s captures and you find two vocabularies side by side. The algebra is standard: a row max becomes `vector.multi_reduction`, the exponential becomes `math.exp`, the divide becomes `arith.divf`, the bf16 round-trip becomes `arith.extf` and `arith.truncf`. Only what is specific to this machine wears the `tpu` prefix, `tpu.matmul` and `tpu.vector_store` among them.',
              'The split is worth internalizing because it tells you where portability ends. Anything expressed in `vector` and `arith` is describing computation any vector machine could do. Anything in the `tpu` dialect is naming a unit that exists on this chip.',
            ],
            code: {
              caption: 'the fused softmax body in Mosaic, captured on this repo (jax 0.4.38)',
              lang: 'mlir',
              text: "%1 = arith.extf %0 : vector<256x512xbf16> to vector<256x512xf32>\n%cst = arith.constant dense<0xFF800000> : vector<256xf32>\n%2 = vector.multi_reduction <maximumf>, %1, %cst [1] : vector<256x512xf32> to vector<256xf32>\n%3 = vector.shape_cast %2 : vector<256xf32> to vector<256x1xf32>\n%4 = vector.broadcast %3 : vector<256x1xf32> to vector<256x512xf32>\n%5 = arith.subf %1, %4 : vector<256x512xf32>\n%6 = math.exp %5 : vector<256x512xf32>",
            },
          },
          {
            h: 'the path that was deprecated, and why to read it anyway',
            ps: [
              'Pallas was designed with Triton as a target, so the original GPU lowering was straightforward: JAX dot products became Triton dot products, unary primitives became their Triton equivalents, and Triton atomics arrived through new Pallas primitives. The GPU path now goes to Mosaic GPU instead, and the design document carries a footnote saying the Pallas-to-Triton lowering path is officially deprecated and is discussed for historical reasons.',
              'Read the deprecated section anyway, for one paragraph in it. Triton has no notion of a `BlockSpec` and addresses memory with pointers rather than indices, so lowering `x_ref[3, 2]` on a `(4, 5)` ref meant computing the row-major pointer by hand, `5 * 3 + 2 * 1`, and lowering a slice like `x_ref[4, :]` meant producing a whole array of pointers.',
              'That arithmetic is what the block index abstraction removed. The blockspec lesson spent its first section on the fact that an index map returns block coordinates and Pallas does the multiplication; this is the layer where you can see who used to do it and what they had to say instead.',
            ],
          },
          {
            h: 'interpret mode is a scan',
            ps: [
              'Because a kernel is JAX primitives plus a few Pallas ones, a `pallas_call` can be lowered to StableHLO directly, implemented as a `lax.scan` over the grid, and compiled by XLA like any other program. That is the whole of interpret mode. It is not a simulator written for Pallas; it is your kernel expressed as a loop that XLA already knows how to run, on any supported platform including CPU.',
              'Three consequences follow, and the design document states all three. Ordinary debugging works, `jax.debug.print` included. The numerics come from XLA, which the document calls more reliable and better tested, and which is used to verify the Triton and Mosaic compilers. And the ordering is the scan\'s, which the document notes could in principle be perturbed to simulate the parallel reads and writes a GPU performs.',
              'What a scan cannot have is a memory hierarchy. VMEM, SMEM and ANY collapse into whatever plain JAX does, so a lattice violation and a VMEM overflow, the two exhibits the museum keeps at /mistakes/kernels, are invisible here by construction. A kernel that passes in interpret mode and fails to lower on hardware almost always failed the second question, not the first.',
            ],
          },
          {
            h: 'vmap adds an axis, grad transposes memory',
            ps: [
              'Batching a kernel from the outside has a natural default. `vmap` of a `pallas_call` augments the call with an extra grid dimension for the new batch axis and rewrites the `BlockSpec`s to index along it, and `jax.custom_vmap` is there for when that default is not the batched kernel you wanted.',
              'Differentiation is where the layering strains, and the design document is specific about why rather than vague. `jax.grad` decomposes into `jvp`, `partial_eval` and `transpose`, and most of JAX\'s existing machinery applies. Then the honest sentence: automatic differentiation of kernels can result in a performance hit due to how memory access is transposed. A kernel with overlapping-and-parallel reads and disjoint-but-parallel writes transposes into one with overlapping-but-parallel writes, which are slow when done atomically, and disjoint-and-parallel reads.',
              'The document names the missing capability too, which is more useful than the warning alone: emitting a good transposed kernel would mean reordering loops and changing the vectorization, and Pallas has no program representation amenable to that. Elementwise kernels transpose fine. For everything else the recommendation is `jax.custom_vjp`, which is exactly what stage 3 does when it wires a hand-derived backward onto flash attention at /s/kernels.',
              'Two more transformations sit on the document\'s speculative list rather than in the shipping surface. `checkify` could plumb error codes out of a kernel for out-of-bounds access or NaNs, and `custom_partitioning` could make a kernel automatically partitionable. Both are written as things one could imagine, and reading them that way keeps the map honest.',
            ],
          },
        ],
        readings: [
          {
            label: 'Pallas design: lowering and transforming Pallas',
            url: 'https://docs.jax.dev/en/latest/pallas/design/design.html',
            note: 'the Mosaic and Triton paths, emulation mode, and the grad caveat verbatim',
          },
          {
            label: 'Pallas documentation index',
            url: 'https://docs.jax.dev/en/latest/pallas/index.html',
            note: 'the current shape of the backend surface, which the design doc predates',
          },
          {
            label: 'Writing TPU kernels with Pallas',
            url: 'https://docs.jax.dev/en/latest/pallas/tpu/details.html',
            note: 'what the Mosaic backend accepts, op by op, when the lowering refuses',
          },
        ],
      },
      {
        id: 'the-tiling-vocabulary',
        num: 2,
        title: 'The tiling vocabulary, precisely',
        lede: 'A vector register is a physical grid of 8 sublanes by 128 lanes, and every vector type in a lowered module is that fact showing through.',
        goal: 'Read the trailing tile pair on any vector type and derive it from the dtype: (8, 128) for f32, (16, 128) for bf16, (32, 128) for int8.',
        sections: [],
        guide: { id: 'mosaic', sections: [0] },
        readings: [
          { label: 'Pallas TPU details', url: 'https://docs.jax.dev/en/latest/pallas/tpu/details.html', note: 'the tiling constraints as the kernel author meets them' },
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the register file these tiles live in' },
        ],
        check: [
          {
            q: 'Why does bf16 tile as (16, 128) when f32 tiles as (8, 128)?',
            a: 'Register width in bytes is fixed, so narrower elements pack more rows into the same footprint: half-width bf16 doubles the sublane count.',
          },
          {
            q: 'What two shapes does the captured module keep side by side, and what does the pairing teach?',
            a: 'The logical array shape you wrote in JAX and the physical vector type Mosaic derived from it. With the packing rule known, the pairing reads as a derivation rather than noise.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'reading-layout-decisions',
        num: 3,
        title: 'Reading layout decisions',
        lede: 'You do not trace every op to read a module. Three signatures carry the story: the cast before a reduction, the grid attributes, and the compiled index maps.',
        goal: 'Scan a Mosaic module for its three layout signatures and read each one back to the source decision that produced it.',
        sections: [],
        guide: { id: 'mosaic', sections: [1, 2] },
        readings: [
          { label: 'MLIR language reference', url: 'https://mlir.llvm.org/docs/LangRef/', note: 'the notation the module is written in' },
          { label: 'Pallas TPU pipelining', url: 'https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html', note: 'the dimension_semantics the grid attributes record' },
        ],
        check: [
          {
            q: 'What does a shape_cast just before a reduction tell you?',
            a: 'The logical shape dropped a dimension but the register grid cannot; the cast is the tiled representation being reconciled with the reduction\'s result shape.',
          },
          {
            q: 'Where did your BlockSpec\'s Python index map go?',
            a: 'It compiled into a transform function attached to the operand, taking grid indices and returning memory offsets. The closure does not survive; the function does.',
          },
          {
            q: 'Which attribute shows how the pipeline actually runs, and what marks a sequential axis?',
            a: 'dimension_semantics on the grid: each axis is parallel or arbitrary, and arbitrary marks the axis Mosaic must run in order.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
    ],
  },
]
