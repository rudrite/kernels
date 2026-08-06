// The compiler wall, arced: l:xla becomes five lessons slicing its guide,
// each one idea with its own frame and checks. l:gap deliberately stays a
// single-article unit (its whole job is one bridging idea); its ledger
// gains an applied item instead, edited in data/mastery.ts.
import type { UnitLessons } from './index'

export const COMPILER_WALL_LESSONS: UnitLessons[] = [
  {
    unit: 'l:xla',
    coversGuide: true,
    lessons: [
      {
        id: 'two-dumps-two-truths',
        num: 1,
        title: 'Two dumps, two truths',
        lede: 'Your program passes through XLA twice: once translated, once decided. Only one of those dumps can answer a performance question.',
        goal: 'Pick the right dump for the question you are asking, and explain why the lowered text cannot testify about performance.',
        sections: [],
        guide: { id: 'xla', sections: [0] },
        readings: [
          { label: 'JAX · ahead-of-time lowering', url: 'https://docs.jax.dev/en/latest/aot.html', note: 'the lower and compile calls, in the official telling' },
          { label: 'XLA tools', url: 'https://openxla.org/xla/tools', note: 'the wider toolbox around the two dumps' },
        ],
        check: [
          {
            q: 'A colleague reads .lower().as_text(), sees an expensive-looking loop, and starts optimizing. What did they get wrong?',
            a: 'The lowered StableHLO is the program before any backend decision: nothing is fused, tiled, or laid out yet, so cost read from it is fiction. Performance questions start at .compile().as_text(), where fusions, layouts, and buffers are real.',
          },
          {
            q: 'The compiled dump is many times larger than the lowered one for the same kernel. What is that size difference telling you?',
            a: 'How much the backend decided on your behalf: the extra lines are the fusion ops, layout choices, and buffer assignments the compiler added between your program and the machine.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'the-fusion-taxonomy',
        num: 2,
        title: 'The fusion taxonomy',
        lede: 'Every fusion op in a compiled dump carries a kind, and the kinds are the difference between a wall of names and a diagnosis.',
        goal: 'Classify the fusions in a compiled dump by kind, and judge whether a boundary between two of them was necessary or a missed merge.',
        sections: [],
        guide: { id: 'xla', sections: [1] },
        readings: [
          { label: 'XLA architecture', url: 'https://openxla.org/xla/architecture', note: 'where fusion sits in the pass pipeline' },
          { label: 'Scaling book · rooflines', url: 'https://jax-ml.github.io/scaling-book/roofline/', note: 'why HBM traffic is the thing fusion exists to cut' },
        ],
        check: [
          {
            q: 'What does a loop fusion buy, in one sentence about memory?',
            a: 'A chain of elementwise ops becomes one pass over the data, so no intermediate round-trips through HBM between the steps.',
          },
          {
            q: 'The naive attention dump at 8192 carries the full score matrix through three separate fusions. Why is that the chapter’s clearest evidence of a fusion limit?',
            a: 'The bf16[8192,8192] intermediate materializes in HBM between fusions because a single pass cannot express softmax’s reduce-then-renormalize dependence; the boundary is structural, and crossing it is exactly what the hand-written streaming kernel exists to do.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'layouts',
        num: 3,
        title: 'Layouts',
        lede: 'The compiled dump is the only place you can see how a tensor sits in memory, and the annotation reads in one breath once you know the order.',
        goal: 'Read {1,0}-style layout annotations on sight, predict when a transpose is free, and spot the copy ops layout disagreements insert.',
        sections: [],
        guide: { id: 'xla', sections: [2] },
        readings: [
          { label: 'XLA shapes and layout', url: 'https://openxla.org/xla/shapes', note: 'minor-to-major, defined by the people who wrote it' },
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the register tiles the layout feeds' },
        ],
        check: [
          {
            q: 'What does {1,0} on a rank-2 tensor say, reading the numbers in order?',
            a: 'Dimensions listed fastest-varying first: dimension 1 is minor, so elements along it sit contiguously, and dimension 0 is major. Row-major, said in the dump’s notation.',
          },
          {
            q: 'When is a transpose free, and what does the dump show when it is not?',
            a: 'Free when the consuming op can read the data under the new annotation, treating a different dimension as minor; the transpose disappears into a layout choice. When ops disagree, XLA inserts a copy op that physically rearranges the bytes, and that copy is the cost you see.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'the-memory-report',
        num: 4,
        title: 'Reading the memory report',
        lede: 'When a kernel asks for more VMEM than exists, the compiler prints the most literal error message in the stack. This lesson reads it line by line.',
        goal: 'Take a VMEM overflow report and name each allocation’s origin, then compute the fix from the verdict line’s own arithmetic.',
        sections: [],
        guide: { id: 'xla', sections: [3] },
        readings: [
          { label: 'Pallas · grids and BlockSpecs', url: 'https://docs.jax.dev/en/latest/pallas/grid_blockspec.html', note: 'the BlockSpec a window allocation came from' },
          { label: 'Cloud TPU system architecture', url: 'https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm', note: 'the ceiling the verdict line states' },
        ],
        check: [
          {
            q: 'A report line says window allocation. What object in your source does that correspond to?',
            a: 'A BlockSpec, turned into physical bytes: the block size you gave the kernel’s grid, now reserved as an actual region of VMEM.',
          },
          {
            q: 'The verdict reads 128.00M requested against 127.94M available. What is the fix, and why is it arithmetic rather than debugging?',
            a: 'Shrink a block spec until the request fits under the stated ceiling. The report lists allocations by size in the same units as the limit, so the amount to cut is a subtraction, not an investigation.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'dumps-on-demand',
        num: 5,
        title: 'Dumps on demand, and where XLA stops',
        lede: 'You can make the compiler write out every pass it runs, and you should know which decisions were never its to make.',
        goal: 'Produce a pass-by-pass dump with the XLA flags, and place the boundary where XLA hands off to libtpu on TPU or to vendor libraries on GPU.',
        sections: [],
        guide: { id: 'xla', sections: [4, 5] },
        readings: [
          { label: 'XLA tools', url: 'https://openxla.org/xla/tools', note: 'the dump flags, catalogued' },
          { label: 'OpenXLA', url: 'https://openxla.org/xla', note: 'the same compiler on both backends, one repository' },
        ],
        check: [
          {
            q: 'You want to watch just the fusion pass transform your module. Which two flags, and what does each do?',
            a: 'XLA_FLAGS=--xla_dump_to=DIR writes every intermediate stage to disk; --xla_dump_hlo_pass_re narrows the dump to passes matching a regex, pointed at fusion to isolate that step’s before and after.',
          },
          {
            q: 'A GPU matmul is slow, and the HLO dump looks fine. Why might staring harder at the dump never find it?',
            a: 'XLA:GPU dispatches big matmuls to cuBLAS rather than compiling them itself, so the slow kernel may be a library selection the HLO never shows. Every path ends in a box the compiler does not open: libtpu and LLO on TPU, ptxas and SASS on GPU.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
    ],
  },
]
