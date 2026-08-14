// Machine lessons: the hardware depth layer of the kernel course, homed in
// the units that own each subject (the tpu and ici layer chapters, and the
// machine stage for the GPU contrast). Chip constants come from the scaling
// book and are quoted with their units; nothing here is measured on site.
import type { UnitLessons } from './index'

export const MACHINE_LESSONS: UnitLessons[] = [
  {
    unit: 'l:tpu',
    coversGuide: true,
    lessons: [
      {
        id: 'tpu-chip',
        num: 1,
        title: 'The TPU chip',
        lede: 'A TPU is one bet cast in silicon: almost all of the work is matrix multiplication, so build one enormous unit for exactly that and spend the rest of the chip feeding it.',
        goal: 'Given a kernel and a TPU generation, name which unit runs each line of it and where every byte sits on its way to the MXU.',
        sections: [
          {
            h: 'one bet, cast in silicon',
            ps: [
              "A CPU spends most of its area deciding what to do next: branch predictors, reorder buffers, cache hierarchies that guess your access pattern. A TPU spends almost none. The designers looked at what training and serving actually run and found matrix multiplication, over and over, at every scale. So the chip commits. One huge matrix unit does that work, a vector unit handles the arithmetic around it, and everything else on the die exists to keep those two fed.",
              'The commitment shows up in what is missing. There is no branch predictor to speak of, no out-of-order engine, no hardware cache deciding for you what stays close to the compute. When this course later has you choreograph data by hand with a `BlockSpec`, this lesson is the reason: the hardware chose not to guess, which means someone has to say. On a TPU that someone is the compiler, and in Pallas it is you.',
            ],
          },
          {
            h: 'the systolic array',
            ps: [
              'Picture the multiply as a piece of machinery rather than a loop. Weights get parked in a square grid of cells, one value per cell, and stay there. Activations enter from one edge, one diagonal per cycle, and every cell they pass does a single multiply-accumulate: take the incoming value, multiply by the parked weight, add to the partial sum flowing through, pass both along. Results drain out the far edge. Nothing fetches an instruction per operation and nothing asks a cache for its operand; the schedule is the geometry. An array built to pulse data through itself this way is called a **systolic array**, and the MXU is one: `128x128` cells on most generations, `256x256` on v6e.',
              "The geometry explains the costs you will keep meeting. The array has a fill phase while the first diagonals march in, a steady state where all cells work every cycle, and a drain phase at the end. Small matmuls live mostly in fill and drain, which is why they waste the unit; the arithmetic below a `(8, 128)`-shaped tile can't even occupy one edge. Big matmuls amortize the ramp and run the array flat out. When stage 0 had you predict that a skinny matmul lands memory-bound and a square one compute-bound, this grid is the machinery behind the prediction.",
              '>> The schedule is the geometry: data moves, and the movement is the computation.',
            ],
            exhibit: 'systolic',
          },
          {
            h: 'everything that is not a matmul',
            ps: [
              'Softmax needs an exponential, layernorm needs a square root, and neither is a matrix product. That work goes to the VPU, a vector unit that applies the same operation across wide registers of data. It is the second citizen of the chip by area and the first by variety: adds, multiplies, exponentials, comparisons, casts, everything elementwise a kernel does between matmuls. On v5p each core carries 64 32-bit vector registers to stage that work.',
              "The split matters for how you read a kernel's cost. A fused attention kernel is not one workload; it is matmuls on the MXU with exponentials and rescaling on the VPU stitched between them. If the elementwise work is thin, it hides behind the matmuls. If it is thick, the VPU becomes the bottleneck while the MXU idles, and no roofline over FLOPs alone will tell you. The habit to build now: when you meet a kernel, sort its lines into MXU lines and VPU lines before you estimate anything.",
            ],
          },
          {
            h: 'the scratchpad and the staging',
            ps: [
              "Between the compute units and HBM sits VMEM, roughly `128 MiB` of on-chip memory on v5e. Calling it a cache would miss the design. A cache decides for you what stays close; VMEM holds exactly what software staged into it, nothing more. Every block a kernel touches was placed there by a DMA that something explicitly issued, and the compute units read only from there. Kernel engineering on TPU is mostly the choreography of that staging: which block arrives when, and whether the next transfer overlaps the current compute.",
              'Two smaller memories complete the picture. SMEM holds scalars: loop bounds, block indices, flags, the values a kernel branches on. And a scalar core runs alongside the vector units executing the control flow your kernel compiles to, issuing the DMA descriptors that move blocks from HBM into VMEM. The guide sections below walk these two in detail, with the diagram most block diagrams omit; here it is enough to hold the division: MXU and VPU compute, the scalar core decides and fetches, VMEM is where the two worlds meet.',
            ],
          },
          {
            h: 'sparsecore, the odd one out',
            ps: [
              "One more unit sits on the die and no kernel in this course touches it. SparseCore exists for embedding lookups, the wide scattered reads that recommendation models hammer and that a dense `128x128` grid is exactly wrong for. It earns its area on those workloads and stays dark on ours. It belongs in your picture of the machine anyway, as the exception that proves the bet: when a workload class mattered enough and fit the MXU badly enough, it got its own silicon rather than bending the array.",
            ],
          },
        ],
        guide: { id: 'tpu', sections: [0] },
        readings: [
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the backbone text; every constant in this lesson lives in its tables' },
          { label: 'Cloud TPU system architecture', url: 'https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm', note: 'the vendor description of the same units, generation by generation' },
        ],
        check: [
          {
            q: "A kernel's inner loop is mostly exponentials and rescaling, with a small matmul at the end. Which unit is your bottleneck candidate, and why won't a FLOPs roofline warn you?",
            a: 'The VPU. The elementwise work runs there while the MXU idles, and a roofline over FLOPs mostly counts matmul arithmetic, so a VPU-bound kernel can sit far below the FLOPs roof and still be at its own ceiling.',
          },
          {
            q: 'Why does a small matmul waste the MXU even when its operands are already resident in VMEM?',
            a: 'The systolic array spends its opening cycles filling the wavefront and its closing cycles draining it. A small operand spends most of its time in those ramps, so few cycles run with every cell active, no matter what the memory did.',
          },
          {
            q: 'VMEM is often described as a cache. What breaks in that analogy?',
            a: 'A cache decides for itself what stays close. VMEM holds exactly what software staged into it by DMA, nothing is fetched or evicted behind your back, and that is why the staging is yours to choreograph.',
          },
        ],
        work: [
          { id: 'systolic', label: 'systolic stepper: run a full pass and name the three phases', href: '#the-systolic-array' },
          { id: 'sort', label: 'take one kernel from stage 1 and sort its lines into MXU lines and VPU lines' },
          { id: 'check', label: 'answer the three checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'generations',
        num: 2,
        title: 'Six generations of the same idea',
        lede: 'Every TPU generation is the same machine with different constants. The constants are the personality, and you can read a generation like a datasheet once you know which ratios matter.',
        goal: 'Given the spec table and a kernel you know, predict how the kernel’s bottleneck moves when it lands on a different generation.',
        sections: [
          {
            h: 'what a generation changes',
            ps: [
              'The architecture of the previous lesson has held for a decade: a matrix unit, a vector unit, a software-managed scratchpad, HBM behind them, links to neighbors. What moves between generations is the numbers, and the numbers move unevenly. Compute has grown faster than memory bandwidth in almost every step, which drags one derived quantity upward: the ridge, peak FLOPs divided by HBM bytes per second. That single ratio is the personality of a generation. It says which kernels the chip rewards and which it starves.',
            ],
          },
          {
            h: 'the table',
            ps: [
              'Read the table as columns of the same machine growing at different rates. Units are exact: FLOPs per second at bf16, bytes and bytes per second for memory, one-way bytes per second per link for ICI.',
            ],
            table: {
              caption: 'chip constants from jax-ml.github.io/scaling-book/tpus/ (retrieved 2026-08-06) · blank cells are figures the source does not state',
              cols: ['chip', 'bf16 FLOPs/s', 'HBM', 'HBM B/s', 'ICI one-way B/s', 'topology', 'max slice'],
              rows: [
                ['v3', '1.4e14', '32 GB', '9.0e11', '1.0e11', '2D torus', '32x32'],
                ['v4p', '2.75e14', '32 GB', '1.2e12', '4.5e10', '3D torus', '16x16x16'],
                ['v5e', '1.97e14', '16 GB', '8.2e11', '4.5e10', '2D torus', '16x16'],
                ['v5p', '4.59e14', '96 GB', '2.8e12', '9.0e10', '3D torus', '16x20x28'],
                ['v6e', '9.20e14', '32 GB', '1.6e12', '9.0e10', '2D torus', '16x16'],
                ['7x', '2.30e15', '192 GB', '7.4e12', '9.0e10', '3D torus', '4x4x576'],
              ],
            },
          },
          {
            h: 'reading the table like an engineer',
            ps: [
              "Take the step from v5e to v6e, the two chips this site's bench actually ran. Compute jumped 4.7x, from `1.97e14` to `9.20e14`, because the MXU doubled in both dimensions. HBM bandwidth only doubled, `8.2e11` to `1.6e12`. Divide and the ridge moves from about `240` FLOPs per byte to about `575`. Every op whose intensity sits between those two numbers changed teams: compute-bound on v5e, memory-bound on v6e. A kernel tuned on one chip can cross the ridge on the next without a single line changing, and the table told you before the profiler did.",
              "Notice also what v5e is in this table: the only generation whose compute went down relative to its predecessor's line. It is the efficiency part, 16 GB of HBM and a quarter of v5p's bandwidth, priced for serving. The lettered split matters when you rent: e parts trade capacity and bandwidth for cost, p parts keep the headline numbers. Stage 0 taught the roofline as a skill; this table is the terrain the skill runs on, and the roofline playground on that page now carries every generation listed here.",
            ],
          },
          {
            h: 'cores, chips, and what counts as one device',
            ps: [
              "One wrinkle keeps confusing profiler output and device counts: a v5p chip carries two cores, each with its own MXU and VMEM, and `jax.devices()` shows each as a device. v5e and v6e run one core per chip, so chip and device coincide. The table's host rows explain the other ratio you meet in practice: a v5e host manages a `4x2` block of chips, a v5p host a `2x2x1` block. None of this changes the math of a kernel; all of it changes what the numbers in your tooling refer to, and reading them wrong by a factor of two is a rite of passage this paragraph exists to skip.",
            ],
          },
        ],
        readings: [
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the source of every row; check the live tables, they gain generations' },
          { label: 'Cloud TPU pricing and configurations', url: 'https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm', note: 'the e and p split as the vendor frames it' },
        ],
        check: [
          {
            q: "What single ratio is a generation's personality, and why?",
            a: 'Peak FLOPs over HBM bytes per second, the ridge: it says which kernels the chip rewards and which it starves, and it moves whenever compute and bandwidth grow unevenly.',
          },
          {
            q: 'A kernel tuned on v5e lands on v6e with no code change. What can flip?',
            a: "Its side of the ridge: v6e's ridge sits near 575 FLOPs per byte against v5e's 240, so any op between those intensities switches from compute-bound to memory-bound.",
          },
        ],
        work: [
          { id: 'ridge', label: 'compute the ridge for all six rows and rank the generations by it' },
          { id: 'cross', label: 'find one op from stage 0 that crosses the ridge between v5e and v6e' },
        ],
      },
      {
        id: 'two-ways-to-hide-latency',
        num: 3,
        title: 'Two ways to hide latency',
        lede: 'A GPU hides memory latency with thousands of threads. A TPU has one big core and no threads to switch, so every wait has to be planned away.',
        goal: 'Explain both latency-hiding strategies and name who does the hiding on each machine: the warp scheduler at runtime, or the compiled pipeline ahead of time.',
        sections: [],
        guide: { id: 'tpu', sections: [1, 2] },
        readings: [
          { label: 'Scaling book · How to think about GPUs', url: 'https://jax-ml.github.io/scaling-book/gpus/', note: 'the thread-switching half of the contrast' },
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the pipelined half' },
        ],
        check: [
          {
            q: 'Both machines wait on HBM. What hides the wait on each?',
            a: 'The GPU switches among resident warps at runtime, so another thread computes while one waits. The TPU runs one sequential core, so the compiler and the pipeline overlap transfers against compute ahead of time.',
          },
          {
            q: 'Why does the TPU approach demand more of the compiler?',
            a: 'There is no runtime scheduler to cover a miss: any wait not planned away at compile time is a stall the machine simply takes.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'xprof-and-the-three-accounts',
        num: 4,
        title: 'XProf, and the three accounts',
        lede: 'The profiler is how prediction meets the machine, and a program has three cost tellings that this site once caught disagreeing in public.',
        goal: 'Capture a trace both ways, then rank the cost model, the roofline, and the timeline by trustworthiness for the question you are asking.',
        sections: [],
        guide: { id: 'tpu', sections: [3, 4, 5] },
        readings: [
          { label: 'JAX profiling', url: 'https://docs.jax.dev/en/latest/profiling.html', note: 'the capture API and the viewer, officially' },
          { label: 'Scaling book · rooflines', url: 'https://jax-ml.github.io/scaling-book/roofline/', note: 'the prediction the trace is judged against' },
        ],
        check: [
          {
            q: 'What are the three accounts of a program, and which one arbitrates?',
            a: 'The cost model from compiled.cost_analysis(), the roofline prediction, and the measured timeline. The timeline arbitrates: the first two predict, the device plane records what ran.',
          },
          {
            q: 'What are the two ways to reach XProf?',
            a: 'Programmatically, wrapping the code you care about with the capture API, and interactively through the profiler UI; both land in the same viewer.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'the-timeline-op-by-op',
        num: 5,
        title: 'The timeline, op by op',
        lede: 'One plane of the trace matters for kernel work: the device plane, where every event carries a name you already know how to read.',
        goal: 'Walk a device-plane trace, match events to fusions and custom calls by name, and say what the host and device sides each contribute to a capture.',
        sections: [],
        guide: { id: 'tpu', sections: [6, 7] },
        readings: [
          { label: 'JAX profiling', url: 'https://docs.jax.dev/en/latest/profiling.html', note: 'where the planes come from' },
          { label: 'XLA tools', url: 'https://openxla.org/xla/tools', note: 'the dump the event names trace back to' },
        ],
        check: [
          {
            q: 'Which plane answers kernel questions, and what is on it?',
            a: 'The device plane (/device:TPU:0): the ops that actually ran, under names that match the compiled module, fusions and custom calls included.',
          },
          {
            q: 'Where does a capture come from?',
            a: 'Two recorders: the host runtime logs python and dispatch events while the TPU side logs device execution, and the viewer aligns the two.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'reading-it-like-an-operator',
        num: 6,
        title: 'Reading it like an operator',
        lede: 'Six habits turn a wall of events into a diagnosis, and the first is always the same: name the envelope before judging anything inside it.',
        goal: 'Apply the six operator habits to a fresh trace and produce a one-sentence diagnosis with the envelope number attached.',
        sections: [],
        guide: { id: 'tpu', sections: [8, 9] },
        readings: [
          { label: 'JAX profiling', url: 'https://docs.jax.dev/en/latest/profiling.html', note: 'the tool the habits run on' },
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the constants a diagnosis quotes' },
        ],
        check: [
          {
            q: 'What is the envelope, and why does it come first?',
            a: 'The jit event for the function: wall time per call. Every other number is judged against it, so naming it first anchors the reading.',
          },
          {
            q: 'The compute events sum to far less than the envelope. What is the habit?',
            a: 'Find what fills the gap: the difference is unhidden transfer or overhead, and chasing it beats tuning a kernel that already looks fast.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
    ],
  },
  {
    unit: 'l:ici',
    coversGuide: true,
    lessons: [
      {
        id: 'tpu-fabric',
        num: 1,
        title: 'The TPU fabric',
        lede: 'TPUs scale by wiring each chip to its neighbors and to nothing else. Everything about distributed kernels follows from that one wiring decision.',
        goal: 'Given a slice shape and a collective, count the hops and put a lower bound on its time using only link constants.',
        sections: [
          {
            h: 'neighbors, not switches',
            ps: [
              "Ask how 4096 chips talk to each other and the obvious answer is a network: switches in the middle, every chip a cable away from every other. TPUs refuse the switch. Each chip carries direct links, ICI, to its physical neighbors: four of them on v5e and v6e, six on v4p, v5p, and 7x. A message to a distant chip hops chip to chip, forwarded by the chips between. What the design buys is brutal simplicity of scale: doubling the pod doubles the chips and the cables in the same proportion, with no switch tier growing in the middle to pay for.",
              'What it costs is distance. Two chips across the pod are many hops apart, and every hop spends link bandwidth on forwarding. The entire discipline of distributed TPU work, deciding which axis of your mesh maps to which axis of the machine, is the management of that distance.',
            ],
          },
          {
            h: 'the torus and the wraparound',
            ps: [
              'Four neighbors make a grid; six make a 3D lattice. Then the edges fold: the last chip in a row links back to the first, turning each straight axis into a ring. A grid whose every axis is a ring is a **torus**, and the fold is worth exactly a factor of two, because the farthest chip on a ring of n is n/2 hops away instead of n-1. The fold is physical cabling, so it exists only when a slice spans a full axis: on 3D generations a slice gets wraparound as full `4x4x4` cubes and their multiples, on 2D generations when an axis reaches the full 16.',
              'The maximum shapes from the generation table are torus dimensions: `16x16` on v5e and v6e, `16x16x16` on v4p, `16x20x28` on v5p, and the tall `4x4x576` of a full 7x pod, 9216 chips. Pick two chips in the stepper below and watch what the fold does to the hop count; then take an axis away and watch it come back.',
            ],
            exhibit: 'torus',
          },
          {
            h: 'link arithmetic',
            ps: [
              "The links themselves are constants you can hold: `4.5e10` bytes per second one-way per link on v4p and v5e, `9.0e10` on v5p, v6e, and 7x, double it for both directions at once. A chip's total egress is its per-link rate times its neighbor count, so a v6e chip can push `3.6e11` bytes per second outward if all four links run hot, and a collective is well designed exactly when they do.",
              "That is enough to bound real operations from the armchair. An all-gather of D bytes around one ring axis of n chips must move each shard past n-1 positions; run the ring perfectly and the time floor is D(n-1)/n divided by the one-way link rate. Nothing in that formula is measured, and yet it called the shape of every ring-collective benchmark on this site's bench within the factor the roofline habit taught you to expect. When a measured collective misses the floor badly, the miss has a name (a congested axis, a slice without wraparound, DCN in the path), and finding which is the debugging.",
              '>> A collective is well designed exactly when every link is busy.',
            ],
          },
          {
            h: 'past the slice',
            ps: [
              'ICI ends at the slice boundary. Beyond it, traffic falls onto the data-center network, and the cliff is the point of this section: DCN carries `3.125e9` bytes per second per chip on v5e, `6.25e9` on v5p, `1.25e10` on v6e and 7x, between one and two orders of magnitude below the ICI numbers above. The host connection has the same flavor, PCIe at around `1.6e10` bytes per second per TPU generally and `3.2e10` on v6e. The hierarchy dictates strategy outright: the parallelism axes that communicate most stay inside the slice, and whatever crosses slices had better communicate rarely. When a later stage places data parallel across DCN and model parallel inside ICI, it is reading this table, not expressing a preference.',
            ],
          },
        ],
        readings: [
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'link constants, torus shapes, and the wraparound rules, with diagrams' },
          { label: 'Scaling book · Sharding matmuls', url: 'https://jax-ml.github.io/scaling-book/sharding/', note: 'where the hop arithmetic starts paying rent' },
        ],
        check: [
          {
            q: 'What does wraparound buy, and when does a slice have it?',
            a: "A factor of two on worst-case hops, since a ring's far chip is n/2 away instead of n-1. It exists when the slice spans full 4x4x4 cubes on 3D generations or a full axis of 16 on 2D ones.",
          },
          {
            q: 'Why do the chattiest parallelism axes stay inside the slice?',
            a: 'DCN carries one to two orders of magnitude less per chip than ICI, so whatever crosses slices has to communicate rarely.',
          },
        ],
        work: [
          { id: 'torus', label: 'torus stepper: find the worst-case pair with and without wraparound', href: '#the-torus-and-the-wraparound' },
          { id: 'floor', label: 'bound an all-gather on one v6e ring axis by hand, then check it against the bench' },
        ],
      },
      {
        id: 'the-vocabulary-as-ring-movements',
        num: 2,
        title: 'The vocabulary, as ring movements',
        lede: 'All-gather, reduce-scatter, all-reduce: each one is the same ring step repeated, shards moving neighbor to neighbor by remote DMA.',
        goal: "Describe each collective as ring movements precisely enough to draw every chip's shards after any given step.",
        sections: [],
        guide: { id: 'ici', sections: [0] },
        readings: [
          { label: 'Scaling book · sharded matmuls', url: 'https://jax-ml.github.io/scaling-book/sharding/', note: 'where these movements start earning money' },
          { label: 'JAX · shard_map', url: 'https://docs.jax.dev/en/latest/notebooks/shard_map.html', note: 'the collectives as you call them' },
        ],
        check: [
          {
            q: 'What single primitive underlies the ring collectives?',
            a: 'A neighbor-to-neighbor shard move by remote DMA, repeated around the ring; the collectives differ only in what each chip does with what arrives.',
          },
          {
            q: 'How does all-reduce decompose on a ring?',
            a: 'Reduce-scatter then all-gather: first every chip ends holding one fully reduced shard, then the shards circulate until everyone holds all of them.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'cost-formulas-in-your-head',
        num: 3,
        title: 'Cost formulas you can do in your head',
        lede: 'Ring collective cost is desk arithmetic, not measurement: bytes, chips, and one link rate are the whole formula.',
        goal: 'Bound any ring collective on a given axis from link constants alone, and say when the bound is tight.',
        sections: [],
        guide: { id: 'ici', sections: [1] },
        readings: [
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the link constants the formulas quote' },
          { label: 'Scaling book · sharded matmuls', url: 'https://jax-ml.github.io/scaling-book/sharding/', note: 'the formulas at work on real layouts' },
        ],
        check: [
          {
            q: 'What is the all-gather floor for D bytes on a ring of n chips?',
            a: 'D(n-1)/n over the one-way link rate: each shard travels n-1 hops, and the bound is tight exactly when every link stays busy the whole time.',
          },
          {
            q: 'When does the desk formula miss, and what does the miss mean?',
            a: 'When a link is contended, the slice lacks wraparound, or traffic leaves ICI. A measured collective far off its floor is naming its own problem.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'the-collective-is-a-kernel',
        num: 4,
        title: 'The collective is a kernel',
        lede: 'Nothing about a collective is special hardware. It is the same async remote DMA and semaphore pairing, written as a kernel you can read.',
        goal: 'Read a ring collective kernel and point at the remote copy, the semaphore pair, and the accumulation that make it a collective.',
        sections: [],
        guide: { id: 'ici', sections: [2, 3] },
        readings: [
          { label: 'Pallas TPU pipelining', url: 'https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html', note: 'the local version of the same copy discipline' },
          { label: 'JAX · multi-process', url: 'https://docs.jax.dev/en/latest/multi_process.html', note: 'the scale these kernels run at' },
        ],
        check: [
          {
            q: 'What distinguishes a collective kernel from the manual-DMA kernel you already know?',
            a: 'Only the destination: the async copy lands on a neighboring chip over ICI instead of in local memory. The issue, wait, semaphore discipline is identical.',
          },
          {
            q: 'Why does that identity matter when a collective hangs?',
            a: 'It is the same bug class as a missed semaphore wait in any kernel, debugged with the same discipline, rather than a network mystery.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: "gpu-fabric",
        num: 5,
        title: "The GPU fabric",
        lede: "Inside a node, every GPU is one hop from every other. Cross the node boundary and you are on a different network with different numbers.",
        goal: "Given a collective and the GPUs it spans, name the layer of fabric that carries it and the bandwidth that layer actually delivers rather than the one on the slide.",
        sections: [
          {
            h: "eight GPUs, one hop",
            ps: [
              "The unit above a single GPU is the node, which is 8 GPUs for everything through B200 and up to 72 for GB200. Inside it they are wired all-to-all through NVSwitches, so any GPU reaches any other in one hop at full bandwidth. The topology has moved around across generations; on an H100 node there are 4 NVSwitches and the GPUs attach to them in a 5 + 4 + 4 + 5 link pattern.",
              "Each NVLink link on Hopper carries `25 GB/s` full-duplex, meaning `25` each way with the two directions independent, so `50` total but never more than `25` in one direction. Each GPU has 18 of those links, which gives `18 * 25 = 450 GB/s` of full-duplex bandwidth from every GPU into the network.",
              "The switches are not the constraint. An NVSwitch carries up to 64 NVLink ports, so four of them come to `64 * 25e9 * 4 = 6.4 TB/s`. The GPUs get there first: at `450 GB/s` each, 8 of them cap the node at `3.6 TB/s`. Split that node in half any way you like and the same `3.6 TB/s` crosses the cut, which is the node's bisection bandwidth.",
            ],
            diagram: "nvlink-node",
          },
          {
            h: "three generations of link",
            ps: [
              "What changed generation to generation is worth reading as three separate decisions. Ampere had 12 links per GPU at `25 GB/s`, for `300`. Hopper left the link speed alone and added ports, 18 of them, for `450`. Blackwell kept 18 ports and doubled the link to `50 GB/s`, for `900`. Meanwhile the switch count inside a node fell from 6 to 4 to 2, because each switch generation carries more.",
              "The last row is really two machines. A B200 node is still 8 GPUs with 2 switches. A GB200 NVL72 puts 72 GPUs into one NVLink domain behind 18 switches, at the same `900 GB/s` per GPU.",
            ],
            table: {
              caption: "NVLink generations, from the scaling book's networking table",
              cols: ["NVLink gen", "GPU gen", "per link (GB/s, full-duplex)", "links / GPU", "per GPU (GB/s)", "node size", "NVSwitches / node"],
              rows: [
                ["3.0", "Ampere", "25", "12", "300", "8", "6"],
                ["4.0", "Hopper", "25", "18", "450", "8", "4"],
                ["5.0", "Blackwell", "50", "18", "900", "8 or 72", "2 or 18"],
              ],
            },
          },
          {
            h: "above the node, a fat tree",
            ps: [
              "Past the NVLink domain the fabric changes technology. Every GPU in a reference H100 node has its own 400 Gbps CX7 NIC into an InfiniBand network, so the node egresses `8 * 400 / 8 = 400 GB/s` into what NVIDIA calls the **scale-out network**, as against the **NVLink domain** below it. Thirty-two of those nodes, 256 GPUs, sit under a single set of 8 leaf InfiniBand switches and are called a Scalable Unit. The cabling is NDR, `50 GB/s` full-duplex per port, through 64-port switches.",
              "Four Scalable Units under 16 top-level spine switches make the reference SuperPod: 1024 GPUs, 512 node-level NVSwitches, 32 leaf IB switches, 16 spine IB switches, 560 switches in total. All leaf switches connect to all spine switches, and the cable and port counts are chosen so that bisecting the pod anywhere still gives every node its full `400 GB/s`. That is what fat tree means here, and it is why AllReduce bandwidth in the scale-out network stays roughly constant as you add nodes.",
              "So the shape of the whole machine is two numbers. `450 GB/s` per GPU while you stay inside a node, `400 GB/s` per node once you leave it. Every sharding decision on a GPU cluster is an argument about which side of that boundary your biggest collective lands on.",
            ],
          },
          {
            h: "what NVL72 changes",
            ps: [
              "GB200 NVL72 racks put 72 GPUs in one NVLink domain, each with the full `900 GB/s` to every other. Counting the egress from that node into the leaf level gives `4 * 18 * 400 / 8 = 3.6 TB/s`, which is 9x an H100 node's `400 GB/s`, exactly matching the 9x in GPU count.",
              "Divide it out and the per-GPU story is flat: `400 / 8` is `50 GB/s` of egress per GPU on an H100 node, and `3600 / 72` is also `50 GB/s` on an NVL72. The rate at which a GPU can talk to the world outside its domain did not change. The domain got nine times bigger, so nine times as much of your traffic never has to leave it, and the rooflines you compute for a sharding move accordingly.",
            ],
          },
          {
            h: "the claimed number and the measured one",
            ps: [
              "An 8xH100 node claims `450 GB/s` per GPU. Measured AllReduce with SHARP disabled gets close to `370 GB/s`, and only at array sizes around 10 GB per device. The estimate is not wrong. It just takes an enormous message to get there.",
              "Smaller messages are much worse, and they are the ones you actually send. A LLaMA-3 70B MLP with 8-way model sharding is `bf16[8192, 3584]`, about 58 MB, and an AllReduce over it achieves only around `150 GB/s` against the nominal `450`. TPUs reach peak bandwidth at considerably smaller messages, which is one of the sharper practical differences between the two fabrics.",
              "Since Hopper, NVIDIA switches support in-network reduction (SHARP): the switch performs the reduction itself and multicasts the result, so each GPU does not have to egress its data twice. In theory that close to halves an AllReduce. In measurements it gives about 30%.",
            ],
          },
          {
            h: "switch against torus",
            ps: [
              "TPUs are wired as 2D or 3D tori, where each chip connects only to its neighbours. A message between two distant TPUs passes through every chip in between, and that forces uniform communication patterns across the mesh. In exchange, the number of links per chip is constant, so a pod grows without any per-chip bandwidth loss, up to 8960 chips on a v5p. A v5p link carries about `90 GB/s`, or `540 GB/s` summed across all axes of the torus.",
              "The GPU tree buys the opposite property. Any GPU can send to any other in an arbitrary pattern, and you can extend the fabric to any size by adding switches or another layer of indirection, which the 4096-GPU configuration does with 128 spine and 64 core switches. You pay for that in latency and in switches, at every level, forever. The generational split is in the protocols themselves: NVLink behaves like a souped-up PCIe with low latency and little protocol overhead but no design for scale or fault tolerance, while InfiniBand is closer to Ethernet, built for large lossy networks.",
              ">> A torus costs the same to wire at any size. A tree costs another layer of switches.",
            ],
            exhibit: "fabric",
          },
        ],
        readings: [
          {
            label: "Scaling book · How to think about GPUs",
            url: "https://jax-ml.github.io/scaling-book/gpus/",
            note: "the networking half of the chapter; do the node bandwidth quiz by hand",
          },
          {
            label: "NVIDIA · DGX H100 SuperPod reference architecture",
            url: "https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf",
            note: "the 1024-GPU design the leaf and spine counts come from",
          },
          {
            label: "NVSwitch at Hot Chips 2022",
            url: "https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf",
            note: "the switch itself, from the people who built it",
          },
          {
            label: "NVIDIA · GB200 NVL72 network fabrics",
            url: "https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html",
            note: "how the 72-GPU domain is cabled once you go above the rack",
          },
        ],
          check: [
          {
            q: 'What are the two numbers that shape a GPU cluster?',
            a: '450 GB/s per GPU inside an H100 node over NVLink, and 400 GB/s per node once traffic leaves for the InfiniBand tree. Sharding decisions are arguments about which side of that boundary a collective lands on.',
          },
          {
            q: 'What did NVL72 change, and what stayed fixed?',
            a: 'The NVLink domain grew to 72 GPUs at the same 900 GB/s each, while per-GPU egress to the outside stayed near 50 GB/s: nine times more traffic never has to leave, and leaving costs what it did.',
          },
        ],
          work: [
            { id: 'fabric', label: 'fabric instrument: move the source and watch which machine cares', href: '#switch-against-torus' },
            { id: 'bound', label: 'bound an 8-GPU all-reduce at 58 MB and set it against the measured 150 GB/s' },
          ],
      },
      {
        id: 'the-optical-patch-panel',
        num: 6,
        title: 'The optical patch panel',
        lede: 'A TPU v4 pod has almost no long fixed cables. Forty-eight optical switches aim mirrors at fibers, and the shape of your slice is a routing table they load when the job starts.',
        goal: 'Count the switches, ports, and fiber strands behind a 4096-chip pod from the block geometry alone, and say what a topology change costs when the wiring never moves.',
        sections: [
          {
            h: 'the cable that had to be light',
            ps: [
              'A wraparound link on a TPU v3 pod sometimes had to reach across the room. Electrical cable gives out at that distance, so those particular links were already optical, and an optical link costs more than ten times what an electrical one costs. Then v4 asked for four times the chips. More of the expensive links, a 2D torus whose bisection bandwidth was thin at that size, and one problem sitting under both: a machine of 4096 chips is rarely a machine where everything works at once.',
              "So TPU v4 stopped fixing the long links in place. Between the racks sits a set of switches that join fiber to fiber by aiming a mirror, an **optical circuit switch**, and Google's is called Palomar. The paper's own word for what it does is plugboard. The pod gets patched per job, and a dead unit gets patched around.",
              'The fabric lesson that opens this chapter took the torus as given: chips wired to neighbors, axes folded into rings. This is where the given comes from. On v4 nobody wired your neighbor at install time. The scheduler chose it, and the choice is a routing table loaded when your slice is allocated.',
            ],
          },
          {
            h: 'a mirror, not a router',
            ps: [
              'Palomar is built on 3D MEMS mirrors, micro-machined and tilted to steer light out of one fiber and into another, and the tilt settles in milliseconds. After that the path is passive. Nothing reads a header, nothing queues, nothing decides, because there is no packet: the light that leaves the source tray is the light that arrives at the destination tray. What the design gains is a list of absences. No congestion. No protocol layers adding latency. Almost no power, since holding a mirror at an angle costs a fraction of what packet processing costs.',
              'The switch is `136x136`, which the paper unpacks as 128 ports plus 8 spares kept for link testing and repairs. Circulators send light both ways down a single fiber, halving the ports and cables the design would otherwise need. Every connection is one input to one output, so the switch cannot fan out and cannot buffer a byte. Whatever shape a job wants has to be expressible as a permutation, set once at slice allocation and held for the life of the job.',
            ],
          },
          {
            h: 'counting the cables',
            ps: [
              'Why `4x4x4`? A cube has the best bisection bandwidth of any 3D box, which argues for building blocks of 64 or 512 chips. Sixty-four chips and their 16 CPU hosts fit comfortably in one rack; 512 chips do not. The rack became the block, wired inside as a `4x4x4` mesh by passive electrical cable, with nothing optical in it at all.',
              "The optics start at the tray. A tray's fiber connector turns electrons into light, and the next conversion back happens at the connector of the destination tray, with nothing in between. Each rack presents six faces, 16 links per face, 96 optical links in total. Wraparound decides where those 96 go: for a dimension to fold into a ring, the links on its two opposing faces must arrive at the same switch. That pairs them, `6 x 16 / 2 = 48`, so a rack cables to 48 switches and every rack cables to every one of them.",
              "The same counting works one level down. A v4 chip has six ICI links, two on the board and four leaving through OSFP connectors. Four chips per board with two on-board links each pair into the `2x2` mesh the PCB embeds, and four off-board links times four chips are the 16 OSFP connectors on the tray's front panel.",
              'Two further counts fall out, and both close against published figures, which is the reason to do the arithmetic rather than read it. Sixty-four racks each hand one in-and-out pair to a given switch, which is 128 fiber ends, exactly the usable port count of a `136x136` Palomar. Multiply by 48 switches and you get 6,144 strands, which is what the Hot Chips deck says a system contains.',
            ],
            table: {
              caption: 'the v4 wiring, counted from the ISCA 2023 paper (sections 2.1 and 2.2) and the Hot Chips 2023 deck',
              cols: ['quantity', 'value', 'where it comes from'],
              rows: [
                ['chips per building block', '64, a `4x4x4` cube filling one rack', 'cubes maximise bisection, and 64 chips plus 16 hosts fit a rack'],
                ['link ends per block', '384, from 64 chips at 6 ICI links each', 'derived; the rack-internal mesh consumes 288 of them as 144 links'],
                ['optical links per block', '96, 16 on each of 6 faces', 'stated in the paper; equals the 384 ends minus the 288 spent inside the rack'],
                ['switches per block', '48, from `6 x 16 / 2`', 'opposing faces must land on the same switch for the wraparound to exist'],
                ['ports used per switch', '128, one in-and-out pair from each of 64 racks', 'derived; matches the published 128 usable ports of a `136x136` Palomar'],
                ['fiber strands per system', '6,144, from `48 x 128`', 'derived; matches the deck\'s "6,144 Fiber Strands"'],
                ['chips per system', '4,096, from 64 blocks of 64', '64 racks deployed in 8 groups of 8'],
              ],
            },
          },
          {
            h: 'shapes without recabling',
            ps: [
              'A v4 slice is any `4i x 4j x 4k` with the dimensions sorted, so 192 chips can be `4x4x12` and nothing has to be a power of two. What the shape decides is which distances your collectives pay for. A cigar like `4x4x32` suits pipeline parallelism, where traffic runs mostly along one axis. A cube like `8x8x8` gives the highest bisection bandwidth, which is what embedding-heavy work wants. The cost formulas earlier in this chapter take a ring length n and hand back a floor; the shape is where n comes from.',
              'Past the rectangle there is a second choice. A twisted torus rewires some of the links between cubes so the worst-case distance shrinks, and on this machine that rewiring is not physical. The optical connections move from a rectangular torus to a twisted one by loading different routing tables, while the electrical cabling inside the racks never moves. Measured on all-to-all with large aggregate transfers and 4 KiB DMAs, twisting bought 1.63x on a `4x4x8` slice and 1.31x on `4x8x8`.',
              'Not every slice can twist. The geometry has to be `n x n x 2n` or `n x 2n x 2n` with n at least 4, which covered 33% of slices in a November 2022 production sample; the ones that actually ran twisted were 28%. Counted only among slices of a full cube or larger, 40% ran twisted. Hold the other end of that distribution too: 29% of slices were smaller than a `4x4x4` cube, and those get a 2D mesh with no wraparound at all.',
              "The paper's sharpest performance case mixes two changes, and reading it as topology alone would be wrong. A 512-chip LLM slice went from 17.9 to 41.3 sequences per second, 2.3x, when the shape moved from `4x8x16` to `8x8x8` and the model-parallel split moved from `16x32` to `64x8` with 1D/2D activation and weight partitioning. A GPT-3 pre-training case, starting from an expert's configuration rather than a novice's, gained 1.2x from a similar joint change. Reconfiguring the topology pays when the partitioning moves with it.",
              '>> The shape of the machine is a routing table, not a recabling.',
            ],
          },
          {
            h: 'skipping the dead ones',
            ps: [
              'Four TPUs share a CPU host, so a 4096-chip system carries 1024 hosts, and the host is the part most likely to be down. Without an OCS the chips behind a dead host are out of the slice, and the paper states the requirement plainly: host availability has to reach 99.9% before large slices deliver reasonable goodput. With the switches in the path, 99.0% and 99.5% still give fair goodput at most slice sizes, because the scheduler patches a live block in where the dead one sat.',
              'Goodput at the largest sizes stays awkward anyway, for a reason that has nothing to do with optics. A pool needs spares. Ask for 2048 of 4096 chips and half the machine sits spare, which the paper reads as 50% goodput; ask for 3072 and a quarter sits spare, for 75%. Two 2K slices out of a 4K machine is not a realistic schedule.',
              'The plugboard also changed how a pod gets built. A v3 system was unusable until all 1024 chips and every cable were installed and tested, so one late component held the whole machine. On v4 each `4x4x4` block went into production as soon as its own 64 chips and cables were in and tested. Racks arrive, racks join the pool.',
              'Scheduling got easier in the same move. A 256-chip slice on v3 meant finding 256 contiguous idle chips; on v4 it means finding four blocks anywhere in the machine. And since each job holds its own physical light paths, slices are air gapped from one another, which is the isolation story for a pod with several tenants on it.',
            ],
          },
          {
            h: 'the bill, and the end of the record',
            ps: [
              'All of that is bought with an optical fabric, and the paper prices the fabric: under 5% of total TPU v4 supercomputer capital cost and under 3% of total power, counting the optics modules, the fiber, and the OCS infrastructure. The power figure follows from what the switch does. A mirror held at an angle draws almost nothing next to a switch that parses packets.',
              "The counterfactual in the paper's discussion is the comparison worth carrying. Replacing the 48 128-port switches with InfiniBand, following NVIDIA's own fat-tree guidance, takes 568 IB switches for the same 4096 chips, with the NICs on top. Each ICI link carries 400 Gbit/s against IB's 200. The switch-against-torus argument in the GPU fabric lesson is this same argument read from the other machine.",
              'What is published thins out fast after v4. The engineering record for the OCS is the ISCA paper and the Hot Chips deck, both describing v4, and the ISCA papers stop there. The ICI figures themselves come in different currencies per generation, which matters when you try to compare them. The deck and the paper do agree for v4 once you convert: 400 Gb/s each direction is `50 GB/s`, four off-board links plus two on-board make six, and six times fifty is the 300 GB/s per chip that Table 4 states.',
              'A per-link number for v5e, v5p, or v6e is not something you can pull from those pages, because they publish an aggregate. The one-way per-link constants in the generations table over in the TPU chapter come from the scaling book, which is secondary and says so. One more mismatch worth knowing about: the v4 product page calls the topology a 3D mesh where the paper calls it a 3D torus, and the wraparound rule from the fabric lesson is what reconciles them.',
            ],
            table: {
              caption: "ICI as each source states it; Google's per-chip GBps figures are aggregate and bidirectional, the papers' are per link",
              cols: ['generation', 'published figure', 'source'],
              rows: [
                ['v3', '4 links @ 70 GB/s', 'ISCA 2023, Table 4'],
                ['v4', '6 links @ 50 GB/s, so 300 GB/s per chip', 'ISCA 2023, Table 4'],
                ['v4, physically', '4 OSFP connectors off-board at 400 Gb/s each direction, plus 2 on-board links', 'Hot Chips 2023, slide 8'],
                ['v5e', '400 GBps per chip, 4 ports', 'Google Cloud v5e page'],
                ['v5p', '1200 GBps per chip', 'Google Cloud v5p page'],
                ['v6e', '800 GBps per chip, 4 ports', 'Google Cloud v6e page'],
              ],
            },
          },
        ],
        readings: [
          {
            label: 'Jouppi et al. · TPU v4, an optically reconfigurable supercomputer (ISCA 2023)',
            url: 'https://arxiv.org/abs/2304.01433',
            note: 'sections 2.1 through 2.10 are the whole OCS story, cabling counts included',
          },
          {
            label: 'Jouppi and Swing · Hot Chips 2023 deck',
            url: 'https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf',
            note: 'the physical system: 64 racks, 48 switches, 6,144 fiber strands, and the board photos',
          },
          {
            label: 'Google Cloud · TPU v4',
            url: 'https://docs.cloud.google.com/tpu/docs/v4',
            note: 'the product-facing wording, which says 3D mesh where the paper says 3D torus',
          },
        ],
        check: [
          {
            q: 'Why does a 4x4x4 block cable to exactly 48 optical switches?',
            a: 'Ninety-six optical links leave the block, 16 on each of six faces, and a wraparound link needs the two opposing faces of a dimension to arrive at the same switch. That pairs the links: 6 x 16 / 2 = 48.',
          },
          {
            q: 'The paper reports 2.3x on a 512-chip LLM slice. What actually changed?',
            a: 'Two things together, not the topology alone: the shape went from 4x8x16 to 8x8x8 and the model-parallel split from 16x32 to 64x8 with 1D/2D activation and weight partitioning. Throughput went from 17.9 to 41.3 sequences per second.',
          },
          {
            q: 'What host availability does a large slice need without an OCS, and what changes with one?',
            a: 'The paper puts it at 99.9% without. With the switches in the path, 99.0% and 99.5% still give fair goodput at most slice sizes, because a dead host is patched around instead of stranding the 4 chips behind it.',
          },
        ],
        work: [
          { id: 'ports', label: 'close the port arithmetic yourself: 64 racks to 128 ports to 6,144 strands', href: '#counting-the-cables' },
          { id: 'shape', label: 'pick a 512-chip shape for pipeline parallelism and one for all-to-all, and say what each costs in ring hops', href: '#shapes-without-recabling' },
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
    ],
  },
  {
    unit: 's:machine',
    lessons: [
      {
        id: "gpu-chip",
        num: 1,
        title: "The GPU chip",
        lede: "A TPU v5p has at most two big compute units. An H100 has 132 small ones, and most of the other differences follow from that count.",
        goal: "Given an H100 SM, name what sits in each of its four subpartitions and say which resource (registers, SMEM, or resident warps) your kernel runs out of first.",
        sections: [
          {
            h: "a hundred and thirty-two small machines",
            ps: [
              "Count the independent compute units on each chip and the two designs separate on the first line. A TPU has at most two TensorCores. An H100 has `132` streaming multiprocessors and a B200 has `148`, each one independent of the others, so a GPU can run hundreds of separate tasks at once. Any single SM is much weaker than a TPU TensorCore. The chip as a whole is much more flexible.",
              "That independence has a ceiling, and the ceiling is the L2 cache. All `132` SMs share roughly `50 MB` of it, which means units that are architecturally independent still end up coordinating in practice, because they are competing for the same lines. You change a memory access pattern in one kernel and a different kernel's throughput moves. The scaling book's phrasing for this is action at a distance, and it is the reason GPU programmers talk about cache behavior the way TPU programmers talk about block sizes.",
            ],
          },
          {
            h: "inside one SM",
            ps: [
              "Open an SM and you find four identical quadrants, which NVIDIA calls subpartitions. Each one holds a Tensor Core, a warp scheduler, `32` fp32 CUDA cores that all execute the same instruction in a given cycle, and its own register file of `16,384` 32-bit words. Four register files put `256 kB` of register memory in every SM. Next to them, shared across all four subpartitions, sits `256 kB` of SMEM: an on-chip cache you can either leave to the hardware or drive yourself as shared memory.",
              "Nearly all the arithmetic lives in the Tensor Cores. An H100 does `990` bf16 TFLOPs/s through them against `66` TFLOPs/s from the CUDA cores, a factor of about 15. Divide that `990` by 132 SMs, 4 subpartitions, and the `1.76 GHz` the scaling book uses for the arithmetic, and each Tensor Core is doing roughly `1024` bf16 FLOPs per cycle, about the work of an `8x8x8` matmul. NVIDIA publishes almost nothing about the internals, so read that as an inference from the published totals rather than a datasheet line. The same division on a B200 lands near `2048`.",
              "Blackwell grew the Tensor Core past what the older memories could feed it. In Ampere a single warp could feed the unit, in Hopper it takes a full warpgroup, and in Blackwell it is fed from two SMs at once; by then the accumulator no longer fits in registers or SMEM. So a B200 adds a separate `256 kB` of Tensor Memory per SM to hold the arguments.",
            ],
            diagram: "sm-anatomy",
          },
          {
            h: "SIMT, and the divergence a TPU cannot have",
            ps: [
              "The `32` CUDA cores in a subpartition execute the same instruction each cycle, which is exactly what the ALUs in a TPU's VPU do. The difference sits one level up. Each CUDA core, called a thread in CUDA's vocabulary, has its own instruction pointer, and a warp is the group of 32 threads that one warp scheduler dispatches together. NVIDIA calls the model **SIMT**, single instruction multiple threads, against the TPU's **SIMD**.",
              "So you can write a branch whose two sides are taken by different threads of the same warp. The hardware will not refuse it. It runs both sides and masks off the cores that should not be executing the current one, so a warp that diverges pays for both branches. Nothing on a TPU can do this: the VPU has one instruction pointer for the whole unit and no per-lane state to diverge with. Divergence is not a TPU hazard you learn to avoid, it is a category that does not exist there. Threads are also freer about memory, reaching individual values in shared memory and keeping per-thread state, where the VPU only operates on contiguous blocks.",
              "Scheduling is the second freedom. An SM can hold up to `64` resident warps and the warp scheduler switches between them to hide memory loads, roughly the way a multi-threaded CPU does. Registers are what stop you. A thread can address at most `256` registers, and at that ceiling only `8` warps fit at once in an SM's `256 kB` register file. TPUs are single threaded by comparison, which is why their loads have to be pipelined by the compiler instead of covered by a scheduler at runtime.",
            ],
          },
          {
            h: "the memory ladder",
            ps: [
              "Below registers and SMEM sits the L2, about `50 MB` on an H100. It is physically split in two, so half the SMs reach `25 MB` a piece, with a link between the halves at lower bandwidth. NVIDIA does not publish its bandwidth; measurements put it near `5.5 TB/s`, roughly 1.6x HBM bandwidth, and because it is full-duplex the effective bidirectional figure is closer to 3x. It is the nearest thing the GPU has to VMEM by size. It is much slower, and unlike VMEM you do not control it.",
              "Under everything is HBM: `80 GB` at `3.4e12 bytes/s` on an H100, `192 GB` at `8.0e12 bytes/s` on a B200. Those two numbers are the ones you divide against FLOPs to get a ridge, the same arithmetic stage 0 does for a v5e, and the reason the whole memory ladder above exists is to keep traffic off that bottom rung.",
            ],
            table: {
              caption: "per-chip capacities, from the scaling book's spec tables",
              cols: ["level", "H100", "B200"],
              rows: [
                ["SMs / chip", "132", "148"],
                ["registers / SM", "256 kB", "256 kB"],
                ["SMEM / SM", "256 kB", "256 kB"],
                ["TMEM / SM", "none", "256 kB"],
                ["L2 / chip", "50 MB", "126 MB"],
                ["HBM / chip", "80 GB", "192 GB"],
                ["HBM bandwidth", "3.4e12 B/s", "8.0e12 B/s"],
              ],
            },
          },
          {
            h: "two things named Tensor Core",
            ps: [
              "One term does two different jobs across the two vendors, and it causes real confusion when you read both sets of docs in a week. On a TPU, the TensorCore is the umbrella unit: it contains the MXU, the VPU, and the machinery around them. On a GPU, the Tensor Core is only the matrix multiplication sub-unit inside a subpartition, one of four in an SM. The unit that plays the TPU TensorCore's role on a GPU is the SM itself.",
              ">> A TPU TensorCore contains an MXU. A GPU Tensor Core is one.",
              "Hold that straight and the rest of the vocabulary lines up cleanly: warp scheduler against VPU, CUDA core against VPU ALU, SMEM against VMEM, HBM against HBM. The next lesson puts the counts side by side, which is where the two designs stop looking alike.",
            ],
          },
        ],
        readings: [
          {
            label: "Scaling book · How to think about GPUs",
            url: "https://jax-ml.github.io/scaling-book/gpus/",
            note: "every number in this lesson comes from here; start at the SM section",
          },
          {
            label: "SemiAnalysis · the Tensor Core from Volta to Blackwell",
            url: "https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/",
            note: "how the matmul unit grew until it needed a memory of its own",
          },
          {
            label: "Cornell · understanding GPU architecture",
            url: "https://cvw.cac.cornell.edu/gpu-architecture",
            note: "the slower walk through SMs, warps, and occupancy",
          },
          {
            label: "How to optimize a CUDA matmul",
            url: "https://siboehm.com/articles/22/CUDA-MMM",
            note: "the same hierarchy seen from inside a kernel, step by step",
          },
        ],
          check: [
          {
            q: 'What is the ceiling on SM independence?',
            a: "The shared L2: all the SMs compete for the same lines, so one kernel's access pattern moves another kernel's throughput.",
          },
          {
            q: 'What stops you keeping 64 warps resident on an SM?',
            a: 'Registers: at 256 registers per thread only 8 warps fit the 256 kB register file, so occupancy falls as register pressure rises.',
          },
        ],
          work: [
            { id: 'sketch', label: 'sketch one SM from memory: four subpartitions and what each holds' },
            { id: 'occupancy', label: 'compute how many warps fit an SM at 256 registers per thread, then at 64' },
          ],
      },
      {
        id: "two-machines",
        num: 2,
        title: "Two machines, one job",
        lede: "Both chips exist to multiply matrices. One does it with 132 small units the hardware schedules, the other with two big ones the compiler schedules.",
        goal: "Given a piece of TPU kernel vocabulary, name its GPU counterpart and the count on each chip, and say what the difference costs the person writing the kernel.",
        sections: [
          {
            h: "the mapping, with counts",
            ps: [
              "Term for term, the two machines line up better than the marketing suggests. An SM plays the part of a TensorCore, a warp scheduler the part of the VPU, a CUDA core the part of a VPU ALU, SMEM the part of VMEM, and a GPU Tensor Core the part of the MXU. HBM is HBM on both. Once you can translate in both directions, most GPU documentation stops being foreign.",
              "The counts are where the family resemblance ends. A TPU v5p has 2 TensorCores with 8 MXUs between them; an H100 has 132 SMs carrying 528 Tensor Cores. Each TPU TensorCore has one big VPU built from 4 independently programmable `8x128` units, 4096 ALUs in total, while the H100 has 528 independent 32-wide SIMD units, about 16k ALUs. Counting individual lanes, an H100 has `132 * 4 * 32 = 16,896` CUDA cores against a v5p's `2 * 4 * 8 * 128 = 8192` ALUs, running at roughly the same frequency.",
              "The memory asymmetry runs the other way, and it is bigger than the table makes it look. An H100's on-chip fast memory is `32 MB` of SMEM against a TPU's `128 MB` of VMEM. Bandwidth widens the gap again: TPU VMEM runs at around `40 TB/s`, while the closest GPU equivalent by size, the L2, has been measured near `5.5 TB/s` and is not under your control. That is the single fact behind most claims that TPUs are better at inference, since weights that live in VMEM load fast enough to change the roofline.",
            ],
            table: {
              caption: "the scaling book's 1:1 comparison, plus the cache bandwidth line from the same chapter",
              cols: ["GPU", "TPU", "H100", "TPU v5p"],
              rows: [
                ["SM (streaming multiprocessor)", "TensorCore", "132", "2"],
                ["warp scheduler", "VPU slots", "528", "8"],
                ["Tensor Core", "MXU", "528", "8"],
                ["SMEM (L1)", "VMEM", "32 MB", "128 MB"],
                ["registers", "vector registers (VRegs)", "32 MB", "256 kB"],
                ["L2 cache", "VMEM", "~50 MB at ~5.5 TB/s", "128 MB at ~40 TB/s"],
              ],
            },
          },
          {
            h: "why the shapes differ",
            ps: [
              "Both chips spend a transistor budget; they spend it on different problems. The GPU buys many small independent units plus the hardware to decide, at runtime, which warp runs next on which subpartition. That machine is **hardware-scheduled**. The TPU buys a few very large units with a single thread of control and only VPU-wide vector instructions, which leaves the compiler to place every load and every matmul in a schedule ahead of time. That machine is **compiler-scheduled**.",
              ">> The GPU schedules at runtime. The TPU schedules at compile time.",
              "The TPU side is cheaper to build and simpler to reason about, and it moves the whole burden into the compiler, which must pipeline every memory load against MXU and VPU work or the machine stalls with nothing to hide the wait. The GPU side asks much less of its compiler. You can launch dozens of unrelated kernels and each one lands on an independent SM, and they will run.",
              "They may also run badly. Kernels that thrash the shared L2 or fail to coalesce their loads are slow for reasons the source does not show, and because the hardware owns so much of the runtime it is hard to see which reason applies. TPUs more often reach close to roofline with less work, precisely because less was left to runtime in the first place.",
            ],
          },
          {
            h: "what it costs the author",
            ps: [
              "On a GPU, the tuning surface is occupancy and locality. You watch registers per thread against the `64` resident warps an SM can hold, knowing that 256 registers per thread leaves room for only 8. You size shared-memory tiles against `256 kB` of SMEM. You order your accesses so that threads in a warp touch adjacent addresses, and you think about whether SMs are cooperating or fighting over the shared L2.",
              "On a TPU the surface is the schedule you write down. The `BlockSpec` index map decides which block each grid step sees, the grid order decides what streams in behind the current step, the `(8, 128)` lattice decides whether your shapes compile well at all, and the VMEM budget decides how large a block you may ask for. Stage 0 and stage 1 of the path are almost entirely this vocabulary.",
              "Underneath, it is the same question on both machines. An H100's bf16 matmul ridge is `9.9e14 / 3.4e12`, about `290 FLOPs per byte`, the same shape of number as the v5e's `240` from stage 0. What differs is who is responsible for staying above it.",
            ],
          },
          {
            h: "flash attention on both, shaped differently",
            ps: [
              "Attention is the clearest case of the same idea landing twice. The intermediate score matrix is the thing you cannot afford to write out to HBM and read back, on either machine, which is why the algorithm gets restructured to keep it on chip. The original paper frames the whole method as IO-awareness: tiling to reduce reads and writes between HBM and the on-chip SRAM, rather than reducing FLOPs.",
              "On a GPU that on-chip memory is one SM's SMEM, so the kernel is written as tiles staged into shared memory with threads inside a warp cooperating on each tile, and correctness of the online softmax expressed at thread granularity. On a TPU the same algebra becomes a grid over KV blocks: a `BlockSpec` stages each block into VMEM, the running maximum, running sum, and accumulator are carried across grid steps, and the pipeline hides the next block's transfer under the current block's math.",
              "The algebra is identical in both cases. What changes is who does the staging, which is the same split you have been reading about since the top of this lesson.",
            ],
          },
          {
            h: "back to the kernel",
            ps: [
              "You can now read a GPU spec sheet without translating twice: SM against TensorCore, Tensor Core against MXU, SMEM against VMEM, NVLink domain against torus axis, and a measured `370 GB/s` against a claimed `450`. That is enough hardware to hold both machines in your head while reading anyone's kernel.",
              "It is also where the hardware questions stop being the interesting ones. Everything from here is authorship, and the path picks it straight back up at Pallas fundamentals (`/s/pallas`), where the grid stops being a loop nest and becomes a pipeline you are responsible for keeping full.",
            ],
          },
        ],
        readings: [
          {
            label: "Scaling book · How to think about GPUs",
            url: "https://jax-ml.github.io/scaling-book/gpus/",
            note: "the chip-level comparison table this lesson is built on",
          },
          {
            label: "Scaling book · All about TPUs",
            url: "https://jax-ml.github.io/scaling-book/tpus/",
            note: "the other side of every row; the v5p figures live here",
          },
          {
            label: "FlashAttention: fast and memory-efficient exact attention with IO-awareness",
            url: "https://arxiv.org/abs/2205.14135",
            note: "the tiling argument both implementations descend from",
          },
          {
            label: "HuggingFace · the ultra-scale playbook",
            url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook",
            note: "the GPU side of all of this, at cluster scale and in practice",
          },
        ],
        check: [
          {
            q: 'What are the mappings for SMEM and for the GPU Tensor Core?',
            a: "SMEM plays VMEM's role, and a GPU Tensor Core maps to the MXU. The TPU TensorCore is the umbrella unit; its GPU counterpart is the SM.",
          },
          {
            q: 'Who schedules each machine?',
            a: 'The GPU schedules at runtime through its warp schedulers; the TPU is scheduled at compile time, with the compiler pipelining every load or the machine stalls.',
          },
        ],
        work: [
          { id: 'translate', label: 'translate one page of CUDA docs into TPU vocabulary with the table' },
          { id: 'ridge', label: 'derive the H100 ridge yourself and place three stage-0 ops against it' },
        ],
      },
    ],
  },
]
