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
        id: 'tensorcore-complex',
        num: 3,
        title: 'The TensorCore complex',
        lede: 'The v4 chip is the last TPU with a paper behind it, and that paper prints enough parts to rebuild the headline number from scratch.',
        goal: 'Given a generation’s published parts, derive its peak FLOPs and say which of three things happened: it closed, it closed only against a secondary clock, or it did not close at all.',
        sections: [
          {
            h: 'the last TPU with a paper behind it',
            ps: [
              'Google publishes two kinds of document about a TPU. Product pages give you a peak number, a memory size, and a link to pricing. ISCA papers give you the parts. The papers stop at v4, which is why v4 is the specimen worth opening: one paragraph of the 2023 paper names every compute block on the die, and Table 4 of the same paper prints the clock those blocks run at. After v4 there are product pages and launch posts, and neither kind of document lists parts.',
              '>> each TPU v4 contains two TensorCores (TC). Each TC contains four 128x128 Matrix Multiply Units (MXUs) and a Vector Processing Unit (VPU) with 128 lanes (16 ALUs per lane) and a 16 MiB Vector Memory (VMEM). The two TCs share a 128 MiB Common Memory (CMEM).',
              'Two things in that sentence never reach a product page. CMEM is `128 MiB` of memory the two TensorCores share, and no later generation is documented as keeping it or dropping it. The VPU is given as `128` lanes with `16` ALUs each, a count you should always carry its generation with: the scaling book describes a v5 VPU as a `(8, 128)` grid with `4` ALUs per lane-sublane pair, which is a different chip counted a different way. Neither source is wrong. Quoting one shape without its generation is.',
              'Two cores sharing one pool of memory raises a question the paper does not answer: does software see one accelerator or two? It never uses the word MegaCore for the arrangement either. Google Cloud’s docs and the scaling book do the answering, describing a chip whose two cores share memory and can be treated as one large accelerator. So the fusion is a software-visible arrangement resting on a hardware fact, and the hardware fact is CMEM. Which core `jax.devices()` hands you is the generations lesson at /l/tpu/generations.',
              'The chip lesson at /l/tpu/tpu-chip already said what these units do, and what VMEM is instead of a cache. This one counts them. Counting is what turns a spec sheet from a claim into arithmetic you can check.',
            ],
            table: {
              caption: 'TPU v4 as published · Jouppi et al., ISCA 2023, section 2 and Table 4 · the MAC line is the only derived row',
              cols: ['part', 'what the paper says', 'where'],
              rows: [
                ['TensorCores per chip', '2', 'section 2; Table 4 "Processors / Chip"'],
                ['MXUs per TensorCore', '4, each 128x128', 'section 2'],
                ['MACs per chip', '131,072 (2 x 4 x 128 x 128)', 'derived from section 2'],
                ['VPU per TensorCore', '128 lanes, 16 ALUs per lane', 'section 2'],
                ['VMEM', '16 MiB per TensorCore, 32 MiB per chip', 'section 2; Table 4'],
                ['CMEM', '128 MiB, shared by the two TensorCores', 'section 2; Table 4'],
                ['register file', '0.25 MiB', 'Table 4'],
                ['clock', '1050 MHz', 'Table 4'],
                ['node, die, transistors', '7 nm, under 600 mm2, 22 billion', 'Table 4'],
                ['HBM2', '32 GiB at 1200 GB/s', 'Table 4'],
              ],
            },
          },
          {
            h: 'multiply it out',
            ps: [
              'A peak FLOPs figure is three numbers multiplied together: how many multiply-accumulate cells the chip has, two operations per cell per cycle, and the clock. Start with v1, where all three sit in one paper. Its matrix unit is a single `256x256` grid, `65,536` cells, running at `700 MHz`. That gives `91.75` TOPS against a published `92`.',
              'Now v4, where the cells are spread across eight smaller arrays instead of one big one. Two TensorCores, four MXUs each, `128x128` per MXU: `131,072` cells, exactly twice what v1 carried in its single array. At `1050 MHz` that is `275.25` TFLOPS against a published `275`.',
              'Agreement at three digits is worth pausing on, because of what it tells you about the number being agreed with. There is no derating in it. Nothing accounts for the cycles the systolic array spends filling and draining, nothing accounts for a kernel that cannot keep the array fed. The vendor did the same multiplication you just did. A peak figure is geometry times a clock, and every real kernel lives underneath it.',
              'The same arithmetic survives one more zoom. `275` TFLOPS across a full 4096-chip v4 pod is `1.126` exaflops, and Google’s v4 page advertises `1.1 exaflops`. Peak numbers at pod scale are the chip number times the chip count, nothing else.',
            ],
            code: {
              caption: 'peak from parts · every input on these two derivations sits in one table of one paper',
              lang: 'text',
              text: "v1   256 x 256 MACs         =  65,536         ISCA 2017, section 2\n     x 2 ops per MAC        = 131,072         one multiply, one add\n     x 700 MHz             =  91.75 TOPS     ISCA 2017, Table 2\n     Google publishes         92 TOPS        ISCA 2017, Table 2\n\nv4   2 TC x 4 MXU x 128x128 = 131,072 MACs   ISCA 2023, section 2\n     x 2 ops per MAC        = 262,144\n     x 1050 MHz            = 275.25 TFLOPS   ISCA 2023, Table 4\n     Google publishes        275 TFLOPS      ISCA 2023, Table 4",
              full: {
                label: 'all four derivations, including the two that need a footnote and the one that fails',
                text: "v1   256 x 256 MACs         =  65,536         ISCA 2017, section 2\n     x 2 ops per MAC        = 131,072         one multiply, one add\n     x 700 MHz             =  91.75 TOPS     ISCA 2017, Table 2\n     Google publishes         92 TOPS        ISCA 2017, Table 2\n\nv4   2 TC x 4 MXU x 128x128 = 131,072 MACs   ISCA 2023, section 2\n     x 2 ops per MAC        = 262,144\n     x 1050 MHz            = 275.25 TFLOPS   ISCA 2023, Table 4\n     Google publishes        275 TFLOPS      ISCA 2023, Table 4\n\nv5p  2 TC x 4 MXU x 128x128 = 131,072 MACs   Google v5p page (2 TC);\n     x 2 ops per MAC        = 262,144        4 MXUs/TC read across from v4 and v5e\n     x about 1.75 GHz      = 458.8 TFLOPS    scaling book, SECONDARY, no vendor clock\n     Google publishes        459 TFLOPS      Google v5p page\n\nv6e  1 TC x 2 MXU x 256x256 = 131,072 MACs   Google v6e page + architecture page\n     x 2 ops per MAC        = 262,144\n     Google publishes        918 TFLOPS      Google v6e page\n     implied clock          = 918e12 / 262,144 = 3.50 GHz\n     no v6e clock is published, and 3.50 GHz is not a plausible TPU clock",
              },
            },
          },
          {
            h: 'the clock you cannot look up',
            ps: [
              'v5p is where the derivation starts needing a footnote. Google’s v5p page gives two TensorCores per chip and a peak of `459` TFLOPS bf16. It does not print an MXU count per TensorCore, though v4 and v5e are both documented at four, and the v5p page is written to the same shape. Take that layout and the MAC count comes out at `131,072`, exactly what v4 had. Same cells, `1.67x` the FLOPs. The extra has to be clock, and Google publishes no TPU clock after v4.',
              'One public figure fills the gap, and it is secondary: the scaling book states that a TPU v5p runs at about `1.75 GHz`. Run it forward. `262,144` FLOPs per clock times `1.75e9` is `458.8` TFLOPS against a published `459`. A secondary number that reproduces a vendor number to three digits has earned a place in the lesson, with its label still attached. If Google ever prints a v5p clock, that is the citation this paragraph should carry instead.',
              'Notice what the closing derivation buys you: it runs backwards. Whenever a vendor publishes a peak and enough geometry, you can solve for whatever constant they withheld, and then judge a third-party claim by whether it lands where the arithmetic says it must. What that clock did to the ratio between compute and bandwidth is the generations lesson at /l/tpu/generations, which owns the ridge.',
            ],
          },
          {
            h: 'v6e, where the arithmetic stops closing',
            ps: [
              'Trillium, the v6e generation, breaks the pattern. Google’s v6e page gives one TensorCore per chip with two matrix-multiply units on it, and a peak of `918` TFLOPS bf16. The architecture page gives the v6e MXU as `256x256`. Multiply those together and you get `131,072` MACs again, `262,144` FLOPs per clock, and an implied clock of `3.50 GHz`.',
              'That clock is not credible. The fastest TPU Google itself has printed is v4 at `1050 MHz`, and the fastest figure in the public record of any kind is the scaling book’s `1.75 GHz` for v5p. So at least one of the three inputs is wrong, and no published clock exists to arbitrate: the Trillium launch post says the team expanded the MXUs and raised the clock speed, and attaches no number to either.',
              'Google’s own architecture page also undercuts one of the inputs. It gives the MXU size for v6e and TPU7x as `256 x 256` multiply-accumulators, and elsewhere in the same passage it puts an MXU at 16K multiply-accumulate operations per cycle. `16K` is `16,384`, which is `128 x 128`. Whichever of those two describes v6e, they do not describe it together, and anyone quoting both in one breath is publishing a contradiction.',
              'You can make the numbers close by assuming four `256x256` MXUs instead of two, which puts the implied clock at `1.75 GHz` and matches v5p exactly. That is arithmetic, not evidence. Google’s page says two. So the honest result is an open derivation, written down as open: the published v6e MXU shape and the published `918` TFLOPS cannot both hold at any plausible clock, and the lesson stops there rather than inventing the number that would rescue it.',
            ],
          },
          {
            h: 'the count that means nothing on its own',
            ps: [
              'Every figure in this lesson is a count of cells, never a count of cores, and that is deliberate. Google’s TensorCore is a whole core: four MXUs, a vector unit, and a scalar unit under one instruction stream, two per chip on v4 and 8192 across a full 4096-chip pod. NVIDIA’s Tensor Core is a functional unit inside one of an SM’s four processing blocks, `528` of them on an H100 SXM5. The GPU chapter works that name collision through properly at /s/machine/gpu-chip; what it does to arithmetic is this lesson’s problem.',
              'A core count cannot be multiplied by anything. One chip has two of them and the other has `528`, and neither figure enters a derivation, because the two vendors are counting objects at different scales. MACs per clock per chip is the quantity both sides can produce, which is why every derivation above turns a TensorCore count into `131,072` cells in the same breath.',
              'The count also moves between generations for reasons that have little to do with capability. v4 and v5p carry two TensorCores, v5e and v6e carry one. v6e drops to a single core with two `256x256` MXUs, and on those published numbers the cell count lands exactly where it has sat since v4, at `131,072`. That is the same shape whose peak refuses to close, so hold it loosely; the point stands either way, which is that the core count is packaging and the cell count is the machine.',
            ],
          },
          {
            h: 'where the public record ends',
            ps: [
              'Past v4 the parts list thins out fast, and the gaps are worth knowing by name rather than discovering mid-argument. No clock is published for v5e, v5p, or v6e. VMEM is public for v3 and v4 and, secondhand through the scaling book, for v5e; for v5p and v6e it is not public at all. SMEM gets named by both Google and the scaling book and sized by neither, at any generation. Whether CMEM survives past v4 is not stated anywhere. Die size, transistor count, and process node stop at v4 with the ISCA papers.',
              'What to do with a gap is the part worth practising. The temptation is to divide two published numbers and print the quotient as a spec, which is exactly the move that yields a `3.50 GHz` TPU. Write the derivation instead, with its inputs and their sources, and label the result: closed against vendor numbers, closed only against a secondary source, or open. A lesson that says which figures are not public is more useful than one that quietly fills them in, because the reader can tell what to trust.',
            ],
            table: {
              caption: 'what Google publishes, by generation, as of 2026-08 · v4 from ISCA 2023 Table 4, v5p and v6e from the Cloud TPU pages · "about 1.75 GHz" is the scaling book, secondary',
              cols: ['figure', 'v4', 'v5p', 'v6e'],
              rows: [
                ['peak bf16', '275 TFLOPS', '459 TFLOPS', '918 TFLOPS'],
                ['clock', '1050 MHz', 'not published; about 1.75 GHz secondary', 'not published'],
                ['MXUs per TensorCore', '4, each 128x128', 'not printed; 4 by the page structure', '2, each 256x256'],
                ['VMEM', '16 MiB per TensorCore', 'not published', 'not published'],
                ['CMEM', '128 MiB shared', 'not stated', 'not stated'],
                ['SMEM', 'not published', 'not published', 'not published'],
                ['node, die, transistors', '7 nm, under 600 mm2, 22 billion', 'not published', 'not published'],
              ],
            },
          },
        ],
        readings: [
          { label: 'TPU v4, ISCA 2023', url: 'https://arxiv.org/abs/2304.01433', note: 'section 2 names every block on the die and Table 4 prints the clock; the last TPU documented this way' },
          { label: 'TPU v1, ISCA 2017', url: 'https://arxiv.org/abs/1704.04760', note: 'the 256x256 MAC array and the 700 MHz clock that make the first derivation close' },
          { label: 'Cloud TPU system architecture', url: 'https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm', note: 'the per-generation MXU sizes, and the passage that disagrees with itself about how many MACs an MXU has' },
          { label: 'Introducing Trillium, sixth-generation TPUs', url: 'https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus', note: 'says the clock went up and prints no number; the gap this lesson refuses to fill' },
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'secondary, and the only public source for a v5p clock; it reproduces the published 459 TFLOPS exactly' },
          { label: 'NVIDIA H100 architecture whitepaper', url: 'https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf', note: 'v1.02 mirror; p.18 and Figure 7 are where the other kind of tensor core is counted, four per SM' },
        ],
        check: [
          {
            q: 'A v4 chip and a v5p chip carry the same 131,072 MACs, yet Google publishes 275 TFLOPS for one and 459 for the other. Where does the difference come from, and what is the evidence?',
            a: 'The clock. v4 runs at 1050 MHz by ISCA Table 4; the scaling book puts v5p at about 1.75 GHz, and 262,144 FLOPs per clock times 1.75e9 gives 458.8 TFLOPS against the published 459. The clock figure is secondary, and it earns its place by reproducing the vendor number.',
          },
          {
            q: 'Why can no honest lesson print a clock speed for v6e?',
            a: 'Because the published parts do not close. One TensorCore with two 256x256 MXUs is 131,072 MACs, so 918 TFLOPS needs 3.50 GHz, which no TPU approaches. Google publishes no v6e clock, and its own architecture page gives an MXU MAC count that fits 128x128 rather than 256x256, so an input is wrong and nothing public says which.',
          },
          {
            q: 'The v4 paper prints 16 MiB of VMEM per TensorCore. What is the v5p figure, and how should a lesson write it?',
            a: 'There is none to print. VMEM is published for v3 and v4 and reaches v5e only through the scaling book, which is secondary; for v5p and v6e it is not public at all. Write it as not published, with the generations that are documented named beside it, rather than scaling a v4 number forward.',
          },
        ],
        work: [
          { id: 'derive', label: 'derive v1, v4, and v5p peak from parts with the code block folded shut', href: '#multiply-it-out' },
          { id: 'open', label: 'write the v6e derivation in three lines and name the one input you would ask Google for', href: '#v6e-where-the-arithmetic-stops-closing' },
          { id: 'check', label: 'answer the three checks without opening them', href: '#check' },
        ],
      },
      // NOTE for the integrator: `num` must equal the 1-based index of this
      // object inside the l:tpu lessons array. 4 assumes the block order
      // tpu-chip, generations, tensorcore-complex, sparsecore. See §3.
      {
        id: 'sparsecore',
        num: 4,
        title: 'SparseCore',
        lede: 'Every TPU since v2 carries a third kind of core that no kernel in this course touches. It exists because one workload class was worth 5% of the die rather than a rewrite of the array.',
        goal: 'Say why an embedding lookup cannot be made to fit a systolic array, name the units inside one v4 SparseCore, and state what the paper paid in die area and what it bought.',
        sections: [
          {
            h: 'the lookup that has no matmul in it',
            ps: [
              "A ranking model's first layer is a table read. The ISCA paper's example is a table with 80,000 rows, one per word in the English language, each row 100 numbers wide; one training example looks up a single row, or a small and dynamic number of rows that are then summed. Nothing in that is a matrix multiply. It is a gather on the way forward and a scatter on the way back, over tables the paper sizes anywhere from O(10 MiB) to O(100 GiB), with all the tables of one model reaching several TiB together.",
              'The TPU chip lesson gave the reason a systolic array punishes small work: fill and drain dominate whenever the operand does not fill the grid. Lookups are not merely small, they are shaped wrong. The paper describes them as small gather or scatter memory accesses with low arithmetic intensity, so what decides their end-to-end speed is memory bandwidth, memory capacity and vector throughput rather than chip FLOPS per second. Hand that to an MXU and most of the grid holds still while the memory system does all the work.',
              "The tables are also too large for one chip, so they get cut up three ways the paper names: column sharding along the width, row sharding along the vocabulary, or whole tables placed on different chips. Traffic follows from the cut. Under model parallelism the pattern is a variable-length all-to-all whose ceiling is bisection bandwidth, and the fabric lesson in the ICI unit carries that arithmetic. The sparsity is unstructured on top of that, since a few feature values are far hotter than the rest, so compute, memory and network all skew together, and deduplicating the frequent values before the lookup is part of what the hardware has to make cheap.",
            ],
          },
          {
            h: 'the two places it could have gone',
            ps: [
              'Before there was a third core, the work had two possible homes, and the paper argues against both in a paragraph each. The TensorCore is the obvious candidate and the wrong one: wide VPU, large matrix units, tuned end to end for dense operations, asked here to perform small gathers and scatters and to exchange variable-length data with other chips. It would run the work. The paper calls placing embeddings there suboptimal, and names those two access patterns as the reason.',
              'The host CPUs are the other candidate. Push every lookup out to host memory and the CPU DRAM interface becomes the Amdahl bottleneck, amplified on v4 by four chips sharing one CPU host, with tail latency and data-center network bandwidth constraining whatever survives. This is not a thought experiment in the paper. They built the configuration and measured it, and that measurement is where the widely quoted 5x to 7x comes from.',
              "What the codesign chose instead was the supercomputer's own memory. Every chip's HBM joins one flat, globally addressable space, 128 TiB of it on a v4 pod, reached over a dedicated ICI network with gather and scatter support in hardware. The paper's term for the arrangement is a sea of cores. Because the SparseCores are separate cores rather than a mode of an existing one, dense compute, embedding work and ICI traffic proceed at the same time instead of taking turns.",
            ],
          },
          {
            h: 'sixteen tiles and five cross-channel units',
            ps: [
              'Open one v4 SparseCore and nothing inside resembles a grid of multipliers. The paper calls it a dataflow architecture, meaning data flows out of memory into a set of directly connected specialized units, and a v4 chip carries four of these next to its two TensorCores.',
              "Sixteen compute tiles do the general work. Each tile owns an HBM channel and keeps many memory accesses outstanding at once, which is the design answering the access pattern directly: thousands of independent small reads, none of them waiting on the answer to the last. Inside a tile sit three units in a row. A Fetch Unit reads activations and parameters out of HBM. A programmable 8-wide SIMD vector unit, the scVPU, does the arithmetic on them, reusing the same ALUs as the TensorCore's VPU. A Flush Unit writes updated parameters back to HBM during the backward pass.",
              'The tiles work out of Spmem, the sparse vector memory: 2.5 MiB per SparseCore, with each tile reading and writing its own slice of it. Multiply that out and the per-chip totals in the paper table fall out exactly, four SparseCores on v4 for 10 MiB of spMEM and two on v3 for 5 MiB. It is a small memory next to the tens of MiB the TensorCore side works from, and it is sized for vectors a hundred numbers wide rather than for the tiles of a matmul.',
              'Five cross-channel units sit beside the tiles and perform the embedding operations no single tile can, because they operate across all sixteen banks of Spmem collectively. They take CISC-like instructions over variable-length inputs, so how long an instruction runs depends on the data handed to it. That is the opposite property from the array next door, where a matmul costs a cycle count fixed by its shape before it begins.',
            ],
            diagram: 'sparsecore',
          },
          {
            h: 'five percent of the die',
            ps: [
              'The abstract prices the whole thing in one line, and section 3.5 repeats it with a tilde on each figure: about 5% of the die area and about 5% of the power.',
              '>> SparseCores, dataflow processors that accelerate models that rely on embeddings by 5x-7x yet use only 5% of die area and power.',
              'The 5x to 7x is a subtraction rather than a headline. The paper takes TPU v4, changes nothing else, moves the embeddings into CPU host memory, and measures what the production model DLRM0 loses at 128 chips: a factor of 5 to 7, bottlenecked on CPU memory bandwidth. Read it as the price of not having the core. An area-and-power claim can only be honest if the comparison names what was removed.',
              'The neighbouring numbers set the scale. On the same model against 576 Skylake sockets, TPU v3 is 9.8x faster; v4 beats v3 by 3.1x and the CPU configuration by 30.1x. Moving from the 2D torus of v3 to the 3D torus of v4 raises bisection bandwidth 2x to 4x at a given chip count and accelerates embeddings by 1.1x to 2.0x, which is the fabric doing embedding work rather than the core. Past about 1024 chips the paper reports SparseCore overheads starting to dominate, at which point bisection bandwidth matters less.',
              'From the pod, the SparseCores look like a single machine. The Hot Chips deck describes non-coherent shared memory spanning the pod and millions of outstanding references reaching any node in it, hidden behind multithreading, with a full v4 pod presenting 8K TensorCores and 16K SparseCores across the optical switches.',
            ],
          },
          {
            h: 'counting them, generation by generation',
            ps: [
              'SparseCore has been part of TPUs since v2, and the per-chip count has moved twice in the public record. The ISCA paper gives 2 on v3 and 4 on v4. The Cloud TPU architecture page gives 4 on v5p and on TPU7x, and 2 on v6e. The v5e page lists none at all, which is a gap in the documentation and not a documented zero: write unknown, not absent.',
              'The generation labels deserve more care than the counts. The Hot Chips 2023 deck lists a third-generation embeddings coprocessor among the innovations of the fourth-generation system, meaning v4. The 2024 Trillium blog says v6e is equipped with third-generation SparseCore. Two chips, two generations apart, both called third. Nothing published reconciles the two, so carry the source with the label every time and never quote the number on its own.',
            ],
            table: {
              caption: 'per-chip SparseCore counts as published (retrieved 2026-08-14) · not published means the vendor page or paper does not state it, which is not the same as zero',
              cols: ['generation', 'SparseCores / chip', 'spMEM / chip', 'where it is published'],
              rows: [
                ['v2', 'present, count not published', 'not published', 'ISCA 2023, section 1'],
                ['v3', '2', '5 MiB', 'ISCA 2023, Table 4'],
                ['v4', '4', '10 MiB', 'ISCA 2023, Table 4'],
                ['v5e', 'not listed', 'not published', 'Cloud TPU v5e page'],
                ['v5p', '4', 'not published', 'Cloud TPU architecture page'],
                ['v6e', '2', 'not published', 'Cloud TPU architecture page'],
                ['7x', '4', 'not published', 'Cloud TPU architecture page'],
              ],
            },
          },
          {
            h: 'a sibling core, not a bigger core',
            ps: [
              "One piece of hygiene before the principle, because two vendors put sparse in the name of very different things. NVIDIA's structured sparsity is a Tensor Core mode: prune the weights until at least two values in every group of four contiguous ones are zero, and the unit processes only the nonzeros for roughly twice the dense rate. What that does to a datasheet is read out in the machine stage, at `/s/machine/inside-the-sm`. SparseCore is not a mode of anything. It is a separate programmable processor with its own memory, its own instructions and its own HBM channels, and the sparsity it serves is the unstructured kind that arrives from a lookup rather than from pruning.",
              'The general move is worth taking away from the TPU entirely. When a workload class is large enough to matter and its access pattern fights the execution model of the main core, the answer that keeps winning is a sibling core beside it, not a wider or more general version of it. Widening the array would have cost area on every workload and still left gather latency where it was.',
              "The condition on that move is the arithmetic in this lesson. DLRMs are about a quarter of Google's ML workload by the paper's own table, the core costs around 5% of area and power, and its absence costs 5x to 7x on that quarter. Change any of the three and the answer changes: a workload at one percent of the fleet does not earn silicon, and a workload that tiles into the array does not need it. The TPU chip lesson put SparseCore on the die in a single sentence and moved on, which was the right depth for a first pass; this is what that sentence was standing on.",
            ],
          },
        ],
        readings: [
          {
            label: 'Jouppi et al. · TPU v4, with hardware support for embeddings (ISCA 2023)',
            url: 'https://arxiv.org/abs/2304.01433',
            note: 'section 3 is the whole SparseCore argument, and Figure 7 is the block diagram this lesson redraws',
          },
          {
            label: 'Jouppi and Swing · the same machine at Hot Chips 2023',
            url: 'https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf',
            note: 'the pod-level view: non-coherent shared memory, millions of outstanding references, 16K SparseCores',
          },
          {
            label: 'Cloud TPU system architecture',
            url: 'https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm',
            note: 'the per-generation counts, and the page that quietly stops listing them for v5e',
          },
          {
            label: 'Google Cloud · introducing Trillium',
            url: 'https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus',
            note: 'the third-generation label for v6e that collides with the deck above',
          },
          {
            label: 'NVIDIA · structured sparsity in the Ampere architecture',
            url: 'https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/',
            note: 'the other thing called sparse: two zeros in every four weights, and a Tensor Core mode rather than a core',
          },
        ],
        check: [
          {
            q: 'An embedding lookup and a large matmul both stream bytes out of HBM. Why does only one of them suit the MXU?',
            a: 'The lookup is small gathers and scatters with low arithmetic intensity, and the number of rows varies per example, so bandwidth, capacity and vector throughput decide its speed while most of a systolic array would stand idle.',
          },
          {
            q: 'What exactly was measured to produce the 5x to 7x figure, and why does the condition matter?',
            a: 'TPU v4 with embeddings moved into CPU host memory instead of onto SparseCore, on DLRM0 at 128 chips, bottlenecked on CPU memory bandwidth. It prices the absence of the core on one workload rather than a general speedup, so quoting it bare overstates it.',
          },
          {
            q: 'Google calls two different chips third-generation SparseCore. What goes in your notes?',
            a: 'Both labels with their sources: the Hot Chips 2023 deck for v4, the Trillium blog for v6e. Nothing public reconciles them, so a generation number only means something with its source attached.',
          },
        ],
        work: [
          { id: 'tile', label: 'redraw one SparseCore from memory: sixteen tiles, three units each, five cross-channel units', href: '#sixteen-tiles-and-five-cross-channel-units' },
          { id: 'spmem', label: 'check the Spmem arithmetic against the ISCA table: 2.5 MiB per core against the v3 and v4 chip totals' },
          { id: 'counts', label: 'fill the generation table from the vendor pages yourself and mark every cell the record does not state', href: '#counting-them-generation-by-generation' },
          { id: 'check', label: 'answer the three checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'two-ways-to-hide-latency',
        num: 5,
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
        num: 6,
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
        num: 7,
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
        num: 8,
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
      {
        id: 'vliw-bundles-and-llo',
        num: 9,
        title: 'The bundle, and the floor below it',
        lede: 'One TPU instruction is 322 bits wide and speaks to several units at once. Filling those slots is the compiler’s whole job, and the layer where it happens is the one you cannot read.',
        goal: 'Say what a single TPU instruction actually is, name what the compiler decides that no hardware will fix at runtime, and state exactly which parts of the layer below Mosaic are public and which are not.',
        sections: [
          {
            h: 'what counts as one instruction',
            ps: [
              "A TPUv2 core fetches one 322-bit instruction at a time, and that instruction speaks to several units at once. Two of its slots carry scalar work. Four carry vector work, two of those reserved for vector load and store. Two more carry matrix work, one push and one pop. Then a miscellaneous slot, and six immediates. Read the layout as a budget rather than a format: those slots are what a compiler has to fill, per cycle, for the machine to be doing anything at all.",
              "A machine whose one instruction word issues to several units at once, with the parallelism settled before the program runs, is a **VLIW** machine, for very long instruction word. The design paper says why the TPU is one. A VLIW architecture was the simplest way for the hardware to express instruction level parallelism, and it let the team use compiler techniques that already existed.",
              "The scalar unit is the front of this machine. It pulls whole bundles from a local instruction memory, runs the scalar slots itself, and forwards the decoded rest onward to the vector and matrix units, where execution happens later and decoupled from the scalar side. The chip lesson at /l/tpu/tpu-chip introduced that unit from the data side, as the thing that decides which block comes next and fetches it. This is the same unit from the instruction side, and it is one job, not two.",
              "One vector slot is also not one number. The vector unit is 128 lanes wide and each lane carries an eight-deep sublane dimension, so a vector slot operates on eight sets of 128-wide values in a cycle. The `(8, 128)` shape the course keeps meeting as a tiling rule is that geometry, read from the instruction that drives it.",
            ],
            code: {
              caption: 'the bundle, verbatim from the TPUv2 and TPUv3 design paper (Norrie et al., IEEE Micro 41(2), 2021); line breaks are this panel’s',
              lang: 'text',
              text: 'The scalar unit is where computation originates. It fetches complete VLIW\nbundles from a local instruction memory, executes the scalar operations slots\nlocally, and then forwards decoded instructions on to the vector and matrix\nunits where execution happens later, decoupled from scalar execution. The\nVLIW bundle is 322 bits and is composed of two scalar slots, four vector\nslots (two used for vector load/store), two matrix slots (a push and a pop),\none miscellaneous slot (a simple example would be a delay instruction), and\nsix immediates.',
            },
          },
          {
            h: 'the instruction stream is staged, like everything else',
            ps: [
              "Those bundles have to reach the core somehow, and the answer is the one this chip gives for everything else. There is no instruction cache backed by HBM. There is a local instruction memory that DMA writes into, and the paper is candid about the decision: a cache would have been nice, and a DMA target for software-managed instruction overlays was easier to build. Code gets staged the way data gets staged, by a transfer that something scheduled.",
              "That symmetry closes a gap in the picture the unit has been building. VMEM holds what software put there. SMEM holds what software put there. So does the memory the program itself lives in. Nothing on this chip fetches ahead on your behalf, and the part that fetches your instructions is not an exception to its own design.",
              "Put a rate on the stream and the tradeoff gets concrete. One 322-bit bundle per cycle at v3’s 940 MHz is `3.78e10` bytes per second of instruction fetch, about 4 percent of the `9.0e11` bytes per second v3’s HBM delivers. Two assumptions sit under that arithmetic and neither is small. It assumes a bundle issues every cycle, which is the ceiling and not the average. And 322 bits is the published width for v2 and v3 alone, which the next two sections are about.",
            ],
          },
          {
            h: 'nothing reorders, so the schedule is the program',
            ps: [
              "The shorthand for a VLIW machine is that it has no interlocks, and the TPU papers do not say that. What the design paper describes is hold conditions on instructions for execution interlock, plus synchronization flags for interlocking against software-managed DMA. Stalls exist on this chip. A wait is a real event with real cycles in it.",
              "What is missing is the other half of a modern CPU. Nothing here picks a different instruction to run while one waits. A hold stalls the pipeline and does not fill it, because there is no pool of ready work to draw from and no unit whose job is to choose. The latency lesson at /l/tpu/two-ways-to-hide-latency drew that contrast against a GPU’s warp scheduler; this is the same fact one level lower, in the instruction word itself.",
              "One piece of hardware exists to buy the compiler slack, and it shows how tight the schedule is otherwise. Matrix results land in a Result FIFO, which the paper says lets them avoid strict execution schedule constraints for the long-latency matrix operations and shorten register lifetimes, simplifying the compiler. That is why push and pop are two separate slots. The two ends of a matrix operation get scheduled independently, and the FIFO absorbs the distance between them.",
              "Static scheduling puts the performance in the compiler, and the TPU team has published both the risk and the return. Compiler problems likely sank the Itanium’s VLIW architecture, the TPUv4i paper says, and it names that before observing that many domain-specific accelerators rely on VLIW anyway. Then it gives the return. Over the twenty months from MLPerf Training 0.5 to 0.7, CUDA compilation improved the GPU by 1.8x and XLA raised the TPU by 2.2x, against the 1 to 2 percent a year that C compilers move general-purpose code.",
              '>> Twenty months of compiler work moved the same silicon 2.2x.',
              "You have already scrolled past this compiler estimating a schedule. Every TPU fusion in this repo’s capture carries a `window_config` inside its `backend_config`, and two of its fields are the compiler talking about cycles and bundles for that window. The cycle number is populated here and the bundle count is not, which is worth taking as the shape of the boundary rather than as a shortage. The plan has a place to say how many bundles it expects. What the dump hands you is the cycle estimate alone.",
            ],
            code: {
              caption: 'the compiler’s own schedule estimate for the attention fusion, from site/src/data/hlo-pairs.json (TPU v6 lite, jax 0.11.0); fields trimmed at the ellipsis, values verbatim',
              lang: 'text',
              text: '%fusion = bf16[1024,1024]{1,0:T(8,128)(2,1)S(1)} fusion(%copy-done, %k.1), kind=kOutput, ...\n  backend_config={... "window_config":{"buffering_level":"2",\n    "cost_model_type":"COST_MODEL_TYPE_CLASSIC",\n    "estimated_bundle_count":"0","estimated_cycles":"3528", ...}}',
            },
          },
          {
            h: 'compiler compatible, not binary compatible',
            ps: [
              "A 322-bit word with a fixed slot layout sounds like an instruction set you could target, and Google decided it would not be one. TPUv2 and TPUv3 share the bundle length. TPUv4i broke with it on purpose, and the paper spends a section saying why rather than leaving it to be discovered.",
              "The argument starts from what VLIW was for. Putting the parallelism in the instruction word is what lets a recompile use a wider machine, and binary compatibility freezes exactly that. The paper adds an admission worth reading twice: many engineers built Itanium compilers, some of them now on the XLA team, where they learned the drawbacks of binary compatibility for a VLIW compiler and its hardware.",
              "So the stable thing is a split inside the compiler. XLA divides the compiling task into High-Level Operations that are machine independent and Low-Level Operations that are machine dependent, and the second name is **LLO**. Optimizations at the HLO level apply to every platform. A new TPU that confines its compiler changes to the LLO half, a wider VLIW for instance, keeps compiler compatibility while breaking every binary. The contract with the outside world is HLO, and the bundle sits on the far side of it.",
              "The GPU stack draws its line one level lower, which is the comparison to hold. PTX is a published virtual instruction set with a documented stability story, and SASS below it is per-generation and undocumented, so a GPU author gets one readable layer past the compiler’s portable output. There is no PTX on the TPU side. The last level anyone outside can print is the one the Mosaic lesson at /l/mosaic/to-machine-code already draws the boundary at, and below it the encoding is per-generation because being per-generation is the design.",
              '>> 322 bits is published for v2 and v3, and for nothing since.',
            ],
          },
          {
            h: 'where the public record ends',
            ps: [
              "Start with what you can see, because it is more than the usual telling admits. XProf has a public LLO surface, documented in the open-source repo rather than inferred. The current instructions set two XLA flags together, `xla_xprof_enable_custom_call_tracing` and `xla_xprof_register_llo_debug_info`, and what the doc shows those producing is LLO traces in the trace viewer, at the level of ops and of individual instructions. A Pallas kernel arrives as a custom call, so the surface is pointed at exactly the kernels this course writes.",
              "An older recipe sits further down the same file under a heading that marks it old, and it is the only place the utilization view is documented. There, `xla_enable_custom_call_region_trace` pairs with the same debug-info flag, and the doc says an LLO utilization line then appears in the trace viewer for each TPU core or device executing the custom call. Nothing in the file attributes that line to the newer tracing flag. If you want the utilization view, the old flag is the one to set.",
              "Back on the current recipe, the parameters that tune trace insertion are where bundles stop being a paper fact. Trace insertion is configured in units of bundles. `trace_best_effort_frequency` sets the target interval for opportunistic traces packed into existing bundles, and `trace_guaranteed_frequency` sets the maximum number of bundles allowed between two traces. The doc is exact about the difference: the best-effort pass will not create new bundles, and when the guarantee cannot be met by packing, the compiler creates a new bundle and places a trace there by itself. A profiler that can slip its own instruction into a bundle’s spare slot is describing a real slot budget.",
              "The same document publishes the per-instruction cycle costs the compiler models, which is the nearest thing to an LLO timing manual in public. It names two units this course has not: XLU for transposes, and EUP, which the doc glosses as vector math functions like tanh, log and exp. Read the table as the model and not the measurement, in the doc’s own terms. The compiler calculates an intrinsic cycle cost per LLO instruction from the target generation and the execution unit resolving it, and XProf interpolates between its trace points with those numbers.",
              "The 2026 XProf work goes a step further and says so in the same vocabulary. LLO bundle data is now exposed for Pallas authors, described there as the specific machine instructions issued to the TPU’s functional units during every clock cycle, with traces inserted by dynamic instrumentation so the times are exact rather than static compiler estimates. The worked example is idle cycles inside the MXU pipeline, latency between a `vmatmul` and a `vpop`. Two instruction names in public, doing precisely what the push and pop matrix slots of a 2021 paper describe.",
              "Now the other half, stated as plainly as the XLA course states its own limits at /xla/pathways. The instruction set is not published. The encoding is not published past v3’s width. The scheduler that fills the slots ships inside libtpu and no flag prints it. Even the name is unsettled: the scaling book and the XProf docs expand LLO as low-level optimizer, while the TPUv4i paper and Google’s own 2026 announcement expand it as Low-Level Operations. Four public sources, two expansions, no correction anywhere. Treat everything here about the bundle’s internals as coming from papers about v2, v3 and v4i, and everything about the tooling as documentation for chips those papers never described.",
            ],
            table: {
              caption: 'modeled per-instruction cycle costs, from openxla/xprof docs/custom_call_profiling.md at commit d9a61f5 (2026-08-14); the compiler’s estimates XProf interpolates with, not measurements',
              cols: ['unit', 'instruction', 'v5e / v5p', 'v6e / v7x'],
              rows: [
                ['MXU', 'vector matmul, f32', '8', '8'],
                ['MXU', 'vector matmul, packed bf16', '2', '2'],
                ['MXU', 'vector matmul, integer (u8, s8, u4, s4)', '1', '1'],
                ['XLU', 'packed transpose', '17', '4'],
                ['XLU', 'b16 transpose', '17', '4'],
                ['EUP', 'vector math (tanh, exp, log)', '2', '1'],
              ],
            },
          },
        ],
        readings: [
          {
            label: 'The design process for Google’s training chips: TPUv2 and TPUv3',
            url: 'https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf',
            note: 'the bundle, the scalar unit, and the Result FIFO, in the architects’ own words',
          },
          {
            label: 'Ten lessons from three generations shaped Google’s TPUv4i',
            url: 'https://www.cs.cmu.edu/~18742/papers/Jouppi2021.pdf',
            note: 'why compiler compatibility beat binary compatibility, and what XLA was worth in MLPerf',
          },
          {
            label: 'XProf · custom call profiling',
            url: 'https://github.com/openxla/xprof/blob/master/docs/custom_call_profiling.md',
            note: 'the flags, the bundle-level trace knobs, and the modeled cycle tables',
          },
          {
            label: 'Advanced TPU optimization with XProf: LLO bundles',
            url: 'https://opensource.googleblog.com/2026/03/advanced-tpu-optimization-with-xprof-continuous-profiling-utilization-insights-and-llo-bundles.html',
            note: 'the 2026 announcement that put per-cycle bundle data in front of kernel authors',
          },
        ],
        check: [
          {
            q: 'The shorthand says a VLIW machine has no interlocks. What do the TPU papers actually describe, and what is genuinely absent?',
            a: 'Hold conditions interlock execution and synchronization flags interlock against software-managed DMA, so stalls are real. What is absent is reordering: nothing picks another instruction while one waits, so a gap the compiler left is a gap the machine takes.',
          },
          {
            q: 'TPUv2 and TPUv3 share a 322-bit bundle. Why did TPUv4i decline to stay binary compatible with it?',
            a: 'The point of VLIW is that a recompile lets the compiler use new hardware resources, which binary compatibility freezes. XLA already splits machine-independent HLO from machine-dependent LLO, so a wider VLIW changes only the LLO half and compiler compatibility survives instead.',
          },
          {
            q: 'What can someone outside Google actually see of LLO, and what stays closed?',
            a: 'The XProf surface: LLO debug info behind an XLA flag, an LLO utilization line, bundle-level traces, and a published table of modeled per-instruction cycle costs. The instruction set, the encoding past v3, and the scheduler are closed, and the acronym itself is expanded two different ways across public sources.',
          },
        ],
        work: [
          { id: 'slots', label: 'account for all 322 bits: name every slot and the unit it drives', href: '#what-counts-as-one-instruction' },
          { id: 'fetch', label: 'redo the instruction-fetch rate for a generation of your choice and state every assumption you had to make', href: '#the-instruction-stream-is-staged-like-everything-else' },
          { id: 'boundary', label: 'write the LLO boundary in three sentences: what is public, what is inferred, what is closed', href: '#where-the-public-record-ends' },
          { id: 'check', label: 'answer the three checks without opening them', href: '#check' },
        ],
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
        id: "die-and-reticle",
        num: 1,
        title: "The die and the reticle",
        lede: "Every headline number on a chip starts inside one rectangle: the largest area a lithography scanner can expose in a single shot, 26 by 33 millimetres.",
        goal: "Given a chip's die area, transistor count and HBM figures, work out what the reticle limit and the yield curve allowed the designer to build, and name which of those numbers the vendor never published.",
        sections: [
          {
            h: "one exposure, 26 by 33",
            ps: [
              "A scanner prints a chip by shining light through a mask onto the wafer, one rectangle at a time, and that rectangle has a fixed size. ASML publishes it per machine rather than as an industry constant: the TWINSCAN NXT:2050i lists a `full 26 x 33 mm field size` with `4X reduction`, so the pattern drawn on the six-inch mask blank is demagnified four times on its way down to the resist. Multiply the field out and you get `858 mm2`. That is the ceiling every chip designer works under, and NVIDIA's own phrase for Blackwell, `the largest die possible within the limits of reticle size`, is a designer telling you they hit it.",
              "Put those fields on a wafer and the second constraint appears. A 300 mm wafer has `706 cm2` of area, so at most `82` full-field dies fit on it before you account for the round edge, and a die half that size gets you at most `164`. The edge is not a rounding detail. Leachman's yield note lists edge loss as a mechanism of its own, films deposited poorly near the rim causing `wholesale die yield losses` that have nothing to do with a particle landing on the die.",
              "Lithography is not about to relax the limit either. High-NA EUV, the generation after the current one, uses anamorphic optics and exposes a half field of `26 x 16.5 mm`, so a design that wants the full width has to be printed as two stitched exposures. Semiconductor Engineering's survey of the question is the readable account. The size limit that shapes the largest AI chips gets tighter before it gets looser.",
            ],
          },
          {
            h: "yield falls faster than area grows",
            ps: [
              "Defects land on a wafer roughly at random, and one landing in the wrong place kills the die under it. Write `D0` for defects per square centimetre and `A` for die area, and the simplest model of the survivors is Poisson, `DY = exp(-D0 * A)`. Double the area and you square the survival probability. That single fact is why a reticle-sized die costs much more than twice a half-sized one, and it is the pressure behind every packaging trick in this lesson.",
              "Poisson is also the pessimistic model. Leachman is explicit that it `tends to underestimate die yield when the expected number of defects per chip is greater than one or when the die area is relatively large`, because real defects cluster, and a wafer that dumps six particles onto one die leaves its neighbours clean. Murphy's fix at Bell Labs was to let `D0` itself be a distribution and integrate over it. A triangular distribution gives the model that carries his name, `DY = ((1 - exp(-A*D0)) / (A*D0))^2`. An exponential one gives Seeds, `DY = 1 / (1 + A*D0)`.",
              "One model absorbs the others. Assume a Gamma distribution and the integral collapses to the negative binomial, `DY = (1 + A*D0/alpha)^(-alpha)`, where `alpha` is a cluster parameter you estimate from defect data as mean squared over variance. Leachman gives the correspondence outright: at `alpha >= 10` it is essentially Poisson, at `alpha = 5` it closely approximates Murphy, at `alpha = 1` it closely approximates Seeds. Choosing `alpha` is choosing how clustered you believe the defects are, and nothing more.",
              "Foundries do not publish defect density for a leading node, so the `D0 = 0.1` below is a teaching value you can vary, not a TSMC figure. The shape of the answer survives whatever you set it to.",
            ],
            code: {
              caption: "the three models at one illustrative defect density; formulas from Leachman, IEOR 130, equations 2, 9, 10 and 12",
              lang: "python",
              text: `from math import exp

D0 = 0.10              # defects/cm2: a teaching value, not a TSMC figure
FULL = 26 * 33 / 100   # 8.58 cm2, one full reticle field
HALF = FULL / 2

poisson = lambda a: exp(-D0 * a)
murphy = lambda a: ((1 - exp(-D0 * a)) / (D0 * a)) ** 2
negbin = lambda a, k: (1 + D0 * a / k) ** -k

for name, f in [("poisson", poisson), ("murphy", murphy),
                ("negbin k=5", lambda a: negbin(a, 5))]:
    print(f"{name:11} full {f(FULL):.3f}  half {f(HALF):.3f}")

# dual-die parts per cm2 of wafer: harvested halves against whole dies
pairs = poisson(HALF) / HALF / 2
whole = poisson(FULL) / FULL
print(f"pairs / whole = {pairs / whole:.2f}")

# poisson     full 0.424  half 0.651
# murphy      full 0.451  half 0.661
# negbin k=5  full 0.453  half 0.663
# pairs / whole = 1.54`,
            },
          },
          {
            h: "cut it in half and harvest the wafer",
            ps: [
              "The yield argument for splitting a die is easy to state wrongly. Under Poisson, the probability that one specific pair of half-dies are both good is exactly the yield of the undivided die, because the exponents add and nothing has changed. You are never required to use a specific pair. You test every half on the wafer, discard the dead ones, and assemble parts from any two survivors.",
              "So count parts per unit of wafer rather than per die site. At `D0 = 0.1` a full-field die yields `0.42` and a half-field die yields `0.65`, and pairing harvested halves gives `1.54x` as many dual-die parts out of the same silicon area. Murphy, being kinder to large dies, puts the same ratio at `1.47x`. The gain is real and it is not enormous, which is why yield on its own rarely decides the question.",
              "Harvesting also happens inside a single die, and that version ships in the highest volume of all. Draw the design with more SMs, more memory controllers and more L2 than any part will be sold with, test what comes off the wafer, then sort each die by how much of it works. A die with a few dead SMs is not scrap, it is a cheaper part. NVIDIA sells no full GH100 at all, and both H100 SKUs run with SMs switched off. The SM lesson prints the whitepaper's three configurations of that one design side by side, and works from them. Designing in redundancy and then selling by what survived is how a reticle-sized die stays economically viable, which means the SM count printed on a datasheet is a yield outcome as much as a design decision.",
              "Two other reasons carry more weight. One is the wall from the first section: once a design wants more transistors than `858 mm2` can hold, there is no single-die version left to compare against, so no yield argument is needed. The other is that different circuits want different processes. TSMC advertises CoWoS as designed `for both heterogeneous and homogeneous integration`, which is the packaging way of saying the logic can sit on a leading node while I/O and analog blocks stay on an older, cheaper one.",
              "Every cut has a price, and it is paid in the join. A wire that used to run inside a die now crosses a package boundary, costing energy per bit and latency, and it needs a physical interface at each end that occupies area on both dies. That is why the number worth looking up on any multi-die part is the bandwidth of the join, and why the Blackwell worked example below starts there.",
            ],
          },
          {
            h: "the interposer and the stacks",
            ps: [
              "The join happens on a slab of silicon underneath. TSMC describes CoWoS as `TSMC's proprietary chip-last on interposer process`, and ships it as three families, CoWoS-S, CoWoS-L and CoWoS-R. Whatever the process details, a multi-die part only pays off if each die can be tested before it is committed to a package, because one bad die would otherwise take every good die on the same interposer down with it.",
              "The interposer itself runs well past a single reticle field. TSMC says it `completed the certification of CoWoS advanced packaging solution for 5.5 times mask/reticle size interposers` in 2025 and will start volume production in 2026. Reporting points further, at a nine-reticle carrier with twelve HBM4 stacks around 2027, though that is press rather than a TSMC statement and should be read as such.",
              "Around the dies sit the memory stacks, and their arithmetic is division. An H100 SXM5 carries five HBM3 stacks for `80 GB` and `3.35 TB/s`, so each stack is `16 GB` at `670 GB/s`. A B200 carries eight HBM3e stacks for `192 GB` and `8 TB/s`, so each is `24 GB` at `1.0 TB/s`. The GPU chip lesson quotes those same totals rounded to `3.4e12` and `8.0e12` bytes per second; this is where the totals come from.",
              "Divide once more and you learn something about headroom. JEDEC's HBM3 update sets `6.4 Gb/s` per pin and `819 GB/s` per device, which is a 1024-bit interface if you do that division too. The H100's `670 GB/s` per stack works out to `5.23 Gb/s` per pin, about `82%` of what the standard allows, so five stacks at the ceiling would have been `4.1 TB/s`. The B200's `1.0 TB/s` per stack needs `7.8 Gb/s` per pin, past anything HBM3 permitted, which is what the E generation was for.",
              "Blackwell Ultra makes the last point on its own. It keeps eight stacks and the same `8 TB/s` but moves to 12-high stacks at `36 GB` each, `288 GB` in total. Capacity grew by stacking more dies in each tower. Bandwidth did not move at all, because bandwidth is set by the interface width and the pin rate rather than by how tall the stack is.",
            ],
            table: {
              caption: "HBM per part; the per-stack and per-pin columns are division, not vendor lines (H100 whitepaper p.18 and Table 3 · nvidia.com H100 datasheet · NVIDIA, Inside NVIDIA Blackwell Ultra · JEDEC JESD238)",
              cols: ["part", "stacks", "per stack", "total", "per pin"],
              rows: [
                ["H100 SXM5", "5 x HBM3", "16 GB, 670 GB/s", "80 GB, 3.35 TB/s", "5.23 Gb/s"],
                ["H100 PCIe", "5 x HBM2e", "16 GB, 400 GB/s", "80 GB, 2.0 TB/s (preliminary)", "3.13 Gb/s"],
                ["B200", "8 x HBM3e", "24 GB, 1.0 TB/s", "192 GB, 8 TB/s", "7.81 Gb/s"],
                ["B300", "8 x HBM3e, 12-Hi", "36 GB, 1.0 TB/s", "288 GB, 8 TB/s", "7.81 Gb/s"],
                ["JEDEC HBM3 ceiling", "n/a", "up to 64 GB, 819 GB/s", "n/a", "6.4 Gb/s"],
              ],
            },
          },
          {
            h: "SRAM buys area, DRAM buys a trip",
            ps: [
              "Every level of the memory hierarchy is one tradeoff applied at a different scale. An SRAM bit is a latch of a few transistors sitting on the same die as the logic, fast to reach and stable as long as the power is on. A DRAM bit is a capacitor whose charge leaks away, so it has to be refreshed, and it is small enough that you can afford billions of them on a die of their own.",
              "The area cost shows up plainly in a published floorplan. TPU v1 put `24 MiB` of Unified Buffer on chip, and Jouppi's ISCA paper describes that one block as `almost a third of the die`, on a 28 nm part whose total area Google withheld beyond a footnote saying `The TPU die is <= half the Haswell die size`. Twenty-four mebibytes of SRAM cost a third of a chip.",
              "Set that against the DRAM beside it. An H100 carries roughly `50 MB` of L2 on its `814 mm2` die and `80 GB` of HBM3 in five stacks around it, a capacity ratio near `1600` to one. The hierarchy exists because the cheap memory cannot be on the die and the fast memory cannot be large, and no amount of design cleverness moves either half of that sentence.",
              "The GPU chip lesson walks the ladder from registers down to HBM with the H100's capacities and bandwidths on it, and stage 0 turns the bottom rung into a ridge. What sits underneath both is the economics above: the fast levels are paid for in die area under a reticle limit, and the large level is paid for in stacks of a different kind of silicon placed on the same interposer.",
            ],
          },
          {
            h: "Blackwell: two dies at the limit",
            ps: [
              "NVIDIA states the whole design decision in one paragraph of the Blackwell technical brief.",
              ">> Each of the two dies are the largest die possible within the limits of reticle size, as big as can possibly be built today. The two dies are connected and unified with a single 10 terabyte-per-second (TB/s) chip-to-chip NVIDIA High-Bandwidth Interface (NV-HBI), providing one fully coherent, unified GPU.",
              "Read that against the three reasons for splitting a die and only one of them applies. Both halves sit at the reticle limit, so the harvest argument buys nothing here, and NVIDIA names a single process for the part, TSMC's 4NP, so it is not mixed-node integration either. Blackwell is split because the design wanted more transistors than one exposure can print: `208` billion of them, which the brief notes is `more than 2.5x the amount of transistors in NVIDIA Hopper GPUs`.",
              "You cannot get a per-die area out of NVIDIA. The brief says only `the largest die possible within the limits of reticle size` and prints no square millimetres, so a B200 die area quoted anywhere is somebody's estimate. A bound is still available. If both dies sat exactly at `858 mm2` the package would hold `1716 mm2` of silicon, and `208e9 / 1716` is `121` million transistors per mm2. An H100 puts `80` billion on `814 mm2`, which is `98` million per mm2, and an A100 put `54.2` billion on `826 mm2`, which is `66` million. So Blackwell's density is at least `1.23x` Hopper's, and higher still if the dies come in under the full field. That is a floor derived from a limit rather than a measurement, and it is worth more than a guessed area.",
              "The `10 TB/s` on NV-HBI has a gap of its own. NVIDIA does not say whether the figure is per direction or the sum of both, and publishes neither the width of the interface nor its signalling rate. Compare it to a number the same brief does decompose. For fifth-generation NVLink it prints the link count, the per-link rate, the direction and the total those multiply out to, so every term of the product is checkable by hand. NV-HBI gets none of that, so quote it as 10 TB/s with the direction unspecified and stop there.",
              "Both gaps are worth keeping as a habit rather than a complaint. A vendor brief is a marketing document that happens to contain engineering, and the parts it decomposes are the parts you can check by hand. Where the decomposition stops, saying so is the honest move, and it is the same discipline the TPU chapters apply to Google's unpublished clocks.",
            ],
            diagram: "blackwell-package",
          },
        ],
        readings: [
          {
            label: "ASML · TWINSCAN NXT:2050i",
            url: "https://www.asml.com/en/products/duv-lithography-systems/twinscan-nxt2050i",
            note: "the 26 x 33 mm field and the 4X reduction, from the company that builds the scanner",
          },
          {
            label: "Leachman · Yield modeling and analysis (IEOR 130, Berkeley)",
            url: "https://fog.misty.com/perry/cod/references/yield_models.pdf",
            note: "Poisson, Murphy, Seeds and the negative binomial, with the derivations and the cluster parameter",
          },
          {
            label: "Semiconductor Engineering · are larger reticle sizes on the horizon?",
            url: "https://semiengineering.com/are-larger-reticle-sizes-on-the-horizon/",
            note: "secondary, and the readable account of the high-NA half field and mask blank geometry",
          },
          {
            label: "TSMC · wafer-level system integration",
            url: "https://www.tsmc.com/english/dedicatedFoundry/technology/platform_HPC_tech_WLSI",
            note: "TSMC's own CoWoS wording, the three families, and the 5.5x reticle interposer certification",
          },
          {
            label: "NVIDIA · Blackwell architecture technical brief",
            url: "https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief",
            note: "the two-die paragraph, NV-HBI, and the 208 billion transistors, early in the document",
          },
          {
            label: "NVIDIA · H100 Tensor Core GPU architecture whitepaper",
            url: "https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf",
            note: "a mirror of NVIDIA's gated PDF; Table 3 carries the die areas and transistor counts used above",
          },
          {
            label: "JEDEC · HBM3 standard update",
            url: "https://www.jedec.org/news/pressreleases/jedec-publishes-hbm3-update-high-bandwidth-memory-hbm-standard",
            note: "6.4 Gb/s per pin and 819 GB/s per device, the ceiling every stack is measured against",
          },
        ],
        check: [
          {
            q: "Why does splitting a reticle-sized die into halves raise the number of dual-die parts per wafer, when the odds on any specific pair of halves are unchanged?",
            a: "Because you pair harvested survivors rather than fixed neighbours. Every half is tested and the dead ones are discarded, so the yield that counts is the per-half yield, 0.65 against 0.42 at D0 = 0.1, which is 1.54x as many parts from the same wafer area under Poisson.",
          },
          {
            q: "What is a B200 die's area, and what can you derive in its place?",
            a: "NVIDIA never publishes it; the brief says only that each die is the largest possible within the limits of reticle size. Assuming both dies sit at the 858 mm2 field gives a density floor of 121 million transistors per mm2, at least 1.23x an H100's 98 million.",
          },
          {
            q: "An H100 stack runs 5.23 Gb/s per pin against a JEDEC HBM3 ceiling of 6.4. What does that say about the 3.35 TB/s?",
            a: "That the total is about 82 percent of what the standard allowed, so it reflects a product decision rather than a hard limit of HBM3. Five stacks at the ceiling would have delivered 4.1 TB/s instead.",
          },
        ],
        work: [
          { id: 'yield', label: 'run the three yield models at D0 = 0.05 and 0.2 and say where the split stops paying', href: '#yield-falls-faster-than-area-grows' },
          { id: 'stacks', label: 'derive per-stack capacity and bandwidth for one part not in the table, from vendor figures only', href: '#the-interposer-and-the-stacks' },
          { id: 'bound', label: 'redo the Blackwell density floor assuming 800 mm2 per die and name what changes', href: '#blackwell-two-dies-at-the-limit' },
        ],
      },
      {
        id: "gpu-chip",
        num: 2,
        title: "The GPU chip",
        lede: "A TPU v5p has at most two big compute units. An H100 has 132 small ones, and most of the other differences follow from that count.",
        goal: "Given an H100 SM, name what sits in each of its four subpartitions and say which resource (registers, SMEM, or resident warps) your kernel runs out of first.",
        sections: [
          {
            h: "a hundred and thirty-two small machines",
            ps: [
              "Count the independent compute units on each chip and the two designs separate on the first line. A TPU has at most two TensorCores. An H100 has `132` streaming multiprocessors and a B200 measures at `148` (a microbenchmark figure; NVIDIA publishes no count), each one independent of the others, so a GPU can run hundreds of separate tasks at once. Any single SM is much weaker than a TPU TensorCore. The chip as a whole is much more flexible.",
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
                ["SMs / chip", "132", "148, measured"],
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
              "Hold that straight and the rest of the vocabulary lines up cleanly: warp scheduler against VPU, CUDA core against VPU ALU, SMEM against VMEM, HBM against HBM. The two-machines lesson later in this stage puts the counts side by side, which is where the two designs stop looking alike.",
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
        id: 'inside-the-sm',
        num: 3,
        title: 'Inside the SM, multiplied out',
        lede: "Multiply an H100's SM contents by a clock and NVIDIA's headline FLOPS come back out. NVIDIA never prints the clock, and the rates that would tell you one disagree with each other on the same page.",
        goal: 'Re-derive a chip’s headline FLOPS from unit counts and a clock, and when the arithmetic misses, name which SKU, which sparsity convention, which document, or which unpublished constant is responsible.',
        sections: [
          {
            h: 'the die you were shipped',
            ps: [
              "Read the H100 whitepaper's own configuration list and there are three chips on the page, not one. The full GH100 carries `144` SMs across 8 GPCs, `6` HBM stacks behind `12` 512-bit memory controllers, and `60 MB` of L2. The SXM5 part in the rack carries `132` SMs, `5` HBM3 stacks behind `10` controllers, `50 MB` of L2. The PCIe part carries `114` SMs and HBM2e. One design, three harvests, and the SKU decides which counts your arithmetic is allowed to start from.",
              'Twelve SMs and one memory stack are switched off somewhere between the drawing and the server. Why an `814 mm²` die ships with pieces disabled is the yield story, and this lesson needs only the consequence: put `144` into a FLOPS calculation for a part that has `132` and every number after it is high by about 9%.',
              'The tour of one SM, the SIMT model, and the divergence a TPU cannot have are the previous lesson (`/s/machine/gpu-chip`). What follows treats the same silicon as a parts list with a missing constant in it.',
            ],
            table: {
              caption: 'three configurations of one design · H100 whitepaper p.18',
              cols: ['unit', 'GH100 full', 'H100 SXM5', 'H100 PCIe'],
              rows: [
                ['SMs', '144', '132', '114'],
                ['FP32 cores', '18,432', '16,896', '14,592'],
                ['Tensor Cores', '576', '528', '456'],
                ['memory', '6 stacks', '5 x HBM3, 80 GB', '5 x HBM2e, 80 GB'],
                ['memory controllers', '12 x 512-bit', '10 x 512-bit', '10 x 512-bit'],
                ['L2', '60 MB', '50 MB', '50 MB'],
              ],
            },
          },
          {
            h: 'the parts list you multiply',
            ps: [
              'An SM is four processing blocks and a shared floor. Each block holds an L0 instruction cache, one warp scheduler issuing `32 thread/clk`, one dispatch unit at the same rate, a register file of `16,384` 32-bit words, `32` FP32 lanes, `16` INT32, `16` FP64, `8` load/store units, one SFU, and one fourth-generation Tensor Core. The floor underneath them is `256 KB` of combined L1 data cache and shared memory, plus a Tensor Memory Accelerator and four texture units.',
              'Multiply the blocks out and the whitepaper’s comparison table confirms every total in text: `128` FP32 cores per SM, `64` FP64 excluding the Tensor Cores, `64` INT32, `4` Tensor Cores, and `65,536` 32-bit registers, which is the `256 KB` register file. Shared memory is carved out of the `256 KB` block and is "configurable up to 228 KB"; the whitepaper describes the block itself only as "1.33x larger than A100", which is how you get A100’s `192 KB`.',
              'Keep the A100 column in view. It is the one that carries a clock, and that is what makes it useful for anything past nostalgia.',
            ],
            table: {
              caption: 'per SM · whitepaper Table 3 (p.39) and Table 4 (p.41); the last row is what this lesson is about',
              cols: ['per SM', 'A100', 'H100'],
              rows: [
                ['FP32 cores', '64', '128'],
                ['FP64 cores, excluding Tensor', '32', '64'],
                ['INT32 cores', '64', '64'],
                ['Tensor Cores', '4', '4'],
                ['register file', '256 KB (65,536 x 32-bit)', '256 KB (65,536 x 32-bit)'],
                ['L1 + shared block', '192 KB', '256 KB'],
                ['shared memory', 'configurable up to 164 KB', 'configurable up to 228 KB'],
                ['max resident warps', '64', '64'],
                ['GPU boost clock', '1410 MHz', 'Not Finalized'],
              ],
            },
          },
          {
            h: 'sparse or dense, on the same line',
            ps: [
              'The shipping datasheet lists BF16 Tensor Core at `1,979 TFLOPS` with an asterisk, and the asterisk resolves at the bottom of the page to "With sparsity". The whitepaper writes the same kind of entry as a pair, `1000/2000`, under footnote 2: "Effective TOPS / TFLOPS using the Sparsity feature". Two notations, one convention, and the larger number is never the one you multiply against.',
              'What is being assumed is 2:4 structured sparsity, a Tensor Core mode that the whitepaper says works by "exploit[ing] fine-grained structured sparsity in deep learning networks, doubling the performance of standard Tensor Core operations". Half of every group of four weights is zero, the hardware skips them, and the marketing number doubles. Dense is the starred value halved. FP32, FP64 and FP64 Tensor Core carry no asterisk and no multiplier.',
              'Get this wrong and the error is loud rather than subtle. Feed `1,979` into the clock derivation below and it asks for `3.66 GHz`, which no GPU has ever run at. On a narrower question the same mistake would have passed quietly, which is the argument for doing the multiplication at all.',
            ],
          },
          {
            h: 'predict the clock',
            ps: [
              'One constant is missing and NVIDIA does not publish it: FLOPs per SM per clock through the Tensor Cores. Pin it on the chip whose clock is printed. The A100 row gives `108` SMs at `1410 MHz` and FP16 Tensor `312/624`, so dense is `312 TFLOPS`, and `312e12 / (108 x 1.41e9)` is `2048.9`. Call it `2048` and the A100 reproduces itself to a tenth of a TFLOP.',
              'Hopper is then one sentence away. The whitepaper says its Tensor Cores "deliver 2x the MMA (Matrix Multiply-Accumulate) computational rates of the A100 SM on equivalent data types", per SM and per clock, so the constant is `4096` per SM, `1024` per Tensor Core. Across `132` SMs that is `540,672` FLOPs per clock, and the datasheet’s dense BF16 rate of `989.5 TFLOPS` divides out to `1.830 GHz`.',
              'The previous lesson runs the same equation with the other unknown fixed: it takes the scaling book’s `1.76 GHz` as given and solves for the per-Tensor-Core rate, landing near `1024`. Pinning the constant instead and solving for the clock is the version you can source end to end, because `2048` comes from a table that prints its own clock. Third-party GPU databases list `1830 MHz` boost for H100 SXM5, which agrees. Treat that agreement as a check on the arithmetic, not as the missing datasheet line.',
            ],
            code: {
              caption: 'the derivation, pinned on A100 and carried to Hopper',
              lang: 'text',
              text: `pin the constant where the clock is published
  A100    108 SMs x 2048 FP16 FLOP/SM/clk x 1.41 GHz = 311.9 TFLOPS
  printed                                              312   TFLOPS   [Table 3, p.39]

carry it across one documented sentence
  "2x the MMA ... rates of the A100 SM"             -> 4096 FLOP/SM/clk [p.22]
  132 SMs x 4096                                    =  540,672 FLOP/clk
  datasheet BF16 Tensor  1,979 TFLOPS "* With sparsity"
  dense                                             =  989.5 TFLOPS
  implied clock  989.5e12 / 540,672                 =  1.830 GHz`,
            },
          },
          {
            h: 'the datasheet that does not close',
            ps: [
              'Now run the same move on the rates that do not go through a Tensor Core. FP32 is `67 TFLOPS` over `132 x 128 x 2 = 33,792` FLOPs per clock, which asks for `1.983 GHz`. FP64 is `34 TFLOPS` over `132 x 64 x 2 = 16,896`, which asks for `2.012 GHz`. Neither is `1.830`.',
              'Some of that spread is rounding, and the reconciliation is worth doing rather than waving at. A printed `67` covers any clock from `1.968` to `1.998 GHz`, and a printed `34` covers `1.983` to `2.042`, so FP32 and FP64 are both consistent with a single clock in a narrow band just under `2 GHz`. The `1.830 GHz` the Tensor rows demand is nowhere near that band. Two families of numbers, two clocks, one page.',
              'One row on the same page cannot be run through this arithmetic at all. FP64 Tensor Core prints `67 TFLOPS`, which is the FP32 number over again, and the whitepaper repeats that coincidence in all three of its columns: `60` for both on SXM5, `48` for both on PCIe, `19.5` for both on A100. Set it against the plain FP64 row instead and the ratio is exactly two in the whitepaper’s H100 columns, `60` over `30` and `48` over `24`, and only roughly two on the shipping datasheet, where doubling `34` gives `68` against a printed `67`. Rounding accounts for that gap: a single underlying rate between `33.5` and `33.75 TFLOPS` prints as `34` and doubles into a `67`, the same way A100’s `9.7` is a rounded `9.746`. But NVIDIA never prints an FP64 rate per clock through the Tensor Cores, so that row’s divisor can only be inferred from the doubling rather than counted off a table, and the table below records it as unavailable rather than filling it in.',
              'The whitepaper’s preliminary column does the same thing with different values: its FP32 `60` and its FP64 `30 TFLOPS` both imply `1.776 GHz` to four figures, while its dense FP16 Tensor rate of `1000` implies `1.850`. The PCIe column repeats the pattern, `48` and `24` both giving `1.645 GHz` against `1.713` from its Tensor row. The gap survives the change of document and of SKU, and it even flips direction between the preliminary table and the shipping one.',
              'Which leaves the clock itself. The whitepaper prints "Not Finalized" where the boost clock goes, for both H100 parts, and the shipping datasheet page lists no clock at all. So there is no vendor number to reconcile these two families against, and the honest end of the chain is to say that. A derivation that fails to close and reports why is worth more than one that quietly picks the clock that made it work.',
              '>> A peak-FLOPS table is not necessarily quoted at one clock.',
            ],
            table: {
              caption: 'every clock the two documents imply, and neither document prints one',
              cols: ['rate on the page', 'value', 'FLOPs per clock', 'implied clock'],
              rows: [
                ['datasheet BF16/FP16 Tensor, dense', '989.5 TFLOPS', '132 x 4096 = 540,672', '1.830 GHz'],
                ['datasheet FP32', '67 TFLOPS', '132 x 128 x 2 = 33,792', '1.983 GHz'],
                ['datasheet FP64', '34 TFLOPS', '132 x 64 x 2 = 16,896', '2.012 GHz'],
                ['datasheet FP64 Tensor Core', '67 TFLOPS', 'not published', 'cannot be derived'],
                ['whitepaper preliminary FP32', '60 TFLOPS', '33,792', '1.776 GHz'],
                ['whitepaper preliminary FP64', '30 TFLOPS', '16,896', '1.776 GHz'],
                ['whitepaper preliminary FP16 Tensor, dense', '1,000 TFLOPS', '540,672', '1.850 GHz'],
                ['printed clock, either document', 'none', 'n/a', 'Not Finalized'],
              ],
            },
          },
          {
            h: 'the same move on the memory side',
            ps: [
              'Bandwidth divides as cleanly as FLOPS multiply. Five HBM3 stacks carry the datasheet’s `3.35 TB/s`, so each stack delivers `670 GB/s`, and `80 GB` over five stacks is `16 GB` a stack. The `5120-bit` memory interface across `10` 512-bit controllers puts `1024` bits on each stack, which turns `670 GB/s` into `5.23 Gb/s` per pin.',
              'That last figure is the one with an external yardstick. JEDEC’s HBM3 update tops out at `6.4 Gb/s` per pin and `819 GB/s` per device, so an H100 runs its memory at about `82%` of the standard’s ceiling. This division closes because both of its inputs are printed, which is the difference between it and the clock.',
              'Check which document you are holding here too. The whitepaper’s preliminary table says `3000 GB/sec` for the same part that the shipping datasheet sells at `3.35 TB/s`, and a roofline built on the older figure is off by 12% before you start.',
              'The habit underneath all of this is short. Write the multiplication before you look up the answer, then reconcile: wrong SKU, sparsity convention, preliminary document, unpublished constant, in that order. When the gap survives every one of those, you have found something about the vendor’s numbers rather than about your own arithmetic.',
            ],
          },
        ],
        readings: [
          {
            label: 'NVIDIA H100 Tensor Core GPU architecture whitepaper',
            url: 'https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf',
            note: 'v1.02, mirrored; the three configurations on p.18, the SM figure on p.21, and "Not Finalized" in Table 3',
          },
          {
            label: 'NVIDIA H100 product page and datasheet',
            url: 'https://www.nvidia.com/en-us/data-center/h100/',
            note: 'the shipping rates with the sparsity asterisk, and no clock anywhere on the page',
          },
          {
            label: 'JEDEC publishes HBM3 update',
            url: 'https://www.jedec.org/news/pressreleases/jedec-publishes-hbm3-update-high-bandwidth-memory-hbm-standard',
            note: 'the 6.4 Gb/s per pin and 819 GB/s per device the H100 is measured against',
          },
        ],
        check: [
          {
            q: 'The datasheet prints 1,979 TFLOPS for BF16 Tensor Core. What happens if you put that straight into the clock derivation?',
            a: 'It asks for 3.66 GHz. The asterisk means sparsity, so the dense rate is half of it, 989.5 TFLOPS, and the derivation lands on 1.830 GHz instead.',
          },
          {
            q: 'Why does the FLOPS arithmetic close on an A100 and fail on an H100?',
            a: "The A100's table prints 1410 MHz, so 108 SMs x 2048 FP16 FLOP/SM/clk gives 311.9 TFLOPS against a printed 312. The H100 prints no clock at all, and its own rates imply 1.830 GHz from the Tensor rows against roughly 1.98 to 2.01 GHz from FP32 and FP64.",
          },
          {
            q: 'An H100 SXM5 moves 3.35 TB/s across five HBM3 stacks. How close is that to the JEDEC ceiling?',
            a: '670 GB/s per stack, which over a 1024-bit stack is 5.23 Gb/s per pin, against JEDEC HBM3 at 6.4 Gb/s per pin and 819 GB/s per device. About 82 percent of the ceiling.',
          },
        ],
        work: [
          { id: 'derive', label: 'derive the H100 tensor clock from the A100 constant with the answer covered', href: '#predict-the-clock' },
          { id: 'preliminary', label: "run both derivations on the whitepaper's preliminary column and show FP32 and FP64 land on one clock", href: '#the-datasheet-that-does-not-close' },
          { id: 'pcie', label: 'redo the FP32, FP64 and Tensor derivations for the 114-SM PCIe part and say which inputs changed' },
        ],
      },
      {
        id: "two-machines",
        num: 4,
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
              "It is also where the two chips stop being the whole story. One question is left at this scale, which is what happens when the thing you shard across is a rack rather than a chip, and the last lesson of the stage answers it before the path turns to authorship.",
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
      {
        id: 'the-isa-contract',
        num: 5,
        title: 'The ISA contract',
        lede: 'Two words get used as though they named the same machine. One is a promise that has to survive a decade of silicon; the other is silicon that gets thrown away every two years.',
        goal: 'Take any GPU dump and say which instruction set it is, which tool produced it, and what is free to change underneath it before your binary stops running.',
        sections: [
          {
            h: 'the same instruction, twice',
            ps: [
              "Compile four multiplies and four adds on an ordinary laptop, disassemble the object file, and every line comes back in two columns. On the right, `addss %xmm0, %xmm2`. On the left, the four bytes `f3 0f 58 d0`. They are not two things that correspond. They are one instruction written down twice, once for a person and once for the decoder, and an assembler turns either into the other without losing anything.",
              'The left column is **machine code**, the encoding the hardware actually eats. The right column is **assembly**, a notation for humans with mnemonics instead of opcodes and names instead of register numbers. Because the two map one to one, people say "writing assembly" and "the machine\'s instructions" as if they were the same claim, and mostly nothing goes wrong. The distinction that does matter sits one level up from this pair.',
            ],
            code: {
              caption: 'clang 21.0.0 on x86-64, `clang -O2 -c dot4.c -o dot4.o` then `objdump -d dot4.o`; first seven lines of the function',
              lang: 'text',
              text: `0000000000000000 <_dot4>:
       0: 55                           	pushq	%rbp
       1: 48 89 e5                     	movq	%rsp, %rbp
       4: f3 0f 10 07                  	movss	(%rdi), %xmm0
       8: f3 0f 10 4f 04               	movss	0x4(%rdi), %xmm1
       d: f3 0f 59 06                  	mulss	(%rsi), %xmm0
      11: 0f 57 d2                     	xorps	%xmm2, %xmm2
      14: f3 0f 58 d0                  	addss	%xmm0, %xmm2`,
            },
          },
          {
            h: 'what the chip promises',
            ps: [
              'Those four bytes have meant the same thing since 1999, and they mean it on parts from two different companies. The agreement covers which instructions exist, how they are encoded, which registers they may name, what the memory model guarantees about the order other cores see writes in. All of that together is the **instruction set architecture**, and it is a contract: code compiled against it keeps running on machines nobody had designed when it was compiled.',
              'What the contract deliberately leaves out is everything about how the work gets done. How many multiplies issue per cycle, how deep the pipeline runs, whether there is a reorder buffer at all, how large the caches are, how the branch predictor is organized. That is the **microarchitecture**, one implementation of the contract, and it is redesigned every generation or two. The dump above was produced on a Core i9-9880H; the same bytes run unchanged on an AMD part that shares none of its internals.',
              '>> The ISA is the part that may not change. Everything under it is free to.',
              'On a CPU the contract and the machine sit at the same address, so the two words collapse in casual speech and no harm comes of it. NVIDIA moved the line. There are two instruction sets in a CUDA binary, they are not versions of each other, and only one of them is the contract.',
            ],
          },
          {
            h: 'a machine that does not exist',
            ps: [
              'The first instruction set is PTX, and the specification opens by saying exactly what it is: "a low-level parallel thread execution virtual machine and instruction set architecture (ISA)". Virtual machine, meaning no chip decodes it. PTX has as many registers as a kernel wants, one type per instruction, and special registers with names like `%tid.x` rather than an address. It is an ISA in the sense of being a complete, documented, versioned contract, and in no other sense.',
              'What the contract is for shows up in the goals list, one line of which reads "Provide a stable ISA that spans multiple GPU generations." A **virtual ISA** is a contract with the implementation deliberately missing: something for a compiler to target and something for a translator to consume, with the translation postponed. The specification says when the postponement ends, too: "PTX programs are translated at install time to the target hardware instruction set."',
              'Read the dump below and the virtual part is visible in the register names. Every value gets a fresh one, `%f1` through `%f4` for floats and `%rd1` through `%rd7` for addresses, because nothing here is competing for a physical register file yet. The thread index arrives through `%ctaid.x`, `%ntid.x`, and `%tid.x`; the bounds check becomes a `setp.ge.s32` writing a predicate and a `@%p1 bra` reading it; the multiply-add is one `fma.rn.f32` with its rounding mode spelled out.',
            ],
            code: {
              caption: 'saxpy at the virtual level: nvcc 13.3.0, `-arch=sm_90 -O3`, PTX pane (captured via the Compiler Explorer API, 2026-08-14)',
              lang: 'text',
              text: `.visible .entry saxpy(
	.param .f32 saxpy_param_0,
	.param .u64 saxpy_param_1,
	.param .u64 saxpy_param_2,
	.param .u32 saxpy_param_3
)
{
	ld.param.f32 	%f1, [saxpy_param_0];
	ld.param.u64 	%rd1, [saxpy_param_1];
	ld.param.u64 	%rd2, [saxpy_param_2];
	ld.param.u32 	%r2, [saxpy_param_3];
	mov.u32 	%r3, %ctaid.x;
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;
	cvta.to.global.u64 	%rd3, %rd2;
	cvta.to.global.u64 	%rd4, %rd1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.f32 	%f2, [%rd6];
	add.s64 	%rd7, %rd3, %rd5;
	ld.global.f32 	%f3, [%rd7];
	fma.rn.f32 	%f4, %f2, %f1, %f3;
	st.global.f32 	[%rd7], %f4;
$L__BB0_2:
	ret;
}`,
            },
          },
          {
            h: 'the floor that actually runs',
            ps: [
              'Push the same kernel one step further and the second instruction set appears. SASS is what the streaming multiprocessors decode, and everything postponed in PTX has been settled in it. Registers are physical and numbered, `R0` through `R7` here. The kernel parameters are no longer named; they are offsets into a constant bank, `c[0x0][0x210]` and `c[0x0][0x218]`. The bounds check is now `ISETP.GE.AND` writing the predicate register `P0`, and the threads that fail it leave immediately through a predicated `@P0 EXIT`.',
              'One line is worth pausing on. The whole body of the PTX, four separate address computations and two loads, has collapsed into `IMAD.WIDE` pairs and `LDG.E` with a descriptor operand. Nothing in the source asked for that. It is the assembler choosing instructions for one particular generation of hardware, which is what an assembler for a real ISA does.',
            ],
            code: {
              caption: 'the same kernel after ptxas: nvcc 13.3.0, `-arch=sm_90 -O3`, SASS pane (Compiler Explorer API, 2026-08-14); trailing NOP padding elided',
              lang: 'text',
              text: `saxpy:
 LDC R1, c[0x0][0x28]
 S2R R0, SR_TID.X
 S2UR UR4, SR_CTAID.X
 LDC R7, c[0x0][RZ]
 IMAD R7, R7, UR4, R0
 ULDC UR4, c[0x0][0x228]
 ISETP.GE.AND P0, PT, R7, UR4, PT
 @P0 EXIT
 LDC.64 R2, c[0x0][0x218]
 ULDC.64 UR4, c[0x0][0x208]
 ULDC UR6, c[0x0][0x210]
 LDC.64 R4, c[0x0][0x220]
 IMAD.WIDE R2, R7, 0x4, R2
 LDG.E R2, desc[UR4][R2.64]
 IMAD.WIDE R4, R7, 0x4, R4
 LDG.E R7, desc[UR4][R4.64]
 FFMA R7, R2, UR6, R7
 STG.E desc[UR4][R4.64], R7
 EXIT`,
            },
          },
          {
            h: 'one contract, many floors',
            ps: [
              'Now run the experiment that makes the split concrete. Take that source, one nvcc, and ask for two architectures. The PTX comes back byte for byte identical, the same thirty lines both times. The SASS does not, and the differences are not cosmetic: `MOV R1` where Hopper used `LDC R1`, a bare `LDG.E.SYS R2, [R2]` where Hopper carried a descriptor, an `IMAD.WIDE` that folds the base address straight out of the constant bank, and every parameter at a different offset, `0x160` here against `0x210` there.',
              'Nothing about the program changed between those two dumps. The contract held and the floor moved, which is the arrangement the goals list was describing. It also explains why NVIDIA behaves so differently about the two levels. The PTX specification is a document with a version number you can read cover to cover. For SASS the binary utilities manual gives you `nvdisasm`, which "extracts information from standalone cubin files and presents them in human readable format", an opcode table per architecture, and no assembler. SASS is something you read.',
            ],
            code: {
              caption: 'same source, same compiler (nvcc 12.9.1), `-arch=sm_70 -O3`: the PTX is identical to the block above, the SASS is not',
              lang: 'text',
              text: `saxpy:
 MOV R1, c[0x0][0x28]
 @!PT SHFL.IDX PT, RZ, RZ, RZ, RZ
 S2R R4, SR_CTAID.X
 S2R R3, SR_TID.X
 IMAD R4, R4, c[0x0][0x0], R3
 ISETP.GE.AND P0, PT, R4, c[0x0][0x178], PT
 @P0 EXIT
 MOV R5, 0x4
 IMAD.WIDE R2, R4, R5, c[0x0][0x168]
 IMAD.WIDE R4, R4, R5, c[0x0][0x170]
 LDG.E.SYS R2, [R2]
 LDG.E.SYS R7, [R4]
 FFMA R7, R2, c[0x0][0x160], R7
 STG.E.SYS [R4], R7
 EXIT`,
            },
          },
          {
            h: 'who translates, and when',
            ps: [
              'Two tools cross the gap, and knowing which one ran explains most surprising launch times. At build time it is `ptxas`, which the nvcc manual calls "the PTX optimizing assembler". You tell nvcc which architectures to target, and it names them in two vocabularies: it produces "a true binary load image for each real architecture (such as sm_100), and PTX code for the virtual architecture (such as compute_100)". Both go into the same object file, so a shipped binary usually carries several cubins and one copy of the PTX they all came from.',
              'The second tool is the driver, and it runs when the first list comes up short: "During runtime, such embedded PTX code is dynamically compiled by the CUDA runtime system if no binary load image is found for the current GPU." That is the mechanism behind a binary from two years ago starting on a card that did not exist when it was built. It is also the mechanism behind that binary taking an unreasonable amount of time on its first launch, because a compiler you never invoked is running inside the driver, on your critical path, over code you cannot inspect.',
              'The wall lesson in the compiler unit already puts `ptxas` and SASS where the toolchain stops (`/l/xla/dumps-on-demand`, which names libtpu and LLO in the same breath). This lesson says what is behind that door and why the vendor wants it shut: the floor is redesigned every generation, and a contract one level up is what lets them do it. The next question is not which instructions are down there. It is who decided which of them run at the same time.',
            ],
          },
        ],
        readings: [
          {
            label: 'NVIDIA · PTX ISA',
            url: 'https://docs.nvidia.com/cuda/parallel-thread-execution/',
            note: 'section 1.2 is the entire contract argument in six bullet points',
          },
          {
            label: 'NVIDIA · CUDA compiler driver NVCC',
            url: 'https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/',
            note: 'virtual against real architectures, and where ptxas sits in the trajectory',
          },
          {
            label: 'NVIDIA · CUDA binary utilities',
            url: 'https://docs.nvidia.com/cuda/cuda-binary-utilities/',
            note: 'cuobjdump, nvdisasm, and one opcode table per architecture with no assembler beside it',
          },
          {
            label: 'Compiler Explorer',
            url: 'https://godbolt.org/',
            note: 'produces both dumps for a kernel you type, with no GPU anywhere in the room',
          },
        ],
        check: [
          {
            q: 'One nvcc, one source file, two architectures: the PTX comes back identical and the SASS does not. Which of the two is the contract, and what changed underneath it?',
            a: 'PTX is the contract, a virtual ISA whose stated goal is spanning multiple GPU generations. What changed is the machine ISA below it, SASS, which ptxas re-selects per architecture: different opcodes for the same load, different constant-bank offsets for the same parameters.',
          },
          {
            q: 'A binary built two years ago launches on a card that did not exist then, and the first launch takes far longer than the second. What ran?',
            a: 'The driver JIT. No cubin in the fatbinary matched the current GPU, so the embedded PTX was compiled at runtime by the CUDA runtime system, on your critical path, before the kernel could start.',
          },
          {
            q: 'Why can you author PTX by hand but not SASS?',
            a: 'PTX is published as a versioned specification with a documented assembler path through ptxas. For SASS NVIDIA publishes a per-architecture opcode table and disassemblers, and no assembler, so the only supported producer is ptxas and the only supported use is reading.',
          },
        ],
        work: [
          { id: 'trace', label: 'take one PTX instruction from the dump and find what it became in both SASS listings', href: '#one-contract-many-floors' },
          { id: 'godbolt', label: 'put a kernel of your own through Compiler Explorer and read the PTX and SASS panes side by side' },
          { id: 'check', label: 'answer the three checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'who-schedules',
        num: 6,
        title: 'Who decides what runs at once',
        lede: 'Four famous names get taught as though they were four rungs of one ladder. They are answers to two different questions, and each sorts in a sentence once you know which question it answers.',
        goal: 'Sort VLIW, superscalar out-of-order execution, SIMD, and SIMT by the question each one answers, and for a given machine say what had to be decided before the program started running.',
        sections: [
          {
            h: 'one question, and the vocabulary it sorts',
            ps: [
              'Whenever two operations run in the same cycle, somebody decided they were allowed to. There are only two candidates for who. A compiler could have decided, before the program ran, by proving the two operations independent and placing them accordingly. Or hardware could decide while the program runs, by inspecting instructions as they arrive. Nearly every architecture word in this area is an answer to that question, and the answers split cleanly by when the deciding happens.',
              'A second question gets mixed into the same conversation and should not be. How much data does one instruction move? That is width, and it is independent of scheduling: a machine can be wide and statically scheduled, wide and dynamically scheduled, or narrow and either. Two of the four names below answer the scheduling question and two answer the width question, which is why they never quite line up when taught as a sequence.',
              'The table sorts them. Read the second column first, because it says which of the two questions a row is even answering, and only then read across to who decides. Of the two rows that answer the scheduling question, exactly one hands the work to a compiler.',
            ],
            table: {
              caption: 'four names, two questions',
              cols: ['name', 'the question it answers', 'who decides', 'when'],
              rows: [
                ['superscalar, out-of-order', 'which nearby instructions may issue together', 'the hardware, from a window it maintains itself', 'run time, on every run'],
                ['VLIW', 'which operations may issue together', 'the compiler, by placing them in one wide instruction', 'compile time, once'],
                ['SIMD', 'how many elements one instruction moves', 'the compiler, by choosing that instruction', 'compile time, once'],
                ['SIMT', 'how many threads share one instruction stream', 'a warp scheduler, one warp at a time', 'run time, on every run'],
              ],
            },
          },
          {
            h: 'the schedule the machine may ignore',
            ps: [
              'A modern CPU core fetches several instructions per cycle, renames their registers so false dependencies disappear, and issues each one from a window as soon as its operands are ready, in whatever order that turns out to be. Retirement puts the program order back for anything the outside world can see. Issuing several per cycle is what **superscalar** names; issuing them in a different order than written is what **out-of-order** names, and the two travel together in practice.',
              'The compiler still emits one linear sequence, and it still tries to order that sequence well for a particular chip. Ask clang for the same function twice with the instruction set pinned and only the tuning target changed, and you get the two listings below: the same thirteen instructions with the same operands, issued in a different order. Under `-mtune=skylake` each load is paired with its multiply. Under `-mtune=znver3` the three loads are hoisted to the front.',
              'On a machine with a reorder buffer that ordering is a hint. The window is free to rearrange it, and `-mtune` mostly shifts where the rearranging starts from. What the hardware buys with all that area is knowledge the compiler could not have had: whether a load hit in cache, which way a branch actually went, whether two pointers turned out to alias. When those answers change from run to run, only a decision made at run time can use them.',
            ],
            code: {
              caption: 'clang 21.0.0, `clang -O2 -march=x86-64 -mtune=<target> -S`, arithmetic lines only; the instruction multiset is identical between the two',
              lang: 'text',
              text: `# -mtune=skylake
	movss	(%rdi), %xmm0
	movss	4(%rdi), %xmm1
	mulss	(%rsi), %xmm0
	xorps	%xmm2, %xmm2
	addss	%xmm0, %xmm2
	mulss	4(%rsi), %xmm1
	movss	8(%rdi), %xmm3

# -mtune=znver3, same -march, same source
	movss	(%rdi), %xmm0
	movss	4(%rdi), %xmm1
	movss	8(%rdi), %xmm3
	xorps	%xmm2, %xmm2
	mulss	(%rsi), %xmm0
	mulss	4(%rsi), %xmm1
	mulss	8(%rsi), %xmm3`,
            },
          },
          {
            h: 'one instruction, many lanes',
            ps: [
              'Width is the other axis, and the baseline x86-64 instruction set already has it. Compile a saxpy loop with no tuning flags at all and the inner loop comes back holding `mulps %xmm1, %xmm2`, which multiplies four single-precision floats in one instruction, and `movups`, which moves sixteen bytes. One instruction, four lanes, one program counter for all of them. The model has a name older than any of the chips in this course: **SIMD**, single instruction multiple data.',
              'Notice what SIMD does not tell you. It says how much data an instruction covers and nothing about who scheduled the instruction. Here the compiler chose the width, and the out-of-order core will still reorder the result. On a TPU the VPU is SIMD too, at the `(8, 128)` shape the chip lesson works through, and there the compiler chooses the width and the schedule. Same width model, opposite answers on the other axis.',
            ],
            code: {
              caption: 'clang 21.0.0, `clang -O2 -march=x86-64 -S`, the vectorized inner loop of `y[i] = a * x[i] + y[i]`',
              lang: 'text',
              text: `LBB0_6:
	movups	(%rsi,%r8), %xmm2
	movups	16(%rsi,%r8), %xmm3
	movups	(%rdi,%r8), %xmm4
	movups	16(%rdi,%r8), %xmm5
	mulps	%xmm1, %xmm2
	addps	%xmm4, %xmm2
	mulps	%xmm1, %xmm3
	addps	%xmm5, %xmm3
	movups	%xmm2, (%rdi,%r8)
	movups	%xmm3, 16(%rdi,%r8)
	addq	$32, %r8
	cmpq	%r8, %rdx
	jne	LBB0_6`,
            },
          },
          {
            h: 'one instruction, many threads',
            ps: [
              'This unit already introduced the GPU\'s answer to the width question, and the hazard that comes with it (`/s/machine/gpu-chip`). The vendor states both plainly: "Each SM creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps", and "A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path." When they disagree, "the warp executes each branch path taken, disabling threads that are not on that path". The etymology in the same section is worth carrying: the term warp comes from weaving.',
              'That description is usually taken on faith. It does not have to be. Compile a kernel whose two arms write to different arrays, look at the SASS, and the masking is written down: every instruction of the else arm carries the predicate `@!P0`, the store included, and so does the `@!P0 EXIT` that retires those threads. A warp holding both kinds of thread walks that entire block with the wrong lanes switched off, then walks the if arm. Both arms are in the instruction stream, one after the other, exactly as advertised.',
              'On the sorting question, **SIMT** answers width the way SIMD does and then adds something SIMD has no equivalent for. Each of the 32 threads carries its own program counter, and a warp scheduler picks which resident warp issues next. That second half is a scheduling decision made by hardware at run time, which is why SIMT lands in two rows of the table at once. One more thing to hold before you count branches in a kernel: when both arms are cheap enough, the compiler removes the divergence entirely by turning the branch into a select, and the same experiment run on a two-line body comes back with an `FSEL` and no predicate at all.',
            ],
            code: {
              caption: 'the else arm of a divergent kernel, every line predicated: nvcc 13.3.0, `-arch=sm_90 -O3`, SASS pane (Compiler Explorer API, 2026-08-14)',
              lang: 'text',
              text: ` FSETP.GT.AND P0, PT, R0, RZ, PT
 @!P0 LDC.64 R4, c[0x0][0x220]
 @!P0 FADD R9, -R0, -RZ
 @!P0 LEA R4, P1, R7, R4, 0x2
 @!P0 LEA.HI.X R5, R7, R5, R6, 0x2, P1
 @!P0 STG.E desc[UR4][R4.64], R9
 @!P0 EXIT`,
              full: {
                text: `two_paths:
 LDC R1, c[0x0][0x28]
 S2R R7, SR_TID.X
 LDC.64 R2, c[0x0][0x210]
 ULDC.64 UR4, c[0x0][0x208]
 IMAD.WIDE R2, R7, 0x4, R2
 LDG.E R0, desc[UR4][R2.64]
 SHF.R.S32.HI R6, RZ, 0x1f, R7
 FSETP.GT.AND P0, PT, R0, RZ, PT
 @!P0 LDC.64 R4, c[0x0][0x220]
 @!P0 FADD R9, -R0, -RZ
 @!P0 LEA R4, P1, R7, R4, 0x2
 @!P0 LEA.HI.X R5, R7, R5, R6, 0x2, P1
 @!P0 STG.E desc[UR4][R4.64], R9
 @!P0 EXIT
 ULDC.64 UR6, c[0x0][0x218]
 FMUL R5, R0, R0
 LEA R2, P0, R7, UR6, 0x2
 LEA.HI.X R3, R7, UR7, R6, 0x2, P0
 STG.E desc[UR4][R2.64], R5
 EXIT`,
                label: 'the whole kernel, both arms',
              },
            },
          },
          {
            h: 'why the compiler wins on this workload',
            ps: [
              'This unit has already put the two machines on opposite sides of this question and left it there (`/s/machine/two-machines`): the GPU is hardware-scheduled, the TPU is compiler-scheduled. The part worth arguing now is why the second bet pays on dense linear algebra specifically, because as a general architectural choice it has lost before.',
              'Out-of-order execution earns its transistors by covering uncertainty. A cache miss whose latency nobody knows in advance, a branch whose direction depends on data, pointers that might or might not alias. A blocked matmul supplies none of that. Trip counts are compile-time constants, addresses are affine functions of the loop indices, the only branches are loop back-edges, and the memory is a software-managed scratchpad, so a load takes a number of cycles rather than a distribution of them. Every question the reorder buffer exists to answer at run time already has an answer at compile time.',
              'The TPU team stated the trade in the first TPU paper: "The TPU\'s deterministic execution model is a better match to the 99th-percentile response-time requirement of our NN applications than are the time-varying optimizations of CPUs and GPUs (caches, out-of-order execution, multithreading, multiprocessing, prefetching, ...) that help average throughput more than guaranteed latency." The later training-chip paper lists what got deleted to pay for the multipliers, "dropping general-purpose features irrelevant for DNNs but critical for CPUs such as caches and branch predictors".',
            ],
          },
          {
            h: 'the bundle, and the line it forces',
            ps: [
              'Delete the scheduler and something has to emit the schedule. The instruction format is where it goes: on TPUv2 and v3, "the 322-bit VLIW instruction can launch eight operations: two scalar, two vector ALU, vector load and store, and a pair of slots that queue data to and from the matrix multiply and transpose units". An instruction word with independent slots that all issue together is a **VLIW**, a very long instruction word, and filling the slots is the compiler\'s job. The same paper names which compiler: "TPUs use a VLIW architecture to express instruction-level parallelism to the many compute units of a TensorCore. XLA uses standard VLIW compilation techniques including loop unrolling, instruction scheduling, and software pipelining to keep all compute units busy."',
              '>> Hardware schedules what it can see. A compiler sees the whole loop nest.',
              'A VLIW schedule is correct for the latencies it was built against. Change the depth of one unit by a cycle and the bundles are wrong, so the format is tied to a generation in a way an ISA is not supposed to be. Both vendors reached the same arrangement: publish a stable layer, keep the schedule-bearing layer private, and translate between them behind a closed door. The previous lesson walked the GPU half of that split; the Mosaic layer chapter (`/l/mosaic`) walks the TPU half. Set them beside each other and the shape is the same twice over.',
            ],
            table: {
              caption: 'the same line, drawn by two vendors',
              cols: ['machine', 'the public layer', 'the private floor', 'who translates'],
              rows: [
                ['NVIDIA GPU', 'PTX, a virtual ISA you can hand-write', 'SASS, respecified per architecture', 'ptxas at build time, or the driver at load time'],
                ['Google TPU', 'Mosaic, an MLIR dialect any kernel will print', 'LLO, closed inside libtpu', 'the TPU backend, ahead of time'],
              ],
            },
          },
        ],
        readings: [
          {
            label: 'A domain-specific supercomputer for training deep neural networks',
            url: 'https://dl.acm.org/doi/10.1145/3360307',
            note: 'the eight VLIW slots and the sequencer, from the people who drew the floorplan',
          },
          {
            label: 'In-datacenter performance analysis of a tensor processing unit',
            url: 'https://arxiv.org/abs/1704.04760',
            note: 'the determinism argument, made in the abstract and defended for fifteen pages',
          },
          {
            label: 'NVIDIA · the SIMT execution model',
            url: 'https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html',
            note: 'warps, divergence, and the weaving etymology, first-party',
          },
          {
            label: 'Agner Fog · the microarchitecture of Intel, AMD and VIA CPUs',
            url: 'https://www.agner.org/optimize/microarchitecture.pdf',
            note: 'the out-of-order half in more detail than any vendor manual offers',
          },
        ],
        check: [
          {
            q: 'Out-of-order hardware and a VLIW compiler are hunting the same thing. What does the hardware know that the compiler cannot, and why does a blocked matmul erase that advantage?',
            a: 'The hardware knows run-time facts: whether a load hit cache, which way a data-dependent branch went, whether pointers aliased. A blocked matmul has constant trip counts, affine addresses, no data-dependent branches, and a software-managed scratchpad instead of a cache, so every one of those facts is already known at compile time and the window has nothing left to discover.',
          },
          {
            q: 'Why are SIMD and SIMT not two more answers to the question this lesson asks?',
            a: 'They answer width, not scheduling: how much data one instruction covers. SIMD says nothing at all about who scheduled it. SIMT answers width and then adds a separate scheduling answer, since a warp scheduler picks the next resident warp at run time.',
          },
          {
            q: 'PTX is to SASS what Mosaic is to LLO. What is the shared reason both vendors draw the line in that place?',
            a: 'The lower layer carries the schedule and the per-generation latencies, so it has to be free to change every generation. Publishing a stable layer above it and keeping the translation private is what lets the floor move without breaking anything compiled against the contract.',
          },
        ],
        work: [
          { id: 'sort', label: 'take four instructions from the dumps above and label each with the axis it belongs to' },
          { id: 'predicate', label: 'find every @!P0 in the divergence dump and say which threads of a mixed warp execute it', href: '#one-instruction-many-threads' },
          { id: 'check', label: 'answer the three checks without opening them', href: '#check' },
        ],
      },
      {
        id: "scale-up",
        num: 7,
        title: "The scale-up domain",
        lede: "NVIDIA prints 900 GB/s for an H100's NVLink and the scaling book prints 450 for the same eighteen links. Both are correct. Sorting out why is what lets you read every other number in the rack.",
        goal: "Given any NVLink or NVL72 bandwidth figure, say which directions and which endpoints it sums over, and say whether the traffic you care about stays inside one address space or leaves it.",
        sections: [
          {
            h: "a load that lands on another chip",
            ps: [
              "Two GPUs in the same node are wired together by NVLink, and that wiring buys something a network does not. One GPU can issue a request against memory that physically sits on the other chip, and the request is routed there by GPU physical address, because inside the wiring all the GPUs share one address space. Nothing is packed into a message, addressed to a peer, and unpacked at the far end. The remote memory is memory.",
              "That property is what **scale-up** names. You make the machine bigger by making one memory system bigger, as far as the fabric reaches, and code written against it reads like code written against a single very large GPU. **Scale-out** is the other move: more machines, each with its own memory, talking in messages. Nearly every confusing bandwidth figure in an NVIDIA rack becomes readable once you ask which of the two it is describing.",
              "The GPU fabric lesson (`/l/ici/gpu-fabric`) counted the links and switches inside a node and set the per-GPU numbers against measured collectives. This lesson asks a different question about the same wires: how far the shared address space reaches, what the published figures are actually summing, and what changes in your program at the edge.",
            ],
          },
          {
            h: "where the address space stops",
            ps: [
              "Eight GPUs sharing one address space is a manageable thing to build. Two hundred and fifty six of them across many chassis is a different problem, because any GPU could then reach any address on any other with no isolation between tenants, so one job's stray write lands in another job's memory. Hopper's answer was to keep the wires and change the semantics.",
              "The whitepaper states the split plainly. Regular NVLink is the case where \"all GPUs share a common address space and requests are routed directly using GPU physical addresses\"; NVLink Network \"introduces a new Network Address Space supported by new address translation hardware in H100 to isolate all GPUs' address spaces from one another and from the network address space\". The wires below are still NVLink. What sits on them is no longer one memory system.",
              "The consequence lands in your program, and NVIDIA names the comparison itself: because those endpoints do not share a common memory address space, connections \"are not automatically established across the entire system\", and instead, \"similar to other networking interfaces such as InfiniBand, the user software should explicitly establish connections between endpoints as needed\". So the place where InfiniBand takes over is not first a bandwidth cliff. It is the point where a peer stops being an address and starts being an endpoint you connect to, and NVLink Network has already crossed that line while still running on NVLink cable.",
              ">> Inside the domain a peer is an address. Outside it, a peer is a connection.",
              "Hopper put 256 GPUs behind that scheme. The Blackwell brief says the fifth generation \"can scale up to 576 GPUs\". Those are the ceilings on how far one vendor-supported NVLink fabric stretches, not on how many GPUs share a single address space, which is the smaller number set by the node or rack you actually bought.",
            ],
          },
          {
            h: "eighteen links, counted two ways",
            ps: [
              "Three generations of link are three different decisions, and the wire count tells you which. A100's NVLink 3 used four differential pairs in each direction to make one link carrying 25 GB/s each way, and put 12 links on the chip. H100's NVLink 4 got the same 25 GB/s each way out of two pairs, so the rate per wire doubled, and NVIDIA spent the saved wires on more links: 18 instead of 12. Blackwell's NVLink 5 keeps both the two pairs and the 18 links, and doubles the link itself to `50 GB/s` in each direction.",
              "Now the totals. NVIDIA writes that H100 \"includes 18 fourth-generation NVLink links to provide 900 GB/sec total bandwidth\", and 18 links at 25 GB/s each way comes to 450 in one direction, `900` with both summed. Blackwell's brief spells the convention out rather than leaving it to arithmetic: \"1.8 TB/sec total bandwidth, 900 GB/sec in each direction\". Every NVIDIA total for NVLink is both directions added together.",
              "The scaling book counts the same eighteen links one direction at a time and prints `450 GB/s` for an H100, which is where the GPU fabric lesson got its per-GPU figure. Neither source is wrong and neither is being sloppy. Before you compare two NVLink numbers from two authors, divide or multiply by two until they are describing the same directions.",
              "NVIDIA's own ratios confirm they never switch conventions mid-document. Fourth-generation NVLink is \"7x the bandwidth of PCIe Gen 5\" and fifth-generation is \"over 14X\", which puts the PCIe Gen5 baseline at about `129 GB/s` under both, and that is the x16 figure with both directions counted. Same convention, twice, two generations apart.",
            ],
            table: {
              caption: "three NVLink generations, from the H100 whitepaper p.47 and the Blackwell brief p.8",
              cols: ["generation", "GPU", "links / GPU", "pairs per direction", "per link, each direction", "NVIDIA's stated total"],
              rows: [
                ["NVLink 3", "A100", "12", "4", "25 GB/s", "600 GB/s"],
                ["NVLink 4", "H100", "18", "2", "25 GB/s", "900 GB/s"],
                ["NVLink 5", "B200", "18", "2", "50 GB/s", "1.8 TB/s"],
              ],
            },
          },
          {
            h: "what 130 TB/s counts",
            ps: [
              "The GB200 NVL72 headline is that its NVLink Switch \"enables 130TB/s GPU bandwidth in one 72 GPU NVLink domain\". Do the multiplication before you do anything else with it. Seventy-two Blackwell GPUs at `1.8 TB/s` each is `129.6 TB/s`, so the figure is the sum of every GPU's own link capacity, both directions, and it says nothing at all about the switch tier those links plug into.",
              "That makes it the wrong number for the question people usually ask of it. Bisection bandwidth is what crosses a cut, and cutting a 72-GPU domain leaves 36 GPUs on each side, so at most half the endpoints can be pushing across it in a given direction. Under the same both-directions accounting that produced 130, a 36-against-36 cut tops out at `64.8 TB/s`, and only if every GPU on both sides sent exclusively across the cut. An all-GPU sum can never be a bisection figure, for any fabric, because half the endpoints are always on the wrong side.",
              "The real ceiling could be lower still, and the public record will not tell you. NVIDIA publishes no NVL72 switch-tier capacity: the per-chip port count and the number of NVSwitch chips in the rack appear in secondary reporting but were not found in an NVIDIA document. Treat `64.8 TB/s` as an upper bound you derived, not as a spec.",
              "Compare a figure NVIDIA does construct carefully. For the 256-GPU Hopper NVLink Switch System, the whitepaper says each node \"exposes a 2:1 tapered level of all the NVLink bandwidth of the GPUs in the node\" and the connected nodes are \"capable of delivering 57.6 TBs of all-to-all bandwidth\". Work it through: 8 GPUs at 900 GB/s is 7.2 TB/s per node, the taper halves that to 3.6, and 32 nodes give 115.2 TB/s. The published 57.6 is exactly half of that, which is the one-direction reading. NVIDIA never states which convention the 57.6 uses. The arithmetic closing to the digit is the evidence.",
              "So the same family of wires produces figures that look alike and mean four different things, and a fifth kind hides in the SuperPod material: the \"9x increase in bisection bandwidth\" quoted there is a ratio against the previous generation's InfiniBand system, not an absolute figure about anything.",
            ],
            table: {
              caption: "five published NVIDIA figures for the same family of wires, and what each one counts",
              cols: ["figure", "NVIDIA's words", "what it sums", "scope"],
              rows: [
                ["900 GB/s", '"18 fourth-generation NVLink links to provide 900 GB/sec total bandwidth"', "both directions of 18 links", "one H100"],
                ["1.8 TB/s", '"1.8 TB/sec total bandwidth, 900 GB/sec in each direction"', "both directions of 18 links", "one B200"],
                ["57.6 TB/s", '"capable of delivering 57.6 TBs of all-to-all bandwidth"', "one direction of the 2:1 tapered uplink from 32 nodes", "256 Hopper GPUs"],
                ["130 TB/s", '"enables 130TB/s GPU bandwidth in one 72 GPU NVLink domain"', "both directions of every GPU's links, summed over 72", "one NVL72 rack"],
                ["9x bisection", '"9x increase in bisection bandwidth"', "a ratio against the prior generation's InfiniBand system", "256 Hopper GPUs"],
              ],
            },
          },
          {
            h: "the boundary you shard against",
            ps: [
              "Turn all of that into one decision you make while sharding. The collective that runs several times per layer, the tensor-parallel all-reduce, wants to stay inside the domain, and the domain size is therefore a hard cap on how wide you can make that axis: eight on an HGX node, seventy-two in an NVL72 rack. The collective that runs once per step, the data-parallel gradient reduction, is the one you can afford to send across the boundary.",
              "The cost of crossing is not only the bandwidth drop. Inside, a peer read is an address the hardware routes; outside, it is an endpoint your software connected to first, with its own setup, its own failure modes, and a different piece of the stack responsible for it. Two collectives with identical shapes are different programs depending on which side of that line they land on.",
              "TPUs draw no line of this kind, which is worth one paragraph before you carry the GPU habit across. ICI never offered a shared address space to begin with: every remote access is already an explicit DMA to a named neighbor, at one hop or at twenty, so there is no semantic edge to fall off. The boundary a TPU programmer manages is the slice edge where the data-center network takes over, and it is a bandwidth cliff rather than a change of model. The hop arithmetic that prices it lives in the ICI unit (`/l/ici/tpu-fabric`), and the numbers there are the ones to use.",
              "That is the last of the hardware in this stage. You can read a rack spec sheet the way you now read a chip spec sheet, asking of every figure what it sums over and where it stops being true. Stage 1 picks it up at Pallas fundamentals (`/s/pallas`), where the questions stop being about what the machine is and start being about the schedule you write for it.",
            ],
          },
        ],
        readings: [
          {
            label: "NVIDIA · H100 Tensor Core GPU Architecture whitepaper",
            url: "https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf",
            note: "pp.47-49 carry NVLink 4, NVLink Network, and the tapered Switch System; every Hopper quote in this lesson comes from them",
          },
          {
            label: "NVIDIA · Blackwell Architecture Technical Brief",
            url: "https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief",
            note: "p.8 is the whole fifth-generation NVLink story, including the sentence that states the both-directions convention outright",
          },
          {
            label: "NVIDIA · GB200 NVL72",
            url: "https://www.nvidia.com/en-us/data-center/gb200-nvl72/",
            note: "the product page the 130 TB/s figure is usually quoted from; read it after you have done the 72 x 1.8 multiplication",
          },
          {
            label: "Scaling book · How to think about GPUs",
            url: "https://jax-ml.github.io/scaling-book/gpus/",
            note: "the other counting convention, one direction at a time; this is where the 450 GB/s per H100 comes from",
          },
        ],
        check: [
          {
            q: 'You read 900 GB/s for an H100 in one document and 450 GB/s in another. Which one is wrong?',
            a: "Neither. Eighteen links at 25 GB/s each direction is 450 one way and 900 with both summed; NVIDIA always prints the both-directions total, the scaling book prints one direction.",
          },
          {
            q: 'Why can 130 TB/s not be the NVL72 bisection bandwidth?',
            a: "It is 72 x 1.8 TB/s, the sum of every GPU's own link capacity. Any cut leaves half the GPUs on the far side, so at most 64.8 TB/s can cross under the same accounting, and NVIDIA publishes no switch-tier figure confirming even that.",
          },
          {
            q: 'What changes in your program when a collective leaves the NVLink domain?',
            a: "Inside, GPUs share an address space and a request routes by GPU physical address. Outside, each endpoint has its own address space, so software must establish connections explicitly, the way it does for InfiniBand.",
          },
        ],
        work: [
          { id: 'convention', label: 'reconcile the 450 GB/s from the GPU fabric lesson against NVIDIA\'s 900 in one sentence', href: '/l/ici/gpu-fabric' },
          { id: 'bisect', label: 'bound what can cross a 36-against-36 cut of an NVL72, then say what NVIDIA has not published', href: '#what-130-tb-s-counts' },
        ],
      },
    ],
  },
]
