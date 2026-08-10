// The XLA path: fifteen chapters from the PJRT seam down through the pass
// pipeline and back up the runtime stack (IFRT, multi-controller JAX,
// Pathways). Facts are anchored to the public source tree at a verified
// commit; the pass dump and HLO excerpts are real captures (jax 0.4.38,
// CPU, 2026-07-27) with paths shortened. Mastery is manual work plus
// streak items served by the xla gym floor (docs/plans/xla-mastery-path.md).
import type { WorkItem } from '../mastery'

export interface XlaCode {
  caption: string
  text: string
  lang: 'python' | 'cpp' | 'mlir' | 'text'
}
export interface XlaSection {
  h: string
  /** Paragraphs; a string starting '>> ' renders as a pull quote. */
  ps: string[]
  code?: XlaCode
}
export interface XlaReading {
  label: string
  url: string
  note: string
}
export interface XlaChapter {
  id: string
  num: number
  part: 'i' | 'ii'
  title: string
  lede: string
  goal: string
  sections: XlaSection[]
  readings: XlaReading[]
  /** A path diagram pinned after the section at this index. */
  diagram?: { id: string; after: number }
}

export const XLA_CHAPTERS: XlaChapter[] = [
  {
    "id": "pjrt",
    "num": 1,
    "part": "i",
    "title": "PJRT",
    "lede": "Every frontend that talks to XLA, JAX included, crosses the exact same runtime seam to get there.",
    "goal": "Given a single jit-and-run call, name every PJRT object it passes through, in order, and explain how a hardware vendor joins the same path without touching XLA's source.",
    "sections": [
      {
        "h": "the seam everything crosses",
        "ps": [
          "JAX, TensorFlow, and PyTorch/XLA are three different frontends with three different programming models, and none of them talk to XLA's optimizer directly. Each one calls into PJRT instead, a runtime API that sits between whatever language or framework produced a program and the compiler that turns it into a running executable. Learn this boundary first and the rest of the compiler stops feeling like one undifferentiated block: everything below PJRT is XLA's business, and everything above it is the frontend's.",
          "That is also why this chapter comes first here. The JAX track ended its own path right above this line, at the point where a lowered function became a StableHLO string ready to hand off. This track picks up exactly where that one stopped, because PJRT is the boundary every later chapter keeps returning to: it is where a device gets named, where a compiled program gets loaded, and where a result actually comes back.",
          ">> Every framework speaks to XLA through the same door."
        ]
      },
      {
        "h": "three objects, three jobs",
        "ps": [
          "Start with the question of who is in charge. Something has to hold the device list, know the topology wiring those devices together, and take your StableHLO when you ask for it to become runnable. That something is PjRtClient. A process typically keeps one per backend, and every other PJRT object you will ever touch traces back to one.",
          "The memory side is smaller than you might expect. A PjRtBuffer is memory on one device, and nothing more; you get one by handing the client a host array, or by running an executable and catching what falls out. One device, exactly. Nothing at this layer spans two chips, which is why an array sharded eight ways shows up here as eight separate buffers, each knowing only its own device.",
          "And the thing you actually run arrives already spoken for: a PjRtLoadedExecutable comes back from Compile bound to its devices. Its Execute wants buffers in a nested list, one inner list per partition, and hands results back in the same shape. So the division of labor comes out clean. The client owns devices and compilation, buffers are per-device memory, executables run, and almost everything else in PJRT is plumbing between those three."
        ]
      },
      {
        "h": "two ways to reach it",
        "ps": [
          "There are two ways a frontend actually calls into PjRtClient. The first is in-process C++: PjRtStreamExecutorClient links directly into the same binary as the frontend and gets called through ordinary virtual methods. This is the simpler path, and it is how XLA talked to its callers for years.",
          "The second is the C ABI, a plain struct of function pointers defined once and versioned carefully. A backend that wants to ship without merging a single line into XLA's own source builds a shared library exporting that struct, and the frontend loads it at runtime rather than linking it at build time. This is not a hypothetical: Intel's oneAPI backend, Apple's Metal backend, and AWS Trainium all ship as exactly this kind of plugin.",
          "It is easy to assume a new accelerator has to get its support baked into XLA's own repository before anything works. The C ABI exists specifically so that is false: the struct is the entire contract, and a vendor who honors it never needs XLA's build system, its review queue, or even its source code."
        ]
      },
      {
        "h": "what actually crosses the wire",
        "ps": [
          "Compile takes a StableHLO module in. Execute takes buffers in and returns buffers out. That is the entire data-plane contract PJRT enforces, and it is worth sitting with what it implies about sharded arrays: one logical array spread across eight devices is not one PjRtBuffer with a fancy shape. It is eight separate PjRtBuffers, one per device, and PJRT itself has no notion that they belong to the same array at all. Chapter 11 is where a layer above PJRT gives that array a single identity again.",
          "On a real machine the objects are easy to see without reading a line of C++. Ask a device for its platform, its kind, and the type of the client underneath it, and PJRT is right there, one attribute access away."
        ],
        "code": {
          "caption": "the PJRT client, one attribute access away (verified, jax 0.4.38, CPU)",
          "text": "import jax\n\ndev = jax.devices()[0]\nprint(dev.platform, dev.device_kind)   # cpu cpu  (this machine)\nprint(type(dev.client).__name__)       # Client: the PJRT client underneath",
          "lang": "python"
        }
      }
    ],
    "readings": [
      {
        "label": "PJRT \u00b7 openxla.org",
        "url": "https://openxla.org/xla/pjrt",
        "note": "what PJRT is for, from the source"
      },
      {
        "label": "PJRT integration guide",
        "url": "https://openxla.org/xla/pjrt/pjrt_integration",
        "note": "how a hardware vendor plugs in"
      },
      {
        "label": "the C ABI, verbatim",
        "url": "https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h",
        "note": "the struct-of-function-pointers itself"
      }
    ]
  },
  {
    "id": "ingestion",
    "num": 2,
    "part": "i",
    "title": "Ingestion",
    "lede": "What jit hands XLA and what XLA actually optimizes are two different IRs, and confusing them is the easiest way to misread a dump.",
    "goal": "Given any HLO or StableHLO dump, identify which IR it is and explain, in one sentence, why XLA keeps them separate instead of optimizing StableHLO directly.",
    "sections": [
      {
        "h": "one program, two representations",
        "ps": [
          "Every compile starts the same way: a frontend emits StableHLO, hands it to PJRT, and PJRT converts it into something else before a single optimization pass runs. StableHLO is a versioned, portable dialect of MLIR, vendored inside XLA's tree so every frontend can rely on the exact same wire format regardless of which compiler version is on the other end. It is designed to be read, written, and serialized across process and even version boundaries.",
          "That conversion is not cosmetic. The function doing it, ConvertStablehloToHlo (xla/hlo/translate/stablehlo.h), with the PJRT-side wrapper sitting at xla/pjrt/mlir_to_hlo.cc, turns a StableHLO module into HLO, XLA's internal representation, and from that point on StableHLO is gone. Nothing downstream of ingestion ever looks at it again.",
          ">> StableHLO is the wire; HLO is the workbench."
        ]
      },
      {
        "h": "HLO is not MLIR",
        "ps": [
          "The natural assumption, especially if you have spent any time in the MLIR ecosystem, is that HLO must be another MLIR dialect sitting one step below StableHLO. It is not. HLO is a hand-rolled, mutable C++ graph IR (xla/hlo/ir/hlo_module.h, xla/hlo/ir/hlo_instruction.h) that predates MLIR entirely, with its own class hierarchy, its own mutation APIs, and none of MLIR's operation or attribute machinery underneath it.",
          "There is a third IR in the picture that makes the confusion worse: MHLO, XLA's legacy MLIR dialect, still lives in the tree at xla/mlir_hlo/. PJRT does not speak it. Round-trip utilities between MHLO and HLO exist purely for older tooling that has not migrated yet; on the path this chapter is teaching, from a jit call to a running executable, MHLO never appears."
        ]
      },
      {
        "h": "why keep two IRs at all",
        "ps": [
          "The layering earns its keep rather than being an accident of history. StableHLO's whole job is to be stable: it carries explicit versioning and compatibility guarantees, so a program serialized by one compiler version can still be read by a slightly different one on the other end of a network or a saved model. HLO has no interest in any of that. Its job is to be mutated, by pass after pass, in place, and it is built for exactly that: XLA runs upward of two hundred passes against it, each one free to rewrite the graph directly.",
          "One format optimized for staying the same across versions, one format optimized for changing constantly during a single compile. Trying to make a single IR do both jobs well is what the split avoids."
        ]
      },
      {
        "h": "reading it yourself",
        "ps": [
          "This is what jit actually hands to XLA, captured from a small attention program before any optimization pass has touched it. Read the shape annotations closely: every tensor states its dtype and dimensions inline, every op names its exact operands, and the contraction in the dot_general spells out which axes multiply against which, information Python never made you write down explicitly."
        ],
        "code": {
          "caption": "what jit hands to XLA, jax.jit(attend).lower(x, x, x).as_text() (verified, jax 0.4.38, CPU)",
          "text": "module @jit_attend attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n  func.func public @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> (tensor<64x64xf32> {jax.result_info = \"\"}) {\n    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>\n    %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>\n    %cst = stablehlo.constant dense<6.400000e+01> : tensor<f32>\n    %2 = stablehlo.sqrt %cst : tensor<f32>\n    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>\n    %4 = stablehlo.divide %1, %3 : tensor<64x64xf32>",
          "lang": "mlir"
        }
      }
    ],
    "readings": [
      {
        "label": "the StableHLO spec",
        "url": "https://openxla.org/stablehlo/spec",
        "note": "the wire format's contract"
      },
      {
        "label": "XLA architecture",
        "url": "https://openxla.org/xla/architecture",
        "note": "the official overview this chapter deepens"
      }
    ],
    "diagram": {
      "id": "xla-stack",
      "after": 0
    }
  },
  {
    "id": "hlo",
    "num": 3,
    "part": "i",
    "title": "HLO",
    "lede": "By the time a program reaches HLO, every optimization decision XLA will ever make happens on this one representation and nowhere else.",
    "goal": "Given a dumped HLO computation, read its shape, layout, and metadata fields correctly, and state which downstream analysis depends on which fact.",
    "sections": [
      {
        "h": "the shape of a module",
        "ps": [
          "An HloModule is the top of the tree: it holds one entry computation, the function actually called when the program runs, plus however many other computations get called from inside it. Each HloComputation is in turn a list of HloInstructions connected in something close to SSA form, every value defined exactly once and consumed by name wherever it is used.",
          "Every one of those instructions carries its own shape and its own layout from the moment it is created, and both survive every pass that touches it. Some instructions carry more than a shape: a reduction needs to know how to combine two elements, so it carries an entire sub-computation of its own, a small nested function that XLA calls once per reduction step. Those show up in a dump as separate %region blocks sitting right next to the instruction that owns them.",
          ">> Optimization happens on HLO, nowhere else."
        ]
      },
      {
        "h": "the metadata that survives everything",
        "ps": [
          "Every HLO instruction also carries a metadata field linking it back to the Python that produced it: an op_name describing the operation's place in the traced program, a source_file, a source_line. None of the two hundred passes that run between here and codegen strip this out. It survives fusion, it survives layout assignment, it survives scheduling.",
          "That survival is not incidental. It is the entire reason reading a dump is a viable debugging technique at all. Pick any instruction out of an optimized module, no matter how many passes have rewritten the graph around it, and its metadata still points at the exact line of Python that caused it to exist. Losing that thread anywhere along the pipeline would make every later chapter's dump-reading exercises useless."
        ]
      },
      {
        "h": "reading a real region",
        "ps": [
          "Below is one actual reduction sub-computation from an optimized attention program, captured whole. Two parameters come in, a maximum gets taken, and the metadata on that maximum instruction still names reduce_max, still names attend.py, still names line 12, exactly as if no pass had ever touched the module."
        ],
        "code": {
          "caption": "the reduce_max region, metadata intact (verified, jax 0.4.38, CPU, paths shortened)",
          "text": "%region_0.13 (Arg_0.14: f32[], Arg_1.15: f32[]) -> f32[] {\n  %Arg_0.14 = f32[] parameter(0), metadata={op_name=\"jit(attend)/jit(main)/reduce_max\"}\n  %Arg_1.15 = f32[] parameter(1), metadata={op_name=\"jit(attend)/jit(main)/reduce_max\"}\n  ROOT %maximum.16 = f32[] maximum(f32[] %Arg_0.14, f32[] %Arg_1.15), metadata={op_name=\"jit(attend)/jit(main)/reduce_max\" source_file=\"attend.py\" source_line=12}\n}",
          "lang": "text"
        }
      },
      {
        "h": "two analyses, two different questions",
        "ps": [
          "Everything above sets up the load-bearing point: later passes do not re-derive facts about the graph from scratch every time they need one. Two static analyses sit underneath the whole pass pipeline, and they answer different questions. Dataflow analysis asks what values can reach a given point in the program; fusion depends on this, because deciding whether one instruction can merge into another requires knowing exactly which values flow between them.",
          "Alias analysis asks a narrower and stricter question: which positions in the program are required to share the exact same physical buffer. Buffer assignment depends on this one, because it is the pass that actually hands out memory, and it cannot hand two required-aliased positions two different addresses without producing a wrong program. Fusion cares about value flow; buffer assignment cares about physical sharing. They sound similar and they answer genuinely different questions, which is exactly why XLA keeps them as two separate analyses instead of one."
        ]
      }
    ],
    "readings": [
      {
        "label": "HLO operation semantics",
        "url": "https://openxla.org/xla/operation_semantics",
        "note": "the op-by-op contract"
      },
      {
        "label": "XLA architecture",
        "url": "https://openxla.org/xla/architecture",
        "note": "the official overview, for the wider map"
      }
    ]
  },
  {
    "id": "pipeline",
    "num": 4,
    "part": "i",
    "title": "Pipeline",
    "lede": "Run one program through a real compile and the dump folder holds every decision the compiler made, in order, as dozens of numbered files.",
    "goal": "Given a real xla_dump_to output, read the numbered pass sequence as a story: identify the watershed pass and explain what changes on either side of it.",
    "sections": [
      {
        "h": "two hundred decisions, one pass at a time",
        "ps": [
          "XLA's optimizer is not one monolithic algorithm. It is roughly two hundred separate passes, each one a class implementing HloPassInterface and run inside HloPassPipelines (xla/hlo/pass/hlo_pass_pipeline.h), one after another. Pipelines can contain other pipelines, so the sequence you would see if you unrolled the whole thing is deep, hardware-aware, and different on CPU than on GPU or TPU.",
          "None of that has to stay invisible. XLA can be told to write the module out after every single pass runs, and the resulting files are not a black box: they are the compiler explaining itself, one decision at a time, in the exact order it made them."
        ]
      },
      {
        "h": "asking for the dump",
        "ps": [
          "Two flags do the whole job. --xla_dump_to points at a directory; --xla_dump_hlo_pass_re=.* tells XLA to match every pass name against that regex and dump after each match, which for .* means every pass, no exceptions. Set both through XLA_FLAGS before jax is imported, compile anything, and the directory fills with one file per pass step.",
          "This ran on this machine against the same small attention program from earlier chapters, and it produced forty-two numbered files. Forty-two decisions, on a program with three matmuls and a softmax, is already a useful thing to internalize: production models with far more operations produce dumps that are correspondingly larger, but the same story-reading skill applies at any size."
        ],
        "code": {
          "caption": "the dump driver: two flags, forty-two files (verified, jax 0.4.38, CPU)",
          "text": "import os\nos.environ[\"XLA_FLAGS\"] = \"--xla_dump_to=/tmp/xla-dump --xla_dump_hlo_pass_re=.*\"\n\nimport jax\nimport jax.numpy as jnp\n\ndef attend(q, k, v):\n    s = q @ k.T / jnp.sqrt(jnp.float32(q.shape[-1]))\n    return jax.nn.softmax(s) @ v\n\nx = jnp.ones((64, 64))\njax.jit(attend).lower(x, x, x).compile()   # 42 numbered dump files appear",
          "lang": "python"
        }
      },
      {
        "h": "the pipeline, spelled out in filenames",
        "ps": [
          "The numbered filenames are not arbitrary; read them in order and they tell the actual story of the compile. Early files cluster under a name like simplification, and several of them repeat: algsimp runs, something changes, algsimp runs again, because that stage is wrapped to iterate until two consecutive runs report no change at all. That repetition is a fixed point, not a bug in the dump.",
          "Partway through, the filenames shift to a different family: HLO_passes_through_layout_assignment. That name change marks a real boundary, not just a naming convention. Before it, shapes carry no physical layout at all; after layout-assignment itself runs, every shape in the module has one, and passes on the far side of that line are working with information the ones before it never had.",
          "Near the very end, copy-insertion appears, adding copies wherever the analyses run earlier in the pipeline found interference they could not resolve any other way. Its position in the sequence is not an accident either: it exists precisely because it needs a finished layout to know where interference actually lives."
        ],
        "code": {
          "caption": "a real pipeline, from the numbered dump filenames; jax 0.4.38 CPU, trimmed",
          "text": "0003 simplification.after_pipeline-start.before_algsimp\n0004 simplification.after_algsimp.before_simplify-sorts\n0005 simplification.after_tree_reduction_rewriter.before_zero_sized_hlo_elimination\n0010 HLO_passes_through_layout_assignment.after_transpose-folding.before_cse\n0011 HLO_passes_through_layout_assignment.after_cse.before_cse_barrier_expander\n0012 HLO_passes_through_layout_assignment.after_flatten-call-graph.before_layout-assignment\n0013 HLO_passes_through_layout_assignment.after_layout-assignment.before_sub-byte-size-setter\n0017 HLO_passes_after_layout_assignment.after_fusion.before_simplification_after_layout_assignment\n0019 copy-insertion.after_adding_copies_to_resolve_interference\n0022 HLO_passes_after_layout_assignment.after_copy-insertion.before_dce",
          "lang": "text"
        }
      },
      {
        "h": "the watershed, named",
        "ps": [
          "If this chapter has one fact worth carrying forward, it is that layout assignment is the irreversible line in the whole pipeline. Every pass before it can reason about a module using shape alone; every pass after it has to reason using shape and physical layout together, and nothing later ever goes back to being layout-free. Chapter 6 spends its entire time on what a layout actually is and what layout assignment does to earn that boundary.",
          "The habit worth building here is simpler than any one fact: when a decision in an optimized module looks mysterious, the dump usually already explains it. The file where an instruction first appears, set against the file just before it where the instruction did not yet exist, is the compiler showing its own reasoning, and reading that pair of files is very often faster than guessing.",
          ">> The compiler will show you every decision, if you ask for the dump."
        ]
      },
      {
        "h": "measured: the same program, two backends",
        "ps": [
          "The driver above ran on both a CPU and a Colab TPU v6e, and the dumps disagree about almost everything. CPU produced 42 files for this program; the TPU produced 88. That difference is not a deeper pipeline in any interesting sense, it is a different pipeline, built for different hardware, as the section on hardware-aware passes claimed and the file names now prove.",
          "The names are where it gets specific. The TPU dump carries stages a CPU compile has no reason to own: `hlo_device_type_async_wrapper`, `Before_X64_rewriter`, `X64_elimination`, `Phase_1_pre_layout_assignment_passes`, and targets like `add-random-host-offloading` and `tpu-embedding-thread-annotator`. Read that list next to chapter 9 and the shape of the backend shows through: host offloading, embedding hardware, and a 64-bit rewrite the TPU wants handled before layout assignment ever runs.",
          "One practical detail costs an afternoon if you meet it cold. TPU dump files insert a build id between the module name and the step number, as in `module_0007.jit_attend.cl_948136882.0000.hlo_device_type_async_wrapper`, while CPU files go straight from the module to the step. A pattern written against one backend silently matches nothing on the other, and a diff of nothing against something looks exactly like a real answer. Print the filenames before you parse them."
        ]
      }
    ],
    "readings": [
      {
        "label": "XLA tools",
        "url": "https://openxla.org/xla/tools",
        "note": "the dump flags and the tooling around them"
      },
      {
        "label": "XLA architecture",
        "url": "https://openxla.org/xla/architecture",
        "note": "the official overview, for the wider map"
      }
    ],
    "diagram": {
      "id": "xla-pipeline",
      "after": 2
    }
  },
  {
    "id": "fusion",
    "num": 5,
    "part": "i",
    "title": "Fusion",
    "lede": "XLA has exactly one lever on memory traffic, and it never touches the math.",
    "goal": "Given an HLO dump, find where XLA fused two operations, where it refused, and say in one sentence why the refusal was correct.",
    "sections": [
      {
        "h": "one loop instead of two",
        "ps": [
          "Picture two operations chained together, one feeding the next. Run them unfused and the first op writes its whole result to memory before the second op is allowed to start reading it back. Merge them into a single kernel instead and the picture changes completely: the first op's output for one element flows straight into the second op's computation on that element, inside one loop, and the intermediate value never leaves fast memory at all. That merge is what XLA calls fusion, and the instruction it produces is a kFusion, one op standing in for what used to be several.",
          "It is easy to picture fusion as the compiler finding a smarter formula, some algebraic shortcut dressed up as an optimization pass. That picture is wrong in a way worth fixing early, because it will mislead you at every later chapter. A fused kernel computes the identical sequence of arithmetic the unfused version computed; nothing about the math is rewritten, simplified, or reordered at the level of results. The only thing fusion changes is whether an intermediate value round-trips through memory or stays resident where the next op can reach it immediately.",
          ">> Fusion is about the intermediates, never the math."
        ]
      },
      {
        "h": "who decides, and by how much",
        "ps": [
          "Two questions gate every fusion decision XLA makes, and the compiler runs the cheaper one first. Fusibility asks a structural question: can these two instructions even be merged, given their opcodes and how their operands connect? Profitability asks the harder one: would merging them actually help? XLA answers profitability with a cost model, backed by an is_fusible callback each backend gets to customize for its own hardware, since what counts as a win on a CPU and what counts as a win on a GPU are not the same arithmetic.",
          "On GPU that cost model runs as a priority queue: every candidate producer gets ranked by the time it would save if fused, and the pass fuses in that order, highest expected benefit first. Nothing here is exhaustive search. It is a greedy ordering over real cost estimates, and you can watch it work directly: the fusion x-ray at /gym/xla#fusion holds three real before-and-after pairs pulled from a TPU, each one a case where this exact ranking made a call you can check against your own intuition. The kernel path already measured what a fusion like this is worth in practice, a fused softmax beating the unfused chain outright, recorded with full provenance at /bench."
        ]
      },
      {
        "h": "the wall fusion cannot cross",
        "ps": [
          "Every fusion pass, however aggressive, is bounded by one hard rule: it can only merge instructions that already sit on a dataflow edge in the graph it was handed. It never invents a new algorithm and it never restructures the computation into a shape that does less work. If two values are related in your head but not connected by a direct producer-consumer edge in the HLO, fusion has nothing to offer them.",
          "This is the wall the kernel path spends an entire stage teaching you to see, at /l/xla: naive softmax needs the row maximum before it can compute a single exponential, so the score matrix gets written to memory, read back, and only then reduced. No pass in this pipeline can fold that spill away, because folding it away would mean changing the algorithm from multi-pass to streaming, and fusion does not do algorithms. It does edges."
        ]
      },
      {
        "h": "one fusion, caught in the act",
        "ps": [
          "The dump below is real, pulled from the optimized module for the attention program this path has been tracking. XLA looked at a broadcast feeding directly into a divide and merged the two into one %fused_computation: the broadcast materializes its input into the divide's shape, and the divide consumes it, all inside a single computation with its own parameters and one ROOT instruction. Nothing outside this fused block ever sees the broadcast's output on its own."
        ],
        "code": {
          "caption": "the compiler's own fusion, from the optimized module",
          "text": "%fused_computation (param_0: f32[64,64], param_1.1: f32[64]) -> f32[64,64] {\n  %param_0 = f32[64,64]{1,0} parameter(0)\n  %param_1.1 = f32[64]{0} parameter(1)\n  %broadcast.3 = f32[64,64]{1,0} broadcast(f32[64]{0} %param_1.1), dimensions={0}, metadata={op_name=\"jit(attend)/jit(main)/div\"}\n  ROOT %divide.0 = f32[64,64]{1,0} divide(f32[64,64]{1,0} %param_0, f32[64,64]{1,0} %broadcast.3), metadata={op_name=\"jit(attend)/jit(main)/div\" source_file=\"attend.py\" source_line=12}\n}",
          "lang": "text"
        }
      },
      {
        "h": "measured, then retracted: a flag that did nothing",
        "ps": [
          "This section used to publish a disagreement. LAB\u00b7X2 timed this chapter's attention on a Colab TPU v6e at `2048` by `128` in bf16, medians of twenty runs in isolated processes, and reported `187.5` microseconds with the pipeline untouched against `139.3` microseconds with `--xla_disable_hlo_passes=fusion`, a ratio of `0.74x`. Disabling fusion appeared to make the program faster, which is the opposite of everything above, so it went up here as an open question.",
          "The question is now closed, and not in the direction anyone expected. Compiling both versions and reading what each one emitted returned the same module twice: 66 instructions either way, seven fusions either way, no custom calls in either, and the same fusion at the root. The flag changed nothing. XLA accepted a pass name, found nothing registered under it in this backend's pipeline, and carried on without a word. Both timings measured the same program, so the ratio was never about fusion at all.",
          ">> A flag that names a pass the backend does not have is not an experiment. It is a typo the compiler agreed to.",
          "Pass names are backend-specific, which chapter 4's dump shows plainly: the TPU pipeline files its work under names like `Phase_1_pre_layout_assignment_passes` and `X64_elimination` that a CPU compile never mentions. A name lifted from one backend's vocabulary can be silently absent from another's, and nothing in the output distinguishes a disabled pass from a misspelled one. The habit that costs ten seconds and would have caught this before it was published: compile both ways, diff the modules, and only time them once you have proof the flag moved something.",
          "The follow-up run answered the measurement half. Six identical processes, no flags anywhere, timing the same program on the same chip came back `170.2`, `147.6`, `147.6`, `182.7`, `166.5`, and `200.4` microseconds. The first process sat within two percent of the median of the rest, so position was never the issue, but the spread across identical work is roughly `35%`. A single pair of process timings on this hardware cannot resolve a `26%` difference, which is all the retracted measurement ever was.",
          ">> If the same work twice spans 35 percent, a single pair of numbers is not a measurement.",
          "The vocabulary half turned up something the flag hunt had missed. Grepping the dumped pass names for fusion returns not one stage but five: `tpu_fusion`, `main_fusion`, `tpu_multi_output_fusion`, `fusion`, and `async-collective-fusion`. Disabling each in turn and diffing the compiled module settles which of them do the work here: `tpu_fusion` and `tpu_multi_output_fusion` change it, while `fusion`, `main_fusion`, and `async-collective-fusion` leave it untouched. So the backend does register a pass called exactly `fusion`, and disabling it does nothing for this program, which is why the original flag looked accepted and behaved like a no-op at the same time.",
          "The experiment then ran twice, with `tpu_fusion` confirmed to change the module first. Six alternating pairs, then six more: `191.9` against `168.9`, `153.4` against `164.9`, and so on through twelve. Fusion came out ahead in ten of the twelve, and the marginal ranges still overlap in both runs, which is what the lab reported at the time.",
          "The overlap test was the wrong test, and this section owes that correction as much as it owed the first one. The design pairs the two configurations and alternates them precisely so that drift lands on both sides of a pair, which means the paired difference is the measurement and the marginal range is not. Read that way the twelve pairs give a median difference of `16.7` microseconds in fusion's favour, ten of twelve in the same direction, which a sign test puts at about `p = 0.02`. Comparing the outer ranges threw the pairing away and hid a result that was sitting in the data.",
          ">> Pair the runs, then compare the pairs. Comparing the envelopes discards the design.",
          "So the answer, finally: on a v6e, disabling `tpu_fusion` for this attention program costs somewhere around ten percent, and the mechanism this chapter argues for holds. It took a retraction, a flag that did nothing, a stage hunt, twelve paired runs, and one more correction to the analysis to say that with a straight face, which is roughly the honest price of a performance claim."
        ]
      }
    ],
    "readings": [
      {
        "label": "XLA architecture",
        "url": "https://openxla.org/xla/architecture",
        "note": "the official overview; read it now against a fusion decision instead of in the abstract"
      },
      {
        "label": "HLO operation semantics",
        "url": "https://openxla.org/xla/operation_semantics",
        "note": "the op-by-op contract fusibility checks are built on"
      }
    ]
  },
  {
    "id": "layout-memory",
    "num": 6,
    "part": "i",
    "title": "Layout and memory",
    "lede": "Two arrays with the same shape can still disagree about where every number actually sits in memory.",
    "goal": "Given a shape with its layout notation, predict the physical stride of each dimension, and trace why XLA inserted, or skipped, a copy at a given point in the module.",
    "sections": [
      {
        "h": "the promise shapes alone do not make",
        "ps": [
          "Every shape carries a layout, and until you have seen one written out, it is easy to assume shape settles memory order by itself, the way it quietly does in plain NumPy. XLA makes no such assumption. Look back at any dump from earlier chapters and the notation was already there, sitting right after the shape: f32[64,64]{1,0}. That trailing {1,0} is the layout, a minor-to-major ordering of dimensions, and it is a decision the compiler makes explicitly and separately from the shape.",
          "Read {1,0} as an instruction: dimension 1 is minor, meaning it varies fastest in memory, and dimension 0 is major. For a 64x64 array of 4-byte floats laid out this way, stepping to the next column costs 4 bytes and stepping to the next row costs 256 bytes, the ordinary row-major layout most code assumes by habit. Flip the order to {0,1} and the strides flip with it: rows become contiguous, columns become the expensive direction. Same shape, same values, a completely different walk through memory."
        ]
      },
      {
        "h": "a constraint that has to travel",
        "ps": [
          "Some instructions arrive with an opinion about layout already fixed. A dot or a convolution wants specific operand layouts to hit its fast path; an entry parameter's layout is whatever the caller promised. LayoutAssignment (`xla/service/layout_assignment.h`) is the pass that starts from these fixed points and propagates outward, deciding a layout for every remaining shape in the module by following the dataflow from instructions that demand one toward instructions that do not care.",
          ">> A layout is a promise the whole module must keep.",
          "Propagation does not always land two neighbors on the same page. When an operand's chosen layout does not match what its consumer needs, XLA cannot just reinterpret the bytes and hope; a layout is a promise the whole module must keep, so the only honest fix is to insert a transpose or a copy that actually rearranges the data into the layout the next instruction was promised. Every copy you see in a dump is a broken agreement being repaired on the spot."
        ]
      },
      {
        "h": "buffers do not care how you got here",
        "ps": [
          "Layout assignment finishes before buffer assignment ever starts, and that ordering is load-bearing: once every shape has a fixed layout, BufferAssignment (`xla/service/buffer_assignment.h`) can treat each value as a fixed number of bytes and ask where those bytes should live on the device, using liveness and aliasing analysis to decide which values can share a physical allocation because they are never alive at the same time.",
          "This is where donation from the JAX path finally cashes out. Marking an argument for donation with donate_argnums does not do anything to memory by itself; it tells the compiler that once this function runs, nobody upstream still needs that input. Buffer assignment is what turns that promise into an actual input-output alias, letting the output reuse the exact allocation the donated input already occupied instead of paying for a second one."
        ]
      },
      {
        "h": "the pipeline exists because one pass cannot fix another pass's leftovers",
        "ps": [
          "Buffer assignment's liveness analysis is not perfect, and it does not try to be. Some interference between values only becomes visible after allocations are chosen, and resolving that interference is a separate job: copy-insertion, the pass whose handiwork you already saw in chapter 4's dump. It exists precisely because buffer assignment cannot fix the conflicts it creates on its own.",
          "That is the honest reason this whole stretch of the compiler runs as an ordered pipeline rather than one pass doing everything at once. Layout has to be fixed before buffer assignment can count bytes meaningfully; buffer assignment's allocation choices have to exist before copy-insertion knows which conflicts still need patching. Each pass leaves work for the next one on purpose, and the order is not arbitrary."
        ]
      }
    ],
    "readings": [
      {
        "label": "XLA architecture",
        "url": "https://openxla.org/xla/architecture",
        "note": "the official overview; read the layout and buffer sections against this chapter's stride example"
      },
      {
        "label": "XLA tools",
        "url": "https://openxla.org/xla/tools",
        "note": "the dump flags that let you watch copy-insertion patch a real conflict"
      }
    ]
  },
  {
    "id": "spmd",
    "num": 7,
    "part": "i",
    "title": "SPMD",
    "lede": "The mesh and the PartitionSpec you wrote from the JAX path did not vanish; they became the input to a pass that rewrites the whole module around them.",
    "goal": "Given a sharded HLO module, name where sharding propagation must have made a choice, and predict which collective the partitioner inserted to resolve it.",
    "sections": [
      {
        "h": "the pass behind the surface you already used",
        "ps": [
          "The JAX path taught the surface of this from above, at /jax/sharding: a Mesh, a PartitionSpec, a NamedSharding, and a compiler that quietly inserted whatever collectives your program's math required. This chapter is the mechanism underneath that surface. SpmdPartitioner (`xla/service/spmd/spmd_partitioner.h:274`) is an HloModulePass, and its job is exactly what it sounds like: take a module written for one logical, global-shape array and rewrite it into the module every device actually runs, one physically local shape per participant.",
          "Every all-gather, reduce-scatter, and all-reduce you never wrote by hand in the JAX path came from this pass deciding, at a specific point in the dataflow, that a sharding boundary demanded data movement. Nothing about your Python asked for a collective explicitly. The collective is what SpmdPartitioner inserts when two adjacent operations disagree about how a value should be split."
        ]
      },
      {
        "h": "sharding has to be decided before it can be rewritten",
        "ps": [
          "Rewriting the module is the second half of this pass's job; propagation is the first, and it runs to completion before any rewriting starts. Sharding annotations you gave explicitly flow forward and backward through the graph until nothing changes anymore, a fixpoint, and every instruction you did not annotate simply inherits whatever its neighbors settled on.",
          "It is easy to assume every instruction needs its own explicit annotation, and that assumption makes the mental model far more work than it needs to be. Most of a real module carries no sharding you wrote at all; propagation supplies it. The cases worth watching are the disagreements, where two neighbors would prefer conflicting shardings for the same value, and those get resolved by priority rules that favor the more informative or more constrained side."
        ]
      },
      {
        "h": "one program, every device",
        "ps": [
          "GSPMD, the design this whole mechanism implements, gets its name from the acronym: generalized single program, multiple data. Every device that participates ends up running the identical compiled binary; what differs from device to device is never the code, only the shard of data that binary happens to be looking at.",
          ">> One program, partitioned, is still one program.",
          "That single sentence is the whole contract, and it is why the partitioner has to be this careful: a collective it inserts on one device's copy of the program has to be matched by the exact same collective, at the exact same logical point, on every other device's copy. Get that wrong and you are not looking at a slow program, you are looking at chapter 8's failure mode."
        ]
      },
      {
        "h": "the annotation is moving, the target is not",
        "ps": [
          "A newer dialect, sdy, short for Shardy, is changing where the sharding annotation actually lives: in MLIR, attached to StableHLO, before the module ever becomes HLO at all. That is an earlier altitude than the mechanism this chapter describes, not a different mechanism. Whether the annotation started life in MLIR or was inherited by an HLO-level pass, the destination is the same partitioned module SpmdPartitioner has always produced."
        ]
      }
    ],
    "readings": [
      {
        "label": "GSPMD paper",
        "url": "https://arxiv.org/abs/2105.04663",
        "note": "the design, from its authors"
      },
      {
        "label": "XLA architecture",
        "url": "https://openxla.org/xla/architecture",
        "note": "the official overview; read the partitioning section against this chapter's mechanism"
      }
    ]
  },
  {
    "id": "collectives",
    "num": 8,
    "part": "i",
    "title": "Collectives",
    "lede": "A collective instruction looks like one op in the dump; underneath, every device in the clique has to show up and call it in the same order or the whole fleet stalls.",
    "goal": "Given a distributed HLO dump, state the deadlock condition a given collective ordering would trigger, and name what a scheduling pass is allowed to run in an async collective's gap.",
    "sections": [
      {
        "h": "a contract, not a broadcast",
        "ps": [
          "An all-reduce, an all-gather, a reduce-scatter, an all-to-all, a collective-permute: each of these shows up in an HLO dump as a single instruction, and that appearance is misleading in a specific way worth correcting early. No single device is in charge of a collective. Every participant in what the runtime calls a clique has to independently issue the matching call, and the operation only completes once all of them have.",
          "That is the sense in which a collective is a contract rather than a command. Nobody sends an all-reduce to the fleet the way you might send a message to one recipient; every device's own compiled program contains the identical instruction, at the point the partitioner decided a sharding boundary demanded it, and the correctness of the whole thing depends on every copy of that program agreeing to run it.",
          ">> A collective is a contract between every participant."
        ]
      },
      {
        "h": "how the contract gets broken",
        "ps": [
          "Break that agreement and the failure mode is deadlock: one device waiting on a collective that another device never issues, or issues in a different order relative to some other collective it also needed. The kernel path met this exact failure at the lowest possible level, raw semaphores in a Pallas kernel at /s/distributed, a send waiting on a recv that is itself waiting on that same send.",
          "This chapter meets the identical failure two layers higher up, where the ordering constraint is enforced by the compiled schedule rather than by a semaphore you wrote by hand. The mechanism looks different at each layer. The failure is the same failure, because the contract being violated, every participant issues the same collective in the same order, does not change shape depending on which layer you are reading it at."
        ]
      },
      {
        "h": "splitting one collective into two instructions",
        "ps": [
          "A synchronous all-reduce forces compute to stop and wait for the network. XLA has an escape from that: an async pair, all-reduce-start followed later by all-reduce-done, splits one logical collective into an instruction that launches the communication and a second instruction that waits for it to finish.",
          "Everything scheduled between the start and the done runs concurrently with the network transfer, which is the entire point. A scheduling pass decides how much independent compute fits into that gap, stretching it as wide as the dataflow allows, and the wider that gap gets filled, the less the collective costs the step that contains it."
        ]
      },
      {
        "h": "moving data is the same problem as running a collective",
        "ps": [
          "Host offloading, moving a tensor out to host memory across some region of the program and back, sounds unrelated to a collective on its face, but it rests on the identical piece of machinery: an async value that resolves once an operation has actually finished, not once it was merely launched. A collective's start instruction and a host transfer both hand back a token immediately; what differs is only what that token is waiting on underneath.",
          "That shared machinery is what makes both overlap patterns real rather than theoretical. Compute can run ahead of a collective's completion, or ahead of a host transfer's completion, because the program has an honest way to ask later whether the thing it is waiting on is actually done, instead of guessing based on how much time has passed."
        ]
      }
    ],
    "readings": [
      {
        "label": "HLO operation semantics",
        "url": "https://openxla.org/xla/operation_semantics",
        "note": "the collective ops section, where every one of these instructions is specified exactly"
      },
      {
        "label": "XLA architecture",
        "url": "https://openxla.org/xla/architecture",
        "note": "the official overview; read the execution section against this chapter's async pairs"
      }
    ]
  },
  {
    "id": "codegen",
    "num": 9,
    "part": "i",
    "title": "The floors",
    "lede": "One fusion, dumped on three chips, comes back three different kinds of code.",
    "goal": "Given an HLO fusion after optimization, name which backend it lowers to and the artifact each one emits before any kernel actually runs.",
    "sections": [
      {
        "h": "one dispatcher, three backends",
        "ps": [
          "Fusion decides what stays together in fast memory. It says nothing about what a chip actually runs. Once the HLO pipeline finishes, every fusion in the module still has to become real code, and that is the point where XLA stops behaving like one compiler and splits into three.",
          ">> Below fusion, every backend goes its own way.",
          "On CPU, a fusion becomes LLVM IR wrapped in a thunk runtime (`xla/backends/cpu/runtime/thunk.h`): a typed sequence of operation nodes an executor fires in dataflow order on a host thread pool. On GPU, a fusion becomes MLIR, or Triton, or a direct call into cuDNN, replayed as its own thunk sequence (`xla/backends/gpu/runtime/thunk.h`). On TPU, the fusion crosses into libtpu, and libtpu is closed: XLA hands off a program and nothing in this track can read what happens after that hand-off. This is exactly where the kernel path's Mosaic chapter picks up.",
          "Follow any of those routes to the bottom and the artifacts are concrete. The CPU's LLVM IR compiles down to ordinary host machine code. On GPU, every route that claims a fusion converges on the same tail: LLVM IR, or Triton's own pipeline, lowers to PTX, and PTX to a cubin the runtime will load onto the card. So the executable that comes back from Compile is two things traveling together: generated device code, and the thunk sequence, an ordered to-do list whose entries say launch this kernel, call cuBLAS here, run this collective now."
        ]
      },
      {
        "h": "what CPU actually runs",
        "ps": [
          "`Thunk` (`xla/backends/cpu/runtime/thunk.h:70`) is CPU's uniform dispatch unit: an abstract base class whose `Execute` is the only virtual call the runtime makes, and whose `Kind` enum enumerates every concrete flavor a fusion can become, a kernel, a convolution, a dot, a collective, a custom call, and several more. Each thunk declares up front which buffers it reads and writes, and the `ThunkExecutor` reads those declarations to build a dependency graph, then fires whichever thunk has every input ready, on an Eigen thread pool.",
          "For a small enough computation, eight thunks or fewer, or every buffer under 512 bytes, the executor skips the thread pool and runs the sequence straight through. Dataflow scheduling has a real cost, and paying it to schedule a handful of scalar ops would lose more time than it saves."
        ]
      },
      {
        "h": "GPU: four ways to a kernel",
        "ps": [
          "GPU fusion resolves through a single dispatcher, `GetFusionEmitter`, sitting behind a decision tree that runs ahead of it. `HloFusionAnalysis` looks at each fusion first: if a backend config was already stamped `kTritonGemmFusionKind` or `kCuDnnFusionKind` by an earlier pass, the fusion goes straight to Triton or cuDNN. Otherwise the dispatcher inspects the fusion's dominant instruction, its shape decides whether the result is a reduction kernel, a transpose kernel, a scatter, or the generic elementwise loop every unclaimed fusion falls through to.",
          "Triton and cuDNN are not the only way a dot instruction leaves HLO. `GemmRewriter` can rewrite a supported matmul directly into a `__cublas$gemm` or `__cublas$lt$matmul` custom call, no fusion emitter involved at all, and cuBLAS owns the kernel from there. Which path a given dot takes, Triton-GEMM if it is enabled, cuBLAS otherwise, the generic loop emitter if neither wants it, is settled by a fixed priority order before any of this chapter's codegen machinery runs."
        ]
      },
      {
        "h": "StreamExecutor, the seam below PJRT",
        "ps": [
          "Chapter 1 named PJRT the seam between every frontend and XLA. StreamExecutor (`xla/stream_executor/stream_executor.h`) is the seam on the other side: one abstract interface per physical device that owns memory allocation, execution streams, kernel loading, and the BLAS and DNN library handles a backend exposes. GPU thunks call through this interface and nothing lower. `Stream` is the ordered queue thunks enqueue work onto: operations on one stream run in order, operations on different streams can overlap, and a `CudaStream` (or its ROCm counterpart) is the concrete object underneath.",
          "One misreading deserves heading off before it settles in. StreamExecutor is not a compile target, and there is no StreamExecutor IR: nothing in the pipeline lowers to it. Code generation already finished, in the sections above, with device code and a thunk sequence; StreamExecutor is the interface those thunks call when the executable finally runs. Allocate this buffer, load this cubin, enqueue this kernel on that stream. Compilation ends with code. This interface is how that code reaches silicon.",
          "The description a `StreamExecutor` returns for its own device, `GetDeviceDescription()`, is what makes GPU codegen decisions possible at all: threads per warp, shared memory per block, register limits, the exact compute-capability variant. A GPU pass asking whether it can use a particular warp-level trick is really asking this object a question, not guessing from a chip name string."
        ]
      },
      {
        "h": "the escape hatch: FFI",
        "ps": [
          "Custom calls are how a fusion decision gets skipped on purpose. XLA's FFI (`xla/ffi/ffi.h`) is a stable, versioned C ABI that lets a `custom-call` instruction dispatch straight to a handler XLA never compiled: a hand-written kernel, a cuDNN graph, or the pattern-matched kernel the timeline x-ray's online-softmax call reveals in the kernel path's TPU dumps, dispatched by XLA:TPU recognizing that exact program shape.",
          "The call frame a handler receives carries typed buffer descriptors, a sorted attribute list, an execution-stage flag, and a context object for state that needs to survive across calls, all defined by one struct-of-function-pointers vtable. A correct custom call means someone wrote the kernel entirely outside the compiler; the FFI's job is making that handoff safe, not making the kernel unnecessary."
        ]
      }
    ],
    "readings": [
      {
        "label": "Custom calls and the FFI",
        "url": "https://openxla.org/xla/custom_call",
        "note": "the typed FFI's own account of the interface"
      },
      {
        "label": "XLA GPU architecture",
        "url": "https://openxla.org/xla/gpu_architecture",
        "note": "the official GPU pipeline overview"
      }
    ]
  },
  {
    "id": "autotuning",
    "num": 10,
    "part": "i",
    "title": "Measure when models run out",
    "lede": "When XLA cannot decide which kernel wins on paper, it stops guessing and runs a race.",
    "goal": "Given a GPU lowering decision, say whether XLA cost-models it or measures it, and predict what invalidates a cached answer.",
    "sections": [
      {
        "h": "when the model runs out",
        "ps": [
          "Most of what XLA's GPU compiler decides comes from a cost model: count the FLOPs, count the bytes, pick the option the arithmetic favors. A few choices resist that treatment. A GEMM's algorithm, a Triton kernel's tile sizes, a convolution's algorithm: the option space is wide enough, and the candidates close enough in theory, that no cost model built from constants can call the winner reliably.",
          ">> When the cost model runs out, the compiler measures.",
          "For exactly those lowerings, the autotuner (`xla/backends/autotuner/`) compiles every candidate, runs each one on the real device at compile time, and keeps the fastest result that is also correct. The winner gets cached, keyed on the device and a fingerprint of the program, and that cache can be serialized to disk so a second compile of the same program on the same chip skips the race entirely."
        ]
      },
      {
        "h": "how a candidate gets timed",
        "ps": [
          "The autotuner treats identical instructions as one job. Every candidate gets a 128-bit fingerprint from its own printed form, duplicates collapse to a single representative, and whatever configuration wins for that representative gets stamped onto every equivalent instruction afterward. Config resolution then runs a three-tier decision: a cache hit returns immediately, a default-config request takes the first backend willing to answer, and only a genuinely new candidate pays for the full race.",
          "The race compiles every supported backend-and-config pair, in parallel when a thread pool is available, then profiles them one mutex at a time: a single lock covers the whole profiling phase, so compilation can overlap but measurement cannot, because the device state being timed is exactly what a concurrent measurement would corrupt. Outputs get clustered by numerical closeness, trusted backends checked first, and the winning cluster has to contain a value XLA actually trusts before anything gets declared a winner.",
          "Among candidates in the winning cluster, the fastest one usually takes it outright. One narrow exception exists: any config within a small time window of the fastest, two microseconds by default, counts as tied, and the tiebreak goes to whichever of those uses the least scratch memory. On fast hardware that window is tight enough that a slower config rarely steals a win on memory alone."
        ]
      },
      {
        "h": "two sides of measurement",
        "ps": [
          "Autotuning is one half of how XLA knows a kernel is fast; the other half is measuring the same fact after the program actually runs. `compiled.cost_analysis()`, the same call the JAX path meets when it lowers a function ahead of time, is the compiler's own estimate: flops and bytes, computed from the program, no device involved. XProf sits at the opposite end entirely, a trace of what really happened on real hardware, the same instrument the kernel path's timeline reads down to individual bytes moved.",
          "Autotuning sits between the two. It does not estimate the way cost analysis does, and it does not merely observe the way a profiler does after the fact; it runs real candidates on a real device, during compilation, specifically to settle a question neither a static model nor a post-hoc trace can answer alone."
        ]
      },
      {
        "h": "reproducibility has a price",
        "ps": [
          "An autotuned answer is a fact about one chip and one driver, not a fact about the algorithm. The same program compiled on a different chip generation, or even a different driver version on the same chip, can autotune to a different winner. Pinning the autotune cache, shipping it alongside the compiled program instead of recomputing it, is how a production build keeps that variable out of the loop.",
          "Distributed autotuning splits the candidate list across processes and exchanges results through a key-value store, and one deliberate shortcut in that path is worth knowing: the store's key hashes the program's fingerprint and the backend name, not the full compilation config. Two compiles of the same HLO under different debug options can therefore collide in the store and swap results neither run actually produced, a trade XLA accepts on purpose to avoid a slower, less deterministic key."
        ]
      }
    ],
    "readings": [
      {
        "label": "XLA tools",
        "url": "https://openxla.org/xla/tools",
        "note": "the dump flags and the tooling around autotuning"
      },
      {
        "label": "XLA GPU architecture",
        "url": "https://openxla.org/xla/gpu_architecture",
        "note": "where autotuned decisions sit in the GPU pipeline"
      }
    ]
  },
  {
    "id": "ifrt",
    "num": 11,
    "part": "ii",
    "title": "The array moves up",
    "lede": "PJRT hands a caller a pile of per-device buffers. IFRT hands back one array.",
    "goal": "Given a sharded array, describe it correctly at both the PJRT level and the IFRT level, and name what IFRT can do that PJRT cannot.",
    "sections": [
      {
        "h": "one object, not many buffers",
        "ps": [
          "PJRT's unit of data is the per-device buffer: a `PjRtBuffer` holds one shard's worth of memory on one device, and a single logical array spread across eight devices is eight separate `PjRtBuffer` objects, with the framework above keeping track of how they fit together. IFRT (`xla/python/ifrt/`) exists to erase that bookkeeping. `ifrt::Array` (`xla/python/ifrt/array.h:65`) is one object: it carries a `dtype()`, a `shape()`, and a `sharding()` of its own, and an eight-device array is one `ifrt::Array`, not eight of anything.",
          ">> The array moved up the stack; the buffers stayed behind.",
          "`ifrt::Client` is what builds these objects. `MakeArrayFromHostBuffer` turns a plain host buffer into one; `AssembleArrayFromSingleDeviceArrays` does the adjacent job in reverse, taking per-device shards a caller already has and gluing them into a single logical array. Where PJRT asks the framework to track eight buffers and a sharding by hand, IFRT asks for one array that already knows its own sharding."
        ]
      },
      {
        "h": "what the header actually holds",
        "ps": [
          "Four methods carry the whole idea: `dtype()`, `shape()`, `sharding()`, and `layout()`, each a pure virtual with no default implementation. An `Array` cannot be copied or moved either, on purpose: every reference to one is a `tsl::RCReference<Array>`, aliased as `ArrayRef`, a ref-counted handle rather than a value that could get duplicated and quietly drift out of sync with its own shards.",
          "The rest of the interface is about taking that one object apart or putting it back together without touching shard data unnecessarily. `DisassembleIntoSingleDeviceArrays` is the exact inverse of `AssembleArrayFromSingleDeviceArrays`; `FullyReplicatedShard` is a shortcut for the common case where every shard is identical, one shard read instead of all of them. An `ArrayCopySemantics` enum, `kAlwaysCopy`, `kReuseInput`, `kDonateInput`, governs every one of these calls the same way donation governs a PJRT buffer: some operations promise a fresh copy, others let the caller trade ownership of the input away for speed."
        ],
        "code": {
          "caption": "verbatim, trimmed, from xla/python/ifrt/array.h",
          "text": "class Array : public llvm::RTTIExtends<Array, Value> {\n public:\n  Array() = default;\n\n  // Not copyable or movable.\n  Array(const Array&) = delete;\n  Array(Array&&) = delete;\n  Array& operator=(const Array&) = delete;\n  Array& operator=(Array&&) = delete;\n\n  virtual DType dtype() const = 0;\n  virtual const Shape& shape() const = 0;\n  virtual const Sharding& sharding() const = 0;",
          "lang": "cpp"
        }
      },
      {
        "h": "two implementations, one interface",
        "ps": [
          "IFRT is an interface, not a runtime, and more than one thing implements it. The PJRT-backed implementation wraps `PjRtClient` and `PjRtBuffer` objects directly, in the same process as the Python that called it. The line dividing the two implementations is where the runtime lives relative to your process, not how many hosts the job spans: the multi-controller jobs of the next chapter still use this in-process implementation, one client per host, each wrapping that host's local PJRT client. The proxy implementation, under `xla/python/ifrt_proxy/`, exists for when client and runtime do not share a process: the process holding the Python `ifrt::Client` talks over a wire to a server process that owns the actual runtime, wherever that runtime happens to live.",
          "Chapter 13 depends entirely on that split existing. Swapping what sits behind the proxy's server, from an ordinary PJRT-backed runtime to something like Pathways, changes nothing about the JAX code running above the client. The abstraction is not decoration; it is the exact seam a single-controller runtime needs in order to be a drop-in replacement underneath code nobody had to touch."
        ]
      },
      {
        "h": "why JAX needed this",
        "ps": [
          "A `jax.Array` sits on an `ifrt::Array` underneath, and that is the concrete reason a single `jax.Array` can span multiple host processes in JAX's multi-process world, the training run the JAX path meets from above (at /jax/training-run). The sharding lives on the array object itself, not stitched together by the framework from a pile of per-device buffers it has to track by hand.",
          "`Client` exposes operations that only make sense once an array is a first-class object: `RemapArrays` shuffles shards between arrays, `ReshardArrays` changes an array's sharding, `BitcastArrays` reinterprets its bytes, all as metadata operations rather than data movement. None of this replaces PJRT. Every IFRT array still bottoms out in real per-device memory, and the PJRT-backed implementation is proof that IFRT is a layer built on top, not a rewrite underneath. What moved is where the bookkeeping lives: up, onto an object that already knows what it is, instead of a set of buffers a caller has to remember to keep in sync."
        ]
      }
    ],
    "readings": [
      {
        "label": "The IFRT tree",
        "url": "https://github.com/openxla/xla/tree/main/xla/python/ifrt",
        "note": "the interface, headers first"
      },
      {
        "label": "PJRT \u00b7 openxla.org",
        "url": "https://openxla.org/xla/pjrt",
        "note": "the layer it abstracts over"
      }
    ],
    "diagram": {
      "id": "xla-runtime-ladder",
      "after": 0
    }
  },
  {
    "id": "mcjax",
    "num": 12,
    "part": "ii",
    "title": "Multi-controller JAX",
    "lede": "Stock JAX has no coordinator: every host in the job runs the identical script and simply trusts that every other host is doing the same.",
    "goal": "Given a JAX job with jax.distributed.initialize() called, trace which objects a single jit call touches from your Python process down to the PJRT client, and explain why the whole job dies for one host's failure.",
    "sections": [
      {
        "h": "One script, run identically everywhere",
        "ps": [
          "Open a JAX training script built for a thousand-host pod and you might look for a coordinator: some process handing out work, some scheduler routing batches. Stock JAX has none. It starts one ordinary Python process per host and hands every one of them the exact same script; `jax.distributed.initialize()` is the single call that lets those processes discover each other, and after it returns nothing else about how you write the program changes.",
          "What that one call does is smaller than the name suggests. Process zero stands up a coordination service; every other process connects to it, announces which devices it holds, and receives the full picture of the job in return. Discovery, topology exchange, health checks, and that is the end of the service's duties: it never hands out work, and after startup it mostly just watches for a process going quiet. The visible result on every host is a split view of the hardware. `jax.devices()` lists every device in the job; `jax.local_devices()` lists the ones this process can actually launch work on, and the second list is the only one it can touch.",
          "The architecture has a name in the community: McJAX, for multi-controller JAX, because every host is its own controller running an identical copy of the program rather than one controller dispatching work to workers. Chapter 11's IFRT array is what lets a single logical array span every one of those processes in your Python code even though the processes never share memory; this chapter sits one layer above that, at the level where the processes themselves come into existence and find each other."
        ]
      },
      {
        "h": "Isolated except when a collective says otherwise",
        "ps": [
          "Each host runs its copy of the program forward on its own, at its own pace, with no message passing except at one specific kind of instruction: a collective. An all-reduce or an all-gather is the sole channel between hosts, and outside of those calls, a host genuinely cannot tell whether the others are ahead of it, behind it, or have crashed.",
          ">> Every host runs the same program and believes it is alone.",
          "Nobody writes those collectives by hand, and no Python process sends them. Chapter 7's `SpmdPartitioner` put them there, at compile time, when it rewrote the global-shape program into the per-device local program plus exactly the all-reduces and all-gathers the math required. And it did that rewrite on every host, identically, which is the architecture's entire correctness argument: the same script, lowered to the same StableHLO, compiled by the same deterministic compiler, yields the same executable everywhere, so every host reaches the same collective at the same point in its program without anyone checking that it will. At runtime the collective itself is device-to-device traffic over the interconnect, ICI on TPU, NCCL on GPU, launched by the executable; Python never touches a byte of it.",
          "That isolation has a cost on the other side of the same coin: because every host runs the same program in lockstep, one host dying kills the whole gang. There is no spare controller to reroute around a dead worker, because there is no controller at all, only peers running the same script. Elasticity in the face of a failed host is not something this architecture offers; it is the problem chapter 13's design exists to solve."
        ]
      },
      {
        "h": "Finding a backend to run on",
        "ps": [
          "Before any of this can dispatch anything, JAX has to find a backend to talk to, and `xla_bridge` is the module that does the finding. CPU support is built in; everything else, TPU, GPU, and any third-party accelerator, arrives as a PJRT plugin, registered through Python entry points that ship inside packages like `jax[tpu]` or `jax[cuda]`. Install one of those extras and `xla_bridge` discovers it automatically at import time.",
          "When more than one backend is available and you want to choose explicitly rather than let discovery decide, the `JAX_PLATFORMS` environment variable does that job. Set it before your script runs and the backend selection is fixed before a single line of your program executes."
        ]
      },
      {
        "h": "From a jit call to a cached executable",
        "ps": [
          "The path a single `jax.jit` call takes, once a backend is chosen, follows the chapters before this one almost exactly: trace to a jaxpr, lower that jaxpr to StableHLO, hand the StableHLO to the backend's PJRT or IFRT client to compile, and keep the resulting executable cached in memory so the next call with the same signature skips straight to execution. Nothing here is new machinery; it is the same pipeline chapters 1 through 10 built, now running once per host, every host compiling the identical program in parallel.",
          "What is new is what happens when the process restarts. An in-memory cache dies with the process that held it, and recompiling a large program through chapter 4's two hundred passes is not free. A persistent compilation cache, which you opt into, serializes compiled executables to disk, keyed on a fingerprint of the program and the compiler version, so a second run of the same script, even in a fresh process, can skip compilation entirely and load the executable straight from disk. It is the production answer to a cost this track measured directly two stages back."
        ]
      }
    ],
    "readings": [
      {
        "label": "JAX multi-process",
        "url": "https://docs.jax.dev/en/latest/multi_process.html",
        "note": "the McJAX contract, official"
      },
      {
        "label": "PJRT integration guide",
        "url": "https://openxla.org/xla/pjrt/pjrt_integration",
        "note": "how plugins register"
      }
    ]
  },
  {
    "id": "pathways",
    "num": 13,
    "part": "ii",
    "title": "Single controller",
    "lede": "One Python program drives an entire pod of chips, and the workers underneath it never see your script at all.",
    "goal": "Given the McJAX architecture from chapter 12, explain what changes and what stays fixed when the same JAX code runs against Pathways instead, and state plainly which parts of the system you can and cannot verify yourself.",
    "sections": [
      {
        "h": "One client, not one per host",
        "ps": [
          "Pathways inverts the shape chapter 12 built. Instead of one Python process per host, each blindly trusting the others to be in lockstep, there is exactly one Python client process, full stop. Underneath it sit a resource manager and a fleet of per-host workers that execute whatever programs the client hands them, gang-scheduled across however many pods the job spans.",
          "The design is not a product improvement bolted onto McJAX; it is a different paper's answer to the same problem. The Pathways paper describes an asynchronous distributed dataflow system with centralized scheduling, amortized over large operations so that one controller issuing instructions to thousands of chips does not become the bottleneck it sounds like it should be."
        ]
      },
      {
        "h": "The swap happens at IFRT, not in your code",
        "ps": [
          "The place chapter 11 spent an entire chapter building an abstraction is exactly the place Pathways plugs in. The client speaks `ifrt_proxy`, a client and server split of the IFRT interface, to a server that fronts the Pathways runtime on the other side. Your JAX code above that boundary does not change at all: the same `jax.jit`, the same arrays, the same sharding specs from chapter seven, now dispatching across a proxy instead of an in-process client.",
          ">> One controller, thousands of chips, the same JAX.",
          "That is the direct answer to a question worth asking plainly: how do you swap out the entire runtime underneath a model without touching the model. You do not touch the model. You touch the client IFRT talks to, and IFRT was built with exactly that seam in mind."
        ]
      },
      {
        "h": "Two IFRT implementations, stacked",
        "ps": [
          "Set `JAX_PLATFORMS=proxy` and JAX's backend becomes the open-source IFRT proxy client, the wire half of chapter 11's split. The product documentation names what sits on the other side, component by component: a proxy server, a gRPC front that receives the client's requests, and behind it a component the docs call the Pathways client, described there as an IFRT implementation in its own right, one that receives HLO programs and works with a resource manager to place them. Read that list twice and the shape appears: two IFRT implementations stacked in one chain, the open proxy pair doing transport, the closed Pathways one behind it doing the work.",
          "The resource manager is the piece McJAX never had. It runs on plain CPUs, owns allocation across every worker, monitors their health, pauses and resumes jobs, and serves as the single place errors surface. Chapter 12's architecture had no process that could play this role, because every process was busy being a peer."
        ]
      },
      {
        "h": "Life of a program, as the paper tells it",
        "ps": [
          "The paper fills in what the product page abstracts away. A traced program becomes a location-agnostic intermediate representation, a custom MLIR dialect, in which each compiled function is one node in a dataflow graph; a tracer can wrap a Python block that calls several jitted functions and capture the whole block as one program, which is what makes MPMD expressible at all. Buffers enter that IR through a sharded-buffer abstraction, one logical array distributed over many devices, bookkept once. Chapter 11 built exactly this instinct into IFRT, and the resemblance runs in that direction, paper first.",
          "Placement is a negotiation with the resource manager. The client asks for virtual devices, optionally constrained by type, location, or interconnect topology; the manager maps virtual onto physical, and the IR is lowered until it carries real device locations plus explicit transfer operations, scatters and gathers, between computation shards. The lowered program then becomes a dataflow program on PLAQUE, a closed-source production sharded dataflow system that carries all cross-host coordination over the data center network.",
          "Execution is gang-scheduled per island: one centralized scheduler consistently orders every computation on its island, so two programs contending for the same chips cannot deadlock holding half each. Each host runs an executor and a sharded object store, an HBM-aware cousin of Ray's, holding buffers behind opaque handles the system can migrate. And the controller does not wait its turn per operation. A compiled function's resource needs are statically known, so host-side setup for a successor runs before its predecessor finishes, and one message describing an entire subgraph lets the scheduler sequence all of its shards back to back. That is the paper's answer to the objection its own design invites, one controller in front of thousands of chips, and it is the reason the system is asynchronous dataflow rather than remote procedure calls."
        ]
      },
      {
        "h": "What a single controller buys you",
        "ps": [
          "Centralizing control is not free, but what it buys is real. MPMD becomes possible: different programs running on different islands of the same job, the shape pipelining needs. The controller also outlives any individual worker, so a worker failing is a recoverable event instead of a gang-wide one, the elasticity chapter 12 explicitly could not offer. And the scale ceiling moves: a single gang-scheduled program is no longer the largest unit of work the system can express, because one controller can orchestrate many gangs across many pods at once."
        ]
      },
      {
        "h": "What actually sits on a worker",
        "ps": [
          "The product documentation gives the worker one sentence of contract: a process on a TPU VM that receives compiled executables and performs the computations. Sit with the first half of that sentence, because it inverts chapter 12. Executables arrive at the worker already compiled; compilation happens once, in the head components, instead of once per host across the whole job, and McJAX's thousand-identical-compiles problem simply does not exist here. A sidecar gRPC server on the same VM rounds out the picture, running user-supplied Python next to the chips so data-adjacent work skips the round trip through the controller.",
          "What launches those executables is the part no public source names. Whatever it is must do a PJRT client's exact job, load a compiled program, hold device buffers, fire execution through the TPU runtime, and the paper says Pathways builds on XLA to represent and execute TPU computations. Whether that layer speaks the literal PJRT C API or Google-internal machinery that predates it is stated nowhere you can read, and this track will not guess. What is certain is that the XLA compiler never left the loop: IFRT receives HLO programs, workers receive compiled executables, and something in between is running the same passes chapter 4 dumped. Pathways replaced the runtime and the coordination, not the codegen."
        ]
      },
      {
        "h": "What you can actually verify",
        "ps": [
          "Here is the part worth stating without hedging: the Pathways runtime itself is not open source. You can read the paper that describes its design, you can read the `ifrt_proxy` client and server code in the public XLA tree, since that boundary has to be public for anything to plug into it, and you can read the product documentation for the cloud surface built on top of it. Be precise about what that public server code is, though: scaffolding, a wire front that serves whichever `ifrt::Client` it is handed. The proxy server a Pathways cluster runs stands at the same boundary and speaks the same protocol to your client, but whether that deployed binary shares a line of code with the public one is not knowable from the tree, and the runtime it fronts is closed either way, the same way the kernel path marks `libtpu` closed rather than pretending otherwise. Treat every claim in this chapter about the runtime's internals as coming from the paper and the product surface, not from code you can read yourself."
        ]
      }
    ],
    "readings": [
      {
        "label": "Pathways paper",
        "url": "https://arxiv.org/abs/2203.12533",
        "note": "the design, from its authors"
      },
      {
        "label": "Pathways on Cloud",
        "url": "https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro",
        "note": "the component list: proxy client and server, Pathways client, resource manager, workers, sidecar"
      },
      {
        "label": "Port JAX workloads to Pathways",
        "url": "https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/porting-jax-workloads",
        "note": "JAX_PLATFORMS=proxy and the worker's one-sentence contract"
      }
    ],
    "diagram": {
      "id": "xla-controllers",
      "after": 1
    }
  },
  {
    "id": "interfaces",
    "num": 14,
    "part": "ii",
    "title": "Two seams, four implementations",
    "lede": "The interfaces are lists of promises, and XLA ships working implementations of every one. This chapter walks them function by function.",
    "goal": "For each core function of PjRtClient, PjRtBuffer, PjRtLoadedExecutable, and their IFRT counterparts, state what the function promises and what XLA's own implementation does to keep the promise.",
    "sections": [
      {
        "h": "How to read this chapter",
        "ps": [
          "An abstract class is a list of promises with no bodies, so this chapter's body is code: two guided walks below step through working implementations line by line, first LAB·X5's mock plugin keeping every PJRT promise in plain C, then the real IFRT adapter from the tree keeping the same kind of promises by delegation. The walks carry the function-by-function story. The short sections here hold only what the walked code cannot show: what XLA's real clients do behind the same signatures, and how the ABI crossing works. Everything was read at openxla/xla commit `881f236` on 2026-08-10.",
          "The cast, so every name has a home. `PjRtCpuClient` (`xla/pjrt/cpu/cpu_client.h`) and `StreamExecutorGpuClient` (`xla/pjrt/gpu/se_gpu_pjrt_client.h`, built on `PjRtStreamExecutorClient`, now under `xla/pjrt/se/`) implement PJRT in-process over a shared `CommonPjRtClient` base. The C API and its client-side wrapper sit in `xla/pjrt/c/` and `xla/pjrt/c_api_client/`. The PJRT-backed IFRT adapter is `xla/python/pjrt_ifrt/`, and the proxy pair is `xla/python/ifrt_proxy/`."
        ]
      },
      {
        "h": "What the real client does behind the same functions",
        "ps": [
          "The mock's `Compile` is a memcpy; the real one is every earlier chapter in order. Follow `PjRtCpuClient::CompileAndLoad` and it calls `CompileAndAssignDevices`, which resolves layouts and runs chapter 2's StableHLO-to-HLO setup, then `CompileInternal`, then a static function named `JitCompile` where the curtain pulls back: a `cpu::CpuCompiler` runs `RunHloPasses`, chapter 4's entire pass pipeline behind one method call, then `RunBackend`, chapter 9's thunk-emitting codegen behind another. `LoadInternal` binds the device assignment last; plain `Compile`, the unloaded flavor ahead-of-time compilation needs, is the same chain stopping just short of that step.",
          "The mock's buffers are also hiding a real mechanism: the definition event. `CommonPjRtClient::BufferFromHostBuffer` branches on the `HostBufferSemantics` the caller declared. Zero-copy semantics call `ImportForeignMemory` and the host allocation simply becomes the buffer; otherwise `AllocateRawBuffer` gets device memory, `LinearizeHostBufferInto` starts the copy, and the `PjRtBuffer` returns immediately carrying an event that records when the bytes will actually be resident. Everything else queues behind that event: `ToLiteral` returns a `Future<>` because the data may not exist yet, `Delete()` drops the caller's claim at once but frees memory only after every enqueued reader finishes, and donation rides the same bookkeeping in reverse, an input surrendered to Execute becoming eligible output storage, with `non_donatable_input_indices` as the opt-out.",
          "At execute time, the CPU's loaded executable holds the `cpu_executable_` chapter 9 named: its `buffer_assignment()` decides which allocation every thunk touches, and Execute's own work is bookkeeping around the thunk run, wait on input definition events, fire the dataflow graph, wrap each output allocation in a fresh buffer with a fresh event. The GPU path is the same shape with streams in it. The signature below is the contract both keep, and the walks' transpose step shows who assembles it."
        ],
        "code": {
          "caption": "verbatim, from xla/pjrt/pjrt_client.h (openxla/xla @ 881f236, read 2026-08-10)",
          "text": "virtual absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>\nExecute(absl::Span<const std::vector<PjRtBuffer*>> argument_handles,\n        const ExecuteOptions& options,\n        std::optional<std::vector<Future<>>>& returned_futures) const = 0;",
          "lang": "cpp"
        }
      },
      {
        "h": "The ABI crossing",
        "ps": [
          "The walk ends at `GetPjrtApi` returning a struct of function pointers; this is what happens on the other side of it. `PJRT_Api_Version` carries a major and a minor (0 and 114 at this reading), the plugin reports the pair it compiled against, and every args struct's `struct_size` lets a newer caller and an older plugin read only as much of each other as they both know. Optional capability chains off `PJRT_Extension_Base` structs so extensions never touch the core ABI.",
          "And the frontend never calls the table directly. `PjRtCApiClient` (`xla/pjrt/c_api_client/pjrt_c_api_client.h`) wraps the function pointers back into the same C++ `PjRtClient` interface, while XLA's own plugins implement the table by wrapping a real C++ client, which the excerpt below catches in the act: `args->client->client->CompileAndLoad(...)`, the outer `client` a C struct, the inner one a C++ object. A program crossing the plugin boundary meets the C++ interface twice, with a function table and two protobufs in between and nothing else."
        ],
        "code": {
          "caption": "verbatim, trimmed, from xla/pjrt/c/pjrt_c_api_wrapper_impl.cc (openxla/xla @ 881f236)",
          "text": "PJRT_ASSIGN_OR_RETURN(\n    std::unique_ptr<xla::PjRtLoadedExecutable> executable,\n    std::visit(absl::Overload{\n                   [args, &options](xla::MaybeOwningMlirModule module) {\n                     return args->client->client->CompileAndLoad(\n                         std::move(module), options);\n                   },\n                   [args, &options](xla::XlaComputation program) {\n                     return args->client->client->CompileAndLoad(program,\n                                                                 options);\n                   },\n               },\n               std::move(module_or_hlo)));",
          "lang": "cpp"
        }
      },
      {
        "h": "The proxy, and the third route",
        "ps": [
          "The proxy client (`xla/python/ifrt_proxy/client/client.cc`) implements the same `ifrt::Client` functions the adapter walk stepped through, a second way: each one serializes its arguments into a protobuf request, one message type per interface function, and the server's `IfrtBackend` (`xla/python/ifrt_proxy/server/ifrt_backend.cc`) is a switch over every request case, replaying calls onto whichever in-process client it wraps. Read the two files side by side and the interface appears a third time, as a protocol: every promise in `client.h` has a proto twin.",
          "Which is the general lesson this chapter has been circling. An interface can be implemented by doing the work, the CPU client; by delegating to something that does, the adapter and the C API wrapper; or by shipping the call to a process that does, the proxy. XLA's tree holds all three in the open. The fourth, keeping the promises with machinery that is not XLA's at all, is chapter 13 standing behind the same seams, and nothing in the code you just walked would notice."
        ]
      }
    ],
    "readings": [
      {
        "label": "pjrt_client.h",
        "url": "https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_client.h",
        "note": "the promises, primary source"
      },
      {
        "label": "cpu_client.cc",
        "url": "https://github.com/openxla/xla/blob/main/xla/pjrt/cpu/cpu_client.cc",
        "note": "the worked example: CompileAndLoad down to RunHloPasses"
      },
      {
        "label": "The PJRT C ABI",
        "url": "https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h",
        "note": "the struct of function pointers, verbatim"
      },
      {
        "label": "The PJRT-backed IFRT adapter",
        "url": "https://github.com/openxla/xla/tree/main/xla/python/pjrt_ifrt",
        "note": "way one, one file per wrapped class"
      },
      {
        "label": "The IFRT proxy",
        "url": "https://github.com/openxla/xla/tree/main/xla/python/ifrt_proxy",
        "note": "way two, client and server halves"
      }
    ]
  },
  {
    "id": "capstone",
    "num": 15,
    "part": "ii",
    "title": "The dump is the referee",
    "lede": "Three projects, one deliverable each, and one rule that applies to all three: no claim survives without a dump or a measurement standing behind it.",
    "goal": "Pick one of the three candidate projects below, plan it as a one-page proposal, ship the deliverable it describes, and hand your write-up to someone else to see whether they can reproduce a single number or dump from it alone.",
    "sections": [
      {
        "h": "A different kind of chapter",
        "ps": [
          "Every chapter before this one taught you to read something the compiler or the runtime already produced: a dump, a header, a trace. This one asks you to produce something instead. Pick one of the three projects below, and the deliverable is yours to ship, not ours to hand you.",
          "The standard is the same one this whole site holds itself to: a claim without a dump or a measurement behind it is not a claim, it is an assertion. Every one of the three projects ends the same way, with a write-up someone else could check against your evidence without having to take your word for it."
        ]
      },
      {
        "h": "Project A, the plugin skeleton",
        "ps": [
          "Implement the minimal PJRT C API surface, following the integration guide, until `jax.devices()` on your machine actually sees a device you built. You will not write a fast backend; you will write the smallest one that compiles and runs at all, filling in the struct-of-function-pointers chapter one introduced with real, if trivial, implementations.",
          "The deliverable is the repository plus a write-up of every struct you had to fill in and why, in the order you filled them. That ordering is the real artifact: it is the map of what PJRT actually requires from a backend, stripped of everything a mature plugin adds on top."
        ]
      },
      {
        "h": "Project B, the pipeline teardown",
        "ps": [
          "Pick one program relevant to a model you actually care about, and dump it with every pass, the same flags chapter four taught. Then read the whole dump, start to finish, and annotate the ten passes that changed the program the most.",
          "The deliverable inlines before-and-after excerpts for each of those ten passes and does the arithmetic on what moved: bytes written, bytes read, in the same spill-style accounting chapter three's metadata and chapter six's buffer assignment set up. A reader should be able to follow your ten annotations and understand the compile the way you do, without opening the dump themselves first."
        ]
      },
      {
        "h": "Project C, the fusion brief",
        "ps": [
          "Pick one fusion decision from this site's own fusion x-ray and reconstruct why the compiler made it, arguing from the policy chapter five laid out, profitability and fusibility, not intuition about what looks efficient.",
          "Then verify your reconstruction the only way that counts: dump the same program with `--xla_disable_hlo_passes=fusion`, measure both versions on whatever backend you have, and report the difference. If your reasoning about why the compiler fused what it fused is right, disabling fusion should cost exactly what your argument predicts."
        ]
      },
      {
        "h": "The bar, stated once for all three",
        "ps": [
          "Whichever project you pick, the bar is identical: every claim in your write-up is carried by a dump or a measurement with its provenance stated, chip, dtype, shapes, method, the same discipline every gate on this site has held to since chapter one. Ship the deliverable, then hand the write-up alone, with no verbal explanation, to someone else and watch whether they can reproduce one number or one dump from it. If they can, the write-up did its job. If they cannot, the gap is the next thing to fix, not the deliverable to declare finished."
        ]
      }
    ],
    "readings": [
      {
        "label": "PJRT integration guide",
        "url": "https://openxla.org/xla/pjrt/pjrt_integration",
        "note": "project A's manual"
      },
      {
        "label": "XLA tools",
        "url": "https://openxla.org/xla/tools",
        "note": "projects B and C live in these flags"
      }
    ]
  }
]

export const XLA_MASTERY: Record<string, WorkItem[]> = {
  "xla:pjrt": [
    {
      "id": "read",
      "label": "Read this chapter and its linked readings."
    },
    {
      "id": "trace-objects",
      "label": "Write down, in order, every PJRT object a single jit-and-run call touches."
    },
    {
      "id": "skim-c-abi",
      "label": "Skim the PJRT C ABI header and find the fields that carry the version."
    },
    {
      "id": "name-handshake",
      "label": "Name the mechanism that lets a plugin's minor version lag the framework's without breaking."
    }
  ],
  "xla:ingestion": [
    {
      "id": "read",
      "label": "Read this chapter and its linked readings."
    },
    {
      "id": "diff-jaxpr",
      "label": "Write down how this StableHLO differs from the same program's jaxpr from the tracing chapter of the JAX track."
    },
    {
      "id": "find-converter",
      "label": "Find ConvertStablehloToHlo in the public XLA tree and read its signature."
    },
    {
      "id": "state-why",
      "label": "State, in one sentence, why XLA keeps StableHLO and HLO as two separate IRs instead of optimizing StableHLO directly."
    }
  ],
  "xla:hlo": [
    {
      "id": "read",
      "label": "Read this chapter and its linked readings."
    },
    {
      "id": "annotate-metadata",
      "label": "Annotate every metadata field in one dumped computation of your own."
    },
    {
      "id": "name-analyses",
      "label": "Name which analysis fusion depends on and which buffer assignment depends on, and say in one sentence why they differ."
    },
    {
      "id": "find-provenance",
      "label": "Find one instruction in your own dump whose metadata still names an exact Python source line."
    }
  ],
  "xla:pipeline": [
    {
      "id": "read",
      "label": "read the chapter and its linked readings"
    },
    {
      "id": "run-driver",
      "label": "run the driver on any machine and read your own dump end to end"
    },
    {
      "id": "spot",
      "label": "spot the decision: streak of 3",
      "href": "/gym/xla#drills",
      "auto": {
        "type": "streak",
        "key": "gym.spot.streak",
        "goal": 3
      }
    },
    {
      "id": "name-watershed",
      "label": "name the watershed pass and write, in your own words, what changes on either side of it"
    },
    {
      "id": "explain-fixpoint",
      "label": "find one pass in your dump that ran more than once and explain the fixpoint loop that caused it"
    },
    {
      "id": "labs",
      "label": "run LAB\u00b7X1",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB\u00b7X1"
        ]
      }
    },
    {
      "id": "passnames",
      "label": "which pipeline filed this pass: streak of 5",
      "href": "/gym/xla#drills",
      "auto": {
        "type": "streak",
        "key": "gym.pass.streak",
        "goal": 5
      }
    }
  ],
  "xla:fusion": [
    {
      "id": "read",
      "label": "read the chapter: why fusion is about intermediates, never the math"
    },
    {
      "id": "diff",
      "label": "in your own chapter 4 dump, find the fusion pass step and diff the module before and after it"
    },
    {
      "id": "spill",
      "label": "state in writing the exact reason the softmax spill survives fusion untouched"
    },
    {
      "id": "xray",
      "label": "work the fusion x-ray and check its calls against your own",
      "href": "/gym/xla#fusion"
    },
    {
      "id": "labs",
      "label": "run LAB\u00b7X2",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB\u00b7X2"
        ]
      }
    },
    {
      "id": "countfusions",
      "label": "count the fusions: streak of 5",
      "href": "/gym/xla#drills",
      "auto": {
        "type": "streak",
        "key": "gym.fusioncount.streak",
        "goal": 5
      }
    }
  ],
  "xla:layout-memory": [
    {
      "id": "read",
      "label": "read the chapter: what a layout promises and who has to keep it"
    },
    {
      "id": "stride",
      "label": "decode {1,0} versus {0,1} in writing, with a concrete stride example of your own"
    },
    {
      "id": "alias",
      "label": "trace one donated input through a dump to the input-output alias buffer assignment gave it"
    },
    {
      "id": "order",
      "label": "explain in one paragraph why layout assignment has to run before buffer assignment, not after"
    }
  ],
  "xla:spmd": [
    {
      "id": "read",
      "label": "read the chapter: propagation first, rewriting second, one program throughout"
    },
    {
      "id": "dump",
      "label": "take the JAX path's chapter 10 matmul, dump it with a sharding, and find every inserted collective op"
    },
    {
      "id": "rule",
      "label": "state in writing the propagation rule that decided one intermediate's sharding in that dump"
    },
    {
      "id": "match",
      "label": "name which collective each inserted op corresponds to and where in your Python the boundary that demanded it sits"
    },
    {
      "id": "labs",
      "label": "run LAB\u00b7X3",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB\u00b7X3"
        ]
      }
    }
  ],
  "xla:collectives": [
    {
      "id": "read",
      "label": "read the chapter: why a collective is a contract, not a command"
    },
    {
      "id": "deadlock",
      "label": "write the deadlock condition for a two-device all-reduce in one paragraph"
    },
    {
      "id": "gap",
      "label": "find a start/done pair in any distributed dump and name exactly what runs in the gap"
    },
    {
      "id": "trace",
      "label": "trace how a psum from the JAX path resolves down to a single participant's collective call"
    },
    {
      "id": "labs",
      "label": "run LAB·X6",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB·X6"
        ]
      }
    },
    {
      "id": "collective",
      "label": "name the collective: streak of 5",
      "href": "/gym/pytorch#collectives",
      "auto": {
        "type": "streak",
        "key": "gym.collective.streak",
        "goal": 5
      }
    }
  ],
  "xla:codegen": [
    {
      "id": "read",
      "label": "read the chapter: one dispatcher, three backends, one closed floor"
    },
    {
      "id": "artifacts",
      "label": "name the artifact each backend emits (LLVM IR thunks, Triton or cuDNN or cuBLAS thunks, libtpu) and the exact layer where the three stop sharing code"
    },
    {
      "id": "dispatch-tree",
      "label": "trace one HLO fusion through the GPU dispatch decision tree in writing and name which of the four codegen paths claims it"
    },
    {
      "id": "custom-call",
      "label": "find the custom call in the timeline x-ray and explain, in writing, who wrote that kernel"
    }
  ],
  "xla:autotuning": [
    {
      "id": "read",
      "label": "read the chapter: cost-modeled versus measured, and where the line falls"
    },
    {
      "id": "classify",
      "label": "state in writing which GPU lowering decisions are autotuned (GEMM algorithm, Triton tile sizes, convolution algorithm) versus purely cost-modeled"
    },
    {
      "id": "invalidate",
      "label": "explain what invalidates the autotune cache: a changed device, a changed program fingerprint, or a debug-options collision in the distributed key-value store"
    },
    {
      "id": "tiebreak",
      "label": "work one scratch-byte tiebreak by hand: two candidates within the two-microsecond window, and say in writing which one wins and why"
    },
    {
      "id": "timeline",
      "label": "read the timeline x-ray and name the op nobody in the program wrote",
      "href": "/gym/xla#timeline"
    }
  ],
  "xla:ifrt": [
    {
      "id": "read",
      "label": "read the chapter: one array object where PJRT had a pile of buffers"
    },
    {
      "id": "ledger",
      "label": "write the buffer-versus-array ledger for one 8-device sharded array: eight PjRtBuffers plus a framework-tracked sharding at the PJRT level, against one ifrt::Array carrying its own sharding() at the IFRT level"
    },
    {
      "id": "client-surface",
      "label": "skim client.h and list three Client responsibilities that have no PJRT counterpart"
    },
    {
      "id": "shortcut",
      "label": "read DisassembleIntoSingleDeviceArrays and FullyReplicatedShard in array.h and explain in writing why the shortcut method needs to exist at all"
    }
  ],
  "xla:mcjax": [
    {
      "id": "read",
      "label": "Read the chapter and both readings"
    },
    {
      "id": "objects",
      "label": "Draw, in writing, every object a single jit call touches from your Python process down to the PJRT client"
    },
    {
      "id": "cache",
      "label": "Enable the persistent compilation cache in a real script and confirm the second run skips compilation entirely"
    },
    {
      "id": "failure-domain",
      "label": "Write the paragraph explaining what happens to the whole job when one host process dies, and why"
    }
  ],
  "xla:pathways": [
    {
      "id": "read",
      "label": "Read the chapter and both readings"
    },
    {
      "id": "paper",
      "label": "Read sections 1 through 4 of the Pathways paper closely, not skimmed"
    },
    {
      "id": "ledger",
      "label": "Write the McJAX-versus-Pathways ledger, controllers, failure domains, program models, from memory, then check it against the paper"
    },
    {
      "id": "ifrt",
      "label": "Write one paragraph explaining why IFRT had to exist for this chapter to be possible"
    }
  ],
  "xla:interfaces": [
    {
      "id": "read",
      "label": "Read this chapter and open both quoted headers in the tree while you do"
    },
    {
      "id": "match-classes",
      "label": "Open pjrt_client.h and ifrt/client.h side by side and match each PJRT class to the IFRT class that wraps it"
    },
    {
      "id": "name-four",
      "label": "From memory, name both implementation routes for each interface and the header or directory where each lives"
    },
    {
      "id": "trace-compile",
      "label": "Follow one Compile call from PjRtCApiClient through the function table into a plugin and back"
    },
    {
      "id": "labs",
      "label": "run LAB·X5",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB·X5"
        ]
      }
    }
  ],
  "xla:capstone": [
    {
      "id": "read",
      "label": "Read this chapter in full before picking a project"
    },
    {
      "id": "plan",
      "label": "Pick one project and write the one-page plan before touching code"
    },
    {
      "id": "ship",
      "label": "Ship the deliverable exactly as the project describes it"
    },
    {
      "id": "verify",
      "label": "Hand the write-up alone to someone else and confirm they can reproduce one number or one dump from it"
    }
  ]
}

export interface XlaPathChapter {
  key: string
  href: string
  num: number
  title: string
  part: 'i' | 'ii'
}

/** The path's order; every next/previous control on /xla derives from this. */
export const XLA_PATH: XlaPathChapter[] = XLA_CHAPTERS.map((c) => ({
  key: `xla:${c.id}`,
  href: `/xla/${c.id}`,
  num: c.num,
  title: c.title,
  part: c.part,
}))

export const xlaChapterAt = (
  key: string,
): { current: XlaPathChapter; prev?: XlaPathChapter; next?: XlaPathChapter } => {
  const idx = XLA_PATH.findIndex((c) => c.key === key)
  if (idx === -1) throw new Error(`unknown xla path chapter: ${key}`)
  return { current: XLA_PATH[idx]!, prev: XLA_PATH[idx - 1], next: XLA_PATH[idx + 1] }
}
