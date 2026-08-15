// New file: site/src/data/lessons/torch-tpu-practice.ts
// SPMD, profiling and Pallas below the survey chapter 11 gives: what mark_sharding
// does on each of its two branches, where an annotation becomes HLO, what an
// xp.Trace name does to the compiled module, and how a Pallas kernel reaches a
// torch graph as a string. Every torch_xla excerpt is verbatim from pytorch/xla at
// commit 41398bf; the tpu-inference excerpts are from 878eb5e. No TPU ran for any of
// it: two snippets ran on this machine and say so, and everything that needs a chip
// is marked Colab-pending.
import type { UnitLessons } from './index'

export const TORCH_TPU_PRACTICE_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:tpu-practice',
    lessons: [
      {
        id: 'marking-a-sharding',
        num: 1,
        title: 'Marking a sharding',
        lede: 'One call annotates a tensor, and which of its two branches runs decides whether you paid nothing or paid a round trip to the host. The spec you passed decides the rest.',
        goal: 'Given a torch tensor, a mesh and a partition spec, name the tile assignment the spec produces, say which branch mark_sharding takes and what that costs, and point at the line where the annotation becomes part of the compiled module.',
        sections: [
          {
            h: 'one device, and a mesh that spans all of them',
            ps: [
              "Turn SPMD on with `xr.use_spmd()` and the first thing that changes is a number you can check. The frontend stops reporting eight devices and reports one: `_xla_num_devices` returns 1 whenever the virtual device is in use, and tensors land on a virtual `SPMD:0` instead of on `TPU:3`. From that point the program is written as though the whole slice were a single large chip, which is the arrangement the mode exists to give you.",
              "The mesh is the object that still remembers there were eight. Building one asserts that the flat list of device ids matches `xr.global_runtime_device_count()` exactly, with a comment saying XLA requires it, so a mesh is never a subset of the machine and never a superset of it. Reshape those ids however you like; a 512-core slice can be a 16 by 16 by 2 mesh or a flat 512, and the shape you pick is a logical arrangement rather than a claim about the wires.",
              "One latch is worth knowing before the first line runs. `UseVirtualDevice` sets `spmd_config_is_locked` true the first time anything queries it, and `use_spmd()` checks that lock: if tensors were already built on non-virtual devices it warns, calls `_xla_force_spmd_device()` and waits on device ops, replicating what already existed. The warning names the fix itself, which is to call `use_spmd()` at the top of the program rather than after the model is on the chip.",
            ],
            code: {
              caption: 'verbatim, torch_xla/distributed/spmd/xla_sharding.py:78-84 at 41398bf, inside Mesh.__init__',
              lang: 'python',
              text: '    # At the moment, XLA requires that the Mesh uses the global number of\n    # devices.\n    num_devices = xr.global_runtime_device_count()\n    assert num_devices > 0, "This requires XLA supported device(s)."\n    assert num_devices == len(\n        device_ids\n    ), f"Number of device IDs ({len(device_ids)}) must match the global number of devices ({num_devices})"',
            },
          },
          {
            h: 'the spec is a permutation of the mesh',
            ps: [
              "A partition spec has one entry per tensor dimension, and `mark_sharding` refuses anything shorter. The assert is deliberate rather than defensive: the comment above it says unspecified dimensions are not filled in with replication, because partial replication builds its group assignment from the spec and the groups would vary with rank if the fill were implicit. So a rank-3 tensor gets a three-entry spec, `None` included, every time.",
              "What that spec does is reorder the mesh. The tiled entries become a permutation, the untouched mesh axes are appended after them, and the logical mesh is transposed by that permutation. Then any entry that was a tuple collapses its adjacent axes into one, which is how a single tensor dimension gets sharded over two mesh axes at once. The device ordering that falls out is the tile assignment, and it is the whole content of the annotation.",
              "The kind of sharding falls out of the same spec by three cheap tests. Every entry `None` is replication. Any entry `None`, with others tiled, is partial replication. Neither of those, and it is tiled. A single device short-circuits all three and gives maximal. The jax path's own lesson at /jax/sharding/mesh-and-spec teaches the mesh and the spec as concepts; the arithmetic below is what the torch side does with them.",
            ],
            code: {
              caption: 'verbatim, torch_xla/distributed/spmd/xla_sharding.py:520-536 at 41398bf, the body of _get_tile_assignment',
              lang: 'python',
              text: '  # Flatten the partition spec and ensure that it is fully specified over the\n  # mesh for permutation.\n  tiled_dims = [x for x in partition_spec if x is not None]\n  permutation = np.hstack(tiled_dims).tolist() if tiled_dims else []\n  missing_axes = sorted(set(range(len(mesh.shape()))) - set(permutation))\n  tile_assignment = mesh.get_logical_mesh().transpose(permutation +\n                                                      missing_axes)\n\n  # For any tuples in the partition_spec, the grouped axes will be adjacent\n  # after the permutation. Combine these dimensions into a single axis.\n  for i, spec in enumerate(tiled_dims):\n    if isinstance(spec, tuple):\n      shape = tile_assignment.shape\n      tile_assignment = tile_assignment.reshape(shape[:i] + (-1,) +\n                                                shape[i + len(spec):])\n\n  return tile_assignment',
            },
            table: {
              caption: 'run on this machine (python 3.12, numpy 2.2.6, no torch_xla installed): _get_tile_assignment and _get_sharding_type executed verbatim out of the pinned checkout against a stub mesh of 8 devices shaped (data=4, model=2)',
              cols: ['partition spec', 'sharding type', 'tile shape', 'device order'],
              rows: [
                ['(0, 1)', 'TILED', '(4, 2)', '0 1 2 3 4 5 6 7'],
                ['(0, None)', 'PARTIAL', '(4, 2)', '0 1 2 3 4 5 6 7'],
                ['(None, 1)', 'PARTIAL', '(2, 4)', '0 2 4 6 1 3 5 7'],
                ['((0, 1), None)', 'PARTIAL', '(8,)', '0 1 2 3 4 5 6 7'],
                ['(None, None)', 'REPLICATED', '(4, 2)', '0 1 2 3 4 5 6 7'],
              ],
            },
          },
          {
            h: 'two branches inside one call',
            ps: [
              "The bridge arc's second lesson lists `mark_sharding` among the lines that force materialization, with the qualifier that it only does so sometimes. The qualifier is a branch, four lines into `XlaMarkSharding`, and reading it tells you which of your own annotations are free.",
              "If the tensor's current IR value is anything other than a `DeviceData` node, the call hands off to `XlaAnnotateCustomSharding` and returns. That path splices a `CustomSharding` node into the graph, which lowers to a custom call, and nothing moves. Annotating an activation halfway through a forward pass costs one IR node.",
              "If the IR is a `DeviceData` node, meaning a parameter or an input, the sharding is physical and the call has to deal with bytes. When the host copy is still around, which is the ordinary case under the virtual device because the initial upload was deferred, it reuses that copy and bumps a `VirtualDeviceUsage` counter. When it is not, the tensor is pulled back from the device through `GetTensors` first. Either way the data is re-created against the virtual device as shards.",
              "Two guards sit on that same path. A second annotation with the same spec returns early and does nothing. A second annotation with a different spec is refused outright unless the existing one was replicated or unknown, with the message asking you to clear the old annotation first. Both branches raise their own counter, so a metrics report says which one your program took without you having to reason it out.",
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/xla_sharding_util.cpp at 41398bf: the branch at 800-809, then the deferred-upload case at 819-825',
              lang: 'c',
              text: '  // For Non DeviceData IR values, we directly attach the sharding spec to the\n  // xtensor.\n  const DeviceData* device_data_node = nullptr;\n  if (xtensor->CurrentIrValue()) {\n    device_data_node = DeviceData::Cast(xtensor->CurrentIrValue().node.get());\n    if (!device_data_node) {\n      XlaAnnotateCustomSharding(xtensor, sharding);\n      return;\n    }\n  }\n\n  if (xtensor->CurrentTensorData().has_value()) {\n    TORCH_LAZY_COUNTER("VirtualDeviceUsage", 1);\n    // When virtual device is enabled for SPMD, we defer the initial\n    // data transfer to the device and retain the original data on the\n    // host, until the sharded data transfer.\n    cpu_tensor = xtensor->CurrentTensorData().value();\n  } else {',
            },
          },
          {
            h: 'where an annotation becomes HLO',
            ps: [
              "Up to compile time the sharding is torch-side bookkeeping, mirrored in two places: on the tensor, where a comment calls it the source of truth for every lookup, and on the node, as its output shardings. Neither of those is something XLA can read. One function does the translation, and it runs once per compile, from inside `XLAGraphExecutor::Compile`.",
              "`SetHloSharding` walks every emitted output of the lowering context, casts the node back to an `XlaNode`, reads the sharding for that output index, and writes it straight into the HLO instruction proto. Anything typed unknown is skipped, which is how implicit replication stays implicit rather than becoming an assertion the partitioner has to honour.",
              "Input parameters take a different road entirely, because a parameter is not an emitted output. `LoweringContext::GetParameter` checks whether the backing data carries a sharding and, when it does, wraps the `xla::Parameter` call in an `xla::XlaScopedShardingAssignment`. Same annotation, written at a different moment, which is the underlying reason marking a parameter had to be physical two sections ago.",
              "What the compiler then does with those annotations is not this arc's story. The propagation pass and the collectives it inserts are taught on the xla path at /xla/spmd/the-partitioner, and /xla/collectives/six-shardings-one-matmul captures the collectives for one matmul under six different shardings on this repo's own hardware. Chapter 11's mastery item points you at that comparison for a reason: it is the answer key for the sharding you are about to write.",
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/xla_sharding_util.cpp:172-188 at 41398bf',
              lang: 'c',
              text: 'bool ShardingUtil::SetHloSharding(LoweringContext* lowering_ctx) {\n  bool is_sharded = false;\n  for (std::pair<torch::lazy::Output, xla::XlaOp> elem :\n       lowering_ctx->GetEmittedOutputs()) {\n    const torch::lazy::Node* node = elem.first.node;\n    const XlaNode* xla_node = dynamic_cast<const XlaNode*>(node);\n    xla::HloInstructionProto* instruction =\n        XlaBuilderFriend::GetInstruction(elem.second);\n    const std::shared_ptr<xla::OpSharding> sharding =\n        xla_node->GetSharding(elem.first.index);\n    if (sharding != nullptr && sharding->type() != xla::OpSharding::UNKNOWN) {\n      *instruction->mutable_sharding() = *sharding;\n      is_sharded = true;\n    }\n  }\n  return is_sharded;\n}',
            },
          },
          {
            h: 'the gradient needs its own annotation',
            ps: [
              "`mark_sharding` edits the tensor in place and hands back an `XLAShardedTensor` view of it. On a parameter or an input that is the whole job. On an activation in the middle of a model it is half of one, because the backward pass will compute a gradient for that activation and nothing has said where the gradient lives.",
              "The docstring on `MarkShardingFunction` gives the reason in one sentence, quoted here in full: \"This is required to guide GSPMD sharding propagation better during the backward pass as during complicated workloads the compiler can introduce extra collectives that can hurt performance.\" A collective you did not ask for is the failure mode, and it appears in the profile as device time rather than as an error.",
              "The mechanism is an autograd Function whose forward and backward call the same custom op. That op is registered through `torch.library`, marks a clone rather than the input, and has a fake implementation returning an empty tensor of the same shape, which is what lets it survive AOTAutograd and appear in a dynamo-captured graph. The bridge arc's third lesson explains why forward and backward arrive at the bridge separately in the first place.",
              '>> Annotating the activation and annotating its gradient are two different instructions to the same partitioner.',
            ],
            code: {
              caption: 'verbatim, torch_xla/distributed/spmd/xla_sharding.py:1556-1586 at 41398bf, with the import and parse lines inside the custom op elided at the ellipsis; the sentence quoted in the prose above is the class docstring of the same file at 1545-1547',
              lang: 'python',
              text: '  @staticmethod\n  def forward(ctx, torch_tensor: torch.Tensor, mesh: Mesh,\n              partition_spec: PartitionSpec) -> torch.Tensor:\n    o = _aot_mark_sharding(torch_tensor, str(mesh), str(partition_spec))\n    ctx.partition_spec = partition_spec\n    ctx.mesh = mesh\n    return o\n\n  @staticmethod\n  def backward(ctx, grad_output: torch.Tensor):  # type: ignore\n    partition_spec = ctx.partition_spec\n    mesh = ctx.mesh\n    o = _aot_mark_sharding(grad_output, str(mesh), str(partition_spec))\n    return o, None, None\n\n\n@torch.library.custom_op("xla::aot_mark_sharding", mutates_args=())\ndef _aot_mark_sharding(t: torch.Tensor, mesh: str,\n                       partition_spec: str) -> torch.Tensor:\n  ...\n  return xs.mark_sharding(t.clone(), the_mesh,\n                          partition_spec_eval).global_tensor',
            },
          },
        ],
        readings: [
          { label: 'xla_sharding.py at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/distributed/spmd/xla_sharding.py', note: 'Mesh, PartitionSpec, the tile assignment, and both marking APIs, in one file' },
          { label: 'xla_sharding_util.cpp at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/xla_sharding_util.cpp', note: 'the two branches at 791 and the HLO write at 172' },
          { label: 'the SPMD user guide at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/perf/spmd_basic.md', note: 'the maintainers\' own worked example, including which device holds which shard' },
        ],
        check: [
          {
            q: 'SPMD is on and the frontend reports one device. How is a mesh over eight chips still legal?',
            a: 'Because a mesh is sized against xr.global_runtime_device_count(), the runtime device count, not against what the frontend reports. The assert in Mesh.__init__ requires the flat id list to match that count exactly.',
          },
          {
            q: 'The same parameter gets two mark_sharding calls with two different specs. What happens on the second?',
            a: 'It is refused. Equal specs return early and do nothing; a different spec raises unless the existing annotation was replicated or unknown, with a message asking for the existing annotation to be cleared first.',
          },
          {
            q: 'Why does mark_sharding_with_gradients exist when mark_sharding already annotates the tensor?',
            a: 'Because it annotates the gradient too. Forward and backward both call the xla::aot_mark_sharding custom op on a clone, which guides GSPMD propagation through the backward pass and survives AOTAutograd; the docstring names extra collectives as the cost of leaving it out.',
          },
        ],
        work: [
          { id: 'tile-assignment', label: 'take one tensor from a model you have written, pick a spec, and write out the tile assignment by hand before checking it against the helper', href: '#the-spec-is-a-permutation-of-the-mesh' },
          { id: 'branch-audit', label: 'mark every mark_sharding call in one program with which of the two branches it takes, and say which ones move bytes', href: '#two-branches-inside-one-call' },
        ],
      },
      {
        id: 'reading-a-tpu-profile',
        num: 2,
        title: 'Reading a TPU profile',
        lede: 'A name you write on the host reappears as a prefix on device instructions, which is the only reason a timeline of a lazy program is readable at all. Two other instruments answer questions the timeline cannot.',
        goal: 'Name the three capture surfaces and what each produces, say what an xp.Trace name does to the compiled module and which flag gates it, and say which question belongs to the counters rather than to the trace.',
        sections: [
          {
            h: 'two ways to ask for a profile',
            ps: [
              "The on-demand route starts a server inside the training process and samples it from outside. `xp.start_server(port)` returns an object whose lifetime is the server's, and `xp.trace(service_addr, logdir, duration_ms)` blocks while it collects. Its defaults are worth reading once: host tracing at level 2, device tracing on, and a retry loop that keeps re-sending the request every `interval_s` until `timeout_s` runs out, because the thing being profiled may be busy compiling when the first request lands.",
              "The in-process route skips the server. `xp.start_trace(log_dir)` opens a profiler session, `xp.stop_trace()` stops it and exports to that directory, and a second `start_trace` while one is running raises rather than quietly nesting. The implementation says it is based on the jax profiler, and the shape of the API is the same shape jax users already know.",
              "Both routes produce the same kind of capture, and neither one knows anything about torch. What makes the capture legible is the third surface, the annotations you put in the loop, which is the rest of this lesson.",
            ],
            table: {
              caption: 'the three capture surfaces, torch_xla/debug/profiler.py at 41398bf',
              cols: ['what you call', 'where it runs', 'what you get'],
              rows: [
                ['xp.start_server(port), then xp.trace(addr, logdir, duration_ms)', 'the trace client blocks somewhere else while the server keeps training', 'a capture in logdir, retried every interval_s until timeout_s'],
                ['xp.start_trace(log_dir), then xp.stop_trace()', 'inside the training process, one session at a time', 'the same capture, exported on stop; a second start raises'],
                ['xp.Trace(name), xp.StepTrace(name, step_num=n)', 'inside the loop, around a region or a step', 'a host event, plus a name that travels into the compiled module'],
              ],
            },
            code: {
              caption: 'verbatim, torch_xla/debug/profiler.py:82-94 at 41398bf, the body of trace',
              lang: 'python',
              text: '  options = {\n      \'host_tracer_level\': host_tracer_level,\n      \'device_tracer_level\': device_tracer_level,\n      \'delay_ms\': delay_ms,\n  }\n  torch_xla._XLAC.profiler.trace(\n      service_addr,\n      logdir,\n      duration_ms=duration_ms,\n      num_tracing_attempts=num_tracing_attempts,\n      timeout_s=timeout_s,\n      interval_s=interval_s,\n      options=options)',
            },
          },
          {
            h: 'a name on the host becomes a prefix in the module',
            ps: [
              "Wrap a region in `xp.Trace('fwd')` and two things happen at once. A host trace event opens, which is the part you expected. The context manager also pushes a lazy scope, and every IR node built while that scope is open records the name in its metadata.",
              "That recorded scope is what makes a device timeline readable. When the lowering writes HLO metadata, a node whose metadata carries a scope gets that scope as its op name prefix, so an instruction that would have been called `xla__add` is called `fwd/xla__add` instead. The comment above that code says why the string manipulation is there: the xprof backend groups and nests traces by op name and type patterns, so the naming is the grouping.",
              "One flag gates the whole thing. The metadata is only populated when `XLA_HLO_DEBUG` is set or the lazy IR debug flag is on, which means a profile captured without either shows your host annotations and shows unnamed device work. Nothing errors. The names simply do not arrive on the far side.",
              "`Trace` also enters a jax named scope when jax is importable, so the same annotation carries through a torchax lowering rather than needing a second, jax-shaped version of itself.",
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/lowering_context.cpp:87-99 at 41398bf, inside PopulateXlaOpMetadata',
              lang: 'c',
              text: '    std::string op_name_prefix;\n    size_t max_stack_depth = nmeta.frame_info.size();\n\n    if (custom_opname_meta != nullptr) {\n      op_name_prefix = custom_opname_meta->op_name_prefix;\n      max_stack_depth = custom_opname_meta->max_stack_depth;\n    }\n\n    else if (!nmeta.scope.empty()) {\n      op_name_prefix =\n          absl::StrCat(absl::StrReplaceAll(nmeta.scope, {{":", "_"}}), "/");\n    }\n    metadata.set_op_name(absl::StrCat(op_name_prefix, op_type));',
            },
          },
          {
            h: 'StepTrace ends with a sync',
            ps: [
              "`xp.StepTrace` looks like `xp.Trace` with a step number attached, and its exit does something the plain version does not. It deletes the scope first, then calls `torch_xla.sync()`, then closes the host event. The comment explains the ordering: the step marker checks that no scope is still open, so the scope has to go before the barrier does.",
              "That makes the step annotation a cut point as well as a label. Whatever the bridge arc's second lesson taught about lines that force materialization applies to this one, with the difference that here the cut is the intended behaviour rather than an accident you are hunting.",
              "The profiling guide puts one restriction on the plain version, and it follows from the same ordering. A region wrapped in `xp.Trace()` must not contain a call to `torch_xla.sync()`. Reach for `StepTrace` when a barrier belongs inside the region, and for `Trace` when it does not.",
            ],
            code: {
              caption: 'verbatim, torch_xla/debug/profiler.py:167-180 at 41398bf, the whole of StepTrace past its docstring',
              lang: 'python',
              text: '  def __init__(self, name: str, **kwargs):\n    super().__init__(name, _r=1, **kwargs)\n\n  def __enter__(self):\n    set_tracer_marked_step(True)\n    super().__enter__()\n\n  def __exit__(self, type, value, traceback):\n    if getattr(self, \'scope\', None):\n      # In ir.cpp ResetScopeContext we ensure that we have no remaining scope\n      # before marking step.\n      del self.scope\n    torch_xla.sync()\n    super().__exit__(type, value, traceback)',
            },
          },
          {
            h: 'what a gap means',
            ps: [
              "The repo ships a worked reading of two real profiles, and it is the closest thing to an answer key this course can point at while it has no chip of its own. A Stable Diffusion 2.1 pipeline on a v4-8 profiles as busy with one small gap in the middle, and without annotations there is no way to find out what the host was doing during it. Adding `xp.Trace` calls to the pipeline and the U-Net turns that gap into a named Python region.",
              "The XL version of the same model shows two shapes at once. There is one large gap caused by watermarking, which the walkthrough diagnoses by noticing that the gap is preceded by a `TransferFromDevice`, and which turns out to be tensors moving to CPU and becoming numpy arrays for two libraries that need them there. There are also many small gaps inside the denoising loop, traced to `.item()` and `.nonzero()` calls inside the scheduler's step, each one cutting the loop graph into smaller pieces.",
              '>> If the gaps in the profile are due to Python code tracing that happens on the host, then this might be a bottleneck and there is no further straightforward optimization that can be done.',
              "That quote is the guide's own conclusion, and it is the honest half of profiling a lazy program. Some gaps are your program asking for a value it did not need. Others are tracing itself, and no amount of rearranging the loop removes them.",
            ],
          },
          {
            h: 'counters answer the question a timeline cannot',
            ps: [
              "A trace tells you where time sat. It does not tell you how many times the frontend crossed into the runtime, or which of the seam's named paths it took, and those are usually the first questions worth asking. The metrics report answers them by name, counting calls raised inside `PjRtComputationClient` itself.",
              "LAB·P5 does that reading end to end against the pinned line numbers, and its reference run on a Colab TPU v6e-1 is the number to compare your own against: the same two-step loop reports one compile and two executes on the lazy path, and 23 compiles and 58 executes under the eager names. The lab hangs off chapter 10 with the rest of the runnable work, and this lesson does not repeat its table; the point of naming it here is the ordering.",
              "Read the counters first, because they are exact and cost nothing. Capture a trace when the counts already look the way you expected and the wall clock still does not. A profile of a program that is compiling four times per step is a picture of the wrong problem.",
            ],
          },
          {
            h: 'what this machine could not do',
            ps: [
              "None of the captures above ran here. The machine this course was written on has no TPU, `xp.trace` needs a profiler server on a machine that has one, and a device plane without a device is nothing. Everything in this lesson is read from the pinned source and the pinned docs.",
              "The pending half is small and specific, which is how a pending marker should read. Start a server in a Colab TPU runtime, wrap one step in `StepTrace` with `XLA_HLO_DEBUG=1` set, capture, and check two things: that your step name appears on device instructions and not only on host events, and that the gaps you predicted are the gaps you got. LAB·P5 already holds the counter half of that same run.",
            ],
          },
        ],
        readings: [
          { label: 'profiler.py at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/debug/profiler.py', note: 'the whole profiling surface in 256 lines: server, trace, Trace, StepTrace' },
          { label: 'the profiling walkthrough at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/learn/xla-profiling.md', note: 'two real profiles read gap by gap, with the annotations that found each cause' },
          { label: 'Profiling PyTorch/XLA on a TPU VM', url: 'https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm', note: 'the capture workflow the doc above assumes, official' },
        ],
        check: [
          {
            q: 'Your xp.Trace names show up on host events but nothing on the device plane is named. What is missing?',
            a: 'The metadata flag. HLO op metadata is only populated when XLA_HLO_DEBUG is set or the lazy IR debug flag is on, and the op name prefix that groups device work in xprof comes from that metadata.',
          },
          {
            q: 'Why does StepTrace delete its scope before calling torch_xla.sync() rather than after?',
            a: 'Because the step marker resets the scope context and checks that no scope remains open. The comment in the source says so directly, which is also why the plain Trace must not contain a sync at all.',
          },
          {
            q: 'A step shows one long host gap and the counters show four compiles per step. Which do you fix first?',
            a: 'The compiles. The counters are exact and say the boundary is being crossed more than the loop should need; a timeline of a program that recompiles every step is a picture of compilation, not of the step you meant to measure.',
          },
        ],
        work: [
          { id: 'annotate-a-step', label: 'annotate one step of a loop you have written and name both places the annotation is supposed to appear', href: '#a-name-on-the-host-becomes-a-prefix-in-the-module' },
          { id: 'gap-prediction', label: 'before capturing anything, write down the three lines of your loop you expect to show up as host gaps, and why', href: '#what-a-gap-means' },
        ],
      },
      {
        id: 'calling-a-pallas-kernel',
        num: 3,
        title: 'Calling a Pallas kernel',
        lede: 'The kernel is compiled by jax, and what reaches the torch graph is a base64 string. Everything torch does afterwards follows from that one fact, including why editing two lines of kernel body recompiles the step.',
        goal: 'Trace a Pallas kernel from a Python function to an HLO custom call inside a torch graph, name what gets cached along the way and what does not, and say what the shipping serving stack reuses from the same path.',
        sections: [
          {
            h: 'jax lowers the kernel, torch never sees it',
            ps: [
              "There is no torch-side Pallas. `trace_pallas` takes the kernel and the tensors you would call it with, replaces every tensor with a `jax.ShapeDtypeStruct` carrying only shape and dtype, and jits the kernel over those meta values. Nothing executes, because meta values have no storage; the point is the module that lowering produces.",
              "In that module the kernel appears as a single `stablehlo.custom_call` named `tpu_custom_call`, and its `backend_config` holds the compiled Mosaic body as a base64 blob. `_extract_backend_config` walks two levels of the module looking for exactly that operation and returns its config string. The docstring in the source shows a full example module, which is worth reading once because it makes the shape of the thing concrete.",
              "So the object that crosses from jax to torch is text. Not a function pointer, not a compiled binary the torch runtime loads, and not anything torch can inspect. A string, handed onward.",
            ],
            code: {
              caption: 'verbatim, torch_xla/experimental/custom_kernel.py:247-257 at 41398bf, the end of trace_pallas',
              lang: 'python',
              text: '  # Here we ignore the kwargs for execution as most of the time, the kwargs is only used in traced code.\n  ir = jax.jit(\n      kernel, static_argnums=static_argnums,\n      static_argnames=static_argnames).lower(*jax_args, **kwargs).compiler_ir()\n  payload = _extract_backend_config(ir)\n\n  if use_cache:\n    # if we reach here it means we have a cache miss.\n    trace_pallas_arg_to_payload[hash_key] = payload\n\n  return payload, tensor_args',
            },
          },
          {
            h: 'what this machine could and could not do',
            ps: [
              "That lowering step is the one piece of the path a machine without a TPU can attempt, so it was attempted here. jax 0.4.38 on CPU imports the Mosaic registration happily and then refuses the lowering, in one sentence: only interpret mode is supported on the CPU backend.",
              "Run the same kernel with `interpret=True` and it lowers fine, into ordinary StableHLO with no custom call in it at all. Feed that module to the extractor from the pinned source and it returns `None`, which is the correct answer rather than a bug: interpret mode inlines the kernel, so there is no payload to extract and nothing for torch to carry.",
              "The runnable half of this lesson is therefore Colab-pending, in the same sense LAB·P5 uses the term. The mechanism below is read from source at the pinned commit. Producing a real payload, printing its first eighty characters, and finding `tpu_custom_call` in a dumped HLO module all need a TPU runtime, and they belong in the same lab pass that closes chapter 12.",
            ],
            code: {
              caption: 'run on this machine (jax 0.4.38, CPU, no TPU attached), three short scripts, output trimmed to the answering lines; traceback frames and file paths cut',
              lang: 'text',
              text: '$ python pallas_lower.py           # the trace_pallas lowering, unmodified\njax 0.4.38 [CpuDevice(id=0)]\nmosaic registration: imported\nValueError: Only interpret mode is supported on CPU backend.\n\n$ python pallas_interp.py          # same kernel, interpret=True\ntpu_custom_call in module: False\ncustom_call in module: False\nmodule lines: 45\n\n$ python extract_none.py           # _extract_backend_config, verbatim from 41398bf\npayload: None',
            },
          },
          {
            h: 'a string crosses into the graph',
            ps: [
              "On the torch side the payload goes to `_xla_tpu_custom_call`, which lands in `tensor_methods::tpu_custom_call` and builds one `TpuCustomCall` IR node. Look at what that node's constructor does with the payload: it passes it as the hash seed. The bridge arc's first lesson enumerated the ingredients of a graph hash; this is the kernel body entering that number.",
              "The consequence is worth stating plainly, because it surprises people who think of a kernel as a library call. Edit two lines inside the kernel body, and the payload changes, so the node hash changes, so the graph hash changes, so the step compiles again. A kernel is part of the program's identity, not a dependency the program links against.",
              "Lowering the node emits an XLA custom call whose target name is the string `tpu_custom_call`, with layouts forced to torch's own major-to-minor order on both the inputs and the outputs, because Mosaic requires that ordering. A single-output kernel skips the tuple wrapper entirely, with a comment saying Mosaic rejects a tuple of one.",
            ],
            code: {
              caption: 'two verbatim excerpts under added path headings, both at 41398bf: torch_xla/csrc/ops/tpu_custom_call.cpp:9-15, then torch_xla/csrc/xla_lower_util.cpp:1307-1314',
              lang: 'c',
              text: '// torch_xla/csrc/ops/tpu_custom_call.cpp: the payload is the node hash seed\nTpuCustomCall::TpuCustomCall(torch::lazy::OpList inputs,\n                             xla::Shape output_shape,\n                             const std::string& payload)\n    : XlaNode(xla_tpu_custom_call, inputs, output_shape,\n              /*num_outputs=*/output_shape.tuple_shapes_size(),\n              torch::lazy::MHash(payload)),\n      payload_(payload) {}\n\n// torch_xla/csrc/xla_lower_util.cpp: and the lowering names the call target\n  // Mosaic has some weird checks that disallow using a tuple output for single\n  // element.\n  if (output_shapes.size() == 1) {\n    return {xla::CustomCallWithLayout(inputs[0].builder(),\n                                      /*call_target_name=*/"tpu_custom_call",\n                                      inputs, output_shapes[0], input_shapes,\n                                      payload)};\n  }',
            },
          },
          {
            h: 'the payload cache is a python dict',
            ps: [
              "Lowering a kernel through jax is not free, and doing it on every call would put a jit trace inside your training step. There is a cache, and it is a module-level dictionary rather than anything the runtime knows about.",
              "Read its key and you learn what counts as the same kernel: the default matmul precision, the kernel function object itself, the static argnums and argnames, the meta arguments with their shapes and dtypes, and the keyword arguments repr'd and sorted. A hit bumps a `trace_pallas_cache_hit` counter you can read in the same metrics report as the seam names from LAB·P5.",
              "The cache is opt-in per call site, and the comment says why: the key assumes every keyword argument is hashable and not a tensor, which holds for the grouped matmul kernels that use it and is not promised in general. A kernel wrapper of your own that turns `use_cache` on has just taken on that assumption.",
            ],
            code: {
              caption: 'verbatim, torch_xla/experimental/custom_kernel.py:234-245 at 41398bf',
              lang: 'python',
              text: '  hash_key = ()\n  if use_cache:\n    global trace_pallas_arg_to_payload\n    # implcit assumption here that everything in kwargs is hashable and not a tensor,\n    # which is true for the gmm and tgmm.\n    hash_key = (jax.config.jax_default_matmul_precision, kernel, static_argnums,\n                tuple(static_argnames)\n                if static_argnames is not None else static_argnames,\n                tuple(jax_args), repr(sorted(kwargs.items())).encode())\n    if hash_key in trace_pallas_arg_to_payload:\n      torch_xla._XLAC._xla_increment_counter(\'trace_pallas_cache_hit\', 1)\n      return trace_pallas_arg_to_payload[hash_key], tensor_args',
            },
          },
          {
            h: 'a kernel that is also a torch op',
            ps: [
              "The kernels the repo ships are not left as bare Python functions. Flash attention, paged attention, grouped matmul and the rest are each defined as a torch library op with a schema, then given two implementations. The XLA one calls the real thing. The `CompositeExplicitAutograd` one returns an empty tensor shaped like the query, and warns when the tensors are not on a meta device.",
              "That second implementation is what makes the kernel usable under `torch.compile`. Dynamo builds fake outputs with meta tensors while it is capturing, and a fake tensor cannot run a Mosaic kernel; what it needs is a shape rule. The empty tensor is that rule, written as code.",
              "The same registration is why a model containing one of these kernels still imports and traces on a CPU box, with a warning rather than an import error. It will not compute anything correct there, and the warning says so, but the graph is capturable, which is usually what you were doing on the CPU box anyway.",
            ],
            code: {
              caption: 'verbatim, torch_xla/experimental/custom_kernel.py:1515-1545 at 41398bf, the fallback and the two registrations, spelling as in the source',
              lang: 'python',
              text: 'def non_xla_attetion(q, k, v, attention_type):\n  # This will be called when dynamo use fake tensor to construct the fake output.\n  # We need to make sure output tensor\'s shape is correct.\n  if k.device != torch.device("meta"):\n    warnings.warn(\n        f\'XLA {attention_type} attention should only be applied to tensors on XLA device\'\n    )\n\n  # Return orignal shape of q.\n  return torch.empty_like(q)\n\n\nXLA_LIB.define(\n    "flash_attention(Tensor q, Tensor k, Tensor v, bool casual=False) -> Tensor",\n)\n\n\n@impl(XLA_LIB, "flash_attention", "XLA")\ndef flash_attention_xla(q: torch.Tensor,\n                        k: torch.Tensor,\n                        v: torch.Tensor,\n                        causal: bool = False):\n  return flash_attention(q, k, v, causal=causal)\n\n\n@impl(XLA_LIB, "flash_attention", "CompositeExplicitAutograd")\ndef flash_attention_non_xla(q: torch.Tensor,\n                            k: torch.Tensor,\n                            v: torch.Tensor,\n                            causal: bool = False):\n  return non_xla_attetion(q, k, v, "flash")',
            },
          },
          {
            h: 'sharding around a kernel is a different instruction',
            ps: [
              "A kernel runs on one device's worth of data, so a sharded model cannot hand it a global tensor and hope. The wrapper for that is `_shard_map`, which calls `enable_manual_sharding` on each input, runs the single-device function, and calls `disable_manual_sharding` on each output after computing what the full shape should be from the mesh axis sizes.",
              "Flash attention uses exactly that when a mesh is passed: nine input specs, three output specs, and the same single-device forward underneath either way. The specs for the log-sum-exp outputs differ from the specs for the queries, which is the kind of detail that only shows up when the kernel has more outputs than the obvious one.",
              "Set this next to lesson one and the difference is the interesting part. `mark_sharding` leaves an annotation for the partitioner to propagate, and the compiler decides what the shard shapes are. Manual sharding ends propagation at a boundary and hands you the shard shapes as your problem. A kernel is the case where you wanted the second one.",
            ],
          },
          {
            h: 'the same path, in production',
            ps: [
              "The chapter above says vLLM's TPU backend runs PyTorch model definitions through a jax lowering path, and the repository states it in its own words: tpu-inference is described as a hardware plugin unifying JAX and PyTorch under a single lowering path within the vLLM project, whose aim includes running PyTorch model definitions performantly on TPU without any additional code changes.",
              "The wrapper that does it is one file, and its imports are the argument. It imports torchax, pulls `jax_view` and `torch_view` out of `torchax.interop`, calls the torch model through `torch.func.functional_call` with jax-backed parameters viewed as torch tensors, and shards its inputs with `jax.sharding.NamedSharding` over a `PartitionSpec`. Those are the same two objects the jax path's chapter 10 built, applied to a torch module, which is the convergence chapter 10 of this path described made into imports.",
              "Read that as a map rather than a recipe. The kernels sit in their own directory, the sharding is jax's, the model definition is torch's, and each of those layers has a chapter on this site that teaches it. Placing them is chapter 11's mastery item, and doing it yourself is worth more than reading someone else's answer.",
              "Provenance for this section: read on 2026-08-15 from tpu-inference at commit 878eb5e, not run. Nothing here was measured.",
            ],
          },
        ],
        readings: [
          { label: 'custom_kernel.py at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/experimental/custom_kernel.py', note: 'the whole path: trace_pallas, the payload cache, shard_map, and every shipped kernel op' },
          { label: 'tpu_custom_call.cpp at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/ops/tpu_custom_call.cpp', note: '37 lines, and the payload appears in two of them' },
          { label: 'vllm_model_wrapper.py at 878eb5e', url: 'https://github.com/vllm-project/tpu-inference/blob/878eb5e68611691ee4805517a54318cf34621a5c/tpu_inference/models/vllm/vllm_model_wrapper.py', note: 'the torch path of the serving stack, in its imports and its shardings' },
        ],
        check: [
          {
            q: 'Where does a Pallas kernel actually get compiled, and what reaches the torch graph?',
            a: 'jax compiles it. trace_pallas jits the kernel over ShapeDtypeStruct meta values, lowers it, and pulls the backend_config off the stablehlo.custom_call in the result. What reaches torch is that base64 payload string and nothing else of the kernel.',
          },
          {
            q: 'You edited two lines inside a kernel body and the whole step recompiled. Why?',
            a: 'Because the payload is the hash seed of the TpuCustomCall node. A different kernel body is a different payload, so a different node hash, so a different graph hash, so a cache miss at the step barrier.',
          },
          {
            q: 'Why does flash_attention register a second implementation that returns an empty tensor?',
            a: 'To give dynamo a shape rule. Capture builds fake outputs with meta tensors, which cannot run a Mosaic kernel, so the CompositeExplicitAutograd implementation returns torch.empty_like(q) and warns when the tensors are not on meta.',
          },
        ],
        work: [
          { id: 'payload-walk', label: 'take one Pallas kernel and name every stage between the Python function and an HLO instruction, in order', href: '#jax-lowers-the-kernel-torch-never-sees-it' },
          { id: 'two-impls', label: 'write the two implementations a kernel op of your own would need, and say exactly what each returns and why', href: '#a-kernel-that-is-also-a-torch-op' },
        ],
      },
    ],
  },
]
