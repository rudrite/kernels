// New file: site/src/data/lessons/torch-bridge.ts
// The torch_xla frontend, below the survey depth chapter 10 teaches: the lazy
// tensor engine and its hash rules, the cut points, the three execution modes,
// and the ComputationClient seam. Every C++ and Python excerpt is verbatim from
// pytorch/xla at commit 41398bf; the one openxla excerpt is from a6c8e17. No TPU
// ran for any of it, and the only throughput numbers are the maintainers' own.
import type { UnitLessons } from './index'

export const TORCH_BRIDGE_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:bridges',
    lessons: [
      {
        id: 'inside-the-lazy-tensor',
        num: 1,
        title: 'Inside the lazy tensor',
        lede: 'A tensor on the xla device is in one of four states, and only one of them holds numbers. What the other three hold is a recording, and how that recording gets fingerprinted decides everything you will later call a recompile.',
        goal: 'Given a torch_xla program, say which of the four states each tensor is in, name every ingredient of the graph hash in the order they merge, and predict from the hash rules alone which edit forces a fresh compile.',
        sections: [
          {
            h: 'four states, one of them numbers',
            ps: [
              "The chapter above this lesson says the operations get recorded rather than run, which is the right sentence to carry around and the wrong place to stop. Open `XLATensor::Data` and the recording turns out to be a four-way state machine, one state per constructor. A tensor holds a device handle, meaning real bytes on the chip. Or it holds an IR value, meaning a pending expression nothing has computed. Or it holds an `at::Tensor` still sitting on the host, uploaded to nothing yet. Or it holds a view, a chain of offsets into some other tensor's storage.",
              "Reading a value forces the first state. Everything else in this arc is about which lines of your program do that reading and what it costs, so it pays to be able to answer, for any tensor in a loop you are debugging, which of the four it is in right now.",
              "The transition into the recording has one detail people trip on. `GetIrValue()` is where a materialized tensor gets lazified: if the tensor currently holds only a device handle, the method mints a `DeviceData` IR node wrapping that handle. It deliberately does not clear the handle afterward. That is why calling it twice returns the same node instead of two parameters pointing at one buffer.",
            ],
          },
          {
            h: 'one node, three hashes',
            ps: [
              "torch_xla does not own a graph walker. It subclasses the LazyTensor core that ships inside PyTorch itself and overrides the parts that need an `xla::Shape` where upstream wanted a `torch::lazy::Shape`. What `XlaNode` adds is worth enumerating, because two of the three additions are hashes and the third is the sharding annotation that feeds one of them.",
              "The interior constructor's four initializers tell you the rule directly. `node_hash_` combines the op kind with a seed. `dag_hash_` folds in every operand's hash, so it identifies the whole subgraph rooted at this node rather than the node alone. Look at what is missing from both: the shape. An interior node's hash does not know its own dimensions.",
              "Leaves are the exception, and `GetOpHash` is where the exception lives. A `DeviceData` or `Scalar` node hashes its op kind together with `shape.ToString()`, so dimensions enter the graph through the leaves and nowhere else. There is a third hash for sharding, updated whenever an annotation lands, folded into `hash()` only when the node actually carries one, so two identical graphs with different shardings can never share a compiled executable.",
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/ir.cpp at 41398bf: the interior constructor at line 67, the leaf hash at 214',
              lang: 'c',
              text: 'XlaNode::XlaNode(torch::lazy::OpKind op, torch::lazy::OpList operands,\n                 std::vector<torch::lazy::Shape>&& shapes, xla::Shape xla_shape,\n                 size_t num_outputs, torch::lazy::hash_t hash_seed)\n    : torch::lazy::Node(op, operands, std::move(shapes), num_outputs),\n      xla_shape_(std::move(xla_shape)),\n      node_hash_(torch::lazy::HashCombine(op.hash(), hash_seed)),\n      dag_hash_(GetOperandHashes(operands, node_hash_)) {}\n\ntorch::lazy::hash_t XlaNode::GetOpHash(torch::lazy::OpKind op,\n                                       const xla::Shape& shape,\n                                       torch::lazy::hash_t hash_seed) {\n  torch::lazy::hash_t h =\n      torch::lazy::HashCombine(op.hash(), torch::lazy::Hash(shape.ToString()));\n  return torch::lazy::HashCombine(h, hash_seed);\n}',
            },
            table: {
              caption: 'the three hashes on an XlaNode, torch_xla/csrc/ir.cpp at 41398bf',
              cols: ['hash', 'built from', 'what it decides'],
              rows: [
                ['node_hash_', 'op kind plus seed; for leaves, op kind plus shape.ToString() plus seed', 'the identity of this one op'],
                ['dag_hash_', 'node_hash_ combined with every operand hash', 'the identity of the whole subgraph under it'],
                ['sharding_hash_', 'tile assignment dims and devices, sharding type, tile shape proto; layout deliberately excluded', 'whether two identically shaped graphs may share an executable'],
              ],
            },
          },
          {
            h: 'so shapes arrive from underneath',
            ps: [
              "Put the two constructors together and a debugging rule falls out. When a graph recompiles after you changed something in the middle of a model, the middle is not where the hash changed. Interior hashes cover op kind and operand structure, so a shape change reaches the graph hash by travelling up from the leaf that owns the shape.",
              "The same rule has a mirror image that surprises people the other way. Two different device buffers of the same shape hash identically, because `DeviceData` uses the leaf constructor and the leaf constructor never saw a buffer address. Buffer identity reaches the compile decision through a separate channel: the parameter sequence, the ordered list of parameter indices recorded as the lowering walks the graph, hashed alongside everything else.",
              "Shape inference itself is memoized in a global cache keyed by the node hash, sized by `XLA_IR_SHAPE_CACHE_SIZE` at a default of 12288 entries. A repeated subgraph shape is not re-inferred, which is a small thing until you are staring at a trace wondering why the second identical block cost nothing.",
            ],
          },
          {
            h: 'the graph hash, in merge order',
            ps: [
              "One function assembles the number that decides compile against cache hit, and reading as far as its first `MergeHash` gives you most of the recompilation story. The very first ingredient is not the graph. It is `config.force_ltc_data`, a boolean about whether this sync is allowed to alias buffers, merged before any tensor is looked at, with a comment saying exactly why.",
              "Next in are the compilation environment hash and two git revisions, the pytorch one and the torch_xla one, generated into the build. Upgrading either library invalidates every entry in your persistent compilation cache, and that is the intent rather than an accident: a bug fixed in torch_xla should not be masked by a stale executable on disk.",
              "After that come the per-tensor ingredients: each synced tensor's IR value hash, the parameter sequence, the buffer donor indices when there are any, and the auto-sharding settings. Absent by design are the bytes in any buffer and the HLO text itself.",
              '>> Two runs with different weights and identical shapes land on the same cache entry. That is the point of hashing structure rather than contents.',
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/xla_graph_executor.cpp:644-655 at 41398bf, inside CollectSyncTensors',
              lang: 'c',
              text: '  // The force_ltc_data controls aliasing compilation, so effectively the same\n  // graph with on/off force_ltc_data should not match, hash wise.\n  coll.hash = torch::lazy::MHash(config.force_ltc_data);\n  // Ensure that the compilation environment and git revisions are reflected\n  // in the hash, so that different versions of the code can produce different\n  // hashes for the same graph.\n  XLA_ASSIGN_OR_THROW(runtime::ComputationClient * absl_nonnull const client,\n                      runtime::GetComputationClient());\n  MergeHash(\n      {client->HashCompilationEnv(), torch::lazy::StringHash(TORCH_GITREV),\n       torch::lazy::StringHash(XLA_GITREV)},\n      &coll.hash);',
            },
          },
          {
            h: 'what the rules predict',
            ps: [
              "You can now derive the recompile list instead of memorizing it. An input shape changes, so a leaf hash changes. A new op appears, or the same ops run in a different order, so the DAG hash changes. A stray tensor is alive at the barrier that was not alive last step, so a different set of IR value hashes merges. A sharding annotation moves. A library upgrade lands. Each of those is one ingredient of one number.",
              "The last entry on the list is not about your program at all. The compiled result lands in an LRU sized by `XLA_COMPILATION_CACHE_SIZE`, default 2048 entries, or in a persistent on-disk cache when `XLA_PERSISTENT_CACHE_PATH` is set. Evict an entry and the next hit becomes a miss, which on the lazy path costs a compile and on the dynamo path costs something worse. The third lesson in this arc takes that apart.",
              "This is the same cache-key discipline chapter 6 taught for dynamo guards, paid a second time at a different layer, against a key you now know the ingredients of. The two layers do not share a key and they do not share a cache, so a program can be perfectly stable under dynamo and still recompile every step down here.",
            ],
          },
        ],
        readings: [
          { label: 'ir.cpp at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/ir.cpp', note: 'the node, the three hashes, and the shape cache, in 342 lines' },
          { label: 'xla_graph_executor.cpp at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/xla_graph_executor.cpp', note: 'CollectSyncTensors at 626 and Compile at 1402 are the two halves of the hash' },
          { label: 'sources of recompilation', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/notes/source_of_recompilation.md', note: "the maintainers' own design essay on why dynamic shapes fix only half of it" },
        ],
        check: [
          {
            q: 'An interior IR node changed its output shape and nothing recompiled. Why not?',
            a: 'Because an interior node hashes its op kind and its operand hashes, not its shape. Shapes enter the graph hash only through leaf DeviceData and Scalar nodes, whose GetOpHash folds in shape.ToString().',
          },
          {
            q: 'Two device buffers with the same shape feed the same graph. What distinguishes them to the cache?',
            a: 'Nothing in their node hashes, which are identical. Buffer identity reaches the graph hash through the parameter sequence, the ordered list of parameter indices recorded while the graph is lowered.',
          },
          {
            q: 'Why are the pytorch and torch_xla git revisions inside every graph hash?',
            a: 'So a version change invalidates the persistent compilation cache. A fix landing in either library should not be masked by an executable compiled before the fix existed.',
          },
        ],
        work: [
          { id: 'hash-ingredients', label: 'write the seven graph-hash ingredients from memory, in merge order, then check them against CollectSyncTensors', href: '#the-graph-hash-in-merge-order' },
          { id: 'four-states', label: 'take one training step you have written and label every tensor in it with which of the four states it is in at the barrier' },
        ],
      },
      {
        id: 'where-the-graph-gets-cut',
        num: 2,
        title: 'Where the graph gets cut',
        lede: 'Nothing in a lazy program runs because you asked it to run. It runs because something needed a value, and the list of things that quietly need a value is longer than the list you would guess.',
        goal: 'Given a training loop, name every line that forces materialization and say why, predict which of those lines shares a compiled graph with the step barrier and which compiles its own, and read a PT_XLA_DEBUG report back to the line that caused it.',
        sections: [
          {
            h: 'the barrier is three calls',
            ps: [
              "`torch_xla.sync()` is a small Python function that hands off to a C++ step marker, and the step marker does three things. It syncs the live tensors graph. It bumps a counter and resets the tracing scopes. Then, only under a debug flag, it prints a report about what just happened.",
              "The first of those is the one with teeth. `SyncLiveTensorsGraph` fetches every live tensor on the device from the arena that tracks them, and syncs all of them with `sync_ltc_data` true. Not the tensors you mentioned; every tensor that still exists. That is the barrier semantics, and it is why an accidentally retained tensor from a previous step changes the graph rather than being ignored.",
              "Two details in the Python are worth having: metrics are saved only on the master ordinal, so you are not reading N copies of the same numbers, and the step closures registered with `add_step_closure` run after the marker, which is what lets several closures share one execution instead of forcing one each.",
            ],
            code: {
              caption: 'verbatim, torch_xla/torch_xla.py:78-96 at 41398bf',
              lang: 'python',
              text: 'def sync(wait: bool = False, reset_scope: bool = True):\n  """Launches all pending graph operations.\n\n  Args:\n    wait (bool): whether to block the current process until the execution finished.\n    reset_scope (bool): whether to reset the torch::lazy::ScopeContext of the IR Nodes.\n  """\n  if xu.getenv_as(\'XLA_EMIT_STEPLOG\', bool, False):\n    print(\'torch_xla.torch_xla::sync\\n\', end=\'\', file=sys.stderr, flush=True)\n  torch_xla._XLAC._xla_step_marker(\n      torch_xla._XLAC._xla_get_default_device(), [],\n      wait=xu.getenv_as(\'XLA_SYNC_WAIT\', bool, wait),\n      reset_scope=reset_scope)\n  # Only emit metrics from the first local device index, to avoid emitting the\n  # same values from different threads.\n  if xm.is_master_ordinal():\n    xm.ms.save_metrics()\n  devctx = xm._run_step_closures()\n  torch_xla._XLAC._set_all_reduce_token(devctx.device, None)',
            },
          },
          {
            h: 'every road leads to one function',
            ps: [
              'Five entry points force materialization and all five funnel into `SyncTensorsGraphInternal`. What separates them is two booleans on a config struct, and those booleans are not bookkeeping: one of them enters the hash, so the same graph reached by two different roads compiles twice.',
              '`sync_ltc_data` says whether the synced tensors should be turned into device data afterward. `force_ltc_data` says whether this sync may alias input buffers into output buffers. A step marker sets both. A `print()` sets neither, because reading a value must not consume the buffer you are about to read.',
            ],
            table: {
              caption: 'the five roads into SyncTensorsGraphInternal, torch_xla/csrc/xla_graph_executor.cpp at 41398bf',
              cols: ['what you wrote', 'where it lands', 'force_ltc_data'],
              rows: [
                ['torch_xla.sync(), xm.mark_step()', 'SyncLiveTensorsGraph, every live tensor on the device (line 430)', 'true'],
                ['.item(), .cpu(), print(), .tolist(), numpy()', 'GetTensors (line 493)', 'false'],
                ['any op at all, in eager mode', 'ApplyEagerSync (line 281), with sync_ltc_data false', 'default'],
                ['the dynamo bridge warming the cache', 'SyncTensorsGraph with warm_up_cache_only (line 421)', 'forced false'],
                ['the dynamo bridge taking a hash', 'GetGraphHash (line 519), hash only, nothing executes', 'false'],
              ],
            },
          },
          {
            h: 'print and sync compile two programs',
            ps: [
              "Since `force_ltc_data` is merged into the hash before any tensor is examined, printing a graph and syncing the same graph land on two different cache entries and pay for two compiles. The comment in the source says so in one line, and the reason is aliasing: a sync at a barrier may donate an input buffer to an output, and a print may not, so the two cannot execute the same module.",
              "The knock-on effect shows up in a shape people write without thinking. Debug a loop by printing a loss every iteration and you have added a second compiled graph to every step, permanently, not just while you are watching. Remove the print and the second entry stops being touched, though it stays in the LRU taking a slot.",
              "The buffer donation half of that story has its own home. Which parameters may be donated, and why the answer is only safe at a barrier, is the fourth lesson in this arc; the twenty-four-line comment inside `GetBufferDonors` is where the source argues it out with two examples.",
            ],
          },
          {
            h: 'the forcers you did not write',
            ps: [
              "Beyond the five explicit roads, a handful of ordinary-looking lines cut a graph. Python control flow on a tensor value is the obvious one: `if x[0][0] == 3` needs a Python bool, so the graph runs to produce it, and the design essay in the repo is blunt that this cannot be fixed without lowering control flow into the graph.",
              "Calling `tensor.size(d)` on a dynamic-shape tensor is the same story with a smaller footprint. The docs put it plainly: the op forces XLA to cut the graph and evaluate, because the alternative is returning the padded shape, which is wrong.",
              "Then there are two that leave no trace in your source at all. Any op without a lowering hits the boxed CPU fallback, which is a device round-trip and therefore a materialization, counted per op in a fallback counter you can read. And `mark_sharding` on a tensor whose IR is a `DeviceData` node downloads the buffer to host and re-uploads it as shards, so an annotation that reads like metadata is a synchronous round-trip. On an intermediate tensor the same call is a pure IR annotation and costs nothing.",
            ],
          },
          {
            h: 'views are left out on purpose',
            ps: [
              "One category of tensor is deliberately excluded from the step barrier, and the eighteen-line comment above `ApplyEagerSync` explains why. View tensors are not synced, because updating the base tensor replays every view on it anyway, so syncing views would compile work that is about to be redone.",
              "The comment states the consequence and accepts it: users who want to print a view still can, and will incur a small graph compile. An extra compile you did not ask for, chosen on purpose to avoid a larger set of compiles you also did not ask for. Worth knowing before you file it as a bug.",
              "Underneath, torch_xla is running two view systems at once. The legacy `View` and `Alias` machinery with its generation counter coexists with PyTorch's own functionalization dispatch key, and roughly a dozen comments in `tensor.h` mark the migration as unfinished. That is the kind of detail that makes a stack trace legible when nothing else does.",
            ],
          },
          {
            h: 'the report that names the cause',
            ps: [
              "torch_xla ships its own explainer for all of this. Set `PT_XLA_DEBUG=1` and every graph execution walks the Python frames and classifies itself into one of eight causes, from the parallel loader's step end to a profiler region exiting to the dynamo bridge processing input graphs. The final else is the one you will meet most often, and its wording is a good summary of the whole lesson.",
              "The report carries the graph name, the graph hash, the input and output counts, and up to eight Python frames, capped by `PT_XLA_DEBUG_MAX_FRAME`. At debug level 1 only compilation analyses print; the environment variable set to 1 actually implies level 100, so the loud version is the default one.",
              "Reading this report is the difference between guessing and knowing which line cut your graph. It is also the mechanism, not just the diagnostic, behind `full_graph=True`, which the next lesson gets to.",
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/debug_util.cpp:369-378 at 41398bf, the last two branches of the classifier, misspelling included',
              lang: 'c',
              text: '  } else if (frames[0].function == "extract_graph_helper" &&\n             endsWith(frames[0].file, "dynamo_bridge.py")) {\n    ss << debug_output_prefix << "  dynamo is compiling a FX graph to HLO\\n";\n  } else {\n    // TODO(JackCaoG): be more specific about  exeuction caused by printing\n    // tensor or fallback or some weird indexing.\n    ss << debug_output_prefix\n       << "  most likely user code trying to access tensor value before "\n          "torch_xla.sync\\n";\n  }',
            },
          },
        ],
        readings: [
          { label: 'torch_xla.py at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/torch_xla.py', note: 'sync, step, compile and device, in 285 readable lines' },
          { label: 'debug_util.cpp at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/debug_util.cpp', note: 'the eight causes, and the error that enforces full_graph' },
          { label: 'recompilation, the practical guide', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/perf/recompilation.md', note: 'the same material as a checklist you can run down' },
        ],
        check: [
          {
            q: 'Why do print(x) and torch_xla.sync() over the same graph cost two compiles?',
            a: 'force_ltc_data differs between them and it is the first thing merged into the graph hash, before any tensor is examined. The two roads cannot share a cache entry because their aliasing decisions differ.',
          },
          {
            q: 'A mark_sharding call on an input tensor stalled the loop. What did it actually do?',
            a: 'It re-sharded physically. When the tensor IR is a DeviceData node, XlaMarkSharding downloads the buffer to host and re-uploads it as shards. On an intermediate tensor the same call is a pure IR annotation.',
          },
          {
            q: 'Printing a view compiled an extra graph. Is that a bug?',
            a: 'No, it is the documented design. View tensors are excluded from step-boundary sync because updating the base replays every view anyway, and the source comment accepts the small extra compile as the price.',
          },
        ],
        work: [
          { id: 'forcer-audit', label: 'take one loop of your own and mark every line that forces materialization, then say for each whether it shares the step graph or compiles its own', href: '#every-road-leads-to-one-function' },
          { id: 'debug-report', label: 'write out what a PT_XLA_DEBUG report would say for a loop that prints its loss, cause line included' },
        ],
      },
      {
        id: 'three-modes-one-machine',
        num: 3,
        title: 'Three modes, one machine',
        lede: 'Eager mode on an XLA device is not a second execution engine. It is one boolean and a call to the same sync function after every op, which is why its cache, its lowering, and its recompiles behave exactly like the lazy path you already know.',
        goal: 'For lazy tracing, eager mode, torch_xla.compile and torch.compile with the openxla backend, name what each changes about the machinery from lesson one, and say which one a given workload should use and what it costs.',
        sections: [
          {
            h: 'eager is a bool and a one-line function',
            ps: [
              "The whole Python surface of eager mode is 32 lines: a setter, a getter, and a context manager that saves and restores. On the C++ side it is one boolean on the graph executor. Three call sites in the tensor layer read that boolean, and when it is set they call `ApplyEagerSync` on the tensors they just created.",
              "`ApplyEagerSync` is one statement. It syncs the given tensors with `wait` false and `sync_ltc_data` false, which is to say it runs the same barrier machinery, per op, without turning everything into device data and without blocking. Same lowering, same builder, same LRU computation cache.",
              "The cache part has an observable consequence the test suite pins. Run `logsumexp` on a 5 by 5 tensor twice and the eager compile counter stays at one while the eager execute counter goes to two. Eager mode is compile-per-op-shape, not compile-per-op. Multi-output ops go further: they construct every output with execution delayed, then call `ApplyEagerSync` once, so a multi-output op costs one compilation rather than N.",
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/xla_graph_executor.cpp:281-283 at 41398bf: the entire difference between eager and lazy',
              lang: 'c',
              text: 'void XLAGraphExecutor::ApplyEagerSync(std::vector<XLATensorPtr>& tensors) {\n  SyncTensorsGraph(&tensors, {}, /*wait=*/false, /*sync_ltc_data=*/false);\n}',
            },
          },
          {
            h: 'what per-op sync costs, in their numbers',
            ps: [
              "The repo publishes a measurement rather than an adjective, and since nothing in this course has a TPU attached, their measurement is the one to quote. A two-layer decoder-only model, fake data, 300 steps, one chip of a v4-8.",
              "The caveat under the table matters more than the table. Pure eager reaches about 45 percent of compiled performance for decoder-only models, and about 1 percent for ResNet50. A number that swings by a factor of forty across two ordinary models is not a performance characteristic you can carry between workloads.",
              "The docs draw the conclusion themselves: eager mode is for data preprocessing, random number generation, custom utilities and debugging, where immediate execution beats throughput. Not for the training loop, and not for the inference loop.",
            ],
            table: {
              caption: 'quoted from docs/source/learn/eager.md at 41398bf; measured by the maintainers on one chip of a v4-8, not on any machine this course touched',
              cols: ['mode', 'token/s'],
              rows: [
                ['tracing mode (base line)', '147'],
                ['eager mode', '65'],
                ['eager + torch_xla compile', '147'],
              ],
            },
          },
          {
            h: 'torch_xla.compile flushes at both ends',
            ps: [
              "The decorator that recovers the lost throughput is a context manager, and its body is a sequence you can read in one screen. It turns eager off inside the region. It names the pending graph, syncs to flush whatever was already queued, then renames the graph to yours, so dataloading does not leak into the step graph and show up as a shape change.",
              "It sets an `allow_execution` flag to the negation of `full_graph`. It optionally opens a dynamic-shape-detector session with a cap on how many distinct graphs the region may produce. Then it yields, and on the way out it restores everything and syncs again.",
              "The `full_graph=True` enforcement is the part worth pausing on, because there is no strict-mode code path. When `allow_execution` is false, any execution inside the region reaches the debug analyzer, which prints an unexpected-execution report and then raises. The diagnostic is the enforcement mechanism. That also means the report you get names the Python frame that broke your full graph, which is more than a strict mode would have told you.",
            ],
            code: {
              caption: 'verbatim, torch_xla/torch_xla.py:190-215 at 41398bf, the body of the compile context manager',
              lang: 'python',
              text: '  @contextlib.contextmanager\n  def _compile():\n    saved_eager_mode_status = torch_xla._XLAC._get_use_eager_mode()\n    saved_allow_execution = torch_xla._XLAC._get_allow_execution()\n    saved_current_graph_name = torch_xla._XLAC._get_current_graph_name()\n    torch_xla._XLAC._set_use_eager_mode(False)\n    if name is not None:\n      torch_xla._XLAC._set_current_graph_name(name + \'_clear_pending\')\n    # Clear pending operations\n    _clear_pending_ops_before_compile()\n\n    if name is not None:\n      torch_xla._XLAC._set_current_graph_name(name)\n\n    # if full_graph sets to true execution can not happen before the sync below\n    torch_xla._XLAC._set_allow_execution(not full_graph)\n\n    if max_different_graphs is not None:\n      torch_xla._XLAC._dynamic_shape_detector_set_max_different_graphs(\n          max_different_graphs)\n      torch_xla._XLAC._dynamic_shape_detector_start_session(current_id)\n\n    try:\n      yield\n    finally:\n      torch_xla._XLAC._set_allow_execution(saved_allow_execution)',
            },
          },
          {
            h: 'the openxla backend is not in this repository',
            ps: [
              "Go looking for where `torch.compile(backend=\"openxla\")` gets registered and you will not find it in pytorch/xla. There is no `register_backend` call for that name anywhere in the tree. Registration lives in PyTorch itself, and the definition is three lines that decide the shape of everything after.",
              "`openxla` is `aot_autograd` wrapping a forward compiler. Because it goes through AOTAutograd, forward and backward reach the torch_xla bridge as separate FX graphs. That single fact is the source of the training-side cost, and the docs state the consequence outright: three graphs per training step instead of one.",
              "What dynamo buys in exchange is real. Per step it skips FX interpretation, ATen dispatch, IR construction, the post-order walk, hashing and cache lookup by graph content, because the bridge holds a precomputed hash and jumps straight to execution. What it still pays per step is a Python loop splitting tensors from symbolic constants, an input materialization check that blocks if any input carries pending IR, an input list rebuild, and a copy back into every in-place-updated argument.",
            ],
            code: {
              caption: 'verbatim from torch/_dynamo/backends/torchxla.py in pytorch itself (definition at line 30, registration at 70-73, joined here); checked against an installed torch 2.2.2 on this machine',
              lang: 'python',
              text: 'def openxla_eval_boxed(model, fake_tensor_inputs):\n    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)\n\n\nopenxla = aot_autograd(\n    fw_compiler=openxla_eval_boxed,\n)\nregister_backend(name="openxla", compiler_fn=openxla)',
            },
          },
          {
            h: 'the cliff at the end of the dynamo path',
            ps: [
              "The lazy path can always recover from a cache miss by lowering the graph again. The dynamo path cannot, and the source says so with a `TODO` next to the check that fails.",
              "`ExecuteComputationWithBarrier` looks its computation up by hash alone. There is no graph left to re-lower, because the whole point was to stop retracing. If the entry has been evicted from the 2048-entry LRU, the run throws. A long job with more distinct shapes than cache slots does not degrade; it stops.",
              "Two smaller cliffs sit next to it. Under SPMD, dynamo cannot see an input sharding change, so the bridge compares shardings for a few steps and then stops checking; a sharding change after that crashes in the execution thread and leaves the result as a placeholder. And a dynamic-shape program does not retrace under dynamo but still compiles a fresh HLO per new shape, so the compile bill arrives even when the trace bill does not.",
              '>> The lazy path can always re-lower. The dynamo path has thrown away the graph, so eviction is an error, not a recompile.',
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/xla_graph_executor.cpp:756-762 at 41398bf, in the dynamo execution path',
              lang: 'c',
              text: '  // TODO implement a fallback mechanism, or make sure those entries\n  // never get kicked out\n  XLA_CHECK(cachedComputation)\n      << "Failed to get computation by hash " << torch::lazy::HashToString(hash)\n      << ". Maybe the entry get "\n         "kicked out of the LRU cache";',
            },
          },
          {
            h: 'which mode, for what',
            ps: [
              "The repo's own recommendation as of this commit splits by workload. Wrap the whole step function in `torch_xla.compile` for training. Use `torch.compile(backend=\"openxla\")` for inference, where it lowers tracing overhead and the three-graph problem does not arise because there is no backward. The docs add that the long-term aim is for `torch.compile` to be the single compilation API for both.",
              "Eager mode sits outside that split, as a debugging and utility mode. It is still `torch_xla.experimental.eager_mode` at this commit, and the docs say it is likely to become the default in future releases. The last lesson in this arc explains why that future release is probably not torch_xla.",
              "Whichever mode you pick, the machinery under it is the one from lesson one. The hash rules do not change, the cut points do not change, and the seam under all three is the same class, which is where this arc goes next.",
            ],
          },
        ],
        readings: [
          { label: 'eager mode, from the maintainers', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/learn/eager.md', note: 'the mode landscape, the benchmark table, and the recommendation, in 168 lines' },
          { label: 'the dynamo integration doc', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/docs/source/perf/dynamo.md', note: 'three graphs per training step, and the gap that blocks larger training' },
          { label: 'dynamo_bridge.py at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/_dynamo/dynamo_bridge.py', note: 'the openxla backend end to end; the clearest inventory of what the lazy path costs per step' },
        ],
        check: [
          {
            q: 'What exactly changes when eager mode is on?',
            a: 'One boolean on the graph executor, read at three tensor-layer call sites that then call ApplyEagerSync, which is one call to SyncTensorsGraph with sync_ltc_data false and wait false. Same lowering, same builder, same LRU cache.',
          },
          {
            q: 'Why does the openxla backend produce three graphs per training step?',
            a: 'Because it is registered as aot_autograd, so forward and backward arrive at the torch_xla bridge as separate FX graphs, with the sync-input graph alongside them. The lazy path gets one graph per step.',
          },
          {
            q: 'What happens when a dynamo graph hash is evicted from the computation cache?',
            a: 'The run throws. ExecuteComputationWithBarrier looks up by hash alone and XLA_CHECKs on the miss, next to a TODO about a fallback. The lazy path never hits this because it can lower the graph again.',
          },
        ],
        work: [
          { id: 'mode-ledger', label: 'write the four modes side by side: what each changes about tracing, what it caches, and what its failure mode is', href: '#which-mode-for-what' },
          { id: 'eviction-math', label: 'work out how many distinct input shapes your own workload would need before it hits the 2048-entry cache ceiling' },
        ],
      },
      {
        id: 'the-seam',
        num: 4,
        title: 'The seam',
        lede: 'The frontend never calls PJRT. It calls one abstract class with two implementations, only one of which is switched on, and every crossing a tensor makes on its way to the chip and back goes through it.',
        goal: 'Name the interface between the torch_xla frontend and any runtime, list the crossings one training step makes and in what order, and say where buffer donation is expressed and where it deliberately is not.',
        sections: [
          {
            h: 'one class, two implementations, one live',
            ps: [
              "`ComputationClient` is a pure-virtual class in `torch_xla/csrc/runtime/computation_client.h`, and it exists because the LazyTensor core above it deals in opaque handles. Its nested `Data` inherits the core's backend data type; its nested `Computation` inherits the core's computation type. The class re-types those generic handles into XLA-shaped ones carrying an `xla::Shape` and an `xla::OpSharding`, and it hides whether the bytes underneath are a PJRT buffer or an IFRT array.",
              "Two subclasses implement it. There is no registry and no dynamic dispatch beyond the vtable; one function picks the implementation once, at first use. Read that function and the choice turns out not to be a choice at this commit. The IFRT client is compiled into the binary, it is a dependency of the runtime build target, and it is unreachable, because the environment variable that used to select it is commented out and the boolean above the branch is a hard-coded false.",
              "What that parked client would have changed is worth knowing without retelling: an IFRT array carries its own sharding across devices where PJRT hands out per-device buffers. The xla path's lesson at /xla/ifrt/above-the-compiler is where that comparison lives. What the torch_xla copy adds is a list of what does not work yet. It refuses non-SPMD compilation outright, `ExecuteComputation` returns unimplemented, and roughly a dozen methods throw, serialization among them, which means no persistent compilation cache.",
            ],
            code: {
              caption: 'verbatim, torch_xla/csrc/runtime/runtime.cpp:27-45 at 41398bf',
              lang: 'c',
              text: '  // TODO: enable IFRT once it\'s not crashing anymore.\n  // Ref: https://github.com/pytorch/xla/pull/8267\n  //\n  // static bool use_ifrt = sys_util::GetEnvBool("XLA_USE_IFRT", false);\n  const bool use_ifrt = false;\n  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") == "") {\n    return XLA_ERROR_WITH_LOCATION(\n        absl::FailedPreconditionError("$PJRT_DEVICE is not set."));\n  }\n\n  ABSL_CHECK(!g_computation_client_initialized)\n      << "ComputationClient can only be initialized once.";\n\n  std::unique_ptr<ComputationClient> client;\n  if (use_ifrt) {\n    XLA_ASSIGN_OR_RETURN(client, IfrtComputationClient::Create());\n  } else {\n    XLA_ASSIGN_OR_RETURN(client, PjRtComputationClient::Create());\n  }',
            },
          },
          {
            h: 'thirteen crossings',
            ps: [
              "Follow one tensor from a torch op to device memory and back and you cross this interface thirteen times, in thirteen distinct places. The table below pairs each frontend site with the client method it calls, at file and line, so you can check every row rather than take the count on faith.",
              "Read the middle column and the shape of the interface shows itself. Compile and execute, transfers in both directions, a placeholder allocator for the asynchronous path, an environment hash, serialization for the persistent cache, two topology queries and one sharding query. That is a runtime contract, not a device API. Nothing in it mentions a stream, a kernel, or a collective.",
              "One honest caveat about the count: the topology queries in particular are called from more places than this table lists, including the Python bindings and the ATen bridge. The table follows one tensor's path rather than inventorying every call site in the tree.",
            ],
            table: {
              caption: 'the crossings on one tensor path, pytorch/xla at 41398bf; paths relative to torch_xla/csrc/',
              cols: ['frontend site', 'client method', 'file:line'],
              rows: [
                ['XLAGraphExecutor::Compile', 'Compile', 'xla_graph_executor.cpp:1498'],
                ['ScheduleSyncTensorsGraph, unsharded branch', 'ExecuteComputation', 'xla_graph_executor.cpp:1159'],
                ['ScheduleSyncTensorsGraph, sharded branch', 'ExecuteReplicated', 'xla_graph_executor.cpp:1144'],
                ['ExecuteComputationWithBarrier, the dynamo path', 'ExecuteReplicated, or the LTC backend ExecuteComputation', 'xla_graph_executor.cpp:865, 873'],
                ['CreateTensorsData, from CollectSyncTensors', 'TransferToDevice, TransferShardsToDevice', 'tensor_util.cpp:572, 842, 880'],
                ['ReleaseGilAndTransferData, from GetTensors', 'TransferFromDevice', 'tensor_util.cpp:921'],
                ['ClearPendingIrs and the dynamo barrier', 'CreateDataPlaceholder', 'xla_graph_executor.cpp:611, 797'],
                ['CollectSyncTensors, building the hash', 'HashCompilationEnv', 'xla_graph_executor.cpp:653'],
                ['CreateComputationCache, persistent cache', 'SerializeComputation, DeserializeComputation', 'xla_graph_executor.cpp:98, 106'],
                ['XLAGraphExecutor::Compile', 'GetCompilationDevices', 'xla_graph_executor.cpp:1466'],
                ['ScheduleSyncTensorsGraph, WaitDeviceOps', 'GetLocalDevices', 'xla_graph_executor.cpp:1134, 481'],
                ['DeviceData::PropagateShardingFromData', 'GetDataSharding', 'ops/device_data.cpp:64'],
                ['torch_xla.set_custom_compile_options', 'SetCustomCompileOptions', 'init_python_bindings.cpp:3344'],
              ],
            },
          },
          {
            h: 'the branch where a step actually leaves',
            ps: [
              "Two of those thirteen rows sit next to each other in one if-else, on a worker thread, inside the lambda the executor schedules. Whether SPMD is on decides which arm runs, and the arms differ in more than a name: the replicated arm hands the client a device list and gets back sharded data, and the single-device arm hands it one device string and options carrying the eager-mode flag from lesson three.",
              "Neither arm blocks. Both return handles to buffers that may not be ready, which the executor then assigns onto placeholders created earlier. Python returns from `sync()` with nothing computed and nothing waited on. The first blocking call comes later, when something reads a value, and it lands on `TransferFromDevice`, whose header comment carries a warning worth internalizing: it blocks until the data is ready, and calling it from Python while holding the GIL can deadlock.",
              "The order for a full step, then, reads: compile if the hash missed, transfer any host-side parameters up, schedule the execute on a worker thread, assign results into placeholders, return. Read a value and only then wait. Everything about this pipeline is designed so the Python thread is never the thing waiting.",
            ],
            code: {
              caption: 'torch_xla/csrc/xla_graph_executor.cpp:1142-1165 at 41398bf, the two arms of the execute branch; every line verbatim, the TF_VLOG logging lines between them trimmed',
              lang: 'c',
              text: '        XLA_ASSIGN_OR_THROW(\n            std::vector<runtime::ComputationClient::DataPtr> outputs,\n            client->ExecuteReplicated(*async->cached_computation->computation,\n                                      UnwrapXlaData(async->parameters_data),\n                                      devices, execute_options));\n        results = WrapXlaData(outputs);\n        TORCH_LAZY_COUNTER("ExecuteReplicated", 1);\n      } else {\n        XLA_ASSIGN_OR_THROW(\n            std::vector<runtime::ComputationClient::DataPtr> outputs,\n            client->ExecuteComputation(*async->cached_computation->computation,\n                                       UnwrapXlaData(async->parameters_data),\n                                       async->device.toString(),\n                                       {/*explode_tuple=*/true,\n                                        /*eager_mode=*/use_eager_mode}));\n        results = WrapXlaData(outputs);\n        TORCH_LAZY_COUNTER("ExecuteComputation", 1);',
            },
          },
          {
            h: 'donation is compiled in, never passed in',
            ps: [
              "Look for a donation list on the execute call and there is none. torch_xla never puts one there. Which parameters may be donated is decided before compilation and written into the module itself, by calling `AddBufferDonor` on the XLA builder for each donor index, so the compiled executable carries the input-output alias as part of its own definition.",
              "PJRT's side of that arrangement is stated in the class comment above `PjRtLoadedExecutable`, and the two quotes below are worth reading as a pair. One side writes the alias into the computation; the other side reads it off the computation and donates the parameter at execution time. The runtime is never told; it is compiled at.",
              "There is a consequence for the hash from lesson one. Because donation changes the module, the donor indices are merged into the graph hash, and they are merged as a sorted set specifically so the ordering cannot make the hash unstable. There is a matching consequence for parameter tupling: when a program crosses 3200 parameters, all parameters get wrapped into one tuple, and the donors are re-emitted against that tuple by shape index instead of by parameter number.",
              "What donation buys you, and what it costs the donated array, is the xla path's story and it is told at /xla/layout-memory. The opt-out on the PJRT side, the non-donatable input indices field on `ExecuteOptions`, is told in the xla path's chapter 14. The torch_xla-specific fact is only this: that field is never the mechanism here.",
            ],
            code: {
              caption: 'two verbatim excerpts under added path headings: torch_xla/csrc/xla_graph_executor.cpp:1392-1400 at 41398bf, then the class comment at xla/pjrt/pjrt_client.h:1386-1390 in openxla/xla at a6c8e17',
              lang: 'c',
              text: '// torch_xla/csrc/xla_graph_executor.cpp: the alias goes into the module\nvoid XLAGraphExecutor::SetBufferDonors(\n    LoweringContext* lowering_ctx,\n    const std::vector<size_t>& buffer_donor_indexs) {\n  for (size_t i : buffer_donor_indexs) {\n    lowering_ctx->builder()->AddBufferDonor(/*param_number=*/i,\n                                            /*param_index=*/{});\n  }\n  TORCH_LAZY_VALUE_METRIC("InputOutputAliasCount", buffer_donor_indexs.size());\n}\n\n// xla/pjrt/pjrt_client.h: and PJRT reads it back off the module\n// Represents a compiled computation that can be executed given handles to\n// device-allocated literals. If any input/output alias has been specified in\n// the computation, the parameter containing the input buffer will be donated\n// when passed to the execution.\nclass PjRtLoadedExecutable {',
            },
          },
          {
            h: 'what never crosses',
            ps: [
              "Two absences from the interface say as much as the thirteen crossings. There is no collective call on it at all. `all_reduce` and its relatives lower into the graph as HLO collectives, so by the time anything reaches the runtime the communication is already an instruction inside the module. The runtime layer's only contribution is the device assignment it built at compile time, plus the coordinator that let a multi-host plugin exchange topology in the first place.",
              "There is no kernel call either, and no stream. Whatever the backend does with warps or vector units or systolic arrays happens entirely behind `Compile` and `Execute`. That is the same boundary the xla path opens at /xla/pjrt/pjrt-boundary from the plugin author's side, and the same one chapter 14's walks step through implementation by implementation, the third of them through the very client this lesson maps. This lesson is the view from directly above it.",
              "One leak is worth naming rather than glossing. `GetPjRtBuffer` sits on the supposedly runtime-neutral interface and returns a PJRT type, because DLPack export needs a raw buffer pointer. The IFRT client throws on it. An abstraction with exactly one working implementation tends to grow a hole shaped like that implementation.",
            ],
          },
        ],
        readings: [
          { label: 'computation_client.h at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/runtime/computation_client.h', note: 'the whole contract: Data, Computation, CompileInstance, then the pure virtuals' },
          { label: 'runtime.cpp at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/torch_xla/csrc/runtime/runtime.cpp', note: '74 lines that pick the client and prove IFRT is off' },
          { label: 'pjrt_client.h at a6c8e17', url: 'https://github.com/openxla/xla/blob/a6c8e1767d219fc6dbc3493f0e02a8b345a2e7b4/xla/pjrt/pjrt_client.h', note: 'the other side of the seam, including the donation semantics comment' },
        ],
        check: [
          {
            q: 'Which class does the torch_xla frontend call to reach a runtime, and how many implementations of it can actually run?',
            a: 'ComputationClient, in torch_xla/csrc/runtime/computation_client.h. Two subclasses exist, PjRt and Ifrt, but the selecting boolean in runtime.cpp is a hard-coded false, so only the PJRT one is reachable.',
          },
          {
            q: 'Where does torch_xla express buffer donation, and where does it not?',
            a: 'In the compiled module, through AddBufferDonor on the XLA builder before compilation. It never passes a donation list in ExecuteOptions; PJRT reads the input-output alias off the executable and donates at execution time.',
          },
          {
            q: 'Why is there no collective call anywhere on the ComputationClient interface?',
            a: 'Because collectives lower into the graph as HLO instructions during tracing. By execution time the communication is inside the module, so the runtime layer only supplies the device assignment and the coordinator behind it.',
          },
        ],
        work: [
          { id: 'thirteen', label: 'check three rows of the crossing table against the source at the pinned commit and confirm the line numbers yourself', href: '#thirteen-crossings' },
          { id: 'donation-trace', label: 'trace one donated parameter from GetBufferDonors through AddBufferDonor to the alias PJRT honours at execute' },
        ],
      },
      {
        id: 'what-survives',
        num: 5,
        title: 'What survives',
        lede: 'The engine you just took apart is scheduled for replacement, and the announcement is at the top of the README in a note dated April 2026. Which means the useful question is which half of this arc describes mechanism and which half describes a boundary.',
        goal: 'State what TorchTPU changes about PyTorch on TPU and what it keeps, sort the four preceding lessons into mechanism that is being replaced and boundary knowledge that outlives it, and justify the sort from the announced design rather than from a guess.',
        sections: [
          {
            h: 'the notice at the top of the file',
            ps: [
              "Read the pytorch/xla README from line one and the third line is a replacement notice. It is worth quoting rather than paraphrasing, because the wording is unambiguous about direction and vague about date, and both halves of that matter.",
              "The tip of `master` at the time this lesson was written was a documentation commit from 23 April 2026, and a remote check on 14 August 2026 confirmed nothing had landed since. Four months without a commit is not by itself a verdict, but with the notice above it, the two readings agree.",
              "The repository has been shedding pieces for a while, so the direction is not new. The XRT runtime is gone entirely, PJRT is the only one left. XLA:CUDA is gone, and a CUDA device type now aborts with a message telling you to report a bug. `xm.mark_step` is a deprecation wrapper around `torch_xla.sync`. torchax, the other bridge chapter 10 names, moved out to its own repository in October 2025 and left only a README behind.",
            ],
            code: {
              caption: 'the first two notices in README.md at 41398bf; wording verbatim, rewrapped to fit, with the two markdown link targets listed underneath',
              lang: 'text',
              text: '> [!NOTE]\n> <b>4/22/2026</b>: To read more on our TorchTPU announcement see our latest\n> blog post. Once TorchTPU is public it will replace PyTorch/XLA.\n> <b>10/2025</b>: Based on community feedback, we have proposed a more native\n> direction for PyTorch on TPU. Read the RFC and comment at #9684.\n\n  blog post: https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/\n  RFC:       https://github.com/pytorch/xla/issues/9684',
            },
          },
          {
            h: 'what the successor proposes',
            ps: [
              "The RFC behind that notice describes a native TPU backend rather than a lazy one. Ops dispatch eagerly through PyTorch's `PrivateUse1` dispatch key, compilation happens asynchronously in the background rather than at a barrier you place by hand, and `torch.compile` and DTensor are the first-class paths instead of `torch_xla.compile` and a torch_xla-specific sharding API.",
              "The blog post fills in the shape: three eager modes, called Debug Eager, Strict Eager and Fused Eager, built on that dispatch key. Notice what that does to the mode landscape from lesson three. Eager stops being a boolean that reroutes a lazy engine and becomes the actual dispatch path, with compilation as the exception rather than the rule.",
              "One thing survives explicitly, and it is the one that matters most for anyone who has read the xla path. XLA is retained, as the `torch.compile` backend, reached through StableHLO. The compiler does not go away. The tracing frontend in front of it does.",
            ],
          },
          {
            h: 'sorting the arc',
            ps: [
              "Lessons one and two are mechanism. Interior nodes excluding shapes from their hashes, `force_ltc_data` merging first, the step barrier syncing every live tensor, views excluded on purpose: all of that is how one specific design solved the problem of turning eager PyTorch into whole compiled graphs. A design that dispatches eagerly and compiles in the background does not need most of it.",
              "Lesson four is boundary. A compile-and-execute interface over an accelerator runtime, with transfers in both directions and asynchronous handles, is not a torch_xla idea. It is what every framework on every accelerator ends up with, which is why the same shape appears at /xla/pjrt/pjrt-boundary from the plugin side and at /xla/ifrt/above-the-compiler one layer up. Donation living inside the compiled module rather than in an execute call is a property of how XLA expresses aliasing, not of how torch_xla drives it.",
              "Lesson three sits across both. `torch_xla.compile` is mechanism and will go. The cache-key discipline underneath it will not, because a compiled program is only reusable if something can decide two invocations are the same, and something has to pay when that decision says no. The eviction cliff is a design bug in one implementation; the fact that a compilation cache has a policy is permanent.",
              '>> The lazy engine is one implementation. The seam is the design, and it is what will still be there afterwards.',
            ],
          },
          {
            h: 'the questions to carry forward',
            ps: [
              "Once TorchTPU is public, this arc becomes a checklist rather than a manual, and these are the entries worth checking first. Does the compilation cache key still carry the framework's git revisions, or has background compilation changed what invalidation means? Is there still a hard failure on cache eviction, or did the fallback in that `TODO` finally get written?",
              "Where does buffer donation live now, given that eager dispatch has no barrier at which aliasing is provably safe? What replaces the step barrier as the place a program's shape is decided, and what is the diagnostic that tells you which line forced a compile, now that the eight-way classifier no longer has a step marker to classify?",
              "Every one of those questions is answerable in an afternoon by someone who has read this arc, and not answerable at all by someone who has only read the migration guide. That is the case for learning a system on its way out: not the API, which expires, but the set of problems the API was solving, which does not.",
            ],
          },
        ],
        readings: [
          { label: 'the TorchTPU RFC', url: 'https://github.com/pytorch/xla/issues/9684', note: 'the native-backend proposal the README notice points at, opened October 2025' },
          { label: 'the TorchTPU announcement', url: 'https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/', note: 'three eager modes on PrivateUse1, with XLA kept as the torch.compile backend' },
          { label: 'pytorch/xla README at 41398bf', url: 'https://github.com/pytorch/xla/blob/41398bfff334fc8d3b1c00be6ea8cc5411f6d6bf/README.md', note: 'the notice, quoted above, in its own context' },
        ],
        check: [
          {
            q: 'What does TorchTPU keep from the stack this arc describes?',
            a: 'XLA itself, retained as the torch.compile backend and reached through StableHLO. What it replaces is the lazy-tensor frontend in front of the compiler, with eager dispatch on the PrivateUse1 key and background compilation.',
          },
          {
            q: 'Which parts of this arc are mechanism that expires, and which are boundary knowledge?',
            a: 'The hash rules, the step barrier and torch_xla.compile are mechanism of one design. The compile-and-execute runtime interface, donation living inside the compiled module, and the existence of a cache-key policy outlive it.',
          },
          {
            q: 'What evidence beyond the README supports reading torch_xla as frozen?',
            a: 'The tip of master was a documentation commit dated 23 April 2026, and a remote check on 14 August 2026 found nothing newer. XRT and XLA:CUDA have already been removed, and torchax left the repository in October 2025.',
          },
        ],
        work: [
          { id: 'sort-the-arc', label: 'sort the four preceding lessons into mechanism and boundary yourself, in writing, and defend any row where you disagree with this lesson', href: '#sorting-the-arc' },
          { id: 'rfc', label: 'read RFC #9684 end to end and note every place it names a problem this arc explained' },
        ],
      },
    ],
  },
]
