// XLA lessons: the compiler at interface and implementation level, one depth
// lesson homed in each unit it deepens. C snippets are quoted from the public
// tree; HLO excerpts are verbatim from this repo's captures (hlo-pairs.json,
// TPU v6 lite, jax 0.11.0; the CPU and TPU pass corpora under data/xla/).
import type { UnitLessons } from './index'

export const XLA_LESSONS: UnitLessons[] = [
  {
    unit: 'xla:pjrt',
    lessons: [
  {
        id: 'pjrt-boundary',
        check: [
          {
            q: 'What is the entire public surface of a PJRT plugin?',
            a: 'One exported symbol, GetPjrtApi, returning a PJRT_Api pointer the caller never owns; everything else is function pointers behind it.',
          },
          {
            q: 'How does JAX find your plugin?',
            a: 'Through jax_plugins namespace packages or entry points: discovery imports the module and calls initialize(), which registers via xb.register_plugin, default priority 400.',
          },
        ],
        num: 1,
        work: [
          { id: 'symbols', label: 'read the cpu plugin version script and find the one global symbol yourself' },
        ],
        title: 'PJRT, the boundary',
        lede: "A PJRT plugin is a shared library that exports exactly one symbol, and every other decision a backend makes hangs off that one function's return value.",
        goal: "Given an empty repository and a working C++ device runtime, name the symbol your plugin must export, the four function pointers a minimal PJRT_Api actually needs, and the two things JAX has to find before jax.devices() returns a device you built.",
        sections: [
          {
            h: 'one exported symbol',
            ps: [
              "The chapter above this lesson opens at the seam and stops there, which is the right altitude for orientation: it names the boundary, names the three nouns on either side of it, and moves on. This lesson starts one line below that, in the symbol table of a plugin's shared object. Read the linker version script XLA ships for its own CPU plugin and the entire public surface of a PJRT backend turns out to be a single name. GetPjrtApi is global. Everything else in the library is local.",
              "The spelling is worth pinning down before you type it from memory, because the tree and its own documentation disagree. Every header in xla/pjrt/c declares the function as GetPjrtApi, lowercase r and t, and both version scripts export it under that spelling. The integration guide's prose calls the step GetPjRtApi. Match the header.",
              "The declaration is three lines, and the comment above it carries an ownership rule that costs you a crash if you miss it. The caller does not take ownership of what comes back, so the struct a plugin returns has to outlive every call made through it. In practice that means a function-local static, built once, returned by pointer forever after.",
            ],
            code: {
              caption: 'verbatim, xla/pjrt/c/pjrt_c_api_cpu.h; the tpu and gpu headers declare the same function',
              text: '#include "xla/pjrt/c/pjrt_c_api.h"\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n\n// Does not pass ownership of returned PJRT_Api* to caller.\nconst PJRT_Api* GetPjrtApi();\n\n#ifdef __cplusplus\n}\n#endif',
              lang: 'c',
            },
          },
          {
            h: 'two routes to the same struct',
            ps: [
              "PJRT_Api is a struct of function pointers, one member per operation, and there are two honest ways to fill it in. The first route implements the C API directly. You write a PJRT_Error* function for every member, you define your own concrete types behind the opaque PJRT_Client and PJRT_Buffer handles, and you hand the finished struct back. Nothing in XLA needs to know your project exists, and nothing in your project needs to link XLA.",
              "The second route is shorter by a few thousand lines, and it is the one the tree's own example plugin takes. Write an ordinary C++ subclass of xla::PjRtClient, the same class an in-tree backend would write, then let XLA generate the C surface around it. pjrt::CreateWrapperClient takes that C++ client and returns the opaque PJRT_Client handle the framework will hold. pjrt::CreatePjrtApi builds the whole PJRT_Api out of XLA's stock implementations, and asks you for only the pieces it cannot invent.",
              "Look at what CreatePjrtApi's parameter list actually demands and the size of a minimal plugin becomes concrete. Four creates: a client, an execute context, a topology description, and a plugin initializer. The last two parameters have defaults, and the example plugin fills the initializer with pjrt::PJRT_Plugin_Initialize_NoOp because it has nothing to set up. Everything else in the struct, every buffer call and every executable call, is XLA's wrapper code forwarding into your C++ client's virtual methods.",
            ],
            code: {
              caption: 'the wrapper route: the signatures in xla/pjrt/c/pjrt_c_api_wrapper_impl.h and the example plugin calling them in xla/pjrt/plugin/example_plugin/myplugin_c_pjrt_internal.cc (one printf line trimmed)',
              text: '// xla/pjrt/c/pjrt_c_api_wrapper_impl.h\nPJRT_Client* CreateWrapperClient(const PJRT_Api* api,\n                                 std::unique_ptr<xla::PjRtClient> cpp_client);\n\nPJRT_Api CreatePjrtApi(\n    PJRT_Client_Create* create_fn,\n    PJRT_ExecuteContext_Create* execute_context_create_fn,\n    PJRT_TopologyDescription_Create* topology_create_fn,\n    PJRT_Plugin_Initialize* plugin_initialize_fn,\n    PJRT_Extension_Base* extension_start = nullptr,\n    PJRT_Plugin_Attributes* plugin_attributes_fn =\n        pjrt::PJRT_Plugin_Attributes_Empty);\n\n// xla/pjrt/plugin/example_plugin/myplugin_c_pjrt_internal.cc\nPJRT_Error* PJRT_MypluginClient_Create(PJRT_Client_Create_Args* args) {\n  std::unique_ptr<xla::PjRtClient> client = CreateMyPluginPjrtClient();\n  args->client =\n      pjrt::CreateWrapperClient(GetMyPluginPjrtApi(), std::move(client));\n  return nullptr;\n}\n\nstatic const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(\n    myplugin_pjrt::PJRT_MypluginClient_Create,\n    myplugin_pjrt::PJRT_MypluginExecuteContext_Create,\n    myplugin_pjrt::PJRT_MypluginDeviceTopology_Create,\n    pjrt::PJRT_Plugin_Initialize_NoOp, &example_extension.base,\n    pjrt::PJRT_Plugin_Attributes_Xla);',
              lang: 'c',
            },
          },
          {
            h: 'the create call, field by field',
            ps: [
              "Every function in the C API takes exactly one argument, a pointer to its own Args struct, and returns PJRT_Error*, where a null pointer means success. The typedef reads `typedef PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args);` and the pattern repeats for all of them. Inputs and outputs both live in the struct, so reading an Args declaration tells you the whole contract of a call without hunting for a return type.",
              "PJRT_Client_Create_Args is worth reading field by field, because three of its members carry a distributed system into a struct that otherwise looks single-host. The kv_get, kv_put, and kv_try_get callbacks are the framework handing your plugin a key-value store it already owns. When a multi-host job starts, that store is how a plugin on host 3 learns what a plugin on host 0 decided, and it is the concrete thing on the far side of the jax.distributed.initialize() call the path meets at /xla/mcjax.",
              "One detail in the field ordering is a small lesson in reading C ABIs. The output field, client, sits in the middle of the struct rather than at the end, with the kv_try_get pair after it. Fields get appended, never inserted, because inserting one would move every field after it and break every already-compiled caller. So the tail of any Args struct is a rough history of what got added last.",
            ],
            table: {
              caption: 'PJRT_Client_Create_Args in declaration order, xla/pjrt/c/pjrt_c_api.h',
              cols: ['field', 'direction', 'what it carries'],
              rows: [
                ['struct_size', 'in', 'the size the caller was compiled against; how both sides survive a struct that grew'],
                ['extension_start', 'in', 'head of the extension chain, or nullptr'],
                ['create_options', 'in', 'a PJRT_NamedValue array of backend-specific options'],
                ['num_options', 'in', 'length of that array'],
                ['kv_get_callback, kv_get_user_arg', 'in', 'blocking read from the framework key-value store'],
                ['kv_put_callback, kv_put_user_arg', 'in', 'write into that same store'],
                ['client', 'out', 'the opaque PJRT_Client the plugin allocates and the framework then owns'],
                ['kv_try_get_callback, kv_try_get_user_arg', 'in', 'non-blocking read, appended after the out field'],
              ],
            },
          },
          {
            h: 'how JAX finds you',
            ps: [
              "A shared object with the right symbol in it is inert until something loads it, and JAX has two ways of finding one. The first is a namespace package: create a directory called jax_plugins, put a module under it, and JAX will import it. The second is packaging metadata, an entry point registered under the group jax_plugins in your pyproject.toml. discover_pjrt_plugins() in jax/_src/xla_bridge.py walks both, and for each thing it finds it calls one function by name, initialize().",
              "What initialize() has to do is call xb.register_plugin. The signature is worth copying exactly rather than approximating, because two of its arguments are alternatives rather than companions: library_path points at the .so JAX should dlopen, and c_api is for a plugin already resident in the process. The default priority is 400, so a plugin that wants to outrank the stock backends passes a higher number.",
              "Priority only decides anything when JAX_PLATFORMS is unset. Set that variable and selection becomes explicit, and it fails loudly rather than quietly falling back to CPU, which is the behaviour you want while a new plugin is still half-written. A plugin that silently loses to CPU looks exactly like a plugin that loaded fine and produced wrong devices.",
            ],
            code: {
              caption: 'the discovery hook, written against the verified register_plugin signature in jax/_src/xla_bridge.py',
              text: '# jax_plugins/my_backend/__init__.py\nfrom jax._src import xla_bridge as xb\n\n\ndef initialize() -> None:\n    xb.register_plugin(\n        "my_backend",\n        priority=500,          # any value above the 400 default outranks cpu\n        library_path="/opt/my_backend/libmy_backend_pjrt_plugin.so",\n        options=None,\n    )\n\n\n# the signature you are calling, from jax/_src/xla_bridge.py:\n#\n# def register_plugin(plugin_name, *, priority=400, library_path=None,\n#                     options=None, c_api=None, factory=None,\n#                     make_topology=None)',
              lang: 'python',
            },
          },
          {
            h: 'versions, and where libtpu sits in exactly this scheme',
            ps: [
              "The header carries its own version in two macros and one struct. PJRT_API_MAJOR is 0 and PJRT_API_MINOR is 114 at the commit this lesson was written against. The major number is incremented when an ABI-incompatible change is made to the interface; the minor is incremented when the interface is updated in a way that is potentially ABI-compatible with older versions. PJRT_Api_Version carries both across the boundary at runtime, so a framework can ask a plugin what it was built against before calling anything newer than that.",
              "That machinery describes an aspiration more than the current rule. The integration guide states the practical constraint plainly: you need to match the jaxlib version with the PJRT C API version, and the recommendation for out-of-tree work is nightly jaxlib from the same date as the XLA commit you built against. ABI stability is described as coming, not arrived. Budget for rebuilding your plugin against each jaxlib you support, and treat struct_size and extension_start as the mechanism that makes that rebuild cheap rather than one that makes it unnecessary.",
              "libtpu is this scheme running in production, which is the useful thing to hold in mind whenever the TPU stack feels like a special case. It is a wheel on PyPI, described by its own publisher as the Google Cloud TPU runtime library, providing core functionality for compilation, inter-chip communication, and runtime execution. Version 0.0.45 went out on 2026-08-01 at 215.9 MB per wheel. JAX finds it through the same plugin discovery as anything else, dlopens it, and calls the same three-line GetPjrtApi that the CPU header declares.",
              '>> The four-line header the CPU plugin exports and the 215 MB TPU wheel present the identical symbol.',
              "So the difference between a toy plugin and libtpu is not the interface. It is what sits behind the function pointers, which is a compiler and a runtime for a chip nobody outside Google can read. The codegen unit's lesson at /xla/codegen/backend-seam takes that seam apart properly.",
            ],
          },
        ],
        readings: [
          { label: 'the PJRT C API header', url: 'https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h', note: 'every struct this lesson quotes, in its own words' },
          { label: 'PJRT integration guide', url: 'https://github.com/openxla/xla/blob/main/docs/pjrt/pjrt_integration.md', note: 'three steps, from the people who review the plugins' },
          { label: 'PJRT on openxla.org', url: 'https://openxla.org/xla/pjrt', note: 'the same seam, one altitude higher' },
        ],
      },
    ],
  },
  {
    unit: 'xla:hlo',
    lessons: [
  {
        id: 'hlo-invariants',
        check: [
          {
            q: 'What does StableHLO have that HLO does not?',
            a: 'A written grammar, a spec, and versioned serialization. HLO has a C++ class hierarchy and a set of invariants passes must leave standing.',
          },
          {
            q: 'In bf16[1024,512]{1,0:T(8,128)(2,1)S(1)}, what are the last two fragments?',
            a: 'The (2,1) is the packing tile that appears only on 16-bit types in the capture; S(1) is a memory space, reached through a copy-start and copy-done pair.',
          },
        ],
        num: 1,
        work: [
          { id: 'decode', label: 'decode one full shape brace from the repo capture, fragment by fragment' },
        ],
        title: 'HLO and its invariants',
        lede: "StableHLO arrives at the door with a written grammar. HLO inside has no grammar at all, only a set of invariants that two hundred passes are required to leave standing.",
        goal: "Given any dumped module, tell StableHLO from HLO on sight, decode a shape annotation down to its tiling and memory space, and name which invariant a given analysis is relying on when it reads that instruction.",
        sections: [
          {
            h: 'two representations, one with a grammar',
            ps: [
              "The path separates these two at /xla/ingestion, and the chapter above this lesson reads HLO; the split they teach is correct: StableHLO is the wire format, HLO is the workbench. What that pairing leaves out is how differently the two are specified, which turns out to matter every time you go looking for the rules.",
              "StableHLO has a spec with a grammar in it. Programs are a sequence of functions, `Program ::= {Func}`, each function written `func.func @name(inputs) -> (outputs) { body }`. Operations are named `stablehlo.mnemonic`, they consume values and attributes and produce outputs, and the type system is written down: tensor types have a shape and an element type, shapes cannot be unranked, and quantized element types map an integer storage range onto an expressed floating-point type. Versioning lives in a separate dialect, VHLO, which is what lets a serialized program survive a compiler upgrade.",
              "HLO has none of that. There is no published grammar for the text format, no versioned dialect, and no stability promise about how a module prints. What HLO has instead is a C++ class hierarchy and a set of invariants those classes hold. HloModule owns its computations and names one of them the entry. HloComputation owns an ordered list of instructions and marks exactly one ROOT. Every HloInstruction carries a Shape from the moment it is constructed. Passes are allowed to rewrite anything they like as long as those three statements stay true afterwards.",
            ],
            code: {
              caption: 'StableHLO at the door: the head of naive attention before any pass runs (site/src/data/hlo-pairs.json, TPU v6 lite, jax 0.11.0)',
              text: 'module @jit_naive_attention attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n  func.func public @main(%arg0: tensor<1024x128xbf16>, %arg1: tensor<1024x128xbf16>, %arg2: tensor<1024x128xbf16>) -> (tensor<1024x128xbf16> {jax.result_info = "result"}) {\n    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<1024x128xbf16>) -> tensor<128x1024xbf16>\n    %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x128xbf16>, tensor<128x1024xbf16>) -> tensor<1024x1024xbf16>\n    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>\n    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<1024x1024xbf16>, tensor<bf16>) -> tensor<1024xbf16>',
              lang: 'mlir',
            },
          },
          {
            h: 'what a name promises',
            ps: [
              "Read a few HLO computations and the shape of the invariant set shows up in the syntax itself. A name is defined once, on the left of exactly one instruction, and referenced by name everywhere it is used. That single-definition property is what makes dataflow analysis a matter of following names rather than reasoning about mutation, and it is why the path can say at /xla/hlo that fusion depends on dataflow analysis without that sounding like a heavy claim.",
              "The second promise is that no pass ever has to infer a type. The shape is printed on every instruction because it is stored on every instruction. A pass that wants to know whether two operands match reads two Shape objects and compares them; it does not walk backwards through the graph reconstructing what something must be. That is a real design decision with a cost, since every rewrite has to compute and set the new shape explicitly, and a real payoff, since a partially rewritten module is still fully typed.",
              "The third is that structure nests through named computations rather than through blocks. A reduce carries `to_apply=%region_0.1` and the region is a separate computation printed alongside. A fusion carries `calls=%fused_computation.1` the same way. So a computation you find in a dump is either the entry, or something named by an instruction elsewhere, and searching the module for its name finds the caller in one step.",
            ],
            code: {
              caption: "one reduce region, verbatim from the repo's own TPU capture of row softmax",
              text: '%region_0.1 (reduce_max.3: bf16[], reduce_max.4: bf16[]) -> bf16[] {\n %reduce_max.4 = bf16[]{:T(256)} parameter(1), metadata={op_name="reduce_max"}\n %reduce_max.3 = bf16[]{:T(256)} parameter(0), metadata={op_name="reduce_max"}\n ROOT %reduce_max.5 = bf16[]{:T(256)} maximum(%reduce_max.3, %reduce_max.4), metadata={op_name="jit(row_softmax)/reduce_max" stack_frame_id=20}\n}',
              lang: 'text',
            },
          },
          {
            h: 'decoding a shape all the way down',
            ps: [
              "The path teaches the layout brace at /xla/layout-memory and stops at the minor-to-major list, which is the part that changes strides. A real TPU dump prints more inside that brace, and the rest of it is readable too. Take `bf16[1024,512]{1,0:T(8,128)(2,1)S(1)}` and pull it apart left to right.",
              "The interesting fragments are the two after the colon. The tiles are a physical blocking of the last dimensions, and the second one is easy to identify from the capture itself rather than from documentation: every bf16 shape in this module carries `T(8,128)(2,1)` and every f32 shape carries `T(8,128)` alone. One tile appears only on the 16-bit type, which is what a packing tile for two 16-bit values looks like. Scalars get a different tile entirely, `T(256)`, as the reduce region above shows.",
              "S(1) is a memory space, and the capture proves it is not decoration. In the softmax module a `copy-start` takes `%x.1` in the default space and produces a shape that differs from its input in exactly one way, the trailing S(1); a matching `copy-done` completes it. Same element type, same dimensions, same layout, same tiles. A copy whose only visible effect is the memory space is a move between memories, and it is scheduled as an async pair because it takes real time.",
            ],
            table: {
              caption: 'bf16[1024,512]{1,0:T(8,128)(2,1)S(1)}, fragment by fragment',
              cols: ['fragment', 'what it says'],
              rows: [
                ['bf16', 'element type'],
                ['[1024,512]', 'logical dimensions, major to minor as written'],
                ['{1,0', 'minor-to-major order: dimension 1 varies fastest, so rows are contiguous'],
                ['T(8,128)', 'physical tiling of the last two dimensions'],
                ['(2,1)', 'a second tile present on every bf16 shape in this capture and on no f32 shape in it'],
                ['S(1)', 'memory space 1, reached in this module through a copy-start and copy-done pair'],
              ],
            },
          },
          {
            h: 'the header line is a contract with PJRT',
            ps: [
              "Every dumped HLO module opens with one long line that most readers skip, and two of its fields are worth not skipping. `entry_computation_layout={(bf16[1024,512]{1,0:T(8,128)(2,1)})->bf16[1024,512]{1,0:T(8,128)(2,1)}}` is the promise the compiled program makes to whoever calls it, stated in exactly the vocabulary the previous section decoded. That promise is what a PjRtBuffer has to satisfy before Execute will accept it, which is where the pjrt lesson and this one meet.",
              "`is_scheduled=true` changes how you read everything below it. A module dumped before scheduling lists instructions in an order that only has to respect dataflow; a scheduled module lists them in the order they will actually run. So in a scheduled dump the distance between a `copy-start` and its `copy-done` is not a formatting accident, it is the window the scheduler chose to overlap that transfer against, and counting what sits between them tells you how much compute is covering the move.",
              "The rest of the header line is sharding and frontend bookkeeping: `allow_spmd_sharding_propagation_to_parameters`, `arg_layout_modes`, `arg_memory_spaces`. On a single-device capture these read as trivia. They stop being trivia the moment the spmd unit's lesson gets to a partitioned module, because they are the flags that decided whether the partitioner was allowed to change your inputs' shardings at all.",
            ],
            code: {
              caption: 'the header line of the optimized row-softmax module, verbatim',
              text: 'HloModule jit_row_softmax, is_scheduled=true, entry_computation_layout={(bf16[1024,512]{1,0:T(8,128)(2,1)})->bf16[1024,512]{1,0:T(8,128)(2,1)}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}, frontend_attributes={arg_layout_modes="default",arg_memory_spaces="0",out_layout_modes="default",out_memory_spaces="0"}',
              lang: 'text',
            },
          },
          {
            h: 'fluency, drilled on one capture',
            ps: [
              "Reading fluency here is not a matter of understanding, it is a matter of having done it enough times that the eye stops stalling. The three captures in this repo are enough material: naive attention, an MLP, and a row softmax, each dumped before and after optimization on a TPU v6 lite. The routine that pays off is to start at ROOT in the ENTRY computation and walk backwards through operands, which reconstructs the program in the order the compiler thinks about it.",
              "Doing that on the softmax capture turns up a naming trap worth meeting once, deliberately, instead of by accident. The ROOT instruction is `%fusion` and it calls `%fused_computation.1`. The instruction feeding it is `%fusion.1` and that one calls `%fused_computation.2`. Instruction suffixes and computation suffixes are independent uniquifying counters, and a fusion numbered 1 has no particular relationship to a computation numbered 1.",
              '>> The instruction number and the computation number are two different counters.',
              "The rest of what you need is repetition against real modules, which is what the drills on /gym/xla exist for. The one thing worth carrying out of this lesson in advance is the habit of decoding the full brace rather than the first two numbers, because on a TPU capture the tiles and the memory space are where the interesting decisions ended up.",
            ],
          },
        ],
        readings: [
          { label: 'the StableHLO spec', url: 'https://openxla.org/stablehlo/spec', note: 'the grammar and type system HLO does not have' },
          { label: 'HLO operation semantics', url: 'https://openxla.org/xla/operation_semantics', note: 'op by op, including dot_general dimension numbers' },
          { label: 'XLA architecture', url: 'https://openxla.org/xla/architecture', note: 'where the two representations sit in the flow' },
        ],
      },
    ],
  },
  {
    unit: 'xla:pipeline',
    lessons: [
  {
        id: 'pass-pipeline',
        check: [
          {
            q: "What does a pass's Run return, and what pipeline behavior does that enable?",
            a: 'StatusOr of bool, the bool meaning whether the module changed; a pipeline can rerun its members until every one returns false, which is why algsimp repeats in dumps.',
          },
          {
            q: 'Why diff the modules before timing a disable flag?',
            a: 'A flag can be accepted, name a pass that exists, and still change nothing for your program; the byte-level diff catches that in seconds where a benchmark would mislead for an afternoon.',
          },
        ],
        num: 1,
        work: [
          { id: 'dump', label: 'dump any jit function with xla_dump_to and draw the pipeline tree from filenames alone' },
        ],
        title: 'The pass pipeline, from a real dump',
        lede: "Two hundred passes is a number people repeat. A dump directory turns it into a list of filenames you can read end to end in an afternoon.",
        goal: "Given a dump directory from any backend, reconstruct the pipeline nesting from filenames alone, find the pass that introduced a given instruction, and prove that a --xla_disable_hlo_passes flag actually changed something before you time anything.",
        sections: [
          {
            h: 'what a pass is, exactly',
            ps: [
              "A pass in XLA is a class with two members that matter. `name()` returns the string the dump will print, and `Run()` does the work and reports whether it changed anything. The return type is what makes the rest of the pipeline's behaviour legible: `absl::StatusOr<bool>`, where the comment in the header says the bool means whether it modified the module.",
              "That bool is the reason a dump has repeats in it. A pipeline can be wrapped so that it reruns its member passes until every one of them returns false, and the chapter above notices this when algsimp shows up several times in a row. The mechanism behind the observation is a loop reading return values, and there is a companion entry point for it, `RunOnChangedComputations`, which lets a pass in a fixpoint loop revisit only what the previous iteration touched instead of the whole module.",
              "Most passes derive from HloModulePass rather than implementing the interface directly. The distinction is about scope: a module pass gets the whole HloModule and may rewrite across computation boundaries, which is what layout assignment and buffer assignment need. Knowing which base a pass uses tells you, before reading a line of its body, whether it could possibly have caused a change you found two computations away.",
            ],
            code: {
              caption: 'verbatim, trimmed, from xla/hlo/pass/hlo_pass_interface.h',
              text: 'class HloPassInterface {\n public:\n  virtual absl::string_view name() const = 0;\n\n  // Run the pass on the given HLO module ...\n  // Returns whether it modified the module.\n  absl::StatusOr<bool> Run(\n      HloModule* module,\n      const absl::flat_hash_set<absl::string_view>& execution_threads = {});\n\n  // Run the pass on computations changed from last iteration in given HLO\n  // module.\n  // ...\n};\n\nclass HloModulePass : public HloPassInterface {\n  // ...\n};',
              lang: 'c',
            },
          },
          {
            h: 'the flags, quoted rather than remembered',
            ps: [
              "The dump flags are defined in xla/debug_options_flags.cc with help strings, and the help strings answer questions the tools page does not. `--xla_dump_hlo_pass_re` dumps HLO before and after optimization passes which match this regular expression, in addition to dumping at the very beginning and end of compilation. Two facts follow from that sentence: the endpoints are always dumped whether or not your regex matches anything, and the per-pass dumps come in before-and-after pairs rather than snapshots.",
              "`--xla_dump_to` takes a directory and has two special values. The literal `-` means stdout, and `sponge` or `test_undeclared_outputs_dir` route into the directory named by the TEST_UNDECLARED_OUTPUTS_DIR environment variable. If you point it at a path that does not exist you get nothing and no complaint, which is the most common reason a first dump attempt appears to do nothing at all.",
              "The disabling flags carry a warning in their own help text that the path had to learn the hard way. `--xla_disable_hlo_passes` takes a comma-separated list, and the help says these names must exactly match the passes' names, with no whitespace around commas. Nothing in that contract promises the compiler will tell you when a name matched nothing. The retraction on /xla/fusion is exactly that failure, published and then corrected.",
            ],
            table: {
              caption: 'the dump and pass-selection flags, from their help strings in xla/debug_options_flags.cc',
              cols: ['flag', 'what its help string says'],
              rows: [
                ['--xla_dump_to', 'directory for debugging data; "-" means stdout, "sponge" uses TEST_UNDECLARED_OUTPUTS_DIR'],
                ['--xla_dump_hlo_pass_re', 'dump before and after passes matching this regex, on top of the always-dumped endpoints'],
                ['--xla_dump_hlo_module_re', 'limit dumping to modules matching this regex; default is every module'],
                ['--xla_dump_hlo_as_text', 'modules as text before and after optimizations'],
                ['--xla_dump_hlo_as_proto', 'modules as HloProtos into the dump directory'],
                ['--xla_dump_hlo_as_html', 'modules rendered as HTML files'],
                ['--xla_dump_hlo_as_dot', 'modules rendered as dot files'],
                ['--xla_disable_hlo_passes', 'comma-separated names, exact match required, no whitespace around commas'],
                ['--xla_enable_hlo_passes_only', 'the inverse: named passes on, everything unspecified off'],
              ],
            },
          },
          {
            h: 'the filename is the pipeline',
            ps: [
              "A dump filename encodes four things, and once you can read all four you no longer need to open most of the files. The parts are a zero-padded module number, the module name, a zero-padded step number, and then the interesting half: which pipeline the step belongs to, which pass just finished, and which pass is about to run. `HLO_passes_through_layout_assignment.after_cse.before_cse_barrier_expander` is a complete sentence about where the compiler was.",
              "Nesting is recoverable from that alone, without reading any source. In this repo's CPU capture, step 15 sits in pipeline `HLO_passes_after_layout_assignment` with `before_after_layout_assignment`, and step 16 sits in pipeline `after_layout_assignment` going from `pipeline-start` to `pipeline-end`. A name that appears as a pass in one line and as the pipeline in the next is a nested pipeline being entered. Do that for a whole dump and you have drawn the tree.",
              "The other thing the filenames give you free is the fixpoint. Steps 3 through 8 of that capture are `simplification` running `algsimp`, then `simplify-sorts`, then `algsimp` again from `pipeline-start`, twice more. Three entries into the same pipeline from its start is three iterations of the loop the previous section described, and the loop stopped when the third pass produced no change.",
              '>> A nested pipeline announces itself twice: once as a pass name, then as the pipeline on the next line.',
              "One backend-specific trap costs an afternoon if you meet it cold. TPU dumps insert a build id between the module name and the step number, as in `module_0007.jit_attend.cl_948136882.0000.hlo_device_type_async_wrapper`, while CPU dumps go straight from module name to step. A filename pattern written against one backend matches nothing on the other, and a diff of nothing against something looks exactly like a real result. List the filenames before you parse them.",
            ],
            code: {
              caption: "the CPU pipeline as filenames, from this repo's own capture (site/src/data/xla/pipeline-corpus.json, jax 0.4.38, attention 64x64)",
              text: '  n  pipeline                                after            -> before\n  0  sharding-removal                        pipeline-start      sharding-remover\n  1  SubbytePacker_pipeline                  pipeline-start      sub-byte-size-setter\n  2  HLO_passes_through_layout_assignment    pipeline-start      gather_scatter_normalizer\n  3  simplification                          pipeline-start      algsimp\n  4  simplification                          algsimp             simplify-sorts\n  5  simplification                          tree_reduction_rewriter  zero_sized_hlo_elimination\n  6  simplification                          pipeline-start      algsimp\n  8  simplification                          pipeline-start      algsimp\n 12  HLO_passes_through_layout_assignment    flatten-call-graph  layout-assignment\n 13  HLO_passes_through_layout_assignment    layout-assignment   sub-byte-size-setter\n 15  HLO_passes_after_layout_assignment      pipeline-start      after_layout_assignment\n 16  after_layout_assignment                 pipeline-start      pipeline-end\n 17  HLO_passes_after_layout_assignment      fusion              simplification_after_layout_assignment\n 22  HLO_passes_after_layout_assignment      copy-insertion      dce',
              lang: 'text',
            },
          },
          {
            h: 'the reading protocol EX·15 stops short of',
            ps: [
              "The chapter above ships an instrument, EX·15, that steps twenty of these files so you can scrub the sequence, watch the reruns appear, and find the watershed at layout assignment. Stepping is the first half of the skill. The second half is a loop you run against your own dump, and it is four steps, in this order, because each one prevents a mistake the next one would otherwise hide.",
              "List the filenames first and read the pattern off them, rather than assuming the pattern from another backend. Then grep the pass vocabulary before you name a pass in a flag, because names are backend-specific and a name lifted from a CPU dump can be simply absent from a TPU pipeline. Then diff the pair around a pass and count instructions rather than lines, since a metadata-only reprint can produce a hundred changed lines and zero changed program. Only then, if you disabled something, time it.",
              "That third step is where this repo has receipts. Its pass corpus records that both the CPU and TPU pipelines register a pass called exactly `fusion`, and that disabling `fusion` on the TPU leaves an attention module byte-identical, while `tpu_fusion` and `tpu_multi_output_fusion` change it. So a flag can be accepted, name a pass that genuinely exists, and still be a no-op for your program. Diffing the compiled module is the only step that catches that, and it costs ten seconds.",
              "Read the same corpus the other direction and it doubles as a vocabulary map. Names like `add-random-host-offloading`, `tpu-embedding-thread-annotator`, and `legalize-scheduling-annotations` appear only on TPU; `tree_reduction_rewriter`, `cse_barrier_expander`, and `sharding-remover` only on CPU. Two names, `fusion` and `bitcast_dtypes_expander`, appear in both. That overlap is small enough to be worth internalizing: almost nothing you learn about one backend's pipeline transfers to another by name.",
            ],
          },
          {
            h: 'from a printed name back to the source file',
            ps: [
              "The last skill in this lesson is turning a name in a dump into a file you can open. Hardware-independent passes live under xla/hlo/transforms, which has subdirectories for simplifiers, expanders, and collectives, plus a README and its own BUILD file. Backend passes live under xla/backends/<backend>/transforms. If a grep through xla/hlo/transforms comes back empty for a name your dump printed, that is usually the answer rather than a mystery: the pass belongs to a backend.",
              "The name-to-filename mapping is by hand, in both directions, because there is no consistent convention. The printed string is whatever the pass's `name()` returns, and the corpus above shows both styles side by side: `layout-assignment`, `transpose-folding`, `flatten-call-graph`, and `copy-insertion` print with hyphens, while `tree_reduction_rewriter`, `zero_sized_hlo_elimination`, and `gather_scatter_normalizer` print with underscores. Filenames are snake_case regardless. Convert by hand and search both spellings.",
              "Browsing that directory once is worth more than it sounds, because the file list is a catalogue of what a compiler at this level actually worries about. Host offloading gets three passes of its own (`host_offloader`, `host_offload_legalize`, `host_offloading_prepare`). There is a `bfloat16_propagation` and an `operand_upcaster` for precision, a `while_loop_trip_count_annotator` for loops, and a `defuser` that undoes fusion, which exists because debugging and testing need a way back out.",
            ],
          },
        ],
        readings: [
          { label: 'HloPassInterface', url: 'https://github.com/openxla/xla/blob/main/xla/hlo/pass/hlo_pass_interface.h', note: 'the two virtuals every pass implements' },
          { label: 'xla/hlo/transforms', url: 'https://github.com/openxla/xla/tree/main/xla/hlo/transforms', note: 'the hardware-independent passes, as a directory' },
          { label: 'XLA tools', url: 'https://openxla.org/xla/tools', note: 'hlo-opt, run_hlo_module, and the dump flags in context' },
        ],
      },
    ],
  },
  {
    unit: 'xla:fusion',
    lessons: [
  {
        id: 'reading-fusion',
        check: [
          {
            q: 'How do you match a fusion instruction to the computation it runs?',
            a: 'Only through its calls= field. The numeric suffixes are independent counters, and matching fusion.N to fused_computation.N by number will be wrong more often than right.',
          },
          {
            q: 'What are the three ways a merge does not happen?',
            a: 'No producer-consumer edge exists in the graph; the priority queue judged the merge unprofitable; or something unfusable sits between the two, like a memory-space copy.',
          },
        ],
        num: 1,
        work: [
          { id: 'trace', label: 'follow every calls= link in the softmax capture and diagram the fusion graph' },
        ],
        title: 'Fusion',
        lede: "The compiler writes down which kind of fusion it made and which computation that fusion calls, and both facts are sitting in the dump you already have.",
        goal: "Given an optimized module, classify every fusion by its FusionKind, trace each fusion instruction to the computation it calls, and say which of the two gates, fusibility or profitability, rejected a merge that did not happen.",
        sections: [
          {
            h: 'four kinds, named in the enum',
            ps: [
              "The chapter above makes the load-bearing argument, that fusion changes where intermediates live and never changes the math, and it is the right thing to hold onto. What a chapter does not have room for is the vocabulary, and the vocabulary is small enough to learn in one sitting. XLA distinguishes four kinds of fusion, and it prints the kind on every fusion instruction it emits.",
              "kLoop is the default shape: the fused computation's root is the primary node, and codegen emits an element-at-a-time loop over it. kInput also has the primary node at the root but generates code close to what the unfused version would produce, which is the shape reductions take. kOutput is the one where the primary node is not the root, and it carries a constraint the other kinds do not: one operand buffer has to alias the output buffer. kCustom is the backend's own category for anything that fits none of these.",
              "In this repo's TPU capture, every fusion in all three programs prints `kind=kLoop`, including the ones that exist only to wrap a single bitcast. That uniformity is itself informative: on this backend and these programs, the interesting variation is not in the kind field, it is in what got pulled inside the computation.",
            ],
            code: {
              caption: 'verbatim, xla/hlo/ir/hlo_instruction.h',
              text: 'enum class FusionKind {\n  kLoop,\n  kInput,\n  kOutput,\n  kCustom,\n};',
              lang: 'c',
            },
          },
          {
            h: 'the queue, and the formula that orders it',
            ps: [
              "Fusion on GPU runs as a priority queue, and the priority has an actual formula written in the header. PriorityFusion is an HloModulePass whose `name()` returns `priority-fusion`, and its documented algorithm is: compute `priority = time_unfused - time_fused` for each producer, put the producers with positive benefit into a queue ordered by that benefit, then pop the top one, fuse it, and update the priorities of everything the fusion touched.",
              "Read the formula rather than the word and one common mistake disappears. The queue is ordered by estimated time saved, not by how many operations a merge absorbs. A fusion that swallows one enormous intermediate outranks a fusion that swallows six small ones, because the difference of two time estimates is dominated by the memory traffic the merge removed.",
              '>> The queue is ordered by time saved, not by operations absorbed.',
              "The private members of the pass name the machinery behind the estimate. `GpuHloCostAnalysis::Options cost_analysis_options_` is described in the header as what defines the priorities in the queue, and `HloFusionAnalysisCache fusion_analysis_cache_` keeps that analysis from being recomputed for every candidate as the graph changes underneath it. There is a third member worth knowing about as a debugging surface: `std::unique_ptr<FusionProcessDumpProto> fusion_process_dump_`, which logs the decisions the queue made. Fusion decisions are dumpable as their own artifact, separately from the HLO text.",
            ],
          },
          {
            h: 'reading fusion names in a dump',
            ps: [
              "A fusion appears twice in a module and the two appearances look confusingly similar. There is an instruction, printed inside a computation, that looks like `%fusion.1 = bf16[1024] fusion(%copy-done, %reduce_max.7), kind=kLoop, calls=%fused_computation.2`. And there is the computation itself, printed at module scope with its own parameter list and its own ROOT. The `calls=` field is the only link between them.",
              "The numbers on the two are independent counters, which is the trap worth meeting deliberately. In the softmax capture, the ROOT instruction `%fusion` calls `%fused_computation.1`, and the instruction feeding it, `%fusion.1`, calls `%fused_computation.2`. Matching a `fusion.42` to a `fused_computation.42` by number will be wrong more often than right. Follow the `calls=` field.",
              "Once you are inside the computation, the parameter list tells you what the fusion actually reads, which is more useful than the instruction's operand list because it is annotated with shapes. `%fused_computation.2 (param_0.9: bf16[1024,512], param_1.10: bf16[1024])` says this fusion consumes the whole matrix and a vector of row maxima. The ROOT tells you what it produces and, in this case, what anchored it: `ROOT %reduce.1 = bf16[1024] reduce(%convert.7, %constant.6), dimensions={1}, to_apply=%region_1.2.clone`.",
            ],
            code: {
              caption: "one fusion, verbatim from the repo's TPU capture of row softmax, backend_config blocks trimmed",
              text: '%fused_computation.2 (param_0.9: bf16[1024,512], param_1.10: bf16[1024]) -> bf16[1024] {\n %param_0.9 = bf16[1024,512]{1,0:T(8,128)(2,1)S(1)} parameter(0)\n %convert.5 = f32[1024,512]{1,0:T(8,128)} convert(%param_0.9)\n %param_1.10 = bf16[1024]{0:T(1024)(128)(2,1)S(1)} parameter(1)\n %sub.10 = bf16[1024,512]{1,0:T(8,128)(2,1)} broadcast(%param_1.10), dimensions={0}\n %convert.6 = f32[1024,512]{1,0:T(8,128)} convert(%sub.10)\n %sub.8 = f32[1024,512]{1,0:T(8,128)} subtract(%convert.5, %convert.6)\n %exp.5 = f32[1024,512]{1,0:T(8,128)} exponential(%sub.8)\n %convert.7 = bf16[1024,512]{1,0:T(8,128)(2,1)} convert(%exp.5)\n %constant.6 = bf16[]{:T(256)} constant(0)\n ROOT %reduce.1 = bf16[1024]{0:T(1024)(128)(2,1)S(1)} reduce(%convert.7, %constant.6), dimensions={1}, to_apply=%region_1.2.clone\n}',
              lang: 'text',
            },
          },
          {
            h: 'what a reduction does to a fusion',
            ps: [
              "The computation above is worth one more pass, because it shows the general shape of what fusion buys on a memory-bound program. Six instructions went inside: a convert up to f32, a broadcast of the row maxima, a convert, a subtract, an exponential, a convert back down to bf16. Then a reduce. Every one of those intermediates is a full 1024 by 512 array that now never exists in memory, because the loop that computes them feeds the reduction directly.",
              "The reduction is the root, and that is the pattern rather than a coincidence. This repo's own note on the softmax capture says it plainly: reductions become fusion roots, and the max and the sum each anchor one. A reduction is a natural stopping point because its output is small; everything upstream of it that is elementwise can be pulled in without the fused kernel having to materialize anything wide.",
              "Notice also what did not get merged. `%fused_computation.1`, the ROOT fusion, recomputes the same subtract and the same exponential that `%fused_computation.2` already computed. The compiler chose to do the exponential twice rather than write a 1024 by 512 f32 intermediate to memory and read it back. That is the cost model deciding that arithmetic is cheaper than traffic, visible in the dump, on a program small enough to check by hand.",
            ],
          },
          {
            h: 'the three ways a merge does not happen',
            ps: [
              "When two operations you expected to fuse did not, there are three distinct reasons, and telling them apart is most of the diagnostic skill. The first is structural. The pass can only merge instructions that already sit on a producer-consumer edge, so if two values are related in your head but not connected in the graph, no fusion pass has anything to consider. This is the wall the path spends a whole stage on at /l/xla and returns to at /xla/fusion, and it is why the naive softmax spill survives every pass in the pipeline.",
              "The second is profitability. The candidate existed, the queue ranked it, and the difference of time estimates came out negative or too small. These are the ones the FusionProcessDumpProto can tell you about directly, and the ones the fusion x-ray at /gym/xla#fusion is built out of: three real before-and-after pairs where the ranking made a call you can check against your own intuition.",
              "The third is that something sits between the two operations that is not fusible at all. The softmax capture has a clean instance. Between the entry parameter and the fusion that reads it there is a `copy-start` and `copy-done` pair whose only effect is to move the value into memory space S(1). A fusion cannot absorb a memory-space move, so the copy stays outside, and the fusion's parameter is the `copy-done` rather than the parameter itself. Bitcasts get the opposite treatment in the same capture: `%bitcast_fusion` is a fusion whose entire body is one bitcast, which exists because the backend wanted a fusion-shaped node in that position.",
            ],
          },
        ],
        readings: [
          { label: 'FusionKind in hlo_instruction.h', url: 'https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.h', note: 'the four kinds and the comment describing each' },
          { label: 'priority_fusion.h', url: 'https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.h', note: 'the priority formula, in the header that implements it' },
          { label: 'HLO operation semantics', url: 'https://openxla.org/xla/operation_semantics', note: 'the op contracts fusibility checks are built on' },
        ],
      },
    ],
  },
  {
    unit: 'xla:spmd',
    lessons: [
  {
        id: 'the-partitioner',
        check: [
          {
            q: 'Who authors every collective in an optimized module?',
            a: 'The SPMD partitioner, the pass named spmd-partitioning: each collective is the answer to a resharding request between two shardings.',
          },
          {
            q: 'What does shard_map change about how you read a dump?',
            a: "Inside manual regions the shapes are already per-device and the collectives are yours, not the pass's; propagation had nothing to fill in.",
          },
        ],
        num: 1,
        work: [
          { id: 'collectives', label: 'count collectives in a partitioned dump and name the resharding each one answers' },
        ],
        title: 'SPMD',
        lede: "One annotated program goes in and one program per device comes out, with every collective the math required inserted by a pass that has a name, an options struct, and a fixpoint.",
        goal: "Given a partitioned module, name the pass that rewrote it, predict which resharding operation produced a given collective, and say whether a region of your program was under propagation or under manual control.",
        sections: [
          {
            h: 'the pass, named, and the paper behind it',
            ps: [
              "The chapter above introduces this as the mechanism under JAX's Mesh and PartitionSpec, and names SpmdPartitioner correctly. Two details make the source navigable from there. The pass is an HloModulePass whose `name()` returns `spmd-partitioning`, so that is the string to grep for in a dump filename and the string to pass to a disable flag. And it is the pass that inserts all-gather, all-reduce, collective-permute, and all-to-all, which means every collective in your optimized module has exactly one author.",
              "The design has a paper, and the paper's numbers are worth carrying because they set the scale this machinery was built for. GSPMD: General and Scalable Parallelization for ML Computation Graphs, by Xu, Lee, Chen, Hechtman and thirteen others, reports 50% to 62% compute utilization on up to 2048 Cloud TPUv3 cores for models with up to one trillion parameters. The design brief in the abstract is the part that explains the annotation model: practitioners write code as if targeting a single device, then distribute it through minimal annotations.",
              "Minimal is the operative word, and it is a claim about the propagation algorithm rather than about ergonomics. The system determines operator partitioning from limited user guidance, which is to say most instructions in a real partitioned module carry a sharding nobody wrote.",
            ],
          },
          {
            h: 'propagation, then rewriting',
            ps: [
              "Propagation runs first and completes before any rewriting starts. Annotations you gave flow forward and backward through the graph until nothing changes, and every unannotated instruction inherits whatever its neighbours settled on. The interesting cases are the disagreements, where two neighbours would prefer different shardings for the same value.",
              "That those conflicts are ordinary rather than exceptional is visible in the options struct. SpmdPartitionerOptions carries a field called `need_resolve_conflicts`, which only exists because conflicting shardings are a routine input to this pass rather than a malformed one. Reading an options struct is often the fastest way to learn what a pass considers a normal day.",
              "Where the annotation lives is currently moving. Shardy, the `sdy` dialect, attaches sharding in MLIR to StableHLO before the module ever becomes HLO. That is an earlier altitude for the same information, not a different partitioner: the destination is still a module where every instruction has a sharding and the rewriting can begin.",
            ],
          },
          {
            h: 'PartitionedHlo, where a collective actually gets chosen',
            ps: [
              "The rewriting half has two supporting classes, and knowing them changes how you read a partitioned dump. SpmdBuilder wraps HloComputation::Builder and tracks derived instructions and broadcast dimensions as it goes, so the partitioner can reason about what it just created. PartitionedHlo is the more useful one to know: it represents a value together with its current partitioned state, and it exposes the operations that change that state, including `Reshard()`, `PadWithZero()`, and `Replicate()`.",
              "So a collective in your module is the output of a resharding request. Something asked for a value in a sharding it did not currently have, PartitionedHlo produced the instructions that get it there, and the collective you are looking at is that answer. Reading a dump backwards from a collective becomes a specific question with a specific answer: which two shardings does this instruction sit between, and which consumer demanded the second one.",
              "PartitionedHlo also caches its reshards, which is the reason the same conversion appearing twice in your Python does not always appear twice in the module. If you are counting collectives to reason about network time, count them in the optimized HLO rather than in your source, and expect the two counts to differ.",
            ],
            table: {
              caption: 'SpmdPartitionerOptions, and what each field tells you the pass has to handle',
              cols: ['option', 'what its existence implies'],
              rows: [
                ['need_resolve_conflicts', 'conflicting shardings are a routine input, not a malformed one'],
                ['cache_all_gather', 'gathered results can be retained, trading memory for repeat cost'],
                ['enable_windowed_einsum_for_all_gather', 'large contractions can overlap communication with compute rather than gathering first'],
                ['enable_windowed_einsum_for_reduce_scatter', 'the same treatment on the output side of a contraction'],
                ['conv_halo_exchange_always_on_lhs', 'spatially partitioned convolutions need halo exchange, and the side is a choice'],
                ['enable_dynamic_slice_collective_broadcast', 'a broadcast can be narrowed to the participants that need it'],
                ['preferred_gather_partition_methods', 'gathers have several partitioning strategies with no universal winner'],
                ['preferred_scatter_partition_methods', 'the same for scatters'],
              ],
            },
          },
          {
            h: 'where the topology re-enters',
            ps: [
              "The partitioner decides that an all-reduce belongs at a particular point in the dataflow. It does not decide how that all-reduce runs. An HLO collective names the operation, the participant groups, and the shapes; it says nothing about ring versus tree, about which links carry which fragment, or about how the transfer is chunked.",
              "That choice belongs to the runtime, below the compiler and below the HLO you are reading. On TPU it lives inside libtpu, whose own description names inter-chip communication alongside compilation and runtime execution as the three things the library provides. The ICI topology the kernel path teaches is a property of the physical pod, and the collective implementation is chosen against it at a level the dump does not show you.",
              "There is one compiler-visible handle on that boundary, and /xla/collectives is where the path picks it up: the async pair. A collective split into `all-reduce-start` and `all-reduce-done` gives the scheduler a gap to fill with independent compute, and the width of that gap is a compiler decision made in full ignorance of which link the bytes will cross. Both layers optimize the same collective and neither can see the other's choice.",
            ],
          },
          {
            h: 'the manual escape',
            ps: [
              "Propagation is not the only way to get a partitioned program, and knowing where the other door is matters for reading dumps. shard_map is JAX's manual mode, described in its own documentation as a single-program multiple-data multi-device parallelism API to map a function over shards of data. The contrast with jit is stated directly in the same document: with jit you write code as if for a single device and the compiler partitions it and generates the collectives behind the scenes.",
              "Inside a shard_map you do neither of those things. The function body sees per-device shapes already, you write the communication yourself, and the available primitives are a short list: psum for an all-reduce sum, all_gather, all_to_all, psum_scatter for reduce-scatter, and axis_index when a device needs to know which one it is. The mesh, in_specs, and out_specs arguments say how the inputs get split and how the outputs get reassembled.",
              "The two modes compose rather than compete. The documented pattern is manual control across groups of devices with compiler-based partitioning inside each group, which is what pipelining across islands with data parallelism inside them looks like. When you read a dump from such a program, the manual regions are the parts where the shapes are already per-device and the partitioner had nothing to propagate, and the collectives there are yours rather than the pass's.",
              '>> Inside a manual region, every collective in the dump is one you wrote.',
            ],
            code: {
              caption: 'the two doors, written against the documented shard_map signature and collective list',
              text: '# propagated: you annotate the edges, the partitioner fills in the middle\ny = jax.jit(f, out_shardings=NamedSharding(mesh, P("data", None)))(x)\n\n\n# manual: the body sees per-device shapes and you write the communication\n@partial(shard_map, mesh=mesh, in_specs=P("data", None), out_specs=P(None, None))\ndef g(x_shard):\n    local = x_shard @ w          # already a per-device shape here\n    return jax.lax.psum(local, "data")   # the only collective in this region',
              lang: 'python',
            },
          },
        ],
        readings: [
          { label: 'GSPMD paper', url: 'https://arxiv.org/abs/2105.04663', note: 'the design and the utilization numbers, from its authors' },
          { label: 'spmd_partitioner.h', url: 'https://github.com/openxla/xla/blob/main/xla/service/spmd/spmd_partitioner.h', note: 'the options struct and PartitionedHlo, in the header' },
          { label: 'shard_map', url: 'https://docs.jax.dev/en/latest/notebooks/shard_map.html', note: 'the manual door, with its collective list' },
        ],
      },
    ],
  },
  {
    unit: 'xla:codegen',
    lessons: [
  {
        id: 'backend-seam',
        check: [
          {
            q: 'Where exactly does shared XLA end?',
            a: 'At the xla/backends directory: five entries, none of them tpu. The TPU backend lives in the libtpu wheel behind the same exported PJRT symbol.',
          },
          {
            q: "Must a backend run XLA's pass pipeline?",
            a: 'No. PJRT_Client_Compile takes a program blob plus a format string; a backend may lower StableHLO through its own compiler and never construct an HloModule.',
          },
        ],
        num: 1,
        work: [
          { id: 'seam', label: 'list xla/backends yourself and place libtpu in the picture the pjrt lesson drew' },
        ],
        title: 'Codegen and the backend seam',
        lede: "The shared pipeline ends at a directory listing, and that listing has five entries in it, none of which is tpu.",
        goal: "Name the exact point where XLA stops behaving as one compiler, say which artifact each backend emits before a kernel runs, and describe what a third-party backend has to build against what it gets for free.",
        sections: [
          {
            h: 'five directories, and the one that is missing',
            ps: [
              "There is a fast way to see where the shared part of XLA ends, and it is a directory listing. xla/hlo/transforms holds the hardware-independent passes: the simplifiers, the expanders, the collective transforms, host offloading, precision propagation. Everything past that point lives under xla/backends, and xla/backends has exactly five entries: autotuner, cpu, gpu, interpreter, profiler.",
              "No tpu. That absence is not an oversight and it is the shape of this whole page. The TPU backend is not in the XLA tree, so the boundary between what you can read and what you cannot is not somewhere fuzzy inside codegen. It is a directory that does not exist.",
              "The official architecture describes three stages, and the five directories map onto the last two. First, target-independent optimization: common subexpression elimination, target-independent operation fusion, buffer analysis. Second, backend-specific HLO-level optimization done with target-specific information and needs in mind, including pattern matching for library calls. Third, code generation, with the CPU and GPU backends leveraging LLVM for low-level IR, optimization, and code generation. The first stage is shared. The second and third are per backend, and one backend keeps them in a wheel.",
            ],
            code: {
              caption: 'the seam as two directory listings, from the public tree',
              text: 'xla/hlo/transforms/          # hardware-independent, shared by every backend\n  collectives/\n  expanders/\n  simplifiers/\n  host_offloader.h  bfloat16_propagation.h  defuser.h  ...\n\nxla/backends/                # everything past the shared pipeline\n  autotuner/\n  cpu/\n  gpu/\n  interpreter/\n  profiler/\n                             # there is no tpu/ here, and that is the point',
              lang: 'text',
            },
          },
          {
            h: 'what CPU and GPU actually emit',
            ps: [
              "The chapter above walks the emitters, and the thing worth adding here is that each of these backends produces two artifacts, not one, and confusing them makes the runtime look more mysterious than it is. LLVM generates the machine code for a kernel. A separate structure, the thunk sequence, is the program that decides which kernel runs when and against which buffers. Compiled code and execution plan, produced by the same compile, consumed by different parts of the runtime.",
              "GPU adds a complication the CPU path does not have: not every operation becomes XLA-generated code at all. A matmul can be rewritten into a cuBLAS custom call, a fusion can be handed to Triton, and a convolution can go to cuDNN. Each of those is a decision made before codegen about whether XLA emits anything for that instruction, and the fusion kind kCustom from the fusion unit's lesson is the marker for a fusion claimed by that kind of path.",
              "The interpreter backend in that listing is easy to skip past and worth a sentence. It is the reference implementation, the thing run_hlo_module compares a compiled result against by default. A backend that produces a different answer from the interpreter has a bug, and having a slow, obvious, hardware-free implementation in the tree is what makes that statement checkable.",
            ],
          },
          {
            h: 'what the libtpu artifact actually contains',
            ps: [
              "libtpu is one file that holds three things people usually think of as separate. Its own publisher describes it as the core library that enables JAX, PyTorch, and TensorFlow to execute models on Google Cloud TPUs, providing core functionality for compilation, inter-chip communication, and runtime execution. So: a compiler, a network layer, and a runtime, shipped as a wheel. Version 0.0.45 published on 2026-08-01, at 215.9 MB.",
              "The fourth thing in there is the one the pjrt lesson established. It is a PJRT plugin, exporting GetPjrtApi, discovered by JAX through the same mechanism as any other plugin. Every layer above it, IFRT, jax.jit, your model code, is talking to a function-pointer struct and does not know or care that a compiler is on the other side.",
              "What stays readable through that boundary is more than you would expect, and /xla/pipeline is built on it. The closed compiler dumps through the same flags, so its pass names come out: `Phase_1_pre_layout_assignment_passes`, `X64_elimination`, `hlo_device_type_async_wrapper`, `add-random-host-offloading`, `tpu-embedding-thread-annotator`. You cannot read the implementation of a single one of those. You can read the whole list, in order, and diff the module on either side of any of them.",
              '>> A closed compiler still prints its own pass names, and still dumps the module on both sides of each one.',
            ],
          },
          {
            h: 'what a third-party backend owns',
            ps: [
              "Now the question this lesson exists for. If you are building a backend, what do you have to write? The contract is narrower than the size of the XLA tree suggests, and the reason is in a struct from the pjrt lesson. PJRT_Client_Compile takes a PJRT_Program, and a PJRT_Program is a code blob plus a `format` string. Nothing in that contract says the code is HLO. Nothing says XLA's pass pipeline runs.",
              "So XLA's optimizer is a library available to a backend, not a stage a backend has to pass through. You can take the StableHLO you were handed, lower it into your own IR, and never construct an HloModule at all. Or you can link XLA, run its pipeline, and write only an emitter, which is what the CPU and GPU backends do. Both choices satisfy the same PJRT contract, and a framework above cannot tell which one you made.",
              "There is a conformance harness for the interface itself. `RegisterPjRtCApiTestFactory` gives a plugin the standard tests of basic PJRT C API behaviours, and the integration guide's own acceptance criteria are ordinary JAX operations: element-wise arithmetic, a jit compile, and a pmap with a collective reduction. That last one is the real gate, because it exercises the multi-device path that the key-value callbacks in the pjrt lesson exist to bootstrap.",
            ],
            table: {
              caption: 'building a backend: what you write against what the contract gives you',
              cols: ['piece', 'who provides it'],
              rows: [
                ['StableHLO on the way in', 'the frontend, unconditionally'],
                ['the PJRT_Api struct', 'XLA, via pjrt::CreatePjrtApi, or you by hand'],
                ['buffer and executable handles', 'XLA wrapper code, if you subclass xla::PjRtClient'],
                ['device and topology description', 'you, always'],
                ['StableHLO to machine code', 'you, or XLA if you choose to link its pipeline'],
                ['HLO passes', 'optional; nothing in the PJRT contract requires HLO to exist'],
                ['collective implementation', 'you, against your own interconnect'],
                ['interface conformance tests', 'XLA, via RegisterPjRtCApiTestFactory'],
              ],
            },
          },
          {
            h: 'the seam, stated as a build decision',
            ps: [
              "Both routes exist in public, at both extremes of scale, which makes the decision concrete rather than theoretical. The example plugin under xla/pjrt/plugin/example_plugin is the linked route at its smallest: a C++ PjRtClient subclass, wrapped by pjrt::CreateWrapperClient, exported through a PJRT_Api that pjrt::CreatePjrtApi assembled. libtpu is the other route at its largest: an entire compiler and runtime behind the same exported symbol, sharing none of XLA's codegen.",
              "The question that decides which one fits is not about performance, it is about how close your hardware's execution model is to the one XLA's passes assume. If shapes, layouts, and buffers mean roughly what they mean on a GPU, linking the pipeline saves you years. If they do not, the pipeline will spend those years fighting you, and the PJRT contract is deliberately loose enough to let you skip it.",
              "Either way the surface a framework sees is identical, and that is the property the ifrt unit's lesson depends on. A layer above PJRT can treat every backend as the same kind of object precisely because the seam was drawn at a struct of function pointers rather than at an IR.",
            ],
          },
        ],
        readings: [
          { label: 'XLA architecture', url: 'https://openxla.org/xla/architecture', note: 'the three stages, and where LLVM enters' },
          { label: 'xla/backends', url: 'https://github.com/openxla/xla/tree/main/xla/backends', note: 'five directories; count them yourself' },
          { label: 'libtpu on PyPI', url: 'https://pypi.org/project/libtpu/', note: 'compilation, ICI, and runtime, in one wheel' },
        ],
      },
    ],
  },
  {
    unit: 'xla:ifrt',
    lessons: [
  {
        id: 'above-the-compiler',
        check: [
          {
            q: "What is IFRT's unit of data, against PJRT's?",
            a: 'One ifrt::Array carrying its own dtype, shape, sharding, and layout across devices, where PJRT hands out per-device buffers the framework must track by hand.',
          },
          {
            q: 'What makes the proxy the swap point?',
            a: 'It splits the interface across a wire, so a different runtime, a Pathways-style single controller included, can sit behind the same client with unchanged JAX above.',
          },
        ],
        num: 1,
        work: [
          { id: 'capstone', label: 'open the capstone PJRT project and write down the first struct you would fill in' },
        ],
        title: 'Above the compiler',
        lede: "PJRT hands you one device's buffers. IFRT hands you one array that spans devices, and that difference is what lets a whole runtime be swapped underneath JAX code nobody edited.",
        goal: "State what IFRT adds over PJRT in interface terms rather than in adjectives, name the specific seam a single-controller runtime plugs into, and explain why this stack has three layers instead of two.",
        sections: [
          {
            h: 'the bifurcation, in the README words',
            ps: [
              "IFRT's own README is unusually direct about why it exists, and it is worth reading before any diagram. IFRT stands for Interim Framework Runtime, and it is described as a high-level ML runtime API designed to be used as the interface between a user-facing framework, such as JAX, PyTorch, or TensorFlow, and the runtimes below.",
              "The history is stated just as plainly. PjRt was a low-level API that abstracted hardware differences on a single host, and it was, in the README's own phrasing, pretty much just another runtime. Distributed execution across thousands of accelerators went past what that API was shaped for, so the teams decided to bifurcate the current PjRt API and let the two deviate to better support their intended use cases. IFRT is not a rewrite of PJRT and it does not replace it.",
              "One sentence in there is the design brief for everything else in this lesson: frameworks should more or less declaratively express the work that needs to be done, and delegate policy choices about how to efficiently execute that work to the runtime implementations. Declare what, not how. Every difference between the two APIs follows from taking that seriously.",
            ],
          },
          {
            h: 'the unit of data changed, and so did the vocabulary',
            ps: [
              "The chapter above makes the central comparison and quotes array.h directly: PJRT's unit is a per-device buffer, so an eight-device array is eight PjRtBuffer objects with the framework tracking how they fit together, while `ifrt::Array` is one object carrying its own dtype, shape, sharding, and layout. That is the correct headline. What a chapter cannot do is show you the rest of the directory, and the rest of the directory is where the second half of the story is.",
              "The headers in xla/python/ifrt include array.h, client.h, device.h, device_list.h, executable.h, compiler.h, program.h, dtype.h, layout.h, memory.h, and bundle.h. Two of those names say something the array comparison alone does not. compiler.h and program.h mean IFRT has its own notion of compilation, so a runtime behind it can be handed a program to compile rather than only arrays to move. device_list.h means a set of devices is a first-class value with its own type, which PJRT has no equivalent of.",
              "Put those together and the layer is doing more than bookkeeping. A framework can hand IFRT a program and a set of devices and a sharding, and let the implementation decide how to place and execute it. That is the declarative brief from the README, expressed as a header list.",
            ],
            code: {
              caption: 'xla/python/ifrt, read as a map of what the layer claims to own',
              text: 'array.h        one logical array: dtype(), shape(), sharding(), layout()\nclient.h       what builds and reshapes those arrays\ndevice.h       a device\ndevice_list.h  a SET of devices, as a value type; PJRT has no counterpart\ncompiler.h     IFRT compiles, it does not only move data\nprogram.h      ... and it has its own notion of what a program is\nexecutable.h   the compiled result, across devices\ndtype.h  layout.h  memory.h  bundle.h',
              lang: 'text',
            },
            table: {
              caption: 'the same three concerns at two altitudes',
              cols: ['concern', 'PJRT', 'IFRT'],
              rows: [
                ['data', 'PjRtBuffer, one device, one shard', 'ifrt::Array, one object, its own sharding()'],
                ['devices', 'a list of PjRtDevice on the client', 'DeviceList as a first-class value type'],
                ['compilation', 'PJRT_Client_Compile, one program for one client', 'its own Compiler and Program abstractions'],
                ['who tracks a sharded array', 'the framework above, by hand', 'the array itself'],
              ],
            },
          },
          {
            h: 'the proxy is the swap point',
            ps: [
              "An interface only earns the name if something other than the obvious implementation can sit behind it, and IFRT has two. The in-process implementation wraps PjRtClient and PjRtBuffer directly, and a single-process JAX program runs on it without ever mentioning IFRT. The other one, under xla/python/ifrt_proxy, splits the interface in half: the process holding the Python client talks over a wire to a server that owns the actual runtime.",
              "That split is the entire swap mechanism, and /xla/pathways is where the path spends a chapter on what gets swapped in. Pathways puts a single controller in front of a fleet of workers, which buys MPMD, worker failure as a recoverable event rather than a job-ending one, and a scale ceiling above a single gang-scheduled program. Its design is published as a paper. Its runtime is not open source, and the honest version of this lesson says so in the same breath: what you can read is the proxy boundary, because that boundary has to be public for anything to plug into it at all.",
              "So the property that makes the layering worth its complexity is testable in one sentence. The same JAX program, unchanged, runs against an in-process PJRT-backed IFRT client and against a proxy client with a different runtime behind it. Nothing above the client had to know which.",
            ],
          },
          {
            h: 'why three layers and not two',
            ps: [
              "It is fair to ask whether this is one abstraction too many, and the answer is easiest to see by trying to collapse it in each direction. Fold IFRT into PJRT and every framework goes back to tracking eight buffers and a sharding by hand, which is exactly the bookkeeping JAX had before and exactly the thing that does not survive a single logical array spanning host processes.",
              "Fold the runtime into IFRT and the interface stops being an interface. The proxy split only means something because there is a real seam there with two implementations on the far side of it, and a single-controller runtime is not a variant of an in-process client, it is a different system with a different failure model.",
              "Which leaves three layers with one job each. PJRT owns per-device memory and one compiled executable on one client. IFRT owns arrays, shardings, device sets, and a compilation interface across devices. The runtime behind the proxy owns scheduling and placement policy. Each of them is the smallest thing that can answer its own question, and none of them can answer another's.",
            ],
          },
          {
            h: 'where this leaves you',
            ps: [
              "These lessons went down the stack and back up it: the exported symbol at pjrt, the invariants a pass may not break at hlo, the pipeline as filenames, the fusion queue, the partitioner, the seam where backends part ways, and the layer above all of it here. What is left is not more depth pages. It is the path's own closing chapters and the work in them.",
              "/xla/ifrt reads array.h line by line, /xla/mcjax has the multi-controller architecture these arrays span, and /xla/pathways states carefully which claims come from a paper rather than from code. Read them in that order if you have not, or reread them now that the interfaces underneath have names.",
              "Then /xla/capstone, and specifically its first project, which is the one these lessons were assembled to make possible: implement the minimal PJRT C API surface until jax.devices() on your own machine returns a device you built. The pjrt lesson already listed what CreatePjrtApi will ask you for. The deliverable that project wants is a write-up of every struct you filled in and the order you filled them, which is a map nobody has published and which you will only have after doing it.",
            ],
          },
        ],
        readings: [
          { label: 'the IFRT tree', url: 'https://github.com/openxla/xla/tree/main/xla/python/ifrt', note: 'the README and the headers, read together' },
          { label: 'Pathways paper', url: 'https://arxiv.org/abs/2203.12533', note: 'the runtime you can read about but not read' },
          { label: 'PJRT on openxla.org', url: 'https://openxla.org/xla/pjrt', note: 'the layer underneath, where these lessons started' },
        ],
      },
    ],
  },
  {
    unit: 'xla:ingestion',
    lessons: [
      {
        id: 'the-translation-walk',
        num: 1,
        title: 'The translation walk',
        lede: 'One six-line function, three representations: the Python you wrote, the jaxpr tracing recorded, and the StableHLO XLA received. This lesson walks the same program across both borders.',
        goal: 'Given any traced program, predict where each piece of it lands in the StableHLO: which equation becomes which op, where the carry goes, and what disappears.',
        sections: [
          {
            h: 'the program, on the source side',
            ps: [
              'The walk needs a program small enough to hold whole and rich enough to exercise the interesting machinery. A running mean over rows does it: there is a carry (the total and the count), a loop (`lax.scan`), and elementwise arithmetic inside the step. Six lines of Python, and every one of them will be findable on the far side of two translations.',
            ],
            code: {
            caption: 'the program, verbatim from the corpus (generated by gen_cf_corpus.py, jax 0.4.38)',
            lang: 'python',
            text: "def running_mean(xs):\n    def step(carry, x):\n        total, n = carry\n        return (total + x, n + 1.0), (total + x) / (n + 1.0)\n    (_, _), means = jax.lax.scan(step, (jnp.zeros(xs.shape[1]), 0.0), xs)\n    return means",
          },
          },
          {
            h: 'the jaxpr side',
            ps: [
              'Trace it and the recording shows the grammar the jaxpr lessons taught: one `scan` equation at the top level, with the step function as a nested jaxpr inside its params and the carry as explicit state. Notice what already happened here, before XLA saw anything: the tuple you wrote became positional values, the closure over nothing stayed empty, and every value carries its shape.',
            ],
            code: {
            caption: 'the head of the traced jaxpr, verbatim (the full dump is in the gym corpus)',
            lang: 'haskell',
            text: "{ lambda ; a:f32[8,16]. let\n    b:f32[16] = broadcast_in_dim[\n      broadcast_dimensions=()\n      shape=(16,)\n      sharding=None\n    ] 0.0\n    _:f32[16] _:f32[] c:f32[8,16] = scan[\n      _split_transpose=False\n      jaxpr={ lambda ; d:f32[16] e:f32[] f:f32[16]. let\n          g:f32[16] = add d f\n          h:f32[] = add e 1.0\n          i:f32[16] = add d f\n          j:f32[] = add e 1.0\n          k:f32[] = convert_element_type[new_dtype=float32 weak_type=False] j",
          },
          },
          {
            h: 'the StableHLO side',
            ps: [
              'Lower it and the same structure re-appears in the other notation. The `scan` equation became `stablehlo.while`; the carry you threaded became the operand tuple riding the loop, visible in the `%iterArg` list; the nested jaxpr became the `cond` and `do` regions attached to the op. Nothing about the computation moved, and everything about the notation did.',
            ],
            code: {
            caption: 'the while op with its carry tuple, verbatim from the same corpus entry',
            lang: 'mlir',
            text: "    %2:5 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c, %iterArg_3 = %0, %iterArg_4 = %cst_0, %iterArg_5 = %1) : tensor<8x16xf32>, tensor<i32>, tensor<16xf32>, tensor<f32>, tensor<8x16xf32>\n     cond {\n      %c_6 = stablehlo.constant dense<8> : tensor<i32>\n      %3 = stablehlo.compare  LT, %iterArg_2, %c_6,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>\n      stablehlo.return %3 : tensor<i1>\n    } do {",
          },
          },
          {
            h: 'the translator\'s dictionary',
            ps: [
              'The walk generalizes to a small dictionary you can apply to any program. Equations become ops one for one, with the primitive name usually surviving recognizably. Nested jaxprs become regions on the op that carried them. Explicit carry state stays explicit, changing only its clothes: params-and-binders on one side, an operand tuple on the other. And names die at the border: your Python identifiers were already gone in the jaxpr, and the jaxpr\'s letter variables give way to numbered `%` values. What survives every translation is shapes, dtypes, and structure, which is exactly the set of things the compiler is allowed to care about.',
              'The chapter above this lesson explains why the two IRs exist at all; the corpus x-ray on the gym floor has seventeen more programs to walk. Fluency is the point: the border should feel like notation, not like a wall.',
            ],
          },
        ],
        readings: [
          { label: 'JAX · understanding jaxprs', url: 'https://docs.jax.dev/en/latest/jaxpr.html', note: 'the left side of the dictionary' },
          { label: 'StableHLO spec', url: 'https://openxla.org/stablehlo/spec', note: 'the right side, op by op' },
        ],
        check: [
          {
            q: 'Where does the scan\'s carry live on each side of the translation?',
            a: 'In the jaxpr, as explicit state on the scan equation with the body nested in its params. In StableHLO, as the operand tuple threading through the while op\'s regions. Same contract, two notations.',
          },
          {
            q: 'What survives translation unchanged, and what changes form?',
            a: 'Shapes, dtypes, and the structure of the computation survive. Notation changes: equations become ops, nested jaxprs become regions, and names die at each border, ending as numbered values.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
    ],
  },
  {
    unit: 'xla:layout-memory',
    lessons: [
      {
        id: 'copies-and-buffers',
        num: 1,
        title: 'Two kinds of copies, and who owns a buffer',
        lede: 'Some copies exist because two ops disagree about memory order. Others exist because a buffer someone still needs was about to be overwritten. Telling them apart is most of reading a memory story.',
        goal: 'Classify any copy in a dump as a layout copy or a correctness copy, and say what donating an input changes about the buffers underneath.',
        sections: [
          {
            h: 'layout copies, briefly, and where the pass lives',
            ps: [
              'The layouts lesson under the XLA layer chapter owns the reading of `{1,0}` annotations and the copies a disagreement inserts; this lesson picks up where that one stops, at position and ownership. Layout assignment has an address in the pipeline: in this repo\'s CPU capture it sits mid-flight, after `flatten-call-graph` and before `sub-byte-size-setter`, which means everything after it must either respect the chosen layouts or pay for a rearrangement. A layout copy is that payment, and it is attached to a disagreement you can name: two ops, one value, two opinions about which dimension varies fastest.',
            ],
          },
          {
            h: 'correctness copies: the pass named copy-insertion',
            ps: [
              'Late in the same capture, one step before dead-code elimination, runs a pass called `copy-insertion`, and it has nothing to do with layouts. Its job is protecting live values under aliasing. A while loop\'s state buffer gets updated in place across iterations; an output that reuses an input\'s buffer overwrites it; and in every such case, if someone still needs the old value after the new one lands, the program is wrong. The pass finds those hazards and inserts a copy exactly where an alias would corrupt a value that still has a reader.',
              'So the two copy families answer different questions. A layout copy answers "these two ops disagree about memory order." A correctness copy answers "this buffer is about to be reused while its old contents still matter." When a dump shows a copy you did not expect, deciding which family it belongs to is the first diagnostic step, because the fixes live in different places: layouts are argued with the compiler, aliasing hazards are argued with your own program structure.',
            ],
          },
          {
            h: 'donation reaches the buffers',
            ps: [
              'Aliasing is not only something the compiler discovers; you can hand it over deliberately. `jax.jit(f, donate_argnums=...)` tells XLA an input\'s buffer may be consumed: the output can be written into the donated input\'s memory, saving an allocation and often a copy, which matters exactly when the arrays are large and the update is the point, an optimizer state, a KV cache, a params tree. The price is that the donated array is dead to your program afterward; touch it again and you get an error rather than stale data.',
              'Donation is a request, not a command. A backend that cannot honor it ignores it, and jax warns when that happens, so the honest workflow is to donate, then check the warning stream and the memory numbers rather than assuming the reuse happened. Between this and the correctness copies above, buffer ownership stops being folklore: you can say who owns a region of memory at any point in the compiled program, and what evidence backs the claim.',
            ],
          },
        ],
        readings: [
          { label: 'JAX · buffer donation', url: 'https://docs.jax.dev/en/latest/faq.html#buffer-donation', note: 'the request, its price, and the warning when ignored' },
          { label: 'XLA shapes and layout', url: 'https://openxla.org/xla/shapes', note: 'the layout half of the copy story' },
        ],
        check: [
          {
            q: 'The pass named copy-insertion is not about layouts. What is it protecting?',
            a: 'Live values under aliasing: while-loop state, in-place reuse, and donated inputs must not be clobbered while a reader remains, so the pass inserts copies exactly where an alias would corrupt a value someone still needs.',
          },
          {
            q: 'What does donating an input buy, and what does it cost?',
            a: 'XLA may write the output into the donated buffer, saving an allocation and often a copy. The cost is the array is dead to your program afterward, and since donation is a request, jax warns when a backend ignored it.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
    ],
  },
  {
    unit: 'xla:collectives',
    lessons: [
      {
        id: 'six-shardings-one-matmul',
        num: 1,
        title: 'Six shardings, one matmul',
        lede: 'The same (512,256) by (256,512) matmul, six ways of sharding it across eight devices, and the collectives XLA actually inserted for each, captured on this repo.',
        goal: 'For any sharded matmul, predict whether resharding is needed at all, and read the inserted collectives as the partitioner\'s chosen plan rather than the only possible one.',
        sections: [
          {
            h: 'the table, from the capture',
            ps: [
              'The corpus behind the collective picker instrument (EX·16 on the spmd chapter) holds six real cases: one matmul, shapes `(512,256) x (256,512)`, on an eight-device mesh laid out `(4 data, 2 model)`, jax 0.4.38. Each row names the two input shardings, the requested output sharding, and the collectives the compiled module actually contains. The table is worth reading before any theory, because two of its rows are silent and one of them is louder than you would guess.',
            ],
            table: {
              caption: 'verbatim from site/src/data/xla/sharding-corpus.json · mesh (4 data, 2 model), 8 devices, jax 0.4.38',
              cols: ['a sharded', 'b sharded', 'out requested', 'collectives inserted'],
              rows: [
              ["rows over data", "replicated", "rows over data", "none"],
              ["rows over data", "replicated", "replicated", "all-gather"],
              ["cols over model", "rows over data", "replicated", "all-gather"],
              ["replicated", "cols over model", "cols over model", "none"],
              ["rows data, cols model", "replicated", "rows over data", "all-gather"],
              ["cols over model", "cols over model", "replicated", "all-gather + collective-permute"],
              ],
            },
          },
          {
            h: 'the silent rows',
            ps: [
              'Rows one and four insert nothing, and they share a shape: the contraction axis arrives whole on every shard, and the local results already sit exactly as the requested output sharding wants them. Row one splits `a` by rows across data and leaves `b` replicated, so each shard computes its own rows of the answer; row four mirrors it on the other side, splitting `b` by columns. No resharding question is ever asked, so the partitioner has nothing to answer. Free is not a special case; it is what alignment looks like.',
            ],
          },
          {
            h: 'the rows where the partitioner chose',
            ps: [
              'Row three is the teaching row. Both inputs are split along the contraction axis, `a` over model and `b` over data, so no shard can finish its arithmetic alone; something has to move. A partial-sum analysis predicts a reduction. The capture shows an `all-gather`: the partitioner chose to reassemble an operand and compute locally rather than reduce partials, and the module records that choice without asking your opinion. Row six goes further, splitting both inputs over the same model axis, and pays with two collectives, an `all-gather` and a `collective-permute`, to reconcile the two different splits.',
              'The lesson in those rows is not the specific plans; it is who decides. The partitioner picks among mathematically equal strategies with a cost model you do not see, and the only reliable account of the plan you got is the dump. The partitioner lesson under the spmd unit gives every collective its author; this table is where you watch the author make six different calls on one program. Step through the same six rows live in EX·16, then predict row five before revealing it.',
            ],
          },
        ],
        readings: [
          { label: 'GSPMD paper', url: 'https://arxiv.org/abs/2105.04663', note: 'the cost-model machinery behind the choices in the table' },
          { label: 'JAX · distributed arrays', url: 'https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html', note: 'the sharding vocabulary the rows are written in' },
        ],
        check: [
          {
            q: 'Two rows insert nothing. What makes a sharded matmul free?',
            a: 'The contraction axis arrives whole on every shard and the local results already match the requested output sharding, so no resharding question is ever asked. Alignment, not luck.',
          },
          {
            q: 'Row three gathers an operand where an analysis predicts a reduction of partial sums. What is the lesson in that?',
            a: 'The partitioner picks among mathematically equal plans with its own cost model, and the dump records the choice it made. Reading the inserted collectives is how you learn the plan you actually got, rather than the one you assumed.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
    ],
  },
]
