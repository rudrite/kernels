// New file: site/src/data/lessons/torch-dynamo.ts
// Dynamo's capture layer, below the survey chapter 6 teaches: the bytecode it
// rewrites and the resume functions it synthesizes, the guard predicates as
// printed text, the four families of graph break, and the cache line with its
// two limits and its symbolic shapes. Every printed value was produced on one
// x86_64 macOS machine with torch 2.2.2 CPU (git_version 39901f2) on Python
// 3.11.15, with backend="eager" so the numbers are capture-side only. Reason
// strings and guard wording are version-specific; re-read them on your torch.
import type { UnitLessons } from './index'

export const TORCH_DYNAMO_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:dynamo',
    lessons: [
      {
        id: 'the-frame-it-rewrites',
        num: 1,
        title: 'The frame it rewrites',
        lede: 'Nothing about your function changes when you compile it. What changes is the code object the interpreter runs while the compiled wrapper is on the stack, and you can print both versions side by side.',
        goal: 'Given a compiled function, print its original and modified bytecode, name the three artifacts dynamo produced, read the frame prefix on any dynamo log line, and say exactly what the two counters in torch._dynamo.utils count.',
        sections: [
          {
            h: 'the callback, not the wrapper',
            ps: [
              "The chapter above calls it attaching dynamo to a function, which is the right picture for reasoning about guards and the wrong one for reasoning about what runs. Open `torch/_dynamo/eval_frame.py` and the wrapper's body is a context. On entry it calls `set_eval_frame(callback)` and saves whatever callback was installed before; in the `finally` it puts the old one back. Between those two lines the interpreter asks dynamo about every Python frame it is about to execute, not only the one you decorated.",
              "That single fact explains behaviour you would otherwise file as magic. A function dynamo has given up on can still have its callees compiled, because their frames are offered to the callback too while the wrapper is on the stack, which is where lesson four ends. And the C extension module holding the machinery is small enough to inventory in one line of Python: `set_eval_frame`, `skip_code`, `reset_code`, `unsupported`, `set_guard_error_hook`, `_debug_get_cache_entry_list`, and the two types `_CacheEntry` and `_PyInterpreterFrame`.",
              "The unit of everything that follows is the code object, not the function object. Guards attach to a code object, the cache attaches to a code object, and the compile counter in every log line counts per code object. Build two closures from one `def` with equal captured values and they share a single cache entry and a single graph between them.",
            ],
            code: {
              caption: 'run on this machine (verified, torch 2.2.2 CPU, Python 3.11): the whole C surface dynamo uses to take over frame evaluation',
              lang: 'python',
              text: 'import torch\nfrom torch._C._dynamo import eval_frame\n\nprint(sorted(n for n in dir(eval_frame) if not n.startswith("__")))\n# [\'_CacheEntry\', \'_PyInterpreterFrame\', \'_debug_get_cache_entry_list\',\n#  \'reset_code\', \'set_eval_frame\', \'set_guard_error_hook\', \'skip_code\',\n#  \'unsupported\']',
            },
          },
          {
            h: 'eleven instructions become eight',
            ps: [
              "Set `TORCH_LOGS=bytecode` and dynamo prints what it read and what it wrote. The function below is two tensor ops and a return. Its original code object is eleven instructions: a global load for `torch`, an attribute load for `sin`, a call, a store, then the add and the return.",
              "The modified version is eight instructions and none of them mention `torch`. The entire body has been replaced by a call to `__compiled_fn_0`, a global dynamo installed on your module, followed by `UNPACK_SEQUENCE` because compiled functions return a tuple of outputs. Everything the tracer learned about your Python is gone from the bytecode; what remains is a call and an unpack.",
              "Worth doing once by hand: the modified bytecode has no guard check in it. Nothing here tests a shape. The guard runs earlier, in C, before this code object is chosen at all, which is why lesson two can print guards and this listing cannot show them.",
            ],
            code: {
              caption: 'TORCH_LOGS=bytecode on a two-op function (verified, torch 2.2.2 CPU, Python 3.11); log timestamps trimmed, absolute paths replaced by the file name',
              lang: 'text',
              text: "[0/0] ORIGINAL BYTECODE f simple.py line 3\n[0/0]   3           0 RESUME                   0\n[0/0]   5           2 LOAD_GLOBAL              1 (NULL + torch)\n[0/0]              14 LOAD_ATTR                1 (sin)\n[0/0]              24 LOAD_FAST                0 (x)\n[0/0]              26 PRECALL                  1\n[0/0]              30 CALL                     1\n[0/0]              40 STORE_FAST               1 (y)\n[0/0]   6          42 LOAD_FAST                1 (y)\n[0/0]              44 LOAD_CONST               1 (1)\n[0/0]              46 BINARY_OP                0 (+)\n[0/0]              50 RETURN_VALUE\n\n[0/0] MODIFIED BYTECODE f simple.py line 3\n[0/0]   3           0 RESUME                   0\n[0/0]               2 PUSH_NULL\n[0/0]               4 LOAD_GLOBAL              4 (__compiled_fn_0)\n[0/0]              16 LOAD_FAST                0 (x)\n[0/0]              18 PRECALL                  1\n[0/0]              22 CALL                     1\n[0/0]              32 UNPACK_SEQUENCE          1\n[0/0]              36 RETURN_VALUE",
            },
          },
          {
            h: 'a break leaves a second function behind',
            ps: [
              "Put a `print()` between the two tensor ops and the same log shows what a graph break actually costs. The rewritten frame calls `__compiled_fn_0` for the prefix, stores the result, runs the real `print` with the real argument, and then calls a global named `__resume_at_70_1`. The number 70 is the bytecode offset the break happened at.",
              "That resume function is not in your source. `ContinueExecutionCache.generate` in `torch/_dynamo/resume_execution.py` synthesizes a code object whose first three instructions are a `RESUME`, a load of a synthetic local called `___stack0`, and a `JUMP_FORWARD` over the part that already ran. Live variables come in as arguments. The interpreter enters it, jumps past the prefix, and lands on the continuation, which dynamo then compiles as a frame in its own right.",
              "Read the two listings together and the cost model stops being a slogan about fusion. A break means a second code object, a second capture, a second set of guards, and a Python call on the boundary between them, on every single call after that.",
            ],
            code: {
              caption: 'the same log with one print() in the middle (verified, torch 2.2.2 CPU, Python 3.11); the resume function is synthesized, not written',
              lang: 'text',
              text: "[0/0_1] MODIFIED BYTECODE f break.py line 3\n[0/0_1]               4 LOAD_GLOBAL              6 (__compiled_fn_0)\n[0/0_1]              22 CALL                     1\n[0/0_1]              32 STORE_FAST               3 (graph_out_0)\n[0/0_1]              36 LOAD_GLOBAL              4 (print)\n[0/0_1]              48 LOAD_CONST               1 ('mid')\n[0/0_1]              70 CALL                     1\n[0/0_1]              84 LOAD_GLOBAL              8 (__resume_at_70_1)\n[0/0_1]              98 LOAD_FAST                1 (y)\n[0/0_1]             104 CALL                     2\n[0/0_1]             114 RETURN_VALUE\n\n[1/0] ORIGINAL BYTECODE resume_in_f break.py line 6\n[1/0]   6           0 RESUME                   0\n[1/0]               2 LOAD_FAST                0 (___stack0)\n[1/0]               4 JUMP_FORWARD            35 (to 76)\n[1/0]         >>   76 POP_TOP\n[1/0]   7          78 LOAD_FAST                1 (y)\n[1/0]              82 BINARY_OP                0 (+)\n[1/0]              86 RETURN_VALUE",
            },
          },
          {
            h: 'reading the prefix on every log line',
            ps: [
              "The bracket in front of every dynamo log line is three numbers and it is the fastest diagnostic in the system. `torch/_guards.py` defines `CompileId` as a frame id and a per-frame compile count, printed as `0/1`, and `TraceId` adds an attempt counter that appears as a trailing `_1` when analysis had to restart.",
              "So `[0/0]` is the first frame dynamo saw, compiled for the first time, captured on the first attempt. `[0/1]` is that same frame being recompiled: a guard missed. `[1/0]` is a different frame entirely, which for the listing above is the resume function. And `[0/0_1]` says the first attempt was abandoned, which is what a graph break does: `convert_frame.py` loops over attempts, catches `RestartAnalysis`, and gives up only past a hundred restarts.",
              "Two glances at the same log now answer two different questions. Frames climbing means your program is being split. The second number climbing means a guard is missing and the same frame is being compiled again.",
            ],
            table: {
              caption: 'the log prefix, from torch/_guards.py and torch/_dynamo/convert_frame.py at 39901f2',
              cols: ['prefix', 'reads as', 'what it tells you'],
              rows: [
                ['[0/0]', 'frame 0, compile 0, first attempt', 'a clean first capture'],
                ['[0/1]', 'frame 0, compile 1', 'a recompile: some guard on frame 0 failed'],
                ['[1/0]', 'frame 1, compile 0', 'a second code object, usually a resume function'],
                ['[0/0_1]', 'frame 0, compile 0, attempt 1', 'analysis restarted, which is how a break is handled'],
              ],
            },
          },
          {
            h: 'what the two counters count',
            ps: [
              "`torch._dynamo.utils.counters` is the instrument the chapter uses and it repays five minutes of reading. `unique_graphs` is incremented once per compiled graph, next to the call into your backend. `calls_captured` is incremented by `count_calls(self.graph)`, the number of call nodes in the FX graph that was just built.",
              "The measurement below pins both. A one-op function called five times reports `calls_captured` 1; a three-op function called five times reports 3. Neither number moves on a call that hits the cache, because both are incremented inside the compile path and a cache hit never enters it. What `calls_captured` counts is captured ops, summed over every compile.",
              "Use `unique_graphs` when the question is how many times you paid, which is what the rest of this arc does. The lab LAB·P3 has the side-effect experiment that shows Python around the graph still running on every call, and it uses a plain global counter for it, which is the instrument that actually answers that question.",
            ],
            code: {
              caption: 'the counters, pinned (verified, torch 2.2.2 CPU, Python 3.11): op count decides calls_captured, call count does not',
              lang: 'python',
              text: 'import torch\nimport torch._dynamo as dynamo\nfrom torch._dynamo.utils import counters\n\ndef run(label, fn, n):\n    dynamo.reset(); counters.clear()\n    cf = torch.compile(fn, backend="eager")\n    for _ in range(n):\n        cf(torch.randn(4))\n    print(label, dict(counters["stats"]))\n\nrun("1 op, 1 call  ", lambda x: x + 1, 1)\nrun("1 op, 5 calls ", lambda x: x + 1, 5)\nrun("3 ops, 1 call ", lambda x: torch.sin(x) * 2 + 1, 1)\nrun("3 ops, 5 calls", lambda x: torch.sin(x) * 2 + 1, 5)\n# 1 op, 1 call   {\'calls_captured\': 1, \'unique_graphs\': 1}\n# 1 op, 5 calls  {\'calls_captured\': 1, \'unique_graphs\': 1}\n# 3 ops, 1 call  {\'calls_captured\': 3, \'unique_graphs\': 1}\n# 3 ops, 5 calls {\'calls_captured\': 3, \'unique_graphs\': 1}',
            },
          },
        ],
        readings: [
          { label: 'eval_frame.py at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/eval_frame.py', note: 'the wrapper that installs a frame-evaluation callback and restores the previous one' },
          { label: 'resume_execution.py at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/resume_execution.py', note: 'ContinueExecutionCache, where the __resume_at_ functions are built' },
          { label: 'PEP 523', url: 'https://peps.python.org/pep-0523/', note: 'the interpreter hook this whole design stands on, from 2016' },
        ],
        check: [
          {
            q: 'A helper function you never decorated shows up inside the captured graph. Why?',
            a: 'Because torch.compile installs a frame-evaluation callback for the duration of the call, not a rewrite of one function. Every frame executed inside that extent is offered to dynamo, so callees are captured too.',
          },
          {
            q: 'What does the trailing _1 in a log prefix like [0/0_1] mean?',
            a: 'The attempt counter from TraceId: analysis of that frame restarted once. A graph break is handled by raising RestartAnalysis and re-running the transform, so broken frames usually carry it.',
          },
          {
            q: 'Five calls to a compiled two-op function leave calls_captured at 2. What happened on calls two through five?',
            a: 'They hit the cache. Both counters are incremented inside the compile path, calls_captured by the number of call nodes in the graph just built, so a hit moves neither.',
          },
        ],
        work: [
          { id: 'print-both', label: 'run TORCH_LOGS=bytecode on one function of your own and mark, instruction by instruction, which of your Python survived into the modified code object', href: '#eleven-instructions-become-eight' },
          { id: 'read-a-prefix', label: 'take any dynamo log you have and classify every distinct prefix into new frame, recompile, or restart', href: '#reading-the-prefix-on-every-log-line' },
        ],
      },
      {
        id: 'what-a-guard-checks',
        num: 2,
        title: 'What a guard checks',
        lede: 'A guard is not a concept. It is a line of generated Python you can print, and the tensor one checks seven properties, of which shape is a single field.',
        goal: 'Print the guard set for any compiled function, say which entries are ambient and which came from your code, name the seven fields of the tensor guard, and derive from them five recompiles that have nothing to do with shape.',
        sections: [
          {
            h: 'the guard set, printed',
            ps: [
              "`TORCH_LOGS=guards` dumps the compiled guard for every capture. For a function taking a tensor and a float, the whole set is seven lines, and reading them in order tells you where each came from. The first three are about your arguments. The next three are ambient state nobody wrote down. The last is the tensor check.",
              "Look at what `check_tensor` compares. Type, dispatch key set, dtype, device, requires_grad, size, stride: seven fields for one argument. Shape is one of them, which is why the chapter's shorthand about shapes and dtypes is true and incomplete. The dispatch key set entry is why a tensor from a different device or with a different autograd state cannot reuse this entry even at identical dimensions.",
              "Two of the ambient lines deserve names. `___compile_config_hash()` pins the dynamo config in force at capture time, so flipping a config flag anywhere in the process invalidates every entry compiled under the old one. `___skip_backend_check() or ___current_backend() == ...` pins the backend, which is why the same function compiled twice under two backends does not share a thing.",
            ],
            code: {
              caption: 'TORCH_LOGS=guards for one two-argument function (verified, torch 2.2.2 CPU, Python 3.11); timestamps and the log prefix trimmed, paths shortened',
              lang: 'text',
              text: "GUARDS:\nhasattr(L['x'], '_dynamo_dynamic_indices') == False           # return torch.sin(x) * alpha  # guards.py:5 in f\n___check_type_id(L['alpha'], 4570165824)                      # return torch.sin(x) * alpha  # guards.py:5 in f\nL['alpha'] == 2.0                                             # return torch.sin(x) * alpha  # guards.py:5 in f\nutils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:379 in init_ambient_guards\n(___skip_backend_check() or ___current_backend() == ___lookup_backend(4643794512))  # _dynamo/output_graph.py:385 in init_ambient_guards\n___compile_config_hash() == '56b03c84a72a07721b34d987281ef6a5'  # _dynamo/output_graph.py:387 in init_ambient_guards\ncheck_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[4], stride=[1])  # return torch.sin(x) * alpha  # guards.py:5 in f",
            },
          },
          {
            h: 'eleven guard objects, eight printed lines',
            ps: [
              "`torch._dynamo.explain(f)(*args).out_guards` gives the same guards as objects rather than text, and the two lists are not the same length. For the function in the table below, explain reports eleven guard objects while the compiled check for that same function prints eight predicate lines. The gap is not arbitrary. `GuardBuilder.GRAD_MODE`, `DETERMINISTIC_ALGORITHMS` and `TORCH_FUNCTION_STATE` are each a bare `pass` with the comment that they are always guarded via `GlobalStateGuard()`, and `SHAPE_ENV` emits code only when a dimension is symbolic, which is lesson four's subject.",
              "The `create_fn` on each object is the builder method that wrote it, and the names are the taxonomy: `TENSOR_MATCH`, `CONSTANT_MATCH`, `FUNCTION_MATCH`, `CLOSURE_MATCH`, `GRAD_MODE`, `DETERMINISTIC_ALGORITHMS`, `TORCH_FUNCTION_STATE`, `DEFAULT_DEVICE`, `BACKEND_MATCH`, `CONFIG_HASH_MATCH`, `SHAPE_ENV`. The `name` field is the source expression: `L['x']` for a local, `G['scale']` for a global, empty for ambient state.",
              "The printed set is also two predicates short of what runs. Every check function begins with `___guarded_code.valid` and `___check_global_state()`, which `guards.py` builds first and then declines to log, with the comment that reporting it would be useless because it is always the same. That second call is the one covering grad mode, deterministic algorithms and torch function state together, and it is why the next section's `no_grad` recompile has nothing more specific to report.",
              "Reach for the objects when you want to enumerate what a capture depends on, and for the text when you want to know what will actually be tested per call. They answer different questions and the gap between them is a fact about dynamo, not about your program.",
            ],
            table: {
              caption: "explain(f)(torch.randn(4), True).out_guards for a function closing over a global (verified, torch 2.2.2 CPU, Python 3.11)",
              cols: ['name', 'create_fn', 'what it pins'],
              rows: [
                ["L['x']", 'TENSOR_MATCH', 'type, dispatch keys, dtype, device, requires_grad, size, stride'],
                ["L['flag']", 'CONSTANT_MATCH', 'the exact value of a Python argument dynamo specialized on'],
                ["G['scale']", 'CONSTANT_MATCH', 'the value of a module-level constant the graph baked in'],
                ["G['torch']", 'FUNCTION_MATCH', 'the module object the calls were resolved against'],
                ['', 'GRAD_MODE', 'whether grad was enabled at capture'],
                ['', 'DEFAULT_DEVICE', 'the ambient default device'],
                ['', 'BACKEND_MATCH', 'the compiler backend in force'],
                ['', 'CONFIG_HASH_MATCH', 'the dynamo config in force'],
                ['', 'SHAPE_ENV', 'the symbolic-shape assumptions, when there are any'],
              ],
            },
          },
          {
            h: 'five compiles, one shape',
            ps: [
              "Five calls, one function, one shape and one dtype throughout, and five separate compiles come out the other end. A transposed tensor is the same shape with a different stride. A tensor with `requires_grad` on is the same shape with a different flag. A call inside `torch.no_grad()` changes ambient state rather than any argument. Rebinding the module-level float the graph baked in changes a global.",
              "The recompile log names the failed predicate every time, and it is the fastest way to close a recompile investigation. Three of these four report the exact line: a stride mismatch with the expected value, a requires_grad mismatch, a `G['scale'] == 3.0` that no longer holds. The `no_grad` one reports `___check_global_state()`, the unprinted predicate from the previous section, which is as specific as that failure ever gets.",
              "Notice the shape of the output too. On the fourth recompile there are three failure lines, one per existing cache entry, because dynamo checked every entry in the line before deciding to compile. Lesson four picks that up.",
            ],
            code: {
              caption: 'five captures from one function with one shape (verified, torch 2.2.2 CPU, Python 3.11), run under TORCH_LOGS=recompiles',
              lang: 'python',
              text: 'import torch\nfrom torch._dynamo.utils import counters\n\nscale = 3.0\n\n@torch.compile(backend="eager")\ndef f(x):\n    return x * scale\n\ndef graphs():\n    return counters["stats"]["unique_graphs"]\n\nf(torch.randn(4, 4));                       print("contiguous   ", graphs())  # 1\nf(torch.randn(4, 4).t());                   print("transposed   ", graphs())  # 2\nf(torch.randn(4, 4, requires_grad=True));   print("requires_grad", graphs())  # 3\nwith torch.no_grad():\n    f(torch.randn(4, 4))\nprint("under no_grad", graphs())                                              # 4\nscale = 4.0\nf(torch.randn(4, 4));                       print("scale rebound", graphs())  # 5\n\n# the four guard failures TORCH_LOGS=recompiles reports, in order:\n#   - tensor \'L[\'x\']\' stride mismatch at index 0. expected 4, actual 1\n#   - tensor \'L[\'x\']\' requires_grad mismatch. expected requires_grad=0\n#   - ___check_global_state()\n#   - G[\'scale\'] == 3.0                        # return x * scale  # misses.py:8 in f',
            },
          },
          {
            h: 'the module you built twice',
            ps: [
              "One guard kind matters more than the rest for real models. When dynamo captures an `nn.Module` method, it guards `self` by object identity: `___check_obj_id(L['self'], 4621695952)`. Not by class, not by parameter shapes. By address.",
              "Two instances of one module class therefore compile twice, which is measurable in four lines and surprises people who assume the class is the cache key. An ensemble of sixteen identical members compiles sixteen times. A model rebuilt inside an evaluation loop compiles once per rebuild, forever, and the second lesson of that story is in lesson four, because identity guards are exactly why dynamo carries two cache limits instead of one.",
              "The GYM·10 drill on the gym page runs eight of these scenarios as a guess-the-verdict exercise against counters measured on this same torch. Take it after this section rather than before: the drill asks which verdict, and this section is where the answer stops being a guess.",
            ],
            code: {
              caption: 'two instances, two graphs (verified, torch 2.2.2 CPU, Python 3.11); the guard line comes from TORCH_LOGS=guards on the same run',
              lang: 'python',
              text: 'import torch\nfrom torch._dynamo.utils import counters\n\nclass M(torch.nn.Module):\n    def __init__(self, k):\n        super().__init__()\n        self.lin = torch.nn.Linear(4, 4)\n        self.k = k\n    def forward(self, x):\n        return self.lin(x) * self.k\n\nc1 = torch.compile(M(2.0), backend="eager")\nc2 = torch.compile(M(3.0), backend="eager")\nc1(torch.randn(2, 4)); print(dict(counters["stats"]))  # unique_graphs: 1\nc1(torch.randn(2, 4)); print(dict(counters["stats"]))  # unique_graphs: 1\nc2(torch.randn(2, 4)); print(dict(counters["stats"]))  # unique_graphs: 2\n\n# [0/0] ___check_obj_id(L[\'self\'], 4621695952)   # return self.lin(x) * self.k\n# [0/1] ___check_obj_id(L[\'self\'], 4443764496)   # return self.lin(x) * self.k',
            },
          },
          {
            h: 'the hole the guards leave',
            ps: [
              "Guard coverage is a contract, and the interesting part of any contract is what it excludes. On this torch, rebinding a module-level function that dynamo inlined does not invalidate the graph that inlined it. The compiled path keeps running the old body while eager Python runs the new one, and the two disagree until something else forces a recompile.",
              "The sequence below is five prints. Compile once with a helper that doubles. Rebind the helper to multiply by a hundred. Eager gives 100, compiled still gives 2. Then call with a new shape, which misses the tensor guard, and the fresh capture picks up the new body, after which even the old shape returns the new answer, because the recompiled graph took a dynamic dimension and now serves both.",
              "Whether it still holds on the torch you are running is a question for your torch, so re-run the snippet before trusting either outcome. The durable lesson is the shape of the risk: a guard set is a finite list of predicates, hot-swapping code under a compiled function is outside it, and the stale window closes at the next miss rather than at the moment you changed something.",
            ],
            code: {
              caption: 'a stale graph, measured (verified, torch 2.2.2 CPU, Python 3.11); re-run this on your own version before relying on either outcome',
              lang: 'python',
              text: 'import torch\n\ndef helper(t):\n    return t * 2\n\n@torch.compile(backend="eager")\ndef f(x):\n    return helper(x)\n\nx = torch.ones(3)\nprint("compiled once    :", f(x).tolist())          # [2.0, 2.0, 2.0]\n\ndef helper(t):                 # same name, new function object, new body\n    return t * 100\n\nprint("eager now        :", helper(x).tolist())     # [100.0, 100.0, 100.0]\nprint("compiled still   :", f(x).tolist())          # [2.0, 2.0, 2.0]\nprint("after a shape miss:", f(torch.ones(5)).tolist())\n# [100.0, 100.0, 100.0, 100.0, 100.0]\nprint("and back at 3    :", f(x).tolist())          # [100.0, 100.0, 100.0]',
            },
          },
        ],
        readings: [
          { label: 'guards.py at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/guards.py', note: 'GuardBuilder, one method per guard kind, including the CLOSURE_MATCH that only fires on locals' },
          { label: 'output_graph.py at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/output_graph.py', note: 'init_ambient_guards, the six guards nobody wrote, and where both counters are incremented' },
          { label: 'torch.compiler troubleshooting at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/docs/source/torch.compiler_troubleshooting.rst', note: "the maintainers' own recompile checklist, written against these log lines" },
        ],
        check: [
          {
            q: 'Two tensors have the same shape and dtype, and the second one recompiles. Name three fields that could have differed?',
            a: 'Any of the seven in check_tensor: stride, requires_grad, device, dispatch key set, or the tensor type itself. Shape is one field among several, not the whole guard.',
          },
          {
            q: 'Why does an ensemble of sixteen identical modules compile sixteen times?',
            a: 'Because a captured nn.Module method guards self with ___check_obj_id, an identity check on the instance. The class and the parameter shapes are not the cache key; the object address is.',
          },
          {
            q: 'A guard object appears in explain().out_guards but no line for it appears under TORCH_LOGS=guards. Is that a bug?',
            a: 'No. Some guard kinds emit no runtime code, and SHAPE_ENV emits code only when a dimension is symbolic. The object list is what the capture depends on; the printed lines are what gets tested per call.',
          },
        ],
        work: [
          { id: 'seven-fields', label: 'write the seven fields of check_tensor from memory, then trigger a miss on each one you can reach without changing shape', href: '#five-compiles-one-shape' },
          { id: 'guard-your-model', label: 'print the guard set of one model of your own and mark every line as ambient, argument, or global', href: '#the-guard-set-printed' },
        ],
      },
      {
        id: 'anatomy-of-a-break',
        num: 3,
        title: 'Anatomy of a break',
        lede: 'Every graph break carries a reason string, and the strings sort into four families with four different fixes. The count the survey hands you is a subtraction, which is why it sometimes reports a break that never happened and sometimes misses one that did.',
        goal: 'Given a compiled function, name which of the four break families each break belongs to from its reason string, predict the number of frames and guard sets a break costs, and say why explain and the counters can disagree.',
        sections: [
          {
            h: 'nine repros, four families',
            ps: [
              "Reason strings are dynamo telling you which internal handler gave up. Nine one-line repros produce nine distinct strings, and sorting them by what would fix them gives four families rather than nine special cases.",
              "In the first family a tensor value is being forced into Python, whether by a branch on a comparison, a `bool()`, an `.item()`, or `id()` on a tensor. None of those can be answered without running the graph, so dynamo runs what it has. The second family is a callable with no handler, which is where `print` and `os.getpid()` land; dynamo knows exactly what the object is and has no rule for tracing it.",
              "The third family is a callee inside a skipped directory. `logging.getLogger` and `torch.nn.Parameter` both report `skipfiles.SKIP_DIRS` by name, which is a policy list rather than a limitation: dynamo declines to trace into those trees. The name is version-specific, and a `trace_rules.py` already sits beside `skipfiles.py` in this same 2.2.2 tree, so re-read the string on whatever torch you run. The fourth family is one op, something whose output shape is data, with `aten.nonzero.default` as the canonical case.",
              ">> Four families, four fixes: move the value out of Python, teach dynamo the call, hoist the skipped callee, or accept a dynamic output shape.",
            ],
            table: {
              caption: 'nine repros, each run through torch._dynamo.explain (verified, torch 2.2.2 CPU, Python 3.11); reason strings verbatim, absolute paths shortened',
              cols: ['what the line does', 'reason string', 'family'],
              rows: [
                ['`if x.sum() > 0`', 'generic_jump TensorVariable()', 'a tensor value forced into Python'],
                ['`bool(x.sum())`', 'call_function BuiltinVariable(bool) [TensorVariable()] {}', 'a tensor value forced into Python'],
                ['`x.max().item()`', 'Tensor.item', 'a tensor value forced into Python'],
                ['`id(y)`', 'call_id with args (TensorVariable(),)', 'a tensor value forced into Python'],
                ['`print()`', 'call_function BuiltinVariable(print) [] {}', 'a callable with no handler'],
                ['`os.getpid()`', 'call_method UserDefinedObjectVariable(getpid) __call__ [] {}', 'a callable with no handler'],
                ['`logging.getLogger("p")`', "'skip function getLogger in file .../logging/__init__.py'', skipped according skipfiles.SKIP_DIRS'", 'a callee in a skipped directory'],
                ['`torch.nn.Parameter(...)`', "'skip function Parameter in file .../torch/nn/parameter.py'", 'a callee in a skipped directory'],
                ['`torch.nonzero(x)`', 'dynamic shape operator: aten.nonzero.default', 'an output shape that is data'],
              ],
            },
          },
          {
            h: 'the cost is frames, not fusion',
            ps: [
              "Two prints in one function, and the accounting comes out at three of everything: three frames, three graphs, three guard sets. The guard sets are the part worth staring at, because each one guards a different local. The original frame guards `L['x']`, the first resume frame guards `L['y']`, the second guards `L['z']`.",
              "Which means the intermediate tensors flowing across a break are now cache keys. A break in the middle of a model turns an internal activation into something whose stride and requires_grad get checked on every call, and into something a later shape change can invalidate independently of the input that caused it.",
              "One extra line shows up at each break site in the guard dump: `not ___needs_nopython()`. The break is recorded in the guard itself, which is how the same cached code refuses to run when the function is later called under `fullgraph=True`.",
            ],
            code: {
              caption: 'two breaks in one function (verified, torch 2.2.2 CPU, Python 3.11); guard lines trimmed to the tensor check, counters printed at the end',
              lang: 'text',
              text: "[0/0_1] check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[4], stride=[1])  # y = torch.sin(x)  # frames.py:6 in f\n[0/0_1] not ___needs_nopython()                                       # print()  # frames.py:7 in f\n[1/0_1] check_tensor(L['y'], ... size=[4], stride=[1])  # z = y * 2  # frames.py:8 in resume_in_f\n[1/0_1] not ___needs_nopython()                                       # print()  # frames.py:9 in resume_in_f\n[2/0]   check_tensor(L['z'], ... size=[4], stride=[1])  # return z + 1  # frames.py:10 in resume_in_f\n\nframes     : {'total': 3, 'ok': 3}\nstats      : {'calls_captured': 3, 'unique_graphs': 3}\ngraph_break: {'call_function BuiltinVariable(print) [] {}': 2}",
            },
          },
          {
            h: 'the count that subtracts',
            ps: [
              "`explain(...).graph_break_count` is not a count of breaks. In `torch/_dynamo/eval_frame.py` it is one line, `graph_break_count = graph_count - 1`, where `graph_count` is the number of graphs the accumulating backend actually received. Empty graphs never reach a backend, so any break with nothing captured before it is invisible to the subtraction.",
              "Two measurements make the gap concrete. A function whose first statement is `print()` compiles two frames and reports zero breaks, because the leading segment held no ops. A function with no tensor math at all reports minus one. Neither number is wrong for what it measures; they are wrong for the question people ask them.",
              "`counters[\"graph_break\"]` is the counted instrument: a dict keyed by reason string, incremented once per break as it happens. Two prints give a value of 2 under one key. Use the counter dict when you want to know how many and why, and use explain when you want the graphs and the guards it also returns.",
            ],
            code: {
              caption: 'the two instruments disagreeing, on purpose (verified, torch 2.2.2 CPU, Python 3.11)',
              lang: 'python',
              text: 'import torch\nimport torch._dynamo as dynamo\nfrom torch._dynamo.utils import counters\n\ndef print_first(x):\n    print()\n    return x + 1\n\ndef no_tensors(x):\n    try:\n        raise ValueError("no")\n    except ValueError:\n        return 1\n\nprint(dynamo.explain(print_first)(torch.randn(4)).graph_break_count)   # 0\nprint(dynamo.explain(no_tensors)(torch.randn(4)).graph_break_count)    # -1\n\ndynamo.reset(); counters.clear()\ntorch.compile(print_first, backend="eager")(torch.randn(4))\nprint(dict(counters["frames"]))        # {\'total\': 2, \'ok\': 2}\nprint(dict(counters["graph_break"]))   # {\'call_function BuiltinVariable(print) [] {}\': 1}',
            },
          },
          {
            h: 'the same conditions, as errors',
            ps: [
              "`fullgraph=True` does not change what dynamo can trace. It changes what happens when it cannot, and the two exception types it raises are worth telling apart. A data-dependent branch raises `torch._dynamo.exc.UserError` with the message about dynamic control flow and a pointer at `functorch.experimental.control_flow.cond`. A `print` raises `torch._dynamo.exc.Unsupported` carrying the same reason string the counter would have recorded.",
              "The split is a rough map of intent. `UserError` says the program as written has no whole-graph meaning and someone has to choose a different formulation. `Unsupported` says dynamo has no rule for this, which may be true only of this version.",
              "Chapter 7 covers the other tool that refuses instead of splitting, `torch.export`, and the difference in contract between the two is the mastery item on that chapter rather than this one. What belongs here is the mechanism: the refusal is the same trace, with the break site turned into a raise.",
            ],
          },
        ],
        readings: [
          { label: 'symbolic_convert.py at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/symbolic_convert.py', note: 'generic_jump and the unimplemented() calls that produce most of these strings' },
          { label: 'skipfiles.py at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/skipfiles.py', note: 'SKIP_DIRS itself, alongside the trace_rules.py that is replacing it in the same tree' },
          { label: 'eval_frame.py explain at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/eval_frame.py', note: 'the one-line subtraction behind graph_break_count, in context' },
        ],
        check: [
          {
            q: 'A break reports skipfiles.SKIP_DIRS in its reason. What kind of fix does that point at?',
            a: 'A structural one: the callee lives in a directory dynamo refuses to trace into, so hoist that call out of the compiled region rather than trying to make it traceable.',
          },
          {
            q: 'Why can explain report zero breaks for a function that clearly broke?',
            a: 'Because graph_break_count is graph_count minus one, and empty segments never reach the backend. A break before any tensor op leaves one graph and reports zero.',
          },
          {
            q: 'What do the guard sets look like after two breaks in one function?',
            a: 'Three of them, one per frame, each guarding the locals live at its own entry point. The intermediates crossing the breaks become cache keys with their own stride and requires_grad checks.',
          },
        ],
        work: [
          { id: 'sort-your-breaks', label: 'collect the reason strings from one model of your own and sort every break into one of the four families', href: '#nine-repros-four-families' },
          { id: 'count-both-ways', label: 'count the breaks in one function with explain and with counters["graph_break"], and explain any disagreement', href: '#the-count-that-subtracts' },
        ],
      },
      {
        id: 'the-cache-line',
        num: 4,
        title: 'The cache line',
        lede: 'The cache is a linked list hanging off your code object, and you can walk it from Python. What it does when it fills up is throw away everything you paid for, and what it does about changing shapes is stop testing equality and start testing a range.',
        goal: 'Walk the cache line of any compiled function, name both limits and which one applies, predict what a program does after it bursts the cache, and say when a shape becomes symbolic and what guard replaces the equality check.',
        sections: [
          {
            h: 'walk the linked list yourself',
            ps: [
              "The note at the top of `torch/_dynamo/cache_size.py` describes the structure in one sentence: a linked list of entries, each a check function, an output code object, and a next pointer, hanging off the code object's `co_extra` scratch space. `_debug_get_cache_entry_list` hands you the head, and the walk is six lines.",
              "Three captures of one function give three entries, each with a `check_fn` whose qualified name is `___make_guard_fn.<locals>.guard`, a closure generated at compile time and holding the predicates lesson two printed. Every call walks this list from the head and runs check functions until one passes.",
              "The recompile log therefore prints one failure line per entry rather than one per call. Three identical failure lines are not a repeated message; they are three cache entries each rejecting the same arguments for the same reason.",
            ],
            code: {
              caption: 'the cache line, walked from Python (verified, torch 2.2.2 CPU, Python 3.11)',
              lang: 'python',
              text: 'import torch\nfrom torch._C._dynamo import eval_frame\n\ndef f(x, alpha):\n    return torch.sin(x) * alpha\n\ncf = torch.compile(f, backend="eager")\ncf(torch.randn(4), 2.0)\ncf(torch.randn(8), 2.0)\ncf(torch.randn(4), 5.0)\n\nentry, n = eval_frame._debug_get_cache_entry_list(f.__code__), 0\nwhile entry is not None:\n    n += 1\n    print(n, entry.code.co_name, entry.check_fn.__qualname__)\n    entry = entry.next\nprint("entries on f.__code__:", n)\n# 1 f ___make_guard_fn.<locals>.guard\n# 2 f ___make_guard_fn.<locals>.guard\n# 3 f ___make_guard_fn.<locals>.guard\n# entries on f.__code__: 3',
            },
          },
          {
            h: 'two limits, one message',
            ps: [
              "`cache_size_limit` is 8 and `accumulated_cache_size_limit` is 64, and the note in `cache_size.py` explains why one number was not enough. Identity guards on `nn.Module` instances mean a normal model produces many entries on one code object through no fault of the author, so the small limit counts entries per identity bucket and the large one caps the code object overall.",
              "Sixty-four is therefore the number that stops an ensemble. Ten instances of one module class compile ten graphs and nothing complains; the sixty-fifth instance trips the ceiling. Eight is the number that stops a function specializing on a Python value, which is the more common accident: nine distinct floats through one argument is enough.",
              "Both cases print the same warning, and it names `config.cache_size_limit (8)` either way, because `will_compilation_exceed` checks the bucket limit and the total and the log line only ever formats the first. When you see that warning after more than eight compiles, read it as the accumulated limit and go look for identity guards.",
            ],
            table: {
              caption: 'the two limits, from torch/_dynamo/cache_size.py and config.py at 39901f2; both ceilings hit on this machine',
              cols: ['limit', 'default', 'counts', 'what trips it'],
              rows: [
                ['cache_size_limit', '8', 'entries sharing the same ID_MATCH objects', 'a function specializing on a Python value, or one module seeing nine guard sets'],
                ['accumulated_cache_size_limit', '64', 'all entries on the code object', 'many instances of one module class, each with its own identity guard'],
              ],
            },
          },
          {
            h: 'the ninth guard set throws the other eight away',
            ps: [
              "The failure mode is worse than a stop. Watch the entry count across twelve calls with twelve distinct constants: it climbs to eight, and on the ninth call it goes to zero. The whole cache line is dropped, the code object is skipped from then on, the frame counter stops moving, and every later call runs eager. Eight compiles paid for and discarded in one call.",
              "Dynamo does not stop working at that point, it moves down. In the variant below, `f` calls a helper; once `f` is skipped, the callback still sees the helper's frame and compiles that instead, so the helper's own code object grows an entry while `f` has none. Capture drops one level and quietly keeps a fraction of the benefit.",
              "Compare the layer below, where the same cache-key discipline is paid a second time. The torch_xla dynamo bridge, in the lesson /pytorch/bridges/three-modes-one-machine, throws on a cache miss it cannot recover from, because it discarded the graph. Here, bursting the limit silently returns you to eager. One failure is loud and one is a slow benchmark, and neither is a recompile.",
            ],
            code: {
              caption: 'twelve calls, twelve constants (verified, torch 2.2.2 CPU, Python 3.11); the warning fires once, on call 8, the ninth distinct guard set',
              lang: 'text',
              text: "i  entries  stats                                        frames\n0  1        {'calls_captured': 2,  'unique_graphs': 1}   {'total': 1, 'ok': 1}\n3  4        {'calls_captured': 8,  'unique_graphs': 4}   {'total': 4, 'ok': 4}\n7  8        {'calls_captured': 16, 'unique_graphs': 8}   {'total': 8, 'ok': 8}\n8  0        {'calls_captured': 16, 'unique_graphs': 8}   {'total': 9, 'ok': 8}\n9  0        {'calls_captured': 16, 'unique_graphs': 8}   {'total': 9, 'ok': 8}\n11 0        {'calls_captured': 16, 'unique_graphs': 8}   {'total': 9, 'ok': 8}\n\nWARNING torch._dynamo hit config.cache_size_limit (8)\nWARNING    function: 'f' (burst.py:5)\nWARNING    last reason: L['alpha'] == 0.0     # return torch.sin(x) * alpha\n\n# and with a helper in the middle, capture moves one frame down:\n# call 7   f: 8 entries   helper: 0 entries   frames {'total': 8,  'ok': 8}\n# call 8   f: 0 entries   helper: 1 entry     frames {'total': 10, 'ok': 9}",
            },
          },
          {
            h: 'a size that stops being a number',
            ps: [
              "The chapter's rule is that a new shape triggers a new capture. True once. `automatic_dynamic_shapes` is on by default, and the second distinct size for a dimension makes dynamo recompile that dimension as symbolic rather than as another constant. Shapes 4 then 8 cost two compiles. Shapes 16 and 32 after them cost nothing.",
              "The guard is where the change is legible. The first entry checks `size=[4]`. The second checks `size=[None]` and adds a separate line, `2 <= L['x'].size()[0]`, because zero and one are specialized: a dimension of length 1 broadcasts, and code that is correct for 1 is not automatically correct for n. So a call at size 1 recompiles even against a dynamic entry, and that third graph checks `size=[1]`.",
              "`mark_dynamic(x, 0)` moves the same decision one compile earlier. Marked from the first call, sizes 4, 8, 16 and 32 share one graph and there is no static entry at all. It buys the first compile back, not correctness, and the size-1 call still compiles separately. `dynamic=False` goes the other way, turning automatic dynamic off and giving three shapes three graphs.",
              "What dynamic shapes do not do is remove guards. Branch on `x.shape[0] > 4` inside a marked-dynamic function and the equality check becomes an inequality: `L['x'].size()[0] > 4`, which holds for 8 and 16 and fails at 3. Symbolic shapes move the predicate from equality to a range; the range still comes from your Python.",
            ],
            code: {
              caption: 'automatic dynamic, mark_dynamic, and a branch on a symbolic size (verified, torch 2.2.2 CPU, Python 3.11); guard lines from TORCH_LOGS=guards on the same runs',
              lang: 'python',
              text: 'import torch\nfrom torch._dynamo.utils import counters\n\n@torch.compile(backend="eager")\ndef f(x):\n    return torch.sin(x) + 1\n\nfor n in (4, 8, 16, 32, 1):\n    f(torch.randn(n))\n    print("plain ", n, counters["stats"]["unique_graphs"])\n# plain  4 1     [0/0] check_tensor(... size=[4], stride=[1])\n# plain  8 2     [0/1] check_tensor(... size=[None], stride=[1])\n# plain  16 2          [0/1] 2 <= L[\'x\'].size()[0]\n# plain  32 2\n# plain  1 3     [0/2] check_tensor(... size=[1], stride=[1])\n\ncounters.clear(); torch._dynamo.reset()\n\n@torch.compile(backend="eager")\ndef g(x):\n    return torch.sin(x) + 1\n\nfor n in (4, 8, 16, 32, 1):\n    y = torch.randn(n)\n    torch._dynamo.mark_dynamic(y, 0)\n    g(y)\n    print("marked", n, counters["stats"]["unique_graphs"])\n# marked 4 1 · marked 8 1 · marked 16 1 · marked 32 1 · marked 1 2',
            },
          },
        ],
        readings: [
          { label: 'cache_size.py at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/cache_size.py', note: 'the note on why there are two limits, then the twenty lines that implement it' },
          { label: 'dynamic shapes, from the maintainers', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/docs/source/torch.compiler_dynamic_shapes.rst', note: 'automatic dynamic and the 0/1 specialization, in their own words' },
          { label: 'config.py at 39901f2', url: 'https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/_dynamo/config.py', note: 'the defaults quoted here: 8, 64, automatic_dynamic_shapes, assume_static_by_default' },
        ],
        check: [
          {
            q: 'Your function compiled nine times and now runs slower than eager. What happened to the eight graphs?',
            a: 'They were discarded. On the ninth distinct guard set the cache line is dropped to zero entries and the code object is skipped, so every later call runs eager with nothing cached.',
          },
          {
            q: 'A model warns about cache_size_limit (8) after forty compiles. Which limit actually stopped it?',
            a: 'The accumulated one, at 64 entries on the code object. The warning always formats config.cache_size_limit, but the check is bucket limit or total, and forty entries means identity guards spreading across buckets.',
          },
          {
            q: 'Why does a dynamic-shape entry carry a guard reading 2 <= size()[0]?',
            a: 'Because 0 and 1 are specialized rather than symbolic. A length-1 dimension broadcasts, so code compiled for a general n is not valid for it, and a call at size 1 compiles its own entry.',
          },
        ],
        work: [
          { id: 'walk-the-line', label: 'walk the cache line of one compiled function of your own and name what each entry is specialized on', href: '#walk-the-linked-list-yourself' },
          { id: 'burst-on-purpose', label: 'burst the cache on purpose with nine guard sets, then find the moment the entry count drops to zero', href: '#the-ninth-guard-set-throws-the-other-eight-away' },
          { id: 'mark-one-dim', label: 'take a loop with a ragged last batch and count its graphs with and without mark_dynamic on the batch dimension', href: '#a-size-that-stops-being-a-number' },
        ],
      },
    ],
  },
]
