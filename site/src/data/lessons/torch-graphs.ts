// New file: site/src/data/lessons/torch-graphs.ts
// What happens to a captured graph between dynamo and the code that finally runs:
// the joint forward-backward trace and the cut that splits it, the FX node grammar,
// the three op vocabularies decompositions move between, and the contract
// torch.export enforces that dynamo does not. Every printout is a local run on
// torch 2.2.2 CPU, Python 3.11; the aot_autograd and partitioner source lines are
// from the v2.2.2 tag.
import type { UnitLessons } from './index'

export const TORCH_GRAPHS_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:graphs',
    lessons: [
      {
        id: 'one-call-three-graphs',
        num: 1,
        title: 'One call, three graphs',
        lede: 'Compile a module, call it once, and three graphs get built before a single fused kernel exists. Only two of them speak ATen, and the one you have to read to understand memory is the middle one.',
        goal: 'Given a compiled training step, name which component builds each of the three graphs, read a forward graph\u2019s return list as the list of tensors the backward will be handed, and predict from the partitioner alone which of those tensors gets recomputed instead of saved.',
        sections: [
          {
            h: 'the backend sees a graph that still speaks python',
            ps: [
              "A compiler backend under torch.compile is just a function that takes an FX graph and gives back something callable. That makes it the easiest instrument in the whole stack: write a backend that prints its argument and returns it unchanged, and you get to see exactly what dynamo produced, with nothing between you and the object.",
              "Do that on a two-layer step and the graph that arrives is not what chapter 7's closing line would have you expect. `aot_module_simplified` is the function that turns it into ATen, and it runs inside the backend, after dynamo is already done. So there is a moment where the captured graph exists and is not ATen at all.",
              "The script below stacks three printers: one for the graph dynamo hands over, one for the forward that comes out of aot_autograd, and one for the backward. All three fire from a single compiled call plus a single `.backward()`.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a backend that prints everything it is handed',
              lang: 'python',
              text: 'import torch\nimport torch.nn as nn\nfrom torch._functorch.aot_autograd import aot_module_simplified\nfrom functorch.compile import make_boxed_func\n\nclass MLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 4)\n    def forward(self, x):\n        return torch.relu(self.fc(x)).sum()\n\ndef show(tag):\n    def compiler(gm, inputs):\n        print(f"--- {tag}")\n        print(gm.code.strip())\n        return make_boxed_func(gm.forward)\n    return compiler\n\ndef backend(gm, example_inputs):\n    print("--- what dynamo handed the backend")\n    print(gm.code.strip())\n    return aot_module_simplified(gm, example_inputs,\n                                 fw_compiler=show("aot forward"),\n                                 bw_compiler=show("aot backward"))\n\ntorch.manual_seed(0)\nloss = torch.compile(MLP(), backend=backend)(torch.randn(2, 3))\nloss.backward()',
            },
          },
          {
            h: 'three printouts, three vocabularies',
            ps: [
              "Read the first block and count the targets. `self.L__self___fc(l_x_)` is a module call, still a whole `nn.Linear` sitting unopened in the graph. `torch.relu` is the Python-level function, not `aten.relu.default`. `relu.sum()` is a method call on a tensor. Nothing in that graph is an operator overload yet.",
              "The second block is the same computation with every one of those opened up. The Linear became a transpose and an `addmm`; the method call became `aten.sum.default`. Dynamo never did that rewriting. It happened when aot_autograd re-traced the graph under a dispatcher that records what each Python-level call actually dispatches to.",
              "The third block is the piece chapter 7 promises exists and never shows. It takes `primals_3` and `relu`, which are two of the three values the forward returned, plus `tangents_1`, which is the incoming gradient. `threshold_backward` is the relu derivative; the two transposes and the `mm` are the weight gradient; `sum.dim_IntList` over dimension 0 is the bias gradient.",
              '>> The forward graph returns more than the forward result. The extras are the backward\u2019s inputs.',
            ],
            code: {
              caption: 'the verbatim output of the script above (verified, torch 2.2.2 CPU)',
              lang: 'text',
              text: '--- what dynamo handed the backend\ndef forward(self, L_x_ : torch.Tensor):\n    l_x_ = L_x_\n    l__self___fc = self.L__self___fc(l_x_);  l_x_ = None\n    relu = torch.relu(l__self___fc);  l__self___fc = None\n    sum_1 = relu.sum();  relu = None\n    return (sum_1,)\n--- aot forward\ndef forward(self, primals_1, primals_2, primals_3):\n    t = torch.ops.aten.t.default(primals_1);  primals_1 = None\n    addmm = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = t = None\n    relu = torch.ops.aten.relu.default(addmm);  addmm = None\n    sum_1 = torch.ops.aten.sum.default(relu)\n    return [sum_1, primals_3, relu]\n--- aot backward\ndef forward(self, primals_3, relu, tangents_1):\n    detach = torch.ops.aten.detach.default(relu);  relu = None\n    expand = torch.ops.aten.expand.default(tangents_1, [2, 4]);  tangents_1 = None\n    detach_1 = torch.ops.aten.detach.default(detach);  detach = None\n    threshold_backward = torch.ops.aten.threshold_backward.default(expand, detach_1, 0);  expand = detach_1 = None\n    t_1 = torch.ops.aten.t.default(threshold_backward)\n    mm = torch.ops.aten.mm.default(t_1, primals_3);  t_1 = primals_3 = None\n    t_2 = torch.ops.aten.t.default(mm);  mm = None\n    sum_2 = torch.ops.aten.sum.dim_IntList(threshold_backward, [0], True);  threshold_backward = None\n    view = torch.ops.aten.view.default(sum_2, [4]);  sum_2 = None\n    t_3 = torch.ops.aten.t.default(t_2);  t_2 = None\n    return [t_3, view, None]',
            },
            table: {
              caption: 'the same computation at three stages, from the printout above',
              cols: ['stage', 'built by', 'what the node targets look like'],
              rows: [
                ['captured', 'dynamo, from Python bytecode', 'call_module on L__self___fc, call_function on torch.relu, call_method sum'],
                ['forward', 'aot_autograd, after partitioning the joint trace', 'aten.t.default, aten.addmm.default, aten.relu.default, aten.sum.default'],
                ['backward', 'the same joint trace, other half', 'aten.threshold_backward.default, aten.mm.default, aten.sum.dim_IntList'],
              ],
            },
          },
          {
            h: 'before the cut there was one joint graph',
            ps: [
              "Two separate graphs come out, but nothing traced them separately. aot_autograd builds one graph containing both halves and then hands it to a partitioner, and the docstring on `create_aot_dispatcher_function` says so in order: trace to a joint graph, pass it through `partition_fn` to isolate the forward and backward portions, compile each with its own compiler.",
              "You can watch the cut happen by writing a `partition_fn` that prints its argument and then delegates. The joint graph below is from a smaller program than the MLP, `(x @ w).sin().sum()`, because the joint form is easier to read when the whole thing fits on a screen.",
              "Two naming conventions are worth learning here, because they show up in every aot printout you will ever read. `primals` are the forward inputs. `tangents` are the incoming gradients. In the joint graph they arrive as arguments to the same function, and the return carries the forward output and the input gradients together, with `None` in the slot for the input that did not require grad.",
            ],
            code: {
              caption: 'the joint graph, verbatim, before any partitioner touched it (verified, torch 2.2.2 CPU)',
              lang: 'text',
              text: 'def forward(self, primals, tangents):\n    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)\n    mm = torch.ops.aten.mm.default(primals_1, primals_2);  primals_2 = None\n    sin = torch.ops.aten.sin.default(mm)\n    sum_1 = torch.ops.aten.sum.default(sin);  sin = None\n    expand = torch.ops.aten.expand.default(tangents_1, [2, 4]);  tangents_1 = None\n    cos = torch.ops.aten.cos.default(mm);  mm = None\n    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None\n    t = torch.ops.aten.t.default(primals_1);  primals_1 = None\n    mm_1 = torch.ops.aten.mm.default(t, mul);  t = mul = None\n    return pytree.tree_unflatten([sum_1, None, mm_1], self._out_spec)',
              full: {
                label: 'the script that printed it, and its output',
                text: 'import torch\nfrom functorch.compile import aot_function, make_boxed_func, default_partition\n\ndef nop(gm, inputs):\n    return make_boxed_func(gm.forward)\n\ndef peek(joint_gm, joint_inputs, **kwargs):\n    print(joint_gm.code.strip())\n    return default_partition(joint_gm, joint_inputs, **kwargs)\n\ndef f(x, w):\n    return (x @ w).sin().sum()\n\nx = torch.randn(2, 3)\nw = torch.randn(3, 4, requires_grad=True)\naot_function(f, fw_compiler=nop, bw_compiler=nop, partition_fn=peek)(x, w).backward()\n\n# def forward(self, primals, tangents):\n#     primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)\n#     mm = torch.ops.aten.mm.default(primals_1, primals_2);  primals_2 = None\n#     sin = torch.ops.aten.sin.default(mm)\n#     sum_1 = torch.ops.aten.sum.default(sin);  sin = None\n#     expand = torch.ops.aten.expand.default(tangents_1, [2, 4]);  tangents_1 = None\n#     cos = torch.ops.aten.cos.default(mm);  mm = None\n#     mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None\n#     t = torch.ops.aten.t.default(primals_1);  primals_1 = None\n#     mm_1 = torch.ops.aten.mm.default(t, mul);  t = mul = None\n#     return pytree.tree_unflatten([sum_1, None, mm_1], self._out_spec)',
              },
            },
          },
          {
            h: 'two partitioners, two memory bills',
            ps: [
              "Where the cut falls is a choice, and PyTorch ships two answers to it. `default_partition` collects the operators between the forward inputs and the forward outputs, and its docstring names the consequence directly: the stashed tensors become the output of the generated forward graph. `min_cut_rematerialization_partition` opens with a different sentence, that the backward recomputes the forward, trading memory bandwidth against computation.",
              "Run the same three-op chain through both and the difference is countable. Under the default cut, the forward returns four values and the backward takes four arguments. Under the min-cut, both numbers drop to two, and `sin` and `cos` reappear as the first two lines of the backward, recomputed from the input rather than carried across the boundary.",
              "Which one you get depends on how you entered. `aot_function` defaults to `default_partition`. Inductor does not use that default: `compile_fx.py` wraps `min_cut_rematerialization_partition` in its own `partition_fn` and passes that down, so an ordinary `torch.compile` on the inductor backend is already recomputing rather than saving wherever the heuristic says it should.",
              "This is activation checkpointing, decided per tensor by a solver instead of per block by you. It also explains a shape of profile that otherwise looks like a bug: a backward that runs more ops than the forward did, on a model you never wrote a checkpoint wrapper for.",
            ],
            code: {
              caption: 'both partitioners on x.sin().cos().sin().sum(), verbatim (verified, torch 2.2.2 CPU)',
              lang: 'text',
              text: 'default_partition: forward returns 4, backward takes 4\ndef forward(self, primals_1):\n    sin = torch.ops.aten.sin.default(primals_1)\n    cos = torch.ops.aten.cos.default(sin)\n    sin_1 = torch.ops.aten.sin.default(cos)\n    sum_1 = torch.ops.aten.sum.default(sin_1);  sin_1 = None\n    return [sum_1, primals_1, sin, cos]\nmin_cut_rematerialization_partition: forward returns 2, backward takes 2\ndef forward(self, primals_1):\n    sin = torch.ops.aten.sin.default(primals_1)\n    cos = torch.ops.aten.cos.default(sin);  sin = None\n    sin_1 = torch.ops.aten.sin.default(cos);  cos = None\n    sum_1 = torch.ops.aten.sum.default(sin_1);  sin_1 = None\n    return [sum_1, primals_1]',
            },
            table: {
              caption: 'the two partitioners, measured on the chain above',
              cols: ['partitioner', 'forward returns', 'backward placeholders', 'used by default from'],
              rows: [
                ['default_partition', '4 (loss plus 3 saved)', '4 (3 saved plus 1 tangent)', 'aot_function, aot_module'],
                ['min_cut_rematerialization_partition', '2 (loss plus 1 saved)', '2 (1 saved plus 1 tangent)', 'inductor, so plain torch.compile'],
              ],
            },
          },
          {
            h: 'the backward compiles when you call backward',
            ps: [
              "One more thing the printers tell you, and it is about time rather than structure. Swap the graph printing for a one-line marker in each compiler and the ordering is unambiguous: the forward compiler runs during the compiled call, the backward compiler does not run until `.backward()` is invoked.",
              "So a first training step pays for two compilations at two different moments, and a benchmark that times only the forward will miss the second one entirely. Chapter 9's harness exists for exactly this class of measurement error, and this is one more thing its warmup has to cover.",
              "The lazy backward also means a model you compile and only ever run under `torch.no_grad()` never builds a backward graph at all. Nothing warns you either way. The compiler simply never gets called.",
            ],
            code: {
              caption: 'markers instead of graphs, same script otherwise (verified, torch 2.2.2 CPU)',
              lang: 'text',
              text: 'before the forward call\n[forward compiler ran]\nforward call returned\n[backward compiler ran]\nbackward returned',
            },
          },
        ],
        readings: [
          { label: 'aot_autograd.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/_functorch/aot_autograd.py', note: 'create_aot_dispatcher_function is where the joint-graph-then-partition contract is written down' },
          { label: 'partitioners.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/_functorch/partitioners.py', note: 'both partitioners, and the min-cut solver that decides save against recompute' },
          { label: 'compile_fx.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/_inductor/compile_fx.py', note: 'the partition_fn inductor actually installs, near line 1126' },
        ],
        check: [
          {
            q: 'A forward graph returns three values but the module returns one tensor. What are the other two?',
            a: 'Saved tensors. The partitioner turns every value the backward needs into an extra output of the forward graph, and those extras become the backward graph\u2019s leading placeholders.',
          },
          {
            q: 'Why does a backward graph sometimes contain ops the forward already ran?',
            a: 'Because the min-cut partitioner chose to recompute them rather than save them. Inductor installs min_cut_rematerialization_partition, so a plain torch.compile trades extra backward flops for lower peak memory.',
          },
          {
            q: 'You timed a compiled step by calling the module in a loop and never calling backward. What did you fail to measure?',
            a: 'The backward compilation, which does not happen until .backward() is first called, and the backward execution itself. The forward compiler runs during the call; the backward compiler waits.',
          },
        ],
        work: [
          { id: 'print-three-graphs', label: 'run the printing backend on one of your own modules and label every line of the backward graph with the forward op it differentiates', href: '#three-printouts-three-vocabularies' },
          { id: 'count-the-cut', label: 'run one module through both partitioners and write down the two saved-tensor counts before you look at the output', href: '#two-partitioners-two-memory-bills' },
        ],
      },
      {
        id: 'reading-a-real-graph',
        num: 2,
        title: 'Reading a real graph',
        lede: 'An FX printout has exactly six kinds of line, and one small module can produce all of them. Once you can name the kind, the target, and the user count, nothing in a graph dump is decoration.',
        goal: 'Given any FX printout from anywhere in the torch.compile stack, name each node\u2019s opcode, say what its target is an instance of, explain what the trailing overload name selects, and find the shape of any intermediate without running the graph.',
        sections: [
          {
            h: 'six kinds of node, one six-line graph',
            ps: [
              "FX has a fixed vocabulary of node opcodes, and there are only six. A module with one linear layer, one buffer, one operator, and two method calls produces five of them, and the sixth is the return. That is the whole grammar, which is why the same reading habit works on a dynamo capture, an aot forward, and an exported program.",
              "`torch.fx.symbolic_trace` is the oldest and simplest way to get one. It runs your forward with proxy objects and records what happens, which is not what dynamo does and not what export does, but it produces the same node types, and it leaves modules unopened so you can see `call_module` and `get_attr` in the same picture.",
              "Read the printout below top to bottom. `%x` is the input. `%fc` calls a submodule by name. `%shift` fetches a registered buffer off the module, no computation at all. `%add` calls `operator.add`, the Python built-in behind the `+`. `%relu` and `%sum_1` are method calls on the tensor that came before. The last line is the return.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): every FX opcode in one trace',
              lang: 'python',
              text: 'import torch\nimport torch.nn as nn\nfrom collections import Counter\n\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 4)\n        self.register_buffer("shift", torch.zeros(4))\n    def forward(self, x):\n        return (self.fc(x) + self.shift).relu().sum()\n\ngm = torch.fx.symbolic_trace(M())\nprint(gm.graph)\nprint(Counter(n.op for n in gm.graph.nodes))',
              full: {
                label: 'the script and its verbatim output',
                text: 'import torch\nimport torch.nn as nn\nfrom collections import Counter\n\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 4)\n        self.register_buffer("shift", torch.zeros(4))\n    def forward(self, x):\n        return (self.fc(x) + self.shift).relu().sum()\n\ngm = torch.fx.symbolic_trace(M())\nprint(gm.graph)\nprint(Counter(n.op for n in gm.graph.nodes))\n\n# graph():\n#     %x : [num_users=1] = placeholder[target=x]\n#     %fc : [num_users=1] = call_module[target=fc](args = (%x,), kwargs = {})\n#     %shift : [num_users=1] = get_attr[target=shift]\n#     %add : [num_users=1] = call_function[target=operator.add](args = (%fc, %shift), kwargs = {})\n#     %relu : [num_users=1] = call_method[target=relu](args = (%add,), kwargs = {})\n#     %sum_1 : [num_users=1] = call_method[target=sum](args = (%relu,), kwargs = {})\n#     return sum_1\n# Counter({\'call_method\': 2, \'placeholder\': 1, \'call_module\': 1, \'get_attr\': 1, \'call_function\': 1, \'output\': 1})',
              },
            },
            table: {
              caption: 'the six FX opcodes, each one present in the trace above',
              cols: ['opcode', 'target is', 'appears in the printout as'],
              rows: [
                ['placeholder', 'the argument name', '%x, an input to the graph'],
                ['get_attr', 'a dotted path into the module', '%shift, the registered buffer'],
                ['call_function', 'a free Python callable or an OpOverload', '%add, targeting operator.add'],
                ['call_method', 'a method name, called on args[0]', '%relu and %sum_1'],
                ['call_module', 'a submodule path, invoked whole', '%fc, the nn.Linear left unopened'],
                ['output', 'nothing; args[0] carries the returned structure', 'the final return line'],
              ],
            },
          },
          {
            h: 'the anatomy of one printed line',
            ps: [
              "Every non-return line has the same five parts, and the printout spaces them out the same way each time: `%name : [num_users=N] = opcode[target=T](args = (...), kwargs = {...})`. The name is the SSA-style handle other lines refer to. The opcode is one of the six. The target is what the opcode dispatches on.",
              "`num_users` is the one people skim past, and it is the field that tells you whether a value is about to be freed. It counts how many later nodes take this node as an argument, which is why the `gm.code` view of the same graph is full of `;  x = None` statements: the codegen drops the reference as soon as the last user has consumed it, so the buffer can be released mid-graph.",
              "It is also the field a graph pass reads to decide whether an edit is legal. Dead code elimination is exactly the rule that a node with no users and no side effects can go, and you can see the result rather than the rule in the min-cut forward from lesson one, where the values the backward stopped needing simply do not appear in the return list.",
            ],
          },
          {
            h: 'default is an overload name',
            ps: [
              "The `.default` that ends almost every ATen target in these graphs is not punctuation. `torch.ops.aten.sum` is an overload packet, a family of C++ signatures sharing one name, and the last component picks one member of that family. Ask the packet directly and it lists them.",
              "So `aten.sum.default` is the reduce-everything signature and `aten.sum.dim_IntList` is the one taking a dimension list, and the aot backward in lesson one used the second because the bias gradient sums over dimension 0 only. A graph that names a specific overload has already resolved which C++ kernel it means, which is why an FX graph at this level is unambiguous in a way your Python source is not.",
              "Counting the surface makes the scale concrete. On this build, `torch.ops.aten` exposes 817 packets, which between them carry 2124 overloads. Seven of the packets do not report overloads at all. `torch.ops.prims` exposes 132. Those three numbers are the reason decompositions exist, which is the next lesson.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the size of the operator surface, and one packet\u2019s overloads',
              lang: 'python',
              text: 'import torch\n\npackets = dir(torch.ops.aten)\ntotal = 0\nskipped = 0\nfor name in packets:\n    try:\n        total += len(getattr(torch.ops.aten, name).overloads())\n    except Exception:\n        skipped += 1\nprint("aten packets:", len(packets), "overloads:", total, "packets with none:", skipped)\nprint("prims packets:", len(dir(torch.ops.prims)))\nprint("aten.sum overloads:", torch.ops.aten.sum.overloads())\n\n# aten packets: 817 overloads: 2124 packets with none: 7\n# prims packets: 132\n# aten.sum overloads: [\'dim_IntList\', \'default\', \'dim_DimnameList\', \'DimnameList_out\', \'IntList_out\', \'out\', \'int\', \'float\', \'complex\', \'bool\']',
            },
          },
          {
            h: 'the shapes are in the metadata',
            ps: [
              "The printed graph shows no shapes and no dtypes, which makes it look like less information than it holds. Every node carries a `meta` dict, and for graphs that went through fake-tensor tracing the key `val` holds a tensor with the right shape, dtype and device and no storage behind it. Reading that dict turns a graph dump into a shape table.",
              "This is the same shape-propagation you would otherwise do by hand while staring at a matmul, done once during tracing and kept. Note where the `t` node lands in the table below: `(4, 3)` in, `(3, 4)` out, which is the transpose an `nn.Linear` needs before `addmm` can consume the weight.",
              "One caution on the placeholder names in that output. `arg0_1` and `l_x_` are what torch 2.2.2 prints; current torch names lifted parameters after their module path instead, so an exported graph there begins with something like `p_fc_weight`. The structure is the same and the names are not, so re-run this on your own build before you quote a name.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): shapes recovered from node.meta, no execution',
              lang: 'python',
              text: 'import torch\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 4)\n    def forward(self, x):\n        return torch.relu(self.fc(x)).sum()\n\ntorch.manual_seed(0)\nep = torch.export.export(Net(), (torch.randn(2, 3),))\nfor n in ep.graph.nodes:\n    v = n.meta.get("val")\n    shape = tuple(v.shape) if isinstance(v, torch.Tensor) else v\n    print(f"{n.op:13s} {n.name:9s} {str(n.target)[:22]:22s} {shape}")\n\n# placeholder   arg0_1    arg0_1                 (4, 3)\n# placeholder   arg1_1    arg1_1                 (4,)\n# placeholder   l_x_      l_x_                   (2, 3)\n# call_function t         aten.t.default         (3, 4)\n# call_function addmm     aten.addmm.default     (2, 4)\n# call_function relu      aten.relu.default      (2, 4)\n# call_function sum_1     aten.sum.default       ()\n# output        output    output                 None',
            },
          },
          {
            h: 'what disappears as you descend',
            ps: [
              "Put the printouts from this lesson and the last one side by side and a pattern falls out that is worth carrying as a reading rule. The deeper the graph, the fewer opcodes it uses. The symbolic trace above holds all six. The dynamo capture in lesson one used `call_module` and `call_method` with no `get_attr` at all, because its parameters were still inside an unopened Linear. The aot and exported graphs have none of those three: every computing node is a `call_function` on an `OpOverload`.",
              "That narrowing is what makes the bottom of the stack compilable. A backend that has to understand `call_module` has to understand arbitrary Python classes. A backend that only sees `call_function` on resolved ATen overloads has a finite list to implement, and the size of that list is the whole subject of the next lesson.",
              "The narrowing also tells you where you are when someone hands you a graph with no context. Count the opcodes present. If you see a module call, you are above aot_autograd; if every line is an overload, you are below it.",
            ],
          },
        ],
        readings: [
          { label: 'torch.fx reference', url: 'https://docs.pytorch.org/docs/stable/fx.html', note: 'the six opcodes, the Node and Graph API, and the symbolic tracer used above' },
          { label: 'fx/node.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/fx/node.py', note: 'where op, target, args, users and meta are defined, and what the printer does with them' },
          { label: 'graph transformations', url: 'https://docs.pytorch.org/docs/stable/torch.compiler_transformations.html', note: 'writing a pass over these nodes, which is the other reason to know the grammar' },
        ],
        check: [
          {
            q: 'A graph line reads call_module[target=L__self___fc]. What does that tell you about where in the stack the graph came from?',
            a: 'That it is above aot_autograd. Module calls survive dynamo capture but not the aot re-trace, so an aot or exported graph has only call_function nodes on resolved ATen overloads.',
          },
          {
            q: 'What does the .default at the end of torch.ops.aten.sum.default select?',
            a: 'One overload out of the packet named sum. On torch 2.2.2 that packet lists ten overloads; default is the reduce-everything signature, while dim_IntList is the one that takes a dimension list.',
          },
          {
            q: 'How would you get the output shape of the third node in an exported graph without running anything?',
            a: 'Read node.meta["val"], which holds a fake tensor with the real shape, dtype and device but no storage. It is filled in during tracing, so the whole graph is a shape table already.',
          },
        ],
        work: [
          { id: 'opcode-census', label: 'take three graphs from different depths of your own model and write the opcode census of each before printing it', href: '#six-kinds-of-node-one-six-line-graph' },
          { id: 'shape-table', label: 'build the shape table for one exported module from node.meta alone, then check two entries by hand', href: '#the-shapes-are-in-the-metadata' },
        ],
      },
      {
        id: 'aten-core-aten-prims',
        num: 3,
        title: 'ATen, core ATen, prims',
        lede: 'One call to gelu is one node, or one node, or five, depending on which vocabulary you ask it to speak. The rewriting between those vocabularies is what decompositions are, and you can watch it happen on one line of code.',
        goal: 'Given an op, say which of the three vocabularies it belongs to, predict whether the core decomposition table will rewrite it and into what, and read a decomposed graph well enough to say why the rewrite left the node count higher but the signature count lower.',
        sections: [
          {
            h: 'one function, three vocabularies deep',
            ps: [
              "`make_fx` traces a function into an FX graph and takes a decomposition table as an argument, which makes it the cleanest instrument for this. Trace `torch.nn.functional.gelu` three times, changing only the table, and you get three graphs of the same function at three depths.",
              "Plain, it is one node: `aten.gelu.default`. With the core table, it is still one node, because gelu is already in the core set and the table has no rule that touches it. Under `TorchRefsMode`, which routes every torch call through the reference implementations in `torch._refs`, it becomes five `prims` nodes and the arithmetic is finally visible: multiply by a half, multiply by one over the square root of two, take the error function, add one, multiply the two halves together.",
              "The constant `0.7071067811865476` in that printout is the whole point of the bottom rung. At the prims level nothing is named after a neural network idea any more. There are only elementwise operations on shaped buffers, which is a small enough vocabulary that a new hardware backend can implement all of it.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same gelu at three depths',
              lang: 'python',
              text: 'import torch\nfrom torch.fx.experimental.proxy_tensor import make_fx\nfrom torch._decomp import core_aten_decompositions\nfrom torch._prims.context import TorchRefsMode\n\ndef f(x):\n    return torch.nn.functional.gelu(x)\n\nx = torch.randn(3)\nprint("aten:")\nprint(make_fx(f)(x).code.strip())\nprint("core aten:")\nprint(make_fx(f, decomposition_table=core_aten_decompositions())(x).code.strip())\nprint("prims:")\nwith TorchRefsMode():\n    print(make_fx(f)(x).code.strip())\nprint("core_aten_decompositions entries:", len(core_aten_decompositions()))',
            },
          },
          {
            h: 'counting the three surfaces',
            ps: [
              "The output of that script, verbatim, is below. Read the last line first: the core decomposition table on this build holds 373 entries, which is the number of rewrite rules, not the size of the core set itself.",
              "Set that against the numbers from the last lesson. There are 817 ATen packets carrying 2124 overloads, and 132 prims packets. A backend author facing 2124 signatures has an impossible job; facing 132 has a finite one, at the cost of writing every fused kernel out of pieces. Core ATen sits between the two, small enough to implement and still coarse enough to keep the fusion opportunities the prims level has already thrown away.",
              "Which rung you want depends on what you are building. A compiler that fuses aggressively wants ops small enough to rearrange. A vendor library with a hand-written attention kernel wants the op left whole so it can pattern-match it. Both of those preferences are expressed by choosing a decomposition table, not by changing the model.",
            ],
            code: {
              caption: 'the verbatim output of the script above (verified, torch 2.2.2 CPU)',
              lang: 'text',
              text: 'aten:\ndef forward(self, x_1):\n    gelu = torch.ops.aten.gelu.default(x_1);  x_1 = None\n    return gelu\ncore aten:\ndef forward(self, x_1):\n    gelu = torch.ops.aten.gelu.default(x_1);  x_1 = None\n    return gelu\nprims:\ndef forward(self, x_1):\n    mul = torch.ops.prims.mul.default(x_1, 0.5)\n    mul_1 = torch.ops.prims.mul.default(x_1, 0.7071067811865476);  x_1 = None\n    erf = torch.ops.prims.erf.default(mul_1);  mul_1 = None\n    add = torch.ops.prims.add.default(erf, 1.0);  erf = None\n    mul_2 = torch.ops.prims.mul.default(mul, add);  mul = add = None\n    return mul_2\ncore_aten_decompositions entries: 373',
            },
            table: {
              caption: 'the three surfaces, counted on this build (verified, torch 2.2.2 CPU)',
              cols: ['vocabulary', 'measured size', 'what it is for'],
              rows: [
                ['torch.ops.aten', '817 packets, 2124 overloads', 'everything eager PyTorch can dispatch, including every convenience signature'],
                ['core ATen', '373 rewrite rules in core_aten_decompositions()', 'the target a backend is expected to cover; the rules map the rest onto it'],
                ['torch.ops.prims', '132 packets', 'the reference layer, elementwise and shape primitives with no fusion left in them'],
              ],
            },
          },
          {
            h: 'what the core set actually buys',
            ps: [
              "Four ops through the same comparison show the shape of the boundary better than any list. `silu` is not core, so the table rewrites it into `sigmoid` plus a `mul`. `hardswish` is not core either, and comes out as add, two clamps, a mul and a div. `_softmax` and `native_layer_norm` both survive untouched, because both are in the core set and a backend is expected to implement them.",
              "The fourth case is the interesting one, and it is not about neural network ops at all. `nn.functional.linear` traces to `aten.t.default` followed by `addmm`. Under the core table, the `addmm` is unchanged and the `t` becomes `aten.permute.default` with an explicit `[1, 0]`. Same node count, same work, one fewer signature to implement.",
              "That is the rule the core set is built on. Decomposition is not simplification and it is not always a rewrite into more nodes. It is a reduction in the number of distinct signatures a backend has to understand, and sometimes the cheapest way to reach it is to replace a convenience op with the general one it is a special case of.",
            ],
            table: {
              caption: 'the same trace with and without core_aten_decompositions (verified, torch 2.2.2 CPU)',
              cols: ['written as', 'traced plainly', 'under the core table'],
              rows: [
                ['F.silu(x)', 'aten.silu.default', 'aten.sigmoid.default, aten.mul.Tensor'],
                ['F.hardswish(x)', 'aten.hardswish.default', 'add.Tensor, clamp, clamp, mul.Tensor, div.Tensor'],
                ['torch.softmax(x, -1)', 'aten._softmax.default', 'unchanged, already core'],
                ['F.layer_norm(x, (3,))', 'aten.native_layer_norm.default', 'unchanged, already core'],
                ['F.linear(x, w, b)', 'aten.t.default, aten.addmm.default', 'aten.permute.default, aten.addmm.default'],
              ],
            },
          },
          {
            h: 'decomposing an exported program',
            ps: [
              "You do not have to build the trace by hand to get this. An `ExportedProgram` carries `run_decompositions()`, which returns a new program with the table applied, and running it on a linear layer followed by a silu shows both rewrites from the table above happening at once on real graph.",
              "Three things change in that printout and only two of them are decompositions. `t` becomes `permute` with an explicit permutation. `silu` becomes `sigmoid` and `mul`, and the `addmm` node picks up a second user because both of them consume it. The third change is cosmetic: the user input, printed as `%l_x_` before, comes back as `%arg2_1`, because the pass renumbers every placeholder into one sequence.",
              "Chapter 10 says an exported program reaches the bridges decomposed toward a core set of ATen ops. This is the call that does it, and the reason the bridge cares is the same reason a backend author cares: on the other side of the crossing there is a compiler whose op list is finite and does not include `silu`.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): an exported program before and after run_decompositions()',
              lang: 'python',
              text: 'import torch\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 4)\n    def forward(self, x):\n        return torch.nn.functional.silu(self.fc(x))\n\ntorch.manual_seed(0)\nep = torch.export.export(Net(), (torch.randn(2, 3),))\nprint(ep.graph)\nprint(ep.run_decompositions().graph)',
              full: {
                label: 'the script and its verbatim output',
                text: 'import torch\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 4)\n    def forward(self, x):\n        return torch.nn.functional.silu(self.fc(x))\n\ntorch.manual_seed(0)\nep = torch.export.export(Net(), (torch.randn(2, 3),))\nprint(ep.graph)\nprint(ep.run_decompositions().graph)\n\n# graph():\n#     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]\n#     %arg1_1 : [num_users=1] = placeholder[target=arg1_1]\n#     %l_x_ : [num_users=1] = placeholder[target=l_x_]\n#     %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%arg0_1,), kwargs = {})\n#     %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg1_1, %l_x_, %t), kwargs = {})\n#     %silu : [num_users=1] = call_function[target=torch.ops.aten.silu.default](args = (%addmm,), kwargs = {})\n#     return (silu,)\n# graph():\n#     %arg0_1 : [num_users=1] = placeholder[target=arg0_1]\n#     %arg1_1 : [num_users=1] = placeholder[target=arg1_1]\n#     %arg2_1 : [num_users=1] = placeholder[target=arg2_1]\n#     %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0]), kwargs = {})\n#     %addmm : [num_users=2] = call_function[target=torch.ops.aten.addmm.default](args = (%arg1_1, %arg2_1, %permute), kwargs = {})\n#     %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm,), kwargs = {})\n#     %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, %sigmoid), kwargs = {})\n#     return (mul,)',
              },
            },
          },
          {
            h: 'where this lands for a backend author',
            ps: [
              "The jax path solves the same problem with the same move and different names, which is worth noticing because it tells you the problem is structural rather than a PyTorch quirk. There, a client-level op either has an HLO counterpart and survives, or it falls through a table of decomposition patterns into spec ops. The lesson on how a jaxpr becomes HLO works one such table line by line.",
              "The difference is where the table lives. PyTorch's decompositions are written in Python, in `torch._decomp` and `torch._refs`, and you can swap the table per compilation. The XLA equivalent is compiled into the compiler. That is why a torch backend can be selective about which ops it wants left whole, and why the same model can reach two backends with two different op sets from one export.",
              "A practical consequence for reading profiles: when a fused kernel you expected does not appear, check whether the op survived decomposition before you go looking at the codegen. An op that got rewritten into five pieces upstream never reached the fusion pass as one thing.",
            ],
          },
        ],
        readings: [
          { label: 'IR reference: core ATen and prims', url: 'https://docs.pytorch.org/docs/stable/torch.compiler_ir.html', note: 'the two op sets listed out, which is the authoritative membership list behind the counts above' },
          { label: 'torch/_decomp at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/_decomp/__init__.py', note: 'core_aten_decompositions and the registry the tables are built from' },
          { label: 'torch/_refs at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/_refs/__init__.py', note: 'the reference implementations TorchRefsMode routes through on the way to prims' },
        ],
        check: [
          {
            q: 'Why did core_aten_decompositions() leave gelu and softmax alone but rewrite silu?',
            a: 'Because gelu and _softmax are in the core ATen set and silu is not. The table only carries rules for ops outside the set, so membership is what decides whether a node survives.',
          },
          {
            q: 'The core table turned aten.t.default into aten.permute.default and the node count did not change. What was the gain?',
            a: 'One fewer signature for a backend to implement. Decomposition reduces the number of distinct ops a backend must cover, which sometimes means replacing a convenience op with the general one rather than expanding it.',
          },
          {
            q: 'You want a vendor kernel to match a whole attention op, but your graph arrives already broken into elementwise pieces. Where do you intervene?',
            a: 'At the decomposition table, before the backend sees the graph. Which ops survive is chosen by the table passed to tracing or to run_decompositions, not by anything in the model.',
          },
        ],
        work: [
          { id: 'guess-the-rung', label: 'pick five ops your model uses and predict for each whether the core table rewrites it, then check with make_fx', href: '#what-the-core-set-actually-buys' },
          { id: 'decompose-your-export', label: 'run run_decompositions() on one of your own exported modules and write down every node that changed and why', href: '#decomposing-an-exported-program' },
        ],
      },
      {
        id: 'what-export-refuses',
        num: 4,
        title: 'What export refuses',
        lede: 'Export is called the strict one, and it is, but not in the direction most people guess. It refuses Python it cannot prove away, and it captures a data-dependent value that costs dynamo a graph break.',
        goal: 'Given a module, predict whether torch.export accepts it in strict mode, in non-strict mode, or in neither, name which tracer produced the error you got, and say what happened to any Python side effect the module performed.',
        sections: [
          {
            h: 'two tracers, two error messages',
            ps: [
              "Chapter 7 frames export as the thing that refuses the compromise dynamo makes. The frame is right and it hides a fork: there are two ways to export, and they fail differently because they are built out of different machinery.",
              "Strict mode, the default in torch 2.2.2, traces with dynamo. It analyses bytecode, so it knows which Python line offended and says so. Non-strict mode runs your forward on the actual Python interpreter with fake tensors flowing through it, so anything that is not a tensor operation simply executes and leaves no trace in the graph.",
              "Put a `print` inside a forward and both behaviours show up in one run. Strict refuses with `Unsupported: call_function BuiltinVariable(print)`, and it names the two arguments it saw, a string constant and a tuple, because it read the call out of the bytecode rather than executing it. Non-strict accepts, and the graph that comes back has four nodes with no print among them.",
              "The line `tracing (3,)` lands on your terminal twice, so the forward ran twice inside the one export call. The shape in it is not a guess. `tuple(x.shape)` was evaluated on the fake tensor the interpreter was carrying, and a fake tensor holds an exact shape, dtype and device with no data behind them. The shape was there to read. The values were not, which is what stops the second module.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same two modules, both modes',
              lang: 'python',
              text: 'import torch\n\nclass Logged(torch.nn.Module):\n    def forward(self, x):\n        w = x.tanh()\n        print("tracing", tuple(x.shape))\n        return w * 3\n\nclass Guarded(torch.nn.Module):\n    def forward(self, x):\n        if (x < 0).any():\n            return x.abs()\n        return x.tanh()\n\nfor name, mod in [("logged", Logged()), ("guarded", Guarded())]:\n    for strict in (True, False):\n        try:\n            ep = torch.export.export(mod, (torch.ones(3),), strict=strict)\n            print(f"{name} strict={strict}: accepted, {len(ep.graph.nodes)} nodes")\n        except Exception as e:\n            print(f"{name} strict={strict}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")',
            },
          },
          {
            h: 'the branch neither mode will take',
            ps: [
              "The output below is verbatim, `tracing` lines included, in the order they were printed. Read the two `guarded` lines together. Both modes refuse the data-dependent `if`, and the wording of the refusal tells you which tracer produced it.",
              "Strict mode raises a `UserError` naming the feature and pointing at `functorch.experimental.control_flow.cond`, the operator that would let the branch live inside the graph as a node with two subgraphs. Non-strict raises `DataDependentOutputException` naming `aten._local_scalar_dense.default`, which is the op that pulls a Python number out of a tensor. `(x < 0).any()` produces a one-element bool tensor and the `if` needs a real bool out of it, which means reading the data a fake tensor does not have.",
              "Same refusal, two different vantage points on it, and the practical value is diagnostic. A `UserError` quoting your source line means dynamo tripped. A fake-tensor exception naming an ATen op means you were in non-strict mode and the interpreter got as far as executing before the shape machinery stopped it.",
            ],
            code: {
              caption: 'the verbatim output of the script above, stdout in order (verified, torch 2.2.2 CPU)',
              lang: 'text',
              text: 'logged strict=True: Unsupported: call_function BuiltinVariable(print) [ConstantVariable(str), TupleVariable()] {}\ntracing (3,)\ntracing (3,)\nlogged strict=False: accepted, 4 nodes\nguarded strict=True: UserError: Dynamic control flow is not supported at the moment. Please use functorch.experimental.control_flow.cond to ex\nguarded strict=False: DataDependentOutputException: aten._local_scalar_dense.default',
            },
            table: {
              caption: 'four attempts, four outcomes, from the run above',
              cols: ['module', 'strict=True', 'strict=False'],
              rows: [
                ['print inside forward', 'Unsupported, from dynamo, with both print arguments named', 'accepted; the print runs at export time and vanishes from the graph'],
                ['if (x < 0).any()', 'UserError naming dynamic control flow', 'DataDependentOutputException on aten._local_scalar_dense.default'],
              ],
            },
          },
          {
            h: 'the one dynamo breaks on and export keeps',
            ps: [
              "Now the case that goes the other way, and it is the one worth remembering because it inverts the usual story. The guard-or-break drill on the gym's pytorch floor has a `.item()` scenario among its eight, and its measured verdict is a graph break. `.item()` has to hand back a real Python number, and at capture time there is no such number, so capture stops at that line and Python runs it.",
              "Export takes the same construct. Subtract a mean pulled out with `.item()` and the graph that comes back is one placeholder, three ATen calls and a return. The middle of the three is `aten._local_scalar_dense.default`, the scalar extraction written down as an operation instead of appearing as a gap in the capture. Nothing broke because there was nothing to break: export has no fallback path to Python, so it either represents a thing in the graph or refuses.",
              "The line between the two cases is whether the value gets used to decide control flow. A scalar pulled out and fed back into arithmetic is a node. The same scalar compared in an `if` is a branch, and a branch needs both sides in the graph, which is what `cond` is for and what the error message asks you to write.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the scalar extraction, captured as a node',
              lang: 'python',
              text: 'import torch\n\nclass Center(torch.nn.Module):\n    def forward(self, x):\n        return x - x.mean().item()\n\nprint(torch.export.export(Center(), (torch.ones(4),)).graph)\n\n# graph():\n#     %l_x_ : [num_users=2] = placeholder[target=l_x_]\n#     %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%l_x_,), kwargs = {})\n#     %_local_scalar_dense : [num_users=1] = call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%mean,), kwargs = {})\n#     %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%l_x_, %_local_scalar_dense), kwargs = {})\n#     return (sub,)',
            },
          },
          {
            h: 'a side effect that fires once',
            ps: [
              "Chapter 6 proves that a Python side effect inside a compiled function keeps firing on every call, and contrasts that with jax's trace-once model where it fires exactly once. Export lands on the jax side of that line, and it is worth measuring rather than assuming.",
              "Append the input's shape to a module-level list, export, and the list holds one entry, `(3,)`: the trace ran the forward once, on the example input you handed it. Empty the list, replay the exported program three times, and it stays empty. The graph is a placeholder, one `tanh`, and the return, with no record that a Python list was ever touched.",
              "That is the contract you are buying. An `ExportedProgram` is a description of tensor computation and nothing else, which is what lets it run in a process with no Python interpreter. Anything your forward did that was not tensor math happened once, during export, in your process, and is gone.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the list before and after',
              lang: 'python',
              text: 'import torch\n\nseen = []\n\nclass Recorder(torch.nn.Module):\n    def forward(self, x):\n        seen.append(tuple(x.shape))\n        return x.tanh()\n\nep = torch.export.export(Recorder(), (torch.ones(3),))\nprint("seen after export:", seen)\nseen.clear()\nfor _ in range(3):\n    ep.module()(torch.ones(3))\nprint("seen after three replays:", seen)\nprint(ep.graph)\n\n# seen after export: [(3,)]\n# seen after three replays: []\n# graph():\n#     %l_x_ : [num_users=1] = placeholder[target=l_x_]\n#     %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%l_x_,), kwargs = {})\n#     return (tanh,)',
            },
          },
          {
            h: 'shapes you promise in advance',
            ps: [
              "One graph for the whole program means one graph for every input shape you intend to serve, so export makes you say which dimensions vary. The module below is chapter 7's `Sin`, the same two ops it exports there, taken back to the export call with one argument added. A `Dim` with a name and a range turns a concrete size into a symbol, and the placeholder's shape comes back as `(s0,)` instead of `(6,)`.",
              "The range is not decoration. `ep.range_constraints` records it, and the exported module enforces both ends at replay: a size of 3 or of 9 against a `Dim(\"b\", min=4, max=8)` raises `Input l_x_.shape[0] is outside of specified dynamic range [4, 8]`, while 4 and 8 run. Compare that to the dynamo layer, where an unexpected shape is a guard miss and a silent recompile.",
              "The two failure modes are the same disagreement chapter 6 draws between guarded capture and tracing once, met a second time at the artifact boundary. Under compile, a shape you did not anticipate costs you a compile. Under export, it costs you an exception, which is what you want from something running in a serving stack with no compiler attached.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a symbolic dimension, and both ends of its range',
              lang: 'python',
              text: 'import torch\nfrom torch.export import Dim\n\nclass Sin(torch.nn.Module):\n    def forward(self, x):\n        return torch.sin(x).sum()\n\nb = Dim("b", min=4, max=8)\nep = torch.export.export(Sin(), (torch.randn(6),), dynamic_shapes={"x": {0: b}})\nprint("range_constraints:", ep.range_constraints)\nprint("placeholder shape:", tuple(list(ep.graph.nodes)[0].meta["val"].shape))\nfor n in (3, 4, 8, 9):\n    try:\n        print(n, "->", round(ep.module()(torch.ones(n)).item(), 6))\n    except Exception as e:\n        print(n, "->", type(e).__name__ + ":", str(e).splitlines()[0][:70])\n\n# range_constraints: {s0: ValueRanges(lower=4, upper=8, is_bool=False)}\n# placeholder shape: (s0,)\n# 3 -> RuntimeError: Input l_x_.shape[0] is outside of specified dynamic range [4, 8]\n# 4 -> 3.365884\n# 8 -> 6.731767\n# 9 -> RuntimeError: Input l_x_.shape[0] is outside of specified dynamic range [4, 8]',
            },
          },
          {
            h: 'what the artifact carries',
            ps: [
              "The graph is the part everyone looks at, and it is not the part that makes an `ExportedProgram` portable. Parameters do not live inside the graph as attributes; they are lifted to placeholders, and a separate `graph_signature` records which placeholder was which parameter. That is why the printed graph of a two-parameter linear layer starts with three inputs when the module takes one.",
              "Alongside the signature there is a `state_dict` holding the actual weights under their original names, so nothing about the mapping is lost. `torch.export.save` writes the whole thing to a stream, and that stream is a zip archive with four entries: the serialized program, the state dict, any constants, and a version file. Reload it in another process and the module replays with no reference to the class that produced it.",
              "This is the object chapter 10 hands to the bridges. What each bridge does with it after that, and the StableHLO it becomes on the other side, is that chapter's material and the torch_xla lessons underneath it. From here, the thing worth carrying is what the artifact is: a graph, a signature that maps its inputs back to names, and the weights.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the signature, the weights, and a save-load round trip',
              lang: 'python',
              text: 'import io\nimport zipfile\nimport torch\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 4)\n    def forward(self, x):\n        return torch.relu(self.fc(x)).sum()\n\ntorch.manual_seed(0)\nep = torch.export.export(Net(), (torch.randn(2, 3),))\nfor spec in ep.graph_signature.input_specs:\n    print(spec.kind, spec.arg.name, "->", spec.target)\nprint("state_dict keys:", list(ep.state_dict))\nbuf = io.BytesIO()\ntorch.export.save(ep, buf)\nbuf.seek(0)\nprint("archive entries:", zipfile.ZipFile(buf).namelist())\nbuf.seek(0)\nprint("reloaded, replayed:", round(torch.export.load(buf).module()(torch.ones(2, 3)).item(), 6))\n\n# InputKind.PARAMETER arg0_1 -> fc.weight\n# InputKind.PARAMETER arg1_1 -> fc.bias\n# InputKind.USER_INPUT l_x_ -> None\n# state_dict keys: [\'fc.weight\', \'fc.bias\']\n# archive entries: [\'serialized_exported_program.json\', \'serialized_state_dict.json\', \'serialized_constants.json\', \'version\']\n# reloaded, replayed: 0.314205',
            },
          },
        ],
        readings: [
          { label: 'torch.export reference', url: 'https://docs.pytorch.org/docs/stable/export.html', note: 'the ExportedProgram contract, Dim, and the strict flag; read it against the version you actually run' },
          { label: 'what strict=False means', url: 'https://dev-discuss.pytorch.org/t/meaning-of-strict-false-in-torch-export-export/1952', note: 'the maintainers arguing out what non-strict tracing does and does not guarantee' },
          { label: 'export tutorial', url: 'https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html', note: 'the current-version walkthrough, useful mainly for spotting what has changed since 2.2.2' },
        ],
        check: [
          {
            q: 'An export failed with DataDependentOutputException naming an ATen op. Which mode were you in, and how do you know?',
            a: 'Non-strict. Strict mode traces with dynamo and reports a UserError naming the Python feature and the source line; a fake-tensor exception naming an ATen op comes from the interpreter path.',
          },
          {
            q: 'Dynamo graph breaks on a scalar pulled out with .item(), but export captures the same line. Why is that not a contradiction?',
            a: 'Because export has no Python fallback, so it must represent the scalar extraction as a node, aten._local_scalar_dense.default. Dynamo can afford to stop capturing and run the line in Python instead.',
          },
          {
            q: 'What happens to a Python list your forward appends to, once the module is exported?',
            a: 'It gets exactly one entry, appended while the export trace ran the forward, and never another. The replayed program contains only tensor operations, which is what lets it run without a Python interpreter.',
          },
        ],
        work: [
          { id: 'both-modes', label: 'export one of your own modules in both strict and non-strict mode and write down every difference in the two graphs', href: '#two-tracers-two-error-messages' },
          { id: 'name-the-dims', label: 'give one model an explicit Dim for its batch dimension and prove the range is enforced at both ends', href: '#shapes-you-promise-in-advance' },
          { id: 'audit-side-effects', label: 'find one non-tensor thing your forward does and say what an exported copy of it would silently stop doing', href: '#a-side-effect-that-fires-once' },
        ],
      },
    ],
  },
]
