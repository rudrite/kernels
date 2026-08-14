// New file: site/src/data/lessons/torch-tape.ts
// The autograd tape, below the survey chapter 2 teaches: the Node object behind
// grad_fn and the edges it holds, who writes .grad and in what order, what each
// node keeps alive and how it notices a rewrite, and a hand-written backward
// checked against a numerical Jacobian. Every printed value was produced on this
// machine with torch 2.2.2 on CPU; the two C++ excerpts are verbatim from
// pytorch/pytorch at tag v2.2.2, the version the runs used.
import type { UnitLessons } from './index'

export const TORCH_TAPE_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:autograd',
    lessons: [
      {
        id: 'the-node-behind-grad-fn',
        num: 1,
        title: 'The node behind grad_fn',
        lede: 'Printing y.grad_fn gives you a name and an address, which is enough to believe the tape exists and not enough to read it. The object behind that repr holds a list of edges, and each edge is a pair.',
        goal: 'Given a short program, write out the node list autograd built with the edges between them, say what the second element of every next_functions pair is for, and predict the order the engine will run those nodes in from the numbers the tape assigned during the forward pass.',
        sections: [
          {
            h: 'grad_fn is an object with a type',
            ps: [
              "The chapter above this lesson says `y.grad_fn` is the tape, which is true and stops one step short of useful. `type(y.grad_fn)` comes back as `SumBackward0`, a class generated at build time from the derivative formula table, and the object is an instance of the C++ `torch::autograd::Node` base exposed to Python. It answers `name()`, it carries a `next_functions` tuple, and for most ops it carries the tensors it saved under `_saved_` attributes that lesson three takes apart.",
              "Two names for the same node disagree on purpose. The repr prints the short class name, `AccumulateGrad`, while `name()` returns the fully qualified `torch::autograd::AccumulateGrad`, because the leaf accumulator is a hand-written C++ node rather than a generated one. When you are matching a profiler trace against a graph walk, the long name is the one the profiler emitted.",
              "Run the walk on the simplest expression that branches and something shows up that a drawing would get wrong. `x * x` has two operands, so `MulBackward0` holds two edges, and both edges point at the same `AccumulateGrad` object. Not two accumulators for one tensor. One accumulator, reached twice, whose `.variable` attribute is `x` itself.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one accumulator, reached down two edges',
              lang: 'python',
              text: 'import torch\n\nx = torch.ones(3, requires_grad=True)\ny = (x * x).sum()\nmul = y.grad_fn.next_functions[0][0]\nprint(y.grad_fn.name())              # SumBackward0\nprint(len(mul.next_functions))       # 2\na, b = (n for n, _ in mul.next_functions)\nprint(a.name())                      # torch::autograd::AccumulateGrad\nprint(a is b, a.variable is x)       # True True',
            },
          },
          {
            h: 'an edge is a node and a slot',
            ps: [
              "`next_functions` is a tuple of pairs, and both halves matter. The first half is the node this edge leads to, or `None` when the operand on that side has no gradient to receive. The second half is an integer nobody looks at until they need it, and the table below is the cheapest way to learn what fills each position.",
              "Read the `None` rows first. Multiplying by a constant produces the same `MulBackward0` in both spellings, but the edge for the constant side is `(None, 0)`, so the engine has somewhere to put the operand's slot without having anywhere to send a gradient. That is how autograd represents a dead branch: not by omitting the edge, but by keeping the position and nulling the destination.",
              "The chapter's tape exhibit walks four of these chains node by node, and two of its four entries are the pair worth holding side by side. The one-leaf multiply and the two-leaf matmul both print a node with two `AccumulateGrad` children, and they are not the same shape underneath. In the multiply, both children are one object; in the matmul, they are two objects with different `.variable` tensors.",
            ],
            table: {
              caption: 'next_functions for eight expressions, read off the live nodes (verified, torch 2.2.2 CPU); x requires grad, c does not',
              cols: ['expression', 'grad_fn', 'next_functions'],
              rows: [
                ['x * x', 'MulBackward0', '(AccumulateGrad, 0) (AccumulateGrad, 0)'],
                ['x * c', 'MulBackward0', '(AccumulateGrad, 0) (None, 0)'],
                ['c * x', 'MulBackward0', '(None, 0) (AccumulateGrad, 0)'],
                ['x + c', 'AddBackward0', '(AccumulateGrad, 0) (None, 0)'],
                ['x.exp()', 'ExpBackward0', '(AccumulateGrad, 0)'],
                ['(x * x).sum()', 'SumBackward0', '(MulBackward0, 0)'],
                ['x.split(1)[1]', 'SplitBackward0', '(AccumulateGrad, 0)'],
                ['x.view(3, 1)', 'ViewBackward0', '(AccumulateGrad, 0)'],
              ],
            },
          },
          {
            h: 'the slot number earns its keep on multi-output ops',
            ps: [
              "Every row in that table has a zero in the slot position, which is what happens when every op you try produces one output. Split a tensor and the zero stops being automatic. `x.split(3)` returns two tensors that share one `SplitBackward0` node, and they are told apart by `output_nr`: the first is output zero, the second is output one.",
              "Now the pair makes sense. An edge says which node to call and which of that node's outputs this gradient belongs to, because a node with two outputs receives two gradients and has to know which is which. Take the gradient of something built from the second half of the split and the edge reads `('SplitBackward0', 1)`.",
              "One tensor attribute carries the same number on the forward side. `hi.output_nr` is 1 before any backward pass runs, so you can read a tensor's position in its producing node without touching the graph at all. It is also the number that appears in the version-counter error the museum wing keeps, in the phrase naming which output of which node was modified.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one node, two outputs, and the slot that tells them apart',
              lang: 'python',
              text: "import torch\n\nx = torch.arange(6.0, requires_grad=True)\nlo, hi = x.split(3)\nprint(lo.grad_fn is hi.grad_fn)      # True: one node, two outputs\nprint(lo.output_nr, hi.output_nr)    # 0 1\nedge = (hi * 2).sum().grad_fn.next_functions[0][0].next_functions\nprint([(n.name() if n else None, k) for n, k in edge])\n# [('SplitBackward0', 1), (None, 0)]",
            },
          },
          {
            h: 'the tape numbers itself as it is built',
            ps: [
              "Each node gets an integer when it is constructed, from a counter that starts at zero in every thread and only goes up. Four ops in forward order come out zero, one, two, three, which reads like bookkeeping until you notice which node breaks the pattern.",
              "`AccumulateGrad` is not on the counter. Its number is the largest a 64-bit unsigned integer holds, 18446744073709551615, set by hand rather than drawn. A leaf accumulator is created whenever a leaf is first used, so on the counter it would land wherever that happened to be, and its position on the counter is exactly what the engine sorts by.",
              "You can read the number off any node with `_sequence_nr()`. The underscore is honest about the stability promise, and the value is worth reading anyway, because it is the same integer the profiler stamps on a backward event when it correlates that event with the forward op that produced it.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): four ops in forward order, then the accumulator that sits outside the counter',
              lang: 'python',
              text: 'import torch\n\nx = torch.ones(3, requires_grad=True)\na, b = x * 2, x * 3\nc = a + b\ny = c.sum()\nfor t in (a, b, c, y):\n    print(t.grad_fn.name(), t.grad_fn._sequence_nr())\n# MulBackward0 0 / MulBackward0 1 / AddBackward0 2 / SumBackward0 3\nacc = c.grad_fn.next_functions[0][0].next_functions[0][0]\nprint(acc.name(), acc._sequence_nr())\n# torch::autograd::AccumulateGrad 18446744073709551615',
            },
          },
          {
            h: 'and the engine reads those numbers backwards',
            ps: [
              "The comment above `sequence_nr()` in the C++ header states both jobs the number does, and the first one turns the measurement above into a scheduling rule. Higher runs first, so the op that ran last in the forward pass is the first one the backward pass reaches. Reverse order falls out of a priority queue rather than out of a stored ordering.",
              "The caveat in the middle of that comment is the reason `AccumulateGrad` sits at the ceiling. Give the leaf accumulator the maximum priority and a gradient that has arrived at a leaf gets written the moment it can be, instead of waiting behind interior nodes that were created later. The buffer holding that gradient is released earlier as a result.",
              "That is one number doing scheduling and profiling at once, and it is worth knowing that the second job is why it exists at all in a profiled build. The rest of this arc stays on the first job: which node runs, what it reads, and what it writes.",
            ],
            code: {
              caption: 'verbatim, torch/csrc/autograd/function.h:309-318 at pytorch v2.2.2, the tag this machine has installed',
              lang: 'c',
              text: '  /// NOTE [ Sequence Number]\n  ///\n  /// The sequence_nr has two main usages in autograd:\n  ///\n  /// 1) Helps determine the node\'s execution priority in the engine.\n  ///    All else being equal, nodes with higher priority numbers are executed\n  ///    first. Thus, nodes corresponding to ops executed later are the first to\n  ///    be executed in the backward pass. One caveat is that we prioritize\n  ///    AccumulateGrad nodes by explicitly setting its sequence_nr to be\n  ///    UINT64_MAX.',
            },
          },
        ],
        readings: [
          { label: 'torch.autograd.graph.Node', url: 'https://docs.pytorch.org/docs/stable/autograd.html', note: 'the Python surface of a node: name, next_functions, metadata, register_hook' },
          { label: 'How computational graphs are constructed in PyTorch', url: 'https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/', note: 'the maintainers walking the same construction from the C++ side, with the generated Backward classes' },
          { label: 'function.h at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/csrc/autograd/function.h', note: 'the Node base class in 600 lines; the Sequence Number note is at 309 and the Topological Number note right under it' },
        ],
        check: [
          {
            q: 'MulBackward0 for x * x holds two edges. How many AccumulateGrad nodes are on the other end?',
            a: 'One. Both edges point at the same object, which you can check with `is`, and its `.variable` attribute is x itself. Two distinct accumulators appear only when two distinct leaves feed the op, as in a matmul of two parameters.',
          },
          {
            q: 'What is the second element of a next_functions pair, and when is it ever not zero?',
            a: 'The output slot of the node the edge leads to. It is zero for every single-output op, and becomes non-zero for ops that produce several outputs from one node, such as split, where the second output is reached by an edge carrying slot 1.',
          },
          {
            q: 'Why does AccumulateGrad carry a sequence number of 18446744073709551615?',
            a: 'Because the engine runs higher numbers first, and setting the leaf accumulator to the 64-bit maximum makes a gradient that has reached a leaf get written as early as possible rather than queuing behind interior nodes. The header comment states it as a deliberate exception to the counter.',
          },
        ],
        work: [
          { id: 'edge-table', label: 'extend the edge table with four expressions of your own and predict each row before you run it', href: '#an-edge-is-a-node-and-a-slot' },
          { id: 'slot-hunt', label: 'find a second op besides split whose node carries more than one output, and read its slot numbers off the edges' },
        ],
      },
      {
        id: 'who-owns-grad',
        num: 2,
        title: 'Who owns .grad',
        lede: 'A gradient reaching a leaf gets written into an attribute, and the timing of that write, the object it lands in, and how many times it happens are three separate questions the word accumulate quietly runs together.',
        goal: 'For any tensor in a program, say whether a backward pass will populate its .grad and why, name how many times the accumulator runs for a tensor used several times in one forward pass, and predict what a reference you kept to .grad holds after zero_grad and one more step.',
        sections: [
          {
            h: 'a leaf is a tensor with no grad_fn',
            ps: [
              "The chapter defines a leaf as a tensor you made rather than one an operation returned. The runtime definition is narrower and easier to check: `is_leaf` is true when `grad_fn` is `None`. Anything an op produced under a recording context carries a node, so it is not a leaf, so no accumulator was ever built for it, so there is nothing to write into.",
              "Reading `.grad` on such a tensor returns `None` and prints a warning first, and the warning is unusually good. It says the attribute will not be populated, names `retain_grad()` as the way to change that, and then guesses out loud that you probably meant to read the leaf instead. Half of PyTorch's autograd errors read like this: a statement of what happened, followed by the two things you might have meant.",
              "The warning fires on the read, not on the backward pass, so it can appear long before or long after the pass that would have filled the attribute. That timing is worth knowing when the warning shows up in a log with nothing around it.",
            ],
            code: {
              caption: 'the warning verbatim on a non-leaf .grad read (verified, torch 2.2.2 CPU); the trailing internal-location parenthesis is trimmed',
              lang: 'text',
              text: 'UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being\naccessed. Its .grad attribute won\'t be populated during autograd.backward(). If\nyou indeed want the .grad field to be populated for a non-leaf Tensor, use\n.retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by\nmistake, make sure you access the leaf Tensor instead. See\ngithub.com/pytorch/pytorch/pull/30531 for more informations.',
            },
          },
          {
            h: 'retain_grad hangs a mailbox on an intermediate',
            ps: [
              "Call `retain_grad()` on a non-leaf and the tensor keeps its node, keeps its place in the graph, and gains somewhere for a gradient to land on the way past. Nothing about the walk changes; one extra hook copies the gradient into `.grad` as it flows through.",
              "The two numbers below are the point of doing it. With `h = x * 2` and a loss of `(h * h).sum()`, the gradient at `h` is 4 and the gradient at `x` is 8, because the chain rule multiplies by the 2 that `h` was built with. Reading only `x.grad` you would have to divide to recover the middle; reading both, the factor is right there.",
              "The cost is a live tensor per retained intermediate, held until you clear it. That makes `retain_grad` a debugging instrument rather than something to leave switched on, which is the same trade the next lesson's saved-tensor accounting makes explicit in bytes.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the gradient at the middle of the chain, and at the leaf',
              lang: 'python',
              text: 'import torch\n\nx = torch.ones(3, requires_grad=True)\nh = x * 2\nh.retain_grad()\n(h * h).sum().backward()\nprint(h.is_leaf, h.grad_fn.name())   # False MulBackward0\nprint(h.grad)                        # tensor([4., 4., 4.])\nprint(x.grad)                        # tensor([8., 8., 8.])',
            },
          },
          {
            h: 'the engine sums before it accumulates',
            ps: [
              "Use one leaf twice in a forward pass and the natural picture is two gradients arriving at the accumulator and being added there, one after the other. That picture is wrong, and it is easy to falsify: hang a hook on the `AccumulateGrad` node and count how many times it runs.",
              "It runs once. For `((x * 2) + (x * 3)).sum()` the accumulator is called a single time and the tensor it receives is already 5, which is the sum of the 2 and the 3 that arrived down the two paths. The addition happened in a buffer the engine keeps per node while it waits for every incoming edge to report, not at the leaf.",
              "So the word accumulate covers two different mechanisms with different scopes. Within one backward pass, contributions are summed in that buffer and delivered once. Across backward passes, the accumulator adds into whatever `.grad` already held, which is the behavior `zero_grad` exists to undo and the one the chapter's training loop is built around.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one call, carrying an already-summed gradient',
              lang: 'python',
              text: 'import torch\n\ngot = []\nx = torch.ones(3, requires_grad=True)\ny = ((x * 2) + (x * 3)).sum()\nadd = y.grad_fn.next_functions[0][0]\nacc = add.next_functions[0][0].next_functions[0][0]\nacc.register_hook(lambda grad_in, grad_out: got.append(grad_out[0].clone()))\ny.backward()\nprint(len(got), got[0])   # 1 tensor([5., 5., 5.])\nprint(x.grad)             # tensor([5., 5., 5.])',
            },
          },
          {
            h: 'grad is one tensor, and zero_grad does not zero it',
            ps: [
              "Across steps, the accumulator adds in place into the tensor `.grad` already points at. Hold a reference to `p.grad` after one backward pass and it reads 4 after the next one, without you touching it, because it is the same object.",
              "Then `opt.zero_grad()` runs and the reference stops tracking. `set_to_none` defaults to `True` here, and has since PyTorch 2.0, so the call assigns `None` to `.grad` rather than filling the existing tensor with zeros, and the tensor you were holding is left behind at its old value. Fewer kernel launches and one less live buffer per parameter, at the price of a name that describes what the method used to do.",
              "Two consequences follow for anyone reading gradients between steps. A logger that captured `p.grad` once and reuses the handle is reading a tensor the optimizer abandoned. And `p.grad` is `None` rather than zero before the first backward pass of every step, so code that assumes a zero tensor is there needs `set_to_none=False` or a `None` check.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same object across two steps, then dropped rather than zeroed',
              lang: 'python',
              text: 'import torch\n\np = torch.ones(3, requires_grad=True)\n(p * p).sum().backward()\nheld = p.grad\n(p * p).sum().backward()\nprint(held, p.grad is held)       # tensor([4., 4., 4.]) True\n\nopt = torch.optim.SGD([p], lr=0.0)\nopt.zero_grad()                   # set_to_none=True is the default\nprint(p.grad, held)               # None tensor([4., 4., 4.])',
            },
          },
          {
            h: 'the call that skips the accumulator entirely',
            ps: [
              "`torch.autograd.grad` walks the same graph with the same engine and never reaches an `AccumulateGrad` node. It takes the outputs, the inputs you want gradients for, and returns a tuple. Nothing is written anywhere; `.grad` stays exactly as it was, `None` included.",
              "That makes it the right call for anything that is not a training step. Second derivatives, influence functions, a gradient norm you want to inspect, a per-sample gradient you are about to reduce yourself: none of those want a side effect on the parameters, and all of them are one line with `grad` and three lines with `backward` plus a save and a restore.",
              "It is also the call lesson four leans on, because `gradcheck` is built on it. A checker that accumulated into `.grad` while checking would corrupt the state of the model it was called on.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same number, returned rather than stored',
              lang: 'python',
              text: 'import torch\n\nz = torch.ones(3, requires_grad=True)\nprint(torch.autograd.grad((z * z).sum(), z))   # (tensor([2., 2., 2.]),)\nprint(z.grad)                                  # None\n\n(z * z).sum().backward()\nprint(z.grad)                                  # tensor([2., 2., 2.])',
            },
          },
        ],
        readings: [
          { label: 'Autograd mechanics: how autograd encodes the history', url: 'https://docs.pytorch.org/docs/stable/notes/autograd.html', note: 'the contract for leaves, requires_grad propagation, and the graph the engine walks' },
          { label: 'How computational graphs are executed in PyTorch', url: 'https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/', note: 'the engine, its ready queue, and the input buffer that sums before a node runs' },
          { label: 'input_buffer.cpp at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/csrc/autograd/input_buffer.cpp', note: 'the 200 lines that do the summing the accumulator gets credit for' },
        ],
        check: [
          {
            q: 'A leaf is used twice in one forward pass. How many times does its AccumulateGrad node run during one backward pass?',
            a: 'Once. The engine holds an input buffer per node and sums every incoming edge there, then calls the node a single time with the total. Measured on ((x * 2) + (x * 3)).sum(), the accumulator ran once and received 5.',
          },
          {
            q: 'You saved a reference to p.grad, then called opt.zero_grad(). What does your reference hold?',
            a: 'The gradient from before the call, unchanged. zero_grad defaults to set_to_none=True, so it assigns None to p.grad rather than writing zeros into the existing tensor, and your reference now points at a tensor the optimizer no longer uses.',
          },
          {
            q: 'When should a gradient be taken with torch.autograd.grad instead of backward?',
            a: 'Whenever the gradient is being inspected rather than applied. grad returns a tuple and writes nothing, so .grad on every parameter is left alone; backward accumulates into .grad and would need a save and a restore around it to be side-effect free.',
          },
        ],
        work: [
          { id: 'accumulator-count', label: 'hang a hook on one AccumulateGrad in a model of your own and check the call count against the number of times that parameter appears in forward', href: '#the-engine-sums-before-it-accumulates' },
          { id: 'none-audit', label: 'find every place your training code reads .grad and decide for each whether it survives set_to_none=True' },
        ],
      },
      {
        id: 'what-the-tape-kept',
        num: 3,
        title: 'What the tape kept',
        lede: 'Between the forward pass and the backward pass, some tensors stay alive because a node is holding them. Which ones, how many bytes, and what happens when you write into one are all readable from Python before backward ever runs.',
        goal: 'List what a given node saved and whether it saved an input or an output, measure the bytes a forward pass holds alive and explain why freezing a layer changes that number, and say exactly what the version counter compares and the one situation in which it compares nothing.',
        sections: [
          {
            h: 'every node advertises what it saved',
            ps: [
              "The derivative of `sin` is `cos`, which needs the input. The derivative of `exp` is `exp`, which is the output already computed. PyTorch encodes that choice per op and then exposes it: any `_saved_` attribute on a node names a tensor that node is keeping alive, and `dir()` lists them.",
              "So `SinBackward0` carries `_saved_self` and `ExpBackward0` carries `_saved_result`, and `ReluBackward0` carries `_saved_result` too, because a relu's gradient only needs to know where the output was positive. `MulBackward0` carries both operands. Nothing here is documentation you have to trust; it is the node telling you.",
              "One detail in the last two lines of the run is the reason lesson one bothered with output slots. `y.grad_fn._saved_result` is not `y`. It is a different Python tensor over the same storage, rebuilt on each access, because a node holding a strong reference to its own output would be a reference cycle that never collects. The rebuild is why the saved value can be checked against the live one at all.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): four nodes naming what they kept',
              lang: 'python',
              text: 'import torch\n\nx = torch.ones(3, requires_grad=True)\nsaved = lambda t: [a for a in dir(t.grad_fn) if a.startswith("_saved")]\nprint(x.sin().grad_fn.name(), saved(x.sin()))       # SinBackward0 [\'_saved_self\']\nprint(x.exp().grad_fn.name(), saved(x.exp()))       # ExpBackward0 [\'_saved_result\']\nprint(torch.relu(x).grad_fn.name(), saved(torch.relu(x)))\n# ReluBackward0 [\'_saved_result\']\nprint((x * x).grad_fn.name(), saved(x * x))\n# MulBackward0 [\'_saved_other\', \'_saved_self\']\n\ny = x.exp()\ns = y.grad_fn._saved_result\nprint(s is y, s.data_ptr() == y.data_ptr())   # False True',
            },
          },
          {
            h: 'what gets saved depends on what you asked for',
            ps: [
              "`saved_tensors_hooks` installs a pack function that fires once per save, so a forward pass under it prints its own memory bill. Run a three-layer MLP with a 256 by 512 input and the bill comes to five distinct storages and 2,623,488 bytes, itemized below.",
              "The interesting entry is the one that is absent. `0.weight` is 512 by 512 and never gets saved, while `2.weight` of exactly the same shape does. The first layer's matmul needs to produce a gradient for its weight, which needs the input; it does not need to produce a gradient for its input, because the input is data. The second layer needs both, so it keeps its weight too.",
              "Turn that around and it is a memory knob with a measured size. Set `requires_grad` on the input and `0.weight` joins the list, taking the total to 3,672,064 bytes. Freeze the first layer instead and the total drops to 1,050,624, because two of the five storages stop being needed at once. Same model, same batch, a factor of 3.5 between the two ends, decided entirely by which gradients the program asked for.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the forward pass printing what it holds alive',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Linear(512, 512), nn.ReLU(),\n                      nn.Linear(512, 512), nn.ReLU(),\n                      nn.Linear(512, 1))\nx = torch.randn(256, 512)\nnames = {p.data_ptr(): n for n, p in model.named_parameters()}\n\nheld = {}\ndef pack(t):\n    held.setdefault(t.data_ptr(), (tuple(t.shape), t.numel() * t.element_size()))\n    return t\n\nwith torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):\n    model(x).sum()\n\nfor ptr, (shape, nbytes) in held.items():\n    print(shape, nbytes, names.get(ptr, "x" if ptr == x.data_ptr() else "an activation"))\nprint("total", sum(b for _, b in held.values()))',
            },
            table: {
              caption: 'saved storages for one forward pass of the same 3-layer MLP on a 256x512 input, three ways (verified, torch 2.2.2 CPU)',
              cols: ['what the program asks for', 'distinct storages held', 'bytes'],
              rows: [
                ['as written: data in, all parameters training', '5', '2,623,488'],
                ['input carries requires_grad too', '6', '3,672,064'],
                ['first Linear frozen, data in', '3', '1,050,624'],
              ],
            },
          },
          {
            h: 'the counter that notices a rewrite',
            ps: [
              "Every tensor carries a version counter, readable as `_version`, that starts at zero and increments on each in-place write. Views share the counter with their base, so writing through a slice bumps the number the base reports, which is how a mutation reaches a node that never saw the view.",
              "In-place writes are not banned, and the run below is the case that works. `y = x * 2` then `y.add_(1)` bumps `y._version` to 1 and also rebases `y.grad_fn` from `MulBackward0` to `AddBackward0`, and the backward pass afterwards is correct, because `MulBackward0` saved `x` rather than `y` and nothing it kept was touched.",
              "Swap `exp` for the multiply and the same two lines fail, because `exp` saved its own output and that output is what got rewritten. The museum wing holds that failure with its verbatim error. The mechanism worth carrying is the one this run makes visible: the counter is always running, and whether a write matters depends only on whether some node saved that exact tensor first.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): an in-place write that rebases the node and keeps the gradient right',
              lang: 'python',
              text: 'import torch\n\nx = torch.ones(3, requires_grad=True)\ny = x * 2\nprint(y.grad_fn.name(), y._version)   # MulBackward0 0\ny.add_(1)\nprint(y.grad_fn.name(), y._version)   # AddBackward0 1\ny.sum().backward()\nprint(x.grad)                         # tensor([2., 2., 2.])\n\nb = torch.ones(4)\nv = b[:2]\nv.add_(1)\nprint(b._version, v._version)         # 1 1',
            },
          },
          {
            h: 'what the check actually compares',
            ps: [
              "Two lines of C++ from opposite ends of `SavedVariable` are the whole comparison. On the way in, the constructor copies the tensor's current version into `saved_version_`. On the way out, unpack reads the version again and raises if the two differ. Nothing is checksummed and nothing is copied; a saved tensor is a pointer plus an integer taken at save time.",
              "That is why the failure is reported at the backward pass rather than at the write. Nobody is watching the tensor. The counter increments silently, and the mismatch is discovered when the node finally asks for what it saved, which can be many lines later in a place with no obvious connection to the mutation.",
              "It also means you can run the check yourself, early. `_version` on a tensor you know a node saved tells you before you ever call backward whether the number has moved.",
            ],
            code: {
              caption: 'verbatim, torch/csrc/autograd/saved_variable.cpp at pytorch v2.2.2: line 50 from the constructor, then lines 160-167 from unpack, joined here under added headings',
              lang: 'c',
              text: '// on the way in, at SavedVariable::SavedVariable\n    saved_version_ = version_counter.current_version();\n\n// on the way out, at SavedVariable::unpack\n  // Only check version counter in the case without hooks\n  // If user provides hooks, we can\'t track versions through the hooks\n  if (!hooks_) {\n    auto current_version = saved_original_\n        ? impl::version_counter(data_).current_version()\n        : version_counter_.current_version();\n\n    if (saved_version_ != current_version) {',
            },
          },
          {
            h: 'and the hook that turns the check off',
            ps: [
              "The `if (!hooks_)` in that excerpt is a guard with a consequence most people meet by accident. Install `saved_tensors_hooks` and the version check is skipped, because a pack function may have returned something that is not a tensor at all and there is nothing left to read a version from.",
              "So the exact program that raises without hooks computes a wrong answer with them. Under the hooks, `e.grad` comes back 3.7183, which is the mutated saved value; the gradient of `sum(exp(e))` at `e = 1` is 2.7183. No warning, no error, a number that is off by exactly the size of the in-place add.",
              "The same hook pair is the mechanism behind offloading saved tensors to host memory, which is the reason it exists and a good thing to do. What it costs is the guardrail, and any pack and unpack pair you write should be treated as code that has to be right on its own, because nothing downstream is going to check it.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): hooks installed, check skipped, answer silently wrong',
              lang: 'python',
              text: "import torch\n\ne = torch.ones(3, requires_grad=True)\nwith torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):\n    z = e.exp()\n    z.add_(1)           # the write the museum's version-counter exhibit catches\n    z.sum().backward()\nprint(e.grad)           # tensor([3.7183, 3.7183, 3.7183]); the answer is 2.7183",
            },
          },
        ],
        readings: [
          { label: 'Hooks for autograd saved tensors', url: 'https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html', note: 'the official tutorial for pack and unpack, including the offload-to-CPU pattern' },
          { label: 'saved_variable.cpp at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/csrc/autograd/saved_variable.cpp', note: 'the save, the unpack, the version comparison, and the message it formats, in 250 lines' },
          { label: 'Autograd mechanics: in-place operations', url: 'https://docs.pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd', note: "the maintainers' own case against in-place ops, and the two reasons they still support them" },
        ],
        check: [
          {
            q: 'Two Linear layers in the same model have identically shaped weights. Why is only one of them saved for backward?',
            a: 'Because a matmul saves its weight only when a gradient for its input is needed, and the first layer feeds on data that carries no gradient. Measured on a 3-layer MLP, 0.weight was absent and 2.weight present, a difference of 1,048,576 bytes.',
          },
          {
            q: 'What exactly does the version counter compare, and when is the comparison made?',
            a: 'The integer recorded when the tensor was saved against the integer the tensor reports when the node unpacks it, which happens during the backward pass. Nothing is checksummed, so a mutation is discovered at unpack rather than at the write that caused it.',
          },
          {
            q: 'You installed saved_tensors_hooks and a program that used to raise now returns a number. Should you trust it?',
            a: 'No. The version check runs only when no hooks are installed, so the same in-place write that raised before now passes silently. The measured case returned 3.7183 where the correct gradient is 2.7183.',
          },
        ],
        work: [
          { id: 'byte-audit', label: 'run the pack-hook audit on one of your own models and reconcile every saved storage against the op that needs it', href: '#what-gets-saved-depends-on-what-you-asked-for' },
          { id: 'version-probe', label: 'take a loop that raises the version-counter error and locate the offending write by reading _version alone, before any backward call' },
        ],
      },
      {
        id: 'a-backward-you-can-defend',
        num: 4,
        title: 'A backward you can defend',
        lede: 'The chapter calls torch.autograd.Function the escape hatch for when the automatic rule is wrong. Two spellings of the vector norm, mathematically identical, disagree about the gradient at the origin, which is a concrete answer to when that happens.',
        goal: 'Write a torch.autograd.Function whose backward returns one value per forward argument and honours needs_input_grad, check it against a numerical Jacobian and read the failure output when it disagrees, and say why a deliberately wrong backward can still be the right thing to ship.',
        sections: [
          {
            h: 'two spellings, two gradients, same point',
            ps: [
              "`x.norm()` and `torch.sqrt((x * x).sum())` compute the same number for every input. At `x = 0` their gradients differ: the first returns zeros, the second returns NaN. Neither is a bug. The norm has no derivative at the origin, PyTorch's `linalg_vector_norm` derivative formula special-cases it to zero, and the spelled-out version inherits the derivative of `sqrt`, which is unbounded as its argument goes to zero.",
              "That gap is the practical form of the chapter's claim that the automatic rule is sometimes the wrong one. Autograd differentiates the ops you called, correctly and one at a time. It has no view of the function you meant, so two decompositions of one function can have two different numerical characters and it will faithfully give you whichever you wrote.",
              "A NaN gradient at the origin is not hypothetical either. Normalizing a vector that can be zero, a distance between two points that can coincide, a standard deviation over a constant window: all of them reach that point during training, and once one NaN enters a parameter, every subsequent step carries it.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same function, two spellings, two gradients at zero',
              lang: 'python',
              text: 'import torch\n\nx = torch.zeros(3, requires_grad=True)\nx.norm().backward()\nprint(x.grad)                     # tensor([0., 0., 0.])\n\ny = torch.zeros(3, requires_grad=True)\ntorch.sqrt((y * y).sum()).backward()\nprint(y.grad)                     # tensor([nan, nan, nan])',
            },
          },
          {
            h: 'writing the node yourself',
            ps: [
              "A `Function` subclass adds one node to the graph in place of everything its forward did. `forward` computes the value and stashes what backward will need with `ctx.save_for_backward`, which routes through the same `SavedVariable` machinery lesson three took apart, version counter included. `backward` receives one incoming gradient per output and returns one gradient per forward argument.",
              "Two details in the code below are the ones people get wrong first. The return has two entries because `forward` took two arguments, and the second is `None` because `eps` is a float that no gradient can flow to. And `ctx.needs_input_grad` is a tuple of booleans in the same positions, so a backward that skips work for arguments nobody asked about is a one-line guard rather than a redesign.",
              "The eps inside the square root is the whole fix. Smoothing the norm to `sqrt(sum(x*x) + eps*eps)` makes the function differentiable everywhere and moves the gradient at the origin to a clean zero, at the cost of a value that is off by eps. Whether that trade is acceptable is a modelling question, and it is one you can only make deliberately once the derivative is yours.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a smoothed norm, its own backward, both checks green',
              lang: 'python',
              text: 'import torch\nfrom torch.autograd import Function, gradcheck, gradgradcheck\n\nclass SafeNorm(Function):\n    @staticmethod\n    def forward(ctx, x, eps):\n        n = torch.sqrt((x * x).sum() + eps * eps)\n        ctx.save_for_backward(x, n)\n        return n\n\n    @staticmethod\n    def backward(ctx, g):\n        x, n = ctx.saved_tensors\n        gx = g * x / n if ctx.needs_input_grad[0] else None\n        return gx, None          # one return per forward argument; eps gets None\n\nz = torch.zeros(3, dtype=torch.double, requires_grad=True)\nSafeNorm.apply(z, 1e-6).backward()\nprint(z.grad)                    # tensor([0., 0., 0.], dtype=torch.float64)\n\nt = torch.randn(4, dtype=torch.double, requires_grad=True)\nprint(gradcheck(lambda a: SafeNorm.apply(a, 1e-6), (t,)))      # True\nprint(gradgradcheck(lambda a: SafeNorm.apply(a, 1e-6), (t,)))  # True',
            },
          },
          {
            h: 'what gradcheck is actually comparing',
            ps: [
              "`gradcheck` perturbs each input entry by `eps`, defaulting to 1e-6, and divides the change in each output by it to build a numerical Jacobian. Then it builds the analytic Jacobian by calling your backward once per output entry. If the two agree within `atol` of 1e-5 and `rtol` of 1e-3, it returns `True`; otherwise it raises `GradcheckError` and prints both matrices.",
              "Those three constants explain the double-precision rule that every tutorial states and few justify. A float32 tensor holds about seven decimal digits, and a perturbation of 1e-6 against values near 1 destroys most of them before the subtraction. The run below is the same correct `Square` backward checked twice: float32 fails with a numerical column of 2.9802 against an analytic 3.0820, float64 passes.",
              "Read the failure output as a diagnosis rather than a verdict. A numerical column near zero where the analytic one is not means the function is locally flat. Disagreement in the last digits means precision. Disagreement in sign or magnitude means the derivative is wrong. Only the last one is your bug.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one correct backward, failing in float32 and passing in float64',
              lang: 'python',
              text: 'import torch\nfrom torch.autograd import Function, gradcheck\n\nclass Square(Function):\n    @staticmethod\n    def forward(ctx, x):\n        ctx.save_for_backward(x)\n        return x * x\n\n    @staticmethod\n    def backward(ctx, g):\n        (x,) = ctx.saved_tensors\n        return g * 2 * x\n\ntorch.manual_seed(0)\nf = torch.randn(2, requires_grad=True)          # float32\ntry:\n    gradcheck(Square.apply, (f,))\nexcept Exception as e:\n    print(e)\n# Jacobian mismatch for output 0 with respect to input 0,\n# numerical:tensor([[ 2.9802,  0.0000],\n#         [ 0.0000, -0.5960]])\n# analytical:tensor([[ 3.0820,  0.0000],\n#         [-0.0000, -0.5869]])\nprint(gradcheck(Square.apply, (f.detach().double().requires_grad_(True),)))  # True',
            },
          },
          {
            h: 'a backward that is wrong on purpose',
            ps: [
              "Rounding has a derivative of zero almost everywhere, and autograd reports exactly that: `x.round().sum().backward()` fills `x.grad` with zeros. A quantized layer built on it trains nowhere, because no gradient survives the round.",
              "The standard answer is to lie in one specific place. A straight-through estimator rounds in forward and passes the incoming gradient through untouched in backward, so the rest of the network trains as if the round were the identity. Ten lines, and `gradcheck` rejects it flatly: the numerical Jacobian is all zeros, the analytic one is the identity, and the error prints both.",
              "That failure is the correct result and the reason to run the check anyway. Now the disagreement is documented instead of assumed, you know it is total rather than partial, and the next person to read the class can see which of the two Jacobians was chosen on purpose. A custom Function is a claim about a derivative, and running the check turns the claim into either a green result or a known, deliberate exception.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the true derivative, the useful one, and the check rejecting the useful one',
              lang: 'python',
              text: 'import torch\nfrom torch.autograd import Function, gradcheck\n\nclass RoundSTE(Function):\n    @staticmethod\n    def forward(ctx, x):\n        return torch.round(x)\n\n    @staticmethod\n    def backward(ctx, g):\n        return g\n\nx = torch.tensor([0.3, 1.7], requires_grad=True)\nx.round().sum().backward()\nprint(x.grad)                    # tensor([0., 0.]): the true derivative\n\ny = torch.tensor([0.3, 1.7], requires_grad=True)\nRoundSTE.apply(y).sum().backward()\nprint(y.grad)                    # tensor([1., 1.]): the one that trains\n\nt = torch.tensor([0.3, 1.7], dtype=torch.double, requires_grad=True)\ntry:\n    gradcheck(RoundSTE.apply, (t,))\nexcept Exception as e:\n    print(type(e).__name__)      # GradcheckError\n    print(e)\n# Jacobian mismatch for output 0 with respect to input 0,\n# numerical:tensor([[0., 0.],\n#         [0., 0.]], dtype=torch.float64)\n# analytical:tensor([[1., 0.],\n#         [0., 1.]], dtype=torch.float64)',
            },
          },
          {
            h: 'what carries over to the other path',
            ps: [
              "Three of the four decisions in `SafeNorm` are not PyTorch decisions. Something has to hold the values backward will read, something has to know which inputs want gradients, and something has to check the analytic rule against a numerical one. The JAX path names those `res` in a `custom_vjp` residual tuple, the argnums of the transform, and `jax.test_util.check_grads`.",
              "What is genuinely local to this arc is where the state lives. A `ctx` object is mutable, per call, and carries a version counter into a comparison that happens later; a residual tuple is a value returned from one function and passed to another. That is the same split the modules chapter draws between state on an object and state threaded through a signature, showing up one layer down in the autodiff machinery itself.",
              "Everything else in this arc holds for both. A tape is nodes and edges, a node keeps what its rule needs, and a rule you wrote yourself is worth exactly as much as the check you ran against it.",
            ],
          },
        ],
        readings: [
          { label: 'Extending PyTorch: torch.autograd.Function', url: 'https://docs.pytorch.org/docs/stable/notes/extending.html', note: 'the full contract: setup_context, needs_input_grad, mark_dirty, once_differentiable, and when not to use it' },
          { label: 'torch.autograd.gradcheck', url: 'https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html', note: 'every argument, and the double-precision requirement stated in the first paragraph' },
          { label: 'gradcheck.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/autograd/gradcheck.py', note: 'the perturbation loop, the Jacobian comparison, and the message the failures above were printed by' },
        ],
        check: [
          {
            q: 'Why do x.norm() and torch.sqrt((x * x).sum()) return different gradients at x = 0?',
            a: 'Because autograd differentiates the ops you called, not the function you meant. The norm has its own derivative formula that special-cases the origin to zero; the spelled-out version inherits the derivative of sqrt, which is unbounded there, so it returns NaN.',
          },
          {
            q: 'Your backward returns a single tensor and forward took two arguments. What goes wrong?',
            a: 'The return has to have one entry per forward argument, in order. An argument no gradient can flow to, such as a float eps, gets None in its position, and ctx.needs_input_grad carries a boolean per position so backward can skip work nobody asked for.',
          },
          {
            q: 'gradcheck fails on a backward you know is correct. What do you check before changing the derivative?',
            a: 'The dtype. gradcheck perturbs by 1e-6 and compares within atol 1e-5, which float32 cannot resolve. The same correct Square backward failed in float32 with numerical 2.9802 against analytic 3.0820, and passed in float64.',
          },
        ],
        work: [
          { id: 'own-function', label: 'write one Function for an op in your own code, honour needs_input_grad, and run both gradcheck and gradgradcheck on it in double precision', href: '#writing-the-node-yourself' },
          { id: 'ste-audit', label: 'find one straight-through estimator in a codebase you use and write down, in one sentence, which Jacobian it chose and why' },
        ],
      },
    ],
  },
]
