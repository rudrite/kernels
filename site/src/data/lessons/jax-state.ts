// New file: site/src/data/lessons/jax-state.ts
// Under chapter 9's pattern: where threaded state lands in the recording, what
// a donated state tree gets granted leaf by leaf, and which fields earn a
// place in a train state. The single-array side of donation is the JAX
// performance chapter's, and this arc leans on it rather than retelling it.
// Every snippet here and every
// value it prints ran on this machine, jax 0.4.38 on CPU, Python 3.12,
// 2026-08-15. No optax in this arc: the local environment has none, so the
// optimizer state is written out by hand where every leaf is visible, and the
// library that packages the same pair of functions is named, not quoted.
import type { UnitLessons } from './index'

export const JAX_STATE_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:state',
    lessons: [
      {
        id: 'the-step-returns-its-state',
        num: 1,
        title: 'The step that returns its state',
        lede: 'A traced function has two ways to reach a value: through an argument, or by reading a name from the scope around it. The jaxpr keeps those in two different slots, and only one of the two ever gets a new value again.',
        goal: 'Point at the constvar slot in a jaxpr and say what landed there, read a step function’s state off its invars and outvars, and name what a scan carry demands of a state tree that a Python loop never asks for.',
        sections: [
          {
            h: 'two slots in the recording, and only one takes new values',
            ps: [
              "Trace a function that reads an array from the enclosing scope and the jaxpr has something before the semicolon that chapter 2's softmax never had. `{ lambda a:f32[3]; b:f32[3]. ... }` names two slots: constvars on the left of the semicolon, invars on the right. The closed-over `bias` went left. The argument `x` went right.",
              "The two slots differ in when they receive a value. An invar gets a fresh one on every call, because it is an argument. A constvar received its value once, during the trace, and the executable carries that array inside itself from then on.",
              "Rebinding the Python name is what makes the second half visible. After `bias = jnp.zeros(3)` the compiled function still adds ones, because it holds the array that was there when it traced, not the name that used to point at it. Nothing raises and nothing warns. A loop that reassigns a global every step keeps computing with the value from step zero.",
              '>> State you read from scope is a constvar. State you pass is an invar.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the constvar slot, and a rebind the executable never hears about',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nbias = jnp.ones(3)\n\ndef add_bias(x):\n    return x + bias\n\nprint(jax.make_jaxpr(add_bias)(jnp.zeros(3)))\n\nfast = jax.jit(add_bias)\nprint(fast(jnp.zeros(3)))\nbias = jnp.zeros(3)          # rebind the name the function reads\nprint(fast(jnp.zeros(3)))\n\n# { lambda a:f32[3]; b:f32[3]. let c:f32[3] = add b a in (c,) }\n# [1. 1. 1.]\n# [1. 1. 1.]',
            },
          },
          {
            h: 'the state is the leading invars and the leading outvars',
            ps: [
              "Thread the same value as an argument instead and it moves to the other slot, where it can be replaced. The step below carries `(count, total)`, and the jaxpr opens with `a:i32[] b:f32[3] c:f32[3]`: the state tuple flattened into two invars, then the batch. It closes with `(d, e, g)`, which is the new count, the new total, and the per-step output, in that order. One leaf of state, one invar, one outvar.",
              "The equation list also prices the counter. `f:f32[] = convert_element_type[new_dtype=float32 weak_type=False] d` is in the program because an `i32[]` count met a float division, and the recording will not do that silently. A state field's dtype is visible work, not an implementation detail of your dataclass.",
              "That one-leaf-one-slot correspondence is what the next lesson's alias map indexes into. Donation is expressed per output position against per input position, so a state tree with eight leaves gives the compiler eight independent decisions to make.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a step, and the recording it produces',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef step(state, x):\n    count, total = state\n    count = count + 1\n    total = total + x\n    return (count, total), total / count\n\nprint(jax.make_jaxpr(step)((jnp.int32(0), jnp.zeros(3)), jnp.ones(3)))\n\n# { lambda ; a:i32[] b:f32[3] c:f32[3]. let\n#     d:i32[] = add a 1\n#     e:f32[3] = add b c\n#     f:f32[] = convert_element_type[new_dtype=float32 weak_type=False] d\n#     g:f32[3] = div e f\n#   in (d, e, g) }',
            },
          },
          {
            h: 'the contract belongs to scan too',
            ps: [
              "`(state, x) -> (new_state, y)` is not only how a training step is written. It is the exact signature `jax.lax.scan` wants of a body, which chapter 6 covers from the loop's side. Read from the state's side, the claim is narrower and more useful: the carry is your state tree, and the two names describe one object.",
              "The carry adds one requirement a Python loop never enforces. What comes out has to have the same type as what went in, leaf for leaf, and `count + 1.0` instead of `count + 1` is enough to break it: an `int32[]` carry component comes back as `float32[]` and scan refuses before running anything.",
              "Read the refusal closely and it names `state[0]`, rooted at the body's own parameter name and indexed by position in the flattened tree. On a deep state that path is how you find the field, without bisecting the step by hand.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one float where an int was, and the message that comes back, hard-wrapped here to the panel and otherwise verbatim',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef step(state, x):\n    count, total = state\n    return (count + 1.0, total + x), total    # count leaves as a float\n\ninit = (jnp.int32(0), jnp.zeros(3))\ntry:\n    jax.lax.scan(step, init, jnp.ones((4, 3)))\nexcept TypeError as err:\n    print(err)\n\n# scan body function carry input and carry output must have equal types (e.g.\n# shapes and dtypes of arrays), but they differ:\n#\n# The input carry component state[0] has type int32[] but the corresponding\n# output carry component has type float32[], so the dtypes do not match.\n#\n# Revise the function so that all output types (e.g. shapes and dtypes) match\n# the corresponding input types.',
            },
          },
          {
            h: 'the same eight steps, two schedules',
            ps: [
              "Run the step eight times in a Python loop, then hand the identical function to scan over the identical inputs. The final carry matches, and so does the last per-step output: `(Array(8, dtype=int32), Array(36., dtype=float32))` and `3.5` from both.",
              "Nothing about the step was edited between the two runs. The Python loop rebinds a name eight times and dispatches eight programs; scan hands the carry from one iteration to the next inside one. Which of those you want is a compile-time and memory question that chapters 6 and 11 both weigh.",
              "The portability is the part worth keeping. A step that threads its state runs unchanged under a loop, under scan, under jit, and under grad. A step that mutates something outside itself runs correctly under none of them, and the first section showed what it computes instead.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a Python loop and a scan over one unchanged step',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef step(state, x):\n    count, total = state\n    return (count + 1, total + x), total / (count + 1)\n\nxs = jnp.arange(1.0, 9.0)\n\nstate = (jnp.int32(0), jnp.float32(0.0))\nfor x in xs:\n    state, mean = step(state, x)\n\nscanned, means = jax.lax.scan(step, (jnp.int32(0), jnp.float32(0.0)), xs)\nprint(state, mean)\nprint(scanned, means[-1])\nprint(state[1] == scanned[1], mean == means[-1])\n\n# (Array(8, dtype=int32), Array(36., dtype=float32)) 3.5\n# (Array(8, dtype=int32), Array(36., dtype=float32)) 3.5\n# True True',
            },
          },
        ],
        readings: [
          { label: 'JAX · jaxpr reference', url: 'https://docs.jax.dev/en/latest/jaxpr.html', note: 'the grammar behind the two slots: constvars, then invars' },
          { label: 'JAX · Stateful computations', url: 'https://docs.jax.dev/en/latest/stateful-computations.html', note: 'the official walk from a mutating counter to a threaded one' },
          { label: 'JAX · jax.lax.scan', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html', note: 'the carry contract this lesson reads as a state contract' },
        ],
        check: [
          {
            q: 'A jaxpr prints as { lambda a:f32[3]; b:f32[3]. let ... }. Which value sits in which slot, and which of the two can a later call replace?',
            a: 'The constvar a is a value closed over from scope and frozen into the executable at trace time; the invar b is the argument. Only the invar gets a new value per call, so rebinding the Python name behind a constvar changes nothing about what runs.',
          },
          {
            q: 'scan refuses a step with a message naming state[0]. What did the body do, and how does that path help?',
            a: 'It returned a carry component whose type differs from the one it received, such as int32[] in and float32[] out after a count + 1.0. The path is rooted at the body’s parameter name and indexed into the flattened tree, so it names the offending leaf rather than the whole state.',
          },
          {
            q: 'A step threads its state and one running mean is kept in a module-level list instead. What breaks first?',
            a: 'The list is appended at trace time only, so it records one entry and never updates again, while the compiled program keeps whatever value was frozen into it. Nothing raises; the numbers are simply the trace-time ones.',
          },
        ],
        work: [
          { id: 'name-the-constvar', label: 'print the jaxpr of one function of your own that reads a value from scope, then name every entry in its constvar slot', href: '#two-slots-in-the-recording-and-only-one-takes-new-values' },
          { id: 'loop-to-scan', label: 'hand a step you wrote for a Python loop to scan unchanged, and fix whatever the carry rule refuses', href: '#the-contract-belongs-to-scan-too' },
        ],
      },
      {
        id: 'donating-the-state-tree',
        num: 2,
        title: 'Donating the state tree',
        lede: 'Donating one array is a single yes or no. Donating a state is one answer per leaf, and the compiled module carries all of those answers before a step has run.',
        goal: 'Read a per-leaf alias map off a compiled step, check its byte total against the state’s own leaves, explain how one leaf loses its alias while its siblings keep theirs, and name what else in a loop a donated state takes with it.',
        sections: [
          {
            h: 'one answer per leaf',
            ps: [
              "What `donate_argnums` promises for a single argument, what becomes of the array you gave away, and the two mismatches that get a donation refused all belong to the performance chapter's lesson on donation. Read that one first; this lesson starts where the donated argument stops being an array and starts being a tree. What the compiler then does with a granted alias, and where it inserts copies to keep a live value from being clobbered, is the XLA path's at /xla/layout-memory/copies-and-buffers.",
              "Mark the state argument and the header of the compiled module comes back with one entry per leaf: `input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias) }`. Output position zero may take input position zero's buffer, and so on down the flattened tree. That is the one-leaf-one-slot correspondence from lesson one, read back from the compiler's side.",
              "`alias_size_in_bytes` totals the map, and the total is checkable against the state itself. It reads 84 here, and `sum(leaf.nbytes for leaf in jax.tree.leaves(state))` also reads 84, so every leaf of this state is covered and none was left out. Check that equality on a real parameter tree rather than assuming it, because a state can come back with most of its leaves aliased and one of them not.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one alias entry per state leaf, and the bytes they add up to',
              lang: 'python',
              text: 'from dataclasses import dataclass\n\nimport jax\nimport jax.numpy as jnp\n\n@jax.tree_util.register_dataclass\n@dataclass\nclass State:\n    w: jax.Array\n    b: jax.Array\n    step: jax.Array\n\ndef sgd(state, g):\n    return State(w=state.w - 0.1 * g, b=state.b + 1.0, step=state.step + 1)\n\nstate = State(w=jnp.ones((4, 4)), b=jnp.zeros(4), step=jnp.zeros((), jnp.int32))\nc = jax.jit(sgd, donate_argnums=0).lower(state, jnp.ones((4, 4))).compile()\nhead = c.as_text().splitlines()[0]\nprint(head[: head.index("entry_computation_layout")])\nprint(c.memory_analysis().alias_size_in_bytes)\nprint(sum(leaf.nbytes for leaf in jax.tree.leaves(state)))\n\n# HloModule jit_sgd, is_scheduled=true, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias) },\n# 84\n# 84',
            },
          },
          {
            h: 'one leaf can lose its alias while the rest keep theirs',
            ps: [
              "A donated array either aliases or it does not. A donated tree has a middle case: most leaves keep their alias, one drops out of the map, and the executable ends up doing part of what you asked for.",
              "The two functions below differ in a single character. One returns the step counter as `state.step + 1`, the other as `state.step + 1.0`, so an `int32[]` goes in and a `float32[]` comes back. The map loses its `{2}` entry, the total falls from 84 bytes to 80, and the missing 4 is the counter. The weight and bias entries are identical in both.",
              "The refusal also arrives on stderr, in the same warning the performance lesson reads, naming the aval it could not place. What the map adds is arithmetic the warning cannot give you: which leaf, out of how many, and what fraction of the promise survived. A whole-tree answer would say the donation failed. This one says you got 80 of the 84 bytes you asked for.",
              '>> On a state, donation is not granted or refused. It is granted leaf by leaf.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same step twice, one character apart; stdout only, since the refusal warning belongs to the performance lesson',
              lang: 'python',
              text: 'from dataclasses import dataclass\n\nimport jax\nimport jax.numpy as jnp\n\n@jax.tree_util.register_dataclass\n@dataclass\nclass State:\n    w: jax.Array\n    b: jax.Array\n    step: jax.Array\n\ndef kept(state, g):\n    return State(w=state.w - 0.1 * g, b=state.b + 1.0, step=state.step + 1)\n\ndef drifted(state, g):\n    return State(w=state.w - 0.1 * g, b=state.b + 1.0, step=state.step + 1.0)\n\nstate = State(w=jnp.ones((4, 4)), b=jnp.zeros(4), step=jnp.zeros((), jnp.int32))\nfor fn in (kept, drifted):\n    c = jax.jit(fn, donate_argnums=0).lower(state, jnp.ones((4, 4))).compile()\n    head = c.as_text().splitlines()[0]\n    print(fn.__name__, head[head.index("input_output_alias") : head.index("entry_computation")])\n    print("   ", c.memory_analysis().alias_size_in_bytes, "bytes")\n\n# kept input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias) },\n#     84 bytes\n# drifted input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) },\n#     80 bytes',
            },
          },
          {
            h: 'donation reaches leaves you kept a name for',
            ps: [
              "Donating a state donates its leaves, and a leaf is an object, not a location. Pull one out and stash it somewhere else, last epoch's weights in a metrics dict, a slice held back for a diff, and the thing you stashed is the same array the compiler was given permission to overwrite. It goes with the tree.",
              "The run below keeps `state.w` under a second name, calls one donating step, and asks both names what survived. The kept name reports its array deleted; the batch, which was never donated, reports False. So the audit before you add the keyword is not about the step at all. It is about every other place in the loop that still holds a leaf.",
              "That also settles which argument takes the keyword. The state is replaced by the step's own output, so nothing upstream needs it once the call returns. The batch arrives from the host and downstream code usually still wants it for a metric, which makes donating it a way to delete an array you are about to read.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one donating step, one leaf held under a second name, and one batch that was not donated',
              lang: 'python',
              text: 'from dataclasses import dataclass\n\nimport jax\nimport jax.numpy as jnp\n\n@jax.tree_util.register_dataclass\n@dataclass\nclass State:\n    w: jax.Array\n    b: jax.Array\n\ndef step(state, batch):\n    return State(w=state.w - 0.1 * batch, b=state.b + 1.0)\n\nfast = jax.jit(step, donate_argnums=0)\n\nstate = State(w=jnp.ones((4, 4)), b=jnp.zeros(4))\nbatch = jnp.ones((4, 4))\nkept = state.w                    # one leaf, held under a second name\nstate = fast(state, batch)\nprint("the leaf held elsewhere:", kept.is_deleted())\nprint("the batch:", batch.is_deleted())\nprint("the new weight:", state.w[0, 0])\n\n# the leaf held elsewhere: True\n# the batch: False\n# the new weight: 0.9',
            },
          },
        ],
        readings: [
          { label: 'JAX · Buffer donation', url: 'https://docs.jax.dev/en/latest/buffer_donation.html', note: 'the rule as the source states it, including what donating a pytree rather than an array means' },
          { label: 'JAX · jax.jit', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.jit.html', note: 'donate_argnums and donate_argnames, in the signature they belong to' },
          { label: 'JAX · Ahead-of-time compilation', url: 'https://docs.jax.dev/en/latest/aot.html', note: 'lower and compile without running, which is how the map gets read before a step happens' },
        ],
        check: [
          {
            q: 'A compiled step reports 80 alias bytes where the state’s leaves total 84. What happened, and to which leaf?',
            a: 'One leaf lost its alias and the others kept theirs. Its input and output stopped matching in shape or dtype, here a counter returning float32 where an int32 went in, so the map came back with two entries instead of three and the missing 4 bytes are the counter.',
          },
          {
            q: 'You donate a state, and a metrics dict elsewhere still holds last epoch’s weight array. What happens to that array?',
            a: 'It goes with the tree. Donation is granted per leaf, and that leaf is the same object the metrics dict points at, so a second name does not protect it. Either keep a copy made before the step or keep the array out of the state.',
          },
          {
            q: 'Why does a training step donate its state and not its batch?',
            a: 'The step returns a new state of the same shapes, so every old leaf has an output leaf that can take its buffer, and the caller rebinds the name anyway. The batch comes from the host and downstream code usually still reads it, so donating it deletes an array that is about to be used.',
          },
        ],
        work: [
          { id: 'read-the-map', label: 'compile one step of your own with the state donated, then read its alias map entry by entry against the state’s own leaves and check the byte totals match', href: '#one-answer-per-leaf' },
          { id: 'hunt-the-reference', label: 'list every other name your loop holds on a state leaf, and decide for each one whether it survives the donation', href: '#donation-reaches-leaves-you-kept-a-name-for' },
        ],
      },
      {
        id: 'a-state-that-earns-its-shape',
        num: 3,
        title: 'A state that earns its shape',
        lede: 'Four kinds of thing want a seat in a training state: parameters, optimizer moments, a step counter, and settings that are not arrays at all. Each has to answer the same questions, and only one of the four answers them by not being a leaf.',
        goal: 'Say for any field whether it lands in the leaves or in the treedef, predict what changing it costs in traces or in bytes, compute what an Adam state adds to a parameter tree, and produce the abstract template a restore reads into.',
        sections: [
          {
            h: 'the moments are the parameter tree, twice',
            ps: [
              "Write Adam's two functions out by hand once and the state stops being an object with a name. `init` returns a count and two trees of zeros built with `jax.tree.map` over the params, so `jax.tree.structure(opt[\"mu\"]) == jax.tree.structure(params)` is True by construction, not by convention. `update` reads that dict and returns a new one.",
              "The leaves for a 512 by 512 weight and a 512 bias come out as `['int32[]', 'float32[512]', 'float32[512, 512]', 'float32[512]', 'float32[512, 512]']`: the count, then mu, then nu, with each tree's keys in sorted order. Five leaves for a two-leaf model.",
              "The bytes make the budget concrete. The parameters take 1050624 bytes, the optimizer state takes 2101252, and 2101252 minus twice 1050624 is 4, which is the int32 count and nothing else. So a step holds three copies of the parameter tree before the gradients it computes are counted, which is the memory that donation from lesson two is trying to stop doubling again.",
              "Optax packages this same pair of functions, and chapter 9 names the pieces. The arithmetic above does not change when you switch to it, because the leaves are the same leaves.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): Adam’s state, written out so every leaf is visible',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef adam_init(params):\n    zeros = jax.tree.map(jnp.zeros_like, params)\n    return {"count": jnp.zeros((), jnp.int32), "mu": zeros, "nu": zeros}\n\ndef adam_update(grads, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):\n    count = state["count"] + 1\n    mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["mu"], grads)\n    nu = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["nu"], grads)\n    updates = jax.tree.map(\n        lambda m, v: -lr * (m / (1 - b1**count)) / (jnp.sqrt(v / (1 - b2**count)) + eps),\n        mu,\n        nu,\n    )\n    return updates, {"count": count, "mu": mu, "nu": nu}\n\nparams = {"w": jnp.ones((512, 512)), "b": jnp.zeros(512)}\nopt = adam_init(params)\n\nprint(jax.tree.structure(opt["mu"]) == jax.tree.structure(params))\nprint([f"{leaf.dtype.name}{list(leaf.shape)}" for leaf in jax.tree.leaves(opt)])\n\np_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(params))\no_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(opt))\nprint(p_bytes, o_bytes, o_bytes - 2 * p_bytes)\n\n# True\n# [\'int32[]\', \'float32[512]\', \'float32[512, 512]\', \'float32[512]\', \'float32[512, 512]\']\n# 1050624 2101252 4',
            },
          },
          {
            h: 'the counter that is nearly an array',
            ps: [
              "Two ways to hold a step number look interchangeable. `0` and `jnp.zeros((), jnp.int32)` both mean zero, and `jax.eval_shape` prints the difference in one word: the Python int comes back as `ShapeDtypeStruct(shape=(), dtype=int32, weak_type=True)`, the array without the weak flag.",
              "The usual warning about the Python version is that it recompiles every step. Measured here, it does not. Three calls, two traces, and the second trace arrives with the strong-typed zero, not with a new value: a weak leaf and its own weak output share one executable however many times the number changes.",
              "Donation does not separate them either. Compiled with `donate_argnums=0`, both forms report 4 alias bytes, so nothing about buffer reuse argues for one over the other.",
              "The reason to write `jnp.zeros((), jnp.int32)` is the one measurement leaves standing: an aval that stays the same everywhere the state travels. A counter restored from a checkpoint is a strong int32 array, and the first call that mixes it with a weak-typed one changes the key, which is the flip chapter 3 warns about arriving through the state tree rather than through an argument you wrote by hand.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the weak flag, the trace it costs when the two forms meet, and the alias bytes it does not change',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nprint(jax.eval_shape(lambda s: s, 0))\nprint(jax.eval_shape(lambda s: s, jnp.zeros((), jnp.int32)))\n\ntraces = []\n\ndef advance(step):\n    traces.append(1)\n    return step + 1\n\nbump = jax.jit(advance)\nbump(0)                          # a Python int: weak_type=True\nbump(bump(0))                    # its own output, still weak\nbump(jnp.zeros((), jnp.int32))   # the same zero, read back from a checkpoint\nprint(len(traces))\n\nfor step in (0, jnp.zeros((), jnp.int32)):\n    compiled = jax.jit(advance, donate_argnums=0).lower(step).compile()\n    print(compiled.memory_analysis().alias_size_in_bytes, end=\' \')\nprint()\n\n# ShapeDtypeStruct(shape=(), dtype=int32, weak_type=True)\n# ShapeDtypeStruct(shape=(), dtype=int32)\n# 2\n# 4 4',
            },
          },
          {
            h: 'the template a restore reads into',
            ps: [
              "`jax.eval_shape(step, state, batch)` runs the step abstractly: no compile, no device work, no allocation. What comes back is the state with a `ShapeDtypeStruct` at every leaf, and `jax.tree.structure(out) == jax.tree.structure(s0)` is True, which is the cheapest available proof that a step returns the state it was given rather than something the same size.",
              "The leaf listing is in the treedef's own order, `['float32[4]', 'float32[8, 4]', 'float32[4]', 'float32[8, 4]', 'int32[]']`: the params dict with its keys sorted, then the optimizer dict, then the step. Any flatten of that treedef produces that order, which is what a restore has to line up against leaf for leaf.",
              "Chapter 12 sets the rule that the unit of checkpointing is the whole state tree, and the PyTorch path measures what each missing piece costs after a restore at /pytorch/the-loop/the-resume-that-matches. What this lesson adds is where the target tree comes from: an abstract state you can build before the first step runs, from the same function that will produce the real one.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the state’s shapes and dtypes, without running a step',
              lang: 'python',
              text: 'from dataclasses import dataclass\n\nimport jax\nimport jax.numpy as jnp\n\n@jax.tree_util.register_dataclass\n@dataclass\nclass TrainState:\n    params: dict\n    opt: dict\n    step: jax.Array\n\ndef step(state, batch):\n    x, y = batch\n\n    def loss_fn(p):\n        return jnp.mean((x @ p["w"] + p["b"] - y) ** 2)\n\n    loss, grads = jax.value_and_grad(loss_fn)(state.params)\n    mu = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, state.opt["mu"], grads)\n    params = jax.tree.map(lambda p, m: p - 0.01 * m, state.params, mu)\n    return TrainState(params=params, opt={"mu": mu}, step=state.step + 1), loss\n\np0 = {"w": jnp.ones((8, 4)) * 0.1, "b": jnp.zeros(4)}\ns0 = TrainState(params=p0, opt={"mu": jax.tree.map(jnp.zeros_like, p0)}, step=jnp.zeros((), jnp.int32))\n\nout, loss = jax.eval_shape(step, s0, (jnp.ones((16, 8)), jnp.zeros((16, 4))))\nprint(jax.tree.structure(out) == jax.tree.structure(s0))\nprint([f"{leaf.dtype.name}{list(leaf.shape)}" for leaf in jax.tree.leaves(out)])\nprint(loss)\n\n# True\n# [\'float32[4]\', \'float32[8, 4]\', \'float32[4]\', \'float32[8, 4]\', \'int32[]\']\n# ShapeDtypeStruct(shape=(), dtype=float32)',
            },
          },
          {
            h: 'the ledger',
            ps: [
              "Put the four kinds of field side by side and the design rule is short enough to hold in your head. A field is either a leaf or part of the structure. Leaves cost memory and can be donated; structure costs executables and cannot.",
              "One row is borrowed rather than proved here. A field declared static lands in the treedef instead of the leaves, and every distinct value it takes buys its own executable, which the pytrees chapter's lesson on registered nodes counts in its section on the aux slot and the cache key. What a train state adds is the last column: a static field is the one kind that cannot be donated, because it never became an array in the first place.",
              "The bottom row is the one that catches people out. A Python scalar in a state tree is a leaf like any other, and it costs nothing extra until the day something hands you the array-typed version of the same number.",
            ],
            table: {
              caption: 'byte and trace figures measured on this machine (jax 0.4.38, CPU), for params of one 512x512 weight and one 512 bias; the static row is cited to the pytrees lesson, not measured again here',
              cols: ['field', 'where it lives', 'what a change costs', 'donatable'],
              rows: [
                ['params, a dict of arrays', 'leaves', '1050624 bytes; a retrace when a shape or dtype moves', 'yes: the step returns the same shapes'],
                ['mu and nu, treedef equal to params', 'leaves', '2101252 bytes for the optimizer state, twice params plus the count', 'yes'],
                ['count or step as i32[]', 'one leaf, 4 bytes', 'nothing per value: one executable for every step number', 'yes, 4 alias bytes'],
                ['reduction as a static str', 'the treedef', 'one executable per distinct value it takes', 'no, it never became an array'],
                ['step as a Python int', 'one leaf, weak_type=True', 'a second trace the first time a strong int32 arrives', 'yes, the same 4 alias bytes'],
              ],
            },
          },
        ],
        readings: [
          { label: 'JAX · jax.tree_util.register_dataclass', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_dataclass.html', note: 'the data and static field split, and what each one becomes' },
          { label: 'JAX · jax.eval_shape', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html', note: 'shapes and dtypes without a compile or an allocation' },
          { label: 'Optax · transformations', url: 'https://optax.readthedocs.io/en/latest/api/transformations.html', note: 'the same init and update pair, with the state each transformation carries listed' },
        ],
        check: [
          {
            q: 'Your params take 1050624 bytes. What does an Adam state add, and where does the odd remainder come from?',
            a: 'Another 2101252 bytes, which is twice the parameter tree for mu and nu plus 4 bytes. The 4 is the int32 count, the only leaf in the optimizer state that is not shaped like a parameter.',
          },
          {
            q: 'Which field of a train state can never be donated, whatever the step does with it?',
            a: 'One declared static, such as a reduction mode, because it lives in the treedef rather than the leaves. Donation hands an input buffer to an output, and a static field never became a buffer; it is part of the structure that says where the leaves go.',
          },
          {
            q: 'If a Python int step counter does not recompile per step, why write it as jnp.zeros((), jnp.int32)?',
            a: 'Because the Python int carries weak_type=True in its aval, and a checkpoint restore hands back a strong int32. The first call that mixes the two changes the cache key and retraces, so the array form keeps one aval everywhere the state travels.',
          },
        ],
        work: [
          { id: 'byte-ledger', label: 'write the byte ledger for your own model: params, optimizer state, and the gradient tree the step builds, before you size the batch', href: '#the-moments-are-the-parameter-tree-twice' },
          { id: 'field-audit', label: 'take one train state you have written and put every field in one of the ledger’s rows, then move the ones that landed in the wrong column', href: '#the-ledger' },
        ],
      },
    ],
  },
]
