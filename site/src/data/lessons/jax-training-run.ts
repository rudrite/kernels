// New file: site/src/data/lessons/jax-training-run.ts
// The capstone loop under JAX chapter 12, taken apart: the state tree and what
// jit does with it, the three shapes a 200-step loop can take, and a resume
// proved by equality rather than by a curve. Every printed value ran on this
// machine (jax 0.4.38, jaxlib 0.4.38, CPU, Python 3.12, numpy 2.2.6,
// 2026-08-15). Adam is written out by hand here so the whole state tree is
// visible on the page; chapter 9 tells the optax version, and the chapter's own
// capstone runs it.
import type { UnitLessons } from './index'

export const JAX_TRAINING_RUN_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:training-run',
    lessons: [
      {
        id: 'one-tree-in-one-tree-out',
        num: 1,
        title: 'One tree in, one tree out',
        lede: 'A training run has no object holding its progress. It has one pytree that goes into a function and a new one that comes out, and every property that matters later, resumability included, is a property of that tree.',
        goal: 'Given a state tree, count its leaves and name the dtype of each, say how many executables a run of it compiles and what adds another, and name the way of building an optimizer state that gets a donated step refused.',
        sections: [
          {
            h: 'the whole run is one value',
            ps: [
              "Four parameter arrays, two Adam moments for each of them, and a step counter. Thirteen leaves, and that is the entire run: there is nothing else to save, nothing else to move to a device, nothing else a later step reads. Chapter 7 established that every transformation sees a model as leaves on a tree; this is what the tree looks like once an optimizer and a clock are in it.",
              "Two things in the printout are worth slowing down for. The step counter is `int32` while everything above it is `float32`, so a saver that assumes one dtype for the whole tree is already wrong. And the paths come back in the order `b1, b2, w1, w2`, not the order the dict was written, because dict keys flatten sorted. Match leaves to names by position and a rename of `w1` to `a1` silently reorders your checkpoint.",
              "The `State` type here is a `NamedTuple`, which JAX already knows how to flatten. A dataclass needs `register_dataclass` first, which the chapter 7 arc covers along with the sorted-key rule this printout is obeying, and the arc's third lesson depends only on the tree being flattenable, not on which of the two you picked.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): thirteen leaves, and the order they come out in; the treedef line is wrapped to fit',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nfrom typing import NamedTuple\n\n\nclass State(NamedTuple):\n    params: dict\n    m: dict            # Adam\'s first moment\n    v: dict            # Adam\'s second\n    step: jnp.ndarray\n\n\ndef init(key):\n    k1, k2 = jax.random.split(key)\n    params = {\n        "w1": jax.random.normal(k1, (8, 32)) * 0.1, "b1": jnp.zeros(32),\n        "w2": jax.random.normal(k2, (32, 1)) * 0.1, "b2": jnp.zeros(1),\n    }\n    return State(params,\n                 jax.tree.map(jnp.zeros_like, params),\n                 jax.tree.map(jnp.zeros_like, params),\n                 jnp.zeros((), jnp.int32))\n\n\nstate = init(jax.random.key(0))\npaths, treedef = jax.tree_util.tree_flatten_with_path(state)\nprint(len(paths), paths[-1][1].dtype)\nprint([jax.tree_util.keystr(k) for k, _ in paths[:5]])\nprint(treedef)\n\n# 13 int32\n# [".params[\'b1\']", ".params[\'b2\']", ".params[\'w1\']", ".params[\'w2\']", ".m[\'b1\']"]\n# PyTreeDef(CustomNode(namedtuple[State], [{\'b1\': *, \'b2\': *, \'w1\': *, \'w2\': *},\n# {\'b1\': *, \'b2\': *, \'w1\': *, \'w2\': *}, {\'b1\': *, \'b2\': *, \'w1\': *, \'w2\': *}, *]))',
            },
            table: {
              caption: 'the thirteen leaves of this run, measured on jax 0.4.38 CPU; the order is flatten order, which sorts dict keys',
              cols: ['leaves', 'shapes', 'dtype', 'what a resume needs it for'],
              rows: [
                ['params.b1, b2, w1, w2', '(32,) (1,) (8, 32) (32, 1)', 'float32', 'the model, and the only part a params-only checkpoint carries'],
                ['m.b1, b2, w1, w2', 'same as params', 'float32', "Adam's first moment; without it the step after the resume is already wrong"],
                ['v.b1, b2, w1, w2', 'same as params', 'float32', "Adam's second moment, and the denominator of every update"],
                ['step', '()', 'int32', 'the bias correction, and the key for the next batch'],
              ],
            },
          },
          {
            h: 'one step, one executable',
            ps: [
              "The step below is the chapter's step with the optimizer written out instead of called. `value_and_grad` returns the loss and the gradient tree together, the moments update leafwise through `tree.map`, and the bias correction reads the counter that rides in the same tree. Nothing is stored anywhere; the function takes a `State` and returns a `State`.",
              "The batch is drawn inside the step from a key the caller derives with `fold_in`, so the step number alone decides which 32 rows this step sees. Chapter 8 made that choice for readability under `scan`. Lesson three spends the whole page on what it buys at resume time.",
              "After twenty steps the loss reads 1.720861 and the jit cache holds exactly one entry. One entry for twenty calls is the steady state a training loop is supposed to reach, and `_cache_size` is the cheapest way to check it. It is a private helper rather than public API, which is fine for a check you run once and delete.",
              '>> Twenty calls, one executable. That number is the health check.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): Adam by hand, twenty steps, one compile',
              lang: 'python',
              text: 'LR, B1, B2, EPS = 1e-2, 0.9, 0.999, 1e-8\n\n\ndef predict(p, x):\n    h = jax.nn.relu(x @ p["w1"] + p["b1"])\n    return h @ p["w2"] + p["b2"]\n\n\ndef loss_fn(p, batch):\n    x, y = batch\n    return jnp.mean((predict(p, x) - y) ** 2)\n\n\ndef step_fn(state, data, key):\n    x, y = data\n    idx = jax.random.randint(key, (32,), 0, x.shape[0])\n    loss, g = jax.value_and_grad(loss_fn)(state.params, (x[idx], y[idx]))\n    t = state.step + 1\n    m = jax.tree.map(lambda m, g: B1 * m + (1 - B1) * g, state.m, g)\n    v = jax.tree.map(lambda v, g: B2 * v + (1 - B2) * g * g, state.v, g)\n    params = jax.tree.map(\n        lambda p, m, v: p - LR * (m / (1 - B1 ** t)) / (jnp.sqrt(v / (1 - B2 ** t)) + EPS),\n        state.params, m, v,\n    )\n    return State(params, m, v, t), loss\n\n\ntrain_step = jax.jit(step_fn)\n\nx = jax.random.normal(jax.random.key(1), (256, 8))\ndata = (x, jnp.sum(x, axis=1, keepdims=True))\nbase = jax.random.key(2)\n\nstate = init(jax.random.key(0))\nfor step in range(20):\n    state, loss = train_step(state, data, jax.random.fold_in(base, step))\n\nprint(int(state.step), f"{loss:.6f}", type(loss).__name__)\nprint(train_step._cache_size())\n\n# 20 1.720861 ArrayImpl\n# 1',
            },
          },
          {
            h: 'a Python int in the tree costs two more executables',
            ps: [
              "Start the same run with `step=0` as a plain Python integer instead of a zero-dimensional array and the loop still produces the right numbers. The cache goes from one entry to three.",
              "Three, not two, and the third one is the part worth understanding. The first call sees a Python `int` leaf and compiles for it. What that call returns is an `int32` array whose `weak_type` is `True`, because it was built from a weakly typed input, and a weak `int32` is a different signature from the strong `int32` the first run used. So the second call compiles again, and only from the third call does the cache stop growing.",
              "Chapter 3 owns the cache-key model, its lesson arc prices a weak Python scalar handed in as an argument, and chapter 11 owns the recompile hunt. What is left for here is that a state tree is the easiest place in a training loop to leak a Python scalar into a leaf, and the symptom is small enough to miss: not a compile per step, just two extra compiles and a cache that will not agree with itself.",
            ],
            code: {
              caption: 'continuing the block above (verified, jax 0.4.38 CPU): the same twenty numbers, three executables',
              lang: 'python',
              text: 's = init(jax.random.key(0))._replace(step=0)          # a Python int, not an array\nfor step in range(4):\n    s, _ = train_step(s, data, jax.random.fold_in(base, step))\n\nprint(s.step.dtype, s.step.aval.weak_type, state.step.aval.weak_type)\nprint(train_step._cache_size())\n\n# int32 True False\n# 3',
            },
          },
          {
            h: 'two leaves, one buffer',
            ps: [
              "Two lessons elsewhere own donation and this section stays off both. What `donate_argnums` promises for a single array, what becomes of the array you handed over, and the two mismatches that get a donation refused belong to chapter 11's lesson on it. How a state tree's donation is granted one leaf at a time, and which other names in a loop lose their array when it is, belong to chapter 9's lesson on donating a state tree. Read either one before you add the keyword. What is left over is a failure you can only produce while building an optimizer state, so it belongs here.",
              "Make the zeros once, use them for both Adam moments, and the state looks tidier than the version in the first block. It also has two leaves backed by one buffer, and `unsafe_buffer_pointer` reports that directly.",
              "Nothing goes wrong until the step is donated. Then one execution is asked to hand the same buffer to two donated slots and refuses with `Attempt to donate the same buffer twice`, naming the flattened argument index and the earlier use it collided with. The message comes from XLA rather than from JAX, which is why it reads like a runtime status rather than a Python exception.",
              "The two separate `tree.map` calls of the first block give distinct pointers, and the same donated step then runs and returns step 1. The rule worth carrying past this example: leaves of a donated tree have to be distinct buffers, and sharing that is invisible under normal use turns into a hard failure the moment you promise the runtime it can reuse them.",
            ],
            code: {
              caption: 'continuing the block above (verified, jax 0.4.38 CPU): one zeros tree used twice, the refusal it earns, and the same step donated on a tree built the other way',
              lang: 'python',
              text: 'donating = jax.jit(step_fn, donate_argnums=0)\n\n\ndef init_shared(key):\n    k1, k2 = jax.random.split(key)\n    params = {\n        "w1": jax.random.normal(k1, (8, 32)) * 0.1, "b1": jnp.zeros(32),\n        "w2": jax.random.normal(k2, (32, 1)) * 0.1, "b2": jnp.zeros(1),\n    }\n    zeros = jax.tree.map(jnp.zeros_like, params)     # one tree, used twice\n    return State(params, zeros, zeros, jnp.zeros((), jnp.int32))\n\n\ns = init_shared(jax.random.key(0))\nprint(s.m["w1"].unsafe_buffer_pointer() == s.v["w1"].unsafe_buffer_pointer())\ntry:\n    donating(s, data, jax.random.fold_in(base, 0))\nexcept Exception as err:\n    print(err)\n\nfresh = init(jax.random.key(0))                      # two tree.map calls, two buffers\nprint(fresh.m["w1"].unsafe_buffer_pointer() == fresh.v["w1"].unsafe_buffer_pointer())\nnew_state, _ = donating(fresh, data, jax.random.fold_in(base, 0))\nprint(int(new_state.step))\n\n# True\n# INVALID_ARGUMENT: Attempt to donate the same buffer twice in Execute()\n# (flattened argument 8, replica 0, partition 0, first use: 4). Toy example for\n# this bug: `f(donate(a), donate(a))`.\n# False\n# 1',
            },
          },
        ],
        readings: [
          { label: 'jax.jit reference', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.jit.html', note: 'donate_argnums and donate_argnames, the keyword the last section hands a shared buffer to' },
          { label: 'Buffer donation', url: 'https://docs.jax.dev/en/latest/buffer_donation.html', note: 'what donation is allowed to reuse, and where it is a no-op; the old faq anchor now points here' },
          { label: 'Working with pytrees', url: 'https://docs.jax.dev/en/latest/working-with-pytrees.html', note: 'key paths and flatten order, which is what the printed paths above are showing' },
        ],
        check: [
          {
            q: 'A state tree flattens to thirteen leaves and one of them is not float32. Which one, and what breaks if a saver ignores it?',
            a: 'The step counter, at int32 and shape (). A saver that writes one dtype for the whole tree either corrupts the counter or reloads it as a float, and the counter is what the bias correction and the next batch key both read.',
          },
          {
            q: 'Why does a Python int in the state cost three cache entries rather than two?',
            a: 'The first call compiles for the Python int leaf. It returns an int32 array with weak_type True, which is a second signature, and that call returns a strong int32, which is the third. From the third call on the cache stops growing.',
          },
          {
            q: 'Two Adam moments were built from one zeros tree and donation failed. What did the runtime object to?',
            a: 'Two leaves of the donated tree pointed at one buffer, so one execution was asked to donate the same buffer twice. Building the moments with two separate tree.map calls gives distinct buffers and the donated call runs.',
          },
        ],
        work: [
          { id: 'count-your-tree', label: 'flatten the state tree of a loop you already have, count the leaves, and name the dtype of every leaf that is not float32', href: '#the-whole-run-is-one-value' },
          { id: 'cache-of-one', label: 'drive your own training loop to a cache size of one and write down what the first two extra entries were', href: '#one-step-one-executable' },
        ],
      },
      {
        id: 'three-shapes-of-loop',
        num: 2,
        title: 'Three shapes of loop',
        lede: 'Two hundred steps can be two hundred dispatches, one scan, or one enormous compiled program. All three produce the same numbers to the last bit, and they differ in what the host is allowed to do between steps.',
        goal: 'Given a training loop, choose between a Python loop over a jitted step, a scan, and an unrolled program by naming what each leaves the host able to do between steps, and write the chunked form that keeps a compiled loop and a checkpoint boundary together.',
        sections: [
          {
            h: 'the loop moves into the program',
            ps: [
              "The Python loop from lesson one calls a compiled step two hundred times. Move that same body into `jax.lax.scan` and the loop stops being Python: the state becomes the carry, the step indices become the scanned input, and the per-step losses come back stacked into one `(200,)` array.",
              "The two paths are not merely close. Every loss matches, and every leaf of the final state matches, as `array_equal` over the whole tree. That is worth confirming once yourself rather than assuming, because it is the licence to move a loop into `scan` after it already works.",
              "Chapter 6 states the compile-time property that makes this attractive: `scan` traces its body once whatever the length, while a Python loop written inside `jit` unrolls. Its lesson arc prices the two against each other, in equations and in compile seconds, at three lengths. The rest of this lesson is what a scan takes away from a training run and how the run gets it back.",
            ],
            code: {
              caption: 'run it, continuing lesson one\'s definitions (verified, jax 0.4.38 CPU): the same two hundred steps, both ways',
              lang: 'python',
              text: 'def body(state, step):\n    return step_fn(state, data, jax.random.fold_in(base, step))\n\n\n@jax.jit\ndef run(state, steps):\n    return jax.lax.scan(body, state, steps)\n\n\ndef python_loop(state, n):\n    losses = []\n    for step in range(n):\n        state, loss = train_step(state, data, jax.random.fold_in(base, step))\n        losses.append(loss)\n    return state, jnp.stack(losses)\n\n\nstart = init(jax.random.key(0))\nlooped, l_loop = python_loop(start, 200)\nscanned, l_scan = run(start, jnp.arange(200))\n\nprint(l_scan.shape, f"{l_scan[-1]:.6f}")\nprint(jnp.array_equal(l_loop, l_scan))\nprint(all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(looped), jax.tree.leaves(scanned))))\n\n# (200,) 0.005003\n# True\n# True',
            },
          },
          {
            h: 'one equation instead of two hundred copies',
            ps: [
              "The jaxpr says exactly what happened. One step of this model is 124 equations. The scanned version of two hundred of them is a single equation, whose primitive is `scan` and whose `length` parameter reads 200, with 126 equations in the body it carries.",
              "126 rather than 124 because the scanned body derives its own key from the step index, so it carries a `random_fold_in` and a `convert_element_type` that the standalone step was handed ready-made. The step count lives in a parameter, not in the program's size, which is the difference between a loop the compiler sees as a loop and a loop it sees as a very long straight line.",
              "Reading the outer jaxpr is also how you catch the mistake of scanning a function that is already jitted. Wrap `run` in `make_jaxpr` and the whole thing collapses to one `pjit` equation, because the trace stops at the jit boundary. Trace the unjitted function to see the scan.",
            ],
            code: {
              caption: 'continuing the block above (verified, jax 0.4.38 CPU): the equation counts, from the jaxprs themselves',
              lang: 'python',
              text: 'one = jax.make_jaxpr(step_fn)(start, data, jax.random.fold_in(base, 0))\nmany = jax.make_jaxpr(lambda st, xs: jax.lax.scan(body, st, xs))(start, jnp.arange(200))\n\nprint(len(one.eqns))\nprint(len(many.eqns), many.eqns[0].primitive, many.eqns[0].params["length"])\nprint(len(many.eqns[0].params["jaxpr"].eqns))\n\n# 124\n# 1 scan 200\n# 126',
            },
          },
          {
            h: 'what XLA prints when you unroll it',
            ps: [
              "The third shape is the one people reach for when they want the whole run inside a single `jit`: a Python `for` in the traced function. It is correct, and what it costs is compile time, because the program the compiler receives carries one copy of the body per step. Chapter 6's arc prices that against a scan in equations and in seconds, and those counts are the ones to trust; they are exact where a stopwatch on a laptop is not.",
              "What this loop adds is what the compiler prints while that happens. Compiling the hundred-step unrolled module here tripped XLA's own slow-operation alarm, which prints the module name, offers to help you file a bug, and reports the duration when the compile finally lands. A training script that has ever printed those asterisks unrolled something.",
              "Read the duration as XLA's own output and not as a benchmark. Load average on this laptop was above 500 while it ran, and the claim being made is only that XLA's threshold for calling a compile slow was crossed by a program you could write by accident.",
            ],
            code: {
              caption: 'run it, continuing the block above (verified, jax 0.4.38 CPU): the hundred-step unrolled module, and what XLA printed to stderr while compiling it; the two alarm lines are wrapped to fit and the repeated alarm block at the end is trimmed',
              lang: 'python',
              text: '@jax.jit\ndef unrolled(state):\n    losses = []\n    for step in range(100):\n        state, loss = body(state, step)\n        losses.append(loss)\n    return state, jnp.stack(losses)\n\n\njax.block_until_ready(unrolled(start))\n\n# nothing on stdout, and this on stderr while it compiles:\n# 2026-08-15 06:23:03.016556: E external/xla/xla/service/slow_operation_alarm.cc:73]\n# ********************************\n# [Compiling module jit_unrolled] Very slow compile? If you want to file a bug, run\n# with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.\n# ********************************\n# 2026-08-15 06:24:05.368570: E external/xla/xla/service/slow_operation_alarm.cc:140]\n# The operation took 3m2.369747s',
            },
          },
          {
            h: 'chunks put the host back in the loop',
            ps: [
              "What `scan` takes away is every opportunity the host had between steps. No printing a loss as it arrives, no writing a checkpoint at step 137, no early stop, no adjusting anything from Python. The device runs all two hundred steps and comes back when it is done.",
              "The shape that keeps both is a scan inside a Python loop. Scan fifty steps, return to the host, do whatever the host is for, scan the next fifty. Four chunks of fifty produce the same losses and the same final state as one scan of two hundred, leaf for leaf.",
              "The chunk length is part of the signature, so the four chunks share one executable and the cache holds two entries: one for the length-200 call, one for the length-50 calls. Pick a chunk length and keep it, and the compile happens once no matter how long the run goes.",
              '>> Scan the chunk, checkpoint at the boundary, scan the next.',
            ],
            code: {
              caption: 'continuing the block above (verified, jax 0.4.38 CPU): four chunks of fifty against one scan of two hundred',
              lang: 'python',
              text: 'CHUNK = 50\nstate = init(jax.random.key(0))\nchunks = []\nfor c in range(4):\n    state, losses = run(state, jnp.arange(c * CHUNK, (c + 1) * CHUNK))\n    chunks.append(losses)            # a checkpoint can be written here\n\nprint(jnp.array_equal(jnp.concatenate(chunks), l_scan))\nprint(all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(scanned))))\nprint(run._cache_size())\n\n# True\n# True\n# 2',
            },
          },
          {
            h: 'picking one',
            ps: [
              "The choice is not about speed first. It is about where the host stands. A Python loop over a jitted step leaves the host between every pair of steps, which is what you want while a loop is still being debugged and what makes per-step logging free to write.",
              "A scan removes the host entirely and is the right end state for a step whose body has stopped changing. Unrolling inside one jit gives you neither the host nor a small program, and the alarm above is the reason it is a last resort rather than a default on this loop, with chapter 6's arc supplying the counts behind the general case.",
              "Measuring which of the first two is faster on your own hardware is a job for the harness chapter 11 builds: warm up, loop, block once at the end. The gap between them is dispatch overhead against a per-step compute cost, so it depends on your model size, and a number measured on this eight-by-thirty-two MLP would tell you nothing about yours.",
            ],
            table: {
              caption: 'the three shapes, with the equation counts and cache entries measured above on jax 0.4.38 CPU',
              cols: ['shape', 'compiled program', 'host between steps', 'when it is right'],
              rows: [
                ['Python loop over a jitted step', 'one step, 124 equations, one cache entry', 'yes, every step', 'while the step is still changing, and wherever per-step logging matters'],
                ['scan under jit', 'one scan equation, 126 in the body, length in a parameter', 'no, not until the scan returns', 'a settled step, run in chunks so checkpoints still land'],
                ['Python loop inside one jit', 'one program carrying a copy of the body per step', 'no', 'rarely; XLA raised its own slow-compile alarm at 100 steps here'],
              ],
            },
          },
        ],
        readings: [
          { label: 'jax.lax.scan reference', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html', note: 'the carry and xs contract, and the unroll parameter this lesson leaves at its default' },
          { label: 'Control flow and logical operators', url: 'https://docs.jax.dev/en/latest/control-flow.html', note: 'the official telling of why a Python loop in a traced function unrolls' },
          { label: 'Understanding jaxprs', url: 'https://docs.jax.dev/en/latest/jaxpr.html', note: 'how to read the scan equation and its nested jaxpr parameter' },
        ],
        check: [
          {
            q: 'A scan of 200 steps and a Python loop of 200 jitted steps produced the same losses. What is different about the compiled artifact?',
            a: 'The Python loop compiles one step, 124 equations, and dispatches it 200 times. The scan compiles one equation whose primitive is scan, carrying a 126-equation body and the length 200 as a parameter, and dispatches once.',
          },
          {
            q: 'Why does scanning fifty steps at a time compile only once, however long the run is?',
            a: 'Because the length of the scanned indices is part of the call signature, and every chunk has the same length. Four chunks of fifty share one executable; the cache held two entries only because a length-200 call had been made as well.',
          },
          {
            q: 'What does a scan take away that a Python loop over a jitted step gives you?',
            a: 'Every host opportunity between steps: printing a loss as it arrives, writing a checkpoint mid-run, stopping early. Scanning in chunks and returning to the host at the boundary is how you keep the compiled loop and those opportunities together.',
          },
        ],
        work: [
          { id: 'count-the-equations', label: 'count the jaxpr equations of your own step, then of the same loop scanned and unrolled, and write the three numbers down before you choose a shape', href: '#one-equation-instead-of-two-hundred-copies' },
          { id: 'chunk-a-run', label: 'convert one Python training loop of your own into chunked scans and prove the losses are unchanged with array_equal, not by eye', href: '#chunks-put-the-host-back-in-the-loop' },
        ],
      },
      {
        id: 'kill-it-and-prove-it',
        num: 3,
        title: 'Kill it and prove it',
        lede: 'The chapter sets the bar at bitwise: resume, and every later loss matches the run that was never interrupted. Here is the whole saver, the two things that make the proof possible in JAX, and the two resumes that look correct for exactly one step.',
        goal: 'Serialize and restore a state tree without a checkpointing library, say what a numpy archive does with a typed key and what this loop stores instead, and prove a resume by equality on the loss list and on every leaf of the final state.',
        sections: [
          {
            h: 'thirteen arrays and a step number',
            ps: [
              "A saver for a pytree is short enough to read in one sitting. Flatten with paths, turn each path into a name, and write the leaves into a `.npz`. Restore by flattening a freshly initialized state to recover the structure, then unflattening the loaded arrays back into it.",
              "Using the key path as the archive name is what makes the file survive a change to the model. Position would not: lesson one showed that dict keys flatten sorted, so renaming a parameter reorders the leaves while the count stays the same, and a positional restore would load silently and wrongly.",
              "The archive holds thirteen arrays and the step comes back as `int32` with `weak_type` False, which matters for the reason lesson one measured: restore that counter as a Python int and the resumed run compiles a second and third executable for no reason.",
            ],
            code: {
              caption: 'run it, continuing lesson one\'s definitions (verified, jax 0.4.38 CPU, numpy 2.2.6): the whole saver, and a restore that matches leaf for leaf',
              lang: 'python',
              text: 'import numpy as np\n\n\ndef save(path, state):\n    flat, _ = jax.tree_util.tree_flatten_with_path(state)\n    np.savez(path, **{jax.tree_util.keystr(p): np.asarray(leaf) for p, leaf in flat})\n\n\ndef load(path, like):\n    z = np.load(path)\n    flat, treedef = jax.tree_util.tree_flatten_with_path(like)\n    return jax.tree.unflatten(treedef, [jnp.asarray(z[jax.tree_util.keystr(p)]) for p, _ in flat])\n\n\nstate = init(jax.random.key(0))\nfor step in range(10):\n    state, _ = train_step(state, data, jax.random.fold_in(base, step))\n\nsave("run.npz", state)\nz = np.load("run.npz")\nprint(len(z.files), z[".step"], z[".step"].dtype)\n\nrestored = load("run.npz", init(jax.random.key(0)))\nprint(jax.tree.structure(restored) == jax.tree.structure(state))\nprint(all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored))))\nprint(restored.step.aval.weak_type)\n\n# 13 10 int32\n# True\n# True\n# False',
            },
          },
          {
            h: 'the key numpy will not take',
            ps: [
              "Notice what is not in that archive. There is no key in the state at all, and if there were, `np.asarray` would refuse it: a typed key has dtype `key<fry>`, and the error says so and names the function that gets you out.",
              "So a state that carries a key is still serializable. `key_data` on the way out, `wrap_key_data` on the way in, two extra lines and one decision about which form the archive stores. Chapter 8's arc owns both calls and the comparison this arc leans on: which key discipline survives a restore, and what a resumed run looks like when the wrong one was used.",
              "The reason there is no key here is that the batch key is `fold_in(base, step)`, so the base key is a constant of the program rather than state, and the counter carries the stream. That is what makes the archive thirteen arrays with nothing left to decide, and the next two sections measure what it is worth.",
              '>> Restore the step number and the randomness comes back with it.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, numpy 2.2.6): what a numpy archive does with a typed key, and the one call that changes its mind',
              lang: 'python',
              text: 'import jax\nimport numpy as np\n\nkey = jax.random.key(7)\ntry:\n    np.asarray(key)\nexcept TypeError as err:\n    print(err)\n\nnp.savez("key.npz", key=jax.random.key_data(key))\nwords = np.load("key.npz")["key"]\nprint(words.dtype, words.shape)\n\n# JAX array with PRNGKey dtype cannot be converted to a NumPy array. Use\n# jax.random.key_data(arr) if you wish to extract the underlying integer array.\n# uint32 (2,)',
            },
          },
          {
            h: 'the proof is an equality, not a curve',
            ps: [
              "Twenty steps straight through. The same run killed at ten, saved, restored into objects built from scratch, and continued. The claim is that the last ten losses are the same list, and the check is `==` on the list, not a plot.",
              "They match to six decimals as printed, and the whole final state matches leaf for leaf as `array_equal`. Both halves are worth running: the losses agreeing tells you the forward pass lined up, and the state agreeing tells you the moments and the counter did too, which is what the eleventh step depends on.",
              "The PyTorch path's own resume lesson runs the same experiment against a global generator and a scheduler, and LAB·P4 draws the overlaid curves on a TPU bridge. What is different here is how little the checkpoint has to hold: no generator block, no scheduler object, thirteen arrays and a step number, because the randomness is a function of the step and the learning rate is a constant in the program.",
            ],
            code: {
              caption: 'continuing the first block of this lesson (verified, jax 0.4.38 CPU): ten resumed losses against the run that never stopped',
              lang: 'python',
              text: 'def run_from(state, start, n):\n    losses = []\n    for step in range(start, start + n):\n        state, loss = train_step(state, data, jax.random.fold_in(base, step))\n        losses.append(round(float(loss), 6))\n    return state, losses\n\n\nstraight, uninterrupted = run_from(init(jax.random.key(0)), 0, 20)\nafter, resumed = run_from(restored, 10, 10)\n\nprint(uninterrupted[10:])\nprint(resumed)\nprint(resumed == uninterrupted[10:])\nprint(all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(straight), jax.tree.leaves(after))))\n\n# [5.617665, 7.358051, 6.792553, 5.592591, 2.720668, 3.390179, 4.089592,\n#  3.357401, 3.01731, 1.720861]\n# [5.617665, 7.358051, 6.792553, 5.592591, 2.720668, 3.390179, 4.089592,\n#  3.357401, 3.01731, 1.720861]\n# True\n# True',
            },
          },
          {
            h: 'two resumes that look fine for one step',
            ps: [
              "Break the resume two ways and watch how long each one takes to show. Restore the state correctly but restart the step index at zero, and the first resumed loss is 8.487146 against the correct 5.617665, because a different 32 rows were drawn. That failure is loud and lands immediately.",
              "The second one shows up a step later. Restore the parameters and the counter, zero the two Adam moments, and the first resumed loss is 5.617665, exactly right. The loss is measured before the update, so missing moments cannot change the number until the step after. Step twelve reads 7.644585 where the uninterrupted run had 7.358051.",
              "One correct step is enough to fool a check that only looks at the first number after a restore, and this is a small model on synthetic data where the gap opens fast. On a real run the divergence is a slightly worse curve that nobody attributes to the restore. Compare the list, not the first element.",
            ],
            code: {
              caption: 'continuing the block above (verified, jax 0.4.38 CPU): the first three resumed losses under two broken restores',
              lang: 'python',
              text: 'zeros = jax.tree.map(jnp.zeros_like, restored.params)\n\n_, from_zero = run_from(restored, 0, 10)\n_, no_moments = run_from(restored._replace(m=zeros, v=zeros), 10, 10)\n\nprint(uninterrupted[10:13])\nprint(from_zero[:3])\nprint(no_moments[:3])\n\n# [5.617665, 7.358051, 6.792553]\n# [8.487146, 5.50882, 5.776051]\n# [5.617665, 7.644585, 7.207603]',
            },
            table: {
              caption: 'three resumes from the same step-10 checkpoint, first three losses each (verified, jax 0.4.38 CPU)',
              cols: ['restore', 'step 11', 'step 12', 'step 13'],
              rows: [
                ['the whole tree, resumed at step 10', '5.617665', '7.358051', '6.792553'],
                ['the whole tree, step index restarted at 0', '8.487146', '5.50882', '5.776051'],
                ['params and step only, moments zeroed', '5.617665', '7.644585', '7.207603'],
              ],
            },
          },
          {
            h: 'what a checkpointing library is for',
            ps: [
              "The eight-line saver above is enough to prove a resume and not enough to run one. It writes the whole file synchronously, so a training step waits on the disk. It writes in place, so a process killed mid-write leaves an archive that loads and is wrong. And it has no notion of a mesh, so a sharded run has nothing to restore onto.",
              "Orbax is the library the chapter names for those three problems: asynchronous saves so the step does not block, atomic directory commits so a partial write is never mistaken for a checkpoint, and sharding-aware restore so a tree written from one device layout comes back onto another. The tree is the same tree either way, which is why the saver above is a good thing to have written once before adopting a library that hides it.",
              "The bar does not change when the library does. Kill the run, restore, continue, and compare the list. A checkpoint that exists is not evidence; a resumed loss list equal to the uninterrupted one is.",
            ],
          },
        ],
        readings: [
          { label: 'Orbax checkpointing', url: 'https://orbax.readthedocs.io/en/latest/', note: 'the async save, the atomic commit, and the sharding-aware restore this lesson does without' },
          { label: 'Pseudorandom numbers in JAX', url: 'https://docs.jax.dev/en/latest/random-numbers.html', note: 'key, key_data and fold_in, the three calls the resume proof rests on' },
          { label: 'jax.tree_util reference', url: 'https://docs.jax.dev/en/latest/jax.tree_util.html', note: 'tree_flatten_with_path and keystr, which is what makes the archive names survive a rename' },
        ],
        check: [
          {
            q: 'Why does the archive use key paths as names rather than leaf positions?',
            a: 'Because flatten order sorts dict keys, so renaming a parameter reorders the leaves without changing their count. A positional restore would load without complaint and put the wrong array in each slot; a path-named restore fails loudly or matches correctly.',
          },
          {
            q: 'A resume restored the params and the step but zeroed Adam\'s moments, and the first resumed loss was exactly right. Why, and when does it go wrong?',
            a: 'The loss is measured before the update, so the missing moments cannot affect it until the following step. Here step 11 read 5.617665 on both runs and step 12 read 7.644585 against the correct 7.358051.',
          },
          {
            q: 'The archive holds thirteen arrays and no random key. What about this loop makes that possible, and what would a saver need if a key were in the state?',
            a: 'Every batch is drawn from fold_in(base, step), so the base key is a constant of the program rather than state and the counter carries the stream. A state that did carry a key would need key_data before np.savez accepts it and wrap_key_data on the way back, and chapter 8 owns both calls and the comparison of the two disciplines.',
          },
        ],
        work: [
          { id: 'save-your-tree', label: 'write the path-keyed saver for your own state tree and check that a restore matches every leaf with array_equal before you trust it', href: '#thirteen-arrays-and-a-step-number' },
          { id: 'prove-the-resume', label: 'kill one of your own runs at a checkpoint, resume, and compare the loss lists as lists; then break one piece of the restore and record how many steps pass before it shows', href: '#two-resumes-that-look-fine-for-one-step' },
        ],
      },
    ],
  },
]
