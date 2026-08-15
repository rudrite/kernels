// New file: site/src/data/lessons/jax-jit.ts
// The jit cache, measured rather than described: two counters that disagree,
// what a static value costs in milliseconds, and the four ahead-of-time
// stages. Chapter 03 names the four parts of the key and LAB·J2 drives them
// with jax_log_compiles; nothing here repeats either, and donate_argnums stays
// under chapter 11, where chapter 1 sends it. Every printed value ran on this
// machine (jax 0.4.38, CPU, Python 3.12, 2026-08-15). Timings are single-run
// medians on one laptop and vary; every ratio and every exact integer
// reproduces.
import type { UnitLessons } from './index'

export const JAX_JIT_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:jit',
    lessons: [
      {
        id: 'two-counters',
        num: 1,
        title: 'Two counters, one cache',
        lede: 'You do not have to infer a recompile from a log line. A side effect inside the function counts traces, jit counts its own dispatch entries, and when those two numbers move apart they tell you something a single counter cannot.',
        goal: 'Instrument a jitted function with a trace counter and _cache_size, then predict which of shape, dtype, weak type, pytree structure and function identity moves which counter, and by how much.',
        sections: [
          {
            h: 'the counter that only moves at trace time',
            ps: [
              "Chapter 02 established that a Python side effect inside a jitted function fires once, at trace time, and never again. Read that as a measuring instrument rather than a warning and you get the cheapest recompile detector there is: a `global` counter incremented in the function body ticks exactly once per trace, so its value is the number of times JAX has walked your Python.",
              "The second number comes from the jit object itself. `f._cache_size()` reports how many entries the dispatch cache holds for the function `f` wraps. The leading underscore is a real caveat and worth saying out loud: it is not part of the promised API, and a future release may rename it. As an instrument in a lesson it is fine, and nothing else in the public surface answers the same question.",
              "Run the two together and the first three lines are unsurprising. The same argument twice moves neither counter, which is the cache doing its job.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): five calls, two counters',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nimport numpy as np\n\ntraces = 0\n\n@jax.jit\ndef f(x, s):\n    global traces\n    traces += 1          # a trace-time side effect: one tick per trace\n    return x * s\n\nx = jnp.ones(4)\n\ndef call(label, s):\n    f(x, s)\n    print(f"{label:<16} traces={traces} entries={f._cache_size()}")\n\ncall("jnp float32", jnp.float32(2.0))\ncall("the same again", jnp.float32(2.0))\ncall("numpy float32", np.float32(2.0))\ncall("python float", 2.0)\ncall("python int", 2)\n\n# jnp float32      traces=1 entries=1\n# the same again   traces=1 entries=1\n# numpy float32    traces=1 entries=2\n# python float     traces=2 entries=3\n# python int       traces=3 entries=4',
            },
          },
          {
            h: 'two counters, and they disagree',
            ps: [
              "Line three is where the instrument earns its place. Swapping `jnp.float32(2.0)` for `np.float32(2.0)` adds a cache entry and does not add a trace. Nothing about the program changed, because both scalars abstract to the same shape and dtype, so JAX reused the jaxpr it already had. What it did not reuse was the fast dispatch path, which is keyed more finely than the trace is, and a NumPy scalar reaches it as a different kind of Python object than a JAX array does.",
              "Lines four and five move both counters, and chapter 01's weak scalars are why. A Python `2.0` abstracts to `float32[]` with `weak_type=True`, a Python `2` to `int32[]` with `weak_type=True`, and each of those is a different abstract value from the strong `float32[]` the first three calls produced. A different abstract value is a different program, so it is a real retrace.",
              "The practical reading is a rule about argument hygiene rather than about dtypes. Passing a hyperparameter as a bare Python number, and sometimes as an array, and sometimes as a NumPy scalar, spends cache entries on a value that never changed. Pick one form at the call site and the counters flatten out.",
              '>> Two numbers that move apart are telling you which layer noticed.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same 2, four ways, and the aval each one gets',
              lang: 'python',
              text: 'import jax.numpy as jnp\nimport numpy as np\n\nfor label, v in [("jnp.float32(2.0)", jnp.float32(2.0)),\n                 ("np.float32(2.0)", np.float32(2.0)),\n                 ("2.0", 2.0),\n                 ("2", 2)]:\n    a = jnp.asarray(v)\n    print(f"{label:<17} {a.dtype} weak_type={a.weak_type}")\n\n# jnp.float32(2.0)  float32 weak_type=False\n# np.float32(2.0)   float32 weak_type=False\n# 2.0               float32 weak_type=True\n# 2                 int32 weak_type=True',
            },
          },
          {
            h: 'structure counts, key order does not',
            ps: [
              "Pytree structure is the part of the key people most often guess wrong about, in both directions. Handing the same two arrays as a dict whose keys are written in a different order costs nothing, because flattening sorts dict keys before it builds the treedef, so `{\"b\": x, \"a\": x}` and `{\"a\": x, \"b\": x}` produce the same `PyTreeDef` and land on the same entry.",
              "Handing them as a list instead of a tuple costs a full retrace. `PyTreeDef((*, *))` and `PyTreeDef([*, *])` are different structures, and JAX makes no attempt to treat one container as the other. The counter proves it in one line, and printing the treedef shows you exactly what the cache compared.",
              "That asymmetry is worth carrying into real code. A data pipeline that yields a list on one step and a tuple on the next, or a config that arrives as a dict here and a `NamedTuple` there, is a recompile per shape of container, and none of it shows up as a change in shapes or dtypes.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): four containers, three traces',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ntraces = 0\n\n@jax.jit\ndef g(t):\n    global traces\n    traces += 1\n    if isinstance(t, dict):\n        return t["a"] + t["b"]\n    return t[0] + t[1]\n\nx = jnp.ones(3)\ng({"a": x, "b": x}); print("dict a,b ", traces, g._cache_size())\ng({"b": x, "a": x}); print("dict b,a ", traces, g._cache_size())\ng((x, x));           print("tuple    ", traces, g._cache_size())\ng([x, x]);           print("list     ", traces, g._cache_size())\n\nprint(jax.tree_util.tree_structure({"b": x, "a": x}))\nprint(jax.tree_util.tree_structure((x, x)))\nprint(jax.tree_util.tree_structure([x, x]))\n\n# dict a,b  1 1\n# dict b,a  1 1\n# tuple     2 2\n# list      3 3\n# PyTreeDef({\'a\': *, \'b\': *})\n# PyTreeDef((*, *))\n# PyTreeDef([*, *])',
            },
          },
          {
            h: 'the wrapper is cheap, the function is not',
            ps: [
              "The advice you usually hear is to hoist `jax.jit(f)` out of the loop, and the counters say that advice is aimed at the wrong object. Build a fresh `jax.jit` wrapper around the same named function on every iteration and the calls stay at cache-hit speed, because the key holds the function jit wraps, not the wrapper. `jax.jit(step)._cache_size()` still reports one entry after all of it.",
              "Build a fresh function on every iteration and the cost shows up on the first call. A `lambda` written inside a loop, or an inner function returned by a factory, is a new function object each time round, so it is a new key, so it traces and compiles from scratch. In the run below that is 0.4 milliseconds against 83.3, on a computation small enough that the arithmetic itself is a rounding error, and on a busier machine the gap widens rather than closes.",
              "The factory pattern is the version that shows up in real training code, because it reads as configuration rather than as churn. `make_step(lr)` called once per epoch to bake in a new learning rate hands jit a new function every epoch, and every one of them compiles. Building it once and passing the learning rate as an argument costs one compile in total.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one run\'s medians of five; the milliseconds move with machine and load, the gap of two orders of magnitude or more does not',
              lang: 'python',
              text: 'import statistics as st\nimport time\n\nimport jax\nimport jax.numpy as jnp\n\nx = jnp.ones((512, 512))\n\ndef step(v):\n    return v * 2.0 + 1.0\n\ndef make_step(scale):\n    def inner(v):\n        return v * scale + 1.0\n    return inner\n\ndef median_ms(call, n=5):\n    ts = []\n    for _ in range(n):\n        t0 = time.perf_counter()\n        call()\n        ts.append(1e3 * (time.perf_counter() - t0))\n    return round(st.median(ts), 2)\n\njax.jit(step)(x).block_until_ready()          # compile once, for real\njax.jit(make_step(2.0))(x).block_until_ready()\n\nprint("a new wrapper over one function", median_ms(lambda: jax.jit(step)(x).block_until_ready()), "ms")\nprint("a new closure from a factory  ", median_ms(lambda: jax.jit(make_step(2.0))(x).block_until_ready()), "ms")\nprint("entries under step:", jax.jit(step)._cache_size())\n\n# a new wrapper over one function 0.4 ms\n# a new closure from a factory   83.3 ms\n# entries under step: 1',
            },
          },
        ],
        readings: [
          { label: 'jax.jit', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.jit.html', note: 'the parameter list the cache key is built from, argument by argument' },
          { label: 'Type promotion semantics', url: 'https://docs.jax.dev/en/latest/type_promotion.html', note: 'where weak types are defined, and why a Python scalar is not a float32 array' },
          { label: 'Pytrees', url: 'https://docs.jax.dev/en/latest/pytrees.html', note: 'the flattening rules, including what happens to dict keys on the way in' },
        ],
        check: [
          {
            q: 'A call adds a cache entry but the trace counter does not move. What changed?',
            a: 'Something the dispatch path keys on that abstraction throws away, such as passing np.float32(2.0) where the last call passed jnp.float32(2.0). Both abstract to a strong float32 scalar, so the jaxpr was reused and only the fast path needed a new entry.',
          },
          {
            q: 'Why does reordering a dict of arguments cost nothing while swapping a tuple for a list costs a retrace?',
            a: 'Flattening sorts dict keys, so both orderings build the same PyTreeDef and hit the same entry. A tuple and a list build different PyTreeDefs, and pytree structure is part of the key, so the second one traces and compiles from scratch.',
          },
          {
            q: 'You rebuild jax.jit(step) inside your training loop. How much does that cost, and what would cost a lot?',
            a: 'Close to nothing: the key holds the wrapped function, so every wrapper over the same step object shares one entry. Building a fresh function each iteration, from a lambda or a closure factory, is a new key and a full trace and compile every time.',
          },
        ],
        work: [
          { id: 'counter-kit', label: 'put a trace counter and _cache_size on one function of your own, then call it five ways and predict both numbers before each call', href: '#the-counter-that-only-moves-at-trace-time' },
          { id: 'hunt-the-factory', label: 'search your code for a jitted function built inside a loop or returned from a factory, and count the compiles it costs per epoch', href: '#the-wrapper-is-cheap-the-function-is-not' },
        ],
      },
      {
        id: 'the-price-of-static',
        num: 2,
        title: 'The price of a static argument',
        lede: 'Marking an argument static removes a parameter from the compiled signature and plants a constant in its place. Both halves of that trade are measurable, and for a plain scalar the measurement is unkind.',
        goal: 'Read a static value off a lowered signature, put a millisecond price on one extra cache entry, state the hash-and-equality contract a static argument must satisfy, and decide from evidence whether an argument earns being static.',
        sections: [
          {
            h: 'the parameter that turns into a constant',
            ps: [
              "The clearest picture of what `static_argnums` does is the lowered program, before any of the caching argument comes up. Lower the same two-argument function twice, once plain and once with the second argument static, and the signatures do not match: the traced version takes `tensor<4xf32>` and `tensor<f32>`, the static version takes `tensor<4xf32>` alone and carries `stablehlo.constant dense<2.000000e+00>` in the body.",
              "That is the whole mechanism, stated in the artifact rather than in prose. A static argument never becomes a parameter. Its value is read at trace time and written into the program, exactly the way a closed-over Python constant is, which is why chapter 03 describes it as baked in rather than passed in.",
              "It also explains the shape of the cost. A constant in the program means the program is only valid for that constant, so a second value needs a second program, and the cache has no choice about it.",
            ],
            code: {
              caption: 'verbatim StableHLO, jax 0.4.38 CPU: two lowerings of one function, joined here under added comment headings',
              lang: 'mlir',
              text: '// jax.jit(scale).lower(jnp.ones(4), 2.0).as_text()\nmodule @jit_scale attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n  func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> (tensor<4xf32> {jax.result_info = ""}) {\n    %0 = stablehlo.convert %arg1 : tensor<f32>\n    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>\n    %2 = stablehlo.multiply %arg0, %1 : tensor<4xf32>\n    return %2 : tensor<4xf32>\n  }\n}\n\n// jax.jit(scale, static_argnums=1).lower(jnp.ones(4), 2.0).as_text()\nmodule @jit_scale attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32> {jax.result_info = ""}) {\n    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>\n    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>\n    %1 = stablehlo.multiply %arg0, %0 : tensor<4xf32>\n    return %1 : tensor<4xf32>\n  }\n}',
            },
          },
          {
            h: 'what that constant is worth, in milliseconds',
            ps: [
              "A folded constant sounds like free speed, so it is worth asking what it actually bought. Take a 2048 by 2048 elementwise scale-and-add, compile it both ways, and time the steady state. On this machine the traced version ran at 5.33 milliseconds and the static version at 5.40, which is to say the two are the same number wearing different error bars. XLA broadcasts a scalar parameter about as cheaply as it broadcasts a scalar constant, and there was nothing else in the program for the constant to unlock.",
              "The other side of the ledger is not within noise at all. The first call at a new static value cost 139.22 milliseconds against a 5.33 millisecond steady call, so one new value costs roughly twenty-six calls of debt and returns nothing measurable on this program. Converting a compile into steady steps is chapter 11's habit, taught on a six-layer MLP in its lesson on what the clock caught; it is borrowed here to price a single keyword.",
              "So the decision is not about whether specialization is good in general. It is arithmetic over two numbers: how many distinct values this argument takes across a run, and what one compile costs at this shape. LAB·J2 counts the values for you with the compile log. What this lesson adds is the other factor, measured for the case where the answer comes out against static: a scalar that changes no shape.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one run\'s medians of nine, plus one cold call; a loaded machine inflates the cold compile most',
              lang: 'python',
              text: 'import statistics as st\nimport time\n\nimport jax\nimport jax.numpy as jnp\n\nx = jnp.ones((2048, 2048))\n\ndef scale_a(x, s):\n    return x * s + 1.0\n\ndef scale_b(x, s):\n    return x * s + 1.0\n\ntraced = jax.jit(scale_a)\nstatic = jax.jit(scale_b, static_argnums=1)\n\ndef median_ms(call, n=9):\n    ts = []\n    for _ in range(n):\n        t0 = time.perf_counter()\n        call().block_until_ready()\n        ts.append(1e3 * (time.perf_counter() - t0))\n    return round(st.median(ts), 2)\n\ntraced(x, 2.0).block_until_ready()\nstatic(x, 2.0).block_until_ready()\nprint("s traced, steady state", median_ms(lambda: traced(x, 2.0)), "ms")\nprint("s static, steady state", median_ms(lambda: static(x, 2.0)), "ms")\n\nt0 = time.perf_counter()\nstatic(x, 3.0).block_until_ready()\nprint("s static, one new value", round(1e3 * (time.perf_counter() - t0), 2), "ms")\nprint("entries: traced", traced._cache_size(), "static", static._cache_size())\n\n# s traced, steady state 5.33 ms\n# s static, steady state 5.4 ms\n# s static, one new value 139.22 ms\n# entries: traced 1 static 2',
            },
          },
          {
            h: 'hash and eq are the real contract',
            ps: [
              "A static value has to hash. The museum files the error you get when it does not, under a list handed to `static_argnames`, so that half of the contract already has a home. The other half fails silently instead of raising, which is why it is worth a section here.",
              "Two static values land on the same cache entry when they are equal, and a plain Python object without `__eq__` compares by identity. So a small config object, constructed fresh at each call site with the same contents, is a different key every time. Three equal-looking configs produce three cache entries and three compiles, and nothing anywhere prints a warning.",
              "Give the class `__hash__` and `__eq__` by value and the same three calls collapse to one entry. That is the fix for the config-object pattern generally: a frozen dataclass, a `NamedTuple`, or an explicit pair of dunders. The compiler is comparing your keys with `==`, so whatever `==` means for your type is what the cache means by the same program.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): three equal configs, three compiles or one',
              lang: 'python',
              text: 'from functools import partial\n\nimport jax\nimport jax.numpy as jnp\n\nx = jnp.ones(4)\n\nclass Cfg:\n    def __init__(self, n):\n        self.n = n\n\nclass KeyedCfg:\n    def __init__(self, n):\n        self.n = n\n    def __hash__(self):\n        return hash(self.n)\n    def __eq__(self, other):\n        return isinstance(other, KeyedCfg) and self.n == other.n\n\n@partial(jax.jit, static_argnums=1)\ndef loose(x, cfg):\n    return x * cfg.n\n\n@partial(jax.jit, static_argnums=1)\ndef keyed(x, cfg):\n    return x * cfg.n\n\nfor _ in range(3):\n    loose(x, Cfg(2))\n    keyed(x, KeyedCfg(2))\n\nprint("three equal Cfg objects     ", loose._cache_size())\nprint("three equal KeyedCfg objects", keyed._cache_size())\n\n# three equal Cfg objects      3\n# three equal KeyedCfg objects 1',
            },
          },
          {
            h: 'when static is the only answer',
            ps: [
              "None of the above says avoid `static_argnums`. It says spend it where a traced argument cannot do the job at all, which is any argument the output shape depends on. A repeat count, a number of heads, a window size, a boolean that selects between two differently shaped returns: each of those has to be known while the trace is running, because the trace is where shapes get fixed.",
              "For those arguments the cost analysis inverts. The cardinality is small and bounded by the model rather than by the data, so a handful of compiles buys programs that could not otherwise exist. A per-example sequence length, by contrast, is bounded by the data, and marking it static is the churn bug the chapter warns about.",
              "There is a third option that is easy to forget between the two. An argument whose value is genuinely fixed for the life of the program does not need to be an argument: close over it, and it is baked in at trace time for free, with no cache key component to keep track of and no way for a caller to accidentally vary it.",
            ],
            table: {
              caption: 'the decision, from the measurements above (jax 0.4.38 CPU) and from the shape rules chapter 03 states',
              cols: ['the argument', 'traced or static', 'why'],
              rows: [
                ['a learning rate, a loss scale', 'traced', 'no shape depends on it, and the folded constant measured no faster at 2048 by 2048'],
                ['a repeat count, a head count, a window size', 'static', 'the output shape needs it at trace time; cardinality is bounded by the model'],
                ['a per-example sequence length', 'neither, pad or bucket it', 'static gives one executable per length seen; traced cannot fix the shape at all'],
                ['a config object built per call site', 'static, with __eq__ and __hash__', 'identity comparison gave 3 entries for 3 equal configs, value comparison gave 1'],
                ['a value that never changes after startup', 'close over it', 'baked in at trace time with no key component and no way for a caller to vary it'],
              ],
            },
          },
        ],
        readings: [
          { label: 'jit compilation', url: 'https://docs.jax.dev/en/latest/jit-compilation.html', note: 'the official tutorial, including the static-argument section this lesson prices' },
          { label: 'jax.jit', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.jit.html', note: 'static_argnums and static_argnames, and the hashability requirement stated in the docstring' },
          { label: 'Ahead-of-time compilation', url: 'https://docs.jax.dev/en/latest/aot.html', note: 'lower and as_text, the two calls that produced the signatures above' },
        ],
        check: [
          {
            q: 'Looking at two lowered signatures, how do you tell which one had a static argument?',
            a: 'Count the parameters. The static version is one parameter shorter and carries the value as a stablehlo.constant in the body, because a static argument is read at trace time and written into the program rather than passed to it.',
          },
          {
            q: 'Three calls pass configs that hold identical values, and the cache grows to three entries. What is wrong and how do you fix it?',
            a: 'The config class has no __eq__, so the cache compares the objects by identity and three fresh instances are three distinct keys. Define __eq__ and __hash__ by value, or use a frozen dataclass or a NamedTuple, and the three calls collapse to one entry.',
          },
          {
            q: 'When is a folded constant worth a compile per distinct value?',
            a: 'When the output shape depends on the argument, so a traced version cannot be built at all, and the number of distinct values is bounded by the model rather than the data. For a plain scalar that changes no shape, the steady state measured the same either way, at 5.33 against 5.40 milliseconds, while one new value cost 139.22.',
          },
        ],
        work: [
          { id: 'signature-diff', label: 'lower one of your own functions twice, plain and with an argument static, and diff the two @main signatures', href: '#the-parameter-that-turns-into-a-constant' },
          { id: 'price-a-compile', label: 'measure one cold compile and one steady-state call at your real shapes, then divide: that is how many calls a new static value costs you', href: '#what-that-constant-is-worth-in-milliseconds' },
        ],
      },
      {
        id: 'before-the-first-call',
        num: 3,
        title: 'Before the first call',
        lede: 'Tracing, lowering and compiling are three separate stages with three separate artifacts, and jit only hides them. Take them apart and most of the wait turns out to sit in one place.',
        goal: 'Drive eval_shape, trace, lower and compile by hand, say which question each stage\'s artifact can answer, name where the time goes, and explain how an ahead-of-time Compiled object behaves differently from the jitted function it came from.',
        sections: [
          {
            h: 'four stops, and where the time actually goes',
            ps: [
              "Chapter 03 introduces `lower` and `compile` as a way to see a program before running it. There are four stops on that road, not two, and running them separately tells you something the combined call cannot: which stage you are actually waiting on.",
              "For a small MLP forward pass at 128 by 512 by 128, the answer on this machine is not close. Shape inference took 3.83 milliseconds, tracing 2.22, lowering to StableHLO 47.94, and XLA compiling that module 205.29. Everything upstream of the compiler is a rounding error against the compiler.",
              "That ratio is what makes shape work cheap and worth doing eagerly. `f.eval_shape(*specs)` runs the abstract evaluation and hands back the output's shape and dtype without producing a program at all, which makes it the right tool for validating that a model's shapes line up across a config sweep. Nothing gets compiled, so nothing costs 205 milliseconds.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one run after two warmups; the milliseconds move with machine and load, the ordering held in every run',
              lang: 'python',
              text: 'import time\n\nimport jax\nimport jax.numpy as jnp\n\nspec = lambda *s: jax.ShapeDtypeStruct(s, jnp.float32)\nargs = (spec(128, 512), spec(512, 512), spec(512, 128))\n\ndef make(bias):\n    def net(x, w1, w2):\n        h = jnp.tanh(x @ w1 + bias)\n        return jnp.sum(h @ w2)\n    return jax.jit(net)\n\nfor warm in (0.0, 1.0):                    # warm every stage, then throw it away\n    make(warm).trace(*args).lower().compile()\n\ndef ms(call):\n    t0 = time.perf_counter()\n    out = call()\n    return round(1e3 * (time.perf_counter() - t0), 2), out\n\nt_eval, shape = ms(lambda: make(2.0).eval_shape(*args))\nf = make(3.0)\nt_trace, traced = ms(lambda: f.trace(*args))\nt_lower, lowered = ms(lambda: traced.lower())\nt_compile, compiled = ms(lambda: lowered.compile())\nt_again, _ = ms(lambda: lowered.compile())\n\nprint("eval_shape   ", t_eval, "ms ->", shape)\nprint("trace        ", t_trace, "ms ->", type(traced).__name__)\nprint("lower        ", t_lower, "ms ->", type(lowered).__name__)\nprint("compile      ", t_compile, "ms ->", type(compiled).__name__)\nprint("compile again", t_again, "ms, the same Lowered")\n\n# eval_shape    3.83 ms -> ShapeDtypeStruct(shape=(), dtype=float32)\n# trace         2.22 ms -> Traced\n# lower         47.94 ms -> Lowered\n# compile       205.29 ms -> Compiled\n# compile again 0.02 ms, the same Lowered',
            },
          },
          {
            h: 'reading a program you have not run',
            ps: [
              "Each stage hands back an object, and each object answers a different class of question. `Traced` carries the jaxpr, so it answers what JAX recorded. `Lowered` carries the StableHLO and can be printed with `as_text`, so it answers what the compiler was handed. `Compiled` carries the executable, so it answers what the compiler decided, in bytes and in flops.",
              "One of those answers this arc has already used: the signature diff in lesson two came off a `Lowered`. The other two are read elsewhere. LAB·J2 takes its flop count off a `Compiled`, and so does the aliased-byte count that proves a buffer donation, in chapter 11's lesson on buffers you promise away.",
              "The one caution worth carrying is the one the AOT page states about itself: these analysis results are diagnostics, not a stable interface, and their type and contents may differ across backends and versions. Treat a number from `memory_analysis` or `cost_analysis` as evidence about this build on this backend, not as a documented value you can assert against.",
            ],
            table: {
              caption: 'the four stages, their artifacts and their measured cost at 128x512x128 (jax 0.4.38 CPU, one run after two warmups)',
              cols: ['stage', 'what it returns', 'what you can ask it', 'cost'],
              rows: [
                ['f.eval_shape(*specs)', 'ShapeDtypeStruct', 'do the shapes line up, and what comes out', '3.83 ms'],
                ['f.trace(*specs)', 'Traced', 'the jaxpr, in_avals, in_tree, out_info', '2.22 ms'],
                ['.lower()', 'Lowered', 'as_text, the StableHLO the compiler receives', '47.94 ms'],
                ['.compile()', 'Compiled', 'memory_analysis, cost_analysis, the executable', '205.29 ms'],
                ['.compile() again', 'Compiled', 'the same answers, from the compilation cache', '0.02 ms'],
              ],
            },
          },
          {
            h: 'a second cache underneath the first',
            ps: [
              "The last row of that table is the surprising one. Calling `.compile()` twice on the same `Lowered` object cost 205.29 milliseconds and then 0.02, so something below jit is caching compiled modules keyed on the module itself, not on the Python function that produced it.",
              "That is a different cache from the one lesson one instrumented, and it explains a pattern that otherwise looks like magic: two unrelated functions that happen to lower to identical StableHLO pay for one compile between them. It also means a benchmark that compiles the same module twice and reports the second number is measuring the cache, not the compiler.",
              "In a single process this cache is a convenience. Across processes it is configuration, since JAX also ships a persistent compilation cache that writes compiled modules to a directory and reads them back on the next run. That one is off unless you turn it on, and it is the difference between paying compile time once per run and once ever.",
            ],
          },
          {
            h: 'the compiled object refuses instead of retracing',
            ps: [
              "A `Compiled` is not a jitted function, and the difference shows up the moment you call it with something it was not compiled for. A jitted function meeting a new shape traces and compiles a new entry, silently; a `Compiled` meeting a new shape raises.",
              "The message is specific about both sides of the mismatch, naming the argument, the type it was compiled with, and the type it was called with. That is the behaviour you want at a serving boundary, where a silent recompile at a request boundary is a latency spike nobody asked for.",
              "Read that against lesson one and the arc closes. jit's cache is a convenience that hides a decision, and the decision is which of four things changed. The ahead-of-time surface removes the hiding: you choose the signature, you pay for the compile when you choose to, and anything that does not match gets an error instead of a recompile.",
              '>> jit recompiles quietly. A compiled program refuses out loud.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one compiled program, two arguments it will not take',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nf = jax.jit(lambda x: (x @ x).sum())\ncompiled = f.lower(jax.ShapeDtypeStruct((256, 256), jnp.float32)).compile()\n\nprint(compiled(jnp.ones((256, 256))))\nfor arg in (jnp.ones((128, 128)), jnp.ones((256, 256), jnp.bfloat16)):\n    try:\n        compiled(arg)\n    except TypeError as e:\n        print(f"{type(e).__name__}: {e}")\n\n# 16777216.0\n# TypeError: Argument types differ from the types for which this computation was compiled. The mismatches are:\n# Argument \'x\' compiled with float32[256,256] and called with float32[128,128]\n# TypeError: Argument types differ from the types for which this computation was compiled. The mismatches are:\n# Argument \'x\' compiled with float32[256,256] and called with bfloat16[256,256]',
            },
          },
        ],
        readings: [
          { label: 'Ahead-of-time compilation', url: 'https://docs.jax.dev/en/latest/aot.html', note: 'the four stages and the stability caveat on every analysis result' },
          { label: 'jax.stages', url: 'https://docs.jax.dev/en/latest/jax.stages.html', note: 'the Traced, Lowered and Compiled classes, attribute by attribute' },
          { label: 'jax.eval_shape', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html', note: 'shape inference with no tracing artifact and no compile' },
          { label: 'Persistent compilation cache', url: 'https://docs.jax.dev/en/latest/persistent_compilation_cache.html', note: 'the across-process version of the cache the last row of the table shows' },
        ],
        check: [
          {
            q: 'Of the four stages, which one dominates, and what does that make cheap?',
            a: 'Compiling: 205.29 ms against 3.83 for eval_shape, 2.22 for trace and 47.94 for lower on a small MLP. That makes shape checking with eval_shape essentially free, so validating a config sweep\'s shapes need not compile anything.',
          },
          {
            q: 'Calling compile() twice on one Lowered took 205.29 ms then 0.02 ms. What does that tell you?',
            a: 'There is a compilation cache below jit keyed on the module rather than on the Python function, so two functions lowering to identical StableHLO share one compile. It also means a benchmark that compiles the same module twice is timing the cache.',
          },
          {
            q: 'What does an ahead-of-time Compiled object do when handed the wrong dtype, and why prefer that?',
            a: 'It raises a TypeError naming the argument, the compiled type and the called type, instead of tracing and compiling a new entry. At a serving boundary an explicit refusal beats a silent recompile that shows up as a latency spike.',
          },
        ],
        work: [
          { id: 'stage-timing', label: 'time the four stages on your own model and write down the ratio; if compile does not dominate, find out what does', href: '#four-stops-and-where-the-time-actually-goes' },
          { id: 'aot-the-step', label: 'compile one function ahead of time from ShapeDtypeStructs and call the Compiled object directly in your loop', href: '#the-compiled-object-refuses-instead-of-retracing' },
        ],
      },
    ],
  },
]
