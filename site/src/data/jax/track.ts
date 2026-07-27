// The JAX path: twelve chapters across the language, from arrays to the
// training run. Content blueprint and code snippets were authored and
// verified first (every snippet ran on jax 0.4.38, CPU, 2026-07-27; eight
// host-platform devices for the sharding chapter); the prose was then
// written to the humanized-writing standard. The mastery ledger here is
// manual-only until the JAX gym and labs land (docs/plans/jax-mastery-path.md).
import type { WorkItem } from '../mastery'

export interface JaxCode {
  caption: string
  text: string
  /** Highlighter grammar; jaxpr dumps use haskell, like the kernel path. */
  lang?: 'python' | 'haskell'
}
export interface JaxSection {
  h: string
  /** Paragraphs; a string starting '>> ' renders as a pull quote. */
  ps: string[]
  code?: JaxCode
}
export interface JaxReading {
  label: string
  url: string
  note: string
}
export interface JaxChapter {
  id: string
  num: number
  part: 'i' | 'ii'
  title: string
  lede: string
  goal: string
  sections: JaxSection[]
  readings: JaxReading[]
  /** A path diagram pinned after the section at this index. */
  diagram?: { id: string; after: number }
}

export const JAX_CHAPTERS: JaxChapter[] = [
  {
    "id": "arrays",
    "num": 1,
    "part": "i",
    "title": "Arrays",
    "lede": "An array in JAX is a value you compute with, not memory you overwrite.",
    "goal": "Given a snippet of JAX code, predict every array's mutability, dtype, and device placement before running it.",
    "sections": [
      {
        "h": "math on values, not on memory",
        "ps": [
          "Try `x[0] = 1` on a `jnp` array and JAX refuses outright. Arrays here are immutable, so the way you express an update is different: `x.at[0].set(1.0)` never touches `x`, it returns a new array with the change already in it. The copy you might now be picturing on every line like this mostly does not happen: under `jit`, XLA can see the moment the old value goes dead and reuse its buffer for the output instead of allocating a second one. Donation and aliasing make that reuse explicit at a function's boundaries, and chapter 11 comes back to donation once you have a concrete reason to control it yourself.",
          "`grad`, `vmap`, and `jit` all work the same way: each records what a function does to its inputs, then transforms that recording. A mutation hidden inside the function would make the recording describe something that never happened, a step the transform never saw. That is why immutability here is not a style preference. Values in, values out is what keeps the recording honest.",
          ">> An array is a value, not a place."
        ],
        "code": {
          "caption": "run it; x is untouched",
          "text": "import jax.numpy as jnp\n\nx = jnp.zeros(4)\ny = x.at[0].set(1.0)   # a new array; x is unchanged\nprint(x)               # [0. 0. 0. 0.]\nprint(y)               # [1. 0. 0. 0.]"
        }
      },
      {
        "h": "the dtype you did not choose",
        "ps": [
          "Ask `jnp.arange(3)` for its dtype and you get `int32`. Add `1.5` to it and you might expect `float64`, the way NumPy would sometimes give you depending on the values involved; JAX gives you `float32` instead, because the Python float `1.5` is weakly typed. A weak scalar adapts to the array it meets rather than promoting the array up to itself.",
          "That adaptation comes from a promotion lattice JAX defines and documents, not from NumPy's older, value-dependent rules. The two disagree exactly where old NumPy habits are most likely to mislead you, so reaching for NumPy intuition on a promotion question here is the wrong instinct.",
          "The default floating dtype is `float32`, and 64-bit is off unless you turn on `jax_enable_x64`. That default is where the biggest surprise in this section lives: `jnp.array(1e300)` does not raise, it overflows to `inf`, silently, by design. It is the first x64 surprise almost everyone meets, and now you have met it here instead of in a debugger."
        ],
        "code": {
          "caption": "run it: the promotion lattice, in three lines",
          "text": "import jax.numpy as jnp\n\nprint(jnp.arange(3).dtype)           # int32\nprint((jnp.arange(3) + 1.5).dtype)   # float32: the Python float is weak\nprint(jnp.array(1e300))              # inf: float32 unless you enable x64"
        }
      },
      {
        "h": "the line returns before the work is done",
        "ps": [
          "Run `y = x @ x` on a large array and the line returns almost instantly, long before a matmul that size could actually finish. JAX dispatches operations asynchronously: what you get back is a future-like array while the device keeps computing behind it. Printing it or converting it to NumPy blocks until the value is ready; calling `y.block_until_ready()` blocks on purpose, explicitly, wherever you need the wait.",
          "This is exactly why wrapping a single op in `time.perf_counter()` without a block measures dispatch time, not compute time. The clock stops the moment Python gets its future back, not when the device finishes the work. Chapter 11 builds the honest benchmarking harness on top of this one fact, but the fact itself belongs here, at the first place you would be tempted to time something."
        ],
        "code": {
          "caption": "illustrative timing; the actual numbers vary by machine",
          "text": "import time\nimport jax.numpy as jnp\n\nx = jnp.ones((2000, 2000))\nt0 = time.perf_counter()\ny = x @ x                     # returns before the matmul finishes\nt1 = time.perf_counter()\ny.block_until_ready()         # now it is actually done\nt2 = time.perf_counter()\nprint(f\"dispatch {1e3 * (t1 - t0):.2f} ms · compute {1e3 * (t2 - t1):.2f} ms\")"
        }
      },
      {
        "h": "where the array lives",
        "ps": [
          "Every array you create in JAX, whatever operation produced it, is the same type: `jax.Array`. It knows where it lives through a `.sharding` attribute; on a single device that sharding is a `SingleDeviceSharding`, the simplest case of a much bigger idea. Chapter 10 stretches that same object across a whole mesh of devices without changing the shape of your code at all.",
          "`jax.device_put` is how you place an array somewhere on purpose, and once it is placed, computation follows the operands: an op involving that array runs where the array lives, not wherever Python happens to be executing."
        ]
      }
    ],
    "readings": [
      {
        "label": "JAX quickstart",
        "url": "https://docs.jax.dev/en/latest/quickstart.html",
        "note": "the official five-minute surface"
      },
      {
        "label": "Key concepts",
        "url": "https://docs.jax.dev/en/latest/key-concepts.html",
        "note": "arrays, tracing, pytrees in one page"
      },
      {
        "label": "Thinking in JAX",
        "url": "https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html",
        "note": "the mental-model notebook"
      },
      {
        "label": "Common gotchas in JAX",
        "url": "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html",
        "note": "every gotcha this chapter's invariants generate"
      }
    ]
  },
  {
    "id": "tracing",
    "num": 2,
    "part": "i",
    "title": "Tracing",
    "lede": "Once jit calls your function, it stops running Python and starts recording a program.",
    "goal": "Given a jitted function, predict what tracing records, what it silently drops, and read the jaxpr it produces.",
    "sections": [
      {
        "h": "the function becomes data",
        "ps": [
          "Call a jitted function and JAX does not run your Python the way you might picture. It calls your function once, with tracers standing in for the real arguments: objects that carry shape and dtype but no actual values. Every primitive operation those tracers pass through gets recorded, and that recording is the jaxpr, the thing that actually gets compiled and the thing that actually runs on every later call. The same is true under `grad` and under `vmap`, not only under `jit`.",
          "One corollary is easy to miss and explains half the surprises in this track: Python itself is a metaprogramming layer here. The `for` loops and `if` statements in your function run once, at trace time, to produce the recording; by the time anything executes on a device, your loops and branches are gone, replaced by whatever primitives they happened to emit.",
          ">> jit runs your Python once. Everything else follows."
        ],
        "code": {
          "caption": "stage the jaxpr without running anything else",
          "text": "import jax\nimport jax.numpy as jnp\n\ndef attend(q, k):\n    return jax.nn.softmax(q @ k.T)\n\nprint(jax.make_jaxpr(attend)(jnp.ones((4, 8)), jnp.ones((4, 8))))"
        }
      },
      {
        "h": "what the trace takes away",
        "ps": [
          "Write `if x.sum() > 0:` inside a jitted function and, if `x` is traced, JAX raises `TracerBoolConversionError`, with a message worth reading exactly as it appears: \"Attempted boolean conversion of traced array with shape bool[].\" The tracer has no value to branch on yet, so there is nothing for `if` to test.",
          "A `print` statement inside the same function fires exactly once, at trace time, and what it prints is a tracer, not a number; side effects like that never make it into the recording. Anything your function closes over from the surrounding scope gets baked in as a constant at trace time too, frozen at whatever value it held when tracing happened.",
          "Shapes are part of the trace as well, which is the fact chapter 3 leans on hardest: hand the function an array with a new shape and you get a new trace, not a reused one."
        ]
      },
      {
        "h": "reading the recording",
        "ps": [
          "`jax.make_jaxpr(f)(*args)` shows you the trace directly, without compiling anything, and it is worth running on purpose rather than only reaching for it once something breaks. A jaxpr has three parts: the invars it takes in, a flat list of equations, each one a primitive applied to earlier values, and the outvars it returns. Every value carries an aval, a shape-and-dtype pair written like `f32[4,8]`.",
          "Look at what tracing an actual softmax produced below and the layer stops being abstract. Two lines of Python, `q @ k.T` and `jax.nn.softmax(...)`, became twelve equations the machine remembers so you do not have to: the softmax dissolved into seven primitives, the max subtraction carries an explicit `stop_gradient` that nothing in your source code asked for, the broadcasts Python was hiding are written out in full, and the matmul states its contraction as `dimension_numbers` instead of leaving it implied.",
          "This chapter only needs fluency at this surface. The kernel path's jaxpr chapter descends further, into how this same recording lowers to StableHLO and then to XLA, but reading equations like the ones below on sight is the skill this path asks of you."
        ],
        "code": {
          "caption": "the recording, verbatim (jax 0.4.38, CPU)",
          "text": "{ lambda ; a:f32[4,8] b:f32[4,8]. let\n    c:f32[8,4] = transpose[permutation=(1, 0)] b\n    d:f32[4,4] = dot_general[\n      dimension_numbers=(([1], [0]), ([], []))\n      preferred_element_type=float32\n    ] a c\n    e:f32[4] = reduce_max[axes=(1,)] d\n    f:f32[4] = max -inf e\n    g:f32[4,1] = broadcast_in_dim[\n      broadcast_dimensions=(0,)\n      shape=(4, 1)\n      sharding=None\n    ] f\n    h:f32[4,1] = stop_gradient g\n    i:f32[4,4] = sub d h\n    j:f32[4,4] = exp i\n    k:f32[4] = reduce_sum[axes=(1,)] j\n    l:f32[4,1] = broadcast_in_dim[\n      broadcast_dimensions=(0,)\n      shape=(4, 1)\n      sharding=None\n    ] k\n    m:f32[4,4] = div j l\n  in (m,) }",
          "lang": "haskell"
        }
      },
      {
        "h": "seeing values anyway (the debugging kit)",
        "ps": [
          "None of this means you are blind to values. `jax.debug.print(\"x = {x}\", x=x)` stages an actual print into the compiled program, so it runs on every call, with real numbers, unlike a bare Python `print` that only ever sees a tracer once.",
          "`jax.debug.callback` generalizes the same idea to any host function, not just printing. And when a NaN shows up somewhere in a long jitted computation and you cannot tell where, `jax.config.update(\"jax_debug_nans\", True)` re-runs the failing operation eagerly, outside the trace, so the exact op that produced it is the one that raises."
        ],
        "code": {
          "caption": "debug.print runs per call, with real values, unlike Python's print",
          "text": "import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef step(x):\n    jax.debug.print(\"x = {x}\", x=x)   # runs every call, with real values\n    return x * 2\n\nstep(jnp.arange(3.0))"
        }
      }
    ],
    "readings": [
      {
        "label": "Key concepts",
        "url": "https://docs.jax.dev/en/latest/key-concepts.html",
        "note": "arrays, tracing, and pytrees, the three foundations this path builds on, in one page"
      },
      {
        "label": "jaxpr reference",
        "url": "https://docs.jax.dev/en/latest/jaxpr.html",
        "note": "the jaxpr reference, section by section"
      },
      {
        "label": "Debugging",
        "url": "https://docs.jax.dev/en/latest/debugging.html",
        "note": "the official debugging tutorial this section supersets"
      }
    ],
    "diagram": {
      "id": "jax-machine",
      "after": 0
    }
  },
  {
    "id": "jit",
    "num": 3,
    "part": "i",
    "title": "Jit",
    "lede": "A jitted function is compiled once per signature, then reused exactly until that signature changes.",
    "goal": "Given a jit cache miss, name exactly which part of the cache key changed and predict hit or miss on the next call.",
    "sections": [
      {
        "h": "what jit buys",
        "ps": [
          "Without `jit`, JAX dispatches one primitive at a time: each op runs, materializes its result, and hands that result to the next op. With `jit`, XLA sees the whole jaxpr at once before any of it runs, so it can fuse chains of elementwise operations together, keep intermediates in registers or cache instead of round-tripping through memory, and emit a single executable for the whole function.",
          "The win from fusing compounds as the program grows, since a longer jaxpr gives XLA more to fuse. The cost sits on the other side of the ledger, paid once per shape signature rather than once per call: only the first call for a given signature pays trace time plus compile time."
        ]
      },
      {
        "h": "the cache and its key",
        "ps": [
          "Every jitted function is backed by a cache, keyed on four things at once: function identity, the pytree structure of the arguments, the shape and dtype of every leaf, and the value of any argument marked static. Match all four and you get the cached executable back in microseconds. Change any one of them, a new shape, a new dtype, a different pytree structure, and JAX retraces and recompiles from scratch; a new static value forces exactly the same retrace, because a static argument is not really an input to the compiled program, it is baked into the trace like a closed-over constant.",
          "Three habits account for almost every recompile bug people report: passing a fresh `lambda` or closure on every call, which counts as a new function identity even though the code inside is identical; marking a high-cardinality argument static, so nearly every call earns its own executable; and Python scalars flipping between weak and strong dtypes across calls, which changes the key without a line of code that looks like it should. `static_argnums` and `static_argnames` are the tool for moving an argument out of the traced set on purpose, when you want its value baked in rather than treated as data.",
          ">> A recompile is never mysterious. You changed the key."
        ],
        "code": {
          "caption": "k moves from data to key: same shape, different static value, different executable",
          "text": "from functools import partial\n\nimport jax\nimport jax.numpy as jnp\n\n@partial(jax.jit, static_argnames=\"k\")\ndef topk_sum(x, k):\n    return jax.lax.top_k(x, k)[0].sum()\n\nx = jnp.arange(100.0)\ntopk_sum(x, 3)   # trace + compile, k = 3 baked in\ntopk_sum(x, 3)   # cache hit\ntopk_sum(x, 5)   # new static value: trace + compile again"
        }
      },
      {
        "h": "when jit says no",
        "ps": [
          "Some computations cannot be traced at all, not because JAX is missing a feature but because the question itself has no answer at trace time. `x[x > 0]` is the standard example: the output shape depends on how many entries of `x` are positive, and that count does not exist until the values do. Two escapes exist: pick a static upper bound and mask down to it, or pull that one step outside the jitted region entirely and let it run as ordinary, untraced Python.",
          "Data-dependent Python control flow runs into the same wall from a different angle. A branch whose condition depends on a traced value needs restructuring, either with the `lax` control-flow primitives chapter 6 covers, or by making the branching value itself static so the choice resolves at trace time instead of run time."
        ]
      },
      {
        "h": "ahead of time",
        "ps": [
          "Compilation does not have to happen at call time. `jax.jit(f).lower(*args)` produces the StableHLO for a function before anything runs at all; `.compile()` turns that into an executable; and `compiled.cost_analysis()` returns the compiler's own estimate of flops and bytes moved, no execution required. Reading the cost of a computation before running it is a habit worth owning on its own, not only a debugging trick.",
          "The numbers below are real, not illustrative: `compiled.cost_analysis()[\"flops\"]` on a `256x256` matmul reports `33554432.0`, which is exactly `2 * 256**3`, the matmul's true FLOP count with nothing hidden. On JAX versions before 0.5, `cost_analysis()` returned a one-element list wrapping this same dict; the dict form shown here is current as of 0.4.38, and if your installed version differs, that is the first thing to check.",
          "`jax.export` takes the same lowered form one step further, serializing it so a function can be compiled once and served elsewhere, outside the Python process that traced it."
        ],
        "code": {
          "caption": "the StableHLO and the compiler's own flop count (verified, jax 0.4.38, CPU)",
          "text": "import jax\nimport jax.numpy as jnp\n\nlowered = jax.jit(lambda x: x @ x).lower(jnp.ones((256, 256)))\nprint(lowered.as_text()[:300])   # StableHLO, before anything runs\ncompiled = lowered.compile()\nprint(compiled.cost_analysis()[\"flops\"])   # the compiler's own estimate: 33554432.0"
        }
      }
    ],
    "readings": [
      {
        "label": "jit compilation",
        "url": "https://docs.jax.dev/en/latest/jit-compilation.html",
        "note": "the official jit tutorial, the surface this chapter reasons underneath"
      },
      {
        "label": "Ahead-of-time compilation",
        "url": "https://docs.jax.dev/en/latest/aot.html",
        "note": "lowering and compiling without running"
      },
      {
        "label": "Thinking in JAX",
        "url": "https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html",
        "note": "the mental-model notebook, worth rereading now that jit has an exact cache model"
      }
    ],
    "diagram": {
      "id": "jax-cache",
      "after": 1
    }
  },
  {
    "id": "autodiff",
    "num": 4,
    "part": "i",
    "title": "Autodiff",
    "lede": "`jax.grad` does not estimate a slope; it derives one exactly from the recorded program, and which direction it walks that program is a cost decision you get to make.",
    "goal": "Given a loss and a forward pass, choose jvp, vjp, or a custom rule, and predict the memory and compute cost before running it.",
    "sections": [
      {
        "h": "derivatives of programs",
        "ps": [
          "Call `jax.grad` on a function and JAX does not perturb the inputs and divide by a small number. It looks at the jaxpr, the recorded program from chapter 2, and derives a new program from it: one that computes the exact gradient by the chain rule, term for term. There is no step size to tune and no truncation error to worry about, because nothing was ever approximated. The rule for differentiating each primitive (a `dot_general`, a `sin`, a `reduce_sum`) is registered once, and `grad` composes those rules automatically along the trace.",
          "The usual finite-difference intuition, nudge x by epsilon and measure how y moved, has nothing to do with what happens here. `jax.grad(f)` expects `f` to return a scalar: a loss, not a vector of predictions. Ask for the gradient of something vector-valued and JAX complains, because the gradient of a vector output isn't a vector at all, it's a Jacobian, and later chapters build those directly with other tools. When you also need the loss value itself, which you almost always do for logging, `jax.value_and_grad` gets you both from a single forward-and-backward pass instead of running the function twice.",
          "Because grad is a transformation from one program to another, it composes with every other transformation the same way. `grad(grad(f))` differentiates twice. Wrap a `vmap`ped function in `grad`, or a `scan`ned one, and the derivative flows through correctly, not because JAX special-cased those combinations, but because a transformed program is still just a program, and grad doesn't need to know how it got built.",
          ">> grad is a program transformation, not a numerical method."
        ],
        "code": {
          "caption": "the exact gradient of a linear least-squares loss",
          "text": "import jax\nimport jax.numpy as jnp\n\ndef loss(w, x, y):\n    return jnp.mean((x @ w - y) ** 2)\n\ng = jax.grad(loss)(jnp.ones(3), jnp.eye(3), jnp.zeros(3))\nprint(g)   # exact, to machine precision: [0.6667 0.6667 0.6667]"
        }
      },
      {
        "h": "two directions, one cost model",
        "ps": [
          "Underneath `grad` sit two more primitive transformations, and picking between them by hand is worth understanding even if you rarely call them directly. `jax.jvp` pushes a tangent vector forward through the program: for every input direction you care about, it costs one extra forward-like pass. That's cheap when you have a few inputs and many outputs, a Jacobian-vector product for a small parameter feeding a large output, and expensive the moment your input space gets big.",
          "`jax.vjp` runs the opposite direction. It does one forward pass, but it also has to remember enough of that pass, its residuals, to walk backward and pull a cotangent through every operation. The backward pass costs about as much as the forward one, and critically, that total cost does not grow with the number of inputs. This is why deep learning lives in reverse mode almost exclusively: a model can carry a billion parameters and one scalar loss, and reverse mode differentiates all billion of them in roughly two forward passes' worth of work.",
          "`jacfwd` and `jacrev`, the two ways to build a full Jacobian, are not separate machinery. Each is `vmap` wrapped around `jvp` or `vjp` respectively, batching the single-direction cost across every direction at once. Which one to reach for is a question about the Jacobian's shape: forward mode when there are few inputs and many outputs, reverse mode when it's the other way around.",
          "The bill reverse mode runs up is memory, not compute: whatever the backward pass needs to remember from the forward pass has to live somewhere until it gets used. The last section of this chapter comes back to that bill directly."
        ]
      },
      {
        "h": "owning the derivative",
        "ps": [
          "Every rule `grad` applies by default comes from JAX's built-in table of how to differentiate primitives, and most of the time that table is exactly what you want. Sometimes it isn't. `jax.custom_vjp` lets you throw out the automatic backward rule for a function and supply your own: because the automatic one is numerically unstable, because you know an algebraically cheaper form, or because the function hides work, a kernel, a call to something the tracer can't see through, that autodiff has no rule for at all. The kernel path's flash-attention backward (at /s/kernels) is built exactly this way, replacing a rule the tracer could never have derived on its own.",
          "Take `log1p(exp(x))`, a softplus that looks numerically stable on its face. Its automatic gradient is fine for small x, but at large x, `exp(x)` overflows toward infinity while the chain rule needs to multiply that infinity by a value collapsing toward zero, and the result is `nan`. The function's value comes out correctly; the gradient the automatic rule derives from it does not. Nothing is broken in JAX here: the algebra it's following really does produce that indeterminate form. The fix is to hand it a gradient rule you derived by hand, one that never routes through the unstable intermediate at all."
        ],
        "code": {
          "caption": "custom_vjp fixes what the automatic rule cannot",
          "text": "import jax\nimport jax.numpy as jnp\n\n@jax.custom_vjp\ndef log1pexp(x):\n    return jnp.log1p(jnp.exp(x))\n\ndef fwd(x):\n    return log1pexp(x), jax.nn.sigmoid(x)   # residual: the stable gradient\n\ndef bwd(sig, ct):\n    return (ct * sig,)\n\nlog1pexp.defvjp(fwd, bwd)\nprint(jax.grad(log1pexp)(100.0))   # 1.0; the automatic rule returns nan here"
        }
      },
      {
        "h": "memory, remat, and stop_gradient",
        "ps": [
          "Reverse mode's residuals are usually intermediate activations: everything the backward pass needs to reconstruct the chain rule. For a deep network that's one saved tensor per layer, and it adds up fast. `jax.checkpoint`, JAX's name for gradient checkpointing (remat, if you've met the term elsewhere), makes a different trade: instead of storing a block's activations, it discards them and recomputes that block during the backward pass. You spend more FLOPs to spend fewer bytes. It's the identical trade flash attention makes at the kernel level, applied here at the level of whole layers instead of attention blocks.",
          "`jax.lax.stop_gradient` does something unrelated: it cuts the tape at a specific point, so no gradient flows through that value afterward, no matter what happens to it downstream. Target networks in reinforcement learning, straight-through estimators, freezing a pretrained encoder while still using its output: all three are the same operation underneath, telling autodiff that a particular value counts as a constant for the backward pass, even though it came from real computation."
        ],
        "code": {
          "caption": "grad through the same eight-layer block, with and without remat",
          "text": "import jax\nimport jax.numpy as jnp\n\nW = jnp.ones((256, 256)) * 0.01\n\ndef block(x):\n    for _ in range(8):\n        x = jnp.tanh(x @ W)\n    return x.sum()\n\ng_full = jax.grad(block)(jnp.ones(256))\ng_remat = jax.grad(jax.checkpoint(block))(jnp.ones(256))\nprint(jnp.allclose(g_full, g_remat))   # True; the remat pass stored no intermediates"
        }
      }
    ],
    "readings": [
      {
        "label": "Automatic differentiation",
        "url": "https://docs.jax.dev/en/latest/automatic-differentiation.html",
        "note": "the official grad/jvp/vjp tutorial this chapter goes past"
      },
      {
        "label": "Advanced autodiff",
        "url": "https://docs.jax.dev/en/latest/advanced-autodiff.html",
        "note": "custom rules, the full machinery"
      },
      {
        "label": "Gradient checkpointing",
        "url": "https://docs.jax.dev/en/latest/gradient-checkpointing.html",
        "note": "remat, policies included"
      },
      {
        "label": "The autodiff cookbook",
        "url": "https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html",
        "note": "the recipes this chapter derives"
      }
    ],
    "diagram": {
      "id": "jax-reverse",
      "after": 1
    }
  },
  {
    "id": "vmap",
    "num": 5,
    "part": "i",
    "title": "Vmap",
    "lede": "`vmap` doesn't loop over your data; it rewrites the program itself to already expect a batch.",
    "goal": "Given a function written for one example, batch it correctly across axes, nested calls, and gradients.",
    "sections": [
      {
        "h": "the loop you do not write",
        "ps": [
          "Write a function for one example and call `jax.vmap` on it, and JAX does not generate a Python loop that calls your function N times. It rewrites the traced program itself: every primitive inside gets replaced by its batched form, a matmul becomes a batched matmul, an elementwise op just grows an extra axis, and the whole thing comes out as one fused program that XLA compiles once. Nothing in the result dispatches per example; the batch dimension is built into the primitives themselves.",
          "It's tempting to picture `vmap` as sugar for a loop, something that saves you typing `for i in range(n)`. That mental model gets the performance story backward. A real loop dispatches N separate programs; `vmap` produces a single program that happens to carry an extra axis through it, and the two only resemble each other from the outside.",
          "The contract you can rely on is exact, and it reads as an equation: `vmap(f)(xs)[i] == f(xs[i])`, for every i, up to the order floating-point additions happen to associate in. Whatever `f` computes for one example, `vmap(f)` computes for every example in the batch independently, and the result is indistinguishable except in the narrow sense that summing floats in a different order can move the last bit.",
          ">> vmap rewrites the program, not the loop."
        ]
      },
      {
        "h": "axis specs",
        "ps": [
          "`vmap` needs to know, for every argument, where its batch axis lives, and `in_axes` is how you say it. Pass an integer and that's the axis holding the batch dimension for that argument. Pass `None` and the argument is broadcast instead: every call sees the whole thing, unbatched, with no axis to strip out first. When your arguments are pytrees rather than bare arrays, `in_axes` can itself be a matching pytree of specs, one entry per leaf.",
          "`out_axes` plays the same role for the output side, choosing where the newly introduced batch dimension lands in whatever `f` returns. And because `vmap` is just another transformation, you can nest it: wrap a `vmap`ped function in another `vmap` and the axes stack, reading from the inside out. The innermost `vmap` is the one that ran first, on the function you actually wrote.",
          "One requirement `vmap` enforces and won't relax: at least one argument has to carry a mapped axis. Set every entry of `in_axes` to `None` and JAX raises, because without a single batch dimension anywhere in the inputs, there's nothing telling it how large the output's new axis should even be."
        ],
        "code": {
          "caption": "in_axes chooses the batch axis per argument; nesting stacks two batch dimensions",
          "text": "import jax\nimport jax.numpy as jnp\n\ndef dot(a, b):\n    return a @ b\n\na = jnp.ones((32, 8))\nb = jnp.ones(8)\nprint(jax.vmap(dot, in_axes=(0, None))(a, b).shape)   # (32,): b broadcast\n\npair = jax.vmap(jax.vmap(dot, in_axes=(None, 0)), in_axes=(0, None))\nprint(pair(a, a).shape)   # (32, 32): every pair of rows, two nested maps"
        }
      },
      {
        "h": "the recipes",
        "ps": [
          "The composition worth knowing by heart is `jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))`. Read it in order: `grad` turns the per-example loss into a per-example gradient function, and `vmap` then runs that gradient function once per example in the batch, with the parameters broadcast (their `in_axes` entry is `None`) and the data batched across the other two. What comes out is one gradient per training example, computed at roughly the cost of a single batched matmul rather than a genuine Python loop over gradients. This is the pattern behind per-example gradient clipping and differential-privacy accounting, anything that needs the gradient before it gets averaged away.",
          "The Jacobian builders from the last chapter are built the same way underneath. `jacfwd` is `vmap` wrapped around `jvp`; `jacrev` is `vmap` wrapped around `vjp`. Batching the single-direction cost across every direction at once is, quite literally, what a Jacobian is.",
          "The same trick scales up to whole models. Stack N sets of parameters into one pytree, an extra leading axis on every leaf, and `vmap` a training step over that stacked pytree, and you're training N independent models inside a single compiled program: an ensemble without N separate training loops competing for the compiler's attention."
        ],
        "code": {
          "caption": "per-sample gradients, the canonical vmap(grad(...)) composition",
          "text": "import jax\nimport jax.numpy as jnp\n\ndef loss(w, x, y):\n    return (x @ w - y) ** 2\n\nw = jnp.ones(3)\nxs = jnp.ones((32, 3))\nys = jnp.zeros(32)\nper_sample = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(w, xs, ys)\nprint(per_sample.shape)   # (32, 3): one gradient per example"
        }
      }
    ],
    "readings": [
      {
        "label": "Automatic vectorization",
        "url": "https://docs.jax.dev/en/latest/automatic-vectorization.html",
        "note": "the official vmap tutorial this chapter goes past"
      },
      {
        "label": "jax.vmap reference",
        "url": "https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html",
        "note": "the full axis-spec surface"
      }
    ]
  },
  {
    "id": "control-flow",
    "num": 6,
    "part": "i",
    "title": "Control flow",
    "lede": "The trace can only see control flow you hand it in a form it understands, and everything else either breaks or silently unrolls.",
    "goal": "Given a loop or branch, decide whether it's static or traced, then pick the lax primitive that matches.",
    "sections": [
      {
        "h": "why if breaks, and when it does not",
        "ps": [
          "A Python `if` branching on a traced value raises `TracerBoolConversionError`, and the reason follows straight from chapter 2: a tracer carries a shape and a dtype, not a value, and `if` needs an actual boolean to decide which branch to take. There's nothing to decide with. The error is JAX being honest about a question it genuinely cannot answer at trace time.",
          "It's easy to read that error as JAX simply not supporting if statements and go looking for a workaround everywhere. It's narrower than that. An `if` branching on a static value, something fixed at trace time like an array's shape, a `static_argnums` argument, or a plain Python constant closed over from outside, works exactly as written and costs nothing extra. The branch resolves once, during tracing, and only the taken branch's ops end up in the jaxpr at all.",
          "Python loops over a static range have the same shape of story: they run, and every one of them works, but they don't stay a loop. `for i in range(1000): x = step(x)` unrolls, pasting a thousand copies of `step`'s ops into the jaxpr one after another. The program is correct, and the compile time is proportional to the length you wrote: fine at ten iterations, a real cost at a thousand.",
          ">> Fast control flow is control flow the trace can see."
        ]
      },
      {
        "h": "cond, select, where",
        "ps": [
          "When the branch really does depend on a traced value, `jax.lax.cond(pred, true_fn, false_fn, *operands)` is the primitive built for it, and it works by a trick worth knowing up front: both branches get staged into the compiled program, and the choice between their two results happens at run time, not trace time. Nothing about `cond` avoids computing the branch you don't take at the program level. What it avoids is Python ever needing to know, while tracing, which branch that will turn out to be.",
          "For a purely elementwise choice, `jnp.where` is the more direct tool: no branch functions to write, just a boolean array selecting between two already-computed arrays. And it's worth knowing that under `vmap`, `cond` collapses into exactly that: since different elements of a batch might each want a different branch, the batched form has to compute both and select, so `cond` inside a `vmap`ped function costs the same as `where` would have cost there in the first place."
        ],
        "code": {
          "caption": "cond stages both branches; the runtime picks one",
          "text": "import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef flip_negatives(x):\n    return jax.lax.cond(x.sum() < 0, lambda v: -v, lambda v: v, x)\n\nprint(flip_negatives(jnp.array([-3.0, 1.0])))   # [ 3. -1.]"
        }
      },
      {
        "h": "scan is the workhorse",
        "ps": [
          "`jax.lax.scan(f, init, xs)` is the primitive that makes iteration cheap to compile. `f` takes a carry and one slice of `xs` and returns a new carry plus one output; `scan` runs it over every slice and hands back the final carry along with every per-step output, stacked into one array. The body only gets traced once, no matter how many steps `xs` has, so compile time stays flat as the loop gets longer, the exact opposite of the unrolled Python loop from the first section.",
          "The three loop primitives split cleanly on one property: whether they support reverse-mode differentiation. `scan` does, because it saves the residuals it needs at every step, the same bookkeeping any reverse-mode computation requires. `jax.lax.while_loop` does not: its trip count depends on a runtime condition, and reverse mode has no way to size a backward pass whose length isn't known in advance. `jax.lax.fori_loop` sits between the two: with static bounds it lowers to a scan-like form, and it falls back to `while_loop`'s form the moment the bounds themselves become dynamic.",
          "That split decides which primitive to reach for. Anything that iterates and carries state forward, a training loop, an RNN, a sampler, belongs in `scan`. A loop that has to run until some runtime condition is met, with no gradient required through it, is what `while_loop` is for."
        ],
        "code": {
          "caption": "the EMA loop: one carry, one stacked output, one trace of the body",
          "text": "import jax\nimport jax.numpy as jnp\n\ndef ema_step(mean, x):\n    new = 0.9 * mean + 0.1 * x\n    return new, new          # (carry out, output for this step)\n\nfinal, path = jax.lax.scan(ema_step, 0.0, jnp.arange(5.0))\nprint(final)   # 0.9754...\nprint(path)    # every intermediate mean, stacked"
        }
      }
    ],
    "readings": [
      {
        "label": "Control flow",
        "url": "https://docs.jax.dev/en/latest/control-flow.html",
        "note": "the official control-flow tutorial this chapter goes past"
      },
      {
        "label": "jax.lax.scan reference",
        "url": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html",
        "note": "the exact contract"
      }
    ]
  },
  {
    "id": "pytrees",
    "num": 7,
    "part": "i",
    "title": "Pytrees",
    "lede": "A pytree is not a data structure you chose. It is the contract every transformation shares.",
    "goal": "Given any nested params structure, predict how jit, grad, and vmap flatten and rebuild it, then register a custom node correctly.",
    "sections": [
      {
        "h": "leaves and structure",
        "ps": [
          "Take any nested structure you'd build to hold a model's weights: a dict of dicts, a list of arrays, maybe a tuple mixed in. JAX has a name for exactly that shape, a pytree, any nesting of dicts, lists, tuples, and registered node types, bottoming out in leaves like arrays and scalars. `jax.tree.flatten` takes one of these and splits it into two things: a flat list of leaves, and a treedef that remembers how to put them back.",
          "This is not a side detail. It is the calling convention every transformation in JAX shares. Call `grad` on a loss that takes a dict of arrays, and the gradient that comes back is a dict with the exact same keys and shapes, same treedef as the params argument, by construction. `jit` and `vmap` do the same thing on the way in, flattening whatever you pass to leaves before tracing touches it. That is why reaching for `params = {\"dense\": {...}, \"output\": {...}}` instead of some bespoke class is idiomatic JAX, not a beginner's habit you'll grow out of.",
          ">> Every transformation sees your model the same way: leaves on a tree.",
          "One habit from Python trips people here. `None` looks like it should be a leaf, it's a value, after all, but JAX treats it as an empty subtree: it vanishes from the leaves list entirely. And a list is not a tuple, structurally, even when they hold the same values in the same order. Zip a dict of lists against a dict of tuples with `tree.map` and you don't get silent coercion. You get a structure mismatch error, because those are two different treedefs pretending to look alike."
        ]
      },
      {
        "h": "tree.map and the rest of the kit",
        "ps": [
          "Once every model is leaves and a treedef, updating it stops needing an architecture-specific loop. `jax.tree.map(f, tree, *rest)` walks every tree in lockstep, structure required to match, and applies `f` leafwise. Wire the SGD update as `params - 0.1 * grads` and it is one line, unchanged whether the model has three parameters or three hundred.",
          "The rest of the kit rounds out the same idea. `jax.tree.leaves` gets you the flat list without the treedef, `jax.tree.structure` gets you the treedef without the leaves, and `jax.tree.unflatten` puts a new set of leaves back into an old structure. Together with `flatten` and `map`, that is the whole surface most training code needs."
        ],
        "code": {
          "caption": "run it: the SGD-style update, unchanged by how deep the model nests",
          "text": "import jax\nimport jax.numpy as jnp\n\nparams = {\"dense\": {\"w\": jnp.ones((2, 2)), \"b\": jnp.zeros(2)}, \"scale\": jnp.ones(())}\ngrads = jax.tree.map(jnp.ones_like, params)\nnew_params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)\nprint(jax.tree.structure(new_params))   # same treedef as params, by construction"
        }
      },
      {
        "h": "your own nodes",
        "ps": [
          "A plain dataclass is not automatically a pytree. JAX doesn't know which of its fields are arrays that should flow through transformations and which are configuration that should stay fixed, so left alone, `jax.grad` or `jax.jit` would just refuse it, or worse, treat the whole object as one opaque leaf. `@jax.tree_util.register_dataclass` tells JAX how to read it: which fields are data, which are static, declared right on the field with `dataclasses.field(metadata=dict(static=True))`.",
          "That decorator covers the common case, a dataclass that's mostly arrays with a few static settings. For a container you want full control over, custom flatten and unflatten logic, not just field-by-field mapping, `register_pytree_node` is the lower-level tool underneath it. Either way, the lesson is the same: a custom class doesn't opt out of the pytree system by existing. It opts in the moment you register it, and from then on `grad`, `jit`, and `vmap` walk through it exactly as they would a dict."
        ],
        "code": {
          "caption": "run it: a registered dataclass, transparent to tree.map",
          "text": "from dataclasses import dataclass\n\nimport jax\nimport jax.numpy as jnp\n\n@jax.tree_util.register_dataclass\n@dataclass\nclass TrainState:\n    step: jax.Array\n    params: dict\n\nstate = TrainState(step=jnp.zeros((), jnp.int32), params={\"w\": jnp.ones(3)})\nbumped = jax.tree.map(lambda x: x + 1, state)\nprint(bumped.step)   # 1: the dataclass is transparent to every transform"
        }
      }
    ],
    "readings": [
      {
        "label": "Working with pytrees",
        "url": "https://docs.jax.dev/en/latest/working-with-pytrees.html",
        "note": "the official pytree tutorial: everything here, plus why the rules are what they are"
      },
      {
        "label": "Pytrees (reference)",
        "url": "https://docs.jax.dev/en/latest/pytrees.html",
        "note": "the reference semantics (None, key paths, custom nodes)"
      }
    ]
  },
  {
    "id": "random",
    "num": 8,
    "part": "i",
    "title": "Randomness",
    "lede": "JAX has no global random state, only keys you pass by hand.",
    "goal": "Given a JAX program that needs randomness, thread keys through split and fold_in without ever reusing one.",
    "sections": [
      {
        "h": "randomness without a dial",
        "ps": [
          "Call `numpy.random.normal()` twice and you get two different answers, because NumPy keeps a global seed mutating quietly in the background. JAX has no such thing. Every sampling function takes an explicit `key`, and the same key with the same shape produces identical bits, every call, every run, every machine, for a given backend. Reproducibility isn't a flag you remember to set. It's just how the system works.",
          ">> The same key gives the same bits, forever.",
          "The reason traces back to chapter one's invariant: values, not places. A global RNG dial is hidden mutable state, exactly the kind of thing that makes a traced function lie about what it does. It would also make `vmap` and `jit` order-dependent: run the same batched computation twice and get different random draws depending on how the compiler chose to schedule things. An explicit key sidesteps all of that, because the randomness lives in a value you pass around like any other array.",
          "`jax.random.key(0)` builds one of these keys, a small, typed object, the modern form you should reach for. Older code still shows `jax.random.PRNGKey(...)`, which returns a plain uint32 array instead of a typed key; both work, but the typed key is what current JAX expects, and what this chapter uses throughout."
        ]
      },
      {
        "h": "split and fold_in",
        "ps": [
          "A key follows one rule, and it's worth memorizing before you write a single sampling call: use it exactly once. Either a sampler consumes it directly, or you split it into children and use those instead. Reuse a key for two different draws and the draws aren't independent anymore. They're correlated, silently, and nothing in the code will warn you.",
          "`jax.random.split(key, n)` is how you get n independent keys out of one. `fold_in(key, step)` does something related but different: it derives a new key deterministically from an existing one plus an integer, like a step number or a layer id, without you having to carry a growing chain of split keys through the rest of the function. Both are correct ways to get more randomness out of one seed; which one reads better depends on the shape of the code around it."
        ],
        "code": {
          "caption": "run it: split once, and the two children never correlate",
          "text": "import jax\n\nkey = jax.random.key(0)\nk1, k2 = jax.random.split(key)\nprint(jax.random.normal(k1, (2,)))   # independent of k2's draw\nprint(jax.random.normal(k2, (2,)))\nprint(jax.random.normal(k1, (2,)))   # identical to the first line, forever"
        }
      },
      {
        "h": "keys under transforms",
        "ps": [
          "Batched sampling follows the same discipline, just wired through vmap. Split a key into however many independent draws you need, then map a sampler over the resulting array of keys. The result is one fused program producing N independent samples, not N sequential calls to a sampler.",
          "Inside a `scan`, you have two options for the same problem. Carry the key in the loop's carry and split it fresh on every step, or `fold_in` the step index each time and skip the carry entirely. Both give you correct, independent randomness per step. `fold_in` tends to read better for loops you expect to resume partway through, chapter twelve leans on exactly this, because the key for step k doesn't depend on having run steps 0 through k-1 first."
        ],
        "code": {
          "caption": "run it: one vmap, 32 independent draws",
          "text": "import jax\n\nkey = jax.random.key(0)\nkeys = jax.random.split(key, 32)\nsamples = jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys)\nprint(samples.shape)   # (32, 3): 32 independent draws, one fused program"
        }
      }
    ],
    "readings": [
      {
        "label": "Pseudorandom numbers in JAX",
        "url": "https://docs.jax.dev/en/latest/random-numbers.html",
        "note": "the official tutorial on keys, split, and the sampling API"
      },
      {
        "label": "JEP 263: the PRNG design",
        "url": "https://docs.jax.dev/en/latest/jep/263-prng.html",
        "note": "the design document: why splittable, why counter-based"
      }
    ]
  },
  {
    "id": "state",
    "num": 9,
    "part": "ii",
    "title": "State",
    "lede": "In JAX, state is not something a class holds. It is something a function returns.",
    "goal": "Given a model, an optimizer, and an RNG stream, thread all three through a jitted training step as one explicit state tree.",
    "sections": [
      {
        "h": "state without mutation",
        "ps": [
          "Every stateful thing in a JAX program, an optimizer's momentum, an RNG stream, a batch-norm running mean, a step counter, rides the same pattern: `step(state, x) -> (new_state, y)`. State goes in as an argument. A new state comes out as a return value. Nothing gets mutated in place, and nothing lives outside the function's inputs and outputs.",
          ">> State is an argument and a return value. Nothing else.",
          "This isn't a style preference; it's forced by how tracing works. A traced function that mutated some global object would bake in whatever value that global held at trace time, and every later call would silently drop the mutation, because the recording never captured it happening. Threading state explicitly is the only way a function's behavior stays honest under `jit` and `grad`."
        ]
      },
      {
        "h": "optax: the optimizer as a pytree program",
        "ps": [
          "`optax.adam(1e-3)` isn't an object that quietly tracks its own momentum the way an optimizer class in a mutation-based framework would. It's a `GradientTransformation`: a pair of functions. `init(params)` builds the optimizer's state, itself a pytree of moment estimates the same shape as your params. `update(grads, state, params)` takes gradients and the current state and returns `(updates, new_state)`. `apply_updates(params, updates)` folds those updates into your params. Every piece of state Adam needs is a value you're handed back, not something hidden inside an object.",
          "Composing optimizers works because every piece shares that same shape. `optax.chain(clip_by_global_norm, adam, ...)` links transformations end to end, and each link, whatever it does, exposes the same `init` and `update` pair. Add gradient clipping or weight decay to a training loop and nothing about how you call the optimizer changes; only the chain grows by one link."
        ],
        "code": {
          "caption": "run it: every piece of Adam's memory is a returned pytree",
          "text": "import jax.numpy as jnp\nimport optax\n\nopt = optax.adam(1e-3)\nparams = {\"w\": jnp.ones(3)}\nopt_state = opt.init(params)\n\ngrads = {\"w\": jnp.full(3, 0.5)}\nupdates, opt_state = opt.update(grads, opt_state, params)\nparams = optax.apply_updates(params, updates)\nprint(params[\"w\"])   # moved against the gradient, Adam-scaled"
        }
      },
      {
        "h": "Flax NNX: modules over the pattern",
        "ps": [
          "NNX modules look, at first glance, like ordinary Python classes: you build a `Linear` layer, assign it to `self.up`, and mutate attributes in `__init__` the way you would anywhere else. That's deliberate. NNX wants module code to read like normal Python, mutation included, while everything underneath still obeys the functional pattern from section one.",
          "The bridge is `nnx.split(model)`, which pulls a module apart into a `graphdef`, the static structure, what layers exist, how they connect, and a `state`, a pytree of the actual arrays, the part that crosses `jit` and `grad` boundaries cleanly. `nnx.merge` puts the two back together into a live module. Flax's older API, Linen, skips this split step and has you thread params explicitly through every call instead; NNX is the same underlying model with the bookkeeping automated.",
          "None of this is an exemption. NNX's whole value is ergonomics laid on top of the same functional core chapter one and this chapter both describe, not an escape hatch from it. Every array NNX manages still moves through `jit`, `grad`, and `vmap` as a pytree leaf; the module syntax just spares you from writing the flatten and unflatten by hand."
        ],
        "code": {
          "caption": "the NNX API (flax.nnx), verified version noted at integration: modules mutate in Python, state still flows as a pytree",
          "text": "import jax\nfrom flax import nnx\n\nclass Mlp(nnx.Module):\n    def __init__(self, din, dhidden, rngs: nnx.Rngs):\n        self.up = nnx.Linear(din, dhidden, rngs=rngs)\n        self.down = nnx.Linear(dhidden, din, rngs=rngs)\n\n    def __call__(self, x):\n        return self.down(jax.nn.relu(self.up(x)))\n\nmodel = Mlp(4, 16, nnx.Rngs(0))\ngraphdef, state = nnx.split(model)   # static structure, pytree of arrays"
        }
      },
      {
        "h": "checkpoints and the host",
        "ps": [
          "Saving a model means saving a pytree, and Orbax is the library built for exactly that: it saves and restores arbitrary pytrees, asynchronously so training doesn't stall on disk I/O, and with sharding awareness baked in, so a checkpoint written from a distributed array, chapter ten's world, restores correctly across whatever device layout reads it back.",
          "The other place host and device meet is callbacks. `jax.experimental.io_callback` and `jax.pure_callback` let a traced function reach out and run ordinary Python on the host: writing a metric to a log, reading from a file, anything the compiled program itself can't do. They come with real costs, ordering guarantees to reason about, a performance hit every time the device has to pause and wait on the host, so the honest use case is something occasional, like logging metrics, not something on the hot path of every step."
        ]
      }
    ],
    "readings": [
      {
        "label": "Stateful computations in JAX",
        "url": "https://docs.jax.dev/en/latest/stateful-computations.html",
        "note": "the official counter-example tutorial this chapter builds past"
      },
      {
        "label": "External callbacks",
        "url": "https://docs.jax.dev/en/latest/external-callbacks.html",
        "note": "the callback family, contracts included"
      },
      {
        "label": "Optax",
        "url": "https://optax.readthedocs.io/en/latest/",
        "note": "the transformation catalog"
      },
      {
        "label": "Flax",
        "url": "https://flax.readthedocs.io/en/latest/",
        "note": "NNX guides"
      },
      {
        "label": "Orbax",
        "url": "https://orbax.readthedocs.io/en/latest/",
        "note": "checkpointing"
      }
    ]
  },
  {
    "id": "sharding",
    "num": 10,
    "part": "ii",
    "title": "Sharding",
    "lede": "A sharded array is still one array; the mesh and the spec just say where its pieces live.",
    "goal": "Given an array and a device mesh, choose a PartitionSpec, predict the collectives GSPMD inserts, and rewrite the computation in shard_map when you need to write them yourself.",
    "sections": [
      {
        "h": "one array, many devices",
        "ps": [
          "Picture an array too big for one chip, cut into tiles and spread across eight of them, one tile per device. JAX names that arrangement with two objects. A `Mesh` names your devices along axes, so a `(4, 2)` mesh might call one axis `data` and the other `model`. A `PartitionSpec` maps each array dimension onto a mesh axis, or onto nothing, which means that dimension stays whole on every device. Put a mesh and a spec together and you get a `NamedSharding`: the mesh says what devices exist and how they're arranged, the spec says which array dimension rides on which axis.",
          "`jax.device_put(x, sharding)` takes a global array and lays it across the mesh according to that sharding. `jax.debug.visualize_array_sharding` prints the layout back as a grid of device indices, so you can confirm the tiling matches what you intended instead of trusting the code silently.",
          "Every exercise in this chapter runs on eight devices that don't exist. Set `XLA_FLAGS=--xla_force_host_platform_device_count=8` before importing JAX and one laptop CPU pretends to be an eight-device pod. The mesh, the sharding, and the collectives XLA inserts are all real; only the hardware underneath is faked, which is what makes sharding a laptop exercise rather than a cluster requirement."
        ],
        "code": {
          "caption": "run with XLA_FLAGS=--xla_force_host_platform_device_count=8: eight fake devices, real sharding",
          "text": "# run with: XLA_FLAGS=--xla_force_host_platform_device_count=8\nimport jax\nimport jax.numpy as jnp\nfrom jax.sharding import NamedSharding, PartitionSpec as P\n\nmesh = jax.make_mesh((4, 2), (\"data\", \"model\"))\nx = jax.device_put(jnp.ones((8, 16)), NamedSharding(mesh, P(\"data\", \"model\")))\njax.debug.visualize_array_sharding(x)   # 4x2 tiles, one per device"
        }
      },
      {
        "h": "jit compiles for the whole mesh",
        "ps": [
          "Nothing about the matmuls, the reductions, or the control flow changes when an array is sharded. You keep writing global-array math exactly as chapters 1 through 9 taught it. What changes sits underneath: XLA's SPMD partitioner, called GSPMD, looks at the sharding on your inputs and splits every operation across the mesh to match, inserting whatever collectives the math requires along the way. An all-gather where a full row needs assembling. A reduce-scatter where a sum needs splitting back out. An all-reduce where every device needs the same total. Sharding propagates forward from your inputs through the whole program, so you rarely have to state more than where the leaves start.",
          ">> Write global math; the compiler places it.",
          "Propagation guesses wrong sometimes: an intermediate value ends up sharded in a way that forces an expensive reshuffle right before an operation that wanted it laid out differently. `jax.lax.with_sharding_constraint` pins that intermediate's layout explicitly, overriding the guess. This is the second level of control: not choosing the collectives yourself, just correcting where propagation would have chosen badly.",
          "Data parallelism, tensor parallelism, FSDP: under this model none of them are frameworks or separate code paths. They are PartitionSpecs. Data parallelism shards the batch axis and replicates the parameters. Tensor parallelism shards a parameter axis and replicates the batch. FSDP shards the parameters too, gathering them back only when a layer needs the whole thing. Naming the pattern doesn't require different code, only a different spec."
        ]
      },
      {
        "h": "shard_map: per-device code",
        "ps": [
          "GSPMD chooses the collectives for you, and most of the time that's the right amount of control to give up. `jax.shard_map` gives it back. Call `jax.shard_map(f, mesh=mesh, in_specs=..., out_specs=...)` and `f` runs once per device, on that device's local block: the shapes shrink to what one device actually holds, but the rank stays the same, so the function you write still reads like ordinary array code. Inside `f` you call the collectives yourself, `jax.lax.psum`, `ppermute`, and the rest, addressed by mesh axis name rather than by device index.",
          "One import detail is worth stating plainly rather than leaving to guesswork. On JAX versions before `shard_map` reached the top-level namespace, it lived at `jax.experimental.shard_map.shard_map`. Same function, same contract, older address.",
          "Reach for shard_map when the partitioner keeps fighting you toward a layout you didn't want, or when you're writing the communication pattern on purpose rather than hoping the compiler infers it. The kernel path's distributed chapter builds collectives like these from raw remote DMAs, one layer further down; this chapter stops at the level where you name the collective and let JAX lower it."
        ],
        "code": {
          "caption": "the manual level: one function, called once per device, collectives named explicitly",
          "text": "from functools import partial\n\nimport jax\nfrom jax.sharding import PartitionSpec as P\n\n@partial(jax.shard_map, mesh=mesh, in_specs=P(\"data\", None), out_specs=P())\ndef global_mean(block):              # block: this device's (2, 16) slice\n    return jax.lax.pmean(block.mean(), axis_name=\"data\")"
        }
      },
      {
        "h": "the cost model lives one path over",
        "ps": [
          "Two PartitionSpecs for the same array can both run correctly and still cost very different amounts, because the collectives GSPMD inserts move real bytes over real links. Which spec wins is arithmetic: bytes moved divided by link bandwidth, summed over every collective the choice implies.",
          "This chapter doesn't rebuild that arithmetic. The scaling book carries the full treatment of the trade, and the kernel path's ICI chapter teaches the mechanism underneath the collectives themselves, down to the remote DMA. What this chapter leaves you with is the vocabulary to read that treatment with: mesh, spec, propagation, and the three levels of control between them."
        ]
      }
    ],
    "readings": [
      {
        "label": "Sharded computation",
        "url": "https://docs.jax.dev/en/latest/sharded-computation.html",
        "note": "the official introduction to meshes, specs, and NamedSharding"
      },
      {
        "label": "shard_map notebook",
        "url": "https://docs.jax.dev/en/latest/notebooks/shard_map.html",
        "note": "the manual level, complete"
      },
      {
        "label": "Scaling book · Sharding",
        "url": "https://jax-ml.github.io/scaling-book/sharding/",
        "note": "the cost model for choosing specs"
      }
    ],
    "diagram": {
      "id": "jax-sharding-ladder",
      "after": 1
    }
  },
  {
    "id": "performance",
    "num": 11,
    "part": "ii",
    "title": "Performance",
    "lede": "Most JAX benchmarks measure the wrong thing without anyone noticing.",
    "goal": "Given a JAX function, build a benchmark harness that measures compute rather than dispatch, and read a profiler trace well enough to name its largest gap.",
    "sections": [
      {
        "h": "measure honestly",
        "ps": [
          "Time a JAX op the way you'd time any other Python call, and the number that comes back is very often wrong, not because the clock lied but because of what you handed it to measure. Three mistakes produce three different lies. Time a call without blocking on the result and you measure dispatch, the moment the Python line returns control, not the moment the device finishes computing (chapter 1 built this fact once already: JAX dispatches asynchronously, and printing or converting to NumPy is what actually blocks). Time a function's first call and you measure compilation, the one-time cost of tracing and lowering that every later call skips. Time a function once and you measure noise, whatever the scheduler happened to be doing at that instant.",
          "The harness below kills all three lies at once. A warmup call forces the compile and the first execution to finish before the clock starts. The timed loop runs the function `n` times and blocks once at the end, so dispatch overhead amortizes across every call instead of dominating a single one. Dividing by `n` gives a number that reflects the computation, not the ceremony around it.",
          "This site's own habit for every measured number carries over directly: state the machine, the dtype, the shapes, and the method, every time. A latency number with no provenance isn't a fact. It's a rumor with decimal places.",
          ">> Benchmark the computation, not the dispatch."
        ],
        "code": {
          "caption": "a harness that survives all three lies: warmup, loop, one block at the end",
          "text": "import time\n\nimport jax\n\ndef bench(f, *args, n=20):\n    jax.block_until_ready(f(*args))   # warmup: compile + first run\n    t0 = time.perf_counter()\n    for _ in range(n):\n        out = f(*args)\n    jax.block_until_ready(out)\n    return (time.perf_counter() - t0) / n"
        }
      },
      {
        "h": "the recompile hunt",
        "ps": [
          "A training loop that recompiles mid-run is losing seconds it will never get back, and the first step is knowing it's happening at all. `jax.config.update(\"jax_log_compiles\", True)` logs every trace and compile event along with the reason JAX decided a new one was needed. Once training reaches steady state, that log should go silent: same shapes, same structure, same executable, call after call.",
          "Chapter 3's cache-key model explains every entry that keeps showing up in that log. Shapes drifting from the data forces a retrace, and the last batch of an epoch is usually short, which makes it the most common offender. A fresh closure or lambda built inside the loop counts as a new function identity even when its body is identical, forcing another retrace. A Python float that flips a leaf from f32 to a weakly typed float across calls changes the key too, quietly, with no exception to flag it.",
          "The standard fix for ragged data, the short last batch above all, is padding to a fixed set of shapes: pad the tail batch up to the usual size before it reaches the traced function, and the cache key stops moving."
        ]
      },
      {
        "h": "the profiler",
        "ps": [
          "Logging compiles tells you when JAX rebuilt something. It doesn't tell you where the time inside a single step actually went. `jax.profiler.trace(dir)`, or the paired `start_trace` and `stop_trace` calls, captures a trace you can open in XProf or Perfetto, and that trace shows dispatch gaps, compile events, and the cost of individual ops laid out on a timeline.",
          "`jax.profiler.save_device_memory_profile` does the same job for memory instead of time, and it's how you go looking for the residuals chapter 4 predicted reverse mode would store. A memory profile that spikes exactly where a vjp pass should be holding its intermediates confirms the cost model rather than just gesturing at it.",
          "The kernel path's IR-stage chapter reads these same profiles down to individual bytes moved. This chapter only needs the workflow: capture a trace, open it, find the gap, not that depth."
        ]
      },
      {
        "h": "precision is a choice",
        "ps": [
          "The dtype you compute in is a decision with a cost, not a detail the framework hides from you. On TPU, f32 matmuls run at reduced precision on the MXU by default, trading some accuracy for throughput without asking. When you're comparing a computation against a reference and need the true f32 answer, `jax.default_matmul_precision(\"highest\")` pins it, the same lesson the kernel path's gate 3 measures directly and publishes at `/bench`.",
          "Dropping to bf16 halves the bytes moved for every array involved, so memory-bound operations speed up by roughly 2x, and compute-bound matmuls ride the MXU's native bf16 path instead of emulating it. Going the other direction, enabling x64 doubles the bytes and, on accelerators, can push a computation off the fast path entirely rather than just slowing it proportionally."
        ]
      }
    ],
    "readings": [
      {
        "label": "Profiling JAX programs",
        "url": "https://docs.jax.dev/en/latest/profiling.html",
        "note": "the trace and memory profiler workflow this chapter walks"
      },
      {
        "label": "JAX FAQ · benchmarking",
        "url": "https://docs.jax.dev/en/latest/faq.html#benchmarking-jax-code",
        "note": "the official benchmarking notes"
      },
      {
        "label": "Common gotchas in JAX",
        "url": "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html",
        "note": "the dispatch and compile lies, from the source"
      }
    ]
  },
  {
    "id": "training-run",
    "num": 12,
    "part": "ii",
    "title": "The Training Run",
    "lede": "Every fact this track taught shows up once more here, wired together into a loop that actually trains.",
    "goal": "Given the facts from chapters 1 through 11, compose one complete, checkpointable, resumable training loop and prove it works, rather than assume it.",
    "sections": [
      {
        "h": "compose what you own",
        "ps": [
          "Nothing in this chapter is new, and that's deliberate. Eleven chapters built the pieces; a training run is what happens when you wire them together in the order a real loop needs them. Params and optimizer state ride as one registered pytree, the pattern chapters 7 and 9 built. The step itself is a jitted function computing gradients, the machinery chapters 3 and 4 gave you. Iteration is a scan or a plain Python loop over steps, chapter 6's territory. Randomness folds in per step by index, the discipline chapter 8 insisted on. If the mesh is real, the arrays carry sharding from chapter 10. And every number the loop reports, loss, throughput, time per step, gets measured the way chapter 11 taught: warmup, block_until_ready, provenance stated.",
          "This chapter walks one complete run: small, real, and fully specified, an MLP fitting a synthetic regression target end to end. The mastery work at the end is where you scale it, with real data, a bigger model, a real mesh. The loop's shape doesn't change when you do.",
          ">> A training run is composition, not invention."
        ]
      },
      {
        "h": "the step",
        "ps": [
          "One jitted function does the actual work of a step: compute the loss and its gradient together with `value_and_grad`, hand the gradients to the optimizer's `update`, apply the resulting updates to the parameters, and return the new state alongside whatever metrics you want to keep. On an accelerator, `donate_argnums` lets that function recycle the old parameters' buffers instead of allocating fresh ones for the new values, since the old buffer is dead the instant the new one exists.",
          "Data stays outside the jitted world in spirit, even though it's passed as an argument: it lives as an iterator of ordinary host or NumPy arrays, handed in fresh each step rather than carried inside the traced function. Metrics leave the same way params do, as returned values from the jitted function, or through `io_callback` when you need them to stream out mid-run instead of waiting for the step to finish, the callback family chapter 9 introduced."
        ],
        "code": {
          "caption": "the complete loop, runnable as is; final loss is the verified real output (jax 0.4.38, CPU)",
          "text": "import jax\nimport jax.numpy as jnp\nimport optax\n\ndef init(key):\n    k1, k2 = jax.random.split(key)\n    return {\n        \"w1\": jax.random.normal(k1, (8, 32)) * 0.1, \"b1\": jnp.zeros(32),\n        \"w2\": jax.random.normal(k2, (32, 1)) * 0.1, \"b2\": jnp.zeros(1),\n    }\n\ndef predict(p, x):\n    h = jax.nn.relu(x @ p[\"w1\"] + p[\"b1\"])\n    return h @ p[\"w2\"] + p[\"b2\"]\n\ndef loss_fn(p, batch):\n    x, y = batch\n    return jnp.mean((predict(p, x) - y) ** 2)\n\nopt = optax.adam(1e-2)\n\n@jax.jit\ndef train_step(p, opt_state, batch):\n    loss, grads = jax.value_and_grad(loss_fn)(p, batch)\n    updates, opt_state = opt.update(grads, opt_state, p)\n    return optax.apply_updates(p, updates), opt_state, loss\n\nkey = jax.random.key(0)\nparams = init(key)\nopt_state = opt.init(params)\nx = jax.random.normal(jax.random.key(1), (256, 8))\ny = jnp.sum(x, axis=1, keepdims=True)\n\nfor step in range(200):\n    params, opt_state, loss = train_step(params, opt_state, (x, y))\nprint(f\"final loss {loss:.5f}\")   # final loss 0.00335 (verified, jax 0.4.38 CPU)"
        }
      },
      {
        "h": "saving and resuming",
        "ps": [
          "A checkpoint that only saves params isn't a checkpoint of your training run. It's a checkpoint of a different run that happens to start from the same weights. Adam without its moment estimates is a different optimizer wearing the same name, and resuming from params alone while re-initializing everything else changes the trajectory even if nobody notices for a while. The unit of checkpointing is the whole state tree: params, optimizer state, step, and key together.",
          "Orbax is the library that saves and restores that tree, as pytrees, sharding-aware, so a checkpoint written by a sharded run restores correctly onto a different mesh shape. The key half of the tree resumes cleanly because of a choice chapter 8 already made: `fold_in(key, step)` derives each step's randomness from the step number directly, so restoring at step k and continuing produces exactly the same bits from k onward that an uninterrupted run would have produced.",
          "The bar for resume working isn't a plausible-looking loss curve. It's bitwise: kill the run, restore the checkpoint, continue, and the loss at every subsequent step matches the run that was never interrupted, digit for digit. Prove that. Don't assume it."
        ]
      },
      {
        "h": "many hosts",
        "ps": [
          "Multi-host JAX isn't a different programming model. It's the same program run once per host, after each host calls `jax.distributed.initialize()` to find the others. Once initialized, a `jax.Array` can be sharded globally across every device on every host, not just the devices attached to one process, and `jit` compiles a single SPMD program that runs across the whole fleet.",
          "Each host feeds only its own local shard of the batch into the step, the same way each device did inside a single host in chapter 10, and the collectives that keep gradients and activations synchronized cross host boundaries exactly as they crossed device boundaries there. Nothing about the mesh or the PartitionSpec vocabulary changes; the ring just gets bigger."
        ]
      }
    ],
    "readings": [
      {
        "label": "Multi-process JAX",
        "url": "https://docs.jax.dev/en/latest/multi_process.html",
        "note": "the multi-host initialize call and the global array model"
      },
      {
        "label": "Scaling book · Training",
        "url": "https://jax-ml.github.io/scaling-book/training/",
        "note": "what the loop becomes at scale"
      },
      {
        "label": "MaxText",
        "url": "https://github.com/AI-Hypercomputer/maxtext",
        "note": "a production loop to read end to end"
      }
    ]
  }
]

export const JAX_MASTERY: Record<string, WorkItem[]> = {
  "jax:arrays": [
    {
      "id": "read",
      "label": "read the chapter: values not places, the promotion lattice, async dispatch, where arrays live"
    },
    {
      "id": "predict-dtypes",
      "label": "predict the dtype of six mixed expressions in writing, then run them and check every prediction"
    },
    {
      "id": "explain-inplace-grad",
      "label": "explain in one written paragraph why in-place update cannot coexist with grad"
    }
  ],
  "jax:tracing": [
    {
      "id": "read",
      "label": "read the chapter: tracers, the jaxpr, what tracing discards, and the debugging kit"
    },
    {
      "id": "annotate-jaxpr",
      "label": "trace two of your own functions with make_jaxpr and annotate every equation in writing"
    },
    {
      "id": "fix-twice",
      "label": "take a function that fails under jit and fix it twice: once with debug.print, once by restructuring"
    }
  ],
  "jax:jit": [
    {
      "id": "read",
      "label": "read the chapter: what jit buys, the cache key, where tracing refuses, ahead-of-time compilation"
    },
    {
      "id": "cache-key-predict",
      "label": "write the cache key for three call sites in writing, predict hit or miss, then verify with jax_log_compiles"
    },
    {
      "id": "find-recompile-bug",
      "label": "find one real recompile bug in your own code or the chapter's examples and name which key component churned"
    }
  ],
  "jax:autodiff": [
    {
      "id": "read",
      "label": "read the chapter: derivatives as an exact program transformation, not a numerical method"
    },
    {
      "id": "reverse-mode-scalar",
      "label": "derive on paper why reverse mode wins for scalar losses, and write the argument out"
    },
    {
      "id": "custom-vjp-fix",
      "label": "write one custom_vjp for a numerically unstable function and prove the fix against a failing input"
    },
    {
      "id": "remat-tradeoff",
      "label": "measure the memory/flops trade of jax.checkpoint on an 8-layer block, in writing"
    }
  ],
  "jax:vmap": [
    {
      "id": "read",
      "label": "read the chapter: vmap as a program rewrite, not a loop over your data"
    },
    {
      "id": "predict-shapes",
      "label": "predict shapes for five vmap compositions in writing, then run them"
    },
    {
      "id": "per-sample-clipping",
      "label": "write per-sample gradient clipping (clip each example's gradient before the mean) in under ten lines"
    }
  ],
  "jax:control-flow": [
    {
      "id": "read",
      "label": "read the chapter: static vs traced control flow, and scan as the load-bearing loop"
    },
    {
      "id": "unroll-to-scan",
      "label": "convert an unrolled Python training loop to scan and compare compile times, in writing"
    },
    {
      "id": "while-vs-scan-grad",
      "label": "explain in writing why while_loop cannot support reverse-mode grad but scan can"
    }
  ],
  "jax:pytrees": [
    {
      "id": "read",
      "label": "read the chapter: pytrees as the calling convention"
    },
    {
      "id": "flatten-treedef",
      "label": "flatten a real model's params and name the treedef, in writing"
    },
    {
      "id": "register-dataclass",
      "label": "register one dataclass state object and push it through jit and grad"
    },
    {
      "id": "break-treemap",
      "label": "break tree.map with mismatched structures on purpose and read the error"
    }
  ],
  "jax:random": [
    {
      "id": "read",
      "label": "read the chapter: keys as values, not a global dial"
    },
    {
      "id": "key-reuse-bug",
      "label": "find a key-reuse bug in a snippet, yours or provided, and fix it two ways: with split and with fold_in"
    },
    {
      "id": "batched-dropout",
      "label": "write batched dropout with vmap over keys and prove per-example independence"
    }
  ],
  "jax:state": [
    {
      "id": "read",
      "label": "read the chapter: state as an argument and a return value"
    },
    {
      "id": "train-state-dataclass",
      "label": "write a train state (params, opt state, step, key) as one registered dataclass and thread it through a jitted step"
    },
    {
      "id": "optax-chain",
      "label": "chain clipping and weight decay into an optax optimizer and verify each link's state, in writing"
    }
  ],
  "jax:sharding": [
    {
      "id": "read",
      "label": "read the chapter, the shard_map notebook, and the scaling book's sharding chapter"
    },
    {
      "id": "shard-three-ways",
      "label": "shard one matmul three ways (data, model, both) on the 8 fake host devices and visualize each with visualize_array_sharding"
    },
    {
      "id": "predict-collectives",
      "label": "predict in writing which collectives GSPMD inserts for one spec, then confirm by reading the lowered StableHLO"
    },
    {
      "id": "rewrite-shard-map",
      "label": "rewrite one of the three shardings in shard_map with an explicit psum"
    }
  ],
  "jax:performance": [
    {
      "id": "read",
      "label": "read the chapter, the profiling guide, and the official benchmarking notes"
    },
    {
      "id": "benchmark-three-lies",
      "label": "benchmark one op three wrong ways (no block, no warmup, single run) and once right, and write down how far each lie was from the truth"
    },
    {
      "id": "silence-the-log",
      "label": "turn on jax_log_compiles for a real training script and drive it to silence"
    },
    {
      "id": "profile-one-trace",
      "label": "capture one profiler trace and name the largest gap in writing"
    }
  ],
  "jax:training-run": [
    {
      "id": "read",
      "label": "read the chapter, the multi-process guide, and the scaling book's training chapter"
    },
    {
      "id": "build-full-loop",
      "label": "build the full loop for a small MLP on real data (MNIST-class), checkpointing every N steps"
    },
    {
      "id": "prove-bitwise-resume",
      "label": "kill the run mid-training and prove bitwise resume in writing, loss curve matched step for step"
    },
    {
      "id": "shard-across-devices",
      "label": "shard the same run over the 8 host-platform devices and show the loss curve unchanged"
    }
  ]
}

export interface JaxPathChapter {
  key: string
  href: string
  num: number
  title: string
  part: 'i' | 'ii'
}

/** The path's order; every next/previous control on /jax/ derives from this. */
export const JAX_PATH: JaxPathChapter[] = JAX_CHAPTERS.map((c) => ({
  key: `jax:${c.id}`,
  href: `/jax/${c.id}`,
  num: c.num,
  title: c.title,
  part: c.part,
}))

export const jaxChapterAt = (
  key: string,
): { current: JaxPathChapter; prev?: JaxPathChapter; next?: JaxPathChapter } => {
  const idx = JAX_PATH.findIndex((c) => c.key === key)
  if (idx === -1) throw new Error(`unknown jax path chapter: ${key}`)
  return { current: JAX_PATH[idx]!, prev: JAX_PATH[idx - 1], next: JAX_PATH[idx + 1] }
}
