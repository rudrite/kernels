// New file: site/src/data/lessons/jax-tracing.ts
// Tracers below the survey chapter 2 teaches: the abstract value a tracer
// carries, the calls it refuses and the four error classes those refusals
// come from, and the difference between a value passed in and a value the
// closure froze. Every printed value ran on this machine (jax 0.4.38 CPU,
// Python 3.12, 2026-08-15) and is quoted verbatim; snippets were run through
// python3 -c, which is why the trace-site lines in the error text name
// <string> rather than a file.
import type { UnitLessons } from './index'

export const JAX_TRACING_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:tracing',
    lessons: [
      {
        id: 'shape-dtype-and-no-numbers',
        num: 1,
        title: 'Shape, dtype, and no numbers',
        lede: 'The stand-in that walks through your function has a class name, a type of its own, and a short list of things it knows. Print all three and tracing stops being a metaphor.',
        goal: 'Name the tracer class a given transform installs, read any printed aval and say what the trace knows from it, and use eval_shape to answer a shape question without allocating an array or compiling anything.',
        sections: [
          {
            h: 'what shows up instead of your array',
            ps: [
              "Put a `print(type(x))` at the top of a jitted function and the class that comes back is `DynamicJaxprTracer`. The chapter above says JAX calls your function with stand-ins; this is the stand-in, named, and the print fires at trace time for the reason that chapter already gave.",
              "Ask the tracer for its `aval` and it answers `ShapedArray(float32[3])`. That object is an abstract value: a shape, a dtype, and a couple of flags, with no storage behind it. Two arrays holding completely different numbers at the same shape and dtype have the same aval, so one recording serves both.",
              "Plenty of what you would ask a real array still works, and it is worth being exact about why. `x.shape` gives a tuple of ordinary Python ints, `len(x)` gives an int, `x.dtype` gives a dtype. Each of those answers was already sitting in the aval, so nothing had to be computed and nothing had to exist.",
              "The last line of the run is the one to carry into the next lesson. `x.sum() > 0` inside the trace has the aval `ShapedArray(bool[])`, and `bool[]` is exactly the phrase the branch error prints back at you. That message is not describing your data. It is printing the aval.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): what the stand-in knows about itself; on jax 0.8.2 and later a Tracer no longer inherits from jax.Array at runtime, though isinstance still answers True (jax changelog, 0.8.2)',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef probe(x):\n    print(type(x).__name__)\n    print(x.aval, x.aval == jax.core.ShapedArray((3,), jnp.float32))\n    print(x.shape, x.dtype, x.ndim, len(x))\n    print((x.sum() > 0).aval)\n    print(isinstance(x, jax.Array))\n    return x\n\nprobe(jnp.ones(3))\n\n# DynamicJaxprTracer\n# ShapedArray(float32[3]) True\n# (3,) float32 1 3\n# ShapedArray(bool[])\n# True',
            },
          },
          {
            h: 'the aval is the whole vocabulary',
            ps: [
              "Everything the trace can reason about a value is in its aval, and the aval is small. A shape as a tuple of ints, a dtype, and a `weak_type` flag that records whether the value came from a Python scalar rather than an array. Nothing else about the value survives into the recording.",
              "That `weak_type` flag is where the promotion rule from chapter 1 is stored. Pass the Python float `2.0` into a jitted function and its aval prints as `ShapedArray(float32[], weak_type=True)`; pass `jnp.float32(2.0)` and the flag is gone. Chapter 1 owns what the flag does to a promotion, and chapter 3 owns what it does to a cache key. Here it is one more field of an abstract value, and it is why two calls that look identical in Python can present different avals.",
              "Comparing avals is the cheapest test of the mental model there is. `x.aval == jax.core.ShapedArray((3,), jnp.float32)` answers True in the run above, which means you can write down what the trace sees before you run anything, then check yourself in one line.",
            ],
            table: {
              caption: 'four printed avals from the runs in this lesson (verified, jax 0.4.38 CPU)',
              cols: ['printed form', 'where it came from', 'what the trace knows'],
              rows: [
                ['ShapedArray(float32[3])', 'the argument of a jitted function, called with jnp.ones(3)', 'rank 1, three elements, float32, and no values at all'],
                ['ShapedArray(bool[])', 'x.sum() > 0 inside that same trace', 'a scalar boolean that will exist at run time; the phrase the branch error prints'],
                ['ShapedArray(float32[], weak_type=True)', 'the Python float 2.0 passed into a jitted function', 'a scalar whose dtype yields to whatever array it meets'],
                ['ShapeDtypeStruct(shape=(8,), dtype=float32)', 'jax.eval_shape over a two-op function', 'the same information, in the public struct eval_shape hands back'],
              ],
            },
          },
          {
            h: 'tracing with the arrays left out',
            ps: [
              "If the trace only ever sees shapes and dtypes, you can run one without owning the arrays. `jax.eval_shape(f, *args)` does exactly that: it traces `f`, computes the output avals, and allocates nothing, so the arguments can be `ShapeDtypeStruct` descriptions instead of real data.",
              "The use for that is practical rather than decorative. A model whose parameters would not fit on the machine in front of you still answers shape questions in milliseconds, and a shape bug that would have surfaced after a long allocation surfaces before it. No compile happens either, which chapter 3 measures the cost of.",
              "One thing does carry over, and it is the subject of the next lesson. `eval_shape` runs your Python once, exactly as `jit` does, so a function that refuses to trace refuses here too. Cheap tracing is still tracing.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): output shapes with no arrays anywhere, and the weak-type flag on a Python scalar',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nfrom jax import ShapeDtypeStruct\n\ndef block(x, w):\n    return jnp.tanh(x @ w).sum(axis=1)\n\nprint(jax.eval_shape(block, ShapeDtypeStruct((8, 16), jnp.float32),\n                            ShapeDtypeStruct((16, 4), jnp.float32)))\nprint(jax.eval_shape(lambda x: x, 2.0))\nprint(jax.eval_shape(lambda x: x, jnp.float32(2.0)))\n\n# ShapeDtypeStruct(shape=(8,), dtype=float32)\n# ShapeDtypeStruct(shape=(), dtype=float32, weak_type=True)\n# ShapeDtypeStruct(shape=(), dtype=float32)',
            },
          },
          {
            h: 'one tracer per transform, and they stack',
            ps: [
              "`jit` is not the only thing that traces, and the three transforms do not send the same object through your function. Under `jit` you get a `DynamicJaxprTracer`, under `vmap` a `BatchTracer`, under `grad` a `JVPTracer`. Same interface, three different jobs.",
              "Look at what `vmap` reports and the design shows through. The input was `(2, 3)` and the aval inside the function is `float32[3]`, one row, because `vmap` hides the mapped axis from the code it wraps. Your function is written for one example and the batch axis is bookkeeping that happens around it.",
              "Stack all three and the stand-ins nest in the order the transforms were applied. `jit(vmap(grad(f)))` hands your function a `JVPTracer` whose primal is a `BatchTracer` whose value is a `DynamicJaxprTracer`, and only that innermost one still carries the full `float32[2,3]`. Unwrapping the stack by hand is a two-line trick and the fastest way to see which transform is currently in charge.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): three transforms, three stand-ins, unwrapped one layer at a time',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef who(x):\n    print(type(x).__name__, x.aval)\n    print(type(x.primal).__name__, x.primal.aval)\n    inner = x.primal.val\n    print(type(inner).__name__, inner.aval)\n    return jnp.sum(x)\n\njax.jit(jax.vmap(jax.grad(who)))(jnp.ones((2, 3)))\n\n# JVPTracer ShapedArray(float32[3])\n# BatchTracer ShapedArray(float32[3])\n# DynamicJaxprTracer ShapedArray(float32[2,3])',
            },
            table: {
              caption: 'which stand-in each transform installs, and the aval it shows inside the function (verified, jax 0.4.38 CPU)',
              cols: ['transform', 'class inside the function', 'aval it reports'],
              rows: [
                ['jax.jit(f)', 'DynamicJaxprTracer', 'float32[3], the argument as it was passed'],
                ['jax.vmap(f)', 'BatchTracer', 'float32[3], one row of the (2, 3) input'],
                ['jax.grad(f)', 'JVPTracer', 'float32[3], the primal side of the pair'],
                ['jax.jit(jax.vmap(jax.grad(f)))', 'JVPTracer over BatchTracer over DynamicJaxprTracer', 'float32[3], then float32[3], then float32[2,3]'],
              ],
            },
          },
        ],
        readings: [
          { label: 'jax.eval_shape', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html', note: 'the whole API in one page, including what it promises not to do' },
          { label: 'jax.ShapeDtypeStruct', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.ShapeDtypeStruct.html', note: 'the public stand-in for an array you do not want to allocate' },
          { label: 'Type promotion semantics', url: 'https://docs.jax.dev/en/latest/type_promotion.html', note: 'where the weak_type flag on an aval comes from, and the lattice it feeds' },
        ],
        check: [
          {
            q: 'Inside a jitted function, x.shape answers instantly while float(x.sum()) raises. What separates the two calls?',
            a: 'The shape is already in the aval, so answering costs nothing and needs no data. A float conversion asks for a number, and the aval holds no numbers, so there is nothing to convert.',
          },
          {
            q: 'Two calls pass arrays holding completely different numbers at the same shape and dtype. What does the trace see?',
            a: 'The same aval both times, so the same recording serves both calls. Values never enter the trace; a shape, a dtype and the weak_type flag are the whole vocabulary.',
          },
          {
            q: 'Under vmap the function prints an aval of float32[3] when the input was (2, 3). Where did the other axis go?',
            a: 'vmap hides the mapped axis from the wrapped function, so the code inside is written for one example. The batch axis reappears one layer out, on the tracer belonging to whatever transform sits above vmap.',
          },
        ],
        work: [
          { id: 'aval-ledger', label: 'print the aval of every intermediate in one function of your own, then predict the aval of the next line you add before you run it', href: '#the-aval-is-the-whole-vocabulary' },
          { id: 'eval-shape-first', label: 'answer one shape question about the largest model you have with eval_shape, before allocating anything', href: '#tracing-with-the-arrays-left-out' },
        ],
      },
      {
        id: 'what-a-tracer-refuses',
        num: 2,
        title: 'What a tracer refuses',
        lede: 'Six calls that all mean the same thing to a Python programmer produce four different error classes, and the class you get tells you which method the tracer was asked for.',
        goal: 'Predict which of the four tracer error classes a given line raises, read the message down to the method it names, and recognize the three refusals that arrive as plain ValueError and TypeError instead.',
        sections: [
          {
            h: 'the questions that cost nothing',
            ps: [
              "Before the refusals, the permissions, because they are wider than people expect. Shape, length, dtype and rank all answer from the aval. Arithmetic on those answers is ordinary Python arithmetic, so `jnp.zeros(len(x) * 2)` builds a `(6,)` array inside a trace with nothing traced about it at all.",
              "Anything that produces a new traced value is fine too. `x.astype(jnp.int32)` records a convert equation, `x.at[0].set(9)` records a scatter, and both hand back another tracer for the next line to use. The recording grows and nothing needs a number.",
              "The line between the two groups is one question: does this call need to know a value right now, in Python, before the trace can continue? Shape arithmetic does not. A branch does.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): four answers from the aval, one shape built from them, and two ops that record instead of asking',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef free(x):\n    print(x.shape, len(x), x.dtype, x.ndim)\n    print(jnp.zeros(len(x) * 2).shape)\n    return x.astype(jnp.int32).at[0].set(9)\n\nprint(free(jnp.ones(3)))\n\n# (3,) 3 float32 1\n# (6,)\n# [9 1 1]',
            },
          },
          {
            h: 'six calls, four error classes',
            ps: [
              "The six lines below all ask a tracer for a value, and JAX distinguishes them by which Python method got called. An `if` reaches `__bool__` and raises `TracerBoolConversionError`. A `float()` or an `.item()` reaches the concretization path and raises `ConcretizationTypeError`. Handing a tracer to NumPy reaches `__array__` and raises `TracerArrayConversionError`. Indexing a Python list with a tracer reaches `__index__` and raises `TracerIntegerConversionError`.",
              "Two entries in that run are worth pausing on. `min(x[0], x[1])` never mentions a branch in your source, and it raises the branch error, because `min` compares its arguments and a comparison of tracers has to be resolved to a Python bool. Anything built on comparison lands in the same place: `max`, `sorted` with a key that compares traced values, a `while` condition, an `assert`.",
              "The other one is `.item()`, which people reach for precisely when they want to look at a number. It shares a class with `float()` rather than getting one of its own, and the message says which method asked, so the class narrows the cause and the first line finishes the job.",
              '>> The class of the error names the method that asked. The message names the line.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): six refusals, first line of each message quoted as printed',
              lang: 'python',
              text: "import jax\nimport jax.numpy as jnp\nimport numpy as np\n\ndef refuse(name, fn):\n    try:\n        jax.jit(fn)(jnp.ones(3))\n    except Exception as e:\n        print(name, type(e).__name__, '|', str(e).splitlines()[0])\n\nrefuse('if      ', lambda x: x if x.sum() > 0 else -x)\nrefuse('float() ', lambda x: float(x.sum()))\nrefuse('.item() ', lambda x: x.sum().item())\nrefuse('np.array', lambda x: np.asarray(x))\nrefuse('index   ', lambda x: [1.0, 2.0][x.sum().astype(int)])\nrefuse('min()   ', lambda x: min(x[0], x[1]))\n\n# if       TracerBoolConversionError | Attempted boolean conversion of traced array with shape bool[].\n# float()  ConcretizationTypeError | Abstract tracer value encountered where concrete value is expected: traced array with shape float32[]\n# .item()  ConcretizationTypeError | Abstract tracer value encountered where concrete value is expected: traced array with shape float32[]\n# np.array TracerArrayConversionError | The numpy.ndarray conversion method __array__() was called on traced array with shape float32[3]\n# index    TracerIntegerConversionError | The __index__() method was called on traced array with shape int32[]\n# min()    TracerBoolConversionError | Attempted boolean conversion of traced array with shape bool[].",
            },
            table: {
              caption: 'the six calls above, by the method they reach (verified, jax 0.4.38 CPU); the fix column names where each one is taught, not what this lesson teaches',
              cols: ['the call', 'method reached', 'error class', 'where the fix lives'],
              rows: [
                ['if x.sum() > 0', '__bool__', 'TracerBoolConversionError', 'chapter 6, lax.cond; the museum exhibit tracer-bool'],
                ['min(x[0], x[1])', '__bool__ through the comparison', 'TracerBoolConversionError', 'jnp.minimum, which records instead of comparing'],
                ['float(x.sum())', 'the concretization path', 'ConcretizationTypeError', 'x.astype(float), which the message itself suggests'],
                ['x.sum().item()', 'the item() method of jax.Array', 'ConcretizationTypeError', 'return the value and call .item() outside the trace'],
                ['np.asarray(x)', '__array__', 'TracerArrayConversionError', 'keep the value in jnp, or move the NumPy call outside'],
                ['[1.0, 2.0][i]', '__index__', 'TracerIntegerConversionError', 'jnp.take or a lax switch on a traced index'],
              ],
            },
          },
          {
            h: 'the message names the function that asked',
            ps: [
              "Read one of those messages in full and it has four parts doing four jobs. The first line states the abstract-value problem and prints the aval. The second names the Python function that asked, `float` here, and offers the conversion that would have worked. The third points at the traced function and the line it was defined on. The fourth is a link to the error's own documentation page.",
              "The third line is the one people skim and the one that saves the most time on a large program, because it names the function being traced rather than the frame you happened to be looking at. When a refusal fires four calls deep inside a library, that line is what tells you whose trace you are inside.",
              "The message also states, plainly, why the value was unavailable: it depends on the value of the argument. That is the distinction chapter 3 turns into a decision about `static_argnums`, and reading it here is what makes that decision obvious later.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one ConcretizationTypeError, verbatim; the trace-site line names <string> because this ran through python3 -c, and names your file and line when it comes from a file',
              lang: 'text',
              text: 'Abstract tracer value encountered where concrete value is expected: traced array with shape float32[]\nThe problem arose with the `float` function. If trying to convert the data type of a value, try using `x.astype(float)` or `jnp.array(x, float)` instead.\nThe error occurred while tracing the function scale at <string>:4 for jit. This concrete value was not available in Python because it depends on the value of the argument x.\n\nSee https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError\n\n# produced by:\n# import jax\n# import jax.numpy as jnp\n#\n# @jax.jit\n# def scale(x):\n#     return x / float(x.sum())\n#\n# try:\n#     scale(jnp.ones(3))\n# except jax.errors.ConcretizationTypeError as e:\n#     print(e)',
            },
          },
          {
            h: 'three refusals that never say tracer',
            ps: [
              "Not every failure inside a trace comes from the tracer error family, and the ones that do not are the ones that waste an afternoon. `bool(x)` on a three-element array raises a plain `ValueError` about ambiguous truth values, the same sentence NumPy raises, because the shape check runs first and the shape is known. Rank is not the problem the trace has, so that check answers before the value check ever runs.",
              "Formatting is the second. Putting a traced scalar in an f-string raises `TypeError: unsupported format string passed to DynamicJaxprTracer.__format__`, which is the only refusal in this lesson that names the tracer class in the message and gives no advice at all.",
              "The third is the one that shows up in real code most often. Ask for an array whose size comes from a traced value and the shape machinery refuses before any tracer method is reached: `Shapes must be 1D sequences of concrete values of integer type`, with the tracer printed inside the tuple it was found in. The museum's boolean-mask exhibit is the same wall approached from the other side.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): three failures inside a trace that are not tracer errors, first line of each message quoted as printed',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef fails(fn):\n    try:\n        jax.jit(fn)(jnp.ones(3))\n    except Exception as e:\n        print(type(e).__name__, \'|\', str(e).splitlines()[0])\n\nfails(lambda x: x if bool(x) else -x)\nfails(lambda x: f"{x.sum():.2f}")\nfails(lambda x: jnp.zeros(x.sum().astype(int)))\n\n# ValueError | The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()\n# TypeError | unsupported format string passed to DynamicJaxprTracer.__format__\n# TypeError | Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[])>with<DynamicJaxprTrace>,).',
            },
          },
          {
            h: 'one family tree under all of them',
            ps: [
              "The four tracer error classes are not siblings in the way the messages suggest. `TracerBoolConversionError` is a subclass of `ConcretizationTypeError`, so catching the general one catches the branch failure too. `TracerIntegerConversionError` and `TracerArrayConversionError` are not: they sit directly under `JAXTypeError`, alongside `ConcretizationTypeError` rather than beneath it.",
              "Everything in the tree bottoms out at Python's `TypeError`, `UnexpectedTracerError` from the next lesson included. A bare `except TypeError` around a traced call therefore swallows every refusal in this lesson, which is a good reason not to write one.",
              "For a fixture that has to survive any refusal, `jax.errors.JAXTypeError` is the one class that covers all five without reaching outside JAX.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the first three bases of each error class, printed from the class objects themselves',
              lang: 'python',
              text: 'import jax\n\nfor name in ["TracerBoolConversionError", "TracerIntegerConversionError",\n             "TracerArrayConversionError", "ConcretizationTypeError",\n             "UnexpectedTracerError"]:\n    cls = getattr(jax.errors, name)\n    print(name, "<-", " <- ".join(b.__name__ for b in cls.__mro__[1:4]))\n\n# TracerBoolConversionError <- ConcretizationTypeError <- JAXTypeError <- _JAXErrorMixin\n# TracerIntegerConversionError <- JAXTypeError <- _JAXErrorMixin <- TypeError\n# TracerArrayConversionError <- JAXTypeError <- _JAXErrorMixin <- TypeError\n# ConcretizationTypeError <- JAXTypeError <- _JAXErrorMixin <- TypeError\n# UnexpectedTracerError <- JAXTypeError <- _JAXErrorMixin <- TypeError',
            },
          },
        ],
        readings: [
          { label: 'JAX errors', url: 'https://docs.jax.dev/en/latest/errors.html', note: 'every class in this lesson, with the official one-paragraph cause for each' },
          { label: 'Control flow and logical operators with JIT', url: 'https://docs.jax.dev/en/latest/control-flow.html', note: 'the structured replacements for the branch that raised' },
          { label: 'jax.numpy.ndarray.at', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html', note: 'the update syntax that records instead of asking for a value' },
        ],
        check: [
          {
            q: 'min(x[0], x[1]) raises TracerBoolConversionError even though the line contains no if. Why?',
            a: 'min compares its arguments, and comparing two tracers produces a traced boolean that Python has to resolve right now. Every builtin that compares lands on the same method and the same error, max and sorted included.',
          },
          {
            q: 'One line raises TracerArrayConversionError and another raises ConcretizationTypeError. What does the difference tell you?',
            a: 'Which Python method was reached. The first means something called __array__, so a NumPy conversion is in the path; the second means a scalar conversion such as float, int or .item() asked for a value the aval does not hold.',
          },
          {
            q: 'jnp.zeros(n) with a traced n raises a plain TypeError rather than a tracer error. What ran first?',
            a: 'The shape check, which requires concrete integers and refuses before any tracer method is reached. Its message prints the tracer inside the tuple it was found in, and the fix is a static bound rather than a different conversion.',
          },
        ],
        work: [
          { id: 'predict-six', label: 'write six lines that each ask a tracer for a value, predict the error class of each in writing, then run them', href: '#six-calls-four-error-classes' },
          { id: 'read-one-whole', label: 'take one refusal from your own code and account for all four parts of its message before you change a line', href: '#the-message-names-the-function-that-asked' },
        ],
      },
      {
        id: 'frozen-passed-or-escaped',
        num: 3,
        title: 'Frozen, passed, or escaped',
        lede: 'A jitted function reads a closed-over array once, keeps a copy, and never looks again. The same rule, seen from the other end, is why a tracer stored in a list raises when you touch it later.',
        goal: 'Predict what a jitted function returns after a closed-over value changes, say what a closed-over scalar and a closed-over array each become in the recording, and read an UnexpectedTracerError down to the line that leaked.',
        sections: [
          {
            h: 'a closure is a snapshot, an argument is a promise',
            ps: [
              "The run below changes a closed-over array between two calls and the jitted function does not notice. First call prints ones. Mutate `bias[0]` to 100. Second call prints ones again, while the same expression outside `jit` prints 101, so there is no doubt about what the array now holds.",
              "Nothing failed here, and nothing recompiled, which is the part worth being precise about. The trace read `bias` once and the value went into the recording. The second call arrived with the same shape and dtype, matched the cached executable, and never ran the Python body at all. `add_bias._cache_size()` says one, because a closed-over value is not part of the key that chapter 3 describes.",
              "So the rule to carry is short. A value that changes between calls belongs in the argument list, where every call rebinds it. A value that will not change can be closed over, and the trace will keep its own copy.",
              '>> Change a closed-over array and nothing recompiles, because nothing in the key changed.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a mutation the jitted function never sees, with the eager result underneath for contrast',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nimport numpy as np\n\nbias = np.zeros(3)\n\n@jax.jit\ndef add_bias(x):\n    return x + bias\n\nprint(add_bias(jnp.ones(3)))\nbias[0] = 100.0\nprint(add_bias(jnp.ones(3)))\nprint(add_bias._cache_size())\nprint(jnp.ones(3) + bias)\n\n# [1. 1. 1.]\n# [1. 1. 1.]\n# 1\n# [101.   1.   1.]',
            },
          },
          {
            h: 'what the snapshot is made of',
            ps: [
              "Trace a function that closes over the Python float `2.0` and the number appears inside the equation itself: `mul a 2.0`. There is no separate storage for it, and no way to change it short of retracing.",
              "Close over an array instead and the recording grows a slot in front of the arguments, with the array's value carried alongside as a constant. The `.consts` list on the traced object holds it, one entry, shape `(2, 3)` in the run below. The kernel path's source lesson at /l/source names the two slots the jaxpr grammar uses for this; what matters here is that a copy of the value is now part of the program.",
              "That copy has a size. A closed-over parameter tree of a gigabyte is a gigabyte of constants in the trace, held for as long as the compiled function is alive, and duplicated per signature you trace. Passing the same tree as an argument costs nothing extra, which is most of why idiomatic JAX threads parameters through the call rather than reaching for them from an enclosing scope.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a closed-over scalar written into the equation, a closed-over array carried beside it',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nscale = 2.0\nweights = jnp.ones((2, 3))\n\nprint(jax.make_jaxpr(lambda x: x * scale)(jnp.ones(3)))\nclosed = jax.make_jaxpr(lambda x: weights @ x)(jnp.ones(3))\nprint(closed)\nprint(len(closed.consts), closed.consts[0].shape)\n\n# { lambda ; a:f32[3]. let b:f32[3] = mul a 2.0 in (b,) }\n# { lambda a:f32[2,3]; b:f32[3]. let\n#     c:f32[2] = dot_general[\n#       dimension_numbers=(([1], [0]), ([], []))\n#       preferred_element_type=float32\n#     ] a b\n#   in (c,) }\n# 1 (2, 3)',
            },
          },
          {
            h: 'the tracer that outlived its trace',
            ps: [
              "Append the argument to a module-level list inside a jitted function and nothing complains during the call. The append happens at trace time, the function returns its real result, and the list now holds a `DynamicJaxprTracer` whose trace finished a moment ago.",
              "Touching that object is what raises, and the message is unusually generous. It names the type that escaped, states the rule it broke, names the function being traced when the leak happened, and prints the stack frames from where the value was created. Read the creation line carefully: it points at the call site, `<string>:11` here, not at the `append` that did the storing, because the value was created when the trace began.",
              "The last two lines hand you the follow-up before you go looking for it: an environment variable and a context manager that catch the leak earlier, which the next section runs.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the leak, and the whole UnexpectedTracerError verbatim; line numbers refer to the snippet, which ran through python3 -c',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nseen = []\n\n@jax.jit\ndef step(x):\n    seen.append(x)\n    return x * 2\n\nstep(jnp.ones(3))\nprint(type(seen[0]).__name__)\ntry:\n    print(seen[0] + 1)\nexcept jax.errors.UnexpectedTracerError as e:\n    print(e)\n\n# DynamicJaxprTracer\n# Encountered an unexpected tracer. A function transformed by JAX had a side effect, allowing for a reference to an intermediate value with type float32[3] wrapped in a DynamicJaxprTracer to escape the scope of the transformation.\n# JAX transformations require that functions explicitly return their outputs, and disallow saving intermediate values to global state.\n# The function being traced when the value leaked was step at <string>:6 traced for jit.\n# ------------------------------\n# The leaked intermediate value was created on line <string>:11 (<module>).\n# ------------------------------\n# When the value was created, the final 5 stack frames (most recent last) excluding JAX-internal frames were:\n# ------------------------------\n# <string>:11 (<module>)\n# ------------------------------\n#\n# To catch the leak earlier, try setting the environment variable JAX_CHECK_TRACER_LEAKS or using the `jax.checking_leaks` context manager.\n# See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.UnexpectedTracerError',
            },
          },
          {
            h: 'catching it while the trace is still open',
            ps: [
              "Wrap the same call in `jax.checking_leaks()` and the failure moves from the moment you use the tracer to the moment the trace ends. That matters when the two are far apart, which they usually are: a leak stored during setup and read during evaluation gives you an error with nothing useful nearby.",
              "What this message adds is the referrer chain. It reports the leaked tracer, then the list holding it at index 0, then the module global holding the list, so you get the path to the object that kept the reference rather than the line that used it. The same check is available without editing code, through the `JAX_CHECK_TRACER_LEAKS` environment variable.",
              "It is a debugging mode rather than a setting to leave on. The check walks referrers for every trace, so it is slow, and the documentation says so plainly.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same leak under jax.checking_leaks; the three object ids are Python ids and differ on every run',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nseen = []\n\n@jax.jit\ndef step(x):\n    seen.append(x)\n    return x * 2\n\ntry:\n    with jax.checking_leaks():\n        step(jnp.ones(3))\nexcept Exception as e:\n    print(type(e).__name__)\n    print(e)\n\n# Exception\n# Leaked trace DynamicJaxprTrace. Leaked tracer(s):\n#\n# Traced<ShapedArray(float32[3])>with<DynamicJaxprTrace>\n# The error occurred while tracing the function step at <string>:6 for jit. This concrete value was not available in Python because it depends on the value of the argument x.\n# <DynamicJaxprTracer 4624049616> is referred to by <list 4520251328>[0]\n# <list 4520251328> is referred to by __main__.seen',
            },
          },
          {
            h: 'the closure over a tracer that is fine',
            ps: [
              "None of this makes closing over a tracer illegal. Define an inner jitted lambda inside a jitted function, let it close over the outer function's argument, and it runs without complaint, because the inner trace is still inside the outer one.",
              "The jaxpr shows what that closure turned into. The inner `pjit` equation takes two operands, `c` and `d`, where `c` is the outer function's argument and `d` is the array the lambda was called with. A tracer closed over inside its own trace becomes an ordinary operand of the inner call.",
              "So the line is not about closures and not about tracers. It is about whether the trace that made the tracer is still running when you use it. Inside, the value is a normal intermediate; outside, it is a reference to a program that has already been recorded and handed off.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a closed-over tracer becomes an operand of the inner pjit equation',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef outer(x):\n    inner = jax.jit(lambda y: y + x)\n    return inner(jnp.ones(3))\n\nprint(outer(jnp.arange(3.0)))\nprint(jax.make_jaxpr(outer)(jnp.arange(3.0)))\n\n# [1. 2. 3.]\n# { lambda ; a:f32[3]. let\n#     b:f32[3] = pjit[\n#       name=outer\n#       jaxpr={ lambda ; c:f32[3]. let\n#           d:f32[3] = broadcast_in_dim[\n#             broadcast_dimensions=()\n#             shape=(3,)\n#             sharding=None\n#           ] 1.0\n#           e:f32[3] = pjit[\n#             name=<lambda>\n#             jaxpr={ lambda ; f:f32[3] g:f32[3]. let h:f32[3] = add g f in (h,) }\n#           ] c d\n#         in (e,) }\n#     ] a\n#   in (b,) }',
            },
          },
        ],
        readings: [
          { label: 'UnexpectedTracerError', url: 'https://docs.jax.dev/en/latest/errors.html#jax.errors.UnexpectedTracerError', note: 'the official account of the leak, with the two other shapes it takes' },
          { label: 'jax.checking_leaks', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.checking_leaks.html', note: 'the context manager, and the warning about what it costs' },
          { label: 'JAX changelog', url: 'https://docs.jax.dev/en/latest/changelog.html', note: 'worth a scan per upgrade: 0.8.2 changed what a Tracer inherits from and deprecated a list of jax.core symbols' },
        ],
        check: [
          {
            q: 'You mutate a closed-over NumPy array between two calls to one jitted function and the output does not change. Why does nothing recompile?',
            a: 'Because a closed-over value is not part of the cache key. The trace copied it once, the second call matched on shape and dtype alone, and the Python body never ran again, so the mutation had no path into the program.',
          },
          {
            q: 'After a jitted call returns, a module-level list holds a DynamicJaxprTracer. What raises, and what does the message tell you to change?',
            a: 'Using that object raises UnexpectedTracerError. The message says transformations require functions to return their outputs explicitly rather than saving intermediates to global state, and it names the traced function and the line where the leaked value was created.',
          },
          {
            q: 'An inner jitted lambda closes over the outer function’s tracer and nothing complains. What did the closure become in the jaxpr?',
            a: 'An operand of the inner pjit equation, passed in alongside the lambda’s own argument. Closing over a tracer inside its own trace is ordinary; only using one after its trace has ended is the leak.',
          },
        ],
        work: [
          { id: 'closure-audit', label: 'find one closed-over array in code of your own and decide in writing whether it should be an argument instead', href: '#a-closure-is-a-snapshot-an-argument-is-a-promise' },
          { id: 'leak-twice', label: 'reproduce one leak on purpose, then run the same program under jax.checking_leaks and compare what the two messages tell you', href: '#catching-it-while-the-trace-is-still-open' },
        ],
      },
    ],
  },
]
