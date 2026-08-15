// New file: site/src/data/lessons/jax-control-flow.ts
// The tracing semantics under chapter 6's survey: what cond and switch put in a
// jaxpr, the carry contract scan and while_loop share, and the params that make
// one scan equation readable as a training loop. Every printed value ran on this
// machine (jax 0.4.38, jaxlib 0.4.38, Python 3.12, CPU, one device, 2026-08-15).
// Every source excerpt is verbatim from the installed 0.4.38 wheel; loops.py,
// conditionals.py and common.py in that wheel are byte-identical to the
// jax-v0.4.38 tag, so the quoted line numbers hold against the public tree.
// Timings are stated with their method; equation counts are exact.
import type { UnitLessons } from './index'

export const JAX_CONTROL_FLOW_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:control-flow',
    lessons: [
      {
        id: 'both-branches-one-index',
        num: 1,
        title: 'Both branches, one index',
        lede: 'Two branch functions go into a cond and both of them run while you trace. The predicate stops being a boolean before the primitive sees it, and what finally picks a branch is an integer.',
        goal: 'Predict what a cond or a switch puts into a jaxpr, name everything the branches have to agree about, and say what batching changes about the cost of the branch nobody took and about the gradient through it.',
        sections: [
          {
            h: 'both branch functions run while you trace',
            ps: [
              "Put a `print` inside each branch function and trace the function once. Both prints fire. A jaxpr for each branch has to exist before the primitive can be built, so your Python runs once per branch, at trace time, whatever the predicate later turns out to be.",
              "The jaxpr then shows what became of the predicate. `gt` produced a `bool[]`, and the very next equation converts it to `i32[]`. `cond` never receives a boolean at all; it receives an index and a tuple of branches.",
              "Read the branch order twice, because it is the reverse of how the call is written. `div` sits in the first slot and `mul` in the second, and `div` was the false branch. One line in the source says why: `cond_p.bind(index, *consts, *ops, branches=(false_jaxpr, true_jaxpr))`. Slot 0 is false, slot 1 is true, so a two-way `cond` is a two-entry switch that was handed a converted boolean.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): two prints at trace time, and the index the predicate turned into',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef flip(x):\n    def hot(v):\n        print("tracing hot")\n        return v * 2.0\n    def cold(v):\n        print("tracing cold")\n        return v / 2.0\n    return jax.lax.cond(x.sum() > 0, hot, cold, x)\n\nprint(jax.make_jaxpr(flip)(jnp.ones(3)))\n\n# tracing hot\n# tracing cold\n# { lambda ; a:f32[3]. let\n#     b:f32[] = reduce_sum[axes=(0,)] a\n#     c:bool[] = gt b 0.0\n#     d:i32[] = convert_element_type[new_dtype=int32 weak_type=False] c\n#     e:f32[3] = cond[\n#       branches=(\n#         { lambda ; f:f32[3]. let g:f32[3] = div f 2.0 in (g,) }\n#         { lambda ; h:f32[3]. let i:f32[3] = mul h 2.0 in (i,) }\n#       )\n#     ] d a\n#   in (e,) }',
            },
          },
          {
            h: 'a literal predicate does not delete a branch',
            ps: [
              "Hand `cond` a Python `True` and nothing collapses. Both branches are still in the jaxpr and the index is the literal `1`. The primitive was built the same way it always is; only the value feeding it is known early.",
              "That is worth holding against the chapter above, where the branching that really does resolve during tracing is a Python `if` on a static value. The two cases look similar in the source and land in different places, and the jaxpr is where you tell them apart: an `if` leaves no `cond` equation behind at all, and a `cond` leaves one even when its predicate is a constant.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a constant predicate, and still two branches',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef always(x):\n    return jax.lax.cond(True, lambda v: v * 2.0, lambda v: v / 2.0, x)\n\nprint(jax.make_jaxpr(always)(jnp.ones(3)))\n\n# { lambda ; a:f32[3]. let\n#     b:f32[3] = cond[\n#       branches=(\n#         { lambda ; c:f32[3]. let d:f32[3] = div c 2.0 in (d,) }\n#         { lambda ; e:f32[3]. let f:f32[3] = mul e 2.0 in (f,) }\n#       )\n#     ] 1 a\n#   in (b,) }',
            },
          },
          {
            h: 'switch clamps the index and says nothing',
            ps: [
              "`jax.lax.switch` is the n-way form, and its docstring states the semantics as three lines of Python you can hold in your head. Clamp the index into range, then apply that branch. The clamp is not a footnote: it is an equation in the jaxpr, sitting right before the `cond`.",
              "So an index of 7 against three branches runs branch 2, and an index of -5 runs branch 0. No error, no warning, and a result that looks like a legitimate answer. If your index is computed from data, the failure mode for an off-by-one is a plausible number rather than a raise.",
              "The rows below came from one jitted function called five times. Only the index changed between calls, and the same executable served all five, because the index is a traced value like any other.",
            ],
            code: {
              caption: 'verbatim, jax/_src/lax/control_flow/conditionals.py:71-79 at jax-v0.4.38 (the switch docstring), then a run on this machine (jax 0.4.38 CPU)',
              lang: 'python',
              text: '  """Apply exactly one of the ``branches`` given by ``index``.\n\n  If ``index`` is out of bounds, it is clamped to within bounds.\n\n  Has the semantics of the following Python::\n\n    def switch(index, branches, *operands):\n      index = clamp(0, index, len(branches) - 1)\n      return branches[index](*operands)\n\n# >>> three = [lambda v: v + 1.0, lambda v: v + 2.0, lambda v: v + 3.0]\n# >>> def pick(i, x): return jax.lax.switch(i, three, x)\n# >>> print(jax.make_jaxpr(pick)(jnp.int32(0), jnp.ones(2)))\n# { lambda ; a:i32[] b:f32[2]. let\n#     c:i32[] = clamp 0 a 2\n#     d:f32[2] = cond[\n#       branches=(\n#         { lambda ; e:f32[2]. let f:f32[2] = add e 1.0 in (f,) }\n#         { lambda ; g:f32[2]. let h:f32[2] = add g 2.0 in (h,) }\n#         { lambda ; i:f32[2]. let j:f32[2] = add i 3.0 in (j,) }\n#       )\n#     ] c b\n#   in (d,) }',
            },
            table: {
              caption: 'the same jitted switch called five times with three branches, measured on jax 0.4.38 CPU',
              cols: ['index passed', 'branch that ran', 'result'],
              rows: [
                ['-5', '0', '[2. 2.]'],
                ['0', '0', '[2. 2.]'],
                ['1', '1', '[3. 3.]'],
                ['2', '2', '[4. 4.]'],
                ['7', '2', '[4. 4.]'],
              ],
            },
          },
          {
            h: 'what the two branches have to agree about',
            ps: [
              "The branches meet at one output type, and the check is exact rather than sympathetic. Different shapes raise. Different dtypes raise. Different pytree structures raise, and that one is reported separately, naming both treedefs instead of comparing avals.",
              "One disagreement is allowed through, and it is the one people expect to break. A branch returning the Python float `1.0` against a branch returning an `f32[]` array is fine, because the weak scalar promotes to meet the array. Weakness is a property of the type here, not a special case in `cond`.",
              "The habit worth building is to read the two avals in the message before rereading your code. `DIFFERENT ShapedArray(float32[3]) vs. ShapedArray(float32[2])` tells you which axis moved, and it is usually a slice or a reduction inside one branch that you did not mirror in the other.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): three refusals and one acceptance; current jax has reworded the structure message to report differences per output path',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nx = jnp.ones(3)\npred = x.sum() > 0\n\ndef show(true_fn, false_fn):\n    try:\n        jax.lax.cond(pred, true_fn, false_fn, x)\n        print("accepted")\n    except TypeError as err:\n        print(err)\n\nshow(lambda v: v, lambda v: v[:2])                    # shapes disagree\nshow(lambda v: v, lambda v: v.astype(jnp.float16))    # dtypes disagree\nshow(lambda v: (v, v), lambda v: v)                   # structures disagree\nshow(lambda v: v.sum(), lambda v: 1.0)                # a weak python float\n\n# true_fun and false_fun output must have identical types, got\n# DIFFERENT ShapedArray(float32[3]) vs. ShapedArray(float32[2]).\n# true_fun and false_fun output must have identical types, got\n# DIFFERENT ShapedArray(float32[3]) vs. ShapedArray(float16[3]).\n# true_fun and false_fun output must have same type structure, got PyTreeDef((*, *)) and PyTreeDef(*).\n# accepted',
            },
          },
          {
            h: 'a value one branch uses enters both',
            ps: [
              "Close over a value in one branch only and watch what the jaxpr does with it. Both branch signatures grow, both get the value, and the one that has no use for it marks its binder with a trailing underscore. The operand list carries it twice, once for each branch.",
              "The comment above the function that does this states the reason in three sentences: the staged jaxprs are the branches of one primitive, so their input signatures have to match, and the fix is to make each accept all the constants and drop the ones it does not need.",
              "This is where a closure over something large stops being free. Two branches closing over two different big arrays produce one primitive whose operand list holds both, whichever branch runs. Passing the value as an operand instead of closing over it changes nothing about that, but it does make the cost visible at the call site.",
            ],
            code: {
              caption: 'verbatim, jax/_src/lax/control_flow/common.py:78-83 at jax-v0.4.38, then a run on this machine (jax 0.4.38 CPU) where only one branch uses w',
              lang: 'python',
              text: '  # When staging the branches of a conditional into jaxprs, constants are\n  # extracted from each branch and converted to jaxpr arguments. To use the\n  # staged jaxprs as the branches to a conditional *primitive*, we need for\n  # their (input) signatures to match. This function "joins" the staged jaxprs:\n  # for each one, it makes another that accepts *all* constants, but only uses\n  # those that it needs (dropping the rest).\n\n# >>> def f(x, w):\n# ...     return jax.lax.cond(x.sum() > 0, lambda v: v * w, lambda v: v - w, x)\n# >>> print(jax.make_jaxpr(f)(jnp.ones(3), jnp.full((3,), 2.0)))\n# { lambda ; a:f32[3] b:f32[3]. let\n#     c:f32[] = reduce_sum[axes=(0,)] a\n#     d:bool[] = gt c 0.0\n#     e:i32[] = convert_element_type[new_dtype=int32 weak_type=False] d\n#     f:f32[3] = cond[\n#       branches=(\n#         { lambda ; g_:f32[3] h:f32[3] i:f32[3]. let\n#             j:f32[3] = convert_element_type[new_dtype=float32 weak_type=False] h\n#             k:f32[3] = sub i j\n#           in (k,) }\n#         { lambda ; l:f32[3] m_:f32[3] n:f32[3]. let\n#             o:f32[3] = convert_element_type[new_dtype=float32 weak_type=False] l\n#             p:f32[3] = mul n o\n#           in (p,) }\n#       )\n#     ] e b b a\n#   in (f,) }',
            },
          },
          {
            h: 'the branch nobody took is free until you batch it',
            ps: [
              "Give one branch five chained 512 by 512 matmuls and the other an add, then time the same compiled function under both predicates. The heavy predicate costs milliseconds and the light one costs almost nothing, so on this backend the executable really is skipping the branch it did not select.",
              "Batch that same function and the timing stops depending on the predicate. Two lanes with both predicates false still pay for the matmuls, because the batched form has to produce both results before it can select between them per lane. Chapter 6 states that cost; the stopwatch is here so you can see how large it gets when one branch is expensive.",
              "Which gives a rule for where to put a `cond` in a batched program. A branch that guards a cheap correction costs little either way. A branch that guards the expensive half of your model is worth hoisting out of the `vmap`, so the predicate applies to the whole batch instead of per element.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): best of 50 calls each, on the timings in the table below',
              lang: 'python',
              text: 'import time\nimport jax\nimport jax.numpy as jnp\n\nA = jnp.ones((512, 512))\nheavy = lambda v: (A @ A @ A @ A @ A).sum() + v\nlight = lambda v: v + 1.0\none = jax.jit(lambda p, x: jax.lax.cond(p, heavy, light, x))\nmany = jax.jit(jax.vmap(lambda p, x: jax.lax.cond(p, heavy, light, x)))\n\ndef best(fn, *a):\n    fn(*a).block_until_ready()\n    ts = []\n    for _ in range(50):\n        t0 = time.perf_counter()\n        fn(*a).block_until_ready()\n        ts.append(time.perf_counter() - t0)\n    return 1e3 * min(ts)\n\nx, xs = jnp.float32(1.0), jnp.ones(2)\nprint(f"cond, predicate True   {best(one, jnp.bool_(True), x):6.2f} ms")\nprint(f"cond, predicate False  {best(one, jnp.bool_(False), x):6.2f} ms")\nprint(f"vmapped, both False    {best(many, jnp.array([False, False]), xs):6.2f} ms")\nprint(f"vmapped, both True     {best(many, jnp.array([True, True]), xs):6.2f} ms")',
            },
            table: {
              caption: 'measured on this machine, jax 0.4.38 CPU, best of 50 calls each; across three runs of the same script the first row stayed near 3.5 ms, the second under 0.1 ms, and the two batched rows moved between 5.3 and 8.5 ms',
              cols: ['what ran', 'best of 50 calls'],
              rows: [
                ['cond, predicate True', '3.42 ms'],
                ['cond, predicate False', '0.03 ms'],
                ['vmap of the same cond, two lanes, both predicates False', '5.26 ms'],
                ['vmap of the same cond, two lanes, both predicates True', '8.48 ms'],
              ],
            },
          },
          {
            h: 'the batched branch still blocks the gradient',
            ps: [
              "Guard a `1 / x` with `jnp.where` and differentiate it at zero, and back comes `nan`. The masked-out value was still computed, its derivative was still taken, and multiplying an infinite derivative by a zero cotangent produces the indeterminate form. This is the classic reason people distrust masking.",
              "Write the same guard as a `cond`, batch it, differentiate it, and the answer is `0`. That is not because the batched form skipped anything. It computed both branches, the same way the timings above showed.",
              "The jaxpr says where the protection comes from. Before each branch runs, its operand goes through a `select_n` against `stop_gradient` of itself: the lanes that branch owns receive the real value, and every other lane receives a gradient-blocked copy. So the `1 / x` still evaluates at zero, and no cotangent from it can reach the input.",
              '>> Batching a cond costs you both branches. It does not cost you the gradient.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same guard two ways, then the batched jaxpr that explains the difference',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nmasked = lambda x: jnp.where(x > 0, 1.0 / x, 0.0)\nbranched = lambda x: jax.lax.cond(x > 0, lambda v: 1.0 / v, lambda v: 0.0 * v, x)\n\nxs = jnp.array([0.0, 2.0])\nprint(jax.vmap(masked)(xs), jax.vmap(branched)(xs))\nprint(jax.vmap(jax.grad(masked))(xs))\nprint(jax.vmap(jax.grad(branched))(xs))\nprint(jax.make_jaxpr(jax.vmap(branched))(xs))\n\n# [0.  0.5] [0.  0.5]\n# [  nan -0.25]\n# [ 0.   -0.25]\n# { lambda ; a:f32[2]. let\n#     b:bool[2] = gt a 0.0\n#     c:i32[2] = convert_element_type[new_dtype=int32 weak_type=False] b\n#     d:bool[2] = eq c 0\n#     e:f32[2] = stop_gradient a\n#     f:f32[2] = select_n d e a\n#     g:f32[2] = mul 0.0 f\n#     h:bool[2] = eq c 1\n#     i:f32[2] = stop_gradient a\n#     j:f32[2] = select_n h i a\n#     k:f32[2] = div 1.0 j\n#     l:f32[2] = select_n c g k\n#   in (l,) }',
            },
          },
        ],
        readings: [
          { label: 'jax.lax.cond reference', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html', note: 'the contract, and the note that both branches are traced' },
          { label: 'jax.lax.switch reference', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html', note: 'the clamp is documented, one line above the semantics' },
          { label: 'conditionals.py at jax-v0.4.38', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/lax/control_flow/conditionals.py', note: 'switch at 69, the bind that orders the branches false-first at 263' },
        ],
        check: [
          {
            q: 'A cond jaxpr shows div in the first branch slot and mul in the second, but you wrote mul as the true branch. What happened?',
            a: 'cond binds branches as (false_jaxpr, true_jaxpr) and passes an int32 index, so slot 0 is the false branch and slot 1 the true one. Your bool predicate was converted to i32 by a convert_element_type equation before the primitive saw it.',
          },
          {
            q: 'switch got an index of 7 against three branches. What runs, and what tells you?',
            a: 'Branch 2 runs, and nothing tells you. switch emits a clamp equation ahead of the cond, so out-of-range indices are pinned to the ends silently: -5 runs branch 0 and 7 runs branch 2, both returning a plausible-looking answer.',
          },
          {
            q: 'grad through a jnp.where guard returned nan at zero where the same guard written as lax.cond returned 0, even under vmap. What in the batched jaxpr accounts for that?',
            a: 'Each branch gets its operand through a select_n against stop_gradient of that operand: the lanes the branch owns see the real value, the others see a gradient-blocked copy. Both branches still compute, but no cotangent flows out of the branch a lane did not select.',
          },
        ],
        work: [
          { id: 'branch-order', label: 'trace one cond of your own and name, without running it again, which branch is in slot 0 and what dtype the predicate has when the primitive gets it', href: '#both-branch-functions-run-while-you-trace' },
          { id: 'switch-off-by-one', label: 'write a switch whose index is computed from data, feed it an out-of-range index on purpose, and decide where in your code the range check belongs', href: '#switch-clamps-the-index-and-says-nothing' },
        ],
      },
      {
        id: 'the-carry-is-a-type',
        num: 2,
        title: 'The carry is a type',
        lede: 'State threaded through a loop is not just a value. It is a shape, a dtype and a pytree structure, and the one going out has to equal the one that came in, checked by the same function for scan and while_loop alike.',
        goal: 'Read a carry mismatch error and name which primitive checked it and which component broke, predict when a weakly typed init makes the loop body trace twice, and choose between scan, while_loop and fori_loop from the trip count and the differentiation you need.',
        sections: [
          {
            h: 'in and out, or it does not run',
            ps: [
              "A loop primitive builds its body once, against the types of the initial carry. For that single body to be correct on every iteration, whatever comes out of it has to have exactly the types that went in. Nothing adapts between steps, because there is nothing between steps: it is one traced body, run again.",
              "One function enforces that, and both primitives call it. `_check_carry_type` runs at line 303 for a scan body and at line 1357 for a while_loop body, which is why the two errors read identically apart from the name in front. Learn one message and you can read both.",
              "The message text is worth knowing by sight, because the useful half is the middle. The wrapper sentences are constant; the lines between them name the component and give you both types.",
            ],
            code: {
              caption: 'verbatim, jax/_src/lax/control_flow/loops.py at jax-v0.4.38: the two call sites at 303 and 1357, then the raise at 411-417, joined here under added path headings',
              lang: 'python',
              text: '# loops.py:303, inside scan\n  _check_carry_type(\'scan body\', f, init, out_tree_children[0], carry_avals_out)\n\n# loops.py:1357, inside while_loop\n  _check_carry_type(\'while_loop body\', body_fun, new_init_val, body_tree,\n                    body_jaxpr.out_avals)\n\n# loops.py:411-417, the raise both of them reach\n    raise TypeError(\n        f"{name} function carry input and carry output must have equal types "\n        "(e.g. shapes and dtypes of arrays), "\n        "but they differ:\\n\\n"\n        f"{differences}\\n"\n        "Revise the function so that all output types (e.g. shapes "\n        "and dtypes) match the corresponding input types.")',
            },
          },
          {
            h: 'one defect, three names in front',
            ps: [
              "Grow the carry and the message says shapes. A carry that starts as `float32[1]` and comes back as `float32[2]` is the shape of every accumulate-into-a-growing-buffer idea people bring from NumPy, and a loop primitive refuses it before it runs a single step.",
              "Hand the identical defect to `while_loop` and the sentence that comes back is the same sentence, with a different name in front of it. That is the one check function speaking from its two call sites, which is what makes the message worth learning once instead of three times.",
              "`fori_loop` is the case that repays a second look. The name in front reads `scan body` for a call you never wrote as a scan, and the component it names is `loop_carry[1]` rather than a parameter, because the loop index is carry component 0 and your value sits behind it. Both of those are the lowering showing through, and the bounds that chose that lowering get their own section later in this lesson.",
              "The other two ways a carry breaks are told elsewhere, from the side that owns them. A dtype that drifts is read from the state side in chapter 9's lesson `The step that returns its state`, where the message names the offending field by its path into the state tree. A structure that changes is read from the tree side in chapter 7's lesson `Structure, then leaves`, together with the rule that structure is checked before any type is.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one growing carry against three primitives; current jax has dropped the parenthetical from the first and last sentences of this message',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ngrow = lambda c: jnp.concatenate([c, jnp.ones(1)])\n\ndef show(call):\n    try:\n        call()\n    except TypeError as err:\n        print(err)\n    print("---")\n\nshow(lambda: jax.lax.scan(lambda c, x: (grow(c), None), jnp.zeros(1), jnp.arange(4.)))\nshow(lambda: jax.lax.while_loop(lambda c: c.sum() < 3.0, grow, jnp.zeros(1)))\nshow(lambda: jax.lax.fori_loop(0, 4, lambda i, c: grow(c), jnp.zeros(1)))\n\n# scan body function carry input and carry output must have equal types (e.g. shapes and dtypes of arrays), but they differ:\n#\n# The input carry c has type float32[1] but the corresponding output carry component has type float32[2], so the shapes do not match.\n#\n# Revise the function so that all output types (e.g. shapes and dtypes) match the corresponding input types.\n# ---\n# while_loop body function carry input and carry output must have equal types (e.g. shapes and dtypes of arrays), but they differ:\n#\n# The input carry c has type float32[1] but the corresponding output carry component has type float32[2], so the shapes do not match.\n#\n# Revise the function so that all output types (e.g. shapes and dtypes) match the corresponding input types.\n# ---\n# scan body function carry input and carry output must have equal types (e.g. shapes and dtypes of arrays), but they differ:\n#\n# The input carry component loop_carry[1] has type float32[1] but the corresponding output carry component has type float32[2], so the shapes do not match.\n#\n# Revise the function so that all output types (e.g. shapes and dtypes) match the corresponding input types.\n# ---',
            },
            table: {
              caption: 'the same growing carry, three call sites, measured on jax 0.4.38 CPU',
              cols: ['what you called', 'the name in front', 'the component it names'],
              rows: [
                ['jax.lax.scan', 'scan body function', 'c, the body parameter'],
                ['jax.lax.while_loop', 'while_loop body function', 'c, the body parameter'],
                ['jax.lax.fori_loop, static bounds', 'scan body function', 'loop_carry[1], one slot past the index'],
              ],
            },
          },
          {
            h: 'a python scalar traces the body twice',
            ps: [
              "Start a scan with a plain `0` and put a counter in the body. The body traces twice. The first pass traces against `int32[]` weak, discovers the carry comes back `float32[]`, promotes the init, and traces again. The literal in the final jaxpr is `0.0`, not `0`.",
              "The source says so directly, in a comment sitting above the two calls that do it. A weakly typed init may be compatible with the output despite not matching it, and the way that gets resolved is two passes, the second one with modified init values.",
              "This is why a side effect inside a loop body is a bad place to keep a count. A `print`, an append to a list, a metric you increment: any of those fire once per trace, and the number of traces depends on how you wrote the init. Passing `jnp.float32(0.0)` instead of `0.0` is enough to make it one.",
              "Current jax keeps the same mechanism and has added a note that it costs more than it looks: the comment on main now carries a TODO calling the two-pass approach expensive, exponential in scan nesting depth, and incomplete in the general case.",
            ],
            code: {
              caption: 'verbatim, jax/_src/lax/control_flow/loops.py:290-299 at jax-v0.4.38, then a run on this machine (jax 0.4.38 CPU) counting the traces',
              lang: 'python',
              text: '  # The carry input and output avals must match exactly. However, we want to account for\n  # the case when init contains weakly-typed values (e.g. Python scalars), with avals that\n  # may not match the output despite being compatible by virtue of their weak type.\n  # To do this, we compute the jaxpr in two passes: first with the raw inputs, and if\n  # necessary, a second time with modified init values.\n  init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(init)\n  new_init_flat, changed = _promote_weak_typed_inputs(init_flat, carry_avals, carry_avals_out)\n  if changed:\n    init = tree_unflatten(init_tree, new_init_flat)\n    init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(init)\n\n# >>> traces = []\n# >>> def step(total, x):\n# ...     traces.append(jax.eval_shape(lambda v: v, total))\n# ...     return total + x, None\n# >>> print(jax.make_jaxpr(lambda xs: jax.lax.scan(step, 0, xs))(jnp.arange(4.)))\n# { lambda ; a:f32[4]. let\n#     b:f32[] = scan[\n#       _split_transpose=False\n#       jaxpr={ lambda ; c:f32[] d:f32[]. let e:f32[] = add c d in (e,) }\n#       length=4\n#       linear=(False, False)\n#       num_carry=1\n#       num_consts=0\n#       reverse=False\n#       unroll=1\n#     ] 0.0 a\n#   in (b,) }\n# >>> for t in traces: print(t)\n# ShapeDtypeStruct(shape=(), dtype=int32, weak_type=True)\n# ShapeDtypeStruct(shape=(), dtype=float32)',
            },
          },
          {
            h: 'the loop that stops when the numbers say so',
            ps: [
              "`while_loop` adds one requirement on top of the carry contract, and it is about the condition rather than the body. `cond_fun` has to return a boolean scalar. Compare an array by mistake and it hands back a `bool[3]`, which is refused before the body is looked at.",
              "Its trip count is decided at run time, and that decides what you can differentiate. Forward mode is fine: a tangent rides through the loop alongside the value, one step at a time, and needs no advance knowledge of how many steps there will be. The run below takes a jvp through a while_loop and gets both numbers back.",
              "Reverse mode raises, and the message is unusually direct about the alternatives: use `lax.scan`, or use `fori_loop` with static start and stop. Both suggestions amount to the same thing, which is giving the loop a trip count known before it runs.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): forward mode through a while_loop, then the two refusals',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef grow(x):\n    cond_fn = lambda c: c[0] < 3.0\n    body_fn = lambda c: (c[0] + 1.0, c[1] * x)\n    return jax.lax.while_loop(cond_fn, body_fn, (0.0, 1.0))[1]\n\nprint(jax.jvp(grow, (2.0,), (1.0,)))\ntry:\n    jax.grad(grow)(2.0)\nexcept ValueError as err:\n    print(err)\ntry:\n    jax.lax.while_loop(lambda v: v < 5.0, lambda v: v + 1.0, jnp.ones(3))\nexcept TypeError as err:\n    print(err)\n\n# (Array(8., dtype=float32, weak_type=True), Array(12., dtype=float32, weak_type=True))\n# Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop.\n# cond_fun must return a boolean scalar, but got output type(s) [ShapedArray(bool[3])].',
            },
          },
          {
            h: 'fori_loop picks its road from its bounds',
            ps: [
              "Trace the same `fori_loop` twice with different bounds and you get two different primitives. Static bounds produce a `scan` with `length=5` and a carry of two, the loop index riding along as the first carried value. A traced upper bound produces a `while` instead, with a `cond_jaxpr` doing the comparison and a `body_jaxpr` doing the work.",
              "The trap in that is where the bound becomes traced. Outside `jit`, a concrete `jnp.int32(5)` is an ordinary value, so the scan road is taken and reverse mode works. Wrap the same call in `jit` and the bound is a tracer, the while road is taken, and the identical code raises the reverse-mode error.",
              "So a gradient that works in a notebook and fails in a jitted training step is not a mystery. It is one argument that stopped being concrete, and the fix is to make the bound static rather than to restructure the loop.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the two lowerings, then the same gradient outside and inside jit; three scan params that do not vary here (_split_transpose, linear, unroll) are trimmed from the excerpt and printed in the fold',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndecay = lambda i, c: c * 1.1 + 1.0\nloop = lambda x, n: jax.lax.fori_loop(0, n, decay, x)\n\nprint(jax.make_jaxpr(lambda x: jax.lax.fori_loop(0, 5, decay, x))(jnp.float32(1.0)))\nprint(jax.make_jaxpr(loop)(jnp.float32(1.0), jnp.int32(5)))\nprint(jax.grad(loop)(jnp.float32(1.0), jnp.int32(5)))\ntry:\n    jax.jit(jax.grad(loop))(jnp.float32(1.0), jnp.int32(5))\nexcept ValueError as err:\n    print(err)\n\n# { lambda ; a:f32[]. let\n#     _:i32[] b:f32[] = scan[\n#       jaxpr={ lambda ; c:i32[] d:f32[]. let\n#           e:i32[] = add c 1\n#           f:f32[] = mul d 1.100000023841858\n#           g:f32[] = add f 1.0\n#         in (e, g) }\n#       length=5\n#       num_carry=2\n#       num_consts=0\n#       reverse=False\n#     ] 0 a\n#   in (b,) }\n# { lambda ; a:f32[] b:i32[]. let\n#     _:i32[] _:i32[] c:f32[] = while[\n#       body_jaxpr={ lambda ; d:i32[] e:i32[] f:f32[]. let\n#           g:i32[] = add d 1\n#           h:f32[] = mul f 1.100000023841858\n#           i:f32[] = add h 1.0\n#         in (g, e, i) }\n#       body_nconsts=0\n#       cond_jaxpr={ lambda ; j:i32[] k:i32[] l:f32[]. let\n#           m:bool[] = lt j k\n#         in (m,) }\n#       cond_nconsts=0\n#     ] 0 b a\n#   in (c,) }\n# 1.6105101\n# Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop.',
              full: {
                text: 'import jax\nimport jax.numpy as jnp\n\ndecay = lambda i, c: c * 1.1 + 1.0\nloop = lambda x, n: jax.lax.fori_loop(0, n, decay, x)\n\nprint(jax.make_jaxpr(lambda x: jax.lax.fori_loop(0, 5, decay, x))(jnp.float32(1.0)))\nprint(jax.make_jaxpr(loop)(jnp.float32(1.0), jnp.int32(5)))\nprint(jax.grad(loop)(jnp.float32(1.0), jnp.int32(5)))\ntry:\n    jax.jit(jax.grad(loop))(jnp.float32(1.0), jnp.int32(5))\nexcept ValueError as err:\n    print(err)\n\n# { lambda ; a:f32[]. let\n#     _:i32[] b:f32[] = scan[\n#       _split_transpose=False\n#       jaxpr={ lambda ; c:i32[] d:f32[]. let\n#           e:i32[] = add c 1\n#           f:f32[] = mul d 1.100000023841858\n#           g:f32[] = add f 1.0\n#         in (e, g) }\n#       length=5\n#       linear=(False, False)\n#       num_carry=2\n#       num_consts=0\n#       reverse=False\n#       unroll=1\n#     ] 0 a\n#   in (b,) }\n# { lambda ; a:f32[] b:i32[]. let\n#     _:i32[] _:i32[] c:f32[] = while[\n#       body_jaxpr={ lambda ; d:i32[] e:i32[] f:f32[]. let\n#           g:i32[] = add d 1\n#           h:f32[] = mul f 1.100000023841858\n#           i:f32[] = add h 1.0\n#         in (g, e, i) }\n#       body_nconsts=0\n#       cond_jaxpr={ lambda ; j:i32[] k:i32[] l:f32[]. let\n#           m:bool[] = lt j k\n#         in (m,) }\n#       cond_nconsts=0\n#     ] 0 b a\n#   in (c,) }\n# 1.6105101\n# Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop.',
                label: 'the same run with every scan param printed',
              },
            },
          },
          {
            h: 'what scan needs from xs',
            ps: [
              "The length of a scan is inferred, not declared, and it comes from the leading axis of whatever you scan over. Every leaf of `xs` has to agree on that axis, and when they do not the error prints the sizes it found rather than a shape.",
              "There is a form with nothing to scan over at all. Pass `xs=None` and a `length`, and you get a pure iteration: the body takes the carry and an ignored `None`, and the stacked output still comes back with `length` rows. That is the shape a sampler or a fixed-step optimizer usually wants.",
              "Leave out both and the error is exactly what it should be. With no `xs` and no `length`, there is no number anywhere in the call that says how many times to run.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): two refusals and the xs-free form',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nadd = lambda c, x: (c + x, None)\n\ntry:\n    jax.lax.scan(lambda c, xy: (c + xy[0].sum(), None), 0.0, (jnp.ones((4, 2)), jnp.ones((3, 2))))\nexcept ValueError as err:\n    print(err)\ntry:\n    jax.lax.scan(add, 0.0, None)\nexcept ValueError as err:\n    print(err)\nprint(jax.lax.scan(lambda c, _: (c + 1.0, c), 0.0, None, length=3))\n\n# scan got values with different leading axis sizes: 4, 3.\n# scan got no values to scan over and `length` not provided.\n# (Array(3., dtype=float32, weak_type=True), Array([0., 1., 2.], dtype=float32, weak_type=True))',
            },
          },
        ],
        readings: [
          { label: 'loops.py at jax-v0.4.38', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/lax/control_flow/loops.py', note: 'scan at 129, the two-pass init promotion at 290, the carry check at 354' },
          { label: 'jax.lax.while_loop reference', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html', note: 'the cond_fun contract, stated as the Python it stands in for' },
          { label: 'jax.lax.fori_loop reference', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html', note: 'the docs say which lowering you get, and the differentiability that follows' },
          { label: 'Type promotion semantics', url: 'https://docs.jax.dev/en/latest/type_promotion.html', note: 'what weak means, which is what makes a python scalar init a two-pass trace' },
        ],
        check: [
          {
            q: 'Your scan raised with int32[] going in and float32[] coming out. What is the smallest fix, and what would a plain 0 as the init have done instead?',
            a: 'Make the init match the body: jnp.float32(0.0) rather than jnp.int32(0). A plain Python 0 is weakly typed, so no error appears at all; the init is promoted and the body traces a second time against float32[].',
          },
          {
            q: 'A print inside a four-step scan body fired twice. What does that tell you about the init you passed?',
            a: 'That it held a weakly typed Python scalar whose type had to be promoted to match the carry coming out. The body traces once with the raw init and once with the promoted one, so anything with a side effect in it happens twice.',
          },
          {
            q: 'The same gradient runs in a notebook and raises inside jit, and the loop is a fori_loop. Why?',
            a: 'Because the bound was concrete outside jit, so fori_loop lowered to a scan and reverse mode worked. Inside jit the bound is a tracer, so it lowered to a while, and reverse-mode differentiation refuses a loop whose trip count is not known in advance.',
          },
        ],
        work: [
          { id: 'carry-ledger', label: 'take one loop of your own and write its carry down as types before you run it: shape, dtype and structure per leaf, and the same list for what the body returns', href: '#in-and-out-or-it-does-not-run' },
          { id: 'trace-count', label: 'put a counter in a scan body, run it with a python scalar init and again with a typed one, and explain the difference in the count you get', href: '#a-python-scalar-traces-the-body-twice' },
        ],
      },
      {
        id: 'one-trace-however-many-steps',
        num: 3,
        title: 'One trace, however many steps',
        lede: 'A training loop compiles down to a single equation, and that equation carries a ledger: which operands are constants, which are carried, how many steps run, in which direction, and how much of the body the lowering should copy out.',
        goal: 'Read a scan equation and name every operand as a constant, a carry or a slice; say what unroll changes and what it leaves alone; and name what a gradient adds to a scanned loop and where a written-out loop is still the right call.',
        sections: [
          {
            h: 'consts first, then carry, then slices',
            ps: [
              "Here is a linear model taking one gradient step per batch, scanned over six batches. The whole loop is one equation, and its params tell you how to read the four operands after it. `num_consts=1` claims the first, `num_carry=1` claims the second, and everything left is sliced along its leading axis, one row per step.",
              "Line the operand list up against the body signature and the correspondence is exact. The operands are `d a b c`: the learning rate, the weights, the inputs, the targets. The body binders are `g h i j`: the learning rate whole, the weights as carry, and one slice each of inputs and targets, `f32[2,3]` and `f32[2]` cut from `f32[6,2,3]` and `f32[6,2]`.",
              "What decides that split is your closure, not a keyword. The learning rate is a traced argument closed over by the step function, so it became a constant operand. Pass `0.1` as a Python float instead and it stops being an operand at all: `num_consts` drops to 0 and the number is a literal inside the body.",
              "The kernel path's jaxpr chapter teaches the nesting grammar this equation is written in, and the XLA path's ingestion chapter walks one scan across the border into StableHLO. What this section adds is the reading of the params themselves, which is the part you need when the loop is your own training step and you want to know what got hoisted.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the traced program, with 16 of the body\'s 18 equations elided; the whole dump unfolds below',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef train(w, xs, ys, lr):\n    def step(w, batch):\n        x, y = batch\n        loss = lambda w: jnp.mean((x @ w - y) ** 2)\n        return w - lr * jax.grad(loss)(w), loss(w)\n    return jax.lax.scan(step, w, (xs, ys))\n\nprint(jax.make_jaxpr(train)(jnp.zeros(3), jnp.ones((6, 2, 3)), jnp.ones((6, 2)), jnp.float32(0.1)))\n\n# { lambda ; a:f32[3] b:f32[6,2,3] c:f32[6,2] d:f32[]. let\n#     e:f32[3] f:f32[6] = scan[\n#       _split_transpose=False\n#       jaxpr={ lambda ; g:f32[] h:f32[3] i:f32[2,3] j:f32[2]. let\n#           k:f32[2] = dot_general[\n#             dimension_numbers=(([1], [0]), ([], []))\n#             preferred_element_type=float32\n#           ] i h\n#\n#           ... 16 more equations: the loss, the gradient, the update ...\n#\n#           ba:f32[] = div z 2.0\n#         in (v, ba) }\n#       length=6\n#       linear=(False, False, False, False)\n#       num_carry=1\n#       num_consts=1\n#       reverse=False\n#       unroll=1\n#     ] d a b c\n#   in (e, f) }',
              full: {
                text: 'import jax\nimport jax.numpy as jnp\n\ndef train(w, xs, ys, lr):\n    def step(w, batch):\n        x, y = batch\n        loss = lambda w: jnp.mean((x @ w - y) ** 2)\n        return w - lr * jax.grad(loss)(w), loss(w)\n    return jax.lax.scan(step, w, (xs, ys))\n\nprint(jax.make_jaxpr(train)(jnp.zeros(3), jnp.ones((6, 2, 3)), jnp.ones((6, 2)), jnp.float32(0.1)))\n\n# { lambda ; a:f32[3] b:f32[6,2,3] c:f32[6,2] d:f32[]. let\n#     e:f32[3] f:f32[6] = scan[\n#       _split_transpose=False\n#       jaxpr={ lambda ; g:f32[] h:f32[3] i:f32[2,3] j:f32[2]. let\n#           k:f32[2] = dot_general[\n#             dimension_numbers=(([1], [0]), ([], []))\n#             preferred_element_type=float32\n#           ] i h\n#           l:f32[2] = sub k j\n#           m:f32[2] = integer_pow[y=2] l\n#           n:f32[2] = integer_pow[y=1] l\n#           o:f32[2] = mul 2.0 n\n#           p:f32[] = reduce_sum[axes=(0,)] m\n#           _:f32[] = div p 2.0\n#           q:f32[] = div 1.0 2.0\n#           r:f32[2] = broadcast_in_dim[\n#             broadcast_dimensions=()\n#             shape=(2,)\n#             sharding=None\n#           ] q\n#           s:f32[2] = mul r o\n#           t:f32[3] = dot_general[\n#             dimension_numbers=(([0], [0]), ([], []))\n#             preferred_element_type=float32\n#           ] s i\n#           u:f32[3] = mul g t\n#           v:f32[3] = sub h u\n#           w:f32[2] = dot_general[\n#             dimension_numbers=(([1], [0]), ([], []))\n#             preferred_element_type=float32\n#           ] i h\n#           x:f32[2] = sub w j\n#           y:f32[2] = integer_pow[y=2] x\n#           z:f32[] = reduce_sum[axes=(0,)] y\n#           ba:f32[] = div z 2.0\n#         in (v, ba) }\n#       length=6\n#       linear=(False, False, False, False)\n#       num_carry=1\n#       num_consts=1\n#       reverse=False\n#       unroll=1\n#     ] d a b c\n#   in (e, f) }',
                label: 'the whole traced program, 56 lines',
              },
            },
            table: {
              caption: 'the params on that one equation, with the values from the run above; the descriptions of unroll and _split_transpose are the docstring\'s, jax 0.4.38',
              cols: ['param', 'value here', 'what it decides'],
              rows: [
                ['num_consts', '1', 'how many leading operands go into the body unchanged on every step'],
                ['num_carry', '1', 'how many operands after those are threaded in and out as state'],
                ['length', '6', 'how many steps run, taken from the leading axis of the sliced operands'],
                ['reverse', 'False', 'which end the loop starts from, equivalent to reversing xs and ys'],
                ['unroll', '1', 'how many scan iterations the lowering runs inside one iteration of its loop'],
                ['linear', '(False, False, False, False)', 'one flag per operand, marking the ones autodiff knows are linear'],
                ['_split_transpose', 'False', 'experimental: split a transposed scan into a scan plus a map'],
              ],
            },
          },
          {
            h: 'unroll is a note for the lowering',
            ps: [
              "`unroll` does not change the jaxpr. Ask for `unroll=8` on an eight-step scan and the equation still says `length=8` with a two-equation body; the only difference is the `unroll` param itself, carried along for the lowering to act on.",
              "What it buys is measurable, and it is a trade rather than a win. On a 1024-step scan over a `tanh` body, going from `unroll=1` to `unroll=64` cut the run from 0.38 ms to 0.21 ms and pushed compile from 0.33 s to 1.07 s. Most of the run-time gain arrived by `unroll=16`, and most of the compile cost arrived after it.",
              "The docstring is worth reading before you set it, because the units are easy to misread. The number says how many scan iterations happen inside a single iteration of the underlying loop, so `unroll=16` on a 1024-step scan means 64 loop iterations of 16 bodies each, not 16 copies total.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): unroll survives into the params and leaves the body alone',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nxs = jnp.arange(8.)\nfor u in (1, 4, 8):\n    jp = jax.make_jaxpr(lambda xs: jax.lax.scan(lambda c, x: (c + x, None), 0.0, xs, unroll=u))(xs)\n    eqn = jp.jaxpr.eqns[0]\n    print(u, eqn.params["length"], eqn.params["unroll"], len(eqn.params["jaxpr"].jaxpr.eqns))\n\n# 1 8 1 2\n# 4 8 4 2\n# 8 8 8 2',
            },
            table: {
              caption: 'a 1024-step scan over a tanh body, jax 0.4.38 CPU: one first compile per fresh process, run time best of 50 calls; a second pass of the same script agreed within 0.02 ms and 0.12 s',
              cols: ['unroll', 'compile', 'run'],
              rows: [
                ['1', '0.33 s', '0.38 ms'],
                ['4', '0.35 s', '0.28 ms'],
                ['16', '0.42 s', '0.23 ms'],
                ['64', '1.07 s', '0.21 ms'],
              ],
            },
          },
          {
            h: 'the gradient adds a second scan, running backwards',
            ps: [
              "Differentiate a scanned function and the jaxpr grows a twin. Two scan equations come back where there was one: the forward loop with `reverse=False`, and below it another with `reverse=True` and a carry of two rather than one.",
              "The second one is the backward pass, written in the same primitive as the first. It starts from the last step and walks to the first, carrying the cotangent as state, which is why it needs a trip count known in advance and why the primitive that has one is the primitive you can differentiate.",
              "Look at `linear` on the backward equation and it reads `(True, True, False)` where the forward one read all False. Those flags mark the operands autodiff knows enter linearly, the cotangent carries here, and they exist so the transpose rule can be applied to exactly those and no others.",
              '>> The backward pass of a scan is a scan. Same primitive, other direction.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one scan in, two scans out',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef total(w, xs):\n    step = lambda c, x: (c * x + w, c)\n    last, _ = jax.lax.scan(step, 1.0, xs)\n    return last\n\nprint(jax.make_jaxpr(jax.grad(total))(jnp.float32(0.5), jnp.arange(1., 5.)))\n\n# { lambda ; a:f32[] b:f32[4]. let\n#     _:f32[] _:f32[4] = scan[\n#       _split_transpose=False\n#       jaxpr={ lambda ; c:f32[] d:f32[] e:f32[]. let\n#           f:f32[] = convert_element_type[new_dtype=float32 weak_type=False] d\n#           g:f32[] = mul f e\n#           h:f32[] = add g c\n#         in (h, d) }\n#       length=4\n#       linear=(False, False, False)\n#       num_carry=1\n#       num_consts=1\n#       reverse=False\n#       unroll=1\n#     ] a 1.0 b\n#     i:f32[] _:f32[] = scan[\n#       _split_transpose=False\n#       jaxpr={ lambda ; j:f32[] k:f32[] l:f32[]. let\n#           m:f32[] = mul k l\n#           n:f32[] = convert_element_type[new_dtype=float32 weak_type=True] m\n#           o:f32[] = add_any j k\n#         in (o, n) }\n#       length=4\n#       linear=(True, True, False)\n#       num_carry=2\n#       num_consts=0\n#       reverse=True\n#       unroll=1\n#     ] 0.0 1.0 b\n#   in (i,) }',
            },
          },
          {
            h: 'when the python loop is the right loop',
            ps: [
              "A scan runs one body against one carry type, so the steps have to be the same shape of work. A stack of layers with different widths is not: their weights cannot be stacked into a leading axis at all, and the refusal comes from `jnp.stack` long before any scan sees it. Loops like that stay Python loops, and every layer contributes its own equations to the program.",
              "How much that costs is measured next door rather than here. Chapter 12's lesson `Three shapes of loop` counts what an unrolled training loop puts in a program at four lengths, against the single equation the scanned form puts there, and it is worth reading before you commit to writing a long loop out. The count comes from the jaxpr, so it is exact and needs no compile to get.",
              "The case for writing it out anyway is when you want the steps to differ deliberately: a warmup phase with a different body, a per-layer remat policy, a debug print you only want on step zero. Wrapping those in a `cond` inside a scan works and costs you both branches every step, which lesson 1 timed. Writing them out costs you trace size once.",
              "The last chapter of this path assembles a real training run out of these pieces, and the mistake museum keeps the tracer-boolean failure that sends most people to these primitives in the first place. What this lesson leaves you with is the reading: one equation, seven params, and an operand list you can name before the loop ever runs.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): why layers of different widths cannot be scanned',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nwidths = [(jnp.ones((4, 8)), jnp.zeros(8)), (jnp.ones((8, 3)), jnp.zeros(3))]\ntry:\n    jnp.stack([w for w, _ in widths])\nexcept ValueError as err:\n    print(err)\n\n# All input arrays must have the same shape.',
            },
          },
        ],
        readings: [
          { label: 'jax.lax.scan reference', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html', note: 'the docstring defines unroll and _split_transpose; the params in a jaxpr are these arguments' },
          { label: 'Understanding jaxprs', url: 'https://docs.jax.dev/en/latest/jaxpr.html', note: 'the grammar the scan equation is written in, params and nested jaxprs included' },
          { label: 'Gradient checkpointing', url: 'https://docs.jax.dev/en/latest/gradient-checkpointing.html', note: 'what to do about the residuals the backward scan needs, when the loop is long' },
        ],
        check: [
          {
            q: 'A scan equation reads num_consts=1, num_carry=1, length=6 and takes four operands. Which operand is which?',
            a: 'The first is a closed-over constant, passed to the body whole on every step. The second is the carry, threaded in and out. The last two are sliced along their leading axis of 6, one row of each per step, which is also where the length came from.',
          },
          {
            q: 'You set unroll=16 on a 1024-step scan and the traced program looks unchanged. Did anything happen?',
            a: 'Not in the jaxpr, which keeps the same length and the same body; only the unroll param differs, and it is a note the lowering acts on. Measured here it took the run from 0.38 ms to 0.23 ms and compile from 0.33 s to 0.42 s, and it means 64 loop iterations of 16 bodies each rather than 16 copies in total.',
          },
          {
            q: 'grad of a scanned function produced two scan equations. What does the second one do, and what marks it?',
            a: 'It is the backward pass, carrying cotangents from the last step to the first. It is marked by reverse=True, a larger num_carry, and linear flags set True on the cotangent operands so the transpose rule applies to exactly those.',
          },
        ],
        work: [
          { id: 'params-ledger', label: 'trace one training step of your own as a scan and name every operand from the params alone: constant, carry or slice, before you look at the body', href: '#consts-first-then-carry-then-slices' },
          { id: 'unroll-sweep', label: 'sweep unroll on a scan you already have and write down where the run time stops improving and where the compile time starts climbing', href: '#unroll-is-a-note-for-the-lowering' },
        ],
      },
    ],
  },
]
