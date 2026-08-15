// New file: site/src/data/lessons/jax-autodiff.ts
// The machinery under jax.grad, below the survey chapter 4: jvp as the only rule a
// primitive registers, linearize as the place residuals appear, transposition as
// where reverse mode comes from, the two cost models measured rather than asserted,
// custom_jvp against a gradient that is nan where the value is fine, and what
// stop_gradient removes from a derived program. Every printed value ran on this
// machine (jax 0.4.38 CPU, CPython 3.12, 2026-08-15) and is quoted verbatim. The two
// source excerpts are byte-identical to jax/_src/api.py in the 0.4.38 wheel; line
// numbers are that file's. Flop counts come from XLA's own cost model through
// cost_analysis(), which is an estimate the compiler reports, not a hardware counter;
// they repeat exactly across runs on this machine.
import type { UnitLessons } from './index'

export const JAX_AUTODIFF_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:autodiff',
    lessons: [
      {
        id: 'the-rule-under-grad',
        num: 1,
        title: 'The rule under grad',
        lede: 'A primitive in JAX registers one derivative rule, and it points forward. Reverse mode is that same rule linearized and then run backwards, which you can do by hand in four lines.',
        goal: 'Read a jvp jaxpr against the program it came from, name the arrays linearize stores and what each one is, and rebuild grad from linearize and linear_transpose without calling grad.',
        sections: [
          {
            h: 'a tangent goes in, a tangent comes out',
            ps: [
              "`jax.jvp` takes three things and returns two. The function, a tuple of primal arguments, a tuple of tangents matching them one for one, and back come the value and the directional derivative at that value. No Jacobian is built anywhere in that call, which is why it costs a forward pass rather than a Jacobian's worth of them.",
              "The run below checks the tangent against a Jacobian built the slow way. `jax.jacrev(f)(x) @ v` assembles the full 2-by-2 matrix and multiplies; `jax.jvp` never assembles anything. On this input the two agree bit for bit, which is the difference between a derivative rule composed exactly and a finite difference that would agree to a few digits at best.",
              "One shape detail decides how you call it. The primals and tangents are tuples over the function's positional arguments, not over the array's elements, so a one-argument function takes `(x,)` and `(v,)` with the commas that make them tuples. Forgetting a comma is the most common way this call fails.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one tangent pushed through, checked against an assembled Jacobian',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef f(x):\n    return jnp.stack([jnp.sin(x[0]) * x[1], x[0] ** 2 + x[1]])\n\nx = jnp.array([0.5, 2.0])\nv = jnp.array([1.0, 0.0])\n\ny, dy = jax.jvp(f, (x,), (v,))\nprint(y, dy)\nprint(jnp.array_equal(dy, jax.jacrev(f)(x) @ v))\n\n# [0.9588511 2.25     ] [1.7551651 1.       ]\n# True',
            },
          },
          {
            h: 'the jvp program has no second half',
            ps: [
              "Trace `sin(x) * x` and you get two equations. Trace its jvp and you get seven, and every one of them is either a primal equation you already had or a tangent equation sitting next to it. `cos a` appears because the sine rule needs it. The two `mul` equations feeding `add_any` are the product rule, one term per factor, added together at the end.",
              "Read the output line of the jvp jaxpr and the structure is plain: `in (f, i)`, the primal result and the tangent result, produced by one program in one pass. There is no second program, no tape, and nothing held back for later. A tangent is consumed by the next equation the moment it is produced.",
              "That is why forward mode's memory cost is the same as the forward pass's. It also explains why the cost scales with the number of directions you ask about: one tangent vector per call, and the tangent equations run once for each.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same function, plain and under jvp; jaxpr printing is an internal representation, so a different jax release may print different parameter lines',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef f(x):\n    return jnp.sin(x) * x\n\nx = jnp.arange(3.0)\nprint(jax.make_jaxpr(f)(x))\nprint(jax.make_jaxpr(lambda x, v: jax.jvp(f, (x,), (v,)))(x, jnp.ones(3)))\n\n# { lambda ; a:f32[3]. let b:f32[3] = sin a; c:f32[3] = mul b a in (c,) }\n# { lambda ; a:f32[3] b:f32[3]. let\n#     c:f32[3] = sin a\n#     d:f32[3] = cos a\n#     e:f32[3] = mul b d\n#     f:f32[3] = mul c a\n#     g:f32[3] = mul e a\n#     h:f32[3] = mul c b\n#     i:f32[3] = add_any g h\n#   in (f, i) }',
            },
          },
          {
            h: 'linearize freezes the primal half and hands you the rest',
            ps: [
              "`jax.linearize(f, x)` runs the primal half once and gives you back the value plus a function that is linear in the tangent. The docs describe it in one line as producing a linear approximation using `jvp()` and partial eval, and the jaxpr of what comes back shows exactly which half got evaluated.",
              "Look at where the semicolon falls. Three arrays sit before it, as constvars, and only one variable sits after it, the tangent. Those three constants are the residuals: the values from the forward pass that the linear program still needs. Print them and they are `cos(x)`, `x`, and `sin(x)`, in that order, which are precisely the three factors the product rule and the sine rule asked for.",
              "This is the memory that reverse mode is famous for spending, made countable. It is not a mystery buffer inside the framework; it is a list of arrays you can print, and the next lesson counts them for a deep block.",
              '>> The residuals are not hidden. They are the constvars of the linearized program.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the linear part of one function, and the three arrays it kept',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef f(x):\n    return jnp.sum(jnp.sin(x) * x)\n\nx = jnp.arange(3.0)\ny, f_lin = jax.linearize(f, x)\nlin = jax.make_jaxpr(f_lin)(jnp.ones(3))\nprint(lin)\nprint([c.tolist() for c in lin.consts])\n\n# { lambda a:f32[3] b:f32[3] c:f32[3]; d:f32[3]. let\n#     e:f32[3] = mul d a\n#     f:f32[3] = mul e b\n#     g:f32[3] = mul c d\n#     h:f32[3] = add_any f g\n#     i:f32[] = reduce_sum[axes=(0,)] h\n#   in (i,) }\n# [[1.0, 0.5403022766113281, -0.416146844625473], [0.0, 1.0, 2.0], [0.0, 0.8414709568023682, 0.9092974066734314]]',
            },
          },
          {
            h: 'transpose the linear program and grad falls out',
            ps: [
              "A linear function has a transpose, and JAX will build it for you. Hand `jax.linear_transpose` the linear part from the previous section and the point it was linearized at, feed the transposed function a cotangent of 1.0, and what comes back is the gradient, element for element the same array `jax.grad` returns.",
              "So a primitive never registers a backward rule. It registers a forward one, JAX linearizes the composition, and transposition turns that linear program around. Reverse mode is a consequence of two transformations rather than a second differentiation algorithm sitting beside the first.",
              "The Pallas arc reaches this same decomposition from the other end, quoting the design document on why a kernel with overlapping parallel reads transposes into something slow. Same three steps, read there as a limit on what can be compiled; read here as where your gradient comes from.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): grad, rebuilt without calling grad',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef f(x):\n    return jnp.sum(jnp.sin(x) * x)\n\nx = jnp.arange(3.0)\n_, f_lin = jax.linearize(f, x)\nprint(jax.linear_transpose(f_lin, x)(1.0))\nprint(jax.grad(f)(x))\n\n# (Array([0.        , 1.3817732 , 0.07700372], dtype=float32),)\n# [0.         1.3817732  0.07700372]',
            },
          },
          {
            h: 'a tangent has a dtype, and integers have none',
            ps: [
              "Tangents live in a different space from primals, and JAX enforces the match rather than casting quietly. Hand a float16 tangent to a float32 primal and the call raises with a message that names both dtypes and the one it expected. Nothing is promoted for you, because a promotion here would change the numerics of a derivative you asked to be exact.",
              "Integer primals get the interesting answer. The tangent space of an integer array has no directions in it, so JAX gives it a dtype called `float0` whose itemsize really is zero bytes. Differentiate through an integer with `allow_int=True` and what comes back is a `float0` array of the right shape: a stated absence, not a buffer of zeros you might mistake for a gradient.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the dtype rule, and the zero-byte tangent type',
              lang: 'python',
              text: 'import numpy as np\nimport jax\nimport jax.numpy as jnp\nfrom jax import dtypes\n\ntry:\n    jax.jvp(lambda t: t * 2.0, (jnp.arange(3.0),), (jnp.ones(3, dtype=jnp.float16),))\nexcept TypeError as err:\n    print(err)\n\nn = jnp.arange(3)\nprint(jax.jvp(lambda t: t * 2, (n,), (np.zeros(3, dtype=dtypes.float0),))[1].dtype)\nprint(np.dtype(dtypes.float0).itemsize)\nprint(jax.grad(lambda v: jnp.sum(v * 2.0), allow_int=True)(n).dtype)\n\n# primal and tangent arguments to jax.jvp do not match; dtypes must be equal, or in\n# case of int/bool primal dtype the tangent dtype must be float0.Got primal dtype\n# float32 and so expected tangent dtype float32, but got tangent dtype float16 instead.\n# [(\'float0\', \'V\')]\n# 0\n# [(\'float0\', \'V\')]',
            },
          },
        ],
        readings: [
          { label: 'jax.linearize', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html', note: 'the one-line definition this lesson unpacks: a linear approximation from jvp and partial eval' },
          { label: 'jax.linear_transpose', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.linear_transpose.html', note: 'the other half of the rebuild, with the rules about what counts as linear' },
          { label: 'Autodidax: JAX core from scratch', url: 'https://docs.jax.dev/en/latest/autodidax.html', note: 'jvp rules, partial eval and transposition implemented in the open, in that order' },
        ],
        check: [
          {
            q: 'jax.jvp returned a tangent equal to the Jacobian times your vector. Why was no Jacobian built?',
            a: 'Because each primitive contributes a tangent equation next to its primal equation, and those run once for the one direction you passed in. The Jacobian would be that same program run once per input direction, which is what jacfwd does.',
          },
          {
            q: 'You linearize a function and print the linear part as a jaxpr. What are the constvars?',
            a: 'The residuals: the forward-pass values the linear program still needs. For sum(sin(x) * x) they are cos(x), x and sin(x), and they are the memory reverse mode holds between the forward and backward halves.',
          },
          {
            q: 'If every primitive registers only a forward rule, where does reverse mode come from?',
            a: 'From linearizing the composition and transposing the linear program that results. jax.linear_transpose on the output of jax.linearize, fed a cotangent of 1.0, reproduces jax.grad exactly.',
          },
        ],
        work: [
          { id: 'rebuild-grad', label: 'rebuild grad for a function of your own out of linearize and linear_transpose, and check it element for element against jax.grad', href: '#transpose-the-linear-program-and-grad-falls-out' },
          { id: 'name-residuals', label: 'print the constvars of one linearized function you wrote and name each array in terms of your source line', href: '#linearize-freezes-the-primal-half-and-hands-you-the-rest' },
        ],
      },
      {
        id: 'which-direction-costs-what',
        num: 2,
        title: 'Which direction costs what',
        lede: 'Forward and reverse compute the same derivative and send you different bills. Both bills are readable before you run a training step: one as a flop count off the compiled program, one as a list of arrays.',
        goal: 'Predict from a function’s input and output widths which of jacfwd and jacrev costs less, say what reverse mode holds in memory and how that grows with depth, and measure both for a function of your own.',
        sections: [
          {
            h: 'the gradient stays at twice the forward pass',
            ps: [
              "A compiled JAX program will tell you what it thinks it costs. `jax.jit(fn).lower(x).compile().cost_analysis()` hands back a list with one dictionary in it, and `flops` is one of its keys. It is XLA's own cost model rather than a hardware counter, so treat it as the compiler's arithmetic and not as a measurement of your machine. It repeats exactly across runs, which makes it a good instrument for comparing two programs.",
              "Run it on the same function at three input widths and the shape of the cost model appears. The gradient sits at almost exactly twice the forward pass every time: 2.03, then 2.01, then 2.00. Widen the input by four and the ratio does not move, because reverse mode makes one backward sweep no matter how many inputs feed it.",
              "`jacfwd` on the same function tells the opposite story. It costs 19 forward passes at 16 inputs, 68 at 64, and 260 at 256. That is the same tangent program run once per input direction, and the table below is what the chapter's cost argument looks like when you make the compiler count it.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): flop counts off the compiled program, at one width',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef flops(fn, *args):\n    return jax.jit(fn).lower(*args).compile().cost_analysis()[0][\'flops\']\n\nn = 64\nW = jnp.ones((n, n)) * 0.01\n\ndef f(x):\n    return jnp.sum(jnp.tanh(x @ W))\n\nx = jnp.ones(n)\nprint(flops(f, x), flops(jax.grad(f), x))\nprint(flops(jax.jacfwd(f), x), flops(jax.jacrev(f), x))\n\n# 8255.0 16576.0\n# 557056.0 16576.0',
            },
            table: {
              caption: 'f(x) = sum(tanh(x @ W)) with W square at each width; flops as XLA’s cost model reports them (jax 0.4.38 CPU), ratios computed from those numbers',
              cols: ['inputs', 'forward', 'grad', 'grad / forward', 'jacfwd', 'jacfwd / forward'],
              rows: [
                ['16', '527', '1072', '2.03', '10240', '19.4'],
                ['64', '8255', '16576', '2.01', '557056', '67.5'],
                ['256', '131327', '262912', '2.00', '34078720', '259.5'],
              ],
            },
          },
          {
            h: 'the aspect ratio picks the direction',
            ps: [
              "Turn the function around and the winner turns around with it. A map from 4 inputs to 256 outputs costs 13,600 flops through `jacfwd` and 788,736 through `jacrev`, a factor of 58 in favour of the direction that was hopeless a moment ago.",
              "The rule behind both tables is one sentence about basis vectors. Forward mode runs once per input direction, reverse mode runs once per output direction, and the Jacobian needs every direction on one side or the other. Count inputs, count outputs, pick the smaller one.",
              "Deep learning sits at one extreme of that rule and never moves: a billion parameters in, one loss out. The interesting cases are the ones that are not training steps, where a small parameter vector drives a large simulated trajectory and forward mode is the cheap direction by two orders of magnitude.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): 4 inputs, 256 outputs, and the two Jacobian builders',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef flops(fn, *args):\n    return jax.jit(fn).lower(*args).compile().cost_analysis()[0][\'flops\']\n\nW = jnp.ones((4, 256)) * 0.01\n\ndef g(x):\n    return jnp.tanh(x @ W)\n\nx = jnp.ones(4)\nprint(flops(g, x))\nprint(flops(jax.jacfwd(g), x), flops(jax.jacrev(g), x))\nprint(jax.jacfwd(g)(x).shape)\n\n# 2048.0\n# 13600.0 788736.0\n# (256, 4)',
            },
          },
          {
            h: 'both Jacobians are one direction, batched',
            ps: [
              "Open the wheel and the two builders are four lines each. `jacfwd` partially applies `jvp` at the point you passed, then vmaps it over `_std_basis(dyn_args)`. `jacrev` calls `_vjp`, keeps the pullback, and vmaps that over `_std_basis(y)`. Same shape of code, one built from the forward primitive and one from the reverse one.",
              "`_std_basis` is where the flop ratios come from. It flattens the tree, counts the elements, and builds `jnp.eye(ndim)`: an identity basis with one row per element of the thing it was handed. `jacfwd` gets the basis of the inputs, so it runs n times. `jacrev` gets the basis of the outputs, so it runs m times. The docs put the same fact in four words each, calling `jacfwd` column-by-column and `jacrev` row-by-row.",
              "The `out_axes=(None, -1)` on the forward one is worth a second look. The primal output is shared across the batch rather than stacked, and the tangents stack on the last axis, which is what puts a Jacobian's columns where columns belong.",
            ],
            code: {
              caption: 'verbatim, jax/_src/api.py in the jax 0.4.38 wheel: the decisive line of jacfwd at 581-582, of jacrev at 669 and 673, and _std_basis at 763-769, joined here under added headings',
              lang: 'python',
              text: '# jax/_src/api.py:581-582, inside jacfwd\n      pushfwd: Callable = partial(_jvp, f_partial, dyn_args)\n      y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))\n\n# jax/_src/api.py:669 and 673, inside jacrev\n      y, pullback = _vjp(f_partial, *dyn_args)\n    jac = vmap(pullback)(_std_basis(y))\n\n# jax/_src/api.py:763-769, the basis both of them batch over\ndef _std_basis(pytree):\n  import jax.numpy as jnp\n  leaves, _ = tree_flatten(pytree)\n  ndim = sum(map(np.size, leaves))\n  dtype = dtypes.result_type(*leaves)\n  flat_basis = jnp.eye(ndim, dtype=dtype)\n  return _unravel_array_into_pytree(pytree, 1, None, flat_basis)',
            },
          },
          {
            h: 'what reverse mode holds while it waits',
            ps: [
              "The flop side of the bill favours reverse mode. The memory side is where it pays, and lesson one already showed where to look: the constvars of the linearized program. Count them for a block of tanh layers and the growth is exactly linear, 3 arrays at one layer and 17 at eight.",
              "Print the shapes rather than the count and the arithmetic stops being abstract. One 128 by 128 weight matrix at 64 KiB, held once, plus two 32 by 128 activations per layer at 16 KiB each. Every layer you add costs 32 KiB at this batch size and nothing else, so a batch of 512 instead of 32 would cost 512 KiB per layer by the same rule.",
              "Forward mode holds none of this, which is the other half of the trade the previous sections measured in flops. A tangent is consumed by the next equation; a residual has to survive until the backward sweep reaches it. LAB·J3 takes the next step from here, trading those bytes back for recomputation.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the residual arrays of a tanh block, counted and sized',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nW = jnp.ones((128, 128)) * 0.01\nx = jnp.ones((32, 128))\n\ndef residuals(depth):\n    def block(x):\n        for _ in range(depth):\n            x = jnp.tanh(x @ W)\n        return jnp.sum(x)\n    _, lin = jax.linearize(block, x)\n    avals = [v.aval for v in jax.make_jaxpr(lin)(x).jaxpr.constvars]\n    return len(avals), sum(a.size * a.dtype.itemsize for a in avals)\n\nfor depth in (1, 2, 4, 8):\n    print(depth, residuals(depth))\n\n# 1 (3, 98304)\n# 2 (5, 131072)\n# 4 (9, 196608)\n# 8 (17, 327680)',
            },
            table: {
              caption: 'residuals of depth layers of tanh(x @ W), W float32[128,128], x float32[32,128] (jax 0.4.38 CPU); the shapes column is the aval list printed for depth 2',
              cols: ['depth', 'residual arrays', 'bytes', 'what they are'],
              rows: [
                ['1', '3', '98304', 'W, plus one layer’s input and its tanh output'],
                ['2', '5', '131072', 'W once, two arrays per layer'],
                ['4', '9', '196608', 'same rule: 2 * depth + 1'],
                ['8', '17', '327680', '65536 for W, 32768 per layer'],
              ],
            },
          },
        ],
        readings: [
          { label: 'jax.jacfwd', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html', note: 'the column-by-column wording, and the argnums rules this lesson skips' },
          { label: 'jax.jacrev', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.jacrev.html', note: 'the row-by-row twin, including what allow_int changes' },
          { label: 'jax/_src/api.py at jax-v0.4.38', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/api.py', note: 'the file the excerpt is cut from; read jacfwd, jacrev and _std_basis together' },
        ],
        check: [
          {
            q: 'Your function maps 8 inputs to 4096 outputs and you need the whole Jacobian. Which builder, and what are you avoiding?',
            a: 'jacfwd, because forward mode runs once per input direction and reverse mode once per output direction. On a 4-input, 256-output function the measured gap was 13,600 flops against 788,736, a factor of 58, and it widens with the output width.',
          },
          {
            q: 'Why does the gradient of a scalar loss cost about two forward passes whatever the input width?',
            a: 'Because one cotangent sweeps backward once, regardless of how many inputs feed it. Measured at 16, 64 and 256 inputs the grad-to-forward flop ratio was 2.03, 2.01 and 2.00, while jacfwd went from 19 forward passes to 260.',
          },
          {
            q: 'You linearize a 12-layer block of the shape measured here. How many residual arrays, and what sets the bytes?',
            a: '25, from the rule of two per layer plus the shared weight matrix. The bytes are one 64 KiB weight matrix plus 32 KiB per layer at batch 32, since each layer keeps its input and its tanh output at 16 KiB each.',
          },
        ],
        work: [
          { id: 'flops-of-your-own', label: 'write the flops helper, then predict the grad-to-forward ratio for a function of your own before you print it', href: '#the-gradient-stays-at-twice-the-forward-pass' },
          { id: 'residual-ledger', label: 'count the residual arrays of one block you wrote and name what each array is for, in your own source lines', href: '#what-reverse-mode-holds-while-it-waits' },
        ],
      },
      {
        id: 'when-you-write-the-rule',
        num: 3,
        title: 'When you write the rule',
        lede: 'A gradient can be nan while the value beside it is exactly right. The repair is not a smaller step or a bigger epsilon; it is handing JAX the derivative you already know, in the direction the rest of your program needs.',
        goal: 'Recognize a gradient that fails where the value does not, choose custom_jvp or custom_vjp from what the surrounding code has to support, and defend a hand-written rule against finite differences.',
        sections: [
          {
            h: 'the value is zero and the gradient is nan',
            ps: [
              "The Euclidean norm is three lines of arithmetic with no special functions in it. Ask for its value at the origin and you get 0.0, which is correct. Ask for its gradient there and every component comes back nan.",
              "The algebra says why without any appeal to floating point. The derivative of the norm is x divided by the norm, and at the origin that is zero over zero. JAX is not wrong here; it evaluated the rule it was given at a point where the rule has no value.",
              "What matters for a real model is that this is a region and not a single point. At 1e-20 the squares underflow to zero in float32, so the norm is zero while x is not, and the first component comes back inf while the others come back nan. Any padded row, masked token, or freshly zeroed embedding walks into that region on the first step.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same norm at three magnitudes, value then gradient',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef norm(x):\n    return jnp.sqrt(jnp.sum(x ** 2))\n\nfor pt in ([0.0, 0.0, 0.0], [1e-20, 0.0, 0.0], [1e-3, 0.0, 0.0]):\n    p = jnp.array(pt)\n    print(norm(p).item(), jax.grad(norm)(p).tolist())\n\n# 0.0 [nan, nan, nan]\n# 0.0 [inf, nan, nan]\n# 0.0010000000474974513 [1.0, 0.0, 0.0]',
            },
          },
          {
            h: 'why masking it first does not help',
            ps: [
              "The reflex is to guard the output: compute the norm, and where the input was zero, return zero instead. It does exactly nothing for the gradient. `jnp.where` evaluates both of its branches and then selects, so the nan tangent from the branch you discarded is still multiplied by a zero from the selection, and nan times zero is nan.",
              "The fix the FAQ prescribes is to guard the input as well, so the unsafe operation never sees the value it cannot handle. Substitute a harmless 1.0 inside the square root, then select on the same predicate outside it. Written that way the gradient at the origin is a clean zero.",
              "It works and it is fragile in a specific way: the predicate is now written twice, and anyone editing either copy has to know that the inner one exists for the derivative rather than for the value. When the derivative is something you can state outright, saying so is the more durable repair.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one where, then the nested pair the FAQ prescribes',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef masked(x):\n    s = jnp.sum(x ** 2)\n    return jnp.where(s > 0.0, jnp.sqrt(s), 0.0)\n\ndef twice_masked(x):\n    s = jnp.sum(x ** 2)\n    safe = jnp.where(s > 0.0, s, 1.0)\n    return jnp.where(s > 0.0, jnp.sqrt(safe), 0.0)\n\nzero = jnp.zeros(3)\nprint(masked(zero), jax.grad(masked)(zero))\nprint(twice_masked(zero), jax.grad(twice_masked)(zero))\n\n# 0.0 [nan nan nan]\n# 0.0 [0. 0. 0.]',
            },
          },
          {
            h: 'one rule, both directions',
            ps: [
              "`jax.custom_jvp` replaces the forward rule for a function, and lesson one is why that is the load-bearing one: every other mode is built from it. The decorated rule takes the primals and tangents as two tuples and returns the primal output beside the tangent output, which for a norm is the projection of the tangent onto the unit vector, `dot(x, dx) / |x|`.",
              "Two things are true about the guard inside that rule. It is written once, in the derivative, where it belongs. And it encodes a choice rather than a fact: the norm has no derivative at the origin, so returning zero there picks the subgradient of smallest length, which is the one that leaves an optimizer standing still instead of producing nan.",
              "With the rule in place the gradient at the origin is zero, the gradient at (3, 4, 0) is the unit vector (0.6, 0.8, 0), and `jax.jvp` works on the same function, because a forward rule is what forward mode wanted in the first place.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the rule, at the boundary point and away from it',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\n@jax.custom_jvp\ndef norm(x):\n    return jnp.sqrt(jnp.sum(x ** 2))\n\n@norm.defjvp\ndef norm_jvp(primals, tangents):\n    (x,), (dx,) = primals, tangents\n    y = norm(x)\n    safe = jnp.where(y > 0.0, y, 1.0)\n    return y, jnp.where(y > 0.0, jnp.dot(x, dx) / safe, 0.0)\n\np = jnp.array([3.0, 4.0, 0.0])\nprint(jax.grad(norm)(jnp.zeros(3)))\nprint(jax.grad(norm)(p))\nprint(jax.jvp(norm, (p,), (jnp.array([1.0, 0.0, 0.0]),)))\n\n# [0. 0. 0.]\n# [0.6 0.8 0. ]\n# (Array(5., dtype=float32), Array(0.6, dtype=float32))',
            },
          },
          {
            h: 'custom_vjp buys one direction only',
            ps: [
              "Write the same repair as a `custom_vjp` and reverse mode is happy: the gradient at (3, 4, 0) is the same unit vector. Then ask for a jvp of it and the answer is a refusal in one sentence, `can't apply forward-mode autodiff (jvp) to a custom_vjp function.`",
              "That refusal propagates further than it first appears. Anything that needs a forward pass over the gradient is closed off too, which includes second derivatives and every Hessian-vector product, since those are a jvp of a grad. The next lesson measures what you would have been buying there.",
              "So the choice is not a matter of taste. Reach for `custom_jvp` when the derivative is something you can state going forward, and the function keeps working under every transformation and every order. Reach for `custom_vjp` when only the backward rule exists as a separate program: a fused kernel whose backward is its own kernel, which is what the Pallas arc and the flash-attention stage are about.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same repair through custom_vjp, and the direction it will not give you',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\n@jax.custom_vjp\ndef norm_v(x):\n    return jnp.sqrt(jnp.sum(x ** 2))\n\ndef fwd(x):\n    y = jnp.sqrt(jnp.sum(x ** 2))\n    safe = jnp.where(y > 0.0, y, 1.0)\n    return y, jnp.where(y > 0.0, x / safe, 0.0)\n\ndef bwd(res, ct):\n    return (ct * res,)\n\nnorm_v.defvjp(fwd, bwd)\n\np = jnp.array([3.0, 4.0, 0.0])\nprint(jax.grad(norm_v)(p))\ntry:\n    jax.jvp(norm_v, (p,), (jnp.ones(3),))\nexcept TypeError as err:\n    print(err)\n\n# [0.6 0.8 0. ]\n# can\'t apply forward-mode autodiff (jvp) to a custom_vjp function.',
            },
            table: {
              caption: 'what each decorator gives you, from the runs in this lesson (jax 0.4.38 CPU)',
              cols: ['ask', 'custom_jvp', 'custom_vjp'],
              rows: [
                ['jax.grad', 'works, through linearize and transpose', 'works, the rule is the backward pass'],
                ['jax.jvp', 'works', 'raises: no forward-mode rule exists'],
                ['jax.hessian, jvp of grad', 'works', 'raises, for the same reason'],
                ['you choose the residuals', 'no, JAX linearizes your rule', 'yes, fwd returns them explicitly'],
                ['the backward is its own program', 'not expressible', 'the case it exists for'],
              ],
            },
          },
          {
            h: 'prove it against finite differences',
            ps: [
              "A hand-written rule is a claim, and `jax.test_util.check_grads` is how you test it. The docs describe it as checking gradients from automatic differentiation against finite differences, and `order=2` runs the check on the derivative of the derivative as well. On the custom rule above, at (3, 4, 0), it passes.",
              "Then look at the Hessian it produces, because there is a closed form to compare against. The exact answer is the projection away from the unit vector divided by the norm, which for this point is 0.128 and 0.072 on the diagonal, -0.096 off it, and 0.2 in the direction with no component. Those are the printed numbers, which means the rule survived being differentiated twice and stayed correct.",
              "One guard rail comes free. A rule that returns a tangent of the wrong shape raises before it can hand you a wrong number, and the message names both the shape it expected and the shape it got.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the check, the second derivative, and what a wrong rule shape raises',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nfrom jax.test_util import check_grads\n\n@jax.custom_jvp\ndef norm(x):\n    return jnp.sqrt(jnp.sum(x ** 2))\n\n@norm.defjvp\ndef norm_jvp(primals, tangents):\n    (x,), (dx,) = primals, tangents\n    y = norm(x)\n    safe = jnp.where(y > 0.0, y, 1.0)\n    return y, jnp.where(y > 0.0, jnp.dot(x, dx) / safe, 0.0)\n\ncheck_grads(norm, (jnp.array([3.0, 4.0, 0.0]),), order=2)\nprint(\'order 2 ok\')\nprint(jax.hessian(norm)(jnp.array([3.0, 4.0, 0.0])))\n\n@jax.custom_jvp\ndef bad(x):\n    return jnp.sum(x ** 2)\n\n@bad.defjvp\ndef bad_jvp(primals, tangents):\n    (x,), (dx,) = primals, tangents\n    return bad(x), 2.0 * x * dx\n\ntry:\n    jax.jvp(bad, (jnp.ones(3),), (jnp.ones(3),))\nexcept TypeError as err:\n    print(err)\n\n# order 2 ok\n# [[ 0.128 -0.096  0.   ]\n#  [-0.096  0.072  0.   ]\n#  [ 0.     0.     0.2  ]]\n# Custom JVP rule must produce primal and tangent outputs with corresponding shapes\n# and dtypes. Expected float32[] (tangent type of float32[]) but got float32[3].',
            },
          },
        ],
        readings: [
          { label: 'jax.custom_jvp', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html', note: 'defjvp and defjvps, and the nondiff_argnums rules this lesson leaves alone' },
          { label: 'FAQ · Gradients contain NaN where using where', url: 'https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where', note: 'the nested-where prescription, stated by the people who wrote the transform' },
          { label: 'jax.test_util.check_grads', url: 'https://docs.jax.dev/en/latest/jax.test_util.html', note: 'the finite-difference check, with the order and tolerance arguments' },
          { label: 'Advanced autodiff', url: 'https://docs.jax.dev/en/latest/advanced-autodiff.html', note: 'both decorators in full, including closures and integer arguments' },
        ],
        check: [
          {
            q: 'A loss that calls a Euclidean norm returns a clean number and its gradient is nan. Where do you look first?',
            a: 'At the inputs that make the norm zero or nearly zero, since the derivative is x over the norm and that is zero over zero at the origin. In float32 the failure covers a region, not a point: at 1e-20 the squares underflow and the gradient comes back inf and nan.',
          },
          {
            q: 'Why does wrapping the unstable operation in a single jnp.where leave the gradient nan?',
            a: 'Because where evaluates both branches and then selects. The discarded branch still produces a nan tangent, and multiplying that by the zero the selection contributes leaves nan. Guarding the input as well, so the unsafe operation never sees the bad value, is what fixes it.',
          },
          {
            q: 'When do you need custom_jvp rather than custom_vjp?',
            a: 'Whenever anything downstream needs forward mode: a jvp, a Hessian, a Hessian-vector product, any second derivative. custom_vjp refuses all of those with "can\'t apply forward-mode autodiff (jvp) to a custom_vjp function"; it earns its place when the backward pass is a separate program, like a fused kernel.',
          },
        ],
        work: [
          { id: 'defend-a-rule', label: 'take one function of your own that can reach a boundary point, write a custom_jvp for it, and check_grads it at order 2', href: '#prove-it-against-finite-differences' },
          { id: 'both-directions', label: 'run jax.jvp and jax.hessian against your rule; if either raises, write down which decorator you picked and what forced it', href: '#custom-vjp-buys-one-direction-only' },
        ],
      },
      {
        id: 'stopping-and-stacking',
        num: 4,
        title: 'Stopping and stacking',
        lede: 'Two things you can do once a gradient is a program you can read: delete a term from it deliberately, and run the transformation again over what came out.',
        goal: 'Predict which product-rule terms stop_gradient removes and read that in the jaxpr, write a straight-through estimator and say what it costs, and choose between a dense Hessian and a Hessian-vector product on measured flops.',
        sections: [
          {
            h: 'one factor of the product rule, removed',
            ps: [
              "Multiply a vector by itself and differentiate: the answer is 2x, one term per factor. Now wrap the second factor in `lax.stop_gradient`. The value is unchanged at 14.0, and the gradient is x rather than 2x.",
              "The gradient jaxpr shows what happened with no interpretation required. `stop_gradient a` runs in the forward half, the product uses its output, and the backward half has exactly one multiply, against that stopped value. The term that would have differentiated the second factor was never emitted, because there was no rule to emit it through.",
              "Reading it as deletion from a program, rather than as zeroing something at runtime, is what makes the effect predictable. Nothing is multiplied by zero and nothing is masked. One path through the derived program does not exist.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same product, with and without the cut, and the program that results',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nfrom jax import lax\n\ndef cut(x):\n    return jnp.sum(x * lax.stop_gradient(x))\n\nx = jnp.array([1.0, 2.0, 3.0])\nprint(cut(x), jax.grad(cut)(x))\nprint(jax.grad(lambda v: jnp.sum(v * v))(x))\nprint(jax.make_jaxpr(jax.grad(cut))(x))\n\n# 14.0 [1. 2. 3.]\n# [2. 4. 6.]\n# { lambda ; a:f32[3]. let\n#     b:f32[3] = stop_gradient a\n#     c:f32[3] = mul a b\n#     _:f32[] = reduce_sum[axes=(0,)] c\n#     d:f32[3] = broadcast_in_dim[\n#       broadcast_dimensions=()\n#       shape=(3,)\n#       sharding=None\n#     ] 1.0\n#     e:f32[3] = mul d b\n#   in (e,) }',
            },
          },
          {
            h: 'a value forward, a different derivative back',
            ps: [
              "Rounding has a derivative of zero wherever it has one at all, so a model with a rounding step in it learns nothing through that step. `jnp.round` differentiates to zeros, as the run below confirms.",
              "The construction that gets around it is `x + stop_gradient(round(x) - x)`. The value is exactly `round(x)`, since the added term is the difference. The derivative is the identity, because the only part of the expression with a live gradient is the leading x. Quantizers and discrete samplers are built from this pattern.",
              "Be precise about the price. The gradient you get is not an approximation of this function's gradient; it is the exact gradient of a different function, chosen because it is useful for optimization. That is a modelling decision made in the code, and the only place it is written down is the `stop_gradient` call itself.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the value of a rounding step and the derivative of a linear one',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nfrom jax import lax\n\ndef ste(x):\n    return x + lax.stop_gradient(jnp.round(x) - x)\n\ny = jnp.array([0.2, 1.7, -0.5])\nprint(ste(y))\nprint(jax.grad(lambda v: jnp.sum(ste(v)))(y))\nprint(jax.grad(lambda v: jnp.sum(jnp.round(v)))(y))\n\n# [0. 2. 0.]\n# [1. 1. 1.]\n# [0. 0. 0.]',
            },
          },
          {
            h: 'differentiate the derivative',
            ps: [
              "`jax.grad` returns a function, and that function is a program of the same kind as the one it came from, so `grad` applies to it again. Stack four of them on `sin` and what prints is sin, cos, -sin, -cos at the same point, matching the library's own values digit for digit.",
              "Nothing about the second application is special-cased. The gradient program was assembled out of the same per-primitive rules as the program it came from, so those rules apply to it in turn, which is also what gives an order-2 `check_grads` something to check.",
              "Where the stacking gets interesting is the order you stack in, and what the composition costs. The next two sections are those two questions.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): four derivatives of sin at one point, against the closed forms',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nd0 = jnp.sin\nd1 = jax.grad(d0)\nd2 = jax.grad(d1)\nd3 = jax.grad(d2)\nprint(d0(0.7), d1(0.7), d2(0.7), d3(0.7))\nprint(jnp.sin(0.7), jnp.cos(0.7), -jnp.sin(0.7), -jnp.cos(0.7))\n\n# 0.64421767 0.7648422 -0.64421767 -0.7648422\n# 0.64421767 0.7648422 -0.64421767 -0.7648422',
            },
          },
          {
            h: 'hessian is two transforms, in that order',
            ps: [
              "`jax.hessian` is not separate machinery. Its whole body in the wheel is one expression, `jacfwd(jacrev(fun, ...), ...)`, and reading the order gives you the shape. `jacrev` produces the gradient as a row, then `jacfwd` pushes one tangent per input through that gradient program and fills the matrix column by column.",
              "Whether the other order would cost more is a question you can put to the compiler instead of arguing about. Priced with the counter from lesson two, `jacfwd(jacrev(loss))` and `jacrev(jacrev(loss))` come out the same on a 32-input loss: 147,552 flops each, 12,288 bytes of scratch each, and the same matrix. On this function XLA does not separate them, so take the order in the body as the one JAX fixes for you and not as a measured win.",
              "The order also explains the refusal from the previous lesson. A Hessian needs a forward pass over a gradient, so a function whose only custom rule is a backward one cannot supply one at all.",
            ],
            code: {
              caption: 'verbatim, jax/_src/api.py:760-761 in the jax 0.4.38 wheel: the whole body of jax.hessian, docstring above it omitted',
              lang: 'python',
              text: '  return jacfwd(jacrev(fun, argnums, has_aux=has_aux, holomorphic=holomorphic),\n                argnums, has_aux=has_aux, holomorphic=holomorphic)',
            },
          },
          {
            h: 'the dense Hessian you probably do not need',
            ps: [
              "Stacking transforms is cheap to write and expensive to run, and the same flop counter from lesson two prices it. On a 32-input loss the forward pass is 2,111 flops, the gradient 4,256, and the full Hessian 147,552: about 35 gradients to fill a 32 by 32 matrix.",
              "Most methods that want curvature never want the matrix. Newton-style solvers, conjugate gradient, and Gauss-Newton approximations all consume the Hessian one product at a time, and one product is `jax.jvp(jax.grad(loss), (x,), (u,))[1]`: push a tangent forward through the gradient program. That costs 6,624 flops, a little over one and a half gradients, and 22 times less than building the matrix.",
              "The two agree to 7.5e-09 in the largest component, which is float32 rounding rather than a different answer. When the input is a model rather than a 32-vector, the matrix is not merely expensive but unrepresentable, and the product is the only form that exists.",
              '>> A Hessian-vector product is a jvp of a grad. Both transforms you already have, composed in the order that skips the matrix.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): four flop counts and one agreement check',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef flops(fn, *args):\n    return jax.jit(fn).lower(*args).compile().cost_analysis()[0][\'flops\']\n\nW = jnp.ones((32, 32)) * 0.01\n\ndef loss(x):\n    return jnp.sum(jnp.tanh(x @ W) ** 2)\n\nx = jnp.ones(32)\nu = jnp.arange(32.) / 32.0\n\nprint(flops(loss, x), flops(jax.grad(loss), x))\nprint(flops(jax.hessian(loss), x))\nprint(flops(lambda x, u: jax.jvp(jax.grad(loss), (x,), (u,))[1], x, u))\n\nhvp = jax.jvp(jax.grad(loss), (x,), (u,))[1]\nprint(float(jnp.max(jnp.abs(hvp - jax.hessian(loss)(x) @ u))))\n\n# 2111.0 4256.0\n# 147552.0\n# 6624.0\n# 7.450580596923828e-09',
            },
            table: {
              caption: 'loss(x) = sum(tanh(x @ W) ** 2), W float32[32,32]; flops as XLA’s cost model reports them (jax 0.4.38 CPU)',
              cols: ['program', 'flops', 'against the gradient'],
              rows: [
                ['loss(x)', '2111', '0.50x'],
                ['jax.grad(loss)', '4256', '1.00x'],
                ['jvp of grad, one product', '6624', '1.56x'],
                ['jax.hessian, the full matrix', '147552', '34.7x'],
              ],
            },
          },
        ],
        readings: [
          { label: 'jax.lax.stop_gradient', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html', note: 'the primitive itself, with the worked example of a squared value that differentiates to zero' },
          { label: 'jax.hessian', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html', note: 'the pytree rules for what shape a Hessian of a nested params tree even has' },
          { label: 'The autodiff cookbook', url: 'https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html', note: 'Hessian-vector products derived three ways, with the one this lesson measures' },
        ],
        check: [
          {
            q: 'sum(x * stop_gradient(x)) differentiates to x rather than 2x. What does the gradient jaxpr look like?',
            a: 'It carries the stop_gradient equation in the forward half and exactly one multiply in the backward half, against the stopped value. The second product-rule term is absent from the program rather than zeroed at runtime.',
          },
          {
            q: 'A straight-through estimator returns round(x) and differentiates as the identity. What did you give up?',
            a: 'Any claim that the gradient belongs to the function you evaluated. It is the exact gradient of x, used in place of the rounding step\'s real derivative, which is zero everywhere it exists. The substitution is a modelling choice recorded only in the stop_gradient call.',
          },
          {
            q: 'You need H times v for a 32-input loss. Why not build H first?',
            a: 'Because the product is a jvp of a grad and costs 6,624 flops against 147,552 for the matrix, 22 times less, and the two agreed to 7.5e-09 here. At model scale the matrix cannot be stored at all, while the product costs about 1.6 gradients.',
          },
        ],
        work: [
          { id: 'stop-gradient-jaxpr', label: 'print the gradient jaxpr for one stop_gradient of your own and name the equation that is missing from it', href: '#one-factor-of-the-product-rule-removed' },
          { id: 'hvp-instead', label: 'write a Hessian-vector product as a jvp of a grad and compare its flop count with jax.hessian on the same loss', href: '#the-dense-hessian-you-probably-do-not-need' },
        ],
      },
    ],
  },
]
