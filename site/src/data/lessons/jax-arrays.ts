// New file: site/src/data/lessons/jax-arrays.ts
// Below the survey chapter 1 teaches: what jax.Array is as a type, what a
// concrete array holds beside its metadata, the committed bit that decides
// placement, the weak_type flag and the promotion lattice under it, and what
// block_until_ready waits on. Every printed value ran on this machine (jax
// 0.4.38 CPU, jaxlib 0.4.38, Python 3.12.0, 2026-08-15). Every source excerpt
// is verbatim from a file the jax 0.4.38 wheel ships and is byte-identical to
// the jax-v0.4.38 tag; line numbers are deliberately absent because the wheel
// and the tagged tree disagree on offsets, so excerpts are cited by file and
// symbol instead.
import type { UnitLessons } from './index'

export const JAX_ARRAYS_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:arrays',
    lessons: [
      {
        id: 'three-classes-one-check',
        num: 1,
        title: 'Three classes, one instance check',
        lede: 'One type name covers three unrelated Python classes, two of which reach it by different routes, and one of which holds no numbers at all. Underneath the name, an array is metadata plus buffers plus a single boolean about where it is allowed to be.',
        goal: 'Say what isinstance(x, jax.Array) does and does not prove, name the two halves a concrete array is made of and what survives when one half is freed, and predict the device and the committed flag of any expression built from placed and unplaced operands.',
        sections: [
          {
            h: 'the check that says yes inside a trace',
            ps: [
              "Use `isinstance(x, jax.Array)` to find out whether you are holding real numbers and the answer will mislead you. Inside a jitted function the argument is a tracer, carrying a shape and a dtype and no values whatsoever, and the check still returns True. The class docstring promises exactly that, in a line written as an example: True both inside and outside traced functions.",
              "So the check answers a narrower question than the one people ask it. It says the object speaks the array interface. It says nothing about whether anything has been computed, which is why it is useless as a guard against tracers and fine as a type annotation. Chapter 2 owns what a tracer is and what tracing discards; the fact that belongs here is only that a tracer passes this test.",
              "A NumPy array fails the same check, and `jnp` functions accept one anyway. The union in the source that describes what they really take is `ArrayLike`, which is `Array` plus `np.ndarray` plus the Python and NumPy scalar types. What a function accepts and what `isinstance` admits are two different sets, and only the second one is `jax.Array`.",
              "`jnp.ndarray` is not a second type either. It is the same object under a second name, so the NumPy-shaped spelling of the check is the identical check, down to object identity.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): four answers, one of them from inside a trace',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nimport numpy as np\n\nx = jnp.arange(3.)\nprint(type(x).__name__, type(x).__mro__[1].__name__)\nprint(isinstance(x, jax.Array), isinstance(np.ones(3), jax.Array))\nprint(jnp.ndarray is jax.Array)\n\n@jax.jit\ndef f(v):\n    print(type(v).__name__, isinstance(v, jax.Array))\n    return v * 2\n\nf(x)\n\n# ArrayImpl object\n# True False\n# True\n# DynamicJaxprTracer True',
            },
          },
          {
            h: 'one abstract class, reached two ways',
            ps: [
              "`jax.Array` is an abstract base class with no implementation in it, and the classes that satisfy it arrive by two different routes. `ArrayImpl`, the concrete array jaxlib hands back, is registered as a virtual subclass from Python, which is why its method resolution order runs two entries long and lands on `object`. A TODO above the registration line says the plan is true inheritance at the C++ level, eventually.",
              "`Tracer` takes the other route and inherits outright, so `Array` really does appear in its method resolution order. `PRNGKeyArray`, the type `jax.random.key` returns, inherits too. Chapter 8 owns what a key is and how to fold one; the fact for here is that it answers this instance check like anything else.",
              "Three lines above the class statement, a comment says the type is not meant to include non-standard array types like KeyArray. The class statement in `prng.py` inherits from it directly. The comment predates that class and nothing updated it, so the class statements are the ones to read.",
            ],
            code: {
              caption: 'verbatim, the jax 0.4.38 wheel: the stale comment and the class statement in jax/_src/basearray.py, the registration in jax/_src/array.py, and the two class statements that inherit instead, joined here under added path headings',
              lang: 'python',
              text: '# jax/_src/basearray.py\n# Array is a type annotation for standard JAX arrays and tracers produced by\n# core functions in jax.lax and jax.numpy; it is not meant to include\n# future non-standard array types like KeyArray and BInt.\nclass Array(abc.ABC):\n\n# jax/_src/array.py, the last line of the module\n# TODO(jakevdp) replace this with true inheritance at the C++ level.\nbasearray.Array.register(ArrayImpl)\n\n# jax/_src/core.py\nclass Tracer(typing.Array, metaclass=StrictABCMeta):\n\n# jax/_src/prng.py, the type the comment above says is not included\nclass PRNGKeyArray(jax.Array):',
            },
            table: {
              caption: 'three classes that pass isinstance(x, jax.Array), with their method resolution orders read on jax 0.4.38 CPU',
              cols: ['class', 'how it satisfies jax.Array', 'mro after itself', 'holds buffers'],
              rows: [
                ['ArrayImpl, from jaxlib.xla_extension', 'registered as a virtual subclass', 'object', 'yes, one per addressable device'],
                ['DynamicJaxprTracer, from jax._src.core', 'inherits, through Tracer', 'Tracer, Array, ABC, object', 'no, it has an aval and a trace'],
                ['PRNGKeyArray, from jax._src.prng', 'inherits directly', 'Array, ABC, object', 'yes, under a key dtype'],
                ['np.ndarray', 'it does not; ArrayLike accepts it anyway', 'not applicable', 'not applicable'],
              ],
            },
          },
          {
            h: 'an aval on one side, buffers on the other',
            ps: [
              "A concrete array is two things fastened together. An abstract value, the `ShapedArray` carrying shape and dtype, and a list of device buffers with one entry per addressable device. Ask a tracer for that buffer list and the attribute is not there at all, because a tracer is the first half and never the second.",
              "Deleting proves the seam is real. `x.delete()` frees the buffers and leaves the Python object standing, so `x.shape`, `x.dtype`, and `x.sharding` all keep answering while the numbers are gone. Blocking on the same array afterward raises out of the runtime with a message naming a deleted or donated buffer. Donation, the other way buffers vanish under you, is chapter 11's to explain.",
              '>> The metadata and the memory come apart. Deleting is the operation that takes them apart on purpose.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the buffers freed, the metadata still answering',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nx = jnp.ones((4, 4))\nprint(x.is_deleted(), x.is_ready())\nx.delete()\nprint(x.is_deleted())\nprint(x.shape, x.dtype, x.sharding)\ntry:\n    x.block_until_ready()\nexcept Exception as err:\n    print(type(err).__name__)\n    print(err)\n\n# False True\n# True\n# (4, 4) float32 SingleDeviceSharding(device=CpuDevice(id=0), memory_kind=unpinned_host)\n# XlaRuntimeError\n# INVALID_ARGUMENT: BlockHostUntilReady() called on deleted or donated buffer',
            },
          },
          {
            h: 'the bit that records you meant it',
            ps: [
              "Two arrays can sit on the same device for two different reasons. One landed there because it had to land somewhere, the other because you said so, and the array remembers which. `committed` is a boolean the abstract base class requires of every implementation, and its docstring is the plainest statement of JAX's placement rules anywhere in the codebase.",
              "`jax.device_put(x, device)` sets it. `jax.device_put(x)` with no device does not, and on an array already living on a device that call is the identity function, handing back the same object. Name a device, even the device the array is already sitting on, and a different object comes back with the bit set.",
              "In that second case nothing about the bytes moved. What changed is a commitment: from here on, this array will not be relocated to meet another operand.",
            ],
            code: {
              caption: 'verbatim, the committed property docstring in jax/_src/basearray.py from the jax 0.4.38 wheel; current jax carries the same text, though the FAQ anchor it links to no longer exists in the docs',
              lang: 'python',
              text: '  @property\n  @abc.abstractmethod\n  def committed(self) -> bool:\n    """Whether the array is committed or not.\n\n    An array is committed when it is explicitly placed on device(s) via JAX\n    APIs. For example, `jax.device_put(np.arange(8), jax.devices()[0])` is\n    committed to device 0. While `jax.device_put(np.arange(8))` is uncommitted\n    and will be placed on the default device.\n\n    Computations involving some committed inputs will happen on the committed\n    device(s) and the result will be committed on the same device(s).\n    Invoking an operation on arguments that are committed to different device(s)\n    will raise an error.\n    """',
            },
          },
          {
            h: 'commitment travels through the computation',
            ps: [
              "Add an uncommitted array to a committed one and the result comes back committed, on the committed operand's device. That single rule is why one `device_put` near the top of a program can decide where a long chain of downstream arrays ends up, with none of the lines in between mentioning a device.",
              "`jax.default_device` changes where new arrays land without committing any of them. Arrays built inside the context manager land on the device you named and stay uncommitted, so they remain free to be pulled somewhere else by the first committed operand they meet. Leave the block and the array keeps the device it got, still uncommitted.",
              "Two uncommitted arrays on two different devices do not raise. They resolve onto the default device, quietly, and the result is uncommitted like both of its parents. Commitment is what turns a placement disagreement into an error, which the next section is about.",
            ],
            code: {
              caption: 'run with XLA_FLAGS=--xla_force_host_platform_device_count=2 (verified, jax 0.4.38 CPU); chapter 10 runs the same flag at eight devices to build a mesh, this needs only two',
              lang: 'python',
              text: '# run with: XLA_FLAGS=--xla_force_host_platform_device_count=2\nimport jax\nimport jax.numpy as jnp\n\nd0, d1 = jax.devices()\nu = jnp.ones(3)\nprint(u.committed, u.device)\n\nc = jax.device_put(u, d1)\nprint(c.committed, c.device, c is u)\nprint(jax.device_put(u).committed)\n\nwith jax.default_device(d1):\n    v = jnp.ones(3)\nprint(v.committed, v.device)\n\nprint((u + c).committed, (u + c).device)\nprint((u + v).committed, (u + v).device)\n\n# False TFRT_CPU_0\n# True TFRT_CPU_1 False\n# False\n# False TFRT_CPU_1\n# True TFRT_CPU_1\n# False TFRT_CPU_0',
            },
            table: {
              caption: 'placement outcomes measured on jax 0.4.38 CPU with two host-platform devices; d0 is the default device',
              cols: ['operands', 'result device', 'result committed'],
              rows: [
                ['uncommitted on d0, uncommitted on d0', 'd0', 'False'],
                ['uncommitted on d1 (via default_device), uncommitted on d0', 'd0, the default', 'False'],
                ['uncommitted on d0, committed to d1', 'd1', 'True'],
                ['committed to d1, passed through jit', 'd1', 'True'],
                ['committed to d0, committed to d1', 'nothing; it raises', 'not applicable'],
              ],
            },
          },
          {
            h: 'the one placement error there is',
            ps: [
              "Commit two arrays to two devices, add them, and JAX refuses before anything runs. The message names the primitive, both shapes, and both device id lists, which is more detail than most placement problems give you. It is a plain `ValueError`, not a runtime error out of the device like the one the deleted buffer produced.",
              "Read what the message leaves out. It never uses the word committed, so the sentence you get is a report of where the arguments were, not a diagnosis of why they could not move. The diagnosis is the previous section: each argument had been pinned, and a pinned array does not migrate to meet another operand.",
              "Which makes the fix mechanical once you can name the cause. Either drop one of the commitments, or add a `device_put` that moves one argument onto the other's device on purpose, and the copy becomes a line you wrote instead of a line JAX refused to write for you.",
            ],
            code: {
              caption: 'run with XLA_FLAGS=--xla_force_host_platform_device_count=2 (verified, jax 0.4.38 CPU): the error text in full, as printed',
              lang: 'python',
              text: '# run with: XLA_FLAGS=--xla_force_host_platform_device_count=2\nimport jax\nimport jax.numpy as jnp\n\nd0, d1 = jax.devices()\na = jax.device_put(jnp.ones(3), d0)\nb = jax.device_put(jnp.ones(3), d1)\ntry:\n    a + b\nexcept Exception as err:\n    print(type(err).__name__)\n    print(err)\n\n# ValueError\n# Received incompatible devices for jitted computation. Got argument x of add\n# with shape float32[3] and device ids [0] on platform CPU and argument y of\n# add with shape float32[3] and device ids [1] on platform CPU',
            },
          },
        ],
        readings: [
          { label: 'jax.Array in the API docs', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.Array.html', note: 'the interface as published: fifty-odd methods and sixteen attributes, sharding and committed among them' },
          { label: 'basearray.py at jax-v0.4.38', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/basearray.py', note: '190 lines: the abstract class, the ArrayLike union, and the docstrings the rest of this lesson quotes' },
          { label: 'jax.device_put', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html', note: 'the identity case with device=None spelled out, next to the committing case' },
        ],
        check: [
          {
            q: 'A function guards its input with isinstance(x, jax.Array) before touching values. Why does the guard not do what it looks like it does?',
            a: 'Because a tracer passes it. Tracer inherits from jax.Array, so the check returns True inside a jitted function where there are no values at all. The check proves the object speaks the array interface, nothing more.',
          },
          {
            q: 'You call x.delete() and then ask for x.shape. What comes back, and what would raise?',
            a: 'The shape still answers, along with dtype and sharding, because those live on the abstract value rather than in the buffers. Blocking on the array raises from the runtime with INVALID_ARGUMENT about a deleted or donated buffer.',
          },
          {
            q: 'Two uncommitted arrays sit on different devices and add without error, while two committed arrays on different devices raise. What is the difference?',
            a: 'An uncommitted array is free to be relocated, so the pair resolves onto the default device. device_put with a named device sets the committed bit, and a committed array will not move, so two of them pinned to different devices leave the operation nowhere to run.',
          },
        ],
        work: [
          { id: 'name-the-route', label: 'for three array-ish objects of your own, say before you run it whether isinstance(x, jax.Array) is True and by which of the two routes', href: '#one-abstract-class-reached-two-ways' },
          { id: 'placement-table', label: 'reproduce the placement table on two host-platform devices, predicting each device and committed flag in writing first', href: '#commitment-travels-through-the-computation' },
        ],
      },
      {
        id: 'the-weak-ones-adapt',
        num: 2,
        title: 'The weak ones adapt',
        lede: 'Weakness is not a property of Python literals. It is a flag on an abstract value that some arrays carry and others do not, it survives arithmetic, and it decides which of two operands gets converted when their dtypes disagree.',
        goal: 'Read the weak_type flag off any value, predict the dtype and the weakness of a mixed expression from the lattice, say which operand a promotion converts and why, and name the two cases where the lattice answers something the machine will not give you.',
        sections: [
          {
            h: 'two float32 scalars that are not the same scalar',
            ps: [
              "Build a scalar four ways and all four report `float32`. Two of them are weakly typed and two are not, and nothing in the dtype tells them apart. `jnp.array(1.0)` comes out weak. `jnp.array(1.0, jnp.float32)` does not, because naming the dtype is what strips the weakness. `jnp.float32(1.0)` is strong, and so is a NumPy scalar handed in from outside.",
              "The flag lives on the abstract value, not on the array, which is where to look for it. `x.aval.weak_type` reads it directly, and `jax.eval_shape` prints it as part of the `ShapeDtypeStruct` without running anything. Chapter 1 established that a Python float is weak; what it did not say is that a JAX array can be weak too, and stay weak for a long time.",
              "The last of the four is the one that changes an expression without changing anything you can see in it. A NumPy scalar is strong here even though a Python scalar is weak, so swapping `0.5` for `np.float32(0.5)` in a half-precision expression changes the output dtype. The two spellings look interchangeable and are not.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): four float32 scalars, two weak, and what each does to a float16 array',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nimport numpy as np\n\nfor label, v in [\n    ("jnp.array(1.0)", jnp.array(1.0)),\n    ("jnp.array(1.0, jnp.float32)", jnp.array(1.0, jnp.float32)),\n    ("jnp.float32(1.0)", jnp.float32(1.0)),\n    ("np.float32(1.0)", jnp.asarray(np.float32(1.0))),\n]:\n    print(f"{label:28s} {v.dtype} weak={v.aval.weak_type}")\n\nprint(jax.eval_shape(lambda a: a, jnp.array(1.0)))\nh = jnp.arange(3, dtype=jnp.float16)\nprint((h + jnp.array(1.0)).dtype, (h + jnp.float32(1.0)).dtype)\n\n# jnp.array(1.0)               float32 weak=True\n# jnp.array(1.0, jnp.float32)  float32 weak=False\n# jnp.float32(1.0)             float32 weak=False\n# np.float32(1.0)              float32 weak=False\n# ShapeDtypeStruct(shape=(), dtype=float32, weak_type=True)\n# float16 float32',
            },
          },
          {
            h: 'the lattice, written as a dict of edges',
            ps: [
              "The promotion rules are not a table of pairs. They are a directed graph, written in `dtypes.py` as a dict mapping each type to the types immediately above it, and every promotion question is answered by finding the lowest node above both operands. Two dozen lines define the whole thing.",
              "Three edges in that dict explain most of what surprises people. Weak int sits directly below the narrowest integer types, so an integer literal never forces a width on the array it meets. Weak float sits below `bf`, `f2`, and the small float types, which is why a Python float meeting a half-precision array leaves it half precision. And the int branches both terminate at weak float rather than at any concrete float, so an integer array plus a Python float does not jump to `float64`.",
              "The fourth edge worth reading is a pair. `bf: [f4]` and `f2: [f4]` put bfloat16 and float16 side by side with no ordering between them, so their only common upper bound is float32. The design note gives the reason in one sentence: bfloat16 carries a larger range at lower precision and float16 a smaller range at higher precision, and neither is a refinement of the other, so promoting between them would have to lose something either way.",
              '>> Promotion is a lookup for the lowest node above both operands. Everything else is naming.',
            ],
            code: {
              caption: 'verbatim, _type_promotion_lattice in jax/_src/dtypes.py from the jax 0.4.38 wheel, the standard branch; current jax renames the locals to bit-width form (u8 means uint8 there, uint64 here) but the edges have the same shape',
              lang: 'python',
              text: 'def _type_promotion_lattice(jax_numpy_dtype_promotion: str) -> dict[JAXType, list[JAXType]]:\n  """\n  Return the type promotion lattice in the form of a DAG.\n  This DAG maps each type to its immediately higher type on the lattice.\n  """\n  b1, = _bool_types\n  ...\n  uint4, u1, u2, u4, u8, int4, i1, i2, i4, i8 = _int_types\n  *f1_types, bf, f2, f4, f8 = _float_types\n  c4, c8 = _complex_types\n  i_, f_, c_ = _weak_types\n  if jax_numpy_dtype_promotion == \'standard\':\n    out: dict[JAXType, list[JAXType]]\n    out = {\n      b1: [i_],\n      i_: [u1, uint4, i1, int4],\n      uint4: [], u1: [i2, u2], u2: [i4, u4], u4: [i8, u8], u8: [f_],\n      int4: [], i1: [i2], i2: [i4], i4: [i8], i8: [f_],\n      f_: [*f1_types, bf, f2, c_],\n      **{t: [] for t in f1_types}, bf: [f4], f2: [f4], f4: [f8, c4], f8: [c8],\n      c_: [c4], c4: [c8], c8: [],\n    }',
            },
          },
          {
            h: 'which operand moves, in the recording',
            ps: [
              "Trace the same expression twice, once with a weak scalar and once with a strong one, and the two jaxprs disagree about who gets converted. Against a float16 array, the weak float32 scalar is converted down to float16 and the array is left alone. The strong float32 scalar leaves itself alone and converts the array up, three elements of it, to float32.",
              "The `convert_element_type` equation is the whole difference, and which line it sits on is the whole consequence. On a real array that second jaxpr doubles the bytes the operation reads and writes, which is the cost chapter 11 measures from the other end when it talks about dtype and bandwidth.",
              "One detail in the printout is worth flagging so it does not mislead later. The aval notation prints `f32[]` for both scalars, with no mark for weakness, so a jaxpr will never tell you which kind you passed. Chapter 2 teaches reading these; `jax.eval_shape` is the instrument for the part the notation drops.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same add, traced twice, with the convert on opposite sides; both jaxprs are verbatim, with the two call lines added above them as comments and a blank line between',
              lang: 'haskell',
              text: '# jax.make_jaxpr(lambda a, b: a + b)(jnp.array(1.0), h)   # weak scalar\n{ lambda ; a:f32[] b:f16[3]. let\n    c:f16[] = convert_element_type[new_dtype=float16 weak_type=False] a\n    d:f16[3] = add c b\n  in (d,) }\n\n# jax.make_jaxpr(lambda a, b: a + b)(jnp.float32(1.0), h)  # strong scalar\n{ lambda ; a:f32[] b:f16[3]. let\n    c:f32[3] = convert_element_type[new_dtype=float32 weak_type=False] b\n    d:f32[3] = add a c\n  in (d,) }',
            },
          },
          {
            h: 'sixteen expressions, predicted then read',
            ps: [
              "Cover the right two columns and work down the list. Every row is decided by the lattice plus one extra rule, which is that weakness survives an operation only when no strong operand took part, and dies the moment one does.",
              "The propagation rows at the bottom are the ones worth slowing down for. `jnp.sin` of a weak scalar is still weak, and a weak plus a weak is still weak, so weakness travels an arbitrary distance through a computation. `astype` is the call that ends it, which is how you fix a dtype at a point in the program where you want the adapting to stop.",
              "Row eight is the one that reaches furthest. An int32 array plus a Python float comes back weak float32, so an integer array that touched a literal is still adapting several lines later, and will still bend to a float16 array it meets downstream.",
            ],
            table: {
              caption: 'sixteen expressions run on jax 0.4.38 CPU; h is jnp.arange(3, dtype=jnp.float16), b is bfloat16, f is float32, i is jnp.arange(3), u is uint8, m is a bool array',
              cols: ['expression', 'dtype', 'weak'],
              rows: [
                ['h + 1.0', 'float16', 'False'],
                ['h + jnp.array(1.0)', 'float16', 'False'],
                ['h + jnp.float32(1.0)', 'float32', 'False'],
                ['h + np.float32(1.0)', 'float32', 'False'],
                ['h + f', 'float32', 'False'],
                ['b + h', 'float32', 'False'],
                ['i + 1', 'int32', 'False'],
                ['i + 1.0', 'float32', 'True'],
                ['i / 2', 'float32', 'False'],
                ['i * jnp.float16(2)', 'float16', 'False'],
                ['u + 1', 'uint8', 'False'],
                ['u + 300', 'uint8', 'False'],
                ['m + 1', 'int32', 'True'],
                ['jnp.sin(jnp.array(1.0))', 'float32', 'True'],
                ['jnp.array(1.0) + jnp.array(1.0)', 'float32', 'True'],
                ['jnp.array(1.0).astype(jnp.float32)', 'float32', 'False'],
              ],
            },
          },
          {
            h: 'when the lattice answers wider than the machine will go',
            ps: [
              "`jnp.promote_types('int8', 'uint32')` returns int64, and adding those two arrays gives you int32. Both answers are correct in their own frame: the lattice says int64 because the smallest node above a signed 8-bit and an unsigned 32-bit type is a signed 64-bit one, and then x64 is off, so the result is narrowed on the way out. No warning is printed.",
              "The other narrow case runs the opposite direction. `u + 300` on a uint8 array stays uint8 and wraps, giving 44 where you asked for 300, because a weak integer literal adapts to the array rather than promoting it. NumPy 2.2.6 raises `OverflowError` on the identical expression. Chapter 1 has the other silent-loss story, the one where a literal too large for float32 becomes inf; this is a different mechanism arriving at the same kind of quiet.",
              "One habit catches both. Ask the lattice what it intends with `jnp.promote_types`, then check what the array actually reports, and treat any disagreement between the two as a decision you now have to make explicitly.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, numpy 2.2.6): the lattice answer, the runtime answer, and what NumPy does with the same line',
              lang: 'python',
              text: 'import jax.numpy as jnp\nimport numpy as np\n\ni8 = jnp.arange(3, dtype=jnp.int8)\nu32 = jnp.arange(3, dtype=jnp.uint32)\nprint(jnp.promote_types(\'int8\', \'uint32\'), (i8 + u32).dtype)\n\nu8 = jnp.arange(3, dtype=jnp.uint8)\nprint((u8 + 300).dtype, u8 + 300)\ntry:\n    np.arange(3, dtype=np.uint8) + 300\nexcept Exception as err:\n    print(type(err).__name__, err)\n\n# int64 int32\n# uint8 [44 45 46]\n# OverflowError Python integer 300 out of bounds for uint8',
            },
          },
          {
            h: 'the setting that turns a promotion into an error',
            ps: [
              "Set `jax_numpy_dtype_promotion` to strict and the lattice collapses to the weak types alone. Python scalars still adapt to whatever array they meet, so ordinary code with literals in it keeps working. Any implicit conversion between two real dtypes stops being allowed and raises `TypePromotionError` instead, naming both inputs.",
              "Turning it on across a codebase is a search tool rather than a permanent setting. Every raise marks a line where two dtypes met and one of them silently changed, and each one is then a choice: cast on purpose, or fix the dtype further upstream where it was decided.",
              "The knob also has a context-manager form, which is the more usable shape for this. Wrap the section of a model you actually care about instead of the whole program, and the failures come back scoped to the code you were reading.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): weak scalars still adapt under strict, array-to-array promotion does not; the two dtype names come out of a set, so their order inside the message swaps between runs and the rest of it does not',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\njax.config.update("jax_numpy_dtype_promotion", "strict")\nh = jnp.arange(3, dtype=jnp.float16)\nb = jnp.arange(3, dtype=jnp.bfloat16)\nprint((h + 1.0).dtype)\ntry:\n    b + h\nexcept Exception as err:\n    print(type(err).__name__)\n    print(err)\n\n# float16\n# TypePromotionError\n# Input dtypes (\'bfloat16\', \'float16\') have no available implicit dtype\n# promotion path when jax_numpy_dtype_promotion=strict. Try explicitly casting\n# inputs to the desired output type, or set jax_numpy_dtype_promotion=standard.',
            },
          },
        ],
        readings: [
          { label: 'Type promotion semantics', url: 'https://docs.jax.dev/en/latest/type_promotion.html', note: 'the promotion tables as published, with the weak-type columns marked' },
          { label: 'Design of Type Promotion Semantics for JAX', url: 'https://docs.jax.dev/en/latest/jep/9407-type-promotion.html', note: 'why the lattice has this shape: the four options considered, and why bfloat16 and float16 stay unordered' },
          { label: 'dtypes.py at jax-v0.4.38', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/dtypes.py', note: 'the lattice as edges, the upper-bound closure built from it, and the result_type path both feed' },
        ],
        check: [
          {
            q: 'jnp.array(1.0) and jnp.float32(1.0) both report float32. Added to a float16 array, why do they give different dtypes?',
            a: 'The first is weakly typed and the second is not. A weak float sits below float16 in the lattice, so it converts down and leaves the array at float16; a strong float32 sits above it, so the array converts up and the result is float32.',
          },
          {
            q: 'Why do bfloat16 and float16 promote to float32 rather than to one another?',
            a: 'Because the lattice gives them no edge between them: bf and f2 each point only at f4, so their lowest common upper bound is float32. The design note explains the deliberate choice, since bfloat16 trades precision for range and float16 the reverse, and neither refines the other.',
          },
          {
            q: 'promote_types says int64 and the array you get is int32, with nothing printed. What happened?',
            a: 'The lattice answered correctly for the two input types, and then x64 being off narrowed the result on the way out. The lattice and the runtime can disagree, silently, so checking promote_types against the array dtype is how you catch it.',
          },
        ],
        work: [
          { id: 'weak-audit', label: 'read weak_type off six values in your own code with jax.eval_shape and mark which of them you expected to be weak', href: '#two-float32-scalars-that-are-not-the-same-scalar' },
          { id: 'strict-sweep', label: 'run one training step under jax_numpy_dtype_promotion=strict and decide, for each raise, whether to cast or to fix the dtype upstream', href: '#the-setting-that-turns-a-promotion-into-an-error' },
        ],
      },
      {
        id: 'what-blocking-waits-on',
        num: 3,
        title: 'What blocking actually waits on',
        lede: 'block_until_ready is four lines long and none of them is a device sync. It waits on the buffers of one array, returns that array, and in two situations that come up in real benchmarking code it does nothing at all.',
        goal: 'State what block_until_ready waits on and what it leaves running, use is_ready to observe dispatch without a timer, and name the two ways a block in a harness can silently become a no-op.',
        sections: [
          {
            h: 'four lines, and none of them a device sync',
            ps: [
              "Open the method and there is less machinery than the name suggests. It checks the array has not been deleted, walks its list of device buffers, blocks on each one, and returns the array itself. No device queue is drained, no other computation is waited on, and nothing is copied to the host.",
              "Returning `self` is what makes the harness idiom work. `jax.block_until_ready(f(*args))` reads as one expression because the call hands the array back, so a warmup line can block and bind on one line. Chapter 11 owns the harness; the reason the idiom composes is this line.",
              "The buffer list is the part that generalizes. On one device it has one entry, so the loop runs once. On a sharded array it has one entry per addressable device, and the call returns when the slowest of them is ready, which is the sense in which it waits for a computation and not for a device.",
            ],
            code: {
              caption: 'verbatim, ArrayImpl.block_until_ready in jax/_src/array.py from the jax 0.4.38 wheel',
              lang: 'python',
              text: '  @use_cpp_method()\n  def block_until_ready(self):\n    self._check_if_deleted()\n    for db in self._arrays:\n      db.block_until_ready()\n    return self',
            },
          },
          {
            h: 'is_ready asks the same question without waiting',
            ps: [
              "Every array carries a non-blocking poll beside the blocking one. `is_ready()` returns whether the buffers are done, right now, and never waits, which makes asynchronous dispatch something you can watch rather than something you infer from a stopwatch.",
              "Dispatch a large matmul on an already-compiled function and the poll says False on the line right after it. Block, and it says True. Two booleans, no timer, and the observation does not vary with what else the machine happens to be doing, which a timing number would.",
              "Compilation has to be out of the way first or the poll measures the wrong thing. The warmup call in the snippet is there for that reason, and chapter 11 is where compilation-in-the-measurement gets its full treatment.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU; identical across six runs): dispatch, poll, block, poll',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nmm = jax.jit(lambda a: a @ a)\nx = jnp.ones((3000, 3000))\nmm(x).block_until_ready()      # compile, then start clean\n\ny = mm(x)\nprint(y.is_ready())\nprint(y.block_until_ready() is y)\nprint(y.is_ready())\n\n# False\n# True\n# True',
            },
          },
          {
            h: 'one array, not one device',
            ps: [
              "Dispatch a small computation and then a large one, and block on the small one's output. The large one is still running when the call returns, and its poll says so. Nothing about blocking on an array drains the work queued behind it.",
              "That is worth checking against the reverse order, because the reverse order looks like the same experiment and is not. Dispatch the large one first, block on the small one's output, and the large one has finished too, since a single CPU device runs its queue in dispatch order. The independence you can observe here is one-directional, and reading only the second case would teach the wrong rule.",
              "For a benchmark the consequence is direct. If a step produces several outputs and the harness blocks on one of them, it has timed the path to that one output, not the step. Blocking the whole pytree of results is the version that measures what the loop actually does.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU; identical across six runs): the second dispatch is still in flight when the first one is blocked and returned',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nmm = jax.jit(lambda a: a @ a)\nbig, small = jnp.ones((3000, 3000)), jnp.ones((64, 64))\nmm(big).block_until_ready()\nmm(small).block_until_ready()\n\ns = mm(small)                  # dispatched first\nb = mm(big)                    # dispatched second\nprint(s.block_until_ready() is s)\nprint(b.is_ready())\nprint(b.block_until_ready().is_ready())\n\n# True\n# False\n# True',
            },
          },
          {
            h: 'the pytree wrapper skips what it cannot block',
            ps: [
              "The module-level `jax.block_until_ready` is the version to use on a step's outputs, and it does three things the method cannot. It flattens a pytree, it batches the arrays into a single runtime call rather than blocking them one at a time, and it hands the original object back so the call sits anywhere in an expression.",
              "The skipping is where care is needed. Any leaf without a `block_until_ready` method is passed to a helper that catches `AttributeError` and returns it unchanged, so NumPy arrays, Python floats, and strings pass through unchanged and unmentioned. Hand it a pytree with no JAX arrays in it at all and the function does nothing and reports nothing, which in a harness reads exactly like a successful block.",
              "The batched path is a real difference for a step with many outputs. One `batched_block_until_ready` call across the whole list beats a Python loop calling the method per array, and defaulting to the module-level function is how you get that without deciding it at each call site.",
            ],
            code: {
              caption: 'verbatim, jax.block_until_ready in jax/_src/api.py from the jax 0.4.38 wheel, the docstring trimmed',
              lang: 'python',
              text: 'def block_until_ready(x):\n  """Tries to call a ``block_until_ready`` method on pytree leaves."""\n  def try_to_block(x):\n    try:\n      return x.block_until_ready()\n    except AttributeError:\n      return x\n\n  arrays = []\n  for leaf in tree_leaves(x):\n    if isinstance(leaf, array.ArrayImpl):\n      arrays.append(leaf)\n    else:\n      try_to_block(leaf)\n\n  if not arrays:\n    # `arrays` will be empty if tree_leaves(x) is empty or all leaves are not\n    # jax.Array.\n    pass\n  elif len(arrays) == 1:\n    # Fast path for single array.\n    try_to_block(arrays[0])\n  else:\n    # Optimized for multiple arrays.\n    xc.batched_block_until_ready(arrays)\n\n  return x',
            },
          },
          {
            h: 'a tracer refuses, and the refusal is deliberate',
            ps: [
              "Put a block inside a jitted function and it never runs as a block. `Tracer` defines `block_until_ready` as a property that raises `AttributeError`, and the comment above it says why: raising that particular exception keeps `hasattr` and `getattr` checks working the way callers expect.",
              "Which means the module-level wrapper, whose helper catches exactly `AttributeError`, treats a tracer as an unblockable leaf and moves on without a word. A `jax.block_until_ready` call inside a jitted function is a no-op with no diagnostic, and it is a plausible thing to write when a harness gets refactored and a block ends up on the wrong side of the decorator.",
              "The second no-op is the one from the previous section, a pytree that turned out to hold no JAX arrays. Both fail the same way, quietly and while looking correct, so a harness worth trusting asserts on the arrays it meant to block rather than assuming the call found them.",
              '>> A block that found nothing to block looks exactly like a block that worked.',
            ],
            code: {
              caption: 'verbatim, the Tracer property in jax/_src/core.py from the jax 0.4.38 wheel, then a run on this machine (jax 0.4.38 CPU)',
              lang: 'python',
              text: '  @property\n  def block_until_ready(self):\n    # Raise AttributeError for backward compatibility with hasattr() and getattr() checks.\n    raise AttributeError(self,\n      f"The \'block_until_ready\' method is not available on {self._error_repr()}."\n      f"{self._origin_msg()}")\n\n# >>> params = {"w": jnp.ones((2, 2)), "cached": np.ones(4), "lr": 0.01}\n# >>> out = jax.block_until_ready(params)\n# >>> out is params, out["w"] is params["w"]\n# (True, True)\n#\n# >>> @jax.jit\n# ... def step(v):\n# ...     try:\n# ...         v.block_until_ready()\n# ...     except AttributeError:\n# ...         print("a tracer has no block_until_ready")\n# ...     return v * 2\n# >>> step(jnp.ones(3)).block_until_ready()\n# a tracer has no block_until_ready',
            },
          },
          {
            h: 'the wait that block_until_ready is not',
            ps: [
              "Side effects have their own wait. `jax.effects_barrier()` is one line, and it blocks on the runtime's effect tokens rather than on any array, so it is the call that waits for a `debug.print` or an `io_callback` to have actually happened. Blocking on a function's output array says nothing about them, because the output buffer and the effect token are separate things to be ready.",
              "Nor does blocking fetch anything. `block_until_ready` leaves the values where they are; `np.asarray(x)` is the call that moves them to the host, and `copy_to_host_async` is the call that starts that move without waiting for it. On this CPU backend the difference costs nothing, since the buffer is already host memory, so the separation is easiest to keep straight by reading the three call sites rather than by timing them here.",
              "Three waits, then, with three different meanings. One array's buffers, the whole runtime's pending effects, and a transfer to the host. Using the first when you needed one of the other two is what produces a benchmark or a log that is wrong without saying so.",
            ],
            code: {
              caption: 'verbatim, jax.effects_barrier in jax/_src/api.py from the jax 0.4.38 wheel',
              lang: 'python',
              text: 'def effects_barrier():\n  """Waits until existing functions have completed any side-effects."""\n  dispatch.runtime_tokens.block_until_ready()',
            },
          },
        ],
        readings: [
          { label: 'Asynchronous dispatch', url: 'https://docs.jax.dev/en/latest/async_dispatch.html', note: 'the official page, including the note that blocking without transferring back to Python is the faster measurement' },
          { label: 'jax.block_until_ready', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html', note: 'the pytree signature, which is the form that batches and the form worth defaulting to' },
          { label: 'array.py at jax-v0.4.38', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/array.py', note: 'ArrayImpl in full: the buffer list, the deletion check, and copy_to_host_async next door' },
        ],
        check: [
          {
            q: 'A step returns three arrays and the harness blocks on the first one. What has been timed?',
            a: 'The path to that one array. block_until_ready walks only that array’s buffers, so work producing the other two can still be in flight when the call returns. Blocking the whole returned pytree is what times the step.',
          },
          {
            q: 'A refactor moved a jax.block_until_ready call inside the jitted function. What does it do there, and what does it report?',
            a: 'Nothing, and nothing. Tracer raises AttributeError from the property on purpose, and the pytree wrapper catches exactly AttributeError and moves on, so the block silently disappears with no diagnostic.',
          },
          {
            q: 'You blocked on a jitted function’s output and a jax.debug.print from inside it has not appeared. Which call do you need?',
            a: 'jax.effects_barrier, which blocks on the runtime’s effect tokens rather than on any array. Output buffers and effect tokens become ready independently, so blocking on the result says nothing about the side effects.',
          },
        ],
        work: [
          { id: 'poll-not-time', label: 'reproduce the dispatch-then-poll observation on a function of your own, using is_ready rather than a timer', href: '#is-ready-asks-the-same-question-without-waiting' },
          { id: 'harness-audit', label: 'take one benchmarking harness you have written and check every block for the two no-op cases: a tracer, and a pytree with no jax arrays in it', href: '#a-tracer-refuses-and-the-refusal-is-deliberate' },
        ],
      },
    ],
  },
]
