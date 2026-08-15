// New file: site/src/data/lessons/jax-vmap.ts
// The batching interpreter below the survey chapter 5 teaches: the tracer that
// carries one integer, the per-primitive rule table it dispatches through, where
// an axis move gets paid for, and what the composition with grad and jit looks
// like on the jaxpr. Every printed value ran on this machine (jax 0.4.38 CPU,
// jaxlib 0.4.38, numpy 2.2.6, Python 3.12, 2026-08-15). Every excerpt from
// jax/_src/interpreters/batching.py is verbatim from the installed file, which is
// byte-identical to the jax-v0.4.38 tag, so the quoted line numbers hold against
// the public tree. Where current jax has moved, the caption says so.
import type { UnitLessons } from './index'

export const JAX_VMAP_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:vmap',
    lessons: [
      {
        id: 'the-rule-per-primitive',
        num: 1,
        title: 'The rule per primitive',
        lede: 'There is no batched copy of your function anywhere, and no loop. There is a tracer carrying one extra integer, and a dictionary from primitive to rule that the tracer looks things up in.',
        goal: 'Say what a BatchTracer holds and what your function sees instead, walk the interpreter’s four-way dispatch for a given primitive, and predict what happens to a primitive that has no rule registered.',
        sections: [
          {
            h: 'a tracer that carries one integer',
            ps: [
              "Print the type of an argument from inside a vmapped function and a `BatchTracer` comes back with three slots on it: the real value, an integer naming which of that value's axes is the batch, and a source location kept for error messages. The array underneath is the entire batch. Only the integer says which axis it is.",
              "The tracer then hides that axis from you. Map axis 1 of a (3, 7, 5) array and the tracer reports its shape as (3, 5) while the value it holds is still (3, 7, 5) with `batch_dim` 1. You wrote the function for one example, and one example is the shape it gets handed, which is why the body needs no rewriting.",
              "The second argument in the run below is the more interesting one. Give it `None` in `in_axes` and it does not arrive as a tracer at all; it arrives as an `ArrayImpl`, the concrete array, unwrapped. An unbatched argument is not a batch of size one and not a broadcast copy. It is simply not in the transformation.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): what the mapped argument is, and what the unmapped one is not',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef peek(x, y):\n    print(type(x).__name__, x.shape, x.val.shape, x.batch_dim)\n    print(type(y).__name__, y.shape)\n    return x * y\n\njax.vmap(peek, in_axes=(1, None))(jnp.ones((3, 7, 5)), jnp.ones(5))\njax.vmap(peek, in_axes=(0, None))(jnp.ones((7, 3, 5)), jnp.ones(5))\n\n# BatchTracer (3, 5) (3, 7, 5) 1\n# ArrayImpl (5,)\n# BatchTracer (3, 5) (7, 3, 5) 0\n# ArrayImpl (5,)',
            },
          },
          {
            h: 'the sentinel is the None you typed',
            ps: [
              "The annotation on that integer field is `NotMapped | int | RaggedAxis`, and `NotMapped` is defined two lines above as `type(None)`. So the `None` you write in `in_axes` and the internal marker for a value with no batch axis are the same object, aliased as `not_mapped` for readability. A TODO sitting above the alias asks for a real sentinel type instead.",
              "The `aval` property is where the axis disappears. When `batch_dim` is an integer it calls `core.mapped_aval`, which returns the abstract value with that axis deleted, and that abstract value is what every primitive inside your function checks its shapes against. Nothing downstream has to know a batch exists.",
              "That TODO has since been answered, in the direction of deleting the alias rather than replacing it. On jax's current main the name `not_mapped` does not appear in the file at all and the code compares against `None` directly, so if you read the modern file expecting the alias, it is gone rather than moved.",
            ],
            code: {
              caption: 'verbatim, jax/_src/interpreters/batching.py at jax 0.4.38: the sentinel at 377-381, the tracer’s fields at 384-397, and the aval property at 399-405; on current main the not_mapped alias is deleted and the comparisons read `is None`',
              lang: 'python',
              text: '### tracer\n\n# TODO(mattjj): use a special sentinel type rather than None\nNotMapped = type(None)\nnot_mapped = None\n\n\nclass BatchTracer(Tracer):\n  __slots__ = [\'val\', \'batch_dim\', \'source_info\']\n\n  def __init__(self, trace, val, batch_dim: NotMapped | int | RaggedAxis,\n               source_info: source_info_util.SourceInfo | None = None):\n    if config.enable_checks.value:\n      assert type(batch_dim) in (NotMapped, int, RaggedAxis)\n      if type(batch_dim) is int:\n        aval = core.get_aval(val)\n        assert 0 <= batch_dim < len(aval.shape)\n    self._trace = trace\n    self.val = val\n    self.batch_dim = batch_dim\n    self.source_info = source_info\n\n  @property\n  def aval(self):\n    aval = core.get_aval(self.val)\n    if self.batch_dim is not_mapped:\n      return aval\n    elif type(self.batch_dim) is int:\n      return core.mapped_aval(aval.shape[self.batch_dim], self.batch_dim, aval)',
            },
          },
          {
            h: 'the dispatch, in twenty lines',
            ps: [
              "Every primitive your function calls lands in one function, and that function is short enough to hold in your head. It collects the values and their batch dims, then takes one of four branches, and which branch it takes is decided entirely by two dictionary lookups and one boolean.",
              "Read the `elif args_not_mapped` branch before the others, because it is the one people do not expect. When no argument carries a batch dim, the primitive binds on the parent trace with the raw values, no rule is consulted, and the result is not wrapped in a tracer. Work that does not touch the batch stays outside the transformation entirely, at no cost.",
              "The last branch is the refusal. A primitive in neither table raises `NotImplementedError`, with the primitive's own name in the message. Batching is opt-in per primitive, and the opt-in is a dictionary entry.",
            ],
            code: {
              caption: 'verbatim, jax/_src/interpreters/batching.py:460-482 at jax 0.4.38, BatchTrace.process_primitive down to the refusal; the tracer-wrapping tail at 483-491 is trimmed',
              lang: 'python',
              text: '  def process_primitive(self, p, tracers, params):\n    if config.dynamic_shapes.value:\n      p.abstract_eval(*(map(core.get_aval, tracers)), **params)\n    vals_in, dims_in = unzip2(map(self.to_batch_info, tracers))\n    args_not_mapped = all(bdim is not_mapped for bdim in dims_in)\n    if p in fancy_primitive_batchers:\n      if (args_not_mapped\n          and p in skippable_batchers\n          and not any(self.axis_data.name == axis_name\n                      for axis_name in skippable_batchers[p](params))):\n        # no-op shortcut\n        return p.bind_with_trace(self.parent_trace, vals_in, params)\n      else:\n        with core.set_current_trace(self.parent_trace):\n          val_out, dim_out = fancy_primitive_batchers[p](self.axis_data, vals_in, dims_in, **params)\n    elif args_not_mapped:\n      # no-op shortcut\n      return p.bind_with_trace(self.parent_trace, vals_in, params)\n    elif p in primitive_batchers:\n      with core.set_current_trace(self.parent_trace):\n        val_out, dim_out = primitive_batchers[p](vals_in, dims_in, **params)\n    else:\n      raise NotImplementedError("Batching rule for \'{}\' not implemented".format(p))',
            },
            table: {
              caption: 'the four branches of process_primitive, read off the source above',
              cols: ['when', 'what runs', 'what comes back'],
              rows: [
                ['the primitive is in fancy_primitive_batchers', 'that rule, handed axis_data first', 'a value and a batch dim, wrapped in a BatchTracer'],
                ['no argument is mapped (and no fancy rule claims the axis)', 'the primitive itself, on the parent trace', 'a plain value, not wrapped at all'],
                ['the primitive is in primitive_batchers', 'that rule, handed values and dims', 'a value and a batch dim, wrapped in a BatchTracer'],
                ['neither table has it', 'nothing', 'NotImplementedError naming the primitive'],
              ],
            },
          },
          {
            h: 'a rule is three lines when the primitive is elementwise',
            ps: [
              "`sin` does not care where the batch axis sits, because it treats every element the same. Its rule asserts that all the incoming batch dims agree, binds the primitive on the batched values unchanged, and hands the same dim back out. Three lines, and `defvectorized` is the one-liner that installs it.",
              "Binary primitives need a little more, because their two operands can disagree about where the batch is or whether there is one. The broadcasting rule moves each batched operand's axis to the front and gives each unmapped operand a size-1 axis there instead, then leans on the primitive's own broadcasting to finish. A comment above that line says exactly why the inserted axis has size 1 rather than the batch size.",
              "Reductions are the third shape of rule, and they only have to fix up a parameter. Adding a batch axis at position 0 shifts every axis in the `axes` parameter up by one, which is bookkeeping, not new work: `reduce_sum[axes=(0,)]` over a (3, 5) becomes `reduce_sum[axes=(1,)]` over a (4, 3, 5).",
            ],
            code: {
              caption: 'verbatim, jax/_src/interpreters/batching.py at jax 0.4.38: the elementwise rule at 945-950, then the size-1 comment inside broadcast_batcher at 973-978',
              lang: 'python',
              text: 'def defvectorized(prim):\n  primitive_batchers[prim] = partial(vectorized_batcher, prim)\n\ndef vectorized_batcher(prim, batched_args, batch_dims, **params):\n  assert all(batch_dims[0] == bd for bd in batch_dims[1:]), batch_dims\n  return prim.bind(*batched_args, **params), batch_dims[0]\n\n  else:\n    # We pass size of 1 here because (1) at least one argument has a real batch\n    # dimension and (2) all unmapped axes can have a singleton axis inserted and\n    # then rely on the primitive\'s built-in broadcasting.\n    args = [bdim_at_front(x, d, 1) if np.ndim(x) else x\n            for x, d in zip(args, dims)]',
            },
          },
          {
            h: 'two tables, and what the second one knows',
            ps: [
              "Count the tables and there are two, holding 180 and 21 entries on this machine once one vmap has run. The counts move with what has been imported, since a rule registers when its module loads, so treat those two numbers as a reading rather than a constant.",
              "What separates the tables is one extra argument. A rule in the second table is handed `axis_data` first, carrying the mapped axis's name and size, and the twenty-one primitives that need it are exactly the ones for which the identity of the axis matters: the collectives, the control-flow primitives, `pjit`, `remat2`, and the custom-derivative calls.",
              "`sin`, `add`, `dot_general`, `reduce_sum` and `transpose` all sit in the first table, and so does `pallas_call`. A Pallas kernel is batchable because somebody wrote it a rule, not because vmap can see inside it.",
              "Current jax has collapsed this split. On main every rule lives in the fancy table, `primitive_batchers` survives only as a compatibility proxy that wraps old-style rules, and `defvectorized` now registers into the fancy table. The two-argument shape of a rule is the one that won.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the two tables counted after one warm vmap, then the whole second table by name',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nimport jax._src.interpreters.batching as B\n\njax.vmap(lambda x: jnp.sum(jnp.tanh(x)))(jnp.ones((2, 2)))\nprint(len(B.primitive_batchers), len(B.fancy_primitive_batchers))\nprint(sorted(p.name for p in B.fancy_primitive_batchers))\n\n# 180 21\n# [\'all_gather\', \'all_to_all\', \'axis_index\', \'cond\', \'custom_linear_solve\',\n#  \'custom_vjp_call_jaxpr\', \'host_local_array_to_global_array\', \'pbroadcast\',\n#  \'pgather\', \'pjit\', \'pmax\', \'pmin\', \'ppermute\', \'psum\', \'psum2\',\n#  \'reduce_scatter\', \'remat2\', \'remat_opt\', \'scan\', \'sharding_constraint\',\n#  \'while\']',
            },
          },
          {
            h: 'a primitive with no rule at all',
            ps: [
              "Define a primitive of your own with an implementation and an abstract eval and it runs, and it traces, and it lowers. Call `vmap` on it and the refusal arrives with the primitive's name in it, because nothing about an impl or an abstract eval implies anything about batching.",
              "One line fixes it. `defvectorized` says this primitive is elementwise, which is enough for the interpreter to leave the batch axis alone, and the same call that raised a moment ago now returns a (4, 3). Look at the jaxpr afterward and the equation is still `mystery a`, one primitive, operating on a batched shape.",
              '>> A primitive is batchable when someone has written it a rule, and not before.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the refusal, the one-line fix, and the jaxpr that comes out',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nimport jax._src.interpreters.batching as B\nfrom jax._src import core as jcore\n\nmystery_p = jcore.Primitive("mystery")\nmystery_p.def_impl(lambda v: v * 2.0)\nmystery_p.def_abstract_eval(lambda v: jcore.ShapedArray(v.shape, v.dtype))\n\nprint(mystery_p.bind(jnp.ones(3)))\ntry:\n    jax.vmap(mystery_p.bind)(jnp.ones((4, 3)))\nexcept NotImplementedError as err:\n    print(err)\n\nB.defvectorized(mystery_p)\nprint(jax.vmap(mystery_p.bind)(jnp.ones((4, 3))).shape)\nprint(jax.make_jaxpr(jax.vmap(mystery_p.bind))(jnp.ones((4, 3))))\n\n# [2. 2. 2.]\n# Batching rule for \'mystery\' not implemented\n# (4, 3)\n# { lambda ; a:f32[4,3]. let b:f32[4,3] = mystery a in (b,) }',
            },
          },
        ],
        readings: [
          { label: 'batching.py at jax-v0.4.38', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/interpreters/batching.py', note: 'the whole interpreter in 1119 lines; start at the tracer on 384 and the dispatch on 460' },
          { label: 'Autodidax, part 1', url: 'https://docs.jax.dev/en/latest/autodidax.html', note: 'builds the same BatchTracer from nothing under "Vectorized batching with vmap"' },
          { label: 'JAX primitives, the batching section', url: 'https://docs.jax.dev/en/latest/jax-primitives.html', note: 'a rule written by hand for a custom primitive, registered into the same table' },
        ],
        check: [
          {
            q: 'Inside a vmapped function an argument reports shape (3, 5) while the array under it is (3, 7, 5). What is the tracer holding, and where did the axis go?',
            a: 'It holds the full (3, 7, 5) value plus batch_dim 1. The aval property calls core.mapped_aval to delete that axis, so every shape check inside your function sees one example while the value carries the whole batch.',
          },
          {
            q: 'A primitive you defined yourself runs under jit but raises NotImplementedError under vmap. Why?',
            a: 'Because a batching rule is a separate registration from the impl and the abstract eval, and nothing infers one from the others. Until the primitive is a key in primitive_batchers or fancy_primitive_batchers, process_primitive falls through to the raise; defvectorized is enough if the primitive is elementwise.',
          },
          {
            q: 'Every argument to some primitive inside a vmapped function arrives unmapped. Which branch runs, and what is the result’s batch dim?',
            a: 'The no-op shortcut. The primitive binds on the parent trace with the raw values, no rule is looked up, and the result is not wrapped in a BatchTracer, so its batch dim is not_mapped, which is None.',
          },
        ],
        work: [
          { id: 'walk-the-dispatch', label: 'take three primitives from a jaxpr of your own and say, for each, which of the four branches of process_primitive it takes and why', href: '#the-dispatch-in-twenty-lines' },
          { id: 'register-a-rule', label: 'define one primitive of your own, watch vmap refuse it, register a rule, and watch the same call pass', href: '#a-primitive-with-no-rule-at-all' },
        ],
      },
      {
        id: 'where-the-axis-goes',
        num: 2,
        title: 'Where the axis goes',
        lede: 'A batch axis placed where the primitives already want it costs nothing at all. Placed anywhere else it costs one transpose, and in_axes and out_axes together decide which of those you get.',
        goal: 'Predict both the result shape and the extra equations for any in_axes and out_axes pair, including negatives, None entries and pytree prefixes, and name the boundary where a move is inserted.',
        sections: [
          {
            h: 'the axis you name is the axis the rule receives',
            ps: [
              "Batch an elementwise function along axis 1 and nothing inside the function moves. `sin` and `mul` run on the (3, 4) array exactly as it arrived, because neither of them cares which axis is which, and a single `transpose` appears at the very end to put the batch where `out_axes` asked for it.",
              "Ask for it back where it started and even that goes away. With `in_axes=1` and `out_axes=1` the jaxpr is character for character the jaxpr of the unbatched function, one `sin` and one `mul`, no transpose anywhere. The rewrite added nothing because nothing needed to move.",
              "So a move is not something vmap does to your data on the way in. It is something it does at whichever boundary the axis specs disagree about, and if they agree with each other and with the primitives, there is no boundary work to do.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same function batched three ways, and where the transpose shows up',
              lang: 'haskell',
              text: '>>> f = lambda x: jnp.sin(x) * 2.0\n>>> jax.make_jaxpr(jax.vmap(f, in_axes=0))(jnp.ones((4, 3)))\n{ lambda ; a:f32[4,3]. let b:f32[4,3] = sin a; c:f32[4,3] = mul b 2.0 in (c,) }\n\n>>> jax.make_jaxpr(jax.vmap(f, in_axes=1))(jnp.ones((3, 4)))\n{ lambda ; a:f32[3,4]. let\n    b:f32[3,4] = sin a\n    c:f32[3,4] = mul b 2.0\n    d:f32[4,3] = transpose[permutation=(1, 0)] c\n  in (d,) }\n\n>>> jax.make_jaxpr(jax.vmap(f, in_axes=1, out_axes=1))(jnp.ones((3, 4)))\n{ lambda ; a:f32[3,4]. let b:f32[3,4] = sin a; c:f32[3,4] = mul b 2.0 in (c,) }',
            },
          },
          {
            h: 'matchaxis, and its four outcomes',
            ps: [
              "The function that reconciles where the axis is with where you asked for it is `matchaxis`, and it is a chain of four cases. Same place, nothing happens. Two integers that differ, one `moveaxis`, which is the transpose you saw. Not mapped going to mapped, a broadcast. Mapped going to not mapped, an error, unless the caller passed `sum_match`, in which case the axis is summed away.",
              "That last case is not reachable from `out_axes`. It exists for the transpose of a batched program, which the next lesson is about, and its presence here is the reason a broadcast in the forward direction turns into a sum in the backward one.",
              "The error the fourth case raises is the one you meet when you write `out_axes=None` for something that really did get batched. JAX reformats it before you see it, and what arrives reads `at vmap out_axes, got axis spec None but output was batched on axis 0`.",
            ],
            code: {
              caption: 'verbatim, jax/_src/interpreters/batching.py:1077-1088 at jax 0.4.38, the four cases of matchaxis; the jumble branch and the type check above them are folded away',
              lang: 'python',
              text: '  if src == dst:\n    return x\n  elif type(src) == type(dst) == int:\n    return moveaxis(x, src, dst)\n  elif src is not_mapped and dst is not not_mapped:\n    return broadcast(x, sz, canonicalize_axis(dst, np.ndim(x) + 1))\n  elif dst is not_mapped and sum_match:\n    return x.sum(src)\n  else:\n    if (not isinstance(axis_name, core._TempAxisName) and\n        axis_name is not core.no_axis_name):\n      raise ValueError(f\'vmap has mapped output ({axis_name=}) but out_axes is {dst}\')',
              full: {
                text: 'def matchaxis(axis_name, sz, src, dst, x, sum_match=False):\n  if dst == jumble_axis:\n    x = bdim_at_front(x, src, sz)\n    elt_ty = x.aval.update(shape=x.shape[1:])\n    aval = JumbleTy(core.Var(\'\', core.ShapedArray((), np.dtype(\'int32\'))),\n                    x.shape[0], elt_ty)\n    return Jumble(aval, x)\n  try:\n    _ = core.get_aval(x)\n  except TypeError as e:\n    raise TypeError(f"Output from batched function {x!r} with type "\n                    f"{type(x)} is not a valid JAX type") from e\n  if src == dst:\n    return x\n  elif type(src) == type(dst) == int:\n    return moveaxis(x, src, dst)\n  elif src is not_mapped and dst is not not_mapped:\n    return broadcast(x, sz, canonicalize_axis(dst, np.ndim(x) + 1))\n  elif dst is not_mapped and sum_match:\n    return x.sum(src)\n  else:\n    if (not isinstance(axis_name, core._TempAxisName) and\n        axis_name is not core.no_axis_name):\n      raise ValueError(f\'vmap has mapped output ({axis_name=}) but out_axes is {dst}\')\n    else:\n      raise SpecMatchError(None, None, None)',
                label: 'matchaxis in full, jax/_src/interpreters/batching.py:1065-1091 at jax 0.4.38',
              },
            },
          },
          {
            h: 'dot_general takes the axis into its parameters',
            ps: [
              "A matmul does care which axis is which, and its rule handles that without adding an equation. Batch the left operand on axis 0 and the contraction moves from axis 0 to axis 1 of that operand, written into `dimension_numbers`. Batch both operands and a genuine batch-dimension pair appears in the second half of that parameter, which is what `dot_general` has always had for batched matmuls.",
              "Count the equations across those variants and the count does not move. Four equations for the unbatched program, four for `in_axes=0`, four for `in_axes=1`. The rewrite is inside a parameter, not in the equation list, which is worth knowing before you go looking for the batching in a jaxpr and fail to find it.",
              "Reductions behave the same way one level simpler. `reduce_sum[axes=(0,)]` over a (3, 5) becomes `reduce_sum[axes=(1,)]` over a (4, 3, 5) when the batch goes in front, and `axes=(0,)` again when the batch goes in the middle, because the axis numbers shift around the inserted one.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one dot_general, three batchings, and the parameter that absorbs each one; each equation’s preferred_element_type=float32 line is elided for width, and nothing else is',
              lang: 'haskell',
              text: '>>> dot = lambda a, b: a @ b\n>>> jax.make_jaxpr(dot)(jnp.ones(8), jnp.ones((8, 5)))\n{ lambda ; a:f32[8] b:f32[8,5]. let\n    c:f32[5] = dot_general[dimension_numbers=(([0], [0]), ([], []))] a b\n  in (c,) }\n\n>>> jax.make_jaxpr(jax.vmap(dot, in_axes=(0, None)))(jnp.ones((32, 8)), jnp.ones((8, 5)))\n{ lambda ; a:f32[32,8] b:f32[8,5]. let\n    c:f32[32,5] = dot_general[dimension_numbers=(([1], [0]), ([], []))] a b\n  in (c,) }\n\n>>> jax.make_jaxpr(jax.vmap(dot, in_axes=(0, 0)))(jnp.ones((32, 8)), jnp.ones((32, 8, 5)))\n{ lambda ; a:f32[32,8] b:f32[32,8,5]. let\n    c:f32[32,5] = dot_general[dimension_numbers=(([1], [1]), ([0], [0]))] a b\n  in (c,) }',
            },
          },
          {
            h: 'None does not make N copies',
            ps: [
              "Broadcast a 1000-element weight vector across a batch of 8 and you might expect an (8, 1000) to appear in the jaxpr. What appears is a (1, 1000). The rule gives the unmapped operand a size-1 axis and lets `mul` do the rest, which is the comment from the previous lesson made visible: nothing is materialized per example.",
              "Push it further and the broadcast disappears too. When an output does not depend on any batched input, the no-op shortcut keeps it unmapped end to end, so a two-output function with `out_axes=(None, 0)` produces a jaxpr where the first output was computed once, at its original shape, and never widened.",
              "The widening only happens when you ask for it. Set `out_axes=0` on an unmapped output and a `broadcast_in_dim` shows up at the end to give it the batch axis it never had, which is `matchaxis` taking its third case.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the size-1 broadcast, and an output that stays unmapped; the np.int64(...) inside broadcast_dimensions is numpy 2.2.6 reprs coming through this version’s jaxpr printer, not a different value',
              lang: 'haskell',
              text: '>>> useboth = lambda w, x: jnp.sum(w * x)\n>>> jax.make_jaxpr(jax.vmap(useboth, in_axes=(None, 0)))(jnp.ones(1000), jnp.ones((8, 1000)))\n{ lambda ; a:f32[1000] b:f32[8,1000]. let\n    c:f32[1,1000] = broadcast_in_dim[\n      broadcast_dimensions=(np.int64(1),)\n      shape=(1, 1000)\n      sharding=None\n    ] a\n    d:f32[8,1000] = mul c b\n    e:f32[8] = reduce_sum[axes=(np.int64(1),)] d\n  in (e,) }\n\n>>> k = lambda x, y: (jnp.sin(y), x * 2.0)\n>>> jax.make_jaxpr(jax.vmap(k, in_axes=(0, None), out_axes=(None, 0)))(jnp.ones(4), jnp.ones(3))\n{ lambda ; a:f32[4] b:f32[3]. let\n    c:f32[3] = sin b\n    d:f32[4] = mul a 2.0\n  in (c, d) }',
            },
          },
          {
            h: 'the drill',
            ps: [
              "Eight calls of one function, `mv(A, v) = A @ v`, with the batch put in different places. Predict the result shape and the equation list for each before you read the right-hand columns, and check the third and fourth rows against each other: a negative `in_axes` counts from the right of that argument, so -2 on a rank-3 array is axis 1 and behaves identically.",
              "The pair worth staring at is rows five and seven. Same call, same inputs, and moving `out_axes` from its default 0 to 1 removes an equation, because batching `v` on axis 0 naturally puts the batch at position 1 of the result and asking for it at position 0 is what forces the transpose. Reaching for the default out here costs a transpose that the code did not need.",
              "Two rules on the table's edges are worth carrying. Axis integers must be in `[-ndim, ndim)` for the array they apply to, and keyword arguments are always mapped along axis 0 with no way to say otherwise, both of which the `vmap` docstring states outright. When arguments are pytrees, `in_axes` has to be a tree prefix of the argument tuple, so a spec may stop short and cover a whole subtree with one entry, but it may never disagree with the structure below it.",
            ],
            table: {
              caption: 'eight calls of mv(A, v) = A @ v, run on jax 0.4.38 CPU; equations are the jaxpr’s primitives in order',
              cols: ['call', 'argument shapes', 'result', 'equations'],
              rows: [
                ['mv(A, v), no vmap', 'A (2, 3), v (3,)', '(2,)', 'dot_general'],
                ['vmap(mv, in_axes=(0, None))', 'A (5, 2, 3), v (3,)', '(5, 2)', 'dot_general'],
                ['vmap(mv, in_axes=(1, None))', 'A (2, 5, 3), v (3,)', '(5, 2)', 'dot_general, transpose'],
                ['vmap(mv, in_axes=(-2, None))', 'A (2, 5, 3), v (3,)', '(5, 2)', 'dot_general, transpose'],
                ['vmap(mv, in_axes=(None, 0))', 'A (2, 3), v (5, 3)', '(5, 2)', 'dot_general, transpose'],
                ['vmap(mv, in_axes=(0, 0))', 'A (5, 2, 3), v (5, 3)', '(5, 2)', 'dot_general'],
                ['vmap(mv, in_axes=(None, 0), out_axes=1)', 'A (2, 3), v (5, 3)', '(2, 5)', 'dot_general'],
                ['vmap(mv, in_axes=(None, 0), out_axes=-1)', 'A (2, 3), v (5, 3)', '(2, 5)', 'dot_general'],
              ],
            },
          },
          {
            h: 'nesting stacks axes and nests nothing',
            ps: [
              "Three nested vmaps over a product of three vectors give a (2, 3, 4), the axes stacking outside in, and the jaxpr holds no nested structure of any kind. Seven equations come out, five `broadcast_in_dim`s inserting size-1 axes and two `mul`s. What a reader would call an outer product is what the nesting compiled to, because each layer only ever rewrote the layer below it.",
              "Two nested maps over a matmul collapse further still, into one `dot_general` with `dimension_numbers=(([1], [1]), ([], []))` and no batch dimensions at all. The pairwise structure lives entirely in which axes get contracted.",
              "Swapping the nesting order does not change the result's shape here, only which axis is which. `vmap(vmap(mv, in_axes=(None, 0)), in_axes=(0, None))` and the same pair reversed both return a (5, 5, 2), and one is the other with axes 0 and 1 exchanged. Shape is a weak check on a nest; if you want to know you got the order right, transpose one against the other and compare.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): three levels deep, and an order swap that keeps the shape',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ng = lambda a, b, c: a * b * c\nthree = jax.vmap(jax.vmap(jax.vmap(g, in_axes=(None, None, 0)),\n                          in_axes=(None, 0, None)), in_axes=(0, None, None))\nprint(three(jnp.ones(2), jnp.ones(3), jnp.ones(4)).shape)\n\nmv = lambda A, v: A @ v\nA, vs = jnp.ones((5, 2, 3)), jnp.ones((5, 3))\np = jax.vmap(jax.vmap(mv, in_axes=(None, 0)), in_axes=(0, None))\nq = jax.vmap(jax.vmap(mv, in_axes=(0, None)), in_axes=(None, 0))\nprint(p(A, vs).shape, q(A, vs).shape)\nprint(bool(jnp.all(p(A, vs) == jnp.swapaxes(q(A, vs), 0, 1))))\n\n# (2, 3, 4)\n# (5, 5, 2) (5, 5, 2)\n# True',
            },
          },
        ],
        readings: [
          { label: 'jax.vmap reference', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html', note: 'the in_axes and out_axes paragraphs, including the range rule and the keyword-argument rule' },
          { label: 'Applying optional parameters to pytrees', url: 'https://docs.jax.dev/en/latest/pytrees.html', note: 'what a tree prefix is, which is the whole of the pytree in_axes rule' },
          { label: 'matchaxis and moveaxis at jax-v0.4.38', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/interpreters/batching.py', note: 'the reconciliation functions at the bottom of the file, from 1065' },
        ],
        check: [
          {
            q: 'A vmapped elementwise function traces to exactly the jaxpr of the unbatched function, with no transpose anywhere. What must in_axes and out_axes have been?',
            a: 'Equal to each other, and the primitives inside must be ones that ignore axis position. With in_axes=1 and out_axes=1 on sin then mul, matchaxis takes its src == dst case and returns the value untouched, so nothing is inserted at either boundary.',
          },
          {
            q: 'Why does vmap(mv, in_axes=(None, 0)) add a transpose while vmap(mv, in_axes=(None, 0), out_axes=1) does not?',
            a: 'Because batching v on axis 0 makes dot_general produce the batch at position 1 of the result. out_axes=0 asks for it at position 0, which is a moveaxis, and out_axes=1 asks for it where it already is, which is nothing.',
          },
          {
            q: 'You map an argument with in_axes=None and its array has 1000 elements against a batch of 8. What shape does it take in the jaxpr, and why not (8, 1000)?',
            a: 'It becomes (1, 1000). The batching rule inserts a size-1 axis and lets the primitive’s own broadcasting cover the batch, so no per-example copy is materialized; a full-width broadcast only appears when out_axes asks an unmapped value to gain a real batch axis.',
          },
        ],
        work: [
          { id: 'predict-the-equations', label: 'cover the last two columns of the drill table, predict shape and equation list for all eight rows, then check with make_jaxpr', href: '#the-drill' },
          { id: 'find-a-free-out-axes', label: 'take one vmap in code of your own and find whether a different out_axes removes a transpose from its jaxpr', href: '#dot-general-takes-the-axis-into-its-parameters' },
        ],
      },
      {
        id: 'composed-with-grad-and-jit',
        num: 3,
        title: 'Composed with grad and jit',
        lede: 'Every ordering of vmap, grad and jit compiles. They do not all compute the same thing, and the jaxpr says which one you asked for before you have to reason about it.',
        goal: 'Read an ordering of vmap, grad and jit off a jaxpr, say whether it yields per-example gradients or a summed one, explain what vmap does when it meets a pjit equation, and name the refusals that belong to vmap alone.',
        sections: [
          {
            h: 'the ordering decides what you get',
            ps: [
              "Put `vmap` outside `grad` and the whole program grows a batch axis, gradient equations included, and what comes back is a (4, 3) with one row per example. Put `grad` outside `vmap` and the batch axis is still there through the middle of the program, but the result is a (3,), one gradient for the batch.",
              "The difference lands in three equations at the end of the second jaxpr, and they are worth finding by eye: `reduce_sum`, then `reshape`, then `reduce_sum` again. That tail is the transpose of the broadcast that the batching rule inserted for the unmapped `w` on the way in, collapsing the batch axis back out of the cotangent.",
              "Which means the two orderings are related by exactly that sum, and you can check it in one line: the rows of the per-example result add up to the batch gradient, to the last bit on this machine. If you only ever need the sum, either ordering works; if you need the rows, only one does.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the two orderings, their jaxpr tails, and the identity between them; the equations before each tail are elided at the marked line, and the full dumps run eleven and fifteen equations',
              lang: 'haskell',
              text: '>>> loss = lambda w, x: jnp.sum(jnp.tanh(x * w))\n>>> jax.make_jaxpr(jax.vmap(jax.grad(loss), in_axes=(None, 0)))(jnp.ones(3), jnp.ones((4, 3)))\n{ lambda ; a:f32[3] b:f32[4,3]. let\n    ...\n    i:f32[4,3] = mul h f\n    j:f32[4,3] = mul i e\n    k:f32[4,3] = add_any i j\n    l:f32[4,3] = mul b k\n  in (l,) }\n\n>>> batched = lambda w, xs: jnp.sum(jax.vmap(lambda x: jnp.sum(jnp.tanh(x * w)))(xs))\n>>> jax.make_jaxpr(jax.grad(batched))(jnp.ones(3), jnp.ones((4, 3)))\n{ lambda ; a:f32[3] b:f32[4,3]. let\n    ...\n    m:f32[4,3] = mul b l\n    n:f32[3] = reduce_sum[axes=(0,)] m\n    o:f32[1,3] = reshape[dimensions=None new_sizes=(1, 3) sharding=None] n\n    p:f32[3] = reduce_sum[axes=(np.int64(0),)] o\n  in (p,) }\n\n>>> xs = jnp.arange(12.0).reshape(4, 3)\n>>> per = jax.vmap(jax.grad(loss), in_axes=(None, 0))(jnp.ones(3), xs)\n>>> per.shape, per.sum(0)\n((4, 3), Array([0.02974708, 0.4253622 , 0.14220996], dtype=float32))\n>>> jax.grad(batched)(jnp.ones(3), xs)\nArray([0.02974708, 0.4253622 , 0.14220996], dtype=float32)',
            },
          },
          {
            h: 'the broadcast comes back as a sum',
            ps: [
              "That tail is not a special case anyone wrote for gradients of batched functions. It is the fourth branch of `matchaxis` from the last lesson, the one guarded by `sum_match`, doing what the transpose of a broadcast has to do.",
              "Read it as a pair and it stops needing memorising. An unmapped input gets broadcast on the way forward, so its cotangent gets summed on the way back, because the derivative of copying a value into N places is adding the N gradients up. The batch axis you introduced with `in_axes=None` is exactly the axis that disappears in the reverse pass.",
              "This also tells you where per-example gradients stop being free of assumptions. They exist because `vmap` sat outside `grad` and the sum never happened, not because JAX kept N separate tapes. There is one program, one backward pass, and an axis that was never collapsed.",
            ],
          },
          {
            h: 'vmap walks into a pjit, not around it',
            ps: [
              "`pjit` is one of the twenty-one primitives with a rule in the second table, and its rule batches the jaxpr held in the equation's parameters rather than calling the equation once per example. So an inner `jit` does not hide its contents from the batching interpreter, and it does not force a per-example dispatch either.",
              "The consequence is easier to state as a measurement. `vmap(jit(f))` and `jit(vmap(f))` trace to the same jaxpr, text for text, on this machine. Both hold one `pjit` equation whose inner jaxpr is already batched, and the string comparison of the two dumps is `True`.",
              "That is a useful thing to know when you inherit code with `jit` decorators scattered through it. Wrapping the outside in `vmap` does not defeat them and is not defeated by them; the batch axis goes through the call boundary and comes out the other side.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the two orderings compared as text, and the jaxpr they share',
              lang: 'haskell',
              text: '>>> f = lambda x: jnp.sin(x) * 2.0\n>>> x = jnp.ones((4, 3))\n>>> str(jax.make_jaxpr(jax.vmap(jax.jit(f)))(x)) == str(jax.make_jaxpr(jax.jit(jax.vmap(f)))(x))\nTrue\n>>> jax.make_jaxpr(jax.vmap(jax.jit(f)))(x)\n{ lambda ; a:f32[4,3]. let\n    b:f32[4,3] = pjit[\n      name=<lambda>\n      jaxpr={ lambda ; c:f32[4,3]. let\n          d:f32[4,3] = sin c\n          e:f32[4,3] = mul d 2.0\n        in (e,) }\n    ] a\n  in (b,) }',
            },
          },
          {
            h: 'the four refusals vmap owns by itself',
            ps: [
              "Four errors come from the vmap machinery and nowhere else, and all four are about the axis specs rather than about your function. Every one of them fires before any tracing happens, which is why they are quick to fix: the message already contains the two things that disagree.",
              "The first has an answer the error does not mention. Mapping nothing is legal if you say how wide the batch is, and `axis_size=4` with `in_axes=None` runs the body once at its original shape and broadcasts the result, which is a cheap way to shape a program for a batch it does not read yet.",
              "Two other failures look like vmap's and are not. An `in_axes` that names a batch of 32 against an argument whose axis is 8 raises the inconsistent-sizes error, which the museum keeps as its own exhibit. And a function whose output shape depends on its values fails under `vmap` for the same reason it fails under `jit`, since that is a rule about tracing, not about batching, and the message you get is the tracing one.",
            ],
            table: {
              caption: 'the four vmap-only refusals, messages quoted verbatim from runs on jax 0.4.38 CPU',
              cols: ['what you wrote', 'the message', 'the fix'],
              rows: [
                ['in_axes=None on every argument', 'vmap must have at least one non-None value in in_axes', 'pass axis_size=n, or map something'],
                ['out_axes=None on a batched output', 'at vmap out_axes, got axis spec None but output was batched on axis 0', 'give it an integer, or stop batching what feeds it'],
                ['in_axes=2 on a rank-2 array', 'vmap was requested to map its argument along axis 2, which implies that its rank should be at least 3, but is only 2 (its shape is (4, 3))', 'axis integers live in [-ndim, ndim)'],
                ['in_axes={"w": None} against a two-key dict', 'vmap in_axes specification must be a tree prefix of the corresponding value, got specification ({\'w\': None}, 0) for value tree PyTreeDef(({\'b\': *, \'w\': *}, *))', 'name every key, or replace the dict spec with one None'],
              ],
            },
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the all-None refusal and the axis_size that answers it',
              lang: 'haskell',
              text: '>>> jax.vmap(lambda x: x * 2.0, in_axes=None)(jnp.ones(3))\nValueError: vmap must have at least one non-None value in in_axes\n\n>>> jax.vmap(lambda x: x * 2.0, in_axes=None, axis_size=4)(jnp.ones(3)).shape\n(4, 3)\n>>> jax.make_jaxpr(jax.vmap(lambda x: x * 2.0, in_axes=None, axis_size=4))(jnp.ones(3))\n{ lambda ; a:f32[3]. let\n    b:f32[3] = mul a 2.0\n    c:f32[4,3] = broadcast_in_dim[\n      broadcast_dimensions=(np.int64(1),)\n      shape=(4, 3)\n      sharding=None\n    ] b\n  in (c,) }',
            },
          },
          {
            h: 'when the primitive is yours',
            ps: [
              "Everything above rests on the primitives already having rules, and the moment you add a primitive of your own that stops being true. A Pallas kernel is batchable because `pallas_call` is a key in the first rule table, a `custom_vjp` because its call primitive is a key in the second. Neither is batchable because vmap can see inside it.",
              "So the question to ask of any new primitive is the one lesson one ended on. Does it have a rule, is that rule the elementwise one, and does the elementwise one actually describe what the primitive does to an extra axis. Get that wrong and nothing raises; you get a program that runs and computes the wrong thing.",
              "The kernel path's Pallas arc is where that stops being hypothetical, since a kernel with a block spec has real opinions about which axis is which. The rule table is the same table.",
            ],
          },
        ],
        readings: [
          { label: 'Automatic vectorization', url: 'https://docs.jax.dev/en/latest/automatic-vectorization.html', note: 'the official tutorial; read it for the axis specs, then come back for the orderings' },
          { label: 'Autodidax, part 3', url: 'https://docs.jax.dev/en/latest/autodidax.html', note: 'builds the transpose rule that turns a forward broadcast into a backward sum' },
          { label: 'Understanding jaxprs', url: 'https://docs.jax.dev/en/latest/jaxpr.html', note: 'the grammar the comparisons in this lesson are read in, pjit equations included' },
        ],
        check: [
          {
            q: 'One jaxpr ends in reduce_sum, reshape, reduce_sum and returns a (3,); another ends in a mul and returns a (4, 3). Which is grad of vmap and which is vmap of grad?',
            a: 'The (3,) one is grad of vmap: that tail is the transpose of the broadcast the batching rule inserted for the unmapped argument, collapsing the batch axis out of the cotangent. The (4, 3) one is vmap of grad, where the axis is never collapsed and each row is one example’s gradient.',
          },
          {
            q: 'You wrap vmap around a function that already has a jit inside it. What does the batching interpreter do with the pjit equation?',
            a: 'It looks pjit up in the fancy rule table and batches the jaxpr stored in the equation’s parameters. The result is the same jaxpr you would get from jit(vmap(f)), text for text: one pjit equation whose inner jaxpr already carries the batch axis.',
          },
          {
            q: 'Which vmap error does axis_size answer, and what does the resulting jaxpr look like?',
            a: 'The one raised when every in_axes entry is None. With axis_size=n the body traces once at its unbatched shape and a single broadcast_in_dim at the end gives the result its batch axis, so no work is repeated per example.',
          },
        ],
        work: [
          { id: 'both-orderings', label: 'take one loss of your own and dump both orderings, then find the equations that differ and say which broadcast each sum is undoing', href: '#the-ordering-decides-what-you-get' },
          { id: 'read-a-pjit', label: 'wrap vmap around a function with an inner jit and read the batched shapes inside the pjit equation before you look at the outer ones', href: '#vmap-walks-into-a-pjit-not-around-it' },
        ],
      },
    ],
  },
]
