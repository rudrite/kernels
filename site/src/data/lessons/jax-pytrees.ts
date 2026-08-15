// The pytree arc under jax chapter 7: where flattening happens, the two
// checks every transformation runs against a tree, and what registering a
// node commits you to. Every snippet ran on this machine (jax 0.4.38, CPU,
// Python 3.12, 2026-08-15) and is published with the stdout it produced,
// appended as comments. Error text is quoted from those runs, never
// paraphrased; the two docs sentences quoted in lesson 3 come from the
// installed 0.4.38 docstrings and the current API reference, which agree.
import type { UnitLessons } from './index'

export const JAX_PYTREE_LESSONS: UnitLessons[] = [
  {
    "unit": "jax:pytrees",
    "lessons": [
      {
        "id": "flatten-happens-first",
        "num": 1,
        "title": "Flatten happens first",
        "lede": "Hand a jitted function a dict of dicts and the program it compiles takes four separate arrays. The nesting is a Python-side convention that stops at the trace boundary, and knowing where it stops explains most of what the transformations do with a model.",
        "goal": "Predict the leaf order any container flattens to, say what a treedef stores and what it drops, and name a leaf by the same key path JAX uses when it reports an error about that leaf.",
        "sections": [
          {
            "h": "four arguments where you passed two",
            "ps": [
              "The forward pass below takes two arguments: a params dict nested two levels deep, and a batch of rows. Trace it and the header line of the jaxpr lists four invars, one per array. Neither the dict nor the word `dense` is anywhere in the recording.",
              "The two printed lines say the same thing at two altitudes. Chapter 2's recording and chapter 3's lowered StableHLO agree that this callable takes four flat tensors in one fixed order. Flattening runs on the way in, before tracing starts, and the treedef stays behind on the Python side to rebuild whatever comes back.",
              "Read the order off the header and it is not the order anyone typed. `a:f32[2]` is `dense.b`, `b:f32[3,2]` is `dense.w`, `c:f32[2,1]` is `out.w`, and `d:f32[4,3]` is the batch. Inside each dict the keys sorted themselves; across the two arguments, positional order held.",
              ">> The compiler never sees a dict. It sees leaves, in an order you can work out beforehand."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): the jaxpr header line and the StableHLO entry point, both flat",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\n\nparams = {\"dense\": {\"w\": jnp.ones((3, 2)), \"b\": jnp.zeros(2)},\n          \"out\": {\"w\": jnp.ones((2, 1))}}\n\ndef fwd(p, x):\n    h = jnp.tanh(x @ p[\"dense\"][\"w\"] + p[\"dense\"][\"b\"])\n    return h @ p[\"out\"][\"w\"]\n\nx = jnp.ones((4, 3))\nprint(str(jax.make_jaxpr(fwd)(params, x)).split(\"\\n\")[0])\nprint(next(l for l in jax.jit(fwd).lower(params, x).as_text().split(\"\\n\") if \"@main\" in l).strip())\n\n# { lambda ; a:f32[2] b:f32[3,2] c:f32[2,1] d:f32[4,3]. let\n# func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<2x1xf32>, %arg3: tensor<4x3xf32>) -> (tensor<4x1xf32> {jax.result_info = \"\"}) {"
            }
          },
          {
            "h": "the order comes from the registry",
            "ps": [
              "The jaxpr above put `dense.b` ahead of `dense.w` though `w` was typed first, and nothing in the function chose that. Each container type carries its own flatten function in the registry, and the dict's walks the keys in sorted order, so which array arrives as argument one is settled by the keys before tracing starts. The jit lessons under chapter 3 take the same rule the other way, into what it means for a cache entry; here it decides only the order.",
              "`OrderedDict` does not inherit that behaviour. It is registered as a node whose auxiliary data is its key tuple, so insertion order is part of the structure and the leaves come out in the order they went in. The first two lines below differ only in the container, and the leaf shapes come out swapped.",
              "A namedtuple is a pytree node without anyone registering it, and its treedef never compares equal to the plain tuple holding the same two fields. The printed forms show why: one is `PyTreeDef(CustomNode(namedtuple[P], [*, *]))` and the other is `PyTreeDef((*, *))`, so the class name is carried in the structure rather than looked past.",
              "Keys that cannot be compared to each other stop the sort before any of this happens. A dict mixing an int key with a str key raises while flattening, and the message names the sort rather than the transformation you were calling."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): four containers holding the same two arrays, and the order each one imposes",
              "lang": "python",
              "text": "import collections\n\nimport jax\nimport jax.numpy as jnp\n\nw, b = jnp.ones((3, 2)), jnp.zeros(2)\nP = collections.namedtuple(\"P\", [\"w\", \"b\"])\n\nprint([l.shape for l in jax.tree.leaves({\"w\": w, \"b\": b})])\nprint([l.shape for l in jax.tree.leaves(collections.OrderedDict([(\"w\", w), (\"b\", b)]))])\nprint(jax.tree.structure(P(w, b)) == jax.tree.structure((w, b)))\ntry:\n    jax.tree.leaves({1: w, \"a\": b})\nexcept ValueError as e:\n    print(e)\n\n# [(2,), (3, 2)]\n# [(3, 2), (2,)]\n# False\n# Comparator raised exception while sorting pytree dictionary keys."
            },
            "table": {
              "caption": "leaf order and treedef for one pair of arrays, w of shape (3, 2) and b of shape (2,), as printed on this machine (verified, jax 0.4.38 CPU)",
              "cols": [
                "container",
                "leaf shapes, in order",
                "treedef as printed"
              ],
              "rows": [
                [
                  "{'w': w, 'b': b}",
                  "[(2,), (3, 2)]",
                  "PyTreeDef({'b': *, 'w': *})"
                ],
                [
                  "OrderedDict([('w', w), ('b', b)])",
                  "[(3, 2), (2,)]",
                  "PyTreeDef(CustomNode(OrderedDict[('w', 'b')], [*, *]))"
                ],
                [
                  "(w, b)",
                  "[(3, 2), (2,)]",
                  "PyTreeDef((*, *))"
                ],
                [
                  "P(w, b), a namedtuple",
                  "[(3, 2), (2,)]",
                  "PyTreeDef(CustomNode(namedtuple[P], [*, *]))"
                ],
                [
                  "{'w': w, 'cfg': None}",
                  "[(3, 2)]",
                  "PyTreeDef({'cfg': None, 'w': *})"
                ]
              ]
            }
          },
          {
            "h": "what a treedef carries",
            "ps": [
              "A treedef has a printable form, and reading it beats reasoning about it. `PyTreeDef({'cfg': None, 'opt': [*, *], 'w': *})` says there are three leaves, sitting at `opt[0]`, `opt[1]` and `w`, and that `cfg` is present as an empty subtree contributing no leaf. Chapter 7 told you `None` drops out of the leaves list. The printed form adds that it did not drop out of the structure.",
              "The two counts a treedef reports answer different questions. `num_leaves` is 3 for that tree and `num_nodes` is 6, because the nodes tally counts the containers and the `None` alongside the leaves.",
              "Equality looks at the containers and stops there. Two dicts with the same keys give equal treedefs even when one holds arrays and the other holds Python ints, because the leaves left the structure the moment they were flattened out of it and nothing about them is recorded in what remains.",
              "Unflattening is arity-checked, so a treedef will not silently accept a short list of leaves. Hand `jax.tree.unflatten` two leaves for a three-leaf structure and it counts them back at you."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): one treedef, printed, counted, compared, and refused",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\n\nt = {\"w\": jnp.ones((2, 2)), \"cfg\": None, \"opt\": [jnp.zeros(1), 3]}\nleaves, treedef = jax.tree.flatten(t)\nprint(treedef)\nprint(treedef.num_leaves, treedef.num_nodes)\nprint(jax.tree.structure({\"a\": 1, \"b\": 2}) == jax.tree.structure({\"a\": jnp.ones(3), \"b\": jnp.zeros((2, 2))}))\ntry:\n    jax.tree.unflatten(treedef, leaves[:2])\nexcept ValueError as e:\n    print(e)\n\n# PyTreeDef({'cfg': None, 'opt': [*, *], 'w': *})\n# 3 6\n# True\n# Too few leaves for PyTreeDef; expected 3, got 2"
            }
          },
          {
            "h": "the path that names a leaf",
            "ps": [
              "`jax.tree_util.tree_flatten_with_path` hands back each leaf together with the route taken to reach it, and `keystr` prints that route as `['dense']['w']`. The same notation shows up in error messages, which is the reason to learn to read it.",
              "Put a string in a params dict and jit refuses, and the refusal locates the offender by path rather than by leaf number. `at path p['dense']['act']` finds one bad entry in a tree of hundreds with no printing on your side.",
              "The path survives further down than you might expect. A jitted function returning a nested dict lowers to a StableHLO entry point whose two results carry `jax.result_info = \"['aux']['norm']\"` and `jax.result_info = \"['loss']\"`, recording which slot of the output tree each result belongs to. The tree is gone from the computation and the labels for rebuilding it are not."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): key paths, the same path inside a jit error, and the paths written into the lowered module",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\n\nparams = {\"dense\": {\"w\": jnp.ones((3, 2)), \"act\": \"relu\"}}\nfor path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:\n    print(jax.tree_util.keystr(path), type(leaf).__name__)\ntry:\n    jax.jit(lambda p: p[\"dense\"][\"w\"] * 2)(params)\nexcept TypeError as e:\n    print(str(e).split(\"abstract array. \", 1)[1].split(\"\\n\")[0])\n\ndef split(x):\n    return {\"loss\": x.sum(), \"aux\": {\"norm\": (x * x).sum()}}\n\ntext = jax.jit(split).lower(jnp.ones(4)).as_text()\nprint(next(l for l in text.split(\"\\n\") if \"@main\" in l).strip())\n\n# ['dense']['act'] str\n# ['dense']['w'] ArrayImpl\n# The problematic value is of type <class 'str'> and was passed to the function at path p['dense']['act'].\n# func.func public @main(%arg0: tensor<4xf32>) -> (tensor<f32> {jax.result_info = \"['aux']['norm']\"}, tensor<f32> {jax.result_info = \"['loss']\"}) {"
            }
          }
        ],
        "readings": [
          {
            "label": "Pytrees (reference)",
            "url": "https://docs.jax.dev/en/latest/pytrees.html",
            "note": "the flattening rules the registry implements, container type by container type"
          },
          {
            "label": "jax.tree_util.tree_flatten_with_path",
            "url": "https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.tree_flatten_with_path.html",
            "note": "the key-path API this lesson reads error messages with"
          },
          {
            "label": "jax.jit lowering",
            "url": "https://docs.jax.dev/en/latest/aot.html",
            "note": "how to get the lowered text the last section reads result_info out of"
          }
        ],
        "check": [
          {
            "q": "You pass params = {'out': w2, 'dense': w1} to a jitted function. Which of the two arrays is the compiled program's first argument, and why?",
            "a": "w1, the one under dense. The dict's flatten function walks its keys in sorted order, so the invars come out alphabetical by key rather than in the order the literal was typed, and you can read that off a jaxpr header before running anything."
          },
          {
            "q": "A treedef prints as PyTreeDef({'cfg': None, 'opt': [*, *], 'w': *}). How many leaves does it have, and what is cfg doing there?",
            "a": "Three leaves, at opt[0], opt[1] and w. The cfg entry is an empty subtree: None contributes no leaf but still occupies a slot in the structure, so a tree with it and a tree without it are different treedefs."
          },
          {
            "q": "jit reports a problem at path p['dense']['act']. What did you build, and where do you look?",
            "a": "A params tree with a non-array leaf, a string in this case, at the act key of the dense subtree. The path is the flatten route to that one leaf, so you go straight to it instead of printing shapes across the whole tree."
          }
        ],
        "work": [
          {
            "id": "predict-order",
            "label": "take one real params tree of your own, write the leaf order down before running anything, then check it against jax.tree.leaves",
            "href": "#the-order-comes-from-the-registry"
          },
          {
            "id": "read-result-info",
            "label": "lower one jitted function that returns a nested dict and read every jax.result_info label in the StableHLO against the output tree you wrote",
            "href": "#the-path-that-names-a-leaf"
          }
        ]
      },
      {
        "id": "structure-then-leaves",
        "num": 2,
        "title": "Structure, then leaves",
        "lede": "Every transformation checks a tree twice, first the containers and then what is inside them, and it has a different error message for each. Knowing which message belongs to which pass turns most failures into a one-line diagnosis.",
        "goal": "Given a structure complaint from tree.map, scan, cond or a vjp pullback, say whether the treedefs disagreed or the leaves did, name which argument was acting as the template, and fix the call without guessing.",
        "sections": [
          {
            "h": "two failures, two messages",
            "ps": [
              "Run a scan whose body returns a carry with one extra key and the message says pytree structure. Run one whose carry starts as int32 and comes back float32 and the message says equal types. Same primitive, same call shape, two entirely different sentences.",
              "The first check compares treedefs and nothing else. The second compares the avals of the leaves, shape and dtype together, and it only runs once the structures already line up. So a structure error means you assembled the wrong containers, and a type error means the containers were right and something inside one of them changed.",
              "Both messages name the component that differs, and the structure one goes as far as printing the symmetric difference of the key sets, {'n'}, so the diffing is done for you. Chapter 6 covers what scan is for. This is what it says when the carry contract breaks, and the wording alone tells you which half broke.",
              ">> Structure first, then leaves. The message says which pass raised it.",
              "Not every tree complaint is one of these two. `vmap` can refuse an `in_axes` spec before either check runs, on the shape of the spec rather than on anything in your arguments, and the vmap lessons under chapter 5 keep that refusal with the others the transformation owns."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): the same scan, failed twice, once on structure and once on dtype",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\n\ndef grows(carry, x):\n    return {\"m\": carry[\"m\"] + x, \"n\": carry[\"m\"]}, x\n\ndef drifts(carry, x):\n    return carry + x, x\n\nfor body, init in [(grows, {\"m\": 0.0}), (drifts, jnp.zeros((), jnp.int32))]:\n    try:\n        jax.lax.scan(body, init, jnp.arange(3.0))\n    except TypeError as e:\n        print(e)\n        print(\"=\" * 8)\n\n# scan body function carry input and carry output must have the same pytree structure, but they differ:\n#\n# The input carry carry is a <class 'dict'> with 1 child but the corresponding component of the carry output is a <class 'dict'> with 2 children, so the numbers of children do not match, with the symmetric difference of key sets: {'n'}.\n#\n# Revise the function so that the carry output has the same pytree structure as the carry input.\n# ========\n# scan body function carry input and carry output must have equal types (e.g. shapes and dtypes of arrays), but they differ:\n#\n# The input carry carry has type int32[] but the corresponding output carry component has type float32[], so the dtypes do not match.\n#\n# Revise the function so that all output types (e.g. shapes and dtypes) match the corresponding input types.\n# ========"
            },
            "table": {
              "caption": "four more of the same two checks, each message quoted from a run on this machine (verified, jax 0.4.38 CPU)",
              "cols": [
                "the call",
                "the message, verbatim",
                "which pass"
              ],
              "rows": [
                [
                  "jax.tree.map(f, [1.0, 2.0], (1.0, 2.0))",
                  "Expected list, got (1.0, 2.0).",
                  "structure"
                ],
                [
                  "jax.tree.map(f, {'a': .., 'b': ..}, {'a': .., 'c': ..})",
                  "Dict key mismatch; expected keys: ['a', 'b']; dict: {'a': 1.0, 'c': 2.0}.",
                  "structure"
                ],
                [
                  "jax.lax.cond(True, lambda: {'a': ..}, lambda: {'b': ..})",
                  "true_fun and false_fun output must have same type structure, got PyTreeDef({'a': *}) and PyTreeDef({'b': *}).",
                  "structure"
                ],
                [
                  "jax.grad(f)({'w': f32 array, 'step': int32 scalar})",
                  "grad requires real- or complex-valued inputs (input dtype that is a sub-dtype of np.inexact), but got int32.",
                  "leaves"
                ]
              ]
            }
          },
          {
            "h": "map reads the first tree and trusts the rest",
            "ps": [
              "`jax.tree.map` is not symmetric in its arguments. The first tree is the template, and every later tree is flattened only as far down as that template goes. A deeper structure in argument two is therefore not an error at all: the subtree arrives at your function whole, as one value.",
              "The first printed line shows it happening. The template has a leaf at `a`, the second tree has a dict there, and the function receives that dict. Nothing checked below the template's depth, because nothing was asked to.",
              "This is why a missing gradient does not raise a structure error. Write `None` where a layer has no gradient and the template still has a leaf in that position, so `None` is handed straight to your function and the failure surfaces as arithmetic on a NoneType, several frames away from the tree that caused it.",
              "`is_leaf` takes the decision back. Pass `is_leaf=lambda x: x is None` and `None` stops being an empty subtree for that one call, so your function receives it and decides what a missing gradient means. The structure that comes back out is the template's, unchanged."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): the template rule, the None that slips through it, and the is_leaf that catches it",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\n\nparams = {\"w\": jnp.ones(3), \"b\": jnp.zeros(3)}\ngrads = {\"w\": jnp.ones(3), \"b\": None}          # a layer with no gradient\n\nprint(jax.tree.map(lambda a, b: (a, type(b).__name__), {\"a\": 1.0}, {\"a\": {\"deeper\": 2.0}}))\ntry:\n    jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)\nexcept TypeError as e:\n    print(e)\nout = jax.tree.map(lambda p, g: p if g is None else p - 0.1 * g, params, grads,\n                   is_leaf=lambda x: x is None)\nprint(jax.tree.structure(out))\n\n# {'a': (1.0, 'dict')}\n# unsupported operand type(s) for *: 'float' and 'NoneType'\n# PyTreeDef({'b': *, 'w': *})"
            }
          },
          {
            "h": "the pullback wants the output tree",
            "ps": [
              "Chapter 4 sets up `vjp` as the reverse direction and chapter 7 states half of its tree contract, that the gradient comes back carrying the input's treedef. The other half sits on the argument you pass to the pullback, and it has to carry the output's treedef instead.",
              "Return a dict of two things from the function and the pullback takes a dict of two cotangents. Drop one key and the message prints both treedefs side by side, so the diff is done for you.",
              "That symmetry matters as soon as a model returns more than a loss. Whatever shape the forward pass hands back is the shape the backward pass expects to be seeded with, key for key, and a tuple will not stand in for a dict holding the same two entries."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): the pullback takes the output's tree, and says so when it does not get it",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\n\ndef f(p):\n    return {\"y\": p[\"w\"] * 2, \"z\": p[\"w\"].sum()}\n\nout, pullback = jax.vjp(f, {\"w\": jnp.ones(3)})\nprint(jax.tree.structure(out))\nprint(pullback({\"y\": jnp.ones(3), \"z\": jnp.ones(())})[0])\ntry:\n    pullback({\"y\": jnp.ones(3)})\nexcept ValueError as e:\n    print(e)\n\n# PyTreeDef({'y': *, 'z': *})\n# {'w': Array([3., 3., 3.], dtype=float32)}\n# unexpected tree structure of argument to vjp function: got PyTreeDef({'y': *}), but expected to match PyTreeDef({'y': *, 'z': *})"
            }
          }
        ],
        "readings": [
          {
            "label": "JAX errors",
            "url": "https://docs.jax.dev/en/latest/errors.html",
            "note": "the official catalogue of messages, worth skimming once so the wording is familiar before you need it"
          },
          {
            "label": "jax.tree.map",
            "url": "https://docs.jax.dev/en/latest/_autosummary/jax.tree.map.html",
            "note": "the signature the template rule is written into, plus what is_leaf is allowed to stop"
          },
          {
            "label": "jax.vjp",
            "url": "https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html",
            "note": "the pullback's contract, both directions of it"
          }
        ],
        "check": [
          {
            "q": "A scan tells you the carry input and output must have equal types, not that they must have the same pytree structure. What does that wording rule out?",
            "a": "It rules out a container mismatch. The structure check runs first and passed, so the keys and containers line up; what differs is a leaf, meaning a shape or a dtype, and the message names which one and how."
          },
          {
            "q": "You call tree.map(lambda p, g: p - 0.1 * g, params, grads) and get a TypeError about NoneType, not a structure error. What happened?",
            "a": "grads had None where params had an array. tree.map only flattens later trees up to the first one's structure, so the None was passed through as a value rather than compared as a structure, and it blew up inside your own function."
          },
          {
            "q": "Your function returned both y and z, and you seed its pullback with a dict holding only y. Which check refuses that, and what is it comparing?",
            "a": "The structure check, before a single leaf is looked at. The cotangent argument has to carry the output's treedef, so vjp compares PyTreeDef({'y': *}) against PyTreeDef({'y': *, 'z': *}) and prints both sides of the disagreement."
          }
        ],
        "work": [
          {
            "id": "sort-the-errors",
            "label": "collect five real structure failures from your own code and sort each one into structure or leaves before reading past the first line of the message",
            "href": "#two-failures-two-messages"
          },
          {
            "id": "none-grads",
            "label": "write one tree.map that handles a None gradient correctly with is_leaf, and prove the output treedef still matches params",
            "href": "#map-reads-the-first-tree-and-trusts-the-rest"
          }
        ]
      },
      {
        "id": "nodes-you-register",
        "num": 3,
        "title": "Nodes you register",
        "lede": "An object JAX has never heard of does not raise. It becomes a single leaf, and the complaint arrives later from somewhere else, which is why registration is worth getting exactly right the first time.",
        "goal": "Register a container correctly, decide field by field what belongs in the children and what belongs in the aux slot, predict which of the two forces a recompile, and write an unflatten function that survives being called with objects you never put there.",
        "sections": [
          {
            "h": "an unregistered object is a single leaf",
            "ps": [
              "Wrap two arrays in a plain dataclass, flatten it, and there is exactly one leaf: the object itself. `PyTreeDef(*)` is JAX reporting that it found nothing it knew how to walk into, and it reports that quietly, at flatten time, with no error anywhere.",
              "The refusal comes later and from a different direction entirely. jit takes that single leaf, tries to turn it into an abstract array, and fails on the type. The message names the path, which here is just the argument, and suggests static_argnums, which is the wrong fix for a state object holding arrays you want traced.",
              "Chapter 7 puts it as opting in: a class joins the pytree system the moment you register it. This is what the interval before that looks like from the inside, and it is quiet until an actual transformation touches the object."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): a dataclass JAX has not been told about, flattened and then jitted",
              "lang": "python",
              "text": "from dataclasses import dataclass\n\nimport jax\nimport jax.numpy as jnp\n\n@dataclass\nclass State:\n    w: jax.Array\n    step: int\n\ns = State(jnp.ones(3), 0)\nprint(jax.tree.structure(s), len(jax.tree.leaves(s)))\ntry:\n    jax.jit(lambda st: st.w * 2)(s)\nexcept TypeError as e:\n    print(str(e).split(\"abstract array. \", 1)[1].split(\"\\n\")[0])\n\n# PyTreeDef(*) 1\n# The problematic value is of type <class '__main__.State'> and was passed to the function at path st."
            }
          },
          {
            "h": "two functions and a slot for everything else",
            "ps": [
              "`register_pytree_node` takes a type and two functions. Flatten returns the children plus one more value, and unflatten receives that value back along with the rebuilt children. The reference calls the extra value \"some hashable auxiliary data to be stored in the treedef and to be passed to the unflatten_func\", and both halves of that phrase carry a requirement.",
              "The printed treedef shows the split in one line. `PyTreeDef(CustomNode(Layer[relu], [*, *]))` puts the aux value in brackets after the type name and the children in the star positions after it. Everything in the brackets is structure. Everything in the stars is data.",
              "Which side a field lands on decides how it behaves under every transformation. Children get traced, batched and differentiated like any other leaf. Aux data is carried along untouched and handed back to your constructor on the way out, which is why the activation name below survives a round trip through jit without ever becoming a tracer."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): one registered container, its treedef, and a round trip through jit",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\n\nclass Layer:\n    def __init__(self, w, b, act):\n        self.w, self.b, self.act = w, b, act\n\njax.tree_util.register_pytree_node(\n    Layer,\n    lambda l: ((l.w, l.b), l.act),        # children first, then the aux slot\n    lambda act, children: Layer(*children, act=act),\n)\n\nlayer = Layer(jnp.ones((2, 2)), jnp.zeros(2), \"relu\")\nprint(jax.tree.structure(layer))\nprint([x.shape for x in jax.tree.leaves(layer)])\ndoubled = jax.jit(lambda l: Layer(l.w * 2, l.b, l.act))(layer)\nprint(doubled.act, doubled.w[0, 0])\n\n# PyTreeDef(CustomNode(Layer[relu], [*, *]))\n# [(2, 2), (2,)]\n# relu 2.0"
            }
          },
          {
            "h": "the aux slot is part of the cache key",
            "ps": [
              "`register_dataclass` writes the same split for you from field metadata. Mark a field static and it goes into the aux tuple, so a `Cfg` with `depth=2` prints as `Cfg[(2,)]` and that 2 is now part of the structure rather than part of the data.",
              "Changing it therefore retraces. The counter below sits at 1 across two calls whose arrays differ, then steps to 2 the moment depth goes from 2 to 3. Chapter 3 and LAB J2 name the components of the jit cache key; this is the pytree component moving, driven by a field you declared static.",
              "That gives you a design rule with a cost attached. A field in the aux slot is free to read inside the function and can be branched on with an ordinary Python `if`, because it is a real value at trace time. It also multiplies your executables by its number of distinct values, so anything with more than a handful of them belongs in the children."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): a static field lands in the treedef, and changing it costs a trace",
              "lang": "python",
              "text": "from dataclasses import dataclass, field\n\nimport jax\nimport jax.numpy as jnp\n\n@jax.tree_util.register_dataclass\n@dataclass\nclass Cfg:\n    w: jax.Array\n    depth: int = field(metadata=dict(static=True), default=2)\n\ntraces = 0\n\n@jax.jit\ndef scaled(c):\n    global traces\n    traces += 1\n    return c.w * c.depth\n\nprint(jax.tree.structure(Cfg(jnp.ones(3), 2)))\nscaled(Cfg(jnp.ones(3), 2)); print(traces)\nscaled(Cfg(jnp.zeros(3), 2)); print(traces)\nscaled(Cfg(jnp.zeros(3), 3)); print(traces)\n\n# PyTreeDef(CustomNode(Cfg[(2,)], [*]))\n# 1\n# 1\n# 2"
            }
          },
          {
            "h": "an array in the aux slot breaks the comparison",
            "ps": [
              "The API docs state the requirement plainly: \"Metadata fields must be static, hashable, immutable objects, as these objects are used to generate JIT cache keys. In particular, metadata fields cannot contain jax.Array or numpy.ndarray objects.\" Put one there anyway and the first call goes through without complaint.",
              "The second call is where it lands. Looking up the cache means comparing this call's treedef against the stored one, that comparison reaches the aux values, and comparing two numpy arrays produces an array of booleans rather than a yes or no. What surfaces is numpy's ambiguous-truth-value error, raised from inside a lookup you never wrote.",
              "The museum's exhibit on a static argument that cannot be a cache key shows the same requirement arriving through jit's own arguments. This is that requirement reached from a custom node instead, and the fix has the same shape: put a hashable, comparable stand-in in the aux slot, and keep the array itself among the children."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): a numpy array in the aux slot, fine once and fatal twice",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\nimport numpy as np\n\nclass Masked:\n    def __init__(self, w, mask):\n        self.w, self.mask = w, mask\n\njax.tree_util.register_pytree_node(\n    Masked,\n    lambda m: ((m.w,), m.mask),           # a numpy array in the aux slot\n    lambda mask, children: Masked(children[0], mask),\n)\n\ndouble = jax.jit(lambda m: m.w * 2)\nprint(double(Masked(jnp.ones(3), np.array([True, False]))))\ntry:\n    double(Masked(jnp.zeros(3), np.array([True, False])))\nexcept ValueError as e:\n    print(e)\n\n# [2. 2. 2.]\n# The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
            }
          },
          {
            "h": "unflatten gets called with things you did not put there",
            "ps": [
              "Unflatten does not only run on your arrays. JAX rebuilds your node whenever it needs a copy of the structure, and whatever it happens to be carrying goes into the leaf positions. So a constructor that validates its arguments, which is ordinary good Python, is the wrong constructor to point unflatten at.",
              "The class below refuses anything that is not a `jax.Array`. Under jit it survives, because the leaves are tracers and tracers are arrays. Under `vmap` it is handed a bare `object`, and under `eval_shape` a `ShapeDtypeStruct`, and it raises in both.",
              "JAX's own source shows why one of those placeholders exists. In `jax/_src/tree_util.py`, `equality_errors_pytreedef` builds a sentinel whose repr is the string `pytree leaf`, unflattens both treedefs with that sentinel repeated once per leaf, and walks the two results to produce a readable diff. Your unflatten function runs during that walk, with the sentinel in hand and no arrays anywhere.",
              "So keep unflatten to assembly. No validation, no `jnp.asarray`, no shape checks, no normalizing of defaults. Put all of that in a separate factory that your own code calls, and let the registered constructor accept whatever it is given."
            ],
            "code": {
              "caption": "run it (verified, jax 0.4.38 CPU): a validating constructor, fine under jit, wrong under vmap and eval_shape",
              "lang": "python",
              "text": "import jax\nimport jax.numpy as jnp\n\nclass Strict:\n    def __init__(self, w):\n        if not isinstance(w, jax.Array):\n            raise TypeError(f\"Strict wants an array, got {type(w).__name__}\")\n        self.w = w\n\njax.tree_util.register_pytree_node(\n    Strict, lambda s: ((s.w,), None), lambda aux, children: Strict(*children)\n)\n\ns = Strict(jnp.ones(3))\nprint(jax.jit(lambda x: Strict(x.w * 2))(s).w)          # tracers and arrays: fine\nfor label, call in [\n    (\"vmap\", lambda: jax.vmap(lambda x: Strict(x.w * 2))(Strict(jnp.ones((4, 3))))),\n    (\"eval_shape\", lambda: jax.eval_shape(lambda x: Strict(x.w * 2), s)),\n]:\n    try:\n        call()\n    except TypeError as e:\n        print(label, \"->\", e)\n\n# [2. 2. 2.]\n# vmap -> Strict wants an array, got object\n# eval_shape -> Strict wants an array, got ShapeDtypeStruct"
            }
          }
        ],
        "readings": [
          {
            "label": "jax.tree_util.register_pytree_node",
            "url": "https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_node.html",
            "note": "the two-function contract, with the hashable-aux requirement in the argument docs"
          },
          {
            "label": "jax.tree_util.register_dataclass",
            "url": "https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_dataclass.html",
            "note": "the data-field and meta-field split, and the sentence about cache keys quoted in this lesson"
          },
          {
            "label": "jax.tree_util.register_static",
            "url": "https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_static.html",
            "note": "the smallest case, a type whose whole value is structure and which carries no leaves at all"
          },
          {
            "label": "jax.eval_shape",
            "url": "https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html",
            "note": "where the ShapeDtypeStruct in the last section comes from"
          }
        ],
        "check": [
          {
            "q": "You flatten a custom object and get PyTreeDef(*) with one leaf. What did you forget, and when will you find out?",
            "a": "You never registered the type, so JAX treats the whole object as an opaque leaf. Nothing raises at flatten time; the failure comes when a transformation tries to make an abstract array out of that leaf, and the message will suggest marking it static, which is not the fix."
          },
          {
            "q": "A field of your registered dataclass is marked static and takes one of two hundred values across a run. What does that cost?",
            "a": "Up to two hundred executables. A static field goes into the aux slot, which lives in the treedef, which is part of the jit cache key, so each distinct value traces and compiles its own program. High-cardinality values belong in the children instead."
          },
          {
            "q": "Why must an unflatten function avoid validating or converting its children?",
            "a": "Because JAX calls it with placeholders. vmap passes bare object instances, eval_shape passes ShapeDtypeStructs, and the structure-diff machinery passes a sentinel once per leaf, so any isinstance check or jnp.asarray in the constructor turns an internal bookkeeping step into a crash."
          }
        ],
        "work": [
          {
            "id": "register-two-ways",
            "label": "register one container of your own twice, once with register_pytree_node and once with register_dataclass, and check both treedefs print the same split",
            "href": "#two-functions-and-a-slot-for-everything-else"
          },
          {
            "id": "count-executables",
            "label": "count the distinct values every static field of your train state takes across one run, and move anything past a handful into the children",
            "href": "#the-aux-slot-is-part-of-the-cache-key"
          },
          {
            "id": "placeholder-proof",
            "label": "call vmap and eval_shape on your own registered node and confirm unflatten survives both without touching the leaves",
            "href": "#unflatten-gets-called-with-things-you-did-not-put-there"
          }
        ]
      }
    ]
  }
]
