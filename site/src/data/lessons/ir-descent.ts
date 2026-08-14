// The IR descent, arced into lessons: source, jaxpr, and stablehlo. These
// units' depth already lives in their guides, told once; each lesson slices
// the guide's sections into one ordered idea and adds the frame around it
// (lede, goal, checks). The hub keeps orientation and stops rendering the
// guide inline (coversGuide).
import type { UnitLessons } from './index'

export const IR_DESCENT_LESSONS: UnitLessons[] = [
  {
    unit: 'l:source',
    coversGuide: true,
    lessons: [
      {
        id: 'what-tracing-takes',
        num: 1,
        title: 'What tracing takes, exactly',
        lede: 'Four things leave your Python the moment a trace runs: closed-over arrays, side effects, shapes, and one side of every branch.',
        goal: 'Given a traced function, predict what froze, what fired once, what forces a retrace, and which branch survived, before running it.',
        sections: [],
        guide: { id: 'source', sections: [0] },
        readings: [
          { label: 'JAX · How JAX primitives work', url: 'https://docs.jax.dev/en/latest/notebooks/How_JAX_primitives_work.html', note: 'the machinery under the freeze, one level down' },
          { label: 'JAX · Common gotchas', url: 'https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html', note: 'the official list of everything this lesson predicts' },
        ],
        check: [
          {
            q: 'A print statement inside a jitted function fires once, then goes silent on later calls with the same shapes. What single sentence explains it?',
            a: 'The print runs at trace time, while JAX walks the function with symbolic tracers, and that walk happens once per distinct shape and dtype; compiled runs replay the recorded computation, which has no print in it.',
          },
          {
            q: 'You call the same jitted function with a (32, 64) input and then a (48, 64) input. What happens, and why?',
            a: 'A fresh trace and a fresh compile. Shapes and dtypes freeze into the jaxpr, so a new input shape cannot reuse the recorded computation; it triggers the whole pipeline again for the new signature.',
          },
          {
            q: 'When does an ordinary Python if inside a traced function give the answer you meant, and when do you need lax.cond?',
            a: 'The if resolves once at trace time to whichever branch the tracer took, so it is only right when the condition does not depend on traced values. lax.cond records both branches and picks at runtime, which is what a data-dependent choice needs.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'inside-the-tracer',
        num: 2,
        title: 'Inside the tracer',
        lede: 'A description of a tracer is not the same thing as watching one run. This lesson is the watching: autodidax, the grammar, and seventeen real programs.',
        goal: 'Build the tracer mechanism in your head well enough that an unfamiliar jaxpr dump reads as the output of a machine you could sketch.',
        sections: [],
        guide: { id: 'source', sections: [1, 2] },
        readings: [
          { label: 'Autodidax: JAX core from scratch', url: 'https://docs.jax.dev/en/latest/autodidax.html', note: 'the working tracer this lesson asks you to read' },
          { label: 'JAX · Understanding jaxprs', url: 'https://docs.jax.dev/en/latest/jaxpr.html', note: 'the grammar, defined precisely; keep it open for the exercises' },
        ],
        check: [
          {
            q: 'You close over a numpy array and trace the function. Where does the array surface in the jaxpr, and where does it move if you pass it as an argument instead?',
            a: 'Closed over, it freezes into the constvars, matched against the consts list on the ClosedJaxpr. Passed as an argument, it becomes an invar, bound fresh on every call.',
          },
          {
            q: 'What is the fastest way to check whether your mental model of tracing survives contact with real programs?',
            a: 'The corpus x-ray in the gym: seventeen real programs with their traced output side by side. Skim a handful and test the rules, frozen constants, vanished branches, shape-locked retraces, against what actually printed.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
    ],
  },
  {
    unit: 'l:jaxpr',
    coversGuide: true,
    lessons: [
      {
        id: 'the-anatomy',
        num: 1,
        title: 'The anatomy of a jaxpr',
        lede: 'A ClosedJaxpr is a jaxpr plus the values it closed over, and the grammar underneath is small enough to hold whole.',
        goal: 'Read any printed jaxpr and name each part: invars, constvars, equations, params, outvars, and say why each value sits where it does.',
        sections: [],
        guide: { id: 'jaxpr', sections: [0] },
        readings: [
          { label: 'JAX · Understanding jaxprs', url: 'https://docs.jax.dev/en/latest/jaxpr.html', note: 'the formal grammar behind this lesson' },
          { label: 'JAX · key concepts', url: 'https://docs.jax.dev/en/latest/key-concepts.html', note: 'where tracing hands over to the jaxpr, in the official telling' },
        ],
        check: [
          {
            q: 'What is the difference between a value in invars and a value behind constvars, in terms of what it says about your function?',
            a: 'Invars are whatever the caller passes, different on every trace. Constvars name values the Python closure fixed at trace time, the same on every call until you retrace.',
          },
          {
            q: 'Where does a dot_general equation keep its dimension numbers, and why are they not a traced value?',
            a: 'In its params, the square-bracket configuration on the equation. Params are the static part a primitive needs fixed before it can run; they configure the operation rather than flow through it.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'transforms-rewrite',
        num: 2,
        title: 'Transforms are jaxpr rewrites',
        lede: 'grad, vmap, scan, remat, shard_map: one small dense function, traced once, then rewritten five ways in front of you.',
        goal: 'Predict what each major transform does to a jaxpr (which grow equations, which only move shapes, which nest bodies) before opening the gallery.',
        sections: [],
        guide: { id: 'jaxpr', sections: [1] },
        readings: [
          { label: 'JAX · automatic differentiation', url: 'https://docs.jax.dev/en/latest/automatic-differentiation.html', note: 'why grad appends a backward half rather than replacing the forward' },
          { label: 'JAX · automatic vectorization', url: 'https://docs.jax.dev/en/latest/automatic-vectorization.html', note: 'why vmap moves shapes and leaves the equation list alone' },
        ],
        check: [
          {
            q: 'Under jax.grad, does the forward jaxpr get replaced by a derivative program? What actually happens?',
            a: 'No. The forward equations stay and reverse-mode autodiff appends the backward pass as more equations in the same jaxpr: transposes undoing layouts, mul and sub chains computing the derivative, flowing the cotangent back.',
          },
          {
            q: 'You diff a jaxpr before and after vmap and the equation list is identical. Where did the transform go?',
            a: 'Into the shapes. Every traced value gained a leading batch dimension matching the mapped axis; dot_general is still dot_general, applied to batched operands.',
          },
          {
            q: 'Where does the loop body live after lax.scan, and what is the carry in that picture?',
            a: 'As a nested jaxpr inside the single scan equation’s params. The carry is the explicit state threaded from one iteration to the next, part of the equation rather than a hidden closure.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'nested-jaxprs',
        num: 3,
        title: 'Reading nested jaxprs',
        lede: 'Some equations carry whole programs in their params. The indented block underneath is a jaxpr like any other, and kernels nest exactly this way.',
        goal: 'Read a cond or scan equation’s nested body as fluently as a flat jaxpr, and recognize the same nesting when a Pallas grid body appears inside one.',
        sections: [],
        guide: { id: 'jaxpr', sections: [2] },
        readings: [
          { label: 'JAX · control flow', url: 'https://docs.jax.dev/en/latest/control-flow.html', note: 'why cond and scan carry bodies instead of branching the trace' },
          { label: 'JAX · Pallas quickstart', url: 'https://docs.jax.dev/en/latest/pallas/quickstart.html', note: 'where the same nesting shows up under a kernel' },
        ],
        check: [
          {
            q: 'lax.cond puts two nested jaxprs in one equation. What must be true of both for the program to type-check?',
            a: 'Each branch is a complete function from the same inputs to the same output shapes and dtypes, so whichever one the runtime picks, the surrounding program sees identical types.',
          },
          {
            q: 'Why does fluency with nested jaxprs pay off in the kernel stages, not just here?',
            a: 'A pipelined Pallas kernel’s per-step body is exactly this kind of nested jaxpr: an inner function with its own binders run once per grid step. Reading scan’s nesting is reading a kernel’s, one level shallower.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'params-and-drills',
        num: 4,
        title: 'Params, and the drills that fix them',
        lede: 'Params are the configuration a primitive needs before it can run. Fluency is repetition, and the gym exists for exactly that.',
        goal: 'Read dimension_numbers, axes, and friends on real equations without a lookup, at drill speed.',
        sections: [],
        guide: { id: 'jaxpr', sections: [3, 4] },
        readings: [
          { label: 'JAX · jax.lax reference', url: 'https://docs.jax.dev/en/latest/jax.lax.html', note: 'every primitive’s params, defined where they live' },
          { label: 'Autodidax: JAX core from scratch', url: 'https://docs.jax.dev/en/latest/autodidax.html', note: 'how tracing produces the params you’ve been reading' },
        ],
        check: [
          {
            q: 'What kind of thing belongs in an equation’s params rather than its operands?',
            a: 'Anything that must be fixed before the primitive can run and is not itself a traced value: dimension numbers for a dot_general, the axes of a reduction, a scan’s body and lengths.',
          },
          {
            q: 'The same primitives keep appearing under Pallas kernels. What changes about the grammar there, and what does not?',
            a: 'Nothing about the grammar changes; a kernel body still lowers through dot_general with dimension_numbers and reductions with explicit axes. What changes is the shapes and the memory the values live in.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'one-name-one-definition',
        num: 5,
        title: 'One name, one definition',
        lede: 'Python let you assign to x three times. The jaxpr that came back has four names, and the rule behind that is the one property every IR below this one also keeps.',
        goal: 'Name the property a jaxpr already has, verify it by counting binders on a program of your own, and say where the phi node an LLVM loop needs went in a scan equation.',
        sections: [
          {
            h: 'three assignments, three names',
            ps: [
              'Write a function that assigns to the same Python name three times, `x = x * 2.0`, then `x = x + 1.0`, then `x = jnp.tanh(x)`, and trace it on one `f32[3]` argument. Python had a single `x` the whole way down, rebound at each line. The jaxpr has four variables.',
              'Every equation binds a new letter on its left and no equation ever writes to a letter twice. `mul` binds `b`, `add` binds `c`, `tanh` binds `d`, and `a`, the argument, is never assigned again. Ask what `x` held at the second line and the printed program cannot answer, because there is no `x` in it. There is a value called `b`, and `b` means that value for the rest of the body.',
              'The property has a name, and it is worth learning here because from this point down the stack you never meet an IR without it: static single assignment, SSA in every compiler paper you will read. Two rules, and the second one carries as much weight as the first. Each variable is bound by exactly one equation. Every use of a variable sits below the equation that bound it.',
              'The word static is doing real work in that name. The rule is about the program text, not about how often a value gets computed while the program runs; a loop body binds its result once on the page and once per iteration on the machine. A jaxpr gets the second rule almost for free, since the body is a flat list in the order the trace recorded it, so reading top to bottom is reading in dependency order.',
            ],
            code: {
              caption: 'a function that rebinds x three times, traced (jax 0.4.38, CPU, jax.make_jaxpr)',
              lang: 'haskell',
              text: '{ lambda ; a:f32[3]. let\n    b:f32[3] = mul a 2.0\n    c:f32[3] = add b 1.0\n    d:f32[3] = tanh c\n  in (d,) }',
            },
          },
          {
            h: 'counted, not taken on faith',
            ps: [
              'A property this load-bearing is worth checking rather than believing, and checking it takes about ten lines. Walk the equations, count left-hand sides by object identity so two different variables that happen to print as the same letter stay separate, and recurse into any jaxpr a primitive carries in its params so nested bodies are counted too.',
              'Run it on the chapter gallery function `dense` and you get 4 bound, 4 distinct, 0 rebound. Run it on `jax.grad` of that function, a program with three times the equations, and you get 12 bound, 12 distinct, 0 rebound. A scan, whose body is a separate nested jaxpr, comes back 4 bound, 4 distinct, 0 rebound. The count never diverges, on any program, which is what a structural guarantee looks like from the outside.',
              'The other half of the property answers to a second pass. Walk the same grad program, keep a set of everything bound so far, and check each equation only reads names already in it: 12 equations, 0 uses before their definition. Nothing forward-references, so there is no order to reconstruct.',
              'The letters themselves belong to the printer, not to the program. Two jaxprs both using `b` are saying nothing to each other, which is why the count works on identity rather than on names.',
            ],
            code: {
              caption: 'the binder count, nested bodies included; run on this repo (jax 0.4.38, CPU)',
              lang: 'python',
              text: 'from collections import Counter\nfrom jax.extend.core import Jaxpr\n\ndef binders(jaxpr, counts):\n    for eqn in jaxpr.eqns:\n        for v in eqn.outvars:\n            counts[id(v)] += 1              # one entry per left-hand side\n        for p in eqn.params.values():\n            for sub in (p if isinstance(p, (tuple, list)) else [p]):\n                inner = getattr(sub, "jaxpr", sub)\n                if isinstance(inner, Jaxpr):\n                    binders(inner, counts)  # scan and cond bodies count too\n    return counts',
            },
          },
          {
            h: 'where another IR pays for the join',
            ps: [
              'Write the same accumulation in C, a carry starting at zero and a loop adding one element at a time, and hand it to clang at `-O1`. The loop block that comes back opens with two instructions that have no counterpart anywhere in a jaxpr. `%9` is the induction variable and `%10` is the carry, and both are phi nodes.',
              'The LLVM reference states what one does in a single sentence: at runtime, the phi instruction logically takes on the value specified by the pair corresponding to the predecessor basic block that executed just prior to the current block. Read the operands and the sentence explains itself. `%10` is `0.000000e+00` if control arrived from block `%4`, the preheader, and `%13` if it arrived from block `%8`, which is the block itself looping around.',
              'The reason LLVM needs the instruction is structural. A function body there is a graph of basic blocks joined by edges, control can reach a block from more than one predecessor, and SSA still insists on one definition per name. So the merge itself has to be an instruction, and that instruction has to name the incoming edges. Nothing else in the language can express a value whose definition depends on how you got here.',
              '>> A phi node is what SSA costs you when control flow is a graph of blocks.',
              'A jaxpr body has no blocks and no edges to merge. It is one straight list of equations, and everything that would branch or repeat is a single equation holding whole jaxprs in its params. The next two sections read what that buys.',
            ],
            code: {
              caption: 'the loop block of a five-line C accumulate, from Apple clang 21.0.0 with -O1 -fno-vectorize -fno-unroll-loops -S -emit-llvm',
              lang: 'text',
              text: '8:                                                ; preds = %4, %8\n  %9 = phi i64 [ 0, %4 ], [ %14, %8 ]\n  %10 = phi float [ 0.000000e+00, %4 ], [ %13, %8 ]\n  %11 = getelementptr inbounds nuw float, ptr %0, i64 %9\n  %12 = load float, ptr %11, align 4, !tbaa !6\n  %13 = fadd float %10, %12\n  %14 = add nuw nsw i64 %9, 1\n  %15 = icmp eq i64 %14, %5\n  br i1 %15, label %6, label %8, !llvm.loop !10',
              full: {
                text: 'define float @accumulate(ptr noundef readonly captures(none) %0, i32 noundef %1) local_unnamed_addr #0 {\n  %3 = icmp sgt i32 %1, 0\n  br i1 %3, label %4, label %6\n\n4:                                                ; preds = %2\n  %5 = zext nneg i32 %1 to i64\n  br label %8\n\n6:                                                ; preds = %8, %2\n  %7 = phi float [ 0.000000e+00, %2 ], [ %13, %8 ]\n  ret float %7\n\n8:                                                ; preds = %4, %8\n  %9 = phi i64 [ 0, %4 ], [ %14, %8 ]\n  %10 = phi float [ 0.000000e+00, %4 ], [ %13, %8 ]\n  %11 = getelementptr inbounds nuw float, ptr %0, i64 %9\n  %12 = load float, ptr %11, align 4, !tbaa !6\n  %13 = fadd float %10, %12\n  %14 = add nuw nsw i64 %9, 1\n  %15 = icmp eq i64 %14, %5\n  br i1 %15, label %6, label %8, !llvm.loop !10\n}',
                label: 'the whole function: four basic blocks, three phi nodes',
              },
            },
          },
          {
            h: 'a branch that binds one name',
            ps: [
              'Take a `lax.cond` on a scalar predicate, doubling on the true side and adding one on the false side. Three equations come back. `gt` binds `c`, the boolean, and `convert_element_type` binds `d`, turning that boolean into an `int32` index. Then one `cond` equation binds `e`, and `e` is the answer whichever way the predicate goes.',
              'Both branches sit in the params as nested jaxprs, each with its own binders, each ending in its own `outvars`. Neither one writes to `e`. They return, and the equation that called them binds the result. The merge that LLVM needed a phi instruction for is the ordinary act of a function returning a value into a caller that names it once.',
              'Two details in this capture are worth pointing at. The branches are a tuple indexed by that `int32`, and index 0 is the false branch, which is why the `add f 1.0` body prints first even though `cond` takes the true function first. And the printer draws every name from one running alphabet, so no letter repeats anywhere on the page, inside branch bodies included. The nesting lesson at /l/jaxpr/nested-jaxprs reads the indentation itself; the point here is only that the inner binders are separate from the outer ones and that the join has a single name.',
            ],
            code: {
              caption: 'lax.cond, doubling against adding one, traced on this repo (jax 0.4.38, CPU)',
              lang: 'haskell',
              text: '{ lambda ; a:f32[] b:f32[3]. let\n    c:bool[] = gt a 0.0\n    d:i32[] = convert_element_type[new_dtype=int32 weak_type=False] c\n    e:f32[3] = cond[\n      branches=(\n        { lambda ; f:f32[3]. let g:f32[3] = add f 1.0 in (g,) }\n        { lambda ; h:f32[3]. let i:f32[3] = mul h 2.0 in (i,) }\n      )\n    ] d b\n  in (e,) }',
            },
          },
          {
            h: 'a loop whose carry is a parameter',
            ps: [
              'Now the accumulation the C loop was doing, written as a `lax.scan`: a scalar carry, an array of four elements, and a body that adds the element to the carry and returns it twice, once as the new carry and once as the per-step output. The whole loop is one equation.',
              'The body is `{ lambda ; e:f32[] f:f32[]. let g:f32[] = add e f in (g, g) }`, and `num_carry=1` in the params says how many of its arguments are carried state. So the loop-carried value is `e`, a formal parameter of the body, bound fresh every time the body is applied. That is the same value the C version needed a phi for: zero on the first iteration, the previous `%13` on every later one. One IR names the incoming edges, the other names an argument. The params lesson at /l/jaxpr/params-and-drills reads the rest of that bracket list.',
              'The equivalence is old and it was stated plainly. Appel wrote a four-page argument in 1998 that SSA and functional programming are the same thing in different notation, and the sentence that matters for reading a scan is this one: the left-hand side of the phi assignment is the formal parameter of the corresponding function, and each right-hand side argument of the phi assignment is the actual parameter of some call to the corresponding function.',
              '>> Wherever there is a formal parameter of a function (in the functional form), there is a phi (in the SSA form).',
              'MLIR made the same choice one level below you, and says so in its rationale: regions represent SSA using block arguments rather than the phi instructions used in LLVM, a choice it calls representationally identical. The StableHLO lesson at /l/stablehlo/control-flow-is-regions reads a `while` carrying its state exactly that way, so the shape you are learning here survives the next lowering intact.',
            ],
            code: {
              caption: 'lax.scan over four elements, the same accumulation as the C loop (jax 0.4.38, CPU)',
              lang: 'haskell',
              text: '{ lambda ; a:f32[] b:f32[4]. let\n    c:f32[] d:f32[4] = scan[\n      _split_transpose=False\n      jaxpr={ lambda ; e:f32[] f:f32[]. let g:f32[] = add e f in (g, g) }\n      length=4\n      linear=(False, False)\n      num_carry=1\n      num_consts=0\n      reverse=False\n      unroll=1\n    ] a b\n  in (c, d) }',
            },
          },
          {
            h: 'what the property buys the reader',
            ps: [
              'Reading a large jaxpr, the question you ask most often is where a value came from. Under SSA that question is a search, not an analysis: the name appears on exactly one left-hand side, somewhere above the line you are staring at, and that equation is the whole answer. In a representation that allows rebinding you would have to reason about which assignment reached this point, and along which path.',
              'The compiler gets the same discount, and the cheapest pass shows it best. Trace a function whose first two lines feed nothing, and the trace already prints the dead product as `_`, a variable nothing reads. Dead code elimination then walks the equations backward once, marking what is used, and drops the rest: the `mul` goes because nothing reads it, and the `sin` goes because the only thing that read it is gone. One reverse pass, no fixed point, because a use can only point upward.',
              'The same reasoning is why the transforms in /l/jaxpr/transforms-rewrite can append and re-shape equations without ever renaming anything around them, and why every IR further down keeps the property rather than dropping it. StableHLO and the MLIR it is written in keep it with block arguments, and XLA keeps it in HLO, where each instruction defines one value that later instructions reference by name. Learning to read equations this way once is learning to read the rest of the descent.',
            ],
            code: {
              caption: 'dead code elimination over a traced jaxpr, before and after jax._src.interpreters.partial_eval.dce_jaxpr (jax 0.4.38, CPU)',
              lang: 'text',
              text: '{ lambda ; a:f32[3]. let\n    b:f32[3] = sin a\n    _:f32[3] = mul b 3.0\n    c:f32[3] = add a 1.0\n    d:f32[3] = tanh c\n  in (d,) }\n\nafter dce_jaxpr(jaxpr, [True]):\n\n{ lambda ; a:f32[3]. let b:f32[3] = add a 1.0; c:f32[3] = tanh b in (c,) }',
            },
          },
        ],
        readings: [
          {
            label: 'LLVM language reference · the phi instruction',
            url: 'https://llvm.org/docs/LangRef.html#phi-instruction',
            note: 'the instruction a jaxpr never needs, defined by the people who ship it',
          },
          {
            label: 'MLIR rationale · block arguments vs PHI nodes',
            url: 'https://mlir.llvm.org/docs/Rationale/Rationale/#block-arguments-vs-phi-nodes',
            note: 'the same equivalence, argued as a design choice with five reasons attached',
          },
          {
            label: 'Appel · SSA is Functional Programming',
            url: 'https://www.cs.princeton.edu/~appel/papers/ssafun.pdf',
            note: 'four pages on why a phi is a formal parameter; the scan section is this paper in JAX notation',
          },
        ],
        check: [
          {
            q: 'Python assigns to x three times and the traced program has four variables in it. What rule produced the extra names?',
            a: 'Static single assignment: each equation binds a fresh variable and nothing is ever rebound, so every assignment in the source becomes a new name. A Python name is a slot the interpreter overwrites; a jaxpr name is a definition, and there is exactly one of them per value.',
          },
          {
            q: 'An LLVM loop needs a phi instruction at the top of its body block, and the equivalent scan jaxpr has nothing like it. Where did the phi go?',
            a: 'Into the body jaxpr, as a binder. The carry is one of the body’s invars, and num_carry in the params says how many, bound fresh each time the body is applied. Passing an argument does the merge that a phi does, which is the correspondence Appel states: a phi’s left-hand side is the formal parameter of the corresponding function.',
          },
          {
            q: 'You are looking at a two-hundred-equation jaxpr and want to know where one value came from. Why is that a search rather than an analysis?',
            a: 'Because the name is bound by exactly one equation, and that equation sits above every use of it. Find the single line with that variable on the left and you have the definition, with no question of which assignment reached this point along which path. The same guarantee is why dead code elimination is one backward pass over the equations.',
          },
        ],
        work: [
          { id: 'binders', label: 'run the binder count on a jaxpr of your own and report all three numbers' },
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
    ],
  },
  {
    unit: 'l:stablehlo',
    coversGuide: true,
    lessons: [
      {
        id: 'the-type-system',
        num: 1,
        title: 'The type system',
        lede: 'Every value carries its type in the text: dimensions first, element type second, and a scalar is just a tensor with no dimensions.',
        goal: 'Read tensor types fluently enough that a shape error diagnoses itself from the dump’s operand types.',
        sections: [],
        guide: { id: 'stablehlo', sections: [0] },
        readings: [
          { label: 'StableHLO spec', url: 'https://openxla.org/stablehlo/spec', note: 'every op with its exact type constraints' },
          { label: 'MLIR language reference', url: 'https://mlir.llvm.org/docs/LangRef/', note: 'the notation StableHLO builds on, for when a dump uses it' },
        ],
        check: [
          {
            q: 'Read tensor<32x64xbf16> aloud, in order. What is each part?',
            a: 'A tensor of 32 by 64 elements, stored in bfloat16: dimensions first, left to right, then the element type. The whole shape story of the value is in that one token.',
          },
          {
            q: 'How does StableHLO write a scalar, and why is that a design decision rather than an accident?',
            a: 'As a rank-zero tensor, tensor<f32>, with no dimension list. One rule, tensor plus dtype, covers everything from a scalar epsilon to a 4096 by 4096 matmul operand, so the type system needs no second case.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'control-flow-is-regions',
        num: 2,
        title: 'Control flow is regions',
        lede: 'No gotos, no basic blocks: a while or an if is one op holding nested bodies, and the op decides which body runs.',
        goal: 'Find your scan’s carry and your cond’s branches inside a real dump, and read region nesting as ordinary control flow.',
        sections: [],
        guide: { id: 'stablehlo', sections: [1] },
        readings: [
          { label: 'StableHLO spec · while', url: 'https://openxla.org/stablehlo/spec#while', note: 'the carry tuple’s contract, in the op’s own definition' },
          { label: 'JAX · control flow', url: 'https://docs.jax.dev/en/latest/control-flow.html', note: 'the source-side of the same lowering' },
        ],
        check: [
          {
            q: 'Where did your lax.scan’s carry go in the StableHLO dump?',
            a: 'Onto the stablehlo.while op itself, as an explicit tuple of operands carried into the body region every iteration and returned updated, not as a hidden closure.',
          },
          {
            q: 'After tracing chose one path for a Python if, why does a lax.cond dump still show two bodies?',
            a: 'cond lowers to stablehlo.if with both branches present as regions on the op; a tensor<i1> predicate selects which region executes at runtime, so the choice survived tracing instead of being resolved by it.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'the-escape-hatch',
        num: 3,
        title: 'custom_call, the escape hatch',
        lede: 'The opset is closed on purpose, and everything outside it, your Pallas kernels included, travels through one op.',
        goal: 'Find the custom_call in a lowered kernel, name its target, and say what rides in the payload and where StableHLO’s responsibility ends.',
        sections: [],
        guide: { id: 'stablehlo', sections: [2] },
        readings: [
          { label: 'StableHLO spec · custom_call', url: 'https://openxla.org/stablehlo/spec#custom_call', note: 'the op’s contract: an opaque target and a payload' },
          { label: 'Pallas documentation', url: 'https://docs.jax.dev/en/latest/pallas/index.html', note: 'the kernel side of the door this lesson opens' },
        ],
        check: [
          {
            q: 'What does pallas_call put into the StableHLO dump, and what does it deliberately not try to do?',
            a: 'One stablehlo.custom_call targeting tpu_custom_call, with the serialized Mosaic module attached as the payload. It does not try to express the kernel body in StableHLO ops; the closed opset stays closed.',
          },
          {
            q: 'A kernel fails somewhere past the custom_call boundary. Why is that a different debugging world?',
            a: 'Past the seam, the payload is compiled by Mosaic, a separate compiler with its own passes and failure modes. StableHLO’s job ended at carrying the payload; the museum’s VMEM overflow capture shows exactly that boundary in a real error.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'why-stable',
        num: 4,
        title: 'Why it is called Stable',
        lede: 'The name is a serialization promise with numbers attached, and it is the reason a saved model outlives the toolchain that made it.',
        goal: 'State what the compatibility guarantee covers, its backward and forward windows, and what that buys a checkpoint.',
        sections: [],
        guide: { id: 'stablehlo', sections: [3, 4] },
        readings: [
          { label: 'StableHLO compatibility', url: 'https://openxla.org/stablehlo/compatibility', note: 'the exact windows this lesson summarizes' },
          { label: 'StableHLO bytecode', url: 'https://openxla.org/stablehlo/bytecode', note: 'the format the guarantee is made about' },
        ],
        check: [
          {
            q: 'What exactly is stable about StableHLO?',
            a: 'The serialized form. Modules serialize to a versioned bytecode with an explicit compatibility guarantee, roughly five years backward and two years forward, not a promise that the opset never changes.',
          },
          {
            q: 'Why does that guarantee matter to someone who just trains models?',
            a: 'A checkpoint exported as StableHLO under one JAX release can be deserialized and compiled by a toolchain years later without a re-export. The saved artifact outlives the framework version that produced it.',
          },
        ],
        work: [
          { id: 'xray', label: 'corpus x-ray: narrate two dumps end to end using all three lessons', href: '/gym/kernels#xray' },
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
      {
        id: 'the-hlo-family',
        num: 5,
        title: 'The HLO family tree',
        lede: 'Five names one letter apart, and none of them is a rung on the same ladder. Each one exists because something broke that the one before it could not fix.',
        goal: 'Given any dump with an hlo in the name, say which family member you are reading, which problem it was created to solve, and what it converts into next.',
        sections: [
          {
            h: 'three prefixes in five lines',
            ps: [
              'Lower a single arctangent and JAX hands back a module carrying three different prefixes. `func.func` is plain upstream MLIR. `chlo.atan` is an op the StableHLO spec does not contain at all. And bolted onto the module itself are `mhlo.num_partitions` and `mhlo.num_replicas`, named for a dialect whose own README says it is deprecated and slated for removal.',
              'None of that is cruft to route around. The five names in this family, HLO and MHLO and StableHLO and VHLO and CHLO, are not five levels of one stack; they are five answers to five different problems, and they arrived in the order the problems did. Read them as history and the prefixes on this dump stop being noise.',
            ],
            code: {
              caption: 'jnp.arctan lowered, verbatim (jax 0.4.38, jaxlib 0.4.38, CPU)',
              lang: 'mlir',
              text: 'module @jit_atan_fn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32> {jax.result_info = ""}) {\n    %0 = chlo.atan %arg0 : tensor<4xf32> -> tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n}',
              full: {
                text: 'module @jit_atan_fn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32> {jax.result_info = ""}) {\n    %0 = chlo.atan %arg0 : tensor<4xf32> -> tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n}\n\nHloModule jit_atan_fn, is_scheduled=true, entry_computation_layout={(f32[4]{0})->f32[4]{0}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}\n\n%fused_computation (param_0: f32[4]) -> f32[4] {\n  %param_0 = f32[4]{0} parameter(0)\n  %constant.0 = f32[] constant(1)\n  %broadcast.0 = f32[4]{0} broadcast(f32[] %constant.0), dimensions={}\n  ROOT %atan2.0 = f32[4]{0} atan2(f32[4]{0} %param_0, f32[4]{0} %broadcast.0), metadata={op_name="jit(atan_fn)/jit(main)/atan" source_file="<string>" source_line=4}\n}\n\nENTRY %main.5 (Arg_0.1: f32[4]) -> f32[4] {\n  %Arg_0.1 = f32[4]{0} parameter(0), metadata={op_name="v"}\n  ROOT %broadcast_atan2_fusion = f32[4]{0} fusion(f32[4]{0} %Arg_0.1), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(atan_fn)/jit(main)/atan" source_file="<string>" source_line=4}\n}\n\n\nmodule @jit_erf_fn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32> {jax.result_info = ""}) {\n    %0 = chlo.erf %arg0 : tensor<4xf32> -> tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n}\n\nHloModule jit_erf_fn, is_scheduled=true, entry_computation_layout={(f32[4]{0})->f32[4]{0}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}\n\nENTRY %main.3 (Arg_0.1: f32[4]) -> f32[4] {\n  %Arg_0.1 = f32[4]{0} parameter(0), metadata={op_name="v"}\n  ROOT %erf.2 = f32[4]{0} erf(f32[4]{0} %Arg_0.1), metadata={op_name="jit(erf_fn)/jit(main)/erf" source_file="<string>" source_line=5}\n}',
                label: 'the whole capture, unedited: two functions, each lowered and then compiled',
              },
            },
          },
          {
            h: 'the compiler had an instruction set before it had a dialect',
            ps: [
              'HLO is the oldest thing in this picture by years, and it was never a dialect of anything. It is an enum. `xla/hlo/ir/hlo_opcode.h` lists 132 opcodes as macro entries of the form `V(kAtan2, "atan2", 2)`, name and printed spelling and arity, and the comment above the list calls them "High-level optimizer instruction opcodes" and describes them as "a flattened form of the UnaryOp, BinaryOp, ... opcodes present in the XLA service protobuf".',
              'That origin explains what HLO is missing. It was the internal vocabulary of one compiler talking to itself, so it needed no written grammar, no version number, and no story for what happens when the compiler upgrades under a saved file. The xla:hlo unit takes apart the invariants HLO offers instead of a spec; what matters here is only the date. Everything else in this family was built after HLO, around HLO, or to replace a job HLO was doing badly.',
            ],
          },
          {
            h: 'MLIR arrived, and HLO moved in',
            ps: [
              'MLIR gave compiler authors a way to keep several op vocabularies inside one module and lower between them under shared infrastructure. XLA wanted its ops in that world, so the ops got an MLIR dialect: MHLO, still sitting in the tree at `xla/mlir_hlo/mhlo`, with 138 op definitions that mirror the HLO opcodes. For a few years MHLO was the bridge every frontend crossed to reach XLA.',
              'That bridge is closed now. The README on that directory opens with a deprecation notice and a line that settles the question of whether you should learn MHLO in any depth.',
              '>> Users of MHLO should migrate to StableHLO whenever possible.',
              'The dialect is going away and the prefix is not, which is why your lowered module still says `mhlo.num_partitions`. XLA keeps a single header listing every attribute a StableHLO module is allowed to carry across the border, and the `mhlo.` namespace is where the facts live that the StableHLO spec has no field for: replica counts, SPMD shardings, entry layouts, buffer donation. An attribute outside that list does not survive the trip.',
            ],
            code: {
              caption: 'openxla/xla at 2c73111, xla/mlir_hlo/utils/unregistered_attributes.h, lines 20 to 46 (excerpted: five of the thirteen constants, comments unedited)',
              lang: 'c',
              text: 'namespace xla {\n\n// This file captures all discardable attributes that XLA supports.\n// Attributes not in this list will be dropped when exporting to StableHLO.\n\n// Module level attributes require namespacing.\ninline constexpr char kMhloCrossProgramPrefetches[] =\n    "mhlo.cross_program_prefetches";\ninline constexpr char kMhloInputOutputAlias[] = "mhlo.input_output_alias";\ninline constexpr char kMhloSpmdOutputSharding[] = "mhlo.spmd_output_sharding";\ninline constexpr char kMhloNumPartitions[] = "mhlo.num_partitions";\ninline constexpr char kMhloNumReplicas[] = "mhlo.num_replicas";',
            },
          },
          {
            h: 'bootstrapped from CHLO and MHLO',
            ps: [
              'The openxla/stablehlo repository opens with an empty initial commit on 3 August 2022. Two weeks later the second commit lands, and its title is the whole design decision: "Bootstrap StableHLO from CHLO/MHLO (#1)". StableHLO did not invent an opset. It copied the one MHLO already had, from a compiler that had been running it in production for years, and then made a promise about it.',
              'The commit message is worth reading because it names the problem in the language of the moment it was written. Frameworks and compilers were passing dialects between repositories with no agreement about what a version bump was allowed to break, and the proposal on the table was a shallow dialect that producers could vendor and upgrade under stated compatibility windows.',
            ],
            code: {
              caption: 'openxla/stablehlo commit d6c918d (2022-08-17), message excerpted',
              lang: 'text',
              text: 'Bootstrap StableHLO from CHLO/MHLO (#1)\n\nRecent discussions highlight an acute need for stability of interchange\ndialects in between ML frameworks and ML compilers in the opensource\ncommunity.\n\nIn a Discourse post, a potential solution was called out: something called\n"shallow dialects" that producers could vendor into their repositories and\nupgrade with well-defined backward compatibility windows.\n\nI think this presents a great opportunity for StableHLO: to start as shallow\nMHLO which will bootstrap us from a well-understood baseline and will enable\nus to provide a service to the community right away - backward compatibility\nguarantees for MHLO (and its sister dialect CHLO as well).',
            },
          },
          {
            h: 'a copy that is only allowed to grow',
            ps: [
              'A promise about serialized bytes needs somewhere to keep the old shapes of things, because the live dialect only ever holds the current shape. That somewhere is VHLO, added to the repo in December 2022, and its dialect definition describes itself as a shallow versioned copy of StableHLO simplified down to a bare minimum, used for upgrades, downgrades, and serialization. Every op in it carries a version range instead of a definition that can be edited.',
              'Watch one change land. StableHLO 1.5.0 made the collectives variadic so `all_gather` could take several operands at once. The op in StableHLO was rewritten in place; the op in VHLO could not be, so `all_gather_v1` had its upper bound closed at 1.4.0 and a new `all_gather_v2` opened at 1.5.0. Both definitions are still in the file, side by side, differing by one `Variadic<>` wrapper.',
              'Count the file and the growth shows: 142 op definitions in VHLO against 117 in StableHLO, and 23 of the VHLO ones have a closed upper bound. Those 23 are ops nothing will ever emit again. They stay because a reader opening a three-year-old artifact needs something to deserialize it into, which is the mechanism behind the compatibility windows the previous lesson quoted.',
            ],
            code: {
              caption: 'openxla/stablehlo at 806a684: VhloOps.td lines 95 to 115, with the matching line from VhloDialect.td',
              lang: 'text',
              text: '// VhloDialect.td, version log\n//   1.5.0: Make collective ops (`all_reduce`, `all_gather`, `all_to_all`) variadic.\n\ndef VHLO_AllGatherOpV1 : VHLO_Op<"all_gather_v1", "0.9.0", "1.4.0"> {\n  let arguments = (ins\n    VHLO_AnyType:$operand,\n    VHLO_AnyAttr:$all_gather_dim,\n    VHLO_AnyAttr:$replica_groups,\n    VHLO_AnyAttr:$channel_id,\n    VHLO_AnyAttr:$use_global_device_ids\n  );\n  let results = (outs VHLO_AnyType:$result);\n}\n\ndef VHLO_AllGatherOpV2 : VHLO_Op<"all_gather_v2", "1.5.0", "current"> {\n  let arguments = (ins\n    Variadic<VHLO_AnyType>:$operands,\n    VHLO_AnyAttr:$all_gather_dim,\n    VHLO_AnyAttr:$replica_groups,\n    VHLO_AnyAttr:$channel_id,\n    VHLO_AnyAttr:$use_global_device_ids\n  );\n  let results = (outs Variadic<VHLO_AnyType>:$results);\n}',
            },
          },
          {
            h: 'the target version is in the first bytes',
            ps: [
              'You can watch VHLO do its job from Python without leaving this machine. `jaxlib.mlir.dialects.stablehlo` exposes StableHLO\'s own serializer, and it takes a target version: the version of the reader you are writing for, not the version you are running. Ask it for a 0.9.0 reader and the same four-op module comes out at 575 bytes; ask for 1.0.0 and it comes out at 459. Both artifacts name their target in a producer string right after the MLIR bytecode magic.',
              "Ask for a version this build does not have and it refuses rather than guessing. The message comes from `VhloToVersion.cpp`, which is also where the harder failure lives: when a program uses an op that has no definition at the requested version, serialization fails with `failed to convert VHLO to v<version>` instead of emitting bytes the old reader would misread. Refusing to downgrade is the promise working, not the promise breaking.",
            ],
            code: {
              caption: 'the script first, then its verbatim stdout and stderr (jax 0.4.38 / jaxlib 0.4.38, CPU); stderr is placed under the call that raised it',
              lang: 'text',
              text: 'print(\'current\', shlo.get_current_version(), \'minimum\', shlo.get_minimum_version())\nfor target in (\'0.9.0\', \'1.0.0\', \'1.8.7\'):\n    blob = shlo.serialize_portable_artifact_str(module_text, target)\n    print(target, len(blob), blob[:4], blob[5:21])\nshlo.serialize_portable_artifact_str(module_text, \'1.9.0\')\n\ncurrent 1.8.7 minimum 0.9.0\n0.9.0 575 b\'ML\\xefR\' b\'StableHLO_v0.9.0\'\n1.0.0 459 b\'ML\\xefR\' b\'StableHLO_v1.0.0\'\n1.8.7 459 b\'ML\\xefR\' b\'StableHLO_v1.8.7\'\nloc("-":1:1): error: target version 1.9.0 is greater than current version 1.8.7\nValueError: failed to serialize module',
            },
          },
          {
            h: 'two fates for one frontend op',
            ps: [
              'That leaves the op the lesson opened on. CHLO is older than StableHLO and came along in the same bootstrap; its dialect description says it models the API surface of the XlaBuilder C++ API, and that whenever the client library uses syntactic sugar or a composition of several ops for one API call, CHLO models the call and supplies conversion patterns to materialize it into lower level dialects. StableHLO has `atan2` and no `atan`, so `jnp.arctan` has nowhere to land except CHLO.',
              'The conversion pattern for it is three lines of TableGen and the comment above it states the identity outright: `atan(x) = atan2(x, 1)`. Compile that same function on CPU and the prediction is sitting in the HLO: a `constant(1)` broadcast to the operand shape, then `atan2(f32[4]{0} %param_0, f32[4]{0} %broadcast.0)`, and the frontend name survives in the instruction metadata as `op_name="jit(atan_fn)/jit(main)/atan"`.',
              'Now do the same with `jax.scipy.special.erf` and something different happens. It lowers to `chlo.erf`, and the compiled entry computation reads the parameter straight into one instruction, `ROOT %erf.2 = f32[4]{0} erf(f32[4]{0} %Arg_0.1)`. No constant, no fusion, no decomposition at all. The reason is that HLO has an `erf` opcode and no `atan` opcode, and XLA ingestion knows it: the function `StablehloToMhlo` in `xla/hlo/translate/stablehlo.cc` runs a CHLO to high-level MHLO pass first, summarized in its own definition as legalizing "CHLO\'s with XLA counterparts, like TopK and Erf", and only then decomposes whatever is left into spec ops.',
              '>> Which fate an op meets is decided by whether HLO already had an opcode for it.',
              'So the family sorts by the problem each member was built for, and the whole tree is legible from that one dump. Here they are on one card.',
            ],
            code: {
              caption: 'openxla/stablehlo at 806a684, stablehlo/transforms/ChloDecompositionPatterns.td lines 51 to 58 and 119 to 120',
              lang: 'text',
              text: '// Express `atan` as\n//   atan(x) = atan2(x, 1)\ndef : Pat<(CHLO_AtanOp NonComplexElementType:$input),\n  (StableHLO_Atan2Op\n    $input,\n    (StableHLO_ConstantLike<"1"> $input)\n  )>;\n\ndef : Pat<(CHLO_TanOp $input),\n          (StableHLO_TanOp $input, ConstDefaultResultAccuracyAttr)>;',
            },
            table: {
              caption: 'op definitions counted by grep at openxla/stablehlo 806a684 and openxla/xla 2c73111',
              cols: ['name', 'where it lives', 'defs', 'the problem it answers'],
              rows: [
                ['HLO', 'openxla/xla, xla/hlo/ir/hlo_opcode.h', '132 opcodes', 'one compiler needs an internal instruction set'],
                ['MHLO', 'openxla/xla, xla/mlir_hlo/mhlo', '138 ops', 'those opcodes need to exist inside MLIR; deprecated'],
                ['StableHLO', 'openxla/stablehlo, StablehloOps.td', '117 ops', 'many frameworks and many backends need one spec to agree on'],
                ['VHLO', 'openxla/stablehlo, VhloOps.td', '142 ops, 23 frozen', 'saved bytes must outlive the compiler that wrote them'],
                ['CHLO', 'openxla/stablehlo, ChloOps.td', '50 ops', 'frontends have ops the spec chose not to carry'],
              ],
            },
          },
        ],
        readings: [
          { label: 'The VHLO dialect', url: 'https://openxla.org/stablehlo/vhlo', note: 'the add-only rule and the versioned-op recipe, written by the people who enforce it' },
          { label: 'StableHLO PR #1, bootstrap from CHLO/MHLO', url: 'https://github.com/openxla/stablehlo/pull/1', note: 'the founding argument in the words it was made in' },
          { label: 'MLIR-HLO README', url: 'https://github.com/openxla/xla/blob/main/xla/mlir_hlo/README.md', note: "MHLO's own deprecation notice, still sitting in the tree" },
          { label: 'CHLO decomposition patterns', url: 'https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/ChloDecompositionPatterns.td', note: 'every rule that turns a client op into spec ops, one file' },
        ],
        check: [
          {
            q: 'Your lowered module carries mhlo.num_partitions, and the MHLO dialect is deprecated. What is actually going on?',
            a: 'The dialect and the attribute namespace are different things. MHLO the dialect is being removed; the mhlo. prefix is the reserved namespace for facts the StableHLO spec has no field for, listed exhaustively in xla/mlir_hlo/utils/unregistered_attributes.h. Anything not on that list is dropped when a module is exported to StableHLO.',
          },
          {
            q: 'StableHLO 1.5.0 made all_gather variadic. Why does VHLO still define all_gather_v1?',
            a: 'Because VHLO is add-only: a change means a new versioned op, never an edit. all_gather_v1 had its upper bound closed at 1.4.0 and all_gather_v2 opened at 1.5.0, so an artifact serialized before the change still has a definition to deserialize into. 23 of VHLO\'s 142 op definitions are frozen that way.',
          },
          {
            q: 'chlo.atan and chlo.erf both leave JAX as CHLO ops. Why does only one of them reach the compiled HLO under its own name?',
            a: 'HLO has an erf opcode and no atan opcode. XLA ingestion runs a CHLO to high-level MHLO pass for the ops with XLA counterparts, so erf stays erf; everything else falls through to the decomposition patterns, where atan becomes atan2(x, 1).',
          },
        ],
        work: [
          { id: 'lower', label: 'lower jnp.arctan and jax.scipy.special.erf, compile both, and find the two different fates yourself', href: '#two-fates-for-one-frontend-op' },
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
        ],
      },
    ],
  },
]
