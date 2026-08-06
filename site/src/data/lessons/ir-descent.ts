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
    ],
  },
]
