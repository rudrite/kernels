// The compiler wall, arced: l:xla becomes seven lessons, five slicing its guide,
// each one idea with its own frame and checks. l:gap deliberately stays a
// single-article unit (its whole job is one bridging idea); its ledger
// gains an applied item instead, edited in data/mastery.ts.
import type { UnitLessons } from './index'

export const COMPILER_WALL_LESSONS: UnitLessons[] = [
  {
    unit: 'l:xla',
    coversGuide: true,
    lessons: [
      {
        id: 'two-dumps-two-truths',
        num: 1,
        title: 'Two dumps, two truths',
        lede: 'Your program passes through XLA twice: once translated, once decided. Only one of those dumps can answer a performance question.',
        goal: 'Pick the right dump for the question you are asking, and explain why the lowered text cannot testify about performance.',
        sections: [],
        guide: { id: 'xla', sections: [0] },
        readings: [
          { label: 'JAX · ahead-of-time lowering', url: 'https://docs.jax.dev/en/latest/aot.html', note: 'the lower and compile calls, in the official telling' },
          { label: 'XLA tools', url: 'https://openxla.org/xla/tools', note: 'the wider toolbox around the two dumps' },
        ],
        check: [
          {
            q: 'A colleague reads .lower().as_text(), sees an expensive-looking loop, and starts optimizing. What did they get wrong?',
            a: 'The lowered StableHLO is the program before any backend decision: nothing is fused, tiled, or laid out yet, so cost read from it is fiction. Performance questions start at .compile().as_text(), where fusions, layouts, and buffers are real.',
          },
          {
            q: 'The compiled dump is many times larger than the lowered one for the same kernel. What is that size difference telling you?',
            a: 'How much the backend decided on your behalf: the extra lines are the fusion ops, layout choices, and buffer assignments the compiler added between your program and the machine.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'the-fusion-taxonomy',
        num: 2,
        title: 'The fusion taxonomy',
        lede: 'Every fusion op in a compiled dump carries a kind, and the kinds are the difference between a wall of names and a diagnosis.',
        goal: 'Classify the fusions in a compiled dump by kind, and judge whether a boundary between two of them was necessary or a missed merge.',
        sections: [],
        guide: { id: 'xla', sections: [1] },
        readings: [
          { label: 'XLA architecture', url: 'https://openxla.org/xla/architecture', note: 'where fusion sits in the pass pipeline' },
          { label: 'Scaling book · rooflines', url: 'https://jax-ml.github.io/scaling-book/roofline/', note: 'why HBM traffic is the thing fusion exists to cut' },
        ],
        check: [
          {
            q: 'What does a loop fusion buy, in one sentence about memory?',
            a: 'A chain of elementwise ops becomes one pass over the data, so no intermediate round-trips through HBM between the steps.',
          },
          {
            q: 'The naive attention dump at 8192 carries the full score matrix through three separate fusions. Why is that the chapter’s clearest evidence of a fusion limit?',
            a: 'The bf16[8192,8192] intermediate materializes in HBM between fusions because a single pass cannot express softmax’s reduce-then-renormalize dependence; the boundary is structural, and crossing it is exactly what the hand-written streaming kernel exists to do.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'layouts',
        num: 3,
        title: 'Layouts',
        lede: 'The compiled dump is the only place you can see how a tensor sits in memory, and the annotation reads in one breath once you know the order.',
        goal: 'Read {1,0}-style layout annotations on sight, predict when a transpose is free, and spot the copy ops layout disagreements insert.',
        sections: [],
        guide: { id: 'xla', sections: [2] },
        readings: [
          { label: 'XLA shapes and layout', url: 'https://openxla.org/xla/shapes', note: 'minor-to-major, defined by the people who wrote it' },
          { label: 'Scaling book · All about TPUs', url: 'https://jax-ml.github.io/scaling-book/tpus/', note: 'the register tiles the layout feeds' },
        ],
        check: [
          {
            q: 'What does {1,0} on a rank-2 tensor say, reading the numbers in order?',
            a: 'Dimensions listed fastest-varying first: dimension 1 is minor, so elements along it sit contiguously, and dimension 0 is major. Row-major, said in the dump’s notation.',
          },
          {
            q: 'When is a transpose free, and what does the dump show when it is not?',
            a: 'Free when the consuming op can read the data under the new annotation, treating a different dimension as minor; the transpose disappears into a layout choice. When ops disagree, XLA inserts a copy op that physically rearranges the bytes, and that copy is the cost you see.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'the-memory-report',
        num: 4,
        title: 'Reading the memory report',
        lede: 'When a kernel asks for more VMEM than exists, the compiler prints the most literal error message in the stack. This lesson reads it line by line.',
        goal: 'Take a VMEM overflow report and name each allocation’s origin, then compute the fix from the verdict line’s own arithmetic.',
        sections: [],
        guide: { id: 'xla', sections: [3] },
        readings: [
          { label: 'Pallas · grids and BlockSpecs', url: 'https://docs.jax.dev/en/latest/pallas/grid_blockspec.html', note: 'the BlockSpec a window allocation came from' },
          { label: 'Cloud TPU system architecture', url: 'https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm', note: 'the ceiling the verdict line states' },
        ],
        check: [
          {
            q: 'A report line says window allocation. What object in your source does that correspond to?',
            a: 'A BlockSpec, turned into physical bytes: the block size you gave the kernel’s grid, now reserved as an actual region of VMEM.',
          },
          {
            q: 'The verdict reads 128.00M requested against 127.94M available. What is the fix, and why is it arithmetic rather than debugging?',
            a: 'Shrink a block spec until the request fits under the stated ceiling. The report lists allocations by size in the same units as the limit, so the amount to cut is a subtraction, not an investigation.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'dumps-on-demand',
        num: 5,
        title: 'Dumps on demand, and where XLA stops',
        lede: 'You can make the compiler write out every pass it runs, and you should know which decisions were never its to make.',
        goal: 'Produce a pass-by-pass dump with the XLA flags, and place the boundary where XLA hands off to libtpu on TPU or to vendor libraries on GPU.',
        sections: [],
        guide: { id: 'xla', sections: [4, 5] },
        readings: [
          { label: 'XLA tools', url: 'https://openxla.org/xla/tools', note: 'the dump flags, catalogued' },
          { label: 'OpenXLA', url: 'https://openxla.org/xla', note: 'the same compiler on both backends, one repository' },
        ],
        check: [
          {
            q: 'You want to watch just the fusion pass transform your module. Which two flags, and what does each do?',
            a: 'XLA_FLAGS=--xla_dump_to=DIR writes every intermediate stage to disk; --xla_dump_hlo_pass_re narrows the dump to passes matching a regex, pointed at fusion to isolate that step’s before and after.',
          },
          {
            q: 'A GPU matmul is slow, and the HLO dump looks fine. Why might staring harder at the dump never find it?',
            a: 'XLA:GPU dispatches big matmuls to cuBLAS rather than compiling them itself, so the slow kernel may be a library selection the HLO never shows. Every path ends in a box the compiler does not open: libtpu and LLO on TPU, ptxas and SASS on GPU.',
          },
        ],
        work: [{ id: 'check', label: 'answer the checks without opening them', href: '#check' }],
      },
      {
        id: 'the-one-level-world',
        num: 6,
        title: 'The one-level world',
        lede: 'Below every tensor compiler sits an IR with no tensors in it. Reading one small dump tells you exactly which facts get dropped on the way down, and why the wall this unit teaches has to exist where it does.',
        goal: 'Read a small LLVM IR dump without a reference open, then name, from the dump itself, the two things the level above knew for free that this level has to be told or has to prove at runtime.',
        sections: [
          {
            h: 'four lines of C, and what clang makes of them',
            ps: [
              'Start with a loop small enough to hold in your head. Three float pointers, a length, one add per element, nothing clever anywhere in it.',
              "Compile that with `clang -S -emit-llvm -O0` and roughly sixty lines of text come back. LLVM IR is what that text is called, and it is the representation every LLVM-based backend optimizes before it emits machine code. Clang emits it. Rust emits it. XLA's CPU backend emits it, and its GPU backend emits it too, which is why this floor is worth an hour of your time even if you never write C again.",
              'One property to hold onto while reading: everything in the dump came from those four lines. Nothing was added by a library, and nothing is hidden in a runtime call.',
            ],
            code: {
              caption: 'vadd.c, the whole input for the clang captures in this lesson',
              text: "void vadd(const float *a, const float *b, float *out, int n) {\n  for (int i = 0; i < n; i++) {\n    out[i] = a[i] + b[i];\n  }\n}",
              lang: 'c',
            },
          },
          {
            h: 'values, blocks, and address arithmetic',
            ps: [
              'Three habits get you through any dump at this level. Read `%N` as a value that is defined exactly once, on the left of one instruction, and referenced by name after that. The jaxpr chapter at /l/jaxpr teaches that same one-name-one-definition property where you first met it, and it means here what it means there.',
              'Read a label with a `; preds =` comment after it as a basic block, a straight run of instructions with one entry and one exit. There is no `for` at this level. The loop is three blocks, one that tests, one that does the work, one that increments, wired together by explicit branches back to the test. Five blocks in the whole function at `-O0`, counting the entry and the exit.',
              'Read `getelementptr` as arithmetic that touches no memory. It computes an address from a base pointer, an element type, and an index, and hands it to a separate `load` or `store`. That leaves exactly one instruction in the body doing what the C source asked for, `%25 = fadd float %19, %24`. Everything around it is bookkeeping, most of it because `-O0` keeps every variable in an `alloca` slot and reloads it each time round.',
            ],
            code: {
              caption: 'clang -S -emit-llvm -O0 vadd.c: the three loop blocks, verbatim (Apple clang 21.0.0, target x86_64-apple-macosx26.0.0)',
              text: "10:                                               ; preds = %30, %4\n  %11 = load i32, ptr %9, align 4\n  %12 = load i32, ptr %8, align 4\n  %13 = icmp slt i32 %11, %12\n  br i1 %13, label %14, label %33\n\n14:                                               ; preds = %10\n  %15 = load ptr, ptr %5, align 8\n  %16 = load i32, ptr %9, align 4\n  %17 = sext i32 %16 to i64\n  %18 = getelementptr inbounds float, ptr %15, i64 %17\n  %19 = load float, ptr %18, align 4\n  %20 = load ptr, ptr %6, align 8\n  %21 = load i32, ptr %9, align 4\n  %22 = sext i32 %21 to i64\n  %23 = getelementptr inbounds float, ptr %20, i64 %22\n  %24 = load float, ptr %23, align 4\n  %25 = fadd float %19, %24\n  %26 = load ptr, ptr %7, align 8\n  %27 = load i32, ptr %9, align 4\n  %28 = sext i32 %27 to i64\n  %29 = getelementptr inbounds float, ptr %26, i64 %28\n  store float %25, ptr %29, align 4\n  br label %30\n\n30:                                               ; preds = %14\n  %31 = load i32, ptr %9, align 4\n  %32 = add nsw i32 %31, 1\n  store i32 %32, ptr %9, align 4\n  br label %10, !llvm.loop !6",
              lang: 'text',
              full: {
                text: "; ModuleID = 'vadd.c'\nsource_filename = \"vadd.c\"\ntarget datalayout = \"e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\ntarget triple = \"x86_64-apple-macosx26.0.0\"\n\n; Function Attrs: noinline nounwind optnone ssp uwtable\ndefine void @vadd(ptr noundef %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) #0 {\n  %5 = alloca ptr, align 8\n  %6 = alloca ptr, align 8\n  %7 = alloca ptr, align 8\n  %8 = alloca i32, align 4\n  %9 = alloca i32, align 4\n  store ptr %0, ptr %5, align 8\n  store ptr %1, ptr %6, align 8\n  store ptr %2, ptr %7, align 8\n  store i32 %3, ptr %8, align 4\n  store i32 0, ptr %9, align 4\n  br label %10\n\n10:                                               ; preds = %30, %4\n  %11 = load i32, ptr %9, align 4\n  %12 = load i32, ptr %8, align 4\n  %13 = icmp slt i32 %11, %12\n  br i1 %13, label %14, label %33\n\n14:                                               ; preds = %10\n  %15 = load ptr, ptr %5, align 8\n  %16 = load i32, ptr %9, align 4\n  %17 = sext i32 %16 to i64\n  %18 = getelementptr inbounds float, ptr %15, i64 %17\n  %19 = load float, ptr %18, align 4\n  %20 = load ptr, ptr %6, align 8\n  %21 = load i32, ptr %9, align 4\n  %22 = sext i32 %21 to i64\n  %23 = getelementptr inbounds float, ptr %20, i64 %22\n  %24 = load float, ptr %23, align 4\n  %25 = fadd float %19, %24\n  %26 = load ptr, ptr %7, align 8\n  %27 = load i32, ptr %9, align 4\n  %28 = sext i32 %27 to i64\n  %29 = getelementptr inbounds float, ptr %26, i64 %28\n  store float %25, ptr %29, align 4\n  br label %30\n\n30:                                               ; preds = %14\n  %31 = load i32, ptr %9, align 4\n  %32 = add nsw i32 %31, 1\n  store i32 %32, ptr %9, align 4\n  br label %10, !llvm.loop !6\n\n33:                                               ; preds = %10\n  ret void\n}\n\nattributes #0 = { noinline nounwind optnone ssp uwtable \"darwin-stkchk-strong-link\" \"frame-pointer\"=\"all\" \"min-legal-vector-width\"=\"0\" \"no-trapping-math\"=\"true\" \"probe-stack\"=\"___chkstk_darwin\" \"stack-protector-buffer-size\"=\"8\" \"target-cpu\"=\"penryn\" \"target-features\"=\"+cmov,+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87\" \"tune-cpu\"=\"generic\" }\n\n!llvm.module.flags = !{!0, !1, !2, !3, !4}\n!llvm.ident = !{!5}\n\n!0 = !{i32 2, !\"SDK Version\", [2 x i32] [i32 26, i32 5]}\n!1 = !{i32 1, !\"wchar_size\", i32 4}\n!2 = !{i32 8, !\"PIC Level\", i32 2}\n!3 = !{i32 7, !\"uwtable\", i32 2}\n!4 = !{i32 7, !\"frame-pointer\", i32 2}\n!5 = !{!\"Apple clang version 21.0.0 (clang-2100.1.1.101)\"}\n!6 = distinct !{!6, !7}\n!7 = !{!\"llvm.loop.mustprogress\"}",
                label: 'the whole 67-line module, allocas and metadata included',
              },
            },
          },
          {
            h: 'the type system, and the word missing from it',
            ps: [
              "LLVM's type system is short enough to list from memory. Integers of any width, a handful of floating-point types, `ptr`, vectors, labels, tokens, metadata, and two aggregates: arrays and structs. Fetch the language reference, search it for the word tensor, and you get zero hits. That absence is the cleanest statement of this lesson available.",
              'The closest LLVM comes is an intrinsic. `llvm.matrix.multiply.*` takes two flat vectors and three constant integers, and the reference describes it as treating `%A` as a `<OuterRows> x <Inner>` matrix. So the shape sits in the argument list of one call rather than in the type of any value, and nothing in the verifier connects the two.',
              '>> A shape at this level is either an array type used for address arithmetic or three integers handed to an intrinsic. It is never a property of a value.',
              'Notice what `ptr` does not say, either. It carries no element type at all, so `load float, ptr %18` is where the float appears, and the same address could be read as an i32 by the next instruction without the type system objecting.',
            ],
            table: {
              caption: 'one 8 by 1024 array of f32, and what each notation still knows about it',
              cols: ['written as', 'what it fixes', 'what it leaves open'],
              rows: [
                ['tensor<8x1024xf32>', 'element type, rank, both extents, and that this is one whole value', 'everything about memory: no address, no stride, and no aliasing question to ask'],
                ['[8 x [1024 x float]]', 'element type and both extents, as arithmetic for one getelementptr', 'whether two such pointers overlap, and whether anything downstream still treats this as one value'],
                ['ptr', 'that this is an address', 'element type, extent, alignment, provenance, all of which move to the load, the store, or metadata'],
                ['<8 x float>', 'a register-width value, one machine operation wide', 'any relation to the array it was read out of'],
              ],
            },
          },
          {
            h: 'the same add, emitted by XLA',
            ps: [
              'None of this is theoretical for a JAX program. Set `XLA_FLAGS=--xla_dump_to=DIR` around a CPU compile and the dump directory gains two `.ll` files per module, one as XLA emitted it and one after LLVM has optimized it. Which flag does what belongs to another lesson in this unit; what matters here is what the emission looks like.',
              'Lower `a + b` on two arrays of shape (8, 1024) and XLA writes the loop nest itself, two levels deep, with block names that carry the HLO instruction that produced them. The shape survives into `getelementptr inbounds [8 x [1024 x float]]`, and that array type is the last trace of it. No value in the module has that type.',
              "The metadata is the part worth staring at. Both loads carry `!noalias !3`, the store carries `!alias.scope !3`, and the domain string is XLA's own: `!5 = !{!\"XLA host kernel add.3 AA domain\"}`. Buffer assignment already knew these three slices do not overlap, and metadata is the only channel that fact had left. The `!dereferenceable` note says `i64 32768`, which is 8 by 1024 by 4 bytes, spelled out because the type no longer says it.",
            ],
            code: {
              caption: 'module_0005.jit__lambda_.ir-no-opt.ll, the inner loop body XLA:CPU emitted for one stablehlo.add (jax 0.4.38, CPU backend, XLA_FLAGS=--xla_dump_to)',
              text: "add.3.loop_body.dim.1:                            ; preds = %add.3.loop_header.dim.1\n  %4 = getelementptr inbounds [8 x [1024 x float]], ptr %arg0, i64 0, i64 %add.3.indvar.dim.0, i64 %add.3.indvar.dim.1\n  %5 = load float, ptr %4, align 4, !invariant.load !0, !noalias !3\n  %6 = getelementptr inbounds [8 x [1024 x float]], ptr %arg1, i64 0, i64 %add.3.indvar.dim.0, i64 %add.3.indvar.dim.1\n  %7 = load float, ptr %6, align 4, !invariant.load !0, !noalias !3\n  %add.3 = fadd float %5, %7\n  %8 = getelementptr inbounds [8 x [1024 x float]], ptr %arg2, i64 0, i64 %add.3.indvar.dim.0, i64 %add.3.indvar.dim.1\n  store float %add.3, ptr %8, align 4, !alias.scope !3\n  %invar.inc5 = add nuw nsw i64 %add.3.indvar.dim.1, 1\n  store i64 %invar.inc5, ptr %add.3.invar_address.dim.1, align 4\n  br label %add.3.loop_header.dim.1",
              lang: 'text',
              full: {
                text: "; ModuleID = '__compute_module'\nsource_filename = \"__compute_module\"\ntarget datalayout = \"e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\ntarget triple = \"x86_64-unknown-darwin25.5.0\"\n\n%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }\n%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }\n%XLA_CPU_KernelThread = type { i64, i64, i64 }\n%XLA_CPU_KernelArg = type { ptr, i64 }\n\n; Function Attrs: uwtable\ndefine ptr @add.3(ptr %0) #0 {\n  %add.3.invar_address.dim.1 = alloca i64, align 8\n  %add.3.invar_address.dim.0 = alloca i64, align 8\n  %tdims_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0\n  %tdims = load ptr, ptr %tdims_gep, align 8\n  %tdim_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 0\n  %tdim_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 1\n  %tdim_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 2\n  %tdim_x = load i64, ptr %tdim_x_gep, align 4\n  %tdim_y = load i64, ptr %tdim_y_gep, align 4\n  %tdim_z = load i64, ptr %tdim_z_gep, align 4\n  %tid_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1\n  %tids = load ptr, ptr %tid_gep, align 8\n  %tid_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 0\n  %tid_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 1\n  %tid_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 2\n  %tid_x = load i64, ptr %tid_x_gep, align 4\n  %tid_y = load i64, ptr %tid_y_gep, align 4\n  %tid_z = load i64, ptr %tid_z_gep, align 4\n  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3\n  %args = load ptr, ptr %args_gep, align 8\n  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0\n  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !2\n  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3\n  %args2 = load ptr, ptr %args_gep1, align 8\n  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0\n  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !2\n  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3\n  %args4 = load ptr, ptr %args_gep3, align 8\n  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0\n  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !2\n  store i64 0, ptr %add.3.invar_address.dim.0, align 4\n  br label %add.3.loop_header.dim.0\n\nadd.3.loop_header.dim.0:                          ; preds = %add.3.loop_exit.dim.1, %1\n  %add.3.indvar.dim.0 = load i64, ptr %add.3.invar_address.dim.0, align 4\n  %2 = icmp uge i64 %add.3.indvar.dim.0, 8\n  br i1 %2, label %add.3.loop_exit.dim.0, label %add.3.loop_body.dim.0\n\nadd.3.loop_body.dim.0:                            ; preds = %add.3.loop_header.dim.0\n  store i64 0, ptr %add.3.invar_address.dim.1, align 4\n  br label %add.3.loop_header.dim.1\n\nadd.3.loop_header.dim.1:                          ; preds = %add.3.loop_body.dim.1, %add.3.loop_body.dim.0\n  %add.3.indvar.dim.1 = load i64, ptr %add.3.invar_address.dim.1, align 4\n  %3 = icmp uge i64 %add.3.indvar.dim.1, 1024\n  br i1 %3, label %add.3.loop_exit.dim.1, label %add.3.loop_body.dim.1\n\nadd.3.loop_body.dim.1:                            ; preds = %add.3.loop_header.dim.1\n  %4 = getelementptr inbounds [8 x [1024 x float]], ptr %arg0, i64 0, i64 %add.3.indvar.dim.0, i64 %add.3.indvar.dim.1\n  %5 = load float, ptr %4, align 4, !invariant.load !0, !noalias !3\n  %6 = getelementptr inbounds [8 x [1024 x float]], ptr %arg1, i64 0, i64 %add.3.indvar.dim.0, i64 %add.3.indvar.dim.1\n  %7 = load float, ptr %6, align 4, !invariant.load !0, !noalias !3\n  %add.3 = fadd float %5, %7\n  %8 = getelementptr inbounds [8 x [1024 x float]], ptr %arg2, i64 0, i64 %add.3.indvar.dim.0, i64 %add.3.indvar.dim.1\n  store float %add.3, ptr %8, align 4, !alias.scope !3\n  %invar.inc5 = add nuw nsw i64 %add.3.indvar.dim.1, 1\n  store i64 %invar.inc5, ptr %add.3.invar_address.dim.1, align 4\n  br label %add.3.loop_header.dim.1\n\nadd.3.loop_exit.dim.1:                            ; preds = %add.3.loop_header.dim.1\n  %invar.inc = add nuw nsw i64 %add.3.indvar.dim.0, 1\n  store i64 %invar.inc, ptr %add.3.invar_address.dim.0, align 4\n  br label %add.3.loop_header.dim.0\n\nadd.3.loop_exit.dim.0:                            ; preds = %add.3.loop_header.dim.0\n  br label %return\n\nreturn:                                           ; preds = %add.3.loop_exit.dim.0\n  ret ptr null\n}\n\nattributes #0 = { uwtable \"frame-pointer\"=\"all\" \"prefer-vector-width\"=\"256\" }\n\n!0 = !{}\n!1 = !{i64 32768}\n!2 = !{i64 64}\n!3 = !{!4}\n!4 = !{!\"result slice: {index:0, offset:0, size:32768}\", !5}\n!5 = !{!\"XLA host kernel add.3 AA domain\"}",
                label: 'the whole emitted kernel, 91 lines, call frame unpacking included',
              },
            },
          },
          {
            h: 'told, or proved at runtime',
            ps: [
              'Compile the same four-line C loop at `-O2` and the vectorizer does not start with vectors. It starts with a guard: two pointer subtractions, two unsigned compares against 32, and a branch that skips the vector path if either distance is under 32 bytes. That is the compiler proving at runtime what it could not prove at compile time, that the output does not overlap either input. The function goes from five basic blocks to eleven, and most of the new ones are the guard and the scalar tails it needs.',
              "XLA's version of the same add never asks. Its optimized dump walks from the entry block straight to `br label %vector.ph` and loads `<8 x float>` in the body, because the `!noalias` metadata was already sitting there. Same optimizer, same machine, same arithmetic.",
              '>> Same vectorizer, same machine. One of them was told.',
              'That gap is the shape of every question this unit asks. The level above knows things by construction, that a value is a whole array, that two buffers are distinct, that one op covers every element. The level below can only be told those things or re-derive them. So the decisions the unit reads out of a compiled dump have to be made while that knowledge still exists in the IR, and by the time you are reading LLVM IR they have been made for you. The next lesson is about the design that refuses to accept one level at all.',
            ],
            code: {
              caption: 'clang -S -emit-llvm -O2 vadd.c: the runtime aliasing guard, ahead of any vector instruction',
              text: "9:                                                ; preds = %4\n  %10 = zext nneg i32 %3 to i64\n  %11 = icmp ult i32 %3, 8\n  br i1 %11, label %38, label %12\n\n12:                                               ; preds = %9\n  %13 = sub i64 %7, %6\n  %14 = icmp ult i64 %13, 32\n  %15 = sub i64 %7, %5\n  %16 = icmp ult i64 %15, 32\n  %17 = or i1 %14, %16\n  br i1 %17, label %38, label %18",
              lang: 'text',
              full: {
                text: "; ModuleID = 'vadd.c'\nsource_filename = \"vadd.c\"\ntarget datalayout = \"e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\ntarget triple = \"x86_64-apple-macosx26.0.0\"\n\n; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite) uwtable\ndefine void @vadd(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2, i32 noundef %3) local_unnamed_addr #0 {\n  %5 = ptrtoint ptr %1 to i64\n  %6 = ptrtoint ptr %0 to i64\n  %7 = ptrtoint ptr %2 to i64\n  %8 = icmp sgt i32 %3, 0\n  br i1 %8, label %9, label %58\n\n9:                                                ; preds = %4\n  %10 = zext nneg i32 %3 to i64\n  %11 = icmp ult i32 %3, 8\n  br i1 %11, label %38, label %12\n\n12:                                               ; preds = %9\n  %13 = sub i64 %7, %6\n  %14 = icmp ult i64 %13, 32\n  %15 = sub i64 %7, %5\n  %16 = icmp ult i64 %15, 32\n  %17 = or i1 %14, %16\n  br i1 %17, label %38, label %18\n\n18:                                               ; preds = %12\n  %19 = and i64 %10, 2147483640\n  br label %20\n\n20:                                               ; preds = %20, %18\n  %21 = phi i64 [ 0, %18 ], [ %34, %20 ]\n  %22 = getelementptr inbounds nuw float, ptr %0, i64 %21\n  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16\n  %24 = load <4 x float>, ptr %22, align 4, !tbaa !6\n  %25 = load <4 x float>, ptr %23, align 4, !tbaa !6\n  %26 = getelementptr inbounds nuw float, ptr %1, i64 %21\n  %27 = getelementptr inbounds nuw i8, ptr %26, i64 16\n  %28 = load <4 x float>, ptr %26, align 4, !tbaa !6\n  %29 = load <4 x float>, ptr %27, align 4, !tbaa !6\n  %30 = fadd <4 x float> %24, %28\n  %31 = fadd <4 x float> %25, %29\n  %32 = getelementptr inbounds nuw float, ptr %2, i64 %21\n  %33 = getelementptr inbounds nuw i8, ptr %32, i64 16\n  store <4 x float> %30, ptr %32, align 4, !tbaa !6\n  store <4 x float> %31, ptr %33, align 4, !tbaa !6\n  %34 = add nuw i64 %21, 8\n  %35 = icmp eq i64 %34, %19\n  br i1 %35, label %36, label %20, !llvm.loop !10\n\n36:                                               ; preds = %20\n  %37 = icmp eq i64 %19, %10\n  br i1 %37, label %58, label %38\n\n38:                                               ; preds = %12, %9, %36\n  %39 = phi i64 [ 0, %12 ], [ 0, %9 ], [ %19, %36 ]\n  %40 = and i64 %10, 3\n  %41 = icmp eq i64 %40, 0\n  br i1 %41, label %54, label %42\n\n42:                                               ; preds = %38, %42\n  %43 = phi i64 [ %51, %42 ], [ %39, %38 ]\n  %44 = phi i64 [ %52, %42 ], [ 0, %38 ]\n  %45 = getelementptr inbounds nuw float, ptr %0, i64 %43\n  %46 = load float, ptr %45, align 4, !tbaa !6\n  %47 = getelementptr inbounds nuw float, ptr %1, i64 %43\n  %48 = load float, ptr %47, align 4, !tbaa !6\n  %49 = fadd float %46, %48\n  %50 = getelementptr inbounds nuw float, ptr %2, i64 %43\n  store float %49, ptr %50, align 4, !tbaa !6\n  %51 = add nuw nsw i64 %43, 1\n  %52 = add i64 %44, 1\n  %53 = icmp eq i64 %52, %40\n  br i1 %53, label %54, label %42, !llvm.loop !14\n\n54:                                               ; preds = %42, %38\n  %55 = phi i64 [ %39, %38 ], [ %51, %42 ]\n  %56 = sub nsw i64 %39, %10\n  %57 = icmp ugt i64 %56, -4\n  br i1 %57, label %58, label %59\n\n58:                                               ; preds = %54, %59, %36, %4\n  ret void\n\n59:                                               ; preds = %54, %59\n  %60 = phi i64 [ %88, %59 ], [ %55, %54 ]\n  %61 = getelementptr inbounds nuw float, ptr %0, i64 %60\n  %62 = load float, ptr %61, align 4, !tbaa !6\n  %63 = getelementptr inbounds nuw float, ptr %1, i64 %60\n  %64 = load float, ptr %63, align 4, !tbaa !6\n  %65 = fadd float %62, %64\n  %66 = getelementptr inbounds nuw float, ptr %2, i64 %60\n  store float %65, ptr %66, align 4, !tbaa !6\n  %67 = add nuw nsw i64 %60, 1\n  %68 = getelementptr inbounds nuw float, ptr %0, i64 %67\n  %69 = load float, ptr %68, align 4, !tbaa !6\n  %70 = getelementptr inbounds nuw float, ptr %1, i64 %67\n  %71 = load float, ptr %70, align 4, !tbaa !6\n  %72 = fadd float %69, %71\n  %73 = getelementptr inbounds nuw float, ptr %2, i64 %67\n  store float %72, ptr %73, align 4, !tbaa !6\n  %74 = add nuw nsw i64 %60, 2\n  %75 = getelementptr inbounds nuw float, ptr %0, i64 %74\n  %76 = load float, ptr %75, align 4, !tbaa !6\n  %77 = getelementptr inbounds nuw float, ptr %1, i64 %74\n  %78 = load float, ptr %77, align 4, !tbaa !6\n  %79 = fadd float %76, %78\n  %80 = getelementptr inbounds nuw float, ptr %2, i64 %74\n  store float %79, ptr %80, align 4, !tbaa !6\n  %81 = add nuw nsw i64 %60, 3\n  %82 = getelementptr inbounds nuw float, ptr %0, i64 %81\n  %83 = load float, ptr %82, align 4, !tbaa !6\n  %84 = getelementptr inbounds nuw float, ptr %1, i64 %81\n  %85 = load float, ptr %84, align 4, !tbaa !6\n  %86 = fadd float %83, %85\n  %87 = getelementptr inbounds nuw float, ptr %2, i64 %81\n  store float %86, ptr %87, align 4, !tbaa !6\n  %88 = add nuw nsw i64 %60, 4\n  %89 = icmp eq i64 %88, %10\n  br i1 %89, label %58, label %59, !llvm.loop !16\n}",
                label: 'the whole -O2 function: guard, unrolled vector body, and two scalar tails',
              },
            },
          },
        ],
        readings: [
          { label: 'LLVM language reference', url: 'https://llvm.org/docs/LangRef.html', note: 'the entire type system on one page; search it for the word tensor' },
          { label: 'The often misunderstood GEP instruction', url: 'https://llvm.org/docs/GetElementPtr.html', note: 'why address arithmetic is its own instruction, written by people tired of the question' },
          { label: 'Performance tips for frontend authors', url: 'https://llvm.org/docs/Frontend/PerformanceTips.html', note: 'what a compiler sitting above LLVM is expected to emit, aliasing metadata included' },
        ],
        check: [
          {
            q: 'In `%18 = getelementptr inbounds float, ptr %15, i64 %17`, which part of the original array reached this instruction and which part did not?',
            a: 'The element type reached it, as the `float` the index gets scaled by. Nothing else did: the extent is absent, the rank is absent, and `ptr` does not say what it points at. In the XLA emission the extents survive one step further, inside the array type `[8 x [1024 x float]]`, but only as arithmetic; no value in that module has that type.',
          },
          {
            q: 'The -O2 loop opens with two pointer subtractions and two unsigned compares against 32. What is that code for, and why has the XLA emission no equivalent?',
            a: 'It is a runtime overlap check. The vectorizer may only use wide loads and stores if the output does not alias the inputs, and at this level nothing proves that, so it tests the byte distances and falls back to a scalar loop when they are too small. XLA attaches `!noalias` and `!alias.scope` from buffer assignment instead, so the same vectorizer enters the vector body unconditionally.',
          },
          {
            q: 'Someone proposes skipping the tensor compiler and letting LLVM fuse the loops instead. Which facts would LLVM have to re-derive first?',
            a: 'That the buffers are distinct, that a loop nest is one operation over one whole array, and that a producer has exactly one consumer. All three are free above and cost a dependence analysis over scalar address math here. LLVM does have loop fusion and vectorization; what it lacks is the information. And the algebraic rewrite the chapter at /l/xla is about, the one that turns two-pass softmax into a streaming kernel, is out of reach at every level anyway.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
          { id: 'dump', label: 'compile a four-line loop to LLVM IR at -O0 and -O2, and find the aliasing guard yourself' },
        ],
      },
      {
        id: 'levels-as-a-first-class-idea',
        num: 7,
        title: 'Levels as a first-class idea',
        lede: 'One elementwise add, written three ways in one framework: whole tensors, then loops over buffers, then register-width vectors. What is computed never changes. What has been decided changes at every step.',
        goal: 'Read one computation in three MLIR dialects, say what each rung wrote down that the rung above left open, and define legalization precisely enough to predict which ops a pass expands and which it leaves alone.',
        sections: [
          {
            h: 'the top of the ladder',
            ps: [
              'The vocabulary this lesson runs on was settled one chapter back. The StableHLO chapter at /l/stablehlo pins what MLIR is and what a dialect is; take both as given and put them to work on a program small enough that nothing hides.',
              'Two arrays of shape (8, 1024), added. At the top of the ladder that is one operation over one value, and the type carries the whole shape.',
              'Read it once more for what it does not say. No loop, no order, no memory, no tile, no vector width. A backend is free to choose all of that, and every rung below is a record of it choosing.',
            ],
            code: {
              caption: 'jax 0.4.38 on CPU: jax.jit(lambda a, b: a + b).lower(x, x).as_text(), x of shape (8, 1024) float32',
              text: "module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {\n  func.func public @main(%arg0: tensor<8x1024xf32>, %arg1: tensor<8x1024xf32>) -> (tensor<8x1024xf32> {jax.result_info = \"\"}) {\n    %0 = stablehlo.add %arg0, %arg1 : tensor<8x1024xf32>\n    return %0 : tensor<8x1024xf32>\n  }\n}",
              lang: 'mlir',
            },
          },
          {
            h: 'the middle, where the loops get written down',
            ps: [
              "XLA's CPU tiled pipeline has a pass whose entire job is one rung of this ladder, and its test file shows both ends of the step at once. The input is `linalg.elementwise kind=add` over three `memref<8x1024xf32>` operands. Values have become buffers, which is what the `ins` and `outs` lists are about: an op that used to produce a result now writes into one you hand it.",
              'The expected output is written as FileCheck patterns, so `%[[IV0]]` is a capture name rather than an SSA value. Read past that syntax and the assertion is precise. Two `scf.for` loops with bounds 8 and 1024, the inner one stepping by 8, a `vector.transfer_read` per operand pulling `vector<8xf32>` out of the memref, one `arith.addf`, one `vector.transfer_write`.',
              'Count what arrived at this rung. Loop order, loop bounds, a tile of eight elements along the minor dimension, the register width, and the exact points where memory gets touched. The add itself is untouched. Every new fact is a decision about how, and the pass wrote it into the IR rather than keeping it in a data structure off to the side.',
            ],
            code: {
              caption: 'openxla/xla at commit a6c8e17, xla/backends/cpu/codegen/tiled/transforms/tests/linalg_elementwise_to_vector_pass.mlir: input op and expected output, verbatim',
              text: "func.func @elementwise_add_to_vector(\n    %arg0 : memref<8x1024xf32>,\n    %arg1 : memref<8x1024xf32>,\n    %arg2 : memref<8x1024xf32>) {\n  // CHECK-DAG: %[[MASK:.*]] = ub.poison : f32\n  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index\n  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index\n  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index\n  // CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index\n  // CHECK: scf.for %[[IV0:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {\n  // CHECK:   scf.for %[[IV1:.*]] = %[[C0]] to %[[C1024]] step %[[C8]] {\n  // CHECK:     %[[LHS:.*]] = vector.transfer_read %arg0[%[[IV0]], %[[IV1]]],\n  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x1024xf32>, vector<8xf32>\n  // CHECK:     %[[RHS:.*]] = vector.transfer_read %arg1[%[[IV0]], %[[IV1]]],\n  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x1024xf32>, vector<8xf32>\n  // CHECK:     %[[OUT:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<8xf32>\n  // CHECK:     vector.transfer_write %[[OUT]], %arg2[%[[IV0]], %[[IV1]]]\n  // CHECK-SAME:  {in_bounds = [true]} : vector<8xf32>, memref<8x1024xf32>\n  // CHECK:   }\n  // CHECK: }\n  linalg.elementwise kind=#linalg.elementwise_kind<add>\n    ins(%arg0, %arg1 : memref<8x1024xf32>, memref<8x1024xf32>)\n    outs(%arg2 : memref<8x1024xf32>)\n  return\n}",
              lang: 'mlir',
              full: {
                text: "func.func @elementwise_add_to_vector(\n    %arg0 : memref<8x1024xf32>,\n    %arg1 : memref<8x1024xf32>,\n    %arg2 : memref<8x1024xf32>) {\n  // CHECK-DAG: %[[MASK:.*]] = ub.poison : f32\n  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index\n  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index\n  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index\n  // CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index\n  // CHECK: scf.for %[[IV0:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {\n  // CHECK:   scf.for %[[IV1:.*]] = %[[C0]] to %[[C1024]] step %[[C8]] {\n  // CHECK:     %[[LHS:.*]] = vector.transfer_read %arg0[%[[IV0]], %[[IV1]]],\n  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x1024xf32>, vector<8xf32>\n  // CHECK:     %[[RHS:.*]] = vector.transfer_read %arg1[%[[IV0]], %[[IV1]]],\n  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x1024xf32>, vector<8xf32>\n  // CHECK:     %[[OUT:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<8xf32>\n  // CHECK:     vector.transfer_write %[[OUT]], %arg2[%[[IV0]], %[[IV1]]]\n  // CHECK-SAME:  {in_bounds = [true]} : vector<8xf32>, memref<8x1024xf32>\n  // CHECK:   }\n  // CHECK: }\n  linalg.elementwise kind=#linalg.elementwise_kind<add>\n    ins(%arg0, %arg1 : memref<8x1024xf32>, memref<8x1024xf32>)\n    outs(%arg2 : memref<8x1024xf32>)\n  return\n}\n\n//------\n\nfunc.func @elementwise_add_to_vector_non_multiple_of_8(\n    %arg0 : memref<8x100xf32>,\n    %arg1 : memref<8x100xf32>,\n    %arg2 : memref<8x100xf32>) {\n  // CHECK-DAG: %[[MASK:.*]] = ub.poison : f32\n  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index\n  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index\n  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index\n  // CHECK-DAG: %[[C96:.*]] = arith.constant 96 : index\n  // CHECK: scf.for %[[IV0:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {\n  // CHECK:   scf.for %[[IV1:.*]] = %[[C0]] to %[[C96]] step %[[C8]] {\n  // CHECK:     %[[LHS:.*]] = vector.transfer_read %arg0[%[[IV0]], %[[IV1]]],\n  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x100xf32>, vector<8xf32>\n  // CHECK:     %[[RHS:.*]] = vector.transfer_read %arg1[%[[IV0]], %[[IV1]]],\n  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x100xf32>, vector<8xf32>\n  // CHECK:     %[[OUT:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<8xf32>\n  // CHECK:     vector.transfer_write %[[OUT]], %arg2[%[[IV0]], %[[IV1]]]\n  // CHECK-SAME:  {in_bounds = [true]} : vector<8xf32>, memref<8x100xf32>\n  // CHECK:   }\n  // CHECK: %[[UNROLL_LHS:.*]] = vector.transfer_read %arg0[%[[IV0]], %[[C96]]], %[[MASK]]\n  // CHECK-SAME: {in_bounds = [true]} : memref<8x100xf32>, vector<4xf32>\n  // CHECK: %[[UNROLL_RHS:.*]] = vector.transfer_read %arg1[%[[IV0]], %[[C96]]], %[[MASK]]\n  // CHECK-SAME: {in_bounds = [true]} : memref<8x100xf32>, vector<4xf32>\n  // CHECK: %[[UNROLL_OUT:.*]] = arith.addf %[[UNROLL_LHS]], %[[UNROLL_RHS]] : vector<4xf32>\n  // CHECK: vector.transfer_write %[[UNROLL_OUT]], %arg2[%[[IV0]], %[[C96]]]\n  // CHECK-SAME: {in_bounds = [true]} : vector<4xf32>, memref<8x100xf32>\n  // CHECK: }\n  linalg.elementwise kind=#linalg.elementwise_kind<add>\n    ins(%arg0, %arg1 : memref<8x100xf32>, memref<8x100xf32>)\n    outs(%arg2 : memref<8x100xf32>)\n  return\n}\n\n//------\n\n#map = affine_map<(d0, d1) -> (d0, d1)>\nfunc.func @fused(%arg0: memref<8x1024xf32>,\n                 %arg1: memref<8x1024xf32>,\n                 %arg2: memref<8x1024xf32>) {\n  // CHECK: scf.for\n  // CHECK: scf.for\n  // CHECK-NEXT: vector.transfer_read\n  // CHECK-NEXT: vector.transfer_read\n  // CHECK-NEXT: arith.mulf\n  // CHECK-NEXT: arith.addf\n  // CHECK-NEXT: vector.transfer_write\n  linalg.generic\n    {indexing_maps = [#map, #map, #map], iterator_types = [\"parallel\", \"parallel\"]}\n    ins(%arg0, %arg1 : memref<8x1024xf32>, memref<8x1024xf32>)\n    outs(%arg2 : memref<8x1024xf32>) {\n  ^bb0(%lhs: f32, %rhs: f32, %out: f32):\n    %mul = arith.mulf %lhs, %rhs : f32\n    %res = arith.addf %mul, %rhs : f32\n    linalg.yield %res : f32\n  }\n  return\n}\n\n// -----\n\nfunc.func @elementwise_add_to_vector_small_minor(\n    %arg0 : memref<8x3xf32>,\n    %arg1 : memref<8x3xf32>,\n    %arg2 : memref<8x3xf32>) {\n  linalg.elementwise kind=#linalg.elementwise_kind<add>\n    ins(%arg0, %arg1 : memref<8x3xf32>, memref<8x3xf32>)\n    outs(%arg2 : memref<8x3xf32>)\n  return\n}",
                label: 'all four cases in the test, including the ragged 8x100 tail and a two-op linalg.generic fused into one loop body',
              },
            },
          },
          {
            h: 'several dialects, one module',
            ps: [
              'Six op prefixes appear in that one file: `func`, `linalg`, `scf`, `vector`, `arith`, and `ub`. Nothing was translated into another language or written to another file. Ops from one dialect were replaced in place by ops from others, in the same module, under the same verifier, and a module halfway through a pipeline legitimately holds several dialects at once.',
              'That is what it means for levels to be first-class. Each of those dialects is defined in the same framework as every other, so a pass can rewrite between two of them the way an ordinary pass rewrites within one. The alternative, which the previous lesson walked through, is a single fixed level that every frontend must reach in one jump.',
              'The consequence for tensor programs is direct. A fact survives as long as some dialect on the way down can still express it, so shapes, whole-value semantics, and loop structure can each be dropped at the rung that no longer needs them, instead of all at once at the door.',
            ],
            table: {
              caption: 'the dialects this site quotes, and what each one contributes',
              cols: ['dialect', 'what it adds', 'where it shows up here'],
              rows: [
                ['stablehlo', 'whole-tensor ops with the shape in the type', 'the lowered jax dump above, and the type system lesson at /l/stablehlo'],
                ['linalg', 'structured ops over buffers or tensors: what to compute, not yet in what order', 'the input side of the pass test'],
                ['scf', 'structured control flow, for and if and while, as regions rather than labels', 'the output side of the same test'],
                ['vector', 'register-width values, and the transfers that move them in and out of memory', 'vector<8xf32> and the two transfer ops'],
                ['arith', 'the arithmetic itself, on scalars and on vectors alike', 'arith.addf here, and the tanh expansion below'],
                ['tpu', 'the ops that name one chip, which is Mosaic', 'the module a pallas_call prints, taught at /l/mosaic'],
              ],
            },
          },
          {
            h: 'legalization, and what a target refuses to take',
            ps: [
              'A pass that lowers between dialects needs a definition of acceptable output. MLIR calls that the conversion target, and every op is marked against it as legal, illegal, or dynamic, that last one meaning only some instances qualify. The framework documentation gives its own example, `arith.addi` legal only on 32-bit integers, and rewriting an illegal op into legal ones is what legalization means.',
              'A test in the XLA tree makes the idea concrete with one operation and three types. At f64, `math.tanh` is left exactly as written, because the target accepts it. At f32 it becomes 46 operations, 16 of them constants of a rational approximation. At f16 the pass widens with `arith.extf`, runs the f32 expansion, and narrows back with `arith.truncf`.',
              'So legality belongs to an op together with its types and a target, never to the op alone. When a dump answers one line of your source with a wall of arithmetic, a legalization pattern like this one is the usual reason, and the identity it applied is written into the pass, not derived on the spot.',
            ],
            code: {
              caption: 'openxla/xla at commit a6c8e17, xla/mlir_hlo/tests/legalize-trigonometric-to-approximation.mlir: the RUN line names the pass, then f64 untouched and f32 expanded',
              text: "// RUN: mlir-hlo-opt --mhlo-legalize-trigonometric-to-approximation --split-input-file %s | FileCheck %s\n\n// CHECK-LABEL: @tanh_f64\nfunc.func @tanh_f64(%arg0 : f64) -> f64 {\n  // CHECK: tanh\n  %res = math.tanh %arg0 : f64\n  func.return %res : f64\n}\n\n// -----\n\n// CHECK-LABEL: @tanh_f32\n// CHECK-SAME: (%[[ARG:.*]]: f32) -> f32\nfunc.func @tanh_f32(%arg0 : f32) -> f32 {\n  // CHECK-DAG: %[[C:.*]] = arith.constant -2.76076837E-16 : f32\n  // CHECK-DAG: %[[C0:.*]] = arith.constant 2.00018794E-13 : f32\n  // CHECK-DAG: %[[C1:.*]] = arith.constant -8.60467184E-11 : f32\n  // CHECK-DAG: %[[C2:.*]] = arith.constant 5.12229725E-8 : f32\n  // CHECK-DAG: %[[C3:.*]] = arith.constant 1.48572235E-5 : f32",
              lang: 'mlir',
              full: {
                text: "// RUN: mlir-hlo-opt --mhlo-legalize-trigonometric-to-approximation --split-input-file %s | FileCheck %s\n\n// CHECK-LABEL: @tanh_f64\nfunc.func @tanh_f64(%arg0 : f64) -> f64 {\n  // CHECK: tanh\n  %res = math.tanh %arg0 : f64\n  func.return %res : f64\n}\n\n// -----\n\n// CHECK-LABEL: @tanh_f32\n// CHECK-SAME: (%[[ARG:.*]]: f32) -> f32\nfunc.func @tanh_f32(%arg0 : f32) -> f32 {\n  // CHECK-DAG: %[[C:.*]] = arith.constant -2.76076837E-16 : f32\n  // CHECK-DAG: %[[C0:.*]] = arith.constant 2.00018794E-13 : f32\n  // CHECK-DAG: %[[C1:.*]] = arith.constant -8.60467184E-11 : f32\n  // CHECK-DAG: %[[C2:.*]] = arith.constant 5.12229725E-8 : f32\n  // CHECK-DAG: %[[C3:.*]] = arith.constant 1.48572235E-5 : f32\n  // CHECK-DAG: %[[C4:.*]] = arith.constant 6.37261954E-4 : f32\n  // CHECK-DAG: %[[C5:.*]] = arith.constant 0.00489352457 : f32\n  // CHECK-DAG: %[[C6:.*]] = arith.constant 1.19825836E-6 : f32\n  // CHECK-DAG: %[[C7:.*]] = arith.constant 1.18534706E-4 : f32\n  // CHECK-DAG: %[[C8:.*]] = arith.constant 0.00226843474 : f32\n  // CHECK-DAG: %[[C9:.*]] = arith.constant 0.00489352504 : f32\n  // CHECK-DAG: %[[C10:.*]] = arith.constant 4.000000e-04 : f32\n  // CHECK-DAG: %[[C11:.*]] = arith.constant 7.90531111 : f32\n  // CHECK-DAG: %[[C12:.*]] = arith.constant -7.90531111 : f32\n  // CHECK-DAG: %[[C13:.*]] = arith.constant 1.000000e+00 : f32\n  // CHECK-DAG: %[[C14:.*]] = arith.constant -1.000000e+00 : f32\n  // CHECK-DAG: %[[TMP0:.*]] = arith.mulf %[[ARG]], %[[ARG]] : f32\n  // CHECK-DAG: %[[TMP1:.*]] = arith.mulf %[[TMP0]], %[[C]] : f32\n  // CHECK-DAG: %[[TMP2:.*]] = arith.addf %[[TMP1]], %[[C0]] : f32\n  // CHECK-DAG: %[[TMP3:.*]] = arith.mulf %[[TMP0]], %[[TMP2]] : f32\n  // CHECK-DAG: %[[TMP4:.*]] = arith.addf %[[TMP3]], %[[C1]] : f32\n  // CHECK-DAG: %[[TMP5:.*]] = arith.mulf %[[TMP0]], %[[TMP4]] : f32\n  // CHECK-DAG: %[[TMP6:.*]] = arith.addf %[[TMP5]], %[[C2]] : f32\n  // CHECK-DAG: %[[TMP7:.*]] = arith.mulf %[[TMP0]], %[[TMP6]] : f32\n  // CHECK-DAG: %[[TMP8:.*]] = arith.addf %[[TMP7]], %[[C3]] : f32\n  // CHECK-DAG: %[[TMP9:.*]] = arith.mulf %[[TMP0]], %[[TMP8]] : f32\n  // CHECK-DAG: %[[TMP10:.*]] = arith.addf %[[TMP9]], %[[C4]] : f32\n  // CHECK-DAG: %[[TMP11:.*]] = arith.mulf %[[TMP0]], %[[TMP10]] : f32\n  // CHECK-DAG: %[[TMP12:.*]] = arith.addf %[[TMP11]], %[[C5]] : f32\n  // CHECK-DAG: %[[TMP13:.*]] = arith.mulf %[[ARG]], %[[TMP12]] : f32\n  // CHECK-DAG: %[[TMP14:.*]] = arith.mulf %[[TMP0]], %[[C6]] : f32\n  // CHECK-DAG: %[[TMP15:.*]] = arith.addf %[[TMP14]], %[[C7]] : f32\n  // CHECK-DAG: %[[TMP16:.*]] = arith.mulf %[[TMP0]], %[[TMP15]] : f32\n  // CHECK-DAG: %[[TMP17:.*]] = arith.addf %[[TMP16]], %[[C8]] : f32\n  // CHECK-DAG: %[[TMP18:.*]] = arith.mulf %[[TMP0]], %[[TMP17]] : f32\n  // CHECK-DAG: %[[TMP19:.*]] = arith.addf %[[TMP18]], %[[C9]] : f32\n  // CHECK-DAG: %[[TMP20:.*]] = arith.divf %[[TMP13]], %[[TMP19]] : f32\n  // CHECK-DAG: %[[TMP21:.*]] = math.absf %[[ARG]] : f32\n  // CHECK-DAG: %[[TMP22:.*]] = arith.cmpf olt, %[[TMP21]], %[[C10]] : f32\n  // CHECK-DAG: %[[TMP23:.*]] = arith.select %[[TMP22]], %[[ARG]], %[[TMP20]] : f32\n  // CHECK-DAG: %[[TMP24:.*]] = arith.cmpf ugt, %[[ARG]], %[[C11]] : f32\n  // CHECK-DAG: %[[TMP25:.*]] = arith.cmpf ult, %[[ARG]], %[[C12]] : f32\n  // CHECK-DAG: %[[IS_NAN:.*]] = arith.cmpf une, %[[ARG]], %[[ARG]] : f32\n  // CHECK-DAG: %[[TMP26:.*]] = arith.select %[[TMP24]], %[[C13]], %[[TMP23]] : f32\n  // CHECK-DAG: %[[TMP27:.*]] = arith.select %[[TMP25]], %[[C14]], %[[TMP26]] : f32\n  // CHECK-DAG: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[ARG]], %[[TMP27]] : f32\n  // CHECK: return %[[RESULT]] : f32\n  %res = math.tanh %arg0 : f32\n  func.return %res : f32\n}\n\n// -----\n\nfunc.func @tanh_f16(%arg0 : f16) -> f16 {\n  // CHECK-LABEL: func @tanh_f16\n  // CHECK-SAME: (%[[ARG:.*]]: f16) -> f16\n  // CHECK: %{{.*}} = arith.extf %[[ARG]] : f16 to f32\n  // CHECK: %[[RES:.*]] = arith.truncf %{{.*}} : f32 to f16\n  // CHECK: return %[[RES]] : f16\n  %res = math.tanh %arg0 : f16\n  func.return %res : f16\n}\n",
                label: 'all three tanh cases, f64 through f16, with the whole f32 approximation',
              },
            },
          },
          {
            h: 'what the ladder never does',
            ps: [
              'Walk the rungs back up and each one added a decision. Buffers, then loop order, then a tile, then a vector width, then instructions. Not one of them changed what gets computed, and that is the contract every lowering pass signs.',
              'Which is exactly why the ceiling this unit states holds all the way down. Two-pass softmax stays two-pass in every dialect on the descent, because no legalization pattern says that a reduce and a divide over the same values are a streaming recurrence. The chapter states the reason, and the fusion taxonomy lesson shows the boundary in a real dump; the ladder is not a second chance at it.',
              '>> Lowering rewrites how. Nothing in the pipeline is allowed to rewrite what.',
              'The bottom rung is the one-level world of the previous lesson, where the dialect distinction disappears along with everything it was carrying. What MLIR keeps explicit as ops and types, LLVM keeps as address arithmetic and metadata, and after that there is only the machine.',
            ],
          },
        ],
        readings: [
          { label: 'MLIR dialect conversion', url: 'https://mlir.llvm.org/docs/DialectConversion/', note: "legal, illegal, and dynamic in the framework's own words, with the partial and full conversion modes" },
          { label: "MLIR: a compiler infrastructure for the end of Moore's law", url: 'https://arxiv.org/abs/2002.11054', note: 'the paper that argued for levels as a first-class idea, by the people who then built it' },
          { label: 'The linalg dialect', url: 'https://mlir.llvm.org/docs/Dialects/Linalg/', note: 'the structured-op layer the middle rung of this lesson came from' },
          { label: 'the pass test this lesson quotes', url: 'https://github.com/openxla/xla/blob/a6c8e1767d219fc6dbc3493f0e02a8b345a2e7b4/xla/backends/cpu/codegen/tiled/transforms/tests/linalg_elementwise_to_vector_pass.mlir', note: "pinned at the commit the excerpt was cut from, in XLA's own tree" },
        ],
        check: [
          {
            q: 'Which facts does the vector form of that add state that the linalg form left open?',
            a: 'The loop order and both bounds, the tile of eight along the minor dimension, the register width as vector<8xf32>, and the exact instructions where memory is read and written. What is computed is identical in the two forms; only the how got written down.',
          },
          {
            q: 'A pass leaves math.tanh alone at f64 and expands it into 46 operations at f32. Which legality action is that, and what does the f16 case add to the picture?',
            a: 'Dynamic legality: the op is legal for some types and illegal for others, so the pass expands only the illegal instances. The f16 case shows the second move available to a legalization pattern, which is converting a type it cannot expand into one it can, with arith.extf in and arith.truncf out around the f32 path.',
          },
          {
            q: 'Progressive lowering added buffers, loops, tiles, and vector widths to one add. Why does the same machinery never turn two-pass softmax into a one-pass kernel?',
            a: 'Because every rung is semantics-preserving by contract. Lowering commits to how a computation runs and is forbidden from changing what it computes, while the streaming rewrite is an algebraic identity about exponentials that no pass in the pipeline holds. The chapter at /l/xla states the same limit from the fusion side.',
          },
        ],
        work: [
          { id: 'check', label: 'answer the checks without opening them', href: '#check' },
          { id: 'name-the-rung', label: 'take any MLIR snippet on this site and name its dialect and its rung before reading the caption' },
        ],
      },
    ],
  },
]
