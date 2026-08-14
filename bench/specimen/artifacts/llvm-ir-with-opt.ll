; ==== module part 00 of 04 ====
; ModuleID = '__compute_module_part_00'
source_filename = "__compute_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-darwin25.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @multiply_reduce_fusion(ptr nocapture readonly %0) local_unnamed_addr #0 {
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !2
  br label %middle.block

middle.block:                                     ; preds = %middle.block, %1
  %multiply_reduce_fusion.invar_address.dim.0.05 = phi i64 [ 0, %1 ], [ %invar.inc, %middle.block ]
  %2 = getelementptr inbounds nuw [16 x [16 x float]], ptr %arg0, i64 0, i64 %multiply_reduce_fusion.invar_address.dim.0.05, i64 0
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %wide.load8 = load <8 x float>, ptr %3, align 32, !invariant.load !0, !noalias !3
  %4 = fmul <8 x float> %wide.load8, splat (float 0x3FC6A09E60000000)
  %wide.load = load <8 x float>, ptr %2, align 64, !invariant.load !0, !noalias !3
  %5 = fmul <8 x float> %wide.load, splat (float 0x3FC6A09E60000000)
  %rdx.minmax = tail call reassoc <8 x float> @llvm.maximum.v8f32(<8 x float> %5, <8 x float> %4)
  %6 = tail call reassoc float @llvm.vector.reduce.fmaximum.v8f32(<8 x float> %rdx.minmax)
  %7 = getelementptr inbounds nuw [16 x float], ptr %arg1, i64 0, i64 %multiply_reduce_fusion.invar_address.dim.0.05
  store float %6, ptr %7, align 4, !alias.scope !3
  %invar.inc = add nuw nsw i64 %multiply_reduce_fusion.invar_address.dim.0.05, 1
  %exitcond6 = icmp eq i64 %invar.inc, 16
  br i1 %exitcond6, label %return, label %middle.block

return:                                           ; preds = %middle.block
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fmaximum.v8f32(<8 x float>) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{}
!1 = !{i64 1024}
!2 = !{i64 64}
!3 = !{!4}
!4 = !{!"result slice: {index:12, offset:0, size:64}", !5}
!5 = !{!"XLA host kernel multiply_reduce_fusion AA domain"}

; ==== module part 01 of 04 ====
; ModuleID = '__compute_module_part_01'
source_filename = "__compute_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-darwin25.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @subtract_exponential_fusion(ptr nocapture readonly %0) local_unnamed_addr #0 {
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !0, !dereferenceable !1, !align !1
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !1
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !1
  br label %vector.ph

vector.ph:                                        ; preds = %vector.ph, %1
  %subtract_exponential_fusion.invar_address.dim.0.02 = phi i64 [ 0, %1 ], [ %invar.inc, %vector.ph ]
  %2 = getelementptr inbounds nuw [16 x float], ptr %arg0, i64 0, i64 %subtract_exponential_fusion.invar_address.dim.0.02
  %3 = load float, ptr %2, align 4, !invariant.load !0, !noalias !3
  %broadcast.splatinsert = insertelement <8 x float> poison, float %3, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  %4 = getelementptr inbounds nuw [16 x [16 x float]], ptr %arg1, i64 0, i64 %subtract_exponential_fusion.invar_address.dim.0.02, i64 0
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 32
  %wide.load = load <8 x float>, ptr %4, align 64, !invariant.load !0, !noalias !3
  %wide.load4 = load <8 x float>, ptr %5, align 32, !invariant.load !0, !noalias !3
  %6 = fmul <8 x float> %wide.load, splat (float 0x3FC6A09E60000000)
  %7 = fmul <8 x float> %wide.load4, splat (float 0x3FC6A09E60000000)
  %8 = fsub <8 x float> %6, %broadcast.splat
  %9 = fsub <8 x float> %7, %broadcast.splat
  %10 = fcmp uge <8 x float> %8, splat (float 0xC055F33340000000)
  %11 = select <8 x i1> %10, <8 x float> %8, <8 x float> splat (float 0xC055F33340000000)
  %12 = fcmp ule <8 x float> %11, splat (float 0x4056333340000000)
  %13 = select <8 x i1> %12, <8 x float> %11, <8 x float> splat (float 0x4056333340000000)
  %exp_f32.i5 = fmul <8 x float> %13, splat (float 0x3FF7154760000000)
  %exp_f321.i6 = fadd <8 x float> splat (float 5.000000e-01), %exp_f32.i5
  %14 = call <8 x float> @llvm.floor.v8f32(<8 x float> %exp_f321.i6)
  %15 = fcmp uge <8 x float> %14, splat (float -1.270000e+02)
  %16 = select <8 x i1> %15, <8 x float> %14, <8 x float> splat (float -1.270000e+02)
  %17 = fcmp ule <8 x float> %16, splat (float 1.270000e+02)
  %18 = select <8 x i1> %17, <8 x float> %16, <8 x float> splat (float 1.270000e+02)
  %exp_f322.i7 = fmul <8 x float> splat (float 0x3FE6300000000000), %18
  %19 = fsub <8 x float> %13, %exp_f322.i7
  %exp_f323.i8 = fmul <8 x float> splat (float 0xBF2BD01060000000), %18
  %20 = fsub <8 x float> %19, %exp_f323.i8
  %exp_f324.i9 = fmul <8 x float> %20, splat (float 0x3F2A0D2CE0000000)
  %exp_f325.i10 = fadd <8 x float> splat (float 0x3F56E879C0000000), %exp_f324.i9
  %exp_f326.i11 = fmul <8 x float> %exp_f325.i10, %20
  %exp_f327.i12 = fadd <8 x float> splat (float 0x3F81112100000000), %exp_f326.i11
  %exp_f328.i13 = fmul <8 x float> %exp_f327.i12, %20
  %exp_f329.i14 = fadd <8 x float> splat (float 0x3FA5553820000000), %exp_f328.i13
  %exp_f3210.i15 = fmul <8 x float> %exp_f329.i14, %20
  %exp_f3211.i16 = fadd <8 x float> splat (float 0x3FC5555540000000), %exp_f3210.i15
  %exp_f3212.i17 = fmul <8 x float> %exp_f3211.i16, %20
  %exp_f3213.i18 = fadd <8 x float> splat (float 5.000000e-01), %exp_f3212.i17
  %exp_f3214.i19 = fmul <8 x float> %20, %20
  %exp_f3215.i20 = fmul <8 x float> %exp_f3213.i18, %exp_f3214.i19
  %exp_f3216.i21 = fadd <8 x float> %20, %exp_f3215.i20
  %exp_f3217.i22 = fadd <8 x float> splat (float 1.000000e+00), %exp_f3216.i21
  %21 = fptosi <8 x float> %18 to <8 x i32>
  %22 = add <8 x i32> %21, splat (i32 127)
  %23 = shl <8 x i32> %22, splat (i32 23)
  %24 = bitcast <8 x i32> %23 to <8 x float>
  %exp_f3218.i23 = fmul <8 x float> %exp_f3217.i22, %24
  %25 = fcmp uge <8 x float> %9, splat (float 0xC055F33340000000)
  %26 = select <8 x i1> %25, <8 x float> %9, <8 x float> splat (float 0xC055F33340000000)
  %27 = fcmp ule <8 x float> %26, splat (float 0x4056333340000000)
  %28 = select <8 x i1> %27, <8 x float> %26, <8 x float> splat (float 0x4056333340000000)
  %exp_f32.i = fmul <8 x float> %28, splat (float 0x3FF7154760000000)
  %exp_f321.i = fadd <8 x float> splat (float 5.000000e-01), %exp_f32.i
  %29 = call <8 x float> @llvm.floor.v8f32(<8 x float> %exp_f321.i)
  %30 = fcmp uge <8 x float> %29, splat (float -1.270000e+02)
  %31 = select <8 x i1> %30, <8 x float> %29, <8 x float> splat (float -1.270000e+02)
  %32 = fcmp ule <8 x float> %31, splat (float 1.270000e+02)
  %33 = select <8 x i1> %32, <8 x float> %31, <8 x float> splat (float 1.270000e+02)
  %exp_f322.i = fmul <8 x float> splat (float 0x3FE6300000000000), %33
  %34 = fsub <8 x float> %28, %exp_f322.i
  %exp_f323.i = fmul <8 x float> splat (float 0xBF2BD01060000000), %33
  %35 = fsub <8 x float> %34, %exp_f323.i
  %exp_f324.i = fmul <8 x float> %35, splat (float 0x3F2A0D2CE0000000)
  %exp_f325.i = fadd <8 x float> splat (float 0x3F56E879C0000000), %exp_f324.i
  %exp_f326.i = fmul <8 x float> %exp_f325.i, %35
  %exp_f327.i = fadd <8 x float> splat (float 0x3F81112100000000), %exp_f326.i
  %exp_f328.i = fmul <8 x float> %exp_f327.i, %35
  %exp_f329.i = fadd <8 x float> splat (float 0x3FA5553820000000), %exp_f328.i
  %exp_f3210.i = fmul <8 x float> %exp_f329.i, %35
  %exp_f3211.i = fadd <8 x float> splat (float 0x3FC5555540000000), %exp_f3210.i
  %exp_f3212.i = fmul <8 x float> %exp_f3211.i, %35
  %exp_f3213.i = fadd <8 x float> splat (float 5.000000e-01), %exp_f3212.i
  %exp_f3214.i = fmul <8 x float> %35, %35
  %exp_f3215.i = fmul <8 x float> %exp_f3213.i, %exp_f3214.i
  %exp_f3216.i = fadd <8 x float> %35, %exp_f3215.i
  %exp_f3217.i = fadd <8 x float> splat (float 1.000000e+00), %exp_f3216.i
  %36 = fptosi <8 x float> %33 to <8 x i32>
  %37 = add <8 x i32> %36, splat (i32 127)
  %38 = shl <8 x i32> %37, splat (i32 23)
  %39 = bitcast <8 x i32> %38 to <8 x float>
  %exp_f3218.i = fmul <8 x float> %exp_f3217.i, %39
  %40 = getelementptr inbounds nuw [16 x [16 x float]], ptr %arg2, i64 0, i64 %subtract_exponential_fusion.invar_address.dim.0.02, i64 0
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 32
  store <8 x float> %exp_f3218.i23, ptr %40, align 64, !alias.scope !3
  store <8 x float> %exp_f3218.i, ptr %41, align 32, !alias.scope !3
  %invar.inc = add nuw nsw i64 %subtract_exponential_fusion.invar_address.dim.0.02, 1
  %exitcond3 = icmp eq i64 %invar.inc, 16
  br i1 %exitcond3, label %return, label %vector.ph

return:                                           ; preds = %vector.ph
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.floor.v4f32(<4 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.floor.v8f32(<8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x float> @llvm.floor.v16f32(<16 x float>) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{}
!1 = !{i64 64}
!2 = !{i64 1024}
!3 = !{!4}
!4 = !{!"result slice: {index:3, offset:0, size:1024}", !5}
!5 = !{!"XLA host kernel subtract_exponential_fusion AA domain"}

; ==== module part 02 of 04 ====
; ModuleID = '__compute_module_part_02'
source_filename = "__compute_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-darwin25.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @reduce.33(ptr nocapture readonly %0) local_unnamed_addr #0 {
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !3, !align !2
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !2
  br label %vector.ph

vector.ph:                                        ; preds = %vector.ph, %1
  %reduce.33.invar_address.dim.0.07 = phi i64 [ 0, %1 ], [ %invar.inc, %vector.ph ]
  %accumulator_0.03 = load float, ptr %arg1, align 64
  %2 = insertelement <8 x float> <float poison, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, float %accumulator_0.03, i64 0
  %3 = getelementptr inbounds nuw [16 x [16 x float]], ptr %arg0, i64 0, i64 %reduce.33.invar_address.dim.0.07, i64 0
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %wide.load10 = load <8 x float>, ptr %4, align 32, !invariant.load !0, !noalias !4
  %wide.load = load <8 x float>, ptr %3, align 64, !invariant.load !0, !noalias !4
  %5 = fadd reassoc <8 x float> %2, %wide.load
  %bin.rdx = fadd reassoc <8 x float> %wide.load10, %5
  %6 = tail call reassoc float @llvm.vector.reduce.fadd.v8f32(float -0.000000e+00, <8 x float> %bin.rdx)
  %7 = getelementptr inbounds nuw [16 x float], ptr %arg2, i64 0, i64 %reduce.33.invar_address.dim.0.07
  store float %6, ptr %7, align 4, !alias.scope !4
  %invar.inc = add nuw nsw i64 %reduce.33.invar_address.dim.0.07, 1
  %exitcond8 = icmp eq i64 %invar.inc, 16
  br i1 %exitcond8, label %return, label %vector.ph

return:                                           ; preds = %vector.ph
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v8f32(float, <8 x float>) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{}
!1 = !{i64 1024}
!2 = !{i64 64}
!3 = !{i64 4}
!4 = !{!5}
!5 = !{!"result slice: {index:12, offset:0, size:64}", !6}
!6 = !{!"XLA host kernel reduce.33 AA domain"}

; ==== module part 03 of 04 ====
; ModuleID = '__compute_module_part_03'
source_filename = "__compute_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-darwin25.5.0"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast_divide_fusion(ptr nocapture readonly %0) local_unnamed_addr #0 {
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !2
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  br label %vector.ph

vector.ph:                                        ; preds = %vector.ph, %1
  %broadcast_divide_fusion.invar_address.dim.0.02 = phi i64 [ 0, %1 ], [ %invar.inc, %vector.ph ]
  %2 = getelementptr inbounds nuw [16 x float], ptr %arg1, i64 0, i64 %broadcast_divide_fusion.invar_address.dim.0.02
  %3 = load float, ptr %2, align 4, !invariant.load !0, !noalias !3
  %broadcast.splatinsert = insertelement <8 x float> poison, float %3, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  %4 = getelementptr inbounds nuw [16 x [16 x float]], ptr %arg0, i64 0, i64 %broadcast_divide_fusion.invar_address.dim.0.02, i64 0
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 32
  %wide.load = load <8 x float>, ptr %4, align 64, !invariant.load !0, !noalias !3
  %wide.load4 = load <8 x float>, ptr %5, align 32, !invariant.load !0, !noalias !3
  %6 = fdiv <8 x float> %wide.load, %broadcast.splat
  %7 = fdiv <8 x float> %wide.load4, %broadcast.splat
  %8 = getelementptr inbounds nuw [16 x [16 x float]], ptr %arg2, i64 0, i64 %broadcast_divide_fusion.invar_address.dim.0.02, i64 0
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 32
  store <8 x float> %6, ptr %8, align 64, !alias.scope !3
  store <8 x float> %7, ptr %9, align 32, !alias.scope !3
  %invar.inc = add nuw nsw i64 %broadcast_divide_fusion.invar_address.dim.0.02, 1
  %exitcond3 = icmp eq i64 %invar.inc, 16
  br i1 %exitcond3, label %return, label %vector.ph

return:                                           ; preds = %vector.ph
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!0 = !{}
!1 = !{i64 1024}
!2 = !{i64 64}
!3 = !{!4}
!4 = !{!"result slice: {index:12, offset:2048, size:1024}", !5}
!5 = !{!"XLA host kernel broadcast_divide_fusion AA domain"}
