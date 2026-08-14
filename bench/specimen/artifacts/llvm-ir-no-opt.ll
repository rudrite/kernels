; ==== module part 00 of 04 ====
; ModuleID = '__compute_module_part_00'
source_filename = "__compute_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-darwin25.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.1 = private unnamed_addr constant [4 x i8] c"\00\00\80\FF"
@__llvmsplit_unnamed.2 = private unnamed_addr constant [4 x i8] c"\F3\045>"

; Function Attrs: uwtable
define ptr @multiply_reduce_fusion(ptr %0) #0 {
  %reduce_function_parameter_addresses = alloca ptr, i32 2, align 8
  %reduce_function_return_value_addr = alloca float, align 4
  %arg_addr4 = alloca float, align 4
  %arg_addr = alloca float, align 4
  %reduce.0.inner.invar_address.reduction_dim.1 = alloca i64, align 8
  %accumulator_0 = alloca float, align 4
  %multiply_reduce_fusion.invar_address.dim.0 = alloca i64, align 8
  %tdims_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %tdims = load ptr, ptr %tdims_gep, align 8
  %tdim_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 0
  %tdim_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 1
  %tdim_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 2
  %tdim_x = load i64, ptr %tdim_x_gep, align 4
  %tdim_y = load i64, ptr %tdim_y_gep, align 4
  %tdim_z = load i64, ptr %tdim_z_gep, align 4
  %tid_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 0
  %tid_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 1
  %tid_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 2
  %tid_x = load i64, ptr %tid_x_gep, align 4
  %tid_y = load i64, ptr %tid_y_gep, align 4
  %tid_z = load i64, ptr %tid_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !2
  store i64 0, ptr %multiply_reduce_fusion.invar_address.dim.0, align 4
  br label %multiply_reduce_fusion.loop_header.dim.0

multiply_reduce_fusion.loop_header.dim.0:         ; preds = %reduce.0.inner.loop_exit.reduction_dim.1, %1
  %multiply_reduce_fusion.indvar.dim.0 = load i64, ptr %multiply_reduce_fusion.invar_address.dim.0, align 4
  %2 = icmp uge i64 %multiply_reduce_fusion.indvar.dim.0, 16
  br i1 %2, label %multiply_reduce_fusion.loop_exit.dim.0, label %multiply_reduce_fusion.loop_body.dim.0

multiply_reduce_fusion.loop_body.dim.0:           ; preds = %multiply_reduce_fusion.loop_header.dim.0
  %constant.3 = load float, ptr @__llvmsplit_unnamed.1, align 4
  store float %constant.3, ptr %accumulator_0, align 4
  store i64 0, ptr %reduce.0.inner.invar_address.reduction_dim.1, align 4
  br label %reduce.0.inner.loop_header.reduction_dim.1

reduce.0.inner.loop_header.reduction_dim.1:       ; preds = %reduce.0.inner.loop_body.reduction_dim.1, %multiply_reduce_fusion.loop_body.dim.0
  %reduce.0.inner.indvar.reduction_dim.1 = load i64, ptr %reduce.0.inner.invar_address.reduction_dim.1, align 4
  %3 = icmp uge i64 %reduce.0.inner.indvar.reduction_dim.1, 16
  br i1 %3, label %reduce.0.inner.loop_exit.reduction_dim.1, label %reduce.0.inner.loop_body.reduction_dim.1

reduce.0.inner.loop_body.reduction_dim.1:         ; preds = %reduce.0.inner.loop_header.reduction_dim.1
  %4 = load float, ptr %accumulator_0, align 4
  %5 = getelementptr inbounds [16 x [16 x float]], ptr %arg0, i64 0, i64 %multiply_reduce_fusion.indvar.dim.0, i64 %reduce.0.inner.indvar.reduction_dim.1
  %6 = load float, ptr %5, align 4, !invariant.load !0, !noalias !3
  %constant.4 = load float, ptr @__llvmsplit_unnamed.2, align 4
  %multiply.2 = fmul float %6, %constant.4
  store float %4, ptr %arg_addr, align 4
  store float %multiply.2, ptr %arg_addr4, align 4
  %7 = getelementptr inbounds ptr, ptr %reduce_function_parameter_addresses, i64 0
  store ptr %arg_addr, ptr %7, align 8
  %8 = getelementptr inbounds ptr, ptr %reduce_function_parameter_addresses, i64 1
  store ptr %arg_addr4, ptr %8, align 8
  call void @reduce_function(ptr %reduce_function_return_value_addr, ptr null, ptr %reduce_function_parameter_addresses, ptr null, ptr null, ptr null)
  %9 = load float, ptr %reduce_function_return_value_addr, align 4
  store float %9, ptr %accumulator_0, align 4
  %invar.inc3 = add nuw nsw i64 %reduce.0.inner.indvar.reduction_dim.1, 1
  store i64 %invar.inc3, ptr %reduce.0.inner.invar_address.reduction_dim.1, align 4
  br label %reduce.0.inner.loop_header.reduction_dim.1

reduce.0.inner.loop_exit.reduction_dim.1:         ; preds = %reduce.0.inner.loop_header.reduction_dim.1
  %10 = load float, ptr %accumulator_0, align 4
  %11 = getelementptr inbounds [16 x float], ptr %arg1, i64 0, i64 %multiply_reduce_fusion.indvar.dim.0
  store float %10, ptr %11, align 4, !alias.scope !3
  %invar.inc = add nuw nsw i64 %multiply_reduce_fusion.indvar.dim.0, 1
  store i64 %invar.inc, ptr %multiply_reduce_fusion.invar_address.dim.0, align 4
  br label %multiply_reduce_fusion.loop_header.dim.0

multiply_reduce_fusion.loop_exit.dim.0:           ; preds = %multiply_reduce_fusion.loop_header.dim.0
  br label %return

return:                                           ; preds = %multiply_reduce_fusion.loop_exit.dim.0
  ret ptr null
}

; Function Attrs: alwaysinline uwtable
define internal void @reduce_function(ptr %retval, ptr noalias %run_options, ptr noalias %params, ptr noalias %buffer_table, ptr noalias %status, ptr noalias %prof_counters) #1 {
entry:
  %maximum.20 = alloca float, align 4
  %0 = getelementptr inbounds ptr, ptr %params, i64 0
  %Arg_0.18 = load ptr, ptr %0, align 8, !dereferenceable !6, !align !6
  %1 = getelementptr inbounds ptr, ptr %params, i64 1
  %Arg_1.19 = load ptr, ptr %1, align 8, !dereferenceable !6, !align !6
  %2 = load float, ptr %Arg_0.18, align 4, !alias.scope !7, !noalias !10
  %3 = load float, ptr %Arg_1.19, align 4, !alias.scope !12, !noalias !10
  %4 = call reassoc float @llvm.maximum.f32(float %2, float %3)
  store float %4, ptr %maximum.20, align 4, !alias.scope !10
  %load_ret_value = load float, ptr %maximum.20, align 4
  store float %load_ret_value, ptr %retval, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline uwtable "denormal-fp-math"="preserve-sign" "frame-pointer"="none" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{}
!1 = !{i64 1024}
!2 = !{i64 64}
!3 = !{!4}
!4 = !{!"result slice: {index:12, offset:0, size:64}", !5}
!5 = !{!"XLA host kernel multiply_reduce_fusion AA domain"}
!6 = !{i64 4}
!7 = !{!8}
!8 = !{!"buffer: {index:7, offset:0, size:4}", !9}
!9 = !{!"XLA global AA domain"}
!10 = !{!11}
!11 = !{!"buffer: {index:9, offset:0, size:4}", !9}
!12 = !{!13}
!13 = !{!"buffer: {index:8, offset:0, size:4}", !9}

; ==== module part 01 of 04 ====
; ModuleID = '__compute_module_part_01'
source_filename = "__compute_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-darwin25.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.3 = private unnamed_addr constant [4 x i8] c"\00\00\80\FF"
@__llvmsplit_unnamed.4 = private unnamed_addr constant [4 x i8] c"\F3\045>"

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

; Function Attrs: uwtable
define ptr @subtract_exponential_fusion(ptr %0) #1 {
  %subtract_exponential_fusion.invar_address.dim.1 = alloca i64, align 8
  %subtract_exponential_fusion.invar_address.dim.0 = alloca i64, align 8
  %tdims_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %tdims = load ptr, ptr %tdims_gep, align 8
  %tdim_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 0
  %tdim_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 1
  %tdim_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 2
  %tdim_x = load i64, ptr %tdim_x_gep, align 4
  %tdim_y = load i64, ptr %tdim_y_gep, align 4
  %tdim_z = load i64, ptr %tdim_z_gep, align 4
  %tid_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 0
  %tid_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 1
  %tid_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 2
  %tid_x = load i64, ptr %tid_x_gep, align 4
  %tid_y = load i64, ptr %tid_y_gep, align 4
  %tid_z = load i64, ptr %tid_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !1
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !1
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !1
  store i64 0, ptr %subtract_exponential_fusion.invar_address.dim.0, align 4
  br label %subtract_exponential_fusion.loop_header.dim.0

subtract_exponential_fusion.loop_header.dim.0:    ; preds = %subtract_exponential_fusion.loop_exit.dim.1, %1
  %subtract_exponential_fusion.indvar.dim.0 = load i64, ptr %subtract_exponential_fusion.invar_address.dim.0, align 4
  %2 = icmp uge i64 %subtract_exponential_fusion.indvar.dim.0, 16
  br i1 %2, label %subtract_exponential_fusion.loop_exit.dim.0, label %subtract_exponential_fusion.loop_body.dim.0

subtract_exponential_fusion.loop_body.dim.0:      ; preds = %subtract_exponential_fusion.loop_header.dim.0
  store i64 0, ptr %subtract_exponential_fusion.invar_address.dim.1, align 4
  br label %subtract_exponential_fusion.loop_header.dim.1

subtract_exponential_fusion.loop_header.dim.1:    ; preds = %subtract_exponential_fusion.loop_body.dim.1, %subtract_exponential_fusion.loop_body.dim.0
  %subtract_exponential_fusion.indvar.dim.1 = load i64, ptr %subtract_exponential_fusion.invar_address.dim.1, align 4
  %3 = icmp uge i64 %subtract_exponential_fusion.indvar.dim.1, 16
  br i1 %3, label %subtract_exponential_fusion.loop_exit.dim.1, label %subtract_exponential_fusion.loop_body.dim.1

subtract_exponential_fusion.loop_body.dim.1:      ; preds = %subtract_exponential_fusion.loop_header.dim.1
  %4 = getelementptr inbounds [16 x [16 x float]], ptr %arg1, i64 0, i64 %subtract_exponential_fusion.indvar.dim.0, i64 %subtract_exponential_fusion.indvar.dim.1
  %5 = load float, ptr %4, align 4, !invariant.load !0, !noalias !3
  %constant.2 = load float, ptr @__llvmsplit_unnamed.4, align 4
  %multiply.1 = fmul float %5, %constant.2
  %6 = getelementptr inbounds [16 x float], ptr %arg0, i64 0, i64 %subtract_exponential_fusion.indvar.dim.0
  %7 = load float, ptr %6, align 4, !invariant.load !0, !noalias !3
  %constant.1 = load float, ptr @__llvmsplit_unnamed.3, align 4
  %8 = call float @llvm.maximum.f32(float %7, float %constant.1)
  %subtract.0 = fsub float %multiply.1, %8
  %9 = call float @llvm.exp.f32(float %subtract.0)
  %10 = getelementptr inbounds [16 x [16 x float]], ptr %arg2, i64 0, i64 %subtract_exponential_fusion.indvar.dim.0, i64 %subtract_exponential_fusion.indvar.dim.1
  store float %9, ptr %10, align 4, !alias.scope !3
  %invar.inc5 = add nuw nsw i64 %subtract_exponential_fusion.indvar.dim.1, 1
  store i64 %invar.inc5, ptr %subtract_exponential_fusion.invar_address.dim.1, align 4
  br label %subtract_exponential_fusion.loop_header.dim.1

subtract_exponential_fusion.loop_exit.dim.1:      ; preds = %subtract_exponential_fusion.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %subtract_exponential_fusion.indvar.dim.0, 1
  store i64 %invar.inc, ptr %subtract_exponential_fusion.invar_address.dim.0, align 4
  br label %subtract_exponential_fusion.loop_header.dim.0

subtract_exponential_fusion.loop_exit.dim.0:      ; preds = %subtract_exponential_fusion.loop_header.dim.0
  br label %return

return:                                           ; preds = %subtract_exponential_fusion.loop_exit.dim.0
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

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

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

; Function Attrs: uwtable
define ptr @reduce.33(ptr %0) #0 {
  %reduce_function_parameter_addresses = alloca ptr, i32 2, align 8
  %reduce_function_return_value_addr = alloca float, align 4
  %arg_addr6 = alloca float, align 4
  %arg_addr = alloca float, align 4
  %reduce.33.inner.invar_address.reduction_dim.1 = alloca i64, align 8
  %accumulator_0 = alloca float, align 4
  %reduce.33.invar_address.dim.0 = alloca i64, align 8
  %tdims_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %tdims = load ptr, ptr %tdims_gep, align 8
  %tdim_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 0
  %tdim_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 1
  %tdim_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 2
  %tdim_x = load i64, ptr %tdim_x_gep, align 4
  %tdim_y = load i64, ptr %tdim_y_gep, align 4
  %tdim_z = load i64, ptr %tdim_z_gep, align 4
  %tid_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 0
  %tid_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 1
  %tid_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 2
  %tid_x = load i64, ptr %tid_x_gep, align 4
  %tid_y = load i64, ptr %tid_y_gep, align 4
  %tid_z = load i64, ptr %tid_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !3, !align !2
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !2
  store i64 0, ptr %reduce.33.invar_address.dim.0, align 4
  br label %reduce.33.loop_header.dim.0

reduce.33.loop_header.dim.0:                      ; preds = %reduce.33.inner.loop_exit.reduction_dim.1, %1
  %reduce.33.indvar.dim.0 = load i64, ptr %reduce.33.invar_address.dim.0, align 4
  %2 = icmp uge i64 %reduce.33.indvar.dim.0, 16
  br i1 %2, label %reduce.33.loop_exit.dim.0, label %reduce.33.loop_body.dim.0

reduce.33.loop_body.dim.0:                        ; preds = %reduce.33.loop_header.dim.0
  %3 = load float, ptr %arg1, align 4, !invariant.load !0, !noalias !4
  store float %3, ptr %accumulator_0, align 4
  store i64 0, ptr %reduce.33.inner.invar_address.reduction_dim.1, align 4
  br label %reduce.33.inner.loop_header.reduction_dim.1

reduce.33.inner.loop_header.reduction_dim.1:      ; preds = %reduce.33.inner.loop_body.reduction_dim.1, %reduce.33.loop_body.dim.0
  %reduce.33.inner.indvar.reduction_dim.1 = load i64, ptr %reduce.33.inner.invar_address.reduction_dim.1, align 4
  %4 = icmp uge i64 %reduce.33.inner.indvar.reduction_dim.1, 16
  br i1 %4, label %reduce.33.inner.loop_exit.reduction_dim.1, label %reduce.33.inner.loop_body.reduction_dim.1

reduce.33.inner.loop_body.reduction_dim.1:        ; preds = %reduce.33.inner.loop_header.reduction_dim.1
  %5 = load float, ptr %accumulator_0, align 4
  %6 = getelementptr inbounds [16 x [16 x float]], ptr %arg0, i64 0, i64 %reduce.33.indvar.dim.0, i64 %reduce.33.inner.indvar.reduction_dim.1
  %7 = load float, ptr %6, align 4, !invariant.load !0, !noalias !4
  store float %5, ptr %arg_addr, align 4
  store float %7, ptr %arg_addr6, align 4
  %8 = getelementptr inbounds ptr, ptr %reduce_function_parameter_addresses, i64 0
  store ptr %arg_addr, ptr %8, align 8
  %9 = getelementptr inbounds ptr, ptr %reduce_function_parameter_addresses, i64 1
  store ptr %arg_addr6, ptr %9, align 8
  call void @reduce_function__1(ptr %reduce_function_return_value_addr, ptr null, ptr %reduce_function_parameter_addresses, ptr null, ptr null, ptr null)
  %10 = load float, ptr %reduce_function_return_value_addr, align 4
  store float %10, ptr %accumulator_0, align 4
  %invar.inc5 = add nuw nsw i64 %reduce.33.inner.indvar.reduction_dim.1, 1
  store i64 %invar.inc5, ptr %reduce.33.inner.invar_address.reduction_dim.1, align 4
  br label %reduce.33.inner.loop_header.reduction_dim.1

reduce.33.inner.loop_exit.reduction_dim.1:        ; preds = %reduce.33.inner.loop_header.reduction_dim.1
  %11 = load float, ptr %accumulator_0, align 4
  %12 = getelementptr inbounds [16 x float], ptr %arg2, i64 0, i64 %reduce.33.indvar.dim.0
  store float %11, ptr %12, align 4, !alias.scope !4
  %invar.inc = add nuw nsw i64 %reduce.33.indvar.dim.0, 1
  store i64 %invar.inc, ptr %reduce.33.invar_address.dim.0, align 4
  br label %reduce.33.loop_header.dim.0

reduce.33.loop_exit.dim.0:                        ; preds = %reduce.33.loop_header.dim.0
  br label %return

return:                                           ; preds = %reduce.33.loop_exit.dim.0
  ret ptr null
}

; Function Attrs: alwaysinline uwtable
define internal void @reduce_function__1(ptr %retval, ptr noalias %run_options, ptr noalias %params, ptr noalias %buffer_table, ptr noalias %status, ptr noalias %prof_counters) #1 {
entry:
  %add.32 = alloca float, align 4
  %0 = getelementptr inbounds ptr, ptr %params, i64 0
  %Arg_0.30 = load ptr, ptr %0, align 8, !dereferenceable !3, !align !3
  %1 = getelementptr inbounds ptr, ptr %params, i64 1
  %Arg_1.31 = load ptr, ptr %1, align 8, !dereferenceable !3, !align !3
  %2 = load float, ptr %Arg_0.30, align 4, !alias.scope !7, !noalias !10
  %3 = load float, ptr %Arg_1.31, align 4, !alias.scope !12, !noalias !10
  %add.321 = fadd reassoc float %2, %3
  store float %add.321, ptr %add.32, align 4, !alias.scope !10
  %load_ret_value = load float, ptr %add.32, align 4
  store float %load_ret_value, ptr %retval, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline uwtable "denormal-fp-math"="preserve-sign" "frame-pointer"="none" }

!0 = !{}
!1 = !{i64 1024}
!2 = !{i64 64}
!3 = !{i64 4}
!4 = !{!5}
!5 = !{!"result slice: {index:12, offset:0, size:64}", !6}
!6 = !{!"XLA host kernel reduce.33 AA domain"}
!7 = !{!8}
!8 = !{!"buffer: {index:10, offset:0, size:4}", !9}
!9 = !{!"XLA global AA domain"}
!10 = !{!11}
!11 = !{!"buffer: {index:6, offset:0, size:4}", !9}
!12 = !{!13}
!13 = !{!"buffer: {index:11, offset:0, size:4}", !9}

; ==== module part 03 of 04 ====
; ModuleID = '__compute_module_part_03'
source_filename = "__compute_module"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-darwin25.5.0"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

; Function Attrs: uwtable
define ptr @broadcast_divide_fusion(ptr %0) #0 {
  %broadcast_divide_fusion.invar_address.dim.1 = alloca i64, align 8
  %broadcast_divide_fusion.invar_address.dim.0 = alloca i64, align 8
  %tdims_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %tdims = load ptr, ptr %tdims_gep, align 8
  %tdim_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 0
  %tdim_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 1
  %tdim_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 2
  %tdim_x = load i64, ptr %tdim_x_gep, align 4
  %tdim_y = load i64, ptr %tdim_y_gep, align 4
  %tdim_z = load i64, ptr %tdim_z_gep, align 4
  %tid_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 0
  %tid_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 1
  %tid_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 2
  %tid_x = load i64, ptr %tid_x_gep, align 4
  %tid_y = load i64, ptr %tid_y_gep, align 4
  %tid_z = load i64, ptr %tid_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !0, !dereferenceable !2, !align !2
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !0, !dereferenceable !1, !align !2
  store i64 0, ptr %broadcast_divide_fusion.invar_address.dim.0, align 4
  br label %broadcast_divide_fusion.loop_header.dim.0

broadcast_divide_fusion.loop_header.dim.0:        ; preds = %broadcast_divide_fusion.loop_exit.dim.1, %1
  %broadcast_divide_fusion.indvar.dim.0 = load i64, ptr %broadcast_divide_fusion.invar_address.dim.0, align 4
  %2 = icmp uge i64 %broadcast_divide_fusion.indvar.dim.0, 16
  br i1 %2, label %broadcast_divide_fusion.loop_exit.dim.0, label %broadcast_divide_fusion.loop_body.dim.0

broadcast_divide_fusion.loop_body.dim.0:          ; preds = %broadcast_divide_fusion.loop_header.dim.0
  store i64 0, ptr %broadcast_divide_fusion.invar_address.dim.1, align 4
  br label %broadcast_divide_fusion.loop_header.dim.1

broadcast_divide_fusion.loop_header.dim.1:        ; preds = %broadcast_divide_fusion.loop_body.dim.1, %broadcast_divide_fusion.loop_body.dim.0
  %broadcast_divide_fusion.indvar.dim.1 = load i64, ptr %broadcast_divide_fusion.invar_address.dim.1, align 4
  %3 = icmp uge i64 %broadcast_divide_fusion.indvar.dim.1, 16
  br i1 %3, label %broadcast_divide_fusion.loop_exit.dim.1, label %broadcast_divide_fusion.loop_body.dim.1

broadcast_divide_fusion.loop_body.dim.1:          ; preds = %broadcast_divide_fusion.loop_header.dim.1
  %4 = getelementptr inbounds [16 x [16 x float]], ptr %arg0, i64 0, i64 %broadcast_divide_fusion.indvar.dim.0, i64 %broadcast_divide_fusion.indvar.dim.1
  %5 = load float, ptr %4, align 4, !invariant.load !0, !noalias !3
  %6 = getelementptr inbounds [16 x float], ptr %arg1, i64 0, i64 %broadcast_divide_fusion.indvar.dim.0
  %7 = load float, ptr %6, align 4, !invariant.load !0, !noalias !3
  %divide.0 = fdiv float %5, %7
  %8 = getelementptr inbounds [16 x [16 x float]], ptr %arg2, i64 0, i64 %broadcast_divide_fusion.indvar.dim.0, i64 %broadcast_divide_fusion.indvar.dim.1
  store float %divide.0, ptr %8, align 4, !alias.scope !3
  %invar.inc5 = add nuw nsw i64 %broadcast_divide_fusion.indvar.dim.1, 1
  store i64 %invar.inc5, ptr %broadcast_divide_fusion.invar_address.dim.1, align 4
  br label %broadcast_divide_fusion.loop_header.dim.1

broadcast_divide_fusion.loop_exit.dim.1:          ; preds = %broadcast_divide_fusion.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %broadcast_divide_fusion.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast_divide_fusion.invar_address.dim.0, align 4
  br label %broadcast_divide_fusion.loop_header.dim.0

broadcast_divide_fusion.loop_exit.dim.0:          ; preds = %broadcast_divide_fusion.loop_header.dim.0
  br label %return

return:                                           ; preds = %broadcast_divide_fusion.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!0 = !{}
!1 = !{i64 1024}
!2 = !{i64 64}
!3 = !{!4}
!4 = !{!"result slice: {index:12, offset:2048, size:1024}", !5}
!5 = !{!"XLA host kernel broadcast_divide_fusion AA domain"}
