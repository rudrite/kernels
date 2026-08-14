module @jit_block attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) -> (tensor<16x32xf32> {jax.result_info = ""}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %2 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %3 = stablehlo.transpose %1, dims = [1, 0] : (tensor<16x32xf32>) -> tensor<32x16xf32>
    %4 = stablehlo.dot_general %0, %3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
    %cst = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %5 = stablehlo.sqrt %cst : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
    %7 = stablehlo.divide %4, %6 : tensor<16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %10 = stablehlo.maximum %9, %8 : tensor<16xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<16xf32>) -> tensor<16x1xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<16x1xf32>) -> tensor<16x16xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<16x16xf32>
    %14 = stablehlo.exponential %13 : tensor<16x16xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.reduce(%14 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<16xf32>) -> tensor<16x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<16x1xf32>) -> tensor<16x16xf32>
    %18 = stablehlo.divide %14, %17 : tensor<16x16xf32>
    %19 = stablehlo.dot_general %18, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x16xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    return %19 : tensor<16x32xf32>
  }
}
