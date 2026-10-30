// RUN: vision-opt %s --vision-pipeline | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @mlp_layer_forward
  // CHECK: llvm.fmul
  // CHECK: llvm.fadd
  // CHECK: llvm.return
  func.func @mlp_layer_forward(
      %input: tensor<1x64xf32>,
      %w1: tensor<64x32xf32>,
      %b1: tensor<1x32xf32>) -> tensor<1x32xf32> {

    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x32xf32>
    %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<1x32xf32>) -> tensor<1x32xf32>

    %mm = linalg.matmul
      ins(%input, %w1 : tensor<1x64xf32>, tensor<64x32xf32>)
      outs(%fill : tensor<1x32xf32>) -> tensor<1x32xf32>

    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm, %b1 : tensor<1x32xf32>, tensor<1x32xf32>)
      outs(%init : tensor<1x32xf32>) {
    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):
      %sum = arith.addf %in_val, %bias_val : f32
      %relu = arith.maximumf %sum, %c0 : f32
      linalg.yield %relu : f32
    } -> tensor<1x32xf32>

    return %out : tensor<1x32xf32>
  }
}
