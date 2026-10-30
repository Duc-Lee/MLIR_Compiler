// =========================================================================
//  Multi-Layer Perceptron (MLP) MNIST Digit Classifier (3-Layer Neural Net)
//  Input:  [1, 784]
//  Layer 1: Linear(784 -> 128) + ReLU
//  Layer 2: Linear(128 -> 64)  + ReLU
//  Layer 3: Linear(64 -> 10)   (Logits)
// =========================================================================

module {
  func.func @mlp_forward(
      %input: tensor<1x784xf32>,
      %w1: tensor<784x128xf32>,
      %b1: tensor<1x128xf32>,
      %w2: tensor<128x64xf32>,
      %b2: tensor<1x64xf32>,
      %w3: tensor<64x10xf32>,
      %b3: tensor<1x10xf32>) -> tensor<1x10xf32> {

    %c0 = arith.constant 0.0 : f32

    // -------------------------------------------------------------
    // LAYER 1: MatMul (1x784 * 784x128 -> 1x128) + Bias + ReLU
    // -------------------------------------------------------------
    %init_l1 = tensor.empty() : tensor<1x128xf32>
    %fill_l1 = linalg.fill ins(%c0 : f32) outs(%init_l1 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %mm1 = linalg.matmul
      ins(%input, %w1 : tensor<1x784xf32>, tensor<784x128xf32>)
      outs(%fill_l1 : tensor<1x128xf32>) -> tensor<1x128xf32>

    // Bias Add & ReLU
    %l1_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm1, %b1 : tensor<1x128xf32>, tensor<1x128xf32>)
      outs(%init_l1 : tensor<1x128xf32>) {
    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):
      %sum = arith.addf %in_val, %bias_val : f32
      %relu = arith.maximumf %sum, %c0 : f32
      linalg.yield %relu : f32
    } -> tensor<1x128xf32>

    // -------------------------------------------------------------
    // LAYER 2: MatMul (1x128 * 128x64 -> 1x64) + Bias + ReLU
    // -------------------------------------------------------------
    %init_l2 = tensor.empty() : tensor<1x64xf32>
    %fill_l2 = linalg.fill ins(%c0 : f32) outs(%init_l2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %mm2 = linalg.matmul
      ins(%l1_out, %w2 : tensor<1x128xf32>, tensor<128x64xf32>)
      outs(%fill_l2 : tensor<1x64xf32>) -> tensor<1x64xf32>

    // Bias Add & ReLU
    %l2_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm2, %b2 : tensor<1x64xf32>, tensor<1x64xf32>)
      outs(%init_l2 : tensor<1x64xf32>) {
    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):
      %sum = arith.addf %in_val, %bias_val : f32
      %relu = arith.maximumf %sum, %c0 : f32
      linalg.yield %relu : f32
    } -> tensor<1x64xf32>

    // -------------------------------------------------------------
    // LAYER 3: MatMul (1x64 * 64x10 -> 1x10) + Bias (Logits)
    // -------------------------------------------------------------
    %init_l3 = tensor.empty() : tensor<1x10xf32>
    %fill_l3 = linalg.fill ins(%c0 : f32) outs(%init_l3 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %mm3 = linalg.matmul
      ins(%l2_out, %w3 : tensor<1x64xf32>, tensor<64x10xf32>)
      outs(%fill_l3 : tensor<1x10xf32>) -> tensor<1x10xf32>

    // Bias Add (Linear Output Logits)
    %l3_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm3, %b3 : tensor<1x10xf32>, tensor<1x10xf32>)
      outs(%init_l3 : tensor<1x10xf32>) {
    ^bb0(%in_val: f32, %bias_val: f32, %out_val: f32):
      %sum = arith.addf %in_val, %bias_val : f32
      linalg.yield %sum : f32
    } -> tensor<1x10xf32>

    return %l3_out : tensor<1x10xf32>
  }
}
