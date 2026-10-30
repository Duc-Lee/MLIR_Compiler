module {
  func.func @mlp_forward(
      %input: tensor<1x784xf32>,
      %w1: tensor<784x128xf32>,
      %b1: tensor<1x128xf32>,
      %w2: tensor<128x64xf32>,
      %b2: tensor<1x64xf32>,
      %w3: tensor<64x10xf32>,
      %b3: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %c0 = arith.constant 0.000000e+00 : f32

    // Layer 1 Linear
    %init_l1 = tensor.empty() : tensor<1x128xf32>
    %fill_l1 = linalg.fill ins(%c0 : f32) outs(%init_l1 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %mm1 = linalg.matmul
      ins(%input, %w1 : tensor<1x784xf32>, tensor<784x128xf32>)
      outs(%fill_l1 : tensor<1x128xf32>) -> tensor<1x128xf32>

    // Bias Add for Layer 1
    %bias_l1 = linalg.generic {
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
      linalg.yield %sum : f32
    } -> tensor<1x128xf32>

    // Activation: ReLU (relu)
    %relu_relu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias_l1 : tensor<1x128xf32>)
      outs(%init_l1 : tensor<1x128xf32>) {
    ^bb0(%in_val: f32, %out_val: f32):
      %act = arith.maximumf %in_val, %c0 : f32
      linalg.yield %act : f32
    } -> tensor<1x128xf32>

    // Layer 2 Linear
    %init_l2 = tensor.empty() : tensor<1x64xf32>
    %fill_l2 = linalg.fill ins(%c0 : f32) outs(%init_l2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %mm2 = linalg.matmul
      ins(%relu_relu, %w2 : tensor<1x128xf32>, tensor<128x64xf32>)
      outs(%fill_l2 : tensor<1x64xf32>) -> tensor<1x64xf32>

    // Bias Add for Layer 2
    %bias_l2 = linalg.generic {
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
      linalg.yield %sum : f32
    } -> tensor<1x64xf32>

    // Activation: ReLU (relu_1)
    %relu_relu_1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias_l2 : tensor<1x64xf32>)
      outs(%init_l2 : tensor<1x64xf32>) {
    ^bb0(%in_val: f32, %out_val: f32):
      %act = arith.maximumf %in_val, %c0 : f32
      linalg.yield %act : f32
    } -> tensor<1x64xf32>

    // Layer 3 Linear
    %init_l3 = tensor.empty() : tensor<1x10xf32>
    %fill_l3 = linalg.fill ins(%c0 : f32) outs(%init_l3 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %mm3 = linalg.matmul
      ins(%relu_relu_1, %w3 : tensor<1x64xf32>, tensor<64x10xf32>)
      outs(%fill_l3 : tensor<1x10xf32>) -> tensor<1x10xf32>

    // Bias Add for Layer 3
    %bias_l3 = linalg.generic {
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

    return %bias_l3 : tensor<1x10xf32>
  }
}
