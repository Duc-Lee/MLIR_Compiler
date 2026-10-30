// mau IR con toan tu Conv2D + Bias Add + ReLU (TOSA Dialect)
// dau vao cho custom lowering pipeline
module {
  func.func @conv2d_bias_relu(
      %input: tensor<1x56x56x64xf32>,
      %weight: tensor<64x3x3x64xf32>,
      %zero_bias: tensor<64xf32>,
      %bias: tensor<64xf32>) -> tensor<1x56x56x64xf32> {

    // toan tu conv2d voi zero bias
    %0 = "tosa.conv2d"(%input, %weight, %zero_bias) {
      pad = array<i64: 1, 1, 1, 1>,
      stride = array<i64: 1, 1>,
      dilation = array<i64: 1, 1>
    } : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

    // cong bias (hoac folded batchnorm parameters)
    %1 = "tosa.add"(%0, %bias) : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

    // ham kich hoat relu (clamp gia tri tu 0.0 den max)
    %2 = "tosa.clamp"(%1) {
      min_int = 0 : i64,
      max_int = 0 : i64,
      min_fp = 0.0 : f32,
      max_fp = 3.40282347e+38 : f32
    } : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

    return %2 : tensor<1x56x56x64xf32>
  }
}
