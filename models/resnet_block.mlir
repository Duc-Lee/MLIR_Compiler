// mau IR bieu dien 1 Residual Block (ResNet-18) co Skip-connection
// input -> Conv1 -> ReLU -> Conv2 -> Add(x) -> ReLU
module {
  func.func @resnet_basic_block(
      %input: tensor<1x56x56x64xf32>,
      %w1: tensor<64x3x3x64xf32>,
      %b1: tensor<64xf32>,
      %w2: tensor<64x3x3x64xf32>,
      %b2: tensor<64xf32>) -> tensor<1x56x56x64xf32> {

    // lop conv1
    %c1 = "tosa.conv2d"(%input, %w1, %b1) {
      pad = array<i64: 1, 1, 1, 1>,
      stride = array<i64: 1, 1>,
      dilation = array<i64: 1, 1>
    } : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

    // relu1
    %r1 = "tosa.clamp"(%c1) {
      min_int = 0 : i64,
      max_int = 0 : i64,
      min_fp = 0.0 : f32,
      max_fp = 3.40282347e+38 : f32
    } : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

    // lop conv2
    %c2 = "tosa.conv2d"(%r1, %w2, %b2) {
      pad = array<i64: 1, 1, 1, 1>,
      stride = array<i64: 1, 1>,
      dilation = array<i64: 1, 1>
    } : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

    // skip-connection: cong duong tat voi tensor dau vao ban dau
    %residual = "tosa.add"(%c2, %input) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

    // relu2 ket thuc block
    %out = "tosa.clamp"(%residual) {
      min_int = 0 : i64,
      max_int = 0 : i64,
      min_fp = 0.0 : f32,
      max_fp = 3.40282347e+38 : f32
    } : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

    return %out : tensor<1x56x56x64xf32>
  }
}
