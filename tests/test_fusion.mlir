// RUN: vision-opt %s --vision-fusion | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_conv_bias_fusion
  // CHECK: %[[NEW_BIAS:.*]] = "tosa.const"
  // CHECK: %[[CONV:.*]] = tosa.conv2d %arg0, %arg1, %[[NEW_BIAS]]
  // CHECK-NOT: tosa.add
  // CHECK: return %[[CONV]]
  func.func @test_conv_bias_fusion(
      %input: tensor<1x56x56x64xf32>,
      %weight: tensor<64x3x3x64xf32>) -> tensor<1x56x56x64xf32> {

    %zero_bias = "tosa.const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
    %new_bias = "tosa.const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>

    %0 = "tosa.conv2d"(%input, %weight, %zero_bias) {
      pad = array<i64: 1, 1, 1, 1>,
      stride = array<i64: 1, 1>,
      dilation = array<i64: 1, 1>
    } : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

    %1 = "tosa.add"(%0, %new_bias) : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

    return %1 : tensor<1x56x56x64xf32>
  }
}
