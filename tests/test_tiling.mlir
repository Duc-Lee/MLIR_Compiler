// RUN: vision-opt %s --vision-tiling | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_matmul_tiling
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       linalg.matmul
  func.func @test_matmul_tiling(
      %A: tensor<128x128xf32>,
      %B: tensor<128x128xf32>,
      %C: tensor<128x128xf32>) -> tensor<128x128xf32> {

    %res = linalg.matmul
      ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>

    return %res : tensor<128x128xf32>
  }
}
