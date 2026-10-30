// RUN: vision-opt %s --vision-vectorize | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_vectorize_add
  // CHECK: vector.transfer_read
  // CHECK: vector.transfer_read
  // CHECK: arith.addf
  // CHECK: vector.transfer_write
  func.func @test_vectorize_add(
      %lhs: memref<128xf32>,
      %rhs: memref<128xf32>,
      %out: memref<128xf32>) {

    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%lhs, %rhs : memref<128xf32>, memref<128xf32>)
      outs(%out : memref<128xf32>) {
    ^bb0(%in1: f32, %in2: f32, %res: f32):
      %add = arith.addf %in1, %in2 : f32
      linalg.yield %add : f32
    }

    return
  }
}
