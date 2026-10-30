#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace vision {

// pass vectorize cac phep toan linalg sang vector dialect (AVX2/SIMD)
std::unique_ptr<Pass> createVisionVectorizationPass();
std::unique_ptr<Pass> createVisionVectorizationPass(int64_t vectorWidth);

} // namespace vision
} // namespace mlir
