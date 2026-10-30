#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace vision {

/// Creates an operation pass that vectorizes Linalg operations into Vector Dialect ops.
std::unique_ptr<Pass> createVisionVectorizationPass();
std::unique_ptr<Pass> createVisionVectorizationPass(int64_t vectorWidth);

} // namespace vision
} // namespace mlir
