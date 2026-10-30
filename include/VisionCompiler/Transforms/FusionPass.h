#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace vision {

/// Creates an operation pass that fuses Conv2D/MatMul with bias addition and activation.
std::unique_ptr<Pass> createVisionFusionPass();

/// Populates rewrite patterns for operator fusion (GEMM/Conv epilogue fusion).
void populateVisionFusionPatterns(RewritePatternSet &patterns);

} // namespace vision
} // namespace mlir
