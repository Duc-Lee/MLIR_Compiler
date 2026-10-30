#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace vision {

// pass hop nhat toan tu: Conv2D + BatchNorm + ReLU/SiLU
std::unique_ptr<Pass> createVisionFusionPass();

// dang ky pattern fusion vao rewrite set
void populateVisionFusionPatterns(RewritePatternSet &patterns);

} // namespace vision
} // namespace mlir
