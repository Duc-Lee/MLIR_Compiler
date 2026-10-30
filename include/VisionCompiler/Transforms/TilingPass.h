#pragma once

#include "VisionCompiler/Common/CompilerTypes.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace vision {

/// Creates an operation pass that tiles Linalg operations for L1/L2 cache locality.
std::unique_ptr<Pass> createVisionTilingPass();
std::unique_ptr<Pass> createVisionTilingPass(const TilingConfig &config);

} // namespace vision
} // namespace mlir
