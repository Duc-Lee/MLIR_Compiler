#pragma once

#include "mlir/Pass/Pass.h"
#include "VisionCompiler/Common/CompilerTypes.h"
#include <memory>
#include <vector>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace vision {

// pass chia nho vong lap (tiling) de vua khit L1/L2 cache
std::unique_ptr<Pass> createVisionTilingPass();
std::unique_ptr<Pass> createVisionTilingPass(const TilingConfig &config);

} // namespace vision
} // namespace mlir
