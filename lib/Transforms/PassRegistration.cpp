#include "VisionCompiler/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::vision {

void registerVisionPasses() {
  // Register Pass 1: Operator Fusion
  registerPass([]() -> std::unique_ptr<Pass> {
    return createVisionFusionPass();
  });

  // Register Pass 2: Loop Tiling
  registerPass([]() -> std::unique_ptr<Pass> {
    return createVisionTilingPass();
  });

  // Register Pass 3: AVX2 Vectorization
  registerPass([]() -> std::unique_ptr<Pass> {
    return createVisionVectorizationPass();
  });
}

} // namespace mlir::vision
