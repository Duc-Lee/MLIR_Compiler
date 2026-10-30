#include "VisionCompiler/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::vision {

void registerVisionPasses() {
  // dang ky pass 1: fusion
  registerPass([]() -> std::unique_ptr<Pass> {
    return createVisionFusionPass();
  });

  // dang ky pass 2: tiling
  registerPass([]() -> std::unique_ptr<Pass> {
    return createVisionTilingPass();
  });

  // dang ky pass 3: vectorization
  registerPass([]() -> std::unique_ptr<Pass> {
    return createVisionVectorizationPass();
  });
}

} // namespace mlir::vision
