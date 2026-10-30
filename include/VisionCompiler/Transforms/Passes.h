#pragma once
#include "VisionCompiler/Transforms/FusionPass.h"
#include "VisionCompiler/Transforms/TilingPass.h"
#include "VisionCompiler/Transforms/VectorizationPass.h"

namespace mlir::vision {

// dang ky toan bo custom pass vao mlir pass registry
void registerVisionPasses();

} // namespace mlir::vision
