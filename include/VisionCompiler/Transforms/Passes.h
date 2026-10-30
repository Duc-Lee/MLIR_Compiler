#pragma once

#include "VisionCompiler/Transforms/FusionPass.h"
#include "VisionCompiler/Transforms/TilingPass.h"
#include "VisionCompiler/Transforms/VectorizationPass.h"

namespace mlir::vision {

/// Registers all VisionCompiler transformation passes with the MLIR PassRegistry.
void registerVisionPasses();

} // namespace mlir::vision
