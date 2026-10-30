#pragma once

#include "VisionCompiler/Common/CompilerTypes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::vision {

/// Configuration options for the complete Vision Lowering Pipeline.
struct VisionPipelineOptions {
  TilingConfig tilingConfig;
  TargetHardwareConfig hwConfig;
  bool runFusion = true;
  bool runTiling = true;
  bool runVectorization = true;
  bool lowerToLLVM = true;
  bool emitC = false;
};

/// Builds the end-to-end progressive lowering pipeline from high-level dialects to LLVM.
void buildVisionLoweringPipeline(OpPassManager &pm,
                                 const VisionPipelineOptions &options);

/// Registers the "vision-pipeline" string name with the MLIR PassPipelineRegistry.
void registerVisionPipeline();

} // namespace mlir::vision
