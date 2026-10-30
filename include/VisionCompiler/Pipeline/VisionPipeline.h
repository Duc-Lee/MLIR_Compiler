#pragma once

#include "mlir/Pass/PassManager.h"
#include "VisionCompiler/Common/CompilerTypes.h"

namespace mlir::vision {

// tuy chon cho toan bo pipeline ha cap
struct VisionPipelineOptions {
  TilingConfig tilingConfig;
  TargetHardwareConfig hwConfig;
  bool runFusion = true;
  bool runTiling = true;
  bool runVectorization = true;
  bool lowerToLLVM = true;
  bool emitC = false;
};

// xay dung pipeline ha cap tu high-level IR (TOSA/Linalg) xuong LLVM / EmitC
void buildVisionLoweringPipeline(OpPassManager &pm,
                                 const VisionPipelineOptions &options);

// dang ky pipeline pipeline string voi MLIR PassManager
void registerVisionPipeline();

} // namespace mlir::vision
