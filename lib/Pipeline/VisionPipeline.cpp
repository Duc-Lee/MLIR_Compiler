#include "VisionCompiler/Pipeline/VisionPipeline.h"
#include "VisionCompiler/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::vision {

void buildVisionLoweringPipeline(OpPassManager &pm,
                                 const VisionPipelineOptions &options) {
  // Phase 1: High-level TOSA dialect lowering to Linalg on tensors
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalgNamed());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToTensor());

  // Canonicalization and common subexpression elimination
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Phase 2: Operator Fusion (Conv2D/MatMul + Bias/BN + ReLU)
  if (options.runFusion) {
    pm.addNestedPass<func::FuncOp>(createVisionFusionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // Phase 3: One-shot bufferization (lowers immutable Tensors to allocated MemRefs)
  bufferization::OneShotBufferizationOptions bufOptions;
  bufOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOptions));

  // Phase 4: Loop Tiling for L1/L2 cache locality on memrefs
  if (options.runTiling) {
    pm.addNestedPass<func::FuncOp>(createVisionTilingPass(options.tilingConfig));
    pm.addPass(createCanonicalizerPass());
  }

  // Phase 5: Vectorization down to Vector Dialect (AVX2 256-bit SIMD)
  if (options.runVectorization) {
    pm.addNestedPass<func::FuncOp>(createVisionVectorizationPass(
        options.hwConfig.vectorWidthBits / 32));
    pm.addPass(createCanonicalizerPass());
  }

  // Phase 6: Progressive lowering to LLVM Dialect
  if (options.lowerToLLVM) {
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createConvertVectorToSCFPass());
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
  }
}

void registerVisionPipeline() {
  PassPipelineRegistration<>(
      "vision-pipeline",
      "End-to-end lowering pipeline for vision models with Fusion, Tiling, and Vectorization",
      [](OpPassManager &pm) {
        VisionPipelineOptions defaultOptions;
        buildVisionLoweringPipeline(pm, defaultOptions);
      });
}

} // namespace mlir::vision
