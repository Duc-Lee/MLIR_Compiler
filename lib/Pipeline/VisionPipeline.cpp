#include "VisionCompiler/Pipeline/VisionPipeline.h"
#include "VisionCompiler/Transforms/Passes.h"

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
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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
  // giai doan 1: ha tosa sang linalg on tensors neu input la tosa
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalgNamed());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToTensor());

  // lam sach cac phep toan va don gian hoa do thi
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // giai doan 2: pass 1 - operator fusion (conv + bias/bn + relu)
  if (options.runFusion) {
    pm.addNestedPass<func::FuncOp>(createVisionFusionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // giai doan 3: bufferization (chuyen tensor bat bien sang memref vung nho thuc)
  bufferization::OneShotBufferizationOptions bufOptions;
  bufOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOptions));

  // giai doan 4: pass 2 - loop tiling de toi uu L1/L2 cache tren buffers
  if (options.runTiling) {
    pm.addNestedPass<func::FuncOp>(createVisionTilingPass(options.tilingConfig));
    pm.addPass(createCanonicalizerPass());
  }

  // giai doan 5: pass 3 - vectorization xuong vector dialect (AVX2/SIMD)
  if (options.runVectorization) {
    pm.addNestedPass<func::FuncOp>(createVisionVectorizationPass(
        options.hwConfig.vectorWidthBits / 32));
    pm.addPass(createCanonicalizerPass());
  }

  // giai doan 6: ha cap xuong llvm ir neu duoc yeu cau
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
      "End-to-end lowering pipeline cho vision models voi Fusion, Tiling va Vectorization",
      [](OpPassManager &pm) {
        VisionPipelineOptions defaultOptions;
        buildVisionLoweringPipeline(pm, defaultOptions);
      });
}

} // namespace mlir::vision
