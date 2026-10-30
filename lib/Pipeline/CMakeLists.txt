add_mlir_library(VisionPipeline
  VisionPipeline.cpp

  DEPENDS
  VisionTransforms

  LINK_LIBS PUBLIC
  VisionTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTosaToLinalg
  MLIRTosaToTensor
  MLIRLinalgTransforms
  MLIRVectorToLLVM
  MLIRSCFToControlFlow
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMemRefTransforms
)
