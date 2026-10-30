add_mlir_library(VisionTransforms
  FusionPass.cpp
  TilingPass.cpp
  VectorizationPass.cpp
  PassRegistration.cpp

  DEPENDS
  MLIRPass
  MLIRIR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRBufferizationDialect
)
