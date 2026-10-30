#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "VisionCompiler/Pipeline/VisionPipeline.h"
#include "VisionCompiler/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // nap toan bo standard dialects trong mlir
  mlir::registerAllDialects(registry);

  // nap standard passes
  mlir::registerAllPasses();

  // dang ky custom passes va lowering pipeline cua vision compiler
  mlir::vision::registerVisionPasses();
  mlir::vision::registerVisionPipeline();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Vision MLIR Optimizer Driver\n", registry));
}
