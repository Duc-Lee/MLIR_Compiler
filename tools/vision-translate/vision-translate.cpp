#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

int main(int argc, char **argv) {
  // dang ky toan bo ban dich chuan (MLIR sang LLVM IR, C-runner, ...)
  mlir::registerAllTranslations();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "Vision MLIR Translation Driver\n"));
}
