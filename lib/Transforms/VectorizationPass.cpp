#include "VisionCompiler/Transforms/VectorizationPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::vision {

namespace {

struct VisionVectorizationPass
    : public PassWrapper<VisionVectorizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VisionVectorizationPass)

  VisionVectorizationPass() = default;
  VisionVectorizationPass(int64_t width) : vectorWidth(width) {}

  StringRef getArgument() const final { return "vision-vectorize"; }
  StringRef getDescription() const final {
    return "Vectorize cac toan tu Linalg thanh Vector Dialect de sinh ma AVX2 (256-bit)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect,
                    linalg::LinalgDialect,
                    arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    // duyet qua cac linalg op va thuc hien vectorize
    SmallVector<linalg::LinalgOp> linalgOps;
    func.walk([&](linalg::LinalgOp op) {
      linalgOps.push_back(op);
    });

    for (linalg::LinalgOp op : linalgOps) {
      rewriter.setInsertionPoint(op);
      // goi vectorize cua linalg de ha thanh vector.transfer_read/write va vector.contract
      (void)linalg::vectorize(rewriter, op);
    }

    // toi uu cac lenh vector vua sinh ra (vi du: rut gon transfer, can bang chuyen vi)
    RewritePatternSet patterns(context);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    GreedyRewriteConfig config;
    config.maxIterations = 5;
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), config);
  }
private:
  int64_t vectorWidth = 8; // 8 phan tu float32 cho thanh ghi YMM 256-bit
};

} // namespace

std::unique_ptr<Pass> createVisionVectorizationPass() {
  return std::make_unique<VisionVectorizationPass>();
}

std::unique_ptr<Pass> createVisionVectorizationPass(int64_t vectorWidth) {
  return std::make_unique<VisionVectorizationPass>(vectorWidth);
}

} // namespace mlir::vision
