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

/// Operation pass that vectorizes structured Linalg operations into Vector Dialect ops.
struct VisionVectorizationPass
    : public PassWrapper<VisionVectorizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VisionVectorizationPass)

  VisionVectorizationPass() = default;
  VisionVectorizationPass(int64_t width) : vectorWidth(width) {}

  StringRef getArgument() const final { return "vision-vectorize"; }
  StringRef getDescription() const final {
    return "Vectorizes Linalg operations into Vector Dialect for AVX2 (256-bit SIMD)";
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

    // Collect Linalg operations to vectorize
    SmallVector<linalg::LinalgOp> linalgOps;
    func.walk([&](linalg::LinalgOp op) {
      linalgOps.push_back(op);
    });

    for (linalg::LinalgOp op : linalgOps) {
      rewriter.setInsertionPoint(op);
      (void)linalg::vectorize(rewriter, op);
    }

    // Optimize resulting vector operations (drop unit dims, canonicalize transfer ops)
    RewritePatternSet patterns(context);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    GreedyRewriteConfig config;
    config.maxIterations = 5;
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), config);
  }

private:
  int64_t vectorWidth = 8; // 8 x float32 elements per 256-bit YMM register
};

} // namespace

std::unique_ptr<Pass> createVisionVectorizationPass() {
  return std::make_unique<VisionVectorizationPass>();
}

std::unique_ptr<Pass> createVisionVectorizationPass(int64_t vectorWidth) {
  return std::make_unique<VisionVectorizationPass>(vectorWidth);
}

} // namespace mlir::vision
