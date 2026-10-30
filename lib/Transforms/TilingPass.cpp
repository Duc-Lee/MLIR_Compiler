#include "VisionCompiler/Transforms/TilingPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::vision {

namespace {

/// Operation pass that tiles structured Linalg operations using SCF loops.
struct VisionTilingPass
    : public PassWrapper<VisionTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VisionTilingPass)

  VisionTilingPass() = default;
  VisionTilingPass(const TilingConfig &cfg) : config(cfg) {}

  StringRef getArgument() const final { return "vision-tiling"; }
  StringRef getDescription() const final {
    return "Applies loop tiling to Linalg operators (Conv2D, GEMM) to fit L1/L2 cache";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect,
                    linalg::LinalgDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    // Collect candidate Linalg operations for loop tiling
    SmallVector<linalg::LinalgOp> candidateOps;
    func.walk([&](linalg::LinalgOp op) {
      if (op->hasAttr("tiled_by_vision"))
        return;
      candidateOps.push_back(op);
    });

    for (linalg::LinalgOp op : candidateOps) {
      rewriter.setInsertionPoint(op);
      unsigned numLoops = op.getNumLoops();
      if (numLoops == 0)
        continue;

      SmallVector<OpFoldResult> tileSizes;
      tileSizes.reserve(numLoops);

      // Configure tile sizes according to operator semantics
      if (isa<linalg::MatmulOp>(op)) {
        // Matmul has 3 iteration dimensions: [M, N, K]
        tileSizes.push_back(rewriter.getIndexAttr(config.tileM));
        tileSizes.push_back(rewriter.getIndexAttr(config.tileN));
        tileSizes.push_back(rewriter.getIndexAttr(config.tileK));
      } else if (isa<linalg::Conv2DNhwcHwcfOp>(op)) {
        // Conv2D NHWC: [N=1, H=tileM, W=tileN, C_out=tileK, ...]
        tileSizes.push_back(rewriter.getIndexAttr(1));
        tileSizes.push_back(rewriter.getIndexAttr(config.tileM));
        tileSizes.push_back(rewriter.getIndexAttr(config.tileN));
        tileSizes.push_back(rewriter.getIndexAttr(config.tileK));
        while (tileSizes.size() < numLoops)
          tileSizes.push_back(rewriter.getIndexAttr(0));
      } else {
        // Generic elementwise or reduce operations: tile outermost dimensions
        for (unsigned i = 0; i < numLoops; ++i) {
          if (i < 2)
            tileSizes.push_back(rewriter.getIndexAttr(config.tileM));
          else
            tileSizes.push_back(rewriter.getIndexAttr(0));
        }
      }

      // Mark operation to prevent recursive tiling
      op->setAttr("tiled_by_vision", rewriter.getBoolAttr(true));

      // Apply TilingUsingInterface
      if (auto tilingInterfaceOp = dyn_cast<TilingInterface>(op.getOperation())) {
        scf::SCFTilingOptions options;
        options.setTileSizes(tileSizes);

        FailureOr<scf::SCFTilingResult> tilingResult =
            scf::tileUsingSCFForOp(rewriter, tilingInterfaceOp, options);

        if (succeeded(tilingResult)) {
          rewriter.replaceOp(op, tilingResult->replacements);
        }
      }
    }
  }

private:
  TilingConfig config;
};

} // namespace

std::unique_ptr<Pass> createVisionTilingPass() {
  return std::make_unique<VisionTilingPass>();
}

std::unique_ptr<Pass> createVisionTilingPass(const TilingConfig &config) {
  return std::make_unique<VisionTilingPass>(config);
}

} // namespace mlir::vision
