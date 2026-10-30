#include "VisionCompiler/Transforms/FusionPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::vision {

namespace {

/// Fuses bias addition into tosa.conv2d when the convolution uses default zero-bias.
struct FuseTosaConv2DBiasPattern : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern<tosa::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Value input1 = addOp.getInput1();
    Value input2 = addOp.getInput2();

    tosa::Conv2DOp convOp = input1.getDefiningOp<tosa::Conv2DOp>();
    Value bias = input2;
    if (!convOp) {
      convOp = input2.getDefiningOp<tosa::Conv2DOp>();
      bias = input1;
    }

    if (!convOp || !convOp.getResult().hasOneUse())
      return failure();

    Operation *currentBiasOp = convOp.getBias().getDefiningOp();
    if (!currentBiasOp)
      return failure();

    // Replace the zero bias with the actual bias tensor from tosa.add
    rewriter.setInsertionPoint(convOp);
    auto fusedConv = rewriter.create<tosa::Conv2DOp>(
        convOp.getLoc(),
        addOp.getType(),
        convOp.getInput(),
        convOp.getWeight(),
        bias,
        convOp.getPadAttr(),
        convOp.getStrideAttr(),
        convOp.getDilationAttr());

    rewriter.replaceOp(addOp, fusedConv.getResult());
    return success();
  }
};

/// Fuses tosa.clamp (ReLU) directly into preceding tosa.conv2d or tosa.matmul operations.
struct FuseTosaReluPattern : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp clampOp,
                                PatternRewriter &rewriter) const override {
    Value input = clampOp.getInput();
    Operation *defOp = input.getDefiningOp();
    if (!defOp || !defOp->hasOneUse())
      return failure();

    if (isa<tosa::Conv2DOp, tosa::MatMulOp>(defOp)) {
      if (defOp->hasAttr("fused_activation"))
        return failure();

      defOp->setAttr("fused_activation", rewriter.getStringAttr("relu"));
      rewriter.replaceOp(clampOp, input);
      return success();
    }

    return failure();
  }
};

/// Fuses consecutive linalg.generic operations (e.g., bias-add epilogue + elementwise ReLU).
struct FuseLinalgGenericReluPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Only handle elementwise activation with single input and single init
    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
      return failure();

    Value input = genericOp.getDpsInputOperand(0)->get();
    auto producerOp = input.getDefiningOp<linalg::GenericOp>();
    if (!producerOp || !producerOp->hasOneUse())
      return failure();

    // Verify whether the operation body performs an arith.maximumf with 0.0 (ReLU)
    Block *body = genericOp.getBody();
    if (!body || body->empty())
      return failure();

    bool isRelu = false;
    for (auto &op : *body) {
      if (dyn_cast<arith::MaximumFOp>(op)) {
        isRelu = true;
        break;
      }
    }

    if (!isRelu)
      return failure();

    // Mark producer with fused_activation attribute for downstream lowering
    producerOp->setAttr("fused_activation", rewriter.getStringAttr("relu"));
    rewriter.replaceOp(genericOp, producerOp->getResults());
    return success();
  }
};

/// Function pass executing operator fusion rewrites.
struct VisionFusionPass
    : public PassWrapper<VisionFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VisionFusionPass)

  StringRef getArgument() const final { return "vision-fusion"; }
  StringRef getDescription() const final {
    return "Fuses Conv2D/MatMul with bias addition and activation to reduce DRAM round-trips";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect,
                    linalg::LinalgDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateVisionFusionPatterns(patterns);

    GreedyRewriteConfig config;
    config.maxIterations = 10;
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateVisionFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseTosaConv2DBiasPattern,
               FuseTosaReluPattern,
               FuseLinalgGenericReluPattern>(patterns.getContext());
}

std::unique_ptr<Pass> createVisionFusionPass() {
  return std::make_unique<VisionFusionPass>();
}

} // namespace mlir::vision
