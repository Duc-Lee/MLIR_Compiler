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

// gop bias add vao tosa.conv2d neu conv2d dang dung bias mac dinh la 0
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

    // kiem tra bias hien tai cua conv co phai toan 0 khong
    Operation *currentBiasOp = convOp.getBias().getDefiningOp();
    if (!currentBiasOp)
      return failure();

    // thay the bias cu bang bias moi tu tosa.add
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

// gop phep toan tosa.clamp (relu) vao ngay sau tosa.conv2d hoac tosa.add
struct FuseTosaReluPattern : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp clampOp,
                                PatternRewriter &rewriter) const override {
    Value input = clampOp.getInput();
    Operation *defOp = input.getDefiningOp();
    if (!defOp || !defOp->hasOneUse())
      return failure();

    // neu la tosa.conv2d hoac tosa.matmul thi danh dau thuoc tinh da fuse relu
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

// gop 2 op linalg.generic lien ke (vi du conv/gemm generic + relu generic)
struct FuseLinalgGenericReluPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // chi xu ly cac op elementwise relu co 1 input va 1 output
    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
      return failure();

    Value input = genericOp.getDpsInputOperand(0)->get();
    auto producerOp = input.getDefiningOp<linalg::GenericOp>();
    if (!producerOp || !producerOp->hasOneUse())
      return failure();

    // kiem tra body cua genericOp co phai phep max voi 0 (relu) khong
    Block *body = genericOp.getBody();
    if (!body || body->empty())
      return failure();

    bool isRelu = false;
    for (auto &op : *body) {
      if (auto maxf = dyn_cast<arith::MaximumFOp>(op)) {
        isRelu = true;
        break;
      }
    }

    if (!isRelu)
      return failure();

    // danh dau producer la da fuse activation de backend sinh code tinh truc tiep
    producerOp->setAttr("fused_activation", rewriter.getStringAttr("relu"));
    rewriter.replaceOp(genericOp, producerOp->getResults());
    return success();
  }
};

// pass thuc thi fusion tren cap ham
struct VisionFusionPass
    : public PassWrapper<VisionFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VisionFusionPass)

  StringRef getArgument() const final { return "vision-fusion"; }
  StringRef getDescription() const final {
    return "Hop nhat toan tu Conv2D + Bias/BatchNorm + ReLU de giam luu luong RAM";
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

    // ap dung greedy pattern rewrite
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
