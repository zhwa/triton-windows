//===-- TritonToLinalg.cpp - Convert Triton IR to Linalg + Tensor ---------===//
//
// Vulkan/SPIR-V backend for Triton.
// Adapted from triton-shared (Microsoft, MIT license).
// Ported to Triton 3.7.0 APIs.
//
// Pipeline: TTIR → Linalg/Tensor/Arith → MemRef → SPIR-V → Vulkan
//
// 16 converters:
//   Core: splat, make_range, broadcast, expand_dims, transpose, reshape,
//         get_program_id, get_num_programs, dot, reduce, bitcast
//   Pointer/Memory: addptr (PtrState), load, store, atomic_rmw
//   Constant: dense splat → linalg.fill
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonToLinalg.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

static SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                             RankedTensorType dst) {
  SmallVector<int64_t> broadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();
  for (size_t i = 0; i < srcShape.size(); i++) {
    if (dstShape[i] != srcShape[i]) {
      assert(srcShape[i] == 1);
      broadcastDims.push_back(i);
    }
  }
  assert(!broadcastDims.empty() && "cannot identify broadcast dimension");
  return broadcastDims;
}

static AffineMap getBroadcastAffineMap(MLIRContext *context,
                                       ArrayRef<int64_t> inputShape,
                                       ArrayRef<int64_t> broadcastToShape) {
  assert(broadcastToShape.size() >= inputShape.size());
  SmallVector<AffineExpr> outExpr;
  size_t diff = broadcastToShape.size() - inputShape.size();
  for (size_t i = 0; i < broadcastToShape.size(); i++) {
    if (i < diff)
      continue;
    size_t j = i - diff;
    if (inputShape[j] == 1) {
      outExpr.push_back(mlir::getAffineConstantExpr(0, context));
    } else {
      outExpr.push_back(mlir::getAffineDimExpr(i, context));
    }
  }
  return AffineMap::get(broadcastToShape.size(), 0, outExpr, context);
}

static Value getTransposedValue(Value source, Location loc,
                                PatternRewriter &rewriter,
                                ArrayRef<int32_t> order = {}) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(sourceRank);
  SmallVector<int64_t> transposedShape(sourceType.getShape());
  if (order.empty()) {
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[sourceRank - 1], perm[sourceRank - 2]);
    std::swap(transposedShape[sourceRank - 1], transposedShape[sourceRank - 2]);
  } else {
    assert(static_cast<int64_t>(order.size()) == sourceRank);
    for (int64_t i = 0; i < sourceRank; ++i) {
      perm[i] = order[i];
      transposedShape[i] = sourceType.getShape()[order[i]];
    }
  }

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];
  return transpose;
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// tt.splat %scalar → tensor.empty + linalg.fill
// For pointer splats, produce an unrealized_conversion_cast that
// AddPtrConverter can trace through to find the base memref.
struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Pointer splat: the type converter maps tensor<N x ptr<T>> to memref<N x T>.
    // We create a memref.cast from the unranked base memref to a ranked one,
    // then a reinterpret_cast to get the right shape. This serves as the
    // "base pointer" for subsequent AddPtrConverter to create views from.
    if (isa<triton::PointerType>(op.getSrc().getType())) {
      Value baseMem = adaptor.getSrc();
      auto ptrType = cast<triton::PointerType>(op.getSrc().getType());
      auto elemType = ptrType.getPointeeType();
      auto resultTensorType = cast<RankedTensorType>(op.getType());
      auto resultShape = resultTensorType.getShape();

      // Cast unranked memref → ranked 1D dynamic memref
      auto ranked1D = MemRefType::get({ShapedType::kDynamic}, elemType);
      Value ranked = rewriter.create<memref::CastOp>(loc, ranked1D, baseMem);

      // Create a reinterpret_cast with zero offset to match expected shape
      auto resultMemType = MemRefType::get(
          resultShape, elemType,
          StridedLayoutAttr::get(rewriter.getContext(),
                                 ShapedType::kDynamic,
                                 SmallVector<int64_t>(resultShape.size(),
                                                      ShapedType::kDynamic)));
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      SmallVector<OpFoldResult> sizes, strides;
      for (auto dim : resultShape) {
        sizes.push_back(rewriter.getIndexAttr(dim));
        strides.push_back(rewriter.getIndexAttr(1));
      }
      Value result = rewriter.create<memref::ReinterpretCastOp>(
          loc, resultMemType, ranked, /*offset=*/zero, sizes, strides);
      rewriter.replaceOp(op, result);
      return success();
    }

    auto opType = cast<TensorType>(op.getType());

    auto init = rewriter.create<tensor::EmptyOp>(loc, opType.getShape(),
                                                 opType.getElementType());
    auto filledTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{adaptor.getSrc()},
                                    ValueRange{init})
            .result();
    rewriter.replaceOp(op, filledTensor);
    return success();
  }
};

// tt.make_range {start, end} → linalg.generic with linalg.index
struct MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<TensorType>(op.getResult().getType());
    auto shape = type.getShape();
    auto elementType = type.getElementType();
    auto context = rewriter.getContext();

    SmallVector<AffineMap> indexingMaps{AffineMap::get(
        1, 0,
        SmallVector<AffineExpr>{mlir::getAffineDimExpr(0, context)}, context)};

    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), ValueRange{}, ValueRange{init}, indexingMaps,
        getNParallelLoopsAttrs(1),
        [&](OpBuilder &b, Location nestedLoc, ValueRange blockArgs) {
          Value index = b.create<linalg::IndexOp>(loc, 0);
          Value res = b.create<arith::IndexCastOp>(loc, elementType, index);
          if (op.getStart() != 0) {
            auto start = rewriter.create<arith::ConstantIntOp>(
                loc, op.getStart(),
                elementType.getIntOrFloatBitWidth());
            res = b.create<arith::AddIOp>(loc, res, start);
          }
          b.create<linalg::YieldOp>(loc, res);
        });
    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

// tt.broadcast → linalg.generic with broadcast affine map
struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    RankedTensorType sourceType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    RankedTensorType resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(getBroadcastAffineMap(
        op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));

    auto init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 elementType);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), ValueRange{adaptor.getSrc()},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &b, Location nestedLoc, ValueRange blockArgs) {
          b.create<linalg::YieldOp>(loc, blockArgs[0]);
        });
    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

// tt.expand_dims → tensor.expand_shape
struct ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto srcRank = cast<RankedTensorType>(src.getType()).getRank();
    auto resType = cast<RankedTensorType>(op->getResultTypes()[0]);

    SmallVector<ReassociationIndices> reassoc;
    int64_t c = 0;
    for (int64_t i = 0; i < srcRank; i++) {
      ReassociationIndices g;
      g.push_back(c++);
      if (op.getAxis() == i) {
        g.push_back(c++);
      } else if (op.getAxis() == i + 1 && i == srcRank - 1) {
        g.push_back(c++);
      }
      reassoc.push_back(g);
    }

    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        op.getLoc(), resType, src, reassoc);
    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
  }
};

// tt.trans → linalg.transpose
struct TransposeConverter : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<int32_t> order(op.getOrder().begin(), op.getOrder().end());
    auto res = getTransposedValue(adaptor.getSrc(), op.getLoc(), rewriter,
                                  order);
    rewriter.replaceOp(op, res);
    return success();
  }
};

// tt.get_program_id → extract from function args
struct GetProgramIDConverter
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern::OpConversionPattern;
  static constexpr uint32_t LAUNCH_GRID_RANK = 3; // X, Y, Z

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = static_cast<uint32_t>(op.getAxis());
    assert(axis < LAUNCH_GRID_RANK);
    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    // pid args are at numArgs-6..numArgs-4 (local_id is last 3)
    auto id = func.getArgument(numArgs - 2 * LAUNCH_GRID_RANK + axis);
    rewriter.replaceOp(op, id);
    return success();
  }
};

// tt.get_num_programs → extract from function args
struct GetNumProgramsConverter
    : public OpConversionPattern<triton::GetNumProgramsOp> {
  using OpConversionPattern::OpConversionPattern;
  static constexpr uint32_t LAUNCH_GRID_RANK = 3;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = static_cast<uint32_t>(op.getAxis());
    assert(axis < LAUNCH_GRID_RANK);
    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    // num_programs args are at numArgs-9..numArgs-7
    auto id = func.getArgument(numArgs - 3 * LAUNCH_GRID_RANK + axis);
    rewriter.replaceOp(op, id);
    return success();
  }
};

// tt.dot → linalg.matmul
struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  static bool isZeroTensor(Value v) {
    if (auto splatOp = v.getDefiningOp<triton::SplatOp>()) {
      if (auto constOp =
              splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
        if (auto val = dyn_cast<FloatAttr>(constOp.getValue()))
          return val.getValueAsDouble() == 0.;
        if (auto val = dyn_cast<IntegerAttr>(constOp.getValue()))
          return val.getValue() == 0;
      }
    }
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat()) {
          if (denseAttr.getElementType().isInteger())
            return denseAttr.getSplatValue<APInt>().isZero();
          return denseAttr.getSplatValue<APFloat>().isZero();
        }
      }
    }
    return false;
  }

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto a = op.getA();
    auto b = op.getB();
    auto c = op.getC();

    auto dstType = cast<RankedTensorType>(op.getType());
    auto elementType = dstType.getElementType();
    bool isInt = elementType.isInteger();
    bool skipC = isZeroTensor(c);

    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elementType);

    TypedAttr zeroAttr =
        isInt ? static_cast<TypedAttr>(rewriter.getIntegerAttr(elementType, 0))
              : static_cast<TypedAttr>(rewriter.getFloatAttr(elementType, 0));
    auto zero =
        rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto zeroes =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{init})
            .result();

    auto res = rewriter
                   .create<linalg::MatmulOp>(loc, ValueRange{a, b},
                                             ValueRange{zeroes})
                   .getResult(0);

    if (!skipC) {
      if (isInt)
        res = rewriter.create<arith::AddIOp>(loc, c, res);
      else
        res = rewriter.create<arith::AddFOp>(loc, c, res);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

// tt.reduce → linalg.reduce
// Clones the combiner region from Triton's reduce into linalg.reduce.
struct ReduceConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  // Determine initial value for a reduction based on the combiner op.
  static Value getInitValue(Operation *combiner, Type elementType,
                            Location loc, OpBuilder &builder) {
    return TypeSwitch<Operation *, Value>(combiner)
        .Case<arith::AddFOp, arith::AddIOp>([&](auto) {
          return builder.create<arith::ConstantOp>(
              loc, cast<TypedAttr>(builder.getZeroAttr(elementType)));
        })
        .Case<arith::MulFOp>([&](auto) {
          return builder.create<arith::ConstantOp>(
              loc, cast<TypedAttr>(builder.getFloatAttr(elementType, 1.0)));
        })
        .Case<arith::MulIOp>([&](auto) {
          return builder.create<arith::ConstantOp>(
              loc, cast<TypedAttr>(builder.getIntegerAttr(elementType, 1)));
        })
        .Case<arith::MaximumFOp, arith::MaxSIOp>([&](auto) {
          if (elementType.isInteger()) {
            auto intType = cast<IntegerType>(elementType);
            auto minVal = APInt::getSignedMinValue(intType.getWidth());
            return builder.create<arith::ConstantOp>(
                loc, cast<TypedAttr>(builder.getIntegerAttr(elementType, minVal)));
          }
          return builder.create<arith::ConstantOp>(
              loc,
              cast<TypedAttr>(builder.getFloatAttr(
                  elementType,
                  APFloat::getInf(
                      cast<FloatType>(elementType).getFloatSemantics(),
                      /*Negative=*/true))));
        })
        .Case<arith::MinimumFOp, arith::MinSIOp>([&](auto) {
          if (elementType.isInteger()) {
            auto intType = cast<IntegerType>(elementType);
            auto maxVal = APInt::getSignedMaxValue(intType.getWidth());
            return builder.create<arith::ConstantOp>(
                loc, cast<TypedAttr>(builder.getIntegerAttr(elementType, maxVal)));
          }
          return builder.create<arith::ConstantOp>(
              loc,
              cast<TypedAttr>(builder.getFloatAttr(
                  elementType,
                  APFloat::getInf(
                      cast<FloatType>(elementType).getFloatSemantics(),
                      /*Negative=*/false))));
        })
        .Case<arith::AndIOp>([&](auto) {
          auto intType = cast<IntegerType>(elementType);
          auto allOnes = APInt::getAllOnes(intType.getWidth());
          return builder.create<arith::ConstantOp>(
              loc, cast<TypedAttr>(builder.getIntegerAttr(elementType, allOnes)));
        })
        .Case<arith::OrIOp, arith::XOrIOp>([&](auto) {
          return builder.create<arith::ConstantOp>(
              loc, cast<TypedAttr>(builder.getZeroAttr(elementType)));
        })
        .Default([&](Operation *op) -> Value {
          op->emitWarning("Unknown reduce combiner — using zero identity. "
                          "This may produce incorrect results for min/max or "
                          "custom combiners.");
          return builder.create<arith::ConstantOp>(
              loc, cast<TypedAttr>(builder.getZeroAttr(elementType)));
        });
  }

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto axis = op.getAxis();
    auto srcs = adaptor.getSrcs();

    // Find the single combiner op to determine init values
    auto *combiner = op.getSingleCombiner();

    // Create init values (identity elements for the reduction)
    SmallVector<Value> initValues;
    for (auto src : srcs) {
      auto srcType = cast<RankedTensorType>(src.getType());
      auto elemType = srcType.getElementType();

      // Compute result shape (drop the reduction axis)
      SmallVector<int64_t> resultShape;
      for (int64_t i = 0; i < srcType.getRank(); i++) {
        if (i != axis)
          resultShape.push_back(srcType.getShape()[i]);
      }

      Value init;
      if (resultShape.empty()) {
        // Scalar result → 0-d tensor init (linalg.reduce requires tensor)
        auto initScalar = getInitValue(combiner, elemType, loc, rewriter);
        auto emptyTensor =
            rewriter.create<tensor::EmptyOp>(loc, resultShape, elemType);
        init = rewriter
                   .create<linalg::FillOp>(loc, ValueRange{initScalar},
                                           ValueRange{emptyTensor})
                   .result();
      } else {
        auto emptyTensor =
            rewriter.create<tensor::EmptyOp>(loc, resultShape, elemType);
        auto initVal = getInitValue(combiner, elemType, loc, rewriter);
        init = rewriter
                   .create<linalg::FillOp>(loc, ValueRange{initVal},
                                           ValueRange{emptyTensor})
                   .result();
      }
      initValues.push_back(init);
    }

    // Create linalg.reduce with body builder
    SmallVector<int64_t> dimensions{axis};
    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, srcs, initValues, dimensions,
        [&](OpBuilder &b, Location nestedLoc, ValueRange blockArgs) {
          // Clone the combiner body from tt.reduce into linalg.reduce
          auto &srcBlock = op.getCombineOp().front();
          IRMapping mapping;
          for (auto [srcArg, dstArg] :
               llvm::zip(srcBlock.getArguments(), blockArgs))
            mapping.map(srcArg, dstArg);

          for (auto &srcOp : srcBlock.without_terminator())
            b.clone(srcOp, mapping);

          // Map tt.reduce.return operands → linalg.yield
          auto *terminator = srcBlock.getTerminator();
          SmallVector<Value> yieldValues;
          for (auto operand : terminator->getOperands())
            yieldValues.push_back(mapping.lookupOrDefault(operand));
          b.create<linalg::YieldOp>(nestedLoc, yieldValues);
        });

    // If reducing to scalar (1D→scalar), linalg.reduce produces
    // tensor<f32> (0-d tensor) but tt.reduce expects bare f32.
    // Extract scalars from 0-d tensor results.
    SmallVector<Value> results;
    for (auto result : reduceOp->getResults()) {
      if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
        if (tensorType.getRank() == 0) {
          auto scalar = rewriter.create<tensor::ExtractOp>(
              loc, result, ValueRange{});
          results.push_back(scalar);
          continue;
        }
      }
      results.push_back(result);
    }
    rewriter.replaceOp(op, results);

    return success();
  }
};

// tt.bitcast → arith.bitcast
struct BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, op.getType(),
                                                  adaptor.getSrc());
    return success();
  }
};

// tt.reshape → tensor.reshape
struct ReshapeConverter : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto src = adaptor.getSrc();
    auto srcType = cast<RankedTensorType>(src.getType());

    // Try expand_shape/collapse_shape first (more efficient, no runtime shape)
    auto reassoc =
        getReassociationIndicesForReshape(srcType, resultType);
    if (reassoc.has_value()) {
      if (srcType.getRank() < resultType.getRank()) {
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, resultType, src, *reassoc);
      } else {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, resultType, src, *reassoc);
      }
      return success();
    }

    // Fallback: use tensor.reshape with dynamic shape
    SmallVector<Value> shapeSizes;
    for (auto dim : resultType.getShape()) {
      shapeSizes.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dim));
    }
    auto shapeType =
        RankedTensorType::get({resultType.getRank()}, rewriter.getIndexType());
    auto shape =
        rewriter.create<tensor::FromElementsOp>(loc, shapeType, shapeSizes);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultType, src, shape);
    return success();
  }
};

// Dense constant tensor → linalg.fill (for splat constants)
struct DenseConstantConverter
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return failure();

    auto elementType = denseAttr.getElementType();
    if (!isa<FloatType, IntegerType>(elementType))
      return failure();

    auto loc = op.getLoc();
    auto splatAttr = denseAttr.getSplatValue<Attribute>();
    Value scalar = rewriter.create<arith::ConstantOp>(
        loc, cast<TypedAttr>(splatAttr));

    auto init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 elementType);
    auto filled = rewriter
                      .create<linalg::FillOp>(loc, ValueRange{scalar},
                                              ValueRange{init})
                      .result();
    rewriter.replaceOp(op, filled);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pointer / Memory Converters
//
// Simplified pointer analysis: handles flat pointer tensors where
//   ptr_tensor = tt.splat(base_ptr) + tt.addptr(ptr_tensor, offset_tensor)
// Produces memref.reinterpret_cast from the base pointer + offset.
//
// Supported:
//   - Unmasked loads/stores (direct memref.copy / materialize_in_destination)
//   - Masked loads/stores with continuous masks (subview-based)
// NOT handled (future):
//   - Wraparound/modulo pointers
//   - Complex loop-carried pointer state
//===----------------------------------------------------------------------===//

/// Simplified pointer state: tracks base memref + per-dimension offset/size/stride.
struct PtrState {
  Value source;                        // Base memref (from type-converted ptr)
  SmallVector<OpFoldResult> offsets;   // Per-dimension offsets
  SmallVector<OpFoldResult> sizes;     // Per-dimension sizes
  SmallVector<OpFoldResult> strides;   // Per-dimension strides
  Value scalar;                        // For scalar pointer case

  int64_t getRank() const { return sizes.size(); }
  bool isEmpty() const { return sizes.empty() && !scalar; }

  /// Combine two states (pointer + offset) via addition.
  void addState(const PtrState &lhs, const PtrState &rhs,
                Location loc, OpBuilder &builder) {
    source = lhs.source ? lhs.source : rhs.source;

    if (lhs.scalar && rhs.scalar) {
      scalar = builder.create<arith::AddIOp>(loc, lhs.scalar, rhs.scalar);
    } else {
      scalar = lhs.scalar ? lhs.scalar : rhs.scalar;
    }

    for (size_t i = 0; i < lhs.sizes.size(); i++) {
      // Add offsets
      auto lhsOff = lhs.offsets[i], rhsOff = rhs.offsets[i];
      auto lhsConst = dyn_cast_if_present<Attribute>(lhsOff);
      auto rhsConst = dyn_cast_if_present<Attribute>(rhsOff);
      if (lhsConst && rhsConst) {
        auto lv = cast<IntegerAttr>(lhsConst).getInt();
        auto rv = cast<IntegerAttr>(rhsConst).getInt();
        offsets.push_back(builder.getIndexAttr(lv + rv));
      } else {
        Value lv = lhsConst ? builder.create<arith::ConstantIndexOp>(
                                  loc, cast<IntegerAttr>(lhsConst).getInt())
                            : cast<Value>(lhsOff);
        Value rv = rhsConst ? builder.create<arith::ConstantIndexOp>(
                                  loc, cast<IntegerAttr>(rhsConst).getInt())
                            : cast<Value>(rhsOff);
        offsets.push_back(builder.create<arith::AddIOp>(loc, lv, rv).getResult());
      }

      // Add strides
      auto lhsStr = lhs.strides[i], rhsStr = rhs.strides[i];
      auto lsConst = dyn_cast_if_present<Attribute>(lhsStr);
      auto rsConst = dyn_cast_if_present<Attribute>(rhsStr);
      if (lsConst && rsConst) {
        strides.push_back(builder.getIndexAttr(
            cast<IntegerAttr>(lsConst).getInt() +
            cast<IntegerAttr>(rsConst).getInt()));
      } else {
        strides.push_back(lhs.strides[i]); // Take non-zero one
      }

      sizes.push_back(lhs.sizes[i]);
    }
  }

  /// Create a memref.reinterpret_cast from the base source + accumulated offset.
  Value createCastOp(ArrayRef<int64_t> resultShape, Location loc,
                     OpBuilder &builder) const {
    // Accumulate scalar offset
    OpFoldResult targetOffset = builder.getIndexAttr(0);
    for (auto o : offsets) {
      auto oConst = dyn_cast_if_present<Attribute>(o);
      auto tConst = dyn_cast_if_present<Attribute>(targetOffset);
      if (oConst && tConst) {
        targetOffset = builder.getIndexAttr(
            cast<IntegerAttr>(tConst).getInt() +
            cast<IntegerAttr>(oConst).getInt());
      } else {
        Value tv = tConst ? builder.create<arith::ConstantIndexOp>(
                                loc, cast<IntegerAttr>(tConst).getInt())
                          : cast<Value>(targetOffset);
        Value ov = oConst ? builder.create<arith::ConstantIndexOp>(
                                loc, cast<IntegerAttr>(oConst).getInt())
                          : cast<Value>(o);
        targetOffset = builder.create<arith::AddIOp>(loc, tv, ov).getResult();
      }
    }

    // Ensure source is a ranked memref (may be unranked from type conversion)
    Value rankedSource = source;
    Type sourceType = source.getType();
    Type elemType;
    if (auto unrankedType = dyn_cast<UnrankedMemRefType>(sourceType)) {
      elemType = unrankedType.getElementType();
      auto ranked1D = MemRefType::get({ShapedType::kDynamic}, elemType);
      rankedSource = builder.create<memref::CastOp>(loc, ranked1D, source);
    } else if (auto memrefType = dyn_cast<MemRefType>(sourceType)) {
      elemType = memrefType.getElementType();
    } else {
      // Shouldn't reach here — source should always be a memref
      llvm::errs() << "PtrState::createCastOp: unexpected source type: "
                   << sourceType << "\n";
      return Value();
    }

    // Build result memref type with dynamic offset + strides
    auto resultType = MemRefType::get(
        resultShape, elemType,
        StridedLayoutAttr::get(builder.getContext(),
                               ShapedType::kDynamic,
                               SmallVector<int64_t>(resultShape.size(),
                                                    ShapedType::kDynamic)));

    return builder.create<memref::ReinterpretCastOp>(
        loc, resultType, rankedSource, targetOffset, sizes, strides);
  }
};

/// Walk the def-chain of a Value to build PtrState.
static void visitOperand(Value operand, PtrState &state, Location loc,
                          OpBuilder &builder,
                          const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  // Check if already analyzed
  auto it = knownPtrs.find(operand);
  if (it != knownPtrs.end()) {
    state = it->second;
    return;
  }

  auto *defOp = operand.getDefiningOp();
  if (!defOp) {
    // Block argument (type-converted ptr → memref)
    if (isa<MemRefType>(operand.getType()) ||
        isa<UnrankedMemRefType>(operand.getType())) {
      state.source = operand;
    }
    return;
  }

  TypeSwitch<Operation *>(defOp)
      .Case<triton::SplatOp>([&](auto splatOp) {
        auto src = splatOp.getSrc();
        PtrState srcState;
        visitOperand(src, srcState, loc, builder, knownPtrs);
        state.source = srcState.source;
        if (!state.source && isa<MemRefType>(src.getType()))
          state.source = src;

        auto resultType = cast<RankedTensorType>(splatOp.getType());
        auto elemType = resultType.getElementType();
        bool isIntSplat = elemType.isInteger(32) || elemType.isInteger(64) ||
                          elemType.isIndex();

        for (auto dim : resultType.getShape()) {
          // If splatting an integer scalar (offset), use it as the base offset
          // so that pid-dependent offsets propagate through addState
          if (isIntSplat && srcState.scalar) {
            // Propagate runtime scalar as dynamic offset (enables multi-block
            // and 2D pointer patterns where pid-dependent values flow through)
            Value idxVal = builder.create<arith::IndexCastOp>(
                loc, builder.getIndexType(), srcState.scalar);
            state.offsets.push_back(idxVal);
          } else {
            state.offsets.push_back(builder.getIndexAttr(0));
          }
          state.sizes.push_back(builder.getIndexAttr(dim));
          state.strides.push_back(builder.getIndexAttr(0));
        }
      })
      .Case<triton::AddPtrOp>([&](auto addptrOp) {
        PtrState ptrState, offsetState;
        visitOperand(addptrOp.getPtr(), ptrState, loc, builder, knownPtrs);
        visitOperand(addptrOp.getOffset(), offsetState, loc, builder, knownPtrs);

        if (ptrState.getRank() == 1 && offsetState.getRank() == 0 &&
            offsetState.scalar) {
          offsetState.sizes.push_back(builder.getIndexAttr(1));
          offsetState.offsets.push_back(offsetState.scalar);
          offsetState.strides.push_back(builder.getIndexAttr(0));
        }

        if (ptrState.getRank() > 0 && offsetState.getRank() > 0)
          state.addState(ptrState, offsetState, loc, builder);
        else
          state = ptrState;
      })
      .Case<triton::MakeRangeOp>([&](auto rangeOp) {
        auto start = rangeOp.getStart();
        auto end = rangeOp.getEnd();
        state.offsets.push_back(builder.getIndexAttr(start));
        state.sizes.push_back(builder.getIndexAttr(end - start));
        state.strides.push_back(builder.getIndexAttr(1));
      })
      .Case<arith::AddIOp>([&](auto addOp) {
        PtrState lhs, rhs;
        visitOperand(addOp.getLhs(), lhs, loc, builder, knownPtrs);
        visitOperand(addOp.getRhs(), rhs, loc, builder, knownPtrs);

        if (lhs.getRank() > 0 && rhs.getRank() > 0) {
          state.addState(lhs, rhs, loc, builder);
        } else if (lhs.scalar && rhs.scalar) {
          state.scalar =
              builder.create<arith::AddIOp>(loc, lhs.scalar, rhs.scalar);
        } else {
          // One scalar, one tensor — add scalar as base offset to tensor state
          auto &tensorState = lhs.getRank() > 0 ? lhs : rhs;
          auto &scalarState = lhs.getRank() > 0 ? rhs : lhs;
          state = tensorState;
          if (scalarState.scalar) {
            // Convert scalar i32 to index and add to the tensor's first offset
            Value scalarIdx = builder.create<arith::IndexCastOp>(
                loc, builder.getIndexType(), scalarState.scalar);
            if (!state.offsets.empty()) {
              auto existingOff = state.offsets[0];
              auto oConst = dyn_cast_if_present<Attribute>(existingOff);
              if (oConst) {
                Value ov = builder.create<arith::ConstantIndexOp>(
                    loc, cast<IntegerAttr>(oConst).getInt());
                state.offsets[0] =
                    builder.create<arith::AddIOp>(loc, ov, scalarIdx)
                        .getResult();
              } else {
                state.offsets[0] =
                    builder.create<arith::AddIOp>(loc, cast<Value>(existingOff),
                                                  scalarIdx)
                        .getResult();
              }
            } else {
              state.offsets.push_back(scalarIdx);
            }
          }
        }
      })
      .Case<arith::MulIOp>([&](auto mulOp) {
        PtrState lhs, rhs;
        visitOperand(mulOp.getLhs(), lhs, loc, builder, knownPtrs);
        visitOperand(mulOp.getRhs(), rhs, loc, builder, knownPtrs);

        if (lhs.scalar && rhs.scalar) {
          state.scalar =
              builder.create<arith::MulIOp>(loc, lhs.scalar, rhs.scalar);
        }
      })
      .Case<arith::ExtSIOp>([&](auto extOp) {
        visitOperand(extOp.getIn(), state, loc, builder, knownPtrs);
      })
      .Case<triton::ExpandDimsOp>([&](auto expandOp) {
        PtrState srcState;
        visitOperand(expandOp.getSrc(), srcState, loc, builder, knownPtrs);
        state.source = srcState.source;
        auto axis = expandOp.getAxis();
        for (int64_t i = 0; i < srcState.getRank() + 1; i++) {
          if (i == axis) {
            state.offsets.push_back(builder.getIndexAttr(0));
            state.sizes.push_back(builder.getIndexAttr(1));
            state.strides.push_back(builder.getIndexAttr(0));
          } else {
            auto j = i < axis ? i : i - 1;
            state.offsets.push_back(srcState.offsets[j]);
            state.sizes.push_back(srcState.sizes[j]);
            state.strides.push_back(srcState.strides[j]);
          }
        }
      })
      .Case<triton::BroadcastOp>([&](auto broadcastOp) {
        PtrState srcState;
        visitOperand(broadcastOp.getSrc(), srcState, loc, builder, knownPtrs);
        state.source = srcState.source;
        auto resultType = cast<RankedTensorType>(broadcastOp.getType());
        for (int64_t i = 0; i < srcState.getRank(); i++) {
          state.offsets.push_back(srcState.offsets[i]);
          state.sizes.push_back(builder.getIndexAttr(resultType.getShape()[i]));
          state.strides.push_back(srcState.strides[i]);
        }
      })
      .Default([&](Operation *op) {
        // For unrecognized ops producing scalars, use the value directly
        if (!isa<ShapedType>(operand.getType())) {
          if (operand.getType().isIntOrIndex()) {
            if (operand.getType().isIndex()) {
              state.scalar = operand;
            } else {
              state.scalar = builder.create<arith::IndexCastOp>(
                  loc, builder.getIndexType(), operand);
            }
          }
        }
      });
}

// tt.addptr → memref.reinterpret_cast
//
// The AddPtrConverter walks the ORIGINAL def chain to understand pointer
// structure (offsets, sizes, strides from splat+range+add patterns), then
// uses the CONVERTED base memref (from adaptor) for the reinterpret_cast.
struct AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;

    // Walk the original def chain to understand pointer structure
    PtrState state;
    visitOperand(op, state, loc, rewriter, knownPtrs);

    // Override source with the type-converted ptr from the adaptor.
    // After SplatConverter runs on pointer splats, adaptor.getPtr() is the
    // base memref (from function arg type conversion).
    Value convertedPtr = adaptor.getPtr();
    if (isa<MemRefType>(convertedPtr.getType()) ||
        isa<UnrankedMemRefType>(convertedPtr.getType())) {
      state.source = convertedPtr;
    } else {
      return rewriter.notifyMatchFailure(
          op, "expected memref from type-converted ptr");
    }

    // For scalar addptr
    if (state.getRank() == 0 && state.scalar) {
      state.sizes.push_back(rewriter.getIndexAttr(1));
      state.offsets.push_back(state.scalar);
      state.strides.push_back(rewriter.getIndexAttr(0));
    }

    if (state.getRank() == 0)
      return rewriter.notifyMatchFailure(op, "cannot determine ptr shape");

    SmallVector<int64_t> resultShape;
    if (auto shapedType = dyn_cast<ShapedType>(op.getResult().getType()))
      resultShape.assign(shapedType.getShape().begin(),
                         shapedType.getShape().end());
    else
      resultShape.push_back(1);

    // Fix zero strides for size-1 dims
    int64_t accumSize = 1;
    for (int i = state.getRank() - 1; i >= 0; i--) {
      auto sizeAttr = dyn_cast_if_present<Attribute>(state.sizes[i]);
      auto strideAttr = dyn_cast_if_present<Attribute>(state.strides[i]);
      if (sizeAttr && strideAttr) {
        auto sz = cast<IntegerAttr>(sizeAttr).getInt();
        auto st = cast<IntegerAttr>(strideAttr).getInt();
        if (sz == 1 && st == 0)
          state.strides[i] = rewriter.getIndexAttr(accumSize);
        accumSize *= sz;
      }
    }

    Value result = state.createCastOp(resultShape, loc, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// tt.load → memref.copy + bufferization.to_tensor
struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto mask = adaptor.getMask();
    auto other = adaptor.getOther();
    auto loc = op.getLoc();

    // Scalar load
    if (!isa<ShapedType>(op.getResult().getType())) {
      Value memPtr = ptr;
      // Handle unranked memref from scalar pointer conversion
      if (isa<UnrankedMemRefType>(ptr.getType())) {
        auto elemType =
            cast<UnrankedMemRefType>(ptr.getType()).getElementType();
        auto ranked1D = MemRefType::get({1}, elemType);
        memPtr = rewriter.create<memref::CastOp>(loc, ranked1D, ptr);
      }
      if (!isa<MemRefType>(memPtr.getType()))
        return rewriter.notifyMatchFailure(op, "expected memref for scalar load");
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      auto loadVal = rewriter.create<affine::AffineLoadOp>(
          loc, memPtr, zeroMap, ValueRange{});
      rewriter.replaceOp(op, loadVal.getResult());
      return success();
    }

    auto memrefType = dyn_cast<MemRefType>(ptr.getType());
    if (!memrefType)
      return rewriter.notifyMatchFailure(op, "expected memref type");

    auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                             memrefType.getElementType());

    // Allocate destination buffer
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(memrefType.getShape(),
                             memrefType.getElementType()));

    if (!mask) {
      // Unmasked load: straight copy
      rewriter.create<memref::CopyOp>(loc, ptr, alloc);
    } else {
      // Masked load: fill with 'other' value first, then conditionally
      // copy from source only where mask is true.
      // Step 1: Fill alloc with 'other' (or zero if no 'other' provided)
      auto elemType = memrefType.getElementType();
      if (other) {
        auto otherVal = other;
        if (auto splatOp = otherVal.getDefiningOp<triton::SplatOp>())
          otherVal = splatOp.getSrc();
        if (isa<ShapedType>(otherVal.getType())) {
          TypedAttr zeroAttr;
          if (isa<FloatType>(elemType))
            zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
          else
            zeroAttr = rewriter.getIntegerAttr(elemType, 0);
          auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
          rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                          ValueRange{alloc});
        } else {
          rewriter.create<linalg::FillOp>(loc, ValueRange{otherVal},
                                          ValueRange{alloc});
        }
      } else {
        TypedAttr zeroAttr;
        if (isa<FloatType>(elemType))
          zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
        else
          zeroAttr = rewriter.getIntegerAttr(elemType, 0);
        auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                        ValueRange{alloc});
      }

      // Step 2: Conditional copy — use linalg.generic with arith.select
      // to copy from source only where mask is true, keeping the 'other'
      // fill value where mask is false.
      // The TypeConverter converts tensor<Nxi1> → memref<Nxi1>, so
      // adaptor.getMask() is already a memref. Cast to the expected type
      // if shapes/element types match but the memref type differs.
      Value maskMemref = mask;
      auto maskMemrefType = MemRefType::get(memrefType.getShape(),
                                            rewriter.getI1Type());
      if (maskMemref.getType() != maskMemrefType)
        maskMemref = rewriter.create<memref::CastOp>(loc, maskMemrefType,
                                                      maskMemref);

      unsigned rank = memrefType.getRank();
      auto ctx = rewriter.getContext();
      SmallVector<AffineMap> maps(
          3, AffineMap::getMultiDimIdentityMap(rank, ctx));
      SmallVector<utils::IteratorType> iterTypes(
          rank, utils::IteratorType::parallel);

      rewriter.create<linalg::GenericOp>(
          loc, /*resultTypes=*/TypeRange{},
          /*inputs=*/ValueRange{maskMemref, ptr},
          /*outputs=*/ValueRange{alloc}, maps, iterTypes,
          [](OpBuilder &b, Location l, ValueRange args) {
            // args[0]: mask (i1), args[1]: source, args[2]: other (from fill)
            auto selected =
                b.create<arith::SelectOp>(l, args[0], args[1], args[2]);
            b.create<linalg::YieldOp>(l, ValueRange{selected});
          });
    }

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

// tt.store → bufferization.materialize_in_destination
struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto val = adaptor.getValue();
    auto mask = adaptor.getMask();
    auto loc = op.getLoc();

    // Scalar store
    if (!isa<ShapedType>(val.getType())) {
      Value memPtr = ptr;
      // Handle unranked memref from scalar pointer conversion
      if (isa<UnrankedMemRefType>(ptr.getType())) {
        auto elemType =
            cast<UnrankedMemRefType>(ptr.getType()).getElementType();
        auto ranked1D = MemRefType::get({1}, elemType);
        memPtr = rewriter.create<memref::CastOp>(loc, ranked1D, ptr);
      }
      if (!isa<MemRefType>(memPtr.getType()))
        return rewriter.notifyMatchFailure(op, "expected memref for scalar store");
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      rewriter.create<affine::AffineStoreOp>(loc, val, memPtr, zeroMap,
                                             ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    if (!mask) {
      // Unmasked store
      auto storeOp =
          rewriter.create<bufferization::MaterializeInDestinationOp>(
              loc, val, ptr);
      storeOp.setWritable(true);
    } else {
      // Masked store: only write elements where mask is true.
      // Use linalg.generic to select between new value and existing
      // destination value based on mask, then materialize the result.
      auto memrefType = dyn_cast<MemRefType>(ptr.getType());
      if (!memrefType)
        return rewriter.notifyMatchFailure(op, "expected memref for masked store");

      // adaptor.getMask() is a memref (TypeConverter converts tensor → memref)
      Value maskMemref = mask;
      auto maskMemrefType = MemRefType::get(memrefType.getShape(),
                                            rewriter.getI1Type());
      if (maskMemref.getType() != maskMemrefType)
        maskMemref = rewriter.create<memref::CastOp>(loc, maskMemrefType,
                                                      maskMemref);

      // adaptor.getValue() may be a tensor or memref depending on conversion
      Value valMemref = val;
      if (isa<RankedTensorType>(val.getType()))
        valMemref = rewriter.create<bufferization::ToMemrefOp>(
            loc, memrefType, val);
      else if (val.getType() != memrefType)
        valMemref = rewriter.create<memref::CastOp>(loc, memrefType, val);

      unsigned rank = memrefType.getRank();
      auto ctx = rewriter.getContext();
      SmallVector<AffineMap> maps(
          3, AffineMap::getMultiDimIdentityMap(rank, ctx));
      SmallVector<utils::IteratorType> iterTypes(
          rank, utils::IteratorType::parallel);

      // In-place update of ptr: where mask=true write val, else keep existing
      rewriter.create<linalg::GenericOp>(
          loc, /*resultTypes=*/TypeRange{},
          /*inputs=*/ValueRange{maskMemref, valMemref},
          /*outputs=*/ValueRange{ptr}, maps, iterTypes,
          [](OpBuilder &b, Location l, ValueRange args) {
            // args[0]: mask (i1), args[1]: new value, args[2]: existing dest
            auto selected =
                b.create<arith::SelectOp>(l, args[0], args[1], args[2]);
            b.create<linalg::YieldOp>(l, ValueRange{selected});
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// tt.atomic_rmw → sequential load-modify-store loop
// In single-threaded OpenCL/Vulkan execution, atomics are regular RMW.
// Handles both:
//   - Scalar ptr: direct load/op/store
//   - Splat ptr tensor (all same location): loop accumulating into base[0]
//   - Offset ptr tensor (addptr): loop with per-element RMW
struct AtomicRMWConverter
    : public OpConversionPattern<triton::AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;

  static Value applyRMW(OpBuilder &b, Location loc, triton::RMWOp kind,
                         Value old, Value val) {
    switch (kind) {
    case triton::RMWOp::FADD:
      return b.create<arith::AddFOp>(loc, old, val);
    case triton::RMWOp::ADD:
      return b.create<arith::AddIOp>(loc, old, val);
    case triton::RMWOp::MAX:
      return b.create<arith::MaxSIOp>(loc, old, val);
    case triton::RMWOp::MIN:
      return b.create<arith::MinSIOp>(loc, old, val);
    case triton::RMWOp::UMAX:
      return b.create<arith::MaxUIOp>(loc, old, val);
    case triton::RMWOp::UMIN:
      return b.create<arith::MinUIOp>(loc, old, val);
    case triton::RMWOp::AND:
      return b.create<arith::AndIOp>(loc, old, val);
    case triton::RMWOp::OR:
      return b.create<arith::OrIOp>(loc, old, val);
    case triton::RMWOp::XOR:
      return b.create<arith::XOrIOp>(loc, old, val);
    case triton::RMWOp::XCHG:
      return val; // exchange — just return new value
    default:
      llvm_unreachable("unsupported RMW operation");
    }
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto rmwKind = op.getAtomicRmwOp();
    auto ptrVal = adaptor.getPtr();
    auto valVal = adaptor.getVal();

    // Determine if result is scalar or tensor
    auto resultType = op.getResult().getType();
    bool isScalar = !isa<RankedTensorType>(resultType);

    if (isScalar) {
      // Scalar atomic: ptr is unranked memref, val is scalar
      auto elemType = resultType;
      auto rankedTy = MemRefType::get({1}, elemType);
      auto ranked = rewriter.create<memref::CastOp>(loc, rankedTy, ptrVal);
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto old = rewriter.create<memref::LoadOp>(loc, ranked,
                                                  ValueRange{zero});
      auto newVal = applyRMW(rewriter, loc, rmwKind, old, valVal);
      rewriter.create<memref::StoreOp>(loc, newVal, ranked,
                                        ValueRange{zero});
      rewriter.replaceOp(op, old);
      return success();
    }

    // Tensor atomic: with typeConverter, adaptor gives us memref operands.
    // ptrVal = memref<NxT> (from tensor<Nx!tt.ptr<T>>)
    // valVal = memref<NxT> (from tensor<NxT>)
    auto valOrigTy = cast<RankedTensorType>(op.getVal().getType());
    auto elemType = valOrigTy.getElementType();
    auto shape = valOrigTy.getShape();
    int64_t n = shape[0];

    // Allocate result buffer to hold old values
    auto resultMemTy = MemRefType::get(shape, elemType);
    auto resultBuf = rewriter.create<memref::AllocOp>(loc, resultMemTy);

    // Determine if ptr is unranked (from splat — all same location)
    bool isSplatPtr = isa<UnrankedMemRefType>(ptrVal.getType());

    Value ptrMemref;
    if (isSplatPtr) {
      // All pointers to same location — cast to memref<1xelemTy>
      auto rankedTy = MemRefType::get({1}, elemType);
      ptrMemref = rewriter.create<memref::CastOp>(loc, rankedTy, ptrVal);
    } else {
      ptrMemref = ptrVal;
    }

    // Build a sequential loop: for i = 0..n
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto ub = rewriter.create<arith::ConstantIndexOp>(loc, n);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Sequential RMW loop using memref load/store (no tensor ops needed)
    rewriter.create<scf::ForOp>(
        loc, zero, ub, step, ValueRange{},
        [&](OpBuilder &b, Location l, Value iv, ValueRange) {
          // Load val[i] from memref
          auto valI = b.create<memref::LoadOp>(l, valVal, ValueRange{iv});

          // Load old value from target
          Value loadIdx = isSplatPtr ? zero : iv;
          auto old = b.create<memref::LoadOp>(l, ptrMemref,
                                              ValueRange{loadIdx});

          // Apply RMW operation
          auto newVal = applyRMW(b, l, rmwKind, old, valI);

          // Store new value to target
          b.create<memref::StoreOp>(l, newVal, ptrMemref,
                                    ValueRange{loadIdx});

          // Save old value into result buffer
          b.create<memref::StoreOp>(l, old, resultBuf, ValueRange{iv});

          b.create<scf::YieldOp>(l, ValueRange{});
        });

    // Convert result memref to tensor for replacement
    auto resultTensorTy = RankedTensorType::get(shape, elemType);
    auto resultTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultTensorTy, resultBuf, /*restrict=*/true, /*writable=*/false);
    rewriter.replaceOp(op, resultTensor.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Populate patterns
//===----------------------------------------------------------------------===//

void mlir::triton::vulkan::populateTritonToLinalgConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank) {
  auto *ctx = patterns.getContext();

  // Core tensor manipulation
  patterns.add<SplatConverter>(ctx);
  patterns.add<MakeRangeConverter>(ctx);
  patterns.add<BroadcastConverter>(ctx);
  patterns.add<ExpandDimsConverter>(ctx);
  patterns.add<TransposeConverter>(ctx);
  patterns.add<ReshapeConverter>(ctx);
  patterns.add<BitcastConverter>(ctx);

  // Program info
  patterns.add<GetProgramIDConverter>(ctx);
  patterns.add<GetNumProgramsConverter>(ctx);

  // Compute
  patterns.add<MatmulConverter>(ctx);
  patterns.add<ReduceConverter>(ctx);

  // Constants
  patterns.add<DenseConstantConverter>(ctx);

  // Pointer / Memory
  patterns.add<AddPtrConverter>(ctx);
  patterns.add<LoadConverter>(ctx);
  patterns.add<StoreConverter>(ctx);

  // Atomics — uses typeConverter for ptr type resolution
  patterns.add<AtomicRMWConverter>(typeConverter, ctx);

  // Elementwise arith/math on tensors → linalg.generic
  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}
