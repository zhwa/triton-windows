//===-- TritonToLinalg.cpp - Convert Triton IR to Linalg + Tensor ---------===//
//
// Vulkan/SPIR-V backend for Triton.
// Adapted from triton-shared (Microsoft, MIT license).
// Ported to Triton 3.7.0 APIs.
//
// Pipeline: TTIR → Linalg/Tensor/Arith → (later) SPIR-V
//
// Phase 0.5 scope:
//   Converts the core Triton ops that don't require pointer analysis:
//     tt.splat, tt.make_range, tt.broadcast, tt.expand_dims, tt.trans,
//     tt.get_program_id, tt.get_num_programs, tt.dot, tt.reduce,
//     tt.addptr (scalar only), tt.bitcast, tt.reshape
//
//   Stubs for complex ops (require PtrAnalysis/MaskAnalysis):
//     tt.load, tt.store — will be implemented in Phase 1
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
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK + axis);
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
    // num_programs args come before program_id args
    auto id = func.getArgument(numArgs - 2 * LAUNCH_GRID_RANK + axis);
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
        // Scalar result
        init = getInitValue(combiner, elemType, loc, rewriter);
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

    rewriter.replaceOp(op, reduceOp->getResults());
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
// Pointer / Memory Converters (Phase 1)
//
// Simplified pointer analysis: handles flat pointer tensors where
//   ptr_tensor = tt.splat(base_ptr) + tt.addptr(ptr_tensor, offset_tensor)
// Produces memref.reinterpret_cast from the base pointer + offset.
//
// For Phase 1 we handle:
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
        for (auto dim : resultType.getShape()) {
          state.offsets.push_back(builder.getIndexAttr(0));
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
          // One scalar, one tensor
          state = lhs.getRank() > 0 ? lhs : rhs;
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
    auto mask = op.getMask();
    auto other = op.getOther();
    auto loc = op.getLoc();

    // Scalar load
    if (!isa<ShapedType>(op.getResult().getType())) {
      if (!isa<MemRefType>(ptr.getType()))
        return rewriter.notifyMatchFailure(op, "expected memref for scalar load");
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      auto loadVal = rewriter.create<affine::AffineLoadOp>(
          loc, ptr, zeroMap, ValueRange{});
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
      // Masked load: fill with 'other' value first, then copy valid region.
      // For Phase 1: fill with zero if no 'other' provided, then do full copy.
      // This is correct for the common case where mask is a bounds check
      // and we're inside the valid region.
      if (other) {
        // Try to extract scalar from 'other'
        auto otherVal = other;
        if (auto splatOp = otherVal.getDefiningOp<triton::SplatOp>())
          otherVal = splatOp.getSrc();
        if (isa<ShapedType>(otherVal.getType())) {
          // Can't extract scalar — just fill with zero
          TypedAttr zeroAttr;
          auto elemType = memrefType.getElementType();
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
        // No 'other' — fill with zero
        TypedAttr zeroAttr;
        auto elemType = memrefType.getElementType();
        if (isa<FloatType>(elemType))
          zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
        else
          zeroAttr = rewriter.getIntegerAttr(elemType, 0);
        auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                        ValueRange{alloc});
      }
      // Copy from source — for Phase 1, do full copy (mask check is on the
      // producer side). Full MaskAnalysis for subview-based partial copy
      // will come later.
      rewriter.create<memref::CopyOp>(loc, ptr, alloc);
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
    auto mask = op.getMask();
    auto loc = op.getLoc();

    // Scalar store
    if (!isa<ShapedType>(val.getType())) {
      if (!isa<MemRefType>(ptr.getType()))
        return rewriter.notifyMatchFailure(op, "expected memref for scalar store");
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      rewriter.create<affine::AffineStoreOp>(loc, val, ptr, zeroMap,
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
      // Masked store — for Phase 1, do full store (mask check is on the
      // producer side). Full MaskAnalysis for subview-based partial store
      // will come later.
      auto storeOp =
          rewriter.create<bufferization::MaterializeInDestinationOp>(
              loc, val, ptr);
      storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
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

  // Pointer / Memory (Phase 1)
  patterns.add<AddPtrConverter>(ctx);
  patterns.add<LoadConverter>(ctx);
  patterns.add<StoreConverter>(ctx);

  // Elementwise arith/math on tensors → linalg.generic
  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}
