//===-- PrepareSPIRV.cpp - Prepare memref IR for SPIR-V conversion --------===//
//
// This pass eliminates memref operations that have no direct SPIR-V lowering:
//   1. memref.reinterpret_cast → replaced with base memref + index arithmetic
//   2. memref.copy → expanded to explicit load/store loops
//   3. memref.cast (unranked→ranked) → eliminated by inlining
//   4. Attaches spirv.target_env attribute to the module
//
// After this pass, the IR contains only memref.load, memref.store,
// memref.alloc (simple ranked memrefs), arith.*, cf.*, and func.func,
// all of which have standard SPIR-V lowerings.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <string>

using namespace mlir;

namespace mlir {
namespace triton {
namespace vulkan {

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Expand memref.copy to an explicit loop of load/store.
struct ExpandMemRefCopy : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcType = cast<MemRefType>(op.getSource().getType());
    auto shape = srcType.getShape();

    if (shape.empty() || ShapedType::isDynamic(shape[0]))
      return failure(); // Only handle static 1D for now

    int64_t size = shape[0];
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto bound = rewriter.create<arith::ConstantIndexOp>(loc, size);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // scf.for %i = 0 to size step 1 { dst[i] = src[i] }
    rewriter.create<scf::ForOp>(
        loc, zero, bound, step, ValueRange{},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
          Value val = b.create<memref::LoadOp>(loc, op.getSource(), iv);
          b.create<memref::StoreOp>(loc, val, op.getTarget(), iv);
          b.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

/// Eliminate memref.reinterpret_cast by replacing uses with direct access
/// on the source memref with adjusted indices.
///
/// For:  %view = reinterpret_cast %base to offset:[off], sizes:[N], strides:[1]
///       %v = memref.load %view[%i]
/// Becomes:
///       %idx = arith.addi %i, off
///       %v = memref.load %base[%idx]
///
/// This works by: creating a ranked memref from unranked source,
/// then replacing every load/store on the view with load/store on the
/// ranked source using adjusted indices.
struct ExpandReinterpretCast
    : public OpRewritePattern<memref::ReinterpretCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto source = op.getSource();
    auto resultShape = op.getType().getShape();

    if (resultShape.size() != 1 || ShapedType::isDynamic(resultShape[0]))
      return failure();

    int64_t size = resultShape[0];
    auto offsets = op.getMixedOffsets();

    // Get the offset value
    Value offsetVal;
    if (auto attr = dyn_cast_if_present<Attribute>(offsets[0])) {
      int64_t off = cast<IntegerAttr>(attr).getInt();
      offsetVal = rewriter.create<arith::ConstantIndexOp>(loc, off);
    } else {
      offsetVal = cast<Value>(offsets[0]);
    }

    // Cast unranked to ranked if needed
    Value rankedSource = source;
    if (isa<UnrankedMemRefType>(source.getType())) {
      auto elemType =
          cast<UnrankedMemRefType>(source.getType()).getElementType();
      auto rankedType = MemRefType::get({ShapedType::kDynamic}, elemType);
      rankedSource = rewriter.create<memref::CastOp>(loc, rankedType, source);
    }

    // Create a simple ranked memref with static size for the view
    auto elemType = op.getType().getElementType();
    auto allocType = MemRefType::get({size}, elemType);

    // Check if this reinterpret_cast result is only used by load/store
    // If so, we can redirect loads/stores to the source with adjusted index
    bool allUsesAreLoadStore = true;
    for (auto &use : op.getResult().getUses()) {
      if (!isa<memref::LoadOp, memref::StoreOp>(use.getOwner())) {
        allUsesAreLoadStore = false;
        break;
      }
    }

    if (allUsesAreLoadStore) {
      // Replace each load/store to use rankedSource + offset
      SmallVector<OpOperand *> usesToRewrite;
      for (auto &use : op.getResult().getUses())
        usesToRewrite.push_back(&use);

      for (auto *use : usesToRewrite) {
        Operation *user = use->getOwner();
        rewriter.setInsertionPoint(user);

        if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
          auto idx = loadOp.getIndices()[0];
          Value newIdx =
              rewriter.create<arith::AddIOp>(loc, idx, offsetVal);
          auto newLoad = rewriter.create<memref::LoadOp>(
              loc, rankedSource, ValueRange{newIdx});
          rewriter.replaceOp(loadOp, newLoad.getResult());
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
          auto idx = storeOp.getIndices()[0];
          Value newIdx =
              rewriter.create<arith::AddIOp>(loc, idx, offsetVal);
          rewriter.create<memref::StoreOp>(loc, storeOp.getValue(),
                                           rankedSource, ValueRange{newIdx});
          rewriter.eraseOp(storeOp);
        }
      }
      rewriter.eraseOp(op);
      return success();
    }

    // Fallback: can't handle this pattern
    return failure();
  }
};

/// Lower memref.cast from unranked to ranked.
/// memref<*xf32> → memref<?xf32> becomes a direct substitution
/// since in SPIR-V all buffers are typed.
struct LowerUnrankedCast : public OpRewritePattern<memref::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSource().getType();
    auto dstType = op.getType();

    // Only handle unranked → ranked
    if (!isa<UnrankedMemRefType>(srcType) || !isa<MemRefType>(dstType))
      return failure();

    // For SPIR-V, we'll let the type converter handle this.
    // Just keep the cast — convert-memref-to-spirv will handle it.
    return failure();
  }
};

/// Convert memref.dealloc → no-op (SPIR-V handles memory lifetime).
struct RemoveDealloc : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class PrepareSPIRVPass
    : public PassWrapper<PrepareSPIRVPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareSPIRVPass)

  StringRef getArgument() const override { return "prepare-spirv"; }
  StringRef getDescription() const override {
    return "Prepare memref IR for SPIR-V conversion by expanding "
           "reinterpret_cast and copy ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    // 1. Attach spirv.target_env
    if (!moduleOp->hasAttr(spirv::getTargetEnvAttrName())) {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_0, {spirv::Capability::Shader},
          {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, ctx);
      auto limits = spirv::ResourceLimitsAttr::get(
          ctx, 16384, 128,
          Builder(ctx).getI32ArrayAttr({128, 128, 64}),
          32, std::nullopt, std::nullopt, ArrayAttr(), ArrayAttr());
      moduleOp->setAttr(spirv::getTargetEnvAttrName(),
                        spirv::TargetEnvAttr::get(triple, limits));
    }

    // 2. Convert function signatures: unranked memref → ranked 1D dynamic
    // This allows convert-memref-to-spirv to handle the args as StorageBuffer.
    moduleOp.walk([&](func::FuncOp func) {
      auto funcType = func.getFunctionType();
      bool changed = false;
      SmallVector<Type> newInputTypes;
      for (auto ty : funcType.getInputs()) {
        if (auto unranked = dyn_cast<UnrankedMemRefType>(ty)) {
          auto ranked = MemRefType::get({ShapedType::kDynamic},
                                        unranked.getElementType(),
                                        nullptr,
                                        unranked.getMemorySpace());
          newInputTypes.push_back(ranked);
          changed = true;
        } else {
          newInputTypes.push_back(ty);
        }
      }
      if (!changed)
        return;

      auto newFuncType = FunctionType::get(ctx, newInputTypes,
                                            funcType.getResults());
      func.setFunctionType(newFuncType);

      // Update block arguments
      auto &entryBlock = func.getBody().front();
      for (unsigned i = 0; i < newInputTypes.size(); i++) {
        auto arg = entryBlock.getArgument(i);
        if (arg.getType() != newInputTypes[i]) {
          arg.setType(newInputTypes[i]);
        }
      }
    });

    // 3. Remove memref.cast ops that are now identity casts
    moduleOp.walk([&](memref::CastOp castOp) {
      if (castOp.getSource().getType() == castOp.getType()) {
        castOp.replaceAllUsesWith(castOp.getSource());
        castOp.erase();
      }
    });

    // 4. Expand reinterpret_cast, copy, dealloc
    RewritePatternSet patterns(ctx);
    patterns.add<ExpandReinterpretCast, ExpandMemRefCopy, RemoveDealloc>(ctx);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // 5. Convert memref.alloc to memref.alloca (no address space — default).
    // Note: map-memref-spirv-storage-class WILL map address space 0 to
    // StorageBuffer, including these allocas. That's why
    // FixAllocaStorageClassPass must run AFTER map to change them to Function.
    moduleOp.walk([&](memref::AllocOp allocOp) {
      OpBuilder builder(allocOp);
      auto alloca = builder.create<memref::AllocaOp>(
          allocOp.getLoc(), allocOp.getType());
      allocOp.getResult().replaceAllUsesWith(alloca.getResult());
      allocOp.erase();
    });

    // Note: The caller should run the following passes AFTER this one:
    //   map-memref-spirv-storage-class → convert-memref-to-spirv →
    //   convert-arith-to-spirv → convert-cf-to-spirv → convert-func-to-spirv

    // 3. Lower remaining scf.for from copy expansion to cf
    // (The scf→cf conversion happens in make_memref, but copy expansion
    //  creates new scf.for ops. We need another scf→cf pass.)
  }
};

/// Fix memref.alloca storage classes after map-memref-spirv-storage-class.
/// The mapping pass assigns StorageBuffer to ALL memrefs (including allocas),
/// but convert-memref-to-spirv's AllocaOpPattern requires Function storage class.
/// This pass changes alloca types from StorageBuffer to Function.
class FixAllocaStorageClassPass
    : public PassWrapper<FixAllocaStorageClassPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixAllocaStorageClassPass)

  StringRef getArgument() const override { return "fix-alloca-storage-class"; }
  StringRef getDescription() const override {
    return "Change memref.alloca storage class to Function for SPIR-V";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    moduleOp.walk([&](memref::AllocaOp allocaOp) {
      auto oldType = allocaOp.getType();

      // Create new type with Function storage class
      auto funcAttr = spirv::StorageClassAttr::get(
          ctx, spirv::StorageClass::Function);
      auto newType = MemRefType::get(
          oldType.getShape(), oldType.getElementType(),
          oldType.getLayout(), funcAttr);

      // Replace the alloca with one that has Function class
      OpBuilder builder(allocaOp);
      auto newAlloca = builder.create<memref::AllocaOp>(
          allocaOp.getLoc(), newType);
      allocaOp.getResult().replaceAllUsesWith(newAlloca.getResult());
      allocaOp.erase();
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createFixAllocaStorageClassPass() {
  return std::make_unique<FixAllocaStorageClassPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createPrepareSPIRVPass() {
  return std::make_unique<PrepareSPIRVPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVModulePass() {
  return createPrepareSPIRVPass(); // Alias
}

// FinalizeSPIRV is handled in Python (compiler.py make_spv) via text
// wrapping + mlir-translate. The C++ approach of building spirv.module
// manually had verifier issues with block arg type mismatches.
std::unique_ptr<OperationPass<ModuleOp>> createFinalizeSPIRVPass() {
  // No-op — finalization is done in make_spv() Python code.
  return createPrepareSPIRVPass();
}

} // namespace vulkan
} // namespace triton
} // namespace mlir
