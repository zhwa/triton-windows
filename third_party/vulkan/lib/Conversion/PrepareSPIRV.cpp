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

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"

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

/// Expand memref.expand_shape by replacing 2D loads/stores with linearized 1D.
///
/// For:  %view = expand_shape %base [[0,1]] : memref<256xf32> into memref<16x16xf32>
///       %v = memref.load %view[%i, %j]
/// Becomes:
///       %idx = arith.addi(arith.muli(%i, 16), %j)
///       %v = memref.load %base[%idx]
struct ExpandExpandShape
    : public OpRewritePattern<memref::ExpandShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto source = op.getSrc();
    auto resultType = op.getResultType();

    // Only handle 1D → 2D expansion
    if (resultType.getRank() != 2)
      return failure();

    int64_t dim0 = resultType.getShape()[0];
    int64_t dim1 = resultType.getShape()[1];
    if (ShapedType::isDynamic(dim0) || ShapedType::isDynamic(dim1))
      return failure();

    // Replace all load/store users with linearized access on source
    bool allUsesHandled = true;
    SmallVector<OpOperand *> usesToRewrite;
    for (auto &use : op.getResult().getUses()) {
      if (isa<memref::LoadOp, memref::StoreOp>(use.getOwner()))
        usesToRewrite.push_back(&use);
      else if (isa<memref::CollapseShapeOp>(use.getOwner()))
        usesToRewrite.push_back(&use); // collapse undoes expand
      else
        allUsesHandled = false;
    }

    if (!allUsesHandled)
      return failure();

    for (auto *use : usesToRewrite) {
      Operation *user = use->getOwner();
      rewriter.setInsertionPoint(user);

      if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
        auto indices = loadOp.getIndices();
        if (indices.size() == 2) {
          Value dimConst = rewriter.create<arith::ConstantIndexOp>(loc, dim1);
          Value linearIdx = rewriter.create<arith::MulIOp>(loc, indices[0], dimConst);
          linearIdx = rewriter.create<arith::AddIOp>(loc, linearIdx, indices[1]);
          auto newLoad = rewriter.create<memref::LoadOp>(loc, source, ValueRange{linearIdx});
          rewriter.replaceOp(loadOp, newLoad.getResult());
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
        auto indices = storeOp.getIndices();
        if (indices.size() == 2) {
          Value dimConst = rewriter.create<arith::ConstantIndexOp>(loc, dim1);
          Value linearIdx = rewriter.create<arith::MulIOp>(loc, indices[0], dimConst);
          linearIdx = rewriter.create<arith::AddIOp>(loc, linearIdx, indices[1]);
          rewriter.create<memref::StoreOp>(loc, storeOp.getValue(), source, ValueRange{linearIdx});
          rewriter.eraseOp(storeOp);
        }
      } else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) {
        // collapse_shape undoes expand_shape → just use the original source
        collapseOp.getResult().replaceAllUsesWith(source);
        rewriter.eraseOp(collapseOp);
      }
    }
    rewriter.eraseOp(op);
    return success();
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

    // 1. Attach spirv.target_env with all capabilities we might need.
    // VulkanizePass will set the actual spirv.module VCE triple based on
    // which features are used (subgroup ops, cooperative matrix, etc.)
    if (!moduleOp->hasAttr(spirv::getTargetEnvAttrName())) {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_6,
          {spirv::Capability::Shader,
           spirv::Capability::Float16,
           spirv::Capability::StorageBuffer16BitAccess,
           spirv::Capability::GroupNonUniform,
           spirv::Capability::GroupNonUniformArithmetic,
           spirv::Capability::CooperativeMatrixKHR},
          {spirv::Extension::SPV_KHR_storage_buffer_storage_class,
           spirv::Extension::SPV_KHR_16bit_storage,
           spirv::Extension::SPV_KHR_cooperative_matrix}, ctx);
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
    patterns.add<ExpandReinterpretCast, ExpandMemRefCopy, RemoveDealloc,
                 ExpandExpandShape>(ctx);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // 5b. Flatten 2D allocs: memref<MxNxT> → memref<M*NxT>
    // This handles allocs that were created directly as 2D (e.g., matmul output)
    // and their collapse_shape users.
    moduleOp.walk([&](memref::AllocOp allocOp) {
      auto type = allocOp.getType();
      if (type.getRank() != 2) return;
      auto shape = type.getShape();
      if (ShapedType::isDynamic(shape[0]) || ShapedType::isDynamic(shape[1]))
        return;

      int64_t dim0 = shape[0], dim1 = shape[1];
      int64_t total = dim0 * dim1;
      auto flatType = MemRefType::get({total}, type.getElementType());

      OpBuilder builder(allocOp);
      auto flatAlloc = builder.create<memref::AllocOp>(
          allocOp.getLoc(), flatType);

      // Replace all users: linearize 2D indices
      SmallVector<Operation *> users(allocOp->getUsers().begin(),
                                     allocOp->getUsers().end());
      for (auto *user : users) {
        builder.setInsertionPoint(user);
        auto loc = user->getLoc();

        if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
          if (loadOp.getIndices().size() == 2) {
            Value d1 = builder.create<arith::ConstantIndexOp>(loc, dim1);
            Value idx = builder.create<arith::MulIOp>(loc, loadOp.getIndices()[0], d1);
            idx = builder.create<arith::AddIOp>(loc, idx, loadOp.getIndices()[1]);
            auto newLoad = builder.create<memref::LoadOp>(loc, flatAlloc, ValueRange{idx});
            loadOp.replaceAllUsesWith(newLoad.getResult());
            loadOp.erase();
          }
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
          if (storeOp.getIndices().size() == 2) {
            Value d1 = builder.create<arith::ConstantIndexOp>(loc, dim1);
            Value idx = builder.create<arith::MulIOp>(loc, storeOp.getIndices()[0], d1);
            idx = builder.create<arith::AddIOp>(loc, idx, storeOp.getIndices()[1]);
            builder.create<memref::StoreOp>(loc, storeOp.getValue(), flatAlloc, ValueRange{idx});
            storeOp.erase();
          }
        } else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) {
          collapseOp.getResult().replaceAllUsesWith(flatAlloc);
          collapseOp.erase();
        }
      }
      allocOp.erase();
    });

    // Do the same for alloca
    moduleOp.walk([&](memref::AllocaOp allocaOp) {
      auto type = allocaOp.getType();
      if (type.getRank() != 2) return;
      auto shape = type.getShape();
      if (ShapedType::isDynamic(shape[0]) || ShapedType::isDynamic(shape[1]))
        return;

      int64_t dim0 = shape[0], dim1 = shape[1];
      int64_t total = dim0 * dim1;
      auto flatType = MemRefType::get({total}, type.getElementType());

      OpBuilder builder(allocaOp);
      auto flatAlloca = builder.create<memref::AllocaOp>(
          allocaOp.getLoc(), flatType);

      SmallVector<Operation *> users(allocaOp->getUsers().begin(),
                                     allocaOp->getUsers().end());
      for (auto *user : users) {
        builder.setInsertionPoint(user);
        auto loc = user->getLoc();

        if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
          if (loadOp.getIndices().size() == 2) {
            Value d1 = builder.create<arith::ConstantIndexOp>(loc, dim1);
            Value idx = builder.create<arith::MulIOp>(loc, loadOp.getIndices()[0], d1);
            idx = builder.create<arith::AddIOp>(loc, idx, loadOp.getIndices()[1]);
            auto newLoad = builder.create<memref::LoadOp>(loc, flatAlloca, ValueRange{idx});
            loadOp.replaceAllUsesWith(newLoad.getResult());
            loadOp.erase();
          }
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
          if (storeOp.getIndices().size() == 2) {
            Value d1 = builder.create<arith::ConstantIndexOp>(loc, dim1);
            Value idx = builder.create<arith::MulIOp>(loc, storeOp.getIndices()[0], d1);
            idx = builder.create<arith::AddIOp>(loc, idx, storeOp.getIndices()[1]);
            builder.create<memref::StoreOp>(loc, storeOp.getValue(), flatAlloca, ValueRange{idx});
            storeOp.erase();
          }
        } else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) {
          collapseOp.getResult().replaceAllUsesWith(flatAlloca);
          collapseOp.erase();
        }
      }
      allocaOp.erase();
    });

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

//===----------------------------------------------------------------------===//
// ConvertReductionToParallel: Transform linalg.reduce → parallel tree reduce
//===----------------------------------------------------------------------===//

/// Replaces linalg.reduce with a parallel tree reduction using shared memory
/// (memref.global with address space 3 → Workgroup) and gpu.barrier for
/// synchronization. Each workgroup invocation loads one element and
/// participates in a log2(N) tree reduction.
///
/// This pass runs AFTER bufferize but BEFORE linalg-to-loops, so it can
/// consume linalg.reduce ops directly.
class ConvertReductionToParallel
    : public PassWrapper<ConvertReductionToParallel, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertReductionToParallel)

  StringRef getArgument() const override {
    return "convert-reduction-to-parallel";
  }
  StringRef getDescription() const override {
    return "Convert linalg.reduce to parallel tree reduction with shared memory";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, arith::ArithDialect,
                    scf::SCFDialect, func::FuncDialect>();
  }

  /// Emit func.call @__vulkan_barrier() as a placeholder.
  /// VulkanizePass will replace these with spirv.ControlBarrier.
  void emitBarrier(OpBuilder &builder, Location loc) {
    builder.create<func::CallOp>(
        loc, "__vulkan_barrier", TypeRange{}, ValueRange{});
  }

  /// Ensure the barrier declaration exists in the module.
  void declareBarrier(ModuleOp moduleOp, OpBuilder &builder) {
    if (moduleOp.lookupSymbol("__vulkan_barrier"))
      return;
    auto loc = moduleOp.getLoc();
    builder.setInsertionPointToStart(
        &moduleOp.getBodyRegion().front());
    auto funcType = builder.getFunctionType({}, {});
    auto decl = builder.create<func::FuncOp>(
        loc, "__vulkan_barrier", funcType);
    decl.setPrivate();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    // Collect all linalg.reduce ops first (can't modify while walking)
    SmallVector<linalg::ReduceOp> reduces;
    moduleOp.walk([&](linalg::ReduceOp op) { reduces.push_back(op); });

    if (reduces.empty())
      return;

    // Declare the barrier placeholder function
    OpBuilder moduleBuilder(ctx);
    declareBarrier(moduleOp, moduleBuilder);

    if (reduces.empty())
      return;

    for (auto reduceOp : reduces) {
      auto func = reduceOp->getParentOfType<func::FuncOp>();
      if (!func) continue;

      // Get input memref type and block size
      auto inputMemref = reduceOp.getInputs()[0];
      auto inputType = dyn_cast<MemRefType>(inputMemref.getType());
      if (!inputType || inputType.getRank() != 1) continue;

      int64_t blockSize = inputType.getShape()[0];
      if (ShapedType::isDynamic(blockSize)) continue;
      // Must be power of 2
      if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0) continue;

      auto elemType = inputType.getElementType();
      auto loc = reduceOp.getLoc();
      OpBuilder builder(reduceOp);

      // Set module attribute for VulkanizePass to read LocalSize
      // (func attributes may not survive func-to-spirv conversion)
      moduleOp->setAttr("vulkan.local_size",
                         builder.getI64ArrayAttr({blockSize, 1, 1}));

      // Allocate shared memory with address space 3 (Workgroup).
      // map-memref-spirv-storage-class will map AS 3 → Workgroup.
      // FixAllocaStorageClassPass only touches StorageBuffer, so this
      // alloca keeps its Workgroup storage class through conversion.
      // VulkanizePass will then promote it from function-scope Variable
      // to module-scope GlobalVariable (required by SPIR-V Workgroup).
      auto sharedType = MemRefType::get(
          {blockSize}, elemType, MemRefLayoutAttrInterface{},
          builder.getI64IntegerAttr(3));
      auto sharedAlloca = builder.create<memref::AllocaOp>(loc, sharedType);
      Value sharedRef = sharedAlloca.getResult();

      // Get local_id_x from function args (last arg)
      auto localIdArg = func.getArgument(func.getNumArguments() - 3);
      Value tidIdx = builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), localIdArg);

      // Each thread loads one element from input → shared[tid]
      Value val = builder.create<memref::LoadOp>(
          loc, inputMemref, tidIdx);
      builder.create<memref::StoreOp>(
          loc, val, sharedRef, tidIdx);

      // Barrier: ensure all threads have stored
      emitBarrier(builder, loc);

      // Determine subgroup size for optimization.
      // If blockSize > subgroupSize AND the combiner maps to a known
      // subgroup op, use shared memory tree reduction for outer strides
      // (>= subgroupSize) and a single subgroup reduce for inner strides.
      // Otherwise, fall back to full shared memory tree reduction.
      // TODO: make configurable via module attribute instead of hardcoding.
      constexpr int64_t SUBGROUP_SIZE = 32; // Turing/Ampere/Hopper

      // Classify combiner op for subgroup reduce placeholder
      auto *combiner = reduceOp.getCombiner().front().getTerminator();
      std::string subgroupOpName;
      if (!combiner->getOperands().empty()) {
        auto *combinerOp = combiner->getOperand(0).getDefiningOp();
        if (combinerOp) {
          if (isa<arith::AddFOp>(combinerOp))
            subgroupOpName = "__vulkan_subgroup_reduce_fadd";
          else if (isa<arith::AddIOp>(combinerOp))
            subgroupOpName = "__vulkan_subgroup_reduce_iadd";
          else if (isa<arith::MaximumFOp>(combinerOp))
            subgroupOpName = "__vulkan_subgroup_reduce_fmax";
          else if (isa<arith::MaxSIOp>(combinerOp))
            subgroupOpName = "__vulkan_subgroup_reduce_smax";
          else if (isa<arith::MinimumFOp>(combinerOp))
            subgroupOpName = "__vulkan_subgroup_reduce_fmin";
          else if (isa<arith::MinSIOp>(combinerOp))
            subgroupOpName = "__vulkan_subgroup_reduce_smin";
        }
      }

      // Tree reduction: stride halving (only for strides >= SUBGROUP_SIZE)
      int64_t stopStride = (!subgroupOpName.empty() && blockSize > SUBGROUP_SIZE)
                               ? SUBGROUP_SIZE : 1;

      for (int64_t stride = blockSize / 2; stride >= stopStride; stride /= 2) {
        auto strideConst =
            builder.create<arith::ConstantIndexOp>(loc, stride);
        auto cmp = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, tidIdx, strideConst);

        auto ifOp = builder.create<scf::IfOp>(loc, cmp,
                                               /*withElseRegion=*/false);
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(
              &ifOp.getThenRegion().front());

          Value a = builder.create<memref::LoadOp>(
              loc, sharedRef, tidIdx);
          Value bIdx = builder.create<arith::AddIOp>(
              loc, tidIdx, strideConst);
          Value b = builder.create<memref::LoadOp>(
              loc, sharedRef, bIdx);

          // Clone the reduction combiner from linalg.reduce body
          Block &combinerBlock = reduceOp.getCombiner().front();
          IRMapping mapping;
          mapping.map(combinerBlock.getArgument(0), a);
          mapping.map(combinerBlock.getArgument(1), b);

          Value result;
          Operation *terminator = combinerBlock.getTerminator();
          for (Operation &bodyOp : combinerBlock.getOperations()) {
            if (&bodyOp == terminator)
              break;
            Operation *cloned = builder.clone(bodyOp, mapping);
            for (unsigned r = 0; r < bodyOp.getNumResults(); r++)
              mapping.map(bodyOp.getResult(r), cloned->getResult(r));
          }
          result = mapping.lookup(terminator->getOperand(0));

          builder.create<memref::StoreOp>(
              loc, result, sharedRef, tidIdx);
        }

        emitBarrier(builder, loc);
      }

      Value finalVal;
      if (!subgroupOpName.empty() && blockSize > SUBGROUP_SIZE) {
        // Subgroup reduce for inner strides (replaces strides 16→1).
        // After shared memory tree reduction to stride=SUBGROUP_SIZE,
        // threads 0..31 hold partial results in shared[0..31].
        // A subgroup reduce across all 32 threads gives the final value.
        //
        // Emit: %result = call @__vulkan_subgroup_reduce_*(%shared[tid])
        // VulkanizePass converts to spirv.GroupNonUniform* Reduce Subgroup.

        // Declare the subgroup reduce function if needed
        {
          auto funcType = FunctionType::get(ctx, {elemType}, {elemType});
          if (!moduleOp.lookupSymbol(subgroupOpName)) {
            OpBuilder declBuilder(ctx);
            declBuilder.setInsertionPointToStart(
                &moduleOp.getBodyRegion().front());
            auto decl = declBuilder.create<func::FuncOp>(
                loc, subgroupOpName, funcType);
            decl.setPrivate();
          }
        }

        Value partialVal = builder.create<memref::LoadOp>(
            loc, sharedRef, tidIdx);
        auto callOp = builder.create<func::CallOp>(
            loc, subgroupOpName, TypeRange{elemType},
            ValueRange{partialVal});
        Value subgroupResult = callOp.getResult(0);

        // Store result to shared[tid] so all threads see it
        builder.create<memref::StoreOp>(
            loc, subgroupResult, sharedRef, tidIdx);

        // Barrier: ensure subgroup 0's store to shared[0] is visible
        // to all threads before reading
        emitBarrier(builder, loc);

        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        finalVal = builder.create<memref::LoadOp>(
            loc, sharedRef, zero);
      } else {
        // No subgroup optimization — shared[0] already has the result
        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        finalVal = builder.create<memref::LoadOp>(
            loc, sharedRef, zero);
      }

      Value outputMemref = reduceOp.getDpsInits()[0];
      auto outType = cast<MemRefType>(outputMemref.getType());
      SmallVector<Value> outIndices;
      Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
      for (int64_t i = 0; i < outType.getRank(); i++)
        outIndices.push_back(zeroIdx);
      builder.create<memref::StoreOp>(
          loc, finalVal, outputMemref, outIndices);

      // Erase the original linalg.reduce
      reduceOp.erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// ConvertMatmulToCooperative: Replace linalg.matmul with coop matrix placeholder
//===----------------------------------------------------------------------===//

/// Traces a bufferized matmul operand backward to its original function arg.
/// Pattern: alloc → memref.copy(source, alloc) → source traces through
/// reinterpret_cast/cast/subview to a BlockArgument.
static std::optional<unsigned> traceToFuncArg(Value operand) {
  // Operand is typically an alloc result
  auto allocOp = operand.getDefiningOp<memref::AllocOp>();
  if (!allocOp) return std::nullopt;

  // Find memref.copy where this alloc is the destination
  for (auto *user : allocOp->getUsers()) {
    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getTarget() != allocOp.getResult())
      continue;

    // Walk the source through view-like ops to find the func arg
    Value source = copyOp.getSource();
    while (true) {
      if (auto blockArg = dyn_cast<BlockArgument>(source))
        return blockArg.getArgNumber();
      if (auto op = source.getDefiningOp<memref::ReinterpretCastOp>()) {
        source = op.getSource(); continue;
      }
      if (auto op = source.getDefiningOp<memref::CastOp>()) {
        source = op.getSource(); continue;
      }
      if (auto op = source.getDefiningOp<memref::SubViewOp>()) {
        source = op.getSource(); continue;
      }
      break;
    }
  }
  return std::nullopt;
}

/// Traces a matmul output forward: alloc → collapse_shape/copy → destination
/// → through view ops to a BlockArgument.
static std::optional<unsigned> traceOutputToFuncArg(Value initOperand) {
  auto allocOp = initOperand.getDefiningOp<memref::AllocOp>();
  if (!allocOp) return std::nullopt;

  // Walk users to find copy/collapse that leads to the output buffer
  SmallVector<Value> worklist = {allocOp.getResult()};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (auto *user : current.getUsers()) {
      // memref.copy where current is SOURCE → trace dest to func arg
      if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
        if (copyOp.getSource() == current) {
          Value dest = copyOp.getTarget();
          while (true) {
            if (auto blockArg = dyn_cast<BlockArgument>(dest))
              return blockArg.getArgNumber();
            if (auto op = dest.getDefiningOp<memref::ReinterpretCastOp>()) {
              dest = op.getSource(); continue;
            }
            if (auto op = dest.getDefiningOp<memref::CastOp>()) {
              dest = op.getSource(); continue;
            }
            break;
          }
        }
      }
      // collapse_shape → follow its result
      if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user))
        worklist.push_back(collapseOp.getResult());
    }
  }
  return std::nullopt;
}

/// Replaces linalg.matmul on small static tiles (16x16 f16) with a no-arg
/// placeholder call. Buffer arg indices are stored as module attributes so
/// VulkanizePass can access StorageBuffer GlobalVariables directly (bypassing
/// the alloca-copy pattern that produces Function-class pointers).
class ConvertMatmulToCooperative
    : public PassWrapper<ConvertMatmulToCooperative, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMatmulToCooperative)

  StringRef getArgument() const override {
    return "convert-matmul-to-cooperative";
  }
  StringRef getDescription() const override {
    return "Convert linalg.matmul to cooperative matrix placeholder";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    SmallVector<linalg::MatmulOp> matmuls;
    moduleOp.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

    if (matmuls.empty())
      return;

    for (auto matmulOp : matmuls) {
      // Only handle 2D static 16x16 f16 matmuls
      auto outType = dyn_cast<MemRefType>(
          matmulOp.getDpsInits()[0].getType());
      if (!outType || outType.getRank() != 2) continue;

      auto shape = outType.getShape();
      if (ShapedType::isDynamic(shape[0]) || ShapedType::isDynamic(shape[1]))
        continue;

      auto elemType = outType.getElementType();
      if (!elemType.isF16()) continue;
      if (shape[0] != 16 || shape[1] != 16) continue;

      auto aType = dyn_cast<MemRefType>(
          matmulOp.getDpsInputs()[0].getType());
      if (!aType || aType.getRank() != 2) continue;
      if (aType.getShape()[0] != 16 || aType.getShape()[1] != 16) continue;

      // Trace operands back to function args
      auto argA = traceToFuncArg(matmulOp.getDpsInputs()[0]);
      auto argB = traceToFuncArg(matmulOp.getDpsInputs()[1]);
      auto argC = traceOutputToFuncArg(matmulOp.getDpsInits()[0]);

      if (!argA || !argB || !argC) {
        matmulOp.emitWarning("Could not trace coop matmul operands to "
                             "function args; skipping cooperative conversion");
        continue;
      }

      auto loc = matmulOp.getLoc();
      OpBuilder builder(matmulOp);

      // Store metadata as module attributes for VulkanizePass
      moduleOp->setAttr("vulkan.coop_matmul", builder.getUnitAttr());
      moduleOp->setAttr("vulkan.coop_buffer_args",
                         builder.getI64ArrayAttr(
                             {static_cast<int64_t>(*argA),
                              static_cast<int64_t>(*argB),
                              static_cast<int64_t>(*argC)}));
      moduleOp->setAttr("vulkan.coop_dims",
                         builder.getI64ArrayAttr({shape[0], shape[1],
                             aType.getShape()[1]}));

      // Emit no-arg placeholder call.
      // VulkanizePass will use vulkan.coop_buffer_args to access the
      // correct StorageBuffer GlobalVariables directly.
      std::string funcName = "__vulkan_coop_matmul";
      if (!moduleOp.lookupSymbol(funcName)) {
        auto funcType = FunctionType::get(ctx, {}, {});
        OpBuilder declBuilder(ctx);
        declBuilder.setInsertionPointToStart(
            &moduleOp.getBodyRegion().front());
        auto decl = declBuilder.create<func::FuncOp>(
            loc, funcName, funcType);
        decl.setPrivate();
      }

      builder.create<func::CallOp>(loc, funcName, TypeRange{}, ValueRange{});

      matmulOp.erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// VulkanizePass: Convert spirv.func args → GlobalVariables for Vulkan dispatch
//===----------------------------------------------------------------------===//

/// After convert-func-to-spirv, the IR has:
///   spirv.func @kernel(%arg0: !spirv.ptr<struct<rtarray<f32>>, StorageBuffer>,
///                      %arg1: !spirv.ptr<...>, ..., %argN: i32) { ... }
///
/// Vulkan requires StorageBuffer bindings as global variables with descriptor
/// decorations, not function parameters. This pass:
///   1. Creates spirv.GlobalVariable for each StorageBuffer arg
///   2. Replaces arg uses with spirv.mlir.addressof
///   3. Removes the args from the function signature
///   4. Wraps in spirv.module with EntryPoint + ExecutionMode
class VulkanizePass
    : public PassWrapper<VulkanizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VulkanizePass)

  StringRef getArgument() const override { return "vulkanize-spirv"; }
  StringRef getDescription() const override {
    return "Convert spirv.func args to Vulkan-compatible GlobalVariables";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();
    auto loc = moduleOp.getLoc();

    // Find the spirv.func
    spirv::FuncOp funcOp;
    moduleOp.walk([&](spirv::FuncOp f) { funcOp = f; });
    if (!funcOp)
      return;

    auto funcType = funcOp.getFunctionType();
    auto &entryBlock = funcOp.getBody().front();
    std::string kernelName = funcOp.getName().str();

    // Classify args: StorageBuffer pointers vs scalars (i32)
    SmallVector<unsigned> bufferArgIndices;
    SmallVector<unsigned> scalarArgIndices;
    for (unsigned i = 0; i < funcType.getNumInputs(); i++) {
      auto ty = funcType.getInput(i);
      if (auto ptrTy = dyn_cast<spirv::PointerType>(ty)) {
        if (ptrTy.getStorageClass() == spirv::StorageClass::StorageBuffer)
          bufferArgIndices.push_back(i);
        else
          scalarArgIndices.push_back(i);
      } else {
        scalarArgIndices.push_back(i);
      }
    }

    // Create spirv.module to wrap everything
    OpBuilder moduleBuilder(ctx);
    moduleBuilder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
    auto spirvModule = moduleBuilder.create<spirv::ModuleOp>(
        loc, spirv::AddressingModel::Logical,
        spirv::MemoryModel::GLSL450);

    // VCE triple will be set after all replacements (need to know if
    // subgroup ops are used)

    // Build inside the spirv.module body
    auto *spirvBody = spirvModule.getBody();
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(spirvBody);

    // Create GlobalVariables for each StorageBuffer arg
    SmallVector<spirv::GlobalVariableOp> globalVars;
    SmallVector<Attribute> interfaceVarRefs;  // for EntryPoint interface list
    for (unsigned idx = 0; idx < bufferArgIndices.size(); idx++) {
      unsigned argIdx = bufferArgIndices[idx];
      auto argType = funcType.getInput(argIdx);
      std::string varName = "__buffer_" + std::to_string(idx);

      // Build GlobalVariableOp with descriptor set/binding decorations.
      // Use OperationState since the build() signature is limited.
      OperationState state(loc, spirv::GlobalVariableOp::getOperationName());
      state.addAttribute("sym_name", builder.getStringAttr(varName));
      state.addAttribute("type", TypeAttr::get(argType));
      state.addAttribute("descriptor_set", builder.getI32IntegerAttr(0));
      state.addAttribute("binding", builder.getI32IntegerAttr(idx));
      auto *op = builder.create(state);
      globalVars.push_back(cast<spirv::GlobalVariableOp>(op));
      interfaceVarRefs.push_back(
          FlatSymbolRefAttr::get(ctx, varName));
    }

    // Move the function into the spirv.module
    funcOp->remove();
    spirvBody->push_back(funcOp.getOperation());

    // Promote the shared memory Variable to Workgroup storage class.
    // ConvertReductionToParallel created a memref.alloca with AS 3 (Workgroup)
    // which FixAllocaStorageClassPass changed to Function. After SPIR-V
    // conversion, it's a spirv.Variable Function. We identify it as the
    // LAST Variable matching the block size (the shared alloca was created
    // after the load buffer alloca).
    SmallVector<spirv::VariableOp> sharedVars;
    if (moduleOp->hasAttr("vulkan.local_size")) {
      auto lsAttr = moduleOp->getAttrOfType<ArrayAttr>("vulkan.local_size");
      int64_t blockSize = cast<IntegerAttr>(lsAttr[0]).getInt();

      SmallVector<spirv::VariableOp> candidates;
      SmallVector<spirv::VariableOp> loadBuffers;
      funcOp.walk([&](spirv::VariableOp varOp) {
        auto ptrType = dyn_cast<spirv::PointerType>(varOp.getType());
        if (!ptrType) return;
        auto structType = dyn_cast<spirv::StructType>(
            ptrType.getPointeeType());
        if (!structType || structType.getNumElements() != 1) return;
        auto arrayType = dyn_cast<spirv::ArrayType>(
            structType.getElementType(0));
        if (!arrayType) return;
        if (arrayType.getNumElements() == static_cast<unsigned>(blockSize))
          candidates.push_back(varOp);
      });
      // The first matching variable is the load buffer (from tt.load → alloc).
      // All subsequent matching variables are shared memory (from
      // ConvertReductionToParallel). Skip the first, promote the rest.
      for (unsigned i = 1; i < candidates.size(); i++)
        sharedVars.push_back(candidates[i]);
    }
    for (auto varOp : sharedVars) {
      auto ptrType = cast<spirv::PointerType>(varOp.getType());
      auto pointeeType = ptrType.getPointeeType();
      auto workgroupPtrType = spirv::PointerType::get(
          pointeeType, spirv::StorageClass::Workgroup);

      std::string varName = "__shared_" + std::to_string(
          interfaceVarRefs.size());

      builder.setInsertionPoint(funcOp);
      OperationState gvState(loc, spirv::GlobalVariableOp::getOperationName());
      gvState.addAttribute("sym_name", builder.getStringAttr(varName));
      gvState.addAttribute("type", TypeAttr::get(workgroupPtrType));
      auto *gvOp = builder.create(gvState);
      auto sharedGlobal = cast<spirv::GlobalVariableOp>(gvOp);

      interfaceVarRefs.push_back(
          FlatSymbolRefAttr::get(ctx, varName));

      // Replace the function-scope Variable with addressof and update
      // all downstream AccessChain/Load/Store to use Workgroup class
      builder.setInsertionPointAfter(varOp);
      auto addrOf = builder.create<spirv::AddressOfOp>(loc, sharedGlobal);

      // Walk users and rebuild AccessChain ops with Workgroup ptr types
      SmallVector<Operation *> users(varOp->getUsers().begin(),
                                     varOp->getUsers().end());
      for (auto *user : users) {
        if (auto acOp = dyn_cast<spirv::AccessChainOp>(user)) {
          // Compute new result type: same element type, Workgroup class
          auto oldResultType = cast<spirv::PointerType>(acOp.getType());
          auto newResultType = spirv::PointerType::get(
              oldResultType.getPointeeType(),
              spirv::StorageClass::Workgroup);

          builder.setInsertionPointAfter(acOp);
          auto newAC = builder.create<spirv::AccessChainOp>(
              acOp.getLoc(), newResultType, addrOf, acOp.getIndices());

          // Replace users of the old AccessChain
          SmallVector<Operation *> acUsers(acOp->getUsers().begin(),
                                           acOp->getUsers().end());
          for (auto *acUser : acUsers) {
            if (auto loadOp = dyn_cast<spirv::LoadOp>(acUser)) {
              builder.setInsertionPointAfter(loadOp);
              auto newLoad = builder.create<spirv::LoadOp>(
                  loadOp.getLoc(), newAC);
              loadOp.replaceAllUsesWith(newLoad.getOperation());
              loadOp.erase();
            } else if (auto storeOp = dyn_cast<spirv::StoreOp>(acUser)) {
              builder.setInsertionPointAfter(storeOp);
              builder.create<spirv::StoreOp>(
                  storeOp.getLoc(), newAC, storeOp.getValue());
              storeOp.erase();
            }
          }
          acOp.erase();
        }
      }
      varOp.erase();
    }

    // Split scalar args: last 6 are local_id(3) + program_id(3),
    // replaced with LocalInvocationId and WorkgroupId builtins.
    // Convention: [..., num_programs(3), pid(3), local_id(3)]
    constexpr unsigned NUM_BUILTIN_ARGS = 6; // 3 pid + 3 local_id
    constexpr unsigned NUM_PID_ARGS = 3;
    constexpr unsigned NUM_LID_ARGS = 3;
    SmallVector<unsigned> pushConstArgIndices;
    SmallVector<unsigned> pidArgIndices;
    SmallVector<unsigned> lidArgIndices;
    if (scalarArgIndices.size() < NUM_BUILTIN_ARGS) {
      funcOp.emitError("Expected at least 6 scalar args for program_id + local_id");
      return signalPassFailure();
    }
    for (unsigned i = 0; i < scalarArgIndices.size(); i++) {
      unsigned fromEnd = scalarArgIndices.size() - i;
      if (fromEnd <= NUM_LID_ARGS)
        lidArgIndices.push_back(scalarArgIndices[i]);
      else if (fromEnd <= NUM_LID_ARGS + NUM_PID_ARGS)
        pidArgIndices.push_back(scalarArgIndices[i]);
      else
        pushConstArgIndices.push_back(scalarArgIndices[i]);
    }

    // Create WorkgroupId builtin variable (provides program_id via dispatch).
    auto vec3i32 = VectorType::get({3}, builder.getI32Type());
    auto inputPtrType = spirv::PointerType::get(
        vec3i32, spirv::StorageClass::Input);

    builder.setInsertionPoint(funcOp);
    OperationState wgState(loc, spirv::GlobalVariableOp::getOperationName());
    wgState.addAttribute("sym_name",
                         builder.getStringAttr("__builtin_workgroup_id"));
    wgState.addAttribute("type", TypeAttr::get(inputPtrType));
    wgState.addAttribute("built_in", builder.getStringAttr("WorkgroupId"));
    auto *wgOp = builder.create(wgState);
    auto workgroupIdVar = cast<spirv::GlobalVariableOp>(wgOp);
    interfaceVarRefs.push_back(
        FlatSymbolRefAttr::get(ctx, "__builtin_workgroup_id"));

    // Create LocalInvocationId builtin variable (thread ID within workgroup)
    OperationState lidState(loc, spirv::GlobalVariableOp::getOperationName());
    lidState.addAttribute("sym_name",
                          builder.getStringAttr("__builtin_local_invocation_id"));
    lidState.addAttribute("type", TypeAttr::get(inputPtrType));
    lidState.addAttribute("built_in",
                          builder.getStringAttr("LocalInvocationId"));
    auto *lidOp = builder.create(lidState);
    auto localInvocationIdVar = cast<spirv::GlobalVariableOp>(lidOp);
    interfaceVarRefs.push_back(
        FlatSymbolRefAttr::get(ctx, "__builtin_local_invocation_id"));

    // Create push constant struct for non-pid scalar args only
    spirv::GlobalVariableOp pushConstVar;
    if (!pushConstArgIndices.empty()) {
      SmallVector<Type> memberTypes;
      for (auto idx : pushConstArgIndices)
        memberTypes.push_back(funcType.getInput(idx));

      // Create struct with Offset decorations based on actual type sizes
      SmallVector<spirv::StructType::OffsetInfo> offsets;
      unsigned currentOffset = 0;
      for (unsigned i = 0; i < memberTypes.size(); i++) {
        offsets.push_back(currentOffset);
        unsigned typeSize = 4; // default i32
        if (memberTypes[i].isInteger(64) || memberTypes[i].isF64())
          typeSize = 8;
        else if (memberTypes[i].isInteger(16) || memberTypes[i].isF16())
          typeSize = 2;
        currentOffset += typeSize;
      }

      auto structType = spirv::StructType::get(memberTypes, offsets);
      auto ptrType = spirv::PointerType::get(
          structType, spirv::StorageClass::PushConstant);

      builder.setInsertionPoint(funcOp);
      OperationState pcState(loc, spirv::GlobalVariableOp::getOperationName());
      pcState.addAttribute("sym_name", builder.getStringAttr("__push_constants"));
      pcState.addAttribute("type", TypeAttr::get(ptrType));
      auto *pcOp = builder.create(pcState);
      pushConstVar = cast<spirv::GlobalVariableOp>(pcOp);

      // Add to interface variables
      interfaceVarRefs.push_back(
          FlatSymbolRefAttr::get(ctx, pushConstVar.getSymName()));
    }

    // Replace arg uses: buffer args → addressof
    builder.setInsertionPointToStart(&entryBlock);
    for (unsigned i = 0; i < bufferArgIndices.size(); i++) {
      unsigned argIdx = bufferArgIndices[i];
      auto arg = entryBlock.getArgument(argIdx);
      auto addrOf = builder.create<spirv::AddressOfOp>(
          loc, globalVars[i]);
      arg.replaceAllUsesWith(addrOf.getResult());
    }

    // Replace push constant args (non-pid scalars)
    if (pushConstVar) {
      auto pcAddrOf = builder.create<spirv::AddressOfOp>(loc, pushConstVar);
      for (unsigned i = 0; i < pushConstArgIndices.size(); i++) {
        unsigned argIdx = pushConstArgIndices[i];
        auto arg = entryBlock.getArgument(argIdx);
        auto memberIdx = builder.create<spirv::ConstantOp>(
            loc, builder.getI32Type(), builder.getI32IntegerAttr(i));
        auto elemPtrType = spirv::PointerType::get(
            arg.getType(), spirv::StorageClass::PushConstant);
        auto accessChain = builder.create<spirv::AccessChainOp>(
            loc, elemPtrType, pcAddrOf, ValueRange{memberIdx});
        auto loaded = builder.create<spirv::LoadOp>(loc, accessChain);
        arg.replaceAllUsesWith(loaded);
      }
    }

    // Replace program_id args with WorkgroupId builtin components
    {
      auto wgAddrOf = builder.create<spirv::AddressOfOp>(loc, workgroupIdVar);
      auto wgLoaded = builder.create<spirv::LoadOp>(loc, wgAddrOf);
      for (unsigned i = 0; i < pidArgIndices.size(); i++) {
        unsigned argIdx = pidArgIndices[i];
        auto arg = entryBlock.getArgument(argIdx);
        auto extracted = builder.create<spirv::CompositeExtractOp>(
            loc, builder.getI32Type(), wgLoaded,
            builder.getI32ArrayAttr({static_cast<int32_t>(i)}));
        arg.replaceAllUsesWith(extracted);
      }
    }

    // Replace local_id args with LocalInvocationId builtin components
    {
      auto lidAddrOf = builder.create<spirv::AddressOfOp>(
          loc, localInvocationIdVar);
      auto lidLoaded = builder.create<spirv::LoadOp>(loc, lidAddrOf);
      for (unsigned i = 0; i < lidArgIndices.size(); i++) {
        unsigned argIdx = lidArgIndices[i];
        auto arg = entryBlock.getArgument(argIdx);
        auto extracted = builder.create<spirv::CompositeExtractOp>(
            loc, builder.getI32Type(), lidLoaded,
            builder.getI32ArrayAttr({static_cast<int32_t>(i)}));
        arg.replaceAllUsesWith(extracted);
      }
    }

    // Replace barrier placeholder calls with spirv.ControlBarrier
    SmallVector<spirv::FunctionCallOp> barrierCalls;
    funcOp.walk([&](spirv::FunctionCallOp callOp) {
      if (callOp.getCallee() == "__vulkan_barrier")
        barrierCalls.push_back(callOp);
    });
    for (auto callOp : barrierCalls) {
      OpBuilder barrierBuilder(callOp);
      OperationState state(callOp.getLoc(),
                           spirv::ControlBarrierOp::getOperationName());
      state.addAttribute("execution_scope",
          spirv::ScopeAttr::get(ctx, spirv::Scope::Workgroup));
      state.addAttribute("memory_scope",
          spirv::ScopeAttr::get(ctx, spirv::Scope::Workgroup));
      state.addAttribute("memory_semantics",
          spirv::MemorySemanticsAttr::get(
              ctx, spirv::MemorySemantics::WorkgroupMemory |
                   spirv::MemorySemantics::AcquireRelease));
      barrierBuilder.create(state);
      callOp.erase();
    }
    // Also remove the barrier function declaration from the spirv.module
    if (auto barrierDecl = dyn_cast_or_null<spirv::FuncOp>(
            spirvModule.lookupSymbol("__vulkan_barrier"))) {
      barrierDecl.erase();
    }

    // Replace subgroup reduce placeholder calls with spirv.GroupNonUniform* ops
    bool hasSubgroupOps = false;
    SmallVector<spirv::FunctionCallOp> subgroupCalls;
    funcOp.walk([&](spirv::FunctionCallOp callOp) {
      auto callee = callOp.getCallee();
      if (callee.starts_with("__vulkan_subgroup_reduce_"))
        subgroupCalls.push_back(callOp);
    });
    for (auto callOp : subgroupCalls) {
      hasSubgroupOps = true;
      auto callee = callOp.getCallee();
      auto inputVal = callOp.getOperand(0);
      auto resultType = callOp.getResult(0).getType();
      auto subgroupScope = spirv::ScopeAttr::get(ctx, spirv::Scope::Subgroup);
      auto reduceOp = spirv::GroupOperationAttr::get(
          ctx, spirv::GroupOperation::Reduce);

      OpBuilder sgBuilder(callOp);
      Value replacement;
      if (callee == "__vulkan_subgroup_reduce_fadd") {
        replacement = spirv::GroupNonUniformFAddOp::create(
            sgBuilder, callOp.getLoc(), resultType,
            subgroupScope, reduceOp, inputVal, /*cluster_size=*/nullptr);
      } else if (callee == "__vulkan_subgroup_reduce_iadd") {
        replacement = spirv::GroupNonUniformIAddOp::create(
            sgBuilder, callOp.getLoc(), resultType,
            subgroupScope, reduceOp, inputVal, /*cluster_size=*/nullptr);
      } else if (callee == "__vulkan_subgroup_reduce_fmax") {
        replacement = spirv::GroupNonUniformFMaxOp::create(
            sgBuilder, callOp.getLoc(), resultType,
            subgroupScope, reduceOp, inputVal, /*cluster_size=*/nullptr);
      } else if (callee == "__vulkan_subgroup_reduce_smax") {
        replacement = spirv::GroupNonUniformSMaxOp::create(
            sgBuilder, callOp.getLoc(), resultType,
            subgroupScope, reduceOp, inputVal, /*cluster_size=*/nullptr);
      } else if (callee == "__vulkan_subgroup_reduce_fmin") {
        replacement = spirv::GroupNonUniformFMinOp::create(
            sgBuilder, callOp.getLoc(), resultType,
            subgroupScope, reduceOp, inputVal, /*cluster_size=*/nullptr);
      } else if (callee == "__vulkan_subgroup_reduce_smin") {
        replacement = spirv::GroupNonUniformSMinOp::create(
            sgBuilder, callOp.getLoc(), resultType,
            subgroupScope, reduceOp, inputVal, /*cluster_size=*/nullptr);
      }
      if (replacement) {
        callOp.getResult(0).replaceAllUsesWith(replacement);
        callOp.erase();
      }
    }
    // Remove subgroup reduce function declarations
    for (auto name : {"__vulkan_subgroup_reduce_fadd",
                      "__vulkan_subgroup_reduce_iadd",
                      "__vulkan_subgroup_reduce_fmax",
                      "__vulkan_subgroup_reduce_smax",
                      "__vulkan_subgroup_reduce_fmin",
                      "__vulkan_subgroup_reduce_smin"}) {
      if (auto decl = dyn_cast_or_null<spirv::FuncOp>(
              spirvModule.lookupSymbol(name))) {
        decl.erase();
      }
    }

    // Replace cooperative matmul placeholder calls with
    // spirv.KHRCooperativeMatrix{Load,MulAdd,Store} ops.
    // Unlike other placeholders, coop matmul uses StorageBuffer GlobalVariables
    // directly (from vulkan.coop_buffer_args attribute) instead of the
    // Function-class alloca copies that the placeholder operands would provide.
    bool hasCoopMatmul = false;
    SmallVector<spirv::FunctionCallOp> coopCalls;
    funcOp.walk([&](spirv::FunctionCallOp callOp) {
      if (callOp.getCallee() == "__vulkan_coop_matmul")
        coopCalls.push_back(callOp);
    });
    if (!coopCalls.empty() && moduleOp->hasAttr("vulkan.coop_buffer_args")) {
      hasCoopMatmul = true;
      auto argsAttr = moduleOp->getAttrOfType<ArrayAttr>("vulkan.coop_buffer_args");
      auto dimsAttr = moduleOp->getAttrOfType<ArrayAttr>("vulkan.coop_dims");

      // Map buffer arg index → GlobalVariable
      // bufferArgIndices[i] → globalVars[i], so we need to find which
      // position in bufferArgIndices matches the coop arg index.
      auto findGlobalVar = [&](int64_t funcArgIdx) -> spirv::GlobalVariableOp {
        for (unsigned i = 0; i < bufferArgIndices.size(); i++) {
          if (bufferArgIndices[i] == static_cast<unsigned>(funcArgIdx))
            return globalVars[i];
        }
        return nullptr;
      };

      int64_t argIdxA = cast<IntegerAttr>(argsAttr[0]).getInt();
      int64_t argIdxB = cast<IntegerAttr>(argsAttr[1]).getInt();
      int64_t argIdxC = cast<IntegerAttr>(argsAttr[2]).getInt();
      int64_t stride = dimsAttr
          ? cast<IntegerAttr>(dimsAttr[1]).getInt() : 16;

      auto gvA = findGlobalVar(argIdxA);
      auto gvB = findGlobalVar(argIdxB);
      auto gvC = findGlobalVar(argIdxC);

      if (!gvA || !gvB || !gvC) {
        funcOp.emitError("Could not find GlobalVariables for coop matrix args");
        return signalPassFailure();
      }

      for (auto callOp : coopCalls) {
        OpBuilder cmBuilder(callOp);
        auto cLoc = callOp.getLoc();

        auto f16Type = cmBuilder.getF16Type();
        auto f32Type = cmBuilder.getF32Type();
        auto subgroupScope = spirv::Scope::Subgroup;

        auto coopTypeA = spirv::CooperativeMatrixType::get(
            f16Type, 16, 16, subgroupScope,
            spirv::CooperativeMatrixUseKHR::MatrixA);
        auto coopTypeB = spirv::CooperativeMatrixType::get(
            f16Type, 16, 16, subgroupScope,
            spirv::CooperativeMatrixUseKHR::MatrixB);
        auto coopTypeAcc = spirv::CooperativeMatrixType::get(
            f32Type, 16, 16, subgroupScope,
            spirv::CooperativeMatrixUseKHR::MatrixAcc);
        auto coopTypeResult = spirv::CooperativeMatrixType::get(
            f16Type, 16, 16, subgroupScope,
            spirv::CooperativeMatrixUseKHR::MatrixAcc);

        // Get StorageBuffer pointers via AddressOf → AccessChain[0][0]
        auto cst0 = cmBuilder.create<spirv::ConstantOp>(
            cLoc, cmBuilder.getI32Type(), cmBuilder.getI32IntegerAttr(0));

        auto getStorageBufferElemPtr = [&](spirv::GlobalVariableOp gv) -> Value {
          auto addrOf = cmBuilder.create<spirv::AddressOfOp>(cLoc, gv);
          auto elemPtrType = spirv::PointerType::get(
              f16Type, spirv::StorageClass::StorageBuffer);
          return cmBuilder.create<spirv::AccessChainOp>(
              cLoc, elemPtrType, addrOf, ValueRange{cst0, cst0});
        };

        auto aPtrElem = getStorageBufferElemPtr(gvA);
        auto bPtrElem = getStorageBufferElemPtr(gvB);
        auto cPtrElem = getStorageBufferElemPtr(gvC);

        auto strideVal = cmBuilder.create<spirv::ConstantOp>(
            cLoc, cmBuilder.getI32Type(),
            cmBuilder.getI32IntegerAttr(stride));
        auto layoutAttr = spirv::CooperativeMatrixLayoutKHRAttr::get(
            ctx, spirv::CooperativeMatrixLayoutKHR::RowMajor);

        // Load A and B from StorageBuffer
        auto loadA = spirv::KHRCooperativeMatrixLoadOp::create(
            cmBuilder, cLoc, coopTypeA, aPtrElem, layoutAttr, strideVal,
            /*memory_operand=*/nullptr, /*alignment=*/nullptr);
        auto loadB = spirv::KHRCooperativeMatrixLoadOp::create(
            cmBuilder, cLoc, coopTypeB, bPtrElem, layoutAttr, strideVal,
            /*memory_operand=*/nullptr, /*alignment=*/nullptr);

        // Zero accumulator
        auto zeroF32 = cmBuilder.create<spirv::ConstantOp>(
            cLoc, f32Type, cmBuilder.getF32FloatAttr(0.0f));
        auto zeroAcc = cmBuilder.create<spirv::CompositeConstructOp>(
            cLoc, coopTypeAcc, ValueRange{zeroF32});

        // MulAdd: result = A * B + 0
        auto mulAdd = spirv::KHRCooperativeMatrixMulAddOp::create(
            cmBuilder, cLoc, coopTypeAcc,
            loadA.getResult(), loadB.getResult(), zeroAcc,
            /*matrix_operands=*/nullptr);

        // Convert f32 accumulator to f16 for storage
        auto narrowed = cmBuilder.create<spirv::FConvertOp>(
            cLoc, coopTypeResult, mulAdd.getResult());

        // Store to StorageBuffer directly
        spirv::KHRCooperativeMatrixStoreOp::create(
            cmBuilder, cLoc, cPtrElem, narrowed, layoutAttr, strideVal,
            /*memory_operand=*/nullptr, /*alignment=*/nullptr);

        callOp.erase();
      }
    }
    // Remove coop matmul function declaration
    if (auto decl = dyn_cast_or_null<spirv::FuncOp>(
            spirvModule.lookupSymbol("__vulkan_coop_matmul"))) {
      decl.erase();
    }

    // Build new function type with NO args (all via globals/push constants/builtins)
    auto newFuncType = FunctionType::get(ctx, {}, {});
    funcOp.setFunctionType(newFuncType);

    // Remove ALL args from block (in reverse)
    while (entryBlock.getNumArguments() > 0)
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);

    // Set VCE triple for serialization (after all replacements are done).
    // Conditionally add capabilities for subgroup ops and cooperative matrix.
    {
      SmallVector<spirv::Capability> caps = {spirv::Capability::Shader};
      SmallVector<spirv::Extension> exts = {
          spirv::Extension::SPV_KHR_storage_buffer_storage_class};
      auto spirvVersion = spirv::Version::V_1_0;
      if (hasSubgroupOps) {
        spirvVersion = spirv::Version::V_1_3;
        caps.push_back(spirv::Capability::GroupNonUniform);
        caps.push_back(spirv::Capability::GroupNonUniformArithmetic);
      }
      if (hasCoopMatmul) {
        if (spirvVersion < spirv::Version::V_1_6)
          spirvVersion = spirv::Version::V_1_6;
        caps.push_back(spirv::Capability::Float16);
        caps.push_back(spirv::Capability::StorageBuffer16BitAccess);
        caps.push_back(spirv::Capability::CooperativeMatrixKHR);
        exts.push_back(spirv::Extension::SPV_KHR_16bit_storage);
        exts.push_back(spirv::Extension::SPV_KHR_cooperative_matrix);
      }
      auto triple = spirv::VerCapExtAttr::get(
          spirvVersion, caps, exts, ctx);
      spirvModule.setVceTripleAttr(triple);
    }

    // Add spirv.EntryPoint (interfaceVarRefs already populated above)
    builder.setInsertionPointAfter(funcOp);
    {
      OperationState state(loc, spirv::EntryPointOp::getOperationName());
      state.addAttribute("execution_model",
          spirv::ExecutionModelAttr::get(ctx, spirv::ExecutionModel::GLCompute));
      state.addAttribute("fn", SymbolRefAttr::get(ctx, kernelName));
      state.addAttribute("interface", builder.getArrayAttr(interfaceVarRefs));
      builder.create(state);
    }

    // Add ExecutionMode: LocalSize from function attribute or default 1,1,1
    {
      SmallVector<int32_t, 3> localSize = {1, 1, 1};
      // Check if the original func had vulkan.local_size attribute
      // (set by ConvertReductionToParallel for reduction kernels)
      if (auto lsAttr = moduleOp->getAttrOfType<ArrayAttr>("vulkan.local_size")) {
        for (unsigned i = 0; i < 3 && i < lsAttr.size(); i++)
          localSize[i] = cast<IntegerAttr>(lsAttr[i]).getInt();
      }

      OperationState state(loc, spirv::ExecutionModeOp::getOperationName());
      state.addAttribute("fn", SymbolRefAttr::get(ctx, kernelName));
      state.addAttribute("execution_mode",
          spirv::ExecutionModeAttr::get(ctx, spirv::ExecutionMode::LocalSize));
      state.addAttribute("values", builder.getI32ArrayAttr(localSize));
      builder.create(state);
    }

    // Remove the old spirv.func from outer module (it's now in spirv.module)
    // The outer module should only contain the spirv.module.
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createVulkanizePass() {
  return std::make_unique<VulkanizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createPrepareSPIRVPass() {
  return std::make_unique<PrepareSPIRVPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertReductionToParallelPass() {
  return std::make_unique<ConvertReductionToParallel>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertMatmulToCooperativePass() {
  return std::make_unique<ConvertMatmulToCooperative>();
}

} // namespace vulkan
} // namespace triton
} // namespace mlir
