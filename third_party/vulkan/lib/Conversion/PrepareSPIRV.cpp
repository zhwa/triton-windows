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

    // Set VCE triple for serialization
    auto triple = spirv::VerCapExtAttr::get(
        spirv::Version::V_1_0, {spirv::Capability::Shader},
        {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, ctx);
    spirvModule.setVceTripleAttr(triple);

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

    // Create push constant struct for scalar args
    spirv::GlobalVariableOp pushConstVar;
    if (!scalarArgIndices.empty()) {
      SmallVector<Type> memberTypes;
      for (auto idx : scalarArgIndices)
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

    // Replace arg uses: buffer args → addressof, scalar args → push constant access
    builder.setInsertionPointToStart(&entryBlock);
    for (unsigned i = 0; i < bufferArgIndices.size(); i++) {
      unsigned argIdx = bufferArgIndices[i];
      auto arg = entryBlock.getArgument(argIdx);
      auto addrOf = builder.create<spirv::AddressOfOp>(
          loc, globalVars[i]);
      arg.replaceAllUsesWith(addrOf.getResult());
    }

    // Replace scalar args with push constant loads
    if (pushConstVar) {
      auto pcAddrOf = builder.create<spirv::AddressOfOp>(loc, pushConstVar);
      for (unsigned i = 0; i < scalarArgIndices.size(); i++) {
        unsigned argIdx = scalarArgIndices[i];
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

    // Build new function type with NO args (all via globals/push constants)
    auto newFuncType = FunctionType::get(ctx, {}, {});
    funcOp.setFunctionType(newFuncType);

    // Remove ALL args from block (in reverse)
    while (entryBlock.getNumArguments() > 0)
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);

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

    // Add ExecutionMode: LocalSize 1,1,1
    {
      OperationState state(loc, spirv::ExecutionModeOp::getOperationName());
      state.addAttribute("fn", SymbolRefAttr::get(ctx, kernelName));
      state.addAttribute("execution_mode",
          spirv::ExecutionModeAttr::get(ctx, spirv::ExecutionMode::LocalSize));
      state.addAttribute("values", builder.getI32ArrayAttr({1, 1, 1}));
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

} // namespace vulkan
} // namespace triton
} // namespace mlir
