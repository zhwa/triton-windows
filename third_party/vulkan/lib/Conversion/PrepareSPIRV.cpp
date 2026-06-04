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
    // Leave the address space as 0 so map-memref-spirv-storage-class
    // will NOT map it to StorageBuffer. The convert-memref-to-spirv pass
    // handles alloca with default address space → spirv.Variable(Function).
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

/// Build spirv.module + spirv.func manually, handling remaining memref.alloca
/// by converting them to spirv.GlobalVariable.
///
/// After convert-{memref,arith,cf}-to-spirv, we have:
///   func.func @kernel(%arg0: memref<?xf32, StorageBuffer>, ...) {
///     %cast_arg = unrealized_conversion_cast %arg0 → !spirv.ptr<...>
///     %alloca   = memref.alloca() : memref<256xf32, Workgroup>
///     %cast_buf = unrealized_conversion_cast %alloca → !spirv.ptr<...>
///     spirv.Branch / spirv.Load / spirv.Store / spirv.FAdd / ...
///   }
///
/// This pass creates:
///   spirv.module Logical GLSL450 {
///     spirv.GlobalVariable @buf : !spirv.ptr<struct<array<256xf32>>, Workgroup>
///     spirv.func @kernel(%arg0: !spirv.ptr<...>, ...) "None" {
///       %addr = spirv.mlir.addressof @buf
///       spirv.Branch / ...  (same body, with casts replaced)
///     }
///     spirv.EntryPoint "GLCompute" @kernel
///   }
class FinalizeSPIRVPass
    : public PassWrapper<FinalizeSPIRVPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeSPIRVPass)

  StringRef getArgument() const override { return "finalize-spirv"; }
  StringRef getDescription() const override {
    return "Build spirv.module from func.func with SPIR-V ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    // Find the kernel function
    func::FuncOp funcOp;
    moduleOp.walk([&](func::FuncOp f) { funcOp = f; });
    if (!funcOp) {
      llvm::errs() << "FinalizeSPIRV: no func.func found, skipping\n";
      return;
    }

    llvm::errs() << "FinalizeSPIRV: processing " << funcOp.getName() << "\n";

    OpBuilder builder(ctx);
    auto loc = funcOp.getLoc();

    // Get target env
    auto targetEnv = moduleOp->getAttrOfType<spirv::TargetEnvAttr>(
        spirv::getTargetEnvAttrName());

    // 1. Create spirv.module
    builder.setInsertionPointToEnd(moduleOp.getBody());
    auto spvModule = builder.create<spirv::ModuleOp>(
        loc, spirv::AddressingModel::Logical,
        spirv::MemoryModel::GLSL450);
    if (targetEnv)
      spvModule->setAttr(spirv::getTargetEnvAttrName(), targetEnv);

    builder.setInsertionPointToStart(spvModule.getBody());

    // 2. Collect alloca→cast mappings and create GlobalVariables
    llvm::DenseMap<Value, spirv::GlobalVariableOp> allocaToGlobal;
    int varIdx = 0;

    funcOp.walk([&](memref::AllocaOp alloca) {
      for (auto *user : alloca.getResult().getUsers()) {
        auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
        if (!cast)
          continue;

        auto spirvPtrType =
            dyn_cast<spirv::PointerType>(cast.getResultTypes()[0]);
        if (!spirvPtrType)
          continue;

        // Create GlobalVariable in the spirv.module
        std::string name = "__buf_" + std::to_string(varIdx++);
        auto typeAttr = TypeAttr::get(spirvPtrType.getPointeeType());
        auto nameAttr = builder.getStringAttr(name);
        auto globalVar = spirv::GlobalVariableOp::create(
            builder, loc, typeAttr, nameAttr);
        // The storage class is encoded in the pointer type, but we also
        // need it as an attribute for proper SPIR-V serialization.

        allocaToGlobal[cast.getResult(0)] = globalVar;
      }
    });

    // 3. Determine spirv.func argument types from the unrealized_conversion_casts
    SmallVector<Type> spirvArgTypes;
    SmallVector<Value> argCastResults; // cast results for each arg
    llvm::DenseMap<Value, unsigned> castToArgIdx;

    auto &entryBlock = funcOp.getBody().front();
    for (auto &op : entryBlock) {
      auto cast = dyn_cast<UnrealizedConversionCastOp>(&op);
      if (!cast)
        continue;
      // Check if this cast converts a function argument (memref → spirv.ptr)
      if (cast.getNumOperands() == 1 &&
          isa<BlockArgument>(cast.getOperand(0))) {
        auto blockArg = llvm::dyn_cast<BlockArgument>(cast.getOperand(0));
        if (blockArg && blockArg.getOwner() == &entryBlock) {
          spirvArgTypes.push_back(cast.getResultTypes()[0]);
          castToArgIdx[cast.getResult(0)] = spirvArgTypes.size() - 1;
          argCastResults.push_back(cast.getResult(0));
        }
      }
    }

    // Add remaining scalar args (i32 etc.) that don't have casts
    for (auto arg : entryBlock.getArguments()) {
      bool hasCast = false;
      for (auto *user : arg.getUsers()) {
        if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
          if (castToArgIdx.count(cast.getResult(0)))
            hasCast = true;
        }
      }
      if (!hasCast) {
        // This arg is used directly (e.g., i32 scalar)
        spirvArgTypes.push_back(arg.getType());
      }
    }

    // 4. Create spirv.func and MOVE the function body from func.func.
    // This avoids all cloning/mapping issues — we just take the existing
    // blocks and transplant them into the spirv.func.
    auto spvFuncType =
        FunctionType::get(ctx, spirvArgTypes, /*results=*/{});
    auto spvFunc = builder.create<spirv::FuncOp>(
        loc, funcOp.getName(), spvFuncType);
    spvFunc->setAttr("function_control",
        spirv::FunctionControlAttr::get(
            ctx, spirv::FunctionControl::None));

    // Move blocks from func.func into spirv.func (after auto-created entry)
    auto &funcBody = funcOp.getBody();
    auto &spvBody = spvFunc.getBody();
    spvBody.getBlocks().splice(spvBody.end(), funcBody.getBlocks());

    // Now erase the auto-created empty entry block (first block)
    spvBody.front().erase();

    llvm::errs() << "FinalizeSPIRV: moved body, fixing up types...\n";

    // 5. Fix up the moved body: replace casts and allocas in-place.
    auto &spvEntry = spvFunc.getBody().front();

    // First, map each old arg to the corresponding new arg by index.
    // Since we cloneInto, the cloned block args have the old (memref) types.
    // We need to update them to the spirv.func arg types.
    // Actually, cloneInto preserves types. The spirv.func was created with
    // spirvArgTypes, but the cloned entry block has the original memref arg types.
    // We need to merge: use the cloned blocks but with the correct arg types.

    // Simpler approach: just find and replace unrealized_conversion_cast ops
    // inside the spirv.func with the appropriate values.

    // Handle arg casts: the cloned casts take cloned block args (memref types)
    // and produce spirv.ptr types. We want the spirv.func args to BE the
    // spirv.ptr types, so we need to update the entry block arg types.

    // Replace arg casts: collect first, then process.
    SmallVector<std::pair<BlockArgument, UnrealizedConversionCastOp>> argCasts;
    for (auto &op : spvEntry) {
      auto castOp = dyn_cast<UnrealizedConversionCastOp>(&op);
      if (!castOp || castOp.getNumOperands() != 1)
        continue;
      auto blockArg = llvm::dyn_cast<BlockArgument>(castOp.getOperand(0));
      if (!blockArg || blockArg.getOwner() != &spvEntry)
        continue;
      argCasts.push_back({blockArg, castOp});
    }
    for (auto &[blockArg, castOp] : argCasts) {
      auto spirvType = castOp.getResultTypes()[0];
      blockArg.setType(spirvType);
      castOp.getResult(0).replaceAllUsesWith(blockArg);
      castOp.erase();
    }

    // Handle alloca casts: replace with spirv.mlir.addressof
    for (auto &block : spvFunc.getBody()) {
      for (auto &op : llvm::make_early_inc_range(block)) {
        auto allocaOp = dyn_cast<memref::AllocaOp>(&op);
        if (!allocaOp)
          continue;

        // Find the cast user
        for (auto *user :
             llvm::make_early_inc_range(allocaOp.getResult().getUsers())) {
          auto castOp = dyn_cast<UnrealizedConversionCastOp>(user);
          if (!castOp)
            continue;

          auto spirvPtrType =
              dyn_cast<spirv::PointerType>(castOp.getResultTypes()[0]);
          if (!spirvPtrType)
            continue;

          // Create GlobalVariable with correct storage class
          OpBuilder moduleBuilder(spvModule.getBody(),
                                  spvModule.getBody()->begin());
          std::string name = "__buf_" + std::to_string(varIdx++);
          auto typeAttr = TypeAttr::get(spirvPtrType);
          auto nameAttr = moduleBuilder.getStringAttr(name);
          auto globalVar = spirv::GlobalVariableOp::create(
              moduleBuilder, allocaOp.getLoc(), typeAttr, nameAttr);

          // Create addressof — returns same pointer type as global var
          OpBuilder funcBuilder(castOp);
          auto addressOf = funcBuilder.create<spirv::AddressOfOp>(
              allocaOp.getLoc(), spirvPtrType,
              SymbolRefAttr::get(ctx, name));
          castOp.getResult(0).replaceAllUsesWith(addressOf.getResult());
          if (castOp.use_empty())
            castOp.erase();
          else
            llvm::errs() << "WARNING: cast still has uses after replace! "
                         << "Cast type: " << castOp.getResultTypes()[0]
                         << ", AddressOf type: " << addressOf.getType()
                         << "\n";
        }
        if (allocaOp.use_empty())
          allocaOp.erase();
      }
    }

    // Replace func::ReturnOp with spirv::ReturnOp
    for (auto &block : spvFunc.getBody()) {
      auto *term = block.getTerminator();
      if (isa<func::ReturnOp>(term)) {
        OpBuilder b(term);
        b.create<spirv::ReturnOp>(term->getLoc());
        term->erase();
      }
    }

    // 6. Add EntryPoint
    builder.setInsertionPointToEnd(spvModule.getBody());
    builder.create<spirv::EntryPointOp>(
        loc, spirv::ExecutionModel::GLCompute,
        spvFunc, SmallVector<Attribute>());

    // 7. Leave the old func.func in place (body was moved)
    // funcOp.erase() would crash if it still has attribute uses.
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPrepareSPIRVPass() {
  return std::make_unique<PrepareSPIRVPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVModulePass() {
  return createPrepareSPIRVPass(); // Alias
}

std::unique_ptr<OperationPass<ModuleOp>> createFinalizeSPIRVPass() {
  return std::make_unique<FinalizeSPIRVPass>();
}

} // namespace vulkan
} // namespace triton
} // namespace mlir
