//===-- TritonToLinalgPass.cpp - Pass wrapper for TritonToLinalg -----------===//
//
// Vulkan/SPIR-V backend for Triton.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonToLinalg.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;

namespace {

/// Type converter: maps Triton pointer types to MemRef types.
class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addConversion([](TensorType tensorType) -> Type {
      auto elemType = tensorType.getElementType();
      if (auto ptrType = dyn_cast<triton::PointerType>(elemType))
        elemType = ptrType.getPointeeType();
      return MemRefType::get(tensorType.getShape(), elemType);
    });

    // Handle materialization when SplatConverter produces an unranked memref
    // but a consumer expects a ranked memref (e.g., AtomicRMWConverter).
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs,
                                Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      auto input = inputs[0];
      if (type == input.getType())
        return input;
      // Cast between memref types (unranked ↔ ranked)
      if ((isa<MemRefType>(type) || isa<UnrankedMemRefType>(type)) &&
          (isa<MemRefType>(input.getType()) ||
           isa<UnrankedMemRefType>(input.getType()))) {
        return builder.create<memref::CastOp>(loc, type, input).getResult();
      }
      // tensor → memref (needed when TypeConverter converts tensor<NxT> →
      // memref<NxT> for AtomicRMWOp's val operand)
      if (isa<MemRefType>(type) && isa<RankedTensorType>(input.getType())) {
        return builder.create<bufferization::ToBufferOp>(loc, type, input)
            .getResult();
      }
      return Value();
    });
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs,
                                Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      auto input = inputs[0];
      if (type == input.getType())
        return input;
      if ((isa<MemRefType>(type) || isa<UnrankedMemRefType>(type)) &&
          (isa<MemRefType>(input.getType()) ||
           isa<UnrankedMemRefType>(input.getType()))) {
        return builder.create<memref::CastOp>(loc, type, input).getResult();
      }
      // memref → tensor (reverse bridge)
      if (isa<RankedTensorType>(type) && isa<MemRefType>(input.getType())) {
        return builder.create<bufferization::ToTensorOp>(
                   loc, type, input, /*restrict=*/true, /*writable=*/true)
            .getResult();
      }
      return Value();
    });
  }
};

static constexpr uint32_t LAUNCH_GRID_RANK = 3; // X, Y, Z
static constexpr unsigned TRITON_PROGRAM_INFO_ARG_COUNT =
    LAUNCH_GRID_RANK * 2; // num_programs(3) + program_id(3)

/// Add 6 i32 args to a triton::FuncOp: num_programs (x,y,z) + program_id
/// (x,y,z). GetProgramIDConverter and GetNumProgramsConverter extract from
/// these.
static void addProgramInfo(triton::FuncOp func) {
  OpBuilder b(func);
  auto origType = func.getFunctionType();
  auto origInputs = origType.getInputs();
  SmallVector<Type> newInputs(origInputs);
  newInputs.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

  auto newType = b.getFunctionType(newInputs, origType.getResults());
  func.setFunctionType(newType);

  if (func.getAllArgAttrs()) {
    SmallVector<DictionaryAttr> newArgAttrs;
    func.getAllArgAttrs(newArgAttrs);
    newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
    func.setAllArgAttrs(newArgAttrs);
  }

  for (unsigned i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++)
    func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
}

//===----------------------------------------------------------------------===//
// TritonToLinalgPass
//===----------------------------------------------------------------------===//

class TritonToLinalgPass
    : public PassWrapper<TritonToLinalgPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonToLinalgPass)

  StringRef getArgument() const override { return "triton-to-linalg"; }
  StringRef getDescription() const override {
    return "Convert Triton IR to Linalg/Tensor/Arith dialects";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Set up conversion target
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonTypeConverter typeConverter;

    // Legal dialects (output)
    target.addLegalDialect<func::FuncDialect, arith::ArithDialect,
                           math::MathDialect, linalg::LinalgDialect,
                           affine::AffineDialect, scf::SCFDialect,
                           cf::ControlFlowDialect, tensor::TensorDialect,
                           bufferization::BufferizationDialect,
                           memref::MemRefDialect>();
    target.addLegalOp<ModuleOp>();

    // Mark specific Triton ops as illegal (must be converted)
    target.addIllegalOp<triton::SplatOp, triton::MakeRangeOp,
                        triton::BroadcastOp, triton::ExpandDimsOp,
                        triton::TransOp, triton::GetProgramIdOp,
                        triton::GetNumProgramsOp, triton::DotOp,
                        triton::ReduceOp, triton::BitcastOp,
                        triton::ReshapeOp,
                        triton::AddPtrOp, triton::LoadOp,
                        triton::StoreOp, triton::AtomicRMWOp>();

    // Dense splat constants must be lowered
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      if (!isa<RankedTensorType>(op.getType()))
        return true;
      auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
      if (denseAttr && denseAttr.isSplat() &&
          isa<FloatType, IntegerType>(denseAttr.getElementType()))
        return false;
      return true;
    });

    // Tensor arith/math ops → linalg.generic
    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          if (isa<arith::ConstantOp>(op))
            return true;
          return !llvm::all_of(op->getOperandTypes(), [](Type type) {
            return isa<RankedTensorType>(type);
          });
        });

    // Triton FuncOp: legal only after type conversion of signature
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addLegalOp<triton::ReturnOp>();

    // Add program info args before conversion
    for (auto func : moduleOp.getOps<triton::FuncOp>())
      addProgramInfo(func);

    // Populate patterns — includes function signature conversion
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        patterns, typeConverter);
    vulkan::populateTritonToLinalgConversionPatterns(typeConverter, patterns,
                                                     LAUNCH_GRID_RANK);

    if (failed(
            applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Convert tt.func/tt.return → func.func/func.return
    moduleOp.walk([&](triton::FuncOp func) {
      OpBuilder builder(func);
      auto name = func.getName();
      auto type = func.getFunctionType();

      SmallVector<DictionaryAttr> argAttrs, resAttrs;
      func.getAllArgAttrs(argAttrs);
      func.getAllResultAttrs(resAttrs);

      auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, type);
      funcFunc.setAllArgAttrs(argAttrs);
      funcFunc.setAllResultAttrs(resAttrs);

      auto &funcFuncBody = funcFunc.getBody();
      auto &funcBody = func.getBody();

      IRMapping map;
      funcBody.cloneInto(&funcFuncBody, map);

      for (Block &block : funcFuncBody.getBlocks()) {
        auto *term = block.getTerminator();
        builder.setInsertionPoint(term);
        builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
        term->erase();
      }
      func.erase();
    });

    // Clean up dead ops
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::vulkan::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
