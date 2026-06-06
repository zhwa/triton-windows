//===-- triton_vulkan.cc - pybind11 module for Vulkan backend passes ------===//
//
// Exposes TritonToLinalg and standard MLIR passes to Python.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Target/SPIRV/Serialization.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"

#include "Conversion/TritonToLinalg.h"

#ifdef HAVE_VULKAN_RUNTIME
#include "VulkanCompute.h"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_vulkan_passes_ttir_to_linalg(py::module &&m) {
  m.def("triton_to_linalg", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::vulkan::createTritonToLinalgPass());
  });
}

void init_triton_vulkan_passes_linalg_to_memref(py::module &&m) {
  m.def("one_shot_bufferize", [](mlir::PassManager &pm) {
    mlir::bufferization::OneShotBufferizePassOptions opts;
    opts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
  });
  m.def("convert_reduction_to_parallel", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::vulkan::createConvertReductionToParallelPass());
  });
  m.def("convert_linalg_to_loops", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertLinalgToLoopsPass());
  });
  m.def("lower_affine", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createLowerAffinePass());
  });
  m.def("convert_scf_to_cf", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createSCFToControlFlowPass());
  });
}

void init_triton_vulkan_passes_spirv(py::module &&m) {
  m.def("prepare_spirv", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::vulkan::createPrepareSPIRVPass());
  });
  m.def("lower_scf_to_cf", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createSCFToControlFlowPass());
  });
  m.def("map_storage_class", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createMapMemRefStorageClassPass());
  });
  m.def("convert_memref_to_spirv", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertMemRefToSPIRVPass());
  });
  m.def("convert_arith_to_spirv", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertArithToSPIRVPass());
  });
  m.def("convert_math_to_spirv", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertMathToSPIRVPass());
  });
  m.def("convert_cf_to_spirv", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertControlFlowToSPIRVPass());
  });
  m.def("convert_func_to_spirv", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertFuncToSPIRVPass());
  });
  m.def("convert_gpu_to_spirv", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertGPUToSPIRVPass());
  });
  m.def("fix_alloca_storage_class", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::vulkan::createFixAllocaStorageClassPass());
  });
  m.def("vulkanize", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::vulkan::createVulkanizePass());
  });
}

// Serialize a spirv.module op to a SPIR-V binary (bytes).
static py::bytes serialize_spirv_module(mlir::ModuleOp module) {
  // Find the spirv.module inside the outer module
  mlir::spirv::ModuleOp spirvModule;
  module.walk([&](mlir::spirv::ModuleOp op) { spirvModule = op; });

  if (!spirvModule) {
    throw std::runtime_error("No spirv.module found in the IR");
  }

  llvm::SmallVector<uint32_t, 0> binary;
  if (mlir::failed(mlir::spirv::serialize(spirvModule, binary))) {
    throw std::runtime_error("Failed to serialize SPIR-V module");
  }

  return py::bytes(reinterpret_cast<const char *>(binary.data()),
                   binary.size() * sizeof(uint32_t));
}

void init_triton_vulkan(py::module &&m) {
  m.doc() = "Vulkan/SPIR-V backend for Triton";
  auto passes = m.def_submodule("passes");
  init_triton_vulkan_passes_ttir_to_linalg(passes.def_submodule("linalg"));
  init_triton_vulkan_passes_linalg_to_memref(passes.def_submodule("memref"));
  init_triton_vulkan_passes_spirv(passes.def_submodule("spirv"));

  m.def("serialize_spirv", &serialize_spirv_module,
        "Serialize spirv.module to SPIR-V binary bytes");

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
    registry.insert<mlir::spirv::SPIRVDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

#ifdef HAVE_VULKAN_RUNTIME
  // Vulkan compute runtime — dispatch SPIR-V shaders on GPU
  auto vk = m.def_submodule("runtime", "Vulkan compute dispatch");

  py::class_<VulkanCompute>(vk, "VulkanCompute")
      .def(py::init<>())
      .def("device_name", &VulkanCompute::getDeviceName)
      .def("load_shader",
           [](VulkanCompute &vc, py::bytes spirv_binary,
              const std::string &entry_point) {
             std::string data = spirv_binary;
             if (data.size() % 4 != 0)
               throw std::runtime_error(
                   "SPIR-V binary size must be a multiple of 4");
             std::vector<uint32_t> words(data.size() / 4);
             std::memcpy(words.data(), data.data(), data.size());
             vc.loadShader(words, entry_point);
           },
           py::arg("spirv_binary"), py::arg("entry_point") = "main")
      .def("set_workgroups", &VulkanCompute::setWorkgroups, py::arg("x"),
           py::arg("y") = 1, py::arg("z") = 1)
      .def("create_buffer", &VulkanCompute::createBuffer, py::arg("binding"),
           py::arg("size_bytes"))
      .def("write_buffer",
           [](VulkanCompute &vc, size_t buf_idx, py::buffer data) {
             py::buffer_info info = data.request();
             vc.writeBuffer(buf_idx, info.ptr,
                            static_cast<size_t>(info.size * info.itemsize));
           })
      .def("read_buffer",
           [](VulkanCompute &vc, size_t buf_idx, py::buffer data) {
             py::buffer_info info = data.request(true);
             vc.readBuffer(buf_idx, info.ptr,
                           static_cast<size_t>(info.size * info.itemsize));
           })
      .def("set_push_constants",
           [](VulkanCompute &vc, py::buffer data) {
             py::buffer_info info = data.request();
             vc.setPushConstants(info.ptr,
                                static_cast<size_t>(info.size * info.itemsize));
           })
      .def("dispatch", &VulkanCompute::dispatch)
      .def("reset", &VulkanCompute::resetShaderState);
#endif
}
