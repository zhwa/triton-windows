# Vulkan/SPIR-V backend for Triton
#
# Phase 0.5: TritonToLinalg conversion pass
# Future phases will add:
#   - TritonToSPIRV conversion passes
#   - SPIR-V serialization target
#   - Vulkan runtime launcher

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(VulkanTritonToLinalg STATIC
  lib/Conversion/TritonToLinalg.cpp
  lib/Conversion/TritonToLinalgPass.cpp
  lib/Conversion/PrepareSPIRV.cpp
)

target_include_directories(VulkanTritonToLinalg PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(VulkanTritonToLinalg PUBLIC
  TritonIR
  MLIRIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSPIRVTransforms
  MLIRTensorDialect
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRDialectUtils
  MLIRArithToSPIRV
  MLIRControlFlowToSPIRV
  MLIRFuncToSPIRV
  MLIRMemRefToSPIRV
)

set_property(GLOBAL APPEND PROPERTY TRITON_LIBS VulkanTritonToLinalg)

# pybind11 module — exposes passes to Python as `vulkan.passes.linalg.*`
add_triton_plugin(TritonVulkan
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_vulkan.cc
  LINK_LIBS
  VulkanTritonToLinalg
  MLIRBufferizationTransforms
  MLIRArithTransforms
  MLIRLinalgTransforms
  MLIRTensorTransforms
  MLIRSCFTransforms
  MLIRMemRefTransforms
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRSPIRVSerialization
  MLIRSPIRVDialect
  MLIRSPIRVTransforms
)
target_link_libraries(TritonVulkan PRIVATE Python3::Module pybind11::headers)
