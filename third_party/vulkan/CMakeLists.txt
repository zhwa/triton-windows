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
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRAffineDialect
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRDialectUtils
)

set_property(GLOBAL APPEND PROPERTY TRITON_LIBS VulkanTritonToLinalg)

