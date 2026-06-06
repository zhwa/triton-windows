# Vulkan/SPIR-V backend for Triton
#
# Components:
#   - VulkanTritonToLinalg: 16 converters + bridge passes + VulkanizePass
#   - VulkanRuntime: Vulkan compute dispatch engine
#   - TritonVulkan: pybind11 module exposing passes + runtime

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
  MLIRGPUDialect
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

# Vulkan compute runtime — dispatches SPIR-V shaders on GPU
find_path(VULKAN_INCLUDE_DIR vulkan/vulkan.h
  PATHS "$ENV{VULKAN_SDK}/Include"
        "C:/ProgramData/anaconda3/envs/mlir-dev/Library/include"
  NO_DEFAULT_PATH
)
find_library(VULKAN_LIBRARY vulkan-1
  PATHS "$ENV{VULKAN_SDK}/Lib"
        "C:/ProgramData/anaconda3/envs/mlir-dev/Library/lib"
  NO_DEFAULT_PATH
)

if(VULKAN_INCLUDE_DIR AND VULKAN_LIBRARY)
  message(STATUS "Vulkan found: ${VULKAN_LIBRARY}")
  add_library(VulkanRuntime STATIC
    lib/Runtime/VulkanCompute.cpp
  )
  target_include_directories(VulkanRuntime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/Runtime
    ${VULKAN_INCLUDE_DIR}
  )
  target_link_libraries(VulkanRuntime PUBLIC ${VULKAN_LIBRARY})
  set(HAVE_VULKAN_RUNTIME TRUE)
else()
  message(WARNING "Vulkan SDK not found — VulkanRuntime disabled")
  set(HAVE_VULKAN_RUNTIME FALSE)
endif()

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
  MLIRMathToSPIRV
  MLIRGPUToSPIRV
)
target_link_libraries(TritonVulkan PRIVATE Python3::Module pybind11::headers)
if(HAVE_VULKAN_RUNTIME)
  target_link_libraries(TritonVulkan PRIVATE VulkanRuntime)
  target_compile_definitions(TritonVulkan PRIVATE HAVE_VULKAN_RUNTIME=1)
endif()
