#ifndef TRITON_VULKAN_CONVERSION_TRITONTOLINALG_H
#define TRITON_VULKAN_CONVERSION_TRITONTOLINALG_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace vulkan {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgPass();

std::unique_ptr<OperationPass<ModuleOp>> createPrepareSPIRVPass();

std::unique_ptr<OperationPass<ModuleOp>> createFixAllocaStorageClassPass();

std::unique_ptr<OperationPass<ModuleOp>> createVulkanizePass();

void populateTritonToLinalgConversionPatterns(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns,
                                              unsigned int launchGridRank);

} // namespace vulkan
} // namespace triton
} // namespace mlir

#endif // TRITON_VULKAN_CONVERSION_TRITONTOLINALG_H
