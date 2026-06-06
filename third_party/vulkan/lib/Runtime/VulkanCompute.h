#pragma once

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

/// VulkanCompute: A minimal Vulkan compute dispatch engine.
///
/// Replaces mlir::ExecutionEngine for SPIR-V compute shaders.
/// Manages Vulkan device, buffers, and compute pipeline.
///
/// Usage:
///   VulkanCompute vc;
///   vc.loadShader(spirvBinary, "main");
///   vc.setWorkgroups(numGroups);
///   auto bufA = vc.createBuffer(0, sizeBytes);
///   vc.writeBuffer(bufA, data, sizeBytes);
///   vc.dispatch();
///   vc.readBuffer(bufA, output, sizeBytes);
class VulkanCompute {
public:
    VulkanCompute();
    ~VulkanCompute();

    // Non-copyable
    VulkanCompute(const VulkanCompute&) = delete;
    VulkanCompute& operator=(const VulkanCompute&) = delete;

    /// Load a SPIR-V compute shader from binary data.
    void loadShader(const std::vector<uint32_t>& spirvBinary,
                    const std::string& entryPoint = "main");

    /// Set workgroup count for dispatch.
    void setWorkgroups(uint32_t x, uint32_t y = 1, uint32_t z = 1);

    /// Create a storage buffer bound to a descriptor set binding.
    /// Returns a buffer index for later reference.
    size_t createBuffer(uint32_t binding, size_t sizeBytes);

    /// Write data from host to a GPU buffer.
    void writeBuffer(size_t bufferIndex, const void* data, size_t sizeBytes);

    /// Read data from a GPU buffer to host.
    void readBuffer(size_t bufferIndex, void* data, size_t sizeBytes);

    /// Set push constant data (for scalar kernel args like N, block_size, pid).
    void setPushConstants(const void* data, size_t sizeBytes);

    /// Build pipeline and execute the compute shader.
    void dispatch();

    /// Get device name for display.
    std::string getDeviceName() const;

    /// Clean up per-shader state (buffers, pipeline) for reuse.
    void resetShaderState();

private:
    // Vulkan instance/device (created once)
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamily_ = 0;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::string deviceName_;

    // Per-shader state
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    std::string entryPoint_;

    // Buffer management
    struct BufferInfo {
        VkBuffer buffer = VK_NULL_HANDLE;        // device-local storage buffer
        VkDeviceMemory memory = VK_NULL_HANDLE;   // device-local memory
        VkBuffer staging = VK_NULL_HANDLE;        // host-visible staging buffer
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        size_t size = 0;
        uint32_t binding = 0;
        bool deviceLocal = false;  // true if buffer uses device-local memory
    };
    std::vector<BufferInfo> buffers_;

    // Push constant data
    std::vector<uint8_t> pushConstantData_;

    // Dispatch dimensions
    uint32_t groupCountX_ = 1, groupCountY_ = 1, groupCountZ_ = 1;

    // Helpers
    void initInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void destroyShaderState();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    int32_t findMemoryTypeFallback(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    void buildPipeline();
};

/// Helper: throw on Vulkan error
inline void vkCheck(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " (VkResult=" + std::to_string(result) + ")");
    }
}
