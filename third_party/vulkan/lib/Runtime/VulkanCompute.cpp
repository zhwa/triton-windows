#include "VulkanCompute.h"
#include <cstring>
#include <algorithm>

// ============================================================================
// Construction / Destruction
// ============================================================================

VulkanCompute::VulkanCompute() {
    initInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
}

VulkanCompute::~VulkanCompute() {
    destroyShaderState();

    if (commandPool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, commandPool_, nullptr);
    if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);
    if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);
}

// ============================================================================
// Vulkan Initialization (done once)
// ============================================================================

void VulkanCompute::initInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Triton-Vulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "VulkanCompute";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    vkCheck(vkCreateInstance(&createInfo, nullptr, &instance_),
            "Failed to create Vulkan instance");
}

void VulkanCompute::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    // Pick first device with a compute queue
    for (auto& dev : devices) {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice_ = dev;
                computeQueueFamily_ = i;

                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(dev, &props);
                deviceName_ = props.deviceName;

                // Query subgroup size (Vulkan 1.1+)
                VkPhysicalDeviceSubgroupProperties subgroupProps{};
                subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
                VkPhysicalDeviceProperties2 props2{};
                props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                props2.pNext = &subgroupProps;
                vkGetPhysicalDeviceProperties2(dev, &props2);
                subgroupSize_ = subgroupProps.subgroupSize;

                return;
            }
        }
    }

    throw std::runtime_error("No GPU with compute queue found");
}

void VulkanCompute::createLogicalDevice() {
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamily_;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Enable device extensions for cooperative matrix and 16-bit storage
    std::vector<const char*> deviceExtensions = {
        "VK_KHR_cooperative_matrix",
        "VK_KHR_16bit_storage",
        "VK_KHR_shader_float16_int8",
    };

    // Enable cooperative matrix + f16 features
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures{};
    coopFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopFeatures.cooperativeMatrix = VK_TRUE;

    VkPhysicalDevice16BitStorageFeatures f16Features{};
    f16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    f16Features.storageBuffer16BitAccess = VK_TRUE;
    f16Features.pNext = &coopFeatures;

    VkPhysicalDeviceShaderFloat16Int8Features f16ShaderFeatures{};
    f16ShaderFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    f16ShaderFeatures.shaderFloat16 = VK_TRUE;
    f16ShaderFeatures.pNext = &f16Features;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    deviceCreateInfo.pNext = &f16ShaderFeatures;

    vkCheck(vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &device_),
            "Failed to create logical device");

    vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);
}

void VulkanCompute::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    vkCheck(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_),
            "Failed to create command pool");
}

// ============================================================================
// Shader Loading
// ============================================================================

void VulkanCompute::loadShader(const std::vector<uint32_t>& spirvBinary,
                               const std::string& entryPoint) {
    // Clean up previous shader state if any
    destroyShaderState();

    entryPoint_ = entryPoint;

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvBinary.size() * sizeof(uint32_t);
    createInfo.pCode = spirvBinary.data();

    vkCheck(vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule_),
            "Failed to create shader module");
}

// ============================================================================
// Workgroup Configuration
// ============================================================================

void VulkanCompute::setWorkgroups(uint32_t x, uint32_t y, uint32_t z) {
    groupCountX_ = x;
    groupCountY_ = y;
    groupCountZ_ = z;
}

// ============================================================================
// Buffer Management
// ============================================================================

size_t VulkanCompute::createBuffer(uint32_t binding, size_t sizeBytes) {
    BufferInfo info;
    info.binding = binding;
    info.size = sizeBytes;

    // Try device-local memory for the storage buffer (fast VRAM on discrete GPUs).
    // The buffer needs TRANSFER_DST (for writeBuffer) and TRANSFER_SRC (for readBuffer).
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCheck(vkCreateBuffer(device_, &bufferInfo, nullptr, &info.buffer),
            "Failed to create buffer");

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device_, info.buffer, &memReqs);

    // Try device-local (non-host-visible) memory first
    int32_t deviceLocalIdx = findMemoryTypeFallback(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (deviceLocalIdx >= 0) {
        // Device-local memory available — use it + create staging buffer
        info.deviceLocal = true;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = static_cast<uint32_t>(deviceLocalIdx);

        vkCheck(vkAllocateMemory(device_, &allocInfo, nullptr, &info.memory),
                "Failed to allocate device-local buffer memory");
        vkCheck(vkBindBufferMemory(device_, info.buffer, info.memory, 0),
                "Failed to bind device-local buffer memory");

        // Create host-visible staging buffer for transfers
        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = sizeBytes;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCheck(vkCreateBuffer(device_, &stagingInfo, nullptr, &info.staging),
                "Failed to create staging buffer");

        VkMemoryRequirements stagingReqs;
        vkGetBufferMemoryRequirements(device_, info.staging, &stagingReqs);

        VkMemoryAllocateInfo stagingAlloc{};
        stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        stagingAlloc.allocationSize = stagingReqs.size;
        stagingAlloc.memoryTypeIndex = findMemoryType(
            stagingReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkCheck(vkAllocateMemory(device_, &stagingAlloc, nullptr, &info.stagingMemory),
                "Failed to allocate staging buffer memory");
        vkCheck(vkBindBufferMemory(device_, info.staging, info.stagingMemory, 0),
                "Failed to bind staging buffer memory");
    } else {
        // No device-local memory (integrated GPU) — fall back to host-visible
        info.deviceLocal = false;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType(
            memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkCheck(vkAllocateMemory(device_, &allocInfo, nullptr, &info.memory),
                "Failed to allocate buffer memory");
        vkCheck(vkBindBufferMemory(device_, info.buffer, info.memory, 0),
                "Failed to bind buffer memory");
    }

    size_t index = buffers_.size();
    buffers_.push_back(info);
    return index;
}

void VulkanCompute::writeBuffer(size_t bufferIndex, const void* data, size_t sizeBytes) {
    if (bufferIndex >= buffers_.size()) {
        throw std::runtime_error("Invalid buffer index");
    }
    auto& buf = buffers_[bufferIndex];
    if (sizeBytes > buf.size) {
        throw std::runtime_error("Write size exceeds buffer size");
    }

    if (buf.deviceLocal) {
        // Write to staging buffer, then copy to device-local
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device_, buf.stagingMemory, 0, sizeBytes, 0, &mapped),
                "Failed to map staging buffer for write");
        std::memcpy(mapped, data, sizeBytes);
        vkUnmapMemory(device_, buf.stagingMemory);
        copyBuffer(buf.staging, buf.buffer, sizeBytes);
    } else {
        // Direct map (integrated GPU fallback)
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device_, buf.memory, 0, sizeBytes, 0, &mapped),
                "Failed to map buffer memory for write");
        std::memcpy(mapped, data, sizeBytes);
        vkUnmapMemory(device_, buf.memory);
    }
}

void VulkanCompute::readBuffer(size_t bufferIndex, void* data, size_t sizeBytes) {
    if (bufferIndex >= buffers_.size()) {
        throw std::runtime_error("Invalid buffer index");
    }
    auto& buf = buffers_[bufferIndex];
    if (sizeBytes > buf.size) {
        throw std::runtime_error("Read size exceeds buffer size");
    }

    if (buf.deviceLocal) {
        // Copy from device-local to staging, then read
        copyBuffer(buf.buffer, buf.staging, sizeBytes);
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device_, buf.stagingMemory, 0, sizeBytes, 0, &mapped),
                "Failed to map staging buffer for read");
        std::memcpy(data, mapped, sizeBytes);
        vkUnmapMemory(device_, buf.stagingMemory);
    } else {
        void* mapped = nullptr;
        vkCheck(vkMapMemory(device_, buf.memory, 0, sizeBytes, 0, &mapped),
                "Failed to map buffer memory for read");
        std::memcpy(data, mapped, sizeBytes);
        vkUnmapMemory(device_, buf.memory);
    }
}

// ============================================================================
// Pipeline Build + Dispatch
// ============================================================================

void VulkanCompute::buildPipeline() {
    // 1. Create descriptor set layout from buffer bindings
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    for (auto& buf : buffers_) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = buf.binding;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutBindings.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();

    vkCheck(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_),
            "Failed to create descriptor set layout");

    // 2. Create pipeline layout (with push constants if needed)
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;

    VkPushConstantRange pushConstRange{};
    if (!pushConstantData_.empty()) {
        pushConstRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstRange.offset = 0;
        pushConstRange.size = static_cast<uint32_t>(pushConstantData_.size());
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstRange;
    }

    vkCheck(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_),
            "Failed to create pipeline layout");

    // 3. Create compute pipeline
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule_;
    stageInfo.pName = entryPoint_.c_str();

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout_;

    vkCheck(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_),
            "Failed to create compute pipeline");

    // 4. Create descriptor pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(buffers_.size());

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    vkCheck(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_),
            "Failed to create descriptor pool");

    // 5. Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;

    vkCheck(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_),
            "Failed to allocate descriptor set");

    // 6. Update descriptor set with buffer bindings
    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers_.size());
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers_.size());

    for (size_t i = 0; i < buffers_.size(); ++i) {
        bufferInfos[i].buffer = buffers_[i].buffer;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = buffers_[i].size;

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].pNext = nullptr;
        descriptorWrites[i].dstSet = descriptorSet_;
        descriptorWrites[i].dstBinding = buffers_[i].binding;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSets(device_,
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);
}

void VulkanCompute::setPushConstants(const void* data, size_t sizeBytes) {
    pushConstantData_.resize(sizeBytes);
    std::memcpy(pushConstantData_.data(), data, sizeBytes);
    // Force full pipeline rebuild — push constant range change requires
    // pipeline layout rebuild. Destroy all dependent state to avoid leaks.
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
        descriptorSet_ = VK_NULL_HANDLE;  // implicitly freed with pool
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        descriptorSetLayout_ = VK_NULL_HANDLE;
    }
}

void VulkanCompute::dispatch() {
    // Build pipeline only if not already built (supports re-dispatch)
    if (pipeline_ == VK_NULL_HANDLE)
        buildPipeline();

    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkCheck(vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer),
            "Failed to allocate command buffer");

    // Record commands
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkCheck(vkBeginCommandBuffer(commandBuffer, &beginInfo),
            "Failed to begin command buffer");

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
    if (!pushConstantData_.empty()) {
        vkCmdPushConstants(commandBuffer, pipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(pushConstantData_.size()),
                           pushConstantData_.data());
    }
    vkCmdDispatch(commandBuffer, groupCountX_, groupCountY_, groupCountZ_);

    vkCheck(vkEndCommandBuffer(commandBuffer),
            "Failed to end command buffer");

    // Submit and wait
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkCheck(vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE),
            "Failed to submit compute work");

    vkCheck(vkQueueWaitIdle(computeQueue_),
            "Failed waiting for compute work to complete");

    // Free command buffer
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
}

// ============================================================================
// Utilities
// ============================================================================

std::string VulkanCompute::getDeviceName() const {
    return deviceName_;
}

uint32_t VulkanCompute::getSubgroupSize() const {
    return subgroupSize_;
}

uint32_t VulkanCompute::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

int32_t VulkanCompute::findMemoryTypeFallback(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            // Skip memory types that are also host-visible — those are the
            // PCIe BAR region on discrete GPUs, not true device-local VRAM.
            if (properties == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT &&
                (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
                continue;
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

void VulkanCompute::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuf;
    vkCheck(vkAllocateCommandBuffers(device_, &allocInfo, &cmdBuf),
            "Failed to allocate transfer command buffer");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkCheck(vkBeginCommandBuffer(cmdBuf, &beginInfo),
            "Failed to begin transfer command buffer");

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmdBuf, src, dst, 1, &copyRegion);

    vkCheck(vkEndCommandBuffer(cmdBuf),
            "Failed to end transfer command buffer");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    vkCheck(vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE),
            "Failed to submit transfer command");
    vkCheck(vkQueueWaitIdle(computeQueue_),
            "Failed waiting for transfer to complete");

    vkFreeCommandBuffers(device_, commandPool_, 1, &cmdBuf);
}

void VulkanCompute::resetShaderState() {
    destroyShaderState();
}

void VulkanCompute::destroyShaderState() {
    // Wait for device to be idle before destroying
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }

    // Destroy pipeline
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
        descriptorSet_ = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        descriptorSetLayout_ = VK_NULL_HANDLE;
    }

    // Destroy buffers (device-local + staging)
    for (auto& buf : buffers_) {
        if (buf.staging != VK_NULL_HANDLE)
            vkDestroyBuffer(device_, buf.staging, nullptr);
        if (buf.stagingMemory != VK_NULL_HANDLE)
            vkFreeMemory(device_, buf.stagingMemory, nullptr);
        if (buf.buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(device_, buf.buffer, nullptr);
        if (buf.memory != VK_NULL_HANDLE)
            vkFreeMemory(device_, buf.memory, nullptr);
    }
    buffers_.clear();

    // Destroy shader module
    if (shaderModule_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, shaderModule_, nullptr);
        shaderModule_ = VK_NULL_HANDLE;
    }
}
