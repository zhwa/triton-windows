from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class VulkanDriver(DriverBase):

    def __init__(self):
        super().__init__()

    @staticmethod
    def is_active():
        # Phase 0: always inactive — we're not ready to run kernels yet.
        return False

    def get_current_target(self):
        return GPUTarget("vulkan", 0, 32)

    def get_active_torch_device(self):
        # Vulkan kernels don't use torch devices (yet)
        raise NotImplementedError("Vulkan backend does not support torch devices")

    def get_benchmarker(self):
        raise NotImplementedError("Vulkan backend benchmarking not implemented")

    def map_python_to_cpp_type(self, ty: str) -> str:
        mapping = {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u1": "uint32_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "fp64": "double",
        }
        if ty.startswith("*"):
            return "uint64_t"  # VkDeviceAddress
        return mapping.get(ty, "int32_t")
