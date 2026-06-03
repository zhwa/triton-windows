import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget


@dataclass(frozen=True)
class VulkanOptions:
    backend_name: str = "vulkan"
    num_warps: int = 1
    num_stages: int = 1
    num_ctas: int = 1
    cluster_dims: tuple = (1, 1, 1)
    extern_libs: dict = None
    debug: bool = False

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join(f"{name}-{val}" for name, val in sorted(hash_dict.items()))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class VulkanBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "vulkan"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "spv"

    def hash(self) -> str:
        return f"vulkan-{self.target.arch}-{self.target.warp_size}"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in VulkanOptions.__dataclass_fields__.keys() if k in opts}
        return VulkanOptions(**args)

    def add_stages(self, stages, options, language=None):
        # Phase 0 stub — no stages yet.
        # Future pipeline: ttir → linalg → spirv → spv binary
        pass

    def load_dialects(self, ctx):
        # Will load SPIR-V dialect once C++ bindings exist
        pass

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
