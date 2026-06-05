import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict
from types import ModuleType

from triton._C.libtriton import ir, passes
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
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["linalg"] = lambda src, metadata: self.make_linalg(src, metadata, options)
        stages["memref"] = lambda src, metadata: self.make_memref(src, metadata, options)
        stages["spirv"] = lambda src, metadata: self.make_spirv(src, metadata, options)
        stages["spv"] = lambda src, metadata: self.make_spv(src, metadata, options)

    def load_dialects(self, ctx):
        try:
            from triton._C.libtriton import vulkan
            vulkan.load_dialects(ctx)
        except ImportError:
            pass

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    @staticmethod
    def make_ttir(mod, metadata, opt):
        """TTIR optimization passes (shared with NVIDIA backend)."""
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_linalg(mod, metadata, opt):
        """Convert TTIR → Linalg/Tensor/MemRef dialects."""
        try:
            from triton._C.libtriton import vulkan
        except ImportError:
            raise RuntimeError(
                "Vulkan backend C++ passes not available. "
                "Rebuild triton with vulkan backend enabled."
            )
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        vulkan.passes.linalg.triton_to_linalg(pm)
        pm.run(mod, 'make_linalg')

        # Extract kernel name for metadata
        src = str(mod)
        names = re.findall(r'func\.func @(\w+)\(', src)
        if names:
            metadata["name"] = names[0]

        return mod

    @staticmethod
    def make_memref(mod, metadata, opt):
        """Lower Linalg/Tensor → MemRef + loops + control flow."""
        try:
            from triton._C.libtriton import vulkan
        except ImportError:
            raise RuntimeError("Vulkan backend C++ passes not available.")
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        vulkan.passes.memref.one_shot_bufferize(pm)
        vulkan.passes.memref.convert_linalg_to_loops(pm)
        vulkan.passes.memref.lower_affine(pm)
        vulkan.passes.memref.convert_scf_to_cf(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, 'make_memref')
        return mod

    @staticmethod
    def make_opencl(src, metadata, opt):
        """Emit OpenCL C source from fully-lowered MemRef IR.

        Alternative output path — not in the default pipeline but available
        for debugging or OpenCL-based execution via pyopencl.
        """
        from triton.backends.vulkan.emitter import emit_opencl
        mlir_text = str(src)
        opencl_src = emit_opencl(mlir_text)
        return opencl_src

    @staticmethod
    def make_memref_bufonly(mod, metadata, opt):
        """Bufferize only — keep linalg.generic for parallel emission."""
        try:
            from triton._C.libtriton import vulkan
        except ImportError:
            raise RuntimeError("Vulkan backend C++ passes not available.")
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        vulkan.passes.memref.one_shot_bufferize(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, 'make_memref_bufonly')
        return mod

    @staticmethod
    def make_opencl_parallel(src, metadata, opt):
        """Emit parallel OpenCL C from bufferized IR with linalg.generic ops.

        Each workitem processes one element using get_local_id(0).
        Returns OpenCL C source. Sets metadata['block_size'] for dispatch.
        """
        from triton.backends.vulkan.emitter_parallel import emit_opencl_parallel
        mlir_text = src.str_nodebug()
        opencl_src, block_size = emit_opencl_parallel(mlir_text)
        metadata['block_size'] = block_size
        return opencl_src

    @staticmethod
    def make_spirv(mod, metadata, opt):
        """Convert MemRef IR → SPIR-V dialect via MLIR conversion passes."""
        try:
            from triton._C.libtriton import vulkan
        except ImportError:
            raise RuntimeError("Vulkan backend C++ passes not available.")

        # Step 1: Prepare — expand reinterpret_cast/copy, alloc→alloca, target_env
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        vulkan.passes.spirv.prepare_spirv(pm)
        vulkan.passes.spirv.lower_scf_to_cf(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, 'make_spirv_prep')

        # Step 2: Map storage classes, then fix alloca storage classes
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        vulkan.passes.spirv.map_storage_class(pm)
        vulkan.passes.spirv.fix_alloca_storage_class(pm)
        pm.run(mod, 'make_spirv_map')

        # Step 3: Convert to SPIR-V
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        vulkan.passes.spirv.convert_memref_to_spirv(pm)
        vulkan.passes.spirv.convert_arith_to_spirv(pm)
        vulkan.passes.spirv.convert_math_to_spirv(pm)
        vulkan.passes.spirv.convert_cf_to_spirv(pm)
        vulkan.passes.spirv.convert_func_to_spirv(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, 'make_spirv_convert')

        # Step 4: Vulkanize — convert func args → GlobalVariables, wrap in spirv.module
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        vulkan.passes.spirv.vulkanize(pm)
        pm.run(mod, 'make_spirv_vulkanize')

        return mod

    @staticmethod
    def make_spv(mod, metadata, opt):
        """Serialize spirv.module to SPIR-V binary via C++ serializer."""
        from triton._C.libtriton import vulkan
        return vulkan.serialize_spirv(mod)
