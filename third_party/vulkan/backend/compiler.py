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
        vulkan.passes.spirv.convert_cf_to_spirv(pm)
        vulkan.passes.spirv.convert_func_to_spirv(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, 'make_spirv_convert')
        return mod

    @staticmethod
    def make_spv(mod, metadata, opt):
        """Serialize to SPIR-V binary via mlir-opt + mlir-translate tools."""
        # Find MLIR tools — search common locations
        import os
        import subprocess
        import sys
        import tempfile

        ext = ".exe" if sys.platform == "win32" else ""
        mlir_opt = None
        mlir_translate = None

        # Search paths
        search_roots = []
        # From triton source tree
        for base in [os.path.dirname(os.path.dirname(os.path.dirname(
                         os.path.dirname(os.path.abspath(__file__))))),
                     os.path.dirname(os.path.dirname(os.path.dirname(
                         os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))]:
            candidate = os.path.join(base, "build", "llvm-project", "build", "bin")
            if os.path.isdir(candidate):
                search_roots.append(candidate)

        for root in search_roots:
            opt = os.path.join(root, f"mlir-opt{ext}")
            trans = os.path.join(root, f"mlir-translate{ext}")
            if os.path.exists(opt) and os.path.exists(trans):
                mlir_opt, mlir_translate = opt, trans
                break

        if not mlir_opt:
            raise RuntimeError(
                "mlir-opt/mlir-translate not found. "
                "Searched: " + str(search_roots))

        # Write the SPIR-V IR, wrapping spirv.func in spirv.module
        ir_text = mod.str_nodebug()

        # The IR has spirv.func inside builtin.module. We need it inside
        # spirv.module for serialization. Extract via brace-matching.
        import re as _re

        # Find any spirv.GlobalVariable ops
        globals_text = ""
        for m in _re.finditer(r'(spirv\.GlobalVariable\s+@\w+\s*:.+)', ir_text):
            globals_text += f"  {m.group(1)}\n"

        # Find kernel function name
        fname_match = _re.search(r'spirv\.func @(\w+)\(', ir_text)
        kernel_name = fname_match.group(1) if fname_match else "kernel"

        # Extract spirv.func + full body with brace matching
        func_start = ir_text.find("spirv.func")
        depth = 0
        func_end = func_start
        for i in range(func_start, len(ir_text)):
            if ir_text[i] == '{':
                depth += 1
            elif ir_text[i] == '}':
                depth -= 1
                if depth == 0:
                    func_end = i + 1
                    break
        func_text = ir_text[func_start:func_end]

        wrapped = (
            f'spirv.module Logical GLSL450 '
            f'requires #spirv.vce<v1.0, [Shader], '
            f'[SPV_KHR_storage_buffer_storage_class]> {{\n'
            f'{globals_text}'
            f'  {func_text}\n'
            f'  spirv.EntryPoint "GLCompute" @{kernel_name}\n'
            f'}}\n'
        )

        with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w",
                                         delete=False) as f:
            f.write(wrapped)
            ir_path = f.name

        try:
            # Serialize to SPIR-V binary
            spv_path = ir_path + ".spv"
            result = subprocess.run([
                mlir_translate, ir_path,
                "--no-implicit-module",
                "--serialize-spirv",
                "-o", spv_path,
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(
                    f"mlir-translate --serialize-spirv failed:\n{result.stderr}")

            with open(spv_path, "rb") as f:
                binary = f.read()

            return binary
        finally:
            for p in [ir_path, ir_path + ".spv"]:
                if os.path.exists(p):
                    os.unlink(p)
