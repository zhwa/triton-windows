"""Shared utilities for triton development tools."""
import os, re
from triton._C.libtriton import ir, passes


def load_ttir(path):
    """Load a .ttir file and return (module, context)."""
    c = ir.context()
    ir.load_dialects(c)
    try:
        from triton._C.libtriton import vulkan
        vulkan.load_dialects(c)
    except ImportError:
        pass
    m = ir.parse_mlir_module(path, c)
    m.context = c
    return m, c


def get_ir_text(mod):
    """Get IR text without debug locations (cleaner for diffing)."""
    return mod.str_nodebug()


def count_ops(ir_text):
    """Count operations by dialect prefix in IR text.

    Matches MLIR operations like 'spirv.Constant', 'arith.addi', 'memref.alloc'.
    Requires the dialect prefix to start with a letter (filters out numeric
    literals like '0.000' and version strings like 'v1.6').
    """
    ops = re.findall(r'\b([a-z][a-z_]*\.[a-zA-Z]\w*)', ir_text)
    counts = {}
    for op in ops:
        dialect = op.split('.')[0]
        counts[dialect] = counts.get(dialect, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def get_vulkan_backend():
    """Get VulkanBackend + GPUTarget for pipeline use."""
    from triton.backends.vulkan.compiler import VulkanBackend, GPUTarget
    b = VulkanBackend(GPUTarget("vulkan", 0, 32))
    return b, b.parse_options({})


def get_stage_defs():
    """Return the Vulkan pipeline stage definitions.

    Each stage is (name, [(pass_name, add_pass_fn), ...]).
    Centralized here so inspector.py and timer.py stay in sync.
    """
    from triton._C.libtriton import vulkan

    return [
        ("ttir", [
            ("inliner", lambda pm: passes.common.add_inliner(pm)),
            ("canonicalizer", lambda pm: passes.common.add_canonicalizer(pm)),
            ("combine", lambda pm: passes.ttir.add_combine(pm)),
            ("reorder_broadcast", lambda pm: passes.ttir.add_reorder_broadcast(pm)),
            ("cse", lambda pm: passes.common.add_cse(pm)),
            ("symbol_dce", lambda pm: passes.common.add_symbol_dce(pm)),
            ("loop_unroll", lambda pm: passes.ttir.add_loop_unroll(pm)),
        ]),
        ("linalg", [
            ("triton_to_linalg", lambda pm: vulkan.passes.linalg.triton_to_linalg(pm)),
        ]),
        ("memref", [
            ("one_shot_bufferize", lambda pm: vulkan.passes.memref.one_shot_bufferize(pm)),
            ("reduction_to_parallel", lambda pm: vulkan.passes.memref.convert_reduction_to_parallel(pm)),
            ("matmul_to_cooperative", lambda pm: vulkan.passes.memref.convert_matmul_to_cooperative(pm)),
            ("linalg_to_loops", lambda pm: vulkan.passes.memref.convert_linalg_to_loops(pm)),
            ("lower_affine", lambda pm: vulkan.passes.memref.lower_affine(pm)),
            ("scf_to_cf", lambda pm: vulkan.passes.memref.convert_scf_to_cf(pm)),
            ("canonicalizer", lambda pm: passes.common.add_canonicalizer(pm)),
            ("cse", lambda pm: passes.common.add_cse(pm)),
        ]),
        ("spirv_prep", [
            ("prepare_spirv", lambda pm: vulkan.passes.spirv.prepare_spirv(pm)),
            ("lower_scf_to_cf", lambda pm: vulkan.passes.spirv.lower_scf_to_cf(pm)),
            ("canonicalizer", lambda pm: passes.common.add_canonicalizer(pm)),
        ]),
        ("spirv_map", [
            ("map_storage_class", lambda pm: vulkan.passes.spirv.map_storage_class(pm)),
            ("fix_alloca_storage_class", lambda pm: vulkan.passes.spirv.fix_alloca_storage_class(pm)),
        ]),
        ("spirv_convert", [
            ("convert_memref_to_spirv", lambda pm: vulkan.passes.spirv.convert_memref_to_spirv(pm)),
            ("convert_arith_to_spirv", lambda pm: vulkan.passes.spirv.convert_arith_to_spirv(pm)),
            ("convert_math_to_spirv", lambda pm: vulkan.passes.spirv.convert_math_to_spirv(pm)),
            ("convert_cf_to_spirv", lambda pm: vulkan.passes.spirv.convert_cf_to_spirv(pm)),
            ("convert_func_to_spirv", lambda pm: vulkan.passes.spirv.convert_func_to_spirv(pm)),
            ("canonicalizer", lambda pm: passes.common.add_canonicalizer(pm)),
        ]),
        ("vulkanize", [
            ("vulkanize", lambda pm: vulkan.passes.spirv.vulkanize(pm)),
        ]),
    ]
