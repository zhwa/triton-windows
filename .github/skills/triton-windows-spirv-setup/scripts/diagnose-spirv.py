#!/usr/bin/env python3
"""Diagnose SPIR-V conversion issues in the Vulkan backend.

Usage:
    python diagnose-spirv.py <mlir-file> [stage]
    python diagnose-spirv.py --pipeline <ttir-file>   # Run full pipeline and diagnose each stage

Stages: prep, map, convert, final
"""

import os
import re
import sys


def diagnose(ir_text: str, stage: str = "unknown") -> list[str]:
    """Check for known SPIR-V conversion traps in the IR."""
    issues = []

    # TRAP #1: reinterpret_cast
    n = len(re.findall(r'memref\.reinterpret_cast', ir_text))
    if n > 0:
        issues.append(
            f"TRAP #1: {n} memref.reinterpret_cast ops remain. "
            f"Run prepare_spirv to expand them.")

    # TRAP #2: memref.copy
    n = len(re.findall(r'memref\.copy\b', ir_text))
    if n > 0:
        issues.append(
            f"TRAP #2: {n} memref.copy ops remain. "
            f"Run prepare_spirv to expand them to loops.")

    # TRAP #3: alloca with StorageBuffer
    n = len(re.findall(r'memref\.alloca.*StorageBuffer', ir_text))
    if n > 0:
        issues.append(
            f"TRAP #3: {n} allocas have StorageBuffer class! "
            f"Run fix_alloca_storage_class AFTER map_storage_class.")

    # TRAP #3 variant: alloca with Function class (good)
    n_func = len(re.findall(r'memref\.alloca.*Function', ir_text))
    if n_func > 0 and stage in ('map', 'convert'):
        issues.append(
            f"OK: {n_func} allocas have Function class (correct for SPIR-V).")

    # TRAP #4: spirv.func without spirv.module
    if 'spirv.func' in ir_text and 'spirv.module' not in ir_text:
        remaining = set(re.findall(r'(memref\.\w+|func\.\w+)', ir_text))
        if remaining:
            issues.append(
                f"TRAP #4: spirv.func exists but no spirv.module. "
                f"Remaining non-SPIR-V ops: {remaining}")
        else:
            issues.append(
                f"TRAP #4: spirv.func exists but no spirv.module. "
                f"All ops are SPIR-V — wrap with spirv.module text in make_spv().")

    # TRAP #6: missing target_env
    if 'spirv.target_env' not in ir_text and stage in ('spirv', 'convert', 'map'):
        issues.append(
            f"TRAP #6: No spirv.target_env attribute found. "
            f"Run prepare_spirv first to attach it.")

    # TRAP #8: unranked memrefs
    n = len(re.findall(r'memref<\*x\w+>', ir_text))
    if n > 0:
        issues.append(
            f"TRAP #8: {n} unranked memref types remain. "
            f"Run prepare_spirv to convert to ranked memref<?x...>.")

    # Unrealized conversion casts
    n = len(re.findall(r'builtin\.unrealized_conversion_cast', ir_text))
    if n > 0:
        issues.append(
            f"INFO: {n} unrealized_conversion_cast ops (type bridges).")

    # Remaining memref ops
    remaining_memref = set(re.findall(r'memref\.(\w+)', ir_text))
    if remaining_memref:
        issues.append(
            f"INFO: Remaining memref ops: {remaining_memref}")

    # SPIR-V ops count
    spirv_ops = len(re.findall(r'spirv\.\w+', ir_text))
    if spirv_ops > 0:
        issues.append(f"INFO: {spirv_ops} SPIR-V ops in IR.")

    # spirv.Variable check
    if 'spirv.Variable' in ir_text:
        issues.append("OK: spirv.Variable found (alloca converted).")

    return issues


def run_pipeline_diagnosis(ttir_path: str):
    """Run the full pipeline and diagnose each stage."""
    try:
        from triton._C.libtriton import ir, passes, vulkan
        from triton.backends.vulkan.compiler import VulkanBackend
        from triton.backends.compiler import GPUTarget
    except ImportError:
        print("ERROR: Cannot import triton. Set TRITON_BACKENDS_IN_TREE=1")
        sys.exit(1)

    ctx = ir.context()
    ir.load_dialects(ctx)
    vulkan.load_dialects(ctx)

    mod = ir.parse_mlir_module(ttir_path, ctx)
    mod.context = ctx
    backend = VulkanBackend(GPUTarget('vulkan', 0, 32))
    opt = backend.parse_options({})
    metadata = {}

    stages = [
        ("make_ttir", lambda: backend.make_ttir(mod, metadata, opt)),
        ("make_linalg", lambda: backend.make_linalg(mod, metadata, opt)),
        ("make_memref", lambda: backend.make_memref(mod, metadata, opt)),
    ]

    for name, fn in stages:
        try:
            result = fn()
            if result is not None:
                pass  # mod is modified in-place for most stages
            print(f"\n{'='*60}")
            print(f"After {name}:")
            print(f"{'='*60}")
            issues = diagnose(mod.str_nodebug(), name)
            for issue in issues:
                print(f"  {issue}")
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            return

    # SPIR-V stages — run individually with diagnosis
    spv_stages = [
        ("prepare_spirv", lambda pm: (
            vulkan.passes.spirv.prepare_spirv(pm),
            vulkan.passes.spirv.lower_scf_to_cf(pm),
            passes.common.add_canonicalizer(pm),
        )),
        ("map_storage + fix_alloca", lambda pm: (
            vulkan.passes.spirv.map_storage_class(pm),
            vulkan.passes.spirv.fix_alloca_storage_class(pm),
        )),
        ("convert_to_spirv", lambda pm: (
            vulkan.passes.spirv.convert_memref_to_spirv(pm),
            vulkan.passes.spirv.convert_arith_to_spirv(pm),
            vulkan.passes.spirv.convert_cf_to_spirv(pm),
            vulkan.passes.spirv.convert_func_to_spirv(pm),
            passes.common.add_canonicalizer(pm),
        )),
    ]

    for name, setup_fn in spv_stages:
        try:
            pm = ir.pass_manager(mod.context)
            setup_fn(pm)
            pm.run(mod, name)
            print(f"\n{'='*60}")
            print(f"After {name}:")
            print(f"{'='*60}")
            issues = diagnose(mod.str_nodebug(), 'convert')
            for issue in issues:
                print(f"  {issue}")
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            # Still try to diagnose
            issues = diagnose(mod.str_nodebug(), 'convert')
            for issue in issues:
                print(f"  {issue}")
            return

    # Try serialization
    print(f"\n{'='*60}")
    print("Serialization:")
    print(f"{'='*60}")
    try:
        binary = backend.make_spv(mod, metadata, opt)
        import struct
        magic = struct.unpack('<I', binary[:4])[0]
        print(f"  [OK] SPIR-V binary: {len(binary)} bytes")
        print(f"  Magic: 0x{magic:08X} (valid: {magic == 0x07230203})")
    except Exception as e:
        print(f"  [FAIL] Serialization failed: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--pipeline":
        if len(sys.argv) < 3:
            print("Usage: diagnose-spirv.py --pipeline <ttir-file>")
            sys.exit(1)
        os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
        run_pipeline_diagnosis(sys.argv[2])
    else:
        with open(sys.argv[1]) as f:
            ir_text = f.read()
        stage = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        issues = diagnose(ir_text, stage)
        if not issues:
            print("[OK] No known SPIR-V conversion issues found.")
        else:
            print(f"[WARN] Found {len(issues)} issue(s):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")


if __name__ == "__main__":
    main()
