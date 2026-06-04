---
name: triton-windows-spirv-setup
description: "SPIR-V backend setup for triton-windows. Covers all traps, workarounds, and pass ordering for the TTIR→Linalg→MemRef→SPIR-V pipeline. Use for: adding SPIR-V output, fixing SPIR-V conversion failures, debugging memref-to-spirv issues, or understanding the alloca/storage-class problem."
argument-hint: "diagnose | pipeline-order | alloca-fix | reinterpret-cast | storage-class | serialize | full-setup"
user-invocable: true
---

# Triton-Windows SPIR-V Setup Skill

You are an expert at the SPIR-V backend for triton-windows. This skill documents
every trap, workaround, and design decision discovered during implementation.

## Architecture Overview

```
TTIR → make_ttir → make_linalg → make_memref → make_spirv → make_spv
         │             │              │              │           │
    shared TTIR    C++ pass      MLIR passes    MLIR passes   mlir-translate
    passes         TritonToLinalg  bufferize     prepare_spirv  --serialize-spirv
                                  linalg→loops  map_storage
                                  lower_affine  fix_alloca
                                  scf→cf        memref→spirv
                                                arith→spirv
                                                cf→spirv
                                                func→spirv
```

## Key Files

| File | Purpose |
|------|---------|
| `third_party/vulkan/lib/Conversion/PrepareSPIRV.cpp` | Eliminates SPIR-V-incompatible memref ops |
| `third_party/vulkan/lib/Conversion/TritonToLinalg.cpp` | Triton IR → Linalg/MemRef conversion |
| `third_party/vulkan/triton_vulkan.cc` | pybind11 pass registration |
| `third_party/vulkan/backend/compiler.py` | Python pipeline orchestration |
| `third_party/vulkan/CMakeLists.txt` | Build rules + SPIR-V link libraries |

---

## TRAP #1: `memref.reinterpret_cast` Has No SPIR-V Lowering

**Symptom:** `convert-memref-to-spirv` silently skips `memref.reinterpret_cast`.
The op stays unconverted, blocking `convert-func-to-spirv` from creating
`spirv.module`.

**Root Cause:** MLIR's `MemRefToSPIRV` conversion doesn't have a pattern for
`memref::ReinterpretCastOp`. It only handles `load`, `store`, `alloc`, `alloca`,
`subview`, `cast`, etc.

**Why It Exists:** Our `AddPtrConverter` in `TritonToLinalg.cpp` converts
`tt.addptr(splat(base_ptr), offsets)` into `memref.reinterpret_cast` to create
a view into the base pointer with computed offset/stride. This is the natural
MLIR lowering for Triton's pointer arithmetic.

**Fix:** The `ExpandReinterpretCast` pattern in `PrepareSPIRV.cpp` replaces
`reinterpret_cast` by redirecting all load/store users to the source memref
with adjusted indices:
```
BEFORE: %view = reinterpret_cast %base offset:[off] sizes:[N] strides:[1]
        %v = memref.load %view[%i]
AFTER:  %idx = arith.addi %i, off
        %v = memref.load %base[%idx]
```

**Diagnostic Script:**
```python
# Check if any reinterpret_cast ops remain after prepare_spirv
import re
ir_text = mod.str_nodebug()
casts = re.findall(r'memref\.reinterpret_cast', ir_text)
if casts:
    print(f"TRAP #1: {len(casts)} memref.reinterpret_cast ops remain!")
    print("Run prepare_spirv pass BEFORE convert-memref-to-spirv")
```

---

## TRAP #2: `memref.copy` Has No SPIR-V Lowering

**Symptom:** Same as Trap #1 — `memref.copy` stays unconverted.

**Root Cause:** MLIR's `MemRefToSPIRV` has no pattern for `memref::CopyOp`.

**Why It Exists:** Our `LoadConverter` creates local buffers:
`alloc → fill(zero) → memref.copy(input, alloc) → loads from alloc`.

**Fix:** `ExpandMemRefCopy` in `PrepareSPIRV.cpp` expands `memref.copy` to an
explicit `scf.for` loop with `memref.load`/`memref.store`. The scf.for must
then be lowered with `convert-scf-to-cf` before SPIR-V conversion.

**Important:** After expanding copies, a SECOND `convert-scf-to-cf` pass is
needed because `make_memref` already lowered the original SCF ops, but the
copy expansion creates NEW `scf.for` ops.

---

## TRAP #3: `map-memref-spirv-storage-class` Maps Allocas to StorageBuffer

**THIS IS THE MOST CRITICAL TRAP.**

**Symptom:** `convert-memref-to-spirv` silently skips `memref.alloca` ops.
The alloca stays unconverted, blocking `convert-func-to-spirv`.

**Root Cause:** `map-memref-spirv-storage-class` maps ALL address-space-0
memrefs to `#spirv.storage_class<StorageBuffer>`. This includes local
`memref.alloca` scratch buffers. But `convert-memref-to-spirv`'s
`AllocaOpPattern` has this check:
```cpp
static bool isAllocationSupported(Operation *allocOp, MemRefType type) {
    auto storageClass = type.getMemorySpace();
    if (storageClass) {
        auto spirvStorageClass = dyn_cast<spirv::StorageClassAttr>(storageClass);
        if (!spirvStorageClass ||
            spirvStorageClass.getValue() != spirv::StorageClass::Function)
            return false;  // ← REJECTS StorageBuffer allocas!
    }
}
```

**Fix:** `FixAllocaStorageClassPass` runs AFTER `map-memref-spirv-storage-class`
and BEFORE `convert-memref-to-spirv`. It changes alloca storage class from
`StorageBuffer` to `Function`:
```cpp
auto funcAttr = spirv::StorageClassAttr::get(ctx, spirv::StorageClass::Function);
auto newType = MemRefType::get(oldType.getShape(), oldType.getElementType(),
                               oldType.getLayout(), funcAttr);
```

**Pipeline Order (CRITICAL — do NOT reorder):**
```
1. prepare_spirv          (alloc→alloca, expand reinterpret_cast/copy)
2. convert_scf_to_cf      (lower new scf.for from copy expansion)
3. canonicalize
4. map_storage_class       (maps addr_space 0 → StorageBuffer)
5. fix_alloca_storage_class (StorageBuffer → Function on allocas ONLY)
6. convert_memref_to_spirv (alloca→spirv.Variable, load/store→AccessChain)
7. convert_arith_to_spirv
8. convert_cf_to_spirv
9. convert_func_to_spirv   (func.func → spirv.func)
10. canonicalize
```

**Diagnostic Script:**
```python
# After map_storage_class, check if allocas have wrong storage class
import re
ir_text = mod.str_nodebug()
bad_allocas = re.findall(r'memref\.alloca.*StorageBuffer', ir_text)
if bad_allocas:
    print(f"TRAP #3: {len(bad_allocas)} allocas have StorageBuffer class!")
    print("Run fix_alloca_storage_class BEFORE convert-memref-to-spirv")
```

---

## TRAP #4: `convert-func-to-spirv` Doesn't Create `spirv.module`

**Symptom:** After all conversion passes, the IR has `spirv.func` but NO
`spirv.module`. Serialization fails with "expected a 'spirv.module' op".

**Root Cause:** MLIR's `convert-func-to-spirv` creates `spirv.func` from
`func.func`, but only creates the `spirv.module` wrapper if the function body
is FULLY converted (no unconverted ops). If ANY `memref.*` or
`builtin.unrealized_conversion_cast` ops remain, it skips the wrapper.

**Fix:** In `make_spv()`, we wrap the `spirv.func` in `spirv.module` via text
manipulation:
```python
wrapped = (
    f'spirv.module Logical GLSL450 '
    f'requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {{\n'
    f'  {func_text}\n'
    f'  spirv.EntryPoint "GLCompute" @{kernel_name}\n'
    f'}}\n'
)
```

**Diagnostic:** Check if `spirv.module` exists in the IR:
```python
if 'spirv.module' not in mod.str_nodebug():
    remaining = set(re.findall(r'(memref\.\w+|func\.\w+|builtin\.unrealized)', ir))
    print(f"TRAP #4: spirv.module missing. Remaining ops: {remaining}")
```

---

## TRAP #5: `mlir-translate --serialize-spirv` Needs `--no-implicit-module`

**Symptom:** `mlir-translate --serialize-spirv` fails with
"expected a 'spirv.module' op, got 'builtin.module'"

**Root Cause:** MLIR's parser wraps everything in an implicit `builtin.module`.
The SPIR-V serializer expects `spirv.module` as the top-level op.

**Fix:** Use `--no-implicit-module` flag:
```
mlir-translate --no-implicit-module --serialize-spirv input.mlir -o output.spv
```

---

## TRAP #6: `spirv.target_env` Must Be Attached Before Any Conversion

**Symptom:** SPIR-V conversion passes produce empty or wrong output.

**Root Cause:** All `convert-*-to-spirv` passes check the `spirv.target_env`
attribute on the module to determine what capabilities are available. Without it,
they fall back to minimal capabilities and may skip many conversions.

**Fix:** `PrepareSPIRV` attaches it as the first step:
```cpp
auto triple = spirv::VerCapExtAttr::get(
    spirv::Version::V_1_0, {spirv::Capability::Shader},
    {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, ctx);
auto limits = spirv::ResourceLimitsAttr::get(
    ctx, 16384, 128,
    Builder(ctx).getI32ArrayAttr({128, 128, 64}),
    32, std::nullopt, std::nullopt, ArrayAttr(), ArrayAttr());
moduleOp->setAttr(spirv::getTargetEnvAttrName(),
                  spirv::TargetEnvAttr::get(triple, limits));
```

**Note:** `ResourceLimitsAttr::get` signature varies by LLVM version. The
4th argument is `ArrayAttr` (not `DenseI32ArrayAttr`). Use
`Builder(ctx).getI32ArrayAttr({128, 128, 64})` for workgroup size.

---

## TRAP #7: `SPIRVUpdateVCEPass` Requires `spirv.module` Context

**Symptom:** `spirv-update-vce` fails with "trying to schedule pass on an
unsupported operation" when run on `builtin.module`.

**Root Cause:** `SPIRVUpdateVCEPass` is designed to run on `spirv.module`, not
`builtin.module`. It must run AFTER `convert-func-to-spirv` creates the module.

**Fix:** Don't include `spirv-update-vce` in the main pipeline. Either run it
separately after wrapping, or skip it entirely (the serializer handles VCE).

---

## TRAP #8: `UnrankedMemRefType` Not Supported by SPIR-V Conversion

**Symptom:** Function args like `memref<*xf32>` are not converted by
`convert-memref-to-spirv`.

**Root Cause:** Triton's type converter maps `!tt.ptr<T>` to `UnrankedMemRefType`.
SPIR-V doesn't have unranked memrefs — all buffers must be typed.

**Fix:** `PrepareSPIRV` converts function signatures from
`memref<*xf32>` → `memref<?xf32>` before the SPIR-V passes:
```cpp
auto ranked = MemRefType::get({ShapedType::kDynamic},
                              unranked.getElementType(),
                              nullptr, unranked.getMemorySpace());
```

---

## TRAP #9: Pointer Splat Conversion Must Not Create `linalg.fill`

**Symptom:** `tt.splat %ptr` on pointer types causes crashes or wrong IR.

**Root Cause:** `SplatConverter` in `TritonToLinalg.cpp` normally creates
`tensor.empty + linalg.fill`. But for pointer types, the "value" is a memref,
not a scalar — `linalg.fill` can't broadcast a memref.

**Fix:** `SplatConverter` checks `isa<triton::PointerType>(op.getSrc().getType())`
and replaces the op with `adaptor.getSrc()` directly (the type-converted memref
from the function argument). Non-pointer splats use the normal fill path.

---

## TRAP #10: MLIR API Varies by LLVM Commit

These are specific to the LLVM commit used by Triton 3.7.0:

| API | Correct Usage |
|-----|---------------|
| `linalg::ReduceOp` | Use `ReduceOp::create(builder, loc, inputs, inits, dims, bodyBuilder)` static method |
| `arith::ConstantOp` | Requires `TypedAttr`: `cast<TypedAttr>(builder.getZeroAttr(elemType))` |
| `FunctionInterfaces.h` | Path is `mlir/Interfaces/FunctionInterfaces.h` (NOT `mlir/IR/`) |
| `populateElementwiseToLinalgConversionPatterns` | In `mlir/Dialect/Linalg/Transforms/Transforms.h` |
| `ResourceLimitsAttr::get` | 4th arg is `ArrayAttr` (use `Builder.getI32ArrayAttr`) |
| `createSCFToControlFlowPass` | NOT `createConvertSCFToCFPass` |
| `spirv::ModuleOp::getBody()` | Returns `Block*` (NOT `Region&`) |

---

## Diagnostic Script

Save as `third_party/vulkan/tools/diagnose-spirv.py`:

```python
#!/usr/bin/env python3
"""Diagnose SPIR-V conversion issues in the Vulkan backend."""

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
            f"TRAP #3: {n} allocas have StorageBuffer class. "
            f"Run fix_alloca_storage_class AFTER map_storage_class.")

    # TRAP #3b: alloca with no storage class after mapping
    if 'spirv.storage_class' in ir_text:
        n = len(re.findall(r'memref\.alloca\(\)\s*:\s*memref<\d+x\w+>', ir_text))
        if n > 0:
            issues.append(
                f"TRAP #3b: {n} allocas have no storage class after mapping. "
                f"They may not convert to spirv.Variable.")

    # TRAP #4: spirv.func without spirv.module
    if 'spirv.func' in ir_text and 'spirv.module' not in ir_text:
        remaining = set(re.findall(r'(memref\.\w+|func\.\w+)', ir_text))
        issues.append(
            f"TRAP #4: spirv.func exists but no spirv.module. "
            f"Remaining non-SPIR-V ops: {remaining if remaining else 'none (wrap manually)'}.")

    # TRAP #6: missing target_env
    if 'spirv.target_env' not in ir_text and stage in ('spirv', 'convert'):
        issues.append(
            f"TRAP #6: No spirv.target_env attribute found. "
            f"Run prepare_spirv first to attach it.")

    # TRAP #8: unranked memrefs
    n = len(re.findall(r'memref<\*x\w+>', ir_text))
    if n > 0:
        issues.append(
            f"TRAP #8: {n} unranked memref types remain. "
            f"Run prepare_spirv to convert to ranked memref<?x...>.")

    # General: unrealized conversion casts
    n = len(re.findall(r'builtin\.unrealized_conversion_cast', ir_text))
    if n > 0:
        issues.append(
            f"INFO: {n} unrealized_conversion_cast ops remain. "
            f"These are type bridges — should be eliminated by reconcile-unrealized-casts.")

    # General: remaining memref ops
    remaining_memref = set(re.findall(r'memref\.(\w+)', ir_text))
    if remaining_memref:
        issues.append(
            f"INFO: Remaining memref ops: {remaining_memref}")

    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: diagnose-spirv.py <mlir-file> [stage]")
        print("Stages: prep, map, convert, final")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        ir_text = f.read()

    stage = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    issues = diagnose(ir_text, stage)

    if not issues:
        print("✅ No known SPIR-V conversion issues found.")
    else:
        print(f"⚠️  Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")


if __name__ == "__main__":
    main()
```

---

## Quick Reference: Pass Pipeline in Python

```python
from triton._C.libtriton import ir, passes, vulkan

# Step 1: Prepare (expand reinterpret_cast/copy, alloc→alloca, target_env)
pm = ir.pass_manager(mod.context)
vulkan.passes.spirv.prepare_spirv(pm)
vulkan.passes.spirv.lower_scf_to_cf(pm)  # for new scf.for from copy expansion
passes.common.add_canonicalizer(pm)
pm.run(mod, 'prep')

# Step 2: Map storage classes, then fix allocas
pm = ir.pass_manager(mod.context)
vulkan.passes.spirv.map_storage_class(pm)
vulkan.passes.spirv.fix_alloca_storage_class(pm)  # CRITICAL — Trap #3
pm.run(mod, 'map')

# Step 3: Convert to SPIR-V
pm = ir.pass_manager(mod.context)
vulkan.passes.spirv.convert_memref_to_spirv(pm)
vulkan.passes.spirv.convert_arith_to_spirv(pm)
vulkan.passes.spirv.convert_cf_to_spirv(pm)
vulkan.passes.spirv.convert_func_to_spirv(pm)
passes.common.add_canonicalizer(pm)
pm.run(mod, 'convert')

# Step 4: Serialize
# spirv.func exists but spirv.module doesn't → wrap via text + mlir-translate
binary = backend.make_spv(mod, metadata, opt)
```

---

## CMake Libraries Required

```cmake
target_link_libraries(VulkanTritonToLinalg PUBLIC
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSPIRVTransforms
  MLIRArithToSPIRV
  MLIRControlFlowToSPIRV
  MLIRFuncToSPIRV
  MLIRMemRefToSPIRV
  # ... plus standard MLIR libs
)

target_link_libraries(TritonVulkan PRIVATE
  MLIRSPIRVSerialization  # for vulkan.serialize_spirv()
  MLIRSPIRVDialect
  MLIRSPIRVTransforms
)
```

---

## Common Error Messages and What They Mean

| Error | Trap | Fix |
|-------|------|-----|
| "expected a 'spirv.module' op, got 'builtin.module'" | #5 | Add `--no-implicit-module` to mlir-translate |
| "trying to schedule pass 'SPIRVUpdateVCEPass' on an unsupported operation" | #7 | Don't run update_vce before spirv.module exists |
| "operation destroyed but still has uses" | FinalizeSPIRV | Don't erase func.func while block args have users |
| "'memref.reinterpret_cast' op failed to legalize" | #1 | Run prepare_spirv first |
| "cannot open file 'MLIRMapMemRefStorageClass.lib'" | CMake | Use `MLIRMemRefToSPIRV` instead (contains the pass) |
| `ResourceLimitsAttr::get` wrong arg count | #10 | Check LLVM version; 4th arg is ArrayAttr not DenseI32ArrayAttr |
| alloca silently not converted | #3 | Check storage class: must be Function, not StorageBuffer |
