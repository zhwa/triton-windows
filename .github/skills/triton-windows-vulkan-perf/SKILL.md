---
name: triton-windows-vulkan-perf
description: "Incremental Vulkan/SPIR-V performance improvements (Path C+). Covers WorkgroupId dispatch, device-local memory, shared memory tree reductions, and 10 documented traps. Use for: adding Vulkan compute features, fixing SPIR-V conversion issues, understanding the shared memory promotion pipeline, or planning C+4/C+5 work."
argument-hint: "workgroup-id | device-local | shared-memory | barriers | traps | strategy"
user-invocable: true
---

# Triton-Windows Vulkan Performance (Path C+)

Incremental enhancements to the MLIR SPIR-V pipeline. Each step adds one
Vulkan compute feature while preserving all existing tests.

**Philosophy:** _"Premature optimization is the root of all evil."_ — Knuth.
Get correctness first (base skill), then improve performance one feature
at a time. Every step is a working commit with 11/11 tests passing.

**Prerequisite:** The base `triton-windows-vulkan` skill must be complete
(16 converters, 7 bridge passes, VulkanizePass, VulkanCompute runtime).

## 1. C+ Roadmap

| Step | Feature | Files Changed | Traps Hit | Status |
|------|---------|--------------|-----------|--------|
| C+1 | WorkgroupId for program_id | PrepareSPIRV.cpp, test | 1 | ✅ |
| C+2 | Device-local memory | VulkanCompute.{h,cpp} | 1 | ✅ |
| C+3 | Workgroup shared memory | PrepareSPIRV.cpp, TritonToLinalg*.cpp, compiler.py, triton_vulkan.cc | 8 | ✅ |
| C+4 | Subgroup operations | (future) | — | 🔲 |
| C+5 | Cooperative matrix | (future) | — | 🔲 |

---

## 2. C+1: WorkgroupId for program_id

**Goal:** Replace push-constant `program_id` with SPIR-V `WorkgroupId` builtin
so all blocks execute in a single `vkCmdDispatch(num_blocks, 1, 1)`.

### What Changed

| Component | Change |
|-----------|--------|
| VulkanizePass | Last 3 scalar args → `__builtin_workgroup_id` GlobalVariable with `BuiltIn WorkgroupId`, read via `spirv.CompositeExtract` |
| Push constants | Reduced by 12 bytes (3×i32 pid removed) |
| Test | `vadd_multiblock`: 1024 elements across 4 parallel workgroups |

### Arg Layout (after C+1)

```
func.func @kernel(%ptr: memref<*xf32>, ..., %N: i32,
                  %num_progs_x: i32, %num_progs_y: i32, %num_progs_z: i32,
                  %pid_x: i32, %pid_y: i32, %pid_z: i32)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  replaced by WorkgroupId builtin
```

Push constants: `[N, num_programs(3)]` — no pid.

### Trap C1-1: assert → emitError

Using `assert` in an MLIR pass aborts in debug builds, is no-op in release.
Always use `funcOp.emitError(...); return signalPassFailure();`.

---

## 3. C+2: Device-Local Memory

**Goal:** Use device-local VRAM instead of PCIe BAR memory for storage buffers.

### What Changed

| Component | Change |
|-----------|--------|
| `BufferInfo` struct | New fields: `staging`, `stagingMemory`, `deviceLocal` flag |
| `createBuffer` | Tries DEVICE_LOCAL VRAM; creates host-visible staging buffer alongside; falls back on integrated GPUs |
| `writeBuffer` | Host → staging (memcpy) → device (vkCmdCopyBuffer) |
| `readBuffer` | Device → staging (vkCmdCopyBuffer) → host (memcpy) |
| `copyBuffer` (new) | One-shot command buffer for vkCmdCopyBuffer |
| `findMemoryTypeFallback` (new) | Returns -1 instead of throwing; skips BAR memory |
| `destroyShaderState` | Cleans up staging buffers too |
| Test | `vadd_65k`: 65536 elements across 256 workgroups |

### Trap C2-1: BAR Memory Masquerades as VRAM

On discrete GPUs, there's a ~256MB memory type that is BOTH `DEVICE_LOCAL`
and `HOST_VISIBLE` — the PCIe BAR, not true VRAM. `findMemoryTypeFallback`
explicitly skips types with both flags to find the large non-host-visible heap.

Many Vulkan tutorials get this wrong.

---

## 4. C+3: Workgroup Shared Memory (Most Complex)

**Goal:** Parallel tree reductions using SPIR-V Workgroup shared memory +
`spirv.ControlBarrier` for synchronization.

### Architecture

```
linalg.reduce                        (consumed by ConvertReductionToParallel)
    ↓
memref.alloca(AS 3) + scf.if + func.call @__vulkan_barrier
    ↓ bufferize already done; linalg-to-loops + scf-to-cf + canonicalize
cf.* + memref.load/store + func.call @__vulkan_barrier
    ↓ map_storage_class: AS 3 → Workgroup
memref<256xf32, #spirv.storage_class<Workgroup>>
    ↓ fix_alloca_storage_class: ALL allocas → Function (including shared!)
memref<256xf32, #spirv.storage_class<Function>>
    ↓ convert-memref-to-spirv: alloca → spirv.Variable Function
spirv.Variable : !spirv.ptr<struct<array<256 x f32>>, Function>
    ↓ convert-func-to-spirv: func.func → spirv.func
spirv.FunctionCall @__vulkan_barrier()
    ↓ VulkanizePass:
      1. Promote shared Variable → Workgroup GlobalVariable
      2. Rebuild all AccessChain/Load/Store with Workgroup ptr types
      3. Replace @__vulkan_barrier calls → spirv.ControlBarrier
```

### What Changed

| Component | Change |
|-----------|--------|
| `TritonToLinalgPass.cpp` | 9 program info args: num_programs(3) + pid(3) + local_id(3) |
| `TritonToLinalg.cpp` | `GetProgramIDConverter`: numArgs-6. `GetNumProgramsConverter`: numArgs-9 |
| `ConvertReductionToParallel` (new pass) | linalg.reduce → parallel tree reduction with shared alloca(AS 3) + barrier placeholders |
| VulkanizePass | LocalInvocationId builtin, configurable LocalSize, shared Variable→Workgroup GlobalVariable, barrier replacement |
| `compiler.py` | `convert_reduction_to_parallel` after bufferize, before linalg-to-loops |
| `triton_vulkan.cc` | Register new pass |

### The 8 Traps of C+3

| # | Trap | What Happened | Fix |
|---|------|--------------|-----|
| C3-1 | `gpu.barrier` not converted | `convert-gpu-to-spirv` silently ignores `gpu.barrier` inside `func.func` — the pass only works on `gpu.func` or requires specific context | Use `func.call @__vulkan_barrier` placeholder; replace in VulkanizePass with `spirv.ControlBarrier` |
| C3-2 | `spirv.ControlBarrier` at MemRef level | Inserting `spirv.*` ops at the MemRef level blocks `convert-func-to-spirv` — the function contains non-func ops and the converter refuses to process it | Same fix as C3-1: defer barrier creation to VulkanizePass |
| C3-3 | `convert-memref-to-spirv` forces Function class | ALL `memref.alloca` → `spirv.Variable Function`, even those with Workgroup address space. The conversion ignores the memory space for the Variable's storage_class attribute | Detect by array size + block size from `vulkan.local_size`; promote in VulkanizePass |
| C3-4 | Workgroup Variable must be module-scope | SPIR-V spec requires Workgroup storage class variables at module scope, not function scope. The verifier rejects function-scope Workgroup variables | VulkanizePass creates module-scope GlobalVariable and replaces the function-scope Variable |
| C3-5 | Type cascade on Variable promotion | Replacing a Function ptr Variable with a Workgroup ptr AddressOf causes type mismatches in all downstream AccessChain/Load/Store ops | Rebuild ALL AccessChain + Load + Store ops with corrected Workgroup pointer types |
| C3-6 | Arg layout shifted by 3 | Adding 3 local_id args shifts program_id from `numArgs-3` to `numArgs-6` and num_programs from `numArgs-6` to `numArgs-9` | Update `GetProgramIDConverter` and `GetNumProgramsConverter` |
| C3-7 | Softmax has 2 reductions → 2 shared vars | Initially only promoted the LAST matching Variable. Softmax produces 2 shared allocas (max + sum) | Skip first matching Variable (load buffer), promote ALL subsequent |
| C3-8 | MSVC `auto` cascading failures | One failed type deduction (`MemRefType::get` missing `#include`) causes ALL subsequent `auto` to fail with "cannot be used before initialized" | Always `#include` all used dialect headers; use explicit types for complex expressions |

### Key Design Decisions

1. **Barrier placeholder pattern:** `func.call @__vulkan_barrier()` survives
   through all MLIR conversion passes unchanged (func→spirv converts it to
   `spirv.FunctionCall`). VulkanizePass replaces and erases the declaration.

2. **Shared memory detection by array size:** After SPIR-V conversion, the
   original Workgroup address space is lost. We match `spirv.array<N>` where
   N = block size from `vulkan.local_size`. First match = load buffer (from
   `tt.load`), subsequent matches = shared memory.

3. **Module-level attributes:** `vulkan.local_size` is set on the module (not
   the function) because function attributes don't survive `func-to-spirv`.

### Pipeline Order (C+3)

```python
# make_memref:
bufferize → convert_reduction_to_parallel → linalg_to_loops → lower_affine → scf_to_cf

# make_spirv:
prepare_spirv → scf_to_cf → canonicalize
  → map_storage_class → fix_alloca_storage_class
  → convert_{memref,arith,math,cf,func}_to_spirv → canonicalize
  → vulkanize  # shared var promotion + barrier replacement + builtins
```

---

## 5. Traps Summary (All C+ Steps)

Quick reference for the 10 documented traps across C+1 through C+3:

| ID | Category | One-liner |
|----|----------|-----------|
| C1-1 | MLIR API | `assert` in pass → use `emitError` + `signalPassFailure` |
| C2-1 | Vulkan | BAR memory (HOST_VISIBLE+DEVICE_LOCAL) is not true VRAM |
| C3-1 | MLIR Pipeline | `gpu.barrier` silently ignored by `convert-gpu-to-spirv` |
| C3-2 | MLIR Pipeline | `spirv.*` ops at MemRef level block `convert-func-to-spirv` |
| C3-3 | MLIR SPIR-V | `convert-memref-to-spirv` forces ALL Variables to Function class |
| C3-4 | SPIR-V Spec | Workgroup Variables must be at module scope |
| C3-5 | MLIR IR | Storage class change cascades to ALL AccessChain/Load/Store users |
| C3-6 | Convention | Adding args shifts existing arg indices in converters |
| C3-7 | Semantics | Softmax = 2 reductions = 2 shared variables to promote |
| C3-8 | MSVC | Missing `#include` causes cascading `auto` deduction failures |

---

## 6. C+4 / C+5 Planning (Future)

### C+4: Subgroup Operations

Replace the innermost reduction strides (< subgroupSize) with
`OpGroupNonUniform{IAdd,FAdd,SMax,...}`. Requires `VK_1_1` or
`VK_KHR_shader_subgroup_extended_types`.

**Expected approach:**
- Add `subgroup_size` to VulkanCompute device query
- In `ConvertReductionToParallel`, stop tree reduction at `stride = subgroupSize`
- Emit `spirv.GroupNonUniformFAdd` etc. for the final strides
- No shared memory needed for the subgroup portion

### C+5: Cooperative Matrix

Use `VK_KHR_cooperative_matrix` for matmul. Requires RTX 2080 Ti support
check (Turing supports 16x16 fp16 cooperative matrix).

**Expected approach:**
- Query `VkPhysicalDeviceCooperativeMatrixPropertiesKHR`
- Replace `linalg.matmul` → cooperative matrix load/multiply/store
- Requires FP16 input support

---

## 7. Strategy Lessons

1. **Get it working first, then make it fast.** The base skill achieves
   correctness. This skill achieves performance. Never mix the two.

2. **MLIR SPIR-V conversion passes are black boxes.** They silently drop
   operations, force storage classes, and ignore attributes. Always verify
   IR at every pipeline step.

3. **The barrier problem is fundamental.** There is NO clean way to express
   workgroup barriers at the MemRef level that survives through MLIR's
   SPIR-V conversion. The `func.call` placeholder pattern is the pragmatic
   solution.

4. **Module attributes > function attributes.** Function attributes don't
   survive dialect conversion. Store metadata on the module.

5. **Detection by structure, not by attribute.** Custom attributes don't
   survive through `convert-*-to-spirv`. Detect shared memory by matching
   array dimensions against known block size.

6. **Test after EVERY change.** Run all 11 tests after every edit. The
   pipeline is fragile — a change in arg layout breaks everything downstream.

7. **Debug with `str_nodebug()`.** Print IR at each pipeline step. Search
   for non-spirv ops with regex. The verifier error messages reference
   source locations, not the actual offending op.
