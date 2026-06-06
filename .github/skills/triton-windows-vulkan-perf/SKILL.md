---
name: triton-windows-vulkan-perf
description: "Incremental Vulkan/SPIR-V performance improvements (Path C+). Covers WorkgroupId dispatch, device-local memory, shared memory tree reductions, subgroup operations, cooperative matrix, and 23 documented traps. Use for: adding Vulkan compute features, fixing SPIR-V conversion issues, understanding the buffer-forwarding architecture, or extending the pipeline."
argument-hint: "workgroup-id | device-local | shared-memory | barriers | subgroup | traps | strategy"
user-invocable: true
---

# Triton-Windows Vulkan Performance (Path C+)

Incremental enhancements to the MLIR SPIR-V pipeline. Each step adds one
Vulkan compute feature while preserving all existing tests.

**Philosophy:** _"Premature optimization is the root of all evil."_ — Knuth.
Get correctness first (base skill), then improve performance one feature
at a time. Every step is a working commit with 12/12 tests passing.

**Prerequisite:** The base `triton-windows-vulkan` skill must be complete
(16 converters, 7 bridge passes, VulkanizePass, VulkanCompute runtime).

**Ordering is critical.** Each C+ step builds infrastructure used by later steps:
- C+1 invents the **placeholder pattern** and **builtin variable creation**
- C+3 invents **module attributes**, **shared memory promotion**, and **type rebuilding**
- C+5 combines all of the above into the **buffer-forwarding architecture**

Do NOT attempt C+5 without completing C+1–C+4 first.

## 1. C+ Roadmap

| Step | Feature | Files Changed | Traps Hit | Status |
|------|---------|--------------|-----------|--------|
| C+1 | WorkgroupId for program_id | PrepareSPIRV.cpp, test | 1 | ✅ |
| C+2 | Device-local memory | VulkanCompute.{h,cpp} | 1 | ✅ |
| C+3 | Workgroup shared memory | PrepareSPIRV.cpp, TritonToLinalg*.cpp, compiler.py, triton_vulkan.cc, CMakeLists.txt | 8 | ✅ |
| C+4 | Subgroup operations | PrepareSPIRV.cpp, VulkanCompute.{h,cpp}, triton_vulkan.cc | 2 | ✅ |
| C+5 | Cooperative matrix | PrepareSPIRV.cpp, VulkanCompute.cpp, compiler.py, test | 5 | ✅ |

---

## 2. C+1: WorkgroupId for program_id

**Goal:** Replace push-constant `program_id` with SPIR-V `WorkgroupId` builtin
so all blocks execute in a single `vkCmdDispatch(num_blocks, 1, 1)`.

### How to Reproduce

1. In `PrepareSPIRV.cpp` VulkanizePass, split scalar args: last 3 are
   program_id → replaced with `__builtin_workgroup_id` GlobalVariable
   (`BuiltIn WorkgroupId`), read via `spirv.CompositeExtract`
2. Remaining scalars (N, num_programs) → push constants (12 bytes less)
3. Use `emitError() + signalPassFailure()`, NOT `assert()` (trap C1-1)
4. Add test: `vadd_multiblock` — 1024 elements, 4 workgroups

### Arg Layout After C+1

```
func @kernel(%ptrs..., %N, %num_progs(3), %pid(3))
                                         ^^^^^^^^ → WorkgroupId builtin
Push constants: [N, num_progs_x, num_progs_y, num_progs_z]  (no pid)
```

### Trap C1-1: assert → emitError

`assert` in an MLIR pass aborts in debug builds, no-op in release.
Always use `funcOp.emitError(...); return signalPassFailure();`.

---

## 3. C+2: Device-Local Memory

**Goal:** Use device-local VRAM instead of PCIe BAR memory for storage buffers.

### How to Reproduce

1. `VulkanCompute.h`: Add `staging`, `stagingMemory`, `deviceLocal` fields to
   `BufferInfo`; add `findMemoryTypeFallback()` and `copyBuffer()` methods
2. `createBuffer()`: Try `DEVICE_LOCAL` (non-host-visible) memory first;
   create host-visible staging buffer alongside; fall back on integrated GPUs
3. `writeBuffer()`: Host → staging (memcpy) → device (`vkCmdCopyBuffer`)
4. `readBuffer()`: Device → staging (`vkCmdCopyBuffer`) → host (memcpy)
5. `destroyShaderState()`: Clean up staging buffers
6. Add test: `vadd_65k` — 65536 elements, 256 workgroups

### Trap C2-1: BAR Memory Masquerades as VRAM

On discrete GPUs, a ~256MB memory type is BOTH `DEVICE_LOCAL` AND
`HOST_VISIBLE` — the PCIe BAR, not true VRAM. `findMemoryTypeFallback`
must skip types with both flags. Many Vulkan tutorials get this wrong.

---

## 4. C+3: Workgroup Shared Memory (Most Complex Step)

**Goal:** Parallel tree reductions using SPIR-V Workgroup shared memory +
`spirv.ControlBarrier` for synchronization.

### How to Reproduce

**Step 1: Add local_id args.** In `TritonToLinalgPass.cpp`, change
`PROGRAM_INFO_ARG_COUNT` from 6 to 9 (add 3 local_id args). Fix arg indices:
- `GetProgramIDConverter`: `numArgs - 2*3 + axis` (was `numArgs - 3 + axis`)
- `GetNumProgramsConverter`: `numArgs - 3*3 + axis` (was `numArgs - 2*3 + axis`)

**Step 2: Create `ConvertReductionToParallel` pass** (PrepareSPIRV.cpp).
Runs AFTER bufferize, BEFORE linalg-to-loops. For each `linalg.reduce`:
1. Set `vulkan.local_size = [blockSize, 1, 1]` on module (not function!)
2. Create `memref.alloca` with address space 3 (→ Workgroup after mapping)
3. Each thread: load input[tid] → shared[tid]
4. Emit barrier: `func.call @__vulkan_barrier()` (placeholder)
5. Tree reduction loop (stride halving): load/combine/store + barrier per stride
6. All threads read shared[0] → write to output
7. Erase the `linalg.reduce`

**Step 3: Register the pass** in `triton_vulkan.cc` and call in
`compiler.py` (`convert_reduction_to_parallel` after `one_shot_bufferize`).

**Step 4: Update VulkanizePass** to handle new features:
- Detect Workgroup-bound Variables by matching `spirv.array<N>` against
  `vulkan.local_size`. First match = load buffer (skip), rest = shared
- Promote each to module-scope `spirv.GlobalVariable Workgroup`
- Rebuild ALL downstream AccessChain/Load/Store with Workgroup ptr types
- Replace `spirv.FunctionCall @__vulkan_barrier` → `spirv.ControlBarrier`
- Add `__builtin_local_invocation_id` (BuiltIn LocalInvocationId)
- Read `vulkan.local_size` for `ExecutionMode LocalSize`

### The IR Flow (What Happens to Shared Memory)

```
linalg.reduce                        → consumed by ConvertReductionToParallel
    ↓
memref.alloca(AS 3) + scf.if + func.call @__vulkan_barrier
    ↓ linalg-to-loops + scf-to-cf
cf.* + memref.load/store + func.call @__vulkan_barrier
    ↓ map_storage_class (AS 3 → Workgroup)
memref<256xf32, #spirv.storage_class<Workgroup>>
    ↓ fix_alloca_storage_class (ALL allocas → Function!)
memref<256xf32, #spirv.storage_class<Function>>
    ↓ convert-memref-to-spirv
spirv.Variable : !spirv.ptr<struct<array<256 x f32>>, Function>
    ↓ convert-func-to-spirv
spirv.FunctionCall @__vulkan_barrier()
    ↓ VulkanizePass
      1. Promote Variable → Workgroup GlobalVariable (rebuild types)
      2. Replace @__vulkan_barrier → spirv.ControlBarrier
      3. Add builtins (WorkgroupId, LocalInvocationId)
```

### The 8 Traps (in the order you will hit them)

| # | What Goes Wrong | Why | Fix |
|---|----------------|-----|-----|
| C3-8 | MSVC: cascading `auto` failures | Missing `#include "Linalg/IR/Linalg.h"` in PrepareSPIRV.cpp | Include ALL used dialect headers first |
| C3-6 | Existing converters read wrong args | Adding 3 args shifts indices | Fix GetProgramID (numArgs-6) and GetNumPrograms (numArgs-9) |
| C3-1 | `gpu.barrier` silently ignored | `convert-gpu-to-spirv` doesn't work on `func.func` | Use `func.call @__vulkan_barrier` placeholder |
| C3-2 | `spirv.ControlBarrier` blocks func conversion | `convert-func-to-spirv` rejects non-func-dialect ops | Defer barrier creation to VulkanizePass |
| C3-3 | Shared alloca becomes Function class | `convert-memref-to-spirv` forces ALL Variables to Function | Detect by array size in VulkanizePass |
| C3-4 | Verifier rejects function-scope Workgroup Variable | SPIR-V spec: Workgroup must be module-scope | Promote to GlobalVariable in VulkanizePass |
| C3-5 | Type mismatch cascade after promotion | Replacing Function ptr → Workgroup ptr breaks AccessChain types | Rebuild ALL downstream AccessChain/Load/Store ops |
| C3-7 | Softmax fails (only promote last variable) | Softmax has 2 reductions = 2 shared allocas | Skip first match (load buffer), promote ALL rest |

### Design Patterns Worth Remembering

1. **Placeholder pattern:** `func.call @placeholder()` survives through ALL
   MLIR conversion passes (func→spirv converts to `spirv.FunctionCall`).
   Replace in VulkanizePass. Erase declaration. Works for barriers AND
   subgroup ops.

2. **Module attributes beat function attributes:** `vulkan.local_size` on
   the module, not the func. Function attributes don't survive `func-to-spirv`.

3. **Detect by structure, not by attribute:** Custom attributes don't survive
   `convert-*-to-spirv`. Match array dimensions against known values instead.

### Pipeline Order After C+3

```python
# make_memref:
one_shot_bufferize
→ convert_reduction_to_parallel   # linalg.reduce → tree reduction
→ convert_matmul_to_cooperative   # linalg.matmul 16x16 f16 → coop placeholder
→ convert_linalg_to_loops
→ lower_affine → convert_scf_to_cf → canonicalize + cse

# make_spirv:
prepare_spirv → lower_scf_to_cf → canonicalize
→ map_storage_class → fix_alloca_storage_class
→ convert_{memref,arith,math,cf,func}_to_spirv → canonicalize
→ vulkanize   # shared mem promotion + barrier replacement + builtins
```

---

## 5. C+4: Subgroup Operations

**Goal:** Replace the innermost tree reduction strides (< subgroupSize=32)
with a single `spirv.GroupNonUniform*` op. Eliminates 5 barriers for
256-element reductions (8 → 3 barriers).

### How to Reproduce

1. `VulkanCompute.{h,cpp}`: Add `subgroupSize_` field + `getSubgroupSize()`
   method. Query via `VkPhysicalDeviceSubgroupProperties` in `pickPhysicalDevice`.
   **Must use `VK_API_VERSION_1_1`** — trap C4-1.
2. `triton_vulkan.cc`: Expose `subgroup_size()` to Python via pybind11
3. In `ConvertReductionToParallel`:
   a. Classify combiner op (arith.addf → fadd, arith.maximumf → fmax, etc.)
   b. Set `stopStride = SUBGROUP_SIZE` if combiner maps to a subgroup op
   c. Tree reduction loop runs only for strides ≥ stopStride
   d. After loop: emit `func.call @__vulkan_subgroup_reduce_*(%shared[tid])`
   e. Store subgroup result to shared[tid]
   f. **Add barrier AFTER store** (trap C4-2!)
   g. Read shared[0] as final result
4. VulkanizePass: replace placeholder calls with `spirv.GroupNonUniform*Op`
   (scope=Reduce, execution=Subgroup). Conditionally set VCE triple to
   V_1_3 + `GroupNonUniform` + `GroupNonUniformArithmetic`.
5. Clean up: erase all 6 `__vulkan_subgroup_reduce_*` function declarations

### Combiner → Subgroup Op Mapping

| Combiner | Placeholder | SPIR-V Op |
|----------|------------|----------|
| `arith.addf` | `__vulkan_subgroup_reduce_fadd` | `GroupNonUniformFAdd` |
| `arith.addi` | `__vulkan_subgroup_reduce_iadd` | `GroupNonUniformIAdd` |
| `arith.maximumf` | `__vulkan_subgroup_reduce_fmax` | `GroupNonUniformFMax` |
| `arith.maxsi` | `__vulkan_subgroup_reduce_smax` | `GroupNonUniformSMax` |
| `arith.minimumf` | `__vulkan_subgroup_reduce_fmin` | `GroupNonUniformFMin` |
| `arith.minsi` | `__vulkan_subgroup_reduce_smin` | `GroupNonUniformSMin` |

### Reduction Architecture (blockSize=256)

```
stride 128 → shared mem tree + barrier   ┐
stride  64 → shared mem tree + barrier   ├ 3 strides, 3 barriers
stride  32 → shared mem tree + barrier   ┘
────── subgroup boundary (32 threads) ──────
stride 16→1 → GroupNonUniformFAdd Reduce  ← 1 op replaces 5 strides
→ barrier → read shared[0]
Total: 4 barriers (was 9 without subgroup ops)
```

### Trap C4-1: Vulkan API Version

`vkGetPhysicalDeviceProperties2` (for subgroup size query) requires
`VK_API_VERSION_1_1`. With 1.0, `subgroupSize` silently returns 0.

### Trap C4-2: Missing Barrier After Subgroup Reduce

After the subgroup reduce, each subgroup stores its result to shared[tid].
Subgroup 0 stores the correct answer in shared[0]. But threads in other
subgroups (32+) may read shared[0] before subgroup 0's store completes.
Without a barrier, **softmax fails with ~0.23 error** (race on shared[0]).
This bug is hard to catch because reduce_sum and reduce_max pass — only
softmax (2 reductions sharing the same shared memory) exposes the race.

### Known Limitation

Subgroup size is hardcoded to 32 (`constexpr SUBGROUP_SIZE = 32`). Correct
for all NVIDIA GPUs. For AMD (64) or Intel (8/16/32), should be made
configurable via module attribute.

---

## 6. C+5: Cooperative Matrix (Buffer-Forwarding Architecture)

**Goal:** Use `VK_KHR_cooperative_matrix` for 16×16 f16 matmul, loading tiles
directly from StorageBuffer GlobalVariables (bypassing Function-class allocas).

### Architecture

The key innovation is **buffer-forwarding**: the cooperative matrix ops
skip the alloca-copy pattern entirely and reference the original
StorageBuffer GlobalVariables via module attributes.

```
ConvertMatmulToCooperative (runs after bufferize):
  traces matmul operands backward: alloc → memref.copy → reinterpret_cast → func arg
  stores vulkan.coop_buffer_args = [argA, argB, argC] on module
  emits: call @__vulkan_coop_matmul()  (no operands!)

VulkanizePass:
  reads vulkan.coop_buffer_args → finds GlobalVariables @binding0/1/2
  AccessChain @binding0[0][0] → spirv.ptr<f16, StorageBuffer>
  KHRCooperativeMatrixLoad(ptr, RowMajor, stride)
  KHRCooperativeMatrixMulAdd(A, B, zero_acc) → f32 result
  FConvert f32→f16
  KHRCooperativeMatrixStore(ptr, result, RowMajor, stride)
```

### What Changed

| Component | Change |
|-----------|--------|
| `ConvertMatmulToCooperative` | `traceToFuncArg()` walks alloc→copy→reinterpret_cast→BlockArgument. No-arg placeholder. |
| VulkanizePass | Maps `vulkan.coop_buffer_args` → GlobalVariables. AccessChain with `StorageBuffer` pointers. |
| `createLogicalDevice` | Queries `vkEnumerateDeviceExtensionProperties`. Enables coop matrix only if supported. |
| VCE triple | V_1_6 + CooperativeMatrixKHR + Float16 + StorageBuffer16BitAccess (conditional) |
| Test | `matmul_coop_f16`: 16×16 f16 cooperative matmul, error ~0.015 |

### How to Reproduce

**CRITICAL: Cooperative matrix ops require StorageBuffer pointers.** The
`tt.load` → `memref.alloc` → `memref.copy` pattern produces Function-class
pointers in SPIR-V, which are INVALID for cooperative matrix Load/Store
(driver crashes at `vkCreateComputePipelines`). The solution is
**buffer-forwarding**: trace operands backward to find the original buffer
function args, pass their indices via module attributes, and have
VulkanizePass AccessChain into StorageBuffer GlobalVariables directly.

1. **`ConvertMatmulToCooperative` pass** (PrepareSPIRV.cpp):
   - Only match static 16×16 f16 `linalg.matmul` (after bufferize)
   - `traceToFuncArg()`: walk `AllocOp` → find `CopyOp(source, alloc)` →
     trace source through `ReinterpretCastOp`/`CastOp` → `BlockArgument`
   - `traceOutputToFuncArg()`: walk alloc users forward →
     find `CopyOp(alloc, dest)` → trace dest to `BlockArgument`
   - Store `vulkan.coop_buffer_args = [argA, argB, argC]` on module
   - Store `vulkan.coop_dims = [M, N, K]` on module
   - Emit **no-arg** placeholder: `call @__vulkan_coop_matmul()`
   - **No-arg is essential** — avoids FixAllocaStorageClass type mangling

2. **VulkanizePass** coop replacement:
   - Read `vulkan.coop_buffer_args` → `findGlobalVar(argIdx)` maps to `@bindingN`
   - `AddressOf @bindingN` → `AccessChain [0][0]` → `ptr<f16, StorageBuffer>`
   - `KHRCooperativeMatrixLoad(ptr, RowMajor, stride)` for A and B
   - `CompositeConstruct(0.0f)` → zero f32 accumulator
   - `KHRCooperativeMatrixMulAdd(A, B, zero)` → f32 result
   - `FConvert` f32→f16 → `KHRCooperativeMatrixStore`

3. **VulkanCompute.cpp**: Query `vkEnumerateDeviceExtensionProperties`
   before enabling `VK_KHR_cooperative_matrix`. **Never enable
   unconditionally** — crashes GPUs without support (trap C5-3).

4. **VCE triple**: Conditionally add `V_1_6 + CooperativeMatrixKHR +
   Float16 + StorageBuffer16BitAccess + SPV_KHR_cooperative_matrix +
   SPV_KHR_16bit_storage` only when coop ops are emitted.

5. **Test**: `matmul_coop_f16` — 16×16 f16 matmul, tolerance 5e-2.

### The 5 Traps of C+5

| # | Trap | Fix |
|---|------|-----|
| C5-1 | Function-class pointers crash `vkCreateComputePipelines` | Buffer-forwarding: use StorageBuffer GlobalVariables directly |
| C5-2 | `FixAllocaStorageClass` rewrites placeholder declarations | No-arg placeholder eliminates the issue |
| C5-3 | Unconditional extension enabling crashes unsupported GPUs | Query `vkEnumerateDeviceExtensionProperties` first |
| C5-4 | Stride must match original matrix row length, not tile size | Store stride in `vulkan.coop_dims` attribute |
| C5-5 | AccessChain result storage class must match base pointer | Use `spirv::StorageClass::StorageBuffer` explicitly |

---

## 7. Traps Reference (All 23)

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
| C4-1 | Vulkan | `vkGetPhysicalDeviceProperties2` needs Vulkan 1.1 API version |
| C4-2 | Synchronization | Missing barrier after subgroup reduce → race on shared[0] |
| C5-1 | SPIR-V Spec | Function-class pointers invalid for cooperative matrix ops |
| C5-2 | MLIR Pipeline | FixAllocaStorageClass rewrites placeholder declarations |
| C5-3 | Vulkan | Unconditional extension enabling crashes unsupported GPUs |
| C5-4 | Convention | Stride = matrix row length, not tile size |
| C5-5 | SPIR-V | AccessChain result storage class must match base pointer |
| G-1 | Correctness | Masked load/store must apply mask — never do full copy/store |
| G-2 | SPIR-V Spec | Push-constant struct needs natural alignment per member type |
| G-3 | SPIR-V Spec | Push constants are NOT interface variables in EntryPoint |
| G-4 | Portability | Subgroup size varies by vendor (NVIDIA=32, AMD=64, Intel=8–32) |
| G-5 | Correctness | Reduce identity must match combiner — zero is wrong for min/max |
| G-6 | MLIR API | In ConversionPattern, use `adaptor.getXxx()` not `op.getXxx()` for operands |

---

## 8. Strategy Lessons

1. **Get it working first, then make it fast.** The base skill achieves
   correctness. This skill achieves performance. Never mix the two.

2. **MLIR SPIR-V conversion passes are black boxes.** They silently drop
   operations, force storage classes, and ignore attributes. Always verify
   IR at every pipeline step with `m.str_nodebug()`.

3. **The placeholder pattern is the key innovation.** `func.call @name()`
   survives through ALL MLIR conversion passes unchanged. VulkanizePass
   replaces them with actual SPIR-V ops. Used for barriers, subgroup ops,
   and cooperative matrix. This solves a fundamental MLIR infrastructure gap.

4. **Module attributes > function attributes.** Function attributes don't
   survive dialect conversion. Store metadata on the module.

5. **Detection by structure, not by attribute.** Custom attributes don't
   survive through `convert-*-to-spirv`. Detect shared memory by matching
   array dimensions against known block size.

6. **Test after EVERY change.** Run all 12 tests after every edit. The
   pipeline is fragile — a change in arg layout breaks everything downstream.

7. **Debug with `str_nodebug()` + regex.** Print IR at each pipeline step.
   Search for non-spirv ops: `re.findall(r'(memref\.\w+|gpu\.\w+)', text)`.
   The verifier error messages reference source locations, not actual ops.

8. **VCE triple must match actual ops.** `spirv.module` VCE triple must
   include capabilities for ALL ops used. Non-reduction shaders stay at
   V_1_0. Subgroup ops bump to V_1_3. Coop matrix bumps to V_1_6.

9. **Buffer-forwarding bypasses pipeline limitations.** When MLIR passes
   force wrong storage classes, trace backward through the IR to find the
   original buffer args and reference their GlobalVariables directly.
