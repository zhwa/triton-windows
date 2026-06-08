---
name: triton-windows-vulkan-perf
description: "Incremental Vulkan/SPIR-V performance improvements (Path C+). Covers WorkgroupId dispatch, device-local memory, shared memory tree reductions, subgroup operations, cooperative matrix, and documented traps and strategy lessons. Use for: adding Vulkan compute features, fixing SPIR-V conversion issues, understanding the buffer-forwarding architecture, or extending the pipeline."
argument-hint: "workgroup-id | device-local | shared-memory | barriers | subgroup | traps | strategy"
user-invocable: true
---

# Triton-Windows Vulkan Performance (Path C+)

Incremental enhancements to the MLIR SPIR-V pipeline. Each step adds one
Vulkan compute feature while preserving all existing tests.

**Philosophy:** _"Premature optimization is the root of all evil."_ — Knuth.
Get correctness first (base skill), then improve performance one feature
at a time. Every step is a working commit with all Vulkan tests passing
(run `python third_party/vulkan/test/test_kernels_vulkan.py` to verify).

**Prerequisite:** The base `triton-windows-vulkan` skill must be complete
(converter set, bridge passes, VulkanizePass, VulkanCompute runtime).

**Ordering is critical.** Each C+ step builds infrastructure used by later steps:
- C+1 invents the **placeholder pattern** and **builtin variable creation**
- C+3 invents **module attributes**, **shared memory promotion**, and **type rebuilding**
- C+5 combines all of the above into the **buffer-forwarding architecture**

Do NOT attempt C+5 without completing C+1–C+4 first.

## 1. C+ Roadmap

| Step | Feature | Files Changed | Key Traps | Status |
|------|---------|--------------|-----------|--------|
| C+1 | WorkgroupId for program_id | PrepareSPIRV.cpp, test | C1-1 | ✅ |
| C+2 | Device-local memory | VulkanCompute.{h,cpp} | C2-1 | ✅ |
| C+3 | Workgroup shared memory | PrepareSPIRV.cpp, TritonToLinalg*.cpp, compiler.py, triton_vulkan.cc, CMakeLists.txt | C3-* | ✅ |
| C+4 | Subgroup operations | PrepareSPIRV.cpp, VulkanCompute.{h,cpp}, triton_vulkan.cc | C4-* | ✅ |
| C+5 | Cooperative matrix | PrepareSPIRV.cpp, VulkanCompute.cpp, compiler.py, test | C5-* | ✅ |
| C+6 | Discrete GPU selection | VulkanCompute.cpp | — | ✅ |
| — | Performance baseline | test_kernels_vulkan.py | — | ✅ |

---

## 2. C+1: WorkgroupId for program_id

**Goal:** Replace push-constant `program_id` with SPIR-V `WorkgroupId` builtin
so all blocks execute in a single `vkCmdDispatch(num_blocks, 1, 1)`.

### How to Reproduce

1. In `PrepareSPIRV.cpp` VulkanizePass, split scalar args: the `program_id`
   args (last group in the program-info args) → replaced with
   `__builtin_workgroup_id` GlobalVariable
   (`BuiltIn WorkgroupId`), read via `spirv.CompositeExtract`
2. Remaining scalars (N, num_programs) → push constants with a smaller payload
3. Use `emitError() + signalPassFailure()`, NOT `assert()` (trap C1-1)
4. Add test: `vadd_multiblock` — 1024 elements, 4 workgroups

### Arg Layout After C+1

Program-info arg count is defined by `TRITON_PROGRAM_INFO_ARG_COUNT` in
`TritonToLinalgPass.cpp`.

```
func @kernel(%ptrs..., %scalar args..., %program_info_args...)
                                      ^^^^^^^^^^^^^^^^^^^^^ the last program-info group → WorkgroupId builtin
Push constants: scalar args except the program_id group (no pid)
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

On discrete GPUs, a host-visible `DEVICE_LOCAL` memory type may actually be
the PCIe BAR, not true VRAM. `findMemoryTypeFallback`
must skip types with both flags. Many Vulkan tutorials get this wrong.

---

## 4. C+3: Workgroup Shared Memory (Most Complex Step)

**Goal:** Parallel tree reductions using SPIR-V Workgroup shared memory +
`spirv.ControlBarrier` for synchronization.

### How to Reproduce

**Step 1: Add local_id args.** In `TritonToLinalgPass.cpp`, update
`TRITON_PROGRAM_INFO_ARG_COUNT` to include the `local_id` group. Fix arg
indices:
- `GetProgramIDConverter`: index from the final program-info group
- `GetNumProgramsConverter`: index from the preceding program-info group

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

### Documented Traps for C+3 (in the order you will hit them)

| # | What Goes Wrong | Why | Fix |
|---|----------------|-----|-----|
| C3-8 | MSVC: cascading `auto` failures | Missing `#include "Linalg/IR/Linalg.h"` in PrepareSPIRV.cpp | Include ALL used dialect headers first |
| C3-6 | Existing converters read wrong args | Adding the `local_id` group shifts indices | Fix `GetProgramID` and `GetNumPrograms` index math using `TRITON_PROGRAM_INFO_ARG_COUNT` |
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

**Goal:** Replace the innermost tree reduction strides below `SUBGROUP_SIZE`
(search for `SUBGROUP_SIZE` in `PrepareSPIRV.cpp`) with a single
`spirv.GroupNonUniform*` op. This reduces shared-memory barriers for
large block reductions.

### How to Reproduce

1. `VulkanCompute.{h,cpp}`: Add `subgroupSize_` field + `getSubgroupSize()`
   method. Query via `VkPhysicalDeviceSubgroupProperties` in `pickPhysicalDevice`.
   **Must use the API version that exposes `vkGetPhysicalDeviceProperties2`** — trap C4-1.
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

`vkGetPhysicalDeviceProperties2` (for subgroup size query) requires the
API version that exposes that query path. With older API settings,
`subgroupSize` silently returns 0.

### Trap C4-2: Missing Barrier After Subgroup Reduce

After the subgroup reduce, each subgroup stores its result to shared[tid].
Subgroup 0 stores the correct answer in shared[0]. But threads in later
subgroups may read shared[0] before subgroup 0's store completes.
Without a barrier, **softmax fails** (race on shared[0]).
This bug is hard to catch because reduce_sum and reduce_max pass — only
softmax (2 reductions sharing the same shared memory) exposes the race.

### Known Limitation

Subgroup size is defined by `SUBGROUP_SIZE` (search for it in
`PrepareSPIRV.cpp`). For GPUs with non-default subgroup widths, it should
be made configurable via module attribute.

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
| Test | `matmul_coop_f16`: cooperative matrix validation (run the Vulkan test suite to verify) |

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

### Documented Traps of C+5

| # | Trap | Fix |
|---|------|-----|
| C5-1 | Function-class pointers crash `vkCreateComputePipelines` | Buffer-forwarding: use StorageBuffer GlobalVariables directly |
| C5-2 | `FixAllocaStorageClass` rewrites placeholder declarations | No-arg placeholder eliminates the issue |
| C5-3 | Unconditional extension enabling crashes unsupported GPUs | Query `vkEnumerateDeviceExtensionProperties` first |
| C5-4 | Stride must match original matrix row length, not tile size | Store stride in `vulkan.coop_dims` attribute |
| C5-5 | AccessChain result storage class must match base pointer | Use `spirv::StorageClass::StorageBuffer` explicitly |

---

## 7. C+6: Discrete GPU Selection + Performance Baseline

**Goal:** Prefer discrete GPUs over integrated, and add compile/dispatch
timing to every test for regression tracking.

### How to Reproduce

1. **`pickPhysicalDevice()`** in `VulkanCompute.cpp` (search for `pickPhysicalDevice`):
   - Enumerate all physical devices
   - Score each by `VkPhysicalDeviceType`: discrete=3, integrated=2, virtual=1, other=0
   - For each device, find the first compute-capable queue family
   - Pick the highest-scoring device (use `std::max_element`)

2. **Compile timing** in `compiler.py` `comp()` function:
   - Wrap the 5-stage pipeline (`make_ttir` → `make_spv`) with `time.perf_counter()`
   - Store result as `md["compile_ms"]`

3. **Dispatch timing** in `test_kernels_vulkan.py` `run()` function:
   - 1 warmup dispatch (not timed)
   - 5 timed dispatches, average as `md["dispatch_us"]`
   - Return timing in the results tuple: `(name, error, tolerance, compile_ms, dispatch_us)`

4. **Output format**: Table with Compile (ms) and Dispatch (µs) columns,
   plus summary line with totals.

### What Changed

| Component | Change |
|-----------|--------|
| `pickPhysicalDevice()` | Scoring system instead of "first match wins" |
| `comp()` | `time.perf_counter()` around pipeline stages |
| `run()` | Warmup + 5-run average dispatch timing |
| Results tuple | 5-element: `(name, err, tol, compile_ms, dispatch_us)` |

---

## 8. Traps Reference

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
| C4-1 | Vulkan | `vkGetPhysicalDeviceProperties2` needs the API version that exposes subgroup properties |
| C4-2 | Synchronization | Missing barrier after subgroup reduce → race on shared[0] |
| C5-1 | SPIR-V Spec | Function-class pointers invalid for cooperative matrix ops |
| C5-2 | MLIR Pipeline | FixAllocaStorageClass rewrites placeholder declarations |
| C5-3 | Vulkan | Unconditional extension enabling crashes unsupported GPUs |
| C5-4 | Convention | Stride = matrix row length, not tile size |
| C5-5 | SPIR-V | AccessChain result storage class must match base pointer |
| G-1 | Correctness | Masked load/store must apply mask — never do full copy/store |
| G-2 | SPIR-V Spec | Push-constant struct needs natural alignment per member type |
| G-3 | SPIR-V Spec | Push constants are NOT interface variables in EntryPoint |
| G-4 | Portability | Subgroup size varies by vendor; do not assume the default fits every GPU |
| G-5 | Correctness | Reduce identity must match combiner — zero is wrong for min/max |
| G-6 | MLIR API | In ConversionPattern, use `adaptor.getXxx()` not `op.getXxx()` for operands |

---

## 9. Strategy Lessons

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

6. **Test after EVERY change.** Run `python third_party/vulkan/test/test_kernels_vulkan.py` after every edit. The
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

---

## 10. Adapting to Upstream Changes

This skill documents patterns and principles, not exact code locations.
When upstream Triton or MLIR changes:

1. **Search, don't assume line numbers.** Use `rg "__vulkan_|ConvertReductionToParallel|ConvertMatmulToCooperative|SUBGROUP_SIZE" third_party\vulkan\lib\Conversion` to find the current placeholder, reduction, cooperative-matrix, and subgroup code.
2. **Verify arg layout.** Check `TRITON_PROGRAM_INFO_ARG_COUNT` in `third_party\vulkan\lib\Conversion\TritonToLinalgPass.cpp`; if it changed, update every `program_id`, `num_programs`, `local_id`, and push-constant index calculation.
3. **Verify test suite.** Run `python third_party\vulkan\test\test_kernels_vulkan.py` — the test count may change, but all tests should pass.
4. **Verify IR after each pass.** Print `str_nodebug()` after bufferize, bridge passes, and VulkanizePass; look for leftover `memref.*`, `gpu.*`, or `func.call @__vulkan_*` operations.
5. **Verify capabilities and runtime probing.** Make sure the `spirv.module` VCE triple, subgroup assumptions, and cooperative-matrix extension enabling all match the ops actually emitted and the device features actually reported.
