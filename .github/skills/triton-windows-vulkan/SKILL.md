---
name: triton-windows-vulkan
description: "Foundation Vulkan/SPIR-V backend for triton-windows. Covers the full TTIR→SPIR-V→Vulkan dispatch pipeline: TritonToLinalg conversion, bridge passes, VulkanizePass, VulkanCompute runtime, and documented traps and strategy lessons. For performance improvements (C+ steps: WorkgroupId, device-local memory, shared memory, subgroups, cooperative matrix), see `triton-windows-vulkan-perf`."
argument-hint: "pipeline | converters | bridge-passes | vulkanize | push-constants | runtime | traps | diagnostics"
user-invocable: true
---

# Triton-Windows Vulkan/SPIR-V Backend

Complete guide to the Vulkan backend for triton-windows. Covers the full
pipeline from Triton IR to GPU execution via native Vulkan compute dispatch.

**Vulkan kernels verified on the target NVIDIA GPU via Vulkan SPIR-V dispatch (run `python third_party/vulkan/test/test_kernels_vulkan.py` to verify).**

## 1. Architecture

```
TTIR → make_ttir → make_linalg → make_memref → make_spirv → make_spv → VulkanCompute → GPU
         shared      C++ pass     bufferize     bridge+convert  serialize   Vulkan dispatch
         passes      converter set  +loops+cf    +vulkanize      C++ API     vkCmdDispatch
```

### File Layout

| File | Purpose |
|------|---------|
| `lib/Conversion/TritonToLinalg.cpp` | The full converter set: Triton→Linalg/Tensor/MemRef (search for `OpConversionPattern` or `patterns.add<`) |
| `lib/Conversion/TritonToLinalgPass.cpp` | Pass wrapper, TritonTypeConverter, illegal ops |
| `lib/Conversion/PrepareSPIRV.cpp` | The bridge passes (search for pass structs/pattern classes) + VulkanizePass |
| `lib/Runtime/VulkanCompute.{h,cpp}` | Vulkan compute dispatch engine |
| `triton_vulkan.cc` | pybind11 module exposing passes + runtime |
| `backend/compiler.py` | Python pipeline orchestration |
| `test/test_kernels_vulkan.py` | Vulkan GPU test suite (run `python third_party/vulkan/test/test_kernels_vulkan.py` to verify) |

### Build Commands

```powershell
# Activate MSVC + build
cmd /c '"...vcvars64.bat" -vcvars_ver=14.44 >nul 2>&1 && set' | ...
cd build/cmake.win-amd64-cpython-3.14
cmake --build . --target triton  # rebuilds C++ + links Vulkan

# Sync Python files
Copy-Item third_party/vulkan/backend/*.py python/triton/backends/vulkan/ -Force

# Run Vulkan tests
$env:TRITON_BACKENDS_IN_TREE="1"
python third_party/vulkan/test/test_kernels_vulkan.py
```

---

## 2. TritonToLinalg Converters

### Type System (TritonTypeConverter)

| Triton Type | Converted Type |
|-------------|---------------|
| `!tt.ptr<T>` | `memref<*xT>` (UnrankedMemRefType) |
| `tensor<N x !tt.ptr<T>>` | `memref<NxT>` (RankedMemRefType) |

### Converter Catalog

| Converter | Triton Op | Lowering |
|-----------|-----------|----------|
| SplatConverter | `tt.splat` | Pointer: pass through. Scalar: `tensor.empty` + `linalg.fill` |
| MakeRangeConverter | `tt.make_range` | `linalg.generic` + `linalg.index` |
| BroadcastConverter | `tt.broadcast` | `linalg.generic` with broadcast affine map |
| ExpandDimsConverter | `tt.expand_dims` | `tensor.expand_shape` |
| TransposeConverter | `tt.trans` | `linalg.transpose` |
| ReshapeConverter | `tt.reshape` | `tensor.expand/collapse_shape` |
| GetProgramIDConverter | `tt.get_program_id` | Function arg (the `program_id` group in the program-info args; count defined by `TRITON_PROGRAM_INFO_ARG_COUNT` in `TritonToLinalgPass.cpp`) |
| GetNumProgramsConverter | `tt.get_num_programs` | Function arg (program-info args; count defined by `TRITON_PROGRAM_INFO_ARG_COUNT` in `TritonToLinalgPass.cpp`) |
| MatmulConverter | `tt.dot` | `linalg.matmul` + zero-init |
| ReduceConverter | `tt.reduce` | `linalg.reduce` with cloned combiner |
| BitcastConverter | `tt.bitcast` | `arith.bitcast` |
| DenseConstantConverter | dense splat | `tensor.empty` + `linalg.fill` |
| AddPtrConverter | `tt.addptr` | `memref.reinterpret_cast` via PtrState |
| LoadConverter | `tt.load` | `alloc` + `fill(zero)` + `memref.copy` + `to_tensor` |
| StoreConverter | `tt.store` | `bufferization.materialize_in_destination` |
| AtomicRMWConverter | `tt.atomic_rmw` | Sequential `scf.for` with memref load-modify-store |

### Key Patterns

**PtrState/AddPtr chain:** `tt.splat(ptr) → tt.make_range → arith.addi → tt.addptr`
chains are walked by `visitOperand()` to extract offset/size/stride → `memref.reinterpret_cast`.
Dynamic offsets (e.g., `pid * BLOCK_SIZE`) propagate via SplatOp integer scalar tracking.

**Reduction identity values:** `addf→0`, `mulf→1`, `maximumf→-inf`, `minimumf→+inf`, `andi→all-ones`

**Registration:** Use `patterns.add<Foo>(ctx)` (NO typeConverter) for all converters
except AtomicRMWConverter which needs `patterns.add<Foo>(typeConverter, ctx)`.

---

## 3. Bridge Passes (PrepareSPIRV.cpp)

The upstream MLIR `convert-*-to-spirv` passes assume structured IR.
Triton's pointer-heavy IR needs bridge passes first. This is standard
practice (IREE, Intel XPU do the same).

| Pass | Problem | Fix |
|------|---------|-----|
| ExpandReinterpretCast | No SPIR-V lowering for `reinterpret_cast` | Redirect load/store to source + offset |
| ExpandMemRefCopy | No SPIR-V lowering for `memref.copy` | Expand to `scf.for` loop (needs 2nd scf→cf) |
| ExpandExpandShape | No SPIR-V lowering for `expand_shape` | Linearize 2D indices: `i*dim1+j` |
| Flatten2DAllocs | 2D allocs not supported | Walk: `memref<MxN>` → `memref<M*N>`, linearize all users |
| alloc→alloca | `memref.alloc` not supported | Replace with `memref.alloca` |
| Unranked→Ranked | `memref<*xT>` not supported | Function sigs: `memref<*xT>` → `memref<?xT>` |
| target_env | Conversion passes need it | Attach `spirv.target_env` attribute to module |

**Pass ordering in compiler.py:**
```
Step 1: prepare_spirv + scf→cf + canonicalize
Step 2: map_storage_class + fix_alloca_storage_class
Step 3: convert_{memref,arith,math,cf,func}_to_spirv + canonicalize
Step 4: vulkanize
```

---

## 4. VulkanizePass

Converts `spirv.func @kernel(args...)` → Vulkan-compatible `spirv.module`.
This is the most complex pass. Core responsibilities (C+ additions in perf skill):

**a) Buffer args → GlobalVariables:**
StorageBuffer pointer args → `spirv.GlobalVariable` with `bind(0, N)`

**b) WorkgroupId builtin (C+1):**
The `program_id` args (last group in the program-info args; see
`TRITON_PROGRAM_INFO_ARG_COUNT` in `TritonToLinalgPass.cpp`) →
`spirv.GlobalVariable @__builtin_workgroup_id`
with `BuiltIn WorkgroupId` + `CompositeExtract` per axis

**c) LocalInvocationId builtin (C+3):**
The `local_id` args (the group immediately before `program_id` in the
program-info args) → `spirv.GlobalVariable @__builtin_local_invocation_id`
with `BuiltIn LocalInvocationId` + `CompositeExtract`

**d) Push constants:**
Remaining scalar args from the program-info layout (`TRITON_PROGRAM_INFO_ARG_COUNT`
in `TritonToLinalgPass.cpp`) → `PushConstant` struct

**e) Shared memory promotion (C+3):**
Detects `spirv.Variable` with array size matching `vulkan.local_size` attribute.
Promotes Function-scope → module-scope `spirv.GlobalVariable Workgroup`.
Rebuilds all downstream `AccessChain`/`Load`/`Store` with Workgroup pointer types.

**f) Barrier replacement (C+3):**
`spirv.FunctionCall @__vulkan_barrier` → `spirv.ControlBarrier Workgroup`

**g) Module wrapping:**
Creates `spirv.module Logical GLSL450` with VCE triple, `EntryPoint GLCompute`,
`ExecutionMode LocalSize` from `vulkan.local_size` (default 1,1,1).
Function ends with ZERO parameters (all access via globals/builtins).

**Dispatch model:** One Triton program = one Vulkan workgroup. Host calls
`vkCmdDispatch(num_blocks, 1, 1)` and each workgroup reads its `WorkgroupId`
as `program_id`. All blocks execute in parallel in a single dispatch.

---

## 5. VulkanCompute Runtime

C++ Vulkan engine exposed via pybind11:

```python
vc = vulkan.runtime.VulkanCompute()          # VkInstance → VkDevice
vc.load_shader(spv_binary, "kernel_name")    # VkShaderModule
vc.set_workgroups(num_blocks)                 # parallel dispatch
buf = vc.create_buffer(0, N * 4)             # binding=0, VkBuffer
vc.write_buffer(buf, numpy_array)
vc.set_push_constants(np.array([N, 1,1,1], dtype=np.int32))  # no pid!
vc.dispatch()                                 # vkCmdDispatch(num_blocks,1,1)
vc.read_buffer(buf, output_array)
```

**Push constants** carry non-pid scalar args (N, num_programs). `program_id`
comes from the SPIR-V `WorkgroupId` builtin — no push constant needed.
Struct offsets computed from actual type sizes. Host calls `vkCmdPushConstants`.

**Buffer memory:** Storage buffers use device-local VRAM on discrete GPUs,
with host-visible staging buffers for transfers (`vkCmdCopyBuffer`).
Falls back to host-visible-only on integrated GPUs. BAR memory
(host-visible + device-local) is explicitly skipped to find true VRAM.

---

## 6. Traps Reference

### SPIR-V Conversion Traps

| # | Trap | Fix |
|---|------|-----|
| 1 | `memref.reinterpret_cast` no SPIR-V lowering | ExpandReinterpretCast bridge pass |
| 2 | `memref.copy` no SPIR-V lowering | ExpandMemRefCopy bridge pass |
| 3 | `map-memref-spirv-storage-class` maps allocas to StorageBuffer | FixAllocaStorageClassPass after map |
| 4 | `convert-func-to-spirv` doesn't create `spirv.module` | VulkanizePass creates it |
| 5 | `mlir-translate` needs `--no-implicit-module` | Use C++ `serialize()` instead |
| 6 | `spirv.target_env` must be attached first | PrepareSPIRV attaches it |
| 7 | `expand_shape`/`collapse_shape` no lowering | ExpandExpandShape + 2D flatten |
| 8 | `UnrankedMemRefType` not supported | Unranked→ranked in PrepareSPIRV |
| 9 | Pointer splat must not `linalg.fill` | SplatConverter checks pointer type |
| 10 | `math.*` ops not handled by arith-to-spirv | Separate `convert-math-to-spirv` pass |

### Vulkan Dispatch Traps

| # | Trap | Fix |
|---|------|-----|
| V1 | `spirv.func` args ≠ Vulkan interface vars | VulkanizePass: args → GlobalVariables |
| V2 | `spirv.module` needs `vce_triple` | `setVceTripleAttr()` after creation |
| V3 | PushConstant AccessChain: 1 index, not 2 | `AccessChain %pc[%memberIdx]` only |
| V4 | Second scf→cf needed after copy expansion | Extra `lower_scf_to_cf` in Step 1 |
| V5 | 2D alloc flatten: walk, not pattern | Pattern causes domination errors |
| V6 | `convert-math-to-spirv` required | Link `MLIRMathToSPIRV`, add to pipeline |

### MLIR API Traps (for the MLIR API version pinned by the repo; see `cmake/llvm-hash.txt`)

| API | Correct Usage |
|-----|---------------|
| `linalg::ReduceOp` | `ReduceOp::create()` static method with body builder |
| `arith::ConstantOp` | Requires `TypedAttr`: `cast<TypedAttr>(...)` |
| `FunctionInterfaces.h` | Path: `mlir/Interfaces/FunctionInterfaces.h` |
| `bufferization::ToTensorOp` | Explicit result type arg: `create(loc, tensorType, buf, ...)` |
| `bufferization::ToBufferOp` | NOT `ToMemrefOp` in this LLVM version |
| `ResourceLimitsAttr::get` | 4th arg: `Builder(ctx).getI32ArrayAttr({128,128,64})` |
| `OpConversionPattern` operands | **Always use `adaptor.getXxx()`, never `op.getXxx()`** for operands. The TypeConverter may have changed operand types (e.g., `tensor<Nxi1>` → `memref<Nxi1>`). Using `op.getXxx()` references the original stale value. (Trap G-6) |

---

## 7. Debugging

```python
# Dump IR at any stage
print(m.str_nodebug())

# Check remaining non-SPIR-V ops
import re
remaining = set(re.findall(r'(memref\.\w+|linalg\.\w+|math\.\w+)', m.str_nodebug()))
print("Non-SPIR-V:", sorted(remaining))

# Validate SPIR-V binary
import struct
magic = struct.unpack('<I', spv[:4])[0]
assert magic == 0x07230203

# Use triton-opt for isolated converter testing
& "python/triton/_C/triton-opt.exe" "--triton-to-linalg" test.ttir
# --debug flag shows pattern matching attempts
```

---

## 8. Known Limitations

These are current limitations in the codebase, not bugs:

| Limitation | Location | Impact |
|-----------|----------|--------|
| `pickPhysicalDevice` scores by type but no multi-GPU testing | `VulkanCompute.cpp` | Only tested with single-GPU systems |
| `LowerUnrankedCast` dead code | `PrepareSPIRV.cpp` | Defined but always returns `failure()`; never contributes |
| `vkQueueWaitIdle` per transfer | `VulkanCompute.cpp` copyBuffer | Serializes transfers; optimize with batched command buffers later |
| `ConvertReductionToParallel` 1D-only | `PrepareSPIRV.cpp` | Only handles 1D static power-of-2 reductions |
| `memref.copy` expansion 1D-only | `PrepareSPIRV.cpp` | Multi-dim or dynamic copies remain unlowered |
| 2D flattening rank-2 static only | `PrepareSPIRV.cpp` | Other ranks/dynamic shapes fall through |
| Subgroup size default comes from `SUBGROUP_SIZE` (search for it in `PrepareSPIRV.cpp`) | `PrepareSPIRV.cpp` | Set `vulkan.subgroup_size` module attribute for GPUs whose subgroup width differs from the default |
| `driver.py` is a stub | `backend/driver.py` | `is_active()` always returns False; manual test flow only |

### Correctness Invariants (must not violate)

| Rule | Why | Trap ID |
|------|-----|---------|
| Masked load must conditionally copy per element | Full `memref.copy` ignores mask → wrong for non-aligned sizes | G-1 |
| Masked store must conditionally write per element | Full `MaterializeInDestination` ignores mask | G-1 |
| Push-constant struct members must be naturally aligned | Packed offsets violate SPIR-V alignment rules for i64/f64 | G-2 |
| Push constants are NOT interface variables | Adding to `spirv.EntryPoint` interface list is invalid SPIR-V | G-3 |
| Reduce identity must match combiner op | Zero identity is wrong for min (should be +inf) and max (-inf) | G-5 |
| In `OpConversionPattern`, always use `adaptor.getXxx()` | `op.getXxx()` returns stale pre-conversion values (e.g., tensor instead of memref) | G-6 |

For performance-related items (shared memory, subgroups, cooperative matrix), see
the `triton-windows-vulkan-perf` skill.

---

## 9. Reproducing from Scratch (Optimal Journey)

If starting from a fresh clone, follow this order:

### Phase 1: Foundation (base skill — this document)

1. **Build triton-windows** — use `triton-windows-build` skill
2. **Implement the full TritonToLinalg converter set** (`TritonToLinalg.cpp`; search for `OpConversionPattern` or `patterns.add<` to see the current list)
   - Start with FMA kernel (simplest: no masks, no scalars)
   - Add converters incrementally: splat → range → broadcast → addptr → load → store → dot → reduce
   - Test each with `str_nodebug()` dumps, NOT with GPU dispatch yet
3. **Implement the bridge passes** (`PrepareSPIRV.cpp`; search for the pass structs and pattern classes to find the current set)
   - Must run BEFORE upstream `convert-*-to-spirv`
   - Key insight: upstream SPIR-V passes assume structured IR; Triton's pointer-heavy IR needs bridge
4. **Implement VulkanizePass** (`PrepareSPIRV.cpp`; search for `struct VulkanizePass` or `class VulkanizePass`)
   - This is the critical innovation: func args → Vulkan GlobalVariables/builtins
   - Without it, NVIDIA driver crashes at pipeline creation
5. **Implement VulkanCompute runtime** (`VulkanCompute.{h,cpp}`)
   - VkInstance → VkDevice → VkBuffer → VkShaderModule → VkPipeline → vkCmdDispatch
6. **Wire Python pipeline** (compiler.py) + pybind11 (triton_vulkan.cc)
7. **First GPU test**: vector_add (verify end-to-end)

### Phase 2: Performance (perf skill — `triton-windows-vulkan-perf`)

Follow C+1 → C+6 **in order**. Each builds on the previous:

| Step | What You Learn | Infrastructure Created |
|------|---------------|----------------------|
| C+1 | Placeholder pattern, builtin variables | WorkgroupId, `emitError` pattern |
| C+2 | Vulkan memory management | Staging buffers, device-local |
| C+3 | Shared memory, barriers, module attributes | ConvertReductionToParallel, shared var promotion |
| C+4 | Subgroup operations | GroupNonUniform ops, combiner classification |
| C+5 | Buffer-forwarding architecture | traceToFuncArg, attribute-based operand passing |
| C+6 | Device selection, perf baseline | GPU scoring, compile/dispatch timing |

**Do NOT skip steps.** C+5 depends on the placeholder pattern (C+1/C+3),
module attributes (C+3), and GlobalVariable manipulation (C+1/C+3).

### What to Skip

- **OpenCL emitters** — optional debugging aids. Debug via `str_nodebug()` instead.
- **Path A (TTGIR→LLVM→SPIR-V)** — not viable for Turing. See `development/intel-xpu-backend-study.md`.

---

## 10. Lessons Learned

1. **Bridge passes are standard practice** — IREE, Intel XPU all do this. The
   MLIR SPIR-V dialect is fine; the upstream bridges assume different IR shapes.
2. **VulkanizePass is the critical innovation** — func args → descriptor bindings.
   Without it, NVIDIA driver crashes at pipeline creation.
3. **`str_nodebug()` for parsing** — `str()` includes locs that corrupt regex.
4. **Start with simplest kernel** — FMA first (no masks, no scalars). Add
   complexity incrementally.
5. **Don't pass TypeConverter to all patterns** — it changes adaptor behavior.
   Only AtomicRMWConverter needs it.
6. **OpenCL was scaffolding** — useful for debugging, not a prerequisite.
   New developers should skip it and debug via MLIR IR dumps.

---

## 11. Adapting to Upstream Changes

This skill documents patterns and principles, not exact code locations.
When upstream Triton or MLIR changes:

1. **Search, don't assume line numbers.** Use `rg "OpConversionPattern|patterns.add<|struct .*Pass|class VulkanizePass" third_party\vulkan` to find the current converter and pass definitions.
2. **Verify arg layout.** Check `TRITON_PROGRAM_INFO_ARG_COUNT` in `third_party\vulkan\lib\Conversion\TritonToLinalgPass.cpp`; if it changed, update all `get_program_id`, `get_num_programs`, `local_id`, and push-constant index math.
3. **Verify test suite.** Run `python third_party\vulkan\test\test_kernels_vulkan.py` — the test count may change, but all tests should pass.
4. **Verify pass ordering.** Print IR after each stage with `str_nodebug()` and look for unconverted `memref.*`, `gpu.*`, or `func.*` ops.
5. **Verify VCE triple.** Check that `spirv.module` capabilities match the ops actually emitted for basic shaders, subgroup ops, and cooperative matrix ops.
