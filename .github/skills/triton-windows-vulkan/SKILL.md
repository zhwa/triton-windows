---
name: triton-windows-vulkan
description: "Foundation Vulkan/SPIR-V backend for triton-windows. Covers the full TTIR→SPIR-V→Vulkan dispatch pipeline: 16 TritonToLinalg converters, 7 bridge passes, VulkanizePass, VulkanCompute runtime, and 18 documented traps. For performance improvements (C+ steps: WorkgroupId, device-local memory, shared memory), see `triton-windows-vulkan-perf`."
argument-hint: "pipeline | converters | bridge-passes | vulkanize | push-constants | runtime | traps | diagnostics"
user-invocable: true
---

# Triton-Windows Vulkan/SPIR-V Backend

Complete guide to the Vulkan backend for triton-windows. Covers the full
pipeline from Triton IR to GPU execution via native Vulkan compute dispatch.

**11/11 kernels verified on RTX 2080 Ti via Vulkan SPIR-V dispatch (incl. multi-block + 65k).**

## 1. Architecture

```
TTIR → make_ttir → make_linalg → make_memref → make_spirv → make_spv → VulkanCompute → GPU
         shared      C++ pass     bufferize     bridge+convert  serialize   Vulkan dispatch
         passes      16 converters  +loops+cf    +vulkanize      C++ API     vkCmdDispatch
```

### File Layout

| File | Purpose |
|------|---------|
| `lib/Conversion/TritonToLinalg.cpp` | 16 converters: Triton→Linalg/Tensor/MemRef |
| `lib/Conversion/TritonToLinalgPass.cpp` | Pass wrapper, TritonTypeConverter, illegal ops |
| `lib/Conversion/PrepareSPIRV.cpp` | 7 bridge passes + VulkanizePass |
| `lib/Runtime/VulkanCompute.{h,cpp}` | Vulkan compute dispatch engine |
| `triton_vulkan.cc` | pybind11 module exposing passes + runtime |
| `backend/compiler.py` | Python pipeline orchestration |
| `test/test_kernels_vulkan.py` | 11-kernel Vulkan GPU test suite (incl. multi-block + large-N) |

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

## 2. TritonToLinalg Converters (16 total)

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
| GetProgramIDConverter | `tt.get_program_id` | Function arg (last 3 i32s = pid x,y,z) |
| GetNumProgramsConverter | `tt.get_num_programs` | Function arg |
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
This is the most complex pass — it handles six responsibilities:

**a) Buffer args → GlobalVariables:**
StorageBuffer pointer args → `spirv.GlobalVariable` with `bind(0, N)`

**b) WorkgroupId builtin (C+1):**
Last 3 scalar args (program_id) → `spirv.GlobalVariable @__builtin_workgroup_id`
with `BuiltIn WorkgroupId` + `CompositeExtract` per axis

**c) LocalInvocationId builtin (C+3):**
Next 3 scalar args (local_id) → `spirv.GlobalVariable @__builtin_local_invocation_id`
with `BuiltIn LocalInvocationId` + `CompositeExtract`

**d) Push constants:**
Remaining scalar args (N, num_programs, etc.) → `PushConstant` struct

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

### MLIR API Traps (LLVM commit 87717bf)

| API | Correct Usage |
|-----|---------------|
| `linalg::ReduceOp` | `ReduceOp::create()` static method with body builder |
| `arith::ConstantOp` | Requires `TypedAttr`: `cast<TypedAttr>(...)` |
| `FunctionInterfaces.h` | Path: `mlir/Interfaces/FunctionInterfaces.h` |
| `bufferization::ToTensorOp` | Explicit result type arg: `create(loc, tensorType, buf, ...)` |
| `bufferization::ToBufferOp` | NOT `ToMemrefOp` in this LLVM version |
| `ResourceLimitsAttr::get` | 4th arg: `Builder(ctx).getI32ArrayAttr({128,128,64})` |

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
| `pickPhysicalDevice` no discrete preference | `VulkanCompute.cpp` | May select integrated GPU over discrete |
| `LowerUnrankedCast` dead code | `PrepareSPIRV.cpp` | Defined but always returns `failure()`; never contributes |
| `vkQueueWaitIdle` per transfer | `VulkanCompute.cpp` copyBuffer | Serializes transfers; optimize with batched command buffers later |
| `ConvertReductionToParallel` 1D-only | `PrepareSPIRV.cpp` | Only handles 1D static power-of-2 reductions |

For performance-related items (shared memory, subgroups, cooperative matrix), see
the `triton-windows-vulkan-perf` skill.

---

## 9. Lessons Learned

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
