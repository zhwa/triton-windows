# Intel XPU Backend for Triton: Comprehensive Study

**Purpose:** A thorough analysis of Intel's SPIR-V backend implementation,
examined as a reference for building a real SPIR-V backend for triton-windows
(beyond the current Linalg→SPIR-V pipeline).

This study was conducted at a specific point in time. The architectural
conclusions remain valid, but specific code references may have drifted.

**Audience:** Compiler engineers working on the Vulkan/SPIR-V backend who
want to understand what a production-grade SPIR-V backend looks like, what
we can borrow from Intel's approach, and what's fundamentally different
about our NVIDIA+Vulkan target vs Intel's native XPU target.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Compilation Pipeline: Stage by Stage](#3-compilation-pipeline-stage-by-stage)
4. [The TritonIntelGPU Dialect](#4-the-tritonintelgpu-dialect)
5. [TTGIR → LLVM-IR Conversion](#5-ttgir--llvm-ir-conversion)
6. [LLVM-IR → SPIR-V Translation](#6-llvm-ir--spir-v-translation)
7. [The TritonGEN Dialect and SPIR-V Lowering](#7-the-tritongen-dialect-and-spir-v-lowering)
8. [Runtime: SYCL + Level Zero](#8-runtime-sycl--level-zero)
9. [Key Design Patterns Worth Borrowing](#9-key-design-patterns-worth-borrowing)
10. [What's Fundamentally Different for Us](#10-whats-fundamentally-different-for-us)
11. [Feasibility Assessment for triton-windows](#11-feasibility-assessment-for-triton-windows)
12. [Recommended Approach](#12-recommended-approach)
13. [Appendix: File Inventory](#13-appendix-file-inventory)

---

## 1. Executive Summary

Intel's XPU backend is the only production SPIR-V backend for Triton.
It follows Triton's standard Route A pipeline:

```
TTIR → TTGIR → LLVM-IR (MLIR) → LLVM-IR (LLVM) → SPIR-V binary
```

Key insights for our project:

1. **Intel does NOT use MLIR's SPIR-V dialect for code generation.** They go
   TTGIR → LLVM-IR → SPIR-V via the `llvm-spirv` translator (or LLVM's SPIR-V
   backend). The MLIR SPIR-V dialect is only used for barriers and subgroup ops.

2. **The heavy lifting is in TTGIR → LLVM-IR.** This is where thread/workgroup
   mapping, shared memory allocation, matrix multiply (DPAS), and memory
   coalescing happen. It's a large codebase across dozens of files.

3. **The SPIR-V translation itself is thin** — `SPIRVTranslation.cpp` is a
   small amount of code. The real complexity is upstream in LLVM-IR generation.

4. **For our NVIDIA+Vulkan target, we cannot reuse Intel's TTGIR.** Their
   TTGIR encodes Intel-specific concepts (DPAS layouts, subgroup sizes,
   2D block loads, GRF allocation) that don't exist on NVIDIA hardware.

5. **The most transferable piece is the architecture pattern** — not the code.
   How they structure `compiler.py`, the pass pipeline, TritonGEN as an
   intermediate dialect, and the SPIR-V serialization approach.

---

## 2. Architecture Overview

### Repository Structure

```
intel-xpu-backend-for-triton/
├── third_party/intel/              # The actual backend
│   ├── backend/                    # Python: compiler.py, driver.py, driver.c
│   │   ├── compiler.py             # XPUBackend(BaseBackend) — pipeline stages
│   │   ├── driver.py               # Python runtime (kernel launch, memory)
│   │   ├── driver.c                # C runtime (SYCL + Level Zero bindings)
│   │   └── arch/                   # Architecture-specific submodules
│   ├── include/                    # C++ headers for Intel passes/dialects
│   │   ├── Dialect/                # TritonIntelGPU, TritonGEN dialect defs
│   │   ├── TritonIntelGPUToLLVM/   # TTGIR → LLVM conversion headers
│   │   ├── TritonGENToSPIRV/       # TritonGEN → SPIR-V barrier lowering
│   │   ├── TritonGENToLLVM/        # TritonGEN → LLVM intrinsic lowering
│   │   └── Target/SPIRV/           # SPIR-V translation API
│   ├── lib/                        # C++ implementations
│   │   ├── Dialect/TritonIntelGPU/ # Intel GPU dialect (DPAS, layouts)
│   │   ├── TritonIntelGPUToLLVM/   # TTGIR → LLVM conversion (30+ files)
│   │   ├── TritonIntelGPUTransforms/ # TTGIR optimization passes
│   │   ├── TritonGENToSPIRV/       # Barrier → spirv.ControlBarrier
│   │   ├── TritonGENToLLVM/        # Hardware intrinsic lowering
│   │   └── Target/SPIRV/           # LLVM-IR → SPIR-V translation
│   └── triton_xpu.cc              # Pybind11 module registration
├── lib/                            # Upstream Triton core (shared)
│   ├── Conversion/TritonToTritonGPU/  # TTIR → TTGIR (shared across backends)
│   ├── Conversion/TritonGPUToLLVM/    # TTGIR → LLVM (shared patterns)
│   └── Dialect/                       # Core Triton dialects
└── python/triton/backends/         # Backend discovery (entry points)
```

### Backend Registration

Intel registers as the `xpu` target via Python entry points:

```python
class XPUBackend(BaseBackend, metaclass=XPUBackendMeta):
    binary_ext = "spv"
    target_arch = "spir64"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'xpu'
```

The `XPUBackendMeta` metaclass supports architecture-specific subclasses
(e.g., `arch.bmg`, `arch.pvc`) that override device-specific parameters.

---

## 3. Compilation Pipeline: Stage by Stage

### Pipeline Registration

```python
def add_stages(self, stages, options, language):
    stages["ttir"]  = lambda src, metadata: self.make_ttir(src, metadata, options)
    stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.properties)
    stages["llir"]  = lambda src, metadata: self.make_llir(src, metadata, options)
    stages["spv"]   = lambda src, metadata: self.make_spv(src, metadata, options)
    if options.generate_native_code:
        stages["zebin"] = lambda src, metadata: self.make_zebin(src, metadata, options)
```

The pipeline stages are `ttir → ttgir → llir → spv → zebin` (optional native binary).

### Stage 1: `make_ttir` — Triton IR Optimization

**Input:** Raw Triton IR from the frontend.
**Output:** Optimized Triton IR.

Intel-specific TTIR passes (beyond upstream):
- `rewrite_tensor_descriptor_to_pointer` — Intel tensor descriptor support
- `remove_masks` — Simplify masked operations
- `stride_versioning` — Specialize for stride patterns
- `fuse_reshape` — Fuse reshape operations
- `fold_true_cmpi` — Constant fold comparisons
- `simplify_signed_arithmetic` — Clean up signed integer ops

Standard upstream passes: inliner, combine, reorder_broadcast, CSE, DCE, loop_unroll.

### Stage 2: `make_ttgir` — GPU Scheduling

**Input:** Optimized TTIR.
**Output:** TritonGPU IR with hardware-specific layouts and scheduling.

This is the most complex stage (~65 lines of pass registration). Key steps:

```python
# 1. Annotate module with hardware capabilities
intel.passes.ttgpuir.add_triton_annotate_module(pm, module_opts)

# 2. Convert TTIR → TTGIR (upstream, shared with all backends)
passes.ttir.add_convert_to_ttgpuir(pm, "xpu", num_warps, warp_size, num_ctas)

# 3. Optimize data layout
passes.ttgpuir.add_coalesce(pm)
intel.passes.ttgpuir.add_widen_load_store_encoding(pm)
intel.passes.ttgpuir.add_remove_layout_conversions(pm)

# 4. Accelerate matrix multiply (DPAS)
intel.passes.ttgpuir.add_accelerate_matmul(pm)
intel.passes.ttgpuir.add_materialize_block_pointer(pm)

# 5. Software pipeline (multi-stage)
intel.passes.ttgpuir.add_pipeline(pm, num_stages, use_barrier)

# 6. Further optimization
passes.ttgpuir.add_optimize_thread_locality(pm)
passes.ttgpuir.add_prefetch(pm)
intel.passes.ttgpuir.add_annotate_cache_control(pm)
passes.ttgpuir.add_reorder_instructions(pm)
```

**Hardware annotation** (`annotate_module`) sets module attributes:
- `ttig.target_arch` — Device architecture (e.g., "pvc", "bmg")
- `ttig.min_sg_size` — Minimum subgroup size (8, 16, or 32)
- `ttig.support_subgroup_matrix_multiply_accumulate` — DPAS available
- `ttig.support_2d_block_io` — 2D block load/store available

**DPAS acceleration** (`accelerate_matmul`) rewrites `tt.dot` operations
to use Intel's `#ttig.dpas` encoding, which maps to DPAS/XMX hardware
instructions. The encoding captures:
- `executionSize` (8 or 16 lanes)
- `systolicDepth` (8)
- `repeatCount` (8)
- `opsChanBitWidths` (32)

### Stage 3: `make_llir` — LLVM-IR Generation

**Input:** TritonGPU IR with Intel-specific layouts.
**Output:** LLVM-IR as a string.

This stage has two sub-phases:

**Phase A: MLIR-level lowering (TTGIR → LLVM dialect)**
```python
intel.passes.ttgpuir.add_lower_to_2d_block_load(pm)    # 2D block I/O
passes.convert.add_scf_to_cf(pm)                        # Loops → branches
passes.convert.add_index_to_llvmir(pm)                   # Index → LLVM i64
intel.passes.ttgpuir.add_allocate_shared_memory(pm)      # SLM allocation
intel.passes.ttgpuir.add_to_llvmir(pm)                   # Main TTGIR → LLVM
intel.passes.ttgpuir.add_gen_to_llvm(pm)                 # TritonGEN → LLVM
passes.convert.add_arith_to_llvmir(pm)                   # Arith → LLVM
```

**Phase B: MLIR → LLVM native module**
```python
llvm.init_targets()
context = llvm.context()
llvm_mod = llvm.to_module(mod, context)
intel.set_fast_math(llvm_mod, metadata['enable_fp_fusion'])
cls.optimize_llvm_mod(llvm_mod, options)
intel.post_process_llir(llvm_mod)
ret = str(llvm_mod)
```

The `ConvertTritonGPUToLLVM` pass (TritonGPUToLLVM.cpp) is the heart
of code generation. It:
1. Registers `spirv::SPIRVDialect` as a dependent dialect
2. Allocates shared memory (`global_smem`, `__local` in OpenCL terms)
3. Runs `ModuleMembarAnalysis` to insert memory barriers
4. Converts functions (calling conventions, arg attributes)
5. Converts all TTGIR ops to LLVM dialect ops via pattern rewriting

### Stage 4: `make_spv` — SPIR-V Serialization

**Input:** LLVM-IR string.
**Output:** SPIR-V binary bytes.

```python
@classmethod
def make_spv(cls, src, metadata, options):
    spirv, name = intel.translate_to_spirv(src)
    metadata["name"] = name
    # GRF mode flags (128, 256, 512, auto)
    if options.grf_mode == '256':
        metadata["build_flags"] += " -cl-intel-256-GRF-per-thread"
    return spirv
```

The `translate_to_spirv` function calls into `SPIRVTranslation.cpp`:

```cpp
std::string translateLLVMIRToSPIRV(llvm::Module &module) {
    SPIRV::TranslatorOpts SPIRVOpts = getSPIRVOpts();
    // Two paths: LLVM SPIR-V backend or llvm-spirv translator
    auto success = SpvTranslateMode
        ? llvm::runSpirvBackend(&module, OS, Err, SPIRVOpts)
        : llvm::writeSpirv(&module, SPIRVOpts, OS, Err);
    return result;
}
```

Two SPIR-V translation backends are supported:
1. **`llvm-spirv` translator** (default) — The Khronos SPIRV-LLVM-Translator
   library. Mature, well-tested.
2. **LLVM SPIR-V backend** (opt-in via `TRITON_USE_SPIRV_BACKEND=1`) — LLVM's
   native SPIR-V target backend. Newer, potentially faster.

### Stage 5: `make_zebin` — Native Binary (Optional)

Uses Intel's `ocloc` offline compiler to convert SPIR-V → native binary:

```python
ocloc_cmd = ['ocloc', 'compile', '-file', fsrc.name, '-o', fbin,
             '-spirv_input', '-device', cls.device_arch, ...]
```

This is equivalent to JIT-compiling the SPIR-V for a specific GPU model.

---

## 4. The TritonIntelGPU Dialect

### Purpose

The `TritonIntelGPU` dialect (`ttig`) extends the upstream `TritonGPU`
dialect with Intel-specific concepts:

- **DPAS encoding** (`#ttig.dpas`) — Maps to Intel's Dot Product
  Accumulate Systolic instructions (XMX engine)
- **Subgroup 2D block encoding** — For 2D block load/store operations
- **Warp encoding** — Intel-specific warp (subgroup) layout

### DPAS Layout

The `DPASCapability` attribute models hardware limits:

```tablegen
def DPASCapability : Attr {
    int executionSize;      // 8 or 16 (number of SIMD lanes)
    int systolicDepth;      // 8 (fixed for current hardware)
    int repeatCount;        // 8 (number of repeat operations)
    int opsChanBitWidths;   // 32 (operand channel bit width)
}
```

The `DPAStoLinearLayout` function maps DPAS encoding to Triton's
`LinearLayout` framework (register/lane/warp dimensions), enabling
automatic data marshaling between memory layouts and compute layouts.

### Module Annotations

The `annotate_module` pass attaches hardware capabilities as module
attributes, enabling subsequent passes to make hardware-aware decisions:

```
module attributes {
  ttig.target_arch = "pvc",
  ttig.min_sg_size = 16,
  ttig.support_subgroup_matrix_multiply_accumulate = true,
  ttig.support_2d_block_io = true
}
```

### Key Passes (from Passes.td)

| Pass | Purpose |
|------|---------|
| `tritonintelgpu-accelerate-matmul` | Rewrite `tt.dot` to DPAS encoding |
| `tritonintelgpu-remove-layout-conversions` | Eliminate redundant layout changes |
| `tritonintelgpu-pipeline` | Software pipelining (multi-stage) |
| `tritonintelgpu-materialize-block-pointer` | 2D block pointer → 2D block I/O |
| `tritonintelgpu-widen-load-store-encoding` | Use wider (256-bit) loads/stores |
| `tritonintelgpu-optimize-reduction-locality` | Improve reduction data locality |

---

## 5. TTGIR → LLVM-IR Conversion

### The Central Pass: `ConvertTritonGPUToLLVM`

**File:** `third_party/intel/lib/TritonIntelGPUToLLVM/TritonGPUToLLVM.cpp`

This pass is the most complex component (~170 lines for the pass wrapper,
but it delegates to 30+ pattern files). Key responsibilities:

1. **Shared memory allocation**: Creates `@global_smem` with
   `TritonGEN::TritonGENMemorySpace::kWorkgroup` address space
2. **Memory barrier analysis**: `ModuleMembarAnalysis` inserts barriers
   where needed based on data dependency analysis
3. **Function lowering**: Converts `tt.func` to `llvm.func` with correct
   calling conventions
4. **Op conversion**: Delegates to type-specific patterns via
   the pattern population entry point

### Target Info Abstraction

The `SPIRVTargetInfo` class provides hardware-specific implementations:

```cpp
class SPIRVTargetInfo : public TargetInfoBase {
    // Shared memory uses Workgroup address space
    unsigned getSharedAddressSpace() { return kWorkgroup; }

    // Warp reduce uses SPIR-V subgroup ops
    Value genWarpReduce(RewriterBase &rewriter, Location loc,
                        Value acc, Operation *reduceOp,
                        unsigned numLanesToReduce, unsigned warpSize);

    // Check if an arith op has a SPIR-V subgroup equivalent
    bool isSupportedWarpReduceOp(Operation *op, ...);
};
```

This abstraction is what allows the same TTGIR → LLVM conversion
framework to target different hardware backends.

### Subgroup Operations for Reductions

Intel maps `arith::*` reduce operations to SPIR-V subgroup ops:

| Arith Op | SPIR-V Subgroup Op |
|----------|-------------------|
| `arith.addf` | `spirv.GroupNonUniformFAdd` |
| `arith.addi` | `spirv.GroupNonUniformIAdd` |
| `arith.mulf` | `spirv.GroupNonUniformFMul` |
| `arith.maxsi` | `spirv.GroupNonUniformSMax` |
| `arith.minui` | `spirv.GroupNonUniformUMin` |
| `arith.andi` | `spirv.GroupNonUniformBitwiseAnd` |
| `arith.ori` | `spirv.GroupNonUniformBitwiseOr` |
| `arith.xori` | `spirv.GroupNonUniformBitwiseXor` |

These are true hardware-accelerated reductions using the GPU's subgroup
shuffle network — no `__local` memory or explicit barriers needed.
This is the SPIR-V equivalent of CUDA's `__shfl_xor_sync`.

---

## 6. LLVM-IR → SPIR-V Translation

### The Translation Path

Intel does NOT generate SPIR-V from MLIR's SPIR-V dialect. Instead:

```
TTGIR → LLVM dialect (MLIR) → LLVM-IR (native) → SPIR-V binary
                                                    ↑
                                                llvm-spirv translator
                                                or LLVM SPIR-V backend
```

### Why This Works

LLVM-IR has a well-defined mapping to SPIR-V:
- `llvm.func` → `OpFunction` / `OpFunctionEnd`
- `llvm.alloca` → `OpVariable` (Function storage class)
- `llvm.load` / `llvm.store` → `OpLoad` / `OpStore`
- `llvm.add` / `llvm.fadd` → `OpIAdd` / `OpFAdd`
- Address space annotations → SPIR-V storage classes
- SPIR-V intrinsics (GenISA) → vendor-specific SPIR-V extensions

### The Translator Options

```cpp
SPIRV::TranslatorOpts getSPIRVOpts() {
    SPIRVOpts.setMaxVersion(VersionNumber::SPIRV_1_4);
    SPIRVOpts.setDesiredBIsRepresentation(BIsRepresentation::SPIRVFriendlyIR);
    SPIRVOpts.setAllowedToUseExtension("SPV_INTEL_*", true);
    SPIRVOpts.setSPIRVAllowUnknownIntrinsics({"llvm.genx.GenISA."});
    return SPIRVOpts;
}
```

Key settings:
- **SPIR-V 1.4** — Required for subgroup operations
- **SPIRVFriendlyIR** — Use SPIR-V-friendly built-in representations
- **Intel extensions allowed** — Enables vendor-specific SPIR-V extensions
- **GenISA intrinsics allowed** — Intel GPU Assembly intrinsics pass through

### Target Triple

Before translation, the LLVM module's target triple is set to SPIR-V:

```python
# In compiler.py make_llir:
intel.set_spv_target_triple(llvm_mod)
# Sets to "spir64-unknown-unknown" or "spirv64v1.6-unknown-unknown"
```

The translator then corrects `spir64` to `spirv64` if needed.

---

## 7. The TritonGEN Dialect and SPIR-V Lowering

### What is TritonGEN?

`TritonGEN` is Intel's intermediate dialect for GPU-specific operations
that don't have direct LLVM-IR equivalents:

- `TritonGEN::BarrierOp` — Workgroup/subgroup barriers
- `TritonGEN::MatrixDPASOp` — Matrix multiply via DPAS hardware
- `TritonGEN::Matrix2DBlockLoadOp` — 2D block load from memory
- Various cache control and prefetch operations

### TritonGEN → SPIR-V Lowering

**File:** `third_party/intel/lib/TritonGENToSPIRV/TritonGENToSPIRVPass.cpp`
(116 lines)

This pass is surprisingly small — it only handles barrier lowering:

```cpp
struct TritonGENBarrierLowering
    : public OpConversionPattern<TritonGEN::BarrierOp> {
    LogicalResult matchAndRewrite(...) {
        switch (op.getMemFence()) {
        case TritonGEN::MemFence::LOCAL:
            memorySemantics = AcquireRelease | WorkgroupMemory;
            break;
        case TritonGEN::MemFence::GLOBAL:
            memorySemantics = SequentiallyConsistent | CrossWorkgroupMemory;
            break;
        }
        rewriter.replaceOpWithNewOp<spirv::ControlBarrierOp>(
            op, scope, scope, memorySemantics);
    }
};
```

Other TritonGEN ops (DPAS, 2D block loads) go through TritonGEN → LLVM
lowering, where they become LLVM intrinsics (`llvm.genx.GenISA.*`) that
the SPIR-V translator passes through as vendor extensions.

### The Two-Path Architecture

```
TritonGEN::BarrierOp ─────→ spirv::ControlBarrierOp (MLIR SPIR-V dialect)
                                    │
                                    └→ SPIR-V OpControlBarrier (in binary)

TritonGEN::MatrixDPASOp ──→ LLVM intrinsic (llvm.genx.GenISA.dpas.*)
                                    │
                                    └→ SPIR-V vendor extension (in binary)
```

Barriers go through MLIR's SPIR-V dialect because they have clean,
standard SPIR-V semantics. Hardware intrinsics (DPAS, block loads) go
through LLVM intrinsics because they're vendor-specific and have no
standard SPIR-V equivalent.

---

## 8. Runtime: SYCL + Level Zero

### Runtime Stack

```
Python (driver.py)
    ↓
C bindings (driver.c)
    ↓
SYCL runtime
    ↓
Level Zero API (ze_api.h)
    ↓
Intel GPU driver (IGC)
    ↓
Intel GPU hardware
```

### SPIR-V Loading

```c
// driver.c: Load SPIR-V into Level Zero module
ze_module = create_module(l0_context, l0_device, binary_ptr, binary_size, build_flags);
ze_function = create_function(ze_module, kernel_name);
// Wrap into SYCL objects
sycl_kernel_bundle = make_kernel_bundle(ze_module, sycl_context);
sycl_kernel = make_kernel(sycl_kernel_bundle, ze_function);
```

### Kernel Launch

```c
// driver.c: Launch via SYCL
global_range = gridX * threads_per_warp * num_warps;
local_range = num_warps * threads_per_warp;
cgh.parallel_for(nd_range(global_range, local_range), sycl_kernel);

// Shared memory allocation
sycl::local_accessor<int8_t, 1> local_buffer(shared_memory_size, cgh);
```

### Memory Management

Intel uses SYCL Unified Shared Memory (USM):

```c
void* dev_ptr = sycl::malloc_device<char>(nbytes, sycl_queue);
sycl_queue.memcpy(dev_ptr, host_ptr, nbytes);           // H2D
sycl_queue.memcpy(host_ptr, dev_ptr, nbytes);           // D2H
sycl::free(dev_ptr, sycl_queue);
```

### SPIRVRunner Utility

Intel provides a standalone SPIR-V execution tool:

```
SPIRVRunner -d <dump_dir> -o tensor_2 -p
```

It reads `args_data.json` + `tensor_*.pt` + `.spv`, loads via Level Zero,
and executes. This is useful for debugging without the full Triton stack.

---

## 9. Key Design Patterns Worth Borrowing

### Pattern 1: Intermediate GPU Dialect (TritonGEN)

Intel introduces `TritonGEN` as a bridge between TTGIR and LLVM/SPIR-V.
This separates concerns:
- TTGIR optimizations don't need to know about SPIR-V
- SPIR-V lowering doesn't need to know about TTGIR layouts

**For our backend:** We could introduce a `TritonVulkan` dialect for
Vulkan-specific ops (compute shader dispatch, descriptor set binding,
push constants) before lowering to SPIR-V.

### Pattern 2: TargetInfo Abstraction

The `SPIRVTargetInfo` class provides a clean interface for hardware-specific
code generation decisions (shared memory address space, warp reduce
implementation, etc.). The same conversion patterns work for different
targets by querying TargetInfo.

**For our backend:** We'd implement a `VulkanTargetInfo` that maps:
- Shared memory → SPIR-V `Workgroup` storage class
- Warp reduce → `spirv.GroupNonUniform*` ops (same as Intel)
- Thread ID → `spirv.GlobalInvocationId` builtin

### Pattern 3: LLVM-IR → SPIR-V (Not MLIR → SPIR-V)

Intel's most important lesson: **don't try to generate SPIR-V from MLIR's
SPIR-V dialect**. Instead:
1. Lower everything to LLVM-IR (MLIR's LLVM dialect)
2. Convert MLIR LLVM to native LLVM-IR
3. Use the `llvm-spirv` translator for the final step

This works because:
- LLVM-IR → SPIR-V is a well-tested, maintained translation
- MLIR's SPIR-V dialect lacks some ops needed for GPU compute
- The LLVM path inherits all of LLVM's optimization passes

**For our backend:** This is the recommended approach. Our current
`PrepareSPIRV.cpp` tries to use MLIR's SPIR-V conversion passes, which is
fragile and incomplete. The LLVM path would be more robust.

### Pattern 4: Subgroup Ops for Reductions

Intel maps reductions to SPIR-V `GroupNonUniform*` ops instead of
explicit `__local` memory + barrier tree reductions. This is:
- Faster (hardware shuffle network, no memory access)
- Simpler (one op instead of log₂(N) barrier rounds)
- Portable (SPIR-V subgroup ops are standard, not vendor-specific)

**For our backend:** We should target `spirv.GroupNonUniform*` for
reductions on NVIDIA too (via Vulkan compute). NVIDIA's Vulkan driver
supports subgroup operations via `VK_KHR_shader_subgroup_*`.

### Pattern 5: Architecture Metaclass

The `XPUBackendMeta` metaclass enables architecture-specific backend
subclasses without modifying the base backend:

```python
class XPUBackend(BaseBackend, metaclass=XPUBackendMeta):
    arch_to_impl = {}  # Auto-populated by subclasses
```

**For our backend:** Less relevant (we target one architecture via Vulkan)
but useful if we ever support multiple Vulkan device types.

---

## 10. What's Fundamentally Different for Us

### Intel vs Our Situation

| Aspect | Intel XPU | Our Vulkan Backend |
|--------|-----------|-------------------|
| **Target hardware** | Intel GPUs (native) | NVIDIA GPUs (via Vulkan) |
| **Runtime API** | SYCL + Level Zero | Vulkan Compute (vkCmdDispatch) |
| **SPIR-V flavor** | OpenCL SPIR-V (compute kernel) | Vulkan SPIR-V (compute shader) |
| **Memory model** | OpenCL SVM / USM | Vulkan descriptor sets + push constants |
| **Subgroup size** | 8, 16, or 32 (configurable) | 32 (NVIDIA warp size, fixed) |
| **Matrix multiply** | DPAS/XMX hardware | Tensor Cores (via Vulkan cooperative matrix) |
| **Shared memory** | SLM (Shared Local Memory) | Workgroup storage class |
| **Thread model** | Work-item in work-group | Invocation in workgroup |

### Key Differences in Detail

**1. SPIR-V Execution Environment**

Intel uses **OpenCL SPIR-V** (`spir64` target triple). Their kernels are
OpenCL kernels with `__kernel` entry points.

We need **Vulkan SPIR-V** (`vulkan1.2` target environment). Our shaders
are compute shaders with `OpEntryPoint GLCompute` and `OpExecutionMode
LocalSize`. This is a fundamentally different SPIR-V profile with different
capabilities, decorations, and memory model.

**2. Argument Passing**

Intel passes kernel arguments as flat scalar/pointer parameters
(OpenCL-style `__global float* arg0, int arg1`).

Vulkan requires **descriptor sets** and **push constants**. Buffer
arguments go through `VkDescriptorSetLayout` bindings. Small scalar
arguments use push constants. This requires SPIR-V decorations
(`OpDecorate %var DescriptorSet 0`, `OpDecorate %var Binding N`).

**3. No Direct Pointer Arithmetic**

OpenCL SPIR-V supports `OpPtrAccessChain` for pointer arithmetic.
Vulkan SPIR-V (before SPIR-V 1.6 with `PhysicalStorageBuffer`) requires
all buffer access through `OpAccessChain` on descriptor-bound variables.
Our `memref.reinterpret_cast` → pointer+offset pattern would need to become
buffer[index] accesses.

**4. TTGIR is Hardware-Specific**

Intel's TTGIR encodes Intel GPU architecture details:
- DPAS layouts (8×8 systolic array shape)
- 2D block load/store (Intel-specific memory ops)
- GRF allocation modes (128/256/512 registers per thread)
- Subgroup sizes (8/16/32, configurable per kernel)

None of these concepts exist on NVIDIA hardware. We cannot reuse Intel's
TTGIR passes. We would need NVIDIA-specific TTGIR passes (which already
exist in upstream Triton as `TritonNvidiaGPU`).

**5. Translation Path**

Intel: LLVM-IR → `llvm-spirv` → OpenCL SPIR-V
Us: LLVM-IR → ??? → Vulkan SPIR-V

The `llvm-spirv` translator emits OpenCL-flavored SPIR-V by default.
Getting Vulkan-flavored SPIR-V requires either:
- A different translator (`spirv-link` with Vulkan target)
- Post-processing the SPIR-V to add Vulkan decorations
- Using LLVM's native SPIR-V backend with Vulkan triple

---

## 11. Feasibility Assessment for triton-windows

### What Would a Real SPIR-V Backend Require?

To build a production-grade SPIR-V backend for NVIDIA via Vulkan, we need:

#### Must Have (Equivalent to Intel's Implementation)

1. **TTIR → TTGIR conversion** — Already exists in upstream Triton
   (`TritonToTritonGPU`). We'd use the NVIDIA variant.

2. **TTGIR → LLVM-IR conversion** — Already exists upstream
   (`TritonNvidiaGPUToLLVM`). But it targets NVPTX, not SPIR-V. We'd
   need to fork/modify it to target SPIR-V address spaces and built-in
   variables instead of PTX-specific intrinsics.

3. **LLVM-IR → Vulkan SPIR-V translation** — The `llvm-spirv` translator
   can emit Vulkan-flavored SPIR-V with the right target environment, but
   this path is less tested than the OpenCL path. LLVM's SPIR-V backend
   with `--vulkan` target is another option.

4. **Vulkan runtime** — Need a runtime that creates Vulkan compute
   pipelines, allocates device memory via `VkBuffer`, binds descriptor
   sets, and dispatches compute shaders. This is ~2000-5000 lines of C++
   wrapping the Vulkan API (vs Intel's SYCL/Level Zero wrapper).

5. **NVIDIA-to-SPIR-V op mapping** — Map NVIDIA-specific ops
   (tensor core `mma`, shared memory `cp.async`, warp shuffle) to
   SPIR-V equivalents or polyfills.

#### Estimated Effort

| Component | Intel's LOC | Our Estimate | Difficulty |
|-----------|------------|-------------|------------|
| compiler.py (pipeline) | ~600 | ~400 | Medium |
| TTGIR passes (Intel-specific) | ~15,000 | 0 (use upstream NVIDIA) | N/A |
| TTGIR → LLVM conversion | ~30,000 | ~5,000 (modify NVIDIA→SPIR-V) | **Very Hard** |
| SPIR-V translation | ~200 | ~200 | Easy |
| TritonGEN equivalents | ~3,000 | ~1,000 | Medium |
| Vulkan runtime (driver.c) | ~500 | ~3,000 | Hard |
| Vulkan runtime (driver.py) | ~400 | ~300 | Medium |
| Tests | ~2,000 | ~1,000 | Medium |

**Total: ~10,000-15,000 lines** of new/modified C++ and Python code.
The critical difficulty is in the TTGIR → LLVM conversion, which requires
deep knowledge of both Triton's GPU abstractions and SPIR-V/Vulkan
compute semantics.

---

## 12. Updated Feasibility Assessment (June 2025)

### Status Update

Since the original study, significant progress has been made:
- **Path C is complete.** VulkanizePass, push constants, VulkanCompute
  runtime all working. All kernels dispatching via native Vulkan SPIR-V.
- **Path B is complete.** Serial and parallel OpenCL emitters are working.
  Useful for debugging but not production.
- The question now is: **should we pursue Path A (TTGIR→LLVM→SPIR-V)?**

### Deep Analysis: Why Intel's Approach Doesn't Transfer

The original §12 recommended "Path A as the next upgrade." After a deeper
study of the actual intrinsic dependencies, **this recommendation is revised.**

#### Intel's Code Is 80% Intel-Specific

| Component | Scale | Intel-Specific? |
|-----------|-------|-----------------|
| `TritonIntelGPUToLLVM/` | Large | ~80% (DPAS, 2D block load, Xe asm, SPV_INTEL_*) |
| `TritonNVIDIAGPUToLLVM/` | Large | ~85% (PTX inline asm, NVVM intrinsics) |
| Generic `TritonGPUToLLVM/` | Substantial generic implementation | ~55-60% generic, 40-45% vendor stubs |

Intel's path cannot be "followed." Their code assumes Intel hardware
intrinsics that don't exist on NVIDIA: DPAS matrix engines, 2D block
load/store, Xe assembly format, SPV_INTEL_* extensions.

#### NVIDIA's Code Is Equally Vendor-Locked

Forking NVIDIA's `TritonNVIDIAGPUToLLVM` (the actual Path A proposal)
requires replacing **every PTX intrinsic** with a SPIR-V equivalent:

| NVIDIA Intrinsic Category | Count | SPIR-V Equivalent | On Turing (RTX 2080 Ti)? |
|--------------------------|-------|-------------------|--------------------------|
| Thread/block IDs | ~10 | `gl_LocalInvocationID`, `gl_WorkGroupID` | ✅ Yes |
| Basic barriers (`bar.sync`) | ~5 | `OpControlBarrier` | ✅ Yes |
| Shared memory load/store | ~20 | `Workgroup` storage class | ✅ Yes |
| Atomics (`atom.*`) | ~10 | `OpAtomic*` | ✅ Yes |
| Warp shuffle (`shfl.sync`) | ~8 | `OpGroupNonUniformShuffle` | ✅ Yes (subgroup) |
| Warp vote | ~3 | `OpGroupNonUniformAll/Any` | ✅ Yes |
| Tensor cores (`mma.sync`) | ~15 | `VK_KHR_cooperative_matrix` | ✅ Yes (Turing supports it) |
| Async copy (`cp.async`) | ~10 | ❌ None | ❌ N/A (Ampere+ only anyway) |
| TMA (`cp.async.bulk.tensor`) | ~15 | ❌ None | ❌ N/A (Hopper+ only) |
| Async barriers (`mbarrier.*`) | ~10 | ❌ None | ❌ N/A (Ampere+ only) |
| Cache control (evict, L2 hint) | ~8 | ❌ None | ❌ Vulkan doesn't expose |
| PTX inline asm (misc) | ~50+ | Must be rewritten case-by-case | ⚠️ Partial |

**The problem is deeper than intrinsic replacement.** The entire code
generation strategy in `TritonNVIDIAGPUToLLVM` is designed around PTX:
- **Layout encodings** (blocked, MMA, shared) assume warp/SM topology
- **Memory coalescing** algorithms assume warp-level access patterns
- **Software pipelining** uses `cp.async` (no SPIR-V equivalent)
- **Warp specialization** is deeply NVIDIA-specific
- **Register allocation hints** (`.reg`, `.pred`) are PTX-specific

You can't find-and-replace intrinsics. You'd be rewriting the code
generation strategy for ~7,000 lines of C++.

#### LLVM→SPIR-V Translation Has Its Own Costs

Even if you got clean LLVM-IR, translating to Vulkan SPIR-V requires:

1. **SPIRV-LLVM-Translator** — external dependency, Intel patches it
   with custom patches (`3122.patch`, `revert_3609.patch`)
2. The translator emits **OpenCL-flavored SPIR-V** by default — getting
   Vulkan-flavored SPIR-V needs the LLVM SPIR-V backend with Vulkan triple
3. Our LLVM build **doesn't include the SPIR-V backend target** — we'd
   need to add `SPIRV` to `LLVM_TARGETS_TO_BUILD`
4. The LLVM SPIR-V backend is conditional on `LLVM_SPIRV_BACKEND_TARGET_PRESENT`
   — another build configuration to manage

#### The Generic TritonGPU→LLVM Can't Stand Alone

The generic `TritonGPUToLLVM` implementation is ~55-60% generic but
**requires a vendor backend** to fill the remaining 40-45%:
- `TargetInfo` interface: thread/block IDs, barrier, shuffle, printf
- Vendor-specific allocation: shared memory sizing, register pressure
- Layout-to-hardware mapping: which encoding uses which instructions

Creating a "generic SPIR-V TargetInfo" is possible but amounts to
writing a new vendor backend (~3-5K lines of C++).

#### triton-shared Is Not an Alternative

Microsoft's triton-shared (TTIR→Linalg converter with 35+ converters)
is **no longer maintained** (README states this explicitly). Last commit
December 2025. No SPIR-V output path. Not viable as a foundation.

### Revised Path Assessment

| Path | Pipeline | Effort | Perf Ceiling | Risk | Recommended? |
|------|----------|--------|-------------|------|-------------|
| **A** | TTGIR → LLVM → SPIR-V (fork NVIDIA) | ~10-15K lines, 3-6 months | 50-80% CUDA | **Very High** | ❌ Not now |
| **A-lite** | Same but SM 7.x only, no TMA/async | ~8K lines, 2-3 months | 30-50% CUDA | High | ❌ Poor ROI |
| **B** | TTIR → Linalg → OpenCL C text | Done | 0.04-10% CUDA | None | ✅ Done (debug tool) |
| **C** | TTIR → Linalg → MLIR SPIR-V → Vulkan | Done (all kernels) | 5-20% CUDA | None | ✅ Done |
| **C+** | Path C + incremental GPU features | ~2-4K lines, 4-8 weeks | **20-40% CUDA** | **Low** | ✅ **Recommended** |

### Path C+ Details (Completed)

The MLIR SPIR-V pipeline has been enhanced with GPU compute features through
incremental steps (C+1 through C+5), all now complete and passing the current
test suite. No TTGIR, no LLVM-IR, no vendor-specific code — just standard MLIR
passes and Vulkan SPIR-V extensions that NVIDIA Turing supports.

**Confirmed: RTX 2080 Ti (Turing) supports these Vulkan extensions:**
- `VK_KHR_cooperative_matrix` (tensor cores via Vulkan!)
- `VK_KHR_shader_subgroup_*` (warp-level ops via subgroup)
- `VK_KHR_shader_float16_int8`
- `VK_KHR_16bit_storage`, `VK_KHR_8bit_storage`
- `SPV_KHR_cooperative_matrix`

**Incremental improvement roadmap:**

| Step | Feature | SPIR-V Mechanism | Status |
|------|---------|-----------------|--------|
| C+1 | Multi-threaded workgroups | `WorkgroupId` builtin | ✅ Done |
| C+2 | Device-local memory + staging | `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | ✅ Done |
| C+3 | Workgroup shared memory | `Workgroup` storage class + barriers | ✅ Done |
| C+4 | Subgroup ops for reductions | `OpGroupNonUniformFAdd/FMax` | ✅ Done |
| C+5 | Cooperative matrix for matmul | `OpCooperativeMatrixMulAddKHR` | ✅ Done |
| C+6 | Discrete GPU selection | `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` | ✅ Done |

**Why C+ beats Path A for our situation:**

1. **Lower risk.** Each step is independently testable. Step C+1 alone
   could give 10-50× speedup on elementwise kernels.
2. **No new dependencies.** No SPIRV-LLVM-Translator, no LLVM SPIR-V
   backend, no vendor-specific dialect.
3. **Uses proven infrastructure.** Our VulkanizePass + VulkanCompute
   runtime are already working. We're extending, not rewriting.
4. **RTX 2080 Ti benefits limited from Path A.** Turing has no TMA,
   no async copy, no wgmma — the features that make Path A worthwhile
   are Ampere/Hopper features we can't use anyway.
5. **cooperative_matrix via MLIR SPIR-V.** MLIR has `spirv.KHR.CooperativeMatrixMulAdd`
   ops. We can emit them directly from our Linalg matmul, no LLVM-IR needed.

### When Path A Becomes Worth It

Path A makes sense only when ALL of these are true:
- Target GPU is Ampere+ (async copy, `mma.sync.aligned.m16n8k16`)
- Need software pipelining (Triton's key optimization)
- Need >50% CUDA performance
- Have 2-3 months of dedicated compiler engineering
- Have deep Triton internals expertise

For the RTX 2080 Ti on Windows, **Path C+ is the right ceiling.**
Path A would be the next step if you later target datacenter GPUs
(A100, H100) where async copy and TMA provide the real perf gains.

### Path A: What It Would Actually Take (For Future Reference)

If/when Path A is pursued, here's the honest breakdown:

1. **Add SPIR-V to LLVM build** — Add `SPIRV` to `LLVM_TARGETS_TO_BUILD`
   in `build-llvm.ps1`. Rebuild LLVM (~2 hours).

2. **Build SPIRV-LLVM-Translator** — Clone KhronosGroup repo, apply
   Intel's patches, build as static library.

3. **Create SPIRVTargetInfo** — New `TargetInfo` implementation (~800 lines):
   - Thread/block IDs → SPIR-V built-in variables
   - `barrier()` → `OpControlBarrier`
   - `shfl.sync` → `OpGroupNonUniformShuffle`
   - `printf` → stub or `spirv.DebugPrintf`

4. **Fork key conversion files** (~4 files, ~3000 lines):
   - `LoadStoreOpToLLVM.cpp` — remove PTX asm, use LLVM loads
   - `BarrierOpToLLVM.cpp` — replace with SPIR-V barriers
   - `DotOpToLLVM.cpp` — replace MMA with cooperative_matrix
   - `SPMDOpToLLVM.cpp` — replace tid/ctaid

5. **Modify address spaces** — NVPTX uses `addrspace(0-5)`, SPIR-V
   uses different numbering for Generic/Workgroup/CrossWorkgroup.

6. **Add Vulkan wrapper pass** — SPIR-V from LLVM needs entry point
   decoration, descriptor sets, push constants (similar to VulkanizePass).

7. **SPIR-V serialization** — Wire up translator: LLVM-IR → SPIR-V binary.

8. **Test with existing VulkanCompute runtime** — The runtime is reusable.

**Estimated: ~10K lines C++, ~500 lines Python, 3-4 months.**

---

## 13. Appendix: File Inventory

### Intel Backend — Critical Files

| File | Lines | Role |
|------|-------|------|
| `backend/compiler.py` | ~600 | Pipeline stages, options, pass registration |
| `backend/driver.py` | ~520 | Python runtime: launch, memory, serialization |
| `backend/driver.c` | ~450 | C runtime: SYCL + Level Zero kernel launch |
| `triton_xpu.cc` | ~250 | Pybind11 module: pass/dialect registration |

### Intel Backend — MLIR Passes

| Directory | Files | Purpose |
|-----------|-------|---------|
| `lib/TritonIntelGPUToLLVM/` | ~35 | TTGIR → LLVM (main code generation) |
| `lib/TritonIntelGPUTransforms/` | ~15 | TTGIR optimization passes |
| `lib/TritonGENToSPIRV/` | 1 | Barrier → spirv.ControlBarrier |
| `lib/TritonGENToLLVM/` | ~10 | TritonGEN → LLVM intrinsics |
| `lib/Target/SPIRV/` | 1 | LLVM-IR → SPIR-V binary |
| `lib/Dialect/TritonIntelGPU/` | ~10 | Intel GPU dialect definition |
| `lib/Dialect/TritonGEN/` | ~5 | TritonGEN dialect definition |

### Intel Backend — Key Include Files

| File | Purpose |
|------|---------|
| `include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td` | DPAS, Warp, Subgroup2DBlock encodings |
| `include/Dialect/TritonIntelGPU/Transforms/Passes.td` | Pass declarations |
| `include/TritonIntelGPUToLLVM/SPIRVTargetInfo.h` | TargetInfo for SPIR-V target |
| `include/TritonIntelGPUToLLVM/SPIRVSubgroupOps.h` | arith → SPIR-V subgroup op mapping |
| `include/Target/SPIRV/SPIRVTranslation.h` | SPIR-V translation API |

### Comparison with Our Backend

| Component | Intel (Production) | Ours (Current) |
|-----------|-------------------|----------------|
| Pipeline flow | ttir→ttgir→llir→spv→zebin | ttir→linalg→memref→opencl |
| C++ conversion code | ~50,000 lines | ~1,400 lines |
| SPIR-V generation | LLVM-IR → llvm-spirv | regex emitter → OpenCL C |
| GPU scheduling | Full (coalesce, pipeline, prefetch) | None (1:1 element mapping) |
| Subgroup ops | Hardware shuffle | __local + barrier |
| Tests | Thousands | serial and parallel OpenCL suites |
| Performance vs CUDA | ~80-95% | ~0.04-10% |
