# Triton Vulkan/SPIR-V Backend: Comprehensive Guide

**Scope:** The unified Vulkan/SPIR-V backend for triton-windows, from backend discovery and TTIR→Linalg lowering through pointer analysis, SPIR-V conversion, core operation support, emitter improvements, and testing.

**Audience:** Compiler engineers who want to understand, maintain, or extend the Vulkan/SPIR-V backend for triton-windows.

**Prerequisites:** Familiarity with MLIR concepts (dialects, passes, type conversion, pattern rewriting) and Triton's compilation model.

This document is self-contained and reflects the current unified backend organization.

> **Note on durability:** This guide describes architectural patterns and design
> rationale that remain valid across upstream changes. Specific line numbers,
> file sizes, and API names may drift — use `grep` to find current locations.
> When in doubt, print IR with `str_nodebug()` after each pipeline stage to
> verify the transformation.

---

## Table of Contents

1. [Context and Motivation](#1-context-and-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Backend Skeleton](#3-backend-skeleton)
4. [TritonToLinalg Conversion](#4-tritontolinalg-conversion)
5. [Pointer Analysis and Memory Ops](#5-pointer-analysis-and-memory-ops)
6. [Core Ops and Testing Scope](#6-core-ops-and-testing-scope)
7. [AtomicRMWConverter: Design and Implementation](#7-atomicrmwconverter-design-and-implementation)
8. [Reduction Scalar Fix: The 0-d Tensor Problem](#8-reduction-scalar-fix-the-0-d-tensor-problem)
9. [Scalar Load/Store with Unranked MemRef](#9-scalar-loadstore-with-unranked-memref)
10. [TypeConverter Materializations](#10-typeconverter-materializations)
11. [SPIR-V Conversion](#11-spir-v-conversion)
12. [PrepareSPIRV Cleanup](#12-preparespirv-cleanup)
13. [Emitter Improvements](#13-emitter-improvements)
14. [Build Integration](#14-build-integration)
15. [Testing and Diagnostics](#15-testing-and-diagnostics)
16. [Test Suite Architecture](#16-test-suite-architecture)
17. [Kernel-by-Kernel Analysis](#17-kernel-by-kernel-analysis)
18. [Path C+: Incremental Performance Improvements](#18-path-c-incremental-performance-improvements)
19. [Remaining Work and Future Directions](#19-remaining-work-and-future-directions)
20. [Lessons Learned](#20-lessons-learned)
21. [Appendix: File Inventory and Change Summary](#21-appendix-file-inventory-and-change-summary)

---

## 1. Context and Motivation

### 1.1 What Triton Is

Triton is a Python-based GPU programming language and compiler developed by
OpenAI. Users write kernels in Python using `@triton.jit`, and the Triton
compiler lowers them through a series of intermediate representations to
GPU machine code:

```
Python source (@triton.jit)
  │ AST → TTIR
  ▼
 TTIR (Triton IR — tensor-level operations)
  │ TritonToTritonGPU
  ▼
 TTGIR (TritonGPU IR — GPU-specific layout encodings)
  │ TritonGPUToLLVM
  ▼
 LLVM IR → PTX → cubin
```

Each backend (NVIDIA, AMD) implements the GPU-specific portions: layout
assignment, memory coalescing, tensor core mapping, and final code emission.

### 1.2 Why a Vulkan/SPIR-V Backend

Triton officially supports only NVIDIA (CUDA) and AMD (ROCm). A Vulkan/SPIR-V
backend enables:

- **Vendor-neutral execution** on any Vulkan 1.1+ GPU (NVIDIA, AMD, Intel,
 Qualcomm, Apple via MoltenVK)
- **Windows-native compute** without CUDA toolkit for non-NVIDIA GPUs
- **Intel Arc/Xe** support via standard Vulkan (vs. proprietary oneAPI)
- **Hardware tensor cores** via `VK_KHR_cooperative_matrix` extension

### 1.3 The Two Routes

There are two viable insertion points for a new backend:

**Route A — Native (replace TTGIR→backend):**
```
TTIR → TTGIR → [Vulkan-specific TTGIR passes] → SPIR-V
```
Consume `TritonGPU` IR directly, write Vulkan-specific lowering patterns.
This preserves layout optimizations (blocked, MMA encodings) and can map
`tl.dot` to `spv.KHR.CooperativeMatrixMulAdd`. Intel's XPU backend follows
this route. Downside: enormous implementation effort (~20+ conversion patterns
for TTGIR ops, each requiring deep understanding of both TritonGPU semantics
and SPIR-V's strict type system).

**Route B — Portable (replace at TTIR level via Linalg):**
```
TTIR → Linalg/Tensor (via TritonToLinalg) → MemRef → SPIR-V
```
Bypass TTGIR entirely. Convert Triton IR to standard MLIR dialects (Linalg,
Tensor, MemRef, Arith), then use MLIR's built-in SPIR-V conversion passes.
Much less code to write, but loses all GPU-specific optimizations — matmul
becomes a serial triple-nested loop.

**Our choice: Route B.** It is proven by `triton-ocl` (an independent project
that successfully ran matmul on NVIDIA via OpenCL C), requires 10x less code,
and produces a working end-to-end pipeline that can be optimized incrementally.
Route A remains a stretch goal for future work.

### 1.4 The Output Format Question: OpenCL C vs SPIR-V Binary

Route B produces standard MLIR dialects (memref, arith, cf, func). Two output
formats are possible:

| Output | Pros | Cons |
|--------|------|------|
| **OpenCL C** | Human-readable, debuggable, proven by triton-ocl | Deprecated API, NVIDIA-only OpenCL is limited |
| **SPIR-V binary** | Vulkan-native, vendor-neutral, forward-looking | More complex pipeline, harder to debug |

**Our approach:** Implement both. OpenCL C (via Python regex emitter) serves as
a debugging tool and reference implementation. SPIR-V binary (via MLIR's
built-in conversion passes) is the primary production path.

---

## 2. Architecture Overview

### 2.1 Directory Structure

```
third_party/vulkan/
├── backend/
│  ├── __init__.py     # Backend discovery exports
│  ├── compiler.py     # VulkanBackend: 6-stage pipeline orchestration
│  ├── driver.py      # VulkanDriver: GPU target info skeleton
│  └── emitter.py      # OpenCL C emitter (debugging output path)
├── include/Conversion/
│  └── TritonToLinalg.h   # Public API: pass factory functions
├── lib/Conversion/
│  ├── TritonToLinalg.cpp  # Main converter file: TritonToLinalg + memory ops
│  ├── TritonToLinalgPass.cpp # Pass wrapper + type converter
│  └── PrepareSPIRV.cpp   # Largest backend file: SPIR-V bridge + Vulkanization passes
├── test/
│  ├── lit.cfg.py      # Lit test configuration
│  ├── test_triton_to_linalg.mlir
│  ├── test_vector_add.ttir
│  ├── test_arith_to_spirv.mlir
│  ├── test_math_to_spirv.mlir
│  └── test_scf_to_spirv.mlir
├── tools/
│  └── vulkan-opt.py    # SPIR-V conversion + serialization wrapper
├── CMakeLists.txt      # Build rules and link libraries
└── triton_vulkan.cc     # pybind11 bindings (passes + serialization)
```

### 2.2 The 6-Stage Pipeline

```
     ┌───────────────────────────────────────────────────────────┐
Stage 1: │ make_ttir   (shared TTIR passes: inline, CSE, etc.)  │
     └────────────────────────┬──────────────────────────────────┘
                 │ TTIR (tensor-level Triton IR)
     ┌────────────────────────▼──────────────────────────────────┐
Stage 2: │ make_linalg  (C++ TritonToLinalg pass)         │
     │  tt.splat → tensor.empty + linalg.fill         │
     │  tt.dot  → linalg.matmul               │
     │  tt.load → memref.alloc + memref.copy         │
     │  tt.func → func.func                 │
     └────────────────────────┬──────────────────────────────────┘
                 │ Linalg/Tensor/MemRef IR
     ┌────────────────────────▼──────────────────────────────────┐
Stage 3: │ make_memref  (standard MLIR lowering passes)      │
     │  one_shot_bufferize  (tensor → memref)        │
     │  convert_linalg_to_loops (linalg.generic → scf.for)  │
     │  lower_affine     (affine.for → scf.for)      │
     │  convert_scf_to_cf  (scf.for → cf.br/cf.cond_br)  │
     └────────────────────────┬──────────────────────────────────┘
                 │ MemRef + Arith + CF IR
     ┌────────────────────────▼──────────────────────────────────┐
Stage 4: │ make_spirv  (SPIR-V preparation + conversion)   │
     │  prepare_spirv    (expand reinterpret_cast/copy)  │
     │  map_storage_class  (addr space 0 → StorageBuffer)  │
     │  fix_alloca      (alloca: StorageBuffer→Function) │
     │  convert_{memref,arith,cf,func}_to_spirv        │
     └────────────────────────┬──────────────────────────────────┘
                 │ SPIR-V dialect IR
     ┌────────────────────────▼──────────────────────────────────┐
Stage 5: │ make_spv   (wrap in spirv.module + serialize)    │
     │  Extract spirv.func, wrap in spirv.module       │
     │  Add spirv.EntryPoint for GLCompute          │
     │  mlir-translate --serialize-spirv → binary .spv    │
     └────────────────────────┬──────────────────────────────────┘
                 │ SPIR-V binary bytes
                 ▼
             Vulkan dispatch / runtime execution
```

`PrepareSPIRV.cpp` now hosts multiple Stage-4/Stage-5 passes rather than one
small cleanup pass: `PrepareSPIRVPass` (bridge pass),
`ConvertReductionToParallel`, `ConvertMatmulToCooperative`,
`FixAllocaStorageClassPass`, and `VulkanizePass`.

### 2.3 Relation to Triton's Backend Interface

Every Triton backend implements `BaseBackend` with these methods:

```python
class VulkanBackend(BaseBackend):
  def supports_target(target: GPUTarget) -> bool  # "vulkan" match
  def add_stages(stages, options, language=None)   # register pipeline
  def load_dialects(ctx)               # register MLIR dialects
  def parse_options(opts) -> VulkanOptions      # backend-specific options
  def hash() -> str                 # cache key
  def get_module_map() -> Dict[str, ModuleType]   # empty for now
```

The `add_stages()` method registers callable lambdas for each stage. Triton's
compiler core calls them in sequence, passing the MLIR module through each
stage. Each stage either transforms the module in-place (for MLIR passes) or
returns a new representation (for code emission).

### 2.4 Current Implementation Notes

A few details in the original split notes have changed since they were first written:

- Push-constant member offsets are computed from the actual scalar type sizes in `PrepareSPIRV.cpp`; they are no longer hardcoded as `i * 4`.
- The Vulkan runtime sets `VkApplicationInfo::pApplicationName` to `"Triton-Vulkan"`.
- `VulkanCompute::setPushConstants()` tears down dependent pipeline and descriptor state before rebuilding, so the earlier descriptor-layout leak is fixed.
- `VulkanCompute.cpp` now queries subgroup size during device selection, conditionally enables device extensions, and prefers device-local buffers with host-visible staging buffers when discrete-GPU VRAM is available.
- The obsolete `RemoveCollapseShape` dead code is gone from `PrepareSPIRV.cpp`; collapse-shape cleanup now lives only in the active rewrite paths.
- The old stage-specific skills were consolidated into `triton-windows-vulkan` and `triton-windows-opencl`.

---

## 3. Backend Skeleton

### 3.1 Goal

Create the minimum viable backend that Triton can discover and load without
crashing. No compilation, no code generation — just the interface contract.

### 3.2 Backend Discovery

Triton discovers backends by scanning `python/triton/backends/` for Python
packages that export `compiler` and `driver` modules. The discovery is in
`python/triton/backends/__init__.py`:

```python
# Simplified — actual code uses importlib.metadata entry_points
for name in os.listdir(backends_dir):
  try:
    mod = importlib.import_module(f"triton.backends.{name}")
    # Expects mod.compiler.SomeBackend(BaseBackend)
  except (ImportError, ModuleNotFoundError):
    pass
```

For our backend to be found:

1. `third_party/vulkan/backend/__init__.py` must exist (can be empty for now,
  but should eventually export the backend classes)
2. `setup.py` must include `"vulkan"` in the backend list
3. `compiler.py` must define a class extending `BaseBackend`
4. `driver.py` must define a class extending `DriverBase`

### 3.3 The Driver Skeleton

The driver provides GPU target information. Initially, it reports inactive:

```python
class VulkanDriver(DriverBase):
  @staticmethod
  def is_active():
    return False # Not ready to run kernels yet

  def get_current_target(self):
    return GPUTarget("vulkan", 0, 32) # backend, arch, warp_size
```

The `is_active() = False` means Triton won't try to auto-detect Vulkan as the
current device. Users must explicitly request the Vulkan backend.

### 3.4 The Compiler Skeleton

The compiler defines `VulkanOptions` (backend-specific configuration) and
`VulkanBackend` (the pipeline orchestrator):

```python
@dataclass(frozen=True)
class VulkanOptions:
  backend_name: str = "vulkan"
  num_warps: int = 1    # Vulkan doesn't have warps, but the API requires it
  num_stages: int = 1    # No software pipelining in Route B
  num_ctas: int = 1
  cluster_dims: tuple = (1, 1, 1)
  extern_libs: dict = None
  debug: bool = False
```

The `num_warps = 1` is notable. NVIDIA's backend uses this to configure warp
counts for register allocation and occupancy. Since Route B produces serial
loops (no explicit parallelism), this parameter is effectively ignored. A
future Route A implementation would map this to Vulkan workgroup size.

### 3.5 Build System Integration

Three integration points connect the backend to Triton's build:

1. **`setup.py`** — includes `"vulkan"` in the backends list so CMake builds it
2. **`CMakeLists.txt`** — defines the C++ library and pybind11 module
3. **`bin/RegisterTritonDialects.h`** — registers the pass for `triton-opt`

The CMake integration uses `add_triton_plugin()`, the same mechanism as
NVIDIA/AMD backends:

```cmake
add_triton_plugin(TritonVulkan
 ${CMAKE_CURRENT_SOURCE_DIR}/triton_vulkan.cc
 LINK_LIBS
 VulkanTritonToLinalg
 # ... MLIR libraries ...
)
```

This creates a pybind11 shared library that Triton loads at runtime as
`triton._C.libtriton.vulkan`.

---

## 4. TritonToLinalg Conversion

This is the heart of the backend — a C++ MLIR pass that converts Triton IR
to standard MLIR dialects.

### 4.1 Design Principles

**Dialect conversion framework.** We use MLIR's `applyPartialConversion` with
a `ConversionTarget` that marks Triton ops as illegal and standard dialects as
legal. Each Triton op gets a conversion pattern that rewrites it to legal ops.

**Type conversion.** Triton's pointer types (`!tt.ptr<f32>`) have no equivalent
in standard MLIR. We use a `TypeConverter` to map them:

```
!tt.ptr<f32>          → memref<*xf32>   (unranked memref)
tensor<256x!tt.ptr<f32>>    → memref<256xf32>  (ranked memref)
tensor<64x32xf32>        → tensor<64x32xf32> (unchanged)
f32, i32, index         → f32, i32, index  (unchanged)
```

The key insight is that Triton's pointer tensors (`tensor<N x !tt.ptr<T>>`)
represent N pointers into a buffer. After conversion, these become a single
memref view of that buffer. The offset/stride information comes from the
`tt.addptr` pattern, not from the pointer type itself.

**Program info injection.** Triton kernels call `tl.program_id(axis)` and
`tl.num_programs(axis)` to get their position in the launch grid. Later C+
passes also need `local_id`. In NVIDIA, these map to `%ctaid.*`, `%nctaid.*`,
and `%tid.*` registers. In our portable backend, `TritonToLinalgPass`
initially appends explicit function arguments that later map to Vulkan
builtins and push constants:

```
BEFORE: tt.func @kernel(%arg0: !tt.ptr<f32>) {
     %pid = tt.get_program_id x : i32
     ...
    }

AFTER: func.func @kernel(%arg0: memref<*xf32>,
              %num_x: i32, %num_y: i32, %num_z: i32,
              %pid_x: i32, %pid_y: i32, %pid_z: i32,
              %lid_x: i32, %lid_y: i32, %lid_z: i32) {
     // %pid_x replaces tt.get_program_id x during TritonToLinalg
     ...
    }
```

Program-info i32 arguments are appended (see
`TRITON_PROGRAM_INFO_ARG_COUNT` in `TritonToLinalgPass.cpp`):
`num_programs(x,y,z)`, `program_id(x,y,z)`, and `local_id(x,y,z)`.
`GetProgramIDConverter` extracts the `program_id` arg group using the offset
derived from `TRITON_PROGRAM_INFO_ARG_COUNT`; the trailing group is
`local_id`, and `GetNumProgramsConverter` extracts the leading
`num_programs` group.

### 4.2 Core Conversion Patterns

Each converter is an `OpConversionPattern<TritonOp>` subclass that implements
`matchAndRewrite`. Here is what each does and why:

#### 4.2.1 SplatConverter (tt.splat → tensor.empty + linalg.fill)

Triton's `tt.splat %scalar : T -> tensor<NxT>` broadcasts a scalar to a
tensor. In standard MLIR:

```
%init = tensor.empty [N] : tensor<NxT>
%filled = linalg.fill ins(%scalar) outs(%init) → tensor<NxT>
```

**Special case for pointer splats:** When the input is `!tt.ptr<T>`, the type
converter maps the result to `memref<NxT>`. We can't `linalg.fill` a memref
with a memref, so instead we create a `memref.reinterpret_cast` from the base
pointer. This creates a view that `AddPtrConverter` will later adjust with
offset/stride information.

```
BEFORE: %ptrs = tt.splat %base_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
AFTER: %ranked = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %view = memref.reinterpret_cast %ranked to
        offset: [0], sizes: [256], strides: [1]
        : memref<?xf32> to memref<256xf32, strided<[?], offset: ?>>
```

#### 4.2.2 MakeRangeConverter (tt.make_range → linalg.generic + index)

`tt.make_range {start, end}` creates a 1D tensor `[start, start+1, ..., end-1]`.
In standard MLIR, we use a `linalg.generic` with a `linalg.index` op:

```mlir
%init = tensor.empty [end-start] : tensor<Nxi32>
%result = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} outs(%init) {
  %idx = linalg.index 0 : index
  %val = arith.index_cast %idx : index to i32
  %res = arith.addi %val, %start : i32
  linalg.yield %res : i32
} -> tensor<Nxi32>
```

#### 4.2.3 BroadcastConverter (tt.broadcast → linalg.generic)

Broadcasting replicates a tensor along size-1 dimensions. We use
`linalg.generic` with an affine map that maps broadcast dimensions to
constant 0:

```
input shape: [128, 1]  → affine_map<(d0, d1) -> (d0, 0)>
output shape: [128, 64]  → affine_map<(d0, d1) -> (d0, d1)>
```

The body simply yields the input value — the affine map handles the
broadcast semantics.

#### 4.2.4 ExpandDimsConverter (tt.expand_dims → tensor.expand_shape)

Inserts a size-1 dimension at the specified axis. This maps directly to
`tensor.expand_shape` with appropriate reassociation indices.

```
BEFORE: tensor<128xf32> with axis=1
AFTER: tensor<128x1xf32> via tensor.expand_shape [[0, 1]]
```

#### 4.2.5 TransposeConverter (tt.trans → linalg.transpose)

Direct mapping. The `order` attribute specifies the permutation.

#### 4.2.6 MatmulConverter (tt.dot → linalg.matmul)

Triton's `tt.dot %a, %b, %c` computes `C += A @ B`. We decompose this into:

```
%zero_filled = linalg.fill(%zero, %init)     # Zero-initialized output
%product = linalg.matmul ins(%a, %b) outs(%zero) # A @ B
%result = arith.addf %c, %product         # C + (A @ B)
```

If `%c` is provably zero (a `tt.splat` of 0.0), we skip the addition.

**Performance note:** After `linalg-to-loops`, this becomes three nested
`scf.for` loops with no tiling. This is the primary reason Route B matmul
achieves <5% of CUDA performance. Tiling, vectorization, and cooperative
matrix support remain future work.

#### 4.2.7 ReduceConverter (tt.reduce → linalg.reduce)

Triton's `tt.reduce` has an arbitrary combiner region (the body of the
Python `triton.language.reduce`). We clone this region into a `linalg.reduce`:

```python
# Triton Python
result = tl.sum(x, axis=0) # Uses arith.addf as combiner
```

The converter:
1. Determines the identity element from the combiner op type (0 for add,
  -∞ for max, etc.)
2. Creates a `linalg.reduce` with the appropriate dimension
3. Clones the combiner body from `tt.reduce` into the linalg body
4. Maps `tt.reduce.return` to `linalg.yield`

This handles arbitrary reductions — sum, max, min, product, and/or, xor —
because we clone the combiner rather than pattern-matching specific ops.

#### 4.2.8 Supporting Patterns

- **BitcastConverter:** `tt.bitcast → arith.bitcast` (direct mapping)
- **ReshapeConverter:** `tt.reshape → tensor.expand_shape / collapse_shape`
 (tries static reassociation first, falls back to `tensor.reshape`)
- **DenseConstantConverter:** Splat `arith.constant dense<0.0>` →
 `tensor.empty + linalg.fill` (needed because tensor constants can't be
 bufferized directly)
- **Elementwise promotion:** `linalg::populateElementwiseToLinalgConversionPatterns`
 handles `arith.addf %a, %b : tensor<Nxf32>` → `linalg.generic` with
 per-element `arith.addf`.

### 4.3 The Pass Wrapper (TritonToLinalgPass)

The pass wrapper in `TritonToLinalgPass.cpp` ties everything together:

```cpp
void runOnOperation() override {
  // 1. Set up type converter, conversion target, patterns
  // 2. Mark standard dialects as legal, Triton ops as illegal
  // 3. Add the program-info arg block (see TRITON_PROGRAM_INFO_ARG_COUNT)
  //    to each tt.func
  // 4. Apply partial conversion
  // 5. Convert tt.func/tt.return → func.func/func.return
  // 6. Canonicalize
}
```

**Why partial conversion?** We use `applyPartialConversion` rather than
`applyFullConversion` because some ops (like `arith.constant` with non-splat
values) are legal and should pass through unchanged. The `ConversionTarget`
uses `addDynamicallyLegalOp` to make this distinction.

**The tt.func → func.func conversion** is done in a separate walk after the
main conversion, because MLIR's function type conversion infrastructure
handles the signature (types, attributes) but not the op itself. We
manually create `func.func`, clone the body, and replace `tt.return` with
`func.return`.

---

## 5. Pointer Analysis and Memory Ops

This section adds the ability to actually load from and store to memory — the
operations that make kernels do useful work.

### 5.1 The Pointer Problem

Triton uses pointer arithmetic to express memory access patterns:

```python
# Triton Python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x_ptrs = x_ptr + offsets     # tt.addptr
mask = offsets < N         # bounds check
x = tl.load(x_ptrs, mask=mask)  # tt.load with mask
```

This produces TTIR like:

```mlir
%base = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
%range = tt.make_range {start=0, end=256} : tensor<256xi32>
%pid_splat = tt.splat %pid : i32 -> tensor<256xi32>
%offsets = arith.addi %pid_splat, %range : tensor<256xi32>
%ptrs = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
%mask = arith.cmpi slt, %offsets, %N_splat : tensor<256xi32>
%x = tt.load %ptrs, %mask : tensor<256x!tt.ptr<f32>>
```

The challenge: we need to convert this pointer arithmetic into memref
operations with explicit offset/size/stride parameters.

### 5.2 PtrState: Simplified Pointer Analysis

Rather than implementing the full `PtrAnalysis` from triton-shared (which
handles loop-carried pointer state, wraparound, and complex scatter/gather),
we implement a simplified analysis that handles the most common pattern:

```
base_ptr → tt.splat → tt.addptr(splat, range+scalar) → tt.load/tt.store
```

The `PtrState` struct tracks:

```cpp
struct PtrState {
  Value source;           // Base memref
  SmallVector<OpFoldResult> offsets; // Per-dim offsets
  SmallVector<OpFoldResult> sizes;  // Per-dim sizes
  SmallVector<OpFoldResult> strides; // Per-dim strides
  Value scalar;           // Scalar offset (for scalar ptrs)
};
```

The `visitOperand` function walks the def-chain of a value backward,
building up `PtrState` as it goes:

```
tt.addptr ← visits ptr and offset operands recursively
 ├── tt.splat %base_ptr  → sets source = base memref, offsets = [0], strides = [0]
 └── arith.addi
   ├── tt.splat %pid  → scalar = pid value
   └── tt.make_range  → offset = [start], size = [end-start], stride = [1]
```

The `addState` method combines two `PtrState`s element-wise (adding offsets
and strides). After the full walk, `PtrState::createCastOp` produces:

```mlir
%view = memref.reinterpret_cast %base
  to offset: [computed_offset],
    sizes: [BLOCK_SIZE],
    strides: [1]
  : memref<?xf32> to memref<256xf32, strided<[?], offset: ?>>
```

### 5.3 AddPtrConverter

The `AddPtrConverter` is the glue between Triton's pointer arithmetic and
memref views:

```cpp
LogicalResult matchAndRewrite(triton::AddPtrOp op, ...) {
  // 1. Walk the ORIGINAL (pre-conversion) def chain to understand
  //  the pointer structure (offsets, sizes, strides)
  PtrState state;
  visitOperand(op, state, ...);

  // 2. Override source with the type-converted ptr from the adaptor.
  //  After SplatConverter runs, adaptor.getPtr() is the base memref.
  state.source = adaptor.getPtr();

  // 3. Fix up zero strides for size-1 dimensions
  // 4. Create memref.reinterpret_cast
  Value result = state.createCastOp(resultShape, loc, rewriter);
  rewriter.replaceOp(op, result);
}
```

**Key insight:** We walk the *original* (pre-conversion) def chain to
understand the pointer pattern, but use the *converted* base memref
(from the adaptor) for the actual reinterpret_cast. This is because after
`SplatConverter` runs on pointer splats, the adaptor provides the type-
converted memref, while the original ops still have the structural
information we need.

### 5.4 LoadConverter

`tt.load %ptrs, %mask` loads data from memory, optionally with masking.

```cpp
LogicalResult matchAndRewrite(triton::LoadOp op, ...) {
  // ptr is now a memref (from AddPtrConverter)

  // 1. Allocate a destination buffer
  auto alloc = memref.alloc() : memref<256xf32>

  // 2. Fill with zero (or 'other' value for masked loads)
  linalg.fill(%zero, %alloc)

  // 3. Copy from source
  memref.copy(%ptr, %alloc)

  // 4. Return as tensor
  %tensor = bufferization.to_tensor %alloc
}
```

**Current limitation:** The mask is not actually applied — we do a full
copy regardless. This is correct only when the source buffer is large
enough to cover the full load region (which it usually is for the common
`offsets < N` pattern where N ≥ BLOCK_SIZE for all but the last block).
Full mask analysis (using subviews for partial copies) is covered later in this guide.

### 5.5 StoreConverter

`tt.store %ptrs, %val, %mask` stores data to memory.

```cpp
// Unmasked:
bufferization.materialize_in_destination %val, %ptr

// Scalar:
affine.store %val, %ptr[0]
```

The `materialize_in_destination` op copies a tensor into a memref, which
is exactly the semantics of `tt.store` for contiguous stores.

### 5.6 The make_memref Stage

After `make_linalg`, we have IR in Linalg/Tensor/MemRef/Arith dialects.
The `make_memref` stage lowers this to a form suitable for code generation:

```python
def make_memref(mod, metadata, opt):
  pm.add(one_shot_bufferize)   # tensor → memref
  pm.add(convert_linalg_to_loops) # linalg.generic → scf.for
  pm.add(lower_affine)      # affine.for → scf.for
  pm.add(convert_scf_to_cf)   # scf.for → cf.br + cf.cond_br
  pm.add(canonicalize)
  pm.add(cse)
```

After this stage, the IR contains only: `memref.load`, `memref.store`,
`memref.alloc`, `memref.reinterpret_cast`, `memref.copy`, `memref.cast`,
`arith.*`, `cf.br`, `cf.cond_br`, and `func.func/func.return`.

### 5.7 Worked Example: vector_add

Here is the complete transformation of a vector addition kernel through
all stages up to the end of the memory-ops lowering.

**Input (TTIR):**
```mlir
tt.func @vector_add(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>,
           %out_ptr: !tt.ptr<f32>, %N: i32) {
 %pid = tt.get_program_id x : i32
 %c256 = arith.constant 256 : i32
 %start = arith.muli %pid, %c256 : i32
 %range = tt.make_range {start=0, end=256} : tensor<256xi32>
 %start_splat = tt.splat %start : i32 -> tensor<256xi32>
 %offsets = arith.addi %start_splat, %range : tensor<256xi32>
 %mask = arith.cmpi slt, %offsets, %N_splat : tensor<256xi32>

 %x_base = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
 %x_ptrs = tt.addptr %x_base, %offsets
 %x = tt.load %x_ptrs, %mask : tensor<256x!tt.ptr<f32>>

 %y_base = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
 %y_ptrs = tt.addptr %y_base, %offsets
 %y = tt.load %y_ptrs, %mask : tensor<256x!tt.ptr<f32>>

 %result = arith.addf %x, %y : tensor<256xf32>

 %out_base = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
 %out_ptrs = tt.addptr %out_base, %offsets
 tt.store %out_ptrs, %result, %mask
 tt.return
}
```

**After make_linalg (TritonToLinalg pass):**

The kernel becomes a `func.func` with:
- 3 `memref<*xf32>` args (type-converted pointers) + 1 `i32` (N) +
  the program-info arg block (see `TRITON_PROGRAM_INFO_ARG_COUNT`:
  `num_programs×3`, `program_id×3`, `local_id×3`)
- `memref.reinterpret_cast` for each pointer+offset pattern
- `memref.alloc + linalg.fill + memref.copy` for each load
- `arith.addf` on tensors (will become `linalg.generic` via elementwise promotion)
- `bufferization.materialize_in_destination` for the store

**After make_memref (bufferize + lower):**

All tensor ops become memref ops. Linalg generics become `scf.for` loops.
SCF loops become `cf.br`/`cf.cond_br` with block arguments. The result is
a flat control flow graph with explicit loads, stores, and branches.

---

## 6. Core Ops and Testing Scope

The [roadmap](../references/roadmap.md) groups the following work under the unified Vulkan/SPIR-V backend:

```
Core ops + testing
```

Specifically:

| Goal | Description | Status |
|------|-------------|--------|
| Elementwise ops | add, mul, fma, gelu, swiglu | ✅ Complete |
| Reductions | sum, max, softmax (compound) | ✅ Complete |
| Matrix multiply | Small tile matmul (16×16) | ✅ Complete |
| 2D operations | Transpose, broadcast add | ✅ Complete |
| Atomic operations | atomic_rmw (fadd, add, max, etc.) | ✅ Complete |
| End-to-end GPU tests | Serial OpenCL + Vulkan SPIR-V execution with numpy reference | ✅ Complete |
| Performance baseline | Timing vs CUDA backend | ❌ Future work |

### 6.1 What Changed (File Summary)

```
Modified core implementation files:
 third_party/vulkan/lib/Conversion/TritonToLinalg.cpp
   - AtomicRMWConverter, reduction scalar fix, scalar load/store fixes
 third_party/vulkan/lib/Conversion/TritonToLinalgPass.cpp
   - TypeConverter materializations and conversion target updates
 third_party/vulkan/backend/emitter.py
   - Hex float, N-D indexing, 0-d memref, alloca, and reshape-alias support

Test assets:
 third_party/vulkan/test/test_*.ttir             (the TTIR test kernels)
 third_party/vulkan/test/test_kernels.py         (the OpenCL test suite)
 third_party/vulkan/test/test_kernels_vulkan.py  (the Vulkan kernel test suite)

Cleanup:
 third_party/vulkan/lib/Conversion/PrepareSPIRV.cpp
   - standalone finalization logic removed
 third_party/vulkan/backend/compiler.py
   - `make_spv()` extraction and binary extension cleanup
```

---

## 7. AtomicRMWConverter: Design and Implementation

### 7.1 The Problem

Triton's `tt.atomic_rmw` operation performs read-modify-write on GPU memory,
typically used for cross-workgroup accumulation (e.g., gradient updates,
histogram computation). The operation takes a pointer (scalar or tensor),
a value, and an RMW kind (fadd, add, max, xchg, etc.), and returns the
old values that were at those memory locations before the modification.

In the CUDA backend, these map to hardware atomic instructions (`atomicAdd`,
`atomicMax`, etc.). In our Vulkan backend, which currently executes as a
single OpenCL workgroup, we can implement them as sequential load-modify-store
operations — semantically correct because there's no data race.

### 7.2 Three Cases

The converter handles three distinct pointer patterns:

```
Case 1: Scalar atomic (rare)
 tt.atomic_rmw fadd, %scalar_ptr, %scalar_val → %old_scalar

Case 2: Splat pointer tensor (all same location)
 %ptrs = tt.splat %base : !tt.ptr<f32> → tensor<256x!tt.ptr<f32>>
 tt.atomic_rmw fadd, %ptrs, %vals → %old_vals
 → All 256 elements accumulate into base[0] sequentially

Case 3: Offset pointer tensor (different locations)
 %ptrs = tt.addptr %base_splat, %offsets
 tt.atomic_rmw fadd, %ptrs, %vals → %old_vals
 → Each element RMWs its own location: base[offsets[i]]
```

### 7.3 The Conversion Pattern

**Location:** search for `struct AtomicRMWConverter` in
`third_party/vulkan/lib/Conversion/TritonToLinalg.cpp`.

```cpp
struct AtomicRMWConverter
  : public OpConversionPattern<triton::AtomicRMWOp> {
```

Unlike most other converters in this file that use
`patterns.add<FooConverter>(ctx)`, the AtomicRMWConverter is registered with
the type converter:

```cpp
patterns.add<AtomicRMWConverter>(typeConverter, ctx);
```

This is because atomic ops receive type-converted operands through the
`OpAdaptor` — the pointer has already been converted from `!tt.ptr<f32>` to
`memref<*xf32>` (unranked) or `memref<256xf32>` (ranked) by the
TritonTypeConverter. Using the typeConverter ensures the adaptor provides
correctly-typed values.

### 7.4 The RMW Operation Switch

```cpp
static Value applyRMW(OpBuilder &b, Location loc, triton::RMWOp kind,
            Value old, Value val) {
 switch (kind) {
 case triton::RMWOp::FADD: return b.create<arith::AddFOp>(loc, old, val);
 case triton::RMWOp::ADD:  return b.create<arith::AddIOp>(loc, old, val);
 case triton::RMWOp::MAX:  return b.create<arith::MaxSIOp>(loc, old, val);
 case triton::RMWOp::MIN:  return b.create<arith::MinSIOp>(loc, old, val);
 case triton::RMWOp::UMAX: return b.create<arith::MaxUIOp>(loc, old, val);
 case triton::RMWOp::UMIN: return b.create<arith::MinUIOp>(loc, old, val);
 case triton::RMWOp::AND:  return b.create<arith::AndIOp>(loc, old, val);
 case triton::RMWOp::OR:  return b.create<arith::OrIOp>(loc, old, val);
 case triton::RMWOp::XOR:  return b.create<arith::XOrIOp>(loc, old, val);
 case triton::RMWOp::XCHG: return val;
 default:          llvm_unreachable("unsupported RMW operation");
 }
}
```

Key design decisions:

1. **`XCHG` returns `val` directly** — exchange just replaces the old value.
2. **`default` uses `llvm_unreachable`** — fails loudly on unsupported ops
  (e.g., `CAS`/compare-and-swap which requires different IR structure).
3. **Float vs integer separation** — `FADD` uses `AddFOp` (float), `ADD`
  uses `AddIOp` (integer). The caller must ensure type consistency.

### 7.5 Scalar Atomic Path

For scalar atomics, the pointer arrives as an unranked memref (`memref<*xf32>`)
from the TritonTypeConverter's scalar pointer conversion. We cast to ranked
`memref<1xf32>` for load/store:

```cpp
if (isScalar) {
 auto rankedTy = MemRefType::get({1}, elemType);
 auto ranked = rewriter.create<memref::CastOp>(loc, rankedTy, ptrVal);
 auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
 auto old = rewriter.create<memref::LoadOp>(loc, ranked, ValueRange{zero});
 auto newVal = applyRMW(rewriter, loc, rmwKind, old, valVal);
 rewriter.create<memref::StoreOp>(loc, newVal, ranked, ValueRange{zero});
 rewriter.replaceOp(op, old); // returns the OLD value
}
```

### 7.6 Tensor Atomic Path with scf.for (Memref-Based)

For tensor atomics, we need to:
1. Allocate a result memref buffer to hold old values
2. Loop over each element, performing load-modify-store on the target
3. Save each old value into the result buffer
4. Convert the result memref back to a tensor for replacement

**Critical design insight:** Because `AtomicRMWConverter` is registered with
the `typeConverter`, `adaptor.getVal()` returns a **memref** (not a tensor).
This means we must use `memref.load/store` throughout — NOT `tensor.extract/
insert`. An earlier design using tensor ops crashed because `tensor::ExtractOp`
was called on a memref value.

```cpp
// Allocate result buffer to hold old values
auto resultMemTy = MemRefType::get(shape, elemType);
auto resultBuf = rewriter.create<memref::AllocOp>(loc, resultMemTy);

// Sequential RMW loop using memref load/store (no tensor ops needed)
rewriter.create<scf::ForOp>(
  loc, zero, ub, step, ValueRange{},
  [&](OpBuilder &b, Location l, Value iv, ValueRange) {
   auto valI = b.create<memref::LoadOp>(l, valVal, ValueRange{iv});
   Value loadIdx = isSplatPtr ? zero : iv;
   auto old = b.create<memref::LoadOp>(l, ptrMemref, ValueRange{loadIdx});
   auto newVal = applyRMW(b, l, rmwKind, old, valI);
   b.create<memref::StoreOp>(l, newVal, ptrMemref, ValueRange{loadIdx});
   b.create<memref::StoreOp>(l, old, resultBuf, ValueRange{iv});
   b.create<scf::YieldOp>(l, ValueRange{});
  });

// Convert result memref to tensor for replacement
auto resultTensor = rewriter.create<bufferization::ToTensorOp>(
  loc, resultTensorTy, resultBuf, /*restrict=*/true, /*writable=*/false);
rewriter.replaceOp(op, resultTensor.getResult());
```

**Why no iter_args?** Unlike the tensor-based approach, memref stores are
side-effecting — the result buffer is mutated in-place. No need to thread
a tensor through the loop. The `scf.for` has no iter_args, making it simpler.

**Why `bufferization::ToTensorOp`?** The original `tt.atomic_rmw` returns a
tensor (`tensor<256xf32>`), but our result is a memref. `ToTensorOp` bridges
this gap. The `restrict=true` flag tells the optimizer this is the only
reference to the buffer, enabling later optimization.

### 7.7 Splat vs Offset Pointer Detection

```cpp
bool isSplatPtr = isa<UnrankedMemRefType>(ptrVal.getType());
```

This works because:
- `tt.splat %base` → SplatConverter produces `memref<*xf32>` (unranked)
- `tt.addptr %splat, %offsets` → AddPtrConverter produces `memref<256xf32>`
 (ranked, with strides encoding the offsets)

When all pointers go to the same location (splat), we load/store from index 0
regardless of loop iteration. When pointers have offsets, we use the loop
induction variable as the index.

### 7.8 Semantic Note: Sequential vs True Atomics

This implementation is **sequentially correct** but not atomically correct in
the hardware sense. For single-workgroup execution (our current model), this
is fine — there are no concurrent writers. For multi-workgroup execution
we would need to lower to actual SPIR-V atomic operations
(`spirv.AtomicIAdd`, `spirv.AtomicFAddEXT`, etc.).

---

## 8. Reduction Scalar Fix: The 0-d Tensor Problem

### 8.1 The Bug

When reducing a 1D tensor to a scalar:

```mlir
%sum = "tt.reduce"(%x) ({...}) {axis = 0} : (tensor<256xf32>) -> f32
```

Triton's `tt.reduce` returns a bare `f32` scalar. But `linalg.reduce`
operates on tensors and produces a **0-d tensor** (`tensor<f32>`), not a
scalar. The mismatch caused type verification failures.

### 8.2 The Fix (Two Parts)

**Part 1: 0-d tensor initialization** (search for
`if (resultShape.empty())` in `ReduceConverter`)

Previously, scalar reduction tried to pass a bare scalar as the init value
to `linalg.reduce`. But `linalg.reduce` requires tensor operands. Fix:

```cpp
if (resultShape.empty()) {
 // Scalar result → 0-d tensor init (linalg.reduce requires tensor)
 auto initScalar = getInitValue(combiner, elemType, loc, rewriter);
 auto emptyTensor =
   rewriter.create<tensor::EmptyOp>(loc, resultShape, elemType);
 init = rewriter
       .create<linalg::FillOp>(loc, ValueRange{initScalar},
                   ValueRange{emptyTensor})
       .result();
}
```

`resultShape` is empty (`{}`) for a 0-d tensor, so `tensor.empty` creates
a `tensor<f32>` and `linalg.fill` fills it with the init value (0.0 for sum,
-inf for max).

**Part 2: 0-d tensor extraction** (search for `tensor::ExtractOp` in
`ReduceConverter`)

After `linalg.reduce` produces a `tensor<f32>` (0-d), we need to extract the
bare `f32` to match `tt.reduce`'s result type:

```cpp
for (auto result : reduceOp->getResults()) {
 if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
  if (tensorType.getRank() == 0) {
   auto scalar = rewriter.create<tensor::ExtractOp>(
     loc, result, ValueRange{});
   results.push_back(scalar);
   continue;
  }
 }
 results.push_back(result);
}
```

`tensor.extract` with empty indices extracts the single element from a 0-d
tensor. For non-scalar reductions (e.g., 2D→1D), the result is already the
correct tensor type and passes through unchanged.

### 8.3 Why This Matters

Without this fix, every reduction kernel (reduce_sum, reduce_max, softmax)
would fail at MLIR verification. Softmax uses **two** reductions (max, then
sum of exp), so this fix is exercised twice per softmax compilation.

---

## 9. Scalar Load/Store with Unranked MemRef

### 9.1 The Problem

When a Triton kernel stores a scalar result through a scalar pointer:

```mlir
tt.store %out_ptr, %sum : !tt.ptr<f32>
```

The TritonTypeConverter converts `%out_ptr` from `!tt.ptr<f32>` to
`memref<*xf32>` (unranked memref). But `affine.store` requires a ranked
memref with known dimensions.

### 9.2 The Fix

Both `LoadConverter` and `StoreConverter` now handle unranked memrefs
by inserting a `memref.cast` to `memref<1xelemType>`:

```cpp
// In the unranked-memref handling paths of LoadConverter and StoreConverter:
if (isa<UnrankedMemRefType>(ptr.getType())) {
 auto elemType = cast<UnrankedMemRefType>(ptr.getType()).getElementType();
 auto ranked1D = MemRefType::get({1}, elemType);
 memPtr = rewriter.create<memref::CastOp>(loc, ranked1D, ptr);
}
```

This pattern is consistent with how `AtomicRMWConverter` handles scalar
pointers (§2.5), creating a uniform approach across all memory operations.

### 9.3 The Type Conversion Chain

For a scalar pointer argument, the full conversion chain is:

```
!tt.ptr<f32>          (Triton IR)
  → memref<*xf32>      (TritonTypeConverter, unranked)
  → memref<1xf32>      (memref.cast in Load/Store/AtomicRMW converters)
  → affine.load/store [0]  (actual memory access)
```

---

## 10. TypeConverter Materializations

### 10.1 What Are Materializations?

When MLIR's dialect conversion framework converts types, it sometimes needs
to bridge between the old and new type systems. A **target materialization**
tells the framework "how to convert a value from source type to target type",
and a **source materialization** does the reverse.

Without materializations, the conversion framework inserts
`unrealized_conversion_cast` ops as placeholders, which fail verification
if not cleaned up.

### 10.2 The Implementation

**Location:** search for `addTargetMaterialization` in
`third_party/vulkan/lib/Conversion/TritonToLinalgPass.cpp`.

```cpp
// Target materialization: convert source type → target type
addTargetMaterialization([](OpBuilder &builder, Type type,
              ValueRange inputs, Location loc) -> Value {
 if (inputs.size() != 1) return Value();
 auto input = inputs[0];
 if (type == input.getType()) return input;
 // Cast between memref types (unranked ↔ ranked)
 if ((isa<MemRefType>(type) || isa<UnrankedMemRefType>(type)) &&
   (isa<MemRefType>(input.getType()) ||
    isa<UnrankedMemRefType>(input.getType()))) {
  return builder.create<memref::CastOp>(loc, type, input).getResult();
 }
 // tensor → memref (needed when TypeConverter converts tensor<NxT> →
 // memref<NxT> for AtomicRMWOp's val operand)
 if (isa<MemRefType>(type) && isa<RankedTensorType>(input.getType())) {
  return builder.create<bufferization::ToBufferOp>(loc, type, input)
    .getResult();
 }
 return Value();
});
```

Source materialization has the same memref.cast logic, plus the reverse
bridge for `memref → tensor`:

```cpp
// memref → tensor (reverse bridge)
if (isa<RankedTensorType>(type) && isa<MemRefType>(input.getType())) {
 return builder.create<bufferization::ToTensorOp>(
       loc, type, input, /*restrict=*/true, /*writable=*/true)
   .getResult();
}
```

### 10.3 Why Both Directions?

- **Target materialization:** When `AtomicRMWConverter` receives a splat
 pointer (unranked memref) but needs a ranked memref for `memref.load`.
 Also when a tensor value needs to be viewed as memref for atomic ops
 (`bufferization::ToBufferOp`).
- **Source materialization:** When an unconverted op still expects the
 original Triton type but receives a converted memref type. Also when
 a memref result needs to be viewed as tensor (`bufferization::ToTensorOp`).

The memref.cast materializations are no-ops at runtime. The
`bufferization::ToBufferOp/ToTensorOp` materializations create actual
buffer↔tensor bridges that are later resolved by the bufferization passes.

---

## 11. SPIR-V Conversion

### 11.1 The Problem

After `make_memref`, the IR uses `memref`, `arith`, `cf`, and `func` dialects.
MLIR provides `convert-{memref,arith,cf,func}-to-spirv` passes. In theory,
running them in sequence should produce valid SPIR-V. In practice, there are
10 traps that require a custom preparation pass.

### 11.2 The Traps

#### Trap #1: memref.reinterpret_cast Has No SPIR-V Lowering

**The problem:** MLIR's `MemRefToSPIRV` conversion has no pattern for
`memref::ReinterpretCastOp`. It was never implemented because SPIR-V doesn't
have an equivalent concept — SPIR-V uses typed pointer arithmetic via
`spirv.AccessChain`.

**Why we have it:** Our `AddPtrConverter` creates `memref.reinterpret_cast`
to represent pointer views with offset/stride. This is idiomatic MLIR for
expressing pointer arithmetic, but it's a dead end for SPIR-V.

**The fix:** The `ExpandReinterpretCast` pattern in `PrepareSPIRV.cpp`
eliminates reinterpret_casts by inlining their offset arithmetic into each
load/store user:

```
BEFORE: %view = memref.reinterpret_cast %base offset:[off] sizes:[N] strides:[1]
    %v = memref.load %view[%i]

AFTER: %idx = arith.addi %i, %off
    %v = memref.load %base[%idx]
```

This redirects all memory accesses to the original base memref with adjusted
indices. The reinterpret_cast is then dead and can be erased.

#### Trap #2: memref.copy Has No SPIR-V Lowering

**The problem:** Same issue — `MemRefToSPIRV` has no pattern for
`memref::CopyOp`.

**Why we have it:** `LoadConverter` creates `memref.copy` to transfer data
from the source buffer to a local buffer.

**The fix:** `ExpandMemRefCopy` expands the copy to an explicit loop:

```mlir
scf.for %i = 0 to N step 1 {
  %val = memref.load %src[%i]
  memref.store %val, %dst[%i]
}
```

**Important subtlety:** This creates new `scf.for` ops, but `make_memref`
already lowered all SCF ops to control flow. We need a SECOND
`convert-scf-to-cf` pass after `prepare_spirv` to handle these new loops.

#### Trap #3: Storage Class Mapping for Allocas (THE CRITICAL TRAP)

This is the most insidious trap because it fails silently.

**The problem:** MLIR's `map-memref-spirv-storage-class` pass maps ALL
address-space-0 memrefs to `#spirv.storage_class<StorageBuffer>`. This
includes function arguments (correct — they ARE storage buffers) AND local
`memref.alloca` scratch buffers (WRONG — they should be Function-local
variables).

The subsequent `convert-memref-to-spirv` pass has this check:

```cpp
// In AllocaOpPattern::matchAndRewrite:
if (storageClass != spirv::StorageClass::Function)
  return false; // Silently skip!
```

So allocas with `StorageBuffer` class are **silently not converted**. They
remain as `memref.alloca` ops in the IR, blocking `convert-func-to-spirv`
from creating `spirv.module`.

**The fix:** `FixAllocaStorageClassPass` runs AFTER `map_storage_class` and
BEFORE `convert_memref_to_spirv`. It changes alloca storage class from
`StorageBuffer` to `Function`:

```cpp
moduleOp.walk([&](memref::AllocaOp allocaOp) {
  auto funcAttr = spirv::StorageClassAttr::get(
    ctx, spirv::StorageClass::Function);
  auto newType = MemRefType::get(
    oldType.getShape(), oldType.getElementType(),
    oldType.getLayout(), funcAttr);
  // Replace alloca with one that has Function class
});
```

#### Trap #4: convert-func-to-spirv Doesn't Create spirv.module

**The problem:** `convert-func-to-spirv` creates `spirv.func` from
`func.func`, but only creates the `spirv.module` wrapper if the function
body is FULLY converted. If ANY unconverted ops remain (memref ops,
`unrealized_conversion_cast`), it skips the wrapper.

**The underlying fix:** The backend must perform an explicit Vulkan/SPIR-V
finalization step once conversion is complete. Earlier revisions experimented
with both a dedicated C++ finalization pass and Python-side wrapping logic; the
important invariant is that no non-SPIR-V residue can remain when the entry
function is packaged for serialization and dispatch.

#### Trap #5: mlir-translate Needs --no-implicit-module

MLIR's parser wraps everything in an implicit `builtin.module`. The SPIR-V
serializer expects `spirv.module` as the top-level op. Solution:
use `--no-implicit-module` or call `spirv::serialize()` C++ API directly.

#### Trap #6: spirv.target_env Must Be Attached First

All `convert-*-to-spirv` passes check the `spirv.target_env` attribute on
the module to determine available capabilities. Without it, they fall back
to minimal capabilities. `PrepareSPIRV` attaches it as the very first step:

```cpp
auto triple = spirv::VerCapExtAttr::get(
  spirv::Version::V_1_0,
  {spirv::Capability::Shader},
  {spirv::Extension::SPV_KHR_storage_buffer_storage_class},
  ctx);
moduleOp->setAttr(spirv::getTargetEnvAttrName(),
         spirv::TargetEnvAttr::get(triple, limits));
```

#### Trap #7: SPIRVUpdateVCEPass Requires spirv.module Context

The `spirv-update-vce` pass is designed to run on `spirv.module`, not
`builtin.module`. It cannot be run before the module is created.

#### Trap #8: UnrankedMemRefType Not Supported

Our type converter maps `!tt.ptr<T>` to `UnrankedMemRefType`. SPIR-V
doesn't support unranked memrefs. `PrepareSPIRV` converts function
signatures from `memref<*xf32>` to `memref<?xf32>` before conversion.

#### Trap #9: Pointer Splat Must Not Create linalg.fill

For pointer-typed splats, the "value" is a memref, not a scalar.
`linalg.fill` can't broadcast a memref. `SplatConverter` detects pointer
types and handles them specially (see §4.2.1).

#### Trap #10: MLIR API Variations

The LLVM commit used by the current Triton version (check
`python/triton/__init__.py`) has specific API signatures that differ from
MLIR documentation. Key differences:

| API | Correct for This Commit |
|-----|-------------------------|
| `ReduceOp` creation | `ReduceOp::create(builder, loc, ...)` static method |
| `ConstantOp` value | Requires `cast<TypedAttr>(...)` |
| `FunctionInterfaces.h` | `mlir/Interfaces/FunctionInterfaces.h` |
| `ResourceLimitsAttr::get` | 4th arg is `ArrayAttr` |
| SCF→CF pass | `createSCFToControlFlowPass()` |

### 11.3 The SPIR-V Pass Pipeline (Critical Ordering)

The passes MUST be run in this exact order. Reordering causes silent failures.

```
Step 1: prepare_spirv
 ├── Attach spirv.target_env          (TRAP #6)
 ├── Convert unranked memref → ranked memref  (TRAP #8)
 ├── Remove identity memref.cast ops
 ├── Expand memref.reinterpret_cast      (TRAP #1)
 ├── Expand memref.copy → scf.for loop     (TRAP #2)
 ├── Remove memref.dealloc
 └── Convert memref.alloc → memref.alloca

Step 2: convert_scf_to_cf
 └── Lower new scf.for from copy expansion   (TRAP #2 follow-up)

Step 3: canonicalize
 └── Clean up dead ops

Step 4: map_storage_class
 └── Map addr space 0 → StorageBuffer     (for function args)

Step 5: fix_alloca_storage_class
 └── Change alloca: StorageBuffer → Function  (TRAP #3)

Step 6: convert_memref_to_spirv
 └── memref.load → spirv.Load, memref.alloca → spirv.Variable

Step 7: convert_arith_to_spirv
 └── arith.addi → spirv.IAdd, etc.

Step 8: convert_cf_to_spirv
 └── cf.br → spirv.Branch, cf.cond_br → spirv.BranchConditional

Step 9: convert_func_to_spirv
 └── func.func → spirv.func          (TRAP #4: may not create module)

Step 10: canonicalize
 └── Clean up
```

In `compiler.py`, steps 1-3 run in one pass manager, steps 4-5 in another,
and steps 6-10 in a third. Separate pass managers are needed because some
passes modify the type system in ways that require re-initialization.

### 11.4 The make_spv Stage (Serialization)

After `make_spirv`, the IR contains `spirv.func`, `spirv.Load`, `spirv.Store`,
`spirv.IAdd`, etc. inside a `builtin.module`. Serialization requires:

1. Extract the `spirv.func` (and any `spirv.GlobalVariable`) from the IR
2. Wrap in `spirv.module Logical GLSL450 requires #spirv.vce<...> { ... }`
3. Add `spirv.EntryPoint "GLCompute" @kernel_name`
4. Serialize via `mlir-translate --no-implicit-module --serialize-spirv`

The result is a valid SPIR-V binary (magic number `0x07230203`, version 1.0)
that can be loaded by any Vulkan implementation.

### 11.5 Worked Example: vector_add SPIR-V

Continuing from §5.7, after SPIR-V conversion the vector_add kernel becomes:

```mlir
spirv.module Logical GLSL450
 requires #spirv.vce<v1.0, [Shader],
           [SPV_KHR_storage_buffer_storage_class]> {

 spirv.GlobalVariable @__buf_0 : !spirv.ptr<
  !spirv.struct<(!spirv.array<256 x f32>)>, Function>

 spirv.func @vector_add_kernel(
   %x: !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32>)>, StorageBuffer>,
   %y: !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32>)>, StorageBuffer>,
   %out: !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32>)>, StorageBuffer>,
   %N: i32, ...) "None" {

  // Local buffer for masked load
  %buf_addr = spirv.mlir.addressof @__buf_0

  // Loop: copy x[pid*256+i] to local buffer
  spirv.Branch ^bb1(...)
  ^bb1(%i: i32):
   %ptr = spirv.AccessChain %x[%0, %idx]  // x[offset + i]
   %val = spirv.Load "StorageBuffer" %ptr  // load from global
   %local = spirv.AccessChain %buf_addr[%0, %i]
   spirv.Store "Function" %local, %val    // store to local

   // ... (similar for y, then add, then store to out)

  spirv.Return
 }

 spirv.EntryPoint "GLCompute" @vector_add_kernel
}
```

Serialized: ~1772 bytes of SPIR-V binary.

---

## 12. PrepareSPIRV Cleanup

### 12.1 FinalizeSPIRV Removal

Earlier revisions included a standalone `FinalizeSPIRVPass`
that attempted to build `spirv.module` + `spirv.func` in C++, transplanting
function bodies and fixing up types. This approach had fundamental issues:

1. **Block argument type mismatches** — moved blocks retained their original
  memref types, but the spirv.func expected spirv.ptr types
2. **Alloca→GlobalVariable conversion** — complex and fragile, requiring
  matching unrealized_conversion_cast ops
3. **Multiple traversals** — needed separate passes over arg casts, alloca
  casts, and return ops

That intermediate fix was to **remove the standalone C++ pass** and handle
finalization in Python (`compiler.py make_spv()`) using text manipulation:

```python
# Extract spirv.func with brace matching
func_start = ir_text.find("spirv.func")
depth = 0
for i in range(func_start, len(ir_text)):
  if ir_text[i] == '{': depth += 1
  elif ir_text[i] == '}':
    depth -= 1
    if depth == 0:
      func_end = i + 1
      break

# Wrap in spirv.module
wrapped = f'spirv.module Logical GLSL450 ... {{\n {func_text}\n}}\n'
```

This is simpler and more reliable because:
- The IR is already fully converted to SPIR-V ops at this point
- Text extraction avoids all the type-matching complexity
- `mlir-translate --serialize-spirv` handles the actual serialization

### 12.2 compiler.py Improvements

The `make_spv()` method was also improved:

1. **Brace-matching extraction** replaces fragile regex for spirv.func body
2. **`binary_ext = "spv"`** (was "cl") — the default output format is now
  SPIR-V binary, matching the backend's name and purpose
3. **`make_opencl` docstring** clarified as an alternative debug/execution path

### 12.3 Current Status

The temporary `createFinalizeSPIRVPass()` compatibility stub described in the
earlier notes is gone. `PrepareSPIRV.cpp` is now the large shared home for the
live SPIR-V bridge/finalization passes: `PrepareSPIRVPass`,
`ConvertReductionToParallel`, `ConvertMatmulToCooperative`,
`FixAllocaStorageClassPass`, and `VulkanizePass`.

The old `RemoveCollapseShape` dead code is gone as well. Collapse-shape cleanup
now happens only in the active rewrite paths that still participate in lowering.

---

## 13. Emitter Improvements

For the dedicated emitter-focused walkthrough, see `opencl-emitter-guide.md`.

The OpenCL C emitter (`third_party/vulkan/backend/emitter.py`) received
significant improvements to handle the IR patterns produced by the current backend kernels.

### 13.1 IEEE 754 Hex Float Decoding

**Problem:** MLIR sometimes emits float constants as hex-encoded IEEE 754
bit patterns:

```mlir
%cst = arith.constant 0xFF800000 : f32  // -infinity
%cst = arith.constant 0x7F800000 : f32  // +infinity
```

The emitter now detects these and converts to OpenCL C constants:

```python
if ctype == "float" and re.match(r"^0x[0-9A-Fa-f]+$", val):
  bits = int(val, 16) & 0xFFFFFFFF
  fval = struct.unpack('f', struct.pack('I', bits))[0]
  if math.isinf(fval) and fval < 0:
    val = "-INFINITY"
  elif math.isinf(fval) and fval > 0:
    val = "INFINITY"
  elif math.isnan(fval):
    val = "NAN"
  else:
    val = f"{fval:.8e}f"
```

This is critical for `reduce_max` where `getInitValue` emits
`arith.constant 0xFF800000` (-infinity) as the max reduction identity.

### 13.2 N-Dimensional MemRef Linearization

**Problem:** Earlier emitter revisions only handled 1D memref indexing (`buf[i]`).
The current kernels produce 2D and higher operations (matmul, transpose) that
lower to multi-dimensional memref accesses:

```mlir
memref.load %buf[%i, %j] : memref<16x16xf32>
```

**Solution:** The new `_linearize_nd_index` static method computes flat
indices from N-dimensional subscripts:

```python
@staticmethod
def _linearize_nd_index(line, idx_parts, map_val_fn):
  all_dims = re.findall(r'memref<([\dx]+)x\w+', line)
  if all_dims:
    dim_strs = [d for d in all_dims[0].split('x') if d.isdigit()]
    if len(dim_strs) == len(idx_parts) and len(dim_strs) >= 2:
      dims = [int(d) for d in dim_strs]
      terms = []
      for k, idx in enumerate(idx_parts):
        stride = 1
        for d in dims[k + 1:]:
          stride *= d
        mapped = map_val_fn(idx)
        if stride == 1:
          terms.append(mapped)
        else:
          terms.append(f"{mapped} * {stride}")
      return " + ".join(terms)
  return None
```

For `memref<16x16xf32>`, indices `[i, j]` become `i * 16 + j`.
For `memref<2x16x16xf32>`, indices `[b, i, j]` become `b * 256 + i * 16 + j`.

This replaces the earlier hardcoded 2D-only pattern and handles arbitrary
dimensionality.

### 13.3 0-d MemRef Load/Store

Scalar reduction results produce 0-d memref accesses:

```mlir
memref.store %val, %buf[] : memref<f32>  // empty index list
memref.load %buf[] : memref<f32>
```

The emitter now detects empty indices and emits `buf[0]`:

```python
if indices.strip():
  # ... normal indexed access
else:
  self._line(f"{self._map_val(buf)}[0] = {self._map_val(val)};")
```

### 13.4 memref.alloca Support

Current kernels produce `memref.alloca` (stack allocation) in addition to
`memref.alloc`. The emitter regex now matches both:

```python
m = re.match(
  r"(%[\w]+)\s*=\s*memref\.alloca?\(\)(?:\s*\{[^}]*\})?\s*:\s*memref<([^>]+)>",
  line)
```

The `alloca?` pattern matches both `alloc` and `alloca`. The optional
`\{[^}]*\}` handles attribute groups that MLIR may attach (e.g., alignment).

Additionally, the alloc handler now:
- Strips SPIR-V storage class annotations from type strings
- Computes total size for N-d memrefs (product of all dimensions)
- Handles 0-d memrefs (scalar, size=1)

### 13.5 Reshape Aliases

```python
def _emit_reshape_alias(self, line: str):
  """expand_shape/collapse_shape are view aliases — same 1D buffer."""
  m = re.match(
    r"(%[\w]+)\s*=\s*memref\.(expand|collapse)_shape\s+(%[\w]+)", line)
  if m:
    self.ssa_map[dst] = self._map_val(src) # alias, no copy
```

`memref.expand_shape` and `memref.collapse_shape` are view operations —
they reinterpret the same underlying buffer with different dimensions.
In OpenCL C, where all buffers are flat `__global float*`, these are
no-ops. The emitter creates an alias in its SSA map.

### 13.6 Multi-Dimensional memref.copy

Earlier `_emit_copy` revisions extracted only the first dimension from the type.
For `memref<16x16xf32>`, it would copy only 16 elements instead of 256.
The fix computes total size:

```python
dims = re.findall(r'memref<([\dx]+)x\w+', line)
if dims:
  parts = dims[0].split('x')
  total = 1
  for p in parts:
    if p.isdigit():
      total *= int(p)
  size = str(total)
```

---

## 14. Build Integration

### 14.1 CMakeLists.txt

The build defines two targets:

**`VulkanTritonToLinalg`** — Static library containing all C++ passes:
```cmake
add_library(VulkanTritonToLinalg STATIC
 lib/Conversion/TritonToLinalg.cpp
 lib/Conversion/TritonToLinalgPass.cpp
 lib/Conversion/PrepareSPIRV.cpp
)
```

Linked against: `TritonIR`, `MLIRIR`, all standard MLIR dialect libraries,
and SPIR-V-specific libraries (`MLIRSPIRVDialect`, `MLIRSPIRVConversion`,
`MLIRArithToSPIRV`, `MLIRMemRefToSPIRV`, etc.).

**`TritonVulkan`** — pybind11 shared library exposing passes to Python:
```cmake
add_triton_plugin(TritonVulkan
 triton_vulkan.cc
 LINK_LIBS VulkanTritonToLinalg ...
)
```

This creates `triton._C.libtriton.vulkan` with submodules:
- `vulkan.passes.linalg.triton_to_linalg(pm)` — TritonToLinalg pass
- `vulkan.passes.memref.{one_shot_bufferize, convert_linalg_to_loops, ...}(pm)`
- `vulkan.passes.spirv.{prepare_spirv, map_storage_class, ...}(pm)`
- `vulkan.serialize_spirv(module)` — C++ SPIR-V serialization
- `vulkan.load_dialects(ctx)` — register all required dialects

### 14.2 pybind11 Bindings (triton_vulkan.cc)

The bindings expose three pass groups and a serialization function:

```cpp
void init_triton_vulkan(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_vulkan_passes_ttir_to_linalg(passes.def_submodule("linalg"));
  init_triton_vulkan_passes_linalg_to_memref(passes.def_submodule("memref"));
  init_triton_vulkan_passes_spirv(passes.def_submodule("spirv"));

  m.def("serialize_spirv", &serialize_spirv_module);
  m.def("load_dialects", [](MLIRContext &context) { ... });
}
```

The SPIR-V serialization is implemented in C++ using MLIR's
`spirv::serialize()` API, which produces a `SmallVector<uint32_t>`. This is
returned to Python as `bytes`.

### 14.3 triton-opt Registration

For testing with `triton-opt` (the Triton-specific `mlir-opt`), the pass is
registered in `bin/RegisterTritonDialects.h`:

```cpp
#include "vulkan/include/Conversion/TritonToLinalg.h"

// In registerTritonDialectsAndPasses():
mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
  return mlir::triton::vulkan::createTritonToLinalgPass();
});
```

This allows running:
```bash
triton-opt --triton-to-linalg test_vector_add.ttir
```

---

## 15. Testing and Diagnostics

### 15.1 MLIR Lit Tests

Five test files verify individual conversion stages:

| File | Tests | Approach |
|------|-------|----------|
| `test_triton_to_linalg.mlir` | TritonToLinalg pass | `triton-opt --triton-to-linalg` |
| `test_vector_add.ttir` | Full pipeline through load/store | End-to-end TTIR |
| `test_arith_to_spirv.mlir` | Arithmetic → SPIR-V | `mlir-opt` pipeline |
| `test_math_to_spirv.mlir` | Math → SPIR-V | `mlir-opt` pipeline |
| `test_scf_to_spirv.mlir` | SCF → SPIR-V | `mlir-opt` pipeline |

The `lit.cfg.py` configures the test runner with the path to `triton-opt`.
Run with:
```bash
cd <build_dir> && lit -v third_party/vulkan/test/
```

### 15.2 vulkan-opt.py (Manual Pipeline Testing)

A Python wrapper around `mlir-opt` and `mlir-translate` with named pipelines:

```bash
# Convert arith to SPIR-V
python vulkan-opt.py test.mlir --pipeline arith

# Full conversion + serialization
python vulkan-opt.py test.mlir --pipeline full -o test.spv

# Roundtrip verification
python vulkan-opt.py test.mlir --pipeline arith --roundtrip
```

Available pipelines: `arith`, `linalg`, `gpu`, `memref`, `full`, `math`.

### 15.3 diagnose-spirv.py (Trap Detection)

A diagnostic tool that checks IR for known SPIR-V conversion issues:

```bash
# Check a single MLIR file
python diagnose-spirv.py lowered.mlir convert

# Run the full pipeline and diagnose each stage
python diagnose-spirv.py --pipeline vector_add.ttir
```

The `--pipeline` mode instantiates the `VulkanBackend`, runs each stage, and
reports per-stage diagnostics. It checks for all 10 traps documented in the
unified `triton-windows-vulkan` skill.

### 15.4 Manual End-to-End Verification

To verify a kernel works end-to-end:

```python
import triton
from triton._C.libtriton import ir, passes, vulkan
from triton.backends.vulkan.compiler import VulkanBackend
from triton.backends.compiler import GPUTarget

# Load TTIR
ctx = ir.context()
ir.load_dialects(ctx)
vulkan.load_dialects(ctx)
mod = ir.parse_mlir_module("test_vector_add.ttir", ctx)
mod.context = ctx

# Run pipeline
backend = VulkanBackend(GPUTarget('vulkan', 0, 32))
opt = backend.parse_options({})
metadata = {}

mod = backend.make_ttir(mod, metadata, opt)
mod = backend.make_linalg(mod, metadata, opt)
mod = backend.make_memref(mod, metadata, opt)

# Check intermediate IR
print(mod.str_nodebug())

mod = backend.make_spirv(mod, metadata, opt)
binary = backend.make_spv(mod, metadata, opt)

import struct
magic = struct.unpack('<I', binary[:4])[0]
assert magic == 0x07230203, "Invalid SPIR-V magic"
print(f"SPIR-V binary: {len(binary)} bytes")
```

---

## 16. Test Suite Architecture

### 16.1 Test Philosophy

The test suite takes a **compilation-first** approach:

1. Write hand-crafted TTIR (Triton IR) for each kernel pattern
2. Compile through the full pipeline: TTIR → Linalg → MemRef → OpenCL C or SPIR-V
3. Execute on GPU via pyopencl or the Vulkan runtime
4. Compare against numpy reference implementations

This tests the **entire backend** end-to-end, not just individual passes.
If any pass produces incorrect IR, the GPU result will diverge from numpy.
Today that coverage is split across `test_kernels.py` (the serial OpenCL
suite) and `test_kernels_vulkan.py` (the Vulkan SPIR-V dispatch suite).

### 16.2 Pipeline Under Test

```
.ttir file → parse → make_ttir → make_linalg → make_memref → make_opencl
                                  │
           ┌──────────────────────────────────────────────┘
           ▼
        OpenCL C source
           │ cl.Program.build()
           ▼
        GPU execution → numpy comparison
```

The `compile_ttir()` function in `test_kernels.py` drives this serial OpenCL reference path:

```python
def compile_ttir(ttir_path):
  c = ir.context()
  ir.load_dialects(c)
  vulkan.load_dialects(c)
  m = ir.parse_mlir_module(ttir_path, c)
  m.context = c
  backend = VulkanBackend(GPUTarget("vulkan", 0, 32))
  options = backend.parse_options({})
  metadata = {}
  m = backend.make_ttir(m, metadata, options)
  m = backend.make_linalg(m, metadata, options)
  m = backend.make_memref(m, metadata, options)
  return backend.make_opencl(m, metadata, options), metadata
```

### 16.3 Test Harness Design

Each test function follows this pattern:

```python
def test_foo():
  src, md = compile_ttir("test_foo.ttir")
  # Create input data with numpy
  x = np.random.randn(N).astype(np.float32)
  # Upload to GPU
  xb = cl.Buffer(ctx, RO, hostbuf=x)
  ob = cl.Buffer(ctx, WO, N * 4)
  # Run kernel (append the program-info arg block)
  program_info = [np.int32(0)] * PROGRAM_INFO_ARG_COUNT
  args = [xb, ob, np.int32(N)] + program_info
  run_kernel(src, md, args)
  # Read result and compare
  o = read_buf(ob, N)
  return np.max(np.abs(o - expected)) # max absolute error
```

**The appended program-info args** (see `TRITON_PROGRAM_INFO_ARG_COUNT` in
`TritonToLinalgPass.cpp`): these are the program info arguments that the Vulkan
backend adds to every kernel:
- `num_programs_x`, `num_programs_y`, `num_programs_z` — grid dimensions
- `program_id_x`, `program_id_y`, `program_id_z` — current program index
- `local_id_x`, `local_id_y`, `local_id_z` — thread index within the workgroup

For single-block serial tests, `num_programs` is typically 1 and both
`program_id` and `local_id` are 0.

### 16.4 Error Tolerances

| Category | Tolerance | Rationale |
|----------|-----------|-----------|
| Exact ops (add, mul, copy) | 1e-6 | IEEE 754 exact for simple ops |
| Transcendentals (exp, sigmoid) | 1e-5 | GPU exp() may use fast-math |
| Reductions (sum, atomic_add) | 1e-3 | Accumulation order differs |
| Softmax | 1e-5 | Exp/div partially cancel errors |
| Matrix multiply | 1e-4 | 16 multiply-accumulate steps |

### 16.5 TTIR Test File Convention

All test files follow this naming pattern:

```
test_{kernel_name}.ttir
```

Each file contains a single `tt.func public @{kernel_name}_kernel(...)` with
clearly documented parameters and operations. Tests are self-contained —
no dependencies between test files.

---

## 17. Kernel-by-Kernel Analysis

### 17.1 Elementwise Operations

#### test_vector_add.ttir (existing earlier in the bring-up)
```
Pattern: out[i] = x[i] + y[i]
Ops: tt.load (masked), arith.addf, tt.store (masked)
Coverage: Masked load/store path, basic elementwise
```

#### test_elementwise_mul.ttir
```
Pattern: out[i] = x[i] * y[i]
Ops: tt.load (masked), arith.mulf, tt.store (masked)
Coverage: Masked load/store, multiplication
```

#### test_fma.ttir
```
Pattern: out[i] = a[i] * b[i] + c[i]
Ops: tt.load (×3), arith.mulf, arith.addf, tt.store
Coverage: Multiple input buffers, compound arithmetic
```

### 17.2 Activation Functions

#### test_gelu.ttir
```
Pattern: GELU(x) ≈ x · sigmoid(1.702 · x)
Ops: dense constant, arith.mulf, arith.negf, math.exp,
   arith.addf, arith.divf
Coverage: Dense constant lowering, math dialect ops,
     transcendental functions, masked load/store
```

This uses the sigmoid approximation of GELU, which avoids `erf()` (not in
OpenCL C standard). The computation chain:

```
scaled = x * 1.702
neg = -scaled
exp_neg = exp(neg)
denom = 1.0 + exp_neg
sigmoid = 1.0 / denom
result = x * sigmoid
```

#### test_swiglu.ttir
```
Pattern: SwiGLU(x, gate) = x · sigmoid(x) · gate
Ops: arith.negf, math.exp, arith.mulf (×2), arith.divf
Coverage: Two-input activation, SiLU subexpression
```

### 17.3 Reductions

#### test_reduce_sum.ttir
```
Pattern: out = sum(x[0..256])
Ops: tt.load (masked), tt.reduce(addf), tt.store (scalar)
Coverage: Scalar reduction, masked load, scalar pointer store
New ops: 0-d tensor handling (§8), scalar store via unranked memref (§9)
```

This kernel exercises the full reduction scalar fix chain:
1. `tt.reduce` produces `f32` scalar
2. Internally, `linalg.reduce` uses 0-d `tensor<f32>` init
3. `tensor.extract` unwraps 0-d tensor to scalar
4. `tt.store %out_ptr, %sum` stores through unranked memref

#### test_reduce_max.ttir
```
Pattern: out = max(x[0..256])
Ops: tt.load, tt.reduce(maximumf), tt.store (scalar)
Coverage: Different reduction combiner, -inf init value
New ops: IEEE 754 hex float in init value (0xFF800000 → -INFINITY)
```

#### test_softmax.ttir
```
Pattern: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
Ops: tt.reduce (×2), tt.splat (scalar→vector), arith.subf,
   math.exp, arith.divf
Coverage: Compound kernel with two reductions, scalar broadcast,
     transcendental + arithmetic chain
```

Softmax is the most complex single kernel in the test suite. It exercises:
- Two separate reductions (max, then sum)
- Scalar-to-vector broadcast via `tt.splat`
- The full reduce → extract → splat → elementwise pipeline

### 17.4 Matrix Operations

#### test_matmul_simple.ttir
```
Pattern: C[16×16] = A[16×16] @ B[16×16]
Ops: tt.load (flat 256), tt.reshape (256→16×16), tt.dot,
   tt.reshape (16×16→256), tt.store (flat 256)
Coverage: tt.dot → linalg.matmul, reshape for 2D computation,
     dense constant (zero accumulator)
```

The matmul test uses a clever flat-load pattern:
1. Load 256 elements as 1D vector (avoids 2D pointer computation)
2. `tt.reshape` to 16×16 matrix
3. `tt.dot` for matrix multiply
4. `tt.reshape` back to 1D
5. Store as flat 256

This avoids multi-dimensional pointer patterns that aren't yet fully
supported, while still exercising the matmul conversion.

#### test_transpose.ttir
```
Pattern: out[j,i] = x[i,j] (16×16 transpose)
Ops: tt.load (flat), tt.reshape (1D→2D), tt.trans,
   tt.reshape (2D→1D), tt.store (flat)
Coverage: tt.trans with order=[1,0], reshape round-trip
```

### 17.5 Multi-Load Pattern

#### test_broadcast_add.ttir
```
Pattern: out[i] = x[i] + bias[i]
Ops: tt.load (×2, different ptrs), arith.addf, tt.store
Coverage: Dual-pointer load pattern, elementwise addition
     with separate input buffers
```

Note: Despite the name "broadcast_add", this tests 1D elementwise addition
with two separate pointer sources. A true 2D broadcast
(`x[M,N] + bias[N]` with `expand_dims`) requires multi-dimensional pointer
support in TritonToLinalg which is future work.

### 17.6 Atomic Operations

#### test_atomic_add.ttir
```
Pattern: out[i] += x[i] for i in 0..256 (per-element RMW)
Ops: tt.load, tt.splat (ptr), tt.atomic_rmw(fadd)
Coverage: AtomicRMWConverter with tensor operands, memref-based
     sequential loop, bufferization::ToTensorOp result bridge
```

The TTIR uses `tt.splat %out_ptr` to create a pointer tensor, but the
TritonTypeConverter converts the tensor-of-pointers to a ranked
`memref<256xf32>`. The splat information is lost at the type level — the
converter sees a regular ranked memref and performs per-element RMW:
`out[i] = out[i] + x[i]` for each i. This is correct for the test (the
Python harness validates `out == out_init + x`).

**Note:** True accumulate-into-one-location semantics would require
scalar pointer atomics (not tensor), or a pre-processing pass that detects
the splat pattern before type conversion.

---

## 18. Path C+: Incremental Performance Improvements

After achieving correctness with the base pipeline (§1-§17), the backend was
enhanced with five incremental GPU compute features. Each step added one
Vulkan/SPIR-V capability while preserving all existing tests.

### 18.1 Architecture: The Placeholder Pattern

The key innovation across C+1 through C+5 is the **placeholder pattern**:
insert `func.call @__vulkan_*()` calls at the MemRef level, let them survive
unchanged through all MLIR conversion passes (`func→spirv` converts them to
`spirv.FunctionCall`), then replace them in VulkanizePass with actual SPIR-V
ops. This solves a fundamental MLIR infrastructure gap — there is no clean way
to express GPU-specific operations (barriers, subgroup ops, cooperative matrix)
at the MemRef level that survives through dialect conversion.

### 18.2 C+1: WorkgroupId for program_id

**Problem:** Multi-block dispatch required N serial dispatches, each setting
`program_id` as a push constant.

**Solution:** Replace the last 3 scalar args (program_id x,y,z) with a SPIR-V
`WorkgroupId` builtin. VulkanizePass creates `spirv.GlobalVariable
@__builtin_workgroup_id` with `BuiltIn WorkgroupId` decoration, reads via
`spirv.CompositeExtract`. Host calls `vkCmdDispatch(num_blocks, 1, 1)` and
all blocks execute in parallel.

**Result:** Push constants reduced by 12 bytes. All blocks in a single dispatch.

### 18.3 C+2: Device-Local Memory

**Problem:** Storage buffers used host-visible PCIe BAR memory (~256MB), not
true GPU VRAM.

**Solution:** `createBuffer` now tries `DEVICE_LOCAL` VRAM first, creates a
host-visible staging buffer alongside, and uses `vkCmdCopyBuffer` for
transfers. `findMemoryTypeFallback` explicitly skips BAR memory (both
`DEVICE_LOCAL` and `HOST_VISIBLE`) to find the large non-host-visible heap.
Falls back to host-visible-only on integrated GPUs.

**Trap C2-1:** On discrete GPUs, there's a ~256MB memory type that is BOTH
`DEVICE_LOCAL` and `HOST_VISIBLE` — the PCIe BAR, not true VRAM. Many Vulkan
tutorials get this wrong.

### 18.4 C+3: Workgroup Shared Memory

**Problem:** Reductions (sum, max, softmax) ran serially on one thread.

**Solution:** `ConvertReductionToParallel` pass transforms `linalg.reduce` →
parallel tree reduction using shared memory (`memref.alloca` in address space 3
→ Workgroup) with `func.call @__vulkan_barrier` placeholders. VulkanizePass
promotes Function-scope shared Variables to module-scope Workgroup
GlobalVariables and replaces barrier calls with `spirv.ControlBarrier`.

This was the most complex step (8 traps documented). Key challenges:
- `gpu.barrier` is silently ignored by `convert-gpu-to-spirv` (trap C3-1)
- `convert-memref-to-spirv` forces ALL Variables to Function class (trap C3-3)
- Shared Variables must be at module scope per SPIR-V spec (trap C3-4)
- Adding the `local_id` arg group shifted all existing arg indices (trap C3-6)

**Arg layout after C+3:** `[...original..., num_programs×3, pid×3, local_id×3]`

### 18.5 C+4: Subgroup Operations

**Problem:** Tree reduction used 8 barriers for 256-element reductions.

**Solution:** Stop the shared-memory tree reduction at stride = subgroupSize
(32), then emit a single `spirv.GroupNonUniformFAdd/IAdd/FMax/SMax` for the
final 32→1 reduction. Reduces barriers from 8 to 3 for 256-element reductions.

Uses the same placeholder pattern: `func.call @__vulkan_subgroup_reduce_*`
→ VulkanizePass replaces with `spirv.GroupNonUniform*Op`. VCE triple
conditionally upgraded to V_1_3 with GroupNonUniform capabilities.

### 18.6 C+5: Cooperative Matrix (Buffer-Forwarding)

**Problem:** Cooperative matrix Load/Store requires StorageBuffer pointers,
but the `tt.load` → `memref.alloc` → `memref.copy` pattern produces
Function-class pointers in SPIR-V (invalid → driver crashes).

**Solution:** Buffer-forwarding architecture:
1. `ConvertMatmulToCooperative` traces matmul operands backward through the
   IR (alloc → memref.copy → reinterpret_cast → BlockArgument) to find
   the original buffer function arg indices
2. Stores them as module attributes: `vulkan.coop_buffer_args = [0, 1, 2]`
3. Emits a **no-arg** placeholder: `call @__vulkan_coop_matmul()`
4. VulkanizePass maps arg indices → StorageBuffer GlobalVariables,
   creates `AccessChain @bindingN[0][0]` → StorageBuffer pointer
5. Emits `KHRCooperativeMatrixLoad`, `MulAdd`, `FConvert`, `Store`

**Key insight:** The no-arg placeholder is essential — it eliminates
`FixAllocaStorageClass` type mangling that caused earlier attempts to fail.

VCE triple conditionally upgraded to V_1_6 with CooperativeMatrixKHR,
Float16, and StorageBuffer16BitAccess capabilities. Device extensions
queried via `vkEnumerateDeviceExtensionProperties` before enabling.

### 18.7 C+6: Discrete GPU Selection + Performance Baseline

**Problem:** `pickPhysicalDevice` selected the first device with a compute
queue, which could be an integrated GPU even when a discrete GPU was available.
No timing data existed for regression tracking.

**Solution:**
1. **GPU scoring** in `pickPhysicalDevice()`: discrete=3, integrated=2, virtual=1.
   Uses `std::max_element` to pick the highest-scoring device with a compute queue.
2. **Compile timing**: `time.perf_counter()` around the 5-stage pipeline in `comp()`.
3. **Dispatch timing**: 1 warmup + 5 timed dispatches, averaged, in `run()`.
4. **Results format**: Table with Compile (ms) and Dispatch (µs) columns per kernel.

### 18.8 Current VulkanizePass Responsibilities

After C+1 through C+5, VulkanizePass handles 9 responsibilities:
1. Buffer args → GlobalVariables with descriptor bindings
2. WorkgroupId builtin for program_id (C+1)
3. LocalInvocationId builtin for local_id (C+3)
4. Push constants for remaining scalar args
5. Shared memory Variable promotion to Workgroup (C+3)
6. Barrier replacement (C+3)
7. Subgroup reduce replacement (C+4)
8. Cooperative matrix replacement with buffer-forwarding (C+5)
9. Module wrapping (spirv.module, EntryPoint, ExecutionMode)

### 18.9 Test Suite Evolution

| Milestone | Tests | Key additions |
|-----------|-------|---------------|
| Base | 7 | vector_add, fma, gelu, swiglu, reduce_sum, reduce_max, softmax |
| +matmul | 9 | matmul_16x16, transpose |
| +C+1 | 10 | vadd_multiblock (1024 elements, 4 workgroups) |
| +C+2 | 11 | vadd_65k (65536 elements, 256 workgroups) |
| +C+5 | 12 | matmul_coop_f16 (16×16 cooperative matrix) |

## 19. Remaining Work and Future Directions

### 19.1 Current Gaps

| Item | Priority | Description |
|------|----------|-------------|
| ~~Performance baseline~~ | ~~Medium~~ | ✅ Done — compile (ms) + dispatch (µs) timing per kernel, warmup + 5-run average |
| ~~Discrete GPU preference~~ | ~~Medium~~ | ✅ Done — `pickPhysicalDevice` scores: discrete=3, integrated=2, virtual=1 (C+6) |
| Dynamic shapes | Medium | Relax fixed-shape assumptions in tests, metadata, and push-constant packing |
| Autotuning | Medium | Explore block-size, subgroup-size, and cooperative-matrix tuning strategies |
| `@triton.jit` integration | Medium | Wire the backend into Triton's higher-level launch/runtime flow |

### 19.2 Native TTG→SPIR-V as a Future Direction

A longer-term direction still targets the **Route A** approach from the
roadmap: consuming TritonGPU IR directly and lowering to SPIR-V, bypassing the
Linalg path. After C+1 through C+5, the motivation is no longer basic feature
coverage — it is richer layout-aware codegen, dynamic shapes, and deeper
performance tuning.

This would enable:

- **Direct TritonGPU layout lowering** instead of reconstructing structure later
- **Stronger autotuning hooks** for tile sizes, subgroup strategies, and MMA paths
- **Tighter `@triton.jit` integration** with less backend-specific glue
- **A cleaner long-term path** toward performance parity with CUDA on supported ops

The key challenge is implementing TTGIR layout encodings (blocked, sliced,
shared) for Vulkan's compute model, which differs significantly from
CUDA's warp-level execution.

### 19.3 Converter Coverage Summary

| Converter | Coverage Area | Ops Covered |
|-----------|---------------|-------------|
| SplatConverter | TritonToLinalg conversion | tt.splat |
| MakeRangeConverter | TritonToLinalg conversion | tt.make_range |
| BroadcastConverter | TritonToLinalg conversion | tt.broadcast |
| ExpandDimsConverter | TritonToLinalg conversion | tt.expand_dims |
| TransposeConverter | TritonToLinalg conversion | tt.trans |
| ReshapeConverter | TritonToLinalg conversion | tt.reshape |
| BitcastConverter | TritonToLinalg conversion | tt.bitcast |
| GetProgramIDConverter | TritonToLinalg conversion | tt.get_program_id |
| GetNumProgramsConverter | TritonToLinalg conversion | tt.get_num_programs |
| MatmulConverter | TritonToLinalg conversion | tt.dot |
| ReduceConverter | TritonToLinalg + later scalar-reduction fixes | tt.reduce (scalar fix covered by the reduction section in this guide) |
| DenseConstantConverter | TritonToLinalg conversion | arith.constant(dense) |
| AddPtrConverter | Pointer analysis and memory ops | tt.addptr |
| LoadConverter | Memory ops + later scalar-pointer fixes | tt.load (scalar unranked fix covered by the scalar load/store section) |
| StoreConverter | Memory ops + later scalar-pointer fixes | tt.store (scalar unranked fix covered by the scalar load/store section) |
| **AtomicRMWConverter** | **Core ops + testing expansion** | **tt.atomic_rmw** (10 RMW operations) |

Total: **16 converters** covering the unified backend scope described in this guide.

For the dedicated OpenCL emitter material, see `opencl-emitter-guide.md`.

---

## 20. Lessons Learned

### 20.1 On MLIR Pass Development

**MLIR's partial conversion is your friend.** Full conversion fails hard when
any op can't be converted. Partial conversion lets legal ops pass through and
converts only what it can. Use `addDynamicallyLegalOp` for ops that are
sometimes legal (like `arith.constant` — legal for scalars, illegal for
dense splat tensors).

**Type conversion is the hardest part.** Getting the `TypeConverter` right
(especially for Triton's pointer types) determines whether the rest of the
pipeline works. Debug type conversion failures by printing the
`adaptor.getOperands()` types vs. the original `op.getOperands()` types.

**NEVER use `op.getXxx()` for operands in a ConversionPattern (G-6).** In an
`OpConversionPattern`, the `TypeConverter` may have already converted operand
types — e.g., `tensor<256xi1>` → `memref<256xi1>`. Calling `op.getMask()`
returns the **original** tensor value, which is stale. Always use
`adaptor.getMask()` to get the type-converted value. Use `op` only for
attributes (`op.getLoc()`, `op.getAxis()`) and result types. This bug is
silent until you actually _use_ the operand — if you just null-check it
(`if (!mask)`), both versions work, hiding the problem.

**The walk-then-erase pattern is fragile.** When walking the IR to modify
or erase ops, always collect ops first, then process them. Erasing during
a walk invalidates iterators. Use `llvm::make_early_inc_range` or collect
into a `SmallVector` first.

### 20.2 On SPIR-V Conversion

**Silent failures are the norm.** Most SPIR-V conversion passes silently
skip ops they don't recognize. The only symptom is that `convert-func-to-spirv`
doesn't create `spirv.module`, because the function body has unconverted ops.
Always check for remaining non-SPIR-V ops after each pass.

**Pass ordering is a minefield.** The 10-step pipeline in §11.3 was discovered
empirically. Each ordering change was motivated by a silent failure. Document
the ordering and the reason for each step.

**Storage classes are semantic, not syntactic.** SPIR-V has strict rules about
which storage class each variable can have. `StorageBuffer` is for global
buffers (function args), `Function` is for local variables (alloca),
`Workgroup` is for shared memory. Getting these wrong causes silent
conversion failures.

### 20.3 On Triton Backend Development

**Start with the simplest possible kernel.** Vector addition is ideal: one
load pattern, one store pattern, one elementwise op, one program_id usage.
It exercises the full pipeline without edge cases.

**The `emitter.py` approach (regex-based MLIR→C) is fragile but useful for
debugging.** Being able to see human-readable C output from the MLIR IR is
invaluable for understanding what the pipeline produces. Don't rely on it
for production, but keep it as a debugging tool. For a dedicated emitter
walkthrough, see `opencl-emitter-guide.md`.

**Test intermediate representations, not just end-to-end.** Bugs in early
stages (e.g., wrong pointer analysis) manifest as wrong code in later stages
(e.g., wrong SPIR-V). Print IR after each stage and verify it looks correct.

### 20.4 On Correctness Pitfalls (Code Review Findings)

A comprehensive code review after completing Path C+ revealed several
correctness issues. These are now fixed but documented here to prevent
recurrence:

**Masked load/store must actually apply the mask (G-1).** The original
`LoadConverter` filled the destination with the `other` value, then did a
full `memref.copy` from the source — overwriting the fill, making the mask
useless. The fix uses `linalg.generic` with `arith.select` to conditionally
copy per element: `select(mask, source, other)`. Same pattern for stores:
use `linalg.generic` to selectively write where mask is true.

**Push-constant struct alignment matters (G-2).** SPIR-V requires each struct
member to be aligned to its natural alignment. The original code packed offsets
sequentially (0, 4, 8, ...) which is only correct for i32. With i64/f64 members,
the offset must be aligned to 8 bytes: `offset = (offset + size - 1) & ~(size - 1)`.

**Push constants are not interface variables (G-3).** The original code added
the `__push_constants` global to the `spirv.EntryPoint` interface list. Per
SPIR-V spec, only `StorageBuffer`/`Uniform`/`Input`/`Output` globals are
interface variables. Push constants are accessed via the `PushConstant` storage
class and are implicitly available.

**Subgroup size is not universal (G-4).** NVIDIA = 32, AMD = 64, Intel = 8/16/32.
The `ConvertReductionToParallel` pass now reads `vulkan.subgroup_size` from
the module, defaulting to 32. Set this attribute from the runtime when targeting
non-NVIDIA GPUs.

**Reduce identity must match the combiner (G-5).** Using zero as the identity
for `max` reduction produces wrong results (zero might be larger than all
elements for negative inputs). The `ReduceConverter` now emits a warning for
unrecognized combiners instead of silently returning zero.

### 20.5 On MSVC-Specific Issues

The Triton build on Windows (MSVC) has its own set of challenges documented
in the build skill. The Vulkan backend adds:

- **SPIR-V link libraries:** The library naming follows LLVM conventions
 (`MLIRSPIRVDialect`, not `MLIR_SPIRV_Dialect`). Check exact names in
 the LLVM build directory.
- **std::to_string and <string>:** MSVC implicitly includes `<string>` via
 other headers, but this isn't portable. Add explicit includes.
- **pybind11 and py::bytes:** The SPIR-V binary is returned as `py::bytes`
 from C++. Make sure the `reinterpret_cast<const char*>` is used correctly
 for the `SmallVector<uint32_t>` → bytes conversion.

---

## 21. Appendix: File Inventory and Change Summary

### 21.1 Source Files

| File | Relative Size | Coverage Area | Description |
|------|---------------|-------|-------------|
| `lib/Conversion/TritonToLinalg.cpp` | Main converter file | TritonToLinalg + memory ops | 16 conversion patterns + PtrState analysis |
| `lib/Conversion/TritonToLinalgPass.cpp` | Pass wrapper | TritonToLinalg conversion | Pass wrapper, type converter, target setup |
| `lib/Conversion/PrepareSPIRV.cpp` | Largest backend file | SPIR-V conversion | PrepareSPIRV, ConvertReductionToParallel, ConvertMatmulToCooperative, FixAllocaStorageClass, and Vulkanize passes |
| `triton_vulkan.cc` | Small binding layer | Backend skeleton + SPIR-V conversion | pybind11 bindings for all passes + serialization |
| `include/Conversion/TritonToLinalg.h` | Small public header | TritonToLinalg conversion | Public pass factory declarations |
| `backend/compiler.py` | Pipeline orchestrator | Pipeline orchestration across the backend | VulkanBackend with 6 pipeline stages |
| `backend/driver.py` | Small driver skeleton | Backend skeleton | VulkanDriver skeleton |
| `backend/emitter.py` | Medium debugging emitter | Pointer analysis and memory ops | OpenCL C emitter (debugging path) |
| `backend/__init__.py` | Tiny export surface | Backend skeleton | Backend discovery exports |
| `CMakeLists.txt` | Small build file | Pipeline orchestration across the backend | Build rules for static lib + pybind11 module |

### 21.2 Test Files

| File | Coverage Area | Tests |
|------|-------|-------|
| `test/test_triton_to_linalg.mlir` | TritonToLinalg conversion | splat, range, broadcast, expand, dot, trans, elementwise |
| `test/test_vector_add.ttir` | Pointer analysis and memory ops | Full kernel: splat→addptr→load→add→store |
| `test/test_arith_to_spirv.mlir` | SPIR-V conversion | addf, mulf, addi, muli → spirv |
| `test/test_math_to_spirv.mlir` | SPIR-V conversion | math ops → spirv |
| `test/test_scf_to_spirv.mlir` | SPIR-V conversion | scf.for → spirv structured control flow |
| `test/lit.cfg.py` | SPIR-V conversion | Lit test runner configuration |

### 21.3 Tool Files

| File | Purpose |
|------|---------|
| `tools/vulkan-opt.py` | SPIR-V conversion + serialization CLI wrapper |

### 21.4 Skill Files

| File | Purpose |
|------|---------|
| `.github/skills/triton-windows-vulkan/SKILL.md` | Unified Vulkan/SPIR-V backend skill with the current traps, passes, push-constant, and runtime notes |
| `.github/skills/triton-windows-opencl/SKILL.md` | Unified OpenCL/emitter skill covering the serial and parallel debugging paths |

### 21.5 Conversion Pattern Reference

| Pattern | Triton Op | Output | Covered In |
|---------|-----------|--------|-------|
| `SplatConverter` | `tt.splat` | `tensor.empty + linalg.fill` or `memref.reinterpret_cast` | TritonToLinalg conversion |
| `MakeRangeConverter` | `tt.make_range` | `linalg.generic + linalg.index` | TritonToLinalg conversion |
| `BroadcastConverter` | `tt.broadcast` | `linalg.generic + broadcast affine map` | TritonToLinalg conversion |
| `ExpandDimsConverter` | `tt.expand_dims` | `tensor.expand_shape` | TritonToLinalg conversion |
| `TransposeConverter` | `tt.trans` | `linalg.transpose` | TritonToLinalg conversion |
| `GetProgramIDConverter` | `tt.get_program_id` | Block argument extraction | TritonToLinalg conversion |
| `GetNumProgramsConverter` | `tt.get_num_programs` | Block argument extraction | TritonToLinalg conversion |
| `MatmulConverter` | `tt.dot` | `linalg.matmul + arith.addf` | TritonToLinalg conversion |
| `ReduceConverter` | `tt.reduce` | `linalg.reduce (cloned combiner)` | TritonToLinalg conversion |
| `BitcastConverter` | `tt.bitcast` | `arith.bitcast` | TritonToLinalg conversion |
| `ReshapeConverter` | `tt.reshape` | `tensor.expand/collapse_shape` | TritonToLinalg conversion |
| `DenseConstantConverter` | `arith.constant (splat)` | `tensor.empty + linalg.fill` | TritonToLinalg conversion |
| `AddPtrConverter` | `tt.addptr` | `memref.reinterpret_cast` | Pointer analysis and memory ops |
| `LoadConverter` | `tt.load` | `memref.alloc + fill + select(mask, src, other)` | Pointer analysis and memory ops |
| `StoreConverter` | `tt.store` | `select(mask, val, existing) → dest` or `materialize_in_destination` | Pointer analysis and memory ops |
| `ExpandReinterpretCast` | `memref.reinterpret_cast` | `memref.load/store with adjusted index` | SPIR-V conversion |
| `ExpandMemRefCopy` | `memref.copy` | `scf.for { load; store }` | SPIR-V conversion |
| `RemoveDealloc` | `memref.dealloc` | (erased) | SPIR-V conversion |

### 21.6 Modified Files

> **Historical note:** the original bring-up tracked exact line deltas here.
> Those counts drift quickly, so this appendix keeps only the durable summary
> of what changed.

| File | What Changed |
|------|--------------|
| `TritonToLinalg.cpp` | AtomicRMWConverter (memref-based), reduce scalar fix, scalar load/store unranked fix |
| `TritonToLinalgPass.cpp` | TypeConverter materializations (memref.cast + bufferization tensor↔memref), AtomicRMWOp illegal |
| `emitter.py` | Hex float, N-d linearization, 0-d memref, alloca, reshape aliases |
| `compiler.py` | `binary_ext` fix, brace-matching extraction, `make_opencl` docstring |
| `PrepareSPIRV.cpp` | `FinalizeSPIRVPass` removed, comments updated |
| `lit.cfg.py` | Simplified path detection |
| `.github/skills/triton-windows-vulkan/SKILL.md` | Skill documentation aligned with the unified backend naming and current SPIR-V/Vulkan notes |

### 21.7 New Test Files

| File | Pattern Tested |
|------|----------------|
| `test_vector_add.ttir` | Masked load + addf + masked store |
| `test_elementwise_mul.ttir` | Masked load + mulf + masked store |
| `test_fma.ttir` | 3-input multiply-add |
| `test_gelu.ttir` | Sigmoid GELU activation (masked) |
| `test_swiglu.ttir` | SiLU × gate activation |
| `test_reduce_sum.ttir` | Sum reduction (masked) + scalar store |
| `test_reduce_max.ttir` | Max reduction + scalar store |
| `test_softmax.ttir` | Compound: 2× reduce + exp + div |
| `test_matmul_simple.ttir` | 16×16 dot product with reshape |
| `test_broadcast_add.ttir` | Dual-source elementwise add |
| `test_transpose.ttir` | 16×16 transpose with reshape |
| `test_atomic_add.ttir` | Per-element atomic fadd (splat ptr → ranked memref) |
| `test_kernels.py` | End-to-end serial OpenCL test harness (OpenCL test suite) |
| `test_kernels_vulkan.py` | Vulkan SPIR-V dispatch harness (Vulkan kernel test suite) |

### 21.8 Op Coverage Matrix

| Op | test_vec | test_mul | test_fma | test_gelu | test_swi | test_rsum | test_rmax | test_soft | test_mat | test_brd | test_trn | test_atm |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| tt.load | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| tt.store | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| tt.splat | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| tt.addptr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| tt.make_range | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| tt.reduce | | | | | | ✓ | ✓ | ✓ | | | | |
| tt.dot | | | | | | | | | ✓ | | | |
| tt.reshape | | | | | | | | | ✓ | | ✓ | |
| tt.trans | | | | | | | | | | | ✓ | |
| tt.atomic_rmw | | | | | | | | | | | | ✓ |
| arith.mulf | | ✓ | ✓ | ✓ | ✓ | | | | | | | |
| arith.addf | ✓ | | ✓ | ✓ | | | | | | ✓ | | |
| arith.negf | | | | ✓ | ✓ | | | | | | | |
| arith.divf | | | | ✓ | ✓ | | | ✓ | | | | |
| arith.subf | | | | | | | | ✓ | | | | |
| math.exp | | | | ✓ | ✓ | | | ✓ | | | | |
| arith.maximumf | | | | | | | ✓ | ✓ | | | | |
| mask (load) | ✓ | | | ✓ | | ✓ | | | | | | |
| mask (store) | ✓ | | | ✓ | | | | | | | | |
| scalar store | | | | | | ✓ | ✓ | | | | | |
| dense const | | | | ✓ | ✓ | | | | ✓ | | | |
