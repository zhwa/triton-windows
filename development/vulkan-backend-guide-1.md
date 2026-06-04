# Triton Vulkan/SPIR-V Backend: A Comprehensive Guide

**Scope:** Phase 0 through Phase 1.5 — from backend skeleton to working SPIR-V
binary output.

**Audience:** Compiler engineers who want to understand, maintain, or extend
the Vulkan/SPIR-V backend for triton-windows.

**Prerequisites:** Familiarity with MLIR concepts (dialects, passes, type
conversion, pattern rewriting) and Triton's compilation model.

---

## Table of Contents

1. [Context and Motivation](#1-context-and-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 0: Backend Skeleton](#3-phase-0-backend-skeleton)
4. [Phase 0.5: TritonToLinalg Conversion](#4-phase-05-tritontolinalg-conversion)
5. [Phase 1: Pointer Analysis and Memory Ops](#5-phase-1-pointer-analysis-and-memory-ops)
6. [Phase 1.5: SPIR-V Conversion](#6-phase-15-spir-v-conversion)
7. [Build Integration](#7-build-integration)
8. [Testing and Diagnostics](#8-testing-and-diagnostics)
9. [Lessons Learned](#9-lessons-learned)
10. [Appendix: Complete File Inventory](#10-appendix-complete-file-inventory)

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
  TTIR  (Triton IR — tensor-level operations)
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
│   ├── __init__.py          # Backend discovery exports
│   ├── compiler.py          # VulkanBackend: 6-stage pipeline orchestration
│   ├── driver.py            # VulkanDriver: GPU target info (Phase 0 skeleton)
│   └── emitter.py           # OpenCL C emitter (debugging output path)
├── include/Conversion/
│   └── TritonToLinalg.h     # Public API: pass factory functions
├── lib/Conversion/
│   ├── TritonToLinalg.cpp   # 16 conversion patterns (1125 lines)
│   ├── TritonToLinalgPass.cpp  # Pass wrapper + type converter (210 lines)
│   └── PrepareSPIRV.cpp     # SPIR-V preparation + finalization (626 lines)
├── test/
│   ├── lit.cfg.py           # Lit test configuration
│   ├── test_triton_to_linalg.mlir
│   ├── test_vector_add.ttir
│   ├── test_arith_to_spirv.mlir
│   ├── test_math_to_spirv.mlir
│   └── test_scf_to_spirv.mlir
├── tools/
│   └── vulkan-opt.py        # SPIR-V conversion + serialization wrapper
├── CMakeLists.txt           # Build rules and link libraries
└── triton_vulkan.cc         # pybind11 bindings (passes + serialization)
```

### 2.2 The 6-Stage Pipeline

```
         ┌───────────────────────────────────────────────────────────┐
Stage 1: │  make_ttir     (shared TTIR passes: inline, CSE, etc.)   │
         └────────────────────────┬──────────────────────────────────┘
                                  │ TTIR (tensor-level Triton IR)
         ┌────────────────────────▼──────────────────────────────────┐
Stage 2: │  make_linalg   (C++ TritonToLinalg pass)                 │
         │    tt.splat → tensor.empty + linalg.fill                 │
         │    tt.dot   → linalg.matmul                              │
         │    tt.load  → memref.alloc + memref.copy                 │
         │    tt.func  → func.func                                  │
         └────────────────────────┬──────────────────────────────────┘
                                  │ Linalg/Tensor/MemRef IR
         ┌────────────────────────▼──────────────────────────────────┐
Stage 3: │  make_memref   (standard MLIR lowering passes)           │
         │    one_shot_bufferize    (tensor → memref)               │
         │    convert_linalg_to_loops (linalg.generic → scf.for)   │
         │    lower_affine         (affine.for → scf.for)           │
         │    convert_scf_to_cf    (scf.for → cf.br/cf.cond_br)    │
         └────────────────────────┬──────────────────────────────────┘
                                  │ MemRef + Arith + CF IR
         ┌────────────────────────▼──────────────────────────────────┐
Stage 4: │  make_spirv    (SPIR-V conversion — 3 sub-steps)         │
         │    prepare_spirv        (expand reinterpret_cast/copy)   │
         │    map_storage_class    (addr space 0 → StorageBuffer)   │
         │    fix_alloca           (alloca: StorageBuffer→Function) │
         │    convert_{memref,arith,cf,func}_to_spirv               │
         └────────────────────────┬──────────────────────────────────┘
                                  │ SPIR-V dialect IR
         ┌────────────────────────▼──────────────────────────────────┐
Stage 5: │  make_spv      (wrap in spirv.module + serialize)        │
         │    Extract spirv.func, wrap in spirv.module              │
         │    Add spirv.EntryPoint for GLCompute                    │
         │    mlir-translate --serialize-spirv → binary .spv        │
         └────────────────────────┬──────────────────────────────────┘
                                  │ SPIR-V binary bytes
                                  ▼
                         (future: Vulkan dispatch)
```

### 2.3 Relation to Triton's Backend Interface

Every Triton backend implements `BaseBackend` with these methods:

```python
class VulkanBackend(BaseBackend):
    def supports_target(target: GPUTarget) -> bool    # "vulkan" match
    def add_stages(stages, options, language=None)     # register pipeline
    def load_dialects(ctx)                             # register MLIR dialects
    def parse_options(opts) -> VulkanOptions           # backend-specific options
    def hash() -> str                                  # cache key
    def get_module_map() -> Dict[str, ModuleType]      # empty for now
```

The `add_stages()` method registers callable lambdas for each stage. Triton's
compiler core calls them in sequence, passing the MLIR module through each
stage. Each stage either transforms the module in-place (for MLIR passes) or
returns a new representation (for code emission).

---

## 3. Phase 0: Backend Skeleton

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

The driver provides GPU target information. For Phase 0, it reports inactive:

```python
class VulkanDriver(DriverBase):
    @staticmethod
    def is_active():
        return False  # Not ready to run kernels yet

    def get_current_target(self):
        return GPUTarget("vulkan", 0, 32)  # backend, arch, warp_size
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
    num_warps: int = 1        # Vulkan doesn't have warps, but the API requires it
    num_stages: int = 1       # No software pipelining in Route B
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

## 4. Phase 0.5: TritonToLinalg Conversion

This is the heart of the backend — a C++ MLIR pass that converts Triton IR
to standard MLIR dialects.

### 4.1 Design Principles

**Dialect conversion framework.** We use MLIR's `applyPartialConversion` with
a `ConversionTarget` that marks Triton ops as illegal and standard dialects as
legal. Each Triton op gets a conversion pattern that rewrites it to legal ops.

**Type conversion.** Triton's pointer types (`!tt.ptr<f32>`) have no equivalent
in standard MLIR. We use a `TypeConverter` to map them:

```
!tt.ptr<f32>                    → memref<*xf32>     (unranked memref)
tensor<256x!tt.ptr<f32>>        → memref<256xf32>   (ranked memref)
tensor<64x32xf32>               → tensor<64x32xf32> (unchanged)
f32, i32, index                 → f32, i32, index   (unchanged)
```

The key insight is that Triton's pointer tensors (`tensor<N x !tt.ptr<T>>`)
represent N pointers into a buffer. After conversion, these become a single
memref view of that buffer. The offset/stride information comes from the
`tt.addptr` pattern, not from the pointer type itself.

**Program info injection.** Triton kernels call `tl.program_id(axis)` and
`tl.num_programs(axis)` to get their position in the launch grid. In NVIDIA,
these map to `%ctaid.x` and `%nctaid.x` PTX registers. In our portable
backend, we pass them as explicit function arguments:

```
BEFORE: tt.func @kernel(%arg0: !tt.ptr<f32>) {
          %pid = tt.get_program_id x : i32
          ...
        }

AFTER:  func.func @kernel(%arg0: memref<*xf32>,
                           %num_x: i32, %num_y: i32, %num_z: i32,
                           %pid_x: i32, %pid_y: i32, %pid_z: i32) {
          // %pid_x replaces tt.get_program_id x
          ...
        }
```

Six i32 arguments are appended: `num_programs(x,y,z)` and `program_id(x,y,z)`.
The `GetProgramIDConverter` and `GetNumProgramsConverter` patterns simply
extract the corresponding block argument.

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
AFTER:  %ranked = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
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
input shape:  [128, 1]    → affine_map<(d0, d1) -> (d0, 0)>
output shape: [128, 64]   → affine_map<(d0, d1) -> (d0, d1)>
```

The body simply yields the input value — the affine map handles the
broadcast semantics.

#### 4.2.4 ExpandDimsConverter (tt.expand_dims → tensor.expand_shape)

Inserts a size-1 dimension at the specified axis. This maps directly to
`tensor.expand_shape` with appropriate reassociation indices.

```
BEFORE: tensor<128xf32>  with axis=1
AFTER:  tensor<128x1xf32> via tensor.expand_shape [[0, 1]]
```

#### 4.2.5 TransposeConverter (tt.trans → linalg.transpose)

Direct mapping. The `order` attribute specifies the permutation.

#### 4.2.6 MatmulConverter (tt.dot → linalg.matmul)

Triton's `tt.dot %a, %b, %c` computes `C += A @ B`. We decompose this into:

```
%zero_filled = linalg.fill(%zero, %init)          # Zero-initialized output
%product = linalg.matmul ins(%a, %b) outs(%zero)  # A @ B
%result = arith.addf %c, %product                 # C + (A @ B)
```

If `%c` is provably zero (a `tt.splat` of 0.0), we skip the addition.

**Performance note:** After `linalg-to-loops`, this becomes three nested
`scf.for` loops with no tiling. This is the primary reason Route B matmul
achieves <5% of CUDA performance. Tiling, vectorization, and cooperative
matrix support are Phase 3+ work.

#### 4.2.7 ReduceConverter (tt.reduce → linalg.reduce)

Triton's `tt.reduce` has an arbitrary combiner region (the body of the
Python `triton.language.reduce`). We clone this region into a `linalg.reduce`:

```python
# Triton Python
result = tl.sum(x, axis=0)  # Uses arith.addf as combiner
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
    // 3. Add program info args (6 × i32) to each tt.func
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

## 5. Phase 1: Pointer Analysis and Memory Ops

Phase 1 adds the ability to actually load from and store to memory — the
operations that make kernels do useful work.

### 5.1 The Pointer Problem

Triton uses pointer arithmetic to express memory access patterns:

```python
# Triton Python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x_ptrs = x_ptr + offsets          # tt.addptr
mask = offsets < N                 # bounds check
x = tl.load(x_ptrs, mask=mask)   # tt.load with mask
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
    Value source;                      // Base memref
    SmallVector<OpFoldResult> offsets;  // Per-dim offsets
    SmallVector<OpFoldResult> sizes;    // Per-dim sizes
    SmallVector<OpFoldResult> strides;  // Per-dim strides
    Value scalar;                      // Scalar offset (for scalar ptrs)
};
```

The `visitOperand` function walks the def-chain of a value backward,
building up `PtrState` as it goes:

```
tt.addptr ← visits ptr and offset operands recursively
  ├── tt.splat %base_ptr   → sets source = base memref, offsets = [0], strides = [0]
  └── arith.addi
      ├── tt.splat %pid    → scalar = pid value
      └── tt.make_range    → offset = [start], size = [end-start], stride = [1]
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
    //    the pointer structure (offsets, sizes, strides)
    PtrState state;
    visitOperand(op, state, ...);

    // 2. Override source with the type-converted ptr from the adaptor.
    //    After SplatConverter runs, adaptor.getPtr() is the base memref.
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
Full mask analysis (using subviews for partial copies) is deferred to Phase 2.

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
    pm.add(one_shot_bufferize)     # tensor → memref
    pm.add(convert_linalg_to_loops) # linalg.generic → scf.for
    pm.add(lower_affine)           # affine.for → scf.for
    pm.add(convert_scf_to_cf)      # scf.for → cf.br + cf.cond_br
    pm.add(canonicalize)
    pm.add(cse)
```

After this stage, the IR contains only: `memref.load`, `memref.store`,
`memref.alloc`, `memref.reinterpret_cast`, `memref.copy`, `memref.cast`,
`arith.*`, `cf.br`, `cf.cond_br`, and `func.func/func.return`.

### 5.7 Worked Example: vector_add

Here is the complete transformation of a vector addition kernel through
all stages up to the end of Phase 1.

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
- 3 `memref<*xf32>` args (type-converted pointers) + 1 `i32` (N) + 6 `i32` (program info)
- `memref.reinterpret_cast` for each pointer+offset pattern
- `memref.alloc + linalg.fill + memref.copy` for each load
- `arith.addf` on tensors (will become `linalg.generic` via elementwise promotion)
- `bufferization.materialize_in_destination` for the store

**After make_memref (bufferize + lower):**

All tensor ops become memref ops. Linalg generics become `scf.for` loops.
SCF loops become `cf.br`/`cf.cond_br` with block arguments. The result is
a flat control flow graph with explicit loads, stores, and branches.

---

## 6. Phase 1.5: SPIR-V Conversion

### 6.1 The Problem

After `make_memref`, the IR uses `memref`, `arith`, `cf`, and `func` dialects.
MLIR provides `convert-{memref,arith,cf,func}-to-spirv` passes. In theory,
running them in sequence should produce valid SPIR-V. In practice, there are
10 traps that require a custom preparation pass.

### 6.2 The Traps

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

AFTER:  %idx = arith.addi %i, %off
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
    return false;  // Silently skip!
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

**The fix:** We wrap `spirv.func` in `spirv.module` manually, either via
the `FinalizeSPIRVPass` (C++ path) or via text manipulation in `make_spv()`
(Python path). The wrapper includes `spirv.EntryPoint "GLCompute" @kernel`
for Vulkan compute dispatch.

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

The LLVM commit used by Triton 3.7.0 has specific API signatures that
differ from MLIR documentation. Key differences:

| API | Correct for This Commit |
|-----|-------------------------|
| `ReduceOp` creation | `ReduceOp::create(builder, loc, ...)` static method |
| `ConstantOp` value | Requires `cast<TypedAttr>(...)` |
| `FunctionInterfaces.h` | `mlir/Interfaces/FunctionInterfaces.h` |
| `ResourceLimitsAttr::get` | 4th arg is `ArrayAttr` |
| SCF→CF pass | `createSCFToControlFlowPass()` |

### 6.3 The SPIR-V Pass Pipeline (Critical Ordering)

The passes MUST be run in this exact order. Reordering causes silent failures.

```
Step 1: prepare_spirv
  ├── Attach spirv.target_env                   (TRAP #6)
  ├── Convert unranked memref → ranked memref   (TRAP #8)
  ├── Remove identity memref.cast ops
  ├── Expand memref.reinterpret_cast            (TRAP #1)
  ├── Expand memref.copy → scf.for loop         (TRAP #2)
  ├── Remove memref.dealloc
  └── Convert memref.alloc → memref.alloca

Step 2: convert_scf_to_cf
  └── Lower new scf.for from copy expansion     (TRAP #2 follow-up)

Step 3: canonicalize
  └── Clean up dead ops

Step 4: map_storage_class
  └── Map addr space 0 → StorageBuffer          (for function args)

Step 5: fix_alloca_storage_class
  └── Change alloca: StorageBuffer → Function   (TRAP #3)

Step 6: convert_memref_to_spirv
  └── memref.load → spirv.Load, memref.alloca → spirv.Variable

Step 7: convert_arith_to_spirv
  └── arith.addi → spirv.IAdd, etc.

Step 8: convert_cf_to_spirv
  └── cf.br → spirv.Branch, cf.cond_br → spirv.BranchConditional

Step 9: convert_func_to_spirv
  └── func.func → spirv.func                    (TRAP #4: may not create module)

Step 10: canonicalize
  └── Clean up
```

In `compiler.py`, steps 1-3 run in one pass manager, steps 4-5 in another,
and steps 6-10 in a third. Separate pass managers are needed because some
passes modify the type system in ways that require re-initialization.

### 6.4 The make_spv Stage (Serialization)

After `make_spirv`, the IR contains `spirv.func`, `spirv.Load`, `spirv.Store`,
`spirv.IAdd`, etc. inside a `builtin.module`. Serialization requires:

1. Extract the `spirv.func` (and any `spirv.GlobalVariable`) from the IR
2. Wrap in `spirv.module Logical GLSL450 requires #spirv.vce<...> { ... }`
3. Add `spirv.EntryPoint "GLCompute" @kernel_name`
4. Serialize via `mlir-translate --no-implicit-module --serialize-spirv`

The result is a valid SPIR-V binary (magic number `0x07230203`, version 1.0)
that can be loaded by any Vulkan implementation.

### 6.5 Worked Example: vector_add SPIR-V

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
      %ptr = spirv.AccessChain %x[%0, %idx]    // x[offset + i]
      %val = spirv.Load "StorageBuffer" %ptr    // load from global
      %local = spirv.AccessChain %buf_addr[%0, %i]
      spirv.Store "Function" %local, %val       // store to local

      // ... (similar for y, then add, then store to out)

    spirv.Return
  }

  spirv.EntryPoint "GLCompute" @vector_add_kernel
}
```

Serialized: ~1772 bytes of SPIR-V binary.

---

## 7. Build Integration

### 7.1 CMakeLists.txt

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

### 7.2 pybind11 Bindings (triton_vulkan.cc)

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

### 7.3 triton-opt Registration

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

## 8. Testing and Diagnostics

### 8.1 MLIR Lit Tests

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

### 8.2 vulkan-opt.py (Manual Pipeline Testing)

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

### 8.3 diagnose-spirv.py (Trap Detection)

A diagnostic tool that checks IR for known SPIR-V conversion issues:

```bash
# Check a single MLIR file
python diagnose-spirv.py lowered.mlir convert

# Run the full pipeline and diagnose each stage
python diagnose-spirv.py --pipeline vector_add.ttir
```

The `--pipeline` mode instantiates the `VulkanBackend`, runs each stage, and
reports per-stage diagnostics. It checks for all 10 traps documented in the
SPIR-V skill.

### 8.4 Manual End-to-End Verification

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

## 9. Lessons Learned

### 9.1 On MLIR Pass Development

**MLIR's partial conversion is your friend.** Full conversion fails hard when
any op can't be converted. Partial conversion lets legal ops pass through and
converts only what it can. Use `addDynamicallyLegalOp` for ops that are
sometimes legal (like `arith.constant` — legal for scalars, illegal for
dense splat tensors).

**Type conversion is the hardest part.** Getting the `TypeConverter` right
(especially for Triton's pointer types) determines whether the rest of the
pipeline works. Debug type conversion failures by printing the
`adaptor.getOperands()` types vs. the original `op.getOperands()` types.

**The walk-then-erase pattern is fragile.** When walking the IR to modify
or erase ops, always collect ops first, then process them. Erasing during
a walk invalidates iterators. Use `llvm::make_early_inc_range` or collect
into a `SmallVector` first.

### 9.2 On SPIR-V Conversion

**Silent failures are the norm.** Most SPIR-V conversion passes silently
skip ops they don't recognize. The only symptom is that `convert-func-to-spirv`
doesn't create `spirv.module`, because the function body has unconverted ops.
Always check for remaining non-SPIR-V ops after each pass.

**Pass ordering is a minefield.** The 10-step pipeline in §6.3 was discovered
empirically. Each ordering change was motivated by a silent failure. Document
the ordering and the reason for each step.

**Storage classes are semantic, not syntactic.** SPIR-V has strict rules about
which storage class each variable can have. `StorageBuffer` is for global
buffers (function args), `Function` is for local variables (alloca),
`Workgroup` is for shared memory. Getting these wrong causes silent
conversion failures.

### 9.3 On Triton Backend Development

**Start with the simplest possible kernel.** Vector addition is ideal: one
load pattern, one store pattern, one elementwise op, one program_id usage.
It exercises the full pipeline without edge cases.

**The `emitter.py` approach (regex-based MLIR→C) is fragile but useful for
debugging.** Being able to see human-readable C output from the MLIR IR is
invaluable for understanding what the pipeline produces. Don't rely on it
for production, but keep it as a debugging tool.

**Test intermediate representations, not just end-to-end.** Bugs in early
stages (e.g., wrong pointer analysis) manifest as wrong code in later stages
(e.g., wrong SPIR-V). Print IR after each stage and verify it looks correct.

### 9.4 On MSVC-Specific Issues

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

## 10. Appendix: Complete File Inventory

### 10.1 Source Files

| File | Lines | Phase | Description |
|------|-------|-------|-------------|
| `lib/Conversion/TritonToLinalg.cpp` | 1002 | 0.5+1 | 16 conversion patterns + PtrState analysis |
| `lib/Conversion/TritonToLinalgPass.cpp` | 176 | 0.5 | Pass wrapper, type converter, target setup |
| `lib/Conversion/PrepareSPIRV.cpp` | 321 | 1.5 | PrepareSPIRV + FixAlloca + FinalizeSPIRV passes |
| `triton_vulkan.cc` | 141 | 0+1.5 | pybind11 bindings for all passes + serialization |
| `include/Conversion/TritonToLinalg.h` | 32 | 0.5 | Public pass factory declarations |
| `backend/compiler.py` | 273 | 0-1.5 | VulkanBackend with 6 pipeline stages |
| `backend/driver.py` | 44 | 0 | VulkanDriver skeleton |
| `backend/emitter.py` | 490 | 1 | OpenCL C emitter (debugging path) |
| `backend/__init__.py` | — | 0 | Backend discovery exports |
| `CMakeLists.txt` | 68 | 0-1.5 | Build rules for static lib + pybind11 module |

### 10.2 Test Files

| File | Phase | Tests |
|------|-------|-------|
| `test/test_triton_to_linalg.mlir` | 0.5 | splat, range, broadcast, expand, dot, trans, elementwise |
| `test/test_vector_add.ttir` | 1 | Full kernel: splat→addptr→load→add→store |
| `test/test_arith_to_spirv.mlir` | 1.5 | addf, mulf, addi, muli → spirv |
| `test/test_math_to_spirv.mlir` | 1.5 | math ops → spirv |
| `test/test_scf_to_spirv.mlir` | 1.5 | scf.for → spirv structured control flow |
| `test/lit.cfg.py` | 1.5 | Lit test runner configuration |

### 10.3 Tool Files

| File | Purpose |
|------|---------|
| `tools/vulkan-opt.py` | SPIR-V conversion + serialization CLI wrapper |

### 10.4 Skill Files

| File | Purpose |
|------|---------|
| `.github/skills/triton-windows-spirv-setup/SKILL.md` | 10 traps documented with fixes |
| `.github/skills/triton-windows-spirv-setup/scripts/diagnose-spirv.py` | Automated trap detection |

### 10.5 Conversion Pattern Reference

| Pattern | Triton Op | Output | Phase |
|---------|-----------|--------|-------|
| `SplatConverter` | `tt.splat` | `tensor.empty + linalg.fill` or `memref.reinterpret_cast` | 0.5 |
| `MakeRangeConverter` | `tt.make_range` | `linalg.generic + linalg.index` | 0.5 |
| `BroadcastConverter` | `tt.broadcast` | `linalg.generic + broadcast affine map` | 0.5 |
| `ExpandDimsConverter` | `tt.expand_dims` | `tensor.expand_shape` | 0.5 |
| `TransposeConverter` | `tt.trans` | `linalg.transpose` | 0.5 |
| `GetProgramIDConverter` | `tt.get_program_id` | Block argument extraction | 0.5 |
| `GetNumProgramsConverter` | `tt.get_num_programs` | Block argument extraction | 0.5 |
| `MatmulConverter` | `tt.dot` | `linalg.matmul + arith.addf` | 0.5 |
| `ReduceConverter` | `tt.reduce` | `linalg.reduce (cloned combiner)` | 0.5 |
| `BitcastConverter` | `tt.bitcast` | `arith.bitcast` | 0.5 |
| `ReshapeConverter` | `tt.reshape` | `tensor.expand/collapse_shape` | 0.5 |
| `DenseConstantConverter` | `arith.constant (splat)` | `tensor.empty + linalg.fill` | 0.5 |
| `AddPtrConverter` | `tt.addptr` | `memref.reinterpret_cast` | 1 |
| `LoadConverter` | `tt.load` | `memref.alloc + fill + copy + to_tensor` | 1 |
| `StoreConverter` | `tt.store` | `materialize_in_destination` | 1 |
| `ExpandReinterpretCast` | `memref.reinterpret_cast` | `memref.load/store with adjusted index` | 1.5 |
| `ExpandMemRefCopy` | `memref.copy` | `scf.for { load; store }` | 1.5 |
| `RemoveDealloc` | `memref.dealloc` | (erased) | 1.5 |

---

*Document created: June 2026*
*Covers: triton-windows Vulkan/SPIR-V backend, Phases 0–1.5*
*Based on: Triton 3.7.0, LLVM/MLIR from cmake/llvm-hash.txt, MSVC v143 (14.44)*
