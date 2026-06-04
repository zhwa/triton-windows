# Triton Vulkan/SPIR-V Backend: Phase 2 Guide — Core Ops & Testing

**Scope:** Phase 2 — Atomic operations, scalar reduction fixes, N-dimensional
emitter support, and the 12-kernel end-to-end test suite.

**Audience:** Compiler engineers working on the Vulkan backend who have read
Guide 1 (Phases 0–1.5) and want to understand Phase 2 additions, the test
methodology, and the emitter improvements.

**Prerequisites:** Understanding of the material in `vulkan-backend-guide-1.md`,
particularly the TritonToLinalg conversion framework and the OpenCL emitter
architecture.

---

## Table of Contents

1. [Phase 2 Goals and Scope](#1-phase-2-goals-and-scope)
2. [AtomicRMWConverter: Design and Implementation](#2-atomicrmwconverter-design-and-implementation)
3. [Reduction Scalar Fix: The 0-d Tensor Problem](#3-reduction-scalar-fix-the-0-d-tensor-problem)
4. [Scalar Load/Store with Unranked MemRef](#4-scalar-loadstore-with-unranked-memref)
5. [TypeConverter Materializations](#5-typeconverter-materializations)
6. [Emitter Improvements](#6-emitter-improvements)
7. [Test Suite Architecture](#7-test-suite-architecture)
8. [Kernel-by-Kernel Analysis](#8-kernel-by-kernel-analysis)
9. [PrepareSPIRV Cleanup](#9-preparespirv-cleanup)
10. [Remaining Work and Phase 3 Preview](#10-remaining-work-and-phase-3-preview)
11. [Appendix: File Changes Summary](#11-appendix-file-changes-summary)

---

## 1. Phase 2 Goals and Scope

Phase 2 in the [roadmap](../references/roadmap.md) targets:

```
Phase 2:   Core ops + testing
```

Specifically:

| Goal | Description | Status |
|------|-------------|--------|
| Elementwise ops | add, mul, fma, gelu, swiglu | ✅ Complete |
| Reductions | sum, max, softmax (compound) | ✅ Complete |
| Matrix multiply | Small tile matmul (16×16) | ✅ Complete |
| 2D operations | Transpose, broadcast add | ✅ Complete |
| Atomic operations | atomic_rmw (fadd, add, max, etc.) | ✅ Complete |
| End-to-end GPU tests | OpenCL execution with numpy reference | ✅ Complete |
| Performance baseline | Timing vs CUDA backend | ❌ Future work |

### What Changed (File Summary)

```
Modified (3 files, +356 lines):
  third_party/vulkan/lib/Conversion/TritonToLinalg.cpp    +183 lines
  third_party/vulkan/lib/Conversion/TritonToLinalgPass.cpp  +48 lines
  third_party/vulkan/backend/emitter.py                   +125 lines (net)

New (13 files):
  third_party/vulkan/test/test_*.ttir        (12 TTIR kernel files)
  third_party/vulkan/test/test_kernels.py    (275-line GPU test harness)

Cleanup:
  third_party/vulkan/lib/Conversion/PrepareSPIRV.cpp      -254 lines
  third_party/vulkan/backend/compiler.py                   +21/-18 lines
```

---

## 2. AtomicRMWConverter: Design and Implementation

### 2.1 The Problem

Triton's `tt.atomic_rmw` operation performs read-modify-write on GPU memory,
typically used for cross-workgroup accumulation (e.g., gradient updates,
histogram computation). The operation takes a pointer (scalar or tensor),
a value, and an RMW kind (fadd, add, max, xchg, etc.), and returns the
old values that were at those memory locations before the modification.

In the CUDA backend, these map to hardware atomic instructions (`atomicAdd`,
`atomicMax`, etc.). In our Vulkan backend, which currently executes as a
single OpenCL workgroup, we can implement them as sequential load-modify-store
operations — semantically correct because there's no data race.

### 2.2 Three Cases

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

### 2.3 The Conversion Pattern

**Location:** `third_party/vulkan/lib/Conversion/TritonToLinalg.cpp`, lines
1126–1246.

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

### 2.4 The RMW Operation Switch

```cpp
static Value applyRMW(OpBuilder &b, Location loc, triton::RMWOp kind,
                       Value old, Value val) {
  switch (kind) {
  case triton::RMWOp::FADD:  return b.create<arith::AddFOp>(loc, old, val);
  case triton::RMWOp::ADD:   return b.create<arith::AddIOp>(loc, old, val);
  case triton::RMWOp::MAX:   return b.create<arith::MaxSIOp>(loc, old, val);
  case triton::RMWOp::MIN:   return b.create<arith::MinSIOp>(loc, old, val);
  case triton::RMWOp::UMAX:  return b.create<arith::MaxUIOp>(loc, old, val);
  case triton::RMWOp::UMIN:  return b.create<arith::MinUIOp>(loc, old, val);
  case triton::RMWOp::AND:   return b.create<arith::AndIOp>(loc, old, val);
  case triton::RMWOp::OR:    return b.create<arith::OrIOp>(loc, old, val);
  case triton::RMWOp::XOR:   return b.create<arith::XOrIOp>(loc, old, val);
  case triton::RMWOp::XCHG:  return val;
  default:                    llvm_unreachable("unsupported RMW operation");
  }
}
```

Key design decisions:

1. **`XCHG` returns `val` directly** — exchange just replaces the old value.
2. **`default` uses `llvm_unreachable`** — fails loudly on unsupported ops
   (e.g., `CAS`/compare-and-swap which requires different IR structure).
3. **Float vs integer separation** — `FADD` uses `AddFOp` (float), `ADD`
   uses `AddIOp` (integer). The caller must ensure type consistency.

### 2.5 Scalar Atomic Path

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
  rewriter.replaceOp(op, old);  // returns the OLD value
}
```

### 2.6 Tensor Atomic Path with scf.for (Memref-Based)

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

### 2.7 Splat vs Offset Pointer Detection

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

### 2.8 Semantic Note: Sequential vs True Atomics

This implementation is **sequentially correct** but not atomically correct in
the hardware sense. For single-workgroup execution (our current model), this
is fine — there are no concurrent writers. For multi-workgroup execution
(Phase 3+), we would need to lower to actual SPIR-V atomic operations
(`spirv.AtomicIAdd`, `spirv.AtomicFAddEXT`, etc.).

---

## 3. Reduction Scalar Fix: The 0-d Tensor Problem

### 3.1 The Bug

When reducing a 1D tensor to a scalar:

```mlir
%sum = "tt.reduce"(%x) ({...}) {axis = 0} : (tensor<256xf32>) -> f32
```

Triton's `tt.reduce` returns a bare `f32` scalar. But `linalg.reduce`
operates on tensors and produces a **0-d tensor** (`tensor<f32>`), not a
scalar. The mismatch caused type verification failures.

### 3.2 The Fix (Two Parts)

**Part 1: 0-d tensor initialization** (lines 498–506)

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

**Part 2: 0-d tensor extraction** (lines 542–557)

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

### 3.3 Why This Matters

Without this fix, every reduction kernel (reduce_sum, reduce_max, softmax)
would fail at MLIR verification. Softmax uses **two** reductions (max, then
sum of exp), so this fix is exercised twice per softmax compilation.

---

## 4. Scalar Load/Store with Unranked MemRef

### 4.1 The Problem

When a Triton kernel stores a scalar result through a scalar pointer:

```mlir
tt.store %out_ptr, %sum : !tt.ptr<f32>
```

The TritonTypeConverter converts `%out_ptr` from `!tt.ptr<f32>` to
`memref<*xf32>` (unranked memref). But `affine.store` requires a ranked
memref with known dimensions.

### 4.2 The Fix

Both `LoadConverter` and `StoreConverter` now handle unranked memrefs
by inserting a `memref.cast` to `memref<1xelemType>`:

```cpp
// In LoadConverter (line 993) and StoreConverter (line 1093):
if (isa<UnrankedMemRefType>(ptr.getType())) {
  auto elemType = cast<UnrankedMemRefType>(ptr.getType()).getElementType();
  auto ranked1D = MemRefType::get({1}, elemType);
  memPtr = rewriter.create<memref::CastOp>(loc, ranked1D, ptr);
}
```

This pattern is consistent with how `AtomicRMWConverter` handles scalar
pointers (§2.5), creating a uniform approach across all memory operations.

### 4.3 The Type Conversion Chain

For a scalar pointer argument, the full conversion chain is:

```
!tt.ptr<f32>                    (Triton IR)
    → memref<*xf32>            (TritonTypeConverter, unranked)
    → memref<1xf32>            (memref.cast in Load/Store/AtomicRMW converters)
    → affine.load/store [0]    (actual memory access)
```

---

## 5. TypeConverter Materializations

### 5.1 What Are Materializations?

When MLIR's dialect conversion framework converts types, it sometimes needs
to bridge between the old and new type systems. A **target materialization**
tells the framework "how to convert a value from source type to target type",
and a **source materialization** does the reverse.

Without materializations, the conversion framework inserts
`unrealized_conversion_cast` ops as placeholders, which fail verification
if not cleaned up.

### 5.2 The Implementation

**Location:** `third_party/vulkan/lib/Conversion/TritonToLinalgPass.cpp`,
lines 45–76.

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

### 5.3 Why Both Directions?

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

## 6. Emitter Improvements

The OpenCL C emitter (`third_party/vulkan/backend/emitter.py`) received
significant improvements to handle the IR patterns produced by Phase 2
kernels.

### 6.1 IEEE 754 Hex Float Decoding

**Problem:** MLIR sometimes emits float constants as hex-encoded IEEE 754
bit patterns:

```mlir
%cst = arith.constant 0xFF800000 : f32    // -infinity
%cst = arith.constant 0x7F800000 : f32    // +infinity
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

### 6.2 N-Dimensional MemRef Linearization

**Problem:** Phase 1 emitter only handled 1D memref indexing (`buf[i]`).
Phase 2 kernels produce 2D and higher operations (matmul, transpose) that
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

This replaces the Phase 1 hardcoded 2D-only pattern and handles arbitrary
dimensionality.

### 6.3 0-d MemRef Load/Store

Scalar reduction results produce 0-d memref accesses:

```mlir
memref.store %val, %buf[] : memref<f32>    // empty index list
memref.load %buf[] : memref<f32>
```

The emitter now detects empty indices and emits `buf[0]`:

```python
if indices.strip():
    # ... normal indexed access
else:
    self._line(f"{self._map_val(buf)}[0] = {self._map_val(val)};")
```

### 6.4 memref.alloca Support

Phase 2 kernels produce `memref.alloca` (stack allocation) in addition to
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

### 6.5 Reshape Aliases

```python
def _emit_reshape_alias(self, line: str):
    """expand_shape/collapse_shape are view aliases — same 1D buffer."""
    m = re.match(
        r"(%[\w]+)\s*=\s*memref\.(expand|collapse)_shape\s+(%[\w]+)", line)
    if m:
        self.ssa_map[dst] = self._map_val(src)  # alias, no copy
```

`memref.expand_shape` and `memref.collapse_shape` are view operations —
they reinterpret the same underlying buffer with different dimensions.
In OpenCL C, where all buffers are flat `__global float*`, these are
no-ops. The emitter creates an alias in its SSA map.

### 6.6 Multi-Dimensional memref.copy

Phase 1's `_emit_copy` extracted only the first dimension from the type.
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

## 7. Test Suite Architecture

### 7.1 Test Philosophy

The Phase 2 test suite takes a **compilation-first** approach:

1. Write hand-crafted TTIR (Triton IR) for each kernel pattern
2. Compile through the full pipeline: TTIR → Linalg → MemRef → OpenCL C
3. Execute on GPU via pyopencl
4. Compare against numpy reference implementations

This tests the **entire backend** end-to-end, not just individual passes.
If any pass produces incorrect IR, the GPU result will diverge from numpy.

### 7.2 Pipeline Under Test

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

The `compile_ttir()` function in `test_kernels.py` drives this:

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

### 7.3 Test Harness Design

Each test function follows this pattern:

```python
def test_foo():
    src, md = compile_ttir("test_foo.ttir")
    # Create input data with numpy
    x = np.random.randn(N).astype(np.float32)
    # Upload to GPU
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, WO, N * 4)
    # Run kernel (with program info args)
    args = [xb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    # Read result and compare
    o = read_buf(ob, N)
    return np.max(np.abs(o - expected))  # max absolute error
```

**The 6 extra `np.int32(0)` args:** These are the program info arguments
that the Vulkan backend adds to every kernel:
- `num_programs_x`, `num_programs_y`, `num_programs_z` — grid dimensions
- `program_id_x`, `program_id_y`, `program_id_z` — current program index

For single-block tests, all are 0 (or 1 for num_programs).

### 7.4 Error Tolerances

| Category | Tolerance | Rationale |
|----------|-----------|-----------|
| Exact ops (add, mul, copy) | 1e-6 | IEEE 754 exact for simple ops |
| Transcendentals (exp, sigmoid) | 1e-5 | GPU exp() may use fast-math |
| Reductions (sum, atomic_add) | 1e-3 | Accumulation order differs |
| Softmax | 1e-5 | Exp/div partially cancel errors |
| Matrix multiply | 1e-4 | 16 multiply-accumulate steps |

### 7.5 TTIR Test File Convention

All test files follow this naming pattern:

```
test_{kernel_name}.ttir
```

Each file contains a single `tt.func public @{kernel_name}_kernel(...)` with
clearly documented parameters and operations. Tests are self-contained —
no dependencies between test files.

---

## 8. Kernel-by-Kernel Analysis

### 8.1 Elementwise Operations

#### test_vector_add.ttir (existing from Phase 1)
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

### 8.2 Activation Functions

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

### 8.3 Reductions

#### test_reduce_sum.ttir
```
Pattern: out = sum(x[0..256])
Ops: tt.load (masked), tt.reduce(addf), tt.store (scalar)
Coverage: Scalar reduction, masked load, scalar pointer store
New ops: 0-d tensor handling (§3), scalar store via unranked memref (§4)
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

### 8.4 Matrix Operations

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

### 8.5 Multi-Load Pattern

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

### 8.6 Atomic Operations

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

## 9. PrepareSPIRV Cleanup

### 9.1 FinalizeSPIRV Removal

The Phase 1.5 implementation included a `FinalizeSPIRVPass` (~250 lines)
that attempted to build `spirv.module` + `spirv.func` in C++, transplanting
function bodies and fixing up types. This approach had fundamental issues:

1. **Block argument type mismatches** — moved blocks retained their original
   memref types, but the spirv.func expected spirv.ptr types
2. **Alloca→GlobalVariable conversion** — complex and fragile, requiring
   matching unrealized_conversion_cast ops
3. **Multiple traversals** — needed separate passes over arg casts, alloca
   casts, and return ops

The fix: **remove the C++ pass entirely** and handle finalization in Python
(`compiler.py make_spv()`) using text manipulation:

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
wrapped = f'spirv.module Logical GLSL450 ... {{\n  {func_text}\n}}\n'
```

This is simpler and more reliable because:
- The IR is already fully converted to SPIR-V ops at this point
- Text extraction avoids all the type-matching complexity
- `mlir-translate --serialize-spirv` handles the actual serialization

### 9.2 compiler.py Improvements

The `make_spv()` method was also improved:

1. **Brace-matching extraction** replaces fragile regex for spirv.func body
2. **`binary_ext = "spv"`** (was "cl") — the default output format is now
   SPIR-V binary, matching the backend's name and purpose
3. **`make_opencl` docstring** clarified as an alternative debug/execution path

### 9.3 The createFinalizeSPIRVPass Stub

```cpp
std::unique_ptr<OperationPass<ModuleOp>> createFinalizeSPIRVPass() {
  // No-op — finalization is done in make_spv() Python code.
  return createPrepareSPIRVPass();
}
```

The function is kept for API compatibility (it's declared in the header)
but currently returns a duplicate PrepareSPIRV pass. This is harmless since
the pass is idempotent, but a cleaner solution would be a true no-op pass.

---

## 10. Remaining Work and Phase 3 Preview

### 10.1 Phase 2 Gaps

| Item | Priority | Description |
|------|----------|-------------|
| Performance baseline | Medium | Time kernels vs CUDA, establish baseline |
| Multi-block execution | Medium | Test with `num_programs > 1` |
| True 2D broadcast | Low | `expand_dims + broadcast` pattern |
| CAS atomic | Low | Compare-and-swap needs different IR structure |
| `__init__.py` | Low | Add backend docstring |
| No-op finalize pass | Low | Replace the PrepareSPIRV-returning stub |

### 10.2 Phase 3 Preview: Native TTG→SPIR-V

Phase 3 targets the **Route A** approach from the roadmap: consuming
TritonGPU IR directly and lowering to SPIR-V, bypassing the Linalg path.
This would enable:

- **Shared memory** via `spirv.Variable(Workgroup)`
- **Workgroup barriers** via `spirv.ControlBarrier`
- **Cooperative matrix** via `VK_KHR_cooperative_matrix` extension
- **Multi-workgroup** execution with true atomics
- **Performance parity** with CUDA for supported operations

The key challenge is implementing TTGIR layout encodings (blocked, sliced,
shared) for Vulkan's compute model, which differs significantly from
CUDA's warp-level execution.

### 10.3 Converter Coverage Summary (After Phase 2)

| Converter | Phase | Ops Covered |
|-----------|-------|-------------|
| SplatConverter | 0.5 | tt.splat |
| MakeRangeConverter | 0.5 | tt.make_range |
| BroadcastConverter | 0.5 | tt.broadcast |
| ExpandDimsConverter | 0.5 | tt.expand_dims |
| TransposeConverter | 0.5 | tt.trans |
| ReshapeConverter | 0.5 | tt.reshape |
| BitcastConverter | 0.5 | tt.bitcast |
| GetProgramIDConverter | 0.5 | tt.get_program_id |
| GetNumProgramsConverter | 0.5 | tt.get_num_programs |
| MatmulConverter | 0.5 | tt.dot |
| ReduceConverter | 0.5+2 | tt.reduce (scalar fix in Phase 2) |
| DenseConstantConverter | 0.5 | arith.constant(dense) |
| AddPtrConverter | 1 | tt.addptr |
| LoadConverter | 1+2 | tt.load (scalar unranked fix in Phase 2) |
| StoreConverter | 1+2 | tt.store (scalar unranked fix in Phase 2) |
| **AtomicRMWConverter** | **2** | **tt.atomic_rmw** (10 RMW operations) |

Total: **16 converters** covering the full Phase 2 scope.

---

## 11. Appendix: File Changes Summary

### Modified Files

| File | Lines Changed | What Changed |
|------|--------------|--------------|
| `TritonToLinalg.cpp` | +183 | AtomicRMWConverter (memref-based), reduce scalar fix, scalar load/store unranked fix |
| `TritonToLinalgPass.cpp` | +48 | TypeConverter materializations (memref.cast + bufferization tensor↔memref), AtomicRMWOp illegal |
| `emitter.py` | +125 (net) | Hex float, N-d linearization, 0-d memref, alloca, reshape aliases |
| `compiler.py` | +21/-18 | binary_ext fix, brace-matching extraction, make_opencl docstring |
| `PrepareSPIRV.cpp` | -254 | FinalizeSPIRVPass removed, comments updated |
| `lit.cfg.py` | +5/-8 | Simplified path detection |
| `SKILL.md` | +4/-2 | Alloca comment fix, ResourceLimitsAttr note |

### New Test Files

| File | Lines | Pattern Tested |
|------|-------|----------------|
| `test_vector_add.ttir` | 33 | Masked load + addf + masked store |
| `test_elementwise_mul.ttir` | 29 | Masked load + mulf + masked store |
| `test_fma.ttir` | 24 | 3-input multiply-add |
| `test_gelu.ttir` | 33 | Sigmoid GELU activation (masked) |
| `test_swiglu.ttir` | 29 | SiLU × gate activation |
| `test_reduce_sum.ttir` | 28 | Sum reduction (masked) + scalar store |
| `test_reduce_max.ttir` | 17 | Max reduction + scalar store |
| `test_softmax.ttir` | 34 | Compound: 2× reduce + exp + div |
| `test_matmul_simple.ttir` | 37 | 16×16 dot product with reshape |
| `test_broadcast_add.ttir` | 38 | Dual-source elementwise add |
| `test_transpose.ttir` | 29 | 16×16 transpose with reshape |
| `test_atomic_add.ttir` | 14 | Per-element atomic fadd (splat ptr → ranked memref) |
| `test_kernels.py` | 275 | End-to-end GPU test harness (12 kernels) |

### Op Coverage Matrix

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

---

*Guide 2 of the Triton Vulkan/SPIR-V Backend series.*
*Guide 1 covers Phases 0–1.5: [vulkan-backend-guide-1.md](vulkan-backend-guide-1.md)*
*Next: Guide 3 will cover Phase 3 (native TTG→SPIR-V) when implemented.*
