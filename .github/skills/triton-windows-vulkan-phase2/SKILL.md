---
name: triton-windows-vulkan-phase2
description: "Complete guide for the Triton Vulkan backend Phase 2: core kernel ops, OpenCL emitter, and GPU testing. Covers all 16 converters, the OpenCL C emitter architecture, N-d memref handling, reduction identity values, matmul via linalg, transpose/reshape patterns, and the full 12-kernel GPU test suite. Use for: adding new kernel support, fixing emitter output, debugging TritonToLinalg failures, understanding the PtrState/AddPtr pattern, or extending the test suite."
argument-hint: "new-converter | emitter-fix | reduce-init | matmul | pointer-pattern | test-suite | full-overview"
user-invocable: true
---

# Triton-Windows Vulkan Backend — Phase 2 Skill

You are an expert at the Triton Vulkan/SPIR-V backend. This skill documents
every converter, emitter pattern, workaround, and testing technique discovered
during Phase 2 implementation. Phase 1 is covered by `triton-windows-spirv-setup`.

## Quick Status

**14/14 GPU-verified kernels on RTX 2080 Ti via pyopencl:**
vector_add, elementwise_mul, fma, gelu, swiglu, reduce_sum, reduce_max,
softmax, matmul_16x16, broadcast_add, transpose_16x16, atomic_add,
vector_add_4blk (multi-block), broadcast_2d (expand_dims+broadcast)

**Pipeline:** TTIR → make_ttir → make_linalg → make_memref → make_opencl → GPU

---

## Architecture

### File Layout

| File | Lines | Purpose |
|------|-------|---------|
| `third_party/vulkan/lib/Conversion/TritonToLinalg.cpp` | ~1300 | 16 converters: Triton IR → Linalg/Tensor/MemRef |
| `third_party/vulkan/lib/Conversion/TritonToLinalgPass.cpp` | ~210 | Pass wrapper, TritonTypeConverter, illegal ops |
| `third_party/vulkan/lib/Conversion/PrepareSPIRV.cpp` | ~380 | SPIR-V prep passes (Phase 1.5, see spirv-setup skill) |
| `third_party/vulkan/backend/compiler.py` | ~275 | Python pipeline orchestration |
| `third_party/vulkan/backend/emitter.py` | ~580 | OpenCL C emitter (Python, regex-based) |
| `third_party/vulkan/test/test_kernels.py` | ~280 | 12-kernel GPU test suite |
| `third_party/vulkan/test/*.ttir` | ~30 each | Hand-written TTIR test inputs |
| `third_party/vulkan/triton_vulkan.cc` | ~140 | pybind11 pass registration |

### Build & Sync Commands

```powershell
# Activate MSVC (required before cmake)
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul 2>&1 && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }

# Rebuild after C++ changes
cd build/cmake.win-amd64-cpython-3.14
cmake --build . --target triton

# Sync Python files after editing backend/*.py
Copy-Item third_party/vulkan/backend/*.py python/triton/backends/vulkan/ -Force

# Run test suite
$env:TRITON_BACKENDS_IN_TREE="1"
python third_party/vulkan/test/test_kernels.py
```

**CRITICAL:** Only rebuild C++ (`cmake --build`) when you change `.cpp`/`.h`
files. Python-only changes just need the `Copy-Item` sync.

---

## Converter Reference

### Type System

The `TritonTypeConverter` in `TritonToLinalgPass.cpp` maps:

| Triton Type | Converted Type | Notes |
|-------------|---------------|-------|
| `!tt.ptr<T>` | `memref<*xT>` (UnrankedMemRefType) | Scalar pointer → unranked memref |
| `tensor<NxM x !tt.ptr<T>>` | `memref<NxMxT>` (RankedMemRefType) | Pointer tensor → ranked memref with same shape |
| `tensor<NxT>` | `memref<NxT>` | Data tensor → ranked memref (via TypeConverter; only affects ops registered with typeConverter. Most data tensors pass through unchanged and are bufferized later by one-shot-bufferize) |
| Everything else | Pass-through | i32, f32, index, etc. unchanged |

**TRAP: Pattern registration with/without TypeConverter**

Registering patterns with `patterns.add<Foo>(typeConverter, ctx)` changes how
the adaptor resolves operands — the framework will auto-convert operand types
through the TypeConverter. For most of our converters, we do NOT want this
because they manually handle the original Triton types internally (e.g.,
`SplatConverter` checks `isa<triton::PointerType>(op.getSrc().getType())`
on the ORIGINAL op, not the adapted type).

**Rule:** Register with `patterns.add<Foo>(ctx)` (no typeConverter) for all
converters except `AtomicRMWConverter`. The atomic converter needs
typeConverter because its operands come from other already-converted ops.

Even with typeConverter, `AtomicRMWConverter` currently has a known issue
where the MLIR dialect conversion framework doesn't dispatch to it. Only
`ConvertAnyElementwiseMappableOpOnRankedTensors` is tried for
`tt.atomic_rmw`. Root cause is likely the `Optional<mask>` operand +
`MemoryEffectOpInterface` traits on `AtomicRMWOp`. Fix requires a
pre-processing pass to decompose atomics before conversion.

### Converter Catalog

#### Tensor Manipulation (6 converters)

| Converter | Triton Op | Lowering |
|-----------|-----------|----------|
| `SplatConverter` | `tt.splat` | Pointer: pass through base memref. Scalar: `tensor.empty` + `linalg.fill` |
| `MakeRangeConverter` | `tt.make_range` | `linalg.generic` with `linalg.index(0)` + optional offset |
| `BroadcastConverter` | `tt.broadcast` | `linalg.generic` with broadcast affine map (dim=0 for broadcast dims) |
| `ExpandDimsConverter` | `tt.expand_dims` | `tensor.expand_shape` with reassociation indices |
| `TransposeConverter` | `tt.trans` | `linalg.transpose` with order permutation |
| `ReshapeConverter` | `tt.reshape` | `tensor.expand_shape` or `tensor.collapse_shape`; fallback to `tensor.reshape` |

#### Program Info (2 converters)

| Converter | Triton Op | Lowering |
|-----------|-----------|----------|
| `GetProgramIDConverter` | `tt.get_program_id` | Reads from injected function arg (last 3 i32 args = pid x,y,z) |
| `GetNumProgramsConverter` | `tt.get_num_programs` | Reads from injected function arg (args[-6:-3] = num_programs x,y,z) |

The pass injects 6 extra i32 args to every `tt.func` via `addProgramInfo()`.
The OpenCL emitter maps these to `__kernel` args which the host sets at launch.

#### Compute (2 converters)

| Converter | Triton Op | Lowering |
|-----------|-----------|----------|
| `MatmulConverter` | `tt.dot` | `linalg.matmul` with zero-init + optional accumulation via `arith.addf` |
| `ReduceConverter` | `tt.reduce` | `linalg.reduce` with cloned combiner region |

#### Memory (3 converters)

| Converter | Triton Op | Lowering |
|-----------|-----------|----------|
| `AddPtrConverter` | `tt.addptr` | Walks def-chain via `visitOperand()` → `memref.reinterpret_cast` |
| `LoadConverter` | `tt.load` | Scalar: `memref.cast` unranked→ranked + `memref.load`. Tensor: `alloc` + `fill(zero)` + `memref.copy` + `bufferization.to_tensor` |
| `StoreConverter` | `tt.store` | Scalar: `memref.cast` + `memref.store`. Tensor: `bufferization.materialize_in_destination` |

#### Other (3 converters)

| Converter | Triton Op | Lowering |
|-----------|-----------|----------|
| `BitcastConverter` | `tt.bitcast` | `arith.bitcast` |
| `DenseConstantConverter` | dense splat constants | `tensor.empty` + `linalg.fill` |
| `AtomicRMWConverter` | `tt.atomic_rmw` | Sequential `scf.for` with memref load-modify-store. Requires `typeConverter` registration + tensor↔memref materializations |

---

## Key Patterns & Workarounds

### PATTERN 1: The PtrState / AddPtr Chain

The most complex converter. Triton IR builds pointer tensors via chains like:
```
%base = tt.splat %ptr          // broadcast scalar ptr to tensor<256x!tt.ptr<f32>>
%range = tt.make_range 0..256  // tensor<256xi32>
%ptrs = tt.addptr %base, %range  // tensor<256x!tt.ptr<f32>>
%data = tt.load %ptrs
```

`AddPtrConverter` walks the def-chain backward via `visitOperand()` to extract:
- **source**: the base memref (from SplatConverter)
- **offset**: constant or SSA offset value
- **sizes/strides**: from the shape of the pointer tensor

Then creates: `memref.reinterpret_cast %base offset:[off] sizes:[N] strides:[1]`

**Dynamic offsets (pid-dependent):** When the offset chain involves
`tt.get_program_id` (e.g., `pid * BLOCK_SIZE + range`), the offset must be
dynamic. The key is `visitOperand.SplatOp`: when splatting an integer scalar
(not a pointer), the scalar value is propagated as a dynamic offset via
`arith.index_cast`. This enables multi-block execution where each block
accesses a different slice of the buffer.

**TRAP:** Before this fix, `SplatOp` for integer scalars set `offsets=[0]`
(discarding the runtime value), causing all blocks to read from offset 0.
The fix checks `isIntSplat && srcState.scalar` and uses the scalar as the
base offset. This also enables 2D pointer patterns with `expand_dims +
broadcast + addptr`.

**2D pointer patterns:** `visitOperand` handles `ExpandDimsOp` (inserts a
size-1 dimension with stride 0) and `BroadcastOp` (updates sizes to match
broadcast shape). With the SplatOp dynamic offset fix, chains like
`splat(N) → broadcast → muli(rows, N) → addi(cols)` produce correct
multi-dimensional reinterpret_cast ops.

### PATTERN 2: Scalar Load/Store with Unranked MemRef

**Problem:** `tt.store %scalar_ptr, %value` where `%scalar_ptr` is
`!tt.ptr<f32>` converts the ptr to `memref<*xf32>` (unranked). But
`memref.load/store` requires ranked memref.

**Solution in LoadConverter/StoreConverter:**
```cpp
// Cast unranked to ranked memref<1xT>
auto rankedTy = MemRefType::get({1}, elemType);
auto ranked = rewriter.create<memref::CastOp>(loc, rankedTy, ptr);
auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
auto val = rewriter.create<memref::LoadOp>(loc, ranked, ValueRange{zero});
```

### PATTERN 3: ReduceConverter 0-d Tensor

**Problem:** 1D tensor → scalar reduction (e.g., `tensor<256xf32> → f32`)
requires `linalg.reduce` which works on tensors, not scalars. The result is
a 0-d tensor (`tensor<f32>`), but Triton expects bare `f32`.

**Solution:**
1. Create 0-d init tensor: `tensor.empty<f32>` → `linalg.fill` with identity
2. Run `linalg.reduce` producing `tensor<f32>`
3. Extract scalar: `tensor.extract` (no indices = 0-d extract)

**Identity values by combiner op:**

| Combiner | Float Init | Int Init |
|----------|-----------|----------|
| `arith.addf/addi` | 0.0 | 0 |
| `arith.mulf` | 1.0 | — |
| `arith.muli` | — | 1 |
| `arith.maximumf` | -∞ (`APFloat::getInf(sem, true)`) | `APInt::getSignedMinValue` |
| `arith.minimumf` | +∞ | `APInt::getSignedMaxValue` |
| `arith.andi` | all-ones | — |
| `arith.ori/xori` | 0 | — |

**TRAP:** The `getInitValue` method returns `Value` via `TypeSwitch`. If a new
combiner op is encountered, the `Default` case returns zero — this is wrong for
max/min reductions. The default should be `llvm_unreachable` for safety, but
we keep zero as a fallback since linalg's elementwise-to-linalg conversion can
produce unexpected combiner ops.

### PATTERN 4: MatmulConverter

`tt.dot %A, %B, %C` (fused multiply-add) becomes:
```
%init = arith.constant dense<0.0> → tensor.empty + linalg.fill
%result = linalg.matmul ins(%A, %B) outs(%init)
%final = arith.addf %result, %C   // if C is non-zero
```

The matmul TTIR uses `tt.reshape` to go from flat 1D loads to 2D:
```mlir
%x_flat = tt.load %xp : tensor<256x!tt.ptr<f32>>
%x = tt.reshape %x_flat : tensor<256xf32> -> tensor<16x16xf32>
```
This avoids the 2D pointer pattern limitation (Pattern 1).

### PATTERN 5: DenseConstantConverter

**Problem:** Triton's constant folder sometimes produces dense splat constants:
```mlir
%c = arith.constant dense<0.0> : tensor<256xf32>
```
These are `RankedTensorType` constants that `arith-to-spirv` and the emitter
can't handle directly.

**Solution:** A dynamically-legal check marks splat `arith.constant` ops as
illegal only when they produce `RankedTensorType` with `DenseElementsAttr`.
`DenseConstantConverter` replaces them with `tensor.empty` + `linalg.fill`.

---

## OpenCL Emitter Architecture

The emitter in `emitter.py` is a line-by-line regex-based MLIR→OpenCL C
transpiler. It processes the lowered MemRef-level IR (after `make_memref`),
NOT the original TTIR.

### Design Decisions

1. **Regex, not MLIR Python bindings** — The MLIR Python API doesn't expose
   easy line-by-line walking of lowered IR. Regex on `str(module)` is fragile
   but works for the limited op set we produce.

2. **Flat arrays, not multi-d C arrays** — All memref allocations are flattened
   to 1D: `memref<16x16xf32>` → `float buf[256]`. Multi-d indices are linearized
   at load/store time.

3. **SSA tracking** — `ssa_map` maps MLIR SSA values (`%0`, `%arg0`) to C
   variable names. `ssa_types` tracks their C types. `_def_var` generates fresh
   variable names.

4. **Block args for control flow** — `cf.br` and `cf.cond_br` pass block
   arguments. The emitter tracks these via `block_arg_map` and generates
   variable assignments before gotos.

### Key Emitter Methods

| Method | Purpose | Traps |
|--------|---------|-------|
| `_emit_constant` | `arith.constant` → C literal | Must decode IEEE 754 hex (`0xFF800000` → `-INFINITY`) |
| `_emit_alloc` | `memref.alloc/alloca` → `__private float buf[N]` | Must handle `{alignment=64:i64}` attributes |
| `_emit_store` | `memref.store` → `buf[idx] = val` | Multi-d linearization needed |
| `_emit_load` | `memref.load` → `float v = buf[idx]` | Multi-d linearization needed |
| `_emit_copy` | `memref.copy` → `for` loop | Must compute total size for N-d |
| `_emit_reshape_alias` | `memref.expand/collapse_shape` → no-op alias | Just aliases SSA name |
| `_emit_reinterpret_cast` | `memref.reinterpret_cast` → pointer arithmetic | `ptr + offset` |
| `_emit_memref_cast` | `memref.cast` → no-op alias | Identity in flat OpenCL |
| `_linearize_nd_index` | N-d index → flat 1D index | Computes `i*s0 + j*s1 + k` for arbitrary dims |

### Emitter Traps

**TRAP E1: IEEE 754 Hex Floats**

MLIR emits `-infinity` as `0xFF800000` (raw IEEE 754 bits). OpenCL C interprets
hex literals as integers, so `float c = 0xFF800000` compiles but gives a huge
integer value, not -inf.

**Fix:** Decode hex via `struct.unpack`:
```python
import struct, math
bits = int(val, 16) & 0xFFFFFFFF
fval = struct.unpack('f', struct.pack('I', bits))[0]
if math.isinf(fval) and fval < 0:
    val = "-INFINITY"
```

**TRAP E2: Multi-d Memref Indexing**

After `linalg-to-loops`, `memref.load %buf[%i, %j]` appears for 2D memrefs.
Since we flatten all allocations to 1D, we must linearize:
`buf[i * d1 + j]` for 2D, `buf[i * d1 * d2 + j * d2 + k]` for 3D, etc.

The `_linearize_nd_index()` static method handles arbitrary N-d:
```python
@staticmethod
def _linearize_nd_index(line, idx_parts, map_val_fn):
    all_dims = re.findall(r'memref<([\dx]+)x\w+', line)
    if all_dims:
        dim_strs = [d for d in all_dims[0].split('x') if d.isdigit()]
        if len(dim_strs) == len(idx_parts):
            dims = [int(d) for d in dim_strs]
            terms = []
            for k, idx in enumerate(idx_parts):
                stride = 1
                for d in dims[k + 1:]:
                    stride *= d
                mapped = map_val_fn(idx)
                terms.append(f"{mapped} * {stride}" if stride > 1 else mapped)
            return " + ".join(terms)
    return None
```

**TRAP E3: `memref.expand_shape` / `memref.collapse_shape`**

These ops are view reshapes (e.g., `memref<256xf32>` → `memref<16x16xf32>`).
In flat OpenCL, they're no-ops — the underlying buffer doesn't change.
The emitter aliases the output SSA name to the input:
```python
self.ssa_map[dst] = self._map_val(src)
```

But subsequent `memref.load %expand_shape[%i, %j]` uses 2D indices on the
aliased 1D buffer — that's where `_linearize_nd_index` (Trap E2) kicks in.

**TRAP E4: `memref.alloca` with `{alignment=N:i64}`**

MLIR sometimes emits: `memref.alloca() {alignment = 64 : i64} : memref<f32>`
The regex must handle the optional attribute group:
```python
r"(%[\w]+)\s*=\s*memref\.alloca?\(\)(?:\s*\{[^}]*\})?\s*:\s*memref<([^>]+)>"
```

**TRAP E5: 0-d Memref**

Scalar reduction results produce `memref<f32>` (no dimensions). The emitter
must handle:
- `memref.alloca() : memref<f32>` → `float buf[1]`
- `memref.store %v, %buf[]` → `buf[0] = v` (empty index list)
- `memref.load %buf[]` → `v = buf[0]`

---

## Test Suite Design

### TTIR Test Files

Each test is a hand-written `.ttir` file in `third_party/vulkan/test/`.
They use fixed block sizes (N=256 or 16×16) to avoid masking complexity.

| File | Pattern | Key Ops |
|------|---------|---------|
| `test_vector_add.ttir` | 1D elementwise | load, addptr, addf, store |
| `test_elementwise_mul.ttir` | 1D elementwise | mulf |
| `test_fma.ttir` | Ternary elementwise | mulf + addf, 3 input buffers |
| `test_gelu.ttir` | Activation | negf, math.exp, divf |
| `test_swiglu.ttir` | Gated activation | negf, math.exp, mulf, divf, dense constant |
| `test_reduce_sum.ttir` | 1D→scalar reduction | tt.reduce(addf), scalar store |
| `test_reduce_max.ttir` | 1D→scalar reduction | tt.reduce(maximumf), scalar store |
| `test_softmax.ttir` | Compound: 2× reduce + broadcast | tt.reduce(maximumf), tt.reduce(addf), splat, math.exp, divf |
| `test_matmul_simple.ttir` | Matrix multiply | tt.reshape (1D→2D), tt.dot, dense constant |
| `test_broadcast_add.ttir` | Dual-load + addition | Two separate load chains + addf |
| `test_transpose.ttir` | Reshape + transpose | tt.reshape, tt.trans |

### Test Harness Pattern

```python
def compile_ttir(ttir_path):
    c = ir.context(); ir.load_dialects(c); vulkan.load_dialects(c)
    m = ir.parse_mlir_module(ttir_path, c)
    m.context = c  # REQUIRED — dynamic attribute for pass_manager
    b = VulkanBackend(GPUTarget("vulkan", 0, 32))
    o = b.parse_options({}); md = {}
    m = b.make_ttir(m, md, o)
    m = b.make_linalg(m, md, o)
    m = b.make_memref(m, md, o)
    return b.make_opencl(m, md, o), md
```

**CRITICAL:** `m.context = c` must be set. The `ir.pass_manager` constructor
reads `mod.context` as a Python attribute. Without it, you get
`AttributeError: context`.

### Program Info Args

Every kernel gets 6 extra i32 args appended by `addProgramInfo()`:
`num_programs_x, num_programs_y, num_programs_z, program_id_x, program_id_y, program_id_z`

For single-block execution:
```python
args = [buf_a, buf_b, buf_out, np.int32(N)] + [np.int32(0)] * 6
```

For multi-block execution, use `run_kernel_multiblock()`:
```python
def run_kernel_multiblock(src, metadata, base_args, n_blocks):
    prog = cl.Program(ctx, src).build()
    kernel = getattr(prog, metadata["name"])
    for bid in range(n_blocks):
        for i, arg in enumerate(base_args):
            kernel.set_arg(i, arg)
        n = len(base_args)
        kernel.set_arg(n + 0, np.int32(n_blocks))  # num_programs_x
        kernel.set_arg(n + 1, np.int32(1))          # num_programs_y
        kernel.set_arg(n + 2, np.int32(1))          # num_programs_z
        kernel.set_arg(n + 3, np.int32(bid))        # program_id_x
        kernel.set_arg(n + 4, np.int32(0))          # program_id_y
        kernel.set_arg(n + 5, np.int32(0))          # program_id_z
        cl.enqueue_nd_range_kernel(queue, kernel, (1,), (1,))
    queue.finish()
```

### Performance Baseline

Measured on RTX 2080 Ti (Route B via Linalg, single-threaded OpenCL):

| Kernel | N | OpenCL µs | CUDA µs | Ratio | Notes |
|--------|---|-----------|---------|-------|-------|
| vector_add | 65536 | 47,234 | 19 | 2539× | 256 sequential blocks |
| reduce_sum | 256 | 277 | 26 | 10.6× | Single block, serial loop |
| softmax | 256 | 221 | 18 | 12.1× | Single block, 2 reductions |
| matmul_16×16 | 16 | 345 | 37 | 9.4× | Triple nested loop |

The 2539× gap for vector_add is expected: we dispatch 256 single-threaded
blocks sequentially. Single-block kernels are 10-12×: serial execution vs
CUDA's parallel threads. Route A (TTG→SPIR-V) would use Vulkan compute
workgroups to close this gap.

### Error Tolerances

| Kernel Type | Typical Error | Tolerance | Why |
|-------------|---------------|-----------|-----|
| Integer/exact ops | 0.00e+00 | 1e-6 | Bit-perfect |
| Single float op (add, mul) | 0 – 5e-7 | 1e-6 | Float32 ULP |
| Chained float ops (gelu, swiglu) | 1e-7 – 5e-7 | 1e-5 | Error accumulation |
| Reduction (sum, N=256) | 1e-5 – 2e-5 | 1e-3 | Accumulation over 256 elements |
| Matmul (16×16) | 5e-7 – 2e-6 | 1e-4 | 16 multiply-adds per element |
| Softmax | 1e-9 – 5e-9 | 1e-5 | Exp/div cancel errors |

---

## Known Limitations

### 1. No Loop-Carried Pointer State

`scf.for` loops that update pointer state across iterations (e.g., tiled
matmul with pointer increment per tile) are not supported. `visitOperand`
only traces static def-chains, not loop-carried values.

### 2. AtomicRMW Requires TypeConverter + Materializations

The `AtomicRMWConverter` must be registered with `typeConverter` (unlike
other converters) because its operands come from already-converted ops.
This means `adaptor.getVal()` returns `memref<NxT>` (not `tensor<NxT>`),
so the converter must use `memref.load/store` — NOT `tensor.extract/insert`.
The `TritonTypeConverter` needs `tensor↔memref` materializations
(`bufferization::ToBufferOp` and `bufferization::ToTensorOp`) to bridge
the type gap when non-atomic ops produce tensors consumed by the atomic.

### 3. Sequential Block Dispatch

Multi-block kernels are dispatched via host-side loop (one OpenCL enqueue per
block). This is ~2500× slower than CUDA for large N. Route A (Phase 3) would
use Vulkan compute workgroups for parallel dispatch.

### 4. Masked Operations Coverage

Four tests use masked loads: `test_vector_add.ttir`, `test_elementwise_mul.ttir`,
`test_gelu.ttir`, and `test_reduce_sum.ttir`. The remaining tests use
N = BLOCK_SIZE without masks. `test_vector_add.ttir` and
`test_elementwise_mul.ttir` also use masked stores.

---

## Debugging Recipes

### Recipe 1: Dump IR at Each Stage

```python
def comp_debug(ttir_path):
    c = ir.context(); ir.load_dialects(c); vulkan.load_dialects(c)
    m = ir.parse_mlir_module(ttir_path, c); m.context = c
    b = VulkanBackend(GPUTarget("vulkan", 0, 32))
    o = b.parse_options({}); md = {}
    m = b.make_ttir(m, md, o);   print("=== TTIR ===\n", str(m))
    m = b.make_linalg(m, md, o); print("=== LINALG ===\n", str(m))
    m = b.make_memref(m, md, o); print("=== MEMREF ===\n", str(m))
    s = b.make_opencl(m, md, o); print("=== OPENCL ===\n", s)
    return s, md
```

### Recipe 2: Check Why Conversion Fails

```powershell
# Run triton-opt with debug output to trace pattern matching
$env:MLIR_ENABLE_DUMP=""
& "python/triton/_C/triton-opt.exe" `
    "--triton-to-linalg" "--debug" "test.ttir" 2>&1 |
    Select-String "pattern|Trying|FAIL|SUCCESS"
```

Key lines to look for:
- `Trying to match "FooConverter"` — pattern was attempted
- `-> matchAndRewrite failed` — pattern matched but conversion failed
- `no matched legalization pattern` — NO pattern tried for this op
- `unrealized_conversion_cast` — type bridge left behind (usually harmless)

### Recipe 3: Debug OpenCL Compilation Errors

```python
import os; os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
# Now cl.Program(ctx, src).build() will print NVIDIA's compiler errors
```

Common OpenCL compilation errors:
- `undeclared identifier 'v_expand_shape'` → Missing `_emit_reshape_alias` handler
- `expected ';'` → Missing semicolon in emitted code; check `_emit_*` method
- `use of old-style cast` → Warning only, not fatal

### Recipe 4: Verify a New TTIR File Converts

```powershell
& "python/triton/_C/triton-opt.exe" "--triton-to-linalg" "test.ttir"
# Exit code 0 = success, non-zero = conversion failure
# stderr will show which op failed to legalize
```

---

## Adding a New Kernel (Step-by-Step)

1. **Write TTIR:** Create `third_party/vulkan/test/test_foo.ttir` using existing
   tests as templates. Both 1D flat addressing and 2D pointer patterns
   (expand_dims + broadcast + addptr) are supported.

2. **Test conversion:** Run through `triton-opt --triton-to-linalg test_foo.ttir`.
   If it fails, check if you need a new converter or if an existing op isn't
   marked illegal.

3. **Test emitter:** Run through the full Python pipeline and dump the OpenCL:
   ```python
   s, md = comp_debug("test_foo.ttir")
   ```
   Check for `// UNLOWERED:` comments in the output — these are ops the emitter
   doesn't handle.

4. **Add emitter handler:** If needed, add a new `_emit_foo` method and wire it
   into the `_emit_op` dispatcher. Follow existing patterns.

5. **Add GPU test:** Add `test_foo()` function in `test_kernels.py` following
   the pattern of existing tests. Add to the `TESTS` list at the bottom.

6. **Run test suite:** `python test_kernels.py` — all tests should pass (currently 14).

---

## Adding a New Converter (Step-by-Step)

1. **Define the converter struct** in `TritonToLinalg.cpp`:
   ```cpp
   struct FooConverter : public OpConversionPattern<triton::FooOp> {
     using OpConversionPattern::OpConversionPattern;
     LogicalResult matchAndRewrite(triton::FooOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const override {
       // ...
       rewriter.replaceOp(op, result);  // or rewriter.eraseOp(op) for stores
       return success();
     }
   };
   ```

2. **Register it** in `populateTritonToLinalgConversionPatterns()`:
   ```cpp
   patterns.add<FooConverter>(ctx);  // NOT typeConverter — see Pattern System trap
   ```

3. **Mark the op illegal** in `TritonToLinalgPass.cpp`:
   ```cpp
   target.addIllegalOp<..., triton::FooOp>();
   ```

4. **Rebuild:** `cmake --build . --target triton`

5. **Test:** `triton-opt --triton-to-linalg test.ttir`

**TRAP: `rewriter.eraseOp` vs `rewriter.replaceOp`**
- `replaceOp(op, results)` for ops with results (load, reduce, etc.)
- `eraseOp(op)` for ops without results (store, return)
- Using the wrong one causes "operation destroyed but still has uses" crash

---

## Lessons Learned (Phase 2 Retrospective)

1. **Start with the emitter, not SPIR-V.** OpenCL C output is human-readable
   and debuggable. SPIR-V binary is opaque. Getting the emitter working first
   gives you fast iteration on new kernel patterns.

2. **Test each converter individually.** Run `triton-opt --triton-to-linalg`
   on minimal TTIR before trying the full pipeline. The Linalg→MemRef→OpenCL
   stages add complexity you don't need when debugging a converter.

3. **The emitter will break on every new op.** MLIR's lowering produces ops
   you don't expect. Always dump the lowered IR (`make_memref` output) and
   check what ops appear before assuming the emitter handles them.

4. **Reduction identity values are non-trivial.** Getting `-INFINITY` for
   max-reduce requires `APFloat::getInf(semantics, /*Negative=*/true)` on
   the C++ side AND `struct.unpack` hex decoding on the emitter side.

5. **`memref.expand_shape`/`collapse_shape` are invisible traps.** They look
   like they should be lowered by bufferization, but they survive into the
   final IR as views. The emitter must alias them and then linearize multi-d
   loads/stores that reference the aliased buffer.

6. **Don't pass TypeConverter to all patterns.** It changes how operands are
   resolved in the adaptor and can break converters that inspect original
   Triton types. Only pass it when the converter genuinely needs adapted types.

7. **The MLIR dialect conversion framework has pattern-matching blind spots.**
   Not all `OpConversionPattern<FooOp>` implementations get dispatched. Ops
   with `Optional` operands or certain trait combinations may silently skip
   your pattern. Debug with `--debug` flag and check `Trying to match`.

8. **Sync Python files after every edit.** The backend code lives in
   `third_party/vulkan/backend/` but Python imports from
   `python/triton/backends/vulkan/`. Forgetting to copy causes stale behavior
   with no error message.

9. **`visitOperand.SplatOp` must propagate integer scalar values.** When
   `tt.splat` broadcasts a runtime integer (e.g., `pid * BLOCK_SIZE`), the
   scalar value must become the dynamic offset in PtrState — NOT zero. Without
   this, multi-block kernels silently read from offset 0 for all blocks, and
   2D pointer patterns (expand_dims + broadcast) lose their row/column offsets.
   This was the single most impactful bug fix in Phase 2.

10. **Use `--debug` to verify pattern dispatch.** When a converter appears to
    not work, the `--debug` flag shows exactly which patterns are tried. The
    AtomicRMW "not dispatched" claim turned out to be wrong — the pattern WAS
    dispatched but crashed because `adaptor.getVal()` returned `memref` (not
    `tensor`) due to TypeConverter registration. Always verify with evidence.
