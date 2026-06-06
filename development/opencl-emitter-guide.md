# Triton OpenCL Emitter Guide — Parallel Execution

> **Note:** The OpenCL emitter is an **optional debugging aid**. The primary
> backend path is SPIR-V → Vulkan dispatch (see `vulkan-backend-guide.md`
> and the `triton-windows-vulkan` / `triton-windows-vulkan-perf` skills).
> Use the OpenCL emitter when you need human-readable C output for debugging
> converter correctness.

**Scope:** The parallel OpenCL emitter that maps `linalg` ops directly
to OpenCL workitems, achieving 253× speedup over the serial emitter for
elementwise kernels.

**Audience:** Compiler engineers who want to understand the parallel execution
model, the tree reduction algorithm, barrier semantics, and how to extend the
emitter to new ops.

**Prerequisites:** Understanding of OpenCL work-groups and work-items,
`linalg.generic` semantics (iterator types, indexing maps, block arguments),
and the serial emitter architecture (see `vulkan-backend-guide.md` §Emitter).

---

## Table of Contents

1. [The Problem: Why Serial Is Slow](#1-the-problem-why-serial-is-slow)
2. [The Insight: Stop Before Loop Lowering](#2-the-insight-stop-before-loop-lowering)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [The ParallelOpenCLEmitter Class](#4-the-parallelopenclemitter-class)
5. [Op-by-Op Emission Guide](#5-op-by-op-emission-guide)
6. [Memory Model: Global, Local, Private](#6-memory-model-global-local-private)
7. [Barrier Correctness](#7-barrier-correctness)
8. [Tree Reduction: The Algorithm](#8-tree-reduction-the-algorithm)
9. [Type Detection](#9-type-detection)
10. [Test Suite and Verification](#10-test-suite-and-verification)
11. [Performance Analysis](#11-performance-analysis)
12. [Traps Encountered](#12-traps-encountered)
13. [Extension Guide: Adding New Ops](#13-extension-guide-adding-new-ops)
14. [Known Limitations](#14-known-limitations)
15. [Appendix: File Inventory](#15-appendix-file-inventory)

---

## 1. The Problem: Why Serial Is Slow

The serial emitter (`emitter.py`) produces OpenCL kernels where a
**single workitem** executes the entire block — all 256 elements processed
sequentially by one thread:

```
Serial pipeline:
  TTIR → linalg → memref (bufferize + linalg-to-loops + lower-affine + scf-to-cf) → emitter.py

Resulting OpenCL:
  __kernel void vector_add(args...) {
    for (int i = 0; i < 256; i++)     // ONE workitem, sequential loop
      out[i] = x[i] + y[i];
  }

Dispatch:
  cl.enqueue_nd_range_kernel(queue, kern, (1,), (1,))   // 1 workitem
```

For multi-block kernels (e.g., 65536 elements = 256 blocks), the host dispatches
each block sequentially in a Python loop. This means 65536 elements are processed
by 256 serial dispatches, each running 256 elements on one thread.

The result: **48,683 µs** for `vector_add` at N=65536 — dominated by
256 sequential kernel launches with zero GPU parallelism.

---

## 2. The Insight: Stop Before Loop Lowering

The serial pipeline applies four passes after bufferization:

```
one_shot_bufferize → convert_linalg_to_loops → lower_affine → convert_scf_to_cf
```

The key observation: **`convert_linalg_to_loops` destroys parallelism**.
It lowers `linalg.generic {iterator_types = ["parallel"]}` into `scf.for`
loops — sequential iteration that must run on one thread. Once the structured
`linalg` ops are gone, there is no way to recover the parallelism.

The parallel emitter's insight is to **stop after bufferization**:

```
one_shot_bufferize → canonicalize → cse → (done)
```

This preserves `linalg.generic`, `linalg.reduce`, `linalg.matmul`, and
`linalg.transpose` as structured ops. Each op carries its iterator types
and indexing maps, which the emitter can read directly to decide how to
parallelize.

### What Bufferization Does (and Doesn't Do)

After `one_shot_bufferize`, the IR has:

- **`tensor<256xf32>` → `memref<256xf32>`**: All tensors are now buffers
- **`linalg.generic` preserved**: Operands changed from `tensor` to `memref`
  but the op structure (block args, yield, iterator types) is unchanged
- **`linalg.reduce` preserved**: Similarly, operands are `memref` but the
  reduction body is intact
- **`linalg.matmul` preserved**: Named op, not lowered to `linalg.generic`
- **`memref.alloc`/`memref.copy` inserted**: Bufferization creates
  temporaries and inserts copies to maintain SSA semantics

What bufferization does NOT do:
- Does NOT create `scf.for` loops (that's `convert_linalg_to_loops`)
- Does NOT create `cf.br` branches (that's `convert_scf_to_cf`)
- Does NOT create `affine.for` (that's an alternative lowering path)

### Typical IR After `make_memref_bufonly`

```mlir
func.func @vector_add_kernel(
    %arg0: memref<*xf32>, %arg1: memref<*xf32>,
    %arg2: memref<*xf32>, %arg3: i32, ...) {
  %recast_x = memref.reinterpret_cast %arg0 to offset:[0], sizes:[256], strides:[1]
      : memref<*xf32> to memref<256xf32, strided<[1]>>
  %recast_y = memref.reinterpret_cast %arg1 to offset:[0], sizes:[256], strides:[1]
      : memref<*xf32> to memref<256xf32, strided<[1]>>
  %alloc = memref.alloc() : memref<256xf32>
  memref.copy %recast_x, %alloc : memref<256xf32, ...> to memref<256xf32>
  %alloc_0 = memref.alloc() : memref<256xf32>
  memref.copy %recast_y, %alloc_0 : memref<256xf32, ...> to memref<256xf32>
  %alloc_1 = memref.alloc() : memref<256xf32>
  linalg.generic {indexing_maps = [#map, #map, #map],
                  iterator_types = ["parallel"]}
    ins(%alloc, %alloc_0 : memref<256xf32>, memref<256xf32>)
    outs(%alloc_1 : memref<256xf32>) {
  ^bb0(%in: f32, %in2: f32, %out: f32):
    %0 = arith.addf %in, %in2 : f32
    linalg.yield %0 : f32
  }
  %recast_out = memref.reinterpret_cast %arg2 to offset:[0], sizes:[256], strides:[1]
      : memref<*xf32> to memref<256xf32, strided<[1]>>
  memref.copy %alloc_1, %recast_out : memref<256xf32> to memref<256xf32, ...>
  return
}
```

The pattern is clear:
1. `memref.reinterpret_cast` — compute pointer + offset into global buffer
2. `memref.alloc` + `memref.copy` — load from global into local buffer
3. `linalg.generic` — compute (each element independently)
4. `memref.copy` — store from local buffer back to global

The parallel emitter maps each step to a parallel workitem operation.

---

## 3. Pipeline Architecture

### Stage Comparison

```
Serial pipeline:
  ┌──────────┐   ┌──────────┐   ┌──────────────────────────────┐   ┌────────────┐
  │ make_ttir│ → │make_linalg│ → │make_memref (buf+loops+cf)    │ → │make_opencl │
  └──────────┘   └──────────┘   └──────────────────────────────┘   └────────────┘
                                  one_shot_bufferize                 emitter.py
                                  convert_linalg_to_loops            (serial)
                                  lower_affine
                                  convert_scf_to_cf

Parallel pipeline:
  ┌──────────┐   ┌──────────┐   ┌──────────────────────────────┐   ┌─────────────────┐
  │ make_ttir│ → │make_linalg│ → │make_memref_bufonly (buf only)│ → │make_opencl_par  │
  └──────────┘   └──────────┘   └──────────────────────────────┘   └─────────────────┘
                                  one_shot_bufferize                 emitter_parallel.py
                                  canonicalize                       (parallel)
                                  cse
```

The first two stages are identical. The difference is in stages 3 and 4.

### `make_memref_bufonly` (compiler.py, lines 129–142)

```python
@staticmethod
def make_memref_bufonly(mod, metadata, opt):
    """Bufferize only — keep linalg.generic for parallel emission."""
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    vulkan.passes.memref.one_shot_bufferize(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    pm.run(mod, 'make_memref_bufonly')
    return mod
```

This runs exactly one pass (`one_shot_bufferize`) followed by cleanup.
The three passes that destroy parallelism (`convert_linalg_to_loops`,
`lower_affine`, `convert_scf_to_cf`) are deliberately omitted.

### `make_opencl_parallel` (compiler.py, lines 144–155)

```python
@staticmethod
def make_opencl_parallel(src, metadata, opt):
    """Emit parallel OpenCL C from bufferized IR with linalg.generic ops."""
    from triton.backends.vulkan.emitter_parallel import emit_opencl_parallel
    mlir_text = src.str_nodebug()
    opencl_src, block_size = emit_opencl_parallel(mlir_text)
    metadata['block_size'] = block_size
    return opencl_src
```

Two important details:
1. **`str_nodebug()`** — Produces MLIR without `loc(...)` annotations, which
   is critical for clean regex parsing (see Trap P6 in §12).
2. **`metadata['block_size']`** — The emitter detects the block size from
   the first `memref<Nxtype>` in the IR and returns it. The caller uses this
   to dispatch with `global_size = local_size = block_size`.

### Pipeline Registration

The parallel stages are NOT in `add_stages()` (line 44–49 of compiler.py).
Only the serial+SPIR-V pipeline is registered with Triton's framework:

```python
def add_stages(self, stages, options, language=None):
    stages["ttir"]  = lambda src, metadata: self.make_ttir(src, metadata, options)
    stages["linalg"] = lambda src, metadata: self.make_linalg(src, metadata, options)
    stages["memref"] = lambda src, metadata: self.make_memref(src, metadata, options)
    stages["spirv"]  = lambda src, metadata: self.make_spirv(src, metadata, options)
    stages["spv"]    = lambda src, metadata: self.make_spv(src, metadata, options)
```

The parallel stages are called manually in test scripts. This is deliberate —
The parallel emitter is a proof-of-concept that validates the approach without modifying
the framework's stage dispatch mechanism.

---

## 4. The ParallelOpenCLEmitter Class

**File:** `third_party/vulkan/backend/emitter_parallel.py` (~790 lines)

### Class Design

```python
class ParallelOpenCLEmitter:
    TYPE_MAP = {"f16": "half", "f32": "float", "f64": "double", ...}
    ARITH_OP_MAP = {"arith.addf": "+", "arith.mulf": "*", ...}
    MATH_FN_MAP = {"math.exp": "exp", "math.sqrt": "sqrt", ...}

    def __init__(self):
        self.lines: List[str] = []       # Output C lines
        self.indent = 0                   # Current indentation depth
        self.ssa_map: Dict[str, str] = {} # %ssa_name → C variable name
        self.ssa_types: Dict[str, str] = {} # C var name → C type string
        self.var_counter = 0              # For generating unique variable names
        self.block_size = 256             # Detected from IR
        self.needs_local_reduce = False   # Whether tree reduction is used
        self.local_arrays: List[str] = [] # __local declarations to insert
```

The emitter maintains an SSA map (`%name` → `varN`) and a type map
(`varN` → `"float"`, `"float*"`, etc.) that grow as ops are processed.
This is the same architecture as the serial emitter but with parallel-aware
op handlers.

### Emission Flow (the `emit()` method, lines 84–146)

```
1. Parse function signature → extract args (name, type, is_memref)
2. Detect block_size from first memref<NxT> in IR
3. Extract function body between matching braces
4. Map function args to C names (arg0, arg1, ...)
5. Emit kernel header (__kernel void name(...))
6. Emit `int _tid = get_local_id(0);`
7. Save line index for __local declarations (inserted later)
8. Walk body ops → dispatch to type-specific emitters
9. Emit closing brace
10. Insert __local declarations at saved position
11. Return (source, block_size)
```

The deferred `__local` insertion (step 10) is critical. OpenCL requires
`__local` variables at function scope, but the emitter discovers the need
for `__local` arrays while walking the body. By saving the insertion point
and adding declarations after the walk, the emitter produces valid OpenCL
regardless of op ordering.

### SSA Tracking

Three helper methods manage the SSA map:

```python
def _fresh(self, prefix: str) -> str:
    """Generate a unique C variable name like 'ptr_3', 'f_7', 'sh_12'."""
    self.var_counter += 1
    return f"{prefix}_{self.var_counter}"

def _def(self, ssa: str, prefix: str, ctype: str) -> str:
    """Define a new SSA value: maps %name to a fresh C variable."""
    name = self._fresh(prefix)
    self.ssa_map[ssa.strip()] = name
    self.ssa_types[name] = ctype
    return name

def _map_val(self, ssa: str) -> str:
    """Look up a value: returns the C name or a fallback v_ name."""
    ssa = ssa.strip().rstrip(",:")
    return self.ssa_map.get(ssa, ssa.replace("%", "v_"))
```

Prefixes encode the role: `ptr_` for pointers, `sh_` for shared arrays,
`f_` for float results, `i_` for integer results, `c_` for constants,
`ld_` for loads, `m_` for math function results, `loc_` for local scalars.

### Body Dispatcher (`_emit_body`, lines 197–287)

The body dispatcher walks MLIR lines and routes each to a handler:

| MLIR Op | Handler | Parallelization |
|---------|---------|----------------|
| `memref.reinterpret_cast` | `_emit_reinterpret_cast` | Pointer arithmetic (all workitems) |
| `memref.alloc/alloca` | `_emit_alloc_skip` | `__local` array or private scalar |
| `memref.copy` | `_emit_parallel_copy` | Per-workitem element copy + barrier |
| `memref.dealloc` | (skip) | No-op |
| `linalg.fill` | `_emit_linalg_fill` | Per-workitem element write |
| `linalg.generic` | `_emit_linalg_generic` | One workitem per element |
| `linalg.reduce` | `_emit_linalg_reduce` | Tree reduction with `__local` |
| `linalg.matmul` | `_emit_linalg_matmul` | One workitem per output element |
| `linalg.transpose` | `_emit_linalg_transpose` | One workitem per element (index swap) |
| `memref.expand/collapse_shape` | `_emit_reshape_alias` | SSA alias (same buffer) |
| `memref.load` | `_emit_memref_load` | Scalar read |
| `memref.cast` | `_emit_memref_cast` | SSA alias + type update |
| `affine.store` | `_emit_affine_store` | Guarded scalar write (tid==0) |
| `arith.constant` | `_emit_constant` | Per-workitem constant |
| `arith.index_cast` | `_emit_index_cast` | Per-workitem cast |
| `arith.*` (binary) | `_emit_arith_binary` | Per-workitem arithmetic |
| (anything else) | catch-all | `// UNHANDLED: ...` comment |

For multi-line ops (`linalg.generic`, `linalg.reduce`), the dispatcher uses
a two-phase brace collector: first it scans forward until the opening `{` is
found (MLIR may put it on the next line), then it collects until the matching
closing `}`. This handles MLIR's variable brace placement (see Trap P4 in §12).

---

## 5. Op-by-Op Emission Guide

### 5.1. `memref.reinterpret_cast` → Pointer with Offset

**MLIR input:**
```mlir
%ptr_1 = memref.reinterpret_cast %arg0 to offset:[%off], sizes:[256], strides:[1]
    : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
```

**Generated OpenCL:**
```c
__global float* ptr_1 = arg0 + ic_2;   // when offset is dynamic
__global float* ptr_1 = arg0;           // when offset is 0
```

The handler extracts the source SSA, offset value, and element type from the
memref. Zero offsets produce a simple alias; dynamic offsets emit pointer
arithmetic. The element type is extracted via regex from the `memref<...xTYPE>`
pattern.

### 5.2. `memref.alloc` → `__local` Array or Private Scalar

**Decision rule (lines 311–343):**

| MLIR Type | OpenCL Declaration | Scope |
|-----------|-------------------|-------|
| `memref<f32>` (0-d) | `float loc_N;` | Private (one per workitem) |
| `memref<256xf32>` (1-d) | `__local float sh_N[256];` | Shared (one per workgroup) |
| `memref<16x16xf32>` (2-d) | `__local float sh_N[256];` | Shared, flattened (16×16=256) |

**Why `__local` for arrays:** In the parallel model, `linalg.generic` writes
to `outs(%alloc)` where each workitem writes position `[_tid]`. Subsequent
ops read from the same `%alloc` at `[_tid]`. For cross-workitem ops
(matmul, transpose, reduce), workitem N reads positions written by OTHER
workitems. This requires `__local` visibility — `__private` arrays are
invisible across workitems.

**Why private for scalars:** `memref<f32>` represents a scalar result
(e.g., the output of a reduction). Each workitem needs its own copy of the
scalar for subsequent elementwise ops. A `float` declaration is per-workitem.

Multi-dimensional arrays are flattened: `memref<16x16xf32>` → `__local float[256]`.
All `__local` declarations are collected in `self.local_arrays` and inserted
at function scope after the body walk (see §4).

### 5.3. `memref.copy` → Per-Workitem Element Copy

**Generated OpenCL (lines 345–369):**
```c
// Array → array copy
sh_dst[_tid] = ptr_src[_tid];
barrier(CLK_LOCAL_MEM_FENCE);

// Array → scalar (not common)
loc_dst = ptr_src[_tid];

// Scalar → array (reduction output → global)
if (_tid == 0) ptr_dst[0] = loc_src;

// Scalar → scalar
loc_dst = loc_src;
```

All array-to-array copies are per-workitem element copies followed by a
barrier. This ensures all workitems have completed their copy before any
subsequent operation reads from the destination.

**Why not alias?** An earlier design aliased `memref.copy src, dst` by
mapping `dst → src` in the SSA map (no actual copy). This seemed efficient
but is **catastrophically wrong** when `linalg.generic outs(dst)` writes
back to `dst` — it would overwrite the original source data. Softmax was the
worst case: two reductions and three elementwise ops sharing "aliased"
buffers caused `inf`/`nan` results. See Trap P1 in §12.

### 5.4. `linalg.generic` → One Workitem Per Element

This is the core of the parallel emitter.

**MLIR input:**
```mlir
linalg.generic {indexing_maps = [#map, #map, #map],
                iterator_types = ["parallel"]}
  ins(%a, %b : memref<256xf32>, memref<256xf32>)
  outs(%c : memref<256xf32>) {
^bb0(%in: f32, %in2: f32, %out: f32):
  %0 = arith.mulf %in, %in2 : f32
  linalg.yield %0 : f32
}
```

**Generated OpenCL:**
```c
float v_1 = sh_a[_tid];    // block arg 0 ← ins[0][_tid]
float v_2 = sh_b[_tid];    // block arg 1 ← ins[1][_tid]
float v_3 = sh_c[_tid];    // block arg 2 ← outs[0][_tid]
float f_4 = v_1 * v_2;     // body: arith.mulf
sh_c[_tid] = f_4;          // yield → write to outs[0][_tid]
```

**How it works (lines 386–465):**

1. **Extract ins/outs operands**: Parse `ins(%a, %b : ...)` to get `[%a, %b]`.
   Parse `outs(%c : ...)` to get `[%c]`.

2. **Map block arguments**: `^bb0(%in: f32, %in2: f32, %out: f32)` maps to:
   - `%in` → `ins[0][_tid]` (read from first input array at workitem index)
   - `%in2` → `ins[1][_tid]` (read from second input)
   - `%out` → `outs[0][_tid]` (read current output value)

   If the operand is a pointer (`*` in type), read at `[_tid]`. If it's a
   scalar (private variable), read directly.

3. **Emit body ops**: Each op in the body is delegated to `_emit_body_op`
   (arith operations, math functions, casts, constants).

4. **Handle yield**: `linalg.yield %0` → write the yielded value to
   `outs[0][_tid]`. Only the first output is handled (multi-output generics
   are a known limitation).

### 5.5. `linalg.fill` → Per-Workitem Fill

```c
sh_alloc[_tid] = c_1;    // array fill: each workitem fills its element
loc_val = c_1;            // scalar fill: per-workitem value
```

No barrier after fill because the filled buffer is typically the `outs` of
a subsequent `linalg.generic` — each workitem only reads the element it
filled.

### 5.6. `linalg.matmul` → One Workitem Per Output Element

**MLIR input:**
```mlir
linalg.matmul ins(%a, %b : memref<16x16xf32>, memref<16x16xf32>)
              outs(%c : memref<16x16xf32>)
```

**Generated OpenCL (lines 620–659):**
```c
// Parallel matmul 16x16 @ 16x16
int _row = _tid / 16;      // M*N workitems, row = tid / N
int _col = _tid % 16;      // col = tid % N
float _sum = 0.0f;
for (int _k = 0; _k < 16; _k++) {
  _sum += sh_a[_row * 16 + _k] * sh_b[_k * 16 + _col];
}
sh_c[_row * 16 + _col] = _sum;   // = not += (see Trap P7)
barrier(CLK_LOCAL_MEM_FENCE);
```

**Key design choices:**

- **Workgroup size = M × N**: Each workitem computes one element of the
  output matrix. For 16×16, that's 256 workitems. The `self.block_size`
  is overridden to `M * N`.

- **Row-major indexing**: Both `A` and `B` are stored row-major (flattened
  from 2D memref via `memref.collapse_shape`). Each workitem iterates over
  the K dimension with a serial loop.

- **`=` not `+=`**: The `_sum` variable already accumulates the full dot
  product. Writing `C[idx] = _sum` is correct. Using `+=` would add to
  whatever was previously in the buffer (which could be garbage if
  `linalg.fill` was skipped or the buffer was recycled).

- **Barrier after write**: The output array is a `__local` shared buffer.
  The subsequent `memref.copy` reads from it at `[_tid]`, but the matmul
  wrote at `[_row * N + _col]` (which equals `_tid` by construction).
  The barrier is conservative but harmless.

### 5.7. `linalg.transpose` → One Workitem Per Element (Index Swap)

**Generated OpenCL (lines 661–685):**
```c
// Parallel transpose 16x16
int _row = _tid / 16;
int _col = _tid % 16;
sh_dst[_col * 16 + _row] = sh_src[_row * 16 + _col];   // swap indices
barrier(CLK_LOCAL_MEM_FENCE);
```

Each workitem reads from `[row * C + col]` and writes to `[col * R + row]`.
The barrier is essential here: workitem 0 writes to position `[0]`, workitem
1 writes to position `[16]`, etc. The subsequent `memref.copy` reads at
`[_tid]`, which may be a position written by a different workitem.

### 5.8. `memref.load` → Scalar Read

```python
# 0-d memref load (no indices)
float ld_1 = loc_buf;

# Single-index load
float ld_2 = ptr_buf[ic_3];

# Multi-index load (rare after bufferization)
float ld_3 = ptr_buf[v_i + v_j]; // multi-idx linearized
```

The handler (lines 698–724) splits indices on commas. Single-index loads
produce `buf[idx]`. Multi-index loads (which shouldn't appear after
bufferization since all arrays are flattened) generate a sum of the indices
as a conservative linearization.

### 5.9. `memref.expand_shape` / `memref.collapse_shape` → Alias

Both reshape ops are pure SSA aliases — the underlying buffer doesn't change.
The emitter maps `%dst → %src` in the SSA map and emits a comment.

### 5.10. `arith.constant` → C Constant with Hex Float Decoding

```python
# Normal constant
int c_1 = 256;
float c_2 = 0.5;

# IEEE 754 hex float (MLIR sometimes emits these)
float c_3 = -INFINITY;   // from 0xFF800000
float c_4 = INFINITY;    // from 0x7F800000
float c_5 = NAN;          // from 0x7FC00000
```

The hex float decoder (lines 751–770) unpacks 32-bit hex values via
`struct.unpack('f', struct.pack('I', bits))` and detects special values
(infinity, NaN). Normal hex floats are emitted as `%.8ef` formatted strings.

---

## 6. Memory Model: Global, Local, Private

### Three Address Spaces

| Address Space | OpenCL Keyword | Scope | Lifetime | Source in MLIR |
|--------------|---------------|-------|----------|---------------|
| Global | `__global float*` | All workgroups | Kernel | Function args (`memref<*xf32>`) |
| Local | `__local float[]` | One workgroup | Kernel | `memref.alloc() : memref<NxT>` |
| Private | `float` (no qualifier) | One workitem | Kernel | `memref.alloc() : memref<T>` |

### Memory Flow for a Typical Kernel

```
Global input → memref.copy → __local array → linalg.generic → __local array → memref.copy → Global output
    (args)                    (sh_N)           (compute)        (sh_N)                       (args)
```

Each step:
1. **Global → Local**: `sh_1[_tid] = ptr_x[_tid]; barrier();`
2. **Compute**: `sh_2[_tid] = f(sh_1[_tid]); // linalg.generic`
3. **Local → Global**: `ptr_out[_tid] = sh_2[_tid]; barrier();`

### Why Not Read Directly from Global?

In principle, `linalg.generic ins(ptr_x)` could read directly from the
global pointer. But bufferization inserts `memref.alloc + memref.copy`
to maintain SSA semantics — the original MLIR creates new tensors for
each operation result, and bufferization materializes these as separate
buffers with copies.

For performance-critical code, a future optimization could fold
global→local→compute into a single step, eliminating the `memref.copy`.

---

## 7. Barrier Correctness

### When Barriers Are Needed

A barrier (`barrier(CLK_LOCAL_MEM_FENCE)`) is needed when:

1. **Workitem N writes to `__local[X]` and workitem M reads from `__local[X]`
   where M ≠ N** — without a barrier, M might read before N has written.

2. **All workitems must see a consistent view of `__local` memory** — e.g.,
   after a tree reduction step, all workitems must see the updated partial
   sums before the next step.

### When Barriers Are NOT Needed

If every workitem only reads from `__local[_tid]` (the position it wrote),
no barrier is needed — each workitem reads its own data.

### Barrier Placement in the Emitter

| Operation | Barrier After? | Why |
|-----------|---------------|-----|
| `memref.copy` (ptr→ptr) | Yes | Cooperative copy — each workitem copies its element |
| `linalg.fill` (array) | No | Each workitem fills `[_tid]`, reads only `[_tid]` |
| `linalg.generic` | No | Each workitem computes and writes `[_tid]` |
| `linalg.reduce` (each step) | Yes | Tree reduction reads cross-workitem data |
| `linalg.matmul` | Yes | Output `[row*N+col]` may be read by other workitems |
| `linalg.transpose` | Yes | Output `[col*R+row]` read by other workitems |

### Conservative Policy

The emitter follows a **barrier-after-every-shared-write** policy. Some
barriers are technically redundant (e.g., after `memref.copy` when the
subsequent op only reads at `[_tid]`), but the cost is negligible (~20ns
per barrier on modern GPUs) and it eliminates the risk of subtle
race-condition bugs.

---

## 8. Tree Reduction: The Algorithm

### The Problem

`linalg.reduce` takes a 256-element array and produces a scalar.
With 256 workitems, how do you compute a single value from 256 inputs?

### The Algorithm (lines 541–618)

**Step 1: Load into `__local`**
```c
_shared[_tid] = sh_data[_tid];
barrier(CLK_LOCAL_MEM_FENCE);
```

**Step 2: Tree reduction**
```c
for (int _s = 128; _s > 0; _s >>= 1) {
  if (_tid < _s) {
    _shared[_tid] = _shared[_tid] + _shared[_tid + _s];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
```

Each iteration halves the active workitems:
- Iteration 1 (s=128): Workitems 0–127 add elements 128–255
- Iteration 2 (s=64): Workitems 0–63 add elements 64–127
- ...
- Iteration 8 (s=1): Workitem 0 adds element 1

After 8 iterations (log₂(256)), the result is in `_shared[0]`.

**Step 3: Broadcast result**
```c
barrier(CLK_LOCAL_MEM_FENCE);
loc_init = _shared[0];    // ALL workitems read
```

ALL workitems read the result (not just workitem 0). This is critical:
subsequent `linalg.generic` ops need the reduction result on every workitem.
For example, softmax's `exp(x - max) / sum(exp(x - max))` needs both
`max` and `sum` on all 256 workitems.

### Power-of-2 Requirement

The tree reduction assumes `block_size` is a power of 2. The assertion
at line 600–601 enforces this:

```python
assert self.block_size > 0 and (self.block_size & (self.block_size - 1)) == 0, \
    f"Tree reduction requires power-of-2 block size, got {self.block_size}"
```

For non-power-of-2 sizes, the halving stride `s >>= 1` would skip
elements beyond the nearest lower power of 2. Triton typically uses
powers of 2 (256, 512, 1024), so this is not a practical limitation.

### Reduction Op Detection

The emitter detects the reduction operation from the body:

| Body contains | red_op | OpenCL emission |
|---------------|--------|----------------|
| `arith.addf` | `+` | `_shared[tid] = _shared[tid] + _shared[tid+s]` |
| `arith.maximumf` | `fmax` | `_shared[tid] = fmax(_shared[tid], _shared[tid+s])` |
| `arith.minimumf` | `fmin` | `_shared[tid] = fmin(_shared[tid], _shared[tid+s])` |
| `arith.addi` | `+` | `_shared[tid] = _shared[tid] + _shared[tid+s]` |

### `__local` Type Tracking

The `__local float _shared[N]` declaration must match the reduction element
type. The emitter extracts this from the ins memref:

```python
tm = re.search(r'memref<\d+x(\w+)>', text)
reduce_elem = self.TYPE_MAP.get(tm.group(1), tm.group(1))
self.reduce_elem_type = reduce_elem
```

This type is used when inserting the `__local` declaration at function scope.

**Caveat:** `reduce_elem_type` is shared across all reductions in a kernel.
For softmax (which has both `reduce(max)` and `reduce(sum)`), both reductions
share the same `_shared` array. This works because they're sequential (not
concurrent) and both operate on `f32`. Mixed-type reduction chains (e.g.,
`f32` max + `i32` count) would require per-reduction `__local` arrays.

---

## 9. Type Detection

### The Problem

The serial emitter hardcodes `float` for all arithmetic operations. The
parallel emitter must support `half` (`f16`), `double` (`f64`), and
integer types. MLIR ops carry type annotations that must be extracted.

### Body Op Type Detection (lines 468–473)

Inside `linalg.generic` bodies, each op has a trailing type annotation:

```mlir
%0 = arith.mulf %in, %in2 : f32     // ← type is f32
%1 = arith.addf %0, %cst : f64      // ← type is f64
```

The emitter extracts this:

```python
type_hint = re.search(r':\s*(\w+)\s*$', line)
ftype = "float"
if type_hint:
    ftype = self.TYPE_MAP.get(type_hint.group(1), "float")
```

Then uses `ftype` for the variable declaration:

```python
cvar = self._def(dst, "f", ftype)
self._line(f"{ftype} {cvar} = {self._map_val(lhs)} {sym} {self._map_val(rhs)};")
```

This produces `double f_4 = v_1 * v_2;` for `f64` operations instead of
the incorrect `float f_4 = v_1 * v_2;`.

### Block Arg Type Detection

`linalg.generic` block arguments carry types too:

```mlir
^bb0(%in: f32, %in2: f64, %out: f32):
```

The emitter extracts these via:
```python
block_args = re.findall(r'(%\w+):\s*(\w+)', block_args_str)
```

And uses `self._map_type(ba_type)` for the variable declaration.

### Memref Element Type Extraction

Several handlers need to extract the element type from memref types.
The regex pattern varies by context:

```python
# reinterpret_cast, memref.cast: type is after last 'x' before '>'
tm = re.search(r'x?(\w+)>(?:\s|,|$)', line)

# memref.load: handle both memref<f32> and memref<256xf32>
tm = re.search(r'x(\w+)>|memref<(\w+)>', line)
raw = tm.group(1) or tm.group(2)

# linalg.reduce: memref<Nxtype>
tm = re.search(r'memref<\d+x(\w+)>', text)
```

The alternation pattern `x(\w+)>|memref<(\w+)>` is critical for handling
`memref<f32>` (0-d) — the greedy `[^>]*` in a single pattern would consume
`f3`, leaving only `2` for the type capture (see Trap P5 in §12).

---

## 10. Test Suite and Verification

### Test File: `test_kernels_parallel.py` (~198 lines)

The parallel test suite uses the same TTIR input files as the serial test
suite but compiles through the parallel pipeline:

```python
def compile_parallel(ttir_path):
    c = ir.context(); ir.load_dialects(c); vulkan.load_dialects(c)
    m = ir.parse_mlir_module(ttir_path, c); m.context = c
    b = VulkanBackend(GPUTarget("vulkan", 0, 32))
    o = b.parse_options({}); md = {}
    m = b.make_ttir(m, md, o)
    m = b.make_linalg(m, md, o)
    m = b.make_memref_bufonly(m, md, o)     # ← bufferize only
    return b.make_opencl_parallel(m, md, o), md  # ← parallel emitter
```

### Dispatch

```python
def run_par(src, md, args):
    prog = cl.Program(ctx, src).build()
    k = getattr(prog, md["name"])
    for i, a in enumerate(args):
        k.set_arg(i, a)
    bs = md["block_size"]
    cl.enqueue_nd_range_kernel(queue, k, (bs,), (bs,))  # N workitems
    queue.finish()
```

Key: `global_size = local_size = block_size`. This creates one workgroup
with exactly `block_size` workitems. Each workitem gets a unique `_tid`
from `get_local_id(0)`.

### 10 Test Kernels

| # | Kernel | Pattern | Tolerance | Key Validation |
|---|--------|---------|-----------|----------------|
| 1 | vector_add | `linalg.generic(parallel)` | 1e-6 | Basic elementwise |
| 2 | elementwise_mul | `linalg.generic(parallel)` | 1e-6 | Binary op |
| 3 | fma | 2× `linalg.generic` | 1e-5 | Multi-generic chain |
| 4 | gelu | `fill` + 4× `generic` | 1e-5 | Complex activation |
| 5 | swiglu | `fill` + `generic` chain | 1e-5 | Multi-input activation |
| 6 | reduce_sum | `linalg.reduce(addf)` | 1e-3 | Tree reduction (sum) |
| 7 | reduce_max | `linalg.reduce(maximumf)` | 1e-6 | Tree reduction (max) |
| 8 | softmax | 2× `reduce` + 3× `generic` | 1e-5 | Full parallel reduction pipeline |
| 9 | matmul_16×16 | `linalg.matmul` | 1e-4 | Per-output-element compute |
| 10 | transpose_16×16 | `linalg.transpose` | 1e-6 | Index-swap pattern |

### Softmax: The Ultimate Stress Test

Softmax is the most complex test: `out = exp(x - max(x)) / sum(exp(x - max(x)))`.
After bufferization, this becomes:

```
1. memref.copy x → alloc_x        (load input)
2. linalg.reduce(max) alloc_x     (tree reduction: max)
3. linalg.generic: alloc_x - max  (elementwise: subtract)
4. linalg.generic: exp(shifted)   (elementwise: exp)
5. linalg.reduce(sum) exp_vals    (tree reduction: sum)
6. linalg.generic: exp / sum      (elementwise: divide)
7. memref.copy alloc_out → global  (store output)
```

This exercises:
- Two tree reductions with different ops (max vs sum)
- Three elementwise generics between and after reductions
- Reduction result broadcast to all workitems (each workitem needs `max`
  and `sum` for the final `exp(x-max)/sum` computation)
- Multiple `__local` arrays with barriers between operations

If softmax passes with tolerance 1e-5, the barrier correctness, reduction
broadcast, and inter-op memory model are all verified.

### Running the Tests

```powershell
# Sync emitter to installed location
Copy-Item third_party/vulkan/backend/*.py python/triton/backends/vulkan/ -Force

# Run parallel tests
$env:TRITON_BACKENDS_IN_TREE = "1"
python third_party/vulkan/test/test_kernels_parallel.py

# Verify serial tests still pass (no regression)
python third_party/vulkan/test/test_kernels.py
```

---

## 11. Performance Analysis

### Benchmark Tool: `bench_kernels.py` (~275 lines)

The benchmark compares three execution modes:

1. **Serial**: `compile_serial` → `make_memref` → `make_opencl` → dispatch
   256 blocks sequentially, 1 workitem each
2. **Parallel**: `compile_parallel` → `make_memref_bufonly` → `make_opencl_parallel`
   → dispatch 1 workgroup with N workitems
3. **CUDA**: PyTorch operations on GPU (optional, requires torch)

### Vector Add Results (N=65536, RTX 2080 Ti)

| Mode | Time | Speedup |
|------|------|---------|
| Serial (256 blocks × 1 workitem) | 48,683 µs | 1× |
| Parallel (1 dispatch × 256 workitems) | 192 µs | **253×** |
| CUDA (PyTorch) | 19 µs | 2,539× vs serial |

### Why 253× Speedup

The serial mode dispatches 256 separate kernel invocations from Python,
each processing 256 elements with one workitem. The dominant cost is
**kernel launch overhead** (~190 µs × 256 launches ≈ 48,683 µs).

The parallel mode dispatches ONE kernel with 256 workitems processing
elements in parallel. The single launch overhead (~190 µs) is the
dominant cost, but there's only one launch.

The speedup is approximately `n_blocks × (launch_overhead / block_compute_time)`,
which for lightweight elementwise kernels approaches `n_blocks` (256 in this case).

### Why Still 10× Slower Than CUDA

1. **OpenCL compatibility layer**: NVIDIA's OpenCL implementation runs
   through a translation layer to CUDA PTX, adding overhead.
2. **Single workgroup**: The parallel emitter launches one workgroup of 256
   workitems. CUDA would launch 256 thread blocks with 256 threads each.
3. **No memory coalescing optimization**: The emitter doesn't analyze access
   patterns for optimal memory transaction sizing.
4. **Python dispatch overhead**: `pyopencl` kernel launch overhead is higher
   than CUDA's C-level driver API.

### Running the Benchmark

```powershell
Copy-Item third_party/vulkan/backend/*.py python/triton/backends/vulkan/ -Force
$env:TRITON_BACKENDS_IN_TREE = "1"
python third_party/vulkan/test/bench_kernels.py
```

---

## 12. Traps Encountered

### Trap P1: Never Alias `__local` to `__global`

**Symptom:** Softmax produces `inf`/`nan`. Elementwise kernels silently
overwrite their inputs.

**Root Cause:** An earlier design aliased `memref.copy %global, %alloc`
by mapping `%alloc → %global` in the SSA map. No actual copy occurred.
But `linalg.generic outs(%alloc)` writes the result BACK to `%alloc`.
Since `%alloc` is aliased to `%global` (the input), this overwrites the
input data.

For softmax: `reduce(max)` produces the correct maximum. Then
`x - max` computes correctly, writing the shifted values to `%alloc`.
But `%alloc` IS `%x` (aliased), so the original `x` data is destroyed.
When `exp(shifted)` runs, it reads from `shifted` (which IS `x`, now
corrupted), producing garbage → `inf` → `nan`.

**Fix:** All `memref.copy` ops do real per-workitem copies:
```c
sh_alloc[_tid] = ptr_global[_tid];
barrier(CLK_LOCAL_MEM_FENCE);
```

### Trap P2: Barrier After Cross-Workitem Writes

**Symptom:** Transpose/matmul produces partial garbage.

**Root Cause:** After `linalg.transpose`, workitem 0 writes to
`[col*R + row]` = `[0]`, workitem 1 writes to `[16]`, etc. The
subsequent `memref.copy` reads at `[_tid]` — workitem 1 reads `[1]`,
which was written by workitem `col=0, row=1` = workitem 1 (happens to
be the same workitem here, but for non-diagonal positions they differ).

Without a barrier, the `memref.copy` might read stale data before
all workitems have completed their transpose writes.

**Fix:** `barrier(CLK_LOCAL_MEM_FENCE)` after every transpose, matmul,
and cooperative copy.

### Trap P3: Reduction Result Must Be Broadcast

**Symptom:** After `linalg.reduce`, only workitem 0's subsequent
`linalg.generic` produces correct results. Workitems 1–255 use
uninitialized values.

**Root Cause:** Initial implementation: `if (_tid == 0) loc = _shared[0]`.
Only workitem 0 gets the reduction result. When softmax computes
`exp(x - max) / sum`, workitems 1–255 use whatever was in `loc` before
(garbage).

**Fix:** ALL workitems read the result:
```c
barrier(CLK_LOCAL_MEM_FENCE);
loc_init = _shared[0];    // no guard — every workitem reads
```

### Trap P4: MLIR Brace Placement Varies

**Symptom:** `linalg.reduce` body ops leak into the main dispatcher,
producing undeclared variables.

**Root Cause:** MLIR sometimes puts `{` on a SEPARATE line from the op:
```
linalg.reduce ins(...) outs(...) dimensions = [0]
  (%in: f32, %init: f32) {        ← brace is here, not on previous line
    %0 = arith.addf ...
```

A simple `depth = line.count('{')` on the first line sees depth=0 and
stops collecting, leaving the body ops to be processed by the main
dispatcher (which doesn't know about block arguments).

**Fix:** Two-phase collection: first scan until `{` is found, then
match until the closing `}`:
```python
depth = line.count('{') - line.count('}')
while i < len(lines) and depth <= 0:    # Phase 1: find opening brace
    generic_text += '\n' + lines[i]
    depth += lines[i].count('{') - lines[i].count('}')
    i += 1
while i < len(lines) and depth > 0:     # Phase 2: match closing brace
    generic_text += '\n' + lines[i]
    depth += lines[i].count('{') - lines[i].count('}')
    i += 1
```

### Trap P5: Greedy Regex on `memref<f32>`

**Symptom:** `memref.load %buf[] : memref<f32>` emits `2 ld_N = ...`
instead of `float ld_N = ...`.

**Root Cause:** Regex `memref<[^>]*x?(\w+)>` on `memref<f32>`: the greedy
`[^>]*` consumes `f3`, leaving only `2` for `(\w+)`.

**Fix:** Use alternation to handle both ranked and unranked forms:
```python
tm = re.search(r'x(\w+)>|memref<(\w+)>', line)
raw = tm.group(1) or tm.group(2)
```

`memref<256xf32>` matches the first alternative: `x(f32)>`.
`memref<f32>` matches the second alternative: `memref<(f32)>`.

### Trap P6: `str()` vs `str_nodebug()` for Clean IR

**Symptom:** `memref.alloc()` becomes `memref.al` after location stripping.

**Root Cause:** `str()` produces MLIR with `loc(...)` annotations.
A naive `re.sub(r'loc\([^)]*\)', '', ir)` matches the `loc()` substring
inside `alloc()`, destroying the op name.

**Fix:** Use `mod.str_nodebug()` in `compiler.py` which produces clean
IR with no loc annotations at all. The emitter receives parseable MLIR
without any stripping needed.

### Trap P7: Matmul `=` Not `+=`

**Symptom:** Matmul produces wrong results when output buffer has non-zero
initial values.

**Root Cause:** The inner loop `for (_k) _sum += A[...] * B[...]`
accumulates the full dot product in `_sum`. Writing `C[idx] += _sum`
adds `_sum` to whatever was already in `C[idx]` (which could be the
`linalg.fill` zero, or garbage from a previous kernel).

**Fix:** `C[row * N + col] = _sum;` — the dot product is complete in `_sum`.

### Trap P8: `__local` Declarations at Function Scope

**Symptom:** OpenCL compilation error: `__local` declared inside control flow.

**Root Cause:** The `_emit_alloc_skip` handler is called during body walking,
which might be inside a block that the dispatcher processes. OpenCL requires
`__local` declarations at function scope, not inside loops or conditionals.

**Fix:** Collect declarations in `self.local_arrays` during the walk.
After the walk completes, insert all declarations at the saved
`local_decl_idx` position (right after `int _tid = get_local_id(0);`).

---

## 13. Extension Guide: Adding New Ops

### Step 1: Inspect the Bufferized IR

Compile a TTIR through `make_memref_bufonly` and print the result:

```python
from triton._C.libtriton import ir, passes, vulkan
from triton.backends.vulkan.compiler import VulkanBackend, GPUTarget

c = ir.context(); ir.load_dialects(c); vulkan.load_dialects(c)
m = ir.parse_mlir_module("your_kernel.ttir", c); m.context = c
b = VulkanBackend(GPUTarget("vulkan", 0, 32))
o = b.parse_options({}); md = {}
m = b.make_ttir(m, md, o)
m = b.make_linalg(m, md, o)
m = b.make_memref_bufonly(m, md, o)
print(m.str_nodebug())
```

Look at the exact MLIR op structure: what operands does it have? Does it
have a body? Is the opening brace on the same line or the next?

### Step 2: Add to the Body Dispatcher

In `_emit_body` (line 197), add a new `elif` branch:

```python
elif 'linalg.newop' in line:
    # If the op has a body (braces), collect until matching close
    generic_text = line
    depth = line.count('{') - line.count('}')
    while i < len(lines) and depth <= 0:
        generic_text += '\n' + lines[i]
        depth += lines[i].count('{') - lines[i].count('}')
        i += 1
    while i < len(lines) and depth > 0:
        generic_text += '\n' + lines[i]
        depth += lines[i].count('{') - lines[i].count('}')
        i += 1
    self._emit_linalg_newop(generic_text)
```

For single-line ops (no braces), just call the handler directly:
```python
elif 'memref.newop' in line:
    self._emit_memref_newop(line)
```

### Step 3: Implement the Handler

Ask three questions:

1. **Does each workitem compute independently?**
   → Simple parallel: `result[_tid] = f(input[_tid]);`

2. **Do workitems need each other's results?**
   → Need `__local` memory + barrier for communication

3. **Is the output index different from `_tid`?**
   → Need barrier after write (other workitems will read different positions)

### Step 4: Add Barrier If Needed

If the output index pattern differs from `[_tid]`, add a barrier:
```python
self._line(f"barrier(CLK_LOCAL_MEM_FENCE);")
```

### Step 5: Add a Test

In `test_kernels_parallel.py`:

```python
def test_newop():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_newop.ttir"))
    x = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, WO, N * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), ob, np.int32(N)]
                   + [np.int32(0)] * 6)
    expected = your_numpy_reference(x)
    return np.max(np.abs(read(ob) - expected))
```

Add to the `TESTS` list with an appropriate tolerance.

### Step 6: Run Both Test Suites

```powershell
Copy-Item third_party/vulkan/backend/*.py python/triton/backends/vulkan/ -Force
$env:TRITON_BACKENDS_IN_TREE = "1"
python third_party/vulkan/test/test_kernels_parallel.py   # new tests
python third_party/vulkan/test/test_kernels.py             # regression check
```

---

## 14. Known Limitations

### 1. Single Workgroup Only

All workitems run in one workgroup. `barrier(CLK_LOCAL_MEM_FENCE)` only
synchronizes within a workgroup. Multi-workgroup execution (needed for
N > ~1024 on most GPUs) would require either:
- Splitting the kernel into multiple dispatches (like the serial path)
- Using atomic operations for cross-group synchronization
- Moving to a Vulkan compute pipeline with `subgroupBarrier`

### 2. Power-of-2 Block Sizes for Reduction

The tree reduction halves the stride (`s >>= 1`), which only works for
power-of-2 sizes. An assertion enforces this. Non-power-of-2 would
require padding or a more complex reduction algorithm.

### 3. No Multi-Output `linalg.generic`

Only the first output of `linalg.yield` is handled. Multi-output generics
(where `linalg.yield %a, %b`) would lose the second value. Triton rarely
produces these.

### 4. Not in `add_stages`

The parallel pipeline stages (`make_memref_bufonly`, `make_opencl_parallel`)
are not registered in `VulkanBackend.add_stages()`. They can only be called
manually. This is intentional for the proof-of-concept.

### 5. Block Size Override by Matmul/Transpose

`_emit_linalg_matmul` and `_emit_linalg_transpose` override `self.block_size`
to `M * N` (the total output elements). If a kernel mixes a 256-element
`linalg.generic` with a 512-element matmul, the generic was already emitted
with the original block_size, but the matmul changes it for dispatch. In
practice, all ops in a Triton kernel use compatible tensor sizes.

### 6. Shared `reduce_elem_type` Across Reductions

All `linalg.reduce` ops in a kernel share one `_shared` array with one
element type. This works when all reductions use the same type (common:
all `f32`). Mixed-type reduction chains would require per-reduction
`__local` arrays.

---

## 15. Appendix: File Inventory

### New Files (OpenCL Parallel Emitter)

| File | Lines | Purpose |
|------|-------|---------|
| `third_party/vulkan/backend/emitter_parallel.py` | ~790 | Parallel OpenCL C emitter |
| `third_party/vulkan/test/test_kernels_parallel.py` | ~198 | 10-kernel parallel GPU test suite |
| `.github/skills/triton-windows-opencl/SKILL.md` | ~80 | OpenCL emitter skill |

### Modified Files (OpenCL Parallel Emitter)

| File | Lines Added | Change |
|------|-------------|--------|
| `third_party/vulkan/backend/compiler.py` | +28 | `make_memref_bufonly`, `make_opencl_parallel` |
| `third_party/vulkan/test/bench_kernels.py` | ~50 | Added `compile_parallel`, serial vs parallel comparison |

### Unchanged From Converter Infrastructure

The C++ conversion pass (`TritonToLinalg.cpp`), serial emitter (`emitter.py`),
driver (`driver.py`), and all TTIR test files are unchanged. The parallel
emitter is purely a Python-side addition that reuses the existing TTIR → Linalg
conversion and bufferization infrastructure.

### Class Hierarchy

```
emitter.py::OpenCLEmitter        (serial)
  └─ Walks fully-lowered IR (scf.for, cf.br)
  └─ 1 workitem per block

emitter_parallel.py::ParallelOpenCLEmitter  (parallel)
  └─ Walks bufferized IR (linalg.generic, linalg.reduce)
  └─ N workitems per block
  └─ Tree reduction with __local memory
```

Both emitters share the same design pattern (SSA map, type map, regex-based
op walking) but handle fundamentally different IR: the serial emitter sees
loops and branches, while the parallel emitter sees structured ops with
iterator types and block arguments.
