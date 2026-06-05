---
name: triton-windows-vulkan-phase3
description: "Parallel OpenCL emitter for the Triton Vulkan backend. Covers the linalg→parallel-OpenCL pipeline, workgroup tree reductions, __local memory management, barrier placement, and all traps discovered during Phase 3. Use for: parallelizing kernels, fixing barrier issues, debugging __local memory, understanding the parallel emitter architecture, or extending to new linalg ops."
argument-hint: "architecture | barrier-trap | reduction | matmul | transpose | softmax | copy-semantics | type-detection | full-overview"
user-invocable: true
---

# Triton-Windows Vulkan Backend — Phase 3 Parallel Emitter Skill

You are an expert at the parallel OpenCL emitter for triton-windows. This skill
documents the architecture, every trap, and every workaround discovered during
Phase 3 implementation.

**Prerequisites:** Phase 1 (`triton-windows-spirv-setup`) and Phase 2
(`triton-windows-vulkan-phase2`) skills cover the serial pipeline.

## Quick Status

**10/10 parallel GPU tests pass on RTX 2080 Ti via pyopencl.**
**253× speedup** for vector_add at N=65536 over the serial emitter.

**Pipeline comparison:**
```
Serial (Phase 2):   TTIR → linalg → memref (bufferize+loops+cf) → emitter.py → serial OpenCL
Parallel (Phase 3): TTIR → linalg → memref_bufonly (bufferize)  → emitter_parallel.py → parallel OpenCL
```

The key difference: Phase 3 stops before loop lowering, keeping `linalg.generic`
/ `linalg.reduce` / `linalg.matmul` / `linalg.transpose` ops intact. The
parallel emitter maps each op directly to OpenCL workitems.

---

## Architecture

### File Layout

| File | Lines | Purpose |
|------|-------|---------|
| `third_party/vulkan/backend/emitter_parallel.py` | ~690 | Parallel OpenCL C emitter |
| `third_party/vulkan/backend/compiler.py` | ~300 | `make_memref_bufonly`, `make_opencl_parallel` |
| `third_party/vulkan/test/test_kernels_parallel.py` | ~190 | 10-kernel parallel GPU test suite |

### Pipeline Stages

```python
# In compiler.py:
mod = backend.make_ttir(mod, md, opt)      # Shared TTIR passes
mod = backend.make_linalg(mod, md, opt)    # TritonToLinalg C++ pass
mod = backend.make_memref_bufonly(mod, md, opt)  # Bufferize only — NO loop lowering
src = backend.make_opencl_parallel(mod, md, opt) # Parallel emitter
# md['block_size'] is set by the emitter (e.g., 256)
```

**CRITICAL:** `make_memref_bufonly` does NOT call `convert_linalg_to_loops`,
`lower_affine`, or `convert_scf_to_cf`. This preserves `linalg.*` ops for
the parallel emitter.

### IR Format After Bufferization

The emitter receives clean MLIR via `mod.str_nodebug()` (no loc annotations).
A typical bufferized kernel looks like:

```mlir
func.func @kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, ...) {
  %reinterpret = memref.reinterpret_cast %arg0 ...  // pointer + offset
  %alloc = memref.alloc() : memref<256xf32>          // local buffer
  memref.copy %reinterpret, %alloc                    // load from global
  linalg.generic {iterator_types = ["parallel"]}      // computation
    ins(%alloc ...) outs(%alloc ...) { body }
  memref.copy %alloc, %reinterpret_out                // store to global
  return
}
```

### OpenCL Execution Model

- Each kernel runs as ONE workgroup with `block_size` workitems
- `get_local_id(0)` gives each workitem its element index (`_tid`)
- `__local` arrays are shared across all workitems in the workgroup
- `barrier(CLK_LOCAL_MEM_FENCE)` synchronizes all workitems

**Why `get_local_id(0)` not `get_global_id(0)`:** The tree reduction uses
`barrier(CLK_LOCAL_MEM_FENCE)` which only synchronizes within a workgroup.
`get_local_id(0)` is correct. For single-workgroup dispatch, they're identical.

---

## Op Mapping Reference

### linalg.generic (parallel) → One Workitem Per Element

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

**Emitted OpenCL:**
```c
float v_1 = sh_a[_tid];   // ins[0][_tid]
float v_2 = sh_b[_tid];   // ins[1][_tid]
float v_3 = sh_c[_tid];   // outs[0][_tid]
float f_4 = v_1 * v_2;
sh_c[_tid] = f_4;         // yield → write to outs[0][_tid]
```

### linalg.reduce → Workgroup Tree Reduction

```mlir
linalg.reduce ins(%data : memref<256xf32>) outs(%init : memref<f32>)
  dimensions = [0] {
^bb0(%in: f32, %acc: f32):
  %0 = arith.addf %in, %acc : f32
  linalg.yield %0 : f32
}
```

**Emitted OpenCL:**
```c
_shared[_tid] = sh_data[_tid];
barrier(CLK_LOCAL_MEM_FENCE);
for (int _s = 128; _s > 0; _s >>= 1) {
  if (_tid < _s)
    _shared[_tid] = _shared[_tid] + _shared[_tid + _s];
  barrier(CLK_LOCAL_MEM_FENCE);
}
barrier(CLK_LOCAL_MEM_FENCE);
loc_init = _shared[0];  // ALL workitems read the result
```

### linalg.matmul → One Workitem Per Output Element

```c
int _row = _tid / N;
int _col = _tid % N;
float _sum = 0.0f;
for (int _k = 0; _k < K; _k++)
  _sum += A[_row * K + _k] * B[_k * N + _col];
C[_row * N + _col] = _sum;  // = not += (sum already complete)
barrier(CLK_LOCAL_MEM_FENCE);
```

### linalg.transpose → One Workitem Per Element

```c
int _row = _tid / C;
int _col = _tid % C;
dst[_col * R + _row] = src[_row * C + _col];
barrier(CLK_LOCAL_MEM_FENCE);
```

---

## Memory Model

### Three Memory Types

| Type | Keyword | Scope | Source |
|------|---------|-------|--------|
| Global | `__global float*` | All workgroups | Function args (`memref<*xf32>`) |
| Local (shared) | `__local float[]` | One workgroup | `memref.alloc() : memref<Nxf32>` (array) |
| Private | `float` | One workitem | `memref.alloc() : memref<f32>` (scalar) |

### Allocation Rules

The emitter's `_emit_alloc_skip` decides based on memref shape:

| MLIR Type | OpenCL | Why |
|-----------|--------|-----|
| `memref<f32>` (0-d) | `float loc_N;` | Scalar result (e.g., reduction output) |
| `memref<256xf32>` (1-d) | `__local float sh_N[256];` | Shared across workitems |
| `memref<16x16xf32>` (2-d) | `__local float sh_N[256];` | Flattened shared array |

**CRITICAL:** Array allocs MUST be `__local`, not `__private`. Each workitem
writes ONE element, and other ops (matmul, transpose, output copy) read
elements written by OTHER workitems. `__private` arrays are per-workitem
and invisible to other workitems.

---

## Traps & Workarounds

### TRAP P1: `memref.copy` Global→Local Must Not Alias

**Symptom:** Softmax produces `inf`/`nan`. Elementwise kernels overwrite inputs.

**Root Cause:** An earlier design aliased `%alloc → global_ptr` to avoid copies.
But `linalg.generic outs(%alloc)` writes BACK to `%alloc`. If aliased to the
global input pointer, this overwrites the input data — catastrophic when
subsequent ops read from it (e.g., softmax's `exp(x - max)` after `max = reduce(x)`).

**Fix:** ALL `memref.copy` operations do real per-workitem element copies:
```c
sh_alloc[_tid] = ptr_global[_tid];  // each workitem copies one element
barrier(CLK_LOCAL_MEM_FENCE);       // ensure all elements visible
```
No aliasing. The `__local` array is a real separate buffer.

**Rule:** NEVER alias `__local` arrays to `__global` pointers in the parallel
emitter. The serial emitter can alias because it's single-threaded.

### TRAP P2: Barrier Placement After Cross-Workitem Writes

**Symptom:** Transpose/matmul produces partial garbage. Output copy reads
stale values.

**Root Cause:** After `linalg.transpose` or `linalg.matmul`, each workitem
writes to a DIFFERENT position in the `__local` array. The subsequent
`memref.copy` reads position `[_tid]` from the SAME array. Without a barrier,
workitem N's write to position `[col*R+row]` may not be visible to the
workitem that reads that position.

**Fix:** Emit `barrier(CLK_LOCAL_MEM_FENCE)` after EVERY operation that writes
to `__local` memory where other workitems may read:
- After `linalg.transpose` write
- After `linalg.matmul` write
- After `memref.copy` to `__local`
- After each step of tree reduction

**Rule:** When in doubt, add a barrier. Extra barriers cost ~20ns each on
modern GPUs — negligible vs the cost of debugging race conditions.

### TRAP P3: Reduction Result Must Be Broadcast

**Symptom:** After `linalg.reduce`, only workitem 0 has the correct result.
Subsequent `linalg.generic` ops that use the reduction result (e.g., softmax's
`x / sum(exp(x))`) produce wrong values for workitems 1-255.

**Root Cause:** The initial implementation had `if (_tid == 0) loc = _shared[0]`,
but ALL workitems need the reduction result for subsequent elementwise ops.

**Fix:** ALL workitems read from `_shared[0]` after the reduction:
```c
barrier(CLK_LOCAL_MEM_FENCE);  // ensure reduction is complete
loc_init = _shared[0];         // ALL workitems read
```
No `if (_tid == 0)` guard — every workitem needs the value.

**When to guard:** Only for scalar stores to global memory:
```c
if (_tid == 0) global_out[0] = result;  // only one workitem writes
```

### TRAP P4: linalg.reduce Block Collection

**Symptom:** Reduce body ops leak into the main body dispatcher, producing
undeclared variables like `v_in`, `v_init`.

**Root Cause:** MLIR prints `linalg.reduce` with the opening `{` on a
SEPARATE LINE from the `linalg.reduce` keyword:
```
linalg.reduce ins(...) outs(...) dimensions = [0]   ← no brace
  (%in: f32, %init: f32) {                           ← brace here
    %0 = arith.addf %in, %init : f32
    linalg.yield %0 : f32
  }
```

The brace-matching collector must handle depth=0 on the first line:
```python
depth = line.count('{') - line.count('}')
# If no opening brace on first line, keep collecting until we find one
while i < len(lines) and depth <= 0:
    generic_text += '\n' + lines[i]
    depth += lines[i].count('{') - lines[i].count('}')
    i += 1
while i < len(lines) and depth > 0:
    generic_text += '\n' + lines[i]
    depth += lines[i].count('{') - lines[i].count('}')
    i += 1
```

The same pattern applies to `linalg.generic` — both need the two-phase
collection (find opening brace, then match closing brace).

### TRAP P5: Type Extraction from `memref<f32>`

**Symptom:** `memref.load %buf[] : memref<f32>` emits `2 ld_N = ...` instead
of `float ld_N = ...` (the `2` is the SSA number `%2`).

**Root Cause:** The regex `memref<[^>]*x?(\w+)>` is greedy. On `memref<f32>`,
`[^>]*` consumes `f3`, leaving only `2` for `(\w+)`.

**Fix:** Use alternation to handle both forms:
```python
tm = re.search(r'x(\w+)>|memref<(\w+)>', line)
if tm:
    raw = tm.group(1) or tm.group(2)
    elem = self.TYPE_MAP.get(raw, raw)
```

### TRAP P6: `str_nodebug()` vs `str()` for Clean IR

**Symptom:** `loc(...)` annotations corrupt op parsing. `memref.alloc()` becomes
`memref.al` because `loc(...)` stripping eats the `loc()` in `alloc()`.

**Root Cause:** A naive `re.sub(r'loc\([^)]*\)', '', ir)` matches ANY `(...)` 
after `loc` — including the `()` in `alloc()`, `memref.load %buf[]`, etc.

**Fix:** Use `mod.str_nodebug()` in `compiler.py` which produces clean IR
with no loc annotations at all. No stripping needed.

```python
# In compiler.py make_opencl_parallel:
mlir_text = src.str_nodebug()  # NOT str(src) which includes locs
```

### TRAP P7: Matmul Output Uses `=` Not `+=`

**Symptom:** Matmul produces wrong results when output buffer has non-zero
initial values.

**Root Cause:** The dot product loop `for (_k) _sum += A[...] * B[...]`
already accumulates the full result. Writing `C[...] += _sum` would
double-count if `C` was initialized by `linalg.fill` to zero — but if
fill was skipped or C had stale data, the `+=` adds to garbage.

**Fix:** Always use `=`:
```c
C[_row * N + _col] = _sum;  // NOT +=
```

### TRAP P8: `__local` Array Declarations Must Be at Function Scope

**Symptom:** OpenCL compilation error: `__local` variable declared inside
control flow.

**Root Cause:** `__local` declarations must be at kernel function scope,
not inside loops or if blocks. But `_emit_alloc_skip` is called during
body walking, which may be at any nesting level.

**Fix:** Collect `__local` declarations in `self.local_arrays` during body
walking. Insert them ALL at the top of the function after `get_local_id`:
```python
# After emitting body, insert all __local declarations at saved position
for decl in self.local_arrays:
    self.lines.insert(local_decl_idx, decl)
```

---

## Body Op Type Detection

### The Problem

MLIR ops carry type annotations: `%0 = arith.mulf %a, %b : f32`. The
serial emitter hardcodes `float` everywhere. The parallel emitter must
detect the type from the annotation to support `half`/`double`/`int`.

### The Fix

Extract type from the trailing `: type` annotation:
```python
type_hint = re.search(r':\s*(\w+)\s*$', line)
ftype = "float"
if type_hint:
    ftype = self.TYPE_MAP.get(type_hint.group(1), "float")
```

Then use `ftype` for variable declarations:
```python
cvar = self._def(dst, "f", ftype)
self._line(f"{ftype} {cvar} = {self._map_val(lhs)} {sym} {self._map_val(rhs)};")
```

**Also applies to:** `_emit_linalg_reduce` — the `__local` declaration must
match the element type, not hardcode `float`:
```python
tm = re.search(r'memref<\d+x(\w+)>', text)
reduce_elem = self.TYPE_MAP.get(tm.group(1), "float")
# Later: __local {reduce_elem} _shared[N];
```

---

## Test Suite

### Running Tests

```powershell
# Sync Python files
Copy-Item third_party/vulkan/backend/*.py python/triton/backends/vulkan/ -Force

# Run parallel tests (10 kernels)
$env:TRITON_BACKENDS_IN_TREE="1"
python third_party/vulkan/test/test_kernels_parallel.py

# Run serial tests (14 kernels) — must not regress
python third_party/vulkan/test/test_kernels.py
```

### Test Coverage

| Kernel | Op Pattern | Parallelization |
|--------|-----------|----------------|
| vector_add | linalg.generic(parallel) | 1 workitem/element |
| elementwise_mul | linalg.generic(parallel) | 1 workitem/element |
| fma | 2× linalg.generic(parallel) | 1 workitem/element |
| gelu | linalg.fill + 4× linalg.generic | linalg.fill + parallel |
| swiglu | linalg.fill + linalg.generic chain | linalg.fill + parallel |
| reduce_sum | linalg.reduce(addf) | Tree reduction |
| reduce_max | linalg.reduce(maximumf) | Tree reduction |
| softmax | 2× linalg.reduce + 3× linalg.generic | Mixed: reduce + parallel |
| matmul_16×16 | linalg.matmul | Per-output-element |
| transpose_16×16 | linalg.transpose | Per-element index swap |

### Dispatch Pattern

```python
prog = cl.Program(ctx, src).build()
kernel = getattr(prog, metadata["name"])
for i, a in enumerate(args):
    kernel.set_arg(i, a)
bs = metadata['block_size']  # set by emitter (e.g., 256)
cl.enqueue_nd_range_kernel(queue, kernel, (bs,), (bs,))
queue.finish()
```

**Key:** `global_size = local_size = block_size`. One workgroup, N workitems.

---

## Performance

| Config | N | Time | Speedup |
|--------|---|------|---------|
| Serial (256 blocks × 1 workitem) | 65536 | 48,683 µs | 1× |
| Parallel (1 dispatch × 256 workitems) | 65536 | 192 µs | 253× |
| CUDA (PyTorch) | 65536 | 19 µs | 2,539× vs serial |

At N=256 (single block), launch overhead dominates (~30µs both modes).
The speedup scales with `block_size × num_blocks`.

**Why still slower than CUDA:** OpenCL on NVIDIA runs through a compatibility
layer. Native CUDA has lower launch overhead and better memory coalescing.
Route A (Phase 3.5: native TTG→SPIR-V via Vulkan compute) would close this gap.

---

## Adding a New Linalg Op Handler

1. **Check the bufferized IR:** Run `make_memref_bufonly` and print `str_nodebug()`
   to see the exact op structure.

2. **Add to dispatcher** in `_emit_body`:
   ```python
   elif 'linalg.newop' in line:
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

3. **Implement handler:** Emit per-workitem code. Key questions:
   - Does each workitem compute independently? → Simple parallel
   - Do workitems need each other's results? → Need `__local` + barrier
   - Is the output at a different index than `_tid`? → Need barrier after write

4. **Add barrier after writes** if other workitems read the output:
   ```python
   self._line(f"barrier(CLK_LOCAL_MEM_FENCE);")
   ```

5. **Add test** in `test_kernels_parallel.py`.

6. **Run both test suites** to verify no regressions.

---

## Known Limitations

1. **Single workgroup only.** All workitems must be in one workgroup for
   `barrier` to work. Multi-workgroup execution requires splitting the kernel
   or using atomic operations for cross-group synchronization.

2. **Power-of-2 block sizes for reduction.** The tree reduction halves the
   stride (`s >>= 1`), which only works correctly for power-of-2 sizes.
   Non-power-of-2 drops elements silently. An assertion guards this.

3. **No multi-output linalg.generic.** Only the first output of
   `linalg.generic` is handled. Multi-output generics (rare in Triton)
   would lose data.

4. **Parallel pipeline not in `add_stages`.** The `VulkanBackend.add_stages()`
   only registers the serial+SPIR-V pipeline. Parallel stages must be called
   manually. This is intentional for Phase 3 (toy/proof-of-concept).

---

## Lessons Learned

1. **Never alias `__local` to `__global`.** It seems clever to avoid copies,
   but linalg.generic outs writes BACK to the buffer. If it's aliased to a
   global input, you corrupt your input data. Softmax was the most painful
   debugging session because of this — two reductions + three elementwise ops
   all sharing the same "aliased" buffer.

2. **Barrier after EVERY cross-workitem write.** Not just reductions — also
   matmul output, transpose output, and cooperative `memref.copy` to `__local`.
   The cost of a barrier (~20ns) is negligible; the cost of a missing barrier
   is hours of debugging non-deterministic wrong results.

3. **Use `str_nodebug()` for parsing.** `str()` includes `loc(...)` annotations
   that corrupt regex-based parsing. We lost an hour to `alloc()` becoming
   `al` because `loc(...)` stripping ate the parentheses.

4. **The MLIR brace is not always on the same line.** `linalg.reduce` puts
   `{` on the NEXT line after `dimensions = [0]`. The block collector must
   handle depth=0 on the first line by continuing to collect until a `{` is
   found.

5. **Greedy regex is the #1 bug source.** `memref<[^>]*x?(\w+)>` on
   `memref<f32>` consumes `f3` in the greedy `[^>]*`, leaving only `2` for
   the type capture. Always test regex on ALL memref shapes: `memref<f32>`,
   `memref<256xf32>`, `memref<16x16xf32>`, `memref<*xf32>`.

6. **Reduction results need ALL-workitem broadcast.** After a tree reduction,
   `_shared[0]` has the result. If only tid=0 reads it, the subsequent
   elementwise ops (softmax's `exp(x-max)/sum`) get uninitialized values
   for tids 1-255. This was the softmax `inf` bug.

7. **`=` not `+=` for matmul output.** The inner loop already accumulates
   the full dot product. `+=` would add to whatever was in the output buffer
   before `linalg.fill` — which could be garbage if fill didn't execute first.
