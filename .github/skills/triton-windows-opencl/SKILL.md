---
name: triton-windows-opencl
description: "Optional OpenCL C debug output for the Triton Vulkan backend. Two emitters: serial (emitter.py, line-by-line MLIR→C) and parallel (emitter_parallel.py, linalg→workitem-per-element). Useful for debugging converters with human-readable output. NOT required for the production Vulkan/SPIR-V pipeline. Use for: debugging converter output, understanding lowered IR, testing via pyopencl, or comparing serial vs parallel execution."
argument-hint: "serial-emitter | parallel-emitter | debugging | benchmarking"
user-invocable: true
---

# Triton-Windows OpenCL Debug Emitters

**This is an optional debugging tool.** The production Vulkan backend uses
SPIR-V binary dispatch (see `triton-windows-vulkan`). The OpenCL emitters
provide human-readable C output for debugging TritonToLinalg converters, and
should be treated as a debugging aid rather than the primary execution path.

## When to Use This

- **Debugging a new converter:** Dump the OpenCL C to see what ops your
  converter produces, then verify with pyopencl
- **Quick correctness check:** Paste OpenCL C into any GPU's OpenCL runtime
- **Performance comparison:** Serial vs parallel vs Vulkan SPIR-V

## Two Emitters

### Serial Emitter (`emitter.py`)

Pipeline: `make_ttir → make_linalg → make_memref → make_opencl`

Walks fully-lowered MemRef/CF IR line-by-line. Each MLIR op → one C statement.
Single-threaded: one workitem does all computation in serial loops.

### Parallel Emitter (`emitter_parallel.py`)

Pipeline: `make_ttir → make_linalg → make_memref_bufonly → make_opencl_parallel`

Walks bufferized IR with linalg ops preserved (no loop lowering).
Each workitem processes one element via `get_local_id(0)`.

| linalg Op | Parallelization |
|-----------|----------------|
| `linalg.generic {parallel}` | 1 workitem/element |
| `linalg.reduce` | Tree reduction: `__local` + `barrier` |
| `linalg.matmul` | 1 workitem/output element |
| `linalg.transpose` | 1 workitem/element swap |

**Key traps:** Never alias `__local` to `__global`; barrier after every
cross-workitem write; reduction broadcast to ALL workitems; use `=` not `+=`
for matmul output; `__local` declarations at function scope only.

## Test Suites

- `test_kernels.py` — the current serial-emitter test suite via pyopencl
- `test_kernels_parallel.py` — the current parallel-emitter test suite via pyopencl
- `bench_kernels.py` — Serial vs parallel vs CUDA benchmarks

## Architecture Note

The OpenCL and Vulkan paths share `make_ttir → make_linalg → make_memref`
(the TritonToLinalg converters). They diverge at step 4:

```
              make_linalg (shared)
             ┌────────┴────────┐
      make_memref          make_memref_bufonly
      (bufferize+loops+cf)  (bufferize only)
           │                      │
      make_opencl          make_opencl_parallel
      (serial text)        (parallel text)
           │                      │
         pyopencl               pyopencl
```

The Vulkan/SPIR-V path uses `make_memref → make_spirv` (see `triton-windows-vulkan`).

Zero code crosses between them. The OpenCL emitters never touch
PrepareSPIRV.cpp; the Vulkan path never imports emitter.py.

## Adapting to Upstream Changes

If upstream Triton or the Vulkan bridge changes, keep the shared pipeline model
(`make_ttir → make_linalg → ...`) but locate the current implementation by
searching for the stage or converter name rather than relying on file size or
test counts. Use the current test suite plus a known kernel to compare the
serial and parallel emitters against the Vulkan path semantically.
