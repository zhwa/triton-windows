---
name: triton-windows-dev
description: "Develop, test, debug, and profile triton-windows kernels and compiler pipeline. Use for: writing/testing kernels (pytest), debugging triton-opt or JIT, tracing IR (TTIR→TTGIR→LLVM→PTX), running lit tests, benchmarking, interpreter mode, or understanding the compilation pipeline."
argument-hint: "test-kernel [pattern] | debug-pipeline | dump-ir | run-lit [path] | benchmark | write-kernel <desc> | explain-pipeline | help"
user-invocable: true
---

# Triton-Windows Development & Debugging Skill

You help users develop, test, debug, profile, and understand triton kernels and
the compiler pipeline on Windows.

## Compilation Pipeline

```
@triton.jit Python kernel
    │ ast_to_ttir()              python/triton/compiler/code_generator.py
    ▼
  TTIR  ──► make_ttir()          canonicalize, combine, CSE
    │
    ▼
  TTGIR ──► make_ttgir()         convert_to_ttgpuir, coalesce, matmul accel,
    │                            pipeline, warp-specialize, fence-insertion
    ▼
  LLVM IR ► make_llir()          TritonGPU→LLVM lowering, optimize(O3)
    │
    ▼
  PTX ────► make_ptx()           llvm.translate_to_asm()
    │
    ▼
  cubin ──► make_cubin()         external ptxas assembler
```

Stages are defined in `python/triton/backends/nvidia/compiler.py` → `CUDABackend.add_stages()`.

## Tool: `triton-opt.exe`

Located at `python/triton/_C/triton-opt.exe` after building.
Standard MLIR `opt` tool with all Triton dialects/passes registered.

```powershell
# Parse and verify TTIR
triton-opt.exe input.ttir -verify-diagnostics

# Run combine pass
triton-opt.exe input.ttir -canonicalize -triton-combine

# TTIR → TTGIR conversion
triton-opt.exe input.ttir -convert-triton-to-tritongpu="target=cuda:80 num-warps=4 threads-per-warp=32 num-ctas=1"

# Run coalescing
triton-opt.exe input.ttgir -tritongpu-coalesce

# TTGIR → LLVM (NVIDIA)
triton-opt.exe input.ttgir -convert-triton-gpu-to-llvm="compute-capability=80"
```

Build `triton-opt` specifically: from `build/cmake.win-amd64-cpython-3.14/`, run `ninja triton-opt`.

## VS Code Integration

The skill provides full VS Code integration for test discovery, running, and C++ debugging.

### Setup

VS Code configuration files are in `.vscode/`:
- `settings.json` — pytest discovery pointing to test kernels
- `launch.json` — Python test + C++ triton-opt debug configurations
- `c_cpp_properties.json` — IntelliSense via triton's `compile_commands.json`
- `tasks.json` — rebuild triton, generate debug IR, run tests

### Test Explorer (Python)

Tests appear in VS Code's **Testing** sidebar (beaker icon). Click the green triangle to run
a test, or the bug icon to debug it with Python breakpoints.

The test runner uses the **build Python** (from the `triton-dev` conda env)
which has the freshly built triton. Never use a separately installed triton for testing.

### Debugging C++ Compiler Passes

This is the key workflow for compiler development. Each test kernel has pre-generated
TTIR (`.vscode/debug/*.ttir`). To debug a specific compiler pass:

1. Open the **Run and Debug** sidebar (Ctrl+Shift+D)
2. Select **"triton-opt: debug kernel pass"** from the dropdown
3. Pick the kernel TTIR (e.g., `matmul.ttir` for DotOp debugging)
4. Pick the compiler pass pipeline (e.g., `accelerate-matmul`)
5. Set C++ breakpoints in the compiler source (e.g., `lib/Dialect/TritonGPU/Transforms/`)
6. Press F5 — triton-opt launches under `cppvsdbg` with full PDB symbols

Available TTIR files and their primary compiler paths:

| TTIR File | Kernel | Key Passes to Debug |
|---|---|---|
| `vector_add.ttir` | elementwise | `triton-combine`, `ElementwiseOpToLLVM` |
| `reduce_sum.ttir` | reduction | `ReduceOpToLLVM`, `atomic_add` |
| `softmax.ttir` | fused reduction | `coalesce`, `reorder-broadcast` |
| `matmul.ttir` | dot product | `accelerate-matmul`, `DotOpToLLVM`, `pipeline` |
| `transpose.ttir` | layout change | `RemoveLayoutConversions`, `ViewOpToLLVM` |
| `gelu.ttir` | math ops | `tl.math.exp` lowering, libdevice |
| `broadcast_add.ttir` | 2D broadcast | `reorder-broadcast`, 2D tensor layout |

Regenerate TTIR after kernel changes:
```powershell
$env:TRITON_PTXAS_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/ptxas.exe"
conda run -n triton-dev python .github/skills/triton-windows-dev/samples/generate_debug_ir.py
```
Or use VS Code task: **Terminal > Run Task > triton: generate debug IR**

### Mixed-Mode Debugging (Python + C++)

To debug C++ compiler code during Python kernel JIT compilation:

1. Start **"Python: pytest current file (GPU)"** launch config
2. Set a Python breakpoint before the kernel launch (e.g., before `kernel[grid](...)`)
3. When Python stops at the breakpoint, launch **"Attach C++ to Python (mixed debug)"**
4. Pick the `python.exe` process from the list
5. Set C++ breakpoints in compiler code (e.g., `TritonGPUToLLVM/ElementwiseOpToLLVM.cpp`)
6. Continue Python — when JIT compiles the kernel, C++ breakpoints will hit

> Note: Mixed-mode requires GPU + triton built with debug symbols (release-with-asserts).

## Task Dispatch

---

### Task: `setup-vscode`

Set up VS Code for test discovery and C++ debugging against the **built triton**.
This is the first task to run on a fresh machine after building triton.

**Important:** All configs use the build Python (`triton-dev` conda env) that has the
freshly built triton. Never use a separately pip-installed triton for testing.

Steps:

1. **Install test dependencies** in the build Python (triton-dev):
   ```powershell
   conda run -n triton-dev pip install pytest torch --quiet
   ```
   Note: `pip install torch` gets CPU-only torch by default. For GPU testing, use:
   ```powershell
   conda run -n triton-dev pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

2. **Create/verify `.vscode/settings.json`** with:
   - `python.defaultInterpreterPath`: build Python from `triton-dev` conda env
   - `python.testing.pytestEnabled`: true
   - `python.testing.pytestArgs`: `[".github/skills/triton-windows-dev/samples", "-s", "--tb=short"]`
   - `C_Cpp.default.compileCommands`: `${workspaceFolder}/build/cmake.win-amd64-cpython-3.14/compile_commands.json`

3. **Create/verify `.vscode/launch.json`** with these configurations:

   Python test configs (use build Python with built triton):
   - `"Python: pytest current file (interpreter)"` — runs current file with `TRITON_INTERPRET=1`
   - `"Python: pytest current file (GPU)"` — runs current file on GPU
   - `"Python: pytest all test kernels"` — runs full test suite (interpreter)

   C++ kernel pass debugging (each kernel has pre-generated TTIR):
   - `"triton-opt: debug kernel pass"` — pick kernel + pass from dropdown, debug C++ code
   - Uses `${input:kernelIR}` and `${input:compilerPass}` input variables

   Existing triton-opt configs:
   - `"triton-opt: parse vecadd"`, `"triton-opt: combine pass"`, `"triton-opt: current file"`

   Mixed-mode C++ attach:
   - `"Attach C++ to Python (mixed debug)"` — attach C++ debugger to running Python test

4. **Create/verify `.vscode/c_cpp_properties.json`** — points to triton's `compile_commands.json`
5. **Create/verify `.vscode/tasks.json`** — rebuild triton, rebuild triton-opt, generate IR, run tests
6. **Generate debug TTIR files:**
   ```powershell
   conda run -n triton-dev python .github/skills/triton-windows-dev/samples/generate_debug_ir.py
   ```
7. **Verify test discovery:**
   ```powershell
   conda run -n triton-dev python -m pytest .github/skills/triton-windows-dev/samples --collect-only -q
   ```

---

### Task: `test-kernel [pattern]`

Run the kernel test suite. Tests cover 6 complexity levels with proper pytest
fixtures, parametrization, dtype sweeps, and tolerance handling.

```powershell
# Run all kernel tests
pytest .github/skills/triton-windows-dev/samples/test_kernels.py -s --tb=short

# Run specific test by name pattern
pytest .github/skills/triton-windows-dev/samples/test_kernels.py -k "softmax" -s --tb=short

# Run in interpreter mode (no GPU required)
$env:TRITON_INTERPRET = "1"
pytest .github/skills/triton-windows-dev/samples/test_kernels.py -s --tb=short -k "not benchmark"

# Force recompilation
$env:TRITON_ALWAYS_COMPILE = "1"
pytest .github/skills/triton-windows-dev/samples/test_kernels.py -s --tb=short

# Run upstream triton tests (GPU required)
pytest python/test/unit/language/test_core.py -s --tb=short --device cuda
pytest python/test/unit/language/test_core.py::test_bin_op -s --tb=short
```

**Test suite levels** (in `samples/test_kernels.py`):

| Level | Kernel | Pattern | Compiler Paths Exercised |
|---|---|---|---|
| 1 | `vector_add` | Elementwise | ElementwiseOp, MemoryOp, SPMDOp, MakeRangeOp |
| 2 | `fma` | Fused multiply-add | + ArithTypeConversion (dtype cast) |
| 3 | `reduce_sum` | Block reduction | + ReduceOpToLLVM, atomic_add |
| 4 | `softmax` | Row-wise softmax | + max/exp/sum reductions, next_power_of_2 |
| 5 | `rmsnorm` | Normalize + scale | + broadcast, rsqrt pattern |
| 6 | `matmul` | Tiled GEMM | + DotOp, AccelerateMatmul, loop→scf.for |
| 7 | `relu_dropout` | Control flow | + CombineTensorSelectAndIf, tl.where, constexpr branch, tl.rand |
| 8 | `transpose` | 2D layout | + ViewOp, RemoveLayoutConversions, multi-dim grid |
| 9 | `gelu` | Libdevice math | + `tl.math.exp`, extern elementwise, manual tanh via exp |
| 10 | `cumsum` | Prefix scan | + ScanOpToLLVM (distinct from reduce) |
| 11 | `reduce_max` | Atomic max | + atomic_max codegen path |
| 12 | `autotuned_add` | Autotune | + @triton.autotune runtime, Config, key |
| 13 | `broadcast_add` | Broadcast | + ReorderBroadcast, expand_dims, 2D ops |
| 14 | `pipelined_matmul` | Pipelining | + pipeline, schedule-loops, tl.range(num_stages) |

Plus benchmarks (vector_add GB/s, matmul TFLOPS) and an IR dump smoke test.

**Writing new tests — follow these conventions:**

```python
# 1. Use pytest parametrize for dtype/shape sweeps
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("N", [128, 1024, 65537])
def test_my_kernel(dtype, N):

# 2. Use tolerance by dtype
TOLERANCES = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=1e-1, rtol=1e-1),
}

# 3. Compare against PyTorch reference
ref = torch.softmax(x, dim=-1)
assert torch.allclose(out.float(), ref.float(), **tol)

# 4. Skip if no GPU (or use interpreter)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() and not IS_INTERPRET,
    reason="CUDA required (or set TRITON_INTERPRET=1)")

# 5. Skip bf16/fp8 on older hardware
HAS_BF16 = torch.cuda.get_device_capability()[0] >= 8  # Ampere+
HAS_FP8 = torch.cuda.get_device_capability()[0] >= 9   # Hopper+
```

**Upstream test infrastructure** (for reference):
- `python/test/conftest.py` — `device`, `fresh_triton_cache`, `fresh_knobs` fixtures
- `python/triton/_internal_testing.py` — `numpy_random`, `to_triton`, `to_numpy`, dtype lists
- `python/triton/testing.py` — `do_bench`, `do_bench_cudagraph`, `perf_report`, `assert_close`

---

### Task: `write-kernel <description>`

Write a new triton kernel based on the user's description. Follow the patterns
from the sample library and `references/gpu-tutorial/src/`:

**Pattern catalog** (reference implementations in `references/gpu-tutorial/`):

| Pattern | Reference File | Key APIs |
|---|---|---|
| Elementwise | `ch01/.../vector_add.py` | `tl.load`, `tl.store`, mask |
| GEMV | `ch02/.../gemv.py` | Row reduction, `tl.sum` |
| Reduction | `ch03/.../reduce.py` | `tl.sum`, `tl.max`, `tl.atomic_add` |
| Transpose | `ch04/.../transpose_triton.py` | 2D blocked load/store |
| GEMM | `ch05/.../gemm.py` | `tl.dot`, `@triton.autotune`, tiled loop |
| FP16 GEMM | `ch06/.../hgemm.py` | Tensor cores, fp32 accumulator |
| RMSNorm | `ch07/.../normalization_triton.py` | Row-wise reduction + scale |
| LayerNorm | `ch07/.../normalization_triton.py` | Mean/variance, eps stability |
| Softmax | `ch03/.../reduce.py` | Online max, `tl.exp`, `tl.sum` |
| TopK | `ch09/.../topk_triton.py` | Repeated max-and-mask |
| MoE GEMM | `ch10/.../moe_triton.py` | Gather/scatter, variable-size workloads |
| FlashAttention | `ch11/.../flash_attention_triton.py` | Online softmax, 2D tiling, `tl.dot` |
| PagedAttention | `ch13/.../paged_attention.py` | KV-cache, split-K |
| CrossEntropy | `ch15/.../cross_entropy.py` | Fused fwd+bwd, online softmax |
| RoPE | `ch15/.../rope.py` | In-place transform, `autograd.Function` |
| SwiGLU | `ch15/.../swiglu.py` | Fused activation, memory-efficient bwd |
| FusedFFN | `ch16/.../fused_ffn.py` | SiLU+mul epilogue fusion |
| FlashDecode | `ch18/.../flash_decode.py` | Split-KV, `torch.library.custom_op` |

**Key coding conventions:**
- Always include `mask = offs < N` for boundary safety
- Use `triton.next_power_of_2(n)` for dynamic block sizes
- Accumulate in fp32 (`tl.zeros(..., dtype=tl.float32)`) for precision
- Use `tl.constexpr` for compile-time constants (BLOCK sizes, flags)
- Write a matching test with `torch.allclose` against PyTorch reference

---

### Task: `benchmark`

Profile kernel performance:

```python
import triton

# Standard benchmark with L2 cache flush
ms = triton.testing.do_bench(
    lambda: my_kernel[grid](...),
    warmup=25,              # warmup ms
    rep=100,                # measurement ms
)

# With quantiles for statistical distribution
ms, min_ms, max_ms = triton.testing.do_bench(
    lambda: my_kernel[grid](...),
    quantiles=[0.5, 0.2, 0.8],   # median, p20, p80
)

# Compute throughput
gbps = 3 * N * 4 * 1e-9 / (ms * 1e-3)        # elementwise: 3 tensors × 4B
tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)  # matmul: 2×M×N×K FLOPs
```

**Lower host-overhead benchmarking:**
```python
ms = triton.testing.do_bench_cudagraph(lambda: my_kernel[grid](...), rep=20)
```

**Publication-quality benchmark report:**
```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'], x_vals=[2**i for i in range(12, 25)],
        line_arg='provider', line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='GB/s', plot_name='vector-add',
    ))
def benchmark(N, provider):
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    if provider == 'triton':
        fn = lambda: vector_add_kernel[(triton.cdiv(N, 256),)](x, y, torch.empty_like(x), N, BLOCK=256)
    else:
        fn = lambda: x + y
    ms = triton.testing.do_bench(fn)
    return 3 * N * 4 * 1e-9 / (ms * 1e-3)

benchmark.run(print_data=True)
```

**Occupancy analysis (pre-compile):**
```python
kernel = softmax_kernel.warmup(out, inp, *args, BLOCK_SIZE=BLOCK, num_warps=4, grid=(1,))
kernel._init_handles()
print(f"Registers: {kernel.n_regs}, Shared Memory: {kernel.metadata.shared} bytes")
```

**PTX register spill analysis:**
```powershell
$env:TRITON_DUMP_PTXAS_LOG = "1"
python my_kernel.py   # look for "lmem" warnings in output
```

---

### Task: `debug-pipeline`

Set up VS Code debugging for `triton-opt.exe`:

1. Ensure `triton-opt.exe` exists at `python/triton/_C/triton-opt.exe`
   (if not: activate vcvars, cd to cmake build dir, run `ninja triton-opt`)
2. Use these launch configs (already in `.vscode/launch.json`):
   - **triton-opt: parse vecadd** — basic parse/verify
   - **triton-opt: combine pass** — watch rewrite patterns
   - **triton-opt: TTIR → TTGIR** — watch layout assignment
   - **triton-opt: current file** — debug whatever .mlir file is open
   - **Attach to Python** — debug JIT compilation from Python

3. Key breakpoint locations:

| Stage | File | Function | What to watch |
|---|---|---|---|
| Entry | `bin/triton-opt.cpp:5` | `main()` | `argc`, `argv` |
| Registration | `bin/RegisterTritonDialects.h:~70` | `registerTritonDialects()` | All passes loaded |
| Combine | `lib/Dialect/Triton/Transforms/Combine.cpp:~283` | `runOnOperation()` | `getOperation()` |
| Combine pattern | same file ~78 | `matchAndRewrite()` | `op->getName()`, rewrite actions |
| TTIR→TTGIR | `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:~728` | `runOnOperation()` | `numWarps`, `threadsPerWarp` |
| Generic convert | same file ~38 | `GenericOpPattern::matchAndRewrite()` | op types gaining `#triton_gpu.blocked` |
| Coalesce | `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:~77` | `runOnOperation()` | Layout changes |
| TTGIR→LLVM | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:~97` | `runOnOperation()` | LLVM IR generation |

4. **Debugger watch expressions:**
   - `op->getName().getStringRef()` — operation name (`tt.load`, `arith.addf`)
   - `op->getResult(0).getType()` — result type (shows encoding after conversion)
   - `op->getNumOperands()` / `op->getNumResults()` — op shape
   - `op->getLoc()` — source location

**Debugging JIT compilation from Python:**

```powershell
# Attach VS Code debugger to a running Python kernel
python -c "import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); exec(open('my_kernel.py').read())"
# Then in VS Code: Run > Attach to Process > localhost:5678
```

**Debugging compilation errors:**
```python
from triton.compiler.errors import CompilationError

@triton.jit
def bad_kernel():
    a += 1   # undefined variable

# Catch and inspect error with source location
try:
    triton.compile(triton.compiler.ASTSource(fn=bad_kernel, signature={}, constexprs={}))
except CompilationError as e:
    print(e)   # shows "is not defined at 2:4:"
```

---

### Task: `dump-ir`

Dump intermediate IR at each compilation stage:

```powershell
# Dump IR before/after every pass (to stderr)
$env:MLIR_ENABLE_DUMP = "1"
$env:TRITON_ALWAYS_COMPILE = "1"
python my_kernel.py 2> ir_dump.txt

# Dump only passes touching a specific kernel
$env:MLIR_ENABLE_DUMP = "add_kernel"

# Save final .ttir/.ttgir/.llir/.ptx/.cubin files to a directory
$env:TRITON_KERNEL_DUMP = "1"
$env:TRITON_DUMP_DIR = "C:\Users\$env:USERNAME\.triton\dump"
$env:TRITON_ALWAYS_COMPILE = "1"
python my_kernel.py

# Override compiled kernels (edit IR, re-run without recompile)
$env:TRITON_OVERRIDE_DIR = "C:\Users\$env:USERNAME\.triton\override"
# Copy and edit .ttgir/.llir files there, triton loads them instead

# Pass timing information
$env:MLIR_ENABLE_TIMING = "1"

# PTX dump
$env:NVPTX_ENABLE_DUMP = "1"

# LLVM debug output
$env:TRITON_ENABLE_LLVM_DEBUG = "1"

# Crash reproducer (auto-saved on compiler crash)
$env:TRITON_REPRODUCER_PATH = "C:\tmp\triton_repro"
# Then replay: triton-opt.exe C:\tmp\triton_repro\crash.mlir --run-reproducer
```

**Programmatic IR inspection:**
```python
from contextlib import contextmanager
import os

@contextmanager
def dump_ir(kernel_name="1"):
    os.environ["MLIR_ENABLE_DUMP"] = kernel_name
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    try:
        yield
    finally:
        os.environ.pop("MLIR_ENABLE_DUMP", None)
        os.environ.pop("TRITON_ALWAYS_COMPILE", None)

# Usage:
with dump_ir("add_kernel"):
    add_kernel[(N // 256,)](x, y, out, N, BLOCK=256)
# IR printed to stderr — look for "IR Dump Before/After <pass-name>"
```

**Stage inspection hook (intercept compilation at any stage):**
```python
import triton

def my_hook(stages=None, options=None, **kwargs):
    original_ttgir = stages["ttgir"]
    def wrapped_ttgir(src, metadata):
        print("TTGIR input:", src[:200])  # inspect intermediate IR
        result = original_ttgir(src, metadata)
        print("TTGIR output:", result[:200])
        return result
    stages["ttgir"] = wrapped_ttgir

triton.knobs.runtime.add_stages_inspection_hook = my_hook
```

---

### Task: `run-lit [test-path]`

Run MLIR lit tests (no GPU required):

```powershell
# From the cmake build dir:
cd build\cmake.win-amd64-cpython-3.14

# Build triton-opt first
ninja triton-opt

# Run a single test
lit -v test/Triton/vecadd.mlir

# Run all tests in a directory
lit -v test/Triton/
lit -v test/TritonGPU/
lit -v test/TritonNvidiaGPU/

# Verbose with all FileCheck output
lit -v --show-all test/Triton/combine.mlir
```

Test directories:
| Directory | Content |
|---|---|
| `test/Triton/` | TTIR-level pass tests |
| `test/TritonGPU/` | TTGIR-level pass tests |
| `test/TritonNvidiaGPU/` | NVIDIA-specific passes |
| `test/Conversion/` | Dialect conversion tests |
| `test/Hopper/` | Hopper-specific tests |
| `test/Analysis/` | Analysis pass tests |

---

### Task: `explain-pipeline`

Explain what happens at each pipeline stage for a given kernel.

Read `python/triton/backends/nvidia/compiler.py` and trace through:
1. `make_ttir()` — runs canonicalize, combine, CSE, loop-unroll on TTIR
2. `make_ttgir()` — converts to TritonGPU IR, adds GPU layouts, runs coalesce, matmul acceleration, pipelining, warp-specialization
3. `make_llir()` — lowers to LLVM IR, allocates shared memory, links libdevice, runs LLVM O3
4. `make_ptx()` — LLVM PTX backend generates assembly
5. `make_cubin()` — ptxas assembles to GPU binary

---

### Task: `help`

Print available tasks and the pipeline overview.

---

## Interpreter Mode (No GPU)

Run any triton kernel on the CPU — perfect for development and debugging
without a GPU:

```powershell
$env:TRITON_INTERPRET = "1"
python my_kernel.py
# Or with pytest:
$env:TRITON_INTERPRET = "1"
pytest test_kernels.py -s --tb=short -k "not benchmark"
```

**Key behaviors:**
- Kernels run on CPU, single-threaded (THREADS_PER_WARP = 1)
- Standard Python debugger (pdb, debugpy, VS Code) can step through kernel code
- `tl.device_assert` works without `TRITON_DEBUG=1`
- `bfloat16` is **NOT supported** — skip those tests
- `get_current_target()` returns `None` — always check `is_interpreter()` first
- Performance is slow — only for correctness, not benchmarking

## In-Kernel Debug Primitives

```python
@triton.jit
def my_kernel(...):
    # Compile-time diagnostics (no runtime cost)
    tl.static_print(BLOCK_SIZE)           # print constexpr values during compilation
    tl.static_assert(BLOCK_SIZE >= 32)    # compile-time assertion

    # Runtime diagnostics (requires TRITON_DEBUG=1 or debug=True)
    tl.device_print("x_vals", x)          # print tensor on device
    tl.device_assert(offs < N, "OOB!")    # runtime assert (raises RuntimeError)

# Enable debug mode:
$env:TRITON_DEBUG = "1"                   # env var
# Or per-kernel:  @triton.jit(debug=True)
# Or per-call:    kernel[grid](..., debug=True)
```

## Key Source Files (Reading Order)

### Python Layer
1. `python/tutorials/01-vector-add.py` — simplest kernel
2. `python/triton/runtime/jit.py` — `@triton.jit`, `kernel[grid]()` dispatch
3. `python/triton/compiler/compiler.py` — `compile()` orchestrator
4. `python/triton/compiler/code_generator.py` — Python AST → TTIR
5. `python/triton/backends/nvidia/compiler.py` — NVIDIA backend stages
6. `python/triton/knobs.py` — all env var knobs (authoritative reference)
7. `python/triton/testing.py` — `do_bench`, `perf_report`, `assert_close`
8. `python/triton/_internal_testing.py` — `numpy_random`, `to_triton`, dtype lists

### C++ Layer
9. `bin/RegisterTritonDialects.h` — all registered dialects/passes overview
10. `include/triton/Dialect/Triton/IR/` — Triton dialect (tt.load, tt.store, tt.dot)
11. `include/triton/Dialect/TritonGPU/IR/` — TritonGPU dialect (layout encodings)
12. `lib/Dialect/Triton/Transforms/` — TTIR passes (Combine, LoopUnroll)
13. `lib/Dialect/TritonGPU/Transforms/` — TTGIR passes (Coalesce, Pipeline, WarpSpec)
14. `lib/Conversion/TritonToTritonGPU/` — TTIR → TTGIR conversion
15. `lib/Conversion/TritonGPUToLLVM/` — TTGIR → LLVM lowering

### Test Resources
- `.github/skills/triton-windows-dev/samples/test_kernels.py` — graduated kernel test suite
- `.github/skills/triton-windows-dev/debug-vecadd.ttir` — minimal TTIR for triton-opt debugging
- `test/Triton/vecadd.mlir` — vector add with loop
- `test/Triton/combine.mlir` — combine pass test cases
- `references/gpu-tutorial/src/` — 19 chapters of kernel examples (see catalog above)

### Upstream Test Infrastructure
- `python/test/conftest.py` — pytest fixtures (`device`, `fresh_triton_cache`)
- `python/test/unit/language/test_core.py` — 260KB comprehensive language tests
- `python/test/unit/language/test_compile_errors.py` — error message quality tests
- `python/test/unit/test_debug.py` — device_assert, TRITON_DEBUG tests
- `python/test/unit/test_debug_dump.py` — MLIR_ENABLE_DUMP tests

## Kernel Pattern Catalog (from gpu-tutorial)

| Category | Chapter | Pattern | Complexity |
|---|---|---|---|
| Elementwise | ch01 | vector add | simple |
| GEMV | ch02 | tiled row reduction | simple |
| Reduction | ch03 | sum, max, softmax | medium |
| Transpose | ch04 | 2D blocked load/store | simple |
| GEMM | ch05 | `tl.dot`, autotune, tiled loop | medium |
| Tensor Cores | ch06 | FP16 GEMM, fp32 accumulator | medium |
| Normalization | ch07 | RMSNorm, LayerNorm | medium |
| TopK / MoE | ch09-10 | max-and-mask, gather/scatter, expert GEMM | advanced |
| FlashAttention | ch11 | online softmax, 2D tiling, `tl.dot` | advanced |
| Attention variants | ch12-13 | state composition, paged KV-cache | advanced |
| Fused kernels | ch15-16 | RoPE, SwiGLU, cross-entropy, FFN fusion | advanced |
| Distributed | ch17 | AllGather+GEMM, GEMM+ReduceScatter overlap | advanced |
| Framework | ch18 | `autograd.Function`, `torch.library.custom_op` | advanced |

## Environment Variables Reference

| Variable | Effect |
|---|---|
| **IR Dumping** | |
| `MLIR_ENABLE_DUMP=1` | Print IR before/after every pass (stderr) |
| `MLIR_ENABLE_DUMP=func_name` | Filter dump to specific function |
| `TRITON_KERNEL_DUMP=1` | Save .ttir/.ttgir/.llir/.ptx/.cubin files |
| `TRITON_DUMP_DIR=path` | Output directory for kernel dump |
| `TRITON_OVERRIDE_DIR=path` | Load IR from dir instead of compiling |
| `NVPTX_ENABLE_DUMP=1` | Dump generated PTX |
| **Debug** | |
| `TRITON_DEBUG=1` | Enable `device_print`/`device_assert` |
| `TRITON_ENABLE_ASAN=1` | Address sanitizer for GPU memory |
| `TRITON_FRONT_END_DEBUGGING=1` | Extra frontend error detail |
| **Compilation** | |
| `TRITON_ALWAYS_COMPILE=1` | Bypass cache, force recompilation |
| `TRITON_INTERPRET=1` | CPU interpreter mode (no GPU) |
| `TRITON_REPRODUCER_PATH=path` | Save crash reproducers as .mlir |
| **Profiling** | |
| `MLIR_ENABLE_TIMING=1` | Print pass timing |
| `TRITON_ENABLE_LLVM_DEBUG=1` | LLVM debug output |
| `TRITON_DUMP_PTXAS_LOG=1` | ptxas log (register spills, lmem) |
| `DISABLE_PTXAS_OPT=1` | Disable ptxas optimizations |
| `PTXAS_OPTIONS="-warn-spills"` | Extra ptxas flags |
| **Autotuning** | |
| `TRITON_PRINT_AUTOTUNING=1` | Print best config per shape |
| `TRITON_CACHE_AUTOTUNING=1` | Cache autotune results across runs |
| **Cache** | |
| `TRITON_CACHE_DIR=path` | Cache location (default `~/.triton/cache`) |
| `TRITON_HOME=path` | Home for all .triton/ dirs |
| `TRITON_INSTRUMENTATION_MODE=fpsan` | Enable floating-point sanitizer |

## Windows-Specific Notes

- **Script naming**: Never name your script `triton.py` — it shadows the real module
- **Cache location**: `C:\Users\<name>\.triton\cache\` — delete when switching Python/Triton versions
- **DLL errors**: `ImportError: DLL load failed` → install latest `vc_redist.x64.exe` or delete `.triton/cache`
- **`multiprocessing` context**: `run_in_process` uses `forkserver` (unavailable on Windows) — use `spawn` instead
- **`CC` env var**: If set in Windows Environment Variables, must be a plain string, not a list (lists add trailing `;`)
- **Path length**: Use short paths or enable Win32 long paths to avoid build issues
- **Interpreter + tl.dot**: `tl.dot` in interpreter mode crashes on Windows (OpenMP duplicate library) — skip matmul tests when `TRITON_INTERPRET=1`

## Compiler Pass Coverage Map

Shows which compilation passes are exercised by each test kernel.

### Always-run passes (exercised by ALL kernels)
- `inliner`, `canonicalize`, `cse`, `symbol-dce` (make_ttir)
- `convert-triton-to-tritongpu`, `coalesce`, `remove-layout-conversions` (make_ttgir)
- `plan-cta`, `reduce-data-duplication`, `reorder-instructions` (make_ttgir)
- `allocate-shared-memory`, `scf-to-cf` (make_llir)
- `to-llvmir`, `canonicalize-llvm-ir`, `llvm.optimize_module(O3)` (make_llir)

### Passes triggered by specific kernel patterns

| Pass | Triggered By | Test Kernel |
|---|---|---|
| `combine` (TTIR simplification) | Algebraic patterns | all (basic patterns) |
| `reorder-broadcast` | Broadcast-heavy shapes | `broadcast_add` |
| `loop-unroll` | Fixed-trip loops | `matmul`, `pipelined_matmul` |
| `accelerate-matmul` | `tl.dot` | `matmul`, `pipelined_matmul` |
| `f32-dot-tc` | `tl.dot` on SM80+ | `matmul`, `pipelined_matmul` |
| `optimize-dot-operands` | `tl.dot` on SM80+ | `matmul`, `pipelined_matmul` |
| `lower-mma` | `tl.dot` | `matmul`, `pipelined_matmul` |
| `fuse-nested-loops` | Loop nests on SM80+ | `pipelined_matmul` |
| `triton-licm` | Loop-invariant code | `pipelined_matmul` |
| `pipeline` | `tl.range(num_stages=N)` | `pipelined_matmul` |
| `schedule-loops` | Pipelined loops | `pipelined_matmul` |
| `combine-tensor-select-and-if` | `tl.where`, `if/else` | `relu_dropout` |
| `fence-insertion` | Memory ops | all with load/store |

### LLVM lowering paths exercised

| Lowering | API | Test Kernel |
|---|---|---|
| `ElementwiseOpToLLVM` | arith ops | `vector_add`, `fma`, `gelu` |
| `ReduceOpToLLVM` | `tl.sum`, `tl.max` | `reduce_sum`, `softmax`, `reduce_max` |
| `ScanOpToLLVM` | `tl.associative_scan` | `cumsum` |
| `MemoryOpToLLVM` | `tl.load`, `tl.store` | all |
| `DotOpToLLVM` | `tl.dot` | `matmul`, `pipelined_matmul` |
| `SPMDOpToLLVM` | `tl.program_id` | all |
| `MakeRangeOpToLLVM` | `tl.arange` | all |
| `ViewOpToLLVM` | layout transforms | `transpose` |
| `PrintOpToLLVM` | `tl.device_print` | (manual debug) |
| `AssertOpToLLVM` | `tl.device_assert` | (manual debug) |

### Passes NOT exercised (require specific hardware or features)
- `warp-specialize` — requires SM90+ (Hopper) or explicit annotation
- `tma-lowering` — requires SM90+ with tensor descriptors
- `optimize-tmem-layouts`, `hoist-tmem-alloc` — requires SM100+ (Blackwell)
- `promote-lhs-to-tmem`, `remove-tmem-tokens` — requires SM100+
- `prefetch` — only on SM80 (Ampere)
- `HistogramOpToLLVM` — requires `tl.histogram` (niche)
- `GatherOpToLLVM` — requires `tl.gather` (niche)
- `global-sanitizer` — requires `TRITON_INSTRUMENTATION_MODE=gsan`
- `fp-sanitizer` — requires `TRITON_INSTRUMENTATION_MODE=fpsan`
