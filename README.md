# Triton Vulkan/SPIR-V Backend

An experimental Vulkan/SPIR-V backend for [Triton](https://github.com/triton-lang/triton),
enabling GPU compute on any Vulkan-capable device — NVIDIA, AMD, Intel, or mobile.

> For the standard CUDA/NVIDIA Triton experience on Windows (pip install, PyTorch
> integration, troubleshooting), see [triton-windows](https://github.com/triton-lang/triton-windows).

## What This Does

Triton normally compiles GPU kernels through: **TTIR → TTGIR → LLVM IR → PTX → cubin** (NVIDIA only).

This backend takes a different route: **TTIR → Linalg → MemRef → SPIR-V → Vulkan** — portable across any GPU with Vulkan drivers.

```
@triton.jit kernel
    │
    ▼
TTIR (Triton IR)                        ← shared with NVIDIA backend
    │ 16 TritonToLinalg converters
    ▼
Linalg / Tensor / MemRef                ← standard MLIR dialects
    │ bufferize → loops → control flow
    ▼
MemRef / Arith / CF                     ← fully lowered
    │ 7 bridge passes + SPIR-V conversion + VulkanizePass
    ▼
SPIR-V binary (.spv)                    ← Vulkan compute shader
    │ VulkanCompute runtime
    ▼
GPU result                              ← via vkCmdDispatch
```

### Verified Results

Multiple kernel types, verified on RTX 2080 Ti via native Vulkan compute:

| Kernel | What it tests | Error |
|--------|--------------|-------|
| vector_add | Elementwise `a + b` | 0.00 (exact) |
| fma | `a * b + c` | 0.00 (exact) |
| gelu | Activation: `x / (1 + exp(-1.702x))` | 2.38e-07 |
| swiglu | Gated: `x · σ(x) · gate` | 4.77e-07 |
| reduce_sum | `sum(x)` over 256 elements | 5.72e-06 |
| reduce_max | `max(x)` | 0.00 (exact) |
| softmax | `exp(x-max) / sum(exp(x-max))` | 5.59e-09 |
| matmul | 16×16 matrix multiply | 1.91e-06 |
| transpose | 16×16 transpose | 0.00 (exact) |
| vadd_multiblock | Multi-block dispatch (1024 elements, 4 workgroups) | 0.00 (exact) |
| vadd_65k | Large tensor (65536 elements, 256 workgroups) | 0.00 (exact) |
| matmul_coop_f16 | Cooperative matrix 16×16 (f16 via tensor cores) | ~1e-3 (f16 precision) |

---

## Getting Started

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10 or 11 |
| **GPU** | Any with Vulkan 1.0+ (NVIDIA, AMD, Intel) |
| **C++ compiler** | Visual Studio 2022+ with MSVC v14.44+ |
| **Python** | 3.10+ (check conda for latest compatible version) |
| **Vulkan SDK** | Headers + lib (conda: `conda install vulkan-headers vulkan-loader`) |

### Step 1: Build LLVM and Triton

```powershell
git clone https://github.com/triton-lang/triton-windows.git
cd triton-windows
# See BUILD.md or .github/skills/triton-windows-build/SKILL.md for full LLVM build steps
```

### Step 2: Build the Vulkan backend

```powershell
# Activate MSVC (required in every new terminal)
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul 2>&1 && set' | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

# Build (Vulkan SDK auto-detected from conda or VULKAN_SDK env var)
$env:PYTHONPATH = ".\python"
$buildDir = python -c "from build_helpers import get_cmake_dir; print(get_cmake_dir())"
cd $buildDir
cmake --build . --target triton
```

### Step 3: Test

```powershell
Copy-Item third_party/vulkan/backend/*.py python/triton/backends/vulkan/ -Force
$env:TRITON_BACKENDS_IN_TREE = "1"
python third_party/vulkan/test/test_kernels_vulkan.py
```

Expected: `All tests PASS`

---

## Tutorial: Your First Vulkan Kernel

### 1. Write a kernel in Triton IR

Save as `test_fma.ttir` — computes `out[i] = a[i] * b[i] + c[i]`:

```mlir
module {
  tt.func public @fma_kernel(
      %a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>,
      %c_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32) {
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %ab = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %ap = tt.addptr %ab, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %a = tt.load %ap : tensor<256x!tt.ptr<f32>>
    %bb = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %bp = tt.addptr %bb, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %b = tt.load %bp : tensor<256x!tt.ptr<f32>>
    %cb = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %cp = tt.addptr %cb, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %c = tt.load %cp : tensor<256x!tt.ptr<f32>>
    %mul = arith.mulf %a, %b : tensor<256xf32>
    %add = arith.addf %mul, %c : tensor<256xf32>
    %ob = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %op = tt.addptr %ob, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %op, %add : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
```

### 2. Compile to SPIR-V binary

```python
from triton._C.libtriton import ir, passes, vulkan
from triton.backends.vulkan.compiler import VulkanBackend, GPUTarget

ctx = ir.context(); ir.load_dialects(ctx); vulkan.load_dialects(ctx)
mod = ir.parse_mlir_module("test_fma.ttir", ctx)
mod.context = ctx  # required — pass_manager reads this attribute

backend = VulkanBackend(GPUTarget("vulkan", 0, 32))
opt = backend.parse_options({}); md = {}

mod = backend.make_ttir(mod, md, opt)    # optimize
mod = backend.make_linalg(mod, md, opt)  # Triton → Linalg (16 converters)
mod = backend.make_memref(mod, md, opt)  # bufferize + loops + control flow
mod = backend.make_spirv(mod, md, opt)   # bridge passes + SPIR-V convert + vulkanize
spv = backend.make_spv(mod, md, opt)     # serialize to .spv binary

print(f"SPIR-V: {len(spv)} bytes, kernel: {md['name']}")
```

### 3. Dispatch on GPU via Vulkan

```python
import numpy as np

vc = vulkan.runtime.VulkanCompute()
print(f"GPU: {vc.device_name()}")  # e.g. "NVIDIA GeForce RTX 2080 Ti"

vc.load_shader(spv, md["name"])
vc.set_workgroups(1)

N = 256
buf_a   = vc.create_buffer(0, N * 4)  # binding 0
buf_b   = vc.create_buffer(1, N * 4)
buf_c   = vc.create_buffer(2, N * 4)
buf_out = vc.create_buffer(3, N * 4)

a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.random.randn(N).astype(np.float32)
vc.write_buffer(buf_a, a)
vc.write_buffer(buf_b, b)
vc.write_buffer(buf_c, c)

# Push constants: scalar args (N, num_programs_xyz, program_id_xyz)
vc.set_push_constants(np.array([N, 1, 1, 1, 0, 0, 0], dtype=np.int32))

vc.dispatch()

out = np.zeros(N, dtype=np.float32)
vc.read_buffer(buf_out, out)
print(f"Max error: {np.max(np.abs(out - (a * b + c)))}")
# Output: Max error: 0.0
```

### 4. Debug with IR dumps

Inspect the MLIR IR at any pipeline stage:

```python
mod = backend.make_linalg(mod, md, opt)
print(mod.str_nodebug())  # clean IR, no location annotations
```

---

## How the Pipeline Works

### `make_linalg` — 16 TritonToLinalg Converters

| Triton Op | → MLIR Op | Notes |
|-----------|-----------|-------|
| `tt.splat` | `linalg.fill` / memref passthrough | pointer vs scalar |
| `tt.make_range` | `linalg.generic` + `linalg.index` | |
| `tt.addptr` | `memref.reinterpret_cast` | PtrState def-chain walk |
| `tt.load` | `alloc` + `fill` + `memref.copy` + `to_tensor` | |
| `tt.store` | `materialize_in_destination` | |
| `tt.dot` | `linalg.matmul` | zero-init + accumulation |
| `tt.reduce` | `linalg.reduce` | cloned combiner region |
| `tt.broadcast` | `linalg.generic` with broadcast map | |
| `tt.expand_dims` | `tensor.expand_shape` | |
| `tt.trans` | `linalg.transpose` | |
| `tt.reshape` | `tensor.expand/collapse_shape` | |
| `tt.bitcast` | `arith.bitcast` | |
| `tt.get_program_id` | function arg (injected by `addProgramInfo`) | |
| `tt.get_num_programs` | function arg | |
| `tt.atomic_rmw` | `scf.for` with memref load-modify-store | |
| dense splat constants | `tensor.empty` + `linalg.fill` | |

### `make_spirv` — 7 Bridge Passes + VulkanizePass

The upstream MLIR `convert-*-to-spirv` passes assume structured IR. Triton's
pointer-heavy IR needs bridge passes first (same approach as IREE and Intel XPU):

| Bridge Pass | What it solves |
|-------------|---------------|
| ExpandReinterpretCast | `reinterpret_cast` → direct load/store + offset |
| ExpandMemRefCopy | `memref.copy` → `scf.for` loop |
| ExpandExpandShape | 2D reshape → linearized 1D access |
| Flatten2DAllocs | `memref<MxN>` → `memref<M*N>` |
| alloc → alloca | heap → stack allocation |
| Unranked → Ranked | `memref<*xT>` → `memref<?xT>` |
| spirv.target_env | attach capability requirements |

**VulkanizePass** (the key innovation): converts `spirv.func` into a complete
Vulkan compute shader module. Handles 9 responsibilities: buffer args →
GlobalVariables with descriptor bindings, WorkgroupId/LocalInvocationId builtins,
push constants, shared memory promotion, barrier insertion, subgroup reductions,
cooperative matrix buffer-forwarding, and module/entry-point wrapping.

---

## Project Structure

```
third_party/vulkan/
├── backend/
│   ├── compiler.py              # Pipeline: make_ttir → make_spv
│   ├── emitter.py               # OpenCL C emitter (optional debug)
│   └── emitter_parallel.py      # Parallel OpenCL (optional debug)
├── lib/
│   ├── Conversion/
│   │   ├── TritonToLinalg.cpp   # 16 converters (~1700 lines)
│   │   ├── TritonToLinalgPass.cpp
│   │   └── PrepareSPIRV.cpp     # Bridge passes + VulkanizePass + C+ passes (the largest backend file)
│   └── Runtime/
│       └── VulkanCompute.cpp    # the Vulkan dispatch engine
├── triton_vulkan.cc             # pybind11 module
├── CMakeLists.txt
└── test/
    ├── test_kernels_vulkan.py   # Vulkan GPU test suite (primary)
    ├── test_kernels.py          # OpenCL test suite (optional)
    └── *.ttir                   # test kernels
```

## Skills (Developer Docs)

`.github/skills/` has detailed technical guides:

| Skill | Content |
|-------|---------|
| `triton-windows-build` | LLVM + Triton build on Windows (MSVC patches, vcvars, build scripts) |
| `triton-windows-vulkan` | Base backend: 16 converters, 7 bridge passes, VulkanizePass, runtime API |
| `triton-windows-vulkan-perf` | C+1–C+5 performance roadmap, documented traps, strategy lessons |
| `triton-windows-dev` | Testing, debugging, profiling |
| `triton-windows-opencl` | Optional OpenCL debug emitters |

## Development Docs

`development/` has textbook-style deep technical guides:

| Document | Content |
|----------|---------|
| `vulkan-backend-guide.md` | Complete 21-section guide: architecture, converters, bridge passes, C+ journey |
| `intel-xpu-backend-study.md` | Path A vs Path C+ feasibility analysis, Intel XPU backend research |
| `opencl-emitter-guide.md` | Parallel OpenCL emitter internals (debugging aid, not primary path) |

## Roadmap

| Feature | Status | Description |
|---------|--------|-------------|
| TTIR→Linalg converters | ✅ | 16 converters: splat, range, broadcast, reduce, matmul, load/store, atomics |
| SPIR-V bridge passes | ✅ | 7 passes resolving MLIR→SPIR-V gaps (reinterpret_cast, copy, expand_shape, etc.) |
| Native Vulkan dispatch | ✅ | VulkanizePass + VulkanCompute runtime, all current tests pass |
| WorkgroupId parallel dispatch | ✅ | program_id via SPIR-V WorkgroupId builtin (C+1) |
| Device-local memory | ✅ | Staging buffers + vkCmdCopyBuffer (C+2) |
| Shared memory reductions | ✅ | Workgroup storage class + tree reduction + ControlBarrier (C+3) |
| Subgroup operations | ✅ | `OpGroupNonUniform*` for fast intra-warp reductions (C+4) |
| Cooperative matrix | ✅ | `VK_KHR_cooperative_matrix` + buffer-forwarding for matmul (C+5) |
| Discrete GPU selection | 🔲 | Prefer `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` (C+6) |

> **Why not TritonGPU → LLVM → SPIR-V (Path A)?** Intel's TTGIR→LLVM is 80%
> Intel-specific (DPAS, 2D block loads). NVIDIA's is equally PTX-locked (~7K
> lines of inline asm). Forking either requires 3-6 months and the performance
> features (TMA, async copy, wgmma) don't exist on Turing GPUs. Path C+
> enhances our working MLIR SPIR-V pipeline with Vulkan features the hardware
> actually supports. See `development/intel-xpu-backend-study.md` for analysis.