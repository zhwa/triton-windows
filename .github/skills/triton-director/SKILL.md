---
name: triton-director
description: "Triton compiler pipeline inspection, debugging, and development tools. Maps every pipeline stage (AST→TTIR→TTGIR→LLVM→PTX→cubin), lists upstream/custom tools for each, and provides inspector/timer scripts. Use for: understanding pipeline stages, debugging pass failures, timing compilation, capturing IR at each stage, or learning what tools professional triton developers use."
argument-hint: "help | scan | init | inspect | time | env | test"
user-invocable: true
---

# Triton Director — Compiler Pipeline Development Toolkit

Complete guide to the Triton compiler pipeline stages, available debugging
tools, and custom inspection scripts for triton-windows development.

## Quick Start (for beginners)

The `director.py` CLI is the single entry point for all development tasks:

```powershell
$env:TRITON_BACKENDS_IN_TREE = "1"

# What's in this repo?
python .github\skills\triton-director\scripts\director.py scan

# How do I build from scratch?
python .github\skills\triton-director\scripts\director.py init

# Is my environment set up correctly?
python .github\skills\triton-director\scripts\director.py env --check

# Inspect a kernel's compilation pipeline
python .github\skills\triton-director\scripts\director.py inspect test.ttir --stats

# Time each pass
python .github\skills\triton-director\scripts\director.py time test.ttir

# Run the GPU test suite
python .github\skills\triton-director\scripts\director.py test
```

### CLI Commands

| Command | What It Does |
|---------|-------------|
| `help` | List all commands with usage examples |
| `scan` | Discover skills, tools, docs, and test kernels in the repo |
| `init` | Step-by-step guide from fresh clone to 12/12 PASS |
| `env [--check]` | Show/verify environment (Python, conda, MSVC, triton, Vulkan) |
| `inspect <ttir> [--stats] [--diff] [--passes] [--out DIR] [--json]` | Capture IR at every stage |
| `time <ttir...> [--format table\|csv\|json] [--runs N]` | Per-pass compilation timing |
| `test` | Run the Vulkan GPU test suite |

---

## 1. The Triton Compilation Pipeline

```
Python @triton.jit kernel
    ↓ [Frontend: AST → code_generator.py]
TTIR  (Triton Tensor IR — platform-independent)
    ↓ [Bridge: convert-triton-to-tritongpu]
TTGIR (TritonGPU IR — layout-annotated, target-aware)
    ↓ [Optimization: coalesce → propagate → remat → accelerate_matmul]
TTGIR (optimized — MMA encoding, coalesced memory, propagated layouts)
    ↓ [Lowering: convert-tritongpu-to-llvm]
LLVM IR (nvvm dialect + vendor PTX inline asm)
    ↓ [Backend: mlir-translate → ptxas]
PTX → cubin → SASS (GPU binary)
```

### For Vulkan backend (triton-windows):
```
TTIR
    ↓ [triton-to-linalg: 16 converters]
Linalg/Tensor/MemRef IR
    ↓ [bufferize + loops + cooperative matrix]
MemRef IR (fully lowered)
    ↓ [bridge passes + convert-*-to-spirv + vulkanize]
SPIR-V IR
    ↓ [serialize]
SPIR-V binary → VulkanCompute dispatch
```

---

## 2. Pipeline Stages — Detailed Reference

### Stage 1: Frontend (Python → TTIR)

| Component | File | What It Does |
|-----------|------|-------------|
| `@triton.jit` decorator | `python/triton/runtime/jit.py` | Captures kernel function, builds AST |
| `code_generator.py` | `python/triton/compiler/code_generator.py` | AST → TTIR via builder pattern |
| `ir.py` | `python/triton/compiler/ir.py` | Type system, TTIR op constructors |

**Key ops produced:** `tt.load`, `tt.store`, `tt.dot`, `tt.addptr`, `tt.splat`,
`tt.make_range`, `tt.get_program_id`, `tt.reduce`, `tt.expand_dims`, `tt.broadcast`

**How to inspect:**
```python
# After JIT compilation
ccinfo = triton.compile(kernel, signature=..., ...)
print(ccinfo.asm['ttir'])  # TTIR as text
```

### Stage 2: TTIR Optimization

| Pass | What It Does |
|------|-------------|
| `inliner` | Inline function calls |
| `canonicalizer` | Fold constants, simplify patterns |
| `combine` | Triton-specific combining (e.g., addptr chains) |
| `reorder_broadcast` | Move broadcasts closer to consumers |
| `cse` | Common subexpression elimination |
| `symbol_dce` | Dead code elimination |
| `loop_unroll` | Unroll loops with known bounds |

### Stage 3: TTIR → TTGIR (NVIDIA path only)

| Pass | What It Does |
|------|-------------|
| `convert-triton-to-tritongpu` | Add BlockedEncoding to all tensors |
| `coalesce` | Reorder thread layout for memory coalescing |
| `remove-layout-conversions` | Propagate layouts to remove redundant converts |
| `rewrite-tensor-pointer` | Lower tensor pointers |
| `accelerate-matmul` | BlockedEncoding → MMAEncoding for `tt.dot` |
| `pipeline` | Software pipeline shared memory loads |

**How to inspect:**
```python
print(ccinfo.asm['ttgir'])  # TTGIR as text
```

### Stage 4: TTGIR → LLVM (NVIDIA path only)

| Pass | What It Does |
|------|-------------|
| `convert-tritongpu-to-llvm` | Lower TTGIR ops to LLVM IR + nvvm intrinsics |
| `allocate-shared-memory` | Compute shared memory layout |
| `convert-nv-gpu-to-llvm` | Lower NV-specific ops (TMA, wgmma) |

### Stage 5: LLVM → PTX → cubin (NVIDIA path only)

| Tool | What It Does |
|------|-------------|
| `mlir-translate --mlir-to-llvmir` | MLIR LLVM dialect → LLVM IR |
| `llc` | LLVM IR → PTX assembly |
| `ptxas` | PTX → cubin GPU binary |
| `cuobjdump -sass` | cubin → SASS disassembly |

### Vulkan-Specific Stages (see `triton-windows-vulkan` skill)

| Stage | Passes | Input → Output |
|-------|--------|----------------|
| `linalg` | `triton-to-linalg` (16 converters) | TTIR → Linalg/Tensor/MemRef |
| `memref` | bufferize, reduction→parallel, matmul→coop, loops, affine, scf→cf | Linalg → MemRef+CF |
| `spirv_prep` | prepare_spirv, scf→cf, canonicalize | MemRef → bridge-expanded MemRef |
| `spirv_map` | map-storage-class, fix-alloca | MemRef → storage-class-mapped |
| `spirv_convert` | convert-{memref,arith,math,cf,func}-to-spirv | MemRef → SPIR-V dialect |
| `vulkanize` | VulkanizePass | SPIR-V func → Vulkan module |

---

## 3. Upstream Triton Tools

### Environment Variables (set before running)

| Variable | Effect |
|----------|--------|
| `MLIR_ENABLE_DUMP=1` | Dump IR before every MLIR pass (stderr) |
| `MLIR_ENABLE_DUMP=kernelName` | Dump for a specific kernel only |
| `MLIR_ENABLE_TIMING=1` | Per-pass wall time report |
| `LLVM_ENABLE_TIMING=1` | Per-LLVM-pass timing |
| `TRITON_KERNEL_DUMP=1` | Dump IR at all stage boundaries |
| `TRITON_DUMP_DIR=<path>` | Output dir for kernel dumps |
| `TRITON_KERNEL_OVERRIDE=1` | Re-inject modified IR from `TRITON_OVERRIDE_DIR` |
| `TRITON_OVERRIDE_DIR=<path>` | Dir with modified `.ttir`/`.ttgir`/`.llir`/`.ptx` |
| `TRITON_ALWAYS_COMPILE=1` | Bypass cache (force recompile) |
| `TRITON_INTERPRET=1` | CPU interpreter mode (numpy-based, supports pdb) |
| `TRITON_DEBUG=1` | Enable `device_assert` / `device_print` |
| `TRITON_FRONT_END_DEBUGGING=1` | Full tracebacks (no exception wrapping) |
| `TRITON_REPRODUCER_PATH=<path>` | Generate MLIR crash reproducers |
| `TRITON_DUMP_PTXAS_LOG=1` | Show ptxas register/shared mem usage |
| `DISABLE_LLVM_OPT` | Skip LLVM optimization passes |
| `USE_IR_LOC={ttir,ttgir}` | Remap source locations for profiler attribution |

### Kernel Override Workflow (edit IR at any stage)

```bash
# Step 1: Dump all stages
$env:TRITON_ALWAYS_COMPILE = "1"
$env:TRITON_KERNEL_DUMP = "1"
$env:TRITON_DUMP_DIR = "./kernel_stages"
python my_kernel.py

# Step 2: Copy and edit
cp kernel_stages/<hash>/ override_dir/
# Edit override_dir/<hash>/xxx.ttgir

# Step 3: Re-inject
$env:TRITON_KERNEL_OVERRIDE = "1"
$env:TRITON_OVERRIDE_DIR = "./override_dir"
python my_kernel.py  # uses your modified IR
```

### Python API (programmatic access)

```python
# After compilation — all IR as strings
ccinfo = triton.compile(src, signature=..., ...)
ccinfo.asm['ttir']    # Triton IR
ccinfo.asm['ttgir']   # TritonGPU IR
ccinfo.asm['llir']    # LLVM IR
ccinfo.asm['ptx']     # PTX assembly
ccinfo.asm['cubin']   # cubin binary
ccinfo.asm['sass']    # SASS (lazily disassembled)

# Compilation timing listener
from triton import knobs
def my_listener(*, src, metadata, metadata_group, times, cache_hit):
    print(f"Total: {times.total}µs, cache_hit={cache_hit}")
    for stage, duration in times.lowering_stages:
        print(f"  {stage}: {duration}µs")
knobs.compilation.listener = my_listener

# Pipeline stage hook
def inspect_stages(_self, stages, options, language, capability):
    print(list(stages.keys()))  # ['ttir', 'ttgir', 'llir', 'ptx', 'cubin']
knobs.runtime.add_stages_inspection_hook = inspect_stages
```

### In-Kernel Debugging

```python
@triton.jit
def kernel(...):
    tl.static_print(x.shape)         # compile-time shape info
    tl.static_assert(BLOCK > 0)      # compile-time assert
    tl.device_print("val", x)        # runtime print (needs TRITON_DEBUG=1)
    tl.device_assert(x > 0, "neg!")  # runtime assert (needs TRITON_DEBUG=1)
```

### CPU Interpreter Mode

```bash
$env:TRITON_INTERPRET = "1"
python my_kernel.py          # runs on CPU with numpy
# Supports pdb breakpoints inside @triton.jit kernels!
```

---

## 4. GPU Profiling Tools

### NVIDIA

| Tool | Command | Use Case |
|------|---------|----------|
| Nsight Systems | `nsys profile python script.py` | System timeline: kernel launches, transfers, SM utilization |
| Nsight Compute | `ncu --set full python script.py` | Per-kernel hardware counters, occupancy, memory BW |
| compute-sanitizer | `compute-sanitizer --tool memcheck python script.py` | OOB, race, uninit detection |
| cuobjdump | `cuobjdump -sass kernel.cubin` | SASS disassembly |
| nvdisasm | `nvdisasm kernel.cubin` | Detailed SASS with control codes |

**Pro tip:** Use `USE_IR_LOC=ttgir` with Nsight Compute to get source attribution
at the TTGIR level instead of Python line numbers.

### Triton Proton Profiler

```python
from triton.profiler import start, finalize
start("session", context="shadow", backend="cupti")
# ... run kernels ...
finalize()  # saves profile data
```

### Community: triton-viz

```bash
pip install triton-viz
triton-sanitizer script.py     # OOB symbolic checking (no GPU needed)
triton-profiler script.py      # load/store profiling
triton-visualizer trace.tvz    # web UI for access patterns
```

---

## 5. Built-in Triton Tools

| Tool | Location | Purpose |
|------|----------|---------|
| `disasm.py` | `python/triton/tools/disasm.py` | SASS disassembler with control encoding |
| `compile.py` | `python/triton/tools/compile.py` | AOT ahead-of-time kernel compiler |
| `link.py` | `python/triton/tools/link.py` | Link AOT-compiled kernel objects |
| `gsan.py` | `python/triton/tools/gsan.py` | GPU memory sanitizer |

---

## 6. Custom Inspector & Timer Scripts

This skill includes two Python scripts for Vulkan pipeline inspection.
They live in `.github/skills/triton-director/scripts/`.

### `inspector.py` — Pipeline IR Capture & Diff

Captures IR at every Vulkan pipeline stage boundary.

```powershell
$env:TRITON_BACKENDS_IN_TREE = "1"

# Basic — stage summary table
python .github\skills\triton-director\scripts\inspector.py test.ttir

# With dialect statistics (shows which ops remain at each stage)
python .github\skills\triton-director\scripts\inspector.py test.ttir --stats

# With diffs between consecutive stages
python .github\skills\triton-director\scripts\inspector.py test.ttir --diff

# Per-pass IR capture (slower — runs each pass individually)
python .github\skills\triton-director\scripts\inspector.py test.ttir --passes

# Write IR files to directory (one .mlir per stage + manifest.json)
python .github\skills\triton-director\scripts\inspector.py test.ttir --out ./ir_dumps

# Machine-readable JSON
python .github\skills\triton-director\scripts\inspector.py test.ttir --json
```

**Example output (--stats):**
```
Stage              Time   Lines    Bytes  Top Dialects
----------------------------------------------------------------------
input             0.0ms      23     1145  tt:30, arith:5
ttir             11.1ms      23     1145  tt:30, arith:5
linalg            9.8ms      31     2174  arith:9, memref:7, linalg:4
memref            9.5ms      49     2435  arith:13, memref:13, cf:9
spirv_prep        7.6ms      68     3012  arith:22, cf:14, memref:13
spirv_map         1.4ms      68     3554  arith:22, spirv:20, cf:14
spirv_convert    20.6ms      78     5092  spirv:124
vulkanize         1.3ms     114     8465  spirv:189
----------------------------------------------------------------------
Total            61.3ms
```

**What to look for:**
- After `linalg`: no `tt.*` ops remain (all converted)
- After `memref`: no `linalg.*` or `tensor.*` ops remain
- After `spirv_convert`: only `spirv.*` ops (+ maybe stale `memref.store` on i1)
- After `vulkanize`: valid `spirv.module` with `EntryPoint`

### `timer.py` — Per-Pass Timing

Reports wall time for every individual pass in the pipeline.

**Caveat:** Each pass runs in its own PassManager instance, so per-pass
times include PM creation overhead (~0.5ms each). Total will be slightly
higher than batched execution. Use `inspector.py` (without `--passes`)
for representative total stage times.

```powershell
$env:TRITON_BACKENDS_IN_TREE = "1"

# Table with bar chart
python .github\skills\triton-director\scripts\timer.py test.ttir

# Batch all kernels to CSV (good for spreadsheet analysis)
python .github\skills\triton-director\scripts\timer.py third_party\vulkan\test\test_*.ttir --format csv

# Average over 5 runs (more stable numbers)
python .github\skills\triton-director\scripts\timer.py test.ttir --runs 5

# Machine-readable JSON
python .github\skills\triton-director\scripts\timer.py test.ttir --format json
```

**Example output:**
```
test_vector_add (total: 69.7ms, 1 run)
  Stage           Pass                           Time      %
  --------------- ------------------------------ -------- ------
  ttir            inliner                           6.4ms   9.2% ████
  ttir            canonicalizer                     2.1ms   2.9% █
  linalg          triton_to_linalg                 10.5ms  15.0% ███████
  memref          linalg_to_loops                   2.5ms   3.5% █
  spirv_convert   convert_arith_to_spirv            4.2ms   6.0% ███
  ...
                  TOTAL                            69.7ms
```

### Script Dependencies

Both scripts require:
- Python with triton-windows installed (`pip install -e .` in an env)
- `TRITON_BACKENDS_IN_TREE=1` for Vulkan backend
- Input: `.ttir` files (test kernels in `third_party/vulkan/test/`)

### Adding New Scripts

To add a new tool, create a Python file in `scripts/` that imports `_common.py`:
```python
import sys, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from _common import load_ttir, get_ir_text, count_ops, get_vulkan_backend
```

---

## 7. Stage-by-Stage Debugging Playbook

### "My kernel compiles but gives wrong results"

1. **Run inspector with diff**: identify which stage changes the computation
   ```
   python .github\skills\triton-director\scripts\inspector.py kernel.ttir --diff
   ```
2. **Dump IR files**: examine each stage in detail
   ```
   python .github\skills\triton-director\scripts\inspector.py kernel.ttir --out ./debug
   ```
3. **Check for mask issues**: search for `memref.copy` in MemRef stage (may ignore mask)
4. **Use interpreter**: `TRITON_INTERPRET=1` to run on CPU with pdb

### "Compilation crashes at a specific pass"

1. **Run inspector**: it catches errors and shows which pass failed
2. **Set reproducer**: `$env:TRITON_REPRODUCER_PATH = "./repro.mlir"`
3. **Run triton-opt** on the dumped IR with the failing pass:
   ```
   triton-opt.exe --prepare-spirv repro.mlir
   ```
4. **Use --debug** for verbose MLIR pattern matching output

### "Compilation is slow"

1. **Run timer**: identify the slowest passes
   ```
   python .github\skills\triton-director\scripts\timer.py kernel.ttir --runs 3
   ```
2. **Common bottlenecks**: `spirv_convert` (6 passes), `canonicalizer` (runs 3x)
3. **Use MLIR timing**: `$env:MLIR_ENABLE_TIMING = "1"` for upstream pass detail

### "I need to test a modified IR"

1. **Dump**: `inspector.py kernel.ttir --out ./stages`
2. **Edit**: modify `stages/03_memref.mlir`
3. **Re-inject**: load the modified IR and run remaining stages:
   ```python
   from _common import load_ttir
   m, c = load_ttir("stages/03_memref.mlir")
   # ... run spirv_prep, spirv_convert, vulkanize manually
   ```

---

## 8. Key Metrics to Track

| Metric | Where | Healthy Range |
|--------|-------|---------------|
| Total compile time | timer.py | 50-200ms per kernel |
| `spirv_convert` % | timer.py | <40% of total |
| Non-SPIR-V ops after convert | inspector --stats | Must be 0 |
| `spirv.module` ops after vulkanize | inspector --stats | Should dominate |
| Dispatch time (µs) | test_kernels_vulkan.py | <1000µs for 256-element |
| Test pass rate | test_kernels_vulkan.py | 12/12 |

---

## 9. Adapting to Upstream Changes

- **New passes added upstream**: Add to `get_stage_defs()` in
  `_common.py` (search for the stage name — both inspector.py and timer.py
  import from there)
- **Pass API changes**: The tools use `ir.pass_manager` + `passes.*` bindings.
  If a pass is renamed, search for the old name in `python/src/passes.cc`
  to find the new binding
- **New pipeline stages**: Add a new entry to `stage_defs` between the
  appropriate existing stages
- **New env vars**: Check `python/triton/knobs.py` — it's the canonical
  registry of all Triton configuration knobs
