---
name: triton-windows-build
description: "Build triton-windows on Windows with Visual Studio 2026. Handles LLVM cloning, patching, building, source inspection, and triton compilation. Use for: clone-llvm, prebuild-inspect, build-llvm, build-triton, or any triton-windows build question."
argument-hint: "clone-llvm | prebuild-inspect | build-llvm [release|debug] | build-triton | help"
user-invocable: true
---

# Triton-Windows Build Skill

You are an expert at building triton-windows on Windows with Visual Studio 2026.
Parse the user's argument to determine which task to perform.

## Prerequisites

- **Visual Studio 2026** with MSVC v143 toolset (14.44) installed
- **Conda env** `mlir-dev` with cmake, ninja: `conda create -n mlir-dev -c conda-forge ninja cmake`
- **Python** 3.10+ (3.14 free-threaded works but shows GIL warnings)
- **CUDA Toolkit** installed (for NVPTX backend)

## Directory Layout

External dependencies live in `build/` (git-ignored):
- `build/llvm-project/` — LLVM/MLIR source and build
- `build/json/` — nlohmann/json headers
- `build/cmake.win-amd64-cpython-*` — triton cmake build (created by pip)

Build scripts live in TWO locations (identical copies):
- `.github/skills/triton-windows-build/scripts/` — canonical, tracked by git
- `development/` — working copies (may be untracked)

Prefer the `.github/skills/` copies since they always exist after clone.

## PowerShell vcvars Persistence

**CRITICAL:** Running `vcvars64.bat` directly in PowerShell does NOT persist
the environment. You MUST use this pattern:

```powershell
$f = [IO.Path]::GetTempFileName()
C:\Windows\System32\cmd.exe /c "`"C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat`" -vcvars_ver=14.44 >nul 2>&1 && set >`"$f`""
Get-Content $f | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item "Env:$($matches[1])" $matches[2] }
}
Remove-Item $f
```

The build scripts (`build-llvm.ps1`, `build-triton.ps1`) handle this automatically.
If running cmake/ninja manually, ensure vcvars env is active or you'll get
`Cannot open include file: 'stddef.h'`.

## Task Dispatch

Parse the user's input and run the matching task below.

---

### Task: `clone-llvm`

Clone LLVM and download JSON into `build/`. Steps:

1. Read the required commit from `cmake/llvm-hash.txt`
2. Clone LLVM:
   ```powershell
   git clone --filter=blob:none https://github.com/llvm/llvm-project.git build/llvm-project
   cd build/llvm-project
   git checkout <hash>
   ```
3. Download JSON:
   ```powershell
   Invoke-WebRequest "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip" -OutFile build/json.zip
   Expand-Archive build/json.zip -DestinationPath build/json -Force
   Remove-Item build/json.zip
   ```
4. Apply LLVM MSVC patches — two options:
   - **Quick:** `cd build/llvm-project && git apply ../../.github/skills/triton-windows-build/scripts/llvm-msvc-patches.diff`
   - **Semantic:** Apply patches L1-L3 from the Patch Reference below (works even if upstream changed)
5. Run inspector to verify: `python .github/skills/triton-windows-build/scripts/inspect-build.py --fix`
   - The `--fix` flag shows fix hints but does NOT auto-apply patches
   - The agent should read the output and apply fixes for any detected issues
   - Read the JSON version from `cmake/json-version.txt` to confirm v3.11.3 is correct

---

### Task: `prebuild-inspect`

Run the build inspector and fix any detected issues:

1. Run: `python .github/skills/triton-windows-build/scripts/inspect-build.py --fix`
2. The inspector reports issues with IDs like `ENV-01`, `LLVM-02`, `TRI-05`, etc.
   Map them to patches using the cross-reference table below.
3. For each detected issue, apply the corresponding patch from the Patch Reference
4. Re-run inspector to confirm all checks pass
5. Expected results:
   - `ENV-01` (cl.exe not on PATH) is OK if vcvars hasn't been run yet
   - All `LLVM-*` checks require LLVM source to exist
   - All `TRI-*` checks scan the triton source code

**Inspector ID → Patch mapping:**
| Inspector | Patch | Inspector | Patch |
|---|---|---|---|
| LLVM-01 | L1 | TRI-07 | T7 |
| LLVM-02 | L2 | TRI-08 | T8 |
| LLVM-03 | L3 | TRI-09 | T9 |
| TRI-01 | T1 | TRI-10 | T10 |
| TRI-02 | T2 | TRI-11 | T11 |
| TRI-03 | T3 | TRI-12 | T12 |
| TRI-04 | T4 | TRI-13 | (part of T6) |
| TRI-05 | T5 | TRI-14 | T15 |
| TRI-06 | T6 | TRI-15 | T14 |

---

### Task: `build-llvm [release|debug]`

Build LLVM/MLIR. Default is `release` (REQUIRED for triton — see Issue #2 below).

1. Activate MSVC 14.44 (see PowerShell vcvars Persistence above)
2. Run the build script:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .github/skills/triton-windows-build/scripts/build-llvm.ps1 -Action all -BuildType release
   ```
   The script auto-detects the correct MSVC toolset from conda's `vc14_runtime` package.

Or manually:
```powershell
$llvm = "build/llvm-project"
cmake -B "$llvm/build" -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl `
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" `
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" `
  -DLLVM_BUILD_TOOLS=ON -DLLVM_BUILD_UTILS=ON `
  -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=ON `
  -DLLVM_ENABLE_DIA_SDK=OFF -DMLIR_BUILD_MLIR_C_DYLIB=OFF `
  "$llvm/llvm"
cmake --build "$llvm/build" --config Release
```

**CRITICAL:** LLVM MUST be Release (`/MD` CRT). Debug uses `/MDd` → linker error LNK2038.

---

### Task: `build-triton`

Build triton-windows. Requires LLVM already built.

1. Activate MSVC 14.44 (see PowerShell vcvars Persistence section above)
2. Set environment (use ABSOLUTE paths for `LLVM_SYSPATH` and `JSON_SYSPATH`):
   ```powershell
   $env:LLVM_SYSPATH = "$PWD/build/llvm-project/build"   # MUST be absolute
   $env:JSON_SYSPATH = "$PWD/build/json"
   $env:TRITON_OFFLINE_BUILD = "1"
   $env:TRITON_BUILD_PROTON = "0"
   $env:TRITON_BUILD_UT = "0"
   $env:CMAKE_PREFIX_PATH = "$env:CONDA_PREFIX/Library"  # for zlib/zstd
   # pip subprocess may not inherit CMAKE_PREFIX_PATH, so also pass via:
   $env:TRITON_APPEND_CMAKE_ARGS = "-DCMAKE_PREFIX_PATH=$env:CONDA_PREFIX/Library"
   ```
3. Install deps: `pip install "pybind11>=2.13.1,<3.0" "setuptools>=40.8.0" wheel`
4. Build: `pip install --no-build-isolation --verbose -e .`
5. Remove AMD backend symlink (it causes ImportError since AMD plugin wasn't built):
   ```powershell
   Remove-Item python/triton/backends/amd -ErrorAction SilentlyContinue
   Remove-Item python/triton/language/extra/hip -ErrorAction SilentlyContinue
   Remove-Item python/triton/tools/extra/hip -ErrorAction SilentlyContinue
   ```
   Note: these symlinks get recreated on every `pip install -e .`, so repeat after rebuilding.
6. Verify: `python -c "import triton; print(triton.__version__)"`

---

### Task: `help`

Print a summary of available tasks and the build workflow.

---

## Critical Build Requirements

| Requirement | Why |
|---|---|
| **MSVC 14.44 (v143)** | VS 2026's 14.51 has template deduction bugs breaking MLIR `walk()` |
| **LLVM = Release** | Triton uses `/MD` (Release CRT); Debug LLVM uses `/MDd` → LNK2038 |
| **pybind11 < 3.0** | pybind11 3.0 broke `PYBIND11_MODULE` macro |
| **vcvars must persist** | Running vcvars64.bat in PS doesn't persist. Use Import-BatchEnvironment pattern (see above) |
| **CMAKE_PREFIX_PATH** | Conda Library dir for zlib/zstd. Also pass via `TRITON_APPEND_CMAKE_ARGS` for pip subprocesses |
| **setuptools >= 40.8.0** | Required as the build backend for pip install |
| **Absolute LLVM_SYSPATH** | Relative paths break when pip changes cwd during build |

---

## Patch Reference

Apply these patches **semantically** — search for the code pattern and replace.
Do NOT use git patches since upstream changes frequently.

### LLVM Patches (apply to `build/llvm-project/`)

**L1: /utf-8 + complex deprecation** (`llvm/CMakeLists.txt`)
Find the `if(MSVC)` block with `BUILD_SHARED_LIBS` check. Add after the `endif()`:
```cmake
add_compile_options("/utf-8")
add_compile_options("/D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")
```

**L2: PassPlugin dllexport** (`llvm/include/llvm/Plugins/PassPlugin.h`)
Replace `extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK` with:
```cpp
extern "C" ::llvm::PassPluginLibraryInfo
#ifdef _MSC_VER
__declspec(dllexport)
#else
LLVM_ATTRIBUTE_WEAK
#endif
```

**L3: TypeID namespace** (`mlir/include/mlir/Support/TypeID.h`)
Change `return detail::TypeIDResolver<T>::resolveTypeID();`
to `return ::mlir::detail::TypeIDResolver<T>::resolveTypeID();`

### Triton Patches (apply to project root)

**T1: GSan clang++ optional** (`third_party/nvidia/CMakeLists.txt`)
Remove `REQUIRED` from `find_program(TRITON_GSAN_CLANGXX ...)`. Add warning
if not found. Guard `add_custom_target(TritonNVIDIAGSanRuntime)` and
`add_dependencies` with `if(TRITON_GSAN_CLANGXX)...endif()`.

**T2: AMD LLD optional** (`third_party/amd/CMakeLists.txt`)
Remove `REQUIRED` from `find_package(LLD ...)`. Guard the `add_triton_plugin`
and `target_link_libraries` with `if(LLD_FOUND)...endif()`.

**T3: FileCheck.exe** (`CMakeLists.txt`)
Replace `configure_file("${LLVM_SYSPATH}/bin/FileCheck" ...)` with WIN32
conditional for `.exe` suffix and `if(EXISTS ...)` guard.

**T4: /bigobj** (`CMakeLists.txt`)
In the MSVC `else()` block (near TritonRelBuildWithAsserts flags), add:
```cmake
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
```

**T5: walk() workaround** (`lib/Analysis/Utility.cpp` + `lib/Dialect/TritonGPU/Transforms/Utility.cpp`)
Replace `.walk([&](Operation* ...)` calls with `::mlir::detail::walk<::mlir::ForwardIterator>(...)`.
Add `#include "mlir/IR/Visitors.h"`. Four instances total (1 in Analysis, 3 in TritonGPU).

**T6: dlfcn.h shim** (NEW: `third_party/nvidia/include/dlfcn_win32.h`)
Create shared header with `#pragma once`, `NOMINMAX`, `WIN32_LEAN_AND_MEAN`,
inline `dlopen/dlsym/dlclose/dlerror` using Win32 `LoadLibrary/GetProcAddress`.
Replace `#include <dlfcn.h>` with `#include "dlfcn_win32.h"` in:
- `third_party/nvidia/triton_nvidia.cc`
- `third_party/nvidia/include/cublas_instance.h`

**T7: Ternary type mismatch**
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`:
  Replace ternary `isMin ? MinimumOp::create(...) : MaximumOp::create(...)`
  with if/else returning `Value`.
- `third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp`:
  Add `static_cast<Operation*>()` to both sides of ternary.

**T8: Lambda capture** (`third_party/amd/lib/TritonAMDGPUTransforms/OptimizeBufferOpPtr.cpp`)
Change `[elemByteWidth]` to `[elemByteWidth, maxOffsetValue]`.

**T9: Overload ambiguity** (`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp`)
Change `getPartition((int)0)` to `getPartition((unsigned)0)`.

**T10: GSan.h types** (`python/triton/experimental/gsan/src/GSan.h`)
Add BEFORE `namespace gsan {`:
```cpp
#ifdef _MSC_VER
#include <cstddef>
#include <cstdint>
#endif
```
Inside namespace, guard type aliases with `#ifdef _MSC_VER` using `::size_t` etc.

**T11: Proton conditional** (`CMakeLists.txt`)
Move `list(APPEND TRITON_PLUGIN_NAMES "proton")` INSIDE `if(TRITON_BUILD_PROTON)`.

**T12: AMDGPU libs conditional** (`CMakeLists.txt`)
Make `LLVMAMDGPUCodeGen`/`LLVMAMDGPUAsmParser` conditional:
```cmake
if(EXISTS "${LLVM_LIBRARY_DIR}/LLVMAMDGPUCodeGen.lib" OR EXISTS "${LLVM_LIBRARY_DIR}/libLLVMAMDGPUCodeGen.a")
  list(APPEND TRITON_LIBRARIES LLVMAMDGPUCodeGen LLVMAMDGPUAsmParser)
endif()
```
Also add missing libs: `LLVMCodeGen`, `LLVMMIRParser`, `LLVMPlugins`.

**T13: AMD backend tuple removal** (`CMakeLists.txt`)
After the `foreach(CODEGEN_BACKEND ...)` loop, add:
```cmake
if(NOT TARGET TritonAMD)
  list(REMOVE_ITEM TRITON_CODEGEN_BACKENDS "amd")
endif()
```

**T14: GSan testing excluded** (`CMakeLists.txt` + `python/src/main.cc`)
In CMakeLists: use `set(TRITON_SHARED_SRCS ...)` + `if(NOT MSVC) list(APPEND ... gsan_testing.cc)`.
In main.cc: `#ifdef _MSC_VER` no-op stub for `init_gsan_testing`.

**T15: Backend discovery** (`python/triton/backends/__init__.py`)
Wrap entry-point import loop in `try/except (ImportError, ModuleNotFoundError): pass`.

---

## Known Issues Summary

| # | Issue | Severity | Root Cause |
|---|---|---|---|
| 1 | MSVC 14.51 template bugs | BLOCKER | walk() RetT deduction failure |
| 2 | CRT mismatch Debug/Release | BLOCKER | /MDd vs /MD linker error |
| 3 | pybind11 3.0 API break | BLOCKER | PYBIND11_MODULE changed |
| 4 | GSan needs clang++ | ERROR | REQUIRED in find_program |
| 5 | AMD needs LLD | ERROR | REQUIRED in find_package |
| 6 | FileCheck vs .exe | ERROR | No WIN32 extension |
| 7 | /bigobj missing in Debug | ERROR | Only in TritonRelBuildWithAsserts |
| 8 | walk() MSVC deduction | ERROR | walkResultType SFINAE failure |
| 9 | TypeID ambiguous detail | ERROR | Unqualified namespace lookup |
| 10 | dlfcn.h missing | ERROR | POSIX header on Windows |
| 11 | Ternary type mismatch | ERROR | MSVC requires same ?: types |
| 12 | constexpr lambda capture | WARNING | MSVC needs explicit capture |
| 13 | getPartition ambiguous | ERROR | int→unsigned overload |
| 14 | GSan GCC types | ERROR | __SIZE_TYPE__ unavailable |
| 15 | Proton unconditional | ERROR | Always in backends tuple |
| 16 | AMDGPU libs unconditional | ERROR | Linked without target check |
| 17 | windows.h min/max | ERROR | NOMINMAX not defined |
| 18 | Backend discovery crash | WARNING | No try/except on import |
| 19 | GSan testing on MSVC | ERROR | GCC-only code compiled |
