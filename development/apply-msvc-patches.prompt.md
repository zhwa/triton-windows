---
description: "Apply MSVC compatibility patches for building triton-windows on Visual Studio 2026 (Windows). Use when the user wants to build triton-windows, fix MSVC build errors, or apply Windows compatibility changes. This skill knows all 19 MSVC issues and how to fix them."
---

# Triton-Windows MSVC Patch Skill

When asked to apply MSVC patches for triton-windows, apply the following changes.
Each fix targets a specific MSVC compatibility issue. Search for the exact code
pattern and replace it — do NOT use git patches since the upstream changes frequently.

## LLVM Patches (apply to references/llvm-project/)

### Patch L1: UTF-8 and Complex Deprecation (llvm/CMakeLists.txt)
**Find** the `if(MSVC)` block containing `BUILD_SHARED_LIBS` check.
**Add** after the `endif()` for BUILD_SHARED_LIBS:
```cmake
  add_compile_options("/utf-8")
  add_compile_options("/D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")
```

### Patch L2: PassPlugin dllexport (llvm/include/llvm/Plugins/PassPlugin.h)
**Find** `extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK`
**Replace** with:
```cpp
extern "C" ::llvm::PassPluginLibraryInfo
#ifdef _MSC_VER
__declspec(dllexport)
#else
LLVM_ATTRIBUTE_WEAK
#endif
```

### Patch L3: TypeID namespace (mlir/include/mlir/Support/TypeID.h)
**Find** `return detail::TypeIDResolver<T>::resolveTypeID();`
**Replace** with `return ::mlir::detail::TypeIDResolver<T>::resolveTypeID();`

## Triton Patches (apply to triton-windows root)

### Patch T1: GSan clang++ Optional (third_party/nvidia/CMakeLists.txt)
**Find** `find_program(TRITON_GSAN_CLANGXX ... REQUIRED NO_DEFAULT_PATH)`
**Remove** `REQUIRED`. Add warning if not found. Guard the custom target and
`add_dependencies` with `if(TARGET TritonNVIDIAGSanRuntime)`.

### Patch T2: AMD LLD Optional (third_party/amd/CMakeLists.txt)
**Find** `find_package(LLD REQUIRED CONFIG ...)`
**Remove** `REQUIRED`. Guard the `add_triton_plugin` and link commands with
`if(LLD_FOUND)`.

### Patch T3: FileCheck.exe (CMakeLists.txt)
**Find** `configure_file("${LLVM_SYSPATH}/bin/FileCheck" ...)`
**Replace** with conditional `.exe` suffix on Windows and existence check.

### Patch T4: /bigobj for All MSVC Builds (CMakeLists.txt)
**Find** the `else()` block for MSVC flags (near `TritonRelBuildWithAsserts`).
**Add** after the linker flags: `add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")`

### Patch T5: Proton Conditional (CMakeLists.txt)
**Find** `list(APPEND TRITON_PLUGIN_NAMES "proton")` outside `if(TRITON_BUILD_PROTON)`.
**Move** inside the `if(TRITON_BUILD_PROTON)` block.

### Patch T6: GSan Testing Excluded on MSVC (CMakeLists.txt + main.cc)
**In CMakeLists.txt:** Remove `gsan_testing.cc` from the `add_library(triton ...)` sources
on MSVC. Use `if(NOT MSVC) list(APPEND ...)`.
**In main.cc:** Provide a no-op `init_gsan_testing` stub under `#ifdef _MSC_VER`.

### Patch T7: AMDGPU Libs Conditional (CMakeLists.txt)
**Find** `LLVMAMDGPUCodeGen` and `LLVMAMDGPUAsmParser` in the library list.
**Make** conditional on the lib file existing. Also add missing libs:
`LLVMCodeGen`, `LLVMMIRParser`, `LLVMPlugins`.

### Patch T8: AMD Backend Removal from Tuple (CMakeLists.txt)
After the `foreach(CODEGEN_BACKEND ...)` loop, add:
```cmake
if(NOT TARGET TritonAMD)
  list(REMOVE_ITEM TRITON_CODEGEN_BACKENDS "amd")
endif()
```

### Patch T9: walk() API Workaround (lib/Analysis/Utility.cpp)
**Find** `op.walk([&](Operation *nestedOp) -> WalkResult {` in `isAssociative()`
**Replace** with `::mlir::detail::walk<::mlir::ForwardIterator>(op.getOperation(), ...)`.
Add `#include "mlir/IR/Visitors.h"`.

### Patch T10: walk() API Workaround (lib/Dialect/TritonGPU/Transforms/Utility.cpp)
Three instances of `.walk([&](Operation *...)` need the same treatment:
1. `func.walk(...)` in `GraphDumper::dump()`
2. `top->walk(...)` in dead arg elimination
3. `op->walk(...)` in operand visitor

### Patch T11: Ternary Fix (third_party/nvidia/.../TensorMemoryToLLVM.cpp)
**Find** the ternary with `MinimumOp`/`MaximumOp` and `MinNumOp`/`MaxNumOp`.
**Replace** with if/else returning `Value`.

### Patch T12: Ternary Fix (third_party/amd/.../BlockPingpong.cpp)
**Find** `useAsyncCopy ? asyncCopyOps[1] : gLoadOps[1]`
**Add** `static_cast<Operation*>()` to both sides.

### Patch T13: Lambda Capture (third_party/amd/.../OptimizeBufferOpPtr.cpp)
**Find** `[elemByteWidth](const llvm::APInt &x)`
**Replace** with `[elemByteWidth, maxOffsetValue](const llvm::APInt &x)`

### Patch T14: Overload Ambiguity (lib/.../PartitionLoops.cpp)
**Find** `partitions.getPartition((int)0)`
**Replace** with `partitions.getPartition((unsigned)0)`

### Patch T15: GSan.h GCC Types (python/triton/experimental/gsan/src/GSan.h)
**Add** before `namespace gsan {`:
```cpp
#ifdef _MSC_VER
#include <cstddef>
#include <cstdint>
#endif
```
**Inside** `namespace gsan {`, guard the type aliases:
```cpp
#ifdef _MSC_VER
using size_t = ::size_t; // etc.
#else
using size_t = __SIZE_TYPE__; // etc.
#endif
```

### Patch T16: dlfcn.h Shim (NEW FILE: third_party/nvidia/include/dlfcn_win32.h)
Create a shared header with `#pragma once`, `NOMINMAX`, `WIN32_LEAN_AND_MEAN`,
and inline `dlopen`/`dlsym`/`dlclose`/`dlerror` implementations using Win32 API.

### Patch T17: Use dlfcn Shim (triton_nvidia.cc + cublas_instance.h)
Replace `#include <dlfcn.h>` with `#include "dlfcn_win32.h"` in both files.

### Patch T18: Backend Discovery Tolerance (python/triton/backends/__init__.py)
Wrap the entry-point backend import loop in `try/except (ImportError, ModuleNotFoundError)`.

## Build Environment Requirements

- **MSVC toolset:** 14.44 (v143) via `vcvars64.bat -vcvars_ver=14.44`
- **LLVM build type:** Release (to match triton's `/MD` CRT)
- **pybind11:** 2.x (NOT 3.0+)
- **`CMAKE_PREFIX_PATH`:** Must include conda Library dir for zlib/zstd
