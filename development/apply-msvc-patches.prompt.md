---
description: "Apply MSVC compatibility patches for building triton-windows on VS 2026. Use when asked to build triton, fix MSVC errors, or apply Windows patches. Covers LLVM patches + 17 triton source fixes."
---

# Triton-Windows MSVC Patch Skill

Apply these patches semantically — search for patterns and replace. Do NOT use
git patches since upstream changes frequently.

External deps (LLVM, JSON) are in `build/` (git-ignored).

## LLVM Patches (build/llvm-project/)

### L1: /utf-8 (llvm/CMakeLists.txt)
In `if(MSVC)` block after `BUILD_SHARED_LIBS` check, add:
```cmake
add_compile_options("/utf-8")
add_compile_options("/D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")
```

### L2: dllexport (llvm/include/llvm/Plugins/PassPlugin.h)
Replace `LLVM_ATTRIBUTE_WEAK` before `llvmGetPassPluginInfo` with:
```cpp
#ifdef _MSC_VER
__declspec(dllexport)
#else
LLVM_ATTRIBUTE_WEAK
#endif
```

### L3: TypeID (mlir/include/mlir/Support/TypeID.h)
Change `return detail::TypeIDResolver<T>::resolveTypeID();`
to `return ::mlir::detail::TypeIDResolver<T>::resolveTypeID();`

## Triton Patches (project root)

### T1: GSan optional (third_party/nvidia/CMakeLists.txt)
Remove `REQUIRED` from `find_program(TRITON_GSAN_CLANGXX ...)`.
Guard build target with `if(TRITON_GSAN_CLANGXX)...else() warning...endif()`.

### T2: LLD optional (third_party/amd/CMakeLists.txt)
Remove `REQUIRED` from `find_package(LLD ...)`.
Guard plugin with `if(LLD_FOUND)`.

### T3: FileCheck.exe (CMakeLists.txt)
Replace `configure_file("${LLVM_SYSPATH}/bin/FileCheck" ...)` with
WIN32 conditional for `.exe` extension and existence check.

### T4: /bigobj (CMakeLists.txt)
Add `add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")` in MSVC block.

### T5: walk() (lib/Analysis/Utility.cpp + lib/Dialect/TritonGPU/Transforms/Utility.cpp)
Replace `.walk([&](Operation* ...)` with `::mlir::detail::walk<::mlir::ForwardIterator>(...)`.
Add `#include "mlir/IR/Visitors.h"`.

### T6: dlfcn shim (NEW: third_party/nvidia/include/dlfcn_win32.h)
Create with `NOMINMAX`, `WIN32_LEAN_AND_MEAN`, inline `dlopen/dlsym/dlclose/dlerror`.
Replace `#include <dlfcn.h>` with `#include "dlfcn_win32.h"` in triton_nvidia.cc and cublas_instance.h.

### T7: Ternary (TensorMemoryToLLVM.cpp + BlockPingpong.cpp)
Replace `?:` with if/else or `static_cast<Operation*>()`.

### T8: Lambda capture (OptimizeBufferOpPtr.cpp)
Add `maxOffsetValue` to `[elemByteWidth]` capture.

### T9: Overload (PartitionLoops.cpp)
Change `(int)0` to `(unsigned)0`.

### T10: GSan types (GSan.h)
Add `#ifdef _MSC_VER` with `<cstddef>/<cstdint>` BEFORE namespace, then
type aliases using `::size_t` etc.

### T11: Proton (CMakeLists.txt)
Move `list(APPEND TRITON_PLUGIN_NAMES "proton")` inside `if(TRITON_BUILD_PROTON)`.

### T12: AMDGPU libs (CMakeLists.txt)
Make `LLVMAMDGPUCodeGen`/`AsmParser` conditional on file existence.
Add `LLVMCodeGen`, `LLVMMIRParser`, `LLVMPlugins`.

### T13: AMD tuple (CMakeLists.txt)
After `foreach(CODEGEN_BACKEND ...)` add `if(NOT TARGET TritonAMD) list(REMOVE_ITEM ...)`.

### T14: GSan testing (CMakeLists.txt + main.cc)
Exclude `gsan_testing.cc` on MSVC. Add `#ifdef _MSC_VER` no-op stub in main.cc.

### T15: Backend discovery (python/triton/backends/__init__.py)
Wrap entry-point import loop in `try/except (ImportError, ModuleNotFoundError)`.

## Build Environment

- MSVC: 14.44 via `vcvars64.bat -vcvars_ver=14.44`
- LLVM: Release at `build/llvm-project/build/`
- JSON: at `build/json/`
- pybind11: 2.x (NOT 3.0+)
- `CMAKE_PREFIX_PATH`: conda Library dir
