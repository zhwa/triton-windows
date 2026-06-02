# Building Triton-Windows on VS 2026: Troubleshooting Guide

All 19 issues encountered building triton-windows 3.7.0 on Windows 11 with VS 2026.

External deps live in `build/` (git-ignored): `build/llvm-project/`, `build/json/`.

Run `python development/inspect-build.py --fix` to auto-detect all issues.

## Environment

| Component | Version |
|---|---|
| OS | Windows 11 |
| Visual Studio | 2026 (v18), Enterprise |
| MSVC Toolsets | 14.44 (v143, required), 14.51 (v144, default) |
| Conda env | `mlir-dev` (cmake, ninja from conda-forge) |
| Python | 3.14.3 (free-threaded, from conda) |
| CUDA Toolkit | 13.2.51 |
| LLVM commit | `87717bf9f81f7b29466c5d9a30a3453bdfc93941` |

---

## Issue 1: MSVC Toolset Version (CRITICAL)

**Symptom:** `walk()` template deduction failures (error C2672), `walkResultType`
SFINAE failures across many files.

**Root cause:** VS 2026 ships MSVC 19.51 (toolset 14.51) by default. The
triton-windows project (and its CI) is built with MSVC v143 (toolset 14.44,
equivalent to VS 2022). MSVC 19.51 has template deduction regressions that break
MLIR's `walk()` API — specifically, the default template parameter
`RetT = detail::walkResultType<FnT>` is evaluated eagerly before SFINAE can
exclude the wrong overload.

**Fix:** Pass `-vcvars_ver=14.44` when calling `vcvars64.bat` to select the
14.44 toolset (must be installed via VS Installer). Both LLVM and triton must
be built with the same toolset.

```powershell
vcvars64.bat -vcvars_ver=14.44
```

**How we discovered this:** The conda-forge triton recipe skips Windows entirely.
The triton-windows CI uses `ilammy/msvc-dev-cmd@v1` which defaults to v143.
Checking the CI workflow revealed the toolset requirement.

---

## Issue 2: CRT Mismatch (LLVM Debug vs Triton Release)

**Symptom:** `LNK2038: mismatch detected for 'RuntimeLibrary': value
'MDd_DynamicDebug' doesn't match value 'MD_DynamicRelease'`

**Root cause:** LLVM built with `Debug` uses `/MDd` (Debug CRT), while triton's
default build type `TritonRelBuildWithAsserts` uses `/MD` (Release CRT). MSVC
strictly enforces CRT consistency at link time.

**Fix:** Build LLVM with `Release` (not `Debug`):
```
cmake -DCMAKE_BUILD_TYPE=Release ...
```

Despite building LLVM in Release, `LLVM_ENABLE_ASSERTIONS=ON` still gives you
assertion checks for debugging.

---

## Issue 3: pybind11 3.0 API Break

**Symptom:** `PYBIND11_MODULE` macro errors — `error C2062: type 'int'
unexpected`, `pybind11_exec_libtriton: undeclared identifier`.

**Root cause:** pybind11 3.0 (released 2026) changed the `PYBIND11_MODULE` macro
signature. Triton requires `pybind11>=2.13.1` but pip installed 3.0.4 which is
a major breaking change.

**Fix:** Pin pybind11 to 2.x:
```
pip install "pybind11>=2.13.1,<3.0"
```

---

## Issue 4: Missing `clang++` for GSan Runtime

**Symptom:** `CMake Error: Could not find TRITON_GSAN_CLANGXX using the
following names: clang++`

**Root cause:** The NVIDIA backend compiles a GPU sanitizer runtime (`gsan.ll`)
from CUDA source using clang++. We didn't build clang as part of LLVM
(only `mlir;llvm`).

**Fix:** Make clang++ optional in `third_party/nvidia/CMakeLists.txt`. Change
`REQUIRED` to a warning and guard the custom target.

---

## Issue 5: AMD Backend Requires LLD

**Symptom:** `Could not find a package configuration file provided by "LLD"`

**Root cause:** The AMD backend plugin links against `lldCommon` and `lldELF`.
We didn't build LLD.

**Fix:** Make LLD optional in `third_party/amd/CMakeLists.txt`. Skip the
`TritonAMD` plugin when LLD is not found. Also remove `amd` from
`TRITON_CODEGEN_BACKENDS` when the plugin wasn't built.

---

## Issue 6: `FileCheck` vs `FileCheck.exe`

**Symptom:** `CMake Error: File .../build/bin/FileCheck does not exist.`

**Root cause:** Triton's CMakeLists copies `FileCheck` to the wheel dir, but on
Windows the binary has a `.exe` extension.

**Fix:** Add `.exe` suffix conditionally:
```cmake
if(WIN32)
  set(_filecheck_name "FileCheck.exe")
endif()
```

---

## Issue 7: `/bigobj` Missing for Debug Builds

**Symptom:** `fatal error C1128: number of sections exceeded object file format
limit: compile with /bigobj` (on `ir.cc`)

**Root cause:** `/bigobj` was only set for the `TritonRelBuildWithAsserts` build
type, not for `Debug`. Large translation units like `ir.cc` exceed the default
COFF section limit.

**Fix:** Add `/bigobj` unconditionally for MSVC:
```cmake
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
```

---

## Issue 8: `walk()` Template Deduction (MSVC-specific)

**Symptom:** `error C2672: 'mlir::OpState::walk': no matching overloaded
function found` — even with the correct v143 toolset.

**Root cause:** MLIR's `walk()` uses `walkResultType<FnT>` which is defined as
`decltype(walk(nullptr, std::declval<FnT>()))`. MSVC evaluates this default
template argument before SFINAE kicks in, causing overload resolution failures
when passing lambdas.

**Fix:** Use `::mlir::detail::walk<::mlir::ForwardIterator>()` free function
directly, bypassing the template member function entirely:
```cpp
::mlir::detail::walk<::mlir::ForwardIterator>(
    op.getOperation(),
    llvm::function_ref<WalkResult(Operation *)>([&](Operation *nestedOp) -> WalkResult {
        // ...
    }),
    mlir::WalkOrder::PostOrder);
```

---

## Issue 9: `TypeID.h` Ambiguous `detail` Namespace

**Symptom:** `error C2872: 'detail': ambiguous symbol` in
`mlir/Support/TypeID.h`

**Root cause:** MSVC's name lookup finds `mlir::detail`, `mlir::triton::detail`,
and `mlir::triton::gpu::detail` when resolving unqualified `detail` in a template
instantiation context.

**Fix:** Patch `TypeID.h` to use fully qualified `::mlir::detail::`:
```cpp
return ::mlir::detail::TypeIDResolver<T>::resolveTypeID();
```

---

## Issue 10: `dlfcn.h` Not Available on Windows

**Symptom:** `fatal error C1083: Cannot open include file: 'dlfcn.h'`

**Root cause:** `dlfcn.h` is a POSIX header for dynamic loading (`dlopen`,
`dlsym`, etc.). Windows uses `LoadLibrary`/`GetProcAddress` instead. Both
`triton_nvidia.cc` and `cublas_instance.h` include it.

**Fix:** Created `third_party/nvidia/include/dlfcn_win32.h` — a shared header
that provides Windows-compatible inline implementations using the Win32 API.
Must use `NOMINMAX` and `WIN32_LEAN_AND_MEAN` to prevent `windows.h` macros
from breaking LLVM headers (`min`/`max` conflict).

---

## Issue 11: Ternary Operator Type Mismatches

**Symptom:** `error C2446: ':': no conversion from 'MaximumOp' to 'MinimumOp'`

**Root cause:** MSVC requires both branches of `?:` to have the same type or a
common base. GCC/Clang are more lenient with implicit conversions. Affects:
- `TensorMemoryToLLVM.cpp`: `MinimumOp` vs `MaximumOp` in ternary
- `BlockPingpong.cpp`: `AsyncCopyGlobalToLocalOp` vs `LoadOp` in ternary

**Fix:** Replace ternary with if/else, or use explicit `static_cast<Operation*>()`.

---

## Issue 12: Lambda Capture of `constexpr` Local

**Symptom:** `error C3493: 'maxOffsetValue' cannot be implicitly captured`

**Root cause:** MSVC v143 doesn't always allow implicit capture of `constexpr`
local variables in lambdas (even though C++17 permits it).

**Fix:** Explicitly capture the variable: `[elemByteWidth, maxOffsetValue](...)`

---

## Issue 13: `getPartition((int)0)` Ambiguous Overload

**Symptom:** `error C2668: 'getPartition': ambiguous call to overloaded function`

**Root cause:** MSVC won't implicitly convert `int` (from `(int)0`) to
`unsigned` when there are overloads for both `unsigned` and `Operation*`.

**Fix:** Change `(int)0` to `(unsigned)0`.

---

## Issue 14: GSan Header Uses GCC Built-in Types

**Symptom:** `error C2061: syntax error: identifier '__SIZE_TYPE__'`

**Root cause:** `GSan.h` uses GCC/Clang built-in type aliases (`__SIZE_TYPE__`,
`__UINT8_TYPE__`, etc.) which don't exist on MSVC.

**Fix:** `#ifdef _MSC_VER` guard with `#include <cstddef>` / `<cstdint>` and
use `::size_t`, `::uint8_t`, etc. The `#include` must be OUTSIDE the
`namespace gsan {}` block.

---

## Issue 15: `proton` Always in TRITON_BACKENDS_TUPLE

**Symptom:** `error C2065: 'proton': undeclared identifier` in `main.cc`

**Root cause:** `CMakeLists.txt` unconditionally adds `proton` to
`TRITON_PLUGIN_NAMES` with the comment "We always build proton dialect", even
when `TRITON_BUILD_PROTON=0`.

**Fix:** Guard the `list(APPEND TRITON_PLUGIN_NAMES "proton")` with
`if(TRITON_BUILD_PROTON)`.

---

## Issue 16: Missing LLVM Libraries (AMDGPU, MIRParser, Plugins)

**Symptom:** `LNK1104: cannot open file 'LLVMAMDGPUCodeGen.lib'` and
`LNK2019: unresolved external symbol llvm::MIRParser`

**Root cause:** LLVM was built with `host;NVPTX` targets only (no AMDGPU), and
some LLVM libs (`LLVMCodeGen`, `LLVMMIRParser`, `LLVMPlugins`) weren't listed
in triton's link dependencies.

**Fix:** Make AMDGPU libs conditional on their existence, and add missing libs
(`LLVMCodeGen`, `LLVMMIRParser`, `LLVMPlugins`) to the link list.

---

## Issue 17: LLVM `PassPlugin.h` — `LLVM_ATTRIBUTE_WEAK` on Windows

**Symptom:** Undefined symbol `llvmGetPassPluginInfo`

**Root cause:** `LLVM_ATTRIBUTE_WEAK` maps to `__attribute__((weak))` which MSVC
doesn't support. The symbol needs `__declspec(dllexport)` instead.

**Fix:** Documented in triton-windows `BUILD.md`. Patch `PassPlugin.h`:
```cpp
#ifdef _MSC_VER
__declspec(dllexport)
#else
LLVM_ATTRIBUTE_WEAK
#endif
```

---

## Issue 18: LLVM `/utf-8` and Complex Deprecation Warning

**Symptom:** LLVM build fails with encoding errors or `_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING`.

**Root cause:** MSVC requires `/utf-8` for source files with non-ASCII characters.
Newer MSVC STL triggers a deprecation warning for `<complex>`.

**Fix:** Add to LLVM's `llvm/CMakeLists.txt` in the `if(MSVC)` block:
```cmake
add_compile_options("/utf-8")
add_compile_options("/D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")
```

---

## Issue 19: `windows.h` `min`/`max` Macro Pollution

**Symptom:** `error C2589: '(': illegal token on right side of '::'` in
LLVM headers after including `windows.h`.

**Root cause:** `windows.h` defines `#define min(...)` and `#define max(...)`
macros that conflict with `std::min`/`std::max` and any template using `min`/`max`.

**Fix:** Always define `NOMINMAX` and `WIN32_LEAN_AND_MEAN` before including
`windows.h`.

---

## Summary: Files Modified

### LLVM (3 files)
| File | Change |
|---|---|
| `llvm/CMakeLists.txt` | `/utf-8` + `_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING` |
| `llvm/include/llvm/Plugins/PassPlugin.h` | `__declspec(dllexport)` for MSVC |
| `mlir/include/mlir/Support/TypeID.h` | `::mlir::detail::` fully qualified |

### Triton (15 files + 1 new)
| File | Change |
|---|---|
| `CMakeLists.txt` | `/bigobj`, FileCheck.exe, proton conditional, gsan skip, AMDGPU libs conditional, extra LLVM libs |
| `setup.py` | (reverted — kept nvidia+amd) |
| `python/src/main.cc` | `init_gsan_testing` stub on MSVC |
| `python/triton/backends/__init__.py` | Tolerant backend discovery (try/except) |
| `python/triton/experimental/gsan/src/GSan.h` | `#ifdef _MSC_VER` for stdint types |
| `lib/Analysis/Utility.cpp` | `::mlir::detail::walk` free function |
| `lib/Dialect/TritonGPU/Transforms/Utility.cpp` | `::mlir::detail::walk` free function (3 instances) |
| `lib/.../PartitionLoops.cpp` | `(unsigned)0` instead of `(int)0` |
| `third_party/nvidia/CMakeLists.txt` | GSan clang++ optional |
| `third_party/nvidia/include/dlfcn_win32.h` | **NEW** — shared dlfcn shim |
| `third_party/nvidia/include/cublas_instance.h` | Use `dlfcn_win32.h` |
| `third_party/nvidia/triton_nvidia.cc` | Use `dlfcn_win32.h` |
| `third_party/nvidia/.../TensorMemoryToLLVM.cpp` | Ternary → if/else |
| `third_party/amd/CMakeLists.txt` | LLD optional |
| `third_party/amd/.../BlockPingpong.cpp` | `static_cast<Operation*>` |
| `third_party/amd/.../OptimizeBufferOpPtr.cpp` | Explicit lambda capture |
