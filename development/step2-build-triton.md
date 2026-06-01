# Step 2: Build Triton-Windows

Complete guide to building triton-windows 3.7.0 on Windows 11 with VS 2026.

## Prerequisites

| Requirement | Details |
|---|---|
| **Step 1 completed** | LLVM/MLIR built with `Release` + `LLVM_ENABLE_ASSERTIONS=ON` + MSVC 14.44 |
| **nlohmann/json** | Downloaded to `references/json/` |
| **pybind11 2.x** | `pip install "pybind11>=2.13.1,<3.0"` (NOT 3.0+) |
| **setuptools** | `pip install setuptools wheel` |

## 1. Download nlohmann/json

```powershell
$jsonDir = "references\json"
$url = "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip"
Invoke-WebRequest -Uri $url -OutFile "$jsonDir.zip"
Expand-Archive "$jsonDir.zip" -DestinationPath $jsonDir -Force
Remove-Item "$jsonDir.zip"
```

## 2. Install Python Build Dependencies

```powershell
python -m pip install "setuptools>=40.8.0" "pybind11>=2.13.1,<3.0" wheel
```

**Warning:** pybind11 3.0+ breaks the `PYBIND11_MODULE` macro. Must use 2.x.

## 3. Apply MSVC Compatibility Patches

The triton-windows codebase needs several MSVC-specific fixes. Rather than using
git patches (which break when upstream changes), use the Copilot agent skill:

```
@workspace Apply MSVC patches for triton-windows build on VS 2026
```

Or apply manually — see `troubleshooting-guide.md` for the full list of 19 issues.

### Quick Summary of Required Changes

**LLVM patches** (3 files):
- `llvm/CMakeLists.txt` — `/utf-8` + complex deprecation
- `llvm/include/llvm/Plugins/PassPlugin.h` — `__declspec(dllexport)`
- `mlir/include/mlir/Support/TypeID.h` — `::mlir::detail::` qualification

**Triton patches** (15 files):
- CMake: `/bigobj`, FileCheck.exe, optional GSan/LLD/AMDGPU, extra LLVM libs
- C++: `walk()` API workarounds, dlfcn.h shim, ternary fixes, lambda captures
- Python: Tolerant backend discovery, GSan testing stub

## 4. Build

Activate MSVC 14.44 (v143) toolset and set environment variables:

```powershell
# Activate MSVC 14.44 (CRITICAL: must use v143, not the latest VS 2026 toolset)
vcvars64.bat -vcvars_ver=14.44

# Set triton build variables
$env:LLVM_SYSPATH = "D:\code-dive\triton-windows\references\llvm-project\build"
$env:JSON_SYSPATH = "D:\code-dive\triton-windows\references\json"
$env:TRITON_OFFLINE_BUILD = "1"
$env:TRITON_BUILD_PROTON = "0"
$env:TRITON_BUILD_UT = "0"
$env:CMAKE_PREFIX_PATH = "C:\ProgramData\anaconda3\envs\mlir-dev\Library"

# Build and install (editable)
pip install --no-build-isolation --verbose -e .
```

### Critical Notes

1. **MSVC toolset MUST be 14.44** — VS 2026's default (14.51) has template
   deduction regressions that break MLIR's `walk()` API.

2. **LLVM MUST be Release** — triton uses `/MD` (Release CRT) by default.
   Debug LLVM uses `/MDd` which causes `LNK2038` mismatches.

3. **vcvars environment MUST persist** — if the environment is lost between
   cmake configure and build, you'll get `Cannot open include file: 'stddef.h'`.

4. **`CMAKE_PREFIX_PATH`** must include conda's Library dir for zlib/zstd.

## 5. Verify

```powershell
python -c "import triton; print('Version:', triton.__version__); print('Backends:', list(triton.backends.backends.keys()))"
# Expected output:
# Version: 3.7.0
# Backends: ['nvidia']
```

## 6. Incremental Rebuilds

After modifying C++ source files, rebuild with:

```powershell
# Make sure vcvars is activated first!
cd build\cmake.win-amd64-cpython-3.14
ninja
```

Or use `make` from the triton root (which calls ninja internally).
