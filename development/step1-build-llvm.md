# Step 1: Build LLVM/MLIR for Triton-Windows

Triton requires a specific LLVM/MLIR commit built from source. This guide walks
through the full process on Windows 11 with Visual Studio 2026 and Conda.

## Prerequisites

| Requirement | Details |
|---|---|
| **Visual Studio 2026** | Any edition (Community/Professional/Enterprise). Needs C++ desktop workload. |
| **Conda** | Anaconda or Miniconda, with the `mlir-dev` env providing `cmake`, `ninja`. |
| **Git** | For cloning llvm-project. |
| **Disk space** | ~40 GB for Debug build, ~15 GB for Release build. |

The conda `mlir-dev` environment is used only for `cmake` and `ninja` (not for
its MLIR libraries — Triton needs a specific LLVM commit).

```powershell
# Create conda env if you don't have it yet
conda create -n mlir-dev -c conda-forge ninja cmake
```

## 1. Clone LLVM

Clone llvm-project into `references\llvm-project` (this directory is
git-ignored by triton-windows, so it won't interfere with triton's own repo):

```powershell
cd d:\code-dive\triton-windows\references
git clone --filter=blob:none https://github.com/llvm/llvm-project.git
```

The `--filter=blob:none` flag avoids downloading the full history (saves time
and disk).

## 2. Checkout the Required Commit

Triton pins to a specific LLVM commit recorded in `cmake/llvm-hash.txt`:

```powershell
# Read the required commit hash
$hash = Get-Content d:\code-dive\triton-windows\cmake\llvm-hash.txt
Write-Host "Required LLVM commit: $hash"

# Checkout
cd d:\code-dive\triton-windows\references\llvm-project
git checkout $hash
```

As of this writing, the commit is `87717bf9f81f7b29466c5d9a30a3453bdfc93941`.

## 3. Apply MSVC Patches

Two patches are needed for MSVC compatibility. You can apply them all at once
using the saved patch file:

```powershell
cd d:\code-dive\triton-windows\references\llvm-project
git apply d:\code-dive\triton-windows\development\llvm-msvc-patches.diff
```

Or apply them manually — details below.

### Patch 1: UTF-8 and complex deprecation warning (llvm/CMakeLists.txt)

Find the `if(MSVC)` block around line 947 and add two `add_compile_options`
lines after the `BUILD_SHARED_LIBS` check:

```diff
 if(MSVC)
   option(LLVM_BUILD_LLVM_C_DYLIB "Build LLVM-C.dll (Windows only)" ON)
   if (BUILD_SHARED_LIBS)
     message(FATAL_ERROR "BUILD_SHARED_LIBS options is not supported on Windows.")
   endif()
+  add_compile_options("/utf-8")
+  add_compile_options("/D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")
 else()
```

**Why:** MSVC needs `/utf-8` for source files with non-ASCII characters. The
deprecation warning comes from `<complex>` in newer MSVC STL headers.

### Patch 2: dllexport for PassPlugin (llvm/include/llvm/Plugins/PassPlugin.h)

Replace `LLVM_ATTRIBUTE_WEAK` with `__declspec(dllexport)` for MSVC:

```diff
-extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
+extern "C" ::llvm::PassPluginLibraryInfo
+#ifdef _MSC_VER
+__declspec(dllexport)
+#else
+LLVM_ATTRIBUTE_WEAK
+#endif
 llvmGetPassPluginInfo();
```

**Why:** MSVC doesn't support `__attribute__((weak))`. Without this fix, Triton's
`PrintLoadStoreMemSpaces.cpp` will fail to link with undefined symbol errors.
See: https://github.com/llvm/llvm-project/pull/115431

## 4. Build

Copy the build script to the llvm-project directory and run it:

```powershell
# Copy the build script
Copy-Item d:\code-dive\triton-windows\development\build-llvm.ps1 `
          d:\code-dive\triton-windows\references\llvm-project\build-for-triton.ps1

# Release build (REQUIRED — triton uses /MD Release CRT, must match)
powershell -ExecutionPolicy Bypass -File `
    d:\code-dive\triton-windows\references\llvm-project\build-for-triton.ps1 `
    -Action all -BuildType release
```

**Important:** You MUST use Release (not Debug). Triton's default build type
`TritonRelBuildWithAsserts` uses `/MD` (Release CRT). Building LLVM with Debug
(`/MDd`) causes `LNK2038: mismatch detected for 'RuntimeLibrary'` at link time.

### What the build script does

1. Activates conda env `mlir-dev` for `cmake` and `ninja`
2. Detects MSVC toolset version from conda's `vc14_runtime` package
3. Activates VS build environment via `vcvars64.bat`
4. Runs CMake with Ninja generator and these key options:
   - `LLVM_ENABLE_PROJECTS=mlir;llvm` — builds MLIR alongside LLVM
   - `LLVM_TARGETS_TO_BUILD=host;NVPTX` — X86 host + NVIDIA PTX backend
   - `LLVM_ENABLE_ASSERTIONS=ON` — keeps assertion checks for development
   - `LLVM_BUILD_TOOLS=ON` — builds `mlir-opt`, `mlir-translate`, etc.
   - `CMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded` — uses `/Z7` to avoid PDB
     locking with parallel Ninja builds
5. Builds everything with Ninja (uses all available CPU cores)

### Build script actions

| Action | Description |
|---|---|
| `all` (default) | Configure + build |
| `configure` | CMake configure only |
| `build` | Build only (assumes already configured) |
| `rebuild` | Clean + configure + build |
| `clean` | Delete the build directory |

### Expected build time

| Configuration | Targets | Approximate time (24 cores) |
|---|---|---|
| Debug (host+NVPTX) | ~5,500 | 1-2 hours |
| Release (host+NVPTX) | ~5,500 | 30-60 minutes |

## 5. Verify the Build

After the build completes, verify key outputs:

```powershell
$build = "d:\code-dive\triton-windows\references\llvm-project\build"

# Check tools
Test-Path "$build\bin\mlir-opt.exe"          # MLIR optimizer
Test-Path "$build\bin\mlir-translate.exe"    # MLIR translator
Test-Path "$build\bin\mlir-tblgen.exe"       # TableGen for MLIR
Test-Path "$build\bin\FileCheck.exe"         # Lit test checker

# Check CMake configs (needed by Triton's find_package)
Test-Path "$build\lib\cmake\mlir\MLIRConfig.cmake"
Test-Path "$build\lib\cmake\llvm\LLVMConfig.cmake"

# Count libraries
(Get-ChildItem "$build\lib\*MLIR*.lib").Count   # Should be ~457
(Get-ChildItem "$build\lib\*LLVM*.lib").Count   # Should be ~176
```

## 6. Next Step: Build Triton

Once LLVM is built, set the environment variable and proceed to build
triton-windows (see Step 2 doc):

```powershell
$Env:LLVM_SYSPATH = "d:\code-dive\triton-windows\references\llvm-project\build"
$Env:TRITON_OFFLINE_BUILD = "1"
```

## Troubleshooting

### "Unknown value for CMAKE_BUILD_TYPE"
The build type must be exactly `Debug` or `Release` (case-sensitive in CMake).
The build script handles this automatically.

### MASM warnings about `/utf-8`
Harmless — the `/utf-8` flag is meant for the C/C++ compiler, and MASM simply
ignores flags it doesn't understand.

### Linker errors about `llvmGetPassPluginInfo`
You forgot to apply Patch 2. Apply it and rebuild.

### MSVC toolset version mismatch
The script tries to match conda's `vc14_runtime` version via `-vcvars_ver=`.
If it picks the wrong toolset, you can override by editing the script's
`$VcVarsVerFlag` variable directly.

### Out of disk space
Debug builds with symbols can consume ~40 GB. Use Release if space is tight.
