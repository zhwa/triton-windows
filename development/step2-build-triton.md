# Step 2: Build Triton-Windows

## Prerequisites
- Step 1 completed (LLVM built at `build/llvm-project/build/`)
- JSON downloaded to `build/json/`
- `pip install "pybind11>=2.13.1,<3.0" "setuptools>=40.8.0" wheel`

## 1. Apply Triton MSVC Patches

Run the inspector first to see what needs fixing:
```powershell
python development\inspect-build.py --fix
```

Apply patches via Copilot: `@workspace Apply MSVC patches for triton-windows build`

## 2. Build

```powershell
# Activate MSVC 14.44 (CRITICAL)
vcvars64.bat -vcvars_ver=14.44

# Set environment
$env:LLVM_SYSPATH = "build\llvm-project\build"  # relative or absolute
$env:JSON_SYSPATH = "build\json"
$env:TRITON_OFFLINE_BUILD = "1"
$env:TRITON_BUILD_PROTON = "0"
$env:TRITON_BUILD_UT = "0"
$env:CMAKE_PREFIX_PATH = "$env:CONDA_PREFIX\Library"  # for zlib/zstd

# Build (editable install)
pip install --no-build-isolation --verbose -e .
```

## 3. Verify

```powershell
python -c "import triton; print(triton.__version__); print(list(triton.backends.backends.keys()))"
# 3.7.0
# ['nvidia']
```

## Critical Notes

1. **MSVC 14.44 required** — 14.51 has template deduction bugs
2. **LLVM must be Release** — matches triton's `/MD` CRT
3. **vcvars must persist** — environment lost = `stddef.h` not found
4. **`CMAKE_PREFIX_PATH`** — conda Library dir needed for zlib/zstd
