# Step 1: Clone and Build LLVM/MLIR

External dependencies (LLVM, JSON) live in `build/` which is git-ignored.

## 1. Clone LLVM

```powershell
cd <triton-windows-root>
git clone --filter=blob:none https://github.com/llvm/llvm-project.git build/llvm-project
```

## 2. Checkout Required Commit

```powershell
$hash = Get-Content cmake\llvm-hash.txt
cd build\llvm-project
git checkout $hash
```

## 3. Apply MSVC Patches

```powershell
git apply development\llvm-msvc-patches.diff
```

Or ask Copilot: `@workspace Apply MSVC patches for triton-windows build`

**3 patches needed:**
- `llvm/CMakeLists.txt` — `/utf-8` + `_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING`
- `llvm/include/llvm/Plugins/PassPlugin.h` — `__declspec(dllexport)` for MSVC
- `mlir/include/mlir/Support/TypeID.h` — `::mlir::detail::` qualified

## 4. Download nlohmann/json

```powershell
cd <triton-windows-root>
Invoke-WebRequest "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip" -OutFile json.zip
Expand-Archive json.zip -DestinationPath build\json -Force
Remove-Item json.zip
```

## 5. Build LLVM

```powershell
powershell -ExecutionPolicy Bypass -File development\build-llvm.ps1 -Action all -BuildType release
```

**MUST use Release** — triton's CRT is `/MD` (Release). Debug LLVM uses `/MDd`.

### Build options

| Action | Description |
|---|---|
| `all` | Configure + build (default) |
| `configure` | CMake configure only |
| `build` | Build only |
| `rebuild` | Clean + configure + build |
| `clean` | Delete build directory |

## 6. Verify

```powershell
Test-Path build\llvm-project\build\bin\mlir-opt.exe        # True
Test-Path build\llvm-project\build\lib\cmake\mlir\MLIRConfig.cmake  # True
```

## 7. Run Inspector

```powershell
python development\inspect-build.py --fix
```

All ENV and LLVM checks should pass before proceeding to Step 2.
