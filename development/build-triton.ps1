param(
    [string]$BuildType = "release"
)

# ============================================================================
# Build Triton-Windows (VS 2026 + Conda + local LLVM)
# ============================================================================
# Usage:
#   powershell -ExecutionPolicy Bypass -File build-triton.ps1 [-BuildType debug|release]
#
# Prerequisites:
#   1. LLVM/MLIR built (see step1-build-llvm.md)
#   2. nlohmann/json downloaded to build\json
#   3. Conda env 'triton-dev' with cmake, ninja, pybind11
#   4. Visual Studio 2026
# ============================================================================

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Find triton root: walk up until we find CMakeLists.txt (supports both
# development/ (1 level) and .github/skills/.../scripts/ (4 levels))
$TritonRoot = $ScriptDir
for ($i = 0; $i -lt 5; $i++) {
    $TritonRoot = Split-Path -Parent $TritonRoot
    if (Test-Path (Join-Path $TritonRoot "CMakeLists.txt")) { break }
}
if (-not (Test-Path (Join-Path $TritonRoot "CMakeLists.txt"))) {
    Write-Host "Error: Cannot find triton-windows root from $ScriptDir" -ForegroundColor Red; exit 1
}
$LlvmBuild = Join-Path $TritonRoot "build\llvm-project\build"
$JsonPath = Join-Path $TritonRoot "build\json"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Triton-Windows Build" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# === Validate prerequisites ===
if (-not (Test-Path "$LlvmBuild\lib\cmake\mlir\MLIRConfig.cmake")) {
    Write-Host "Error: LLVM build not found at $LlvmBuild" -ForegroundColor Red
    Write-Host "Run step 1 first: build LLVM/MLIR" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "$JsonPath\include\nlohmann\json.hpp")) {
    Write-Host "Error: nlohmann/json not found at $JsonPath" -ForegroundColor Red
    exit 1
}
Write-Host "LLVM build: $LlvmBuild" -ForegroundColor Green
Write-Host "JSON path:  $JsonPath" -ForegroundColor Green

# === Import batch environment helper ===
function Import-BatchEnvironment {
    param([string]$BatchFile, [string]$Args = "")
    $tempFile = [IO.Path]::GetTempFileName()
    if ($Args) {
        & C:\Windows\System32\cmd.exe /c " `"$BatchFile`" $Args > nul 2>&1 && set > `"$tempFile`" "
    } else {
        & C:\Windows\System32\cmd.exe /c " `"$BatchFile`" > nul 2>&1 && set > `"$tempFile`" "
    }
    Get-Content $tempFile | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Item "Env:$($matches[1])" $matches[2]
        }
    }
    Remove-Item $tempFile
}

# === Activate Conda env ===
$CondaEnv = "triton-dev"
$CondaRoots = @(
    "$env:USERPROFILE\anaconda3",
    "$env:USERPROFILE\miniconda3",
    "C:\ProgramData\anaconda3",
    "C:\ProgramData\miniconda3",
    "$env:LOCALAPPDATA\anaconda3",
    "$env:LOCALAPPDATA\miniconda3"
)
$CondaRoot = $CondaRoots | Where-Object { Test-Path "$_\Scripts\activate.bat" } | Select-Object -First 1
if (-not $CondaRoot) {
    Write-Host "Error: Conda not found." -ForegroundColor Red; exit 1
}

$condaHook = "$CondaRoot\shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) {
    & $condaHook *>&1 | Out-Null
    conda activate $CondaEnv *>&1 | Where-Object { $_ -is [System.Management.Automation.InformationalRecord] -or $_ -notmatch "WARNING" } | Out-Null
} else {
    Import-BatchEnvironment "$CondaRoot\Scripts\activate.bat" $CondaEnv
}

if (-not $env:CONDA_PREFIX -or $env:CONDA_PREFIX -notlike "*$CondaEnv*") {
    Write-Host "Error: Failed to activate conda env '$CondaEnv'." -ForegroundColor Red
    exit 1
}
Write-Host "Conda env:  $env:CONDA_PREFIX" -ForegroundColor Green

# === Detect MSVC toolset from conda runtime ===
$VcVarsVerFlag = ""
$vcRuntimeInfo = & "$env:CONDA_PREFIX\python.exe" -c "import subprocess; r=subprocess.run(['conda','list','vc14_runtime','-n','$CondaEnv'],capture_output=True,text=True); print(r.stdout)" 2>$null
$vcRuntimeLine = $vcRuntimeInfo | Select-String "^vc14_runtime"
if ($vcRuntimeLine) {
    $ver = ($vcRuntimeLine -split '\s+')[1]
    $major, $minor = $ver.Split('.')[0..1]
    $VcVarsVerFlag = "-vcvars_ver=$major.$minor"
    Write-Host "Conda vc14_runtime: $ver -> MSVC toolset $major.$minor" -ForegroundColor Green
}

# === Find and activate Visual Studio ===
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Host "Error: Visual Studio not found." -ForegroundColor Red; exit 1
}
$vsPath = & $vswhere -latest -property installationPath
$vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    Write-Host "Error: vcvars64.bat not found." -ForegroundColor Red; exit 1
}
Write-Host "Visual Studio: $vsPath" -ForegroundColor Green
Import-BatchEnvironment $vcvars $VcVarsVerFlag

# Validate vcvars actually selected the requested toolset — it may silently
# fall back to the default (14.51) when called from certain environments.
$wrongVersion = $env:VCToolsVersion
if ($VcVarsVerFlag -and $env:VCToolsVersion -and -not $env:VCToolsVersion.StartsWith("$major.$minor")) {
    Write-Host "Warning: Requested toolset $major.$minor but got VCToolsVersion=$env:VCToolsVersion" -ForegroundColor Yellow
    $toolDir = Get-ChildItem "$vsPath\VC\Tools\MSVC" | Where-Object { $_.Name.StartsWith("$major.$minor") } | Select-Object -First 1
    if ($toolDir -and (Test-Path $toolDir.FullName)) {
        $correctVersion = $toolDir.Name
        Write-Host "Forcing toolset $wrongVersion -> $correctVersion" -ForegroundColor Yellow
        $env:VCToolsInstallDir = "$($toolDir.FullName)\"
        $env:VCToolsVersion = $correctVersion
        # Replace ALL occurrences of wrong version in PATH, INCLUDE, LIB, LIBPATH
        foreach ($var in @("PATH", "INCLUDE", "LIB", "LIBPATH")) {
            $val = [Environment]::GetEnvironmentVariable($var)
            if ($val) {
                $val = $val -replace [regex]::Escape($wrongVersion), $correctVersion
                [Environment]::SetEnvironmentVariable($var, $val)
            }
        }
    }
}
Write-Host "MSVC activated (VCToolsVersion=$env:VCToolsVersion)." -ForegroundColor Green

# === Set Triton build environment ===
$env:LLVM_SYSPATH = $LlvmBuild
$env:JSON_SYSPATH = $JsonPath
$env:TRITON_OFFLINE_BUILD = "1"
$env:TRITON_BUILD_PROTON = "0"
$env:TRITON_BUILD_UT = "0"
$env:CMAKE_PREFIX_PATH = "$env:CONDA_PREFIX\Library"
$env:TRITON_APPEND_CMAKE_ARGS = "-DCMAKE_PREFIX_PATH=$env:CONDA_PREFIX\Library"

# Set build type
if ($BuildType -eq "debug") {
    $env:DEBUG = "1"
    $env:TRITON_REL_BUILD_WITH_ASSERTS = ""
} else {
    $env:DEBUG = ""
    $env:TRITON_REL_BUILD_WITH_ASSERTS = "1"
}

Write-Host ""
Write-Host "=== Build Configuration ===" -ForegroundColor Yellow
Write-Host "  LLVM_SYSPATH:         $env:LLVM_SYSPATH"
Write-Host "  JSON_SYSPATH:         $env:JSON_SYSPATH"
Write-Host "  TRITON_OFFLINE_BUILD: $env:TRITON_OFFLINE_BUILD"
Write-Host "  TRITON_BUILD_PROTON:  $env:TRITON_BUILD_PROTON"
Write-Host "  TRITON_BUILD_UT:      $env:TRITON_BUILD_UT"
Write-Host "  CMAKE_PREFIX_PATH:    $env:CMAKE_PREFIX_PATH"
Write-Host "  VCToolsVersion:       $env:VCToolsVersion"
Write-Host "  Build type:           $(if ($BuildType -eq 'debug') {'Debug'} else {'TritonRelBuildWithAsserts'})"
Write-Host ""

# === Clean stale cmake cache if toolset changed ===
$BuildCacheDir = Join-Path $TritonRoot "build\cmake.win-amd64-cpython-*"
$cacheDirs = Get-Item $BuildCacheDir -ErrorAction SilentlyContinue
foreach ($cacheDir in $cacheDirs) {
    $cacheFile = Join-Path $cacheDir "CMakeCache.txt"
    if (Test-Path $cacheFile) {
        $cachedCompiler = Select-String -Path $cacheFile -Pattern "CMAKE_CXX_COMPILER:.*MSVC[\\/]([0-9.]+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
        if ($cachedCompiler -and $cachedCompiler -ne $env:VCToolsVersion) {
            Write-Host "Cleaning stale cmake cache (was $cachedCompiler, now $($env:VCToolsVersion))..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $cacheDir
        }
    }
}

# === Build ===
Write-Host "=== Building Triton (editable install) ===" -ForegroundColor Yellow
Push-Location $TritonRoot
try {
    & pip install --no-build-isolation --verbose -e .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "Triton build complete!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
} finally {
    Pop-Location
}
