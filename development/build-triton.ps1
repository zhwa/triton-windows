param(
    [string]$BuildType = "debug"
)

# ============================================================================
# Build Triton-Windows (VS 2026 + Conda + local LLVM)
# ============================================================================
# Usage:
#   powershell -ExecutionPolicy Bypass -File build-triton.ps1 [-BuildType debug|release]
#
# Prerequisites:
#   1. LLVM/MLIR built (see step1-build-llvm.md)
#   2. nlohmann/json downloaded to references\json
#   3. Conda env 'mlir-dev' with cmake, ninja, pybind11
#   4. Visual Studio 2026
# ============================================================================

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TritonRoot = Split-Path -Parent $ScriptDir
$LlvmBuild = Join-Path $TritonRoot "references\llvm-project\build"
$JsonPath = Join-Path $TritonRoot "references\json"

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
$CondaEnv = "mlir-dev"
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
$vcRuntimeInfo = C:\ProgramData\anaconda3\envs\mlir-dev\python.exe -c "import subprocess; r=subprocess.run(['conda','list','vc14_runtime','-n','mlir-dev'],capture_output=True,text=True); print(r.stdout)" 2>$null
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
Write-Host "MSVC activated (toolset: $VcVarsVerFlag)." -ForegroundColor Green

# === Set Triton build environment ===
$env:LLVM_SYSPATH = $LlvmBuild
$env:JSON_SYSPATH = $JsonPath
$env:TRITON_OFFLINE_BUILD = "1"
$env:TRITON_BUILD_PROTON = "0"
$env:TRITON_BUILD_UT = "0"

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
Write-Host "  Build type:           $(if ($BuildType -eq 'debug') {'Debug'} else {'TritonRelBuildWithAsserts'})"
Write-Host ""

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
