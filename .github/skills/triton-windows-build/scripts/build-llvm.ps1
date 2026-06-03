param(
    [string]$Action = "all",
    [string]$BuildType = "release"
)

# ============================================================================
# Build LLVM/MLIR for Triton-Windows (VS 2026 + Conda)
# ============================================================================
# Usage:
#   powershell -ExecutionPolicy Bypass -File build-llvm.ps1 [-Action all|configure|build|rebuild|clean] [-BuildType release|debug]
#
# Prerequisites:
#   1. Visual Studio 2026 (any edition)
#   2. Conda env 'triton-dev' with cmake and ninja:
#      conda create -n triton-dev -c conda-forge python=3.12 ninja cmake pip
#   3. LLVM repo checked out at the commit from triton-windows/cmake/llvm-hash.txt
#   4. MSVC patches applied (see step1-build-llvm.md)
#
# Place this script in the development/ directory of triton-windows.
# LLVM source is expected at build/llvm-project/ (next to the llvm/ folder).
# ============================================================================

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
$LlvmSrcDir = Join-Path $TritonRoot "build\llvm-project"
$BuildDir = Join-Path $LlvmSrcDir "build"

if (-not (Test-Path "$LlvmSrcDir\llvm")) {
    Write-Host "Error: LLVM source not found at $LlvmSrcDir" -ForegroundColor Red
    Write-Host "Clone it: git clone --filter=blob:none https://github.com/llvm/llvm-project.git `"$LlvmSrcDir`"" -ForegroundColor Yellow
    exit 1
}

$CmakeBuildType = if ($BuildType -eq "debug") { "Debug" } else { "Release" }

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "LLVM/MLIR Build for Triton ($CmakeBuildType)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

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

# Use conda's PowerShell hook (suppressing stderr warnings)
$condaHook = "$CondaRoot\shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) {
    & $condaHook *>&1 | Out-Null
    conda activate $CondaEnv *>&1 | Where-Object { $_ -is [System.Management.Automation.InformationalRecord] -or $_ -notmatch "WARNING" } | Out-Null
} else {
    # Fallback: batch activate
    Import-BatchEnvironment "$CondaRoot\Scripts\activate.bat" $CondaEnv
}

if (-not $env:CONDA_PREFIX -or $env:CONDA_PREFIX -notlike "*$CondaEnv*") {
    Write-Host "Error: Failed to activate conda env '$CondaEnv'. CONDA_PREFIX=$env:CONDA_PREFIX" -ForegroundColor Red
    exit 1
}
Write-Host "Conda env: $env:CONDA_PREFIX" -ForegroundColor Green

# === Detect MSVC toolset from conda runtime ===
$VcVarsVerFlag = ""
$vcRuntimeInfo = & "$env:CONDA_PREFIX\python.exe" -c "import subprocess; r=subprocess.run(['conda','list','vc14_runtime','-n','$CondaEnv'],capture_output=True,text=True); print(r.stdout)" 2>$null
$vcRuntimeLine = $vcRuntimeInfo | Select-String "^vc14_runtime"
if ($vcRuntimeLine) {
    $ver = ($vcRuntimeLine -split '\s+')[1]
    $major, $minor = $ver.Split('.')[0..1]
    $VcVarsVerFlag = "-vcvars_ver=$major.$minor"
    Write-Host "Conda vc14_runtime: $ver -> MSVC toolset $major.$minor" -ForegroundColor Green
} else {
    Write-Host "Warning: vc14_runtime not found in conda env '$CondaEnv'." -ForegroundColor Yellow
    Write-Host "  VS 2026 default toolset (14.51) has known template bugs." -ForegroundColor Yellow
    Write-Host "  Install: conda install -n $CondaEnv -c conda-forge vc14_runtime=14.44" -ForegroundColor Yellow
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

# From here on, stop on errors
$ErrorActionPreference = "Stop"

# === Actions ===
function Invoke-Configure {
    Write-Host ""
    Write-Host "=== Configuring LLVM ($CmakeBuildType) ===" -ForegroundColor Yellow
    $cmakeArgs = @(
        "-B", $BuildDir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=$CmakeBuildType",
        "-DCMAKE_C_COMPILER=cl",
        "-DCMAKE_CXX_COMPILER=cl",
        "-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded",
        "-DLLVM_ENABLE_PROJECTS=mlir;llvm",
        "-DLLVM_TARGETS_TO_BUILD=host;NVPTX",
        "-DLLVM_BUILD_TOOLS=ON",
        "-DLLVM_BUILD_UTILS=ON",
        "-DLLVM_INSTALL_UTILS=ON",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_ENABLE_DIA_SDK=OFF",
        "-DMLIR_BUILD_MLIR_C_DYLIB=OFF",
        "$LlvmSrcDir\llvm"
    )
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "Configure failed!" -ForegroundColor Red; exit 1 }
}

function Invoke-Build {
    Write-Host ""
    Write-Host "=== Building LLVM ===" -ForegroundColor Yellow
    & cmake --build $BuildDir --config $CmakeBuildType
    if ($LASTEXITCODE -ne 0) { Write-Host "Build failed!" -ForegroundColor Red; exit 1 }
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "Build complete!" -ForegroundColor Green
    Write-Host "LLVM build dir: $BuildDir" -ForegroundColor Green
    Write-Host "To build triton: `$Env:LLVM_SYSPATH = `"$BuildDir`"" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
}

switch ($Action) {
    "clean" {
        Write-Host "Cleaning $BuildDir..."
        if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
        Write-Host "Done."
    }
    "configure" { Invoke-Configure }
    "build" { Invoke-Build }
    "rebuild" {
        if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
        Invoke-Configure
        Invoke-Build
    }
    "all" {
        Invoke-Configure
        Invoke-Build
    }
}
