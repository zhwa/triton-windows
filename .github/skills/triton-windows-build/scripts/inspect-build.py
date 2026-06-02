#!/usr/bin/env python3
"""
Triton-Windows Build Environment Inspector

Scans the build environment and source code for known MSVC compatibility issues.
Works on a fresh clone — no build artifacts required.

Usage:
    python inspect-build.py [--fix] [--llvm-dir PATH] [--json-dir PATH]

Exit codes:  0 = all passed,  N = number of blockers
"""

import argparse, os, re, subprocess, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Ensure Unicode output works on Windows consoles (cp1252 can't encode ✓/✗)
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

@dataclass
class Issue:
    id: str; title: str; severity: str; description: str; fix_hint: str
    detected: bool = False; details: str = ""

def make_issues():
    return {
        "ENV-01": Issue("ENV-01", "MSVC Toolset Version", "BLOCKER",
            "VS 2026 default toolset (14.51) has template bugs. Need 14.44 (v143).",
            "vcvars64.bat -vcvars_ver=14.44"),
        "ENV-02": Issue("ENV-02", "pybind11 Version", "BLOCKER",
            "pybind11 3.0+ breaks PYBIND11_MODULE. Need 2.x.",
            'pip install "pybind11>=2.13.1,<3.0"'),
        "ENV-03": Issue("ENV-03", "LLVM Source Not Cloned", "BLOCKER",
            "LLVM source must be at build/llvm-project/.",
            "git clone --filter=blob:none https://github.com/llvm/llvm-project.git build/llvm-project"),
        "ENV-04": Issue("ENV-04", "LLVM Commit Mismatch", "BLOCKER",
            "LLVM must match cmake/llvm-hash.txt.",
            "cd build/llvm-project && git checkout <hash from cmake/llvm-hash.txt>"),
        "ENV-05": Issue("ENV-05", "LLVM Not Built", "BLOCKER",
            "LLVM/MLIR must be built before triton.",
            "powershell -File development/build-llvm.ps1 -Action all -BuildType release"),
        "ENV-06": Issue("ENV-06", "LLVM Build Type Mismatch", "BLOCKER",
            "LLVM must be Release (/MD) to match triton's CRT.",
            "Rebuild LLVM with -BuildType release"),
        "ENV-07": Issue("ENV-07", "nlohmann/json Not Found", "BLOCKER",
            "JSON headers required for offline build.",
            "Download v3.11.3 to build/json/"),
        "ENV-08": Issue("ENV-08", "setuptools Not Installed", "BLOCKER",
            "setuptools required as build backend.",
            'pip install "setuptools>=40.8.0"'),
        "LLVM-01": Issue("LLVM-01", "LLVM /utf-8 Missing", "ERROR",
            "MSVC needs /utf-8 for non-ASCII sources.",
            'Add add_compile_options("/utf-8") in llvm/CMakeLists.txt if(MSVC)'),
        "LLVM-02": Issue("LLVM-02", "PassPlugin dllexport Missing", "ERROR",
            "LLVM_ATTRIBUTE_WEAK unsupported on MSVC.",
            "Patch PassPlugin.h with __declspec(dllexport)"),
        "LLVM-03": Issue("LLVM-03", "TypeID.h Ambiguous Namespace", "ERROR",
            "Unqualified detail:: ambiguous with triton dialects.",
            "Use ::mlir::detail:: in TypeID.h"),
        "TRI-01": Issue("TRI-01", "GSan clang++ Required", "ERROR",
            "nvidia/CMakeLists.txt has REQUIRED for clang++.",
            "Remove REQUIRED, guard with if(TRITON_GSAN_CLANGXX)"),
        "TRI-02": Issue("TRI-02", "AMD LLD Required", "ERROR",
            "amd/CMakeLists.txt has find_package(LLD REQUIRED).",
            "Make LLD optional, guard plugin with if(LLD_FOUND)"),
        "TRI-03": Issue("TRI-03", "FileCheck.exe Extension", "ERROR",
            "FileCheck copied without .exe on Windows.",
            "Add WIN32 conditional for .exe suffix"),
        "TRI-04": Issue("TRI-04", "/bigobj Missing", "ERROR",
            "/bigobj only in TritonRelBuildWithAsserts.",
            "Add add_compile_options(/bigobj) for MSVC"),
        "TRI-05": Issue("TRI-05", "walk() Template Deduction", "ERROR",
            "MSVC can't deduce walk() from lambdas.",
            "Use ::mlir::detail::walk free function"),
        "TRI-06": Issue("TRI-06", "dlfcn.h Missing", "ERROR",
            "POSIX dlfcn.h unavailable on Windows.",
            "Create/use dlfcn_win32.h shim"),
        "TRI-07": Issue("TRI-07", "Ternary Type Mismatch", "ERROR",
            "MSVC requires same type in ?: branches.",
            "Use if/else or static_cast"),
        "TRI-08": Issue("TRI-08", "constexpr Lambda Capture", "WARNING",
            "MSVC may need explicit constexpr capture.",
            "Add to lambda capture list"),
        "TRI-09": Issue("TRI-09", "getPartition((int)0)", "ERROR",
            "Ambiguous overload with int literal.",
            "Change (int)0 to (unsigned)0"),
        "TRI-10": Issue("TRI-10", "GSan.h GCC Types", "ERROR",
            "__SIZE_TYPE__ unavailable on MSVC.",
            "#ifdef _MSC_VER with <cstdint>"),
        "TRI-11": Issue("TRI-11", "Proton Unconditional", "ERROR",
            "proton always in TRITON_PLUGIN_NAMES.",
            "Guard with if(TRITON_BUILD_PROTON)"),
        "TRI-12": Issue("TRI-12", "AMDGPU Libs Unconditional", "ERROR",
            "LLVMAMDGPUCodeGen linked without check.",
            "Make conditional on lib existence"),
        "TRI-13": Issue("TRI-13", "windows.h NOMINMAX", "ERROR",
            "windows.h min/max macros break LLVM.",
            "Define NOMINMAX before windows.h"),
        "TRI-14": Issue("TRI-14", "Backend Discovery", "WARNING",
            "Import failures crash backend init.",
            "Add try/except around imports"),
        "TRI-15": Issue("TRI-15", "GSan Testing on MSVC", "ERROR",
            "gsan_testing.cc uses GCC-only types.",
            "Exclude on MSVC, stub init_gsan_testing"),
        "TRI-16": Issue("TRI-16", "Missing LLVM Libs", "ERROR",
            "LLVMCodeGen/MIRParser/Plugins not linked.",
            "Add to TRITON_LIBRARIES"),
        "TRI-17": Issue("TRI-17", "AMD Not Removed from Tuple", "WARNING",
            "AMD stays in backends when plugin not built.",
            "Add if(NOT TARGET TritonAMD) removal"),
    }

def fc(p: Path, pat: str) -> bool:
    """file contains pattern"""
    if not p.exists(): return False
    return bool(re.search(pat, p.read_text(encoding="utf-8", errors="replace")))

def fnc(p: Path, pat: str) -> bool:
    """file NOT contains pattern"""
    if not p.exists(): return True
    return not re.search(pat, p.read_text(encoding="utf-8", errors="replace"))

def cmd(args, **kw):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10, **kw)
        return r.stdout.strip() if r.returncode == 0 else None
    except: return None

def inspect(root: Path, llvm: Path, json_d: Path):
    issues = make_issues()

    # ENV-01: MSVC
    try:
        r = subprocess.run(["cl"], capture_output=True, text=True, timeout=10)
        m = re.search(r"Version (\d+\.\d+)", r.stderr or "")
        if m:
            if not m.group(1).startswith("19.44"):
                issues["ENV-01"].detected = True
                issues["ENV-01"].details = f"Have {m.group(1)}, need 19.44"
        else:
            issues["ENV-01"].detected = True
            issues["ENV-01"].details = "cl.exe not on PATH"
    except:
        issues["ENV-01"].detected = True
        issues["ENV-01"].details = "cl.exe not found"

    # ENV-02: pybind11
    pb = cmd([sys.executable, "-c", "import pybind11;print(pybind11.__version__)"])
    if not pb: issues["ENV-02"].detected = True; issues["ENV-02"].details = "Not installed"
    elif pb.startswith("3."): issues["ENV-02"].detected = True; issues["ENV-02"].details = f"Have {pb}"

    # ENV-03: LLVM source
    if not (llvm / "llvm" / "CMakeLists.txt").exists():
        issues["ENV-03"].detected = True; issues["ENV-03"].details = str(llvm)

    # ENV-04: LLVM commit
    hf = root / "cmake" / "llvm-hash.txt"
    if hf.exists() and (llvm / ".git").exists():
        exp = hf.read_text().strip()
        act = cmd(["git", "rev-parse", "HEAD"], cwd=str(llvm))
        if act and not act.startswith(exp[:12]):
            issues["ENV-04"].detected = True
            issues["ENV-04"].details = f"Need {exp[:12]}..., have {act[:12]}..."

    # ENV-05: LLVM built
    lb = llvm / "build"
    if not (lb / "lib" / "cmake" / "mlir" / "MLIRConfig.cmake").exists():
        issues["ENV-05"].detected = True
        issues["ENV-05"].details = "Not built" if (llvm/"llvm").exists() else "Not cloned"

    # ENV-06: LLVM build type
    cc = lb / "CMakeCache.txt"
    if cc.exists():
        m = re.search(r"CMAKE_BUILD_TYPE:STRING=(\w+)", cc.read_text(errors="replace"))
        if m and m.group(1) != "Release":
            issues["ENV-06"].detected = True; issues["ENV-06"].details = f"Have {m.group(1)}"

    # ENV-07: JSON
    if not (json_d / "include" / "nlohmann" / "json.hpp").exists():
        issues["ENV-07"].detected = True

    # ENV-08: setuptools
    if cmd([sys.executable, "-c", "import setuptools;print(1)"]) is None:
        issues["ENV-08"].detected = True

    # LLVM patches (only check if source exists)
    lc = llvm / "llvm" / "CMakeLists.txt"
    if lc.exists() and fnc(lc, r'add_compile_options\("/utf-8"\)'):
        issues["LLVM-01"].detected = True
    pp = llvm / "llvm" / "include" / "llvm" / "Plugins" / "PassPlugin.h"
    if pp.exists() and fnc(pp, r'__declspec\(dllexport\)'):
        issues["LLVM-02"].detected = True
    tid = llvm / "mlir" / "include" / "mlir" / "Support" / "TypeID.h"
    if tid.exists() and fc(tid, r'return detail::TypeIDResolver') and fnc(tid, r'::mlir::detail::TypeIDResolver'):
        issues["LLVM-03"].detected = True

    # Triton source checks
    rc = root / "CMakeLists.txt"
    if rc.exists():
        t = rc.read_text(encoding="utf-8", errors="replace")
        if re.search(r'configure_file.*FileCheck"', t) and 'FileCheck.exe' not in t:
            issues["TRI-03"].detected = True
        if not re.search(r'add_compile_options.*bigobj', t):
            issues["TRI-04"].detected = True
        # proton
        lines = t.split('\n'); in_if = False
        for l in lines:
            if 'TRITON_BUILD_PROTON' in l and 'if' in l.lower(): in_if = True
            if in_if and 'endif' in l.lower(): in_if = False
            if 'TRITON_PLUGIN_NAMES' in l and '"proton"' in l and not in_if:
                issues["TRI-11"].detected = True
        # TRI-12: Check if AMDGPU is in unconditional TRITON_LIBRARIES list
        in_exists_block = False
        for line in lines:
            if 'EXISTS' in line and ('AMDGPU' in line or 'LLVMAMDGPUCodeGen' in line):
                in_exists_block = True
            if 'LLVMAMDGPUCodeGen' in line and 'list' in line.lower() and not in_exists_block:
                issues["TRI-12"].detected = True; break
            if in_exists_block and 'endif' in line.lower():
                in_exists_block = False
        if 'gsan_testing.cc' in t and 'NOT MSVC' not in t:
            issues["TRI-15"].detected = True
        if 'LLVMNVPTXCodeGen' in t and 'LLVMMIRParser' not in t:
            issues["TRI-16"].detected = True
        if 'TRITON_CODEGEN_BACKENDS' in t and 'NOT TARGET TritonAMD' not in t:
            issues["TRI-17"].detected = True

    nc = root / "third_party" / "nvidia" / "CMakeLists.txt"
    if nc.exists() and fc(nc, r'REQUIRED\s+NO_DEFAULT_PATH'):
        issues["TRI-01"].detected = True
    ac = root / "third_party" / "amd" / "CMakeLists.txt"
    if ac.exists() and fc(ac, r'find_package\(LLD\s+REQUIRED'):
        issues["TRI-02"].detected = True

    for p in [root/"lib"/"Analysis"/"Utility.cpp",
              root/"lib"/"Dialect"/"TritonGPU"/"Transforms"/"Utility.cpp"]:
        if p.exists():
            t = p.read_text(encoding="utf-8", errors="replace")
            if re.search(r'(?<!\w)\.walk\(\[', t) or re.search(r'->\s*walk\(\[', t):
                issues["TRI-05"].detected = True; issues["TRI-05"].details = p.name; break

    for f in [root/"third_party"/"nvidia"/"triton_nvidia.cc",
              root/"third_party"/"nvidia"/"include"/"cublas_instance.h"]:
        if f.exists() and fc(f, r'#include\s*<dlfcn\.h>'):
            issues["TRI-06"].detected = True; issues["TRI-06"].details = f.name; break

    tm = root/"third_party"/"nvidia"/"lib"/"TritonNVIDIAGPUToLLVM"/"TensorMemoryToLLVM.cpp"
    if tm.exists() and fc(tm, r'isMin\s*\?\s*LLVM::Minimum'):
        issues["TRI-07"].detected = True
    bp = root/"third_party"/"amd"/"lib"/"TritonAMDGPUTransforms"/"BlockPingpong.cpp"
    if bp.exists():
        t = bp.read_text(errors="replace")
        if re.search(r'useAsyncCopy\s*\?\s*asyncCopyOps', t) and 'static_cast' not in t:
            issues["TRI-07"].detected = True

    ob = root/"third_party"/"amd"/"lib"/"TritonAMDGPUTransforms"/"OptimizeBufferOpPtr.cpp"
    if ob.exists():
        m = re.search(r'\[([\w,\s]*)\]\s*\(\s*const llvm::APInt', ob.read_text(errors="replace"))
        if m and 'maxOffsetValue' not in m.group(1): issues["TRI-08"].detected = True

    pl = root/"lib"/"Dialect"/"TritonGPU"/"Transforms"/"WarpSpecialization"/"PartitionLoops.cpp"
    if pl.exists() and fc(pl, r'getPartition\(\(int\)0\)'): issues["TRI-09"].detected = True

    gs = root/"python"/"triton"/"experimental"/"gsan"/"src"/"GSan.h"
    if gs.exists():
        t = gs.read_text(errors="replace")
        if '__SIZE_TYPE__' in t and '_MSC_VER' not in t: issues["TRI-10"].detected = True

    dl = root/"third_party"/"nvidia"/"include"/"dlfcn_win32.h"
    if dl.exists() and fc(dl, r'windows\.h') and not fc(dl, r'NOMINMAX'):
        issues["TRI-13"].detected = True

    bi = root/"python"/"triton"/"backends"/"__init__.py"
    if bi.exists():
        t = bi.read_text(errors="replace")
        if 'entry_points' in t and not re.search(r'except.*Import', t):
            issues["TRI-14"].detected = True

    return issues

# ─── Output ─────────────────────────────────────────────────────────────

C = {"BLOCKER":"\033[91m","ERROR":"\033[93m","WARNING":"\033[33m","OK":"\033[92m","R":"\033[0m"}

def report(issues, fix):
    cats = {"BLOCKER":[],"ERROR":[],"WARNING":[]}; ok = []
    for i in issues.values():
        if i.detected: cats.get(i.severity, cats["WARNING"]).append(i)
        else: ok.append(i)
    nb,ne,nw = len(cats["BLOCKER"]),len(cats["ERROR"]),len(cats["WARNING"])
    print(f"\n{'='*72}\n  Triton-Windows Build Inspector\n{'='*72}")
    print(f"\n  {len(issues)} checks: {C['BLOCKER']}{nb} blockers{C['R']}, "
          f"{C['ERROR']}{ne} errors{C['R']}, {C['WARNING']}{nw} warnings{C['R']}, "
          f"{C['OK']}✓ {len(ok)} OK{C['R']}")
    for sev,lbl in [("BLOCKER","BLOCKERS"),("ERROR","ERRORS"),("WARNING","WARNINGS")]:
        if not cats[sev]: continue
        print(f"\n{'─'*72}\n  {C[sev]}{lbl}{C['R']}\n{'─'*72}")
        for i in cats[sev]:
            print(f"\n  {C['BLOCKER']}✗{C['R']} [{i.id}] {i.title}")
            print(f"     {i.description}")
            if i.details: print(f"     → {i.details}")
            if fix: print(f"     {C['OK']}Fix:{C['R']} {i.fix_hint}")
    if ok:
        print(f"\n{'─'*72}\n  {C['OK']}✓ PASSED{C['R']}\n{'─'*72}")
        for i in ok: print(f"  {C['OK']}✓{C['R']} [{i.id}] {i.title}")
    print()
    if nb+ne+nw == 0: print(f"  {C['OK']}✓ All checks passed! Ready to build.{C['R']}")
    elif nb: print(f"  {C['BLOCKER']}✗ Fix {nb} blocker(s) first.{C['R']} Use --fix for commands.")
    else: print(f"  {C['ERROR']}✗ {ne+nw} issue(s) to fix.{C['R']} Use --fix for commands.")
    print()

def main():
    ap = argparse.ArgumentParser(description="Triton-Windows Build Inspector")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--llvm-dir", type=Path, default=None)
    ap.add_argument("--json-dir", type=Path, default=None)
    a = ap.parse_args()
    sd = Path(__file__).resolve().parent
    # Support running from development/ (1 level up) or .github/skills/.../scripts/ (4 levels up)
    for depth in [1, 4]:
        root = sd
        for _ in range(depth):
            root = root.parent
        if (root / "CMakeLists.txt").exists():
            break
    else:
        root = Path.cwd()
    if not (root/"CMakeLists.txt").exists():
        print("Error: Run from triton-windows root."); sys.exit(1)
    llvm = a.llvm_dir or root/"build"/"llvm-project"
    json_d = a.json_dir or root/"build"/"json"
    issues = inspect(root, llvm, json_d)
    report(issues, a.fix)
    sys.exit(sum(1 for i in issues.values() if i.detected and i.severity == "BLOCKER"))

if __name__ == "__main__":
    main()
