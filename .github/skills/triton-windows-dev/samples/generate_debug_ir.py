#!/usr/bin/env python3
"""Generate TTIR debug files for triton-opt C++ debugging.

Compiles each test kernel to raw TTIR (pre-canonicalization) so that
triton-opt can be launched under a C++ debugger with any pass combination.

Usage (must use triton-dev Python with built triton):
    $env:TRITON_PTXAS_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/ptxas.exe"
    conda run -n triton-dev python generate_debug_ir.py

Output: {repo}/.vscode/debug/*.ttir  (one file per kernel)
"""

import os
import sys
import sysconfig
from pathlib import Path

# Resolve repo root (4 levels up from this script)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

# Windows knobs.py bug: env var name includes ".exe" suffix
# Set TRITON_PTXAS.EXE_PATH if TRITON_PTXAS_PATH is set but the dotted one isn't
exe_suffix = sysconfig.get_config_var("EXE") or ""
if exe_suffix:
    plain_key = "TRITON_PTXAS_PATH"
    dotted_key = f"TRITON_PTXAS{exe_suffix.upper()}_PATH"
    if plain_key in os.environ and dotted_key not in os.environ:
        os.environ[dotted_key] = os.environ[plain_key]

    # Also try common CUDA paths as fallback
    if dotted_key not in os.environ:
        for cuda_dir in [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        ]:
            if os.path.isdir(cuda_dir):
                versions = sorted(os.listdir(cuda_dir), reverse=True)
                for v in versions:
                    ptxas = os.path.join(cuda_dir, v, "bin", "ptxas.exe")
                    if os.path.isfile(ptxas):
                        os.environ[dotted_key] = ptxas
                        break
                if dotted_key in os.environ:
                    break

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir
from triton.backends.nvidia.compiler import CUDABackend

# ── Kernel definitions (no torch dependency) ──────────────────────────────

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Level 1: ElementwiseOp, MemoryOp, ArithTypeConversion"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def reduce_sum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Level 3: ReduceOpToLLVM, atomic_add"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, total)


@triton.jit
def softmax_kernel(inp_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Level 4: max/exp/sum reduction chain, broadcast"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    row = tl.load(inp_ptr + pid * N + offs, mask=mask, other=float('-inf'))
    row_max = tl.max(row, axis=0)
    row_exp = tl.exp(row - row_max)
    row_sum = tl.sum(row_exp, axis=0)
    out = row_exp / row_sum
    tl.store(out_ptr + pid * N + offs, out, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Level 6: DotOp, AccelerateMatmul, scf.for loop"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def transpose_kernel(in_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Level 8: ViewOp, RemoveLayoutConversions, multi-dim grid"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    src = tl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    tl.store(out_ptr + offs_n[:, None] * M + offs_m[None, :],
             tl.trans(src), mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))


@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Level 9: tl.math.exp (libdevice), extern elementwise"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    c = 0.7978845608028654
    arg = c * (x + 0.044715 * x * x * x)
    exp2a = tl.math.exp(2.0 * arg)
    tanh_val = (exp2a - 1.0) / (exp2a + 1.0)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def broadcast_add_kernel(x_ptr, bias_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Level 13: ReorderBroadcast, 2D tensor ops"""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    out = x + bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


# ── Kernel registry ──────────────────────────────────────────────────────

KERNELS = [
    {
        "name": "vector_add",
        "fn": vector_add_kernel,
        "signature": {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "N": "i32"},
        "constexprs": {"BLOCK": 256},
        "passes": "canonicalize -triton-combine -cse",
    },
    {
        "name": "reduce_sum",
        "fn": reduce_sum_kernel,
        "signature": {"x_ptr": "*fp32", "out_ptr": "*fp32", "N": "i32"},
        "constexprs": {"BLOCK": 256},
        "passes": "canonicalize -triton-combine → then -convert-triton-to-tritongpu for ReduceOpToLLVM",
    },
    {
        "name": "softmax",
        "fn": softmax_kernel,
        "signature": {"inp_ptr": "*fp32", "out_ptr": "*fp32", "N": "i32"},
        "constexprs": {"BLOCK": 256},
        "passes": "canonicalize -triton-combine → max/exp/sum reduction chain",
    },
    {
        "name": "matmul",
        "fn": matmul_kernel,
        "signature": {
            "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
            "M": "i32", "N": "i32", "K": "i32",
            "stride_am": "i32", "stride_ak": "i32",
            "stride_bk": "i32", "stride_bn": "i32",
            "stride_cm": "i32", "stride_cn": "i32",
        },
        "constexprs": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
        "passes": "accelerate-matmul, DotOpToLLVM, loop pipeline",
    },
    {
        "name": "transpose",
        "fn": transpose_kernel,
        "signature": {"in_ptr": "*fp32", "out_ptr": "*fp32", "M": "i32", "N": "i32"},
        "constexprs": {"BLOCK_M": 32, "BLOCK_N": 32},
        "passes": "RemoveLayoutConversions, ViewOpToLLVM",
    },
    {
        "name": "gelu",
        "fn": gelu_kernel,
        "signature": {"x_ptr": "*fp32", "out_ptr": "*fp32", "N": "i32"},
        "constexprs": {"BLOCK": 256},
        "passes": "tl.math.exp → libdevice, extern elementwise lowering",
    },
    {
        "name": "broadcast_add",
        "fn": broadcast_add_kernel,
        "signature": {"x_ptr": "*fp32", "bias_ptr": "*fp32", "out_ptr": "*fp32", "M": "i32", "N": "i32"},
        "constexprs": {"BLOCK_M": 32, "BLOCK_N": 64},
        "passes": "reorder-broadcast, 2D tensor ops",
    },
]


def compile_to_ttir(fn, signature, constexprs, target):
    """Compile a kernel function to raw TTIR (pre-canonicalization)."""
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    backend = CUDABackend(target)
    options = backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    codegen_fns = backend.get_codegen_implementation(options)
    module = src.make_ir(target, options, codegen_fns, {}, ctx)
    return str(module)


def main():
    out_dir = REPO / ".vscode" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = GPUTarget("cuda", 80, 32)  # SM80 (Ampere)

    for k in KERNELS:
        print(f"  Generating {k['name']}.ttir ... ", end="", flush=True)
        try:
            ttir = compile_to_ttir(k["fn"], k["signature"], k["constexprs"], target)
            path = out_dir / f"{k['name']}.ttir"
            path.write_text(ttir, encoding="utf-8")
            print(f"OK ({len(ttir)} bytes)")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nGenerated {len(list(out_dir.glob('*.ttir')))} TTIR files in {out_dir}")
    print("\nTo debug a kernel's compiler pass in VS Code:")
    print("  1. Open launch.json -> select 'triton-opt: debug kernel pass'")
    print("  2. Pick the kernel and pass from the dropdown")
    print("  3. Set C++ breakpoints in the compiler source")
    print("  4. Press F5")


if __name__ == "__main__":
    main()
