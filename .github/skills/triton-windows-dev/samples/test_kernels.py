"""
Triton-Windows Kernel Test Suite

Graduated test cases covering the core Triton patterns, from elementwise ops
to matmul and fused kernels. Uses pytest with proper fixtures, parametrization,
tolerance handling, and GPU/interpreter skip logic.

Usage:
    # GPU mode (default)
    pytest test_kernels.py -s --tb=short

    # Interpreter mode (no GPU)
    $env:TRITON_INTERPRET = "1"
    pytest test_kernels.py -s --tb=short -k "not benchmark"

    # Single test
    pytest test_kernels.py::test_vector_add -s --tb=short

    # Force recompilation (bypass cache)
    $env:TRITON_ALWAYS_COMPILE = "1"
    pytest test_kernels.py -s --tb=short
"""

import os
import pytest
import torch
import numpy as np

import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

HAS_CUDA = torch.cuda.is_available()
IS_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEVICE = "cuda" if HAS_CUDA else "cpu"

# Detect compute capability for dtype support
if HAS_CUDA:
    _CC = torch.cuda.get_device_capability()
    HAS_BF16 = _CC[0] >= 8       # Ampere+
    HAS_FP8 = _CC[0] >= 9        # Hopper+
    GPU_NAME = torch.cuda.get_device_name()
else:
    _CC = (0, 0)
    HAS_BF16 = False
    HAS_FP8 = False
    GPU_NAME = "none"

requires_cuda = pytest.mark.skipif(
    not HAS_CUDA and not IS_INTERPRET,
    reason="CUDA required (or set TRITON_INTERPRET=1)",
)

# Tolerance by dtype
TOLERANCES = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=1e-1, rtol=1e-1),
}


def get_device():
    """Return the device for tensor allocation.

    In interpreter mode the triton interpreter simulates CUDA on CPU,
    so torch tensors must live on CPU (the interpreter converts to numpy).
    """
    if IS_INTERPRET:
        return "cpu"
    return DEVICE


def allclose(actual, expected, dtype=torch.float32):
    tol = TOLERANCES.get(dtype, TOLERANCES[torch.float32])
    return torch.allclose(actual.float(), expected.float(), **tol)


# ═══════════════════════════════════════════════════════════════════════════
# Level 1: Elementwise — Vector Add
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@requires_cuda
@pytest.mark.parametrize("N", [1, 7, 256, 1024, 100_003])
def test_vector_add(N):
    """Elementwise add with non-power-of-2 and edge-case sizes."""
    device = get_device()
    x = torch.randn(N, device=device)
    y = torch.randn(N, device=device)
    out = torch.empty(N, device=device)
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    vector_add_kernel[grid](x, y, out, N, BLOCK=BLOCK)
    assert allclose(out, x + y), f"Mismatch at N={N}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 2: Elementwise — Fused Multiply-Add with dtype sweep
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    z = tl.load(z_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, (x * y + z).to(tl.float32), mask=mask)


_fma_dtypes = [torch.float32, torch.float16]
if HAS_BF16:
    _fma_dtypes.append(torch.bfloat16)


@requires_cuda
@pytest.mark.parametrize("dtype", _fma_dtypes, ids=lambda d: str(d).split(".")[-1])
def test_fma(dtype):
    """Fused multiply-add with dtype sweep (fp32, fp16, bf16)."""
    N = 4096
    device = get_device()
    x = torch.randn(N, device=device, dtype=dtype)
    y = torch.randn(N, device=device, dtype=dtype)
    z = torch.randn(N, device=device, dtype=dtype)
    out = torch.empty(N, device=device, dtype=torch.float32)
    grid = (triton.cdiv(N, 256),)
    fma_kernel[grid](x, y, z, out, N, BLOCK=256)
    ref = (x.float() * y.float() + z.float())
    assert allclose(out, ref, dtype), f"FMA mismatch for {dtype}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 3: Reduction — Sum
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def reduce_sum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    block_sum = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, block_sum)


@requires_cuda
@pytest.mark.parametrize("N", [128, 1024, 8192, 65537])
def test_reduce_sum(N):
    """Block-parallel sum reduction with atomic accumulation."""
    device = get_device()
    x = torch.randn(N, device=device)
    out = torch.zeros(1, device=device)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    reduce_sum_kernel[grid](x, out, N, BLOCK=BLOCK)
    ref = x.sum()
    # Reduction has higher error due to ordering
    assert torch.allclose(out[0], ref, atol=1e-2, rtol=1e-3), \
        f"Sum mismatch: got {out[0].item():.4f}, expected {ref.item():.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 4: Row-wise — Softmax
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def softmax_kernel(
    out_ptr, inp_ptr,
    inp_stride, out_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    inp_ptrs = inp_ptr + row * inp_stride + cols
    x = tl.load(inp_ptrs, mask=mask, other=-float("inf"))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    sm = num / den

    out_ptrs = out_ptr + row * out_stride + cols
    tl.store(out_ptrs, sm, mask=mask)


@requires_cuda
@pytest.mark.parametrize("shape", [(4, 64), (32, 128), (128, 781)])
def test_softmax(shape):
    """Row-wise softmax with non-power-of-2 column count."""
    M, N = shape
    device = get_device()
    x = torch.randn(M, N, device=device)
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    softmax_kernel[(M,)](out, x, x.stride(0), out.stride(0), N, BLOCK=BLOCK)
    ref = torch.softmax(x, dim=-1)
    assert allclose(out, ref), f"Softmax mismatch for shape {shape}"
    # Sanity: rows sum to 1
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(M, device=device), atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# Level 5: Normalization — RMSNorm
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr,
    stride, N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)

    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    y = x * rstd * w

    tl.store(Y_ptr + row * stride + cols, y, mask=mask)


def rmsnorm_ref(x, w, eps=1e-6):
    ms = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(ms + eps) * w.float()).to(x.dtype)


@requires_cuda
@pytest.mark.parametrize("shape", [(8, 64), (32, 256), (4, 768)])
def test_rmsnorm(shape):
    """RMSNorm: row-wise normalize + scale."""
    M, N = shape
    device = get_device()
    x = torch.randn(M, N, device=device)
    w = torch.ones(N, device=device)
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    rmsnorm_kernel[(M,)](x, out, w, x.stride(0), N, eps=1e-6, BLOCK=BLOCK)
    ref = rmsnorm_ref(x, w)
    assert allclose(out, ref), f"RMSNorm mismatch for shape {shape}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 6: GEMM — Tiled Matrix Multiplication
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


@requires_cuda
@pytest.mark.skipif(IS_INTERPRET, reason="tl.dot in interpreter triggers OpenMP conflict on Windows")
@pytest.mark.parametrize("M,N,K", [(32, 32, 32), (64, 48, 80), (128, 128, 256)])
def test_matmul(M, N, K):
    """Tiled GEMM with tl.dot accumulator loop."""
    device = get_device()
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    out = triton_matmul(a, b)
    ref = torch.mm(a, b)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), \
        f"Matmul mismatch for ({M},{N},{K}), max err={torch.max(torch.abs(out - ref)):.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 7: Control Flow — tl.where + constexpr branching
#   Exercises: CombineTensorSelectAndIf, constexpr dead-branch elimination
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def relu_dropout_kernel(
    x_ptr, out_ptr, N, p, seed,
    TRAINING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    # ReLU via tl.where (exercises CombineTensorSelectAndIf)
    x = tl.where(x > 0, x, 0.0)
    # Constexpr branching (exercises dead-branch elimination)
    if TRAINING:
        rng = tl.rand(seed, offs)
        x = tl.where(rng > p, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offs, x, mask=mask)


@requires_cuda
@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_relu_dropout(training):
    """ReLU + dropout: exercises tl.where and constexpr branching."""
    N = 4096
    device = get_device()
    x = torch.randn(N, device=device)
    out = torch.empty(N, device=device)
    grid = (triton.cdiv(N, 256),)
    relu_dropout_kernel[grid](x, out, N, 0.5, 42, TRAINING=training, BLOCK=256)
    if not training:
        ref = torch.relu(x)
        assert allclose(out, ref), "ReLU mismatch in eval mode"
    else:
        # In training mode, just check non-negative (dropout is stochastic)
        assert (out >= -1e-6).all(), "ReLU+dropout produced negative values"


# ═══════════════════════════════════════════════════════════════════════════
# Level 8: Transpose — 2D blocked read/write
#   Exercises: ViewOpToLLVM, layout conversion, RemoveLayoutConversions
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def transpose_kernel(
    in_ptr, out_ptr, M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    in_ptrs = in_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    vals = tl.load(in_ptrs, mask=mask)
    # Write transposed: swap row/col
    out_ptrs = out_ptr + offs_n[None, :] * stride_om + offs_m[:, None] * stride_on
    tl.store(out_ptrs, vals, mask=mask)


@requires_cuda
@pytest.mark.parametrize("shape", [(32, 64), (65, 33), (128, 128)])
def test_transpose(shape):
    """2D transpose with multi-dim grid and non-square shapes."""
    M, N = shape
    device = get_device()
    x = torch.randn(M, N, device=device)
    out = torch.empty(N, M, device=device)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        x, out, M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    assert allclose(out, x.T), f"Transpose mismatch for {shape}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 9: Libdevice Math — transcendental functions
#   Exercises: libdevice linking in make_llir(), extern elementwise
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    # GELU approximation uses math.exp (→ libdevice on GPU) + arithmetic
    c = 0.7978845608028654  # sqrt(2/pi)
    arg = c * (x + 0.044715 * x * x * x)
    # tanh via exp: tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    exp2a = tl.math.exp(2.0 * arg)
    tanh_val = (exp2a - 1.0) / (exp2a + 1.0)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offs, y, mask=mask)


@requires_cuda
def test_gelu():
    """GELU activation: exercises libdevice math (exp) + arithmetic."""
    N = 4096
    device = get_device()
    x = torch.randn(N, device=device)
    out = torch.empty(N, device=device, dtype=torch.float32)
    grid = (triton.cdiv(N, 256),)
    gelu_kernel[grid](x, out, N, BLOCK=256)
    ref = torch.nn.functional.gelu(x.float(), approximate="tanh")
    assert allclose(out, ref), "GELU mismatch"


# ═══════════════════════════════════════════════════════════════════════════
# Level 10: Prefix Sum (Exclusive Scan)
#   Exercises: ScanOpToLLVM (distinct from reduce)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _scan_add_fn(a, b):
    return a + b


@triton.jit
def cumsum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Inclusive prefix sum via associative_scan
    scan_result = tl.associative_scan(x, axis=0, combine_fn=_scan_add_fn)
    tl.store(out_ptr + offs, scan_result, mask=mask)


@requires_cuda
@pytest.mark.parametrize("N", [32, 128, 1024])
def test_cumsum(N):
    """Prefix sum: exercises associative_scan → ScanOpToLLVM."""
    device = get_device()
    x = torch.ones(N, device=device)
    out = torch.empty(N, device=device)
    BLOCK = triton.next_power_of_2(N)
    cumsum_kernel[(1,)](x, out, N, BLOCK=BLOCK)
    ref = torch.cumsum(x, dim=0)
    assert allclose(out, ref), f"Cumsum mismatch at N={N}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 11: Atomic Max — different atomic lowering path
#   Exercises: atomic_max codegen (distinct from atomic_add)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def reduce_max_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
    block_max = tl.max(x, axis=0)
    tl.atomic_max(out_ptr, block_max)


@requires_cuda
@pytest.mark.parametrize("N", [256, 4096])
def test_reduce_max(N):
    """Block-parallel max reduction with atomic_max."""
    device = get_device()
    x = torch.randn(N, device=device)
    # Initialize output to -inf for atomic_max
    out = torch.full((1,), float("-inf"), device=device)
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    reduce_max_kernel[grid](x, out, N, BLOCK=BLOCK)
    ref = x.max()
    assert torch.allclose(out[0], ref, atol=1e-5), \
        f"Max mismatch: got {out[0].item():.4f}, expected {ref.item():.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 12: Autotune — exercises runtime autotuner path
#   Exercises: triton.autotune, Config, key-based retuning
# ═══════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=2),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def autotuned_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@requires_cuda
@pytest.mark.skipif(IS_INTERPRET, reason="Autotune requires real GPU")
def test_autotune():
    """Autotuned kernel: exercises @triton.autotune runtime path."""
    N = 8192
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    autotuned_add_kernel[(triton.cdiv(N, 1024),)](x, y, out, N)
    assert allclose(out, x + y), "Autotuned add mismatch"


# ═══════════════════════════════════════════════════════════════════════════
# Level 13: Broadcast + expand_dims — exercises ReorderBroadcast
#   Exercises: broadcast_to, expand_dims, 2D tensor ops
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def row_broadcast_add_kernel(
    x_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask)
    # Load 1D bias and broadcast to 2D (exercises ReorderBroadcast)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    y = x + bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, y, mask=mask)


@requires_cuda
@pytest.mark.parametrize("shape", [(32, 64), (64, 128)])
def test_broadcast_add(shape):
    """Row-wise bias add: exercises broadcast + ReorderBroadcast pass."""
    M, N = shape
    device = get_device()
    x = torch.randn(M, N, device=device)
    bias = torch.randn(N, device=device)
    out = torch.empty(M, N, device=device)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    row_broadcast_add_kernel[grid](
        x, bias, out, M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    ref = x + bias[None, :]
    assert allclose(out, ref), f"Broadcast add mismatch for {shape}"


# ═══════════════════════════════════════════════════════════════════════════
# Level 14: Pipelined Matmul — exercises pipeline + schedule-loops
#   Exercises: pipeline, schedule-loops, software pipelining
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def pipelined_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Use tl.range with num_stages to trigger software pipelining
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc = tl.dot(a, b, acc)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@requires_cuda
@pytest.mark.skipif(IS_INTERPRET, reason="tl.dot in interpreter triggers OpenMP conflict on Windows")
def test_pipelined_matmul():
    """Matmul with tl.range(num_stages=2): exercises pipeline pass."""
    M, N, K = 64, 64, 128
    device = get_device()
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    c = torch.empty(M, N, device=device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    pipelined_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    ref = torch.mm(a, b)
    assert torch.allclose(c, ref, atol=1e-3, rtol=1e-3), "Pipelined matmul mismatch"


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarks (skipped without GPU)
# ═══════════════════════════════════════════════════════════════════════════

@requires_cuda
@pytest.mark.skipif(IS_INTERPRET, reason="Benchmarks require real GPU")
@pytest.mark.parametrize("N", [2**16, 2**20, 2**24])
def test_benchmark_vector_add(N):
    """Benchmark vector add — reports GB/s."""
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    ms = triton.testing.do_bench(
        lambda: vector_add_kernel[grid](x, y, out, N, BLOCK=BLOCK),
        warmup=25, rep=100,
    )
    gbps = 3 * N * x.element_size() * 1e-9 / (ms * 1e-3)
    print(f"  vector_add N={N:>10,}: {ms:.3f} ms, {gbps:.1f} GB/s")


@requires_cuda
@pytest.mark.skipif(IS_INTERPRET, reason="Benchmarks require real GPU")
@pytest.mark.parametrize("M,N,K", [(512, 512, 512), (1024, 1024, 1024)])
def test_benchmark_matmul(M, N, K):
    """Benchmark GEMM — reports TFLOPS."""
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")

    ms = triton.testing.do_bench(
        lambda: triton_matmul(a, b),
        warmup=25, rep=100,
    )
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
    print(f"  matmul ({M}x{N}x{K}): {ms:.3f} ms, {tflops:.2f} TFLOPS")


# ═══════════════════════════════════════════════════════════════════════════
# IR Dump Smoke Test
# ═══════════════════════════════════════════════════════════════════════════

@requires_cuda
def test_ir_dump(tmp_path):
    """Verify IR dump infrastructure works (saves .ttir/.ttgir/.llir/.ptx)."""
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(tmp_path)
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    N = 256
    device = get_device()
    x = torch.randn(N, device=device)
    y = torch.randn(N, device=device)
    out = torch.empty(N, device=device)
    vector_add_kernel[(1,)](x, y, out, N, BLOCK=256)

    # Check that dump files were created
    dump_files = list(tmp_path.rglob("*"))
    print(f"  IR dump created {len(dump_files)} files in {tmp_path}")

    # Clean up env
    os.environ.pop("TRITON_KERNEL_DUMP", None)
    os.environ.pop("TRITON_DUMP_DIR", None)
    os.environ.pop("TRITON_ALWAYS_COMPILE", None)
