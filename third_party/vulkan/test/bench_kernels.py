"""
Phase 2 performance baseline: Vulkan (OpenCL) vs CUDA.

Benchmarks key kernels at realistic sizes to establish Route B performance.
Requires: pyopencl, numpy, torch (for CUDA baseline)
"""

import os, sys, time
import numpy as np

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pyopencl as cl
from triton._C.libtriton import ir, passes, vulkan
from triton.backends.vulkan.compiler import VulkanBackend, GPUTarget

# ── OpenCL context ───────────────────────────────────────────────────────────
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
RO = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WO = cl.mem_flags.WRITE_ONLY

TEST_DIR = os.path.join(os.path.dirname(__file__), ".")


def compile_ttir(ttir_path):
    c = ir.context(); ir.load_dialects(c); vulkan.load_dialects(c)
    m = ir.parse_mlir_module(ttir_path, c); m.context = c
    b = VulkanBackend(GPUTarget("vulkan", 0, 32))
    o = b.parse_options({}); md = {}
    m = b.make_ttir(m, md, o); m = b.make_linalg(m, md, o)
    m = b.make_memref(m, md, o)
    return b.make_opencl(m, md, o), md


def bench_opencl(name, setup_fn, n_iter=100, warmup=10):
    """Benchmark an OpenCL kernel. setup_fn returns (kernel, queue, event_fn)."""
    kernel, queue, run_fn = setup_fn()
    # Warmup
    for _ in range(warmup):
        run_fn()
    queue.finish()
    # Timed
    start = time.perf_counter()
    for _ in range(n_iter):
        run_fn()
    queue.finish()
    elapsed = (time.perf_counter() - start) / n_iter
    return elapsed


def bench_cuda(name, fn, n_iter=100, warmup=10):
    """Benchmark a CUDA operation via torch."""
    import torch
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iter
    return elapsed


# ── Benchmarks ───────────────────────────────────────────────────────────────

def bench_vector_add(N=65536):
    """vector_add: out = x + y, N elements."""
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_vector_add.ttir"))
    prog = cl.Program(ctx, src).build()
    kern = getattr(prog, md["name"])

    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    yb = cl.Buffer(ctx, RO, hostbuf=y)
    ob = cl.Buffer(ctx, WO, N * 4)

    n_blocks = (N + 255) // 256
    queue = cl.CommandQueue(ctx)

    def setup():
        def run():
            for bid in range(n_blocks):
                kern.set_arg(0, xb); kern.set_arg(1, yb); kern.set_arg(2, ob)
                kern.set_arg(3, np.int32(N))
                kern.set_arg(4, np.int32(n_blocks)); kern.set_arg(5, np.int32(1)); kern.set_arg(6, np.int32(1))
                kern.set_arg(7, np.int32(bid)); kern.set_arg(8, np.int32(0)); kern.set_arg(9, np.int32(0))
                cl.enqueue_nd_range_kernel(queue, kern, (1,), (1,))
        return kern, queue, run

    ocl_time = bench_opencl("vector_add", setup, n_iter=20, warmup=5)

    # Verify correctness
    queue.finish()
    o = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(queue, o, ob); queue.finish()
    err = np.max(np.abs(o - (x + y)))

    cuda_time = None
    try:
        import torch
        tx = torch.from_numpy(x).cuda()
        ty = torch.from_numpy(y).cuda()
        cuda_time = bench_cuda("vector_add", lambda: tx + ty, n_iter=100)
    except ImportError:
        pass

    return N, ocl_time, cuda_time, err


def bench_reduce_sum(N=65536):
    """reduce_sum: out = sum(x), N elements (single block = 256 elements)."""
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_reduce_sum.ttir"))
    prog = cl.Program(ctx, src).build()
    kern = getattr(prog, md["name"])

    # Our reduce kernel handles 256 elements per block
    x = np.random.randn(256).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, WO, 4)

    queue = cl.CommandQueue(ctx)

    def setup():
        def run():
            kern.set_arg(0, xb); kern.set_arg(1, ob); kern.set_arg(2, np.int32(256))
            for i in range(3, 9): kern.set_arg(i, np.int32(0))
            cl.enqueue_nd_range_kernel(queue, kern, (1,), (1,))
        return kern, queue, run

    ocl_time = bench_opencl("reduce_sum", setup, n_iter=100, warmup=10)

    cuda_time = None
    try:
        import torch
        tx = torch.from_numpy(x).cuda()
        cuda_time = bench_cuda("reduce_sum", lambda: tx.sum(), n_iter=100)
    except ImportError:
        pass

    return 256, ocl_time, cuda_time, 0.0


def bench_softmax(N=256):
    """softmax: out = softmax(x), 256 elements."""
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_softmax.ttir"))
    prog = cl.Program(ctx, src).build()
    kern = getattr(prog, md["name"])

    x = np.random.randn(256).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, WO, 256 * 4)

    queue = cl.CommandQueue(ctx)

    def setup():
        def run():
            kern.set_arg(0, xb); kern.set_arg(1, ob); kern.set_arg(2, np.int32(256))
            for i in range(3, 9): kern.set_arg(i, np.int32(0))
            cl.enqueue_nd_range_kernel(queue, kern, (1,), (1,))
        return kern, queue, run

    ocl_time = bench_opencl("softmax", setup, n_iter=100, warmup=10)

    cuda_time = None
    try:
        import torch
        tx = torch.from_numpy(x).cuda()
        cuda_time = bench_cuda("softmax", lambda: torch.softmax(tx, 0), n_iter=100)
    except ImportError:
        pass

    return 256, ocl_time, cuda_time, 0.0


def bench_matmul(M=16):
    """matmul: C = A @ B, 16x16."""
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_matmul_simple.ttir"))
    prog = cl.Program(ctx, src).build()
    kern = getattr(prog, md["name"])

    A = np.random.randn(256).astype(np.float32)
    B = np.random.randn(256).astype(np.float32)
    ab = cl.Buffer(ctx, RO, hostbuf=A)
    bb = cl.Buffer(ctx, RO, hostbuf=B)
    ob = cl.Buffer(ctx, WO, 256 * 4)

    queue = cl.CommandQueue(ctx)

    def setup():
        def run():
            kern.set_arg(0, ab); kern.set_arg(1, bb); kern.set_arg(2, ob)
            kern.set_arg(3, np.int32(16)); kern.set_arg(4, np.int32(16)); kern.set_arg(5, np.int32(16))
            for i in range(6, 12): kern.set_arg(i, np.int32(0))
            cl.enqueue_nd_range_kernel(queue, kern, (1,), (1,))
        return kern, queue, run

    ocl_time = bench_opencl("matmul", setup, n_iter=100, warmup=10)

    cuda_time = None
    try:
        import torch
        tA = torch.from_numpy(A.reshape(16, 16)).cuda()
        tB = torch.from_numpy(B.reshape(16, 16)).cuda()
        cuda_time = bench_cuda("matmul", lambda: tA @ tB, n_iter=100)
    except ImportError:
        pass

    return 16, ocl_time, cuda_time, 0.0


def main():
    print(f"Performance baseline: Vulkan (OpenCL) vs CUDA — {device.name}")
    print(f"{'Kernel':<18} {'N':>8} {'OpenCL µs':>12} {'CUDA µs':>12} {'Ratio':>8}")
    print("-" * 62)

    benchmarks = [
        ("vector_add",  bench_vector_add),
        ("reduce_sum",  bench_reduce_sum),
        ("softmax",     bench_softmax),
        ("matmul_16x16", bench_matmul),
    ]

    for name, fn in benchmarks:
        try:
            n, ocl, cuda, err = fn()
            ocl_us = ocl * 1e6
            if cuda is not None:
                cuda_us = cuda * 1e6
                ratio = f"{ocl_us / cuda_us:.1f}x"
            else:
                cuda_us = float('nan')
                ratio = "N/A"
            print(f"{name:<18} {n:>8} {ocl_us:>12.1f} {cuda_us:>12.1f} {ratio:>8}")
        except Exception as e:
            print(f"{name:<18} {'':>8} {'ERROR':>12}  {str(e)[:40]}")

    print()
    print("Note: OpenCL kernel dispatches single-threaded blocks sequentially.")
    print("      CUDA uses native parallel execution. Ratio = OpenCL/CUDA time.")


if __name__ == "__main__":
    sys.exit(main() or 0)
