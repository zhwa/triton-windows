"""
Phase 3 parallel emitter GPU test suite.

Tests the parallel OpenCL path:
  TTIR → make_ttir → make_linalg → make_memref_bufonly → make_opencl_parallel → GPU

Verifies correctness for all kernel types against numpy reference.
"""

import os
import sys
import numpy as np

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pyopencl as cl
from triton._C.libtriton import ir, passes, vulkan
from triton.backends.vulkan.compiler import VulkanBackend, GPUTarget

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
RO = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WO = cl.mem_flags.WRITE_ONLY
RW = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR

TEST_DIR = os.path.join(os.path.dirname(__file__), ".")
N = 256


def compile_parallel(ttir_path):
    c = ir.context(); ir.load_dialects(c); vulkan.load_dialects(c)
    m = ir.parse_mlir_module(ttir_path, c); m.context = c
    b = VulkanBackend(GPUTarget("vulkan", 0, 32))
    o = b.parse_options({}); md = {}
    m = b.make_ttir(m, md, o)
    m = b.make_linalg(m, md, o)
    m = b.make_memref_bufonly(m, md, o)
    return b.make_opencl_parallel(m, md, o), md


def run_par(src, md, args):
    prog = cl.Program(ctx, src).build()
    k = getattr(prog, md["name"])
    for i, a in enumerate(args):
        k.set_arg(i, a)
    bs = md["block_size"]
    cl.enqueue_nd_range_kernel(queue, k, (bs,), (bs,))
    queue.finish()


def read(buf, n=N):
    r = np.empty(n, np.float32)
    cl.enqueue_copy(queue, r, buf)
    queue.finish()
    return r


# ── Tests ────────────────────────────────────────────────────────────────────

def test_vector_add():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_vector_add.ttir"))
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, WO, N * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), cl.Buffer(ctx, RO, hostbuf=y),
                    ob, np.int32(N)] + [np.int32(0)] * 6)
    return np.max(np.abs(read(ob) - (x + y)))


def test_elementwise_mul():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_elementwise_mul.ttir"))
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, WO, N * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), cl.Buffer(ctx, RO, hostbuf=y),
                    ob, np.int32(N)] + [np.int32(0)] * 6)
    return np.max(np.abs(read(ob) - (x * y)))


def test_fma():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_fma.ttir"))
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, WO, N * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=a), cl.Buffer(ctx, RO, hostbuf=b),
                    cl.Buffer(ctx, RO, hostbuf=c), ob, np.int32(N)] + [np.int32(0)] * 6)
    return np.max(np.abs(read(ob) - (a * b + c)))


def test_gelu():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_gelu.ttir"))
    x = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, WO, N * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), ob, np.int32(N)] + [np.int32(0)] * 6)
    expected = x / (1 + np.exp(-1.702 * x.astype(np.float64)))
    return np.max(np.abs(read(ob) - expected.astype(np.float32)))


def test_swiglu():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_swiglu.ttir"))
    x = np.random.randn(N).astype(np.float32)
    g = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, WO, N * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), cl.Buffer(ctx, RO, hostbuf=g),
                    ob, np.int32(N)] + [np.int32(0)] * 6)
    sig = 1 / (1 + np.exp(-x.astype(np.float64)))
    return np.max(np.abs(read(ob) - (x * sig * g).astype(np.float32)))


def test_reduce_sum():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_reduce_sum.ttir"))
    x = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, RW, hostbuf=np.zeros(1, np.float32))
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), ob, np.int32(N)] + [np.int32(0)] * 6)
    o = np.empty(1, np.float32); cl.enqueue_copy(queue, o, ob); queue.finish()
    return abs(o[0] - np.sum(x))


def test_reduce_max():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_reduce_max.ttir"))
    x = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, RW, hostbuf=np.zeros(1, np.float32))
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), ob, np.int32(N)] + [np.int32(0)] * 6)
    o = np.empty(1, np.float32); cl.enqueue_copy(queue, o, ob); queue.finish()
    return abs(o[0] - np.max(x))


def test_softmax():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_softmax.ttir"))
    x = np.random.randn(N).astype(np.float32)
    ob = cl.Buffer(ctx, WO, N * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), ob, np.int32(N)] + [np.int32(0)] * 6)
    xd = x.astype(np.float64); ex = np.exp(xd - np.max(xd))
    expected = (ex / np.sum(ex)).astype(np.float32)
    return np.max(np.abs(read(ob) - expected))


def test_matmul():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_matmul_simple.ttir"))
    A = np.random.randn(256).astype(np.float32)
    B = np.random.randn(256).astype(np.float32)
    ob = cl.Buffer(ctx, WO, 256 * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=A), cl.Buffer(ctx, RO, hostbuf=B),
                    ob, np.int32(16), np.int32(16), np.int32(16)] + [np.int32(0)] * 6)
    return np.max(np.abs(read(ob).reshape(16, 16) - A.reshape(16, 16) @ B.reshape(16, 16)))


def test_transpose():
    s, md = compile_parallel(os.path.join(TEST_DIR, "test_transpose.ttir"))
    x = np.random.randn(256).astype(np.float32)
    ob = cl.Buffer(ctx, WO, 256 * 4)
    run_par(s, md, [cl.Buffer(ctx, RO, hostbuf=x), ob,
                    np.int32(16), np.int32(16)] + [np.int32(0)] * 6)
    return np.max(np.abs(read(ob) - x.reshape(16, 16).T.flatten()))


TESTS = [
    ("vector_add",       test_vector_add,       1e-6),
    ("elementwise_mul",  test_elementwise_mul,   1e-6),
    ("fma",              test_fma,               1e-5),
    ("gelu",             test_gelu,              1e-5),
    ("swiglu",           test_swiglu,            1e-5),
    ("reduce_sum",       test_reduce_sum,        1e-3),
    ("reduce_max",       test_reduce_max,        1e-6),
    ("softmax",          test_softmax,           1e-5),
    ("matmul_16x16",     test_matmul,            1e-4),
    ("transpose_16x16",  test_transpose,         1e-6),
]


def main():
    print(f"Phase 3 parallel GPU tests — {device.name}")
    print(f"{'Kernel':<22} {'Error':>12} {'Tol':>10} {'Status':>8}")
    print("-" * 56)

    passed = failed = 0
    for name, fn, tol in TESTS:
        try:
            err = fn()
            ok = err < tol
            status = "PASS" if ok else "FAIL"
            passed += ok; failed += (not ok)
            print(f"{name:<22} {err:>12.2e} {tol:>10.0e} {status:>8}")
        except Exception as e:
            failed += 1
            print(f"{name:<22} {'':>12} {'':>10} {'ERROR':>8}  {str(e)[:60]}")

    print("-" * 56)
    print(f"Result: {passed}/{passed + failed} PASS")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
