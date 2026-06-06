"""
Serial OpenCL C test suite for the Triton Vulkan/SPIR-V backend.

Tests 10 kernel types end-to-end:
  TTIR → make_ttir → make_linalg → make_memref → make_opencl → GPU execution

Requires: pyopencl, numpy, TRITON_BACKENDS_IN_TREE=1
"""

import os
import sys
import numpy as np

# ── Environment ──────────────────────────────────────────────────────────────
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pyopencl as cl
from triton._C.libtriton import ir, passes, vulkan
from triton.backends.vulkan.compiler import VulkanBackend, GPUTarget

# ── OpenCL context ──────────────────────────────────────────────────────────
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
RO = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WO = cl.mem_flags.WRITE_ONLY

TEST_DIR = os.path.join(os.path.dirname(__file__), ".")
N = 256  # vector length for all tests


def compile_ttir(ttir_path: str):
    """Compile a .ttir file through the full pipeline, return (OpenCL src, metadata)."""
    c = ir.context()
    ir.load_dialects(c)
    vulkan.load_dialects(c)
    m = ir.parse_mlir_module(ttir_path, c)
    m.context = c
    backend = VulkanBackend(GPUTarget("vulkan", 0, 32))
    options = backend.parse_options({})
    metadata = {}
    m = backend.make_ttir(m, metadata, options)
    m = backend.make_linalg(m, metadata, options)
    m = backend.make_memref(m, metadata, options)
    return backend.make_opencl(m, metadata, options), metadata


def run_kernel(src, metadata, args, global_size=(1,), local_size=(1,)):
    """Build and run an OpenCL kernel."""
    prog = cl.Program(ctx, src).build()
    kernel = getattr(prog, metadata["name"])
    for i, arg in enumerate(args):
        kernel.set_arg(i, arg)
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    queue.finish()


def run_kernel_multiblock(src, metadata, base_args, n_blocks):
    """Run a kernel with multiple blocks by dispatching n_blocks times.

    base_args should include the kernel's own args but NOT the 6 program info
    args. This function appends num_programs(x,y,z) + program_id(x,y,z) for
    each block, with program_id_x = block_id.
    """
    prog = cl.Program(ctx, src).build()
    kernel = getattr(prog, metadata["name"])
    n_base = len(base_args)
    for bid in range(n_blocks):
        for i, arg in enumerate(base_args):
            kernel.set_arg(i, arg)
        # num_programs: (n_blocks, 1, 1)
        kernel.set_arg(n_base + 0, np.int32(n_blocks))
        kernel.set_arg(n_base + 1, np.int32(1))
        kernel.set_arg(n_base + 2, np.int32(1))
        # program_id: (bid, 0, 0)
        kernel.set_arg(n_base + 3, np.int32(bid))
        kernel.set_arg(n_base + 4, np.int32(0))
        kernel.set_arg(n_base + 5, np.int32(0))
        cl.enqueue_nd_range_kernel(queue, kernel, (1,), (1,))
    queue.finish()


def read_buf(buf, shape, dtype=np.float32):
    """Read an OpenCL buffer back to numpy."""
    out = np.empty(shape, dtype=dtype)
    cl.enqueue_copy(queue, out, buf)
    queue.finish()
    return out


# ── Test functions ───────────────────────────────────────────────────────────

def test_vector_add():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_vector_add.ttir"))
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    yb = cl.Buffer(ctx, RO, hostbuf=y)
    ob = cl.Buffer(ctx, WO, N * 4)
    args = [xb, yb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, N)
    return np.max(np.abs(o - (x + y)))


def test_elementwise_mul():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_elementwise_mul.ttir"))
    x = np.random.randn(N).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    yb = cl.Buffer(ctx, RO, hostbuf=y)
    ob = cl.Buffer(ctx, WO, N * 4)
    args = [xb, yb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, N)
    return np.max(np.abs(o - (x * y)))


def test_fma():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_fma.ttir"))
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.random.randn(N).astype(np.float32)
    ab = cl.Buffer(ctx, RO, hostbuf=a)
    bb = cl.Buffer(ctx, RO, hostbuf=b)
    cb = cl.Buffer(ctx, RO, hostbuf=c)
    ob = cl.Buffer(ctx, WO, N * 4)
    args = [ab, bb, cb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, N)
    return np.max(np.abs(o - (a * b + c)))


def test_gelu():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_gelu.ttir"))
    x = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, WO, N * 4)
    args = [xb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, N)
    expected = x / (1 + np.exp(-1.702 * x.astype(np.float64)))
    return np.max(np.abs(o - expected.astype(np.float32)))


def test_swiglu():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_swiglu.ttir"))
    x = np.random.randn(N).astype(np.float32)
    gate = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    gb = cl.Buffer(ctx, RO, hostbuf=gate)
    ob = cl.Buffer(ctx, WO, N * 4)
    args = [xb, gb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, N)
    sigmoid = 1 / (1 + np.exp(-x.astype(np.float64)))
    expected = (x * sigmoid * gate).astype(np.float32)
    return np.max(np.abs(o - expected))


def test_reduce_sum():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_reduce_sum.ttir"))
    x = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, WO, 4)
    args = [xb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, 1)
    return abs(o[0] - np.sum(x))


def test_reduce_max():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_reduce_max.ttir"))
    x = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, WO, 4)
    args = [xb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, 1)
    return abs(o[0] - np.max(x))


def test_softmax():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_softmax.ttir"))
    x = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, WO, N * 4)
    args = [xb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, N)
    xd = x.astype(np.float64)
    exp_x = np.exp(xd - np.max(xd))
    expected = (exp_x / np.sum(exp_x)).astype(np.float32)
    return np.max(np.abs(o - expected))


def test_matmul():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_matmul_simple.ttir"))
    A = np.random.randn(256).astype(np.float32)
    B = np.random.randn(256).astype(np.float32)
    ab = cl.Buffer(ctx, RO, hostbuf=A)
    bb = cl.Buffer(ctx, RO, hostbuf=B)
    ob = cl.Buffer(ctx, WO, 256 * 4)
    args = [ab, bb, ob, np.int32(16), np.int32(16), np.int32(16)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, 256)
    expected = A.reshape(16, 16) @ B.reshape(16, 16)
    return np.max(np.abs(o.reshape(16, 16) - expected))


def test_broadcast_add():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_broadcast_add.ttir"))
    x = np.random.randn(32).astype(np.float32)
    bias = np.random.randn(32).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    bb = cl.Buffer(ctx, RO, hostbuf=bias)
    ob = cl.Buffer(ctx, WO, 32 * 4)
    args = [xb, bb, ob, np.int32(1), np.int32(32)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, 32)
    return np.max(np.abs(o - (x + bias)))


def test_transpose():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_transpose.ttir"))
    x = np.random.randn(256).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, WO, 256 * 4)
    args = [xb, ob, np.int32(16), np.int32(16)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, 256)
    expected = x.reshape(16, 16).T.flatten()
    return np.max(np.abs(o - expected))


def test_vector_add_multiblock():
    """Multi-block test: 1024 elements = 4 blocks of BLOCK_SIZE=256."""
    total = 1024
    n_blocks = total // 256
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_vector_add.ttir"))
    x = np.random.randn(total).astype(np.float32)
    y = np.random.randn(total).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    yb = cl.Buffer(ctx, RO, hostbuf=y)
    ob = cl.Buffer(ctx, WO, total * 4)
    base_args = [xb, yb, ob, np.int32(total)]
    run_kernel_multiblock(src, md, base_args, n_blocks)
    o = read_buf(ob, total)
    return np.max(np.abs(o - (x + y)))


def test_broadcast_add_2d():
    """True 2D broadcast: x[4,8] + bias[8] → out[4,8]."""
    M, N = 4, 8
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_broadcast_add_2d.ttir"))
    x = np.random.randn(M * N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    bb = cl.Buffer(ctx, RO, hostbuf=bias)
    ob = cl.Buffer(ctx, WO, M * N * 4)
    args = [xb, bb, ob, np.int32(M), np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, M * N)
    expected = x.reshape(M, N) + bias
    return np.max(np.abs(o.reshape(M, N) - expected))


def test_atomic_add():
    src, md = compile_ttir(os.path.join(TEST_DIR, "test_atomic_add.ttir"))
    x = np.random.randn(N).astype(np.float32)
    out_init = np.zeros(N, dtype=np.float32)
    RW = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    xb = cl.Buffer(ctx, RO, hostbuf=x)
    ob = cl.Buffer(ctx, RW, hostbuf=out_init)
    args = [xb, ob, np.int32(N)] + [np.int32(0)] * 6
    run_kernel(src, md, args)
    o = read_buf(ob, N)
    # atomic_rmw fadd with splat ptr (no addptr offsets):
    # In single-threaded sequential execution, this does out[i] += x[i]
    # for each element (the splat ptr is expanded to a ranked memref<256>
    # by the type converter, so the RMW loop iterates per-element).
    expected = out_init + x
    return np.max(np.abs(o - expected))


# ── Main ─────────────────────────────────────────────────────────────────────

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
    ("broadcast_add",    test_broadcast_add,     1e-6),
    ("transpose_16x16",  test_transpose,         1e-6),
    ("atomic_add",       test_atomic_add,        1e-3),
    ("vector_add_4blk",  test_vector_add_multiblock, 1e-6),
    ("broadcast_2d",     test_broadcast_add_2d,  1e-5),
]


def main():
    print(f"Vulkan backend serial OpenCL tests — {device.name}")
    print(f"{'Kernel':<22} {'Error':>12} {'Tol':>10} {'Status':>8}")
    print("-" * 56)

    passed = 0
    failed = 0
    for name, fn, tol in TESTS:
        try:
            err = fn()
            ok = err < tol
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"{name:<22} {err:>12.2e} {tol:>10.0e} {status:>8}")
        except Exception as e:
            failed += 1
            print(f"{name:<22} {'':>12} {'':>10} {'ERROR':>8}  {str(e)[:60]}")

    print("-" * 56)
    print(f"Result: {passed}/{passed + failed} PASS")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
