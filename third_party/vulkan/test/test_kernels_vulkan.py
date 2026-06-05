"""Phase 3.5 Vulkan SPIR-V dispatch test suite."""
import os, sys, numpy as np
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
from triton._C.libtriton import ir, passes, vulkan
from triton.backends.vulkan.compiler import VulkanBackend, GPUTarget

TEST_DIR = os.path.join(os.path.dirname(__file__), ".")
N = 256

def comp(name):
    c = ir.context(); ir.load_dialects(c); vulkan.load_dialects(c)
    m = ir.parse_mlir_module(os.path.join(TEST_DIR, name + ".ttir"), c)
    m.context = c
    b = VulkanBackend(GPUTarget("vulkan", 0, 32))
    o = b.parse_options({}); md = {}
    m = b.make_ttir(m, md, o); m = b.make_linalg(m, md, o)
    m = b.make_memref(m, md, o); m = b.make_spirv(m, md, o)
    return b.make_spv(m, md, o), md

def run(spv, md, bufs, pc):
    vc = vulkan.runtime.VulkanCompute()
    vc.load_shader(spv, md["name"]); vc.set_workgroups(1)
    ids = []
    for i, (d, s) in enumerate(bufs):
        bid = vc.create_buffer(i, s); vc.write_buffer(bid, d); ids.append(bid)
    vc.set_push_constants(np.array(pc, dtype=np.int32))
    vc.dispatch()
    return vc, ids

def read(vc, bid, n):
    o = np.zeros(n, np.float32); vc.read_buffer(bid, o); return o

PC = [N, 1, 1, 1, 0, 0, 0]
results = []

# vector_add
x, y = np.random.randn(N).astype(np.float32), np.random.randn(N).astype(np.float32)
vc, ids = run(*comp("test_vector_add"), [(x,N*4),(y,N*4),(np.zeros(N,np.float32),N*4)], PC)
results.append(("vector_add", np.max(np.abs(read(vc,ids[2],N) - (x+y))), 1e-6))

# fma
a, b, c = [np.random.randn(N).astype(np.float32) for _ in range(3)]
vc, ids = run(*comp("test_fma"), [(a,N*4),(b,N*4),(c,N*4),(np.zeros(N,np.float32),N*4)], PC)
results.append(("fma", np.max(np.abs(read(vc,ids[3],N) - (a*b+c))), 1e-5))

# gelu
x = np.random.randn(N).astype(np.float32)
vc, ids = run(*comp("test_gelu"), [(x,N*4),(np.zeros(N,np.float32),N*4)], PC)
exp = x / (1 + np.exp(-1.702 * x.astype(np.float64)))
results.append(("gelu", np.max(np.abs(read(vc,ids[1],N) - exp.astype(np.float32))), 1e-5))

# swiglu
x, g = np.random.randn(N).astype(np.float32), np.random.randn(N).astype(np.float32)
vc, ids = run(*comp("test_swiglu"), [(x,N*4),(g,N*4),(np.zeros(N,np.float32),N*4)], PC)
sig = 1 / (1 + np.exp(-x.astype(np.float64)))
results.append(("swiglu", np.max(np.abs(read(vc,ids[2],N) - (x*sig*g).astype(np.float32))), 1e-5))

# reduce_sum
x = np.random.randn(N).astype(np.float32)
vc, ids = run(*comp("test_reduce_sum"), [(x,N*4),(np.zeros(1,np.float32),4)], PC)
results.append(("reduce_sum", abs(read(vc,ids[1],1)[0] - np.sum(x)), 1e-3))

# reduce_max
x = np.random.randn(N).astype(np.float32)
vc, ids = run(*comp("test_reduce_max"), [(x,N*4),(np.zeros(1,np.float32),4)], PC)
results.append(("reduce_max", abs(read(vc,ids[1],1)[0] - np.max(x)), 1e-6))

# softmax
x = np.random.randn(N).astype(np.float32)
vc, ids = run(*comp("test_softmax"), [(x,N*4),(np.zeros(N,np.float32),N*4)], PC)
xd = x.astype(np.float64); ex = np.exp(xd - np.max(xd))
results.append(("softmax", np.max(np.abs(read(vc,ids[1],N) - (ex/np.sum(ex)).astype(np.float32))), 1e-5))

# matmul
A, B = np.random.randn(256).astype(np.float32), np.random.randn(256).astype(np.float32)
vc, ids = run(*comp("test_matmul_simple"), [(A,256*4),(B,256*4),(np.zeros(256,np.float32),256*4)], [16,16,16,1,1,1,0,0,0])
results.append(("matmul_16x16", np.max(np.abs(read(vc,ids[2],256).reshape(16,16) - A.reshape(16,16)@B.reshape(16,16))), 1e-4))

# transpose
x = np.random.randn(256).astype(np.float32)
vc, ids = run(*comp("test_transpose"), [(x,256*4),(np.zeros(256,np.float32),256*4)], [16,16,1,1,1,0,0,0])
results.append(("transpose", np.max(np.abs(read(vc,ids[1],256) - x.reshape(16,16).T.flatten())), 1e-6))

print("Phase 3.5 Vulkan SPIR-V dispatch — " + vulkan.runtime.VulkanCompute().device_name())
print(f"{'Kernel':<20s} {'Error':>12s} {'Tol':>10s} {'Status':>8s}")
print("-" * 54)
passed = 0
for name, err, tol in results:
    ok = err < tol; passed += ok
    print(f"{name:<20s} {err:>12.2e} {tol:>10.0e} {'PASS' if ok else 'FAIL':>8s}")
print("-" * 54)
print(f"Result: {passed}/{len(results)} PASS")
sys.exit(0 if passed == len(results) else 1)
