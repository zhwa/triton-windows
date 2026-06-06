"""Vulkan SPIR-V dispatch test suite (11 kernels incl. multi-block + parallel reduction)."""
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

def run(spv, md, bufs, pc, workgroups=1):
    vc = vulkan.runtime.VulkanCompute()
    vc.load_shader(spv, md["name"]); vc.set_workgroups(workgroups)
    ids = []
    for i, (d, s) in enumerate(bufs):
        bid = vc.create_buffer(i, s); vc.write_buffer(bid, d); ids.append(bid)
    vc.set_push_constants(np.array(pc, dtype=np.int32))
    vc.dispatch()
    return vc, ids

def read(vc, bid, n):
    o = np.zeros(n, np.float32); vc.read_buffer(bid, o); return o

PC = [N, 1, 1, 1]  # push constants: N, num_programs(1,1,1) — pid from WorkgroupId
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
vc, ids = run(*comp("test_matmul_simple"), [(A,256*4),(B,256*4),(np.zeros(256,np.float32),256*4)], [16,16,16,1,1,1])
results.append(("matmul_16x16", np.max(np.abs(read(vc,ids[2],256).reshape(16,16) - A.reshape(16,16)@B.reshape(16,16))), 1e-4))

# cooperative matrix matmul (fp16, 16x16)
A16 = np.random.randn(256).astype(np.float16)
B16 = np.random.randn(256).astype(np.float16)
spv, md = comp("test_matmul_coop")
vc = vulkan.runtime.VulkanCompute()
vc.load_shader(spv, md["name"]); vc.set_workgroups(1)
ids = []
for i, (d, s) in enumerate([(A16, 256*2), (B16, 256*2), (np.zeros(256, np.float16), 256*2)]):
    bid = vc.create_buffer(i, s); vc.write_buffer(bid, d); ids.append(bid)
vc.set_push_constants(np.array([16, 16, 16, 1, 1, 1], dtype=np.int32))
vc.dispatch()
C16 = np.zeros(256, np.float16); vc.read_buffer(ids[2], C16)
ref = (A16.reshape(16,16).astype(np.float32) @ B16.reshape(16,16).astype(np.float32)).astype(np.float16)
results.append(("matmul_coop_f16", np.max(np.abs(C16.reshape(16,16) - ref)), 5e-2))

# transpose
x = np.random.randn(256).astype(np.float32)
vc, ids = run(*comp("test_transpose"), [(x,256*4),(np.zeros(256,np.float32),256*4)], [16,16,1,1,1])
results.append(("transpose", np.max(np.abs(read(vc,ids[1],256) - x.reshape(16,16).T.flatten())), 1e-6))

# multi-block vector_add: 1024 elements = 4 workgroups of BLOCK_SIZE=256
N_MB = 1024; NB = 4
x_mb, y_mb = np.random.randn(N_MB).astype(np.float32), np.random.randn(N_MB).astype(np.float32)
vc, ids = run(*comp("test_vector_add"), [(x_mb,N_MB*4),(y_mb,N_MB*4),(np.zeros(N_MB,np.float32),N_MB*4)], [N_MB, NB, 1, 1], workgroups=NB)
results.append(("vadd_multiblock", np.max(np.abs(read(vc,ids[2],N_MB) - (x_mb+y_mb))), 1e-6))

# large-N vector_add: 65536 elements = 256 workgroups, exercises device-local staging
N_LG = 65536; NB_LG = N_LG // 256
x_lg, y_lg = np.random.randn(N_LG).astype(np.float32), np.random.randn(N_LG).astype(np.float32)
vc, ids = run(*comp("test_vector_add"), [(x_lg,N_LG*4),(y_lg,N_LG*4),(np.zeros(N_LG,np.float32),N_LG*4)], [N_LG, NB_LG, 1, 1], workgroups=NB_LG)
results.append(("vadd_65k", np.max(np.abs(read(vc,ids[2],N_LG) - (x_lg+y_lg))), 1e-6))

print("Vulkan SPIR-V dispatch (C+5: coop matrix) — " + vulkan.runtime.VulkanCompute().device_name())
print(f"{'Kernel':<20s} {'Error':>12s} {'Tol':>10s} {'Status':>8s}")
print("-" * 54)
passed = 0
for name, err, tol in results:
    ok = err < tol; passed += ok
    print(f"{name:<20s} {err:>12.2e} {tol:>10.0e} {'PASS' if ok else 'FAIL':>8s}")
print("-" * 54)
print(f"Result: {passed}/{len(results)} PASS")
sys.exit(0 if passed == len(results) else 1)
