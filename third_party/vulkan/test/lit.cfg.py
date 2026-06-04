import os
import lit.formats

config.name = "TritonVulkan"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir", ".ttir"]

# Find triton-opt from the build directory
build_dir = os.environ.get("TRITON_BUILD_DIR", "")
if not build_dir:
    # Guess common path
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    build_dir = os.path.join(src_dir, "build", "cmake.win-amd64-cpython-3.14")

triton_opt = os.path.join(src_dir, "python", "triton", "_C", "triton-opt.exe")
if not os.path.exists(triton_opt):
    triton_opt = os.path.join(src_dir, "python", "triton", "_C", "triton-opt")

config.substitutions.append(("%triton-opt", triton_opt))
