import os
import sys
import lit.formats

config.name = "TritonVulkan"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir", ".ttir"]

# Find triton-opt from the build directory
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

ext = ".exe" if sys.platform == "win32" else ""
triton_opt = os.path.join(src_dir, "python", "triton", "_C", f"triton-opt{ext}")

config.substitutions.append(("%triton-opt", triton_opt))
