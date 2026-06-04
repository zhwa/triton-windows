#!/usr/bin/env python3
"""vulkan-opt: Run MLIR SPIR-V conversion pipeline on input IR.

Phase 0 implementation — wraps mlir-opt and mlir-translate from our LLVM build.
In later phases this will be replaced by a native C++ tool with custom passes.

Usage:
    python vulkan-opt.py input.mlir                    # Convert and print SPIR-V MLIR
    python vulkan-opt.py input.mlir -o output.spv      # Serialize to SPIR-V binary
    python vulkan-opt.py input.mlir --roundtrip         # Serialize + deserialize (verify)
    python vulkan-opt.py input.mlir --pipeline arith    # Run arith→spirv pipeline
    python vulkan-opt.py input.mlir --pipeline linalg   # Run linalg→spirv pipeline
"""

import argparse
import os
import subprocess
import sys
import tempfile

# Find LLVM tools relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Script is at third_party/vulkan/tools/vulkan-opt.py → root is 3 levels up
TRITON_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
LLVM_BIN = os.path.join(TRITON_ROOT, "build", "llvm-project", "build", "bin")
_EXE = ".exe" if sys.platform == "win32" else ""
MLIR_OPT = os.path.join(LLVM_BIN, f"mlir-opt{_EXE}")
MLIR_TRANSLATE = os.path.join(LLVM_BIN, f"mlir-translate{_EXE}")

# Pre-defined conversion pipelines
PIPELINES = {
    "arith": [
        "--convert-arith-to-spirv",
        "--convert-func-to-spirv",
        "--canonicalize",
    ],
    "linalg": [
        "--convert-linalg-to-loops",
        "--convert-scf-to-cf",
        "--convert-cf-to-spirv",
        "--convert-arith-to-spirv",
        "--convert-func-to-spirv",
        "--convert-index-to-spirv",
        "--canonicalize",
    ],
    "gpu": [
        "--convert-gpu-to-spirv",
        "--convert-arith-to-spirv",
        "--convert-func-to-spirv",
        "--canonicalize",
    ],
    "memref": [
        "--convert-memref-to-spirv",
        "--convert-arith-to-spirv",
        "--convert-func-to-spirv",
        "--canonicalize",
    ],
    "full": [
        "--convert-linalg-to-loops",
        "--convert-scf-to-cf",
        "--convert-memref-to-spirv",
        "--convert-cf-to-spirv",
        "--convert-math-to-spirv",
        "--convert-arith-to-spirv",
        "--convert-index-to-spirv",
        "--convert-func-to-spirv",
        "--canonicalize",
    ],
    "math": [
        "--convert-math-to-spirv",
        "--convert-arith-to-spirv",
        "--convert-func-to-spirv",
        "--canonicalize",
    ],
}


def run(cmd, check=True):
    """Run command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result


def convert(input_file, passes):
    """Run mlir-opt with given passes, return SPIR-V MLIR text."""
    cmd = [MLIR_OPT] + passes + [input_file]
    result = run(cmd)
    return result.stdout


def _wrap_in_spirv_module(mlir_text):
    """Wrap bare spirv.func ops in a spirv.module if needed.

    mlir-opt --convert-*-to-spirv produces spirv.func ops inside a
    builtin.module. mlir-translate --serialize-spirv needs a top-level
    spirv.module. This inserts the spirv.module wrapper when missing.
    """
    if "spirv.module" in mlir_text:
        return mlir_text  # already wrapped

    import re
    # Strip outer 'module { ... }' wrapper
    inner = re.sub(r"^module\s*\{", "", mlir_text.strip(), count=1)
    if inner.endswith("}"):
        inner = inner[:-1]

    return (
        'spirv.module Logical GLSL450 '
        'requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {\n'
        + inner
        + '}\n'
    )


def serialize(mlir_text, output_file=None):
    """Serialize SPIR-V MLIR to binary .spv file."""
    mlir_text = _wrap_in_spirv_module(mlir_text)

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        tmp = f.name

    try:
        cmd = [MLIR_TRANSLATE, "--no-implicit-module", "--serialize-spirv", tmp]
        if output_file:
            cmd += ["-o", output_file]
            result = run(cmd)
            size = os.path.getsize(output_file)
            print(f"Wrote {size} bytes to {output_file}", file=sys.stderr)
            return None
        else:
            result = run(cmd)
            return result.stdout
    finally:
        os.unlink(tmp)


def roundtrip(mlir_text):
    """Serialize then deserialize SPIR-V to verify correctness."""
    mlir_text = _wrap_in_spirv_module(mlir_text)

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        tmp = f.name

    try:
        cmd = [MLIR_TRANSLATE, "--no-implicit-module", "--test-spirv-roundtrip", tmp]
        result = run(cmd)
        return result.stdout
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(description="Vulkan/SPIR-V optimizer for Triton")
    parser.add_argument("input", help="Input MLIR file")
    parser.add_argument("-o", "--output", help="Output .spv binary file")
    parser.add_argument("--pipeline", choices=list(PIPELINES.keys()),
                        help="Named conversion pipeline to run")
    parser.add_argument("--passes", type=str,
                        help="Comma-separated mlir-opt passes (e.g. convert-arith-to-spirv,convert-func-to-spirv)")
    parser.add_argument("--roundtrip", action="store_true",
                        help="Serialize and deserialize to verify correctness")
    parser.add_argument("--serialize-only", action="store_true",
                        help="Only serialize (skip conversion)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print intermediate IR")

    args = parser.parse_args()

    # Validate tools exist
    for tool, path in [("mlir-opt", MLIR_OPT), ("mlir-translate", MLIR_TRANSLATE)]:
        if not os.path.exists(path):
            print(f"ERROR: {tool} not found at {path}", file=sys.stderr)
            print(f"Build LLVM first. See .github/skills/triton-windows-build/SKILL.md", file=sys.stderr)
            sys.exit(1)

    if args.serialize_only:
        # Read input and serialize directly
        with open(args.input) as f:
            mlir_text = f.read()
    else:
        # Determine passes
        if args.passes:
            passes = [f"--{p.lstrip('-')}" for p in args.passes.split(",")]
        elif args.pipeline:
            passes = PIPELINES[args.pipeline]
        else:
            passes = PIPELINES["arith"]  # default

        # Convert
        mlir_text = convert(args.input, passes)

        if args.verbose:
            print("=== Converted SPIR-V MLIR ===", file=sys.stderr)
            print(mlir_text, file=sys.stderr)

    # Output
    if args.roundtrip:
        result = roundtrip(mlir_text)
        print(result)
    elif args.output:
        serialize(mlir_text, args.output)
    else:
        print(mlir_text)


if __name__ == "__main__":
    main()
