// Test: math → SPIR-V conversion (math ops)
// RUN: python vulkan-opt.py %s --passes --convert-math-to-spirv --convert-arith-to-spirv --convert-func-to-spirv

func.func @math_ops(%x: f32) -> f32 {
  %exp = math.exp %x : f32
  %sqrt = math.sqrt %exp : f32
  return %sqrt : f32
}
